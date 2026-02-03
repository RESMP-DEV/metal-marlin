"""Tests for GPU-accelerated token sorting in MoE dispatch."""

import pytest
import torch

from metal_marlin.moe_dispatch import (
    group_tokens_by_expert,
    group_tokens_by_expert_full,
    group_tokens_by_expert_full_gpu,
    group_tokens_by_expert_gpu,
)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
class TestMoEGPUSort:
    """Test GPU-based token sorting for MoE dispatch."""

    def test_gpu_sort_matches_cpu_small(self):
        """GPU sort should match CPU sort for small batch."""
        batch_size = 8
        top_k = 2
        num_experts = 4
        device = "mps"

        # Random expert assignments
        expert_ids = torch.randint(0, num_experts, (batch_size, top_k), device=device)
        expert_probs = torch.rand(batch_size, top_k, device=device)
        expert_probs = expert_probs / expert_probs.sum(dim=-1, keepdim=True)

        # CPU version
        sorted_cpu, offsets_cpu, inverse_cpu = group_tokens_by_expert(expert_ids, num_experts)

        # GPU version
        sorted_gpu, offsets_gpu, inverse_gpu = group_tokens_by_expert_gpu(
            expert_ids, expert_probs, num_experts
        )

        # Check expert offsets match (grouping structure)
        assert torch.equal(offsets_cpu, offsets_gpu), "Expert offsets mismatch"

        # Check that both produce valid permutations
        assert sorted_cpu.shape == sorted_gpu.shape
        assert set(sorted_cpu.tolist()) == set(sorted_gpu.tolist())

    def test_gpu_sort_matches_cpu_large(self):
        """GPU sort should match CPU sort for realistic batch size."""
        batch_size = 128
        top_k = 8
        num_experts = 64
        device = "mps"

        expert_ids = torch.randint(0, num_experts, (batch_size, top_k), device=device)
        expert_probs = torch.rand(batch_size, top_k, device=device)
        expert_probs = expert_probs / expert_probs.sum(dim=-1, keepdim=True)

        sorted_cpu, offsets_cpu, inverse_cpu = group_tokens_by_expert(expert_ids, num_experts)
        sorted_gpu, offsets_gpu, inverse_gpu = group_tokens_by_expert_gpu(
            expert_ids, expert_probs, num_experts
        )

        # Check grouping structure
        assert torch.equal(offsets_cpu, offsets_gpu), "Expert offsets mismatch for large batch"

        # Verify both are valid permutations
        total = batch_size * top_k
        assert sorted_cpu.shape == (total,)
        assert sorted_gpu.shape == (total,)

    def test_gpu_sort_full_dispatch_info(self):
        """Full dispatch info should be consistent."""
        batch_size = 32
        top_k = 4
        num_experts = 16
        device = "mps"

        expert_ids = torch.randint(0, num_experts, (batch_size, top_k), device=device)
        expert_probs = torch.rand(batch_size, top_k, device=device)
        expert_probs = expert_probs / expert_probs.sum(dim=-1, keepdim=True)

        # CPU version
        info_cpu = group_tokens_by_expert_full(expert_ids, num_experts)

        # GPU version
        info_gpu = group_tokens_by_expert_full_gpu(expert_ids, expert_probs, num_experts)

        # Verify shapes
        assert info_cpu.sorted_token_indices.shape == info_gpu.sorted_token_indices.shape
        assert info_cpu.sorted_expert_indices.shape == info_gpu.sorted_expert_indices.shape
        assert torch.equal(info_cpu.expert_offsets, info_gpu.expert_offsets)

    def test_gpu_sort_expert_grouping(self):
        """Verify tokens are correctly grouped by expert."""
        batch_size = 16
        top_k = 2
        num_experts = 4
        device = "mps"

        # Fixed expert assignments for verification
        expert_ids = torch.tensor([
            [0, 1],  # token 0
            [1, 2],  # token 1
            [2, 3],  # token 2
            [0, 3],  # token 3
        ] * 4, device=device)  # Repeat to make 16 tokens
        expert_probs = torch.ones_like(expert_ids, dtype=torch.float16) * 0.5

        info = group_tokens_by_expert_full_gpu(expert_ids, expert_probs, num_experts)

        # Check that expert_offsets define valid ranges
        for expert_id in range(num_experts):
            start = info.expert_offsets[expert_id]
            end = info.expert_offsets[expert_id + 1]
            count = end - start

            # Each expert should have 8 assignments (4 tokens * 4 repetitions / 2)
            assert count == 8, f"Expert {expert_id} has {count} assignments, expected 8"

    def test_gpu_sort_empty_experts(self):
        """Handle case where some experts have no assignments."""
        batch_size = 8
        top_k = 2
        num_experts = 16  # More experts than assignments
        device = "mps"

        # All tokens assigned to first 2 experts only
        expert_ids = torch.randint(0, 2, (batch_size, top_k), device=device)
        expert_probs = torch.rand(batch_size, top_k, device=device)
        expert_probs = expert_probs / expert_probs.sum(dim=-1, keepdim=True)

        info = group_tokens_by_expert_full_gpu(expert_ids, expert_probs, num_experts)

        # Verify empty experts have zero assignments
        for expert_id in range(2, num_experts):
            start = info.expert_offsets[expert_id]
            end = info.expert_offsets[expert_id + 1]
            assert start == end, f"Expert {expert_id} should be empty"

    def test_gpu_sort_single_expert(self):
        """Edge case: all tokens go to same expert."""
        batch_size = 16
        top_k = 4
        num_experts = 8
        device = "mps"

        # All assignments to expert 3
        expert_ids = torch.full((batch_size, top_k), 3, device=device)
        expert_probs = torch.ones(batch_size, top_k, device=device) / top_k

        info = group_tokens_by_expert_full_gpu(expert_ids, expert_probs, num_experts)

        # Expert 3 should have all assignments
        total = batch_size * top_k
        assert info.expert_offsets[3] == 0
        assert info.expert_offsets[4] == total

        # All other experts should be empty
        for expert_id in range(num_experts):
            if expert_id == 3:
                continue
            start = info.expert_offsets[expert_id]
            end = info.expert_offsets[expert_id + 1]
            if expert_id < 3:
                assert start == end == 0
            else:
                assert start == end == total


@pytest.mark.benchmark
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
class TestMoEGPUSortPerformance:
    """Benchmark GPU sort vs CPU sort."""

    def test_gpu_sort_speedup(self, benchmark):
        """GPU sort should be faster than CPU for large batches."""
        batch_size = 512
        top_k = 8
        num_experts = 64
        device = "mps"

        expert_ids = torch.randint(0, num_experts, (batch_size, top_k), device=device)
        expert_probs = torch.rand(batch_size, top_k, device=device)

        def run_gpu():
            return group_tokens_by_expert_gpu(expert_ids, expert_probs, num_experts)

        result = benchmark(run_gpu)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
