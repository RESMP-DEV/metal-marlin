"""Test MoE dispatch with various batch sizes.

Validates that MoE token dispatch and expert routing handles batched inputs
correctly across different batch sizes and expert configurations.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# Skip entire module if MPS not available
pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


@pytest.fixture
def hidden_dim() -> int:
    """Default hidden dimension for tests."""
    return 256


@pytest.fixture
def intermediate_dim() -> int:
    """Default intermediate dimension for expert FFN."""
    return 512


@pytest.fixture
def num_experts() -> int:
    """Default number of experts."""
    return 8


@pytest.fixture
def top_k() -> int:
    """Default number of experts per token."""
    return 2


class TestMoEBatchSizes:
    """Tests for MoE dispatch with various batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 64])
    def test_moe_batch_sizes_dispatch_info(
        self, batch_size: int, hidden_dim: int, num_experts: int, top_k: int
    ) -> None:
        """MoE dispatch info has correct shape for various batch sizes."""
        from metal_marlin.moe_dispatch import group_tokens_by_expert_full

        # Generate random expert assignments
        torch.manual_seed(42)
        expert_ids = torch.randint(
            0, num_experts, (batch_size, top_k), dtype=torch.int32, device="mps"
        )

        dispatch_info = group_tokens_by_expert_full(expert_ids, num_experts)

        # Verify structure
        assert dispatch_info.num_tokens == batch_size
        assert dispatch_info.top_k == top_k
        assert dispatch_info.num_experts == num_experts
        assert dispatch_info.total_assignments == batch_size * top_k

        # Verify tensor shapes
        assert dispatch_info.sorted_token_indices.shape == (batch_size * top_k,)
        assert dispatch_info.sorted_expert_indices.shape == (batch_size * top_k,)
        assert dispatch_info.expert_offsets.shape == (num_experts + 1,)
        assert dispatch_info.inverse_indices.shape == (batch_size * top_k,)

        # Verify offsets are valid
        assert dispatch_info.expert_offsets[0].item() == 0
        assert dispatch_info.expert_offsets[-1].item() == batch_size * top_k

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 64])
    def test_moe_batch_sizes_gather_scatter(
        self, batch_size: int, hidden_dim: int, num_experts: int, top_k: int
    ) -> None:
        """MoE gather/scatter roundtrip preserves shape and data for various batch sizes."""
        from metal_marlin.moe_dispatch import (
            gather_for_experts,
            group_tokens_by_expert_full,
            scatter_expert_outputs,
        )

        torch.manual_seed(42)

        # Create test inputs
        activations = torch.randn(batch_size, hidden_dim, device="mps")
        expert_ids = torch.randint(
            0, num_experts, (batch_size, top_k), dtype=torch.int32, device="mps"
        )
        expert_probs = torch.softmax(
            torch.randn(batch_size, top_k, device="mps"), dim=-1
        )

        # Dispatch
        dispatch_info = group_tokens_by_expert_full(expert_ids, num_experts)
        gathered = gather_for_experts(activations, dispatch_info)

        # Gathered should have shape [batch_size * top_k, hidden_dim]
        assert gathered.shape == (batch_size * top_k, hidden_dim)

        # Identity expert pass (output = input)
        expert_outputs = gathered

        # Scatter back
        output = scatter_expert_outputs(expert_outputs, expert_probs, dispatch_info)

        # Output shape should match original
        assert output.shape == (batch_size, hidden_dim)

        # Since we're using identity expert and expert_probs sum to 1,
        # output should equal input activations
        expected = activations
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 64])
    def test_moe_output_shape_with_expert_ffn(
        self, batch_size: int, hidden_dim: int, intermediate_dim: int,
        num_experts: int, top_k: int
    ) -> None:
        """MoE output shape correct for various batch sizes with actual expert FFN."""
        from metal_marlin.moe_dispatch import (
            gather_for_experts,
            group_tokens_by_expert_full,
            scatter_expert_outputs,
        )

        torch.manual_seed(42)

        # Create test inputs
        activations = torch.randn(batch_size, hidden_dim, device="mps")
        expert_ids = torch.randint(
            0, num_experts, (batch_size, top_k), dtype=torch.int32, device="mps"
        )
        expert_probs = torch.softmax(
            torch.randn(batch_size, top_k, device="mps"), dim=-1
        )

        # Create simple expert weights
        expert_weights = torch.randn(
            num_experts, hidden_dim, intermediate_dim, device="mps"
        ) * 0.01
        expert_down = torch.randn(
            num_experts, intermediate_dim, hidden_dim, device="mps"
        ) * 0.01

        # Dispatch
        dispatch_info = group_tokens_by_expert_full(expert_ids, num_experts)
        gathered = gather_for_experts(activations, dispatch_info)

        # Run expert computation per-expert
        expert_outputs = torch.zeros(batch_size * top_k, hidden_dim, device="mps")
        for e in range(num_experts):
            start = int(dispatch_info.expert_offsets[e].item())
            end = int(dispatch_info.expert_offsets[e + 1].item())
            if start < end:
                tokens_for_expert = gathered[start:end]  # [n, hidden_dim]
                # Simple linear: hidden -> intermediate -> hidden
                hidden = torch.matmul(tokens_for_expert, expert_weights[e])
                hidden = torch.relu(hidden)
                output = torch.matmul(hidden, expert_down[e])
                expert_outputs[start:end] = output

        # Scatter back
        result = scatter_expert_outputs(expert_outputs, expert_probs, dispatch_info)

        # Output shape should be [batch_size, hidden_dim]
        assert result.shape == (batch_size, hidden_dim)
        assert not result.isnan().any()
        assert not result.isinf().any()


class TestMoEVariableExpertsPerToken:
    """Tests for expert routing and selection."""

    def test_expert_ids_indexing_is_correct(
        self, hidden_dim: int, num_experts: int, top_k: int
    ) -> None:
        """Verify expert_ids indexing is correct in dispatch."""
        from metal_marlin.moe_dispatch import group_tokens_by_expert_full

        batch_size = 16
        torch.manual_seed(42)

        # Create explicit expert assignments for verification
        expert_ids = torch.randint(
            0, num_experts, (batch_size, top_k), dtype=torch.int32, device="mps"
        )

        dispatch_info = group_tokens_by_expert_full(expert_ids, num_experts)

        # Verify that tokens are grouped correctly by expert
        expert_ids_flat = expert_ids.reshape(-1).cpu().numpy()
        sorted_idx = dispatch_info.sorted_token_indices.cpu().numpy()
        expert_slot = dispatch_info.sorted_expert_indices.cpu().numpy()

        for e in range(num_experts):
            start = int(dispatch_info.expert_offsets[e].item())
            end = int(dispatch_info.expert_offsets[e + 1].item())

            for i in range(start, end):
                # Get the original token and slot
                orig_token = sorted_idx[i]
                orig_slot = expert_slot[i]
                # Reconstruct flat index
                flat_idx = orig_token * top_k + orig_slot
                # Verify the expert ID at this position is e
                assert expert_ids_flat[flat_idx] == e, (
                    f"Position {i} in expert {e}'s group has wrong expert: "
                    f"expected {e}, got {expert_ids_flat[flat_idx]}"
                )

    def test_probability_weighting_is_correct(
        self, hidden_dim: int, num_experts: int, top_k: int
    ) -> None:
        """Verify probability weighting is applied correctly."""
        from metal_marlin.moe_dispatch import (
            gather_for_experts,
            group_tokens_by_expert_full,
            scatter_expert_outputs,
        )

        batch_size = 4
        torch.manual_seed(42)

        # Create inputs with known values for verification
        activations = torch.ones(batch_size, hidden_dim, device="mps")
        expert_ids = torch.tensor(
            [[0, 1], [0, 1], [0, 1], [0, 1]], dtype=torch.int32, device="mps"
        )
        # Explicit probabilities: 70% to first expert, 30% to second
        expert_probs = torch.tensor(
            [[0.7, 0.3], [0.7, 0.3], [0.7, 0.3], [0.7, 0.3]],
            dtype=torch.float32,
            device="mps",
        )

        dispatch_info = group_tokens_by_expert_full(expert_ids, num_experts)
        gathered = gather_for_experts(activations, dispatch_info)

        # Expert 0 outputs 1s, Expert 1 outputs 2s
        expert_outputs = torch.zeros(batch_size * top_k, hidden_dim, device="mps")
        for e in range(num_experts):
            start = int(dispatch_info.expert_offsets[e].item())
            end = int(dispatch_info.expert_offsets[e + 1].item())
            if e == 0:
                expert_outputs[start:end] = 1.0
            elif e == 1:
                expert_outputs[start:end] = 2.0
            # Experts 2-7 have no tokens, leave as zeros

        result = scatter_expert_outputs(expert_outputs, expert_probs, dispatch_info)

        # Expected: 0.7 * 1.0 + 0.3 * 2.0 = 1.3
        expected = torch.full((batch_size, hidden_dim), 1.3, device="mps")
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_unequal_expert_load(
        self, hidden_dim: int, num_experts: int, top_k: int
    ) -> None:
        """Verify dispatch handles unequal expert load correctly."""
        from metal_marlin.moe_dispatch import (
            compute_expert_load,
            group_tokens_by_expert_full,
        )

        batch_size = 8

        # All tokens route to expert 0 and 1
        expert_ids = torch.zeros(batch_size, top_k, dtype=torch.int32, device="mps")
        expert_ids[:, 0] = 0  # All first choices are expert 0
        expert_ids[:, 1] = 1  # All second choices are expert 1

        dispatch_info = group_tokens_by_expert_full(expert_ids, num_experts)
        load = compute_expert_load(expert_ids, num_experts)

        # Expert 0 and 1 should each have batch_size tokens
        assert load[0].item() == batch_size
        assert load[1].item() == batch_size
        # Other experts should have 0
        for e in range(2, num_experts):
            assert load[e].item() == 0

    def test_empty_experts_handled(
        self, hidden_dim: int, num_experts: int
    ) -> None:
        """Verify dispatch handles empty experts (no tokens) correctly."""
        from metal_marlin.moe_dispatch import (
            gather_for_experts,
            group_tokens_by_expert_full,
            scatter_expert_outputs,
        )

        batch_size = 4
        top_k = 1  # Single expert per token

        # Only route to expert 0
        expert_ids = torch.zeros(batch_size, top_k, dtype=torch.int32, device="mps")
        expert_probs = torch.ones(batch_size, top_k, device="mps")
        activations = torch.randn(batch_size, hidden_dim, device="mps")

        dispatch_info = group_tokens_by_expert_full(expert_ids, num_experts)
        gathered = gather_for_experts(activations, dispatch_info)

        # Only expert 0 has tokens
        assert dispatch_info.expert_offsets[0].item() == 0
        assert dispatch_info.expert_offsets[1].item() == batch_size
        # All other experts have zero tokens
        for e in range(2, num_experts + 1):
            assert (
                dispatch_info.expert_offsets[e].item()
                == dispatch_info.expert_offsets[e - 1].item()
            )

        # Identity expert
        expert_outputs = gathered
        result = scatter_expert_outputs(expert_outputs, expert_probs, dispatch_info)

        torch.testing.assert_close(result, activations, rtol=1e-5, atol=1e-5)


class TestMoERouterTopK:
    """Tests for MoE router top-k selection."""

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 64])
    def test_router_topk_output_shapes(
        self, batch_size: int, hidden_dim: int, num_experts: int, top_k: int
    ) -> None:
        """Router produces correct output shapes for various batch sizes."""
        from metal_marlin.kernels import moe_router_topk

        torch.manual_seed(42)

        hidden = torch.randn(batch_size, hidden_dim, device="mps")
        router_weights = torch.randn(hidden_dim, num_experts, device="mps")

        expert_ids, expert_probs = moe_router_topk(hidden, router_weights, top_k=top_k)

        assert expert_ids.shape == (batch_size, top_k)
        assert expert_probs.shape == (batch_size, top_k)

    def test_router_topk_probs_sum_to_one(
        self, hidden_dim: int, num_experts: int, top_k: int
    ) -> None:
        """Router probabilities sum to 1 for each token."""
        from metal_marlin.kernels import moe_router_topk

        batch_size = 16
        torch.manual_seed(42)

        hidden = torch.randn(batch_size, hidden_dim, device="mps")
        router_weights = torch.randn(hidden_dim, num_experts, device="mps")

        expert_ids, expert_probs = moe_router_topk(hidden, router_weights, top_k=top_k)

        # Probabilities should sum to 1 (renormalized)
        prob_sums = expert_probs.sum(dim=-1)
        expected_sums = torch.ones(batch_size, device="mps")
        torch.testing.assert_close(prob_sums, expected_sums, rtol=1e-5, atol=1e-5)

    def test_router_topk_ids_in_range(
        self, hidden_dim: int, num_experts: int, top_k: int
    ) -> None:
        """Router expert IDs are in valid range."""
        from metal_marlin.kernels import moe_router_topk

        batch_size = 32
        torch.manual_seed(42)

        hidden = torch.randn(batch_size, hidden_dim, device="mps")
        router_weights = torch.randn(hidden_dim, num_experts, device="mps")

        expert_ids, expert_probs = moe_router_topk(hidden, router_weights, top_k=top_k)

        assert (expert_ids >= 0).all()
        assert (expert_ids < num_experts).all()

    def test_router_topk_ids_unique_per_token(
        self, hidden_dim: int, num_experts: int, top_k: int
    ) -> None:
        """Each token's top-k expert IDs are unique."""
        from metal_marlin.kernels import moe_router_topk

        batch_size = 16
        torch.manual_seed(42)

        hidden = torch.randn(batch_size, hidden_dim, device="mps")
        router_weights = torch.randn(hidden_dim, num_experts, device="mps")

        expert_ids, _ = moe_router_topk(hidden, router_weights, top_k=top_k)

        # Check each token has unique expert selections
        for i in range(batch_size):
            token_experts = expert_ids[i].cpu().tolist()
            assert len(set(token_experts)) == len(token_experts), (
                f"Token {i} has duplicate experts: {token_experts}"
            )


class TestMoEExpertGEMM:
    """Tests for MoE expert GEMM with FP4 quantization."""

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 64])
    def test_moe_expert_gemm_fp4_output_shape(
        self, batch_size: int, hidden_dim: int, num_experts: int, top_k: int
    ) -> None:
        """MoE expert GEMM produces correct output shape."""
        from metal_marlin.kernels import moe_expert_gemm_fp4

        out_dim = hidden_dim * 2  # Common FFN expansion
        torch.manual_seed(42)

        activations = torch.randn(batch_size, hidden_dim, device="mps", dtype=torch.float16)
        # Packed FP4 weights: hidden_dim / 8 since 2 elements per byte
        expert_weights = torch.randint(
            0, 255, (num_experts, hidden_dim // 8, out_dim),
            dtype=torch.uint8, device="mps"
        )
        # Scales per group (group_size=128)
        scales = torch.randn(
            num_experts, hidden_dim // 128, out_dim,
            device="mps", dtype=torch.float16
        ).abs()
        expert_ids = torch.randint(
            0, num_experts, (batch_size, top_k), dtype=torch.int32, device="mps"
        )
        expert_probs = torch.softmax(
            torch.randn(batch_size, top_k, device="mps"), dim=-1
        )

        output = moe_expert_gemm_fp4(
            activations, expert_weights, scales, expert_ids, expert_probs
        )

        assert output.shape == (batch_size, out_dim)
        assert not output.isnan().any()
        assert not output.isinf().any()


class TestMoEDispatcher:
    """Tests for high-level MoE dispatcher module."""

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 64])
    def test_moe_dispatcher_output_shape(
        self, batch_size: int, hidden_dim: int, num_experts: int, top_k: int
    ) -> None:
        """MoE dispatcher produces correct output shape."""
        from metal_marlin.moe_dispatch import MoEDispatcher

        torch.manual_seed(42)

        # Create simple experts
        experts = []
        for _ in range(num_experts):
            expert = torch.nn.Linear(hidden_dim, hidden_dim, bias=False, device="mps")
            torch.nn.init.normal_(expert.weight, std=0.02)
            experts.append(expert)

        dispatcher = MoEDispatcher(
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            experts=experts,
        )

        hidden = torch.randn(batch_size, hidden_dim, device="mps")
        gate_logits = torch.randn(batch_size, num_experts, device="mps")

        output = dispatcher(hidden, gate_logits)

        assert output.shape == (batch_size, hidden_dim)
        assert not output.isnan().any()

    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_moe_dispatcher_3d_input(
        self, batch_size: int, hidden_dim: int, num_experts: int, top_k: int
    ) -> None:
        """MoE dispatcher handles 3D [batch, seq, hidden] input."""
        from metal_marlin.moe_dispatch import MoEDispatcher

        seq_len = 16
        torch.manual_seed(42)

        experts = []
        for _ in range(num_experts):
            expert = torch.nn.Linear(hidden_dim, hidden_dim, bias=False, device="mps")
            torch.nn.init.normal_(expert.weight, std=0.02)
            experts.append(expert)

        dispatcher = MoEDispatcher(
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            experts=experts,
        )

        hidden = torch.randn(batch_size, seq_len, hidden_dim, device="mps")
        gate_logits = torch.randn(batch_size * seq_len, num_experts, device="mps")

        output = dispatcher(hidden, gate_logits)

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert not output.isnan().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
