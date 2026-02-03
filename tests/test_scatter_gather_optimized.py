"""Tests for optimized MoE scatter/gather kernels.

Verifies correctness of vectorized scatter/gather operations against
PyTorch reference implementations.
"""

import pytest
import torch

# Skip all tests if MPS is not available
pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available"
)


def reference_gather(
    activations: torch.Tensor,
    sorted_indices: torch.Tensor,
) -> torch.Tensor:
    """Reference gather implementation using PyTorch indexing."""
    return activations[sorted_indices]


def reference_scatter_combine(
    expert_outputs: torch.Tensor,
    expert_probs: torch.Tensor,
    inverse_indices: torch.Tensor,
    batch_size: int,
    top_k: int,
) -> torch.Tensor:
    """Reference scatter-combine implementation using PyTorch."""
    hidden_dim = expert_outputs.shape[1]

    # Reorder outputs back to original order
    # inverse_indices maps original flat position -> sorted position
    original_order = expert_outputs[inverse_indices]

    # Reshape to [batch, top_k, hidden]
    original_order = original_order.reshape(batch_size, top_k, hidden_dim)

    # Apply weights and sum
    weights = expert_probs.unsqueeze(-1)  # [batch, top_k, 1]
    weighted = original_order * weights
    output = weighted.sum(dim=1)  # [batch, hidden]

    return output


class TestGatherVec8:
    """Tests for vectorized gather kernel."""

    @pytest.fixture
    def metal_lib(self):
        """Load Metal kernel library."""
        from metal_marlin.metal_dispatch import MetalKernelLibrary

        # from_source_dir compiles all .metal files including moe_scatter_gather_optimized.metal
        return MetalKernelLibrary.from_source_dir()

    def test_gather_basic(self, metal_lib):
        """Test basic gather operation."""
        from metal_marlin.moe_scatter_gather import ScatterGatherDispatcher

        batch_size = 4
        top_k = 2
        hidden_dim = 256
        total_tokens = batch_size * top_k

        # Create test data
        activations = torch.randn(batch_size, hidden_dim, dtype=torch.float16, device="mps")

        # Create sorted indices (simple sequential for testing)
        sorted_indices = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32, device="mps")

        # Reference result
        ref_gathered = reference_gather(activations, sorted_indices)

        # Metal result
        dispatcher = ScatterGatherDispatcher(metal_lib, hidden_dim=hidden_dim)
        metal_gathered = dispatcher.gather(activations, sorted_indices, total_tokens)

        # Compare
        torch.testing.assert_close(metal_gathered, ref_gathered, rtol=1e-3, atol=1e-3)

    def test_gather_random_indices(self, metal_lib):
        """Test gather with random expert assignments."""
        from metal_marlin.moe_scatter_gather import ScatterGatherDispatcher

        batch_size = 32
        top_k = 4
        hidden_dim = 512
        total_tokens = batch_size * top_k

        activations = torch.randn(batch_size, hidden_dim, dtype=torch.float16, device="mps")

        # Random sorted indices (each maps to a token in [0, batch_size))
        sorted_indices = torch.randint(0, batch_size, (total_tokens,), dtype=torch.int32, device="mps")

        ref_gathered = reference_gather(activations, sorted_indices)

        dispatcher = ScatterGatherDispatcher(metal_lib, hidden_dim=hidden_dim)
        metal_gathered = dispatcher.gather(activations, sorted_indices, total_tokens)

        torch.testing.assert_close(metal_gathered, ref_gathered, rtol=1e-3, atol=1e-3)

    def test_gather_large_hidden(self, metal_lib):
        """Test gather with large hidden dimension (Qwen3-235B size)."""
        from metal_marlin.moe_scatter_gather import ScatterGatherDispatcher

        batch_size = 8
        top_k = 8
        hidden_dim = 7168  # Qwen3-235B hidden dim
        total_tokens = batch_size * top_k

        activations = torch.randn(batch_size, hidden_dim, dtype=torch.float16, device="mps")
        sorted_indices = torch.randint(0, batch_size, (total_tokens,), dtype=torch.int32, device="mps")

        ref_gathered = reference_gather(activations, sorted_indices)

        dispatcher = ScatterGatherDispatcher(metal_lib, hidden_dim=hidden_dim)
        metal_gathered = dispatcher.gather(activations, sorted_indices, total_tokens)

        torch.testing.assert_close(metal_gathered, ref_gathered, rtol=1e-3, atol=1e-3)


class TestScatterCombine:
    """Tests for vectorized scatter-combine kernel."""

    @pytest.fixture
    def metal_lib(self):
        """Load Metal kernel library."""
        from metal_marlin.metal_dispatch import MetalKernelLibrary

        # from_source_dir compiles all .metal files including moe_scatter_gather_optimized.metal
        return MetalKernelLibrary.from_source_dir()

    def test_scatter_basic_topk2(self, metal_lib):
        """Test scatter-combine with top_k=2."""
        from metal_marlin.moe_scatter_gather import ScatterGatherDispatcher

        batch_size = 4
        top_k = 2
        hidden_dim = 256

        # Expert outputs in sorted order
        expert_outputs = torch.randn(
            batch_size * top_k, hidden_dim, dtype=torch.float16, device="mps"
        )

        # Routing probabilities
        expert_probs = torch.rand(batch_size, top_k, dtype=torch.float16, device="mps")
        expert_probs = expert_probs / expert_probs.sum(dim=1, keepdim=True)  # Normalize

        # Inverse indices (simple: sorted order = original order)
        inverse_indices = torch.arange(batch_size * top_k, dtype=torch.int32, device="mps")

        # Reference
        ref_output = reference_scatter_combine(
            expert_outputs, expert_probs, inverse_indices, batch_size, top_k
        )

        # Metal
        dispatcher = ScatterGatherDispatcher(metal_lib, hidden_dim=hidden_dim, max_top_k=top_k)
        metal_output = dispatcher.scatter_combine(
            expert_outputs, expert_probs, inverse_indices, batch_size, top_k
        )

        torch.testing.assert_close(metal_output, ref_output, rtol=1e-2, atol=1e-2)

    def test_scatter_topk4(self, metal_lib):
        """Test scatter-combine with top_k=4."""
        from metal_marlin.moe_scatter_gather import ScatterGatherDispatcher

        batch_size = 8
        top_k = 4
        hidden_dim = 512

        expert_outputs = torch.randn(
            batch_size * top_k, hidden_dim, dtype=torch.float16, device="mps"
        )

        expert_probs = torch.rand(batch_size, top_k, dtype=torch.float16, device="mps")
        expert_probs = expert_probs / expert_probs.sum(dim=1, keepdim=True)

        # Random permutation to test non-trivial inverse mapping
        perm = torch.randperm(batch_size * top_k, device="mps")
        inverse_indices = torch.argsort(perm).int()

        ref_output = reference_scatter_combine(
            expert_outputs, expert_probs, inverse_indices, batch_size, top_k
        )

        dispatcher = ScatterGatherDispatcher(metal_lib, hidden_dim=hidden_dim, max_top_k=top_k)
        metal_output = dispatcher.scatter_combine(
            expert_outputs, expert_probs, inverse_indices, batch_size, top_k
        )

        torch.testing.assert_close(metal_output, ref_output, rtol=1e-2, atol=1e-2)

    def test_scatter_topk8(self, metal_lib):
        """Test scatter-combine with top_k=8 (Qwen3-235B)."""
        from metal_marlin.moe_scatter_gather import ScatterGatherDispatcher

        batch_size = 16
        top_k = 8
        hidden_dim = 7168

        expert_outputs = torch.randn(
            batch_size * top_k, hidden_dim, dtype=torch.float16, device="mps"
        )

        expert_probs = torch.rand(batch_size, top_k, dtype=torch.float16, device="mps")
        expert_probs = expert_probs / expert_probs.sum(dim=1, keepdim=True)

        inverse_indices = torch.arange(batch_size * top_k, dtype=torch.int32, device="mps")

        ref_output = reference_scatter_combine(
            expert_outputs, expert_probs, inverse_indices, batch_size, top_k
        )

        dispatcher = ScatterGatherDispatcher(metal_lib, hidden_dim=hidden_dim, max_top_k=top_k)
        metal_output = dispatcher.scatter_combine(
            expert_outputs, expert_probs, inverse_indices, batch_size, top_k
        )

        torch.testing.assert_close(metal_output, ref_output, rtol=1e-2, atol=1e-2)


class TestCountAndOffsets:
    """Tests for token counting and offset computation."""

    @pytest.fixture
    def metal_lib(self):
        """Load Metal kernel library."""
        from metal_marlin.metal_dispatch import MetalKernelLibrary

        # from_source_dir compiles all .metal files including moe_scatter_gather_optimized.metal
        return MetalKernelLibrary.from_source_dir()

    def test_count_tokens(self, metal_lib):
        """Test token counting per expert."""
        from metal_marlin.moe_scatter_gather import ScatterGatherDispatcher

        batch_size = 16
        top_k = 2
        num_experts = 8

        # Create expert assignments
        expert_ids = torch.randint(0, num_experts, (batch_size, top_k), dtype=torch.int32, device="mps")

        # Reference: use bincount
        ref_counts = torch.bincount(
            expert_ids.reshape(-1).long(), minlength=num_experts
        ).int()

        # Metal
        dispatcher = ScatterGatherDispatcher(metal_lib, hidden_dim=256)
        metal_counts = dispatcher.count_tokens(expert_ids, num_experts)

        torch.testing.assert_close(metal_counts, ref_counts)

    def test_prefix_sum(self, metal_lib):
        """Test exclusive prefix sum for offsets."""
        from metal_marlin.moe_scatter_gather import ScatterGatherDispatcher

        num_experts = 64

        # Random counts
        expert_counts = torch.randint(0, 100, (num_experts,), dtype=torch.int32, device="mps")

        # Reference: cumsum with prepended 0
        ref_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device="mps")
        ref_offsets[1:] = torch.cumsum(expert_counts, dim=0)

        # Metal
        dispatcher = ScatterGatherDispatcher(metal_lib, hidden_dim=256)
        metal_offsets = dispatcher.compute_offsets(expert_counts, num_experts)

        torch.testing.assert_close(metal_offsets, ref_offsets)


class TestEndToEnd:
    """End-to-end tests for full scatter/gather pipeline."""

    @pytest.fixture
    def metal_lib(self):
        """Load Metal kernel library."""
        from metal_marlin.metal_dispatch import MetalKernelLibrary

        # from_source_dir compiles all .metal files including moe_scatter_gather_optimized.metal
        return MetalKernelLibrary.from_source_dir()

    def test_full_pipeline(self, metal_lib):
        """Test complete gather -> expert GEMM (simulated) -> scatter pipeline."""
        from metal_marlin.moe_dispatch import group_tokens_by_expert_full
        from metal_marlin.moe_scatter_gather import ScatterGatherDispatcher

        batch_size = 8
        top_k = 4
        num_experts = 16
        hidden_dim = 512

        # Input activations
        activations = torch.randn(batch_size, hidden_dim, dtype=torch.float16, device="mps")

        # Router outputs
        expert_ids = torch.randint(0, num_experts, (batch_size, top_k), dtype=torch.int64, device="mps")
        expert_probs = torch.rand(batch_size, top_k, dtype=torch.float16, device="mps")
        expert_probs = expert_probs / expert_probs.sum(dim=1, keepdim=True)

        # Get dispatch info using existing function
        dispatch_info = group_tokens_by_expert_full(expert_ids, num_experts)

        # Gather
        dispatcher = ScatterGatherDispatcher(metal_lib, hidden_dim=hidden_dim, max_top_k=top_k)

        # Note: sorted_token_indices maps sorted -> original token, so we can use it directly
        gathered = dispatcher.gather(
            activations,
            dispatch_info.sorted_token_indices.int(),
            dispatch_info.total_assignments
        )

        # Simulate expert computation (just pass through for testing)
        expert_outputs = gathered.clone()

        # Scatter combine
        output = dispatcher.scatter_combine(
            expert_outputs,
            expert_probs,
            dispatch_info.inverse_indices.int(),
            batch_size,
            top_k
        )

        # Verify output shape
        assert output.shape == (batch_size, hidden_dim)

        # Verify output is not all zeros (sanity check)
        assert output.abs().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
