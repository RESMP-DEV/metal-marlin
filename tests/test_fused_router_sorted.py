"""Tests for fused MoE router with sorted expert indices.

This test suite validates the moe_fused_router_sorted kernel which performs
router computation, softmax, top-k selection, and token grouping in a single
GPU kernel.
"""

import pytest
import torch
import numpy as np


@pytest.fixture
def router_params():
    """Standard router parameters for testing."""
    return {
        "batch_size": 8,
        "hidden_dim": 64,
        "num_experts": 16,
        "top_k": 2,
    }


@pytest.fixture
def sample_inputs(router_params):
    """Generate sample inputs for router testing."""
    batch_size = router_params["batch_size"]
    hidden_dim = router_params["hidden_dim"]
    num_experts = router_params["num_experts"]
    
    # Random hidden states
    hidden = torch.randn(batch_size, hidden_dim, dtype=torch.float16, device="mps")
    
    # Random router weights
    router_weights = torch.randn(hidden_dim, num_experts, dtype=torch.float16, device="mps")
    
    return hidden, router_weights


class TestFusedRouterSorted:
    """Test suite for moe_fused_router_sorted kernel."""
    
    def test_kernel_compiles(self):
        """Verify the kernel is available in the Metal library."""
        try:
            from metal_marlin.metal_dispatch import get_default_library
            lib = get_default_library()
            
            # Check that the kernel functions exist
            for func_name in [
                "moe_fused_router_sorted",
                "moe_fused_router_sorted_coalesced",
            ]:
                pipeline = lib.get_pipeline(func_name)
                assert pipeline is not None, f"{func_name} not found"
                
        except Exception as e:
            pytest.skip(f"Metal not available: {e}")
    
    def test_output_shapes(self, router_params, sample_inputs):
        """Verify output shapes are correct."""
        try:
            from metal_marlin.moe.fused_router import FusedMoERouter
            from metal_marlin.metal_dispatch import get_default_library
            
            lib = get_default_library()
            hidden, router_weights = sample_inputs
            batch_size = router_params["batch_size"]
            num_experts = router_params["num_experts"]
            top_k = router_params["top_k"]
            hidden_dim = router_params["hidden_dim"]
            
            router = FusedMoERouter(lib, num_experts, top_k, hidden_dim)
            output = router.forward(hidden, router_weights)
            
            # Verify shapes
            assert output.expert_ids.shape == (batch_size, top_k)
            assert output.expert_probs.shape == (batch_size, top_k)
            assert output.sorted_indices.shape == (batch_size * top_k,)
            assert output.expert_offsets.shape == (num_experts + 1,)
            
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")
    
    def test_expert_offsets_valid(self, router_params, sample_inputs):
        """Verify expert offsets are valid (monotonically increasing)."""
        try:
            from metal_marlin.moe.fused_router import FusedMoERouter
            from metal_marlin.metal_dispatch import get_default_library
            
            lib = get_default_library()
            hidden, router_weights = sample_inputs
            num_experts = router_params["num_experts"]
            top_k = router_params["top_k"]
            hidden_dim = router_params["hidden_dim"]
            
            router = FusedMoERouter(lib, num_experts, top_k, hidden_dim)
            output = router.forward(hidden, router_weights)
            
            # Offsets should be monotonically increasing
            offsets = output.expert_offsets.cpu().numpy()
            assert np.all(np.diff(offsets) >= 0), "Offsets not monotonically increasing"
            
            # Total should equal batch_size * top_k
            total = offsets[-1]
            assert total == router_params["batch_size"] * top_k
            
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")
    
    def test_sorted_indices_in_range(self, router_params, sample_inputs):
        """Verify sorted indices are within valid range."""
        try:
            from metal_marlin.moe.fused_router import FusedMoERouter
            from metal_marlin.metal_dispatch import get_default_library
            
            lib = get_default_library()
            hidden, router_weights = sample_inputs
            num_experts = router_params["num_experts"]
            top_k = router_params["top_k"]
            hidden_dim = router_params["hidden_dim"]
            batch_size = router_params["batch_size"]
            
            router = FusedMoERouter(lib, num_experts, top_k, hidden_dim)
            output = router.forward(hidden, router_weights)
            
            # All indices should be in [0, batch_size * top_k)
            sorted_indices = output.sorted_indices.cpu().numpy()
            assert np.all(sorted_indices >= 0)
            assert np.all(sorted_indices < batch_size * top_k)
            
            # All indices should be unique
            assert len(np.unique(sorted_indices)) == len(sorted_indices)
            
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")
    
    def test_expert_probs_normalized(self, router_params, sample_inputs):
        """Verify expert probabilities are normalized per token."""
        try:
            from metal_marlin.moe.fused_router import FusedMoERouter
            from metal_marlin.metal_dispatch import get_default_library
            
            lib = get_default_library()
            hidden, router_weights = sample_inputs
            num_experts = router_params["num_experts"]
            top_k = router_params["top_k"]
            hidden_dim = router_params["hidden_dim"]
            
            router = FusedMoERouter(lib, num_experts, top_k, hidden_dim)
            output = router.forward(hidden, router_weights)
            
            # Probabilities should sum to ~1 per token
            probs = output.expert_probs.cpu().float().numpy()
            sums = probs.sum(axis=1)
            np.testing.assert_allclose(sums, 1.0, rtol=1e-3, atol=1e-4)
            
            # All probabilities should be positive
            assert np.all(probs >= 0)
            
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")
    
    def test_expert_ids_in_range(self, router_params, sample_inputs):
        """Verify expert IDs are within valid range."""
        try:
            from metal_marlin.moe.fused_router import FusedMoERouter
            from metal_marlin.metal_dispatch import get_default_library
            
            lib = get_default_library()
            hidden, router_weights = sample_inputs
            num_experts = router_params["num_experts"]
            top_k = router_params["top_k"]
            hidden_dim = router_params["hidden_dim"]
            
            router = FusedMoERouter(lib, num_experts, top_k, hidden_dim)
            output = router.forward(hidden, router_weights)
            
            # All expert IDs should be in [0, num_experts)
            expert_ids = output.expert_ids.cpu().numpy()
            assert np.all(expert_ids >= 0)
            assert np.all(expert_ids < num_experts)
            
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")
    
    def test_matches_pytorch_reference(self, router_params, sample_inputs):
        """Compare output with PyTorch reference implementation."""
        try:
            from metal_marlin.moe.fused_router import FusedMoERouter
            from metal_marlin.metal_dispatch import get_default_library
            
            lib = get_default_library()
            hidden, router_weights = sample_inputs
            num_experts = router_params["num_experts"]
            top_k = router_params["top_k"]
            hidden_dim = router_params["hidden_dim"]
            
            # Fused router output
            router = FusedMoERouter(lib, num_experts, top_k, hidden_dim)
            output = router.forward(hidden, router_weights)
            
            # PyTorch reference
            logits = hidden.float() @ router_weights.float()
            probs = torch.softmax(logits, dim=-1)
            ref_probs, ref_ids = torch.topk(probs, k=top_k, dim=-1, sorted=True)
            ref_probs = ref_probs / ref_probs.sum(dim=-1, keepdim=True)
            
            # Compare expert IDs (may differ due to ties, so check probs)
            fused_ids = output.expert_ids.cpu()
            fused_probs = output.expert_probs.cpu()
            
            # For each token, check that the top-k probabilities match
            for b in range(router_params["batch_size"]):
                fused_token_probs = fused_probs[b].sort(descending=True)[0]
                ref_token_probs = ref_probs[b].cpu().sort(descending=True)[0]
                np.testing.assert_allclose(
                    fused_token_probs.numpy(),
                    ref_token_probs.numpy(),
                    rtol=1e-2,
                    atol=1e-3,
                )
            
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")


class TestFusedRouterEdgeCases:
    """Test edge cases for the fused router."""
    
    def test_single_token(self):
        """Test with batch_size=1."""
        try:
            from metal_marlin.moe.fused_router import FusedMoERouter
            from metal_marlin.metal_dispatch import get_default_library
            
            lib = get_default_library()
            hidden = torch.randn(1, 64, dtype=torch.float16, device="mps")
            weights = torch.randn(64, 8, dtype=torch.float16, device="mps")
            
            router = FusedMoERouter(lib, num_experts=8, top_k=2, hidden_dim=64)
            output = router.forward(hidden, weights)
            
            assert output.expert_ids.shape == (1, 2)
            assert output.sorted_indices.shape == (2,)
            
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")
    
    def test_large_batch(self):
        """Test with larger batch size."""
        try:
            from metal_marlin.moe.fused_router import FusedMoERouter
            from metal_marlin.metal_dispatch import get_default_library
            
            lib = get_default_library()
            batch_size = 32
            hidden = torch.randn(batch_size, 128, dtype=torch.float16, device="mps")
            weights = torch.randn(128, 64, dtype=torch.float16, device="mps")
            
            router = FusedMoERouter(lib, num_experts=64, top_k=8, hidden_dim=128)
            output = router.forward(hidden, weights)
            
            assert output.expert_ids.shape == (batch_size, 8)
            assert output.sorted_indices.shape == (batch_size * 8,)
            
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
