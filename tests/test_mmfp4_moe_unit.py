import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# Import the module under test
from metal_marlin.layers import mmfp4_moe
from metal_marlin.layers.mmfp4_moe import MMFP4Expert, MMFP4MoE


class TestMMFP4MoEUnit:
    @pytest.fixture
    def mock_dispatch(self):
        """Mock the dispatch module."""
        mock = MagicMock()
        # Patch the global variable in mmfp4_moe that caches the module
        with patch.object(mmfp4_moe, '_moe_dispatch_module', mock):
            yield mock

    @pytest.fixture(autouse=True)
    def setup_router_mock(self):
        """Setup router mock before each test."""
        # Setup default fused router mock
        def mock_fused_router(hidden, gate, top_k, **kwargs):
            batch_size = hidden.shape[0]
            return (
                torch.ones(batch_size, top_k) * 0.5,  # topk_weights
                torch.randint(0, 4, (batch_size, top_k)),  # topk_indices
            )
        
        # Patch the function where it is imported in mmfp4_moe
        with patch('metal_marlin.layers.mmfp4_moe._fused_router_topk', side_effect=mock_fused_router):
            yield

    @pytest.fixture
    def moe_layer(self):
        """Create a small MoE layer for testing."""
        layer = MMFP4MoE(
            n_experts=4,
            n_experts_per_tok=2,
            hidden_size=32,
            moe_intermediate_size=16,
            group_size=32,
            has_shared_expert=True
        )
        return layer

    def test_decode_path_uses_optimized_forward(self, moe_layer, mock_dispatch):
        """Test that single-token input uses _forward_decode_optimized path.
        
        This is the _moe_decode_optimized path - it should:
        1. Skip expensive sort/gather/scatter operations
        2. Use the fused decode kernel directly (via moe_dispatch)
        """
        x = torch.randn(1, 32)  # Single token triggers decode path
        
        # Configure the mock dispatch function to return a tensor of correct shape
        mock_dispatch.decode_optimized_expert_combine_fused.return_value = torch.randn(1, 32)
        
        output = moe_layer(x)
        
        # Verify output shape
        assert output.shape == (1, 32)
        
        # Verify that the fused decode function was called
        # This confirms we are using the optimized path which delegates to the fused implementation
        mock_dispatch.decode_optimized_expert_combine_fused.assert_called_once()
        
        # Verify arguments
        call_args = mock_dispatch.decode_optimized_expert_combine_fused.call_args
        assert call_args is not None
        kwargs = call_args.kwargs
        
        # Check that we passed the correct experts and routing info
        assert kwargs['experts'] is moe_layer.experts
        assert kwargs['shared_expert'] is moe_layer.shared_experts
        
        # Verify topk_indices and topk_weights have correct shape and properties
        assert kwargs['topk_indices'].shape == (1, 2)  # [batch, top_k]
        assert kwargs['topk_weights'].shape == (1, 2)  # [batch, top_k]
        assert kwargs['topk_indices'].dtype == torch.int64
        # _minimal_routing_overhead: Weights should already be normalized (sum to 1)
        weights_sum = kwargs['topk_weights'].sum(dim=-1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5)

    def test_decode_path_skips_group_tokens(self, moe_layer, mock_dispatch):
        """Verify decode path avoids expensive token grouping operations.
        
        The _moe_decode_optimized path should NOT call:
        - group_tokens_by_expert_full
        - gather_for_experts  
        - scatter_expert_outputs
        """
        x = torch.randn(1, 32)
        
        # Configure mock return value
        mock_dispatch.decode_optimized_expert_combine_fused.return_value = torch.randn(1, 32)
        
        output = moe_layer(x)
        
        # Verify output
        assert output.shape == (1, 32)
        
        # In decode path, these should NOT be called
        mock_dispatch.group_tokens_by_expert_full.assert_not_called()
        mock_dispatch.gather_for_experts.assert_not_called()
        mock_dispatch.scatter_expert_outputs.assert_not_called()

    def test_batch_path_not_decode(self, moe_layer, mock_dispatch):
        """Test that batch input (>1 token) does NOT use decode optimized path.
        
        Batch inputs should take the standard dispatch path, not the
        _moe_decode_optimized single-token fast path.
        """
        # Use a shape that results in >1 tokens after reshape
        x = torch.randn(2, 2, 32)  # 4 tokens total
        
        # Configure mocks for batch path
        mock_dispatch.group_tokens_by_expert_full.return_value = MagicMock()
        mock_dispatch.gather_for_experts.return_value = torch.randn(4, 32)
        mock_dispatch.dispatch_experts_batched_dynamic.return_value = torch.randn(4, 32)
        mock_dispatch.scatter_expert_outputs.return_value = torch.randn(4, 32)
        
        # Verify that hidden_flat.shape[0] will be > 1 (batch path)
        hidden_flat = x.reshape(-1, 32)
        assert hidden_flat.shape[0] == 4 > 1
        
        output = moe_layer(x)
        
        # Verify decode path was NOT used
        mock_dispatch.decode_optimized_expert_combine_fused.assert_not_called()
        
        # Verify batch path WAS used
        mock_dispatch.group_tokens_by_expert_full.assert_called_once()
        mock_dispatch.gather_for_experts.assert_called_once()
        mock_dispatch.dispatch_experts_batched_dynamic.assert_called_once()
        mock_dispatch.scatter_expert_outputs.assert_called_once()

    def test_forward_decode_optimized_exists(self):
        """Verify _forward_decode_optimized method exists and is callable."""
        layer = MMFP4MoE(n_experts=4, n_experts_per_tok=2, hidden_size=32)
        assert hasattr(layer, '_forward_decode_optimized')
        assert callable(getattr(layer, '_forward_decode_optimized'))

    def test_decode_optimized_docstring(self):
        """Verify _forward_decode_optimized has proper documentation."""
        layer = MMFP4MoE(n_experts=4, n_experts_per_tok=2, hidden_size=32)
        docstring = layer._forward_decode_optimized.__doc__
        assert docstring is not None
        assert 'decode' in docstring.lower()
        assert '_moe_decode_optimized' in docstring