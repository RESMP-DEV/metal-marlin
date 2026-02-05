"""Tests for pipelined quantization."""

import numpy as np
import pytest
import torch

from metal_marlin.quantization.exl3_quantizer import EXL3Quantizer
from metal_marlin.quantization.pipelined_quant import (
    LayerInfo,
    compute_layer_sensitivity,
    quantize_moe_experts_fast,
)


def dummy_hessian_fn(activations: torch.Tensor) -> np.ndarray:
    """Simple Hessian approximation for testing."""
    X = activations.numpy()
    H = 2 * X.T @ X / X.shape[0]
    H += 0.01 * np.eye(H.shape[0])
    return H.astype(np.float64)


class TestLayerSensitivity:
    def test_sensitivity_increases_with_weight_norm(self):
        """Larger weights should have higher sensitivity."""
        act = torch.randn(100, 64)
        
        small_weight = torch.randn(32, 64) * 0.1
        large_weight = torch.randn(32, 64) * 10.0
        
        sens_small = compute_layer_sensitivity(small_weight, act)
        sens_large = compute_layer_sensitivity(large_weight, act)
        
        assert sens_large > sens_small
    
    def test_sensitivity_increases_with_activation_variance(self):
        """Higher activation variance should increase sensitivity."""
        weight = torch.randn(32, 64)
        
        low_var_act = torch.randn(100, 64) * 0.1
        high_var_act = torch.randn(100, 64) * 10.0
        
        sens_low = compute_layer_sensitivity(weight, low_var_act)
        sens_high = compute_layer_sensitivity(weight, high_var_act)
        
        assert sens_high > sens_low


class TestMoEQuantization:
    @pytest.fixture
    def small_experts(self):
        """Create small expert weights for testing."""
        return {
            "expert_0": torch.randn(64, 128),
            "expert_1": torch.randn(64, 128) * 0.5,
            "expert_2": torch.randn(64, 128) * 2.0,
            "expert_3": torch.randn(64, 128) * 0.1,
        }
    
    @pytest.fixture
    def expert_activations(self, small_experts):
        """Generate activations for each expert."""
        return {
            name: torch.randn(256, w.shape[1])
            for name, w in small_experts.items()
        }
    
    def test_sensitive_experts_get_higher_bits(self, small_experts, expert_activations):
        """Most sensitive experts should use higher precision."""
        quantizer = EXL3Quantizer(bits=3, group_size=64, use_metal=False)
        
        results, metadata = quantize_moe_experts_fast(
            small_experts,
            expert_activations,
            quantizer,
            dummy_hessian_fn,
            min_bits=3,
            max_bits=4,
        )
        
        # Count experts at each bit level
        bit_counts = {}
        for m in metadata.values():
            bits = m['bits']
            bit_counts[bits] = bit_counts.get(bits, 0) + 1
            
        assert 4 in bit_counts
        assert 3 in bit_counts
        assert sum(bit_counts.values()) == len(small_experts)
    
    def test_all_experts_quantized(self, small_experts, expert_activations):
        """All experts should have results."""
        quantizer = EXL3Quantizer(bits=3, group_size=64, use_metal=False)
        
        results, metadata = quantize_moe_experts_fast(
            small_experts,
            expert_activations,
            quantizer,
            dummy_hessian_fn,
            min_bits=3,
            max_bits=4,
        )
        
        assert set(results.keys()) == set(small_experts.keys())
        
        for name, result in results.items():
            assert result.trellis_indices is not None
            assert result.scales is not None
