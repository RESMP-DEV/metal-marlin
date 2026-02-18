"""Test _fast_dequant optimization maintains numerical accuracy."""
import pytest
import torch

from metal_marlin.layers.mmfp4_linear import (
    MMFP4Linear,
    _dequantize_rowwise_mmfp4,
    _fast_dequant,
)
from metal_marlin.quantize_fp4 import quantize_fp4


def create_test_weights(K: int, N: int, group_size: int = 128):
    """Create properly formatted test weights for MMFP4Linear.
    
    Args:
        K: in_features (input dimension)
        N: out_features (output dimension)
        group_size: quantization group size
    
    Returns:
        (packed_weights, scales) in MMFP4Linear format:
        - packed_weights: [N, K/8] uint32
        - scales: [n_groups, N] float16
    """
    # Create weight matrix
    W = torch.randn(N, K, dtype=torch.float32)
    
    # quantize_fp4 expects [out_features, in_features] and returns Marlin layout
    # packed: [K/8, N], scales: [K/gs, N]
    W_packed_marlin, scales_marlin = quantize_fp4(W, group_size=group_size)
    if not hasattr(W_packed_marlin, "to"):
        W_packed_marlin = torch.from_numpy(W_packed_marlin)
        scales_marlin = torch.from_numpy(scales_marlin)
    
    # MMFP4Linear expects:
    # - packed_weights: [out_features, in_features//8] = [N, K/8]
    # - scales: [n_groups, out_features] = [K/gs, N]
    
    # Marlin packed is [K/8, N], need to transpose to [N, K/8]
    W_packed = W_packed_marlin.t()
    # Marlin scales are already [K/gs, N]
    scales = scales_marlin
    
    return W_packed, scales


def test_fast_dequant_accuracy_cpu():
    """Verify _fast_dequant matches _dequantize_rowwise_mmfp4 on CPU."""
    # Create test weights
    K, N = 512, 256  # in_features, out_features
    W_packed, scales = create_test_weights(K, N, group_size=128)
    
    # Test on CPU
    W_packed_cpu = W_packed.cpu()
    scales_cpu = scales.cpu()
    
    # Original dequantization
    weight_orig = _dequantize_rowwise_mmfp4(W_packed_cpu, scales_cpu, group_size=128)
    
    # Optimized dequantization
    weight_fast = _fast_dequant(W_packed_cpu, scales_cpu, group_size=128)
    
    # Check shapes match
    assert weight_orig.shape == weight_fast.shape, f"Shape mismatch: {weight_orig.shape} vs {weight_fast.shape}"
    assert weight_orig.shape == (N, K), f"Expected shape {(N, K)}, got {weight_orig.shape}"
    
    # Check values match (within FP16 precision)
    diff = (weight_orig.float() - weight_fast.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    assert max_diff < 0.01, f"Max difference too large: {max_diff}"
    assert mean_diff < 0.001, f"Mean difference too large: {mean_diff}"
    
    print(f"✓ CPU test passed: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_fast_dequant_accuracy_mps():
    """Verify _fast_dequant matches _dequantize_rowwise_mmfp4 on MPS."""
    # Create test weights
    K, N = 512, 256
    W_packed, scales = create_test_weights(K, N, group_size=128)
    
    # Test on MPS
    device = torch.device("mps")
    W_packed_mps = W_packed.to(device)
    scales_mps = scales.to(device)
    
    # Original dequantization
    weight_orig = _dequantize_rowwise_mmfp4(W_packed_mps, scales_mps, group_size=128)
    
    # Optimized dequantization
    weight_fast = _fast_dequant(W_packed_mps, scales_mps, group_size=128)
    
    # Check shapes match
    assert weight_orig.shape == weight_fast.shape
    
    # Check values match (within FP16 precision)
    diff = (weight_orig.float() - weight_fast.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    assert max_diff < 0.01, f"Max difference too large: {max_diff}"
    assert mean_diff < 0.001, f"Mean difference too large: {mean_diff}"
    
    print(f"✓ MPS test passed: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")


def test_fast_dequant_shape_variations():
    """Test _fast_dequant with various input shapes."""
    test_cases = [
        (128, 64),    # Small
        (512, 256),   # Medium
        (2048, 1024), # Large
        (4096, 4096), # Very large
    ]
    
    for K, N in test_cases:
        W_packed, scales = create_test_weights(K, N, group_size=128)
        
        weight_orig = _dequantize_rowwise_mmfp4(W_packed, scales, group_size=128)
        weight_fast = _fast_dequant(W_packed, scales, group_size=128)
        
        assert weight_orig.shape == weight_fast.shape == (N, K)
        
        # Check values match
        diff = (weight_orig.float() - weight_fast.float()).abs()
        max_diff = diff.max().item()
        
        assert max_diff < 0.01, f"Shape (K={K}, N={N}): max_diff={max_diff} too large"
        print(f"✓ Shape (K={K}, N={N}): max_diff={max_diff:.6f}")


def test_fast_dequant_different_group_sizes():
    """Test _fast_dequant with different group sizes."""
    K, N = 512, 256
    
    for group_size in [32, 64, 128, 256]:
        W_packed, scales = create_test_weights(K, N, group_size=group_size)
        
        weight_orig = _dequantize_rowwise_mmfp4(W_packed, scales, group_size=group_size)
        weight_fast = _fast_dequant(W_packed, scales, group_size=group_size)
        
        diff = (weight_orig.float() - weight_fast.float()).abs()
        max_diff = diff.max().item()
        
        assert max_diff < 0.01, f"Group size {group_size}: max_diff={max_diff} too large"
        print(f"✓ Group size {group_size}: max_diff={max_diff:.6f}")


def test_mmfp4_linear_with_fast_dequant():
    """Test that MMFP4Linear works correctly with _fast_dequant."""
    K, N = 512, 256  # in_features, out_features
    W_packed, scales = create_test_weights(K, N, group_size=128)
    
    # Create MMFP4Linear layer
    layer = MMFP4Linear(
        packed_weights=W_packed,
        scales=scales,
        bias=None,
        group_size=128,
    )
    
    # Test forward pass
    x = torch.randn(1, K, dtype=torch.float32)
    output = layer(x)
    
    assert output.shape == (1, N), f"Expected output shape (1, {N}), got {output.shape}"
    assert torch.isfinite(output).all(), "Output contains non-finite values"
    
    print(f"✓ MMFP4Linear forward pass: output shape {tuple(output.shape)}, mean={output.mean().item():.4f}")


def test_fast_dequant_produces_correct_dtype():
    """Verify _fast_dequant produces float16 output."""
    K, N = 512, 256
    W_packed, scales = create_test_weights(K, N, group_size=128)
    
    weight = _fast_dequant(W_packed, scales, group_size=128)
    
    assert weight.dtype == torch.float16, f"Expected float16, got {weight.dtype}"
    print("✓ Output dtype is float16")


def test_fast_dequant_produces_contiguous():
    """Verify _fast_dequant produces contiguous output."""
    K, N = 512, 256
    W_packed, scales = create_test_weights(K, N, group_size=128)
    
    weight = _fast_dequant(W_packed, scales, group_size=128)
    
    assert weight.is_contiguous(), "Output is not contiguous"
    print("✓ Output is contiguous")


if __name__ == "__main__":
    test_fast_dequant_accuracy_cpu()
    test_fast_dequant_shape_variations()
    test_fast_dequant_different_group_sizes()
    test_mmfp4_linear_with_fast_dequant()
    test_fast_dequant_produces_correct_dtype()
    test_fast_dequant_produces_contiguous()
    print("\n✓ All tests passed!")
