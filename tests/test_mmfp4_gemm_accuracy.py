"""Accuracy tests for MMFP4 GEMM and dequantization paths."""

import pytest
import torch
import numpy as np

from metal_marlin.layers.mmfp4_linear import _fast_dequant, mmfp4_gemm
from metal_marlin.quantize import pack_fp4_weights


def pack_mmfp4_rowwise(weights, scales_ignored, group_size):
    """Wrapper to use pack_fp4_weights and return row-packed layout."""
    # pack_fp4_weights expects [K, N] where K is quantization axis.
    # weights is [out, in]. We want to quantize along 'in'.
    # So pass weights.t() -> [in, out].
    # pack_fp4_weights returns packed [K//8, N] -> [in//8, out]
    # and scales [K//G, N] -> [in//G, out]
    packed, scales, _ = pack_fp4_weights(weights.t().cpu().numpy(), group_size=group_size, output_backend="torch")
    
    # Transpose packed to row-packed [out, in // 8]
    # packed is [in // 8, out] -> t() -> [out, in // 8]
    packed_row = packed.t().contiguous()
    
    # Scales are [in//G, out] i.e. [n_groups, out].
    # MMFP4Linear expects [n_groups, out] or [out, n_groups].
    # So we return scales as is.
    
    return packed_row, scales


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_fast_dequant_accuracy():
    """Test that _fast_dequant produces correct results."""
    torch.manual_seed(42)
    
    out_features, in_features = 128, 256
    group_size = 64
    n_groups = (in_features + group_size - 1) // group_size
    
    # Create test weights
    weights = torch.randn(out_features, in_features, dtype=torch.float32)
    # We ignore input scales since pack_fp4_weights computes them
    scales_dummy = torch.ones(n_groups, out_features, dtype=torch.float16)
    
    # Pack weights
    packed, scales = pack_mmfp4_rowwise(weights, scales_dummy, group_size)
    scales = scales.to(torch.float16)

    # Dequantize
    dequant = _fast_dequant(packed.to("mps"), scales.to("mps"), group_size)
    
    assert dequant.shape == (out_features, in_features)
    assert dequant.dtype == torch.float16
    assert torch.isfinite(dequant).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mmfp4_gemm_accuracy():
    """Test MMFP4 GEMM produces correct output shape and finite values."""
    torch.manual_seed(42)
    
    batch, in_features, out_features = 8, 256, 128
    group_size = 64
    n_groups = (in_features + group_size - 1) // group_size
    
    # Create inputs
    x = torch.randn(batch, in_features, dtype=torch.float16).to("mps")
    weights = torch.randn(out_features, in_features, dtype=torch.float32)
    scales_dummy = torch.ones(n_groups, out_features, dtype=torch.float16)
    
    # Pack weights
    packed, scales = pack_mmfp4_rowwise(weights, scales_dummy, group_size)
    
    # Run GEMM
    out = mmfp4_gemm(x, packed.to("mps"), scales.to("mps"), group_size)
    
    assert out.shape == (batch, out_features)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_fast_dequant_different_group_sizes():
    """Test _fast_dequant with various group sizes."""
    torch.manual_seed(42)
    
    out_features, in_features = 64, 128
    
    for group_size in [32, 64, 128]:
        n_groups = (in_features + group_size - 1) // group_size
        
        weights = torch.randn(out_features, in_features, dtype=torch.float32)
        scales_dummy = torch.ones(n_groups, out_features, dtype=torch.float16)
        
        packed, scales = pack_mmfp4_rowwise(weights, scales_dummy, group_size)
        
        dequant = _fast_dequant(packed.to("mps"), scales.to("mps"), group_size)
        
        assert dequant.shape == (out_features, in_features)
        assert torch.isfinite(dequant).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mmfp4_gemm_small_seq_len():
    """Test MMFP4 GEMM with small sequence length (M=4), which is smaller than TILE_M=64.
    
    This tests the boundary check fix for cases where M < TILE_M.
    """
    torch.manual_seed(42)
    
    seq_len, in_features, out_features = 4, 256, 128
    group_size = 64
    n_groups = (in_features + group_size - 1) // group_size
    
    # Create inputs
    x = torch.randn(seq_len, in_features, dtype=torch.float16).to("mps")
    weights = torch.randn(out_features, in_features, dtype=torch.float32)
    scales_dummy = torch.ones(n_groups, out_features, dtype=torch.float16)
    
    # Pack weights
    packed, scales = pack_mmfp4_rowwise(weights, scales_dummy, group_size)
    
    # Run GEMM
    out = mmfp4_gemm(x, packed.to("mps"), scales.to("mps"), group_size)
    
    assert out.shape == (seq_len, out_features)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all(), "Output contains Inf/NaN for small seq_len"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mmfp4_gemm_various_seq_lens():
    """Test MMFP4 GEMM with various sequence lengths to verify boundary handling."""
    torch.manual_seed(42)
    
    in_features, out_features = 128, 64
    group_size = 32
    n_groups = (in_features + group_size - 1) // group_size
    
    weights = torch.randn(out_features, in_features, dtype=torch.float32)
    scales_dummy = torch.ones(n_groups, out_features, dtype=torch.float16)
    packed, scales = pack_mmfp4_rowwise(weights, scales_dummy, group_size)
    packed = packed.to("mps")
    scales = scales.to("mps")
    
    # Test various sequence lengths, including edge cases around tile boundaries
    for seq_len in [1, 2, 4, 8, 16, 32, 33, 64, 65, 100]:
        x = torch.randn(seq_len, in_features, dtype=torch.float16).to("mps")
        out = mmfp4_gemm(x, packed, scales, group_size)
        
        assert out.shape == (seq_len, out_features), f"Shape mismatch for seq_len={seq_len}"
        assert out.dtype == torch.float16
        assert torch.isfinite(out).all(), f"Output contains Inf/NaN for seq_len={seq_len}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
