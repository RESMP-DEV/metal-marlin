"""Test Viterbi quantization implementation."""

import numpy as np
import pytest

from metal_marlin.quantization.viterbi_quant import (
    TrellisCodebook,
    compute_quantization_error,
    quantize_tile_greedy,
    quantize_tile_viterbi,
)


def test_viterbi_quantizer():
    """Quick sanity test for Viterbi quantizer."""
    np.random.seed(42)
    tile = np.random.randn(16, 16).astype(np.float32)
    codebook = TrellisCodebook(bits=4)
    scale = 1.0

    indices, dequantized = quantize_tile_viterbi(tile, codebook, scale)

    # Basic sanity checks
    assert indices.dtype == np.int16
    assert dequantized.shape == (16, 16)
    assert np.all(indices >= 0)
    assert np.all(indices < 16)

    # Compute error
    error = compute_quantization_error(tile, dequantized)
    print(f"Quantization error: {error:.6f}")

    # Error should be reasonable for 4-bit quantization
    assert error < 1.0


def test_quantize_tile_shape():
    """Test output shapes are correct."""
    tile = np.random.randn(16, 16).astype(np.float32)
    codebook = TrellisCodebook(bits=3)
    scale = 1.0

    indices, dequantized = quantize_tile_viterbi(tile, codebook, scale)

    assert indices.shape == (256,)
    assert indices.dtype == np.int16
    assert dequantized.shape == (16, 16)
    assert dequantized.dtype == np.float32


def test_viterbi_vs_greedy():
    """Viterbi should produce error <= greedy."""
    tile = np.random.randn(16, 16).astype(np.float32)
    codebook = TrellisCodebook(bits=3)
    scale = 1.0

    indices_viterbi, deq_viterbi = quantize_tile_viterbi(tile, codebook, scale)
    indices_greedy, deq_greedy = quantize_tile_greedy(tile, codebook, scale)

    error_viterbi = compute_quantization_error(tile, deq_viterbi)
    error_greedy = compute_quantization_error(tile, deq_greedy)

    # Viterbi is optimal, so error should be <= greedy (with small tolerance)
    assert error_viterbi <= error_greedy * 1.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
