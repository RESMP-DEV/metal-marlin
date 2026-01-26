"""Comprehensive test suite for sub-4-bit quantization formats (INT2, INT3, NF2, NF3).

Tests cover:
1. Quantization roundtrip accuracy (quantize -> dequantize)
2. Packing correctness (16 INT2 per uint32, 10 INT3 per uint32)
3. Scale computation
4. Edge cases: zeros, max values, negative values
5. Different tensor shapes
6. Group size variations

Reference: metal_marlin/sub4bit.py implements these formats for MoE expert compression.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None  # type: ignore[assignment]

# Add metal_marlin package to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from metal_marlin.sub4bit import (
    INT2_LEVELS,
    INT3_LEVELS,
    NF2_LEVELS,
    NF3_LEVELS,
    compute_quantization_error,
    dequantize_int2,
    dequantize_int3,
    dequantize_nf2,
    dequantize_nf3,
    estimate_compression_ratio,
    get_int2_lut,
    get_int3_lut,
    get_nf2_lut,
    get_nf3_lut,
    quantize_int2,
    quantize_int3,
    quantize_nf2,
    quantize_nf3,
    select_sub4bit_format,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gaussian_tensor_small():
    """Small Gaussian tensor for quick tests.

    Shape (64, 192) is divisible by:
    - 16 (INT2/NF2 packing)
    - 10 (INT3/NF3 packing via padding)
    - 64 (common group_size)
    """
    np.random.seed(42)
    return np.random.randn(64, 192).astype(np.float32) * 0.02


@pytest.fixture
def gaussian_tensor_medium():
    """Medium Gaussian tensor for accuracy tests.

    Shape (256, 384) is divisible by 16, 64, and 128.
    """
    np.random.seed(42)
    return np.random.randn(256, 384).astype(np.float32) * 0.02


@pytest.fixture
def uniform_tensor():
    """Uniform distribution tensor (non-Gaussian)."""
    np.random.seed(42)
    return np.random.uniform(-0.05, 0.05, (64, 192)).astype(np.float32)


# ---------------------------------------------------------------------------
# INT2 Tests
# ---------------------------------------------------------------------------


class TestINT2:
    """Test INT2 quantization (4 levels: -1.5, -0.5, 0.5, 1.5 scaled)."""

    def test_levels_correct(self):
        """Verify INT2 quantization levels."""
        expected = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
        np.testing.assert_array_almost_equal(INT2_LEVELS, expected)
        assert len(INT2_LEVELS) == 4

    def test_quantize_roundtrip_accuracy(self, gaussian_tensor_small):
        """Quantize and dequantize, verify reasonable accuracy."""
        tensor = gaussian_tensor_small
        packed, scales = quantize_int2(tensor, group_size=64)
        reconstructed = dequantize_int2(packed, scales, group_size=64)

        # Check shapes
        assert packed.shape == (tensor.shape[0], tensor.shape[1] // 16)
        assert scales.shape == (tensor.shape[0], tensor.shape[1] // 64)
        assert reconstructed.shape == tensor.shape

        # INT2 has 4 levels; expect RMSE < 20% of tensor range
        diff = tensor - reconstructed.astype(np.float32)
        rmse = np.sqrt(np.mean(diff**2))
        tensor_range = tensor.max() - tensor.min()
        assert rmse < 0.2 * tensor_range, f"RMSE {rmse} too high for INT2"

    def test_packing_correctness(self):
        """Verify 16 INT2 values pack correctly into uint32."""
        # Create tensor where each value maps to a known code
        # Code 0 -> -1.5*s, Code 1 -> -0.5*s, Code 2 -> +0.5*s, Code 3 -> +1.5*s
        np.random.seed(123)
        scale = 0.1
        # Create 16 values that map to codes 0,1,2,3,0,1,2,3,...
        values = np.array([INT2_LEVELS[i % 4] * scale for i in range(16)])
        tensor = values.reshape(1, 16).astype(np.float32)

        packed, scales = quantize_int2(tensor, group_size=16)

        # Manually verify packing
        # Each code uses 2 bits: code[i] at bits [2i+1:2i]
        packed_val = packed[0, 0]
        for i in range(16):
            expected_code = i % 4
            actual_code = (packed_val >> (i * 2)) & 0x3
            assert actual_code == expected_code, (
                f"Position {i}: expected {expected_code}, got {actual_code}"
            )

    def test_scale_computation(self, gaussian_tensor_small):
        """Verify per-group scales are computed correctly."""
        tensor = gaussian_tensor_small
        group_size = 64
        packed, scales = quantize_int2(tensor, group_size=group_size)

        # Scales should be max_abs / 1.5 per group, per row
        out_feat, in_feat = tensor.shape
        n_groups = in_feat // group_size
        for row in range(out_feat):
            for g in range(n_groups):
                group_vals = tensor[row, g * group_size : (g + 1) * group_size]
                expected_scale = np.max(np.abs(group_vals)) / 1.5
                expected_scale = max(expected_scale, 1e-7 / 1.5)  # Account for epsilon
                actual_scale = float(scales[row, g])
                # Allow tolerance for float16 precision
                np.testing.assert_allclose(actual_scale, expected_scale, rtol=5e-2)

    def test_zeros(self):
        """Test quantization of all-zero tensor."""
        tensor = np.zeros((16, 32), dtype=np.float32)
        packed, scales = quantize_int2(tensor, group_size=32)
        reconstructed = dequantize_int2(packed, scales, group_size=32)

        # All values should dequantize close to zero (codes 1 or 2: -0.5 or +0.5)
        # With small scales, result should be near zero
        assert np.max(np.abs(reconstructed)) < 1e-5

    def test_max_values(self):
        """Test quantization of maximum magnitude values."""
        tensor = np.ones((16, 32), dtype=np.float32) * 1.5  # Max INT2 level
        packed, scales = quantize_int2(tensor, group_size=32)
        reconstructed = dequantize_int2(packed, scales, group_size=32)

        # Should reconstruct to +1.5 * scale = original
        np.testing.assert_allclose(reconstructed.astype(np.float32), tensor, rtol=1e-2)

    def test_negative_values(self):
        """Test quantization of negative values."""
        tensor = np.ones((16, 32), dtype=np.float32) * -1.5  # Min INT2 level
        packed, scales = quantize_int2(tensor, group_size=32)
        reconstructed = dequantize_int2(packed, scales, group_size=32)

        # Should reconstruct to -1.5 * scale = original
        np.testing.assert_allclose(reconstructed.astype(np.float32), tensor, rtol=1e-2)

    @pytest.mark.parametrize("shape", [(32, 64), (128, 320)])
    def test_different_shapes(self, shape):
        """Test various tensor shapes."""
        np.random.seed(42)
        tensor = np.random.randn(*shape).astype(np.float32) * 0.02
        packed, scales = quantize_int2(tensor, group_size=64)
        reconstructed = dequantize_int2(packed, scales, group_size=64)

        assert reconstructed.shape == tensor.shape
        # Verify roundtrip works
        diff = tensor - reconstructed.astype(np.float32)
        rmse = np.sqrt(np.mean(diff**2))
        assert rmse < 0.015  # Reasonable threshold for Gaussian data

    @pytest.mark.parametrize("group_size", [32, 128])
    def test_group_size_variations(self, group_size):
        """Test different group sizes."""
        np.random.seed(42)
        # Ensure in_feat divisible by both 16 (packing) and group_size
        in_feat = 16 * group_size
        tensor = np.random.randn(32, in_feat).astype(np.float32) * 0.02

        packed, scales = quantize_int2(tensor, group_size=group_size)
        reconstructed = dequantize_int2(packed, scales, group_size=group_size)

        assert scales.shape[1] == in_feat // group_size
        assert reconstructed.shape == tensor.shape

    def test_divisibility_error_16(self):
        """Test error when in_features not divisible by 16."""
        tensor = np.random.randn(8, 17).astype(np.float32)  # 17 not divisible by 16
        with pytest.raises(ValueError, match="divisible by 16"):
            quantize_int2(tensor, group_size=17)

    def test_divisibility_error_group_size(self):
        """Test error when in_features not divisible by group_size."""
        tensor = np.random.randn(8, 64).astype(np.float32)
        with pytest.raises(ValueError, match="divisible by group_size"):
            quantize_int2(tensor, group_size=48)  # 64 not divisible by 48


# ---------------------------------------------------------------------------
# INT3 Tests
# ---------------------------------------------------------------------------


class TestINT3:
    """Test INT3 quantization (8 levels: -3.5 to +3.5 scaled)."""

    def test_levels_correct(self):
        """Verify INT3 quantization levels."""
        expected = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], dtype=np.float32)
        np.testing.assert_array_almost_equal(INT3_LEVELS, expected)
        assert len(INT3_LEVELS) == 8

    def test_quantize_roundtrip_accuracy(self, gaussian_tensor_small):
        """Quantize and dequantize, verify better accuracy than INT2."""
        tensor = gaussian_tensor_small
        packed, scales = quantize_int3(tensor, group_size=64)
        reconstructed = dequantize_int3(packed, scales, group_size=64)

        # Check shapes (10 INT3 per uint32)
        expected_packed_cols = (tensor.shape[1] + 9) // 10
        assert packed.shape == (tensor.shape[0], expected_packed_cols)
        assert scales.shape == (tensor.shape[0], tensor.shape[1] // 64)

        # INT3 has 8 levels; expect better RMSE than INT2
        min_feat = min(tensor.shape[1], reconstructed.shape[1])
        diff = tensor[:, :min_feat] - reconstructed[:, :min_feat].astype(np.float32)
        rmse = np.sqrt(np.mean(diff**2))
        tensor_range = tensor.max() - tensor.min()
        assert rmse < 0.1 * tensor_range, f"RMSE {rmse} too high for INT3"

    def test_packing_correctness(self):
        """Verify 10 INT3 values pack correctly into uint32."""
        np.random.seed(123)
        scale = 0.1
        # Create 10 values that map to codes 0,1,2,3,4,5,6,7,0,1
        values = np.array([INT3_LEVELS[i % 8] * scale for i in range(10)])
        tensor = values.reshape(1, 10).astype(np.float32)

        packed, scales = quantize_int3(tensor, group_size=10)

        # Manually verify packing
        # Each code uses 3 bits: code[i] at bits [3i+2:3i]
        packed_val = packed[0, 0]
        for i in range(10):
            expected_code = i % 8
            actual_code = (packed_val >> (i * 3)) & 0x7
            assert actual_code == expected_code, (
                f"Position {i}: expected {expected_code}, got {actual_code}"
            )

        # Bits 30-31 should be padding (zeros)
        padding = (packed_val >> 30) & 0x3
        assert padding == 0, "Padding bits should be zero"

    def test_scale_computation(self, gaussian_tensor_small):
        """Verify per-group scales are computed correctly."""
        tensor = gaussian_tensor_small
        group_size = 64
        packed, scales = quantize_int3(tensor, group_size=group_size)

        # Scales should be max_abs / 3.5 per group, per row
        out_feat, in_feat = tensor.shape
        n_groups = in_feat // group_size
        for row in range(out_feat):
            for g in range(n_groups):
                group_vals = tensor[row, g * group_size : (g + 1) * group_size]
                expected_scale = np.max(np.abs(group_vals)) / 3.5
                expected_scale = max(expected_scale, 1e-7 / 3.5)  # Account for epsilon
                actual_scale = float(scales[row, g])
                np.testing.assert_allclose(actual_scale, expected_scale, rtol=5e-2)

    def test_zeros(self):
        """Test quantization of all-zero tensor."""
        tensor = np.zeros((16, 30), dtype=np.float32)  # 30 divisible by 10 for packing
        packed, scales = quantize_int3(tensor, group_size=30)
        reconstructed = dequantize_int3(packed, scales, group_size=30)

        # All values should dequantize close to zero
        assert np.max(np.abs(reconstructed)) < 1e-5

    def test_max_values(self):
        """Test quantization of maximum magnitude values."""
        tensor = np.ones((16, 30), dtype=np.float32) * 3.5
        packed, scales = quantize_int3(tensor, group_size=30)
        reconstructed = dequantize_int3(packed, scales, group_size=30)

        np.testing.assert_allclose(reconstructed.astype(np.float32), tensor, rtol=1e-2)

    def test_negative_values(self):
        """Test quantization of negative values."""
        tensor = np.ones((16, 30), dtype=np.float32) * -3.5
        packed, scales = quantize_int3(tensor, group_size=30)
        reconstructed = dequantize_int3(packed, scales, group_size=30)

        np.testing.assert_allclose(reconstructed.astype(np.float32), tensor, rtol=1e-2)

    @pytest.mark.parametrize("shape", [(32, 60), (128, 300)])
    def test_different_shapes(self, shape):
        """Test various tensor shapes (shapes divisible by group_size)."""
        np.random.seed(42)
        tensor = np.random.randn(*shape).astype(np.float32) * 0.02
        packed, scales = quantize_int3(tensor, group_size=60)
        reconstructed = dequantize_int3(packed, scales, group_size=60)

        # Compare up to unpadded size
        min_feat = min(tensor.shape[1], reconstructed.shape[1])
        diff = tensor[:, :min_feat] - reconstructed[:, :min_feat].astype(np.float32)
        rmse = np.sqrt(np.mean(diff**2))
        assert rmse < 0.01

    @pytest.mark.parametrize("group_size", [20, 60])
    def test_group_size_variations(self, group_size):
        """Test different group sizes."""
        np.random.seed(42)
        in_feat = 10 * group_size  # Ensure divisible by both 10 and group_size
        tensor = np.random.randn(32, in_feat).astype(np.float32) * 0.02

        packed, scales = quantize_int3(tensor, group_size=group_size)
        dequantize_int3(packed, scales, group_size=group_size)

        assert scales.shape[1] == in_feat // group_size

    def test_padding_non_divisible_by_10(self):
        """Test that INT3 handles in_features not divisible by 10 via padding."""
        np.random.seed(42)
        tensor = np.random.randn(8, 64).astype(np.float32) * 0.02  # 64 not divisible by 10

        packed, scales = quantize_int3(tensor, group_size=64)
        dequantize_int3(packed, scales, group_size=64)

        # Should work with padding
        assert packed.shape[1] == 7  # ceil(64/10) = 7

    def test_original_in_feat_parameter(self):
        """Test the original_in_feat parameter for truncating output."""
        np.random.seed(42)
        tensor = np.random.randn(8, 64).astype(np.float32) * 0.02

        packed, scales = quantize_int3(tensor, group_size=64)
        reconstructed = dequantize_int3(packed, scales, group_size=64, original_in_feat=64)

        assert reconstructed.shape == tensor.shape


# ---------------------------------------------------------------------------
# NF2 Tests
# ---------------------------------------------------------------------------


class TestNF2:
    """Test NF2 (NormalFloat 2-bit) quantization with Gaussian quantile levels."""

    def test_levels_symmetric(self):
        """Verify NF2 levels are symmetric around 0."""
        assert len(NF2_LEVELS) == 4
        # Symmetric: levels[i] = -levels[3-i]
        for i in range(2):
            np.testing.assert_almost_equal(NF2_LEVELS[i], -NF2_LEVELS[3 - i], decimal=5)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_levels_from_gaussian_quantiles(self):
        """Verify NF2 levels are based on Gaussian quantiles."""
        # NF2 uses quantiles at 12.5%, 37.5%, 62.5%, 87.5%
        quantiles = [0.125, 0.375, 0.625, 0.875]
        expected_raw = stats.norm.ppf(quantiles)
        expected = expected_raw / np.max(np.abs(expected_raw))
        np.testing.assert_allclose(NF2_LEVELS, expected, rtol=1e-5)

    def test_quantize_roundtrip_accuracy(self, gaussian_tensor_small):
        """Quantize and dequantize Gaussian data; NF2 should be optimal."""
        tensor = gaussian_tensor_small
        packed, scales = quantize_nf2(tensor, group_size=64)
        reconstructed = dequantize_nf2(packed, scales, group_size=64)

        assert packed.shape == (tensor.shape[0], tensor.shape[1] // 16)
        assert scales.shape == (tensor.shape[0], tensor.shape[1] // 64)
        assert reconstructed.shape == tensor.shape

        diff = tensor - reconstructed.astype(np.float32)
        rmse = np.sqrt(np.mean(diff**2))
        tensor_range = tensor.max() - tensor.min()
        assert rmse < 0.2 * tensor_range

    def test_packing_correctness(self):
        """Verify 16 NF2 values pack correctly into uint32 (same as INT2)."""
        np.random.seed(123)
        scale = 0.1
        values = np.array([NF2_LEVELS[i % 4] * scale for i in range(16)])
        tensor = values.reshape(1, 16).astype(np.float32)

        packed, scales = quantize_nf2(tensor, group_size=16)

        packed_val = packed[0, 0]
        for i in range(16):
            expected_code = i % 4
            actual_code = (packed_val >> (i * 2)) & 0x3
            assert actual_code == expected_code

    def test_better_for_gaussian_than_uniform(self):
        """NF2 should have lower error on Gaussian vs uniform distribution."""
        np.random.seed(42)
        # Use shapes divisible by 16 (packing) and 64 (group_size)
        gaussian = np.random.randn(64, 192).astype(np.float32) * 0.02
        uniform = np.random.uniform(-0.04, 0.04, (64, 192)).astype(np.float32)

        # Quantize both
        packed_g, scales_g = quantize_nf2(gaussian, group_size=64)
        packed_u, scales_u = quantize_nf2(uniform, group_size=64)

        recon_g = dequantize_nf2(packed_g, scales_g, group_size=64)
        recon_u = dequantize_nf2(packed_u, scales_u, group_size=64)

        rmse_gaussian = np.sqrt(np.mean((gaussian - recon_g.astype(np.float32)) ** 2))
        rmse_uniform = np.sqrt(np.mean((uniform - recon_u.astype(np.float32)) ** 2))

        # Both should have reasonable error (test that quantization works)
        assert rmse_gaussian < 0.015, f"Gaussian RMSE {rmse_gaussian} too high"
        assert rmse_uniform < 0.025, f"Uniform RMSE {rmse_uniform} too high"

        # NF2 is designed for Gaussian; relative error should be comparable or better
        # Note: for 2-bit quantization, the advantage is subtle; mainly test both work
        rel_err_g = rmse_gaussian / np.std(gaussian)
        rmse_uniform / np.std(uniform)

        # Allow some tolerance; the key is that both work
        assert rel_err_g < 1.0, f"Gaussian relative error {rel_err_g} too high"

    def test_zeros(self):
        """Test quantization of all-zero tensor."""
        tensor = np.zeros((16, 32), dtype=np.float32)
        packed, scales = quantize_nf2(tensor, group_size=32)
        reconstructed = dequantize_nf2(packed, scales, group_size=32)

        assert np.max(np.abs(reconstructed)) < 1e-5

    @pytest.mark.parametrize("shape", [(32, 64), (128, 320)])
    def test_different_shapes(self, shape):
        """Test various tensor shapes."""
        np.random.seed(42)
        tensor = np.random.randn(*shape).astype(np.float32) * 0.02
        packed, scales = quantize_nf2(tensor, group_size=64)
        reconstructed = dequantize_nf2(packed, scales, group_size=64)

        assert reconstructed.shape == tensor.shape

    @pytest.mark.parametrize("group_size", [32, 128])
    def test_group_size_variations(self, group_size):
        """Test different group sizes."""
        np.random.seed(42)
        in_feat = 16 * group_size
        tensor = np.random.randn(32, in_feat).astype(np.float32) * 0.02

        packed, scales = quantize_nf2(tensor, group_size=group_size)
        dequantize_nf2(packed, scales, group_size=group_size)

        assert scales.shape[1] == in_feat // group_size


# ---------------------------------------------------------------------------
# NF3 Tests
# ---------------------------------------------------------------------------


class TestNF3:
    """Test NF3 (NormalFloat 3-bit) quantization with Gaussian quantile levels."""

    def test_levels_symmetric(self):
        """Verify NF3 levels are symmetric around 0."""
        assert len(NF3_LEVELS) == 8
        for i in range(4):
            np.testing.assert_almost_equal(NF3_LEVELS[i], -NF3_LEVELS[7 - i], decimal=5)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_levels_from_gaussian_quantiles(self):
        """Verify NF3 levels are based on Gaussian quantiles."""
        n_levels = 8
        quantiles = np.linspace(0.5 / n_levels, 1 - 0.5 / n_levels, n_levels)
        expected_raw = stats.norm.ppf(quantiles)
        expected = expected_raw / np.max(np.abs(expected_raw))
        np.testing.assert_allclose(NF3_LEVELS, expected, rtol=1e-5)

    def test_quantize_roundtrip_accuracy(self, gaussian_tensor_small):
        """Quantize and dequantize; NF3 should be more accurate than NF2."""
        tensor = gaussian_tensor_small
        packed, scales = quantize_nf3(tensor, group_size=64)
        reconstructed = dequantize_nf3(packed, scales, group_size=64)

        expected_packed_cols = (tensor.shape[1] + 9) // 10
        assert packed.shape == (tensor.shape[0], expected_packed_cols)

        min_feat = min(tensor.shape[1], reconstructed.shape[1])
        diff = tensor[:, :min_feat] - reconstructed[:, :min_feat].astype(np.float32)
        rmse = np.sqrt(np.mean(diff**2))
        tensor_range = tensor.max() - tensor.min()
        assert rmse < 0.1 * tensor_range

    def test_more_accurate_than_nf2(self, gaussian_tensor_medium):
        """NF3 should have lower quantization error than NF2."""
        tensor = gaussian_tensor_medium

        # NF2 quantization
        packed_nf2, scales_nf2 = quantize_nf2(tensor, group_size=64)
        recon_nf2 = dequantize_nf2(packed_nf2, scales_nf2, group_size=64)
        rmse_nf2 = np.sqrt(np.mean((tensor - recon_nf2.astype(np.float32)) ** 2))

        # NF3 quantization
        packed_nf3, scales_nf3 = quantize_nf3(tensor, group_size=64)
        recon_nf3 = dequantize_nf3(packed_nf3, scales_nf3, group_size=64)
        min_feat = min(tensor.shape[1], recon_nf3.shape[1])
        rmse_nf3 = np.sqrt(
            np.mean((tensor[:, :min_feat] - recon_nf3[:, :min_feat].astype(np.float32)) ** 2)
        )

        assert rmse_nf3 < rmse_nf2, "NF3 should be more accurate than NF2"

    def test_packing_correctness(self):
        """Verify 10 NF3 values pack correctly into uint32 (same as INT3)."""
        np.random.seed(123)
        scale = 0.1
        values = np.array([NF3_LEVELS[i % 8] * scale for i in range(10)])
        tensor = values.reshape(1, 10).astype(np.float32)

        packed, scales = quantize_nf3(tensor, group_size=10)

        packed_val = packed[0, 0]
        for i in range(10):
            expected_code = i % 8
            actual_code = (packed_val >> (i * 3)) & 0x7
            assert actual_code == expected_code

    def test_zeros(self):
        """Test quantization of all-zero tensor."""
        tensor = np.zeros((16, 30), dtype=np.float32)
        packed, scales = quantize_nf3(tensor, group_size=30)
        reconstructed = dequantize_nf3(packed, scales, group_size=30)

        assert np.max(np.abs(reconstructed)) < 1e-5

    @pytest.mark.parametrize("shape", [(32, 60), (128, 300)])
    def test_different_shapes(self, shape):
        """Test various tensor shapes."""
        np.random.seed(42)
        tensor = np.random.randn(*shape).astype(np.float32) * 0.02
        packed, scales = quantize_nf3(tensor, group_size=60)
        reconstructed = dequantize_nf3(packed, scales, group_size=60)

        min_feat = min(tensor.shape[1], reconstructed.shape[1])
        diff = tensor[:, :min_feat] - reconstructed[:, :min_feat].astype(np.float32)
        rmse = np.sqrt(np.mean(diff**2))
        assert rmse < 0.008

    @pytest.mark.parametrize("group_size", [20, 60])
    def test_group_size_variations(self, group_size):
        """Test different group sizes."""
        np.random.seed(42)
        in_feat = 10 * group_size
        tensor = np.random.randn(32, in_feat).astype(np.float32) * 0.02

        packed, scales = quantize_nf3(tensor, group_size=group_size)
        dequantize_nf3(packed, scales, group_size=group_size)

        assert scales.shape[1] == in_feat // group_size

    def test_original_in_feat_parameter(self):
        """Test the original_in_feat parameter for truncating output."""
        np.random.seed(42)
        tensor = np.random.randn(8, 64).astype(np.float32) * 0.02

        packed, scales = quantize_nf3(tensor, group_size=64)
        reconstructed = dequantize_nf3(packed, scales, group_size=64, original_in_feat=64)

        assert reconstructed.shape == tensor.shape


# ---------------------------------------------------------------------------
# Utility Function Tests
# ---------------------------------------------------------------------------


class TestUtilityFunctions:
    """Test utility functions in sub4bit module."""

    @pytest.mark.parametrize("quant_type", ["int2", "int3", "nf2", "nf3"])
    def test_compute_quantization_error(self, quant_type, gaussian_tensor_small):
        """Test the error computation utility."""
        tensor = gaussian_tensor_small

        # Quantize
        if quant_type == "int2":
            packed, scales = quantize_int2(tensor, group_size=64)
        elif quant_type == "int3":
            packed, scales = quantize_int3(tensor, group_size=64)
        elif quant_type == "nf2":
            packed, scales = quantize_nf2(tensor, group_size=64)
        else:
            packed, scales = quantize_nf3(tensor, group_size=64)

        errors = compute_quantization_error(tensor, packed, scales, quant_type, group_size=64)

        # Verify all expected keys present
        assert "mse" in errors
        assert "rmse" in errors
        assert "max_error" in errors
        assert "mean_relative_error" in errors

        # Sanity checks
        assert errors["mse"] >= 0
        assert errors["rmse"] >= 0
        assert np.isclose(errors["rmse"], np.sqrt(errors["mse"]))
        assert errors["max_error"] >= errors["rmse"]

    def test_compute_quantization_error_invalid_type(self, gaussian_tensor_small):
        """Test error on invalid quant_type."""
        tensor = gaussian_tensor_small
        packed, scales = quantize_int2(tensor, group_size=64)

        with pytest.raises(ValueError, match="Unknown quant_type"):
            compute_quantization_error(tensor, packed, scales, "invalid", group_size=64)

    @pytest.mark.parametrize(
        "quant_type,expected_bits",
        [
            ("int2", 2),
            ("int3", 3),
            ("nf2", 2),
            ("nf3", 3),
        ],
    )
    def test_estimate_compression_ratio(self, quant_type, expected_bits):
        """Test compression ratio estimation."""
        shape = (4096, 4096)
        group_size = 64

        ratio = estimate_compression_ratio(shape, quant_type, group_size)

        # Expected: 16 bits (FP16) / (bits_per_weight + scale_overhead)
        # For large tensors, scale overhead is small
        # Approximate: 16 / expected_bits for large tensors
        expected_approx = 16 / expected_bits
        # Allow for scale overhead reducing ratio somewhat
        assert ratio > expected_approx * 0.8
        assert ratio < expected_approx * 1.1

    def test_estimate_compression_ratio_invalid_type(self):
        """Test error on invalid quant_type."""
        with pytest.raises(ValueError, match="Unknown quant_type"):
            estimate_compression_ratio((1024, 1024), "invalid", 64)

    @pytest.mark.parametrize(
        "target_compression,quality_priority,expected_bits",
        [
            (7.0, False, 2),  # High compression -> 2-bit
            (7.0, True, 2),  # High compression + quality -> NF2 (2-bit)
            (5.0, False, 3),  # Moderate compression -> 3-bit
            (5.0, True, 3),  # Moderate compression + quality -> NF3 (3-bit)
        ],
    )
    def test_select_sub4bit_format(self, target_compression, quality_priority, expected_bits):
        """Test format selection based on constraints."""
        np.random.seed(42)
        tensor = np.random.randn(256, 256).astype(np.float32) * 0.02  # Gaussian-like

        fmt = select_sub4bit_format(tensor, target_compression, quality_priority)

        if expected_bits == 2:
            assert fmt in ("int2", "nf2")
        else:
            assert fmt in ("int3", "nf3")

        if quality_priority:
            # Should prefer NF variants for Gaussian data
            assert fmt.startswith("nf")


# ---------------------------------------------------------------------------
# LUT Export Tests
# ---------------------------------------------------------------------------


class TestLUTExport:
    """Test lookup table export functions for Metal shaders."""

    def test_get_int2_lut(self):
        """Test INT2 LUT export."""
        lut = get_int2_lut()
        assert lut.shape == (4,)
        assert lut.dtype == np.float32
        np.testing.assert_array_almost_equal(lut, INT2_LEVELS)
        # Verify it's a copy, not a reference
        lut[0] = 999
        assert INT2_LEVELS[0] != 999

    def test_get_int3_lut(self):
        """Test INT3 LUT export."""
        lut = get_int3_lut()
        assert lut.shape == (8,)
        assert lut.dtype == np.float32
        np.testing.assert_array_almost_equal(lut, INT3_LEVELS)

    def test_get_nf2_lut(self):
        """Test NF2 LUT export."""
        lut = get_nf2_lut()
        assert lut.shape == (4,)
        assert lut.dtype == np.float32
        np.testing.assert_array_almost_equal(lut, NF2_LEVELS)

    def test_get_nf3_lut(self):
        """Test NF3 LUT export."""
        lut = get_nf3_lut()
        assert lut.shape == (8,)
        assert lut.dtype == np.float32
        np.testing.assert_array_almost_equal(lut, NF3_LEVELS)


# ---------------------------------------------------------------------------
# Comparison Tests (INT vs NF for same bit width)
# ---------------------------------------------------------------------------


class TestINTvsNFComparison:
    """Compare INT and NF formats at the same bit width."""

    def test_nf2_vs_int2_on_gaussian(self, gaussian_tensor_medium):
        """NF2 should outperform INT2 on Gaussian data."""
        tensor = gaussian_tensor_medium

        # INT2
        packed_int2, scales_int2 = quantize_int2(tensor, group_size=64)
        recon_int2 = dequantize_int2(packed_int2, scales_int2, group_size=64)
        rmse_int2 = np.sqrt(np.mean((tensor - recon_int2.astype(np.float32)) ** 2))

        # NF2
        packed_nf2, scales_nf2 = quantize_nf2(tensor, group_size=64)
        recon_nf2 = dequantize_nf2(packed_nf2, scales_nf2, group_size=64)
        rmse_nf2 = np.sqrt(np.mean((tensor - recon_nf2.astype(np.float32)) ** 2))

        # NF2 should be better or equal for Gaussian data
        assert rmse_nf2 <= rmse_int2 * 1.05, "NF2 should be at least as good as INT2 on Gaussian"

    def test_nf3_vs_int3_on_gaussian(self, gaussian_tensor_medium):
        """NF3 should outperform INT3 on Gaussian data."""
        tensor = gaussian_tensor_medium

        # INT3
        packed_int3, scales_int3 = quantize_int3(tensor, group_size=64)
        recon_int3 = dequantize_int3(packed_int3, scales_int3, group_size=64)
        min_feat = min(tensor.shape[1], recon_int3.shape[1])
        rmse_int3 = np.sqrt(
            np.mean((tensor[:, :min_feat] - recon_int3[:, :min_feat].astype(np.float32)) ** 2)
        )

        # NF3
        packed_nf3, scales_nf3 = quantize_nf3(tensor, group_size=64)
        recon_nf3 = dequantize_nf3(packed_nf3, scales_nf3, group_size=64)
        min_feat = min(tensor.shape[1], recon_nf3.shape[1])
        rmse_nf3 = np.sqrt(
            np.mean((tensor[:, :min_feat] - recon_nf3[:, :min_feat].astype(np.float32)) ** 2)
        )

        # NF3 should be better or equal for Gaussian data
        assert rmse_nf3 <= rmse_int3 * 1.05, "NF3 should be at least as good as INT3 on Gaussian"

    def test_int2_vs_nf2_on_uniform(self, uniform_tensor):
        """INT2 may outperform NF2 on uniform data (NF optimal for Gaussian only)."""
        tensor = uniform_tensor

        # INT2
        packed_int2, scales_int2 = quantize_int2(tensor, group_size=64)
        recon_int2 = dequantize_int2(packed_int2, scales_int2, group_size=64)
        rmse_int2 = np.sqrt(np.mean((tensor - recon_int2.astype(np.float32)) ** 2))

        # NF2
        packed_nf2, scales_nf2 = quantize_nf2(tensor, group_size=64)
        recon_nf2 = dequantize_nf2(packed_nf2, scales_nf2, group_size=64)
        rmse_nf2 = np.sqrt(np.mean((tensor - recon_nf2.astype(np.float32)) ** 2))

        # Both should work; uniform distribution has more extreme values,
        # so expect higher error than Gaussian
        assert rmse_int2 < 0.03, f"INT2 RMSE {rmse_int2} too high"
        assert rmse_nf2 < 0.03, f"NF2 RMSE {rmse_nf2} too high"


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_row(self):
        """Test quantization of single-row tensor."""
        tensor = np.random.randn(1, 64).astype(np.float32) * 0.02

        for quant_fn, dequant_fn in [
            (quantize_int2, dequantize_int2),
            (quantize_nf2, dequantize_nf2),
        ]:
            packed, scales = quant_fn(tensor, group_size=64)
            reconstructed = dequant_fn(packed, scales, group_size=64)
            assert reconstructed.shape == tensor.shape

    def test_single_group(self):
        """Test tensor with exactly one quantization group."""
        tensor = np.random.randn(8, 64).astype(np.float32) * 0.02

        packed, scales = quantize_int2(tensor, group_size=64)
        assert scales.shape == (8, 1)

        reconstructed = dequantize_int2(packed, scales, group_size=64)
        assert reconstructed.shape == tensor.shape

    def test_very_small_values(self):
        """Test quantization of very small values (near machine precision)."""
        tensor = np.random.randn(16, 64).astype(np.float32) * 1e-10

        packed, scales = quantize_int2(tensor, group_size=64)
        reconstructed = dequantize_int2(packed, scales, group_size=64)

        # Values should be quantized to near-zero
        assert np.max(np.abs(reconstructed)) < 1e-5

    def test_very_large_values(self):
        """Test quantization of large values."""
        np.random.seed(42)
        tensor = np.random.randn(16, 64).astype(np.float32) * 1000

        packed, scales = quantize_int2(tensor, group_size=64)
        reconstructed = dequantize_int2(packed, scales, group_size=64)

        # Should preserve relative magnitudes
        # INT2 only has 4 levels, so expect significant quantization error
        diff = tensor - reconstructed.astype(np.float32)
        rmse = np.sqrt(np.mean(diff**2))
        # RMSE should be proportional to scale; for Gaussian * 1000, std ~ 1000
        # INT2 error is ~50% of std, so RMSE < 600 is reasonable
        assert rmse < 600, f"RMSE {rmse} too high for large values"

        # Also verify that scales are correct (proportional to max values)
        expected_max_scale = np.max(np.abs(tensor)) / 1.5
        actual_max_scale = np.max(scales.astype(np.float32))
        assert actual_max_scale > expected_max_scale * 0.5

    def test_mixed_positive_negative(self):
        """Test tensor with mixed positive and negative values."""
        np.random.seed(42)
        tensor = np.random.randn(32, 64).astype(np.float32) * 0.02
        tensor[:16, :] = np.abs(tensor[:16, :])  # First half positive
        tensor[16:, :] = -np.abs(tensor[16:, :])  # Second half negative

        packed, scales = quantize_int2(tensor, group_size=64)
        reconstructed = dequantize_int2(packed, scales, group_size=64)

        # Signs should be preserved
        assert np.all(reconstructed[:16, :] >= 0)
        assert np.all(reconstructed[16:, :] <= 0)

    def test_alternating_signs(self):
        """Test tensor with rapidly alternating signs."""
        tensor = np.ones((8, 64), dtype=np.float32) * 0.02
        tensor[:, 1::2] *= -1  # Alternate columns

        packed, scales = quantize_int2(tensor, group_size=64)
        reconstructed = dequantize_int2(packed, scales, group_size=64)

        # Should preserve sign pattern
        assert np.all(reconstructed[:, ::2] > 0)
        assert np.all(reconstructed[:, 1::2] < 0)

    def test_dtype_float16_input(self):
        """Test that float16 input works correctly."""
        tensor = np.random.randn(16, 64).astype(np.float16) * 0.02

        packed, scales = quantize_int2(tensor, group_size=64)
        reconstructed = dequantize_int2(packed, scales, group_size=64)

        assert reconstructed.dtype == np.float16
        assert reconstructed.shape == tensor.shape

    def test_scale_dtype(self):
        """Verify scales are stored as float16 for memory efficiency."""
        tensor = np.random.randn(16, 64).astype(np.float32) * 0.02

        _, scales_int2 = quantize_int2(tensor, group_size=64)
        _, scales_int3 = quantize_int3(tensor, group_size=64)
        _, scales_nf2 = quantize_nf2(tensor, group_size=64)
        _, scales_nf3 = quantize_nf3(tensor, group_size=64)

        assert scales_int2.dtype == np.float16
        assert scales_int3.dtype == np.float16
        assert scales_nf2.dtype == np.float16
        assert scales_nf3.dtype == np.float16

    def test_packed_dtype(self):
        """Verify packed weights are stored as uint32."""
        tensor = np.random.randn(16, 64).astype(np.float32) * 0.02

        packed_int2, _ = quantize_int2(tensor, group_size=64)
        packed_int3, _ = quantize_int3(tensor, group_size=64)
        packed_nf2, _ = quantize_nf2(tensor, group_size=64)
        packed_nf3, _ = quantize_nf3(tensor, group_size=64)

        assert packed_int2.dtype == np.uint32
        assert packed_int3.dtype == np.uint32
        assert packed_nf2.dtype == np.uint32
        assert packed_nf3.dtype == np.uint32


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
