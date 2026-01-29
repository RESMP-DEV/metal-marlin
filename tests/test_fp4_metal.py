"""
Unit tests for FP4Metal - Metal-accelerated FP4 E2M1 quantization.

Tests verify:
1. Correct quantization to E2M1 indices
2. Dequantization accuracy
3. Round-trip error bounds
4. Packing correctness
"""

from __future__ import annotations

import numpy as np
import pytest

from metal_marlin.quantize_fp4 import E2M1_VALUES
from metal_marlin.quantize_fp4 import dequantize_fp4 as dequantize_fp4_cpu
from metal_marlin.quantize_fp4 import quantize_to_fp4 as quantize_to_fp4_cpu

# Optional imports - tests skip if Metal implementation not available
try:
    from metal_marlin.fp4_metal import FP4Metal

    HAS_FP4_METAL = True
except ImportError:
    HAS_FP4_METAL = False
    FP4Metal = None  # type: ignore[misc,assignment]

pytestmark = [
    pytest.mark.skipif(not HAS_FP4_METAL, reason="FP4Metal not available"),
]


class TestFP4MetalInit:
    """Test FP4Metal initialization."""

    def test_init_success(self):
        """FP4Metal should initialize without error."""
        fp4 = FP4Metal()
        assert fp4 is not None

    def test_e2m1_table(self):
        """E2M1 lookup table should have correct values."""
        expected = np.array(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(E2M1_VALUES, expected)


class TestQuantization:
    """Test FP4 quantization."""

    @pytest.fixture
    def fp4(self):
        return FP4Metal()

    def test_quantize_shape(self, fp4):
        """Quantization should preserve shape."""
        values = np.random.randn(100).astype(np.float32)
        indices, scales = fp4.quantize(values, group_size=100)
        assert indices.shape == values.shape

    def test_quantize_range(self, fp4):
        """Indices should be in [0, 15]."""
        values = np.random.randn(1000).astype(np.float32) * 5.0
        indices, _ = fp4.quantize(values, group_size=128)
        assert indices.min() >= 0
        assert indices.max() <= 15

    def test_quantize_exact_values(self, fp4):
        """Exact E2M1 values should quantize correctly."""
        # Test values that are exact E2M1 representable
        values = np.array(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32
        )
        indices, scales = fp4.quantize(values, group_size=8)
        reconstructed = fp4.dequantize(indices, scales, group_size=8)

        # Should reconstruct exactly (within floating point)
        np.testing.assert_array_almost_equal(values, reconstructed, decimal=5)

    def test_quantize_negative(self, fp4):
        """Negative values should use indices 8-15."""
        values = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        indices, _ = fp4.quantize(values, group_size=3)
        # Negative indices should have bit 3 set (8-15)
        assert all(idx >= 8 for idx in indices.flatten())


class TestDequantization:
    """Test FP4 dequantization."""

    @pytest.fixture
    def fp4(self):
        return FP4Metal()

    def test_dequantize_table_lookup(self, fp4):
        """Dequantization should match E2M1 table."""
        indices = np.arange(16, dtype=np.uint8)
        scales = np.ones(1, dtype=np.float32)

        reconstructed = fp4.dequantize(indices, scales, group_size=16)

        # Should match E2M1_VALUES exactly
        np.testing.assert_array_almost_equal(reconstructed, E2M1_VALUES, decimal=5)


class TestRoundTrip:
    """Test quantize-dequantize round trips."""

    @pytest.fixture
    def fp4(self):
        return FP4Metal()

    def test_roundtrip_error_bound(self, fp4):
        """Round-trip error should be bounded."""
        values = np.random.randn(1024).astype(np.float32) * 3.0

        indices, scales = fp4.quantize(values, group_size=128)
        reconstructed = fp4.dequantize(indices, scales, group_size=128)

        # FP4 has limited precision, but error should be reasonable
        mse = np.mean((values - reconstructed) ** 2)
        max_error = np.max(np.abs(values - reconstructed))

        # MSE should be small relative to signal variance
        signal_var = np.var(values)
        snr = signal_var / (mse + 1e-10)

        assert snr > 1.0, f"SNR too low: {snr}"
        print(f"Round-trip MSE: {mse:.4f}, Max error: {max_error:.4f}, SNR: {snr:.2f}")

    def test_roundtrip_consistency(self, fp4):
        """Metal round-trip should be internally consistent."""
        values = np.random.randn(256).astype(np.float32)

        # Metal round-trip
        indices_metal, scales_metal = fp4.quantize(values, group_size=256)
        reconstructed_metal = fp4.dequantize(indices_metal, scales_metal, group_size=256)

        # Re-quantizing the reconstructed values should give similar results
        indices2, scales2 = fp4.quantize(reconstructed_metal, group_size=256)

        # Second quantization should have low error since input is already E2M1-representable
        assert indices2.shape == indices_metal.shape
        assert scales2.shape == scales_metal.shape

        # Most indices should match (allowing for small numerical differences)
        matching_indices = np.sum(indices2 == indices_metal)
        match_ratio = matching_indices / len(indices_metal.flatten())
        assert match_ratio > 0.95, f"Index mismatch ratio too high: {1 - match_ratio:.2%}"


class TestPacking:
    """Test 4-bit packing."""

    @pytest.fixture
    def fp4(self):
        return FP4Metal()

    def test_pack_pair_values(self, fp4):
        """Pack should combine lo and hi correctly."""
        lo = np.array([0, 1, 15, 5], dtype=np.uint8)
        hi = np.array([0, 2, 3, 10], dtype=np.uint8)

        packed = fp4.pack_pair(lo, hi)

        # Verify: packed = lo | (hi << 4)
        expected = lo | (hi << 4)
        np.testing.assert_array_equal(packed, expected)

    def test_pack_pair_unpack(self, fp4):
        """Should be able to unpack packed values."""
        lo = np.array([3, 7, 11, 15], dtype=np.uint8)
        hi = np.array([1, 2, 4, 8], dtype=np.uint8)

        packed = fp4.pack_pair(lo, hi)

        # Unpack
        unpacked_lo = packed & 0x0F
        unpacked_hi = (packed >> 4) & 0x0F

        np.testing.assert_array_equal(unpacked_lo, lo)
        np.testing.assert_array_equal(unpacked_hi, hi)


class TestTorchIntegration:
    """Test PyTorch tensor integration."""

    @pytest.fixture
    def fp4(self):
        return FP4Metal()

    @pytest.fixture
    def torch_available(self):
        """Skip if torch not available."""
        try:
            import torch

            return torch
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_torch_tensor_input(self, fp4, torch_available):
        """Should accept PyTorch tensors."""
        torch = torch_available
        values = torch.randn(256, device="cpu")
        indices, scales = fp4.quantize(values, group_size=128)

        # Should return arrays/tensors
        assert isinstance(indices, (np.ndarray, torch.Tensor))

    @pytest.mark.skipif(
        not HAS_FP4_METAL,
        reason="MPS not available",
    )
    def test_torch_mps_tensor(self, fp4, torch_available):
        """Should handle MPS tensors if available."""
        torch = torch_available
        if not torch.backends.mps.is_available():
            pytest.skip("MPS backend not available")

        values = torch.randn(256, device="mps")
        indices, scales = fp4.quantize(values, group_size=128)
        reconstructed = fp4.dequantize(indices, scales, group_size=128)

        assert reconstructed.shape == values.shape


class TestGroupScaling:
    """Test per-group scaling behavior."""

    @pytest.fixture
    def fp4(self):
        return FP4Metal()

    def test_multiple_groups(self, fp4):
        """Quantization should handle multiple groups."""
        values = np.random.randn(512).astype(np.float32)
        group_size = 128
        n_groups = 512 // group_size

        indices, scales = fp4.quantize(values, group_size=group_size)

        # Should have one scale per group
        assert scales.shape[0] == n_groups

    def test_group_independence(self, fp4):
        """Different groups should have independent scales."""
        # Create values with different magnitudes in different groups
        values = np.concatenate(
            [
                np.ones(128) * 0.1,  # Small values
                np.ones(128) * 5.0,  # Large values
            ]
        ).astype(np.float32)

        indices, scales = fp4.quantize(values, group_size=128)

        # Scales should be different for the two groups
        assert scales[0] != scales[1]


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def fp4(self):
        return FP4Metal()

    def test_zeros(self, fp4):
        """All zeros should quantize correctly."""
        values = np.zeros(128, dtype=np.float32)
        indices, scales = fp4.quantize(values, group_size=128)

        # Should quantize without error
        assert indices.shape == values.shape

    def test_extreme_values(self, fp4):
        """Very large and very small values should be handled."""
        values = np.array([1e-6, 1e6, -1e-6, -1e6], dtype=np.float32)
        indices, scales = fp4.quantize(values, group_size=4)

        # Should quantize without error (may clip)
        assert indices.shape == values.shape
        assert np.all(indices >= 0) and np.all(indices <= 15)

    def test_single_element_groups(self, fp4):
        """Group size equal to array length."""
        values = np.random.randn(64).astype(np.float32)
        indices, scales = fp4.quantize(values, group_size=64)

        assert indices.shape == values.shape
        assert scales.shape[0] == 1
