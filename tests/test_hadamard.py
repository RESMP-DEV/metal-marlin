"""Tests for Hadamard transform outlier dispersal.

Validates:
1. Orthonormality: H @ H.T = I
2. Self-inverse: H @ H = I (for normalized Hadamard)
3. Numerical precision of roundtrip (apply -> inverse)
4. Outlier dispersal effectiveness: max/mean ratio reduction
5. Edge cases: padding, axis selection, block sizes
"""

from __future__ import annotations

import numpy as np
import pytest

from metal_marlin.hadamard import (
    apply_hadamard_rotation,
    compute_outlier_stats,
    hadamard_matrix,
    inverse_hadamard_rotation,
)


class TestHadamardMatrix:
    """Tests for hadamard_matrix construction."""

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 128])
    @pytest.mark.smoke
    def test_orthonormality(self, n: int) -> None:
        """H @ H.T should equal identity matrix."""
        H = hadamard_matrix(n)
        result = H @ H.T
        expected = np.eye(n, dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 128])
    @pytest.mark.smoke
    def test_self_inverse(self, n: int) -> None:
        """Normalized Hadamard should be self-inverse: H @ H = I."""
        H = hadamard_matrix(n)
        # For normalized Hadamard, H^T = H (symmetric) so H @ H = H @ H^T = I
        result = H @ H
        expected = np.eye(n, dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64])
    def test_symmetric(self, n: int) -> None:
        """Normalized Hadamard should be symmetric: H = H.T."""
        H = hadamard_matrix(n)
        np.testing.assert_allclose(H, H.T, rtol=1e-6)

    @pytest.mark.parametrize("n", [2, 4, 8, 16])
    def test_entries_magnitude(self, n: int) -> None:
        """All entries should have magnitude 1/sqrt(n)."""
        H = hadamard_matrix(n)
        expected_mag = 1.0 / np.sqrt(n)
        np.testing.assert_allclose(np.abs(H), expected_mag, rtol=1e-6)

    def test_invalid_not_power_of_two(self) -> None:
        """Should raise ValueError for non-power-of-2 input."""
        with pytest.raises(ValueError, match="power of 2"):
            hadamard_matrix(3)
        with pytest.raises(ValueError, match="power of 2"):
            hadamard_matrix(6)

    def test_invalid_zero_or_negative(self) -> None:
        """Should raise ValueError for zero or negative input."""
        with pytest.raises(ValueError, match="positive"):
            hadamard_matrix(0)
        with pytest.raises(ValueError, match="positive"):
            hadamard_matrix(-4)


class TestApplyHadamardRotation:
    """Tests for apply_hadamard_rotation."""

    @pytest.mark.parametrize("block_size", [8, 16, 32, 64])
    @pytest.mark.smoke
    def test_roundtrip_exact_divisible(self, block_size: int) -> None:
        """Rotation should be perfectly reversible for aligned dimensions."""
        K, N = block_size * 4, 128
        W = np.random.randn(K, N).astype(np.float32)

        W_rot, meta = apply_hadamard_rotation(W, block_size=block_size)
        W_recovered = inverse_hadamard_rotation(W_rot, meta)

        np.testing.assert_allclose(W, W_recovered, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("block_size", [8, 16, 32, 64])
    def test_roundtrip_needs_padding(self, block_size: int) -> None:
        """Rotation should handle non-aligned dimensions with padding."""
        # K not divisible by block_size
        K, N = block_size * 3 + 7, 128
        W = np.random.randn(K, N).astype(np.float32)

        W_rot, meta = apply_hadamard_rotation(W, block_size=block_size)
        W_recovered = inverse_hadamard_rotation(W_rot, meta)

        np.testing.assert_allclose(W, W_recovered, rtol=1e-5, atol=1e-6)

    def test_roundtrip_axis_1(self) -> None:
        """Rotation along N axis should also be reversible."""
        K, N = 256, 256
        W = np.random.randn(K, N).astype(np.float32)

        W_rot, meta = apply_hadamard_rotation(W, block_size=64, axis=1)
        W_recovered = inverse_hadamard_rotation(W_rot, meta)

        np.testing.assert_allclose(W, W_recovered, rtol=1e-5, atol=1e-6)

    def test_metadata_correct(self) -> None:
        """Metadata should correctly capture dimensions and parameters."""
        K, N = 200, 512
        block_size = 64

        _, meta = apply_hadamard_rotation(np.zeros((K, N)), block_size=block_size)

        assert meta.block_size == block_size
        assert meta.orig_k == K
        assert meta.padded_k == 256  # Next multiple of 64
        assert meta.axis == 0

    def test_invalid_block_size(self) -> None:
        """Should raise ValueError for non-power-of-2 block_size."""
        W = np.random.randn(128, 128).astype(np.float32)
        with pytest.raises(ValueError, match="power of 2"):
            apply_hadamard_rotation(W, block_size=100)

    def test_invalid_axis(self) -> None:
        """Should raise ValueError for invalid axis."""
        W = np.random.randn(128, 128).astype(np.float32)
        with pytest.raises(ValueError, match="axis must be 0 or 1"):
            apply_hadamard_rotation(W, axis=2)


class TestOutlierDispersal:
    """Tests for outlier dispersal effectiveness."""

    @pytest.mark.smoke
    def test_single_outlier_dispersed(self) -> None:
        """A single large outlier should be dispersed across the block."""
        K, N = 256, 512
        W = np.random.randn(K, N).astype(np.float32) * 0.1

        # Insert a large outlier
        outlier_value = 100.0
        W[0, 0] = outlier_value

        W_rot, _ = apply_hadamard_rotation(W, block_size=64)

        # The max value should be significantly reduced
        # Hadamard spreads the outlier across 64 elements
        assert W_rot.max() < outlier_value / 2
        # Energy is conserved (due to orthonormality)
        np.testing.assert_allclose(np.sum(W**2), np.sum(W_rot**2), rtol=1e-4)

    @pytest.mark.smoke
    def test_max_mean_ratio_reduced(self) -> None:
        """Max/mean ratio should decrease after rotation."""
        K, N = 256, 512
        W = np.random.randn(K, N).astype(np.float32)

        # Add multiple outliers
        np.random.seed(42)
        outlier_indices = np.random.choice(K * N, size=10, replace=False)
        W.ravel()[outlier_indices] = np.random.uniform(10, 50, size=10)

        stats_before = compute_outlier_stats(W)
        W_rot, _ = apply_hadamard_rotation(W, block_size=64)
        stats_after = compute_outlier_stats(W_rot)

        # Max/mean ratio should decrease (outliers dispersed)
        assert stats_after["max_mean_ratio"] < stats_before["max_mean_ratio"]

    def test_energy_preservation(self) -> None:
        """Total energy (sum of squares) should be preserved."""
        K, N = 256, 512
        W = np.random.randn(K, N).astype(np.float32) * 10

        W_rot, _ = apply_hadamard_rotation(W, block_size=64)

        energy_before = np.sum(W**2)
        energy_after = np.sum(W_rot**2)
        np.testing.assert_allclose(energy_before, energy_after, rtol=1e-4)

    def test_channel_wise_variance_more_uniform(self) -> None:
        """Per-channel variance should become more uniform after rotation."""
        K, N = 256, 512
        W = np.random.randn(K, N).astype(np.float32)

        # Make some channels have much higher variance
        W[:, 0:64] *= 10.0

        channel_var_before = np.var(W, axis=0)
        W_rot, _ = apply_hadamard_rotation(W, block_size=64, axis=1)
        channel_var_after = np.var(W_rot, axis=0)

        # Coefficient of variation of channel variances should decrease
        cv_before = np.std(channel_var_before) / np.mean(channel_var_before)
        cv_after = np.std(channel_var_after) / np.mean(channel_var_after)
        assert cv_after < cv_before


class TestComputeOutlierStats:
    """Tests for compute_outlier_stats helper."""

    def test_basic_stats(self) -> None:
        """Should compute correct basic statistics."""
        W = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        stats = compute_outlier_stats(W)

        assert stats["max_abs"] == 4.0
        np.testing.assert_allclose(stats["mean_abs"], 2.5)
        assert stats["max_mean_ratio"] == 4.0 / 2.5

    def test_kurtosis_normal(self) -> None:
        """Normal distribution should have near-zero excess kurtosis."""
        np.random.seed(42)
        W = np.random.randn(10000).astype(np.float32)
        stats = compute_outlier_stats(W)

        # Excess kurtosis of normal is 0, allow some sampling variance
        assert abs(stats["kurtosis"]) < 0.5

    def test_kurtosis_outliers(self) -> None:
        """Distribution with outliers should have positive excess kurtosis."""
        np.random.seed(42)
        W = np.random.randn(10000).astype(np.float32)
        # Add some extreme outliers
        W[:10] = 100.0
        W[10:20] = -100.0
        stats = compute_outlier_stats(W)

        # Should have positive excess kurtosis (heavy tails)
        assert stats["kurtosis"] > 1.0


class TestLLMScaleDimensions:
    """Tests with LLM-scale weight dimensions."""

    @pytest.mark.parametrize(
        "K,N",
        [
            (4096, 4096),  # Typical hidden dim
            (4096, 11008),  # Llama MLP up-projection
            (11008, 4096),  # Llama MLP down-projection
            (4096, 32000),  # Vocabulary projection
        ],
    )
    def test_roundtrip_llm_dims(self, K: int, N: int) -> None:
        """Rotation should work correctly for LLM-scale dimensions."""
        np.random.seed(42)
        W = np.random.randn(K, N).astype(np.float32) * 0.02  # Xavier-like init

        W_rot, meta = apply_hadamard_rotation(W, block_size=128)
        W_recovered = inverse_hadamard_rotation(W_rot, meta)

        np.testing.assert_allclose(W, W_recovered, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("block_size", [64, 128])
    def test_quantization_group_aligned(self, block_size: int) -> None:
        """Block size should align with typical quantization group sizes."""
        K, N = 4096, 4096
        np.random.seed(42)
        W = np.random.randn(K, N).astype(np.float32) * 0.02

        W_rot, meta = apply_hadamard_rotation(W, block_size=block_size)

        # Padded K should be divisible by block_size
        assert meta.padded_k % block_size == 0
        # Output shape should be correct
        assert W_rot.shape == (meta.padded_k, N)
