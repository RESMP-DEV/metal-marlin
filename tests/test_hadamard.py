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

    @pytest.mark.parametrize("block_size", [96, 160, 192])
    def test_roundtrip_non_power_of_2(self, block_size: int) -> None:
        """Rotation should work for non-power-of-2 block sizes via decomposition."""
        K, N = block_size * 4, 128
        W = np.random.randn(K, N).astype(np.float32)

        W_rot, meta = apply_hadamard_rotation(W, block_size=block_size)
        W_recovered = inverse_hadamard_rotation(W_rot, meta)

        np.testing.assert_allclose(W, W_recovered, rtol=1e-5, atol=1e-6)
        assert meta.block_size == block_size

    @pytest.mark.parametrize("block_size", [96, 160, 192])
    def test_roundtrip_non_power_of_2_needs_padding(self, block_size: int) -> None:
        """Non-power-of-2 rotation should handle padding correctly."""
        # K not divisible by block_size
        K, N = block_size * 3 + 7, 128
        W = np.random.randn(K, N).astype(np.float32)

        W_rot, meta = apply_hadamard_rotation(W, block_size=block_size)
        W_recovered = inverse_hadamard_rotation(W_rot, meta)

        np.testing.assert_allclose(W, W_recovered, rtol=1e-5, atol=1e-6)

    def test_invalid_block_size(self) -> None:
        """Should raise ValueError for unsupported block_size."""
        W = np.random.randn(128, 128).astype(np.float32)
        with pytest.raises(ValueError, match="block_size must be"):
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


def hadamard_transform_numpy(x: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Reference Hadamard transform using matrix multiplication.

    Args:
        x: Input array of shape [..., block_size].
        normalize: If True, normalize by 1/sqrt(block_size).

    Returns:
        Transformed array of same shape.
    """
    block_size = x.shape[-1]
    H = hadamard_matrix(block_size)
    if not normalize:
        # hadamard_matrix() returns normalized matrix, so unnormalize if needed
        H = H * np.sqrt(block_size)

    # Reshape to 2D, apply transform, reshape back
    orig_shape = x.shape
    x_2d = x.reshape(-1, block_size)
    result = x_2d @ H.T  # H @ x for each row
    return result.reshape(orig_shape)


# Check if MPS is available
try:
    import torch

    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_MPS = False

pytestmark = pytest.mark.skipif(not HAS_MPS, reason="MPS not available")


class TestHadamardMetalKernel:
    """Tests for hadamard_transform Metal kernel."""

    def test_import(self):
        """Test that hadamard_transform can be imported."""
        from metal_marlin.kernels import hadamard_transform

        assert callable(hadamard_transform)

    def test_block_size_32(self):
        """Test Hadamard transform with block_size=32."""
        from metal_marlin.kernels import hadamard_transform

        batch = 16
        x = torch.randn(batch, 32, device="mps")

        result = hadamard_transform(x, block_size=32, normalize=True)

        assert result.shape == x.shape
        assert result.dtype == torch.float16

    def test_block_size_64(self):
        """Test Hadamard transform with block_size=64."""
        from metal_marlin.kernels import hadamard_transform

        batch = 16
        x = torch.randn(batch, 64, device="mps")

        result = hadamard_transform(x, block_size=64, normalize=True)

        assert result.shape == x.shape

    def test_block_size_128(self):
        """Test Hadamard transform with block_size=128."""
        from metal_marlin.kernels import hadamard_transform

        batch = 16
        x = torch.randn(batch, 128, device="mps")

        result = hadamard_transform(x, block_size=128, normalize=True)

        assert result.shape == x.shape

    def test_accuracy_block_32(self):
        """Test accuracy of block_size=32 against numpy reference."""
        from metal_marlin.kernels import hadamard_transform

        torch.manual_seed(42)
        np.random.seed(42)
        batch = 8
        x_np = np.random.randn(batch, 32).astype(np.float32)

        # Reference
        expected = hadamard_transform_numpy(x_np, normalize=True)

        # Metal kernel
        x_torch = torch.tensor(x_np, device="mps", dtype=torch.float16)
        result = hadamard_transform(x_torch, block_size=32, normalize=True)
        result_np = result.cpu().float().numpy()

        # FP16 has limited precision
        np.testing.assert_allclose(result_np, expected, rtol=1e-2, atol=1e-2)

    def test_accuracy_block_64(self):
        """Test accuracy of block_size=64 against numpy reference."""
        from metal_marlin.kernels import hadamard_transform

        torch.manual_seed(42)
        np.random.seed(42)
        batch = 8
        x_np = np.random.randn(batch, 64).astype(np.float32)

        # Reference
        expected = hadamard_transform_numpy(x_np, normalize=True)

        # Metal kernel
        x_torch = torch.tensor(x_np, device="mps", dtype=torch.float16)
        result = hadamard_transform(x_torch, block_size=64, normalize=True)
        result_np = result.cpu().float().numpy()

        np.testing.assert_allclose(result_np, expected, rtol=1e-2, atol=1e-2)

    def test_accuracy_block_128(self):
        """Test accuracy of block_size=128 against numpy reference."""
        from metal_marlin.kernels import hadamard_transform

        torch.manual_seed(42)
        np.random.seed(42)
        batch = 8
        x_np = np.random.randn(batch, 128).astype(np.float32)

        # Reference
        expected = hadamard_transform_numpy(x_np, normalize=True)

        # Metal kernel
        x_torch = torch.tensor(x_np, device="mps", dtype=torch.float16)
        result = hadamard_transform(x_torch, block_size=128, normalize=True)
        result_np = result.cpu().float().numpy()

        np.testing.assert_allclose(result_np, expected, rtol=2e-2, atol=2e-2)

    def test_orthogonality(self):
        """Test that H @ H @ x = x (with normalized transforms)."""
        from metal_marlin.kernels import hadamard_transform

        torch.manual_seed(42)
        batch = 8
        x = torch.randn(batch, 64, device="mps")

        # Apply normalized Hadamard twice: H_norm @ H_norm @ x = x
        # because H_norm @ H_norm = (H/sqrt(n)) @ (H/sqrt(n)) = H@H / n = I
        result1 = hadamard_transform(x, block_size=64, normalize=True)
        result2 = hadamard_transform(result1, block_size=64, normalize=True)

        x_np = x.cpu().float().numpy()
        result_np = result2.cpu().float().numpy()

        np.testing.assert_allclose(result_np, x_np, rtol=5e-2, atol=5e-2)

    def test_unnormalized(self):
        """Test unnormalized Hadamard transform."""
        from metal_marlin.kernels import hadamard_transform

        torch.manual_seed(42)
        batch = 4
        x = torch.randn(batch, 64, device="mps")

        # Unnormalized
        result_unnorm = hadamard_transform(x, block_size=64, normalize=False)

        # Normalized
        result_norm = hadamard_transform(x, block_size=64, normalize=True)

        # Unnormalized = normalized * sqrt(block_size)
        scale = np.sqrt(64)
        unnorm_np = result_unnorm.cpu().float().numpy()
        norm_np = result_norm.cpu().float().numpy()

        np.testing.assert_allclose(unnorm_np, norm_np * scale, rtol=1e-2, atol=1e-2)

    def test_invalid_block_size(self):
        """Test that invalid block sizes raise errors."""
        from metal_marlin.kernels import hadamard_transform

        x = torch.randn(4, 64, device="mps")

        with pytest.raises(ValueError, match="block_size must be"):
            hadamard_transform(x, block_size=48)

        with pytest.raises(ValueError, match="block_size must be"):
            hadamard_transform(x, block_size=16)

    def test_mismatched_dimension(self):
        """Test that mismatched dimensions raise errors."""
        from metal_marlin.kernels import hadamard_transform

        x = torch.randn(4, 100, device="mps")

        with pytest.raises(ValueError, match="must equal block_size"):
            hadamard_transform(x, block_size=64)

    def test_3d_input(self):
        """Test with 3D input tensor."""
        from metal_marlin.kernels import hadamard_transform

        x = torch.randn(2, 8, 64, device="mps")

        result = hadamard_transform(x, block_size=64, normalize=True)

        assert result.shape == x.shape

    def test_single_vector(self):
        """Test with a single vector."""
        from metal_marlin.kernels import hadamard_transform

        x = torch.randn(1, 64, device="mps")

        result = hadamard_transform(x, block_size=64, normalize=True)

        assert result.shape == x.shape

    def test_large_batch(self):
        """Test with large batch size."""
        from metal_marlin.kernels import hadamard_transform

        batch = 1024
        x = torch.randn(batch, 64, device="mps")

        result = hadamard_transform(x, block_size=64, normalize=True)

        assert result.shape == (batch, 64)

    def test_dtype_float16_input(self):
        """Test that float16 input is handled correctly."""
        from metal_marlin.kernels import hadamard_transform

        x = torch.randn(4, 64, device="mps", dtype=torch.float16)

        result = hadamard_transform(x, block_size=64)

        assert result.dtype == torch.float16

    def test_dtype_float32_input(self):
        """Test that float32 input is converted to float16 for kernel."""
        from metal_marlin.kernels import hadamard_transform

        x = torch.randn(4, 64, device="mps", dtype=torch.float32)

        result = hadamard_transform(x, block_size=64)

        # Kernel internally converts to half
        assert result.dtype == torch.float16

    def test_energy_preservation(self):
        """Test that total energy (Frobenius norm) is preserved."""
        from metal_marlin.kernels import hadamard_transform

        torch.manual_seed(42)
        batch = 16
        x = torch.randn(batch, 64, device="mps")

        # Normalized transform preserves energy
        result = hadamard_transform(x, block_size=64, normalize=True)

        x_np = x.cpu().float().numpy()
        result_np = result.cpu().float().numpy()

        energy_before = np.sum(x_np**2)
        energy_after = np.sum(result_np**2)

        # For normalized transform, energy is preserved
        np.testing.assert_allclose(energy_before, energy_after, rtol=5e-2)
