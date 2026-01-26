"""Tests for Walsh-Hadamard transform Metal kernel.

Tests the GPU-accelerated hadamard_transform function from kernels.py,
which uses simd_shuffle for register-only butterfly computation.
"""

from __future__ import annotations

import numpy as np
import pytest

# Check if MLX is available
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


def hadamard_matrix_numpy(n: int) -> np.ndarray:
    """Generate Walsh-Hadamard matrix of size n x n using Sylvester construction.

    Args:
        n: Size of the matrix. Must be a power of 2.

    Returns:
        Unnormalized Hadamard matrix (entries are +1 or -1).
    """
    if n == 1:
        return np.array([[1.0]])

    h_half = hadamard_matrix_numpy(n // 2)
    h = np.block([[h_half, h_half], [h_half, -h_half]])
    return h


def hadamard_transform_numpy(x: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Reference Hadamard transform using matrix multiplication.

    Args:
        x: Input array of shape [..., block_size].
        normalize: If True, normalize by 1/sqrt(block_size).

    Returns:
        Transformed array of same shape.
    """
    block_size = x.shape[-1]
    H = hadamard_matrix_numpy(block_size)
    if normalize:
        H = H / np.sqrt(block_size)

    # Reshape to 2D, apply transform, reshape back
    orig_shape = x.shape
    x_2d = x.reshape(-1, block_size)
    result = x_2d @ H.T  # H @ x for each row
    return result.reshape(orig_shape)


class TestHadamardTransformKernel:
    """Tests for hadamard_transform Metal kernel."""

    def test_import(self):
        """Test that hadamard_transform can be imported."""
        from metal_marlin.kernels import hadamard_transform

        assert callable(hadamard_transform)

    def test_block_size_32(self):
        """Test Hadamard transform with block_size=32."""
        from metal_marlin.kernels import hadamard_transform

        batch = 16
        x = mx.random.normal((batch, 32))
        mx.eval(x)

        result = hadamard_transform(x, block_size=32, normalize=True)
        mx.eval(result)

        assert result.shape == x.shape
        assert result.dtype in (mx.bfloat16, mx.float16)

    def test_block_size_64(self):
        """Test Hadamard transform with block_size=64."""
        from metal_marlin.kernels import hadamard_transform

        batch = 16
        x = mx.random.normal((batch, 64))
        mx.eval(x)

        result = hadamard_transform(x, block_size=64, normalize=True)
        mx.eval(result)

        assert result.shape == x.shape

    def test_block_size_128(self):
        """Test Hadamard transform with block_size=128."""
        from metal_marlin.kernels import hadamard_transform

        batch = 16
        x = mx.random.normal((batch, 128))
        mx.eval(x)

        result = hadamard_transform(x, block_size=128, normalize=True)
        mx.eval(result)

        assert result.shape == x.shape

    def test_accuracy_block_32(self):
        """Test accuracy of block_size=32 against numpy reference."""
        from metal_marlin.kernels import hadamard_transform

        np.random.seed(42)
        batch = 8
        x_np = np.random.randn(batch, 32).astype(np.float32)

        # Reference
        expected = hadamard_transform_numpy(x_np, normalize=True)

        # Metal kernel
        x_mx = mx.array(x_np, dtype=mx.float16)
        result = hadamard_transform(x_mx, block_size=32, normalize=True, dtype=mx.float16)
        mx.eval(result)
        result_np = np.array(result).astype(np.float32)

        # FP16 has limited precision
        np.testing.assert_allclose(result_np, expected, rtol=1e-2, atol=1e-2)

    def test_accuracy_block_64(self):
        """Test accuracy of block_size=64 against numpy reference."""
        from metal_marlin.kernels import hadamard_transform

        np.random.seed(42)
        batch = 8
        x_np = np.random.randn(batch, 64).astype(np.float32)

        # Reference
        expected = hadamard_transform_numpy(x_np, normalize=True)

        # Metal kernel
        x_mx = mx.array(x_np, dtype=mx.float16)
        result = hadamard_transform(x_mx, block_size=64, normalize=True, dtype=mx.float16)
        mx.eval(result)
        result_np = np.array(result).astype(np.float32)

        np.testing.assert_allclose(result_np, expected, rtol=1e-2, atol=1e-2)

    def test_accuracy_block_128(self):
        """Test accuracy of block_size=128 against numpy reference."""
        from metal_marlin.kernels import hadamard_transform

        np.random.seed(42)
        batch = 8
        x_np = np.random.randn(batch, 128).astype(np.float32)

        # Reference
        expected = hadamard_transform_numpy(x_np, normalize=True)

        # Metal kernel
        x_mx = mx.array(x_np, dtype=mx.float16)
        result = hadamard_transform(x_mx, block_size=128, normalize=True, dtype=mx.float16)
        mx.eval(result)
        result_np = np.array(result).astype(np.float32)

        np.testing.assert_allclose(result_np, expected, rtol=2e-2, atol=2e-2)

    def test_orthogonality(self):
        """Test that H @ H @ x = x (with normalized transforms)."""
        from metal_marlin.kernels import hadamard_transform

        batch = 8
        x = mx.random.normal((batch, 64))
        mx.eval(x)

        # Apply normalized Hadamard twice: H_norm @ H_norm @ x = x
        # because H_norm @ H_norm = (H/sqrt(n)) @ (H/sqrt(n)) = H@H / n = I
        result1 = hadamard_transform(x, block_size=64, normalize=True, dtype=mx.float16)
        mx.eval(result1)

        result2 = hadamard_transform(result1, block_size=64, normalize=True, dtype=mx.float16)
        mx.eval(result2)

        x_np = np.array(x.astype(mx.float32))
        result_np = np.array(result2.astype(mx.float32))

        np.testing.assert_allclose(result_np, x_np, rtol=5e-2, atol=5e-2)

    def test_unnormalized(self):
        """Test unnormalized Hadamard transform."""
        from metal_marlin.kernels import hadamard_transform

        batch = 4
        x = mx.random.normal((batch, 64))
        mx.eval(x)

        # Unnormalized
        result_unnorm = hadamard_transform(x, block_size=64, normalize=False, dtype=mx.float16)
        mx.eval(result_unnorm)

        # Normalized
        result_norm = hadamard_transform(x, block_size=64, normalize=True, dtype=mx.float16)
        mx.eval(result_norm)

        # Unnormalized = normalized * sqrt(block_size)
        scale = np.sqrt(64)
        unnorm_np = np.array(result_unnorm.astype(mx.float32))
        norm_np = np.array(result_norm.astype(mx.float32))

        np.testing.assert_allclose(unnorm_np, norm_np * scale, rtol=1e-2, atol=1e-2)

    def test_invalid_block_size(self):
        """Test that invalid block sizes raise errors."""
        from metal_marlin.kernels import hadamard_transform

        x = mx.random.normal((4, 64))
        mx.eval(x)

        with pytest.raises(ValueError, match="block_size must be"):
            hadamard_transform(x, block_size=48)

        with pytest.raises(ValueError, match="block_size must be"):
            hadamard_transform(x, block_size=16)

    def test_mismatched_dimension(self):
        """Test that mismatched dimensions raise errors."""
        from metal_marlin.kernels import hadamard_transform

        x = mx.random.normal((4, 100))
        mx.eval(x)

        with pytest.raises(ValueError, match="must equal block_size"):
            hadamard_transform(x, block_size=64)

    def test_3d_input(self):
        """Test with 3D input tensor."""
        from metal_marlin.kernels import hadamard_transform

        x = mx.random.normal((2, 8, 64))
        mx.eval(x)

        result = hadamard_transform(x, block_size=64, normalize=True)
        mx.eval(result)

        assert result.shape == x.shape

    def test_single_vector(self):
        """Test with a single vector."""
        from metal_marlin.kernels import hadamard_transform

        x = mx.random.normal((1, 64))
        mx.eval(x)

        result = hadamard_transform(x, block_size=64, normalize=True)
        mx.eval(result)

        assert result.shape == x.shape

    def test_large_batch(self):
        """Test with large batch size."""
        from metal_marlin.kernels import hadamard_transform

        batch = 1024
        x = mx.random.normal((batch, 64))
        mx.eval(x)

        result = hadamard_transform(x, block_size=64, normalize=True)
        mx.eval(result)

        assert result.shape == (batch, 64)

    def test_dtype_float16(self):
        """Test that float16 dtype is handled correctly."""
        from metal_marlin.kernels import hadamard_transform

        x = mx.random.normal((4, 64))
        mx.eval(x)

        result = hadamard_transform(x, block_size=64, dtype=mx.float16)
        mx.eval(result)

        assert result.dtype == mx.float16

    def test_dtype_bfloat16(self):
        """Test that bfloat16 dtype is handled correctly."""
        from metal_marlin.kernels import hadamard_transform

        x = mx.random.normal((4, 64))
        mx.eval(x)

        result = hadamard_transform(x, block_size=64, dtype=mx.bfloat16)
        mx.eval(result)

        assert result.dtype == mx.bfloat16

    def test_energy_preservation(self):
        """Test that total energy (Frobenius norm) is preserved."""
        from metal_marlin.kernels import hadamard_transform

        batch = 16
        x = mx.random.normal((batch, 64))
        mx.eval(x)

        # Unnormalized transform preserves energy up to scaling
        result = hadamard_transform(x, block_size=64, normalize=True, dtype=mx.float16)
        mx.eval(result)

        x_np = np.array(x.astype(mx.float32))
        result_np = np.array(result.astype(mx.float32))

        energy_before = np.sum(x_np ** 2)
        energy_after = np.sum(result_np ** 2)

        # For normalized transform, energy is preserved
        np.testing.assert_allclose(energy_before, energy_after, rtol=5e-2)


class TestHadamardNumpyReference:
    """Test the numpy reference implementation."""

    def test_numpy_hadamard_orthogonality(self):
        """Verify numpy reference implementation is orthogonal."""
        H = hadamard_matrix_numpy(64) / np.sqrt(64)
        identity = H @ H.T
        np.testing.assert_allclose(identity, np.eye(64), rtol=1e-10, atol=1e-10)

    def test_numpy_hadamard_self_inverse(self):
        """Verify H = H^T = H^-1 (up to normalization)."""
        H = hadamard_matrix_numpy(64)
        # H @ H = 64 * I
        product = H @ H
        np.testing.assert_allclose(product, 64 * np.eye(64), rtol=1e-10, atol=1e-10)

    def test_numpy_hadamard_symmetric(self):
        """Verify Hadamard matrix is symmetric."""
        H = hadamard_matrix_numpy(64)
        np.testing.assert_allclose(H, H.T, rtol=1e-10, atol=1e-10)
