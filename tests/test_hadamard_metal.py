"""Unit tests for HadamardMetal - Metal-accelerated Hadamard transform.

Tests verify:
1. Correct Hadamard matrix generation via transform
2. Transform orthogonality
3. Self-inverse property
4. Numerical accuracy
"""

import numpy as np
import pytest

# Try to import Metal components
try:
    import torch

    from metal_marlin.hadamard_metal import HadamardMetal, hadamard_transform_metal
    from metal_marlin.metal_dispatch import require_metal, require_mps
    HAS_METAL = True
    try:
        require_metal()
        require_mps()
        HAS_MPS = torch.backends.mps.is_available()
    except (ImportError, RuntimeError):
        HAS_MPS = False
except ImportError:
    HAS_METAL = False
    HAS_MPS = False


pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")


def _is_power_of_two(n: int) -> bool:
    """Check if n is a power of two."""
    return n > 0 and (n & (n - 1)) == 0


def _generate_hadamard_matrix(n: int) -> np.ndarray:
    """Generate unnormalized Hadamard matrix via Sylvester construction."""
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


class TestUtilities:
    """Test utility functions."""

    def test_is_power_of_two_valid(self):
        """Valid powers of two."""
        for n in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            assert _is_power_of_two(n), f"{n} should be power of 2"

    def test_is_power_of_two_invalid(self):
        """Invalid values."""
        for n in [0, 3, 5, 6, 7, 9, 15, 100]:
            assert not _is_power_of_two(n), f"{n} should not be power of 2"


class TestHadamardMatrix:
    """Test Hadamard matrix generation properties."""

    def test_matrix_size_2(self):
        """H_2 should be [[1, 1], [1, -1]]."""
        H = _generate_hadamard_matrix(2)
        expected = np.array([[1, 1], [1, -1]], dtype=np.float32)
        np.testing.assert_array_equal(H, expected)

    def test_matrix_entries(self):
        """All entries should be ±1."""
        for n in [4, 8, 16, 32]:
            H = _generate_hadamard_matrix(n)
            assert set(H.flatten().tolist()) == {1.0, -1.0}

    def test_matrix_recursive_construction(self):
        """Sylvester construction should produce valid Hadamard."""
        # H_4 = [[H_2, H_2], [H_2, -H_2]]
        H2 = _generate_hadamard_matrix(2)
        H4 = _generate_hadamard_matrix(4)
        expected = np.block([[H2, H2], [H2, -H2]])
        np.testing.assert_array_equal(H4, expected)


class TestOrthogonality:
    """Test orthogonality properties."""

    def test_orthogonality_unnormalized(self):
        """H @ H^T should equal n * I."""
        for n in [4, 8, 16, 32, 64]:
            H = _generate_hadamard_matrix(n)
            product = H @ H.T
            expected = n * np.eye(n, dtype=np.float32)
            np.testing.assert_array_almost_equal(product, expected, decimal=5)

    def test_orthogonality_normalized(self):
        """(H/√n) @ (H/√n)^T should equal I."""
        for n in [4, 8, 16, 32, 64]:
            H = _generate_hadamard_matrix(n) / np.sqrt(n)
            product = H @ H.T
            expected = np.eye(n, dtype=np.float32)
            max_error = np.abs(product - expected).max()
            assert max_error < 1e-5, f"n={n}: orthogonality error {max_error}"


@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
class TestHadamardTransformMetal:
    """Test Metal-accelerated Hadamard transform."""

    def test_transform_shape_32(self):
        """Transform should preserve shape with block_size=32."""
        batch = 16
        x = torch.randn(batch, 32, device="mps", dtype=torch.float16)
        y = hadamard_transform_metal(x, block_size=32, normalize=True)
        assert y.shape == x.shape

    def test_transform_shape_64(self):
        """Transform should preserve shape with block_size=64."""
        batch = 16
        x = torch.randn(batch, 64, device="mps", dtype=torch.float16)
        y = hadamard_transform_metal(x, block_size=64, normalize=True)
        assert y.shape == x.shape

    def test_transform_shape_128(self):
        """Transform should preserve shape with block_size=128."""
        batch = 16
        x = torch.randn(batch, 128, device="mps", dtype=torch.float16)
        y = hadamard_transform_metal(x, block_size=128, normalize=True)
        assert y.shape == x.shape

    def test_transform_3d(self):
        """Should handle 3D input."""
        x = torch.randn(2, 5, 64, device="mps", dtype=torch.float16)
        y = hadamard_transform_metal(x, block_size=64, normalize=True)
        assert y.shape == x.shape

    def test_invalid_block_size(self):
        """Should raise error for invalid block_size."""
        x = torch.randn(4, 64, device="mps", dtype=torch.float16)
        with pytest.raises(ValueError, match="block_size must be"):
            hadamard_transform_metal(x, block_size=48)

    def test_mismatched_dimension(self):
        """Should raise error when dimension doesn't match block_size."""
        x = torch.randn(4, 100, device="mps", dtype=torch.float16)
        with pytest.raises(ValueError, match="must equal block_size"):
            hadamard_transform_metal(x, block_size=64)

    def test_dtype_float16(self):
        """Should handle float16 input."""
        x = torch.randn(4, 64, device="mps", dtype=torch.float16)
        y = hadamard_transform_metal(x, block_size=64)
        assert y.dtype == torch.float16

    def test_dtype_float32(self):
        """Should handle float32 input (convert internally)."""
        x = torch.randn(4, 64, device="mps", dtype=torch.float32)
        y = hadamard_transform_metal(x, block_size=64)
        # Kernel internally works in half
        assert y.dtype == torch.float16


@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
class TestSelfInverse:
    """Test self-inverse property using Metal kernel."""

    def test_inverse_normalized_32(self):
        """Normalized transform applied twice should equal identity (block_size=32)."""
        torch.manual_seed(42)
        batch = 8
        x = torch.randn(batch, 32, device="mps", dtype=torch.float32)
        x_np = x.cpu().float().numpy()

        # Apply normalized Hadamard twice: H_norm @ H_norm @ x = x
        y = hadamard_transform_metal(x, block_size=32, normalize=True)
        z = hadamard_transform_metal(y, block_size=32, normalize=True)

        z_np = z.cpu().float().numpy()

        # Allow some tolerance for FP16 computation
        np.testing.assert_allclose(z_np, x_np, rtol=5e-2, atol=5e-2)

    def test_inverse_normalized_64(self):
        """Normalized transform applied twice should equal identity (block_size=64)."""
        torch.manual_seed(42)
        batch = 8
        x = torch.randn(batch, 64, device="mps", dtype=torch.float32)
        x_np = x.cpu().float().numpy()

        y = hadamard_transform_metal(x, block_size=64, normalize=True)
        z = hadamard_transform_metal(y, block_size=64, normalize=True)

        z_np = z.cpu().float().numpy()
        np.testing.assert_allclose(z_np, x_np, rtol=5e-2, atol=5e-2)

    def test_inverse_normalized_128(self):
        """Normalized transform applied twice should equal identity (block_size=128)."""
        torch.manual_seed(42)
        batch = 8
        x = torch.randn(batch, 128, device="mps", dtype=torch.float32)
        x_np = x.cpu().float().numpy()

        y = hadamard_transform_metal(x, block_size=128, normalize=True)
        z = hadamard_transform_metal(y, block_size=128, normalize=True)

        z_np = z.cpu().float().numpy()
        np.testing.assert_allclose(z_np, x_np, rtol=5e-2, atol=5e-2)

    def test_unnormalized_scale(self):
        """Unnormalized result should be sqrt(n) times normalized result."""
        torch.manual_seed(42)
        batch = 4
        x = torch.randn(batch, 64, device="mps", dtype=torch.float32)

        result_unnorm = hadamard_transform_metal(x, block_size=64, normalize=False)
        result_norm = hadamard_transform_metal(x, block_size=64, normalize=True)

        # Unnormalized = normalized * sqrt(block_size)
        scale = np.sqrt(64)
        unnorm_np = result_unnorm.cpu().float().numpy()
        norm_np = result_norm.cpu().float().numpy()

        np.testing.assert_allclose(unnorm_np, norm_np * scale, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
class TestHadamardMetalClass:
    """Test HadamardMetal class interface."""

    @pytest.fixture
    def handler(self):
        return HadamardMetal()

    def test_class_transform(self, handler):
        """HadamardMetal.transform should work."""
        x = torch.randn(16, 64, device="mps", dtype=torch.float16)
        y = handler.transform(x, block_size=64, normalize=True)
        assert y.shape == x.shape

    def test_class_transform_numpy(self, handler):
        """HadamardMetal.transform_numpy should work with numpy arrays."""
        x = np.random.randn(16, 64).astype(np.float32)
        y = handler.transform_numpy(x, block_size=64, normalize=True)
        assert y.shape == x.shape
        assert isinstance(y, np.ndarray)

    def test_class_roundtrip_numpy(self, handler):
        """Double transform should recover original (numpy path)."""
        np.random.seed(42)
        x = np.random.randn(8, 64).astype(np.float32)

        y = handler.transform_numpy(x, block_size=64, normalize=True)
        z = handler.transform_numpy(y, block_size=64, normalize=True)

        np.testing.assert_allclose(z, x, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
class TestEnergyPreservation:
    """Test that total energy (Frobenius norm) is preserved."""

    def test_energy_preserved_normalized(self):
        """Normalized transform preserves energy (orthogonal)."""
        torch.manual_seed(42)
        batch = 16
        x = torch.randn(batch, 64, device="mps", dtype=torch.float32)

        result = hadamard_transform_metal(x, block_size=64, normalize=True)

        x_np = x.cpu().float().numpy()
        result_np = result.cpu().float().numpy()

        energy_before = np.sum(x_np ** 2)
        energy_after = np.sum(result_np ** 2)

        # For normalized transform, energy is preserved
        np.testing.assert_allclose(energy_before, energy_after, rtol=5e-2)

    def test_energy_scaled_unnormalized(self):
        """Unnormalized transform scales energy by n."""
        torch.manual_seed(42)
        batch = 16
        x = torch.randn(batch, 64, device="mps", dtype=torch.float32)

        result = hadamard_transform_metal(x, block_size=64, normalize=False)

        x_np = x.cpu().float().numpy()
        result_np = result.cpu().float().numpy()

        energy_before = np.sum(x_np ** 2)
        energy_after = np.sum(result_np ** 2)

        # For unnormalized transform, energy is scaled by block_size
        np.testing.assert_allclose(energy_after, energy_before * 64, rtol=5e-2)


class TestNumpyReference:
    """Test Metal results against numpy reference implementation."""

    def _reference_hadamard_transform(
        self,
        x: np.ndarray,
        block_size: int,
        normalize: bool = True
    ) -> np.ndarray:
        """Reference implementation using numpy."""
        H = _generate_hadamard_matrix(block_size)
        if normalize:
            H = H / np.sqrt(block_size)

        orig_shape = x.shape
        x_2d = x.reshape(-1, block_size)
        result = x_2d @ H.T
        return result.reshape(orig_shape)

    @pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
    def test_accuracy_block_32(self):
        """Metal result should match numpy reference (block_size=32)."""
        torch.manual_seed(42)
        np.random.seed(42)
        batch = 8
        x_np = np.random.randn(batch, 32).astype(np.float32)

        # Reference
        expected = self._reference_hadamard_transform(x_np, block_size=32, normalize=True)

        # Metal kernel
        x_torch = torch.tensor(x_np, device="mps", dtype=torch.float16)
        result = hadamard_transform_metal(x_torch, block_size=32, normalize=True)
        result_np = result.cpu().float().numpy()

        # FP16 has limited precision
        np.testing.assert_allclose(result_np, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
    def test_accuracy_block_64(self):
        """Metal result should match numpy reference (block_size=64)."""
        torch.manual_seed(42)
        np.random.seed(42)
        batch = 8
        x_np = np.random.randn(batch, 64).astype(np.float32)

        expected = self._reference_hadamard_transform(x_np, block_size=64, normalize=True)

        x_torch = torch.tensor(x_np, device="mps", dtype=torch.float16)
        result = hadamard_transform_metal(x_torch, block_size=64, normalize=True)
        result_np = result.cpu().float().numpy()

        np.testing.assert_allclose(result_np, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
    def test_accuracy_block_128(self):
        """Metal result should match numpy reference (block_size=128)."""
        torch.manual_seed(42)
        np.random.seed(42)
        batch = 8
        x_np = np.random.randn(batch, 128).astype(np.float32)

        expected = self._reference_hadamard_transform(x_np, block_size=128, normalize=True)

        x_torch = torch.tensor(x_np, device="mps", dtype=torch.float16)
        result = hadamard_transform_metal(x_torch, block_size=128, normalize=True)
        result_np = result.cpu().float().numpy()

        np.testing.assert_allclose(result_np, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
class TestLargeBatch:
    """Test with large batch sizes."""

    def test_large_batch_64(self):
        """Test with large batch size."""
        batch = 1024
        x = torch.randn(batch, 64, device="mps", dtype=torch.float16)

        result = hadamard_transform_metal(x, block_size=64, normalize=True)

        assert result.shape == (batch, 64)

    def test_large_batch_128(self):
        """Test with large batch size and block_size=128."""
        batch = 512
        x = torch.randn(batch, 128, device="mps", dtype=torch.float16)

        result = hadamard_transform_metal(x, block_size=128, normalize=True)

        assert result.shape == (batch, 128)
