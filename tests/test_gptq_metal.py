"""Unit tests for GPTQMetal - Metal-accelerated GPTQ operations.

Tests verify:
1. Correct initialization and error handling
2. Hessian computation accuracy vs CPU reference
3. Cholesky decomposition correctness
4. Numerical stability with edge cases
"""

from __future__ import annotations

import numpy as np
import pytest

from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch
from metal_marlin.gptq import compute_hessian as compute_hessian_cpu

try:
    from metal_marlin.gptq_metal import GPTQMetal

    HAS_GPTQ_METAL = True
except ImportError:
    HAS_GPTQ_METAL = False

# Skip all tests if Metal is not available
pytestmark = pytest.mark.skipif(
    not HAS_GPTQ_METAL or not HAS_MPS,
    reason="GPTQMetal not available or MPS not available",
)


class TestGPTQMetalInit:
    """Test GPTQMetal initialization."""

    def test_init_success(self):
        """GPTQMetal should initialize without error on Apple Silicon."""
        gptq = GPTQMetal()
        assert gptq is not None
        assert gptq._device is not None
        assert gptq._command_queue is not None

    def test_init_with_device(self):
        """GPTQMetal should accept explicit device."""
        import Metal

        device = Metal.MTLCreateSystemDefaultDevice()
        gptq = GPTQMetal(device=device)
        assert gptq._device == device


class TestHessianComputation:
    """Test Hessian matrix computation."""

    @pytest.fixture
    def gptq(self):
        return GPTQMetal()

    def test_hessian_shape(self, gptq):
        """Hessian should have shape [in_features, in_features]."""
        n_samples, in_features = 100, 256
        X = torch.randn(n_samples, in_features, device="mps")
        H = gptq.compute_hessian(X)
        assert H.shape == (in_features, in_features)

    def test_hessian_symmetry(self, gptq):
        """Hessian should be symmetric."""
        X = torch.randn(50, 128, device="mps")
        H = gptq.compute_hessian(X)
        # H should equal H^T
        diff = torch.abs(H - H.T).max().item()
        assert diff < 1e-5, f"Hessian not symmetric, max diff: {diff}"

    def test_hessian_positive_semidefinite(self, gptq):
        """Hessian should be positive semi-definite."""
        X = torch.randn(100, 64, device="mps")
        H = gptq.compute_hessian(X)
        # All eigenvalues should be >= 0
        eigenvalues = torch.linalg.eigvalsh(H)
        min_eigenvalue = eigenvalues.min().item()
        assert min_eigenvalue >= -1e-6, f"Negative eigenvalue: {min_eigenvalue}"

    def test_hessian_vs_cpu(self, gptq):
        """Metal Hessian should match CPU reference."""
        X_np = np.random.randn(100, 128).astype(np.float32)

        # CPU reference
        H_cpu = compute_hessian_cpu(X_np, normalize=True)

        # Metal computation
        X_tensor = torch.from_numpy(X_np)
        H_metal = gptq.compute_hessian(X_tensor, normalize=True)
        H_metal_np = H_metal.cpu().numpy()

        # Compare
        max_diff = np.abs(H_cpu - H_metal_np).max()
        assert max_diff < 1e-4, f"CPU vs Metal mismatch: {max_diff}"

    def test_hessian_normalization(self, gptq):
        """Test normalize parameter."""
        X = torch.randn(100, 64, device="mps")
        H_norm = gptq.compute_hessian(X, normalize=True)
        H_unnorm = gptq.compute_hessian(X, normalize=False)

        # H_unnorm should be 100x H_norm
        ratio = (H_unnorm / H_norm).mean().item()
        assert abs(ratio - 100) < 1, f"Normalization ratio: {ratio}"


class TestCholeskyDecomposition:
    """Test Cholesky decomposition."""

    @pytest.fixture
    def gptq(self):
        return GPTQMetal()

    def test_cholesky_lower_triangular(self, gptq):
        """Cholesky L should be lower triangular."""
        X = torch.randn(100, 64, device="mps")
        H = gptq.compute_hessian(X)
        L = gptq.cholesky_decompose(H)

        # Check upper triangle is zero
        upper = torch.triu(L, diagonal=1)
        max_upper = upper.abs().max().item()
        assert max_upper < 1e-6, f"Upper triangle not zero: {max_upper}"

    def test_cholesky_reconstruction(self, gptq):
        """L @ L^T should reconstruct H."""
        X = torch.randn(100, 64, device="mps")
        H = gptq.compute_hessian(X)
        L = gptq.cholesky_decompose(H)

        # Reconstruct
        H_reconstructed = L @ L.T

        # Compare
        max_diff = (H - H_reconstructed).abs().max().item()
        assert max_diff < 1e-4, f"Reconstruction error: {max_diff}"

    def test_cholesky_with_regularization(self, gptq):
        """Test regularization parameter."""
        # Create near-singular matrix
        X = torch.randn(10, 64, device="mps")  # rank-deficient
        H = gptq.compute_hessian(X)

        # Should succeed with default regularization
        L = gptq.cholesky_decompose(H, regularization=1e-4)
        assert L is not None
        assert not torch.isnan(L).any()


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def gptq(self):
        return GPTQMetal()

    def test_single_sample(self, gptq):
        """Should handle single sample."""
        X = torch.randn(1, 64, device="mps")
        H = gptq.compute_hessian(X)
        assert H.shape == (64, 64)

    def test_large_features(self, gptq):
        """Should handle large feature dimension."""
        X = torch.randn(50, 1024, device="mps")
        H = gptq.compute_hessian(X)
        assert H.shape == (1024, 1024)

    def test_numpy_input(self, gptq):
        """Should accept numpy array input."""
        X_np = np.random.randn(100, 64).astype(np.float32)
        H = gptq.compute_hessian(X_np)
        assert isinstance(H, torch.Tensor)
        assert H.device.type == "mps"

    def test_invalid_input_shape(self, gptq):
        """Should raise error for non-2D input."""
        X = torch.randn(10, 64, 32, device="mps")
        with pytest.raises(ValueError, match="must be 2D"):
            gptq.compute_hessian(X)
