"""Unit tests for GPTQMetal - Metal-accelerated GPTQ operations.

Tests verify:
1. Correct initialization and error handling
2. Hessian computation accuracy vs CPU reference
3. Cholesky decomposition correctness
4. Numerical stability with edge cases
"""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest

from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch
from metal_marlin.gptq import compute_hessian as compute_hessian_cpu

try:
    from metal_marlin.gptq_metal import GPTQMetal

    HAS_GPTQ_METAL = True
except ImportError:
    HAS_GPTQ_METAL = False
_FORCE_TORCH_MATMUL_ENV = "METAL_MARLIN_GPTQ_FORCE_TORCH_MATMUL"

# Skip all tests if Metal is not available
pytestmark = pytest.mark.skipif(
    not HAS_GPTQ_METAL or not HAS_MPS,
    reason="GPTQMetal not available or MPS not available",
)


def _hessian_reference(X: torch.Tensor, *, normalize: bool) -> torch.Tensor:
    """Reference Hessian computation using torch matmul on MPS."""
    x_fp32 = X.to(device="mps", dtype=torch.float32)
    hessian = (2.0 * (x_fp32.T @ x_fp32)).to(device="mps", dtype=torch.float32).contiguous()
    if normalize:
        hessian /= float(x_fp32.shape[0])
    return hessian


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
        assert diff < 5.0, f"Hessian not symmetric, max diff: {diff}"

    def test_hessian_positive_semidefinite(self, gptq):
        """Hessian should be positive semi-definite."""
        X = torch.randn(100, 64, device="mps")
        H = gptq.compute_hessian(X)
        # All eigenvalues should be >= 0
        eigenvalues = torch.linalg.eigvalsh(H.cpu())
        min_eigenvalue = eigenvalues.min().item()
        assert min_eigenvalue >= -5.0, f"Negative eigenvalue: {min_eigenvalue}"

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
        assert max_diff < 5e-3, f"CPU vs Metal mismatch: {max_diff}"

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
        with pytest.raises((ValueError, RuntimeError)):
            gptq.compute_hessian(X)


class TestPathSelectionStability:
    """Test path selection stability and consistency."""

    @pytest.fixture
    def gptq(self):
        return GPTQMetal()

    def test_path_selection_consistency_across_calls(self, gptq):
        """Same input should make a consistent path decision across repeated calls."""
        X_aligned = torch.randn(100, 64, device="mps")

        dispatch_called: list[bool] = []
        outputs: list[torch.Tensor] = []
        for _ in range(5):
            with patch.dict(os.environ, {_FORCE_TORCH_MATMUL_ENV: "0"}, clear=False):
                with patch.object(
                    gptq,
                    "_dispatch_hessian_compute",
                    wraps=gptq._dispatch_hessian_compute,
                ) as dispatch_spy:
                    outputs.append(gptq.compute_hessian(X_aligned, normalize=True))
            dispatch_called.append(dispatch_spy.called)

        assert all(dispatch_called) or not any(dispatch_called)
        for output in outputs:
            assert torch.isfinite(output).all()
        for output in outputs[1:]:
            torch.testing.assert_close(outputs[0], output, atol=5e-3, rtol=5e-3)

    def test_deterministic_results_with_seeded_inputs(self, gptq):
        """Results should be deterministic with seeded inputs.
        
        Same seed should always produce the same Hessian, even across
        different GPTQMetal instances.
        """
        torch.manual_seed(42)
        X1 = torch.randn(64, 64, device="mps")
        
        torch.manual_seed(42)
        X2 = torch.randn(64, 64, device="mps")
        
        # Compute with same instance
        H1 = gptq.compute_hessian(X1, normalize=True)
        H2 = gptq.compute_hessian(X2, normalize=True)
        
        assert torch.allclose(H1, H2, atol=1e-6, rtol=1e-5), \
            "Results differ with same seed"
        
        # Compute with different instance
        gptq2 = GPTQMetal()
        torch.manual_seed(42)
        X3 = torch.randn(64, 64, device="mps")
        H3 = gptq2.compute_hessian(X3, normalize=True)
        
        assert torch.allclose(H1, H3, atol=1e-6, rtol=1e-5), \
            "Results differ across GPTQMetal instances"

    def test_env_override_forces_torch_path(self, gptq):
        """Env override should disable Metal dispatch and force torch fallback."""
        x = torch.randn(64, 64, device="mps")
        with patch.dict(os.environ, {_FORCE_TORCH_MATMUL_ENV: "1"}, clear=False):
            with patch.object(
                gptq,
                "_dispatch_hessian_compute",
                wraps=gptq._dispatch_hessian_compute,
            ) as dispatch_spy:
                with patch.object(
                    gptq,
                    "_torch_hessian_matmul",
                    wraps=gptq._torch_hessian_matmul,
                ) as torch_spy:
                    hessian = gptq.compute_hessian(x, normalize=True)

        dispatch_spy.assert_not_called()
        torch_spy.assert_called_once()
        torch.testing.assert_close(
            hessian,
            _hessian_reference(x, normalize=True),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_small_shape_fallback_boundary_policy(self, gptq):
        """Verify below-threshold fallback and policy-safe boundary behavior."""
        x_small = torch.randn(8, 31, device="mps")
        with patch.dict(os.environ, {_FORCE_TORCH_MATMUL_ENV: "0"}, clear=False):
            with patch.object(
                gptq,
                "_dispatch_hessian_compute",
                wraps=gptq._dispatch_hessian_compute,
            ) as dispatch_spy:
                h_small = gptq.compute_hessian(x_small, normalize=True)

        dispatch_spy.assert_not_called()
        torch.testing.assert_close(
            h_small,
            _hessian_reference(x_small, normalize=True),
            atol=1e-4,
            rtol=1e-4,
        )

        # At 32 features, assert stability only (no path assumption).
        x_boundary = torch.randn(8, 32, device="mps")
        with patch.dict(os.environ, {_FORCE_TORCH_MATMUL_ENV: "0"}, clear=False):
            h_default = gptq.compute_hessian(x_boundary, normalize=True)
        with patch.dict(os.environ, {_FORCE_TORCH_MATMUL_ENV: "1"}, clear=False):
            h_fallback = gptq.compute_hessian(x_boundary, normalize=True)

        assert torch.isfinite(h_default).all()
        assert h_default.shape == (32, 32)
        torch.testing.assert_close(h_default, h_fallback, atol=5e-2, rtol=5e-2)



class TestNumericalStability:
    """Test numerical stability with challenging inputs."""

    @pytest.fixture
    def gptq(self):
        return GPTQMetal()

    def test_numerical_stability_near_thresholds(self, gptq):
        """Compute Hessians for shapes near alignment/size thresholds.
        
        These are critical boundary cases where numerical issues are most
        likely to occur.
        """
        # Test shapes just above and below threshold
        threshold_shapes = [
            (32, 32),   # Exactly at threshold
            (33, 32),   # Just above
            (31, 32),   # Just below (should fall back to torch)
            (64, 32),   # Well above
            (96, 32),   # 3x threshold
        ]
        
        for n_samples, in_features in threshold_shapes:
            X = torch.randn(n_samples, in_features, device="mps")
            H = gptq.compute_hessian(X, normalize=True)
            
            # Check numerical properties
            assert H.shape == (in_features, in_features)
            assert torch.isfinite(H).all(), f"NaN/Inf detected for shape {(n_samples, in_features)}"
            assert H.abs().max().item() < 1e6, f"Hessian magnitude too large for shape {(n_samples, in_features)}"
            
            # Check symmetry (up to numerical precision)
            asymmetry = (H - H.T).abs().max().item()
            assert asymmetry < 1e-3, f"Hessian not symmetric for shape {(n_samples, in_features)}"

    def test_stability_with_extreme_values(self, gptq):
        """Test stability with extreme input values.
        
        Very small or very large values can cause numerical instability.
        """
        in_features = 64
        
        # Very small values (near FP16 underflow)
        X_small = torch.randn(64, in_features, device="mps", dtype=torch.float16) * 1e-4
        H_small = gptq.compute_hessian(X_small, normalize=True)
        assert torch.isfinite(H_small).all(), "Failed with very small values"
        
        # Very large values (near FP16 overflow)
        X_large = torch.randn(64, in_features, device="mps", dtype=torch.float16) * 1e2
        H_large = gptq.compute_hessian(X_large, normalize=True)
        assert torch.isfinite(H_large).all(), "Failed with very large values"
        
        # Mixed magnitudes
        X_mixed = torch.randn(64, in_features, device="mps")
        X_mixed[:, :32] *= 1e-2
        X_mixed[:, 32:] *= 1e2
        H_mixed = gptq.compute_hessian(X_mixed, normalize=True)
        assert torch.isfinite(H_mixed).all(), "Failed with mixed magnitudes"

    def test_error_handling_with_invalid_shapes(self, gptq):
        """Test error handling for problematic shapes.
        
        Invalid or unsupported shapes should raise clear errors rather than
        crashing or producing incorrect results.
        """
        # Zero dimensions
        with pytest.raises((ValueError, RuntimeError)):
            X = torch.randn(0, 64, device="mps")
            gptq.compute_hessian(X)
        
        # Very large unaligned shape (should handle gracefully)
        X_large_unaligned = torch.randn(100, 1001, device="mps")
        # Should not crash, should fall back to torch or handle properly
        H_large_unaligned = gptq.compute_hessian(X_large_unaligned, normalize=True)
        assert H_large_unaligned.shape == (1001, 1001)
        assert torch.isfinite(H_large_unaligned).all()

    def test_stability_with_repeated_computations(self, gptq):
        """Numerical stability over repeated computations.
        
        Multiple iterations should not accumulate errors.
        """
        X = torch.randn(100, 64, device="mps")
        
        # First computation
        H1 = gptq.compute_hessian(X, normalize=True)
        
        # Reuse same Hessian for Cholesky multiple times
        for _ in range(10):
            L = gptq.cholesky_decompose(H1)
            assert not torch.isnan(L).any(), "NaN detected in repeated Cholesky"
            
            # Reconstruct and verify
            H_reconstructed = L @ L.T
            diff = (H1 - H_reconstructed).abs().max().item()
            assert diff < 1e-4, f"Reconstruction error accumulated: {diff}"
        
        # Final check - recompute Hessian and compare
        H2 = gptq.compute_hessian(X, normalize=True)
        assert torch.allclose(H1, H2, atol=1e-6, rtol=1e-5)




class TestHessianPathEquivalence:
    """Test numerical equivalence between Metal and torch Hessian paths."""

    @pytest.fixture
    def gptq(self):
        return GPTQMetal()

    def _run_paths(self, gptq, X, normalize):
        # Run with Metal kernel (default)
        with patch.dict(os.environ, {_FORCE_TORCH_MATMUL_ENV: "0"}, clear=False):
            H_metal = gptq.compute_hessian(X, normalize=normalize)

        # Run with torch.matmul fallback
        with patch.dict(os.environ, {_FORCE_TORCH_MATMUL_ENV: "1"}, clear=False):
            H_torch = gptq.compute_hessian(X, normalize=normalize)

        return H_metal, H_torch

    @pytest.mark.parametrize(
        ("n_samples", "in_features", "dtype", "atol", "rtol"),
        [
            (32, 64, torch.float32, 2e-2, 2e-2),
            (48, 96, torch.float16, 5e-2, 5e-2),
            (16, 63, torch.float32, 1e-4, 1e-4),  # misaligned -> fallback policy
        ],
    )
    def test_equivalence_default_vs_forced_fallback(
        self,
        gptq,
        n_samples,
        in_features,
        dtype,
        atol,
        rtol,
    ):
        """Default and forced-fallback paths should remain numerically close."""
        X = torch.randn(n_samples, in_features, device="mps", dtype=dtype)
        H_metal, H_torch = self._run_paths(gptq, X, normalize=True)

        assert H_metal.shape == H_torch.shape
        assert torch.isfinite(H_metal).all()
        assert torch.isfinite(H_torch).all()
        torch.testing.assert_close(H_metal, H_torch, atol=atol, rtol=rtol)

    def test_equivalence_bfloat16(self, gptq):
        """BF16 inputs should remain close across default and fallback paths."""
        X = torch.randn(64, 128, device="mps", dtype=torch.bfloat16)
        H_metal, H_torch = self._run_paths(gptq, X, normalize=True)
        torch.testing.assert_close(H_metal, H_torch, atol=8e-2, rtol=8e-2)
