"""Regression tests for GPTQMetal Hessian path selection and stability."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from metal_marlin._compat import HAS_MPS, torch

_FORCE_TORCH_MATMUL_ENV = "METAL_MARLIN_GPTQ_FORCE_TORCH_MATMUL"

try:
    from metal_marlin.gptq_metal import GPTQMetal

    HAS_GPTQ_METAL = True
except ImportError:
    HAS_GPTQ_METAL = False


pytestmark = pytest.mark.skipif(
    not HAS_GPTQ_METAL or not HAS_MPS,
    reason="GPTQMetal not available or MPS not available",
)


def _make_activations(
    *,
    seed: int,
    n_samples: int,
    in_features: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n_samples, in_features, dtype=dtype, device="mps")


def _reference_hessian(X: torch.Tensor, *, normalize: bool) -> torch.Tensor:
    x_fp32 = X.to(device="mps", dtype=torch.float32)
    hessian = (2.0 * (x_fp32.T @ x_fp32)).to(device="mps", dtype=torch.float32).contiguous()
    if normalize:
        hessian /= float(x_fp32.shape[0])
    return hessian


@pytest.fixture
def gptq() -> GPTQMetal:
    return GPTQMetal()


def test_default_path_is_deterministic_and_safe(gptq: GPTQMetal) -> None:
    """Default path should be deterministic across repeated invocations."""
    x = _make_activations(seed=17, n_samples=64, in_features=64, dtype=torch.float16)

    with patch.dict(os.environ, {_FORCE_TORCH_MATMUL_ENV: "0"}, clear=False):
        results = [gptq.compute_hessian(x, normalize=True) for _ in range(5)]

    for result in results:
        assert result.shape == (64, 64)
        assert result.device.type == "mps"
        assert result.dtype == torch.float32
        assert result.is_contiguous()
        assert torch.isfinite(result).all()

    for result in results[1:]:
        torch.testing.assert_close(results[0], result, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize(
    ("n_samples", "in_features", "expect_fallback"),
    [
        (1, 31, True),
        (8, 15, True),
        (8, 33, True),
        (12, 17, True),
        (16, 65, True),
        (24, 128, False),
        (32, 256, False),
    ],
)
def test_default_path_shape_regressions_are_stable(
    gptq: GPTQMetal,
    n_samples: int,
    in_features: int,
    expect_fallback: bool,
) -> None:
    """Known-regression shapes must remain stable and finite."""
    x = _make_activations(
        seed=1000 + (n_samples * 10) + in_features,
        n_samples=n_samples,
        in_features=in_features,
    )

    with patch.dict(os.environ, {_FORCE_TORCH_MATMUL_ENV: "0"}, clear=False):
        with patch.object(
            gptq,
            "_dispatch_hessian_compute",
            wraps=gptq._dispatch_hessian_compute,
        ) as dispatch_spy:
            h_default = gptq.compute_hessian(x, normalize=True)

    assert h_default.shape == (in_features, in_features)
    assert torch.isfinite(h_default).all()

    if expect_fallback:
        dispatch_spy.assert_not_called()
        h_ref = _reference_hessian(x, normalize=True)
        torch.testing.assert_close(h_default, h_ref, atol=1e-4, rtol=1e-4)


def test_env_override_forces_torch_matmul_path(gptq: GPTQMetal) -> None:
    """Env override should bypass Metal dispatch and force torch matmul."""
    x = _make_activations(seed=23, n_samples=32, in_features=64)

    with patch.dict(os.environ, {_FORCE_TORCH_MATMUL_ENV: "1"}, clear=False):
        with patch.object(
            gptq,
            "_torch_hessian_matmul",
            wraps=gptq._torch_hessian_matmul,
        ) as torch_spy:
            with patch.object(
                gptq,
                "_dispatch_hessian_compute",
                wraps=gptq._dispatch_hessian_compute,
            ) as dispatch_spy:
                hessian = gptq.compute_hessian(x, normalize=True)

    torch_spy.assert_called_once()
    dispatch_spy.assert_not_called()
    torch.testing.assert_close(
        hessian,
        _reference_hessian(x, normalize=True),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    ("n_samples", "in_features", "dtype", "atol", "rtol"),
    [
        (32, 64, torch.float32, 2e-2, 2e-2),
        (48, 96, torch.float16, 5e-2, 5e-2),
        (64, 128, torch.bfloat16, 8e-2, 8e-2),
    ],
)
def test_default_and_forced_fallback_are_numerically_close(
    gptq: GPTQMetal,
    n_samples: int,
    in_features: int,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    """Default path should stay numerically close to forced torch fallback."""
    x = _make_activations(
        seed=2000 + (n_samples * 10) + in_features,
        n_samples=n_samples,
        in_features=in_features,
        dtype=dtype,
    )

    with patch.dict(os.environ, {_FORCE_TORCH_MATMUL_ENV: "0"}, clear=False):
        h_default = gptq.compute_hessian(x, normalize=True)

    with patch.dict(os.environ, {_FORCE_TORCH_MATMUL_ENV: "1"}, clear=False):
        h_fallback = gptq.compute_hessian(x, normalize=True)

    assert torch.isfinite(h_default).all()
    assert torch.isfinite(h_fallback).all()
    assert h_default.shape == h_fallback.shape
    torch.testing.assert_close(h_default, h_fallback, atol=atol, rtol=rtol)


def test_storage_offset_view_forces_safe_fallback(gptq: GPTQMetal) -> None:
    """Non-zero storage offset view should avoid custom Metal dispatch."""
    base = _make_activations(seed=3100, n_samples=20, in_features=128)
    x_view = base[:, 32:96]  # [20, 64], non-zero storage offset
    assert x_view.storage_offset() > 0

    with patch.dict(os.environ, {_FORCE_TORCH_MATMUL_ENV: "0"}, clear=False):
        with patch.object(
            gptq,
            "_dispatch_hessian_compute",
            wraps=gptq._dispatch_hessian_compute,
        ) as dispatch_spy:
            hessian = gptq.compute_hessian(x_view, normalize=True)

    dispatch_spy.assert_not_called()
    torch.testing.assert_close(hessian, _reference_hessian(x_view, normalize=True), atol=1e-4, rtol=1e-4)


def test_non_unit_inner_stride_forces_safe_fallback(gptq: GPTQMetal) -> None:
    """Non-unit innermost stride should avoid custom Metal dispatch."""
    base = _make_activations(seed=3200, n_samples=24, in_features=128)
    x_strided = torch.as_strided(base, size=(24, 32), stride=(128, 2))
    assert x_strided.stride(-1) == 2

    with patch.dict(os.environ, {_FORCE_TORCH_MATMUL_ENV: "0"}, clear=False):
        with patch.object(
            gptq,
            "_dispatch_hessian_compute",
            wraps=gptq._dispatch_hessian_compute,
        ) as dispatch_spy:
            hessian = gptq.compute_hessian(x_strided, normalize=True)

    dispatch_spy.assert_not_called()
    torch.testing.assert_close(
        hessian,
        _reference_hessian(x_strided, normalize=True),
        atol=1e-4,
        rtol=1e-4,
    )


def test_small_shape_policy_boundary_without_route_assumptions(gptq: GPTQMetal) -> None:
    """Respect the small-shape fallback policy without assuming the >=32 route."""
    x_small = _make_activations(seed=4100, n_samples=8, in_features=31)
    with patch.dict(os.environ, {_FORCE_TORCH_MATMUL_ENV: "0"}, clear=False):
        with patch.object(
            gptq,
            "_dispatch_hessian_compute",
            wraps=gptq._dispatch_hessian_compute,
        ) as dispatch_spy:
            h_small = gptq.compute_hessian(x_small, normalize=True)

    dispatch_spy.assert_not_called()
    torch.testing.assert_close(h_small, _reference_hessian(x_small, normalize=True), atol=1e-4, rtol=1e-4)

    # At threshold (32), do not assume whether Metal dispatch is selected.
    x_threshold = _make_activations(seed=4200, n_samples=8, in_features=32)
    with patch.dict(os.environ, {_FORCE_TORCH_MATMUL_ENV: "0"}, clear=False):
        h_default = gptq.compute_hessian(x_threshold, normalize=True)

    with patch.dict(os.environ, {_FORCE_TORCH_MATMUL_ENV: "1"}, clear=False):
        h_forced_fallback = gptq.compute_hessian(x_threshold, normalize=True)

    assert h_default.shape == (32, 32)
    assert torch.isfinite(h_default).all()
    torch.testing.assert_close(h_default, h_forced_fallback, atol=5e-2, rtol=5e-2)
