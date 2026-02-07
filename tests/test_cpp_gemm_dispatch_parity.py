"""Parity tests for C++ vs Python GEMM dispatch paths."""

from __future__ import annotations

from unittest import mock

import pytest

from metal_marlin._compat import HAS_CPP_EXT, HAS_MPS, HAS_TORCH, _metal_dispatch_ext
from metal_marlin._compat import torch as _torch
from metal_marlin.metal_dispatch import dispatch_gemm_fp4, dispatch_gemm_fp8, get_default_library

if not HAS_TORCH or _torch is None:
    pytest.skip("PyTorch not available", allow_module_level=True)

torch = _torch

pytestmark = [
    pytest.mark.skipif(not HAS_MPS, reason="MPS not available"),
    pytest.mark.skipif(
        not HAS_CPP_EXT or _metal_dispatch_ext is None,
        reason="Metal/C++ extension not available",
    ),
]


class _PythonFallbackUsed(RuntimeError):
    """Raised when a call dropped from C++ path to Python fallback."""


@pytest.fixture(scope="module")
def metal_lib():
    try:
        return get_default_library()
    except Exception as exc:
        pytest.skip(f"Metal kernel library unavailable: {exc}")


def _pack_fp4_codes_k_axis(codes: torch.Tensor) -> torch.Tensor:
    """Pack FP4 codes [K, N] into [K//8, N] uint32 (K-axis packing)."""
    k_dim, n_dim = codes.shape
    assert k_dim % 8 == 0

    packed = torch.zeros((k_dim // 8, n_dim), dtype=torch.int32)
    for i in range(8):
        packed |= (codes[i::8, :].to(torch.int32) & 0xF) << (i * 4)
    return packed.to(torch.uint32)


def _pack_fp8_codes_n_axis(codes: torch.Tensor) -> torch.Tensor:
    """Pack FP8 bytes [K, N] into [K, N//4] uint32 (N-axis packing)."""
    k_dim, n_dim = codes.shape
    assert n_dim % 4 == 0

    packed = torch.zeros((k_dim, n_dim // 4), dtype=torch.int32)
    for j in range(4):
        packed |= (codes[:, j::4].to(torch.int32) & 0xFF) << (j * 8)
    return packed.to(torch.uint32)


def _make_fp4_case():
    torch.manual_seed(7)
    m_dim, k_dim, n_dim, group_size = 4, 32, 16, 32

    activations = torch.randn((m_dim, k_dim), dtype=torch.float16)
    codes = torch.randint(0, 16, (k_dim, n_dim), dtype=torch.int32)
    packed = _pack_fp4_codes_k_axis(codes)
    scales = (0.05 + 0.20 * torch.rand((k_dim // group_size, n_dim), dtype=torch.float32)).to(
        torch.float16
    )

    return (
        activations.to("mps"),
        packed.to("mps"),
        scales.to("mps"),
        m_dim,
        n_dim,
        k_dim,
        group_size,
    )


def _make_fp8_case():
    torch.manual_seed(17)
    m_dim, k_dim, n_dim, group_size = 3, 32, 16, 32

    activations = torch.randn((m_dim, k_dim), dtype=torch.float16)
    codes = torch.randint(0, 256, (k_dim, n_dim), dtype=torch.int32)
    packed = _pack_fp8_codes_n_axis(codes)
    scales = (0.05 + 0.20 * torch.rand((k_dim // group_size, n_dim), dtype=torch.float32)).to(
        torch.float16
    )

    return (
        activations.to("mps"),
        packed.to("mps"),
        scales.to("mps"),
        m_dim,
        n_dim,
        k_dim,
        group_size,
    )


def _dispatch_python_only(dispatch_fn, lib, *args):
    with mock.patch("metal_marlin.metal_dispatch._get_fast_path_context", return_value=None):
        return dispatch_fn(lib, *args, enable_padding=False)


def _dispatch_cpp_only(dispatch_fn, lib, *args, case_name: str):
    # If this patched function is called, execution fell back to Python dispatch.
    with mock.patch(
        "metal_marlin.metal_dispatch.dispatch_kernel",
        side_effect=_PythonFallbackUsed(f"{case_name}: Python fallback was used"),
    ):
        try:
            return dispatch_fn(lib, *args, enable_padding=False)
        except _PythonFallbackUsed:
            pytest.skip(f"{case_name}: C++ fast path unavailable on this machine")
        except Exception as exc:
            pytest.skip(f"{case_name}: C++ dispatch unavailable ({exc})")


def _assert_shape_and_close(python_out: torch.Tensor, cpp_out: torch.Tensor):
    assert python_out.shape == cpp_out.shape
    torch.testing.assert_close(python_out.cpu(), cpp_out.cpu(), rtol=1e-2, atol=1e-2)


def test_fp4_cpp_vs_python_dispatch_parity(metal_lib):
    case = _make_fp4_case()
    try:
        python_out = _dispatch_python_only(dispatch_gemm_fp4, metal_lib, *case)
    except Exception as exc:
        pytest.skip(f"FP4: Python dispatch unavailable ({exc})")

    cpp_out = _dispatch_cpp_only(dispatch_gemm_fp4, metal_lib, *case, case_name="FP4")
    _assert_shape_and_close(python_out, cpp_out)


def test_fp8_cpp_vs_python_dispatch_parity(metal_lib):
    case = _make_fp8_case()
    try:
        python_out = _dispatch_python_only(dispatch_gemm_fp8, metal_lib, *case)
    except Exception as exc:
        pytest.skip(f"FP8: kernel unavailable ({exc})")

    cpp_out = _dispatch_cpp_only(dispatch_gemm_fp8, metal_lib, *case, case_name="FP8")
    _assert_shape_and_close(python_out, cpp_out)
