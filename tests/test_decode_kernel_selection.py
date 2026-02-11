"""Tests for decode kernel selection in the single-token (M=1) path."""

from __future__ import annotations

import pytest

from metal_marlin.inference.decode import select_decode_kernel


def test_decode_path_uses_base_optimized_kernel_for_m1() -> None:
    """Typical single-token decode should use the optimized GEMV kernel."""
    kernel = select_decode_kernel(M=1, N=256, K=4096)
    assert kernel == "decode_gemv_fp4"


@pytest.mark.parametrize(
    ("n", "k"),
    [
        (256, 4096),   # default decode kernel
        (512, 4096),   # wide decode kernel
        (256, 16384),  # tiled decode kernel
    ],
)
def test_decode_path_uses_optimized_kernels_for_m1(n: int, k: int) -> None:
    """All M=1 variants should stay on decode-optimized kernels."""
    kernel = select_decode_kernel(M=1, N=n, K=k)
    assert kernel in {"decode_gemv_fp4", "decode_gemv_fp4_wide", "decode_gemv_fp4_tiled"}
    assert kernel != "marlin_gemm_fp4"
