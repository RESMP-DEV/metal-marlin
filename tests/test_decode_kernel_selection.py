"""Tests for decode kernel selection in the single-token (M=1) case.

This test module verifies that the decode path correctly selects optimized
GEMV kernels for M=1 (single-token generation) vs. the full GEMM kernels
used for larger batch sizes.

Optimized decode kernels:
- decode_gemv_fp4: Default single-token decode
- decode_gemv_fp4_wide: Wide output dimension (N >= 512)
- decode_gemv_fp4_tiled: Large K dimension (K > 8192)
- decode_gemv_fp4_batched: Small batch (1 < M <= 8)

Fall-back kernel:
- marlin_gemm_fp4: Large batch (M > 8)
"""

from __future__ import annotations

import pytest

from metal_marlin.inference.decode import select_decode_kernel


class TestDecodeKernelSelectionM1:
    """Test optimized kernel selection for M=1 (single-token decode)."""

    def test_decode_path_uses_base_optimized_kernel_for_m1(self) -> None:
        """Typical single-token decode should use the optimized GEMV kernel."""
        kernel = select_decode_kernel(M=1, N=256, K=4096)
        assert kernel == "decode_gemv_fp4"

    def test_decode_path_uses_wide_kernel_for_large_n(self) -> None:
        """M=1 with N >= 512 should use wide kernel for better coalescing."""
        # Boundary: N = 512
        kernel = select_decode_kernel(M=1, N=512, K=4096)
        assert kernel == "decode_gemv_fp4_wide"

        # Above boundary
        kernel = select_decode_kernel(M=1, N=1024, K=4096)
        assert kernel == "decode_gemv_fp4_wide"

        # Large N
        kernel = select_decode_kernel(M=1, N=8192, K=4096)
        assert kernel == "decode_gemv_fp4_wide"

    def test_decode_path_uses_tiled_kernel_for_large_k(self) -> None:
        """M=1 with K > 8192 should use tiled kernel to cache A."""
        # Boundary: K just above 8192
        kernel = select_decode_kernel(M=1, N=256, K=8193)
        assert kernel == "decode_gemv_fp4_tiled"

        # Large K
        kernel = select_decode_kernel(M=1, N=256, K=16384)
        assert kernel == "decode_gemv_fp4_tiled"

        # Very large K
        kernel = select_decode_kernel(M=1, N=256, K=32768)
        assert kernel == "decode_gemv_fp4_tiled"

    def test_decode_path_wide_kernel_takes_precedence_over_tiled(self) -> None:
        """When both N >= 512 and K > 8192, wide kernel should be selected."""
        # N >= 512 takes precedence over K > 8192
        kernel = select_decode_kernel(M=1, N=512, K=16384)
        assert kernel == "decode_gemv_fp4_wide"

    @pytest.mark.parametrize(
        ("n", "k"),
        [
            (256, 4096),   # default decode kernel
            (512, 4096),   # wide decode kernel
            (1024, 4096),  # wider decode kernel
            (256, 8192),   # boundary: K not yet tiled
            (256, 16384),  # tiled decode kernel
            (256, 32768),  # larger tiled decode kernel
        ],
    )
    def test_decode_path_uses_optimized_kernels_for_various_dims(self, n: int, k: int) -> None:
        """All M=1 variants should stay on decode-optimized kernels."""
        kernel = select_decode_kernel(M=1, N=n, K=k)
        assert kernel in {
            "decode_gemv_fp4",
            "decode_gemv_fp4_wide",
            "decode_gemv_fp4_tiled",
        }
        assert kernel != "marlin_gemm_fp4"


class TestDecodeKernelSelectionBatch:
    """Test kernel selection for various batch sizes."""

    def test_decode_path_uses_batched_kernel_for_small_batch(self) -> None:
        """1 < M <= 8 should use batched decode kernel."""
        # M = 2
        kernel = select_decode_kernel(M=2, N=256, K=4096)
        assert kernel == "decode_gemv_fp4_batched"

        # M = 4
        kernel = select_decode_kernel(M=4, N=256, K=4096)
        assert kernel == "decode_gemv_fp4_batched"

        # M = 8 (boundary)
        kernel = select_decode_kernel(M=8, N=256, K=4096)
        assert kernel == "decode_gemv_fp4_batched"

    def test_decode_path_uses_gemm_for_large_batch(self) -> None:
        """M > 8 should fall back to full GEMM kernel."""
        # M = 9
        kernel = select_decode_kernel(M=9, N=256, K=4096)
        assert kernel == "marlin_gemm_fp4"

        # M = 16
        kernel = select_decode_kernel(M=16, N=256, K=4096)
        assert kernel == "marlin_gemm_fp4"

        # Large batch
        kernel = select_decode_kernel(M=128, N=256, K=4096)
        assert kernel == "marlin_gemm_fp4"

    @pytest.mark.parametrize(
        "m",
        [1, 2, 4, 8, 9, 16, 32, 64],
    )
    def test_kernel_selection_for_various_batch_sizes(self, m: int) -> None:
        """Verify kernel selection is consistent across batch sizes."""
        kernel = select_decode_kernel(M=m, N=256, K=4096)

        if m == 1:
            assert kernel == "decode_gemv_fp4"
        elif 1 < m <= 8:
            assert kernel == "decode_gemv_fp4_batched"
        else:  # m > 8
            assert kernel == "marlin_gemm_fp4"


class TestDecodeKernelSelectionEdgeCases:
    """Test edge cases for kernel selection."""

    def test_decode_path_boundary_n_511(self) -> None:
        """N = 511 should use base kernel, not wide."""
        kernel = select_decode_kernel(M=1, N=511, K=4096)
        assert kernel == "decode_gemv_fp4"

    def test_decode_path_boundary_k_8192(self) -> None:
        """K = 8192 should use base kernel, not tiled."""
        kernel = select_decode_kernel(M=1, N=256, K=8192)
        assert kernel == "decode_gemv_fp4"

    def test_decode_path_boundary_n_512_k_large(self) -> None:
        """N = 512 with large K should use wide kernel."""
        # N boundary takes precedence
        kernel = select_decode_kernel(M=1, N=512, K=8192)
        assert kernel == "decode_gemv_fp4_wide"

    def test_decode_path_small_n_large_k(self) -> None:
        """Small N with large K should use tiled kernel."""
        kernel = select_decode_kernel(M=1, N=128, K=16384)
        assert kernel == "decode_gemv_fp4_tiled"

    def test_decode_path_uses_optimized_not_gemm_for_m1(self) -> None:
        """Critical test: M=1 should never use the generic GEMM kernel."""
        # Test various configurations
        configs = [
            (1, 256, 4096),
            (1, 512, 4096),
            (1, 1024, 8192),
            (1, 256, 16384),
            (1, 512, 32768),
            (1, 128, 65536),
        ]

        for m, n, k in configs:
            kernel = select_decode_kernel(M=m, N=n, K=k)
            assert kernel != "marlin_gemm_fp4", (
                f"M=1 should not use marlin_gemm_fp4, got {kernel} for (M={m}, N={n}, K={k})"
            )
            assert kernel in {
                "decode_gemv_fp4",
                "decode_gemv_fp4_wide",
                "decode_gemv_fp4_tiled",
            }
