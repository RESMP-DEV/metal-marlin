"""GEMM boundary tests: non-tile-aligned dimensions, partial tiles, and extremes.

The Metal Marlin kernel uses TILE_M=16, TILE_N=16, TILE_K=32 with 128 threads
(4 simdgroups). These tests exercise dimensions that produce partial tiles
to verify boundary guards handle them correctly.

Constraints on valid dimensions:
  - N must be divisible by 8 (FP4 packing: 8 nibbles per uint32 along N)
  - K must be divisible by group_size (per-group scale lookup)
  - M can be anything (boundary-guarded at tile edges)

Tests are split into:
  1. Reference-only (NumPy): validates quantize/dequant/GEMM logic at boundaries
  2. Metal kernel: validates the kernel handles partial tiles correctly
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from metal_marlin._compat import HAS_MLX

# ---------------------------------------------------------------------------
# FP4 E2M1 reference utilities (same as test_accuracy.py)
# ---------------------------------------------------------------------------

FP4_E2M1_TABLE: np.ndarray = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float16,
)


def _quantize_to_fp4(
    weights: np.ndarray, group_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize FP16 weights [K, N] to packed FP4 format.

    Returns:
        packed: [K, N//8] uint32 (8 consecutive N-values per word)
        scales: [K//group_size, N] float16
    """
    K, N = weights.shape
    assert K % group_size == 0
    assert N % 8 == 0

    num_groups = K // group_size
    weights_grouped = weights.reshape(num_groups, group_size, N)

    # Per-group absmax scale
    absmax = np.abs(weights_grouped).max(axis=1).astype(np.float32)
    absmax = np.maximum(absmax, 1e-7)
    scales = absmax.astype(np.float16)

    max_fp4 = 6.0  # largest E2M1 magnitude

    # Quantize to nearest FP4 code
    packed = np.zeros((K, N // 8), dtype=np.uint32)

    for g in range(num_groups):
        for k in range(group_size):
            row = g * group_size + k
            for n in range(N):
                val = float(weights[row, n])
                s = float(scales[g, n])
                normalized = val / s * max_fp4

                # Find nearest FP4 code
                best_code = 0
                best_dist = float("inf")
                for code in range(16):
                    ref = float(FP4_E2M1_TABLE[code])
                    dist = abs(normalized - ref)
                    if dist < best_dist:
                        best_dist = dist
                        best_code = code

                # Pack: 8 N-consecutive values per uint32
                word_idx = n // 8
                nibble_pos = n % 8
                packed[row, word_idx] |= np.uint32(best_code) << np.uint32(nibble_pos * 4)

    return packed, scales


def _dequant_fp4(
    packed: np.ndarray, scales: np.ndarray, K: int, N: int, group_size: int
) -> np.ndarray:
    """Dequantize packed FP4 [K, N//8] to FP16 [K, N]."""
    max_fp4 = 6.0
    result = np.zeros((K, N), dtype=np.float32)

    for row in range(K):
        group = row // group_size
        for n in range(N):
            word_idx = n // 8
            nibble_pos = n % 8
            code = (int(packed[row, word_idx]) >> (nibble_pos * 4)) & 0xF
            ref_val = float(FP4_E2M1_TABLE[code])
            s = float(scales[group, n])
            result[row, n] = ref_val * s / max_fp4

    return result.astype(np.float16)


def _reference_gemm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """FP16 GEMM with FP32 accumulation (reference)."""
    return (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)


def _get_metal_marlin() -> Any:
    """Import metal_marlin, skip if unavailable."""
    try:
        metal_marlin_path = Path(__file__).parent.parent / "python"
        sys.path.insert(0, str(metal_marlin_path))
        import metal_marlin
        return metal_marlin
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"metal_marlin not available: {e}")


def _get_mlx() -> Any:
    """Import mlx, skip if unavailable."""
    if not HAS_MLX:
        pytest.skip("MLX not available (Apple Silicon only)")
    import mlx
    import mlx.core  # noqa: F401
    return mlx


# ===========================================================================
# Tier 1: Reference GEMM with boundary dimensions (NumPy only)
# ===========================================================================


class TestReferenceGEMMBoundaries:
    """Validate reference quantize-dequant-GEMM at non-tile-aligned dimensions."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=1337)

    # M is unconstrained; N must be mult of 8; K must be mult of group_size.
    # TILE_M=16, TILE_N=16, TILE_K=32 -> test values that don't evenly divide.
    @pytest.mark.slow
    @pytest.mark.parametrize("M", [1, 7, 13, 64, 65, 127])
    @pytest.mark.parametrize("N", [64, 100, 4096, 4097])
    @pytest.mark.parametrize("K", [32, 128, 129, 4096])
    def test_gemm_non_aligned_dims(
        self, rng: np.random.Generator, M: int, N: int, K: int
    ) -> None:
        """Ensure reference GEMM handles partial tiles correctly."""
        group_size = 32
        if N % 8 != 0 or K % group_size != 0:
            pytest.skip(f"Invalid dims for packing: M={M}, N={N}, K={K}")

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = _quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = _dequant_fp4(packed, scales, K, N, group_size)

        ref = _reference_gemm(A, B_dequant)

        # Verify output shape
        assert ref.shape == (M, N)

        # Verify no NaN/Inf in output
        assert np.all(np.isfinite(ref)), (
            f"Non-finite values in reference GEMM output for ({M}, {K}, {N})"
        )

        # Cross-check with direct numpy matmul
        direct = (A @ B_dequant)
        max_diff = np.abs(ref.astype(np.float32) - direct.astype(np.float32)).max()
        # FP16 accumulation vs FP32 accumulation difference scales with sqrt(K)
        atol = float(np.sqrt(K)) * 0.01 * (float(np.abs(ref).max()) + 1e-6)
        assert max_diff < atol, (
            f"Reference/direct mismatch for ({M},{K},{N}): "
            f"max_diff={max_diff:.6f}, atol={atol:.6f}"
        )

    def test_gemm_single_element(self, rng: np.random.Generator) -> None:
        """M=1, N=1, K=128: single output with N padded to pack size."""
        M, N, K = 1, 1, 128
        group_size = 32

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        pad_n = (8 - (N % 8)) % 8
        if pad_n:
            B_fp16 = np.pad(B_fp16, ((0, 0), (0, pad_n)), mode="constant")
        N_padded = B_fp16.shape[1]

        packed, scales = _quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = _dequant_fp4(packed, scales, K, N_padded, group_size)

        ref = _reference_gemm(A, B_dequant)
        assert ref.shape == (1, N_padded)
        assert np.all(np.isfinite(ref))

        # Manual dot product check for first element
        expected_0 = np.float32(0.0)
        for k in range(K):
            expected_0 += float(A[0, k]) * float(B_dequant[k, 0])
        assert abs(float(ref[0, 0]) - np.float16(expected_0)) < 0.1

    @pytest.mark.slow
    def test_gemm_very_long_k(self, rng: np.random.Generator) -> None:
        """M=1, N=4096, K=32768: long reduction dimension stresses accumulation."""
        M, N, K = 1, 4096, 32768
        group_size = 32

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = _quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = _dequant_fp4(packed, scales, K, N, group_size)

        ref = _reference_gemm(A, B_dequant)
        assert ref.shape == (1, 4096)
        assert np.all(np.isfinite(ref))

    @pytest.mark.slow
    def test_gemm_very_wide_n(self, rng: np.random.Generator) -> None:
        """M=1, N=32768, K=4096: wide output stresses N-dimension tiling."""
        M, N, K = 1, 32768, 4096
        group_size = 32

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = _quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = _dequant_fp4(packed, scales, K, N, group_size)

        ref = _reference_gemm(A, B_dequant)
        assert ref.shape == (1, 32768)
        assert np.all(np.isfinite(ref))

    @pytest.mark.parametrize("M", [1, 3, 15, 16, 17])
    def test_gemm_m_tile_boundary(
        self, rng: np.random.Generator, M: int
    ) -> None:
        """M values at or near TILE_M=16 boundary: partial M-tile handling."""
        N, K = 64, 128
        group_size = 32

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = _quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = _dequant_fp4(packed, scales, K, N, group_size)

        ref = _reference_gemm(A, B_dequant)
        direct = (A @ B_dequant)

        assert ref.shape == (M, N)
        # Verify all rows computed correctly
        for row in range(M):
            row_ref = ref[row]
            row_direct = direct[row]
            max_diff = np.abs(
                row_ref.astype(np.float32) - row_direct.astype(np.float32)
            ).max()
            atol = float(np.sqrt(K)) * 0.01 * (float(np.abs(row_ref).max()) + 1e-6)
            assert max_diff < atol, f"Row {row} mismatch for M={M}"

    @pytest.mark.parametrize("N", [8, 16, 24, 32, 40, 48])
    def test_gemm_n_tile_boundary(
        self, rng: np.random.Generator, N: int
    ) -> None:
        """N values at multiples of 8 near TILE_N=16: partial N-tile handling."""
        M, K = 4, 64
        group_size = 32

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = _quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = _dequant_fp4(packed, scales, K, N, group_size)

        ref = _reference_gemm(A, B_dequant)
        assert ref.shape == (M, N)
        assert np.all(np.isfinite(ref))

    @pytest.mark.parametrize("K", [32, 64, 96, 128, 160, 192])
    def test_gemm_k_tile_boundary(
        self, rng: np.random.Generator, K: int
    ) -> None:
        """K values at multiples of 32 near TILE_K=32: partial K-tile handling."""
        M, N = 4, 64
        group_size = 32

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = _quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = _dequant_fp4(packed, scales, K, N, group_size)

        ref = _reference_gemm(A, B_dequant)
        assert ref.shape == (M, N)
        assert np.all(np.isfinite(ref))


# ===========================================================================
# Tier 2: Metal Kernel GEMM at boundary dimensions
# ===========================================================================


@pytest.mark.skipif(not HAS_MLX, reason="Requires MLX (Apple Silicon only)")
class TestMetalGEMMBoundaries:
    """Validate Metal Marlin kernel at non-tile-aligned dimensions.

    The kernel tiles at TILE_M=16, TILE_N=16, TILE_K=32.
    These tests use dimensions that create partial tiles to verify
    the boundary guards in the kernel work correctly.
    """

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=42)

    @pytest.fixture
    def mm(self) -> Any:
        return _get_metal_marlin()

    @pytest.fixture
    def mlx(self) -> Any:
        return _get_mlx()

    def _run_metal_gemm(
        self,
        mm: Any,
        A: np.ndarray,
        packed: np.ndarray,
        scales: np.ndarray,
        M: int,
        N: int,
        K: int,
        group_size: int,
    ) -> np.ndarray:
        """Run Metal Marlin GEMM and return result as numpy."""
        import mlx.core as mx

        A_mx = mx.array(A)
        packed_mx = mx.array(packed)
        scales_mx = mx.array(scales)

        result = mm.quantized_linear(A_mx, packed_mx, scales_mx, group_size=group_size)
        mx.eval(result)
        return np.array(result)

    # -----------------------------------------------------------------------
    # Parametrized boundary tests
    # -----------------------------------------------------------------------

    @pytest.mark.slow
    @pytest.mark.parametrize("M", [1, 7, 13, 64, 65, 127])
    @pytest.mark.parametrize("N", [64, 100, 4096, 4097])
    @pytest.mark.parametrize("K", [32, 128, 129, 4096])
    def test_gemm_non_aligned_dims(
        self, rng: np.random.Generator, mm: Any, M: int, N: int, K: int
    ) -> None:
        """Kernel handles partial tiles correctly for non-aligned M, N, K."""
        group_size = 32
        if N % 8 != 0 or K % group_size != 0:
            pytest.skip(f"Invalid dims for packing: M={M}, N={N}, K={K}")

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = _quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = _dequant_fp4(packed, scales, K, N, group_size)

        ref = _reference_gemm(A, B_dequant)
        metal = self._run_metal_gemm(mm, A, packed, scales, M, N, K, group_size)

        assert metal.shape == (M, N), f"Shape mismatch: {metal.shape} vs {(M, N)}"

        # Allow FP16 accumulation tolerance
        max_abs = max(float(np.abs(ref).max()), 1e-6)
        rtol = 0.02  # 2% relative tolerance
        atol = max(float(np.sqrt(K)) * 5e-3 * max_abs, 1e-3)

        assert np.allclose(
            metal.astype(np.float32), ref.astype(np.float32), rtol=rtol, atol=atol
        ), (
            f"Metal vs reference mismatch for ({M},{K},{N}). "
            f"Max diff: {np.abs(metal.astype(np.float32) - ref.astype(np.float32)).max():.6f}, "
            f"atol={atol:.6f}"
        )

    def test_gemm_single_element(self, rng: np.random.Generator, mm: Any) -> None:
        """M=1, N=1, K=128: single output with N padded to pack size."""
        M, N, K = 1, 1, 128
        group_size = 32

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        pad_n = (8 - (N % 8)) % 8
        if pad_n:
            B_fp16 = np.pad(B_fp16, ((0, 0), (0, pad_n)), mode="constant")
        N_padded = B_fp16.shape[1]

        packed, scales = _quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = _dequant_fp4(packed, scales, K, N_padded, group_size)

        ref = _reference_gemm(A, B_dequant)
        metal = self._run_metal_gemm(mm, A, packed, scales, M, N_padded, K, group_size)

        assert metal.shape == (1, N_padded)
        assert np.allclose(
            metal[:, :N].astype(np.float32),
            ref[:, :N].astype(np.float32),
            rtol=0.02,
            atol=1e-3,
        )

    @pytest.mark.slow
    def test_gemm_very_long_k(self, rng: np.random.Generator, mm: Any) -> None:
        """M=1, N=4096, K=32768: long reduction stresses accumulation precision."""
        M, N, K = 1, 4096, 32768
        group_size = 32

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = _quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = _dequant_fp4(packed, scales, K, N, group_size)

        ref = _reference_gemm(A, B_dequant)
        metal = self._run_metal_gemm(mm, A, packed, scales, M, N, K, group_size)

        assert metal.shape == (1, 4096)
        # Larger K means more accumulation error; relax tolerance
        max_abs = max(float(np.abs(ref).max()), 1e-6)
        atol = float(np.sqrt(K)) * 0.01 * max_abs
        assert np.allclose(
            metal.astype(np.float32), ref.astype(np.float32), rtol=0.05, atol=atol
        ), (
            f"Long-K mismatch. Max diff: "
            f"{np.abs(metal.astype(np.float32) - ref.astype(np.float32)).max():.6f}"
        )

    @pytest.mark.slow
    def test_gemm_very_wide_n(self, rng: np.random.Generator, mm: Any) -> None:
        """M=1, N=32768, K=4096: wide output stresses N-tiling coverage."""
        M, N, K = 1, 32768, 4096
        group_size = 32

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = _quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = _dequant_fp4(packed, scales, K, N, group_size)

        ref = _reference_gemm(A, B_dequant)
        metal = self._run_metal_gemm(mm, A, packed, scales, M, N, K, group_size)

        assert metal.shape == (1, 32768)
        max_abs = max(float(np.abs(ref).max()), 1e-6)
        atol = float(np.sqrt(K)) * 5e-3 * max_abs
        assert np.allclose(
            metal.astype(np.float32), ref.astype(np.float32), rtol=0.02, atol=atol
        )

    @pytest.mark.slow
    def test_gemm_large_m_non_aligned(self, rng: np.random.Generator, mm: Any) -> None:
        """M=127, N=4096, K=4096: large non-aligned M with LLM-scale N,K."""
        M, N, K = 127, 4096, 4096
        group_size = 128

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = _quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = _dequant_fp4(packed, scales, K, N, group_size)

        ref = _reference_gemm(A, B_dequant)
        metal = self._run_metal_gemm(mm, A, packed, scales, M, N, K, group_size)

        assert metal.shape == (127, 4096)
        max_abs = max(float(np.abs(ref).max()), 1e-6)
        atol = float(np.sqrt(K)) * 5e-3 * max_abs
        assert np.allclose(
            metal.astype(np.float32), ref.astype(np.float32), rtol=0.02, atol=atol
        )

    # -----------------------------------------------------------------------
    # Edge cases: zero/constant inputs
    # -----------------------------------------------------------------------

    def test_gemm_zero_activations(self, rng: np.random.Generator, mm: Any) -> None:
        """Zero activations should produce zero output regardless of weights."""
        M, N, K = 7, 24, 64
        group_size = 32

        A = np.zeros((M, K), dtype=np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = _quantize_to_fp4(B_fp16, group_size=group_size)
        metal = self._run_metal_gemm(mm, A, packed, scales, M, N, K, group_size)

        assert metal.shape == (M, N)
        assert np.allclose(metal, 0.0, atol=1e-6), (
            f"Non-zero output for zero input. Max: {np.abs(metal).max()}"
        )

    def test_gemm_constant_activations(
        self, rng: np.random.Generator, mm: Any
    ) -> None:
        """Constant activations: output should equal column sums of dequant(B)."""
        M, N, K = 3, 16, 32
        group_size = 32

        # All activations = 1.0: output[i,j] = sum_k(B_dequant[k,j])
        A = np.ones((M, K), dtype=np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = _quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = _dequant_fp4(packed, scales, K, N, group_size)

        ref = _reference_gemm(A, B_dequant)
        metal = self._run_metal_gemm(mm, A, packed, scales, M, N, K, group_size)

        assert np.allclose(
            metal.astype(np.float32), ref.astype(np.float32), rtol=0.02, atol=1e-3
        )
        # All rows should be identical
        for row in range(1, M):
            assert np.allclose(metal[row], metal[0], atol=1e-6)

    # -----------------------------------------------------------------------
    # Group size interaction with K
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_gemm_varying_group_size(
        self, rng: np.random.Generator, mm: Any, group_size: int
    ) -> None:
        """Different group sizes with K that's a multiple of each."""
        M, N = 7, 64
        K = 384  # divisible by 32, 64, 128

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = _quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = _dequant_fp4(packed, scales, K, N, group_size)

        ref = _reference_gemm(A, B_dequant)
        metal = self._run_metal_gemm(mm, A, packed, scales, M, N, K, group_size)

        max_abs = max(float(np.abs(ref).max()), 1e-6)
        atol = float(np.sqrt(K)) * 5e-3 * max_abs
        assert np.allclose(
            metal.astype(np.float32), ref.astype(np.float32), rtol=0.02, atol=atol
        ), f"Group size {group_size} mismatch"

    # -----------------------------------------------------------------------
    # Stress: many partial tiles simultaneously
    # -----------------------------------------------------------------------

    def test_gemm_all_partial_tiles(self, rng: np.random.Generator, mm: Any) -> None:
        """M=13, N=24, K=96: partial tiles in all three dimensions simultaneously."""
        M, N, K = 13, 24, 96
        group_size = 32

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = _quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = _dequant_fp4(packed, scales, K, N, group_size)

        ref = _reference_gemm(A, B_dequant)
        metal = self._run_metal_gemm(mm, A, packed, scales, M, N, K, group_size)

        assert metal.shape == (13, 24)
        max_abs = max(float(np.abs(ref).max()), 1e-6)
        atol = float(np.sqrt(K)) * 5e-3 * max_abs
        assert np.allclose(
            metal.astype(np.float32), ref.astype(np.float32), rtol=0.02, atol=atol
        )

    def test_gemm_prime_m(self, rng: np.random.Generator, mm: Any) -> None:
        """M=97 (prime): guarantees M never divides evenly into any tile size."""
        M, N, K = 97, 64, 128
        group_size = 32

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = _quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = _dequant_fp4(packed, scales, K, N, group_size)

        ref = _reference_gemm(A, B_dequant)
        metal = self._run_metal_gemm(mm, A, packed, scales, M, N, K, group_size)

        assert metal.shape == (97, 64)
        max_abs = max(float(np.abs(ref).max()), 1e-6)
        atol = float(np.sqrt(K)) * 5e-3 * max_abs
        assert np.allclose(
            metal.astype(np.float32), ref.astype(np.float32), rtol=0.02, atol=atol
        )

    # -----------------------------------------------------------------------
    # Regression: exact tile boundaries (no partial tiles, sanity check)
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (16, 16, 32),   # exactly 1 tile in each dim
            (32, 32, 64),   # exactly 2 tiles in each dim
            (16, 64, 128),  # 1 M-tile, 4 N-tiles, 4 K-tiles
            (64, 128, 256), # 4 M-tiles, 8 N-tiles, 8 K-tiles
        ],
    )
    def test_gemm_exact_tile_multiples(
        self, rng: np.random.Generator, mm: Any, M: int, N: int, K: int
    ) -> None:
        """Exact tile multiples should produce correct results (baseline)."""
        group_size = 32

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = _quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = _dequant_fp4(packed, scales, K, N, group_size)

        ref = _reference_gemm(A, B_dequant)
        metal = self._run_metal_gemm(mm, A, packed, scales, M, N, K, group_size)

        max_abs = max(float(np.abs(ref).max()), 1e-6)
        atol = float(np.sqrt(K)) * 5e-3 * max_abs
        assert np.allclose(
            metal.astype(np.float32), ref.astype(np.float32), rtol=0.02, atol=atol
        ), f"Exact-tile mismatch for ({M},{K},{N})"
