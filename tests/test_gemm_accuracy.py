"""GEMM accuracy tests: Metal Marlin FP4/INT4 kernels vs FP32 reference computation.

Validates that fused dequant-GEMM kernels produce results matching
the reference pipeline: dequantize weights to FP16, then matmul in FP32.

Test coverage:
  - FP4 GEMM across various M/N/K configurations including LLM-scale dims
  - INT4 GEMM with asymmetric quantization (zero-point)
  - Accumulation precision: verify FP32 accumulation for large K
  - Numerical stability: large/small values, cancellation, edge cases

Error budget:
  - FP4 quantization: ~10-25% relative error (only 16 representable values per sign)
  - INT4 quantization: ~5-10% relative error (16 uniform levels)
  - FP16 accumulation: O(sqrt(K) * eps) error from K reduction steps
  - The tests here compare Metal kernel output to the *dequantized* reference,
    so only accumulation error is tested, not quantization quality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from metal_marlin._compat import HAS_MLX

from .reference import dequant_fp4_reference, dequant_int4_reference, gemm_reference

# Add metal_marlin python module to path
_METAL_MARLIN_DIR = Path(__file__).parent.parent / "python"
if str(_METAL_MARLIN_DIR) not in sys.path:
    sys.path.insert(0, str(_METAL_MARLIN_DIR))

# ---------------------------------------------------------------------------
# FP4 E2M1 reference values (matches llama.cpp kvalues_mxfp4_f)
# ---------------------------------------------------------------------------
FP4_E2M1_TABLE: np.ndarray = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Quantization / dequantization helpers (pure numpy, no Metal dependency)
# ---------------------------------------------------------------------------


def quantize_to_fp4(
    weights: np.ndarray, group_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize FP16/FP32 weights to packed FP4 E2M1 with per-group absmax scaling.

    Weight layout assumed: [K, N] (input_features x output_features).
    Packing layout: uint32 packs 8 consecutive N-dimension values.

    Args:
        weights: [K, N] weight matrix (any float dtype, cast to float32 internally)
        group_size: Elements per quantization group along K dimension

    Returns:
        packed: [K, N // 8] uint32 array of packed FP4 nibbles
        scales: [K // group_size, N] float16 scale factors
    """
    K, N = weights.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    assert N % 8 == 0, f"N={N} must be divisible by 8"

    w = weights.astype(np.float32)
    num_groups = K // group_size

    # Per-group scales: max|w| / 6.0 so that the full FP4 dynamic range is used.
    # The kernel reconstructs as: FP4_LUT[code] * scale.
    # With scale = max_abs / 6.0, and w_normalized = w / scale in [-6, 6],
    # we get: FP4_LUT[code] * (max_abs / 6.0) ~= w.
    max_fp4 = 6.0  # max representable E2M1 value
    w_grouped = w.reshape(num_groups, group_size, N)
    group_max = np.abs(w_grouped).max(axis=1)  # [num_groups, N]
    scales = np.maximum(group_max / max_fp4, np.float32(1e-7)).astype(np.float16)

    scales_f32 = scales.astype(np.float32)
    scales_expanded = np.repeat(scales_f32, group_size, axis=0)  # [K, N]
    w_normalized = np.clip(w / scales_expanded, -max_fp4, max_fp4)

    # Quantize each element to nearest FP4 code via distance minimization
    # w_normalized is in [-6, 6], FP4_E2M1_TABLE covers the same range
    # Broadcast: [K, N, 1] vs [16] -> find argmin
    dists = np.abs(w_normalized[:, :, None] - FP4_E2M1_TABLE[None, None, :])  # [K, N, 16]
    codes = np.argmin(dists, axis=2).astype(np.uint32)  # [K, N]

    # Pack 8 consecutive N-dimension codes into uint32
    packed_n = N // 8
    packed = np.zeros((K, packed_n), dtype=np.uint32)
    for j in range(8):
        packed |= codes[:, j::8] << (j * 4)

    return packed, scales


def dequant_fp4(
    packed: np.ndarray,
    scales: np.ndarray,
    K: int,
    N: int,
    group_size: int = 128,
) -> np.ndarray:
    """Dequantize packed FP4 array to float32.

    Reverses the quantize_to_fp4 packing layout.

    Args:
        packed: [K, N // 8] uint32 packed FP4 codes
        scales: [K // group_size, N] float16 scales
        K, N: original weight dimensions
        group_size: quantization group size

    Returns:
        [K, N] float32 dequantized weights
    """
    scales_f32 = scales.astype(np.float32)

    result = np.zeros((K, N), dtype=np.float32)

    for j in range(8):
        # Extract nibble j from each uint32
        nibbles = (packed >> (j * 4)) & 0xF  # [K, N//8]
        # Map nibble codes to float values via LUT
        vals = FP4_E2M1_TABLE[nibbles]  # [K, N//8]
        # Scale: each row k belongs to group k // group_size
        # The kernel computes: FP4_LUT[code] * scale (no max_fp4 divisor)
        for k in range(K):
            group = k // group_size
            # Columns j, j+8, j+16, ... are packed together
            col_indices = np.arange(j, N, 8)
            result[k, col_indices] = vals[k, :] * scales_f32[group, col_indices]

    return result


def metal_gemm_fp4(
    A: np.ndarray,
    B_packed: np.ndarray,
    scales: np.ndarray,
    group_size: int = 128,
) -> np.ndarray:
    """Run Metal Marlin FP4 GEMM via MLX.

    Calls the quantized_linear() kernel from metal_marlin.py.

    Args:
        A: [M, K] float16 activations
        B_packed: [K, N//8] uint32 packed FP4 weights
        scales: [K // group_size, N] float16 scales
        group_size: quantization group size

    Returns:
        [M, N] float16 result
    """
    import mlx.core as mx
    from metal_marlin import quantized_linear

    A_mx = mx.array(A.astype(np.float16))
    B_mx = mx.array(B_packed)
    scales_mx = mx.array(scales.astype(np.float16))

    out = quantized_linear(A_mx, B_mx, scales_mx, group_size=group_size)
    mx.eval(out)
    return np.array(out, dtype=np.float16)


# ---------------------------------------------------------------------------
# FP4/INT4 quantization helpers for marlin_gemm_fp4/marlin_gemm_int4 kernels
# (packing along K: packed shape [K/8, N]).
# ---------------------------------------------------------------------------


def quantize_fp4(
    weights: np.ndarray, group_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize weights to FP4 with K-packed layout for marlin_gemm_fp4."""
    K, N = weights.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    assert K % 8 == 0, f"K={K} must be divisible by 8"

    w = weights.astype(np.float32)
    num_groups = K // group_size
    max_fp4 = 6.0
    w_grouped = w.reshape(num_groups, group_size, N)
    group_max = np.abs(w_grouped).max(axis=1)
    scales = np.maximum(group_max / max_fp4, np.float32(1e-7)).astype(np.float16)

    scales_f32 = scales.astype(np.float32)
    scales_expanded = np.repeat(scales_f32, group_size, axis=0)
    w_normalized = np.clip(w / scales_expanded, -max_fp4, max_fp4)

    dists = np.abs(w_normalized[:, :, None] - FP4_E2M1_TABLE[None, None, :])
    codes = np.argmin(dists, axis=2).astype(np.uint32)

    packed = np.zeros((K // 8, N), dtype=np.uint32)
    for i in range(8):
        packed |= codes[i::8, :] << (i * 4)

    return packed, scales


def quantize_int4(
    weights: np.ndarray, group_size: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize weights to asymmetric INT4 with K-packed layout."""
    K, N = weights.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    assert K % 8 == 0, f"K={K} must be divisible by 8"

    w = weights.astype(np.float32)
    num_groups = K // group_size
    w_grouped = w.reshape(num_groups, group_size, N)
    w_min = w_grouped.min(axis=1)
    w_max = w_grouped.max(axis=1)

    scales = np.maximum((w_max - w_min) / 15.0, np.float32(1e-7))
    zeros = np.clip(np.round(-w_min / scales), 0, 15)

    scales = scales.astype(np.float16)
    zeros = zeros.astype(np.float16)

    scales_expanded = np.repeat(scales.astype(np.float32), group_size, axis=0)
    zeros_expanded = np.repeat(zeros.astype(np.float32), group_size, axis=0)

    q = np.round(w / scales_expanded + zeros_expanded)
    q = np.clip(q, 0, 15).astype(np.uint32)

    packed = np.zeros((K // 8, N), dtype=np.uint32)
    for i in range(8):
        packed |= q[i::8, :] << (i * 4)

    return packed, scales, zeros


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# MLX-dependent test marker (shared via conftest.py)
requires_mlx = pytest.mark.skipif(not HAS_MLX, reason="Requires MLX (Apple Silicon only)")

# Alias for backward compatibility
requires_metal = requires_mlx


# ---------------------------------------------------------------------------
# Test: GEMM accuracy against FP32 reference
# ---------------------------------------------------------------------------


class TestGEMMFP4Accuracy:
    """Metal FP4 GEMM kernel vs FP32 dequant-then-matmul reference."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=42)

    @requires_mlx
    @requires_metal
    @pytest.mark.parametrize("M,K,N", [
        (1, 128, 128),
        (8, 256, 256),
        (32, 512, 512),
        (1, 4096, 4096),
        (32, 4096, 4096),
        (128, 4096, 4096),
        (1, 4096, 11008),      # Llama-2 MLP up_proj / gate_proj
        (1, 11008, 4096),      # Llama-2 MLP down_proj
        (1, 4096, 14336),      # Llama-3 MLP up_proj
        (1, 14336, 4096),      # Llama-3 MLP down_proj
    ])
    def test_gemm_fp4_accuracy(
        self, rng: np.random.Generator, M: int, K: int, N: int
    ) -> None:
        """Metal FP4 GEMM matches FP32 reference within FP16 accumulation error."""
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        # Quantize B to FP4
        group_size = min(128, K)
        B_packed, scales = quantize_to_fp4(B_fp16, group_size=group_size)

        # Reference: dequant then matmul in FP32
        B_dequant = dequant_fp4(B_packed, scales, K, N, group_size=group_size)
        ref = A.astype(np.float32) @ B_dequant.astype(np.float32)

        # Metal kernel
        result = metal_gemm_fp4(A, B_packed, scales, group_size=group_size)

        # Error model: FP16 accumulation introduces O(sqrt(K) * eps) error.
        # With K reduction steps, each adding ~eps * max_val rounding error,
        # the total error scales as sqrt(K) * eps * ||A|| * ||B||.
        # We use relaxed tolerances that account for this.
        result_f32 = result.astype(np.float32)
        abs_diff = np.abs(result_f32 - ref)
        max_diff = abs_diff.max()
        mean_diff = abs_diff.mean()

        # Adaptive tolerance based on K and value magnitudes
        ref_scale = np.abs(ref).max() + 1e-7
        rtol = 5e-2  # 5% relative tolerance
        atol = max(float(np.sqrt(K)) * 5e-3 * ref_scale, 1e-3)

        assert np.allclose(result_f32, ref, rtol=rtol, atol=atol), (
            f"GEMM ({M}x{K}x{N}) accuracy failure. "
            f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
            f"atol={atol:.6f}, rtol={rtol}"
        )

    @requires_mlx
    @requires_metal
    def test_gemm_zero_input(self, rng: np.random.Generator) -> None:
        """Zero activations must produce zero output regardless of weights."""
        M, K, N = 16, 256, 256
        A = np.zeros((M, K), dtype=np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)
        B_packed, scales = quantize_to_fp4(B_fp16, group_size=128)

        result = metal_gemm_fp4(A, B_packed, scales, group_size=128)

        assert np.allclose(result, 0.0, atol=1e-6), (
            f"Zero input produced non-zero output. "
            f"max|result|={np.abs(result).max():.8f}"
        )

    @requires_mlx
    @requires_metal
    def test_gemm_identity_weight(self, rng: np.random.Generator) -> None:
        """Identity-ish quantized weight should approximately preserve input.

        FP4 quantization of an identity matrix loses precision because most
        values are zero (which FP4 represents exactly) but the diagonal
        values (1.0) get quantized with group-level scales. We verify the
        output is highly correlated with the input.
        """
        K = 128  # Must be divisible by 8
        A = rng.standard_normal((1, K)).astype(np.float16)
        B_identity = np.eye(K, dtype=np.float16)

        # Quantize identity. group_size = K means one scale per column,
        # and the identity column has max abs = 1.0.
        B_packed, scales = quantize_to_fp4(B_identity, group_size=K)

        # Reference: dequant then matmul
        B_dequant = dequant_fp4(B_packed, scales, K, K, group_size=K)
        ref = A.astype(np.float32) @ B_dequant.astype(np.float32)

        result = metal_gemm_fp4(A, B_packed, scales, group_size=K)

        # The identity won't be perfectly preserved due to quantization,
        # but the Metal result should match the reference (same dequant)
        result_f32 = result.astype(np.float32)
        assert np.allclose(result_f32, ref, rtol=5e-2, atol=1e-3), (
            f"Identity weight: Metal vs reference max diff = "
            f"{np.abs(result_f32 - ref).max():.6f}"
        )

        # Also verify correlation with original input is high
        corr = np.corrcoef(A.flatten().astype(np.float32),
                           result.flatten().astype(np.float32))[0, 1]
        assert corr > 0.85, (
            f"Identity weight correlation too low: {corr:.4f}"
        )

    @requires_mlx
    @requires_metal
    def test_gemm_zero_weight(self) -> None:
        """All-zero weights must produce zero output regardless of input."""
        M, K, N = 8, 128, 128
        A = np.ones((M, K), dtype=np.float16)
        B_zero = np.zeros((K, N), dtype=np.float16)
        B_packed, scales = quantize_to_fp4(B_zero, group_size=128)

        result = metal_gemm_fp4(A, B_packed, scales, group_size=128)

        assert np.allclose(result, 0.0, atol=1e-6), (
            f"Zero weights produced non-zero output. "
            f"max|result|={np.abs(result).max():.8f}"
        )

    @requires_mlx
    @requires_metal
    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_gemm_group_sizes(
        self, rng: np.random.Generator, group_size: int
    ) -> None:
        """GEMM accuracy holds across different quantization group sizes."""
        M, K, N = 8, 512, 512
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        B_packed, scales = quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = dequant_fp4(B_packed, scales, K, N, group_size=group_size)
        ref = A.astype(np.float32) @ B_dequant.astype(np.float32)

        result = metal_gemm_fp4(A, B_packed, scales, group_size=group_size)
        result_f32 = result.astype(np.float32)

        ref_scale = np.abs(ref).max() + 1e-7
        atol = max(float(np.sqrt(K)) * 5e-3 * ref_scale, 1e-3)

        assert np.allclose(result_f32, ref, rtol=5e-2, atol=atol), (
            f"group_size={group_size}: max diff = "
            f"{np.abs(result_f32 - ref).max():.6f}, atol={atol:.6f}"
        )

    @requires_mlx
    @requires_metal
    def test_gemm_single_token_large_model(
        self, rng: np.random.Generator
    ) -> None:
        """Single-token inference at LLM-scale dimensions (M=1, K=N=4096)."""
        M, K, N = 1, 4096, 4096
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        B_packed, scales = quantize_to_fp4(B_fp16, group_size=128)
        B_dequant = dequant_fp4(B_packed, scales, K, N, group_size=128)
        ref = A.astype(np.float32) @ B_dequant.astype(np.float32)

        result = metal_gemm_fp4(A, B_packed, scales, group_size=128)
        result_f32 = result.astype(np.float32)

        ref_scale = np.abs(ref).max() + 1e-7
        atol = max(float(np.sqrt(K)) * 5e-3 * ref_scale, 1e-3)

        assert np.allclose(result_f32, ref, rtol=5e-2, atol=atol), (
            f"Single-token LLM GEMM: max diff = "
            f"{np.abs(result_f32 - ref).max():.6f}, atol={atol:.6f}"
        )

    @requires_mlx
    @requires_metal
    def test_gemm_uniform_weight(self, rng: np.random.Generator) -> None:
        """Uniform weight (all same value) reduces to scaled row-sum."""
        M, K, N = 4, 128, 128
        A = rng.standard_normal((M, K)).astype(np.float16)
        # Uniform weight: all elements = 2.0 (representable in FP4)
        B_uniform = np.full((K, N), 2.0, dtype=np.float16)

        B_packed, scales = quantize_to_fp4(B_uniform, group_size=128)
        B_dequant = dequant_fp4(B_packed, scales, K, N, group_size=128)
        ref = A.astype(np.float32) @ B_dequant.astype(np.float32)

        result = metal_gemm_fp4(A, B_packed, scales, group_size=128)
        result_f32 = result.astype(np.float32)

        ref_scale = np.abs(ref).max() + 1e-7
        atol = max(float(np.sqrt(K)) * 5e-3 * ref_scale, 1e-3)

        assert np.allclose(result_f32, ref, rtol=5e-2, atol=atol), (
            f"Uniform weight: max diff = "
            f"{np.abs(result_f32 - ref).max():.6f}"
        )

    @requires_mlx
    @requires_metal
    def test_gemm_negative_weights(self, rng: np.random.Generator) -> None:
        """Negative weights (all negative FP4 codes) are handled correctly."""
        M, K, N = 4, 128, 128
        A = rng.standard_normal((M, K)).astype(np.float16)
        # All-negative weight: -3.0 is representable in FP4 (code 13)
        B_neg = np.full((K, N), -3.0, dtype=np.float16)

        B_packed, scales = quantize_to_fp4(B_neg, group_size=128)
        B_dequant = dequant_fp4(B_packed, scales, K, N, group_size=128)
        ref = A.astype(np.float32) @ B_dequant.astype(np.float32)

        result = metal_gemm_fp4(A, B_packed, scales, group_size=128)
        result_f32 = result.astype(np.float32)

        ref_scale = np.abs(ref).max() + 1e-7
        atol = max(float(np.sqrt(K)) * 5e-3 * ref_scale, 1e-3)

        assert np.allclose(result_f32, ref, rtol=5e-2, atol=atol), (
            f"Negative weights: max diff = "
            f"{np.abs(result_f32 - ref).max():.6f}"
        )

    @requires_mlx
    @requires_metal
    @pytest.mark.parametrize("M", [1, 2, 7, 15, 33, 65])
    def test_gemm_non_tile_aligned_M(
        self, rng: np.random.Generator, M: int
    ) -> None:
        """Non-tile-aligned M dimensions are handled correctly (boundary check)."""
        K, N = 128, 128
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        B_packed, scales = quantize_to_fp4(B_fp16, group_size=128)
        B_dequant = dequant_fp4(B_packed, scales, K, N, group_size=128)
        ref = A.astype(np.float32) @ B_dequant.astype(np.float32)

        result = metal_gemm_fp4(A, B_packed, scales, group_size=128)
        result_f32 = result.astype(np.float32)

        ref_scale = np.abs(ref).max() + 1e-7
        atol = max(float(np.sqrt(K)) * 5e-3 * ref_scale, 1e-3)

        assert np.allclose(result_f32, ref, rtol=5e-2, atol=atol), (
            f"M={M}: max diff = {np.abs(result_f32 - ref).max():.6f}"
        )

    @requires_mlx
    @requires_metal
    @pytest.mark.parametrize("N", [16, 24, 48, 72, 128, 256])
    def test_gemm_various_N(
        self, rng: np.random.Generator, N: int
    ) -> None:
        """Various N dimensions (must be multiple of 8 for FP4 packing)."""
        M, K = 8, 256
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        B_packed, scales = quantize_to_fp4(B_fp16, group_size=128)
        B_dequant = dequant_fp4(B_packed, scales, K, N, group_size=128)
        ref = A.astype(np.float32) @ B_dequant.astype(np.float32)

        result = metal_gemm_fp4(A, B_packed, scales, group_size=128)
        result_f32 = result.astype(np.float32)

        ref_scale = np.abs(ref).max() + 1e-7
        atol = max(float(np.sqrt(K)) * 5e-3 * ref_scale, 1e-3)

        assert np.allclose(result_f32, ref, rtol=5e-2, atol=atol), (
            f"N={N}: max diff = {np.abs(result_f32 - ref).max():.6f}"
        )

    @requires_mlx
    @requires_metal
    def test_gemm_determinism(self, rng: np.random.Generator) -> None:
        """Multiple runs of the same GEMM produce identical results."""
        M, K, N = 16, 256, 256
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)
        B_packed, scales = quantize_to_fp4(B_fp16, group_size=128)

        results = []
        for _ in range(5):
            r = metal_gemm_fp4(A, B_packed, scales, group_size=128)
            results.append(r.copy())

        for i in range(1, len(results)):
            assert np.array_equal(results[0], results[i]), (
                f"Run 0 vs run {i}: max diff = "
                f"{np.abs(results[0].astype(np.float32) - results[i].astype(np.float32)).max()}"
            )

    @requires_mlx
    @requires_metal
    def test_gemm_scale_sensitivity(self, rng: np.random.Generator) -> None:
        """Scaling activations by constant C scales output by C."""
        M, K, N = 8, 256, 256
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)
        B_packed, scales = quantize_to_fp4(B_fp16, group_size=128)

        result_1x = metal_gemm_fp4(A, B_packed, scales, group_size=128)
        result_2x = metal_gemm_fp4(
            (A * np.float16(2.0)), B_packed, scales, group_size=128
        )

        # 2x input should give 2x output (within FP16 precision)
        expected_2x = result_1x.astype(np.float32) * 2.0
        actual_2x = result_2x.astype(np.float32)

        ref_scale = np.abs(expected_2x).max() + 1e-7
        atol = max(float(np.sqrt(K)) * 1e-2 * ref_scale, 1e-2)

        assert np.allclose(actual_2x, expected_2x, rtol=5e-2, atol=atol), (
            f"Scale linearity: max diff = "
            f"{np.abs(actual_2x - expected_2x).max():.6f}"
        )


# ---------------------------------------------------------------------------
# Test: Quantization quality (not Metal-specific, validates the quant pipeline)
# ---------------------------------------------------------------------------


class TestQuantizationQuality:
    """Validate FP4 quantize→dequant roundtrip quality."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=123)

    @pytest.mark.parametrize("K,N", [
        (128, 128),
        (256, 256),
        (4096, 4096),
    ])
    def test_roundtrip_error_bounded(
        self, rng: np.random.Generator, K: int, N: int
    ) -> None:
        """Quantize→dequant relative error is bounded (median < 50%)."""
        B = rng.standard_normal((K, N)).astype(np.float16)
        group_size = min(128, K)

        packed, scales = quantize_to_fp4(B, group_size=group_size)
        B_recovered = dequant_fp4(packed, scales, K, N, group_size=group_size)

        # Skip near-zero values where relative error is undefined
        mask = np.abs(B.astype(np.float32)) > 0.01
        if not mask.any():
            return

        rel_error = (
            np.abs(B_recovered[mask] - B.astype(np.float32)[mask])
            / (np.abs(B.astype(np.float32)[mask]) + 1e-7)
        )
        median_err = np.median(rel_error)
        assert median_err < 0.5, (
            f"Median relative quantization error too high: {median_err:.4f}"
        )

    def test_exact_representable_values(self) -> None:
        """Values exactly representable in FP4 should quantize losslessly."""
        K, N = 128, 8
        # Fill with values that are exactly in the FP4 codebook
        exact_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        B = np.zeros((K, N), dtype=np.float16)
        for i in range(K):
            B[i, :] = exact_vals[i % len(exact_vals)]

        packed, scales = quantize_to_fp4(B, group_size=K)
        B_recovered = dequant_fp4(packed, scales, K, N, group_size=K)

        # After normalization by scale and denormalization, exact values
        # should be recovered within FP16 precision
        non_zero = B.astype(np.float32) != 0.0
        if non_zero.any():
            rel_err = np.abs(
                B_recovered[non_zero] - B.astype(np.float32)[non_zero]
            ) / np.abs(B.astype(np.float32)[non_zero])
            assert rel_err.max() < 0.01, (
                f"Exact FP4 values not recovered: max rel error = {rel_err.max():.6f}"
            )

    def test_zero_weights_exact(self) -> None:
        """Zero weights are exactly preserved through quantization."""
        K, N = 128, 64
        B = np.zeros((K, N), dtype=np.float16)
        packed, scales = quantize_to_fp4(B, group_size=128)
        B_recovered = dequant_fp4(packed, scales, K, N, group_size=128)
        assert np.allclose(B_recovered, 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# Tests: marlin_gemm_fp4 / marlin_gemm_int4 vs reference
# ---------------------------------------------------------------------------


class TestGEMMAccuracy:
    @requires_mlx
    @requires_metal
    @pytest.mark.parametrize("M,N,K", [
        (1, 4096, 4096),
        (32, 4096, 4096),
        (1, 4096, 11008),
        (1, 11008, 4096),
        (1, 4096, 14336),
    ])
    def test_fp4_gemm_vs_reference(self, M: int, N: int, K: int) -> None:
        """Compare Metal FP4 GEMM against reference."""
        import mlx.core as mx

        rng = np.random.default_rng(seed=42)
        W_fp16 = (rng.standard_normal((K, N)).astype(np.float16) * 0.1)
        packed, scales = quantize_fp4(W_fp16, group_size=128)

        A = rng.standard_normal((M, K)).astype(np.float16)
        W_dequant = dequant_fp4_reference(packed, scales, group_size=128)
        ref = gemm_reference(A, W_dequant)

        try:
            from kernels import marlin_gemm_fp4
        except ImportError:
            from metal_marlin import marlin_gemm_fp4

        result = marlin_gemm_fp4(
            mx.array(A),
            mx.array(packed),
            mx.array(scales),
            group_size=128,
        )
        mx.eval(result)
        result_np = np.array(result)

        np.testing.assert_allclose(result_np, ref, rtol=0.05, atol=0.01)

    @requires_mlx
    @requires_metal
    @pytest.mark.parametrize("M,N,K", [
        (1, 1024, 4096),
        (16, 1024, 4096),
    ])
    def test_int4_gemm_vs_reference(self, M: int, N: int, K: int) -> None:
        """Compare Metal INT4 GEMM against reference."""
        import mlx.core as mx

        rng = np.random.default_rng(seed=123)
        W_fp16 = (rng.standard_normal((K, N)).astype(np.float16) * 0.5)
        packed, scales, zeros = quantize_int4(W_fp16, group_size=128)

        A = rng.standard_normal((M, K)).astype(np.float16)
        W_dequant = dequant_int4_reference(packed, scales, zeros, group_size=128)
        ref = gemm_reference(A, W_dequant)

        try:
            from kernels import marlin_gemm_int4
        except ImportError:
            from metal_marlin import marlin_gemm_int4

        result = marlin_gemm_int4(
            mx.array(A),
            mx.array(packed),
            mx.array(scales),
            mx.array(zeros),
            group_size=128,
        )
        mx.eval(result)
        result_np = np.array(result)

        np.testing.assert_allclose(result_np, ref, rtol=0.05, atol=0.02)

    @requires_mlx
    @requires_metal
    def test_accumulation_precision(self) -> None:
        """Verify FP32 accumulation (not FP16) on large K."""
        import mlx.core as mx

        M, N, K = 1, 128, 32768
        A = np.ones((M, K), dtype=np.float16)
        W_fp16 = np.ones((K, N), dtype=np.float16)

        packed, scales = quantize_fp4(W_fp16, group_size=128)

        try:
            from kernels import marlin_gemm_fp4
        except ImportError:
            from metal_marlin import marlin_gemm_fp4

        result = marlin_gemm_fp4(
            mx.array(A),
            mx.array(packed),
            mx.array(scales),
            group_size=128,
        )
        mx.eval(result)

        expected = float(K)
        actual = float(np.array(result)[0, 0])
        assert abs(actual - expected) < expected * 0.1


class TestNumericalStability:
    @requires_mlx
    @requires_metal
    def test_large_values(self) -> None:
        """Test with weights/activations near FP16 max."""
        import mlx.core as mx

        rng = np.random.default_rng(seed=7)
        M, N, K = 4, 128, 256
        A = (rng.standard_normal((M, K)) * 6.0e4).astype(np.float16)
        W_fp16 = (rng.standard_normal((K, N)) * 6.0e4).astype(np.float16)

        packed, scales = quantize_fp4(W_fp16, group_size=128)
        W_dequant = dequant_fp4_reference(packed, scales, group_size=128)
        ref = gemm_reference(A, W_dequant)

        try:
            from kernels import marlin_gemm_fp4
        except ImportError:
            from metal_marlin import marlin_gemm_fp4

        result = marlin_gemm_fp4(
            mx.array(A),
            mx.array(packed),
            mx.array(scales),
            group_size=128,
        )
        mx.eval(result)
        result_np = np.array(result)

        assert not np.isnan(result_np).any()
        np.testing.assert_allclose(result_np, ref, rtol=0.1, atol=1.0)

    @requires_mlx
    @requires_metal
    def test_small_values(self) -> None:
        """Test with weights/activations near FP16 min."""
        import mlx.core as mx

        rng = np.random.default_rng(seed=11)
        M, N, K = 4, 128, 256
        A = (rng.standard_normal((M, K)) * 1.0e-5).astype(np.float16)
        W_fp16 = (rng.standard_normal((K, N)) * 1.0e-5).astype(np.float16)

        packed, scales = quantize_fp4(W_fp16, group_size=128)
        W_dequant = dequant_fp4_reference(packed, scales, group_size=128)
        ref = gemm_reference(A, W_dequant)

        try:
            from kernels import marlin_gemm_fp4
        except ImportError:
            from metal_marlin import marlin_gemm_fp4

        result = marlin_gemm_fp4(
            mx.array(A),
            mx.array(packed),
            mx.array(scales),
            group_size=128,
        )
        mx.eval(result)
        result_np = np.array(result)

        np.testing.assert_allclose(result_np, ref, rtol=0.2, atol=1e-5)

    @requires_mlx
    @requires_metal
    def test_mixed_signs(self) -> None:
        """Test cancellation with mixed positive/negative."""
        import mlx.core as mx

        rng = np.random.default_rng(seed=19)
        M, N, K = 8, 256, 512
        A = rng.standard_normal((M, K)).astype(np.float16)
        W_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_fp4(W_fp16, group_size=128)
        W_dequant = dequant_fp4_reference(packed, scales, group_size=128)
        ref = gemm_reference(A, W_dequant)

        try:
            from kernels import marlin_gemm_fp4
        except ImportError:
            from metal_marlin import marlin_gemm_fp4

        result = marlin_gemm_fp4(
            mx.array(A),
            mx.array(packed),
            mx.array(scales),
            group_size=128,
        )
        mx.eval(result)
        result_np = np.array(result)

        np.testing.assert_allclose(result_np, ref, rtol=0.05, atol=0.02)
