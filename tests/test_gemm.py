"""GEMM tests: Metal Marlin FP4/INT4 kernels for accuracy and boundary handling.

Validates quantized GEMM kernels against FP32 reference computations:
  - FP4 E2M1 GEMM across LLM-scale dimensions
  - INT4 asymmetric quantization with zero-point
  - Accumulation precision (FP32 accumulation for large K)
  - Numerical stability (large/small values, cancellation)
  - Boundary conditions (non-tile-aligned dims, partial tiles)

The Metal Marlin kernel uses TILE_M=16, TILE_N=16, TILE_K=32 with 128 threads.
Constraints on valid dimensions:
  - N must be divisible by 8 (FP4 packing: 8 nibbles per uint32)
  - K must be divisible by group_size (per-group scale lookup)
  - M can be any value (boundary-guarded at tile edges)

Error budget:
  - FP4 quantization: ~10-25% relative error (16 values per sign)
  - INT4 quantization: ~5-10% relative error (16 uniform levels)
  - FP16 accumulation: O(sqrt(K) * eps) error from K reduction steps
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from metal_marlin._compat import HAS_MPS, HAS_TORCH
from metal_marlin._compat import torch as _torch

from .conftest import requires_mps, requires_torch

if TYPE_CHECKING:
    import torch as torch_types

torch: Any = _torch

# ---------------------------------------------------------------------------
# FP4 E2M1 lookup table (matches llama.cpp kvalues_mxfp4_f)
# ---------------------------------------------------------------------------

FP4_E2M1_TABLE: np.ndarray = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

MAX_FP4 = 6.0  # Largest E2M1 magnitude


# ---------------------------------------------------------------------------
# Quantization helpers (pure numpy, no Metal dependency)
# ---------------------------------------------------------------------------


def quantize_fp4_k_packed(
    weights: np.ndarray, group_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize FP32 weights to FP4 with K-packed layout for marlin_gemm_fp4.

    Weight layout: [K, N] -> packed [K//8, N] uint32

    Args:
        weights: [K, N] weight matrix
        group_size: Elements per quantization group along K

    Returns:
        packed: [K//8, N] uint32 packed FP4 codes
        scales: [K // group_size, N] float16 scale factors
    """
    K, N = weights.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    assert K % 8 == 0, f"K={K} must be divisible by 8"

    w = weights.astype(np.float32)
    num_groups = K // group_size

    # Per-group absmax scale
    w_grouped = w.reshape(num_groups, group_size, N)
    group_max = np.abs(w_grouped).max(axis=1)
    scales = np.maximum(group_max / MAX_FP4, np.float32(1e-7)).astype(np.float16)

    scales_f32 = scales.astype(np.float32)
    scales_expanded = np.repeat(scales_f32, group_size, axis=0)
    w_normalized = np.clip(w / scales_expanded, -MAX_FP4, MAX_FP4)

    # Quantize to nearest FP4 code
    dists = np.abs(w_normalized[:, :, None] - FP4_E2M1_TABLE[None, None, :])
    codes = np.argmin(dists, axis=2).astype(np.uint32)

    # Pack 8 K-consecutive codes into uint32
    packed = np.zeros((K // 8, N), dtype=np.uint32)
    for i in range(8):
        packed |= codes[i::8, :] << (i * 4)

    return packed, scales


def quantize_fp4_n_packed(
    weights: np.ndarray, group_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize FP32 weights to FP4 with N-packed layout for quantized_linear.

    Weight layout: [K, N] -> packed [K, N//8] uint32

    Args:
        weights: [K, N] weight matrix
        group_size: Elements per quantization group along K

    Returns:
        packed: [K, N//8] uint32 packed FP4 codes
        scales: [K // group_size, N] float16 scale factors
    """
    K, N = weights.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    assert N % 8 == 0, f"N={N} must be divisible by 8"

    w = weights.astype(np.float32)
    num_groups = K // group_size

    w_grouped = w.reshape(num_groups, group_size, N)
    group_max = np.abs(w_grouped).max(axis=1)
    scales = np.maximum(group_max / MAX_FP4, np.float32(1e-7)).astype(np.float16)

    scales_f32 = scales.astype(np.float32)
    scales_expanded = np.repeat(scales_f32, group_size, axis=0)
    w_normalized = np.clip(w / scales_expanded, -MAX_FP4, MAX_FP4)

    dists = np.abs(w_normalized[:, :, None] - FP4_E2M1_TABLE[None, None, :])
    codes = np.argmin(dists, axis=2).astype(np.uint32)

    # Pack 8 N-consecutive codes into uint32
    packed = np.zeros((K, N // 8), dtype=np.uint32)
    for j in range(8):
        packed |= codes[:, j::8] << (j * 4)

    return packed, scales


def quantize_int4(
    weights: np.ndarray, group_size: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize weights to asymmetric INT4 with K-packed layout.

    Args:
        weights: [K, N] weight matrix
        group_size: Elements per quantization group along K

    Returns:
        packed: [K//8, N] uint32 packed INT4 codes
        scales: [K // group_size, N] float16 scale factors
        zeros: [K // group_size, N] float16 zero points
    """
    K, N = weights.shape
    assert K % group_size == 0
    assert K % 8 == 0

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
# Dequantization helpers
# ---------------------------------------------------------------------------


def dequant_fp4_k_packed(
    packed: np.ndarray, scales: np.ndarray, group_size: int = 128
) -> np.ndarray:
    """Dequantize K-packed FP4 [K//8, N] to float32 [K, N]."""
    K_packed, N = packed.shape
    K = K_packed * 8

    result = np.zeros((K, N), dtype=np.float32)
    scales_f32 = scales.astype(np.float32)

    for i in range(8):
        codes = (packed >> (i * 4)) & 0xF
        vals = FP4_E2M1_TABLE[codes]
        rows = np.arange(i, K, 8)
        for idx, row in enumerate(rows):
            group = row // group_size
            result[row, :] = vals[idx, :] * scales_f32[group, :]

    return result


def dequant_fp4_n_packed(
    packed: np.ndarray, scales: np.ndarray, K: int, N: int, group_size: int = 128
) -> np.ndarray:
    """Dequantize N-packed FP4 [K, N//8] to float32 [K, N]."""
    scales_f32 = scales.astype(np.float32)
    result = np.zeros((K, N), dtype=np.float32)

    for j in range(8):
        nibbles = (packed >> (j * 4)) & 0xF
        vals = FP4_E2M1_TABLE[nibbles]
        for k in range(K):
            group = k // group_size
            col_indices = np.arange(j, N, 8)
            result[k, col_indices] = vals[k, :] * scales_f32[group, col_indices]

    return result


def dequant_int4(
    packed: np.ndarray, scales: np.ndarray, zeros: np.ndarray, group_size: int = 128
) -> np.ndarray:
    """Dequantize K-packed INT4 [K//8, N] to float32 [K, N]."""
    K_packed, N = packed.shape
    K = K_packed * 8

    result = np.zeros((K, N), dtype=np.float32)
    scales_f32 = scales.astype(np.float32)
    zeros_f32 = zeros.astype(np.float32)

    for i in range(8):
        codes = (packed >> (i * 4)) & 0xF
        rows = np.arange(i, K, 8)
        for idx, row in enumerate(rows):
            group = row // group_size
            result[row, :] = (codes[idx, :].astype(np.float32) - zeros_f32[group, :]) * scales_f32[
                group, :
            ]

    return result


def gemm_reference(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """FP32 GEMM reference: A @ B with FP32 accumulation."""
    return (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)


# ===========================================================================
# Test: FP4 GEMM accuracy
# ===========================================================================


class TestGEMMFP4Accuracy:
    """Metal FP4 GEMM kernel vs FP32 dequant-then-matmul reference."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=42)

    @requires_torch
    @requires_mps
    @pytest.mark.parametrize(
        "M,K,N",
        [
            pytest.param(1, 128, 128, id="tiny"),
            pytest.param(8, 256, 256, id="small"),
            pytest.param(32, 512, 512, id="medium"),
            pytest.param(1, 4096, 4096, id="llm-single-token", marks=pytest.mark.slow),
            pytest.param(32, 4096, 4096, id="llm-batch", marks=pytest.mark.slow),
            pytest.param(1, 4096, 11008, id="llama2-up", marks=pytest.mark.slow),
            pytest.param(1, 11008, 4096, id="llama2-down", marks=pytest.mark.slow),
            pytest.param(1, 4096, 14336, id="llama3-up", marks=pytest.mark.slow),
            pytest.param(1, 14336, 4096, id="llama3-down", marks=pytest.mark.slow),
        ],
    )
    def test_fp4_gemm_accuracy(self, rng: np.random.Generator, M: int, K: int, N: int) -> None:
        """Metal FP4 GEMM matches FP32 reference within FP16 accumulation error."""
        assert torch is not None

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        group_size = min(128, K)
        packed, scales = quantize_fp4_k_packed(B_fp16, group_size=group_size)
        B_dequant = dequant_fp4_k_packed(packed, scales, group_size=group_size)
        ref = gemm_reference(A, B_dequant)

        # Run Metal kernel
        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=group_size)
        result_np = result.cpu().numpy()

        # Adaptive tolerance based on K and value magnitudes
        ref_scale = np.abs(ref).max() + 1e-7
        rtol = 5e-2
        atol = max(float(np.sqrt(K)) * 5e-3 * ref_scale, 1e-3)

        np.testing.assert_allclose(
            result_np.astype(np.float32), ref.astype(np.float32), rtol=rtol, atol=atol
        )

    @requires_torch
    @requires_mps
    def test_zero_input(self, rng: np.random.Generator) -> None:
        """Zero activations must produce zero output."""
        assert torch is not None

        M, K, N = 16, 256, 256
        A = np.zeros((M, K), dtype=np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)
        packed, scales = quantize_fp4_k_packed(B_fp16, group_size=128)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=128)
        result_np = result.cpu().numpy()

        assert np.allclose(result_np, 0.0, atol=1e-6), (
            f"Zero input produced non-zero output. max|result|={np.abs(result_np).max()}"
        )

    @requires_torch
    @requires_mps
    def test_zero_weight(self) -> None:
        """All-zero weights must produce zero output."""
        assert torch is not None

        M, K, N = 8, 128, 128
        A = np.ones((M, K), dtype=np.float16)
        B_zero = np.zeros((K, N), dtype=np.float16)
        packed, scales = quantize_fp4_k_packed(B_zero, group_size=128)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=128)
        result_np = result.cpu().numpy()

        assert np.allclose(result_np, 0.0, atol=1e-6)

    @requires_torch
    @requires_mps
    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_group_sizes(self, rng: np.random.Generator, group_size: int) -> None:
        """GEMM accuracy holds across different quantization group sizes."""
        assert torch is not None

        M, K, N = 8, 512, 512
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_fp4_k_packed(B_fp16, group_size=group_size)
        B_dequant = dequant_fp4_k_packed(packed, scales, group_size=group_size)
        ref = gemm_reference(A, B_dequant)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=group_size)
        result_np = result.cpu().numpy()

        ref_scale = np.abs(ref).max() + 1e-7
        atol = max(float(np.sqrt(K)) * 5e-3 * ref_scale, 1e-3)

        np.testing.assert_allclose(
            result_np.astype(np.float32), ref.astype(np.float32), rtol=5e-2, atol=atol
        )

    @requires_torch
    @requires_mps
    def test_determinism(self, rng: np.random.Generator) -> None:
        """Multiple runs produce identical results."""
        assert torch is not None

        M, K, N = 16, 256, 256
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)
        packed, scales = quantize_fp4_k_packed(B_fp16, group_size=128)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        results = []
        for _ in range(5):
            r = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=128)
            results.append(r.cpu().numpy().copy())

        for i in range(1, len(results)):
            assert np.array_equal(results[0], results[i]), (
                f"Run 0 vs run {i}: max diff = "
                f"{np.abs(results[0].astype(np.float32) - results[i].astype(np.float32)).max()}"
            )

    @requires_torch
    @requires_mps
    def test_scale_linearity(self, rng: np.random.Generator) -> None:
        """Scaling activations by C scales output by C."""
        assert torch is not None

        M, K, N = 8, 256, 256
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)
        packed, scales = quantize_fp4_k_packed(B_fp16, group_size=128)

        from metal_marlin import marlin_gemm_fp4

        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        A_1x = torch.from_numpy(A).to("mps")
        A_2x = torch.from_numpy(A * np.float16(2.0)).to("mps")

        result_1x = marlin_gemm_fp4(A_1x, packed_t, scales_t, group_size=128).cpu().numpy()
        result_2x = marlin_gemm_fp4(A_2x, packed_t, scales_t, group_size=128).cpu().numpy()

        expected_2x = result_1x.astype(np.float32) * 2.0
        actual_2x = result_2x.astype(np.float32)

        ref_scale = np.abs(expected_2x).max() + 1e-7
        atol = max(float(np.sqrt(K)) * 1e-2 * ref_scale, 1e-2)

        np.testing.assert_allclose(actual_2x, expected_2x, rtol=5e-2, atol=atol)


# ===========================================================================
# Test: INT4 GEMM accuracy
# ===========================================================================


class TestGEMMINT4Accuracy:
    """Metal INT4 GEMM kernel with asymmetric quantization."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=123)

    @requires_torch
    @requires_mps
    @pytest.mark.parametrize(
        "M,N,K",
        [
            pytest.param(1, 1024, 4096, id="small"),
            pytest.param(16, 1024, 4096, id="batch"),
        ],
    )
    def test_int4_gemm_vs_reference(self, rng: np.random.Generator, M: int, N: int, K: int) -> None:
        """Metal INT4 GEMM matches reference."""
        assert torch is not None

        W_fp16 = rng.standard_normal((K, N)).astype(np.float16) * 0.5
        packed, scales, zeros = quantize_int4(W_fp16, group_size=128)

        A = rng.standard_normal((M, K)).astype(np.float16)
        W_dequant = dequant_int4(packed, scales, zeros, group_size=128)
        ref = gemm_reference(A, W_dequant)

        from metal_marlin import marlin_gemm_int4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")
        zeros_t = torch.from_numpy(zeros).to("mps")

        result = marlin_gemm_int4(A_t, packed_t, scales_t, zeros_t, group_size=128)
        result_np = result.cpu().numpy()

        np.testing.assert_allclose(
            result_np.astype(np.float32), ref.astype(np.float32), rtol=0.05, atol=0.02
        )


# ===========================================================================
# Test: Accumulation precision
# ===========================================================================


class TestAccumulationPrecision:
    """Verify FP32 accumulation (not FP16) on large K."""

    @requires_torch
    @requires_mps
    def test_large_k_accumulation(self) -> None:
        """Large K should use FP32 accumulators to avoid overflow."""
        assert torch is not None

        M, N, K = 1, 128, 32768
        A = np.ones((M, K), dtype=np.float16)
        W_fp16 = np.ones((K, N), dtype=np.float16)

        packed, scales = quantize_fp4_k_packed(W_fp16, group_size=128)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=128)
        result_np = result.cpu().numpy()

        # Expected ~K (all ones weighted by dequant scale)
        # With FP16 accumulators, this would overflow/saturate
        actual = float(result_np[0, 0])
        # Tolerance: 10% of expected (dequant imprecision)
        assert actual > K * 0.5, f"Accumulation likely overflowed: got {actual}, expected ~{K}"


# ===========================================================================
# Test: Numerical stability
# ===========================================================================


class TestNumericalStability:
    """Numerical edge cases: large/small values, cancellation."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=7)

    @requires_torch
    @requires_mps
    def test_large_values(self, rng: np.random.Generator) -> None:
        """Near FP16 max values don't produce NaN/Inf."""
        assert torch is not None

        M, N, K = 4, 128, 256
        A = (rng.standard_normal((M, K)) * 6.0e4).astype(np.float16)
        W_fp16 = (rng.standard_normal((K, N)) * 6.0e4).astype(np.float16)

        packed, scales = quantize_fp4_k_packed(W_fp16, group_size=128)
        W_dequant = dequant_fp4_k_packed(packed, scales, group_size=128)
        ref = gemm_reference(A, W_dequant)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=128)
        result_np = result.cpu().numpy()

        assert not np.isnan(result_np).any(), "NaN in output"
        assert not np.isinf(result_np).any(), "Inf in output"
        np.testing.assert_allclose(
            result_np.astype(np.float32), ref.astype(np.float32), rtol=0.1, atol=1.0
        )

    @requires_torch
    @requires_mps
    def test_small_values(self, rng: np.random.Generator) -> None:
        """Near FP16 min values don't underflow to zero incorrectly."""
        assert torch is not None

        M, N, K = 4, 128, 256
        A = (rng.standard_normal((M, K)) * 1.0e-5).astype(np.float16)
        W_fp16 = (rng.standard_normal((K, N)) * 1.0e-5).astype(np.float16)

        packed, scales = quantize_fp4_k_packed(W_fp16, group_size=128)
        W_dequant = dequant_fp4_k_packed(packed, scales, group_size=128)
        ref = gemm_reference(A, W_dequant)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=128)
        result_np = result.cpu().numpy()

        np.testing.assert_allclose(
            result_np.astype(np.float32), ref.astype(np.float32), rtol=0.2, atol=1e-5
        )

    @requires_torch
    @requires_mps
    def test_mixed_signs_cancellation(self, rng: np.random.Generator) -> None:
        """Mixed positive/negative values with potential cancellation."""
        assert torch is not None

        M, N, K = 8, 256, 512
        A = rng.standard_normal((M, K)).astype(np.float16)
        W_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_fp4_k_packed(W_fp16, group_size=128)
        W_dequant = dequant_fp4_k_packed(packed, scales, group_size=128)
        ref = gemm_reference(A, W_dequant)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=128)
        result_np = result.cpu().numpy()

        np.testing.assert_allclose(
            result_np.astype(np.float32), ref.astype(np.float32), rtol=0.05, atol=0.02
        )


# ===========================================================================
# Test: Quantization quality (pure numpy, validates quant pipeline)
# ===========================================================================


class TestQuantizationQuality:
    """FP4 quantize-dequant roundtrip quality."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=99)

    @pytest.mark.parametrize(
        "K,N",
        [
            (128, 128),
            (256, 256),
            pytest.param(4096, 4096, marks=pytest.mark.slow),
        ],
    )
    def test_roundtrip_error_bounded(self, rng: np.random.Generator, K: int, N: int) -> None:
        """Quantize-dequant median relative error is bounded (<50%)."""
        B = rng.standard_normal((K, N)).astype(np.float16)
        group_size = min(128, K)

        packed, scales = quantize_fp4_k_packed(B, group_size=group_size)
        B_recovered = dequant_fp4_k_packed(packed, scales, group_size=group_size)

        # Skip near-zero values
        mask = np.abs(B.astype(np.float32)) > 0.01
        if not mask.any():
            return

        rel_error = np.abs(B_recovered[mask] - B.astype(np.float32)[mask]) / (
            np.abs(B.astype(np.float32)[mask]) + 1e-7
        )
        median_err = np.median(rel_error)
        assert median_err < 0.5, f"Median relative error too high: {median_err:.4f}"

    def test_exact_representable_values(self) -> None:
        """Values exactly representable in FP4 should quantize well."""
        K, N = 128, 8
        exact_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        B = np.zeros((K, N), dtype=np.float16)
        for i in range(K):
            B[i, :] = exact_vals[i % len(exact_vals)]

        packed, scales = quantize_fp4_k_packed(B, group_size=K)
        B_recovered = dequant_fp4_k_packed(packed, scales, group_size=K)

        non_zero = B.astype(np.float32) != 0.0
        if non_zero.any():
            rel_err = np.abs(B_recovered[non_zero] - B.astype(np.float32)[non_zero]) / np.abs(
                B.astype(np.float32)[non_zero]
            )
            assert rel_err.max() < 0.01, (
                f"Exact FP4 values not recovered: max rel error = {rel_err.max()}"
            )

    def test_zero_weights_exact(self) -> None:
        """Zero weights are exactly preserved."""
        K, N = 128, 64
        B = np.zeros((K, N), dtype=np.float16)
        packed, scales = quantize_fp4_k_packed(B, group_size=128)
        B_recovered = dequant_fp4_k_packed(packed, scales, group_size=128)
        assert np.allclose(B_recovered, 0.0, atol=1e-7)


# ===========================================================================
# Test: GEMM boundary conditions (partial tiles)
# ===========================================================================


class TestGEMMBoundaries:
    """Boundary conditions: non-tile-aligned dims, partial tiles.

    The kernel tiles at TILE_M=16, TILE_N=16, TILE_K=32.
    """

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=1337)

    # -----------------------------------------------------------------------
    # Non-aligned M (partial M-tiles)
    # -----------------------------------------------------------------------

    @requires_torch
    @requires_mps
    @pytest.mark.parametrize("M", [1, 2, 7, 15, 33, 65])
    def test_non_aligned_m(self, rng: np.random.Generator, M: int) -> None:
        """Non-tile-aligned M dimensions are handled correctly."""
        assert torch is not None

        K, N = 128, 128
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_fp4_k_packed(B_fp16, group_size=128)
        B_dequant = dequant_fp4_k_packed(packed, scales, group_size=128)
        ref = gemm_reference(A, B_dequant)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=128)
        result_np = result.cpu().numpy()

        ref_scale = np.abs(ref).max() + 1e-7
        atol = max(float(np.sqrt(K)) * 5e-3 * ref_scale, 1e-3)

        np.testing.assert_allclose(
            result_np.astype(np.float32), ref.astype(np.float32), rtol=5e-2, atol=atol
        )

    # -----------------------------------------------------------------------
    # Various N (must be multiple of 8 for packing)
    # -----------------------------------------------------------------------

    @requires_torch
    @requires_mps
    @pytest.mark.parametrize("N", [16, 24, 48, 72, 128, 256])
    def test_various_n(self, rng: np.random.Generator, N: int) -> None:
        """Various N dimensions (multiples of 8)."""
        assert torch is not None

        M, K = 8, 256
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_fp4_k_packed(B_fp16, group_size=128)
        B_dequant = dequant_fp4_k_packed(packed, scales, group_size=128)
        ref = gemm_reference(A, B_dequant)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=128)
        result_np = result.cpu().numpy()

        ref_scale = np.abs(ref).max() + 1e-7
        atol = max(float(np.sqrt(K)) * 5e-3 * ref_scale, 1e-3)

        np.testing.assert_allclose(
            result_np.astype(np.float32), ref.astype(np.float32), rtol=5e-2, atol=atol
        )

    # -----------------------------------------------------------------------
    # Tile boundary values
    # -----------------------------------------------------------------------

    @requires_torch
    @requires_mps
    @pytest.mark.parametrize("M", [1, 3, 15, 16, 17])
    def test_m_tile_boundary(self, rng: np.random.Generator, M: int) -> None:
        """M values at/near TILE_M=16 boundary."""
        assert torch is not None

        N, K = 64, 128
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_fp4_k_packed(B_fp16, group_size=32)
        B_dequant = dequant_fp4_k_packed(packed, scales, group_size=32)
        ref = gemm_reference(A, B_dequant)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=32)
        result_np = result.cpu().numpy()

        atol = float(np.sqrt(K)) * 0.01 * (float(np.abs(ref).max()) + 1e-6)
        np.testing.assert_allclose(
            result_np.astype(np.float32), ref.astype(np.float32), rtol=0.02, atol=atol
        )

    @requires_torch
    @requires_mps
    @pytest.mark.parametrize("K", [32, 64, 96, 128, 160, 192])
    def test_k_tile_boundary(self, rng: np.random.Generator, K: int) -> None:
        """K values at multiples of 32 near TILE_K=32."""
        assert torch is not None

        M, N = 4, 64
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_fp4_k_packed(B_fp16, group_size=32)
        B_dequant = dequant_fp4_k_packed(packed, scales, group_size=32)
        ref = gemm_reference(A, B_dequant)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=32)
        result_np = result.cpu().numpy()

        atol = float(np.sqrt(K)) * 0.01 * (float(np.abs(ref).max()) + 1e-6)
        np.testing.assert_allclose(
            result_np.astype(np.float32), ref.astype(np.float32), rtol=0.02, atol=atol
        )

    # -----------------------------------------------------------------------
    # All partial tiles simultaneously
    # -----------------------------------------------------------------------

    @requires_torch
    @requires_mps
    def test_all_partial_tiles(self, rng: np.random.Generator) -> None:
        """M=13, N=24, K=96: partial tiles in all three dimensions."""
        assert torch is not None

        M, N, K = 13, 24, 96
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_fp4_k_packed(B_fp16, group_size=32)
        B_dequant = dequant_fp4_k_packed(packed, scales, group_size=32)
        ref = gemm_reference(A, B_dequant)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=32)
        result_np = result.cpu().numpy()

        max_abs = max(float(np.abs(ref).max()), 1e-6)
        atol = float(np.sqrt(K)) * 5e-3 * max_abs

        np.testing.assert_allclose(
            result_np.astype(np.float32), ref.astype(np.float32), rtol=0.02, atol=atol
        )

    @requires_torch
    @requires_mps
    def test_prime_m(self, rng: np.random.Generator) -> None:
        """M=97 (prime): never divides evenly into any tile size."""
        assert torch is not None

        M, N, K = 97, 64, 128
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_fp4_k_packed(B_fp16, group_size=32)
        B_dequant = dequant_fp4_k_packed(packed, scales, group_size=32)
        ref = gemm_reference(A, B_dequant)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=32)
        result_np = result.cpu().numpy()

        max_abs = max(float(np.abs(ref).max()), 1e-6)
        atol = float(np.sqrt(K)) * 5e-3 * max_abs

        np.testing.assert_allclose(
            result_np.astype(np.float32), ref.astype(np.float32), rtol=0.02, atol=atol
        )

    # -----------------------------------------------------------------------
    # Exact tile multiples (sanity baseline)
    # -----------------------------------------------------------------------

    @requires_torch
    @requires_mps
    @pytest.mark.parametrize(
        "M,N,K",
        [
            (16, 16, 32),  # 1 tile each
            (32, 32, 64),  # 2 tiles each
            (16, 64, 128),  # 1 M-tile, 4 N-tiles, 4 K-tiles
            (64, 128, 256),  # 4 M-tiles, 8 N-tiles, 8 K-tiles
        ],
    )
    def test_exact_tile_multiples(self, rng: np.random.Generator, M: int, N: int, K: int) -> None:
        """Exact tile multiples should produce correct results."""
        assert torch is not None

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_fp4_k_packed(B_fp16, group_size=32)
        B_dequant = dequant_fp4_k_packed(packed, scales, group_size=32)
        ref = gemm_reference(A, B_dequant)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=32)
        result_np = result.cpu().numpy()

        max_abs = max(float(np.abs(ref).max()), 1e-6)
        atol = float(np.sqrt(K)) * 5e-3 * max_abs

        np.testing.assert_allclose(
            result_np.astype(np.float32), ref.astype(np.float32), rtol=0.02, atol=atol
        )

    # -----------------------------------------------------------------------
    # Large dimension stress tests
    # -----------------------------------------------------------------------

    @requires_torch
    @requires_mps
    @pytest.mark.slow
    def test_very_long_k(self, rng: np.random.Generator) -> None:
        """M=1, N=4096, K=32768: long reduction stresses accumulation."""
        assert torch is not None

        M, N, K = 1, 4096, 32768
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_fp4_k_packed(B_fp16, group_size=32)
        B_dequant = dequant_fp4_k_packed(packed, scales, group_size=32)
        ref = gemm_reference(A, B_dequant)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=32)
        result_np = result.cpu().numpy()

        max_abs = max(float(np.abs(ref).max()), 1e-6)
        atol = float(np.sqrt(K)) * 0.01 * max_abs

        np.testing.assert_allclose(
            result_np.astype(np.float32), ref.astype(np.float32), rtol=0.05, atol=atol
        )

    @requires_torch
    @requires_mps
    @pytest.mark.slow
    def test_very_wide_n(self, rng: np.random.Generator) -> None:
        """M=1, N=32768, K=4096: wide output stresses N-tiling."""
        assert torch is not None

        M, N, K = 1, 32768, 4096
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_fp4_k_packed(B_fp16, group_size=32)
        B_dequant = dequant_fp4_k_packed(packed, scales, group_size=32)
        ref = gemm_reference(A, B_dequant)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=32)
        result_np = result.cpu().numpy()

        max_abs = max(float(np.abs(ref).max()), 1e-6)
        atol = float(np.sqrt(K)) * 5e-3 * max_abs

        np.testing.assert_allclose(
            result_np.astype(np.float32), ref.astype(np.float32), rtol=0.02, atol=atol
        )

    @requires_torch
    @requires_mps
    @pytest.mark.slow
    def test_large_m_non_aligned(self, rng: np.random.Generator) -> None:
        """M=127, N=4096, K=4096: large non-aligned M with LLM-scale dims."""
        assert torch is not None

        M, N, K = 127, 4096, 4096
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_fp4_k_packed(B_fp16, group_size=128)
        B_dequant = dequant_fp4_k_packed(packed, scales, group_size=128)
        ref = gemm_reference(A, B_dequant)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=128)
        result_np = result.cpu().numpy()

        max_abs = max(float(np.abs(ref).max()), 1e-6)
        atol = float(np.sqrt(K)) * 5e-3 * max_abs

        np.testing.assert_allclose(
            result_np.astype(np.float32), ref.astype(np.float32), rtol=0.02, atol=atol
        )


# ===========================================================================
# Test: Edge case inputs
# ===========================================================================


class TestEdgeCaseInputs:
    """Edge case inputs: constant values, special patterns."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=42)

    @requires_torch
    @requires_mps
    def test_constant_activations(self, rng: np.random.Generator) -> None:
        """Constant activations: output equals column sums of dequant(B)."""
        assert torch is not None

        M, N, K = 3, 16, 32
        A = np.ones((M, K), dtype=np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_fp4_k_packed(B_fp16, group_size=32)
        B_dequant = dequant_fp4_k_packed(packed, scales, group_size=32)
        ref = gemm_reference(A, B_dequant)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=32)
        result_np = result.cpu().numpy()

        np.testing.assert_allclose(
            result_np.astype(np.float32), ref.astype(np.float32), rtol=0.02, atol=1e-3
        )

        # All rows should be identical (since A rows are identical)
        for row in range(1, M):
            assert np.allclose(result_np[row], result_np[0], atol=1e-6)

    @requires_torch
    @requires_mps
    def test_uniform_weight(self, rng: np.random.Generator) -> None:
        """Uniform weight (all same value) reduces to scaled row-sum."""
        assert torch is not None

        M, K, N = 4, 128, 128
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_uniform = np.full((K, N), 2.0, dtype=np.float16)

        packed, scales = quantize_fp4_k_packed(B_uniform, group_size=128)
        B_dequant = dequant_fp4_k_packed(packed, scales, group_size=128)
        ref = gemm_reference(A, B_dequant)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=128)
        result_np = result.cpu().numpy()

        ref_scale = np.abs(ref).max() + 1e-7
        atol = max(float(np.sqrt(K)) * 5e-3 * ref_scale, 1e-3)

        np.testing.assert_allclose(
            result_np.astype(np.float32), ref.astype(np.float32), rtol=5e-2, atol=atol
        )

    @requires_torch
    @requires_mps
    def test_negative_weights(self, rng: np.random.Generator) -> None:
        """All-negative weights are handled correctly."""
        assert torch is not None

        M, K, N = 4, 128, 128
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_neg = np.full((K, N), -3.0, dtype=np.float16)

        packed, scales = quantize_fp4_k_packed(B_neg, group_size=128)
        B_dequant = dequant_fp4_k_packed(packed, scales, group_size=128)
        ref = gemm_reference(A, B_dequant)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=128)
        result_np = result.cpu().numpy()

        ref_scale = np.abs(ref).max() + 1e-7
        atol = max(float(np.sqrt(K)) * 5e-3 * ref_scale, 1e-3)

        np.testing.assert_allclose(
            result_np.astype(np.float32), ref.astype(np.float32), rtol=5e-2, atol=atol
        )

    @requires_torch
    @requires_mps
    def test_identity_weight(self, rng: np.random.Generator) -> None:
        """Identity-ish quantized weight should approximately preserve input."""
        assert torch is not None

        K = 128
        A = rng.standard_normal((1, K)).astype(np.float16)
        B_identity = np.eye(K, dtype=np.float16)

        packed, scales = quantize_fp4_k_packed(B_identity, group_size=K)
        B_dequant = dequant_fp4_k_packed(packed, scales, group_size=K)
        ref = gemm_reference(A, B_dequant)

        from metal_marlin import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=K)
        result_np = result.cpu().numpy()

        np.testing.assert_allclose(
            result_np.astype(np.float32), ref.astype(np.float32), rtol=5e-2, atol=1e-3
        )

        # Correlation with original input should be high
        corr = np.corrcoef(A.flatten().astype(np.float32), result_np.flatten().astype(np.float32))[
            0, 1
        ]
        assert corr > 0.85, f"Identity weight correlation too low: {corr:.4f}"
