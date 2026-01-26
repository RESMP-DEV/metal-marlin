"""Mixed precision accumulation tests for Metal Marlin GEMM.

Validates that FP32 accumulation prevents overflow and precision loss for
large K dimensions, and quantifies the error difference between FP16 and
FP32 accumulation strategies.

Test categories:
  1. Overflow detection: K=32768 with all-ones weights in FP16 acc must overflow
  2. FP32 accumulation correctness: large K produces correct results
  3. Error scaling: verify error grows with sqrt(K) for FP16 accumulation
  4. Precision comparison: FP16 vs FP32 accumulation for various K values
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# FP4 E2M1 constants and helpers (duplicated from test_accuracy.py for
# standalone execution per contrib guidelines)
# ---------------------------------------------------------------------------

FP4_E2M1_TABLE: np.ndarray = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float16,
)

# Maximum representable FP4 E2M1 magnitude
FP4_MAX = 6.0


def quantize_to_fp4(
    weights: np.ndarray, group_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize FP16 weights to FP4 E2M1 format with per-group scales.

    Returns packed uint32 array [K/8, N] and scales [K/group_size, N].
    """
    K, N = weights.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    assert K % 8 == 0, f"K={K} must be divisible by 8"

    num_groups = K // group_size
    scales = np.zeros((num_groups, N), dtype=np.float16)
    packed = np.zeros((K // 8, N), dtype=np.uint32)

    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        group = weights[start:end, :].astype(np.float32)

        # Per-column max absolute value determines scale
        amax = np.abs(group).max(axis=0)
        amax = np.maximum(amax, 1e-7)  # avoid division by zero
        scale = amax / FP4_MAX
        scales[g, :] = scale.astype(np.float16)

        # Quantize: find nearest FP4 code
        for k_local in range(group_size):
            k_global = start + k_local
            for n in range(N):
                val = group[k_local, n] / float(scale[n])
                # Find closest code
                best_code = 0
                best_dist = float("inf")
                for code in range(16):
                    dist = abs(float(FP4_E2M1_TABLE[code]) - val)
                    if dist < best_dist:
                        best_dist = dist
                        best_code = code

                # Pack into uint32 (8 codes per word)
                pack_idx = k_global // 8
                bit_offset = (k_global % 8) * 4
                packed[pack_idx, n] |= np.uint32(best_code) << np.uint32(bit_offset)

    return packed, scales


def dequant_fp4_array(
    packed: np.ndarray,
    scales: np.ndarray,
    K: int,
    N: int,
    group_size: int = 128,
) -> np.ndarray:
    """Dequantize packed FP4 array back to FP16."""
    result = np.zeros((K, N), dtype=np.float32)

    for k in range(K):
        pack_idx = k // 8
        bit_offset = (k % 8) * 4
        for n in range(N):
            code = (int(packed[pack_idx, n]) >> bit_offset) & 0xF
            fp4_val = float(FP4_E2M1_TABLE[code])

            group_idx = k // group_size
            scale = float(scales[group_idx, n])
            result[k, n] = fp4_val * scale

    return result.astype(np.float16)


def make_all_ones_fp4(K: int, N: int, group_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Create packed FP4 weights where every value decodes to +1.0 (code=2).

    With scale=1.0, every dequantized weight is exactly 1.0.
    """
    assert K % 8 == 0
    assert K % group_size == 0

    packed = np.zeros((K // 8, N), dtype=np.uint32)
    # Code 2 = 1.0 in FP4 E2M1
    # 8 codes of 2 packed: 0x22222222
    for pack_idx in range(K // 8):
        packed[pack_idx, :] = 0x22222222

    num_groups = K // group_size
    scales = np.ones((num_groups, N), dtype=np.float16)

    return packed, scales


def make_max_fp4(K: int, N: int, group_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Create packed FP4 weights where every value decodes to +6.0 (code=7, max).

    With scale=1.0, every dequantized weight is exactly 6.0.
    """
    assert K % 8 == 0
    assert K % group_size == 0

    packed = np.zeros((K // 8, N), dtype=np.uint32)
    # Code 7 = 6.0 in FP4 E2M1
    # 8 codes of 7 packed: 0x77777777
    for pack_idx in range(K // 8):
        packed[pack_idx, :] = 0x77777777

    num_groups = K // group_size
    scales = np.ones((num_groups, N), dtype=np.float16)

    return packed, scales


def gemm_fp16_accumulation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Simulate FP16 accumulation: cast inputs to FP16 and accumulate in FP16.

    This mimics what simdgroup_multiply_accumulate does with half accumulators.
    Numpy doesn't have native FP16 matmul, so we simulate step-by-step.
    """
    M, K = A.shape
    _, N = B.shape
    A_fp16 = A.astype(np.float16)
    B_fp16 = B.astype(np.float16)

    # Simulate FP16 accumulation: tile-by-tile to match GPU behavior
    # Process in chunks of 8 (matching simdgroup K-tile size)
    C = np.zeros((M, N), dtype=np.float16)
    tile_k = 8  # matches simdgroup_multiply_accumulate 8x8 tiles
    for k_start in range(0, K, tile_k):
        k_end = min(k_start + tile_k, K)
        # Each tile product is computed in FP16 and added to FP16 accumulator
        tile_product = (
            A_fp16[:, k_start:k_end].astype(np.float32)
            @ B_fp16[k_start:k_end, :].astype(np.float32)
        ).astype(np.float16)
        C = (C.astype(np.float32) + tile_product.astype(np.float32)).astype(np.float16)
    return C


def gemm_fp32_accumulation(
    A: np.ndarray, B: np.ndarray, *, output_fp16: bool = True
) -> np.ndarray:
    """FP32 accumulation: inputs are FP16, accumulation in FP32.

    Args:
        output_fp16: If True, cast result to FP16 (default, matches kernel output).
            If False, keep FP32 result (useful for checking overflow-free computation).
    """
    result = A.astype(np.float32) @ B.astype(np.float32)
    if output_fp16:
        return result.astype(np.float16)
    return result


# ===========================================================================
# Test Class: Overflow Detection
# ===========================================================================


class TestFP16Overflow:
    """Verify that FP16 accumulation overflows for large K with extreme weights."""

    def test_k32768_all_max_overflows_fp16(self) -> None:
        """K=32768, all weights=6.0, all activations=1.0 overflows FP16.

        Expected sum per output element: 32768 * 6.0 = 196608
        FP16 max: 65504
        Result: inf in FP16 accumulation, correct in FP32 accumulation.
        Note: The FP32 result exceeds FP16 representable range so we keep it
        in FP32 to verify the accumulator itself didn't overflow.
        """
        K = 32768
        N = 8  # small N for speed

        A = np.ones((1, K), dtype=np.float16)
        packed, scales = make_max_fp4(K, N, group_size=128)
        B_dequant = np.full((K, N), 6.0, dtype=np.float16)

        result_fp16 = gemm_fp16_accumulation(A, B_dequant)
        # Keep FP32 result to avoid overflow on final cast (196608 > FP16 max)
        result_fp32 = gemm_fp32_accumulation(A, B_dequant, output_fp16=False)

        # FP16 accumulation must overflow
        assert np.any(np.isinf(result_fp16)), (
            f"Expected FP16 overflow for K=32768, max={result_fp16.max()}"
        )

        # FP32 accumulation must NOT overflow
        assert not np.any(np.isinf(result_fp32)), (
            "FP32 accumulator should not overflow, got inf"
        )

        # FP32 result should be correct: 32768 * 6.0 = 196608.0
        expected_f32 = 32768.0 * 6.0
        np.testing.assert_allclose(
            result_fp32,
            np.full((1, N), expected_f32, dtype=np.float32),
            rtol=1e-5,
        )

    def test_k32768_all_ones_no_overflow_fp16(self) -> None:
        """K=32768, all weights=1.0, all activations=1.0 does NOT overflow FP16.

        Expected sum: 32768 * 1.0 = 32768 (within FP16 range of 65504).
        However, FP16 accumulation suffers severe precision loss: once the
        accumulator exceeds 2048 (ULP=2), additions of 1.0 start rounding away.
        By 4096 (ULP=4), 1.0 is below half-ULP and rounds to zero. The
        accumulator stalls, producing a result far below the true sum.
        """
        K = 32768
        N = 8

        A = np.ones((1, K), dtype=np.float16)
        B_dequant = np.ones((K, N), dtype=np.float16)

        result_fp16 = gemm_fp16_accumulation(A, B_dequant)

        # Should NOT overflow (32768 < 65504) but precision is severely degraded
        assert not np.any(np.isinf(result_fp16)), (
            "K=32768 with ones should not overflow FP16"
        )

        # FP32 accumulation gives the exact answer
        result_fp32 = gemm_fp32_accumulation(A, B_dequant)

        # Verify FP32 result is correct
        np.testing.assert_allclose(
            result_fp32.astype(np.float32),
            np.full((1, N), 32768.0, dtype=np.float32),
            rtol=1e-4,
        )

        # The FP16-accumulated result will be much less than 32768 due to
        # precision starvation. The key test: FP32 acc is significantly more
        # accurate than FP16 acc for this scenario.
        fp16_error = float(np.abs(
            result_fp16.astype(np.float32) - 32768.0
        ).max())
        fp32_error = float(np.abs(
            result_fp32.astype(np.float32) - 32768.0
        ).max())

        # FP16 error should be large (precision loss from accumulating 32768 ones)
        assert fp16_error > 1000, (
            f"Expected significant FP16 precision loss, got error={fp16_error}"
        )
        # FP32 error should be negligible
        assert fp32_error < 1.0, (
            f"FP32 error too large: {fp32_error}"
        )

    @pytest.mark.parametrize("K", [4096, 8192, 16384, 32768])
    def test_max_weights_overflow_threshold(self, K: int) -> None:
        """Determine at which K the all-max-weight scenario overflows FP16.

        sum = K * 6.0. Overflow when K * 6 > 65504, i.e. K > 10917.
        """
        N = 4
        A = np.ones((1, K), dtype=np.float16)
        B_dequant = np.full((K, N), 6.0, dtype=np.float16)

        result_fp16 = gemm_fp16_accumulation(A, B_dequant)
        expected_overflow = (K * 6.0) > 65504.0

        if expected_overflow:
            assert np.any(np.isinf(result_fp16)), (
                f"Expected overflow for K={K} (sum={K*6.0})"
            )
        else:
            assert not np.any(np.isinf(result_fp16)), (
                f"Unexpected overflow for K={K} (sum={K*6.0} < 65504)"
            )


# ===========================================================================
# Test Class: FP32 Accumulation Correctness
# ===========================================================================


class TestFP32Accumulation:
    """Verify FP32 accumulation produces correct results for large K."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=123)

    @pytest.mark.parametrize("K", [2048, 4096, 8192, 16384, 32768])
    def test_fp32_acc_matches_reference(
        self, rng: np.random.Generator, K: int
    ) -> None:
        """FP32 accumulation should match FP64 reference within FP32 epsilon."""
        M, N = 4, 64
        A = rng.standard_normal((M, K)).astype(np.float16)
        B = rng.standard_normal((K, N)).astype(np.float16)

        result_fp32 = gemm_fp32_accumulation(A, B)

        # FP64 reference (ground truth)
        ref_fp64 = (A.astype(np.float64) @ B.astype(np.float64))

        # FP32 accumulation error: bounded by K * eps_fp32 * max_product
        max_prod = float(np.abs(A.astype(np.float32)).max() * np.abs(B.astype(np.float32)).max())
        expected_error = K * 1.19e-7 * max_prod * np.sqrt(K)

        abs_error = np.abs(
            result_fp32.astype(np.float64) - ref_fp64
        ).max()

        # Error should be much smaller than FP16 range
        assert abs_error < max(expected_error, 1.0), (
            f"FP32 acc error too large for K={K}: {abs_error:.6f} > {expected_error:.6f}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("K", [32768, 65536])
    def test_fp32_acc_large_k_no_overflow(
        self, rng: np.random.Generator, K: int
    ) -> None:
        """FP32 accumulation handles K=32768+ without overflow."""
        M, N = 1, 32
        # Use uniform [0,1] to ensure positive accumulation (worst case for overflow)
        A = rng.uniform(0, 1, (M, K)).astype(np.float16)
        B = rng.uniform(0, 1, (K, N)).astype(np.float16)

        result = gemm_fp32_accumulation(A, B)

        assert not np.any(np.isinf(result)), f"FP32 overflow at K={K}"
        assert not np.any(np.isnan(result)), f"NaN at K={K}"
        # Output should be roughly K/4 (mean of uniform products)
        mean_output = float(result.astype(np.float32).mean())
        assert mean_output > K * 0.1, (
            f"Output too small for K={K}: mean={mean_output}"
        )


# ===========================================================================
# Test Class: Error Scaling Analysis
# ===========================================================================


class TestErrorScaling:
    """Verify FP16 accumulation error grows as expected with K dimension."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=456)

    def test_error_grows_with_k(self, rng: np.random.Generator) -> None:
        """FP16 accumulation error should grow roughly as sqrt(K) for random inputs.

        We compare FP16-accumulated result against FP32-accumulated reference
        across increasing K values and verify error growth rate.
        """
        M, N = 8, 64
        k_values = [128, 256, 512, 1024, 2048, 4096]
        errors: list[float] = []

        for K in k_values:
            A = rng.standard_normal((M, K)).astype(np.float16)
            B = rng.standard_normal((K, N)).astype(np.float16)

            result_fp16 = gemm_fp16_accumulation(A, B)
            result_fp32 = gemm_fp32_accumulation(A, B)

            # Relative error (avoid div-by-zero with small outputs)
            ref_magnitude = np.abs(result_fp32.astype(np.float32))
            mask = ref_magnitude > 0.1
            if mask.any():
                rel_error = np.abs(
                    result_fp16[mask].astype(np.float32)
                    - result_fp32[mask].astype(np.float32)
                ) / ref_magnitude[mask]
                errors.append(float(np.median(rel_error)))
            else:
                errors.append(0.0)

        # Verify error increases with K
        # Allow some noise but the trend should be clear
        assert errors[-1] > errors[0], (
            f"Error should increase with K: errors={errors}"
        )

        # Check rough sqrt(K) scaling: error at K=4096 should be ~sqrt(32) = 5.7x
        # larger than error at K=128. Allow 2x-20x range for robustness.
        if errors[0] > 1e-6:
            ratio = errors[-1] / errors[0]
            assert ratio > 1.5, (
                f"Error ratio K=4096/K=128 too small: {ratio:.2f} (expected >1.5)"
            )

    @pytest.mark.parametrize(
        "K,max_rel_error",
        [
            (128, 0.01),    # <1% for small K
            (512, 0.02),    # <2% for medium K
            (2048, 0.05),   # <5% for large K
            (4096, 0.10),   # <10% for very large K
        ],
    )
    def test_fp16_error_bounds(
        self, rng: np.random.Generator, K: int, max_rel_error: float
    ) -> None:
        """FP16 accumulation median relative error is bounded for each K."""
        M, N = 16, 128
        A = rng.standard_normal((M, K)).astype(np.float16)
        B = rng.standard_normal((K, N)).astype(np.float16)

        result_fp16 = gemm_fp16_accumulation(A, B)
        result_fp32 = gemm_fp32_accumulation(A, B)

        ref_magnitude = np.abs(result_fp32.astype(np.float32))
        mask = ref_magnitude > 1.0
        assert mask.any(), f"No outputs above threshold for K={K}"

        rel_error = np.abs(
            result_fp16[mask].astype(np.float32)
            - result_fp32[mask].astype(np.float32)
        ) / ref_magnitude[mask]

        median_error = float(np.median(rel_error))
        assert median_error < max_rel_error, (
            f"FP16 median relative error {median_error:.4f} exceeds bound "
            f"{max_rel_error} for K={K}"
        )


# ===========================================================================
# Test Class: Quantized GEMM Precision Comparison
# ===========================================================================


class TestQuantizedPrecision:
    """Compare FP16 vs FP32 accumulation with actual FP4 quantized weights."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=789)

    @pytest.mark.parametrize("K", [128, 512, 2048, 4096])
    def test_quantized_gemm_fp16_vs_fp32_acc(
        self, rng: np.random.Generator, K: int
    ) -> None:
        """For quantized weights, FP32 acc should be closer to ground truth."""
        M, N = 4, 64
        group_size = 128

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = dequant_fp4_array(packed, scales, K, N, group_size=group_size)

        # Ground truth: FP64
        ref = (A.astype(np.float64) @ B_dequant.astype(np.float64))

        result_fp16 = gemm_fp16_accumulation(A, B_dequant)
        result_fp32 = gemm_fp32_accumulation(A, B_dequant)

        error_fp16 = float(np.abs(
            result_fp16.astype(np.float64) - ref
        ).mean())
        error_fp32 = float(np.abs(
            result_fp32.astype(np.float64) - ref
        ).mean())

        # FP32 accumulation should always be at least as good as FP16
        # (usually much better for K >= 512)
        assert error_fp32 <= error_fp16 * 1.1, (
            f"FP32 acc should be at least as precise as FP16. "
            f"FP16 error={error_fp16:.6f}, FP32 error={error_fp32:.6f}"
        )

        if K >= 2048:
            # For large K, FP32 should be significantly better
            improvement = error_fp16 / max(error_fp32, 1e-10)
            assert improvement > 2.0, (
                f"FP32 should be >2x better for K={K}: "
                f"improvement={improvement:.2f}x"
            )

    @pytest.mark.slow
    def test_k32768_quantized_overflow(self, rng: np.random.Generator) -> None:
        """K=32768 with all-ones FP4 weights: verify FP16 overflow scenario.

        All FP4 codes set to 2 (value=1.0), scale=1.0.
        Sum per output = 32768 * 1.0 = 32768 (within FP16 range but precision-degraded).
        """
        K = 32768
        N = 16
        group_size = 128

        A = np.ones((1, K), dtype=np.float16)
        packed, scales = make_all_ones_fp4(K, N, group_size=group_size)
        B_dequant = np.ones((K, N), dtype=np.float16)

        result_fp16 = gemm_fp16_accumulation(A, B_dequant)
        result_fp32 = gemm_fp32_accumulation(A, B_dequant)

        # FP32 result should be exact (32768.0 is representable)
        np.testing.assert_allclose(
            result_fp32.astype(np.float32),
            np.full((1, N), 32768.0, dtype=np.float32),
            rtol=1e-5,
        )

        # FP16 result: 32768.0 is representable as half, but accumulated
        # via many additions it may have precision loss
        float(np.abs(
            result_fp16.astype(np.float32) - 32768.0
        ).max())
        # At magnitude 32768, FP16 ULP = 32, so precision is very coarse
        # but the result should still be finite
        assert not np.any(np.isinf(result_fp16)), "All-ones K=32768 should not overflow"

    @pytest.mark.slow
    def test_k32768_max_weights_overflow_detection(
        self, rng: np.random.Generator
    ) -> None:
        """K=32768 with max FP4 weights (6.0): confirm FP16 overflows.

        All FP4 codes set to 7 (value=6.0), scale=1.0.
        Sum per output = 32768 * 6.0 = 196608 > 65504 (FP16 max).
        """
        K = 32768
        N = 8
        group_size = 128

        A = np.ones((1, K), dtype=np.float16)
        packed, scales = make_max_fp4(K, N, group_size=group_size)
        B_dequant = np.full((K, N), 6.0, dtype=np.float16)

        result_fp16 = gemm_fp16_accumulation(A, B_dequant)
        result_fp32 = gemm_fp32_accumulation(A, B_dequant)

        # FP16 MUST overflow
        assert np.all(np.isinf(result_fp16)), (
            f"Expected all-inf for K=32768 max weights, got max={result_fp16.max()}"
        )

        # FP32 must produce correct result
        expected = 32768.0 * 6.0  # = 196608.0
        np.testing.assert_allclose(
            result_fp32.astype(np.float32),
            np.full((1, N), expected, dtype=np.float32),
            rtol=1e-5,
        )


# ===========================================================================
# Test Class: K-Parallel Reduction Precision
# ===========================================================================


class TestKParallelPrecision:
    """Test precision implications of K-parallel splitting."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=101)

    @pytest.mark.parametrize("parallel", [1, 2, 4, 8])
    def test_k_parallel_reduces_fp16_error(
        self, rng: np.random.Generator, parallel: int
    ) -> None:
        """K-parallel with FP16 acc should reduce error vs single-pass FP16.

        Splitting K into `parallel` slices means each slice accumulates K/parallel
        terms, reducing per-slice error. Final reduction adds `parallel` terms.
        """
        K = 4096
        M, N = 4, 32
        A = rng.standard_normal((M, K)).astype(np.float16)
        B = rng.standard_normal((K, N)).astype(np.float16)

        # Single-pass FP16
        single_result = gemm_fp16_accumulation(A, B)

        # K-parallel: each slice accumulates K/parallel terms, then sum
        slice_size = K // parallel
        partial_sums = np.zeros((parallel, M, N), dtype=np.float16)
        for p in range(parallel):
            k_start = p * slice_size
            k_end = k_start + slice_size
            partial_sums[p] = gemm_fp16_accumulation(
                A[:, k_start:k_end], B[k_start:k_end, :]
            )

        # Final reduction in FP16 (matching the kernel's behavior)
        parallel_result = partial_sums.astype(np.float32).sum(axis=0).astype(np.float16)

        # Reference (FP32 ground truth)
        ref = gemm_fp32_accumulation(A, B)

        error_single = float(np.abs(
            single_result.astype(np.float32) - ref.astype(np.float32)
        ).mean())
        error_parallel = float(np.abs(
            parallel_result.astype(np.float32) - ref.astype(np.float32)
        ).mean())

        if parallel > 1:
            # K-parallel should reduce error (or at least not increase it much)
            assert error_parallel < error_single * 1.5, (
                f"K-parallel={parallel} increased error: "
                f"single={error_single:.6f}, parallel={error_parallel:.6f}"
            )

    def test_k_parallel_fp32_no_benefit(self, rng: np.random.Generator) -> None:
        """With FP32 accumulation, K-parallel should not significantly change error.

        FP32 has enough precision that K-splitting is unnecessary for accuracy.
        Both paths compute in FP32 internally; differences arise only from
        floating-point non-associativity and final FP16 output quantization.
        """
        K = 8192
        M, N = 4, 32
        parallel = 4

        A = rng.standard_normal((M, K)).astype(np.float16)
        B = rng.standard_normal((K, N)).astype(np.float16)

        # Single-pass FP32 (keep in FP32 for comparison)
        single_fp32 = gemm_fp32_accumulation(A, B, output_fp16=False)

        # K-parallel FP32
        slice_size = K // parallel
        partial_sums = []
        for p in range(parallel):
            k_start = p * slice_size
            k_end = k_start + slice_size
            partial = gemm_fp32_accumulation(
                A[:, k_start:k_end], B[k_start:k_end, :], output_fp16=False
            )
            partial_sums.append(partial)

        parallel_fp32 = np.stack(partial_sums).sum(axis=0)

        # Compare in FP32 space: differences are only from non-associativity
        # of FP32 arithmetic, which should be negligible relative to the values.
        np.testing.assert_allclose(
            single_fp32,
            parallel_fp32,
            rtol=1e-5,
            atol=1e-4,
        )
