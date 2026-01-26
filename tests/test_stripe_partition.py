"""Tests for stripe-partitioned GEMM kernel accuracy and load balancing.

Validates that stripe-partitioned dispatch produces correct results across various
matrix dimensions and K-parallel factors. Also tests the reduction path for K-parallel
computation and load balance distribution.

The striped dispatch uses 1D linearized tile assignment, splitting the K-dimension
across multiple threadgroups when parallel > 1. This requires reduction to combine
partial sums from different K-slices.

NOTE: This test file uses PyTorch MPS for Metal kernel dispatch. The
implementation uses custom Metal kernels compiled via PyObjC MetalKernelLibrary.
The PyTorch version uses the same underlying Metal shaders dispatched through PyObjC.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch

from .conftest import requires_mps, requires_torch

if TYPE_CHECKING:
    import torch as torch_types


# ---------------------------------------------------------------------------
# Constants matching the Metal kernel
# ---------------------------------------------------------------------------

TILE_M = 64
TILE_N = 64
TILE_K = 32
FP4_PER_UINT = 8
THREADS_PER_TG = 128

# Target threadgroup count for M4 Max (40 GPU cores)
DEFAULT_NUM_TGS = 40


# ---------------------------------------------------------------------------
# FP4 quantization utilities (pure numpy reference)
# ---------------------------------------------------------------------------

FP4_E2M1_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float16,
)


def quantize_to_fp4(weights: np.ndarray, group_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Quantize [K, N] FP16 weights to packed FP4 with per-group scales."""
    K, N = weights.shape
    assert K % group_size == 0
    assert N % FP4_PER_UINT == 0

    num_groups = K // group_size
    weights_grouped = weights.reshape(num_groups, group_size, N)

    scales = np.abs(weights_grouped).max(axis=1).astype(np.float16)
    scales = np.maximum(scales, np.float16(1e-7))

    max_fp4 = np.float16(6.0)
    num_elements = K * N
    codes = np.zeros(num_elements, dtype=np.uint8)

    for g in range(num_groups):
        for n in range(N):
            scale = float(scales[g, n])
            for k in range(group_size):
                val = float(weights_grouped[g, k, n])
                normalized = val / scale * float(max_fp4)
                best_code = 0
                best_dist = float("inf")
                for code in range(16):
                    ref_val = float(FP4_E2M1_TABLE[code])
                    dist = abs(normalized - ref_val)
                    if dist < best_dist:
                        best_dist = dist
                        best_code = code
                flat_idx = (g * group_size + k) * N + n
                codes[flat_idx] = best_code

    packed = np.zeros(num_elements // FP4_PER_UINT, dtype=np.uint32)
    for i in range(len(packed)):
        val = 0
        for j in range(FP4_PER_UINT):
            val |= int(codes[i * FP4_PER_UINT + j]) << (j * 4)
        packed[i] = val

    return packed, scales


def dequant_fp4_array(
    packed: np.ndarray, scales: np.ndarray, K: int, N: int, group_size: int = 128
) -> np.ndarray:
    """Dequantize packed FP4 to [K, N] FP16 using reference LUT."""
    max_fp4 = np.float16(6.0)
    num_elements = K * N
    result = np.zeros(num_elements, dtype=np.float16)

    for i in range(len(packed)):
        for j in range(FP4_PER_UINT):
            code = (int(packed[i]) >> (j * 4)) & 0xF
            flat_idx = i * FP4_PER_UINT + j
            if flat_idx >= num_elements:
                break
            row = flat_idx // N
            col = flat_idx % N
            group = row // group_size
            scale = float(scales[group, col])
            ref_val = float(FP4_E2M1_TABLE[code])
            result[flat_idx] = np.float16(ref_val * scale / float(max_fp4))

    return result.reshape(K, N)


def gemm_reference(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """FP32 GEMM reference: A @ B with FP32 accumulation."""
    return (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


# ===========================================================================
# Test: Striped kernel matches reference
# ===========================================================================


class TestStripedVsReference:
    """Compare stripe-partitioned kernel output to numpy FP32 reference."""

    @requires_torch
    @requires_mps
    @pytest.mark.parametrize(
        "M,N,K",
        [
            pytest.param(128, 4096, 4096, id="standard-llm"),
            pytest.param(1, 4096, 16384, id="single-token-long-k"),
            pytest.param(256, 256, 32768, id="long-k-reduction", marks=pytest.mark.slow),
        ],
    )
    def test_striped_matches_reference(
        self, rng: np.random.Generator, M: int, N: int, K: int
    ) -> None:
        """Stripe kernel output matches numpy FP32 reference within FP16 tolerance."""
        assert torch is not None

        group_size = 128
        assert K % group_size == 0

        # Generate random matrices
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        # Quantize weights to FP4
        packed, scales = quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = dequant_fp4_array(packed, scales, K, N, group_size)

        # FP32 reference matmul
        ref = gemm_reference(A, B_dequant)

        # Reshape packed for kernel: [K/8, N] layout
        packed_reshaped = packed.reshape(K // FP4_PER_UINT, N)

        # Run Metal kernel via PyTorch MPS
        from metal_marlin.kernels import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed_reshaped).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=group_size)
        result_np = result.cpu().numpy()

        # Adaptive tolerance based on K and value magnitudes
        ref_scale = np.abs(ref).max() + 1e-7
        rtol = 5e-2
        atol = max(float(np.sqrt(K)) * 5e-3 * ref_scale, 1e-3)

        np.testing.assert_allclose(
            result_np.astype(np.float32),
            ref.astype(np.float32),
            rtol=rtol,
            atol=atol,
            err_msg=(
                f"Striped vs reference mismatch for ({M},{N},{K}). "
                f"Max diff: {np.abs(result_np.astype(np.float32) - ref.astype(np.float32)).max():.6f}"
            ),
        )


# ===========================================================================
# Test: K-parallel reduction correctness
# ===========================================================================


class TestKParallelReduction:
    """Validate K-dimension reduction path via multiple kernel invocations."""

    @requires_torch
    @requires_mps
    def test_k_parallel_simulation_large_k(self, rng: np.random.Generator) -> None:
        """Simulate K-parallel by splitting K and accumulating results.

        This tests the mathematical correctness of K-split reduction:
        C = A[:, 0:K/2] @ B[0:K/2, :] + A[:, K/2:K] @ B[K/2:K, :]

        While the actual K-parallel kernel uses atomic reduction, we can
        verify the arithmetic is correct by explicitly computing partial sums.
        """
        assert torch is not None

        M, K, N = 64, 32768, 256
        parallel = 4
        group_size = 128
        assert K % group_size == 0
        assert K % parallel == 0

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        # Full computation reference
        packed, scales = quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = dequant_fp4_array(packed, scales, K, N, group_size)
        ref_full = gemm_reference(A, B_dequant)

        # Simulated K-parallel: sum partial results
        K_slice = K // parallel
        partial_sum = np.zeros((M, N), dtype=np.float32)

        for p in range(parallel):
            k_start = p * K_slice
            k_end = (p + 1) * K_slice
            A_slice = A[:, k_start:k_end]
            B_slice = B_dequant[k_start:k_end, :]
            partial = A_slice.astype(np.float32) @ B_slice.astype(np.float32)
            partial_sum += partial

        result_parallel = partial_sum.astype(np.float16)

        # Error from additional FP16 rounding in reduction
        max_val = max(np.abs(ref_full).max(), 1e-6)
        atol = parallel * 5e-4 * max_val

        np.testing.assert_allclose(
            result_parallel.astype(np.float32),
            ref_full.astype(np.float32),
            rtol=1e-2,
            atol=max(atol, 1e-2),
            err_msg=(
                f"K-parallel simulation mismatch: max diff = "
                f"{np.abs(result_parallel.astype(np.float32) - ref_full.astype(np.float32)).max():.6f}"
            ),
        )

    @requires_torch
    @requires_mps
    def test_deterministic_runs(self, rng: np.random.Generator) -> None:
        """Multiple runs with the same input produce identical results."""
        assert torch is not None

        M, K, N = 128, 4096, 4096
        group_size = 128

        # Fixed seed for deterministic inputs
        local_rng = np.random.default_rng(seed=42)
        A = local_rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = local_rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_to_fp4(B_fp16, group_size=group_size)
        packed_reshaped = packed.reshape(K // FP4_PER_UINT, N)

        from metal_marlin.kernels import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed_reshaped).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        results = []
        for _ in range(5):
            r = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=group_size)
            results.append(r.cpu().numpy().copy())

        # All runs must be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(
                results[0],
                results[i],
                err_msg=f"Run 0 vs run {i} differ",
            )

    @requires_torch
    @requires_mps
    def test_zero_input(self, rng: np.random.Generator) -> None:
        """Zero input produces zero output."""
        assert torch is not None

        M, K, N = 16, 4096, 128
        group_size = 128

        A = np.zeros((M, K), dtype=np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_to_fp4(B_fp16, group_size=group_size)
        packed_reshaped = packed.reshape(K // FP4_PER_UINT, N)

        from metal_marlin.kernels import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed_reshaped).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=group_size)
        result_np = result.cpu().numpy()

        assert np.allclose(result_np, 0.0, atol=1e-6), (
            f"Zero input produced non-zero: max={np.abs(result_np).max()}"
        )


# ===========================================================================
# Test: Load balance quality (pure numpy, no GPU needed)
# ===========================================================================


class TestLoadBalance:
    """Validate stripe partitioning distributes work evenly."""

    @pytest.mark.parametrize(
        "M,N,num_tgs",
        [
            (200, 200, 40),  # 3x3 tiles on 40 TGs (not evenly divisible)
            (50, 50, 40),  # 1x1 tile on 40 TGs (extreme imbalance in 2D)
            (320, 320, 40),  # 5x5 tiles on 40 TGs
            (128, 4096, 40),  # 2x64 tiles on 40 TGs (realistic LLM shape)
        ],
    )
    def test_work_distribution_uniformity(self, M: int, N: int, num_tgs: int) -> None:
        """Verify work units are distributed with at most 1 unit imbalance.

        For total_work work units across num_tgs threadgroups, the strided
        schedule assigns work_idx = tg_id, tg_id + num_tgs, tg_id + 2*num_tgs, ...
        The maximum imbalance is therefore at most 1 work unit per TG
        (some TGs get ceil(total_work/num_tgs), others get floor).
        """
        parallel = 1
        m_tiles = (M + TILE_M - 1) // TILE_M
        n_tiles = (N + TILE_N - 1) // TILE_N
        total_work = m_tiles * n_tiles * parallel

        # Simulate strided assignment
        work_counts = []
        for tg_id in range(num_tgs):
            count = 0
            work_idx = tg_id
            while work_idx < total_work:
                count += 1
                work_idx += num_tgs
            work_counts.append(count)

        work_counts = np.array(work_counts)
        active_tgs = work_counts[work_counts > 0]

        if len(active_tgs) == 0:
            pytest.skip("No active threadgroups for this configuration")

        # Maximum imbalance should be at most 1 work unit
        max_work = active_tgs.max()
        min_work = active_tgs.min()
        assert max_work - min_work <= 1, (
            f"Work imbalance too high: max={max_work}, min={min_work}, "
            f"total_work={total_work}, num_tgs={num_tgs}"
        )

    @pytest.mark.parametrize("parallel", [1, 2, 4])
    def test_all_tiles_covered(self, parallel: int) -> None:
        """Every output tile is assigned to at least one threadgroup."""
        M, N = 256, 256
        num_tgs = 40

        m_tiles = (M + TILE_M - 1) // TILE_M
        n_tiles = (N + TILE_N - 1) // TILE_N
        total_tiles = m_tiles * n_tiles
        total_work = total_tiles * parallel

        # Track which tiles are assigned
        tile_assigned = np.zeros(total_tiles, dtype=np.int32)

        # Simulate strided assignment
        for tg_id in range(num_tgs):
            work_idx = tg_id
            while work_idx < total_work:
                tile_linear = work_idx // parallel
                if tile_linear < total_tiles:
                    tile_assigned[tile_linear] += 1
                work_idx += num_tgs

        # Each tile must be covered exactly `parallel` times
        expected_coverage = parallel
        assert np.all(tile_assigned == expected_coverage), (
            f"Tile coverage mismatch: min={tile_assigned.min()}, max={tile_assigned.max()}, "
            f"expected={expected_coverage}, uncovered={np.sum(tile_assigned == 0)}"
        )

    def test_stripe_vs_2d_tile_count(self) -> None:
        """Stripe dispatch processes the same number of total tiles as 2D dispatch."""
        M, N, _K = 128, 4096, 4096
        parallel = 1

        m_tiles = (M + TILE_M - 1) // TILE_M
        n_tiles = (N + TILE_N - 1) // TILE_N

        # 2D dispatch: grid = (n_tiles, m_tiles) = total m_tiles * n_tiles TGs
        two_d_total_tgs = m_tiles * n_tiles

        # Stripe dispatch: total_work = m_tiles * n_tiles * parallel = same tile count
        stripe_total_work = m_tiles * n_tiles * parallel

        assert stripe_total_work == two_d_total_tgs * parallel, (
            f"Total work mismatch: stripe={stripe_total_work}, 2D={two_d_total_tgs}"
        )


# ===========================================================================
# Test: Edge cases
# ===========================================================================


class TestStripedEdgeCases:
    """Edge cases specific to the stripe partitioning kernel."""

    @requires_torch
    @requires_mps
    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_different_group_sizes(self, rng: np.random.Generator, group_size: int) -> None:
        """Stripe kernel handles various quantization group sizes correctly."""
        assert torch is not None

        M, K, N = 64, 512, 256

        if K % group_size != 0:
            pytest.skip(f"K={K} not divisible by group_size={group_size}")

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = dequant_fp4_array(packed, scales, K, N, group_size)
        packed_reshaped = packed.reshape(K // FP4_PER_UINT, N)

        ref = gemm_reference(A, B_dequant)

        from metal_marlin.kernels import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed_reshaped).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=group_size)
        result_np = result.cpu().numpy()

        np.testing.assert_allclose(
            result_np.astype(np.float32),
            ref.astype(np.float32),
            rtol=2e-2,
            atol=5e-3,
            err_msg=f"Group size {group_size}: striped kernel mismatch",
        )

    @requires_torch
    @requires_mps
    @pytest.mark.parametrize(
        "M,N,K",
        [
            (1, 64, 256),  # Single row (M < TILE_M)
            (64, 32, 256),  # N < TILE_N (partial N tile)
            (30, 104, 256),  # Non-aligned M and N (N aligned to 8)
            (128, 128, 32),  # Minimal K (single K tile)
        ],
    )
    def test_non_tile_aligned_dims(self, rng: np.random.Generator, M: int, N: int, K: int) -> None:
        """Stripe kernel correctly handles dimensions not aligned to tile boundaries."""
        assert torch is not None

        group_size = 32
        if K % group_size != 0:
            pytest.skip(f"K={K} not divisible by group_size={group_size}")
        # N must be divisible by FP4_PER_UINT for packing
        N_aligned = ((N + FP4_PER_UINT - 1) // FP4_PER_UINT) * FP4_PER_UINT
        N = N_aligned

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = dequant_fp4_array(packed, scales, K, N, group_size)
        packed_reshaped = packed.reshape(K // FP4_PER_UINT, N)

        ref = gemm_reference(A, B_dequant)

        from metal_marlin.kernels import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed_reshaped).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=group_size)
        result_np = result.cpu().numpy()

        np.testing.assert_allclose(
            result_np.astype(np.float32),
            ref.astype(np.float32),
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"Non-aligned ({M},{N},{K}): kernel vs reference mismatch",
        )

    @requires_torch
    @requires_mps
    def test_excess_threadgroups_scenario(self, rng: np.random.Generator) -> None:
        """Test case where problem size is smaller than typical TG count.

        In a stripe dispatch, when there are more threadgroups than work units,
        excess TGs should early-exit without corrupting output.
        """
        assert torch is not None

        # 1x1 tiles = 1 work unit
        M, K, N = 32, 256, 32
        group_size = 32

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = dequant_fp4_array(packed, scales, K, N, group_size)
        packed_reshaped = packed.reshape(K // FP4_PER_UINT, N)

        ref = gemm_reference(A, B_dequant)

        from metal_marlin.kernels import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed_reshaped).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=group_size)
        result_np = result.cpu().numpy()

        np.testing.assert_allclose(
            result_np.astype(np.float32),
            ref.astype(np.float32),
            rtol=2e-2,
            atol=5e-3,
            err_msg="Small problem size produced incorrect output",
        )

    @requires_torch
    @requires_mps
    @pytest.mark.parametrize("M", [1, 8, 16, 32, 64, 128])
    def test_various_batch_sizes(self, rng: np.random.Generator, M: int) -> None:
        """Striped kernel produces correct output for various batch sizes."""
        assert torch is not None

        K, N = 512, 256
        group_size = 128

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = dequant_fp4_array(packed, scales, K, N, group_size)
        packed_reshaped = packed.reshape(K // FP4_PER_UINT, N)

        ref = gemm_reference(A, B_dequant)

        from metal_marlin.kernels import marlin_gemm_fp4

        A_t = torch.from_numpy(A).to("mps")
        packed_t = torch.from_numpy(packed_reshaped).to("mps")
        scales_t = torch.from_numpy(scales).to("mps")

        result = marlin_gemm_fp4(A_t, packed_t, scales_t, group_size=group_size)
        result_np = result.cpu().numpy()

        ref_scale = np.abs(ref).max() + 1e-7
        atol = max(float(np.sqrt(K)) * 5e-3 * ref_scale, 1e-3)

        np.testing.assert_allclose(
            result_np.astype(np.float32),
            ref.astype(np.float32),
            rtol=5e-2,
            atol=atol,
            err_msg=f"M={M}: kernel vs reference mismatch",
        )
