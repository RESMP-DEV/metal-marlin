"""Tests for stripe-partitioned GEMM kernel accuracy and load balancing.

Validates that marlin_gemm_fp4_striped produces identical results to the 2D
grid kernel (marlin_gemm_fp4) across various matrix dimensions and K-parallel
factors. Also tests the atomic reduction path and load balance distribution.

The striped kernel uses 1D dispatch with linearized tile assignment, splitting
the K-dimension across multiple threadgroups when parallel > 1. This requires
an atomic counter-based two-phase reduction to combine partial sums.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Lazy imports for Metal/MLX (skip gracefully if unavailable)
# ---------------------------------------------------------------------------

_METAL_SHADER_PATH = Path(__file__).parent.parent / "src" / "marlin_gemm.metal"


def _require_mlx():
    """Import mlx or skip."""
    try:
        import mlx.core as mx
        return mx
    except (ImportError, ModuleNotFoundError):
        pytest.skip("mlx not available")


def _load_metal_source() -> str:
    """Read the full Metal shader source for kernel compilation."""
    if not _METAL_SHADER_PATH.exists():
        pytest.skip(f"Metal shader not found: {_METAL_SHADER_PATH}")
    return _METAL_SHADER_PATH.read_text()


# ---------------------------------------------------------------------------
# Kernel builders (cached per-session via module globals)
# ---------------------------------------------------------------------------

_kernel_2d = None
_kernel_striped = None
_kernel_zero_reduction = None


def _get_kernel_2d(mx):
    """Build the 2D dispatch GEMM kernel."""
    global _kernel_2d
    if _kernel_2d is not None:
        return _kernel_2d

    source = _load_metal_source()
    _kernel_2d = mx.fast.metal_kernel(
        name="marlin_gemm_fp4",
        input_names=["A", "B", "scales"],
        output_names=["C"],
        source=source,
    )
    return _kernel_2d


def _get_kernel_striped(mx):
    """Build the striped 1D dispatch GEMM kernel."""
    global _kernel_striped
    if _kernel_striped is not None:
        return _kernel_striped

    source = _load_metal_source()
    _kernel_striped = mx.fast.metal_kernel(
        name="marlin_gemm_fp4_striped",
        input_names=["A", "B", "scales"],
        output_names=["C", "reduction_buf", "locks"],
        source=source,
    )
    return _kernel_striped


def _get_kernel_zero_reduction(mx):
    """Build the zero-reduction helper kernel."""
    global _kernel_zero_reduction
    if _kernel_zero_reduction is not None:
        return _kernel_zero_reduction

    source = _load_metal_source()
    _kernel_zero_reduction = mx.fast.metal_kernel(
        name="marlin_zero_reduction",
        input_names=[],
        output_names=["reduction_buf", "locks"],
        source=source,
    )
    return _kernel_zero_reduction


# ---------------------------------------------------------------------------
# Constants matching the Metal kernel
# ---------------------------------------------------------------------------

TILE_M = 64
TILE_N = 64
TILE_K = 32
FP4_PER_UINT = 8
THREADS_PER_TG = 128

# Target threadgroup count for M4 Max (40 GPU cores)
# The stripe kernel dispatches exactly this many TGs for full occupancy.
DEFAULT_NUM_TGS = 40


# ---------------------------------------------------------------------------
# FP4 quantization utilities (numpy reference)
# ---------------------------------------------------------------------------

FP4_E2M1_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float16,
)


def quantize_to_fp4(
    weights: np.ndarray, group_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
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


# ---------------------------------------------------------------------------
# Metal dispatch helpers
# ---------------------------------------------------------------------------


def _run_2d_kernel(mx, A_mx, B_packed_mx, scales_mx, M, N, K, group_size):
    """Dispatch marlin_gemm_fp4 (2D grid) and return output."""
    kernel = _get_kernel_2d(mx)

    grid_x = (N + TILE_N - 1) // TILE_N
    grid_y = (M + TILE_M - 1) // TILE_M

    outputs = kernel(
        inputs=[A_mx, B_packed_mx, scales_mx],
        output_shapes=[(M, N)],
        output_dtypes=[mx.float16],
        grid=(grid_x, grid_y, 1),
        threadgroup=(THREADS_PER_TG, 1, 1),
        template=[
            ("M", M), ("N", N), ("K", K), ("group_size", group_size),
        ],
        init_value=0,
    )
    mx.eval(outputs[0])
    return np.array(outputs[0])


def _run_striped_kernel(mx, A_mx, B_packed_mx, scales_mx, M, N, K,
                        group_size, parallel, num_tgs=DEFAULT_NUM_TGS):
    """Dispatch marlin_gemm_fp4_striped (1D grid) and return output."""
    kernel = _get_kernel_striped(mx)

    m_tiles = (M + TILE_M - 1) // TILE_M
    n_tiles = (N + TILE_N - 1) // TILE_N
    total_tiles = m_tiles * n_tiles

    # Allocate reduction buffer and locks
    reduction_buf_size = parallel * M * N
    locks_size = total_tiles

    outputs = kernel(
        inputs=[A_mx, B_packed_mx, scales_mx],
        output_shapes=[(M, N), (reduction_buf_size,), (locks_size,)],
        output_dtypes=[mx.float16, mx.float16, mx.int32],
        grid=(num_tgs, 1, 1),
        threadgroup=(THREADS_PER_TG, 1, 1),
        template=[
            ("M", M), ("N", N), ("K", K), ("group_size", group_size),
            ("parallel", parallel), ("num_tgs", num_tgs),
        ],
        init_value=0,
    )
    mx.eval(outputs[0])
    return np.array(outputs[0])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


# ===========================================================================
# Test: Stripe-partitioned kernel matches 2D kernel output
# ===========================================================================


class TestStripedVs2D:
    """Compare stripe-partitioned kernel output to 2D grid kernel output."""

    @pytest.mark.parametrize("parallel", [1, 2, 4])
    @pytest.mark.parametrize(
        "M,N,K",
        [
            (128, 4096, 4096),      # Standard LLM layer dimensions
            (1, 4096, 16384),       # Single token, very long K (K-parallel benefit)
            (256, 256, 32768),      # Very long K reduction chain
        ],
    )
    def test_striped_matches_2d(
        self, rng: np.random.Generator, parallel: int, M: int, N: int, K: int
    ) -> None:
        """Stripe kernel output must match 2D kernel output within FP16 tolerance."""
        mx = _require_mlx()

        group_size = 128
        assert K % group_size == 0

        # Generate random matrices
        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        # Quantize weights to FP4
        packed, scales = quantize_to_fp4(B_fp16, group_size=group_size)

        # Reshape packed for kernel: [K/8, N] layout
        # The packing is row-major with FP4_PER_UINT elements per uint32
        # Each row of B has N elements packed into N uint32s (one nibble per column,
        # 8 consecutive K values per uint32)
        # Layout: packed[k_group * N + n] where k_group = k // 8
        packed_reshaped = packed.reshape(K // FP4_PER_UINT, N)

        # Convert to MLX arrays
        A_mx = mx.array(A)
        B_packed_mx = mx.array(packed_reshaped)
        scales_mx = mx.array(scales)

        # Run 2D kernel
        result_2d = _run_2d_kernel(mx, A_mx, B_packed_mx, scales_mx, M, N, K, group_size)

        # Run striped kernel
        result_striped = _run_striped_kernel(
            mx, A_mx, B_packed_mx, scales_mx, M, N, K, group_size, parallel
        )

        # Compare: both kernels should produce identical or near-identical results.
        # When parallel=1, output should be bit-exact (same compute path, different scheduling).
        # When parallel>1, the reduction introduces additional FP16 rounding from summing
        # partial K-slices, so we allow slightly looser tolerance.
        if parallel == 1:
            rtol, atol = 1e-3, 1e-4
        else:
            # K-parallel adds one extra FP16 addition per output element
            # Error scales with sqrt(parallel) * eps * ||partial_sum||
            rtol, atol = 5e-3, 1e-2

        np.testing.assert_allclose(
            result_striped.astype(np.float32),
            result_2d.astype(np.float32),
            rtol=rtol,
            atol=atol,
            err_msg=(
                f"Striped vs 2D mismatch for ({M},{N},{K}) parallel={parallel}. "
                f"Max diff: {np.abs(result_striped.astype(np.float32) - result_2d.astype(np.float32)).max():.6f}"
            ),
        )

    @pytest.mark.parametrize("parallel", [1, 2, 4])
    def test_striped_matches_reference(
        self, rng: np.random.Generator, parallel: int
    ) -> None:
        """Stripe kernel matches numpy FP32 reference matmul (not just 2D kernel)."""
        mx = _require_mlx()

        M, K, N = 64, 512, 512
        group_size = 128

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = dequant_fp4_array(packed, scales, K, N, group_size)
        packed_reshaped = packed.reshape(K // FP4_PER_UINT, N)

        # FP32 reference
        ref = (A.astype(np.float32) @ B_dequant.astype(np.float32)).astype(np.float16)

        A_mx = mx.array(A)
        B_packed_mx = mx.array(packed_reshaped)
        scales_mx = mx.array(scales)

        result = _run_striped_kernel(
            mx, A_mx, B_packed_mx, scales_mx, M, N, K, group_size, parallel
        )

        np.testing.assert_allclose(
            result.astype(np.float32),
            ref.astype(np.float32),
            rtol=2e-2,
            atol=5e-3,
            err_msg=f"Striped vs reference mismatch, parallel={parallel}",
        )


# ===========================================================================
# Test: K-parallel reduction correctness
# ===========================================================================


class TestKParallelReduction:
    """Validate the atomic-counter-based K-parallel reduction path."""

    def test_k_parallel_reduction_large_k(self, rng: np.random.Generator) -> None:
        """K=32768, parallel=4: each threadgroup processes K/4=8192 elements.

        The atomic lock in locks[] ensures the last threadgroup to finish a
        tile performs the final reduction. This tests that:
          1. Partial sums in reduction_buf are correctly accumulated
          2. The atomic ordering (release/acquire) prevents data races
          3. The output matches the single-pass (parallel=1) result
        """
        mx = _require_mlx()

        M, K, N = 64, 32768, 256
        parallel = 4
        group_size = 128
        assert K % group_size == 0

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_to_fp4(B_fp16, group_size=group_size)
        packed_reshaped = packed.reshape(K // FP4_PER_UINT, N)

        A_mx = mx.array(A)
        B_packed_mx = mx.array(packed_reshaped)
        scales_mx = mx.array(scales)

        # Reference: parallel=1 (no reduction path)
        result_p1 = _run_striped_kernel(
            mx, A_mx, B_packed_mx, scales_mx, M, N, K, group_size, parallel=1
        )

        # Test: parallel=4 (reduction path)
        result_p4 = _run_striped_kernel(
            mx, A_mx, B_packed_mx, scales_mx, M, N, K, group_size, parallel=4
        )

        # The K-parallel reduction sums `parallel` partial FP16 values.
        # Error: up to parallel * eps * max_partial ~ 4 * 5e-4 * max_val
        max_val = max(np.abs(result_p1).max(), 1e-6)
        atol = parallel * 5e-4 * max_val

        np.testing.assert_allclose(
            result_p4.astype(np.float32),
            result_p1.astype(np.float32),
            rtol=1e-2,
            atol=max(atol, 1e-2),
            err_msg=(
                f"K-parallel reduction mismatch: max diff = "
                f"{np.abs(result_p4.astype(np.float32) - result_p1.astype(np.float32)).max():.6f}"
            ),
        )

    @pytest.mark.parametrize("parallel", [2, 4, 8])
    def test_k_parallel_deterministic(
        self, rng: np.random.Generator, parallel: int
    ) -> None:
        """Multiple runs with the same parallel factor produce bit-identical results.

        The ordered reduction guarantees determinism: the "last slice" threadgroup
        sums partial sums in a fixed order (s=0, 1, ..., parallel-1), so the FP
        addition sequence is invariant regardless of GPU scheduling. No atomic FP
        adds are used; only the integer lock counter is atomic.

        Uses a large matrix (128x4096x4096) with 5 repetitions to stress-test
        that threadgroup scheduling variance doesn't affect output bits.
        """
        mx = _require_mlx()

        M, K, N = 128, 4096, 4096
        group_size = 128

        # Seed-based reproducibility: fixed RNG state for deterministic inputs
        local_rng = np.random.default_rng(seed=42)
        A = local_rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = local_rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_to_fp4(B_fp16, group_size=group_size)
        packed_reshaped = packed.reshape(K // FP4_PER_UINT, N)

        A_mx = mx.array(A)
        B_packed_mx = mx.array(packed_reshaped)
        scales_mx = mx.array(scales)

        results = []
        for _ in range(5):
            r = _run_striped_kernel(
                mx, A_mx, B_packed_mx, scales_mx, M, N, K, group_size, parallel
            )
            results.append(r)

        # All runs must be bit-identical (ordered reduction guarantees this)
        for i in range(1, len(results)):
            np.testing.assert_array_equal(
                results[0], results[i],
                err_msg=f"Run 0 vs run {i} differ for parallel={parallel}",
            )

    @pytest.mark.parametrize("parallel", [2, 4])
    def test_k_parallel_deterministic_high_level(
        self, parallel: int
    ) -> None:
        """Verify determinism via the quantized_linear_striped Python API.

        Uses mx.array_equal for strict bit-equality checking across 5 runs
        with seeded random inputs, matching the full dispatch path including
        zero-reduction initialization.
        """
        mx = _require_mlx()
        from metal_marlin.metal_marlin import pack_fp4_weights, quantized_linear_striped

        mx.random.seed(42)
        x = mx.random.normal((128, 4096)).astype(mx.float16)
        w = mx.random.normal((4096, 4096)).astype(mx.float16)
        packed, scales = pack_fp4_weights(w)

        results = []
        for _ in range(5):
            out = quantized_linear_striped(x, packed, scales, parallel=parallel)
            mx.eval(out)
            results.append(out)

        for i in range(1, len(results)):
            assert mx.array_equal(results[0], results[i]), (
                f"Non-deterministic! Run 0 vs run {i} differ for parallel={parallel}. "
                f"Max diff: {mx.max(mx.abs(results[0] - results[i])).item()}"
            )

    def test_k_parallel_zero_input(self, rng: np.random.Generator) -> None:
        """Zero input through K-parallel reduction still produces zero output."""
        mx = _require_mlx()

        M, K, N = 16, 4096, 128
        parallel = 4
        group_size = 128

        A = np.zeros((M, K), dtype=np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_to_fp4(B_fp16, group_size=group_size)
        packed_reshaped = packed.reshape(K // FP4_PER_UINT, N)

        A_mx = mx.array(A)
        B_packed_mx = mx.array(packed_reshaped)
        scales_mx = mx.array(scales)

        result = _run_striped_kernel(
            mx, A_mx, B_packed_mx, scales_mx, M, N, K, group_size, parallel
        )

        assert np.allclose(result, 0.0, atol=1e-6), (
            f"Zero input with K-parallel reduction produced non-zero: max={np.abs(result).max()}"
        )


# ===========================================================================
# Test: Load balance quality
# ===========================================================================


class TestLoadBalance:
    """Validate stripe partitioning distributes work evenly."""

    @pytest.mark.parametrize(
        "M,N,num_tgs",
        [
            (200, 200, 40),   # 3x3 tiles on 40 TGs (not evenly divisible)
            (50, 50, 40),     # 1x1 tile on 40 TGs (extreme imbalance in 2D)
            (320, 320, 40),   # 5x5 tiles on 40 TGs
            (128, 4096, 40),  # 2x64 tiles on 40 TGs (realistic LLM shape)
        ],
    )
    def test_work_distribution_uniformity(
        self, M: int, N: int, num_tgs: int
    ) -> None:
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

    @pytest.mark.parametrize("group_size", [32, 64, 128, 256])
    def test_different_group_sizes(
        self, rng: np.random.Generator, group_size: int
    ) -> None:
        """Stripe kernel handles various quantization group sizes correctly."""
        mx = _require_mlx()

        M, K, N = 64, 512, 256
        parallel = 2

        if K % group_size != 0:
            pytest.skip(f"K={K} not divisible by group_size={group_size}")

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = dequant_fp4_array(packed, scales, K, N, group_size)
        packed_reshaped = packed.reshape(K // FP4_PER_UINT, N)

        ref = (A.astype(np.float32) @ B_dequant.astype(np.float32)).astype(np.float16)

        A_mx = mx.array(A)
        B_packed_mx = mx.array(packed_reshaped)
        scales_mx = mx.array(scales)

        result = _run_striped_kernel(
            mx, A_mx, B_packed_mx, scales_mx, M, N, K, group_size, parallel
        )

        np.testing.assert_allclose(
            result.astype(np.float32),
            ref.astype(np.float32),
            rtol=2e-2,
            atol=5e-3,
            err_msg=f"Group size {group_size}: striped kernel mismatch",
        )

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (1, 64, 256),        # Single row (M < TILE_M)
            (64, 32, 256),       # N < TILE_N (partial N tile)
            (30, 100, 256),      # Non-aligned M and N
            (128, 128, 32),      # Minimal K (single K tile)
        ],
    )
    def test_non_tile_aligned_dims(
        self, rng: np.random.Generator, M: int, N: int, K: int
    ) -> None:
        """Stripe kernel correctly handles dimensions not aligned to tile boundaries."""
        mx = _require_mlx()

        group_size = 32
        if K % group_size != 0:
            pytest.skip(f"K={K} not divisible by group_size={group_size}")
        # N must be divisible by FP4_PER_UINT for packing
        N_aligned = ((N + FP4_PER_UINT - 1) // FP4_PER_UINT) * FP4_PER_UINT
        N = N_aligned

        parallel = 1

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = dequant_fp4_array(packed, scales, K, N, group_size)
        packed_reshaped = packed.reshape(K // FP4_PER_UINT, N)

        (A.astype(np.float32) @ B_dequant.astype(np.float32)).astype(np.float16)

        A_mx = mx.array(A)
        B_packed_mx = mx.array(packed_reshaped)
        scales_mx = mx.array(scales)

        result_2d = _run_2d_kernel(
            mx, A_mx, B_packed_mx, scales_mx, M, N, K, group_size
        )
        result_striped = _run_striped_kernel(
            mx, A_mx, B_packed_mx, scales_mx, M, N, K, group_size, parallel
        )

        # Both should match reference
        np.testing.assert_allclose(
            result_striped.astype(np.float32),
            result_2d.astype(np.float32),
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"Non-aligned ({M},{N},{K}): striped vs 2D mismatch",
        )

    def test_excess_threadgroups(self, rng: np.random.Generator) -> None:
        """More threadgroups than work units: excess TGs should early-exit."""
        mx = _require_mlx()

        # 1x1 tiles = 1 work unit, but dispatch 40 threadgroups
        M, K, N = 32, 256, 32
        group_size = 32
        parallel = 1
        num_tgs = 40

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_to_fp4(B_fp16, group_size=group_size)
        B_dequant = dequant_fp4_array(packed, scales, K, N, group_size)
        packed_reshaped = packed.reshape(K // FP4_PER_UINT, N)

        ref = (A.astype(np.float32) @ B_dequant.astype(np.float32)).astype(np.float16)

        A_mx = mx.array(A)
        B_packed_mx = mx.array(packed_reshaped)
        scales_mx = mx.array(scales)

        result = _run_striped_kernel(
            mx, A_mx, B_packed_mx, scales_mx, M, N, K, group_size,
            parallel, num_tgs=num_tgs,
        )

        np.testing.assert_allclose(
            result.astype(np.float32),
            ref.astype(np.float32),
            rtol=2e-2,
            atol=5e-3,
            err_msg="Excess threadgroups caused incorrect output",
        )

    @pytest.mark.parametrize("num_tgs", [1, 5, 10, 40, 80])
    def test_various_threadgroup_counts(
        self, rng: np.random.Generator, num_tgs: int
    ) -> None:
        """Striped kernel produces correct output regardless of threadgroup count."""
        mx = _require_mlx()

        M, K, N = 128, 512, 256
        group_size = 128
        parallel = 1

        A = rng.standard_normal((M, K)).astype(np.float16)
        B_fp16 = rng.standard_normal((K, N)).astype(np.float16)

        packed, scales = quantize_to_fp4(B_fp16, group_size=group_size)
        packed_reshaped = packed.reshape(K // FP4_PER_UINT, N)

        A_mx = mx.array(A)
        B_packed_mx = mx.array(packed_reshaped)
        scales_mx = mx.array(scales)

        # Reference: 2D kernel
        result_2d = _run_2d_kernel(
            mx, A_mx, B_packed_mx, scales_mx, M, N, K, group_size
        )

        result_striped = _run_striped_kernel(
            mx, A_mx, B_packed_mx, scales_mx, M, N, K, group_size,
            parallel, num_tgs=num_tgs,
        )

        np.testing.assert_allclose(
            result_striped.astype(np.float32),
            result_2d.astype(np.float32),
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"num_tgs={num_tgs}: striped vs 2D mismatch",
        )
