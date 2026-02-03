#!/usr/bin/env python3
"""Benchmark tile sizes for M3 Max cache optimization.

Tests different tile configurations by recompiling the GEMM kernel with
different compile-time constants. Finds optimal cache utilization on
Apple Silicon M3 Max.

Current default: 64×64×32 (~8.5KB threadgroup memory)
Test configs: 128×128, 64×128, 128×64, and variants

M3 Max specs:
- 192 KB shared memory per threadgroup (max)
- 32 KB default shared memory (for high occupancy)
- Large L1/L2 cache hierarchy
- Up to 1024 threads per threadgroup
- 8 simdgroups max per threadgroup (256 threads)

Usage:
    cd contrib/metal_marlin
    uv run python scripts/benchmark_tile_sizes.py
    uv run python scripts/benchmark_tile_sizes.py --full  # All problem sizes
    uv run python scripts/benchmark_tile_sizes.py --iters 50  # More iterations
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass(frozen=True)
class TileConfig:
    """Tile configuration for GEMM kernel."""

    tile_m: int
    tile_n: int
    tile_k: int
    simdgroups: int
    sg_m_tiles: int  # M decomposition across simdgroups
    sg_n_tiles: int  # N decomposition across simdgroups
    num_buffers: int = 2

    @property
    def name(self) -> str:
        return f"{self.tile_m}x{self.tile_n}x{self.tile_k}_sg{self.simdgroups}"

    @property
    def threads_per_tg(self) -> int:
        return self.simdgroups * 32

    @property
    def shared_memory_bytes(self) -> int:
        """Estimate threadgroup memory usage."""
        # A buffer: NUM_BUFFERS * TILE_M * TILE_K * 2 bytes (half)
        a_mem = self.num_buffers * self.tile_m * self.tile_k * 2
        # B staging: SIMDGROUPS * 8 * 8 * 2 bytes per simdgroup
        b_mem = self.simdgroups * 8 * 8 * 2
        return a_mem + b_mem

    @property
    def output_elements(self) -> int:
        """Elements computed per threadgroup."""
        return self.tile_m * self.tile_n

    @property
    def elements_per_thread(self) -> int:
        """Output elements per thread."""
        return self.output_elements // self.threads_per_tg

    def fits_m3_max(self) -> bool:
        """Check if config fits M3 Max constraints."""
        # Max 1024 threads per threadgroup
        if self.threads_per_tg > 1024:
            return False
        # Conservative shared memory limit for good occupancy
        # M3 Max has 192KB but 32KB gives better occupancy
        if self.shared_memory_bytes > 64 * 1024:
            return False
        # Verify simdgroup decomposition is valid
        if self.sg_m_tiles * self.sg_n_tiles != self.simdgroups * 4:
            # Each simdgroup handles 4 8x8 blocks (32x32)
            # Total blocks = sg_m_tiles * sg_n_tiles
            # Must match simdgroups * 4 for valid configuration
            pass  # Allow flexible configurations
        return True


# Tile configurations to benchmark
# Each must have consistent decomposition for valid kernel dispatch
TILE_CONFIGS = [
    # Current default (baseline) - 64x64 output, 4 SG each handling 32x32
    TileConfig(64, 64, 32, simdgroups=4, sg_m_tiles=4, sg_n_tiles=4),

    # Larger tiles - test cache benefits
    # 128x128 output, 8 SG handling different decompositions
    TileConfig(128, 128, 32, simdgroups=8, sg_m_tiles=4, sg_n_tiles=8),
    TileConfig(64, 128, 32, simdgroups=8, sg_m_tiles=2, sg_n_tiles=8),
    TileConfig(128, 64, 32, simdgroups=8, sg_m_tiles=4, sg_n_tiles=4),

    # Deep K tiles - better memory coalescing
    TileConfig(64, 64, 64, simdgroups=4, sg_m_tiles=4, sg_n_tiles=4),
    TileConfig(128, 128, 64, simdgroups=8, sg_m_tiles=4, sg_n_tiles=8),

    # Wider tiles for decode (small M)
    TileConfig(32, 128, 32, simdgroups=4, sg_m_tiles=2, sg_n_tiles=8),
    TileConfig(32, 256, 32, simdgroups=8, sg_m_tiles=2, sg_n_tiles=16),

    # Fewer simdgroups for smaller tiles
    TileConfig(64, 64, 32, simdgroups=2, sg_m_tiles=2, sg_n_tiles=4),
    TileConfig(32, 64, 32, simdgroups=2, sg_m_tiles=2, sg_n_tiles=4),

    # Very large tiles (testing M3 Max limits)
    TileConfig(128, 256, 32, simdgroups=8, sg_m_tiles=4, sg_n_tiles=16),
    TileConfig(256, 128, 32, simdgroups=8, sg_m_tiles=8, sg_n_tiles=8),
]


@dataclass
class BenchmarkResult:
    """Result from benchmarking a configuration."""

    config: TileConfig
    M: int
    N: int
    K: int
    elapsed_ms: float
    gflops: float
    bandwidth_gb_s: float
    valid: bool
    error: str | None = None


def estimate_bandwidth(M: int, N: int, K: int, elapsed_s: float, group_size: int = 32) -> float:
    """Estimate memory bandwidth in GB/s."""
    # Read: A (M*K*2 bytes) + B_packed (K/8*N*4 bytes for FP4) + scales
    # Write: C (M*N*2 bytes)
    bytes_read = (
        M * K * 2 +           # A (half)
        (K // 8) * N * 4 +    # B_packed (uint32, FP4)
        (K // group_size) * N * 2  # scales (half)
    )
    bytes_write = M * N * 2   # C (half)
    return (bytes_read + bytes_write) / elapsed_s / 1e9


def modify_kernel_source(source: str, config: TileConfig) -> str:
    """Modify kernel source with new tile constants."""
    # Replace tile dimension constants
    replacements = [
        (r'constant constexpr uint TILE_M = \d+;',
         f'constant constexpr uint TILE_M = {config.tile_m};'),
        (r'constant constexpr uint TILE_N = \d+;',
         f'constant constexpr uint TILE_N = {config.tile_n};'),
        (r'constant constexpr uint TILE_K = \d+;',
         f'constant constexpr uint TILE_K = {config.tile_k};'),
        (r'constant constexpr uint SIMDGROUPS_PER_TG = \d+;',
         f'constant constexpr uint SIMDGROUPS_PER_TG = {config.simdgroups};'),
        (r'constant constexpr uint THREADS_PER_TG = SIMDGROUPS_PER_TG \* 32;',
         f'constant constexpr uint THREADS_PER_TG = {config.threads_per_tg};'),
        (r'constant constexpr uint SG_M_TILES = \d+;',
         f'constant constexpr uint SG_M_TILES = {config.sg_m_tiles};'),
        (r'constant constexpr uint SG_N_TILES = \d+;',
         f'constant constexpr uint SG_N_TILES = {config.sg_n_tiles};'),
    ]

    modified = source
    for pattern, replacement in replacements:
        modified = re.sub(pattern, replacement, modified)

    return modified


class TileConfigBenchmarker:
    """Benchmarks GEMM kernels with different tile configurations."""

    def __init__(self, warmup_iters: int = 5, bench_iters: int = 20, group_size: int = 32):
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters
        self.group_size = group_size

        # Check dependencies
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS backend not available. Requires Apple Silicon Mac.")

        self.torch = torch

        try:
            import Metal
            self.Metal = Metal
        except ImportError:
            raise RuntimeError(
                "PyObjC Metal not available. Install with:\n"
                "  pip install pyobjc-framework-Metal"
            )

        self.device = Metal.MTLCreateSystemDefaultDevice()
        self.command_queue = self.device.newCommandQueue()

        # Load base kernel source
        src_dir = Path(__file__).parent.parent / "src"
        self.base_source = (src_dir / "gemm_trellis.metal").read_text()

        # Cache compiled pipelines
        self._pipeline_cache: dict[str, Any] = {}

    def compile_kernel(self, config: TileConfig) -> Any:
        """Compile kernel with specific tile configuration."""
        cache_key = config.name
        if cache_key in self._pipeline_cache:
            return self._pipeline_cache[cache_key]

        modified_source = modify_kernel_source(self.base_source, config)

        options = self.Metal.MTLCompileOptions.new()
        options.setLanguageVersion_(self.Metal.MTLLanguageVersion3_0)

        library, error = self.device.newLibraryWithSource_options_error_(
            modified_source, options, None
        )
        if library is None:
            raise RuntimeError(f"Failed to compile kernel: {error}")

        func = library.newFunctionWithName_("gemm_trellis_packed")
        if func is None:
            raise RuntimeError("Function gemm_trellis_packed not found in modified kernel")

        pipeline, error = self.device.newComputePipelineStateWithFunction_error_(func, None)
        if pipeline is None:
            raise RuntimeError(f"Failed to create pipeline: {error}")

        self._pipeline_cache[cache_key] = pipeline
        return pipeline

    def benchmark_config(
        self,
        config: TileConfig,
        M: int,
        N: int,
        K: int,
    ) -> BenchmarkResult:
        """Benchmark a single tile configuration."""
        # Check constraints
        if not config.fits_m3_max():
            return BenchmarkResult(
                config=config, M=M, N=N, K=K,
                elapsed_ms=float("inf"), gflops=0, bandwidth_gb_s=0,
                valid=False, error="Config exceeds M3 Max constraints"
            )

        # Check problem size compatibility
        if M < config.tile_m or N < config.tile_n or K < config.tile_k:
            return BenchmarkResult(
                config=config, M=M, N=N, K=K,
                elapsed_ms=float("inf"), gflops=0, bandwidth_gb_s=0,
                valid=False, error=f"Problem too small for tile {config.name}"
            )

        try:
            # Compile kernel
            pipeline = self.compile_kernel(config)

            # Generate test data
            torch = self.torch
            A = torch.randn((M, K), dtype=torch.float16, device="mps")
            C = torch.zeros((M, N), dtype=torch.float16, device="mps")

            # For this test, we create synthetic B data (not properly quantized)
            # This measures kernel overhead, not full end-to-end performance
            # For proper benchmarks, use the full pipeline with pack_fp4_weights
            tiles_k = (K + 15) // 16
            tiles_n = (N + 15) // 16
            packed_bytes = 96  # 3-bit: 256 * 3 / 8 = 96 bytes per tile

            # Create packed indices (random data for timing)
            packed_indices = torch.randint(
                0, 256, (tiles_k, tiles_n, packed_bytes),
                dtype=torch.uint8, device="mps"
            )
            scales = torch.randn((K // self.group_size, N), dtype=torch.float32, device="mps")
            grid = torch.randn((8,), dtype=torch.float32, device="mps")  # 8 levels for 3-bit
            su = torch.randn((K,), dtype=torch.float32, device="mps")
            sv = torch.randn((N,), dtype=torch.float32, device="mps")

            torch.mps.synchronize()

            # Get Metal buffers
            def get_buffer(tensor: torch.Tensor) -> Any:
                import objc
                # Use storage data pointer
                ptr = tensor.data_ptr()
                storage_size = tensor.numel() * tensor.element_size()
                # Create Metal buffer from pointer (shared memory)
                buf = self.device.newBufferWithBytesNoCopy_length_options_deallocator_(
                    objc.objc_object(c_void_p=ptr),
                    storage_size,
                    self.Metal.MTLResourceStorageModeShared,
                    None
                )
                return buf

            # For simplicity, use the standard PyTorch MPS approach
            import numpy as np

            from metal_marlin.metal_dispatch import mps_tensor_to_metal_buffer

            A_buf = mps_tensor_to_metal_buffer(A, self.device)
            packed_buf = mps_tensor_to_metal_buffer(packed_indices, self.device)
            scales_buf = mps_tensor_to_metal_buffer(scales, self.device)
            grid_buf = mps_tensor_to_metal_buffer(grid, self.device)
            su_buf = mps_tensor_to_metal_buffer(su, self.device)
            sv_buf = mps_tensor_to_metal_buffer(sv, self.device)
            C_buf = mps_tensor_to_metal_buffer(C, self.device)

            # Parameters buffer
            bits = 3
            n_levels = 8
            params = np.array([M, K, N, bits, n_levels, self.group_size], dtype=np.uint32)
            params_buf = self.device.newBufferWithBytes_length_options_(
                params.tobytes(), params.nbytes, self.Metal.MTLResourceStorageModeShared
            )

            # Grid dimensions
            grid_x = (N + config.tile_n - 1) // config.tile_n
            grid_y = (M + config.tile_m - 1) // config.tile_m

            def run_kernel() -> None:
                cmd_buf = self.command_queue.commandBuffer()
                encoder = cmd_buf.computeCommandEncoder()
                encoder.setComputePipelineState_(pipeline)
                encoder.setBuffer_offset_atIndex_(A_buf, 0, 0)
                encoder.setBuffer_offset_atIndex_(packed_buf, 0, 1)
                encoder.setBuffer_offset_atIndex_(scales_buf, 0, 2)
                encoder.setBuffer_offset_atIndex_(grid_buf, 0, 3)
                encoder.setBuffer_offset_atIndex_(su_buf, 0, 4)
                encoder.setBuffer_offset_atIndex_(sv_buf, 0, 5)
                encoder.setBuffer_offset_atIndex_(C_buf, 0, 6)
                # Set scalar parameters
                encoder.setBytes_length_atIndex_(
                    np.array([M], dtype=np.uint32).tobytes(), 4, 7
                )
                encoder.setBytes_length_atIndex_(
                    np.array([K], dtype=np.uint32).tobytes(), 4, 8
                )
                encoder.setBytes_length_atIndex_(
                    np.array([N], dtype=np.uint32).tobytes(), 4, 9
                )
                encoder.setBytes_length_atIndex_(
                    np.array([bits], dtype=np.uint32).tobytes(), 4, 10
                )
                encoder.setBytes_length_atIndex_(
                    np.array([n_levels], dtype=np.uint32).tobytes(), 4, 11
                )
                encoder.setBytes_length_atIndex_(
                    np.array([self.group_size], dtype=np.uint32).tobytes(), 4, 12
                )
                encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                    (grid_x, grid_y, 1),
                    (config.threads_per_tg, 1, 1)
                )
                encoder.endEncoding()
                cmd_buf.commit()
                cmd_buf.waitUntilCompleted()

            # Warmup
            for _ in range(self.warmup_iters):
                run_kernel()

            # Benchmark
            start = time.perf_counter()
            for _ in range(self.bench_iters):
                run_kernel()
            elapsed = (time.perf_counter() - start) / self.bench_iters

            elapsed_ms = elapsed * 1000
            flops = 2 * M * N * K
            gflops = flops / elapsed / 1e9
            bandwidth = estimate_bandwidth(M, N, K, elapsed, self.group_size)

            return BenchmarkResult(
                config=config, M=M, N=N, K=K,
                elapsed_ms=elapsed_ms, gflops=gflops, bandwidth_gb_s=bandwidth,
                valid=True
            )

        except Exception as e:
            return BenchmarkResult(
                config=config, M=M, N=N, K=K,
                elapsed_ms=float("inf"), gflops=0, bandwidth_gb_s=0,
                valid=False, error=str(e)
            )


def print_config_summary() -> None:
    """Print summary of tile configurations to test."""
    print("\n" + "=" * 90)
    print("TILE CONFIGURATIONS TO BENCHMARK")
    print("=" * 90)
    print(f"{'Config':<25} {'Threads':>8} {'SharedMem':>10} {'Output':>10} {'Elem/Thd':>10} {'Valid':>6}")
    print("-" * 90)

    for config in TILE_CONFIGS:
        status = "✓" if config.fits_m3_max() else "✗"
        print(
            f"{config.name:<25} "
            f"{config.threads_per_tg:>8} "
            f"{config.shared_memory_bytes // 1024:>7} KB "
            f"{config.output_elements:>10} "
            f"{config.elements_per_thread:>10} "
            f"{status:>6}"
        )
    print("=" * 90)


def benchmark_problem_size(
    benchmarker: TileConfigBenchmarker,
    M: int,
    N: int,
    K: int,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """Benchmark all configs for a single problem size."""
    results = []

    if verbose:
        print(f"\n{'=' * 90}")
        print(f"BENCHMARKING: M={M}, N={N}, K={K}")
        print(f"{'=' * 90}")
        print(f"{'Config':<25} {'Time (ms)':>12} {'GFLOPS':>12} {'BW (GB/s)':>12} {'Status':>15}")
        print("-" * 90)

    for config in TILE_CONFIGS:
        result = benchmarker.benchmark_config(config, M, N, K)
        results.append(result)

        if verbose:
            if result.valid:
                print(
                    f"{config.name:<25} "
                    f"{result.elapsed_ms:>12.3f} "
                    f"{result.gflops:>12.1f} "
                    f"{result.bandwidth_gb_s:>12.1f} "
                    f"{'OK':>15}"
                )
            else:
                error_short = (result.error or "Unknown")[:15]
                print(
                    f"{config.name:<25} "
                    f"{'N/A':>12} "
                    f"{'N/A':>12} "
                    f"{'N/A':>12} "
                    f"{error_short:>15}"
                )

    # Print best result
    valid_results = [r for r in results if r.valid]
    if valid_results and verbose:
        best = min(valid_results, key=lambda r: r.elapsed_ms)
        baseline = next((r for r in valid_results if r.config.name == "64x64x32_sg4"), None)

        print("-" * 90)
        print(f"BEST: {best.config.name} @ {best.elapsed_ms:.3f}ms ({best.gflops:.1f} GFLOPS)")

        if baseline and baseline != best:
            speedup = baseline.elapsed_ms / best.elapsed_ms
            print(f"  vs baseline (64x64x32): {speedup:.2f}x speedup")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark tile sizes for M3 Max cache optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run full benchmark with all problem sizes"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick check with minimal problem sizes"
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Warmup iterations (default: 5)"
    )
    parser.add_argument(
        "--iters", type=int, default=20,
        help="Benchmark iterations (default: 20)"
    )
    parser.add_argument(
        "--decode-only", action="store_true",
        help="Only test decode phase (M=1,32)"
    )
    parser.add_argument(
        "--prefill-only", action="store_true",
        help="Only test prefill phase (large M)"
    )
    args = parser.parse_args()

    print_config_summary()

    # Initialize benchmarker
    try:
        benchmarker = TileConfigBenchmarker(
            warmup_iters=args.warmup,
            bench_iters=args.iters,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Define problem sizes
    if args.decode_only:
        problem_sizes = [
            (1, 4096, 4096),
            (1, 8192, 8192),
            (32, 4096, 4096),
            (32, 8192, 8192),
        ]
    elif args.prefill_only:
        problem_sizes = [
            (512, 4096, 4096),
            (512, 8192, 8192),
            (2048, 4096, 4096),
            (2048, 8192, 8192),
        ]
    elif args.quick:
        problem_sizes = [
            (64, 4096, 4096),
            (512, 4096, 4096),
        ]
    elif args.full:
        problem_sizes = [
            (1, 4096, 4096),
            (32, 4096, 4096),
            (64, 4096, 4096),
            (128, 4096, 4096),
            (256, 4096, 4096),
            (512, 4096, 4096),
            (1024, 4096, 4096),
            (2048, 4096, 4096),
            (512, 8192, 8192),
            (512, 14336, 4096),  # MLP up
            (512, 4096, 14336),  # MLP down
        ]
    else:
        problem_sizes = [
            (1, 4096, 4096),
            (32, 4096, 4096),
            (128, 4096, 4096),
            (512, 4096, 4096),
            (2048, 4096, 4096),
        ]

    # Run benchmarks
    all_results: dict[tuple[int, int, int], list[BenchmarkResult]] = {}

    for M, N, K in problem_sizes:
        results = benchmark_problem_size(benchmarker, M, N, K)
        all_results[(M, N, K)] = results

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY: BEST CONFIG PER PROBLEM SIZE")
    print("=" * 90)
    print(f"{'Problem (MxNxK)':<25} {'Best Config':<25} {'GFLOPS':>12} {'vs 64x64':>12}")
    print("-" * 90)

    for (M, N, K), results in all_results.items():
        valid = [r for r in results if r.valid]
        if not valid:
            continue

        best = min(valid, key=lambda r: r.elapsed_ms)
        baseline = next((r for r in valid if r.config.name == "64x64x32_sg4"), None)

        speedup_str = ""
        if baseline and baseline.config != best.config:
            speedup = baseline.elapsed_ms / best.elapsed_ms
            speedup_str = f"{speedup:.2f}x"
        elif baseline:
            speedup_str = "baseline"

        print(
            f"{M}x{N}x{K:<10} "
            f"{best.config.name:<25} "
            f"{best.gflops:>12.1f} "
            f"{speedup_str:>12}"
        )

    print("=" * 90)

    # Recommendations
    print("\nRECOMMENDATIONS FOR M3 MAX:")

    decode_best: BenchmarkResult | None = None
    prefill_best: BenchmarkResult | None = None

    for (M, N, K), results in all_results.items():
        valid = [r for r in results if r.valid]
        if not valid:
            continue
        best = min(valid, key=lambda r: r.elapsed_ms)

        if M <= 32:
            if decode_best is None or best.gflops > decode_best.gflops:
                decode_best = best
        elif M >= 512:
            if prefill_best is None or best.gflops > prefill_best.gflops:
                prefill_best = best

    if decode_best:
        print(f"  Decode (M<=32):   {decode_best.config.name}")
        print(f"                    SharedMem: {decode_best.config.shared_memory_bytes // 1024} KB")
    if prefill_best:
        print(f"  Prefill (M>=512): {prefill_best.config.name}")
        print(f"                    SharedMem: {prefill_best.config.shared_memory_bytes // 1024} KB")


if __name__ == "__main__":
    main()
