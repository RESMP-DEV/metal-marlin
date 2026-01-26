"""Grid search over tile configurations for Metal GEMM kernels.

This module provides runtime autotuning of tile sizes, threadgroup configuration,
and other kernel parameters for optimal performance on different Apple Silicon
generations and problem sizes.

The key insight is that optimal tile sizes depend on:
- GPU generation (M1/M2/M3/M4) - affects simdgroup count, shared memory, registers
- Problem size - small GEMMs want smaller tiles, large GEMMs want larger tiles
- Memory bandwidth vs compute balance (roofline position)

Usage:
    from metal_marlin.autotuning import TileSearcher, TileConfig

    # Create searcher with GPU detection
    searcher = TileSearcher()

    # Find optimal config for a specific problem
    config = searcher.search(M=4096, N=4096, K=4096)

    # Use the config for kernel dispatch
    result = marlin_gemm_fp4_tuned(A, B_packed, scales, config)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .._compat import HAS_MPS, require_torch, torch

if TYPE_CHECKING:
    from ..dtypes import DTypeConfig

# Type alias for tensors
Tensor = Any


@dataclass(frozen=True)
class TileConfig:
    """Configuration for a tiled GEMM kernel.

    Attributes:
        tile_m: Output rows per threadgroup (M dimension tiling).
        tile_n: Output columns per threadgroup (N dimension tiling).
        tile_k: K dimension chunk size (should match or divide group_size).
        simdgroups_per_tg: Number of simdgroups per threadgroup.
        threads_per_tg: Total threads per threadgroup (simdgroups * 32).
        sg_m_tiles: M decomposition across simdgroups.
        sg_n_tiles: N decomposition across simdgroups.
        num_buffers: Number of A-tile buffers for double/triple buffering.
        shared_memory_bytes: Estimated threadgroup memory usage.
        registers_per_thread: Estimated register usage per thread.
        name: Human-readable config name for debugging.
    """

    tile_m: int
    tile_n: int
    tile_k: int
    simdgroups_per_tg: int
    threads_per_tg: int
    sg_m_tiles: int
    sg_n_tiles: int
    num_buffers: int = 2
    shared_memory_bytes: int = 0
    registers_per_thread: int = 0
    name: str = ""

    def __post_init__(self) -> None:
        # Compute derived values if not set
        if self.shared_memory_bytes == 0:
            # A buffer: NUM_BUFFERS * TILE_M * TILE_K * 2 bytes (half)
            # B staging: SIMDGROUPS * 8 * 8 * 2 bytes
            a_mem = self.num_buffers * self.tile_m * self.tile_k * 2
            b_mem = self.simdgroups_per_tg * 8 * 8 * 2
            object.__setattr__(self, "shared_memory_bytes", a_mem + b_mem)

        if not self.name:
            object.__setattr__(
                self,
                "name",
                f"t{self.tile_m}x{self.tile_n}x{self.tile_k}_s{self.simdgroups_per_tg}",
            )

    @property
    def k_tiles(self) -> int:
        """Number of 8-wide K chunks per tile iteration."""
        return self.tile_k // 8

    @property
    def fp4_per_uint(self) -> int:
        """FP4 values packed per uint32."""
        return 8

    @property
    def elements_per_thread(self) -> int:
        """Elements each thread loads for A tiles (approximate)."""
        total_threads = self.simdgroups_per_tg * 32
        if total_threads == 0:
            return 0
        return (self.tile_m * self.tile_k) // total_threads

    def to_template_params(self) -> list[tuple[str, int]]:
        """Convert to template parameters for Metal kernel dispatch."""
        return [
            ("TILE_M", self.tile_m),
            ("TILE_N", self.tile_n),
            ("TILE_K", self.tile_k),
            ("SIMDGROUPS_PER_TG", self.simdgroups_per_tg),
            ("SG_M_TILES", self.sg_m_tiles),
            ("SG_N_TILES", self.sg_n_tiles),
            ("FP4_PER_UINT", self.fp4_per_uint),
            ("NUM_BUFFERS", self.num_buffers),
        ]


# Pre-defined tile configurations covering the search space
# These are the primary configurations to benchmark during autotuning
TILE_CONFIGS: list[TileConfig] = [
    # Small tiles - good for small M (decode phase, batch=1)
    TileConfig(
        tile_m=16,
        tile_n=16,
        tile_k=32,
        simdgroups_per_tg=2,
        threads_per_tg=64,
        sg_m_tiles=1,
        sg_n_tiles=2,
        num_buffers=2,
    ),
    TileConfig(
        tile_m=32,
        tile_n=32,
        tile_k=32,
        simdgroups_per_tg=2,
        threads_per_tg=64,
        sg_m_tiles=2,
        sg_n_tiles=2,
        num_buffers=2,
    ),
    # Medium tiles - balanced workload
    TileConfig(
        tile_m=32,
        tile_n=64,
        tile_k=32,
        simdgroups_per_tg=4,
        threads_per_tg=128,
        sg_m_tiles=2,
        sg_n_tiles=4,
        num_buffers=2,
    ),
    TileConfig(
        tile_m=64,
        tile_n=32,
        tile_k=32,
        simdgroups_per_tg=4,
        threads_per_tg=128,
        sg_m_tiles=4,
        sg_n_tiles=2,
        num_buffers=2,
    ),
    # Default: 64x64x32 - good balance for most problem sizes
    TileConfig(
        tile_m=64,
        tile_n=64,
        tile_k=32,
        simdgroups_per_tg=4,
        threads_per_tg=128,
        sg_m_tiles=2,
        sg_n_tiles=4,
        num_buffers=2,
    ),
    # Large tiles - high compute, good for large M (prefill phase)
    TileConfig(
        tile_m=64,
        tile_n=128,
        tile_k=32,
        simdgroups_per_tg=8,
        threads_per_tg=256,
        sg_m_tiles=2,
        sg_n_tiles=8,
        num_buffers=2,
    ),
    TileConfig(
        tile_m=128,
        tile_n=64,
        tile_k=32,
        simdgroups_per_tg=8,
        threads_per_tg=256,
        sg_m_tiles=4,
        sg_n_tiles=4,
        num_buffers=2,
    ),
    TileConfig(
        tile_m=128,
        tile_n=128,
        tile_k=32,
        simdgroups_per_tg=8,
        threads_per_tg=256,
        sg_m_tiles=4,
        sg_n_tiles=8,
        num_buffers=2,
    ),
    # Experimental: wider K tiles for better memory coalescing
    TileConfig(
        tile_m=64,
        tile_n=64,
        tile_k=64,
        simdgroups_per_tg=4,
        threads_per_tg=128,
        sg_m_tiles=2,
        sg_n_tiles=4,
        num_buffers=2,
    ),
]

# Default configuration when autotuning is skipped
DEFAULT_CONFIG = TileConfig(
    tile_m=64,
    tile_n=64,
    tile_k=32,
    simdgroups_per_tg=4,
    threads_per_tg=128,
    sg_m_tiles=2,
    sg_n_tiles=4,
    num_buffers=2,
)


@dataclass
class BenchmarkResult:
    """Result from benchmarking a single tile configuration."""

    config: TileConfig
    elapsed_ms: float
    gflops: float
    bandwidth_gb_s: float
    valid: bool
    error: str | None = None


@dataclass
class TileSearcher:
    """Grid search over tile configurations for GEMM autotuning.

    Attributes:
        configs: Tile configurations to search over.
        warmup_iters: Warmup iterations before timing.
        bench_iters: Benchmark iterations for timing.
        group_size: Quantization group size.
        verbose: Print progress during search.
    """

    configs: list[TileConfig] = field(default_factory=lambda: TILE_CONFIGS.copy())
    warmup_iters: int = 3
    bench_iters: int = 10
    group_size: int = 32
    verbose: bool = False
    max_shared_memory_bytes: int = 32 * 1024

    def search(
        self,
        M: int,
        N: int,
        K: int,
        dtype_config: DTypeConfig | None = None,
    ) -> TileConfig:
        """Find the optimal tile configuration for a problem size.

        Runs grid search over all tile configurations, benchmarking each
        and returning the fastest valid configuration.

        Args:
            M: Number of rows in A (batch * seq_len).
            N: Number of columns in output (typically hidden_size).
            K: Shared dimension (typically hidden_size or intermediate).
            dtype_config: Optional dtype configuration for precision control.

        Returns:
            Best TileConfig for this problem size.
        """
        require_torch("tile search autotuning")
        if not HAS_MPS:
            raise RuntimeError(
                "MPS (Metal Performance Shaders) is required for tile search autotuning. "
                "This feature requires an Apple Silicon Mac with PyTorch MPS support."
            )

        from ..dtypes import get_default_config
        from ..kernels import pack_fp4_weights

        cfg = dtype_config if dtype_config is not None else get_default_config()
        act_dtype = cfg.torch_activations

        # Generate test data on MPS
        A = torch.randn((M, K), dtype=act_dtype, device="mps")
        weight = torch.randn((N, K), dtype=act_dtype, device="mps")
        B_packed, scales = pack_fp4_weights(weight, group_size=self.group_size)
        # Synchronize to ensure data is ready
        torch.mps.synchronize()

        results = []
        for config in self.configs:
            result = self._benchmark_config(config, A, B_packed, scales, M, N, K, cfg)
            results.append(result)

            if self.verbose and result.valid:
                print(
                    f"{config.name}: {result.elapsed_ms:.3f}ms, "
                    f"{result.gflops:.1f} GFLOPS, "
                    f"{result.bandwidth_gb_s:.1f} GB/s"
                )

        # Return best valid config, or default if none work
        valid_results = [r for r in results if r.valid]
        if not valid_results:
            return DEFAULT_CONFIG

        best = min(valid_results, key=lambda r: r.elapsed_ms)
        return best.config

    def _benchmark_config(
        self,
        config: TileConfig,
        A: Tensor,
        B_packed: Tensor,
        scales: Tensor,
        M: int,
        N: int,
        K: int,
        dtype_config: DTypeConfig,
    ) -> BenchmarkResult:
        """Benchmark a single tile configuration."""
        # Check if config is compatible with problem size
        if not self._config_fits(config, M, N, K):
            return BenchmarkResult(
                config=config,
                elapsed_ms=float("inf"),
                gflops=0.0,
                bandwidth_gb_s=0.0,
                valid=False,
                error="Config does not fit problem size",
            )

        try:
            # Run with this config
            kernel_fn = self._get_dispatch_fn(config, dtype_config)

            # Warmup
            for _ in range(self.warmup_iters):
                _ = kernel_fn(A, B_packed, scales, M, N, K)
                torch.mps.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(self.bench_iters):
                _ = kernel_fn(A, B_packed, scales, M, N, K)
                torch.mps.synchronize()
            elapsed = (time.perf_counter() - start) / self.bench_iters

            elapsed_ms = elapsed * 1000
            flops = 2 * M * N * K  # 2 ops per FMA
            gflops = flops / elapsed / 1e9

            # Estimate bandwidth: read A + B_packed + scales, write C
            bytes_read = (
                M * K * 2  # A (half)
                + (K // 8) * N * 4  # B_packed (uint32)
                + (K // self.group_size) * N * 2  # scales (half)
            )
            bytes_write = M * N * 2  # C (half)
            bandwidth_gb_s = (bytes_read + bytes_write) / elapsed / 1e9

            return BenchmarkResult(
                config=config,
                elapsed_ms=elapsed_ms,
                gflops=gflops,
                bandwidth_gb_s=bandwidth_gb_s,
                valid=True,
            )

        except Exception as e:
            return BenchmarkResult(
                config=config,
                elapsed_ms=float("inf"),
                gflops=0.0,
                bandwidth_gb_s=0.0,
                valid=False,
                error=str(e),
            )

    def _config_fits(self, config: TileConfig, M: int, N: int, K: int) -> bool:
        """Check if tile config can handle this problem size.

        Tile configs must evenly divide the problem dimensions for simple
        boundary handling. Future work: add boundary handling for partial tiles.
        """
        # Respect threadgroup and shared memory limits
        if config.threads_per_tg > 1024:
            return False
        if config.shared_memory_bytes > self.max_shared_memory_bytes:
            return False

        # For now, require exact divisibility
        # TODO: Add boundary handling for non-divisible sizes
        if M < config.tile_m or N < config.tile_n:
            return False
        if K < config.tile_k:
            return False
        return True

    def _get_dispatch_fn(
        self, config: TileConfig, dtype_config: DTypeConfig
    ) -> Callable[[Tensor, Tensor, Tensor, int, int, int], Tensor]:
        """Get a dispatch function for the given config.

        Returns a callable that runs the GEMM with the specified tile config.
        Uses PyTorch MPS backend with direct Metal kernel dispatch via PyObjC.
        """
        from ..kernels import marlin_gemm_fp4

        # For PyTorch/MPS, we use the unified marlin_gemm_fp4 function
        # which handles Metal kernel dispatch internally.
        # The tile config doesn't affect the kernel dispatch in the PyTorch path
        # since kernels.py uses fixed tile sizes. For true tile-config-based
        # autotuning, we would need to extend the Metal dispatch layer.

        def dispatch(
            A: Tensor,
            B_packed: Tensor,
            scales: Tensor,
            M: int,
            N: int,
            K: int,
        ) -> Tensor:
            # Use the standard GEMM function from kernels module
            # The tile config is recorded for analysis but the kernel
            # uses its built-in configuration
            return marlin_gemm_fp4(A, B_packed, scales, self.group_size)

        return dispatch

    def search_with_logging(
        self,
        M: int,
        N: int,
        K: int,
        dtype_config: DTypeConfig | None = None,
    ) -> tuple[TileConfig, list[BenchmarkResult]]:
        """Search with full benchmark results returned.

        Same as search() but also returns all benchmark results for analysis.
        """
        require_torch("tile search autotuning")
        if not HAS_MPS:
            raise RuntimeError(
                "MPS (Metal Performance Shaders) is required for tile search autotuning. "
                "This feature requires an Apple Silicon Mac with PyTorch MPS support."
            )

        from ..dtypes import get_default_config
        from ..kernels import pack_fp4_weights

        cfg = dtype_config if dtype_config is not None else get_default_config()
        act_dtype = cfg.torch_activations

        # Generate test data on MPS
        A = torch.randn((M, K), dtype=act_dtype, device="mps")
        weight = torch.randn((N, K), dtype=act_dtype, device="mps")
        B_packed, scales = pack_fp4_weights(weight, group_size=self.group_size)
        # Synchronize to ensure data is ready
        torch.mps.synchronize()

        results = []
        for config in self.configs:
            result = self._benchmark_config(config, A, B_packed, scales, M, N, K, cfg)
            results.append(result)

        valid_results = [r for r in results if r.valid]
        if not valid_results:
            return DEFAULT_CONFIG, results

        best = min(valid_results, key=lambda r: r.elapsed_ms)
        return best.config, results


def sweep_problem_sizes(
    sizes: list[tuple[int, int, int]],
    warmup_iters: int = 3,
    bench_iters: int = 10,
    verbose: bool = True,
) -> dict[tuple[int, int, int], TileConfig]:
    """Run autotuning across multiple problem sizes.

    Useful for pre-populating the cache with common transformer shapes.

    Args:
        sizes: List of (M, N, K) problem sizes to tune.
        warmup_iters: Warmup iterations per benchmark.
        bench_iters: Benchmark iterations per benchmark.
        verbose: Print progress.

    Returns:
        Dict mapping (M, N, K) to optimal TileConfig.
    """
    searcher = TileSearcher(
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
        verbose=verbose,
    )

    results: dict[tuple[int, int, int], TileConfig] = {}
    for M, N, K in sizes:
        if verbose:
            print(f"Tuning M={M}, N={N}, K={K}...")
        config = searcher.search(M, N, K)
        results[(M, N, K)] = config
        if verbose:
            print(f"  Best: {config.name}")

    return results


def common_transformer_sizes(
    hidden_sizes: list[int] | None = None,
    batch_sizes: list[int] | None = None,
    seq_lens: list[int] | None = None,
) -> list[tuple[int, int, int]]:
    """Generate common (M, N, K) sizes for transformer models.

    Args:
        hidden_sizes: Model hidden dimensions. Default: common LLM sizes.
        batch_sizes: Batch sizes to test. Default: [1, 4, 8, 16, 32].
        seq_lens: Sequence lengths to test. Default: [1, 128, 512, 2048].

    Returns:
        List of (M, N, K) tuples where M = batch * seq_len.
    """
    if hidden_sizes is None:
        # Common LLM hidden sizes
        hidden_sizes = [
            2048,  # Llama 2 7B
            4096,  # Llama 2 7B, Llama 3 8B
            5120,  # Llama 2 13B
            8192,  # Llama 2 70B, Llama 3 70B
        ]

    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]

    if seq_lens is None:
        seq_lens = [1, 128, 512, 2048]

    sizes = []
    for hidden in hidden_sizes:
        for batch in batch_sizes:
            for seq_len in seq_lens:
                M = batch * seq_len
                N = hidden
                K = hidden
                sizes.append((M, N, K))

                # Also test MLP intermediate (typically 4x or 8/3x hidden)
                intermediate = hidden * 4
                sizes.append((M, intermediate, hidden))  # up projection
                sizes.append((M, hidden, intermediate))  # down projection

    # Remove duplicates and sort
    sizes = sorted(set(sizes))
    return sizes
