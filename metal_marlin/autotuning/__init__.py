"""Runtime autotuning for Metal GEMM kernels.

This module provides automatic tile size selection for optimal performance
on different Apple Silicon generations and problem sizes.

Quick Start:
    from metal_marlin.autotuning import get_tuned_config, AutotuneCache

    # Get optimal config (uses cache, runs autotuning if needed)
    config = get_tuned_config(M=4096, N=4096, K=4096)

    # Or use heuristics for instant config selection (no benchmarking)
    from metal_marlin.autotuning import get_heuristic_config
    config = get_heuristic_config(M=4096, N=4096, K=4096)

    # For manual control over autotuning
    from metal_marlin.autotuning import TileSearcher, AutotuneCache

    cache = AutotuneCache()
    searcher = TileSearcher(verbose=True)

    # Search and cache result
    config = searcher.search(M=4096, N=4096, K=4096)
    cache.put(4096, 4096, 4096, config)
    cache.save()

Key Components:
    - TileConfig: Configuration dataclass for tile sizes and threadgroup settings
    - TileSearcher: Grid search over tile configurations
    - AutotuneCache: Persistent cache for tuning results per GPU
    - Heuristics: Fast fallback when benchmarking is not desired

The autotuning system is designed to:
    1. Run once on first use (per problem size)
    2. Cache results persistently (survives restarts)
    3. Key results by GPU fingerprint (different chips get different configs)
    4. Provide fast heuristic fallback when needed
"""

from .cache import (
    AutotuneCache,
    CacheEntry,
    GPUFingerprint,
    detect_gpu,
    get_cache_dir,
    get_global_cache,
    get_tuned_config,
)
from .heuristics import (
    estimate_throughput,
    get_decode_config,
    get_heuristic_config,
    get_prefill_config,
    select_best_heuristic,
)
from .tile_search import (
    DEFAULT_CONFIG,
    GPU_FAMILY_TILE_CONFIG,
    TILE_CONFIGS,
    BenchmarkResult,
    TileConfig,
    TileSearcher,
    common_transformer_sizes,
    sweep_problem_sizes,
)

__all__ = [
    # Core types
    "TileConfig",
    "BenchmarkResult",
    "GPUFingerprint",
    "CacheEntry",
    # Tile search
    "TileSearcher",
    "sweep_problem_sizes",
    "common_transformer_sizes",
    # Cache
    "AutotuneCache",
    "get_global_cache",
    "get_cache_dir",
    "detect_gpu",
    # High-level API
    "get_tuned_config",
    "get_heuristic_config",
    # Specialized configs
    "get_decode_config",
    "get_prefill_config",
    # Analysis
    "estimate_throughput",
    "select_best_heuristic",
    # Constants
    "DEFAULT_CONFIG",
    "GPU_FAMILY_TILE_CONFIG",
    "TILE_CONFIGS",
]
