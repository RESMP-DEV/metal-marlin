"""Heuristic-based tile selection for fast fallback.

When autotuning data is not available or the overhead of benchmarking is
unacceptable, these heuristics provide reasonable tile configurations based
on problem size and GPU characteristics.

The heuristics are designed to:
1. Be fast (no kernel execution, just arithmetic)
2. Be reasonably good (within 80-90% of optimal)
3. Handle edge cases (small M, large K, etc.)

Usage:
    from metal_marlin.autotuning import get_heuristic_config

    # Fast config selection without benchmarking
    config = get_heuristic_config(M=4096, N=4096, K=4096)

    # With GPU awareness
    config = get_heuristic_config(M=4096, N=4096, K=4096, gpu_cores=38)
"""

from __future__ import annotations

import re

from .cache import GPUFingerprint, detect_gpu
from .tile_search import DEFAULT_CONFIG, GPU_FAMILY_TILE_CONFIG, TILE_CONFIGS, TileConfig


def get_heuristic_config(
    M: int,
    N: int,
    K: int,
    group_size: int = 32,
    gpu_cores: int | None = None,
    memory_gb: int | None = None,
) -> TileConfig:
    """Select a tile configuration using heuristics.

    This provides a fast fallback when autotuning data is not available.
    The heuristics are based on empirical observations:

    1. Small M (batch=1 decode): Use smaller tiles for better thread utilization
    2. Large M (prefill): Use larger tiles for better compute efficiency
    3. More GPU cores: Can handle larger tiles with better occupancy
    4. Memory-bound: Prefer configurations with better memory coalescing

    Args:
        M: Number of rows (batch_size * seq_len).
        N: Number of columns (typically hidden_size).
        K: Shared dimension (typically hidden_size or intermediate).
        group_size: Quantization group size.
        gpu_cores: GPU core count. Auto-detected if None.
        memory_gb: Total GPU memory in GB. Auto-detected if None.

    Returns:
        Heuristically selected TileConfig.
    """
    # Auto-detect GPU if not specified
    if gpu_cores is None or memory_gb is None:
        gpu = detect_gpu()
        gpu_cores = gpu_cores or gpu.cores
        memory_gb = memory_gb or gpu.memory_gb
    else:
        gpu = detect_gpu()

    family_id = _gpu_family_id(gpu)

    # Classify problem size
    problem_class = _classify_problem(M, N, K)

    # M3+ favors larger tiles and arithmetic intensity over occupancy.
    if family_id >= 9 and problem_class in {"medium", "large", "xlarge"}:
        m3_config = _select_m3plus_config(M, N, K)
        if m3_config is not None:
            return m3_config

    # Select configuration based on problem class and hardware
    if problem_class == "tiny":
        return _heuristic_tiny(M, N, K, gpu_cores)
    elif problem_class == "small":
        return _heuristic_small(M, N, K, gpu_cores)
    elif problem_class == "medium":
        return _heuristic_medium(M, N, K, gpu_cores)
    elif problem_class == "large":
        return _heuristic_large(M, N, K, gpu_cores)
    else:  # xlarge
        return _heuristic_xlarge(M, N, K, gpu_cores)


def _classify_problem(M: int, N: int, K: int) -> str:
    """Classify problem size into categories.

    Categories based on total compute (M * N * K):
    - tiny: < 1M ops (single token decode on small models)
    - small: 1M - 64M ops (batch decode, small prefill)
    - medium: 64M - 1B ops (typical inference)
    - large: 1B - 16B ops (large batch or long context)
    - xlarge: > 16B ops (very large batches)
    """
    total_ops = M * N * K

    if total_ops < 1_000_000:
        return "tiny"
    elif total_ops < 64_000_000:
        return "small"
    elif total_ops < 1_000_000_000:
        return "medium"
    elif total_ops < 16_000_000_000:
        return "large"
    else:
        return "xlarge"


def _heuristic_tiny(M: int, N: int, K: int, gpu_cores: int) -> TileConfig:
    """Heuristic for tiny problems (< 1M ops).

    For very small problems, use the smallest tiles to maximize thread
    utilization. The kernel launch overhead dominates anyway.
    """
    # Use 16x16x32 for tiny problems
    return TileConfig(
        tile_m=16,
        tile_n=16,
        tile_k=32,
        simdgroups_per_tg=2,
        threads_per_tg=64,
        sg_m_tiles=1,
        sg_n_tiles=2,
        num_buffers=2,
    )


def _heuristic_small(M: int, N: int, K: int, gpu_cores: int) -> TileConfig:
    """Heuristic for small problems (1M - 64M ops).

    Typical batch=1 decode or small batch sizes. Want enough parallelism
    to saturate GPU cores but not so much that we waste threads.
    """
    # Estimate grid size for different tile configs
    # We want at least gpu_cores * 2 threadgroups for good occupancy

    if M < 64:
        # Very small M (single token or very small batch)
        # Use tall tiles (more M per threadgroup) to reduce grid overhead
        return TileConfig(
            tile_m=32,
            tile_n=32,
            tile_k=32,
            simdgroups_per_tg=2,
            threads_per_tg=64,
            sg_m_tiles=2,
            sg_n_tiles=2,
            num_buffers=2,
        )
    else:
        # M is larger, can use wider tiles
        return TileConfig(
            tile_m=32,
            tile_n=64,
            tile_k=32,
            simdgroups_per_tg=4,
            threads_per_tg=128,
            sg_m_tiles=2,
            sg_n_tiles=4,
            num_buffers=2,
        )


def _heuristic_medium(M: int, N: int, K: int, gpu_cores: int) -> TileConfig:
    """Heuristic for medium problems (64M - 1B ops).

    This is the typical inference workload. Balance between occupancy
    and compute efficiency.
    """
    # Calculate grid dimensions for default 64x64 config
    grid_x = (N + 63) // 64
    grid_y = (M + 63) // 64
    total_tgs = grid_x * grid_y

    # If we have good parallelism, use 64x64
    # Otherwise, try smaller tiles for better occupancy
    min_tgs_for_good_occupancy = gpu_cores * 2

    if total_tgs >= min_tgs_for_good_occupancy:
        # Good occupancy with 64x64 tiles
        return DEFAULT_CONFIG
    elif M >= 128 and N >= 128:
        # Can fit 64x64 but might want 32x64 for more parallelism
        return TileConfig(
            tile_m=32,
            tile_n=64,
            tile_k=32,
            simdgroups_per_tg=4,
            threads_per_tg=128,
            sg_m_tiles=2,
            sg_n_tiles=4,
            num_buffers=2,
        )
    else:
        return TileConfig(
            tile_m=32,
            tile_n=32,
            tile_k=32,
            simdgroups_per_tg=2,
            threads_per_tg=64,
            sg_m_tiles=2,
            sg_n_tiles=2,
            num_buffers=2,
        )


def _heuristic_large(M: int, N: int, K: int, gpu_cores: int) -> TileConfig:
    """Heuristic for large problems (1B - 16B ops).

    For large problems, maximize compute per threadgroup to amortize
    memory latency. Use larger tiles.
    """
    # Check if we have enough work for larger tiles
    if M >= 256 and N >= 128:
        # Use 128x64 for large M
        return TileConfig(
            tile_m=128,
            tile_n=64,
            tile_k=32,
            simdgroups_per_tg=8,
            threads_per_tg=256,
            sg_m_tiles=4,
            sg_n_tiles=4,
            num_buffers=2,
        )
    elif M >= 128 and N >= 128:
        # Use 64x128 for large N
        return TileConfig(
            tile_m=64,
            tile_n=128,
            tile_k=32,
            simdgroups_per_tg=8,
            threads_per_tg=256,
            sg_m_tiles=2,
            sg_n_tiles=8,
            num_buffers=2,
        )
    else:
        # Fall back to default
        return DEFAULT_CONFIG


def _heuristic_xlarge(M: int, N: int, K: int, gpu_cores: int) -> TileConfig:
    """Heuristic for extra-large problems (> 16B ops).

    Very large batches or long sequences. Use the largest tiles
    for maximum compute efficiency.
    """
    # Large tiles, more simdgroups
    if M >= 256 and N >= 256:
        return TileConfig(
            tile_m=128,
            tile_n=128,
            tile_k=32,
            simdgroups_per_tg=8,
            threads_per_tg=256,
            sg_m_tiles=4,
            sg_n_tiles=8,
            num_buffers=2,
        )
    elif M >= 128:
        return TileConfig(
            tile_m=128,
            tile_n=64,
            tile_k=32,
            simdgroups_per_tg=8,
            threads_per_tg=256,
            sg_m_tiles=4,
            sg_n_tiles=4,
            num_buffers=2,
        )
    else:
        return _heuristic_large(M, N, K, gpu_cores)


def _gpu_family_id(gpu: GPUFingerprint) -> int:
    """Map GPU fingerprint to a simplified family id.

    Family IDs:
    - 7: M1
    - 8: M2
    - 9: M3+
    """
    name = gpu.name.lower()
    if "m4" in name or "m3" in name:
        return 9
    if "m2" in name:
        return 8
    if "m1" in name:
        return 7

    match = re.search(r"(\d+)", gpu.metal_family)
    if match:
        family_num = int(match.group(1))
        if family_num >= 8:
            return 9
        if family_num == 7:
            return 7

    return 7


def _find_tile_config(tile_m: int, tile_n: int, tile_k: int) -> TileConfig | None:
    for config in TILE_CONFIGS:
        if (
            config.tile_m == tile_m
            and config.tile_n == tile_n
            and config.tile_k == tile_k
        ):
            return config
    return None


def _family_profile_config(family_id: int) -> TileConfig | None:
    profile = GPU_FAMILY_TILE_CONFIG.get(family_id)
    if not profile:
        return None
    return _find_tile_config(
        profile["TILE_M"],
        profile["TILE_N"],
        profile["TILE_K"],
    )


def _select_m3plus_config(M: int, N: int, K: int) -> TileConfig | None:
    """Prefer larger tiles on M3+ to favor arithmetic intensity."""
    profile = _family_profile_config(9)
    if profile and M >= profile.tile_m and N >= profile.tile_n and K >= profile.tile_k:
        return profile

    if M >= 128 and N >= 64 and K >= 32:
        config = _find_tile_config(128, 64, 32)
        if config is not None:
            return config
    if M >= 64 and N >= 128 and K >= 32:
        config = _find_tile_config(64, 128, 32)
        if config is not None:
            return config

    return None


def get_decode_config(
    N: int, K: int, batch_size: int = 1, gpu_cores: int | None = None
) -> TileConfig:
    """Get optimal config for decode phase (M = batch_size, typically 1).

    Decode is characterized by very small M (often 1), which means:
    - Few threadgroups in M dimension
    - Need to parallelize well in N dimension
    - Smaller tiles often win due to better thread utilization

    Args:
        N: Output dimension (hidden_size).
        K: Input dimension (hidden_size).
        batch_size: Number of sequences being decoded in parallel.
        gpu_cores: GPU core count. Auto-detected if None.

    Returns:
        TileConfig optimized for decode workload.
    """
    if gpu_cores is None:
        gpu_cores = detect_gpu().cores

    M = batch_size

    # For batch=1 decode, use small tiles
    if M == 1:
        # Use 16x16 tiles for single-token decode
        return TileConfig(
            tile_m=16,
            tile_n=16,
            tile_k=32,
            simdgroups_per_tg=2,
            threads_per_tg=64,
            sg_m_tiles=1,
            sg_n_tiles=2,
            num_buffers=2,
        )
    elif M <= 8:
        # Small batch decode
        return TileConfig(
            tile_m=32,
            tile_n=32,
            tile_k=32,
            simdgroups_per_tg=2,
            threads_per_tg=64,
            sg_m_tiles=2,
            sg_n_tiles=2,
            num_buffers=2,
        )
    else:
        # Larger batch decode - use medium heuristic
        return _heuristic_small(M, N, K, gpu_cores)


def get_prefill_config(
    seq_len: int,
    N: int,
    K: int,
    batch_size: int = 1,
    gpu_cores: int | None = None,
) -> TileConfig:
    """Get optimal config for prefill phase (M = batch_size * seq_len).

    Prefill processes the full prompt, so M is typically large:
    - Many threadgroups available
    - Can use larger tiles for better compute efficiency
    - Memory bandwidth often the bottleneck

    Args:
        seq_len: Sequence length being prefilled.
        N: Output dimension (hidden_size).
        K: Input dimension (hidden_size).
        batch_size: Number of sequences being prefilled.
        gpu_cores: GPU core count. Auto-detected if None.

    Returns:
        TileConfig optimized for prefill workload.
    """
    gpu = detect_gpu()
    if gpu_cores is None:
        gpu_cores = gpu.cores
    family_id = _gpu_family_id(gpu)

    M = batch_size * seq_len

    # Prefill typically has large M, use larger tiles
    if family_id >= 9:
        m3_config = _select_m3plus_config(M, N, K)
        if m3_config is not None:
            return m3_config

    if M >= 512:
        return _heuristic_large(M, N, K, gpu_cores)
    elif M >= 128:
        return _heuristic_medium(M, N, K, gpu_cores)
    else:
        return _heuristic_small(M, N, K, gpu_cores)


def estimate_throughput(
    config: TileConfig, M: int, N: int, K: int, gpu_cores: int = 38
) -> dict[str, float]:
    """Estimate throughput for a config without running it.

    Uses a simple roofline model to estimate performance. This is useful
    for quick comparisons without actual benchmarking.

    Args:
        config: Tile configuration to evaluate.
        M, N, K: Problem dimensions.
        gpu_cores: GPU core count for occupancy estimation.

    Returns:
        Dict with estimated metrics:
        - estimated_gflops: Theoretical GFLOPS based on occupancy
        - occupancy: Estimated GPU occupancy (0.0 - 1.0)
        - threadgroups: Total number of threadgroups
        - parallelism_ratio: Threadgroups / GPU cores
    """
    # Calculate grid dimensions
    grid_x = (N + config.tile_n - 1) // config.tile_n
    grid_y = (M + config.tile_m - 1) // config.tile_m
    total_tgs = grid_x * grid_y

    # Estimate occupancy
    # Apple Silicon can run multiple threadgroups per core
    # Rough estimate: 4-8 threadgroups per core at peak
    tgs_per_core = 4.0
    max_parallel_tgs = int(gpu_cores * tgs_per_core)

    parallelism_ratio = total_tgs / gpu_cores
    occupancy = min(1.0, total_tgs / max_parallel_tgs)

    # Estimate GFLOPS
    # Peak GFLOPS for Apple Silicon varies by generation
    # M2 Max: ~13.6 TFLOPS FP16, M1 Max: ~10.4 TFLOPS
    # Use conservative estimate
    peak_gflops = gpu_cores * 256  # Rough: 256 GFLOPS per core FP16

    # Apply occupancy penalty
    estimated_gflops = peak_gflops * occupancy * 0.7  # 70% efficiency factor

    return {
        "estimated_gflops": estimated_gflops,
        "occupancy": occupancy,
        "threadgroups": float(total_tgs),
        "parallelism_ratio": parallelism_ratio,
    }


def select_best_heuristic(
    M: int,
    N: int,
    K: int,
    configs: list[TileConfig] | None = None,
    gpu: GPUFingerprint | None = None,
) -> TileConfig:
    """Select the best config from a list using heuristic scoring.

    This provides a fast way to select from multiple configurations
    without benchmarking. Useful for narrowing down the search space.

    Args:
        M, N, K: Problem dimensions.
        configs: Configurations to choose from. Uses TILE_CONFIGS if None.

    Returns:
        Best scoring configuration.
    """
    if configs is None:
        configs = TILE_CONFIGS

    gpu = gpu or detect_gpu()
    family_id = _gpu_family_id(gpu)
    best_config = DEFAULT_CONFIG
    best_score = -float("inf")

    for config in configs:
        # Skip configs that don't fit
        if M < config.tile_m or N < config.tile_n or K < config.tile_k:
            continue

        # Score the config
        metrics = estimate_throughput(config, M, N, K, gpu.cores)

        # Score: balance occupancy and compute efficiency
        # Prefer high occupancy but also reward larger tiles for amortization
        tile_size_factor = (config.tile_m * config.tile_n) / (64 * 64)  # Normalized
        occupancy_factor = metrics["occupancy"]

        # Compute score: weighted combination
        if family_id >= 9:
            score = occupancy_factor * 0.4 + min(1.0, tile_size_factor) * 0.6
        else:
            score = occupancy_factor * 0.6 + min(1.0, tile_size_factor) * 0.4

        # Bonus for good parallelism ratio
        if metrics["parallelism_ratio"] >= 2.0:
            score += 0.1

        if score > best_score:
            best_score = score
            best_config = config

    return best_config
