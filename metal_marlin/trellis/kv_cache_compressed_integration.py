"""Integration helpers for optimized CompressedKVCacheMLA with Trellis attention.

This module provides utility functions to integrate the optimized
CompressedKVCacheMLA with TrellisMLAttention, enabling on-the-fly
decompression, block-sparse layout, and prefetch optimizations.

Usage:
    from metal_marlin.trellis.kv_cache_compressed_integration import (
        get_optimized_kv_from_cache,
        decompress_kv_optimized,
        prefetch_with_optimization,
        update_with_prefetch_and_cache,
        create_compressed_kv_cache,
    )

    # Create optimized cache
    cache = create_compressed_kv_cache(
        config=trellis_config,
        max_batch_size=1,
        max_seq_len=8192,
        device=torch.device("mps"),
        quantize_mode="fp8",
    )

    # Update cache with new tokens
    compressed_kv = torch.randn(1, 1, 576, device="mps", dtype=torch.float16)
    update_with_prefetch_and_cache(
        cache,
        layer_idx=0,
        compressed_kv=compressed_kv,
        trigger_prefetch=True,
    )

    # Get decompressed KV for attention
    k, v = decompress_kv_optimized(
        cache,
        layer_idx=0,
        kv_b_proj_weight=kv_b_proj.weight,
        kv_a_layernorm=kv_a_layernorm,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .config import TrellisModelConfig
    from .kv_cache_compressed import CompressedKVCacheMLA


def get_optimized_kv_from_cache(
    kv_cache: CompressedKVCacheMLA,
    layer_idx: int,
    batch_indices: list[int] | None = None,
) -> torch.Tensor | None:
    """Get compressed KV from optimized cache.

    This function integrates with CompressedKVCacheMLA's block-sparse
    layout and threadgroup cache to retrieve compressed KV efficiently.

    Args:
        kv_cache: CompressedKVCacheMLA instance
        layer_idx: Layer index
        batch_indices: Optional list of batch indices to retrieve

    Returns:
        Compressed KV tensor [batch, seq_len, kv_lora_rank + qk_rope_head_dim]
    """
    return kv_cache.get_compressed_kv(layer_idx, batch_indices)


def decompress_kv_optimized(
    kv_cache: CompressedKVCacheMLA,
    layer_idx: int,
    kv_b_proj_weight: torch.Tensor,
    kv_a_layernorm: torch.nn.Module | None = None,
    batch_indices: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Decompress KV using optimized cache method.

    This leverages CompressedKVCacheMLA's on-the-fly decompression
    with block-sparse layout and threadgroup cache.

    Args:
        kv_cache: CompressedKVCacheMLA instance
        layer_idx: Layer index
        kv_b_proj_weight: Decompression weight [num_kv_heads * (qk_nope + v), kv_lora_rank]
        kv_a_layernorm: Optional layernorm to apply to latent
        batch_indices: Optional list of batch indices

    Returns:
        k: Decompressed keys [batch, seq_len, num_kv_heads, qk_nope_head_dim]
        v: Decompressed values [batch, seq_len, num_kv_heads, v_head_dim]
    """
    return kv_cache.decompress_kv(
        layer_idx,
        kv_b_proj_weight,
        kv_a_layernorm,
        batch_indices,
    )


def prefetch_with_optimization(
    kv_cache: CompressedKVCacheMLA,
    layer_idx: int,
    block_indices: list[int] | None = None,
) -> None:
    """Prefetch next layer's blocks using optimized cache.

    This triggers the prefetch optimization in CompressedKVCacheMLA,
    which loads next blocks into cache hierarchy during attention.

    Args:
        kv_cache: CompressedKVCacheMLA instance
        layer_idx: Layer index to prefetch
        block_indices: Optional list of specific block indices to prefetch
    """
    kv_cache.prefetch_layer_async(layer_idx, block_indices)


def update_with_prefetch_and_cache(
    kv_cache: CompressedKVCacheMLA,
    layer_idx: int,
    compressed_kv: torch.Tensor,
    trigger_prefetch: bool = True,
    update_threadgroup_cache: bool = True,
) -> torch.Tensor:
    """Update cache and optionally trigger prefetch and cache updates.

    This is a wrapper around cache.update() that also triggers
    prefetch for the next layer and updates threadgroup cache when enabled.

    Args:
        kv_cache: CompressedKVCacheMLA instance
        layer_idx: Layer index
        compressed_kv: Compressed KV to add [batch, seq_len, cache_dim]
        trigger_prefetch: Whether to trigger prefetch for next layer
        update_threadgroup_cache: Whether to update threadgroup cache

    Returns:
        Updated compressed KV sequence
    """
    result = kv_cache.update(layer_idx, compressed_kv)

    if trigger_prefetch and layer_idx + 1 < kv_cache.num_layers:
        # Prefetch next layer's blocks
        prefetch_with_optimization(kv_cache, layer_idx + 1)

    if update_threadgroup_cache:
        # Update threadgroup cache with current layer's blocks
        # This is a hint - actual caching happens during decompression
        pass

    return result


def is_optimized_cache(kv_cache) -> bool:
    """Check if kv_cache is an optimized CompressedKVCacheMLA.

    Args:
        kv_cache: KV cache instance

    Returns:
        True if kv_cache has optimized decompression methods
    """
    return (
        hasattr(kv_cache, "decompress_kv")
        and callable(kv_cache.decompress_kv)
        and hasattr(kv_cache, "get_block_sparse_stats")
    )


def get_cache_optimization_stats(kv_cache) -> dict:
    """Get optimization statistics from cache.

    Returns statistics about block-sparse mode, prefetch,
    threadgroup cache, and memory savings.

    Args:
        kv_cache: CompressedKVCacheMLA instance

    Returns:
        Dictionary with optimization statistics
    """
    if not is_optimized_cache(kv_cache):
        return {}

    stats = {}

    if hasattr(kv_cache, "get_block_sparse_stats"):
        stats.update(kv_cache.get_block_sparse_stats())

    if hasattr(kv_cache, "memory_usage_mb"):
        stats["memory_usage_mb"] = kv_cache.memory_usage_mb()

    if hasattr(kv_cache, "_threadgroup_cache"):
        stats["threadgroup_cache_size"] = len(kv_cache._threadgroup_cache)

    if hasattr(kv_cache, "_prefetch_queue"):
        stats["prefetch_queue_size"] = len(kv_cache._prefetch_queue)

    if hasattr(kv_cache, "get_performance_stats"):
        stats.update(kv_cache.get_performance_stats())

    return stats


def decompress_kv_batch(
    kv_cache: CompressedKVCacheMLA,
    layer_idx: int,
    kv_b_proj_weight: torch.Tensor,
    kv_a_layernorm: torch.nn.Module | None = None,
    batch_indices: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Decompress KV with batch optimization.

    This is optimized for batch inference, using vectorized
    operations where possible.

    Args:
        kv_cache: CompressedKVCacheMLA instance
        layer_idx: Layer index
        kv_b_proj_weight: Decompression weight
        kv_a_layernorm: Optional layernorm
        batch_indices: Optional list of batch indices to decompress

    Returns:
        k: Decompressed keys
        v: Decompressed values
    """
    return decompress_kv_optimized(
        kv_cache,
        layer_idx,
        kv_b_proj_weight,
        kv_a_layernorm,
        batch_indices,
    )


def create_compressed_kv_cache(
    config: TrellisModelConfig,
    max_batch_size: int,
    max_seq_len: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float16,
    block_size: int = 64,
    quantize_mode: str = "none",
    prefetch_enabled: bool = True,
    threadgroup_cache_size: int = 4,
) -> CompressedKVCacheMLA:
    """Create an optimized CompressedKVCacheMLA instance.

    This factory function creates a CompressedKVCacheMLA with optimized
    settings for the given model configuration.

    Args:
        config: TrellisModelConfig
        max_batch_size: Maximum batch size
        max_seq_len: Maximum sequence length
        device: Device to use
        dtype: Data type for cache storage
        block_size: Block size for paging (default: 64)
        quantize_mode: Quantization mode ("none", "fp8", "fp4")
        prefetch_enabled: Enable async prefetching
        threadgroup_cache_size: Size of threadgroup cache

    Returns:
        Configured CompressedKVCacheMLA instance

    Example:
        from metal_marlin.trellis.config import TrellisModelConfig
        config = TrellisModelConfig.from_pretrained("THUDM/glm-4-9b-chat")
        cache = create_compressed_kv_cache(
            config,
            max_batch_size=1,
            max_seq_len=8192,
            device=torch.device("mps"),
            quantize_mode="fp8",
        )
    """
    from .kv_cache_compressed import CompressedKVCacheMLA

    return CompressedKVCacheMLA(
        config=config,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        device=device,
        dtype=dtype,
        block_size=block_size,
        quantize_mode=quantize_mode,
        prefetch_enabled=prefetch_enabled,
        threadgroup_cache_size=threadgroup_cache_size,
    )


def reset_cache_for_batch(
    kv_cache: CompressedKVCacheMLA,
    batch_idx: int,
) -> None:
    """Reset cache for a specific batch entry.

    Useful for freeing memory after a sequence completes.

    Args:
        kv_cache: CompressedKVCacheMLA instance
        batch_idx: Batch index to reset
    """
    kv_cache.reset_batch(batch_idx)


def reset_cache_all(kv_cache: CompressedKVCacheMLA) -> None:
    """Reset all cache state.

    Clears all cached data and resets statistics.

    Args:
        kv_cache: CompressedKVCacheMLA instance
    """
    kv_cache.reset_all()


def get_cached_block(
    kv_cache: CompressedKVCacheMLA,
    layer_idx: int,
    block_idx: int,
) -> torch.Tensor | None:
    """Get a block from threadgroup cache if available.

    This is useful for avoiding memory transfers when the same
    block is needed multiple times.

    Args:
        kv_cache: CompressedKVCacheMLA instance
        layer_idx: Layer index
        block_idx: Block index within layer

    Returns:
        Cached block data if available, None otherwise
    """
    return kv_cache.get_cached_block(layer_idx, block_idx)


def print_cache_stats(kv_cache: CompressedKVCacheMLA) -> None:
    """Print cache statistics for monitoring and debugging.

    Args:
        kv_cache: CompressedKVCacheMLA instance
    """
    block_stats = kv_cache.get_block_sparse_stats()
    perf_stats = kv_cache.get_performance_stats()

    print("=" * 60)
    print("CompressedKVCacheMLA Statistics")
    print("=" * 60)

    print("\nMemory Usage:")
    print(f"  Total blocks: {block_stats['total_blocks']}")
    print(f"  Used blocks: {block_stats['used_blocks']}")
    print(f"  Block size: {block_stats['block_size']}")
    print(f"  Memory: {kv_cache.memory_usage_mb():.2f} MB")

    print("\nCompression:")
    print(f"  Cache dimension: {block_stats['cache_dim']}")
    print(f"  Standard KV dimension: {block_stats['standard_kv_dim']}")
    print(f"  Compression ratio: {block_stats['compression_ratio']:.2f}x")
    print(f"  Quantization mode: {block_stats['quantize_mode']}")

    print("\nFragmentation:")
    print(f"  Fragmentation: {block_stats['fragmentation_pct']:.2f}%")

    print("\nPerformance:")
    print(f"  Total allocations: {perf_stats['total_allocations']}")
    print(f"  Total deallocations: {perf_stats['total_deallocations']}")
    print(f"  Cache hits: {perf_stats['cache_hits']}")
    print(f"  Cache misses: {perf_stats['cache_misses']}")
    if perf_stats['cache_hits'] + perf_stats['cache_misses'] > 0:
        hit_rate = (
            perf_stats['cache_hits']
            / (perf_stats['cache_hits'] + perf_stats['cache_misses'])
            * 100
        )
        print(f"  Cache hit rate: {hit_rate:.2f}%")
    print(f"  Prefetch count: {perf_stats['prefetch_count']}")
    print(f"  Decompression count: {perf_stats['decompression_count']}")

    print("\nThreadgroup Cache:")
    print(f"  Cache size: {perf_stats['threadgroup_cache_size']}")
    print(f"  Prefetch queue size: {perf_stats['prefetch_queue_size']}")

    print("=" * 60)


def estimate_memory_savings(
    config: TrellisModelConfig,
    max_seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    quantize_mode: str = "none",
) -> dict:
    """Estimate memory savings from compressed KV cache.

    Compares standard KV cache memory usage vs compressed KV cache.

    Args:
        config: TrellisModelConfig
        max_seq_len: Maximum sequence length
        num_layers: Number of layers
        num_kv_heads: Number of key-value heads
        head_dim: Head dimension
        dtype: Data type (default: float16)
        quantize_mode: Quantization mode ("none", "fp8", "fp4")

    Returns:
        Dictionary with memory statistics

    Example:
        from metal_marlin.trellis.config import TrellisModelConfig
        config = TrellisModelConfig.from_pretrained("THUDM/glm-4-9b-chat")
        stats = estimate_memory_savings(
            config,
            max_seq_len=8192,
            num_layers=32,
            num_kv_heads=20,
            head_dim=112,
        )
        print(f"Memory savings: {stats['savings_ratio']:.2f}x")
    """
    bytes_per_element = torch.tensor(0, dtype=dtype).element_size()

    # Standard KV cache memory
    # K + V: 2 * seq_len * num_kv_heads * head_dim * bytes * num_layers
    kv_lora_rank = config.kv_lora_rank or 512
    qk_rope_head_dim = config.qk_rope_head_dim
    standard_bytes = (
        2
        * max_seq_len
        * num_kv_heads
        * head_dim
        * bytes_per_element
        * num_layers
    )

    # Compressed KV cache memory
    # Compressed KV: seq_len * (kv_lora_rank + qk_rope_head_dim) * bytes * num_layers
    quant_factor = 1.0
    if quantize_mode == "fp8":
        quant_factor = 0.5
    elif quantize_mode == "fp4":
        quant_factor = 0.25

    compressed_bytes = (
        max_seq_len
        * (kv_lora_rank + qk_rope_head_dim)
        * bytes_per_element
        * quant_factor
        * num_layers
    )

    return {
        "standard_kv_mb": standard_bytes / (1024 * 1024),
        "compressed_kv_mb": compressed_bytes / (1024 * 1024),
        "savings_mb": (standard_bytes - compressed_bytes) / (1024 * 1024),
        "savings_ratio": standard_bytes / compressed_bytes,
        "quantize_mode": quantize_mode,
    }
