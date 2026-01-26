"""Cache utilities for efficient transformer inference.

This module provides:
1. Quantized KV cache for memory-efficient long-context inference
2. Prefix/prompt caching for reusing KV across requests with common prefixes

Key exports:

Quantized KV Cache:
- QuantizedKVCache: FP8/INT8 storage with FP16 compute (numpy-based)
- compress_kv: Quantize K/V tensors for storage
- decompress_kv: Dequantize K/V tensors for attention
- ScalingStrategy: Per-head, per-token, or asymmetric scaling
- CacheStats: Statistics about cache memory usage

When MLX is available:
- QuantizedKVCacheMLX: MLX-accelerated quantized KV cache
- compress_kv_mlx: MLX KV compression
- decompress_kv_mlx: MLX KV decompression

Prefix Caching:
- PrefixCache: Hierarchical prefix cache with GPU/RAM/disk tiers
- RadixPrefixCache: O(log n) prefix matching with radix tree
- PrefixCacheConfig: Configuration for prefix cache
- hash_prefix: Block-aligned token hashing for prefix matching
- hash_tokens: Hash a sequence of tokens

Usage (quantized KV cache):
    from metal_marlin.cache import QuantizedKVCache, ScalingStrategy

    cache = QuantizedKVCache(
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        max_seq_len=8192,
        scaling=ScalingStrategy.PER_HEAD,
    )

Usage (prefix caching):
    from metal_marlin.cache import PrefixCache, PrefixCacheConfig

    config = PrefixCacheConfig(
        num_layers=32,
        num_heads=32,
        head_dim=128,
        max_gpu_blocks=512,
    )
    cache = PrefixCache(config)

    # Match prefix
    match = cache.match_prefix(tokens)
    if match.num_matched_blocks > 0:
        # Reuse cached KV
        kv_blocks = cache.get_blocks(match.block_hashes)
"""

from .prompt_cache import (
    CachedBlock,
    CacheMetrics,
    PrefixCache,
    PrefixCacheConfig,
    PrefixMatch,
    RadixPrefixCache,
    StorageTier,
    hash_prefix,
    hash_tokens,
)
from .quantized_kv import CacheStats, QuantizedKVCache, ScalingStrategy, compress_kv, decompress_kv

__all__ = [
    # Quantized KV cache
    "CacheStats",
    "QuantizedKVCache",
    "ScalingStrategy",
    "compress_kv",
    "decompress_kv",
    # Prefix caching
    "CachedBlock",
    "CacheMetrics",
    "PrefixCache",
    "PrefixCacheConfig",
    "PrefixMatch",
    "RadixPrefixCache",
    "StorageTier",
    "hash_prefix",
    "hash_tokens",
]

# MLX exports when available
try:
    from .quantized_kv import (  # noqa: F401 (re-exported to __all__)
        QuantizedKVCacheMLX,
        compress_kv_mlx,
        decompress_kv_mlx,
    )

    __all__.extend(
        [
            "QuantizedKVCacheMLX",
            "compress_kv_mlx",
            "decompress_kv_mlx",
        ]
    )
except ImportError:
    # MLX not available
    pass
