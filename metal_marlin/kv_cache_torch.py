"""PyTorch-based KV Cache - Consolidated in metal_marlin.kv_cache.

This module is maintained for backward compatibility.
"""

from .kv_cache import KVCacheTorch as KVCache, KVCacheTorch, CacheConfigTorch

__all__ = ["KVCache", "KVCacheTorch", "CacheConfigTorch"]
