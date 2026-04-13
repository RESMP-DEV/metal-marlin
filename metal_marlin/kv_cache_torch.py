"""PyTorch-based KV Cache - Consolidated in metal_marlin.kv_cache.

This module is maintained for backward compatibility.
"""

from .kv_cache import CacheConfigTorch, KVCacheTorch
from .kv_cache import KVCacheTorch as KVCache

__all__ = ["KVCache", "KVCacheTorch", "CacheConfigTorch"]
