"""PyTorch-based KV Cache - Consolidated in metal_marlin.kv_cache.

This module is maintained for backward compatibility.
"""
import logging

from .kv_cache import CacheConfigTorch, KVCacheTorch
from .kv_cache import KVCacheTorch as KVCache


logger = logging.getLogger(__name__)

__all__ = ["KVCache", "KVCacheTorch", "CacheConfigTorch"]
