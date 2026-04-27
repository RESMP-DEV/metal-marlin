"""MLA KV Cache - Consolidated in metal_marlin.kv_cache.

This module is maintained for backward compatibility.
"""
import logging

from .kv_cache import CompressedKVCache, MLAKVCache
from .kv_cache import MLAKVCache as TrellisKVCache


logger = logging.getLogger(__name__)

__all__ = ["TrellisKVCache", "MLAKVCache", "CompressedKVCache"]
