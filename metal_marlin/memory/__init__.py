"""Memory management utilities for MMFP4 inference.

Provides efficient memory optimization techniques for running large FP4
quantized models on Apple Silicon with limited GPU memory.

Modules:
    mmfp4_memory: MMFP4MemoryManager for layer streaming, expert caching,
        KV compression, and activation checkpointing.
    buffer_pool: BufferPool for efficient transient buffer reuse with
        tiered allocation and age-based eviction.
"""

from __future__ import annotations
import logging

from metal_marlin.memory.buffer_pool import (
    BufferPool,
    BufferPoolStats,
    PooledBuffer,
)
from metal_marlin.memory.cuda_pinned_pool import CUDAPinnedPool
from metal_marlin.memory.mmfp4_memory import (
    CompactionConfig,
    CompactionStats,
    ExpertMetadata,
    LayerMetadata,
    MemoryCompactor,
    MemoryStats,
    MLACompressionRatio,
    MMAPWeightConfig,
    MMAPWeightManager,
    MMAPWeightStats,
    MMFP4MemoryManager,
)


logger = logging.getLogger(__name__)

__all__ = [
    "MMFP4MemoryManager",
    "MLACompressionRatio",
    "MemoryStats",
    "LayerMetadata",
    "ExpertMetadata",
    "MemoryCompactor",
    "CompactionConfig",
    "CompactionStats",
    "MMAPWeightConfig",
    "MMAPWeightStats",
    "MMAPWeightManager",
    "BufferPool",
    "BufferPoolStats",
    "PooledBuffer",
    "CUDAPinnedPool",
]
