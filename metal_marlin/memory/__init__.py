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

from metal_marlin.memory.mmfp4_memory import (
    MMFP4MemoryManager,
    MLACompressionRatio,
    MemoryStats,
    LayerMetadata,
    ExpertMetadata,
    MemoryCompactor,
    CompactionConfig,
    CompactionStats,
    MMAPWeightConfig,
    MMAPWeightStats,
    MMAPWeightManager,
)
from metal_marlin.memory.buffer_pool import (
    BufferPool,
    BufferPoolStats,
    PooledBuffer,
)

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
]
