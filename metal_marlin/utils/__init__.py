"""Utilities for metal_marlin.

This module provides helper utilities for memory management, adaptive batch
sizing, system introspection, and tensor prefetching.
"""

from .memory import (
    MemoryInfo,
    compute_optimal_batch_size,
    estimate_tensor_memory,
    get_metal_memory_pressure,
    get_system_memory,
    should_reduce_allocation,
)
from .prefetch import (
    AdaptivePrefetcher,
    SystemMemoryInfo,
    TensorMetadata,
)
from .prefetch import (
    get_system_memory as get_system_memory_detailed,
)

__all__ = [
    # Memory utilities
    "MemoryInfo",
    "compute_optimal_batch_size",
    "estimate_tensor_memory",
    "get_metal_memory_pressure",
    "get_system_memory",
    "should_reduce_allocation",
    # Prefetch utilities
    "AdaptivePrefetcher",
    "SystemMemoryInfo",
    "TensorMetadata",
    "get_system_memory_detailed",
]
