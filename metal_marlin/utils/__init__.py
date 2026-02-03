"""Utilities for metal_marlin.

This module provides helper utilities for memory management, adaptive batch
sizing, system introspection, tensor prefetching, and kernel profiling.
"""

from .memory import (
    MemoryInfo,
    compute_optimal_batch_size,
    estimate_tensor_memory,
    get_metal_memory_pressure,
    get_system_memory,
    should_reduce_allocation,
)
from .padding import (
    pad_numpy_2d,
    pad_torch_2d,
    round_up,
)
from .prefetch import (
    AdaptivePrefetcher,
    SystemMemoryInfo,
    TensorMetadata,
)
from .prefetch import (
    get_system_memory as get_system_memory_detailed,
)
from .profile_ops import (
    LayerFLOPs,
    LayerFLOPsCounter,
    TransformerLayerFLOPs,
    calculate_attention_flops,
    calculate_embedding_flops,
    calculate_ffn_flops,
    calculate_layernorm_flops,
    calculate_matmul_flops,
    profile_model_flops,
)
from .profiling import (
    ProfileRecord,
    ProfileStats,
    clear_profiles,
    disable_profiling,
    enable_profiling,
    get_all_profile_stats,
    get_profile_records,
    get_profile_stats,
    is_profiling_enabled,
    print_profile_summary,
    profile_kernel,
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
    # Padding helpers
    "pad_numpy_2d",
    "pad_torch_2d",
    "round_up",
    # FLOPs profiling utilities
    "LayerFLOPs",
    "LayerFLOPsCounter",
    "TransformerLayerFLOPs",
    "calculate_attention_flops",
    "calculate_embedding_flops",
    "calculate_ffn_flops",
    "calculate_layernorm_flops",
    "calculate_matmul_flops",
    "profile_model_flops",
    # Profiling utilities
    "ProfileRecord",
    "ProfileStats",
    "clear_profiles",
    "disable_profiling",
    "enable_profiling",
    "get_all_profile_stats",
    "get_profile_records",
    "get_profile_stats",
    "is_profiling_enabled",
    "print_profile_summary",
    "profile_kernel",
]
