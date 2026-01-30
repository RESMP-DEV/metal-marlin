"""Mixture of Experts (MoE) support for Metal Marlin.

This module provides optimized MoE inference on Apple Silicon, including:
- Token-to-expert grouping for batched GEMM execution
- Predictive expert prefetching for autoregressive generation
- Expert weight caching with LRU eviction
- Top-K dispatcher with optional shared expert

Key components:
- prefetch: Predictive expert loading for autoregressive generation
- moe_dispatch: Token grouping and Top-K dispatch for batched expert execution
- expert_cache: LRU cache for dequantized expert tiles
"""

from metal_marlin.moe.moe_dispatch import MoEDispatcher
from metal_marlin.moe.prefetch import (
    AsyncExpertLoader,
    ExpertLRUCache,
    ExpertPrefetcher,
    PrefetchConfig,
    PrefetchStats,
    PrefetchStrategy,
    RoutingHistory,
    async_load_experts,
    predict_next_experts,
)

__all__ = [
    "AsyncExpertLoader",
    "ExpertLRUCache",
    "ExpertPrefetcher",
    "MoEDispatcher",
    "PrefetchConfig",
    "PrefetchStats",
    "PrefetchStrategy",
    "RoutingHistory",
    "async_load_experts",
    "predict_next_experts",
]
