"""Mixture of Experts (MoE) support for Metal Marlin.

This module provides optimized MoE inference on Apple Silicon, including:
- Token-to-expert grouping for batched GEMM execution
- Predictive expert prefetching for autoregressive generation
- Expert weight caching with LRU eviction
- Top-K dispatcher with optional shared expert
- Batched dispatch avoiding slow MPS indexing

Key components:
- prefetch: Predictive expert loading for autoregressive generation
- moe_dispatch: Token grouping and Top-K dispatch for batched expert execution
- batched_dispatch: MPS-optimized dispatch avoiding slow advanced indexing
- expert_cache: LRU cache for dequantized expert tiles
"""

from metal_marlin.moe.batched_dispatch import (
    BatchedExpertDispatch,
    DispatchMetrics,
    MetalBatchedDispatch,
    batched_expert_forward,
    create_scatter_indices,
)
from metal_marlin.moe.expert_grouping import (
    ExpertGrouping,
    GroupDispatchInfo,
    GroupedMoEDispatcher,
)
from metal_marlin.moe.moe_dispatch import MoEDispatcher
from metal_marlin.moe.moe_dispatch_metal import (
    AsyncExpertCommandBuffer,
    AsyncMoEExecutor,
    ExpertDispatchJob,
    ExpertExecutionBarrier,
    ExpertWorkItem,
    ParallelExpertExecutor,
    create_expert_constant_buffer,
    dispatch_single_expert_async,
    execute_experts_parallel,
)
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
from metal_marlin.moe.sorted_dispatch import (
    ExpertBoundary,
    MultiPassSortedDispatch,
    SortedDispatchState,
    SortedExpertDispatch,
    sorted_expert_forward,
)
from metal_marlin.moe.sparse_dispatch import (
    SparseMoELayer,
    SparseRouterOnly,
    enable_sparse_routing,
)
from metal_marlin.moe.sparse_routing import (
    CooccurrenceEnhancer,
    SparseExpertPredictor,
    SparseExpertRouter,
    SparseRoutingConfig,
    SparseRoutingStats,
    create_sparse_router_from_profiler,
)

__all__ = [
    "AsyncExpertLoader",
    "BatchedExpertDispatch",
    "CooccurrenceEnhancer",
    "DispatchMetrics",
    "ExpertBoundary",
    "ExpertGrouping",
    "ExpertLRUCache",
    "ExpertPrefetcher",
    "GroupDispatchInfo",
    "GroupedMoEDispatcher",
    "MetalBatchedDispatch",
    "MoEDispatcher",
    "MultiPassSortedDispatch",
    "PrefetchConfig",
    "PrefetchStats",
    "PrefetchStrategy",
    "RoutingHistory",
    "SortedDispatchState",
    "SortedExpertDispatch",
    "SparseExpertPredictor",
    "SparseExpertRouter",
    "SparseMoELayer",
    "SparseRouterOnly",
    "SparseRoutingConfig",
    "SparseRoutingStats",
    "async_load_experts",
    "batched_expert_forward",
    "create_scatter_indices",
    "create_sparse_router_from_profiler",
    "enable_sparse_routing",
    "predict_next_experts",
    "sorted_expert_forward",
]
