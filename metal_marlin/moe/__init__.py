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
from metal_marlin.moe.gpu_grouping import (
    GPUExpertGrouping,
    GPUGroupingResult,
    group_tokens_by_expert_auto,
    group_tokens_by_expert_fast,
    group_tokens_by_expert_gpu_optimized,
)
from metal_marlin.moe.expert_memory_pool import (
    BitWidthPool,
    ExpertMemoryPool,
    PoolConfig,
)
from metal_marlin.moe.expert_selection_cache import (
    CacheLookupResult,
    ExpertSelectionCache,
    ExpertSelectionCacheConfig,
    ExpertSelectionCacheStats,
    create_expert_selection_cache_for_glm47_flash,
    create_glm47_flash_config,
)
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
from metal_marlin.moe.fused_router import (
    FusedMoERouter,
    FusedRouterOutput,
    dispatch_fused_router_sorted,
)
from metal_marlin.moe.sparse_routing import (
    CooccurrenceEnhancer,
    SparseExpertPredictor,
    SparseExpertRouter,
    SparseRoutingConfig,
    SparseRoutingStats,
    create_sparse_router_from_profiler,
)
from metal_marlin.moe.entropy_regularization import (
    EntropyRegularizationOutput,
    EntropyRegularizer,
    MoEGateWithEntropy,
    compute_entropy_regularization,
)
from metal_marlin.moe_dispatch import (
    FusedMoEDispatcher,
    MoEDispatcher,
    MoEDispatchInfo,
    TopKExpertGrouping,
    compute_expert_load,
    compute_load_balancing_loss,
    gather_for_experts,
    group_tokens_by_expert,
    group_tokens_by_expert_full,
    group_tokens_by_expert_full_gpu,
    group_tokens_by_expert_gpu,
    scatter_expert_outputs,
)

__all__ = [
    "AsyncExpertLoader",
    "AsyncExpertCommandBuffer",
    "EntropyRegularizationOutput",
    "EntropyRegularizer",
    "MoEGateWithEntropy",
    "compute_entropy_regularization",
    "AsyncMoEExecutor",
    "BatchedExpertDispatch",
    "CooccurrenceEnhancer",
    "DispatchMetrics",
    "CacheLookupResult",
    "ExpertBoundary",
    "ExpertDispatchJob",
    "ExpertExecutionBarrier",
    "ExpertGrouping",
    "ExpertMemoryPool",
    "PoolConfig",
    "BitWidthPool",
    "ExpertSelectionCache",
    "ExpertSelectionCacheConfig",
    "ExpertSelectionCacheStats",
    "create_expert_selection_cache_for_glm47_flash",
    "create_glm47_flash_config",
    "ExpertLRUCache",
    "ExpertPrefetcher",
    "ExpertWorkItem",
    "FusedMoEDispatcher",
    "FusedMoERouter",
    "FusedRouterOutput",
    "GroupDispatchInfo",
    "GroupedMoEDispatcher",
    "MetalBatchedDispatch",
    "MoEDispatchInfo",
    "MoEDispatcher",
    "MultiPassSortedDispatch",
    "ParallelExpertExecutor",
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
    "TopKExpertGrouping",
    "async_load_experts",
    "batched_expert_forward",
    "compute_expert_load",
    "compute_load_balancing_loss",
    "create_expert_constant_buffer",
    "create_scatter_indices",
    "create_sparse_router_from_profiler",
    "dispatch_fused_router_sorted",
    "dispatch_single_expert_async",
    "enable_sparse_routing",
    "execute_experts_parallel",
    "gather_for_experts",
    "group_tokens_by_expert",
    "group_tokens_by_expert_full",
    "group_tokens_by_expert_full_gpu",
    "group_tokens_by_expert_gpu",
    "predict_next_experts",
    "scatter_expert_outputs",
    "sorted_expert_forward",
]
