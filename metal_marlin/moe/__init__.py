"""Live MoE helpers for Metal Marlin.

This namespace intentionally exports only the active MoE routing and grouping
surface. Retired experiments live in Git history rather than runtime shims.
"""
import logging

from metal_marlin.moe.gpu_grouping import (
    GPUExpertGrouping,
    GPUGroupingResult,
    group_tokens_by_expert_auto,
    group_tokens_by_expert_fast,
    group_tokens_by_expert_gpu_optimized,
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


logger = logging.getLogger(__name__)

__all__ = [
    "FusedMoEDispatcher",
    "GPUExpertGrouping",
    "GPUGroupingResult",
    "MoEDispatchInfo",
    "MoEDispatcher",
    "TopKExpertGrouping",
    "compute_expert_load",
    "compute_load_balancing_loss",
    "gather_for_experts",
    "group_tokens_by_expert",
    "group_tokens_by_expert_auto",
    "group_tokens_by_expert_fast",
    "group_tokens_by_expert_full",
    "group_tokens_by_expert_full_gpu",
    "group_tokens_by_expert_gpu",
    "group_tokens_by_expert_gpu_optimized",
    "scatter_expert_outputs",
]
