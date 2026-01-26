"""Continuous batching serving infrastructure for Metal Marlin."""

from .continuous_batch import (
    BatchScheduler,
    IterationPlan,
    IterationPlanner,
    KVCacheManager,
    KVRegion,
    PreemptionPolicy,
    RequestPriority,
    RequestState,
)
from .continuous_batch import (
    SchedulerConfig as ContinuousSchedulerConfig,
)
from .request import GenerationRequest, RequestStatus, SchedulerOutput
from .runner import BatchedModelRunner, ModelConfig
from .scheduler import FCFSScheduler, SchedulerConfig

__all__ = [
    # Core types
    "GenerationRequest",
    "RequestStatus",
    "SchedulerOutput",
    # Legacy scheduler
    "FCFSScheduler",
    "SchedulerConfig",
    # Continuous batching
    "BatchScheduler",
    "ContinuousSchedulerConfig",
    "IterationPlan",
    "IterationPlanner",
    "KVCacheManager",
    "KVRegion",
    "PreemptionPolicy",
    "RequestPriority",
    "RequestState",
    # Model runner
    "BatchedModelRunner",
    "ModelConfig",
]
