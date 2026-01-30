"""Hybrid scheduler for parallel Metal GPU + ANE execution.

Splits workload between:
- Metal GPU: Custom GEMM kernels (attention projections, FFN)
- ANE: Convolutions, LayerNorm, some MatMuls

Uses async dispatch to overlap GPU and ANE execution.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

import torch


class ComputeUnit(Enum):
    """Compute unit for operation dispatch."""

    METAL_GPU = "metal_gpu"
    ANE = "ane"
    CPU = "cpu"
    AUTO = "auto"


@dataclass
class OpProfile:
    """Profile for an operation type."""

    name: str
    preferred_unit: ComputeUnit
    metal_latency_ms: float = 0.0
    ane_latency_ms: float = 0.0
    cpu_latency_ms: float = 0.0


# Default profiles based on Apple Silicon characteristics
DEFAULT_OP_PROFILES = {
    "linear": OpProfile("linear", ComputeUnit.METAL_GPU),  # Custom GEMM
    "conv1d": OpProfile("conv1d", ComputeUnit.ANE),  # ANE excels at conv
    "layernorm": OpProfile("layernorm", ComputeUnit.ANE),
    "attention": OpProfile("attention", ComputeUnit.METAL_GPU),
    "softmax": OpProfile("softmax", ComputeUnit.ANE),
    "gelu": OpProfile("gelu", ComputeUnit.ANE),
}


class HybridScheduler:
    """Schedule operations across Metal GPU and ANE.

    Maintains separate execution queues for GPU and ANE,
    enabling overlap when possible.
    """

    def __init__(
        self,
        enable_ane: bool = True,
        enable_metal: bool = True,
        max_workers: int = 2,
    ):
        """Initialize scheduler.

        Args:
            enable_ane: Enable ANE execution path
            enable_metal: Enable Metal GPU execution path
            max_workers: Max concurrent operations
        """
        self.enable_ane = enable_ane
        self.enable_metal = enable_metal
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._profiles = dict(DEFAULT_OP_PROFILES)
        self._pending: list[Future] = []

    def dispatch(
        self,
        op: Callable[..., torch.Tensor],
        *args,
        op_type: str = "linear",
        **kwargs,
    ) -> torch.Tensor:
        """Dispatch operation to appropriate compute unit.

        Args:
            op: Operation to execute
            *args: Positional arguments for op
            op_type: Operation type for routing
            **kwargs: Keyword arguments for op

        Returns:
            Operation result
        """
        profile = self._profiles.get(op_type, OpProfile(op_type, ComputeUnit.AUTO))
        unit = self._select_unit(profile)

        if unit == ComputeUnit.ANE and self.enable_ane:
            # ANE path - ensure data is on CPU
            cpu_args = tuple(a.cpu() if isinstance(a, torch.Tensor) else a for a in args)
            result = op(*cpu_args, **kwargs)
            # Move result to MPS if needed
            if (
                isinstance(result, torch.Tensor)
                and args
                and hasattr(args[0], "is_mps")
                and args[0].is_mps
            ):
                result = result.to("mps")
            return result

        elif unit == ComputeUnit.METAL_GPU and self.enable_metal:
            # Metal GPU path - keep on MPS
            return op(*args, **kwargs)

        else:
            # CPU fallback
            cpu_args = tuple(a.cpu() if isinstance(a, torch.Tensor) else a for a in args)
            return op(*cpu_args, **kwargs)

    def dispatch_async(
        self,
        op: Callable[..., torch.Tensor],
        *args,
        op_type: str = "linear",
        **kwargs,
    ) -> Future:
        """Dispatch operation asynchronously.

        Args:
            op: Operation to execute
            *args: Positional arguments
            op_type: Operation type
            **kwargs: Keyword arguments

        Returns:
            Future for the result
        """
        future = self._executor.submit(self.dispatch, op, *args, op_type=op_type, **kwargs)
        self._pending.append(future)
        return future

    def synchronize(self) -> None:
        """Wait for all pending operations to complete."""
        for future in self._pending:
            future.result()
        self._pending.clear()

        # Also sync MPS
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

    def _select_unit(self, profile: OpProfile) -> ComputeUnit:
        """Select compute unit based on profile and availability."""
        if profile.preferred_unit == ComputeUnit.ANE and self.enable_ane:
            return ComputeUnit.ANE
        elif profile.preferred_unit == ComputeUnit.METAL_GPU and self.enable_metal:
            return ComputeUnit.METAL_GPU
        elif self.enable_metal:
            return ComputeUnit.METAL_GPU
        elif self.enable_ane:
            return ComputeUnit.ANE
        else:
            return ComputeUnit.CPU

    def update_profile(self, op_type: str, profile: OpProfile) -> None:
        """Update operation profile based on measured latencies."""
        self._profiles[op_type] = profile

    def shutdown(self) -> None:
        """Shutdown the scheduler."""
        self.synchronize()
        self._executor.shutdown(wait=True)


# Global scheduler instance
_global_scheduler: HybridScheduler | None = None


def get_scheduler() -> HybridScheduler:
    """Get global hybrid scheduler."""
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = HybridScheduler()
    return _global_scheduler


def set_scheduler(scheduler: HybridScheduler) -> None:
    """Set global hybrid scheduler."""
    global _global_scheduler
    _global_scheduler = scheduler
