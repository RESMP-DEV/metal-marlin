"""NaN detection and automatic recovery for MoE inference.

Provides runtime NaN detection with automatic fallback to slow paths when
fast kernels produce numerical instabilities. Tracks NaN statistics for
debugging and identifying problematic input patterns.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

import torch

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=torch.Tensor)

__all__ = [
    "NaNGuard",
    "NaNStatistics",
    "nan_check",
]


@dataclass
class NaNStatistics:
    """Tracks NaN occurrence statistics for debugging.

    Attributes:
        nan_count_by_layer: Count of NaN occurrences per layer index.
        nan_triggering_inputs: Sample of input patterns that triggered NaN.
        total_nan_recoveries: Total number of successful fallback recoveries.
        total_nan_failures: Total number of unrecoverable NaN errors.
        last_nan_timestamp: Timestamp of most recent NaN detection.
    """

    nan_count_by_layer: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    nan_triggering_inputs: list[dict[str, Any]] = field(default_factory=list)
    total_nan_recoveries: int = 0
    total_nan_failures: int = 0
    last_nan_timestamp: float | None = None
    _max_stored_inputs: int = 100

    def record_nan(
        self,
        layer_idx: int | None,
        input_tensor: torch.Tensor | None = None,
        recovered: bool = True,
    ) -> None:
        """Record a NaN occurrence.

        Args:
            layer_idx: Layer index where NaN occurred, or None if unknown.
            input_tensor: Optional input tensor that triggered NaN.
            recovered: Whether the NaN was recovered via fallback.
        """
        self.last_nan_timestamp = time.time()

        if layer_idx is not None:
            self.nan_count_by_layer[layer_idx] += 1

        if recovered:
            self.total_nan_recoveries += 1
        else:
            self.total_nan_failures += 1

        # Store sample of triggering inputs for debugging
        if input_tensor is not None and len(self.nan_triggering_inputs) < self._max_stored_inputs:
            # Store summary statistics, not full tensor
            with torch.no_grad():
                input_flat = input_tensor.float().view(-1)
                self.nan_triggering_inputs.append({
                    "timestamp": self.last_nan_timestamp,
                    "layer_idx": layer_idx,
                    "shape": list(input_tensor.shape),
                    "dtype": str(input_tensor.dtype),
                    "min": input_flat.min().item(),
                    "max": input_flat.max().item(),
                    "mean": input_flat.mean().item(),
                    "std": input_flat.std().item(),
                    "has_inf": input_flat.isinf().any().item(),
                    "has_nan": input_flat.isnan().any().item(),
                })

    def get_summary(self) -> dict[str, Any]:
        """Get summary of NaN statistics.

        Returns:
            Dictionary with NaN statistics summary.
        """
        return {
            "total_recoveries": self.total_nan_recoveries,
            "total_failures": self.total_nan_failures,
            "layers_affected": dict(self.nan_count_by_layer),
            "most_affected_layer": (
                max(self.nan_count_by_layer.items(), key=lambda x: x[1])[0]
                if self.nan_count_by_layer
                else None
            ),
            "last_nan_timestamp": self.last_nan_timestamp,
            "sample_triggering_inputs": self.nan_triggering_inputs[:5],
        }

    def clear(self) -> None:
        """Clear all statistics."""
        self.nan_count_by_layer.clear()
        self.nan_triggering_inputs.clear()
        self.total_nan_recoveries = 0
        self.total_nan_failures = 0
        self.last_nan_timestamp = None


# Global statistics instance
_global_stats = NaNStatistics()


def get_nan_statistics() -> NaNStatistics:
    """Get the global NaN statistics instance."""
    return _global_stats


def nan_check(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """Check if tensor contains NaN values.

    Args:
        tensor: Tensor to check.
        name: Name for logging purposes.

    Returns:
        True if tensor contains NaN, False otherwise.
    """
    has_nan = bool(tensor.isnan().any().item())
    if has_nan:
        logger.warning(f"NaN detected in {name}")
    return has_nan


class NaNGuard:
    """Context manager for NaN detection with automatic recovery.

    Executes a fast path and automatically falls back to a slow path
    if NaN values are detected in the output.

    Example:
        ```python
        with NaNGuard(fallback=slow_moe) as guard:
            output = fast_moe(x)
            guard.check(output)
        # If NaN detected, automatically retries with slow_moe(x)
        ```

    Attributes:
        fallback: Fallback function to use if NaN detected.
        layer_idx: Optional layer index for statistics tracking.
        enabled: Whether NaN checking is enabled.
        stats: NaN statistics instance.
    """

    def __init__(
        self,
        fallback: Callable[[torch.Tensor], torch.Tensor] | None = None,
        layer_idx: int | None = None,
        enabled: bool = True,
        stats: NaNStatistics | None = None,
    ):
        """Initialize NaNGuard.

        Args:
            fallback: Fallback function to call if NaN detected.
            layer_idx: Layer index for statistics tracking.
            enabled: Whether to enable NaN checking.
            stats: Statistics instance to use (default: global stats).
        """
        self.fallback = fallback
        self.layer_idx = layer_idx
        self.enabled = enabled
        self.stats = stats or _global_stats
        self._input: torch.Tensor | None = None
        self._output: torch.Tensor | None = None
        self._nan_detected = False
        self._fallback_output: torch.Tensor | None = None

    def set_input(self, x: torch.Tensor) -> None:
        """Store input tensor for fallback execution.

        Args:
            x: Input tensor to store.
        """
        self._input = x

    def check(self, output: torch.Tensor) -> torch.Tensor:
        """Check output for NaN and trigger fallback if needed.

        Args:
            output: Output tensor to check.

        Returns:
            Original output if no NaN, fallback output otherwise.

        Raises:
            RuntimeError: If NaN detected and no fallback provided.
        """
        if not self.enabled:
            return output

        self._output = output

        # Fast path: no NaN
        if not output.isnan().any().item():
            return output

        self._nan_detected = True
        logger.warning(
            f"NaN detected in output (layer={self.layer_idx}), attempting fallback"
        )

        # Record statistics
        self.stats.record_nan(
            layer_idx=self.layer_idx,
            input_tensor=self._input,
            recovered=self.fallback is not None,
        )

        # Attempt fallback
        if self.fallback is not None and self._input is not None:
            self._fallback_output = self.fallback(self._input)

            # Verify fallback output is clean
            if self._fallback_output.isnan().any().item():
                logger.error(
                    f"NaN persists in fallback output (layer={self.layer_idx})"
                )
                self.stats.record_nan(
                    layer_idx=self.layer_idx,
                    input_tensor=self._input,
                    recovered=False,
                )
                raise RuntimeError(
                    f"NaN detected in both fast and fallback paths at layer {self.layer_idx}"
                )

            logger.info(f"Successfully recovered from NaN via fallback (layer={self.layer_idx})")
            return self._fallback_output

        # No fallback available
        if self.fallback is None:
            raise RuntimeError(
                f"NaN detected at layer {self.layer_idx} but no fallback provided"
            )

        raise RuntimeError(
            f"NaN detected at layer {self.layer_idx} but no input stored for fallback"
        )

    @property
    def nan_detected(self) -> bool:
        """Whether NaN was detected during this guard's scope."""
        return self._nan_detected


@contextmanager
def nan_guarded(
    x: torch.Tensor,
    fallback: Callable[[torch.Tensor], torch.Tensor],
    layer_idx: int | None = None,
    enabled: bool = True,
) -> Generator[NaNGuard, None, None]:
    """Context manager for NaN-guarded execution.

    Convenience wrapper that sets up NaNGuard with input tensor.

    Example:
        ```python
        with nan_guarded(x, slow_moe, layer_idx=5) as guard:
            output = fast_moe(x)
            result = guard.check(output)
        ```

    Args:
        x: Input tensor.
        fallback: Fallback function.
        layer_idx: Layer index for statistics.
        enabled: Whether to enable NaN checking.

    Yields:
        NaNGuard instance.
    """
    guard = NaNGuard(fallback=fallback, layer_idx=layer_idx, enabled=enabled)
    guard.set_input(x)
    yield guard
