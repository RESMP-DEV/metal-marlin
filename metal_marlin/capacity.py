"""Runtime capacity factor adjustment for MoE token routing.

Capacity factor controls the maximum tokens per expert:
    capacity = ceil(n_tokens * capacity_factor / n_experts)

This module provides tools to:
1. Analyze token overflow/dropping rates
2. Auto-tune capacity factor for target drop rates
3. Dynamically adjust capacity per-batch

For inference (as opposed to training), capacity_factor=1.0 often works since
tokens are processed sequentially rather than in large parallel batches. Higher
factors provide safety margin at the cost of memory/compute.

Example:
    >>> from metal_marlin.capacity import CapacityAnalyzer, auto_tune_capacity
    >>>
    >>> # Analyze a model's routing behavior
    >>> analyzer = CapacityAnalyzer(num_experts=64, capacity_factor=1.0)
    >>> for batch_expert_ids in routing_decisions:
    ...     overflow_info = analyzer.record_batch(batch_expert_ids)
    >>> print(analyzer.get_stats())
    >>>
    >>> # Find optimal capacity factor for <1% drop rate
    >>> optimal_factor = auto_tune_capacity(
    ...     expert_load_history, num_experts=64, target_drop_rate=0.01
    ... )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    ArrayLike = NDArray[np.integer] | NDArray[np.floating]


@dataclass
class OverflowInfo:
    """Information about token overflow for a single batch.

    Attributes:
        total_tokens: Number of tokens in the batch.
        total_assignments: Total token-expert assignments (tokens * top_k).
        capacity_per_expert: Maximum slots available per expert.
        dropped_tokens: Number of token-expert assignments that exceeded capacity.
        overflow_experts: Expert IDs that had overflow.
        max_overflow: Maximum overflow count across all experts.
    """

    total_tokens: int
    total_assignments: int
    capacity_per_expert: int
    dropped_tokens: int
    overflow_experts: list[int]
    max_overflow: int

    @property
    def drop_rate(self) -> float:
        """Fraction of token-expert assignments that were dropped."""
        if self.total_assignments == 0:
            return 0.0
        return self.dropped_tokens / self.total_assignments

    @property
    def utilization(self) -> float:
        """Fraction of capacity that was used (excluding drops)."""
        if self.capacity_per_expert == 0:
            return 0.0
        # Total capacity across all experts that received assignments
        return (self.total_assignments - self.dropped_tokens) / (
            self.capacity_per_expert * len(self.overflow_experts)
            if self.overflow_experts
            else self.total_assignments
        )


@dataclass
class CapacityStats:
    """Aggregate statistics for capacity analysis.

    Attributes:
        total_batches: Number of batches analyzed.
        total_tokens: Total tokens processed.
        total_assignments: Total token-expert assignments.
        total_dropped: Total dropped assignments.
        batches_with_overflow: Number of batches that had any overflow.
        expert_overflow_counts: Per-expert overflow count history.
        max_overflow_seen: Maximum single-expert overflow observed.
        capacity_factor: Capacity factor used for analysis.
        num_experts: Number of experts in the model.
    """

    total_batches: int = 0
    total_tokens: int = 0
    total_assignments: int = 0
    total_dropped: int = 0
    batches_with_overflow: int = 0
    expert_overflow_counts: dict[int, int] = field(default_factory=dict)
    max_overflow_seen: int = 0
    capacity_factor: float = 1.0
    num_experts: int = 64

    @property
    def overall_drop_rate(self) -> float:
        """Overall fraction of assignments dropped."""
        if self.total_assignments == 0:
            return 0.0
        return self.total_dropped / self.total_assignments

    @property
    def overflow_rate(self) -> float:
        """Fraction of batches that had overflow."""
        if self.total_batches == 0:
            return 0.0
        return self.batches_with_overflow / self.total_batches

    def top_overflow_experts(self, k: int = 10) -> list[tuple[int, int]]:
        """Get experts with most overflow events.

        Args:
            k: Number of experts to return.

        Returns:
            List of (expert_id, overflow_count) tuples, sorted by count descending.
        """
        sorted_experts = sorted(self.expert_overflow_counts.items(), key=lambda x: -x[1])
        return sorted_experts[:k]


class CapacityAnalyzer:
    """Analyzes token routing to measure overflow rates.

    Use this to understand how often tokens are dropped due to expert
    capacity limits. The analyzer tracks per-batch and aggregate statistics.

    Args:
        num_experts: Number of experts in the MoE layer.
        capacity_factor: Capacity factor for computing expert slots.
        top_k: Number of experts each token routes to (for capacity calculation).

    Example:
        >>> analyzer = CapacityAnalyzer(num_experts=64, capacity_factor=1.0)
        >>> for expert_ids in batches:  # expert_ids shape: [batch, top_k]
        ...     info = analyzer.record_batch(expert_ids)
        ...     if info.dropped_tokens > 0:
        ...         print(f"Dropped {info.dropped_tokens} assignments")
        >>> stats = analyzer.get_stats()
        >>> print(f"Overall drop rate: {stats.overall_drop_rate:.2%}")
    """

    def __init__(
        self,
        num_experts: int,
        capacity_factor: float = 1.0,
        top_k: int = 2,
    ):
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.top_k = top_k
        self._stats = CapacityStats(capacity_factor=capacity_factor, num_experts=num_experts)
        self._batch_history: list[OverflowInfo] = []

    def compute_capacity(self, batch_size: int) -> int:
        """Compute expert capacity for a given batch size.

        Args:
            batch_size: Number of tokens in the batch.

        Returns:
            Maximum tokens per expert.
        """
        return compute_expert_capacity(
            batch_size, self.num_experts, self.capacity_factor, self.top_k
        )

    def record_batch(self, expert_ids: NDArray[np.integer]) -> OverflowInfo:
        """Record routing decisions for a batch and compute overflow.

        Args:
            expert_ids: [batch, top_k] array of expert assignments.

        Returns:
            OverflowInfo with details about this batch's overflow.
        """
        batch_size, top_k = expert_ids.shape
        total_assignments = batch_size * top_k
        capacity = self.compute_capacity(batch_size)

        # Count assignments per expert
        expert_counts = count_expert_assignments(expert_ids, self.num_experts)

        # Find overflow
        overflow_mask = expert_counts > capacity
        overflow_amounts = np.maximum(0, expert_counts - capacity)
        dropped = int(np.sum(overflow_amounts))
        overflow_experts = list(np.where(overflow_mask)[0])
        max_overflow = int(np.max(overflow_amounts)) if dropped > 0 else 0

        info = OverflowInfo(
            total_tokens=batch_size,
            total_assignments=total_assignments,
            capacity_per_expert=capacity,
            dropped_tokens=dropped,
            overflow_experts=overflow_experts,
            max_overflow=max_overflow,
        )

        # Update aggregate stats
        self._stats.total_batches += 1
        self._stats.total_tokens += batch_size
        self._stats.total_assignments += total_assignments
        self._stats.total_dropped += dropped
        if dropped > 0:
            self._stats.batches_with_overflow += 1
            self._stats.max_overflow_seen = max(self._stats.max_overflow_seen, max_overflow)
            for eid in overflow_experts:
                self._stats.expert_overflow_counts[eid] = (
                    self._stats.expert_overflow_counts.get(eid, 0) + 1
                )

        self._batch_history.append(info)
        return info

    def get_stats(self) -> CapacityStats:
        """Get aggregate statistics."""
        return self._stats

    def get_batch_history(self) -> list[OverflowInfo]:
        """Get overflow info for all recorded batches."""
        return self._batch_history.copy()

    def reset(self) -> None:
        """Reset all statistics."""
        self._stats = CapacityStats(
            capacity_factor=self.capacity_factor, num_experts=self.num_experts
        )
        self._batch_history.clear()


def compute_expert_capacity(
    batch_size: int,
    num_experts: int,
    capacity_factor: float,
    top_k: int = 1,
) -> int:
    """Compute maximum tokens per expert given capacity factor.

    Formula: capacity = ceil(batch_size * top_k * capacity_factor / num_experts)

    Args:
        batch_size: Number of tokens in the batch.
        num_experts: Number of experts.
        capacity_factor: Multiplier for capacity (1.0 = exact uniform distribution).
        top_k: Number of experts each token routes to.

    Returns:
        Maximum number of tokens each expert can handle.
    """
    if capacity_factor <= 0:
        raise ValueError("capacity_factor must be positive")
    raw_capacity = batch_size * top_k * capacity_factor / num_experts
    return max(1, math.ceil(raw_capacity))


def count_expert_assignments(
    expert_ids: NDArray[np.integer], num_experts: int
) -> NDArray[np.int64]:
    """Count how many token-expert assignments go to each expert.

    Args:
        expert_ids: [batch, top_k] array of expert indices.
        num_experts: Total number of experts.

    Returns:
        [num_experts] array of assignment counts.
    """
    expert_ids_flat = np.asarray(expert_ids).reshape(-1).astype(np.int32)
    # Use bincount for efficient counting
    counts = np.bincount(expert_ids_flat, minlength=num_experts)
    return counts[:num_experts]


def analyze_overflow_rate(
    expert_load_history: list[NDArray[np.integer]] | NDArray[np.integer],
    num_experts: int,
    capacity_factor: float = 1.0,
    top_k: int = 2,
) -> CapacityStats:
    """Analyze overflow rate from historical routing decisions.

    Takes a history of expert assignments and computes what the overflow
    rate would be at a given capacity factor.

    Args:
        expert_load_history: List of [batch, top_k] expert assignment arrays,
            or single [total_tokens, top_k] array.
        num_experts: Number of experts.
        capacity_factor: Capacity factor to evaluate.
        top_k: Experts per token (inferred from data if not provided).

    Returns:
        CapacityStats with overflow analysis results.
    """
    analyzer = CapacityAnalyzer(
        num_experts=num_experts, capacity_factor=capacity_factor, top_k=top_k
    )

    if isinstance(expert_load_history, np.ndarray):
        # Single array - treat as one batch
        analyzer.record_batch(expert_load_history)
    else:
        for batch_expert_ids in expert_load_history:
            analyzer.record_batch(batch_expert_ids)

    return analyzer.get_stats()


def auto_tune_capacity(
    expert_load_history: list[NDArray[np.integer]],
    num_experts: int,
    target_drop_rate: float = 0.01,
    min_factor: float = 1.0,
    max_factor: float = 4.0,
    tolerance: float = 0.001,
    max_iterations: int = 20,
) -> float:
    """Find optimal capacity factor for a target drop rate.

    Uses binary search to find the minimum capacity factor that achieves
    the target drop rate or better.

    Args:
        expert_load_history: List of [batch, top_k] expert assignment arrays.
        num_experts: Number of experts.
        target_drop_rate: Maximum acceptable drop rate (e.g., 0.01 for 1%).
        min_factor: Minimum capacity factor to consider.
        max_factor: Maximum capacity factor to consider.
        tolerance: Convergence tolerance for binary search.
        max_iterations: Maximum search iterations.

    Returns:
        Optimal capacity factor.

    Example:
        >>> # Collect routing decisions
        >>> history = [router.get_expert_ids(batch) for batch in data]
        >>> # Find factor for <1% drops
        >>> factor = auto_tune_capacity(history, num_experts=64, target_drop_rate=0.01)
        >>> print(f"Optimal capacity factor: {factor:.2f}")
    """
    if not expert_load_history:
        return min_factor

    # Infer top_k from first batch
    top_k = expert_load_history[0].shape[1] if expert_load_history[0].ndim > 1 else 1

    # Check if max_factor achieves target
    stats = analyze_overflow_rate(expert_load_history, num_experts, max_factor, top_k)
    if stats.overall_drop_rate > target_drop_rate:
        # Even max_factor doesn't achieve target, return max
        return max_factor

    # Check if min_factor already achieves target
    stats = analyze_overflow_rate(expert_load_history, num_experts, min_factor, top_k)
    if stats.overall_drop_rate <= target_drop_rate:
        return min_factor

    # Binary search
    low, high = min_factor, max_factor
    best_factor = max_factor

    for _ in range(max_iterations):
        if high - low < tolerance:
            break

        mid = (low + high) / 2
        stats = analyze_overflow_rate(expert_load_history, num_experts, mid, top_k)

        if stats.overall_drop_rate <= target_drop_rate:
            best_factor = mid
            high = mid
        else:
            low = mid

    return best_factor


@dataclass
class DynamicCapacityConfig:
    """Configuration for dynamic capacity adjustment.

    Attributes:
        base_factor: Starting capacity factor.
        min_factor: Minimum allowed factor.
        max_factor: Maximum allowed factor.
        target_drop_rate: Target drop rate to maintain.
        increase_step: Factor increase when drops exceed target.
        decrease_step: Factor decrease when drops are below threshold.
        decrease_threshold: Drop rate below which to decrease factor.
        window_size: Number of batches to consider for adjustment.
        cooldown_batches: Batches to wait after adjustment before next.
    """

    base_factor: float = 1.25
    min_factor: float = 1.0
    max_factor: float = 4.0
    target_drop_rate: float = 0.01
    increase_step: float = 0.1
    decrease_step: float = 0.05
    decrease_threshold: float = 0.001
    window_size: int = 10
    cooldown_batches: int = 5


class DynamicCapacity:
    """Dynamically adjusts capacity factor based on observed overflow.

    Monitors batch-by-batch overflow and adjusts capacity factor to maintain
    target drop rate. Useful for inference workloads with variable batch
    sizes or non-uniform routing patterns.

    For sequential/single-token inference, capacity_factor=1.0 typically
    suffices since each token can be assigned to its chosen experts without
    batching contention.

    Args:
        num_experts: Number of experts.
        top_k: Experts per token.
        config: Configuration for dynamic adjustment.

    Example:
        >>> dyn_capacity = DynamicCapacity(num_experts=64, top_k=2)
        >>> for batch in inference_batches:
        ...     expert_ids = router(batch)
        ...     capacity = dyn_capacity.get_capacity(len(batch))
        ...     outputs = moe_forward(batch, expert_ids, capacity)
        ...     dyn_capacity.update(expert_ids)
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int = 2,
        config: DynamicCapacityConfig | None = None,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.config = config or DynamicCapacityConfig()
        self._current_factor = self.config.base_factor
        self._drop_history: list[float] = []
        self._batches_since_adjust = 0

    @property
    def current_factor(self) -> float:
        """Current capacity factor."""
        return self._current_factor

    def get_capacity(self, batch_size: int) -> int:
        """Get expert capacity for current batch.

        Args:
            batch_size: Number of tokens in batch.

        Returns:
            Maximum tokens per expert.
        """
        return compute_expert_capacity(
            batch_size, self.num_experts, self._current_factor, self.top_k
        )

    def update(self, expert_ids: NDArray[np.integer]) -> float:
        """Update capacity factor based on observed routing.

        Args:
            expert_ids: [batch, top_k] expert assignments.

        Returns:
            Current capacity factor (possibly adjusted).
        """
        batch_size = expert_ids.shape[0]
        capacity = self.get_capacity(batch_size)

        # Count overflow
        expert_counts = count_expert_assignments(expert_ids, self.num_experts)
        overflow = np.sum(np.maximum(0, expert_counts - capacity))
        total_assignments = batch_size * self.top_k
        drop_rate = overflow / total_assignments if total_assignments > 0 else 0.0

        # Record drop rate
        self._drop_history.append(drop_rate)
        if len(self._drop_history) > self.config.window_size:
            self._drop_history.pop(0)

        # Check if we should adjust
        self._batches_since_adjust += 1
        if self._batches_since_adjust >= self.config.cooldown_batches:
            self._maybe_adjust()

        return self._current_factor

    def _maybe_adjust(self) -> None:
        """Adjust capacity factor if needed."""
        if len(self._drop_history) < self.config.window_size:
            return

        avg_drop = sum(self._drop_history) / len(self._drop_history)

        if avg_drop > self.config.target_drop_rate:
            # Increase capacity
            new_factor = min(
                self._current_factor + self.config.increase_step,
                self.config.max_factor,
            )
            if new_factor != self._current_factor:
                self._current_factor = new_factor
                self._batches_since_adjust = 0
        elif avg_drop < self.config.decrease_threshold:
            # Decrease capacity (save memory/compute)
            new_factor = max(
                self._current_factor - self.config.decrease_step,
                self.config.min_factor,
            )
            if new_factor != self._current_factor:
                self._current_factor = new_factor
                self._batches_since_adjust = 0

    def reset(self) -> None:
        """Reset to initial state."""
        self._current_factor = self.config.base_factor
        self._drop_history.clear()
        self._batches_since_adjust = 0

    def get_stats(self) -> dict:
        """Get current state and statistics.

        Returns:
            Dictionary with current factor, recent drop rates, etc.
        """
        return {
            "current_factor": self._current_factor,
            "recent_drop_rate": (
                sum(self._drop_history) / len(self._drop_history) if self._drop_history else 0.0
            ),
            "min_recent_drop": min(self._drop_history) if self._drop_history else 0.0,
            "max_recent_drop": max(self._drop_history) if self._drop_history else 0.0,
            "batches_recorded": len(self._drop_history),
            "batches_since_adjust": self._batches_since_adjust,
            "config": {
                "base_factor": self.config.base_factor,
                "min_factor": self.config.min_factor,
                "max_factor": self.config.max_factor,
                "target_drop_rate": self.config.target_drop_rate,
            },
        }


def dynamic_capacity(
    expert_ids: NDArray[np.integer],
    num_experts: int,
    target_utilization: float = 0.9,
    min_factor: float = 1.0,
    max_factor: float = 2.0,
) -> int:
    """Compute capacity for a single batch based on actual routing.

    This is a per-batch capacity adjustment that looks at the actual
    expert load distribution and computes sufficient capacity.

    For inference, this can be more efficient than using a fixed high
    capacity factor, since it adapts to the actual routing pattern.

    Args:
        expert_ids: [batch, top_k] expert assignments for current batch.
        num_experts: Number of experts.
        target_utilization: Target fraction of capacity to use (0.9 = 10% headroom).
        min_factor: Minimum capacity factor.
        max_factor: Maximum capacity factor.

    Returns:
        Capacity per expert for this batch.
    """
    batch_size, top_k = expert_ids.shape

    # Count actual expert assignments
    expert_counts = count_expert_assignments(expert_ids, num_experts)
    max_load = int(np.max(expert_counts))

    # Compute capacity with headroom
    if target_utilization > 0:
        capacity = max(1, math.ceil(max_load / target_utilization))
    else:
        capacity = max_load

    # Clamp to factor bounds
    min_capacity = compute_expert_capacity(batch_size, num_experts, min_factor, top_k)
    max_capacity = compute_expert_capacity(batch_size, num_experts, max_factor, top_k)
    capacity = max(min_capacity, min(capacity, max_capacity))

    return capacity
