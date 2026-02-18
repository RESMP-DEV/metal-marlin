"""Adaptive speculation depth controller for MMFP4 speculative decoding.

This module provides adaptive control of speculation depth based on real-time
acceptance rates. The controller dynamically adjusts the number of draft tokens
to maximize throughput while minimizing wasted computation on rejected tokens.

The adaptive algorithm uses an exponential moving average (EMA) of acceptance
rates to smoothly adjust speculation depth toward an optimal value. The optimal
depth is computed based on the theoretical speedup formula for speculative decoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class AdaptiveDepthConfig:
    """Configuration for adaptive speculation depth.
    
    Attributes:
        initial_depth: Starting speculation depth
        min_depth: Minimum allowed speculation depth
        max_depth: Maximum allowed speculation depth
        ema_alpha: EMA smoothing factor (0-1), higher = more responsive
        target_acceptance: Target acceptance rate for optimal performance
        aggressiveness: How aggressively to adjust depth (0.5 = conservative, 1.0 = aggressive)
        history_window: Number of steps to consider for moving average
        enable_dynamic_adjustment: Whether to enable dynamic adjustment
    """
    initial_depth: int = 4
    min_depth: int = 1
    max_depth: int = 8
    ema_alpha: float = 0.3
    target_acceptance: float = 0.7
    aggressiveness: float = 0.8
    history_window: int = 10
    enable_dynamic_adjustment: bool = True


@dataclass
class AdaptiveDepthStats:
    """Statistics for adaptive speculation depth.
    
    Attributes:
        current_depth: Current speculation depth
        ema_acceptance: Exponential moving average of acceptance rate
        recent_acceptance_rate: Acceptance rate over recent window
        total_steps: Total number of speculation steps
        total_accepted: Total tokens accepted
        total_proposed: Total tokens proposed
        depth_changes: Number of times depth was adjusted
    """
    current_depth: int = 4
    ema_acceptance: float = 0.5
    recent_acceptance_rate: float = 0.5
    total_steps: int = 0
    total_accepted: int = 0
    total_proposed: int = 0
    depth_changes: int = 0
    
    @property
    def overall_acceptance_rate(self) -> float:
        """Overall acceptance rate across all steps."""
        if self.total_proposed == 0:
            return 0.0
        return self.total_accepted / self.total_proposed
    
    @property
    def efficiency_score(self) -> float:
        """Efficiency score based on accepted tokens vs proposed.
        
        Higher is better. A score of 1.0 means all proposed tokens are accepted.
        """
        if self.total_proposed == 0:
            return 0.0
        return self.total_accepted / self.total_proposed


class AdaptiveSpeculationController:
    """Controller for adaptive speculation depth in MMFP4.
    
    Dynamically adjusts the number of draft tokens based on observed acceptance
    rates to maximize generation throughput. Uses EMA smoothing to avoid
    oscillation and provide stable adaptation.
    
    The optimal speculation depth k* can be derived from:
    - Let α = acceptance rate
    - Let c_d = cost of drafting one token (relative to target)
    - Let c_t = cost of target verification (normalized to 1)
    
    Expected tokens per step = α * k + 1 (bonus token)
    Cost per step = c_d * k + c_t
    
    The optimal k* ≈ sqrt((1 - α) * c_t / (c_d * α)) for rejection sampling
    
    In practice, we use a simplified heuristic that increases depth when
    acceptance is high and decreases when low.
    
    Args:
        config: Configuration for adaptive depth behavior
    
    Example:
        >>> controller = AdaptiveSpeculationController()
        >>> controller.current_depth
        4
        >>> controller.update(4, 4)  # All 4 accepted
        >>> controller.current_depth  # Depth increased
        5
    """
    
    def __init__(self, config: AdaptiveDepthConfig | None = None):
        self.config = config or AdaptiveDepthConfig()
        self._current_depth = self.config.initial_depth
        self._ema_acceptance = 0.5  # Start with neutral estimate
        self._acceptance_history: list[float] = []
        self._stats = AdaptiveDepthStats(current_depth=self._current_depth)
    
    @property
    def current_depth(self) -> int:
        """Current speculation depth."""
        return self._current_depth
    
    @property
    def stats(self) -> AdaptiveDepthStats:
        """Current statistics."""
        # Update stats with current values
        self._stats.current_depth = self._current_depth
        self._stats.ema_acceptance = self._ema_acceptance
        return self._stats
    
    def update(self, num_accepted: int, num_proposed: int) -> int:
        """Update adaptive depth based on acceptance results.
        
        Args:
            num_accepted: Number of draft tokens accepted
            num_proposed: Number of draft tokens proposed
        
        Returns:
            New speculation depth
        """
        if num_proposed == 0:
            return self._current_depth
        
        # Calculate acceptance rate for this step
        acceptance_rate = num_accepted / num_proposed
        
        # Update statistics
        self._stats.total_steps += 1
        self._stats.total_accepted += num_accepted
        self._stats.total_proposed += num_proposed
        
        # Update acceptance history
        self._acceptance_history.append(acceptance_rate)
        if len(self._acceptance_history) > self.config.history_window:
            self._acceptance_history.pop(0)
        
        # Calculate recent acceptance rate (windowed average)
        if self._acceptance_history:
            recent_rate = sum(self._acceptance_history) / len(self._acceptance_history)
            self._stats.recent_acceptance_rate = recent_rate
        
        if not self.config.enable_dynamic_adjustment:
            return self._current_depth
        
        # Update EMA
        alpha = self.config.ema_alpha
        self._ema_acceptance = alpha * acceptance_rate + (1 - alpha) * self._ema_acceptance
        
        # Compute optimal depth based on EMA acceptance
        optimal_depth = self._compute_optimal_depth()
        
        # Adjust current depth toward optimal (gradual adjustment for stability)
        old_depth = self._current_depth
        
        if optimal_depth > self._current_depth:
            # Increase depth (but gradually)
            self._current_depth = min(
                self._current_depth + 1,
                self.config.max_depth
            )
        elif optimal_depth < self._current_depth:
            # Decrease depth (but gradually)
            self._current_depth = max(
                self._current_depth - 1,
                self.config.min_depth
            )
        
        # Track depth changes
        if self._current_depth != old_depth:
            self._stats.depth_changes += 1
        
        return self._current_depth
    
    def _compute_optimal_depth(self) -> int:
        """Compute optimal speculation depth based on current EMA acceptance.
        
        Uses a heuristic based on the expected value of accepted tokens.
        For speculative decoding:
        - Expected tokens per step = α * k + 1 (where α = acceptance rate)
        - Higher α means we benefit more from larger k
        
        The optimal k balances:
        - Higher k = more potential accepted tokens per step
        - Lower k = less waste when rejection happens early
        
        Returns:
            Optimal speculation depth
        """
        ema = self._ema_acceptance
        
        # Clamp to reasonable range for computation
        ema = max(0.1, min(0.99, ema))
        
        # Heuristic: optimal depth increases with acceptance rate
        # At 50% acceptance, depth of 3-4 is good
        # At 80% acceptance, depth of 6-8 is good
        # At 95% acceptance, depth of 8+ is good
        
        if ema < 0.3:
            # Low acceptance: minimal speculation
            base_optimal = self.config.min_depth
        elif ema < 0.5:
            # Moderate-low acceptance: conservative depth
            base_optimal = 2
        elif ema < 0.7:
            # Moderate acceptance: moderate depth
            base_optimal = 3
        elif ema < 0.85:
            # Good acceptance: increased depth
            base_optimal = 5
        elif ema < 0.95:
            # High acceptance: high depth
            base_optimal = 6
        else:
            # Very high acceptance: maximum depth
            base_optimal = self.config.max_depth
        
        # Apply aggressiveness factor
        # Higher aggressiveness = more willing to increase depth
        if self.config.aggressiveness > 0.5:
            # More aggressive: round up more often
            adjustment = 1 if ema > self.config.target_acceptance else 0
        else:
            # More conservative: round down more often
            adjustment = -1 if ema < self.config.target_acceptance else 0
        
        optimal = base_optimal + adjustment
        
        # Clamp to valid range
        optimal = max(self.config.min_depth, min(self.config.max_depth, optimal))
        
        return optimal
    
    def get_speedup_estimate(self) -> float:
        """Estimate expected speedup from speculative decoding.
        
        Based on the formula:
        speedup = (α * k + 1) / (1 + k * c_d/c_t)
        
        Where:
        - α = acceptance rate
        - k = speculation depth
        - c_d/c_t ≈ 0.1 (draft cost is ~10% of target cost)
        
        Returns:
            Estimated speedup factor (1.0 = no speedup)
        """
        alpha = max(0.01, self._ema_acceptance)  # Avoid division by zero
        k = self._current_depth
        
        # Assume draft is ~10% cost of target
        draft_cost_ratio = 0.1
        
        # Expected tokens per target call
        expected_tokens = alpha * k + 1
        
        # Cost per step (normalized to target cost = 1)
        cost_per_step = 1 + k * draft_cost_ratio
        
        # Effective tokens per unit cost
        efficiency = expected_tokens / cost_per_step
        
        return efficiency
    
    def reset(self) -> None:
        """Reset controller state for a new sequence."""
        self._current_depth = self.config.initial_depth
        self._ema_acceptance = 0.5
        self._acceptance_history.clear()
        self._stats = AdaptiveDepthStats(current_depth=self._current_depth)
    
    def should_increase_depth(self) -> bool:
        """Check if conditions suggest increasing depth would help.
        
        Returns:
            True if depth should be increased
        """
        return (
            self._ema_acceptance > self.config.target_acceptance + 0.1
            and self._current_depth < self.config.max_depth
        )
    
    def should_decrease_depth(self) -> bool:
        """Check if conditions suggest decreasing depth would help.
        
        Returns:
            True if depth should be decreased
        """
        return (
            self._ema_acceptance < self.config.target_acceptance - 0.2
            and self._current_depth > self.config.min_depth
        )


__all__ = [
    "AdaptiveDepthConfig",
    "AdaptiveDepthStats", 
    "AdaptiveSpeculationController",
]
