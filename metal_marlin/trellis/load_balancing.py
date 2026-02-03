"""MoE load balancing loss for expert utilization optimization.

Implements auxiliary losses to encourage balanced expert utilization:
- Load balance loss: penalizes correlation between routing probs and expert loads
- Z-loss: regularizes router logits to prevent collapse

Based on Switch Transformer / GShard formulations.

Usage:
    from metal_marlin.trellis.load_balancing import (
        LoadBalancingConfig,
        ExpertLoadTracker,
    )

    # Configure load balancing
    config = LoadBalancingConfig(alpha=0.01, z_loss_coeff=0.001)
    tracker = ExpertLoadTracker(num_experts=64, device=torch.device("mps"))

    # During forward pass, update tracker with routing decisions
    tracker.update(selected_experts, routing_weights, router_logits)

    # After forward pass, compute auxiliary loss
    aux_loss, metrics = tracker.compute_loss(config)
    total_loss = main_loss + aux_loss

    # Reset for next step
    tracker.reset()
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class LoadBalancingConfig:
    """Configuration for MoE load balancing loss.

    The load balancing loss encourages uniform expert utilization by penalizing
    the correlation between routing probabilities and actual expert loads.

    Loss = alpha * num_experts * sum_i(f_i * P_i) where:
    - f_i = fraction of tokens routed to expert i (actual load)
    - P_i = average routing probability for expert i (router's prediction)

    Perfect balance (uniform routing) yields loss = alpha.
    Imbalanced routing (all tokens to one expert) yields loss = alpha * num_experts.

    Attributes:
        alpha: Loss coefficient. Default 0.01 balances quality vs load balance.
            Higher values prioritize balance over accuracy.
        z_loss_coeff: Z-loss coefficient to penalize large router logits.
            Helps prevent router collapse. Set to 0.0 to disable.
        enabled: Whether to track and compute load balancing loss.
    """

    alpha: float = 0.01
    z_loss_coeff: float = 0.001
    enabled: bool = True


@dataclass
class ExpertLoadTracker:
    """Tracks expert utilization statistics for load balancing loss computation.

    Accumulates routing statistics across forward passes within a training step,
    then computes the auxiliary loss when requested.

    Example:
        tracker = ExpertLoadTracker(num_experts=64, device=torch.device("mps"))

        # In forward pass
        tracker.update(selected_experts, routing_weights, router_logits)

        # After forward
        loss, metrics = tracker.compute_loss(config)
        tracker.reset()  # For next step
    """

    num_experts: int
    device: torch.device = field(default_factory=lambda: torch.device("mps"))

    # Accumulated statistics (reset after each loss computation)
    _token_counts: torch.Tensor | None = field(default=None, repr=False)
    _prob_sums: torch.Tensor | None = field(default=None, repr=False)
    _logit_sq_sums: torch.Tensor | None = field(default=None, repr=False)
    _total_tokens: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset accumulated statistics for a new training step."""
        self._token_counts = torch.zeros(
            self.num_experts, dtype=torch.float32, device=self.device
        )
        self._prob_sums = torch.zeros(
            self.num_experts, dtype=torch.float32, device=self.device
        )
        self._logit_sq_sums = torch.zeros(1, dtype=torch.float32, device=self.device)
        self._total_tokens = 0

    def update(
        self,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ) -> None:
        """Update statistics from a forward pass.

        Args:
            selected_experts: [batch_size, top_k] indices of selected experts.
            routing_weights: [batch_size, top_k] normalized routing probabilities.
            router_logits: Optional [batch_size, num_experts] raw router logits.
        """
        # Ensure tensors are initialized (should always be true after __post_init__)
        assert self._token_counts is not None
        assert self._prob_sums is not None
        assert self._logit_sq_sums is not None

        batch_size = selected_experts.shape[0]

        # Count tokens per expert (scatter_add_ requires float)
        flat_experts = selected_experts.reshape(-1)
        ones = torch.ones(
            flat_experts.shape[0], dtype=torch.float32, device=self.device
        )
        self._token_counts.scatter_add_(0, flat_experts.long(), ones)

        # Sum routing probabilities per expert
        flat_probs = routing_weights.reshape(-1).float()
        self._prob_sums.scatter_add_(0, flat_experts.long(), flat_probs)

        # Track total tokens
        self._total_tokens += batch_size

        # Z-loss: penalize large logits to prevent router collapse
        if router_logits is not None:
            logsumexp = torch.logsumexp(router_logits.float(), dim=-1)
            self._logit_sq_sums += (logsumexp**2).sum()

    def compute_loss(
        self, config: LoadBalancingConfig
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute load balancing loss from accumulated statistics.

        Returns:
            Tuple of (loss tensor for backprop, metrics dict for logging).
            Metrics include:
            - load_balance_loss: The auxiliary loss value
            - z_loss: Router logit regularization loss
            - expert_utilization_cv: Coefficient of variation of expert loads
            - max_load_ratio: Ratio of most-loaded to average load
        """
        if self._total_tokens == 0:
            zero = torch.tensor(0.0, device=self.device, requires_grad=True)
            return zero, {
                "load_balance_loss": 0.0,
                "z_loss": 0.0,
                "expert_utilization_cv": 0.0,
                "max_load_ratio": 1.0,
            }

        # Ensure tensors are initialized
        assert self._token_counts is not None
        assert self._prob_sums is not None
        assert self._logit_sq_sums is not None

        # f_i = fraction of tokens routed to expert i
        total_assignments = self._token_counts.sum().clamp(min=1e-8)
        f = self._token_counts / total_assignments

        # P_i = mean routing probability for expert i
        expert_selection_counts = self._token_counts.clamp(min=1)
        P = self._prob_sums / expert_selection_counts

        # Load balance loss: alpha * num_experts * sum(f_i * P_i)
        load_loss = config.alpha * self.num_experts * (f * P).sum()

        # Z-loss: penalize large router logits
        z_loss = torch.tensor(0.0, device=self.device)
        if config.z_loss_coeff > 0:
            z_loss = config.z_loss_coeff * self._logit_sq_sums / self._total_tokens

        total_loss = load_loss + z_loss

        # Compute metrics for logging
        with torch.no_grad():
            loads = self._token_counts / total_assignments
            cv = loads.std() / loads.mean().clamp(min=1e-8)
            max_load_ratio = loads.max() / (1.0 / self.num_experts)

        metrics = {
            "load_balance_loss": load_loss.item(),
            "z_loss": z_loss.item() if isinstance(z_loss, torch.Tensor) else z_loss,
            "expert_utilization_cv": cv.item(),
            "max_load_ratio": max_load_ratio.item(),
        }

        return total_loss, metrics

    def get_expert_loads(self) -> torch.Tensor:
        """Get current expert load distribution as fractions summing to 1."""
        if self._total_tokens == 0:
            return torch.ones(self.num_experts, device=self.device) / self.num_experts
        return self._token_counts / self._token_counts.sum().clamp(min=1)

    def get_hot_experts(self, threshold: float = 1.5) -> list[int]:
        """Get experts with load > threshold * average."""
        loads = self.get_expert_loads()
        avg_load = 1.0 / self.num_experts
        return (loads > threshold * avg_load).nonzero(as_tuple=True)[0].tolist()

    def get_cold_experts(self, threshold: float = 0.5) -> list[int]:
        """Get experts with load < threshold * average."""
        loads = self.get_expert_loads()
        avg_load = 1.0 / self.num_experts
        mask = (loads < threshold * avg_load) & (loads > 0)
        return mask.nonzero(as_tuple=True)[0].tolist()

    def get_dead_experts(self) -> list[int]:
        """Get experts that received no tokens."""
        if self._total_tokens == 0:
            return []
        return (self._token_counts == 0).nonzero(as_tuple=True)[0].tolist()


def compute_load_balance_loss_inline(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    router_logits: torch.Tensor,
    num_experts: int,
    alpha: float = 0.01,
    z_loss_coeff: float = 0.001,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute load balancing loss directly without accumulation.

    This is a convenience function for single-pass loss computation.
    For accumulating across multiple forward passes, use ExpertLoadTracker.

    Args:
        selected_experts: [batch_size, top_k] indices of selected experts.
        routing_weights: [batch_size, top_k] normalized routing probabilities.
        router_logits: [batch_size, num_experts] raw router logits.
        num_experts: Total number of experts.
        alpha: Load balance loss coefficient.
        z_loss_coeff: Z-loss coefficient for logit regularization.

    Returns:
        Tuple of (loss tensor, metrics dict).
    """
    device = selected_experts.device
    batch_size = selected_experts.shape[0]

    # Count tokens per expert
    flat_experts = selected_experts.reshape(-1)
    token_counts = torch.zeros(num_experts, dtype=torch.float32, device=device)
    ones = torch.ones(flat_experts.shape[0], dtype=torch.float32, device=device)
    token_counts.scatter_add_(0, flat_experts.long(), ones)

    # Sum routing probabilities per expert
    prob_sums = torch.zeros(num_experts, dtype=torch.float32, device=device)
    flat_probs = routing_weights.reshape(-1).float()
    prob_sums.scatter_add_(0, flat_experts.long(), flat_probs)

    # f_i = fraction of tokens routed to expert i
    total_assignments = token_counts.sum().clamp(min=1e-8)
    f = token_counts / total_assignments

    # P_i = mean routing probability for expert i
    expert_selection_counts = token_counts.clamp(min=1)
    P = prob_sums / expert_selection_counts

    # Load balance loss
    load_loss = alpha * num_experts * (f * P).sum()

    # Z-loss
    z_loss = torch.tensor(0.0, device=device)
    if z_loss_coeff > 0:
        logsumexp = torch.logsumexp(router_logits.float(), dim=-1)
        z_loss = z_loss_coeff * (logsumexp**2).mean()

    total_loss = load_loss + z_loss

    # Metrics
    with torch.no_grad():
        cv = f.std() / f.mean().clamp(min=1e-8)
        max_load_ratio = f.max() / (1.0 / num_experts)

    metrics = {
        "load_balance_loss": load_loss.item(),
        "z_loss": z_loss.item(),
        "expert_utilization_cv": cv.item(),
        "max_load_ratio": max_load_ratio.item(),
    }

    return total_loss, metrics
