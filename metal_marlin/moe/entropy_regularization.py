"""MoE-Gate Entropy Regularization.

This module implements entropy-based regularization for MoE routing to encourage
diverse expert selection and prevent routing collapse to a small subset of experts.

The entropy regularization loss consists of three components:
1. Entropy Loss: Maximizes the Shannon entropy of routing distributions
   L_entropy = -mean(-sum(p * log(p))) = mean(sum(p * log(p)))
   
2. Balance Loss: Minimizes variance in expert utilization
   L_balance = std(loads) / mean(loads)  (coefficient of variation)
   
3. Z-Loss: Prevents extreme logit values that can cause numerical instability
   L_z = mean(log(sum(exp(logits)))^2)

Total auxiliary loss: L_aux = w_e * L_entropy + w_b * L_balance + w_z * L_z

Example:
    >>> from metal_marlin.moe.entropy_regularization import EntropyRegularizer
    >>> regularizer = EntropyRegularizer(entropy_weight=0.01, balance_weight=0.01)
    >>> 
    >>> # During training
    >>> router_logits = router(hidden_states)  # [batch, num_experts]
    >>> aux_loss = regularizer(router_logits)
    >>> total_loss = moe_loss + aux_loss
    >>> 
    >>> # Access individual components
    >>> print(f"Entropy: {regularizer.last_entropy:.4f}")
    >>> print(f"Balance: {regularizer.last_balance_loss:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from metal_marlin.metal_dispatch import MetalKernelLibrary


@dataclass
class EntropyRegularizationOutput:
    """Output from entropy regularization computation.
    
    Attributes:
        total_loss: Combined auxiliary loss (scalar)
        entropy_loss: Entropy component of loss (scalar)
        balance_loss: Load balance component (scalar)
        z_loss: Z-loss component for logit stability (scalar)
        expert_loads: Average load per expert [num_experts]
        expert_importance: Importance score per expert [num_experts]
    """
    total_loss: torch.Tensor
    entropy_loss: torch.Tensor
    balance_loss: torch.Tensor
    z_loss: torch.Tensor
    expert_loads: torch.Tensor
    expert_importance: torch.Tensor | None = None


class EntropyRegularizer(nn.Module):
    """Entropy regularization for MoE gate routing.
    
    Encourages diverse expert selection by maximizing the entropy of routing
    distributions and balancing expert loads. This prevents the common MoE
    problem of "expert collapse" where only a few experts are used.
    
    The regularization combines three objectives:
    1. Maximize routing entropy (diverse expert selection per token)
    2. Balance expert loads (equal utilization across experts)
    3. Stabilize logits (prevent extreme values that hurt training)
    
    Args:
        num_experts: Number of experts in the MoE layer
        entropy_weight: Weight for entropy loss component (default: 0.01)
        balance_weight: Weight for balance loss component (default: 0.01)
        z_weight: Weight for z-loss component (default: 0.001)
        temperature: Softmax temperature for routing (default: 1.0)
        use_metal: Whether to use Metal kernel acceleration (default: True)
        
    Example:
        >>> regularizer = EntropyRegularizer(
        ...     num_experts=64,
        ...     entropy_weight=0.01,
        ...     balance_weight=0.01,
        ... )
        >>> router_logits = torch.randn(32, 64)  # batch=32, 64 experts
        >>> output = regularizer(router_logits)
        >>> loss = output.total_loss
        >>> print(f"Expert loads: {output.expert_loads}")
    """
    
    def __init__(
        self,
        num_experts: int,
        entropy_weight: float = 0.01,
        balance_weight: float = 0.01,
        z_weight: float = 0.001,
        temperature: float = 1.0,
        use_metal: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.entropy_weight = entropy_weight
        self.balance_weight = balance_weight
        self.z_weight = z_weight
        self.temperature = temperature
        self.use_metal = use_metal
        
        # Statistics tracking
        self.register_buffer("_expert_loads", torch.zeros(num_experts))
        self.register_buffer("_entropy_history", torch.zeros(100))
        self.register_buffer("_balance_history", torch.zeros(100))
        self._history_idx: int = 0
        
        # Cache for last computation
        self.last_entropy: float = 0.0
        self.last_balance_loss: float = 0.0
        self.last_z_loss: float = 0.0
        self.last_total_loss: float = 0.0
    
    def forward(
        self,
        router_logits: torch.Tensor,
        topk_indices: torch.Tensor | None = None,
        topk_probs: torch.Tensor | None = None,
    ) -> EntropyRegularizationOutput:
        """Compute entropy regularization loss.
        
        Args:
            router_logits: Router output logits [num_tokens, num_experts]
            topk_indices: Selected expert indices [num_tokens, topk] (optional)
            topk_probs: Selected expert probabilities [num_tokens, topk] (optional)
            
        Returns:
            EntropyRegularizationOutput with loss components and expert loads
        """
        num_tokens = router_logits.shape[0]
        
        # Compute routing probabilities
        router_probs = F.softmax(router_logits / self.temperature, dim=-1)
        
        # Compute entropy: H = -sum(p * log(p))
        # We want to maximize entropy, so entropy_loss = -H (minimize negative)
        entropy = self._compute_entropy(router_probs)
        entropy_loss = -entropy * self.entropy_weight  # Negative because we want to maximize entropy
        
        # Compute load balance loss
        expert_loads = router_probs.mean(dim=0)  # [num_experts]
        balance_loss = self._compute_balance_loss(expert_loads)
        
        # Compute z-loss (prevents extreme logits)
        z_loss = self._compute_z_loss(router_logits)
        
        # Combine losses
        total_loss = entropy_loss + balance_loss + z_loss
        
        # Update statistics
        self._expert_loads.copy_(expert_loads.detach())
        self._update_history(entropy.item(), balance_loss.item())
        
        # Cache values
        self.last_entropy = entropy.item()
        self.last_balance_loss = balance_loss.item()
        self.last_z_loss = z_loss.item()
        self.last_total_loss = total_loss.item()
        
        return EntropyRegularizationOutput(
            total_loss=total_loss,
            entropy_loss=entropy_loss,
            balance_loss=balance_loss,
            z_loss=z_loss,
            expert_loads=expert_loads.detach(),
        )
    
    def _compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute Shannon entropy: H = -sum(p * log(p)).
        
        Args:
            probs: Probability distribution [num_tokens, num_experts]
            
        Returns:
            Mean entropy across tokens (scalar)
        """
        # Numerical stability: clamp probabilities
        probs_clamped = probs.clamp(min=1e-6)
        
        # Entropy per token
        token_entropy = -(probs * torch.log(probs_clamped)).sum(dim=-1)
        
        # Mean across batch
        return token_entropy.mean()
    
    def _compute_balance_loss(self, expert_loads: torch.Tensor) -> torch.Tensor:
        """Compute load balance loss using coefficient of variation.
        
        Args:
            expert_loads: Average load per expert [num_experts]
            
        Returns:
            Balance loss (scalar)
        """
        mean_load = expert_loads.mean()
        std_load = expert_loads.std()
        
        # Coefficient of variation: CV = std / mean
        # Lower CV = more balanced
        cv = std_load / (mean_load + 1e-6)
        
        return cv * self.balance_weight
    
    def _compute_z_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute z-loss to prevent extreme logit values.
        
        The z-loss penalizes large log-sum-exp values which correspond
        to peaked (confident) routing distributions.
        
        Args:
            logits: Router logits [num_tokens, num_experts]
            
        Returns:
            Z-loss (scalar)
        """
        # log(sum(exp(logits))) is the normalizing constant of softmax
        lse = torch.logsumexp(logits, dim=-1)
        
        # Square to penalize large values more
        z_loss = (lse ** 2).mean()
        
        return z_loss * self.z_weight
    
    def _update_history(self, entropy: float, balance: float) -> None:
        """Update rolling history of entropy and balance values."""
        idx = self._history_idx % 100
        self._entropy_history[idx] = entropy
        self._balance_history[idx] = balance
        self._history_idx += 1
    
    def get_statistics(self) -> dict[str, float]:
        """Get statistics about entropy regularization.
        
        Returns:
            Dictionary with entropy, balance, and load statistics
        """
        valid_entries = min(self._history_idx, 100)
        if valid_entries == 0:
            return {
                "mean_entropy": 0.0,
                "mean_balance": 0.0,
                "current_entropy": self.last_entropy,
                "current_balance": self.last_balance_loss,
            }
        
        return {
            "mean_entropy": self._entropy_history[:valid_entries].mean().item(),
            "mean_balance": self._balance_history[:valid_entries].mean().item(),
            "current_entropy": self.last_entropy,
            "current_balance": self.last_balance_loss,
            "current_z_loss": self.last_z_loss,
            "max_expert_load": self._expert_loads.max().item(),
            "min_expert_load": self._expert_loads.min().item(),
            "load_std": self._expert_loads.std().item(),
        }
    
    def get_expert_importance(self) -> torch.Tensor:
        """Compute importance score for each expert.
        
        Importance is based on both frequency of selection and the
        entropy of routing distributions.
        
        Returns:
            Importance scores [num_experts]
        """
        # Base importance is normalized load
        base_importance = self._expert_loads / (self._expert_loads.sum() + 1e-6)
        
        # Entropy bonus: experts with diverse routing get bonus
        # (This would need per-expert entropy, approximated here)
        uniform_load = 1.0 / self.num_experts
        entropy_bonus = -base_importance * torch.log(base_importance + 1e-6)
        max_entropy = -uniform_load * torch.log(torch.tensor(uniform_load))
        normalized_bonus = entropy_bonus / (max_entropy + 1e-6)
        
        return base_importance * (1.0 + normalized_bonus)


class MoEGateWithEntropy(nn.Module):
    """MoE Gate with integrated entropy regularization.
    
    This module combines the router computation with entropy regularization
    in a single interface for ease of use.
    
    Args:
        hidden_size: Dimension of input hidden states
        num_experts: Number of experts
        top_k: Number of experts to select per token
        entropy_weight: Weight for entropy regularization
        balance_weight: Weight for balance regularization
        
    Example:
        >>> gate = MoEGateWithEntropy(
        ...     hidden_size=768,
        ...     num_experts=64,
        ...     top_k=4,
        ...     entropy_weight=0.01,
        ... )
        >>> hidden_states = torch.randn(32, 768)
        >>> output = gate(hidden_states)
        >>> print(f"Selected experts: {output.indices}")
        >>> print(f"Aux loss: {output.aux_loss}")
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 4,
        entropy_weight: float = 0.01,
        balance_weight: float = 0.01,
        z_weight: float = 0.001,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router linear layer
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Entropy regularizer
        self.regularizer = EntropyRegularizer(
            num_experts=num_experts,
            entropy_weight=entropy_weight,
            balance_weight=balance_weight,
            z_weight=z_weight,
            temperature=temperature,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        compute_aux_loss: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts with entropy regularization.
        
        Args:
            hidden_states: Input [num_tokens, hidden_size]
            compute_aux_loss: Whether to compute auxiliary loss
            
        Returns:
            Tuple of (expert_indices, expert_weights, aux_loss)
            - expert_indices: [num_tokens, top_k] selected expert indices
            - expert_weights: [num_tokens, top_k] routing weights
            - aux_loss: Scalar auxiliary loss for training
        """
        # Compute router logits
        router_logits = self.router(hidden_states)
        
        # Select top-k experts
        topk_weights, topk_indices = torch.topk(
            torch.softmax(router_logits, dim=-1),
            k=self.top_k,
            dim=-1,
        )
        
        # Renormalize weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # Compute auxiliary loss
        if compute_aux_loss and self.training:
            reg_output = self.regularizer(router_logits, topk_indices, topk_weights)
            aux_loss = reg_output.total_loss
        else:
            aux_loss = torch.tensor(0.0, device=hidden_states.device)
        
        return topk_indices, topk_weights, aux_loss
    
    def get_load_balance_stats(self) -> dict[str, Any]:
        """Get load balancing statistics."""
        return self.regularizer.get_statistics()


def compute_entropy_regularization(
    router_probs: torch.Tensor,
    entropy_weight: float = 0.01,
    balance_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Functional interface for entropy regularization.
    
    Args:
        router_probs: Routing probabilities [num_tokens, num_experts]
        entropy_weight: Weight for entropy loss
        balance_weight: Weight for balance loss
        
    Returns:
        Tuple of (total_loss, components_dict)
        - total_loss: Scalar combined loss
        - components: Dict with 'entropy', 'balance', 'loads' tensors
    """
    # Entropy: H = -sum(p * log(p))
    probs_clamped = router_probs.clamp(min=1e-6)
    token_entropy = -(router_probs * torch.log(probs_clamped)).sum(dim=-1)
    entropy = token_entropy.mean()
    entropy_loss = -entropy * entropy_weight
    
    # Balance loss
    expert_loads = router_probs.mean(dim=0)
    mean_load = expert_loads.mean()
    std_load = expert_loads.std()
    cv = std_load / (mean_load + 1e-6)
    balance_loss = cv * balance_weight
    
    total_loss = entropy_loss + balance_loss
    
    components = {
        "entropy": entropy,
        "balance": cv,
        "loads": expert_loads,
    }
    
    return total_loss, components
