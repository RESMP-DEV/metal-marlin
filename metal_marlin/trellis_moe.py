"""Trellis MoE: Mixture-of-Experts with trellis-quantized weights.

This module provides expert implementations using trellis-quantized weights
for efficient inference on Apple Silicon via Metal Performance Shaders.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trellis_linear import TrellisLinear
from .trellis_loader import TrellisWeight

__all__ = [
    "TrellisMoEConfig",
    "TrellisMoELayer",
    "TrellisExpert",
]


@dataclass
class TrellisMoEConfig:
    """Configuration for Trellis MoE layer.

    Attributes:
        num_experts: Total number of experts (e.g., 64 for GLM-4.7-Flash).
        num_experts_per_tok: Number of experts activated per token (top-k).
        hidden_size: Model hidden dimension.
        intermediate_size: FFN intermediate dimension.
        bits: Quantization bit width (2, 3, or 4).
    """

    num_experts: int
    hidden_size: int
    num_experts_per_tok: int = 2
    intermediate_size: int | None = None
    bits: int = 4

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {self.num_experts}")
        if self.bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4, got {self.bits}")
        if self.intermediate_size is None:
            # Default: ~2.75x hidden_size (matches GLM-4.7-Flash)
            self.intermediate_size = int(self.hidden_size * 2.75)


class TrellisMoELayer(nn.Module):
    """Mixture-of-Experts layer with trellis-quantized expert weights.

    Implements sparse MoE where each token is routed to top-k experts.
    Expert weights are stored in trellis-quantized format for memory efficiency.

    Forward pass:
        1. Router computes expert logits for each token
        2. Top-k experts are selected per token
        3. Selected experts are executed
        4. Outputs are combined weighted by routing probabilities
    """

    def __init__(
        self,
        config: TrellisMoEConfig,
        layer_weights: dict[str, dict],
        router_weight: torch.Tensor,
        layer_idx: int,
        device: str = "mps",
    ) -> None:
        """Initialize TrellisMoELayer.

        Args:
            config: MoE configuration.
            layer_weights: Dictionary of expert weights. Keys should be like
                "experts.{i}.gate_proj", "experts.{i}.up_proj", "experts.{i}.down_proj"
                or "experts.{i}.gate_up_proj" (fused).
            router_weight: Router weight tensor [num_experts, hidden_size].
            layer_idx: Layer index for weight key prefixes.
            device: Device to place weights on.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.device = device

        # Router: linear projection to expert logits
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False, device=device)
        self.router.weight.data = router_weight.to(device=device, dtype=torch.float32)

        # Build experts from weights
        self.experts = nn.ModuleList()
        for i in range(config.num_experts):
            expert = self._build_expert(layer_weights, i, device)
            self.experts.append(expert)

    def _build_expert(
        self, layer_weights: dict[str, dict], expert_idx: int, device: str
    ) -> nn.Module:
        """Build a single expert from weight dict.

        Handles both fused (gate_up_proj) and separate (gate_proj, up_proj) formats.
        """
        # Check for fused vs separate weight format
        fused_key = f"experts.{expert_idx}.gate_up_proj"
        gate_key = f"experts.{expert_idx}.gate_proj"
        up_key = f"experts.{expert_idx}.up_proj"
        down_key = f"experts.{expert_idx}.down_proj"

        if fused_key in layer_weights:
            # Fused gate_up format - use FusedExpert
            return _FusedTrellisExpert(
                gate_up_weight=self._dict_to_linear(layer_weights[fused_key], device),
                down_weight=self._dict_to_linear(layer_weights[down_key], device),
                config=self.config,
            )
        elif gate_key in layer_weights:
            # Separate gate/up format
            return TrellisExpert(
                gate_proj=self._dict_to_linear(layer_weights[gate_key], device),
                up_proj=self._dict_to_linear(layer_weights[up_key], device),
                down_proj=self._dict_to_linear(layer_weights[down_key], device),
            )
        else:
            # Fallback: create dummy expert for testing
            return _DummyExpert(self.config.hidden_size, self.config.intermediate_size, device)

    def _dict_to_linear(self, weight_dict: dict, device: str) -> TrellisLinear:
        """Convert weight dict to TrellisLinear module."""
        # Create TrellisWeight-like object from dict
        weight = _DictWeight(weight_dict)
        return TrellisLinear.from_trellis_weight(weight, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MoE layer.

        Args:
            x: Input tensor of shape [batch, seq, hidden] or [tokens, hidden].

        Returns:
            Output tensor with same shape as input.
        """
        # Handle both 2D and 3D inputs
        original_shape = x.shape
        if x.dim() == 3:
            batch, seq_len, hidden = x.shape
            x = x.view(-1, hidden)  # [batch*seq, hidden]
        else:
            batch, seq_len = None, None

        num_tokens, hidden = x.shape

        # Route tokens to experts
        router_logits = self.router(x.float())  # [tokens, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        topk_weights, topk_indices = torch.topk(
            routing_weights, self.config.num_experts_per_tok, dim=-1
        )
        # Normalize weights for selected experts
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Compute expert outputs
        output = torch.zeros_like(x)

        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            expert_mask = (topk_indices == i).any(dim=-1)  # [tokens]
            if not expert_mask.any():
                continue

            # Get tokens for this expert
            expert_tokens = x[expert_mask]  # [n, hidden]

            # Run expert
            expert_out = expert(expert_tokens)  # [n, hidden]

            # Get weights for this expert
            # Expert weight is the topk_weight where topk_indices == i
            weight_positions = (topk_indices == i).float()  # [tokens, k]
            expert_weights = (topk_weights * weight_positions).sum(dim=-1)  # [tokens]
            expert_weights = expert_weights[expert_mask].unsqueeze(-1)  # [n, 1]

            # Accumulate weighted output
            output[expert_mask] += expert_out * expert_weights

        # Restore original shape
        if batch is not None:
            output = output.view(batch, seq_len, hidden)

        return output.to(x.dtype)


class _DictWeight:
    """Adapter to make a dict look like TrellisWeight for from_trellis_weight."""

    def __init__(self, d: dict) -> None:
        self.indices = d["indices"]
        self.scales = d["scales"]
        self.su = d["su"]
        self.sv = d["sv"]
        self.bits = d["bits"]
        self.original_shape = d["original_shape"]


class _FusedTrellisExpert(nn.Module):
    """Expert with fused gate_up projection."""

    def __init__(
        self,
        gate_up_weight: TrellisLinear,
        down_weight: TrellisLinear,
        config: TrellisMoEConfig,
    ) -> None:
        super().__init__()
        self.gate_up = gate_up_weight
        self.down = down_weight
        self.intermediate_size = config.intermediate_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused gate_up: output is [gate, up] concatenated
        gate_up_out = self.gate_up(x)

        # Check if truly fused (2x intermediate) or just named oddly
        if gate_up_out.shape[-1] == 2 * self.intermediate_size:
            gate, up = gate_up_out.chunk(2, dim=-1)
        else:
            # Not truly fused - use same output for both (fallback)
            gate = gate_up_out
            up = gate_up_out

        return self.down(F.silu(gate) * up)


class _DummyExpert(nn.Module):
    """Dummy expert for testing when weights aren't available."""

    def __init__(self, hidden_size: int, intermediate_size: int, device: str) -> None:
        super().__init__()
        # Use regular linear for testing
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False, device=device)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False, device=device)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False, device=device)

        # Initialize with small values
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.normal_(self.down.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TrellisExpert(nn.Module):
    """Single expert with trellis-quantized weights.

    Implements the SwiGLU structure:
        out = down_proj(silu(gate_proj(x)) * up_proj(x))

    Each projection uses trellis-quantized weights for memory efficiency.
    """

    def __init__(
        self,
        gate_proj: TrellisLinear,
        up_proj: TrellisLinear,
        down_proj: TrellisLinear,
    ):
        """Initialize TrellisExpert.

        Args:
            gate_proj: TrellisLinear for gate projection.
            up_proj: TrellisLinear for up projection.
            down_proj: TrellisLinear for down projection.
        """
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the expert.

        Args:
            x: Input tensor [..., hidden_size].

        Returns:
            Output tensor [..., hidden_size].
        """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

    @classmethod
    def from_trellis_weights(
        cls,
        layer_weights: dict[str, TrellisWeight],
        expert_idx: int,
        layer_idx: int,
        device: str = "mps",
    ) -> TrellisExpert:
        """Create TrellisExpert from layer weights dictionary.

        Args:
            layer_weights: Dictionary mapping weight names to TrellisWeight objects.
            expert_idx: Index of the expert to load.
            layer_idx: Index of the transformer layer.
            device: Device to place weights on (default: "mps").

        Returns:
            TrellisExpert initialized with the specified expert's weights.

        Raises:
            KeyError: If required weights are not found in layer_weights.
        """
        prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
        return cls(
            gate_proj=TrellisLinear.from_trellis_weight(
                layer_weights[f"{prefix}.gate_proj.weight"], device=device
            ),
            up_proj=TrellisLinear.from_trellis_weight(
                layer_weights[f"{prefix}.up_proj.weight"], device=device
            ),
            down_proj=TrellisLinear.from_trellis_weight(
                layer_weights[f"{prefix}.down_proj.weight"], device=device
            ),
        )

    def extra_repr(self) -> str:
        """String representation for printing."""
        return (
            f"gate_proj={self.gate_proj.in_features}x{self.gate_proj.out_features}, "
            f"up_proj={self.up_proj.in_features}x{self.up_proj.out_features}, "
            f"down_proj={self.down_proj.in_features}x{self.down_proj.out_features}"
        )
