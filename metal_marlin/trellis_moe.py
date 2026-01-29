"""Trellis MoE: Mixture-of-Experts with trellis-quantized weights.

This module provides expert implementations using trellis-quantized weights
for efficient inference on Apple Silicon via Metal Performance Shaders.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trellis_linear import TrellisLinear
from .trellis_loader import TrellisWeight


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
