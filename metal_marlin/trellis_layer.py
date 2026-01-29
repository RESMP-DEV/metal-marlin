"""Trellis layer modules for EXL3 trellis-quantized models.

Provides layer-level modules like MLP that combine TrellisLinear layers
for use with trellis-quantized models like GLM-4.7-Flash.

Usage:
    from metal_marlin.trellis_loader import TrellisModelLoader
    from metal_marlin.trellis_layer import TrellisDenseMLP

    loader = TrellisModelLoader("models/GLM-4.7-Flash-EXL3-3bpw")
    mlp = TrellisDenseMLP.from_loader(loader, layer_idx=0, device="mps")
    output = mlp(input_tensor)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trellis_linear import TrellisLinear

if TYPE_CHECKING:
    from .trellis_loader import TrellisModelLoader


class TrellisDenseMLP(nn.Module):
    """Dense MLP with trellis-quantized weights (for non-MoE layers).

    Implements the SwiGLU activation pattern used in GLM-4 and similar models:
    - gate_proj: Projects input to gate activations (with SiLU)
    - up_proj: Projects input to up values
    - down_proj: Projects the element-wise product back to hidden size

    This is used for layers 0 and 1 in GLM-4.7-Flash which use dense MLP
    instead of MoE.

    Attributes:
        gate_proj: TrellisLinear for gate projection.
        up_proj: TrellisLinear for up projection.
        down_proj: TrellisLinear for down projection.
    """

    def __init__(
        self,
        gate_proj: TrellisLinear,
        up_proj: TrellisLinear,
        down_proj: TrellisLinear,
    ):
        """Initialize TrellisDenseMLP.

        Args:
            gate_proj: TrellisLinear for gate projection (uses SiLU activation).
            up_proj: TrellisLinear for up projection.
            down_proj: TrellisLinear for down projection.
        """
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SwiGLU activation.

        Args:
            x: Input tensor [..., hidden_size].

        Returns:
            Output tensor [..., hidden_size].
        """
        # SwiGLU activation: silu(gate) * up
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

    @classmethod
    def from_loader(
        cls,
        loader: TrellisModelLoader,
        layer_idx: int,
        device: str = "mps",
    ) -> TrellisDenseMLP:
        """Create TrellisDenseMLP from a TrellisModelLoader.

        Loads the gate_proj, up_proj, and down_proj weights for the specified
        layer and creates a TrellisDenseMLP module.

        Args:
            loader: TrellisModelLoader instance for the model.
            layer_idx: Layer index to load (0-indexed).
            device: Device to place the modules on (default: "mps").

        Returns:
            TrellisDenseMLP module initialized with the layer weights.

        Raises:
            KeyError: If any of the required MLP weights are not found.
        """
        layer_weights = loader.load_layer(layer_idx)
        prefix = f"model.layers.{layer_idx}.mlp"

        return cls(
            gate_proj=TrellisLinear.from_trellis_weight(
                layer_weights[f"{prefix}.gate_proj.weight"],
                device=device,
            ),
            up_proj=TrellisLinear.from_trellis_weight(
                layer_weights[f"{prefix}.up_proj.weight"],
                device=device,
            ),
            down_proj=TrellisLinear.from_trellis_weight(
                layer_weights[f"{prefix}.down_proj.weight"],
                device=device,
            ),
        )

    def extra_repr(self) -> str:
        """String representation for printing."""
        return (
            f"gate_proj={self.gate_proj.out_features}x{self.gate_proj.in_features}, "
            f"up_proj={self.up_proj.out_features}x{self.up_proj.in_features}, "
            f"down_proj={self.down_proj.out_features}x{self.down_proj.in_features}"
        )
