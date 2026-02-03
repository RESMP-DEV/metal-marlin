"""Trellis layer modules for EXL3 trellis-quantized models.

Provides layer-level modules like MLP that combine TrellisLinear layers
for use with trellis-quantized models like GLM-4.7-Flash.

Usage:
    from metal_marlin.trellis.loader import TrellisModelLoader
    from metal_marlin.trellis.layer import TrellisDenseMLP, TrellisFusedDenseMLP

    loader = TrellisModelLoader("models/GLM-4.7-Flash-EXL3-3bpw")
    mlp = TrellisDenseMLP.from_loader(loader, layer_idx=0, device="mps")
    output = mlp(input_tensor)

    # Or use fused gate+up projection for better performance:
    fused_mlp = TrellisFusedDenseMLP.from_loader(loader, layer_idx=0, device="mps")
    output = fused_mlp(input_tensor)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import TrellisLinear

if TYPE_CHECKING:
    from .loader import TrellisModelLoader


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


class TrellisFusedDenseMLP(nn.Module):
    """Dense MLP with fused gate+up projection for better performance.

    Implements the SwiGLU activation pattern with gate and up projections
    fused into a single matrix multiplication:
    - gate_up_proj: Single projection producing concatenated [gate, up] outputs
    - down_proj: Projects the element-wise product back to hidden size

    This provides performance benefits by:
    - Reducing GEMM count from 2 to 1
    - Improving memory access patterns
    - Better cache utilization

    Attributes:
        gate_up: TrellisLinear for fused gate+up projection.
        down: TrellisLinear for down projection.
        intermediate_size: Size of the intermediate dimension.
    """

    def __init__(
        self,
        gate_up: TrellisLinear,
        down: TrellisLinear,
        intermediate_size: int,
    ):
        """Initialize TrellisFusedDenseMLP.

        Args:
            gate_up: TrellisLinear for fused gate+up projection (2x output).
            down: TrellisLinear for down projection.
            intermediate_size: Intermediate dimension size for splitting gate_up.
        """
        super().__init__()
        self.gate_up = gate_up
        self.down = down
        self.intermediate_size = intermediate_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SwiGLU activation using fused gate+up.

        Args:
            x: Input tensor [..., hidden_size].

        Returns:
            Output tensor [..., hidden_size].
        """
        # Fused gate+up: output is [gate, up] concatenated
        gate_up_out = self.gate_up(x)

        # Split fused output into gate and up
        if gate_up_out.shape[-1] == 2 * self.intermediate_size:
            gate, up = gate_up_out.chunk(2, dim=-1)
        else:
            # Fallback: not truly fused, use same output for both
            gate = gate_up_out
            up = gate_up_out

        # SwiGLU activation: silu(gate) * up
        return self.down(F.silu(gate) * up)

    @classmethod
    def from_loader(
        cls,
        loader: TrellisModelLoader,
        layer_idx: int,
        device: str = "mps",
    ) -> TrellisFusedDenseMLP:
        """Create TrellisFusedDenseMLP from a TrellisModelLoader.

        Loads the gate_up_proj and down_proj weights for the specified
        layer and creates a TrellisFusedDenseMLP module.

        Args:
            loader: TrellisModelLoader instance for the model.
            layer_idx: Layer index to load (0-indexed).
            device: Device to place the modules on (default: "mps").

        Returns:
            TrellisFusedDenseMLP module initialized with the layer weights.

        Raises:
            KeyError: If any of the required MLP weights are not found.
        """
        layer_weights = loader.load_layer(layer_idx)
        prefix = f"model.layers.{layer_idx}.mlp"

        gate_up = TrellisLinear.from_trellis_weight(
            layer_weights[f"{prefix}.gate_up_proj.weight"],
            device=device,
        )
        down = TrellisLinear.from_trellis_weight(
            layer_weights[f"{prefix}.down_proj.weight"],
            device=device,
        )

        # Infer intermediate_size from gate_up output (should be 2x)
        intermediate_size = gate_up.out_features // 2

        return cls(
            gate_up=gate_up,
            down=down,
            intermediate_size=intermediate_size,
        )

    @classmethod
    def from_separate_weights(
        cls,
        gate_proj: TrellisLinear,
        up_proj: TrellisLinear,
        down_proj: TrellisLinear,
    ) -> TrellisFusedDenseMLP:
        """Create TrellisFusedDenseMLP by fusing separate gate and up weights.

        Args:
            gate_proj: TrellisLinear for gate projection.
            up_proj: TrellisLinear for up projection.
            down_proj: TrellisLinear for down projection.

        Returns:
            TrellisFusedDenseMLP with gate and up weights fused.
        """
        # Get the packed weights from both projections
        gate_packed = gate_proj.packed_indices
        up_packed = up_proj.packed_indices
        gate_scales = gate_proj.scales
        up_scales = up_proj.scales
        gate_su = gate_proj.su
        up_su = up_proj.su
        gate_sv = gate_proj.sv
        up_sv = up_proj.sv

        # Concatenate along output dimension (first dimension of packed_weights)
        # packed_indices shape: [tiles_out, tiles_in, packed_size]
        fused_packed = torch.cat([gate_packed, up_packed], dim=0)
        fused_scales = torch.cat([gate_scales, up_scales], dim=0)
        fused_su = torch.cat([gate_su, up_su], dim=0)
        fused_sv = torch.cat([gate_sv, up_sv], dim=0)

        # Create a TrellisLinear with the fused weights
        gate_up = TrellisLinear(
            in_features=gate_proj.in_features,
            out_features=gate_proj.out_features + up_proj.out_features,
            bits=gate_proj.bits,
            device=str(gate_proj.scales.device),
            packed_indices=fused_packed,
            scales=fused_scales,
            su=fused_su,
            sv=fused_sv,
            grid=gate_proj.grid,
        )

        intermediate_size = gate_proj.out_features

        return cls(
            gate_up=gate_up,
            down=down_proj,
            intermediate_size=intermediate_size,
        )

    def extra_repr(self) -> str:
        """String representation for printing."""
        return (
            f"gate_up={self.gate_up.out_features}x{self.gate_up.in_features}, "
            f"down={self.down.out_features}x{self.down.in_features}"
        )
