"""MMFP4 expert layer with optional fused decode path."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..kernels import mmfp4_fused_moe_mlp
from .mmfp4_linear import MMFP4Linear


class MMFP4Expert(nn.Module):
    """Single SwiGLU expert using MMFP4-quantized linear layers."""

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        group_size: int = 128,
        use_fused: bool = False,  # Disabled: fused kernel needs tiling fix for intermediate_size > 1536
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.intermediate_size = moe_intermediate_size
        self.group_size = group_size
        self.use_fused = use_fused

        # Create placeholder MMFP4Linear layers - weights loaded later.
        self.gate_proj = _make_placeholder_mmfp4_linear(
            hidden_size, moe_intermediate_size, group_size
        )
        self.up_proj = _make_placeholder_mmfp4_linear(
            hidden_size, moe_intermediate_size, group_size
        )
        self.down_proj = _make_placeholder_mmfp4_linear(
            moe_intermediate_size, hidden_size, group_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused path for single token decode
        # Ensure 2D input [1, hidden_size] for fused kernel
        if self.use_fused and x.shape[0] == 1 and x.ndim == 2:
            return self._fused_forward(x)

        # Standard path for prefill
        return self._standard_forward(x)

    @staticmethod
    @torch.compile(mode="reduce-overhead")
    def _fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Fused SiLU(gate) * up - single kernel to avoid dispatch overhead."""
        # Fused operation: gate * sigmoid(gate) * up
        # This avoids separate SiLU and multiplication dispatches
        sigmoid_gate = torch.sigmoid(gate)
        return gate * sigmoid_gate * up

    @torch.compile(mode="reduce-overhead")
    def _standard_forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Use fused SiLU * up operation to reduce dispatch overhead
        activated = self._fused_silu_mul(gate, up)
        return self.down_proj(activated)

    def _fused_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fused MoE MLP for decode (gate+up+SiLU+down in one kernel)."""
        return mmfp4_fused_moe_mlp(
            input=x,
            gate_proj_packed=self.gate_proj.packed_weights,
            up_proj_packed=self.up_proj.packed_weights,
            down_proj_packed=self.down_proj.packed_weights,
            gate_scales=self.gate_proj.scales,
            up_scales=self.up_proj.scales,
            down_scales=self.down_proj.scales,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            group_size=self.group_size,
        )


def _make_placeholder_mmfp4_linear(
    in_features: int,
    out_features: int,
    group_size: int,
) -> MMFP4Linear:
    """Create MMFP4Linear with placeholder (zeros) weights."""
    in_features_aligned = ((in_features + 7) // 8) * 8
    n_groups = (in_features + group_size - 1) // group_size
    packed_weights = torch.zeros(
        (out_features, in_features_aligned // 8),
        dtype=torch.uint32,
    )
    scales = torch.ones((n_groups, out_features), dtype=torch.float16)
    return MMFP4Linear(
        packed_weights=packed_weights,
        scales=scales,
        bias=None,
        group_size=group_size,
    )