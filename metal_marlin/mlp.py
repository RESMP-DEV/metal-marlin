"""MLP block with Marlin-quantized linear layers.

Supports standard and gated (SwiGLU/GeGLU) architectures using FP4/INT4
quantized GEMM kernels on Apple Silicon.

Usage:
    from metal_marlin.python.mlp import MarlinMLP

    mlp = MarlinMLP(hidden_size=4096, intermediate_size=11008)
    out = mlp(x)  # x: [batch, seq_len, 4096]
"""

from __future__ import annotations

from typing import Literal

import mlx.core as mx
import mlx.nn as nn

from .layers import MarlinLinear


class MarlinMLP(nn.Module):
    """MLP block with Marlin-quantized linear layers.

    Supports:
    - Standard MLP: up -> activation -> down
    - Gated MLP (SwiGLU/GeGLU): (activation(gate) * up) -> down

    The gated variant follows the LLaMA/Mistral convention where gate_proj
    and up_proj are separate linear layers, and the gate output is activated
    before element-wise multiplication with the up output.

    Args:
        hidden_size: Model hidden dimension (input/output size).
        intermediate_size: MLP intermediate dimension.
        activation: Activation function. "silu" for SwiGLU, "gelu" for GeGLU.
        gated: If True, use gated MLP with separate gate and up projections.
        quant_type: Quantization format passed to MarlinLinear.
        group_size: Quantization group size passed to MarlinLinear.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: Literal["silu", "gelu", "relu"] = "silu",
        gated: bool = True,
        quant_type: str = "fp4",
        group_size: int = 128,
    ):
        super().__init__()
        self.gated = gated
        self.activation = activation

        if gated:
            self.gate_proj = MarlinLinear(
                hidden_size, intermediate_size,
                bias=False, quant_type=quant_type, group_size=group_size,
            )
            self.up_proj = MarlinLinear(
                hidden_size, intermediate_size,
                bias=False, quant_type=quant_type, group_size=group_size,
            )
        else:
            self.up_proj = MarlinLinear(
                hidden_size, intermediate_size,
                bias=False, quant_type=quant_type, group_size=group_size,
            )

        self.down_proj = MarlinLinear(
            intermediate_size, hidden_size,
            bias=False, quant_type=quant_type, group_size=group_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.gated:
            gate = self._activate(self.gate_proj(x))
            up = self.up_proj(x)
            x = gate * up
        else:
            x = self._activate(self.up_proj(x))

        return self.down_proj(x)

    def _activate(self, x: mx.array) -> mx.array:
        if self.activation == "silu":
            return nn.silu(x)
        elif self.activation == "gelu":
            return nn.gelu(x)
        elif self.activation == "relu":
            return nn.relu(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
