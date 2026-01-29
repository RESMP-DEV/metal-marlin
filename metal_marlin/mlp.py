"""MLP block with Marlin-quantized linear layers.

Supports standard and gated (SwiGLU/GeGLU) architectures using FP4/INT4
quantized GEMM kernels on Apple Silicon.

Usage:
    from metal_marlin.mlp import MarlinMLP

    mlp = MarlinMLP(hidden_size=4096, intermediate_size=11008)
    out = mlp(x)  # x: [batch, seq_len, 4096]
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ._compat import HAS_TORCH, torch
from .layers import MarlinLinear

# Try to import Metal activation functions
try:
    from .activation_metal import (
        gelu_metal,
        relu_metal,
        silu_metal,
        swiglu_fused_metal,
    )

    _USE_METAL_ACTIVATION = True
except ImportError:
    _USE_METAL_ACTIVATION = False
    silu_metal = None  # type: ignore[misc]
    gelu_metal = None  # type: ignore[misc]
    relu_metal = None  # type: ignore[misc]
    swiglu_fused_metal = None  # type: ignore[misc]


def _silu_numpy(x: np.ndarray) -> np.ndarray:
    """SiLU (Swish) activation: x * sigmoid(x)."""
    return x * (1.0 / (1.0 + np.exp(-x)))


def _gelu_numpy(x: np.ndarray) -> np.ndarray:
    """GELU activation using the approximate formula."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _relu_numpy(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(x, 0)


class _MarlinMLPBase:
    """Base MLP implementation without nn.Module inheritance.

    This class contains the core logic and is used as a mixin.
    """

    gated: bool
    activation: str
    gate_proj: MarlinLinear | None
    up_proj: MarlinLinear
    down_proj: MarlinLinear

    def _init_layers(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: Literal["silu", "gelu", "relu"],
        gated: bool,
        quant_type: str,
        group_size: int,
    ) -> None:
        self.gated = gated
        self.activation = activation

        if gated:
            self.gate_proj = MarlinLinear(
                hidden_size,
                intermediate_size,
                bias=False,
                quant_type=quant_type,
                group_size=group_size,
            )
            self.up_proj = MarlinLinear(
                hidden_size,
                intermediate_size,
                bias=False,
                quant_type=quant_type,
                group_size=group_size,
            )
        else:
            self.gate_proj = None
            self.up_proj = MarlinLinear(
                hidden_size,
                intermediate_size,
                bias=False,
                quant_type=quant_type,
                group_size=group_size,
            )

        self.down_proj = MarlinLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_type=quant_type,
            group_size=group_size,
        )

    def _forward(self, x: Any) -> Any:
        if self.gated and self.gate_proj is not None:
            # Use fused SwiGLU path for MPS tensors when available
            if (
                _USE_METAL_ACTIVATION
                and HAS_TORCH
                and torch is not None
                and isinstance(x, torch.Tensor)
                and x.is_mps
                and swiglu_fused_metal is not None
            ):
                gate = self.gate_proj(x)
                up = self.up_proj(x)
                x = swiglu_fused_metal(gate, up)
            else:
                gate = self._activate(self.gate_proj(x))
                up = self.up_proj(x)
                x = gate * up
        else:
            x = self._activate(self.up_proj(x))

        return self.down_proj(x)

    def _activate(self, x: Any) -> Any:
        if HAS_TORCH and torch is not None:
            return self._activate_torch(x)
        return self._activate_numpy(x)

    def _activate_torch(self, x: Any) -> Any:
        """PyTorch activation functions with Metal dispatch for MPS tensors."""
        assert torch is not None

        # Use Metal activation functions for MPS tensors when available
        if _USE_METAL_ACTIVATION and x.is_mps:
            if self.activation == "silu":
                return silu_metal(x)
            elif self.activation == "gelu":
                return gelu_metal(x)
            elif self.activation == "relu":
                return relu_metal(x)

        # Fallback to standard PyTorch implementations
        if self.activation == "silu":
            return torch.nn.functional.silu(x)
        elif self.activation == "gelu":
            return torch.nn.functional.gelu(x)
        elif self.activation == "relu":
            return torch.nn.functional.relu(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _activate_numpy(self, x: Any) -> Any:
        """Numpy fallback activation functions."""
        x_np = np.asarray(x)
        if self.activation == "silu":
            return _silu_numpy(x_np)
        elif self.activation == "gelu":
            return _gelu_numpy(x_np)
        elif self.activation == "relu":
            return _relu_numpy(x_np)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


if HAS_TORCH and torch is not None:

    class MarlinMLP(torch.nn.Module, _MarlinMLPBase):
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
            torch.nn.Module.__init__(self)
            self._init_layers(
                hidden_size, intermediate_size, activation, gated, quant_type, group_size
            )

        def forward(self, x: Any) -> Any:
            return self._forward(x)

else:

    class MarlinMLP(_MarlinMLPBase):  # type: ignore[no-redef]
        """MLP block with Marlin-quantized linear layers (numpy fallback).

        Supports:
        - Standard MLP: up -> activation -> down
        - Gated MLP (SwiGLU/GeGLU): (activation(gate) * up) -> down

        The gated variant follows the LLaMA/Mistral convention where gate_proj
        and up_proj are separate linear layers, and the gate output is activated
        before element-wise multiplication with the up output.

        Note: This fallback version uses numpy and does not support GPU acceleration.
        Install PyTorch for GPU support.

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
            self._init_layers(
                hidden_size, intermediate_size, activation, gated, quant_type, group_size
            )

        def __call__(self, x: Any) -> Any:
            return self._forward(x)


__all__ = ["MarlinMLP"]
