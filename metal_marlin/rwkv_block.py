"""RWKV Block combining Time Mixing and Channel Mixing.

An RWKV block is the fundamental building unit of RWKV models, analogous
to a transformer block but with linear-complexity attention.

Structure:
    x -> LayerNorm -> TimeMixing -> + -> LayerNorm -> ChannelMixing -> +
    |_______________________________| |__________________________________|

Usage:
    from metal_marlin.rwkv_block import RWKVBlock

    block = RWKVBlock(
        hidden_size=2048,
        num_heads=32,
        intermediate_size=5632,
    )
    output = block(x, state, layer_idx=0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ._compat import HAS_MLX, mx, nn
from .dtypes import DTypeConfig, get_default_config
from .rwkv_channel_mixing import RWKVChannelMixing
from .rwkv_time_mixing import RWKVTimeMixing

if TYPE_CHECKING:
    from .rwkv_state import RWKVState


class RMSNorm:
    """Root Mean Square Layer Normalization for RWKV.

    RWKV typically uses LayerNorm, but RMSNorm is more efficient and
    achieves comparable performance.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        if HAS_MLX and nn is not None:
            nn.Module.__init__(self)

        if HAS_MLX and mx is not None:
            self.weight = mx.ones((hidden_size,))
        else:
            self.weight = np.ones((hidden_size,), dtype=np.float32)

        self.eps = eps
        self.hidden_size = hidden_size

    def __call__(self, x):
        if HAS_MLX and mx is not None:
            variance = mx.mean(x ** 2, axis=-1, keepdims=True)
            x = x * mx.rsqrt(variance + self.eps)
            return self.weight * x
        else:
            variance = np.mean(x ** 2, axis=-1, keepdims=True)
            x = x / np.sqrt(variance + self.eps)
            return self.weight * x


class LayerNorm:
    """Standard Layer Normalization.

    RWKV models typically use LayerNorm rather than RMSNorm.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        if HAS_MLX and nn is not None:
            nn.Module.__init__(self)

        if HAS_MLX and mx is not None:
            self.weight = mx.ones((hidden_size,))
            self.bias = mx.zeros((hidden_size,))
        else:
            self.weight = np.ones((hidden_size,), dtype=np.float32)
            self.bias = np.zeros((hidden_size,), dtype=np.float32)

        self.eps = eps
        self.hidden_size = hidden_size

    def __call__(self, x):
        if HAS_MLX and mx is not None:
            mean = mx.mean(x, axis=-1, keepdims=True)
            variance = mx.var(x, axis=-1, keepdims=True)
            x = (x - mean) * mx.rsqrt(variance + self.eps)
            return self.weight * x + self.bias
        else:
            mean = np.mean(x, axis=-1, keepdims=True)
            variance = np.var(x, axis=-1, keepdims=True)
            x = (x - mean) / np.sqrt(variance + self.eps)
            return self.weight * x + self.bias


class RWKVBlock:
    """Single RWKV block with Time Mixing and Channel Mixing.

    Unlike transformer blocks which use self-attention + FFN, RWKV uses:
    - Time Mixing: Linear-complexity attention-like mechanism with WKV
    - Channel Mixing: Gated FFN with squared ReLU

    Both components use token shifting for temporal context.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        intermediate_size: FFN intermediate dimension.
        head_dim: Dimension per head. Defaults to hidden_size // num_heads.
        quant_type: Quantization format for linear layers.
        group_size: Quantization group size.
        layer_id: Layer index (affects time decay initialization).
        version: RWKV version ("v5" or "v6").
        layer_norm_eps: Epsilon for layer normalization.
        dtype_config: Dtype configuration for activations.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        head_dim: int | None = None,
        quant_type: Literal["fp4", "int4"] = "fp4",
        group_size: int = 128,
        layer_id: int = 0,
        version: Literal["v5", "v6"] = "v6",
        layer_norm_eps: float = 1e-5,
        dtype_config: DTypeConfig | None = None,
    ):
        if HAS_MLX and nn is not None:
            nn.Module.__init__(self)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.layer_id = layer_id
        self.dtype_config = dtype_config if dtype_config is not None else get_default_config()

        # Layer norms (RWKV uses LayerNorm, not RMSNorm)
        self.ln1 = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln2 = LayerNorm(hidden_size, eps=layer_norm_eps)

        # Time mixing (attention-like)
        self.time_mixing = RWKVTimeMixing(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=self.head_dim,
            quant_type=quant_type,
            group_size=group_size,
            layer_id=layer_id,
            version=version,
            dtype_config=dtype_config,
        )

        # Channel mixing (FFN-like)
        self.channel_mixing = RWKVChannelMixing(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_type=quant_type,
            group_size=group_size,
            dtype_config=dtype_config,
        )

    def __call__(
        self,
        hidden_states,  # [batch, seq_len, hidden_size]
        state: RWKVState | None = None,
        layer_idx: int = 0,
    ):
        """Forward pass through the RWKV block.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            state: RWKV state container (optional)
            layer_idx: Layer index for state addressing

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        # Time Mixing with residual
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.time_mixing(hidden_states, state, layer_idx)
        hidden_states = residual + hidden_states

        # Channel Mixing with residual
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.channel_mixing(hidden_states, state, layer_idx)
        hidden_states = residual + hidden_states

        return hidden_states


# Make classes inherit from nn.Module when MLX is available
if HAS_MLX and nn is not None:
    _OriginalRMSNorm = RMSNorm
    _OriginalLayerNorm = LayerNorm
    _OriginalRWKVBlock = RWKVBlock

    class RMSNorm(nn.Module):  # type: ignore[no-redef]
        """Root Mean Square Layer Normalization for RWKV."""

        def __init__(self, hidden_size: int, eps: float = 1e-6):
            super().__init__()
            _OriginalRMSNorm.__init__(self, hidden_size, eps)

        __call__ = _OriginalRMSNorm.__call__

    class LayerNorm(nn.Module):  # type: ignore[no-redef]
        """Standard Layer Normalization."""

        def __init__(self, hidden_size: int, eps: float = 1e-5):
            super().__init__()
            _OriginalLayerNorm.__init__(self, hidden_size, eps)

        __call__ = _OriginalLayerNorm.__call__

    class RWKVBlock(nn.Module):  # type: ignore[no-redef]
        """Single RWKV block with Time Mixing and Channel Mixing."""

        __doc__ = _OriginalRWKVBlock.__doc__

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            _OriginalRWKVBlock.__init__(self, *args, **kwargs)

        __call__ = _OriginalRWKVBlock.__call__
