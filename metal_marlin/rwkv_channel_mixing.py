"""RWKV Channel Mixing layer with Marlin-quantized projections.

Channel Mixing is RWKV's feed-forward network replacement. It uses a similar
token-shift mechanism as Time Mixing but applies a simpler MLP-like transform.

The key differences from standard FFN:
1. Token shifting: mixes current and previous token before projection
2. Squared ReLU activation: provides stronger feature selection
3. Receptance gating: learned sigmoid gate controls output

Usage:
    from metal_marlin.rwkv_channel_mixing import RWKVChannelMixing

    channel_mix = RWKVChannelMixing(
        hidden_size=2048,
        intermediate_size=5632,  # Often hidden_size * 2.75 or similar
    )
    output = channel_mix(x, state, layer_idx=0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ._compat import HAS_MLX, mx, nn
from .dtypes import DTypeConfig, get_default_config
from .layers import MarlinLinear

if TYPE_CHECKING:
    from .rwkv_state import RWKVState


class RWKVChannelMixing:
    """Channel Mixing (FFN-like) layer for RWKV.

    Implements a gated MLP with token shifting. The architecture:
        1. Token shift mixing: x = lerp(x_prev, x, mix_ratio)
        2. Project to key (intermediate) dimension
        3. Apply squared ReLU: relu(x)^2
        4. Project back to hidden dimension
        5. Gate with receptance (sigmoid)

    The squared ReLU is important for RWKV's performance - it provides
    stronger feature selection than standard activations.

    Args:
        hidden_size: Model hidden dimension.
        intermediate_size: FFN intermediate dimension.
        quant_type: Quantization format for linear layers.
        group_size: Quantization group size.
        dtype_config: Dtype configuration for activations.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_type: Literal["fp4", "int4"] = "fp4",
        group_size: int = 128,
        dtype_config: DTypeConfig | None = None,
    ):
        if HAS_MLX and nn is not None:
            nn.Module.__init__(self)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dtype_config = dtype_config if dtype_config is not None else get_default_config()

        # Token shift mixing ratios (learnable)
        if HAS_MLX and mx is not None:
            dtype = self.dtype_config.mlx_activations
            self.channel_mix_k = mx.ones((hidden_size,), dtype=dtype) * 0.5
            self.channel_mix_r = mx.ones((hidden_size,), dtype=dtype) * 0.5
        else:
            self.channel_mix_k = np.ones((hidden_size,), dtype=np.float16) * 0.5
            self.channel_mix_r = np.ones((hidden_size,), dtype=np.float16) * 0.5

        # Quantized projections
        # Key projection (up to intermediate size)
        self.key = MarlinLinear(
            hidden_size, intermediate_size,
            bias=False, quant_type=quant_type, group_size=group_size,
            dtype_config=dtype_config,
        )

        # Value projection (back to hidden size)
        self.value = MarlinLinear(
            intermediate_size, hidden_size,
            bias=False, quant_type=quant_type, group_size=group_size,
            dtype_config=dtype_config,
        )

        # Receptance projection (for gating)
        self.receptance = MarlinLinear(
            hidden_size, hidden_size,
            bias=False, quant_type=quant_type, group_size=group_size,
            dtype_config=dtype_config,
        )

    def _token_shift(self, x, x_prev, mix_ratio):
        """Mix current token with previous token.

        Args:
            x: Current hidden state [batch, seq_len, hidden]
            x_prev: Previous token's hidden state [batch, hidden]
            mix_ratio: Learnable mixing ratio [hidden]

        Returns:
            Mixed hidden state [batch, seq_len, hidden]
        """
        if HAS_MLX and mx is not None:
            batch_size, seq_len, _ = x.shape

            if seq_len == 1:
                return x * mix_ratio + x_prev[:, None, :] * (1 - mix_ratio)
            else:
                x_shifted = mx.concatenate([
                    x_prev[:, None, :],
                    x[:, :-1, :]
                ], axis=1)
                return x * mix_ratio + x_shifted * (1 - mix_ratio)
        else:
            batch_size, seq_len, _ = x.shape

            if seq_len == 1:
                return x * mix_ratio + x_prev[:, None, :] * (1 - mix_ratio)
            else:
                x_shifted = np.concatenate([
                    x_prev[:, None, :],
                    x[:, :-1, :]
                ], axis=1)
                return x * mix_ratio + x_shifted * (1 - mix_ratio)

    def __call__(
        self,
        hidden_states,  # [batch, seq_len, hidden_size]
        state: RWKVState | None = None,
        layer_idx: int = 0,
    ):
        """Forward pass through channel mixing layer.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            state: RWKV state container (optional)
            layer_idx: Layer index for state addressing

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        if HAS_MLX and mx is not None:
            batch_size, seq_len, _ = hidden_states.shape

            # Get previous shift state
            if state is not None:
                shift_state = state.get_channel_mixing_state(layer_idx)
            else:
                shift_state = mx.zeros(
                    (batch_size, self.hidden_size),
                    dtype=self.dtype_config.mlx_activations
                )

            # Token shift mixing
            xk = self._token_shift(hidden_states, shift_state, self.channel_mix_k)
            xr = self._token_shift(hidden_states, shift_state, self.channel_mix_r)

            # Key projection -> squared ReLU -> value projection
            k = self.key(xk)
            k = mx.square(mx.maximum(k, 0))  # Squared ReLU
            kv = self.value(k)

            # Receptance gating
            r = mx.sigmoid(self.receptance(xr))
            output = r * kv

            # Update state
            new_shift = hidden_states[:, -1, :]

            if state is not None:
                state.update_channel_mixing_state(layer_idx, new_shift)

            return output
        else:
            # Numpy fallback
            batch_size, seq_len, _ = hidden_states.shape

            if state is not None:
                shift_state = state.get_channel_mixing_state(layer_idx)
            else:
                shift_state = np.zeros(
                    (batch_size, self.hidden_size),
                    dtype=np.float16
                )

            xk = self._token_shift(hidden_states, shift_state, self.channel_mix_k)
            xr = self._token_shift(hidden_states, shift_state, self.channel_mix_r)

            k = self.key(xk)
            k = np.square(np.maximum(k, 0))  # Squared ReLU
            kv = self.value(k)

            sigmoid_r = 1 / (1 + np.exp(-self.receptance(xr)))
            output = sigmoid_r * kv

            new_shift = hidden_states[:, -1, :]

            if state is not None:
                state.update_channel_mixing_state(layer_idx, new_shift)

            return output


# Make it inherit from nn.Module when MLX is available
if HAS_MLX and nn is not None:
    _OriginalChannelMixing = RWKVChannelMixing

    class RWKVChannelMixing(nn.Module):  # type: ignore[no-redef]
        """Channel Mixing layer for RWKV with Marlin-quantized projections."""

        __doc__ = _OriginalChannelMixing.__doc__

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            _OriginalChannelMixing.__init__(self, *args, **kwargs)

        __call__ = _OriginalChannelMixing.__call__
        _token_shift = _OriginalChannelMixing._token_shift
