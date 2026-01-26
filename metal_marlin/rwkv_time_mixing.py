"""RWKV Time Mixing layer with Marlin-quantized projections.

Time Mixing is RWKV's attention-like mechanism that replaces transformer
self-attention with a linear-complexity recurrent formulation. It uses the
WKV (weighted key-value) operator which approximates attention via
exponential moving averages.

The key insight is that softmax attention can be decomposed:
    softmax(QK^T)V = Σ exp(q_i · k_j) v_j / Σ exp(q_i · k_j)

RWKV reformulates this with learnable time decay:
    wkv_t = Σ exp(-(t-j) * w + k_j) v_j / Σ exp(-(t-j) * w + k_j)

This allows O(1) state update per token instead of O(n) for attention.

Usage:
    from metal_marlin.rwkv_time_mixing import RWKVTimeMixing

    time_mix = RWKVTimeMixing(
        hidden_size=2048,
        num_heads=32,
        head_dim=64,
    )
    output, new_state = time_mix(x, state, layer_idx=0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ._compat import HAS_MLX, mx, nn
from .dtypes import DTypeConfig, get_default_config
from .layers import MarlinLinear

if TYPE_CHECKING:
    from .rwkv_state import RWKVState


class RWKVTimeMixing:
    """Time Mixing (attention-like) layer for RWKV.

    Implements the WKV mechanism with learnable time decay. Unlike transformer
    attention, this has O(1) memory and compute per token during generation.

    RWKV v5/v6 use multi-headed time mixing where each head has independent
    time decay parameters and state.

    Architecture:
        1. Token shift mixing: x = lerp(x_prev, x, mix_ratio)
        2. Project to R, K, V (receptance, key, value)
        3. Apply WKV operator with time decay
        4. Gate output with receptance (sigmoid)
        5. Project to output dimension

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head. Defaults to hidden_size // num_heads.
        quant_type: Quantization format for linear layers.
        group_size: Quantization group size.
        layer_id: Layer index (used for time decay initialization).
        version: RWKV version ("v5" or "v6").
        dtype_config: Dtype configuration for activations.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        quant_type: Literal["fp4", "int4"] = "fp4",
        group_size: int = 128,
        layer_id: int = 0,
        version: Literal["v5", "v6"] = "v6",
        dtype_config: DTypeConfig | None = None,
    ):
        if HAS_MLX and nn is not None:
            nn.Module.__init__(self)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.layer_id = layer_id
        self.version = version
        self.dtype_config = dtype_config if dtype_config is not None else get_default_config()

        # Validate dimensions
        if self.head_dim * num_heads != hidden_size:
            raise ValueError(
                f"head_dim ({self.head_dim}) * num_heads ({num_heads}) "
                f"must equal hidden_size ({hidden_size})"
            )

        # Token shift mixing ratios (learnable, but not quantized)
        # These control how much of the previous token to mix in
        if HAS_MLX and mx is not None:
            dtype = self.dtype_config.mlx_activations
            self.time_mix_r = mx.ones((hidden_size,), dtype=dtype) * 0.5
            self.time_mix_k = mx.ones((hidden_size,), dtype=dtype) * 0.5
            self.time_mix_v = mx.ones((hidden_size,), dtype=dtype) * 0.5
            self.time_mix_g = mx.ones((hidden_size,), dtype=dtype) * 0.5  # v6 gate

            # Time decay parameters (CRITICAL for quality - FP32 recommended)
            # Initialized with layer-dependent values for stability
            decay_speed = 1.0 - (layer_id / 24.0)  # Slower decay in later layers
            self.time_decay = mx.ones((num_heads, self.head_dim), dtype=mx.float32) * decay_speed

            # Time first parameter (bonus for current token)
            self.time_first = mx.zeros((num_heads, self.head_dim), dtype=mx.float32)
        else:
            # Numpy fallback
            self.time_mix_r = np.ones((hidden_size,), dtype=np.float16) * 0.5
            self.time_mix_k = np.ones((hidden_size,), dtype=np.float16) * 0.5
            self.time_mix_v = np.ones((hidden_size,), dtype=np.float16) * 0.5
            self.time_mix_g = np.ones((hidden_size,), dtype=np.float16) * 0.5

            decay_speed = 1.0 - (layer_id / 24.0)
            self.time_decay = np.ones((num_heads, self.head_dim), dtype=np.float32) * decay_speed
            self.time_first = np.zeros((num_heads, self.head_dim), dtype=np.float32)

        # Quantized projections
        # R (receptance), K (key), V (value) projections
        self.receptance = MarlinLinear(
            hidden_size, hidden_size,
            bias=False, quant_type=quant_type, group_size=group_size,
            dtype_config=dtype_config,
        )
        self.key = MarlinLinear(
            hidden_size, hidden_size,
            bias=False, quant_type=quant_type, group_size=group_size,
            dtype_config=dtype_config,
        )
        self.value = MarlinLinear(
            hidden_size, hidden_size,
            bias=False, quant_type=quant_type, group_size=group_size,
            dtype_config=dtype_config,
        )

        # Output projection
        self.output = MarlinLinear(
            hidden_size, hidden_size,
            bias=False, quant_type=quant_type, group_size=group_size,
            dtype_config=dtype_config,
        )

        # Gate projection (v6 only)
        if version == "v6":
            self.gate = MarlinLinear(
                hidden_size, hidden_size,
                bias=False, quant_type=quant_type, group_size=group_size,
                dtype_config=dtype_config,
            )
        else:
            self.gate = None

    def _token_shift(self, x, x_prev, mix_ratio):
        """Mix current token with previous token.

        This is a key RWKV operation that provides temporal context
        without explicit attention over all past tokens.

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
                # Single token decode: simple mix with previous
                return x * mix_ratio + x_prev[:, None, :] * (1 - mix_ratio)
            else:
                # Prefill: shift tokens and mix
                # x_shifted[i] = x[i-1] for i > 0, x_prev for i = 0
                x_shifted = mx.concatenate([
                    x_prev[:, None, :],
                    x[:, :-1, :]
                ], axis=1)
                return x * mix_ratio + x_shifted * (1 - mix_ratio)
        else:
            # Numpy fallback
            batch_size, seq_len, _ = x.shape

            if seq_len == 1:
                return x * mix_ratio + x_prev[:, None, :] * (1 - mix_ratio)
            else:
                x_shifted = np.concatenate([
                    x_prev[:, None, :],
                    x[:, :-1, :]
                ], axis=1)
                return x * mix_ratio + x_shifted * (1 - mix_ratio)

    def _wkv_forward(self, r, k, v, w, u, state, state_denom):
        """Compute WKV (weighted key-value) attention.

        This is the core RWKV operation. For each position t:
            num = state + exp(u + k) * v
            den = state_denom + exp(u + k)
            out = r * (num / den)
            state' = exp(-w) * state + exp(k) * v
            state_denom' = exp(-w) * state_denom + exp(k)

        The exp(-w) term implements time decay: older information fades.

        Args:
            r: Receptance [batch, heads, seq, head_dim]
            k: Key [batch, heads, seq, head_dim]
            v: Value [batch, heads, seq, head_dim]
            w: Time decay [heads, head_dim]
            u: Time first bonus [heads, head_dim]
            state: WKV state [batch, heads, head_dim, head_dim]
            state_denom: WKV denominator [batch, heads, head_dim]

        Returns:
            output: [batch, heads, seq, head_dim]
            new_state: [batch, heads, head_dim, head_dim]
            new_state_denom: [batch, heads, head_dim]
        """
        if HAS_MLX and mx is not None:
            batch_size, num_heads, seq_len, head_dim = k.shape

            # Process sequentially for proper state evolution
            outputs = []
            current_state = state
            current_denom = state_denom

            for t in range(seq_len):
                # Extract current timestep
                kt = k[:, :, t, :]  # [batch, heads, head_dim]
                vt = v[:, :, t, :]
                rt = r[:, :, t, :]

                # Time decay (negative log scale for numerical stability)
                # w is stored as positive values, decay = exp(-w)
                decay = mx.exp(-w)  # [heads, head_dim]

                # Compute attention-like scores
                # The "bonus" for current token (time_first)
                current_bonus = mx.exp(u + kt)  # [batch, heads, head_dim]

                # Numerator: accumulated weighted values + current contribution
                # state has shape [batch, heads, head_dim, head_dim]
                # We compute: sum over source dimension
                state_contrib = mx.sum(current_state, axis=-1)  # [batch, heads, head_dim]
                num = state_contrib + current_bonus * vt

                # Denominator for normalization
                den = current_denom + current_bonus
                den = mx.maximum(den, 1e-8)  # Avoid division by zero

                # Output with receptance gating
                out_t = rt * mx.sigmoid(rt) * (num / den)
                outputs.append(out_t[:, :, None, :])

                # Update state with time decay
                # Outer product for state update: v @ k^T
                # state[i,j] tracks weighted sum of v_i * k_j
                vt_expanded = vt[:, :, :, None]  # [batch, heads, head_dim, 1]
                kt[:, :, None, :]  # [batch, heads, 1, head_dim]
                exp_k = mx.exp(kt)  # [batch, heads, head_dim]
                exp_k_expanded = exp_k[:, :, None, :]  # [batch, heads, 1, head_dim]

                # Decay old state and add new contribution
                current_state = decay[None, :, :, None] * current_state + vt_expanded * exp_k_expanded
                current_denom = decay[None, :, :] * current_denom + exp_k

            output = mx.concatenate(outputs, axis=2)
            return output, current_state, current_denom
        else:
            # Numpy fallback - similar logic
            batch_size, num_heads, seq_len, head_dim = k.shape

            outputs = []
            current_state = state.copy()
            current_denom = state_denom.copy()

            for t in range(seq_len):
                kt = k[:, :, t, :]
                vt = v[:, :, t, :]
                rt = r[:, :, t, :]

                decay = np.exp(-w)
                current_bonus = np.exp(u + kt)

                state_contrib = np.sum(current_state, axis=-1)
                num = state_contrib + current_bonus * vt

                den = current_denom + current_bonus
                den = np.maximum(den, 1e-8)

                sigmoid_r = 1 / (1 + np.exp(-rt))
                out_t = rt * sigmoid_r * (num / den)
                outputs.append(out_t[:, :, None, :])

                vt_expanded = vt[:, :, :, None]
                exp_k = np.exp(kt)
                exp_k_expanded = exp_k[:, :, None, :]

                current_state = decay[None, :, :, None] * current_state + vt_expanded * exp_k_expanded
                current_denom = decay[None, :, :] * current_denom + exp_k

            output = np.concatenate(outputs, axis=2)
            return output, current_state, current_denom

    def __call__(
        self,
        hidden_states,  # [batch, seq_len, hidden_size]
        state: RWKVState | None = None,
        layer_idx: int = 0,
    ):
        """Forward pass through time mixing layer.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            state: RWKV state container (optional, creates new if None)
            layer_idx: Layer index for state addressing

        Returns:
            output: [batch, seq_len, hidden_size]
            If state is provided, updates it in-place and returns output only.
            For stateless mode, returns (output, new_time_state, new_denom, new_shift).
        """
        if HAS_MLX and mx is not None:
            batch_size, seq_len, _ = hidden_states.shape

            # Get previous shift state
            if state is not None:
                wkv_state, wkv_denom, shift_state = state.get_time_mixing_state(layer_idx)
            else:
                wkv_state = mx.zeros(
                    (batch_size, self.num_heads, self.head_dim, self.head_dim),
                    dtype=mx.float32
                )
                wkv_denom = mx.zeros(
                    (batch_size, self.num_heads, self.head_dim),
                    dtype=mx.float32
                )
                shift_state = mx.zeros(
                    (batch_size, self.hidden_size),
                    dtype=self.dtype_config.mlx_activations
                )

            # Token shift mixing
            xr = self._token_shift(hidden_states, shift_state, self.time_mix_r)
            xk = self._token_shift(hidden_states, shift_state, self.time_mix_k)
            xv = self._token_shift(hidden_states, shift_state, self.time_mix_v)

            # Project to R, K, V
            r = self.receptance(xr)
            k = self.key(xk)
            v = self.value(xv)

            # Reshape for multi-head processing
            r = r.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

            # Transpose to [batch, heads, seq, dim]
            r = r.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)

            # WKV computation
            wkv_out, new_wkv_state, new_wkv_denom = self._wkv_forward(
                r, k, v, self.time_decay, self.time_first,
                wkv_state, wkv_denom
            )

            # Reshape back: [batch, heads, seq, dim] -> [batch, seq, hidden]
            wkv_out = wkv_out.transpose(0, 2, 1, 3)
            wkv_out = wkv_out.reshape(batch_size, seq_len, self.hidden_size)

            # Gate (v6) or direct output
            if self.gate is not None:
                xg = self._token_shift(hidden_states, shift_state, self.time_mix_g)
                gate = mx.sigmoid(self.gate(xg))
                output = self.output(wkv_out * gate)
            else:
                output = self.output(wkv_out)

            # Update state
            new_shift = hidden_states[:, -1, :]  # Last token becomes shift state

            if state is not None:
                state.update_time_mixing_state(
                    layer_idx, new_wkv_state, new_wkv_denom, new_shift
                )
                return output
            else:
                return output, new_wkv_state, new_wkv_denom, new_shift
        else:
            # Numpy fallback with same logic
            batch_size, seq_len, _ = hidden_states.shape

            if state is not None:
                wkv_state, wkv_denom, shift_state = state.get_time_mixing_state(layer_idx)
            else:
                wkv_state = np.zeros(
                    (batch_size, self.num_heads, self.head_dim, self.head_dim),
                    dtype=np.float32
                )
                wkv_denom = np.zeros(
                    (batch_size, self.num_heads, self.head_dim),
                    dtype=np.float32
                )
                shift_state = np.zeros(
                    (batch_size, self.hidden_size),
                    dtype=np.float16
                )

            xr = self._token_shift(hidden_states, shift_state, self.time_mix_r)
            xk = self._token_shift(hidden_states, shift_state, self.time_mix_k)
            xv = self._token_shift(hidden_states, shift_state, self.time_mix_v)

            r = self.receptance(xr)
            k = self.key(xk)
            v = self.value(xv)

            r = r.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

            r = np.transpose(r, (0, 2, 1, 3))
            k = np.transpose(k, (0, 2, 1, 3))
            v = np.transpose(v, (0, 2, 1, 3))

            wkv_out, new_wkv_state, new_wkv_denom = self._wkv_forward(
                r, k, v, self.time_decay, self.time_first,
                wkv_state, wkv_denom
            )

            wkv_out = np.transpose(wkv_out, (0, 2, 1, 3))
            wkv_out = wkv_out.reshape(batch_size, seq_len, self.hidden_size)

            if self.gate is not None:
                xg = self._token_shift(hidden_states, shift_state, self.time_mix_g)
                gate_val = 1 / (1 + np.exp(-self.gate(xg)))
                output = self.output(wkv_out * gate_val)
            else:
                output = self.output(wkv_out)

            new_shift = hidden_states[:, -1, :]

            if state is not None:
                state.update_time_mixing_state(
                    layer_idx, new_wkv_state, new_wkv_denom, new_shift
                )
                return output
            else:
                return output, new_wkv_state, new_wkv_denom, new_shift


# Make it inherit from nn.Module when MLX is available
if HAS_MLX and nn is not None:
    _OriginalTimeMixing = RWKVTimeMixing

    class RWKVTimeMixing(nn.Module):  # type: ignore[no-redef]
        """Time Mixing layer for RWKV with Marlin-quantized projections."""

        __doc__ = _OriginalTimeMixing.__doc__

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            _OriginalTimeMixing.__init__(self, *args, **kwargs)

        __call__ = _OriginalTimeMixing.__call__
        _token_shift = _OriginalTimeMixing._token_shift
        _wkv_forward = _OriginalTimeMixing._wkv_forward
