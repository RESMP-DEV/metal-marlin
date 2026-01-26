"""RWKV architecture components for Metal Marlin.

RWKV is an RNN-like architecture with transformer-style training. Its core is
the WKV (weighted key-value) operator, which provides linear attention with
constant memory during generation.

This module provides:
  - Metal WKV kernel dispatch (rwkv_wkv.metal)
  - Time mixing and channel mixing layers
  - Layer-local state management compatible with HybridBlock

Notes on quantization:
  - Time decay parameters are sensitive; keep them in FP32.
  - WKV state accumulation is FP32 for stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .._compat import HAS_MLX, mx, nn, require_mlx
from ..dtypes import DTypeConfig, get_default_config
from ..layers import MarlinLinear
from .base import HybridLayerType, LayerState, StateType

# ---------------------------------------------------------------------------
# Metal kernel dispatch for WKV
# ---------------------------------------------------------------------------

_KERNEL_SOURCE_PATH = Path(__file__).parent.parent / "src" / "rwkv_wkv.metal"
_kernel_source: str | None = None
_kernel_cache: dict[str, object] = {}

MAX_HEAD_DIM = 128
THREADS_PER_TG = 128


def _get_kernel_source() -> str:
    global _kernel_source
    if _kernel_source is None:
        _kernel_source = _KERNEL_SOURCE_PATH.read_text()
    return _kernel_source


def _get_kernel(kernel_name: str, input_names: list[str], output_names: list[str]) -> object:
    if kernel_name not in _kernel_cache:
        kernel = mx.fast.metal_kernel(
            name=kernel_name,
            input_names=input_names,
            output_names=output_names,
            source=_get_kernel_source(),
            ensure_row_contiguous=True,
        )
        _kernel_cache[kernel_name] = kernel
    return _kernel_cache[kernel_name]


def _uint_scalar(value: int) -> Any:
    return mx.array([value], dtype=mx.uint32)


def rwkv_wkv_single_token(
    r: Any,
    k: Any,
    v: Any,
    w: Any,
    u: Any,
    state: Any,
    state_denom: Any,
    output_dtype: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Dispatch single-token WKV kernel.

    Args:
        r, k, v: [batch, heads, head_dim] (float16)
        w, u: [heads, head_dim] (float32)
        state: [batch, heads, head_dim, head_dim] (float32)
        state_denom: [batch, heads, head_dim] (float32)
    """
    require_mlx("RWKV WKV kernel")

    batch_size, num_heads, head_dim = r.shape
    if head_dim > MAX_HEAD_DIM:
        raise ValueError(f"head_dim {head_dim} exceeds MAX_HEAD_DIM {MAX_HEAD_DIM}")

    if output_dtype is None:
        output_dtype = mx.float16

    kernel = _get_kernel(
        "rwkv_wkv_single_token",
        input_names=[
            "r",
            "k",
            "v",
            "w",
            "u",
            "state",
            "state_denom",
            "batch_size",
            "num_heads",
            "head_dim",
        ],
        output_names=["output", "new_state", "new_state_denom"],
    )

    outputs = kernel(
        inputs=[
            r.astype(mx.float16),
            k.astype(mx.float16),
            v.astype(mx.float16),
            w.astype(mx.float32),
            u.astype(mx.float32),
            state.astype(mx.float32),
            state_denom.astype(mx.float32),
            _uint_scalar(batch_size),
            _uint_scalar(num_heads),
            _uint_scalar(head_dim),
        ],
        grid=(batch_size, num_heads, 1),
        threadgroup=(THREADS_PER_TG, 1, 1),
        output_shapes=[
            (batch_size, num_heads, head_dim),
            state.shape,
            state_denom.shape,
        ],
        output_dtypes=[output_dtype, mx.float32, mx.float32],
    )

    return outputs[0], outputs[1], outputs[2]


def rwkv_wkv_batched(
    r: Any,
    k: Any,
    v: Any,
    w: Any,
    u: Any,
    state: Any,
    state_denom: Any,
    output_dtype: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Dispatch batched WKV kernel for prefill.

    Args:
        r, k, v: [batch, seq_len, heads, head_dim] (float16)
        w, u: [heads, head_dim] (float32)
        state: [batch, heads, head_dim, head_dim] (float32)
        state_denom: [batch, heads, head_dim] (float32)
    """
    require_mlx("RWKV WKV kernel")

    batch_size, seq_len, num_heads, head_dim = r.shape
    if head_dim > MAX_HEAD_DIM:
        raise ValueError(f"head_dim {head_dim} exceeds MAX_HEAD_DIM {MAX_HEAD_DIM}")

    if output_dtype is None:
        output_dtype = mx.float16

    kernel = _get_kernel(
        "rwkv_wkv_batched",
        input_names=[
            "r",
            "k",
            "v",
            "w",
            "u",
            "initial_state",
            "initial_denom",
            "batch_size",
            "seq_len",
            "num_heads",
            "head_dim",
        ],
        output_names=["output", "final_state", "final_denom"],
    )

    outputs = kernel(
        inputs=[
            r.astype(mx.float16),
            k.astype(mx.float16),
            v.astype(mx.float16),
            w.astype(mx.float32),
            u.astype(mx.float32),
            state.astype(mx.float32),
            state_denom.astype(mx.float32),
            _uint_scalar(batch_size),
            _uint_scalar(seq_len),
            _uint_scalar(num_heads),
            _uint_scalar(head_dim),
        ],
        grid=(batch_size, num_heads, 1),
        threadgroup=(THREADS_PER_TG, 1, 1),
        output_shapes=[
            (batch_size, seq_len, num_heads, head_dim),
            state.shape,
            state_denom.shape,
        ],
        output_dtypes=[output_dtype, mx.float32, mx.float32],
    )

    return outputs[0], outputs[1], outputs[2]


# ---------------------------------------------------------------------------
# RWKV state (layer-local)
# ---------------------------------------------------------------------------


@dataclass
class RWKVLayerState:
    """Per-layer RWKV recurrent state."""

    wkv_state: Any
    wkv_denom: Any
    time_shift: Any
    channel_shift: Any

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        dtype_config: DTypeConfig,
        state_precision: Literal["full", "fp32"] = "fp32",
    ) -> RWKVLayerState:
        if HAS_MLX and mx is not None:
            act_dtype = dtype_config.mlx_activations
            state_dtype = mx.float32 if state_precision == "fp32" else act_dtype
            return cls(
                wkv_state=mx.zeros((batch_size, num_heads, head_dim, head_dim), dtype=state_dtype),
                wkv_denom=mx.zeros((batch_size, num_heads, head_dim), dtype=state_dtype),
                time_shift=mx.zeros((batch_size, hidden_size), dtype=act_dtype),
                channel_shift=mx.zeros((batch_size, hidden_size), dtype=act_dtype),
            )

        act_dtype = np.float16
        state_dtype = np.float32 if state_precision == "fp32" else act_dtype
        return cls(
            wkv_state=np.zeros((batch_size, num_heads, head_dim, head_dim), dtype=state_dtype),
            wkv_denom=np.zeros((batch_size, num_heads, head_dim), dtype=state_dtype),
            time_shift=np.zeros((batch_size, hidden_size), dtype=act_dtype),
            channel_shift=np.zeros((batch_size, hidden_size), dtype=act_dtype),
        )

    def with_time(self, wkv_state: Any, wkv_denom: Any, time_shift: Any) -> RWKVLayerState:
        return RWKVLayerState(
            wkv_state=wkv_state,
            wkv_denom=wkv_denom,
            time_shift=time_shift,
            channel_shift=self.channel_shift,
        )

    def with_channel(self, channel_shift: Any) -> RWKVLayerState:
        return RWKVLayerState(
            wkv_state=self.wkv_state,
            wkv_denom=self.wkv_denom,
            time_shift=self.time_shift,
            channel_shift=channel_shift,
        )


# ---------------------------------------------------------------------------
# Time mixing and channel mixing
# ---------------------------------------------------------------------------


class RWKVTimeMixing:
    """RWKV time mixing with optional Metal WKV kernel."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        quant_type: Literal["fp4", "int4"] = "fp4",
        group_size: int = 128,
        layer_id: int = 0,
        num_layers: int = 24,
        version: Literal["v5", "v6", "eagle"] = "v6",
        dtype_config: DTypeConfig | None = None,
        state_precision: Literal["full", "fp32"] = "fp32",
        use_wkv_kernel: bool = True,
    ):
        if HAS_MLX and nn is not None:
            nn.Module.__init__(self)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.layer_id = layer_id
        self.version = version
        self.dtype_config = dtype_config if dtype_config is not None else get_default_config()
        self.state_precision = state_precision
        self.use_wkv_kernel = use_wkv_kernel and HAS_MLX

        if self.head_dim * num_heads != hidden_size:
            raise ValueError(
                f"head_dim ({self.head_dim}) * num_heads ({num_heads}) "
                f"must equal hidden_size ({hidden_size})"
            )

        # Token shift mixing ratios
        if HAS_MLX and mx is not None:
            dtype = self.dtype_config.mlx_activations
            self.time_mix_r = mx.ones((hidden_size,), dtype=dtype) * 0.5
            self.time_mix_k = mx.ones((hidden_size,), dtype=dtype) * 0.5
            self.time_mix_v = mx.ones((hidden_size,), dtype=dtype) * 0.5
            self.time_mix_g = mx.ones((hidden_size,), dtype=dtype) * 0.5

            # Time decay and time first remain FP32 (quantization sensitive).
            decay_speed = 1.0 - (layer_id / max(num_layers - 1, 1))
            self.time_decay = mx.ones((num_heads, self.head_dim), dtype=mx.float32) * decay_speed
            self.time_first = mx.zeros((num_heads, self.head_dim), dtype=mx.float32)
        else:
            self.time_mix_r = np.ones((hidden_size,), dtype=np.float16) * 0.5
            self.time_mix_k = np.ones((hidden_size,), dtype=np.float16) * 0.5
            self.time_mix_v = np.ones((hidden_size,), dtype=np.float16) * 0.5
            self.time_mix_g = np.ones((hidden_size,), dtype=np.float16) * 0.5

            decay_speed = 1.0 - (layer_id / max(num_layers - 1, 1))
            self.time_decay = np.ones((num_heads, self.head_dim), dtype=np.float32) * decay_speed
            self.time_first = np.zeros((num_heads, self.head_dim), dtype=np.float32)

        # Quantized projections
        self.receptance = MarlinLinear(
            hidden_size, hidden_size,
            bias=False, quant_type=quant_type, group_size=group_size,
            dtype_config=self.dtype_config,
        )
        self.key = MarlinLinear(
            hidden_size, hidden_size,
            bias=False, quant_type=quant_type, group_size=group_size,
            dtype_config=self.dtype_config,
        )
        self.value = MarlinLinear(
            hidden_size, hidden_size,
            bias=False, quant_type=quant_type, group_size=group_size,
            dtype_config=self.dtype_config,
        )
        self.output = MarlinLinear(
            hidden_size, hidden_size,
            bias=False, quant_type=quant_type, group_size=group_size,
            dtype_config=self.dtype_config,
        )

        if version in ("v6", "eagle"):
            self.gate = MarlinLinear(
                hidden_size, hidden_size,
                bias=False, quant_type=quant_type, group_size=group_size,
                dtype_config=self.dtype_config,
            )
        else:
            self.gate = None

    def _token_shift(self, x: Any, x_prev: Any, mix_ratio: Any) -> Any:
        if HAS_MLX and mx is not None:
            batch_size, seq_len, _ = x.shape
            if seq_len == 1:
                return x * mix_ratio + x_prev[:, None, :] * (1 - mix_ratio)
            x_shifted = mx.concatenate([x_prev[:, None, :], x[:, :-1, :]], axis=1)
            return x * mix_ratio + x_shifted * (1 - mix_ratio)

        batch_size, seq_len, _ = x.shape
        if seq_len == 1:
            return x * mix_ratio + x_prev[:, None, :] * (1 - mix_ratio)
        x_shifted = np.concatenate([x_prev[:, None, :], x[:, :-1, :]], axis=1)
        return x * mix_ratio + x_shifted * (1 - mix_ratio)

    def _wkv_forward(
        self,
        r: Any,
        k: Any,
        v: Any,
        w: Any,
        u: Any,
        state: Any,
        state_denom: Any,
    ) -> tuple[Any, Any, Any]:
        """Compute WKV. Expects r/k/v as [batch, seq, heads, head_dim]."""
        if HAS_MLX and mx is not None and self.use_wkv_kernel:
            batch_size, seq_len, num_heads, head_dim = k.shape
            if head_dim <= MAX_HEAD_DIM:
                if seq_len == 1:
                    r0 = r[:, 0, :, :]
                    k0 = k[:, 0, :, :]
                    v0 = v[:, 0, :, :]
                    out, new_state, new_denom = rwkv_wkv_single_token(
                        r0, k0, v0, w, u, state, state_denom,
                        output_dtype=self.dtype_config.mlx_activations,
                    )
                    return out[:, None, :, :], new_state, new_denom

                out, new_state, new_denom = rwkv_wkv_batched(
                    r, k, v, w, u, state, state_denom,
                    output_dtype=self.dtype_config.mlx_activations,
                )
                return out, new_state, new_denom

        # Fallback: Python loop (works with MLX or NumPy)
        if HAS_MLX and mx is not None:
            batch_size, seq_len, num_heads, head_dim = k.shape
            outputs = []
            current_state = state
            current_denom = state_denom
            decay = mx.exp(-w)
            for t in range(seq_len):
                kt = k[:, t, :, :]
                vt = v[:, t, :, :]
                rt = r[:, t, :, :]
                current_bonus = mx.exp(u + kt)
                state_contrib = mx.sum(current_state, axis=-1)
                num = state_contrib + current_bonus * vt
                den = mx.maximum(current_denom + current_bonus, 1e-8)
                out_t = rt * mx.sigmoid(rt) * (num / den)
                outputs.append(out_t[:, None, :, :])
                exp_k = mx.exp(kt)
                current_state = decay[None, :, :, None] * current_state + vt[:, :, :, None] * exp_k[:, :, None, :]
                current_denom = decay[None, :, :] * current_denom + exp_k
            output = mx.concatenate(outputs, axis=1)
            return output, current_state, current_denom

        batch_size, seq_len, num_heads, head_dim = k.shape
        outputs = []
        current_state = state.copy()
        current_denom = state_denom.copy()
        decay = np.exp(-w)
        for t in range(seq_len):
            kt = k[:, t, :, :]
            vt = v[:, t, :, :]
            rt = r[:, t, :, :]
            current_bonus = np.exp(u + kt)
            state_contrib = np.sum(current_state, axis=-1)
            num = state_contrib + current_bonus * vt
            den = np.maximum(current_denom + current_bonus, 1e-8)
            sigmoid_r = 1 / (1 + np.exp(-rt))
            out_t = rt * sigmoid_r * (num / den)
            outputs.append(out_t[:, None, :, :])
            exp_k = np.exp(kt)
            current_state = decay[None, :, :, None] * current_state + vt[:, :, :, None] * exp_k[:, :, None, :]
            current_denom = decay[None, :, :] * current_denom + exp_k
        output = np.concatenate(outputs, axis=1)
        return output, current_state, current_denom

    def __call__(
        self,
        hidden_states: Any,  # [batch, seq_len, hidden_size]
        state: RWKVLayerState | None = None,
    ) -> tuple[Any, RWKVLayerState]:
        if HAS_MLX and mx is not None:
            batch_size, seq_len, _ = hidden_states.shape
            if state is None:
                state = RWKVLayerState.zeros(
                    batch_size=batch_size,
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    dtype_config=self.dtype_config,
                    state_precision=self.state_precision,
                )

            xr = self._token_shift(hidden_states, state.time_shift, self.time_mix_r)
            xk = self._token_shift(hidden_states, state.time_shift, self.time_mix_k)
            xv = self._token_shift(hidden_states, state.time_shift, self.time_mix_v)

            r = self.receptance(xr)
            k = self.key(xk)
            v = self.value(xv)

            r = r.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

            wkv_out, new_state, new_denom = self._wkv_forward(
                r, k, v, self.time_decay, self.time_first,
                state.wkv_state, state.wkv_denom,
            )

            wkv_out = wkv_out.reshape(batch_size, seq_len, self.hidden_size)

            if self.gate is not None:
                xg = self._token_shift(hidden_states, state.time_shift, self.time_mix_g)
                gate = mx.sigmoid(self.gate(xg))
                output = self.output(wkv_out * gate)
            else:
                output = self.output(wkv_out)

            new_shift = hidden_states[:, -1, :]
            return output, state.with_time(new_state, new_denom, new_shift)

        # NumPy fallback
        batch_size, seq_len, _ = hidden_states.shape
        if state is None:
            state = RWKVLayerState.zeros(
                batch_size=batch_size,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dtype_config=self.dtype_config,
                state_precision=self.state_precision,
            )

        xr = self._token_shift(hidden_states, state.time_shift, self.time_mix_r)
        xk = self._token_shift(hidden_states, state.time_shift, self.time_mix_k)
        xv = self._token_shift(hidden_states, state.time_shift, self.time_mix_v)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)

        r = r.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        wkv_out, new_state, new_denom = self._wkv_forward(
            r, k, v, self.time_decay, self.time_first,
            state.wkv_state, state.wkv_denom,
        )

        wkv_out = wkv_out.reshape(batch_size, seq_len, self.hidden_size)
        if self.gate is not None:
            xg = self._token_shift(hidden_states, state.time_shift, self.time_mix_g)
            gate = 1 / (1 + np.exp(-self.gate(xg)))
            output = self.output(wkv_out * gate)
        else:
            output = self.output(wkv_out)

        new_shift = hidden_states[:, -1, :]
        return output, state.with_time(new_state, new_denom, new_shift)


class RWKVChannelMixing:
    """RWKV channel mixing (FFN-like) layer."""

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

        if HAS_MLX and mx is not None:
            dtype = self.dtype_config.mlx_activations
            self.channel_mix_k = mx.ones((hidden_size,), dtype=dtype) * 0.5
            self.channel_mix_r = mx.ones((hidden_size,), dtype=dtype) * 0.5
        else:
            self.channel_mix_k = np.ones((hidden_size,), dtype=np.float16) * 0.5
            self.channel_mix_r = np.ones((hidden_size,), dtype=np.float16) * 0.5

        self.key = MarlinLinear(
            hidden_size, intermediate_size,
            bias=False, quant_type=quant_type, group_size=group_size,
            dtype_config=self.dtype_config,
        )
        self.value = MarlinLinear(
            intermediate_size, hidden_size,
            bias=False, quant_type=quant_type, group_size=group_size,
            dtype_config=self.dtype_config,
        )
        self.receptance = MarlinLinear(
            hidden_size, hidden_size,
            bias=False, quant_type=quant_type, group_size=group_size,
            dtype_config=self.dtype_config,
        )

    def _token_shift(self, x: Any, x_prev: Any, mix_ratio: Any) -> Any:
        if HAS_MLX and mx is not None:
            batch_size, seq_len, _ = x.shape
            if seq_len == 1:
                return x * mix_ratio + x_prev[:, None, :] * (1 - mix_ratio)
            x_shifted = mx.concatenate([x_prev[:, None, :], x[:, :-1, :]], axis=1)
            return x * mix_ratio + x_shifted * (1 - mix_ratio)

        batch_size, seq_len, _ = x.shape
        if seq_len == 1:
            return x * mix_ratio + x_prev[:, None, :] * (1 - mix_ratio)
        x_shifted = np.concatenate([x_prev[:, None, :], x[:, :-1, :]], axis=1)
        return x * mix_ratio + x_shifted * (1 - mix_ratio)

    def __call__(
        self,
        hidden_states: Any,
        state: RWKVLayerState | None = None,
    ) -> tuple[Any, RWKVLayerState]:
        if HAS_MLX and mx is not None:
            batch_size, seq_len, _ = hidden_states.shape
            if state is None:
                state = RWKVLayerState.zeros(
                    batch_size=batch_size,
                    hidden_size=self.hidden_size,
                    num_heads=1,
                    head_dim=self.hidden_size,
                    dtype_config=self.dtype_config,
                )

            xk = self._token_shift(hidden_states, state.channel_shift, self.channel_mix_k)
            xr = self._token_shift(hidden_states, state.channel_shift, self.channel_mix_r)

            k = self.key(xk)
            k = mx.square(mx.maximum(k, 0))
            kv = self.value(k)

            r = mx.sigmoid(self.receptance(xr))
            output = r * kv

            new_shift = hidden_states[:, -1, :]
            return output, state.with_channel(new_shift)

        batch_size, seq_len, _ = hidden_states.shape
        if state is None:
            state = RWKVLayerState.zeros(
                batch_size=batch_size,
                hidden_size=self.hidden_size,
                num_heads=1,
                head_dim=self.hidden_size,
                dtype_config=self.dtype_config,
            )

        xk = self._token_shift(hidden_states, state.channel_shift, self.channel_mix_k)
        xr = self._token_shift(hidden_states, state.channel_shift, self.channel_mix_r)

        k = self.key(xk)
        k = np.square(np.maximum(k, 0))
        kv = self.value(k)

        r = 1 / (1 + np.exp(-self.receptance(xr)))
        output = r * kv

        new_shift = hidden_states[:, -1, :]
        return output, state.with_channel(new_shift)


class LayerNorm:
    """Standard LayerNorm for RWKV blocks."""

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

    def __call__(self, x: Any) -> Any:
        if HAS_MLX and mx is not None:
            mean = mx.mean(x, axis=-1, keepdims=True)
            variance = mx.var(x, axis=-1, keepdims=True)
            x = (x - mean) * mx.rsqrt(variance + self.eps)
            return self.weight * x + self.bias
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x = (x - mean) / np.sqrt(variance + self.eps)
        return self.weight * x + self.bias


class RWKVBlock(nn.Module if HAS_MLX else object):
    """RWKV block compatible with HybridBlock."""

    layer_type = HybridLayerType.LINEAR_ATTENTION
    state_type = StateType.SSM_STATE

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        head_dim: int | None = None,
        quant_type: Literal["fp4", "int4"] = "fp4",
        group_size: int = 128,
        layer_id: int = 0,
        num_layers: int = 24,
        version: Literal["v5", "v6", "eagle"] = "v6",
        layer_norm_eps: float = 1e-5,
        dtype_config: DTypeConfig | None = None,
        state_precision: Literal["full", "fp32"] = "fp32",
        use_wkv_kernel: bool = True,
    ):
        if HAS_MLX and nn is not None:
            nn.Module.__init__(self)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.layer_id = layer_id
        self.num_layers = num_layers
        self.dtype_config = dtype_config if dtype_config is not None else get_default_config()
        self.state_precision = state_precision

        self.ln1 = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln2 = LayerNorm(hidden_size, eps=layer_norm_eps)

        self.time_mixing = RWKVTimeMixing(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=self.head_dim,
            quant_type=quant_type,
            group_size=group_size,
            layer_id=layer_id,
            num_layers=num_layers,
            version=version,
            dtype_config=self.dtype_config,
            state_precision=state_precision,
            use_wkv_kernel=use_wkv_kernel,
        )

        self.channel_mixing = RWKVChannelMixing(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_type=quant_type,
            group_size=group_size,
            dtype_config=self.dtype_config,
        )

    def __call__(
        self,
        hidden_states: Any,
        state: LayerState | None = None,
        layer_idx: int = 0,
    ) -> tuple[Any, LayerState]:
        if state is None or not isinstance(state.ssm_state, RWKVLayerState):
            if HAS_MLX and mx is not None:
                batch_size = hidden_states.shape[0]
            else:
                batch_size = hidden_states.shape[0]
            rwkv_state = RWKVLayerState.zeros(
                batch_size=batch_size,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dtype_config=self.dtype_config,
                state_precision=self.state_precision,
            )
            state = LayerState(
                state_type=StateType.SSM_STATE,
                layer_idx=layer_idx,
                ssm_state=rwkv_state,
            )
        else:
            rwkv_state = state.ssm_state

        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        time_out, rwkv_state = self.time_mixing(hidden_states, rwkv_state)
        hidden_states = residual + time_out

        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        chan_out, rwkv_state = self.channel_mixing(hidden_states, rwkv_state)
        hidden_states = residual + chan_out

        new_state = LayerState(
            state_type=StateType.SSM_STATE,
            layer_idx=layer_idx,
            ssm_state=rwkv_state,
        )
        return hidden_states, new_state

    def init_state(self, batch_size: int, layer_idx: int) -> LayerState:
        rwkv_state = RWKVLayerState.zeros(
            batch_size=batch_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype_config=self.dtype_config,
            state_precision=self.state_precision,
        )
        return LayerState(
            state_type=StateType.SSM_STATE,
            layer_idx=layer_idx,
            ssm_state=rwkv_state,
        )
