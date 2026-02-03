"""RWKV architecture components for Metal Marlin.

RWKV is an RNN-like architecture with transformer-style training. Its core is
the WKV (weighted key-value) operator, which provides linear attention with
constant memory during generation.

This module provides:
  - Metal WKV kernel dispatch (rwkv_wkv.metal) via PyObjC
  - Time mixing and channel mixing layers
  - Layer-local state management compatible with HybridBlock

Notes on quantization:
  - Time decay parameters are sensitive; keep them in FP32.
  - WKV state accumulation is FP32 for stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._compat import HAS_MPS, HAS_PYOBJC_METAL, HAS_TORCH, torch
from ..dtypes import DTypeConfig, get_default_config
from ..layers import MarlinLinear
from .base import HybridLayerType, LayerState, StateType

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Metal kernel dispatch for WKV
# ---------------------------------------------------------------------------

# Metal dispatch requires both PyTorch MPS and PyObjC Metal framework
HAS_METAL_DISPATCH: bool = HAS_MPS and HAS_PYOBJC_METAL

# Path to kernel source
_KERNEL_SOURCE_PATH = Path(__file__).parent.parent.parent / "src" / "rwkv_wkv.metal"

# Lazy-loaded Metal library
_metal_lib: Any = None

MAX_HEAD_DIM = 128
THREADS_PER_TG = 128


def _get_metal_library() -> Any:
    """Get or create the Metal kernel library with RWKV kernels compiled."""
    global _metal_lib
    if _metal_lib is None:
        from ..metal_dispatch import MetalKernelLibrary

        _metal_lib = MetalKernelLibrary()
        # Compile the RWKV WKV kernel
        source = _KERNEL_SOURCE_PATH.read_text()
        _metal_lib.compile_source("rwkv_wkv", source)
    return _metal_lib


def _dispatch_wkv_single_token(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    state: torch.Tensor,
    state_denom: torch.Tensor,
    batch_size: int,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch single-token WKV kernel via Metal."""
    import Metal

    from ..metal_dispatch import dispatch_kernel, mps_tensor_to_metal_buffer

    lib = _get_metal_library()
    device = lib.device

    # Allocate outputs
    output = torch.empty((batch_size, num_heads, head_dim), dtype=torch.float16, device="mps")
    new_state = torch.empty_like(state, dtype=torch.float32, device="mps")
    new_state_denom = torch.empty_like(state_denom, dtype=torch.float32, device="mps")

    # Convert tensors to Metal buffers
    r_buf = mps_tensor_to_metal_buffer(r.half().contiguous(), device)
    k_buf = mps_tensor_to_metal_buffer(k.half().contiguous(), device)
    v_buf = mps_tensor_to_metal_buffer(v.half().contiguous(), device)
    w_buf = mps_tensor_to_metal_buffer(w.float().contiguous(), device)
    u_buf = mps_tensor_to_metal_buffer(u.float().contiguous(), device)
    state_buf = mps_tensor_to_metal_buffer(state.float().contiguous(), device)
    state_denom_buf = mps_tensor_to_metal_buffer(state_denom.float().contiguous(), device)
    output_buf = mps_tensor_to_metal_buffer(output, device)
    new_state_buf = mps_tensor_to_metal_buffer(new_state, device)
    new_state_denom_buf = mps_tensor_to_metal_buffer(new_state_denom, device)

    # Create constant buffers
    bs_buf = device.newBufferWithBytes_length_options_(
        np.array([batch_size], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    nh_buf = device.newBufferWithBytes_length_options_(
        np.array([num_heads], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    hd_buf = device.newBufferWithBytes_length_options_(
        np.array([head_dim], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )

    # Dispatch
    dispatch_kernel(
        lib,
        function_name="rwkv_wkv_single_token",
        grid=(batch_size, num_heads, 1),
        threadgroup=(THREADS_PER_TG, 1, 1),
        buffers=[
            r_buf,
            k_buf,
            v_buf,
            w_buf,
            u_buf,
            state_buf,
            state_denom_buf,
            output_buf,
            new_state_buf,
            new_state_denom_buf,
            bs_buf,
            nh_buf,
            hd_buf,
        ],
        wait=True,
    )

    return output, new_state, new_state_denom


def _dispatch_wkv_batched(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    state: torch.Tensor,
    state_denom: torch.Tensor,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch batched WKV kernel for prefill via Metal."""
    import Metal

    from ..metal_dispatch import dispatch_kernel, mps_tensor_to_metal_buffer

    lib = _get_metal_library()
    device = lib.device

    # Allocate outputs
    output = torch.empty(
        (batch_size, seq_len, num_heads, head_dim), dtype=torch.float16, device="mps"
    )
    final_state = torch.empty_like(state, dtype=torch.float32, device="mps")
    final_denom = torch.empty_like(state_denom, dtype=torch.float32, device="mps")

    # Convert tensors to Metal buffers
    r_buf = mps_tensor_to_metal_buffer(r.half().contiguous(), device)
    k_buf = mps_tensor_to_metal_buffer(k.half().contiguous(), device)
    v_buf = mps_tensor_to_metal_buffer(v.half().contiguous(), device)
    w_buf = mps_tensor_to_metal_buffer(w.float().contiguous(), device)
    u_buf = mps_tensor_to_metal_buffer(u.float().contiguous(), device)
    state_buf = mps_tensor_to_metal_buffer(state.float().contiguous(), device)
    state_denom_buf = mps_tensor_to_metal_buffer(state_denom.float().contiguous(), device)
    output_buf = mps_tensor_to_metal_buffer(output, device)
    final_state_buf = mps_tensor_to_metal_buffer(final_state, device)
    final_denom_buf = mps_tensor_to_metal_buffer(final_denom, device)

    # Create constant buffers
    bs_buf = device.newBufferWithBytes_length_options_(
        np.array([batch_size], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    sl_buf = device.newBufferWithBytes_length_options_(
        np.array([seq_len], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    nh_buf = device.newBufferWithBytes_length_options_(
        np.array([num_heads], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    hd_buf = device.newBufferWithBytes_length_options_(
        np.array([head_dim], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )

    # Dispatch
    dispatch_kernel(
        lib,
        function_name="rwkv_wkv_batched",
        grid=(batch_size, num_heads, 1),
        threadgroup=(THREADS_PER_TG, 1, 1),
        buffers=[
            r_buf,
            k_buf,
            v_buf,
            w_buf,
            u_buf,
            state_buf,
            state_denom_buf,
            output_buf,
            final_state_buf,
            final_denom_buf,
            bs_buf,
            sl_buf,
            nh_buf,
            hd_buf,
        ],
        wait=True,
    )

    return output, final_state, final_denom


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
        output_dtype: Optional output dtype (torch.float16 default)

    Returns:
        (output, new_state, new_state_denom)
    """
    if HAS_METAL_DISPATCH and torch is not None:
        # Ensure inputs are on MPS
        if not isinstance(r, torch.Tensor):
            r = torch.from_numpy(np.asarray(r)).to("mps")
        elif not r.is_mps:
            r = r.to("mps")

        if not isinstance(k, torch.Tensor):
            k = torch.from_numpy(np.asarray(k)).to("mps")
        elif not k.is_mps:
            k = k.to("mps")

        if not isinstance(v, torch.Tensor):
            v = torch.from_numpy(np.asarray(v)).to("mps")
        elif not v.is_mps:
            v = v.to("mps")

        if not isinstance(w, torch.Tensor):
            w = torch.from_numpy(np.asarray(w)).to("mps")
        elif not w.is_mps:
            w = w.to("mps")

        if not isinstance(u, torch.Tensor):
            u = torch.from_numpy(np.asarray(u)).to("mps")
        elif not u.is_mps:
            u = u.to("mps")

        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(np.asarray(state)).to("mps")
        elif not state.is_mps:
            state = state.to("mps")

        if not isinstance(state_denom, torch.Tensor):
            state_denom = torch.from_numpy(np.asarray(state_denom)).to("mps")
        elif not state_denom.is_mps:
            state_denom = state_denom.to("mps")

        batch_size, num_heads, head_dim = r.shape
        if head_dim > MAX_HEAD_DIM:
            raise ValueError(f"head_dim {head_dim} exceeds MAX_HEAD_DIM {MAX_HEAD_DIM}")

        return _dispatch_wkv_single_token(
            r, k, v, w, u, state, state_denom, batch_size, num_heads, head_dim
        )

    # NumPy fallback
    raise RuntimeError(
        "RWKV WKV kernel requires PyTorch MPS and PyObjC Metal.\n"
        "Install with: pip install torch pyobjc-framework-Metal"
    )


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
        output_dtype: Optional output dtype (torch.float16 default)

    Returns:
        (output, final_state, final_denom)
    """
    if HAS_METAL_DISPATCH and torch is not None:
        # Ensure inputs are on MPS
        if not isinstance(r, torch.Tensor):
            r = torch.from_numpy(np.asarray(r)).to("mps")
        elif not r.is_mps:
            r = r.to("mps")

        if not isinstance(k, torch.Tensor):
            k = torch.from_numpy(np.asarray(k)).to("mps")
        elif not k.is_mps:
            k = k.to("mps")

        if not isinstance(v, torch.Tensor):
            v = torch.from_numpy(np.asarray(v)).to("mps")
        elif not v.is_mps:
            v = v.to("mps")

        if not isinstance(w, torch.Tensor):
            w = torch.from_numpy(np.asarray(w)).to("mps")
        elif not w.is_mps:
            w = w.to("mps")

        if not isinstance(u, torch.Tensor):
            u = torch.from_numpy(np.asarray(u)).to("mps")
        elif not u.is_mps:
            u = u.to("mps")

        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(np.asarray(state)).to("mps")
        elif not state.is_mps:
            state = state.to("mps")

        if not isinstance(state_denom, torch.Tensor):
            state_denom = torch.from_numpy(np.asarray(state_denom)).to("mps")
        elif not state_denom.is_mps:
            state_denom = state_denom.to("mps")

        batch_size, seq_len, num_heads, head_dim = r.shape
        if head_dim > MAX_HEAD_DIM:
            raise ValueError(f"head_dim {head_dim} exceeds MAX_HEAD_DIM {MAX_HEAD_DIM}")

        return _dispatch_wkv_batched(
            r, k, v, w, u, state, state_denom, batch_size, seq_len, num_heads, head_dim
        )

    # NumPy fallback
    raise RuntimeError(
        "RWKV WKV kernel requires PyTorch MPS and PyObjC Metal.\n"
        "Install with: pip install torch pyobjc-framework-Metal"
    )


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
        if HAS_TORCH and torch is not None:
            state_dtype = torch.float32 if state_precision == "fp32" else torch.float16
            act_dtype = torch.float16
            device = "mps" if HAS_MPS else "cpu"
            return cls(
                wkv_state=torch.zeros(
                    (batch_size, num_heads, head_dim, head_dim), dtype=state_dtype, device=device
                ),
                wkv_denom=torch.zeros(
                    (batch_size, num_heads, head_dim), dtype=state_dtype, device=device
                ),
                time_shift=torch.zeros((batch_size, hidden_size), dtype=act_dtype, device=device),
                channel_shift=torch.zeros(
                    (batch_size, hidden_size), dtype=act_dtype, device=device
                ),
            )

        # NumPy fallback
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
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.layer_id = layer_id
        self.version = version
        self.dtype_config = dtype_config if dtype_config is not None else get_default_config()
        self.state_precision: Literal["full", "fp32"] = state_precision
        self.use_wkv_kernel = use_wkv_kernel and HAS_METAL_DISPATCH

        if self.head_dim * num_heads != hidden_size:
            raise ValueError(
                f"head_dim ({self.head_dim}) * num_heads ({num_heads}) "
                f"must equal hidden_size ({hidden_size})"
            )

        # Token shift mixing ratios
        if HAS_TORCH and torch is not None:
            device = "mps" if HAS_MPS else "cpu"
            dtype = torch.float16
            self.time_mix_r = torch.ones((hidden_size,), dtype=dtype, device=device) * 0.5
            self.time_mix_k = torch.ones((hidden_size,), dtype=dtype, device=device) * 0.5
            self.time_mix_v = torch.ones((hidden_size,), dtype=dtype, device=device) * 0.5
            self.time_mix_g = torch.ones((hidden_size,), dtype=dtype, device=device) * 0.5

            # Time decay and time first remain FP32 (quantization sensitive).
            decay_speed = 1.0 - (layer_id / max(num_layers - 1, 1))
            self.time_decay = (
                torch.ones((num_heads, self.head_dim), dtype=torch.float32, device=device)
                * decay_speed
            )
            self.time_first = torch.zeros(
                (num_heads, self.head_dim), dtype=torch.float32, device=device
            )
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
            hidden_size,
            hidden_size,
            bias=False,
            quant_type=quant_type,
            group_size=group_size,
            dtype_config=self.dtype_config,
        )
        self.key = MarlinLinear(
            hidden_size,
            hidden_size,
            bias=False,
            quant_type=quant_type,
            group_size=group_size,
            dtype_config=self.dtype_config,
        )
        self.value = MarlinLinear(
            hidden_size,
            hidden_size,
            bias=False,
            quant_type=quant_type,
            group_size=group_size,
            dtype_config=self.dtype_config,
        )
        self.output = MarlinLinear(
            hidden_size,
            hidden_size,
            bias=False,
            quant_type=quant_type,
            group_size=group_size,
            dtype_config=self.dtype_config,
        )

        if version in ("v6", "eagle"):
            self.gate = MarlinLinear(
                hidden_size,
                hidden_size,
                bias=False,
                quant_type=quant_type,
                group_size=group_size,
                dtype_config=self.dtype_config,
            )
        else:
            self.gate = None

    def _token_shift(self, x: Any, x_prev: Any, mix_ratio: Any) -> Any:
        if HAS_TORCH and torch is not None and isinstance(x, torch.Tensor):
            batch_size, seq_len, _ = x.shape
            if seq_len == 1:
                return x * mix_ratio + x_prev[:, None, :] * (1 - mix_ratio)
            x_shifted = torch.cat([x_prev[:, None, :], x[:, :-1, :]], dim=1)
            return x * mix_ratio + x_shifted * (1 - mix_ratio)

        # NumPy fallback
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
        if HAS_METAL_DISPATCH and torch is not None and self.use_wkv_kernel:
            batch_size, seq_len, num_heads, head_dim = k.shape
            if head_dim <= MAX_HEAD_DIM:
                if seq_len == 1:
                    r0 = r[:, 0, :, :]
                    k0 = k[:, 0, :, :]
                    v0 = v[:, 0, :, :]
                    out, new_state, new_denom = rwkv_wkv_single_token(
                        r0,
                        k0,
                        v0,
                        w,
                        u,
                        state,
                        state_denom,
                        output_dtype=torch.float16,
                    )
                    return out[:, None, :, :], new_state, new_denom

                out, new_state, new_denom = rwkv_wkv_batched(
                    r,
                    k,
                    v,
                    w,
                    u,
                    state,
                    state_denom,
                    output_dtype=torch.float16,
                )
                return out, new_state, new_denom

        # Fallback: Python loop (works with PyTorch or NumPy)
        if HAS_TORCH and torch is not None and isinstance(k, torch.Tensor):
            batch_size, seq_len, num_heads, head_dim = k.shape
            outputs = []
            current_state = state.clone()
            current_denom = state_denom.clone()
            decay = torch.exp(-w)
            for t in range(seq_len):
                kt = k[:, t, :, :]
                vt = v[:, t, :, :]
                rt = r[:, t, :, :]
                current_bonus = torch.exp(u + kt)
                state_contrib = torch.sum(current_state, dim=-1)
                num = state_contrib + current_bonus * vt
                den = torch.maximum(
                    current_denom + current_bonus, torch.tensor(1e-8, device=k.device)
                )
                out_t = rt * torch.sigmoid(rt) * (num / den)
                outputs.append(out_t[:, None, :, :])
                exp_k = torch.exp(kt)
                current_state = (
                    decay[None, :, :, None] * current_state
                    + vt[:, :, :, None] * exp_k[:, :, None, :]
                )
                current_denom = decay[None, :, :] * current_denom + exp_k
            output = torch.cat(outputs, dim=1)
            return output, current_state, current_denom

        # NumPy fallback
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
            current_state = (
                decay[None, :, :, None] * current_state + vt[:, :, :, None] * exp_k[:, :, None, :]
            )
            current_denom = decay[None, :, :] * current_denom + exp_k
        output = np.concatenate(outputs, axis=1)
        return output, current_state, current_denom

    def __call__(
        self,
        hidden_states: Any,  # [batch, seq_len, hidden_size]
        state: RWKVLayerState | None = None,
    ) -> tuple[Any, RWKVLayerState]:
        if HAS_TORCH and torch is not None and isinstance(hidden_states, torch.Tensor):
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
                r,
                k,
                v,
                self.time_decay,
                self.time_first,
                state.wkv_state,
                state.wkv_denom,
            )

            wkv_out = wkv_out.reshape(batch_size, seq_len, self.hidden_size)

            if self.gate is not None:
                xg = self._token_shift(hidden_states, state.time_shift, self.time_mix_g)
                gate = torch.sigmoid(self.gate(xg))
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
            r,
            k,
            v,
            self.time_decay,
            self.time_first,
            state.wkv_state,
            state.wkv_denom,
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
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dtype_config = dtype_config if dtype_config is not None else get_default_config()

        if HAS_TORCH and torch is not None:
            device = "mps" if HAS_MPS else "cpu"
            dtype = torch.float16
            self.channel_mix_k = torch.ones((hidden_size,), dtype=dtype, device=device) * 0.5
            self.channel_mix_r = torch.ones((hidden_size,), dtype=dtype, device=device) * 0.5
        else:
            self.channel_mix_k = np.ones((hidden_size,), dtype=np.float16) * 0.5
            self.channel_mix_r = np.ones((hidden_size,), dtype=np.float16) * 0.5

        self.key = MarlinLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_type=quant_type,
            group_size=group_size,
            dtype_config=self.dtype_config,
        )
        self.value = MarlinLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_type=quant_type,
            group_size=group_size,
            dtype_config=self.dtype_config,
        )
        self.receptance = MarlinLinear(
            hidden_size,
            hidden_size,
            bias=False,
            quant_type=quant_type,
            group_size=group_size,
            dtype_config=self.dtype_config,
        )

    def _token_shift(self, x: Any, x_prev: Any, mix_ratio: Any) -> Any:
        if HAS_TORCH and torch is not None and isinstance(x, torch.Tensor):
            batch_size, seq_len, _ = x.shape
            if seq_len == 1:
                return x * mix_ratio + x_prev[:, None, :] * (1 - mix_ratio)
            x_shifted = torch.cat([x_prev[:, None, :], x[:, :-1, :]], dim=1)
            return x * mix_ratio + x_shifted * (1 - mix_ratio)

        # NumPy fallback
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
        if HAS_TORCH and torch is not None and isinstance(hidden_states, torch.Tensor):
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
            k = torch.square(torch.relu(k))
            kv = self.value(k)

            r = torch.sigmoid(self.receptance(xr))
            output = r * kv

            new_shift = hidden_states[:, -1, :]
            return output, state.with_channel(new_shift)

        # NumPy fallback
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
        if HAS_TORCH and torch is not None:
            device = "mps" if HAS_MPS else "cpu"
            self.weight = torch.ones((hidden_size,), device=device)
            self.bias = torch.zeros((hidden_size,), device=device)
        else:
            self.weight = np.ones((hidden_size,), dtype=np.float32)
            self.bias = np.zeros((hidden_size,), dtype=np.float32)

        self.eps = eps
        self.hidden_size = hidden_size

    def __call__(self, x: Any) -> Any:
        if HAS_TORCH and torch is not None and isinstance(x, torch.Tensor):
            mean = torch.mean(x, dim=-1, keepdim=True)
            variance = torch.var(x, dim=-1, keepdim=True, unbiased=False)
            x = (x - mean) * torch.rsqrt(variance + self.eps)
            return self.weight * x + self.bias
        # NumPy fallback
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x = (x - mean) / np.sqrt(variance + self.eps)
        return self.weight * x + self.bias


class RWKVBlock:
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
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.layer_id = layer_id
        self.num_layers = num_layers
        self.dtype_config = dtype_config if dtype_config is not None else get_default_config()
        self.state_precision: Literal["full", "fp32"] = state_precision

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
            if HAS_TORCH and torch is not None and isinstance(hidden_states, torch.Tensor):
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
