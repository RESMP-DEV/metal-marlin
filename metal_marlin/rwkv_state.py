"""RWKV state management for efficient autoregressive inference.

RWKV uses a constant-size recurrent state per layer, unlike transformer KV cache
which grows with sequence length. This makes RWKV memory-efficient for long
sequences but requires careful state management.

The state consists of:
- Time mixing state: for the WKV attention-like mechanism
- Channel mixing state: for the feed-forward network

Usage:
    from metal_marlin.rwkv_state import RWKVState, RWKVStateConfig

    config = RWKVStateConfig(
        num_layers=24,
        hidden_size=2048,
        num_heads=32,  # RWKV v5/v6 use multi-headed state
    )

    state = RWKVState(config, batch_size=1)

    # During forward pass
    new_time_state = state.update_time_mixing(layer_idx, new_state)
    new_channel_state = state.update_channel_mixing(layer_idx, new_state)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._compat import HAS_MLX, mx
from .dtypes import DTypeConfig, get_default_config


@dataclass
class RWKVStateConfig:
    """Configuration for RWKV state.

    Attributes:
        num_layers: Number of RWKV layers.
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads (for v5/v6 multi-headed state).
        head_dim: Dimension per head. Defaults to hidden_size // num_heads.
        version: RWKV version (v5 or v6). v5 uses simpler state, v6 adds more.
        state_precision: Precision for state storage.
            - "full": Uses dtype_config.activations (bf16/fp16)
            - "fp32": Always use FP32 for numerical stability
    """

    num_layers: int
    hidden_size: int
    num_heads: int = 1
    head_dim: int | None = None
    version: Literal["v5", "v6"] = "v6"
    state_precision: Literal["full", "fp32"] = "fp32"

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads


class RWKVState:
    """RWKV recurrent state for inference.

    Unlike transformer KV cache which stores all past tokens, RWKV compresses
    history into fixed-size state tensors. This enables O(1) memory for
    generation regardless of context length.

    State structure per layer:
    - Time mixing: [batch, num_heads, head_dim, head_dim] - the "wkv" state
    - Time mixing shift: [batch, hidden_size] - previous token's hidden state
    - Channel mixing shift: [batch, hidden_size] - previous token for channel mix

    For RWKV v5/v6, the time mixing state is a running weighted sum that
    approximates attention over all past tokens via exponential decay.
    """

    def __init__(
        self,
        config: RWKVStateConfig,
        batch_size: int = 1,
        dtype_config: DTypeConfig | None = None,
    ):
        self.config = config
        self.batch_size = batch_size
        self.dtype_config = dtype_config if dtype_config is not None else get_default_config()
        self._initialized = False

        # Determine storage dtype
        if config.state_precision == "fp32":
            self._np_dtype = np.float32
            if HAS_MLX and mx is not None:
                self._mlx_dtype = mx.float32
            else:
                self._mlx_dtype = None
        else:
            self._np_dtype = np.float16  # or bf16 via config
            if HAS_MLX and mx is not None:
                self._mlx_dtype = self.dtype_config.mlx_activations
            else:
                self._mlx_dtype = None

        # Pre-allocate state tensors
        self._allocate_states()

    def _allocate_states(self):
        """Allocate state tensors for all layers."""
        cfg = self.config
        num_layers = cfg.num_layers
        batch = self.batch_size
        hidden = cfg.hidden_size
        heads = cfg.num_heads
        head_dim = cfg.head_dim

        if HAS_MLX and mx is not None:
            dtype = self._mlx_dtype
            # Time mixing WKV state: [batch, heads, head_dim, head_dim]
            # This stores the running weighted sum for linear attention
            self.time_state = [
                mx.zeros((batch, heads, head_dim, head_dim), dtype=dtype)
                for _ in range(num_layers)
            ]
            # Time mixing denominator (for normalization): [batch, heads, head_dim]
            self.time_state_denom = [
                mx.zeros((batch, heads, head_dim), dtype=dtype)
                for _ in range(num_layers)
            ]
            # Shift states for token mixing: [batch, hidden]
            self.time_shift = [
                mx.zeros((batch, hidden), dtype=dtype)
                for _ in range(num_layers)
            ]
            self.channel_shift = [
                mx.zeros((batch, hidden), dtype=dtype)
                for _ in range(num_layers)
            ]
        else:
            # Numpy fallback
            dtype = self._np_dtype
            self.time_state = [
                np.zeros((batch, heads, head_dim, head_dim), dtype=dtype)
                for _ in range(num_layers)
            ]
            self.time_state_denom = [
                np.zeros((batch, heads, head_dim), dtype=dtype)
                for _ in range(num_layers)
            ]
            self.time_shift = [
                np.zeros((batch, hidden), dtype=dtype)
                for _ in range(num_layers)
            ]
            self.channel_shift = [
                np.zeros((batch, hidden), dtype=dtype)
                for _ in range(num_layers)
            ]

        self._initialized = True

    def get_time_mixing_state(self, layer_idx: int):
        """Get time mixing state for a layer.

        Returns:
            Tuple of (wkv_state, wkv_denom, shift_state)
            - wkv_state: [batch, heads, head_dim, head_dim]
            - wkv_denom: [batch, heads, head_dim]
            - shift_state: [batch, hidden]
        """
        return (
            self.time_state[layer_idx],
            self.time_state_denom[layer_idx],
            self.time_shift[layer_idx],
        )

    def update_time_mixing_state(
        self,
        layer_idx: int,
        new_wkv_state,
        new_wkv_denom,
        new_shift,
    ):
        """Update time mixing state for a layer.

        Args:
            layer_idx: Layer index.
            new_wkv_state: New WKV state [batch, heads, head_dim, head_dim]
            new_wkv_denom: New WKV denominator [batch, heads, head_dim]
            new_shift: New shift state [batch, hidden]
        """
        self.time_state[layer_idx] = new_wkv_state
        self.time_state_denom[layer_idx] = new_wkv_denom
        self.time_shift[layer_idx] = new_shift

    def get_channel_mixing_state(self, layer_idx: int):
        """Get channel mixing shift state for a layer.

        Returns:
            shift_state: [batch, hidden]
        """
        return self.channel_shift[layer_idx]

    def update_channel_mixing_state(self, layer_idx: int, new_shift):
        """Update channel mixing state for a layer.

        Args:
            layer_idx: Layer index.
            new_shift: New shift state [batch, hidden]
        """
        self.channel_shift[layer_idx] = new_shift

    def reset(self):
        """Reset all states to zeros for a new sequence."""
        self._allocate_states()

    def clone(self) -> RWKVState:
        """Create a deep copy of the state.

        Useful for parallel generation (beam search, speculative decoding).
        """
        new_state = RWKVState(
            self.config,
            self.batch_size,
            self.dtype_config,
        )

        if HAS_MLX and mx is not None:
            for i in range(self.config.num_layers):
                new_state.time_state[i] = mx.array(self.time_state[i])
                new_state.time_state_denom[i] = mx.array(self.time_state_denom[i])
                new_state.time_shift[i] = mx.array(self.time_shift[i])
                new_state.channel_shift[i] = mx.array(self.channel_shift[i])
        else:
            for i in range(self.config.num_layers):
                new_state.time_state[i] = self.time_state[i].copy()
                new_state.time_state_denom[i] = self.time_state_denom[i].copy()
                new_state.time_shift[i] = self.time_shift[i].copy()
                new_state.channel_shift[i] = self.channel_shift[i].copy()

        return new_state

    def memory_usage_mb(self) -> float:
        """Return current memory usage in MB."""
        cfg = self.config
        bytes_per_element = 4.0 if cfg.state_precision == "fp32" else 2.0

        # Per layer: wkv_state + wkv_denom + 2x shift
        wkv_state_size = self.batch_size * cfg.num_heads * cfg.head_dim * cfg.head_dim
        wkv_denom_size = self.batch_size * cfg.num_heads * cfg.head_dim
        shift_size = self.batch_size * cfg.hidden_size

        total_per_layer = wkv_state_size + wkv_denom_size + 2 * shift_size
        total_elements = total_per_layer * cfg.num_layers

        return total_elements * bytes_per_element / 1024 / 1024

    def __repr__(self) -> str:
        return (
            f"RWKVState(layers={self.config.num_layers}, "
            f"hidden={self.config.hidden_size}, "
            f"heads={self.config.num_heads}, "
            f"batch={self.batch_size}, "
            f"memory={self.memory_usage_mb():.2f}MB)"
        )
