"""
KV Cache management for PyTorch MPS-based inference.

Stores key/value tensors from previous tokens to avoid recomputation during
decode phase. Works with PyTorch MPS tensors for Metal-accelerated inference.

Usage:
    from metal_marlin.kv_cache_torch import KVCacheTorch, CacheConfigTorch

    config = CacheConfigTorch(
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,  # GQA
        head_dim=128,
        max_seq_len=4096,
    )

    cache = KVCacheTorch(config, batch_size=1)

    # During forward pass
    k_full, v_full = cache.update(layer_idx=0, k_new=k, v_new=v)
    cache.advance(num_tokens=seq_len)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ._compat import require_torch, torch

if TYPE_CHECKING:
    import torch as torch_typing


@dataclass
class CacheConfigTorch:
    """Configuration for PyTorch-based KV cache.

    Attributes:
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        num_kv_heads: Number of key-value heads (for GQA).
        head_dim: Dimension per head.
        max_seq_len: Maximum sequence length to allocate.
        dtype: Cache data type (fp16, bf16, or fp32).
    """

    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    dtype: Literal["fp16", "bf16", "fp32"] = "fp16"


class KVCacheTorch:
    """
    Key-Value cache for PyTorch MPS-based inference.

    Supports:
    - Standard MHA (num_heads == num_kv_heads)
    - Grouped Query Attention (num_kv_heads < num_heads)
    - FP16, BF16, or FP32 precision

    The cache pre-allocates tensors for max_seq_len to avoid reallocation
    during generation. Only the slice up to current seq_len is used.
    """

    def __init__(
        self,
        config: CacheConfigTorch,
        batch_size: int = 1,
        device: str = "mps",
    ):
        require_torch()

        self.config = config
        self.batch_size = batch_size
        self.seq_len = 0
        self.device = device

        # Determine torch dtype
        if config.dtype == "fp16":
            self._dtype = torch.float16
        elif config.dtype == "bf16":
            self._dtype = torch.bfloat16
        else:
            self._dtype = torch.float32

        # Shape: [batch, num_kv_heads, max_seq_len, head_dim]
        cache_shape = (
            batch_size,
            config.num_kv_heads,
            config.max_seq_len,
            config.head_dim,
        )

        # Pre-allocate cache tensors
        self.k_cache: list[torch_typing.Tensor] = [
            torch.zeros(cache_shape, dtype=self._dtype, device=device)
            for _ in range(config.num_layers)
        ]
        self.v_cache: list[torch_typing.Tensor] = [
            torch.zeros(cache_shape, dtype=self._dtype, device=device)
            for _ in range(config.num_layers)
        ]

    def update(
        self,
        layer_idx: int,
        k_new: torch_typing.Tensor,
        v_new: torch_typing.Tensor,
    ) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """
        Update cache with new K, V and return full cached K, V.

        Args:
            layer_idx: Which transformer layer
            k_new: New key tensor [batch, num_kv_heads, new_seq_len, head_dim]
            v_new: New value tensor [batch, num_kv_heads, new_seq_len, head_dim]

        Returns:
            (k_full, v_full) with shape [batch, num_kv_heads, seq_len + new_seq_len, head_dim]
        """
        new_seq_len = k_new.shape[2]
        end_pos = self.seq_len + new_seq_len

        if end_pos > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {end_pos} exceeds max_seq_len {self.config.max_seq_len}"
            )

        # Update cache slices in-place
        self.k_cache[layer_idx][:, :, self.seq_len : end_pos, :] = k_new.to(self._dtype)
        self.v_cache[layer_idx][:, :, self.seq_len : end_pos, :] = v_new.to(self._dtype)

        # Return the full cached tensors up to end_pos
        k_full = self.k_cache[layer_idx][:, :, :end_pos, :]
        v_full = self.v_cache[layer_idx][:, :, :end_pos, :]

        return k_full, v_full

    def advance(self, num_tokens: int = 1) -> None:
        """Advance sequence position after processing tokens."""
        self.seq_len += num_tokens

    def reset(self) -> None:
        """Clear cache for new sequence."""
        self.seq_len = 0

    def get_kv(
        self, layer_idx: int
    ) -> tuple[torch_typing.Tensor | None, torch_typing.Tensor | None]:
        """Get current cached K, V for a layer.

        Returns None if cache is empty.
        """
        if self.seq_len == 0:
            return None, None

        k = self.k_cache[layer_idx][:, :, : self.seq_len, :]
        v = self.v_cache[layer_idx][:, :, : self.seq_len, :]

        return k, v

    def get_layer(self, layer_idx: int) -> KVCacheLayerView:
        """Get a view of the cache for a specific layer.

        Useful for passing to attention functions that expect
        a single-layer cache interface.
        """
        return KVCacheLayerView(self, layer_idx)

    def memory_usage_mb(self) -> float:
        """Return current memory usage in MB."""
        bytes_per_element = 2 if self._dtype in (torch.float16, torch.bfloat16) else 4

        elements = (
            self.batch_size
            * self.config.num_kv_heads
            * self.seq_len
            * self.config.head_dim
            * 2  # K and V
            * self.config.num_layers
        )
        return elements * bytes_per_element / 1024 / 1024


class KVCacheLayerView:
    """View of a single layer's KV cache.

    Provides a simplified interface for attention modules that
    work with single-layer caches.
    """

    def __init__(self, parent: KVCacheTorch, layer_idx: int):
        self._parent = parent
        self._layer_idx = layer_idx

    @property
    def k(self) -> torch_typing.Tensor | None:
        """Current cached keys for this layer."""
        if self._parent.seq_len == 0:
            return None
        return self._parent.k_cache[self._layer_idx][:, :, : self._parent.seq_len, :]

    @property
    def v(self) -> torch_typing.Tensor | None:
        """Current cached values for this layer."""
        if self._parent.seq_len == 0:
            return None
        return self._parent.v_cache[self._layer_idx][:, :, : self._parent.seq_len, :]

    def update(
        self,
        k_new: torch_typing.Tensor,
        v_new: torch_typing.Tensor,
    ) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Update this layer's cache with new K, V."""
        return self._parent.update(self._layer_idx, k_new, v_new)
