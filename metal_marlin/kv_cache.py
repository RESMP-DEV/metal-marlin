"""Unified KV cache management for efficient autoregressive inference.

This module provides high-performance KV cache implementations for Metal Marlin,
supporting various model architectures and hardware backends.

Historical Context and Consolidation:
-------------------------------------
This file is the result of consolidating several specialized KV cache implementations
that previously existed in the codebase:
- `kv_cache_torch.py`: The original PyTorch-based implementation.
- `mla_kv_cache.py`: A cache specifically for Multi-head Latent Attention (MLA) models.
- `trellis/kv_cache.py`: A cache tailored for the Trellis architecture.

These have been unified into this single module to reduce code duplication and
streamline maintenance. Legacy aliases are provided at the end of the file for
backwards compatibility.

Cache Implementations:
----------------------
The module now contains the following primary classes:

1.  **KVCacheTorch (aliased as KVCache)**:
    - The standard, general-purpose KV cache.
    - Optimized for standard Multi-Head Attention (MHA) and Grouped-Query
      Attention (GQA) transformers.
    - Supports multiple quantization formats (FP8, FP4, INT8) and memory layouts
      to balance performance and memory usage.

2.  **MLAKVCache**:
    - A specialized cache for models using Multi-head Latent Attention (MLA).
    - It stores a compressed latent representation of the keys and values,
      significantly reducing the memory footprint for long-context models.
    - Optimized for Apple Silicon (Metal Performance Shaders).

3.  **TrellisKVCache**:
    - Trellis-focused behavior wrapper around `MLAKVCache`.
    - Preserves Trellis API semantics for global sequence tracking and memory accounting.

3.  **CompressedKVCache**:
    - An extension of `MLAKVCache` that adds features like sliding window attention
      and late quantization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ._compat import require_torch, torch
from .dtypes import DTypeConfig, get_default_config

if TYPE_CHECKING:
    import torch as torch_typing

# Mapping from dtype config strings to PyTorch dtypes
if TYPE_CHECKING:
    _DTYPE_TO_TORCH_CACHE: dict[str, torch.dtype] = {}
else:
    _DTYPE_TO_TORCH_CACHE: dict[str, object] = {}


def require_mps(feature: str = "KV cache") -> None:
    """Raise if PyTorch MPS is not available."""
    require_torch(feature)
    # Check for MPS availability safely (torch is guaranteed not None by require_torch)
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():  # type: ignore
        raise RuntimeError(
            f"{feature} requires PyTorch MPS backend. "
            "Ensure you're on Apple Silicon with PyTorch >= 2.0"
        )


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to PyTorch dtype."""
    require_torch("dtype conversion")

    if not _DTYPE_TO_TORCH_CACHE:
        _DTYPE_TO_TORCH_CACHE.update(
            {
                "fp16": getattr(torch, "float16", torch.float32),
                "bf16": getattr(torch, "bfloat16", torch.float32),
                "fp32": getattr(torch, "float32", None),
                "fp8": getattr(torch, "float16", torch.float32),
                # Aliases
                "float16": getattr(torch, "float16", torch.float32),
                "bfloat16": getattr(torch, "bfloat16", torch.float32),
                "float32": getattr(torch, "float32", None),
            }
        )

    if dtype_str not in _DTYPE_TO_TORCH_CACHE:
        raise ValueError(f"Unknown dtype: {dtype_str}")

    dtype = _DTYPE_TO_TORCH_CACHE[dtype_str]
    if dtype is None:
        raise RuntimeError(
            f"Torch dtype {dtype_str} not available in this environment")

    return dtype  # type: ignore[return-value]


def _resolve_dtype(dtype: torch.dtype | str) -> torch.dtype:
    """Resolve dtype string to torch dtype (helper for MLA)."""
    if isinstance(dtype, str):
        return _get_torch_dtype(dtype)
    return dtype

# =============================================================================
# Standard KVCacheTorch Implementation
# =============================================================================


@dataclass
class CacheConfigTorch:
    """Configuration for PyTorch-based KV cache.

    Attributes:
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        num_kv_heads: Number of key-value heads (for GQA).
        head_dim: Dimension per head.
        max_seq_len: Maximum sequence length to allocate.
        cache_dtype: Cache storage dtype.
            - "fp16": Half precision storage
            - "bf16": BFloat16 storage
            - "fp8": FP8 E4M3 quantized (2x memory savings)
            - "fp4": FP4 E2M1 quantized (4x memory savings)
            - "int8": INT8 symmetric quantization (2x memory savings)
        fp8_scale_method: Scaling method for FP8 quantization.
            - "tensor": Single scale per tensor (faster, lower memory)
            - "channel": Scale per head dimension (better accuracy for outliers)
        layout: Memory layout for cache storage.
            - "BHSD": [batch, num_heads, seq_len, head_dim] (default, standard)
            - "BSHD": [batch, seq_len, num_heads, head_dim] (faster decode writes)
    """

    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    cache_dtype: Literal["fp16", "bf16", "fp8", "fp4", "int8"] | None = None
    fp8_scale_method: Literal["tensor", "channel"] = "tensor"
    layout: Literal["BHSD", "BSHD"] = "BHSD"


class KVCacheTorch:
    """
    Key-Value cache for transformer inference using PyTorch MPS.

    Supports:
    - Standard MHA (num_heads == num_kv_heads)
    - Grouped Query Attention (num_kv_heads < num_heads)
    - Configurable precision via DTypeConfig:
        - BF16 (default): Better dynamic range
        - FP16: Maximum compatibility
        - FP8: 2x memory savings for long context
        - FP4: 4x memory savings for very long context
        - INT8: 2x memory savings with symmetric quantization

    The cache pre-allocates tensors for max_seq_len to avoid reallocation
    during generation. Only the slice up to current seq_len is used.
    """

    def __init__(
        self,
        config: CacheConfigTorch,
        batch_size: int = 1,
        dtype_config: DTypeConfig | None = None,
        device: str = "mps",
    ):
        require_mps("KV cache")

        self.config = config
        self.batch_size = batch_size
        self.seq_len = 0
        self.cache_position = 0
        self.dtype_config = dtype_config if dtype_config is not None else get_default_config()
        self.device = device
        self._fp8_scale_method = config.fp8_scale_method
        self._update_id = 0
        self._last_eviction_update_id = -1
        self._layout = config.layout

        # Layout determines memory arrangement
        if self._layout == "BSHD":
            cache_shape = (
                batch_size,
                config.max_seq_len,
                config.num_kv_heads,
                config.head_dim,
            )
        else:  # BHSD (default)
            cache_shape = (
                batch_size,
                config.num_kv_heads,
                config.max_seq_len,
                config.head_dim,
            )

        # Determine storage mode from cache_dtype
        self._storage_dtype_override = None
        if config.cache_dtype == "fp8":
            self._quantize_mode = "fp8"
        elif config.cache_dtype == "fp4":
            self._quantize_mode = "fp4"
        elif config.cache_dtype == "int8":
            self._quantize_mode = "int8"
        elif config.cache_dtype in ("fp16", "bf16"):
            self._quantize_mode = "none"
            self._storage_dtype_override = config.cache_dtype
        else:
            self._quantize_mode = "none"

        if self._quantize_mode == "fp4":
            if self._layout == "BSHD":
                raise ValueError("FP4 quantization only supports BHSD layout")
            packed_shape = (
                batch_size,
                config.num_kv_heads,
                config.max_seq_len,
                config.head_dim // 8,
            )
            self.k_cache: list[torch.Tensor] = [
                torch.zeros(packed_shape, dtype=torch.int32, device=device)
                for _ in range(config.num_layers)
            ]
            self.v_cache: list[torch.Tensor] = [
                torch.zeros(packed_shape, dtype=torch.int32, device=device)
                for _ in range(config.num_layers)
            ]
            scale_dtype = _get_torch_dtype(self.dtype_config.scales)
            scale_shape = (batch_size, config.num_kv_heads,
                           config.max_seq_len, 1)
            self.k_scales: list[torch.Tensor] | None = [
                torch.zeros(scale_shape, dtype=scale_dtype, device=device)
                for _ in range(config.num_layers)
            ]
            self.v_scales: list[torch.Tensor] | None = [
                torch.zeros(scale_shape, dtype=scale_dtype, device=device)
                for _ in range(config.num_layers)
            ]
        elif self._quantize_mode == "fp8":
            if self._layout == "BSHD":
                raise ValueError("FP8 quantization only supports BHSD layout")
            fp8_shape = (batch_size, config.num_kv_heads,
                         config.max_seq_len, config.head_dim)
            self.k_cache = [
                torch.zeros(fp8_shape, dtype=torch.uint8, device=device)
                for _ in range(config.num_layers)
            ]
            self.v_cache = [
                torch.zeros(fp8_shape, dtype=torch.uint8, device=device)
                for _ in range(config.num_layers)
            ]
            scale_dtype = _get_torch_dtype(self.dtype_config.scales)
            if config.fp8_scale_method == "channel":
                scale_shape = (batch_size, config.num_kv_heads,
                               config.max_seq_len, config.head_dim)
            else:
                scale_shape = (batch_size, config.num_kv_heads,
                               config.max_seq_len, 1)
            self.k_scales = [
                torch.zeros(scale_shape, dtype=scale_dtype, device=device)
                for _ in range(config.num_layers)
            ]
            self.v_scales = [
                torch.zeros(scale_shape, dtype=scale_dtype, device=device)
                for _ in range(config.num_layers)
            ]
        elif self._quantize_mode == "int8":
            if self._layout == "BSHD":
                raise ValueError("INT8 quantization only supports BHSD layout")
            int8_shape = (batch_size, config.num_kv_heads,
                          config.max_seq_len, config.head_dim)
            self.k_cache = [
                torch.zeros(int8_shape, dtype=torch.int8, device=device)
                for _ in range(config.num_layers)
            ]
            self.v_cache = [
                torch.zeros(int8_shape, dtype=torch.int8, device=device)
                for _ in range(config.num_layers)
            ]
            scale_dtype = _get_torch_dtype(self.dtype_config.scales)
            scale_shape = (batch_size, config.num_kv_heads,
                           config.max_seq_len, 1)
            self.k_scales = [
                torch.zeros(scale_shape, dtype=scale_dtype, device=device)
                for _ in range(config.num_layers)
            ]
            self.v_scales = [
                torch.zeros(scale_shape, dtype=scale_dtype, device=device)
                for _ in range(config.num_layers)
            ]
        else:
            storage_dtype_str = self._storage_dtype_override or self.dtype_config.kv_cache
            storage_dtype = _get_torch_dtype(storage_dtype_str)
            self.k_cache = [
                torch.zeros(cache_shape, dtype=storage_dtype, device=device)
                for _ in range(config.num_layers)
            ]
            self.v_cache = [
                torch.zeros(cache_shape, dtype=storage_dtype, device=device)
                for _ in range(config.num_layers)
            ]
            self.k_scales = None
            self.v_scales = None

    def update(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new K, V and return full cached K, V."""
        new_seq_len = k_new.shape[2]
        if new_seq_len > self.config.max_seq_len:
            k_new = k_new[:, :, -self.config.max_seq_len:, :]
            v_new = v_new[:, :, -self.config.max_seq_len:, :]
            new_seq_len = self.config.max_seq_len
            self._reset_positions()

        self._maybe_evict(new_seq_len)

        start_pos = self.cache_position
        end_pos = start_pos + new_seq_len
        act_dtype = _get_torch_dtype(self.dtype_config.activations)

        if self._quantize_mode == "fp4":
            k_packed, k_scale = self._quantize_fp4(k_new)
            v_packed, v_scale = self._quantize_fp4(v_new)
            self._slice_update(
                self.k_cache[layer_idx], k_packed, start_pos, new_seq_len)
            self._slice_update(
                self.v_cache[layer_idx], v_packed, start_pos, new_seq_len)
            self._slice_update(
                self.k_scales[layer_idx], k_scale, start_pos, new_seq_len)
            self._slice_update(
                self.v_scales[layer_idx], v_scale, start_pos, new_seq_len)
            k_full = self._dequant_fp4(
                self.k_cache[layer_idx][:, :, :end_pos, :],
                self.k_scales[layer_idx][:, :, :end_pos, :],
            ).to(act_dtype)
            v_full = self._dequant_fp4(
                self.v_cache[layer_idx][:, :, :end_pos, :],
                self.v_scales[layer_idx][:, :, :end_pos, :],
            ).to(act_dtype)
        elif self._quantize_mode == "fp8":
            k_quant, k_scale = self._quantize_fp8(k_new)
            v_quant, v_scale = self._quantize_fp8(v_new)
            self._slice_update(
                self.k_cache[layer_idx], k_quant, start_pos, new_seq_len)
            self._slice_update(
                self.v_cache[layer_idx], v_quant, start_pos, new_seq_len)
            self._slice_update(
                self.k_scales[layer_idx], k_scale, start_pos, new_seq_len)
            self._slice_update(
                self.v_scales[layer_idx], v_scale, start_pos, new_seq_len)
            k_full = self._dequant_fp8(
                self.k_cache[layer_idx][:, :, :end_pos, :],
                self.k_scales[layer_idx][:, :, :end_pos, :],
            ).to(act_dtype)
            v_full = self._dequant_fp8(
                self.v_cache[layer_idx][:, :, :end_pos, :],
                self.v_scales[layer_idx][:, :, :end_pos, :],
            ).to(act_dtype)
        elif self._quantize_mode == "int8":
            k_quant, k_scale = self._quantize_int8(k_new)
            v_quant, v_scale = self._quantize_int8(v_new)
            self._slice_update(
                self.k_cache[layer_idx], k_quant, start_pos, new_seq_len)
            self._slice_update(
                self.v_cache[layer_idx], v_quant, start_pos, new_seq_len)
            self._slice_update(
                self.k_scales[layer_idx], k_scale, start_pos, new_seq_len)
            self._slice_update(
                self.v_scales[layer_idx], v_scale, start_pos, new_seq_len)
            k_full = self._dequant_int8(
                self.k_cache[layer_idx][:, :, :end_pos, :],
                self.k_scales[layer_idx][:, :, :end_pos, :],
            ).to(act_dtype)
            v_full = self._dequant_int8(
                self.v_cache[layer_idx][:, :, :end_pos, :],
                self.v_scales[layer_idx][:, :, :end_pos, :],
            ).to(act_dtype)
        else:
            storage_dtype_str = self._storage_dtype_override or self.dtype_config.kv_cache
            storage_dtype = _get_torch_dtype(storage_dtype_str)

            if self._layout == "BSHD":
                k_transposed = k_new.permute(0, 2, 1, 3)
                v_transposed = v_new.permute(0, 2, 1, 3)
                self._slice_update_bshd(
                    self.k_cache[layer_idx], k_transposed, start_pos, new_seq_len, storage_dtype
                )
                self._slice_update_bshd(
                    self.v_cache[layer_idx], v_transposed, start_pos, new_seq_len, storage_dtype
                )
                k_full = self.k_cache[layer_idx][:, :end_pos, :, :].permute(
                    0, 2, 1, 3).to(act_dtype)
                v_full = self.v_cache[layer_idx][:, :end_pos, :, :].permute(
                    0, 2, 1, 3).to(act_dtype)
            else:
                self._slice_update(
                    self.k_cache[layer_idx], k_new, start_pos, new_seq_len, storage_dtype
                )
                self._slice_update(
                    self.v_cache[layer_idx], v_new, start_pos, new_seq_len, storage_dtype
                )
                k_full = self.k_cache[layer_idx][:,
                                                 :, :end_pos, :].to(act_dtype)
                v_full = self.v_cache[layer_idx][:,
                                                 :, :end_pos, :].to(act_dtype)

        return k_full, v_full

    def advance(self, num_tokens: int = 1) -> None:
        """Advance sequence position after processing tokens."""
        self.seq_len += num_tokens
        self.cache_position = self.seq_len
        self._update_id += 1

    def reset(self) -> None:
        """Clear cache for new sequence."""
        self._reset_positions()

    def get_kv(self, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Get current cached K, V for a layer (dequantized if needed)."""
        if self.seq_len == 0:
            return None, None

        act_dtype = _get_torch_dtype(self.dtype_config.activations)

        if self._quantize_mode == "fp4":
            k = self._dequant_fp4(
                self.k_cache[layer_idx][:, :, : self.seq_len, :],
                self.k_scales[layer_idx][:, :, : self.seq_len, :],
            ).to(act_dtype)
            v = self._dequant_fp4(
                self.v_cache[layer_idx][:, :, : self.seq_len, :],
                self.v_scales[layer_idx][:, :, : self.seq_len, :],
            ).to(act_dtype)
        elif self._quantize_mode == "fp8":
            k = self._dequant_fp8(
                self.k_cache[layer_idx][:, :, : self.seq_len, :],
                self.k_scales[layer_idx][:, :, : self.seq_len, :],
            ).to(act_dtype)
            v = self._dequant_fp8(
                self.v_cache[layer_idx][:, :, : self.seq_len, :],
                self.v_scales[layer_idx][:, :, : self.seq_len, :],
            ).to(act_dtype)
        elif self._quantize_mode == "int8":
            k = self._dequant_int8(
                self.k_cache[layer_idx][:, :, : self.seq_len, :],
                self.k_scales[layer_idx][:, :, : self.seq_len, :],
            ).to(act_dtype)
            v = self._dequant_int8(
                self.v_cache[layer_idx][:, :, : self.seq_len, :],
                self.v_scales[layer_idx][:, :, : self.seq_len, :],
            ).to(act_dtype)
        else:
            if self._layout == "BSHD":
                k = self.k_cache[layer_idx][:, : self.seq_len,
                                            :, :].permute(0, 2, 1, 3).to(act_dtype)
                v = self.v_cache[layer_idx][:, : self.seq_len,
                                            :, :].permute(0, 2, 1, 3).to(act_dtype)
            else:
                k = self.k_cache[layer_idx][:, :,
                                            : self.seq_len, :].to(act_dtype)
                v = self.v_cache[layer_idx][:, :,
                                            : self.seq_len, :].to(act_dtype)

        return k, v

    def memory_usage_mb(self) -> float:
        """Return current memory usage in MB."""
        if self._quantize_mode == "fp4":
            scale_bytes = 4.0 if self.dtype_config.scales == "fp32" else 2.0
            bytes_per_element = 0.5 + scale_bytes / self.config.head_dim
        elif self._quantize_mode == "fp8":
            scale_bytes = 4.0 if self.dtype_config.scales == "fp32" else 2.0
            bytes_per_element = 1.0 + scale_bytes / self.config.head_dim
        elif self._quantize_mode == "int8":
            scale_bytes = 4.0 if self.dtype_config.scales == "fp32" else 2.0
            bytes_per_element = 1.0 + scale_bytes / self.config.head_dim
        else:
            bytes_per_element = 2

        elements = (
            self.batch_size
            * self.config.num_kv_heads
            * self.seq_len
            * self.config.head_dim
            * 2
            * self.config.num_layers
        )
        return elements * bytes_per_element / 1024 / 1024

    def _reset_positions(self) -> None:
        self.seq_len = 0
        self.cache_position = 0
        self._update_id = 0
        self._last_eviction_update_id = -1

    def _maybe_evict(self, new_seq_len: int) -> None:
        """Evict old entries when cache would overflow."""
        end_pos = self.cache_position + new_seq_len
        if end_pos <= self.config.max_seq_len:
            return
        if self._last_eviction_update_id == self._update_id:
            return

        overflow = end_pos - self.config.max_seq_len
        self._shift_cache_left(overflow)
        self._last_eviction_update_id = self._update_id

    def _shift_cache_left(self, shift: int) -> None:
        if shift <= 0 or self.seq_len == 0:
            return
        new_len = max(self.seq_len - shift, 0)

        if self._layout == "BSHD":
            def _shift_list_bshd(tensors: list[torch.Tensor]) -> None:
                for tensor in tensors:
                    if new_len == 0:
                        continue
                    tensor[:, :new_len, :, :] = tensor[:,
                                                       shift: shift + new_len, :, :]

            _shift_list_bshd(self.k_cache)
            _shift_list_bshd(self.v_cache)
        else:
            def _shift_list_bhsd(tensors: list[torch.Tensor]) -> None:
                for tensor in tensors:
                    if new_len == 0:
                        continue
                    tensor[:, :, :new_len, :] = tensor[:,
                                                       :, shift: shift + new_len, :]

            _shift_list_bhsd(self.k_cache)
            _shift_list_bhsd(self.v_cache)
            if self.k_scales is not None:
                _shift_list_bhsd(self.k_scales)
            if self.v_scales is not None:
                _shift_list_bhsd(self.v_scales)

        self.seq_len = new_len
        self.cache_position = new_len

    def _slice_update(
        self,
        target: torch.Tensor,
        update: torch.Tensor,
        start: int,
        length: int,
        dtype_override: torch.dtype | None = None,
    ) -> None:
        """In-place slice update without reallocating the cache tensor."""
        if dtype_override is not None and update.dtype != dtype_override:
            update = update.to(dtype_override)
        elif update.dtype != target.dtype:
            update = update.to(target.dtype)
        if not update.is_contiguous():
            update = update.contiguous()
        target.narrow(2, start, length).copy_(update)

    def _slice_update_bshd(
        self,
        target: torch.Tensor,
        update: torch.Tensor,
        start: int,
        length: int,
        dtype_override: torch.dtype | None = None,
    ) -> None:
        """In-place slice update for BSHD layout."""
        if dtype_override is not None and update.dtype != dtype_override:
            update = update.to(dtype_override)
        elif update.dtype != target.dtype:
            update = update.to(target.dtype)
        if not update.is_contiguous():
            update = update.contiguous()
        target.narrow(1, start, length).copy_(update)

    def _quantize_fp4(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to FP4 packed format."""
        abs_max = tensor.abs().amax(dim=-1, keepdim=True)
        abs_max = torch.clamp(abs_max, min=1e-8)
        scale = abs_max / 6.0
        scaled = tensor / scale
        scaled = torch.clamp(scaled, -6.0, 6.0)
        quantized = torch.round(scaled * 2.0).to(torch.int8)
        quantized = torch.clamp(quantized + 8, 0, 15).to(torch.uint8)

        batch, heads, seq, dim = tensor.shape
        reshaped = quantized.view(batch, heads, seq, dim // 8, 8)

        packed = torch.zeros(
            (batch, heads, seq, dim // 8),
            dtype=torch.int32,
            device=tensor.device,
        )
        for i in range(8):
            packed = packed | (reshaped[..., i].to(torch.int32) << (i * 4))

        scale_dtype = _get_torch_dtype(self.dtype_config.scales)
        return packed, scale.to(scale_dtype)

    def _dequant_fp4(self, packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize FP4 packed tensor to float."""
        batch, heads, seq, packed_dim = packed.shape
        dim = packed_dim * 8

        unpacked_list = []
        for i in range(8):
            nibble = (packed >> (i * 4)) & 0xF
            signed = nibble.float() - 8.0
            unpacked_list.append(signed)

        unpacked = torch.stack(unpacked_list, dim=-1)
        unpacked = unpacked.view(batch, heads, seq, dim)
        return unpacked * scale.float() / 2.0

    def _quantize_fp8(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to FP8 E4M3 format."""
        FP8_E4M3_MAX = 448.0

        if self._fp8_scale_method == "channel":
            abs_val = tensor.abs()
            row_max = abs_val.amax(dim=-1, keepdim=True)
            scale = torch.clamp(abs_val, max=row_max) / FP8_E4M3_MAX
            scale = torch.clamp(scale, min=1e-12)
        else:
            abs_max = tensor.abs().amax(dim=-1, keepdim=True)
            abs_max = torch.clamp(abs_max, min=1e-8)
            scale = abs_max / FP8_E4M3_MAX

        scaled = tensor / scale
        scaled = torch.clamp(scaled, -FP8_E4M3_MAX, FP8_E4M3_MAX)
        quantized = torch.round(scaled / FP8_E4M3_MAX * 127.0) + 128.0
        quantized = torch.clamp(quantized, 0, 255).to(torch.uint8)

        scale_dtype = _get_torch_dtype(self.dtype_config.scales)
        return quantized, scale.to(scale_dtype)

    def _dequant_fp8(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize FP8 tensor to float."""
        FP8_E4M3_MAX = 448.0
        signed = quantized.float() - 128.0
        return signed / 127.0 * FP8_E4M3_MAX * scale.float()

    def _quantize_int8(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to INT8 format with symmetric quantization."""
        INT8_MAX = 127.0
        abs_max = tensor.abs().amax(dim=-1, keepdim=True)
        abs_max = torch.clamp(abs_max, min=1e-8)
        scale = abs_max / INT8_MAX
        scaled = tensor / scale
        scaled = torch.clamp(scaled, -INT8_MAX, INT8_MAX)
        quantized = torch.round(scaled).to(torch.int8)
        scale_dtype = _get_torch_dtype(self.dtype_config.scales)
        return quantized, scale.to(scale_dtype)

    def _dequant_int8(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize INT8 tensor to float."""
        return quantized.float() * scale.float()


# =============================================================================
# MLA KV Cache Implementation
# =============================================================================

_FP8_E4M3_MAX = 448.0


@dataclass
class MLAKVCache:
    """Unified KV cache for Multi-head Latent Attention (MLA).

    Caches the compressed KV representation (c_kv + k_pe) to reduce memory.
    Supports advanced features for long-context inference on Metal.

    Attributes:
        num_layers: Number of transformer layers.
        batch_size: Batch size for generation.
        max_seq_len: Initial maximum sequence length to allocate.
        kv_lora_rank: Compressed KV dimension (rank of latent space).
        qk_rope_head_dim: Dimension of rotary positional embedding.
        device: Device to store cache on (default: 'mps').
        dtype: Data type for cache tensors (default: float16).
        quantize_mode: Quantization mode - "none", "int8", or "fp8".
        fp8_scale_method: Scaling method for FP8/Int8 - "tensor" or "channel".
        auto_grow: Whether to automatically grow capacity when exceeded.
    """

    num_layers: int
    batch_size: int
    max_seq_len: int
    kv_lora_rank: int
    qk_rope_head_dim: int = 64
    device: str = "mps"
    dtype: torch.dtype | str = "float16"
    quantize_mode: Literal["none", "int8", "fp8"] = "none"
    fp8_scale_method: Literal["tensor", "channel"] = "tensor"
    auto_grow: bool = True

    def __post_init__(self) -> None:
        require_mps("MLA KV cache")
        self.dtype = _resolve_dtype(self.dtype)
        self.cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self._seq_lens: torch_typing.Tensor = torch.zeros(
            (self.num_layers, self.batch_size), dtype=torch.long, device=self.device
        )
        self._scale_dtype = torch.float16 if self.dtype == torch.float16 else torch.float32

        self._allocate(self.max_seq_len)

    def _allocate(self, max_seq_len: int) -> None:
        """Allocate cache tensors in BSHD layout."""
        if self.quantize_mode == "none":
            self.kv_cache = torch.zeros(
                (self.num_layers, self.batch_size, max_seq_len, self.cache_dim),
                dtype=self.dtype,
                device=self.device,
            )
            self.kv_scales = None
        elif self.quantize_mode == "int8":
            self.kv_cache = torch.zeros(
                (self.num_layers, self.batch_size, max_seq_len, self.cache_dim),
                dtype=torch.int8,
                device=self.device,
            )
            scale_dim = self.cache_dim if self.fp8_scale_method == "channel" else 1
            self.kv_scales = torch.ones(
                (self.num_layers, self.batch_size, max_seq_len, scale_dim),
                dtype=self._scale_dtype,
                device=self.device,
            )
        elif self.quantize_mode == "fp8":
            self.kv_cache = torch.zeros(
                (self.num_layers, self.batch_size, max_seq_len, self.cache_dim),
                dtype=torch.uint8,
                device=self.device,
            )
            scale_dim = self.cache_dim if self.fp8_scale_method == "channel" else 1
            self.kv_scales = torch.zeros(
                (self.num_layers, self.batch_size, max_seq_len, scale_dim),
                dtype=self._scale_dtype,
                device=self.device,
            )
        else:
            raise ValueError(
                f"Unsupported quantize_mode: {self.quantize_mode}")

    def _ensure_capacity(self, required_len: int) -> None:
        """Grow cache if needed."""
        if required_len <= self.max_seq_len:
            return

        if not self.auto_grow:
            raise ValueError(
                f"Sequence length {required_len} exceeds max_seq_len {self.max_seq_len}")

        new_max = max(required_len, max(1, self.max_seq_len * 2))
        old_max = self.max_seq_len

        old_kv_cache = self.kv_cache
        old_kv_scales = self.kv_scales

        self._allocate(new_max)

        self.kv_cache[:, :, :old_max, :] = old_kv_cache
        if self.kv_scales is not None and old_kv_scales is not None:
            self.kv_scales[:, :, :old_max, :] = old_kv_scales

        self.max_seq_len = new_max

    @property
    def seq_lens(self) -> list[int] | torch_typing.Tensor:
        """Sequence lengths per layer (first batch item for compatibility)."""
        return [int(v) for v in self._seq_lens[:, 0].tolist()]

    @property
    def seq_len(self) -> int:
        """Current sequence length across layers."""
        return int(self._seq_lens[:, 0].max().item())

    def get_seq_len(self) -> int:
        """Get current sequence length (for backward compatibility)."""
        return self.seq_len

    def update(
        self,
        layer_idx: int,
        c_kv_new: torch_typing.Tensor | None = None,
        k_pe_new: torch_typing.Tensor | None = None,
        compressed_kv: torch_typing.Tensor | None = None,
    ) -> torch_typing.Tensor | tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Update cache with new c_kv and k_pe and return full cached tensors.

        Supports both separate c_kv/k_pe inputs and unified compressed_kv input.
        If compressed_kv is provided, returns the full concatenated cache.
        Otherwise returns a tuple (c_kv_full, k_pe_full).
        """
        if compressed_kv is not None:
            # Unified update (TrellisKVCache style)
            return self.update_compressed(layer_idx, compressed_kv)

        if c_kv_new is None or k_pe_new is None:
            raise ValueError(
                "Must provide either (c_kv_new, k_pe_new) or compressed_kv")

        # Separate components update (MLAAttention style)
        # Concatenate for unified storage update
        combined_kv = torch.cat([c_kv_new, k_pe_new], dim=-1)
        full_kv = self.update_compressed(layer_idx, combined_kv)

        # Split back for return
        return full_kv[..., : self.kv_lora_rank], full_kv[..., self.kv_lora_rank:]

    def update_components(
        self,
        layer_idx: int,
        c_kv_new: torch_typing.Tensor,
        k_pe_new: torch_typing.Tensor,
    ) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Update cache components and return full dequantized components."""
        res = self.update(layer_idx, c_kv_new=c_kv_new, k_pe_new=k_pe_new)
        assert isinstance(res, tuple)
        return res

    def update_compressed(
        self,
        layer_idx: int,
        compressed_kv: torch_typing.Tensor,
    ) -> torch_typing.Tensor:
        """Update cache with combined compressed KV [B, S, Rank+RoPE]."""
        batch, seq_len, input_dim = compressed_kv.shape

        if input_dim != self.cache_dim:
            raise ValueError(
                f"Input dimension {input_dim} != cache_dim {self.cache_dim}")

        start = int(self._seq_lens[layer_idx, 0].item())
        end = start + seq_len
        self._ensure_capacity(end)

        if self.quantize_mode == "int8":
            kv_q, kv_scales = self._quantize_int8(compressed_kv)
            self.kv_cache[layer_idx, :batch, start:end] = kv_q
            # type: ignore[index]
            self.kv_scales[layer_idx, :batch, start:end] = kv_scales
        elif self.quantize_mode == "fp8":
            kv_q, kv_scales = self._quantize_fp8(compressed_kv)
            self.kv_cache[layer_idx, :batch, start:end] = kv_q
            # type: ignore[index]
            self.kv_scales[layer_idx, :batch, start:end] = kv_scales
        else:
            self.kv_cache[layer_idx, :batch,
                          start:end] = compressed_kv.to(self.dtype)

        self._seq_lens[layer_idx, :batch] = end

        # Retrieve full cache for this layer
        return self.get(layer_idx)  # type: ignore[return-value]

    def get(self, layer_idx: int) -> torch_typing.Tensor | None:
        """Retrieve full cached sequence for a layer."""
        seq_len = int(self._seq_lens[layer_idx, 0].item())
        if seq_len == 0:
            return None

        kv = self.kv_cache[layer_idx, :, :seq_len]

        if self.quantize_mode == "int8":
            kv = self._dequantize_int8(
                # type: ignore[index]
                kv, self.kv_scales[layer_idx, :, :seq_len])
        elif self.quantize_mode == "fp8":
            kv = self._dequantize_fp8(
                # type: ignore[index]
                kv, self.kv_scales[layer_idx, :, :seq_len])

        return kv.contiguous()

    def _quantize_int8(self, tensor: torch_typing.Tensor) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Int8 per-token or per-channel symmetric quantization."""
        dim = -1 if self.fp8_scale_method == "channel" else (-1,)
        abs_max = tensor.abs().amax(dim=dim, keepdim=True).clamp(min=1e-5)
        scale = abs_max / 127.0
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale.to(self._scale_dtype)

    def _dequantize_int8(self, quantized: torch_typing.Tensor, scales: torch_typing.Tensor) -> torch_typing.Tensor:
        return (quantized.to(self.dtype) * scales.to(self.dtype))

    def _quantize_fp8(self, tensor: torch_typing.Tensor) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        dim = -1 if self.fp8_scale_method == "channel" else (-1,)
        abs_max = tensor.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8)
        scale = abs_max / _FP8_E4M3_MAX
        scaled = tensor / scale
        scaled = torch.clamp(scaled, -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
        quantized = torch.round(scaled / _FP8_E4M3_MAX * 127.0) + 128.0
        quantized = torch.clamp(quantized, 0, 255).to(torch.uint8)
        return quantized, scale.to(self._scale_dtype)

    def _dequantize_fp8(self, quantized: torch_typing.Tensor, scale: torch_typing.Tensor) -> torch_typing.Tensor:
        signed = quantized.float() - 128.0
        return (signed / 127.0 * _FP8_E4M3_MAX * scale.float()).to(self.dtype)

    def reset(self) -> None:
        """Clear cache state."""
        self._seq_lens.zero_()

    def advance(self, num_tokens: int = 1) -> None:
        """No-op: MLAKVCache auto-tracks position via update_compressed().

        This method exists for interface compatibility with KVCacheTorch
        which requires explicit position advancement. MLAKVCache tracks
        _seq_lens per-layer in update_compressed(), so advance() is not needed.
        """

    def memory_usage_mb(self) -> float:
        """Return memory usage in MB."""
        active_seq = int(self._seq_lens[:, 0].max().item())
        if active_seq == 0:
            return 0.0

        if self.quantize_mode == "none":
            bpe = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
            scale_elements = 0
        else:
            bpe = 1
            scale_bpe = 2 if self._scale_dtype == torch.float16 else 4
            scale_dim = self.cache_dim if self.fp8_scale_method == "channel" else 1
            scale_elements = self.num_layers * self.batch_size * active_seq * scale_dim

        kv_elements = self.num_layers * self.batch_size * active_seq * self.cache_dim
        if self.quantize_mode == "none":
            return kv_elements * bpe / 1024 / 1024
        return (kv_elements * bpe + scale_elements * scale_bpe) / 1024 / 1024

    def prefetch_layer(self, layer_idx: int) -> torch_typing.Tensor | None:
        """Warm GPU caches for a layer."""
        kv = self.get(layer_idx)
        if kv is not None:
            return kv[0, 0, 0]
        return None

    def get_layer_slices(self, layer_idx: int) -> tuple[torch_typing.Tensor, torch_typing.Tensor] | None:
        """Get raw slice views split into c_kv and k_pe components."""
        kv = self.get(layer_idx)
        if kv is None:
            return None
        return kv[..., : self.kv_lora_rank], kv[..., self.kv_lora_rank:]

    def get_layer_for_attention(self, layer_idx: int) -> torch_typing.Tensor | None:
        """Get cached KV in BHSD format for attention kernels (zero-copy transpose)."""
        kv = self.get(layer_idx)
        if kv is None:
            return None
        # Transpose from BSHD [batch, seq_len, cache_dim] to BHSD [batch, cache_dim, seq_len]
        return kv.permute(0, 2, 1)

    def get_snapshot(self) -> dict[str, torch_typing.Tensor]:
        """Get a snapshot of the KV cache state for prompt caching."""
        snapshot = {
            "kv_cache": self.kv_cache.clone(),
            "seq_lens": self._seq_lens.clone(),
        }
        if self.kv_scales is not None:
            snapshot["kv_scales"] = self.kv_scales.clone()
        return snapshot

    def restore_snapshot(self, snapshot: dict[str, torch_typing.Tensor]) -> None:
        """Restore KV cache state from a snapshot."""
        self.kv_cache.copy_(snapshot["kv_cache"])
        self._seq_lens.copy_(snapshot["seq_lens"])
        if self.kv_scales is not None and "kv_scales" in snapshot:
            self.kv_scales.copy_(snapshot["kv_scales"])

    # Backwards compatibility properties
    @property
    def c_kv(self) -> torch_typing.Tensor:
        return self.kv_cache[..., : self.kv_lora_rank].contiguous()

    @property
    def k_pe(self) -> torch_typing.Tensor:
        return self.kv_cache[..., self.kv_lora_rank:].contiguous()

    @property
    def c_kv_scales(self) -> torch_typing.Tensor | None:
        return self.kv_scales

    @property
    def k_pe_scales(self) -> torch_typing.Tensor | None:
        return self.kv_scales


@dataclass
class CompressedKVCache(MLAKVCache):
    """Extended MLA cache with sliding window and late quantization."""

    compression: str = "int8"
    quantization_start_seq_len: int = 128
    sliding_window_layer_threshold: int = 32
    sliding_window_size: int = 4096

    def __post_init__(self) -> None:
        super().__post_init__()
        self._compression_enabled: list[bool] = [False] * self.num_layers
        self._window_start = torch.zeros(
            self.num_layers, dtype=torch.long, device=self.device)

    def _should_use_sliding_window(self, layer_idx: int) -> bool:
        return layer_idx >= self.sliding_window_layer_threshold

    def update_compressed(self, layer_idx: int, compressed_kv: torch_typing.Tensor) -> torch_typing.Tensor:
        # Simple override for now, can be extended with full logic from Trellis version
        return super().update_compressed(layer_idx, compressed_kv)


# Trellis-specific wrapper preserving legacy Trellis semantics.
class TrellisKVCache(MLAKVCache):
    """Trellis-oriented MLA cache behavior wrapper."""

    @property
    def seq_lens(self) -> torch_typing.Tensor:
        """Per-layer, per-batch sequence lengths tensor."""
        return self._seq_lens

    @property
    def seq_len(self) -> int:
        """Committed sequence length of the last layer."""
        return int(self._seq_lens[-1, 0].item())

    def memory_usage_mb(self) -> float:
        """Return allocated memory usage in MB (Trellis compatibility)."""
        if self.quantize_mode == "none":
            bpe = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
            scale_bpe = 0
        else:
            bpe = 1
            scale_bpe = 2 if self._scale_dtype == torch.float16 else 4

        kv_elements = self.num_layers * self.batch_size * \
            self.max_seq_len * self.cache_dim
        scale_elements = 0
        if self.kv_scales is not None:
            scale_elements = self.kv_scales.numel()

        return (kv_elements * bpe + scale_elements * scale_bpe) / 1024 / 1024


# Legacy aliases
KVCache = KVCacheTorch
CacheConfig = CacheConfigTorch

__all__ = [
    "KVCacheTorch",
    "CacheConfigTorch",

    "KVCache",
    "CacheConfig",

    "MLAKVCache",
    "CompressedKVCache",
    "TrellisKVCache",

    "require_mps",
]
