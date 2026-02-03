"""
KV Cache management for efficient autoregressive inference.

Stores key/value tensors from previous tokens to avoid recomputation during
decode phase. Supports configurable precision via DTypeConfig:
- Full precision: BF16 (default) or FP16
- Quantized: FP8 for 2x memory savings on long context
- FP4 for maximum compression (4x savings)
- INT8 for 2x memory savings with symmetric quantization

Uses PyTorch MPS backend with direct Metal kernel dispatch for Apple Silicon
acceleration. No MLX dependency.

Usage:
    from metal_marlin.kv_cache import KVCache, CacheConfig
    from metal_marlin.dtypes import DTypeConfig, memory_efficient_config

    config = CacheConfig(
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,  # GQA
        head_dim=128,
        max_seq_len=4096,
    )

    # Standard BF16 cache
    cache = KVCache(config, batch_size=1)

    # FP8 cache for long context
    cache = KVCache(config, batch_size=1, dtype_config=memory_efficient_config())

    # Mixed: BF16 activations with FP8 KV cache
    mixed_config = DTypeConfig(activations="bf16", kv_cache="fp8")
    cache = KVCache(config, batch_size=1, dtype_config=mixed_config)

    # During forward pass
    k_full, v_full = cache.update(layer_idx=0, k_new=k, v_new=v)
    cache.advance(num_tokens=seq_len)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .dtypes import DTypeConfig, get_default_config

# PyTorch MPS import with availability check
try:
    import torch

    HAS_TORCH = True
    HAS_MPS = hasattr(torch, "backends") and torch.backends.mps.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False
    torch = None

if TYPE_CHECKING:
    import torch


def require_mps(feature: str = "KV cache") -> None:
    """Raise if PyTorch MPS is not available."""
    if not HAS_TORCH:
        raise RuntimeError(
            f"{feature} requires PyTorch. Install with: pip install torch")
    if not HAS_MPS:
        raise RuntimeError(
            f"{feature} requires PyTorch MPS backend. "
            "Ensure you're on Apple Silicon with PyTorch >= 2.0"
        )


# Mapping from dtype config strings to PyTorch dtypes
# Use Any for the value type at runtime to avoid AttributeError when torch isn't fully initialized.
# The actual type is torch.dtype but we defer that to TYPE_CHECKING.
if TYPE_CHECKING:
    _DTYPE_TO_TORCH_CACHE: dict[str, torch.dtype] = {}
else:
    _DTYPE_TO_TORCH_CACHE: dict[str, object] = {}


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to PyTorch dtype."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available")

    if not _DTYPE_TO_TORCH_CACHE:
        # Lazy initialization to handle any weird import ordering issues
        _DTYPE_TO_TORCH_CACHE.update(
            {
                "fp16": getattr(torch, "float16", torch.float32),
                "bf16": getattr(torch, "bfloat16", torch.float32),
                "fp32": getattr(torch, "float32", None),
                "fp8": getattr(torch, "float16", torch.float32),
            }
        )

    if dtype_str not in _DTYPE_TO_TORCH_CACHE:
        raise ValueError(f"Unknown dtype: {dtype_str}")

    dtype = _DTYPE_TO_TORCH_CACHE[dtype_str]
    if dtype is None:
        raise RuntimeError(
            f"Torch dtype {dtype_str} not available in this environment")

    return dtype


@dataclass
class CacheConfig:
    """Configuration for KV cache.

    Attributes:
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        num_kv_heads: Number of key-value heads (for GQA).
        head_dim: Dimension per head.
        max_seq_len: Maximum sequence length to allocate.
        quantize_mode: Cache quantization mode (legacy, prefer cache_dtype).
            - "none": Full precision (uses dtype_config.kv_cache)
            - "fp8": 8-bit floating point (2x memory savings)
            - "fp4": 4-bit floating point (4x memory savings)
            - "int8": 8-bit integer symmetric quantization (2x memory savings)
        cache_dtype: Cache storage dtype (takes precedence over quantize_mode).
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
            BSHD provides ~3x faster single-token writes at decode time because
            all heads for a token are contiguous, enabling coalesced memory access.
            The attention API always returns [B,H,S,D] format; BSHD caches use
            a zero-copy view transpose internally.
    """

    num_layers: int
    num_heads: int
    num_kv_heads: int  # For Grouped Query Attention (GQA)
    head_dim: int
    max_seq_len: int
    quantize_mode: Literal["none", "fp8", "fp4", "int8"] = "none"
    cache_dtype: Literal["fp16", "bf16", "fp8", "fp4", "int8"] | None = None
    fp8_scale_method: Literal["tensor", "channel"] = "tensor"
    layout: Literal["BHSD", "BSHD"] = "BHSD"


class KVCache:
    """
    Key-Value cache for transformer inference using PyTorch MPS.

    Supports:
    - Standard MHA (num_heads == num_kv_heads)
    - Grouped Query Attention (num_kv_heads < num_heads)
    - Configurable precision via DTypeConfig:
        - BF16 (default): Better dynamic range, recommended for most cases
        - FP16: Maximum compatibility
        - FP8: 2x memory savings for long context
        - FP4: 4x memory savings for very long context

    The cache pre-allocates tensors for max_seq_len to avoid reallocation
    during generation. Only the slice up to current seq_len is used.
    """

    def __init__(
        self,
        config: CacheConfig,
        batch_size: int = 1,
        dtype_config: DTypeConfig | None = None,
        device: str = "mps",
    ):
        require_mps("KV cache")

        self.config = config
        self.batch_size = batch_size
        self.seq_len = 0  # Current sequence length
        self.cache_position = 0  # Next write position (tracks seq_len)
        self.dtype_config = dtype_config if dtype_config is not None else get_default_config()
        self.device = device
        self._fp8_scale_method = config.fp8_scale_method
        self._update_id = 0
        self._last_eviction_update_id = -1
        self._layout = config.layout

        # Allocate cache for all layers
        # Layout determines memory arrangement:
        #   BHSD: [batch, num_kv_heads, max_seq_len, head_dim] - standard
        #   BSHD: [batch, max_seq_len, num_kv_heads, head_dim] - faster decode writes
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

        # Determine storage mode: cache_dtype takes precedence over quantize_mode
        if config.cache_dtype is not None:
            # New API: use cache_dtype directly
            if config.cache_dtype == "fp8":
                self._quantize_mode = "fp8"
            elif config.cache_dtype == "fp4":
                self._quantize_mode = "fp4"
            elif config.cache_dtype == "int8":
                self._quantize_mode = "int8"
            elif config.cache_dtype in ("fp16", "bf16"):
                self._quantize_mode = "none"
                # Override dtype_config's kv_cache setting
                self._storage_dtype_override = config.cache_dtype
            else:
                self._quantize_mode = "none"
                self._storage_dtype_override = None
        else:
            # Legacy API: use quantize_mode
            self._quantize_mode = config.quantize_mode
            self._storage_dtype_override = None

        if self._quantize_mode == "fp4":
            # FP4 quantized cache - pack 8 values per int32
            # Note: FP4 only supports BHSD layout currently
            if self._layout == "BSHD":
                raise ValueError(
                    "FP4 quantization only supports BHSD layout currently")
            packed_shape = (
                batch_size,
                config.num_kv_heads,
                config.max_seq_len,
                config.head_dim // 8,  # 8 FP4 values per int32
            )
            self.k_cache: list[torch.Tensor] = [
                torch.zeros(packed_shape, dtype=torch.int32, device=device)
                for _ in range(config.num_layers)
            ]
            self.v_cache: list[torch.Tensor] = [
                torch.zeros(packed_shape, dtype=torch.int32, device=device)
                for _ in range(config.num_layers)
            ]
            # Per-row scales for dequantization
            scale_dtype = _get_torch_dtype(self.dtype_config.scales)
            scale_shape = (
                batch_size,
                config.num_kv_heads,
                config.max_seq_len,
                1,
            )
            self.k_scales: list[torch.Tensor] | None = [
                torch.zeros(scale_shape, dtype=scale_dtype, device=device)
                for _ in range(config.num_layers)
            ]
            self.v_scales: list[torch.Tensor] | None = [
                torch.zeros(scale_shape, dtype=scale_dtype, device=device)
                for _ in range(config.num_layers)
            ]
        elif self._quantize_mode == "fp8":
            # FP8 E4M3 quantized cache - store as uint8 with per-tensor or per-channel scales
            # PyTorch doesn't have native fp8 on MPS, so we simulate with uint8 + scales
            # Note: FP8 only supports BHSD layout currently
            if self._layout == "BSHD":
                raise ValueError(
                    "FP8 quantization only supports BHSD layout currently")
            fp8_shape = (
                batch_size,
                config.num_kv_heads,
                config.max_seq_len,
                config.head_dim,
            )
            self.k_cache = [
                torch.zeros(fp8_shape, dtype=torch.uint8, device=device)
                for _ in range(config.num_layers)
            ]
            self.v_cache = [
                torch.zeros(fp8_shape, dtype=torch.uint8, device=device)
                for _ in range(config.num_layers)
            ]
            # Scales for FP8 dequantization: tensor vs channel scaling
            scale_dtype = _get_torch_dtype(self.dtype_config.scales)
            if config.fp8_scale_method == "channel":
                # Per-channel (head_dim) scaling for better outlier handling
                scale_shape = (
                    batch_size,
                    config.num_kv_heads,
                    config.max_seq_len,
                    config.head_dim,
                )
            else:
                # Per-tensor (per-row) scaling: single scale per sequence position
                scale_shape = (
                    batch_size,
                    config.num_kv_heads,
                    config.max_seq_len,
                    1,
                )
            self.k_scales = [
                torch.zeros(scale_shape, dtype=scale_dtype, device=device)
                for _ in range(config.num_layers)
            ]
            self.v_scales = [
                torch.zeros(scale_shape, dtype=scale_dtype, device=device)
                for _ in range(config.num_layers)
            ]
        elif self._quantize_mode == "int8":
            # INT8 symmetric quantized cache - store as int8 with per-tensor scales
            # Note: INT8 only supports BHSD layout currently
            if self._layout == "BSHD":
                raise ValueError(
                    "INT8 quantization only supports BHSD layout currently")
            int8_shape = (
                batch_size,
                config.num_kv_heads,
                config.max_seq_len,
                config.head_dim,
            )
            self.k_cache = [
                torch.zeros(int8_shape, dtype=torch.int8, device=device)
                for _ in range(config.num_layers)
            ]
            self.v_cache = [
                torch.zeros(int8_shape, dtype=torch.int8, device=device)
                for _ in range(config.num_layers)
            ]
            # Scales for INT8 dequantization: per-tensor (per-row) scaling
            scale_dtype = _get_torch_dtype(self.dtype_config.scales)
            scale_shape = (
                batch_size,
                config.num_kv_heads,
                config.max_seq_len,
                1,
            )
            self.k_scales = [
                torch.zeros(scale_shape, dtype=scale_dtype, device=device)
                for _ in range(config.num_layers)
            ]
            self.v_scales = [
                torch.zeros(scale_shape, dtype=scale_dtype, device=device)
                for _ in range(config.num_layers)
            ]
        else:
            # Full precision cache using cache_dtype override or dtype_config.kv_cache
            storage_dtype_str = (
                getattr(self, "_storage_dtype_override",
                        None) or self.dtype_config.kv_cache
            )
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
        k_new: torch.Tensor,  # [batch, num_kv_heads, new_seq_len, head_dim]
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new K, V and return full cached K, V.

        Args:
            layer_idx: Which transformer layer
            k_new: New key tensor [batch, num_kv_heads, new_seq_len, head_dim]
            v_new: New value tensor [batch, num_kv_heads, new_seq_len, head_dim]

        Returns:
            (k_full, v_full) with shape [batch, num_kv_heads, seq_len + new_seq_len, head_dim]
            Output dtype is activations dtype from dtype_config (for attention computation).
        """
        new_seq_len = k_new.shape[2]
        if new_seq_len > self.config.max_seq_len:
            # Keep the most recent window when a single update exceeds capacity.
            k_new = k_new[:, :, -self.config.max_seq_len:, :]
            v_new = v_new[:, :, -self.config.max_seq_len:, :]
            new_seq_len = self.config.max_seq_len
            self._reset_positions()

        self._maybe_evict(new_seq_len)

        start_pos = self.cache_position
        end_pos = start_pos + new_seq_len

        act_dtype = _get_torch_dtype(self.dtype_config.activations)

        if self._quantize_mode == "fp4":
            # Quantize and store as FP4
            k_packed, k_scale = self._quantize_fp4(k_new)
            v_packed, v_scale = self._quantize_fp4(v_new)

            # Update cache slices in-place (MPS uses sliceUpdateDataTensor under the hood).
            self._slice_update(
                self.k_cache[layer_idx], k_packed, start_pos, new_seq_len)
            self._slice_update(
                self.v_cache[layer_idx], v_packed, start_pos, new_seq_len)
            self._slice_update(
                self.k_scales[layer_idx], k_scale, start_pos, new_seq_len)
            self._slice_update(
                self.v_scales[layer_idx], v_scale, start_pos, new_seq_len)

            # Dequantize full cache for attention (output in activations dtype)
            k_full = self._dequant_fp4(
                self.k_cache[layer_idx][:, :, :end_pos, :],
                self.k_scales[layer_idx][:, :, :end_pos, :],
            ).to(act_dtype)
            v_full = self._dequant_fp4(
                self.v_cache[layer_idx][:, :, :end_pos, :],
                self.v_scales[layer_idx][:, :, :end_pos, :],
            ).to(act_dtype)

        elif self._quantize_mode == "fp8":
            # Quantize and store as FP8
            k_quant, k_scale = self._quantize_fp8(k_new)
            v_quant, v_scale = self._quantize_fp8(v_new)

            # Update cache slices (in-place)
            self._slice_update(
                self.k_cache[layer_idx], k_quant, start_pos, new_seq_len)
            self._slice_update(
                self.v_cache[layer_idx], v_quant, start_pos, new_seq_len)
            self._slice_update(
                self.k_scales[layer_idx], k_scale, start_pos, new_seq_len)
            self._slice_update(
                self.v_scales[layer_idx], v_scale, start_pos, new_seq_len)

            # Dequantize full cache for attention (output in activations dtype)
            k_full = self._dequant_fp8(
                self.k_cache[layer_idx][:, :, :end_pos, :],
                self.k_scales[layer_idx][:, :, :end_pos, :],
            ).to(act_dtype)
            v_full = self._dequant_fp8(
                self.v_cache[layer_idx][:, :, :end_pos, :],
                self.v_scales[layer_idx][:, :, :end_pos, :],
            ).to(act_dtype)

        elif self._quantize_mode == "int8":
            # Quantize and store as INT8
            k_quant, k_scale = self._quantize_int8(k_new)
            v_quant, v_scale = self._quantize_int8(v_new)

            # Update cache slices (in-place)
            self._slice_update(
                self.k_cache[layer_idx], k_quant, start_pos, new_seq_len)
            self._slice_update(
                self.v_cache[layer_idx], v_quant, start_pos, new_seq_len)
            self._slice_update(
                self.k_scales[layer_idx], k_scale, start_pos, new_seq_len)
            self._slice_update(
                self.v_scales[layer_idx], v_scale, start_pos, new_seq_len)

            # Dequantize full cache for attention (output in activations dtype)
            k_full = self._dequant_int8(
                self.k_cache[layer_idx][:, :, :end_pos, :],
                self.k_scales[layer_idx][:, :, :end_pos, :],
            ).to(act_dtype)
            v_full = self._dequant_int8(
                self.v_cache[layer_idx][:, :, :end_pos, :],
                self.v_scales[layer_idx][:, :, :end_pos, :],
            ).to(act_dtype)

        else:
            # Direct storage using kv_cache dtype
            storage_dtype_str = (
                getattr(self, "_storage_dtype_override",
                        None) or self.dtype_config.kv_cache
            )
            storage_dtype = _get_torch_dtype(storage_dtype_str)

            if self._layout == "BSHD":
                # BSHD layout: [batch, seq, heads, dim]
                # Transpose input from [B, H, S, D] to [B, S, H, D] for coalesced writes
                k_transposed = k_new.permute(0, 2, 1, 3)
                v_transposed = v_new.permute(0, 2, 1, 3)
                self._slice_update_bshd(
                    self.k_cache[layer_idx], k_transposed, start_pos, new_seq_len, storage_dtype
                )
                self._slice_update_bshd(
                    self.v_cache[layer_idx], v_transposed, start_pos, new_seq_len, storage_dtype
                )

                # Return view-transposed back to [B, H, S, D] for attention (zero-copy)
                k_full = self.k_cache[layer_idx][:, :end_pos, :, :].permute(
                    0, 2, 1, 3).to(act_dtype)
                v_full = self.v_cache[layer_idx][:, :end_pos, :, :].permute(
                    0, 2, 1, 3).to(act_dtype)
            else:
                # BHSD layout: [batch, heads, seq, dim] (default)
                self._slice_update(
                    self.k_cache[layer_idx], k_new, start_pos, new_seq_len, storage_dtype
                )
                self._slice_update(
                    self.v_cache[layer_idx], v_new, start_pos, new_seq_len, storage_dtype
                )

                # Return in activations dtype for attention computation
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
        """Get current cached K, V for a layer (dequantized if needed).

        Returns tensors in activations dtype from dtype_config.
        """
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
                # BSHD layout: [batch, seq, heads, dim] -> [batch, heads, seq, dim]
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
            # FP4 packed (0.5 bytes) + scales (2 or 4 bytes per row depending on config)
            scale_bytes = 4.0 if self.dtype_config.scales == "fp32" else 2.0
            bytes_per_element = 0.5 + scale_bytes / self.config.head_dim
        elif self._quantize_mode == "fp8":
            # FP8 (1 byte) + scales (2 or 4 bytes per row)
            scale_bytes = 4.0 if self.dtype_config.scales == "fp32" else 2.0
            bytes_per_element = 1.0 + scale_bytes / self.config.head_dim
        elif self._quantize_mode == "int8":
            # INT8 (1 byte) + scales (2 or 4 bytes per row)
            scale_bytes = 4.0 if self.dtype_config.scales == "fp32" else 2.0
            bytes_per_element = 1.0 + scale_bytes / self.config.head_dim
        else:
            # Full precision: 2 bytes for fp16, bf16 (bf16 also stored as 2 bytes)
            bytes_per_element = 2

        elements = (
            self.batch_size
            * self.config.num_kv_heads
            * self.seq_len
            * self.config.head_dim
            * 2  # K and V
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
            # BSHD layout: seq dimension is at index 1
            def _shift_list_bshd(tensors: list[torch.Tensor]) -> None:
                for tensor in tensors:
                    if new_len == 0:
                        continue
                    tensor[:, :new_len, :, :] = tensor[:,
                                                       shift: shift + new_len, :, :]

            _shift_list_bshd(self.k_cache)
            _shift_list_bshd(self.v_cache)
        else:
            # BHSD layout: seq dimension is at index 2
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

    def _quantize_fp4(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to FP4 packed format."""
        # Find max absolute value per row for scaling
        abs_max = tensor.abs().amax(dim=-1, keepdim=True)
        abs_max = torch.clamp(abs_max, min=1e-8)  # Avoid division by zero

        # Scale to [-1, 1] range
        # FP4 E2M1 max value is 6.0 (1.5 * 2^2)
        scale = abs_max / 6.0
        scaled = tensor / scale

        # Clamp to FP4 range
        scaled = torch.clamp(scaled, -6.0, 6.0)

        # Quantize to 4-bit (scale to use integer range 0-15)
        quantized = torch.round(scaled * 2.0).to(torch.int8)
        quantized = torch.clamp(quantized + 8, 0, 15).to(torch.uint8)

        # Pack 8 values per int32
        batch, heads, seq, dim = tensor.shape
        reshaped = quantized.view(batch, heads, seq, dim // 8, 8)

        # Pack using bit shifts
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

        # Unpack 8 values from each int32
        unpacked_list = []
        for i in range(8):
            nibble = (packed >> (i * 4)) & 0xF
            # Convert 4-bit unsigned to signed centered at 8
            signed = nibble.float() - 8.0
            # Scale back and apply per-row scale
            unpacked_list.append(signed)

        # Stack and reshape
        unpacked = torch.stack(unpacked_list, dim=-1)  # [..., packed_dim, 8]
        unpacked = unpacked.view(batch, heads, seq, dim)  # [..., dim]

        # Apply scale and undo the 2.0 scaling
        return unpacked * scale.float() / 2.0

    def _quantize_fp8(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to FP8 E4M3 format (simulated with uint8 + scale).

        Uses E4M3 representation: 4 exponent bits, 3 mantissa bits.
        - Dynamic range: 2^-6 to 448 (positive), with symmetric negative range
        - Max representable value: 448 (1.75 * 2^8)
        - Min representable normal: ~0.015625 (2^-6)
        - Preserves outliers better than INT8 for attention patterns

        Supports two scaling methods:
        - tensor: Single scale per row (last dim reduced) - faster, lower memory
        - channel: Per-element scale along head_dim - better accuracy for outliers
        """
        # FP8 E4M3 max value is 448 (sign bit + 4 exponent + 3 mantissa)
        FP8_E4M3_MAX = 448.0

        if self._fp8_scale_method == "channel":
            # Per-channel scaling: compute scale for each element in head_dim
            # This preserves outliers better but uses more memory for scales
            # Scale shape: [batch, heads, seq, head_dim]
            abs_val = tensor.abs()
            # Per-element, clamped to avoid div by zero
            abs_max = torch.clamp(abs_val, min=1e-8)
            # Use running max smoothed with the tensor's max to handle outliers
            row_max = abs_val.amax(dim=-1, keepdim=True)
            # Scale each channel independently but bounded by row max for stability
            scale = torch.clamp(abs_max, max=row_max) / FP8_E4M3_MAX
            scale = torch.clamp(scale, min=1e-12)  # Ensure non-zero scale
        else:
            # Per-tensor (per-row) scaling: single scale per sequence position
            # Scale shape: [batch, heads, seq, 1]
            abs_max = tensor.abs().amax(dim=-1, keepdim=True)
            abs_max = torch.clamp(abs_max, min=1e-8)  # Avoid division by zero
            scale = abs_max / FP8_E4M3_MAX

        # Scale to E4M3 range
        scaled = tensor / scale

        # Clamp to E4M3 symmetric range
        scaled = torch.clamp(scaled, -FP8_E4M3_MAX, FP8_E4M3_MAX)

        # Convert to uint8 with offset (128-centered for signed range)
        # Map [-448, 448] to [0, 255] with 128 as zero point
        quantized = torch.round(scaled / FP8_E4M3_MAX * 127.0) + 128.0
        quantized = torch.clamp(quantized, 0, 255).to(torch.uint8)

        scale_dtype = _get_torch_dtype(self.dtype_config.scales)
        return quantized, scale.to(scale_dtype)

    def _dequant_fp8(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize FP8 tensor to float.

        Supports both tensor and channel scaling methods - the scale tensor shape
        determines which method was used during quantization.
        """
        FP8_E4M3_MAX = 448.0

        # Reverse the quantization: uint8 centered at 128 -> signed -> scaled
        # Map [0, 255] back to [-448, 448] then apply scale
        signed = quantized.float() - 128.0
        # signed is in range [-128, 127], map to [-448, 448]
        dequant = signed / 127.0 * FP8_E4M3_MAX * scale.float()
        return dequant

    def _quantize_int8(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to INT8 format with symmetric quantization.

        Uses symmetric quantization mapping float range to [-127, 127]:
        - Dynamic range: -127 to 127 (signed 8-bit integer)
        - Per-tensor (per-row) scaling: single scale per sequence position
        - Simple and efficient for KV cache compression (2x memory savings)
        - Zero point at 0 (symmetric around zero)
        """
        INT8_MAX = 127.0

        # Per-tensor (per-row) scaling: single scale per sequence position
        # Scale shape: [batch, heads, seq, 1]
        abs_max = tensor.abs().amax(dim=-1, keepdim=True)
        abs_max = torch.clamp(abs_max, min=1e-8)  # Avoid division by zero
        scale = abs_max / INT8_MAX

        # Scale to INT8 range
        scaled = tensor / scale

        # Clamp to INT8 symmetric range and round
        scaled = torch.clamp(scaled, -INT8_MAX, INT8_MAX)
        quantized = torch.round(scaled).to(torch.int8)

        scale_dtype = _get_torch_dtype(self.dtype_config.scales)
        return quantized, scale.to(scale_dtype)

    def _dequant_int8(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize INT8 tensor to float.

        Reverses symmetric quantization by converting int8 back to float
        and applying the per-tensor scale factor.
        """
        # Convert int8 to float and apply scale
        dequant = quantized.float() * scale.float()
        return dequant
