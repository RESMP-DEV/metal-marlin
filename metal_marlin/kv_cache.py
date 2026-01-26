"""
KV Cache management for efficient autoregressive inference.

Stores key/value tensors from previous tokens to avoid recomputation during
decode phase. Supports configurable precision via DTypeConfig:
- Full precision: BF16 (default) or FP16
- Quantized: FP8 for 2x memory savings on long context
- FP4 for maximum compression (4x savings)

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
    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False
    torch = None

if TYPE_CHECKING:
    import torch


def require_mps(feature: str = "KV cache") -> None:
    """Raise if PyTorch MPS is not available."""
    if not HAS_TORCH:
        raise RuntimeError(f"{feature} requires PyTorch. Install with: pip install torch")
    if not HAS_MPS:
        raise RuntimeError(
            f"{feature} requires PyTorch MPS backend. "
            "Ensure you're on Apple Silicon with PyTorch >= 2.0"
        )


# Mapping from dtype config strings to PyTorch dtypes
_DTYPE_TO_TORCH: dict[str, torch.dtype] = {}
if HAS_TORCH:
    _DTYPE_TO_TORCH = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp8": torch.float16,  # PyTorch doesn't have native fp8, use fp16 storage
    }


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to PyTorch dtype."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available")
    return _DTYPE_TO_TORCH[dtype_str]


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
        cache_dtype: Cache storage dtype (takes precedence over quantize_mode).
            - "fp16": Half precision storage
            - "bf16": BFloat16 storage
            - "fp8": FP8 E4M3 quantized (2x memory savings)
            - "fp4": FP4 E2M1 quantized (4x memory savings)
        fp8_scale_method: Scaling method for FP8 quantization.
            - "tensor": Single scale per tensor (faster, lower memory)
            - "channel": Scale per head dimension (better accuracy for outliers)
    """

    num_layers: int
    num_heads: int
    num_kv_heads: int  # For Grouped Query Attention (GQA)
    head_dim: int
    max_seq_len: int
    quantize_mode: Literal["none", "fp8", "fp4"] = "none"
    cache_dtype: Literal["fp16", "bf16", "fp8", "fp4"] | None = None
    fp8_scale_method: Literal["tensor", "channel"] = "tensor"


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
        self.dtype_config = dtype_config if dtype_config is not None else get_default_config()
        self.device = device
        self._fp8_scale_method = config.fp8_scale_method

        # Allocate cache for all layers
        # Shape: [batch, num_kv_heads, max_seq_len, head_dim]
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
        else:
            # Full precision cache using cache_dtype override or dtype_config.kv_cache
            storage_dtype_str = getattr(self, "_storage_dtype_override", None) or self.dtype_config.kv_cache
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
        end_pos = self.seq_len + new_seq_len

        if end_pos > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {end_pos} exceeds max_seq_len {self.config.max_seq_len}"
            )

        act_dtype = _get_torch_dtype(self.dtype_config.activations)

        if self._quantize_mode == "fp4":
            # Quantize and store as FP4
            k_packed, k_scale = self._quantize_fp4(k_new)
            v_packed, v_scale = self._quantize_fp4(v_new)

            # Update cache slices (in-place for efficiency)
            self.k_cache[layer_idx][:, :, self.seq_len : end_pos, :] = k_packed
            self.v_cache[layer_idx][:, :, self.seq_len : end_pos, :] = v_packed
            self.k_scales[layer_idx][:, :, self.seq_len : end_pos, :] = k_scale
            self.v_scales[layer_idx][:, :, self.seq_len : end_pos, :] = v_scale

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
            self.k_cache[layer_idx][:, :, self.seq_len : end_pos, :] = k_quant
            self.v_cache[layer_idx][:, :, self.seq_len : end_pos, :] = v_quant
            self.k_scales[layer_idx][:, :, self.seq_len : end_pos, :] = k_scale
            self.v_scales[layer_idx][:, :, self.seq_len : end_pos, :] = v_scale

            # Dequantize full cache for attention (output in activations dtype)
            k_full = self._dequant_fp8(
                self.k_cache[layer_idx][:, :, :end_pos, :],
                self.k_scales[layer_idx][:, :, :end_pos, :],
            ).to(act_dtype)
            v_full = self._dequant_fp8(
                self.v_cache[layer_idx][:, :, :end_pos, :],
                self.v_scales[layer_idx][:, :, :end_pos, :],
            ).to(act_dtype)

        else:
            # Direct storage using kv_cache dtype
            storage_dtype = _get_torch_dtype(self.dtype_config.kv_cache)
            self.k_cache[layer_idx][:, :, self.seq_len : end_pos, :] = k_new.to(storage_dtype)
            self.v_cache[layer_idx][:, :, self.seq_len : end_pos, :] = v_new.to(storage_dtype)

            # Return in activations dtype for attention computation
            k_full = self.k_cache[layer_idx][:, :, :end_pos, :].to(act_dtype)
            v_full = self.v_cache[layer_idx][:, :, :end_pos, :].to(act_dtype)

        return k_full, v_full

    def advance(self, num_tokens: int = 1) -> None:
        """Advance sequence position after processing tokens."""
        self.seq_len += num_tokens

    def reset(self) -> None:
        """Clear cache for new sequence."""
        self.seq_len = 0

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
        else:
            k = self.k_cache[layer_idx][:, :, : self.seq_len, :].to(act_dtype)
            v = self.v_cache[layer_idx][:, :, : self.seq_len, :].to(act_dtype)

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
            abs_max = torch.clamp(abs_val, min=1e-8)  # Per-element, clamped to avoid div by zero
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
