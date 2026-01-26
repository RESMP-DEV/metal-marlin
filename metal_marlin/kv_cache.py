"""
KV Cache management for efficient autoregressive inference.

Stores key/value tensors from previous tokens to avoid recomputation during
decode phase. Supports configurable precision via DTypeConfig:
- Full precision: BF16 (default) or FP16
- Quantized: FP8 for 2x memory savings on long context
- FP4 for maximum compression (4x savings)

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
from typing import Literal

import mlx.core as mx

from .dtypes import DTypeConfig, get_default_config


@dataclass
class CacheConfig:
    """Configuration for KV cache.

    Attributes:
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        num_kv_heads: Number of key-value heads (for GQA).
        head_dim: Dimension per head.
        max_seq_len: Maximum sequence length to allocate.
        quantize_mode: Cache quantization mode.
            - "none": Full precision (uses dtype_config.kv_cache)
            - "fp8": 8-bit floating point (2x memory savings)
            - "fp4": 4-bit floating point (4x memory savings)
    """

    num_layers: int
    num_heads: int
    num_kv_heads: int  # For Grouped Query Attention (GQA)
    head_dim: int
    max_seq_len: int
    quantize_mode: Literal["none", "fp8", "fp4"] = "none"


class KVCache:
    """
    Key-Value cache for transformer inference.

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
    ):
        self.config = config
        self.batch_size = batch_size
        self.seq_len = 0  # Current sequence length
        self.dtype_config = dtype_config if dtype_config is not None else get_default_config()

        # Allocate cache for all layers
        # Shape: [batch, num_kv_heads, max_seq_len, head_dim]
        cache_shape = (
            batch_size,
            config.num_kv_heads,
            config.max_seq_len,
            config.head_dim,
        )

        # Determine storage mode based on quantize_mode
        self._quantize_mode = config.quantize_mode

        if config.quantize_mode == "fp4":
            # FP4 quantized cache - pack 8 values per uint32
            packed_shape = (
                batch_size,
                config.num_kv_heads,
                config.max_seq_len,
                config.head_dim // 8,  # 8 FP4 values per uint32
            )
            self.k_cache = [
                mx.zeros(packed_shape, dtype=mx.uint32) for _ in range(config.num_layers)
            ]
            self.v_cache = [
                mx.zeros(packed_shape, dtype=mx.uint32) for _ in range(config.num_layers)
            ]
            # Per-row scales for dequantization (use scales dtype from config)
            scale_shape = (
                batch_size,
                config.num_kv_heads,
                config.max_seq_len,
                1,
            )
            self.k_scales = [
                mx.zeros(scale_shape, dtype=self.dtype_config.mlx_scales) for _ in range(config.num_layers)
            ]
            self.v_scales = [
                mx.zeros(scale_shape, dtype=self.dtype_config.mlx_scales) for _ in range(config.num_layers)
            ]
        elif config.quantize_mode == "fp8":
            # FP8 quantized cache - store as uint8 with per-row scales
            # MLX doesn't have native fp8, so we simulate with uint8 + scales
            fp8_shape = (
                batch_size,
                config.num_kv_heads,
                config.max_seq_len,
                config.head_dim,
            )
            self.k_cache = [
                mx.zeros(fp8_shape, dtype=mx.uint8) for _ in range(config.num_layers)
            ]
            self.v_cache = [
                mx.zeros(fp8_shape, dtype=mx.uint8) for _ in range(config.num_layers)
            ]
            # Per-row scales for FP8 dequantization
            scale_shape = (
                batch_size,
                config.num_kv_heads,
                config.max_seq_len,
                1,
            )
            self.k_scales = [
                mx.zeros(scale_shape, dtype=self.dtype_config.mlx_scales) for _ in range(config.num_layers)
            ]
            self.v_scales = [
                mx.zeros(scale_shape, dtype=self.dtype_config.mlx_scales) for _ in range(config.num_layers)
            ]
        else:
            # Full precision cache using kv_cache dtype from config
            self.k_cache = [
                mx.zeros(cache_shape, dtype=self.dtype_config.mlx_kv_cache) for _ in range(config.num_layers)
            ]
            self.v_cache = [
                mx.zeros(cache_shape, dtype=self.dtype_config.mlx_kv_cache) for _ in range(config.num_layers)
            ]
            self.k_scales = None
            self.v_scales = None

    def update(
        self,
        layer_idx: int,
        k_new: mx.array,  # [batch, num_kv_heads, new_seq_len, head_dim]
        v_new: mx.array,
    ) -> tuple[mx.array, mx.array]:
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

        if self._quantize_mode == "fp4":
            # Quantize and store as FP4
            k_packed, k_scale = self._quantize_fp4(k_new)
            v_packed, v_scale = self._quantize_fp4(v_new)

            # Update cache slices
            self.k_cache[layer_idx] = mx.concatenate(
                [
                    self.k_cache[layer_idx][:, :, : self.seq_len, :],
                    k_packed,
                    self.k_cache[layer_idx][:, :, end_pos:, :],
                ],
                axis=2,
            )
            self.v_cache[layer_idx] = mx.concatenate(
                [
                    self.v_cache[layer_idx][:, :, : self.seq_len, :],
                    v_packed,
                    self.v_cache[layer_idx][:, :, end_pos:, :],
                ],
                axis=2,
            )
            self.k_scales[layer_idx] = mx.concatenate(
                [
                    self.k_scales[layer_idx][:, :, : self.seq_len, :],
                    k_scale,
                    self.k_scales[layer_idx][:, :, end_pos:, :],
                ],
                axis=2,
            )
            self.v_scales[layer_idx] = mx.concatenate(
                [
                    self.v_scales[layer_idx][:, :, : self.seq_len, :],
                    v_scale,
                    self.v_scales[layer_idx][:, :, end_pos:, :],
                ],
                axis=2,
            )

            # Dequantize full cache for attention (output in activations dtype)
            k_full = self._dequant_fp4(
                self.k_cache[layer_idx][:, :, :end_pos, :],
                self.k_scales[layer_idx][:, :, :end_pos, :],
            )
            v_full = self._dequant_fp4(
                self.v_cache[layer_idx][:, :, :end_pos, :],
                self.v_scales[layer_idx][:, :, :end_pos, :],
            )
        elif self._quantize_mode == "fp8":
            # Quantize and store as FP8
            k_quant, k_scale = self._quantize_fp8(k_new)
            v_quant, v_scale = self._quantize_fp8(v_new)

            # Update cache slices
            self.k_cache[layer_idx] = mx.concatenate(
                [
                    self.k_cache[layer_idx][:, :, : self.seq_len, :],
                    k_quant,
                    self.k_cache[layer_idx][:, :, end_pos:, :],
                ],
                axis=2,
            )
            self.v_cache[layer_idx] = mx.concatenate(
                [
                    self.v_cache[layer_idx][:, :, : self.seq_len, :],
                    v_quant,
                    self.v_cache[layer_idx][:, :, end_pos:, :],
                ],
                axis=2,
            )
            self.k_scales[layer_idx] = mx.concatenate(
                [
                    self.k_scales[layer_idx][:, :, : self.seq_len, :],
                    k_scale,
                    self.k_scales[layer_idx][:, :, end_pos:, :],
                ],
                axis=2,
            )
            self.v_scales[layer_idx] = mx.concatenate(
                [
                    self.v_scales[layer_idx][:, :, : self.seq_len, :],
                    v_scale,
                    self.v_scales[layer_idx][:, :, end_pos:, :],
                ],
                axis=2,
            )

            # Dequantize full cache for attention (output in activations dtype)
            k_full = self._dequant_fp8(
                self.k_cache[layer_idx][:, :, :end_pos, :],
                self.k_scales[layer_idx][:, :, :end_pos, :],
            )
            v_full = self._dequant_fp8(
                self.v_cache[layer_idx][:, :, :end_pos, :],
                self.v_scales[layer_idx][:, :, :end_pos, :],
            )
        else:
            # Direct storage using kv_cache dtype
            # Note: MLX doesn't support in-place slice assignment, so we concatenate
            self.k_cache[layer_idx] = mx.concatenate(
                [
                    self.k_cache[layer_idx][:, :, : self.seq_len, :],
                    k_new.astype(self.dtype_config.mlx_kv_cache),
                    self.k_cache[layer_idx][:, :, end_pos:, :],
                ],
                axis=2,
            )
            self.v_cache[layer_idx] = mx.concatenate(
                [
                    self.v_cache[layer_idx][:, :, : self.seq_len, :],
                    v_new.astype(self.dtype_config.mlx_kv_cache),
                    self.v_cache[layer_idx][:, :, end_pos:, :],
                ],
                axis=2,
            )

            # Return in activations dtype for attention computation
            k_full = self.k_cache[layer_idx][:, :, :end_pos, :].astype(self.dtype_config.mlx_activations)
            v_full = self.v_cache[layer_idx][:, :, :end_pos, :].astype(self.dtype_config.mlx_activations)

        return k_full, v_full

    def advance(self, num_tokens: int = 1):
        """Advance sequence position after processing tokens."""
        self.seq_len += num_tokens

    def reset(self):
        """Clear cache for new sequence."""
        self.seq_len = 0

    def get_kv(self, layer_idx: int) -> tuple[mx.array | None, mx.array | None]:
        """Get current cached K, V for a layer (dequantized if needed).

        Returns tensors in activations dtype from dtype_config.
        """
        if self.seq_len == 0:
            return None, None

        if self._quantize_mode == "fp4":
            k = self._dequant_fp4(
                self.k_cache[layer_idx][:, :, : self.seq_len, :],
                self.k_scales[layer_idx][:, :, : self.seq_len, :],
            )
            v = self._dequant_fp4(
                self.v_cache[layer_idx][:, :, : self.seq_len, :],
                self.v_scales[layer_idx][:, :, : self.seq_len, :],
            )
        elif self._quantize_mode == "fp8":
            k = self._dequant_fp8(
                self.k_cache[layer_idx][:, :, : self.seq_len, :],
                self.k_scales[layer_idx][:, :, : self.seq_len, :],
            )
            v = self._dequant_fp8(
                self.v_cache[layer_idx][:, :, : self.seq_len, :],
                self.v_scales[layer_idx][:, :, : self.seq_len, :],
            )
        else:
            k = self.k_cache[layer_idx][:, :, : self.seq_len, :].astype(self.dtype_config.mlx_activations)
            v = self.v_cache[layer_idx][:, :, : self.seq_len, :].astype(self.dtype_config.mlx_activations)

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

    def _quantize_fp4(self, tensor: mx.array) -> tuple[mx.array, mx.array]:
        """Quantize tensor to FP4 packed format."""
        # Find max absolute value per row for scaling
        abs_max = mx.max(mx.abs(tensor), axis=-1, keepdims=True)
        abs_max = mx.maximum(abs_max, 1e-8)  # Avoid division by zero

        # Scale to [-1, 1] range
        # FP4 E2M1 max value is 6.0 (1.5 * 2^2)
        scale = abs_max / 6.0
        scaled = tensor / scale

        # Clamp to FP4 range
        scaled = mx.clip(scaled, -6.0, 6.0)

        # Quantize to FP4 (simplified - in practice use proper rounding)
        # This is a placeholder - real implementation would pack 8 values per uint32
        # For now, we store as uint8 per value and pack later
        quantized = mx.round(scaled * 2.0).astype(mx.int8)  # Scale to use more bits
        quantized = mx.clip(quantized + 8, 0, 15).astype(mx.uint8)

        # Pack 8 values per uint32
        # Reshape to [..., head_dim // 8, 8]
        batch, heads, seq, dim = tensor.shape
        reshaped = quantized.reshape(batch, heads, seq, dim // 8, 8)

        # Pack (simplified)
        packed = mx.zeros((batch, heads, seq, dim // 8), dtype=mx.uint32)
        for i in range(8):
            packed = packed | (reshaped[..., i].astype(mx.uint32) << (i * 4))

        return packed, scale.astype(self.dtype_config.mlx_scales)

    def _dequant_fp4(self, packed: mx.array, scale: mx.array) -> mx.array:
        """Dequantize FP4 packed tensor to activations dtype."""
        batch, heads, seq, packed_dim = packed.shape
        dim = packed_dim * 8
        out_dtype = self.dtype_config.mlx_activations

        # Unpack (simplified)
        unpacked = mx.zeros((batch, heads, seq, dim), dtype=out_dtype)
        for i in range(8):
            nibble = (packed >> (i * 4)) & 0xF
            # Convert 4-bit unsigned to signed centered at 8
            signed = nibble.astype(out_dtype) - 8.0
            unpacked = mx.concatenate(
                [
                    unpacked[..., : packed_dim * i],
                    (signed * scale.astype(out_dtype) / 2.0).reshape(batch, heads, seq, -1),
                    unpacked[..., packed_dim * (i + 1) :],
                ],
                axis=-1,
            )

        # Proper unpack would use index operations
        # This is simplified placeholder
        return unpacked

    def _quantize_fp8(self, tensor: mx.array) -> tuple[mx.array, mx.array]:
        """Quantize tensor to FP8 E4M3 format (simulated with uint8 + scale).

        Uses E4M3 representation: 4 exponent bits, 3 mantissa bits.
        Max representable value is 448, min is ~2e-9.
        """
        # Find max absolute value per row for scaling
        abs_max = mx.max(mx.abs(tensor), axis=-1, keepdims=True)
        abs_max = mx.maximum(abs_max, 1e-8)  # Avoid division by zero

        # E4M3 max value is 448
        scale = abs_max / 448.0
        scaled = tensor / scale

        # Clamp to E4M3 range and quantize to 8-bit signed
        # Range is approximately [-448, 448] in E4M3
        scaled = mx.clip(scaled, -448.0, 448.0)

        # Convert to uint8 with offset (128-centered for signed range)
        # This is a simplified linear quantization; true E4M3 would use
        # non-linear encoding but MLX doesn't support native FP8
        # Use float arithmetic to avoid overflow, then convert to uint8
        quantized_float = mx.round(scaled / 448.0 * 127.0) + 128.0
        quantized = mx.clip(quantized_float, 0, 255).astype(mx.uint8)

        return quantized, scale.astype(self.dtype_config.mlx_scales)

    def _dequant_fp8(self, quantized: mx.array, scale: mx.array) -> mx.array:
        """Dequantize FP8 tensor to activations dtype."""
        out_dtype = self.dtype_config.mlx_activations

        # Reverse the quantization: uint8 centered at 128 -> signed -> scaled
        signed = quantized.astype(out_dtype) - 128.0
        dequant = signed / 127.0 * 448.0 * scale.astype(out_dtype)

        return dequant
