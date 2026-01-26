"""Quantized KV Cache for memory-efficient long-context inference.

Memory savings comparison (for 32-layer, 32-head model with head_dim=128):

Standard FP16 KV cache = 2 * n_layers * seq_len * n_kv_heads * head_dim * 2 bytes
Quantized FP8/INT8 = half the memory (with scales overhead ~1-2%)

For 8K context: FP16 = ~4GB, FP8 = ~2GB
For 32K context: FP16 = ~16GB, FP8 = ~8GB

Usage:
    from metal_marlin.cache import QuantizedKVCache, ScalingStrategy

    cache = QuantizedKVCache(
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        max_seq_len=8192,
        scaling=ScalingStrategy.PER_HEAD,  # Better accuracy
    )

    # During forward pass
    k_compressed, v_compressed = cache.compress_and_store(layer_idx, k_new, v_new)

    # For attention computation
    k_full, v_full = cache.get_kv_for_attention(layer_idx)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np

from .._compat import HAS_MLX, mx

# FP8 E4M3 format constants
# E4M3: 4 exponent bits, 3 mantissa bits
# Max value: 448 (2^8 * 1.75)
# Min positive: ~2^-9 = 0.00195
FP8_E4M3_MAX = 448.0

# INT8 symmetric range
INT8_MAX = 127


class ScalingStrategy(Enum):
    """Strategy for computing quantization scales.

    PER_HEAD: One scale per KV head per position. Better accuracy as each head
        can have different value distributions. Overhead: n_kv_heads * seq_len scales.

    PER_TOKEN: One scale per token position across all heads. Simpler and lower
        overhead, but assumes all heads have similar distributions.
        Overhead: seq_len scales.

    ASYMMETRIC: FP8 for keys (more sensitive to precision), INT8 for values
        (more robust to quantization noise). Combines benefits of both formats.
    """

    PER_HEAD = "per_head"
    PER_TOKEN = "per_token"
    ASYMMETRIC = "asymmetric"


@dataclass
class CacheStats:
    """Statistics about cache memory usage."""

    layers: int
    kv_heads: int
    head_dim: int
    current_seq_len: int
    max_seq_len: int
    scaling: ScalingStrategy

    @property
    def fp16_memory_bytes(self) -> int:
        """Memory if using FP16 storage."""
        # 2 (K+V) * layers * kv_heads * seq_len * head_dim * 2 bytes
        return 2 * self.layers * self.kv_heads * self.current_seq_len * self.head_dim * 2

    @property
    def quantized_memory_bytes(self) -> int:
        """Memory with current quantized storage."""
        # Base: 1 byte per element (FP8/INT8)
        base = 2 * self.layers * self.kv_heads * self.current_seq_len * self.head_dim

        # Scale overhead depends on strategy
        if self.scaling == ScalingStrategy.PER_HEAD:
            # One FP16 scale per head per position for both K and V
            scale_bytes = 2 * self.layers * self.kv_heads * self.current_seq_len * 2
        elif self.scaling == ScalingStrategy.PER_TOKEN:
            # One FP16 scale per position for both K and V
            scale_bytes = 2 * self.layers * self.current_seq_len * 2
        else:  # ASYMMETRIC
            # FP8 K has per-head scales, INT8 V has per-head scales
            scale_bytes = 2 * self.layers * self.kv_heads * self.current_seq_len * 2

        return base + scale_bytes

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs FP16."""
        if self.current_seq_len == 0:
            return 1.0
        return self.fp16_memory_bytes / self.quantized_memory_bytes

    @property
    def memory_saved_mb(self) -> float:
        """Memory saved in MB."""
        return (self.fp16_memory_bytes - self.quantized_memory_bytes) / (1024 * 1024)


def compress_kv(
    k: np.ndarray | None,
    v: np.ndarray | None,
    scaling: ScalingStrategy = ScalingStrategy.PER_HEAD,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Quantize incoming K, V tensors for storage.

    Args:
        k: Key tensor [batch, num_kv_heads, seq_len, head_dim] or None
        v: Value tensor [batch, num_kv_heads, seq_len, head_dim] or None
        scaling: Scaling strategy for quantization

    Returns:
        ((k_quantized, k_scales), (v_quantized, v_scales))
        - k_quantized: uint8 tensor with quantized keys
        - k_scales: FP16 scales for dequantization
        - v_quantized: uint8 tensor with quantized values
        - v_scales: FP16 scales for dequantization
    """
    if k is None or v is None:
        raise ValueError("Both k and v must be provided")

    if scaling == ScalingStrategy.ASYMMETRIC:
        # FP8 for keys, INT8 for values
        k_q, k_s = _quantize_fp8_e4m3(k, per_head=(scaling != ScalingStrategy.PER_TOKEN))
        v_q, v_s = _quantize_int8_symmetric(v, per_head=(scaling != ScalingStrategy.PER_TOKEN))
    else:
        # FP8 for both with specified granularity
        per_head = scaling == ScalingStrategy.PER_HEAD
        k_q, k_s = _quantize_fp8_e4m3(k, per_head=per_head)
        v_q, v_s = _quantize_fp8_e4m3(v, per_head=per_head)

    return (k_q, k_s), (v_q, v_s)


def decompress_kv(
    k_q: np.ndarray,
    k_scales: np.ndarray,
    v_q: np.ndarray,
    v_scales: np.ndarray,
    scaling: ScalingStrategy = ScalingStrategy.PER_HEAD,
    output_dtype: np.dtype = np.float16,
) -> tuple[np.ndarray, np.ndarray]:
    """Dequantize K, V tensors for attention computation.

    Args:
        k_q: Quantized keys [batch, num_kv_heads, seq_len, head_dim] uint8
        k_scales: Key scales (shape depends on scaling strategy)
        v_q: Quantized values [batch, num_kv_heads, seq_len, head_dim] uint8
        v_scales: Value scales (shape depends on scaling strategy)
        scaling: Scaling strategy used during quantization
        output_dtype: Output dtype (default FP16 for attention)

    Returns:
        (k_deq, v_deq): Dequantized tensors in output_dtype
    """
    if scaling == ScalingStrategy.ASYMMETRIC:
        # FP8 for keys, INT8 for values
        k = _dequantize_fp8_e4m3(k_q, k_scales, output_dtype=output_dtype)
        v = _dequantize_int8_symmetric(v_q, v_scales, output_dtype=output_dtype)
    else:
        # FP8 for both
        k = _dequantize_fp8_e4m3(k_q, k_scales, output_dtype=output_dtype)
        v = _dequantize_fp8_e4m3(v_q, v_scales, output_dtype=output_dtype)

    return k, v


def _quantize_fp8_e4m3(
    tensor: np.ndarray,
    per_head: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize to FP8 E4M3 format (simulated with uint8 + scale).

    FP8 E4M3 has range ~[-448, 448] with ~3 bits of mantissa precision.
    We simulate this with linear quantization to uint8 with a scale factor.

    Args:
        tensor: Input tensor [batch, num_kv_heads, seq_len, head_dim]
        per_head: If True, compute one scale per head per position.
                  If False, compute one scale per position (shared across heads).

    Returns:
        (quantized, scales): uint8 quantized values and FP16 scales
    """
    # Compute absolute max for scaling
    if per_head:
        # Scale per [batch, head, position]
        abs_max = np.max(np.abs(tensor), axis=-1, keepdims=True)
    else:
        # Scale per [batch, position] - max across heads and head_dim
        abs_max = np.max(np.abs(tensor), axis=(1, 3), keepdims=True)

    # Avoid division by zero
    abs_max = np.maximum(abs_max, 1e-12)

    # Scale to fit in FP8 E4M3 range
    scale = abs_max / FP8_E4M3_MAX
    scaled = tensor / scale

    # Clamp to FP8 range
    scaled = np.clip(scaled, -FP8_E4M3_MAX, FP8_E4M3_MAX)

    # Linear quantization to uint8 (centered at 128)
    # Map [-448, 448] to [0, 255]
    quantized = np.round(scaled / FP8_E4M3_MAX * 127.0 + 128.0)
    quantized = np.clip(quantized, 0, 255).astype(np.uint8)

    return quantized, scale.astype(np.float16)


def _dequantize_fp8_e4m3(
    quantized: np.ndarray,
    scale: np.ndarray,
    output_dtype: np.dtype = np.float16,
) -> np.ndarray:
    """Dequantize FP8 E4M3 format to floating point.

    Args:
        quantized: uint8 quantized values
        scale: FP16 scale factors
        output_dtype: Output dtype

    Returns:
        Dequantized tensor in output_dtype
    """
    # Convert back from uint8 to signed scaled values
    # [0, 255] -> [-448, 448]
    signed = (quantized.astype(np.float32) - 128.0) / 127.0 * FP8_E4M3_MAX
    dequantized = signed * scale.astype(np.float32)

    return dequantized.astype(output_dtype)


def _quantize_int8_symmetric(
    tensor: np.ndarray,
    per_head: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize to INT8 symmetric format.

    INT8 symmetric maps values to [-127, 127] with a scale factor.
    Values are more robust to this quantization than keys.

    Args:
        tensor: Input tensor [batch, num_kv_heads, seq_len, head_dim]
        per_head: If True, compute one scale per head per position.

    Returns:
        (quantized, scales): uint8 quantized values (offset by 128) and FP16 scales
    """
    # Compute absolute max for scaling
    if per_head:
        abs_max = np.max(np.abs(tensor), axis=-1, keepdims=True)
    else:
        abs_max = np.max(np.abs(tensor), axis=(1, 3), keepdims=True)

    # Avoid division by zero
    abs_max = np.maximum(abs_max, 1e-12)

    # Scale to fit in INT8 range
    scale = abs_max / INT8_MAX
    scaled = tensor / scale

    # Clamp and quantize
    scaled = np.clip(scaled, -INT8_MAX, INT8_MAX)
    quantized = np.round(scaled + 128.0)  # Offset to unsigned
    quantized = np.clip(quantized, 0, 255).astype(np.uint8)

    return quantized, scale.astype(np.float16)


def _dequantize_int8_symmetric(
    quantized: np.ndarray,
    scale: np.ndarray,
    output_dtype: np.dtype = np.float16,
) -> np.ndarray:
    """Dequantize INT8 symmetric format to floating point.

    Args:
        quantized: uint8 quantized values (offset by 128)
        scale: FP16 scale factors
        output_dtype: Output dtype

    Returns:
        Dequantized tensor in output_dtype
    """
    # Convert back from uint8 to signed
    signed = quantized.astype(np.float32) - 128.0
    dequantized = signed * scale.astype(np.float32)

    return dequantized.astype(output_dtype)


class QuantizedKVCache:
    """Quantized KV cache with FP8 storage and FP16 compute.

    Stores K and V tensors in quantized format (FP8 or INT8) to achieve
    approximately 2x memory savings compared to FP16 storage. Dequantization
    happens on-the-fly during attention computation.

    Supports three scaling strategies:
    - PER_HEAD: One scale per KV head per position (best accuracy)
    - PER_TOKEN: One scale per token position (lower overhead)
    - ASYMMETRIC: FP8 for keys, INT8 for values (values more robust)

    Memory formula:
        Standard: 2 * n_layers * seq_len * n_kv_heads * head_dim * 2 bytes
        Quantized: 2 * n_layers * seq_len * n_kv_heads * head_dim * 1 byte + scales

    Expected 2x longer context in same memory.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        batch_size: int = 1,
        scaling: ScalingStrategy = ScalingStrategy.PER_HEAD,
        compute_dtype: Literal["fp16", "bf16"] = "fp16",
    ):
        """Initialize quantized KV cache.

        Args:
            num_layers: Number of transformer layers
            num_kv_heads: Number of KV heads (for GQA this is < num_q_heads)
            head_dim: Dimension per head
            max_seq_len: Maximum sequence length to allocate
            batch_size: Batch size (typically 1 for inference)
            scaling: Quantization scaling strategy
            compute_dtype: Dtype for attention computation output (fp16 or bf16)
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.scaling = scaling
        self.compute_dtype = compute_dtype
        self.seq_len = 0

        # Pre-allocate quantized storage
        cache_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)

        # Quantized data stored as uint8
        self.k_cache: list[np.ndarray] = [
            np.zeros(cache_shape, dtype=np.uint8) for _ in range(num_layers)
        ]
        self.v_cache: list[np.ndarray] = [
            np.zeros(cache_shape, dtype=np.uint8) for _ in range(num_layers)
        ]

        # Scales storage depends on strategy
        if scaling == ScalingStrategy.PER_TOKEN:
            # One scale per position for all heads
            scale_shape = (batch_size, 1, max_seq_len, 1)
        else:
            # One scale per head per position (PER_HEAD and ASYMMETRIC)
            scale_shape = (batch_size, num_kv_heads, max_seq_len, 1)

        self.k_scales: list[np.ndarray] = [
            np.zeros(scale_shape, dtype=np.float16) for _ in range(num_layers)
        ]
        self.v_scales: list[np.ndarray] = [
            np.zeros(scale_shape, dtype=np.float16) for _ in range(num_layers)
        ]

    def compress_and_store(
        self,
        layer_idx: int,
        k_new: np.ndarray,
        v_new: np.ndarray,
    ) -> None:
        """Quantize and store new K, V tensors in cache.

        Args:
            layer_idx: Transformer layer index
            k_new: New keys [batch, num_kv_heads, new_seq_len, head_dim]
            v_new: New values [batch, num_kv_heads, new_seq_len, head_dim]
        """
        new_seq_len = k_new.shape[2]
        end_pos = self.seq_len + new_seq_len

        if end_pos > self.max_seq_len:
            raise ValueError(
                f"Sequence length {end_pos} exceeds max_seq_len {self.max_seq_len}"
            )

        # Quantize
        (k_q, k_s), (v_q, v_s) = compress_kv(k_new, v_new, self.scaling)

        # Store in cache
        self.k_cache[layer_idx][:, :, self.seq_len:end_pos, :] = k_q
        self.v_cache[layer_idx][:, :, self.seq_len:end_pos, :] = v_q
        self.k_scales[layer_idx][:, :, self.seq_len:end_pos, :] = k_s
        self.v_scales[layer_idx][:, :, self.seq_len:end_pos, :] = v_s

    def get_kv_for_attention(
        self,
        layer_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get dequantized K, V for attention computation.

        Args:
            layer_idx: Transformer layer index

        Returns:
            (k, v): Dequantized tensors in compute_dtype
        """
        if self.seq_len == 0:
            return (
                np.zeros(
                    (self.batch_size, self.num_kv_heads, 0, self.head_dim),
                    dtype=np.float16,
                ),
                np.zeros(
                    (self.batch_size, self.num_kv_heads, 0, self.head_dim),
                    dtype=np.float16,
                ),
            )

        k_q = self.k_cache[layer_idx][:, :, :self.seq_len, :]
        k_s = self.k_scales[layer_idx][:, :, :self.seq_len, :]
        v_q = self.v_cache[layer_idx][:, :, :self.seq_len, :]
        v_s = self.v_scales[layer_idx][:, :, :self.seq_len, :]

        output_dtype = np.float16  # numpy doesn't have bf16, use fp16

        return decompress_kv(k_q, k_s, v_q, v_s, self.scaling, output_dtype)

    def update(
        self,
        layer_idx: int,
        k_new: np.ndarray,
        v_new: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update cache with new K, V and return full dequantized cache.

        Convenience method that combines compress_and_store and get_kv_for_attention.
        Returns the full cache including the newly added tokens.

        Note: This method returns cache data including new tokens but does NOT
        advance seq_len. Call advance() separately after updating all layers.

        Args:
            layer_idx: Transformer layer index
            k_new: New keys [batch, num_kv_heads, new_seq_len, head_dim]
            v_new: New values [batch, num_kv_heads, new_seq_len, head_dim]

        Returns:
            (k_full, v_full): Full dequantized KV cache including new tokens
        """
        new_seq_len = k_new.shape[2]
        end_pos = self.seq_len + new_seq_len

        if end_pos > self.max_seq_len:
            raise ValueError(
                f"Sequence length {end_pos} exceeds max_seq_len {self.max_seq_len}"
            )

        # Quantize and store
        (k_q, k_s), (v_q, v_s) = compress_kv(k_new, v_new, self.scaling)
        self.k_cache[layer_idx][:, :, self.seq_len:end_pos, :] = k_q
        self.v_cache[layer_idx][:, :, self.seq_len:end_pos, :] = v_q
        self.k_scales[layer_idx][:, :, self.seq_len:end_pos, :] = k_s
        self.v_scales[layer_idx][:, :, self.seq_len:end_pos, :] = v_s

        # Return dequantized full cache (including new tokens)
        k_q_full = self.k_cache[layer_idx][:, :, :end_pos, :]
        k_s_full = self.k_scales[layer_idx][:, :, :end_pos, :]
        v_q_full = self.v_cache[layer_idx][:, :, :end_pos, :]
        v_s_full = self.v_scales[layer_idx][:, :, :end_pos, :]

        output_dtype = np.float16
        return decompress_kv(k_q_full, k_s_full, v_q_full, v_s_full, self.scaling, output_dtype)

    def advance(self, num_tokens: int = 1) -> None:
        """Advance sequence position after processing tokens.

        Call this AFTER updating all layers for the current token(s).
        """
        self.seq_len += num_tokens

    def reset(self) -> None:
        """Clear cache for new sequence."""
        self.seq_len = 0

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return CacheStats(
            layers=self.num_layers,
            kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            current_seq_len=self.seq_len,
            max_seq_len=self.max_seq_len,
            scaling=self.scaling,
        )

    def memory_usage_mb(self) -> float:
        """Return current memory usage in MB."""
        return self.get_stats().quantized_memory_bytes / (1024 * 1024)

    def memory_saved_mb(self) -> float:
        """Return memory saved vs FP16 in MB."""
        return self.get_stats().memory_saved_mb


# MLX-accelerated versions when available
if HAS_MLX and mx is not None:

    def compress_kv_mlx(
        k: mx.array,
        v: mx.array,
        scaling: ScalingStrategy = ScalingStrategy.PER_HEAD,
    ) -> tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]]:
        """MLX-accelerated KV compression.

        Args:
            k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
            v: Value tensor [batch, num_kv_heads, seq_len, head_dim]
            scaling: Scaling strategy

        Returns:
            ((k_q, k_s), (v_q, v_s)): Quantized tensors and scales
        """
        per_head = scaling != ScalingStrategy.PER_TOKEN

        if scaling == ScalingStrategy.ASYMMETRIC:
            k_q, k_s = _quantize_fp8_e4m3_mlx(k, per_head=per_head)
            v_q, v_s = _quantize_int8_symmetric_mlx(v, per_head=per_head)
        else:
            k_q, k_s = _quantize_fp8_e4m3_mlx(k, per_head=per_head)
            v_q, v_s = _quantize_fp8_e4m3_mlx(v, per_head=per_head)

        return (k_q, k_s), (v_q, v_s)

    def decompress_kv_mlx(
        k_q: mx.array,
        k_scales: mx.array,
        v_q: mx.array,
        v_scales: mx.array,
        scaling: ScalingStrategy = ScalingStrategy.PER_HEAD,
        output_dtype: mx.Dtype = None,
    ) -> tuple[mx.array, mx.array]:
        """MLX-accelerated KV decompression.

        Args:
            k_q: Quantized keys
            k_scales: Key scales
            v_q: Quantized values
            v_scales: Value scales
            scaling: Scaling strategy
            output_dtype: Output dtype (default mx.float16)

        Returns:
            (k, v): Dequantized tensors
        """
        if output_dtype is None:
            output_dtype = mx.float16

        if scaling == ScalingStrategy.ASYMMETRIC:
            k = _dequantize_fp8_e4m3_mlx(k_q, k_scales, output_dtype)
            v = _dequantize_int8_symmetric_mlx(v_q, v_scales, output_dtype)
        else:
            k = _dequantize_fp8_e4m3_mlx(k_q, k_scales, output_dtype)
            v = _dequantize_fp8_e4m3_mlx(v_q, v_scales, output_dtype)

        return k, v

    def _quantize_fp8_e4m3_mlx(
        tensor: mx.array,
        per_head: bool = True,
    ) -> tuple[mx.array, mx.array]:
        """MLX FP8 E4M3 quantization."""
        if per_head:
            abs_max = mx.max(mx.abs(tensor), axis=-1, keepdims=True)
        else:
            abs_max = mx.max(mx.abs(tensor), axis=(1, 3), keepdims=True)

        abs_max = mx.maximum(abs_max, 1e-12)
        scale = abs_max / FP8_E4M3_MAX
        scaled = tensor / scale
        scaled = mx.clip(scaled, -FP8_E4M3_MAX, FP8_E4M3_MAX)

        quantized = mx.round(scaled / FP8_E4M3_MAX * 127.0 + 128.0)
        quantized = mx.clip(quantized, 0, 255).astype(mx.uint8)

        return quantized, scale.astype(mx.float16)

    def _dequantize_fp8_e4m3_mlx(
        quantized: mx.array,
        scale: mx.array,
        output_dtype: mx.Dtype,
    ) -> mx.array:
        """MLX FP8 E4M3 dequantization."""
        signed = (quantized.astype(mx.float32) - 128.0) / 127.0 * FP8_E4M3_MAX
        dequantized = signed * scale.astype(mx.float32)
        return dequantized.astype(output_dtype)

    def _quantize_int8_symmetric_mlx(
        tensor: mx.array,
        per_head: bool = True,
    ) -> tuple[mx.array, mx.array]:
        """MLX INT8 symmetric quantization."""
        if per_head:
            abs_max = mx.max(mx.abs(tensor), axis=-1, keepdims=True)
        else:
            abs_max = mx.max(mx.abs(tensor), axis=(1, 3), keepdims=True)

        abs_max = mx.maximum(abs_max, 1e-12)
        scale = abs_max / INT8_MAX
        scaled = tensor / scale
        scaled = mx.clip(scaled, -INT8_MAX, INT8_MAX)

        quantized = mx.round(scaled + 128.0)
        quantized = mx.clip(quantized, 0, 255).astype(mx.uint8)

        return quantized, scale.astype(mx.float16)

    def _dequantize_int8_symmetric_mlx(
        quantized: mx.array,
        scale: mx.array,
        output_dtype: mx.Dtype,
    ) -> mx.array:
        """MLX INT8 symmetric dequantization."""
        signed = quantized.astype(mx.float32) - 128.0
        dequantized = signed * scale.astype(mx.float32)
        return dequantized.astype(output_dtype)

    class QuantizedKVCacheMLX:
        """MLX-accelerated quantized KV cache.

        Uses Metal compute for quantization/dequantization operations,
        significantly faster than numpy for large caches.
        """

        def __init__(
            self,
            num_layers: int,
            num_kv_heads: int,
            head_dim: int,
            max_seq_len: int,
            batch_size: int = 1,
            scaling: ScalingStrategy = ScalingStrategy.PER_HEAD,
            compute_dtype: mx.Dtype = None,
        ):
            """Initialize MLX quantized KV cache.

            Args:
                num_layers: Number of transformer layers
                num_kv_heads: Number of KV heads
                head_dim: Dimension per head
                max_seq_len: Maximum sequence length
                batch_size: Batch size
                scaling: Quantization scaling strategy
                compute_dtype: MLX dtype for attention (default mx.float16)
            """
            self.num_layers = num_layers
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.max_seq_len = max_seq_len
            self.batch_size = batch_size
            self.scaling = scaling
            self.compute_dtype = compute_dtype if compute_dtype is not None else mx.float16
            self.seq_len = 0

            # Pre-allocate
            cache_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
            self.k_cache: list[mx.array] = [
                mx.zeros(cache_shape, dtype=mx.uint8) for _ in range(num_layers)
            ]
            self.v_cache: list[mx.array] = [
                mx.zeros(cache_shape, dtype=mx.uint8) for _ in range(num_layers)
            ]

            if scaling == ScalingStrategy.PER_TOKEN:
                scale_shape = (batch_size, 1, max_seq_len, 1)
            else:
                scale_shape = (batch_size, num_kv_heads, max_seq_len, 1)

            self.k_scales: list[mx.array] = [
                mx.zeros(scale_shape, dtype=mx.float16) for _ in range(num_layers)
            ]
            self.v_scales: list[mx.array] = [
                mx.zeros(scale_shape, dtype=mx.float16) for _ in range(num_layers)
            ]

        def compress_and_store(
            self,
            layer_idx: int,
            k_new: mx.array,
            v_new: mx.array,
        ) -> None:
            """Quantize and store new K, V tensors."""
            new_seq_len = k_new.shape[2]
            end_pos = self.seq_len + new_seq_len

            if end_pos > self.max_seq_len:
                raise ValueError(
                    f"Sequence length {end_pos} exceeds max_seq_len {self.max_seq_len}"
                )

            (k_q, k_s), (v_q, v_s) = compress_kv_mlx(k_new, v_new, self.scaling)

            # MLX is immutable, so we concatenate slices
            self.k_cache[layer_idx] = mx.concatenate(
                [
                    self.k_cache[layer_idx][:, :, :self.seq_len, :],
                    k_q,
                    self.k_cache[layer_idx][:, :, end_pos:, :],
                ],
                axis=2,
            )
            self.v_cache[layer_idx] = mx.concatenate(
                [
                    self.v_cache[layer_idx][:, :, :self.seq_len, :],
                    v_q,
                    self.v_cache[layer_idx][:, :, end_pos:, :],
                ],
                axis=2,
            )
            self.k_scales[layer_idx] = mx.concatenate(
                [
                    self.k_scales[layer_idx][:, :, :self.seq_len, :],
                    k_s,
                    self.k_scales[layer_idx][:, :, end_pos:, :],
                ],
                axis=2,
            )
            self.v_scales[layer_idx] = mx.concatenate(
                [
                    self.v_scales[layer_idx][:, :, :self.seq_len, :],
                    v_s,
                    self.v_scales[layer_idx][:, :, end_pos:, :],
                ],
                axis=2,
            )

        def get_kv_for_attention(
            self,
            layer_idx: int,
        ) -> tuple[mx.array, mx.array]:
            """Get dequantized K, V for attention."""
            if self.seq_len == 0:
                shape = (self.batch_size, self.num_kv_heads, 0, self.head_dim)
                return mx.zeros(shape, dtype=self.compute_dtype), mx.zeros(
                    shape, dtype=self.compute_dtype
                )

            k_q = self.k_cache[layer_idx][:, :, :self.seq_len, :]
            k_s = self.k_scales[layer_idx][:, :, :self.seq_len, :]
            v_q = self.v_cache[layer_idx][:, :, :self.seq_len, :]
            v_s = self.v_scales[layer_idx][:, :, :self.seq_len, :]

            return decompress_kv_mlx(k_q, k_s, v_q, v_s, self.scaling, self.compute_dtype)

        def update(
            self,
            layer_idx: int,
            k_new: mx.array,
            v_new: mx.array,
        ) -> tuple[mx.array, mx.array]:
            """Update cache and return full dequantized cache including new tokens.

            Note: Does NOT advance seq_len. Call advance() after updating all layers.
            """
            new_seq_len = k_new.shape[2]
            end_pos = self.seq_len + new_seq_len

            if end_pos > self.max_seq_len:
                raise ValueError(
                    f"Sequence length {end_pos} exceeds max_seq_len {self.max_seq_len}"
                )

            # Compress and store
            (k_q, k_s), (v_q, v_s) = compress_kv_mlx(k_new, v_new, self.scaling)

            # MLX concatenate for immutable arrays
            self.k_cache[layer_idx] = mx.concatenate(
                [
                    self.k_cache[layer_idx][:, :, :self.seq_len, :],
                    k_q,
                    self.k_cache[layer_idx][:, :, end_pos:, :],
                ],
                axis=2,
            )
            self.v_cache[layer_idx] = mx.concatenate(
                [
                    self.v_cache[layer_idx][:, :, :self.seq_len, :],
                    v_q,
                    self.v_cache[layer_idx][:, :, end_pos:, :],
                ],
                axis=2,
            )
            self.k_scales[layer_idx] = mx.concatenate(
                [
                    self.k_scales[layer_idx][:, :, :self.seq_len, :],
                    k_s,
                    self.k_scales[layer_idx][:, :, end_pos:, :],
                ],
                axis=2,
            )
            self.v_scales[layer_idx] = mx.concatenate(
                [
                    self.v_scales[layer_idx][:, :, :self.seq_len, :],
                    v_s,
                    self.v_scales[layer_idx][:, :, end_pos:, :],
                ],
                axis=2,
            )

            # Return dequantized full cache including new tokens
            k_q_full = self.k_cache[layer_idx][:, :, :end_pos, :]
            k_s_full = self.k_scales[layer_idx][:, :, :end_pos, :]
            v_q_full = self.v_cache[layer_idx][:, :, :end_pos, :]
            v_s_full = self.v_scales[layer_idx][:, :, :end_pos, :]

            return decompress_kv_mlx(k_q_full, k_s_full, v_q_full, v_s_full, self.scaling, self.compute_dtype)

        def advance(self, num_tokens: int = 1) -> None:
            """Advance sequence position."""
            self.seq_len += num_tokens

        def reset(self) -> None:
            """Clear cache."""
            self.seq_len = 0

        def get_stats(self) -> CacheStats:
            """Get cache statistics."""
            return CacheStats(
                layers=self.num_layers,
                kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                current_seq_len=self.seq_len,
                max_seq_len=self.max_seq_len,
                scaling=self.scaling,
            )

        def memory_usage_mb(self) -> float:
            """Return current memory usage in MB."""
            return self.get_stats().quantized_memory_bytes / (1024 * 1024)
