"""Quantized KV cache for memory-efficient inference.

This module provides a lightweight Torch/MPS KV cache with FP4 or INT8 storage.
It is designed for decode-time workloads where KV bandwidth dominates latency.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


class QuantizedKVCache:
    """KV cache with FP4/INT8 quantization.

    Layout:
    - Quantized cache: ``[seq, layer, head, packed_dim]`` (uint8)
    - Scales: ``[seq, layer, head, num_scale_groups]`` (fp16)

    ``fp4`` stores two values per byte (4-bit nibbles).
    ``int8`` stores one value per byte with a +128 unsigned offset.
    """

    def __init__(
        self,
        max_seq_len: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        quant_dtype: str = "fp4",  # or "int8"
        scale_group_size: int = 128,
        device: str | torch.device | None = None,
    ):
        if quant_dtype not in {"fp4", "int8"}:
            raise ValueError(f"quant_dtype must be 'fp4' or 'int8', got {quant_dtype!r}")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        if num_layers <= 0 or num_heads <= 0 or head_dim <= 0:
            raise ValueError("num_layers, num_heads, and head_dim must be > 0")
        if scale_group_size <= 0:
            raise ValueError("scale_group_size must be > 0")
        if quant_dtype == "fp4" and head_dim % 2 != 0:
            raise ValueError("FP4 cache requires even head_dim (two values packed per byte)")

        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.quant_dtype = quant_dtype
        self.scale_group_size = scale_group_size
        self.device = torch.device(
            device
            if device is not None
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        self.values_per_byte = 2 if quant_dtype == "fp4" else 1
        self.packed_head_dim = head_dim // self.values_per_byte
        self.num_scale_groups = math.ceil(head_dim / scale_group_size)
        self.current_len = 0

        self.k_cache = torch.zeros(
            (max_seq_len, num_layers, num_heads, self.packed_head_dim),
            dtype=torch.uint8,
            device=self.device,
        )
        self.v_cache = torch.zeros_like(self.k_cache)

        self.k_scales = torch.ones(
            (max_seq_len, num_layers, num_heads, self.num_scale_groups),
            dtype=torch.float16,
            device=self.device,
        )
        self.v_scales = torch.ones_like(self.k_scales)

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        position: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add new K/V and return full dequantized cache range for this layer."""
        if not (0 <= layer_idx < self.num_layers):
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")
        if not (0 <= position < self.max_seq_len):
            raise IndexError(f"position {position} out of range [0, {self.max_seq_len})")

        k_tensor = self._normalize_token(new_k, "new_k")
        v_tensor = self._normalize_token(new_v, "new_v")

        k_quant, k_scale = self._quantize(k_tensor)
        v_quant, v_scale = self._quantize(v_tensor)

        self.k_cache[position, layer_idx].copy_(k_quant)
        self.v_cache[position, layer_idx].copy_(v_quant)
        self.k_scales[position, layer_idx].copy_(k_scale)
        self.v_scales[position, layer_idx].copy_(v_scale)

        self.current_len = max(self.current_len, position + 1)
        return self._dequantize_range(0, position + 1, layer_idx)

    def attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        start: int = 0,
        end: int | None = None,
    ) -> torch.Tensor:
        """Run fused dequant+attention on a cached range for one layer.

        Args:
            layer_idx: Transformer layer index.
            query: Query tensor ``[heads, head_dim]`` or ``[1, heads, head_dim]``.
            start: Start position (inclusive).
            end: End position (exclusive). Defaults to current cache length.

        Returns:
            Attention output with shape ``[heads, head_dim]``.
        """
        from ..kernels import quantized_kv_attention_decode

        if end is None:
            end = self.current_len

        query_norm = self._normalize_token(query, "query")
        if end <= start:
            return torch.zeros_like(query_norm)

        if query_norm.device.type != "mps":
            return self._attention_fallback(layer_idx, query_norm, start, end)

        try:
            return quantized_kv_attention_decode(
                query=query_norm,
                k_cache=self.k_cache[start:end, layer_idx],
                v_cache=self.v_cache[start:end, layer_idx],
                k_scales=self.k_scales[start:end, layer_idx],
                v_scales=self.v_scales[start:end, layer_idx],
                num_heads_q=self.num_heads,
                num_kv_heads=self.num_heads,
                head_dim=self.head_dim,
                quant_dtype=self.quant_dtype,
                group_size=self.scale_group_size,
            )
        except (ImportError, RuntimeError):
            return self._attention_fallback(layer_idx, query_norm, start, end)

    def _quantize(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to FP4/INT8 with per-group scales."""
        scales = self._compute_group_scales(tensor)
        expanded_scales = self._expand_scales(scales)

        if self.quant_dtype == "fp4":
            quant = torch.round(tensor / expanded_scales).clamp(-8, 7).to(torch.int16)
            quant_u8 = (quant + 8).to(torch.uint8)
            packed = quant_u8[:, 0::2] | (quant_u8[:, 1::2] << 4)
            return packed, scales

        # INT8 symmetric quantization stored in uint8 with +128 offset.
        quant = torch.round(tensor / expanded_scales).clamp(-127, 127).to(torch.int16)
        quant_u8 = (quant + 128).to(torch.uint8)
        return quant_u8, scales

    def _dequantize_range(self, start: int, end: int, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize a range of the cache for one layer."""
        if not (0 <= layer_idx < self.num_layers):
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")
        if not (0 <= start <= end <= self.max_seq_len):
            raise IndexError(
                f"Invalid range [{start}, {end}) for max_seq_len={self.max_seq_len}"
            )

        k_quant = self.k_cache[start:end, layer_idx]
        v_quant = self.v_cache[start:end, layer_idx]
        k_scale = self.k_scales[start:end, layer_idx]
        v_scale = self.v_scales[start:end, layer_idx]

        k = self._dequantize_tensor(k_quant, k_scale)
        v = self._dequantize_tensor(v_quant, v_scale)
        return k, v

    def reset(self) -> None:
        """Reset logical sequence length for a new request."""
        self.current_len = 0

    def fp16_memory_bytes(self, seq_len: int | None = None) -> int:
        """Estimated FP16 cache footprint for K+V at ``seq_len``."""
        use_len = self.current_len if seq_len is None else seq_len
        return use_len * self.num_layers * 2 * self.num_heads * self.head_dim * 2

    def quantized_memory_bytes(self, seq_len: int | None = None) -> int:
        """Quantized cache footprint for K+V + scales at ``seq_len``."""
        use_len = self.current_len if seq_len is None else seq_len
        data_bytes = use_len * self.num_layers * 2 * self.num_heads * self.packed_head_dim
        scale_bytes = use_len * self.num_layers * 2 * self.num_heads * self.num_scale_groups * 2
        return data_bytes + scale_bytes

    def compression_ratio(self, seq_len: int | None = None) -> float:
        """FP16 bytes / quantized bytes."""
        q_bytes = self.quantized_memory_bytes(seq_len)
        if q_bytes == 0:
            return 1.0
        return self.fp16_memory_bytes(seq_len) / q_bytes

    def _normalize_token(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.ndim == 3:
            if tensor.shape[0] != 1:
                raise ValueError(
                    f"{name} with rank 3 must have batch=1, got shape {tuple(tensor.shape)}"
                )
            tensor = tensor.squeeze(0)
        if tensor.ndim != 2:
            raise ValueError(
                f"{name} must have shape [heads, head_dim] or [1, heads, head_dim], "
                f"got {tuple(tensor.shape)}"
            )
        expected = (self.num_heads, self.head_dim)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} != expected {expected}")
        return tensor.to(device=self.device, dtype=self.dtype)

    def _compute_group_scales(self, tensor: torch.Tensor) -> torch.Tensor:
        max_q = 7.0 if self.quant_dtype == "fp4" else 127.0
        padded_dim = self.num_scale_groups * self.scale_group_size

        if padded_dim != self.head_dim:
            tensor = torch.nn.functional.pad(tensor, (0, padded_dim - self.head_dim))

        grouped = tensor.view(self.num_heads, self.num_scale_groups, self.scale_group_size)
        abs_max = grouped.abs().amax(dim=-1).clamp(min=1e-6)
        return (abs_max / max_q).to(torch.float16)

    def _expand_scales(self, scales: torch.Tensor) -> torch.Tensor:
        expanded = (
            scales.to(torch.float32)
            .unsqueeze(-1)
            .expand(-1, -1, self.scale_group_size)
            .reshape(self.num_heads, -1)
        )
        return expanded[:, : self.head_dim]

    def _dequantize_tensor(self, quant: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        seq_len = quant.shape[0]
        expanded_scales = (
            scales.to(torch.float32)
            .unsqueeze(-1)
            .expand(-1, -1, -1, self.scale_group_size)
            .reshape(seq_len, self.num_heads, -1)
        )[..., : self.head_dim]

        if self.quant_dtype == "fp4":
            low = (quant & 0x0F).to(torch.float32)
            high = ((quant >> 4) & 0x0F).to(torch.float32)
            values = torch.empty(
                (seq_len, self.num_heads, self.head_dim),
                dtype=torch.float32,
                device=quant.device,
            )
            values[..., 0::2] = low - 8.0
            values[..., 1::2] = high - 8.0
            return (values * expanded_scales).to(self.dtype)

        values = quant.to(torch.float32) - 128.0
        return (values * expanded_scales).to(self.dtype)

    def _attention_fallback(
        self,
        layer_idx: int,
        query: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        k_cache, v_cache = self._dequantize_range(start, end, layer_idx)
        # [seq, heads, dim] -> [heads, seq, dim]
        k = k_cache.to(torch.float32).permute(1, 0, 2)
        v = v_cache.to(torch.float32).permute(1, 0, 2)
        q = query.to(torch.float32).unsqueeze(1)  # [heads, 1, dim]
        scale = float(self.head_dim) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).squeeze(1)
        return out.to(self.dtype)
