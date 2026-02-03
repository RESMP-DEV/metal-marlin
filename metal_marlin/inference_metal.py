"""
PyTorch MPS + Metal inference module for quantized LLMs.

This module provides native Metal inference without MLX dependency. All layers use
PyTorch MPS tensors with custom Metal compute kernels for quantized GEMM operations.

Key components:
    - MetalQuantizedLinear: FP4/INT2/FP8 quantized linear with Metal GEMM kernels
    - MetalAttention: Flash attention using Metal dispatch
    - MetalMoELayer: Mixture of Experts with batched expert dispatch
    - MetalTransformerBlock: Full transformer block with pre-norm architecture

Requirements:
    - macOS with Apple Silicon
    - PyTorch with MPS backend
    - PyObjC: pip install pyobjc-framework-Metal
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metal_dispatch import (
    MetalKernelLibrary,
    dispatch_gemm_fp4,
    dispatch_gemm_fp8,
    dispatch_gemm_int2,
    get_default_library,
    mps_tensor_to_metal_buffer,
)

# ---------------------------------------------------------------------------
# Check MPS availability
# ---------------------------------------------------------------------------


def require_mps() -> None:
    """Raise if PyTorch MPS is not available."""
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "Metal inference requires PyTorch with MPS backend.\n"
            "Ensure you're on Apple Silicon with PyTorch >= 2.0"
        )


def _mps_buffer(tensor: torch.Tensor, device: Any) -> Any:
    """Get a zero-copy MTLBuffer for an MPS tensor."""
    return mps_tensor_to_metal_buffer(tensor, device)


# ---------------------------------------------------------------------------
# E2M1 FP4 codebook for reference
# ---------------------------------------------------------------------------

_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float16,
)


# ---------------------------------------------------------------------------
# MetalQuantizedLinear
# ---------------------------------------------------------------------------


class MetalQuantizedLinear(nn.Module):
    """FP4/INT2/FP8 quantized linear layer using Metal GEMM kernels.

    Stores weights in packed format with per-group scales. Forward pass
    dispatches to fused dequant-GEMM Metal kernels for maximum efficiency.

    The kernel selection is based on `bits`:
        - bits=4: FP4 E2M1 quantization (default)
        - bits=8: FP8 E4M3 quantization
        - bits=2: INT2 extreme compression (for cold MoE experts)

    Args:
        in_features: Input dimension (K)
        out_features: Output dimension (N)
        bits: Quantization bit width (2, 4, or 8)
        group_size: Elements per quantization group
        bias: Whether to include bias term
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: Literal[2, 4, 8] = 4,
        group_size: int = 128,
        bias: bool = False,
    ):
        super().__init__()
        require_mps()

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        # Compute pack factor based on bit width
        if bits == 4:
            self.pack_factor = 8  # 8 FP4 values per uint32
        elif bits == 8:
            self.pack_factor = 4  # 4 FP8 values per uint32
        elif bits == 2:
            self.pack_factor = 16  # 16 INT2 values per uint32
        else:
            raise ValueError(f"Unsupported bits: {bits}. Use 2, 4, or 8.")

        # Validate dimensions
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by group_size ({group_size})"
            )
        if in_features % self.pack_factor != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by pack_factor ({self.pack_factor})"
            )

        # Pad out_features to next multiple of pack_factor if needed (for dispatch padding)
        self._needs_output_slice = out_features % self.pack_factor != 0
        self._padded_out_features = (
            ((out_features + self.pack_factor - 1) // self.pack_factor) * self.pack_factor
            if self._needs_output_slice
            else out_features
        )

        # Packed weights: [K // pack_factor, N_padded] as uint32 (K-dim packing)
        # This matches the kernel expectation: B_packed shape [(K+pad)//8, N+pad]
        self.register_buffer(
            "weight_packed",
            torch.zeros(
                (in_features // self.pack_factor, self._padded_out_features),
                dtype=torch.uint32,
                device="mps",
            ),
        )

        # Per-group scales: [K // group_size, N_padded] as half
        num_groups = in_features // group_size
        self.register_buffer(
            "scales",
            torch.ones(
                (num_groups, self._padded_out_features),
                dtype=torch.float16,
                device="mps",
            ),
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features, dtype=torch.float16, device="mps"),
            )
        else:
            self.register_parameter("bias", None)

        # Cache the Metal library reference
        self._lib: MetalKernelLibrary | None = None

    @property
    def lib(self) -> MetalKernelLibrary:
        """Lazily get Metal kernel library."""
        if self._lib is None:
            self._lib = get_default_library()
        return self._lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: fused dequant + GEMM via Metal kernel.

        Args:
            x: Input tensor [*, in_features], must be on MPS device

        Returns:
            Output tensor [*, out_features] as float16
        """
        # Handle batched input
        orig_shape = x.shape
        orig_dtype = x.dtype
        x_2d = x.reshape(-1, self.in_features)
        M = x_2d.shape[0]

        # Dispatch to appropriate Metal GEMM kernel (use padded dimensions)
        if self.bits == 4:
            out = dispatch_gemm_fp4(
                self.lib,
                x_2d,
                self.weight_packed,
                self.scales,
                M=M,
                N=self._padded_out_features,
                K=self.in_features,
                group_size=self.group_size,
            )
        elif self.bits == 8:
            out = dispatch_gemm_fp8(
                self.lib,
                x_2d,
                self.weight_packed,
                self.scales,
                M=M,
                N=self._padded_out_features,
                K=self.in_features,
                group_size=self.group_size,
            )
        elif self.bits == 2:
            out = dispatch_gemm_int2(
                self.lib,
                x_2d,
                self.weight_packed,
                self.scales,
                M=M,
                N=self._padded_out_features,
                K=self.in_features,
                group_size=self.group_size,
            )

        # Slice to original out_features if we used padding
        if self._needs_output_slice:
            out = out[..., : self.out_features]

        # Add bias if present - use in-place to avoid MPS validation error
        if self.bias is not None:
            out.add_(self.bias)

        # Reshape back to original batch dims
        out_shape = list(orig_shape[:-1]) + [self.out_features]
        out = out.reshape(out_shape)

        # Preserve input dtype (Metal kernels output float16, but model may use bfloat16)
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)

        return out

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        bits: Literal[2, 4, 8] = 4,
        group_size: int = 128,
    ) -> MetalQuantizedLinear:
        """Quantize a float linear layer to Metal format.

        Args:
            linear: Source nn.Linear layer
            bits: Target bit width
            group_size: Quantization group size

        Returns:
            Quantized MetalQuantizedLinear layer
        """
        require_mps()

        out_features, in_features = linear.weight.shape
        has_bias = linear.bias is not None

        layer = cls(
            in_features=in_features,
            out_features=out_features,
            bits=bits,
            group_size=group_size,
            bias=has_bias,
        )

        # Quantize weights - pad to _padded_out_features if needed
        weight = linear.weight.detach().float()
        if layer._needs_output_slice:
            pad_cols = layer._padded_out_features - out_features
            weight = torch.nn.functional.pad(weight, (0, 0, 0, pad_cols), value=0.0)

        if bits == 4:
            packed, scales = cls._quantize_fp4(weight, group_size)
        elif bits == 8:
            packed, scales = cls._quantize_fp8(weight, group_size)
        elif bits == 2:
            packed, scales = cls._quantize_int2(weight, group_size)

        layer.weight_packed.copy_(packed.to("mps"))
        layer.scales.copy_(scales.to("mps"))

        if has_bias:
            layer.bias.copy_(linear.bias.half().to("mps"))

        return layer

    @staticmethod
    def _quantize_fp4(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize weight to FP4 E2M1 format.

        Args:
            weight: [out_features, in_features] float tensor
            group_size: Quantization group size along in_features

        Returns:
            (packed [in_features // 8, out_features] uint32 (K-dim packing),
             scales [in_features // group_size, out_features] float16)
        """
        out_features, in_features = weight.shape

        # Transpose to [K, N] for kernel layout
        w = weight.T.contiguous()  # [in_features, out_features]

        # Reshape for per-group quantization
        K, N = w.shape
        n_groups = K // group_size
        w_grouped = w.reshape(n_groups, group_size, N)

        # Per-group absmax scaling
        abs_max = w_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scales = abs_max.squeeze(1) / 6.0  # FP4 E2M1 max is 6.0

        # Scale and quantize
        w_scaled = w_grouped / abs_max * 6.0
        w_scaled = w_scaled.clamp(-6.0, 6.0)

        # Map to FP4 codebook indices (simplified linear quantization)
        # Real implementation would use proper E2M1 rounding
        w_quant = torch.round(w_scaled * 2.0).to(torch.int8) + 8
        w_quant = w_quant.clamp(0, 15).to(torch.uint8)

        # Reshape back to [K, N]
        w_quant = w_quant.reshape(K, N)

        # Pack 8 FP4 values per uint32 along K dimension -> [K // 8, N]
        packed = torch.zeros((K // 8, N), dtype=torch.uint32)
        for i in range(8):
            packed |= w_quant[i::8, :].to(torch.uint32) << (i * 4)

        return packed, scales.to(torch.float16)

    @staticmethod
    def _quantize_fp8(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize weight to FP8 E4M3 format."""
        out_features, in_features = weight.shape

        w = weight.T.contiguous()
        K, N = w.shape
        n_groups = K // group_size
        w_grouped = w.reshape(n_groups, group_size, N)

        abs_max = w_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scales = abs_max.squeeze(1) / 448.0  # E4M3 max is 448

        w_scaled = w_grouped / abs_max * 448.0
        w_scaled = w_scaled.clamp(-448.0, 448.0)

        # Linear quantization to uint8 centered at 128
        w_quant = torch.round(w_scaled / 448.0 * 127.0).to(torch.int8) + 128
        w_quant = w_quant.clamp(0, 255).to(torch.uint8)
        w_quant = w_quant.reshape(K, N)

        # Pack 4 FP8 values per uint32 along K dimension -> [K // 4, N]
        packed = torch.zeros((K // 4, N), dtype=torch.uint32)
        for i in range(4):
            packed |= w_quant[i::4, :].to(torch.uint32) << (i * 8)

        return packed, scales.to(torch.float16)

    @staticmethod
    def _quantize_int2(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize weight to INT2 format (extreme compression)."""
        out_features, in_features = weight.shape

        w = weight.T.contiguous()
        K, N = w.shape
        n_groups = K // group_size
        w_grouped = w.reshape(n_groups, group_size, N)

        abs_max = w_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scales = abs_max.squeeze(1) / 1.5  # INT2 symmetric: {-1.5, -0.5, 0.5, 1.5}

        w_scaled = w_grouped / abs_max * 1.5
        w_scaled = w_scaled.clamp(-1.5, 1.5)

        # Map to {0, 1, 2, 3} representing {-1.5, -0.5, 0.5, 1.5}
        w_quant = torch.round((w_scaled + 1.5) / 1.0).to(torch.uint8)
        w_quant = w_quant.clamp(0, 3)
        w_quant = w_quant.reshape(K, N)

        # Pack 16 INT2 values per uint32 along K dimension -> [K // 16, N]
        packed = torch.zeros((K // 16, N), dtype=torch.uint32)
        for i in range(16):
            packed |= w_quant[i::16, :].to(torch.uint32) << (i * 2)

        return packed, scales.to(torch.float16)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bits={self.bits}, "
            f"group_size={self.group_size}, "
            f"bias={self.bias is not None}"
        )


# ---------------------------------------------------------------------------
# RMSNorm for Metal
# ---------------------------------------------------------------------------


class MetalRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for MPS.

    Normalizes by RMS value and applies learned scale. More efficient than
    LayerNorm since it skips mean centering.
    """

    def __init__(
        self, hidden_size: int, eps: float = 1e-6, device: str | torch.device | None = None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


# ---------------------------------------------------------------------------
# RoPE for Metal
# ---------------------------------------------------------------------------


class MetalRoPE(nn.Module):
    """Rotary Position Embedding for MPS tensors.

    Supports rope_ratio scaling for GLM-style frequency adjustment.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        rope_ratio: float = 1.0,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.rope_ratio = rope_ratio
        self.max_seq_len = max_seq_len

        # Precompute inverse frequencies with rope_ratio scaling
        half_dim = dim // 2
        inv_freq = rope_ratio / (base ** (torch.arange(0, half_dim).float() * 2 / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin cache
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos_cache", freqs.cos().half())
        self.register_buffer("sin_cache", freqs.sin().half())

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        """Apply RoPE to input tensor.

        Args:
            x: [..., seq_len, dim] tensor
            position_offset: Starting position for cache lookup

        Returns:
            Tensor with RoPE applied
        """
        seq_len = x.shape[-2]
        positions = torch.arange(position_offset, position_offset + seq_len, device=x.device)

        cos = self.cos_cache[positions]  # [seq_len, dim/2]
        sin = self.sin_cache[positions]

        # Expand for broadcasting
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        # Split and rotate
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_odd * cos + x_even * sin

        # Interleave back
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        return x_rotated.reshape(x.shape)


# ---------------------------------------------------------------------------
# MetalKVCache
# ---------------------------------------------------------------------------


@dataclass
class MetalKVCacheConfig:
    """Configuration for Metal KV cache."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    dtype: torch.dtype = torch.float16


class MetalKVCache:
    """KV cache for Metal inference using MPS tensors."""

    def __init__(self, config: MetalKVCacheConfig, batch_size: int = 1):
        require_mps()

        self.config = config
        self.batch_size = batch_size
        self.seq_len = 0

        cache_shape = (
            batch_size,
            config.num_kv_heads,
            config.max_seq_len,
            config.head_dim,
        )

        self.k_cache = [
            torch.zeros(cache_shape, dtype=config.dtype, device="mps")
            for _ in range(config.num_layers)
        ]
        self.v_cache = [
            torch.zeros(cache_shape, dtype=config.dtype, device="mps")
            for _ in range(config.num_layers)
        ]

    def update(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new K, V and return full cached K, V."""
        new_seq_len = k_new.shape[2]
        end_pos = self.seq_len + new_seq_len

        if end_pos > self.config.max_seq_len:
            raise ValueError(f"Sequence length {end_pos} exceeds max {self.config.max_seq_len}")

        # Store new values
        self.k_cache[layer_idx][:, :, self.seq_len : end_pos, :] = k_new
        self.v_cache[layer_idx][:, :, self.seq_len : end_pos, :] = v_new

        # Return full sequence
        return (
            self.k_cache[layer_idx][:, :, :end_pos, :],
            self.v_cache[layer_idx][:, :, :end_pos, :],
        )

    def advance(self, num_tokens: int = 1) -> None:
        """Advance sequence position."""
        self.seq_len += num_tokens

    def reset(self) -> None:
        """Reset cache for new sequence."""
        self.seq_len = 0


# ---------------------------------------------------------------------------
# MetalAttention
# ---------------------------------------------------------------------------


class MetalAttention(nn.Module):
    """
    DEPRECATED: Attention implementations should come from Transformers.

    Metal Marlin's value is in MetalQuantizedLinear, not attention patterns.
    The attention mechanism (how Q, K, V interact) is architecture-specific
    and already implemented correctly in Transformers.

    What we optimize:
    - Q/K/V projections: nn.Linear -> MetalQuantizedLinear
    - Output projection: nn.Linear -> MetalQuantizedLinear

    What Transformers handles:
    - Attention pattern (standard, sliding window, MLA, etc.)
    - Position embeddings (RoPE, ALiBi, etc.)
    - KV cache management
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        bits: Literal[2, 4, 8] = 4,
        group_size: int = 128,
        rope_theta: float = 10000.0,
        rope_ratio: float = 1.0,
        max_position_embeddings: int = 4096,
        bias: bool = False,
        warn_if_standalone: bool = True,
    ):
        if warn_if_standalone:
            warnings.warn(
                "MetalAttention is deprecated. Model architecture should come from "
                "Transformers. Use replace_linear_layers() to quantize projections.",
                DeprecationWarning,
                stacklevel=2,
            )
        super().__init__()
        require_mps()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = self.head_dim**-0.5

        # Quantized projections
        self.q_proj = MetalQuantizedLinear(
            hidden_size,
            num_heads * self.head_dim,
            bits=bits,
            group_size=group_size,
            bias=bias,
        )
        self.k_proj = MetalQuantizedLinear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            bits=bits,
            group_size=group_size,
            bias=bias,
        )
        self.v_proj = MetalQuantizedLinear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            bits=bits,
            group_size=group_size,
            bias=bias,
        )
        self.o_proj = MetalQuantizedLinear(
            num_heads * self.head_dim,
            hidden_size,
            bits=bits,
            group_size=group_size,
            bias=bias,
        )

        # RoPE
        self.rope = MetalRoPE(
            dim=self.head_dim,
            base=rope_theta,
            rope_ratio=rope_ratio,
            max_seq_len=max_position_embeddings,
        )

        self._lib: MetalKernelLibrary | None = None

    @property
    def lib(self) -> MetalKernelLibrary:
        if self._lib is None:
            self._lib = get_default_library()
        return self._lib

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache: MetalKVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional causal/padding mask
            kv_cache: Optional KV cache for autoregressive generation
            layer_idx: Layer index for KV cache

        Returns:
            Output [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape: [batch, seq, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose: [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        position_offset = kv_cache.seq_len if kv_cache else 0
        q = self.rope(q, position_offset)
        k = self.rope(k, position_offset)

        # Update KV cache
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        # Expand K, V for GQA
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # Scaled dot-product attention
        # Try to use flash attention if available, fall back to standard
        attn_output = self._attention(q, k, v, attention_mask)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        output = self.o_proj(attn_output)

        return output

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute attention. Uses PyTorch SDPA which dispatches to Metal."""
        # Ensure consistent dtypes (KV cache may store as float16 while q is float32)
        if q.dtype != k.dtype:
            q = q.to(k.dtype)

        # PyTorch 2.0+ scaled_dot_product_attention uses Metal on MPS
        is_causal = mask is None and q.shape[2] == k.shape[2] and q.shape[2] > 1

        if is_causal:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif mask is not None:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            return F.scaled_dot_product_attention(q, k, v)


# ---------------------------------------------------------------------------
# MetalMLP
# ---------------------------------------------------------------------------


class MetalMLP(nn.Module):
    """DEPRECATED: Use Transformers' MLP with MetalQuantizedLinear.

    Gated MLP (SwiGLU) with quantized projections.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bits: Literal[2, 4, 8] = 4,
        group_size: int = 128,
        activation: str = "silu",
        warn_if_standalone: bool = True,
    ):
        if warn_if_standalone:
            warnings.warn(
                "MetalMLP is deprecated. Use Transformers' MLP implementation "
                "and replace_linear_layers() to quantize Linear layers.",
                DeprecationWarning,
                stacklevel=2,
            )
        super().__init__()

        self.gate_proj = MetalQuantizedLinear(
            hidden_size,
            intermediate_size,
            bits=bits,
            group_size=group_size,
        )
        self.up_proj = MetalQuantizedLinear(
            hidden_size,
            intermediate_size,
            bits=bits,
            group_size=group_size,
        )
        self.down_proj = MetalQuantizedLinear(
            intermediate_size,
            hidden_size,
            bits=bits,
            group_size=group_size,
        )

        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)

        if self.activation == "silu":
            gate = F.silu(gate)
        elif self.activation == "gelu":
            gate = F.gelu(gate)
        elif self.activation == "relu":
            gate = F.relu(gate)

        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ---------------------------------------------------------------------------
# MetalMoELayer
# ---------------------------------------------------------------------------


class MetalMoELayer(nn.Module):
    """Mixture of Experts layer with Metal dispatch.

    Implements efficient batched expert execution by grouping tokens by
    their assigned expert and dispatching batched GEMMs per expert.

    Args:
        hidden_size: Model hidden dimension
        intermediate_size: MLP intermediate dimension per expert
        num_experts: Total number of experts
        top_k: Number of experts per token
        bits: Quantization bit width
        group_size: Quantization group size
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
        bits: Literal[2, 4, 8] = 4,
        group_size: int = 128,
    ):
        super().__init__()
        require_mps()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        # Expert MLPs
        self.experts = nn.ModuleList(
            [
                MetalMLP(
                    hidden_size,
                    intermediate_size,
                    bits=bits,
                    group_size=group_size,
                    warn_if_standalone=False,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with top-k expert routing.

        Args:
            hidden_states: [batch, seq, hidden_size]

        Returns:
            [batch, seq, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_size)  # [batch*seq, hidden]

        # Router logits
        router_logits = self.router(hidden_flat)  # [batch*seq, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Simple loop over experts (optimized version would batch by expert)
        # For each token, compute weighted sum of selected experts
        output = torch.zeros_like(hidden_flat)

        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)

            if not expert_mask.any():
                continue

            # Get tokens for this expert
            expert_input = hidden_flat[expert_mask]

            # Run expert
            expert_output = self.experts[expert_idx](expert_input)

            # Get weights for this expert
            weight_mask = top_k_indices[expert_mask] == expert_idx
            weights = (top_k_probs[expert_mask] * weight_mask.float()).sum(dim=-1, keepdim=True)

            # Accumulate weighted output
            output[expert_mask] += weights * expert_output

        return output.view(batch_size, seq_len, hidden_size)

    def forward_fused_gate_up(
        self,
        hidden_states: torch.Tensor,
        gate_up_packed: torch.Tensor,
        gate_up_scales: torch.Tensor,
        down_packed: torch.Tensor,
        down_scales: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for fused gate/up experts using MoE kernels.

        This path expects fused gate_up weights per expert (hidden -> 2*intermediate)
        and performs:
            gate_up = W_gate_up @ hidden
            gate, up = split(gate_up)
            hidden = silu(gate) * up
            output = W_down @ hidden
        """
        from .moe_ops import fused_moe_forward

        return fused_moe_forward(
            hidden_states,
            gate_up_packed,
            gate_up_scales,
            down_packed,
            down_scales,
            expert_ids,
            expert_probs,
        )


# ---------------------------------------------------------------------------
# MetalTransformerBlock
# ---------------------------------------------------------------------------


class MetalTransformerBlock(nn.Module):
    """Single transformer decoder block with pre-norm architecture.

    Structure:
        x -> RMSNorm -> Attention -> + -> RMSNorm -> MLP -> +

    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        intermediate_size: MLP intermediate dimension
        num_kv_heads: KV heads for GQA (default: same as num_heads)
        bits: Quantization bit width
        group_size: Quantization group size
        rms_norm_eps: Epsilon for RMSNorm
        rope_theta: Base frequency for RoPE
        max_position_embeddings: Maximum sequence length
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_kv_heads: int | None = None,
        bits: Literal[2, 4, 8] = 4,
        group_size: int = 128,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_ratio: float = 1.0,
        max_position_embeddings: int = 4096,
    ):
        super().__init__()

        self.input_layernorm = MetalRMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = MetalAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            bits=bits,
            group_size=group_size,
            rope_theta=rope_theta,
            rope_ratio=rope_ratio,
            max_position_embeddings=max_position_embeddings,
            warn_if_standalone=False,
        )

        self.post_attention_layernorm = MetalRMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = MetalMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bits=bits,
            group_size=group_size,
            warn_if_standalone=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache: MetalKVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Forward pass through the transformer block."""
        if self._needs_device_move(hidden_states.device):
            self.to(hidden_states.device)
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def _needs_device_move(self, device: torch.device) -> bool:
        param = next(self.parameters(), None)
        if param is None:
            return False
        return param.device != device


# ---------------------------------------------------------------------------
# MetalMLAAttention (for GLM-4.7-Flash)
# ---------------------------------------------------------------------------


class MetalMLAAttention(nn.Module):
    """Optional MLA implementation for custom optimizations.

    Transformers' MLA implementations work fine for most cases; use this
    class only when you need custom Metal-specific changes.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_lora_rank: int = 512,
        q_lora_rank: int | None = 1536,
        qk_rope_head_dim: int = 64,
        qk_nope_head_dim: int | None = None,
        v_head_dim: int | None = None,
        head_dim: int | None = None,
        rope_theta: float = 10000.0,
        rope_ratio: float = 1.0,
        max_position_embeddings: int = 4096,
        bits: Literal[2, 4, 8] = 4,
        group_size: int = 128,
        o_proj_group_size: int | None = None,
        bias: bool = False,
        warn_if_standalone: bool = True,
    ):
        _ = warn_if_standalone  # retained for backward compatibility
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # GLM-4 uses separate nope/rope dims; fallback to hidden_size/num_heads
        self.qk_nope_head_dim = qk_nope_head_dim or (hidden_size // num_heads - qk_rope_head_dim)
        self.v_head_dim = v_head_dim or (hidden_size // num_heads)
        # head_dim for queries = qk_nope + qk_rope
        self.head_dim = head_dim or (self.qk_nope_head_dim + qk_rope_head_dim)
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.scale = self.head_dim**-0.5

        # Use o_proj_group_size if provided; often checkpoint uses larger gs for o_proj
        if o_proj_group_size is None:
            o_proj_group_size = group_size

        def _compatible_group_size(in_features: int, default_group_size: int) -> int:
            if in_features % default_group_size == 0:
                return default_group_size
            return math.gcd(in_features, default_group_size) or 1

        # Query projections
        if q_lora_rank is not None:
            q_b_group_size = _compatible_group_size(q_lora_rank, group_size)
            self.q_a_proj = MetalQuantizedLinear(
                hidden_size,
                q_lora_rank,
                bits=bits,
                group_size=group_size,
                bias=bias,
            )
            self.q_b_proj = MetalQuantizedLinear(
                q_lora_rank,
                num_heads * self.head_dim,
                bits=bits,
                group_size=q_b_group_size,
                bias=bias,
            )
        else:
            self.q_proj = MetalQuantizedLinear(
                hidden_size,
                num_heads * self.head_dim,
                bits=bits,
                group_size=group_size,
                bias=bias,
            )

        # KV projections
        kv_a_out_dim = kv_lora_rank + qk_rope_head_dim
        self.kv_a_proj = MetalQuantizedLinear(
            hidden_size,
            kv_a_out_dim,
            bits=bits,
            group_size=group_size,
            bias=bias,
        )
        kv_b_group_size = _compatible_group_size(kv_lora_rank, group_size)
        # kv_b projects to k_nope + v for each head
        kv_b_out_dim = num_heads * (self.qk_nope_head_dim + self.v_head_dim)
        self.kv_b_proj = MetalQuantizedLinear(
            kv_lora_rank,
            kv_b_out_dim,
            bits=bits,
            group_size=kv_b_group_size,
            bias=bias,
        )

        # Output projection - input is num_heads * v_head_dim
        # Uses separate o_proj_group_size (checkpoint often uses larger gs for o_proj)
        o_proj_in_dim = num_heads * self.v_head_dim
        actual_o_proj_group_size = _compatible_group_size(o_proj_in_dim, o_proj_group_size)
        self.o_proj = MetalQuantizedLinear(
            o_proj_in_dim,
            hidden_size,
            bits=bits,
            group_size=actual_o_proj_group_size,
            bias=bias,
        )

        # RoPE for Q and k_pe
        self.rope_q = MetalRoPE(
            dim=self.head_dim,
            base=rope_theta,
            rope_ratio=rope_ratio,
            max_seq_len=max_position_embeddings,
        )
        self.rope_k = MetalRoPE(
            dim=qk_rope_head_dim,
            base=rope_theta,
            rope_ratio=rope_ratio,
            max_seq_len=max_position_embeddings,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache: Any | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Query path
        if self.q_lora_rank is not None:
            q_latent = self.q_a_proj(hidden_states)
            q = self.q_b_proj(q_latent)
        else:
            q = self.q_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # KV path
        kv_compressed = self.kv_a_proj(hidden_states)
        c_kv = kv_compressed[..., : self.kv_lora_rank]
        k_pe = kv_compressed[..., self.kv_lora_rank :]

        # RoPE
        position_offset = kv_cache.seq_len if kv_cache else 0
        q = self.rope_q(q, position_offset)
        k_pe = self.rope_k(k_pe.unsqueeze(2), position_offset).squeeze(2)

        # Decompress KV
        kv_full = self.kv_b_proj(c_kv)
        # kv_b outputs [qk_nope_head_dim + v_head_dim] per head
        kv_dim_per_head = self.qk_nope_head_dim + self.v_head_dim
        kv_full = kv_full.view(batch_size, seq_len, self.num_heads, kv_dim_per_head)
        # Split into k_nope and v
        k_nope = kv_full[..., : self.qk_nope_head_dim]
        v = kv_full[..., self.qk_nope_head_dim :]

        # Concatenate k_nope with k_pe to form full key [qk_nope + qk_rope = head_dim]
        # k_pe shape: [B, S, qk_rope] -> expand to [B, S, H, qk_rope]
        k_pe_expanded = k_pe.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        k = torch.cat([k_nope, k_pe_expanded], dim=-1)

        # Transpose for attention
        q = q.transpose(1, 2)  # [B, H, S, head_dim]
        k = k.transpose(1, 2)  # [B, H, S, head_dim]
        v = v.transpose(1, 2)  # [B, H, S, v_head_dim]

        # Attention
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=attention_mask is None and seq_len > 1,
        )

        # Output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Output dimension is num_heads * v_head_dim
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.v_head_dim)
        return self.o_proj(attn_output)


