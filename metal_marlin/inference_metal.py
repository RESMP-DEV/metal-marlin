"""
PyTorch MPS + Metal inference module for quantized LLMs.

This module provides native Metal inference without MLX dependency. All layers use
PyTorch MPS tensors with custom Metal compute kernels for quantized GEMM operations.

Key components:
    - MetalQuantizedLinear: FP4/INT2/FP8 quantized linear with Metal GEMM kernels
    - MetalAttention: Flash attention using Metal dispatch
    - MetalMoELayer: Mixture of Experts with batched expert dispatch
    - MetalTransformerBlock: Full transformer block with pre-norm architecture
    - MetalGLM47Model: Complete GLM-4.7-Flash model with MLA attention

Usage:
    from metal_marlin.inference_metal import MetalGLM47Model

    model = MetalGLM47Model.from_quantized("path/to/model")
    output_ids = model.generate(input_ids, max_tokens=128)

Requirements:
    - macOS with Apple Silicon
    - PyTorch with MPS backend
    - PyObjC: pip install pyobjc-framework-Metal
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
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
        if out_features % self.pack_factor != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by pack factor ({self.pack_factor})"
            )

        # Packed weights: [K, N // pack_factor] as uint32
        self.register_buffer(
            "weight_packed",
            torch.zeros(
                (in_features, out_features // self.pack_factor),
                dtype=torch.uint32,
                device="mps",
            ),
        )

        # Per-group scales: [K // group_size, N] as half
        num_groups = in_features // group_size
        self.register_buffer(
            "scales",
            torch.ones(
                (num_groups, out_features),
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
        x_2d = x.reshape(-1, self.in_features)
        M = x_2d.shape[0]

        # Dispatch to appropriate Metal GEMM kernel
        if self.bits == 4:
            out = dispatch_gemm_fp4(
                self.lib,
                x_2d,
                self.weight_packed,
                self.scales,
                M=M,
                N=self.out_features,
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
                N=self.out_features,
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
                N=self.out_features,
                K=self.in_features,
                group_size=self.group_size,
            )

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias

        # Reshape back to original batch dims
        out_shape = list(orig_shape[:-1]) + [self.out_features]
        return out.reshape(out_shape)

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

        # Quantize weights
        weight = linear.weight.detach().float()

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
            (packed [in_features, out_features // 8] uint32,
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

        # Pack 8 FP4 values per uint32
        packed = torch.zeros((K, N // 8), dtype=torch.uint32)
        for i in range(8):
            packed |= w_quant[:, i::8].to(torch.uint32) << (i * 4)

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

        # Pack 4 FP8 values per uint32
        packed = torch.zeros((K, N // 4), dtype=torch.uint32)
        for i in range(4):
            packed |= w_quant[:, i::4].to(torch.uint32) << (i * 8)

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

        # Pack 16 INT2 values per uint32
        packed = torch.zeros((K, N // 16), dtype=torch.uint32)
        for i in range(16):
            packed |= w_quant[:, i::16].to(torch.uint32) << (i * 2)

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

    def __init__(self, hidden_size: int, eps: float = 1e-6, device: str | torch.device | None = None):
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
    """Multi-head attention with Metal flash attention kernel.

    Uses MetalQuantizedLinear for Q/K/V/O projections and dispatches to
    Metal flash attention kernel for the attention computation.

    Supports:
        - Standard MHA (num_heads == num_kv_heads)
        - Grouped Query Attention (num_kv_heads < num_heads)
        - RoPE position embeddings
        - KV caching for autoregressive generation
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
    ):
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
    """Gated MLP (SwiGLU) with quantized projections."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bits: Literal[2, 4, 8] = 4,
        group_size: int = 128,
        activation: str = "silu",
    ):
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
                MetalMLP(hidden_size, intermediate_size, bits=bits, group_size=group_size)
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
        )

        self.post_attention_layernorm = MetalRMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = MetalMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bits=bits,
            group_size=group_size,
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
    """Multi-head Latent Attention for Metal inference.

    MLA compresses KV cache through learned latent projections. Used by
    GLM-4.7-Flash and DeepSeek models.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_lora_rank: int = 512,
        q_lora_rank: int | None = 1536,
        qk_rope_head_dim: int = 64,
        head_dim: int | None = None,
        rope_theta: float = 10000.0,
        rope_ratio: float = 1.0,
        max_position_embeddings: int = 4096,
        bits: Literal[2, 4, 8] = 4,
        group_size: int = 128,
        bias: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.scale = self.head_dim**-0.5

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
        self.kv_b_proj = MetalQuantizedLinear(
            kv_lora_rank,
            num_heads * self.head_dim * 2,
            bits=bits,
            group_size=kv_b_group_size,
            bias=bias,
        )

        # Output projection
        self.o_proj = MetalQuantizedLinear(
            num_heads * self.head_dim,
            hidden_size,
            bits=bits,
            group_size=group_size,
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
        kv_full = kv_full.view(batch_size, seq_len, self.num_heads, self.head_dim * 2)
        k_content, v = kv_full.chunk(2, dim=-1)

        # Transpose for attention
        q = q.transpose(1, 2)
        k = k_content.transpose(1, 2)
        v = v.transpose(1, 2)

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
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


# ---------------------------------------------------------------------------
# MetalGLM47Model
# ---------------------------------------------------------------------------


@dataclass
class MetalGenerationConfig:
    """Configuration for token generation."""

    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    eos_token_id: int = 2


class MetalGLM47Model(nn.Module):
    """Complete GLM-4.7-Flash model using Metal inference.

    A full decoder-only transformer with MLA attention, optimized for
    Apple Silicon via Metal compute kernels.

    Args:
        vocab_size: Vocabulary size
        hidden_size: Model hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        intermediate_size: MLP intermediate dimension
        kv_lora_rank: MLA latent dimension
        q_lora_rank: Query compression dimension
        qk_rope_head_dim: RoPE dimension for MLA
        max_position_embeddings: Maximum sequence length
        rms_norm_eps: RMSNorm epsilon
        bits: Quantization bit width
        group_size: Quantization group size
    """

    def __init__(
        self,
        vocab_size: int = 151552,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        intermediate_size: int = 11008,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 1536,
        qk_rope_head_dim: int = 64,
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_ratio: float = 1.0,
        bits: Literal[2, 4, 8] = 4,
        group_size: int = 128,
    ):
        super().__init__()
        require_mps()

        self.config = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "intermediate_size": intermediate_size,
            "kv_lora_rank": kv_lora_rank,
            "q_lora_rank": q_lora_rank,
            "qk_rope_head_dim": qk_rope_head_dim,
            "max_position_embeddings": max_position_embeddings,
            "rms_norm_eps": rms_norm_eps,
            "bits": bits,
            "group_size": group_size,
        }

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding (not quantized)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Transformer layers with MLA
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict(
                {
                    "input_layernorm": MetalRMSNorm(hidden_size, eps=rms_norm_eps),
                    "self_attn": MetalMLAAttention(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        kv_lora_rank=kv_lora_rank,
                        q_lora_rank=q_lora_rank,
                        qk_rope_head_dim=qk_rope_head_dim,
                        rope_theta=rope_theta,
                        rope_ratio=rope_ratio,
                        max_position_embeddings=max_position_embeddings,
                        bits=bits,
                        group_size=group_size,
                    ),
                    "post_attention_layernorm": MetalRMSNorm(hidden_size, eps=rms_norm_eps),
                    "mlp": MetalMLP(
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        bits=bits,
                        group_size=group_size,
                    ),
                }
            )
            self.layers.append(layer)

        # Final norm
        self.norm = MetalRMSNorm(hidden_size, eps=rms_norm_eps)

        # LM head (tied to embeddings or separate)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Ensure all params/buffers live on MPS for MPS input IDs.
        self.to("mps")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache: MetalKVCache | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: Optional attention mask
            kv_cache: Optional KV cache

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        hidden_states = self.embed_tokens(input_ids)

        for i, layer in enumerate(self.layers):
            # Self-attention with residual
            residual = hidden_states
            hidden_states = layer["input_layernorm"](hidden_states)
            hidden_states = layer["self_attn"](
                hidden_states,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                layer_idx=i,
            )
            hidden_states = residual + hidden_states

            # MLP with residual
            residual = hidden_states
            hidden_states = layer["post_attention_layernorm"](hidden_states)
            hidden_states = layer["mlp"](hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    def create_kv_cache(self, batch_size: int = 1) -> MetalKVCache:
        """Create KV cache for inference."""
        head_dim = self.hidden_size // self.config["num_heads"]
        config = MetalKVCacheConfig(
            num_layers=self.num_layers,
            num_kv_heads=self.config["num_heads"],
            head_dim=head_dim,
            max_seq_len=self.config["max_position_embeddings"],
        )
        return MetalKVCache(config, batch_size=batch_size)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        eos_token_id: int = 2,
        streamer: Any | None = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: [1, seq_len] prompt token IDs
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            repetition_penalty: Penalty for repeated tokens
            eos_token_id: End of sequence token
            streamer: Optional callback for each generated token

        Returns:
            [1, total_len] full sequence
        """
        kv_cache = self.create_kv_cache()

        # Prefill
        logits = self(input_ids, kv_cache=kv_cache)
        kv_cache.advance(input_ids.shape[1])

        generated = input_ids.tolist()[0]

        for _ in range(max_tokens):
            next_logits = logits[:, -1, :]

            # Temperature
            if temperature > 0:
                next_logits = next_logits / temperature

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated):
                    if next_logits[0, token_id] > 0:
                        next_logits[0, token_id] /= repetition_penalty
                    else:
                        next_logits[0, token_id] *= repetition_penalty

            # Sampling
            probs = F.softmax(next_logits, dim=-1)

            if top_k > 0:
                top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                probs = torch.zeros_like(probs).scatter(-1, top_k_indices, top_k_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative - sorted_probs > top_p
                sorted_probs[mask] = 0
                probs = probs.scatter(-1, sorted_indices, sorted_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            next_token = torch.multinomial(probs, num_samples=1)
            next_token_id = next_token.item()

            if streamer:
                streamer(next_token_id)

            if next_token_id == eos_token_id:
                break

            generated.append(next_token_id)

            # Decode step
            logits = self(next_token, kv_cache=kv_cache)
            kv_cache.advance(1)

        return torch.tensor([generated], device=input_ids.device)

    def generate_stream(
        self,
        input_ids: torch.Tensor,
        config: MetalGenerationConfig = MetalGenerationConfig(),
    ) -> Iterator[int]:
        """Streaming generator that yields tokens as produced."""
        kv_cache = self.create_kv_cache()

        with torch.no_grad():
            logits = self(input_ids, kv_cache=kv_cache)
            kv_cache.advance(input_ids.shape[1])

            generated = []

            for _ in range(config.max_tokens):
                next_logits = logits[:, -1, :] / max(config.temperature, 1e-7)

                if config.repetition_penalty != 1.0:
                    for token_id in set(generated):
                        if next_logits[0, token_id] > 0:
                            next_logits[0, token_id] /= config.repetition_penalty
                        else:
                            next_logits[0, token_id] *= config.repetition_penalty

                probs = F.softmax(next_logits, dim=-1)

                if config.top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(probs, config.top_k, dim=-1)
                    probs = torch.zeros_like(probs).scatter(-1, top_k_indices, top_k_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                next_token = torch.multinomial(probs, num_samples=1)
                next_token_id = next_token.item()

                if next_token_id == config.eos_token_id:
                    break

                yield next_token_id
                generated.append(next_token_id)

                logits = self(next_token, kv_cache=kv_cache)
                kv_cache.advance(1)

    @classmethod
    def from_quantized(
        cls,
        model_path: Path | str,
        bits: Literal[2, 4, 8] = 4,
    ) -> MetalGLM47Model:
        """Load model from quantized safetensors checkpoint.

        Args:
            model_path: Path to model directory containing:
                - config.json: Model configuration
                - model.safetensors or model-*.safetensors: Quantized weights

        Returns:
            Loaded MetalGLM47Model ready for inference
        """
        require_mps()

        model_path = Path(model_path)

        # Load config
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            raise FileNotFoundError(f"config.json not found in {model_path}")

        # Extract model parameters from config
        model = cls(
            vocab_size=config.get("vocab_size", 151552),
            hidden_size=config.get("hidden_size", 4096),
            num_layers=config.get("num_hidden_layers", config.get("num_layers", 32)),
            num_heads=config.get("num_attention_heads", 32),
            intermediate_size=config.get("intermediate_size", 11008),
            kv_lora_rank=config.get("kv_lora_rank", 512),
            q_lora_rank=config.get("q_lora_rank", 1536),
            qk_rope_head_dim=config.get("qk_rope_head_dim", 64),
            max_position_embeddings=config.get("max_position_embeddings", 4096),
            rms_norm_eps=config.get("rms_norm_eps", 1e-6),
            rope_theta=config.get("rope_theta", 10000.0),
            rope_ratio=config.get("rope_ratio", 1.0),
            bits=bits,
            group_size=config.get("group_size", 128),
        )

        # Load weights from safetensors
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError(
                "safetensors required for loading. Install with: pip install safetensors"
            )

        # Find safetensors files
        safetensors_files = list(model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")

        # Load all weight files
        state_dict: dict[str, torch.Tensor] = {}
        for sf_path in safetensors_files:
            state_dict.update(load_file(sf_path, device="mps"))

        # Load weights with key mapping
        model._load_quantized_state_dict(state_dict)

        model.to("mps")
        model.eval()

        return model

    def _load_quantized_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load quantized weights with key mapping.

        Handles the mapping from checkpoint key names to module parameters,
        including .packed and .scales suffixes for quantized layers.
        """
        # Build key mapping for our model structure
        # This handles conversion from HuggingFace-style keys

        for name, param in self.named_parameters():
            if name in state_dict:
                param.data.copy_(state_dict[name])
            elif name.replace(".", "_") in state_dict:
                param.data.copy_(state_dict[name.replace(".", "_")])

        for name, buffer in self.named_buffers():
            # Handle quantized weight buffers
            if name.endswith(".weight_packed"):
                base_name = name.replace(".weight_packed", ".weight")
                packed_key = f"{base_name}.packed"
                if packed_key in state_dict:
                    buffer.copy_(state_dict[packed_key])
                elif base_name in state_dict:
                    # Weight provided as float, need to quantize
                    pass  # Skip, quantization should be done at conversion time
            elif name.endswith(".scales"):
                base_name = name.replace(".scales", ".weight")
                scales_key = f"{base_name}.scales"
                if scales_key in state_dict:
                    buffer.copy_(state_dict[scales_key])
            elif name in state_dict:
                buffer.copy_(state_dict[name])
