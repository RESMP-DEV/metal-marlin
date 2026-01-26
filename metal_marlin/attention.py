"""
Multi-head attention layer with Marlin-quantized projections.

Uses MarlinLinear for Q/K/V/O projections with FP4 quantized weights.
Supports Grouped Query Attention (GQA) where num_kv_heads < num_heads.

This implementation uses PyTorch with MPS device for Apple Silicon,
and delegates attention computation to attention_metal functions.

Usage:
    from metal_marlin.python.attention import MarlinAttention

    attn = MarlinAttention(
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,  # GQA
    )
    output = attn(hidden_states, kv_cache=cache, layer_idx=0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kv_cache import KVCache
from .layers import MarlinLinear

if TYPE_CHECKING:
    pass


def _get_device() -> torch.device:
    """Get the appropriate device (MPS on Apple Silicon, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RoPE(nn.Module):
    """Rotary Position Embedding."""

    def __init__(self, dims: int, traditional: bool = False, base: float = 10000.0):
        super().__init__()
        self.dims = dims
        self.traditional = traditional
        self.base = base

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0,
    ) -> torch.Tensor:
        """Apply RoPE to input tensor.

        Args:
            x: Input tensor [batch, num_heads, seq_len, head_dim]
            offset: Position offset for KV cache

        Returns:
            Tensor with RoPE applied
        """
        shape = x.shape
        seq_len = shape[2]
        head_dim = shape[3]
        device = x.device
        dtype = x.dtype

        # Compute position indices
        positions = torch.arange(offset, offset + seq_len, dtype=torch.float32, device=device)

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dims, 2, dtype=torch.float32, device=device) / self.dims)
        )

        # Compute angles: [seq_len, dims/2]
        freqs = torch.outer(positions, inv_freq)

        # Compute cos and sin
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # Expand dims for broadcasting: [1, 1, seq_len, dims/2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Split into even and odd indices
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # Apply rotation
        if self.traditional:
            # Traditional RoPE
            x_rotated_even = x_even * cos - x_odd * sin
            x_rotated_odd = x_even * sin + x_odd * cos
        else:
            # Llama-style RoPE (half-rotation)
            x_rotated_even = x_even * cos - x_odd * sin
            x_rotated_odd = x_odd * cos + x_even * sin

        # Interleave back
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        x_rotated = x_rotated.view(*shape[:-1], head_dim)

        return x_rotated.to(dtype)


class MarlinAttention(nn.Module):
    """
    Multi-head attention with Marlin-quantized projections.

    Supports:
    - Standard MHA (num_heads == num_kv_heads)
    - Grouped Query Attention (num_kv_heads < num_heads)
    - KV caching for efficient autoregressive generation
    - RoPE position embeddings
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        quant_type: str = "fp4",
        group_size: int = 128,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 4096,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = self.head_dim**-0.5

        # Quantized projections
        self.q_proj = MarlinLinear(
            hidden_size,
            num_heads * self.head_dim,
            bias=bias,
            quant_type=quant_type,
            group_size=group_size,
        )
        self.k_proj = MarlinLinear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=bias,
            quant_type=quant_type,
            group_size=group_size,
        )
        self.v_proj = MarlinLinear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=bias,
            quant_type=quant_type,
            group_size=group_size,
        )
        self.o_proj = MarlinLinear(
            num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            quant_type=quant_type,
            group_size=group_size,
        )

        # RoPE embeddings
        self.rope = RoPE(self.head_dim, base=rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, hidden_size]
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Optional position IDs for RoPE
            kv_cache: Optional KV cache for autoregressive generation
            layer_idx: Layer index for KV cache

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V projections using Marlin kernels
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        position_offset = kv_cache.seq_len if kv_cache else 0
        q = self.rope(q, offset=position_offset)
        k = self.rope(k, offset=position_offset)

        # Update KV cache
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        # Expand K, V for GQA
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # Use Metal-optimized attention via scaled_dot_product_attention
        # This will use the MPS backend's optimized attention implementation
        attn_output = scaled_dot_product_attention_metal(
            q,
            k,
            v,
            attn_mask=attention_mask,
            scale=self.scale,
            is_causal=attention_mask is None,
        )

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.head_dim)
        )
        output = self.o_proj(attn_output)

        return output


def scaled_dot_product_attention_metal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Scaled dot-product attention using PyTorch's native implementation.

    On MPS devices, this uses the Metal-optimized SDPA implementation.
    Falls back to PyTorch's native implementation on other devices.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, kv_seq_len, head_dim]
        v: Value tensor [batch, num_heads, kv_seq_len, head_dim]
        attn_mask: Optional attention mask [batch, num_heads, seq_len, kv_seq_len]
        scale: Optional scale factor. If None, uses 1/sqrt(head_dim)
        is_causal: If True, applies causal masking

    Returns:
        Attention output [batch, num_heads, seq_len, head_dim]
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    # Use PyTorch's scaled_dot_product_attention which has MPS optimization
    # This automatically selects the best implementation for the current device
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
    )


def flash_attention_metal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    causal: bool = True,
    num_kv_heads: int | None = None,
) -> torch.Tensor:
    """
    Flash attention implementation for Metal/MPS devices.

    This function provides a unified API for attention computation that
    can be called from ONNX converters and other components.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_kv_heads, kv_seq_len, head_dim]
        v: Value tensor [batch, num_kv_heads, kv_seq_len, head_dim]
        scale: Optional scale factor. If None, uses 1/sqrt(head_dim)
        causal: Whether to apply causal masking
        num_kv_heads: Number of KV heads for GQA. If provided and different
            from num_heads, K and V will be repeated accordingly.

    Returns:
        Attention output [batch, num_heads, seq_len, head_dim]
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    num_heads = q.shape[1]

    # Handle GQA by repeating K/V heads
    if num_kv_heads is not None and num_kv_heads < num_heads:
        repeat_factor = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

    return scaled_dot_product_attention_metal(
        q,
        k,
        v,
        scale=scale,
        is_causal=causal,
    )


def create_causal_mask(
    seq_len: int,
    kv_seq_len: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    """
    Create causal attention mask.

    Args:
        seq_len: Query sequence length
        kv_seq_len: Key/value sequence length (defaults to seq_len)
        device: Device to create the mask on

    Returns:
        Causal mask with -inf for masked positions, or None for single-token decode
    """
    kv_seq_len = kv_seq_len or seq_len

    # Single token decode - no masking needed
    if seq_len == 1:
        return None

    if device is None:
        device = _get_device()

    # Prefill: standard causal mask
    # Create upper triangular mask with -inf for positions that should be masked
    mask = torch.triu(
        torch.full((seq_len, kv_seq_len), float("-inf"), device=device),
        diagonal=1,
    )
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, kv_seq_len]
