"""
Multi-head attention layer with Marlin-quantized projections.

Uses MarlinLinear for Q/K/V/O projections with FP4 quantized weights.
Supports Grouped Query Attention (GQA) where num_kv_heads < num_heads.

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

import mlx.core as mx
import mlx.nn as nn

from .kv_cache import KVCache
from .layers import MarlinLinear


class RoPE(nn.Module):
    """Rotary Position Embedding."""

    def __init__(self, dims: int, traditional: bool = False, base: float = 10000.0):
        super().__init__()
        self.dims = dims
        self.traditional = traditional
        self.base = base

    def __call__(
        self,
        x: mx.array,
        offset: int = 0,
    ) -> mx.array:
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

        # Compute position indices
        positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)

        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (mx.arange(0, self.dims, 2, dtype=mx.float32) / self.dims))

        # Compute angles: [seq_len, dims/2]
        freqs = mx.outer(positions, inv_freq)

        # Compute cos and sin
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)

        # Expand dims for broadcasting: [1, 1, seq_len, dims/2]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

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
        x_rotated = mx.zeros_like(x)
        # Use index assignment workaround
        x_rotated = mx.concatenate(
            [x_rotated_even[..., None], x_rotated_odd[..., None]], axis=-1
        ).reshape(shape)

        return x_rotated


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

    def __call__(
        self,
        hidden_states: mx.array,  # [batch, seq_len, hidden_size]
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> mx.array:
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
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

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
            k = mx.repeat(k, repeat_factor, axis=1)
            v = mx.repeat(v, repeat_factor, axis=1)

        # Scaled dot-product attention
        # Q: [batch, num_heads, seq_len, head_dim]
        # K: [batch, num_heads, kv_seq_len, head_dim]
        # attn_weights: [batch, num_heads, seq_len, kv_seq_len]
        attn_weights = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = mx.softmax(attn_weights, axis=-1)

        # Compute attention output
        attn_output = attn_weights @ v

        # Reshape and project output
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        output = self.o_proj(attn_output)

        return output


def create_causal_mask(seq_len: int, kv_seq_len: int | None = None) -> mx.array:
    """
    Create causal attention mask.

    Args:
        seq_len: Query sequence length
        kv_seq_len: Key/value sequence length (defaults to seq_len)

    Returns:
        Causal mask with -inf for masked positions
    """
    kv_seq_len = kv_seq_len or seq_len

    # Create upper triangular mask
    # For autoregressive: query position i can attend to all positions <= i
    if seq_len == 1:
        # Single token decode - no masking needed
        return None

    # Prefill: standard causal mask
    mask = mx.triu(mx.full((seq_len, kv_seq_len), float("-inf")), k=1)
    return mask[None, None, :, :]  # [1, 1, seq_len, kv_seq_len]
