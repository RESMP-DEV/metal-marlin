"""
Full transformer block with pre-norm architecture and Marlin-quantized layers.

Implements a single decoder block matching the Llama/Mistral pattern:
    x -> RMSNorm -> Attention -> residual -> RMSNorm -> MLP -> residual

All linear projections use FP4/INT4 quantized GEMM via Metal kernels.

Usage:
    from metal_marlin.python.transformer import MarlinTransformerBlock

    block = MarlinTransformerBlock(
        hidden_size=4096,
        num_heads=32,
        intermediate_size=11008,
        num_kv_heads=8,
    )
    output = block(hidden_states, kv_cache=cache, layer_idx=0)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .attention import MarlinAttention
from .kv_cache import KVCache
from .mlp import MarlinMLP


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes inputs by their RMS value and applies a learned scale.
    More efficient than LayerNorm since it skips mean subtraction and
    bias, while achieving comparable training stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        variance = mx.mean(x ** 2, axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance + self.eps)
        return self.weight * x


class MarlinTransformerBlock(nn.Module):
    """
    Single transformer decoder block with pre-norm architecture.

    Structure:
      x -> RMSNorm -> Attention -> + -> RMSNorm -> MLP -> +
      |______________________________|   |__________________|

    All linear projections (Q/K/V/O in attention, gate/up/down in MLP)
    use Marlin FP4-quantized GEMM kernels.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        intermediate_size: MLP intermediate dimension.
        num_kv_heads: Number of KV heads for GQA. Defaults to num_heads (MHA).
        quant_type: Quantization format for all linear layers.
        group_size: Quantization group size for all linear layers.
        rms_norm_eps: Epsilon for RMSNorm stability.
        rope_theta: Base frequency for RoPE position embeddings.
        max_position_embeddings: Maximum sequence length for RoPE.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_kv_heads: int | None = None,
        quant_type: str = "fp4",
        group_size: int = 128,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 4096,
    ):
        super().__init__()

        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.self_attn = MarlinAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            quant_type=quant_type,
            group_size=group_size,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
        )

        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.mlp = MarlinMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_type=quant_type,
            group_size=group_size,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> mx.array:
        """Forward pass through the transformer block.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional causal/padding mask
            position_ids: Optional position IDs for RoPE
            kv_cache: Optional KV cache for autoregressive decoding
            layer_idx: Layer index for KV cache addressing

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
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
