"""
Multi-head Latent Attention (MLA) with trellis-quantized projections.

MLA compresses KV caches using low-rank decomposition for efficient memory usage.
This implementation uses trellis-quantized weights for all linear projections.

Architecture based on GLM-4 and DeepSeek-V2 MLA:
- KV compression: hidden_states -> kv_compressed via kv_a_proj
- KV decompression: kv_compressed -> kv via kv_b_proj
- Optional query compression via q_lora_rank

Usage:
    from metal_marlin.trellis_attention import TrellisMLAttention, TrellisMLAConfig

    config = TrellisMLAConfig(
        hidden_size=2048,
        num_attention_heads=32,
        num_kv_heads=4,
        kv_lora_rank=512,
    )

    attn = TrellisMLAttention(
        config=config,
        q_proj=q_linear,
        kv_a_proj=kv_a_linear,
        kv_b_proj=kv_b_linear,
        o_proj=o_linear,
    )

    output = attn(hidden_states, kv_cache=cache, layer_idx=0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .attention import RoPE, scaled_dot_product_attention_metal
from .kv_cache import KVCache
from .trellis_kv_cache import TrellisKVCache
from .trellis_linear import TrellisLinear

if TYPE_CHECKING:
    pass


@dataclass
class TrellisMLAConfig:
    """Configuration for TrellisMLAttention."""

    hidden_size: int = 2048
    num_attention_heads: int = 32
    num_kv_heads: int = 4  # MLA uses fewer KV heads
    head_dim: int = 64
    kv_lora_rank: int = 512  # Compression dimension for KV
    q_lora_rank: int | None = None  # Optional query compression
    rope_theta: float = 10000.0
    max_position_embeddings: int = 131072


class TrellisMLAttention(nn.Module):
    """Multi-head Latent Attention with trellis-quantized projections.

    MLA compresses KV caches using low-rank decomposition:
    1. Compression: hidden_states -> kv_compressed [batch, seq, kv_lora_rank]
    2. Decompression: kv_compressed -> kv [batch, num_kv_heads, seq, head_dim]
    3. Split into K and V for attention computation
    """

    def __init__(
        self,
        config: TrellisMLAConfig,
        q_a_proj: TrellisLinear | None,
        q_b_proj: TrellisLinear | None,
        kv_a_proj: TrellisLinear,
        kv_b_proj: TrellisLinear,
        o_proj: TrellisLinear,
    ):
        super().__init__()
        self.config = config
        self.q_a_proj = q_a_proj
        self.q_b_proj = q_b_proj
        self.kv_a_proj = kv_a_proj
        self.kv_b_proj = kv_b_proj
        self.o_proj = o_proj

        # Validate dimensions
        if kv_a_proj.out_features != config.kv_lora_rank:
            raise ValueError(
                f"kv_a_proj.out_features ({kv_a_proj.out_features}) must equal "
                f"config.kv_lora_rank ({config.kv_lora_rank})"
            )

        # kv_b_proj input should match kv_lora_rank, output should match K+V dimensions
        expected_kv_b_input = config.kv_lora_rank
        expected_kv_b_output = config.num_kv_heads * config.head_dim * 2  # K and V concatenated

        if kv_b_proj.in_features != expected_kv_b_input:
            raise ValueError(
                f"kv_b_proj.in_features ({kv_b_proj.in_features}) must equal "
                f"config.kv_lora_rank ({expected_kv_b_input})"
            )

        if kv_b_proj.out_features != expected_kv_b_output:
            raise ValueError(
                f"kv_b_proj.out_features ({kv_b_proj.out_features}) must equal "
                f"num_kv_heads * head_dim * 2 ({expected_kv_b_output})"
            )

        self.rope = RoPE(config.head_dim, base=config.rope_theta)
        self.scale = config.head_dim**-0.5

        # For GQA compatibility
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim

        # GQA repeat factor
        self.qkv_repeat_factor = self.num_heads // self.num_kv_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: KVCache | TrellisKVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Forward pass of TrellisMLAttention.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Optional position IDs (not used, RoPE uses offset)
            kv_cache: Optional KV cache for autoregressive generation
            layer_idx: Layer index for KV cache

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Query projection (low-rank MLA: q_a_proj compresses, q_b_proj decompresses)
        if self.q_a_proj is not None and self.q_b_proj is not None:
            # Low-rank query: hidden_states -> q_compressed -> q
            q_compressed = self.q_a_proj(hidden_states)
            q = self.q_b_proj(q_compressed)
        else:
            # Direct use of hidden_states as queries (no compression)
            q = hidden_states

        # MLA KV compression and decompression
        # Step 1: Compress hidden_states to low-rank representation
        kv_compressed = self.kv_a_proj(hidden_states)  # [batch, seq_len, kv_lora_rank]

        # Step 2: Decompress to KV space
        kv = self.kv_b_proj(kv_compressed)  # [batch, seq_len, num_kv_heads * head_dim * 2]

        # Step 3: Split into K and V and reshape
        # First reshape to [batch, seq_len, num_kv_heads, head_dim * 2]
        kv = kv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim * 2)

        # Split last dimension into K and V
        k, v = kv.chunk(2, dim=-1)  # Each: [batch, seq_len, num_kv_heads, head_dim]

        # Reshape queries to [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch, num_heads, seq_len, head_dim] for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        offset = kv_cache.seq_len if kv_cache else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        # Update KV cache
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)
            kv_cache.advance(seq_len)

        # Handle GQA by repeating K/V heads if needed
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.qkv_repeat_factor, dim=1)
            v = v.repeat_interleave(self.qkv_repeat_factor, dim=1)

        # Scaled dot-product attention
        attn_output = scaled_dot_product_attention_metal(
            q,
            k,
            v,
            attn_mask=attention_mask,
            scale=self.scale,
            is_causal=attention_mask is None,
        )

        # Reshape output: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.head_dim)
        )

        # Output projection
        output = self.o_proj(attn_output)

        return output


def create_mla_projections(
    config: TrellisMLAConfig,
    bits: int = 4,
    bias: bool = False,
    device: torch.device | str | None = None,
) -> dict[str, TrellisLinear]:
    """Create trellis-quantized linear projections for MLA.

    Args:
        config: MLA configuration
        bits: Quantization bit width (2, 3, or 4)
        bias: Whether to include bias terms
        device: Device to create layers on

    Returns:
        Dictionary with 'q_proj', 'kv_a_proj', 'kv_b_proj', 'o_proj' keys
    """
    projections = {}

    # Optional query compression (low-rank: q_a_proj + q_b_proj)
    if config.q_lora_rank is not None:
        projections["q_a_proj"] = TrellisLinear(
            in_features=config.hidden_size,
            out_features=config.q_lora_rank,
            bits=bits,
            bias=bias,
            device=device,
        )
        projections["q_b_proj"] = TrellisLinear(
            in_features=config.q_lora_rank,
            out_features=config.num_attention_heads * config.head_dim,
            bits=bits,
            bias=bias,
            device=device,
        )
    else:
        projections["q_a_proj"] = None
        projections["q_b_proj"] = None

    # KV compression projection: hidden_size -> kv_lora_rank
    projections["kv_a_proj"] = TrellisLinear(
        in_features=config.hidden_size,
        out_features=config.kv_lora_rank,
        bits=bits,
        bias=bias,
        device=device,
    )

    # KV decompression projection: kv_lora_rank -> (K+V)
    # Output dimension is num_kv_heads * head_dim * 2 for K and V
    kv_b_output_dim = config.num_kv_heads * config.head_dim * 2
    projections["kv_b_proj"] = TrellisLinear(
        in_features=config.kv_lora_rank,
        out_features=kv_b_output_dim,
        bits=bits,
        bias=bias,
        device=device,
    )

    # Output projection: num_heads * head_dim -> hidden_size
    projections["o_proj"] = TrellisLinear(
        in_features=config.num_attention_heads * config.head_dim,
        out_features=config.hidden_size,
        bits=bits,
        bias=bias,
        device=device,
    )

    return projections
