"""
Multi-head Latent Attention (MLA) with trellis-quantized projections.

GLM-4 MLA Architecture:
- Q projection with low-rank compression (q_a_proj -> layernorm -> q_b_proj)
- KV compression with MQA: kv_a outputs latent + rope component
- KV decompression: kv_b expands latent to k_nope + v

Q/K dimension handling:
- Q is split into q_nope (192d) + q_rope (64d)
- K is built from k_nope (from kv_b) + k_rope (from kv_a)
- Both Q and K end up with qk_head_dim = 256

Uses GLM's rotary embedding from transformers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

# Import GLM's RoPE from transformers
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteRotaryEmbedding,
    apply_rotary_pos_emb,
)

from ..attention import scaled_dot_product_attention_metal
from ..kv_cache import KVCache
from .kv_cache import TrellisKVCache
from .linear import TrellisLinear

if TYPE_CHECKING:
    pass


@dataclass
class TrellisMLAConfig:
    """Configuration for TrellisMLAttention (GLM-4 MLA).

    GLM-4 MLA dimensions:
    - qk_nope_head_dim: Non-positional part of Q/K (192)
    - qk_rope_head_dim: Rotary positional part of Q/K (64)
    - v_head_dim: Value dimension (256)
    - kv_lora_rank: Latent dimension for KV compression (512)
    - q_lora_rank: Latent dimension for Q compression (768)
    """

    hidden_size: int = 2048
    num_attention_heads: int = 20
    num_kv_heads: int = 20  # Same as attention heads in GLM
    qk_nope_head_dim: int = 192  # Non-positional Q/K dim
    qk_rope_head_dim: int = 64  # Rotary Q/K dim
    v_head_dim: int = 256  # Value dimension
    kv_lora_rank: int = 512  # KV compression dimension
    q_lora_rank: int | None = 768  # Query compression dimension
    rope_theta: float = 1000000.0  # GLM uses 1M
    max_position_embeddings: int = 131072

    @property
    def qk_head_dim(self) -> int:
        """Total Q/K head dimension (nope + rope)."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    # Legacy compatibility
    @property
    def head_dim(self) -> int:
        return self.qk_head_dim

    @property
    def kv_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.v_head_dim


class TrellisMLAttention(nn.Module):
    """Multi-head Latent Attention with trellis-quantized projections.

    GLM-4 MLA architecture:
    1. Q projection: hidden -> q_a_proj -> layernorm -> q_b_proj -> Q
       Q is split into q_nope (192d) + q_rope (64d)
    2. KV compression: hidden -> kv_a_proj -> [latent (512d), k_rope (64d)]
    3. KV decompression: latent -> layernorm -> kv_b_proj -> [k_nope (192d), V (256d)]
    4. K is built from k_nope + k_rope (both 192d + 64d = 256d)
    5. RoPE applied to q_rope and k_rope only
    """

    def __init__(
        self,
        config: TrellisMLAConfig,
        q_a_proj: TrellisLinear | None,
        q_b_proj: TrellisLinear | None,
        kv_a_proj: TrellisLinear,
        kv_b_proj: TrellisLinear,
        o_proj: TrellisLinear,
        q_a_layernorm: nn.Module | None = None,
        kv_a_layernorm: nn.Module | None = None,
    ):
        super().__init__()
        self.config = config
        self.q_a_proj = q_a_proj
        self.q_b_proj = q_b_proj
        self.kv_a_proj = kv_a_proj
        self.kv_b_proj = kv_b_proj
        self.o_proj = o_proj

        # Layer norms for MLA (GLM uses them before decompression)
        self.q_a_layernorm = q_a_layernorm
        self.kv_a_layernorm = kv_a_layernorm

        # Store MLA dimensions
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_head_dim = config.qk_head_dim

        # GLM's rotary embedding - use actual HF config class
        from transformers.models.glm4_moe_lite.configuration_glm4_moe_lite import Glm4MoeLiteConfig

        rope_config = Glm4MoeLiteConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_kv_heads,
            head_dim=config.qk_rope_head_dim,  # RoPE only on rope dim
            qk_rope_head_dim=config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            partial_rotary_factor=1.0,
        )
        self.rotary_emb = Glm4MoeLiteRotaryEmbedding(rope_config)

        self.scale = config.qk_head_dim**-0.5

        # Head configuration
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads

        # GQA repeat factor (1 if num_heads == num_kv_heads)
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
            position_ids: Optional position IDs for RoPE
            kv_cache: Optional KV cache for autoregressive generation
            layer_idx: Layer index for KV cache

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # === Query Projection (Low-rank MLA) ===
        # hidden -> q_a_proj -> layernorm -> q_b_proj -> Q
        if self.q_a_proj is not None and self.q_b_proj is not None:
            q_compressed = self.q_a_proj(hidden_states)
            if self.q_a_layernorm is not None:
                q_compressed = self.q_a_layernorm(q_compressed)
            q = self.q_b_proj(q_compressed)
        else:
            raise ValueError("GLM MLA requires q_a_proj and q_b_proj")

        # Reshape Q: [batch, seq, num_heads * qk_head_dim] -> [batch, num_heads, seq, qk_head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim).transpose(1, 2)

        # Split Q into non-positional and rotary parts
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # === KV Compression (with MQA rope component) ===
        # hidden -> kv_a_proj -> [latent, k_rope]
        compressed_kv = self.kv_a_proj(
            hidden_states
        )  # [batch, seq, kv_lora_rank + qk_rope_head_dim]

        # Split into latent and rope components
        k_latent, k_rope = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # === KV Decompression ===
        # latent -> layernorm -> kv_b_proj -> [k_nope, V]
        if self.kv_a_layernorm is not None:
            k_latent = self.kv_a_layernorm(k_latent)
        kv_decompressed = self.kv_b_proj(k_latent)  # [batch, seq, num_kv_heads * (qk_nope + v)]

        # Reshape: [batch, seq, num_kv_heads * (nope + v)] -> [batch, num_kv_heads, seq, nope + v]
        kv_decompressed = kv_decompressed.view(
            batch_size, seq_len, self.num_kv_heads, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)

        # Split into k_nope and V
        k_nope, v = torch.split(kv_decompressed, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Reshape k_rope: [batch, seq, rope_dim] -> [batch, 1, seq, rope_dim]
        # The "1" represents shared rope across all KV heads (MQA style)
        k_rope = k_rope.view(batch_size, 1, seq_len, self.qk_rope_head_dim)

        # === Apply RoPE ===
        # Create position_ids if not provided
        if position_ids is None:
            offset = kv_cache.seq_len if kv_cache else 0
            position_ids = torch.arange(
                offset, offset + seq_len, device=hidden_states.device
            ).unsqueeze(0)

        # Get cos/sin from GLM rotary embedding
        cos, sin = self.rotary_emb(k_rope, position_ids)

        # Apply RoPE to q_rope and k_rope only
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)

        # Expand k_rope to match num_kv_heads: [batch, 1, seq, rope] -> [batch, kv_heads, seq, rope]
        k_rope = k_rope.expand(batch_size, self.num_kv_heads, seq_len, self.qk_rope_head_dim)

        # === Concatenate Q and K ===
        # Q: cat(q_nope, q_rope) -> [batch, heads, seq, qk_head_dim]
        # K: cat(k_nope, k_rope) -> [batch, kv_heads, seq, qk_head_dim]
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # === KV Cache Update ===
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)
            kv_cache.advance(seq_len)

        # === Handle GQA (repeat K/V heads if needed) ===
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.qkv_repeat_factor, dim=1)
            v = v.repeat_interleave(self.qkv_repeat_factor, dim=1)

        # === Scaled Dot-Product Attention ===
        # Note: Q and K have qk_head_dim, V has v_head_dim
        # For standard attention this works: score = Q @ K.T, out = score @ V
        attn_output = scaled_dot_product_attention_metal(
            q,
            k,
            v,
            attn_mask=attention_mask,
            scale=self.scale,
            is_causal=attention_mask is None,
        )

        # Output has v_head_dim, not qk_head_dim
        # Reshape: [batch, heads, seq, v_head_dim] -> [batch, seq, heads * v_head_dim]
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.v_head_dim)
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
