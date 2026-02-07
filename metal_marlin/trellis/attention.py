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
from transformers.models.glm4_moe.modeling_glm4_moe import apply_rotary_pos_emb

from ..attention import scaled_dot_product_attention_metal
from ..fused_attention_mps import fused_attention  # noqa: F401
from .dispatch import dispatch_fused_qkv_trellis
from ..kv_cache import TrellisKVCache
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
        from transformers.models.glm4_moe.configuration_glm4_moe import \
            Glm4MoeConfig
        from transformers.models.glm4_moe.modeling_glm4_moe import \
            Glm4MoeRotaryEmbedding

        rope_config = Glm4MoeConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_kv_heads,
            head_dim=config.qk_rope_head_dim,  # RoPE only on rope dim
            qk_rope_head_dim=config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            partial_rotary_factor=1.0,
        )
        self.rotary_emb = Glm4MoeRotaryEmbedding(rope_config)

        self.scale = config.qk_head_dim**-0.5

        # Head configuration
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads

        # GQA repeat factor (1 if num_heads == num_kv_heads)
        self.qkv_repeat_factor = self.num_heads // self.num_kv_heads

    def _fused_qkv_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused Q/KV projection for decode.

        Uses fused Metal kernel to compute q_a_proj and kv_a_proj simultaneously.
        """
        # Get lib from q_a_proj (assumed set)
        lib = self.q_a_proj._get_lib()

        # We use Q=q_a, K=kv_a, V=None
        # This computes q_compressed and compressed_kv in one kernel launch
        q_compressed, compressed_kv, _ = dispatch_fused_qkv_trellis(
            lib, x, self.q_a_proj, self.kv_a_proj, None
        )

        # Add bias if present (fused kernel handles matmul only)
        # Clone bias to work around PyTorch MPS validation error where
        # add_dense_scalar kernel declares write access but gets read-only binding
        if self.q_a_proj.bias is not None:
            q_compressed = q_compressed + self.q_a_proj.bias.clone()
        if self.kv_a_proj.bias is not None:
            compressed_kv = compressed_kv + self.kv_a_proj.bias.clone()

        return q_compressed, compressed_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: TrellisKVCache | None = None,
        layer_idx: int = 0,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of TrellisMLAttention.

        MLA KV Caching Strategy:
        - Cache the compressed representation (kv_a_proj output) BEFORE layernorm
        - This reduces KV cache size by ~8x (512+64 vs 20*256*2 per token)
        - On decode: retrieve cached, apply layernorm, then kv_b_proj to decompress

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Optional position IDs for RoPE
            kv_cache: TrellisKVCache for MLA caching (stores compressed KV)
            layer_idx: Layer index for KV cache
            rope_cos: Precomputed RoPE cos values [1, 1, seq_len, rope_dim//2].
                      If provided, skips on-the-fly RoPE computation.
            rope_sin: Precomputed RoPE sin values [1, 1, seq_len, rope_dim//2].
                      If provided, skips on-the-fly RoPE computation.

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Optim for batch=1 decode: fuse q_a and kv_a projections
        is_decode = batch_size == 1 and seq_len == 1
        can_fuse = (
            is_decode and
            self.q_a_proj is not None and isinstance(self.q_a_proj, TrellisLinear) and
            isinstance(self.kv_a_proj, TrellisLinear) and
            # Fused kernel requires same bit width for all projections
            self.q_a_proj.bits == self.kv_a_proj.bits
        )

        if can_fuse:
            q_compressed, compressed_kv = self._fused_qkv_forward(
                hidden_states)
            # Ensure 3D shape for fused output
            if q_compressed.dim() == 2:
                q_compressed = q_compressed.unsqueeze(0)
            if compressed_kv.dim() == 2:
                compressed_kv = compressed_kv.unsqueeze(0)

            # Finish Q path
            if self.q_a_layernorm is not None:
                q_compressed = self.q_a_layernorm(q_compressed)
            q = self.q_b_proj(q_compressed)
            # Ensure 3D shape
            if q.dim() == 2:
                q = q.unsqueeze(0)
        else:
            # === Query Projection (Low-rank MLA) ===
            # hidden -> q_a_proj -> layernorm -> q_b_proj -> Q
            if self.q_a_proj is not None and self.q_b_proj is not None:
                q_compressed = self.q_a_proj(hidden_states)
                # Ensure 3D shape: [batch, seq, dim]
                if q_compressed.dim() == 2:
                    q_compressed = q_compressed.unsqueeze(0)
                if self.q_a_layernorm is not None:
                    q_compressed = self.q_a_layernorm(q_compressed)
                q = self.q_b_proj(q_compressed)
                # Ensure 3D shape: [batch, seq, dim]
                if q.dim() == 2:
                    q = q.unsqueeze(0)
            else:
                raise ValueError("GLM MLA requires q_a_proj and q_b_proj")

            # === KV Compression (with MQA rope component) ===
            # hidden -> kv_a_proj -> [latent, k_rope]
            compressed_kv = self.kv_a_proj(hidden_states)
            # Ensure 3D shape for KV cache: [batch, seq, kv_dim]
            if compressed_kv.dim() == 2:
                compressed_kv = compressed_kv.unsqueeze(0)
            # Debug: verify shape
            assert compressed_kv.dim(
            ) == 3, f"compressed_kv must be 3D, got {compressed_kv.shape}"

        # Reshape Q: [batch, seq, num_heads * qk_head_dim] -> [batch, num_heads, seq, qk_head_dim]
        q = q.view(batch_size, seq_len, self.num_heads,
                   self.qk_head_dim).transpose(1, 2)

        # Split Q into non-positional and rotary parts
        q_nope, q_rope = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # === MLA KV Cache: Store compressed representation ===
        # TrellisKVCache stores [kv_lora_rank + qk_rope_head_dim] per token
        # This is ~8x smaller than storing full K,V ([num_kv_heads * qk_head_dim] * 2)
        if isinstance(kv_cache, TrellisKVCache):
            # Update cache with compressed representation (before layernorm)
            # Returns full sequence of compressed KV: [batch, total_seq, kv_lora_rank + qk_rope_head_dim]
            compressed_kv_full = kv_cache.update(
                layer_idx,
                compressed_kv=compressed_kv,
            )
            total_seq_len = compressed_kv_full.shape[1]

            # Split full sequence into latent and rope components
            k_latent_full, k_rope_full = torch.split(
                compressed_kv_full, [self.kv_lora_rank,
                                     self.qk_rope_head_dim], dim=-1
            )

            # Apply layernorm to latent part (after retrieval from cache)
            if self.kv_a_layernorm is not None:
                k_latent_full = self.kv_a_layernorm(k_latent_full)

            # Decompress to get k_nope and V
            kv_decompressed = self.kv_b_proj(
                k_latent_full
            )  # [batch, total_seq, num_kv_heads * (qk_nope + v)]

            # Reshape: [batch, total_seq, num_kv_heads * (nope + v)] -> [batch, num_kv_heads, total_seq, nope + v]
            kv_decompressed = kv_decompressed.view(
                batch_size,
                total_seq_len,
                self.num_kv_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            ).transpose(1, 2)

            # Split into k_nope and V
            k_nope, v = torch.split(
                kv_decompressed, [self.qk_nope_head_dim,
                                  self.v_head_dim], dim=-1
            )

            # Reshape k_rope: [batch, total_seq, rope_dim] -> [batch, 1, total_seq, rope_dim]
            k_rope_for_attn = k_rope_full.view(
                batch_size, 1, total_seq_len, self.qk_rope_head_dim)

            # Get position IDs for the full cached sequence
            if position_ids is None:
                position_ids_full = torch.arange(
                    total_seq_len, device=hidden_states.device
                ).unsqueeze(0)
            else:
                # position_ids is for current tokens; prepend cached positions
                cached_len = total_seq_len - seq_len
                cached_positions = torch.arange(cached_len, device=hidden_states.device).unsqueeze(
                    0
                )
                position_ids_full = torch.cat(
                    [cached_positions, position_ids], dim=1)

            # Apply RoPE to full k_rope sequence
            # Use precomputed cache if available (fast path), else compute on-the-fly
            if rope_cos is not None and rope_sin is not None:
                # Fast path: slice from precomputed cache
                # rope_cos/rope_sin shape: [1, 1, max_seq_len, rope_dim]
                # After squeeze: [max_seq_len, rope_dim]
                # Index with position_ids: [batch, total_seq, rope_dim]
                cos_full = rope_cos.squeeze(0).squeeze(0)[position_ids_full]
                sin_full = rope_sin.squeeze(0).squeeze(0)[position_ids_full]
                # Cast cos/sin to q_rope dtype to avoid float16+bfloat16->float32 promotion
                cos_full = cos_full.to(q_rope.dtype)
                sin_full = sin_full.to(q_rope.dtype)
                # For Q, we only need the last seq_len positions
                # cos_full is [batch, total_seq, rope_dim] (3D)
                cos_q = cos_full[:, -seq_len:, :]
                sin_q = sin_full[:, -seq_len:, :]
                q_rope, _ = apply_rotary_pos_emb(q_rope, q_rope, cos_q, sin_q)
                _, k_rope_for_attn = apply_rotary_pos_emb(
                    k_rope_for_attn, k_rope_for_attn, cos_full, sin_full
                )
            else:
                # Fallback: compute on-the-fly (slow path)
                cos, sin = self.rotary_emb(k_rope_for_attn, position_ids_full)
                # Glm4MoeRotaryEmbedding returns [batch, seq, rope_dim] (3D)
                # apply_rotary_pos_emb will unsqueeze(1) internally to get
                # [batch, 1, seq, rope_dim] which broadcasts with q/k [batch, heads, seq, dim]
                # For Q, we only need the last seq_len positions
                cos_q = cos[:, -seq_len:]
                sin_q = sin[:, -seq_len:]
                q_rope, _ = apply_rotary_pos_emb(q_rope, q_rope, cos_q, sin_q)
                _, k_rope_for_attn = apply_rotary_pos_emb(
                    k_rope_for_attn, k_rope_for_attn, cos, sin
                )

            # Expand k_rope to match num_kv_heads
            k_rope_for_attn = k_rope_for_attn.expand(
                batch_size, self.num_kv_heads, total_seq_len, self.qk_rope_head_dim
            )

            # Concatenate to form full K
            k = torch.cat([k_nope, k_rope_for_attn], dim=-1)

        else:
            if kv_cache is not None:
                raise ValueError(
                    "TrellisMLAttention requires kv_cache to be TrellisKVCache or None."
                )

            # No cache path (standard inference)
            # Split current tokens into latent and rope components
            k_latent, k_rope = torch.split(
                compressed_kv, [self.kv_lora_rank,
                                self.qk_rope_head_dim], dim=-1
            )

            # === KV Decompression ===
            # latent -> layernorm -> kv_b_proj -> [k_nope, V]
            if self.kv_a_layernorm is not None:
                k_latent = self.kv_a_layernorm(k_latent)
            # [batch, seq, num_kv_heads * (qk_nope + v)]
            kv_decompressed = self.kv_b_proj(k_latent)

            # Reshape: [batch, seq, num_kv_heads * (nope + v)] -> [batch, num_kv_heads, seq, nope + v]
            kv_decompressed = kv_decompressed.view(
                batch_size,
                seq_len,
                self.num_kv_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            ).transpose(1, 2)

            # Split into k_nope and V
            k_nope, v = torch.split(
                kv_decompressed, [self.qk_nope_head_dim,
                                  self.v_head_dim], dim=-1
            )

            # Reshape k_rope: [batch, seq, rope_dim] -> [batch, 1, seq, rope_dim]
            k_rope = k_rope.view(batch_size, 1, seq_len, self.qk_rope_head_dim)

            # === Apply RoPE ===
            if position_ids is None:
                offset = 0
                position_ids = torch.arange(
                    offset, offset + seq_len, device=hidden_states.device
                ).unsqueeze(0)

            # Use precomputed cache if available (fast path), else compute on-the-fly
            if rope_cos is not None and rope_sin is not None:
                # Fast path: slice from precomputed cache
                # rope_cos/rope_sin shape: [1, 1, max_seq_len, rope_dim]
                # Extract: [batch, seq, rope_dim] - apply_rotary_pos_emb will unsqueeze
                cos = rope_cos.squeeze(0).squeeze(0)[position_ids]
                sin = rope_sin.squeeze(0).squeeze(0)[position_ids]
                # Cast cos/sin to q_rope dtype to avoid float16+bfloat16->float32 promotion
                cos = cos.to(q_rope.dtype)
                sin = sin.to(q_rope.dtype)
                q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
            else:
                # Fallback: compute on-the-fly (slow path)
                cos, sin = self.rotary_emb(k_rope, position_ids)
                # Glm4MoeRotaryEmbedding returns [batch, seq, rope_dim] (3D)
                # apply_rotary_pos_emb will unsqueeze(1) internally to get
                # [batch, 1, seq, rope_dim] which broadcasts with q/k [batch, heads, seq, dim]
                q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)

            # Expand k_rope to match num_kv_heads
            k_rope = k_rope.expand(
                batch_size, self.num_kv_heads, seq_len, self.qk_rope_head_dim
            )

            # Concatenate to form K
            k = torch.cat([k_nope, k_rope], dim=-1)

        # Concatenate Q
        q = torch.cat([q_nope, q_rope], dim=-1)

        # Ensure dtypes match for SDPA (after Q concatenation)
        k = k.to(q.dtype)
        v = v.to(q.dtype)

        # === Handle GQA (repeat K/V heads if needed) ===
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.qkv_repeat_factor, dim=1)
            v = v.repeat_interleave(self.qkv_repeat_factor, dim=1)

        # === Fast Decode Path (batch=1, seq_len=1) ===
        # Use PyTorch SDPA for decode - optimized and stable on MPS
        # Note: PyTorch SDPA (~2ms) is faster than fused_attention (~16ms) for decode shapes
        if batch_size == 1 and seq_len == 1:
            # Q: [1, num_heads, 1, qk_head_dim]
            # K: [1, num_heads, total_seq_len, qk_head_dim]
            # V: [1, num_heads, total_seq_len, v_head_dim]
            # causal=False for decode: single query attends to all cached keys
            attn_output_4d = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scale,
            )  # [1, num_heads, 1, v_head_dim]

            # Reshape to [batch, seq, heads * v_head_dim]
            attn_output = attn_output_4d.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.num_heads * self.v_head_dim
            )
        else:
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
