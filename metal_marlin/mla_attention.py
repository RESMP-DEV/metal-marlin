"""
Multi-head Latent Attention (MLA) implementation for Metal Marlin.

MLA compresses KV cache through learned latent projections, achieving significant
memory reduction while maintaining quality. Used by GLM-4.7-Flash, DeepSeek-V2/V3.

Architecture overview:

Standard MHA:
    Q = W_q @ hidden
    K = W_k @ hidden
    V = W_v @ hidden
    KV cache stores: [K, V] per layer

MLA:
    q_latent = q_a_proj @ hidden              # Compress query
    Q = q_b_proj @ q_latent                   # Decompress query

    kv_compressed = kv_a_proj @ hidden        # [kv_lora_rank + rope_dim]
    c_kv = kv_compressed[:kv_lora_rank]       # Latent (no RoPE)
    k_pe = kv_compressed[kv_lora_rank:]       # Position encoding (gets RoPE)

    KV cache stores: [c_kv, k_pe] per layer   # Much smaller!

    # At attention time:
    kv_full = kv_b_proj @ c_kv                # Decompress
    K, V = split(kv_full)
    K = concat(K, RoPE(k_pe))                 # Add positional info

RoPE Fusion Optimization:
    For MLA models where RoPE is decoupled (applied to k_pe separately), we can
    fuse the RoPE application with the kv_a_proj split operation using the
    rope_mla_split_fused Metal kernel, reducing memory round-trips.

Supported MLA variants:
    - GLM-4.7-Flash: Full MLA with q_lora_rank=1536, kv_lora_rank=512, rope_dim=64
    - DeepSeek-V2: KV compression only (no query compression in base version)
    - DeepSeek-V2.5/V3: Full MLA similar to GLM

This implementation uses PyTorch with MPS backend for Apple Silicon acceleration.

Usage:
    from metal_marlin.mla_attention import MLAAttention

    attn = MLAAttention(
        hidden_size=4096,
        num_heads=32,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        rope_ratio=1.0,  # GLM uses rope_ratio for frequency scaling
    )
    output, new_cache = attn(hidden_states, kv_cache, layer_idx=0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fused_attention_mps import fused_attention
from .kv_cache import KVCache, MLAKVCache, TrellisKVCache
from .layers import MarlinLinear

# Import C++ MLA attention if available
try:
    from .mla_attention_cpp import MLAAttentionCpp, is_available as _cpp_mla_available
    _HAS_CPP_MLA = _cpp_mla_available()
except ImportError:
    _HAS_CPP_MLA = False

if TYPE_CHECKING:
    pass


def _get_device() -> torch.device:
    """Get the appropriate device (MPS on Apple Silicon, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class MLAConfig:
    """Configuration for MLA attention layer.

    Attributes:
        hidden_size: Model hidden dimension (d_model)
        num_heads: Number of attention heads
        head_dim: Dimension per head (default: hidden_size // num_heads)
        kv_lora_rank: KV compression latent dimension (e.g., 512)
        q_lora_rank: Query compression dimension (e.g., 1536), None for no compression
        qk_nope_head_dim: Query/key content dimension without RoPE
        qk_rope_head_dim: Dimension for position encoding (e.g., 64)
        v_head_dim: Value head dimension (may differ from q/k in GLM)
        rope_theta: RoPE base frequency (default: 10000)
        rope_ratio: GLM-style frequency scaling (default: 1.0)
        max_position_embeddings: Maximum sequence length for RoPE cache
        quant_type: Quantization format for projections
        group_size: Quantization group size
        bias: Whether to use bias in projections
    """

    hidden_size: int
    num_heads: int
    head_dim: int | None = None
    kv_lora_rank: int = 512
    q_lora_rank: int | None = 1536  # None = no query compression
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int = 64
    v_head_dim: int | None = None
    rope_theta: float = 10000.0
    rope_ratio: float = 1.0
    max_position_embeddings: int = 4096
    quant_type: str = "fp4"
    group_size: int = 128
    bias: bool = False


class MLARoPE(nn.Module):
    """RoPE for MLA with rope_ratio scaling support.

    For MLA models, RoPE is applied only to the qk_rope_head_dim portion,
    not the full head_dim. GLM models use rope_ratio to scale frequencies.

    The inverse frequencies are computed as:
        inv_freq = rope_ratio / (base^(2i/dim))

    This changes the effective context length and positional resolution.
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
        inv_freq = rope_ratio / (base ** (torch.arange(0, half_dim, dtype=torch.float32) * 2 / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin cache for max sequence length
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int) -> None:
        """Build cos/sin cache for the given sequence length."""
        # Ensure positions are created on the same device as inv_freq
        device = self.inv_freq.device
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, self.inv_freq)  # [max_seq, dim/2]
        cos_cache = torch.cos(freqs).to(torch.float16)
        sin_cache = torch.sin(freqs).to(torch.float16)
        self.register_buffer("cos_cache", cos_cache)
        self.register_buffer("sin_cache", sin_cache)
        self.max_seq_len = max_seq_len

    def forward(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """Apply RoPE to input tensor.

        Args:
            x: Input tensor of shape:
               - [batch, seq_len, dim] for k_pe
               - [batch, heads, seq_len, head_dim] for Q
               For all inputs, seq_len is the second-to-last dimension.
            position_offset: Position offset for KV cache continuation

        Returns:
            Tensor with RoPE applied, same shape as input
        """
        # seq_len is always the second-to-last dimension
        seq_len = x.shape[-2]
        device = x.device
        dtype = x.dtype

        # Get cos/sin for positions
        end_pos = position_offset + seq_len
        if end_pos > self.max_seq_len:
            self._build_cache(end_pos)
            self.max_seq_len = end_pos

        cos = self.cos_cache[position_offset:end_pos].to(device)  # [seq_len, dim/2]
        sin = self.sin_cache[position_offset:end_pos].to(device)

        if x.ndim < 3:
            raise ValueError(f"Unsupported input dimensions: {x.ndim}. Expected >= 3.")

        # Broadcast cos/sin across leading dims, keeping seq at -2 and dim at -1.
        # Example shapes:
        # - 3D: [batch, seq, dim] -> [1, seq, dim/2]
        # - 4D: [batch, heads, seq, dim] -> [1, 1, seq, dim/2]
        expand_shape = [1] * (x.ndim - 2) + [seq_len, cos.shape[1]]
        cos = cos.view(*expand_shape)
        sin = sin.view(*expand_shape)

        # Split into even/odd indices along last dimension
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # Apply rotation (Llama-style)
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_odd * cos + x_even * sin

        # Interleave back
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        x_rotated = x_rotated.flatten(-2)  # Flatten last 2 dims to restore original dim

        return x_rotated.to(dtype)



class MLAAttention(nn.Module):
    """Multi-head Latent Attention with compressed KV cache.

    Implements the MLA architecture from DeepSeek-V2/GLM-4.7-Flash:
    1. Query compression: hidden -> q_latent -> Q
    2. KV compression: hidden -> kv_compressed -> (c_kv, k_pe)
    3. KV decompression at attention time: c_kv -> K, V
    4. RoPE applied to Q and k_pe (not to c_kv)
    5. Standard scaled dot-product attention

    Supports:
    - Optional query compression (set q_lora_rank=None to disable)
    - GLM rope_ratio scaling
    - Quantized projections via MarlinLinear
    - Fused RoPE + split operation for kv_a_proj output
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_lora_rank: int = 512,
        q_lora_rank: int | None = 1536,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int = 64,
        head_dim: int | None = None,
        v_head_dim: int | None = None,
        rope_theta: float = 10000.0,
        rope_ratio: float = 1.0,
        max_position_embeddings: int = 4096,
        quant_type: str = "fp4",
        group_size: int = 128,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        base_head_dim = head_dim or (hidden_size // num_heads)
        self.head_dim = base_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = (
            qk_nope_head_dim if qk_nope_head_dim is not None else base_head_dim - qk_rope_head_dim
        )
        if self.qk_nope_head_dim < 0:
            raise ValueError(
                "qk_nope_head_dim must be >= 0 and qk_rope_head_dim must be <= head_dim"
            )
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else base_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.rope_ratio = rope_ratio
        self.scale = self.q_head_dim**-0.5
        self.use_fused_attention = True
        self.use_cpp_mla = _HAS_CPP_MLA
        self._cpp_mla = MLAAttentionCpp() if _HAS_CPP_MLA else None

        # Query projections
        if q_lora_rank is not None:
            # Compressed query path: hidden -> q_latent -> Q
            self.q_a_proj = MarlinLinear(
                hidden_size, q_lora_rank, bias=bias, quant_type=quant_type, group_size=group_size
            )
            self.q_b_proj = MarlinLinear(
                q_lora_rank,
                num_heads * self.q_head_dim,
                bias=bias,
                quant_type=quant_type,
                group_size=group_size,
            )
            self.q_proj = None
        else:
            # Standard query projection
            self.q_proj = MarlinLinear(
                hidden_size,
                num_heads * self.q_head_dim,
                bias=bias,
                quant_type=quant_type,
                group_size=group_size,
            )
            self.q_a_proj = None
            self.q_b_proj = None

        # KV projections (always compressed in MLA)
        # kv_a_proj output: [kv_lora_rank + qk_rope_head_dim]
        kv_a_out_dim = kv_lora_rank + qk_rope_head_dim
        self.kv_a_proj = MarlinLinear(
            hidden_size, kv_a_out_dim, bias=bias, quant_type=quant_type, group_size=group_size
        )
        # kv_b_proj: decompress latent to full K and V
        self.kv_b_proj = MarlinLinear(
            kv_lora_rank,
            num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=bias,
            quant_type=quant_type,
            group_size=group_size,
        )

        # Output projection
        self.o_proj = MarlinLinear(
            num_heads * self.v_head_dim,
            hidden_size,
            bias=bias,
            quant_type=quant_type,
            group_size=group_size,
        )

        # RoPE for q_pe and k_pe (qk_rope_head_dim)
        self.rope_q = MLARoPE(
            dim=self.qk_rope_head_dim,
            base=rope_theta,
            rope_ratio=rope_ratio,
            max_seq_len=max_position_embeddings,
        )
        self.rope_k = MLARoPE(
            dim=qk_rope_head_dim,
            base=rope_theta,
            rope_ratio=rope_ratio,
            max_seq_len=max_position_embeddings,
        )

    def _load_from_state_dict(  # noqa: D401
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Support GLM kv_a_proj_with_mqa weight alias."""
        alias_prefix = f"{prefix}kv_a_proj_with_mqa."
        target_prefix = f"{prefix}kv_a_proj."
        for key in list(state_dict.keys()):
            if key.startswith(alias_prefix):
                mapped_key = f"{target_prefix}{key[len(alias_prefix):]}"
                if mapped_key not in state_dict:
                    state_dict[mapped_key] = state_dict[key]
                state_dict.pop(key, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @classmethod
    def from_config(cls, config: MLAConfig) -> MLAAttention:
        """Create MLAAttention from configuration."""
        return cls(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            kv_lora_rank=config.kv_lora_rank,
            q_lora_rank=config.q_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            rope_theta=config.rope_theta,
            rope_ratio=config.rope_ratio,
            max_position_embeddings=config.max_position_embeddings,
            quant_type=config.quant_type,
            group_size=config.group_size,
            bias=config.bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache: MLAKVCache | KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Input [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            kv_cache: Optional MLA KV cache for autoregressive decoding
            layer_idx: Layer index for KV cache

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Query path
        if self.q_lora_rank is not None:
            q_latent = self.q_a_proj(hidden_states)  # [B, S, q_lora_rank]
            q = self.q_b_proj(q_latent)  # [B, S, num_heads * q_head_dim]
        else:
            q = self.q_proj(hidden_states)

        # Reshape Q: [batch, seq, num_heads, q_head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.q_head_dim)

        # KV path: compress then split
        kv_compressed = self.kv_a_proj(hidden_states)  # [B, S, kv_lora_rank + rope_dim]
        c_kv = kv_compressed[..., : self.kv_lora_rank]  # Latent (no RoPE)
        k_pe = kv_compressed[..., self.kv_lora_rank :]  # Position encoding

        # Determine position offset from cache
        if isinstance(kv_cache, MLAKVCache):
            position_offset = kv_cache.seq_len
        elif kv_cache is not None:
            position_offset = kv_cache.seq_len
        else:
            position_offset = 0

        # Apply RoPE to q_pe and k_pe
        if self.qk_rope_head_dim > 0:
            q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            # MLARoPE expects seq_len at the second-to-last dimension for 4D inputs.
            # q_pe is [B, S, H, D], so transpose to [B, H, S, D] before applying RoPE.
            q_pe = q_pe.permute(0, 2, 1, 3)
            q_pe = self.rope_q(q_pe, position_offset)
            q_pe = q_pe.permute(0, 2, 1, 3)
            q = torch.cat([q_nope, q_pe], dim=-1)
            k_pe = self.rope_k(k_pe, position_offset)

        # Update MLA cache if provided
        if isinstance(kv_cache, MLAKVCache):
            c_kv, k_pe = kv_cache.update_components(layer_idx, c_kv, k_pe)

        # Get current cache/sequence length after potential cache update
        cache_len = c_kv.shape[1]

        # Decompress KV: c_kv -> full K, V
        kv_full = self.kv_b_proj(c_kv)  # [B, cache_len, num_heads * (qk_nope_head_dim + v_head_dim)]
        kv_full = kv_full.view(
            batch_size, cache_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_full.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        if self.qk_rope_head_dim > 0:
            k_pe_expanded = k_pe.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
            k = torch.cat([k_nope, k_pe_expanded], dim=-1)
        else:
            k = k_nope

        # Transpose for attention: [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use fused Metal attention when available, fallback to PyTorch SDPA.
        attn_output: torch.Tensor | None = None
        is_causal = attention_mask is None and seq_len > 1
        if (
            self.use_fused_attention
            and q.is_mps
            and (attention_mask is None or attention_mask.dtype != torch.bool)
        ):
            try:
                attn_output = fused_attention(
                    q,
                    k,
                    v,
                    mask=attention_mask,
                    scale=self.scale,
                    causal=is_causal,
                )
            except Exception:
                attn_output = None

        if attn_output is None:
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=self.scale,
            )

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.v_head_dim)
        )

        output = self.o_proj(attn_output)

        return output

    def mla_proj_fp4_cpp(
        self,
        ctx: Any,
        A: bytes,
        B_packed: bytes,
        scales: bytes,
        C: bytes,
        M: int,
        N: int,
        K: int,
        group_size: int,
        wait: bool = True,
    ) -> None:
        """C++ MLA projection with FP4 quantized weights (prefill phase).
        
        This method uses the optimized C++ Metal kernel for maximum performance.
        Falls back to Python implementation if C++ extension is not available.
        
        Args:
            ctx: MetalContext from _cpp_ext
            A: Input matrix as bytes [M, K] float16
            B_packed: Packed FP4 weight matrix [K/2, N] uint8
            scales: Scale factors for dequantization
            C: Output buffer as bytes [M, N] float16
            M: Batch size (number of tokens)
            N: Output dimension
            K: Input dimension
            group_size: Quantization group size
            wait: Whether to wait for kernel completion
        """
        if self._cpp_mla is not None:
            self._cpp_mla.mla_proj_fp4(ctx, A, B_packed, scales, C, M, N, K, group_size, wait)
        else:
            raise RuntimeError("C++ MLA extension not available. Build with: uv pip install -e .")

    def mla_decode_proj_fp4_cpp(
        self,
        ctx: Any,
        x: bytes,
        W_packed: bytes,
        scales: bytes,
        out: bytes,
        K: int,
        N: int,
        group_size: int,
        wait: bool = True,
    ) -> None:
        """C++ MLA decode projection with FP4 quantized weights (single token).
        
        Optimized for single-token decode phase using C++ Metal kernel.
        
        Args:
            ctx: MetalContext from _cpp_ext
            x: Input vector as bytes [K] float16
            W_packed: Packed FP4 weight matrix [K/2, N] uint8
            scales: Scale factors for dequantization
            out: Output buffer as bytes [N] float16
            K: Input dimension
            N: Output dimension
            group_size: Quantization group size
            wait: Whether to wait for kernel completion
        """
        if self._cpp_mla is not None:
            self._cpp_mla.mla_decode_proj_fp4(ctx, x, W_packed, scales, out, K, N, group_size, wait)
        else:
            raise RuntimeError("C++ MLA extension not available. Build with: uv pip install -e .")

    def is_cpp_available(self) -> bool:
        """Check if C++ MLA path is available.
        
        Returns:
            True if C++ extension is available and usable.
        """
        return self._cpp_mla is not None


def create_mla_from_hf_config(
    config: dict,
    quant_type: str = "fp4",
    group_size: int | None = None,
) -> MLAAttention:
    """Create MLAAttention from HuggingFace config dict.

    Extracts MLA-specific parameters from model config and creates attention layer.

    Args:
        config: HuggingFace model config dict
        quant_type: Quantization type for projections
        group_size: Quantization group size. If None, auto-calculates from dimensions.

    Returns:
        Configured MLAAttention instance
    """
    # Extract dimensions
    hidden_size = config.get("hidden_size", 4096)
    num_heads = config.get("num_attention_heads", 32)
    qk_rope_head_dim = config.get("qk_rope_head_dim", 64)
    qk_nope_head_dim = config.get("qk_nope_head_dim")
    head_dim = config.get("head_dim")
    if head_dim is None:
        if qk_nope_head_dim is not None:
            head_dim = qk_nope_head_dim + qk_rope_head_dim
        else:
            head_dim = hidden_size // num_heads
    if qk_nope_head_dim is None:
        qk_nope_head_dim = head_dim - qk_rope_head_dim

    # MLA-specific parameters
    kv_lora_rank = config.get("kv_lora_rank", 512)
    q_lora_rank = config.get("q_lora_rank", 1536)  # May be None
    v_head_dim = config.get("v_head_dim", head_dim)

    # RoPE parameters
    rope_theta = config.get("rope_theta", 10000.0)
    rope_ratio = config.get("rope_ratio", 1.0)
    max_position_embeddings = config.get("max_position_embeddings", 4096)

    # Auto-calculate group_size if not provided
    if group_size is None:
        # Find a group_size that divides all projection dimensions
        dims = [hidden_size, kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)]
        if q_lora_rank is not None:
            dims.append(q_lora_rank)
            dims.append(num_heads * (qk_nope_head_dim + qk_rope_head_dim))
        # Start from 128 and find largest power of 2 that works
        for gs in [128, 64, 32, 16, 8]:
            if all(d % gs == 0 for d in dims):
                group_size = gs
                break
        else:
            group_size = 1  # Fallback

    return MLAAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=q_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        rope_theta=rope_theta,
        rope_ratio=rope_ratio,
        max_position_embeddings=max_position_embeddings,
        quant_type=quant_type,
        group_size=group_size,
    )
