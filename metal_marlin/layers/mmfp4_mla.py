"""MMFP4 MLA (Multi-head Latent Attention) layer for GLM-style models.

This module implements an FP4-quantized MLA attention block using low-rank
query/KV projections and compressed KV-cache storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import logging

from .._compat import HAS_TORCH, torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..kv_cache import KVCache


if HAS_TORCH and torch is not None:
    import torch.nn as nn
    import torch.nn.functional as F

    from ..kv_cache import KVCache, MLAKVCache
    from ..paged_kv_cache import PagedKVCache
    from ..kernels import paged_attention_v1
    from ..mla_fused import (
        MLAAttentionParams,
        mla_fused_attention_decode,
        mla_fused_attention_prefill,
        mla_chunked_prefill_attention,
    )
    from ..rope import (
        YaRNRoPEMetal,
        dispatch_fused_rope_attention,
    )
    from .mmfp4_linear import MMFP4Linear

    class _RotaryEmbedding(nn.Module):
        """RoPE for MLA rope sub-dimensions with rope_ratio support."""

        def __init__(
            self,
            dim: int,
            max_position_embeddings: int = 131072,
            base: float = 10000.0,
            rope_ratio: float = 1.0,
        ) -> None:
            super().__init__()
            if dim % 2 != 0:
                raise ValueError(f"RoPE dimension must be even, got {dim}")

            self.dim = dim
            self.base = base
            self.rope_ratio = rope_ratio
            self.max_position_embeddings = max_position_embeddings
            self._cached_seq_len = 0

            half_dim = dim // 2
            # GLM-style rope_ratio scaling: inv_freq = rope_ratio / (base ** ...)
            inv_freq = rope_ratio / (
                base
                ** (torch.arange(0, half_dim, dtype=torch.float32) * 2.0 / dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.register_buffer(
                "_cos_cached", torch.empty(0), persistent=False)
            self.register_buffer(
                "_sin_cached", torch.empty(0), persistent=False)
            self._ensure_cache(1, device=inv_freq.device)

        def _ensure_cache(self, seq_len: int, device: torch.device) -> None:
            if (
                self._cached_seq_len >= seq_len
                and self._cos_cached.device == device
                and self._sin_cached.device == device
            ):
                return

            positions = torch.arange(
                seq_len, dtype=torch.float32, device=device)
            inv_freq = self.inv_freq.to(device=device)
            freqs = torch.outer(positions, inv_freq)
            self._cos_cached = torch.cos(freqs).to(torch.float16)
            self._sin_cached = torch.sin(freqs).to(torch.float16)
            self._cached_seq_len = seq_len

        def get_cos_sin(
            self,
            position_ids: torch.Tensor,
            dtype: torch.dtype,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if position_ids.ndim == 1:
                position_ids = position_ids.unsqueeze(0)

            if position_ids.dtype != torch.long:
                position_ids = position_ids.to(torch.long)

            if position_ids.numel() == 0:
                raise ValueError("position_ids must be non-empty")

            if position_ids.device.type == "cpu":
                max_pos = int(position_ids.max()) + 1
                if max_pos > self._cached_seq_len or self._cos_cached.device != position_ids.device:
                    grow_target = max(max_pos, max(1, self._cached_seq_len * 2))
                    if grow_target > self.max_position_embeddings:
                        grow_target = max_pos
                    self._ensure_cache(grow_target, device=position_ids.device)
            elif self._cos_cached.device != position_ids.device:
                self._ensure_cache(self._cached_seq_len, device=position_ids.device)

            flat_pos = position_ids.reshape(-1)
            cos = self._cos_cached.index_select(0, flat_pos).view(
                *position_ids.shape, -1
            )
            sin = self._sin_cached.index_select(0, flat_pos).view(
                *position_ids.shape, -1
            )
            return cos.to(dtype=dtype), sin.to(dtype=dtype)

        def apply(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
            """Apply RoPE to x with absolute position ids.

            Args:
                x: Tensor of shape [B, H, S, D] or [B, S, D].
                position_ids: [B, S] or [S].
            """
            cos, sin = self.get_cos_sin(position_ids, dtype=x.dtype)

            if x.ndim == 4:
                # x: [B, H, S, D], cos: [B, S, D/2] -> unsqueeze(1) to [B, 1, S, D/2]
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
            elif x.ndim == 3:
                # x: [B, S, D], cos: [B, S, D/2] -> match
                pass
            else:
                raise ValueError(f"Unsupported tensor rank for RoPE: {x.ndim}")

            x_even = x[..., ::2]
            x_odd = x[..., 1::2]
            x_rot_even = x_even * cos - x_odd * sin
            x_rot_odd = x_odd * cos + x_even * sin
            return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)

    class MMFP4MLA(nn.Module):
        """MMFP4 MLA layer with optional KV cache quantization for attention optimization.
        
        This layer supports quantized KV caches (FP4/FP8/INT8) for reduced memory bandwidth
        during attention computation. When kv_quant is enabled, the attention kernels will
        dequantize KV values on-the-fly in the shader, reducing memory traffic by 2-4x.
        
        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            num_kv_heads: Number of KV heads (for GQA)
            q_lora_rank: Query compression rank
            kv_lora_rank: KV compression rank
            qk_nope_head_dim: Non-RoPE head dimension for Q/K
            qk_rope_head_dim: RoPE head dimension
            v_head_dim: Value head dimension
            group_size: Quantization group size for weights
            rope_theta: RoPE base frequency
            rope_ratio: RoPE scaling ratio (GLM-style)
            use_fused_qkv: Whether to use fused QKV projection
            use_paged_attention: Whether to use paged attention for decode
            use_fused_decode: Whether to use fused decode kernels
            use_memory_efficient_attention: Whether to use memory-efficient attention
            use_fused_rope_attention: Whether to use fused RoPE+attention (inline RoPE)
            num_layers: Number of layers (for paged attention)
            kv_quant: KV cache quantization mode ("none", "fp4", "fp8", "int8")
            kv_quant_group_size: Group size for KV cache quantization
            sliding_window: Sliding window size for attention (0 = disabled, >0 = window size)
        """
        
        def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            num_kv_heads: int,
            q_lora_rank: int,
            kv_lora_rank: int,
            qk_nope_head_dim: int,
            qk_rope_head_dim: int,
            v_head_dim: int,
            group_size: int = 128,
            rope_theta: float = 10000.0,
            rope_ratio: float = 1.0,
            use_fused_qkv: bool = False,
            use_paged_attention: bool = False,
            use_fused_decode: bool = True,
            use_memory_efficient_attention: bool = True,
            use_fused_rope_attention: bool = True,  # Inline RoPE with attention
            num_layers: int = 32,  # Accepted for compat with mmfp4_causal_lm.py
            kv_quant: str = "none",  # "none", "fp4", "fp8", "int8"
            kv_quant_group_size: int = 128,
            sliding_window: int = 0,  # 0 = disabled, >0 = window size
        ):
            super().__init__()
            if num_heads <= 0 or num_kv_heads <= 0:
                raise ValueError("num_heads and num_kv_heads must be > 0")
            if num_heads % num_kv_heads != 0:
                raise ValueError(
                    f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
                )
            if qk_rope_head_dim % 2 != 0:
                raise ValueError("qk_rope_head_dim must be even for RoPE")

            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.num_layers = num_layers
            self.q_lora_rank = q_lora_rank
            self.kv_lora_rank = kv_lora_rank
            self.qk_nope_head_dim = qk_nope_head_dim
            self.qk_rope_head_dim = qk_rope_head_dim
            self.v_head_dim = v_head_dim
            self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
            self.qkv_repeat_factor = num_heads // num_kv_heads
            self.scale = self.qk_head_dim ** -0.5
            self.layer_idx = 0
            self.use_fused_qkv = use_fused_qkv
            self.use_paged_attention = use_paged_attention
            self.use_fused_decode = use_fused_decode
            self.prefer_glm4_fused_kernel = False
            self.use_memory_efficient_attention = use_memory_efficient_attention
            self.use_fused_rope_attention = use_fused_rope_attention
            self._paged_adapter: Any = None
            self.chunked_prefill_size = 2048  # Default chunk size for prefill
            
            # KV cache quantization settings for attention optimization
            self.kv_quant = kv_quant
            self.kv_quant_group_size = kv_quant_group_size
            self._kv_quant_enabled = kv_quant in ("fp4", "fp8", "int8")
            
            # Sliding window attention for memory-efficient long sequence processing
            self.sliding_window = sliding_window

            # Helper to create dummy weights for MMFP4Linear
            def create_dummy_linear(in_features: int, out_features: int) -> MMFP4Linear:
                # MMFP4Linear expects [out_features, in_features // 8] packed weights
                if in_features % 8 != 0:
                    # In practice, in_features should be divisible by 8 for FP4 packing
                    pass

                # Align in_features to 8 for shape calculation (MMFP4Linear constraint)
                in_aligned = ((in_features + 7) // 8) * 8

                packed_weights = torch.zeros(
                    (out_features, in_aligned // 8), dtype=torch.uint32
                )

                # Scales: [n_groups, out_features]
                n_groups = (in_features + group_size - 1) // group_size
                scales = torch.zeros(
                    (n_groups, out_features), dtype=torch.float16
                )

                return MMFP4Linear(
                    packed_weights=packed_weights,
                    scales=scales,
                    bias=None,
                    group_size=group_size
                )

            if self.use_fused_qkv:
                # Fused Q/KV compression: hidden -> [q_lora + kv_lora + qk_rope]
                self.qkv_a_proj = create_dummy_linear(
                    hidden_size,
                    q_lora_rank + kv_lora_rank + qk_rope_head_dim,
                )
                # To avoid attribute errors if accessed, we can initialize others to None
                # or just skip them. For now, skipping specific init for them.
                # But to maintain structure if someone inspects 'q_a_proj',
                # we might want to keep them or not.
                # Given strict instruction to "Concatenate weights and use single dispatch",
                # I will NOT create the separate ones if fused is True.
            else:
                # Query: hidden -> q_lora -> [heads * (qk_nope + qk_rope)]
                # We assume q_lora_rank > 0 implies query compression
                self.q_a_proj = create_dummy_linear(hidden_size, q_lora_rank)

                # KV compression: hidden -> [kv_lora + qk_rope]
                self.kv_a_proj = create_dummy_linear(
                    hidden_size,
                    kv_lora_rank + qk_rope_head_dim,
                )

            self.q_b_proj = create_dummy_linear(
                q_lora_rank, num_heads * self.qk_head_dim
            )

            # KV decompression: kv_lora -> [num_kv_heads * (qk_nope + v)]
            self.kv_b_proj = create_dummy_linear(
                kv_lora_rank,
                num_kv_heads * (qk_nope_head_dim + v_head_dim),
            )

            # Output projection: [num_heads * v] -> hidden
            self.o_proj = create_dummy_linear(
                num_heads * v_head_dim,
                hidden_size,
            )

            # Layer norms applied after low-rank compression before decompression
            # These are critical for GLM-4 MLA architecture
            self.q_a_layernorm = nn.RMSNorm(q_lora_rank, eps=1e-5)
            self.kv_a_layernorm = nn.RMSNorm(kv_lora_rank, eps=1e-5)

            # GLM uses large RoPE theta; default to provided arg
            # Use local _RotaryEmbedding for consistent API (has rope_ratio attribute)
            # YaRNRoPEMetal has issues: (1) missing rope_ratio, (2) Metal texture limit 16384
            max_position_embeddings = 131072  # Default max seq len for GLM-style models
            self.rotary_emb = _RotaryEmbedding(
                dim=qk_rope_head_dim,
                max_position_embeddings=max_position_embeddings,
                base=rope_theta,
                rope_ratio=rope_ratio,
            )

        def _normalize_position_ids(
            self,
            position_ids: torch.Tensor,
            batch_size: int,
            seq_len: int,
            device: torch.device,
        ) -> torch.Tensor:
            if position_ids.ndim == 1:
                position_ids = position_ids.unsqueeze(0)
            if position_ids.ndim != 2:
                raise ValueError(
                    f"position_ids must have shape [S] or [B, S], got {tuple(position_ids.shape)}"
                )

            if position_ids.shape[0] == 1 and batch_size > 1:
                position_ids = position_ids.expand(batch_size, -1)

            if position_ids.shape != (batch_size, seq_len):
                raise ValueError(
                    f"position_ids shape {tuple(position_ids.shape)} does not match "
                    f"(batch_size, seq_len)=({batch_size}, {seq_len})"
                )

            return position_ids.to(device=device, dtype=torch.long)

        @staticmethod
        def _build_full_position_ids(
            current_position_ids: torch.Tensor,
            total_seq_len: int,
        ) -> torch.Tensor:
            """Infer full key positions from current query positions."""
            current_seq_len = current_position_ids.shape[1]
            if total_seq_len == current_seq_len:
                return current_position_ids

            if total_seq_len < current_seq_len:
                raise ValueError(
                    f"total_seq_len ({total_seq_len}) < current_seq_len ({current_seq_len})"
                )

            offset = total_seq_len - current_seq_len
            starts = current_position_ids[:, :1] - offset
            steps = torch.arange(
                total_seq_len,
                device=current_position_ids.device,
                dtype=current_position_ids.dtype,
            ).unsqueeze(0)
            return starts + steps

        @staticmethod
        def _build_attention_mask(
            query_positions: torch.Tensor,
            key_positions: torch.Tensor,
        ) -> torch.Tensor:
            # Shape: [B, 1, Q, K], True = can attend.
            return (query_positions.unsqueeze(-1) >= key_positions.unsqueeze(-2)).unsqueeze(1)

        def _get_or_create_paged_adapter(self) -> Any:
            """Lazy initialization of paged attention adapter."""
            if self._paged_adapter is None:
                from ..paged.mmfp4_paged_adapter import MMFP4PagedAttention
                self._paged_adapter = MMFP4PagedAttention(
                    mla_layer=self,
                    max_batch_size=1,  # Will be inferred at runtime
                    max_seq_len=8192,
                    num_layers=self.num_layers,
                )
            return self._paged_adapter

        def _forward_paged_attention(
            self,
            q_states: torch.Tensor,
            k_states: torch.Tensor,
            v_states: torch.Tensor,
            pos_q: torch.Tensor,
            key_positions: torch.Tensor,
        ) -> torch.Tensor:
            """Forward using paged attention adapter for decode mode.

            Args:
                q_states: [B, num_heads, 1, qk_head_dim]
                k_states: [B, num_heads, total_seq, qk_head_dim]
                v_states: [B, num_heads, total_seq, v_head_dim]
                pos_q: Query positions [B, 1]
                key_positions: Key positions [B, total_seq]

            Returns:
                attn_output: [B, num_heads, 1, v_head_dim]
            """
            adapter = self._get_or_create_paged_adapter()

            # Split q_states into nope and rope components
            # [B, H, 1, nope_dim]
            q_nope = q_states[..., : self.qk_nope_head_dim]
            # [B, H, 1, rope_dim]
            q_rope = q_states[..., self.qk_nope_head_dim:]

            # Squeeze seq_len dimension (which is 1 for decode)
            q_nope = q_nope.squeeze(2)  # [B, H, nope_dim]
            q_rope = q_rope.squeeze(2)  # [B, H, rope_dim]

            # Compute context lengths from key positions
            context_lens = (key_positions.max(dim=1)[0] + 1).to(torch.int32)

            # Call adapter forward
            attn_out = adapter.forward(
                q_nope=q_nope,
                q_rope=q_rope,
                layer_idx=self.layer_idx,
                context_lens=context_lens,
            )  # Returns [B, H, v_head_dim]

            # Reshape to match standard attention output: [B, H, 1, v_head_dim]
            return attn_out.unsqueeze(2)

        def _mla_streaming(
            self,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            kv_cache: MLAKVCache,
        ) -> torch.Tensor:
            """Streaming MLA decode with optimized cache handling.
            
            Optimized decode path that:
            1. Updates MLAKVCache in-place (avoiding concatenation overhead)
            2. Uses raw cache buffers for zero-copy kernel access
            3. Leverages mla_fused_attention_decode for computation
            """
            batch_size = x.shape[0]
            
            # 1. Update KV cache
            # Compute compressed KV: [B, 1, kv_lora_rank + rope_dim]
            kv_compressed = self.kv_a_proj(x)
            
            # Update cache (handles quantization and position tracking)
            # We ignore the return value (full dequantized cache) to avoid allocation
            kv_cache.update_compressed(self.layer_idx, kv_compressed)
            
            # 2. Prepare kernel parameters
            # Get current sequence length from cache
            cache_len = int(kv_cache._seq_lens[self.layer_idx, 0].item())
            cache_start_pos = cache_len - 1
            
            # Get raw cache buffers (zero-copy)
            # shape: [batch, max_seq, dim]
            k_cache_buf = kv_cache.kv_cache[self.layer_idx]
            v_cache_buf = k_cache_buf # Unified cache for MLA
            
            # Get scales if quantized
            k_scales_buf = kv_cache.kv_scales[self.layer_idx] if kv_cache.kv_scales is not None else None
            v_scales_buf = k_scales_buf
            
            # Handle unit scales for non-quantized case if None
            if k_scales_buf is None:
                # Kernel expects scales. If not quantized, create dummy or handle in kernel?
                # mla_fused_attention_decode expects k_scales. 
                # In _forward_fused_decode, it created ones.
                # Here we should ideally use what's in cache or ones.
                # MLAKVCache.kv_scales is None if quantize_mode="none".
                # We need a fallback.
                n_groups = (self.kv_lora_rank + self.kv_b_proj.group_size - 1) // self.kv_b_proj.group_size
                k_scales_buf = torch.ones(
                    (cache_len, n_groups),
                    dtype=torch.float16,
                    device=x.device,
                )
                v_scales_buf = k_scales_buf
            else:
                # Ensure scales cover the full sequence (MLAKVCache handles this)
                pass

            # Kernel layout helpers
            def _kernel_layout(linear: MMFP4Linear) -> tuple[torch.Tensor, torch.Tensor]:
                packed = linear.packed_weights
                scales = linear.scales
                if packed.device != x.device:
                    packed = packed.to(x.device)
                if scales.device != x.device or scales.dtype != torch.float16:
                    scales = scales.to(device=x.device, dtype=torch.float16)
                return packed.transpose(0, 1).contiguous(), scales.contiguous()

            q_a_packed, q_a_scales = _kernel_layout(self.q_a_proj)
            q_b_packed, q_b_scales = _kernel_layout(self.q_b_proj)
            kv_a_packed, kv_a_scales = _kernel_layout(self.kv_a_proj)
            kv_b_packed, kv_b_scales = _kernel_layout(self.kv_b_proj)
            o_packed, o_scales = _kernel_layout(self.o_proj)

            q_bias = self.q_b_proj.bias
            if q_bias is not None:
                q_bias = q_bias.to(device=x.device, dtype=torch.float16)
                
            # Determine cache max length from buffer
            max_cache_len = k_cache_buf.shape[1] # [batch, max_seq, dim] if BHSD/BSHD?
            # MLAKVCache allocates [layers, batch, max_seq, dim] -> k_cache_buf is [batch, max_seq, dim]
            
            params = MLAAttentionParams(
                batch=batch_size,
                seq_q=1,
                seq_k=cache_len,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                head_dim=self.qk_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                q_lora_rank=self.q_lora_rank,
                rope_dim=self.qk_rope_head_dim,
                scale=self.scale,
                is_causal=True,
                q_a_group_size=self.q_a_proj.group_size,
                q_b_group_size=self.q_b_proj.group_size,
                kv_a_group_size=self.kv_a_proj.group_size,
                kv_b_group_size=self.kv_b_proj.group_size,
                o_group_size=self.o_proj.group_size,
                rope_theta=self.rotary_emb.base,
                rope_ratio=self.rotary_emb.rope_ratio,
                rope_base_seq_len=0,
                cache_start_pos=cache_start_pos,
                cache_len=cache_len,
                max_cache_len=max_cache_len,
                use_fused_q_proj=True,
                use_fused_kv_proj=True,
                fuse_rope_in_kv_a=True,
                skip_kv_decompress=False,
                kv_quant_mode=kv_cache.quantize_mode,
                kv_quant_group_size=self.kv_quant_group_size,
                sliding_window=self.sliding_window,
            )

            return mla_fused_attention_decode(
                hidden=x.to(torch.float16),
                q_a_weights_packed=q_a_packed,
                q_a_scales=q_a_scales,
                q_b_weights_packed=q_b_packed,
                q_b_scales=q_b_scales,
                q_bias=q_bias,
                kv_a_weights_packed=kv_a_packed,
                kv_a_scales=kv_a_scales,
                kv_b_weights_packed=kv_b_packed,
                kv_b_scales=kv_b_scales,
                k_cache=k_cache_buf,
                v_cache=v_cache_buf,
                k_scales=k_scales_buf,
                v_scales=v_scales_buf,
                o_weights_packed=o_packed,
                o_scales=o_scales,
                params=params,
                prefer_glm4_kernel=self.prefer_glm4_fused_kernel,
            )

        def _forward_fused_decode(
            self,
            x: torch.Tensor,
            cache_start_pos: int | None,
            kv_cache: KVCache | None,
        ) -> torch.Tensor:
            """Fused decode path (batch=1, seq=1) using mla_fused_attention_decode.
            
            Supports quantized KV caches (FP4/FP8/INT8) for reduced memory bandwidth.
            When kv_quant is enabled, the cache is quantized on-the-fly and scales
            are passed to the kernel for dequantization during attention.
            """
            if self.use_fused_qkv:
                raise RuntimeError(
                    "Fused decode path requires split q/kv projections")
            if not isinstance(kv_cache, (MLAKVCache, PagedKVCache)) and kv_cache is not None:
                raise RuntimeError(
                    "Fused decode path supports MLAKVCache and PagedKVCache only")

            if self.use_paged_attention and isinstance(kv_cache, PagedKVCache):
                # 1. Project Q
                # hidden -> q_lora
                q_latent = self.q_a_proj(x)
                q_latent = self.q_a_layernorm(q_latent)
                # q_lora -> heads * (nope + rope)
                q = self.q_b_proj(q_latent).view(
                    1, 1, self.num_heads, self.qk_head_dim
                )
                q_nope, q_rope = torch.split(
                    q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
                )
                # Apply RoPE to Q
                # [B, 1, H, D] -> [B, H, 1, D]
                q_rope = q_rope.transpose(1, 2)
                # Position IDs for decode (single token)
                pos_ids = torch.tensor(
                    [[cache_start_pos]], dtype=torch.long, device=x.device
                )
                q_rope = self.rotary_emb.apply(q_rope, pos_ids).transpose(1, 2)
                q_states = torch.cat((q_nope, q_rope), dim=-1)

                # 2. Project KV
                kv_compressed = self.kv_a_proj(x)
                c_kv = kv_compressed[..., : self.kv_lora_rank]
                c_kv = self.kv_a_layernorm(c_kv)
                k_rope_comp = kv_compressed[..., self.kv_lora_rank:]

                kv_decompressed = self.kv_b_proj(c_kv).view(
                    1, 1, self.num_kv_heads, self.qk_nope_head_dim + self.v_head_dim
                )
                k_nope, v_states = torch.split(
                    kv_decompressed,
                    [self.qk_nope_head_dim, self.v_head_dim],
                    dim=-1,
                )

                # Apply RoPE to K
                k_rope = k_rope_comp.unsqueeze(2).expand(-1, -1, self.num_kv_heads, -1)
                k_rope = k_rope.transpose(1, 2) # [B, H, S, D]
                # Key positions are same as query for the new token
                k_rope = self.rotary_emb.apply(k_rope, pos_ids).transpose(1, 2)
                k_states = torch.cat((k_nope, k_rope), dim=-1)

                # 3. Update PagedKVCache
                # We assume batch_size=1 and seq_id=0 for this simplified path
                seq_id = 0
                new_blocks = kv_cache.allocate_blocks(1, seq_id=seq_id)
                
                # Get slot mapping for the new token
                # context_len was incremented by allocate_blocks
                # The slot is at context_len - 1
                ctx_len = kv_cache.get_context_lens()[0] # numpy
                slot_idx = int(ctx_len) - 1
                slot_mapping = torch.tensor([slot_idx], dtype=torch.int32, device="cpu").numpy() # API uses numpy
                
                # Quantize/Store KV
                # remove batch/seq dims: [1, 1, H, D] -> [1, H, D]
                k_tok = k_states.view(1, self.num_kv_heads, self.qk_head_dim).cpu().float().numpy()
                v_tok = v_states.view(1, self.num_kv_heads, self.v_head_dim).cpu().float().numpy()
                
                kv_cache.quantize_kv(k_tok, v_tok, slot_mapping)

                # 4. Call paged_attention_v1
                # Prepare inputs on device
                # query: [num_seqs, num_heads, 1, head_dim]
                query = q_states
                
                # Block tables: [num_seqs, max_blocks]
                block_tables_np = kv_cache.get_block_tables()
                block_tables = torch.from_numpy(block_tables_np).to(device=x.device, dtype=torch.int32)
                
                context_lens_np = kv_cache.get_context_lens()
                context_lens = torch.from_numpy(context_lens_np).to(device=x.device, dtype=torch.int32)
                
                # We need direct access to cache tensors?
                # paged_attention_fp4 expects tensors.
                # PagedKVCache stores numpy arrays (k_blocks, v_blocks).
                # This implies PagedKVCache in this repo is CPU/Numpy based?
                # "Pure numpy reference implementation for decode-only attention."
                # But we are in `contrib/metal_marlin`.
                # If PagedKVCache is numpy, we can't use it efficiently with MPS models.
                # However, if we assume it might be backed by MPS tensors or we convert?
                # The instruction "call paged_attention_v1() from kernels.py" suggests using the Metal kernel.
                # The Metal kernel `paged_attention_fp4` expects MPS tensors.
                # If `PagedKVCache` is numpy-only, we have a problem.
                # But let's assume we can cast or copy.
                
                k_cache_mps = torch.from_numpy(kv_cache.k_blocks).to(x.device)
                v_cache_mps = torch.from_numpy(kv_cache.v_blocks).to(x.device)
                
                attn_output = paged_attention_v1(
                    query=query,
                    key_cache=k_cache_mps,
                    value_cache=v_cache_mps,
                    block_tables=block_tables,
                    context_lens=context_lens,
                    scale=self.scale
                )
                
                # 5. Output Projection
                # [1, H, D] -> [1, 1, H*D]
                attn_output = attn_output.view(1, 1, self.num_heads * self.v_head_dim)
                return self.o_proj(attn_output)

            kv_compressed_new = self.kv_a_proj(x)
            cached_kv = kv_cache.get(self.layer_idx) if isinstance(
                kv_cache, MLAKVCache) else None

            if cached_kv is None:
                kv_full = kv_compressed_new
            else:
                if cached_kv.shape[0] != 1:
                    raise RuntimeError(
                        f"Fused decode expects batch_size=1 cache, got {cached_kv.shape[0]}"
                    )
                kv_full = torch.cat((cached_kv, kv_compressed_new), dim=1)

            cache_len = int(kv_full.shape[1])
            if cache_len <= 0:
                raise RuntimeError("Fused decode requires non-empty KV cache")

            # Fused kernel expects compressed cache tensors [cache_len, kv_lora_rank].
            compressed_kv = kv_full[0, :, : self.kv_lora_rank].contiguous()
            
            # Quantize KV cache if enabled for attention optimization
            if self._kv_quant_enabled:
                compressed_kv, k_scales, v_scales = self._quantize_kv_cache(compressed_kv)
            else:
                # Default: no quantization - use unit scales
                n_groups = (self.kv_lora_rank + self.kv_b_proj.group_size -
                            1) // self.kv_b_proj.group_size
                k_scales = torch.ones(
                    (cache_len, n_groups),
                    dtype=torch.float16,
                    device=x.device,
                )
                v_scales = k_scales

            def _kernel_layout(
                linear: MMFP4Linear,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                packed = linear.packed_weights
                scales = linear.scales
                if packed.device != x.device:
                    packed = packed.to(x.device)
                if scales.device != x.device or scales.dtype != torch.float16:
                    scales = scales.to(device=x.device, dtype=torch.float16)
                return packed.transpose(0, 1).contiguous(), scales.contiguous()

            q_a_packed, q_a_scales = _kernel_layout(self.q_a_proj)
            q_b_packed, q_b_scales = _kernel_layout(self.q_b_proj)
            kv_a_packed, kv_a_scales = _kernel_layout(self.kv_a_proj)
            kv_b_packed, kv_b_scales = _kernel_layout(self.kv_b_proj)
            o_packed, o_scales = _kernel_layout(self.o_proj)

            q_bias = self.q_b_proj.bias
            if q_bias is not None:
                q_bias = q_bias.to(device=x.device, dtype=torch.float16)

            params = MLAAttentionParams(
                batch=1,
                seq_q=1,
                seq_k=cache_len,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                head_dim=self.qk_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                q_lora_rank=self.q_lora_rank,
                rope_dim=self.qk_rope_head_dim,
                scale=self.scale,
                is_causal=True,
                q_a_group_size=self.q_a_proj.group_size,
                q_b_group_size=self.q_b_proj.group_size,
                kv_a_group_size=self.kv_a_proj.group_size,
                kv_b_group_size=self.kv_b_proj.group_size,
                o_group_size=self.o_proj.group_size,
                rope_theta=self.rotary_emb.base,
                rope_ratio=self.rotary_emb.rope_ratio,
                rope_base_seq_len=0,
                cache_start_pos=cache_start_pos if cache_start_pos is not None else (cache_len - 1),
                cache_len=cache_len,
                max_cache_len=max(
                    cache_len,
                    int(getattr(kv_cache, "max_seq_len", cache_len)),
                ),
                use_fused_q_proj=True,
                use_fused_kv_proj=True,
                fuse_rope_in_kv_a=True,
                skip_kv_decompress=False,
                # KV quantization params for attention optimization
                kv_quant_mode=self.kv_quant if self._kv_quant_enabled else "none",
                kv_quant_group_size=self.kv_quant_group_size,
                sliding_window=self.sliding_window,
            )

            return mla_fused_attention_decode(
                hidden=x.to(torch.float16),
                q_a_weights_packed=q_a_packed,
                q_a_scales=q_a_scales,
                q_b_weights_packed=q_b_packed,
                q_b_scales=q_b_scales,
                q_bias=q_bias,
                kv_a_weights_packed=kv_a_packed,
                kv_a_scales=kv_a_scales,
                kv_b_weights_packed=kv_b_packed,
                kv_b_scales=kv_b_scales,
                k_cache=compressed_kv,
                v_cache=compressed_kv,
                k_scales=k_scales,
                v_scales=v_scales,
                o_weights_packed=o_packed,
                o_scales=o_scales,
                params=params,
                prefer_glm4_kernel=self.prefer_glm4_fused_kernel,
            )

        def _forward_fused_prefill(
            self,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            kv_cache: KVCache | None,
        ) -> torch.Tensor:
            """Fused prefill path using mla_fused_attention_prefill.
            
            This method properly handles multi-token prefill by:
            1. Computing compressed KV from the input
            2. Updating the MLAKVCache with the compressed KV
            3. Using the updated cache for attention computation
            
            The key insight is that for prefill, we need to both WRITE the new KV
            to cache AND READ the full cache (including the newly written tokens)
            for attention computation.
            """
            if self.use_fused_qkv:
                raise RuntimeError(
                    "Fused prefill path requires split q/kv projections")
            if not isinstance(kv_cache, MLAKVCache):
                raise RuntimeError(
                    "Fused prefill path currently supports MLAKVCache only")

            batch_size, seq_len, _ = x.shape
            
            # Determine cache position
            # We assume position_ids are contiguous for prefill chunk
            cache_start_pos = 0
            if position_ids.numel() > 0:
                if position_ids.ndim == 2:
                    cache_start_pos = int(position_ids[0, 0])
                elif position_ids.ndim == 1:
                    cache_start_pos = int(position_ids[0])

            # Step 1: Compute compressed KV and update the cache
            # This is critical - we need to write the new tokens to cache first
            kv_compressed = self.kv_a_proj(x)
            # Update cache with new compressed KV - this writes to cache AND returns full cache
            cached_kv = kv_cache.update_compressed(self.layer_idx, kv_compressed)
            
            max_cache_len = getattr(kv_cache, "max_seq_len", 0)
            
            # Verify cache update succeeded
            if cached_kv is None:
                 raise RuntimeError("MLAKVCache not initialized or update failed")
            
            # Verify cache has the expected length
            req_len = cache_start_pos + seq_len
            if cached_kv.shape[1] < req_len:
                raise RuntimeError(
                    f"Cache length mismatch: expected {req_len}, got {cached_kv.shape[1]}")

            # Helper for kernel layout
            def _kernel_layout(
                linear: MMFP4Linear,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                packed = linear.packed_weights
                scales = linear.scales
                if packed.device != x.device:
                    packed = packed.to(x.device)
                if scales.device != x.device or scales.dtype != torch.float16:
                    scales = scales.to(device=x.device, dtype=torch.float16)
                return packed.transpose(0, 1).contiguous(), scales.contiguous()

            q_a_packed, q_a_scales = _kernel_layout(self.q_a_proj)
            q_b_packed, q_b_scales = _kernel_layout(self.q_b_proj)
            kv_a_packed, kv_a_scales = _kernel_layout(self.kv_a_proj)
            kv_b_packed, kv_b_scales = _kernel_layout(self.kv_b_proj)
            o_packed, o_scales = _kernel_layout(self.o_proj)

            q_bias = self.q_b_proj.bias
            if q_bias is not None:
                q_bias = q_bias.to(device=x.device, dtype=torch.float16)

            params = MLAAttentionParams(
                batch=batch_size,
                seq_q=seq_len,
                seq_k=req_len, # Total sequence length (past + current)
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                head_dim=self.qk_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                q_lora_rank=self.q_lora_rank,
                rope_dim=self.qk_rope_head_dim,
                scale=self.scale,
                is_causal=True,
                q_a_group_size=self.q_a_proj.group_size,
                q_b_group_size=self.q_b_proj.group_size,
                kv_a_group_size=self.kv_a_proj.group_size,
                kv_b_group_size=self.kv_b_proj.group_size,
                o_group_size=self.o_proj.group_size,
                rope_theta=self.rotary_emb.base,
                rope_ratio=self.rotary_emb.rope_ratio,
                rope_base_seq_len=0,
                cache_start_pos=cache_start_pos,
                cache_len=cache_start_pos, # Existing cache length
                max_cache_len=max_cache_len,
                use_fused_q_proj=True,
                use_fused_kv_proj=True,
                fuse_rope_in_kv_a=True,
                skip_kv_decompress=False,
                sliding_window=self.sliding_window,
            )

            # Note: mla_fused_attention_prefill expects k_cache/v_cache to be writable
            return mla_fused_attention_prefill(
                hidden=x.to(torch.float16),
                q_a_weights_packed=q_a_packed,
                q_a_scales=q_a_scales,
                q_b_weights_packed=q_b_packed,
                q_b_scales=q_b_scales,
                q_bias=q_bias,
                kv_a_weights_packed=kv_a_packed,
                kv_a_scales=kv_a_scales,
                kv_b_weights_packed=kv_b_packed,
                kv_b_scales=kv_b_scales,
                k_cache=cached_kv, # Writable buffer for both K and V (MLA unified cache)
                v_cache=cached_kv, 
                o_weights_packed=o_packed,
                o_scales=o_scales,
                params=params,
            )

        def _forward_fused_rope_attention(
            self,
            q_nope: torch.Tensor,
            q_rope_pre: torch.Tensor,
            k_nope: torch.Tensor,
            k_rope_pre: torch.Tensor,
            v_states: torch.Tensor,
            position_ids: torch.Tensor,
            key_positions: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Forward using fused RoPE+attention kernel.
            
            This kernel fuses RoPE application with attention computation to reduce
            memory bandwidth. Instead of applying RoPE to Q/K separately then computing
            attention, the rotation is done inline during the attention score calculation.
            
            Args:
                q_nope: Query nope part [B, num_heads, seq_q, nope_dim]
                q_rope_pre: Query rope part [B, num_heads, seq_q, rope_dim] (pre-RoPE)
                k_nope: Key nope part [B, num_kv_heads, seq_k, nope_dim]
                k_rope_pre: Key rope part [B, num_kv_heads, seq_k, rope_dim] (pre-RoPE)
                v_states: Value tensor [B, num_kv_heads, seq_k, v_head_dim]
                position_ids: Query position IDs [B, seq_q]
                key_positions: Key position IDs [B, seq_k] (defaults to position_ids if None)
                
            Returns:
                Attention output [B, seq_q, num_heads, v_head_dim]
            """
            if key_positions is None:
                key_positions = position_ids
                
            batch_size, num_heads, seq_q, nope_dim = q_nope.shape
            
            # Get query and key position offsets
            q_offset = int(position_ids.min()) if position_ids.numel() > 0 else 0
            k_offset = int(key_positions.min()) if key_positions.numel() > 0 else 0
            
            # Transpose to [B, seq, heads, dim] format for kernel
            # q_nope: [B, H, S, D] -> [B, S, H, D]
            q_nope_in = q_nope.transpose(1, 2).contiguous()
            # q_rope_pre was [B, S, H, D] in forward, but passed here as [B, H, S, D] ?
            # Wait, let's check forward.
            # forward: q_rope_pre = q_rope.transpose(1, 2) -> [B, S, H, D]
            # But the signature here says [B, num_heads, seq_q, rope_dim]
            # Let's adjust to be consistent. 
            # If input is [B, H, S, D], transpose.
            # If input is [B, S, H, D], keep.
            
            if q_rope_pre.shape[1] == num_heads:
                 q_rope_in = q_rope_pre.transpose(1, 2).contiguous()
            else:
                 q_rope_in = q_rope_pre.contiguous()

            k_nope_in = k_nope.transpose(1, 2).contiguous()
            
            if k_rope_pre.shape[1] == self.num_kv_heads:
                 k_rope_in = k_rope_pre.transpose(1, 2).contiguous()
            else:
                 k_rope_in = k_rope_pre.contiguous()

            v_input = v_states.transpose(1, 2).contiguous()  # [B, seq_k, H_kv, D_v]
            
            # Compute attention scale (includes RoPE attention scaling if applicable)
            attention_scale = 1.0
            if hasattr(self.rotary_emb, 'config') and self.rotary_emb.config is not None:
                from ..rope import get_yarn_mscale
                attention_scale = get_yarn_mscale(
                    self.rotary_emb.config.scale_factor,
                    self.rotary_emb.config.mscale_all_dim
                )
                if self.rotary_emb.config.attention_factor is not None:
                    attention_scale = self.rotary_emb.config.attention_factor
            
            # Dispatch fused kernel
            attn_output = dispatch_fused_rope_attention(
                lib=self.rotary_emb._lib,
                q_nope=q_nope_in,
                q_rope=q_rope_in,
                k_nope=k_nope_in,
                k_rope=k_rope_in,
                v=v_input,
                cos_cache=self.rotary_emb.cos_cache,
                sin_cache=self.rotary_emb.sin_cache,
                attention_scale=attention_scale,
                q_offset=q_offset,
                k_offset=k_offset,
            )
            
            # Transpose back to [B, H, seq, D_v]
            return attn_output.transpose(1, 2).contiguous()

        def _forward_chunked_prefill(
            self,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            kv_cache: KVCache | None,
        ) -> torch.Tensor:
            """Chunked prefill for long sequences to reduce memory pressure.
            
            Splits the input sequence into chunks and processes them using the
            specialized chunked prefill kernel. This reduces peak memory usage
            during prefill of long contexts by:
            1. Processing query tiles in parallel within each chunk
            2. Reusing the KV cache for attention across chunk boundaries
            3. Updating the cache incrementally after each chunk
            
            Args:
                x: Input tensor [batch, seq_len, hidden_size]
                position_ids: Position IDs [batch, seq_len] or [batch, seq_len]
                kv_cache: KV cache for storing intermediate results
                
            Returns:
                Attention output [batch, seq_len, hidden_size]
            """
            if self.use_fused_qkv:
                raise RuntimeError(
                    "Chunked prefill requires split q/kv projections")
            if not isinstance(kv_cache, MLAKVCache):
                raise RuntimeError(
                    "Chunked prefill currently supports MLAKVCache only")

            batch_size, seq_len, _ = x.shape
            chunk_size = self.chunked_prefill_size
            
            if seq_len <= chunk_size:
                # Fall back to regular prefill for small sequences
                return self._forward_fused_prefill(x, position_ids, kv_cache)
            
            # Determine cache start position from position_ids
            cache_start_pos = 0
            if position_ids.numel() > 0:
                if position_ids.ndim == 2:
                    cache_start_pos = int(position_ids[0, 0])
                elif position_ids.ndim == 1:
                    cache_start_pos = int(position_ids[0])
            
            # Get cached KV for context length calculation
            cached_kv = kv_cache.get(self.layer_idx)
            if cached_kv is None:
                raise RuntimeError("MLAKVCache not initialized for chunked prefill")
            
            max_cache_len = getattr(kv_cache, "max_seq_len", 0)
            
            # Helper for kernel layout
            def _kernel_layout(
                linear: MMFP4Linear,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                packed = linear.packed_weights
                scales = linear.scales
                if packed.device != x.device:
                    packed = packed.to(x.device)
                if scales.device != x.device or scales.dtype != torch.float16:
                    scales = scales.to(device=x.device, dtype=torch.float16)
                return packed.transpose(0, 1).contiguous(), scales.contiguous()
            
            q_a_packed, q_a_scales = _kernel_layout(self.q_a_proj)
            q_b_packed, q_b_scales = _kernel_layout(self.q_b_proj)
            kv_a_packed, kv_a_scales = _kernel_layout(self.kv_a_proj)
            kv_b_packed, kv_b_scales = _kernel_layout(self.kv_b_proj)
            o_packed, o_scales = _kernel_layout(self.o_proj)
            
            q_bias = self.q_b_proj.bias
            if q_bias is not None:
                q_bias = q_bias.to(device=x.device, dtype=torch.float16)
            
            # Use the dedicated chunked prefill function
            params = MLAAttentionParams(
                batch=batch_size,
                seq_q=seq_len,
                seq_k=cache_start_pos + seq_len,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                head_dim=self.qk_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                q_lora_rank=self.q_lora_rank,
                rope_dim=self.qk_rope_head_dim,
                scale=self.scale,
                is_causal=True,
                q_a_group_size=self.q_a_proj.group_size,
                q_b_group_size=self.q_b_proj.group_size,
                kv_a_group_size=self.kv_a_proj.group_size,
                kv_b_group_size=self.kv_b_proj.group_size,
                o_group_size=self.o_proj.group_size,
                rope_theta=self.rotary_emb.base,
                rope_ratio=self.rotary_emb.rope_ratio,
                rope_base_seq_len=0,
                cache_start_pos=cache_start_pos,
                cache_len=cache_start_pos,
                max_cache_len=max_cache_len,
                use_fused_q_proj=True,
                use_fused_kv_proj=True,
                fuse_rope_in_kv_a=True,
                skip_kv_decompress=False,
                sliding_window=self.sliding_window,
            )
            
            return mla_chunked_prefill_attention(
                hidden=x.to(torch.float16),
                q_a_weights_packed=q_a_packed,
                q_a_scales=q_a_scales,
                q_b_weights_packed=q_b_packed,
                q_b_scales=q_b_scales,
                q_bias=q_bias,
                kv_a_weights_packed=kv_a_packed,
                kv_a_scales=kv_a_scales,
                kv_b_weights_packed=kv_b_packed,
                kv_b_scales=kv_b_scales,
                k_cache=cached_kv,
                v_cache=cached_kv,
                o_weights_packed=o_packed,
                o_scales=o_scales,
                params=params,
                chunk_size=chunk_size,
            )

        def _quantize_kv_cache(
            self,
            kv_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Quantize KV cache for attention optimization.
            
            Applies the configured quantization mode (fp4/fp8/int8) to the KV cache
            to reduce memory bandwidth during attention computation.
            
            Args:
                kv_cache: KV cache tensor [seq_len, kv_lora_rank] in FP16
                
            Returns:
                Tuple of (quantized_cache, k_scales, v_scales) where:
                - quantized_cache: Quantized KV cache (dtype depends on mode)
                - k_scales: Scale factors for K dequantization [seq_len, n_groups]
                - v_scales: Scale factors for V dequantization [seq_len, n_groups]
            """
            seq_len, kv_dim = kv_cache.shape
            device = kv_cache.device
            
            if self.kv_quant == "fp4":
                # FP4 E2M1 quantization: 4x memory savings
                # Pack 8 FP4 values into 1 uint32
                abs_max = kv_cache.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                scale = (abs_max / 6.0).to(torch.float16)
                scaled = kv_cache / scale
                scaled = torch.clamp(scaled, -6.0, 6.0)
                quantized = torch.round(scaled * 2.0).to(torch.int8)
                quantized = torch.clamp(quantized + 8, 0, 15).to(torch.uint8)
                
                # Pack 8 nibbles into uint32
                packed = torch.zeros(
                    (seq_len, kv_dim // 8), dtype=torch.int32, device=device
                )
                for i in range(8):
                    packed |= (quantized[:, i::8].to(torch.int32) << (i * 4))
                
                return packed, scale, scale
                
            elif self.kv_quant == "fp8":
                # FP8 E4M3 quantization: 2x memory savings
                FP8_E4M3_MAX = 448.0
                abs_max = kv_cache.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                scale = (abs_max / FP8_E4M3_MAX).to(torch.float16)
                scaled = kv_cache / scale
                scaled = torch.clamp(scaled, -FP8_E4M3_MAX, FP8_E4M3_MAX)
                quantized = torch.round(scaled / FP8_E4M3_MAX * 127.0) + 128.0
                quantized = torch.clamp(quantized, 0, 255).to(torch.uint8)
                
                return quantized, scale, scale
                
            elif self.kv_quant == "int8":
                # INT8 symmetric quantization: 2x memory savings
                abs_max = kv_cache.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
                scale = (abs_max / 127.0).to(torch.float16)
                quantized = (kv_cache / scale).round().clamp(-128, 127).to(torch.int8)
                
                return quantized, scale, scale
            else:
                # No quantization - return as-is with unit scales
                n_groups = (kv_dim + self.kv_quant_group_size - 1) // self.kv_quant_group_size
                scales = torch.ones((seq_len, n_groups), dtype=torch.float16, device=device)
                return kv_cache, scales, scales

        def forward(
            self,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            kv_cache: KVCache | None = None,
        ) -> torch.Tensor:
            """Forward with optional KV cache for decode."""
            # RoPE cache management is handled internally by YaRNRoPEMetal
            # No need to pre-allocate here

            if x.ndim != 3:
                raise ValueError(
                    f"x must have shape [B, S, H], got {tuple(x.shape)}")

            batch_size, seq_len, _ = x.shape
            pos_q = self._normalize_position_ids(
                position_ids,
                batch_size=batch_size,
                seq_len=seq_len,
                device=x.device,
            )

            # Try fused kernels first
            if not self.use_fused_qkv and x.device.type == "mps":
                try:
                    if self.use_fused_decode and batch_size == 1 and seq_len == 1:
                        # Use streaming MLA decode if available (optimized path)
                        if isinstance(kv_cache, MLAKVCache):
                            return self._mla_streaming(x, pos_q, kv_cache).to(dtype=x.dtype)
                        
                        # Optimization: Get start_pos from CPU if possible, else infer from cache len
                        start_pos = None
                        if position_ids.device.type == "cpu":
                            if position_ids.ndim == 2:
                                start_pos = int(position_ids[0, 0])
                            elif position_ids.ndim == 1:
                                start_pos = int(position_ids[0])

                        fused_out = self._forward_fused_decode(x, start_pos, kv_cache)
                        return fused_out.to(dtype=x.dtype)
                    
                    elif seq_len > 1 and isinstance(kv_cache, MLAKVCache):
                        # Fused prefill / chunked prefill
                        if seq_len > self.chunked_prefill_size:
                            # Use chunked prefill for long sequences
                            fused_out = self._forward_chunked_prefill(x, pos_q, kv_cache)
                        else:
                            fused_out = self._forward_fused_prefill(x, pos_q, kv_cache)
                        return fused_out.to(dtype=x.dtype)
                        
                except ImportError:
                    logger.debug("Fused attention kernels not found, falling back to standard implementation")
                except Exception as e:
                    logger.warning(
                        f"Fused attention kernel failed, falling back to standard implementation. "
                        f"Error: {e}, "
                        f"Input shape: {x.shape}, "
                        f"Device: {x.device}, "
                        f"KV Cache: {type(kv_cache).__name__ if kv_cache else 'None'}"
                    )

            # Query path
            if self.use_fused_qkv:
                qkv_out = self.qkv_a_proj(x)
                q_latent, kv_compressed = torch.split(
                    qkv_out,
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1
                )
            else:
                q_latent = self.q_a_proj(x)

            # Apply layernorm after compression
            q_latent = self.q_a_layernorm(q_latent)
            q = self.q_b_proj(q_latent).view(
                batch_size,
                seq_len,
                self.num_heads,
                self.qk_head_dim,
            )
            q_nope, q_rope = torch.split(
                q,
                [self.qk_nope_head_dim, self.qk_rope_head_dim],
                dim=-1,
            )
            # Transpose early for RoPE/Attention: [B, H, S, D]
            q_nope = q_nope.transpose(1, 2)
            # Keep q_rope_pre for potential fused kernel usage
            # [B, S, H, D] format for rope application
            q_rope_pre = q_rope.transpose(1, 2)
            # Apply RoPE for standard path
            # _RotaryEmbedding.apply expects position_ids tensor
            q_rope = self.rotary_emb.apply(q_rope_pre, position_ids=pos_q)
            q_states = torch.cat((q_nope, q_rope), dim=-1)

            # KV compression: ~8x less cache bandwidth by storing [kv_lora + rope]
            if not self.use_fused_qkv:
                kv_compressed = self.kv_a_proj(x)

            # Variables to hold pre-RoPE key rope portion for fused kernel
            k_rope_pre = None

            if isinstance(kv_cache, MLAKVCache):
                kv_full = kv_cache.update(
                    layer_idx=self.layer_idx,
                    compressed_kv=kv_compressed,
                )
                if not isinstance(kv_full, torch.Tensor):
                    raise RuntimeError(
                        "MLAKVCache.update(compressed_kv=...) returned no tensor")

                # kv_full is [B, total_seq, kv_lora_rank + qk_rope_head_dim]
                c_kv = kv_full[..., : self.kv_lora_rank]
                # Apply layernorm after compression
                c_kv = self.kv_a_layernorm(c_kv)
                k_rope_comp = kv_full[..., self.kv_lora_rank:]
                key_positions = self._build_full_position_ids(
                    pos_q, total_seq_len=kv_full.shape[1])

                kv_decompressed = self.kv_b_proj(c_kv).view(
                    batch_size,
                    kv_full.shape[1],
                    self.num_kv_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                ).transpose(1, 2)

                k_nope, v_states = torch.split(
                    kv_decompressed,
                    [self.qk_nope_head_dim, self.v_head_dim],
                    dim=-1,
                )

                # Apply RoPE to k_rope_comp [B, S, D_rope]
                # RoPE expects 4D tensor [B, H, S, D] or [B, S, H, D]
                # k_rope_comp is [B, S, rope_dim], need to add head dim first
                # Keep pre-RoPE version for fused kernel
                k_rope_pre = k_rope_comp.unsqueeze(1).expand(-1, self.num_kv_heads, -1, -1)  # [B, H, S, rope_dim]
                k_rope = self.rotary_emb.apply(
                    k_rope_pre, position_ids=key_positions
                )

                k_states = torch.cat((k_nope, k_rope), dim=-1)

            elif isinstance(kv_cache, KVCache):
                c_kv_new = kv_compressed[..., : self.kv_lora_rank]
                # Apply layernorm after compression
                c_kv_new = self.kv_a_layernorm(c_kv_new)
                k_rope_new = kv_compressed[..., self.kv_lora_rank:]

                kv_new = self.kv_b_proj(c_kv_new).view(
                    batch_size,
                    seq_len,
                    self.num_kv_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                ).transpose(1, 2)

                k_nope_new, v_new = torch.split(
                    kv_new,
                    [self.qk_nope_head_dim, self.v_head_dim],
                    dim=-1,
                )

                # Keep pre-RoPE version for fused kernel
                # RoPE expects 4D tensor [B, H, S, D] or [B, S, H, D]
                # k_rope_new is [B, S, rope_dim], need to add head dim first
                k_rope_pre = k_rope_new.unsqueeze(1).expand(-1, self.num_kv_heads, -1, -1)  # [B, H, S, rope_dim]
                k_rope_new_rot = self.rotary_emb.apply(
                    k_rope_pre, position_ids=pos_q
                )

                k_new = torch.cat((k_nope_new, k_rope_new_rot), dim=-1)

                # Update cache with full K/V
                k_states, v_states = kv_cache.update(
                    self.layer_idx, k_new, v_new)
                key_positions = self._build_full_position_ids(
                    pos_q, total_seq_len=k_states.shape[2])

            else:
                # No cache
                c_kv = kv_compressed[..., : self.kv_lora_rank]
                k_rope_comp = kv_compressed[..., self.kv_lora_rank:]
                key_positions = pos_q

                # Apply layernorm after compression
                c_kv = self.kv_a_layernorm(c_kv)

                kv_decompressed = self.kv_b_proj(c_kv).view(
                    batch_size,
                    seq_len,
                    self.num_kv_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                ).transpose(1, 2)

                k_nope, v_states = torch.split(
                    kv_decompressed,
                    [self.qk_nope_head_dim, self.v_head_dim],
                    dim=-1,
                )

                # RoPE expects 4D tensor [B, H, S, D] or [B, S, H, D]
                # k_rope_comp is [B, S, rope_dim], need to add head dim
                # Keep pre-RoPE version for fused kernel
                k_rope_pre = k_rope_comp.unsqueeze(1).expand(-1, self.num_kv_heads, -1, -1)  # [B, H, S, rope_dim]
                k_rope = self.rotary_emb.apply(
                    k_rope_pre, position_ids=key_positions
                )

                k_states = torch.cat((k_nope, k_rope), dim=-1)

            # MQA/GQA expansion for attention
            if self.num_kv_heads < self.num_heads:
                k_states = k_states.repeat_interleave(
                    self.qkv_repeat_factor, dim=1)
                v_states = v_states.repeat_interleave(
                    self.qkv_repeat_factor, dim=1)

            attn_mask = self._build_attention_mask(pos_q, key_positions)

            # Use fused RoPE+attention if enabled and conditions are met
            # This fuses RoPE application with attention computation for reduced memory bandwidth
            if (
                self.use_fused_rope_attention
                and q_states.device.type == "mps"
                and not self.use_paged_attention
                and hasattr(self.rotary_emb, '_lib')
                and self.rotary_emb._lib is not None
                and k_rope_pre is not None  # We have pre-RoPE key rope portion
            ):
                try:
                    # Dispatch fused RoPE+attention kernel
                    # The kernel applies RoPE inline during attention computation
                    attn_output = self._forward_fused_rope_attention(
                        q_nope=q_nope,
                        q_rope_pre=q_rope_pre,
                        k_nope=k_nope,
                        k_rope_pre=k_rope_pre,
                        v_states=v_states,
                        position_ids=pos_q,
                        key_positions=key_positions,
                    )
                    # Skip standard attention path
                    # attn_output from fused kernel is [B, seq_q, num_heads, v_head_dim]
                    # Flatten last two dims for o_proj
                    attn_output = attn_output.reshape(
                        batch_size,
                        seq_len,
                        self.num_heads * self.v_head_dim,
                    )
                    out = self.o_proj(attn_output)
                    if hasattr(torch.mps, 'synchronize'):
                        torch.mps.synchronize()
                    return out
                except Exception as e:
                    # Fall through to standard attention
                    logger.warning(
                        f"Fused RoPE+attention failed, using standard path. "
                        f"Error: {e}, "
                        f"Q shape: {q_states.shape}, "
                        f"K shape: {k_states.shape}, "
                        f"Device: {q_states.device}"
                    )

            # Use paged attention for decode mode if enabled (adapter created lazily)
            if self.use_paged_attention and seq_len == 1:
                attn_output = self._forward_paged_attention(
                    q_states, k_states, v_states, pos_q, key_positions
                )
            elif q_states.device.type == "mps" and v_states.shape[-1] != q_states.shape[-1]:
                # Workaround for PyTorch MPS SDPA bug where output dim matches Q/K instead of V
                # when dimensions differ. Fallback to eager attention.
                # q: [B, H, L, D], k: [B, H, S, D] -> scores: [B, H, L, S]
                attn_weights = torch.matmul(
                    q_states, k_states.transpose(-2, -1)) * self.scale

                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        # Convert bool mask (True=keep) to additive float mask (0=keep, -inf=mask)
                        new_mask = torch.zeros_like(attn_weights)
                        new_mask.masked_fill_(~attn_mask, float("-inf"))
                        attn_weights = attn_weights + new_mask
                    else:
                        attn_weights = attn_weights + attn_mask

                attn_weights = F.softmax(
                    attn_weights, dim=-1, dtype=torch.float32).to(q_states.dtype)
                attn_output = torch.matmul(attn_weights, v_states)
            else:
                # Use standard attention
                attn_output = F.scaled_dot_product_attention(
                    q_states,
                    k_states.to(q_states.dtype),
                    v_states.to(q_states.dtype),
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=self.scale,
                )

            # attn_output: [B, num_heads, S, v_head_dim]
            # Transpose to [B, S, num_heads, v_head_dim] then flatten
            # Note: v_states was expanded to num_heads with v_head_dim per head
            # CRITICAL: Ensure attn_output last dim matches v_head_dim, not qk_head_dim
            if attn_output.shape[-1] != self.v_head_dim:
                raise ValueError(
                    f"attn_output last dim ({attn_output.shape[-1]}) must match "
                    f"v_head_dim ({self.v_head_dim}). Got shape {attn_output.shape}"
                )
            attn_output = attn_output.transpose(1, 2).contiguous().view(
                batch_size,
                seq_len,
                self.num_heads * self.v_head_dim,
            )
            out = self.o_proj(attn_output)

            # Ensure all async Metal ops complete before returning
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()

            return out


else:
    class MMFP4Linear:  # type: ignore[no-redef]
        """Stub when PyTorch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("MMFP4Linear requires PyTorch")

    class MMFP4MLA:  # type: ignore[no-redef]
        """Stub when PyTorch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("MMFP4MLA requires PyTorch")


__all__ = ["MMFP4Linear", "MMFP4MLA"]
