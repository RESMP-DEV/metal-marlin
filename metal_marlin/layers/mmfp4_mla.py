"""MMFP4 MLA (Multi-head Latent Attention) layer for GLM-style models.

This module implements an FP4-quantized MLA attention block using low-rank
query/KV projections and compressed KV-cache storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._compat import HAS_TORCH, torch

if TYPE_CHECKING:
    from ..kv_cache import KVCache


if HAS_TORCH and torch is not None:
    import torch.nn as nn
    import torch.nn.functional as F

    from ..kv_cache import KVCache, MLAKVCache
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

            max_pos = int(position_ids.max().item()) + 1
            if max_pos > self._cached_seq_len or self._cos_cached.device != position_ids.device:
                grow_target = max(max_pos, max(1, self._cached_seq_len * 2))
                if grow_target > self.max_position_embeddings:
                    grow_target = max_pos
                self._ensure_cache(grow_target, device=position_ids.device)

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
            self._paged_adapter: Any = None

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
            self.rotary_emb = _RotaryEmbedding(
                dim=qk_rope_head_dim,
                base=rope_theta,
                rope_ratio=rope_ratio
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

        def forward(
            self,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            kv_cache: KVCache | None = None,
        ) -> torch.Tensor:
            """Forward with optional KV cache for decode."""
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
            # q_rope needs to be [B, S, H, D] or [B, H, S, D] for RoPE
            # _RotaryEmbedding handles [B, H, S, D] or [B, S, D]
            q_rope = q_rope.transpose(1, 2)
            q_rope = self.rotary_emb.apply(q_rope, pos_q)
            q_states = torch.cat((q_nope, q_rope), dim=-1)

            # KV compression: ~8x less cache bandwidth by storing [kv_lora + rope]
            if not self.use_fused_qkv:
                kv_compressed = self.kv_a_proj(x)

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
                k_rope = self.rotary_emb.apply(
                    k_rope_comp, key_positions
                )
                # Expand to num_kv_heads: [B, S, D] -> [B, 1, S, D] -> [B, H, S, D]
                # But k_nope is [B, H, S, D], so we want k_rope to be [B, H, S, D]
                # _RotaryEmbedding.apply on [B, S, D] returns [B, S, D]
                k_rope = k_rope.unsqueeze(
                    1).expand(-1, self.num_kv_heads, -1, -1)

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

                k_rope_new_rot = self.rotary_emb.apply(
                    k_rope_new, pos_q
                ).unsqueeze(1).expand(-1, self.num_kv_heads, -1, -1)

                k_new = torch.cat((k_nope_new, k_rope_new_rot), dim=-1)

                # Update cache with full K/V
                k_states, v_states = kv_cache.update(
                    self.layer_idx, k_new, v_new)
                key_positions = self._build_full_position_ids(
                    pos_q, total_seq_len=k_states.shape[2])

            else:
                # No cache
                c_kv = kv_compressed[..., : self.kv_lora_rank]
                # Apply layernorm after compression
                c_kv = self.kv_a_layernorm(c_kv)
                k_rope_comp = kv_compressed[..., self.kv_lora_rank:]
                key_positions = pos_q

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

                k_rope = self.rotary_emb.apply(
                    k_rope_comp, key_positions
                ).unsqueeze(1).expand(-1, self.num_kv_heads, -1, -1)

                k_states = torch.cat((k_nope, k_rope), dim=-1)

            # MQA/GQA expansion for attention
            if self.num_kv_heads < self.num_heads:
                k_states = k_states.repeat_interleave(
                    self.qkv_repeat_factor, dim=1)
                v_states = v_states.repeat_interleave(
                    self.qkv_repeat_factor, dim=1)

            attn_mask = self._build_attention_mask(pos_q, key_positions)

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
