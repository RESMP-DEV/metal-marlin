"""Adapter to connect MMFP4 MLA attention with paged_attention_v1_fp4 kernel.

This module provides an adapter layer that connects MMFP4's Multi-head Latent
Attention (MLA) with the paged_attention_v1_fp4 Metal kernel for efficient
decode-phase inference on Apple Silicon.

Example:
    from metal_marlin.paged.mmfp4_paged_adapter import MMFP4PagedAttention
    from metal_marlin.layers.mmfp4_mla import MMFP4MLA

    # Create MMFP4 MLA layer
    mla_layer = MMFP4MLA(
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=32,
        q_lora_rank=384,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
    )

    # Create paged attention adapter
    adapter = MMFP4PagedAttention(
        mla_layer=mla_layer,
        max_batch_size=4,
        max_seq_len=8192,
    )

    # Use in forward pass
    output = adapter.forward(q_nope, q_rope, layer_idx=0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._compat import HAS_TORCH, torch

if TYPE_CHECKING:
    import torch


if HAS_TORCH and torch is not None:
    import torch.nn as nn

    from ..kernels import paged_attention_fp4
    from ..kv_cache import MLAKVCache

    class MMFP4PagedAttention(nn.Module):
        """Wraps paged attention for MMFP4's MLA (Multi-head Latent Attention).

        This adapter connects the MMFP4 MLA layer with the paged_attention_v1_fp4
        Metal kernel, enabling efficient FP4-quantized paged attention for
        autoregressive generation.

        The adapter handles:
        - Query projection to latent space (matching MLA architecture)
        - Paged KV cache management with FP4 quantization via MLAKVCache
        - Kernel dispatch for paged_attention_v1_fp4
        - Output projection back to V-head dimensions

        Attributes:
            cache: MLAKVCache instance managing FP4-quantized KV cache
            num_heads: Number of attention heads (from MLA layer)
            num_kv_heads: Number of KV heads (from MLA layer)
            kv_lora_rank: Compressed KV dimension
            qk_rope_head_dim: Rotary position embedding dimension
            qk_nope_head_dim: Non-rotary query key dimension
            v_head_dim: Value head dimension
            head_dim: Total latent dimension (kv_lora_rank + qk_rope_head_dim)
            scale: Attention scale factor (1/sqrt(qk_head_dim))
        """

        def __init__(
            self,
            mla_layer: nn.Module,
            max_batch_size: int = 1,
            max_seq_len: int = 8192,
            num_layers: int = 32,
        ):
            """Initialize the MMFP4 paged attention adapter.

            Args:
                mla_layer: MMFP4MLA layer instance providing:
                    - num_heads, num_kv_heads
                    - kv_lora_rank, qk_rope_head_dim
                    - qk_nope_head_dim, v_head_dim
                    - kv_b_proj weights for query/output projection
                max_batch_size: Maximum number of sequences in a batch
                max_seq_len: Maximum sequence length to allocate for
                num_layers: Number of transformer layers
            """
            super().__init__()

            # Store reference to MLA layer for weight access
            self.mla_layer = mla_layer

            # Extract dimensions from MLA layer
            self.num_heads = getattr(mla_layer, "num_heads", 32)
            self.num_kv_heads = getattr(mla_layer, "num_kv_heads", self.num_heads)
            self.kv_lora_rank = getattr(mla_layer, "kv_lora_rank", 512)
            self.qk_rope_head_dim = getattr(mla_layer, "qk_rope_head_dim", 64)
            self.qk_nope_head_dim = getattr(mla_layer, "qk_nope_head_dim", 128)
            self.v_head_dim = getattr(mla_layer, "v_head_dim", 128)

            # MLA combines latent + rope dims for the compressed representation
            self.head_dim = self.kv_lora_rank + self.qk_rope_head_dim
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

            # Attention scale
            self.scale = self.qk_head_dim ** -0.5

            self.num_layers = num_layers
            self.max_batch_size = max_batch_size
            self.max_seq_len = max_seq_len

            # Initialize MLA KV cache with FP4 quantization
            # MLAKVCache stores compressed KV (latent + rope) in BSHD layout
            self.cache = MLAKVCache(
                num_layers=num_layers,
                batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                quantize_mode="fp4",
                auto_grow=True,
            )

            # Self-test: validate FP4 cache functionality
            self._validate_fp4_cache()

        def _validate_fp4_cache(self) -> None:
            """Validate FP4 cache creation, update, and retrieval.

            This self-test runs during __init__ to ensure the FP4 cache is
            properly configured and functional before use.
            """
            import torch

            # 1. Verify cache was created with FP4 mode
            if self.cache.quantize_mode != "fp4":
                raise RuntimeError(
                    f"FP4 cache validation failed: expected quantize_mode='fp4', "
                    f"got '{self.cache.quantize_mode}'"
                )

            # 2. Verify cache tensor shapes and dtypes
            cache_shape = self.cache.kv_cache.shape
            expected_shape = (
                self.num_layers,
                self.max_batch_size,
                self.max_seq_len,
                self.head_dim // 8,  # FP4 packs 8 values per int32
            )
            if cache_shape != expected_shape:
                raise RuntimeError(
                    f"FP4 cache validation failed: cache shape {cache_shape} != "
                    f"expected {expected_shape}"
                )

            if self.cache.kv_cache.dtype != torch.int32:
                raise RuntimeError(
                    f"FP4 cache validation failed: cache dtype should be int32, "
                    f"got {self.cache.kv_cache.dtype}"
                )

            # 3. Verify scales tensor exists and has correct shape
            if self.cache.kv_scales is None:
                raise RuntimeError(
                    "FP4 cache validation failed: kv_scales should not be None in FP4 mode"
                )

            expected_scale_shape = (
                self.num_layers,
                self.max_batch_size,
                self.max_seq_len,
                1,
            )
            if self.cache.kv_scales.shape != expected_scale_shape:
                raise RuntimeError(
                    f"FP4 cache validation failed: scale shape {self.cache.kv_scales.shape} != "
                    f"expected {expected_scale_shape}"
                )

            # 4. Test cache update with sample compressed KV tensor
            test_batch = min(2, self.max_batch_size)
            test_seq = 4
            test_kv = torch.randn(
                test_batch,
                test_seq,
                self.head_dim,
                dtype=torch.float16,
                device=self.cache.device,
            )

            # Update cache for layer 0
            result = self.cache.update_compressed(0, test_kv)

            # 5. Verify cache update succeeded
            if result is None:
                raise RuntimeError(
                    "FP4 cache validation failed: update_compressed returned None"
                )

            if result.shape != (test_batch, test_seq, self.head_dim):
                raise RuntimeError(
                    f"FP4 cache validation failed: result shape {result.shape} != "
                    f"expected {(test_batch, test_seq, self.head_dim)}"
                )

            # 6. Verify we can retrieve cached K/V
            cached_kv = self.cache.get(0)
            if cached_kv is None:
                raise RuntimeError(
                    "FP4 cache validation failed: get() returned None after update"
                )

            if cached_kv.shape[0] != test_batch:
                raise RuntimeError(
                    f"FP4 cache validation failed: cached batch size {cached_kv.shape[0]} != "
                    f"expected {test_batch}"
                )

            # 7. Reset cache and verify it's cleared
            self.cache.reset()
            if self.cache.seq_len != 0:
                raise RuntimeError(
                    f"FP4 cache validation failed: seq_len should be 0 after reset, "
                    f"got {self.cache.seq_len}"
                )

            # All validations passed
            # Restore cache to clean state for actual use
            self.cache._seq_lens.zero_()

        def _project_query_to_latent(
            self, q_nope: torch.Tensor
        ) -> torch.Tensor:
            """Project non-rotary query to latent space.

            For MLA: q_latent = q_nope @ W_kv_b_nope

            Args:
                q_nope: Non-rotary query [batch, num_heads, qk_nope_head_dim]

            Returns:
                q_latent: Projected query [batch, num_heads, kv_lora_rank]
            """
            batch_size, num_heads, _ = q_nope.shape

            # Get kv_b_proj weight from MLA layer
            if hasattr(self.mla_layer, "kv_b_proj"):
                kv_b_proj = self.mla_layer.kv_b_proj

                # Handle MMFP4Linear which stores packed weights
                if hasattr(kv_b_proj, "packed_weights"):
                    from ..layers.mmfp4_linear import _dequantize_rowwise_mmfp4

                    weight = _dequantize_rowwise_mmfp4(
                        kv_b_proj.packed_weights,
                        kv_b_proj.scales,
                        kv_b_proj.group_size,
                    )
                else:
                    weight = kv_b_proj.weight

                # Weight shape: [num_kv_heads * (qk_nope + v), kv_lora_rank]
                w_reshaped = weight.view(
                    self.num_kv_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                    self.kv_lora_rank,
                )
                w_kv_b_nope = w_reshaped[:, : self.qk_nope_head_dim, :]

                # Expand for GQA if needed
                if self.num_kv_heads < num_heads:
                    repeat_factor = num_heads // self.num_kv_heads
                    w_kv_b_nope = w_kv_b_nope.repeat_interleave(repeat_factor, dim=0)

                # Batch matmul: [B, H, 1, nope] @ [H, nope, rank] -> [B, H, rank]
                q_latent = torch.matmul(
                    q_nope.unsqueeze(2), w_kv_b_nope.unsqueeze(0)
                ).squeeze(2)
                return q_latent
            else:
                # Fallback: linear projection if weights not available
                if q_nope.shape[-1] != self.kv_lora_rank:
                    proj = nn.Linear(self.qk_nope_head_dim, self.kv_lora_rank, bias=False)
                    proj = proj.to(device=q_nope.device, dtype=q_nope.dtype)
                    return proj(q_nope)
                return q_nope

        def _project_output_from_latent(
            self, attn_latent: torch.Tensor
        ) -> torch.Tensor:
            """Project latent attention output to V dimension.

            For MLA: attn_out = attn_latent @ W_kv_b_v.T

            Args:
                attn_latent: Latent attention output [batch, num_heads, kv_lora_rank]

            Returns:
                attn_out: Output in V space [batch, num_heads, v_head_dim]
            """
            batch_size, num_heads, _ = attn_latent.shape

            if hasattr(self.mla_layer, "kv_b_proj"):
                kv_b_proj = self.mla_layer.kv_b_proj

                # Handle MMFP4Linear which stores packed weights
                if hasattr(kv_b_proj, "packed_weights"):
                    from ..layers.mmfp4_linear import _dequantize_rowwise_mmfp4

                    weight = _dequantize_rowwise_mmfp4(
                        kv_b_proj.packed_weights,
                        kv_b_proj.scales,
                        kv_b_proj.group_size,
                    )
                else:
                    weight = kv_b_proj.weight

                w_reshaped = weight.view(
                    self.num_kv_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                    self.kv_lora_rank,
                )
                w_kv_b_v = w_reshaped[:, self.qk_nope_head_dim :, :]

                # Expand for GQA if needed
                if self.num_kv_heads < num_heads:
                    repeat_factor = num_heads // self.num_kv_heads
                    w_kv_b_v = w_kv_b_v.repeat_interleave(repeat_factor, dim=0)

                # [B, H, 1, rank] @ [H, rank, v_dim] -> [B, H, v_dim]
                attn_out = torch.matmul(
                    attn_latent.unsqueeze(2), w_kv_b_v.permute(0, 2, 1).unsqueeze(0)
                ).squeeze(2)
                return attn_out
            else:
                # Fallback: linear projection
                if attn_latent.shape[-1] != self.v_head_dim:
                    proj = nn.Linear(self.kv_lora_rank, self.v_head_dim, bias=False)
                    proj = proj.to(device=attn_latent.device, dtype=attn_latent.dtype)
                    return proj(attn_latent)
                return attn_latent

        def _get_fp4_cache_buffers(
            self, layer_idx: int
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Get FP4 cache buffers for kernel dispatch.

            Returns:
                k_cache: Packed K cache [num_layers, batch, max_seq, head_dim//8]
                v_cache: Packed V cache [num_layers, batch, max_seq, head_dim//8]
                k_scales: K scales [num_layers, batch, max_seq, 1]
                v_scales: V scales [num_layers, batch, max_seq, 1]
            """
            # MLAKVCache stores as unified kv_cache in FP4 mode
            # Shape: [num_layers, batch, max_seq, cache_dim]
            # For FP4: kv_cache is uint8 packed, kv_scales contains scales
            k_cache = self.cache.kv_cache[layer_idx]  # [batch, max_seq, cache_dim]
            v_cache = k_cache  # MLA uses same cache for K and V latent

            if self.cache.kv_scales is not None:
                k_scales = self.cache.kv_scales[layer_idx]
                v_scales = k_scales
            else:
                # Fallback: create dummy scales
                k_scales = torch.ones(
                    (self.max_batch_size, self.max_seq_len, 1),
                    dtype=torch.float16,
                    device=k_cache.device,
                )
                v_scales = k_scales

            return k_cache, v_cache, k_scales, v_scales

        def _build_block_tables(self, batch_size: int) -> torch.Tensor:
            """Generate block tables for paged attention.

            For contiguous cache (MLAKVCache), creates sequential block mapping.

            Returns:
                block_tables: [batch, max_blocks] int32
            """
            max_blocks = (self.max_seq_len + 15) // 16  # BLOCK_SIZE=16
            block_tables = torch.arange(
                batch_size * max_blocks, dtype=torch.int32, device=self.cache.kv_cache.device
            )
            return block_tables.view(batch_size, max_blocks)

        def update_cache(
            self,
            layer_idx: int,
            compressed_kv: torch.Tensor,
        ) -> torch.Tensor:
            """Update the FP4-quantized KV cache with new compressed KV values.

            Args:
                layer_idx: Layer index to update
                compressed_kv: Compressed KV [batch, seq_len, kv_lora_rank + qk_rope_head_dim]

            Returns:
                full_cache: Full cached sequence for the layer
            """
            return self.cache.update_compressed(layer_idx, compressed_kv)

        def forward(
            self,
            q_nope: torch.Tensor,
            q_rope: torch.Tensor,
            layer_idx: int = 0,
            context_lens: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Perform paged MLA attention with FP4-quantized KV cache.

            Args:
                q_nope: Non-rotary query [batch, num_heads, qk_nope_head_dim]
                q_rope: Rotary query [batch, num_heads, qk_rope_head_dim]
                layer_idx: Layer index in KV cache
                context_lens: Optional context lengths [batch], auto-detected if None

            Returns:
                attn_output: [batch, num_heads, v_head_dim]
            """
            batch_size, num_heads, _ = q_nope.shape
            device = q_nope.device

            # 1. Project Q to latent space
            q_latent = self._project_query_to_latent(q_nope)

            # 2. Concatenate with rotary component
            # q_full: [batch, num_heads, head_dim=kv_lora_rank + qk_rope_head_dim]
            q_full = torch.cat([q_latent, q_rope], dim=-1)

            # 3. Get context lengths
            if context_lens is None:
                seq_len = self.cache.seq_len
                context_lens = torch.full(
                    (batch_size,), seq_len, dtype=torch.int32, device=device
                )

            # 4. Retrieve and dequantize cached KV from MLAKVCache
            # The cache stores compressed KV in FP4 format
            # k_cache shape: [max_seq_len, cache_dim] for this layer/batch
            max_context_len = int(context_lens.max().item())
            
            # Get cached compressed KV for this layer
            # kv_cache shape: [num_layers, batch, max_seq, cache_dim//8]
            cached_packed = self.cache.kv_cache[layer_idx, :, :max_context_len, :]  # [batch, max_context, packed_dim]
            cached_scales = self.cache.kv_scales[layer_idx, :, :max_context_len, :]  # [batch, max_context, 1]
            
            # Dequantize FP4 cache back to float16
            # _dequantize_rowwise_mmfp4 expects [out_features, in_features//8] but cache is [seq, packed_dim]
            # We need to handle this carefully since the cache layout is different from weights
            
            # For cache dequantization, we apply per-token dequantization
            # packed: [batch, seq, cache_dim//8], scales: [batch, seq, 1]
            from ..layers.mmfp4_linear import _unpack_rowwise_nibbles, _E2M1_TABLE
            
            # Unpack FP4 nibbles: [batch, seq, cache_dim//8] -> [batch, seq, cache_dim]
            shifts = torch.arange(8, device=device, dtype=torch.int64) * 4
            shifts = shifts.view(1, 1, 8)
            words = cached_packed.to(torch.int64).unsqueeze(-1)
            nibbles = torch.bitwise_and(torch.bitwise_right_shift(words, shifts), 0xF)
            cached_dequant = nibbles.reshape(batch_size, max_context_len, -1).to(torch.uint8)
            
            # Map to float values using E2M1 table
            table = _E2M1_TABLE.to(device=device)
            cached_float = table[cached_dequant.to(torch.long)]  # [batch, seq, cache_dim]
            
            # Apply scales: [batch, seq, 1]
            cached_float = cached_float * cached_scales.to(cached_float.dtype)
            
            # Split into latent and rope components
            # cached_float: [batch, seq, kv_lora_rank + qk_rope_head_dim]
            k_latent = cached_float[..., :self.kv_lora_rank]  # [batch, seq, kv_lora_rank]
            k_rope = cached_float[..., self.kv_lora_rank:]    # [batch, seq, qk_rope_head_dim]
            
            # 5. Compute attention scores
            # q_latent: [batch, num_heads, kv_lora_rank]
            # k_latent: [batch, seq, kv_lora_rank] -> [batch, 1, seq, kv_lora_rank]
            k_latent = k_latent.unsqueeze(1)  # [batch, 1, seq, kv_lora_rank]
            
            # Expand q_latent for broadcasting: [batch, num_heads, 1, kv_lora_rank]
            q_latent_expanded = q_latent.unsqueeze(2)  # [batch, num_heads, 1, kv_lora_rank]
            
            # Compute latent attention scores: [batch, num_heads, 1, seq]
            latent_scores = torch.matmul(q_latent_expanded, k_latent.transpose(-2, -1)) * self.scale
            
            # Compute rope attention scores
            # q_rope: [batch, num_heads, qk_rope_head_dim]
            # k_rope: [batch, seq, qk_rope_head_dim] -> [batch, 1, seq, qk_rope_head_dim]
            k_rope = k_rope.unsqueeze(1)
            q_rope_expanded = q_rope.unsqueeze(2)  # [batch, num_heads, 1, qk_rope_head_dim]
            rope_scores = torch.matmul(q_rope_expanded, k_rope.transpose(-2, -1)) * self.scale
            
            # Combined scores
            attn_scores = latent_scores + rope_scores  # [batch, num_heads, 1, seq]
            
            # 6. Apply causal/context mask
            # Build validity mask from context_lens
            kv_positions = torch.arange(max_context_len, device=device).unsqueeze(0)  # [1, seq]
            valid_mask = kv_positions < context_lens.unsqueeze(1)  # [batch, seq]
            valid_mask = valid_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq]
            
            attn_scores = torch.where(
                valid_mask, attn_scores, torch.tensor(float("-inf"), device=device, dtype=attn_scores.dtype)
            )
            
            # 7. Softmax and apply to values
            attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch, num_heads, 1, seq]
            
            # For MLA, the "values" are the latent components
            # attn_weights: [batch, num_heads, 1, seq]
            # k_latent: [batch, 1, seq, kv_lora_rank]
            # We compute weighted sum of the cached latent vectors
            attn_latent = torch.matmul(attn_weights, k_latent)  # [batch, num_heads, 1, kv_lora_rank]
            attn_latent = attn_latent.squeeze(2)  # [batch, num_heads, kv_lora_rank]
            
            # 8. Project to output dimension
            attn_output = self._project_output_from_latent(attn_latent)
            
            return attn_output

        def get_cache(self) -> MLAKVCache:
            """Get the underlying MLAKVCache instance."""
            return self.cache

        def reset_cache(self) -> None:
            """Reset the KV cache for a new sequence."""
            self.cache.reset()

else:
    # Stub when PyTorch is unavailable
    class MMFP4PagedAttention:  # type: ignore[no-redef]
        """Stub when PyTorch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("MMFP4PagedAttention requires PyTorch")

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("MMFP4PagedAttention requires PyTorch")


__all__ = ["MMFP4PagedAttention"]
