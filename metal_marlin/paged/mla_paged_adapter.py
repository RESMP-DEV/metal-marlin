"""
MLA Paged Attention Adapter for TrellisKVCache.

Connects TrellisKVCache (compressed latent storage) to paged_attention_v1
Metal kernels by mapping the contiguous cache into a paged format.

Key Dimensions (GLM-4.7-Flash):
- kv_lora_rank = 512
- qk_rope_head_dim = 64
- Combined latent: 576 floats per token per layer
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .._compat import HAS_MPS, HAS_PYOBJC_METAL, HAS_TORCH
from ..kv_cache import TrellisKVCache
from ..trellis.attention import TrellisMLAttention

if HAS_PYOBJC_METAL:
    import Metal

if TYPE_CHECKING:
    import torch

# Constants matching paged_attention.metal
BLOCK_SIZE = 8
THREADS_PER_TG = 128

_paged_kernel_lib: Any = None


def _get_paged_kernel_library() -> Any:
    """Get or create the paged attention kernel library."""
    global _paged_kernel_lib
    if _paged_kernel_lib is None:
        from ..metal_dispatch import MetalKernelLibrary

        _paged_kernel_lib = MetalKernelLibrary()
        # Find paged_attention.metal
        # Try relative to this file first
        base_path = Path(__file__).parent.parent.parent
        kernel_path = base_path / "src" / "paged_attention.metal"

        if not kernel_path.exists():
            # Fallback to current working directory based search
            kernel_path = Path("contrib/metal_marlin/src/paged_attention.metal")

        if kernel_path.exists():
            source = kernel_path.read_text()
            # MLA requires a larger HEAD_DIM_MAX than the default 128.
            # We patch it to 576 for GLM-4.7-Flash support.
            if "constant constexpr uint HEAD_DIM_MAX = 128;" in source:
                source = source.replace(
                    "constant constexpr uint HEAD_DIM_MAX = 128;",
                    "constant constexpr uint HEAD_DIM_MAX = 576;",
                )
            # Patch BLOCK_SIZE and KV_TILES to fit in threadgroup memory with large HEAD_DIM
            if "constant constexpr uint BLOCK_SIZE = 16;" in source:
                source = source.replace(
                    "constant constexpr uint BLOCK_SIZE = 16;",
                    "constant constexpr uint BLOCK_SIZE = 8;",
                )
            if "constant constexpr uint KV_TILES = 2;" in source:
                source = source.replace(
                    "constant constexpr uint KV_TILES = 2;",
                    "constant constexpr uint KV_TILES = 1;",
                )
            _paged_kernel_lib.compile_source("paged_attention", source)
        else:
            raise FileNotFoundError(f"Paged attention kernel not found at {kernel_path}")
    return _paged_kernel_lib


def _mps_tensor_to_metal_buffer(tensor: torch.Tensor, device: Any) -> Any:
    """Convert MPS tensor to Metal buffer."""
    if not tensor.is_mps:
        tensor = tensor.to("mps")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # Use the bridge from metal_dispatch if available
    from ..metal_dispatch import mps_tensor_to_metal_buffer

    return mps_tensor_to_metal_buffer(tensor, device)


class MLAPagedAdapter:
    """Adapter to connect TrellisKVCache to paged attention kernels."""

    def __init__(
        self,
        kv_cache: TrellisKVCache,
        attention_layer: TrellisMLAttention,
    ):
        """Initialize the adapter.

        Args:
            kv_cache: TrellisKVCache instance (contiguous or paged backend).
            attention_layer: TrellisMLAttention instance providing projections.
        """
        self.kv_cache = kv_cache
        self.attn = attention_layer
        self.config = attention_layer.config
        self.device = kv_cache.kv_cache.device

        # Cache dimensions
        self.kv_lora_rank = attention_layer.kv_lora_rank
        self.qk_rope_head_dim = attention_layer.qk_rope_head_dim
        self.head_dim = self.kv_lora_rank + self.qk_rope_head_dim  # 576

    def _get_block_tables(self, batch_size: int) -> torch.Tensor:
        """Generate virtual block tables for contiguous TrellisKVCache."""
        max_seq = self.kv_cache.max_seq_len
        num_blocks_per_seq = (max_seq + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Sequential mapping for contiguous cache
        block_tables = torch.arange(
            batch_size * num_blocks_per_seq, device=self.device, dtype=torch.int32
        )
        return block_tables.view(batch_size, num_blocks_per_seq)

    def _project_query(self, q_nope: torch.Tensor) -> torch.Tensor:
        """Project Q_nope to latent space: Q_latent = Q_nope @ W_kv_b_nope.

        Args:
            q_nope: [batch, num_heads, qk_nope_head_dim]

        Returns:
            q_latent: [batch, num_heads, kv_lora_rank]
        """
        num_heads = self.attn.num_heads
        qk_nope_dim = self.attn.qk_nope_head_dim
        v_dim = self.attn.v_head_dim
        kv_lora_rank = self.attn.kv_lora_rank

        # kv_b_proj weight is [num_kv_heads * (qk_nope + v), kv_lora_rank]
        # For GLM-4, num_kv_heads == num_attention_heads
        weight = self.attn.kv_b_proj.weight

        # Reshape weight to [num_heads, qk_nope + v, kv_lora_rank]
        w_reshaped = weight.view(num_heads, qk_nope_dim + v_dim, kv_lora_rank)
        w_kv_b_nope = w_reshaped[:, :qk_nope_dim, :]  # [num_heads, qk_nope, kv_lora_rank]

        # Use batch matmul: [B, H, 1, nope] @ [1, H, nope, rank] -> [B, H, 1, rank]
        q_latent = torch.matmul(q_nope.unsqueeze(2), w_kv_b_nope.unsqueeze(0))
        return q_latent.squeeze(2)

    def attention(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Perform paged MLA attention.

        Args:
            q_nope: Non-rotary query [batch, num_heads, qk_nope_head_dim]
            q_rope: Rotary query [batch, num_heads, qk_rope_head_dim]
            layer_idx: Layer index in KV cache

        Returns:
            attn_output: [batch, num_heads, v_head_dim]
        """
        batch_size, num_heads, _ = q_nope.shape

        # 1. Project Q to latent space
        q_latent = self._project_query(q_nope)
        # q_proj: [batch, num_heads, dim=576]
        q_proj = torch.cat([q_latent, q_rope], dim=-1)

        # 2. Prepare paged cache format
        # If kv_cache is already paged (e.g. CompressedKVCacheMLA), use its tables.
        # Otherwise, wrap the contiguous TrellisKVCache.
        if hasattr(self.kv_cache, "page_table") and self.kv_cache.page_table is not None:
            block_tables = self.kv_cache.page_table.to(torch.int32)
        else:
            block_tables = self._get_block_tables(batch_size)

        # context_lens: [batch]
        if hasattr(self.kv_cache, "seq_lens") and isinstance(self.kv_cache.seq_lens, torch.Tensor):
            if self.kv_cache.seq_lens.dim() == 2:
                context_lens = self.kv_cache.seq_lens[layer_idx].to(torch.int32)
            else:
                context_lens = self.kv_cache.seq_lens.to(torch.int32)
        else:
            context_lens = torch.full(
                (batch_size,), self.kv_cache.seq_len, device=self.device, dtype=torch.int32
            )

        # 3. Cache buffers
        # For MLA, we use the compressed latent as both K and V.
        raw_cache = self.kv_cache.kv_cache[layer_idx]  # [batch, max_seq, dim]
        max_seq = raw_cache.shape[1]
        num_blocks_per_seq = (max_seq + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Reshape to [num_blocks, num_kv_heads=1, block_size, head_dim]
        paged_cache = raw_cache.view(-1, 1, BLOCK_SIZE, self.head_dim)

        # 4. Dispatch kernel
        quant_mode = getattr(self.kv_cache, "quantize_mode", "none")
        scale = self.attn.scale

        if quant_mode == "none":
            latent_out = self._dispatch_v1(
                q_proj, paged_cache, paged_cache, block_tables, context_lens, scale
            )
        elif quant_mode == "fp8":
            # FP8 requires scales
            scales = self.kv_cache.kv_scales[layer_idx].view(-1, 1, BLOCK_SIZE)
            latent_out = self._dispatch_v1_fp8(
                q_proj, paged_cache, paged_cache, scales, scales, block_tables, context_lens, scale
            )
        elif quant_mode == "fp4":
            # FP4 requires scales and packed cache (already packed in CompressedKVCacheMLA)
            scales = self.kv_cache.kv_scales[layer_idx].view(-1, 1, BLOCK_SIZE)
            # Reinterpret paged_cache as uint32 if it's stored as uint8
            if paged_cache.dtype == torch.uint8:
                # Kernel expects uint32 for packed fp4
                paged_cache = paged_cache.view(torch.int32)
            latent_out = self._dispatch_v1_fp4(
                q_proj, paged_cache, paged_cache, scales, scales, block_tables, context_lens, scale
            )
        elif quant_mode == "int4":
            # INT4 requires scales and packed cache
            scales = self.kv_cache.kv_scales[layer_idx].view(-1, 1, BLOCK_SIZE)
            if paged_cache.dtype == torch.uint8:
                paged_cache = paged_cache.view(torch.int32)
            latent_out = self._dispatch_v1_int4(
                q_proj, paged_cache, paged_cache, scales, scales, block_tables, context_lens, scale
            )
        else:
            raise NotImplementedError(f"Quantization mode {quant_mode} not supported in adapter")

        # 5. Post-project latent attention output to V space
        # latent_out: [batch, num_heads, dim=576]
        attn_latent = latent_out[..., : self.kv_lora_rank]

        # Reshape kv_b_proj weight to get V projection part
        weight = self.attn.kv_b_proj.weight.view(
            num_heads, self.attn.qk_nope_head_dim + self.attn.v_head_dim, self.kv_lora_rank
        )
        w_kv_b_v = weight[:, self.attn.qk_nope_head_dim :, :]  # [num_heads, v_dim, rank]

        # attn_out = attn_latent @ w_kv_b_v.T
        # [B, H, 1, rank] @ [1, H, rank, v_dim] -> [B, H, 1, v_dim]
        attn_out = torch.matmul(attn_latent.unsqueeze(2), w_kv_b_v.permute(0, 2, 1).unsqueeze(0))
        return attn_out.squeeze(2)

    def _dispatch_v1(self, Q, K, V, block_tables, context_lens, scale):
        lib = _get_paged_kernel_library()
        device = lib.device

        num_seqs, num_heads_q, head_dim = Q.shape
        num_kv_heads = K.shape[1]
        max_blocks_per_seq = block_tables.shape[1]

        # Allocate output
        output = torch.empty_like(Q)

        # Bind buffers
        buffers = [
            _mps_tensor_to_metal_buffer(Q, device),
            _mps_tensor_to_metal_buffer(K, device),
            _mps_tensor_to_metal_buffer(V, device),
            _mps_tensor_to_metal_buffer(block_tables, device),
            _mps_tensor_to_metal_buffer(context_lens, device),
            _mps_tensor_to_metal_buffer(output, device),
        ]

        # Scalar arguments
        def make_u32(v):
            return device.newBufferWithBytes_length_options_(
                np.array([v], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
            )

        buffers.extend(
            [
                make_u32(num_seqs),
                make_u32(num_heads_q),
                make_u32(num_kv_heads),
                make_u32(head_dim),
                make_u32(max_blocks_per_seq),
                device.newBufferWithBytes_length_options_(
                    np.array([scale], dtype=np.float32).tobytes(),
                    4,
                    Metal.MTLResourceStorageModeShared,
                ),
            ]
        )

        from ..metal_dispatch import dispatch_kernel

        dispatch_kernel(
            lib, "paged_attention_v1", (num_seqs, num_heads_q, 1), (THREADS_PER_TG, 1, 1), buffers
        )
        return output

    def _dispatch_v1_fp8(self, Q, K, V, Ks, Vs, block_tables, context_lens, scale):
        lib = _get_paged_kernel_library()
        device = lib.device

        num_seqs, num_heads_q, head_dim = Q.shape
        num_kv_heads = K.shape[1]
        max_blocks_per_seq = block_tables.shape[1]

        output = torch.empty_like(Q)

        buffers = [
            _mps_tensor_to_metal_buffer(Q, device),
            _mps_tensor_to_metal_buffer(K, device),
            _mps_tensor_to_metal_buffer(V, device),
            _mps_tensor_to_metal_buffer(Ks, device),
            _mps_tensor_to_metal_buffer(Vs, device),
            _mps_tensor_to_metal_buffer(block_tables, device),
            _mps_tensor_to_metal_buffer(context_lens, device),
            _mps_tensor_to_metal_buffer(output, device),
        ]

        def make_u32(v):
            return device.newBufferWithBytes_length_options_(
                np.array([v], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
            )

        buffers.extend(
            [
                make_u32(num_seqs),
                make_u32(num_heads_q),
                make_u32(num_kv_heads),
                make_u32(head_dim),
                make_u32(max_blocks_per_seq),
                device.newBufferWithBytes_length_options_(
                    np.array([scale], dtype=np.float32).tobytes(),
                    4,
                    Metal.MTLResourceStorageModeShared,
                ),
            ]
        )

        from ..metal_dispatch import dispatch_kernel

        dispatch_kernel(
            lib, "paged_attention_v1_fp8", (num_seqs, num_heads_q, 1), (THREADS_PER_TG, 1, 1), buffers
        )
        return output

    def _dispatch_v1_fp4(self, Q, K, V, Ks, Vs, block_tables, context_lens, scale):
        lib = _get_paged_kernel_library()
        device = lib.device

        num_seqs, num_heads_q, head_dim = Q.shape
        num_kv_heads = K.shape[1]
        max_blocks_per_seq = block_tables.shape[1]

        output = torch.empty_like(Q)

        buffers = [
            _mps_tensor_to_metal_buffer(Q, device),
            _mps_tensor_to_metal_buffer(K, device),
            _mps_tensor_to_metal_buffer(V, device),
            _mps_tensor_to_metal_buffer(Ks, device),
            _mps_tensor_to_metal_buffer(Vs, device),
            _mps_tensor_to_metal_buffer(block_tables, device),
            _mps_tensor_to_metal_buffer(context_lens, device),
            _mps_tensor_to_metal_buffer(output, device),
        ]

        def make_u32(v):
            return device.newBufferWithBytes_length_options_(
                np.array([v], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
            )

        buffers.extend(
            [
                make_u32(num_seqs),
                make_u32(num_heads_q),
                make_u32(num_kv_heads),
                make_u32(head_dim),
                make_u32(max_blocks_per_seq),
                device.newBufferWithBytes_length_options_(
                    np.array([scale], dtype=np.float32).tobytes(),
                    4,
                    Metal.MTLResourceStorageModeShared,
                ),
            ]
        )

        from ..metal_dispatch import dispatch_kernel

        dispatch_kernel(
            lib, "paged_attention_v1_fp4", (num_seqs, num_heads_q, 1), (THREADS_PER_TG, 1, 1), buffers
        )
        return output

    def _dispatch_v1_int4(self, Q, K, V, Ks, Vs, block_tables, context_lens, scale):
        lib = _get_paged_kernel_library()
        device = lib.device

        num_seqs, num_heads_q, head_dim = Q.shape
        num_kv_heads = K.shape[1]
        max_blocks_per_seq = block_tables.shape[1]

        output = torch.empty_like(Q)

        buffers = [
            _mps_tensor_to_metal_buffer(Q, device),
            _mps_tensor_to_metal_buffer(K, device),
            _mps_tensor_to_metal_buffer(V, device),
            _mps_tensor_to_metal_buffer(Ks, device),
            _mps_tensor_to_metal_buffer(Vs, device),
            _mps_tensor_to_metal_buffer(block_tables, device),
            _mps_tensor_to_metal_buffer(context_lens, device),
            _mps_tensor_to_metal_buffer(output, device),
        ]

        def make_u32(v):
            return device.newBufferWithBytes_length_options_(
                np.array([v], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
            )

        buffers.extend(
            [
                make_u32(num_seqs),
                make_u32(num_heads_q),
                make_u32(num_kv_heads),
                make_u32(head_dim),
                make_u32(max_blocks_per_seq),
                device.newBufferWithBytes_length_options_(
                    np.array([scale], dtype=np.float32).tobytes(),
                    4,
                    Metal.MTLResourceStorageModeShared,
                ),
            ]
        )

        from ..metal_dispatch import dispatch_kernel

        dispatch_kernel(
            lib, "paged_attention_v1_int4", (num_seqs, num_heads_q, 1), (THREADS_PER_TG, 1, 1), buffers
        )
        return output
