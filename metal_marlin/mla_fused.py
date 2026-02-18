"""
Fused MLA (Multi-head Latent Attention) kernel wrappers.

Provides optimized fused kernels for MLA-style attention that combine:
- Q projection (hidden -> q_latent -> q_heads)
- KV projection (hidden -> kv_latent -> k, v)
- RoPE (rotary position embedding)
- Attention computation
- Output projection

Fused MLA reduces memory traffic by combining multiple operations into single kernel dispatch.

Architecture benefits:
- Reduces kernel launch overhead (5+ launches -> 1 launch)
- Minimizes global memory reads/writes
- Better cache locality for intermediate activations
- Optimized for GLM-4.7-Flash dimensions (q_lora_rank=768, kv_lora_rank=512, rope_dim=64)

Usage:
    from mla_fused import mla_fused_attention_decode, mla_fused_attention_prefill
    
    # Decode phase (single token, streaming KV cache)
    output = mla_fused_attention_decode(
        hidden,                     # [batch, 1, hidden_size]
        q_weights, kv_weights, o_weights,  # Quantized weights
        k_cache, v_cache,          # Compressed KV cache
        params                     # MLAAttentionParams
    )
    
    # Prefill phase (multiple tokens, full context)
    output = mla_fused_attention_prefill(
        hidden, q_weights, kv_weights, o_weights,
        k_cache, v_cache,
        params
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from metal_marlin._compat import HAS_MPS, HAS_PYOBJC_METAL, HAS_TORCH, torch

# Metal kernel dispatch requires both PyTorch MPS and PyObjC Metal framework
HAS_METAL_DISPATCH: bool = HAS_MPS and HAS_PYOBJC_METAL

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Constants matching attention_mla_fused.metal
# ---------------------------------------------------------------------------

TILE_M_MLA = 256
TILE_N_MLA = 64
TILE_K_MLA_SMALL = 16
TILE_K_MLA_LARGE = 32
TILE_Q = 16
TILE_KV = 24
HEAD_DIM_64 = 64
HEAD_DIM_128 = 128
SIMD_SIZE = 32
NUM_SIMDGROUPS = 4
THREADS_PER_TG = SIMD_SIZE * NUM_SIMDGROUPS
THREADS_PER_TG_DECODE = 64
FP4_PER_UINT = 8


# ---------------------------------------------------------------------------
# MLA Attention Parameters
# ---------------------------------------------------------------------------

@dataclass
class MLAAttentionParams:
    """Parameters for MLA fused attention kernel.
    
    Matches the struct in attention_mla_fused.metal.
    """
    batch: int
    seq_q: int
    seq_k: int
    hidden_size: int
    num_heads: int
    head_dim: int
    kv_lora_rank: int
    q_lora_rank: int
    rope_dim: int
    scale: float
    is_causal: bool
    q_a_group_size: int
    q_b_group_size: int
    kv_a_group_size: int
    kv_b_group_size: int
    o_group_size: int
    rope_theta: float
    rope_ratio: float
    rope_base_seq_len: int
    cache_start_pos: int
    cache_len: int
    max_cache_len: int
    use_fused_q_proj: bool
    use_fused_kv_proj: bool
    fuse_rope_in_kv_a: bool
    skip_kv_decompress: bool
    # KV cache quantization parameters for attention optimization
    kv_quant_mode: str = "none"  # "none", "fp4", "fp8", "int8"
    kv_quant_group_size: int = 128
    # Sliding window attention for memory-efficient long sequence processing
    sliding_window: int = 0  # 0 = disabled, >0 = window size
    
    def to_struct(self) -> np.ndarray:
        """Convert to numpy array for Metal buffer binding."""
        # Map kv_quant_mode to integer for Metal
        kv_quant_mode_map = {"none": 0, "fp4": 1, "fp8": 2, "int8": 3}
        kv_quant_mode_int = kv_quant_mode_map.get(self.kv_quant_mode, 0)
        
        return np.array([
            self.batch, self.seq_q, self.seq_k, self.hidden_size,
            self.num_heads, self.head_dim, self.kv_lora_rank, self.q_lora_rank,
            self.rope_dim,
            *[0.0] * 8,  # Padding for alignment
            self.scale,
            int(self.is_causal),
            self.q_a_group_size, self.q_b_group_size,
            self.kv_a_group_size, self.kv_b_group_size, self.o_group_size,
            self.rope_theta, self.rope_ratio,
            self.rope_base_seq_len, self.cache_start_pos,
            self.cache_len, self.max_cache_len,
            int(self.use_fused_q_proj), int(self.use_fused_kv_proj),
            int(self.fuse_rope_in_kv_a), int(self.skip_kv_decompress),
            kv_quant_mode_int, self.kv_quant_group_size,  # KV quantization params
            self.sliding_window,  # Sliding window attention (0 = disabled)
        ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Metal dispatch infrastructure (lazy imports to avoid circular deps)
# ---------------------------------------------------------------------------

_metal_lib: Any = None


def _get_metal_library() -> Any:
    """Get or create the Metal kernel library."""
    global _metal_lib
    if _metal_lib is None:
        from metal_marlin.metal_dispatch import get_default_library
        _metal_lib = get_default_library()
    return _metal_lib


def _require_mps() -> None:
    """Raise if MPS is not available."""
    if not HAS_MPS:
        raise RuntimeError(
            "MLA fused kernel dispatch requires PyTorch with MPS backend.\n"
            "Ensure you're on Apple Silicon with PyTorch >= 2.0"
        )


# ---------------------------------------------------------------------------
# Numpy fallback implementations
# ---------------------------------------------------------------------------

def _mla_fused_decode_numpy(
    hidden: np.ndarray,
    q_a_weights: np.ndarray,
    kv_a_weights: np.ndarray,
    kv_b_weights: np.ndarray,
    o_weights: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    params: MLAAttentionParams,
) -> np.ndarray:
    """CPU fallback: MLA fused decode via numpy.
    
    This implements the same logic as the Metal kernel but on CPU.
    Used for testing and when Metal is unavailable.
    """
    batch, seq_q, hidden_size = hidden.shape
    
    # Step 1: Compute Q via two-stage projection
    q_latent = hidden @ q_a_weights.T  # [B, S, q_lora_rank]
    q = q_latent.reshape(batch * seq_q, params.q_lora_rank)
    q = q @ q_weights.T  # [B*S, num_heads * head_dim]
    q = q.reshape(batch, seq_q, params.num_heads, params.head_dim)
    
    # Step 2: Compute KV via two-stage projection
    kv_latent = hidden @ kv_a_weights.T  # [B, S, kv_lora_rank + rope_dim]
    c_kv = kv_latent[..., :params.kv_lora_rank]
    k_pe = kv_latent[..., params.kv_lora_rank:]
    
    # Decompress KV
    kv_full = c_kv @ kv_b_weights.T  # [B, S, num_heads * 2 * head_dim]
    kv_full = kv_full.reshape(batch, seq_q, params.num_heads, 2, params.head_dim)
    k_nope = kv_full[..., 0, :]
    v = kv_full[..., 1, :]
    
    # Step 3: Apply RoPE to k_pe and merge with k_nope
    if params.rope_dim > 0:
        k_pe_rope = _apply_rope_numpy(k_pe, params)
        k = np.concatenate([k_nope, k_pe_rope], axis=-1)
    else:
        k = k_nope
    
    # Step 4: Attention computation
    q = q.transpose(0, 2, 1, 3)  # [B, H, S, D]
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    
    attn = torch.from_numpy(q) @ torch.from_numpy(k).transpose(-2, -1)  # [B, H, S, S]
    attn = attn * params.scale
    
    if params.is_causal:
        mask = torch.triu(torch.ones_like(attn), diagonal=1)
        attn = attn.masked_fill(mask.bool(), float('-inf'))
    
    attn = torch.softmax(attn, dim=-1)
    o = attn @ torch.from_numpy(v)  # [B, H, S, D]
    o = o.transpose(0, 2, 1, 3)  # [B, S, H, D]
    o = o.reshape(batch, seq_q, params.num_heads * params.head_dim)
    
    # Step 5: Output projection
    output = o @ o_weights.T  # [B, S, hidden_size]
    
    return output.astype(np.float16)


def _apply_rope_numpy(
    x: np.ndarray,
    params: MLAAttentionParams,
) -> np.ndarray:
    """Apply rotary position embedding (CPU fallback)."""
    # x: [batch, seq, rope_dim]
    batch, seq, rope_dim = x.shape
    
    # Compute frequencies
    freqs = np.exp(
        -np.log(params.rope_theta) * np.arange(0, rope_dim, 2, dtype=np.float32) / rope_dim
    )
    freqs = freqs * params.rope_ratio
    
    # Apply RoPE
    result = x.copy()
    for s in range(seq):
        pos = params.cache_start_pos + s
        angles = pos * freqs
        cos = np.cos(angles)
        sin = np.sin(angles)
        
        for i in range(0, rope_dim, 2):
            x_i = x[:, s, i]
            x_j = x[:, s, i + 1]
            result[:, s, i] = x_i * cos[i // 2] - x_j * sin[i // 2]
            result[:, s, i + 1] = x_i * sin[i // 2] + x_j * cos[i // 2]
    
    return result


# ---------------------------------------------------------------------------
# Metal kernel dispatch functions
# ---------------------------------------------------------------------------

def mla_fused_attention_decode(
    hidden: torch.Tensor,
    q_a_weights_packed: torch.Tensor,
    q_a_scales: torch.Tensor,
    q_b_weights_packed: torch.Tensor,
    q_b_scales: torch.Tensor,
    q_bias: torch.Tensor | None,
    kv_a_weights_packed: torch.Tensor,
    kv_a_scales: torch.Tensor,
    kv_b_weights_packed: torch.Tensor,
    kv_b_scales: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    o_weights_packed: torch.Tensor,
    o_scales: torch.Tensor,
    params: MLAAttentionParams,
) -> torch.Tensor:
    """Dispatch MLA fused attention decode kernel via Metal.
    
    Optimized for single-token inference (decode phase) with streaming KV cache.
    
    Args:
        hidden: Input hidden states [batch, seq_q, hidden_size]
        q_a_weights_packed: Q projection A weights (FP4 packed) [hidden_size/8, q_lora_rank]
        q_a_scales: Q projection A scales [hidden_size/group_size, q_lora_rank]
        q_b_weights_packed: Q projection B weights (FP4 packed) [q_lora_rank/8, num_heads*head_dim]
        q_b_scales: Q projection B scales [q_lora_rank/group_size, num_heads*head_dim]
        q_bias: Q projection bias [num_heads*head_dim] or None
        kv_a_weights_packed: KV projection A weights (FP4 packed)
        kv_a_scales: KV projection A scales
        kv_b_weights_packed: KV projection B weights (FP4 packed)
        kv_b_scales: KV projection B scales
        k_cache: Compressed K cache [cache_len, compressed_dim]
        v_cache: Compressed V cache [cache_len, compressed_dim]
        k_scales: K cache scales [cache_len, compressed_dim/group_size]
        v_scales: V cache scales [cache_len, compressed_dim/group_size]
        o_weights_packed: Output projection weights (FP4 packed)
        o_scales: Output projection scales
        params: MLAAttentionParams configuration
        
    Returns:
        output: Attention output [batch, seq_q, hidden_size]
    """
    _require_mps()
    
    import Metal
    from metal_marlin.metal_dispatch import dispatch_kernel
    
    lib = _get_metal_library()
    
    # Get kernel function
    kernel_name = "mla_fused_attention_decode"
    kernel = lib.get_function(kernel_name)
    
    # Prepare parameters buffer
    params_np = params.to_struct()
    params_buffer = lib.new_buffer(bytes(params_np.tobytes()))
    
    # Compute grid dimensions
    # Grid: (num_heads, seq_q, batch)
    grid_x = (params.num_heads + 31) // 32
    grid_y = (params.seq_q + 15) // 16
    grid_z = params.batch
    
    grid = (grid_x, grid_y, grid_z)
    threadgroup = (NUM_SIMDGROUPS * SIMD_SIZE, 1, 1)
    
    # Get Metal buffers
    hidden_buf = hidden._mps._get_metal_buffer_ptr()
    q_a_buf = q_a_weights_packed._mps._get_metal_buffer_ptr()
    q_a_scales_buf = q_a_scales._mps._get_metal_buffer_ptr()
    q_b_buf = q_b_weights_packed._mps._get_metal_buffer_ptr()
    q_b_scales_buf = q_b_scales._mps._get_metal_buffer_ptr()
    kv_a_buf = kv_a_weights_packed._mps._get_metal_buffer_ptr()
    kv_a_scales_buf = kv_a_scales._mps._get_metal_buffer_ptr()
    kv_b_buf = kv_b_weights_packed._mps._get_metal_buffer_ptr()
    kv_b_scales_buf = kv_b_scales._mps._get_metal_buffer_ptr()
    k_cache_buf = k_cache._mps._get_metal_buffer_ptr()
    v_cache_buf = v_cache._mps._get_metal_buffer_ptr()
    k_scales_buf = k_scales._mps._get_metal_buffer_ptr()
    v_scales_buf = v_scales._mps._get_metal_buffer_ptr()
    o_buf = o_weights_packed._mps._get_metal_buffer_ptr()
    o_scales_buf = o_scales._mps._get_metal_buffer_ptr()
    
    # Prepare output tensor
    output = torch.empty_like(hidden)
    output_buf = output._mps._get_metal_buffer_ptr()
    
    # Prepare buffers for bias (may be None)
    q_bias_buf = 0 if q_bias is None else q_bias._mps._get_metal_buffer_ptr()
    
    # Dispatch kernel
    buffers = [
        hidden_buf, q_a_buf, q_a_scales_buf, q_b_buf, q_b_scales_buf, q_bias_buf,
        kv_a_buf, kv_a_scales_buf, kv_b_buf, kv_b_scales_buf,
        k_cache_buf, v_cache_buf, k_scales_buf, v_scales_buf,
        o_buf, o_scales_buf, output_buf, params_buffer,
    ]
    
    dispatch_kernel(lib, kernel_name, grid, threadgroup, buffers)
    
    return output


def mla_fused_attention_prefill(
    hidden: torch.Tensor,
    q_a_weights_packed: torch.Tensor,
    q_a_scales: torch.Tensor,
    q_b_weights_packed: torch.Tensor,
    q_b_scales: torch.Tensor,
    q_bias: torch.Tensor | None,
    kv_a_weights_packed: torch.Tensor,
    kv_a_scales: torch.Tensor,
    kv_b_weights_packed: torch.Tensor,
    kv_b_scales: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    o_weights_packed: torch.Tensor,
    o_scales: torch.Tensor,
    params: MLAAttentionParams,
    k_scales: torch.Tensor | None = None,
    v_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch MLA fused attention prefill kernel via Metal.
    
    Optimized for prefill phase with multiple tokens and full context.
    Supports quantized KV caches (FP4/FP8/INT8) for reduced memory bandwidth.
    
    Args:
        Same as mla_fused_attention_decode, except cache formats differ
        (full FP16 for prefill vs compressed FP8 for decode).
        k_scales: Optional K cache scales for quantization [cache_len, n_groups]
        v_scales: Optional V cache scales for quantization [cache_len, n_groups]
        
    Returns:
        output: Attention output [batch, seq_q, hidden_size]
    """
    _require_mps()
    
    import Metal
    from metal_marlin.metal_dispatch import dispatch_kernel
    
    lib = _get_metal_library()
    
    # Get kernel function
    kernel_name = "mla_fused_attention_prefill"
    kernel = lib.get_function(kernel_name)
    
    # Prepare parameters buffer
    params_np = params.to_struct()
    params_buffer = lib.new_buffer(bytes(params_np.tobytes()))
    
    # Compute grid dimensions
    grid_x = (params.num_heads + 31) // 32
    grid_y = (params.seq_q + TILE_Q - 1) // TILE_Q
    grid_z = params.batch
    
    grid = (grid_x, grid_y, grid_z)
    threadgroup = (NUM_SIMDGROUPS * SIMD_SIZE, 1, 1)
    
    # Get Metal buffers
    hidden_buf = hidden._mps._get_metal_buffer_ptr()
    q_a_buf = q_a_weights_packed._mps._get_metal_buffer_ptr()
    q_a_scales_buf = q_a_scales._mps._get_metal_buffer_ptr()
    q_b_buf = q_b_weights_packed._mps._get_metal_buffer_ptr()
    q_b_scales_buf = q_b_scales._mps._get_metal_buffer_ptr()
    kv_a_buf = kv_a_weights_packed._mps._get_metal_buffer_ptr()
    kv_a_scales_buf = kv_a_scales._mps._get_metal_buffer_ptr()
    kv_b_buf = kv_b_weights_packed._mps._get_metal_buffer_ptr()
    kv_b_scales_buf = kv_b_scales._mps._get_metal_buffer_ptr()
    k_cache_buf = k_cache._mps._get_metal_buffer_ptr()
    v_cache_buf = v_cache._mps._get_metal_buffer_ptr()
    
    # Handle optional scale buffers for quantized KV cache
    if k_scales is not None and v_scales is not None:
        k_scales_buf = k_scales._mps._get_metal_buffer_ptr()
        v_scales_buf = v_scales._mps._get_metal_buffer_ptr()
    else:
        # Create dummy scale buffers if not provided (for non-quantized case)
        k_scales_buf = 0
        v_scales_buf = 0
    
    o_buf = o_weights_packed._mps._get_metal_buffer_ptr()
    o_scales_buf = o_scales._mps._get_metal_buffer_ptr()
    
    # Prepare output tensor
    output = torch.empty_like(hidden)
    output_buf = output._mps._get_metal_buffer_ptr()
    
    # Prepare buffers for bias
    q_bias_buf = 0 if q_bias is None else q_bias._mps._get_metal_buffer_ptr()
    
    # Dispatch kernel
    buffers = [
        hidden_buf, q_a_buf, q_a_scales_buf, q_b_buf, q_b_scales_buf, q_bias_buf,
        kv_a_buf, kv_a_scales_buf, kv_b_buf, kv_b_scales_buf,
        k_cache_buf, v_cache_buf, k_scales_buf, v_scales_buf,
        o_buf, o_scales_buf,
        output_buf, params_buffer,
    ]
    
    dispatch_kernel(lib, kernel_name, grid, threadgroup, buffers)
    
    return output


# ---------------------------------------------------------------------------
# Convenience functions for GLM-4.7-Flash style MLA
# ---------------------------------------------------------------------------

def mla_chunked_prefill_attention(
    hidden: torch.Tensor,
    q_a_weights_packed: torch.Tensor,
    q_a_scales: torch.Tensor,
    q_b_weights_packed: torch.Tensor,
    q_b_scales: torch.Tensor,
    q_bias: torch.Tensor | None,
    kv_a_weights_packed: torch.Tensor,
    kv_a_scales: torch.Tensor,
    kv_b_weights_packed: torch.Tensor,
    kv_b_scales: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    o_weights_packed: torch.Tensor,
    o_scales: torch.Tensor,
    params: MLAAttentionParams,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Chunked prefill attention that processes long sequences in chunks.
    
    This reduces peak memory usage during prefill by processing the sequence
    in smaller chunks while maintaining the same final output. Each chunk
    is processed with full attention to all previous tokens (cached + current chunk).
    
    Args:
        hidden: Input hidden states [batch, seq_q, hidden_size]
        q_a_weights_packed: Q projection A weights (FP4 packed)
        q_a_scales: Q projection A scales
        q_b_weights_packed: Q projection B weights (FP4 packed)
        q_b_scales: Q projection B scales
        q_bias: Q projection bias or None
        kv_a_weights_packed: KV projection A weights (FP4 packed)
        kv_a_scales: KV projection A scales
        kv_b_weights_packed: KV projection B weights (FP4 packed)
        kv_b_scales: KV projection B scales
        k_cache: K cache buffer [cache_len + seq_q, kv_lora_rank]
        v_cache: V cache buffer [cache_len + seq_q, kv_lora_rank]
        o_weights_packed: Output projection weights (FP4 packed)
        o_scales: Output projection scales
        params: MLAAttentionParams configuration
        chunk_size: Size of each chunk for processing (default: 2048)
        
    Returns:
        output: Attention output [batch, seq_q, hidden_size]
    """
    _require_mps()
    
    batch_size, seq_len, hidden_size = hidden.shape
    
    # For small sequences, use regular prefill
    if seq_len <= chunk_size:
        return mla_fused_attention_prefill(
            hidden, q_a_weights_packed, q_a_scales,
            q_b_weights_packed, q_b_scales, q_bias,
            kv_a_weights_packed, kv_a_scales,
            kv_b_weights_packed, kv_b_scales,
            k_cache, v_cache, o_weights_packed, o_scales,
            params,
        )
    
    import Metal
    from metal_marlin.metal_dispatch import dispatch_kernel
    
    lib = _get_metal_library()
    
    # Get chunked prefill kernel
    kernel_name = "mla_chunked_prefill_attention"
    kernel = lib.get_function(kernel_name)
    
    # Process in chunks
    outputs = []
    cache_len = params.cache_len
    
    # Get write_kv kernel
    write_kv_kernel_name = "mla_write_kv_cache"
    write_kv_kernel = lib.get_function(write_kv_kernel_name)
    
    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        chunk_len = chunk_end - chunk_start
        
        # Extract chunk
        hidden_chunk = hidden[:, chunk_start:chunk_end, :].contiguous()
        
        # Update params for this chunk
        chunk_params = MLAAttentionParams(
            batch=params.batch,
            seq_q=chunk_len,
            seq_k=cache_len + chunk_len,  # Previous cache + current chunk
            hidden_size=params.hidden_size,
            num_heads=params.num_heads,
            head_dim=params.head_dim,
            kv_lora_rank=params.kv_lora_rank,
            q_lora_rank=params.q_lora_rank,
            rope_dim=params.rope_dim,
            scale=params.scale,
            is_causal=params.is_causal,
            q_a_group_size=params.q_a_group_size,
            q_b_group_size=params.q_b_group_size,
            kv_a_group_size=params.kv_a_group_size,
            kv_b_group_size=params.kv_b_group_size,
            o_group_size=params.o_group_size,
            rope_theta=params.rope_theta,
            rope_ratio=params.rope_ratio,
            rope_base_seq_len=params.rope_base_seq_len,
            cache_start_pos=cache_len,  # Start after existing cache
            cache_len=cache_len,
            max_cache_len=params.max_cache_len,
            use_fused_q_proj=params.use_fused_q_proj,
            use_fused_kv_proj=params.use_fused_kv_proj,
            fuse_rope_in_kv_a=params.fuse_rope_in_kv_a,
            skip_kv_decompress=params.skip_kv_decompress,
            kv_quant_mode=params.kv_quant_mode,
            kv_quant_group_size=params.kv_quant_group_size,
            sliding_window=params.sliding_window,
        )
        
        # Prepare parameters buffer
        params_np = chunk_params.to_struct()
        params_buffer = lib.new_buffer(bytes(params_np.tobytes()))
        
        # Get Metal buffers for chunk
        hidden_buf = hidden_chunk._mps._get_metal_buffer_ptr()
        
        # 1. Dispatch KV Cache Write Kernel
        # Grid: (ceil(chunk_len / TILE_Q), batch, 1) - One threadgroup per tile of tokens
        grid_kv_x = (chunk_len + 31) // 32 
        grid_kv = (grid_kv_x, chunk_params.batch, 1)
        threadgroup_kv = (128, 1, 1) # 4 simds
        
        buffers_kv = [
            hidden_buf, 
            kv_a_weights_packed._mps._get_metal_buffer_ptr(),
            kv_a_scales._mps._get_metal_buffer_ptr(),
            k_cache._mps._get_metal_buffer_ptr(),
            params_buffer
        ]
        
        dispatch_kernel(lib, write_kv_kernel_name, grid_kv, threadgroup_kv, buffers_kv)
        
        # 2. Dispatch Attention Kernel
        # Grid dimensions for this chunk
        # One threadgroup per HEAD per TILE of queries
        grid_x = chunk_params.num_heads
        grid_y = (chunk_len + TILE_Q - 1) // TILE_Q
        grid_z = chunk_params.batch
        
        grid = (grid_x, grid_y, grid_z)
        threadgroup = (NUM_SIMDGROUPS * SIMD_SIZE, 1, 1)
        
        # Get Metal buffers
        q_a_buf = q_a_weights_packed._mps._get_metal_buffer_ptr()
        q_a_scales_buf = q_a_scales._mps._get_metal_buffer_ptr()
        q_b_buf = q_b_weights_packed._mps._get_metal_buffer_ptr()
        q_b_scales_buf = q_b_scales._mps._get_metal_buffer_ptr()
        # kv_a/b not needed for attention part if cached, but might be used for on-the-fly?
        # We assume strict cache usage for chunked prefill to avoid re-computation
        # But we pass them to keep signature compatible if needed, or we clean up.
        # We'll pass them but kernel will ignore.
        kv_a_buf = kv_a_weights_packed._mps._get_metal_buffer_ptr()
        kv_a_scales_buf = kv_a_scales._mps._get_metal_buffer_ptr()
        kv_b_buf = kv_b_weights_packed._mps._get_metal_buffer_ptr()
        kv_b_scales_buf = kv_b_scales._mps._get_metal_buffer_ptr()
        k_cache_buf = k_cache._mps._get_metal_buffer_ptr()
        v_cache_buf = v_cache._mps._get_metal_buffer_ptr()
        o_buf = o_weights_packed._mps._get_metal_buffer_ptr()
        o_scales_buf = o_scales._mps._get_metal_buffer_ptr()
        
        # Prepare output tensor for this chunk
        output_chunk = torch.empty_like(hidden_chunk)
        output_buf = output_chunk._mps._get_metal_buffer_ptr()
        
        # Prepare buffers for bias
        q_bias_buf = 0 if q_bias is None else q_bias._mps._get_metal_buffer_ptr()
        
        # Dispatch kernel
        buffers = [
            hidden_buf, q_a_buf, q_a_scales_buf, q_b_buf, q_b_scales_buf, q_bias_buf,
            kv_a_buf, kv_a_scales_buf, kv_b_buf, kv_b_scales_buf,
            k_cache_buf, v_cache_buf, o_buf, o_scales_buf,
            output_buf, params_buffer,
        ]
        
        dispatch_kernel(lib, kernel_name, grid, threadgroup, buffers)
        
        outputs.append(output_chunk)
        
        # Update cache length for next chunk
        cache_len += chunk_len
    
    # Concatenate all chunk outputs
    return torch.cat(outputs, dim=1)


def create_glm_mla_params(
    batch: int,
    seq_q: int,
    seq_k: int,
    hidden_size: int = 4096,
    num_heads: int = 32,
    head_dim: int = 128,
    kv_lora_rank: int = 512,
    q_lora_rank: int = 768,
    rope_dim: int = 64,
    cache_start_pos: int = 0,
    cache_len: int = 0,
    group_size: int = 64,
    kv_quant_mode: str = "none",
    kv_quant_group_size: int = 128,
    sliding_window: int = 0,
) -> MLAAttentionParams:
    """Create MLAAttentionParams for GLM-4.7-Flash.
    
    Args:
        batch: Batch size
        seq_q: Query sequence length
        seq_k: Key sequence length (including cache)
        hidden_size: Model hidden dimension (default: 4096)
        num_heads: Number of attention heads (default: 32)
        head_dim: Head dimension (default: 128)
        kv_lora_rank: KV compression rank (default: 512)
        q_lora_rank: Q compression rank (default: 768)
        rope_dim: RoPE dimension (default: 64)
        cache_start_pos: Starting position in cache (default: 0)
        cache_len: Current cache length (default: 0)
        group_size: Quantization group size (default: 64)
        kv_quant_mode: KV cache quantization mode (default: "none")
        kv_quant_group_size: KV quantization group size (default: 128)
        
    Returns:
        MLAAttentionParams configured for GLM-4.7-Flash
    """
    head_dim = head_dim or hidden_size // num_heads
    scale = head_dim ** -0.5
    
    return MLAAttentionParams(
        batch=batch,
        seq_q=seq_q,
        seq_k=seq_k,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=q_lora_rank,
        rope_dim=rope_dim,
        scale=scale,
        is_causal=True,
        q_a_group_size=group_size,
        q_b_group_size=group_size,
        kv_a_group_size=group_size,
        kv_b_group_size=group_size,
        o_group_size=group_size,
        rope_theta=10000.0,
        rope_ratio=1.0,
        rope_base_seq_len=0,
        cache_start_pos=cache_start_pos,
        cache_len=cache_len,
        max_cache_len=seq_k,
        use_fused_q_proj=True,
        use_fused_kv_proj=True,
        fuse_rope_in_kv_a=True,
        skip_kv_decompress=False,
        kv_quant_mode=kv_quant_mode,
        kv_quant_group_size=kv_quant_group_size,
        sliding_window=sliding_window,
    )
