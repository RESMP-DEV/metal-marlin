"""
MLA (Multi-head Latent Attention) kernel wrappers for Metal.

Provides optimized projections for MLA-style attention mechanisms:
- kv_a_proj: hidden → latent (down-projection)
- kv_b_proj: latent → output (up-projection)
- Fused kv_a + kv_b (skip intermediate materialization)
- RoPE-fused variants for position encoding
- Decode-optimized GEMV for single token inference

MLA Architecture (GLM-4, DeepSeek-V2/V3):
    - Uses compressed KV cache via latent projections
    - kv_lora_rank typically 512-1536 (small latent dimension)
    - Enables 10-20x KV cache compression vs standard MHA

Usage:
    from mla_kernels import mla_proj_fp4, mla_fused_kv_proj_fp4, mla_decode_proj_fp4

    # Single projection (kv_a or kv_b)
    latent = mla_proj_fp4(hidden, W_a_packed, scales_a, group_size=64)

    # Fused projection (skip intermediate)
    output = mla_fused_kv_proj_fp4(
        hidden, W_a_packed, scales_a, W_b_packed, scales_b,
        group_size_a=64, group_size_b=64
    )

    # Decode phase (single token)
    output = mla_decode_proj_fp4(x, W_packed, scales, group_size=64)

Note:
    This module uses PyTorch MPS with direct Metal dispatch via PyObjC.
    When PyTorch/MPS is not available, falls back to numpy implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._compat import HAS_MPS, HAS_PYOBJC_METAL, HAS_TORCH, torch

# Metal kernel dispatch requires both PyTorch MPS and PyObjC Metal framework
HAS_METAL_DISPATCH: bool = HAS_MPS and HAS_PYOBJC_METAL

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Constants matching mla_proj.metal
# ---------------------------------------------------------------------------

TILE_M_MLA = 64
TILE_N_MLA = 64
TILE_K_MLA = 16
TILE_K_MLA_LARGE = 32
SIMDGROUPS_PER_TG_MLA = 4
THREADS_PER_TG_MLA = SIMDGROUPS_PER_TG_MLA * 32  # 128
THREADS_PER_TG_DECODE = 64
SG_M_TILES_MLA = 8
SG_N_TILES_MLA = 2
FP4_PER_UINT = 8


# ---------------------------------------------------------------------------
# Metal dispatch infrastructure (lazy imports to avoid circular deps)
# ---------------------------------------------------------------------------

_metal_lib: Any = None


def _get_metal_library() -> Any:
    """Get or create the Metal kernel library."""
    global _metal_lib
    if _metal_lib is None:
        from .metal_dispatch import get_default_library

        _metal_lib = get_default_library()
    return _metal_lib


def _require_mps() -> None:
    """Raise if MPS is not available."""
    if not HAS_MPS:
        raise RuntimeError(
            "MLA kernel dispatch requires PyTorch with MPS backend.\n"
            "Ensure you're on Apple Silicon with PyTorch >= 2.0"
        )


# ---------------------------------------------------------------------------
# Numpy fallback implementations
# ---------------------------------------------------------------------------

# E2M1 codebook for FP4 dequantization
_E2M1_VALUES: np.ndarray = np.array(
    [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, -0.0, -0.25, -0.5, -0.75, -1.0, -1.5, -2.0, -3.0],
    dtype=np.float32,
)


def _dequant_fp4_numpy(
    packed: np.ndarray,
    scales: np.ndarray,
    K: int,
    N: int,
    group_size: int,
) -> np.ndarray:
    """CPU fallback: dequantize packed FP4 weights to float32."""
    packed_k = K // FP4_PER_UINT

    # Extract nibble indices from packed uint32 words
    indices = np.empty((K, N), dtype=np.uint8)
    for k_pack in range(packed_k):
        k_base = k_pack * FP4_PER_UINT
        for i in range(FP4_PER_UINT):
            indices[k_base + i, :] = ((packed[k_pack, :] >> (i * 4)) & 0xF).astype(np.uint8)

    # Dequantize via E2M1 codebook
    values = _E2M1_VALUES[indices].astype(np.float32)

    # Apply per-group scales
    scales_f32 = scales.astype(np.float32)
    scales_expanded = np.repeat(scales_f32, group_size, axis=0)  # [K, N]
    values *= scales_expanded[:K, :]

    return values


def _mla_proj_numpy(
    A: np.ndarray,
    B_packed: np.ndarray,
    scales: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """CPU fallback: MLA projection via numpy."""
    orig_shape = A.shape
    K = orig_shape[-1]
    M = int(np.prod(orig_shape[:-1]))

    A_2d = A.reshape(M, K).astype(np.float32)
    N = B_packed.shape[1]

    # Dequantize weights
    weights_dequant = _dequant_fp4_numpy(B_packed, scales, K, N, group_size)

    # Matrix multiply
    out = A_2d @ weights_dequant

    out_shape = list(orig_shape[:-1]) + [N]
    return out.reshape(out_shape).astype(np.float16)


def _mla_decode_numpy(
    x: np.ndarray,
    W_packed: np.ndarray,
    scales: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """CPU fallback: MLA decode GEMV via numpy."""
    x_flat = x.reshape(-1).astype(np.float32)
    K = x_flat.shape[0]
    N = W_packed.shape[1]

    # Dequantize weights
    weights_dequant = _dequant_fp4_numpy(W_packed, scales, K, N, group_size)

    # Vector-matrix multiply
    out = x_flat @ weights_dequant

    if x.ndim == 2:
        return out.reshape(1, N).astype(np.float16)
    return out.astype(np.float16)


# ---------------------------------------------------------------------------
# Metal kernel dispatch functions
# ---------------------------------------------------------------------------


def _dispatch_mla_proj(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    scales: torch.Tensor,
    M: int,
    N: int,
    K: int,
    group_size: int,
) -> torch.Tensor:
    """Dispatch MLA projection kernel via Metal."""
    import Metal

    from .metal_dispatch import (
        dispatch_kernel,
        mps_tensor_to_metal_buffer,
    )

    lib = _get_metal_library()
    device = lib.device

    # Allocate output
    C = torch.empty((M, N), dtype=torch.float16, device="mps")

    # Convert tensors to Metal buffers
    A_buf = mps_tensor_to_metal_buffer(A.half().contiguous(), device)
    B_buf = mps_tensor_to_metal_buffer(B_packed.contiguous(), device)
    S_buf = mps_tensor_to_metal_buffer(scales.half().contiguous(), device)
    C_buf = mps_tensor_to_metal_buffer(C, device)

    # Create constant buffers
    M_buf = device.newBufferWithBytes_length_options_(
        np.array([M], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    N_buf = device.newBufferWithBytes_length_options_(
        np.array([N], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    K_buf = device.newBufferWithBytes_length_options_(
        np.array([K], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    gs_buf = device.newBufferWithBytes_length_options_(
        np.array([group_size], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )

    # Select kernel based on K dimension
    kernel_name = "mla_proj_fp4_k16" if K <= 1024 else "mla_proj_fp4_k32"

    # Compute grid
    grid_x = (N + TILE_N_MLA - 1) // TILE_N_MLA
    grid_y = (M + TILE_M_MLA - 1) // TILE_M_MLA

    # Dispatch
    dispatch_kernel(
        lib,
        function_name=kernel_name,
        grid=(grid_x, grid_y, 1),
        threadgroup=(THREADS_PER_TG_MLA, 1, 1),
        buffers=[A_buf, B_buf, S_buf, C_buf, M_buf, N_buf, K_buf, gs_buf],
        wait=True,
    )

    return C


def _dispatch_mla_decode(
    x: torch.Tensor,
    W_packed: torch.Tensor,
    scales: torch.Tensor,
    K: int,
    N: int,
    group_size: int,
) -> torch.Tensor:
    """Dispatch MLA decode GEMV kernel via Metal."""
    import Metal

    from .metal_dispatch import (
        dispatch_kernel,
        mps_tensor_to_metal_buffer,
    )

    lib = _get_metal_library()
    device = lib.device

    # Allocate output
    out = torch.empty((N,), dtype=torch.float16, device="mps")

    # Convert tensors to Metal buffers
    x_buf = mps_tensor_to_metal_buffer(x.half().contiguous(), device)
    W_buf = mps_tensor_to_metal_buffer(W_packed.contiguous(), device)
    S_buf = mps_tensor_to_metal_buffer(scales.half().contiguous(), device)
    out_buf = mps_tensor_to_metal_buffer(out, device)

    # Create constant buffers
    K_buf = device.newBufferWithBytes_length_options_(
        np.array([K], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    N_buf = device.newBufferWithBytes_length_options_(
        np.array([N], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    gs_buf = device.newBufferWithBytes_length_options_(
        np.array([group_size], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )

    # Number of threadgroups to cover all output columns
    num_tgs = (N + THREADS_PER_TG_DECODE - 1) // THREADS_PER_TG_DECODE

    # Dispatch
    dispatch_kernel(
        lib,
        function_name="mla_decode_proj_fp4",
        grid=(num_tgs, 1, 1),
        threadgroup=(THREADS_PER_TG_DECODE, 1, 1),
        buffers=[x_buf, W_buf, S_buf, out_buf, K_buf, N_buf, gs_buf],
        wait=True,
    )

    return out


def _dispatch_mla_fused(
    hidden: torch.Tensor,
    W_a_packed: torch.Tensor,
    scales_a: torch.Tensor,
    W_b_packed: torch.Tensor,
    scales_b: torch.Tensor,
    M: int,
    K_hidden: int,
    K_latent: int,
    N_out: int,
    group_size_a: int,
    group_size_b: int,
) -> torch.Tensor:
    """Dispatch fused kv_a + kv_b projection kernel via Metal."""
    import Metal

    from .metal_dispatch import (
        dispatch_kernel,
        mps_tensor_to_metal_buffer,
    )

    lib = _get_metal_library()
    device = lib.device

    # Allocate output
    out = torch.empty((M, N_out), dtype=torch.float16, device="mps")

    # Convert tensors to Metal buffers
    hidden_buf = mps_tensor_to_metal_buffer(hidden.half().contiguous(), device)
    Wa_buf = mps_tensor_to_metal_buffer(W_a_packed.contiguous(), device)
    Sa_buf = mps_tensor_to_metal_buffer(scales_a.half().contiguous(), device)
    Wb_buf = mps_tensor_to_metal_buffer(W_b_packed.contiguous(), device)
    Sb_buf = mps_tensor_to_metal_buffer(scales_b.half().contiguous(), device)
    out_buf = mps_tensor_to_metal_buffer(out, device)

    # Create constant buffers
    M_buf = device.newBufferWithBytes_length_options_(
        np.array([M], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    Kh_buf = device.newBufferWithBytes_length_options_(
        np.array([K_hidden], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    Kl_buf = device.newBufferWithBytes_length_options_(
        np.array([K_latent], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    Nout_buf = device.newBufferWithBytes_length_options_(
        np.array([N_out], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    gsa_buf = device.newBufferWithBytes_length_options_(
        np.array([group_size_a], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    gsb_buf = device.newBufferWithBytes_length_options_(
        np.array([group_size_b], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )

    # Compute grid
    grid_x = (N_out + TILE_N_MLA - 1) // TILE_N_MLA
    grid_y = (M + TILE_M_MLA - 1) // TILE_M_MLA

    # Dispatch
    dispatch_kernel(
        lib,
        function_name="mla_fused_kv_proj_fp4",
        grid=(grid_x, grid_y, 1),
        threadgroup=(THREADS_PER_TG_MLA, 1, 1),
        buffers=[
            hidden_buf,
            Wa_buf,
            Sa_buf,
            Wb_buf,
            Sb_buf,
            out_buf,
            M_buf,
            Kh_buf,
            Kl_buf,
            Nout_buf,
            gsa_buf,
            gsb_buf,
        ],
        wait=True,
    )

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mla_proj_fp4(
    A: Any,
    B_packed: Any,
    scales: Any,
    group_size: int = 64,
    dtype: Any = None,
) -> Any:
    """
    MLA single projection with FP4 quantized weights.

    Optimized for MLA's tall-skinny (kv_a) or short-wide (kv_b) matrices.
    Automatically selects TILE_K based on K dimension.

    Uses Metal kernel when PyTorch MPS is available, otherwise falls back to numpy.

    Args:
        A: Input activations [*, K]. Arbitrary leading dims.
        B_packed: Packed FP4 weights [K/8, N] as uint32.
        scales: Per-group scales [K/group_size, N].
        group_size: Elements per quantization group (default 64 for MLA).
        dtype: Output dtype (currently unused, output is always float16).

    Returns:
        Output [*, N].
    """
    if HAS_METAL_DISPATCH and torch is not None:
        # Ensure inputs are on MPS
        if not isinstance(A, torch.Tensor):
            A = torch.from_numpy(np.asarray(A)).to("mps")
        elif not A.is_mps:
            A = A.to("mps")

        if not isinstance(B_packed, torch.Tensor):
            B_packed = torch.from_numpy(np.asarray(B_packed)).to("mps")
        elif not B_packed.is_mps:
            B_packed = B_packed.to("mps")

        if not isinstance(scales, torch.Tensor):
            scales = torch.from_numpy(np.asarray(scales)).to("mps")
        elif not scales.is_mps:
            scales = scales.to("mps")

        orig_shape = A.shape
        K = orig_shape[-1]
        M = 1
        for d in orig_shape[:-1]:
            M *= d

        A_2d = A.reshape(M, K)
        N = B_packed.shape[1]

        out = _dispatch_mla_proj(A_2d, B_packed, scales, M, N, K, group_size)

        out_shape = list(orig_shape[:-1]) + [N]
        return out.reshape(out_shape)
    else:
        # Numpy fallback
        A_np = np.asarray(A)
        B_np = np.asarray(B_packed)
        scales_np = np.asarray(scales)
        return _mla_proj_numpy(A_np, B_np, scales_np, group_size)


def mla_decode_proj_fp4(
    x: Any,
    W_packed: Any,
    scales: Any,
    group_size: int = 64,
    dtype: Any = None,
) -> Any:
    """
    MLA decode-phase projection (GEMV) with FP4 weights.

    Optimized for single token inference (batch=1).
    Uses GEMV pattern instead of full GEMM for better efficiency.

    Uses Metal kernel when PyTorch MPS is available, otherwise falls back to numpy.

    Args:
        x: Input vector [K] or [1, K].
        W_packed: Packed FP4 weights [K/8, N] as uint32.
        scales: Per-group scales [K/group_size, N].
        group_size: Elements per quantization group.
        dtype: Output dtype (currently unused, output is always float16).

    Returns:
        Output [N] or [1, N].
    """
    if HAS_METAL_DISPATCH and torch is not None:
        # Ensure inputs are on MPS
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.asarray(x)).to("mps")
        elif not x.is_mps:
            x = x.to("mps")

        if not isinstance(W_packed, torch.Tensor):
            W_packed = torch.from_numpy(np.asarray(W_packed)).to("mps")
        elif not W_packed.is_mps:
            W_packed = W_packed.to("mps")

        if not isinstance(scales, torch.Tensor):
            scales = torch.from_numpy(np.asarray(scales)).to("mps")
        elif not scales.is_mps:
            scales = scales.to("mps")

        orig_ndim = x.ndim
        x_flat = x.reshape(-1)
        K = x_flat.shape[0]
        N = W_packed.shape[1]

        out = _dispatch_mla_decode(x_flat, W_packed, scales, K, N, group_size)

        if orig_ndim == 2:
            return out.reshape(1, N)
        return out
    else:
        # Numpy fallback
        x_np = np.asarray(x)
        W_np = np.asarray(W_packed)
        scales_np = np.asarray(scales)
        return _mla_decode_numpy(x_np, W_np, scales_np, group_size)


def mla_fused_kv_proj_fp4(
    hidden: Any,
    W_a_packed: Any,
    scales_a: Any,
    W_b_packed: Any,
    scales_b: Any,
    group_size_a: int = 64,
    group_size_b: int = 64,
    dtype: Any = None,
) -> Any:
    """
    Fused kv_a + kv_b projection (skip intermediate materialization).

    Computes: output = hidden @ W_a @ W_b
    Without materializing the intermediate latent tensor.

    This is ~15-20% faster than separate projections for small latent dims.

    Args:
        hidden: Input hidden states [batch, seq, hidden_size].
        W_a_packed: kv_a weights [hidden_size/8, kv_lora_rank] FP4.
        scales_a: kv_a scales [hidden_size/group_size_a, kv_lora_rank].
        W_b_packed: kv_b weights [kv_lora_rank/8, n_kv_heads * head_dim] FP4.
        scales_b: kv_b scales [kv_lora_rank/group_size_b, n_kv_heads * head_dim].
        group_size_a: Group size for W_a quantization.
        group_size_b: Group size for W_b quantization.
        dtype: Output dtype (currently unused).

    Returns:
        Output [batch, seq, n_kv_heads * head_dim].

    Note:
        Uses the fused Metal kernel when available (mla_fused_kv_proj_fp4 in mla_proj.metal).
        Falls back to two sequential projections when fused kernel is unavailable.
    """
    if HAS_METAL_DISPATCH and torch is not None:
        # Ensure inputs are on MPS
        if not isinstance(hidden, torch.Tensor):
            hidden = torch.from_numpy(np.asarray(hidden)).to("mps")
        elif not hidden.is_mps:
            hidden = hidden.to("mps")

        if not isinstance(W_a_packed, torch.Tensor):
            W_a_packed = torch.from_numpy(np.asarray(W_a_packed)).to("mps")
        elif not W_a_packed.is_mps:
            W_a_packed = W_a_packed.to("mps")

        if not isinstance(scales_a, torch.Tensor):
            scales_a = torch.from_numpy(np.asarray(scales_a)).to("mps")
        elif not scales_a.is_mps:
            scales_a = scales_a.to("mps")

        if not isinstance(W_b_packed, torch.Tensor):
            W_b_packed = torch.from_numpy(np.asarray(W_b_packed)).to("mps")
        elif not W_b_packed.is_mps:
            W_b_packed = W_b_packed.to("mps")

        if not isinstance(scales_b, torch.Tensor):
            scales_b = torch.from_numpy(np.asarray(scales_b)).to("mps")
        elif not scales_b.is_mps:
            scales_b = scales_b.to("mps")

        orig_shape = hidden.shape
        K_hidden = orig_shape[-1]
        M = 1
        for d in orig_shape[:-1]:
            M *= d

        K_latent = W_a_packed.shape[1]
        N_out = W_b_packed.shape[1]

        hidden_2d = hidden.reshape(M, K_hidden)

        try:
            # Try the fused kernel
            out = _dispatch_mla_fused(
                hidden_2d,
                W_a_packed,
                scales_a,
                W_b_packed,
                scales_b,
                M,
                K_hidden,
                K_latent,
                N_out,
                group_size_a,
                group_size_b,
            )
        except (KeyError, RuntimeError):
            # Fallback to sequential projections
            latent = mla_proj_fp4(hidden, W_a_packed, scales_a, group_size_a, dtype)
            out = mla_proj_fp4(latent, W_b_packed, scales_b, group_size_b, dtype)
            return out

        out_shape = list(orig_shape[:-1]) + [N_out]
        return out.reshape(out_shape)
    else:
        # Numpy fallback: use two sequential projections
        latent = mla_proj_fp4(hidden, W_a_packed, scales_a, group_size_a, dtype)
        output = mla_proj_fp4(latent, W_b_packed, scales_b, group_size_b, dtype)
        return output


def mla_proj_with_rope_fp4(
    A: Any,
    B_packed: Any,
    scales: Any,
    cos_cache: Any,
    sin_cache: Any,
    positions: Any,
    rope_dim: int,
    group_size: int = 64,
    dtype: Any = None,
) -> Any:
    """
    MLA projection with fused RoPE on partial output.

    For GLM-4 style models where RoPE is applied to decoupled rope_head_dim.

    Args:
        A: Input activations [*, K].
        B_packed: Packed FP4 weights [K/8, N] as uint32.
        scales: Per-group scales [K/group_size, N].
        cos_cache: Cosine cache [max_seq, rope_dim/2].
        sin_cache: Sine cache [max_seq, rope_dim/2].
        positions: Position indices [*] matching leading dims of A.
        rope_dim: Number of dimensions to apply RoPE to.
        group_size: Elements per quantization group.
        dtype: Output dtype (currently unused).

    Returns:
        Output [*, N] with RoPE applied to first rope_dim columns.

    Note:
        This is a reference implementation. For maximum performance,
        the mla_proj_with_rope_fp4 Metal kernel should be used directly.
    """
    # Compute projection
    output = mla_proj_fp4(A, B_packed, scales, group_size, dtype)

    if HAS_METAL_DISPATCH and torch is not None:
        # PyTorch path
        if not isinstance(output, torch.Tensor):
            output = torch.from_numpy(output).to("mps")

        if not isinstance(cos_cache, torch.Tensor):
            cos_cache = torch.from_numpy(np.asarray(cos_cache)).to("mps")
        elif not cos_cache.is_mps:
            cos_cache = cos_cache.to("mps")

        if not isinstance(sin_cache, torch.Tensor):
            sin_cache = torch.from_numpy(np.asarray(sin_cache)).to("mps")
        elif not sin_cache.is_mps:
            sin_cache = sin_cache.to("mps")

        if not isinstance(positions, torch.Tensor):
            positions = torch.from_numpy(np.asarray(positions)).to("mps")
        elif not positions.is_mps:
            positions = positions.to("mps")

        orig_shape = output.shape
        output_flat = output.reshape(-1, orig_shape[-1])
        positions_flat = positions.reshape(-1)

        half_rope = rope_dim // 2
        rope_x = output_flat[:, :half_rope]
        rope_y = output_flat[:, half_rope:rope_dim]

        # Gather cos/sin from cache
        cos_vals = cos_cache[positions_flat]  # [M, half_rope]
        sin_vals = sin_cache[positions_flat]  # [M, half_rope]

        # Apply rotation: (x, y) -> (x*cos - y*sin, x*sin + y*cos)
        rope_x_new = rope_x * cos_vals - rope_y * sin_vals
        rope_y_new = rope_x * sin_vals + rope_y * cos_vals

        # Reconstruct output
        output_flat = torch.cat([rope_x_new, rope_y_new, output_flat[:, rope_dim:]], dim=-1)

        return output_flat.reshape(orig_shape)
    else:
        # Numpy fallback
        output_np = np.asarray(output)
        cos_np = np.asarray(cos_cache)
        sin_np = np.asarray(sin_cache)
        positions_np = np.asarray(positions)

        orig_shape = output_np.shape
        output_flat = output_np.reshape(-1, orig_shape[-1])
        positions_flat = positions_np.reshape(-1)

        half_rope = rope_dim // 2
        rope_x = output_flat[:, :half_rope]
        rope_y = output_flat[:, half_rope:rope_dim]

        cos_vals = cos_np[positions_flat]
        sin_vals = sin_np[positions_flat]

        rope_x_new = rope_x * cos_vals - rope_y * sin_vals
        rope_y_new = rope_x * sin_vals + rope_y * cos_vals

        output_flat = np.concatenate([rope_x_new, rope_y_new, output_flat[:, rope_dim:]], axis=-1)

        return output_flat.reshape(orig_shape)


def mla_decode_batched_fp4(
    x: Any,
    W_packed: Any,
    scales: Any,
    group_size: int = 64,
    dtype: Any = None,
) -> Any:
    """
    Batched MLA decode projection (multiple tokens).

    For small batch sizes during decode phase.

    Args:
        x: Input [batch, K].
        W_packed: Packed FP4 weights [K/8, N] as uint32.
        scales: Per-group scales [K/group_size, N].
        group_size: Elements per quantization group.
        dtype: Output dtype (currently unused).

    Returns:
        Output [batch, N].
    """
    # For small batches, use mla_proj_fp4 which handles arbitrary leading dims
    return mla_proj_fp4(x, W_packed, scales, group_size, dtype)


# ---------------------------------------------------------------------------
# Weight packing utilities
# ---------------------------------------------------------------------------


def pack_fp4_weights_mla(
    weight: Any,
    group_size: int = 64,
) -> tuple[Any, Any]:
    """
    Pack weights for MLA FP4 projection.

    Same as standard FP4 packing but with MLA-optimal group sizes.

    Args:
        weight: FP16/FP32 weight matrix [out_features, in_features].
        group_size: Quantization group size (default 64 for MLA).

    Returns:
        (packed_weights, scales) for use with mla_proj_fp4.
    """
    if HAS_TORCH and torch is not None:
        # Use PyTorch-based packing from kernels module
        try:
            from .kernels import pack_fp4_weights

            # Convert to PyTorch tensor if needed
            if not isinstance(weight, torch.Tensor):
                weight = torch.from_numpy(np.asarray(weight))
            return pack_fp4_weights(weight, group_size)
        except ImportError:
            pass  # Fall through to numpy implementation

    # Numpy fallback
    weight_np = np.asarray(weight).astype(np.float32)
    K, N = weight_np.shape

    # Compute per-group scales (max abs value per group)
    num_groups = K // group_size
    weight_grouped = weight_np.reshape(num_groups, group_size, N)
    scales = np.max(np.abs(weight_grouped), axis=1) / 6.0  # FP4 max is 6
    scales = np.maximum(scales, 1e-8)  # Avoid division by zero

    # Quantize to FP4
    scales_expanded = np.repeat(scales, group_size, axis=0)
    weight_scaled = weight_np / scales_expanded
    weight_clipped = np.clip(weight_scaled, -6.0, 6.0)

    # Map to FP4 indices (simplified)
    # This is a rough approximation of E2M1 encoding
    weight_quant = np.round(weight_clipped * 2.0).astype(np.int8) + 8
    weight_quant = np.clip(weight_quant, 0, 15).astype(np.uint8)

    # Pack 8 values per uint32
    packed = np.zeros((K // FP4_PER_UINT, N), dtype=np.uint32)
    for i in range(FP4_PER_UINT):
        packed |= weight_quant[i::FP4_PER_UINT, :].astype(np.uint32) << (i * 4)

    return packed, scales.astype(np.float16)


# ---------------------------------------------------------------------------
# MLA Layer Wrapper
# ---------------------------------------------------------------------------


class MLALinear:
    """
    Quantized linear layer optimized for MLA projections.

    Automatically selects between GEMM and GEMV based on batch size.

    Usage:
        kv_a = MLALinear(hidden_size, kv_lora_rank, group_size=64)
        kv_a.load_quantized(packed_weights, scales)
        latent = kv_a(hidden_states)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 64,
        dtype: Any = None,
    ):
        """
        Initialize MLA linear layer.

        Args:
            in_features: Input dimension (K).
            out_features: Output dimension (N).
            group_size: Quantization group size.
            dtype: Output dtype. If None, uses default.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.dtype = dtype

        self.weight_packed: Any = None
        self.scales: Any = None

    def load_quantized(self, weight_packed: Any, scales: Any) -> None:
        """Load pre-quantized weights."""
        self.weight_packed = weight_packed
        self.scales = scales

    def quantize_weights(self, weight: Any) -> None:
        """Quantize and store FP16/FP32 weights."""
        self.weight_packed, self.scales = pack_fp4_weights_mla(weight, self.group_size)

    def __call__(self, x: Any) -> Any:
        """
        Forward pass.

        Automatically selects GEMV for single token, GEMM otherwise.
        """
        if self.weight_packed is None or self.scales is None:
            raise RuntimeError(
                "Weights not loaded. Call load_quantized() or quantize_weights() first."
            )

        # Check if this is decode phase (single token per batch element)
        if HAS_TORCH and torch is not None and isinstance(x, torch.Tensor):
            batch_size = 1
            for d in x.shape[:-1]:
                batch_size *= d
        else:
            x_np = np.asarray(x)
            batch_size = int(np.prod(x_np.shape[:-1]))

        if batch_size == 1:
            return mla_decode_proj_fp4(
                x, self.weight_packed, self.scales, self.group_size, self.dtype
            )
        else:
            return mla_proj_fp4(x, self.weight_packed, self.scales, self.group_size, self.dtype)


class MLAAttentionProjections:
    """
    Complete MLA projection stack (kv_a, kv_b, q_a, q_b).

    Encapsulates the latent projection pattern for MLA attention.

    MLA Architecture:
        Q: hidden -> q_a -> q_b -> [num_heads, head_dim]
        KV: hidden -> kv_a -> [kv_lora_rank] (cached) -> kv_b -> [num_kv_heads, 2*head_dim]
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_lora_rank: int | None = None,
        kv_lora_rank: int = 512,
        rope_head_dim: int = 0,
        group_size: int = 64,
        dtype: Any = None,
    ):
        """
        Initialize MLA projection layers.

        Args:
            hidden_size: Model hidden dimension.
            num_heads: Number of attention heads.
            num_kv_heads: Number of KV heads (for GQA).
            head_dim: Dimension per head.
            q_lora_rank: Query latent rank (None = no compression).
            kv_lora_rank: KV latent rank.
            rope_head_dim: Dimension for decoupled RoPE.
            group_size: Quantization group size.
            dtype: Compute dtype.
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.rope_head_dim = rope_head_dim
        self.group_size = group_size
        self.dtype = dtype

        # KV projections (always compressed)
        self.kv_a_proj = MLALinear(hidden_size, kv_lora_rank + rope_head_dim, group_size, dtype)
        self.kv_b_proj = MLALinear(kv_lora_rank, num_kv_heads * head_dim * 2, group_size, dtype)

        # Q projections (optionally compressed)
        if q_lora_rank is not None:
            self.q_a_proj = MLALinear(hidden_size, q_lora_rank, group_size, dtype)
            self.q_b_proj = MLALinear(q_lora_rank, num_heads * head_dim, group_size, dtype)
            self.q_proj: MLALinear | None = None
        else:
            self.q_proj = MLALinear(hidden_size, num_heads * head_dim, group_size, dtype)
            self.q_a_proj: MLALinear | None = None
            self.q_b_proj: MLALinear | None = None

    def project_kv(self, hidden_states: Any) -> tuple[Any, Any | None]:
        """
        Project hidden states to KV cache representation.

        Args:
            hidden_states: [batch, seq, hidden_size]

        Returns:
            (kv_latent, rope_latent) where:
                kv_latent: [batch, seq, kv_lora_rank] - to be cached
                rope_latent: [batch, seq, rope_head_dim] - for RoPE (if rope_head_dim > 0)
        """
        # Down-project to latent space
        kv_a_out = self.kv_a_proj(hidden_states)  # [batch, seq, kv_lora_rank + rope_head_dim]

        if self.rope_head_dim > 0:
            # Split into cacheable latent and RoPE portion
            if HAS_TORCH and torch is not None and isinstance(kv_a_out, torch.Tensor):
                kv_latent = kv_a_out[..., : self.kv_lora_rank]
                rope_latent = kv_a_out[..., self.kv_lora_rank :]
            else:
                kv_a_out_np = np.asarray(kv_a_out)
                kv_latent = kv_a_out_np[..., : self.kv_lora_rank]
                rope_latent = kv_a_out_np[..., self.kv_lora_rank :]
            return kv_latent, rope_latent
        else:
            return kv_a_out, None

    def decompress_kv(self, kv_latent: Any) -> Any:
        """
        Decompress KV latent to full K,V tensors.

        Args:
            kv_latent: [batch, seq, kv_lora_rank] from cache

        Returns:
            kv: [batch, seq, num_kv_heads * head_dim * 2]
        """
        return self.kv_b_proj(kv_latent)

    def project_q(self, hidden_states: Any) -> Any:
        """
        Project hidden states to queries.

        Args:
            hidden_states: [batch, seq, hidden_size]

        Returns:
            q: [batch, seq, num_heads * head_dim]
        """
        if self.q_lora_rank is not None and self.q_a_proj is not None and self.q_b_proj is not None:
            q_latent = self.q_a_proj(hidden_states)
            return self.q_b_proj(q_latent)
        elif self.q_proj is not None:
            return self.q_proj(hidden_states)
        else:
            raise RuntimeError("Q projection not configured")
