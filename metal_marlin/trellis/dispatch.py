"""Metal dispatch for trellis dequantization kernels.

Provides Python wrappers for the dequant_trellis.metal kernels:
- dispatch_trellis_dequant: Dequantize trellis indices to FP16
- dispatch_sign_flips: Apply su/sv sign corrections
- dispatch_trellis_dequant_fused: Combined dequant + sign flip

Reference: src/dequant_trellis.metal
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ..fast_inference import fast_dispatch_available, get_fast_context
from ..metal_dispatch import (
    HAS_METAL,
    HAS_MPS,
    MetalKernelLibrary,
    dispatch_kernel,
    mps_tensor_to_metal_buffer,
    require_mps,
)
from ..quantization.trellis_codebook import TrellisCodebook

# Debug logging gated by METAL_DEBUG env var
_METAL_DEBUG = os.environ.get(
    "METAL_DEBUG", "").lower() in ("1", "true", "yes")
_logger = logging.getLogger(__name__)
if _METAL_DEBUG:
    logging.basicConfig(level=logging.DEBUG)


if TYPE_CHECKING:
    from .loader import TrellisWeight

if HAS_METAL:
    import Metal


def dispatch_trellis_dequant(
    lib: MetalKernelLibrary,
    indices: torch.Tensor,
    scales: torch.Tensor,
    grid: torch.Tensor,
    K: int,
    N: int,
    group_size: int = 128,
) -> torch.Tensor:
    """Dequantize trellis indices to FP16 weights.

    Args:
        lib: MetalKernelLibrary with dequant_trellis compiled
        indices: Trellis indices [tiles_k, tiles_n, 256] uint8, MPS tensor
        scales: Per-group scales [n_groups, N] float32, MPS tensor
        grid: Codebook grid [n_levels] float32, MPS tensor
        K: Number of rows in output
        N: Number of columns in output
        group_size: Elements per quantization group (default 128)

    Returns:
        Dequantized weights [K, N] float16, MPS tensor
    """
    require_mps()

    device = lib.device
    n_levels = grid.shape[0]

    # Ensure proper types and contiguity
    indices = indices.contiguous()
    scales = scales.contiguous() if scales.dtype == torch.float32 else scales.half().contiguous()
    grid = grid.contiguous() if grid.dtype == torch.float32 else grid.half().contiguous()

    # Allocate output
    output = torch.zeros(K, N, dtype=torch.float16, device="mps")

    # Create Metal buffers
    indices_buf = mps_tensor_to_metal_buffer(indices, device)
    scales_buf = mps_tensor_to_metal_buffer(scales, device)
    grid_buf = mps_tensor_to_metal_buffer(grid, device)
    output_buf = mps_tensor_to_metal_buffer(output, device, copy_back=True)

    # Parameters
    params = np.array([K, N, n_levels, group_size], dtype=np.uint32)
    params_buf = device.newBufferWithBytes_length_options_(
        params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
    )

    # Dispatch with one thread per output element
    # Grid: (N, K, 1) threads
    threads_per_tg = 16
    grid_x = (N + threads_per_tg - 1) // threads_per_tg
    grid_y = (K + threads_per_tg - 1) // threads_per_tg

    dispatch_kernel(
        lib,
        function_name="dequant_trellis",
        grid=(grid_x, grid_y, 1),
        threadgroup=(threads_per_tg, threads_per_tg, 1),
        buffers=[
            indices_buf,
            scales_buf,
            grid_buf,
            output_buf,
            params_buf,  # K
            params_buf,  # N (offset handled by Metal)
            params_buf,  # n_levels
            params_buf,  # group_size
        ],
        wait=True,
    )

    return output


def dispatch_sign_flips(
    lib: MetalKernelLibrary,
    weights: torch.Tensor,
    su: torch.Tensor,
    sv: torch.Tensor,
) -> torch.Tensor:
    """Apply sign flips to dequantized weights in-place.

    The trellis format stores sign information in su (row signs) and sv (column signs).
    This applies: out[k,n] = weights[k,n] * su[k] * sv[n]

    Args:
        lib: MetalKernelLibrary with dequant_trellis compiled
        weights: Dequantized weights [K, N] float16, MPS tensor (modified in-place)
        su: Row sign vector [K] float32, MPS tensor
        sv: Column sign vector [N] float32, MPS tensor

    Returns:
        Same tensor with signs applied (modified in-place)
    """
    require_mps()

    device = lib.device
    K, N = weights.shape

    # Ensure proper types
    weights = weights.contiguous()
    su = su.float().contiguous()
    sv = sv.float().contiguous()

    # Create Metal buffers
    weights_buf = mps_tensor_to_metal_buffer(weights, device, copy_back=True)
    su_buf = mps_tensor_to_metal_buffer(su, device)
    sv_buf = mps_tensor_to_metal_buffer(sv, device)

    # Parameters
    params = np.array([K, N], dtype=np.uint32)
    params_buf = device.newBufferWithBytes_length_options_(
        params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
    )

    # Dispatch
    threads_per_tg = 16
    grid_x = (N + threads_per_tg - 1) // threads_per_tg
    grid_y = (K + threads_per_tg - 1) // threads_per_tg

    dispatch_kernel(
        lib,
        function_name="apply_sign_flips",
        grid=(grid_x, grid_y, 1),
        threadgroup=(threads_per_tg, threads_per_tg, 1),
        buffers=[weights_buf, su_buf, sv_buf, params_buf, params_buf],
        wait=True,
    )

    return weights


def dispatch_trellis_dequant_fused(
    lib: MetalKernelLibrary,
    indices: torch.Tensor,
    scales: torch.Tensor,
    grid: torch.Tensor,
    su: torch.Tensor,
    sv: torch.Tensor,
    K: int,
    N: int,
    group_size: int = 128,
) -> torch.Tensor:
    """Fused trellis dequantization with sign flip application.

    More efficient than separate dequant + sign_flips calls.

    Args:
        lib: MetalKernelLibrary with dequant_trellis compiled
        indices: Trellis indices [tiles_k, tiles_n, 256] uint8, MPS tensor
        scales: Per-group scales [n_groups, N] float32, MPS tensor
        grid: Codebook grid [n_levels] float32, MPS tensor
        su: Row sign vector [K] float32, MPS tensor
        sv: Column sign vector [N] float32, MPS tensor
        K: Number of rows in output
        N: Number of columns in output
        group_size: Elements per quantization group (default 128)

    Returns:
        Dequantized weights with signs applied [K, N] float16, MPS tensor
    """
    require_mps()

    device = lib.device
    n_levels = grid.shape[0]

    # Ensure proper types
    indices = indices.contiguous()
    scales = scales.float().contiguous()
    grid = grid.float().contiguous()
    su = su.float().contiguous()
    sv = sv.float().contiguous()

    # Allocate output
    output = torch.zeros(K, N, dtype=torch.float16, device="mps")

    # Create Metal buffers
    indices_buf = mps_tensor_to_metal_buffer(indices, device)
    scales_buf = mps_tensor_to_metal_buffer(scales, device)
    grid_buf = mps_tensor_to_metal_buffer(grid, device)
    su_buf = mps_tensor_to_metal_buffer(su, device)
    sv_buf = mps_tensor_to_metal_buffer(sv, device)
    output_buf = mps_tensor_to_metal_buffer(output, device, copy_back=True)

    # Parameters
    params = np.array([K, N, n_levels, group_size], dtype=np.uint32)
    params_buf = device.newBufferWithBytes_length_options_(
        params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
    )

    # Dispatch
    threads_per_tg = 16
    grid_x = (N + threads_per_tg - 1) // threads_per_tg
    grid_y = (K + threads_per_tg - 1) // threads_per_tg

    dispatch_kernel(
        lib,
        function_name="dequant_trellis_with_signs",
        grid=(grid_x, grid_y, 1),
        threadgroup=(threads_per_tg, threads_per_tg, 1),
        buffers=[
            indices_buf,
            scales_buf,
            grid_buf,
            su_buf,
            sv_buf,
            output_buf,
            params_buf,  # K
            params_buf,  # N
            params_buf,  # n_levels
            params_buf,  # group_size
        ],
        wait=True,
    )

    return output


def dispatch_trellis_dequant_packed(
    lib: MetalKernelLibrary,
    packed_indices: torch.Tensor,
    scales: torch.Tensor,
    grid: Any,  # Tensor or MTLTexture
    su: torch.Tensor,
    sv: torch.Tensor,
    K: int,
    N: int,
    bits: int,
    group_size: int = 32,
) -> torch.Tensor:
    """Dequantize packed trellis indices to FP16 weights.

    Uses the packed uint8 format directly without unpacking on CPU.

    Args:
        lib: MetalKernelLibrary with dequant_trellis compiled
        packed_indices: Packed trellis indices [tiles_k, tiles_n, packed_bytes] uint8, MPS tensor
        scales: Per-group scales [n_groups, N] float32, MPS tensor
        grid: Codebook grid [n_levels] float32, MPS tensor OR MTLTexture
        su: Row sign vector [K] float32, MPS tensor
        sv: Column sign vector [N] float32, MPS tensor
        K: Number of rows in output
        N: Number of columns in output
        bits: Quantization bit width (2, 3, or 4)

    Returns:
        Dequantized weights [K, N] float16, MPS tensor
    """
    require_mps()

    device = lib.device

    # Handle grid (Tensor or Texture)
    if hasattr(grid, "textureType"):
        grid_buf = grid
        n_levels = grid.width()
        # If using texture, use texture-specific kernel
        kernel_name = "dequant_trellis_packed_tex"
    else:
        grid = grid.float().contiguous()
        n_levels = grid.shape[0]
        grid_buf = mps_tensor_to_metal_buffer(grid, device)
        kernel_name = "dequant_trellis_packed"

    # Ensure proper types and contiguity
    packed_indices = packed_indices.contiguous()
    scales = scales.float().contiguous()
    su = su.float().contiguous()
    sv = sv.float().contiguous()

    # Allocate output
    output = torch.zeros(K, N, dtype=torch.float16, device="mps")

    # Create Metal buffers
    packed_indices_buf = mps_tensor_to_metal_buffer(packed_indices, device)
    scales_buf = mps_tensor_to_metal_buffer(scales, device)
    su_buf = mps_tensor_to_metal_buffer(su, device)
    sv_buf = mps_tensor_to_metal_buffer(sv, device)
    output_buf = mps_tensor_to_metal_buffer(output, device, copy_back=True)

    # Create separate buffers for each constant parameter
    def make_uint_buffer(val: int) -> Any:
        data = np.array([val], dtype=np.uint32)
        return device.newBufferWithBytes_length_options_(
            data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared
        )

    K_buf = make_uint_buffer(K)
    N_buf = make_uint_buffer(N)
    n_levels_buf = make_uint_buffer(n_levels)
    bits_buf = make_uint_buffer(bits)
    group_size_buf = make_uint_buffer(group_size)

    # Dispatch with one thread per output element
    threads_per_tg = 16
    grid_x = (N + threads_per_tg - 1) // threads_per_tg
    grid_y = (K + threads_per_tg - 1) // threads_per_tg

    dispatch_kernel(
        lib,
        function_name=kernel_name,
        grid=(grid_x, grid_y, 1),
        threadgroup=(threads_per_tg, threads_per_tg, 1),
        buffers=[
            packed_indices_buf,
            scales_buf,
            grid_buf,
            su_buf,
            sv_buf,
            output_buf,
            K_buf,
            N_buf,
            n_levels_buf,
            bits_buf,
            group_size_buf,
        ],
        wait=True,
    )

    return output


def dispatch_fused_qkv_trellis(
    lib: MetalKernelLibrary,
    A: torch.Tensor,
    q_proj: Any,  # TrellisLinear
    k_proj: Any,  # TrellisLinear
    v_proj: Any,  # TrellisLinear
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Q/K/V projections in a single kernel launch.

    Computes all three attention projections simultaneously:
        Q = A @ Wq
        K = A @ Wk
        V = A @ Wv

    Saves 2 kernel launches and loads A only once (3x memory bandwidth reduction).

    Args:
        lib: MetalKernelLibrary with fused_qkv_trellis compiled
        A: Input activations [M, K] float16, MPS tensor
        q_proj: TrellisLinear for Q projection (or None)
        k_proj: TrellisLinear for K projection (or None)
        v_proj: TrellisLinear for V projection (or None)

    Returns:
        Tuple of (Q, K, V) output tensors, each [M, N_*] float16
    """
    require_mps()

    device = lib.device
    M = A.shape[0]

    # Identify at least one valid projection to get common params
    valid_proj = next(p for p in [q_proj, k_proj, v_proj] if p is not None)
    K = valid_proj.in_features
    bits = valid_proj.bits
    n_levels = valid_proj.grid.shape[0]

    # Infer group_size from scales shape
    n_groups = valid_proj.scales.shape[0]
    group_size = (K + n_groups - 1) // n_groups

    # Ensure proper types and contiguity
    A = A.contiguous()
    A_buf = mps_tensor_to_metal_buffer(A, device)

    # Shared grid (all use same codebook)
    grid_buf = mps_tensor_to_metal_buffer(
        valid_proj.grid.float().contiguous(), device)

    # Helpers for dummy buffers
    def get_proj_buffers(proj):
        if proj is None:
            # Dummy buffers
            dummy_idx = device.newBufferWithLength_options_(
                1, Metal.MTLResourceStorageModePrivate)
            dummy_float = device.newBufferWithLength_options_(
                4, Metal.MTLResourceStorageModePrivate)
            return dummy_idx, dummy_float, dummy_float, dummy_float, 0

        return (
            mps_tensor_to_metal_buffer(
                proj.packed_indices.contiguous(), device),
            mps_tensor_to_metal_buffer(
                proj.scales.float().contiguous(), device),
            mps_tensor_to_metal_buffer(proj.su.float().contiguous(), device),
            mps_tensor_to_metal_buffer(proj.sv.float().contiguous(), device),
            proj.out_features
        )

    # Get buffers and dimensions
    Wq_buf, scales_q_buf, su_q_buf, sv_q_buf, Nq = get_proj_buffers(q_proj)
    Wk_buf, scales_k_buf, su_k_buf, sv_k_buf, Nk = get_proj_buffers(k_proj)
    Wv_buf, scales_v_buf, su_v_buf, sv_v_buf, Nv = get_proj_buffers(v_proj)

    # Allocate outputs
    out_q = torch.zeros(M, Nq if Nq > 0 else 1,
                        dtype=torch.float16, device="mps")
    out_k = torch.zeros(M, Nk if Nk > 0 else 1,
                        dtype=torch.float16, device="mps")
    out_v = torch.zeros(M, Nv if Nv > 0 else 1,
                        dtype=torch.float16, device="mps")

    out_q_buf = mps_tensor_to_metal_buffer(out_q, device, copy_back=True)
    out_k_buf = mps_tensor_to_metal_buffer(out_k, device, copy_back=True)
    out_v_buf = mps_tensor_to_metal_buffer(out_v, device, copy_back=True)

    # Create dimension buffers
    def make_uint_buffer(val: int) -> Any:
        data = np.array([val], dtype=np.uint32)
        return device.newBufferWithBytes_length_options_(
            data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared
        )

    M_buf = make_uint_buffer(M)
    K_buf = make_uint_buffer(K)
    Nq_buf = make_uint_buffer(Nq)
    Nk_buf = make_uint_buffer(Nk)
    Nv_buf = make_uint_buffer(Nv)
    bits_buf = make_uint_buffer(bits)
    n_levels_buf = make_uint_buffer(n_levels)
    group_size_buf = make_uint_buffer(group_size)

    # Dispatch configuration
    TILE_M = 32
    TILE_N = 128
    kernel_name = "fused_qkv_trellis_decode"

    # Grid covers max(Nq, Nk, Nv) columns
    max_N = max(Nq, Nk, Nv)
    grid_x = (max_N + TILE_N - 1) // TILE_N
    grid_y = (M + TILE_M - 1) // TILE_M
    threads_per_tg = 128

    dispatch_kernel(
        lib,
        function_name=kernel_name,
        grid=(grid_x, grid_y, 1),
        threadgroup=(threads_per_tg, 1, 1),
        buffers=[
            # Input
            A_buf,
            # Q projection
            Wq_buf, scales_q_buf, su_q_buf, sv_q_buf,
            # K projection
            Wk_buf, scales_k_buf, su_k_buf, sv_k_buf,
            # V projection
            Wv_buf, scales_v_buf, su_v_buf, sv_v_buf,
            # Shared grid
            grid_buf,
            # Outputs
            out_q_buf, out_k_buf, out_v_buf,
            # Dimensions
            M_buf, K_buf, Nq_buf, Nk_buf, Nv_buf,
            bits_buf, n_levels_buf, group_size_buf,
        ],
        wait=True,
    )

    return out_q, out_k, out_v


def dispatch_gemm_trellis_decode(
    lib: MetalKernelLibrary,
    A: torch.Tensor,
    packed_indices: torch.Tensor,
    scales: torch.Tensor,
    grid: torch.Tensor,
    su: torch.Tensor,
    sv: torch.Tensor,
    K: int,
    N: int,
    bits: int,
    group_size: int = 32,
) -> torch.Tensor:
    """Decode-optimized fused GEMM for small M (autoregressive).

    Computes C[M,N] = A[M,K] @ dequant(W[K,N]) where W is trellis-quantized.
    Optimized for M=1 to M=16 (autoregressive decode phase).

    Args:
        lib: MetalKernelLibrary with gemm_trellis compiled
        A: Input activations [M, K] float16, MPS tensor where M <= 16
        packed_indices: Packed trellis indices [tiles_k, tiles_n, packed_bytes] uint8, MPS tensor
        scales: Per-group scales [n_groups, N] float32, MPS tensor
        grid: Codebook grid [n_levels] float32, MPS tensor
        su: Row signs [K] float32, MPS tensor
        sv: Column signs [N] float32, MPS tensor
        K: Number of columns in A / rows in W
        N: Number of columns in W and C
        bits: Quantization bit width (2, 3, or 4)
        group_size: Quantization group size (default 32)

    Returns:
        Output matrix [M, N] float16, MPS tensor
    """
    require_mps()

    device = lib.device
    M = A.shape[0]
    n_levels = grid.shape[0]

    # Ensure proper types and contiguity
    A = A.contiguous()
    packed_indices = packed_indices.contiguous()
    scales = scales.float().contiguous()
    grid = grid.float().contiguous()
    su = su.float().contiguous()
    sv = sv.float().contiguous()

    # Allocate output
    output = torch.zeros(M, N, dtype=torch.float16, device="mps")

    # Create Metal buffers
    A_buf = mps_tensor_to_metal_buffer(A, device)
    packed_indices_buf = mps_tensor_to_metal_buffer(packed_indices, device)
    scales_buf = mps_tensor_to_metal_buffer(scales, device)
    grid_buf = mps_tensor_to_metal_buffer(grid, device)
    su_buf = mps_tensor_to_metal_buffer(su, device)
    sv_buf = mps_tensor_to_metal_buffer(sv, device)
    output_buf = mps_tensor_to_metal_buffer(output, device, copy_back=True)

    # Create separate buffers for each constant parameter
    def make_uint_buffer(val: int) -> Any:
        data = np.array([val], dtype=np.uint32)
        return device.newBufferWithBytes_length_options_(
            data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared
        )

    M_buf = make_uint_buffer(M)
    K_buf = make_uint_buffer(K)
    N_buf = make_uint_buffer(N)
    bits_buf = make_uint_buffer(bits)
    n_levels_buf = make_uint_buffer(n_levels)
    group_size_buf = make_uint_buffer(group_size)

    # Dispatch configuration
    # Tile dimensions from gemm_trellis.metal: DECODE_TILE_M=32, DECODE_TILE_N=128
    TILE_M = 32
    TILE_N = 128
    threads_per_tg = 128  # 4 simdgroups * 32 threads

    grid_x = (N + TILE_N - 1) // TILE_N
    grid_y = (M + TILE_M - 1) // TILE_M

    # Pre-dispatch debug logging
    if _METAL_DEBUG:
        A_cpu = A.cpu().float()
        packed_cpu = packed_indices.cpu()
        scales_cpu = scales.cpu()
        grid_cpu = grid.cpu()
        su_cpu = su.cpu()
        sv_cpu = sv.cpu()

        _logger.debug("=== gemm_trellis_packed_decode PRE-DISPATCH ===")
        _logger.debug(
            "A: min=%.6f, max=%.6f, mean=%.6f, has_nan=%s",
            A_cpu.min().item(),
            A_cpu.max().item(),
            A_cpu.mean().item(),
            bool(torch.isnan(A_cpu).any()),
        )
        _logger.debug(
            "packed_indices: shape=%s, first_10_bytes=%s",
            list(packed_cpu.shape),
            packed_cpu.flatten()[:10].tolist(),
        )
        _logger.debug(
            "scales: min=%.6f, max=%.6f, has_nan=%s",
            scales_cpu.min().item(),
            scales_cpu.max().item(),
            bool(torch.isnan(scales_cpu).any()),
        )
        _logger.debug("grid: values=%s", grid_cpu.tolist())
        _logger.debug(
            "su: min=%.6f, max=%.6f, unique=%s",
            su_cpu.min().item(),
            su_cpu.max().item(),
            torch.unique(su_cpu).tolist(),
        )
        _logger.debug(
            "sv: min=%.6f, max=%.6f, unique=%s",
            sv_cpu.min().item(),
            sv_cpu.max().item(),
            torch.unique(sv_cpu).tolist(),
        )
        _logger.debug(
            "Dimensions: M=%d, K=%d, N=%d, bits=%d, group_size=%d",
            M,
            K,
            N,
            bits,
            group_size,
        )

    dispatch_kernel(
        lib,
        function_name="gemm_trellis_packed_decode",
        grid=(grid_x, grid_y, 1),
        threadgroup=(threads_per_tg, 1, 1),
        buffers=[
            A_buf,
            packed_indices_buf,
            scales_buf,
            grid_buf,
            su_buf,
            sv_buf,
            output_buf,
            M_buf,
            K_buf,
            N_buf,
            bits_buf,
            n_levels_buf,
            group_size_buf,
        ],
        wait=True,
    )

    # Post-dispatch debug logging
    if _METAL_DEBUG:
        output_cpu = output.cpu().float()
        _logger.debug("=== gemm_trellis_packed_decode POST-DISPATCH ===")
        _logger.debug(
            "Output: min=%.6f, max=%.6f, has_nan=%s, first_10=%s",
            output_cpu.min().item(),
            output_cpu.max().item(),
            bool(torch.isnan(output_cpu).any()),
            output_cpu.flatten()[:10].tolist(),
        )

    return output


def dispatch_gemm_trellis_packed(
    lib: MetalKernelLibrary,
    A: torch.Tensor,
    packed_indices: torch.Tensor,
    scales: torch.Tensor,
    grid: torch.Tensor,
    su: torch.Tensor,
    sv: torch.Tensor,
    K: int,
    N: int,
    bits: int,
    group_size: int = 32,
) -> torch.Tensor:
    """Fused trellis dequantization + GEMM.

    Computes C[M,N] = A[M,K] @ dequant(W[K,N]) where W is trellis-quantized.
    Weights are dequantized on-the-fly during the GEMM computation without
    materializing the full FP16 weight matrix.

    Args:
        lib: MetalKernelLibrary with gemm_trellis compiled
        A: Input activations [M, K] float16, MPS tensor
        packed_indices: Packed trellis indices [tiles_k, tiles_n, packed_bytes] uint8, MPS tensor
        scales: Per-group scales [n_groups, N] float32, MPS tensor
        grid: Codebook grid [n_levels] float32, MPS tensor
        su: Row signs [K] float32, MPS tensor
        sv: Column signs [N] float32, MPS tensor
        K: Number of columns in A / rows in W
        N: Number of columns in W and C
        bits: Quantization bit width (2, 3, or 4)
        group_size: Quantization group size (default 32)

    Returns:
        Output matrix [M, N] float16, MPS tensor
    """
    require_mps()

    device = lib.device
    M = A.shape[0]
    n_levels = grid.shape[0]

    # Ensure proper types and contiguity
    A = A.contiguous()
    packed_indices = packed_indices.contiguous()
    scales = scales.float().contiguous()
    grid = grid.float().contiguous()
    su = su.float().contiguous()
    sv = sv.float().contiguous()

    # Allocate output
    output = torch.zeros(M, N, dtype=torch.float16, device="mps")

    # Create Metal buffers
    A_buf = mps_tensor_to_metal_buffer(A, device)
    packed_indices_buf = mps_tensor_to_metal_buffer(packed_indices, device)
    scales_buf = mps_tensor_to_metal_buffer(scales, device)
    grid_buf = mps_tensor_to_metal_buffer(grid, device)
    su_buf = mps_tensor_to_metal_buffer(su, device)
    sv_buf = mps_tensor_to_metal_buffer(sv, device)
    output_buf = mps_tensor_to_metal_buffer(output, device, copy_back=True)

    # Create separate buffers for each constant parameter
    def make_uint_buffer(val: int) -> Any:
        data = np.array([val], dtype=np.uint32)
        return device.newBufferWithBytes_length_options_(
            data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared
        )

    M_buf = make_uint_buffer(M)
    K_buf = make_uint_buffer(K)
    N_buf = make_uint_buffer(N)
    bits_buf = make_uint_buffer(bits)
    n_levels_buf = make_uint_buffer(n_levels)
    group_size_buf = make_uint_buffer(group_size)

    # Dispatch configuration
    # Tile dimensions from gemm_trellis.metal: TILE_M=64, TILE_N=64
    TILE_M = 64
    TILE_N = 64
    threads_per_tg = 128  # 4 simdgroups * 32 threads

    grid_x = (N + TILE_N - 1) // TILE_N
    grid_y = (M + TILE_M - 1) // TILE_M

    dispatch_kernel(
        lib,
        function_name="gemm_trellis_packed",
        grid=(grid_x, grid_y, 1),
        threadgroup=(threads_per_tg, 1, 1),
        buffers=[
            A_buf,
            packed_indices_buf,
            scales_buf,
            grid_buf,
            su_buf,
            sv_buf,
            output_buf,
            M_buf,
            K_buf,
            N_buf,
            bits_buf,
            n_levels_buf,
            group_size_buf,
        ],
        wait=True,
    )

    return output


def dispatch_gemm_trellis_fused_reg(
    lib: MetalKernelLibrary,
    A: torch.Tensor,
    packed_indices: torch.Tensor,
    scales: torch.Tensor,
    grid: torch.Tensor,
    su: torch.Tensor,
    sv: torch.Tensor,
    K: int,
    N: int,
    bits: int,
    group_size: int = 32,
) -> torch.Tensor:
    """Register-optimized fused trellis dequant + GEMM.

    This kernel maximizes dequantization throughput by using all 32 simdgroup
    threads for unpacking and dequantization, rather than just 8. Key optimizations:
    - All 32 threads dequant in parallel (each handles 2 elements of 8x8 B tile)
    - Preload sv[8] into registers, broadcast via simd_shuffle
    - Single-pass staging buffer write (no thread divergence)

    Computes C[M,N] = A[M,K] @ dequant(W[K,N]) where W is trellis-quantized.

    Args:
        lib: MetalKernelLibrary with gemm_trellis compiled
        A: Input activations [M, K] float16, MPS tensor
        packed_indices: Packed trellis indices [tiles_k, tiles_n, packed_bytes] uint8, MPS tensor
        scales: Per-group scales [n_groups, N] float32, MPS tensor
        grid: Codebook grid [n_levels] float32, MPS tensor
        su: Row signs [N] float32, MPS tensor
        sv: Column signs [K] float32, MPS tensor
        K: Number of columns in A / rows in W
        N: Number of columns in W and C
        bits: Quantization bit width (2, 3, or 4)
        group_size: Quantization group size (default 32)

    Returns:
        Output matrix [M, N] float16, MPS tensor
    """
    require_mps()

    device = lib.device
    M = A.shape[0]
    n_levels = grid.shape[0]

    # Ensure proper types and contiguity
    A = A.contiguous()
    packed_indices = packed_indices.contiguous()
    scales = scales.float().contiguous()
    grid = grid.float().contiguous()
    su = su.float().contiguous()
    sv = sv.float().contiguous()

    # Allocate output
    output = torch.zeros(M, N, dtype=torch.float16, device="mps")

    # Create Metal buffers
    A_buf = mps_tensor_to_metal_buffer(A, device)
    packed_indices_buf = mps_tensor_to_metal_buffer(packed_indices, device)
    scales_buf = mps_tensor_to_metal_buffer(scales, device)
    grid_buf = mps_tensor_to_metal_buffer(grid, device)
    su_buf = mps_tensor_to_metal_buffer(su, device)
    sv_buf = mps_tensor_to_metal_buffer(sv, device)
    output_buf = mps_tensor_to_metal_buffer(output, device, copy_back=True)

    # Create separate buffers for each constant parameter
    def make_uint_buffer(val: int) -> Any:
        data = np.array([val], dtype=np.uint32)
        return device.newBufferWithBytes_length_options_(
            data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared
        )

    M_buf = make_uint_buffer(M)
    K_buf = make_uint_buffer(K)
    N_buf = make_uint_buffer(N)
    bits_buf = make_uint_buffer(bits)
    n_levels_buf = make_uint_buffer(n_levels)
    group_size_buf = make_uint_buffer(group_size)

    # Dispatch configuration
    # Tile dimensions from gemm_trellis.metal: TILE_M=64, TILE_N=64
    TILE_M = 64
    TILE_N = 64
    threads_per_tg = 128  # 4 simdgroups * 32 threads

    grid_x = (N + TILE_N - 1) // TILE_N
    grid_y = (M + TILE_M - 1) // TILE_M

    dispatch_kernel(
        lib,
        function_name="gemm_trellis_fused_reg",
        grid=(grid_x, grid_y, 1),
        threadgroup=(threads_per_tg, 1, 1),
        buffers=[
            A_buf,
            packed_indices_buf,
            scales_buf,
            grid_buf,
            su_buf,
            sv_buf,
            output_buf,
            M_buf,
            K_buf,
            N_buf,
            bits_buf,
            n_levels_buf,
            group_size_buf,
        ],
        wait=True,
    )

    return output


def dispatch_gemm_trellis_auto(
    lib: MetalKernelLibrary,
    A: torch.Tensor,
    packed_indices: torch.Tensor,
    scales: torch.Tensor,
    grid: torch.Tensor,
    su: torch.Tensor,
    sv: torch.Tensor,
    K: int,
    N: int,
    bits: int,
    group_size: int = 32,
) -> torch.Tensor:
    """Dispatch GEMM with automatic C++/PyObjC selection.

    Uses C++ extension fast path when available, falls back to PyObjC.

    Args:
        lib: MetalKernelLibrary with gemm_trellis compiled
        A: Input activations [M, K] float16, MPS tensor
        packed_indices: Packed trellis indices [tiles_k, tiles_n, packed_bytes] uint8
        scales: Per-group scales [n_groups, N] float32
        grid: Codebook grid [n_levels] float32
        su: Row signs [K] float32
        sv: Column signs [N] float32
        K: Inner dimension
        N: Output columns
        bits: Quantization bit width (2, 3, or 4)
        group_size: Quantization group size (default 32)

    Returns:
        Output matrix [M, N] float16
    """
    if fast_dispatch_available():
        try:
            ctx = get_fast_context()
            return ctx.gemm_trellis(
                A, packed_indices, scales, grid, su, sv,
                K, N, bits, group_size
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"C++ dispatch failed, falling back to PyObjC: {e}"
            )

    return dispatch_gemm_trellis_packed(
        lib, A, packed_indices, scales, grid, su, sv,
        K, N, bits, group_size
    )


def dispatch_gemm_trellis_decode_auto(
    lib: MetalKernelLibrary,
    A: torch.Tensor,
    packed_indices: torch.Tensor,
    scales: torch.Tensor,
    grid: torch.Tensor,
    su: torch.Tensor,
    sv: torch.Tensor,
    K: int,
    N: int,
    bits: int,
    group_size: int = 32,
) -> torch.Tensor:
    """Dispatch decode GEMM with automatic C++/PyObjC selection.

    Uses C++ extension fast path when available, falls back to PyObjC.

    Args:
        lib: MetalKernelLibrary with gemm_trellis compiled
        A: Input activations [M, K] float16, MPS tensor where M <= 16
        packed_indices: Packed trellis indices [tiles_k, tiles_n, packed_bytes] uint8
        scales: Per-group scales [n_groups, N] float32
        grid: Codebook grid [n_levels] float32
        su: Row signs [K] float32
        sv: Column signs [N] float32
        K: Inner dimension
        N: Output columns
        bits: Quantization bit width (2, 3, or 4)
        group_size: Quantization group size (default 32)

    Returns:
        Output matrix [M, N] float16
    """
    if fast_dispatch_available():
        try:
            ctx = get_fast_context()
            return ctx.gemm_trellis_decode(
                A, packed_indices, scales, grid, su, sv,
                K, N, bits, group_size
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"C++ decode dispatch failed, falling back to PyObjC: {e}"
            )

    return dispatch_gemm_trellis_decode(
        lib, A, packed_indices, scales, grid, su, sv,
        K, N, bits, group_size
    )


def dispatch_fused_qkv_trellis(
    lib: MetalKernelLibrary,
    A: torch.Tensor,
    q_proj: Any,  # TrellisLinear
    k_proj: Any,  # TrellisLinear
    v_proj: Any,  # TrellisLinear
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Q/K/V projections in a single kernel launch.

    Computes all three attention projections simultaneously:
        Q = A @ Wq
        K = A @ Wk
        V = A @ Wv

    Saves 2 kernel launches and loads A only once (3x memory bandwidth reduction).

    Args:
        lib: MetalKernelLibrary with fused_qkv_trellis compiled
        A: Input activations [M, K] float16, MPS tensor
        q_proj: TrellisLinear for Q projection
        k_proj: TrellisLinear for K projection
        v_proj: TrellisLinear for V projection

    Returns:
        Tuple of (Q, K, V) output tensors, each [M, N_*] float16
    """
    require_mps()

    device = lib.device
    M = A.shape[0]
    K = q_proj.in_features  # All projections share input dim

    # Verify consistent input dimension
    assert k_proj.in_features == K, "K proj input dim mismatch"
    if v_proj is not None:
        assert v_proj.in_features == K, "V proj input dim mismatch"

    Nq = q_proj.out_features
    Nk = k_proj.out_features
    Nv = v_proj.out_features if v_proj is not None else 0

    # All projections must use same bit width
    bits = q_proj.bits
    assert k_proj.bits == bits, "K proj bit width mismatch"
    if v_proj is not None:
        assert v_proj.bits == bits, "V proj bit width mismatch"

    n_levels = q_proj.grid.shape[0]

    # Infer group_size from scales shape
    n_groups = q_proj.scales.shape[0]
    group_size = (K + n_groups - 1) // n_groups

    # Ensure proper types and contiguity
    A = A.contiguous()

    # Allocate outputs
    out_q = torch.zeros(M, Nq, dtype=torch.float16, device="mps")
    out_k = torch.zeros(M, Nk, dtype=torch.float16, device="mps")
    out_v = torch.zeros(M, Nv, dtype=torch.float16,
                        device="mps") if v_proj is not None else None

    # Create Metal buffers
    A_buf = mps_tensor_to_metal_buffer(A, device)

    # Q projection buffers
    Wq_buf = mps_tensor_to_metal_buffer(
        q_proj.packed_indices.contiguous(), device)
    scales_q_buf = mps_tensor_to_metal_buffer(
        q_proj.scales.float().contiguous(), device)
    su_q_buf = mps_tensor_to_metal_buffer(
        q_proj.su.float().contiguous(), device)
    sv_q_buf = mps_tensor_to_metal_buffer(
        q_proj.sv.float().contiguous(), device)

    # K projection buffers
    Wk_buf = mps_tensor_to_metal_buffer(
        k_proj.packed_indices.contiguous(), device)
    scales_k_buf = mps_tensor_to_metal_buffer(
        k_proj.scales.float().contiguous(), device)
    su_k_buf = mps_tensor_to_metal_buffer(
        k_proj.su.float().contiguous(), device)
    sv_k_buf = mps_tensor_to_metal_buffer(
        k_proj.sv.float().contiguous(), device)

    # V projection buffers (only if v_proj provided)
    if v_proj is not None:
        Wv_buf = mps_tensor_to_metal_buffer(
            v_proj.packed_indices.contiguous(), device)
        scales_v_buf = mps_tensor_to_metal_buffer(
            v_proj.scales.float().contiguous(), device)
        su_v_buf = mps_tensor_to_metal_buffer(
            v_proj.su.float().contiguous(), device)
        sv_v_buf = mps_tensor_to_metal_buffer(
            v_proj.sv.float().contiguous(), device)
    else:
        # Create dummy empty buffers for V when not used
        dummy_data = np.zeros(1, dtype=np.float32)
        Wv_buf = device.newBufferWithBytes_length_options_(
            dummy_data.tobytes(), dummy_data.nbytes, Metal.MTLResourceStorageModeShared
        )
        scales_v_buf = Wv_buf
        su_v_buf = Wv_buf
        sv_v_buf = Wv_buf

    # Shared grid (all use same codebook)
    grid_buf = mps_tensor_to_metal_buffer(
        q_proj.grid.float().contiguous(), device)

    # Output buffers
    out_q_buf = mps_tensor_to_metal_buffer(out_q, device, copy_back=True)
    out_k_buf = mps_tensor_to_metal_buffer(out_k, device, copy_back=True)
    if out_v is not None:
        out_v_buf = mps_tensor_to_metal_buffer(out_v, device, copy_back=True)
    else:
        # Dummy buffer for V output when not used
        out_v_buf = Wv_buf  # Reuse dummy buffer

    # Create dimension buffers
    def make_uint_buffer(val: int) -> Any:
        data = np.array([val], dtype=np.uint32)
        return device.newBufferWithBytes_length_options_(
            data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared
        )

    M_buf = make_uint_buffer(M)
    K_buf = make_uint_buffer(K)
    Nq_buf = make_uint_buffer(Nq)
    Nk_buf = make_uint_buffer(Nk)
    Nv_buf = make_uint_buffer(Nv)
    bits_buf = make_uint_buffer(bits)
    n_levels_buf = make_uint_buffer(n_levels)
    group_size_buf = make_uint_buffer(group_size)

    # Choose kernel based on M (decode vs prefill)
    if M <= 16:
        kernel_name = "fused_qkv_trellis_decode"
        TILE_M = 16
        TILE_N = 64
    else:
        kernel_name = "fused_qkv_trellis"
        TILE_M = 32
        TILE_N = 32

    # Grid covers max(Nq, Nk, Nv) columns
    max_N = max(Nq, Nk, Nv)
    grid_x = (max_N + TILE_N - 1) // TILE_N
    grid_y = (M + TILE_M - 1) // TILE_M
    threads_per_tg = 128  # 4 simdgroups * 32 threads

    dispatch_kernel(
        lib,
        function_name=kernel_name,
        grid=(grid_x, grid_y, 1),
        threadgroup=(threads_per_tg, 1, 1),
        buffers=[
            # Input
            A_buf,
            # Q projection
            Wq_buf,
            scales_q_buf,
            su_q_buf,
            sv_q_buf,
            # K projection
            Wk_buf,
            scales_k_buf,
            su_k_buf,
            sv_k_buf,
            # V projection
            Wv_buf,
            scales_v_buf,
            su_v_buf,
            sv_v_buf,
            # Shared grid
            grid_buf,
            # Outputs
            out_q_buf,
            out_k_buf,
            out_v_buf,
            # Dimensions
            M_buf,
            K_buf,
            Nq_buf,
            Nk_buf,
            Nv_buf,
            bits_buf,
            n_levels_buf,
            group_size_buf,
        ],
        wait=True,
    )

    return out_q, out_k, out_v


def dequantize_trellis_weight(
    weight: TrellisWeight,
    lib: MetalKernelLibrary | None = None,
    use_metal: bool = True,
) -> torch.Tensor:
    """Dequantize a TrellisWeight to FP16 tensor.

    High-level API that handles device placement and codebook generation.

    Args:
        weight: TrellisWeight from TrellisModelLoader
        lib: Optional MetalKernelLibrary (created if not provided)
        use_metal: Use Metal GPU acceleration (default True)

    Returns:
        Dequantized weight tensor [K, N] float16
    """
    K, N = weight.original_shape

    # Generate codebook grid for this bit width
    codebook = TrellisCodebook(bits=weight.bits)
    grid = torch.from_numpy(codebook.get_grid()).float()

    if use_metal and HAS_METAL and HAS_MPS:
        # Get or create library
        if lib is None:
            lib = MetalKernelLibrary.from_source_dir()

        # Move to MPS
        indices_mps = weight.indices.to("mps")
        scales_mps = weight.scales.to("mps")
        grid_mps = grid.to("mps")
        su_mps = weight.su.to("mps")
        sv_mps = weight.sv.to("mps")

        # Infer group_size from scales shape
        n_groups = weight.scales.shape[0]
        group_size = (K + n_groups - 1) // n_groups

        # Dispatch fused kernel
        output = dispatch_trellis_dequant_fused(
            lib,
            indices_mps,
            scales_mps,
            grid_mps,
            su_mps,
            sv_mps,
            K,
            N,
            group_size,
        )
        return output

    # CPU fallback
    return _dequantize_trellis_cpu(weight, grid)


def _dequantize_trellis_cpu(
    weight: TrellisWeight,
    grid: torch.Tensor,
) -> torch.Tensor:
    """CPU fallback for trellis dequantization.

    Args:
        weight: TrellisWeight to dequantize
        grid: Codebook grid [n_levels] float32

    Returns:
        Dequantized weight tensor [K, N] float16
    """
    K, N = weight.original_shape
    n_groups = weight.scales.shape[0]
    # Groups are along input dimension (N), scales are per output column (K)
    group_size = (N + n_groups - 1) // n_groups

    # Tile dimensions
    TILE_DIM = 16
    tiles_k = (K + TILE_DIM - 1) // TILE_DIM
    tiles_n = (N + TILE_DIM - 1) // TILE_DIM

    # Allocate output as numpy, convert to torch at end
    import numpy as np

    output = np.zeros((K, N), dtype=np.float32)

    # Dequantize each position
    indices = weight.indices.numpy()
    scales = weight.scales.numpy()
    grid_np = grid.numpy()
    su = weight.su.numpy()
    sv = weight.sv.numpy()

    for k in range(K):
        for n in range(N):
            tile_k = k // TILE_DIM
            tile_n = n // TILE_DIM
            local_k = k % TILE_DIM
            local_n = n % TILE_DIM

            tile_offset = tile_k * tiles_n + tile_n
            local_offset = local_k * TILE_DIM + local_n

            idx = indices[tile_k, tile_n, local_offset]
            idx = max(0, min(idx, len(grid_np) - 1))

            # Group index along N (input), scale per K (output)
            group_idx = n // group_size
            scale = scales[group_idx, k]

            dequant_val = grid_np[idx] * scale
            # su is per-input (N), sv is per-output (K)
            dequant_val *= su[n] * sv[k]

            output[k, n] = dequant_val

    return torch.from_numpy(output).half()


def dispatch_gemm_trellis_auto(
    lib: MetalKernelLibrary,
    A: torch.Tensor,
    packed_indices: torch.Tensor,
    scales: torch.Tensor,
    grid: torch.Tensor,
    su: torch.Tensor,
    sv: torch.Tensor,
    K: int,
    N: int,
    bits: int,
    group_size: int = 32,
) -> torch.Tensor:
    """Dispatch GEMM with automatic C++/PyObjC selection.

    Uses C++ extension (~5-15μs) when available, falls back to
    PyObjC (~80-150μs) when not.
    """
    if fast_dispatch_available():
        try:
            ctx = get_fast_context()
            return ctx.gemm_trellis_packed(
                A, packed_indices, scales, grid, su, sv,
                K, N, bits, group_size
            )
        except Exception as e:
            _logger.warning(
                f"C++ dispatch failed, falling back to PyObjC: {e}"
            )

    # Fallback to PyObjC path
    return dispatch_gemm_trellis_packed(
        lib, A, packed_indices, scales, grid, su, sv,
        K, N, bits, group_size
    )


def dispatch_gemm_trellis_decode_auto(
    lib: MetalKernelLibrary,
    A: torch.Tensor,
    packed_indices: torch.Tensor,
    scales: torch.Tensor,
    grid: torch.Tensor,
    su: torch.Tensor,
    sv: torch.Tensor,
    K: int,
    N: int,
    bits: int,
    group_size: int = 32,
) -> torch.Tensor:
    """Dispatch decode GEMM with automatic C++/PyObjC selection.

    Optimized for small batch sizes (M <= 16).
    Uses C++ extension (~5-15μs) when available, falls back to
    PyObjC (~80-150μs) when not.
    """
    if fast_dispatch_available():
        try:
            ctx = get_fast_context()
            return ctx.gemm_trellis_decode(
                A, packed_indices, scales, grid, su, sv,
                K, N, bits, group_size
            )
        except Exception as e:
            _logger.warning(
                f"C++ decode dispatch failed, falling back to PyObjC: {e}"
            )

    # Fallback to PyObjC path
    return dispatch_gemm_trellis_decode(
        lib, A, packed_indices, scales, grid, su, sv,
        K, N, bits, group_size
    )
