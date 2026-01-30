"""Metal dispatch for trellis dequantization kernels.

Provides Python wrappers for the dequant_trellis.metal kernels:
- dispatch_trellis_dequant: Dequantize trellis indices to FP16
- dispatch_sign_flips: Apply su/sv sign corrections
- dispatch_trellis_dequant_fused: Combined dequant + sign flip

Reference: src/dequant_trellis.metal
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ..metal_dispatch import (
    HAS_METAL,
    HAS_MPS,
    MetalKernelLibrary,
    dispatch_kernel,
    mps_tensor_to_metal_buffer,
    require_mps,
)
from ..quantization.trellis_codebook import TrellisCodebook

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
        indices: Trellis indices [tiles_k, tiles_n, 256] int16, MPS tensor
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
    scales = scales.float().contiguous()
    grid = grid.float().contiguous()

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
        indices: Trellis indices [tiles_k, tiles_n, 256] int16, MPS tensor
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
    grid: torch.Tensor,
    su: torch.Tensor,
    sv: torch.Tensor,
    K: int,
    N: int,
    bits: int,
) -> torch.Tensor:
    """Dequantize packed trellis indices to FP16 weights.

    Uses the packed uint8 format directly without unpacking on CPU.

    Args:
        lib: MetalKernelLibrary with dequant_trellis compiled
        packed_indices: Packed trellis indices [tiles_k, tiles_n, packed_bytes] uint8, MPS tensor
        scales: Per-group scales [n_groups, N] float32, MPS tensor
        grid: Codebook grid [n_levels] float32, MPS tensor
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
    n_levels = grid.shape[0]

    # Ensure proper types and contiguity
    packed_indices = packed_indices.contiguous()
    scales = scales.float().contiguous()
    grid = grid.float().contiguous()
    su = su.float().contiguous()
    sv = sv.float().contiguous()

    # Allocate output
    output = torch.zeros(K, N, dtype=torch.float16, device="mps")

    # Create Metal buffers
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

    K_buf = make_uint_buffer(K)
    N_buf = make_uint_buffer(N)
    n_levels_buf = make_uint_buffer(n_levels)
    bits_buf = make_uint_buffer(bits)

    # Dispatch with one thread per output element
    threads_per_tg = 16
    grid_x = (N + threads_per_tg - 1) // threads_per_tg
    grid_y = (K + threads_per_tg - 1) // threads_per_tg

    dispatch_kernel(
        lib,
        function_name="dequant_trellis_packed",
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
        ],
        wait=True,
    )

    return output


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
