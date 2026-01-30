"""INT8 GEMM operations using Metal Marlin kernels.

Provides high-level interface for INT8 quantized matrix multiplication
using Metal-optimized kernels for Apple Silicon.
"""

from __future__ import annotations

from typing import Any

import torch

from ..metal_dispatch import HAS_METAL, HAS_MPS, get_default_library


def load_metal_kernel(filename: str, function_name: str) -> Any:
    """Load a Metal kernel from file.

    Args:
        filename: Metal source file name
        function_name: Kernel function name

    Returns:
        Metal kernel pipeline
    """
    lib = get_default_library()
    return lib.get_pipeline(function_name, filename.replace(".metal", ""))


class GemmInt8:
    """INT8 GEMM with FP16 activations (W8A16)."""

    def __init__(self, device: str = "mps"):
        """Initialize INT8 GEMM operation.

        Args:
            device: Device to use (default: "mps")
        """
        if not HAS_METAL or not HAS_MPS:
            raise ImportError(
                "INT8 GEMM requires Metal and MPS support. "
                "Install PyObjC: pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders"
            )

        self.kernel = load_metal_kernel("gemm_int8.metal", "gemm_int8_tiled")
        self.device = device

    def forward(
        self,
        activations: torch.Tensor,  # [M, K] FP16
        weights: torch.Tensor,  # [K//4, N] uint32 packed
        scales: torch.Tensor,  # [K//group_size, N]
        zeros: torch.Tensor | None = None,
        group_size: int = 32,
    ) -> torch.Tensor:
        """Perform INT8 GEMM operation.

        Args:
            activations: Input activations [M, K] in FP16
            weights: Packed INT8 weights [K//4, N] as uint32
            scales: Per-group scales [K//group_size, N]
            zeros: Optional zero points [K//group_size, N]
            group_size: Quantization group size

        Returns:
            Output tensor [M, N] with FP16 activations
        """
        import numpy as np

        from ..metal_dispatch import (
            THREADS_PER_TG,
            _private_buffer_from_bytes,
            _private_buffer_from_tensor,
            dispatch_kernel,
            mps_tensor_to_metal_buffer,
        )

        # Ensure inputs are on MPS device
        if not activations.is_mps:
            activations = activations.to("mps")
        if not weights.is_mps:
            weights = weights.to("mps")
        if not scales.is_mps:
            scales = scales.to("mps")
        if zeros is not None and not zeros.is_mps:
            zeros = zeros.to("mps")

        # Get dimensions
        M, K = activations.shape
        K_packed, N = weights.shape

        # Verify dimensions match
        if K_packed * 4 != K:
            raise ValueError(
                f"Packed weights K dimension {K_packed * 4} doesn't match activations K {K}"
            )

        # Create output tensor
        output = torch.empty(M, N, dtype=torch.float16, device="mps")

        # Get Metal library
        lib = get_default_library()
        device = lib.device

        # Convert tensors to Metal buffers
        A_half = activations.half().contiguous()
        A_buf = _private_buffer_from_tensor(A_half, lib, device, cache=False)
        B_packed_contig = weights.contiguous()
        B_buf = _private_buffer_from_tensor(B_packed_contig, lib, device, cache=True)
        scales_half = scales if scales.dtype == torch.float16 else scales.half()
        scales_half = scales_half.contiguous()
        S_buf = _private_buffer_from_tensor(scales_half, lib, device, cache=True)
        C_buf = mps_tensor_to_metal_buffer(output, device, copy_back=True)

        # Create param buffers
        M_buf = _private_buffer_from_bytes(lib, device, np.array([M], dtype=np.uint32).tobytes())
        N_buf = _private_buffer_from_bytes(lib, device, np.array([N], dtype=np.uint32).tobytes())
        K_buf = _private_buffer_from_bytes(lib, device, np.array([K], dtype=np.uint32).tobytes())
        gs_buf = _private_buffer_from_bytes(
            lib, device, np.array([group_size], dtype=np.uint32).tobytes()
        )

        # Compute grid dimensions based on INT8 kernel tile sizes
        # From gemm_int8.metal: TILE_M=128, TILE_N=128
        INT8_TILE_M = 128
        INT8_TILE_N = 128
        grid_m = (M + INT8_TILE_M - 1) // INT8_TILE_M
        grid_n = (N + INT8_TILE_N - 1) // INT8_TILE_N

        # Prepare buffers list
        buffers = [A_buf, B_buf, S_buf, C_buf, M_buf, N_buf, K_buf, gs_buf]

        # Add zeros buffer if provided
        if zeros is not None:
            zeros_half = zeros if zeros.dtype == torch.float16 else zeros.half()
            zeros_half = zeros_half.contiguous()
            Z_buf = _private_buffer_from_tensor(zeros_half, lib, device, cache=True)
            buffers.append(Z_buf)

        # The kernel only supports symmetric quantization (no zeros)
        # If zeros are provided, we ignore them for now
        # TODO: Add asymmetric kernel variant
        kernel_name = "gemm_int8_tiled"

        # Dispatch the kernel
        dispatch_kernel(
            lib,
            function_name=kernel_name,
            grid=(grid_n, grid_m, 1),
            threadgroup=(THREADS_PER_TG, 1, 1),
            buffers=buffers,
            wait=True,
        )

        return output


def pack_int8_weights(weights: torch.Tensor) -> torch.Tensor:
    """Pack FP16 weights to INT8 format.

    Args:
        weights: FP16 weights to pack [K, N]

    Returns:
        Packed INT8 weights [K//4, N] as uint32
    """
    # Convert to INT8
    if weights.dtype != torch.float16:
        weights = weights.to(torch.float16)

    # Quantize to INT8
    weights_int8 = torch.clamp(weights * 127, -128, 127).to(torch.int8)

    # Pack 4 INT8 values into uint32
    K, N = weights_int8.shape
    K_packed = (K + 3) // 4
    packed = torch.zeros(K_packed, N, dtype=torch.uint32)

    # Vectorized packing approach
    for i in range(K_packed):
        for j in range(N):
            packed_val = 0
            for k in range(4):
                src_idx = i * 4 + k
                if src_idx < K:
                    # Extract signed INT8 and convert to unsigned representation
                    val = int(weights_int8[src_idx, j]) & 0xFF
                    packed_val |= val << (k * 8)
            packed[i, j] = packed_val

    return packed
