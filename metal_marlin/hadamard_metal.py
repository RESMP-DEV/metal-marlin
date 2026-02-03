"""Metal-accelerated Hadamard transform for activation rotation.

This module provides Walsh-Hadamard transform kernels optimized for Apple Silicon.
Used for QuIP#/HQQ-style activation rotation before quantized GEMM.

The Hadamard transform is self-inverse (H = H^T = H^-1 up to scaling), so it can
be fused as a preprocessing step before quantized matrix multiplication:
    y = (H @ W) @ (H^T @ x) = W @ x  (mathematically equivalent)

Example:
    >>> from metal_marlin.hadamard_metal import hadamard_transform_metal
    >>> import torch
    >>> x = torch.randn(16, 64, device="mps", dtype=torch.float16)
    >>> y = hadamard_transform_metal(x, block_size=64, normalize=True)
    >>> # Apply twice to recover original (H @ H = I for normalized H)
    >>> x_recovered = hadamard_transform_metal(y, block_size=64, normalize=True)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._compat import Metal
from .metal_dispatch import (
    dispatch_kernel,
    get_default_library,
    mps_tensor_to_metal_buffer,
    require_metal,
    require_mps,
)

if TYPE_CHECKING:
    pass


# Compile kernel on first use
_KERNEL_SOURCE: str | None = None


def _get_kernel_source() -> str:
    """Get the Metal kernel source for Hadamard transform."""
    global _KERNEL_SOURCE
    if _KERNEL_SOURCE is None:
        _KERNEL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

inline float butterfly_step_f(float val, uint partner, bool is_upper) {
    float partner_val = simd_shuffle(val, ushort(partner));
    return is_upper ? (partner_val - val) : (val + partner_val);
}

inline float2 butterfly_step2_f(float2 val, uint partner, bool is_upper) {
    float2 partner_val = simd_shuffle(val, ushort(partner));
    return is_upper ? (partner_val - val) : (val + partner_val);
}

inline float4 butterfly_step4_f(float4 val, uint partner, bool is_upper) {
    float4 partner_val = simd_shuffle(val, ushort(partner));
    return is_upper ? (partner_val - val) : (val + partner_val);
}

struct HadamardParams {
    uint N;
    uint normalize;
};

kernel void hadamard_32(
    device const half* input [[buffer(0)]],
    device half* out [[buffer(1)]],
    device const HadamardParams* params [[buffer(2)]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    uint lane_id = thread_index_in_simdgroup;
    uint tg_idx = threadgroup_position_in_grid;
    const uint N = params->N;
    const bool NORMALIZE = params->normalize != 0;

    if (tg_idx >= N) return;

    uint base = tg_idx * 32;
    float val = float(input[base + lane_id]);

    for (uint stage = 0; stage < 5; ++stage) {
        uint stride = 1u << stage;
        uint partner = lane_id ^ stride;
        bool is_upper = (lane_id & stride) != 0;
        val = butterfly_step_f(val, partner, is_upper);
    }

    if (NORMALIZE) {
        val *= 0.1767766953f;  // 1/sqrt(32)
    }

    out[base + lane_id] = half(val);
}

kernel void hadamard_64(
    device const half* input [[buffer(0)]],
    device half* out [[buffer(1)]],
    device const HadamardParams* params [[buffer(2)]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    uint lane_id = thread_index_in_simdgroup;
    uint tg_idx = threadgroup_position_in_grid;
    const uint N = params->N;
    const bool NORMALIZE = params->normalize != 0;

    if (tg_idx >= N) return;

    uint base = tg_idx * 64;
    float2 val;
    val.x = float(input[base + lane_id * 2]);
    val.y = float(input[base + lane_id * 2 + 1]);

    {
        float sum = val.x + val.y;
        float diff = val.x - val.y;
        val.x = sum;
        val.y = diff;
    }

    for (uint stage = 1; stage < 6; ++stage) {
        uint stride = 1u << (stage - 1);
        uint partner = lane_id ^ stride;
        bool is_upper = (lane_id & stride) != 0;
        val = butterfly_step2_f(val, partner, is_upper);
    }

    if (NORMALIZE) {
        val *= 0.125f;  // 1/sqrt(64)
    }

    out[base + lane_id * 2] = half(val.x);
    out[base + lane_id * 2 + 1] = half(val.y);
}

kernel void hadamard_128(
    device const half* input [[buffer(0)]],
    device half* out [[buffer(1)]],
    device const HadamardParams* params [[buffer(2)]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    uint lane_id = thread_index_in_simdgroup;
    uint tg_idx = threadgroup_position_in_grid;
    const uint N = params->N;
    const bool NORMALIZE = params->normalize != 0;

    if (tg_idx >= N) return;

    uint base = tg_idx * 128;
    float4 val;
    val.x = float(input[base + lane_id * 4]);
    val.y = float(input[base + lane_id * 4 + 1]);
    val.z = float(input[base + lane_id * 4 + 2]);
    val.w = float(input[base + lane_id * 4 + 3]);

    {
        float sum0 = val.x + val.y;
        float diff0 = val.x - val.y;
        float sum1 = val.z + val.w;
        float diff1 = val.z - val.w;
        val = float4(sum0, diff0, sum1, diff1);
    }

    {
        float sum0 = val.x + val.z;
        float diff0 = val.x - val.z;
        float sum1 = val.y + val.w;
        float diff1 = val.y - val.w;
        val = float4(sum0, sum1, diff0, diff1);
    }

    for (uint stage = 2; stage < 7; ++stage) {
        uint stride = 1u << (stage - 2);
        uint partner = lane_id ^ stride;
        bool is_upper = (lane_id & stride) != 0;
        val = butterfly_step4_f(val, partner, is_upper);
    }

    if (NORMALIZE) {
        val *= 0.0883883476f;  // 1/sqrt(128)
    }

    out[base + lane_id * 4] = half(val.x);
    out[base + lane_id * 4 + 1] = half(val.y);
    out[base + lane_id * 4 + 2] = half(val.z);
    out[base + lane_id * 4 + 3] = half(val.w);
}
"""
    return _KERNEL_SOURCE


# Cache for compiled kernel library
_kernel_compiled: bool = False


def _ensure_kernel_compiled(lib) -> None:
    """Compile the Hadamard kernel if not already compiled."""
    global _kernel_compiled
    if not _kernel_compiled:
        lib.compile_source("hadamard", _get_kernel_source())
        _kernel_compiled = True


def hadamard_transform_metal(
    x,
    block_size: int = 64,
    normalize: bool = True,
):
    """Apply Walsh-Hadamard transform using Metal GPU acceleration.

    Args:
        x: Input tensor [..., block_size] on MPS device.
        block_size: Size of each Hadamard block. Must be 32, 64, 96, 128, 160, or 192.
        normalize: If True, normalize by 1/sqrt(block_size) after transform.

    Returns:
        Transformed tensor with same shape as input. MPS tensor.

    Raises:
        ImportError: If Metal/MPS is not available.
        ValueError: If block_size is not 32, 64, 96, 128, 160, or 192.
        ValueError: If last dimension of x doesn't equal block_size.

    Example:
        >>> import torch
        >>> x = torch.randn(16, 64, device="mps", dtype=torch.float16)
        >>> y = hadamard_transform_metal(x, block_size=64, normalize=True)
        >>> # H @ H = I for normalized transform
        >>> x_recovered = hadamard_transform_metal(y, block_size=64, normalize=True)
    """
    import torch

    require_metal()
    require_mps()

    if block_size not in (32, 64, 96, 128, 160, 192):
        raise ValueError(f"block_size must be 32, 64, 96, 128, 160, or 192, got {block_size}")
    
    # Non-power-of-2 sizes require precompiled kernels from src/hadamard.metal
    if block_size in (96, 160, 192):
        raise NotImplementedError(
            f"block_size={block_size} requires precompiled library from src/hadamard.metal. "
            "This feature is available in the full C++ build. "
            "For pure Python usage, use power-of-2 sizes: 32, 64, or 128."
        )

    if x.shape[-1] != block_size:
        raise ValueError(
            f"Last dimension of x ({x.shape[-1]}) must equal block_size ({block_size})"
        )

    lib = get_default_library()
    _ensure_kernel_compiled(lib)

    # Flatten to 2D: [N, block_size]
    orig_shape = x.shape
    n = 1
    for d in orig_shape[:-1]:
        n *= d

    x_2d = x.reshape(n, block_size).half().contiguous()
    out = torch.empty_like(x_2d)

    device = lib.device

    x_buf = mps_tensor_to_metal_buffer(x_2d, device)
    out_buf = mps_tensor_to_metal_buffer(out, device, copy_back=True)

    params = np.array([n, 1 if normalize else 0], dtype=np.uint32)
    params_buf = device.newBufferWithBytes_length_options_(
        params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
    )

    kernel_name = f"hadamard_{block_size}"

    dispatch_kernel(
        lib,
        function_name=kernel_name,
        grid=(n, 1, 1),
        threadgroup=(32, 1, 1),  # One simdgroup per vector
        buffers=[x_buf, out_buf, params_buf],
        wait=True,
    )

    return out.reshape(orig_shape)


class HadamardMetal:
    """Metal-accelerated Hadamard transform handler.

    This class provides a convenient interface for applying Hadamard transforms
    using Metal GPU acceleration. It's designed for use in quantization pipelines
    where activation rotation (QuIP#, HQQ) is needed.

    Example:
        >>> from metal_marlin.hadamard_metal import HadamardMetal
        >>> import torch
        >>> handler = HadamardMetal()
        >>> x = torch.randn(16, 64, device="mps", dtype=torch.float16)
        >>> y = handler.transform(x, normalize=True)
    """

    def __init__(self):
        """Initialize the HadamardMetal handler.

        Raises:
            ImportError: If Metal/MPS is not available.
        """
        require_metal()
        require_mps()
        self._lib = get_default_library()
        _ensure_kernel_compiled(self._lib)

    def transform(
        self,
        x,
        block_size: int = 64,
        normalize: bool = True,
    ):
        """Apply Walsh-Hadamard transform to input tensor.

        Args:
            x: Input tensor [..., block_size] on MPS device.
            block_size: Size of each Hadamard block. Must be 32, 64, or 128.
            normalize: If True, normalize by 1/sqrt(block_size).

        Returns:
            Transformed tensor with same shape as input.
        """
        return hadamard_transform_metal(x, block_size, normalize)

    def transform_numpy(
        self,
        x: np.ndarray,
        block_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Apply Walsh-Hadamard transform to numpy array.

        This helper converts numpy arrays to/from MPS tensors for convenience.

        Args:
            x: Input array [..., block_size] as numpy float32.
            block_size: Size of each Hadamard block. Must be 32, 64, or 128.
            normalize: If True, normalize by 1/sqrt(block_size).

        Returns:
            Transformed array as numpy float32.
        """
        import torch

        x_torch = torch.from_numpy(x).to("mps", dtype=torch.float16)
        y_torch = self.transform(x_torch, block_size, normalize)
        return y_torch.cpu().float().numpy()


__all__ = [
    "HadamardMetal",
    "hadamard_transform_metal",
]
