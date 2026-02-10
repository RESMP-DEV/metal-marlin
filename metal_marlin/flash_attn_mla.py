"""
Flash Attention V3 MLA dispatch wrapper for Apple Silicon Metal kernels.

This module provides Python bindings to the optimized Flash Attention V3 Metal
kernels, specifically designed for MLA (Multi-Head Latent Attention) prefill.

Key Features:
    - Wraps `flash_attention_v3_causal_gqa`
    - Optimized for MLA's split K (k_nope + k_rope concatenated)
    - Causal masking support
    - Enhanced register usage for head_dim=64/128

Usage:
    from metal_marlin.flash_attn_mla import flash_attention_v3_mla

    # MLA prefill (concatenated K/V inputs)
    output = flash_attention_v3_mla(Q, K, V, scale=scale, causal=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._compat import HAS_MPS, HAS_PYOBJC_METAL, HAS_TORCH

if HAS_TORCH:
    import torch

if HAS_PYOBJC_METAL:
    import Metal

if TYPE_CHECKING:
    import torch


# Tile dimensions matching the Metal kernel
TILE_Q = 16
TILE_KV = 24  # As defined in flash_attention_v3.metal
NUM_SIMDGROUPS = 4
THREADS_PER_TG = NUM_SIMDGROUPS * 32  # 128


def _require_metal_attention() -> None:
    """Raise if Metal dispatch is not available."""
    if not HAS_TORCH:
        raise RuntimeError(
            "Flash Attention V3 MLA requires PyTorch. Install with: pip install torch")
    if not HAS_MPS:
        raise RuntimeError(
            "Flash Attention V3 MLA requires PyTorch MPS backend (Apple Silicon).")
    if not HAS_PYOBJC_METAL:
        raise RuntimeError(
            "Flash Attention V3 MLA requires PyObjC Metal. Install with:\n"
            "  pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders"
        )


@dataclass
class AttentionParams:
    """Parameters for Flash Attention kernel dispatch."""

    batch: int
    num_heads_q: int
    num_heads_kv: int
    seq_q: int
    seq_k: int
    head_dim: int
    scale: float
    gqa_ratio: int
    is_causal: bool

    def to_buffer(self, device: Any) -> Any:
        """Pack parameters into Metal buffer for kernel dispatch.

        Args:
            device: MTLDevice for buffer allocation

        Returns:
            MTLBuffer containing packed parameters
        """
        # Pack as uint32 array (is_causal as 0/1)
        # Note: scale is reinterpreted as uint32 for the Metal struct
        data = np.array(
            [
                self.batch,
                self.num_heads_q,
                self.num_heads_kv,
                self.seq_q,
                self.seq_k,
                self.head_dim,
                # reinterpret float as uint32
                np.float32(self.scale).view(np.uint32),
                self.gqa_ratio,
                1 if self.is_causal else 0,
            ],
            dtype=np.uint32,
        )
        buffer = device.newBufferWithBytes_length_options_(
            data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared
        )
        return buffer


def _get_kernel_source() -> str:
    """Load Flash Attention V3 Metal kernel source."""
    kernel_path = Path(__file__).parent.parent / \
        "src" / "flash_attention_v3.metal"
    if kernel_path.exists():
        return kernel_path.read_text()
    raise FileNotFoundError(
        f"Flash Attention V3 kernel not found at {kernel_path}")


# ---------------------------------------------------------------------------
# Kernel library singleton
# ---------------------------------------------------------------------------

_kernel_lib: Any = None


def _get_kernel_library() -> Any:
    """Get or create the Flash Attention kernel library."""
    global _kernel_lib
    if _kernel_lib is None:
        from .metal_dispatch import MetalKernelLibrary

        _kernel_lib = MetalKernelLibrary()
        source = _get_kernel_source()
        _kernel_lib.compile_source("flash_attention_v3", source)
    return _kernel_lib


def _mps_tensor_to_metal_buffer(tensor: torch.Tensor, device: Any) -> Any:
    """Get Metal buffer from PyTorch MPS tensor."""
    import ctypes

    if not tensor.is_mps:
        raise ValueError("Tensor must be on MPS device")

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    storage = tensor.untyped_storage()
    ptr = storage.data_ptr()
    size = storage.nbytes()

    # Try zero-copy first
    try:
        buffer = device.newBufferWithBytesNoCopy_length_options_deallocator_(
            ptr, size, Metal.MTLResourceStorageModeShared, None
        )
        if buffer is not None:
            return buffer
    except (TypeError, ValueError):
        pass

    # Fallback to copy
    arr_type = ctypes.c_uint8 * size
    arr = arr_type.from_address(ptr)
    data = bytes(arr)
    buffer = device.newBufferWithBytes_length_options_(
        data, len(data), Metal.MTLResourceStorageModeShared
    )

    if buffer is None:
        raise RuntimeError("Failed to create Metal buffer from tensor")

    return buffer


def _dispatch_attention_kernel(
    lib: Any,
    function_name: str,
    grid: tuple[int, int, int],
    threadgroup: tuple[int, int, int],
    buffers: list[Any],
    wait: bool = True,
) -> None:
    """Dispatch a Flash Attention Metal kernel."""
    pipeline = lib.get_pipeline(
        function_name, library_name="flash_attention_v3")

    if os.getenv("METAL_MARLIN_AUDIT", "0") == "1":
        import os
        print(f"DISPATCH: {function_name} grid={grid} tg={threadgroup}")

    command_buffer = lib.command_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()

    encoder.setComputePipelineState_(pipeline)

    # Bind buffers
    for i, buf in enumerate(buffers):
        encoder.setBuffer_offset_atIndex_(buf, 0, i)

    # Dispatch
    grid_size = Metal.MTLSizeMake(*grid)
    tg_size = Metal.MTLSizeMake(*threadgroup)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)

    encoder.endEncoding()
    command_buffer.commit()

    if wait:
        command_buffer.waitUntilCompleted()


def flash_attention_v3_mla(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Compute Flash Attention V3 for MLA (prefill).

    Wraps `flash_attention_v3_causal_gqa` to handle MLA's split K/V inputs.
    
    Expected Shapes:
        Q: [batch, heads_q, seq_q, head_dim]
        K: [batch, heads_kv, seq_k, head_dim] (concatenated k_nope + k_rope)
        V: [batch, heads_kv, seq_k, head_dim]

    Args:
        Q: Query tensor
        K: Key tensor (concatenated)
        V: Value tensor
        scale: Attention scale factor (default: 1/sqrt(head_dim))
        causal: Apply causal masking

    Returns:
        Output tensor [batch, heads_q, seq_q, head_dim]
    """
    _require_metal_attention()

    batch, num_heads_q, seq_q, head_dim = Q.shape
    _, num_heads_kv, seq_k, _ = K.shape

    if scale is None:
        scale = head_dim**-0.5

    # Auto-detect GQA ratio
    gqa_ratio = num_heads_q // num_heads_kv if num_heads_kv > 0 else 1

    params = AttentionParams(
        batch=batch,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        seq_q=seq_q,
        seq_k=seq_k,
        head_dim=head_dim,
        scale=scale,
        gqa_ratio=gqa_ratio,
        is_causal=causal,
    )

    lib = _get_kernel_library()
    device = lib.device

    # Ensure tensors are FP16 contiguous on MPS
    Q = Q.to(device="mps", dtype=torch.float16).contiguous()
    K = K.to(device="mps", dtype=torch.float16).contiguous()
    V = V.to(device="mps", dtype=torch.float16).contiguous()

    # Allocate output
    output = torch.empty(
        (batch, num_heads_q, seq_q, head_dim), dtype=torch.float16, device="mps"
    )

    # Convert to Metal buffers
    Q_buf = _mps_tensor_to_metal_buffer(Q, device)
    K_buf = _mps_tensor_to_metal_buffer(K, device)
    V_buf = _mps_tensor_to_metal_buffer(V, device)
    O_buf = _mps_tensor_to_metal_buffer(output, device)
    params_buf = params.to_buffer(device)

    # Compute grid for GQA
    # Dispatch: [head_kv, q_tiles, batch]
    q_tiles = (seq_q + TILE_Q - 1) // TILE_Q
    grid = (num_heads_kv, q_tiles, batch)

    # Use the causal GQA kernel
    kernel_name = "flash_attention_v3_causal_gqa" if causal else "flash_attention_v3_gqa"

    _dispatch_attention_kernel(
        lib,
        kernel_name,
        grid=grid,
        threadgroup=(THREADS_PER_TG, 1, 1),
        buffers=[Q_buf, K_buf, V_buf, O_buf, params_buf],
        wait=True,
    )

    return output