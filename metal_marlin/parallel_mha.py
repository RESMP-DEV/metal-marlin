"""
Parallel Multi-Head Attention for Apple Silicon Metal.

This module provides Python bindings to parallel multi-head attention Metal
kernels where each threadgroup independently processes one attention head.

Kernels:
    parallel_multihead_attention         - Base parallel MHA
    parallel_multihead_attention_multirow - Multiple query rows per threadgroup

Key optimizations:
    - Independent threadgroups per attention head (num_heads threadgroups)
    - Each threadgroup processes HEADS_PER_TG query rows in parallel
    - THREADS_PER_HEAD threads per head (32 = 1 simdgroup)
    - Double-buffered K/V tile streaming
    - Online softmax for numerical stability

Thread configuration:
    - X threads per head (THREADS_PER_HEAD = 32)
    - Y heads in parallel per threadgroup (HEADS_PER_TG = 4)
    - Threadgroups = num_heads

This approach maximizes parallelism across attention heads while
maintaining efficiency within each head through simdgroup operations.

Usage:
    from metal_marlin.parallel_mha import parallel_multihead_attention

    # Automatic kernel selection
    output = parallel_multihead_attention(Q, K, V, scale=scale, causal=True)
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

# Thread configuration matching Metal kernel
THREADS_PER_HEAD = 32
HEADS_PER_TG = 4
THREADS_PER_TG = THREADS_PER_HEAD * HEADS_PER_TG
TILE_KV_PARALLEL = 64
HEAD_DIM_MAX_PARALLEL = 128


def _require_metal_mha() -> None:
    """Raise if Metal MHA dispatch is not available."""
    if not HAS_TORCH:
        raise RuntimeError("Parallel MHA requires PyTorch. Install with: pip install torch")
    if not HAS_MPS:
        raise RuntimeError("Parallel MHA requires PyTorch MPS backend (Apple Silicon).")
    if not HAS_PYOBJC_METAL:
        raise RuntimeError(
            "Parallel MHA requires PyObjC Metal. Install with:\n"
            "  pip install pyobjc-framework-Metal"
        )


@dataclass
class ParallelMHAParams:
    """Parameters for Parallel MHA kernel dispatch."""

    batch: int
    num_heads: int
    seq_q: int
    seq_k: int
    head_dim: int
    scale: float
    causal: int

    def to_buffer(self, device: Any) -> Any:
        """Pack parameters into Metal buffer for kernel dispatch."""
        return np.array(
            [
                self.batch,
                self.num_heads,
                self.seq_q,
                self.seq_k,
                self.head_dim,
                self.scale,
                self.causal,
            ],
            dtype=np.float32,
        ).astype(np.float32)


class ParallelMHADispatcher:
    """Dispatcher for parallel multi-head attention kernels."""

    def __init__(self) -> None:
        """Initialize Metal device and compile kernels."""
        _require_metal_mha()

        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("Failed to create Metal device")

        self.command_queue = self.device.newCommandQueue()
        self._kernel_lib = self._compile_kernels()

    def _compile_kernels(self) -> Metal.MTLComputePipelineState:
        """Compile parallel MHA Metal shaders."""
        kernel_path = Path(__file__).parent.parent / "src" / "parallel_multihead_attention.metal"

        if not kernel_path.exists():
            raise FileNotFoundError(f"Metal kernel not found: {kernel_path}")

        with open(kernel_path) as f:
            source = f.read()

        compile_options = Metal.MTLCompileOptions.alloc().init()
        compile_options.languageVersion = Metal.MTLLanguageVersion2_4

        library = self.device.newLibraryWithSource_options_error_(source, compile_options, None)

        if library is None:
            raise RuntimeError("Failed to compile Metal kernels")

        return library

    def _get_pipeline(self, function_name: str) -> Metal.MTLComputePipelineState:
        """Get compiled Metal pipeline for kernel function."""
        if self._kernel_lib is None:
            raise RuntimeError("Kernel library not compiled")

        function = self._kernel_lib.newFunctionWithName_(function_name)
        if function is None:
            raise ValueError(f"Kernel function not found: {function_name}")

        pipeline_state = self.device.newComputePipelineStateWithFunction_error_(function, None)

        if pipeline_state is None:
            raise RuntimeError(f"Failed to create pipeline state: {function_name}")

        return pipeline_state

    def _dispatch(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        O: torch.Tensor,
        params: ParallelMHAParams,
        function_name: str,
    ) -> None:
        """Dispatch Metal kernel for parallel MHA computation."""
        pipeline = self._get_pipeline(function_name)

        command_buffer = self.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)

        # Set buffers
        encoder.setBuffer_offset_atIndex_(Q._mps_handle().__ptr__, 0, 0)
        encoder.setBuffer_offset_atIndex_(K._mps_handle().__ptr__, 0, 1)
        encoder.setBuffer_offset_atIndex_(V._mps_handle().__ptr__, 0, 2)
        encoder.setBuffer_offset_atIndex_(O._mps_handle().__ptr__, 0, 3)

        # Set parameters buffer
        params_array = params.to_buffer(self.device)
        params_buffer = self.device.newBufferWithBytes_options_(
            params_array.tobytes(),
            Metal.MTLResourceStorageModeShared,
        )
        encoder.setBuffer_offset_atIndex_(params_buffer, 0, 4)
        encoder.setBuffer_offset_atIndex_(params_buffer, 64, 5)
        encoder.setBuffer_offset_atIndex_(params_buffer, 68, 6)
        encoder.setBuffer_offset_atIndex_(params_buffer, 72, 7)
        encoder.setBuffer_offset_atIndex_(params_buffer, 76, 8)
        encoder.setBuffer_offset_atIndex_(params_buffer, 80, 9)
        encoder.setBuffer_offset_atIndex_(params_buffer, 84, 10)

        # Dispatch threadgroups = num_heads, threads_per_threadgroup = THREADS_PER_TG
        threadgroups_per_grid = Metal.MTLSize(params.num_heads, 1, 1)
        threads_per_threadgroup = Metal.MTLSize(THREADS_PER_TG, 1, 1)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            threadgroups_per_grid, threads_per_threadgroup
        )

        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        scale: float,
        causal: bool = False,
    ) -> torch.Tensor:
        """Compute parallel multi-head attention.

        Args:
            Q: Query tensor [batch, num_heads, seq_q, head_dim]
            K: Key tensor [batch, num_heads, seq_k, head_dim]
            V: Value tensor [batch, num_heads, seq_k, head_dim]
            scale: Scaling factor (typically 1/sqrt(head_dim))
            causal: Apply causal mask for autoregressive models

        Returns:
            Output tensor [batch, num_heads, seq_q, head_dim]
        """
        _require_metal_mha()

        # Validate input shapes
        batch, num_heads, seq_q, head_dim = Q.shape
        _, _, seq_k, _ = K.shape

        if K.shape[0] != batch or K.shape[1] != num_heads:
            raise ValueError(
                f"K shape mismatch: expected [{batch}, {num_heads}, seq_k, head_dim], got {K.shape}"
            )
        if V.shape != K.shape:
            raise ValueError(f"V and K shapes must match: V={V.shape}, K={K.shape}")

        # Ensure tensors are on MPS and in FP16
        Q = Q.to(device="mps", dtype=torch.float16)
        K = K.to(device="mps", dtype=torch.float16)
        V = V.to(device="mps", dtype=torch.float16)

        # Allocate output tensor
        O = torch.empty_like(Q)

        # Prepare parameters
        params = ParallelMHAParams(
            batch=batch,
            num_heads=num_heads,
            seq_q=seq_q,
            seq_k=seq_k,
            head_dim=head_dim,
            scale=scale,
            causal=1 if causal else 0,
        )

        # Select kernel based on configuration
        if seq_q == 1:
            # Single-query case: use multirow variant with minimal work
            function_name = "parallel_multihead_attention_multirow"
        else:
            # Standard prefill case
            function_name = "parallel_multihead_attention"

        # Dispatch kernel
        self._dispatch(Q, K, V, O, params, function_name)

        return O


# Global dispatcher instance
_dispatcher: ParallelMHADispatcher | None = None


def _get_dispatcher() -> ParallelMHADispatcher:
    """Get or create global dispatcher instance."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = ParallelMHADispatcher()
    return _dispatcher


def parallel_multihead_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float,
    causal: bool = False,
) -> torch.Tensor:
    """Compute parallel multi-head attention with Metal kernel.

    This kernel launches num_heads threadgroups, where each threadgroup
    independently processes one attention head. Within each threadgroup,
    HEADS_PER_TG query rows are processed in parallel with THREADS_PER_HEAD
    threads per head.

    Thread configuration:
        - X threads per head: 32 (1 simdgroup)
        - Y heads in parallel per threadgroup: 4
        - Threadgroups = num_heads

    This maximizes parallelism across heads while maintaining efficiency
    through simdgroup reductions within each head.

    Args:
        Q: Query tensor [batch, num_heads, seq_q, head_dim]
        K: Key tensor [batch, num_heads, seq_k, head_dim]
        V: Value tensor [batch, num_heads, seq_k, head_dim]
        scale: Scaling factor (typically 1/sqrt(head_dim))
        causal: Apply causal mask for autoregressive models

    Returns:
        Output tensor [batch, num_heads, seq_q, head_dim]

    Example:
        >>> import torch
        >>> from metal_marlin.parallel_mha import parallel_multihead_attention
        >>> batch, num_heads, seq_q, seq_k, head_dim = 2, 32, 512, 512, 128
        >>> Q = torch.randn(batch, num_heads, seq_q, head_dim, dtype=torch.float16)
        >>> K = torch.randn(batch, num_heads, seq_k, head_dim, dtype=torch.float16)
        >>> V = torch.randn(batch, num_heads, seq_k, head_dim, dtype=torch.float16)
        >>> scale = 1.0 / (head_dim ** 0.5)
        >>> output = parallel_multihead_attention(Q, K, V, scale=scale, causal=True)
    """
    dispatcher = _get_dispatcher()
    return dispatcher.forward(Q, K, V, scale, causal)
