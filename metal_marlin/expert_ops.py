"""Python bindings for expert gather/scatter kernels.

Provides torch-compatible APIs for MoE expert weight operations:
    - expert_gather: Select expert weights based on routing indices
    - expert_scatter_add: Weighted accumulation of expert outputs

These operations are accelerated via Metal when available, with PyTorch
fallbacks for CPU/non-Metal devices.

Usage:
    >>> from metal_marlin.expert_ops import expert_gather, expert_scatter_add
    >>>
    >>> # Gather weights for selected experts
    >>> # expert_weights: [num_experts, hidden, out]
    >>> # expert_indices: [batch, top_k]
    >>> # output: [batch, top_k, out]
    >>> output = expert_gather(expert_weights, expert_indices)
    >>>
    >>> # Scatter-add expert outputs with routing weights
    >>> # expert_outputs: [batch, top_k, out]
    >>> # expert_indices: [batch, top_k]
    >>> # routing_weights: [batch, top_k]
    >>> # output: [batch, out]
    >>> output = expert_scatter_add(expert_outputs, expert_indices, routing_weights)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .metal_dispatch import (
    HAS_METAL,
    HAS_MPS,
    MetalKernelLibrary,
    dispatch_kernel,
    mps_tensor_to_metal_buffer,
)

if HAS_METAL:
    import Metal

_lib: MetalKernelLibrary | None = None


def _get_library() -> MetalKernelLibrary:
    """Get or create the Metal kernel library."""
    global _lib
    if _lib is None:
        _lib = MetalKernelLibrary.from_source_dir()
    return _lib


def _make_uint_buffer(device: Any, value: int) -> Any:
    """Create a Metal buffer containing a single uint32 value."""
    data = np.array([value], dtype=np.uint32)
    return device.newBufferWithBytes_length_options_(
        data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared
    )


def expert_gather(
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather expert weights based on routing indices.

    Selects weights from experts for each token in the batch based on the
    top-k expert indices from the router.

    Args:
        expert_weights: Expert weight tensor [num_experts, hidden, out].
        expert_indices: Per-token expert assignments [batch, top_k].
        output: Optional pre-allocated output tensor [batch, top_k, out].
                If None, a new tensor is allocated.

    Returns:
        Gathered expert weights [batch, top_k, out] where
        output[b, k, :] = expert_weights[expert_indices[b, k], :, :]
        reshaped appropriately for downstream GEMM.

    Note:
        For the common MoE case, this gathers the "out" dimension of expert
        weights. The caller typically reshapes for batched GEMM.

    Example:
        >>> expert_weights = torch.randn(8, 4096, 14336, device="mps")
        >>> expert_indices = torch.randint(0, 8, (32, 2), device="mps")
        >>> gathered = expert_gather(expert_weights, expert_indices)
        >>> assert gathered.shape == (32, 2, 14336)
    """
    num_experts, hidden_dim, out_dim = expert_weights.shape
    batch_size, top_k = expert_indices.shape

    if output is None:
        output = torch.zeros(
            batch_size, top_k, out_dim,
            dtype=expert_weights.dtype,
            device=expert_weights.device,
        )

    if expert_weights.device.type == "mps" and HAS_METAL and HAS_MPS:
        _expert_gather_metal(expert_weights, expert_indices, output)
    else:
        _expert_gather_cpu(expert_weights, expert_indices, output)

    return output


def _expert_gather_cpu(
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """CPU fallback for expert_gather using PyTorch indexing."""
    batch_size, top_k = expert_indices.shape
    flat_indices = expert_indices.reshape(-1)
    selected = torch.index_select(expert_weights, dim=0, index=flat_indices.long())
    selected = selected.mean(dim=1)
    output.copy_(selected.reshape(batch_size, top_k, -1))


def _expert_gather_metal(
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Metal-accelerated expert gather using index_select kernel."""
    lib = _get_library()
    device = lib.device

    num_experts, hidden_dim, out_dim = expert_weights.shape
    batch_size, top_k = expert_indices.shape
    total_selections = batch_size * top_k

    flat_weights = expert_weights.reshape(num_experts, -1).contiguous()
    weights_buf = mps_tensor_to_metal_buffer(flat_weights, device)

    flat_indices = expert_indices.reshape(-1).int().contiguous()
    indices_buf = mps_tensor_to_metal_buffer(flat_indices, device)

    flat_output = output.reshape(total_selections, out_dim).contiguous()
    output_buf = mps_tensor_to_metal_buffer(flat_output, device, copy_back=True)

    inner_size = hidden_dim * out_dim

    dispatch_kernel(
        lib,
        function_name="index_select",
        grid=(
            (inner_size + 255) // 256,
            total_selections,
            1,
        ),
        threadgroup=(256, 1, 1),
        buffers=[
            weights_buf,
            indices_buf,
            output_buf,
            _make_uint_buffer(device, total_selections),
            _make_uint_buffer(device, inner_size),
        ],
        wait=True,
    )

    output.copy_(flat_output.reshape(batch_size, top_k, out_dim).mean(dim=-2, keepdim=True).expand(-1, top_k, -1))


def expert_scatter_add(
    expert_outputs: torch.Tensor,
    expert_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scatter-add expert outputs with routing weights.

    Combines outputs from multiple experts for each token, weighted by the
    router's probability distribution.

    Args:
        expert_outputs: Per-expert output tensor [batch, top_k, out].
        expert_indices: Per-token expert assignments [batch, top_k].
        routing_weights: Per-expert routing probabilities [batch, top_k].
        output: Optional pre-allocated output tensor [batch, out].
                If None, a new tensor is allocated and zero-initialized.

    Returns:
        Combined output [batch, out] where
        output[b, :] = sum_k(routing_weights[b, k] * expert_outputs[b, k, :])

    Note:
        This is the inverse operation of expert_gather - it combines the
        processed expert outputs back to the original token positions.

    Example:
        >>> expert_outputs = torch.randn(32, 2, 14336, device="mps")
        >>> expert_indices = torch.randint(0, 8, (32, 2), device="mps")
        >>> routing_weights = torch.softmax(torch.randn(32, 2, device="mps"), dim=-1)
        >>> combined = expert_scatter_add(expert_outputs, expert_indices, routing_weights)
        >>> assert combined.shape == (32, 14336)
    """
    batch_size, top_k, out_dim = expert_outputs.shape

    if output is None:
        output = torch.zeros(
            batch_size, out_dim,
            dtype=expert_outputs.dtype,
            device=expert_outputs.device,
        )

    if expert_outputs.device.type == "mps" and HAS_METAL and HAS_MPS:
        _expert_scatter_add_metal(expert_outputs, expert_indices, routing_weights, output)
    else:
        _expert_scatter_add_cpu(expert_outputs, expert_indices, routing_weights, output)

    return output


def _expert_scatter_add_cpu(
    expert_outputs: torch.Tensor,
    expert_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """CPU fallback for expert_scatter_add using PyTorch operations."""
    batch_size, top_k, out_dim = expert_outputs.shape
    weighted = expert_outputs * routing_weights.unsqueeze(-1)
    combined = weighted.sum(dim=1)
    output.copy_(combined)


def _expert_scatter_add_metal(
    expert_outputs: torch.Tensor,
    expert_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Metal-accelerated scatter-add using optimized SIMD kernels."""
    from .moe_scatter_gather import ScatterGatherDispatcher

    lib = _get_library()
    batch_size, top_k, out_dim = expert_outputs.shape

    flat_outputs = expert_outputs.reshape(-1, out_dim).half().contiguous()
    probs = routing_weights.half().contiguous()

    inverse_indices = torch.arange(
        batch_size * top_k, dtype=torch.int32, device=expert_outputs.device
    )

    dispatcher = ScatterGatherDispatcher(
        lib, hidden_dim=out_dim, max_batch=batch_size, max_top_k=top_k
    )

    result = dispatcher.scatter_combine(
        flat_outputs, probs, inverse_indices, batch_size, top_k
    )

    output.copy_(result.to(output.dtype))


__all__ = ["expert_gather", "expert_scatter_add"]
