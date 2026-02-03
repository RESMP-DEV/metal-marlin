"""LayerNorm and RMSNorm operations using Metal/MPS backend.

This module provides normalized linear layer operations optimized for Apple Silicon.
Uses PyTorch MPS for tensor operations with Metal kernel acceleration.
"""

from __future__ import annotations

import numpy as np
import torch

from ._compat import Metal
from .metal_dispatch import (
    dispatch_kernel,
    get_default_library,
    mps_tensor_to_metal_buffer,
    require_metal,
    require_mps,
)


def rmsnorm_metal(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply RMSNorm using Metal kernel.

    RMSNorm is defined as: output = x / sqrt(mean(x^2) + eps) * weight

    Args:
        x: Input tensor of shape [..., hidden_size] on MPS device
        weight: Scale tensor of shape [hidden_size] on MPS device
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of same shape as input

    Example:
        >>> x = torch.randn(32, 128, 4096, device="mps", dtype=torch.float16)
        >>> weight = torch.randn(4096, device="mps", dtype=torch.float16)
        >>> out = rmsnorm_metal(x, weight, eps=1e-6)
    """
    require_metal()
    require_mps()

    # Handle shapes
    orig_shape = x.shape
    if x.dim() == 3:
        batch, seq, hidden = x.shape
        x = x.view(batch * seq, hidden)
    else:
        hidden = x.shape[-1]

    # Ensure contiguous and correct dtype
    x = x.contiguous().half()
    weight = weight.contiguous().half()

    # Allocate output
    output = torch.empty_like(x)

    # Get Metal library and device
    lib = get_default_library()
    device = lib.device

    # Convert tensors to Metal buffers (zero-copy)
    x_buf = mps_tensor_to_metal_buffer(x, device)
    weight_buf = mps_tensor_to_metal_buffer(weight, device)
    output_buf = mps_tensor_to_metal_buffer(output, device, copy_back=True)

    # Create parameter buffers
    num_tokens = x.shape[0]
    hidden_dim = hidden

    num_tokens_buf = device.newBufferWithBytes_length_options_(
        np.array([num_tokens], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    hidden_dim_buf = device.newBufferWithBytes_length_options_(
        np.array([hidden_dim], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    eps_buf = device.newBufferWithBytes_length_options_(
        np.array([eps], dtype=np.float32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )

    # Compute grid dimensions - one threadgroup per token
    # Kernel assumes 256 threads (8 simdgroups) per threadgroup
    grid = (num_tokens, 1, 1)
    threadgroup = (256, 1, 1)

    # Select kernel based on hidden dimension
    # Multi-pass version is more efficient for large hidden dims (>= 8192)
    kernel_name = "rmsnorm_multipass" if hidden_dim >= 8192 else "rmsnorm"

    # Dispatch kernel
    dispatch_kernel(
        lib,
        function_name=kernel_name,
        grid=grid,
        threadgroup=threadgroup,
        buffers=[x_buf, weight_buf, output_buf, num_tokens_buf, hidden_dim_buf, eps_buf],
        wait=True,
    )

    return output.view(orig_shape)


def layernorm_metal(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply LayerNorm using Metal kernel.

    LayerNorm is defined as: output = (x - mean) / sqrt(var + eps) * weight + bias

    Args:
        x: Input tensor of shape [..., hidden_size] on MPS device
        weight: Scale tensor of shape [hidden_size] on MPS device
        bias: Bias tensor of shape [hidden_size] on MPS device, or None
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of same shape as input

    Example:
        >>> x = torch.randn(32, 4096, device="mps", dtype=torch.float16)
        >>> weight = torch.randn(4096, device="mps", dtype=torch.float16)
        >>> bias = torch.randn(4096, device="mps", dtype=torch.float16)
        >>> out = layernorm_metal(x, weight, bias, eps=1e-5)
    """
    require_metal()
    require_mps()

    # Handle shapes
    orig_shape = x.shape
    if x.dim() == 3:
        batch, seq, hidden = x.shape
        x = x.view(batch * seq, hidden)
    else:
        hidden = x.shape[-1]

    # Ensure contiguous and correct dtype
    x = x.contiguous().half()
    weight = weight.contiguous().half()

    # Allocate output
    output = torch.empty_like(x)

    # Get Metal library and device
    lib = get_default_library()
    device = lib.device

    # Convert tensors to Metal buffers (zero-copy)
    x_buf = mps_tensor_to_metal_buffer(x, device)
    weight_buf = mps_tensor_to_metal_buffer(weight, device)
    output_buf = mps_tensor_to_metal_buffer(output, device, copy_back=True)

    # Handle bias (kernel expects a buffer, create dummy if None)
    if bias is not None:
        bias = bias.contiguous().half()
        bias_buf = mps_tensor_to_metal_buffer(bias, device)
    else:
        # Create a dummy buffer with zeros
        bias_dummy = torch.zeros(hidden, dtype=torch.float16, device=x.device)
        bias_buf = mps_tensor_to_metal_buffer(bias_dummy, device)

    # Create parameter buffers
    num_tokens = x.shape[0]
    hidden_dim = hidden

    num_tokens_buf = device.newBufferWithBytes_length_options_(
        np.array([num_tokens], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    hidden_dim_buf = device.newBufferWithBytes_length_options_(
        np.array([hidden_dim], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    eps_buf = device.newBufferWithBytes_length_options_(
        np.array([eps], dtype=np.float32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )

    # Compute grid dimensions - one threadgroup per token
    grid = (num_tokens, 1, 1)
    threadgroup = (256, 1, 1)

    # Select kernel based on hidden dimension
    # Multi-pass version is more efficient for large hidden dims (>= 8192)
    kernel_name = "layernorm_multipass" if hidden_dim >= 8192 else "layernorm"

    # Dispatch kernel
    dispatch_kernel(
        lib,
        function_name=kernel_name,
        grid=grid,
        threadgroup=threadgroup,
        buffers=[x_buf, weight_buf, bias_buf, output_buf, num_tokens_buf, hidden_dim_buf, eps_buf],
        wait=True,
    )

    return output.view(orig_shape)


def rmsnorm_fused_residual_metal(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual add + RMSNorm.

    Fused kernel that performs:
        1. residual_out = x + residual
        2. normed = RMSNorm(residual_out)

    Args:
        x: Input tensor of shape [..., hidden_size] on MPS device
        residual: Residual tensor of shape [..., hidden_size] on MPS device
        weight: Scale tensor of shape [hidden_size] on MPS device
        eps: Small constant for numerical stability

    Returns:
        Tuple of (normed_output, residual_output) where:
        - normed_output: RMSNorm result of same shape as input
        - residual_output: x + residual (for next layer)

    Example:
        >>> x = torch.randn(32, 128, 4096, device="mps", dtype=torch.float16)
        >>> residual = torch.randn(32, 128, 4096, device="mps", dtype=torch.float16)
        >>> weight = torch.randn(4096, device="mps", dtype=torch.float16)
        >>> out, res = rmsnorm_fused_residual_metal(x, residual, weight, eps=1e-6)
    """
    require_metal()
    require_mps()

    # Handle shapes
    orig_shape = x.shape
    if x.dim() == 3:
        batch, seq, hidden = x.shape
        x = x.view(batch * seq, hidden)
        residual = residual.view(batch * seq, hidden)
    else:
        hidden = x.shape[-1]

    # Ensure contiguous and correct dtype
    x = x.contiguous().half()
    residual = residual.contiguous().half()
    weight = weight.contiguous().half()

    # Allocate outputs
    normed_out = torch.empty_like(x)
    residual_out = torch.empty_like(x)

    # Get Metal library and device
    lib = get_default_library()
    device = lib.device

    # Convert tensors to Metal buffers (zero-copy)
    x_buf = mps_tensor_to_metal_buffer(x, device)
    residual_buf = mps_tensor_to_metal_buffer(residual, device)
    weight_buf = mps_tensor_to_metal_buffer(weight, device)
    normed_out_buf = mps_tensor_to_metal_buffer(normed_out, device, copy_back=True)
    residual_out_buf = mps_tensor_to_metal_buffer(residual_out, device, copy_back=True)

    # Create parameter buffers
    num_tokens = x.shape[0]
    hidden_dim = hidden

    num_tokens_buf = device.newBufferWithBytes_length_options_(
        np.array([num_tokens], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    hidden_dim_buf = device.newBufferWithBytes_length_options_(
        np.array([hidden_dim], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    eps_buf = device.newBufferWithBytes_length_options_(
        np.array([eps], dtype=np.float32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )

    # Compute grid dimensions - one threadgroup per token
    grid = (num_tokens, 1, 1)
    threadgroup = (256, 1, 1)

    # Select kernel based on hidden dimension
    # Multi-pass version is more efficient for large hidden dims (>= 8192)
    kernel_name = "rmsnorm_fused_residual_multipass" if hidden_dim >= 8192 else "rmsnorm_fused_residual"

    # Dispatch kernel
    dispatch_kernel(
        lib,
        function_name=kernel_name,
        grid=grid,
        threadgroup=threadgroup,
        buffers=[
            x_buf,
            residual_buf,
            weight_buf,
            normed_out_buf,
            residual_out_buf,
            num_tokens_buf,
            hidden_dim_buf,
            eps_buf,
        ],
        wait=True,
    )

    return normed_out.view(orig_shape), residual_out.view(orig_shape)
