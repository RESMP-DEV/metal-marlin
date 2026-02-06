"""Unified memory buffer utilities for Apple Silicon.

On Apple Silicon, CPU and GPU share the same physical memory.
MPS tensors can be accessed by Metal directly via data_ptr().
"""
import torch
from typing import Any


def mps_tensor_to_metal_buffer_zerocopy(
    tensor: torch.Tensor,
    device: Any,
) -> Any:
    """Create Metal buffer from MPS tensor WITHOUT copying.
    
    Args:
        tensor: MPS tensor (must be on 'mps' device)
        device: Metal device from lib
        
    Returns:
        Metal buffer pointing to same memory as tensor
    """
    if tensor.device.type != 'mps':
        raise ValueError(f"Expected MPS tensor, got {tensor.device}")
    
    # Get raw pointer - this is the unified memory address
    ptr = tensor.data_ptr()
    nbytes = tensor.numel() * tensor.element_size()
    
    # Create Metal buffer with MTLResourceStorageModeShared
    # This tells Metal to use the existing memory, not copy
    return device.newBufferWithBytesNoCopy_length_options_deallocator_(
        ptr, nbytes, 0, None  # 0 = MTLResourceStorageModeShared
    )


def ensure_contiguous_if_needed(tensor: torch.Tensor) -> torch.Tensor:
    """Only call .contiguous() if tensor is not already contiguous."""
    return tensor if tensor.is_contiguous() else tensor.contiguous()
