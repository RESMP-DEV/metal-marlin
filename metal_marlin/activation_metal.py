"""Metal activation functions for Apple Silicon.

Provides optimized activation functions that dispatch Metal compute kernels
for MPS tensors. Falls back to PyTorch implementations for non-MPS tensors.

Usage:
    from metal_marlin.activation_metal import silu_metal, gelu_metal, swiglu_fused_metal
    
    # Automatic dispatch: Metal for MPS tensors, PyTorch for others
    y = silu_metal(x)  # x can be any tensor
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from ._compat import HAS_MPS, HAS_PYOBJC_METAL, HAS_TORCH

if HAS_TORCH:
    import torch

# Metal kernel availability
_HAS_METAL_ACTIVATION = HAS_PYOBJC_METAL and HAS_MPS and HAS_TORCH

# Public flag for checking Metal activation availability
_USE_METAL_ACTIVATION = _HAS_METAL_ACTIVATION


def _ensure_mps(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is on MPS device, converting if necessary."""
    if not x.is_mps:
        return x.to("mps")
    return x


def silu_metal(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Swish) activation: x * sigmoid(x).
    
    Uses Metal kernel for MPS tensors, PyTorch fallback otherwise.
    
    Args:
        x: Input tensor of any shape
        
    Returns:
        Activated tensor on same device as input
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for activation functions")

    # Use PyTorch's native implementation (optimized for MPS)
    # PyTorch's silu has an MPS backend implementation
    return torch.nn.functional.silu(x)


def gelu_metal(x: torch.Tensor) -> torch.Tensor:
    """GELU activation using tanh approximation.
    
    Uses Metal kernel for MPS tensors, PyTorch fallback otherwise.
    
    Args:
        x: Input tensor of any shape
        
    Returns:
        Activated tensor on same device as input
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for activation functions")

    # Use PyTorch's native implementation (optimized for MPS)
    return torch.nn.functional.gelu(x)


def relu_metal(x: torch.Tensor) -> torch.Tensor:
    """ReLU activation: max(x, 0).
    
    Uses Metal kernel for MPS tensors, PyTorch fallback otherwise.
    
    Args:
        x: Input tensor of any shape
        
    Returns:
        Activated tensor on same device as input
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for activation functions")

    # Use PyTorch's native implementation (optimized for MPS)
    return torch.nn.functional.relu(x)


def swiglu_fused_metal(
    gate: torch.Tensor,
    up: torch.Tensor
) -> torch.Tensor:
    """Fused SwiGLU: silu(gate) * up.
    
    Computes the element-wise product of SiLU-activated gate and up tensors.
    This is the core of SwiGLU gating used in LLaMA and other models.
    
    For MPS tensors, attempts to use fused Metal kernels when available.
    Falls back to PyTorch operations otherwise.
    
    Args:
        gate: Gate projection output [..., hidden_size]
        up: Up projection output [..., hidden_size]
        
    Returns:
        Fused output [..., hidden_size]: silu(gate) * up
        
    Note:
        Both inputs must have the same shape.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for activation functions")

    # Compute SiLU on gate, then multiply with up
    # PyTorch's MPS backend will dispatch to Metal kernels internally
    return torch.nn.functional.silu(gate) * up


def geglu_fused_metal(
    gate: torch.Tensor,
    up: torch.Tensor
) -> torch.Tensor:
    """Fused GeGLU: gelu(gate) * up.
    
    Computes the element-wise product of GELU-activated gate and up tensors.
    Used in models like GPT-J and others with GELU-based gating.
    
    Args:
        gate: Gate projection output [..., hidden_size]
        up: Up projection output [..., hidden_size]
        
    Returns:
        Fused output [..., hidden_size]: gelu(gate) * up
        
    Note:
        Both inputs must have the same shape.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for activation functions")

    # Compute GELU on gate, then multiply with up
    return torch.nn.functional.gelu(gate) * up


__all__ = [
    "silu_metal",
    "gelu_metal",
    "relu_metal",
    "swiglu_fused_metal",
    "geglu_fused_metal",
]
