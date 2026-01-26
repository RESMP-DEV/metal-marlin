"""Centralized optional dependency handling for metal_marlin.

This module provides:
- Feature flags (HAS_TORCH, HAS_MPS, HAS_PYOBJC_METAL) for runtime detection
- Typed module references that work with type checkers
- Array conversion utilities for numpy/torch

Usage:
    from metal_marlin._compat import HAS_TORCH, HAS_MPS, torch, to_numpy
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Feature flags
HAS_TORCH: bool = False
HAS_MPS: bool = False
HAS_PYOBJC_METAL: bool = False
HAS_MATPLOTLIB: bool = False
HAS_MPSGRAPH: bool = False

# Module references (None when unavailable)
torch: ModuleType | None = None
Metal: ModuleType | None = None
plt: ModuleType | None = None
MPSGraph: ModuleType | None = None

# Try importing torch
try:
    import torch as _torch

    torch = _torch
    HAS_TORCH = True
    HAS_MPS = _torch.backends.mps.is_available()
except ImportError:
    pass

# Try importing pyobjc Metal framework
try:
    import Metal as _Metal

    Metal = _Metal
    HAS_PYOBJC_METAL = True
except ImportError:
    pass

# Try importing MPSGraph framework
try:
    import MetalPerformanceShadersGraph as _MPSGraph

    MPSGraph = _MPSGraph
    HAS_MPSGRAPH = True
except ImportError:
    pass

# Try importing matplotlib
try:
    import matplotlib.pyplot as _plt

    plt = _plt
    HAS_MATPLOTLIB = True
except ImportError:
    pass


def to_numpy(arr: Any) -> NDArray[Any]:
    """Convert any array type to numpy.

    Handles:
    - PyTorch tensors (moves to CPU if needed)
    - NumPy arrays (no-op)
    - Anything with __array__ protocol

    Args:
        arr: Input array (torch.Tensor, numpy.ndarray, or array-like)

    Returns:
        numpy.ndarray
    """
    # PyTorch tensors
    if HAS_TORCH and torch is not None:
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()

    # Fallback: assume numpy-compatible
    return np.asarray(arr)


def from_numpy(arr: ArrayLike, backend: str = "numpy", dtype: Any = None) -> Any:
    """Convert numpy array to specified backend.

    Args:
        arr: Input array (numpy or array-like)
        backend: Target backend ('numpy' or 'torch')
        dtype: Optional dtype for the target backend

    Returns:
        Array in the specified backend format

    Raises:
        ValueError: If backend is unavailable
    """
    np_arr = np.asarray(arr)

    if backend == "numpy":
        return np_arr.astype(dtype) if dtype is not None else np_arr

    if backend == "torch":
        if not HAS_TORCH or torch is None:
            raise ValueError("PyTorch not available. Install with: pip install torch")
        result = torch.from_numpy(np_arr.copy())  # copy to avoid negative strides
        if dtype is not None:
            result = result.to(dtype)
        return result

    raise ValueError(f"Unknown backend: {backend}. Use 'numpy' or 'torch'")


def get_array_backend(arr: Any) -> str:
    """Detect the backend of an array.

    Args:
        arr: Input array

    Returns:
        'torch' or 'numpy'
    """
    if HAS_TORCH and torch is not None and isinstance(arr, torch.Tensor):
        return "torch"
    return "numpy"


def ensure_contiguous(arr: Any) -> Any:
    """Ensure array is contiguous in memory.

    Args:
        arr: Input array

    Returns:
        Contiguous array (may be a copy)
    """
    backend = get_array_backend(arr)

    if backend == "torch" and torch is not None:
        return arr.contiguous()
    # numpy
    return np.ascontiguousarray(arr)


def get_default_backend() -> str:
    """Return the best available backend for inference.

    Priority:
        1. torch - CUDA/MPS/CPU via PyTorch
        2. numpy - CPU-only fallback (limited functionality)

    Returns:
        Backend name: 'torch' or 'numpy'
    """
    if HAS_TORCH:
        return "torch"
    return "numpy"


def require_torch(feature: str = "this operation") -> None:
    """Raise RuntimeError if PyTorch is not available.

    Args:
        feature: Description of what requires PyTorch for the error message.

    Raises:
        RuntimeError: If PyTorch is not installed.
    """
    if not HAS_TORCH:
        raise RuntimeError(f"PyTorch is required for {feature}. Install with: pip install torch")


def require_backend(backend: str, feature: str = "this operation") -> None:
    """Raise RuntimeError if the specified backend is not available.

    Args:
        backend: Backend name ('torch', 'numpy')
        feature: Description of what requires the backend.

    Raises:
        RuntimeError: If the backend is not available.
        ValueError: If backend name is invalid.
    """
    if backend == "torch":
        require_torch(feature)
    elif backend == "numpy":
        pass  # Always available
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'torch' or 'numpy'.")


# Type hints for static analysis
if TYPE_CHECKING:
    import torch  # noqa: F811
