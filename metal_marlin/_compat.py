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

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import Metal  # type: ignore[import-not-found]
    import MetalPerformanceShadersGraph as MPSGraph  # type: ignore[import-not-found]
    import torch

    # Shared dummy Tensor type for Pyright when torch is broken/missing
    class Tensor:
        dtype: Any
        shape: tuple[int, ...]
        ndim: int
        device: Any
        def contiguous(self) -> Tensor: ...
        def detach(self) -> Tensor: ...
        def cpu(self) -> Tensor: ...
        def numpy(self) -> Any: ...
        def to(self, dtype: Any) -> Tensor: ...
        def permute(self, *dims: int) -> Tensor: ...
        def squeeze(self, dim: int | None = None) -> Tensor: ...
        def unsqueeze(self, dim: int) -> Tensor: ...
        def numel(self) -> int: ...
        def element_size(self) -> int: ...


# Feature flags - declared with type annotations to allow reassignment
def _init_flags() -> tuple[bool, bool, bool, bool, bool, bool]:
    """Initialize feature flags."""
    return False, False, False, False, False, False


HAS_TORCH: bool
HAS_MPS: bool
HAS_PYOBJC_METAL: bool
HAS_MATPLOTLIB: bool
HAS_MPSGRAPH: bool
HAS_CPP_EXT: bool
HAS_TORCH, HAS_MPS, HAS_PYOBJC_METAL, HAS_MATPLOTLIB, HAS_MPSGRAPH, HAS_CPP_EXT = _init_flags()

# Module references (None when unavailable)
torch: ModuleType | None = None
Metal: ModuleType | None = None  # type: ignore[name-defined]
plt: ModuleType | None = None  # type: ignore[name-defined]
MPSGraph: ModuleType | None = None  # type: ignore[name-defined]
_metal_dispatch_ext: ModuleType | None = None

# Try importing torch
try:
    import torch as _torch_module

    torch = _torch_module
    HAS_TORCH = True
    try:
        HAS_MPS = _torch_module.backends.mps.is_available()
    except AttributeError:
        HAS_MPS = False
except ImportError:
    torch = None
    HAS_TORCH = False
    HAS_MPS = False

# Try importing pyobjc Metal framework
try:
    import Metal as _Metal_module  # type: ignore[import-not-found]

    Metal = _Metal_module
    HAS_PYOBJC_METAL = True
except ImportError:
    Metal = None
    HAS_PYOBJC_METAL = False

# Try importing MPSGraph framework
try:
    import MetalPerformanceShadersGraph as _MPSGraph_module  # type: ignore[import-not-found]

    MPSGraph = _MPSGraph_module
    HAS_MPSGRAPH = True
except ImportError:
    MPSGraph = None
    HAS_MPSGRAPH = False

# Try importing matplotlib
try:
    import matplotlib.pyplot as _plt_module

    plt = _plt_module
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False

# Try importing C++ extension for fast Metal dispatch
# The C++ extension provides low-overhead kernel dispatch by bypassing PyObjC
try:
    import metal_marlin._cpp_ext as _ext_module

    _metal_dispatch_ext = _ext_module
    HAS_CPP_EXT = True
except ImportError:
    _metal_dispatch_ext = None
    HAS_CPP_EXT = False  # Fall back to pure Python (PyObjC) path


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
