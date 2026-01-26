"""Centralized optional dependency handling for metal_marlin.

This module provides:
- Feature flags (HAS_MLX, HAS_TORCH) for runtime detection
- Typed module references that work with type checkers
- Array conversion utilities across numpy/torch/mlx

Usage:
    from metal_marlin._compat import HAS_MLX, HAS_TORCH, mx, torch, to_numpy
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Feature flags
HAS_MLX: bool = False
HAS_TORCH: bool = False

# Module references (None when unavailable)
mx: ModuleType | None = None
nn: ModuleType | None = None
torch: ModuleType | None = None

# Try importing mlx
try:
    import mlx.core as _mx
    import mlx.nn as _nn

    mx = _mx
    nn = _nn
    HAS_MLX = True
except ImportError:
    pass

# Try importing torch
try:
    import torch as _torch

    torch = _torch
    HAS_TORCH = True
except ImportError:
    pass


def to_numpy(arr: Any) -> NDArray[Any]:
    """Convert any array type to numpy.

    Handles:
    - MLX arrays (evaluates and converts)
    - PyTorch tensors (moves to CPU if needed)
    - NumPy arrays (no-op)
    - Anything with __array__ protocol

    Args:
        arr: Input array (mlx.core.array, torch.Tensor, numpy.ndarray, or array-like)

    Returns:
        numpy.ndarray
    """
    # MLX arrays
    if HAS_MLX and mx is not None:
        # Check if it's an MLX array by type name (avoids importing for isinstance)
        if type(arr).__module__.startswith("mlx"):
            mx.eval(arr)
            return np.array(arr)

    # PyTorch tensors
    if HAS_TORCH and torch is not None:
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()

    # Fallback: assume numpy-compatible
    return np.asarray(arr)


def from_numpy(
    arr: ArrayLike, backend: str = "numpy", dtype: Any = None
) -> Any:
    """Convert numpy array to specified backend.

    Args:
        arr: Input array (numpy or array-like)
        backend: Target backend ('numpy', 'mlx', or 'torch')
        dtype: Optional dtype for the target backend

    Returns:
        Array in the specified backend format

    Raises:
        ValueError: If backend is unavailable
    """
    np_arr = np.asarray(arr)

    if backend == "numpy":
        return np_arr.astype(dtype) if dtype is not None else np_arr

    if backend == "mlx":
        if not HAS_MLX or mx is None:
            raise ValueError("MLX not available. Install with: pip install mlx")
        result = mx.array(np_arr)
        if dtype is not None:
            result = result.astype(dtype)
        return result

    if backend == "torch":
        if not HAS_TORCH or torch is None:
            raise ValueError("PyTorch not available. Install with: pip install torch")
        result = torch.from_numpy(np_arr.copy())  # copy to avoid negative strides
        if dtype is not None:
            result = result.to(dtype)
        return result

    raise ValueError(f"Unknown backend: {backend}. Use 'numpy', 'mlx', or 'torch'")


def get_array_backend(arr: Any) -> str:
    """Detect the backend of an array.

    Args:
        arr: Input array

    Returns:
        'mlx', 'torch', or 'numpy'
    """
    if HAS_MLX and type(arr).__module__.startswith("mlx"):
        return "mlx"
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

    if backend == "mlx" and mx is not None:
        # MLX arrays are always contiguous
        return arr
    if backend == "torch" and torch is not None:
        return arr.contiguous()
    # numpy
    return np.ascontiguousarray(arr)


def get_default_backend() -> str:
    """Return the best available backend for inference.

    Priority:
        1. mlx - Apple Silicon with Metal acceleration
        2. torch - CUDA/CPU via PyTorch
        3. numpy - CPU-only fallback (limited functionality)

    Returns:
        Backend name: 'mlx', 'torch', or 'numpy'
    """
    if HAS_MLX:
        return "mlx"
    if HAS_TORCH:
        return "torch"
    return "numpy"


def require_mlx(feature: str = "this operation") -> None:
    """Raise RuntimeError if MLX is not available.

    Args:
        feature: Description of what requires MLX for the error message.

    Raises:
        RuntimeError: If MLX is not installed.
    """
    if not HAS_MLX:
        raise RuntimeError(
            f"MLX is required for {feature}. "
            "Install with: pip install mlx"
        )


def require_torch(feature: str = "this operation") -> None:
    """Raise RuntimeError if PyTorch is not available.

    Args:
        feature: Description of what requires PyTorch for the error message.

    Raises:
        RuntimeError: If PyTorch is not installed.
    """
    if not HAS_TORCH:
        raise RuntimeError(
            f"PyTorch is required for {feature}. "
            "Install with: pip install torch"
        )


def require_backend(backend: str, feature: str = "this operation") -> None:
    """Raise RuntimeError if the specified backend is not available.

    Args:
        backend: Backend name ('mlx', 'torch', 'numpy')
        feature: Description of what requires the backend.

    Raises:
        RuntimeError: If the backend is not available.
        ValueError: If backend name is invalid.
    """
    if backend == "mlx":
        require_mlx(feature)
    elif backend == "torch":
        require_torch(feature)
    elif backend == "numpy":
        pass  # Always available
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'mlx', 'torch', or 'numpy'.")


# Type hints for static analysis
if TYPE_CHECKING:
    import mlx.core as mx  # noqa: F811
    import torch  # noqa: F811
