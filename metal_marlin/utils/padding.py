"""Padding helpers for alignment-sensitive kernels."""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # Optional torch support for runtime padding on MPS
    import torch
except ImportError:  # pragma: no cover - torch is optional for CPU-only tooling
    torch = None  # type: ignore[assignment]


def round_up(value: int, multiple: int) -> int:
    """Round value up to the nearest multiple."""
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    return ((value + multiple - 1) // multiple) * multiple


def pad_numpy_2d(
    array: np.ndarray,
    *,
    rows_multiple: int,
    cols_multiple: int,
    value: float = 0.0,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad a 2D numpy array to the requested multiples."""
    rows, cols = array.shape
    pad_rows = (rows_multiple - (rows % rows_multiple)) % rows_multiple
    pad_cols = (cols_multiple - (cols % cols_multiple)) % cols_multiple
    if pad_rows == 0 and pad_cols == 0:
        return array, (0, 0)
    padded = np.pad(
        array,
        ((0, pad_rows), (0, pad_cols)),
        mode="constant",
        constant_values=value,
    )
    return padded, (pad_rows, pad_cols)


def pad_torch_2d(
    tensor: Any,
    *,
    rows_multiple: int,
    cols_multiple: int,
    value: float = 0.0,
) -> tuple[Any, tuple[int, int]]:
    """Pad a 2D torch tensor to the requested multiples."""
    if torch is None:
        raise ImportError("torch is required for pad_torch_2d")
    rows, cols = tensor.shape
    pad_rows = (rows_multiple - (rows % rows_multiple)) % rows_multiple
    pad_cols = (cols_multiple - (cols % cols_multiple)) % cols_multiple
    if pad_rows == 0 and pad_cols == 0:
        return tensor, (0, 0)
    padded = torch.full(
        (rows + pad_rows, cols + pad_cols),
        value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:rows, :cols] = tensor
    return padded, (pad_rows, pad_cols)
