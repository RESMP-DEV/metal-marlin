"""Padding helpers for Metal Marlin tensors."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ._compat import HAS_TORCH, require_torch

if TYPE_CHECKING:
    import torch



logger = logging.getLogger(__name__)

def pad_to_multiple(
    tensor: torch.Tensor,
    dim: int,
    multiple: int = 8,
) -> tuple[torch.Tensor, int]:
    """Pad tensor along dim to the next multiple.

    Returns:
        (padded_tensor, pad_size)
    """
    logger.debug("pad_to_multiple called with tensor=%s, dim=%s, multiple=%s", tensor, dim, multiple)
    if not HAS_TORCH:
        require_torch("padding tensors")

    if multiple <= 0:
        raise ValueError(f"multiple must be > 0, got {multiple}")

    dim_norm = dim % tensor.dim()
    size = tensor.size(dim_norm)
    if size % multiple == 0:
        return tensor, 0

    pad_size = multiple - (size % multiple)
    new_shape = list(tensor.shape)
    new_shape[dim_norm] = size + pad_size

    import torch as _torch

    padded = _torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
    slices: list[slice] = [slice(None)] * tensor.dim()
    slices[dim_norm] = slice(0, size)
    padded[tuple(slices)] = tensor
    return padded, pad_size


def unpad(tensor: torch.Tensor, dim: int, pad_size: int) -> torch.Tensor:
    """Remove padding added by pad_to_multiple."""
    logger.debug("unpad called with tensor=%s, dim=%s, pad_size=%s", tensor, dim, pad_size)
    if pad_size == 0:
        return tensor
    dim_norm = dim % tensor.dim()
    return tensor.narrow(dim_norm, 0, tensor.size(dim_norm) - pad_size)
