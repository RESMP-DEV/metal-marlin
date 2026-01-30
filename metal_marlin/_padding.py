"""Padding helpers for Metal Marlin tensors."""

from __future__ import annotations

from ._compat import HAS_TORCH, require_torch, torch


def pad_to_multiple(
    tensor: torch.Tensor,
    dim: int,
    multiple: int = 8,
) -> tuple[torch.Tensor, int]:
    """Pad tensor along dim to the next multiple.

    Returns:
        (padded_tensor, pad_size)
    """
    if not HAS_TORCH or torch is None:
        require_torch("padding tensors")

    if multiple <= 0:
        raise ValueError(f"multiple must be > 0, got {multiple}")

    dim = dim % tensor.dim()
    size = tensor.size(dim)
    if size % multiple == 0:
        return tensor, 0

    pad_size = multiple - (size % multiple)
    new_shape = list(tensor.shape)
    new_shape[dim] = size + pad_size

    padded = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
    slices = [slice(None)] * tensor.dim()
    slices[dim] = slice(0, size)
    padded[tuple(slices)] = tensor
    return padded, pad_size


def unpad(tensor: torch.Tensor, dim: int, pad_size: int) -> torch.Tensor:
    """Remove padding added by pad_to_multiple."""
    if pad_size == 0:
        return tensor
    dim = dim % tensor.dim()
    return tensor.narrow(dim, 0, tensor.size(dim) - pad_size)
