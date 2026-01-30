"""Hadamard transform for weight quantization outlier dispersal.

Hadamard rotation redistributes outlier magnitudes across channels, making
per-channel quantization scales more uniform. This is critical for low-bit
quantization (4-bit) where large outliers cause catastrophic errors.

The key insight is that Hadamard matrices are orthonormal (H @ H.T = I) and
self-inverse (H^-1 = H^T = H for normalized Hadamard), so the transformation
can be exactly reversed during inference.

Reference:
    QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs (arxiv:2404.00456)

Example:
    >>> weights = np.random.randn(256, 512).astype(np.float32)
    >>> # Simulate an outlier
    >>> weights[0, 0] = 100.0
    >>> rotated, meta = apply_hadamard_rotation(weights, block_size=64)
    >>> # Outlier is now dispersed across the block
    >>> assert rotated.max() < weights.max()
    >>> # Rotation is perfectly reversible
    >>> recovered = inverse_hadamard_rotation(rotated, meta)
    >>> np.testing.assert_allclose(recovered, weights, rtol=1e-5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class HadamardMetadata:
    """Metadata for reversing Hadamard rotation.

    Attributes:
        block_size: Size of diagonal blocks used in rotation.
        orig_k: Original K dimension before padding.
        padded_k: K dimension after padding to block_size multiple.
        axis: Axis along which rotation was applied (0 for K, 1 for N).
    """

    block_size: int
    orig_k: int
    padded_k: int
    axis: int


def hadamard_matrix(n: int) -> NDArray[np.floating[Any]]:
    """Generate normalized Hadamard matrix H_n where H_n @ H_n.T = I.

    Uses Sylvester construction: H_2n = [[H_n, H_n], [H_n, -H_n]].
    The matrix is normalized by 1/sqrt(n) so H @ H.T = I (orthonormal).

    Args:
        n: Size of the matrix. Must be a power of 2.

    Returns:
        Normalized Hadamard matrix of shape [n, n] with H @ H.T = I.

    Raises:
        ValueError: If n is not a positive power of 2.

    Example:
        >>> H = hadamard_matrix(4)
        >>> np.allclose(H @ H.T, np.eye(4))
        True
        >>> H.shape
        (4, 4)
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n & (n - 1) != 0:
        raise ValueError(f"n must be a power of 2, got {n}")

    # Build unnormalized Hadamard via Sylvester construction
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])

    # Normalize so H @ H.T = I
    return H / np.sqrt(n)


def _pad_to_multiple(
    arr: NDArray[np.floating[Any]], axis: int, multiple: int
) -> NDArray[np.floating[Any]]:
    """Pad array along given axis to the next multiple of `multiple`."""
    current = arr.shape[axis]
    if current % multiple == 0:
        return arr
    pad_amount = multiple - (current % multiple)
    pad_widths = [(0, 0)] * arr.ndim
    pad_widths[axis] = (0, pad_amount)
    return np.pad(arr, pad_widths, mode="constant", constant_values=0.0)


def apply_hadamard_rotation(
    weights: NDArray[np.floating[Any]],
    block_size: int = 64,
    axis: int = 0,
) -> tuple[NDArray[np.floating[Any]], HadamardMetadata]:
    """Apply block-diagonal Hadamard rotation to disperse outliers.

    Applies Hadamard transform in non-overlapping blocks along the specified
    axis. This redistributes outlier magnitudes across each block, making
    per-channel quantization scales more uniform.

    The transformation is: W_rotated = H_block @ W_block for each block,
    where H_block is the normalized Hadamard matrix of size block_size.

    Args:
        weights: Weight matrix [K, N] (input_features x output_features).
        block_size: Size of Hadamard blocks. Must be power of 2.
            Typical values: 64 or 128 (matches quantization group size).
        axis: Axis along which to apply rotation (0 for K, 1 for N).
            Default: 0 (rotate input features).

    Returns:
        Tuple of:
            rotated: Rotated weight matrix [K_padded, N] or [K, N_padded].
            metadata: HadamardMetadata for reversing the rotation.

    Raises:
        ValueError: If block_size is not a power of 2.
        ValueError: If axis is not 0 or 1.

    Example:
        >>> W = np.random.randn(256, 512).astype(np.float32)
        >>> W_rot, meta = apply_hadamard_rotation(W, block_size=64)
        >>> W_rot.shape
        (256, 512)
        >>> # Max/mean ratio should decrease (outliers dispersed)
        >>> abs(W_rot).max() / abs(W_rot).mean() < abs(W).max() / abs(W).mean()
        True
    """
    if block_size & (block_size - 1) != 0 or block_size <= 0:
        raise ValueError(f"block_size must be a positive power of 2, got {block_size}")
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    w = np.asarray(weights, dtype=np.float32)
    orig_shape = w.shape
    orig_dim = orig_shape[axis]

    # Pad to multiple of block_size
    if orig_dim % block_size != 0:
        w = _pad_to_multiple(w, axis=axis, multiple=block_size)

    padded_dim = w.shape[axis]
    num_blocks = padded_dim // block_size

    # Generate Hadamard matrix once
    H = hadamard_matrix(block_size)

    # Apply block-diagonal rotation
    if axis == 0:
        # Rotate along K: W_rotated[block_i] = H @ W[block_i]
        # Shape: [K, N] -> reshape to [num_blocks, block_size, N]
        K, N = w.shape
        w_blocks = w.reshape(num_blocks, block_size, N)
        # H @ each block: [block_size, block_size] @ [block_size, N] -> [block_size, N]
        rotated_blocks = np.einsum("ij,bjk->bik", H, w_blocks)
        rotated = rotated_blocks.reshape(K, N)
    else:
        # Rotate along N: W_rotated[:, block_i] = W[:, block_i] @ H.T
        # Shape: [K, N] -> reshape to [K, num_blocks, block_size]
        K, N = w.shape
        w_blocks = w.reshape(K, num_blocks, block_size)
        # Each block @ H.T: [K, block_size] @ [block_size, block_size] -> [K, block_size]
        rotated_blocks = np.einsum("bij,jk->bik", w_blocks, H.T)
        rotated = rotated_blocks.reshape(K, N)

    metadata = HadamardMetadata(
        block_size=block_size,
        orig_k=orig_dim,
        padded_k=padded_dim,
        axis=axis,
    )

    return rotated, metadata


def inverse_hadamard_rotation(
    weights: NDArray[np.floating[Any]],
    metadata: HadamardMetadata,
) -> NDArray[np.floating[Any]]:
    """Reverse the Hadamard rotation applied by apply_hadamard_rotation.

    For normalized Hadamard matrices, H^-1 = H^T = H, so the inverse is
    simply applying the same rotation again and trimming padding.

    Args:
        weights: Rotated weight matrix from apply_hadamard_rotation.
        metadata: HadamardMetadata from apply_hadamard_rotation.

    Returns:
        Original weight matrix [orig_K, N] or [K, orig_N] (padding stripped).

    Example:
        >>> W = np.random.randn(256, 512).astype(np.float32)
        >>> W_rot, meta = apply_hadamard_rotation(W, block_size=64)
        >>> W_recovered = inverse_hadamard_rotation(W_rot, meta)
        >>> np.allclose(W, W_recovered)
        True
    """
    block_size = metadata.block_size
    axis = metadata.axis
    orig_dim = metadata.orig_k

    w = np.asarray(weights, dtype=np.float32)
    padded_dim = w.shape[axis]
    num_blocks = padded_dim // block_size

    # Hadamard is self-inverse: H @ H = I (for normalized H)
    H = hadamard_matrix(block_size)

    if axis == 0:
        K, N = w.shape
        w_blocks = w.reshape(num_blocks, block_size, N)
        # Inverse: H.T @ each block, but H.T = H for Hadamard
        recovered_blocks = np.einsum("ij,bjk->bik", H, w_blocks)
        recovered = recovered_blocks.reshape(K, N)
        # Trim padding
        recovered = recovered[:orig_dim, :]
    else:
        K, N = w.shape
        w_blocks = w.reshape(K, num_blocks, block_size)
        recovered_blocks = np.einsum("bij,jk->bik", w_blocks, H)
        recovered = recovered_blocks.reshape(K, N)
        recovered = recovered[:, :orig_dim]

    return recovered


def compute_outlier_stats(weights: NDArray[np.floating[Any]]) -> dict[str, float]:
    """Compute statistics useful for measuring outlier dispersal effectiveness.

    Args:
        weights: Weight matrix of any shape.

    Returns:
        Dictionary with:
            - max_abs: Maximum absolute value
            - mean_abs: Mean absolute value
            - max_mean_ratio: max_abs / mean_abs (lower = better dispersed)
            - std: Standard deviation
            - kurtosis: Excess kurtosis (higher = more outliers)
    """
    w = np.abs(weights.ravel()).astype(np.float64)
    mean_abs = float(w.mean())
    max_abs = float(w.max())
    std = float(np.std(weights))

    # Excess kurtosis (normal = 0)
    centered = weights.ravel() - weights.mean()
    if std > 0:
        kurtosis = float(np.mean(centered**4) / (std**4) - 3.0)
    else:
        kurtosis = 0.0

    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_mean_ratio": max_abs / mean_abs if mean_abs > 0 else 0.0,
        "std": std,
        "kurtosis": kurtosis,
    }
