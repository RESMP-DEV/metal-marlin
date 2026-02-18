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


def _is_power_of_2(n: int) -> bool:
    """Check if n is a positive power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def _get_block_diagonal_decomposition(n: int) -> tuple[int, ...] | None:
    """Get power-of-2 block sizes for non-power-of-2 Hadamard decomposition.

    For supported non-power-of-2 sizes, returns the tuple of power-of-2 blocks
    that sum to n. Uses block-diagonal decomposition: H_n = diag(H_a, H_b, ...).

    Uses a greedy strategy to decompose n into supported block sizes (128, 64, 32).

    Args:
        n: Target block size.

    Returns:
        Tuple of power-of-2 block sizes that sum to n, or None if not supported.
    """
    if n <= 0:
        return None

    # Supported base block sizes in descending order
    # We prefer larger blocks for better utilization
    supported_bases = [128, 64, 32]
    
    blocks = []
    remaining = n
    
    # Greedy decomposition
    while remaining > 0:
        found = False
        for base in supported_bases:
            if remaining >= base:
                blocks.append(base)
                remaining -= base
                found = True
                break
        
        if not found:
            # Remainder cannot be covered by supported blocks (e.g., 16, 8)
            return None
            
    return tuple(blocks)


def _apply_block_diagonal_hadamard(
    w: NDArray[np.floating[Any]],
    block_sizes: tuple[int, ...],
    axis: int,
) -> NDArray[np.floating[Any]]:
    """Apply block-diagonal Hadamard transform using multiple power-of-2 blocks.

    For non-power-of-2 sizes, decomposes into independent power-of-2 blocks:
        H_96  = diag(H_64, H_32)
        H_160 = diag(H_128, H_32)
        H_192 = diag(H_128, H_64)

    Each block is transformed independently with its own normalization factor.

    Args:
        w: Weight matrix padded to sum(block_sizes).
        block_sizes: Tuple of power-of-2 block sizes (e.g., (64, 32) for 96).
        axis: Axis along which to apply rotation (0 for K, 1 for N).

    Returns:
        Transformed weight matrix with block-diagonal Hadamard applied.
    """
    if axis == 0:
        # Rotate along K: process each block independently
        K, N = w.shape
        result_blocks = []
        offset = 0

        for block_size in block_sizes:
            H = hadamard_matrix(block_size)
            # Extract block: [block_size, N]
            w_block = w[offset : offset + block_size, :]
            # Apply transform: H @ block
            transformed = H @ w_block
            result_blocks.append(transformed)
            offset += block_size

        return np.vstack(result_blocks)
    else:
        # Rotate along N: process each block independently
        K, N = w.shape
        result_blocks = []
        offset = 0

        for block_size in block_sizes:
            H = hadamard_matrix(block_size)
            # Extract block: [K, block_size]
            w_block = w[:, offset : offset + block_size]
            # Apply transform: block @ H.T
            transformed = w_block @ H.T
            result_blocks.append(transformed)
            offset += block_size

        return np.hstack(result_blocks)


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

    Supports both power-of-2 sizes (32, 64, 128, 256) and non-power-of-2
    sizes (96, 160, 192) via block-diagonal decomposition for optimal
    performance on dimensions that don't align with power-of-2 boundaries.

    Non-power-of-2 decomposition:
        - 96  = 64 + 32: Intermediate dimensions
        - 160 = 128 + 32: Custom model dimensions
        - 192 = 128 + 64: Multi-head attention (24 heads x 8)

    Args:
        weights: Weight matrix [K, N] (input_features x output_features).
        block_size: Size of Hadamard blocks.
            Power-of-2 values: 32, 64, 128, 256.
            Non-power-of-2 values: 96, 160, 192.
            Typical values: 64 or 128 (matches quantization group size).
        axis: Axis along which to apply rotation (0 for K, 1 for N).
            Default: 0 (rotate input features).

    Returns:
        Tuple of:
            rotated: Rotated weight matrix [K_padded, N] or [K, N_padded].
            metadata: HadamardMetadata for reversing the rotation.

    Raises:
        ValueError: If block_size is not a supported size.
        ValueError: If axis is not 0 or 1.

    Example:
        >>> W = np.random.randn(256, 512).astype(np.float32)
        >>> W_rot, meta = apply_hadamard_rotation(W, block_size=64)
        >>> W_rot.shape
        (256, 512)
        >>> # Max/mean ratio should decrease (outliers dispersed)
        >>> abs(W_rot).max() / abs(W_rot).mean() < abs(W).max() / abs(W).mean()
        True
        
        >>> # Non-power-of-2 sizes work too
        >>> W96 = np.random.randn(192, 512).astype(np.float32)
        >>> W96_rot, meta96 = apply_hadamard_rotation(W96, block_size=96)
    """
    # Check if block_size is supported (power-of-2 or decomposable)
    # Power of 2 check
    is_pow2 = (block_size > 0) and ((block_size & (block_size - 1)) == 0)
    
    if is_pow2:
        block_decomp = None
    else:
        block_decomp = _get_block_diagonal_decomposition(block_size)
        if block_decomp is None:
             raise ValueError(
                f"block_size {block_size} not supported. Must be power of 2 or "
                "decomposable into 128, 64, 32 blocks."
            )

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

    # Check if we need block-diagonal decomposition for non-power-of-2
    block_decomp = _get_block_diagonal_decomposition(block_size)

    if block_decomp is not None:
        # Non-power-of-2: Use block-diagonal decomposition
        # Process each large block as independent sub-blocks
        if axis == 0:
            # Rotate along K
            K, N = w.shape
            block_results = []
            for block_idx in range(num_blocks):
                block_start = block_idx * block_size
                block_end = block_start + block_size
                w_block = w[block_start:block_end, :]
                # Apply block-diagonal Hadamard to this block
                rotated_block = _apply_block_diagonal_hadamard(
                    w_block, block_decomp, axis=0
                )
                block_results.append(rotated_block)
            rotated = np.vstack(block_results)
        else:
            # Rotate along N
            K, N = w.shape
            block_results = []
            for block_idx in range(num_blocks):
                block_start = block_idx * block_size
                block_end = block_start + block_size
                w_block = w[:, block_start:block_end]
                # Apply block-diagonal Hadamard to this block
                rotated_block = _apply_block_diagonal_hadamard(
                    w_block, block_decomp, axis=1
                )
                block_results.append(rotated_block)
            rotated = np.hstack(block_results)
    else:
        # Power-of-2: Use standard einsum approach
        H = hadamard_matrix(block_size)

        if axis == 0:
            # Rotate along K: W_rotated[block_i] = H @ W[block_i]
            K, N = w.shape
            w_blocks = w.reshape(num_blocks, block_size, N)
            rotated_blocks = np.einsum("ij,bjk->bik", H, w_blocks)
            rotated = rotated_blocks.reshape(K, N)
        else:
            # Rotate along N: W_rotated[:, block_i] = W[:, block_i] @ H.T
            K, N = w.shape
            w_blocks = w.reshape(K, num_blocks, block_size)
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

    Supports both power-of-2 and non-power-of-2 block sizes (96, 160, 192)
    via block-diagonal decomposition.

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
        
        >>> # Non-power-of-2 sizes work too
        >>> W96 = np.random.randn(192, 512).astype(np.float32)
        >>> W96_rot, meta96 = apply_hadamard_rotation(W96, block_size=96)
        >>> W96_recovered = inverse_hadamard_rotation(W96_rot, meta96)
        >>> np.allclose(W96, W96_recovered)
        True
    """
    block_size = metadata.block_size
    axis = metadata.axis
    orig_dim = metadata.orig_k

    w = np.asarray(weights, dtype=np.float32)
    padded_dim = w.shape[axis]
    num_blocks = padded_dim // block_size

    # Check if we need block-diagonal decomposition for non-power-of-2
    block_decomp = _get_block_diagonal_decomposition(block_size)

    if block_decomp is not None:
        # Non-power-of-2: Use block-diagonal decomposition
        # Hadamard is self-inverse, so apply same transform
        if axis == 0:
            K, N = w.shape
            block_results = []
            for block_idx in range(num_blocks):
                block_start = block_idx * block_size
                block_end = block_start + block_size
                w_block = w[block_start:block_end, :]
                rotated_block = _apply_block_diagonal_hadamard(
                    w_block, block_decomp, axis=0
                )
                block_results.append(rotated_block)
            recovered = np.vstack(block_results)
            # Trim padding
            recovered = recovered[:orig_dim, :]
        else:
            K, N = w.shape
            block_results = []
            for block_idx in range(num_blocks):
                block_start = block_idx * block_size
                block_end = block_start + block_size
                w_block = w[:, block_start:block_end]
                rotated_block = _apply_block_diagonal_hadamard(
                    w_block, block_decomp, axis=1
                )
                block_results.append(rotated_block)
            recovered = np.hstack(block_results)
            # Trim padding
            recovered = recovered[:, :orig_dim]
    else:
        # Power-of-2: Use standard einsum approach
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
