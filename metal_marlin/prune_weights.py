"""2:4 structured sparsity pruning for weight matrices.

Implements NVIDIA's 2:4 fine-grained structured sparsity pattern:
for every contiguous block of 4 elements along K, exactly 2 are kept
(the largest by magnitude) and 2 are zeroed. This yields 50% sparsity
with a regular structure that hardware sparse tensor cores can exploit.

The output format packs kept values contiguously and stores 2-bit
position metadata for reconstruction.
"""

from __future__ import annotations

import numpy as np


def prune_to_2_4(
    weights: np.ndarray,
    group_size: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Prune weights to 2:4 structured sparsity.

    For each contiguous block of 4 elements along the K dimension,
    keeps the 2 largest-magnitude values and discards the rest.

    Args:
        weights: [K, N] float16 weight matrix. K must be divisible by 4.
        group_size: Block size for the sparsity pattern. Must be 4
                    (2:4 is the only supported pattern).

    Returns:
        sparse_weights: [K // 2, N] float16 array of kept values,
                        packed contiguously (2 values per block).
        metadata: [K // 4, N] uint8 array encoding the 2-bit position
                  indices of the two kept elements per block.
                  Bits [1:0] = index of first kept element (0-3).
                  Bits [3:2] = index of second kept element (0-3).
                  Upper 4 bits are zero.

    Raises:
        ValueError: If K is not divisible by 4 or group_size != 4.
    """
    if group_size != 4:
        raise ValueError(f"Only 2:4 sparsity supported (group_size=4), got {group_size}")

    K, N = weights.shape
    if K % 4 != 0:
        raise ValueError(f"K={K} must be divisible by 4 for 2:4 sparsity")

    num_blocks = K // 4

    # Reshape into [num_blocks, 4, N] blocks along K
    blocks = weights.reshape(num_blocks, 4, N)

    # Find the top-2 indices by magnitude within each block
    abs_blocks = np.abs(blocks)

    # argsort descending along the group axis (axis=1)
    # Shape: [num_blocks, 4, N]
    sorted_idx = np.argsort(-abs_blocks, axis=1)

    # The two kept positions are the first two along axis=1 after sort
    idx0 = sorted_idx[:, 0, :]  # [num_blocks, N]
    idx1 = sorted_idx[:, 1, :]  # [num_blocks, N]

    # Ensure idx0 < idx1 for canonical ordering (ascending position)
    swap = idx0 > idx1
    keep_first = np.where(swap, idx1, idx0)
    keep_second = np.where(swap, idx0, idx1)

    # Gather the kept values using advanced indexing
    block_idx = np.arange(num_blocks)[:, None]  # [num_blocks, 1]
    n_idx = np.arange(N)[None, :]  # [1, N]

    val0 = blocks[block_idx, keep_first, n_idx]  # [num_blocks, N]
    val1 = blocks[block_idx, keep_second, n_idx]  # [num_blocks, N]

    # Pack kept values: interleave as [val0_block0, val1_block0, val0_block1, ...]
    # Result shape: [K // 2, N]
    sparse_weights = np.empty((num_blocks * 2, N), dtype=weights.dtype)
    sparse_weights[0::2] = val0
    sparse_weights[1::2] = val1

    # Pack metadata: 2-bit indices per kept element
    # Bits [1:0] = keep_first position (0-3)
    # Bits [3:2] = keep_second position (0-3)
    metadata = (keep_first & 0x3).astype(np.uint8) | ((keep_second & 0x3).astype(np.uint8) << 2)

    return sparse_weights, metadata


def unprune_2_4(
    sparse_weights: np.ndarray,
    metadata: np.ndarray,
) -> np.ndarray:
    """Reconstruct a dense weight matrix from 2:4 sparse format.

    Inverse of prune_to_2_4: places kept values back at their original
    positions and fills the pruned positions with zeros.

    Args:
        sparse_weights: [K // 2, N] float16 packed kept values.
        metadata: [K // 4, N] uint8 position indices.

    Returns:
        dense: [K, N] float16 reconstructed weight matrix (with zeros
               at pruned positions).
    """
    half_K, N = sparse_weights.shape
    num_blocks = half_K // 2
    K = num_blocks * 4

    # Extract position indices from metadata
    keep_first = (metadata & 0x3).astype(np.intp)
    keep_second = ((metadata >> 2) & 0x3).astype(np.intp)

    # Extract kept values
    val0 = sparse_weights[0::2]  # [num_blocks, N]
    val1 = sparse_weights[1::2]  # [num_blocks, N]

    # Scatter back into dense blocks
    dense = np.zeros((num_blocks, 4, N), dtype=sparse_weights.dtype)

    block_idx = np.arange(num_blocks)[:, None]
    n_idx = np.arange(N)[None, :]

    dense[block_idx, keep_first, n_idx] = val0
    dense[block_idx, keep_second, n_idx] = val1

    return dense.reshape(K, N)


def measure_pruning_loss(
    original: np.ndarray,
    pruned: np.ndarray,
) -> dict[str, float]:
    """Measure information loss from 2:4 structured pruning.

    Compares the original dense weight matrix against the pruned-then-
    reconstructed version (with zeros at pruned positions).

    Args:
        original: [K, N] original weight matrix.
        pruned: [K, N] reconstructed weight matrix (from unprune_2_4).

    Returns:
        Dictionary with:
            mse: Mean squared error between original and pruned.
            rmse: Root mean squared error.
            relative_error: ||original - pruned||_F / ||original||_F.
            sparsity: Fraction of zero elements in pruned matrix.
            max_abs_error: Maximum absolute element-wise error.
    """
    orig_f32 = original.astype(np.float32)
    pruned_f32 = pruned.astype(np.float32)

    diff = orig_f32 - pruned_f32
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))

    orig_norm = float(np.linalg.norm(orig_f32))
    diff_norm = float(np.linalg.norm(diff))
    relative_error = diff_norm / orig_norm if orig_norm > 0 else float("inf")

    num_zeros = int(np.sum(pruned_f32 == 0))
    sparsity = num_zeros / pruned_f32.size

    max_abs_error = float(np.max(np.abs(diff)))

    return {
        "mse": mse,
        "rmse": rmse,
        "relative_error": relative_error,
        "sparsity": sparsity,
        "max_abs_error": max_abs_error,
    }
