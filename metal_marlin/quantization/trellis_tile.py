"""16x16 tile structure for trellis quantization.

EXL3 processes weights in 16x16 tiles (256 elements) which map
directly to tensor core operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class TrellisTile:
    """16x16 weight tile for trellis quantization.

    EXL3 processes weights in 16x16 tiles (256 elements) which map
    directly to tensor core operations.
    """
    data: NDArray[np.float32]  # [16, 16] tile
    row_offset: int
    col_offset: int

    def __post_init__(self) -> None:
        """Validate tile dimensions."""
        if self.data.shape != (16, 16):
            raise ValueError(f"Tile data must have shape (16, 16), got {self.data.shape}")
        if self.data.dtype != np.float32:
            raise TypeError(f"Tile data must be float32, got {self.data.dtype}")

    @staticmethod
    def extract_tiles(weights: NDArray, tile_size: int = 16) -> list[TrellisTile]:
        """Extract tiles from weight matrix, padding if needed.

        Args:
            weights: Weight matrix of any shape [rows, cols]
            tile_size: Size of each tile (default 16)

        Returns:
            List of TrellisTile objects covering the matrix
        """
        if weights.ndim != 2:
            raise ValueError(f"weights must be 2D, got {weights.ndim}D")

        rows, cols = weights.shape

        # Calculate number of tiles in each dimension (with padding)
        n_row_tiles = (rows + tile_size - 1) // tile_size
        n_col_tiles = (cols + tile_size - 1) // tile_size

        # Pad weights if necessary
        pad_rows = n_row_tiles * tile_size - rows
        pad_cols = n_col_tiles * tile_size - cols

        if pad_rows > 0 or pad_cols > 0:
            weights_padded = np.pad(
                weights,
                ((0, pad_rows), (0, pad_cols)),
                mode='constant',
                constant_values=0
            )
        else:
            weights_padded = weights

        # Cast to float32 if needed
        if weights_padded.dtype != np.float32:
            weights_padded = weights_padded.astype(np.float32)

        tiles: list[TrellisTile] = []
        for i in range(n_row_tiles):
            for j in range(n_col_tiles):
                row_start = i * tile_size
                col_start = j * tile_size
                tile_data = weights_padded[row_start:row_start + tile_size,
                                          col_start:col_start + tile_size]
                tiles.append(TrellisTile(
                    data=tile_data.copy(),
                    row_offset=row_start,
                    col_offset=col_start
                ))

        return tiles

    @staticmethod
    def reconstruct(tiles: list[TrellisTile], shape: tuple[int, int]) -> NDArray:
        """Reconstruct weight matrix from tiles.

        Args:
            tiles: List of TrellisTile objects
            shape: Original shape (rows, cols) of the weight matrix

        Returns:
            Reconstructed weight matrix
        """
        if not tiles:
            raise ValueError("tiles list cannot be empty")

        rows, cols = shape

        # Determine padded shape from tiles
        max_row = max(t.row_offset for t in tiles) + 16
        max_col = max(t.col_offset for t in tiles) + 16

        # Reconstruct padded matrix
        reconstructed = np.zeros((max_row, max_col), dtype=np.float32)

        for tile in tiles:
            r, c = tile.row_offset, tile.col_offset
            reconstructed[r:r + 16, c:c + 16] = tile.data

        # Trim padding to get original shape
        return reconstructed[:rows, :cols]


def tensor_core_perm(device: str = "metal") -> NDArray[np.int64]:
    """Get tensor core permutation for kernel layout.

    Permutes 256-element tile (16x16) to tensor core-friendly layout.
    This reordering optimizes memory access patterns for tensor core operations.

    Args:
        device: Target device ("metal", "cuda", etc.)

    Returns:
        Permutation indices [256] mapping flat index -> permuted index

    Note:
        The permutation pattern follows EXL3's approach where elements
        are reordered to match tensor core SIMD group layouts.
    """
    # Standard tensor core permutation for 16x16 tiles
    # Maps row-major to tensor core-friendly layout
    perm = np.zeros(256, dtype=np.int64)

    if device.lower() in ("metal", "cuda"):
        # For Metal/CUDA: interleave in groups of 8 for optimal SIMD
        # Pattern: process in 8x8 sub-blocks with specific swizzling
        idx = 0
        for block_row in range(2):  # 2 blocks vertically (16/8)
            for block_col in range(2):  # 2 blocks horizontally (16/8)
                for inner_row in range(8):
                    for inner_col in range(8):
                        src_row = block_row * 8 + inner_row
                        src_col = block_col * 8 + inner_col
                        src_idx = src_row * 16 + src_col
                        perm[idx] = src_idx
                        idx += 1
    else:
        # Default: identity permutation (row-major)
        perm = np.arange(256, dtype=np.int64)

    return perm


def tensor_core_perm_i(device: str = "metal") -> NDArray[np.int64]:
    """Get inverse tensor core permutation.

    This is the inverse of tensor_core_perm, used to convert from
    tensor core layout back to row-major.

    Args:
        device: Target device ("metal", "cuda", etc.)

    Returns:
        Inverse permutation indices [256] mapping permuted -> flat index
    """
    perm = tensor_core_perm(device)
    # Inverse permutation: argsort gives us the inverse mapping
    inv_perm = np.zeros_like(perm)
    inv_perm[perm] = np.arange(len(perm))
    return inv_perm


def apply_tensor_core_perm(
    tile: NDArray[np.float32],
    device: str = "metal"
) -> NDArray[np.float32]:
    """Apply tensor core permutation to a 16x16 tile.

    Args:
        tile: Input tile [16, 16]
        device: Target device

    Returns:
        Permuted tile [16, 16] in tensor core layout
    """
    if tile.shape != (16, 16):
        raise ValueError(f"tile must be (16, 16), got {tile.shape}")

    flat = tile.reshape(-1)
    perm = tensor_core_perm(device)
    return flat[perm].reshape(16, 16)


def apply_tensor_core_perm_i(
    tile: NDArray[np.float32],
    device: str = "metal"
) -> NDArray[np.float32]:
    """Apply inverse tensor core permutation to restore row-major.

    Args:
        tile: Input tile in tensor core layout [16, 16]
        device: Target device

    Returns:
        Tile in row-major layout [16, 16]
    """
    if tile.shape != (16, 16):
        raise ValueError(f"tile must be (16, 16), got {tile.shape}")

    flat = tile.reshape(-1)
    inv_perm = tensor_core_perm_i(device)
    return flat[inv_perm].reshape(16, 16)
