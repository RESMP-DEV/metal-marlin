"""LDLQ: LDL-based quantization with block error compensation.

This module implements the LDLQ algorithm from ExllamaV3, which uses
block-wise quantization with error propagation through the LDL decomposition
of the Hessian matrix.

The key insight is to process weight rows in reverse order (bottom-up)
so that quantization error from already-processed rows can be compensated
for in the remaining rows using the L factor from the LDL decomposition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from metal_marlin.quantization.viterbi_quant import (
    TrellisCodebook,
    TrellisTile,
    quantize_tiles_parallel,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_group_scales(
    weight: NDArray[np.float32],
    group_size: int,
    codebook: TrellisCodebook,
) -> NDArray[np.float32]:
    """Compute per-group quantization scales.

    Scales are computed as the max absolute value in each group
    divided by the max grid value, ensuring all quantized values
    fit within the codebook range.

    Args:
        weight: [out_features, in_features] weight matrix
        group_size: Number of columns per scale group
        codebook: TrellisCodebook for quantization grid

    Returns:
        scales: [n_groups, out_features] per-group scales
    """
    out_feat, in_feat = weight.shape
    n_groups = (in_feat + group_size - 1) // group_size

    # Compute max grid value for normalization
    grid = codebook.get_grid()
    max_grid = float(np.max(np.abs(grid)))
    eps = 1e-8  # Prevent division by zero

    scales = np.zeros((n_groups, out_feat), dtype=np.float32)

    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, in_feat)

        # Max absolute value in this group for each output row
        group_max = np.max(np.abs(weight[:, start:end]), axis=1)

        # Scale is max(abs(weight)) / max(abs(grid))
        scales[g] = group_max / (max_grid + eps)

    return scales


def compute_tile_scales(
    tiles: list[TrellisTile],
    codebook: TrellisCodebook,
) -> NDArray[np.float32]:
    """Compute per-tile scales from tile data.

    Scale is computed as max(abs(tile_data)) / max(abs(grid))
    to ensure all quantized values fit within the codebook range.

    Args:
        tiles: List of TrellisTile objects
        codebook: TrellisCodebook for quantization grid

    Returns:
        tile_scales: [n_tiles] per-tile scales
    """
    grid = codebook.get_grid()
    max_grid = float(np.max(np.abs(grid)))
    eps = 1e-8

    tile_scales = np.zeros(len(tiles), dtype=np.float32)
    for i, tile in enumerate(tiles):
        tile_max = float(np.max(np.abs(tile.data)))
        tile_scales[i] = tile_max / (max_grid + eps)

    return tile_scales


def pack_indices(indices: NDArray[np.int16]) -> NDArray[np.int16]:
    """Pack tile indices into encoded format.

    The encoded format stores indices as [tiles_k, tiles_n, 256]
    where tiles_k is the number of tiles along K dimension (in_features)
    and tiles_n is the number of tiles along N dimension (out_features).

    Args:
        indices: [n_tiles, 256] quantized indices from quantize_tiles_parallel

    Returns:
        packed: [tiles_k, tiles_n, 256] packed indices
    """
    n_tiles, tile_size = indices.shape
    assert tile_size == 256, f"Expected tile size 256, got {tile_size}"

    # Reshape to [tiles_k, tiles_n, 256]
    # Assuming square-ish tiling: tiles_k = tiles_n = sqrt(n_tiles)
    tiles_per_dim = int(np.sqrt(n_tiles))
    if tiles_per_dim * tiles_per_dim != n_tiles:
        # Not a perfect square, keep flat and let caller handle
        return indices

    return indices.reshape(tiles_per_dim, tiles_per_dim, 256)


def extract_tiles_from_rows(
    rows: NDArray[np.float32],
    col_offset: int = 0,
) -> list[TrellisTile]:
    """Extract 16x16 tiles from row data.

    Args:
        rows: [out_features, n_cols] row data where n_cols is multiple of 16
        col_offset: Column offset for tile positioning

    Returns:
        tiles: List of TrellisTile objects
    """
    out_feat, n_cols = rows.shape
    assert n_cols % 16 == 0, f"n_cols must be multiple of 16, got {n_cols}"

    n_tiles_n = out_feat // 16

    tiles = []
    for tn in range(n_tiles_n):
        # Extract 16x16 tile
        row_start = tn * 16
        row_end = row_start + 16

        tile_data = rows[row_start:row_end, :]  # [16, 16]
        tiles.append(TrellisTile(
            data=tile_data.astype(np.float32),
            row_offset=row_start,
            col_offset=col_offset,
        ))

    return tiles


def ldlq_quantize_layer(
    weight: NDArray[np.float32],
    L: NDArray[np.float64],  # noqa: N803
    D: NDArray[np.float64],  # noqa: N803
    codebook: TrellisCodebook,
    group_size: int = 128,
    buf_size_k: int = 128,
    max_workers: int | None = None,
) -> tuple[NDArray[np.int16], NDArray[np.float32], NDArray[np.float32]]:
    """LDLQ: LDL-based quantization with block error compensation.

    Processes weight rows in reverse order (bottom-up) for proper
    error propagation, matching ExllamaV3 ldlq() implementation.

    Key insight: Process in 16-row blocks that match tile size.
    Error from already-quantized rows is propagated via L factor.

    Args:
        weight: [out_features, in_features] weight matrix
        L: Lower triangular factor from block_ldl
        D: Diagonal factor from block_ldl
        codebook: TrellisCodebook for quantization grid
        group_size: Scale group size
        buf_size_k: Processing buffer size (>= 16)
        max_workers: Parallel workers for tile quantization

    Returns:
        encoded: [tiles_k, tiles_n, 256] trellis indices
        scales: [n_groups, out_features] per-group scales
        weight_q: [out_features, in_features] dequantized weights
    """
    out_feat, in_feat = weight.shape
    tiles_k = in_feat // 16
    tiles_n = out_feat // 16

    # Initialize outputs
    encoded = np.zeros((tiles_k, tiles_n, 256), dtype=np.int16)
    weight_q = np.zeros_like(weight)
    prod_cache = np.zeros_like(weight, dtype=np.float64)

    # Compute scales per group
    scales = compute_group_scales(weight, group_size, codebook)

    # Process rows in reverse order (bottom-up)
    for j in range(in_feat, 0, -buf_size_k):
        i = max(0, j - buf_size_k)

        b_weight = weight[:, i:j].copy()
        b_L = L[i:j, :]  # noqa: N806

        # Process 16-row blocks within current span
        for bj in range(j - i, 0, -16):
            bi = bj - 16

            # Error compensation from already-quantized rows
            if bj < j - i:
                bb_err = b_weight[:, bj:] - weight_q[:, i + bj:j]
                bb_L = b_L[bj:, i + bi:i + bj]  # noqa: N806
                compensation = prod_cache[:, bi:bj]
                compensation += bb_err @ bb_L  # [out_feat, remaining] @ [remaining, 16]

            # Rows to quantize with compensation
            rows = b_weight[:, bi:bj] + prod_cache[:, bi:bj].astype(np.float32)

            # Extract and quantize tiles
            # rows shape: [out_feat, 16] where out_feat = tiles_n * 16
            tiles = []
            for tn in range(tiles_n):
                row_start = tn * 16
                row_end = row_start + 16
                tile_data = rows[row_start:row_end, :]  # [16, 16]
                tiles.append(TrellisTile(
                    data=tile_data.astype(np.float32),
                    row_offset=row_start,
                    col_offset=i + bi,
                ))

            tile_scales = compute_tile_scales(tiles, codebook)

            indices, dequant = quantize_tiles_parallel(
                tiles, codebook, tile_scales, max_workers
            )

            # Store results
            # dequant shape: [n_tiles, 16, 16]
            for tn in range(tiles_n):
                row_start = tn * 16
                row_end = row_start + 16
                weight_q[row_start:row_end, i + bi:i + bj] = dequant[tn]

            # Pack indices into encoded array
            # indices shape: [n_tiles, 256]
            tile_idx_k = (i + bi) // 16
            for tn in range(tiles_n):
                encoded[tile_idx_k, tn, :] = indices[tn]

        # Update prod_cache for remaining rows
        b_err = b_weight - weight_q[:, i:j]
        prod_cache[:, i:j] += (b_err @ b_L[:, i:j]).astype(np.float64)

    return encoded, scales, weight_q
