"""Viterbi algorithm for trellis quantization.

Implements optimal path search through trellis state space for
quantizing 16x16 weight tiles, following EXL3's trellis encoding approach.

Uses Metal GPU acceleration when available (Apple Silicon), falls back to CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from metal_marlin.quantization.trellis_tile import TrellisTile

# Check for Metal availability
_HAS_METAL = False
_metal_lib = None

try:
    from metal_marlin.metal_dispatch import (
        HAS_METAL,
        HAS_MPS,
        dispatch_viterbi_quantize,
        dispatch_viterbi_quantize_naive,
        get_default_library,
    )

    if HAS_METAL and HAS_MPS:
        _HAS_METAL = True
except ImportError:
    pass


def _get_metal_lib():
    """Get or create Metal library singleton."""
    global _metal_lib
    if _metal_lib is None and _HAS_METAL:
        _metal_lib = get_default_library()
    return _metal_lib


@dataclass
class TrellisCodebook:
    """Codebook for trellis quantization.

    Defines the quantization grid and bit width for trellis encoding.
    Based on EXL3's multi-component grid (MCG) and multiplicative codebooks.

    Attributes:
        bits: Bits per weight (1-8)
        grid: Quantization levels (uniform or learned)
        mcg: Whether to use multi-component grid
        mul1: Whether to use multiplicative codebook
    """

    bits: int
    grid: NDArray[np.float32] | None = None
    mcg: bool = False
    mul1: bool = False

    def __post_init__(self) -> None:
        """Initialize default grid if not provided."""
        if self.grid is None:
            # Uniform symmetric grid centered at 0
            n_levels = 2**self.bits
            self.grid = np.linspace(
                -(n_levels - 1) / 2, (n_levels - 1) / 2, n_levels, dtype=np.float32
            )

    def get_grid(self) -> NDArray[np.float32]:
        """Get quantization grid values.

        Returns:
            Array of quantization levels
        """
        if self.grid is None:
            raise ValueError("Grid not initialized")
        return self.grid


def quantize_tile_viterbi(
    tile: NDArray[np.float32],
    codebook: TrellisCodebook,
    scale: float,
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Quantize 16x16 tile using Viterbi trellis search.

    Viterbi finds the optimal sequence of quantization decisions
    that minimizes total error across the tile.

    Args:
        tile: [16, 16] weight tile (flattened to 256 for processing)
        codebook: TrellisCodebook with grid and bit width
        scale: Scale factor for this tile

    Returns:
        indices: [256] quantized indices (uint8 for packed storage)
        dequantized: [16, 16] reconstructed tile
    """
    grid = codebook.get_grid()
    n_states = len(grid)

    # Flatten tile to 1D array and normalize
    tile_flat = tile.reshape(-1) / scale
    n_elements = len(tile_flat)

    # Viterbi with uniform transitions simplifies to tracking min cost path
    # Since transition costs are uniform, best_prev is independent of current state
    costs = np.zeros((n_elements, n_states), dtype=np.float32)
    edges = np.zeros(n_elements, dtype=np.uint8)  # best_prev is same for all states

    # Initialize first element (vectorized over all states)
    costs[0] = (tile_flat[0] - grid) ** 2

    # Forward pass: fully vectorized per position
    for i in range(1, n_elements):
        min_prev_cost = np.min(costs[i - 1])
        edges[i] = np.argmin(costs[i - 1])
        # Vectorized cost computation for all states
        costs[i] = min_prev_cost + (tile_flat[i] - grid) ** 2

    # Backtrack to find optimal path
    indices = np.zeros(n_elements, dtype=np.uint8)
    indices[-1] = np.argmin(costs[-1])
    for i in range(n_elements - 2, -1, -1):
        indices[i] = edges[i + 1]

    # Reconstruct
    dequantized = (grid[indices] * scale).reshape(16, 16)

    return indices, dequantized


def quantize_tiles_parallel(
    tiles: list[TrellisTile],
    codebook: TrellisCodebook,
    scales: NDArray[np.float32],
    max_workers: int | None = None,
    use_metal: bool = True,
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Quantize multiple tiles in parallel.

    Uses Metal GPU acceleration when available (300k+ tiles/sec on M4),
    falls back to CPU thread pool otherwise.

    Args:
        tiles: List of TrellisTile objects
        codebook: Shared codebook
        scales: Per-tile scales [n_tiles]
        max_workers: Thread pool size for CPU fallback (None = cpu_count)
        use_metal: Use Metal GPU if available (default True)

    Returns:
        all_indices: [n_tiles, 256] packed indices
        all_dequant: [n_tiles, 16, 16] reconstructed tiles
    """
    # Extract data from tiles (allows object list or raw array)
    tiles_np = np.stack([t.data.reshape(256) for t in tiles], axis=0)
    return quantize_tiles_fast(tiles_np, codebook, scales, max_workers, use_metal)


def quantize_tiles_fast(
    tiles_data: NDArray[np.float32],
    codebook: TrellisCodebook,
    scales: NDArray[np.float32],
    max_workers: int | None = None,
    use_metal: bool = True,
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Quantize tiles from raw ndarray (fast path, no TrellisTile overhead).

    Uses Metal GPU acceleration when available (300k+ tiles/sec on M4),
    falls back to CPU thread pool otherwise.

    Args:
        tiles_data: Raw tile data [n_tiles, 256] or [n_tiles, 16, 16], float32
        codebook: Shared codebook
        scales: Per-tile scales [n_tiles]
        max_workers: Thread pool size for CPU fallback (None = cpu_count)
        use_metal: Use Metal GPU if available (default True)

    Returns:
        all_indices: [n_tiles, 256] packed indices
        all_dequant: [n_tiles, 16, 16] reconstructed tiles
    """
    # Reshape to [n_tiles, 256] if needed
    if tiles_data.ndim == 3:
        n_tiles = tiles_data.shape[0]
        tiles_flat = tiles_data.reshape(n_tiles, 256)
    else:
        n_tiles = tiles_data.shape[0]
        tiles_flat = tiles_data

    # Try Metal GPU acceleration
    if use_metal and _HAS_METAL:
        lib = _get_metal_lib()
        if lib is not None:
            tiles_tensor = torch.from_numpy(tiles_flat.astype(np.float32)).to("mps")
            scales_tensor = torch.from_numpy(scales.astype(np.float32)).to("mps")
            grid_tensor = torch.from_numpy(codebook.get_grid()).float().to("mps")

            # Dispatch to Metal
            indices_tensor, dequant_tensor = dispatch_viterbi_quantize(
                lib,
                tiles_tensor,
                scales_tensor,
                grid_tensor,
                use_u4_kernel=(codebook.bits == 4),
            )

            # Convert back to numpy
            all_indices = indices_tensor.cpu().numpy().astype(np.uint8)
            all_dequant = dequant_tensor.cpu().numpy().reshape(n_tiles, 16, 16)

            return all_indices, all_dequant

    # CPU fallback - batched vectorized Viterbi (fast, GIL-friendly)
    grid = codebook.get_grid()
    n_states = len(grid)

    # Normalize all tiles at once: [n_tiles, 256] / [n_tiles, 1]
    tiles_norm = tiles_flat / scales[:, None]

    # Batched Viterbi forward pass
    # costs: [n_tiles, n_states] - cost to reach each state at current position
    costs = (tiles_norm[:, 0:1] - grid[None, :]) ** 2  # [n_tiles, n_states]

    # Track edges for backtracking: [n_tiles, 256]
    edges = np.zeros((n_tiles, 256), dtype=np.uint8)

    # Forward pass
    for i in range(1, 256):
        min_prev = costs.min(axis=1, keepdims=True)  # [n_tiles, 1]
        edges[:, i] = costs.argmin(axis=1)  # [n_tiles]
        costs = min_prev + (tiles_norm[:, i : i + 1] - grid[None, :]) ** 2

    # Backtrack: all tiles in parallel
    all_indices = np.zeros((n_tiles, 256), dtype=np.uint8)
    all_indices[:, -1] = costs.argmin(axis=1)
    for i in range(254, -1, -1):
        all_indices[:, i] = edges[:, i + 1]

    # Reconstruct
    all_dequant = (grid[all_indices] * scales[:, None]).reshape(n_tiles, 16, 16)

    return all_indices, all_dequant


def quantize_tile_greedy(
    tile: NDArray[np.float32],
    codebook: TrellisCodebook,
    scale: float,
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Quantize tile using greedy nearest-neighbor (baseline comparison).

    Args:
        tile: [16, 16] weight tile
        codebook: TrellisCodebook with grid
        scale: Scale factor for this tile

    Returns:
        indices: [256] quantized indices
        dequantized: [16, 16] reconstructed tile
    """
    grid = codebook.get_grid()

    # Flatten and normalize
    tile_flat = tile.reshape(-1)
    normalized = tile_flat / scale

    # Find nearest grid point for each element independently
    indices = np.zeros(len(tile_flat), dtype=np.uint8)
    for i, val in enumerate(normalized):
        indices[i] = np.argmin(np.abs(grid - val))

    # Reconstruct
    dequantized = (grid[indices] * scale).reshape(16, 16)

    return indices, dequantized


def compute_quantization_error(
    original: NDArray[np.float32],
    dequantized: NDArray[np.float32],
) -> float:
    """Compute MSE between original and dequantized tile.

    Args:
        original: Original weight tile [16, 16]
        dequantized: Reconstructed tile [16, 16]

    Returns:
        Mean squared error
    """
    return float(np.mean((original - dequantized) ** 2))
