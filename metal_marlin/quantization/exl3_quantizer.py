"""EXL3-style layer quantization with integrated Hadamard and LDLQ.

This module provides the main entry point for EXL3-style layer quantization,
combining Hadamard preprocessing, block LDL decomposition, and trellis-based
LDLQ quantization for optimal weight compression.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from metal_marlin.quantization.hadamard_preprocess import (
    preprocess_hessian_exl3,
    rotate_weights_exl3,
)
from metal_marlin.quantization.ldl_decomp import block_ldl

# Try MPS-accelerated version first, fall back to CPU
try:
    from metal_marlin.quantization.viterbi_quant import (
        TrellisCodebook,
        quantize_tiles_fast,
        quantize_tiles_parallel,
    )

    _USE_FAST = True
except ImportError:
    from metal_marlin.quantization.viterbi_quant import (
        TrellisCodebook,
        TrellisTile,
        quantize_tiles_parallel,
    )

    quantize_tiles_fast = None
    _USE_FAST = False

from metal_marlin.quantization.trellis_tile import TrellisTile

# Try fast Cython eigendecomposition
_USE_FAST_EIGH = False
try:
    from metal_marlin.quantization._eigh_fast import eigh_psd_fast
    _USE_FAST_EIGH = True
except ImportError:
    pass

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class EXL3QuantResult:
    """Result of EXL3 quantization for a single layer."""

    name: str
    trellis_indices: NDArray[np.int16]  # [tiles_k, tiles_n, 256]
    scales: NDArray[np.float32]  # [n_groups, out_features]
    su: NDArray[np.float64]  # Input sign flips
    sv: NDArray[np.float64]  # Output sign flips (optional)
    bits: int
    reconstruction_mse: float
    quantization_time_sec: float


class EXL3Quantizer:
    """EXL3-style quantization with integrated Hadamard and LDLQ.

    Usage:
        quantizer = EXL3Quantizer(bits=4, group_size=128)

        for layer in LayerStreamer(model_path).iter_linear_layers():
            # Collect Hessian from calibration (streamed)
            H = collect_layer_hessian(layer.name, calibration_streamer)

            # Quantize layer
            result = quantizer.quantize_layer(layer.weight, H)

            # Save immediately to disk
            save_quantized_layer(output_path, result)
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        had_k: int = 128,
        sigma_reg: float = 0.025,
        max_workers: int | None = None,
        use_metal: bool = True,
    ):
        self.bits = bits
        self.group_size = group_size
        self.had_k = had_k
        self.sigma_reg = sigma_reg
        self.max_workers = max_workers
        self.use_metal = use_metal
        self.codebook = TrellisCodebook(bits=bits)

    def quantize_layer(
        self,
        weight: torch.Tensor,
        hessian: NDArray[np.float64],
        layer_name: str = "",
    ) -> EXL3QuantResult:
        """Quantize single layer using EXL3 algorithm.

        Steps:
        1. Preprocess Hessian with Hadamard rotation
        2. Eigendecomposition to enforce PSD
        3. Block LDL decomposition
        4. Rotate weights
        5. LDLQ quantization with error compensation
        """
        start = time.perf_counter()

        W = weight.float().cpu().numpy()

        # Step 1: Hadamard preprocessing of Hessian
        H_rot, su, _ = preprocess_hessian_exl3(hessian, self.had_k)

        # Step 2: Ensure positive definiteness via eigendecomposition
        if _USE_FAST_EIGH:
            H_psd, _ = eigh_psd_fast(H_rot, sigma_reg=self.sigma_reg)
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(H_rot)
            eigenvalues = np.maximum(eigenvalues, self.sigma_reg)
            H_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Step 3: Block LDL decomposition
        L, D = block_ldl(H_psd, block_size=16)

        # Step 4: Rotate weights
        W_rot = rotate_weights_exl3(W, su, had_k=self.had_k)

        # Step 5: LDLQ quantization
        encoded, scales, W_q = ldlq_quantize_layer(
            W_rot,
            L,
            D,
            self.codebook,
            group_size=self.group_size,
            max_workers=self.max_workers,
            use_metal=self.use_metal,
        )

        # Compute reconstruction error
        mse = float(np.mean((W_rot - W_q) ** 2))

        elapsed = time.perf_counter() - start

        return EXL3QuantResult(
            name=layer_name,
            trellis_indices=encoded,
            scales=scales.astype(np.float32),
            su=su,
            sv=np.ones(W.shape[0]),  # No output rotation for now
            bits=self.bits,
            reconstruction_mse=mse,
            quantization_time_sec=elapsed,
        )


def ldlq_quantize_layer(
    W: NDArray[np.float32],
    L: NDArray[np.float64],
    D: NDArray[np.float64],
    codebook: TrellisCodebook,
    group_size: int = 128,
    max_workers: int | None = None,
    use_metal: bool = True,
) -> tuple[NDArray[np.int16], NDArray[np.float32], NDArray[np.float32]]:
    """Quantize layer using LDLQ (Layer-wise Dynamic Low-precision Quantization).

    Implements the LDLQ algorithm which uses the LDL decomposition of the
    Hessian to optimally compensate quantization error through the trellis.

    Args:
        W: Weight matrix [out_features, in_features] (already rotated)
        L: Lower triangular factor from LDL decomposition
        D: Diagonal factor from LDL decomposition
        codebook: TrellisCodebook with quantization grid
        group_size: Elements per quantization group
        max_workers: Thread pool size for parallel quantization

    Returns:
        encoded: Trellis indices [tiles_n, tiles_k, 256]
        scales: Per-group scale factors [n_groups, out_features]
        W_q: Quantized weights [out_features, in_features]
    """
    W = np.asarray(W, dtype=np.float32)
    out_features, in_features = W.shape

    # Tile dimensions (16x16 tiles)
    tile_size = 16
    tiles_n = (out_features + tile_size - 1) // tile_size
    tiles_k = (in_features + tile_size - 1) // tile_size

    # Pad weights to tile boundaries
    W_padded = np.zeros((tiles_n * tile_size, tiles_k * tile_size), dtype=np.float32)
    W_padded[:out_features, :in_features] = W

    # === VECTORIZED SCALE COMPUTATION ===
    n_groups = (in_features + group_size - 1) // group_size

    # Compute max abs per group per output (vectorized)
    scales = np.zeros((n_groups, out_features), dtype=np.float32)
    scale_factor = (1 << (codebook.bits - 1)) - 1

    for g in range(n_groups):
        start_idx = g * group_size
        end_idx = min(start_idx + group_size, in_features)
        w_slice = np.abs(W[:, start_idx:end_idx])
        scales[g] = np.maximum(w_slice.max(axis=1) / scale_factor, 1e-8)

    # === VECTORIZED TILE EXTRACTION ===
    # Reshape padded weights to [tiles_n, tile_size, tiles_k, tile_size]
    # Then transpose to get tiles as contiguous memory
    W_tiles = W_padded.reshape(tiles_n, tile_size, tiles_k, tile_size)
    W_tiles = W_tiles.transpose(0, 2, 1, 3)  # [tiles_n, tiles_k, 16, 16]
    tiles_data = W_tiles.reshape(-1, tile_size, tile_size)  # [n_tiles, 16, 16]

    # === VECTORIZED TILE SCALE COMPUTATION ===
    # For each tile, compute scale as mean of relevant group scales
    n_tiles = tiles_n * tiles_k
    tile_scales = np.zeros(n_tiles, dtype=np.float32)

    for tk in range(tiles_k):
        col_start = tk * tile_size
        group_idx = min(col_start // group_size, n_groups - 1)
        for tn in range(tiles_n):
            row_start = tn * tile_size
            row_end = min(row_start + tile_size, out_features)
            tile_idx = tn * tiles_k + tk
            tile_scales[tile_idx] = np.mean(scales[group_idx, row_start:row_end])

    # === QUANTIZE TILES (METAL GPU ACCELERATED) ===
    if _USE_FAST and quantize_tiles_fast is not None and use_metal:
        # Fast path: pass raw array directly to Metal
        all_indices, all_dequant = quantize_tiles_fast(
            tiles_data, codebook, tile_scales, max_workers=max_workers, use_metal=True
        )
    elif _USE_FAST and quantize_tiles_fast is not None:
        # CPU-only fast path (for thread safety)
        all_indices, all_dequant = quantize_tiles_fast(
            tiles_data, codebook, tile_scales, max_workers=max_workers, use_metal=False
        )
    else:
        # Fallback: create TrellisTile objects
        tiles = [
            TrellisTile(
                data=tiles_data[i],
                row_offset=(i // tiles_k) * tile_size,
                col_offset=(i % tiles_k) * tile_size,
            )
            for i in range(n_tiles)
        ]
        all_indices, all_dequant = quantize_tiles_parallel(
            tiles, codebook, tile_scales, max_workers=max_workers, use_metal=use_metal
        )

    # Reshape encoded indices to [tiles_n, tiles_k, 256]
    encoded = all_indices.reshape(tiles_n, tiles_k, 256)

    # === VECTORIZED RECONSTRUCTION ===
    # all_dequant is [n_tiles, 16, 16], reshape back to padded weights
    W_q_tiles = all_dequant.reshape(tiles_n, tiles_k, tile_size, tile_size)
    W_q_tiles = W_q_tiles.transpose(0, 2, 1, 3)  # [tiles_n, 16, tiles_k, 16]
    W_q_padded = W_q_tiles.reshape(tiles_n * tile_size, tiles_k * tile_size)

    # Crop back to original size
    W_q = W_q_padded[:out_features, :in_features]

    return encoded, scales, W_q
