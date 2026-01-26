"""
MR-GPTQ: Marlin-Replica GPTQ = Hadamard Rotation + GPTQ Quantization.

This module implements the MR-GPTQ quantization pipeline, combining:
1. Hadamard rotation for outlier dispersal (QuaRot-inspired)
2. GPTQ algorithm with Hessian-aware error compensation
3. Marlin FP4 packing for Metal GPU inference

The Hadamard transform disperses outlier activations across all dimensions,
making the weight distribution more amenable to low-bit quantization. GPTQ
then performs optimal per-column quantization with error propagation to
minimize reconstruction error on calibration data.

Usage:
    from metal_marlin.mr_gptq import MRGPTQQuantizer, QuantizationFormat
    from metal_marlin.calibration import CalibrationDataset

    # Create quantizer
    quantizer = MRGPTQQuantizer(
        bits=4,
        format=QuantizationFormat.FP4,
        group_size=128,
        use_hadamard=True,
        hadamard_block_size=64,
        actorder=True,
    )

    # Load calibration dataset
    calibration = CalibrationDataset.v3()

    # Quantize model
    report = quantizer.quantize_model(
        model_path="path/to/model",
        calibration_data=calibration,
        output_path="path/to/output",
    )

Reference:
    - GPTQ: arxiv.org/abs/2210.17323
    - QuaRot: arxiv.org/abs/2404.00456 (Hadamard rotation for outlier dispersal)
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .calibration import CalibrationDataset


# =============================================================================
# Quantization format configuration
# =============================================================================


class QuantizationFormat(str, Enum):
    """Supported quantization formats for GPTQ."""

    FP4 = "fp4"  # FP4 E2M1 (Marlin native)
    INT4 = "int4"  # INT4 symmetric
    NF4 = "nf4"  # NormalFloat 4-bit


# FP4 E2M1 representable values (same as quantize_fp4.py)
FP4_E2M1_GRID = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,  # Positive values (codes 0-7)
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,  # Negative values (codes 8-15)
    ],
    dtype=np.float32,
)

# INT4 symmetric grid: [-8, -7, ..., 7] (15 values, 0 is shared)
INT4_GRID = np.arange(-8, 8, dtype=np.float32)

# NF4 grid: Gaussian quantiles optimized for normal distributions
# These are pre-computed for N(0,1) and scaled by absmax during quantization
_NF4_QUANTILES = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=np.float32,
)
NF4_GRID = _NF4_QUANTILES


# =============================================================================
# Hadamard Transform
# =============================================================================


def hadamard_matrix(n: int) -> NDArray[np.float32]:
    """
    Generate normalized Hadamard matrix of size n x n using Sylvester construction.

    The Hadamard matrix H_n has the property H @ H^T = n * I.
    We return H / sqrt(n) so the transform is orthonormal: H @ H^T = I.

    Args:
        n: Matrix size (must be power of 2)

    Returns:
        Orthonormal Hadamard matrix of shape [n, n]

    Raises:
        ValueError: If n is not a power of 2
    """
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard size must be power of 2, got {n}")

    # Start with H_1 = [[1]]
    h = np.array([[1.0]], dtype=np.float32)

    while h.shape[0] < n:
        # Sylvester construction: H_2n = [[H_n, H_n], [H_n, -H_n]]
        h = np.block([[h, h], [h, -h]])

    # Normalize to make orthonormal
    return h / np.sqrt(n)


def apply_hadamard_rotation(
    weights: NDArray[np.float32],
    block_size: int = 64,
    axis: int = 1,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """
    Apply block-diagonal Hadamard rotation to disperse outliers.

    For a weight matrix W of shape [out_features, in_features], the rotation
    is applied along the input dimension (axis=1) to disperse activation
    outliers across all input channels.

    The transformation is:
        W_rotated = W @ H_block_diag

    where H_block_diag is block-diagonal with Hadamard blocks of size block_size.

    Args:
        weights: Weight matrix [out_features, in_features]
        block_size: Size of each Hadamard block (must be power of 2)
        axis: Axis along which to apply rotation (0=output, 1=input)

    Returns:
        (rotated_weights, metadata) where metadata contains block_size and
        any padding information needed for inverse transform.
    """
    w = weights.astype(np.float32)
    out_feat, in_feat = w.shape

    # Determine dimension to rotate
    if axis == 1:
        dim = in_feat
    else:
        dim = out_feat
        w = w.T  # Transpose so we always rotate along axis 1

    # Pad dimension to multiple of block_size
    pad_needed = (block_size - (dim % block_size)) % block_size
    if pad_needed > 0:
        if axis == 1:
            w = np.pad(w, ((0, 0), (0, pad_needed)), mode="constant")
        else:
            w = np.pad(w, ((0, pad_needed), (0, 0)), mode="constant")

    dim_padded = dim + pad_needed
    n_blocks = dim_padded // block_size

    # Generate Hadamard matrix for one block
    h_block = hadamard_matrix(block_size)

    # Apply rotation block by block (more memory efficient than full block-diagonal)
    # W_rotated[:, i*bs:(i+1)*bs] = W[:, i*bs:(i+1)*bs] @ H_block
    w_rotated = np.zeros_like(w)
    for i in range(n_blocks):
        start = i * block_size
        end = (i + 1) * block_size
        w_rotated[:, start:end] = w[:, start:end] @ h_block

    # Transpose back if we rotated axis 0
    if axis == 0:
        w_rotated = w_rotated.T

    metadata = {
        "block_size": block_size,
        "original_dim": dim,
        "padded_dim": dim_padded,
        "axis": axis,
        "pad_needed": pad_needed,
    }

    return w_rotated, metadata


def inverse_hadamard_rotation(
    weights: NDArray[np.float32],
    metadata: dict[str, Any],
) -> NDArray[np.float32]:
    """
    Apply inverse Hadamard rotation to recover original weights.

    Since Hadamard is orthonormal, H^{-1} = H^T = H (self-inverse up to sign).
    We apply the same rotation to undo the transformation.

    Args:
        weights: Rotated weight matrix
        metadata: Rotation metadata from apply_hadamard_rotation

    Returns:
        Original (unrotated) weights with padding removed
    """
    block_size = metadata["block_size"]
    original_dim = metadata["original_dim"]
    axis = metadata["axis"]

    # Apply same rotation (Hadamard is self-inverse)
    w_restored, _ = apply_hadamard_rotation(weights, block_size=block_size, axis=axis)

    # Remove padding
    if axis == 1:
        w_restored = w_restored[:, :original_dim]
    else:
        w_restored = w_restored[:original_dim, :]

    return w_restored


# =============================================================================
# Hessian Collection
# =============================================================================


@dataclass
class HessianInfo:
    """Accumulated Hessian information for a layer."""

    # X^T @ X accumulated over calibration samples
    hessian: NDArray[np.float32]

    # Number of samples accumulated
    n_samples: int = 0

    # Diagonal for importance ordering
    diag: NDArray[np.float32] | None = None

    # Damping value used
    damp: float = 0.01


def collect_hessian_from_activations(
    activations: list[NDArray[np.float32]],
    damp_ratio: float = 0.01,
) -> HessianInfo:
    """
    Collect Hessian approximation from layer input activations.

    The Hessian H is approximated as:
        H ≈ (2/n) * Σ X_i^T @ X_i

    where X_i is the input activation for sample i. This is the Fisher
    information matrix for the squared error loss.

    Args:
        activations: List of input activations [batch, seq_len, in_features]
                    or [batch * seq_len, in_features] if already flattened
        damp_ratio: Damping ratio λ = damp_ratio * mean(diag(H))

    Returns:
        HessianInfo with accumulated Hessian and metadata
    """
    # Flatten and stack activations
    flat_acts = []
    for act in activations:
        if act.ndim == 3:
            # [batch, seq, hidden] -> [batch * seq, hidden]
            act = act.reshape(-1, act.shape[-1])
        flat_acts.append(act.astype(np.float32))

    X = np.concatenate(flat_acts, axis=0)  # [total_tokens, in_features]
    n_samples, in_features = X.shape

    # Compute H = X^T @ X (unnormalized Hessian)
    H = X.T @ X  # [in_features, in_features]

    # Normalize by number of samples
    H = H / n_samples

    # Compute diagonal for actorder
    diag = np.diag(H).copy()

    # Apply damping: H_damped = H + λI where λ = damp_ratio * mean(diag)
    damp = damp_ratio * np.mean(diag)
    H[np.diag_indices_from(H)] += damp

    return HessianInfo(
        hessian=H,
        n_samples=n_samples,
        diag=diag,
        damp=damp,
    )


# =============================================================================
# GPTQ Core Algorithm
# =============================================================================


def quantize_to_grid(
    values: NDArray[np.float32],
    grid: NDArray[np.float32],
    scale: float,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """
    Quantize values to nearest grid point.

    Args:
        values: Values to quantize
        grid: Quantization grid (normalized, e.g., FP4 grid is [-6, ..., 6])
        scale: Per-group scale factor

    Returns:
        (quantized_values, indices) where quantized_values = grid[indices] * scale
    """
    # Normalize by scale
    normalized = values / max(scale, 1e-10)

    # Find nearest grid point for each value
    # Expand dims for broadcasting: values[..., None] vs grid[None, None, ...]
    flat_norm = normalized.flatten()
    dists = np.abs(flat_norm[:, None] - grid[None, :])
    indices = np.argmin(dists, axis=1).astype(np.int32)

    # Dequantize to get actual quantized values
    quantized = grid[indices] * scale
    quantized = quantized.reshape(values.shape)
    indices = indices.reshape(values.shape)

    return quantized, indices


def gptq_quantize_layer(
    weights: NDArray[np.float32],
    hessian: NDArray[np.float32],
    grid: NDArray[np.float32],
    group_size: int = 128,
    actorder: bool = True,
    percdamp: float = 0.01,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32]]:
    """
    Apply GPTQ quantization to a weight matrix.

    GPTQ quantizes one column at a time, propagating the quantization error
    to subsequent columns using the inverse Hessian. This minimizes the
    reconstruction error on the calibration data distribution.

    Algorithm:
    1. Compute Cholesky decomposition of damped Hessian: H = L @ L^T
    2. If actorder: sort columns by Hessian diagonal (importance)
    3. For each column i (in actorder):
       a. Quantize column i to nearest grid point
       b. Compute quantization error e_i = w_i - q_i
       c. Propagate error to remaining columns: W[:, j>i] -= e_i * H_inv[i, j] / H_inv[i, i]
    4. Compute per-group scales from quantized weights

    Args:
        weights: Weight matrix [out_features, in_features]
        hessian: Hessian matrix [in_features, in_features]
        grid: Quantization grid (FP4, INT4, or NF4)
        group_size: Elements per quantization group
        actorder: If True, quantize columns in importance order
        percdamp: Damping percentage for numerical stability

    Returns:
        (quantized_weights, scales, indices) where:
        - quantized_weights: Dequantized weights for validation [out, in]
        - scales: Per-group scale factors [out, in // group_size]
        - indices: Quantization grid indices [out, in]
    """
    W = weights.astype(np.float64).copy()
    out_features, in_features = W.shape

    # Add damping to Hessian diagonal
    H = hessian.astype(np.float64).copy()
    damp = percdamp * np.mean(np.diag(H))
    H[np.diag_indices_from(H)] += damp

    # Compute inverse Hessian via Cholesky
    try:
        L = np.linalg.cholesky(H)
        H_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(in_features)))
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if Cholesky fails
        H_inv = np.linalg.pinv(H)

    # Determine column processing order
    if actorder:
        # Sort by Hessian diagonal (higher = more important = process first)
        diag = np.diag(H).copy()
        perm = np.argsort(-diag)  # Descending order
        inv_perm = np.argsort(perm)  # Inverse permutation for unpermuting
    else:
        perm = np.arange(in_features)
        inv_perm = perm

    # Reorder weights and Hessian according to permutation
    W = W[:, perm]
    H_inv = H_inv[np.ix_(perm, perm)]

    # Prepare output arrays
    Q = np.zeros_like(W)  # Quantized (dequantized) weights
    Qidx = np.zeros(W.shape, dtype=np.int32)  # Grid indices

    # Compute per-group scales (before quantization for better scale estimation)
    n_groups = in_features // group_size
    scales = np.zeros((out_features, n_groups), dtype=np.float32)

    # Grid max magnitude for scale computation
    grid_max = np.max(np.abs(grid))

    # Process groups
    for g in range(n_groups):
        g_start = g * group_size
        g_end = (g + 1) * group_size

        # Compute scale for this group
        group_max = np.max(np.abs(W[:, g_start:g_end]), axis=1)
        scales[:, g] = group_max / grid_max + 1e-10

        # Process columns within group
        for i in range(g_start, g_end):
            col = W[:, i]
            scale_col = scales[:, g]

            # Quantize column
            q_col, idx_col = quantize_to_grid(col, grid, scale_col[:, None])
            q_col = q_col.flatten()
            idx_col = idx_col.flatten()

            Q[:, i] = q_col
            Qidx[:, i] = idx_col

            # Compute error and propagate to remaining columns
            err = (col - q_col)[:, None]  # [out_features, 1]

            # Error propagation: W[:, j] -= err * H_inv[i, j] / H_inv[i, i]
            # Only propagate to columns within same group (locality)
            for j in range(i + 1, g_end):
                h_ratio = H_inv[i, j] / max(H_inv[i, i], 1e-10)
                W[:, j] -= err.flatten() * h_ratio

    # Unpermute back to original order
    Q = Q[:, inv_perm]
    Qidx = Qidx[:, inv_perm]

    # Recompute scales in original order
    scales_final = np.zeros((out_features, n_groups), dtype=np.float32)
    for g in range(n_groups):
        g_start = g * group_size
        g_end = (g + 1) * group_size
        group_max = np.max(np.abs(Q[:, g_start:g_end]), axis=1)
        scales_final[:, g] = group_max / grid_max + 1e-10

    return Q.astype(np.float32), scales_final.astype(np.float16), Qidx


# =============================================================================
# MR-GPTQ Quantizer
# =============================================================================


@dataclass
class QuantizationLayerReport:
    """Per-layer quantization statistics."""

    name: str
    shape: tuple[int, int]
    group_size: int
    format: QuantizationFormat
    use_hadamard: bool
    rmse: float
    max_error: float
    mean_relative_error: float


@dataclass
class QuantizationReport:
    """Full model quantization report."""

    model_path: str
    output_path: str
    format: QuantizationFormat
    group_size: int
    use_hadamard: bool
    hadamard_block_size: int
    actorder: bool
    quantized_layers: int
    skipped_layers: int
    total_params: int
    quantized_params: int
    compression_ratio: float
    mean_rmse: float
    layers: list[QuantizationLayerReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "model_path": self.model_path,
            "output_path": self.output_path,
            "format": self.format.value,
            "group_size": self.group_size,
            "use_hadamard": self.use_hadamard,
            "hadamard_block_size": self.hadamard_block_size,
            "actorder": self.actorder,
            "quantized_layers": self.quantized_layers,
            "skipped_layers": self.skipped_layers,
            "total_params": self.total_params,
            "quantized_params": self.quantized_params,
            "compression_ratio": self.compression_ratio,
            "mean_rmse": self.mean_rmse,
            "layers": [
                {
                    "name": layer.name,
                    "shape": list(layer.shape),
                    "group_size": layer.group_size,
                    "format": layer.format.value,
                    "use_hadamard": layer.use_hadamard,
                    "rmse": layer.rmse,
                    "max_error": layer.max_error,
                    "mean_relative_error": layer.mean_relative_error,
                }
                for layer in self.layers
            ],
        }


def _get_quantization_grid(fmt: QuantizationFormat) -> NDArray[np.float32]:
    """Get quantization grid for format."""
    if fmt == QuantizationFormat.FP4:
        return FP4_E2M1_GRID
    elif fmt == QuantizationFormat.INT4:
        return INT4_GRID
    elif fmt == QuantizationFormat.NF4:
        return NF4_GRID
    else:
        raise ValueError(f"Unknown format: {fmt}")


def _pack_fp4_weights(
    indices: NDArray[np.int32],
) -> NDArray[np.uint32]:
    """
    Pack FP4 indices into uint32 for Marlin format.

    8 FP4 nibbles per uint32:
        bits [3:0]   = value 0
        bits [7:4]   = value 1
        ...
        bits [31:28] = value 7

    Args:
        indices: FP4 grid indices [out_features, in_features]

    Returns:
        Packed uint32 array [out_features, in_features // 8]
    """
    out_feat, in_feat = indices.shape
    if in_feat % 8 != 0:
        raise ValueError(f"in_features ({in_feat}) must be divisible by 8 for FP4 packing")

    packed = np.zeros((out_feat, in_feat // 8), dtype=np.uint32)
    for i in range(8):
        packed |= (indices[:, i::8].astype(np.uint32) & 0xF) << (i * 4)

    return packed


def _compute_layer_error(
    original: NDArray[np.float32],
    quantized: NDArray[np.float32],
) -> dict[str, float]:
    """Compute quantization error metrics."""
    diff = original.astype(np.float64) - quantized.astype(np.float64)
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    max_err = float(np.max(np.abs(diff)))
    rel_err = np.abs(diff) / (np.abs(original) + 1e-10)
    mean_rel = float(np.mean(rel_err))

    return {
        "mse": mse,
        "rmse": rmse,
        "max_error": max_err,
        "mean_relative_error": mean_rel,
    }


class MRGPTQQuantizer:
    """
    MR-GPTQ Quantizer: Hadamard Rotation + GPTQ for high-quality low-bit quantization.

    Combines:
    1. Hadamard rotation to disperse activation outliers
    2. GPTQ algorithm with Hessian-aware error compensation
    3. Marlin FP4/INT4/NF4 packing for Metal GPU inference

    Args:
        bits: Quantization bits (4 for FP4/INT4/NF4)
        format: Quantization format (FP4, INT4, or NF4)
        group_size: Elements per quantization group
        use_hadamard: Apply Hadamard rotation before quantization
        hadamard_block_size: Block size for Hadamard transform (power of 2)
        actorder: Process columns in activation importance order
        percdamp: Damping ratio for Hessian inversion
    """

    def __init__(
        self,
        bits: int = 4,
        format: str | QuantizationFormat = "fp4",
        group_size: int = 128,
        use_hadamard: bool = True,
        hadamard_block_size: int = 64,
        actorder: bool = True,
        percdamp: float = 0.01,
    ):
        if bits != 4:
            raise ValueError(f"Only 4-bit quantization supported, got {bits}")

        if isinstance(format, str):
            format = QuantizationFormat(format.lower())
        self.format = format

        self.bits = bits
        self.group_size = group_size
        self.use_hadamard = use_hadamard
        self.hadamard_block_size = hadamard_block_size
        self.actorder = actorder
        self.percdamp = percdamp

        # Get quantization grid
        self.grid = _get_quantization_grid(self.format)

    def quantize_layer(
        self,
        weights: NDArray[np.float32],
        hessian: NDArray[np.float32] | None = None,
        layer_name: str = "",
    ) -> tuple[NDArray[np.uint32], NDArray[np.float16], dict[str, Any]]:
        """
        Quantize a single layer using MR-GPTQ.

        Args:
            weights: Weight matrix [out_features, in_features]
            hessian: Hessian matrix [in_features, in_features] or None for RTN
            layer_name: Layer name for logging

        Returns:
            (packed_weights, scales, metadata) where:
            - packed_weights: uint32 packed for Marlin [out, in // 8]
            - scales: FP16 per-group scales [out, in // group_size]
            - metadata: Hadamard rotation metadata and quantization stats
        """
        W = weights.astype(np.float32)
        out_feat, in_feat = W.shape

        metadata: dict[str, Any] = {
            "layer_name": layer_name,
            "original_shape": (out_feat, in_feat),
            "format": self.format.value,
            "group_size": self.group_size,
            "use_hadamard": self.use_hadamard,
        }

        # Ensure dimensions are compatible
        if in_feat % 8 != 0:
            raise ValueError(f"in_features ({in_feat}) must be divisible by 8 for Marlin packing")
        if in_feat % self.group_size != 0:
            raise ValueError(
                f"in_features ({in_feat}) must be divisible by group_size ({self.group_size})"
            )

        # Step 1: Apply Hadamard rotation (optional)
        hadamard_meta = None
        if self.use_hadamard:
            W, hadamard_meta = apply_hadamard_rotation(
                W, block_size=self.hadamard_block_size, axis=1
            )
            metadata["hadamard"] = hadamard_meta
            # Update dimensions if padding was added
            out_feat, in_feat = W.shape

        # Step 2: Quantize using GPTQ or RTN
        if hessian is not None:
            # GPTQ with Hessian-aware error compensation
            Q, scales, indices = gptq_quantize_layer(
                W,
                hessian,
                self.grid,
                group_size=self.group_size,
                actorder=self.actorder,
                percdamp=self.percdamp,
            )
        else:
            # RTN (Round-to-Nearest) fallback
            Q, scales, indices = self._rtn_quantize(W)

        # Compute error metrics
        error = _compute_layer_error(W, Q)
        metadata["error"] = error

        # Step 3: Pack for Marlin format
        packed = _pack_fp4_weights(indices)

        return packed, scales, metadata

    def _rtn_quantize(
        self,
        weights: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float16], NDArray[np.int32]]:
        """
        Round-to-Nearest quantization (fallback when no Hessian available).

        Args:
            weights: Weight matrix [out_features, in_features]

        Returns:
            (quantized_weights, scales, indices)
        """
        W = weights.astype(np.float32)
        out_feat, in_feat = W.shape

        n_groups = in_feat // self.group_size
        grid_max = np.max(np.abs(self.grid))

        scales = np.zeros((out_feat, n_groups), dtype=np.float16)
        Q = np.zeros_like(W)
        Qidx = np.zeros(W.shape, dtype=np.int32)

        for g in range(n_groups):
            g_start = g * self.group_size
            g_end = (g + 1) * self.group_size

            # Compute per-row scale for this group
            group_weights = W[:, g_start:g_end]
            group_max = np.max(np.abs(group_weights), axis=1, keepdims=True)
            group_scale = (group_max / grid_max).flatten() + 1e-10
            scales[:, g] = group_scale.astype(np.float16)

            # Quantize group
            for row in range(out_feat):
                q_vals, q_idx = quantize_to_grid(
                    group_weights[row], self.grid, float(group_scale[row])
                )
                Q[row, g_start:g_end] = q_vals
                Qidx[row, g_start:g_end] = q_idx

        return Q, scales, Qidx

    def quantize_model(
        self,
        model_path: str | Path,
        calibration_data: CalibrationDataset | None = None,
        output_path: str | Path | None = None,
        layers_to_quantize: list[str] | None = None,
        verbose: bool = True,
    ) -> QuantizationReport:
        """
        Full MR-GPTQ quantization pipeline for a model.

        Pipeline:
        1. Load model weights from safetensors
        2. For models with calibration data:
           a. Run forward passes to collect activation Hessians
           b. Apply Hadamard rotation to weights
           c. Run GPTQ with Hessian-aware error compensation
           d. Pack quantized weights in Marlin format
        3. For models without calibration: use RTN quantization
        4. Save quantized model and metadata

        Args:
            model_path: Path to model directory or safetensors file
            calibration_data: Calibration dataset for Hessian collection
            output_path: Directory to save quantized model
            layers_to_quantize: List of layer name patterns to quantize.
                              If None, quantizes all linear layers.
            verbose: Print progress

        Returns:
            QuantizationReport with quality metrics
        """
        from safetensors import safe_open
        from safetensors.numpy import save_file

        model_path = Path(model_path)

        # Find safetensors files
        if model_path.is_file():
            st_files = [model_path]
            model_dir = model_path.parent
        else:
            st_files = sorted(model_path.glob("*.safetensors"))
            if not st_files:
                raise FileNotFoundError(f"No safetensors files found in {model_path}")
            model_dir = model_path

        if output_path is None:
            output_path = model_dir / "quantized"
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if verbose:
            print("MR-GPTQ Quantization")
            print(f"  Format: {self.format.value}")
            print(f"  Group size: {self.group_size}")
            print(f"  Hadamard: {self.use_hadamard} (block_size={self.hadamard_block_size})")
            print(f"  Actorder: {self.actorder}")
            print()

        # Collect Hessians if calibration data provided
        hessians: dict[str, NDArray[np.float32]] = {}
        if calibration_data is not None:
            if verbose:
                print("Collecting Hessians from calibration data...")
            # Note: Full Hessian collection requires model inference
            # This is a placeholder - actual implementation needs model loading
            # For now, we'll use RTN quantization
            if verbose:
                print("  (Hessian collection requires model loading - using RTN fallback)")

        # Process each safetensors file
        output_tensors: dict[str, np.ndarray] = {}
        layer_reports: list[QuantizationLayerReport] = []
        quantized_count = 0
        skipped_count = 0
        total_params = 0
        quantized_params = 0

        for st_file in st_files:
            if verbose:
                print(f"\nProcessing {st_file.name}...")

            with safe_open(str(st_file), framework="numpy") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    total_params += tensor.size

                    # Check if should quantize
                    should_quant = self._should_quantize_tensor(name, tensor, layers_to_quantize)

                    if should_quant:
                        if verbose:
                            print(f"  Quantizing: {name} {tensor.shape}")

                        # Get Hessian if available
                        hessian = hessians.get(name)

                        # Quantize layer
                        packed, scales, meta = self.quantize_layer(tensor, hessian, layer_name=name)

                        # Store packed weights and scales
                        output_tensors[name] = packed
                        output_tensors[f"{name}.scales"] = scales
                        output_tensors[f"{name}.group_size"] = np.array(
                            [self.group_size], dtype=np.int32
                        )

                        # Store Hadamard metadata if used
                        if meta.get("hadamard"):
                            h_meta = meta["hadamard"]
                            output_tensors[f"{name}.hadamard_block_size"] = np.array(
                                [h_meta["block_size"]], dtype=np.int32
                            )

                        # Record statistics
                        err = meta.get("error", {})
                        layer_reports.append(
                            QuantizationLayerReport(
                                name=name,
                                shape=tensor.shape,
                                group_size=self.group_size,
                                format=self.format,
                                use_hadamard=self.use_hadamard,
                                rmse=err.get("rmse", 0.0),
                                max_error=err.get("max_error", 0.0),
                                mean_relative_error=err.get("mean_relative_error", 0.0),
                            )
                        )

                        quantized_count += 1
                        quantized_params += tensor.size
                    else:
                        if verbose:
                            print(f"  Keeping: {name} {tensor.shape}")
                        output_tensors[name] = tensor
                        skipped_count += 1

                    # Memory cleanup
                    gc.collect()

        # Save quantized model
        output_file = output_path / "model.safetensors"
        if verbose:
            print(f"\nSaving to {output_file}...")
        save_file(output_tensors, str(output_file))

        # Compute overall statistics
        mean_rmse = float(np.mean([r.rmse for r in layer_reports])) if layer_reports else 0.0

        # Original size: FP16 per weight
        # Quantized size: 4 bits per weight + FP16 scale per group
        original_bits = quantized_params * 16
        quant_weight_bits = quantized_params * 4
        quant_scale_bits = (quantized_params // self.group_size) * 16
        compression_ratio = original_bits / max(quant_weight_bits + quant_scale_bits, 1)

        report = QuantizationReport(
            model_path=str(model_path),
            output_path=str(output_path),
            format=self.format,
            group_size=self.group_size,
            use_hadamard=self.use_hadamard,
            hadamard_block_size=self.hadamard_block_size,
            actorder=self.actorder,
            quantized_layers=quantized_count,
            skipped_layers=skipped_count,
            total_params=total_params,
            quantized_params=quantized_params,
            compression_ratio=compression_ratio,
            mean_rmse=mean_rmse,
            layers=layer_reports,
        )

        # Save report
        report_file = output_path / "quantization_report.json"
        with open(report_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        if verbose:
            print(f"\n{'=' * 60}")
            print("MR-GPTQ Quantization Complete!")
            print(f"  Quantized: {quantized_count} layers")
            print(f"  Skipped: {skipped_count} layers")
            print(f"  Compression: {compression_ratio:.2f}x")
            print(f"  Mean RMSE: {mean_rmse:.6f}")
            print(f"{'=' * 60}")

        return report

    def _should_quantize_tensor(
        self,
        name: str,
        tensor: np.ndarray,
        layers_to_quantize: list[str] | None,
    ) -> bool:
        """Determine if a tensor should be quantized."""
        # Must be 2D weight matrix
        if tensor.ndim != 2:
            return False

        # Must have "weight" in name
        if "weight" not in name.lower():
            return False

        # Skip embeddings, norms, biases, lm_head
        skip_patterns = [
            "embed",
            "embedding",
            "norm",
            "layernorm",
            "rmsnorm",
            "lm_head",
            "output",
            "bias",
            "router",  # Keep MoE router in full precision
        ]
        name_lower = name.lower()
        if any(pat in name_lower for pat in skip_patterns):
            return False

        # Check dimension compatibility
        out_feat, in_feat = tensor.shape
        if in_feat % 8 != 0 or in_feat % self.group_size != 0:
            return False

        # Check against layer filter if provided
        if layers_to_quantize is not None:
            if not any(pat in name for pat in layers_to_quantize):
                return False

        return True

    def quantize_model_with_calibration(
        self,
        model_path: Path,
        calibration: CalibrationDataset,
        tokenizer,
        output_path: Path,
        precision_config: MoEPrecisionConfig | None = None,
        num_calibration_batches: int = 128,
        batch_size: int = 4,
        max_seq_len: int = 2048,
        checkpoint_dir: Path | None = None,
        resume: bool = True,
        verbose: bool = True,
    ) -> QuantizationReport:
        """
        Full MR-GPTQ pipeline with Hessian-aware quantization.

        This is the main production entry point that:
        1. Loads model for forward passes (streaming to avoid OOM)
        2. Registers Hessian collection hooks
        3. Runs calibration forward passes (multi-domain v3)
        4. Applies Hadamard rotation to weights
        5. Runs GPTQ with collected Hessians
        6. Packs and saves quantized model

        Supports:
        - Streaming large models (loads shards on-demand)
        - Checkpointing progress (saves after each layer)
        - Resume from interruption

        Args:
            model_path: Path to HuggingFace model directory
            calibration: CalibrationDataset (e.g., CalibrationDataset.v3())
            tokenizer: HuggingFace tokenizer with encode/decode methods
            output_path: Directory to save quantized model
            precision_config: Per-layer precision config for MoE models
            num_calibration_batches: Number of calibration batches to run
            batch_size: Samples per calibration batch
            max_seq_len: Maximum sequence length for tokenization
            checkpoint_dir: Directory for checkpoints (default: output_path/checkpoints)
            resume: If True, resume from checkpoint if available
            verbose: Print progress

        Returns:
            QuantizationReport with quality metrics
        """
        import torch
        from safetensors.numpy import save_file

        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if checkpoint_dir is None:
            checkpoint_dir = output_path / "checkpoints"

        # Try to resume from checkpoint
        checkpoint = None
        if resume:
            checkpoint = QuantizationCheckpoint.load(checkpoint_dir)
            if checkpoint is not None and verbose:
                print(f"Resuming from checkpoint: {len(checkpoint.completed_layers)} layers done")

        if checkpoint is None:
            checkpoint = QuantizationCheckpoint.create(checkpoint_dir, str(model_path))

        if verbose:
            print("=" * 60)
            print("MR-GPTQ Quantization with Hessian Calibration")
            print("=" * 60)
            print(f"Model: {model_path}")
            print(f"Output: {output_path}")
            print(f"Format: {self.format.value}")
            print(f"Group size: {self.group_size}")
            print(f"Hadamard: {self.use_hadamard} (block={self.hadamard_block_size})")
            print(f"Actorder: {self.actorder}")
            print(f"Calibration batches: {num_calibration_batches}")
            print()

        # Step 1: Load model for inference
        if verbose:
            print("Step 1: Loading model for calibration...")

        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            ) from e

        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.float32,  # FP32 for accurate Hessians
            device_map="cpu",  # Start on CPU, move to GPU per-layer
        )
        model.eval()

        # Step 2: Register Hessian collection hooks
        if verbose:
            print("Step 2: Registering Hessian collection hooks...")

        hessian_collector = HessianCollector(damp_ratio=self.percdamp)

        # Load existing Hessian checkpoint if available
        hessian_checkpoint_path = checkpoint_dir / "hessians"
        if resume and hessian_checkpoint_path.exists():
            hessian_collector.load_checkpoint(hessian_checkpoint_path)
            if verbose:
                print(f"  Loaded Hessian checkpoint: {len(hessian_collector._hessians)} layers")

        hessian_collector.register_hooks(model)
        if verbose:
            print(f"  Registered hooks on {len(hessian_collector._hooks)} layers")

        # Step 3: Run calibration forward passes
        if verbose:
            print(f"Step 3: Running {num_calibration_batches} calibration batches...")

        calibration_samples = list(calibration.samples)
        n_samples = min(len(calibration_samples), num_calibration_batches * batch_size)

        with torch.no_grad():
            for batch_idx in range(0, n_samples, batch_size):
                batch_end = min(batch_idx + batch_size, n_samples)
                batch_texts = calibration_samples[batch_idx:batch_end]

                # Tokenize batch
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_len,
                    padding=True,
                )

                # Forward pass (hooks accumulate Hessians)
                model(**inputs)

                if verbose and (batch_idx // batch_size + 1) % 10 == 0:
                    batches_done = batch_idx // batch_size + 1
                    total_batches = (n_samples + batch_size - 1) // batch_size
                    print(f"  Batch {batches_done}/{total_batches}")

                # Checkpoint Hessians periodically
                if (batch_idx // batch_size + 1) % 50 == 0:
                    hessian_collector.save_checkpoint(hessian_checkpoint_path)

        # Final Hessian checkpoint
        hessian_collector.save_checkpoint(hessian_checkpoint_path)
        hessian_collector.remove_hooks()

        # Get computed Hessians
        hessians = hessian_collector.get_hessians()
        if verbose:
            print(f"  Collected Hessians for {len(hessians)} layers")
            for name, info in list(hessians.items())[:3]:
                print(f"    {name}: {info.n_samples} samples")

        # Step 4: Quantize weights with Hessians
        if verbose:
            print("\nStep 4: Quantizing weights with GPTQ...")

        # Release model from memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load weights from safetensors and quantize
        from safetensors import safe_open

        st_files = sorted(model_path.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No safetensors files in {model_path}")

        output_tensors: dict[str, np.ndarray] = {}
        layer_reports: list[QuantizationLayerReport] = []
        quantized_count = 0
        skipped_count = 0
        total_params = 0
        quantized_params = 0

        for st_file in st_files:
            if verbose:
                print(f"\n  Processing {st_file.name}...")

            with safe_open(str(st_file), framework="numpy") as f:
                for name in f.keys():
                    # Skip already processed layers
                    if checkpoint.is_layer_complete(name):
                        if verbose:
                            print(f"    Skipping (already done): {name}")
                        continue

                    tensor = f.get_tensor(name)
                    total_params += tensor.size

                    # Determine if should quantize
                    should_quant = self._should_quantize_tensor(
                        name, tensor, layers_to_quantize=None
                    )

                    if should_quant:
                        checkpoint.current_layer = name
                        checkpoint.save()

                        if verbose:
                            print(f"    Quantizing: {name} {tensor.shape}")

                        # Get Hessian for this layer
                        # Map tensor name to hook name (remove .weight suffix)
                        hessian_name = name.replace(".weight", "")
                        hessian_info = hessians.get(hessian_name)
                        hessian = hessian_info.hessian if hessian_info else None

                        if hessian is None:
                            # Try alternative names
                            for h_name in hessians:
                                if h_name in name or name in h_name:
                                    hessian = hessians[h_name].hessian
                                    break

                        # Quantize layer
                        try:
                            packed, scales, meta = self.quantize_layer(
                                tensor, hessian, layer_name=name
                            )
                        except Exception as e:
                            if verbose:
                                print(f"      Warning: GPTQ failed ({e}), using RTN")
                            packed, scales, meta = self.quantize_layer(
                                tensor, hessian=None, layer_name=name
                            )

                        # Store results
                        output_tensors[name] = packed
                        output_tensors[f"{name}.scales"] = scales
                        output_tensors[f"{name}.group_size"] = np.array(
                            [self.group_size], dtype=np.int32
                        )

                        if meta.get("hadamard"):
                            h_meta = meta["hadamard"]
                            output_tensors[f"{name}.hadamard_block_size"] = np.array(
                                [h_meta["block_size"]], dtype=np.int32
                            )

                        # Record stats
                        err = meta.get("error", {})
                        layer_reports.append(
                            QuantizationLayerReport(
                                name=name,
                                shape=tensor.shape,
                                group_size=self.group_size,
                                format=self.format,
                                use_hadamard=self.use_hadamard,
                                rmse=err.get("rmse", 0.0),
                                max_error=err.get("max_error", 0.0),
                                mean_relative_error=err.get("mean_relative_error", 0.0),
                            )
                        )

                        quantized_count += 1
                        quantized_params += tensor.size

                    else:
                        if verbose:
                            print(f"    Keeping: {name} {tensor.shape}")
                        output_tensors[name] = tensor
                        skipped_count += 1

                    # Mark layer complete
                    checkpoint.mark_layer_complete(name)

                    # Memory cleanup
                    gc.collect()

        # Step 5: Save quantized model
        if verbose:
            print("\nStep 5: Saving quantized model...")

        output_file = output_path / "model.safetensors"
        save_file(output_tensors, str(output_file))

        # Compute statistics
        mean_rmse = float(np.mean([r.rmse for r in layer_reports])) if layer_reports else 0.0

        original_bits = quantized_params * 16
        quant_weight_bits = quantized_params * 4
        quant_scale_bits = (quantized_params // self.group_size) * 16
        compression_ratio = original_bits / max(quant_weight_bits + quant_scale_bits, 1)

        report = QuantizationReport(
            model_path=str(model_path),
            output_path=str(output_path),
            format=self.format,
            group_size=self.group_size,
            use_hadamard=self.use_hadamard,
            hadamard_block_size=self.hadamard_block_size,
            actorder=self.actorder,
            quantized_layers=quantized_count,
            skipped_layers=skipped_count,
            total_params=total_params,
            quantized_params=quantized_params,
            compression_ratio=compression_ratio,
            mean_rmse=mean_rmse,
            layers=layer_reports,
        )

        # Save report
        report_file = output_path / "quantization_report.json"
        with open(report_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        if verbose:
            print()
            print("=" * 60)
            print("MR-GPTQ Quantization Complete!")
            print("=" * 60)
            print(f"  Quantized layers: {quantized_count}")
            print(f"  Skipped layers: {skipped_count}")
            print(f"  Total params: {total_params / 1e9:.2f}B")
            print(f"  Quantized params: {quantized_params / 1e9:.2f}B")
            print(f"  Compression ratio: {compression_ratio:.2f}x")
            print(f"  Mean RMSE: {mean_rmse:.6f}")
            print(f"  Output: {output_file}")

        return report


# =============================================================================
# MoE Precision Configuration
# =============================================================================


@dataclass
class MoEPrecisionConfig:
    """Per-layer precision configuration for MoE models.

    MoE models have varying sensitivity across layer types:
    - Router/gate: Critical for expert selection, keep high precision (BF16/FP16)
    - Shared expert: Sees all tokens, moderate precision (FP4 with small group)
    - Routed experts: Redundant (2 of 64 active), can be aggressive (FP4 large group)
    - Attention: Position-sensitive, tight FP4 quantization

    Example:
        config = MoEPrecisionConfig.default_moe()
        # Or create custom:
        config = MoEPrecisionConfig(
            router_precision="fp16",
            expert_group_size=256,
            attention_group_size=64,
        )
    """

    # Router/gate precision (determines expert selection)
    router_precision: str = "bf16"  # "fp16", "bf16", "fp4"

    # Expert precision settings
    expert_format: str = "fp4"
    expert_group_size: int = 128
    shared_expert_group_size: int = 64  # Tighter for shared (always active)

    # Attention precision
    attention_format: str = "fp4"
    attention_group_size: int = 64

    # MLP (dense model) precision
    mlp_format: str = "fp4"
    mlp_group_size: int = 128

    # MTP heads (Multi-Token Prediction)
    mtp_format: str = "fp4"
    mtp_group_size: int = 256  # Aggressive for drafts

    # Hadamard settings per layer type
    use_hadamard_experts: bool = True
    use_hadamard_attention: bool = True
    hadamard_block_size: int = 64

    @classmethod
    def default_dense(cls) -> MoEPrecisionConfig:
        """Default config for dense transformer models."""
        return cls(
            router_precision="fp16",
            expert_format="fp4",
            expert_group_size=128,
            attention_group_size=64,
            mlp_group_size=128,
        )

    @classmethod
    def default_moe(cls) -> MoEPrecisionConfig:
        """Default config for MoE models (Mixtral, etc.)."""
        return cls(
            router_precision="bf16",
            expert_format="fp4",
            expert_group_size=128,
            shared_expert_group_size=64,
            attention_group_size=64,
        )

    @classmethod
    def default_moe_mtp(cls) -> MoEPrecisionConfig:
        """Config for MoE + MTP models like GLM-4.7-Flash."""
        return cls(
            router_precision="bf16",
            expert_format="fp4",
            expert_group_size=128,
            shared_expert_group_size=64,
            attention_group_size=64,
            mtp_format="fp4",
            mtp_group_size=256,
        )

    @classmethod
    def quality_first(cls) -> MoEPrecisionConfig:
        """Prioritize quality over compression."""
        return cls(
            router_precision="fp16",
            expert_format="fp4",
            expert_group_size=64,
            shared_expert_group_size=32,
            attention_group_size=32,
            mlp_group_size=64,
            mtp_group_size=128,
        )

    @classmethod
    def speed_first(cls) -> MoEPrecisionConfig:
        """Prioritize speed/compression over quality."""
        return cls(
            router_precision="bf16",
            expert_format="fp4",
            expert_group_size=256,
            shared_expert_group_size=128,
            attention_group_size=128,
            mlp_group_size=256,
            mtp_group_size=512,
        )


# =============================================================================
# Streaming Hessian Collection
# =============================================================================


class HessianCollector:
    """Streaming Hessian accumulator for memory-efficient calibration.

    Collects Hessian approximation (X^T @ X) incrementally during forward passes,
    avoiding storage of all calibration activations in memory.

    For a 4096-hidden model:
    - Full activations (1000 samples, 2048 seq, FP32): 4096 * 2048 * 1000 * 4 = 33GB
    - Streaming Hessian: 4096 * 4096 * 8 = 134MB (FP64 for precision)

    Usage:
        collector = HessianCollector()
        collector.register_hooks(model)

        for batch in calibration_data:
            model(batch)  # Hooks accumulate automatically

        hessians = collector.get_hessians()
        collector.remove_hooks()
    """

    def __init__(
        self,
        damp_ratio: float = 0.01,
        layer_patterns: list[str] | None = None,
    ):
        """Initialize collector.

        Args:
            damp_ratio: Damping ratio λ = damp_ratio * mean(diag(H)).
            layer_patterns: Layer name patterns to track. If None, tracks all
                Linear layers except embeddings/norms/lm_head.
        """
        self.damp_ratio = damp_ratio
        self.layer_patterns = layer_patterns

        # Running sums: layer_name -> (H_sum, n_samples)
        self._hessians: dict[str, tuple[NDArray[np.float64], int]] = {}
        self._hooks: list = []
        self._layer_dims: dict[str, int] = {}  # Cache in_features per layer

    def register_hooks(self, model) -> None:
        """Register forward hooks on target layers.

        Args:
            model: PyTorch or MLX model with named_modules() method.
        """
        # Lazy import to avoid torch dependency when not needed
        import torch.nn as nn

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            # Skip non-weight layers
            if not self._should_track_layer(name):
                continue

            in_features = module.in_features
            self._layer_dims[name] = in_features

            # Initialize Hessian accumulator
            self._hessians[name] = (
                np.zeros((in_features, in_features), dtype=np.float64),
                0,
            )

            # Register hook
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _should_track_layer(self, name: str) -> bool:
        """Determine if a layer should have Hessian tracked."""
        name_lower = name.lower()

        # Always skip these
        skip_patterns = [
            "embed",
            "embedding",
            "norm",
            "layernorm",
            "rmsnorm",
            "lm_head",
            "output",
            "bias",
        ]
        if any(p in name_lower for p in skip_patterns):
            return False

        # If custom patterns specified, check them
        if self.layer_patterns is not None:
            return any(p in name for p in self.layer_patterns)

        # Default: track linear layers with weight
        return True

    def _make_hook(self, layer_name: str):
        """Create forward hook for Hessian accumulation."""
        import torch

        def hook(module, input, output):
            # Get input activations
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input

            if not isinstance(x, torch.Tensor):
                return

            # Flatten to [n_tokens, in_features]
            with torch.no_grad():
                x_flat = x.view(-1, x.shape[-1]).float()
                x_np = x_flat.cpu().numpy().astype(np.float64)

            # Accumulate H += X^T @ X
            H_sum, n_samples = self._hessians[layer_name]
            H_sum += x_np.T @ x_np
            self._hessians[layer_name] = (H_sum, n_samples + x_np.shape[0])

        return hook

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def get_hessians(self, apply_damping: bool = True) -> dict[str, HessianInfo]:
        """Get computed Hessians for all tracked layers.

        Args:
            apply_damping: If True, apply diagonal damping for numerical stability.

        Returns:
            Dict mapping layer names to HessianInfo objects.
        """
        results = {}

        for name, (H_sum, n_samples) in self._hessians.items():
            if n_samples == 0:
                continue

            # Normalize by sample count
            H = H_sum / n_samples

            # Compute diagonal for actorder
            diag = np.diag(H).copy()

            # Apply damping
            damp = 0.0
            if apply_damping and diag.mean() > 0:
                damp = self.damp_ratio * diag.mean()
                H[np.diag_indices_from(H)] += damp

            results[name] = HessianInfo(
                hessian=H.astype(np.float32),
                n_samples=n_samples,
                diag=diag.astype(np.float32),
                damp=damp,
            )

        return results

    def get_single_hessian(self, layer_name: str, apply_damping: bool = True) -> HessianInfo | None:
        """Get Hessian for a single layer.

        Args:
            layer_name: Layer name to retrieve.
            apply_damping: Apply diagonal damping.

        Returns:
            HessianInfo or None if layer not found.
        """
        if layer_name not in self._hessians:
            return None

        H_sum, n_samples = self._hessians[layer_name]
        if n_samples == 0:
            return None

        H = H_sum / n_samples
        diag = np.diag(H).copy()

        damp = 0.0
        if apply_damping and diag.mean() > 0:
            damp = self.damp_ratio * diag.mean()
            H[np.diag_indices_from(H)] += damp

        return HessianInfo(
            hessian=H.astype(np.float32),
            n_samples=n_samples,
            diag=diag.astype(np.float32),
            damp=damp,
        )

    def clear(self) -> None:
        """Clear accumulated Hessians (but keep hooks)."""
        for name in self._hessians:
            in_features = self._layer_dims[name]
            self._hessians[name] = (
                np.zeros((in_features, in_features), dtype=np.float64),
                0,
            )

    def save_checkpoint(self, path: Path) -> None:
        """Save accumulated Hessians to disk for resume capability."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for name, (H_sum, n_samples) in self._hessians.items():
            # Sanitize layer name for filename
            safe_name = name.replace("/", "_").replace(".", "_")
            np.savez_compressed(
                path / f"{safe_name}.npz",
                H_sum=H_sum,
                n_samples=np.array([n_samples]),
                layer_name=name,
            )

    def load_checkpoint(self, path: Path) -> None:
        """Load Hessians from checkpoint directory."""
        path = Path(path)
        if not path.exists():
            return

        for npz_file in path.glob("*.npz"):
            data = np.load(npz_file, allow_pickle=True)
            layer_name = str(data["layer_name"])
            H_sum = data["H_sum"]
            n_samples = int(data["n_samples"][0])

            self._hessians[layer_name] = (H_sum, n_samples)
            self._layer_dims[layer_name] = H_sum.shape[0]


# =============================================================================
# Quantization Checkpoint for Resume
# =============================================================================


@dataclass
class QuantizationCheckpoint:
    """Checkpoint state for resumable quantization.

    Saves progress during long-running quantization jobs, allowing resume
    after interruption. Stores:
    - List of completed layers
    - Current layer being processed
    - Hessian checkpoint path
    - Partial output tensors
    """

    checkpoint_dir: Path
    model_path: str
    completed_layers: list[str] = field(default_factory=list)
    current_layer: str | None = None
    hessian_checkpoint: Path | None = None
    partial_output_file: Path | None = None

    @classmethod
    def create(cls, checkpoint_dir: Path, model_path: str) -> QuantizationCheckpoint:
        """Create new checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return cls(checkpoint_dir=checkpoint_dir, model_path=model_path)

    @classmethod
    def load(cls, checkpoint_dir: Path) -> QuantizationCheckpoint | None:
        """Load checkpoint from directory, or None if not found."""
        checkpoint_dir = Path(checkpoint_dir)
        state_file = checkpoint_dir / "state.json"
        if not state_file.exists():
            return None

        with open(state_file) as f:
            data = json.load(f)

        return cls(
            checkpoint_dir=checkpoint_dir,
            model_path=data["model_path"],
            completed_layers=data.get("completed_layers", []),
            current_layer=data.get("current_layer"),
            hessian_checkpoint=(
                Path(data["hessian_checkpoint"]) if data.get("hessian_checkpoint") else None
            ),
            partial_output_file=(
                Path(data["partial_output_file"]) if data.get("partial_output_file") else None
            ),
        )

    def save(self) -> None:
        """Save checkpoint state to disk."""
        state = {
            "model_path": self.model_path,
            "completed_layers": self.completed_layers,
            "current_layer": self.current_layer,
            "hessian_checkpoint": (
                str(self.hessian_checkpoint) if self.hessian_checkpoint else None
            ),
            "partial_output_file": (
                str(self.partial_output_file) if self.partial_output_file else None
            ),
        }
        with open(self.checkpoint_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2)

    def mark_layer_complete(self, layer_name: str) -> None:
        """Mark a layer as completed."""
        if layer_name not in self.completed_layers:
            self.completed_layers.append(layer_name)
        self.current_layer = None
        self.save()

    def is_layer_complete(self, layer_name: str) -> bool:
        """Check if a layer has been completed."""
        return layer_name in self.completed_layers


# =============================================================================
# MoE-specific quantization
# =============================================================================


class MoEMRGPTQQuantizer(MRGPTQQuantizer):
    """
    MR-GPTQ quantizer with MoE-specific handling.

    For Mixture-of-Experts models:
    - Each expert gets its own Hessian (experts may see different input distributions)
    - Router weights stay in higher precision (BF16/FP16)
    - Shared expert uses full batch Hessian

    Args:
        Same as MRGPTQQuantizer, plus:
        expert_hessian_per_expert: If True, collect separate Hessian per expert
        router_precision: Precision for router weights ("fp16" or "bf16")
        shared_expert_group_size: Group size for shared expert (often larger)
    """

    def __init__(
        self,
        bits: int = 4,
        format: str | QuantizationFormat = "fp4",
        group_size: int = 128,
        use_hadamard: bool = True,
        hadamard_block_size: int = 64,
        actorder: bool = True,
        percdamp: float = 0.01,
        expert_hessian_per_expert: bool = True,
        router_precision: str = "bf16",
        shared_expert_group_size: int | None = None,
    ):
        super().__init__(
            bits=bits,
            format=format,
            group_size=group_size,
            use_hadamard=use_hadamard,
            hadamard_block_size=hadamard_block_size,
            actorder=actorder,
            percdamp=percdamp,
        )
        self.expert_hessian_per_expert = expert_hessian_per_expert
        self.router_precision = router_precision
        self.shared_expert_group_size = shared_expert_group_size or group_size

    def _should_quantize_tensor(
        self,
        name: str,
        tensor: np.ndarray,
        layers_to_quantize: list[str] | None,
    ) -> bool:
        """Override to handle MoE-specific patterns."""
        # Keep router in full precision
        if "router" in name.lower() or "gate" in name.lower():
            return False

        # Base check
        return super()._should_quantize_tensor(name, tensor, layers_to_quantize)


# =============================================================================
# CLI
# =============================================================================


def main():
    """CLI entry point for MR-GPTQ quantization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="MR-GPTQ: Hadamard + GPTQ quantization for Metal Marlin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic FP4 quantization with Hadamard
    python -m metal_marlin.mr_gptq model/ output/ --format fp4

    # INT4 quantization without Hadamard
    python -m metal_marlin.mr_gptq model/ output/ --format int4 --no-hadamard

    # NF4 quantization with calibration (when implemented)
    python -m metal_marlin.mr_gptq model/ output/ --format nf4 --calibration v3
""",
    )

    parser.add_argument("model_path", help="Path to model directory or safetensors file")
    parser.add_argument("output_path", help="Output directory for quantized model")
    parser.add_argument(
        "--format",
        choices=["fp4", "int4", "nf4"],
        default="fp4",
        help="Quantization format (default: fp4)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Quantization group size (default: 128)",
    )
    parser.add_argument(
        "--no-hadamard",
        action="store_true",
        help="Disable Hadamard rotation",
    )
    parser.add_argument(
        "--hadamard-block-size",
        type=int,
        default=64,
        help="Hadamard block size (default: 64)",
    )
    parser.add_argument(
        "--no-actorder",
        action="store_true",
        help="Disable activation ordering",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Damping ratio for Hessian (default: 0.01)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    quantizer = MRGPTQQuantizer(
        bits=4,
        format=args.format,
        group_size=args.group_size,
        use_hadamard=not args.no_hadamard,
        hadamard_block_size=args.hadamard_block_size,
        actorder=not args.no_actorder,
        percdamp=args.percdamp,
    )

    quantizer.quantize_model(
        model_path=args.model_path,
        output_path=args.output_path,
        verbose=not args.quiet,
    )

    if not args.quiet:
        print(f"\nReport saved to: {args.output_path}/quantization_report.json")


if __name__ == "__main__":
    main()
