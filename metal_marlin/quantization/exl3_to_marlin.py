"""EXL3 to Marlin FP4 format converter.

This module provides conversion from EXL3 trellis-quantized format to
Marlin FP4 format for inference. EXL3 stores trellis indices and requires
decoding to recover weights, while Marlin stores packed FP4 nibbles with
per-group scales.

This is a temporary bridge until native EXL3 inference kernels exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from metal_marlin.hadamard import hadamard_matrix
from metal_marlin.quantization.trellis_codebook import TrellisCodebook

# FP4 E2M1 representable values (NVFP4/MXFP4 format)
# Bits: [sign(1) | exponent(2, bias=1) | mantissa(1)]
E2M1_VALUES = np.array(
    [
        0.0,  # 0000: +0
        0.5,  # 0001: +0.5 (subnormal)
        1.0,  # 0010: +1.0
        1.5,  # 0011: +1.5
        2.0,  # 0100: +2.0
        3.0,  # 0101: +3.0
        4.0,  # 0110: +4.0
        6.0,  # 0111: +6.0
        -0.0,  # 1000: -0 (treat as 0)
        -0.5,  # 1001: -0.5
        -1.0,  # 1010: -1.0
        -1.5,  # 1011: -1.5
        -2.0,  # 1100: -2.0
        -3.0,  # 1101: -3.0
        -4.0,  # 1110: -4.0
        -6.0,  # 1111: -6.0
    ],
    dtype=np.float32,
)

# Precompute for vectorized nearest-value lookup
_E2M1_POSITIVE = E2M1_VALUES[:8]  # [0, 0.5, 1, 1.5, 2, 3, 4, 6]
_E2M1_NEGATIVE = -_E2M1_POSITIVE  # [0, -0.5, -1, -1.5, -2, -3, -4, -6]


def _blockwise_hadamard_transform(
    X: NDArray[np.float64],
    block_size: int,
    axis: int,
) -> NDArray[np.float64]:
    """Apply block-diagonal Hadamard transformation.

    Args:
        X: Input array
        block_size: Size of Hadamard blocks (must be power of 2)
        axis: Axis along which to apply transform (0 or 1)

    Returns:
        Transformed array with same shape as input
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    n = X.shape[axis]

    # Adjust block size if larger than dimension
    if block_size > n:
        # Use the largest power of 2 that divides n
        while block_size > n and block_size > 1:
            block_size //= 2

    if n % block_size != 0:
        # Skip transformation if dimensions don't align
        return X

    # Get normalized Hadamard matrix (hadamard_matrix returns normalized)
    Had = hadamard_matrix(block_size)

    num_blocks = n // block_size

    if axis == 0:
        # Process along rows
        other_axis = 1
        m = X.shape[other_axis]

        # Reshape: [n, m] -> [num_blocks, block_size, m]
        X_reshaped = X.reshape(num_blocks, block_size, m)

        # Apply Hadamard: Had @ X_block for each block
        Y_blocks = Had @ X_reshaped

        # Reshape back
        Y = Y_blocks.reshape(n, m)
    else:
        # Process along columns
        other_axis = 0
        m = X.shape[other_axis]

        # Reshape: [m, n] -> [m, num_blocks, block_size]
        X_reshaped = X.reshape(m, num_blocks, block_size)

        # Apply Hadamard: X_block @ Had.T for each block
        Y_blocks = X_reshaped @ Had.T

        # Reshape back
        Y = Y_blocks.reshape(m, n)

    return Y


def _decode_trellis_indices(
    trellis_indices: NDArray[np.uint8],
    codebook: TrellisCodebook,
) -> NDArray[np.float32]:
    """Decode EXL3 trellis indices to quantized weight values.

    EXL3 stores weights as trellis path indices. This function decodes
    those indices back to the quantized weight values using the
    configured codebook.

    For now, this implements a simplified uniform grid decoder.
    Full EXL3 decoding would require the exact codebook tables and
    trellis path reconstruction.

    Args:
        trellis_indices: Trellis indices from EXL3 format
        codebook: Codebook configuration

    Returns:
        Decoded weight values
    """
    # Simplified decoding: treat indices as direct quantization levels
    # In full EXL3, this would involve:
    # 1. Reconstructing trellis paths from indices
    # 2. Looking up values in MCG/Mul1 codebooks
    # 3. Applying dequantization scales

    n_levels = 2**codebook.bits

    # Map indices to [-1, 1] range (standard symmetric quantization)
    values = trellis_indices.astype(np.float32) / (n_levels - 1) * 2 - 1

    return values


def _inverse_hadamard_weight_rotation(
    weights: NDArray[np.float32],
    su: NDArray[np.float64],
    had_k: int = 128,
) -> NDArray[np.float32]:
    """Apply inverse Hadamard rotation to recover original weight space.

    EXL3 applies Hadamard rotation during quantization:
        W_rotated = Had @ diag(su) @ W @ diag(su) @ Had.T

    This function applies the inverse:
        W = diag(su) @ Had.T @ W_rotated @ Had @ diag(su)

    Since Hadamard matrices are orthogonal (Had.T = Had^-1), the inverse
    is simply the transpose operation.

    Args:
        weights: Hadamard-rotated weights [K, N]
        su: Sign flip vector [K] or [K + N] (±1 values)
            If length K + N, first K are for rows, last N are for columns
        had_k: Hadamard block size

    Returns:
        Weights in original space
    """
    k, n = weights.shape

    # Convert to float64 for precision
    w = weights.astype(np.float64)

    # Handle su vector - may contain both row and column sign flips
    if su.shape[0] == k + n:
        # Separate row and column sign flips
        su_row = su[:k]
        su_col = su[k : k + n]
    elif su.shape[0] == k:
        # Only row sign flips, use ones for columns
        su_row = su
        su_col = np.ones(n, dtype=np.float64)
    elif su.shape[0] == n:
        # Only column sign flips, use ones for rows
        su_row = np.ones(k, dtype=np.float64)
        su_col = su
    else:
        # Fallback: truncate or pad to match
        if su.shape[0] < max(k, n):
            # Pad with ones
            su_padded = np.ones(max(k, n), dtype=np.float64)
            su_padded[: su.shape[0]] = su
            su_row = su_padded[:k]
            su_col = su_padded[:n]
        else:
            # Truncate
            su_row = su[:k]
            su_col = su[:n]

    # Apply inverse Hadamard on both sides
    # Left: apply along axis 0 (rows)
    w = _blockwise_hadamard_transform(w, had_k, axis=0)

    # Right: apply along axis 1 (columns)
    w = _blockwise_hadamard_transform(w, had_k, axis=1)

    # Apply sign flips (su is its own inverse since su[i] ∈ {-1, +1})
    w = w * su_row[:, None]  # Left multiply by diag(su_row)
    w = w * su_col[None, :]  # Right multiply by diag(su_col)

    return w.astype(np.float32)


def _quantize_to_fp4_indices(values: np.ndarray) -> np.ndarray:
    """Quantize float values to FP4 E2M1 indices (0-15).

    Uses vectorized nearest-value matching. Each value maps to the closest
    representable E2M1 value, then encodes as 4-bit index.

    Args:
        values: Float array of any shape

    Returns:
        uint8 array of same shape with values in [0, 15]
    """
    flat = values.flatten().astype(np.float32)
    result = np.zeros(len(flat), dtype=np.uint8)

    # Split by sign for efficient lookup
    # Note: strict positive/negative; zero handled explicitly
    pos_mask = flat > 0
    zero_mask = flat == 0
    neg_mask = flat < 0

    # Positive values: find nearest in [0.5, 1, 1.5, 2, 3, 4, 6]
    # Index 0 is +0, handled separately
    if np.any(pos_mask):
        pos_vals = flat[pos_mask]
        pos_vals = np.clip(pos_vals, 0, 6.0)
        dists = np.abs(pos_vals[:, None] - _E2M1_POSITIVE[None, :])
        result[pos_mask] = np.argmin(dists, axis=1).astype(np.uint8)

    # Zero maps to index 0 (+0)
    if np.any(zero_mask):
        result[zero_mask] = 0

    # Negative values: find nearest in [-0.5, -1, -1.5, -2, -3, -4, -6]
    # Index 8 is -0
    if np.any(neg_mask):
        neg_vals = flat[neg_mask]
        neg_vals = np.clip(neg_vals, -6.0, 0)
        dists = np.abs(neg_vals[:, None] - _E2M1_NEGATIVE[None, :])
        result[neg_mask] = (np.argmin(dists, axis=1) + 8).astype(np.uint8)

    return result.reshape(values.shape)


def _pack_fp4_to_marlin(
    weights: NDArray[np.float32],
    scales: NDArray[np.float32],
    group_size: int = 128,
) -> tuple[NDArray[np.uint32], NDArray[np.float16]]:
    """Pack FP4-quantized weights to Marlin format.

    Marlin format:
    - Packed weights: [K/8, N] uint32, 8 FP4 nibbles per uint32
    - Scales: [K/group_size, N] float16

    The weights are transposed so K = in_features, N = out_features.

    Args:
        weights: Dequantized weights [out_features, in_features]
        scales: Per-group scales (for rescaling)
        group_size: Quantization group size

    Returns:
        (packed_weights, marlin_scales) in Marlin layout
    """
    out_feat, in_feat = weights.shape

    # Transpose to [K, N] = [in_feat, out_feat] for Marlin layout
    w = weights.T.astype(np.float32)  # [K, N]
    K, N = w.shape

    if K % 8 != 0:
        raise ValueError(f"K ({K}) must be divisible by 8 for FP4 packing")
    if K % group_size != 0:
        raise ValueError(f"K ({K}) must be divisible by group_size ({group_size})")

    n_groups = K // group_size

    # Reshape for per-group processing: [n_groups, group_size, N]
    w_grouped = w.reshape(n_groups, group_size, N)

    # Compute per-group max for scaling to [-6, 6] range
    group_max = np.max(np.abs(w_grouped), axis=1, keepdims=True)
    group_max = np.maximum(group_max, 1e-7)  # Avoid division by zero

    # Compute scales: scale = max / 6.0
    marlin_scales = (group_max / 6.0).astype(np.float16).squeeze(1)  # [n_groups, N]

    # Scale weights to [-6, 6] range
    w_scaled = w_grouped / group_max * 6.0
    w_scaled = w_scaled.reshape(K, N)

    # Quantize to FP4 indices
    fp4_indices = _quantize_to_fp4_indices(w_scaled)  # [K, N]

    # Pack 8 FP4 values into each uint32 along K dimension
    # Layout: [v0, v1, v2, v3, v4, v5, v6, v7] -> bits [3:0, 7:4, ..., 31:28]
    packed = np.zeros((K // 8, N), dtype=np.uint32)
    for i in range(8):
        packed |= fp4_indices[i::8, :].astype(np.uint32) << (i * 4)

    return packed, marlin_scales


def exl3_layer_to_marlin(
    trellis_indices: NDArray[np.uint8],
    scales: NDArray[np.float32],
    su: NDArray[np.float64],
    codebook: TrellisCodebook,
) -> tuple[NDArray[np.uint32], NDArray[np.float16]]:
    """Convert single layer from EXL3 to Marlin format.

    Steps:
    1. Decode trellis indices to get quantized values
    2. Apply inverse Hadamard to get original weight space
    3. Re-quantize to Marlin FP4 nibble packing

    Args:
        trellis_indices: EXL3 trellis indices [tiles_k, tiles_n, 256] or similar
        scales: EXL3 per-tile scales [out_features, n_groups]
        su: Sign flip vector [in_features] (±1 values)
        codebook: Trellis codebook configuration

    Returns:
        (packed_weights, marlin_scales) where:
        - packed_weights: [K/8, N] uint32 with 8 FP4 nibbles per element
        - marlin_scales: [K/group_size, N] float16

    Example:
        >>> indices = np.random.randint(0, 16, (16, 16, 256), dtype=np.uint8)
        >>> scales = np.random.randn(256, 2).astype(np.float32)
        >>> su = np.sign(np.random.randn(512))
        >>> codebook = TrellisCodebook(bits=4)
        >>> packed, m_scales = exl3_layer_to_marlin(indices, scales, su, codebook)
    """
    # Step 1: Decode trellis indices to get quantized weight values
    # Reshape indices to 2D weight matrix format
    # EXL3 stores as tiles; we need to reconstruct the full weight matrix

    # For simplicity, assume indices can be reshaped to [out_feat, in_feat]
    # In practice, EXL3 tile layout may need more complex reconstruction
    decoded = _decode_trellis_indices(trellis_indices, codebook)

    # Flatten and reshape to weight matrix
    # Assuming the indices encode a [out_features, in_features] matrix
    # The exact reshaping depends on EXL3's tile structure
    total_elements = decoded.size

    # Infer dimensions from scales shape if needed
    # scales shape: [out_features, n_groups] or similar
    if scales.ndim >= 2:
        out_features = scales.shape[0]
        # Estimate in_features from total elements
        in_features = total_elements // out_features
    else:
        # Square matrix assumption as fallback
        dim = int(np.sqrt(total_elements))
        out_features, in_features = dim, dim

    weights = decoded.flatten()[: out_features * in_features].reshape(out_features, in_features)

    # Step 2: Apply inverse Hadamard rotation
    # su should match the input dimension (in_features)
    if su.shape[0] != in_features:
        # Pad or truncate su to match
        if su.shape[0] < in_features:
            su_padded = np.ones(in_features, dtype=np.float64)
            su_padded[: su.shape[0]] = su
            su = su_padded
        else:
            su = su[:in_features]

    weights_original = _inverse_hadamard_weight_rotation(weights.astype(np.float32), su, had_k=128)

    # Step 3: Re-quantize to Marlin FP4 format
    # Infer group_size from scales shape
    if scales.ndim >= 2:
        n_groups = scales.shape[1] if scales.shape[1] > 1 else scales.shape[0]
        group_size = max(in_features // n_groups, 128)
    else:
        group_size = 128

    packed, marlin_scales = _pack_fp4_to_marlin(weights_original, scales, group_size=group_size)

    return packed, marlin_scales


def convert_exl3_to_marlin(
    exl3_path: Path,
    output_path: Path,
    verbose: bool = True,
) -> dict[str, Any]:
    """Convert EXL3 quantized model to Marlin FP4 format.

    EXL3 stores trellis indices; Marlin needs packed nibbles + scales.
    This conversion dequantizes EXL3 and re-packs for Marlin kernels.

    For inference, you can either:
    1. Use this converter for Marlin kernels (current path)
    2. Add native EXL3 kernels (future work)

    Args:
        exl3_path: Path to EXL3 model directory or file
        output_path: Path for output Marlin-format model
        verbose: Whether to print progress information

    Returns:
        Conversion stats dict with:
        - layers_converted: Number of layers converted
        - total_params: Total parameter count
        - compression_ratio: Original size / converted size
        - errors: List of any errors encountered

    Example:
        >>> stats = convert_exl3_to_marlin(
        ...     Path("model-exl3"),
        ...     Path("model-marlin.safetensors"),
        ...     verbose=True,
        ... )
        >>> print(f"Converted {stats['layers_converted']} layers")
    """
    exl3_path = Path(exl3_path)
    output_path = Path(output_path)

    stats = {
        "layers_converted": 0,
        "total_params": 0,
        "original_bytes": 0,
        "converted_bytes": 0,
        "compression_ratio": 1.0,
        "errors": [],
    }

    if verbose:
        print(f"Converting EXL3 model: {exl3_path}")
        print(f"Output path: {output_path}")

    # Full model loading from EXL3 format would require:
    # 1. Loading EXL3 model files (safetensors with trellis data)
    # 2. Iterating through layers
    # 3. Converting each layer with exl3_layer_to_marlin
    # 4. Saving to output format

    # For now, this is a placeholder that shows the structure
    if not exl3_path.exists():
        stats["errors"].append(f"Input path does not exist: {exl3_path}")
        return stats

    if verbose:
        print("Note: Full EXL3 model loading not yet implemented.")
        print("Use exl3_layer_to_marlin() for individual layer conversion.")

    return stats
