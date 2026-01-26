"""GPTQ quantization adapted for FP4 E2M1 non-uniform quantization grid.

Standard GPTQ assumes uniform quantization levels (INT4: -8 to 7).
FP4 E2M1 has non-uniform levels: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}

This module adapts the GPTQ algorithm for the FP4 grid by:
1. Using nearest-neighbor quantization to the non-uniform grid
2. Optimizing scales to minimize FP4 quantization error
3. Performing column-wise error compensation with FP4-aware rounding

References:
- GPTQ paper: arxiv:2210.17323
- MR-GPTQ approach from vLLM Marlin
- AutoGPTQ implementation: https://github.com/AutoGPTQ/AutoGPTQ

Usage:
    from metal_marlin.gptq_fp4 import FP4GPTQQuantizer

    quantizer = FP4GPTQQuantizer(group_size=128, actorder=True)
    packed, scales, meta = quantizer.quantize_weight(W, H)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

# FP4 E2M1 codebook: the 16 representable values
# Nibble encoding: [sign(1) | exp(2) | mant(1)], bias = 1
# Positive half: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0  (indices 0-7)
# Negative half: -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0  (indices 8-15)
FP4_GRID: NDArray[np.float32] = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

# Maximum representable magnitude
FP4_MAX: float = 6.0

# Positive grid for scale optimization (symmetric, ignore sign)
FP4_POSITIVE_GRID: NDArray[np.float32] = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32
)

# Packing factor: 8 FP4 nibbles per uint32
FP4_PER_U32: int = 8


@dataclass
class FP4GPTQResult:
    """Result of FP4 GPTQ quantization."""

    packed: NDArray[np.uint32]  # [K, N // 8] packed nibbles
    scales: NDArray[np.float16]  # [K // group_size, N] per-group scales
    quant_indices: NDArray[np.uint8]  # [K, N] nibble indices (unpacked, for debugging)
    mse: float  # Mean squared error vs original
    meta: dict  # Metadata (orig_K, orig_N, group_size, etc.)


def quantize_to_fp4_grid_vectorized(
    x: NDArray[np.float32],
    scale: NDArray[np.float32] | float,
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Map scaled values to nearest FP4 E2M1 grid points (vectorized).

    Args:
        x: Float values to quantize.
        scale: Per-element or scalar scale factor(s).

    Returns:
        Tuple of:
            indices: uint8 array of nibble indices (0-15) into FP4_GRID
            quantized: float32 array of dequantized values (scale * FP4_GRID[idx])
    """
    scale = np.asarray(scale, dtype=np.float32)
    scale = np.maximum(scale, 1e-10)
    x_scaled = x / scale

    # Clip to representable range
    x_clipped = np.clip(x_scaled, -FP4_MAX, FP4_MAX)

    # Find nearest FP4 grid point via argmin over distances
    # Process in chunks to avoid memory explosion
    flat = x_clipped.ravel()
    n = len(flat)
    chunk_size = 1 << 20
    indices = np.empty(n, dtype=np.uint8)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = flat[start:end]
        dists = np.abs(chunk[:, None] - FP4_GRID[None, :])
        indices[start:end] = dists.argmin(axis=1).astype(np.uint8)

    indices = indices.reshape(x.shape)
    quantized = FP4_GRID[indices] * scale

    return indices, quantized


def compute_optimal_fp4_scale(
    w_group: NDArray[np.float32],
    method: Literal["maxabs", "mse_grid", "mse_newton"] = "mse_grid",
    grid_points: int = 32,
) -> float:
    """Compute optimal scale for FP4 quantization of a weight group.

    For FP4, optimal scale minimizes: sum((W - scale * FP4_GRID[idx])²)
    This is different from INT4 where scale = max(|W|) / 7 is optimal.

    Args:
        w_group: Weight values to quantize (1D or 2D).
        method: Scale optimization method:
            - "maxabs": Simple max(|W|) / 6.0 (fast but suboptimal)
            - "mse_grid": Grid search over scale values (better quality)
            - "mse_newton": Newton optimization (best quality, slower)
        grid_points: Number of grid points for "mse_grid" method.

    Returns:
        Optimal scale factor.
    """
    w_flat = w_group.ravel().astype(np.float32)
    max_abs = np.max(np.abs(w_flat))

    if max_abs < 1e-10:
        return 1e-7

    if method == "maxabs":
        return max_abs / FP4_MAX

    elif method == "mse_grid":
        base_scale = max_abs / FP4_MAX
        scale_candidates = np.linspace(
            base_scale * 0.5, base_scale * 1.5, grid_points
        )

        best_scale = base_scale
        best_mse = float("inf")

        for scale in scale_candidates:
            _, quantized = quantize_to_fp4_grid_vectorized(w_flat, scale)
            mse = np.mean((w_flat - quantized) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_scale = float(scale)

        return best_scale

    elif method == "mse_newton":
        scale = max_abs / FP4_MAX

        for _ in range(10):
            indices, quantized = quantize_to_fp4_grid_vectorized(w_flat, scale)
            error = w_flat - quantized
            mse = np.mean(error ** 2)

            if mse < 1e-12:
                break

            grid_values = FP4_GRID[indices]
            gradient = -2.0 * np.mean(error * grid_values)
            hessian = 2.0 * np.mean(grid_values ** 2)

            if abs(hessian) < 1e-12:
                break

            step = gradient / (hessian + 0.1 * abs(gradient))
            scale = max(scale - 0.5 * step, 1e-10)

        return float(scale)

    else:
        raise ValueError(f"Unknown scale optimization method: {method}")


class FP4GPTQQuantizer:
    """GPTQ quantizer adapted for FP4 E2M1 non-uniform grid.

    The GPTQ algorithm quantizes weights column-by-column (along the input dimension),
    compensating quantization errors in remaining columns using the inverse Hessian.

    Key differences from standard INT4 GPTQ:
    1. Quantization maps to non-uniform FP4 levels, not uniform integers
    2. Scale optimization accounts for FP4 spacing via grid search
    3. Error compensation uses FP4-aware nearest-neighbor rounding

    Example:
        >>> quantizer = FP4GPTQQuantizer(group_size=128, actorder=True)
        >>> H = compute_hessian(calibration_data)  # [in_features, in_features]
        >>> result = quantizer.quantize_weight(W, H)
        >>> print(f"MSE: {result.mse:.6f}")
    """

    def __init__(
        self,
        group_size: int = 128,
        actorder: bool = True,
        damp_factor: float = 0.01,
        scale_method: Literal["maxabs", "mse_grid", "mse_newton"] = "mse_grid",
        block_size: int = 128,
    ):
        """Initialize FP4 GPTQ quantizer.

        Args:
            group_size: Number of elements per quantization group.
            actorder: If True, quantize columns in Hessian-diagonal order.
            damp_factor: Damping factor for Hessian inverse.
            scale_method: Method for computing optimal scales.
            block_size: Column block size for blocked GPTQ.
        """
        self.group_size = group_size
        self.actorder = actorder
        self.damp_factor = damp_factor
        self.scale_method = scale_method
        self.block_size = block_size

    def quantize_weight(
        self,
        W: NDArray[np.float32],
        H: NDArray[np.float32] | None = None,
    ) -> FP4GPTQResult:
        """Quantize weight matrix using GPTQ with FP4 grid.

        This implements the blocked GPTQ algorithm from the original paper,
        adapted for the non-uniform FP4 E2M1 grid.

        The algorithm processes columns in blocks:
        1. For each block, quantize all columns
        2. Accumulate error compensation within the block
        3. Apply block error to remaining columns outside the block

        Args:
            W: Weight matrix [out_features, in_features].
            H: Hessian matrix [in_features, in_features] from calibration.
               If None, uses identity (equivalent to RTN).

        Returns:
            FP4GPTQResult containing packed weights, scales, and metadata.
        """
        W = np.asarray(W, dtype=np.float32)
        W_orig = W.copy()  # Keep original for MSE computation
        out_features, in_features = W.shape
        orig_K = in_features
        orig_N = out_features

        # Pad K to multiple of group_size
        if in_features % self.group_size != 0:
            pad_k = self.group_size - (in_features % self.group_size)
            W = np.pad(W, ((0, 0), (0, pad_k)), mode="constant", constant_values=0)
            in_features = W.shape[1]

        # Pad N to multiple of 8 for packing
        if out_features % FP4_PER_U32 != 0:
            pad_n = FP4_PER_U32 - (out_features % FP4_PER_U32)
            W = np.pad(W, ((0, pad_n), (0, 0)), mode="constant", constant_values=0)
            out_features = W.shape[0]

        # Handle Hessian
        if H is None:
            H = np.eye(in_features, dtype=np.float32)
        else:
            H = np.asarray(H, dtype=np.float32)
            if H.shape[0] < in_features:
                H_padded = np.eye(in_features, dtype=np.float32) * np.mean(np.diag(H))
                H_padded[: H.shape[0], : H.shape[1]] = H
                H = H_padded

        # Add damping to Hessian diagonal
        damp = self.damp_factor * np.mean(np.diag(H))
        H = H + damp * np.eye(in_features, dtype=np.float32)

        # Compute activation order permutation
        if self.actorder:
            perm = np.argsort(np.diag(H))[::-1]  # Descending by importance
            inv_perm = np.argsort(perm)
            W = W[:, perm]
            H = H[np.ix_(perm, perm)]
        else:
            perm = np.arange(in_features)
            inv_perm = perm

        # Allocate output arrays
        num_groups = in_features // self.group_size
        Q_indices = np.zeros((out_features, in_features), dtype=np.uint8)
        scales = np.zeros((num_groups, out_features), dtype=np.float32)

        # Compute Cholesky decomposition: H = L @ L^T
        try:
            np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            # Add more damping if Cholesky fails
            H = H + 0.1 * np.eye(in_features, dtype=np.float32)
            np.linalg.cholesky(H)

        # Compute H^{-1} for error compensation
        H_inv = np.linalg.inv(H)

        # Pre-compute all scales from original weights before error compensation
        # This is crucial: GPTQ error compensation changes W, but scales should
        # be computed from original weights for consistent quantization
        for g in range(num_groups):
            gs = g * self.group_size
            ge = gs + self.group_size
            for row in range(out_features):
                scales[g, row] = compute_optimal_fp4_scale(
                    W[row, gs:ge], method=self.scale_method
                )

        # Work on a copy of W for GPTQ error compensation
        W_work = W.copy()

        # Process in blocks
        for block_start in range(0, in_features, self.block_size):
            block_end = min(block_start + self.block_size, in_features)
            block_size_actual = block_end - block_start

            # Error accumulator for this block
            Err = np.zeros((out_features, block_size_actual), dtype=np.float32)

            # Quantize columns in this block
            for i in range(block_size_actual):
                col_idx = block_start + i
                group_idx = col_idx // self.group_size

                # Get current column (may be modified by error compensation)
                w_col = W_work[:, col_idx].copy()

                # Get pre-computed scale for this column's group
                col_scale = scales[group_idx, :]

                # Quantize column to FP4 grid
                indices, q_col = quantize_to_fp4_grid_vectorized(w_col, col_scale)
                Q_indices[:, col_idx] = indices

                # Compute quantization error (GPTQ formula)
                err = (w_col - q_col) / H_inv[col_idx, col_idx]
                Err[:, i] = err

                # Update remaining columns in this block with error
                for j in range(i + 1, block_size_actual):
                    W_work[:, block_start + j] -= err * H_inv[col_idx, block_start + j]

            # Apply block error to remaining columns outside this block
            if block_end < in_features:
                W_work[:, block_end:] -= Err @ H_inv[block_start:block_end, block_end:]

        # Apply inverse permutation to get back to original order
        Q_indices = Q_indices[:, inv_perm]
        scales_reordered = np.zeros_like(scales)
        for g in range(num_groups):
            orig_group_idx = inv_perm[g * self.group_size] // self.group_size
            scales_reordered[orig_group_idx, :] = scales[g, :]
        scales = scales_reordered

        # Pack indices into uint32
        # Layout: packed[k, g] contains 8 columns for row k
        n_packed = out_features // FP4_PER_U32
        packed = np.zeros((in_features, n_packed), dtype=np.uint32)

        # Transpose Q_indices to [K, N] for packing
        Q_T = Q_indices.T  # [in_features, out_features]

        for g in range(n_packed):
            col_start = g * FP4_PER_U32
            for i in range(FP4_PER_U32):
                packed[:, g] |= Q_T[:, col_start + i].astype(np.uint32) << (i * 4)

        # Compute MSE for quality assessment
        # Reconstruct from Q_indices (which is in original column order)
        W_reconstructed = np.zeros((out_features, in_features), dtype=np.float32)
        for row in range(out_features):
            for g in range(num_groups):
                gs = g * self.group_size
                ge = gs + self.group_size
                scale = scales[g, row]
                W_reconstructed[row, gs:ge] = FP4_GRID[Q_indices[row, gs:ge]] * scale

        # Compare against original (before padding/permutation)
        mse = float(np.mean((W_orig - W_reconstructed[:orig_N, :orig_K]) ** 2))

        meta = {
            "orig_K": orig_K,
            "orig_N": orig_N,
            "padded_K": in_features,
            "padded_N": out_features,
            "group_size": self.group_size,
            "actorder": self.actorder,
            "scale_method": self.scale_method,
        }

        return FP4GPTQResult(
            packed=packed,
            scales=scales.astype(np.float16),
            quant_indices=Q_indices,
            mse=mse,
            meta=meta,
        )


def fp4_gptq_quantize_weight(
    W: NDArray[np.float32],
    H: NDArray[np.float32] | None = None,
    group_size: int = 128,
    actorder: bool = True,
    scale_method: Literal["maxabs", "mse_grid", "mse_newton"] = "mse_grid",
) -> tuple[NDArray[np.uint32], NDArray[np.float16], dict]:
    """Convenience function for FP4 GPTQ quantization.

    GPTQ quantization to FP4 E2M1 grid.

    Key difference from INT4 GPTQ:
    - Quantization uses non-uniform FP4 grid
    - Scale optimization accounts for FP4 spacing
    - Error compensation uses FP4-aware rounding

    Args:
        W: Weight matrix [out_features, in_features].
        H: Hessian matrix [in_features, in_features] from calibration.
           If None, falls back to RTN (Round-to-Nearest).
        group_size: Elements per quantization group.
        actorder: If True, quantize columns by Hessian diagonal importance.
        scale_method: Method for computing optimal scales.

    Returns:
        Tuple of (packed, scales, meta):
            packed: uint32 array [K_padded, N_padded // 8]
            scales: float16 array [K_padded // group_size, N_padded]
            meta: Dict with 'orig_K', 'orig_N', etc.
    """
    quantizer = FP4GPTQQuantizer(
        group_size=group_size,
        actorder=actorder,
        scale_method=scale_method,
    )
    result = quantizer.quantize_weight(W, H)
    return result.packed, result.scales, result.meta


def dequantize_fp4_gptq(
    packed: NDArray[np.uint32],
    scales: NDArray[np.float16],
    meta: dict,
) -> NDArray[np.float16]:
    """Dequantize FP4 GPTQ packed weights back to float.

    Args:
        packed: uint32 array [K_padded, N_padded // 8].
        scales: float16 array [K_padded // group_size, N_padded].
        meta: Metadata dict from quantization.

    Returns:
        Dequantized weights [orig_N, orig_K] in float16.
    """
    K = meta["padded_K"]
    N = meta["padded_N"]
    group_size = meta["group_size"]
    orig_K = meta["orig_K"]
    orig_N = meta["orig_N"]

    # Unpack nibbles
    n_packed = N // FP4_PER_U32
    indices = np.empty((K, N), dtype=np.uint8)

    for g in range(n_packed):
        col_start = g * FP4_PER_U32
        for i in range(FP4_PER_U32):
            indices[:, col_start + i] = (
                (packed[:, g] >> (i * 4)) & 0xF
            ).astype(np.uint8)

    # Transpose to [N, K] layout
    indices_T = indices.T

    # Dequantize using FP4 grid and scales
    W = np.zeros((N, K), dtype=np.float32)
    num_groups = K // group_size
    scales_f32 = scales.astype(np.float32)

    for row in range(N):
        for group_idx in range(num_groups):
            gs = group_idx * group_size
            ge = gs + group_size
            scale = scales_f32[group_idx, row]
            W[row, gs:ge] = FP4_GRID[indices_T[row, gs:ge]] * scale

    return W[:orig_N, :orig_K].astype(np.float16)


def compare_rtn_vs_gptq(
    W: NDArray[np.float32],
    H: NDArray[np.float32],
    group_size: int = 128,
) -> dict:
    """Compare RTN vs GPTQ quantization quality.

    Args:
        W: Weight matrix.
        H: Hessian matrix.
        group_size: Elements per quantization group.

    Returns:
        Dict with 'rtn_mse', 'gptq_mse', 'improvement_pct'.
    """
    # RTN: H=None (identity Hessian), no actorder
    quantizer_rtn = FP4GPTQQuantizer(group_size=group_size, actorder=False)
    result_rtn = quantizer_rtn.quantize_weight(W, H=None)

    # GPTQ: with actual Hessian and actorder
    quantizer_gptq = FP4GPTQQuantizer(group_size=group_size, actorder=True)
    result_gptq = quantizer_gptq.quantize_weight(W, H)

    improvement = (result_rtn.mse - result_gptq.mse) / result_rtn.mse * 100

    return {
        "rtn_mse": result_rtn.mse,
        "gptq_mse": result_gptq.mse,
        "improvement_pct": improvement,
    }


def compute_hessian_from_activations(
    X: NDArray[np.float32],
    damp_factor: float = 0.01,
) -> NDArray[np.float32]:
    """Compute Hessian approximation from calibration activations.

    The Hessian H = X^T @ X captures the importance of each input feature.
    This is used by GPTQ to prioritize quantization order and error compensation.

    Args:
        X: Calibration activations [num_samples, in_features].
        damp_factor: Damping factor as fraction of mean diagonal.

    Returns:
        Hessian matrix [in_features, in_features].
    """
    X = np.asarray(X, dtype=np.float32)
    num_samples = X.shape[0]

    # H = X^T @ X / num_samples
    H = (X.T @ X) / num_samples

    return H


# ============================================================================
# CLI / Test
# ============================================================================


def main():
    """Test FP4 GPTQ quantization on random tensor."""
    import argparse

    parser = argparse.ArgumentParser(description="Test FP4 GPTQ quantization")
    parser.add_argument(
        "--shape", type=str, default="1024,1024", help="Weight shape (out,in)"
    )
    parser.add_argument("--group-size", type=int, default=128, help="Group size")
    parser.add_argument(
        "--scale-method",
        choices=["maxabs", "mse_grid", "mse_newton"],
        default="mse_grid",
    )
    parser.add_argument("--actorder", action="store_true", help="Use activation order")
    parser.add_argument("--compare", action="store_true", help="Compare RTN vs GPTQ")
    args = parser.parse_args()

    out_feat, in_feat = (int(x) for x in args.shape.split(","))

    # Ensure divisibility
    in_feat = (in_feat // args.group_size) * args.group_size

    print(f"Testing FP4 GPTQ on weight [{out_feat}, {in_feat}]")
    print(f"Group size: {args.group_size}")
    print(f"Scale method: {args.scale_method}")
    print(f"Activation order: {args.actorder}")
    print()

    # Generate random Gaussian weights (transformer-like)
    np.random.seed(42)
    W = np.random.randn(out_feat, in_feat).astype(np.float32) * 0.02

    # Generate synthetic Hessian (X^T @ X for random X)
    calib_samples = 256
    X = np.random.randn(calib_samples, in_feat).astype(np.float32) * 0.1
    H = compute_hessian_from_activations(X)

    if args.compare:
        print("=== RTN vs GPTQ Comparison ===")
        result = compare_rtn_vs_gptq(W, H, args.group_size)
        print(f"  RTN MSE:   {result['rtn_mse']:.8f}")
        print(f"  GPTQ MSE:  {result['gptq_mse']:.8f}")
        print(f"  Improvement: {result['improvement_pct']:.1f}%")
        print()

    print("=== FP4 GPTQ Quantization ===")
    quantizer = FP4GPTQQuantizer(
        group_size=args.group_size,
        actorder=args.actorder,
        scale_method=args.scale_method,
    )
    result = quantizer.quantize_weight(W, H)

    print(f"  Packed shape: {result.packed.shape}")
    print(f"  Scales shape: {result.scales.shape}")
    print(f"  MSE: {result.mse:.8f}")
    print(f"  RMSE: {np.sqrt(result.mse):.8f}")

    # Compute compression ratio
    original_bits = out_feat * in_feat * 32  # FP32
    packed_bits = result.packed.size * 32 + result.scales.size * 16
    compression = original_bits / packed_bits
    print(f"  Compression vs FP32: {compression:.2f}x")


if __name__ == "__main__":
    main()
