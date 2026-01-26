"""
Standalone FP16 → FP4 E2M1 quantizer.

No MLX dependency. Uses numpy for weight processing, outputs packed uint32
arrays ready for Metal kernels.

Usage:
    from quantize_fp4 import quantize_fp4, load_and_quantize_safetensors

    # Quantize a single tensor
    packed, scales = quantize_fp4(weight_fp16, group_size=128)

    # Quantize entire model from safetensors
    tensors = load_and_quantize_safetensors("model.safetensors", group_size=128)
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

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


def quantize_to_fp4(values: np.ndarray) -> np.ndarray:
    """
    Quantize float values to FP4 E2M1 indices (0-15).

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
    pos_mask = flat >= 0
    neg_mask = ~pos_mask

    # Positive values: find nearest in [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    if np.any(pos_mask):
        pos_vals = flat[pos_mask]
        # Clamp to representable range
        pos_vals = np.clip(pos_vals, 0, 6.0)
        # Find nearest (vectorized)
        dists = np.abs(pos_vals[:, None] - _E2M1_POSITIVE[None, :])
        result[pos_mask] = np.argmin(dists, axis=1).astype(np.uint8)

    # Negative values: find nearest in [-0, -0.5, -1, -1.5, -2, -3, -4, -6]
    if np.any(neg_mask):
        neg_vals = flat[neg_mask]
        neg_vals = np.clip(neg_vals, -6.0, 0)
        dists = np.abs(neg_vals[:, None] - _E2M1_NEGATIVE[None, :])
        result[neg_mask] = (np.argmin(dists, axis=1) + 8).astype(np.uint8)

    return result.reshape(values.shape)


def dequantize_fp4(indices: np.ndarray) -> np.ndarray:
    """Dequantize FP4 indices back to float values."""
    return E2M1_VALUES[indices.astype(np.int32)]


def quantize_fp4(
    weight: np.ndarray,
    group_size: int = 128,
    activation_ranges: dict[str, tuple[float, float]] | None = None,
    marlin_layout: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize a 2D weight matrix to FP4 E2M1 with per-group scales.

    If activation_ranges provided, use them to set optimal scale
    instead of just min/max of static weights.

    Calibration-aware scaling:
    1. If we know input activations range [a_min, a_max]
    2. And weight values range [w_min, w_max]
    3. Output range is roughly [a_min*w_min, a_max*w_max]
    4. Use this to set scales that minimize quantization error
       for the actual runtime value distribution

    Args:
        weight: FP16/FP32 weight tensor, shape [out_features, in_features]
        group_size: Elements per quantization group (along reduction axis)
        activation_ranges: Optional dict with calibration data:
            - "input_range": (min, max) of input activations seen during calibration
            - "output_range": (min, max) of output activations (optional, for validation)
            - "percentile": float in (0, 100], use percentile of values instead of max
              for robust outlier handling (e.g., 99.9 clips top 0.1%)
            - "smooth_factor": float in [0, 1], blend between weight-only and
              calibration-aware scales (0 = weight-only, 1 = fully calibrated)
        marlin_layout: If True (default), output Marlin-compatible layout:
            - packed: [K/8, N] where K=in_features, N=out_features
            - scales: [K/group_size, N]
            This transposes the weight and packs along K (input features).
            If False, use legacy layout [out_features, in_features/8].

    Returns:
        (packed_weights, scales) where layout depends on marlin_layout flag.
    """
    w = weight.astype(np.float32)
    out_feat, in_feat = w.shape

    # For Marlin layout, transpose to [K, N] = [in_feat, out_feat]
    # and quantize/pack along K (axis 0)
    if marlin_layout:
        w = w.T  # [in_feat, out_feat] = [K, N]
        K, N = w.shape
        pack_axis_size = K
        other_axis_size = N
    else:
        K, N = in_feat, out_feat  # For error messages
        pack_axis_size = in_feat
        other_axis_size = out_feat

    if pack_axis_size % 8 != 0:
        raise ValueError(f"Pack axis ({pack_axis_size}) must be divisible by 8")
    if pack_axis_size % group_size != 0:
        raise ValueError(
            f"Pack axis ({pack_axis_size}) must be divisible by group_size ({group_size})"
        )

    n_groups = pack_axis_size // group_size

    # Reshape for per-group processing
    # marlin_layout: w is [K, N], groups along K → [n_groups, group_size, N]
    # legacy:        w is [out, in], groups along in → [out, n_groups, group_size]
    if marlin_layout:
        # [K, N] → [n_groups, group_size, N]
        w_grouped = w.reshape(n_groups, group_size, other_axis_size)
        group_axis = 1  # group elements along axis 1
    else:
        # [out, in] → [out, n_groups, group_size]
        w_grouped = w.reshape(other_axis_size, n_groups, group_size)
        group_axis = 2  # group elements along axis 2

    # Compute per-group scales (max abs value → scale to [-6, 6])
    # FP4 E2M1 max representable magnitude is 6.0
    if activation_ranges is not None:
        # Calibration-aware quantization
        percentile = activation_ranges.get("percentile", 100.0)
        smooth_factor = activation_ranges.get("smooth_factor", 1.0)
        input_range = activation_ranges.get("input_range")

        # Compute weight-based scales (baseline)
        if percentile < 100.0:
            # Use percentile for robust outlier handling
            group_max_weight = np.percentile(
                np.abs(w_grouped), percentile, axis=group_axis, keepdims=True
            )
        else:
            group_max_weight = np.max(np.abs(w_grouped), axis=group_axis, keepdims=True)

        group_max_weight = np.maximum(group_max_weight, 1e-7)

        if input_range is not None:
            # Calibration-aware scaling: adjust based on activation magnitudes
            #
            # Key insight: weights that interact with higher-magnitude activations
            # should use larger scales to preserve precision where it matters.
            #
            # For a linear layer y = W @ x:
            #   - If x[i] tends to be large, W[:, i] values contribute more to output
            #   - Quantization error in W[:, i] is amplified by |x[i]|
            #
            # We approximate this by scaling the weight ranges by the expected
            # activation magnitude ratio.
            a_min, a_max = input_range
            act_scale = max(abs(a_min), abs(a_max))

            # Activation-weighted scale adjustment
            # Higher act_scale → larger quantization scale → coarser but fewer outliers
            # Lower act_scale → smaller quantization scale → finer but may clip
            #
            # Heuristic: if activations are typically larger than weights,
            # we want slightly larger scales to reduce quantization error in
            # the high-magnitude product terms.
            if act_scale > 1.0:
                # Activations amplify weights: use percentile to clip outliers
                # rather than increasing scale (which would lose precision)
                calibrated_max = group_max_weight
            else:
                # Activations attenuate weights: can afford smaller scales
                # for better precision on smaller products
                attenuation = np.sqrt(act_scale)  # Geometric mean adjustment
                calibrated_max = group_max_weight * attenuation

            # Blend between weight-only and calibrated scales
            group_max = (1.0 - smooth_factor) * group_max_weight + smooth_factor * calibrated_max
        else:
            # No input range provided, just use percentile-based max
            group_max = group_max_weight
    else:
        # Standard weight-only quantization
        group_max = np.max(np.abs(w_grouped), axis=group_axis, keepdims=True)
        group_max = np.maximum(group_max, 1e-7)  # Avoid division by zero

    # Squeeze and compute scales
    # marlin_layout: group_max is [n_groups, 1, N] → scales [n_groups, N]
    # legacy:        group_max is [out, n_groups, 1] → scales [out, n_groups]
    scales = (group_max / 6.0).astype(np.float16).squeeze(group_axis)

    # Scale weights to [-6, 6] range per group
    w_scaled = w_grouped / group_max.astype(np.float32) * 6.0

    # Reshape back to 2D for quantization
    if marlin_layout:
        w_scaled = w_scaled.reshape(pack_axis_size, other_axis_size)  # [K, N]
    else:
        w_scaled = w_scaled.reshape(other_axis_size, pack_axis_size)  # [out, in]

    # Quantize to FP4 indices
    fp4_indices = quantize_to_fp4(w_scaled)  # same shape as w_scaled

    # Pack 8 FP4 values into each uint32
    # Layout: [v0, v1, v2, v3, v4, v5, v6, v7] → bits [3:0, 7:4, 11:8, 15:12, 19:16, 23:20, 27:24, 31:28]
    if marlin_layout:
        # Pack along axis 0 (K): [K, N] → [K/8, N]
        packed = np.zeros((pack_axis_size // 8, other_axis_size), dtype=np.uint32)
        for i in range(8):
            packed |= fp4_indices[i::8, :].astype(np.uint32) << (i * 4)
    else:
        # Pack along axis 1 (in_feat): [out, in] → [out, in/8]
        packed = np.zeros((other_axis_size, pack_axis_size // 8), dtype=np.uint32)
        for i in range(8):
            packed |= fp4_indices[:, i::8].astype(np.uint32) << (i * 4)

    return packed, scales


def unpack_fp4(
    packed: np.ndarray,
    scales: np.ndarray,
    group_size: int = 128,
    marlin_layout: bool | None = None,
) -> np.ndarray:
    """
    Unpack and dequantize FP4 weights back to float16.

    Supports two layouts:
    - Legacy layout: packed [out_feat, in_feat // 8], scales [out_feat, n_groups]
      Packs along input dimension (axis 1). Returns [out_feat, in_feat].
    - Marlin layout: packed [K/8, N], scales [K/group_size, N]
      Packs along K dimension (axis 0). Returns [K, N] which needs transpose
      to get original [out_feat, in_feat].

    Args:
        packed: uint32 packed weights
        scales: float16 scales
        group_size: Elements per group
        marlin_layout: If None, auto-detect from scale shapes.
            If True, expect Marlin layout [K/8, N].
            If False, expect legacy layout [out, in/8].

    Returns:
        Dequantized float16 weights. Shape depends on layout:
        - Legacy: [out_feat, in_feat]
        - Marlin: [K, N] (transpose back to get original shape)
    """
    # Auto-detect layout if not specified
    if marlin_layout is None:
        # Marlin layout: packed [K/8, N], scales [K/gs, N]
        # K = packed.shape[0] * 8, N = packed.shape[1]
        # scales.shape should be [K/gs, N] = [packed.shape[0] * 8 / gs, packed.shape[1]]
        K_expected = packed.shape[0] * 8
        N_expected = packed.shape[1]
        n_groups_K = K_expected // group_size
        if scales.shape == (n_groups_K, N_expected):
            marlin_layout = True
        else:
            marlin_layout = False

    if marlin_layout:
        # Marlin layout: packed [K/8, N], scales [K/gs, N]
        # Packing is along K (axis 0)
        K = packed.shape[0] * 8
        N = packed.shape[1]

        # Unpack 8 FP4 values from each uint32 along K dimension
        fp4_indices = np.zeros((K, N), dtype=np.uint8)
        for i in range(8):
            # Each packed value at [k_group, n] contains 8 consecutive K values
            fp4_indices[i::8, :] = ((packed >> (i * 4)) & 0xF).astype(np.uint8)

        # Dequantize indices to base values
        values = dequantize_fp4(fp4_indices)  # [K, N]

        # Apply scales: scales [K/gs, N]
        n_groups = K // group_size
        values = values.reshape(n_groups, group_size, N)
        scales_expanded = scales[:, None, :].astype(np.float32)
        values = values * scales_expanded
        values = values.reshape(K, N)

        return values.astype(np.float16)
    else:
        # Legacy layout: packed [out_feat, in_feat // 8], scales [out_feat, n_groups]
        out_feat = packed.shape[0]
        in_feat = packed.shape[1] * 8

        # Unpack 8 FP4 values from each uint32 along in_feat dimension
        fp4_indices = np.zeros((out_feat, in_feat), dtype=np.uint8)
        for i in range(8):
            fp4_indices[:, i::8] = ((packed >> (i * 4)) & 0xF).astype(np.uint8)

        # Dequantize indices to base values
        values = dequantize_fp4(fp4_indices)  # [out_feat, in_feat]

        # Apply scales: scales [out_feat, n_groups] where n_groups = in_feat // group_size
        n_groups = in_feat // group_size
        values = values.reshape(out_feat, n_groups, group_size)
        scales_expanded = scales[:, :, None].astype(np.float32)
        values = values * scales_expanded
        values = values.reshape(out_feat, in_feat)

        return values.astype(np.float16)


def compute_quantization_error(
    original: np.ndarray,
    packed: np.ndarray,
    scales: np.ndarray,
    group_size: int = 128,
    marlin_layout: bool | None = None,
) -> dict[str, float]:
    """Compute quantization error metrics.

    Args:
        original: Original weight tensor [out_feat, in_feat].
        packed: Packed FP4 weights.
        scales: Per-group scales.
        group_size: Quantization group size.
        marlin_layout: If None, auto-detect. If True, packed is [K/8, N]
            where K=in_feat, N=out_feat. If False, packed is [out, in/8].

    Returns:
        Dict with mse, rmse, max_error, mean_relative_error.
    """
    reconstructed = unpack_fp4(packed, scales, group_size, marlin_layout=marlin_layout)

    # If Marlin layout, reconstructed is [K, N] = [in_feat, out_feat]
    # Need to transpose back to [out_feat, in_feat] for comparison
    if marlin_layout is None:
        # Auto-detect same way as unpack_fp4
        K_expected = packed.shape[0] * 8
        N_expected = packed.shape[1]
        n_groups_K = K_expected // group_size
        if scales.shape == (n_groups_K, N_expected):
            marlin_layout = True
        else:
            marlin_layout = False

    if marlin_layout:
        reconstructed = reconstructed.T  # [K, N] -> [N, K] = [out_feat, in_feat]

    orig_f32 = original.astype(np.float32)
    recon_f32 = reconstructed.astype(np.float32)

    diff = orig_f32 - recon_f32
    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    max_err = np.max(np.abs(diff))

    # Relative error (avoid div by zero)
    rel_err = np.abs(diff) / (np.abs(orig_f32) + 1e-7)
    mean_rel_err = np.mean(rel_err)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "max_error": float(max_err),
        "mean_relative_error": float(mean_rel_err),
    }


def load_and_quantize_safetensors(
    path: str | Path,
    group_size: int = 128,
    skip_patterns: list[str] | None = None,
) -> Iterator[tuple[str, np.ndarray, np.ndarray | None]]:
    """
    Stream weights from safetensors, quantizing 2D weight matrices to FP4.

    Non-weight tensors (embeddings, biases, norms) are yielded unchanged.

    Args:
        path: Path to .safetensors file
        group_size: Quantization group size
        skip_patterns: List of substrings - skip quantizing tensors containing these

    Yields:
        (name, tensor_or_packed, scales_or_None):
        - For quantized: (name, packed_uint32, scales_fp16)
        - For non-quantized: (name, original_tensor, None)
    """
    from safetensors import safe_open

    skip_patterns = skip_patterns or ["embed", "norm", "bias", "lm_head"]

    with safe_open(str(path), framework="numpy") as f:
        for name in f.keys():
            tensor = f.get_tensor(name)

            # Check if should skip quantization
            should_skip = any(pat in name.lower() for pat in skip_patterns)

            # Only quantize 2D weight matrices
            if not should_skip and "weight" in name.lower() and tensor.ndim == 2:
                # Ensure divisibility
                out_feat, in_feat = tensor.shape
                if in_feat % 8 == 0 and in_feat % group_size == 0:
                    packed, scales = quantize_fp4(tensor, group_size)
                    yield name, packed, scales
                else:
                    # Can't quantize, yield original
                    yield name, tensor, None
            else:
                yield name, tensor, None


def convert_safetensors_to_fp4(
    input_path: str | Path,
    output_path: str | Path,
    group_size: int = 128,
    validate: bool = True,
) -> dict[str, any]:
    """
    Convert a safetensors model to FP4-quantized format.

    Args:
        input_path: Source .safetensors file
        output_path: Output .safetensors file
        group_size: Quantization group size
        validate: Compute and report quantization errors

    Returns:
        Stats dict with tensor counts, sizes, errors
    """
    from safetensors.numpy import save_file

    output_tensors = {}
    stats = {
        "quantized_count": 0,
        "skipped_count": 0,
        "original_bytes": 0,
        "quantized_bytes": 0,
        "errors": [],
    }

    for name, tensor, scales in load_and_quantize_safetensors(input_path, group_size):
        if scales is not None:
            # Quantized tensor
            output_tensors[name] = tensor  # packed uint32
            output_tensors[f"{name}.scales"] = scales
            stats["quantized_count"] += 1
            stats["original_bytes"] += tensor.shape[0] * tensor.shape[1] * 8 * 2  # original fp16
            stats["quantized_bytes"] += tensor.nbytes + scales.nbytes

            if validate:
                # Load original for error computation
                from safetensors import safe_open

                with safe_open(str(input_path), framework="numpy") as f:
                    original = f.get_tensor(name)
                err = compute_quantization_error(original, tensor, scales, group_size)
                stats["errors"].append({"name": name, **err})
        else:
            # Non-quantized tensor (kept as-is)
            output_tensors[name] = tensor
            stats["skipped_count"] += 1

    # Save output
    save_file(output_tensors, str(output_path))

    # Compute summary stats
    if stats["errors"]:
        stats["mean_rmse"] = np.mean([e["rmse"] for e in stats["errors"]])
        stats["max_error"] = max(e["max_error"] for e in stats["errors"])

    compression = stats["original_bytes"] / max(stats["quantized_bytes"], 1)
    stats["compression_ratio"] = compression

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize safetensors model to FP4 E2M1")
    parser.add_argument("input", help="Input .safetensors file")
    parser.add_argument("output", help="Output .safetensors file")
    parser.add_argument("--group-size", type=int, default=128, help="Quantization group size")
    parser.add_argument("--no-validate", action="store_true", help="Skip error validation")
    args = parser.parse_args()

    print(f"Quantizing {args.input} → {args.output}")
    print(f"Group size: {args.group_size}")

    stats = convert_safetensors_to_fp4(
        args.input,
        args.output,
        group_size=args.group_size,
        validate=not args.no_validate,
    )

    print("\nResults:")
    print(f"  Quantized tensors: {stats['quantized_count']}")
    print(f"  Skipped tensors:   {stats['skipped_count']}")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    if "mean_rmse" in stats:
        print(f"  Mean RMSE:         {stats['mean_rmse']:.6f}")
        print(f"  Max error:         {stats['max_error']:.6f}")
