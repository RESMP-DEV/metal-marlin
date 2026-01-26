"""
Sub-4-bit quantization: INT2, INT3, NF2, NF3 for aggressive MoE expert compression.

For MoE models like GLM-4.7-Flash:
- 64 experts, only 2 active per token
- 62 "cold" experts can be quantized aggressively
- Community benchmarks (llama.cpp IQ2_XXS, IQ3_XXS) show 2-3 bit works well

INT2: 4 levels packed 16 weights per uint32
  - Symmetric: [-1.5, -0.5, 0.5, 1.5] scaled
  - Best for cold MoE experts

INT3: 8 levels packed 10 weights per uint32 (with 2 bits padding)
  - Symmetric: [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5] scaled
  - Better quality than INT2, still 25% smaller than INT4

NF2/NF3: NormalFloat variants
  - Non-uniform quantization levels based on normal distribution
  - Better for transformer weights which are roughly Gaussian
  - Based on bitsandbytes NF4 approach

Reference implementations:
- llama.cpp IQ2_XXS, IQ3_XXS quants (proven quality)
- bitsandbytes NF4 approach (normal distribution quantiles)
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# ============================================================================
# INT2 Quantization (4 levels: -1.5, -0.5, 0.5, 1.5 scaled)
# ============================================================================

# INT2 symmetric levels (centered at 0)
# With scale s, values are: -1.5s, -0.5s, +0.5s, +1.5s
# Codes: 0 -> -1.5, 1 -> -0.5, 2 -> +0.5, 3 -> +1.5
INT2_LEVELS = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)


def quantize_int2(
    weight: np.ndarray,
    group_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize tensor to INT2 (4 levels) with per-group scales.

    Packing: 16 INT2 values per uint32 (2 bits each)
      bits [1:0] = val0, [3:2] = val1, ..., [31:30] = val15

    Args:
        weight: FP16/FP32 tensor [out_features, in_features]
        group_size: Elements per quantization group (default 64)

    Returns:
        (packed_uint32, scales) where:
        - packed: [out_features, in_features // 16] as uint32
        - scales: [out_features, in_features // group_size] as float16
    """
    w = weight.astype(np.float32)
    out_feat, in_feat = w.shape

    if in_feat % 16 != 0:
        raise ValueError(f"in_features ({in_feat}) must be divisible by 16 for INT2")
    if in_feat % group_size != 0:
        raise ValueError(f"in_features ({in_feat}) must be divisible by group_size ({group_size})")

    n_groups = in_feat // group_size

    # Reshape for per-group processing
    w_grouped = w.reshape(out_feat, n_groups, group_size)

    # Compute per-group scales: max abs value / 1.5 (INT2 max magnitude)
    group_max = np.max(np.abs(w_grouped), axis=2, keepdims=True)
    group_max = np.maximum(group_max, 1e-7)
    scales = (group_max / 1.5).astype(np.float16).squeeze(-1)

    # Scale weights and quantize to nearest INT2 level
    w_scaled = w_grouped / (scales[:, :, None].astype(np.float32))
    w_scaled = w_scaled.reshape(out_feat, in_feat)

    # Find nearest INT2 code for each value
    dists = np.abs(w_scaled[:, :, None] - INT2_LEVELS[None, None, :])
    codes = np.argmin(dists, axis=2).astype(np.uint8)  # [out_feat, in_feat]

    # Pack 16 INT2 values per uint32
    n_packed = in_feat // 16
    packed = np.zeros((out_feat, n_packed), dtype=np.uint32)
    for i in range(16):
        packed |= codes[:, i::16].astype(np.uint32) << (i * 2)

    return packed, scales


def dequantize_int2(
    packed: np.ndarray,
    scales: np.ndarray,
    group_size: int = 64,
) -> np.ndarray:
    """Dequantize INT2 packed weights back to float16."""
    out_feat = packed.shape[0]
    in_feat = packed.shape[1] * 16

    # Unpack 16 INT2 values from each uint32
    codes = np.zeros((out_feat, in_feat), dtype=np.uint8)
    for i in range(16):
        codes[:, i::16] = ((packed >> (i * 2)) & 0x3).astype(np.uint8)

    # Map codes to INT2 levels
    values = INT2_LEVELS[codes.astype(np.int32)]

    # Apply per-group scales
    n_groups = in_feat // group_size
    values = values.reshape(out_feat, n_groups, group_size)
    values = values * scales[:, :, None].astype(np.float32)
    values = values.reshape(out_feat, in_feat)

    return values.astype(np.float16)


# ============================================================================
# INT3 Quantization (8 levels: -3.5 to +3.5 scaled)
# ============================================================================

# INT3 symmetric levels (centered at 0)
# Codes: 0 -> -3.5, 1 -> -2.5, ..., 7 -> +3.5
INT3_LEVELS = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], dtype=np.float32)


def quantize_int3(
    weight: np.ndarray,
    group_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize tensor to INT3 (8 levels) with per-group scales.

    Packing: 10 INT3 values per uint32 (3 bits each, 2 bits padding)
      bits [2:0] = val0, [5:3] = val1, ..., [29:27] = val9, [31:30] = padding

    Args:
        weight: FP16/FP32 tensor [out_features, in_features]
        group_size: Elements per quantization group (default 64)

    Returns:
        (packed_uint32, scales) where:
        - packed: [out_features, ceil(in_features / 10)] as uint32
        - scales: [out_features, in_features // group_size] as float16
    """
    w = weight.astype(np.float32)
    out_feat, in_feat = w.shape

    # INT3 packs 10 values per uint32, so we need divisibility by 10
    # Pad if necessary
    pad_needed = (10 - (in_feat % 10)) % 10
    if pad_needed > 0:
        w = np.pad(w, ((0, 0), (0, pad_needed)), mode='constant', constant_values=0)
        in_feat_padded = in_feat + pad_needed
    else:
        in_feat_padded = in_feat

    if in_feat % group_size != 0:
        raise ValueError(f"in_features ({in_feat}) must be divisible by group_size ({group_size})")

    n_groups = in_feat // group_size

    # Reshape for per-group processing (use original in_feat for groups)
    w_for_scale = w[:, :in_feat].reshape(out_feat, n_groups, group_size)

    # Compute per-group scales: max abs value / 3.5 (INT3 max magnitude)
    group_max = np.max(np.abs(w_for_scale), axis=2, keepdims=True)
    group_max = np.maximum(group_max, 1e-7)
    scales = (group_max / 3.5).astype(np.float16).squeeze(-1)

    # Scale weights using appropriate scale for each position
    w_scaled = np.zeros_like(w)
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        w_scaled[:, start:end] = w[:, start:end] / scales[:, g:g+1].astype(np.float32)
    # Handle padding region (use last group's scale)
    if pad_needed > 0:
        w_scaled[:, in_feat:] = w[:, in_feat:] / scales[:, -1:].astype(np.float32)

    # Find nearest INT3 code for each value
    dists = np.abs(w_scaled[:, :, None] - INT3_LEVELS[None, None, :])
    codes = np.argmin(dists, axis=2).astype(np.uint8)

    # Pack 10 INT3 values per uint32
    n_packed = in_feat_padded // 10
    packed = np.zeros((out_feat, n_packed), dtype=np.uint32)
    for i in range(10):
        packed |= codes[:, i::10].astype(np.uint32) << (i * 3)

    return packed, scales


def dequantize_int3(
    packed: np.ndarray,
    scales: np.ndarray,
    group_size: int = 64,
    original_in_feat: int | None = None,
) -> np.ndarray:
    """Dequantize INT3 packed weights back to float16."""
    out_feat = packed.shape[0]
    in_feat_padded = packed.shape[1] * 10

    # Unpack 10 INT3 values from each uint32
    codes = np.zeros((out_feat, in_feat_padded), dtype=np.uint8)
    for i in range(10):
        codes[:, i::10] = ((packed >> (i * 3)) & 0x7).astype(np.uint8)

    # Map codes to INT3 levels
    values = INT3_LEVELS[codes.astype(np.int32)]

    # Apply per-group scales
    n_groups = scales.shape[1]
    in_feat = n_groups * group_size  # Original unpadded size from scales

    # Only scale the valid region
    values_out = np.zeros((out_feat, in_feat), dtype=np.float32)
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        values_out[:, start:end] = values[:, start:end] * scales[:, g:g+1].astype(np.float32)

    if original_in_feat is not None:
        values_out = values_out[:, :original_in_feat]

    return values_out.astype(np.float16)


# ============================================================================
# NF2 Quantization (NormalFloat 2-bit: Gaussian quantiles)
# ============================================================================

def _compute_nf_levels(n_bits: int) -> np.ndarray:
    """
    Compute NormalFloat quantization levels based on Gaussian distribution quantiles.

    The levels are chosen to minimize expected quantization error when the
    input distribution is Gaussian (which transformer weights approximately are).

    For n_bits=2 (4 levels), we use quantiles at 12.5%, 37.5%, 62.5%, 87.5%
    which gives symmetric levels around 0.

    For n_bits=3 (8 levels), we use quantiles at:
    6.25%, 18.75%, 31.25%, 43.75%, 56.25%, 68.75%, 81.25%, 93.75%

    These minimize MSE for Gaussian-distributed inputs, per the NF4 paper.
    """
    n_levels = 2 ** n_bits
    # Compute quantiles at midpoints of n_levels equal-probability bins
    quantiles = np.linspace(0.5 / n_levels, 1 - 0.5 / n_levels, n_levels)
    levels = stats.norm.ppf(quantiles)

    # Normalize so max magnitude is approximately 1 (for scaling)
    levels = levels / np.max(np.abs(levels))
    return levels.astype(np.float32)


# Precomputed NF2 levels (4 values, symmetric around 0)
NF2_LEVELS = _compute_nf_levels(2)

# Precomputed NF3 levels (8 values, symmetric around 0)
NF3_LEVELS = _compute_nf_levels(3)


def quantize_nf2(
    weight: np.ndarray,
    group_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize tensor to NF2 (NormalFloat 2-bit) with per-group scales.

    NF2 uses non-uniform quantization levels based on Gaussian quantiles,
    which is optimal for transformer weights that follow approximately
    normal distributions.

    Packing: Same as INT2 (16 values per uint32)

    Args:
        weight: FP16/FP32 tensor [out_features, in_features]
        group_size: Elements per quantization group (default 64)

    Returns:
        (packed_uint32, scales) where:
        - packed: [out_features, in_features // 16] as uint32
        - scales: [out_features, in_features // group_size] as float16
    """
    w = weight.astype(np.float32)
    out_feat, in_feat = w.shape

    if in_feat % 16 != 0:
        raise ValueError(f"in_features ({in_feat}) must be divisible by 16 for NF2")
    if in_feat % group_size != 0:
        raise ValueError(f"in_features ({in_feat}) must be divisible by group_size ({group_size})")

    n_groups = in_feat // group_size

    # Reshape for per-group processing
    w_grouped = w.reshape(out_feat, n_groups, group_size)

    # Compute per-group scales: max abs value / max NF2 level magnitude
    nf2_max = np.max(np.abs(NF2_LEVELS))
    group_max = np.max(np.abs(w_grouped), axis=2, keepdims=True)
    group_max = np.maximum(group_max, 1e-7)
    scales = (group_max / nf2_max).astype(np.float16).squeeze(-1)

    # Scale weights and quantize to nearest NF2 level
    w_scaled = w_grouped / (scales[:, :, None].astype(np.float32))
    w_scaled = w_scaled.reshape(out_feat, in_feat)

    # Find nearest NF2 code for each value
    dists = np.abs(w_scaled[:, :, None] - NF2_LEVELS[None, None, :])
    codes = np.argmin(dists, axis=2).astype(np.uint8)

    # Pack 16 NF2 values per uint32 (same as INT2)
    n_packed = in_feat // 16
    packed = np.zeros((out_feat, n_packed), dtype=np.uint32)
    for i in range(16):
        packed |= codes[:, i::16].astype(np.uint32) << (i * 2)

    return packed, scales


def dequantize_nf2(
    packed: np.ndarray,
    scales: np.ndarray,
    group_size: int = 64,
) -> np.ndarray:
    """Dequantize NF2 packed weights back to float16."""
    out_feat = packed.shape[0]
    in_feat = packed.shape[1] * 16

    # Unpack 16 NF2 values from each uint32
    codes = np.zeros((out_feat, in_feat), dtype=np.uint8)
    for i in range(16):
        codes[:, i::16] = ((packed >> (i * 2)) & 0x3).astype(np.uint8)

    # Map codes to NF2 levels
    values = NF2_LEVELS[codes.astype(np.int32)]

    # Apply per-group scales
    n_groups = in_feat // group_size
    values = values.reshape(out_feat, n_groups, group_size)
    values = values * scales[:, :, None].astype(np.float32)
    values = values.reshape(out_feat, in_feat)

    return values.astype(np.float16)


# ============================================================================
# NF3 Quantization (NormalFloat 3-bit: Gaussian quantiles)
# ============================================================================

def quantize_nf3(
    weight: np.ndarray,
    group_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize tensor to NF3 (NormalFloat 3-bit) with per-group scales.

    NF3 uses 8 non-uniform quantization levels based on Gaussian quantiles.
    Packing: Same as INT3 (10 values per uint32, 2 bits padding)

    Args:
        weight: FP16/FP32 tensor [out_features, in_features]
        group_size: Elements per quantization group (default 64)

    Returns:
        (packed_uint32, scales) where:
        - packed: [out_features, ceil(in_features / 10)] as uint32
        - scales: [out_features, in_features // group_size] as float16
    """
    w = weight.astype(np.float32)
    out_feat, in_feat = w.shape

    # INT3 packs 10 values per uint32
    pad_needed = (10 - (in_feat % 10)) % 10
    if pad_needed > 0:
        w = np.pad(w, ((0, 0), (0, pad_needed)), mode='constant', constant_values=0)
        in_feat_padded = in_feat + pad_needed
    else:
        in_feat_padded = in_feat

    if in_feat % group_size != 0:
        raise ValueError(f"in_features ({in_feat}) must be divisible by group_size ({group_size})")

    n_groups = in_feat // group_size

    # Reshape for per-group processing
    w_for_scale = w[:, :in_feat].reshape(out_feat, n_groups, group_size)

    # Compute per-group scales
    nf3_max = np.max(np.abs(NF3_LEVELS))
    group_max = np.max(np.abs(w_for_scale), axis=2, keepdims=True)
    group_max = np.maximum(group_max, 1e-7)
    scales = (group_max / nf3_max).astype(np.float16).squeeze(-1)

    # Scale weights
    w_scaled = np.zeros_like(w)
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        w_scaled[:, start:end] = w[:, start:end] / scales[:, g:g+1].astype(np.float32)
    if pad_needed > 0:
        w_scaled[:, in_feat:] = w[:, in_feat:] / scales[:, -1:].astype(np.float32)

    # Find nearest NF3 code for each value
    dists = np.abs(w_scaled[:, :, None] - NF3_LEVELS[None, None, :])
    codes = np.argmin(dists, axis=2).astype(np.uint8)

    # Pack 10 NF3 values per uint32
    n_packed = in_feat_padded // 10
    packed = np.zeros((out_feat, n_packed), dtype=np.uint32)
    for i in range(10):
        packed |= codes[:, i::10].astype(np.uint32) << (i * 3)

    return packed, scales


def dequantize_nf3(
    packed: np.ndarray,
    scales: np.ndarray,
    group_size: int = 64,
    original_in_feat: int | None = None,
) -> np.ndarray:
    """Dequantize NF3 packed weights back to float16."""
    out_feat = packed.shape[0]
    in_feat_padded = packed.shape[1] * 10

    # Unpack 10 NF3 values from each uint32
    codes = np.zeros((out_feat, in_feat_padded), dtype=np.uint8)
    for i in range(10):
        codes[:, i::10] = ((packed >> (i * 3)) & 0x7).astype(np.uint8)

    # Map codes to NF3 levels
    values = NF3_LEVELS[codes.astype(np.int32)]

    # Apply per-group scales
    n_groups = scales.shape[1]
    in_feat = n_groups * group_size

    values_out = np.zeros((out_feat, in_feat), dtype=np.float32)
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        values_out[:, start:end] = values[:, start:end] * scales[:, g:g+1].astype(np.float32)

    if original_in_feat is not None:
        values_out = values_out[:, :original_in_feat]

    return values_out.astype(np.float16)


# ============================================================================
# Utility functions
# ============================================================================

def compute_quantization_error(
    original: np.ndarray,
    packed: np.ndarray,
    scales: np.ndarray,
    quant_type: str,
    group_size: int = 64,
) -> dict[str, float]:
    """
    Compute quantization error metrics for sub-4-bit formats.

    Args:
        original: Original FP16/FP32 tensor
        packed: Packed quantized weights
        scales: Per-group scales
        quant_type: One of 'int2', 'int3', 'nf2', 'nf3'
        group_size: Quantization group size

    Returns:
        Dict with mse, rmse, max_error, mean_relative_error
    """
    dequant_funcs = {
        'int2': dequantize_int2,
        'int3': dequantize_int3,
        'nf2': dequantize_nf2,
        'nf3': dequantize_nf3,
    }

    dequant_func = dequant_funcs.get(quant_type.lower())
    if dequant_func is None:
        raise ValueError(f"Unknown quant_type: {quant_type}")

    reconstructed = dequant_func(packed, scales, group_size)
    orig_f32 = original.astype(np.float32)
    recon_f32 = reconstructed.astype(np.float32)

    # Truncate if INT3/NF3 padding was added
    if recon_f32.shape[1] < orig_f32.shape[1]:
        orig_f32 = orig_f32[:, :recon_f32.shape[1]]

    diff = orig_f32 - recon_f32
    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    max_err = np.max(np.abs(diff))

    rel_err = np.abs(diff) / (np.abs(orig_f32) + 1e-7)
    mean_rel_err = np.mean(rel_err)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "max_error": float(max_err),
        "mean_relative_error": float(mean_rel_err),
    }


def estimate_compression_ratio(
    shape: tuple[int, int],
    quant_type: str,
    group_size: int = 64,
) -> float:
    """
    Estimate compression ratio for a given shape and quantization type.

    Args:
        shape: (out_features, in_features)
        quant_type: One of 'int2', 'int3', 'nf2', 'nf3'
        group_size: Quantization group size

    Returns:
        Compression ratio vs FP16 (e.g., 8.0 means 8x smaller)
    """
    out_feat, in_feat = shape

    # Original size in bits (FP16 = 16 bits)
    original_bits = out_feat * in_feat * 16

    # Quantized weight bits
    if quant_type.lower() in ('int2', 'nf2'):
        weight_bits = out_feat * in_feat * 2
    elif quant_type.lower() in ('int3', 'nf3'):
        weight_bits = out_feat * in_feat * 3
    else:
        raise ValueError(f"Unknown quant_type: {quant_type}")

    # Scale bits (FP16 per group)
    n_groups = in_feat // group_size
    scale_bits = out_feat * n_groups * 16

    total_bits = weight_bits + scale_bits
    return original_bits / total_bits


def select_sub4bit_format(
    tensor: np.ndarray,
    target_compression: float = 6.0,
    quality_priority: bool = False,
) -> str:
    """
    Select the best sub-4-bit format for a tensor based on constraints.

    Args:
        tensor: Weight tensor to analyze
        target_compression: Minimum compression ratio vs FP16
        quality_priority: If True, prefer NF variants over INT

    Returns:
        One of 'int2', 'int3', 'nf2', 'nf3'
    """
    # Analyze tensor distribution
    vals = tensor.flatten().astype(np.float32)
    skewness = stats.skew(vals)
    kurtosis = stats.kurtosis(vals)

    # Normal-like distribution benefits from NF quantization
    is_gaussian_like = abs(skewness) < 0.5 and abs(kurtosis) < 1.0

    if target_compression >= 7.0:
        # Need aggressive compression -> 2-bit
        return 'nf2' if (is_gaussian_like or quality_priority) else 'int2'
    else:
        # Can use 3-bit for better quality
        return 'nf3' if (is_gaussian_like or quality_priority) else 'int3'


# ============================================================================
# Export constants for Metal shaders
# ============================================================================

def get_int2_lut() -> np.ndarray:
    """Get INT2 dequantization lookup table for Metal shaders."""
    return INT2_LEVELS.copy()


def get_int3_lut() -> np.ndarray:
    """Get INT3 dequantization lookup table for Metal shaders."""
    return INT3_LEVELS.copy()


def get_nf2_lut() -> np.ndarray:
    """Get NF2 dequantization lookup table for Metal shaders."""
    return NF2_LEVELS.copy()


def get_nf3_lut() -> np.ndarray:
    """Get NF3 dequantization lookup table for Metal shaders."""
    return NF3_LEVELS.copy()


# ============================================================================
# CLI / Test
# ============================================================================

def main():
    """Test sub-4-bit quantization on random tensor."""
    import argparse

    parser = argparse.ArgumentParser(description="Test sub-4-bit quantization")
    parser.add_argument("--shape", type=str, default="4096,4096", help="Tensor shape (out,in)")
    parser.add_argument("--group-size", type=int, default=64, help="Quantization group size")
    parser.add_argument("--format", choices=['int2', 'int3', 'nf2', 'nf3', 'all'], default='all')
    args = parser.parse_args()

    shape = tuple(int(x) for x in args.shape.split(','))
    out_feat, in_feat = shape

    # Ensure divisibility
    in_feat = (in_feat // 160) * 160  # LCM of 16 and 10

    print(f"Testing sub-4-bit quantization on tensor [{out_feat}, {in_feat}]")
    print(f"Group size: {args.group_size}")
    print()

    # Generate random Gaussian tensor (transformer-like)
    np.random.seed(42)
    tensor = np.random.randn(out_feat, in_feat).astype(np.float32) * 0.02

    formats = ['int2', 'int3', 'nf2', 'nf3'] if args.format == 'all' else [args.format]

    for fmt in formats:
        print(f"=== {fmt.upper()} ===")

        if fmt == 'int2':
            packed, scales = quantize_int2(tensor, args.group_size)
            reconstructed = dequantize_int2(packed, scales, args.group_size)
        elif fmt == 'int3':
            packed, scales = quantize_int3(tensor, args.group_size)
            reconstructed = dequantize_int3(packed, scales, args.group_size)
        elif fmt == 'nf2':
            packed, scales = quantize_nf2(tensor, args.group_size)
            reconstructed = dequantize_nf2(packed, scales, args.group_size)
        else:  # nf3
            packed, scales = quantize_nf3(tensor, args.group_size)
            reconstructed = dequantize_nf3(packed, scales, args.group_size)

        # Compute error (handle potential padding)
        min_feat = min(tensor.shape[1], reconstructed.shape[1])
        diff = tensor[:, :min_feat] - reconstructed[:, :min_feat]
        rmse = np.sqrt(np.mean(diff**2))
        max_err = np.max(np.abs(diff))
        compression = estimate_compression_ratio(shape, fmt, args.group_size)

        print(f"  Packed shape: {packed.shape}")
        print(f"  Scales shape: {scales.shape}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  Max error: {max_err:.6f}")
        print(f"  Compression: {compression:.2f}x vs FP16")
        print()


if __name__ == "__main__":
    main()
