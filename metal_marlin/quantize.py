"""FP4 E2M1 quantization with padding support for non-aligned dimensions.

Handles the case where K (input features) is not evenly divisible by
group_size or by 8 (the packing factor for uint32). Pads with zeros and
tracks original dimensions for correct dequantization.

This module uses the same E2M1 codebook and packing layout as metal_marlin.py
but adds dimension-padding logic for arbitrary weight shapes.

This module uses pure numpy for quantization operations. MLX is optional
and only needed if you want MLX arrays as output (use output_backend='mlx').
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from ._compat import from_numpy, to_numpy

# E2M1 codebook: the 16 representable FP4 values.
# Nibble encoding: [sign(1) | exp(2) | mant(1)], bias = 1.
# Positive: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
# Negative: -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
E2M1_VALUES: np.ndarray = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

# Maximum representable magnitude in E2M1
E2M1_MAX = 6.0

# Packing factor: 8 FP4 nibbles per uint32
FP4_PER_U32 = 8


def _pad_to_multiple(arr: np.ndarray, axis: int, multiple: int) -> np.ndarray:
    """Pad array along given axis to the next multiple of `multiple`."""
    current = arr.shape[axis]
    if current % multiple == 0:
        return arr
    pad_amount = multiple - (current % multiple)
    pad_widths = [(0, 0)] * arr.ndim
    pad_widths[axis] = (0, pad_amount)
    return np.pad(arr, pad_widths, mode='constant', constant_values=0)


def _quantize_to_e2m1(
    normalized: np.ndarray,
) -> np.ndarray:
    """Map normalized float values to nearest E2M1 nibble indices.

    Args:
        normalized: Float array with values pre-divided by group scale.

    Returns:
        uint8 array of nibble indices (0-15) into E2M1_VALUES.
    """
    # Clip to representable range
    clipped = np.clip(normalized, -E2M1_MAX, E2M1_MAX)

    # Nearest-neighbor against the 16-element codebook.
    # For large arrays, process in chunks to avoid memory blowup.
    flat = clipped.ravel()
    n = len(flat)
    chunk_size = 1 << 20  # ~1M elements per chunk
    indices = np.empty(n, dtype=np.uint8)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = flat[start:end]
        # [chunk, 1] - [1, 16] -> [chunk, 16]
        dists = np.abs(chunk[:, None] - E2M1_VALUES[None, :])
        indices[start:end] = dists.argmin(axis=1).astype(np.uint8)

    return indices.reshape(normalized.shape)


def _pack_nibbles_to_uint32(indices: np.ndarray) -> np.ndarray:
    """Pack FP4 nibble indices along the N dimension into uint32 words.

    8 consecutive N-dimension values are packed into one uint32:
      word = nib[0] | (nib[1] << 4) | ... | (nib[7] << 28)

    Args:
        indices: [K, N] uint8 array with values 0-15. N must be divisible by 8.

    Returns:
        [K, N // 8] uint32 array.
    """
    K, N = indices.shape
    assert N % FP4_PER_U32 == 0, f"N={N} must be divisible by {FP4_PER_U32}"

    reshaped = indices.reshape(K, N // FP4_PER_U32, FP4_PER_U32).astype(np.uint32)
    packed = np.zeros((K, N // FP4_PER_U32), dtype=np.uint32)
    for i in range(FP4_PER_U32):
        packed |= reshaped[:, :, i] << (i * 4)
    return packed


def pack_fp4_weights(
    weights: Any,
    group_size: int = 128,
    pad_k: bool = True,
    scales_dtype: np.dtype | None = None,
    output_backend: Literal["numpy", "mlx", "torch"] = "numpy",
) -> tuple[Any, Any, dict[str, Any]]:
    """Pack float weights to Marlin FP4 format with padding for non-aligned K.

    Quantizes weights using the E2M1 codebook with per-group scales along K.
    When K is not divisible by group_size (or by 8 for packing), the weight
    matrix is zero-padded along K. The original dimensions are returned in
    metadata for correct inference.

    Packing layout: weights[K, N] -> packed[K_padded, N_padded // 8] as uint32,
    where 8 consecutive N-dimension values are packed per uint32.

    Args:
        weights: Weight matrix [K, N] (K = input features, N = output features).
            Can be numpy array, MLX array, or PyTorch tensor.
        group_size: Elements per quantization group along K. Default: 128.
        pad_k: If True, pad K to the next multiple of group_size when not
               evenly divisible. If False, raise ValueError on misalignment.
        scales_dtype: Dtype for scales array. Default: np.float16.
        output_backend: Backend for output arrays ('numpy', 'mlx', or 'torch').
            Default: 'numpy'.

    Returns:
        Tuple of:
            packed: uint32 array [K_padded, N_padded // 8].
            scales: float array [K_padded // group_size, N_padded].
            meta: Dict with 'orig_K', 'orig_N', 'padded_K', 'padded_N'
                  for reconstructing correct output dimensions.

    Raises:
        ValueError: If pad_k=False and K is not divisible by group_size.
        ValueError: If output_backend is not available.
    """
    # Default scales dtype
    if scales_dtype is None:
        scales_dtype = np.float16

    # Convert input to numpy (handles MLX eval, torch CPU move, etc.)
    w = to_numpy(weights).astype(np.float32)
    orig_K, orig_N = w.shape

    # --- Pad K to multiple of group_size ---
    if orig_K % group_size != 0:
        if not pad_k:
            raise ValueError(
                f"K={orig_K} not divisible by group_size={group_size}. "
                f"Set pad_k=True to pad automatically."
            )
        w = _pad_to_multiple(w, axis=0, multiple=group_size)

    K = w.shape[0]

    # group_size might not be a multiple of 8 (e.g., group_size=6).
    # Ensure K is also divisible by 8 for nibble packing along K if we
    # ever pack along K. For the N-dimension packing used here, we need
    # N divisible by 8 instead.
    # However, the GEMM kernel processes K in chunks of group_size per
    # scale row, so K must be group_size-aligned (already done above).
    # The uint32 packing is along N, so pad N to multiple of 8.
    N = w.shape[1]
    if orig_N % FP4_PER_U32 != 0:
        w = _pad_to_multiple(w, axis=1, multiple=FP4_PER_U32)
        N = w.shape[1]

    # --- Compute per-group scales ---
    num_groups = K // group_size
    w_grouped = w.reshape(num_groups, group_size, N)
    group_max = np.abs(w_grouped).max(axis=1)  # [num_groups, N]

    # Scale = max_abs / max_codebook_value. Avoid division by zero.
    scales_np = np.where(group_max > 0, group_max / E2M1_MAX, np.float32(1e-7))

    # --- Quantize to E2M1 indices ---
    scales_expanded = np.repeat(scales_np, group_size, axis=0)  # [K, N]
    normalized = w / scales_expanded
    indices = _quantize_to_e2m1(normalized)  # [K, N] uint8

    # --- Pack into uint32 ---
    packed_np = _pack_nibbles_to_uint32(indices)  # [K, N // 8]

    meta = {
        "orig_K": orig_K,
        "orig_N": orig_N,
        "padded_K": K,
        "padded_N": N,
        "group_size": group_size,
    }

    # Convert to output backend
    packed_out = from_numpy(packed_np, backend=output_backend)
    scales_out = from_numpy(scales_np.astype(scales_dtype), backend=output_backend)

    return packed_out, scales_out, meta


def unpack_fp4_weights(
    packed: Any,
    scales: Any,
    meta: dict[str, Any],
    weights_dtype: np.dtype | None = None,
    output_backend: Literal["numpy", "mlx", "torch"] = "numpy",
) -> Any:
    """Unpack Marlin FP4 weights back to float for validation.

    Reverses pack_fp4_weights: extracts nibble indices from uint32 words,
    maps through E2M1 codebook, applies group scales, and trims padding.

    Args:
        packed: uint32 array [K_padded, N_padded // 8].
            Can be numpy array, MLX array, or PyTorch tensor.
        scales: float array [K_padded // group_size, N_padded].
            Can be numpy array, MLX array, or PyTorch tensor.
        meta: Metadata dict from pack_fp4_weights.
        weights_dtype: Dtype for output weights. Default: np.float16.
        output_backend: Backend for output array ('numpy', 'mlx', or 'torch').
            Default: 'numpy'.

    Returns:
        Array [orig_K, orig_N] (padding stripped) in specified dtype and backend.
    """
    # Default weights dtype
    if weights_dtype is None:
        weights_dtype = np.float16

    # Convert inputs to numpy (handles MLX eval, torch CPU move, etc.)
    packed_np = to_numpy(packed)
    scales_np = to_numpy(scales).astype(np.float32)

    K = meta["padded_K"]
    N = meta["padded_N"]
    group_size = meta["group_size"]

    # Unpack uint32 -> nibble indices
    packed_n = N // FP4_PER_U32

    # packed layout: packed[k, g] has nibbles for cols [g*8, g*8+8)
    indices_correct = np.empty((K, N), dtype=np.uint8)
    for g in range(packed_n):
        col_start = g * FP4_PER_U32
        for i in range(FP4_PER_U32):
            indices_correct[:, col_start + i] = (
                (packed_np[:, g] >> (i * 4)) & 0xF
            ).astype(np.uint8)

    # Dequantize via codebook
    values = E2M1_VALUES[indices_correct].astype(np.float32)

    # Apply per-group scales
    scales_expanded = np.repeat(scales_np, group_size, axis=0)  # [K, N]
    values *= scales_expanded

    # Trim padding
    orig_K = meta["orig_K"]
    orig_N = meta["orig_N"]
    values_trimmed = values[:orig_K, :orig_N].astype(weights_dtype)

    # Convert to output backend
    return from_numpy(values_trimmed, backend=output_backend)


# =============================================================================
# INT4 Symmetric Quantization
# =============================================================================

# INT4 symmetric representable values: -8 to +7
INT4_SYM_MIN = -8
INT4_SYM_MAX = 7


def _quantize_to_int4_sym(normalized: np.ndarray) -> np.ndarray:
    """Map normalized float values to INT4 symmetric indices (0-15).

    INT4 symmetric uses range [-8, 7] with zero at index 8.

    Args:
        normalized: Float array with values pre-divided by group scale.

    Returns:
        uint8 array of indices (0-15).
    """
    # Clip and round to nearest integer
    clipped = np.clip(normalized, INT4_SYM_MIN, INT4_SYM_MAX)
    quantized = np.round(clipped).astype(np.int8)

    # Shift to unsigned: -8 -> 0, 7 -> 15
    indices = (quantized - INT4_SYM_MIN).astype(np.uint8)
    return indices


def _dequant_int4_sym(indices: np.ndarray) -> np.ndarray:
    """Dequantize INT4 symmetric indices back to float values."""
    return (indices.astype(np.int8) + INT4_SYM_MIN).astype(np.float32)


def pack_int4_weights(
    weights: Any,
    group_size: int = 128,
    pad_k: bool = True,
    scales_dtype: np.dtype | None = None,
    output_backend: Literal["numpy", "mlx", "torch"] = "numpy",
) -> tuple[Any, Any, dict[str, Any]]:
    """Pack float weights to INT4 symmetric format with per-group scales.

    Similar to pack_fp4_weights but uses uniform INT4 quantization grid
    [-8, -7, ..., 0, ..., 7] instead of FP4 E2M1 non-uniform grid.

    Args:
        weights: Weight matrix [K, N] (K = input features, N = output features).
        group_size: Elements per quantization group along K. Default: 128.
        pad_k: If True, pad K to the next multiple of group_size.
        scales_dtype: Dtype for scales array. Default: np.float16.
        output_backend: Backend for output arrays.

    Returns:
        Tuple of (packed, scales, meta).
    """
    if scales_dtype is None:
        scales_dtype = np.float16

    w = to_numpy(weights).astype(np.float32)
    orig_K, orig_N = w.shape

    # Pad K to multiple of group_size
    if orig_K % group_size != 0:
        if not pad_k:
            raise ValueError(
                f"K={orig_K} not divisible by group_size={group_size}. "
                f"Set pad_k=True to pad automatically."
            )
        w = _pad_to_multiple(w, axis=0, multiple=group_size)

    K = w.shape[0]

    # Pad N to multiple of 8 for packing
    N = w.shape[1]
    if orig_N % FP4_PER_U32 != 0:
        w = _pad_to_multiple(w, axis=1, multiple=FP4_PER_U32)
        N = w.shape[1]

    # Compute per-group scales (symmetric: scale = max_abs / 7)
    num_groups = K // group_size
    w_grouped = w.reshape(num_groups, group_size, N)
    group_max = np.abs(w_grouped).max(axis=1)  # [num_groups, N]

    # INT4 symmetric: max representable is 7 (we use symmetric range)
    scales_np = np.where(group_max > 0, group_max / INT4_SYM_MAX, np.float32(1e-7))

    # Quantize to INT4 indices
    scales_expanded = np.repeat(scales_np, group_size, axis=0)
    normalized = w / scales_expanded
    indices = _quantize_to_int4_sym(normalized)  # [K, N] uint8

    # Pack into uint32 (same layout as FP4)
    packed_np = _pack_nibbles_to_uint32(indices)

    meta = {
        "orig_K": orig_K,
        "orig_N": orig_N,
        "padded_K": K,
        "padded_N": N,
        "group_size": group_size,
        "quant_type": "int4_sym",
    }

    packed_out = from_numpy(packed_np, backend=output_backend)
    scales_out = from_numpy(scales_np.astype(scales_dtype), backend=output_backend)

    return packed_out, scales_out, meta


# =============================================================================
# NF4 (NormalFloat4) Quantization - QLoRA style
# =============================================================================

# NF4 codebook: optimal quantization levels for normally distributed data
# From QLoRA paper, these values minimize quantization error for N(0,1) weights
NF4_VALUES: np.ndarray = np.array(
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

NF4_MAX = 1.0


def _quantize_to_nf4(normalized: np.ndarray) -> np.ndarray:
    """Map normalized float values to NF4 indices (0-15).

    NF4 uses a non-uniform grid optimized for normally distributed weights.

    Args:
        normalized: Float array with values pre-divided by group scale.

    Returns:
        uint8 array of indices (0-15) into NF4_VALUES.
    """
    # Clip to representable range
    clipped = np.clip(normalized, -NF4_MAX, NF4_MAX)

    # Nearest-neighbor against NF4 codebook (same approach as FP4)
    flat = clipped.ravel()
    n = len(flat)
    chunk_size = 1 << 20
    indices = np.empty(n, dtype=np.uint8)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = flat[start:end]
        dists = np.abs(chunk[:, None] - NF4_VALUES[None, :])
        indices[start:end] = dists.argmin(axis=1).astype(np.uint8)

    return indices.reshape(normalized.shape)


def pack_nf4_weights(
    weights: Any,
    group_size: int = 128,
    pad_k: bool = True,
    scales_dtype: np.dtype | None = None,
    output_backend: Literal["numpy", "mlx", "torch"] = "numpy",
) -> tuple[Any, Any, dict[str, Any]]:
    """Pack float weights to NF4 (NormalFloat4) format with per-group scales.

    NF4 uses a non-uniform quantization grid optimized for normally distributed
    weights, as introduced in QLoRA. This provides better quality than uniform
    INT4 for typical neural network weights.

    Args:
        weights: Weight matrix [K, N] (K = input features, N = output features).
        group_size: Elements per quantization group along K. Default: 128.
        pad_k: If True, pad K to the next multiple of group_size.
        scales_dtype: Dtype for scales array. Default: np.float16.
        output_backend: Backend for output arrays.

    Returns:
        Tuple of (packed, scales, meta).
    """
    if scales_dtype is None:
        scales_dtype = np.float16

    w = to_numpy(weights).astype(np.float32)
    orig_K, orig_N = w.shape

    # Pad K to multiple of group_size
    if orig_K % group_size != 0:
        if not pad_k:
            raise ValueError(
                f"K={orig_K} not divisible by group_size={group_size}. "
                f"Set pad_k=True to pad automatically."
            )
        w = _pad_to_multiple(w, axis=0, multiple=group_size)

    K = w.shape[0]

    # Pad N to multiple of 8 for packing
    N = w.shape[1]
    if orig_N % FP4_PER_U32 != 0:
        w = _pad_to_multiple(w, axis=1, multiple=FP4_PER_U32)
        N = w.shape[1]

    # Compute per-group scales
    # NF4 normalizes to [-1, 1] so scale = max_abs
    num_groups = K // group_size
    w_grouped = w.reshape(num_groups, group_size, N)
    group_max = np.abs(w_grouped).max(axis=1)

    scales_np = np.where(group_max > 0, group_max / NF4_MAX, np.float32(1e-7))

    # Quantize to NF4 indices
    scales_expanded = np.repeat(scales_np, group_size, axis=0)
    normalized = w / scales_expanded
    indices = _quantize_to_nf4(normalized)

    # Pack into uint32
    packed_np = _pack_nibbles_to_uint32(indices)

    meta = {
        "orig_K": orig_K,
        "orig_N": orig_N,
        "padded_K": K,
        "padded_N": N,
        "group_size": group_size,
        "quant_type": "nf4",
    }

    packed_out = from_numpy(packed_np, backend=output_backend)
    scales_out = from_numpy(scales_np.astype(scales_dtype), backend=output_backend)

    return packed_out, scales_out, meta


# =============================================================================
# High-level model quantization functions
# =============================================================================

def quantize_to_int4(
    model_path: str | Path,
    output_path: str | Path,
    group_size: int = 128,
    symmetric: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Quantize model weights to INT4 format.

    Args:
        model_path: Path to source model (safetensors or HF directory)
        output_path: Path to save quantized model
        group_size: Quantization group size
        symmetric: Use symmetric quantization (recommended)
        verbose: Print progress

    Returns:
        Stats dict with compression ratio, error metrics, etc.
    """
    from pathlib import Path as PathlibPath

    from safetensors import safe_open
    from safetensors.numpy import save_file

    model_path = PathlibPath(model_path)
    output_path = PathlibPath(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find safetensors files
    if model_path.is_file():
        safetensors_files = [model_path]
    else:
        safetensors_files = sorted(model_path.glob("*.safetensors"))

    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    stats = {
        "quantized_count": 0,
        "skipped_count": 0,
        "original_bytes": 0,
        "quantized_bytes": 0,
        "quant_type": "int4_sym" if symmetric else "int4_asym",
    }

    skip_patterns = ["embed", "norm", "bias", "lm_head"]

    for sf_path in safetensors_files:
        if verbose:
            print(f"  Processing: {sf_path.name}")

        output_tensors = {}

        with safe_open(str(sf_path), framework="numpy") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)

                should_skip = any(pat in name.lower() for pat in skip_patterns)
                is_weight = "weight" in name.lower() and tensor.ndim == 2

                if not should_skip and is_weight:
                    K, N = tensor.shape
                    if N % 8 == 0 and K % group_size == 0:
                        packed, scales, meta = pack_int4_weights(
                            tensor, group_size=group_size
                        )
                        output_tensors[name] = packed
                        output_tensors[f"{name}.scales"] = scales
                        stats["quantized_count"] += 1
                        stats["original_bytes"] += tensor.nbytes
                        stats["quantized_bytes"] += packed.nbytes + scales.nbytes
                    else:
                        output_tensors[name] = tensor
                        stats["skipped_count"] += 1
                else:
                    output_tensors[name] = tensor
                    stats["skipped_count"] += 1

        out_file = output_path / sf_path.name.replace(".safetensors", ".int4.safetensors")
        save_file(output_tensors, str(out_file))

    if stats["quantized_bytes"] > 0:
        stats["compression_ratio"] = stats["original_bytes"] / stats["quantized_bytes"]
    else:
        stats["compression_ratio"] = 1.0

    if verbose:
        print(f"  Quantized: {stats['quantized_count']} tensors")
        print(f"  Skipped: {stats['skipped_count']} tensors")
        print(f"  Compression: {stats['compression_ratio']:.2f}x")

    return stats


def quantize_to_nf4(
    model_path: str | Path,
    output_path: str | Path,
    group_size: int = 128,
    verbose: bool = True,
) -> dict[str, Any]:
    """Quantize model weights to NF4 (NormalFloat4) format.

    Args:
        model_path: Path to source model (safetensors or HF directory)
        output_path: Path to save quantized model
        group_size: Quantization group size
        verbose: Print progress

    Returns:
        Stats dict with compression ratio, error metrics, etc.
    """
    from pathlib import Path as PathlibPath

    from safetensors import safe_open
    from safetensors.numpy import save_file

    model_path = PathlibPath(model_path)
    output_path = PathlibPath(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find safetensors files
    if model_path.is_file():
        safetensors_files = [model_path]
    else:
        safetensors_files = sorted(model_path.glob("*.safetensors"))

    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    stats = {
        "quantized_count": 0,
        "skipped_count": 0,
        "original_bytes": 0,
        "quantized_bytes": 0,
        "quant_type": "nf4",
    }

    skip_patterns = ["embed", "norm", "bias", "lm_head"]

    for sf_path in safetensors_files:
        if verbose:
            print(f"  Processing: {sf_path.name}")

        output_tensors = {}

        with safe_open(str(sf_path), framework="numpy") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)

                should_skip = any(pat in name.lower() for pat in skip_patterns)
                is_weight = "weight" in name.lower() and tensor.ndim == 2

                if not should_skip and is_weight:
                    K, N = tensor.shape
                    if N % 8 == 0 and K % group_size == 0:
                        packed, scales, meta = pack_nf4_weights(
                            tensor, group_size=group_size
                        )
                        output_tensors[name] = packed
                        output_tensors[f"{name}.scales"] = scales
                        stats["quantized_count"] += 1
                        stats["original_bytes"] += tensor.nbytes
                        stats["quantized_bytes"] += packed.nbytes + scales.nbytes
                    else:
                        output_tensors[name] = tensor
                        stats["skipped_count"] += 1
                else:
                    output_tensors[name] = tensor
                    stats["skipped_count"] += 1

        out_file = output_path / sf_path.name.replace(".safetensors", ".nf4.safetensors")
        save_file(output_tensors, str(out_file))

    if stats["quantized_bytes"] > 0:
        stats["compression_ratio"] = stats["original_bytes"] / stats["quantized_bytes"]
    else:
        stats["compression_ratio"] = 1.0

    if verbose:
        print(f"  Quantized: {stats['quantized_count']} tensors")
        print(f"  Skipped: {stats['skipped_count']} tensors")
        print(f"  Compression: {stats['compression_ratio']:.2f}x")

    return stats
