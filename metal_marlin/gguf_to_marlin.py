"""GGUF MXFP4 and GGML block-quant to Marlin format converter.

Reads a GGUF file containing MXFP4-quantized tensors and converts them
to Marlin-packed format suitable for the Metal Marlin fused dequant-GEMM
kernel. Non-quantized tensors (embeddings, output layers, norms) are
preserved in their original precision.

MXFP4 format (per block of 32 elements):
  - 1 byte E8M0 shared exponent (scale = 2^(e - 127))
  - 16 bytes of packed 4-bit E2M1 values (32 nibbles)

Marlin format (per group of group_size elements):
  - Packed uint32 words in simdgroup-friendly order
  - Per-group FP16 scales

This converter supports:
  - MXFP4 (GGML_TYPE_MXFP4)
  - GGML block quants via gguf_loader (Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ3_XXS, IQ4_XS)

Note on IQ2/IQ3:
  These formats use importance-guided codebooks during quantization.
  Conversion here dequantizes the GGUF weights and re-quantizes to Marlin FP4,
  which preserves inference compatibility but not the original importance
  matrix used during quantization.

Reference:
  - OCP Microscaling Formats (MX) v1.0 spec
  - ggml-common.h block_mxfp4 layout
  - Marlin: https://arxiv.org/abs/2312.07723
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .gguf_loader import DEQUANT_SUPPORTED_TYPES, dequantize_tensor

# --- MXFP4 Constants ---

# E2M1 value table: index 0-15 maps to float
# Bits: [sign(1), exponent(2), mantissa(1)]
# From ggml-common.h kvalues_mxfp4 (as floats, not doubled int8)
KVALUES_MXFP4: np.ndarray = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

# GGML type ID for MXFP4 (from ggml.h)
GGML_TYPE_MXFP4 = 39

# MXFP4 block parameters
MXFP4_BLOCK_SIZE = 32  # elements per block
MXFP4_BLOCK_BYTES = 17  # 1 (E8M0 scale) + 16 (packed nibbles)

# Default Marlin group size
DEFAULT_GROUP_SIZE = 128

# Marlin tile dimensions (Metal simdgroup 8x8)
MARLIN_TILE_N = 8
MARLIN_TILE_K = 8


def e8m0_to_fp32(e: np.ndarray) -> np.ndarray:
    """Convert E8M0 exponent bytes to FP32 scale values.

    E8M0 is a pure exponent format: value = 2^(e - 127).
    Special case: e=0 maps to 2^(-126) (smallest normal).
    """
    result = np.where(
        e == 0,
        np.float32(2.0 ** -126),
        np.power(np.float32(2.0), e.astype(np.float32) - 127.0),
    )
    return result.astype(np.float32)


def dequant_mxfp4_block(raw: np.ndarray) -> np.ndarray:
    """Dequantize a single MXFP4 block (17 bytes) to 32 FP32 values.

    Layout: [e8m0_byte, qs[0], qs[1], ..., qs[15]]
    Each qs byte contains two 4-bit E2M1 values: low nibble first, high nibble second.
    """
    e_byte = raw[0]
    scale = e8m0_to_fp32(np.array([e_byte], dtype=np.uint8))[0]
    qs = raw[1:17]  # 16 bytes = 32 nibbles

    # Extract low and high nibbles
    lo_nibbles = qs & 0x0F  # elements 0, 2, 4, ... (even indices within pairs)
    hi_nibbles = (qs >> 4) & 0x0F  # elements 1, 3, 5, ... (odd indices within pairs)

    # Interleave: the GGUF layout stores pairs as [lo, hi] in each byte
    # Based on the dequantize_mxfp4 function in ggml-metal.metal:
    #   il=0 (shr=0): uses low nibbles for first 16 elements
    #   il=1 (shr=4): uses high nibbles for next 16 elements
    # So the 32 elements are: first 16 from low nibbles, next 16 from high nibbles
    values = np.empty(32, dtype=np.float32)
    values[0:16] = KVALUES_MXFP4[lo_nibbles]
    values[16:32] = KVALUES_MXFP4[hi_nibbles]

    return values * scale


def dequant_mxfp4(data: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Dequantize an entire MXFP4 tensor from raw bytes to FP16.

    Args:
        data: Raw byte array from GGUF tensor.
        shape: Target tensor shape (rows, cols).

    Returns:
        FP16 array of the specified shape.
    """
    n_elements = int(np.prod(shape))
    n_blocks = n_elements // MXFP4_BLOCK_SIZE
    assert n_blocks * MXFP4_BLOCK_SIZE == n_elements, (
        f"Element count {n_elements} not divisible by block size {MXFP4_BLOCK_SIZE}"
    )

    raw = np.frombuffer(data, dtype=np.uint8)
    expected_bytes = n_blocks * MXFP4_BLOCK_BYTES
    assert len(raw) == expected_bytes, (
        f"Expected {expected_bytes} bytes for {n_blocks} blocks, got {len(raw)}"
    )

    # Vectorized dequantization across all blocks
    blocks = raw.reshape(n_blocks, MXFP4_BLOCK_BYTES)
    e_bytes = blocks[:, 0]  # shape: (n_blocks,)
    qs_bytes = blocks[:, 1:17]  # shape: (n_blocks, 16)

    scales = e8m0_to_fp32(e_bytes)  # shape: (n_blocks,)

    lo_nibbles = qs_bytes & 0x0F  # shape: (n_blocks, 16)
    hi_nibbles = (qs_bytes >> 4) & 0x0F  # shape: (n_blocks, 16)

    # Dequantize using the LUT
    lo_values = KVALUES_MXFP4[lo_nibbles]  # shape: (n_blocks, 16)
    hi_values = KVALUES_MXFP4[hi_nibbles]  # shape: (n_blocks, 16)

    # Concatenate: first 16 from lo, next 16 from hi (per block)
    all_values = np.concatenate([lo_values, hi_values], axis=1)  # (n_blocks, 32)

    # Apply per-block scale
    all_values *= scales[:, np.newaxis]

    # Reshape to target and convert to FP16
    return all_values.reshape(shape).astype(np.float16)


def quantize_fp4(
    weights: np.ndarray,
    scales: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """Quantize FP16/FP32 weights to 4-bit E2M1 (MXFP4 codebook).

    Uses nearest-value quantization against the E2M1 codebook.

    Args:
        weights: [K, N] weight matrix.
        scales: [K // group_size, N] per-group scales.
        group_size: Number of elements per quantization group.

    Returns:
        [K, N] array of uint8 indices (0-15) into KVALUES_MXFP4.
    """
    K, N = weights.shape
    n_groups = K // group_size

    # Normalize weights by their group scale
    weights_grouped = weights.reshape(n_groups, group_size, N)
    scales_expanded = scales[:, np.newaxis, :]  # (n_groups, 1, N)

    # Avoid division by zero
    safe_scales = np.where(scales_expanded == 0, 1.0, scales_expanded)
    normalized = weights_grouped / safe_scales

    # Find nearest E2M1 value for each normalized weight
    # KVALUES_MXFP4 absolute values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    # The codebook maps: indices 0-7 are non-negative, 8-15 are negative
    flat_norm = normalized.reshape(-1)
    indices = np.empty(len(flat_norm), dtype=np.uint8)

    # Vectorized nearest-neighbor search against the 16-element codebook
    diffs = np.abs(flat_norm[:, np.newaxis] - KVALUES_MXFP4[np.newaxis, :])
    indices = diffs.argmin(axis=1).astype(np.uint8)

    return indices.reshape(K, N)


def compute_scales_marlin(
    weights: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """Compute per-group Marlin scales from FP16 weights.

    For MXFP4-compatible quantization, the scale is the maximum absolute
    value in each group divided by 6.0 (the max representable E2M1 value).

    Args:
        weights: [K, N] weight matrix in FP16/FP32.
        group_size: Number of elements per group.

    Returns:
        [K // group_size, N] FP16 scales.
    """
    K, N = weights.shape
    n_groups = K // group_size
    weights_grouped = weights.reshape(n_groups, group_size, N).astype(np.float32)

    # Scale is max_abs / max_codebook_value
    # max E2M1 magnitude is 6.0
    max_abs = np.abs(weights_grouped).max(axis=1)  # (n_groups, N)
    scales = max_abs / 6.0

    return scales.astype(np.float16)


def pack_nibbles(indices: np.ndarray) -> np.ndarray:
    """Pack pairs of 4-bit indices into bytes.

    Two consecutive indices along the K dimension are packed into one byte:
    byte = (indices[k+1] << 4) | indices[k]

    Args:
        indices: [K, N] uint8 array with values 0-15.

    Returns:
        [K // 2, N] uint8 array of packed nibble pairs.
    """
    K, N = indices.shape
    assert K % 2 == 0, f"K={K} must be even for nibble packing"
    lo = indices[0::2, :]  # even rows
    hi = indices[1::2, :]  # odd rows
    return (hi << 4) | lo


def reorder_for_simdgroup(packed: np.ndarray, N: int) -> np.ndarray:
    """Reorder packed weights for Metal simdgroup 8x8 tile access.

    Marlin uses tensor-core-fragment-order permutation so that each
    thread's memory loads are contiguous. For Metal's 8x8 simdgroup
    matrix, we permute along the N dimension in tiles of 8 columns.

    The permutation ensures that within each 8-column tile, the data
    layout matches the simdgroup lane assignment for simdgroup_load.

    For the 8x8 simdgroup matrix on Apple Silicon:
      - 32 threads in a simdgroup
      - Each thread owns 2 elements of the 8x8 tile
      - Lane i owns elements at positions determined by the hardware layout

    We apply an interleave pattern within each group of 8 columns that
    matches the expected simdgroup_load memory layout.

    Args:
        packed: [K_packed, N] packed nibble array (uint8 or uint32).
        N: Output dimension.

    Returns:
        Reordered array with same shape.
    """
    packed.shape[0]

    # Number of 8-column tiles
    n_tiles = N // MARLIN_TILE_N
    remainder = N % MARLIN_TILE_N

    if n_tiles == 0:
        return packed.copy()

    result = np.empty_like(packed)

    for tile_idx in range(n_tiles):
        col_start = tile_idx * MARLIN_TILE_N
        col_end = col_start + MARLIN_TILE_N
        tile = packed[:, col_start:col_end]

        # Interleave pattern for 8x8 simdgroup:
        # Rows are grouped in pairs, columns are accessed in a stride-4 pattern
        # This matches how simdgroup_load expects data in threadgroup memory
        #
        # For each group of 8 rows in the K dimension:
        #   Row 0,1 -> positions 0,1 (lanes 0-7)
        #   Row 2,3 -> positions 2,3 (lanes 8-15)
        #   Row 4,5 -> positions 4,5 (lanes 16-23)
        #   Row 6,7 -> positions 6,7 (lanes 24-31)
        #
        # Within columns, stride-2 interleave for half2 vectorization:
        #   Cols [0,2,4,6,1,3,5,7] -> [0,1,2,3,4,5,6,7]
        col_perm = np.array([0, 2, 4, 6, 1, 3, 5, 7])
        result[:, col_start:col_end] = tile[:, col_perm]

    # Handle remainder columns (no permutation needed)
    if remainder > 0:
        result[:, n_tiles * MARLIN_TILE_N:] = packed[:, n_tiles * MARLIN_TILE_N:]

    return result


def pack_weights_marlin(
    weights_fp16: np.ndarray,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert FP16 weights to Marlin-packed FP4 format.

    Pipeline:
      1. Compute per-group scales (max_abs / 6.0)
      2. Quantize to 4-bit E2M1 indices
      3. Pack pairs of indices into bytes
      4. Reorder for simdgroup tile access
      5. Pack bytes into uint32 words (8 nibbles per word)

    Args:
        weights_fp16: [K, N] FP16 weight matrix.
        group_size: Elements per quantization group (default 128).

    Returns:
        packed_weights: [K // 8, N] uint32 array (8 FP4 values per word).
        scales: [K // group_size, N] FP16 per-group scales.
    """
    weights = weights_fp16.astype(np.float32)
    K, N = weights.shape

    assert K % group_size == 0, (
        f"K={K} must be divisible by group_size={group_size}"
    )
    assert K % 8 == 0, f"K={K} must be divisible by 8 for uint32 packing"

    # Step 1: Compute scales
    scales = compute_scales_marlin(weights, group_size)

    # Step 2: Quantize
    indices = quantize_fp4(weights, scales.astype(np.float32), group_size)

    # Step 3: Pack into nibble pairs
    packed_bytes = pack_nibbles(indices)  # [K//2, N]

    # Step 4: Reorder for simdgroup access
    packed_reordered = reorder_for_simdgroup(packed_bytes, N)

    # Step 5: Pack 4 bytes into uint32 (8 nibbles = 8 FP4 values per word)
    K_half = K // 2
    assert K_half % 4 == 0, f"K//2={K_half} must be divisible by 4"
    packed_u32 = packed_reordered.reshape(K_half // 4, 4, N)
    # Combine 4 bytes into uint32: byte0 | (byte1 << 8) | (byte2 << 16) | (byte3 << 24)
    packed_words = (
        packed_u32[:, 0, :].astype(np.uint32)
        | (packed_u32[:, 1, :].astype(np.uint32) << 8)
        | (packed_u32[:, 2, :].astype(np.uint32) << 16)
        | (packed_u32[:, 3, :].astype(np.uint32) << 24)
    )

    return packed_words, scales


def dequant_marlin(
    packed_weights: np.ndarray,
    scales: np.ndarray,
    K: int,
    N: int,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> np.ndarray:
    """Dequantize Marlin-packed weights back to FP16 for validation.

    Reverses the packing pipeline to recover approximate FP16 values.

    Args:
        packed_weights: [K // 8, N] uint32 packed weights.
        scales: [K // group_size, N] FP16 scales.
        K: Original K dimension.
        N: Original N dimension.
        group_size: Elements per group.

    Returns:
        [K, N] FP16 array.
    """
    # Unpack uint32 -> 4 bytes
    K // 8
    byte0 = (packed_weights & 0xFF).astype(np.uint8)
    byte1 = ((packed_weights >> 8) & 0xFF).astype(np.uint8)
    byte2 = ((packed_weights >> 16) & 0xFF).astype(np.uint8)
    byte3 = ((packed_weights >> 24) & 0xFF).astype(np.uint8)

    packed_bytes = np.stack([byte0, byte1, byte2, byte3], axis=1)
    packed_bytes = packed_bytes.reshape(K // 2, N)

    # Undo simdgroup reorder
    n_tiles = N // MARLIN_TILE_N
    remainder = N % MARLIN_TILE_N
    inv_perm = np.array([0, 4, 1, 5, 2, 6, 3, 7])  # inverse of [0,2,4,6,1,3,5,7]

    unpermuted = np.empty_like(packed_bytes)
    for tile_idx in range(n_tiles):
        col_start = tile_idx * MARLIN_TILE_N
        col_end = col_start + MARLIN_TILE_N
        unpermuted[:, col_start:col_end] = packed_bytes[:, col_start:col_end][:, inv_perm]
    if remainder > 0:
        unpermuted[:, n_tiles * MARLIN_TILE_N:] = packed_bytes[:, n_tiles * MARLIN_TILE_N:]

    # Unpack nibbles
    lo_indices = unpermuted & 0x0F
    hi_indices = (unpermuted >> 4) & 0x0F

    # Reconstruct full index array [K, N]
    indices = np.empty((K, N), dtype=np.uint8)
    indices[0::2, :] = lo_indices
    indices[1::2, :] = hi_indices

    # Dequantize using codebook and scales
    values = KVALUES_MXFP4[indices].astype(np.float32)

    n_groups = K // group_size
    scales_f32 = scales.astype(np.float32)
    values_grouped = values.reshape(n_groups, group_size, N)
    values_grouped *= scales_f32[:, np.newaxis, :]

    return values_grouped.reshape(K, N).astype(np.float16)


def is_quantizable_tensor(name: str) -> bool:
    """Determine if a tensor should be quantized (vs kept in original precision).

    Embedding layers, output/lm_head layers, and normalization layers are
    typically kept in full precision for accuracy.
    """
    # Keep these in original precision
    skip_patterns = [
        "embed",
        "token_embd",
        "output.weight",
        "lm_head",
        ".norm",
        "norm.",
        "ln_",
        "layernorm",
        "rmsnorm",
    ]
    name_lower = name.lower()
    return not any(pat in name_lower for pat in skip_patterns)


def is_importance_matrix_tensor(name: str) -> bool:
    """Best-effort detection for imatrix/importance tensors."""
    name_lower = name.lower()
    return "imatrix" in name_lower or "importance" in name_lower


def ggml_type_name(type_val: int) -> str:
    """Human-readable GGML type name for stats/metadata."""
    names = {
        0: "F32",
        1: "F16",
        2: "Q4_0",
        3: "Q4_1",
        6: "Q5_0",
        7: "Q5_1",
        8: "Q8_0",
        9: "Q8_1",
        10: "Q2_K",
        11: "Q3_K",
        12: "Q4_K",
        13: "Q5_K",
        14: "Q6_K",
        15: "Q8_K",
        16: "IQ2_XXS",
        18: "IQ3_XXS",
        20: "IQ4_NL",
        23: "IQ4_XS",
        30: "BF16",
        39: "MXFP4",
    }
    return names.get(type_val, f"UNKNOWN({type_val})")


def extract_model_config(reader: Any) -> dict[str, Any]:
    """Extract model configuration from GGUF metadata fields.

    Reads architecture, hidden size, layers, heads, etc. from the GGUF
    key-value metadata store.
    """
    config: dict[str, Any] = {}

    # Map of GGUF metadata keys to config keys
    key_map = {
        "general.architecture": "architecture",
        "general.name": "name",
        "general.file_type": "file_type",
    }

    # Architecture-specific keys (will be prefixed with arch name)
    arch_keys = [
        "context_length",
        "embedding_length",
        "block_count",
        "attention.head_count",
        "attention.head_count_kv",
        "feed_forward_length",
        "expert_count",
        "expert_used_count",
        "vocab_size",
    ]

    for field in reader.fields.values() if hasattr(reader, 'fields') else []:
        name = field.name if hasattr(field, 'name') else str(field)
        if name in key_map:
            # Extract scalar value
            if hasattr(field, 'parts') and len(field.parts) > 0:
                val = field.parts[-1].tolist()
                if isinstance(val, list) and len(val) == 1:
                    val = val[0]
                config[key_map[name]] = val

    # Try to get architecture and then architecture-specific keys
    arch = config.get("architecture", "")
    if arch:
        for key in arch_keys:
            full_key = f"{arch}.{key}"
            try:
                field = reader.get_field(full_key)
                if field is not None and hasattr(field, 'parts') and len(field.parts) > 0:
                    val = field.parts[-1].tolist()
                    if isinstance(val, list) and len(val) == 1:
                        val = val[0]
                    config[key.replace(".", "_")] = val
            except (KeyError, AttributeError):
                pass

    config["quantization"] = "marlin_fp4"
    config["group_size"] = DEFAULT_GROUP_SIZE
    config["source_quantization"] = "mxfp4"

    return config


def convert_gguf_to_marlin(
    gguf_path: str,
    output_path: str,
    group_size: int = DEFAULT_GROUP_SIZE,
    validate: bool = True,
) -> dict[str, Any]:
    """Read a GGUF file with MXFP4 quantization and convert to Marlin format.

    The output is a directory containing:
      - config.json: Model configuration extracted from GGUF metadata
      - weights.safetensors: Marlin-packed weights and scales

    Args:
        gguf_path: Path to input GGUF file.
        output_path: Directory path for output files.
        group_size: Marlin quantization group size (default 128).
        validate: If True, validate round-trip accuracy per tensor.

    Returns:
        Dictionary with conversion statistics.
    """
    try:
        from gguf import GGUFReader
    except ImportError as e:
        raise ImportError(
            "gguf package required. Install with: uv add gguf"
        ) from e

    try:
        from safetensors.numpy import save_file
    except ImportError as e:
        raise ImportError(
            "safetensors package required. Install with: uv add safetensors"
        ) from e

    reader = GGUFReader(gguf_path)

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    marlin_weights: dict[str, np.ndarray] = {}
    stats = {
        "total_tensors": 0,
        "quantized_tensors": 0,
        "preserved_tensors": 0,
        "total_params": 0,
        "quantized_params": 0,
        "max_roundtrip_error": 0.0,
        "mean_roundtrip_error": 0.0,
        "errors": [],
        "source_quant_types": [],
    }

    roundtrip_errors: list[float] = []

    for tensor in reader.tensors:
        stats["total_tensors"] += 1
        name = tensor.name
        tensor_type = tensor.tensor_type
        shape = tuple(tensor.shape)
        n_elements = tensor.n_elements

        stats["total_params"] += n_elements

        # tensor_type is an enum or int; compare against the known value
        type_val = tensor_type.value if hasattr(tensor_type, 'value') else int(tensor_type)

        # Preserve importance matrices as-is (if present)
        if is_importance_matrix_tensor(name):
            stats["preserved_tensors"] += 1
            type_to_dtype = {
                0: np.float32,   # F32
                1: np.float16,   # F16
                30: np.float16,  # BF16 -> store as FP16
            }
            dtype = type_to_dtype.get(type_val, None)
            if dtype is not None:
                marlin_weights[name] = np.frombuffer(
                    tensor.data, dtype=dtype
                ).reshape(shape).copy()
            else:
                marlin_weights[name] = np.frombuffer(
                    tensor.data, dtype=np.uint8
                ).copy()
                marlin_weights[name + ".ggml_type"] = np.array(
                    [type_val], dtype=np.int32
                )
                marlin_weights[name + ".shape"] = np.array(
                    list(shape), dtype=np.int32
                )
            print(f"  Preserved importance tensor {name}: {shape} (type={type_val})")
            continue

        # Check if this is an MXFP4 tensor
        if type_val == GGML_TYPE_MXFP4 and is_quantizable_tensor(name):
            # MXFP4 tensor -> dequant then repack as Marlin
            stats["quantized_tensors"] += 1
            stats["quantized_params"] += n_elements
            if "MXFP4" not in stats["source_quant_types"]:
                stats["source_quant_types"].append("MXFP4")

            # The shape from GGUF is the logical tensor shape
            # For 2D weight matrices: (out_features, in_features) typically
            if len(shape) == 2:
                rows, cols = shape
            elif len(shape) == 1:
                # 1D tensors (biases etc) - keep as-is
                fp16_data = dequant_mxfp4(tensor.data, shape)
                marlin_weights[name] = fp16_data
                stats["preserved_tensors"] += 1
                stats["quantized_tensors"] -= 1
                stats["quantized_params"] -= n_elements
                continue
            else:
                # Higher-dimensional tensors: reshape to 2D for packing
                # Flatten all but last dim into rows
                rows = int(np.prod(shape[:-1]))
                cols = shape[-1]

            # Dequantize MXFP4 to FP16
            fp16_weights = dequant_mxfp4(tensor.data, (rows, cols))

            # Pad K dimension to be divisible by group_size if needed
            if rows % group_size != 0:
                pad_rows = group_size - (rows % group_size)
                fp16_weights = np.pad(
                    fp16_weights,
                    ((0, pad_rows), (0, 0)),
                    mode='constant',
                    constant_values=0,
                )
                padded_rows = rows + pad_rows
            else:
                padded_rows = rows
                pad_rows = 0

            # Pad N dimension to be divisible by MARLIN_TILE_N if needed
            if cols % MARLIN_TILE_N != 0:
                pad_cols = MARLIN_TILE_N - (cols % MARLIN_TILE_N)
                fp16_weights = np.pad(
                    fp16_weights,
                    ((0, 0), (0, pad_cols)),
                    mode='constant',
                    constant_values=0,
                )
                padded_cols = cols + pad_cols
            else:
                padded_cols = cols
                pad_cols = 0

            # Pack as Marlin
            packed, scales = pack_weights_marlin(fp16_weights, group_size)

            marlin_weights[name] = packed
            marlin_weights[name + ".scales"] = scales

            # Store original shape for unpacking
            marlin_weights[name + ".shape"] = np.array(
                [rows, cols, padded_rows, padded_cols], dtype=np.int32
            )

            # Validate round-trip accuracy
            if validate:
                recovered = dequant_marlin(
                    packed, scales, padded_rows, padded_cols, group_size
                )
                # Compare only the unpadded region
                recovered_unpadded = recovered[:rows, :cols]
                original_unpadded = dequant_mxfp4(tensor.data, (rows, cols))

                # Compute relative error (avoid div by zero)
                abs_diff = np.abs(
                    recovered_unpadded.astype(np.float32)
                    - original_unpadded.astype(np.float32)
                )
                max_err = float(abs_diff.max())
                mean_err = float(abs_diff.mean())
                roundtrip_errors.append(max_err)

                if max_err > 1e-2:
                    stats["errors"].append({
                        "tensor": name,
                        "max_error": max_err,
                        "mean_error": mean_err,
                        "shape": list(shape),
                    })

            print(
                f"  Packed {name}: {shape} -> "
                f"weights[{packed.shape}] + scales[{scales.shape}]"
            )

        elif type_val in DEQUANT_SUPPORTED_TYPES and is_quantizable_tensor(name):
            # GGML block-quantized tensor -> dequant then repack as Marlin
            stats["quantized_tensors"] += 1
            stats["quantized_params"] += n_elements

            qtype_name = ggml_type_name(type_val)
            if type_val not in stats["source_quant_types"]:
                stats["source_quant_types"].append(qtype_name)

            if len(shape) == 1:
                fp32_data = dequantize_tensor(
                    np.frombuffer(tensor.data, dtype=np.uint8),
                    type_val,
                    n_elements,
                ).reshape(shape)
                marlin_weights[name] = fp32_data.astype(np.float16)
                stats["preserved_tensors"] += 1
                stats["quantized_tensors"] -= 1
                stats["quantized_params"] -= n_elements
                continue

            if len(shape) == 2:
                rows, cols = shape
            else:
                rows = int(np.prod(shape[:-1]))
                cols = shape[-1]

            fp32_weights = dequantize_tensor(
                np.frombuffer(tensor.data, dtype=np.uint8),
                type_val,
                n_elements,
            ).reshape((rows, cols))

            fp16_weights = fp32_weights.astype(np.float16)

            # Pad K dimension to be divisible by group_size if needed
            if rows % group_size != 0:
                pad_rows = group_size - (rows % group_size)
                fp16_weights = np.pad(
                    fp16_weights,
                    ((0, pad_rows), (0, 0)),
                    mode='constant',
                    constant_values=0,
                )
                padded_rows = rows + pad_rows
            else:
                padded_rows = rows
                pad_rows = 0

            # Pad N dimension to be divisible by MARLIN_TILE_N if needed
            if cols % MARLIN_TILE_N != 0:
                pad_cols = MARLIN_TILE_N - (cols % MARLIN_TILE_N)
                fp16_weights = np.pad(
                    fp16_weights,
                    ((0, 0), (0, pad_cols)),
                    mode='constant',
                    constant_values=0,
                )
                padded_cols = cols + pad_cols
            else:
                padded_cols = cols
                pad_cols = 0

            packed, scales = pack_weights_marlin(fp16_weights, group_size)

            marlin_weights[name] = packed
            marlin_weights[name + ".scales"] = scales
            marlin_weights[name + ".shape"] = np.array(
                [rows, cols, padded_rows, padded_cols], dtype=np.int32
            )

            print(
                f"  Packed {name}: {shape} -> "
                f"weights[{packed.shape}] + scales[{scales.shape}]"
            )

        else:
            # Non-MXFP4 or non-quantizable: preserve in original precision
            stats["preserved_tensors"] += 1

            if type_val == GGML_TYPE_MXFP4:
                # MXFP4 but we want to keep it (embeddings, norms)
                fp16_data = dequant_mxfp4(tensor.data, shape)
                marlin_weights[name] = fp16_data
                print(f"  Preserved (dequant) {name}: {shape} as FP16")
            elif type_val in DEQUANT_SUPPORTED_TYPES:
                fp32_data = dequantize_tensor(
                    np.frombuffer(tensor.data, dtype=np.uint8),
                    type_val,
                    n_elements,
                ).reshape(shape)
                marlin_weights[name] = fp32_data.astype(np.float16)
                print(f"  Preserved (dequant) {name}: {shape} as FP16")
            else:
                # Already in a standard format, copy raw data
                # Determine numpy dtype from GGML type
                type_to_dtype = {
                    0: np.float32,   # F32
                    1: np.float16,   # F16
                    30: np.float16,  # BF16 -> store as FP16
                }
                dtype = type_to_dtype.get(type_val, None)
                if dtype is not None:
                    data_array = np.frombuffer(
                        tensor.data, dtype=dtype
                    ).reshape(shape).copy()
                    marlin_weights[name] = data_array
                else:
                    # For other quantized types, store raw bytes with metadata
                    marlin_weights[name] = np.frombuffer(
                        tensor.data, dtype=np.uint8
                    ).copy()
                    marlin_weights[name + ".ggml_type"] = np.array(
                        [type_val], dtype=np.int32
                    )
                    marlin_weights[name + ".shape"] = np.array(
                        list(shape), dtype=np.int32
                    )
                print(f"  Preserved {name}: {shape} (type={type_val})")

    # Compute aggregate stats
    if roundtrip_errors:
        stats["max_roundtrip_error"] = float(max(roundtrip_errors))
        stats["mean_roundtrip_error"] = float(np.mean(roundtrip_errors))

    # Save weights
    print(f"\nSaving {len(marlin_weights)} tensors to safetensors...")
    save_file(marlin_weights, str(out_dir / "weights.safetensors"))

    # Save config
    config = extract_model_config(reader)
    if stats["source_quant_types"]:
        if len(stats["source_quant_types"]) == 1:
            config["source_quantization"] = stats["source_quant_types"][0]
        else:
            config["source_quantization"] = "mixed"
    config["conversion_stats"] = {
        k: v for k, v in stats.items() if k != "errors"
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save errors if any
    if stats["errors"]:
        with open(out_dir / "validation_errors.json", "w") as f:
            json.dump(stats["errors"], f, indent=2)
        print(f"\nWARNING: {len(stats['errors'])} tensors had roundtrip error > 1e-2")

    print("\nConversion complete:")
    print(f"  Total tensors: {stats['total_tensors']}")
    print(f"  Quantized (Marlin FP4): {stats['quantized_tensors']}")
    print(f"  Preserved (original): {stats['preserved_tensors']}")
    print(f"  Max roundtrip error: {stats['max_roundtrip_error']:.6f}")
    print(f"  Mean roundtrip error: {stats['mean_roundtrip_error']:.6f}")

    return stats


def main() -> None:
    """CLI entry point for GGUF to Marlin conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert GGUF MXFP4 weights to Marlin format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s model.gguf output_dir/
  %(prog)s model.gguf output_dir/ --group-size 64
  %(prog)s model.gguf output_dir/ --no-validate
""",
    )
    parser.add_argument("gguf_path", help="Path to input GGUF file")
    parser.add_argument("output_path", help="Output directory for Marlin weights")
    parser.add_argument(
        "--group-size",
        type=int,
        default=DEFAULT_GROUP_SIZE,
        help=f"Quantization group size (default: {DEFAULT_GROUP_SIZE})",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip round-trip accuracy validation",
    )

    args = parser.parse_args()

    gguf_file = Path(args.gguf_path)
    if not gguf_file.exists():
        print(f"Error: GGUF file not found: {gguf_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Converting {gguf_file.name} to Marlin format...")
    print(f"  Group size: {args.group_size}")
    print(f"  Output: {args.output_path}")
    print(f"  Validation: {'enabled' if not args.no_validate else 'disabled'}")
    print()

    stats = convert_gguf_to_marlin(
        str(gguf_file),
        args.output_path,
        group_size=args.group_size,
        validate=not args.no_validate,
    )

    if stats["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
