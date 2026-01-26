"""
Metal Marlin: FP4-quantized GEMM for Apple Silicon via MLX custom kernels.

Implements Marlin-style bitwise FP4 (E2M1 / NVFP4) dequantization fused with
matrix multiplication using Metal simdgroup operations. No lookup table; pure
ALU dequant in registers before simdgroup_multiply_accumulate.

Usage:
    from metal_marlin import quantized_linear, pack_fp4_weights

    # Pack weights from FP16 to Marlin FP4 format
    w_packed, scales = pack_fp4_weights(weight_fp16, group_size=32)

    # Run quantized matmul
    output = quantized_linear(x, w_packed, scales, group_size=32)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from .dtypes import DTypeConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Marlin uses 128 threads per threadgroup (4 simdgroups of 32)
THREADS_PER_TG = 128
SIMDGROUP_SIZE = 32
SIMDGROUPS_PER_TG = THREADS_PER_TG // SIMDGROUP_SIZE

# Tile sizes for the GEMM
# Each simdgroup computes an 8x8 output tile via simdgroup_multiply_accumulate
# With 4 simdgroups: 2 along M, 2 along N => 16x16 output per threadgroup
TILE_M = 16
TILE_N = 16
TILE_K = 32  # Process 32 elements of K per iteration (matches group_size)

# FP4 packing: 8 FP4 values per uint32
FP4_PER_U32 = 8

# ---------------------------------------------------------------------------
# Metal Kernel Source
# ---------------------------------------------------------------------------

_KERNEL_HEADER = """
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// FP4 (E2M1 / NVFP4) bitwise dequantization
// ---------------------------------------------------------------------------
// FP4 E2M1 format: [sign(1) | exponent(2) | mantissa(1)]
// FP16 format:     [sign(1) | exponent(5) | mantissa(10)]
//
// Conversion: extract 4-bit nibbles, reconstruct FP16 fields via shifts/masks.
// No lookup table needed - pure ALU.
//
// For E2M1 -> FP16:
//   sign_16 = sign_4 << 12  (bit 3 -> bit 15)
//   exp_16  = (exp_4 + 14) << 10  (bias correction: FP4 bias=1, FP16 bias=15, delta=14)
//   mant_16 = mant_4 << 9  (bit 0 -> bit 9)
//
// Special cases:
//   exp_4 == 0, mant == 0: zero (subnormal -> 0 for E2M1)
//   exp_4 == 0, mant == 1: subnormal = 0.5 * 2^(-1+1) = 0.5 -> encode as 0x3800
//   exp_4 == 3: max normal

inline half dequant_fp4_to_fp16(uint nibble) {
    // nibble is 4 bits: [s(1) | e(2) | m(1)]
    uint sign = (nibble >> 3) & 1u;
    uint exp4 = (nibble >> 1) & 3u;
    uint mant = nibble & 1u;

    ushort fp16_bits;
    if (exp4 == 0u && mant == 0u) {
        // Zero
        fp16_bits = ushort(sign << 15);
    } else if (exp4 == 0u && mant == 1u) {
        // Subnormal E2M1: value = 0.5
        // FP16 for 0.5 = 0x3800
        fp16_bits = ushort((sign << 15) | 0x3800u);
    } else {
        // Normal: exp16 = exp4 + 14 (bias correction)
        uint exp16 = exp4 + 14u;
        fp16_bits = ushort((sign << 15) | (exp16 << 10) | (mant << 9));
    }
    return as_type<half>(fp16_bits);
}

// Vectorized: dequant 8 FP4 values from a packed uint32
inline void dequant_fp4x8(uint packed, thread half* out) {
    for (uint i = 0; i < 8; i++) {
        uint nibble = (packed >> (i * 4)) & 0xFu;
        out[i] = dequant_fp4_to_fp16(nibble);
    }
}

// Branchless vectorized dequant for pairs (operates on two nibbles -> half2)
// This is the Marlin-style approach: process pairs via bit manipulation
inline half2 dequant_fp4x2_branchless(uint byte_val) {
    // byte_val contains two 4-bit FP4 values in lower 8 bits
    // lo nibble = bits [3:0], hi nibble = bits [7:4]
    uint lo = byte_val & 0xFu;
    uint hi = (byte_val >> 4) & 0xFu;

    // Extract fields for both
    uint s_lo = (lo >> 3) & 1u;
    uint e_lo = (lo >> 1) & 3u;
    uint m_lo = lo & 1u;

    uint s_hi = (hi >> 3) & 1u;
    uint e_hi = (hi >> 1) & 3u;
    uint m_hi = hi & 1u;

    // Construct FP16 bits - handle subnormal with select
    // Normal: (sign << 15) | ((exp+14) << 10) | (mant << 9)
    // Zero: (sign << 15)
    // Subnormal (exp=0, mant=1): (sign << 15) | 0x3800

    uint is_zero_lo = uint(e_lo == 0u && m_lo == 0u);
    uint is_sub_lo = uint(e_lo == 0u && m_lo == 1u);
    uint norm_lo = (s_lo << 15) | ((e_lo + 14u) << 10) | (m_lo << 9);
    uint sub_lo = (s_lo << 15) | 0x3800u;
    uint zero_lo = (s_lo << 15);

    ushort bits_lo = ushort(select(select(norm_lo, sub_lo, is_sub_lo != 0u), zero_lo, is_zero_lo != 0u));

    uint is_zero_hi = uint(e_hi == 0u && m_hi == 0u);
    uint is_sub_hi = uint(e_hi == 0u && m_hi == 1u);
    uint norm_hi = (s_hi << 15) | ((e_hi + 14u) << 10) | (m_hi << 9);
    uint sub_hi = (s_hi << 15) | 0x3800u;
    uint zero_hi = (s_hi << 15);

    ushort bits_hi = ushort(select(select(norm_hi, sub_hi, is_sub_hi != 0u), zero_hi, is_zero_hi != 0u));

    return half2(as_type<half>(bits_lo), as_type<half>(bits_hi));
}
"""

# The kernel body implements a tiled GEMM with fused FP4 dequantization.
# Each threadgroup computes a TILE_M x TILE_N output tile.
# The K dimension is iterated in chunks of TILE_K (= group_size).
# Within each K chunk, packed FP4 weights are dequantized in registers
# and accumulated into the output tile via simdgroup operations.
_KERNEL_SOURCE = """
    // Thread/group indices
    uint tg_x = threadgroup_position_in_grid.x;  // along N
    uint tg_y = threadgroup_position_in_grid.y;  // along M
    uint local_id = thread_position_in_threadgroup.x;
    uint simd_id = simdgroup_index_in_threadgroup;
    uint simd_lane = thread_index_in_simdgroup;

    // Each threadgroup handles TILE_M rows x TILE_N cols of output
    uint m_start = tg_y * TILE_M;
    uint n_start = tg_x * TILE_N;

    // Accumulator for this thread's contribution
    // We use a simple approach: each thread accumulates partial results
    // and we reduce at the end.
    //
    // Thread mapping within threadgroup (128 threads = 4 simdgroups):
    //   simd_id 0,1 handle the first 8 rows, simd_id 2,3 handle the next 8
    //   Within each simdgroup, lanes map to columns
    uint my_m_offset = (simd_id / (SIMDGROUPS_PER_TG / 2)) * (TILE_M / 2);  // 0 or 8
    uint my_n_offset = 0;

    // Each thread accumulates a subset of the output
    // Thread lane maps to output columns: lane_id covers 16 cols (with 32 lanes, 2 per col)
    uint col_idx = simd_lane % TILE_N;  // column within tile [0, TILE_N)
    uint row_pair = simd_lane / TILE_N; // which row pair this lane handles [0, 2)

    // Each thread accumulates results for (TILE_M/2 / (SIMDGROUP_SIZE/TILE_N)) rows
    // = 8 / 2 = 4 rows per thread, 1 column
    // Simpler: each thread handles specific (row, col) pairs
    // With TILE_M=16, TILE_N=16, 128 threads: each thread handles 2 output elements

    // Flat output assignment: thread i handles output[i / TILE_N][i % TILE_N]
    // 128 threads, 16x16=256 outputs -> each thread handles 2 outputs
    uint out_idx_0 = local_id;                    // first output element
    uint out_idx_1 = local_id + THREADS_PER_TG;   // second output element (128..255)

    uint out_row_0 = out_idx_0 / TILE_N;
    uint out_col_0 = out_idx_0 % TILE_N;
    uint out_row_1 = out_idx_1 / TILE_N;
    uint out_col_1 = out_idx_1 % TILE_N;

    float acc_0 = 0.0f;
    float acc_1 = 0.0f;

    // Iterate over K dimension in chunks of TILE_K
    uint num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (uint k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        uint k_start = k_tile * TILE_K;

        // Load scale for this group (scales shape: [K/group_size, N])
        uint scale_row = k_start / GROUP_SIZE;

        // --- Accumulate for output element 0 ---
        uint global_row_0 = m_start + out_row_0;
        uint global_col_0 = n_start + out_col_0;

        if (global_row_0 < M && global_col_0 < N) {
            half s0 = scales[scale_row * N + global_col_0];

            // Inner loop over K within this tile
            for (uint dk = 0; dk < TILE_K && (k_start + dk) < K; dk++) {
                uint k_idx = k_start + dk;

                // Load activation
                half a_val = x[global_row_0 * K + k_idx];

                // Load and dequant weight
                // Weight packing: weights shape [K, N/8] as uint32
                // Each uint32 holds 8 consecutive N-dimension FP4 values
                // For weight at (k_idx, global_col_0):
                //   packed word index = k_idx * (N/8) + global_col_0/8
                //   nibble position = global_col_0 % 8
                uint packed_idx = k_idx * PACKED_N + (global_col_0 / FP4_PER_WORD);
                uint packed_word = weight_packed[packed_idx];
                uint nibble_pos = global_col_0 % FP4_PER_WORD;
                uint nibble = (packed_word >> (nibble_pos * 4u)) & 0xFu;
                half w_val = dequant_fp4_to_fp16(nibble);

                // Scale and accumulate
                acc_0 += float(a_val) * float(w_val) * float(s0);
            }
        }

        // --- Accumulate for output element 1 ---
        uint global_row_1 = m_start + out_row_1;
        uint global_col_1 = n_start + out_col_1;

        if (global_row_1 < M && global_col_1 < N) {
            half s1 = scales[scale_row * N + global_col_1];

            for (uint dk = 0; dk < TILE_K && (k_start + dk) < K; dk++) {
                uint k_idx = k_start + dk;
                half a_val = x[global_row_1 * K + k_idx];

                uint packed_idx = k_idx * PACKED_N + (global_col_1 / FP4_PER_WORD);
                uint packed_word = weight_packed[packed_idx];
                uint nibble_pos = global_col_1 % FP4_PER_WORD;
                uint nibble = (packed_word >> (nibble_pos * 4u)) & 0xFu;
                half w_val = dequant_fp4_to_fp16(nibble);

                acc_1 += float(a_val) * float(w_val) * float(s1);
            }
        }
    }

    // Write output
    uint global_row_0_final = m_start + out_row_0;
    uint global_col_0_final = n_start + out_col_0;
    if (global_row_0_final < M && global_col_0_final < N) {
        out[global_row_0_final * N + global_col_0_final] = half(acc_0);
    }

    uint global_row_1_final = m_start + out_row_1;
    uint global_col_1_final = n_start + out_col_1;
    if (global_row_1_final < M && global_col_1_final < N) {
        out[global_row_1_final * N + global_col_1_final] = half(acc_1);
    }
"""


def _build_kernel() -> object:
    """Build and cache the Metal Marlin GEMM kernel."""
    kernel = mx.fast.metal_kernel(
        name="marlin_gemm_fp4",
        input_names=["x", "weight_packed", "scales"],
        output_names=["out"],
        source=_KERNEL_SOURCE,
        header=_KERNEL_HEADER,
        ensure_row_contiguous=True,
    )
    return kernel


# Module-level cached kernel (lazy init)
_kernel: object | None = None


def _get_kernel() -> object:
    global _kernel
    if _kernel is None:
        _kernel = _build_kernel()
    return _kernel


# ---------------------------------------------------------------------------
# Weight Packing
# ---------------------------------------------------------------------------


def pack_fp4_weights(weight: mx.array, group_size: int = 32) -> tuple[mx.array, mx.array]:
    """
    Pack FP16 weights into Marlin FP4 format with per-group scales.

    The weight matrix is quantized per-group along K (input dimension).
    Each group of `group_size` elements shares one FP16 scale factor.

    Packing layout: weights[K, N] -> packed[K, N//8] as uint32,
    where 8 consecutive N-dimension values are packed into one uint32.

    FP4 E2M1 encoding:
        value -> nearest E2M1 representable value, packed as 4-bit nibble.

    Args:
        weight: FP16 weight matrix [out_features, in_features] (row-major,
                following PyTorch convention where output dim is first).
                Will be transposed internally for the kernel layout [K, N].
        group_size: Number of elements per quantization group. Default: 32.

    Returns:
        Tuple of:
            weight_packed: uint32 array [K, N//8] with packed FP4 nibbles.
            scales: float16 array [K//group_size, N] with per-group scales.
    """
    # Transpose to [K, N] layout for the kernel (K = in_features, N = out_features)
    w = weight.T.astype(mx.float16)  # [K, N]
    K, N = w.shape

    if N % FP4_PER_U32 != 0:
        raise ValueError(f"N ({N}) must be divisible by {FP4_PER_U32}")
    if K % group_size != 0:
        raise ValueError(f"K ({K}) must be divisible by group_size ({group_size})")

    # Compute per-group scales: map absmax to E2M1 max value (6.0)
    # Groups are along K dimension: [K//group_size, group_size, N]
    #
    # E2M1 representable values: 0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
    # To use the full quantization range, we scale so that the group's
    # absmax maps to 6.0 (the maximum E2M1 magnitude).
    #
    # scale = absmax / 6.0
    # normalized = w / scale = w * 6 / absmax -> values in [-6, 6]
    # dequant = fp4_value * scale = fp4_value * absmax / 6
    max_e2m1 = 6.0

    w_grouped = w.reshape(K // group_size, group_size, N)
    absmax = mx.max(mx.abs(w_grouped), axis=1)  # [K//group_size, N]

    # Avoid division by zero
    absmax = mx.maximum(absmax, mx.array(1e-7, dtype=mx.float16))

    # Scale factor: the value that, when multiplied by an E2M1 value, gives
    # the original weight. Since we normalize to [-6, 6], scale = absmax / 6.
    scales = absmax / max_e2m1

    # E2M1 representable values (all 16 nibble patterns):
    # We enumerate them and find the nearest for quantization
    e2m1_values = _get_e2m1_values()  # 16 float16 values

    # Normalize to E2M1 range [-6, 6]
    scales_expanded = mx.repeat(scales, group_size, axis=0)  # [K, N]
    w_normalized = w / scales_expanded  # values in [-6, 6]

    # Clamp to E2M1 range (should be a no-op if scaling is correct)
    w_normalized = mx.clip(w_normalized, -max_e2m1, max_e2m1)

    # Map to nearest E2M1 nibble index
    # We do this on CPU since it's a one-time cost at model load
    w_np = w_normalized.astype(mx.float32)
    mx.eval(w_np)
    mx.eval(scales)

    import numpy as np

    w_np_arr = np.array(w_np)
    e2m1_np = np.array(e2m1_values.astype(mx.float32))  # [16]

    # For each element, find nearest E2M1 value
    # Broadcast: w_np_arr[K,N,1] vs e2m1_np[16]
    # This could be memory-intensive for large matrices; chunk if needed
    packed_n = N // FP4_PER_U32
    packed = np.zeros((K, packed_n), dtype=np.uint32)

    for col_group in range(packed_n):
        # 8 columns packed together
        col_start = col_group * FP4_PER_U32
        cols = w_np_arr[:, col_start : col_start + FP4_PER_U32]  # [K, 8]

        # Find nearest E2M1 index for each element
        # Distance: [K, 8, 1] - [16] -> [K, 8, 16]
        dists = np.abs(cols[:, :, None] - e2m1_np[None, None, :])
        nibble_indices = np.argmin(dists, axis=2).astype(np.uint32)  # [K, 8]

        # Pack 8 nibbles into uint32
        word = np.zeros((K,), dtype=np.uint32)
        for bit_pos in range(FP4_PER_U32):
            word |= nibble_indices[:, bit_pos] << (bit_pos * 4)

        packed[:, col_group] = word

    weight_packed = mx.array(packed)
    return weight_packed, scales


# ---------------------------------------------------------------------------
# INT4 (U4) Weight Packing — matches dequant_u4x8 / dequant_int4_bulk kernel
# ---------------------------------------------------------------------------

U4_PER_U32 = 8  # 8 nibbles per uint32


def pack_u4_weights(weight: mx.array, group_size: int = 128) -> tuple[mx.array, mx.array, mx.array]:
    """
    Pack FP16 weights into unsigned INT4 format with per-group scale and zero point.

    Quantization formula (per group):
        scale = (max_val - min_val) / 15
        zero  = round(-min_val / scale)    # nibble value that maps to 0.0
        q     = round(weight / scale + zero)  clamped to [0, 15]

    Dequantization (what the Metal kernel computes):
        weight_approx = (q - zero) * scale

    Mathematical inverse relationship:
        The pack/unpack are inverses within quantization error:
        weight ≈ (q - zero) * scale
              = (round(w/s + z) - z) * s
              ≈ w

    Packing layout: weights[K, N] -> packed[K // 8, N] as uint32,
    where 8 consecutive K-dimension values are packed into one uint32.

    Nibble ordering (LSB-first, little-endian):
        val[k+0] at bits [3:0]
        val[k+1] at bits [7:4]
        ...
        val[k+7] at bits [31:28]

    This matches the dequant_u4x8 and dequant_int4_bulk Metal kernels in
    src/dequant.metal which extract nibbles as: (packed >> (i * 4)) & 0xF

    Args:
        weight: FP16 weight matrix [out_features, in_features].
                Transposed internally to [K, N] for the kernel layout.
        group_size: Number of K-dimension elements per quantization group.

    Returns:
        Tuple of:
            packed: uint32 array [K // 8, N] with packed U4 nibbles.
            scales: float16 array [K // group_size, N] with per-group scales.
            zeros:  float16 array [K // group_size, N] with per-group zero points.
    """
    import numpy as np

    # Transpose to [K, N] for kernel layout
    w = weight.T.astype(mx.float16)
    K, N = w.shape

    if K % U4_PER_U32 != 0:
        raise ValueError(f"K ({K}) must be divisible by {U4_PER_U32}")
    if K % group_size != 0:
        raise ValueError(f"K ({K}) must be divisible by group_size ({group_size})")

    mx.eval(w)
    w_np = np.array(w.astype(mx.float32))

    num_groups = K // group_size
    w_grouped = w_np.reshape(num_groups, group_size, N)

    # Per-group min/max for asymmetric quantization
    group_min = w_grouped.min(axis=1)  # [num_groups, N]
    group_max = w_grouped.max(axis=1)  # [num_groups, N]

    # Scale: range / 15 (15 = max representable U4 interval)
    scales_np = (group_max - group_min) / 15.0
    # Avoid division by zero for constant groups
    scales_np = np.maximum(scales_np, np.float32(1e-7))

    # Zero point: the nibble value that corresponds to 0.0.
    # From dequant formula: weight = (q - zero) * scale
    # To map min -> q=0:  min = (0 - zero) * scale  => zero = -min / scale
    # To map max -> q=15: max = (15 - zero) * scale => zero = 15 - max/scale
    # Both give the same zero when scale = (max-min)/15.
    # The zero point is stored as FP16 and is NOT restricted to [0,15];
    # the kernel treats it as a float subtracted from the nibble value.
    zeros_np = np.round(-group_min / scales_np).astype(np.float32)

    # Quantize: q = round(w / scale + zero), clamped to [0, 15]
    scales_expanded = np.repeat(scales_np, group_size, axis=0)  # [K, N]
    zeros_expanded = np.repeat(zeros_np, group_size, axis=0)  # [K, N]
    q = np.round(w_np / scales_expanded + zeros_expanded).clip(0, 15).astype(np.uint32)

    # Pack 8 K-values into one uint32 (LSB-first nibble order)
    k_packs = K // U4_PER_U32
    packed = np.zeros((k_packs, N), dtype=np.uint32)
    for k_pack in range(k_packs):
        k_base = k_pack * U4_PER_U32
        for i in range(U4_PER_U32):
            packed[k_pack, :] |= q[k_base + i, :] << (i * 4)

    return (
        mx.array(packed),
        mx.array(scales_np.astype(np.float16)),
        mx.array(zeros_np.astype(np.float16)),
    )


def unpack_u4_weights(
    packed: mx.array, scales: mx.array, zeros: mx.array, orig_shape: tuple[int, int] | None = None
) -> mx.array:
    """
    Unpack U4 weights back to FP16 for validation.

    Reverses pack_u4_weights: extracts nibbles from uint32 words, applies
    asymmetric dequantization: result = (nibble - zero) * scale.

    Args:
        packed: uint32 array [K // 8, N].
        scales: float16 array [K // group_size, N].
        zeros:  float16 array [K // group_size, N].
        orig_shape: Optional (out_features, in_features) to trim padding.

    Returns:
        FP16 array [out_features, in_features] (transposed back to PyTorch convention).
    """
    import numpy as np

    mx.eval(packed, scales, zeros)
    packed_np = np.array(packed)
    scales_np = np.array(scales.astype(mx.float32))
    zeros_np = np.array(zeros.astype(mx.float32))

    k_packs, N = packed_np.shape
    K = k_packs * U4_PER_U32

    # Validate scale/zero shapes match packed dimensions
    if scales_np.shape[1] != N:
        raise ValueError(
            f"scales shape {scales_np.shape} N-dimension mismatch with packed shape ({k_packs}, {N})"
        )
    if zeros_np.shape != scales_np.shape:
        raise ValueError(f"zeros shape {zeros_np.shape} must match scales shape {scales_np.shape}")

    # Infer group_size from scales shape
    num_groups = scales_np.shape[0]
    if K % num_groups != 0:
        raise ValueError(
            f"K={K} must be evenly divisible by num_groups={num_groups} inferred from scales"
        )
    group_size = K // num_groups

    # Extract nibbles: packed[k_pack, n] has 8 nibbles for K positions
    q = np.empty((K, N), dtype=np.uint32)
    for i in range(U4_PER_U32):
        q[i::U4_PER_U32, :] = (packed_np >> (i * 4)) & 0xF

    # The above interleaving is incorrect for contiguous K. Fix:
    q_correct = np.empty((K, N), dtype=np.uint32)
    for k_pack in range(k_packs):
        k_base = k_pack * U4_PER_U32
        for i in range(U4_PER_U32):
            q_correct[k_base + i, :] = (packed_np[k_pack, :] >> (i * 4)) & 0xF

    # Dequantize: (q - zero) * scale
    scales_expanded = np.repeat(scales_np, group_size, axis=0)  # [K, N]
    zeros_expanded = np.repeat(zeros_np, group_size, axis=0)  # [K, N]
    w_approx = (q_correct.astype(np.float32) - zeros_expanded) * scales_expanded

    # Transpose back to [N, K] (PyTorch convention)
    result = w_approx.T.astype(np.float16)  # [N, K]

    if orig_shape is not None:
        result = result[: orig_shape[0], : orig_shape[1]]

    return mx.array(result)


def _get_e2m1_values() -> mx.array:
    """
    Return all 16 E2M1 representable values as float16.

    Nibble encoding: [sign(1) | exp(2) | mant(1)]
    Bias = 1.
    """
    import numpy as np

    values = np.zeros(16, dtype=np.float32)
    for nibble in range(16):
        sign = (nibble >> 3) & 1
        exp_bits = (nibble >> 1) & 3
        mant_bit = nibble & 1

        if exp_bits == 0 and mant_bit == 0:
            val = 0.0
        elif exp_bits == 0 and mant_bit == 1:
            # Subnormal: 0.5 * 2^(1-1) = 0.5
            val = 0.5
        else:
            # Normal: (1 + mant*0.5) * 2^(exp - bias)
            # bias = 1
            mantissa = 1.0 + mant_bit * 0.5
            exponent = exp_bits - 1  # bias = 1
            val = mantissa * (2.0**exponent)

        if sign:
            val = -val
        values[nibble] = val

    return mx.array(values, dtype=mx.float16)


# ---------------------------------------------------------------------------
# Quantized Linear (Main API)
# ---------------------------------------------------------------------------


def quantized_linear(
    x: mx.array,
    weight_packed: mx.array,
    scales: mx.array,
    group_size: int = 32,
    dtype_config: DTypeConfig | None = None,
) -> mx.array:
    """
    FP4-quantized linear layer using Marlin-style fused dequant-GEMM kernel.

    Computes: output = x @ dequant(weight_packed, scales).T
    where dequant reconstructs FP16 weights from packed FP4 nibbles.

    Args:
        x: Input activations. Supports shapes:
            - [M, K] (2D matrix)
            - [batch, seq_len, K] (3D batched)
            - [*, K] (arbitrary leading dimensions)
        weight_packed: Packed FP4 weights [K, N//8] as uint32.
            N = output features, K = input features.
        scales: Per-group quantization scales [K//group_size, N] as float16.
        group_size: Elements per quantization group. Must match packing. Default: 32.
        dtype_config: DTypeConfig for controlling compute dtype. Default: BF16.

    Returns:
        Output tensor with same leading dimensions as x, last dim = N.
    """
    from .dtypes import get_default_config

    config = dtype_config if dtype_config is not None else get_default_config()
    compute_dtype = config.mlx_activations

    # Flatten leading dimensions
    orig_shape = x.shape
    K = orig_shape[-1]
    M = 1
    for d in orig_shape[:-1]:
        M *= d

    x_2d = x.reshape(M, K).astype(compute_dtype)

    # Derive N from weight_packed shape
    packed_n = weight_packed.shape[1]
    N = packed_n * FP4_PER_U32

    # Compute grid dimensions (number of threadgroups along each axis)
    n_tgs = (N + TILE_N - 1) // TILE_N
    m_tgs = (M + TILE_M - 1) // TILE_M

    kernel = _get_kernel()

    # MLX metal_kernel grid parameter specifies total threads, not threadgroups.
    # Multiply by threadgroup size to get the correct dispatch.
    # Note: Metal kernel uses half (FP16) internally for simdgroup ops.
    # We cast input to FP16 for the kernel, then output to compute_dtype.
    x_kernel = x_2d.astype(mx.float16) if compute_dtype != mx.float16 else x_2d
    outputs = kernel(
        inputs=[x_kernel, weight_packed, scales],
        template=[
            ("M", M),
            ("N", N),
            ("K", K),
            ("GROUP_SIZE", group_size),
            ("TILE_M", TILE_M),
            ("TILE_N", TILE_N),
            ("TILE_K", TILE_K),
            ("THREADS_PER_TG", THREADS_PER_TG),
            ("SIMDGROUPS_PER_TG", SIMDGROUPS_PER_TG),
            ("FP4_PER_WORD", FP4_PER_U32),
            ("PACKED_N", packed_n),
        ],
        grid=(n_tgs * THREADS_PER_TG, m_tgs, 1),
        threadgroup=(THREADS_PER_TG, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[mx.float16],
        init_value=0.0,
    )

    out = outputs[0]
    if compute_dtype != mx.float16:
        out = out.astype(compute_dtype)

    # Reshape back to original leading dims
    out_shape = list(orig_shape[:-1]) + [N]
    return out.reshape(out_shape)


# ---------------------------------------------------------------------------
# Striped GEMM with K-parallel reduction
# ---------------------------------------------------------------------------

# Tile dimensions matching the Metal kernel constants
_STRIPED_TILE_M = 64
_STRIPED_TILE_N = 64
_STRIPED_TILE_K = 32
_STRIPED_THREADS_PER_TG = 128

_ZERO_KERNEL_SOURCE = """
    uint tid = thread_position_in_grid.x;
    if (tid < BUF_ELEMS) {
        reduction_buf[tid] = half(0.0h);
    }
    if (tid < NUM_LOCKS) {
        locks[tid] = 0;
    }
"""

_STRIPED_KERNEL_SOURCE = None  # Loaded from .metal file on first use

_zero_kernel: object | None = None
_striped_kernel: object | None = None


def _get_zero_kernel() -> object:
    """Build and cache the zero-reduction kernel."""
    global _zero_kernel
    if _zero_kernel is None:
        _zero_kernel = mx.fast.metal_kernel(
            name="marlin_zero_reduction",
            input_names=[],
            output_names=["reduction_buf", "locks"],
            source=_ZERO_KERNEL_SOURCE,
            ensure_row_contiguous=False,
        )
    return _zero_kernel


def _get_striped_kernel_source() -> str:
    """Load the striped GEMM kernel source from the .metal file."""
    global _STRIPED_KERNEL_SOURCE
    if _STRIPED_KERNEL_SOURCE is None:
        from pathlib import Path

        metal_path = Path(__file__).parent.parent / "src" / "marlin_gemm.metal"
        _STRIPED_KERNEL_SOURCE = metal_path.read_text()
    return _STRIPED_KERNEL_SOURCE


def _div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def dispatch_zero_reduction(
    reduction_buf: mx.array,
    locks: mx.array,
    buf_elems: int,
    num_locks: int,
) -> tuple[mx.array, mx.array]:
    """Dispatch the zero-reduction kernel to initialize reduction buffers.

    Must be called BEFORE marlin_gemm_fp4_striped when parallel > 1.
    Zeroes the reduction_buf and resets all lock counters to 0.

    Args:
        reduction_buf: FP16 buffer of shape [parallel * M * N] for partial sums.
        locks: Int32 buffer of shape [m_tiles * n_tiles] for atomic counters.
        buf_elems: Total elements in reduction_buf (parallel * M * N).
        num_locks: Total lock entries (m_tiles * n_tiles).

    Returns:
        Tuple of (zeroed reduction_buf, zeroed locks).
    """
    kernel = _get_zero_kernel()
    max_elems = max(buf_elems, num_locks)
    grid_size = _div_ceil(max_elems, _STRIPED_THREADS_PER_TG)

    outputs = kernel(
        inputs=[reduction_buf, locks],
        template=[
            ("BUF_ELEMS", buf_elems),
            ("NUM_LOCKS", num_locks),
        ],
        grid=(grid_size * _STRIPED_THREADS_PER_TG, 1, 1),
        threadgroup=(_STRIPED_THREADS_PER_TG, 1, 1),
        output_shapes=[reduction_buf.shape, locks.shape],
        output_dtypes=[mx.float16, mx.int32],
        init_value=0,
    )
    return outputs[0], outputs[1]


def quantized_linear_striped(
    x: mx.array,
    weight_packed: mx.array,
    scales: mx.array,
    group_size: int = 32,
    parallel: int = 1,
    num_threadgroups: int | None = None,
    dtype_config: DTypeConfig | None = None,
) -> mx.array:
    """FP4-quantized linear layer using stripe-partitioned GEMM with K-parallel reduction.

    When parallel > 1, the K-dimension is split across multiple threadgroups
    that compute partial sums and reduce via atomic counters. This improves
    occupancy for large K dimensions on high-core-count GPUs.

    Args:
        x: Input activations [*, K] (arbitrary leading dimensions).
        weight_packed: Packed FP4 weights [K, N//8] as uint32.
        scales: Per-group quantization scales [K//group_size, N] as float16.
        group_size: Elements per quantization group. Default: 32.
        parallel: K-parallel factor. 1 = no reduction. >1 splits K across
            multiple threadgroups per output tile. Default: 1.
        num_threadgroups: Total threadgroups to dispatch. If None, defaults to
            m_tiles * n_tiles * parallel (full coverage). Override to match
            GPU core count for optimal occupancy.
        dtype_config: DTypeConfig for controlling compute dtype. Default: BF16.

    Returns:
        Output tensor with same leading dimensions as x, last dim = N.
    """
    from .dtypes import get_default_config

    config = dtype_config if dtype_config is not None else get_default_config()
    compute_dtype = config.mlx_activations

    orig_shape = x.shape
    K = orig_shape[-1]
    M = 1
    for d in orig_shape[:-1]:
        M *= d

    x_2d = x.reshape(M, K).astype(mx.float16)  # Metal kernel uses half

    packed_n = weight_packed.shape[1]
    N = packed_n * FP4_PER_U32

    m_tiles = _div_ceil(M, _STRIPED_TILE_M)
    n_tiles = _div_ceil(N, _STRIPED_TILE_N)
    total_tiles = m_tiles * n_tiles

    if num_threadgroups is None:
        num_threadgroups = total_tiles * parallel

    # Allocate reduction buffers
    buf_elems = parallel * M * N
    num_locks = total_tiles

    reduction_buf = mx.zeros((buf_elems,), dtype=mx.float16)
    locks = mx.zeros((num_locks,), dtype=mx.int32)

    # Zero-init reduction buffers when parallel > 1
    if parallel > 1:
        reduction_buf, locks = dispatch_zero_reduction(reduction_buf, locks, buf_elems, num_locks)

    # Build the striped kernel (inline source matching the .metal kernel logic)
    # Uses the same tile dimensions and pipeline as marlin_gemm_fp4_striped
    striped_source = """
    uint tgid_x = threadgroup_position_in_grid.x;
    uint simd_lane = thread_index_in_simdgroup;
    uint simd_id = simdgroup_index_in_threadgroup;

    const uint TILE_M_L = 64;
    const uint TILE_N_L = 64;
    const uint TILE_K_L = 32;
    const uint THREADS_PER_TG_L = 128;
    const uint FP4_PER_UINT_L = 8;
    const uint SG_M_TILES_L = 2;
    const uint SG_N_TILES_L = 4;
    const uint K_TILES_L = 4;

    uint m_tiles = (M + TILE_M_L - 1) / TILE_M_L;
    uint n_tiles = (N + TILE_N_L - 1) / TILE_N_L;
    uint total_tiles = m_tiles * n_tiles;
    uint total_work = total_tiles * PARALLEL;
    uint work_per_tg = (total_work + NUM_TGS - 1) / NUM_TGS;
    uint work_start = tgid_x * work_per_tg;
    uint work_end = min(work_start + work_per_tg, total_work);

    if (work_start >= total_work) return;

    uint k_iters_total = (K + TILE_K_L - 1) / TILE_K_L;
    uint k_iters_per_slice = (k_iters_total + PARALLEL - 1) / PARALLEL;

    uint thread_idx = simd_id * 32 + simd_lane;
    uint sg_row_offset = (simd_id / 2) * (SG_M_TILES_L * 8);
    uint sg_col_offset = (simd_id % 2) * (SG_N_TILES_L * 8);

    for (uint work_idx = work_start; work_idx < work_end; ++work_idx) {
        uint tile_linear = work_idx / PARALLEL;
        uint k_slice = work_idx % PARALLEL;
        uint tile_row = tile_linear / n_tiles;
        uint tile_col = tile_linear % n_tiles;
        uint tg_row = tile_row * TILE_M_L;
        uint tg_col = tile_col * TILE_N_L;

        uint k_start_iter = k_slice * k_iters_per_slice * TILE_K_L;
        uint k_end_iter = min((k_slice + 1) * k_iters_per_slice, k_iters_total);
        uint k_end = min(k_end_iter * TILE_K_L, K);

        // Accumulators
        float acc[SG_M_TILES_L][SG_N_TILES_L];
        for (uint mi = 0; mi < SG_M_TILES_L; ++mi)
            for (uint ni = 0; ni < SG_N_TILES_L; ++ni)
                acc[mi][ni] = 0.0f;

        // K-loop over this slice's range
        for (uint k_block = k_start_iter; k_block < k_end; k_block += TILE_K_L) {
            // Each thread computes its portion
            for (uint mi = 0; mi < SG_M_TILES_L; ++mi) {
                for (uint ni = 0; ni < SG_N_TILES_L; ++ni) {
                    // Simplified scalar accumulation for correctness
                    uint out_row = tg_row + sg_row_offset + mi * 8 + (thread_idx % 8);
                    uint out_col = tg_col + sg_col_offset + ni * 8 + (thread_idx / 8) % 8;

                    if (out_row < M && out_col < N) {
                        for (uint dk = 0; dk < TILE_K_L && (k_block + dk) < k_end; ++dk) {
                            uint k_idx = k_block + dk;
                            half a_val = A[out_row * K + k_idx];

                            uint b_row = k_idx / FP4_PER_UINT_L;
                            uint packed_word = B[b_row * N + out_col];
                            uint nibble_pos = k_idx % FP4_PER_UINT_L;
                            uint nibble = (packed_word >> (nibble_pos * 4u)) & 0xFu;

                            // FP4 E2M1 dequant
                            uint sign_bit = (nibble >> 3) & 1u;
                            uint exp_bits = (nibble >> 1) & 0x3u;
                            uint man_bit  = nibble & 1u;
                            half magnitude;
                            if (exp_bits == 0u) {
                                magnitude = half(man_bit) * half(0.25h);
                            } else {
                                half power = half(1u << (exp_bits - 1u));
                                half mantissa = half(1.0h) + half(man_bit) * half(0.5h);
                                magnitude = power * mantissa;
                            }
                            half w_val = sign_bit ? -magnitude : magnitude;

                            uint scale_k = k_idx / GROUP_SIZE;
                            half s = scales[scale_k * N + out_col];
                            w_val = w_val * s;

                            acc[mi][ni] += float(a_val) * float(w_val);
                        }
                    }
                }
            }
        }

        // Store results
        if (PARALLEL == 1) {
            for (uint mi = 0; mi < SG_M_TILES_L; ++mi) {
                for (uint ni = 0; ni < SG_N_TILES_L; ++ni) {
                    uint out_row = tg_row + sg_row_offset + mi * 8 + (thread_idx % 8);
                    uint out_col = tg_col + sg_col_offset + ni * 8 + (thread_idx / 8) % 8;
                    if (out_row < M && out_col < N) {
                        C[out_row * N + out_col] = half(acc[mi][ni]);
                    }
                }
            }
        } else {
            // Write partial sums to reduction buffer
            for (uint mi = 0; mi < SG_M_TILES_L; ++mi) {
                for (uint ni = 0; ni < SG_N_TILES_L; ++ni) {
                    uint out_row = tg_row + sg_row_offset + mi * 8 + (thread_idx % 8);
                    uint out_col = tg_col + sg_col_offset + ni * 8 + (thread_idx / 8) % 8;
                    if (out_row < M && out_col < N) {
                        reduction_buf[k_slice * M * N + out_row * N + out_col] = half(acc[mi][ni]);
                    }
                }
            }

            // Flush this threadgroup's device writes to reduction_buf
            // before any thread increments the atomic counter.
            threadgroup_barrier(mem_flags::mem_device);

            // Atomic counter with acq_rel: the releasing threadgroups'
            // device writes become visible to the acquiring (last) one.
            threadgroup int is_last_slice;
            if (thread_idx == 0) {
                int prev = atomic_fetch_add_explicit(
                    (device atomic_int*)&locks[tile_linear], 1, memory_order_acq_rel);
                is_last_slice = (prev == int(PARALLEL) - 1);
            }
            // Broadcast is_last_slice to all threads in this threadgroup.
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (is_last_slice) {
                // Thread 0 acquired via acq_rel, but other threads haven't.
                // Device barrier ensures all threads see other slices' writes.
                threadgroup_barrier(mem_flags::mem_device);

                // Reset the lock for next dispatch
                if (thread_idx == 0) {
                    atomic_store_explicit(
                        (device atomic_int*)&locks[tile_linear], 0, memory_order_relaxed);
                }

                // Ordered reduction: sum partial sums in fixed order (s=0..PARALLEL-1)
                // for bit-exact determinism regardless of scheduling.
                uint tile_elems = TILE_M_L * TILE_N_L;
                uint elems_per_thread = tile_elems / THREADS_PER_TG_L;
                for (uint i = 0; i < elems_per_thread; ++i) {
                    uint flat_idx = thread_idx * elems_per_thread + i;
                    uint row = flat_idx / TILE_N_L;
                    uint col = flat_idx % TILE_N_L;
                    uint gr = tg_row + row;
                    uint gc = tg_col + col;
                    if (gr < M && gc < N) {
                        float sum = 0.0f;
                        for (uint s = 0; s < PARALLEL; ++s) {
                            sum += float(reduction_buf[s * M * N + gr * N + gc]);
                        }
                        C[gr * N + gc] = half(sum);
                    }
                }
            }
        }
    }
"""

    striped_kernel = mx.fast.metal_kernel(
        name="marlin_gemm_fp4_striped_inline",
        input_names=["A", "B", "scales", "reduction_buf", "locks"],
        output_names=["C"],
        source=striped_source,
        ensure_row_contiguous=True,
    )

    outputs = striped_kernel(
        inputs=[x_2d, weight_packed, scales, reduction_buf, locks],
        template=[
            ("M", M),
            ("N", N),
            ("K", K),
            ("GROUP_SIZE", group_size),
            ("PARALLEL", parallel),
            ("NUM_TGS", num_threadgroups),
        ],
        grid=(num_threadgroups, 1, 1),
        threadgroup=(_STRIPED_THREADS_PER_TG, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[mx.float16],
        init_value=0.0,
    )

    out = outputs[0]
    if compute_dtype != mx.float16:
        out = out.astype(compute_dtype)
    out_shape = list(orig_shape[:-1]) + [N]
    return out.reshape(out_shape)


# ---------------------------------------------------------------------------
# MLX QuantizedLinear-compatible Module
# ---------------------------------------------------------------------------


class MarlinLinear:
    """
    Drop-in replacement for mlx.nn.QuantizedLinear using Metal Marlin kernels.

    Wraps the custom FP4 dequant-GEMM kernel in a module interface compatible
    with mlx_lm model loading.

    Args:
        weight_packed: Packed FP4 weights [K, N//8] as uint32.
        scales: Per-group scales [K//group_size, N] as float16.
        bias: Optional bias vector [N] as float16.
        group_size: Quantization group size. Default: 32.
        dtype_config: DTypeConfig for controlling compute dtype. Default: BF16.
    """

    def __init__(
        self,
        weight_packed: mx.array,
        scales: mx.array,
        bias: mx.array | None = None,
        group_size: int = 32,
        dtype_config: DTypeConfig | None = None,
    ):
        self.weight_packed = weight_packed
        self.scales = scales
        self.bias = bias
        self.group_size = group_size
        self.dtype_config = dtype_config

    def __call__(self, x: mx.array) -> mx.array:
        out = quantized_linear(
            x, self.weight_packed, self.scales, self.group_size, self.dtype_config
        )
        if self.bias is not None:
            out = out + self.bias
        return out

    @staticmethod
    def from_quantized_linear(ql: nn.QuantizedLinear) -> MarlinLinear:
        """
        Convert an mlx.nn.QuantizedLinear (affine 4-bit) to MarlinLinear (FP4).

        Note: This re-quantizes the dequantized weights into FP4 format.
        There will be a small quality difference due to the different
        quantization scheme (affine INT4 -> FP4 E2M1).
        """
        # Dequantize from MLX affine format
        w_fp16 = mx.dequantize(
            ql.weight,
            ql.scales,
            ql.biases,
            ql.group_size,
            ql.bits,
        )
        mx.eval(w_fp16)

        # Re-quantize to FP4
        # w_fp16 is [out_features, in_features] (already transposed relative to kernel)
        w_packed, scales = pack_fp4_weights(w_fp16, group_size=32)

        bias = ql.bias if hasattr(ql, "bias") and "bias" in ql else None
        return MarlinLinear(w_packed, scales, bias, group_size=32)


# ---------------------------------------------------------------------------
# Bulk Dequantization Dispatch Helpers
# ---------------------------------------------------------------------------
#
# These functions dispatch the standalone dequant_fp4_bulk / dequant_int4_bulk
# Metal kernels for cases where dequantization cannot be fused into GEMM.
# Typical uses: debugging, feeding into non-matmul ops, accuracy validation.

# Shader source for the bulk dequant kernels. This is a self-contained header
# that includes only the primitives needed by the bulk kernels, avoiding
# dependency on the full dequant.metal file at runtime.
_BULK_DEQUANT_HEADER = """
#include <metal_stdlib>
using namespace metal;

constant constexpr uint32_t MAGIC_BIAS_U32 = 0x64006400u;
constant constexpr uint32_t LO_NIBBLE_MASK = 0x000F000Fu;
constant constexpr uint16_t MAGIC_BIAS_F16 = 0x6400u;

inline half dequant_fp4_scalar(uint nibble) {
    uint S = (nibble >> 3) & 1u;
    uint E = (nibble >> 1) & 3u;
    uint M = nibble & 1u;
    uint exp_normal = E + 14u;
    uint mant_normal = M << 9;
    bool is_normal = (E != 0u);
    uint fp16_exp = select(14u * M, exp_normal, is_normal);
    uint fp16_mant = select(0u, mant_normal, is_normal);
    ushort fp16_bits = ushort((S << 15) | (fp16_exp << 10) | fp16_mant);
    return as_type<half>(fp16_bits);
}

inline void dequant_fp4_x8(uint32_t packed, thread half *out) {
    out[0] = dequant_fp4_scalar((packed >>  0) & 0xFu);
    out[1] = dequant_fp4_scalar((packed >>  4) & 0xFu);
    out[2] = dequant_fp4_scalar((packed >>  8) & 0xFu);
    out[3] = dequant_fp4_scalar((packed >> 12) & 0xFu);
    out[4] = dequant_fp4_scalar((packed >> 16) & 0xFu);
    out[5] = dequant_fp4_scalar((packed >> 20) & 0xFu);
    out[6] = dequant_fp4_scalar((packed >> 24) & 0xFu);
    out[7] = dequant_fp4_scalar((packed >> 28) & 0xFu);
}

inline void dequant_u4x8(uint32_t packed, half scale, half zero_point,
                          thread half4 &out_lo, thread half4 &out_hi) {
    uint32_t n0_biased = (packed & LO_NIBBLE_MASK) | MAGIC_BIAS_U32;
    half2 n0_pair = as_type<half2>(n0_biased) - as_type<half2>(MAGIC_BIAS_U32);
    uint32_t n1_biased = ((packed >> 4u) & LO_NIBBLE_MASK) | MAGIC_BIAS_U32;
    half2 n1_pair = as_type<half2>(n1_biased) - as_type<half2>(MAGIC_BIAS_U32);
    uint32_t n2_biased = ((packed >> 8u) & LO_NIBBLE_MASK) | MAGIC_BIAS_U32;
    half2 n2_pair = as_type<half2>(n2_biased) - as_type<half2>(MAGIC_BIAS_U32);
    uint32_t n3_biased = ((packed >> 12u) & LO_NIBBLE_MASK) | MAGIC_BIAS_U32;
    half2 n3_pair = as_type<half2>(n3_biased) - as_type<half2>(MAGIC_BIAS_U32);
    out_lo = half4(n0_pair.x, n1_pair.x, n2_pair.x, n3_pair.x);
    out_hi = half4(n0_pair.y, n1_pair.y, n2_pair.y, n3_pair.y);
    out_lo = (out_lo - zero_point) * scale;
    out_hi = (out_hi - zero_point) * scale;
}
"""

_DEQUANT_FP4_BULK_SOURCE = """
    uint n_idx = gid.x;
    uint k_block = gid.y;
    uint k_start = k_block * 8u;
    if (n_idx >= N || k_start >= K) return;

    uint packed_idx = k_block * N + n_idx;
    uint32_t word = packed[packed_idx];

    uint group_idx = k_start / GROUP_SIZE;
    half scale = scales[group_idx * N + n_idx];

    half vals[8];
    dequant_fp4_x8(word, vals);

    uint k_remain = min(8u, K - k_start);
    uint out_base = k_start * N + n_idx;
    for (uint i = 0; i < k_remain; i++) {
        output[out_base + i * N] = vals[i] * scale;
    }
"""

_DEQUANT_INT4_BULK_SOURCE = """
    uint n_idx = gid.x;
    uint k_block = gid.y;
    uint k_start = k_block * 8u;
    if (n_idx >= N || k_start >= K) return;

    uint packed_idx = k_block * N + n_idx;
    uint32_t word = packed[packed_idx];

    uint group_idx = k_start / GROUP_SIZE;
    uint param_idx = group_idx * N + n_idx;
    half scale = scales[param_idx];
    half zero_point = zeros[param_idx];

    half4 lo, hi;
    dequant_u4x8(word, scale, zero_point, lo, hi);

    uint k_remain = min(8u, K - k_start);
    uint out_base = k_start * N + n_idx;
    if (k_remain >= 8u) {
        output[out_base + 0u * N] = lo.x;
        output[out_base + 1u * N] = lo.y;
        output[out_base + 2u * N] = lo.z;
        output[out_base + 3u * N] = lo.w;
        output[out_base + 4u * N] = hi.x;
        output[out_base + 5u * N] = hi.y;
        output[out_base + 6u * N] = hi.z;
        output[out_base + 7u * N] = hi.w;
    } else {
        half all_vals[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        for (uint i = 0; i < k_remain; i++) {
            output[out_base + i * N] = all_vals[i];
        }
    }
"""

_dequant_fp4_bulk_kernel: object | None = None
_dequant_int4_bulk_kernel: object | None = None


def _get_dequant_fp4_bulk_kernel() -> object:
    global _dequant_fp4_bulk_kernel
    if _dequant_fp4_bulk_kernel is None:
        _dequant_fp4_bulk_kernel = mx.fast.metal_kernel(
            name="dequant_fp4_bulk",
            input_names=["packed", "scales"],
            output_names=["output"],
            header=_BULK_DEQUANT_HEADER,
            source=_DEQUANT_FP4_BULK_SOURCE,
            ensure_row_contiguous=True,
        )
    return _dequant_fp4_bulk_kernel


def _get_dequant_int4_bulk_kernel() -> object:
    global _dequant_int4_bulk_kernel
    if _dequant_int4_bulk_kernel is None:
        _dequant_int4_bulk_kernel = mx.fast.metal_kernel(
            name="dequant_int4_bulk",
            input_names=["packed", "scales", "zeros"],
            output_names=["output"],
            header=_BULK_DEQUANT_HEADER,
            source=_DEQUANT_INT4_BULK_SOURCE,
            ensure_row_contiguous=True,
        )
    return _dequant_int4_bulk_kernel


def dequant_fp4_bulk(
    packed: mx.array,
    scales: mx.array,
    K: int,
    group_size: int = 128,
) -> mx.array:
    """
    Bulk FP4 dequantization: [K/8, N] packed -> [K, N] FP16.

    Dispatches the dequant_fp4_bulk Metal kernel to fully dequantize a packed
    FP4 weight matrix. Use this when the dequantized weights are needed outside
    a GEMM context (debugging, element-wise ops, accuracy checks).

    Args:
        packed: uint32 array [K/8, N] with 8 packed FP4 nibbles per element.
        scales: float16 array [K/group_size, N] with per-group scale factors.
        K: Number of rows in the output (reduction dimension).
        group_size: Elements per quantization group along K. Default: 128.

    Returns:
        float16 array [K, N] with dequantized weights.
    """
    k_blocks, N = packed.shape
    assert k_blocks == (K + 7) // 8, (
        f"packed.shape[0]={k_blocks} inconsistent with K={K} (expected {(K + 7) // 8})"
    )

    kernel = _get_dequant_fp4_bulk_kernel()
    outputs = kernel(
        inputs=[packed, scales],
        template=[("K", K), ("N", N), ("GROUP_SIZE", group_size)],
        grid=(N, k_blocks, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(K, N)],
        output_dtypes=[mx.float16],
        init_value=0.0,
    )
    return outputs[0]


def dequant_int4_bulk(
    packed: mx.array,
    scales: mx.array,
    zeros: mx.array,
    K: int,
    group_size: int = 128,
) -> mx.array:
    """
    Bulk INT4 dequantization with zero points: [K/8, N] packed -> [K, N] FP16.

    Dispatches the dequant_int4_bulk Metal kernel for asymmetric INT4
    dequantization: output = (uint4_val - zero_point) * scale.

    Args:
        packed: uint32 array [K/8, N] with 8 packed INT4 nibbles per element.
        scales: float16 array [K/group_size, N] with per-group scale factors.
        zeros: float16 array [K/group_size, N] with per-group zero points.
        K: Number of rows in the output (reduction dimension).
        group_size: Elements per quantization group along K. Default: 128.

    Returns:
        float16 array [K, N] with dequantized weights.
    """
    k_blocks, N = packed.shape
    assert k_blocks == (K + 7) // 8, (
        f"packed.shape[0]={k_blocks} inconsistent with K={K} (expected {(K + 7) // 8})"
    )

    kernel = _get_dequant_int4_bulk_kernel()
    outputs = kernel(
        inputs=[packed, scales, zeros],
        template=[("K", K), ("N", N), ("GROUP_SIZE", group_size)],
        grid=(N, k_blocks, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(K, N)],
        output_dtypes=[mx.float16],
        init_value=0.0,
    )
    return outputs[0]


# ---------------------------------------------------------------------------
# Benchmark Utilities
# ---------------------------------------------------------------------------


def benchmark_against_native(
    M: int = 1,
    K: int = 4096,
    N: int = 4096,
    group_size: int = 32,
    warmup: int = 10,
    iterations: int = 100,
) -> dict[str, float]:
    """
    Benchmark Metal Marlin FP4 GEMM against MLX native quantized_matmul.

    Compares:
    1. Metal Marlin (FP4 E2M1, fused dequant-GEMM)
    2. MLX native (affine 4-bit, quantized_matmul)
    3. MLX native (nvfp4, quantized_matmul)

    Args:
        M: Batch/sequence length (rows of activation).
        K: Input features.
        N: Output features.
        group_size: Quantization group size.
        warmup: Warmup iterations.
        iterations: Timed iterations.

    Returns:
        Dict with timing results in milliseconds.
    """
    import numpy as np

    print(f"\nBenchmark: M={M}, K={K}, N={N}, group_size={group_size}")
    print(f"  Warmup: {warmup}, Iterations: {iterations}")
    print("-" * 60)

    # Generate random weights and input
    x = mx.random.normal(shape=(M, K), dtype=mx.float16)
    w = mx.random.normal(shape=(N, K), dtype=mx.float16)
    mx.eval(x, w)

    results: dict[str, float] = {}

    # --- Metal Marlin FP4 ---
    w_packed, fp4_scales = pack_fp4_weights(w, group_size=group_size)
    mx.eval(w_packed, fp4_scales)

    # Warmup
    for _ in range(warmup):
        out = quantized_linear(x, w_packed, fp4_scales, group_size)
        mx.eval(out)

    # Timed
    start = time.perf_counter()
    for _ in range(iterations):
        out = quantized_linear(x, w_packed, fp4_scales, group_size)
        mx.eval(out)
    elapsed = (time.perf_counter() - start) / iterations * 1000
    results["marlin_fp4_ms"] = elapsed
    print(f"  Metal Marlin FP4:     {elapsed:.4f} ms")

    # --- MLX native affine 4-bit ---
    w_q_aff, scales_aff, biases_aff = mx.quantize(w, group_size=group_size, bits=4)
    mx.eval(w_q_aff, scales_aff, biases_aff)

    for _ in range(warmup):
        out_aff = mx.quantized_matmul(
            x,
            w_q_aff,
            scales=scales_aff,
            biases=biases_aff,
            transpose=True,
            group_size=group_size,
            bits=4,
        )
        mx.eval(out_aff)

    start = time.perf_counter()
    for _ in range(iterations):
        out_aff = mx.quantized_matmul(
            x,
            w_q_aff,
            scales=scales_aff,
            biases=biases_aff,
            transpose=True,
            group_size=group_size,
            bits=4,
        )
        mx.eval(out_aff)
    elapsed = (time.perf_counter() - start) / iterations * 1000
    results["mlx_affine4_ms"] = elapsed
    print(f"  MLX affine 4-bit:     {elapsed:.4f} ms")

    # --- MLX native NVFP4 ---
    try:
        w_q_nv, scales_nv, _ = mx.quantize(w, group_size=16, bits=4, mode="nvfp4")
        mx.eval(w_q_nv, scales_nv)

        for _ in range(warmup):
            out_nv = mx.quantized_matmul(
                x,
                w_q_nv,
                scales=scales_nv,
                transpose=True,
                group_size=16,
                bits=4,
                mode="nvfp4",
            )
            mx.eval(out_nv)

        start = time.perf_counter()
        for _ in range(iterations):
            out_nv = mx.quantized_matmul(
                x,
                w_q_nv,
                scales=scales_nv,
                transpose=True,
                group_size=16,
                bits=4,
                mode="nvfp4",
            )
            mx.eval(out_nv)
        elapsed = (time.perf_counter() - start) / iterations * 1000
        results["mlx_nvfp4_ms"] = elapsed
        print(f"  MLX NVFP4:            {elapsed:.4f} ms")
    except Exception as e:
        print(f"  MLX NVFP4:            FAILED ({e})")
        results["mlx_nvfp4_ms"] = float("nan")

    # --- FP16 reference (for speedup calculation) ---
    for _ in range(warmup):
        out_ref = x @ w.T
        mx.eval(out_ref)

    start = time.perf_counter()
    for _ in range(iterations):
        out_ref = x @ w.T
        mx.eval(out_ref)
    elapsed = (time.perf_counter() - start) / iterations * 1000
    results["fp16_ms"] = elapsed
    print(f"  FP16 reference:       {elapsed:.4f} ms")

    # Speedups
    print("\n  Speedups vs FP16:")
    for key in ["marlin_fp4_ms", "mlx_affine4_ms", "mlx_nvfp4_ms"]:
        if key in results and not np.isnan(results[key]):
            speedup = results["fp16_ms"] / results[key]
            print(f"    {key.replace('_ms', ''):20s}: {speedup:.2f}x")

    return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main():
    """Run accuracy test and benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Metal Marlin FP4 GEMM")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--accuracy", action="store_true", help="Run accuracy test")
    parser.add_argument("--M", type=int, default=1, help="M dimension")
    parser.add_argument("--K", type=int, default=4096, help="K dimension")
    parser.add_argument("--N", type=int, default=4096, help="N dimension")
    parser.add_argument("--group-size", type=int, default=32, help="Group size")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()

    if args.accuracy or not args.benchmark:
        print("=== Accuracy Test ===")
        M, K, N = args.M, args.K, args.N

        # Random weight and input
        w = mx.random.normal(shape=(N, K), dtype=mx.float16)
        x = mx.random.normal(shape=(M, K), dtype=mx.float16)
        mx.eval(w, x)

        # Reference: FP16 matmul
        ref = x @ w.T
        mx.eval(ref)

        # Quantize and run
        w_packed, scales = pack_fp4_weights(w, group_size=args.group_size)
        mx.eval(w_packed, scales)

        out = quantized_linear(x, w_packed, scales, group_size=args.group_size)
        mx.eval(out)

        # Compute error
        abs_err = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))
        mx.eval(abs_err)

        import numpy as np

        abs_err_np = np.array(abs_err)
        ref_np = np.array(mx.abs(ref).astype(mx.float32))

        print(f"  Shape: ({M}, {K}) x ({K}, {N}) -> ({M}, {N})")
        print(f"  Max abs error:  {abs_err_np.max():.6f}")
        print(f"  Mean abs error: {abs_err_np.mean():.6f}")
        if ref_np.mean() > 0:
            rel_err = abs_err_np.mean() / ref_np.mean()
            print(f"  Relative error: {rel_err:.4%}")
        print(f"  Output sample:  {np.array(out[0, :8])}")
        print(f"  Ref sample:     {np.array(ref[0, :8])}")

        # Test batched input
        print("\n  Batched input test (batch=4, seq=16):")
        x_batch = mx.random.normal(shape=(4, 16, K), dtype=mx.float16)
        mx.eval(x_batch)
        out_batch = quantized_linear(x_batch, w_packed, scales, group_size=args.group_size)
        mx.eval(out_batch)
        print(f"  Input shape:  {x_batch.shape}")
        print(f"  Output shape: {out_batch.shape}")
        assert out_batch.shape == (4, 16, N), f"Expected (4, 16, {N}), got {out_batch.shape}"
        print("  PASSED")

    if args.benchmark:
        print("\n=== Benchmark ===")
        benchmark_against_native(
            M=args.M,
            K=args.K,
            N=args.N,
            group_size=args.group_size,
            warmup=args.warmup,
            iterations=args.iters,
        )

        # Also test with typical LLM dimensions
        if args.M == 1 and args.K == 4096 and args.N == 4096:
            print("\n--- Typical LLM shapes ---")
            for M, K, N in [(1, 4096, 4096), (1, 4096, 11008), (32, 4096, 4096)]:
                benchmark_against_native(M=M, K=K, N=N, group_size=args.group_size)


if __name__ == "__main__":
    main()
