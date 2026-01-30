#include <metal_stdlib>
using namespace metal;

// ============================================================================
// FP8 E4M3 (NVIDIA) → FP16 Bitwise Dequantization for Apple Metal
// ============================================================================
//
// FP8 E4M3 format: [1 sign][4 exponent (bias=7)][3 mantissa]
//
// Value encoding:
//   Normal (0 < E < 15):  (-1)^S * 2^(E-7) * (1 + M/8)
//   Subnormal (E == 0):   (-1)^S * 2^(-6) * (M/8)
//   NaN (E == 15):        NaN (E4M3 has no infinity; all E=15 are NaN)
//   Zero (E=0, M=0):      +/- 0.0
//
// FP16 format: [1 sign][5 exponent (bias=15)][10 mantissa]
//
// Normal conversion (0 < E < 15):
//   FP16 sign = S
//   FP16 exponent = E - 7 + 15 = E + 8
//   FP16 mantissa = M << 7 (3-bit mantissa left-aligned in 10-bit field)
//
// Subnormal conversion (E == 0, M > 0):
//   Value = (-1)^S * 2^(-6) * (M/8) = (-1)^S * M * 2^(-9)
//   Normalize by counting leading zeros in the 3-bit mantissa:
//     M=1 (0b001): shift=2, normalized: exp=15-9=6,  mant=0   -> val=2^(-9)
//     M=2 (0b010): shift=1, normalized: exp=15-8=7,  mant=0   -> val=2^(-8)
//     M=3 (0b011): shift=1, normalized: exp=15-8=7,  mant=1<<9 -> val=3*2^(-9)
//     M=4 (0b100): shift=0, normalized: exp=15-7=8,  mant=0   -> val=2^(-7)
//     M=5 (0b101): shift=0, normalized: exp=15-7=8,  mant=1<<8 -> ...
//     M=6 (0b110): shift=0, normalized: exp=15-7=8,  mant=1<<9 -> ...
//     M=7 (0b111): shift=0, normalized: exp=15-7=8,  mant=3<<8 -> ...
//   But for 3-bit mantissa, we can handle all 7 subnormal cases with:
//     clz3 = number of leading zeros in 3-bit field (2,1,1,0,0,0,0 for M=1..7)
//     fp16_exp = 6 - clz3 + 1 = 7 - clz3
//     fp16_mant = (M << (clz3 + 1)) & 0x7 then << 7
//
// The E4M3 subnormals map to values {0.001953125, 0.00390625, 0.005859375,
// 0.0078125, 0.009765625, 0.01171875, 0.013671875} which are all exactly
// representable in FP16.
//
// NaN handling:
//   E=15 in E4M3 means NaN regardless of mantissa (no infinity in E4M3).
//   Map to FP16 quiet NaN: 0x7E00 (positive) or 0xFE00 (negative).
//
// ============================================================================

// ============================================================================
// Core FP8 E4M3 → FP16 primitive
// ============================================================================

/// Branchless dequantize a single FP8 E4M3 value to half precision.
/// Handles all cases: zero, subnormal, normal, and NaN.
///
/// @param fp8_val  8-bit FP8 E4M3 value
/// @return         Equivalent half-precision value
inline half dequant_fp8_e4m3(uint fp8_val) {
    uint S = (fp8_val >> 7) & 1u;
    uint E = (fp8_val >> 3) & 0xFu;
    uint M = fp8_val & 0x7u;

    // --- NaN: E == 15 (E4M3 has no inf, all exp-all-ones = NaN) ---
    if (E == 15u) {
        return as_type<half>(ushort(0x7E00u | (S << 15)));
    }

    // --- Zero: E == 0, M == 0 ---
    if (E == 0u && M == 0u) {
        return as_type<half>(ushort(S << 15));
    }

    // --- Subnormal: E == 0, M > 0 ---
    if (E == 0u) {
        // Value = (-1)^S * M * 2^(-9)
        // Normalize the 3-bit subnormal mantissa.
        // Count leading zeros in 3-bit field:
        //   M: 1->2, 2->1, 3->1, 4->0, 5->0, 6->0, 7->0
        uint clz3 = select(0u, select(1u, 2u, M < 2u), M < 4u);
        // Normalized exponent: 2^(-9) * 2^(2-clz3) adjusted for FP16 bias
        //   = 2^(-9 + 2 - clz3 + 1) = base is exp = -6 + (2 - clz3)
        //   In FP16: biased_exp = 15 + (-9) + (2 - clz3) + 1 = 9 - clz3
        // Actually: subnormal value = M * 2^(-9)
        //   For M with (2-clz3) leading bit position:
        //   biased_exp = 15 - 9 + (2 - clz3) = 8 - clz3
        // Let's derive directly:
        //   M=1: val=2^(-9),  FP16 = exp=6 (15-9), mant=0
        //   M=2: val=2^(-8),  FP16 = exp=7 (15-8), mant=0
        //   M=3: val=3*2^(-9), FP16 = exp=7, mant=(1<<9)=512  [1.1 * 2^(-8)]
        //   M=4: val=2^(-7),  FP16 = exp=8 (15-7), mant=0
        //   M=5: val=5*2^(-9), FP16 = exp=8, mant=(1<<8)=256  [1.01 * 2^(-7)]
        //   M=6: val=6*2^(-9)=3*2^(-8), FP16 = exp=8, mant=(1<<9)=512  [1.1 * 2^(-7)]
        //   M=7: val=7*2^(-9), FP16 = exp=8, mant=(3<<8)=768  [1.11 * 2^(-7)]
        //
        // Pattern: biased_exp = 6 + (2 - clz3) = 8 - clz3
        //          mant10 = ((M << (clz3 + 1)) & 0x7) << 7
        uint fp16_exp = 8u - clz3;
        uint normalized_m = (M << (clz3 + 1u)) & 0x7u;
        uint fp16_mant = normalized_m << 7;
        return as_type<half>(ushort((S << 15) | (fp16_exp << 10) | fp16_mant));
    }

    // --- Normal: 0 < E < 15 ---
    // FP16 exponent = E + 8 (rebias from 7 to 15)
    // FP16 mantissa = M << 7 (left-align 3 bits into 10-bit field)
    uint fp16_exp = E + 8u;
    uint fp16_mant = M << 7;
    return as_type<half>(ushort((S << 15) | (fp16_exp << 10) | fp16_mant));
}

// ============================================================================
// Vectorized FP8 dequantization: 4 values from one uint32
// ============================================================================

/// Dequantize 4 FP8 E4M3 values packed in a uint32 to a half4.
///
/// Packing: byte layout [b3:b2:b1:b0] where b0 is bits [7:0].
/// Each byte is one FP8 E4M3 value. This matches the natural memory layout
/// when casting a contiguous FP8 buffer to uint32*.
///
/// @param packed  4 packed FP8 E4M3 bytes in a uint32
/// @return        4 dequantized half-precision values
inline half4 dequant_fp8_e4m3_x4(uint32_t packed) {
    return half4(
        dequant_fp8_e4m3((packed >>  0) & 0xFFu),
        dequant_fp8_e4m3((packed >>  8) & 0xFFu),
        dequant_fp8_e4m3((packed >> 16) & 0xFFu),
        dequant_fp8_e4m3((packed >> 24) & 0xFFu)
    );
}

/// Dequantize 4 FP8 E4M3 values with fused scale multiplication.
/// NOTE: Uses float to work around Metal compiler half-precision bug.
///
/// @param packed  4 packed FP8 E4M3 bytes
/// @param scale   Per-group (or per-tensor) scale factor
/// @return        4 dequantized, scaled half-precision values
inline half4 dequant_fp8_e4m3_x4_scaled(uint32_t packed, half scale) {
    float4 raw = float4(dequant_fp8_e4m3_x4(packed));
    return half4(raw * (float)scale);
}

// ============================================================================
// Vectorized FP8 dequantization: 8 values from two uint32s
// ============================================================================

/// Dequantize 8 FP8 E4M3 values (two uint32s) to two half4 vectors.
/// Suitable for GEMM tile loops operating on 8-element slices.
/// NOTE: Uses float to work around Metal compiler half-precision bug.
///
/// @param packed0  First 4 FP8 values (bytes 0-3)
/// @param packed1  Second 4 FP8 values (bytes 4-7)
/// @param scale    Per-group scale factor
/// @param out_lo   Output: dequantized values 0-3, scaled
/// @param out_hi   Output: dequantized values 4-7, scaled
inline void dequant_fp8_e4m3_x8_fused(uint32_t packed0,
                                       uint32_t packed1,
                                       half scale,
                                       thread half4 &out_lo,
                                       thread half4 &out_hi) {
    float fscale = (float)scale;
    float4 raw_lo = float4(dequant_fp8_e4m3_x4(packed0));
    float4 raw_hi = float4(dequant_fp8_e4m3_x4(packed1));
    out_lo = half4(raw_lo * fscale);
    out_hi = half4(raw_hi * fscale);
}

// ============================================================================
// Compute kernel: Bulk FP8 E4M3 → FP16 dequantization
// ============================================================================

/// Kernel that dequantizes a buffer of packed FP8 E4M3 weights to FP16.
/// Each uint32 contains 4 FP8 values (byte-packed).
///
/// @param packed_weights  Input: N/4 uint32 values
/// @param scales          Per-group scale factors
/// @param output          Output: N half values
/// @param num_elements    Total number of FP8 values (N)
/// @param group_size      Number of elements per quantization group
kernel void dequant_fp8_kernel(
    device const uint32_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device half *output                  [[buffer(2)]],
    constant uint32_t &num_elements      [[buffer(3)]],
    constant uint32_t &group_size        [[buffer(4)]],
    uint tid                             [[thread_position_in_grid]])
{
    // Each thread processes one uint32 = 4 FP8 values
    uint base_idx = tid * 4u;
    if (base_idx >= num_elements) return;

    uint32_t packed = packed_weights[tid];

    // Determine quantization group
    uint group_idx = base_idx / group_size;
    half scale = scales[group_idx];

    half4 vals = dequant_fp8_e4m3_x4_scaled(packed, scale);

    // Write output (handle boundary for last group)
    uint remaining = min(4u, num_elements - base_idx);
    if (remaining >= 4u) {
        output[base_idx + 0u] = vals.x;
        output[base_idx + 1u] = vals.y;
        output[base_idx + 2u] = vals.z;
        output[base_idx + 3u] = vals.w;
    } else {
        for (uint i = 0u; i < remaining; i++) {
            output[base_idx + i] = vals[i];
        }
    }
}

/// High-throughput variant using half4 vectorized stores.
/// Requires num_elements is a multiple of 4 and output is half4-aligned.
kernel void dequant_fp8_aligned_kernel(
    device const uint32_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device half4 *output                 [[buffer(2)]],
    constant uint32_t &num_packed        [[buffer(3)]],
    constant uint32_t &group_size        [[buffer(4)]],
    uint tid                             [[thread_position_in_grid]])
{
    if (tid >= num_packed) return;

    uint32_t packed = packed_weights[tid];
    uint base_idx = tid * 4u;
    uint group_idx = base_idx / group_size;
    half scale = scales[group_idx];

    output[tid] = dequant_fp8_e4m3_x4_scaled(packed, scale);
}

// ============================================================================
// 2D kernel variant for weight matrices
// ============================================================================

/// Dequantize a 2D FP8 weight matrix [K, N] stored packed as [K/4, N] uint32s.
///
/// @param packed_weights  [K/4, N] packed FP8 weights
/// @param scales          [K/group_size, N] per-group scales
/// @param output          [K, N] dequantized FP16 output
/// @param K               Reduction dimension
/// @param N               Output columns
/// @param group_size      Elements per quantization group
kernel void dequant_fp8_2d_kernel(
    device const uint32_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device half *output                  [[buffer(2)]],
    constant uint32_t &K                 [[buffer(3)]],
    constant uint32_t &N                 [[buffer(4)]],
    constant uint32_t &group_size        [[buffer(5)]],
    uint2 gid                            [[thread_position_in_grid]]
) {
    // gid.x = column index (along N)
    // gid.y = block index along K (each block = 4 FP8 values)
    uint n_idx = gid.x;
    uint k_block = gid.y;

    if (n_idx >= N || k_block * 4u >= K) return;

    uint packed_idx = k_block * N + n_idx;
    uint32_t packed = packed_weights[packed_idx];

    uint k_start = k_block * 4u;
    uint group_idx = k_start / group_size;
    half scale = scales[group_idx * N + n_idx];

    half4 vals = dequant_fp8_e4m3_x4(packed);

    // Write output (K-major layout)
    uint out_base = k_start * N + n_idx;
    uint k_remain = min(4u, K - k_start);
    for (uint i = 0; i < k_remain; i++) {
        output[out_base + i * N] = vals[i] * scale;
    }
}

// ============================================================================
// Unit test kernels
// ============================================================================

/// Test kernel: dequantize all 256 FP8 E4M3 codes without scaling.
/// Writes 256 half values for host-side verification against reference.
kernel void test_fp8_all_codes(
    device half *output [[buffer(0)]],
    uint tid            [[thread_position_in_grid]]
) {
    if (tid >= 256u) return;
    output[tid] = dequant_fp8_e4m3(tid);
}

/// Test kernel: dequantize a packed uint32 (4 FP8 values) with a given scale.
kernel void test_fp8_packed_scaled(
    device const uint32_t *packed_input [[buffer(0)]],
    device const half *scale           [[buffer(1)]],
    device half *output                [[buffer(2)]],
    uint tid                           [[thread_position_in_grid]]
) {
    if (tid > 0u) return;

    half4 vals = dequant_fp8_e4m3_x4_scaled(packed_input[0], scale[0]);
    output[0] = vals.x;
    output[1] = vals.y;
    output[2] = vals.z;
    output[3] = vals.w;
}

// ============================================================================
// FP8 E5M2 → FP16 Bitwise Dequantization
// ============================================================================
//
// FP8 E5M2 format: [1 sign][5 exponent (bias=15)][2 mantissa]
// FP16 format:     [1 sign][5 exponent (bias=15)][10 mantissa]
//
// Since E5M2 and FP16 share the same exponent width (5 bits) and bias (15),
// conversion is a pure bit-rearrangement: the sign and exponent fields
// transfer directly, and the 2-bit mantissa left-aligns into FP16's 10-bit
// mantissa field (shift left by 8).
//
// This handles all IEEE 754 special cases correctly:
//   Normal (0 < E < 31):  sign/exp copy, mantissa << 8
//   Subnormal (E=0, M>0): same, mantissa << 8 (both formats use bias=15)
//   Zero (E=0, M=0):      +/- 0.0 preserved
//   Infinity (E=31, M=0): +/- Inf preserved (E5M2 has infinity, unlike E4M3)
//   NaN (E=31, M!=0):     NaN preserved (payload left-aligned)
//
// No branches, no lookup tables, no subnormal normalization needed.
// The shared exponent format makes this a trivial field extension.
// ============================================================================

// ============================================================================
// Core FP8 E5M2 → FP16 primitive
// ============================================================================

/// Dequantize a single FP8 E5M2 value to half precision.
///
/// Since E5M2 and FP16 share the same exponent width and bias, this is a
/// direct bit-field rearrangement with no arithmetic. The 2-bit mantissa
/// is placed into the high 2 bits of FP16's 10-bit mantissa field.
///
/// @param fp8_val  8-bit FP8 E5M2 value
/// @return         Equivalent half-precision value
inline half dequant_fp8_e5m2(uint fp8_val) {
    uint S = (fp8_val >> 7) & 1u;
    uint E = (fp8_val >> 2) & 0x1Fu;
    uint M = fp8_val & 0x3u;

    // Direct field placement: same bias, same exponent width.
    // Mantissa: 2 bits → 10 bits, left-aligned (shift by 8).
    uint mant16 = M << 8;
    return as_type<half>(ushort((S << 15) | (E << 10) | mant16));
}

/// Dequantize FP8 E5M2 with per-element scale factor.
inline half dequant_fp8_e5m2_scaled(uint fp8_val, half scale) {
    return dequant_fp8_e5m2(fp8_val) * scale;
}

// ============================================================================
// Vectorized FP8 E5M2 dequantization
// ============================================================================

/// Dequantize 4 FP8 E5M2 values packed in a uint32 to a half4.
///
/// @param packed  4 packed FP8 E5M2 bytes in a uint32
/// @return        4 dequantized half-precision values
inline half4 dequant_fp8_e5m2_x4(uint32_t packed) {
    return half4(
        dequant_fp8_e5m2((packed >>  0) & 0xFFu),
        dequant_fp8_e5m2((packed >>  8) & 0xFFu),
        dequant_fp8_e5m2((packed >> 16) & 0xFFu),
        dequant_fp8_e5m2((packed >> 24) & 0xFFu)
    );
}

/// Dequantize 4 FP8 E5M2 values with fused scale multiplication.
inline half4 dequant_fp8_e5m2_x4_scaled(uint32_t packed, half scale) {
    return dequant_fp8_e5m2_x4(packed) * scale;
}

/// Dequantize 8 FP8 E5M2 values (two uint32s) to two half4 vectors.
inline void dequant_fp8_e5m2_x8_fused(uint32_t packed0,
                                       uint32_t packed1,
                                       half scale,
                                       thread half4 &out_lo,
                                       thread half4 &out_hi) {
    out_lo = dequant_fp8_e5m2_x4(packed0) * scale;
    out_hi = dequant_fp8_e5m2_x4(packed1) * scale;
}

// ============================================================================
// FP8 E5M2 Compute kernels
// ============================================================================

/// Bulk FP8 E5M2 → FP16 dequantization kernel.
kernel void dequant_fp8_e5m2_kernel(
    device const uint32_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device half *output                  [[buffer(2)]],
    constant uint32_t &num_elements      [[buffer(3)]],
    constant uint32_t &group_size        [[buffer(4)]],
    uint tid                             [[thread_position_in_grid]])
{
    uint base_idx = tid * 4u;
    if (base_idx >= num_elements) return;

    uint32_t packed = packed_weights[tid];
    uint group_idx = base_idx / group_size;
    half scale = scales[group_idx];

    half4 vals = dequant_fp8_e5m2_x4_scaled(packed, scale);

    uint remaining = min(4u, num_elements - base_idx);
    if (remaining >= 4u) {
        output[base_idx + 0u] = vals.x;
        output[base_idx + 1u] = vals.y;
        output[base_idx + 2u] = vals.z;
        output[base_idx + 3u] = vals.w;
    } else {
        for (uint i = 0u; i < remaining; i++) {
            output[base_idx + i] = vals[i];
        }
    }
}

/// Aligned variant with half4 vector stores.
kernel void dequant_fp8_e5m2_aligned_kernel(
    device const uint32_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device half4 *output                 [[buffer(2)]],
    constant uint32_t &num_packed        [[buffer(3)]],
    constant uint32_t &group_size        [[buffer(4)]],
    uint tid                             [[thread_position_in_grid]])
{
    if (tid >= num_packed) return;

    uint32_t packed = packed_weights[tid];
    uint base_idx = tid * 4u;
    uint group_idx = base_idx / group_size;
    half scale = scales[group_idx];

    output[tid] = dequant_fp8_e5m2_x4_scaled(packed, scale);
}

// ============================================================================
// FP8 E5M2 unit test kernels
// ============================================================================

/// Test kernel: dequantize all 256 FP8 E5M2 codes without scaling.
/// Writes 256 half values for host-side verification.
kernel void test_fp8_e5m2_all_codes(
    device half *output [[buffer(0)]],
    uint tid            [[thread_position_in_grid]]
) {
    if (tid >= 256u) return;
    output[tid] = dequant_fp8_e5m2(tid);
}

/// Test kernel: dequantize a packed uint32 (4 E5M2 values) with a given scale.
kernel void test_fp8_e5m2_packed_scaled(
    device const uint32_t *packed_input [[buffer(0)]],
    device const half *scale           [[buffer(1)]],
    device half *output                [[buffer(2)]],
    uint tid                           [[thread_position_in_grid]]
) {
    if (tid > 0u) return;

    half4 vals = dequant_fp8_e5m2_x4_scaled(packed_input[0], scale[0]);
    output[0] = vals.x;
    output[1] = vals.y;
    output[2] = vals.z;
    output[3] = vals.w;
}
