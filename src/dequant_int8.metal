#include <metal_stdlib>
using namespace metal;

// ============================================================================
// INT8 (S8) → FP16 Dequantization for Apple Metal (W8A16 Inference)
// ============================================================================
//
// INT8 quantization packs 4 signed bytes per uint32. Unlike INT4 which
// requires magic bias tricks to avoid branches, INT8 dequantization is
// straightforward: extract each byte, sign-extend, and convert to FP16.
//
// Packing format: uint32 = [byte3:byte2:byte1:byte0]
//   byte0 = bits [7:0]    (least significant)
//   byte1 = bits [15:8]
//   byte2 = bits [23:16]
//   byte3 = bits [31:24]  (most significant)
//
// Two quantization modes:
//   Symmetric:   dequant = int8_value * scale
//   Asymmetric:  dequant = (int8_value - zero_point) * scale
//
// Performance notes:
//   - No magic bias or LUT needed; direct int-to-float conversion
//   - Metal's half(int) conversion is a single instruction on Apple Silicon
//   - Processing 4 values per uint32 gives good ALU utilization
//   - Wider variants (x8, x16) amortize scale/zero loads
// ============================================================================

// ============================================================================
// Byte extraction helpers
// ============================================================================

/// Extract byte at position [0-3] from a packed uint32 and sign-extend to int8.
/// Metal's char type is signed 8-bit, so the cast handles sign extension.
inline int8_t extract_s8(uint32_t packed, uint pos) {
    return int8_t((packed >> (pos * 8u)) & 0xFFu);
}

// ============================================================================
// Core dequantization primitives: Symmetric (scale only)
// ============================================================================

/// Dequantize 4 signed INT8 values packed in a uint32 to half4.
/// Symmetric quantization: result = int8_value * scale
/// NOTE: Uses float to work around Metal compiler half-precision bug.
///
/// @param packed  4 packed signed bytes [b3:b2:b1:b0]
/// @param scale   Per-group scale factor
/// @param out     Output: 4 dequantized FP16 values
inline void dequant_s8x4(uint32_t packed,
                          half scale,
                          thread half4 &out) {
    int8_t b0 = extract_s8(packed, 0);
    int8_t b1 = extract_s8(packed, 1);
    int8_t b2 = extract_s8(packed, 2);
    int8_t b3 = extract_s8(packed, 3);

    float fscale = (float)scale;
    out = half4(float4((float)b0, (float)b1, (float)b2, (float)b3) * fscale);
}

/// Dequantize 8 signed INT8 values from 2 packed uint32s to two half4s.
/// Symmetric quantization: result = int8_value * scale
///
/// @param packed_lo  First 4 packed bytes [b3:b2:b1:b0]
/// @param packed_hi  Next 4 packed bytes [b7:b6:b5:b4]
/// @param scale      Per-group scale factor
/// @param out_lo     Output: values 0-3
/// @param out_hi     Output: values 4-7
inline void dequant_s8x8(uint32_t packed_lo,
                          uint32_t packed_hi,
                          half scale,
                          thread half4 &out_lo,
                          thread half4 &out_hi) {
    dequant_s8x4(packed_lo, scale, out_lo);
    dequant_s8x4(packed_hi, scale, out_hi);
}

/// Dequantize 16 signed INT8 values from 4 packed uint32s to four half4s.
/// Maximizes ALU utilization per scale factor load.
///
/// @param p0-p3     4 packed uint32 values (16 bytes total)
/// @param scale     Per-group scale factor
/// @param out0-out3 Output: 4 half4 vectors (16 values)
inline void dequant_s8x16(uint32_t p0, uint32_t p1,
                           uint32_t p2, uint32_t p3,
                           half scale,
                           thread half4 &out0,
                           thread half4 &out1,
                           thread half4 &out2,
                           thread half4 &out3) {
    dequant_s8x4(p0, scale, out0);
    dequant_s8x4(p1, scale, out1);
    dequant_s8x4(p2, scale, out2);
    dequant_s8x4(p3, scale, out3);
}

// ============================================================================
// Core dequantization primitives: Asymmetric (scale + zero_point)
// ============================================================================

/// Dequantize 4 signed INT8 values with asymmetric quantization.
/// result = (int8_value - zero_point) * scale
/// NOTE: Uses float to work around Metal compiler half-precision bug.
///
/// The zero_point is in the INT8 domain (integer), passed as half for
/// fused subtract-multiply without extra int-to-float conversion.
///
/// @param packed      4 packed signed bytes
/// @param scale       Per-group scale factor
/// @param zero_point  Per-group zero point (as FP16)
/// @param out         Output: 4 dequantized values
inline void dequant_s8x4_asym(uint32_t packed,
                               half scale,
                               half zero_point,
                               thread half4 &out) {
    int8_t b0 = extract_s8(packed, 0);
    int8_t b1 = extract_s8(packed, 1);
    int8_t b2 = extract_s8(packed, 2);
    int8_t b3 = extract_s8(packed, 3);

    float fscale = (float)scale;
    float fzero = (float)zero_point;
    out = half4((float4((float)b0, (float)b1, (float)b2, (float)b3) - fzero) * fscale);
}

/// Dequantize 8 signed INT8 values with asymmetric quantization.
///
/// @param packed_lo   First 4 packed bytes
/// @param packed_hi   Next 4 packed bytes
/// @param scale       Per-group scale factor
/// @param zero_point  Per-group zero point (as FP16)
/// @param out_lo      Output: values 0-3
/// @param out_hi      Output: values 4-7
inline void dequant_s8x8_asym(uint32_t packed_lo,
                               uint32_t packed_hi,
                               half scale,
                               half zero_point,
                               thread half4 &out_lo,
                               thread half4 &out_hi) {
    dequant_s8x4_asym(packed_lo, scale, zero_point, out_lo);
    dequant_s8x4_asym(packed_hi, scale, zero_point, out_hi);
}

/// Dequantize 16 signed INT8 values with asymmetric quantization.
inline void dequant_s8x16_asym(uint32_t p0, uint32_t p1,
                                uint32_t p2, uint32_t p3,
                                half scale,
                                half zero_point,
                                thread half4 &out0,
                                thread half4 &out1,
                                thread half4 &out2,
                                thread half4 &out3) {
    dequant_s8x4_asym(p0, scale, zero_point, out0);
    dequant_s8x4_asym(p1, scale, zero_point, out1);
    dequant_s8x4_asym(p2, scale, zero_point, out2);
    dequant_s8x4_asym(p3, scale, zero_point, out3);
}

// ============================================================================
// Unsigned INT8 variants (for activations or unsigned weight schemes)
// ============================================================================

/// Dequantize 4 unsigned INT8 values (symmetric).
/// result = uint8_value * scale
inline void dequant_u8x4(uint32_t packed,
                          half scale,
                          thread half4 &out) {
    uint8_t b0 = uint8_t(packed & 0xFFu);
    uint8_t b1 = uint8_t((packed >> 8u) & 0xFFu);
    uint8_t b2 = uint8_t((packed >> 16u) & 0xFFu);
    uint8_t b3 = uint8_t((packed >> 24u) & 0xFFu);

    out = half4(half(b0), half(b1), half(b2), half(b3)) * scale;
}

/// Dequantize 4 unsigned INT8 values (asymmetric).
/// result = (uint8_value - zero_point) * scale
inline void dequant_u8x4_asym(uint32_t packed,
                               half scale,
                               half zero_point,
                               thread half4 &out) {
    uint8_t b0 = uint8_t(packed & 0xFFu);
    uint8_t b1 = uint8_t((packed >> 8u) & 0xFFu);
    uint8_t b2 = uint8_t((packed >> 16u) & 0xFFu);
    uint8_t b3 = uint8_t((packed >> 24u) & 0xFFu);

    out = (half4(half(b0), half(b1), half(b2), half(b3)) - zero_point) * scale;
}

// ============================================================================
// GEMM-fused inline helpers
// ============================================================================

/// Symmetric S8 dequant for use inside GEMM tile loops.
/// Processes one uint32 (4 weights) and returns a single half4.
inline void dequant_s8x4_fused(uint32_t packed,
                                half scale,
                                thread half4 &tile) {
    dequant_s8x4(packed, scale, tile);
}

/// Symmetric S8 dequant: 2 uint32s -> 8 values for GEMM tiles.
inline void dequant_s8x8_fused(uint32_t packed_lo,
                                uint32_t packed_hi,
                                half scale,
                                thread half4 &tile_lo,
                                thread half4 &tile_hi) {
    dequant_s8x8(packed_lo, packed_hi, scale, tile_lo, tile_hi);
}

/// Asymmetric S8 dequant for GEMM tile loops.
inline void dequant_s8x4_asym_fused(uint32_t packed,
                                     half scale,
                                     half zero_point,
                                     thread half4 &tile) {
    dequant_s8x4_asym(packed, scale, zero_point, tile);
}

/// Asymmetric S8 dequant: 2 uint32s -> 8 values for GEMM tiles.
inline void dequant_s8x8_asym_fused(uint32_t packed_lo,
                                     uint32_t packed_hi,
                                     half scale,
                                     half zero_point,
                                     thread half4 &tile_lo,
                                     thread half4 &tile_hi) {
    dequant_s8x8_asym(packed_lo, packed_hi, scale, zero_point, tile_lo, tile_hi);
}

// ============================================================================
// Compute kernel: Bulk INT8 → FP16 dequantization
// ============================================================================

/// Kernel that dequantizes a buffer of packed INT8 weights to FP16.
/// Each uint32 contains 4 INT8 values.
///
/// Supports both symmetric and asymmetric modes via the mode parameter.
///
/// @param packed_weights  Input: N/4 uint32 values
/// @param scales          Per-group scale factors
/// @param zeros           Per-group zero points (ignored if symmetric)
/// @param output          Output: N half values
/// @param num_elements    Total number of INT8 values
/// @param group_size      Elements per quantization group
/// @param mode            0 = symmetric signed, 1 = asymmetric signed,
///                        2 = symmetric unsigned, 3 = asymmetric unsigned
kernel void dequant_int8_kernel(
    device const uint32_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device const half *zeros             [[buffer(2)]],
    device half *output                  [[buffer(3)]],
    constant uint32_t &num_elements      [[buffer(4)]],
    constant uint32_t &group_size        [[buffer(5)]],
    constant uint32_t &mode              [[buffer(6)]],
    uint tid                             [[thread_position_in_grid]])
{
    // Each thread processes one uint32 = 4 INT8 values
    uint base_idx = tid * 4u;
    if (base_idx >= num_elements) return;

    uint32_t packed = packed_weights[tid];

    // Determine quantization group
    uint group_idx = base_idx / group_size;
    half scale = scales[group_idx];
    half zero_point = zeros[group_idx];

    half4 result;
    switch (mode) {
        case 0u: dequant_s8x4(packed, scale, result); break;
        case 1u: dequant_s8x4_asym(packed, scale, zero_point, result); break;
        case 2u: dequant_u8x4(packed, scale, result); break;
        case 3u: dequant_u8x4_asym(packed, scale, zero_point, result); break;
        default: dequant_s8x4(packed, scale, result); break;
    }

    // Write output with boundary check
    uint remaining = min(4u, num_elements - base_idx);
    if (remaining == 4u) {
        output[base_idx + 0u] = result.x;
        output[base_idx + 1u] = result.y;
        output[base_idx + 2u] = result.z;
        output[base_idx + 3u] = result.w;
    } else {
        for (uint i = 0u; i < remaining; i++) {
            output[base_idx + i] = result[i];
        }
    }
}

// ============================================================================
// Optimized kernel: Vectorized half4 stores (8 values per thread)
// ============================================================================

/// High-throughput variant processing 2 uint32s (8 INT8 values) per thread.
/// Requires num_elements is a multiple of 8 and output is half4-aligned.
kernel void dequant_int8_aligned_kernel(
    device const uint32_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device const half *zeros             [[buffer(2)]],
    device half4 *output                 [[buffer(3)]],
    constant uint32_t &num_packed_pairs  [[buffer(4)]],
    constant uint32_t &group_size        [[buffer(5)]],
    constant uint32_t &mode              [[buffer(6)]],
    uint tid                             [[thread_position_in_grid]])
{
    if (tid >= num_packed_pairs) return;

    // Each thread processes 2 consecutive uint32s = 8 INT8 values
    uint packed_idx = tid * 2u;
    uint32_t packed_lo = packed_weights[packed_idx];
    uint32_t packed_hi = packed_weights[packed_idx + 1u];

    uint base_idx = tid * 8u;
    uint group_idx = base_idx / group_size;
    half scale = scales[group_idx];
    half zero_point = zeros[group_idx];

    half4 lo, hi;
    switch (mode) {
        case 0u:
            dequant_s8x8(packed_lo, packed_hi, scale, lo, hi);
            break;
        case 1u:
            dequant_s8x8_asym(packed_lo, packed_hi, scale, zero_point, lo, hi);
            break;
        case 2u:
            dequant_u8x4(packed_lo, scale, lo);
            dequant_u8x4(packed_hi, scale, hi);
            break;
        case 3u:
            dequant_u8x4_asym(packed_lo, scale, zero_point, lo);
            dequant_u8x4_asym(packed_hi, scale, zero_point, hi);
            break;
        default:
            dequant_s8x8(packed_lo, packed_hi, scale, lo, hi);
            break;
    }

    // Two half4 vectorized stores
    output[tid * 2u]     = lo;
    output[tid * 2u + 1u] = hi;
}

// ============================================================================
// 2D matrix dequantization kernel
// ============================================================================

/// Dequantize a 2D weight matrix stored in packed INT8 format.
///   packed_weights[k_block][n] where k_block = K/4
///
/// Each uint32 holds 4 INT8 values along the K dimension.
/// Scales are per-group along K: scales[K/group_size][N]
///
/// @param packed_weights  [K/4, N] packed INT8 weights
/// @param scales          [K/group_size, N] per-group scales
/// @param zeros           [K/group_size, N] per-group zero points
/// @param output          [K, N] dequantized FP16 output
/// @param K               Reduction dimension size
/// @param N               Output columns
/// @param group_size      Elements per quantization group
/// @param mode            Quantization mode (0=sym_s8, 1=asym_s8, 2=sym_u8, 3=asym_u8)
kernel void dequant_int8_2d_kernel(
    device const uint32_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device const half *zeros             [[buffer(2)]],
    device half *output                  [[buffer(3)]],
    constant uint32_t &K                 [[buffer(4)]],
    constant uint32_t &N                 [[buffer(5)]],
    constant uint32_t &group_size        [[buffer(6)]],
    constant uint32_t &mode              [[buffer(7)]],
    uint2 gid                            [[thread_position_in_grid]])
{
    // gid.x = column index (along N)
    // gid.y = block index along K (each block = 4 INT8 values)
    uint n_idx = gid.x;
    uint k_block = gid.y;

    if (n_idx >= N || k_block * 4u >= K) return;

    uint packed_idx = k_block * N + n_idx;
    uint32_t packed = packed_weights[packed_idx];

    uint k_start = k_block * 4u;
    uint group_idx = k_start / group_size;
    half scale = scales[group_idx * N + n_idx];
    half zero_point = zeros[group_idx * N + n_idx];

    half4 result;
    switch (mode) {
        case 0u: dequant_s8x4(packed, scale, result); break;
        case 1u: dequant_s8x4_asym(packed, scale, zero_point, result); break;
        case 2u: dequant_u8x4(packed, scale, result); break;
        case 3u: dequant_u8x4_asym(packed, scale, zero_point, result); break;
        default: dequant_s8x4(packed, scale, result); break;
    }

    // Write output (column-major within K dimension)
    uint out_base = k_start * N + n_idx;
    uint k_remain = min(4u, K - k_start);
    for (uint i = 0u; i < k_remain; i++) {
        output[out_base + i * N] = result[i];
    }
}

// ============================================================================
// Unit test kernels
// ============================================================================

/// Test kernel: dequantize a single packed uint32 (4 INT8 values) with scale.
kernel void test_int8_symmetric(
    device const uint32_t *packed_input [[buffer(0)]],
    device const half *scale           [[buffer(1)]],
    device half *output                [[buffer(2)]],
    uint tid                           [[thread_position_in_grid]])
{
    if (tid > 0u) return;

    half4 result;
    dequant_s8x4(packed_input[0], scale[0], result);

    output[0] = result.x;
    output[1] = result.y;
    output[2] = result.z;
    output[3] = result.w;
}

/// Test kernel: dequantize with asymmetric quantization.
kernel void test_int8_asymmetric(
    device const uint32_t *packed_input [[buffer(0)]],
    device const half *scale           [[buffer(1)]],
    device const half *zero_point      [[buffer(2)]],
    device half *output                [[buffer(3)]],
    uint tid                           [[thread_position_in_grid]])
{
    if (tid > 0u) return;

    half4 result;
    dequant_s8x4_asym(packed_input[0], scale[0], zero_point[0], result);

    output[0] = result.x;
    output[1] = result.y;
    output[2] = result.z;
    output[3] = result.w;
}

/// Test kernel: dequantize 8 values (2 packed uint32s) to verify x8 variant.
kernel void test_int8_x8(
    device const uint32_t *packed_input [[buffer(0)]],
    device const half *scale           [[buffer(1)]],
    device half *output                [[buffer(2)]],
    uint tid                           [[thread_position_in_grid]])
{
    if (tid > 0u) return;

    half4 lo, hi;
    dequant_s8x8(packed_input[0], packed_input[1], scale[0], lo, hi);

    output[0] = lo.x;
    output[1] = lo.y;
    output[2] = lo.z;
    output[3] = lo.w;
    output[4] = hi.x;
    output[5] = hi.y;
    output[6] = hi.z;
    output[7] = hi.w;
}

/// Test kernel: verify all representable INT8 values dequantize correctly.
/// Thread i dequantizes value (i - 128), covering [-128, 127].
kernel void test_int8_all_values(
    device half *output [[buffer(0)]],
    uint tid           [[thread_position_in_grid]])
{
    if (tid >= 256u) return;

    // Pack single byte into a uint32 at position 0
    int8_t val = int8_t(tid) - int8_t(128);
    uint32_t packed = uint32_t(uint8_t(val));

    half4 result;
    dequant_s8x4(packed, half(1.0h), result);

    // Only result.x has our value; others are zeros
    output[tid] = result.x;
}
