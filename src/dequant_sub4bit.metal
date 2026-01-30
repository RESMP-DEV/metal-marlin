#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Sub-4-bit Dequantization Kernels: INT2, INT3, NF3
// ============================================================================
//
// For aggressive MoE expert compression (cold experts) and low-bandwidth
// scenarios. Based on llama.cpp IQ2/IQ3 and bitsandbytes NF quantization.
//
// Packing conventions:
//   INT2: 16 values per uint32 (2 bits each), LSB-first
//   INT3: 10 values per uint32 (3 bits each, 30 bits used, 2 bits padding)
//   NF3:  Same packing as INT3, but uses non-uniform Gaussian-quantile levels
//
// INT2 levels: [-1.5, -0.5, 0.5, 1.5] * scale
//   Code 0 -> -1.5, Code 1 -> -0.5, Code 2 -> 0.5, Code 3 -> 1.5
//
// INT3 levels: [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5] * scale
//   Code 0 -> -3.5, Code 1 -> -2.5, ..., Code 7 -> 3.5
//
// NF3 levels (NormalFloat, Gaussian quantiles normalized to [-1, 1]):
//   {-1.0, -0.578277, -0.318604, -0.102541, 0.102541, 0.318604, 0.578277, 1.0}
//   These minimize MSE for Gaussian-distributed weights (transformer weights).
// ============================================================================

// ============================================================================
// NF3 Lookup Table (pre-computed Gaussian quantiles)
// ============================================================================

constant float NF3_LUT[8] = {
    -1.000000f, -0.578277f, -0.318604f, -0.102541f,
     0.102541f,  0.318604f,  0.578277f,  1.000000f
};

// ============================================================================
// INT2 Dequantization: 16 values per uint32
// ============================================================================

/// Dequantize 16 INT2 values from a packed uint32.
///
/// Formula: output[i] = (code[i] - 1.5) * scale
/// This maps codes {0,1,2,3} -> {-1.5, -0.5, 0.5, 1.5} * scale
///
/// @param packed  16 packed 2-bit codes, LSB-first
/// @param scale   Per-group scale factor
/// @param out     Pointer to 16 thread-local half values
inline void dequant_int2_x16(uint32_t packed, half scale, thread half *out) {
    float fscale = float(scale);
    for (uint i = 0u; i < 16u; i++) {
        uint code = (packed >> (i * 2u)) & 0x3u;
        float dequant = (float(code) - 1.5f) * fscale;
        out[i] = half(dequant);
    }
}

/// Unrolled INT2 dequant for better ALU utilization.
/// Identical semantics to dequant_int2_x16, no loop overhead.
inline void dequant_int2_x16_unrolled(uint32_t packed, half scale, thread half *out) {
    float s = float(scale);
    #define DQ2(idx) out[idx] = half((float((packed >> ((idx) * 2u)) & 0x3u) - 1.5f) * s)
    DQ2(0);  DQ2(1);  DQ2(2);  DQ2(3);
    DQ2(4);  DQ2(5);  DQ2(6);  DQ2(7);
    DQ2(8);  DQ2(9);  DQ2(10); DQ2(11);
    DQ2(12); DQ2(13); DQ2(14); DQ2(15);
    #undef DQ2
}

/// Dequant first 8 of 16 INT2 values (for GEMM tile alignment with 8-wide tiles).
inline void dequant_int2_x8(uint32_t packed, half scale, thread half *out) {
    float fscale = float(scale);
    for (uint i = 0u; i < 8u; i++) {
        uint code = (packed >> (i * 2u)) & 0x3u;
        out[i] = half((float(code) - 1.5f) * fscale);
    }
}

/// Vectorized INT2 dequant returning half4 pairs for simdgroup ops.
inline void dequant_int2_x16_vec(uint32_t packed,
                                  half scale,
                                  thread half4 &out0,
                                  thread half4 &out1,
                                  thread half4 &out2,
                                  thread half4 &out3) {
    half vals[16];
    dequant_int2_x16_unrolled(packed, scale, vals);
    out0 = half4(vals[0], vals[1], vals[2], vals[3]);
    out1 = half4(vals[4], vals[5], vals[6], vals[7]);
    out2 = half4(vals[8], vals[9], vals[10], vals[11]);
    out3 = half4(vals[12], vals[13], vals[14], vals[15]);
}

// ============================================================================
// INT3 Dequantization: 10 values per uint32 (30 bits used, 2 padding)
// ============================================================================

/// Dequantize 10 INT3 values from a packed uint32.
///
/// Formula: output[i] = (code[i] - 3.5) * scale
/// This maps codes {0..7} -> {-3.5, -2.5, ..., 3.5} * scale
///
/// @param packed  10 packed 3-bit codes in bits [29:0], bits [31:30] unused
/// @param scale   Per-group scale factor
/// @param out     Pointer to 10 thread-local half values
inline void dequant_int3_x10(uint32_t packed, half scale, thread half *out) {
    float fscale = float(scale);
    for (uint i = 0u; i < 10u; i++) {
        uint code = (packed >> (i * 3u)) & 0x7u;
        float dequant = (float(code) - 3.5f) * fscale;
        out[i] = half(dequant);
    }
}

/// Unrolled INT3 dequant for better ALU utilization.
inline void dequant_int3_x10_unrolled(uint32_t packed, half scale, thread half *out) {
    float s = float(scale);
    #define DQ3(idx) out[idx] = half((float((packed >> ((idx) * 3u)) & 0x7u) - 3.5f) * s)
    DQ3(0); DQ3(1); DQ3(2); DQ3(3); DQ3(4);
    DQ3(5); DQ3(6); DQ3(7); DQ3(8); DQ3(9);
    #undef DQ3
}

/// Dequant first 8 of 10 INT3 values (for GEMM tile alignment).
inline void dequant_int3_x8(uint32_t packed, half scale, thread half *out) {
    float fscale = float(scale);
    for (uint i = 0u; i < 8u; i++) {
        uint code = (packed >> (i * 3u)) & 0x7u;
        out[i] = half((float(code) - 3.5f) * fscale);
    }
}

/// Vectorized INT3 dequant returning half4 pairs + half2 remainder.
/// Note: 10 values = 2x half4 + 1x half2
inline void dequant_int3_x10_vec(uint32_t packed,
                                  half scale,
                                  thread half4 &out0,
                                  thread half4 &out1,
                                  thread half2 &out2) {
    half vals[10];
    dequant_int3_x10_unrolled(packed, scale, vals);
    out0 = half4(vals[0], vals[1], vals[2], vals[3]);
    out1 = half4(vals[4], vals[5], vals[6], vals[7]);
    out2 = half2(vals[8], vals[9]);
}

// ============================================================================
// NF3 Dequantization: 10 values per uint32, Gaussian-quantile levels
// ============================================================================

/// Dequantize 10 NF3 values from a packed uint32.
///
/// NF3 uses non-uniform quantization levels based on Gaussian distribution
/// quantiles, which minimizes MSE for transformer weights that follow
/// approximately normal distributions.
///
/// The 8 levels (codes 0-7) map to:
///   {-1.0, -0.578, -0.319, -0.103, 0.103, 0.319, 0.578, 1.0} * scale
///
/// @param packed  10 packed 3-bit codes in bits [29:0], bits [31:30] unused
/// @param scale   Per-group scale factor
/// @param out     Pointer to 10 thread-local half values
inline void dequant_nf3_x10(uint32_t packed, half scale, thread half *out) {
    float fscale = float(scale);
    for (uint i = 0u; i < 10u; i++) {
        uint code = (packed >> (i * 3u)) & 0x7u;
        float dequant = NF3_LUT[code] * fscale;
        out[i] = half(dequant);
    }
}

/// Unrolled NF3 dequant.
inline void dequant_nf3_x10_unrolled(uint32_t packed, half scale, thread half *out) {
    float s = float(scale);
    #define DQ_NF3(idx) out[idx] = half(NF3_LUT[(packed >> ((idx) * 3u)) & 0x7u] * s)
    DQ_NF3(0); DQ_NF3(1); DQ_NF3(2); DQ_NF3(3); DQ_NF3(4);
    DQ_NF3(5); DQ_NF3(6); DQ_NF3(7); DQ_NF3(8); DQ_NF3(9);
    #undef DQ_NF3
}

/// Dequant first 8 of 10 NF3 values (for GEMM tile alignment).
inline void dequant_nf3_x8(uint32_t packed, half scale, thread half *out) {
    float fscale = float(scale);
    for (uint i = 0u; i < 8u; i++) {
        uint code = (packed >> (i * 3u)) & 0x7u;
        out[i] = half(NF3_LUT[code] * fscale);
    }
}

/// Vectorized NF3 dequant returning half4 pairs + half2 remainder.
inline void dequant_nf3_x10_vec(uint32_t packed,
                                 half scale,
                                 thread half4 &out0,
                                 thread half4 &out1,
                                 thread half2 &out2) {
    half vals[10];
    dequant_nf3_x10_unrolled(packed, scale, vals);
    out0 = half4(vals[0], vals[1], vals[2], vals[3]);
    out1 = half4(vals[4], vals[5], vals[6], vals[7]);
    out2 = half2(vals[8], vals[9]);
}

// ============================================================================
// Bulk dequantization compute kernels
// ============================================================================

/// Bulk INT2 -> FP16 dequantization kernel.
///
/// Layout:
///   packed_weights: [K/16, N] - each uint32 holds 16 INT2 values along K
///   scales: [K/group_size, N] - one scale per group per column
///   output: [K, N] - full FP16 output
///
/// Grid: (N, K/16, 1) - each thread handles one packed uint32
kernel void dequant_int2_bulk(
    device const uint32_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device half *output                  [[buffer(2)]],
    constant uint &K                     [[buffer(3)]],
    constant uint &N                     [[buffer(4)]],
    constant uint &group_size            [[buffer(5)]],
    uint2 gid                            [[thread_position_in_grid]]
) {
    uint n_idx = gid.x;     // column
    uint k_block = gid.y;   // which group of 16 K-elements

    uint k_start = k_block * 16u;
    if (n_idx >= N || k_start >= K) return;

    // Load packed word: row-major [K/16, N]
    uint packed_idx = k_block * N + n_idx;
    uint32_t word = packed_weights[packed_idx];

    // Determine quantization group and load scale
    uint group_idx = k_start / group_size;
    half scale = scales[group_idx * N + n_idx];

    // Dequantize 16 INT2 values
    half vals[16];
    dequant_int2_x16_unrolled(word, scale, vals);

    // Write to output [K, N] with boundary check on K
    uint k_remain = min(16u, K - k_start);
    uint out_base = k_start * N + n_idx;
    for (uint i = 0u; i < k_remain; i++) {
        output[out_base + i * N] = vals[i];
    }
}

/// Bulk INT3 -> FP16 dequantization kernel.
///
/// Layout:
///   packed_weights: [ceil(K/10), N] - each uint32 holds 10 INT3 values
///   scales: [K/group_size, N] - one scale per group per column
///   output: [K, N] - full FP16 output
///
/// Grid: (N, ceil(K/10), 1) - each thread handles one packed uint32
kernel void dequant_int3_bulk(
    device const uint32_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device half *output                  [[buffer(2)]],
    constant uint &K                     [[buffer(3)]],
    constant uint &N                     [[buffer(4)]],
    constant uint &group_size            [[buffer(5)]],
    uint2 gid                            [[thread_position_in_grid]]
) {
    uint n_idx = gid.x;     // column
    uint k_block = gid.y;   // which group of 10 K-elements

    uint k_start = k_block * 10u;
    if (n_idx >= N || k_start >= K) return;

    // Load packed word: row-major [K/10, N]
    uint packed_idx = k_block * N + n_idx;
    uint32_t word = packed_weights[packed_idx];

    // Determine quantization group and load scale
    uint group_idx = k_start / group_size;
    half scale = scales[group_idx * N + n_idx];

    // Dequantize 10 INT3 values
    half vals[10];
    dequant_int3_x10_unrolled(word, scale, vals);

    // Write to output [K, N] with boundary check on K
    uint k_remain = min(10u, K - k_start);
    uint out_base = k_start * N + n_idx;
    for (uint i = 0u; i < k_remain; i++) {
        output[out_base + i * N] = vals[i];
    }
}

/// Bulk NF3 -> FP16 dequantization kernel.
///
/// Same layout as INT3 but uses NormalFloat quantization levels.
///
/// Grid: (N, ceil(K/10), 1) - each thread handles one packed uint32
kernel void dequant_nf3_bulk(
    device const uint32_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device half *output                  [[buffer(2)]],
    constant uint &K                     [[buffer(3)]],
    constant uint &N                     [[buffer(4)]],
    constant uint &group_size            [[buffer(5)]],
    uint2 gid                            [[thread_position_in_grid]]
) {
    uint n_idx = gid.x;     // column
    uint k_block = gid.y;   // which group of 10 K-elements

    uint k_start = k_block * 10u;
    if (n_idx >= N || k_start >= K) return;

    // Load packed word: row-major [K/10, N]
    uint packed_idx = k_block * N + n_idx;
    uint32_t word = packed_weights[packed_idx];

    // Determine quantization group and load scale
    uint group_idx = k_start / group_size;
    half scale = scales[group_idx * N + n_idx];

    // Dequantize 10 NF3 values using Gaussian-quantile LUT
    half vals[10];
    dequant_nf3_x10_unrolled(word, scale, vals);

    // Write to output [K, N] with boundary check on K
    uint k_remain = min(10u, K - k_start);
    uint out_base = k_start * N + n_idx;
    for (uint i = 0u; i < k_remain; i++) {
        output[out_base + i * N] = vals[i];
    }
}

// ============================================================================
// Unit test kernels
// ============================================================================

/// Test kernel: dequantize all 4 INT2 codes (0-3) without scaling.
/// Writes 4 half values to output buffer.
kernel void test_int2_all_codes(
    device half *output [[buffer(0)]],
    uint tid           [[thread_position_in_grid]]
) {
    if (tid >= 4u) return;
    // Construct packed with code=tid in position 0
    uint32_t packed = tid;
    half scale = half(1.0h);
    half vals[16];
    dequant_int2_x16(packed, scale, vals);
    output[tid] = vals[0];
}

/// Test kernel: dequantize all 8 INT3 codes (0-7) without scaling.
kernel void test_int3_all_codes(
    device half *output [[buffer(0)]],
    uint tid           [[thread_position_in_grid]]
) {
    if (tid >= 8u) return;
    uint32_t packed = tid;
    half scale = half(1.0h);
    half vals[10];
    dequant_int3_x10(packed, scale, vals);
    output[tid] = vals[0];
}

/// Test kernel: dequantize all 8 NF3 codes (0-7) without scaling.
kernel void test_nf3_all_codes(
    device half *output [[buffer(0)]],
    uint tid           [[thread_position_in_grid]]
) {
    if (tid >= 8u) return;
    uint32_t packed = tid;
    half scale = half(1.0h);
    half vals[10];
    dequant_nf3_x10(packed, scale, vals);
    output[tid] = vals[0];
}

/// Test kernel: verify INT2 packed dequant with a given scale.
kernel void test_int2_packed_scaled(
    device const uint32_t *packed_input [[buffer(0)]],
    device const half *scale           [[buffer(1)]],
    device half *output                [[buffer(2)]],
    uint tid                           [[thread_position_in_grid]]
) {
    if (tid > 0u) return;

    half vals[16];
    dequant_int2_x16_unrolled(packed_input[0], scale[0], vals);

    for (uint i = 0u; i < 16u; i++) {
        output[i] = vals[i];
    }
}

/// Test kernel: verify INT3 packed dequant with a given scale.
kernel void test_int3_packed_scaled(
    device const uint32_t *packed_input [[buffer(0)]],
    device const half *scale           [[buffer(1)]],
    device half *output                [[buffer(2)]],
    uint tid                           [[thread_position_in_grid]]
) {
    if (tid > 0u) return;

    half vals[10];
    dequant_int3_x10_unrolled(packed_input[0], scale[0], vals);

    for (uint i = 0u; i < 10u; i++) {
        output[i] = vals[i];
    }
}

/// Test kernel: verify NF3 packed dequant with a given scale.
kernel void test_nf3_packed_scaled(
    device const uint32_t *packed_input [[buffer(0)]],
    device const half *scale           [[buffer(1)]],
    device half *output                [[buffer(2)]],
    uint tid                           [[thread_position_in_grid]]
) {
    if (tid > 0u) return;

    half vals[10];
    dequant_nf3_x10_unrolled(packed_input[0], scale[0], vals);

    for (uint i = 0u; i < 10u; i++) {
        output[i] = vals[i];
    }
}
