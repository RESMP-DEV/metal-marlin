#include <metal_stdlib>
using namespace metal;

// ============================================================================
// GPTQ Dequantization Kernel
// ============================================================================

// Magic numbers for fast bitwise dequantization (same as dequant.metal)
constant constexpr uint32_t MAGIC_BIAS_U32 = 0x64006400u;
constant constexpr uint32_t LO_NIBBLE_MASK = 0x000F000Fu;
constant constexpr uint16_t MAGIC_BIAS_F16 = 0x6400u;

/// Dequantize 8 unsigned INT4 values packed in a uint32_t to 8 FP16 values.
inline void dequant_u4x8_gptq(uint32_t packed,
                              half scale,
                              half zero_point,
                              thread half4 &out_lo,
                              thread half4 &out_hi) {
    // Treat packed as two uint16 lanes
    // Lane layout:
    //   bits [15:0]  = lo_lane: nibbles [val3][val2][val1][val0]
    //   bits [31:16] = hi_lane: nibbles [val7][val6][val5][val4]

    // Nibbles 0 and 4
    uint32_t n0_biased = (packed & LO_NIBBLE_MASK) | MAGIC_BIAS_U32;
    half2 n0_pair = as_type<half2>(n0_biased) - as_type<half2>(MAGIC_BIAS_U32);

    // Nibbles 1 and 5
    uint32_t n1_shifted = (packed >> 4u) & LO_NIBBLE_MASK;
    uint32_t n1_biased = n1_shifted | MAGIC_BIAS_U32;
    half2 n1_pair = as_type<half2>(n1_biased) - as_type<half2>(MAGIC_BIAS_U32);

    // Nibbles 2 and 6
    uint32_t n2_shifted = (packed >> 8u) & LO_NIBBLE_MASK;
    uint32_t n2_biased = n2_shifted | MAGIC_BIAS_U32;
    half2 n2_pair = as_type<half2>(n2_biased) - as_type<half2>(MAGIC_BIAS_U32);

    // Nibbles 3 and 7
    uint32_t n3_shifted = (packed >> 12u) & LO_NIBBLE_MASK;
    uint32_t n3_biased = n3_shifted | MAGIC_BIAS_U32;
    half2 n3_pair = as_type<half2>(n3_biased) - as_type<half2>(MAGIC_BIAS_U32);

    out_lo = half4(n0_pair.x, n1_pair.x, n2_pair.x, n3_pair.x);
    out_hi = half4(n0_pair.y, n1_pair.y, n2_pair.y, n3_pair.y);

    // GPTQ dequant: (q - z) * s
    // GPTQ zeros are typically stored as (zero + 1) in older versions, 
    // or just raw values. We assume 'zero_point' is already the correct float value.
    out_lo = (out_lo - zero_point) * scale;
    out_hi = (out_hi - zero_point) * scale;
}

/// Dequantize GPTQ weights (packed int32) to FP16.
///
/// @param packed_weights [K/8, N] uint32
/// @param scales        [n_groups, N] half
/// @param zeros         [n_groups, N] half (unpacked)
/// @param output        [K, N] half
/// @param K             Input features
/// @param N             Output features
/// @param group_size    Group size (e.g. 128)
kernel void dequant_gptq_kernel(
    device const uint32_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device const half *zeros             [[buffer(2)]],
    device half *output                  [[buffer(3)]],
    constant uint32_t &K                 [[buffer(4)]],
    constant uint32_t &N                 [[buffer(5)]],
    constant uint32_t &group_size        [[buffer(6)]],
    uint2 gid                            [[thread_position_in_grid]])
{
    uint n_idx = gid.x;
    uint k_block = gid.y; // Block index along K (8 weights per block)

    uint k_start = k_block * 8u;
    if (n_idx >= N || k_start >= K) return;

    // Load packed weights
    uint packed_idx = k_block * N + n_idx;
    uint32_t packed = packed_weights[packed_idx];

    // Load scale and zero for this group
    // Groups are along K dimension
    uint group_idx = k_start / group_size;
    uint param_idx = group_idx * N + n_idx;
    
    half scale = scales[param_idx];
    half zero = zeros[param_idx];

    // Dequantize
    half4 lo, hi;
    dequant_u4x8_gptq(packed, scale, zero, lo, hi);

    // Write output [K, N] column-major-ish (standard row-major matrix flattened)
    // output is [K, N], so index = k * N + n
    
    uint out_base = k_start * N + n_idx;
    
    // Boundary check for K
    uint k_remain = min(8u, K - k_start);
    
    if (k_remain == 8u) {
        output[out_base + 0u * N] = lo.x;
        output[out_base + 1u * N] = lo.y;
        output[out_base + 2u * N] = lo.z;
        output[out_base + 3u * N] = lo.w;
        output[out_base + 4u * N] = hi.x;
        output[out_base + 5u * N] = hi.y;
        output[out_base + 6u * N] = hi.z;
        output[out_base + 7u * N] = hi.w;
    } else {
        half vals[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        for (uint i = 0; i < k_remain; i++) {
            output[out_base + i * N] = vals[i];
        }
    }
}
