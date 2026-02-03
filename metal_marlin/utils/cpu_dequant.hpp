// ARM NEON-optimized CPU fallback dequantizers
#pragma once

#include <cstdint>

namespace metal_marlin {
namespace cpu_dequant {

/**
 * FP4 E2M1 dequantization
 * 
 * Dequantizes packed 4-bit floating point weights.
 * E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit.
 * 
 * @param packed  Packed nibbles [K/8, N] in uint32 format
 * @param scales  Per-group scales [K/group_size, N]
 * @param output  Dequantized output [K, N]
 * @param K       Reduction dimension (rows)
 * @param N       Output dimension (columns)
 * @param group_size  Elements per quantization group
 */
void dequant_fp4_e2m1(
    const uint32_t* packed,
    const float* scales,
    float* output,
    int K,
    int N,
    int group_size
);

/**
 * INT4 symmetric dequantization
 * 
 * Dequantizes 4-bit signed integer weights with symmetric quantization.
 * Range: [-8, 7] (zero at index 8).
 * 
 * @param packed  Packed nibbles [K/8, N]
 * @param scales  Per-group scales [K/group_size, N]
 * @param output  Dequantized output [K, N]
 * @param K, N, group_size  Dimensions
 */
void dequant_int4_symmetric(
    const uint32_t* packed,
    const float* scales,
    float* output,
    int K,
    int N,
    int group_size
);

/**
 * NF4 (NormalFloat4) dequantization
 * 
 * Dequantizes 4-bit weights with non-uniform codebook optimized
 * for normally distributed data (QLoRA-style).
 * 
 * @param packed  Packed nibbles [K/8, N]
 * @param scales  Per-group scales [K/group_size, N]
 * @param output  Dequantized output [K, N]
 * @param K, N, group_size  Dimensions
 */
void dequant_nf4(
    const uint32_t* packed,
    const float* scales,
    float* output,
    int K,
    int N,
    int group_size
);

/**
 * INT8 symmetric dequantization
 * 
 * Dequantizes 8-bit signed integer weights.
 * 
 * @param data    INT8 data [K, N]
 * @param scales  Per-group scales [K/group_size, N]
 * @param output  Dequantized output [K, N]
 * @param K, N, group_size  Dimensions
 */
void dequant_int8(
    const int8_t* data,
    const float* scales,
    float* output,
    int K,
    int N,
    int group_size
);

#ifdef __ARM_NEON

/**
 * FP8 E4M3 dequantization (ARM NEON only)
 * 
 * Dequantizes 8-bit floating point weights.
 * E4M3 format: 1 sign, 4 exponent, 3 mantissa bits.
 * 
 * @param data    FP8 codes [K, N]
 * @param scales  Per-group scales [K/group_size, N]
 * @param output  Dequantized output [K, N]
 * @param K, N, group_size  Dimensions
 */
void dequant_fp8_e4m3(
    const uint8_t* data,
    const float* scales,
    float* output,
    int K,
    int N,
    int group_size
);

/**
 * FP8 E5M2 dequantization (ARM NEON only)
 * 
 * Dequantizes 8-bit floating point weights with wider dynamic range.
 * E5M2 format: 1 sign, 5 exponent, 2 mantissa bits.
 * 
 * @param data    FP8 codes [K, N]
 * @param scales  Per-group scales [K/group_size, N]
 * @param output  Dequantized output [K, N]
 * @param K, N, group_size  Dimensions
 */
void dequant_fp8_e5m2(
    const uint8_t* data,
    const float* scales,
    float* output,
    int K,
    int N,
    int group_size
);

#endif // __ARM_NEON

} // namespace cpu_dequant
} // namespace metal_marlin
