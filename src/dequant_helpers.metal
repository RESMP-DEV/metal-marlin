// dequant_helpers.metal - Shared dequantization helper functions
//
// This header provides common dequantization utilities for FP4 and other
// quantized formats. Include this in kernels that need dequantization
// without duplicating the LUT and helper functions.

#ifndef DEQUANT_HELPERS_METAL
#define DEQUANT_HELPERS_METAL

#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// FP4 E2M1 Dequantization LUT
//
// Format: [sign(1) | exponent(2) | mantissa(1)]
//
// Values for positive numbers (bit3=0):
//   0000 -> 0.0    (exp=0, man=0 -> subnormal 0*0.25)
//   0001 -> 0.25   (exp=0, man=1 -> subnormal 1*0.25)
//   0010 -> 1.0    (exp=1, man=0 -> 2^0 * 1.0)
//   0011 -> 1.5    (exp=1, man=1 -> 2^0 * 1.5)
//   0100 -> 2.0    (exp=2, man=0 -> 2^1 * 1.0)
//   0101 -> 3.0    (exp=2, man=1 -> 2^1 * 1.5)
//   0110 -> 4.0    (exp=3, man=0 -> 2^2 * 1.0)
//   0111 -> 6.0    (exp=3, man=1 -> 2^2 * 1.5)
//
// Negative numbers (bit3=1) are negations of above.
// ===========================================================================

constant half FP4_DEQUANT_LUT[16] = {
    half( 0.00h),  // 0000
    half( 0.25h),  // 0001
    half( 1.00h),  // 0010
    half( 1.50h),  // 0011
    half( 2.00h),  // 0100
    half( 3.00h),  // 0101
    half( 4.00h),  // 0110
    half( 6.00h),  // 0111
    half(-0.00h),  // 1000 (negative zero, same as zero)
    half(-0.25h),  // 1001
    half(-1.00h),  // 1010
    half(-1.50h),  // 1011
    half(-2.00h),  // 1100
    half(-3.00h),  // 1101
    half(-4.00h),  // 1110
    half(-6.00h),  // 1111
};

// ===========================================================================
// Single-value dequantization
// ===========================================================================

// Dequantize a single FP4 nibble with scale
inline half dequant_fp4(uint nibble, half scale) {
    return FP4_DEQUANT_LUT[nibble & 0xF] * scale;
}

// ===========================================================================
// Packed dequantization (multiple values at once)
// ===========================================================================

// Dequantize 8 FP4 values from a packed uint32_t
// packed: 8 nibbles, LSB first (bits 0-3 = value 0, bits 4-7 = value 1, etc.)
// scale: quantization scale factor
// out: output array of 8 half values (must be thread-addressable)
inline void dequant_fp4x8(
    uint32_t packed,
    half scale,
    thread half* out
) {
    out[0] = FP4_DEQUANT_LUT[(packed >>  0) & 0xF] * scale;
    out[1] = FP4_DEQUANT_LUT[(packed >>  4) & 0xF] * scale;
    out[2] = FP4_DEQUANT_LUT[(packed >>  8) & 0xF] * scale;
    out[3] = FP4_DEQUANT_LUT[(packed >> 12) & 0xF] * scale;
    out[4] = FP4_DEQUANT_LUT[(packed >> 16) & 0xF] * scale;
    out[5] = FP4_DEQUANT_LUT[(packed >> 20) & 0xF] * scale;
    out[6] = FP4_DEQUANT_LUT[(packed >> 24) & 0xF] * scale;
    out[7] = FP4_DEQUANT_LUT[(packed >> 28) & 0xF] * scale;
}

// Dequantize 4 FP4 values from half of a packed uint32_t
// packed: 8 nibbles packed into uint32
// offset: 0 for lower 4 nibbles (bits 0-15), 1 for upper 4 nibbles (bits 16-31)
// scale: quantization scale factor
// out: output array of 4 half values
inline void dequant_fp4x4(
    uint32_t packed,
    uint offset,
    half scale,
    thread half* out
) {
    uint shift = offset * 16;  // 0 or 16
    out[0] = FP4_DEQUANT_LUT[(packed >> (shift +  0)) & 0xF] * scale;
    out[1] = FP4_DEQUANT_LUT[(packed >> (shift +  4)) & 0xF] * scale;
    out[2] = FP4_DEQUANT_LUT[(packed >> (shift +  8)) & 0xF] * scale;
    out[3] = FP4_DEQUANT_LUT[(packed >> (shift + 12)) & 0xF] * scale;
}

// Dequantize 2 FP4 values from a single byte
// packed_byte: 2 nibbles packed (bits 0-3 = value 0, bits 4-7 = value 1)
// scale: quantization scale factor
// out: output array of 2 half values
inline void dequant_fp4x2(
    uchar packed_byte,
    half scale,
    thread half* out
) {
    out[0] = FP4_DEQUANT_LUT[(packed_byte >> 0) & 0xF] * scale;
    out[1] = FP4_DEQUANT_LUT[(packed_byte >> 4) & 0xF] * scale;
}

// ===========================================================================
// Vectorized dequantization (returns vector types)
// ===========================================================================

// Dequantize 8 FP4 values and return as two half4 vectors
// out_lo: lower 4 values (nibbles 0-3)
// out_hi: upper 4 values (nibbles 4-7)
inline void dequant_fp4x8_vec(uint32_t packed, half scale, thread half4& out_lo, thread half4& out_hi) {
    out_lo = half4(
        FP4_DEQUANT_LUT[(packed >>  0) & 0xF] * scale,
        FP4_DEQUANT_LUT[(packed >>  4) & 0xF] * scale,
        FP4_DEQUANT_LUT[(packed >>  8) & 0xF] * scale,
        FP4_DEQUANT_LUT[(packed >> 12) & 0xF] * scale
    );
    out_hi = half4(
        FP4_DEQUANT_LUT[(packed >> 16) & 0xF] * scale,
        FP4_DEQUANT_LUT[(packed >> 20) & 0xF] * scale,
        FP4_DEQUANT_LUT[(packed >> 24) & 0xF] * scale,
        FP4_DEQUANT_LUT[(packed >> 28) & 0xF] * scale
    );
}

// Dequantize 4 FP4 values and return as half4
inline half4 dequant_fp4x4_vec(uint32_t packed, uint offset, half scale) {
    uint shift = offset * 16;
    return half4(
        FP4_DEQUANT_LUT[(packed >> (shift +  0)) & 0xF] * scale,
        FP4_DEQUANT_LUT[(packed >> (shift +  4)) & 0xF] * scale,
        FP4_DEQUANT_LUT[(packed >> (shift +  8)) & 0xF] * scale,
        FP4_DEQUANT_LUT[(packed >> (shift + 12)) & 0xF] * scale
    );
}

// Dequantize lower 4 FP4 values (nibbles 0-3) and return as half4
inline half4 dequant_fp4x4_lo_vec(uint32_t packed, half scale) {
    return half4(
        FP4_DEQUANT_LUT[(packed >>  0) & 0xF] * scale,
        FP4_DEQUANT_LUT[(packed >>  4) & 0xF] * scale,
        FP4_DEQUANT_LUT[(packed >>  8) & 0xF] * scale,
        FP4_DEQUANT_LUT[(packed >> 12) & 0xF] * scale
    );
}

// Dequantize upper 4 FP4 values (nibbles 4-7) and return as half4
inline half4 dequant_fp4x4_hi_vec(uint32_t packed, half scale) {
    return half4(
        FP4_DEQUANT_LUT[(packed >> 16) & 0xF] * scale,
        FP4_DEQUANT_LUT[(packed >> 20) & 0xF] * scale,
        FP4_DEQUANT_LUT[(packed >> 24) & 0xF] * scale,
        FP4_DEQUANT_LUT[(packed >> 28) & 0xF] * scale
    );
}

// Dequantize 2 FP4 values and return as half2
inline half2 dequant_fp4x2_vec(uchar packed_byte, half scale) {
    return half2(
        FP4_DEQUANT_LUT[(packed_byte >> 0) & 0xF] * scale,
        FP4_DEQUANT_LUT[(packed_byte >> 4) & 0xF] * scale
    );
}

#endif // DEQUANT_HELPERS_METAL
