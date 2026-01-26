#include <metal_stdlib>
using namespace metal;

// ============================================================================
// INT4 (U4/S4) → FP16 Bitwise Dequantization for Apple Metal
// ============================================================================
//
// Port of Marlin's "magic number" dequantization trick from CUDA.
//
// The core idea: avoid branches and lookup tables by using FP16 bit
// manipulation. We place each 4-bit integer value into the mantissa of a
// half-precision float with a known exponent, then subtract the bias to
// recover the integer value as a float.
//
// FP16 layout: [1 sign][5 exponent][10 mantissa]
//
// Magic bias: 0x6400 = exponent 0x19 (25), mantissa 0
//   Value = 2^(25-15) * (1 + 0/1024) = 1024.0
//
// When we OR a 4-bit value N (0..15) into bits [3:0] of the mantissa:
//   Value = 2^10 * (1 + N/1024) = 1024 + N
//
// Subtract 1024.0 → recovers N as FP16.
//
// For signed INT4 (S4, range [-8, 7]): dequant as U4, then subtract 8.
//
// This is the Metal equivalent of Marlin's lop3-based dequant. Metal lacks
// lop3.b32 (ternary logic op), so we compose from &, |, ^ which the Metal
// compiler will fuse into optimal ALU sequences on Apple Silicon.
//
// ============================================================================
// PACK/UNPACK CONTRACT (for Python interop)
// ============================================================================
//
// Nibble packing convention: LSB-first (little-endian nibble order)
//   - nibble i occupies bits [(i+1)*4-1 : i*4] of the packed uint32
//   - packed = 0x76543210 contains nibbles [0,1,2,3,4,5,6,7]
//   - nibble[i] = (packed >> (i * 4)) & 0xF
//
// Quantization (Python pack_u4_weights):
//   scale = (max_val - min_val) / 15
//   zero  = round(-min_val / scale)        // nibble value mapping to 0.0
//   q     = round(weight / scale + zero)   // clamped to [0, 15]
//
// Dequantization (this Metal code):
//   output = (nibble - zero_point) * scale
//
// The Python pack and Metal unpack are mathematical inverses:
//   weight ≈ (q - zero) * scale
//         = (round(w/s + z) - z) * s
//         ≈ w  (within quantization error)
//
// ============================================================================

// ============================================================================
// Constants
// ============================================================================

// Two FP16 1024.0 values packed into a uint32
constant constexpr uint32_t MAGIC_BIAS_U32 = 0x64006400u;

// Masks for extracting nibbles from packed uint32 (two FP16 lanes)
// For a uint32 interpreted as two uint16: [hi16][lo16]
// Lo nibble mask: bits [3:0] of each uint16 lane
constant constexpr uint32_t LO_NIBBLE_MASK = 0x000F000Fu;
// FP16 representation of 1024.0
constant constexpr uint16_t MAGIC_BIAS_F16 = 0x6400u;

// ============================================================================
// Core dequantization primitives
// ============================================================================

/// Dequantize a single 4-bit unsigned value to FP16 using the magic bias trick.
/// Input: val in [0, 15], occupying bits [3:0] of a uint16.
/// Output: half-precision float equal to val.
inline half dequant_u4_scalar(uint16_t val) {
    // Place val into mantissa of 1024.0
    uint16_t biased = (val & 0x000Fu) | MAGIC_BIAS_F16;
    // Reinterpret as FP16 and subtract bias
    return as_type<half>(biased) - as_type<half>(MAGIC_BIAS_F16);
}

/// Dequantize a single 4-bit signed value to FP16.
/// Input: val in [0, 15] representing two's-complement [-8, 7].
/// Output: half-precision float in [-8, 7].
inline half dequant_s4_scalar(uint16_t val) {
    uint16_t biased = (val & 0x000Fu) | MAGIC_BIAS_F16;
    // Subtract (1024 + 8) = 1032.0 to recover signed value
    return as_type<half>(biased) - half(1032.0h);
}

// ============================================================================
// Vectorized dequantization: 8 values from one uint32_t
// ============================================================================

/// Dequantize 8 unsigned INT4 values packed in a uint32_t to 8 FP16 values.
///
/// Packing format: bits [31:28] = val7, [27:24] = val6, ..., [3:0] = val0
/// (standard little-endian nibble packing, same as Marlin/GPTQ).
///
/// Output: 8 half values in two half4 vectors (lo = vals 0-3, hi = vals 4-7).
///
/// The scale and zero_point implement asymmetric per-group dequantization:
///   result = (uint4_value - zero_point) * scale
///
/// @param packed    8 packed 4-bit unsigned integers
/// @param scale     Per-group scale factor
/// @param zero_point Per-group zero point (in INT4 domain, already FP16)
/// @param out_lo   Output values 0-3
/// @param out_hi   Output values 4-7
inline void dequant_u4x8(uint32_t packed,
                         half scale,
                         half zero_point,
                         thread half4 &out_lo,
                         thread half4 &out_hi) {
    // Process in pairs: treat packed as two uint16 lanes for SIMD-style ops.
    // Each uint16 lane contains 4 nibbles (16 bits = 4 x 4-bit values).
    //
    // Lane layout within uint32:
    //   bits [15:0]  = lo_lane: nibbles [val3][val2][val1][val0]
    //   bits [31:16] = hi_lane: nibbles [val7][val6][val5][val4]

    // --- Nibbles 0 and 4 (bits [3:0] of each lane) ---
    uint32_t n0_biased = (packed & LO_NIBBLE_MASK) | MAGIC_BIAS_U32;
    half2 n0_pair = as_type<half2>(n0_biased) - as_type<half2>(MAGIC_BIAS_U32);

    // --- Nibbles 1 and 5 (bits [7:4] of each lane) ---
    uint32_t n1_shifted = (packed >> 4u) & LO_NIBBLE_MASK;
    uint32_t n1_biased = n1_shifted | MAGIC_BIAS_U32;
    half2 n1_pair = as_type<half2>(n1_biased) - as_type<half2>(MAGIC_BIAS_U32);

    // --- Nibbles 2 and 6 (bits [11:8] of each lane) ---
    uint32_t n2_shifted = (packed >> 8u) & LO_NIBBLE_MASK;
    uint32_t n2_biased = n2_shifted | MAGIC_BIAS_U32;
    half2 n2_pair = as_type<half2>(n2_biased) - as_type<half2>(MAGIC_BIAS_U32);

    // --- Nibbles 3 and 7 (bits [15:12] of each lane) ---
    uint32_t n3_shifted = (packed >> 12u) & LO_NIBBLE_MASK;
    uint32_t n3_biased = n3_shifted | MAGIC_BIAS_U32;
    half2 n3_pair = as_type<half2>(n3_biased) - as_type<half2>(MAGIC_BIAS_U32);

    // Assemble: lo lane values are .x components, hi lane values are .y
    out_lo = half4(n0_pair.x, n1_pair.x, n2_pair.x, n3_pair.x);
    out_hi = half4(n0_pair.y, n1_pair.y, n2_pair.y, n3_pair.y);

    // Apply asymmetric dequantization: (val - zero_point) * scale
    out_lo = (out_lo - zero_point) * scale;
    out_hi = (out_hi - zero_point) * scale;
}

/// Dequantize 8 signed INT4 values packed in a uint32_t to 8 FP16 values.
///
/// Signed INT4 uses offset binary representation in GPTQ/Marlin:
/// stored_value = actual_value + 8, so stored is [0,15] for actual [-8,7].
///
/// The zero_point is in the signed domain (i.e., the actual zero point before
/// the +8 offset was applied).
///
/// result = (stored_uint4 - 8 - zero_point) * scale
///        = (stored_uint4 - (8 + zero_point)) * scale
///
/// @param packed    8 packed 4-bit values (offset binary: actual + 8)
/// @param scale     Per-group scale factor
/// @param zero_point Per-group zero point (signed domain)
/// @param out_lo   Output values 0-3
/// @param out_hi   Output values 4-7
inline void dequant_s4x8(uint32_t packed,
                         half scale,
                         half zero_point,
                         thread half4 &out_lo,
                         thread half4 &out_hi) {
    // Same bit extraction as U4
    half2 bias_pair = as_type<half2>(MAGIC_BIAS_U32);

    uint32_t n0_biased = (packed & LO_NIBBLE_MASK) | MAGIC_BIAS_U32;
    half2 n0_pair = as_type<half2>(n0_biased) - bias_pair;

    uint32_t n1_biased = ((packed >> 4u) & LO_NIBBLE_MASK) | MAGIC_BIAS_U32;
    half2 n1_pair = as_type<half2>(n1_biased) - bias_pair;

    uint32_t n2_biased = ((packed >> 8u) & LO_NIBBLE_MASK) | MAGIC_BIAS_U32;
    half2 n2_pair = as_type<half2>(n2_biased) - bias_pair;

    uint32_t n3_biased = ((packed >> 12u) & LO_NIBBLE_MASK) | MAGIC_BIAS_U32;
    half2 n3_pair = as_type<half2>(n3_biased) - bias_pair;

    out_lo = half4(n0_pair.x, n1_pair.x, n2_pair.x, n3_pair.x);
    out_hi = half4(n0_pair.y, n1_pair.y, n2_pair.y, n3_pair.y);

    // Subtract 8 (S4 offset) and zero_point, then scale
    half combined_offset = half(8.0h) + zero_point;
    out_lo = (out_lo - combined_offset) * scale;
    out_hi = (out_hi - combined_offset) * scale;
}

// ============================================================================
// Alternative: Process packed pairs directly using uint32 reinterpretation
// ============================================================================

/// Dequantize 2 unsigned INT4 values from adjacent nibbles in a uint16.
/// More granular than the x8 variant; useful when integrating into GEMM tiles
/// where you process sub-tile slices.
///
/// @param packed_u16  A uint16 containing 4 nibbles [n3:n2:n1:n0]
/// @param nibble_idx  Which pair to extract: 0 = (n0,n1), 1 = (n2,n3)
/// @param scale       Per-group scale
/// @param zero_point  Per-group zero point
/// @return            Two dequantized FP16 values
inline half2 dequant_u4x2(uint16_t packed_u16,
                          uint nibble_idx,
                          half scale,
                          half zero_point) {
    uint shift = nibble_idx * 8u;
    uint16_t lo_nibble = (packed_u16 >> shift) & 0x000Fu;
    uint16_t hi_nibble = (packed_u16 >> (shift + 4u)) & 0x000Fu;

    uint16_t lo_biased = lo_nibble | MAGIC_BIAS_F16;
    uint16_t hi_biased = hi_nibble | MAGIC_BIAS_F16;
    half lo_val = as_type<half>(lo_biased) - as_type<half>(MAGIC_BIAS_F16);
    half hi_val = as_type<half>(hi_biased) - as_type<half>(MAGIC_BIAS_F16);

    return half2((lo_val - zero_point) * scale,
                 (hi_val - zero_point) * scale);
}

// ============================================================================
// Compute kernel: Bulk dequantize a packed INT4 buffer
// ============================================================================

/// Kernel that dequantizes a buffer of packed INT4 weights to FP16.
/// Each uint32 contains 8 INT4 values.
///
/// @param packed_weights  Input buffer: N/8 uint32 values
/// @param scales          Per-group scale factors (group_size groups)
/// @param zeros           Per-group zero points (group_size groups)
/// @param output          Output buffer: N half values
/// @param num_elements    Total number of INT4 values (N)
/// @param group_size      Number of elements per quantization group (e.g., 128)
/// @param is_signed       0 = unsigned INT4, 1 = signed INT4
kernel void dequant_int4_kernel(
    device const uint32_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device const half *zeros             [[buffer(2)]],
    device half *output                  [[buffer(3)]],
    constant uint32_t &num_elements      [[buffer(4)]],
    constant uint32_t &group_size        [[buffer(5)]],
    constant uint32_t &is_signed         [[buffer(6)]],
    uint tid                             [[thread_position_in_grid]])
{
    // Each thread processes one uint32 = 8 INT4 values
    uint base_idx = tid * 8u;
    if (base_idx >= num_elements) return;

    uint32_t packed = packed_weights[tid];

    // Determine which quantization group this block belongs to
    uint group_idx = base_idx / group_size;
    half scale = scales[group_idx];
    half zero_point = zeros[group_idx];

    half4 lo, hi;
    if (is_signed) {
        dequant_s4x8(packed, scale, zero_point, lo, hi);
    } else {
        dequant_u4x8(packed, scale, zero_point, lo, hi);
    }

    // Write output (handle boundary: last group may not be full 8)
    uint remaining = min(8u, num_elements - base_idx);

    if (remaining >= 4u) {
        // Write first 4
        output[base_idx + 0u] = lo.x;
        output[base_idx + 1u] = lo.y;
        output[base_idx + 2u] = lo.z;
        output[base_idx + 3u] = lo.w;
        if (remaining >= 8u) {
            output[base_idx + 4u] = hi.x;
            output[base_idx + 5u] = hi.y;
            output[base_idx + 6u] = hi.z;
            output[base_idx + 7u] = hi.w;
        } else {
            for (uint i = 4u; i < remaining; i++) {
                output[base_idx + i] = hi[i - 4u];
            }
        }
    } else {
        for (uint i = 0u; i < remaining; i++) {
            output[base_idx + i] = lo[i];
        }
    }
}

// ============================================================================
// Optimized kernel: Vectorized writes using half4 store
// ============================================================================

/// High-throughput variant that assumes num_elements is a multiple of 8.
/// Uses half4 stores for coalesced memory writes.
kernel void dequant_int4_aligned_kernel(
    device const uint32_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device const half *zeros             [[buffer(2)]],
    device half4 *output                 [[buffer(3)]],
    constant uint32_t &num_packed        [[buffer(4)]],
    constant uint32_t &group_size        [[buffer(5)]],
    constant uint32_t &is_signed         [[buffer(6)]],
    uint tid                             [[thread_position_in_grid]])
{
    if (tid >= num_packed) return;

    uint32_t packed = packed_weights[tid];
    uint base_idx = tid * 8u;
    uint group_idx = base_idx / group_size;
    half scale = scales[group_idx];
    half zero_point = zeros[group_idx];

    half4 lo, hi;
    if (is_signed) {
        dequant_s4x8(packed, scale, zero_point, lo, hi);
    } else {
        dequant_u4x8(packed, scale, zero_point, lo, hi);
    }

    // Two half4 stores (8 elements total)
    output[tid * 2u]     = lo;
    output[tid * 2u + 1u] = hi;
}

// ============================================================================
// Inline helpers for use in fused GEMM kernels
// ============================================================================

/// Dequantize 8 U4 values and return as a pair of half4, ready for
/// simdgroup_multiply_accumulate. No separate kernel dispatch needed;
/// call this inline within your GEMM tile loop.
inline void dequant_u4x8_fused(uint32_t packed,
                               half scale,
                               half zero_point,
                               thread half4 &tile_lo,
                               thread half4 &tile_hi) {
    dequant_u4x8(packed, scale, zero_point, tile_lo, tile_hi);
}

/// Same as above for signed INT4.
inline void dequant_s4x8_fused(uint32_t packed,
                               half scale,
                               half zero_point,
                               thread half4 &tile_lo,
                               thread half4 &tile_hi) {
    dequant_s4x8(packed, scale, zero_point, tile_lo, tile_hi);
}

// ============================================================================
// AWQ/GPTQ compatibility wrappers
// ============================================================================

/// Asymmetric INT4 dequantization for AWQ/GPTQ-style quantized models.
///
/// AWQ/GPTQ store per-group zero points as FP16 values representing the
/// zero offset in the integer domain. The dequantization formula is:
///   result = (int4_value - zero_point) * scale
///
/// This is a named wrapper around dequant_u4x8 for clarity; the underlying
/// implementation already performs fused asymmetric dequantization in a single
/// pass. Prefer calling dequant_u4x8 directly in performance-critical paths;
/// this wrapper exists for API discoverability.
///
/// @param packed      8 packed 4-bit unsigned integers
/// @param scale       Per-group scale factor (FP16)
/// @param zero_point  Per-group zero point in INT4 domain (FP16)
/// @param out_lo      Output values 0-3
/// @param out_hi      Output values 4-7
inline void dequant_u4x8_asymmetric(uint32_t packed,
                                    half scale,
                                    half zero_point,
                                    thread half4 &out_lo,
                                    thread half4 &out_hi) {
    dequant_u4x8(packed, scale, zero_point, out_lo, out_hi);
}

/// Signed asymmetric variant for models that use S4 (offset binary) encoding
/// with per-group zero points. The formula is:
///   result = (stored_uint4 - 8 - zero_point) * scale
///
/// @param packed      8 packed 4-bit values (offset binary: actual + 8)
/// @param scale       Per-group scale factor (FP16)
/// @param zero_point  Per-group zero point in signed domain (FP16)
/// @param out_lo      Output values 0-3
/// @param out_hi      Output values 4-7
inline void dequant_s4x8_asymmetric(uint32_t packed,
                                    half scale,
                                    half zero_point,
                                    thread half4 &out_lo,
                                    thread half4 &out_hi) {
    dequant_s4x8(packed, scale, zero_point, out_lo, out_hi);
}

// ============================================================================
// FP8 (E5M2) -> FP16 Dequantization (IEEE-like)
// ============================================================================
//
// FP8 E5M2 format: [1 sign][5 exponent (bias=15)][2 mantissa]
// - exp = 0: subnormal (mant * 2^-16)
// - exp = 31: inf/NaN
// - otherwise: 2^(exp-15) * (1 + mant/4)
// ============================================================================

inline half dequant_fp8_e5m2(uchar fp8_code, half scale) {
    bool sign = (fp8_code & 0x80u) != 0u;
    uint exp = (fp8_code >> 2) & 0x1Fu;
    uint mant = fp8_code & 0x3u;

    half val;
    if (exp == 0u) {
        // Subnormal (includes signed zero when mant == 0)
        val = half(mant) * half(exp2(-16.0f));
    } else if (exp == 31u) {
        // Inf or NaN
        val = (mant == 0u) ? half(INFINITY) : half(NAN);
    } else {
        // Normal: 2^(exp-15) * (1 + mant/4)
        float base = exp2(float(exp) - 15.0f);
        float frac = 1.0f + float(mant) * 0.25f;
        val = half(base * frac);
    }

    if (sign) val = -val;
    return val * scale;
}

// ============================================================================
// FP4 (E2M1 / NVFP4) -> FP16 Bitwise Dequantization
// ============================================================================
//
// FP4 E2M1 format: [1 sign][2 exponent (bias=1)][1 mantissa]
//
// Value encoding:
//   Normal (E != 0):   (-1)^S * 2^(E-1) * (1 + M*0.5)
//                      = (-1)^S * 2^(E-1) * (1.M in binary)
//   Subnormal (E == 0, M == 1): +/- 0.5  (= (-1)^S * 2^0 * 0.5)
//   Zero (E == 0, M == 0):      +/- 0.0
//
// Reference LUT (kvalues_mxfp4_f):
//   Code: 0    1    2    3    4    5    6    7
//   Val:  0   0.5  1.0  1.5  2.0  3.0  4.0  6.0
//   Code: 8    9   10   11   12   13   14   15
//   Val: -0  -0.5 -1.0 -1.5 -2.0 -3.0 -4.0 -6.0
//
// FP16 format: [1 sign][5 exponent (bias=15)][10 mantissa]
//
// Conversion (normal case, E > 0):
//   FP16 sign = S
//   FP16 exponent = E - 1 + 15 = E + 14
//   FP16 mantissa = M << 9 (1-bit mantissa left-aligned in 10-bit field)
//
// Conversion (subnormal, E == 0, M == 1):
//   Value = 0.5 = 2^(-1)
//   FP16: exp = -1 + 15 = 14, mant = 0
//
// Conversion (zero, E == 0, M == 0):
//   FP16: exp = 0, mant = 0 (signed zero preserved)
//
// Key insight vs LUT: this is pure ALU (3-5 ops per value) with no memory
// loads, eliminating the bandwidth bottleneck of the 16-entry lookup table.
// ============================================================================

// ============================================================================
// Core FP4 -> FP16 primitives
// ============================================================================

/// Branchless dequantize a single FP4 E2M1 nibble to half precision.
/// Uses select() to handle the three cases (normal, subnormal, zero) without
/// divergent control flow.
///
/// @param nibble  4-bit FP4 value in bits [3:0] of the input
/// @return        Equivalent half-precision value
inline half dequant_fp4_scalar(uint nibble) {
    uint S = (nibble >> 3) & 1u;
    uint E = (nibble >> 1) & 3u;
    uint M = nibble & 1u;

    // Normal case: exp = E + 14, mantissa = M << 9
    uint exp_normal = E + 14u;
    uint mant_normal = M << 9;

    // Subnormal (E=0, M=1): represents +/- 0.5 -> FP16 exp=14, mant=0
    // Zero (E=0, M=0): FP16 exp=0, mant=0

    // Branchless selection:
    // When E == 0: exp = 14*M (gives 14 for subnormal, 0 for zero)
    //              mant = 0
    // When E != 0: exp = E + 14, mant = M << 9
    bool is_normal = (E != 0u);
    uint fp16_exp = select(14u * M, exp_normal, is_normal);
    uint fp16_mant = select(0u, mant_normal, is_normal);

    ushort fp16_bits = ushort((S << 15) | (fp16_exp << 10) | fp16_mant);
    return as_type<half>(fp16_bits);
}

/// Dequantize two adjacent FP4 nibbles from a byte to half2.
///
/// @param byte_val  8-bit value containing 2 FP4 nibbles: [hi:lo]
/// @return          half2 with lo nibble in .x, hi nibble in .y
inline half2 dequant_fp4_pair(uint byte_val) {
    return half2(dequant_fp4_scalar(byte_val & 0xFu),
                 dequant_fp4_scalar((byte_val >> 4) & 0xFu));
}

// ============================================================================
// Vectorized FP4 dequantization: 8 values from one uint32
// ============================================================================

/// Dequantize 8 FP4 E2M1 values packed in a uint32 to 8 half values.
///
/// Packing: bits [3:0] = val0, [7:4] = val1, ..., [31:28] = val7
/// (LSB-first nibble order, matching MXFP4 / NVFP4 convention)
///
/// @param packed  8 packed FP4 nibbles in a uint32
/// @param out     Pointer to thread-local array of 8 half values
inline void dequant_fp4_x8(uint32_t packed, thread half *out) {
    // Unrolled extraction + dequant for each nibble position.
    // The Metal compiler will schedule these as independent ALU ops.
    out[0] = dequant_fp4_scalar((packed >>  0) & 0xFu);
    out[1] = dequant_fp4_scalar((packed >>  4) & 0xFu);
    out[2] = dequant_fp4_scalar((packed >>  8) & 0xFu);
    out[3] = dequant_fp4_scalar((packed >> 12) & 0xFu);
    out[4] = dequant_fp4_scalar((packed >> 16) & 0xFu);
    out[5] = dequant_fp4_scalar((packed >> 20) & 0xFu);
    out[6] = dequant_fp4_scalar((packed >> 24) & 0xFu);
    out[7] = dequant_fp4_scalar((packed >> 28) & 0xFu);
}

/// Dequantize 8 FP4 values with fused scale multiplication.
/// Returns results as two half4 vectors suitable for simdgroup_matrix ops.
///
/// @param packed   8 packed FP4 nibbles
/// @param scale    Per-group scale factor
/// @param out_lo   Output: dequantized values 0-3, scaled
/// @param out_hi   Output: dequantized values 4-7, scaled
inline void dequant_fp4_x8_scaled(uint32_t packed,
                                  half scale,
                                  thread half4 &out_lo,
                                  thread half4 &out_hi) {
    half vals[8];
    dequant_fp4_x8(packed, vals);

    out_lo = half4(vals[0], vals[1], vals[2], vals[3]) * scale;
    out_hi = half4(vals[4], vals[5], vals[6], vals[7]) * scale;
}

/// Fused dequant+scale for use inside GEMM tile loops.
/// Identical to dequant_fp4_x8_scaled but named for consistency with INT4 API.
inline void dequant_fp4_x8_fused(uint32_t packed,
                                 half scale,
                                 thread half4 &tile_lo,
                                 thread half4 &tile_hi) {
    dequant_fp4_x8_scaled(packed, scale, tile_lo, tile_hi);
}

// ============================================================================
// Vectorized FP4 dequant: paired-lane uint32 approach
// ============================================================================
//
// Instead of calling dequant_fp4_scalar 8 times independently, we process
// nibble pairs simultaneously. For each pair of nibbles we extract S/E/M,
// construct two FP16 bit patterns, pack them into a uint32, and reinterpret
// as half2. This halves the number of select() and shift+mask sequences.
//
// Vectorization benefits on Apple Silicon GPU:
//   - 4 half2 constructions vs 8 scalar calls (fewer register moves)
//   - Paired FP16 pack into uint32 enables single as_type<half2>
//   - half4 scale multiply uses SIMD FP16 FMA
//   - Better instruction-level parallelism from independent pair ops
// ============================================================================

/// Construct two FP16 values from two FP4 E2M1 nibbles, packed as half2.
/// This is the vectorized building block: given two extracted 4-bit values,
/// compute their FP16 representations and return as a single half2.
///
/// @param lo_nibble  First FP4 value (bits [3:0] used)
/// @param hi_nibble  Second FP4 value (bits [3:0] used)
/// @return           half2(fp16(lo_nibble), fp16(hi_nibble))
inline half2 dequant_fp4_pair_vec(uint lo_nibble, uint hi_nibble) {
    // Decompose lo nibble into S/E/M
    uint lo_S = (lo_nibble >> 3) & 1u;
    uint lo_E = (lo_nibble >> 1) & 3u;
    uint lo_M = lo_nibble & 1u;

    // Decompose hi nibble into S/E/M
    uint hi_S = (hi_nibble >> 3) & 1u;
    uint hi_E = (hi_nibble >> 1) & 3u;
    uint hi_M = hi_nibble & 1u;

    // Construct FP16 bit pattern for lo:
    //   Normal (E>0):    exp = E+14, mant = M<<9
    //   Subnormal (E=0,M=1): exp=14, mant=0
    //   Zero (E=0,M=0): exp=0, mant=0
    // Branchless: when E==0, exp = 14*M (14 for sub, 0 for zero), mant = 0
    bool lo_norm = (lo_E != 0u);
    uint lo_exp  = select(14u * lo_M, lo_E + 14u, lo_norm);
    uint lo_mant = select(0u, lo_M << 9, lo_norm);
    ushort lo_fp16 = ushort((lo_S << 15) | (lo_exp << 10) | lo_mant);

    // Same for hi
    bool hi_norm = (hi_E != 0u);
    uint hi_exp  = select(14u * hi_M, hi_E + 14u, hi_norm);
    uint hi_mant = select(0u, hi_M << 9, hi_norm);
    ushort hi_fp16 = ushort((hi_S << 15) | (hi_exp << 10) | hi_mant);

    // Pack two FP16 values into uint32 and reinterpret as half2
    uint32_t packed_fp16 = uint32_t(lo_fp16) | (uint32_t(hi_fp16) << 16);
    return as_type<half2>(packed_fp16);
}

/// Vectorized FP4 E2M1 dequant: processes 8 nibbles as 4 pairs.
///
/// Extracts all 8 nibbles, constructs FP16 values in pairs via
/// dequant_fp4_pair_vec, assembles into half4 vectors, and applies scale.
///
/// Drop-in replacement for dequant_fp4_x8_scaled with identical semantics.
///
/// @param packed   8 packed FP4 nibbles in a uint32 (LSB-first)
/// @param scale    Per-group scale factor
/// @param out_lo   Output: scaled values 0-3
/// @param out_hi   Output: scaled values 4-7
inline void dequant_fp4_x8_vectorized(uint32_t packed,
                                       half scale,
                                       thread half4 &out_lo,
                                       thread half4 &out_hi) {
    // Extract all 8 nibbles (compiler will optimize shared subexpressions)
    uint n0 = (packed >>  0) & 0xFu;
    uint n1 = (packed >>  4) & 0xFu;
    uint n2 = (packed >>  8) & 0xFu;
    uint n3 = (packed >> 12) & 0xFu;
    uint n4 = (packed >> 16) & 0xFu;
    uint n5 = (packed >> 20) & 0xFu;
    uint n6 = (packed >> 24) & 0xFu;
    uint n7 = (packed >> 28) & 0xFu;

    // Process 4 pairs -> 4 half2 results (8 FP16 values total)
    half2 p01 = dequant_fp4_pair_vec(n0, n1);
    half2 p23 = dequant_fp4_pair_vec(n2, n3);
    half2 p45 = dequant_fp4_pair_vec(n4, n5);
    half2 p67 = dequant_fp4_pair_vec(n6, n7);

    // Assemble into half4 and apply scale (FMA-friendly)
    out_lo = half4(p01.x, p01.y, p23.x, p23.y) * scale;
    out_hi = half4(p45.x, p45.y, p67.x, p67.y) * scale;
}

/// Vectorized FP4 dequant: raw array output (no scale), API-compatible with
/// dequant_fp4_x8 for drop-in replacement.
///
/// @param packed  8 packed FP4 nibbles in a uint32
/// @param out     Pointer to 8 thread-local half values
inline void dequant_fp4_x8_vectorized_raw(uint32_t packed, thread half *out) {
    uint n0 = (packed >>  0) & 0xFu;
    uint n1 = (packed >>  4) & 0xFu;
    uint n2 = (packed >>  8) & 0xFu;
    uint n3 = (packed >> 12) & 0xFu;
    uint n4 = (packed >> 16) & 0xFu;
    uint n5 = (packed >> 20) & 0xFu;
    uint n6 = (packed >> 24) & 0xFu;
    uint n7 = (packed >> 28) & 0xFu;

    half2 p01 = dequant_fp4_pair_vec(n0, n1);
    half2 p23 = dequant_fp4_pair_vec(n2, n3);
    half2 p45 = dequant_fp4_pair_vec(n4, n5);
    half2 p67 = dequant_fp4_pair_vec(n6, n7);

    out[0] = p01.x; out[1] = p01.y;
    out[2] = p23.x; out[3] = p23.y;
    out[4] = p45.x; out[5] = p45.y;
    out[6] = p67.x; out[7] = p67.y;
}

// ============================================================================
// Fully branchless FP4 -> FP16 (pure arithmetic, no select/ternary)
// ============================================================================

/// Fully branchless FP4 E2M1 -> FP16 conversion using arithmetic masking.
///
/// Unlike dequant_fp4_scalar which uses select(), this version computes the
/// FP16 encoding purely via integer arithmetic, avoiding even predicate
/// evaluation. Optimal for shuffle-based access patterns where every lane
/// computes a potentially different code and branch coherence cannot be assumed.
///
/// Encoding:
///   S = bit 3, E = bits [2:1], M = bit 0
///   Normal (E>0):    FP16 exp = E+14, mant = M<<9
///   Subnormal (E=0, M=1): 0.5 -> FP16 exp=14, mant=0
///   Zero (E=0, M=0): 0.0 -> all zeros (+ sign)
///
/// Branchless trick:
///   e_zero = 1 - min(E, 1)    // 1 when E==0
///   nz = min(E + M, 1)        // 1 when value != 0
///   exp = (14 + E*(1-e_zero)) * nz
///   mant = ((M*(1-e_zero)) << 9) * nz
///
/// @param nibble  4-bit FP4 value in bits [3:0]
/// @return        Equivalent half-precision value
inline half dequant_fp4_branchless(uint nibble) {
    uint S = (nibble >> 3) & 1u;
    uint E = (nibble >> 1) & 3u;
    uint M = nibble & 1u;

    // e_zero = 1 when E==0, 0 otherwise
    uint e_zero = 1u - min(E, 1u);

    // nz = 1 when value is nonzero (E|M != 0), 0 for true zero
    uint nz = min(E + M, 1u);

    // Exponent: 14 when subnormal, E+14 when normal, 0 when zero
    uint fp16_exp = (14u + E * (1u - e_zero)) * nz;

    // Mantissa: M<<9 for normals only, 0 otherwise
    uint fp16_mant = ((M * (1u - e_zero)) << 9) * nz;

    ushort fp16_bits = ushort((S << 15) | (fp16_exp << 10) | fp16_mant);
    return as_type<half>(fp16_bits);
}

// ============================================================================
// Simdgroup-cooperative FP4 dequantization
// ============================================================================

/// Simdgroup-cooperative dequant: 32 threads each load one uint32 (8 FP4 vals)
/// and dequantize locally, producing 256 FP16 values total per simdgroup.
///
/// @param packed_data  Device pointer to packed FP4 data (at least 32 uints)
/// @param lane_id      thread_index_in_simdgroup [0,31]
/// @param scale        Scale factor for this group
/// @param out          Thread-local output buffer (caller provides >= 8 halfs)
inline void dequant_fp4_simdgroup_local(device const uint32_t *packed_data,
                                        uint lane_id,
                                        half scale,
                                        thread half *out) {
    uint32_t my_packed = packed_data[lane_id];
    dequant_fp4_x8(my_packed, out);
    for (uint i = 0; i < 8; i++) {
        out[i] *= scale;
    }
}

/// Simd-shuffle based element access using dequant_fp4_scalar.
/// See dequant_fp4_shuffle_access for the branchless variant preferred
/// in GEMM tile loading.
///
/// @param my_packed     This thread's pre-loaded packed uint32
/// @param element_idx   Which of the 256 elements to retrieve [0,255]
/// @param scale         Scale factor
/// @return              Dequantized, scaled half value
inline half dequant_fp4_shuffle(uint32_t my_packed,
                                uint element_idx,
                                half scale) {
    uint source_lane = element_idx >> 3;
    uint nibble_pos = element_idx & 7u;

    uint32_t shuffled = simd_shuffle(my_packed, ushort(source_lane));
    uint nibble = (shuffled >> (nibble_pos * 4u)) & 0xFu;
    return dequant_fp4_scalar(nibble) * scale;
}

/// Simd-shuffle access using the fully branchless dequant path.
///
/// In a simdgroup of 32 threads, each lane holds one uint32 (8 FP4 values),
/// giving 256 FP16 values total. This function lets any lane retrieve any of
/// those 256 values via simd_shuffle + branchless dequant, enabling flexible
/// tile layouts without threadgroup memory staging.
///
/// Preferred over dequant_fp4_shuffle for GEMM tile loading where element
/// indices vary per-lane and branch coherence cannot be assumed.
///
/// @param my_packed     This thread's loaded uint32 (8 FP4 values)
/// @param element_idx   Which of the 256 elements to retrieve [0,255]
/// @param scale         Per-group scale factor
/// @return              Dequantized, scaled half value
inline half dequant_fp4_shuffle_access(uint32_t my_packed,
                                       uint element_idx,
                                       half scale) {
    uint source_lane = element_idx >> 3;
    uint nibble_pos = element_idx & 7u;

    uint32_t shuffled = simd_shuffle(my_packed, ushort(source_lane));
    uint nibble = (shuffled >> (nibble_pos * 4u)) & 0xFu;
    return dequant_fp4_branchless(nibble) * scale;
}

/// Load a GEMM tile row (8 elements) from the simdgroup's cooperative data.
///
/// Issues 8 shuffle-based accesses with configurable stride, enabling
/// arbitrary tile shapes from the 256-element simdgroup pool.
///
/// Example: 8x32 tile (8 rows, 32 cols = 256 elements)
///   Each of 32 lanes loads one row of 8 contiguous elements:
///     load_fp4_tile_row(my_packed, lane_id * 8, 1, scale, row)
///
/// @param my_packed   This thread's loaded uint32
/// @param start_idx   Starting element index [0,248]
/// @param stride      Stride between consecutive elements (1 = contiguous)
/// @param scale       Per-group scale
/// @param out_row     Output: 8 dequantized, scaled half values
inline void load_fp4_tile_row(uint32_t my_packed,
                              uint start_idx,
                              uint stride,
                              half scale,
                              thread half *out_row) {
    for (uint i = 0; i < 8u; i++) {
        out_row[i] = dequant_fp4_shuffle_access(my_packed,
                                                start_idx + i * stride,
                                                scale);
    }
}

/// Load a half4 tile fragment using shuffle-based branchless FP4 dequant.
/// Suitable for feeding simdgroup_matrix_storage or half4 accumulator lanes.
///
/// @param my_packed   This thread's loaded uint32
/// @param start_idx   Starting element index [0,255]
/// @param stride      Stride between elements
/// @param scale       Per-group scale
/// @return            4 dequantized, scaled values
inline half4 load_fp4_tile_half4(uint32_t my_packed,
                                 uint start_idx,
                                 uint stride,
                                 half scale) {
    return half4(
        dequant_fp4_shuffle_access(my_packed, start_idx, scale),
        dequant_fp4_shuffle_access(my_packed, start_idx + stride, scale),
        dequant_fp4_shuffle_access(my_packed, start_idx + 2u * stride, scale),
        dequant_fp4_shuffle_access(my_packed, start_idx + 3u * stride, scale)
    );
}

// ============================================================================
// Compute kernels
// ============================================================================

/// Bulk FP4 -> FP16 dequantization kernel.
///
/// Processes a 2D weight matrix stored in packed FP4 format:
///   packed_weights[k_block][n] where k_block = K/8
///
/// Each uint32 holds 8 FP4 values along the K dimension.
/// Scales are per-group along K: scales[K/group_size][N]
///
/// @param packed_weights  [K/8, N] packed FP4 weights
/// @param scales          [K/group_size, N] per-group scales
/// @param output          [K, N] dequantized FP16 output
/// @param K               Number of elements along reduction dimension
/// @param N               Number of output columns
/// @param group_size      Elements per quantization group
kernel void dequant_fp4_kernel(
    device const uint32_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device half *output                  [[buffer(2)]],
    constant uint32_t &K                 [[buffer(3)]],
    constant uint32_t &N                 [[buffer(4)]],
    constant uint32_t &group_size        [[buffer(5)]],
    uint2 gid                            [[thread_position_in_grid]]
) {
    // gid.x = column index (along N)
    // gid.y = block index along K (each block = 8 FP4 values)
    uint n_idx = gid.x;
    uint k_block = gid.y;

    if (n_idx >= N || k_block * 8u >= K) return;

    // Load packed weights
    uint packed_idx = k_block * N + n_idx;
    uint32_t packed = packed_weights[packed_idx];

    // Load scale for this group
    uint k_start = k_block * 8u;
    uint group_idx = k_start / group_size;
    half scale = scales[group_idx * N + n_idx];

    // Dequantize
    half vals[8];
    dequant_fp4_x8(packed, vals);

    // Write output (column-major within the K dimension)
    uint out_base = k_start * N + n_idx;
    uint k_remain = min(8u, K - k_start);
    for (uint i = 0; i < k_remain; i++) {
        output[out_base + i * N] = vals[i] * scale;
    }
}

/// High-throughput FP4 dequant kernel using half4 vectorized stores.
/// Requires that K is a multiple of 8 and output is half4-aligned.
kernel void dequant_fp4_aligned_kernel(
    device const uint32_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device half4 *output                 [[buffer(2)]],
    constant uint32_t &num_packed        [[buffer(3)]],
    constant uint32_t &N                 [[buffer(4)]],
    constant uint32_t &group_size        [[buffer(5)]],
    uint tid                             [[thread_position_in_grid]]
) {
    if (tid >= num_packed) return;

    uint32_t packed = packed_weights[tid];

    // Linear index -> 2D position
    uint n_idx = tid % N;
    uint k_block = tid / N;
    uint k_start = k_block * 8u;
    uint group_idx = k_start / group_size;
    half scale = scales[group_idx * N + n_idx];

    half4 lo, hi;
    dequant_fp4_x8_scaled(packed, scale, lo, hi);

    output[tid * 2u] = lo;
    output[tid * 2u + 1u] = hi;
}

// ============================================================================
// Bulk dequantization kernels (standalone, not fused into GEMM)
// ============================================================================
//
// These kernels dequantize an entire packed weight matrix to FP16 in one
// dispatch. Useful for debugging, validation, or feeding dequantized weights
// into non-GEMM operations (e.g., normalization, element-wise ops).
//
// Layout convention:
//   packed: [K/8, N] — each uint32 holds 8 values along K
//   scales: [K/group_size, N] — one scale per group per column
//   zeros:  [K/group_size, N] — one zero-point per group per column (INT4)
//   output: [K, N] — full FP16 weight matrix
//
// The 2D grid maps (gid.x, gid.y) = (column index, K-block index).

/// Bulk FP4 dequant: [K/8, N] packed -> [K, N] FP16
///
/// Each thread processes one packed uint32 (8 FP4 values along K) for one
/// column, writing 8 consecutive K-rows of that column.
kernel void dequant_fp4_bulk(
    device const uint32_t* packed [[buffer(0)]],
    device const half* scales     [[buffer(1)]],
    device half* output           [[buffer(2)]],
    constant uint& K              [[buffer(3)]],
    constant uint& N              [[buffer(4)]],
    constant uint& group_size     [[buffer(5)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint n_idx = gid.x;     // column
    uint k_block = gid.y;   // which group of 8 K-elements

    uint k_start = k_block * 8u;
    if (n_idx >= N || k_start >= K) return;

    // Load packed word: row-major [K/8, N]
    uint packed_idx = k_block * N + n_idx;
    uint32_t word = packed[packed_idx];

    // Determine quantization group and load scale
    uint group_idx = k_start / group_size;
    half scale = scales[group_idx * N + n_idx];

    // Dequantize 8 FP4 values
    half vals[8];
    dequant_fp4_x8(word, vals);

    // Write to output [K, N] with boundary check on K
    uint k_remain = min(8u, K - k_start);
    uint out_base = k_start * N + n_idx;
    for (uint i = 0; i < k_remain; i++) {
        output[out_base + i * N] = vals[i] * scale;
    }
}

/// Bulk INT4 dequant with zero points: [K/8, N] packed -> [K, N] FP16
///
/// Each thread processes one packed uint32 (8 INT4 values along K) for one
/// column, applying asymmetric dequantization:
///   output = (int4_val - zero_point) * scale
///
/// Uses the magic number trick (same as dequant_u4x8) for branchless INT4->FP16
/// conversion. Zero points are in the unsigned domain [0,15].
kernel void dequant_int4_bulk(
    device const uint32_t* packed [[buffer(0)]],
    device const half* scales     [[buffer(1)]],
    device const half* zeros      [[buffer(2)]],
    device half* output           [[buffer(3)]],
    constant uint& K              [[buffer(4)]],
    constant uint& N              [[buffer(5)]],
    constant uint& group_size     [[buffer(6)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint n_idx = gid.x;     // column
    uint k_block = gid.y;   // which group of 8 K-elements

    uint k_start = k_block * 8u;
    if (n_idx >= N || k_start >= K) return;

    // Load packed word: row-major [K/8, N]
    uint packed_idx = k_block * N + n_idx;
    uint32_t word = packed[packed_idx];

    // Determine quantization group and load scale + zero
    uint group_idx = k_start / group_size;
    uint param_idx = group_idx * N + n_idx;
    half scale = scales[param_idx];
    half zero_point = zeros[param_idx];

    // Dequantize 8 U4 values using magic number trick with asymmetric zero-point
    half4 lo, hi;
    dequant_u4x8(word, scale, zero_point, lo, hi);

    // Write to output [K, N] with boundary check on K
    uint k_remain = min(8u, K - k_start);
    uint out_base = k_start * N + n_idx;

    // Unrolled write for the common aligned case
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
}

// ============================================================================
// Unit test kernels
// ============================================================================

/// Test kernel: dequantize all 16 U4 values (0-15) without scaling.
/// Uses the magic bias trick. Writes 16 half values to output buffer.
kernel void test_u4_all_codes(
    device half *output [[buffer(0)]],
    uint tid           [[thread_position_in_grid]]
) {
    if (tid >= 16u) return;
    output[tid] = dequant_u4_scalar(uint16_t(tid));
}

/// Test kernel: dequantize all 16 FP4 codes (0x0 through 0xF) without scaling.
/// Writes 16 half values to output buffer for host-side verification.
kernel void test_fp4_all_codes(
    device half *output [[buffer(0)]],
    uint tid           [[thread_position_in_grid]]
) {
    if (tid >= 16u) return;
    output[tid] = dequant_fp4_scalar(tid);
}

/// Test kernel: dequantize a packed uint32 (8 FP4 values) with a given scale.
/// Single-threaded for deterministic testing.
kernel void test_fp4_packed_scaled(
    device const uint32_t *packed_input [[buffer(0)]],
    device const half *scale           [[buffer(1)]],
    device half *output                [[buffer(2)]],
    uint tid                           [[thread_position_in_grid]]
) {
    if (tid > 0u) return;

    half4 lo, hi;
    dequant_fp4_x8_scaled(packed_input[0], scale[0], lo, hi);

    output[0] = lo.x;
    output[1] = lo.y;
    output[2] = lo.z;
    output[3] = lo.w;
    output[4] = hi.x;
    output[5] = hi.y;
    output[6] = hi.z;
    output[7] = hi.w;
}

/// Test kernel: verify simd_shuffle-based dequant.
/// Each thread in the simdgroup retrieves a specific element index.
kernel void test_fp4_simdgroup_shuffle(
    device const uint32_t *packed_data [[buffer(0)]],
    device const half *scale          [[buffer(1)]],
    device half *output               [[buffer(2)]],
    constant uint32_t &num_elements   [[buffer(3)]],
    uint lane_id                      [[thread_index_in_simdgroup]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= num_elements) return;

    // Each thread loads its own packed uint (for shuffle source)
    uint32_t my_packed = packed_data[lane_id];

    // Retrieve element at index=tid via shuffle
    output[tid] = dequant_fp4_shuffle(my_packed, tid, scale[0]);
}

/// Test kernel: verify dequant_fp4_branchless matches dequant_fp4_scalar.
/// Dequantizes all 16 codes via branchless path for comparison.
kernel void test_fp4_branchless_all_codes(
    device half *output [[buffer(0)]],
    uint tid           [[thread_position_in_grid]]
) {
    if (tid >= 16u) return;
    output[tid] = dequant_fp4_branchless(tid);
}

/// Test kernel: verify shuffle_access (branchless) path.
/// Each thread in the simdgroup retrieves a specific element index using
/// the branchless dequant, for comparison against the scalar shuffle path.
kernel void test_fp4_simdgroup_shuffle_access(
    device const uint32_t *packed_data [[buffer(0)]],
    device const half *scale          [[buffer(1)]],
    device half *output               [[buffer(2)]],
    constant uint32_t &num_elements   [[buffer(3)]],
    uint lane_id                      [[thread_index_in_simdgroup]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= num_elements) return;

    uint32_t my_packed = packed_data[lane_id];
    output[tid] = dequant_fp4_shuffle_access(my_packed, tid, scale[0]);
}

/// Test kernel: verify load_fp4_tile_row for tile loading patterns.
/// Dispatches exactly 32 threads (one simdgroup). Each lane loads its
/// own row of 8 contiguous elements from the 256-element pool.
/// Output: [32][8] = 256 values, should match linear dequant order.
kernel void test_fp4_tile_row(
    device const uint32_t *packed_data [[buffer(0)]],
    device const half *scale          [[buffer(1)]],
    device half *output               [[buffer(2)]],
    uint lane_id                      [[thread_index_in_simdgroup]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= 32u) return;

    uint32_t my_packed = packed_data[lane_id];

    half row[8];
    load_fp4_tile_row(my_packed, lane_id * 8u, 1u, scale[0], row);

    uint out_base = lane_id * 8u;
    for (uint i = 0; i < 8u; i++) {
        output[out_base + i] = row[i];
    }
}

/// Test kernel: verify vectorized FP4 dequant matches scalar path.
/// Single-threaded, dequantizes one packed uint32 with the vectorized path.
kernel void test_fp4_vectorized(
    device const uint32_t *packed_input [[buffer(0)]],
    device const half *scale           [[buffer(1)]],
    device half *output                [[buffer(2)]],
    uint tid                           [[thread_position_in_grid]]
) {
    if (tid > 0u) return;

    half4 lo, hi;
    dequant_fp4_x8_vectorized(packed_input[0], scale[0], lo, hi);

    output[0] = lo.x;
    output[1] = lo.y;
    output[2] = lo.z;
    output[3] = lo.w;
    output[4] = hi.x;
    output[5] = hi.y;
    output[6] = hi.z;
    output[7] = hi.w;
}

// ============================================================================
// Benchmark kernels: scalar vs vectorized FP4 dequant throughput
// ============================================================================
//
// These kernels process a large buffer of packed FP4 data using either the
// scalar (dequant_fp4_x8) or vectorized (dequant_fp4_x8_vectorized) path.
// Each thread processes ITERS_PER_THREAD packed uint32 values to amortize
// dispatch overhead and measure sustained ALU throughput.
//
// Usage: dispatch num_packed/ITERS_PER_THREAD threads, compare GPU timestamps.
// ============================================================================

constant constexpr uint BENCH_ITERS_PER_THREAD = 16u;

/// Benchmark: scalar FP4 dequant path (calls dequant_fp4_x8 per packed word).
/// Each thread processes BENCH_ITERS_PER_THREAD consecutive packed uint32 values.
kernel void bench_fp4_scalar(
    device const uint32_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device half4 *output                 [[buffer(2)]],
    constant uint32_t &num_packed        [[buffer(3)]],
    constant uint32_t &group_size        [[buffer(4)]],
    uint tid                             [[thread_position_in_grid]]
) {
    uint base = tid * BENCH_ITERS_PER_THREAD;
    if (base >= num_packed) return;

    uint iters = min(BENCH_ITERS_PER_THREAD, num_packed - base);

    for (uint i = 0; i < iters; i++) {
        uint idx = base + i;
        uint32_t packed = packed_weights[idx];
        uint group_idx = (idx * 8u) / group_size;
        half scale = scales[group_idx];

        // Scalar path: call dequant_fp4_x8 then construct half4s
        half vals[8];
        dequant_fp4_x8(packed, vals);

        half4 lo = half4(vals[0], vals[1], vals[2], vals[3]) * scale;
        half4 hi = half4(vals[4], vals[5], vals[6], vals[7]) * scale;

        output[idx * 2u]     = lo;
        output[idx * 2u + 1u] = hi;
    }
}

/// Benchmark: vectorized FP4 dequant path (paired-lane half2 construction).
/// Each thread processes BENCH_ITERS_PER_THREAD consecutive packed uint32 values.
kernel void bench_fp4_vectorized(
    device const uint32_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device half4 *output                 [[buffer(2)]],
    constant uint32_t &num_packed        [[buffer(3)]],
    constant uint32_t &group_size        [[buffer(4)]],
    uint tid                             [[thread_position_in_grid]]
) {
    uint base = tid * BENCH_ITERS_PER_THREAD;
    if (base >= num_packed) return;

    uint iters = min(BENCH_ITERS_PER_THREAD, num_packed - base);

    for (uint i = 0; i < iters; i++) {
        uint idx = base + i;
        uint32_t packed = packed_weights[idx];
        uint group_idx = (idx * 8u) / group_size;
        half scale = scales[group_idx];

        half4 lo, hi;
        dequant_fp4_x8_vectorized(packed, scale, lo, hi);

        output[idx * 2u]     = lo;
        output[idx * 2u + 1u] = hi;
    }
}

// ============================================================================
// High-Performance FP4 Dequantization Kernels for Apple Silicon
// ============================================================================
//
// Target: >400 GB/s effective bandwidth on M4 Max (theoretical 546 GB/s)
//
// Key optimizations:
//   1. Simdgroup-cooperative loads: 32 threads load together for coalescing
//   2. 128-byte aligned vectorized loads: uint4 (16 bytes) per thread
//   3. Threadgroup memory for scale factor caching
//   4. Vectorized half4 stores for coalesced writes
//   5. Multiple elements per thread to amortize overhead
//   6. LUT-based dequantization in threadgroup memory (faster than ALU)
//
// Memory hierarchy exploitation:
//   - L1 cache: 192KB on M4 Max, ~2TB/s bandwidth
//   - L2 cache: 48MB on M4 Max, ~600GB/s bandwidth
//   - DRAM: 546 GB/s theoretical, ~500GB/s achievable
//
// The kernel is memory-bound: 4 bits in -> 16 bits out = 4x expansion.
// Read BW needed for 400 GB/s output = 100 GB/s (easily achievable).
// ============================================================================

// Constants for optimal kernel
constant constexpr uint OPT_SIMDGROUP_SIZE = 32u;
constant constexpr uint OPT_SIMDGROUPS_PER_TG = 4u;
constant constexpr uint OPT_THREADS_PER_TG = OPT_SIMDGROUP_SIZE * OPT_SIMDGROUPS_PER_TG;  // 128

// Each thread processes 4 packed uint32s (32 FP4 values = 64 bytes output)
// This gives 128 threads * 32 values = 4096 values per threadgroup
constant constexpr uint OPT_PACKS_PER_THREAD = 4u;
constant constexpr uint OPT_VALUES_PER_THREAD = OPT_PACKS_PER_THREAD * 8u;  // 32
constant constexpr uint OPT_VALUES_PER_TG = OPT_THREADS_PER_TG * OPT_VALUES_PER_THREAD;  // 4096

// Scale cache: one scale per group, max 4096/32 = 128 groups for group_size=32
constant constexpr uint OPT_MAX_SCALE_CACHE = 128u;

// FP4 E2M1 lookup table in constant memory (16 entries)
// Values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0}
constant half FP4_LUT_CONST[16] = {
    half(0.0h),  half(0.5h),  half(1.0h),  half(1.5h),
    half(2.0h),  half(3.0h),  half(4.0h),  half(6.0h),
    half(-0.0h), half(-0.5h), half(-1.0h), half(-1.5h),
    half(-2.0h), half(-3.0h), half(-4.0h), half(-6.0h)
};

/// LUT-based FP4 dequantization - faster than bitwise for standalone dequant.
/// Uses constant memory LUT which is cached in L1.
inline half dequant_fp4_lut(uint nibble) {
    return FP4_LUT_CONST[nibble & 0xFu];
}

/// Dequant 8 FP4 values using LUT, output to half4 pair.
inline void dequant_fp4_lut_x8(uint32_t packed, half scale,
                               thread half4 &out_lo, thread half4 &out_hi) {
    out_lo = half4(
        FP4_LUT_CONST[(packed >>  0) & 0xFu],
        FP4_LUT_CONST[(packed >>  4) & 0xFu],
        FP4_LUT_CONST[(packed >>  8) & 0xFu],
        FP4_LUT_CONST[(packed >> 12) & 0xFu]
    ) * scale;
    out_hi = half4(
        FP4_LUT_CONST[(packed >> 16) & 0xFu],
        FP4_LUT_CONST[(packed >> 20) & 0xFu],
        FP4_LUT_CONST[(packed >> 24) & 0xFu],
        FP4_LUT_CONST[(packed >> 28) & 0xFu]
    ) * scale;
}

// ============================================================================
// dequant_fp4_optimal - Maximum bandwidth FP4 dequantization
// ============================================================================
//
// Optimized for bulk dequantization of 2D weight matrices [K, N].
// Each threadgroup processes a tile of the matrix, with scale factors
// cached in threadgroup memory for reuse.
//
// Layout:
//   packed_weights: [K/8, N] - each uint32 holds 8 FP4 values along K
//   scales: [K/group_size, N] - per-group scales
//   output: [K, N] - dequantized FP16 output
//
// Grid: 2D dispatch where each threadgroup handles a tile of output.
// Threadgroup size: 128 threads (4 simdgroups)
//
// Memory access pattern optimized for coalescing:
//   - Adjacent threads read adjacent packed weights (coalesced)
//   - Each thread writes 32 consecutive FP16 values (coalesced half4 stores)
// ============================================================================

kernel void dequant_fp4_optimal(
    device const uint32_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device half4 *output                 [[buffer(2)]],
    constant uint32_t &K                 [[buffer(3)]],
    constant uint32_t &N                 [[buffer(4)]],
    constant uint32_t &group_size        [[buffer(5)]],
    uint2 tgid                           [[threadgroup_position_in_grid]],
    uint tid                             [[thread_index_in_threadgroup]],
    uint simd_lane                       [[thread_index_in_simdgroup]],
    uint simd_id                         [[simdgroup_index_in_threadgroup]]
) {
    // Threadgroup handles a tile: OPT_VALUES_PER_TG elements
    // Mapping: tgid.x = N-tile, tgid.y = K-tile

    // Output tile base indices
    const uint k_tile = tgid.y * OPT_VALUES_PER_TG;  // Starting K index for this tile
    const uint n_col = tgid.x;                       // Column index (one col per tgid.x)

    if (n_col >= N) return;

    // Each thread processes OPT_PACKS_PER_THREAD consecutive packed values
    // Thread layout: threads are spread across K dimension
    const uint thread_k_base = k_tile + tid * OPT_VALUES_PER_THREAD;

    // Process OPT_PACKS_PER_THREAD packed words (each = 8 FP4 values)
    #pragma unroll
    for (uint p = 0; p < OPT_PACKS_PER_THREAD; ++p) {
        uint k_start = thread_k_base + p * 8u;

        // Boundary check
        if (k_start >= K) continue;

        // Load packed weight: row = k_start/8, col = n_col
        uint pack_row = k_start / 8u;
        uint32_t packed = packed_weights[pack_row * N + n_col];

        // Load scale for this group
        uint group_idx = k_start / group_size;
        half scale = scales[group_idx * N + n_col];

        // Dequantize using LUT
        half4 lo, hi;
        dequant_fp4_lut_x8(packed, scale, lo, hi);

        // Compute output indices (2 half4s per packed word)
        // Output layout: [K, N], so output[k * N + n] for element (k, n)
        // As half4: output[(k * N + n) / 4] but we need to handle stride
        //
        // For [K, N] layout with half4 stores, we need K-major access.
        // Each half4 store writes 4 consecutive K values for the same N.

        uint out_idx_lo = (k_start * N + n_col) / 4u;
        uint out_idx_hi = ((k_start + 4u) * N + n_col) / 4u;

        // Check if half4 store is aligned (N must divide evenly for half4 stores)
        // For non-aligned cases, fall back to scalar writes
        if (k_start + 8u <= K) {
            if ((k_start * N + n_col) % 4u == 0u) {
                output[out_idx_lo] = lo;
            } else {
                // Scalar fallback for unaligned
                device half *out_scalar = (device half *)output;
                out_scalar[(k_start + 0u) * N + n_col] = lo.x;
                out_scalar[(k_start + 1u) * N + n_col] = lo.y;
                out_scalar[(k_start + 2u) * N + n_col] = lo.z;
                out_scalar[(k_start + 3u) * N + n_col] = lo.w;
            }

            if (((k_start + 4u) * N + n_col) % 4u == 0u) {
                output[out_idx_hi] = hi;
            } else {
                device half *out_scalar = (device half *)output;
                out_scalar[(k_start + 4u) * N + n_col] = hi.x;
                out_scalar[(k_start + 5u) * N + n_col] = hi.y;
                out_scalar[(k_start + 6u) * N + n_col] = hi.z;
                out_scalar[(k_start + 7u) * N + n_col] = hi.w;
            }
        } else {
            // Boundary handling: write remaining elements
            device half *out_scalar = (device half *)output;
            uint k_remain = K - k_start;
            half vals[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
            for (uint i = 0; i < k_remain && i < 8u; ++i) {
                out_scalar[(k_start + i) * N + n_col] = vals[i];
            }
        }
    }
}

// ============================================================================
// dequant_fp4_optimal_rowmajor - Optimized for row-major output layout
// ============================================================================
//
// For output layout [K, N] where we want coalesced writes along N dimension.
// Each threadgroup handles multiple K-rows, with threads spread across N.
//
// This is the preferred kernel when the output will be used in row-major
// access patterns (e.g., feeding into a GEMM where N is the fast dimension).
//
// Grid: 1D dispatch, threadgroups tile the packed array linearly.
// ============================================================================

// Tile size for row-major kernel: process 8 K-rows at a time for scale reuse
constant constexpr uint ROWMAJOR_K_TILE = 8u;  // 8 K values = 1 packed word per N
constant constexpr uint ROWMAJOR_N_TILE = 128u; // Threads spread across N

kernel void dequant_fp4_optimal_rowmajor(
    device const uint32_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device half *output                  [[buffer(2)]],
    constant uint32_t &K                 [[buffer(3)]],
    constant uint32_t &N                 [[buffer(4)]],
    constant uint32_t &group_size        [[buffer(5)]],
    uint2 tgid                           [[threadgroup_position_in_grid]],
    uint tid                             [[thread_index_in_threadgroup]],
    uint simd_lane                       [[thread_index_in_simdgroup]],
    uint simd_id                         [[simdgroup_index_in_threadgroup]]
) {
    // tgid.x = N tile index, tgid.y = K block index (groups of 8)
    const uint n_base = tgid.x * ROWMAJOR_N_TILE + tid;
    const uint k_block = tgid.y;
    const uint k_start = k_block * 8u;

    if (n_base >= N || k_start >= K) return;

    // Load packed weight for this N position
    uint pack_idx = k_block * N + n_base;
    uint32_t packed = packed_weights[pack_idx];

    // Load scale
    uint group_idx = k_start / group_size;
    half scale = scales[group_idx * N + n_base];

    // Dequantize
    half4 lo, hi;
    dequant_fp4_lut_x8(packed, scale, lo, hi);

    // Write output: 8 consecutive K values for this N column
    uint k_remain = min(8u, K - k_start);
    uint out_base = k_start * N + n_base;

    // Unrolled writes for the common case
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
        for (uint i = 0; i < k_remain; ++i) {
            output[out_base + i * N] = vals[i];
        }
    }
}

// ============================================================================
// dequant_fp4_simdgroup_optimal - Maximum throughput with simdgroup cooperation
// ============================================================================
//
// This kernel achieves maximum memory bandwidth by:
//   1. Each simdgroup (32 threads) loads 32 consecutive packed words together
//   2. Vectorized uint4 loads for 128-byte cache line utilization
//   3. Scale factors cached in threadgroup memory
//   4. Coalesced half4 stores across the simdgroup
//
// Layout (linear, for maximum bandwidth measurement):
//   packed_weights: [num_packed] - linear array of packed FP4
//   scales: [num_groups] - linear array of scales
//   output: [num_packed * 8] - linear FP16 output
//
// Dispatch: 1D grid, threadgroups of 128 threads
// ============================================================================

// Threadgroup scale cache
constant constexpr uint SIMD_OPT_THREADS = 128u;
constant constexpr uint SIMD_OPT_PACKS_PER_THREAD = 4u;
constant constexpr uint SIMD_OPT_PACKS_PER_TG = SIMD_OPT_THREADS * SIMD_OPT_PACKS_PER_THREAD;  // 512

kernel void dequant_fp4_simdgroup_optimal(
    device const uint32_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device half4 *output                 [[buffer(2)]],
    constant uint32_t &num_packed        [[buffer(3)]],
    constant uint32_t &group_size        [[buffer(4)]],
    uint tgid                            [[threadgroup_position_in_grid]],
    uint tid                             [[thread_index_in_threadgroup]],
    uint simd_lane                       [[thread_index_in_simdgroup]],
    uint simd_id                         [[simdgroup_index_in_threadgroup]]
) {
    // Base index for this threadgroup
    const uint tg_base = tgid * SIMD_OPT_PACKS_PER_TG;

    // Each thread processes SIMD_OPT_PACKS_PER_THREAD consecutive packs
    const uint thread_base = tg_base + tid * SIMD_OPT_PACKS_PER_THREAD;

    // Load 4 packed words using vectorized load (uint4 = 16 bytes = 128 bits)
    // This achieves optimal memory coalescing when threads access consecutively
    if (thread_base + 3u < num_packed) {
        // Fast path: load 4 uint32s together
        device const uint4 *packed_vec = (device const uint4 *)&packed_weights[thread_base];
        uint4 p4 = packed_vec[0];

        #pragma unroll
        for (uint i = 0; i < 4u; ++i) {
            uint idx = thread_base + i;
            uint32_t packed = (i == 0) ? p4.x : ((i == 1) ? p4.y : ((i == 2) ? p4.z : p4.w));

            // Calculate group index for scale
            uint group_idx = (idx * 8u) / group_size;
            half scale = scales[group_idx];

            // Dequantize
            half4 lo, hi;
            dequant_fp4_lut_x8(packed, scale, lo, hi);

            // Write output (2 half4s per packed word)
            uint out_idx = idx * 2u;
            output[out_idx]     = lo;
            output[out_idx + 1u] = hi;
        }
    } else {
        // Boundary handling
        for (uint i = 0; i < SIMD_OPT_PACKS_PER_THREAD; ++i) {
            uint idx = thread_base + i;
            if (idx >= num_packed) break;

            uint32_t packed = packed_weights[idx];
            uint group_idx = (idx * 8u) / group_size;
            half scale = scales[group_idx];

            half4 lo, hi;
            dequant_fp4_lut_x8(packed, scale, lo, hi);

            uint out_idx = idx * 2u;
            output[out_idx]     = lo;
            output[out_idx + 1u] = hi;
        }
    }
}

// ============================================================================
// dequant_fp4_bandwidth_max - Absolute maximum bandwidth kernel
// ============================================================================
//
// Stripped-down kernel for bandwidth benchmarking. No boundary checks,
// assumes aligned inputs. Uses maximum vectorization.
//
// Each thread loads uint4 (4 packed words = 32 FP4 values) and writes
// 8 half4s (also 128 bits = 16 bytes).
//
// Requirements:
//   - num_packed must be multiple of 512 (threads * 4 packs/thread)
//   - All buffers must be 16-byte aligned
//   - group_size must divide 8 evenly (typical: 32, 64, 128)
//
// Dispatch: 1D grid, threadgroups of 128 threads
// ============================================================================

kernel void dequant_fp4_bandwidth_max(
    device const uint4 *packed_weights   [[buffer(0)]],  // Pre-cast to uint4
    device const half *scales            [[buffer(1)]],
    device half4 *output                 [[buffer(2)]],
    constant uint32_t &num_packed_div4   [[buffer(3)]],  // num_packed / 4
    constant uint32_t &group_size        [[buffer(4)]],
    uint tgid                            [[threadgroup_position_in_grid]],
    uint tid                             [[thread_index_in_threadgroup]]
) {
    // Linear thread index
    const uint gid = tgid * SIMD_OPT_THREADS + tid;
    if (gid >= num_packed_div4) return;

    // Load 4 packed words at once (128-bit load)
    uint4 p4 = packed_weights[gid];

    // Base output index (each uint4 produces 8 half4s)
    uint out_base = gid * 8u;

    // Process all 4 packed words
    #pragma unroll
    for (uint i = 0; i < 4u; ++i) {
        uint packed_idx = gid * 4u + i;
        uint32_t packed = (i == 0) ? p4.x : ((i == 1) ? p4.y : ((i == 2) ? p4.z : p4.w));

        // Scale lookup (group_idx = (packed_idx * 8) / group_size)
        uint group_idx = (packed_idx * 8u) / group_size;
        half scale = scales[group_idx];

        // LUT dequant
        half4 lo = half4(
            FP4_LUT_CONST[(packed >>  0) & 0xFu],
            FP4_LUT_CONST[(packed >>  4) & 0xFu],
            FP4_LUT_CONST[(packed >>  8) & 0xFu],
            FP4_LUT_CONST[(packed >> 12) & 0xFu]
        ) * scale;
        half4 hi = half4(
            FP4_LUT_CONST[(packed >> 16) & 0xFu],
            FP4_LUT_CONST[(packed >> 20) & 0xFu],
            FP4_LUT_CONST[(packed >> 24) & 0xFu],
            FP4_LUT_CONST[(packed >> 28) & 0xFu]
        ) * scale;

        // Write 2 half4s
        output[out_base + i * 2u]     = lo;
        output[out_base + i * 2u + 1u] = hi;
    }
}

// ============================================================================
// Test kernel for the optimal dequant path
// ============================================================================

/// Test kernel: verify dequant_fp4_lut matches dequant_fp4_scalar.
kernel void test_fp4_lut_all_codes(
    device half *output [[buffer(0)]],
    uint tid           [[thread_position_in_grid]]
) {
    if (tid >= 16u) return;
    output[tid] = dequant_fp4_lut(tid);
}

/// Test kernel: verify dequant_fp4_lut_x8 produces correct output.
kernel void test_fp4_lut_x8(
    device const uint32_t *packed_input [[buffer(0)]],
    device const half *scale           [[buffer(1)]],
    device half *output                [[buffer(2)]],
    uint tid                           [[thread_position_in_grid]]
) {
    if (tid > 0u) return;

    half4 lo, hi;
    dequant_fp4_lut_x8(packed_input[0], scale[0], lo, hi);

    output[0] = lo.x;
    output[1] = lo.y;
    output[2] = lo.z;
    output[3] = lo.w;
    output[4] = hi.x;
    output[5] = hi.y;
    output[6] = hi.z;
    output[7] = hi.w;
}
