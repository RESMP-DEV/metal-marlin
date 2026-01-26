#include <metal_stdlib>
using namespace metal;

#pragma metal performance_hint(fast_math)

// ============================================================================
// BFloat16 (BF16) Compatibility Layer for Apple Metal
// ============================================================================
//
// BFloat16 format: [1 sign][8 exponent (bias=127)][7 mantissa]
//
// BF16 matches the top 16 bits of IEEE 754 FP32, providing the same dynamic
// range (8-bit exponent) as FP32 but with reduced precision (7 mantissa bits
// vs FP32's 23). This makes it ideal for training activations where range
// matters more than precision.
//
// Comparison of 16-bit formats:
//   FP16:  [1 sign][5 exp (bias=15)][10 mantissa]  range: ~6e-8 to 65504
//   BF16:  [1 sign][8 exp (bias=127)][7 mantissa]  range: ~1e-38 to ~3.4e38
//
// BF16 advantages for activations:
//   - Same exponent range as FP32: no overflow during training
//   - FP32 <-> BF16 conversion is trivial bit truncation (no rescaling)
//   - Loss landscape exploration benefits from wide dynamic range
//
// Metal does not natively support bfloat16 as of Metal 3.1 (Apple Silicon M1-M4).
// Future M-series chips may add hardware support. This file provides a software
// implementation using uint16 storage with explicit conversions.
//
// ============================================================================

// ============================================================================
// BF16 struct: software emulation via uint16 storage
// ============================================================================

#if __METAL_VERSION__ >= 310 && defined(__METAL_BFLOAT__)
// Hypothetical future Metal version with native bfloat support
typedef bfloat bf16;
typedef bfloat bf16_t;

#else

/// Software BF16 type using uint16 storage.
/// Conversion to/from float is a simple 16-bit shift (truncation for float->bf16,
/// zero-extension for bf16->float). This mirrors the hardware behavior of BF16
/// units on NVIDIA/AMD GPUs.
struct bf16 {
    ushort bits;

    bf16() = default;

    /// Construct BF16 from float via truncation (round-toward-zero).
    /// Drops the low 16 mantissa bits of FP32.
    ///
    /// For round-to-nearest-even, use bf16_from_float_rne() instead.
    bf16(float f) {
        bits = ushort(as_type<uint>(f) >> 16);
    }

    /// Convert BF16 back to float by zero-extending the low 16 bits.
    /// This is exact (no precision loss in this direction).
    operator float() const {
        return as_type<float>(uint(bits) << 16);
    }
};

typedef bf16 bf16_t;

#endif

// ============================================================================
// Rounding modes for FP32 -> BF16 conversion
// ============================================================================

/// Convert FP32 to BF16 with round-toward-zero (truncation).
/// Fastest conversion; introduces a systematic negative bias for positive values.
///
/// @param f  FP32 value
/// @return   BF16 representation
inline bf16_t bf16_from_float_rtz(float f) {
    bf16_t result;
    result.bits = ushort(as_type<uint>(f) >> 16);
    return result;
}

/// Convert FP32 to BF16 with round-to-nearest-even (RNE).
/// Adds the rounding bias (bit 15 of the FP32 representation) plus a
/// tie-breaking adjustment for the round-to-even rule.
///
/// This matches the rounding behavior of hardware BF16 units.
///
/// @param f  FP32 value
/// @return   BF16 representation (correctly rounded)
inline bf16_t bf16_from_float_rne(float f) {
    uint f_bits = as_type<uint>(f);

    // Extract the 8 exponent bits to check for special values
    uint exp_bits = (f_bits >> 23) & 0xFFu;

    bf16_t result;

    // NaN: preserve sign and set quiet NaN in BF16
    // Inf: truncation already produces correct BF16 inf
    if (exp_bits == 0xFFu) {
        // NaN: ensure we produce a quiet NaN (set MSB of mantissa)
        uint mantissa = f_bits & 0x007FFFFFu;
        if (mantissa != 0u) {
            // Input is NaN; make it quiet NaN in BF16
            result.bits = ushort((f_bits >> 16) | 0x0040u);
        } else {
            // Input is Inf; truncation is correct
            result.bits = ushort(f_bits >> 16);
        }
        return result;
    }

    // Round-to-nearest-even:
    // Add rounding bias: bit 15 (round bit) plus an adjustment for tie-breaking.
    // If the round bit is 1 and all lower bits are 0 (exact tie), we round to
    // even by adding the LSB of the BF16 result (bit 16 of the FP32 word).
    uint round_bit = 1u << 15;
    uint lsb_bit = 1u << 16;

    // Rounding constant: round_bit + tie-breaker
    // The tie-breaker adds 1 only when the result's LSB is 1 (making it even)
    uint rounding = round_bit + ((f_bits >> 16) & 1u);

    // Add rounding and truncate. Overflow in mantissa propagates correctly
    // through exponent (incrementing it), matching IEEE behavior.
    f_bits += rounding;
    result.bits = ushort(f_bits >> 16);
    return result;
}

/// Convert FP32 to BF16 with stochastic rounding.
/// Uses a random value to determine whether to round up or down, providing
/// unbiased rounding in expectation. Useful for training where accumulated
/// rounding bias can affect convergence.
///
/// @param f     FP32 value
/// @param rand  16-bit random value for stochastic decision
/// @return      BF16 representation (stochastically rounded)
inline bf16_t bf16_from_float_stochastic(float f, ushort rand) {
    uint f_bits = as_type<uint>(f);
    uint truncated_bits = f_bits & 0xFFFFu;

    // If the truncated portion exceeds the random threshold, round up
    if (truncated_bits > uint(rand)) {
        f_bits += 0x10000u;  // Increment BF16 mantissa by 1 ULP
    }

    bf16_t result;
    result.bits = ushort(f_bits >> 16);
    return result;
}

// ============================================================================
// BF16 -> FP16 and BF16 -> FP32 conversion
// ============================================================================

/// Convert BF16 to FP32 (exact, zero-extension).
///
/// @param v  BF16 value
/// @return   Equivalent FP32 value (no precision loss)
inline float bf16_to_float(bf16_t v) {
    return float(v);
}

/// Convert raw BF16 bits (uint16 storage) to FP32.
inline float bf16_bits_to_float(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

/// Convert BF16 to FP16 (half).
/// May lose precision (BF16 has 7 mantissa bits, FP16 has 10, but BF16 has
/// much wider exponent range). Values outside FP16 range will saturate to
/// +/-inf or flush to zero.
///
/// @param v  BF16 value
/// @return   Closest FP16 representation
inline half bf16_to_half(bf16_t v) {
    return half(float(v));
}

/// Convert FP16 to BF16.
/// FP16 values are always exactly representable in BF16's exponent range
/// (FP16 exp range is a subset), but the 10-bit mantissa is truncated to 7 bits.
///
/// @param h  FP16 value
/// @return   BF16 representation (may lose mantissa precision)
inline bf16_t bf16_from_half(half h) {
    return bf16_t(float(h));
}

// ============================================================================
// Vectorized BF16 operations (uint16 storage in registers)
// ============================================================================

/// Convert 4 packed BF16 values (stored as ushort4) to float4 using direct
/// bit manipulation (no FP16 intermediate).
///
/// @param packed  4 BF16 values in uint16 storage
/// @return        4 FP32 values
inline float4 bf16x4_to_float4_direct(ushort4 packed) {
    ushort lane = ushort(simd_lane_id);
    ushort4 lane_vals = ushort4(
        simd_shuffle(packed.x, lane),
        simd_shuffle(packed.y, lane),
        simd_shuffle(packed.z, lane),
        simd_shuffle(packed.w, lane)
    );
    uint4 widened = uint4(lane_vals) << 16;
    return as_type<float4>(widened);
}

/// Convert 4 packed BF16 values (stored as ushort4) to float4.
///
/// @param packed  4 BF16 values in uint16 storage
/// @return        4 FP32 values
inline float4 bf16x4_to_float4(ushort4 packed) {
    return bf16x4_to_float4_direct(packed);
}

/// Convert 4 FP32 values to packed BF16 (ushort4) with truncation.
///
/// @param vals  4 FP32 values
/// @return      4 BF16 values in uint16 storage
inline ushort4 float4_to_bf16x4_rtz(float4 vals) {
    return ushort4(
        ushort(as_type<uint>(vals.x) >> 16),
        ushort(as_type<uint>(vals.y) >> 16),
        ushort(as_type<uint>(vals.z) >> 16),
        ushort(as_type<uint>(vals.w) >> 16)
    );
}

/// Convert 4 FP32 values to packed BF16 (ushort4) with RNE rounding.
///
/// @param vals  4 FP32 values
/// @return      4 BF16 values in uint16 storage (correctly rounded)
inline ushort4 float4_to_bf16x4_rne_direct(float4 vals) {
    uint4 f_bits = as_type<uint4>(vals);
    uint4 exp_bits = (f_bits >> 23) & 0xFFu;
    uint4 mantissa = f_bits & 0x007FFFFFu;
    uint4 round_bias = 0x8000u + ((f_bits >> 16) & 1u);

    uint4 rounded = f_bits + round_bias;
    ushort4 rne_bits = ushort4(rounded >> 16);

    bool4 is_special = exp_bits == uint4(0xFFu);
    bool4 is_nan = is_special & (mantissa != uint4(0u));
    bool4 is_inf = is_special & (mantissa == uint4(0u));

    ushort4 trunc_bits = ushort4(f_bits >> 16);
    ushort4 nan_bits = trunc_bits | ushort4(0x0040u);

    ushort4 result = select(rne_bits, trunc_bits, is_inf);
    result = select(result, nan_bits, is_nan);
    return result;
}

inline ushort4 float4_to_bf16x4_rne(float4 vals) {
    return float4_to_bf16x4_rne_direct(vals);
}

/// Convert 4 packed BF16 values to half4.
/// Potentially lossy: BF16 values outside FP16 range saturate.
///
/// @param packed  4 BF16 values in uint16 storage
/// @return        4 FP16 values
inline half4 bf16x4_to_half4(ushort4 packed) {
    float4 f = bf16x4_to_float4(packed);
    return half4(f);
}

// ============================================================================
// BF16 activation storage: pack/unpack for simdgroup GEMM integration
// ============================================================================

/// Load 8 BF16 activation values from a device buffer and convert to float4 pairs
/// without going through FP16.
///
/// @param src       Pointer to BF16 values (ushort storage)
/// @param offset    Element offset into the buffer
/// @param out_lo    Output: first 4 values as float4
/// @param out_hi    Output: last 4 values as float4
inline void bf16_load_as_float8_direct(device const ushort *src,
                                       uint offset,
                                       thread float4 &out_lo,
                                       thread float4 &out_hi) {
    ushort4 lo_packed = ushort4(src[offset],     src[offset + 1],
                                src[offset + 2], src[offset + 3]);
    ushort4 hi_packed = ushort4(src[offset + 4], src[offset + 5],
                                src[offset + 6], src[offset + 7]);

    out_lo = bf16x4_to_float4_direct(lo_packed);
    out_hi = bf16x4_to_float4_direct(hi_packed);
}

inline void bf16_load_as_float8(device const ushort *src,
                                uint offset,
                                thread float4 &out_lo,
                                thread float4 &out_hi) {
    bf16_load_as_float8_direct(src, offset, out_lo, out_hi);
}

/// Load 8 BF16 activation values from a device buffer and convert to half4 pairs.
/// Intended for use in fused GEMM kernels where activations are stored in BF16
/// but computation occurs in FP16 (matching Metal's simdgroup_matrix_multiply).
///
/// @param src       Pointer to BF16 values (ushort storage)
/// @param offset    Element offset into the buffer
/// @param out_lo    Output: first 4 values as half4
/// @param out_hi    Output: last 4 values as half4
inline void bf16_load_as_half8(device const ushort *src,
                               uint offset,
                               thread half4 &out_lo,
                               thread half4 &out_hi) {
    // Load 8 BF16 values
    ushort4 lo_packed = ushort4(src[offset],     src[offset + 1],
                                src[offset + 2], src[offset + 3]);
    ushort4 hi_packed = ushort4(src[offset + 4], src[offset + 5],
                                src[offset + 6], src[offset + 7]);

    // Convert BF16 -> FP32 -> FP16
    // Going through FP32 preserves values within FP16 range accurately.
    // Values outside FP16 range will saturate.
    out_lo = bf16x4_to_half4(lo_packed);
    out_hi = bf16x4_to_half4(hi_packed);
}

/// Load 8 BF16 values and convert directly to float (no FP16 intermediate).
/// Intended for FP32 accumulation paths.
///
/// @param src       Pointer to BF16 values (ushort storage)
/// @param offset    Element offset into the buffer
/// @param out_lo    Output: first 4 values as float4
/// @param out_hi    Output: last 4 values as float4
inline void bf16_load_as_float8(device const ushort *src,
                                uint offset,
                                thread float4 &out_lo,
                                thread float4 &out_hi) {
    ushort4 lo_packed = ushort4(src[offset],     src[offset + 1],
                                src[offset + 2], src[offset + 3]);
    ushort4 hi_packed = ushort4(src[offset + 4], src[offset + 5],
                                src[offset + 6], src[offset + 7]);

    out_lo = bf16x4_to_float4(lo_packed);
    out_hi = bf16x4_to_float4(hi_packed);
}

/// Store 8 half values as BF16 to a device buffer (with RNE rounding).
/// Intended for writing GEMM outputs back to BF16 activation memory.
///
/// @param dst       Pointer to BF16 output buffer (ushort storage)
/// @param offset    Element offset into the buffer
/// @param lo        First 4 values (half4)
/// @param hi        Last 4 values (half4)
inline void bf16_store_from_half8(device ushort *dst,
                                  uint offset,
                                  half4 lo,
                                  half4 hi) {
    // half -> float -> BF16 with RNE rounding
    ushort4 lo_packed = float4_to_bf16x4_rne(float4(lo));
    ushort4 hi_packed = float4_to_bf16x4_rne(float4(hi));

    dst[offset]     = lo_packed.x;
    dst[offset + 1] = lo_packed.y;
    dst[offset + 2] = lo_packed.z;
    dst[offset + 3] = lo_packed.w;
    dst[offset + 4] = hi_packed.x;
    dst[offset + 5] = hi_packed.y;
    dst[offset + 6] = hi_packed.z;
    dst[offset + 7] = hi_packed.w;
}

/// Store 8 float values directly as BF16 (RNE rounding, no FP16 intermediate).
/// Intended for FP32 accumulator output paths.
///
/// @param dst       Pointer to BF16 output buffer (ushort storage)
/// @param offset    Element offset into the buffer
/// @param lo        First 4 values (float4)
/// @param hi        Last 4 values (float4)
inline void bf16_store_from_float8(device ushort *dst,
                                   uint offset,
                                   float4 lo,
                                   float4 hi) {
    ushort4 lo_packed = float4_to_bf16x4_rne(lo);
    ushort4 hi_packed = float4_to_bf16x4_rne(hi);

    dst[offset]     = lo_packed.x;
    dst[offset + 1] = lo_packed.y;
    dst[offset + 2] = lo_packed.z;
    dst[offset + 3] = lo_packed.w;
    dst[offset + 4] = hi_packed.x;
    dst[offset + 5] = hi_packed.y;
    dst[offset + 6] = hi_packed.z;
    dst[offset + 7] = hi_packed.w;
}

/// Store 8 float values as BF16 to a device buffer (with RNE rounding).
/// Intended for writing FP32 outputs back to BF16 activation memory.
///
/// @param dst       Pointer to BF16 output buffer (ushort storage)
/// @param offset    Element offset into the buffer
/// @param lo        First 4 values (float4)
/// @param hi        Last 4 values (float4)
inline void bf16_store_from_float8_direct(device ushort *dst,
                                          uint offset,
                                          float4 lo,
                                          float4 hi) {
    ushort4 lo_packed = float4_to_bf16x4_rne_direct(lo);
    ushort4 hi_packed = float4_to_bf16x4_rne_direct(hi);

    dst[offset]     = lo_packed.x;
    dst[offset + 1] = lo_packed.y;
    dst[offset + 2] = lo_packed.z;
    dst[offset + 3] = lo_packed.w;
    dst[offset + 4] = hi_packed.x;
    dst[offset + 5] = hi_packed.y;
    dst[offset + 6] = hi_packed.z;
    dst[offset + 7] = hi_packed.w;
}

inline void bf16_store_from_float8(device ushort *dst,
                                   uint offset,
                                   float4 lo,
                                   float4 hi) {
    bf16_store_from_float8_direct(dst, offset, lo, hi);
}

// ============================================================================
// Compute kernels: BF16 <-> FP16/FP32 bulk conversion
// ============================================================================

/// Bulk convert BF16 activations to FP16 for downstream compute.
/// Each thread processes 4 elements for coalesced access.
kernel void bf16_to_half_kernel(
    device const ushort *input  [[buffer(0)]],
    device half4 *output        [[buffer(1)]],
    constant uint &num_elements [[buffer(2)]],
    uint tid                    [[thread_position_in_grid]])
{
    uint base_idx = tid * 4u;
    if (base_idx >= num_elements) return;

    ushort4 packed = ushort4(input[base_idx],     input[base_idx + 1],
                             input[base_idx + 2], input[base_idx + 3]);
    output[tid] = bf16x4_to_half4(packed);
}

/// Bulk convert FP16 activations to BF16 for storage.
/// Uses RNE rounding for training-grade accuracy.
kernel void half_to_bf16_kernel(
    device const half4 *input   [[buffer(0)]],
    device ushort *output       [[buffer(1)]],
    constant uint &num_elements [[buffer(2)]],
    uint tid                    [[thread_position_in_grid]])
{
    uint base_idx = tid * 4u;
    if (base_idx >= num_elements) return;

    half4 vals = input[tid];
    ushort4 packed = float4_to_bf16x4_rne(float4(vals));

    output[base_idx]     = packed.x;
    output[base_idx + 1] = packed.y;
    output[base_idx + 2] = packed.z;
    output[base_idx + 3] = packed.w;
}

/// Bulk convert BF16 to FP32 (exact).
kernel void bf16_to_float_kernel(
    device const ushort *input  [[buffer(0)]],
    device float4 *output       [[buffer(1)]],
    constant uint &num_elements [[buffer(2)]],
    uint tid                    [[thread_position_in_grid]])
{
    uint base_idx = tid * 4u;
    if (base_idx >= num_elements) return;

    ushort4 packed = ushort4(input[base_idx],     input[base_idx + 1],
                             input[base_idx + 2], input[base_idx + 3]);
    output[tid] = bf16x4_to_float4(packed);
}

/// Bulk convert FP32 to BF16 with selectable rounding mode.
/// @param rounding_mode  0 = truncation (RTZ), 1 = round-to-nearest-even (RNE)
kernel void float_to_bf16_kernel(
    device const float4 *input    [[buffer(0)]],
    device ushort *output         [[buffer(1)]],
    constant uint &num_elements   [[buffer(2)]],
    constant uint &rounding_mode  [[buffer(3)]],
    uint tid                      [[thread_position_in_grid]])
{
    uint base_idx = tid * 4u;
    if (base_idx >= num_elements) return;

    float4 vals = input[tid];
    ushort4 packed;

    if (rounding_mode == 0u) {
        packed = float4_to_bf16x4_rtz(vals);
    } else {
        packed = float4_to_bf16x4_rne(vals);
    }

    output[base_idx]     = packed.x;
    output[base_idx + 1] = packed.y;
    output[base_idx + 2] = packed.z;
    output[base_idx + 3] = packed.w;
}

// ============================================================================
// Test kernel: validate BF16 conversion round-trip accuracy
// ============================================================================

/// Test kernel that converts FP32 -> BF16 -> FP32 and reports the error.
/// Used to validate the software BF16 implementation against reference values.
///
/// @param input   FP32 test values
/// @param output  Round-tripped FP32 values (should differ by at most 1 BF16 ULP)
/// @param errors  Absolute error for each value
/// @param count   Number of test values
kernel void bf16_roundtrip_test(
    device const float *input   [[buffer(0)]],
    device float *output        [[buffer(1)]],
    device float *errors        [[buffer(2)]],
    constant uint &count        [[buffer(3)]],
    uint tid                    [[thread_position_in_grid]])
{
    if (tid >= count) return;

    float original = input[tid];
    bf16_t converted = bf16_from_float_rne(original);
    float roundtripped = bf16_to_float(converted);

    output[tid] = roundtripped;
    errors[tid] = abs(original - roundtripped);
}

/// Test kernel that uses the direct BF16 load/store helpers on float4 pairs.
kernel void bf16_roundtrip_direct_float8(
    device const float4 *input  [[buffer(0)]],
    device float4 *output       [[buffer(1)]],
    device ushort *scratch      [[buffer(2)]],
    constant uint &num_elements [[buffer(3)]],
    uint tid                    [[thread_position_in_grid]])
{
    uint base_idx = tid * 8u;
    if (base_idx >= num_elements) return;

    uint vec_idx = tid * 2u;
    float4 lo = input[vec_idx];
    float4 hi = input[vec_idx + 1u];

    bf16_store_from_float8_direct(scratch, base_idx, lo, hi);

    thread float4 out_lo;
    thread float4 out_hi;
    bf16_load_as_float8_direct(scratch, base_idx, out_lo, out_hi);

    output[vec_idx] = out_lo;
    output[vec_idx + 1u] = out_hi;
}
