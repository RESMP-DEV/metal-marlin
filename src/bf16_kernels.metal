#include "bf16_compat.metal"

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
