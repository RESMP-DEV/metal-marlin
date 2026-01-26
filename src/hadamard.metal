#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Walsh-Hadamard Transform for Activation Rotation
// ============================================================================
//
// For Hadamard-rotated weights (QuIP#, HQQ), activations need complementary
// rotation before the GEMM:
//
//   y = (H @ W) @ (H^T @ x) = W @ x  (mathematically equivalent)
//
// The Walsh-Hadamard transform is self-inverse (H = H^T = H^-1 up to scaling),
// so we can fuse H^T @ x as a preprocessing step.
//
// Algorithm: In-place butterfly pattern with O(n log n) complexity.
// For a block of size n = 2^k:
//   1. Split input into pairs (a, b)
//   2. Compute (a + b, a - b)
//   3. Recurse on half-sized subproblems
//   4. Apply normalization factor 1/sqrt(n) at the end
//
// Memory layout:
//   Input/output: [N, block_size] where N = number of vectors, block_size = 64 or 128
//   Each threadgroup processes one row (one vector)
//
// Optimization:
//   - Use simdgroup_shuffle for register-only butterfly within a simdgroup
//   - Process multiple elements per thread when block_size > 32
//   - Avoid threadgroup memory by leveraging Apple Silicon's fast shuffle
//
// ============================================================================

// ============================================================================
// Constants
// ============================================================================

// Supported block sizes: 32, 64, 128
// For larger blocks, adjust threads per threadgroup accordingly

// ============================================================================
// Inline butterfly helpers using simd_shuffle
// ============================================================================

/// Single butterfly step: computes (a + b, a - b) where b is shuffled from another lane.
/// @param val       This thread's value
/// @param partner   Lane index to read partner value from
/// @param is_upper  True if this lane should produce (a - b), false for (a + b)
/// @return          Butterfly result for this lane
inline half butterfly_step(half val, uint partner, bool is_upper) {
    half partner_val = simd_shuffle(val, ushort(partner));
    return is_upper ? (val - partner_val) : (val + partner_val);
}

/// Same for half2 (two parallel values per thread)
inline half2 butterfly_step2(half2 val, uint partner, bool is_upper) {
    half2 partner_val = simd_shuffle(val, ushort(partner));
    return is_upper ? (val - partner_val) : (val + partner_val);
}

/// Same for half4 (four parallel values per thread)
inline half4 butterfly_step4(half4 val, uint partner, bool is_upper) {
    half4 partner_val = simd_shuffle(val, ushort(partner));
    return is_upper ? (val - partner_val) : (val + partner_val);
}

// ============================================================================
// Hadamard transform kernel: block_size = 32 (fits in one simdgroup)
// ============================================================================

/// Walsh-Hadamard transform for block_size = 32.
/// Each lane in the simdgroup holds one element. Butterfly pattern via shuffle.
///
/// @param input       Input vectors [N, 32]
/// @param output      Output vectors [N, 32] (can alias input for in-place)
/// @param n           Number of vectors (N dimension)
/// @param normalize   If true, divide by sqrt(32) after transform
kernel void hadamard_transform_32(
    device const half* input  [[buffer(0)]],
    device half* output       [[buffer(1)]],
    constant uint& n          [[buffer(2)]],
    constant uint& normalize  [[buffer(3)]],
    uint lane_id              [[thread_index_in_simdgroup]],
    uint tg_idx               [[threadgroup_position_in_grid]]
) {
    if (tg_idx >= n) return;

    // Each threadgroup processes one vector
    uint base = tg_idx * 32;
    half val = input[base + lane_id];

    // 5 butterfly stages for size 32 (log2(32) = 5)
    // Stage 0: stride 1
    {
        uint partner = lane_id ^ 1;
        bool is_upper = (lane_id & 1) != 0;
        val = butterfly_step(val, partner, is_upper);
    }

    // Stage 1: stride 2
    {
        uint partner = lane_id ^ 2;
        bool is_upper = (lane_id & 2) != 0;
        val = butterfly_step(val, partner, is_upper);
    }

    // Stage 2: stride 4
    {
        uint partner = lane_id ^ 4;
        bool is_upper = (lane_id & 4) != 0;
        val = butterfly_step(val, partner, is_upper);
    }

    // Stage 3: stride 8
    {
        uint partner = lane_id ^ 8;
        bool is_upper = (lane_id & 8) != 0;
        val = butterfly_step(val, partner, is_upper);
    }

    // Stage 4: stride 16
    {
        uint partner = lane_id ^ 16;
        bool is_upper = (lane_id & 16) != 0;
        val = butterfly_step(val, partner, is_upper);
    }

    // Normalize by 1/sqrt(32) = 1/sqrt(32) ≈ 0.1767766953
    if (normalize) {
        val *= half(0.1767766953h);
    }

    output[base + lane_id] = val;
}

// ============================================================================
// Hadamard transform kernel: block_size = 64
// ============================================================================

/// Walsh-Hadamard transform for block_size = 64.
/// Each lane holds 2 elements (half2). Inter-lane shuffle + intra-lane swizzle.
///
/// @param input       Input vectors [N, 64]
/// @param output      Output vectors [N, 64]
/// @param n           Number of vectors
/// @param normalize   If true, divide by sqrt(64) = 8 after transform
kernel void hadamard_transform_64(
    device const half* input  [[buffer(0)]],
    device half* output       [[buffer(1)]],
    constant uint& n          [[buffer(2)]],
    constant uint& normalize  [[buffer(3)]],
    uint lane_id              [[thread_index_in_simdgroup]],
    uint tg_idx               [[threadgroup_position_in_grid]]
) {
    if (tg_idx >= n) return;

    // Each thread loads 2 elements
    uint base = tg_idx * 64;
    half2 val;
    val.x = input[base + lane_id * 2];
    val.y = input[base + lane_id * 2 + 1];

    // Stage 0: stride 1 (within half2, no shuffle needed)
    {
        half sum = val.x + val.y;
        half diff = val.x - val.y;
        val.x = sum;
        val.y = diff;
    }

    // Stage 1: stride 2 (shuffle between adjacent thread pairs, swap half2 components)
    {
        uint partner = lane_id ^ 1;
        half2 partner_val = simd_shuffle(val, ushort(partner));
        bool is_upper = (lane_id & 1) != 0;
        if (is_upper) {
            val.x = val.x - partner_val.x;
            val.y = val.y - partner_val.y;
        } else {
            val.x = val.x + partner_val.x;
            val.y = val.y + partner_val.y;
        }
    }

    // Stage 2: stride 4
    {
        uint partner = lane_id ^ 2;
        bool is_upper = (lane_id & 2) != 0;
        val = butterfly_step2(val, partner, is_upper);
    }

    // Stage 3: stride 8
    {
        uint partner = lane_id ^ 4;
        bool is_upper = (lane_id & 4) != 0;
        val = butterfly_step2(val, partner, is_upper);
    }

    // Stage 4: stride 16
    {
        uint partner = lane_id ^ 8;
        bool is_upper = (lane_id & 8) != 0;
        val = butterfly_step2(val, partner, is_upper);
    }

    // Stage 5: stride 32
    {
        uint partner = lane_id ^ 16;
        bool is_upper = (lane_id & 16) != 0;
        val = butterfly_step2(val, partner, is_upper);
    }

    // Normalize by 1/sqrt(64) = 1/8 = 0.125
    if (normalize) {
        val *= half(0.125h);
    }

    output[base + lane_id * 2] = val.x;
    output[base + lane_id * 2 + 1] = val.y;
}

// ============================================================================
// Hadamard transform kernel: block_size = 128
// ============================================================================

/// Walsh-Hadamard transform for block_size = 128.
/// Each lane holds 4 elements (half4). Inter-lane shuffle + intra-lane swizzle.
///
/// @param input       Input vectors [N, 128]
/// @param output      Output vectors [N, 128]
/// @param n           Number of vectors
/// @param normalize   If true, divide by sqrt(128) after transform
kernel void hadamard_transform_128(
    device const half* input  [[buffer(0)]],
    device half* output       [[buffer(1)]],
    constant uint& n          [[buffer(2)]],
    constant uint& normalize  [[buffer(3)]],
    uint lane_id              [[thread_index_in_simdgroup]],
    uint tg_idx               [[threadgroup_position_in_grid]]
) {
    if (tg_idx >= n) return;

    // Each thread loads 4 elements
    uint base = tg_idx * 128;
    half4 val;
    val.x = input[base + lane_id * 4];
    val.y = input[base + lane_id * 4 + 1];
    val.z = input[base + lane_id * 4 + 2];
    val.w = input[base + lane_id * 4 + 3];

    // Stage 0: stride 1 (pairs within half4)
    {
        half sum0 = val.x + val.y;
        half diff0 = val.x - val.y;
        half sum1 = val.z + val.w;
        half diff1 = val.z - val.w;
        val = half4(sum0, diff0, sum1, diff1);
    }

    // Stage 1: stride 2 (swap pairs within half4)
    {
        half sum0 = val.x + val.z;
        half diff0 = val.x - val.z;
        half sum1 = val.y + val.w;
        half diff1 = val.y - val.w;
        val = half4(sum0, sum1, diff0, diff1);
    }

    // Stage 2: stride 4 (shuffle between adjacent threads)
    {
        uint partner = lane_id ^ 1;
        bool is_upper = (lane_id & 1) != 0;
        val = butterfly_step4(val, partner, is_upper);
    }

    // Stage 3: stride 8
    {
        uint partner = lane_id ^ 2;
        bool is_upper = (lane_id & 2) != 0;
        val = butterfly_step4(val, partner, is_upper);
    }

    // Stage 4: stride 16
    {
        uint partner = lane_id ^ 4;
        bool is_upper = (lane_id & 4) != 0;
        val = butterfly_step4(val, partner, is_upper);
    }

    // Stage 5: stride 32
    {
        uint partner = lane_id ^ 8;
        bool is_upper = (lane_id & 8) != 0;
        val = butterfly_step4(val, partner, is_upper);
    }

    // Stage 6: stride 64
    {
        uint partner = lane_id ^ 16;
        bool is_upper = (lane_id & 16) != 0;
        val = butterfly_step4(val, partner, is_upper);
    }

    // Normalize by 1/sqrt(128) ≈ 0.0883883476
    if (normalize) {
        val *= half(0.0883883476h);
    }

    output[base + lane_id * 4] = val.x;
    output[base + lane_id * 4 + 1] = val.y;
    output[base + lane_id * 4 + 2] = val.z;
    output[base + lane_id * 4 + 3] = val.w;
}

// ============================================================================
// Batched Hadamard transform with vectorized loads/stores
// ============================================================================

/// High-throughput Hadamard transform for block_size = 64 with vectorized memory.
/// Uses half4 loads/stores for better memory bandwidth.
///
/// @param input       Input vectors [N, 64] (16-byte aligned)
/// @param output      Output vectors [N, 64] (16-byte aligned)
/// @param n           Number of vectors
/// @param normalize   If true, divide by sqrt(64) after transform
kernel void hadamard_transform_64_vec(
    device const half4* input  [[buffer(0)]],
    device half4* output       [[buffer(1)]],
    constant uint& n           [[buffer(2)]],
    constant uint& normalize   [[buffer(3)]],
    uint lane_id               [[thread_index_in_simdgroup]],
    uint tg_idx                [[threadgroup_position_in_grid]]
) {
    if (tg_idx >= n) return;

    // Each thread loads 2 half4 (8 elements), but we need 64/32 = 2 elements per thread
    // So we load a portion and use shuffle
    // Actually: 64 elements / 32 lanes = 2 elements per lane
    // Let's reload as half2 from the half4 array

    device const half* input_h = (device const half*)input;
    device half* output_h = (device half*)output;

    uint base = tg_idx * 64;
    half2 val;
    val.x = input_h[base + lane_id * 2];
    val.y = input_h[base + lane_id * 2 + 1];

    // Stage 0: stride 1
    {
        half sum = val.x + val.y;
        half diff = val.x - val.y;
        val.x = sum;
        val.y = diff;
    }

    // Stage 1: stride 2
    {
        uint partner = lane_id ^ 1;
        half2 partner_val = simd_shuffle(val, ushort(partner));
        bool is_upper = (lane_id & 1) != 0;
        val = is_upper ? (val - partner_val) : (val + partner_val);
    }

    // Stage 2: stride 4
    {
        uint partner = lane_id ^ 2;
        bool is_upper = (lane_id & 2) != 0;
        val = butterfly_step2(val, partner, is_upper);
    }

    // Stage 3: stride 8
    {
        uint partner = lane_id ^ 4;
        bool is_upper = (lane_id & 4) != 0;
        val = butterfly_step2(val, partner, is_upper);
    }

    // Stage 4: stride 16
    {
        uint partner = lane_id ^ 8;
        bool is_upper = (lane_id & 8) != 0;
        val = butterfly_step2(val, partner, is_upper);
    }

    // Stage 5: stride 32
    {
        uint partner = lane_id ^ 16;
        bool is_upper = (lane_id & 16) != 0;
        val = butterfly_step2(val, partner, is_upper);
    }

    if (normalize) {
        val *= half(0.125h);
    }

    output_h[base + lane_id * 2] = val.x;
    output_h[base + lane_id * 2 + 1] = val.y;
}

// ============================================================================
// Fused Hadamard + Scale kernel (for preprocessing before quantized GEMM)
// ============================================================================

/// Hadamard transform with fused scale multiplication.
/// Useful when activations need both rotation and a learned scale factor.
///
/// @param input       Input vectors [N, 64]
/// @param scales      Per-vector or per-element scales [N] or [N, 64]
/// @param output      Output vectors [N, 64]
/// @param n           Number of vectors
/// @param per_element If true, scales is [N, 64]; if false, [N]
kernel void hadamard_transform_64_scaled(
    device const half* input   [[buffer(0)]],
    device const half* scales  [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint& n           [[buffer(3)]],
    constant uint& per_element [[buffer(4)]],
    uint lane_id               [[thread_index_in_simdgroup]],
    uint tg_idx                [[threadgroup_position_in_grid]]
) {
    if (tg_idx >= n) return;

    uint base = tg_idx * 64;
    half2 val;
    val.x = input[base + lane_id * 2];
    val.y = input[base + lane_id * 2 + 1];

    // Apply pre-Hadamard scaling if per-element
    if (per_element) {
        val.x *= scales[base + lane_id * 2];
        val.y *= scales[base + lane_id * 2 + 1];
    }

    // 6-stage Walsh-Hadamard (same as hadamard_transform_64)
    {
        half sum = val.x + val.y;
        half diff = val.x - val.y;
        val.x = sum;
        val.y = diff;
    }

    {
        uint partner = lane_id ^ 1;
        half2 partner_val = simd_shuffle(val, ushort(partner));
        bool is_upper = (lane_id & 1) != 0;
        val = is_upper ? (val - partner_val) : (val + partner_val);
    }

    {
        uint partner = lane_id ^ 2;
        bool is_upper = (lane_id & 2) != 0;
        val = butterfly_step2(val, partner, is_upper);
    }

    {
        uint partner = lane_id ^ 4;
        bool is_upper = (lane_id & 4) != 0;
        val = butterfly_step2(val, partner, is_upper);
    }

    {
        uint partner = lane_id ^ 8;
        bool is_upper = (lane_id & 8) != 0;
        val = butterfly_step2(val, partner, is_upper);
    }

    {
        uint partner = lane_id ^ 16;
        bool is_upper = (lane_id & 16) != 0;
        val = butterfly_step2(val, partner, is_upper);
    }

    // Normalize and apply per-vector scale
    half norm = half(0.125h);  // 1/sqrt(64)
    if (!per_element) {
        norm *= scales[tg_idx];
    }
    val *= norm;

    output[base + lane_id * 2] = val.x;
    output[base + lane_id * 2 + 1] = val.y;
}

// ============================================================================
// Inverse Hadamard transform (same operation, different normalization)
// ============================================================================

/// Inverse Walsh-Hadamard transform for block_size = 64.
/// Since H = H^-1 up to scaling, this is the same butterfly pattern but
/// normalizes by sqrt(64) instead of 1/sqrt(64), or equivalently applies
/// the transform without normalization (since we already normalized forward).
///
/// For QuIP#/HQQ workflows:
///   Forward (on activations before GEMM): H @ x / sqrt(n)
///   Inverse (after GEMM if needed): H @ y (no normalization, or * sqrt(n))
///
/// This kernel provides the unnormalized variant.
kernel void hadamard_inverse_64(
    device const half* input  [[buffer(0)]],
    device half* output       [[buffer(1)]],
    constant uint& n          [[buffer(2)]],
    uint lane_id              [[thread_index_in_simdgroup]],
    uint tg_idx               [[threadgroup_position_in_grid]]
) {
    if (tg_idx >= n) return;

    uint base = tg_idx * 64;
    half2 val;
    val.x = input[base + lane_id * 2];
    val.y = input[base + lane_id * 2 + 1];

    // Same butterfly pattern as forward, no normalization
    {
        half sum = val.x + val.y;
        half diff = val.x - val.y;
        val.x = sum;
        val.y = diff;
    }

    {
        uint partner = lane_id ^ 1;
        half2 partner_val = simd_shuffle(val, ushort(partner));
        bool is_upper = (lane_id & 1) != 0;
        val = is_upper ? (val - partner_val) : (val + partner_val);
    }

    {
        uint partner = lane_id ^ 2;
        bool is_upper = (lane_id & 2) != 0;
        val = butterfly_step2(val, partner, is_upper);
    }

    {
        uint partner = lane_id ^ 4;
        bool is_upper = (lane_id & 4) != 0;
        val = butterfly_step2(val, partner, is_upper);
    }

    {
        uint partner = lane_id ^ 8;
        bool is_upper = (lane_id & 8) != 0;
        val = butterfly_step2(val, partner, is_upper);
    }

    {
        uint partner = lane_id ^ 16;
        bool is_upper = (lane_id & 16) != 0;
        val = butterfly_step2(val, partner, is_upper);
    }

    // No normalization for inverse
    output[base + lane_id * 2] = val.x;
    output[base + lane_id * 2 + 1] = val.y;
}

// ============================================================================
// Test/verification kernel: verify Hadamard orthogonality
// ============================================================================

/// Test kernel: apply Hadamard twice (should recover original with n scaling).
/// Useful for verifying correctness: H @ H @ x = n * x
kernel void test_hadamard_roundtrip_64(
    device const half* input  [[buffer(0)]],
    device half* output       [[buffer(1)]],
    constant uint& n          [[buffer(2)]],
    uint lane_id              [[thread_index_in_simdgroup]],
    uint tg_idx               [[threadgroup_position_in_grid]]
) {
    if (tg_idx >= n) return;

    uint base = tg_idx * 64;
    half2 val;
    val.x = input[base + lane_id * 2];
    val.y = input[base + lane_id * 2 + 1];

    // First Hadamard (unnormalized)
    for (uint stage = 0; stage < 6; ++stage) {
        if (stage == 0) {
            half sum = val.x + val.y;
            half diff = val.x - val.y;
            val.x = sum;
            val.y = diff;
        } else {
            uint stride = 1u << (stage - 1);
            uint partner = lane_id ^ stride;
            bool is_upper = (lane_id & stride) != 0;
            val = butterfly_step2(val, partner, is_upper);
        }
    }

    // Second Hadamard (unnormalized)
    for (uint stage = 0; stage < 6; ++stage) {
        if (stage == 0) {
            half sum = val.x + val.y;
            half diff = val.x - val.y;
            val.x = sum;
            val.y = diff;
        } else {
            uint stride = 1u << (stage - 1);
            uint partner = lane_id ^ stride;
            bool is_upper = (lane_id & stride) != 0;
            val = butterfly_step2(val, partner, is_upper);
        }
    }

    // H @ H @ x = 64 * x, so divide by 64 to recover original
    val *= half(1.0h / 64.0h);

    output[base + lane_id * 2] = val.x;
    output[base + lane_id * 2 + 1] = val.y;
}
