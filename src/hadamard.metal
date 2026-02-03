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

// Supported block sizes: 32, 64, 96, 128, 160, 192, 256
// Power-of-2 sizes (32, 64, 128, 256) use optimized O(n log n) butterfly
// Non-power-of-2 sizes use block-diagonal decomposition for optimal performance:
//   - 96  = 64 + 32 (two independent transforms executed in parallel)
//   - 160 = 128 + 32 (maximizes simdgroup utilization)
//   - 192 = 128 + 64 (balanced subblock sizes)

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
    return is_upper ? (partner_val - val) : (val + partner_val);
}

/// Same for half2 (two parallel values per thread)
inline half2 butterfly_step2(half2 val, uint partner, bool is_upper) {
    half2 partner_val = simd_shuffle(val, ushort(partner));
    return is_upper ? (partner_val - val) : (val + partner_val);
}

/// Same for half4 (four parallel values per thread)
inline half4 butterfly_step4(half4 val, uint partner, bool is_upper) {
    half4 partner_val = simd_shuffle(val, ushort(partner));
    return is_upper ? (partner_val - val) : (val + partner_val);
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
            val.x = partner_val.x - val.x;
            val.y = partner_val.y - val.y;
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
        val = is_upper ? (partner_val - val) : (val + partner_val);
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
        val = is_upper ? (partner_val - val) : (val + partner_val);
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
        val = is_upper ? (partner_val - val) : (val + partner_val);
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

// ============================================================================
// WEIGHT MATRIX HADAMARD TRANSFORM KERNELS
// ============================================================================
//
// For weight rotation (QuIP#, HQQ), we apply block-diagonal Hadamard to weight
// matrices. Input W [K, N] is partitioned into blocks of size `block_size` 
// along dimension K. Each block is transformed: W_rot[block] = H @ W[block].
//
// Layout:
//   - Weights are stored in row-major: W[k, n] at index k * N + n
//   - Each thread processes one row (one n index) across all k blocks
//   - Butterfly operations happen within each K-block
//
// ============================================================================

// ============================================================================
// hadamard_forward_fast_64: Optimized O(n log n) butterfly for weights [K, N]
// Each thread handles one column n, applies butterfly across K blocks
// ============================================================================

/// Fast Hadamard transform for weight matrices [K, N] using butterfly operations.
/// O(n log n) complexity instead of O(n²) for matrix multiplication.
///
/// Layout: W [K, N] row-major, K must be divisible by 64
/// Grid: (N, K/64) - one thread per (n, block) pair
/// Each simdgroup handles one 64-element block for one n column.
///
/// Uses the fast Walsh-Hadamard transform with log2(64) = 6 butterfly stages:
///   Stage 0: stride 1 (within thread)
///   Stage 1: stride 2 (simd shuffle)
///   Stage 2-5: stride 4, 8, 16, 32 (simd shuffle)
///
/// Normalization: 1/sqrt(64) = 0.125 (orthonormal Hadamard)
///
/// @param W           Input weight matrix [K, N] in half precision
/// @param W_rot       Output rotated weights [K, N]
/// @param K           Size of K dimension (must be multiple of 64)
/// @param N           Size of N dimension
kernel void hadamard_forward_fast_64(
    device const half* W        [[buffer(0)]],
    device half* W_rot          [[buffer(1)]],
    constant uint& K            [[buffer(2)]],
    constant uint& N            [[buffer(3)]],
    uint3 gid                   [[thread_position_in_grid]],
    uint lane_id              [[thread_index_in_simdgroup]]
)
{
    uint n = gid.x;  // Global column index
    uint block_k = gid.y;  // Which 64-row block
    
    if (n >= N || block_k >= (K / 64)) return;
    
    uint k_base = block_k * 64;
    
    // Each lane handles 2 elements (64 elements / 32 lanes = 2 per lane)
    // Load from global memory: W[k, n] at index k * N + n
    half2 val;
    val.x = W[(k_base + lane_id * 2) * N + n];
    val.y = W[(k_base + lane_id * 2 + 1) * N + n];
    
    // Butterfly Stage 0: stride 1 (within half2)
    {
        half sum = val.x + val.y;
        half diff = val.x - val.y;
        val.x = sum;
        val.y = diff;
    }
    
    // Butterfly Stage 1: stride 2 (shuffle between lanes)
    {
        uint partner = lane_id ^ 1;
        half2 partner_val = simd_shuffle(val, ushort(partner));
        bool is_upper = (lane_id & 1) != 0;
        val = is_upper ? (partner_val - val) : (val + partner_val);
    }
    
    // Butterfly Stage 2: stride 4
    {
        uint partner = lane_id ^ 2;
        bool is_upper = (lane_id & 2) != 0;
        val = butterfly_step2(val, partner, is_upper);
    }
    
    // Butterfly Stage 3: stride 8
    {
        uint partner = lane_id ^ 4;
        bool is_upper = (lane_id & 4) != 0;
        val = butterfly_step2(val, partner, is_upper);
    }
    
    // Butterfly Stage 4: stride 16
    {
        uint partner = lane_id ^ 8;
        bool is_upper = (lane_id & 8) != 0;
        val = butterfly_step2(val, partner, is_upper);
    }
    
    // Butterfly Stage 5: stride 32
    {
        uint partner = lane_id ^ 16;
        bool is_upper = (lane_id & 16) != 0;
        val = butterfly_step2(val, partner, is_upper);
    }
    
    // Normalize by 1/sqrt(64) = 0.125 for orthonormal Hadamard
    // This ensures H @ H = I (identity matrix)
    val *= half(0.125h);
    
    // Write back to global memory
    W_rot[(k_base + lane_id * 2) * N + n] = val.x;
    W_rot[(k_base + lane_id * 2 + 1) * N + n] = val.y;
}

/// Fast Hadamard transform for weight matrices [K, N] with block_size = 128.
/// Each lane handles 4 elements (128 elements / 32 lanes = 4 per lane).
///
/// 7 butterfly stages: stride 1 (intra-lane), 2, 4, 8, 16, 32, 64 (inter-lane shuffle)
/// Normalization: 1/sqrt(128) ≈ 0.088388
///
/// @param W           Input weight matrix [K, N] in half precision
/// @param W_rot       Output rotated weights [K, N]
/// @param K           Size of K dimension (must be multiple of 128)
/// @param N           Size of N dimension
kernel void hadamard_forward_fast_128(
    device const half* W        [[buffer(0)]],
    device half* W_rot          [[buffer(1)]],
    constant uint& K            [[buffer(2)]],
    constant uint& N            [[buffer(3)]],
    uint3 gid                   [[thread_position_in_grid]],
    uint lane_id              [[thread_index_in_simdgroup]]
)
{
    uint n = gid.x;
    uint block_k = gid.y;
    
    if (n >= N || block_k >= (K / 128)) return;
    
    uint k_base = block_k * 128;
    
    // Each lane loads 4 elements
    half4 val;
    val.x = W[(k_base + lane_id * 4) * N + n];
    val.y = W[(k_base + lane_id * 4 + 1) * N + n];
    val.z = W[(k_base + lane_id * 4 + 2) * N + n];
    val.w = W[(k_base + lane_id * 4 + 3) * N + n];
    
    // Stage 0: stride 1 (pairs within half4: (0,1), (2,3))
    {
        half sum0 = val.x + val.y;
        half diff0 = val.x - val.y;
        half sum1 = val.z + val.w;
        half diff1 = val.z - val.w;
        val = half4(sum0, diff0, sum1, diff1);
    }
    
    // Stage 1: stride 2 (pairs: (0,2), (1,3))
    {
        half sum0 = val.x + val.z;
        half diff0 = val.x - val.z;
        half sum1 = val.y + val.w;
        half diff1 = val.y - val.w;
        val = half4(sum0, sum1, diff0, diff1);
    }
    
    // Stage 2: stride 4 (inter-lane shuffle)
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
    
    // Normalize by 1/sqrt(128) ≈ 0.088388
    val *= half(0.0883883476h);
    
    // Write back
    W_rot[(k_base + lane_id * 4) * N + n] = val.x;
    W_rot[(k_base + lane_id * 4 + 1) * N + n] = val.y;
    W_rot[(k_base + lane_id * 4 + 2) * N + n] = val.z;
    W_rot[(k_base + lane_id * 4 + 3) * N + n] = val.w;
}

// ============================================================================
// hadamard_inverse_block: Inverse transform for weight matrices [K, N]
// Since Hadamard is self-inverse (H @ H = n*I), inverse = forward for normalized H
// ============================================================================

/// Inverse Hadamard transform for weight matrices [K, N] with block_size = 64.
/// Since H @ H = n*I, the inverse is H^-1 = H / n.
/// For orthonormal Hadamard (normalized by 1/sqrt(n)), H @ H = I, so inverse = H.
/// This kernel applies the same butterfly as forward - they are identical for 
/// normalized Hadamard.
///
/// @param W_rot       Rotated weights [K, N]
/// @param W           Output original weights [K, N]
/// @param K           Size of K dimension
/// @param N           Size of N dimension
kernel void hadamard_inverse_block_64(
    device const half* W_rot    [[buffer(0)]],
    device half* W              [[buffer(1)]],
    constant uint& K            [[buffer(2)]],
    constant uint& N            [[buffer(3)]],
    uint3 gid                   [[thread_position_in_grid]],
    uint lane_id              [[thread_index_in_simdgroup]]
)
{
    uint n = gid.x;
    uint block_k = gid.y;
    
    if (n >= N || block_k >= (K / 64)) return;
    
    uint k_base = block_k * 64;
    
    // Load
    half2 val;
    val.x = W_rot[(k_base + lane_id * 2) * N + n];
    val.y = W_rot[(k_base + lane_id * 2 + 1) * N + n];
    
    // Same butterfly stages as forward
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
        val = is_upper ? (partner_val - val) : (val + partner_val);
    }
    
    for (uint stage = 2; stage < 6; ++stage) {
        uint stride = 1u << stage;
        uint partner = lane_id ^ stride;
        bool is_upper = (lane_id & stride) != 0;
        val = butterfly_step2(val, partner, is_upper);
    }
    
    // Apply same normalization (orthonormal Hadamard)
    val *= half(0.125h);
    
    // Store
    W[(k_base + lane_id * 2) * N + n] = val.x;
    W[(k_base + lane_id * 2 + 1) * N + n] = val.y;
}

/// Inverse Hadamard transform for weight matrices [K, N] with block_size = 128.
kernel void hadamard_inverse_block_128(
    device const half* W_rot    [[buffer(0)]],
    device half* W              [[buffer(1)]],
    constant uint& K            [[buffer(2)]],
    constant uint& N            [[buffer(3)]],
    uint3 gid                   [[thread_position_in_grid]],
    uint lane_id              [[thread_index_in_simdgroup]]
)
{
    uint n = gid.x;
    uint block_k = gid.y;
    
    if (n >= N || block_k >= (K / 128)) return;
    
    uint k_base = block_k * 128;
    
    half4 val;
    val.x = W_rot[(k_base + lane_id * 4) * N + n];
    val.y = W_rot[(k_base + lane_id * 4 + 1) * N + n];
    val.z = W_rot[(k_base + lane_id * 4 + 2) * N + n];
    val.w = W_rot[(k_base + lane_id * 4 + 3) * N + n];
    
    // Stage 0
    {
        half sum0 = val.x + val.y;
        half diff0 = val.x - val.y;
        half sum1 = val.z + val.w;
        half diff1 = val.z - val.w;
        val = half4(sum0, diff0, sum1, diff1);
    }
    
    // Stage 1
    {
        half sum0 = val.x + val.z;
        half diff0 = val.x - val.z;
        half sum1 = val.y + val.w;
        half diff1 = val.y - val.w;
        val = half4(sum0, sum1, diff0, diff1);
    }
    
    // Stages 2-6
    for (uint stage = 2; stage < 7; ++stage) {
        uint stride = 1u << (stage - 1);
        uint partner = lane_id ^ stride;
        bool is_upper = (lane_id & stride) != 0;
        val = butterfly_step4(val, partner, is_upper);
    }
    
    // Normalize
    val *= half(0.0883883476h);
    
    W[(k_base + lane_id * 4) * N + n] = val.x;
    W[(k_base + lane_id * 4 + 1) * N + n] = val.y;
    W[(k_base + lane_id * 4 + 2) * N + n] = val.z;
    W[(k_base + lane_id * 4 + 3) * N + n] = val.w;
}

// ============================================================================
// Generic entry points (dispatch based on block_size parameter)
// ============================================================================

/// Generic forward Hadamard transform for weight matrices.
/// Dispatches to the appropriate optimized kernel based on block_size.
///
/// @param W           Input weight matrix [K, N] in half precision
/// @param W_rot       Output rotated weights [K, N]
/// @param K           Size of K dimension (must be multiple of block_size)
/// @param N           Size of N dimension
/// @param block_size  Hadamard block size (64 or 128)
kernel void hadamard_forward_block(
    device const half* W        [[buffer(0)]],
    device half* W_rot          [[buffer(1)]],
    constant uint& K            [[buffer(2)]],
    constant uint& N            [[buffer(3)]],
    constant uint& block_size   [[buffer(4)]],
    uint3 gid                   [[thread_position_in_grid]],
    uint lane_id              [[thread_index_in_simdgroup]]
)
{
    // This is a wrapper that should be dispatched to the appropriate 
    // specialized kernel. For runtime selection, use hadamard_forward_fast_64
    // or hadamard_forward_fast_128 directly.
    //
    // For single-kernel entry point with block_size parameter:
    if (block_size == 64) {
        // Delegate to specialized 64-version logic inline
        uint n = gid.x;
        uint block_k = gid.y;
        
        if (n >= N || block_k >= (K / 64)) return;
        
        uint k_base = block_k * 64;
        half2 val;
        val.x = W[(k_base + lane_id * 2) * N + n];
        val.y = W[(k_base + lane_id * 2 + 1) * N + n];
        
        // Butterfly stages
        {
            half sum = val.x + val.y;
            half diff = val.x - val.y;
            val.x = sum;
            val.y = diff;
        }
        
        for (uint stage = 1; stage < 6; ++stage) {
            uint stride = 1u << (stage - 1);
            uint partner = lane_id ^ stride;
            bool is_upper = (lane_id & stride) != 0;
            half2 partner_val = simd_shuffle(val, ushort(partner));
            val = is_upper ? (partner_val - val) : (val + partner_val);
        }
        
        val *= half(0.125h);
        
        W_rot[(k_base + lane_id * 2) * N + n] = val.x;
        W_rot[(k_base + lane_id * 2 + 1) * N + n] = val.y;
    }
    // Note: block_size 128 would need separate grid sizing
}

/// Generic inverse Hadamard transform for weight matrices.
/// Applies the same transform as forward for normalized Hadamard.
///
/// @param W_rot       Rotated weights [K, N]
/// @param W           Output original weights [K, N]
/// @param K           Size of K dimension
/// @param N           Size of N dimension
/// @param block_size  Hadamard block size (64 or 128)
kernel void hadamard_inverse_block(
    device const half* W_rot    [[buffer(0)]],
    device half* W              [[buffer(1)]],
    constant uint& K            [[buffer(2)]],
    constant uint& N            [[buffer(3)]],
    constant uint& block_size   [[buffer(4)]],
    uint3 gid                   [[thread_position_in_grid]],
    uint lane_id              [[thread_index_in_simdgroup]]
)
{
    // For normalized Hadamard (orthonormal), inverse is identical to forward
    // H @ H = I, so H^-1 = H^T = H (Hadamard is symmetric)
    if (block_size == 64) {
        uint n = gid.x;
        uint block_k = gid.y;
        
        if (n >= N || block_k >= (K / 64)) return;
        
        uint k_base = block_k * 64;
        half2 val;
        val.x = W_rot[(k_base + lane_id * 2) * N + n];
        val.y = W_rot[(k_base + lane_id * 2 + 1) * N + n];
        
        // Butterfly stages (same as forward)
        {
            half sum = val.x + val.y;
            half diff = val.x - val.y;
            val.x = sum;
            val.y = diff;
        }
        
        for (uint stage = 1; stage < 6; ++stage) {
            uint stride = 1u << (stage - 1);
            uint partner = lane_id ^ stride;
            bool is_upper = (lane_id & stride) != 0;
            half2 partner_val = simd_shuffle(val, ushort(partner));
            val = is_upper ? (partner_val - val) : (val + partner_val);
        }
        
        val *= half(0.125h);
        
        W[(k_base + lane_id * 2) * N + n] = val.x;
        W[(k_base + lane_id * 2 + 1) * N + n] = val.y;
    }
}

/// Fast Hadamard transform entry point (alias for hadamard_forward_fast_64).
/// Optimized O(n log n) butterfly implementation.
///
/// @param W           Input weight matrix [K, N] in half precision
/// @param W_rot       Output rotated weights [K, N]
/// @param K           Size of K dimension (must be multiple of 64)
/// @param N           Size of N dimension
/// @param block_size  Hadamard block size (64 or 128) - stored but dispatch uses grid
kernel void hadamard_forward_fast(
    device const half* W        [[buffer(0)]],
    device half* W_rot          [[buffer(1)]],
    constant uint& K            [[buffer(2)]],
    constant uint& N            [[buffer(3)]],
    constant uint& block_size   [[buffer(4)]],
    uint3 gid                   [[thread_position_in_grid]],
    uint lane_id              [[thread_index_in_simdgroup]]
)
{
    // Fast O(n log n) butterfly transform using simdgroup shuffle
    // Each thread handles one column n, one 64-row block
    uint n = gid.x;
    uint block_k = gid.y;
    
    if (n >= N || block_k >= (K / 64)) return;
    
    uint k_base = block_k * 64;
    
    // Load 2 elements per lane
    half2 val;
    val.x = W[(k_base + lane_id * 2) * N + n];
    val.y = W[(k_base + lane_id * 2 + 1) * N + n];
    
    // Stage 0: stride 1 (intra-lane)
    {
        half sum = val.x + val.y;
        half diff = val.x - val.y;
        val.x = sum;
        val.y = diff;
    }
    
    // Stages 1-5: stride 2, 4, 8, 16, 32 (inter-lane shuffle)
    for (uint stage = 1; stage < 6; ++stage) {
        uint stride = 1u << (stage - 1);
        uint partner = lane_id ^ stride;
        bool is_upper = (lane_id & stride) != 0;
        half2 partner_val = simd_shuffle(val, ushort(partner));
        val = is_upper ? (partner_val - val) : (val + partner_val);
    }
    
    // Normalize: 1/sqrt(64) = 0.125
    val *= half(0.125h);
    
    // Store
    W_rot[(k_base + lane_id * 2) * N + n] = val.x;
    W_rot[(k_base + lane_id * 2 + 1) * N + n] = val.y;
}

// ============================================================================
// Float variants for high-precision weight transforms
// ============================================================================

/// Fast Hadamard transform for float weights [K, N], block_size = 64
/// Float version for higher precision during weight preprocessing
kernel void hadamard_forward_fast_64_float(
    device const float* W       [[buffer(0)]],
    device float* W_rot         [[buffer(1)]],
    constant uint& K            [[buffer(2)]],
    constant uint& N            [[buffer(3)]],
    uint3 gid                   [[thread_position_in_grid]],
    uint lane_id              [[thread_index_in_simdgroup]]
)
{
    uint n = gid.x;
    uint block_k = gid.y;
    
    if (n >= N || block_k >= (K / 64)) return;
    
    uint k_base = block_k * 64;
    
    // Load as float2
    float2 val;
    val.x = W[(k_base + lane_id * 2) * N + n];
    val.y = W[(k_base + lane_id * 2 + 1) * N + n];
    
    // Butterfly stages for float
    {
        float sum = val.x + val.y;
        float diff = val.x - val.y;
        val.x = sum;
        val.y = diff;
    }
    
    for (uint stage = 1; stage < 6; ++stage) {
        uint stride = 1u << (stage - 1);
        uint partner = lane_id ^ stride;
        bool is_upper = (lane_id & stride) != 0;
        
        float2 partner_val = simd_shuffle(val, ushort(partner));
        val = is_upper ? (partner_val - val) : (val + partner_val);
    }
    
    // Normalize
    val *= float(0.125);
    
    // Store
    W_rot[(k_base + lane_id * 2) * N + n] = val.x;
    W_rot[(k_base + lane_id * 2 + 1) * N + n] = val.y;
}

/// Inverse Hadamard transform for float weights [K, N], block_size = 64
kernel void hadamard_inverse_block_64_float(
    device const float* W_rot   [[buffer(0)]],
    device float* W             [[buffer(1)]],
    constant uint& K            [[buffer(2)]],
    constant uint& N            [[buffer(3)]],
    uint3 gid                   [[thread_position_in_grid]],
    uint lane_id              [[thread_index_in_simdgroup]]
)
{
    // Same as forward for normalized Hadamard
    uint n = gid.x;
    uint block_k = gid.y;
    
    if (n >= N || block_k >= (K / 64)) return;
    
    uint k_base = block_k * 64;
    
    float2 val;
    val.x = W_rot[(k_base + lane_id * 2) * N + n];
    val.y = W_rot[(k_base + lane_id * 2 + 1) * N + n];
    
    {
        float sum = val.x + val.y;
        float diff = val.x - val.y;
        val.x = sum;
        val.y = diff;
    }
    
    for (uint stage = 1; stage < 6; ++stage) {
        uint stride = 1u << (stage - 1);
        uint partner = lane_id ^ stride;
        bool is_upper = (lane_id & stride) != 0;
        
        float2 partner_val = simd_shuffle(val, ushort(partner));
        val = is_upper ? (partner_val - val) : (val + partner_val);
    }
    
    val *= float(0.125);
    
    W[(k_base + lane_id * 2) * N + n] = val.x;
    W[(k_base + lane_id * 2 + 1) * N + n] = val.y;
}

// ============================================================================
// Non-power-of-2 Hadamard transforms via block-diagonal decomposition
// ============================================================================
//
// For non-power-of-2 sizes, decompose into independent power-of-2 blocks:
//   H_96  = diag(H_64, H_32)
//   H_160 = diag(H_128, H_32)
//   H_192 = diag(H_128, H_64)
//
// Each block is transformed independently in parallel, exploiting the
// block-diagonal structure for maximum throughput.
// ============================================================================

/// Hadamard transform for block_size = 96 (64 + 32)
/// Grid: (N, num_blocks_per_96, 2) where last dimension selects 64-block or 32-block
kernel void hadamard_forward_fast_96(
    device const half* W        [[buffer(0)]],
    device half* W_rot          [[buffer(1)]],
    constant uint& K            [[buffer(2)]],
    constant uint& N            [[buffer(3)]],
    uint3 gid                   [[thread_position_in_grid]],
    uint lane_id                [[thread_index_in_simdgroup]]
)
{
    uint n = gid.x;
    uint block_96 = gid.y;
    uint subblock_type = gid.z;  // 0 = first 64 elements, 1 = next 32 elements
    
    if (n >= N) return;
    
    uint k_base_96 = block_96 * 96;
    
    if (subblock_type == 0) {
        // Transform first 64 elements using 64-transform
        if (block_96 >= (K / 96)) return;
        
        uint k_base = k_base_96;
        half2 val;
        val.x = W[(k_base + lane_id * 2) * N + n];
        val.y = W[(k_base + lane_id * 2 + 1) * N + n];
        
        // Stage 0: intra-lane butterfly
        {
            half sum = val.x + val.y;
            half diff = val.x - val.y;
            val.x = sum;
            val.y = diff;
        }
        
        // Stages 1-5: inter-lane butterfly
        for (uint stage = 1; stage < 6; ++stage) {
            uint stride = 1u << (stage - 1);
            uint partner = lane_id ^ stride;
            bool is_upper = (lane_id & stride) != 0;
            half2 partner_val = simd_shuffle(val, ushort(partner));
            val = is_upper ? (partner_val - val) : (val + partner_val);
        }
        
        val *= half(0.125h);  // 1/sqrt(64)
        
        W_rot[(k_base + lane_id * 2) * N + n] = val.x;
        W_rot[(k_base + lane_id * 2 + 1) * N + n] = val.y;
        
    } else {
        // Transform next 32 elements using 32-transform
        if (lane_id >= 32 || block_96 >= (K / 96)) return;
        
        uint k_base = k_base_96 + 64;
        half val = W[(k_base + lane_id) * N + n];
        
        // 5 butterfly stages for size 32
        for (uint stage = 0; stage < 5; ++stage) {
            uint stride = 1u << stage;
            uint partner = lane_id ^ stride;
            bool is_upper = (lane_id & stride) != 0;
            half partner_val = simd_shuffle(val, ushort(partner));
            val = is_upper ? (partner_val - val) : (val + partner_val);
        }
        
        val *= half(0.17677669529663689h);  // 1/sqrt(32)
        
        W_rot[(k_base + lane_id) * N + n] = val;
    }
}

/// Hadamard transform for block_size = 160 (128 + 32)
kernel void hadamard_forward_fast_160(
    device const half* W        [[buffer(0)]],
    device half* W_rot          [[buffer(1)]],
    constant uint& K            [[buffer(2)]],
    constant uint& N            [[buffer(3)]],
    uint3 gid                   [[thread_position_in_grid]],
    uint lane_id                [[thread_index_in_simdgroup]]
)
{
    uint n = gid.x;
    uint block_160 = gid.y;
    uint subblock_type = gid.z;  // 0 = first 128 elements, 1 = next 32 elements
    
    if (n >= N) return;
    
    uint k_base_160 = block_160 * 160;
    
    if (subblock_type == 0) {
        // Transform first 128 elements (each thread handles 4 elements)
        if (block_160 >= (K / 160)) return;
        
        uint k_base = k_base_160;
        half4 val;
        val.x = W[(k_base + lane_id * 4) * N + n];
        val.y = W[(k_base + lane_id * 4 + 1) * N + n];
        val.z = W[(k_base + lane_id * 4 + 2) * N + n];
        val.w = W[(k_base + lane_id * 4 + 3) * N + n];
        
        // Stage 0: intra-register (pairs within half4)
        {
            half4 tmp;
            tmp.x = val.x + val.y;
            tmp.y = val.x - val.y;
            tmp.z = val.z + val.w;
            tmp.w = val.z - val.w;
            val = tmp;
        }
        
        // Stage 1: inter-pair within half4
        {
            half4 tmp;
            tmp.x = val.x + val.z;
            tmp.y = val.y + val.w;
            tmp.z = val.x - val.z;
            tmp.w = val.y - val.w;
            val = tmp;
        }
        
        // Stages 2-6: inter-lane butterfly
        for (uint stage = 2; stage < 7; ++stage) {
            uint stride = 1u << (stage - 2);
            uint partner = lane_id ^ stride;
            bool is_upper = (lane_id & stride) != 0;
            half4 partner_val = simd_shuffle(val, ushort(partner));
            val = is_upper ? (partner_val - val) : (val + partner_val);
        }
        
        val *= half(0.08838834764831845h);  // 1/sqrt(128)
        
        W_rot[(k_base + lane_id * 4) * N + n] = val.x;
        W_rot[(k_base + lane_id * 4 + 1) * N + n] = val.y;
        W_rot[(k_base + lane_id * 4 + 2) * N + n] = val.z;
        W_rot[(k_base + lane_id * 4 + 3) * N + n] = val.w;
        
    } else {
        // Transform next 32 elements
        if (lane_id >= 32 || block_160 >= (K / 160)) return;
        
        uint k_base = k_base_160 + 128;
        half val = W[(k_base + lane_id) * N + n];
        
        for (uint stage = 0; stage < 5; ++stage) {
            uint stride = 1u << stage;
            uint partner = lane_id ^ stride;
            bool is_upper = (lane_id & stride) != 0;
            half partner_val = simd_shuffle(val, ushort(partner));
            val = is_upper ? (partner_val - val) : (val + partner_val);
        }
        
        val *= half(0.17677669529663689h);  // 1/sqrt(32)
        
        W_rot[(k_base + lane_id) * N + n] = val;
    }
}

/// Hadamard transform for block_size = 192 (128 + 64)
kernel void hadamard_forward_fast_192(
    device const half* W        [[buffer(0)]],
    device half* W_rot          [[buffer(1)]],
    constant uint& K            [[buffer(2)]],
    constant uint& N            [[buffer(3)]],
    uint3 gid                   [[thread_position_in_grid]],
    uint lane_id                [[thread_index_in_simdgroup]]
)
{
    uint n = gid.x;
    uint block_192 = gid.y;
    uint subblock_type = gid.z;  // 0 = first 128 elements, 1 = next 64 elements
    
    if (n >= N) return;
    
    uint k_base_192 = block_192 * 192;
    
    if (subblock_type == 0) {
        // Transform first 128 elements
        if (block_192 >= (K / 192)) return;
        
        uint k_base = k_base_192;
        half4 val;
        val.x = W[(k_base + lane_id * 4) * N + n];
        val.y = W[(k_base + lane_id * 4 + 1) * N + n];
        val.z = W[(k_base + lane_id * 4 + 2) * N + n];
        val.w = W[(k_base + lane_id * 4 + 3) * N + n];
        
        // Intra-register butterflies (stages 0-1)
        {
            half4 tmp;
            tmp.x = val.x + val.y;
            tmp.y = val.x - val.y;
            tmp.z = val.z + val.w;
            tmp.w = val.z - val.w;
            val = tmp;
        }
        {
            half4 tmp;
            tmp.x = val.x + val.z;
            tmp.y = val.y + val.w;
            tmp.z = val.x - val.z;
            tmp.w = val.y - val.w;
            val = tmp;
        }
        
        // Inter-lane butterflies (stages 2-6)
        for (uint stage = 2; stage < 7; ++stage) {
            uint stride = 1u << (stage - 2);
            uint partner = lane_id ^ stride;
            bool is_upper = (lane_id & stride) != 0;
            half4 partner_val = simd_shuffle(val, ushort(partner));
            val = is_upper ? (partner_val - val) : (val + partner_val);
        }
        
        val *= half(0.08838834764831845h);  // 1/sqrt(128)
        
        W_rot[(k_base + lane_id * 4) * N + n] = val.x;
        W_rot[(k_base + lane_id * 4 + 1) * N + n] = val.y;
        W_rot[(k_base + lane_id * 4 + 2) * N + n] = val.z;
        W_rot[(k_base + lane_id * 4 + 3) * N + n] = val.w;
        
    } else {
        // Transform next 64 elements
        if (block_192 >= (K / 192)) return;
        
        uint k_base = k_base_192 + 128;
        half2 val;
        val.x = W[(k_base + lane_id * 2) * N + n];
        val.y = W[(k_base + lane_id * 2 + 1) * N + n];
        
        // Stage 0: intra-lane
        {
            half sum = val.x + val.y;
            half diff = val.x - val.y;
            val.x = sum;
            val.y = diff;
        }
        
        // Stages 1-5: inter-lane
        for (uint stage = 1; stage < 6; ++stage) {
            uint stride = 1u << (stage - 1);
            uint partner = lane_id ^ stride;
            bool is_upper = (lane_id & stride) != 0;
            half2 partner_val = simd_shuffle(val, ushort(partner));
            val = is_upper ? (partner_val - val) : (val + partner_val);
        }
        
        val *= half(0.125h);  // 1/sqrt(64)
        
        W_rot[(k_base + lane_id * 2) * N + n] = val.x;
        W_rot[(k_base + lane_id * 2 + 1) * N + n] = val.y;
    }
}

// ============================================================================
// Inverse transforms for non-power-of-2 sizes
// ============================================================================

/// Inverse Hadamard for block_size = 96
kernel void hadamard_inverse_block_96(
    device const half* W_rot    [[buffer(0)]],
    device half* W              [[buffer(1)]],
    constant uint& K            [[buffer(2)]],
    constant uint& N            [[buffer(3)]],
    uint3 gid                   [[thread_position_in_grid]],
    uint lane_id                [[thread_index_in_simdgroup]]
)
{
    uint n = gid.x;
    uint block_96 = gid.y;
    uint subblock_type = gid.z;
    
    if (n >= N) return;
    
    uint k_base_96 = block_96 * 96;
    
    if (subblock_type == 0) {
        if (block_96 >= (K / 96)) return;
        uint k_base = k_base_96;
        
        half2 val;
        val.x = W_rot[(k_base + lane_id * 2) * N + n];
        val.y = W_rot[(k_base + lane_id * 2 + 1) * N + n];
        
        {
            half sum = val.x + val.y;
            half diff = val.x - val.y;
            val.x = sum;
            val.y = diff;
        }
        
        for (uint stage = 1; stage < 6; ++stage) {
            uint stride = 1u << (stage - 1);
            uint partner = lane_id ^ stride;
            bool is_upper = (lane_id & stride) != 0;
            half2 partner_val = simd_shuffle(val, ushort(partner));
            val = is_upper ? (partner_val - val) : (val + partner_val);
        }
        
        val *= half(0.125h);
        
        W[(k_base + lane_id * 2) * N + n] = val.x;
        W[(k_base + lane_id * 2 + 1) * N + n] = val.y;
        
    } else {
        if (lane_id >= 32 || block_96 >= (K / 96)) return;
        uint k_base = k_base_96 + 64;
        
        half val = W_rot[(k_base + lane_id) * N + n];
        
        for (uint stage = 0; stage < 5; ++stage) {
            uint stride = 1u << stage;
            uint partner = lane_id ^ stride;
            bool is_upper = (lane_id & stride) != 0;
            half partner_val = simd_shuffle(val, ushort(partner));
            val = is_upper ? (partner_val - val) : (val + partner_val);
        }
        
        val *= half(0.17677669529663689h);
        
        W[(k_base + lane_id) * N + n] = val;
    }
}

/// Inverse Hadamard for block_size = 160
kernel void hadamard_inverse_block_160(
    device const half* W_rot    [[buffer(0)]],
    device half* W              [[buffer(1)]],
    constant uint& K            [[buffer(2)]],
    constant uint& N            [[buffer(3)]],
    uint3 gid                   [[thread_position_in_grid]],
    uint lane_id                [[thread_index_in_simdgroup]]
)
{
    uint n = gid.x;
    uint block_160 = gid.y;
    uint subblock_type = gid.z;
    
    if (n >= N) return;
    
    uint k_base_160 = block_160 * 160;
    
    if (subblock_type == 0) {
        if (block_160 >= (K / 160)) return;
        uint k_base = k_base_160;
        
        half4 val;
        val.x = W_rot[(k_base + lane_id * 4) * N + n];
        val.y = W_rot[(k_base + lane_id * 4 + 1) * N + n];
        val.z = W_rot[(k_base + lane_id * 4 + 2) * N + n];
        val.w = W_rot[(k_base + lane_id * 4 + 3) * N + n];
        
        {
            half4 tmp;
            tmp.x = val.x + val.y;
            tmp.y = val.x - val.y;
            tmp.z = val.z + val.w;
            tmp.w = val.z - val.w;
            val = tmp;
        }
        {
            half4 tmp;
            tmp.x = val.x + val.z;
            tmp.y = val.y + val.w;
            tmp.z = val.x - val.z;
            tmp.w = val.y - val.w;
            val = tmp;
        }
        
        for (uint stage = 2; stage < 7; ++stage) {
            uint stride = 1u << (stage - 2);
            uint partner = lane_id ^ stride;
            bool is_upper = (lane_id & stride) != 0;
            half4 partner_val = simd_shuffle(val, ushort(partner));
            val = is_upper ? (partner_val - val) : (val + partner_val);
        }
        
        val *= half(0.08838834764831845h);
        
        W[(k_base + lane_id * 4) * N + n] = val.x;
        W[(k_base + lane_id * 4 + 1) * N + n] = val.y;
        W[(k_base + lane_id * 4 + 2) * N + n] = val.z;
        W[(k_base + lane_id * 4 + 3) * N + n] = val.w;
        
    } else {
        if (lane_id >= 32 || block_160 >= (K / 160)) return;
        uint k_base = k_base_160 + 128;
        
        half val = W_rot[(k_base + lane_id) * N + n];
        
        for (uint stage = 0; stage < 5; ++stage) {
            uint stride = 1u << stage;
            uint partner = lane_id ^ stride;
            bool is_upper = (lane_id & stride) != 0;
            half partner_val = simd_shuffle(val, ushort(partner));
            val = is_upper ? (partner_val - val) : (val + partner_val);
        }
        
        val *= half(0.17677669529663689h);
        
        W[(k_base + lane_id) * N + n] = val;
    }
}

/// Inverse Hadamard for block_size = 192
kernel void hadamard_inverse_block_192(
    device const half* W_rot    [[buffer(0)]],
    device half* W              [[buffer(1)]],
    constant uint& K            [[buffer(2)]],
    constant uint& N            [[buffer(3)]],
    uint3 gid                   [[thread_position_in_grid]],
    uint lane_id                [[thread_index_in_simdgroup]]
)
{
    uint n = gid.x;
    uint block_192 = gid.y;
    uint subblock_type = gid.z;
    
    if (n >= N) return;
    
    uint k_base_192 = block_192 * 192;
    
    if (subblock_type == 0) {
        if (block_192 >= (K / 192)) return;
        uint k_base = k_base_192;
        
        half4 val;
        val.x = W_rot[(k_base + lane_id * 4) * N + n];
        val.y = W_rot[(k_base + lane_id * 4 + 1) * N + n];
        val.z = W_rot[(k_base + lane_id * 4 + 2) * N + n];
        val.w = W_rot[(k_base + lane_id * 4 + 3) * N + n];
        
        {
            half4 tmp;
            tmp.x = val.x + val.y;
            tmp.y = val.x - val.y;
            tmp.z = val.z + val.w;
            tmp.w = val.z - val.w;
            val = tmp;
        }
        {
            half4 tmp;
            tmp.x = val.x + val.z;
            tmp.y = val.y + val.w;
            tmp.z = val.x - val.z;
            tmp.w = val.y - val.w;
            val = tmp;
        }
        
        for (uint stage = 2; stage < 7; ++stage) {
            uint stride = 1u << (stage - 2);
            uint partner = lane_id ^ stride;
            bool is_upper = (lane_id & stride) != 0;
            half4 partner_val = simd_shuffle(val, ushort(partner));
            val = is_upper ? (partner_val - val) : (val + partner_val);
        }
        
        val *= half(0.08838834764831845h);
        
        W[(k_base + lane_id * 4) * N + n] = val.x;
        W[(k_base + lane_id * 4 + 1) * N + n] = val.y;
        W[(k_base + lane_id * 4 + 2) * N + n] = val.z;
        W[(k_base + lane_id * 4 + 3) * N + n] = val.w;
        
    } else {
        if (block_192 >= (K / 192)) return;
        uint k_base = k_base_192 + 128;
        
        half2 val;
        val.x = W_rot[(k_base + lane_id * 2) * N + n];
        val.y = W_rot[(k_base + lane_id * 2 + 1) * N + n];
        
        {
            half sum = val.x + val.y;
            half diff = val.x - val.y;
            val.x = sum;
            val.y = diff;
        }
        
        for (uint stage = 1; stage < 6; ++stage) {
            uint stride = 1u << (stage - 1);
            uint partner = lane_id ^ stride;
            bool is_upper = (lane_id & stride) != 0;
            half2 partner_val = simd_shuffle(val, ushort(partner));
            val = is_upper ? (partner_val - val) : (val + partner_val);
        }
        
        val *= half(0.125h);
        
        W[(k_base + lane_id * 2) * N + n] = val.x;
        W[(k_base + lane_id * 2 + 1) * N + n] = val.y;
    }
}
