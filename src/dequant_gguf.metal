#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Q4_K Dequantization for Apple Metal
// ============================================================================
//
// Q4_K is llama.cpp's 4-bit quantization format with super-block scaling.
//
// Block Structure (32 weights = 128 bits):
//   - 16 bytes of packed 4-bit weights
//   - 1 byte with 2 scales (4 bits each)
//   - 1 byte with min value (4 bits) + reserved
//   - Total: 18 bytes per block
//
// Scale Encoding:
//   scale[i] = exp2(scale_bits[i]) / 16.0
//
// Min Encoding:
//   min = -exp2(min_bits)
//
// Dequantization:
//   weight = (packed_weight - 8) * scale[block_idx/16] + min
//
// Super-block Structure:
//   - 8 blocks of 32 weights = 256 weights per super-block
//   - Each super-block has 16 scale values (2 per block)
//   - First 16 weights use scale[0], next 16 use scale[1], etc.
// ============================================================================

// ============================================================================
// Constants
// ============================================================================

constant constexpr uint Q4_K_BLOCK_SIZE = 32u;
constant constexpr uint Q4_K_PACKED_SIZE = 18u;  // bytes per block
constant constexpr uint Q4_K_SUPERBLOCK_SIZE = 256u;  // weights per superblock

// ============================================================================
// Q4_K unpacking
// ============================================================================

/// Unpack 32 packed 4-bit weights from 16 bytes.
/// Returns 8 uint32 values, each containing 4 unpacked 4-bit weights
/// in the lower 16 bits of each uint32.
inline void unpack_q4_k_weights(
    device const uchar *packed_block,
    thread uint32_t *weights_lo,
    thread uint32_t *weights_hi)
{
    // Load 16 packed bytes
    uchar p0 = packed_block[0];
    uchar p1 = packed_block[1];
    uchar p2 = packed_block[2];
    uchar p3 = packed_block[3];
    uchar p4 = packed_block[4];
    uchar p5 = packed_block[5];
    uchar p6 = packed_block[6];
    uchar p7 = packed_block[7];
    uchar p8 = packed_block[8];
    uchar p9 = packed_block[9];
    uchar p10 = packed_block[10];
    uchar p11 = packed_block[11];
    uchar p12 = packed_block[12];
    uchar p13 = packed_block[13];
    uchar p14 = packed_block[14];
    uchar p15 = packed_block[15];

    // Unpack bytes to 4-bit weights (LSB-first)
    // Each byte gives 2 weights: lo nibble (bits 0-3), hi nibble (bits 4-7)
    weights_lo[0] = uint32_t(p0 & 0x0Fu) |
                   (uint32_t(p1 & 0x0Fu) << 4) |
                   (uint32_t(p2 & 0x0Fu) << 8) |
                   (uint32_t(p3 & 0x0Fu) << 12);
    weights_hi[0] = uint32_t((p0 >> 4) & 0x0Fu) |
                   (uint32_t((p1 >> 4) & 0x0Fu) << 4) |
                   (uint32_t((p2 >> 4) & 0x0Fu) << 8) |
                   (uint32_t((p3 >> 4) & 0x0Fu) << 12);

    weights_lo[1] = uint32_t(p4 & 0x0Fu) |
                   (uint32_t(p5 & 0x0Fu) << 4) |
                   (uint32_t(p6 & 0x0Fu) << 8) |
                   (uint32_t(p7 & 0x0Fu) << 12);
    weights_hi[1] = uint32_t((p4 >> 4) & 0x0Fu) |
                   (uint32_t((p5 >> 4) & 0x0Fu) << 4) |
                   (uint32_t((p6 >> 4) & 0x0Fu) << 8) |
                   (uint32_t((p7 >> 4) & 0x0Fu) << 12);

    weights_lo[2] = uint32_t(p8 & 0x0Fu) |
                   (uint32_t(p9 & 0x0Fu) << 4) |
                   (uint32_t(p10 & 0x0Fu) << 8) |
                   (uint32_t(p11 & 0x0Fu) << 12);
    weights_hi[2] = uint32_t((p8 >> 4) & 0x0Fu) |
                   (uint32_t((p9 >> 4) & 0x0Fu) << 4) |
                   (uint32_t((p10 >> 4) & 0x0Fu) << 8) |
                   (uint32_t((p11 >> 4) & 0x0Fu) << 12);

    weights_lo[3] = uint32_t(p12 & 0x0Fu) |
                   (uint32_t(p13 & 0x0Fu) << 4) |
                   (uint32_t(p14 & 0x0Fu) << 8) |
                   (uint32_t(p15 & 0x0Fu) << 12);
    weights_hi[3] = uint32_t((p12 >> 4) & 0x0Fu) |
                   (uint32_t((p13 >> 4) & 0x0Fu) << 4) |
                   (uint32_t((p14 >> 4) & 0x0Fu) << 8) |
                   (uint32_t((p15 >> 4) & 0x0Fu) << 12);
}

/// Decode Q4_K scale from 4-bit encoded value.
/// scale = exp2(scale_bits) / 16.0
inline half decode_q4_k_scale(uint scale_bits) {
    float exp = exp2(float(scale_bits));
    return half(exp / 16.0f);
}

/// Decode Q4_K min value from 4-bit encoded value.
/// min = -exp2(min_bits)
inline half decode_q4_k_min(uint min_bits) {
    float val = exp2(float(min_bits));
    return -half(val);
}

// ============================================================================
// Per-block dequantization
// ============================================================================

/// Dequantize a single Q4_K block (32 weights).
///
/// @param packed_block  18 bytes of Q4_K block data
/// @param scale_lo       First scale (for weights 0-15)
/// @param scale_hi       Second scale (for weights 16-31)
/// @param min_val        Min value for the block
/// @param output         Output buffer (at least 32 half values)
inline void dequant_q4_k_block(
    device const uchar *packed_block,
    half scale_lo,
    half scale_hi,
    half min_val,
    thread half *output)
{
    // Unpack packed weights
    uint32_t weights_lo[4], weights_hi[4];
    unpack_q4_k_weights(packed_block, weights_lo, weights_hi);

    // Dequantize first 16 weights with scale_lo
    for (uint i = 0u; i < 16u; i++) {
        uint32_t weight_uint;
        if (i < 4u) {
            weight_uint = (weights_lo[0] >> (i * 4u)) & 0x0Fu;
        } else if (i < 8u) {
            weight_uint = (weights_lo[1] >> ((i - 4u) * 4u)) & 0x0Fu;
        } else if (i < 12u) {
            weight_uint = (weights_lo[2] >> ((i - 8u) * 4u)) & 0x0Fu;
        } else {
            weight_uint = (weights_lo[3] >> ((i - 12u) * 4u)) & 0x0Fu;
        }

        // Convert to signed: stored = actual + 8
        half weight_signed = half(float(weight_uint) - 8.0f);
        output[i] = weight_signed * scale_lo + min_val;
    }

    // Dequantize last 16 weights with scale_hi
    for (uint i = 0u; i < 16u; i++) {
        uint32_t weight_uint;
        if (i < 4u) {
            weight_uint = (weights_hi[0] >> (i * 4u)) & 0x0Fu;
        } else if (i < 8u) {
            weight_uint = (weights_hi[1] >> ((i - 4u) * 4u)) & 0x0Fu;
        } else if (i < 12u) {
            weight_uint = (weights_hi[2] >> ((i - 8u) * 4u)) & 0x0Fu;
        } else {
            weight_uint = (weights_hi[3] >> ((i - 12u) * 4u)) & 0x0Fu;
        }

        half weight_signed = half(float(weight_uint) - 8.0f);
        output[16u + i] = weight_signed * scale_hi + min_val;
    }
}

// ============================================================================
// Bulk dequantization kernel
// ============================================================================

/// Dequantize an entire Q4_K tensor to FP16.
///
/// @param packed_data  Input: N/32 blocks * 18 bytes
/// @param scales       Pre-extracted scale values [2 * num_blocks]
/// @param mins         Pre-extracted min values [num_blocks]
/// @param output       Output: N half values
/// @param num_elements Total number of elements (N)
kernel void dequant_q4_k_kernel(
    device const uchar *packed_data  [[buffer(0)]],
    device const half *scales        [[buffer(1)]],
    device const half *mins          [[buffer(2)]],
    device half *output             [[buffer(3)]],
    constant uint32_t &num_elements [[buffer(4)]],
    uint tid                        [[thread_position_in_grid]])
{
    // Each thread processes one Q4_K block (32 weights)
    uint block_idx = tid;
    uint base_idx = block_idx * Q4_K_BLOCK_SIZE;

    if (base_idx >= num_elements) return;

    // Get scale and min for this block
    half scale_lo = scales[block_idx * 2u];
    half scale_hi = scales[block_idx * 2u + 1u];
    half min_val = mins[block_idx];

    // Get packed data for this block
    device const uchar *block_data = packed_data + block_idx * Q4_K_PACKED_SIZE;

    // Temporary storage for 32 dequantized values
    half block_output[32];

    // Dequantize block
    dequant_q4_k_block(block_data, scale_lo, scale_hi, min_val, block_output);

    // Write output with boundary check
    uint remaining = min(Q4_K_BLOCK_SIZE, num_elements - base_idx);
    for (uint i = 0u; i < remaining; i++) {
        output[base_idx + i] = block_output[i];
    }
}

// ============================================================================
// Optimized kernel: Vectorized half4 stores
// ============================================================================

/// High-throughput variant using half4 vectorized stores.
/// Requires num_elements is a multiple of 32.
kernel void dequant_q4_k_aligned_kernel(
    device const uchar *packed_data  [[buffer(0)]],
    device const half *scales        [[buffer(1)]],
    device const half *mins          [[buffer(2)]],
    device half4 *output            [[buffer(3)]],
    constant uint32_t &num_blocks   [[buffer(4)]],
    uint tid                        [[thread_position_in_grid]])
{
    if (tid >= num_blocks) return;

    // Get scale and min for this block
    half scale_lo = scales[tid * 2u];
    half scale_hi = scales[tid * 2u + 1u];
    half min_val = mins[tid];

    // Get packed data for this block
    device const uchar *block_data = packed_data + tid * Q4_K_PACKED_SIZE;

    // Dequantize block
    half block_output[32];
    dequant_q4_k_block(block_data, scale_lo, scale_hi, min_val, block_output);

    // Write as 8 half4 vectors (32 values)
    uint out_base = tid * 8u;
    for (uint i = 0u; i < 8u; i++) {
        output[out_base + i] = half4(
            block_output[i * 4u + 0u],
            block_output[i * 4u + 1u],
            block_output[i * 4u + 2u],
            block_output[i * 4u + 3u]
        );
    }
}

// ============================================================================
// 2D matrix dequantization (for GEMM weights)
// ============================================================================

/// Dequantize a 2D weight matrix stored in Q4_K format.
///
/// Layout:
///   packed_data[num_blocks][18]  -- 18 bytes per Q4_K block
///   scales[num_blocks][2]        -- 2 scales per block
///   mins[num_blocks]              -- 1 min per block
///   output[K][N]                -- dequantized FP16 matrix
///
/// Each Q4_K block contains 32 weights along the K dimension.
/// Scales and mins are per-block.
///
/// @param packed_data  [num_blocks * 18] packed Q4_K weights
/// @param scales       [num_blocks * 2] scale values
/// @param mins         [num_blocks] min values
/// @param output       [K, N] dequantized FP16 output
/// @param K            Reduction dimension size
/// @param N            Output columns
kernel void dequant_q4_k_2d_kernel(
    device const uchar *packed_data  [[buffer(0)]],
    device const half *scales        [[buffer(1)]],
    device const half *mins          [[buffer(2)]],
    device half *output             [[buffer(3)]],
    constant uint32_t &K            [[buffer(4)]],
    constant uint32_t &N            [[buffer(5)]],
    uint2 gid                      [[thread_position_in_grid]])
{
    // gid.x = column index (along N)
    // gid.y = block index along K (each block = 32 weights)
    uint n_idx = gid.x;
    uint k_block = gid.y;

    if (n_idx >= N || k_block * Q4_K_BLOCK_SIZE >= K) return;

    // Calculate output position
    uint out_base = k_block * Q4_K_BLOCK_SIZE * N + n_idx;

    // Get scale and min for this block
    half scale_lo = scales[k_block * 2u];
    half scale_hi = scales[k_block * 2u + 1u];
    half min_val = mins[k_block];

    // Get packed data for this block
    device const uchar *block_data = packed_data + k_block * Q4_K_PACKED_SIZE;

    // Dequantize block
    half block_output[32];
    dequant_q4_k_block(block_data, scale_lo, scale_hi, min_val, block_output);

    // Write output (column-major within K dimension)
    uint k_remain = min(Q4_K_BLOCK_SIZE, K - k_block * Q4_K_BLOCK_SIZE);
    for (uint i = 0u; i < k_remain; i++) {
        output[out_base + i * N] = block_output[i];
    }
}

// ============================================================================
// Unit test kernels
// ============================================================================

/// Test kernel: dequantize a single Q4_K block.
/// Single-threaded for deterministic testing.
kernel void test_q4_k_block(
    device const uchar *packed_input [[buffer(0)]],
    device const half *scale_lo      [[buffer(1)]],
    device const half *scale_hi      [[buffer(2)]],
    device const half *min_val       [[buffer(3)]],
    device half *output            [[buffer(4)]],
    uint tid                       [[thread_position_in_grid]])
{
    if (tid > 0u) return;

    half block_output[32];
    dequant_q4_k_block(
        packed_input,
        scale_lo[0],
        scale_hi[0],
        min_val[0],
        block_output
    );

    // Copy to output buffer
    for(int i = 0; i < 32; i++) {
        output[i] = block_output[i];
    }
}

/// Test kernel: verify scale decoding.
/// Dequantizes all 16 possible 4-bit scale values.
kernel void test_q4_k_scale_decoding(
    device half *output [[buffer(0)]],
    uint tid           [[thread_position_in_grid]])
{
    if (tid >= 16u) return;

    uint scale_bits = uint(tid);
    output[tid] = decode_q4_k_scale(scale_bits);
}

/// Test kernel: verify min decoding.
/// Dequantizes all 16 possible 4-bit min values.
kernel void test_q4_k_min_decoding(
    device half *output [[buffer(0)]],
    uint tid           [[thread_position_in_grid]])
{
    if (tid >= 16u) return;

    uint min_bits = uint(tid);
    output[tid] = decode_q4_k_min(min_bits);
}

// ============================================================================
// Benchmark kernels
// ============================================================================

constant constexpr uint BENCH_ITERS_PER_THREAD = 16u;

/// Benchmark: scalar Q4_K dequant path.
/// Each thread processes BENCH_ITERS_PER_THREAD consecutive Q4_K blocks.
kernel void bench_q4_k_scalar(
    device const uchar *packed_data  [[buffer(0)]],
    device const half *scales        [[buffer(1)]],
    device const half *mins          [[buffer(2)]],
    device half4 *output            [[buffer(3)]],
    constant uint32_t &num_blocks   [[buffer(4)]],
    uint tid                        [[thread_position_in_grid]])
{
    uint base = tid * BENCH_ITERS_PER_THREAD;
    if (base >= num_blocks) return;

    uint iters = min(BENCH_ITERS_PER_THREAD, num_blocks - base);

    for (uint i = 0u; i < iters; i++) {
        uint block_idx = base + i;
        half scale_lo = scales[block_idx * 2u];
        half scale_hi = scales[block_idx * 2u + 1u];
        half min_val = mins[block_idx];

        device const uchar *block_data = packed_data + block_idx * Q4_K_PACKED_SIZE;

        half block_output[32];
        dequant_q4_k_block(block_data, scale_lo, scale_hi, min_val, block_output);

        uint out_base = block_idx * 8u;
        for (uint j = 0u; j < 8u; j++) {
            output[out_base + j] = half4(
                block_output[j * 4u + 0u],
                block_output[j * 4u + 1u],
                block_output[j * 4u + 2u],
                block_output[j * 4u + 3u]
            );
        }
    }
}