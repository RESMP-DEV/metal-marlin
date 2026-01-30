// norm_linear.metal - Fused RMSNorm + Quantized Linear for Apple Metal
//
// Fuses RMSNorm normalization with a quantized linear projection to eliminate
// the intermediate activation write/read between normalization and matrix multiply.
//
// Standard flow (memory-bound):
//   1. RMSNorm kernel writes normalized output to DRAM
//   2. Linear kernel reads normalized input from DRAM
//   Memory traffic: 2 * batch * seq * hidden * sizeof(half) = 2x bandwidth
//
// Fused flow (register-resident):
//   1. Single kernel computes norm in-place, applies linear directly
//   Memory traffic: 1 * batch * seq * hidden (input) + output
//   Savings: ~50% memory bandwidth for the normalization stage
//
// This is particularly effective for:
//   - QKV projection: RMSNorm + 3 separate projections fused
//   - MLP entry: RMSNorm + gate_proj + up_proj fused
//
// Design considerations:
//   - RMSNorm is embarrassingly parallel across tokens (each position independent)
//   - Quantized GEMM tiles require careful synchronization
//   - We compute RMSNorm once per token, then reuse the normalized value
//     across multiple output tiles
//
// Kernel variants:
//   1. norm_linear_fp4     - Fused RMSNorm + FP4 quantized GEMM
//   2. norm_linear_int4    - Fused RMSNorm + INT4 quantized GEMM
//   3. norm_qkv_fp4        - Fused RMSNorm + QKV projection (3 outputs)
//   4. norm_qkv_int4       - Fused RMSNorm + QKV projection (3 outputs)
//
// Memory layout:
//   Input:   [batch * seq, hidden_size] row-major
//   Weights: [hidden_size/8, out_features] packed FP4/INT4 (K-major)
//   Scales:  [hidden_size/group_size, out_features]
//   Gamma:   [hidden_size] RMSNorm weight
//   Output:  [batch * seq, out_features]
//
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Tile dimensions
// ============================================================================

// Threads per threadgroup for fused kernels
// 128 threads = 4 simdgroups for good occupancy on Apple GPUs
constant constexpr uint FUSED_TG_SIZE = 128;
constant constexpr uint SIMDGROUP_SIZE = 32;

// GEMM tile dimensions - balanced for register pressure vs. reuse
// M-tile: tokens processed per threadgroup
// N-tile: output features per threadgroup
// K-tile: input features per K-loop iteration
constant constexpr uint TILE_M = 16;
constant constexpr uint TILE_N = 64;
constant constexpr uint TILE_K = 32;  // Matches common group_size

// For RMSNorm reduction, we need enough threads to cover hidden_size
// Typical hidden_size: 2048, 4096, 8192
// Each thread handles hidden_size / FUSED_TG_SIZE elements
constant constexpr uint MAX_HIDDEN_PER_THREAD = 64;  // Supports up to 8192

// FP4 packing: 8 values per uint32
constant constexpr uint FP4_PER_UINT = 8;

// Magic numbers for INT4 dequantization
constant constexpr uint32_t MAGIC_BIAS_U32 = 0x64006400u;
constant constexpr uint32_t LO_NIBBLE_MASK = 0x000F000Fu;
constant constexpr uint16_t MAGIC_BIAS_F16 = 0x6400u;

// ============================================================================
// FP4 E2M1 dequantization (branchless)
// ============================================================================

inline half dequant_fp4_scalar(uint nibble) {
    uint S = (nibble >> 3) & 1u;
    uint E = (nibble >> 1) & 3u;
    uint M = nibble & 1u;

    bool is_normal = (E != 0u);
    uint fp16_exp = select(14u * M, E + 14u, is_normal);
    uint fp16_mant = select(0u, M << 9, is_normal);

    ushort fp16_bits = ushort((S << 15) | (fp16_exp << 10) | fp16_mant);
    return as_type<half>(fp16_bits);
}

inline void dequant_fp4_x8_scaled(uint32_t packed, half scale,
                                   thread half4 &out_lo, thread half4 &out_hi) {
    half vals[8];
    vals[0] = dequant_fp4_scalar((packed >>  0) & 0xFu);
    vals[1] = dequant_fp4_scalar((packed >>  4) & 0xFu);
    vals[2] = dequant_fp4_scalar((packed >>  8) & 0xFu);
    vals[3] = dequant_fp4_scalar((packed >> 12) & 0xFu);
    vals[4] = dequant_fp4_scalar((packed >> 16) & 0xFu);
    vals[5] = dequant_fp4_scalar((packed >> 20) & 0xFu);
    vals[6] = dequant_fp4_scalar((packed >> 24) & 0xFu);
    vals[7] = dequant_fp4_scalar((packed >> 28) & 0xFu);

    out_lo = half4(vals[0], vals[1], vals[2], vals[3]) * scale;
    out_hi = half4(vals[4], vals[5], vals[6], vals[7]) * scale;
}

// ============================================================================
// INT4 dequantization (magic bias trick)
// ============================================================================

inline void dequant_u4x8(uint32_t packed, half scale, half zero_point,
                          thread half4 &out_lo, thread half4 &out_hi) {
    uint32_t n0_biased = (packed & LO_NIBBLE_MASK) | MAGIC_BIAS_U32;
    half2 n0_pair = as_type<half2>(n0_biased) - as_type<half2>(MAGIC_BIAS_U32);

    uint32_t n1_biased = ((packed >> 4u) & LO_NIBBLE_MASK) | MAGIC_BIAS_U32;
    half2 n1_pair = as_type<half2>(n1_biased) - as_type<half2>(MAGIC_BIAS_U32);

    uint32_t n2_biased = ((packed >> 8u) & LO_NIBBLE_MASK) | MAGIC_BIAS_U32;
    half2 n2_pair = as_type<half2>(n2_biased) - as_type<half2>(MAGIC_BIAS_U32);

    uint32_t n3_biased = ((packed >> 12u) & LO_NIBBLE_MASK) | MAGIC_BIAS_U32;
    half2 n3_pair = as_type<half2>(n3_biased) - as_type<half2>(MAGIC_BIAS_U32);

    out_lo = half4(n0_pair.x, n1_pair.x, n2_pair.x, n3_pair.x);
    out_hi = half4(n0_pair.y, n1_pair.y, n2_pair.y, n3_pair.y);

    out_lo = (out_lo - zero_point) * scale;
    out_hi = (out_hi - zero_point) * scale;
}

// ============================================================================
// Simdgroup reduction utilities
// ============================================================================

inline float simd_sum_f32(float val) {
    val += simd_shuffle_xor(val, 16);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);
    return val;
}

// ============================================================================
// Fused RMSNorm + FP4 Linear Projection
//
// Computes: output = Linear(RMSNorm(input, gamma))
//
// The RMSNorm and GEMM are fused so the normalized values never hit DRAM.
// Each threadgroup processes TILE_M tokens, computing one output tile.
//
// Algorithm:
//   1. Cooperatively load input tile into threadgroup memory
//   2. Compute RMSNorm per-token (reduction across hidden_dim)
//   3. Stream through weights, computing dot products on normalized values
//   4. Write output tile
// ============================================================================

kernel void norm_linear_fp4(
    device const half* input           [[buffer(0)]],  // [M, K]
    device const half* gamma           [[buffer(1)]],  // [K] RMSNorm weight
    device const uint* weight_packed   [[buffer(2)]],  // [K/8, N] packed FP4
    device const half* scales          [[buffer(3)]],  // [K/group_size, N]
    device half* output                [[buffer(4)]],  // [M, N]
    constant uint& M                   [[buffer(5)]],  // num tokens
    constant uint& K                   [[buffer(6)]],  // hidden_size
    constant uint& N                   [[buffer(7)]],  // out_features
    constant uint& group_size          [[buffer(8)]],
    constant float& rms_eps            [[buffer(9)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint tid_in_tg                     [[thread_index_in_threadgroup]],
    uint lane_id                       [[thread_index_in_simdgroup]],
    uint sg_id                         [[simdgroup_index_in_threadgroup]]
) {
    // Threadgroup mapping:
    //   tgid.x = output column block (0..ceil(N/TILE_N)-1)
    //   tgid.y = input row block (0..ceil(M/TILE_M)-1)
    const uint n_block = tgid.x * TILE_N;
    const uint m_block = tgid.y * TILE_M;

    // Shared memory for:
    // 1. Input tile: [TILE_M, K] - too large for typical K, so we stream
    // 2. Normalized input: [TILE_M, TILE_K] - K-tile at a time
    // 3. RMS denominators: [TILE_M] - one per token in tile
    threadgroup float rms_inv[TILE_M];
    threadgroup half input_tile[TILE_M][TILE_K];

    // Each thread in the simdgroup contributes to RMSNorm computation
    // We compute RMSNorm for TILE_M tokens cooperatively

    // Step 1: Compute RMS denominators for all tokens in this M-tile
    // This is a reduction across K for each of the TILE_M tokens
    const uint tokens_this_tile = min(TILE_M, M - m_block);

    // Each simdgroup handles multiple tokens if TILE_M > num_simdgroups
    const uint num_simdgroups = FUSED_TG_SIZE / SIMDGROUP_SIZE;
    const uint tokens_per_sg = (tokens_this_tile + num_simdgroups - 1) / num_simdgroups;

    for (uint local_token = 0; local_token < tokens_per_sg; ++local_token) {
        uint token_idx = sg_id * tokens_per_sg + local_token;
        if (token_idx >= tokens_this_tile) break;

        uint global_token = m_block + token_idx;
        if (global_token >= M) {
            if (lane_id == 0) rms_inv[token_idx] = 0.0f;
            continue;
        }

        // Compute sum of squares for this token
        // Each lane handles K / SIMDGROUP_SIZE elements
        float sum_sq = 0.0f;
        const uint elems_per_lane = (K + SIMDGROUP_SIZE - 1) / SIMDGROUP_SIZE;

        for (uint i = 0; i < elems_per_lane; ++i) {
            uint k_idx = lane_id + i * SIMDGROUP_SIZE;
            if (k_idx < K) {
                float val = float(input[global_token * K + k_idx]);
                sum_sq += val * val;
            }
        }

        // Reduce across simdgroup
        sum_sq = simd_sum_f32(sum_sq);

        // Compute inverse RMS
        if (lane_id == 0) {
            float variance = sum_sq / float(K);
            rms_inv[token_idx] = rsqrt(variance + rms_eps);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Main GEMM loop with fused normalization
    // We stream through K in TILE_K chunks, applying normalization on-the-fly

    // Output accumulators (each thread accumulates a subset of the N tile)
    // With 128 threads and TILE_N=64, each thread handles TILE_N/128 outputs
    // Simplified: we use a 2D thread layout where each computes one output
    const uint outputs_per_thread = max(1u, TILE_N / FUSED_TG_SIZE);

    float acc[TILE_M][4];  // Support up to 4 outputs per thread per token
    for (uint m = 0; m < TILE_M; ++m) {
        for (uint i = 0; i < outputs_per_thread; ++i) {
            acc[m][i] = 0.0f;
        }
    }

    // Which output columns this thread handles
    const uint my_n_start = (tid_in_tg * outputs_per_thread) % TILE_N + n_block;

    // K-loop: process TILE_K elements per iteration
    const uint k_packed_stride = K / FP4_PER_UINT;

    for (uint k_base = 0; k_base < K; k_base += TILE_K) {
        const uint k_tile_len = min(TILE_K, K - k_base);

        // Load and normalize input tile cooperatively
        // Each thread loads multiple elements
        const uint elems_to_load = tokens_this_tile * k_tile_len;
        const uint loads_per_thread = (elems_to_load + FUSED_TG_SIZE - 1) / FUSED_TG_SIZE;

        for (uint i = 0; i < loads_per_thread; ++i) {
            uint idx = tid_in_tg + i * FUSED_TG_SIZE;
            if (idx < elems_to_load) {
                uint local_m = idx / k_tile_len;
                uint local_k = idx % k_tile_len;
                uint global_m = m_block + local_m;
                uint global_k = k_base + local_k;

                if (global_m < M && global_k < K) {
                    // Load, normalize, and scale by gamma in one shot
                    float val = float(input[global_m * K + global_k]);
                    float normed = val * rms_inv[local_m];
                    float gam = float(gamma[global_k]);
                    input_tile[local_m][local_k] = half(normed * gam);
                } else {
                    input_tile[local_m][local_k] = half(0);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute dot products for this K-tile
        // Each thread computes its assigned output columns across all M rows
        for (uint out_i = 0; out_i < outputs_per_thread; ++out_i) {
            uint n_idx = my_n_start + out_i;
            if (n_idx >= N) continue;

            // Load weights for this output column, this K-tile
            // Weights are packed [K/8, N]
            for (uint k_offset = 0; k_offset < k_tile_len; k_offset += FP4_PER_UINT) {
                uint global_k = k_base + k_offset;
                uint k_block_idx = global_k / FP4_PER_UINT;
                uint group_idx = global_k / group_size;

                uint32_t packed = weight_packed[k_block_idx * N + n_idx];
                half scale = scales[group_idx * N + n_idx];

                // Dequantize 8 weights
                half4 w_lo, w_hi;
                dequant_fp4_x8_scaled(packed, scale, w_lo, w_hi);

                // Accumulate across all M tokens
                for (uint m = 0; m < tokens_this_tile; ++m) {
                    // Dot product with 8 input elements
                    float dot = 0.0f;
                    uint k_local = k_offset;

                    // Elements 0-3
                    if (k_local < k_tile_len) dot += float(input_tile[m][k_local]) * float(w_lo.x);
                    if (k_local + 1 < k_tile_len) dot += float(input_tile[m][k_local + 1]) * float(w_lo.y);
                    if (k_local + 2 < k_tile_len) dot += float(input_tile[m][k_local + 2]) * float(w_lo.z);
                    if (k_local + 3 < k_tile_len) dot += float(input_tile[m][k_local + 3]) * float(w_lo.w);

                    // Elements 4-7
                    if (k_local + 4 < k_tile_len) dot += float(input_tile[m][k_local + 4]) * float(w_hi.x);
                    if (k_local + 5 < k_tile_len) dot += float(input_tile[m][k_local + 5]) * float(w_hi.y);
                    if (k_local + 6 < k_tile_len) dot += float(input_tile[m][k_local + 6]) * float(w_hi.z);
                    if (k_local + 7 < k_tile_len) dot += float(input_tile[m][k_local + 7]) * float(w_hi.w);

                    acc[m][out_i] += dot;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 3: Write output
    for (uint out_i = 0; out_i < outputs_per_thread; ++out_i) {
        uint n_idx = my_n_start + out_i;
        if (n_idx >= N) continue;

        for (uint m = 0; m < tokens_this_tile; ++m) {
            uint global_m = m_block + m;
            if (global_m < M) {
                output[global_m * N + n_idx] = half(acc[m][out_i]);
            }
        }
    }
}

// ============================================================================
// Fused RMSNorm + INT4 Linear Projection
// ============================================================================

kernel void norm_linear_int4(
    device const half* input           [[buffer(0)]],
    device const half* gamma           [[buffer(1)]],
    device const uint* weight_packed   [[buffer(2)]],
    device const half* scales          [[buffer(3)]],
    device const half* zeros           [[buffer(4)]],
    device half* output                [[buffer(5)]],
    constant uint& M                   [[buffer(6)]],
    constant uint& K                   [[buffer(7)]],
    constant uint& N                   [[buffer(8)]],
    constant uint& group_size          [[buffer(9)]],
    constant float& rms_eps            [[buffer(10)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint tid_in_tg                     [[thread_index_in_threadgroup]],
    uint lane_id                       [[thread_index_in_simdgroup]],
    uint sg_id                         [[simdgroup_index_in_threadgroup]]
) {
    const uint n_block = tgid.x * TILE_N;
    const uint m_block = tgid.y * TILE_M;

    threadgroup float rms_inv[TILE_M];
    threadgroup half input_tile[TILE_M][TILE_K];

    const uint tokens_this_tile = min(TILE_M, M - m_block);
    const uint num_simdgroups = FUSED_TG_SIZE / SIMDGROUP_SIZE;
    const uint tokens_per_sg = (tokens_this_tile + num_simdgroups - 1) / num_simdgroups;

    // Compute RMS denominators
    for (uint local_token = 0; local_token < tokens_per_sg; ++local_token) {
        uint token_idx = sg_id * tokens_per_sg + local_token;
        if (token_idx >= tokens_this_tile) break;

        uint global_token = m_block + token_idx;
        if (global_token >= M) {
            if (lane_id == 0) rms_inv[token_idx] = 0.0f;
            continue;
        }

        float sum_sq = 0.0f;
        const uint elems_per_lane = (K + SIMDGROUP_SIZE - 1) / SIMDGROUP_SIZE;

        for (uint i = 0; i < elems_per_lane; ++i) {
            uint k_idx = lane_id + i * SIMDGROUP_SIZE;
            if (k_idx < K) {
                float val = float(input[global_token * K + k_idx]);
                sum_sq += val * val;
            }
        }

        sum_sq = simd_sum_f32(sum_sq);

        if (lane_id == 0) {
            float variance = sum_sq / float(K);
            rms_inv[token_idx] = rsqrt(variance + rms_eps);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint outputs_per_thread = max(1u, TILE_N / FUSED_TG_SIZE);
    float acc[TILE_M][4];
    for (uint m = 0; m < TILE_M; ++m) {
        for (uint i = 0; i < outputs_per_thread; ++i) {
            acc[m][i] = 0.0f;
        }
    }

    const uint my_n_start = (tid_in_tg * outputs_per_thread) % TILE_N + n_block;

    // K-loop
    for (uint k_base = 0; k_base < K; k_base += TILE_K) {
        const uint k_tile_len = min(TILE_K, K - k_base);

        // Load and normalize input tile
        const uint elems_to_load = tokens_this_tile * k_tile_len;
        const uint loads_per_thread = (elems_to_load + FUSED_TG_SIZE - 1) / FUSED_TG_SIZE;

        for (uint i = 0; i < loads_per_thread; ++i) {
            uint idx = tid_in_tg + i * FUSED_TG_SIZE;
            if (idx < elems_to_load) {
                uint local_m = idx / k_tile_len;
                uint local_k = idx % k_tile_len;
                uint global_m = m_block + local_m;
                uint global_k = k_base + local_k;

                if (global_m < M && global_k < K) {
                    float val = float(input[global_m * K + global_k]);
                    float normed = val * rms_inv[local_m];
                    float gam = float(gamma[global_k]);
                    input_tile[local_m][local_k] = half(normed * gam);
                } else {
                    input_tile[local_m][local_k] = half(0);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute dot products
        for (uint out_i = 0; out_i < outputs_per_thread; ++out_i) {
            uint n_idx = my_n_start + out_i;
            if (n_idx >= N) continue;

            for (uint k_offset = 0; k_offset < k_tile_len; k_offset += FP4_PER_UINT) {
                uint global_k = k_base + k_offset;
                uint k_block_idx = global_k / FP4_PER_UINT;
                uint group_idx = global_k / group_size;

                uint32_t packed = weight_packed[k_block_idx * N + n_idx];
                half scale = scales[group_idx * N + n_idx];
                half zero = zeros[group_idx * N + n_idx];

                half4 w_lo, w_hi;
                dequant_u4x8(packed, scale, zero, w_lo, w_hi);

                for (uint m = 0; m < tokens_this_tile; ++m) {
                    float dot = 0.0f;
                    uint k_local = k_offset;

                    if (k_local < k_tile_len) dot += float(input_tile[m][k_local]) * float(w_lo.x);
                    if (k_local + 1 < k_tile_len) dot += float(input_tile[m][k_local + 1]) * float(w_lo.y);
                    if (k_local + 2 < k_tile_len) dot += float(input_tile[m][k_local + 2]) * float(w_lo.z);
                    if (k_local + 3 < k_tile_len) dot += float(input_tile[m][k_local + 3]) * float(w_lo.w);
                    if (k_local + 4 < k_tile_len) dot += float(input_tile[m][k_local + 4]) * float(w_hi.x);
                    if (k_local + 5 < k_tile_len) dot += float(input_tile[m][k_local + 5]) * float(w_hi.y);
                    if (k_local + 6 < k_tile_len) dot += float(input_tile[m][k_local + 6]) * float(w_hi.z);
                    if (k_local + 7 < k_tile_len) dot += float(input_tile[m][k_local + 7]) * float(w_hi.w);

                    acc[m][out_i] += dot;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    for (uint out_i = 0; out_i < outputs_per_thread; ++out_i) {
        uint n_idx = my_n_start + out_i;
        if (n_idx >= N) continue;

        for (uint m = 0; m < tokens_this_tile; ++m) {
            uint global_m = m_block + m;
            if (global_m < M) {
                output[global_m * N + n_idx] = half(acc[m][out_i]);
            }
        }
    }
}

// ============================================================================
// Fused RMSNorm + QKV Projection (3 outputs)
//
// For attention layers, we typically project to Q, K, V simultaneously.
// This kernel fuses RMSNorm with all three projections, computing the norm
// only once and reusing it for three output streams.
//
// output_q = W_q @ RMSNorm(input)
// output_k = W_k @ RMSNorm(input)
// output_v = W_v @ RMSNorm(input)
//
// Memory layout:
//   - Weights are concatenated: [W_q, W_k, W_v] along the N dimension
//   - Output shape: [M, 3 * head_dim * num_heads]
//   - For GQA: [M, (num_q_heads + 2 * num_kv_heads) * head_dim]
// ============================================================================

kernel void norm_qkv_fp4(
    device const half* input           [[buffer(0)]],
    device const half* gamma           [[buffer(1)]],
    device const uint* weight_packed   [[buffer(2)]],  // [K/8, N_total]
    device const half* scales          [[buffer(3)]],
    device half* output_q              [[buffer(4)]],  // [M, N_q]
    device half* output_k              [[buffer(5)]],  // [M, N_kv]
    device half* output_v              [[buffer(6)]],  // [M, N_kv]
    constant uint& M                   [[buffer(7)]],
    constant uint& K                   [[buffer(8)]],
    constant uint& N_q                 [[buffer(9)]],   // Q output size
    constant uint& N_kv                [[buffer(10)]],  // K/V output size
    constant uint& group_size          [[buffer(11)]],
    constant float& rms_eps            [[buffer(12)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint tid_in_tg                     [[thread_index_in_threadgroup]],
    uint lane_id                       [[thread_index_in_simdgroup]],
    uint sg_id                         [[simdgroup_index_in_threadgroup]]
) {
    // Total output size = N_q + N_kv + N_kv
    const uint N_total = N_q + N_kv + N_kv;
    const uint n_block = tgid.x * TILE_N;
    const uint m_block = tgid.y * TILE_M;

    // Determine which output (Q/K/V) this threadgroup contributes to
    uint output_type;  // 0=Q, 1=K, 2=V
    uint local_n_offset;
    uint local_n_size;

    if (n_block < N_q) {
        output_type = 0;
        local_n_offset = 0;
        local_n_size = N_q;
    } else if (n_block < N_q + N_kv) {
        output_type = 1;
        local_n_offset = N_q;
        local_n_size = N_kv;
    } else {
        output_type = 2;
        local_n_offset = N_q + N_kv;
        local_n_size = N_kv;
    }

    threadgroup float rms_inv[TILE_M];
    threadgroup half input_tile[TILE_M][TILE_K];

    const uint tokens_this_tile = min(TILE_M, M - m_block);
    const uint num_simdgroups = FUSED_TG_SIZE / SIMDGROUP_SIZE;
    const uint tokens_per_sg = (tokens_this_tile + num_simdgroups - 1) / num_simdgroups;

    // Compute RMS denominators (same as norm_linear_fp4)
    for (uint local_token = 0; local_token < tokens_per_sg; ++local_token) {
        uint token_idx = sg_id * tokens_per_sg + local_token;
        if (token_idx >= tokens_this_tile) break;

        uint global_token = m_block + token_idx;
        if (global_token >= M) {
            if (lane_id == 0) rms_inv[token_idx] = 0.0f;
            continue;
        }

        float sum_sq = 0.0f;
        const uint elems_per_lane = (K + SIMDGROUP_SIZE - 1) / SIMDGROUP_SIZE;

        for (uint i = 0; i < elems_per_lane; ++i) {
            uint k_idx = lane_id + i * SIMDGROUP_SIZE;
            if (k_idx < K) {
                float val = float(input[global_token * K + k_idx]);
                sum_sq += val * val;
            }
        }

        sum_sq = simd_sum_f32(sum_sq);

        if (lane_id == 0) {
            float variance = sum_sq / float(K);
            rms_inv[token_idx] = rsqrt(variance + rms_eps);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // GEMM computation (same structure as norm_linear_fp4)
    const uint outputs_per_thread = max(1u, TILE_N / FUSED_TG_SIZE);
    float acc[TILE_M][4];
    for (uint m = 0; m < TILE_M; ++m) {
        for (uint i = 0; i < outputs_per_thread; ++i) {
            acc[m][i] = 0.0f;
        }
    }

    const uint my_n_start = (tid_in_tg * outputs_per_thread) % TILE_N + n_block;

    for (uint k_base = 0; k_base < K; k_base += TILE_K) {
        const uint k_tile_len = min(TILE_K, K - k_base);

        const uint elems_to_load = tokens_this_tile * k_tile_len;
        const uint loads_per_thread = (elems_to_load + FUSED_TG_SIZE - 1) / FUSED_TG_SIZE;

        for (uint i = 0; i < loads_per_thread; ++i) {
            uint idx = tid_in_tg + i * FUSED_TG_SIZE;
            if (idx < elems_to_load) {
                uint local_m = idx / k_tile_len;
                uint local_k = idx % k_tile_len;
                uint global_m = m_block + local_m;
                uint global_k = k_base + local_k;

                if (global_m < M && global_k < K) {
                    float val = float(input[global_m * K + global_k]);
                    float normed = val * rms_inv[local_m];
                    float gam = float(gamma[global_k]);
                    input_tile[local_m][local_k] = half(normed * gam);
                } else {
                    input_tile[local_m][local_k] = half(0);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint out_i = 0; out_i < outputs_per_thread; ++out_i) {
            uint n_idx = my_n_start + out_i;
            if (n_idx >= N_total) continue;

            for (uint k_offset = 0; k_offset < k_tile_len; k_offset += FP4_PER_UINT) {
                uint global_k = k_base + k_offset;
                uint k_block_idx = global_k / FP4_PER_UINT;
                uint group_idx = global_k / group_size;

                uint32_t packed = weight_packed[k_block_idx * N_total + n_idx];
                half scale = scales[group_idx * N_total + n_idx];

                half4 w_lo, w_hi;
                dequant_fp4_x8_scaled(packed, scale, w_lo, w_hi);

                for (uint m = 0; m < tokens_this_tile; ++m) {
                    float dot = 0.0f;
                    uint k_local = k_offset;

                    if (k_local < k_tile_len) dot += float(input_tile[m][k_local]) * float(w_lo.x);
                    if (k_local + 1 < k_tile_len) dot += float(input_tile[m][k_local + 1]) * float(w_lo.y);
                    if (k_local + 2 < k_tile_len) dot += float(input_tile[m][k_local + 2]) * float(w_lo.z);
                    if (k_local + 3 < k_tile_len) dot += float(input_tile[m][k_local + 3]) * float(w_lo.w);
                    if (k_local + 4 < k_tile_len) dot += float(input_tile[m][k_local + 4]) * float(w_hi.x);
                    if (k_local + 5 < k_tile_len) dot += float(input_tile[m][k_local + 5]) * float(w_hi.y);
                    if (k_local + 6 < k_tile_len) dot += float(input_tile[m][k_local + 6]) * float(w_hi.z);
                    if (k_local + 7 < k_tile_len) dot += float(input_tile[m][k_local + 7]) * float(w_hi.w);

                    acc[m][out_i] += dot;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write to appropriate output buffer based on n_idx
    for (uint out_i = 0; out_i < outputs_per_thread; ++out_i) {
        uint n_idx = my_n_start + out_i;
        if (n_idx >= N_total) continue;

        // Determine output destination
        device half* out_ptr;
        uint out_col;

        if (n_idx < N_q) {
            out_ptr = output_q;
            out_col = n_idx;
        } else if (n_idx < N_q + N_kv) {
            out_ptr = output_k;
            out_col = n_idx - N_q;
        } else {
            out_ptr = output_v;
            out_col = n_idx - N_q - N_kv;
        }

        for (uint m = 0; m < tokens_this_tile; ++m) {
            uint global_m = m_block + m;
            if (global_m < M) {
                uint out_size = (n_idx < N_q) ? N_q : N_kv;
                out_ptr[global_m * out_size + out_col] = half(acc[m][out_i]);
            }
        }
    }
}

// ============================================================================
// Decode-optimized fused kernel (batch_size=1, seq_len=1)
//
// For autoregressive decoding, we often process a single token at a time.
// This kernel is optimized for M=1, using simdgroup-level parallelism across
// the output dimension rather than across tokens.
// ============================================================================

kernel void norm_linear_fp4_decode(
    device const half* input           [[buffer(0)]],  // [1, K]
    device const half* gamma           [[buffer(1)]],  // [K]
    device const uint* weight_packed   [[buffer(2)]],  // [K/8, N]
    device const half* scales          [[buffer(3)]],  // [K/group_size, N]
    device half* output                [[buffer(4)]],  // [1, N]
    constant uint& K                   [[buffer(5)]],
    constant uint& N                   [[buffer(6)]],
    constant uint& group_size          [[buffer(7)]],
    constant float& rms_eps            [[buffer(8)]],
    uint tgid                          [[threadgroup_position_in_grid]],
    uint tid_in_tg                     [[thread_index_in_threadgroup]],
    uint lane_id                       [[thread_index_in_simdgroup]],
    uint sg_id                         [[simdgroup_index_in_threadgroup]]
) {
    // Step 1: Compute RMS norm (cooperative across entire threadgroup)
    threadgroup float rms_inv_shared;

    float sum_sq = 0.0f;
    const uint elems_per_thread = (K + FUSED_TG_SIZE - 1) / FUSED_TG_SIZE;

    for (uint i = 0; i < elems_per_thread; ++i) {
        uint k_idx = tid_in_tg + i * FUSED_TG_SIZE;
        if (k_idx < K) {
            float val = float(input[k_idx]);
            sum_sq += val * val;
        }
    }

    // Two-level reduction: within simdgroup, then across simdgroups
    sum_sq = simd_sum_f32(sum_sq);

    // Store simdgroup sum in shared memory
    threadgroup float sg_sums[4];  // Max 4 simdgroups per TG
    if (lane_id == 0) {
        sg_sums[sg_id] = sum_sq;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by simdgroup 0
    if (sg_id == 0 && lane_id < 4) {
        float total = sg_sums[lane_id];
        total = simd_sum_f32(total);
        if (lane_id == 0) {
            float variance = total / float(K);
            rms_inv_shared = rsqrt(variance + rms_eps);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Each thread computes one or more output elements
    // tgid indexes which block of N outputs this threadgroup handles
    const uint n_per_tg = FUSED_TG_SIZE;
    const uint n_base = tgid * n_per_tg;
    const uint my_n = n_base + tid_in_tg;

    if (my_n >= N) return;

    float acc = 0.0f;

    // Stream through K, loading normalized input and weights
    for (uint k_base = 0; k_base < K; k_base += FP4_PER_UINT) {
        uint k_block_idx = k_base / FP4_PER_UINT;
        uint group_idx = k_base / group_size;

        uint32_t packed = weight_packed[k_block_idx * N + my_n];
        half scale = scales[group_idx * N + my_n];

        half4 w_lo, w_hi;
        dequant_fp4_x8_scaled(packed, scale, w_lo, w_hi);

        // Load and normalize 8 input elements
        float in_normed[8];
        for (uint i = 0; i < 8 && (k_base + i) < K; ++i) {
            float val = float(input[k_base + i]);
            float gam = float(gamma[k_base + i]);
            in_normed[i] = val * rms_inv_shared * gam;
        }

        // Dot product
        acc += in_normed[0] * float(w_lo.x);
        if (k_base + 1 < K) acc += in_normed[1] * float(w_lo.y);
        if (k_base + 2 < K) acc += in_normed[2] * float(w_lo.z);
        if (k_base + 3 < K) acc += in_normed[3] * float(w_lo.w);
        if (k_base + 4 < K) acc += in_normed[4] * float(w_hi.x);
        if (k_base + 5 < K) acc += in_normed[5] * float(w_hi.y);
        if (k_base + 6 < K) acc += in_normed[6] * float(w_hi.z);
        if (k_base + 7 < K) acc += in_normed[7] * float(w_hi.w);
    }

    output[my_n] = half(acc);
}
