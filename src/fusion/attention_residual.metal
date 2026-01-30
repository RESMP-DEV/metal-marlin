// attention_residual.metal - Fused Attention Output Projection + Residual Add
//
// Fuses the O-projection after attention with the residual connection:
//   output = residual + O_proj(attention_output)
//
// Standard flow (memory-bound):
//   1. Attention kernel writes attention_output to DRAM
//   2. O-projection kernel reads attention_output, writes projected
//   3. Residual add kernel reads projected + residual, writes final
//   Memory traffic: 3x read + 3x write for the full sequence
//
// Fused flow:
//   1. Attention kernel writes attention_output to DRAM (unavoidable)
//   2. Fused kernel reads attention_output + residual, writes final output
//   Memory traffic: 2x read + 2x write (33% reduction in traffic)
//
// This kernel is particularly effective for:
//   - The attention output path where O_proj is followed by residual
//   - MLP output path where down_proj is followed by residual
//
// Design:
//   - O_proj is a quantized GEMM: [batch*seq, num_heads*head_dim] -> [batch*seq, hidden]
//   - Residual is the input to the attention block (pre-norm architecture)
//   - Output has same shape as residual
//
// Memory layout:
//   attention_output: [M, K] where K = num_heads * head_dim
//   residual:         [M, N] where N = hidden_size (typically K == N)
//   weight_packed:    [K/8, N] packed FP4/INT4
//   scales:           [K/group_size, N]
//   output:           [M, N]
//
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants
// ============================================================================

constant constexpr uint FUSED_TG_SIZE = 128;
constant constexpr uint SIMDGROUP_SIZE = 32;
constant constexpr uint TILE_M = 64;
constant constexpr uint TILE_N = 64;
constant constexpr uint TILE_K = 32;
constant constexpr uint FP4_PER_UINT = 8;

constant constexpr uint32_t MAGIC_BIAS_U32 = 0x64006400u;
constant constexpr uint32_t LO_NIBBLE_MASK = 0x000F000Fu;

// ============================================================================
// FP4/INT4 dequantization (copied from norm_linear.metal for self-containment)
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
// Fused Linear + Residual Add (FP4)
//
// Computes: output = residual + Linear(input)
//
// The GEMM output is added to the residual directly without an intermediate
// store, saving one memory round-trip per element.
// ============================================================================

kernel void linear_residual_fp4(
    device const half* input           [[buffer(0)]],  // [M, K] - attention output
    device const half* residual        [[buffer(1)]],  // [M, N] - pre-attention input
    device const uint* weight_packed   [[buffer(2)]],  // [K/8, N]
    device const half* scales          [[buffer(3)]],  // [K/group_size, N]
    device half* output                [[buffer(4)]],  // [M, N]
    constant uint& M                   [[buffer(5)]],
    constant uint& K                   [[buffer(6)]],
    constant uint& N                   [[buffer(7)]],
    constant uint& group_size          [[buffer(8)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint tid_in_tg                     [[thread_index_in_threadgroup]]
) {
    const uint n_block = tgid.x * TILE_N;
    const uint m_block = tgid.y * TILE_M;

    threadgroup half input_tile[TILE_M][TILE_K];

    const uint tokens_this_tile = min(TILE_M, M - m_block);
    const uint outputs_per_thread = max(1u, TILE_N / FUSED_TG_SIZE);

    float acc[TILE_M][4];
    for (uint m = 0; m < TILE_M; ++m) {
        for (uint i = 0; i < outputs_per_thread; ++i) {
            acc[m][i] = 0.0f;
        }
    }

    const uint my_n_start = (tid_in_tg * outputs_per_thread) % TILE_N + n_block;

    // K-loop: stream through input features
    for (uint k_base = 0; k_base < K; k_base += TILE_K) {
        const uint k_tile_len = min(TILE_K, K - k_base);

        // Cooperative load of input tile
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
                    input_tile[local_m][local_k] = input[global_m * K + global_k];
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

    // Write output with fused residual add
    for (uint out_i = 0; out_i < outputs_per_thread; ++out_i) {
        uint n_idx = my_n_start + out_i;
        if (n_idx >= N) continue;

        for (uint m = 0; m < tokens_this_tile; ++m) {
            uint global_m = m_block + m;
            if (global_m < M) {
                // Fused: GEMM output + residual
                float res = float(residual[global_m * N + n_idx]);
                output[global_m * N + n_idx] = half(acc[m][out_i] + res);
            }
        }
    }
}

// ============================================================================
// Fused Linear + Residual Add (INT4)
// ============================================================================

kernel void linear_residual_int4(
    device const half* input           [[buffer(0)]],
    device const half* residual        [[buffer(1)]],
    device const uint* weight_packed   [[buffer(2)]],
    device const half* scales          [[buffer(3)]],
    device const half* zeros           [[buffer(4)]],
    device half* output                [[buffer(5)]],
    constant uint& M                   [[buffer(6)]],
    constant uint& K                   [[buffer(7)]],
    constant uint& N                   [[buffer(8)]],
    constant uint& group_size          [[buffer(9)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint tid_in_tg                     [[thread_index_in_threadgroup]]
) {
    const uint n_block = tgid.x * TILE_N;
    const uint m_block = tgid.y * TILE_M;

    threadgroup half input_tile[TILE_M][TILE_K];

    const uint tokens_this_tile = min(TILE_M, M - m_block);
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
                    input_tile[local_m][local_k] = input[global_m * K + global_k];
                } else {
                    input_tile[local_m][local_k] = half(0);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

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

    // Write with fused residual
    for (uint out_i = 0; out_i < outputs_per_thread; ++out_i) {
        uint n_idx = my_n_start + out_i;
        if (n_idx >= N) continue;

        for (uint m = 0; m < tokens_this_tile; ++m) {
            uint global_m = m_block + m;
            if (global_m < M) {
                float res = float(residual[global_m * N + n_idx]);
                output[global_m * N + n_idx] = half(acc[m][out_i] + res);
            }
        }
    }
}

// ============================================================================
// Decode-optimized Linear + Residual (batch=1, seq=1)
//
// For autoregressive decoding with a single token, we parallelize across
// the output dimension. Each thread computes one output element and adds
// the corresponding residual.
// ============================================================================

kernel void linear_residual_fp4_decode(
    device const half* input           [[buffer(0)]],  // [1, K]
    device const half* residual        [[buffer(1)]],  // [1, N]
    device const uint* weight_packed   [[buffer(2)]],  // [K/8, N]
    device const half* scales          [[buffer(3)]],  // [K/group_size, N]
    device half* output                [[buffer(4)]],  // [1, N]
    constant uint& K                   [[buffer(5)]],
    constant uint& N                   [[buffer(6)]],
    constant uint& group_size          [[buffer(7)]],
    uint tgid                          [[threadgroup_position_in_grid]],
    uint tid_in_tg                     [[thread_index_in_threadgroup]]
) {
    const uint n_per_tg = FUSED_TG_SIZE;
    const uint my_n = tgid * n_per_tg + tid_in_tg;

    if (my_n >= N) return;

    float acc = 0.0f;

    for (uint k_base = 0; k_base < K; k_base += FP4_PER_UINT) {
        uint k_block_idx = k_base / FP4_PER_UINT;
        uint group_idx = k_base / group_size;

        uint32_t packed = weight_packed[k_block_idx * N + my_n];
        half scale = scales[group_idx * N + my_n];

        half4 w_lo, w_hi;
        dequant_fp4_x8_scaled(packed, scale, w_lo, w_hi);

        // Load 8 input elements
        acc += float(input[k_base]) * float(w_lo.x);
        if (k_base + 1 < K) acc += float(input[k_base + 1]) * float(w_lo.y);
        if (k_base + 2 < K) acc += float(input[k_base + 2]) * float(w_lo.z);
        if (k_base + 3 < K) acc += float(input[k_base + 3]) * float(w_lo.w);
        if (k_base + 4 < K) acc += float(input[k_base + 4]) * float(w_hi.x);
        if (k_base + 5 < K) acc += float(input[k_base + 5]) * float(w_hi.y);
        if (k_base + 6 < K) acc += float(input[k_base + 6]) * float(w_hi.z);
        if (k_base + 7 < K) acc += float(input[k_base + 7]) * float(w_hi.w);
    }

    // Fused residual add
    output[my_n] = half(acc + float(residual[my_n]));
}

// ============================================================================
// Fused RMSNorm + Linear + Residual
//
// The ultimate fusion: combines normalization, projection, and residual add
// into a single kernel. This is the complete attention output path:
//
//   output = residual + O_proj(attention_output)
//
// where attention_output has already been normalized by the input_layernorm.
//
// For the post-attention residual, we don't need another norm, but for the
// MLP path we do. This kernel handles the case where normalization IS needed
// (e.g., a hypothetical architecture or a testing scenario).
// ============================================================================

kernel void norm_linear_residual_fp4(
    device const half* input           [[buffer(0)]],   // [M, K] to be normalized
    device const half* gamma           [[buffer(1)]],   // [K] RMSNorm weight
    device const half* residual        [[buffer(2)]],   // [M, N] residual
    device const uint* weight_packed   [[buffer(3)]],   // [K/8, N]
    device const half* scales          [[buffer(4)]],   // [K/group_size, N]
    device half* output                [[buffer(5)]],   // [M, N]
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

        // Simdgroup reduction
        sum_sq += simd_shuffle_xor(sum_sq, 16);
        sum_sq += simd_shuffle_xor(sum_sq, 8);
        sum_sq += simd_shuffle_xor(sum_sq, 4);
        sum_sq += simd_shuffle_xor(sum_sq, 2);
        sum_sq += simd_shuffle_xor(sum_sq, 1);

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

    // K-loop with fused normalization
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
            if (n_idx >= N) continue;

            for (uint k_offset = 0; k_offset < k_tile_len; k_offset += FP4_PER_UINT) {
                uint global_k = k_base + k_offset;
                uint k_block_idx = global_k / FP4_PER_UINT;
                uint group_idx = global_k / group_size;

                uint32_t packed = weight_packed[k_block_idx * N + n_idx];
                half scale = scales[group_idx * N + n_idx];

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

    // Write with fused residual add
    for (uint out_i = 0; out_i < outputs_per_thread; ++out_i) {
        uint n_idx = my_n_start + out_i;
        if (n_idx >= N) continue;

        for (uint m = 0; m < tokens_this_tile; ++m) {
            uint global_m = m_block + m;
            if (global_m < M) {
                float res = float(residual[global_m * N + n_idx]);
                output[global_m * N + n_idx] = half(acc[m][out_i] + res);
            }
        }
    }
}

// ============================================================================
// Fused Attention Output with Optional Scaling
//
// Some architectures scale the attention output before adding to residual.
// This kernel supports an optional output_scale parameter.
//
// output = residual + output_scale * O_proj(attention_output)
// ============================================================================

kernel void linear_residual_scaled_fp4(
    device const half* input           [[buffer(0)]],
    device const half* residual        [[buffer(1)]],
    device const uint* weight_packed   [[buffer(2)]],
    device const half* scales          [[buffer(3)]],
    device half* output                [[buffer(4)]],
    constant uint& M                   [[buffer(5)]],
    constant uint& K                   [[buffer(6)]],
    constant uint& N                   [[buffer(7)]],
    constant uint& group_size          [[buffer(8)]],
    constant float& output_scale       [[buffer(9)]],  // Scale factor for GEMM output
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint tid_in_tg                     [[thread_index_in_threadgroup]]
) {
    const uint n_block = tgid.x * TILE_N;
    const uint m_block = tgid.y * TILE_M;

    threadgroup half input_tile[TILE_M][TILE_K];

    const uint tokens_this_tile = min(TILE_M, M - m_block);
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
                    input_tile[local_m][local_k] = input[global_m * K + global_k];
                } else {
                    input_tile[local_m][local_k] = half(0);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint out_i = 0; out_i < outputs_per_thread; ++out_i) {
            uint n_idx = my_n_start + out_i;
            if (n_idx >= N) continue;

            for (uint k_offset = 0; k_offset < k_tile_len; k_offset += FP4_PER_UINT) {
                uint global_k = k_base + k_offset;
                uint k_block_idx = global_k / FP4_PER_UINT;
                uint group_idx = global_k / group_size;

                uint32_t packed = weight_packed[k_block_idx * N + n_idx];
                half scale = scales[group_idx * N + n_idx];

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

    // Write with scaled GEMM output + residual
    for (uint out_i = 0; out_i < outputs_per_thread; ++out_i) {
        uint n_idx = my_n_start + out_i;
        if (n_idx >= N) continue;

        for (uint m = 0; m < tokens_this_tile; ++m) {
            uint global_m = m_block + m;
            if (global_m < M) {
                float res = float(residual[global_m * N + n_idx]);
                output[global_m * N + n_idx] = half(acc[m][out_i] * output_scale + res);
            }
        }
    }
}
