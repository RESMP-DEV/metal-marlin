// diff_attention.metal - Fused Differential Attention for Apple Silicon
//
// Implements Microsoft's Differential Transformer attention mechanism:
//   Output = softmax(Q1 @ K1^T / sqrt(d)) @ V - lambda * softmax(Q2 @ K2^T / sqrt(d)) @ V
//
// This kernel fuses both attention computations into a single pass through K/V,
// avoiding redundant memory traffic for the shared V tensor.
//
// Key optimizations:
//   - Single K/V tile load serves both attention paths
//   - Online softmax for both score streams simultaneously
//   - Lambda applied in registers before final accumulation
//   - Double-buffering for memory latency hiding
//
// Kernel variants:
//   1. diff_attention              - Standard differential attention
//   2. diff_attention_causal       - With causal masking
//   3. diff_attention_gqa          - Grouped-query attention
//   4. diff_attention_kv_fp4       - FP4-quantized KV cache
//
// Memory layout (all row-major):
//   Q1: [batch, num_heads, seq_q, head_dim]
//   Q2: [batch, num_heads, seq_q, head_dim]
//   K1: [batch, num_heads_k, seq_k, head_dim]
//   K2: [batch, num_heads_k, seq_k, head_dim]
//   V:  [batch, num_heads_k, seq_k, head_dim]
//   O:  [batch, num_heads, seq_q, head_dim]
//   lambda: [num_heads] or [1] (per-head or shared)

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Tile dimensions - same as flash_attention.metal for consistency
// ---------------------------------------------------------------------------

constant constexpr uint TILE_KV = 64;
constant constexpr uint HEAD_DIM_MAX = 128;
constant constexpr uint THREADS_PER_ROW = 32;
constant constexpr uint ROWS_PER_TG = 4;
constant constexpr uint THREADS_PER_TG = THREADS_PER_ROW * ROWS_PER_TG;
constant constexpr uint FP4_PER_UINT = 8;

// ---------------------------------------------------------------------------
// Utility functions using hardware-accelerated simd reductions
//
// Metal's built-in simd_sum/simd_max are significantly faster than manual
// simd_shuffle_xor chains: single instruction vs 5 dependent instructions.
// ---------------------------------------------------------------------------

inline float simd_max_f32(float val, uint lane_id [[thread_index_in_simdgroup]]) {
    (void)lane_id;  // Unused with hardware intrinsic
    return simd_max(val);
}

inline float simd_sum_f32(float val) {
    return simd_sum(val);
}

// ---------------------------------------------------------------------------
// FP4 dequantization (E2M1 format)
// ---------------------------------------------------------------------------

inline half dequant_fp4_scalar(uint nibble, half scale) {
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;

    half magnitude;
    if (exp_bits == 0) {
        magnitude = half(man_bit) * half(0.5h);
    } else {
        half power = half(1u << (exp_bits - 1));
        half mantissa = half(1.0h) + half(man_bit) * half(0.5h);
        magnitude = power * mantissa;
    }

    half result = sign_bit ? -magnitude : magnitude;
    return result * scale;
}

// ---------------------------------------------------------------------------
// Differential Attention kernel - standard (non-causal)
//
// Computes: O = softmax(Q1 @ K1^T / sqrt(d)) @ V - lambda * softmax(Q2 @ K2^T / sqrt(d)) @ V
//
// Dispatch: [num_heads_q, ceil(seq_q / ROWS_PER_TG), batch] threadgroups
//           THREADS_PER_TG threads per threadgroup
// ---------------------------------------------------------------------------

kernel void diff_attention(
    device const half* Q1           [[buffer(0)]],
    device const half* Q2           [[buffer(1)]],
    device const half* K1           [[buffer(2)]],
    device const half* K2           [[buffer(3)]],
    device const half* V            [[buffer(4)]],
    device half* O                  [[buffer(5)]],
    device const half* lambda_vals  [[buffer(6)]],   // [num_heads] or [1]
    constant uint& batch            [[buffer(7)]],
    constant uint& num_heads_q      [[buffer(8)]],
    constant uint& num_heads_k      [[buffer(9)]],
    constant uint& seq_q            [[buffer(10)]],
    constant uint& seq_k            [[buffer(11)]],
    constant uint& head_dim         [[buffer(12)]],
    constant float& scale           [[buffer(13)]],
    constant bool& lambda_per_head  [[buffer(14)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint tid_in_tg                  [[thread_index_in_threadgroup]],
    uint lane_id                    [[thread_index_in_simdgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    // Prefetch hint: consider simdgroup_async_copy for better memory throughput
    const uint head_q = tgid.x;
    const uint q_row_base = tgid.y * ROWS_PER_TG;
    const uint b = tgid.z;

    // For GQA: map Q head to K/V head
    const uint head_k = head_q * num_heads_k / num_heads_q;

    // This simdgroup handles one query row
    const uint q_row = q_row_base + sg_id;
    if (q_row >= seq_q) return;

    // Get lambda for this head
    const float lambda_val = float(lambda_per_head ? lambda_vals[head_q] : lambda_vals[0]);

    // Strides for [batch, heads, seq, head_dim] layout
    const uint q_stride_b = num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;

    const uint k_stride_b = num_heads_k * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;

    // Base pointers
    const uint q1_offset = b * q_stride_b + head_q * q_stride_h + q_row * q_stride_s;
    const uint q2_offset = q1_offset;  // Same layout for Q2
    const uint kv_offset = b * k_stride_b + head_k * k_stride_h;

    // Load Q1 and Q2 rows into registers
    const uint elems_per_lane = head_dim / THREADS_PER_ROW;
    float q1_reg[HEAD_DIM_MAX / THREADS_PER_ROW];
    float q2_reg[HEAD_DIM_MAX / THREADS_PER_ROW];
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        q1_reg[i] = (d < head_dim) ? float(Q1[q1_offset + d]) : 0.0f;
        q2_reg[i] = (d < head_dim) ? float(Q2[q2_offset + d]) : 0.0f;
    }

    // Shared memory for K1, K2, V tiles (double-buffered)
    threadgroup half K1_tile[2][TILE_KV][HEAD_DIM_MAX];
    threadgroup half K2_tile[2][TILE_KV][HEAD_DIM_MAX];
    threadgroup half V_tile[2][TILE_KV][HEAD_DIM_MAX];

    // Online softmax state for both attention paths
    float m1_prev = -INFINITY;  // Running max for path 1
    float l1_prev = 0.0f;       // Running sum for path 1
    float m2_prev = -INFINITY;  // Running max for path 2
    float l2_prev = 0.0f;       // Running sum for path 2

    // Output accumulators (unnormalized)
    float o1_acc[HEAD_DIM_MAX / THREADS_PER_ROW];
    float o2_acc[HEAD_DIM_MAX / THREADS_PER_ROW];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o1_acc[i] = 0.0f;
        o2_acc[i] = 0.0f;
    }

    const uint num_kv_tiles = (seq_k + TILE_KV - 1) / TILE_KV;

    // Preload first tile into buffer 0
    {
        const uint tile_start = 0;
        const uint elems_to_load = TILE_KV * head_dim;
        const uint loads_per_thread = (elems_to_load + THREADS_PER_TG - 1) / THREADS_PER_TG;
        for (uint i = 0; i < loads_per_thread; ++i) {
            uint idx = tid_in_tg + i * THREADS_PER_TG;
            if (idx < elems_to_load) {
                uint kv_row = idx / head_dim;
                uint kv_col = idx % head_dim;
                uint src_row = tile_start + kv_row;
                if (src_row < seq_k) {
                    K1_tile[0][kv_row][kv_col] = K1[kv_offset + src_row * k_stride_s + kv_col];
                    K2_tile[0][kv_row][kv_col] = K2[kv_offset + src_row * k_stride_s + kv_col];
                    V_tile[0][kv_row][kv_col] = V[kv_offset + src_row * k_stride_s + kv_col];
                } else {
                    K1_tile[0][kv_row][kv_col] = half(0);
                    K2_tile[0][kv_row][kv_col] = half(0);
                    V_tile[0][kv_row][kv_col] = half(0);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main loop with double-buffering
    uint buf_compute = 0;

    for (uint tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
        uint buf_load = 1 - buf_compute;

        // Async load next tile (if exists)
        uint next_tile_start = (tile_idx + 1) * TILE_KV;
        if (tile_idx + 1 < num_kv_tiles) {
            const uint elems_to_load = TILE_KV * head_dim;
            const uint loads_per_thread = (elems_to_load + THREADS_PER_TG - 1) / THREADS_PER_TG;
            for (uint i = 0; i < loads_per_thread; ++i) {
                uint idx = tid_in_tg + i * THREADS_PER_TG;
                if (idx < elems_to_load) {
                    uint kv_row = idx / head_dim;
                    uint kv_col = idx % head_dim;
                    uint src_row = next_tile_start + kv_row;
                    if (src_row < seq_k) {
                        K1_tile[buf_load][kv_row][kv_col] = K1[kv_offset + src_row * k_stride_s + kv_col];
                        K2_tile[buf_load][kv_row][kv_col] = K2[kv_offset + src_row * k_stride_s + kv_col];
                        V_tile[buf_load][kv_row][kv_col] = V[kv_offset + src_row * k_stride_s + kv_col];
                    } else {
                        K1_tile[buf_load][kv_row][kv_col] = half(0);
                        K2_tile[buf_load][kv_row][kv_col] = half(0);
                        V_tile[buf_load][kv_row][kv_col] = half(0);
                    }
                }
            }
        }

        // Compute attention scores for this tile (both paths)
        uint tile_start = tile_idx * TILE_KV;
        uint tile_end = min(tile_start + TILE_KV, seq_k);
        uint tile_len = tile_end - tile_start;

        float scores1[TILE_KV];
        float scores2[TILE_KV];
        for (uint ki = 0; ki < tile_len; ++ki) {
            // Path 1: Q1 @ K1
            float dot1 = 0.0f;
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                dot1 += q1_reg[i] * float(K1_tile[buf_compute][ki][d]);
            }
            dot1 = simd_sum_f32(dot1);
            scores1[ki] = dot1 * scale;

            // Path 2: Q2 @ K2
            float dot2 = 0.0f;
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                dot2 += q2_reg[i] * float(K2_tile[buf_compute][ki][d]);
            }
            dot2 = simd_sum_f32(dot2);
            scores2[ki] = dot2 * scale;
        }
        for (uint ki = tile_len; ki < TILE_KV; ++ki) {
            scores1[ki] = -INFINITY;
            scores2[ki] = -INFINITY;
        }

        // Online softmax update for both paths
        float m1_tile = -INFINITY;
        float m2_tile = -INFINITY;
        for (uint ki = 0; ki < tile_len; ++ki) {
            m1_tile = max(m1_tile, scores1[ki]);
            m2_tile = max(m2_tile, scores2[ki]);
        }

        float m1_new = max(m1_prev, m1_tile);
        float m2_new = max(m2_prev, m2_tile);
        float correction1 = exp(m1_prev - m1_new);
        float correction2 = exp(m2_prev - m2_new);

        float l1_new = l1_prev * correction1;
        float l2_new = l2_prev * correction2;
        for (uint ki = 0; ki < tile_len; ++ki) {
            l1_new += exp(scores1[ki] - m1_new);
            l2_new += exp(scores2[ki] - m2_new);
        }

        // Rescale previous accumulators
        for (uint i = 0; i < elems_per_lane; ++i) {
            o1_acc[i] *= correction1;
            o2_acc[i] *= correction2;
        }

        // Accumulate weighted V rows for both paths
        for (uint ki = 0; ki < tile_len; ++ki) {
            float p1 = exp(scores1[ki] - m1_new);
            float p2 = exp(scores2[ki] - m2_new);
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                float v_val = float(V_tile[buf_compute][ki][d]);
                o1_acc[i] += p1 * v_val;
                o2_acc[i] += p2 * v_val;
            }
        }

        m1_prev = m1_new;
        m2_prev = m2_new;
        l1_prev = l1_new;
        l2_prev = l2_new;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // Normalize and compute differential output
    const uint o_offset = b * q_stride_b + head_q * q_stride_h + q_row * q_stride_s;
    float inv_l1 = (l1_prev > 0.0f) ? (1.0f / l1_prev) : 0.0f;
    float inv_l2 = (l2_prev > 0.0f) ? (1.0f / l2_prev) : 0.0f;

    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        if (d < head_dim) {
            // Differential attention: output = attn1 - lambda * attn2
            float out1 = o1_acc[i] * inv_l1;
            float out2 = o2_acc[i] * inv_l2;
            O[o_offset + d] = half(out1 - lambda_val * out2);
        }
    }
}

// ---------------------------------------------------------------------------
// Differential Attention kernel - causal masking
// ---------------------------------------------------------------------------

kernel void diff_attention_causal(
    device const half* Q1           [[buffer(0)]],
    device const half* Q2           [[buffer(1)]],
    device const half* K1           [[buffer(2)]],
    device const half* K2           [[buffer(3)]],
    device const half* V            [[buffer(4)]],
    device half* O                  [[buffer(5)]],
    device const half* lambda_vals  [[buffer(6)]],
    constant uint& batch            [[buffer(7)]],
    constant uint& num_heads_q      [[buffer(8)]],
    constant uint& num_heads_k      [[buffer(9)]],
    constant uint& seq_q            [[buffer(10)]],
    constant uint& seq_k            [[buffer(11)]],
    constant uint& head_dim         [[buffer(12)]],
    constant float& scale           [[buffer(13)]],
    constant bool& lambda_per_head  [[buffer(14)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint tid_in_tg                  [[thread_index_in_threadgroup]],
    uint lane_id                    [[thread_index_in_simdgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    const uint head_q = tgid.x;
    const uint q_row_base = tgid.y * ROWS_PER_TG;
    const uint b = tgid.z;
    const uint head_k = head_q * num_heads_k / num_heads_q;
    const uint q_row = q_row_base + sg_id;
    if (q_row >= seq_q) return;

    const float lambda_val = float(lambda_per_head ? lambda_vals[head_q] : lambda_vals[0]);

    const uint q_stride_b = num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;
    const uint k_stride_b = num_heads_k * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;

    const uint q1_offset = b * q_stride_b + head_q * q_stride_h + q_row * q_stride_s;
    const uint q2_offset = q1_offset;
    const uint kv_offset = b * k_stride_b + head_k * k_stride_h;

    const uint elems_per_lane = head_dim / THREADS_PER_ROW;
    float q1_reg[HEAD_DIM_MAX / THREADS_PER_ROW];
    float q2_reg[HEAD_DIM_MAX / THREADS_PER_ROW];
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        q1_reg[i] = (d < head_dim) ? float(Q1[q1_offset + d]) : 0.0f;
        q2_reg[i] = (d < head_dim) ? float(Q2[q2_offset + d]) : 0.0f;
    }

    threadgroup half K1_tile[2][TILE_KV][HEAD_DIM_MAX];
    threadgroup half K2_tile[2][TILE_KV][HEAD_DIM_MAX];
    threadgroup half V_tile[2][TILE_KV][HEAD_DIM_MAX];

    float m1_prev = -INFINITY;
    float l1_prev = 0.0f;
    float m2_prev = -INFINITY;
    float l2_prev = 0.0f;

    float o1_acc[HEAD_DIM_MAX / THREADS_PER_ROW];
    float o2_acc[HEAD_DIM_MAX / THREADS_PER_ROW];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o1_acc[i] = 0.0f;
        o2_acc[i] = 0.0f;
    }

    // Causal: only iterate tiles where k_pos <= q_row
    const uint causal_limit = min(q_row + 1, seq_k);
    const uint num_kv_tiles = (causal_limit + TILE_KV - 1) / TILE_KV;

    // Preload first tile
    {
        const uint elems_to_load = TILE_KV * head_dim;
        const uint loads_per_thread = (elems_to_load + THREADS_PER_TG - 1) / THREADS_PER_TG;
        for (uint i = 0; i < loads_per_thread; ++i) {
            uint idx = tid_in_tg + i * THREADS_PER_TG;
            if (idx < elems_to_load) {
                uint kv_row = idx / head_dim;
                uint kv_col = idx % head_dim;
                if (kv_row < seq_k) {
                    K1_tile[0][kv_row][kv_col] = K1[kv_offset + kv_row * k_stride_s + kv_col];
                    K2_tile[0][kv_row][kv_col] = K2[kv_offset + kv_row * k_stride_s + kv_col];
                    V_tile[0][kv_row][kv_col] = V[kv_offset + kv_row * k_stride_s + kv_col];
                } else {
                    K1_tile[0][kv_row][kv_col] = half(0);
                    K2_tile[0][kv_row][kv_col] = half(0);
                    V_tile[0][kv_row][kv_col] = half(0);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf_compute = 0;

    for (uint tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
        uint buf_load = 1 - buf_compute;
        uint tile_start = tile_idx * TILE_KV;

        // Load next tile
        uint next_tile_start = (tile_idx + 1) * TILE_KV;
        if (tile_idx + 1 < num_kv_tiles) {
            const uint elems_to_load = TILE_KV * head_dim;
            const uint loads_per_thread = (elems_to_load + THREADS_PER_TG - 1) / THREADS_PER_TG;
            for (uint i = 0; i < loads_per_thread; ++i) {
                uint idx = tid_in_tg + i * THREADS_PER_TG;
                if (idx < elems_to_load) {
                    uint kv_row = idx / head_dim;
                    uint kv_col = idx % head_dim;
                    uint src_row = next_tile_start + kv_row;
                    if (src_row < seq_k) {
                        K1_tile[buf_load][kv_row][kv_col] = K1[kv_offset + src_row * k_stride_s + kv_col];
                        K2_tile[buf_load][kv_row][kv_col] = K2[kv_offset + src_row * k_stride_s + kv_col];
                        V_tile[buf_load][kv_row][kv_col] = V[kv_offset + src_row * k_stride_s + kv_col];
                    } else {
                        K1_tile[buf_load][kv_row][kv_col] = half(0);
                        K2_tile[buf_load][kv_row][kv_col] = half(0);
                        V_tile[buf_load][kv_row][kv_col] = half(0);
                    }
                }
            }
        }

        uint tile_end = min(tile_start + TILE_KV, seq_k);
        uint tile_len = tile_end - tile_start;

        float scores1[TILE_KV];
        float scores2[TILE_KV];
        for (uint ki = 0; ki < tile_len; ++ki) {
            uint k_pos = tile_start + ki;
            if (k_pos > q_row) {
                // Causal mask: future positions get -INF
                scores1[ki] = -INFINITY;
                scores2[ki] = -INFINITY;
                continue;
            }

            float dot1 = 0.0f;
            float dot2 = 0.0f;
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                dot1 += q1_reg[i] * float(K1_tile[buf_compute][ki][d]);
                dot2 += q2_reg[i] * float(K2_tile[buf_compute][ki][d]);
            }
            dot1 = simd_sum_f32(dot1);
            dot2 = simd_sum_f32(dot2);
            scores1[ki] = dot1 * scale;
            scores2[ki] = dot2 * scale;
        }
        for (uint ki = tile_len; ki < TILE_KV; ++ki) {
            scores1[ki] = -INFINITY;
            scores2[ki] = -INFINITY;
        }

        // Online softmax
        float m1_tile = -INFINITY;
        float m2_tile = -INFINITY;
        for (uint ki = 0; ki < tile_len; ++ki) {
            m1_tile = max(m1_tile, scores1[ki]);
            m2_tile = max(m2_tile, scores2[ki]);
        }

        float m1_new = max(m1_prev, m1_tile);
        float m2_new = max(m2_prev, m2_tile);
        float correction1 = exp(m1_prev - m1_new);
        float correction2 = exp(m2_prev - m2_new);

        float l1_new = l1_prev * correction1;
        float l2_new = l2_prev * correction2;
        for (uint ki = 0; ki < tile_len; ++ki) {
            l1_new += exp(scores1[ki] - m1_new);
            l2_new += exp(scores2[ki] - m2_new);
        }

        for (uint i = 0; i < elems_per_lane; ++i) {
            o1_acc[i] *= correction1;
            o2_acc[i] *= correction2;
        }
        for (uint ki = 0; ki < tile_len; ++ki) {
            float p1 = exp(scores1[ki] - m1_new);
            float p2 = exp(scores2[ki] - m2_new);
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                float v_val = float(V_tile[buf_compute][ki][d]);
                o1_acc[i] += p1 * v_val;
                o2_acc[i] += p2 * v_val;
            }
        }

        m1_prev = m1_new;
        m2_prev = m2_new;
        l1_prev = l1_new;
        l2_prev = l2_new;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // Store output
    const uint o_offset = b * q_stride_b + head_q * q_stride_h + q_row * q_stride_s;
    float inv_l1 = (l1_prev > 0.0f) ? (1.0f / l1_prev) : 0.0f;
    float inv_l2 = (l2_prev > 0.0f) ? (1.0f / l2_prev) : 0.0f;

    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        if (d < head_dim) {
            float out1 = o1_acc[i] * inv_l1;
            float out2 = o2_acc[i] * inv_l2;
            O[o_offset + d] = half(out1 - lambda_val * out2);
        }
    }
}

// ---------------------------------------------------------------------------
// Differential Attention kernel - Grouped-Query Attention (GQA)
//
// Explicit GQA ratio parameter for cleaner head mapping.
// ---------------------------------------------------------------------------

kernel void diff_attention_gqa(
    device const half* Q1           [[buffer(0)]],
    device const half* Q2           [[buffer(1)]],
    device const half* K1           [[buffer(2)]],
    device const half* K2           [[buffer(3)]],
    device const half* V            [[buffer(4)]],
    device half* O                  [[buffer(5)]],
    device const half* lambda_vals  [[buffer(6)]],
    constant uint& batch            [[buffer(7)]],
    constant uint& num_heads_q      [[buffer(8)]],
    constant uint& num_heads_k      [[buffer(9)]],
    constant uint& seq_q            [[buffer(10)]],
    constant uint& seq_k            [[buffer(11)]],
    constant uint& head_dim         [[buffer(12)]],
    constant float& scale           [[buffer(13)]],
    constant bool& lambda_per_head  [[buffer(14)]],
    constant uint& gqa_ratio        [[buffer(15)]],  // num_heads_q / num_heads_k
    constant bool& is_causal        [[buffer(16)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint tid_in_tg                  [[thread_index_in_threadgroup]],
    uint lane_id                    [[thread_index_in_simdgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    const uint head_q = tgid.x;
    const uint q_row_base = tgid.y * ROWS_PER_TG;
    const uint b = tgid.z;
    const uint head_k = head_q / gqa_ratio;
    const uint q_row = q_row_base + sg_id;
    if (q_row >= seq_q) return;

    const float lambda_val = float(lambda_per_head ? lambda_vals[head_q] : lambda_vals[0]);

    const uint q_stride_b = num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;
    const uint k_stride_b = num_heads_k * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;

    const uint q1_offset = b * q_stride_b + head_q * q_stride_h + q_row * q_stride_s;
    const uint q2_offset = q1_offset;
    const uint kv_offset = b * k_stride_b + head_k * k_stride_h;

    const uint elems_per_lane = head_dim / THREADS_PER_ROW;
    float q1_reg[HEAD_DIM_MAX / THREADS_PER_ROW];
    float q2_reg[HEAD_DIM_MAX / THREADS_PER_ROW];
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        q1_reg[i] = (d < head_dim) ? float(Q1[q1_offset + d]) : 0.0f;
        q2_reg[i] = (d < head_dim) ? float(Q2[q2_offset + d]) : 0.0f;
    }

    threadgroup half K1_tile[2][TILE_KV][HEAD_DIM_MAX];
    threadgroup half K2_tile[2][TILE_KV][HEAD_DIM_MAX];
    threadgroup half V_tile[2][TILE_KV][HEAD_DIM_MAX];

    float m1_prev = -INFINITY;
    float l1_prev = 0.0f;
    float m2_prev = -INFINITY;
    float l2_prev = 0.0f;

    float o1_acc[HEAD_DIM_MAX / THREADS_PER_ROW];
    float o2_acc[HEAD_DIM_MAX / THREADS_PER_ROW];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o1_acc[i] = 0.0f;
        o2_acc[i] = 0.0f;
    }

    const uint effective_seq = is_causal ? min(q_row + 1, seq_k) : seq_k;
    const uint num_kv_tiles = (effective_seq + TILE_KV - 1) / TILE_KV;

    // Preload first tile
    {
        const uint elems_to_load = TILE_KV * head_dim;
        const uint loads_per_thread = (elems_to_load + THREADS_PER_TG - 1) / THREADS_PER_TG;
        for (uint i = 0; i < loads_per_thread; ++i) {
            uint idx = tid_in_tg + i * THREADS_PER_TG;
            if (idx < elems_to_load) {
                uint kv_row = idx / head_dim;
                uint kv_col = idx % head_dim;
                if (kv_row < seq_k) {
                    K1_tile[0][kv_row][kv_col] = K1[kv_offset + kv_row * k_stride_s + kv_col];
                    K2_tile[0][kv_row][kv_col] = K2[kv_offset + kv_row * k_stride_s + kv_col];
                    V_tile[0][kv_row][kv_col] = V[kv_offset + kv_row * k_stride_s + kv_col];
                } else {
                    K1_tile[0][kv_row][kv_col] = half(0);
                    K2_tile[0][kv_row][kv_col] = half(0);
                    V_tile[0][kv_row][kv_col] = half(0);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf_compute = 0;

    for (uint tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
        uint buf_load = 1 - buf_compute;
        uint tile_start = tile_idx * TILE_KV;

        // Load next tile
        uint next_tile_start = (tile_idx + 1) * TILE_KV;
        if (tile_idx + 1 < num_kv_tiles) {
            const uint elems_to_load = TILE_KV * head_dim;
            const uint loads_per_thread = (elems_to_load + THREADS_PER_TG - 1) / THREADS_PER_TG;
            for (uint i = 0; i < loads_per_thread; ++i) {
                uint idx = tid_in_tg + i * THREADS_PER_TG;
                if (idx < elems_to_load) {
                    uint kv_row = idx / head_dim;
                    uint kv_col = idx % head_dim;
                    uint src_row = next_tile_start + kv_row;
                    if (src_row < seq_k) {
                        K1_tile[buf_load][kv_row][kv_col] = K1[kv_offset + src_row * k_stride_s + kv_col];
                        K2_tile[buf_load][kv_row][kv_col] = K2[kv_offset + src_row * k_stride_s + kv_col];
                        V_tile[buf_load][kv_row][kv_col] = V[kv_offset + src_row * k_stride_s + kv_col];
                    } else {
                        K1_tile[buf_load][kv_row][kv_col] = half(0);
                        K2_tile[buf_load][kv_row][kv_col] = half(0);
                        V_tile[buf_load][kv_row][kv_col] = half(0);
                    }
                }
            }
        }

        uint tile_end = min(tile_start + TILE_KV, seq_k);
        uint tile_len = tile_end - tile_start;

        float scores1[TILE_KV];
        float scores2[TILE_KV];
        for (uint ki = 0; ki < tile_len; ++ki) {
            uint k_pos = tile_start + ki;
            if (is_causal && k_pos > q_row) {
                scores1[ki] = -INFINITY;
                scores2[ki] = -INFINITY;
                continue;
            }
            float dot1 = 0.0f;
            float dot2 = 0.0f;
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                dot1 += q1_reg[i] * float(K1_tile[buf_compute][ki][d]);
                dot2 += q2_reg[i] * float(K2_tile[buf_compute][ki][d]);
            }
            dot1 = simd_sum_f32(dot1);
            dot2 = simd_sum_f32(dot2);
            scores1[ki] = dot1 * scale;
            scores2[ki] = dot2 * scale;
        }
        for (uint ki = tile_len; ki < TILE_KV; ++ki) {
            scores1[ki] = -INFINITY;
            scores2[ki] = -INFINITY;
        }

        float m1_tile = -INFINITY;
        float m2_tile = -INFINITY;
        for (uint ki = 0; ki < tile_len; ++ki) {
            m1_tile = max(m1_tile, scores1[ki]);
            m2_tile = max(m2_tile, scores2[ki]);
        }

        float m1_new = max(m1_prev, m1_tile);
        float m2_new = max(m2_prev, m2_tile);
        float correction1 = exp(m1_prev - m1_new);
        float correction2 = exp(m2_prev - m2_new);

        float l1_new = l1_prev * correction1;
        float l2_new = l2_prev * correction2;
        for (uint ki = 0; ki < tile_len; ++ki) {
            l1_new += exp(scores1[ki] - m1_new);
            l2_new += exp(scores2[ki] - m2_new);
        }

        for (uint i = 0; i < elems_per_lane; ++i) {
            o1_acc[i] *= correction1;
            o2_acc[i] *= correction2;
        }
        for (uint ki = 0; ki < tile_len; ++ki) {
            float p1 = exp(scores1[ki] - m1_new);
            float p2 = exp(scores2[ki] - m2_new);
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                float v_val = float(V_tile[buf_compute][ki][d]);
                o1_acc[i] += p1 * v_val;
                o2_acc[i] += p2 * v_val;
            }
        }

        m1_prev = m1_new;
        m2_prev = m2_new;
        l1_prev = l1_new;
        l2_prev = l2_new;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    const uint o_offset = b * q_stride_b + head_q * q_stride_h + q_row * q_stride_s;
    float inv_l1 = (l1_prev > 0.0f) ? (1.0f / l1_prev) : 0.0f;
    float inv_l2 = (l2_prev > 0.0f) ? (1.0f / l2_prev) : 0.0f;

    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        if (d < head_dim) {
            float out1 = o1_acc[i] * inv_l1;
            float out2 = o2_acc[i] * inv_l2;
            O[o_offset + d] = half(out1 - lambda_val * out2);
        }
    }
}

// ---------------------------------------------------------------------------
// Differential Attention kernel - FP4-quantized KV cache
//
// For memory-constrained inference with quantized KV cache.
// K1, K2, V are stored in FP4 E2M1 format with per-group scales.
// ---------------------------------------------------------------------------

kernel void diff_attention_kv_fp4(
    device const half* Q1              [[buffer(0)]],
    device const half* Q2              [[buffer(1)]],
    device const uint* K1_packed       [[buffer(2)]],  // FP4 packed: 8 nibbles per uint
    device const uint* K2_packed       [[buffer(3)]],
    device const uint* V_packed        [[buffer(4)]],
    device const half* K1_scales       [[buffer(5)]],
    device const half* K2_scales       [[buffer(6)]],
    device const half* V_scales        [[buffer(7)]],
    device half* O                     [[buffer(8)]],
    device const half* lambda_vals     [[buffer(9)]],
    constant uint& batch               [[buffer(10)]],
    constant uint& num_heads_q         [[buffer(11)]],
    constant uint& num_heads_k         [[buffer(12)]],
    constant uint& seq_q               [[buffer(13)]],
    constant uint& seq_k               [[buffer(14)]],
    constant uint& head_dim            [[buffer(15)]],
    constant float& scale              [[buffer(16)]],
    constant bool& lambda_per_head     [[buffer(17)]],
    constant uint& group_size          [[buffer(18)]],
    constant bool& is_causal           [[buffer(19)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint tid_in_tg                     [[thread_index_in_threadgroup]],
    uint lane_id                       [[thread_index_in_simdgroup]],
    uint sg_id                         [[simdgroup_index_in_threadgroup]]
) {
    const uint head_q = tgid.x;
    const uint q_row_base = tgid.y * ROWS_PER_TG;
    const uint b = tgid.z;
    const uint head_k = head_q * num_heads_k / num_heads_q;
    const uint q_row = q_row_base + sg_id;
    if (q_row >= seq_q) return;

    const float lambda_val = float(lambda_per_head ? lambda_vals[head_q] : lambda_vals[0]);

    const uint q_stride_b = num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;
    
    // KV cache stride: packed FP4
    const uint kv_stride_b = num_heads_k * seq_k * head_dim / FP4_PER_UINT;
    const uint kv_stride_h = seq_k * head_dim / FP4_PER_UINT;
    
    // Scale stride: one scale per group
    const uint scale_stride_b = num_heads_k * seq_k * ((head_dim + group_size - 1) / group_size);
    const uint scale_stride_h = seq_k * ((head_dim + group_size - 1) / group_size);
    const uint scales_per_row = (head_dim + group_size - 1) / group_size;

    const uint q1_offset = b * q_stride_b + head_q * q_stride_h + q_row * q_stride_s;
    const uint q2_offset = q1_offset;
    const uint kv_packed_offset = b * kv_stride_b + head_k * kv_stride_h;
    const uint scale_offset = b * scale_stride_b + head_k * scale_stride_h;

    // Load Q1 and Q2 rows
    const uint elems_per_lane = head_dim / THREADS_PER_ROW;
    float q1_reg[HEAD_DIM_MAX / THREADS_PER_ROW];
    float q2_reg[HEAD_DIM_MAX / THREADS_PER_ROW];
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        q1_reg[i] = (d < head_dim) ? float(Q1[q1_offset + d]) : 0.0f;
        q2_reg[i] = (d < head_dim) ? float(Q2[q2_offset + d]) : 0.0f;
    }

    // Shared memory for dequantized tiles
    threadgroup half K1_tile[TILE_KV][HEAD_DIM_MAX];
    threadgroup half K2_tile[TILE_KV][HEAD_DIM_MAX];
    threadgroup half V_tile[TILE_KV][HEAD_DIM_MAX];

    float m1_prev = -INFINITY;
    float l1_prev = 0.0f;
    float m2_prev = -INFINITY;
    float l2_prev = 0.0f;

    float o1_acc[HEAD_DIM_MAX / THREADS_PER_ROW];
    float o2_acc[HEAD_DIM_MAX / THREADS_PER_ROW];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o1_acc[i] = 0.0f;
        o2_acc[i] = 0.0f;
    }

    const uint effective_seq = is_causal ? min(q_row + 1, seq_k) : seq_k;
    const uint num_kv_tiles = (effective_seq + TILE_KV - 1) / TILE_KV;

    for (uint tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
        uint tile_start = tile_idx * TILE_KV;
        uint tile_end = min(tile_start + TILE_KV, seq_k);
        uint tile_len = tile_end - tile_start;

        // Cooperatively dequantize K1, K2, V tiles
        const uint elems_to_dequant = TILE_KV * head_dim;
        const uint loads_per_thread = (elems_to_dequant + THREADS_PER_TG - 1) / THREADS_PER_TG;
        
        for (uint i = 0; i < loads_per_thread; ++i) {
            uint idx = tid_in_tg + i * THREADS_PER_TG;
            if (idx < elems_to_dequant) {
                uint kv_row = idx / head_dim;
                uint kv_col = idx % head_dim;
                uint src_row = tile_start + kv_row;
                
                if (src_row < seq_k) {
                    uint group_idx = kv_col / group_size;
                    half k1_scale = K1_scales[scale_offset + src_row * scales_per_row + group_idx];
                    half k2_scale = K2_scales[scale_offset + src_row * scales_per_row + group_idx];
                    half v_scale = V_scales[scale_offset + src_row * scales_per_row + group_idx];
                    
                    // Load packed nibbles
                    uint packed_idx = src_row * head_dim / FP4_PER_UINT + kv_col / FP4_PER_UINT;
                    uint shift = (kv_col % FP4_PER_UINT) * 4;
                    
                    uint k1_packed = K1_packed[kv_packed_offset + packed_idx];
                    uint k2_packed = K2_packed[kv_packed_offset + packed_idx];
                    uint v_packed = V_packed[kv_packed_offset + packed_idx];
                    
                    uint8_t k1_nibble = (k1_packed >> shift) & 0xF;
                    uint8_t k2_nibble = (k2_packed >> shift) & 0xF;
                    uint8_t v_nibble = (v_packed >> shift) & 0xF;
                    
                    K1_tile[kv_row][kv_col] = dequant_fp4_scalar(k1_nibble, k1_scale);
                    K2_tile[kv_row][kv_col] = dequant_fp4_scalar(k2_nibble, k2_scale);
                    V_tile[kv_row][kv_col] = dequant_fp4_scalar(v_nibble, v_scale);
                } else {
                    K1_tile[kv_row][kv_col] = half(0);
                    K2_tile[kv_row][kv_col] = half(0);
                    V_tile[kv_row][kv_col] = half(0);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute attention scores
        float scores1[TILE_KV];
        float scores2[TILE_KV];
        for (uint ki = 0; ki < tile_len; ++ki) {
            uint k_pos = tile_start + ki;
            if (is_causal && k_pos > q_row) {
                scores1[ki] = -INFINITY;
                scores2[ki] = -INFINITY;
                continue;
            }
            
            float dot1 = 0.0f;
            float dot2 = 0.0f;
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                dot1 += q1_reg[i] * float(K1_tile[ki][d]);
                dot2 += q2_reg[i] * float(K2_tile[ki][d]);
            }
            dot1 = simd_sum_f32(dot1);
            dot2 = simd_sum_f32(dot2);
            scores1[ki] = dot1 * scale;
            scores2[ki] = dot2 * scale;
        }
        for (uint ki = tile_len; ki < TILE_KV; ++ki) {
            scores1[ki] = -INFINITY;
            scores2[ki] = -INFINITY;
        }

        // Online softmax
        float m1_tile = -INFINITY;
        float m2_tile = -INFINITY;
        for (uint ki = 0; ki < tile_len; ++ki) {
            m1_tile = max(m1_tile, scores1[ki]);
            m2_tile = max(m2_tile, scores2[ki]);
        }

        float m1_new = max(m1_prev, m1_tile);
        float m2_new = max(m2_prev, m2_tile);
        float correction1 = exp(m1_prev - m1_new);
        float correction2 = exp(m2_prev - m2_new);

        float l1_new = l1_prev * correction1;
        float l2_new = l2_prev * correction2;
        for (uint ki = 0; ki < tile_len; ++ki) {
            l1_new += exp(scores1[ki] - m1_new);
            l2_new += exp(scores2[ki] - m2_new);
        }

        for (uint i = 0; i < elems_per_lane; ++i) {
            o1_acc[i] *= correction1;
            o2_acc[i] *= correction2;
        }
        for (uint ki = 0; ki < tile_len; ++ki) {
            float p1 = exp(scores1[ki] - m1_new);
            float p2 = exp(scores2[ki] - m2_new);
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                float v_val = float(V_tile[ki][d]);
                o1_acc[i] += p1 * v_val;
                o2_acc[i] += p2 * v_val;
            }
        }

        m1_prev = m1_new;
        m2_prev = m2_new;
        l1_prev = l1_new;
        l2_prev = l2_new;

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store output
    const uint o_offset = b * q_stride_b + head_q * q_stride_h + q_row * q_stride_s;
    float inv_l1 = (l1_prev > 0.0f) ? (1.0f / l1_prev) : 0.0f;
    float inv_l2 = (l2_prev > 0.0f) ? (1.0f / l2_prev) : 0.0f;

    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        if (d < head_dim) {
            float out1 = o1_acc[i] * inv_l1;
            float out2 = o2_acc[i] * inv_l2;
            O[o_offset + d] = half(out1 - lambda_val * out2);
        }
    }
}
