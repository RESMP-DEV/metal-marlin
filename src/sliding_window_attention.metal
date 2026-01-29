// sliding_window_attention.metal - Sliding Window Attention for Apple Silicon
//
// Implements efficient sliding window attention for models like Mistral that use
// local attention patterns. Each token attends only to the most recent
// window_size tokens, providing O(seq * window) memory instead of O(seq^2).
//
// Key benefits:
//   - Linear memory complexity in sequence length
//   - Faster computation for long sequences (Mistral uses 32K window)
//   - Same API as flash attention for easy integration
//   - Supports both prefill (multiple query tokens) and decode (single query)
//
// Algorithm:
//   For each query at position q_pos:
//     k_start = max(0, q_pos - window_size + 1)
//     k_end = q_pos + 1
//     Compute attention only over K[k_start:k_end], V[k_start:k_end]
//
// Kernel variants:
//   sliding_window_attention_prefill     - Prefill with multiple Q tokens
//   sliding_window_attention_decode      - Decode with single Q token (seq_q=1)
//   sliding_window_attention_causal      - Prefill with causal + window mask
//
// Memory layout (all row-major):
//   Q: [batch, heads_q, seq_q, head_dim]
//   K: [batch, heads_kv, seq_k, head_dim]
//   V: [batch, heads_kv, seq_k, head_dim]
//   O: [batch, heads_q, seq_q, head_dim]
//
// The key optimization is that we only load and process K/V entries within
// the sliding window for each query position, rather than the entire sequence.
// This reduces both memory bandwidth and compute proportionally.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Configuration for Apple Silicon
// ---------------------------------------------------------------------------

// Tile dimensions - tuned for M-series with head_dim=64/128
constant constexpr uint TILE_Q_SW = 8;       // Query rows per threadgroup
constant constexpr uint TILE_KV_SW = 32;      // K/V rows per tile (within window)
constant constexpr uint HEAD_DIM_MAX_SW = 128;

// Thread organization
constant constexpr uint SIMD_SIZE_SW = 32;
constant constexpr uint NUM_SIMDGROUPS_SW = 4;  // 128 threads total
constant constexpr uint THREADS_PER_TG_SW = SIMD_SIZE_SW * NUM_SIMDGROUPS_SW;

// ---------------------------------------------------------------------------
// Fast simd reductions
// ---------------------------------------------------------------------------

inline float simd_max_sw(float val) {
    val = max(val, simd_shuffle_xor(val, 16));
    val = max(val, simd_shuffle_xor(val, 8));
    val = max(val, simd_shuffle_xor(val, 4));
    val = max(val, simd_shuffle_xor(val, 2));
    val = max(val, simd_shuffle_xor(val, 1));
    return val;
}

inline float simd_sum_sw(float val) {
    val += simd_shuffle_xor(val, 16);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);
    return val;
}

// ---------------------------------------------------------------------------
// SlidingWindowParams - packed parameters for kernel dispatch
// ---------------------------------------------------------------------------

struct SlidingWindowParams {
    uint batch;
    uint num_heads_q;
    uint num_heads_kv;
    uint seq_q;
    uint seq_k;
    uint head_dim;
    float scale;
    uint gqa_ratio;      // num_heads_q / num_heads_kv
    uint window_size;    // Sliding window size (e.g., 4096 for Mistral)
    uint is_causal;      // Whether to apply causal masking within window
};

// ---------------------------------------------------------------------------
// Sliding Window Attention - Prefill (Non-Causal within window)
//
// Each threadgroup processes TILE_Q_SW query rows.
// For each query position, we only load K/V from the sliding window.
// Uses online softmax to avoid materializing the full attention matrix.
//
// Dispatch: [num_heads_q, ceil(seq_q / TILE_Q_SW), batch] threadgroups
// ---------------------------------------------------------------------------

[[max_total_threads_per_threadgroup(256)]]
kernel void sliding_window_attention_prefill(
    device const half* Q            [[buffer(0)]],
    device const half* K            [[buffer(1)]],
    device const half* V            [[buffer(2)]],
    device half* O                  [[buffer(3)]],
    constant SlidingWindowParams& params [[buffer(4)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint lane_id                    [[thread_index_in_simdgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    const uint head_q = tgid.x;
    const uint q_tile_idx = tgid.y;
    const uint batch_idx = tgid.z;

    const uint head_dim = params.head_dim;
    const uint seq_q = params.seq_q;
    const uint seq_k = params.seq_k;
    const float scale = params.scale;
    const uint window_size = params.window_size;

    // GQA: map Q head to KV head
    const uint head_kv = head_q / params.gqa_ratio;

    // Base indices for this threadgroup
    const uint q_start = q_tile_idx * TILE_Q_SW;
    if (q_start >= seq_q) return;

    const uint q_rows = min(TILE_Q_SW, seq_q - q_start);

    // Strides for [batch, heads, seq, head_dim] layout
    const uint q_stride_b = params.num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;

    const uint k_stride_b = params.num_heads_kv * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;

    // Base offsets
    const uint q_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;
    const uint kv_base = batch_idx * k_stride_b + head_kv * k_stride_h;

    // ---------------------------------------------------------------------------
    // Threadgroup memory for Q tile and sliding window K/V
    // Q tile: [TILE_Q_SW][HEAD_DIM_MAX_SW] = 8 * 128 * 2 = 2 KB
    // K/V tiles: 2 * [TILE_KV_SW][HEAD_DIM_MAX_SW] * 2 = 2 * 32 * 128 * 2 * 2 = 16 KB
    // Total: ~18 KB, well within 32 KB limit
    // ---------------------------------------------------------------------------

    threadgroup half Q_tile[TILE_Q_SW][HEAD_DIM_MAX_SW];
    threadgroup half K_tile[2][TILE_KV_SW][HEAD_DIM_MAX_SW];
    threadgroup half V_tile[2][TILE_KV_SW][HEAD_DIM_MAX_SW];

    // ---------------------------------------------------------------------------
    // Cooperative Q tile load
    // ---------------------------------------------------------------------------
    {
        const uint elems_to_load = q_rows * head_dim;
        const uint loads_per_thread = (elems_to_load + THREADS_PER_TG_SW - 1) / THREADS_PER_TG_SW;

        for (uint i = 0; i < loads_per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG_SW;
            if (idx < elems_to_load) {
                uint q_row = idx / head_dim;
                uint q_col = idx % head_dim;
                Q_tile[q_row][q_col] = Q[q_base + q_row * q_stride_s + q_col];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---------------------------------------------------------------------------
    // Per-simdgroup state: each simdgroup handles TILE_Q_SW/NUM_SIMDGROUPS_SW = 2 query rows
    // ---------------------------------------------------------------------------

    const uint rows_per_sg = TILE_Q_SW / NUM_SIMDGROUPS_SW;  // 2
    const uint sg_q_start = sg_id * rows_per_sg;
    const uint sg_q_rows = min(rows_per_sg, (q_rows > sg_q_start) ? (q_rows - sg_q_start) : 0u);

    // Register allocation for Q (each lane holds head_dim/32 elements per row)
    const uint elems_per_lane = head_dim / SIMD_SIZE_SW;
    float q_reg[2][HEAD_DIM_MAX_SW / SIMD_SIZE_SW];  // 2 rows max, up to 4 elems each

    // Online softmax state per row
    float m_prev[2];  // Running max
    float l_prev[2];  // Running sum
    float o_acc[2][HEAD_DIM_MAX_SW / SIMD_SIZE_SW];  // Output accumulators

    // Initialize
    for (uint r = 0; r < rows_per_sg; ++r) {
        m_prev[r] = -INFINITY;
        l_prev[r] = 0.0f;
        for (uint i = 0; i < elems_per_lane; ++i) {
            o_acc[r][i] = 0.0f;
        }
    }

    // Load Q rows into registers
    for (uint r = 0; r < sg_q_rows; ++r) {
        uint q_row = sg_q_start + r;
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            q_reg[r][i] = float(Q_tile[q_row][d]);
        }
    }

    // ---------------------------------------------------------------------------
    // Process each query row independently with its own sliding window
    // For non-causal attention within window, window is [pos - window_size + 1, pos]
    // but we attend to all positions in window regardless of relative position
    // ---------------------------------------------------------------------------

    for (uint r = 0; r < sg_q_rows; ++r) {
        uint global_q_pos = q_start + sg_q_start + r;

        // Compute window bounds for this query position
        // k_start is the first K position in the window
        // For non-causal within window: attend to symmetric window around q_pos
        // Here we use causal window: [max(0, q_pos - window + 1), q_pos]
        uint k_window_start = (global_q_pos >= window_size - 1) ? (global_q_pos - window_size + 1) : 0;
        uint k_window_end = min(global_q_pos + 1, seq_k);
        uint window_len = k_window_end - k_window_start;

        if (window_len == 0) continue;

        // Number of K/V tiles to process for this window
        uint num_kv_tiles = (window_len + TILE_KV_SW - 1) / TILE_KV_SW;

        // Reset accumulators for this row
        m_prev[r] = -INFINITY;
        l_prev[r] = 0.0f;
        for (uint i = 0; i < elems_per_lane; ++i) {
            o_acc[r][i] = 0.0f;
        }

        // Process K/V tiles within window
        for (uint tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
            uint tile_start_in_window = tile_idx * TILE_KV_SW;
            uint abs_tile_start = k_window_start + tile_start_in_window;
            uint tile_len = min(uint(TILE_KV_SW), window_len - tile_start_in_window);

            // Load K/V tile into threadgroup memory
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                uint elems = tile_len * head_dim;
                uint per_thread = (elems + THREADS_PER_TG_SW - 1) / THREADS_PER_TG_SW;
                for (uint i = 0; i < per_thread; ++i) {
                    uint idx = tid + i * THREADS_PER_TG_SW;
                    if (idx < elems) {
                        uint kv_row = idx / head_dim;
                        uint kv_col = idx % head_dim;
                        uint global_k_idx = abs_tile_start + kv_row;
                        if (global_k_idx < seq_k) {
                            K_tile[0][kv_row][kv_col] = K[kv_base + global_k_idx * k_stride_s + kv_col];
                            V_tile[0][kv_row][kv_col] = V[kv_base + global_k_idx * k_stride_s + kv_col];
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute attention scores for this tile
            float scores[TILE_KV_SW];

            for (uint ki = 0; ki < tile_len; ++ki) {
                float dot = 0.0f;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    dot += q_reg[r][i] * float(K_tile[0][ki][d]);
                }
                dot = simd_sum_sw(dot);
                scores[ki] = dot * scale;
            }

            // Zero pad invalid positions
            for (uint ki = tile_len; ki < TILE_KV_SW; ++ki) {
                scores[ki] = -INFINITY;
            }

            // Online softmax update
            float m_tile = -INFINITY;
            for (uint ki = 0; ki < tile_len; ++ki) {
                m_tile = max(m_tile, scores[ki]);
            }

            float m_new = max(m_prev[r], m_tile);
            float corr = exp(m_prev[r] - m_new);

            // Rescale and accumulate
            l_prev[r] *= corr;
            for (uint i = 0; i < elems_per_lane; ++i) {
                o_acc[r][i] *= corr;
            }

            // Accumulate new contributions
            for (uint ki = 0; ki < tile_len; ++ki) {
                float p = exp(scores[ki] - m_new);
                l_prev[r] += p;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    o_acc[r][i] += p * float(V_tile[0][ki][d]);
                }
            }

            m_prev[r] = m_new;
        }
    }

    // ---------------------------------------------------------------------------
    // Normalize and store output
    // ---------------------------------------------------------------------------

    const uint o_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;

    for (uint r = 0; r < sg_q_rows; ++r) {
        uint global_q = q_start + sg_q_start + r;
        if (global_q >= seq_q) continue;

        float inv_l = (l_prev[r] > 0.0f) ? (1.0f / l_prev[r]) : 0.0f;

        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                O[o_base + (sg_q_start + r) * q_stride_s + d] = half(o_acc[r][i] * inv_l);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sliding Window Attention - Causal (with sliding window constraint)
//
// Combines causal masking with sliding window:
//   For query at position q_pos, attend to K positions in range:
//   [max(0, q_pos - window_size + 1), q_pos]
//
// This is the typical pattern for Mistral-style models.
//
// Dispatch: [num_heads_q, ceil(seq_q / TILE_Q_SW), batch] threadgroups
// ---------------------------------------------------------------------------

kernel void sliding_window_attention_causal(
    device const half* Q            [[buffer(0)]],
    device const half* K            [[buffer(1)]],
    device const half* V            [[buffer(2)]],
    device half* O                  [[buffer(3)]],
    constant SlidingWindowParams& params [[buffer(4)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint lane_id                    [[thread_index_in_simdgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    const uint head_q = tgid.x;
    const uint q_tile_idx = tgid.y;
    const uint batch_idx = tgid.z;

    const uint head_dim = params.head_dim;
    const uint seq_q = params.seq_q;
    const uint seq_k = params.seq_k;
    const float scale = params.scale;
    const uint window_size = params.window_size;

    const uint head_kv = head_q / params.gqa_ratio;

    const uint q_start = q_tile_idx * TILE_Q_SW;
    if (q_start >= seq_q) return;

    const uint q_rows = min(TILE_Q_SW, seq_q - q_start);

    // Strides
    const uint q_stride_b = params.num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;
    const uint k_stride_b = params.num_heads_kv * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;

    const uint q_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;
    const uint kv_base = batch_idx * k_stride_b + head_kv * k_stride_h;

    threadgroup half Q_tile[TILE_Q_SW][HEAD_DIM_MAX_SW];
    threadgroup half K_tile[TILE_KV_SW][HEAD_DIM_MAX_SW];
    threadgroup half V_tile[TILE_KV_SW][HEAD_DIM_MAX_SW];

    // Load Q tile
    {
        const uint elems = q_rows * head_dim;
        const uint per_thread = (elems + THREADS_PER_TG_SW - 1) / THREADS_PER_TG_SW;
        for (uint i = 0; i < per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG_SW;
            if (idx < elems) {
                uint q_row = idx / head_dim;
                uint q_col = idx % head_dim;
                Q_tile[q_row][q_col] = Q[q_base + q_row * q_stride_s + q_col];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint rows_per_sg = TILE_Q_SW / NUM_SIMDGROUPS_SW;
    const uint sg_q_start = sg_id * rows_per_sg;
    const uint sg_q_rows = min(rows_per_sg, (q_rows > sg_q_start) ? (q_rows - sg_q_start) : 0u);

    const uint elems_per_lane = head_dim / SIMD_SIZE_SW;
    float q_reg[2][HEAD_DIM_MAX_SW / SIMD_SIZE_SW];
    float m_prev[2];
    float l_prev[2];
    float o_acc[2][HEAD_DIM_MAX_SW / SIMD_SIZE_SW];

    // Initialize per-row accumulators
    for (uint r = 0; r < rows_per_sg; ++r) {
        m_prev[r] = -INFINITY;
        l_prev[r] = 0.0f;
        for (uint i = 0; i < elems_per_lane; ++i) o_acc[r][i] = 0.0f;
    }

    // Load Q into registers
    for (uint r = 0; r < sg_q_rows; ++r) {
        uint q_row = sg_q_start + r;
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            q_reg[r][i] = float(Q_tile[q_row][d]);
        }
    }

    // ---------------------------------------------------------------------------
    // Compute the union of all sliding windows for this Q tile
    // This allows us to share K/V loading across query rows
    //
    // For Q positions [q_start, q_start + q_rows - 1]:
    //   - Minimum k_start = max(0, q_start - window_size + 1)
    //   - Maximum k_end = min(q_start + q_rows, seq_k)
    // ---------------------------------------------------------------------------

    const uint min_q_pos = q_start;
    const uint max_q_pos = q_start + q_rows - 1;

    // Union window bounds
    const uint union_k_start = (min_q_pos >= window_size - 1) ? (min_q_pos - window_size + 1) : 0;
    const uint union_k_end = min(max_q_pos + 1, seq_k);
    const uint union_window_len = union_k_end - union_k_start;

    if (union_window_len == 0) return;

    const uint num_kv_tiles = (union_window_len + TILE_KV_SW - 1) / TILE_KV_SW;

    // Process K/V tiles
    for (uint tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
        uint tile_start_in_union = tile_idx * TILE_KV_SW;
        uint abs_tile_start = union_k_start + tile_start_in_union;
        uint tile_len = min(uint(TILE_KV_SW), union_window_len - tile_start_in_union);

        // Load K/V tile
        {
            uint elems = tile_len * head_dim;
            uint per_thread = (elems + THREADS_PER_TG_SW - 1) / THREADS_PER_TG_SW;
            for (uint i = 0; i < per_thread; ++i) {
                uint idx = tid + i * THREADS_PER_TG_SW;
                if (idx < elems) {
                    uint kv_row = idx / head_dim;
                    uint kv_col = idx % head_dim;
                    uint global_k_idx = abs_tile_start + kv_row;
                    K_tile[kv_row][kv_col] = K[kv_base + global_k_idx * k_stride_s + kv_col];
                    V_tile[kv_row][kv_col] = V[kv_base + global_k_idx * k_stride_s + kv_col];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute attention for each query row with causal + window masking
        for (uint r = 0; r < sg_q_rows; ++r) {
            uint global_q_pos = q_start + sg_q_start + r;

            // Per-query window bounds
            uint q_k_start = (global_q_pos >= window_size - 1) ? (global_q_pos - window_size + 1) : 0;
            uint q_k_end = global_q_pos + 1;  // Causal: can only attend up to self

            float scores[TILE_KV_SW];

            for (uint ki = 0; ki < tile_len; ++ki) {
                uint global_k_pos = abs_tile_start + ki;

                // Check both window and causal constraints
                bool in_window = (global_k_pos >= q_k_start) && (global_k_pos < q_k_end);

                if (in_window) {
                    // Compute dot product
                    float dot = 0.0f;
                    for (uint i = 0; i < elems_per_lane; ++i) {
                        uint d = lane_id * elems_per_lane + i;
                        dot += q_reg[r][i] * float(K_tile[ki][d]);
                    }
                    dot = simd_sum_sw(dot);
                    scores[ki] = dot * scale;
                } else {
                    scores[ki] = -INFINITY;
                }
            }

            for (uint ki = tile_len; ki < TILE_KV_SW; ++ki) {
                scores[ki] = -INFINITY;
            }

            // Online softmax
            float m_tile = -INFINITY;
            for (uint ki = 0; ki < tile_len; ++ki) {
                m_tile = max(m_tile, scores[ki]);
            }

            float m_new = max(m_prev[r], m_tile);
            float corr = exp(m_prev[r] - m_new);

            l_prev[r] *= corr;
            for (uint i = 0; i < elems_per_lane; ++i) {
                o_acc[r][i] *= corr;
            }

            for (uint ki = 0; ki < tile_len; ++ki) {
                float p = exp(scores[ki] - m_new);
                l_prev[r] += p;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    o_acc[r][i] += p * float(V_tile[ki][d]);
                }
            }

            m_prev[r] = m_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store output
    const uint o_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;

    for (uint r = 0; r < sg_q_rows; ++r) {
        uint global_q = q_start + sg_q_start + r;
        if (global_q >= seq_q) continue;

        float inv_l = (l_prev[r] > 0.0f) ? (1.0f / l_prev[r]) : 0.0f;

        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                O[o_base + (sg_q_start + r) * q_stride_s + d] = half(o_acc[r][i] * inv_l);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sliding Window Attention - Decode (Optimized for seq_q=1)
//
// Specialized kernel for autoregressive decoding with sliding window.
// Each token only attends to the most recent window_size tokens in KV cache.
//
// This is the most common use case for Mistral/Mixtral during generation.
//
// Dispatch: [num_seqs * num_heads_q, 1, 1] threadgroups
// ---------------------------------------------------------------------------

kernel void sliding_window_attention_decode(
    device const half* Q            [[buffer(0)]],
    device const half* K            [[buffer(1)]],
    device const half* V            [[buffer(2)]],
    device half* O                  [[buffer(3)]],
    constant uint& num_seqs         [[buffer(4)]],
    constant uint& num_heads_q      [[buffer(5)]],
    constant uint& num_heads_kv     [[buffer(6)]],
    constant uint& seq_k            [[buffer(7)]],  // KV cache length
    constant uint& head_dim         [[buffer(8)]],
    constant float& scale           [[buffer(9)]],
    constant uint& window_size      [[buffer(10)]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint lane_id                    [[thread_index_in_simdgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    // Decode has seq_q = 1, one threadgroup per (sequence, head) pair
    const uint seq_idx = tgid / num_heads_q;
    const uint head_q = tgid % num_heads_q;

    if (seq_idx >= num_seqs) return;

    const uint gqa_ratio = num_heads_q / num_heads_kv;
    const uint head_kv = head_q / gqa_ratio;

    // Q layout: [num_seqs, num_heads_q, head_dim]
    const uint q_offset = seq_idx * num_heads_q * head_dim + head_q * head_dim;

    // K/V layout: [num_seqs, num_heads_kv, seq_k, head_dim]
    const uint kv_stride_s = head_dim;
    const uint kv_stride_h = seq_k * head_dim;
    const uint kv_stride_b = num_heads_kv * kv_stride_h;
    const uint kv_base = seq_idx * kv_stride_b + head_kv * kv_stride_h;

    // Current query position is seq_k - 1 (the last token in KV cache)
    // For decode, we're generating token at position seq_k - 1
    const uint q_pos = seq_k - 1;

    // Sliding window bounds
    const uint k_start = (q_pos >= window_size - 1) ? (q_pos - window_size + 1) : 0;
    const uint k_end = seq_k;  // Can attend to all cached KV (up to current position)
    const uint window_len = k_end - k_start;

    if (window_len == 0) return;

    // Load Q into registers (distributed across simdgroup 0)
    const uint elems_per_lane = head_dim / SIMD_SIZE_SW;
    float q_reg[HEAD_DIM_MAX_SW / SIMD_SIZE_SW];

    if (sg_id == 0) {
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            q_reg[i] = (d < head_dim) ? float(Q[q_offset + d]) : 0.0f;
        }
    }

    // Threadgroup memory for K/V window (double-buffered)
    threadgroup half K_smem[2][TILE_KV_SW][HEAD_DIM_MAX_SW];
    threadgroup half V_smem[2][TILE_KV_SW][HEAD_DIM_MAX_SW];

    // Online softmax state
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o_acc[HEAD_DIM_MAX_SW / SIMD_SIZE_SW];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o_acc[i] = 0.0f;
    }

    const uint num_tiles = (window_len + TILE_KV_SW - 1) / TILE_KV_SW;

    // Preload first tile from window
    {
        uint tile_len = min(uint(TILE_KV_SW), window_len);
        uint elems = tile_len * head_dim;
        uint per_thread = (elems + THREADS_PER_TG_SW - 1) / THREADS_PER_TG_SW;
        for (uint i = 0; i < per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG_SW;
            if (idx < elems) {
                uint kv_row = idx / head_dim;
                uint kv_col = idx % head_dim;
                uint global_k = k_start + kv_row;
                K_smem[0][kv_row][kv_col] = K[kv_base + global_k * kv_stride_s + kv_col];
                V_smem[0][kv_row][kv_col] = V[kv_base + global_k * kv_stride_s + kv_col];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf = 0;

    for (uint tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        uint buf_load = 1 - buf;
        uint tile_start_in_window = tile_idx * TILE_KV_SW;
        uint tile_len = min(uint(TILE_KV_SW), window_len - tile_start_in_window);

        // Load next tile (all threads)
        if (tile_idx + 1 < num_tiles) {
            uint next_tile_start = (tile_idx + 1) * TILE_KV_SW;
            uint next_len = min(uint(TILE_KV_SW), window_len - next_tile_start);
            uint elems = next_len * head_dim;
            uint per_thread = (elems + THREADS_PER_TG_SW - 1) / THREADS_PER_TG_SW;
            for (uint i = 0; i < per_thread; ++i) {
                uint idx = tid + i * THREADS_PER_TG_SW;
                if (idx < elems) {
                    uint kv_row = idx / head_dim;
                    uint kv_col = idx % head_dim;
                    uint global_k = k_start + next_tile_start + kv_row;
                    K_smem[buf_load][kv_row][kv_col] = K[kv_base + global_k * kv_stride_s + kv_col];
                    V_smem[buf_load][kv_row][kv_col] = V[kv_base + global_k * kv_stride_s + kv_col];
                }
            }
        }

        // Compute (simdgroup 0 only for single Q)
        if (sg_id == 0) {
            float scores[TILE_KV_SW];

            for (uint ki = 0; ki < tile_len; ++ki) {
                float dot = 0.0f;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    dot += q_reg[i] * float(K_smem[buf][ki][d]);
                }
                dot = simd_sum_sw(dot);
                scores[ki] = dot * scale;
            }

            for (uint ki = tile_len; ki < TILE_KV_SW; ++ki) {
                scores[ki] = -INFINITY;
            }

            // Online softmax
            float m_tile = -INFINITY;
            for (uint ki = 0; ki < tile_len; ++ki) {
                m_tile = max(m_tile, scores[ki]);
            }

            float m_new = max(m_prev, m_tile);
            float corr = exp(m_prev - m_new);

            l_prev *= corr;
            for (uint i = 0; i < elems_per_lane; ++i) {
                o_acc[i] *= corr;
            }

            for (uint ki = 0; ki < tile_len; ++ki) {
                float p = exp(scores[ki] - m_new);
                l_prev += p;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    o_acc[i] += p * float(V_smem[buf][ki][d]);
                }
            }

            m_prev = m_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf = buf_load;
    }

    // Store output (simdgroup 0)
    if (sg_id == 0) {
        const uint o_offset = seq_idx * num_heads_q * head_dim + head_q * head_dim;
        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;

        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                O[o_offset + d] = half(o_acc[i] * inv_l);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sliding Window Attention - GQA (Grouped Query Attention)
//
// Optimized for high GQA ratios where multiple Q heads share KV heads.
// Loads KV once and processes multiple Q heads in parallel.
//
// Dispatch: [num_heads_kv, ceil(seq_q / TILE_Q_SW), batch] threadgroups
// ---------------------------------------------------------------------------

kernel void sliding_window_attention_gqa(
    device const half* Q            [[buffer(0)]],
    device const half* K            [[buffer(1)]],
    device const half* V            [[buffer(2)]],
    device half* O                  [[buffer(3)]],
    constant SlidingWindowParams& params [[buffer(4)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint lane_id                    [[thread_index_in_simdgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    // Process multiple Q heads sharing one KV head
    const uint head_kv = tgid.x;
    const uint q_tile_idx = tgid.y;
    const uint batch_idx = tgid.z;

    const uint gqa_ratio = params.gqa_ratio;
    const uint head_dim = params.head_dim;
    const uint seq_q = params.seq_q;
    const uint seq_k = params.seq_k;
    const float scale = params.scale;
    const uint window_size = params.window_size;

    const uint q_start = q_tile_idx * TILE_Q_SW;
    if (q_start >= seq_q) return;

    const uint q_rows = min(TILE_Q_SW, seq_q - q_start);

    // Strides
    const uint q_stride_b = params.num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;
    const uint k_stride_b = params.num_heads_kv * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;

    const uint kv_base = batch_idx * k_stride_b + head_kv * k_stride_h;

    // K/V tiles are shared across all Q heads in this group
    threadgroup half K_tile[TILE_KV_SW][HEAD_DIM_MAX_SW];
    threadgroup half V_tile[TILE_KV_SW][HEAD_DIM_MAX_SW];

    // Each simdgroup handles one Q head within the GQA group
    const uint heads_per_pass = min(gqa_ratio, uint(NUM_SIMDGROUPS_SW));
    const uint num_head_passes = (gqa_ratio + heads_per_pass - 1) / heads_per_pass;

    // Compute union window for this Q tile
    const uint min_q_pos = q_start;
    const uint max_q_pos = q_start + q_rows - 1;
    const uint union_k_start = (min_q_pos >= window_size - 1) ? (min_q_pos - window_size + 1) : 0;
    const uint union_k_end = min(max_q_pos + 1, seq_k);
    const uint union_window_len = union_k_end - union_k_start;

    if (union_window_len == 0) return;

    const uint num_kv_tiles = (union_window_len + TILE_KV_SW - 1) / TILE_KV_SW;

    // Process each Q head pass
    for (uint head_pass = 0; head_pass < num_head_passes; ++head_pass) {
        uint head_offset = head_pass * heads_per_pass + sg_id;
        if (head_offset >= gqa_ratio) continue;

        uint head_q = head_kv * gqa_ratio + head_offset;
        if (head_q >= params.num_heads_q) continue;

        const uint q_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;

        // Load Q and initialize accumulators
        const uint elems_per_lane = head_dim / SIMD_SIZE_SW;
        float q_reg[TILE_Q_SW][HEAD_DIM_MAX_SW / SIMD_SIZE_SW];
        float m_prev[TILE_Q_SW];
        float l_prev[TILE_Q_SW];
        float o_acc[TILE_Q_SW][HEAD_DIM_MAX_SW / SIMD_SIZE_SW];

        for (uint r = 0; r < q_rows; ++r) {
            m_prev[r] = -INFINITY;
            l_prev[r] = 0.0f;
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                q_reg[r][i] = float(Q[q_base + r * q_stride_s + d]);
                o_acc[r][i] = 0.0f;
            }
        }

        // Process K/V tiles
        for (uint tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
            uint tile_start_in_union = tile_idx * TILE_KV_SW;
            uint abs_tile_start = union_k_start + tile_start_in_union;
            uint tile_len = min(uint(TILE_KV_SW), union_window_len - tile_start_in_union);

            // First head pass loads K/V
            if (head_pass == 0) {
                uint elems = tile_len * head_dim;
                uint per_thread = (elems + THREADS_PER_TG_SW - 1) / THREADS_PER_TG_SW;
                for (uint i = 0; i < per_thread; ++i) {
                    uint idx = tid + i * THREADS_PER_TG_SW;
                    if (idx < elems) {
                        uint kv_row = idx / head_dim;
                        uint kv_col = idx % head_dim;
                        uint global_k_idx = abs_tile_start + kv_row;
                        K_tile[kv_row][kv_col] = K[kv_base + global_k_idx * k_stride_s + kv_col];
                        V_tile[kv_row][kv_col] = V[kv_base + global_k_idx * k_stride_s + kv_col];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute attention with causal + window masking
            for (uint r = 0; r < q_rows; ++r) {
                uint global_q_pos = q_start + r;
                uint q_k_start = (global_q_pos >= window_size - 1) ? (global_q_pos - window_size + 1) : 0;
                uint q_k_end = global_q_pos + 1;

                float scores[TILE_KV_SW];

                for (uint ki = 0; ki < tile_len; ++ki) {
                    uint global_k_pos = abs_tile_start + ki;
                    bool in_window = (global_k_pos >= q_k_start) && (global_k_pos < q_k_end);

                    if (in_window) {
                        float dot = 0.0f;
                        for (uint i = 0; i < elems_per_lane; ++i) {
                            uint d = lane_id * elems_per_lane + i;
                            dot += q_reg[r][i] * float(K_tile[ki][d]);
                        }
                        dot = simd_sum_sw(dot);
                        scores[ki] = dot * scale;
                    } else {
                        scores[ki] = -INFINITY;
                    }
                }

                for (uint ki = tile_len; ki < TILE_KV_SW; ++ki) {
                    scores[ki] = -INFINITY;
                }

                // Online softmax
                float m_tile = -INFINITY;
                for (uint ki = 0; ki < tile_len; ++ki) {
                    m_tile = max(m_tile, scores[ki]);
                }

                float m_new = max(m_prev[r], m_tile);
                float corr = exp(m_prev[r] - m_new);

                l_prev[r] *= corr;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    o_acc[r][i] *= corr;
                }

                for (uint ki = 0; ki < tile_len; ++ki) {
                    float p = exp(scores[ki] - m_new);
                    l_prev[r] += p;
                    for (uint i = 0; i < elems_per_lane; ++i) {
                        uint d = lane_id * elems_per_lane + i;
                        o_acc[r][i] += p * float(V_tile[ki][d]);
                    }
                }

                m_prev[r] = m_new;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Store output for this head
        const uint o_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;

        for (uint r = 0; r < q_rows; ++r) {
            float inv_l = (l_prev[r] > 0.0f) ? (1.0f / l_prev[r]) : 0.0f;
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                if (d < head_dim) {
                    O[o_base + r * q_stride_s + d] = half(o_acc[r][i] * inv_l);
                }
            }
        }
    }
}
