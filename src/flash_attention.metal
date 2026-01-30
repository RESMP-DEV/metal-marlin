// flash_attention.metal - Tiled Flash Attention for Apple Silicon
//
// Computes O = softmax(Q @ K^T / sqrt(d)) @ V without materializing the
// full [seq_q, seq_k] attention matrix. Uses the online softmax algorithm
// (Milakov & Gimelshein 2018) to stream through K/V tiles incrementally.
//
// Kernel variants:
//   1. flash_attention              - Standard multi-head attention
//   2. flash_attention_causal       - Causal (autoregressive) masking
//   3. flash_attention_gqa          - Grouped-query attention (K/V heads < Q heads)
//   4. flash_attention_kv_fp4       - FP4-quantized KV cache (per-row scale)
//   5. flash_attention_kv_int4      - INT4-quantized KV cache (signed, per-row scale)
//
// Memory layout (all row-major):
//   Q: [batch, num_heads_q, seq_q, head_dim]
//   K: [batch, num_heads_k, seq_k, head_dim]
//   V: [batch, num_heads_k, seq_k, head_dim]
//   O: [batch, num_heads_q, seq_q, head_dim]
//
// Design notes:
//   - Each threadgroup processes one query row (one position in one head)
//   - K/V are tiled along seq_k dimension; tiles loaded cooperatively
//   - Online softmax maintains running (m, l) statistics per query row
//   - Double-buffering of K/V tiles hides load latency on M4 Max
//   - head_dim is compile-time fixed via function constants for register allocation
//
// CUDA -> Metal mapping:
//   Flash Attention 2 uses warp-level reductions and shared memory for QK^T
//   Metal equivalent uses simdgroup reductions and threadgroup memory
//   __shfl_xor_sync -> simd_shuffle_xor
//   atomicMax        -> simd_max (within simdgroup), threadgroup atomics

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Tile dimensions
//
// TILE_KV: Number of K/V rows processed per outer loop iteration.
//   64 balances threadgroup memory usage against loop overhead.
//   At head_dim=128: K_tile = 64*128*2B = 16KB, V_tile = 16KB, total = 32KB.
//   For head_dim=64: total = 16KB (fits easily in 32KB budget).
//
// HEAD_DIM_MAX: Maximum supported head dimension. Actual dimension passed
//   at dispatch time; we statically allocate for the max to avoid VLAs.
//   Common values: 64 (GPT-2), 80 (Llama 3.2), 128 (Llama, Mistral).
// ---------------------------------------------------------------------------

constant constexpr uint TILE_KV = 64;
constant constexpr uint HEAD_DIM_MAX = 128;

// Threads per threadgroup: one simdgroup (32 threads) handles one query row.
// Multiple simdgroups can process multiple query rows per threadgroup for
// better occupancy, but the base case is 1 row = 1 simdgroup = 32 threads.
constant constexpr uint THREADS_PER_ROW = 32;
constant constexpr uint ROWS_PER_TG = 4;  // 4 query rows per threadgroup
constant constexpr uint THREADS_PER_TG = THREADS_PER_ROW * ROWS_PER_TG;  // 128
constant constexpr uint FP4_PER_UINT = 8;

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

inline float simd_max_f32(float val, uint lane_id [[thread_index_in_simdgroup]]) {
    // Tree reduction across 32 lanes
    val = max(val, simd_shuffle_xor(val, 16));
    val = max(val, simd_shuffle_xor(val, 8));
    val = max(val, simd_shuffle_xor(val, 4));
    val = max(val, simd_shuffle_xor(val, 2));
    val = max(val, simd_shuffle_xor(val, 1));
    return val;
}

inline float simd_sum_f32(float val) {
    val += simd_shuffle_xor(val, 16);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);
    return val;
}

// ---------------------------------------------------------------------------
// FP4/INT4 dequant helpers (per-row scale)
// ---------------------------------------------------------------------------

inline half dequant_fp4_scalar(uint nibble, half scale) {
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;

    half magnitude;
    if (exp_bits == 0) {
        magnitude = half(man_bit) * half(0.25h);
    } else {
        half power = half(1u << (exp_bits - 1));
        half mantissa = half(1.0h) + half(man_bit) * half(0.5h);
        magnitude = power * mantissa;
    }

    half result = sign_bit ? -magnitude : magnitude;
    return result * scale;
}

inline half dequant_s4_scalar(uint nibble, half scale) {
    int signed_val = int(nibble & 0xFu) - 8;
    return half(signed_val) * scale;
}

inline void load_kv_fp4_tile(
    device const uint* packed,
    device const half* scales,
    threadgroup half (&tile)[TILE_KV][HEAD_DIM_MAX],
    uint head_dim,
    uint seq_k,
    uint packed_head_dim,
    uint kv_packed_offset,
    uint scale_offset,
    uint tile_start,
    uint thread_idx
) {
    const uint packed_elems = TILE_KV * packed_head_dim;
    const uint packed_per_thread = (packed_elems + THREADS_PER_TG - 1) / THREADS_PER_TG;

    for (uint i = 0; i < packed_per_thread; ++i) {
        uint idx = thread_idx + i * THREADS_PER_TG;
        if (idx >= packed_elems) {
            continue;
        }

        uint kv_row = idx / packed_head_dim;
        uint packed_col = idx % packed_head_dim;
        uint src_row = tile_start + kv_row;
        uint base_col = packed_col * FP4_PER_UINT;

        if (src_row < seq_k) {
            half s = scales[scale_offset + src_row];
            uint packed_word = packed[kv_packed_offset + src_row * packed_head_dim + packed_col];
            for (uint j = 0; j < FP4_PER_UINT; ++j) {
                uint d = base_col + j;
                if (d < head_dim) {
                    uint nibble = (packed_word >> (j * 4)) & 0xFu;
                    tile[kv_row][d] = dequant_fp4_scalar(nibble, s);
                }
            }
        } else {
            for (uint j = 0; j < FP4_PER_UINT; ++j) {
                uint d = base_col + j;
                if (d < head_dim) {
                    tile[kv_row][d] = half(0);
                }
            }
        }
    }
}

inline void load_kv_int4_tile(
    device const uint* packed,
    device const half* scales,
    threadgroup half (&tile)[TILE_KV][HEAD_DIM_MAX],
    uint head_dim,
    uint seq_k,
    uint packed_head_dim,
    uint kv_packed_offset,
    uint scale_offset,
    uint tile_start,
    uint thread_idx
) {
    const uint packed_elems = TILE_KV * packed_head_dim;
    const uint packed_per_thread = (packed_elems + THREADS_PER_TG - 1) / THREADS_PER_TG;

    for (uint i = 0; i < packed_per_thread; ++i) {
        uint idx = thread_idx + i * THREADS_PER_TG;
        if (idx >= packed_elems) {
            continue;
        }

        uint kv_row = idx / packed_head_dim;
        uint packed_col = idx % packed_head_dim;
        uint src_row = tile_start + kv_row;
        uint base_col = packed_col * FP4_PER_UINT;

        if (src_row < seq_k) {
            half s = scales[scale_offset + src_row];
            uint packed_word = packed[kv_packed_offset + src_row * packed_head_dim + packed_col];
            for (uint j = 0; j < FP4_PER_UINT; ++j) {
                uint d = base_col + j;
                if (d < head_dim) {
                    uint nibble = (packed_word >> (j * 4)) & 0xFu;
                    tile[kv_row][d] = dequant_s4_scalar(nibble, s);
                }
            }
        } else {
            for (uint j = 0; j < FP4_PER_UINT; ++j) {
                uint d = base_col + j;
                if (d < head_dim) {
                    tile[kv_row][d] = half(0);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Flash Attention kernel - standard (non-causal)
//
// Dispatch: [num_heads_q, ceil(seq_q / ROWS_PER_TG), batch] threadgroups
//           THREADS_PER_TG threads per threadgroup
// ---------------------------------------------------------------------------

kernel void flash_attention(
    device const half* Q            [[buffer(0)]],
    device const half* K            [[buffer(1)]],
    device const half* V            [[buffer(2)]],
    device half* O                  [[buffer(3)]],
    constant uint& batch            [[buffer(4)]],
    constant uint& num_heads_q      [[buffer(5)]],
    constant uint& num_heads_k      [[buffer(6)]],
    constant uint& seq_q            [[buffer(7)]],
    constant uint& seq_k            [[buffer(8)]],
    constant uint& head_dim         [[buffer(9)]],
    constant float& scale           [[buffer(10)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint tid_in_tg                  [[thread_index_in_threadgroup]],
    uint lane_id                    [[thread_index_in_simdgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    // Threadgroup mapping:
    //   tgid.x = head index (0..num_heads_q-1)
    //   tgid.y = query row block (0..ceil(seq_q/ROWS_PER_TG)-1)
    //   tgid.z = batch index (0..batch-1)
    const uint head_q = tgid.x;
    const uint q_row_base = tgid.y * ROWS_PER_TG;
    const uint b = tgid.z;

    // For GQA: map Q head to K/V head
    const uint head_k = head_q * num_heads_k / num_heads_q;

    // This simdgroup handles one query row
    const uint q_row = q_row_base + sg_id;
    if (q_row >= seq_q) return;

    // Strides for [batch, heads, seq, head_dim] layout
    const uint q_stride_b = num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;

    const uint k_stride_b = num_heads_k * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;

    // Base pointers for this query row and K/V head
    const uint q_offset = b * q_stride_b + head_q * q_stride_h + q_row * q_stride_s;
    const uint kv_offset = b * k_stride_b + head_k * k_stride_h;

    // ---------------------------------------------------------------------------
    // Load Q row into registers (each lane loads head_dim/32 elements)
    // For head_dim=128: 4 elements per lane. For head_dim=64: 2 per lane.
    // ---------------------------------------------------------------------------
    float q_reg[HEAD_DIM_MAX / THREADS_PER_ROW];
    const uint elems_per_lane = head_dim / THREADS_PER_ROW;
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        q_reg[i] = (d < head_dim) ? float(Q[q_offset + d]) : 0.0f;
    }

    // ---------------------------------------------------------------------------
    // Shared memory for K and V tiles (double-buffered)
    // Layout: [2][TILE_KV][HEAD_DIM_MAX]
    // ---------------------------------------------------------------------------
    threadgroup half K_tile[2][TILE_KV][HEAD_DIM_MAX];
    threadgroup half V_tile[2][TILE_KV][HEAD_DIM_MAX];

    // Online softmax state
    float m_prev = -INFINITY;  // Running max of scores
    float l_prev = 0.0f;       // Running sum of exp(score - m)

    // Output accumulator (unnormalized)
    float o_acc[HEAD_DIM_MAX / THREADS_PER_ROW];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o_acc[i] = 0.0f;
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
                    K_tile[0][kv_row][kv_col] = K[kv_offset + src_row * k_stride_s + kv_col];
                    V_tile[0][kv_row][kv_col] = V[kv_offset + src_row * k_stride_s + kv_col];
                } else {
                    K_tile[0][kv_row][kv_col] = half(0);
                    V_tile[0][kv_row][kv_col] = half(0);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---------------------------------------------------------------------------
    // Main loop: stream through K/V tiles with double-buffering
    // ---------------------------------------------------------------------------
    uint buf_compute = 0;

    for (uint tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
        uint buf_load = 1 - buf_compute;

        // Async load next tile (if exists) into buf_load
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
                        K_tile[buf_load][kv_row][kv_col] = K[kv_offset + src_row * k_stride_s + kv_col];
                        V_tile[buf_load][kv_row][kv_col] = V[kv_offset + src_row * k_stride_s + kv_col];
                    } else {
                        K_tile[buf_load][kv_row][kv_col] = half(0);
                        V_tile[buf_load][kv_row][kv_col] = half(0);
                    }
                }
            }
        }

        // Compute attention scores for this tile: Q[row] @ K_tile^T
        // Each lane computes partial dot products, then reduces across simdgroup
        uint tile_start = tile_idx * TILE_KV;
        uint tile_end = min(tile_start + TILE_KV, seq_k);
        uint tile_len = tile_end - tile_start;

        // Compute all scores in this tile
        float scores[TILE_KV];
        for (uint ki = 0; ki < tile_len; ++ki) {
            // Partial dot product: each lane handles its slice of head_dim
            float dot = 0.0f;
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                dot += q_reg[i] * float(K_tile[buf_compute][ki][d]);
            }
            // Reduce across lanes to get full dot product
            dot = simd_sum_f32(dot);
            scores[ki] = dot * scale;
        }
        // Zero out invalid positions
        for (uint ki = tile_len; ki < TILE_KV; ++ki) {
            scores[ki] = -INFINITY;
        }

        // Online softmax update
        // Find max of current tile scores
        float m_tile = -INFINITY;
        for (uint ki = 0; ki < tile_len; ++ki) {
            m_tile = max(m_tile, scores[ki]);
        }

        float m_new = max(m_prev, m_tile);
        float correction = exp(m_prev - m_new);

        // Update running sum: rescale previous sum + add new exponentials
        float l_new = l_prev * correction;
        for (uint ki = 0; ki < tile_len; ++ki) {
            l_new += exp(scores[ki] - m_new);
        }

        // Update output accumulator:
        //   O_acc = O_acc * correction + sum_i(exp(scores[i] - m_new) * V_tile[i])
        // Rescale previous accumulator
        for (uint i = 0; i < elems_per_lane; ++i) {
            o_acc[i] *= correction;
        }

        // Accumulate weighted V rows
        for (uint ki = 0; ki < tile_len; ++ki) {
            float p = exp(scores[ki] - m_new);
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                o_acc[i] += p * float(V_tile[buf_compute][ki][d]);
            }
        }

        m_prev = m_new;
        l_prev = l_new;

        // Wait for next tile load to finish before swapping buffers
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // ---------------------------------------------------------------------------
    // Normalize and store output
    // ---------------------------------------------------------------------------
    const uint o_offset = b * q_stride_b + head_q * q_stride_h + q_row * q_stride_s;
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;

    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        if (d < head_dim) {
            O[o_offset + d] = half(o_acc[i] * inv_l);
        }
    }
}

// ---------------------------------------------------------------------------
// Flash Attention kernel - causal masking
//
// Identical to flash_attention but applies causal mask:
//   score[q_pos][k_pos] = -INF if k_pos > q_pos
//
// This enables early termination: once tile_start > q_row, all scores are -INF.
// ---------------------------------------------------------------------------

kernel void flash_attention_causal(
    device const half* Q            [[buffer(0)]],
    device const half* K            [[buffer(1)]],
    device const half* V            [[buffer(2)]],
    device half* O                  [[buffer(3)]],
    constant uint& batch            [[buffer(4)]],
    constant uint& num_heads_q      [[buffer(5)]],
    constant uint& num_heads_k      [[buffer(6)]],
    constant uint& seq_q            [[buffer(7)]],
    constant uint& seq_k            [[buffer(8)]],
    constant uint& head_dim         [[buffer(9)]],
    constant float& scale           [[buffer(10)]],
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

    const uint q_stride_b = num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;
    const uint k_stride_b = num_heads_k * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;

    const uint q_offset = b * q_stride_b + head_q * q_stride_h + q_row * q_stride_s;
    const uint kv_offset = b * k_stride_b + head_k * k_stride_h;

    const uint elems_per_lane = head_dim / THREADS_PER_ROW;
    float q_reg[HEAD_DIM_MAX / THREADS_PER_ROW];
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        q_reg[i] = (d < head_dim) ? float(Q[q_offset + d]) : 0.0f;
    }

    threadgroup half K_tile[2][TILE_KV][HEAD_DIM_MAX];
    threadgroup half V_tile[2][TILE_KV][HEAD_DIM_MAX];

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o_acc[HEAD_DIM_MAX / THREADS_PER_ROW];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o_acc[i] = 0.0f;
    }

    // Causal early termination: only iterate tiles where k_pos <= q_row
    // For the causal case with seq_q == seq_k, q_row determines the upper bound
    // For cross-attention (seq_q != seq_k), we use q_row mapped to k-space
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
                    K_tile[0][kv_row][kv_col] = K[kv_offset + kv_row * k_stride_s + kv_col];
                    V_tile[0][kv_row][kv_col] = V[kv_offset + kv_row * k_stride_s + kv_col];
                } else {
                    K_tile[0][kv_row][kv_col] = half(0);
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
                        K_tile[buf_load][kv_row][kv_col] = K[kv_offset + src_row * k_stride_s + kv_col];
                        V_tile[buf_load][kv_row][kv_col] = V[kv_offset + src_row * k_stride_s + kv_col];
                    } else {
                        K_tile[buf_load][kv_row][kv_col] = half(0);
                        V_tile[buf_load][kv_row][kv_col] = half(0);
                    }
                }
            }
        }

        // Compute scores with causal mask
        uint tile_end = min(tile_start + TILE_KV, seq_k);
        uint tile_len = tile_end - tile_start;

        float scores[TILE_KV];
        for (uint ki = 0; ki < tile_len; ++ki) {
            uint k_pos = tile_start + ki;
            if (k_pos > q_row) {
                // Causal mask: future positions get -INF
                scores[ki] = -INFINITY;
                continue;
            }
            float dot = 0.0f;
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                dot += q_reg[i] * float(K_tile[buf_compute][ki][d]);
            }
            dot = simd_sum_f32(dot);
            scores[ki] = dot * scale;
        }
        for (uint ki = tile_len; ki < TILE_KV; ++ki) {
            scores[ki] = -INFINITY;
        }

        // Online softmax
        float m_tile = -INFINITY;
        for (uint ki = 0; ki < tile_len; ++ki) {
            m_tile = max(m_tile, scores[ki]);
        }

        float m_new = max(m_prev, m_tile);
        float correction = exp(m_prev - m_new);
        float l_new = l_prev * correction;
        for (uint ki = 0; ki < tile_len; ++ki) {
            l_new += exp(scores[ki] - m_new);
        }

        for (uint i = 0; i < elems_per_lane; ++i) {
            o_acc[i] *= correction;
        }
        for (uint ki = 0; ki < tile_len; ++ki) {
            float p = exp(scores[ki] - m_new);
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                o_acc[i] += p * float(V_tile[buf_compute][ki][d]);
            }
        }

        m_prev = m_new;
        l_prev = l_new;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // Store
    const uint o_offset = b * q_stride_b + head_q * q_stride_h + q_row * q_stride_s;
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        if (d < head_dim) {
            O[o_offset + d] = half(o_acc[i] * inv_l);
        }
    }
}

// ---------------------------------------------------------------------------
// Flash Attention kernel - Grouped-Query Attention (GQA)
//
// Identical memory layout but num_heads_k < num_heads_q.
// Multiple Q heads share the same K/V head. The head mapping is:
//   head_k = head_q / (num_heads_q / num_heads_k)
//
// This kernel is functionally identical to flash_attention (which already
// handles GQA via the head mapping), but uses an explicit ratio parameter
// for clarity and avoids the division in the inner loop.
// ---------------------------------------------------------------------------

kernel void flash_attention_gqa(
    device const half* Q            [[buffer(0)]],
    device const half* K            [[buffer(1)]],
    device const half* V            [[buffer(2)]],
    device half* O                  [[buffer(3)]],
    constant uint& batch            [[buffer(4)]],
    constant uint& num_heads_q      [[buffer(5)]],
    constant uint& num_heads_k      [[buffer(6)]],
    constant uint& seq_q            [[buffer(7)]],
    constant uint& seq_k            [[buffer(8)]],
    constant uint& head_dim         [[buffer(9)]],
    constant float& scale           [[buffer(10)]],
    constant uint& gqa_ratio        [[buffer(11)]],  // num_heads_q / num_heads_k
    constant bool& is_causal        [[buffer(12)]],
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

    const uint q_stride_b = num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;
    const uint k_stride_b = num_heads_k * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;

    const uint q_offset = b * q_stride_b + head_q * q_stride_h + q_row * q_stride_s;
    const uint kv_offset = b * k_stride_b + head_k * k_stride_h;

    const uint elems_per_lane = head_dim / THREADS_PER_ROW;
    float q_reg[HEAD_DIM_MAX / THREADS_PER_ROW];
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        q_reg[i] = (d < head_dim) ? float(Q[q_offset + d]) : 0.0f;
    }

    threadgroup half K_tile[2][TILE_KV][HEAD_DIM_MAX];
    threadgroup half V_tile[2][TILE_KV][HEAD_DIM_MAX];

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o_acc[HEAD_DIM_MAX / THREADS_PER_ROW];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o_acc[i] = 0.0f;
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
                    K_tile[0][kv_row][kv_col] = K[kv_offset + kv_row * k_stride_s + kv_col];
                    V_tile[0][kv_row][kv_col] = V[kv_offset + kv_row * k_stride_s + kv_col];
                } else {
                    K_tile[0][kv_row][kv_col] = half(0);
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
                        K_tile[buf_load][kv_row][kv_col] = K[kv_offset + src_row * k_stride_s + kv_col];
                        V_tile[buf_load][kv_row][kv_col] = V[kv_offset + src_row * k_stride_s + kv_col];
                    } else {
                        K_tile[buf_load][kv_row][kv_col] = half(0);
                        V_tile[buf_load][kv_row][kv_col] = half(0);
                    }
                }
            }
        }

        uint tile_end = min(tile_start + TILE_KV, seq_k);
        uint tile_len = tile_end - tile_start;

        float scores[TILE_KV];
        for (uint ki = 0; ki < tile_len; ++ki) {
            uint k_pos = tile_start + ki;
            if (is_causal && k_pos > q_row) {
                scores[ki] = -INFINITY;
                continue;
            }
            float dot = 0.0f;
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                dot += q_reg[i] * float(K_tile[buf_compute][ki][d]);
            }
            dot = simd_sum_f32(dot);
            scores[ki] = dot * scale;
        }
        for (uint ki = tile_len; ki < TILE_KV; ++ki) {
            scores[ki] = -INFINITY;
        }

        float m_tile = -INFINITY;
        for (uint ki = 0; ki < tile_len; ++ki) {
            m_tile = max(m_tile, scores[ki]);
        }

        float m_new = max(m_prev, m_tile);
        float correction = exp(m_prev - m_new);
        float l_new = l_prev * correction;
        for (uint ki = 0; ki < tile_len; ++ki) {
            l_new += exp(scores[ki] - m_new);
        }

        for (uint i = 0; i < elems_per_lane; ++i) {
            o_acc[i] *= correction;
        }
        for (uint ki = 0; ki < tile_len; ++ki) {
            float p = exp(scores[ki] - m_new);
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                o_acc[i] += p * float(V_tile[buf_compute][ki][d]);
            }
        }

        m_prev = m_new;
        l_prev = l_new;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    const uint o_offset = b * q_stride_b + head_q * q_stride_h + q_row * q_stride_s;
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        if (d < head_dim) {
            O[o_offset + d] = half(o_acc[i] * inv_l);
        }
    }
}

// ---------------------------------------------------------------------------
// Flash Attention kernel - FP4 KV cache
//
// Layout:
//   K_packed/V_packed: [batch, num_heads_k, seq_k, head_dim/8] (packed 8 values)
//   K_scales/V_scales: [batch, num_heads_k, seq_k] (per-row scale)
// ---------------------------------------------------------------------------

kernel void flash_attention_kv_fp4(
    device const half* Q            [[buffer(0)]],
    device const uint* K_packed     [[buffer(1)]],
    device const uint* V_packed     [[buffer(2)]],
    device const half* K_scales     [[buffer(3)]],
    device const half* V_scales     [[buffer(4)]],
    device half* O                  [[buffer(5)]],
    constant uint& batch            [[buffer(6)]],
    constant uint& num_heads_q      [[buffer(7)]],
    constant uint& num_heads_k      [[buffer(8)]],
    constant uint& seq_q            [[buffer(9)]],
    constant uint& seq_k            [[buffer(10)]],
    constant uint& head_dim         [[buffer(11)]],
    constant float& scale           [[buffer(12)]],
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

    const uint q_stride_b = num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;

    const uint packed_head_dim = (head_dim + FP4_PER_UINT - 1) / FP4_PER_UINT;
    const uint k_packed_stride_b = num_heads_k * seq_k * packed_head_dim;
    const uint k_packed_stride_h = seq_k * packed_head_dim;

    const uint k_scale_stride_b = num_heads_k * seq_k;
    const uint k_scale_stride_h = seq_k;

    const uint q_offset = b * q_stride_b + head_q * q_stride_h + q_row * q_stride_s;
    const uint kv_packed_offset = b * k_packed_stride_b + head_k * k_packed_stride_h;
    const uint k_scale_offset = b * k_scale_stride_b + head_k * k_scale_stride_h;
    const uint v_scale_offset = k_scale_offset;

    const uint elems_per_lane = head_dim / THREADS_PER_ROW;
    float q_reg[HEAD_DIM_MAX / THREADS_PER_ROW];
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        q_reg[i] = (d < head_dim) ? float(Q[q_offset + d]) : 0.0f;
    }

    threadgroup half K_tile[2][TILE_KV][HEAD_DIM_MAX];
    threadgroup half V_tile[2][TILE_KV][HEAD_DIM_MAX];

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o_acc[HEAD_DIM_MAX / THREADS_PER_ROW];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o_acc[i] = 0.0f;
    }

    const uint num_kv_tiles = (seq_k + TILE_KV - 1) / TILE_KV;

    load_kv_fp4_tile(K_packed, K_scales, K_tile[0],
                     head_dim, seq_k, packed_head_dim,
                     kv_packed_offset, k_scale_offset, 0, tid_in_tg);
    load_kv_fp4_tile(V_packed, V_scales, V_tile[0],
                     head_dim, seq_k, packed_head_dim,
                     kv_packed_offset, v_scale_offset, 0, tid_in_tg);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf_compute = 0;
    for (uint tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
        uint buf_load = 1 - buf_compute;
        uint next_tile_start = (tile_idx + 1) * TILE_KV;
        if (tile_idx + 1 < num_kv_tiles) {
            load_kv_fp4_tile(K_packed, K_scales, K_tile[buf_load],
                             head_dim, seq_k, packed_head_dim,
                             kv_packed_offset, k_scale_offset, next_tile_start, tid_in_tg);
            load_kv_fp4_tile(V_packed, V_scales, V_tile[buf_load],
                             head_dim, seq_k, packed_head_dim,
                             kv_packed_offset, v_scale_offset, next_tile_start, tid_in_tg);
        }

        uint tile_start = tile_idx * TILE_KV;
        uint tile_end = min(tile_start + TILE_KV, seq_k);
        uint tile_len = tile_end - tile_start;

        float scores[TILE_KV];
        for (uint ki = 0; ki < tile_len; ++ki) {
            float dot = 0.0f;
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                dot += q_reg[i] * float(K_tile[buf_compute][ki][d]);
            }
            dot = simd_sum_f32(dot);
            scores[ki] = dot * scale;
        }
        for (uint ki = tile_len; ki < TILE_KV; ++ki) {
            scores[ki] = -INFINITY;
        }

        float m_tile = -INFINITY;
        for (uint ki = 0; ki < tile_len; ++ki) {
            m_tile = max(m_tile, scores[ki]);
        }

        float m_new = max(m_prev, m_tile);
        float correction = exp(m_prev - m_new);
        float l_new = l_prev * correction;
        for (uint ki = 0; ki < tile_len; ++ki) {
            l_new += exp(scores[ki] - m_new);
        }

        for (uint i = 0; i < elems_per_lane; ++i) {
            o_acc[i] *= correction;
        }
        for (uint ki = 0; ki < tile_len; ++ki) {
            float p = exp(scores[ki] - m_new);
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                o_acc[i] += p * float(V_tile[buf_compute][ki][d]);
            }
        }

        m_prev = m_new;
        l_prev = l_new;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    const uint o_offset = b * q_stride_b + head_q * q_stride_h + q_row * q_stride_s;
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        if (d < head_dim) {
            O[o_offset + d] = half(o_acc[i] * inv_l);
        }
    }
}

// ---------------------------------------------------------------------------
// Flash Attention kernel - INT4 KV cache (signed, per-row scale)
// ---------------------------------------------------------------------------

kernel void flash_attention_kv_int4(
    device const half* Q            [[buffer(0)]],
    device const uint* K_packed     [[buffer(1)]],
    device const uint* V_packed     [[buffer(2)]],
    device const half* K_scales     [[buffer(3)]],
    device const half* V_scales     [[buffer(4)]],
    device half* O                  [[buffer(5)]],
    constant uint& batch            [[buffer(6)]],
    constant uint& num_heads_q      [[buffer(7)]],
    constant uint& num_heads_k      [[buffer(8)]],
    constant uint& seq_q            [[buffer(9)]],
    constant uint& seq_k            [[buffer(10)]],
    constant uint& head_dim         [[buffer(11)]],
    constant float& scale           [[buffer(12)]],
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

    const uint q_stride_b = num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;

    const uint packed_head_dim = (head_dim + FP4_PER_UINT - 1) / FP4_PER_UINT;
    const uint k_packed_stride_b = num_heads_k * seq_k * packed_head_dim;
    const uint k_packed_stride_h = seq_k * packed_head_dim;

    const uint k_scale_stride_b = num_heads_k * seq_k;
    const uint k_scale_stride_h = seq_k;

    const uint q_offset = b * q_stride_b + head_q * q_stride_h + q_row * q_stride_s;
    const uint kv_packed_offset = b * k_packed_stride_b + head_k * k_packed_stride_h;
    const uint k_scale_offset = b * k_scale_stride_b + head_k * k_scale_stride_h;
    const uint v_scale_offset = k_scale_offset;

    const uint elems_per_lane = head_dim / THREADS_PER_ROW;
    float q_reg[HEAD_DIM_MAX / THREADS_PER_ROW];
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        q_reg[i] = (d < head_dim) ? float(Q[q_offset + d]) : 0.0f;
    }

    threadgroup half K_tile[2][TILE_KV][HEAD_DIM_MAX];
    threadgroup half V_tile[2][TILE_KV][HEAD_DIM_MAX];

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o_acc[HEAD_DIM_MAX / THREADS_PER_ROW];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o_acc[i] = 0.0f;
    }

    const uint num_kv_tiles = (seq_k + TILE_KV - 1) / TILE_KV;

    load_kv_int4_tile(K_packed, K_scales, K_tile[0],
                      head_dim, seq_k, packed_head_dim,
                      kv_packed_offset, k_scale_offset, 0, tid_in_tg);
    load_kv_int4_tile(V_packed, V_scales, V_tile[0],
                      head_dim, seq_k, packed_head_dim,
                      kv_packed_offset, v_scale_offset, 0, tid_in_tg);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf_compute = 0;
    for (uint tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
        uint buf_load = 1 - buf_compute;
        uint next_tile_start = (tile_idx + 1) * TILE_KV;
        if (tile_idx + 1 < num_kv_tiles) {
            load_kv_int4_tile(K_packed, K_scales, K_tile[buf_load],
                              head_dim, seq_k, packed_head_dim,
                              kv_packed_offset, k_scale_offset, next_tile_start, tid_in_tg);
            load_kv_int4_tile(V_packed, V_scales, V_tile[buf_load],
                              head_dim, seq_k, packed_head_dim,
                              kv_packed_offset, v_scale_offset, next_tile_start, tid_in_tg);
        }

        uint tile_start = tile_idx * TILE_KV;
        uint tile_end = min(tile_start + TILE_KV, seq_k);
        uint tile_len = tile_end - tile_start;

        float scores[TILE_KV];
        for (uint ki = 0; ki < tile_len; ++ki) {
            float dot = 0.0f;
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                dot += q_reg[i] * float(K_tile[buf_compute][ki][d]);
            }
            dot = simd_sum_f32(dot);
            scores[ki] = dot * scale;
        }
        for (uint ki = tile_len; ki < TILE_KV; ++ki) {
            scores[ki] = -INFINITY;
        }

        float m_tile = -INFINITY;
        for (uint ki = 0; ki < tile_len; ++ki) {
            m_tile = max(m_tile, scores[ki]);
        }

        float m_new = max(m_prev, m_tile);
        float correction = exp(m_prev - m_new);
        float l_new = l_prev * correction;
        for (uint ki = 0; ki < tile_len; ++ki) {
            l_new += exp(scores[ki] - m_new);
        }

        for (uint i = 0; i < elems_per_lane; ++i) {
            o_acc[i] *= correction;
        }
        for (uint ki = 0; ki < tile_len; ++ki) {
            float p = exp(scores[ki] - m_new);
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                o_acc[i] += p * float(V_tile[buf_compute][ki][d]);
            }
        }

        m_prev = m_new;
        l_prev = l_new;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    const uint o_offset = b * q_stride_b + head_q * q_stride_h + q_row * q_stride_s;
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        if (d < head_dim) {
            O[o_offset + d] = half(o_acc[i] * inv_l);
        }
    }
}
