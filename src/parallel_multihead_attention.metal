// parallel_multihead_attention.metal - Parallel multi-head attention
//
// Computes multiple attention heads in parallel with independent threadgroups.
//
// Thread configuration:
//   - X threads per head (THREADS_PER_HEAD)
//   - Y heads in parallel per threadgroup (HEADS_PER_TG)
//   - Threadgroups = num_heads (each threadgroup handles one head)
//
// Design:
//   Each threadgroup independently processes one attention head, processing
//   multiple query rows in parallel. This approach maximizes parallelism
//   across heads while maintaining efficiency within each head.
//
// Memory layout (all row-major):
//   Q: [batch, num_heads, seq_q, head_dim]
//   K: [batch, num_heads, seq_k, head_dim]
//   V: [batch, num_heads, seq_k, head_dim]
//   O: [batch, num_heads, seq_q, head_dim]
//
// Dispatch: [num_heads, 1, 1] threadgroups
// Threadgroup: [THREADS_PER_TG, 1, 1] where THREADS_PER_TG = HEADS_PER_TG * THREADS_PER_HEAD

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Thread configuration constants
//
// THREADS_PER_HEAD: Number of threads working on a single attention head
//   Each thread computes dot products for a subset of K vectors per tile
//   32 threads = 1 simdgroup, good for reductions
//
// HEADS_PER_TG: Number of heads processed in parallel within one threadgroup
//   Each threadgroup processes HEADS_PER_TG attention heads independently
//   4 heads per TG balances threadgroup memory usage
//
// TILE_KV: K/V tile size (number of K/V vectors loaded per iteration)
//   64 balances threadgroup memory and loop overhead
//
// HEAD_DIM_MAX: Maximum head dimension supported
// ---------------------------------------------------------------------------

constant constexpr uint THREADS_PER_HEAD = 32;
constant constexpr uint HEADS_PER_TG = 4;
constant constexpr uint THREADS_PER_TG = THREADS_PER_HEAD * HEADS_PER_TG;
constant constexpr uint TILE_KV_PARALLEL = 64;
constant constexpr uint HEAD_DIM_MAX_PARALLEL = 128;

// ---------------------------------------------------------------------------
// Utility functions using hardware-accelerated simd reductions
//
// Metal's built-in simd_sum/simd_max are significantly faster than manual
// simd_shuffle_xor chains: single instruction vs 5 dependent instructions.
// ---------------------------------------------------------------------------

inline float simd_max_parallel(float val, uint lane_id [[thread_index_in_simdgroup]]) {
    (void)lane_id;  // Unused with hardware intrinsic
    return simd_max(val);
}

inline float simd_sum_parallel(float val) {
    return simd_sum(val);
}

// ---------------------------------------------------------------------------
// Main parallel multi-head attention kernel
//
// Each threadgroup processes one attention head independently.
// Within the threadgroup, threads are partitioned into HEADS_PER_TG groups,
// each group processing HEADS_PER_TG query rows.
//
// Thread mapping:
//   tgid.x = head index (0..num_heads-1)
//   tgid.y = batch index (0..batch-1)
//
// Within threadgroup:
//   head_group = tid / THREADS_PER_HEAD (which of the HEADS_PER_TG query rows)
//   thread_in_group = tid % THREADS_PER_HEAD (which thread in the simdgroup)
//
// Each head_group processes one query row of the attention head.
// All threads in the group work together to compute the attention for that row.
// ---------------------------------------------------------------------------

kernel void parallel_multihead_attention(
    device const half* Q            [[buffer(0)]],
    device const half* K            [[buffer(1)]],
    device const half* V            [[buffer(2)]],
    device half* O                  [[buffer(3)]],
    constant uint& batch            [[buffer(4)]],
    constant uint& num_heads        [[buffer(5)]],
    constant uint& seq_q            [[buffer(6)]],
    constant uint& seq_k            [[buffer(7)]],
    constant uint& head_dim         [[buffer(8)]],
    constant float& scale           [[buffer(9)]],
    constant uint& causal           [[buffer(10)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint lane_id                   [[thread_index_in_simdgroup]]
) {
    // Threadgroup identifies which head we're processing
    const uint head = tgid.x;
    const uint b = tgid.y;
    
    if (head >= num_heads || b >= batch) return;
    
    // Partition threads within threadgroup
    // Each partition (head_group) processes one query row
    const uint head_group = tid / THREADS_PER_HEAD;
    const uint thread_in_group = tid % THREADS_PER_HEAD;
    
    // Query row for this group: process rows in blocks of HEADS_PER_TG
    // First threadgroup (tgid.x=0) processes rows 0-3, second (tgid.x=1) processes rows 4-7, etc.
    const uint q_row_base = head * HEADS_PER_TG;
    const uint q_row = q_row_base + head_group;
    
    if (q_row >= seq_q) return;
    
    // Strides for [batch, heads, seq, dim] layout
    const uint q_stride_b = num_heads * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;
    
    const uint k_stride_b = num_heads * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;
    
    // Base pointers for this head
    const uint q_offset = b * q_stride_b + head * q_stride_h;
    const uint kv_offset = b * k_stride_b + head * k_stride_h;
    const uint o_offset = b * q_stride_b + head * q_stride_h;
    
    // Load Q vector into registers for this query row
    const uint elems_per_lane = head_dim / THREADS_PER_HEAD;
    float q_reg[HEAD_DIM_MAX_PARALLEL / THREADS_PER_HEAD];
    
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        q_reg[i] = (d < head_dim) ? float(Q[q_offset + q_row * q_stride_s + d]) : 0.0f;
    }
    
    // Threadgroup memory for K and V tiles (double-buffered)
    // Shared by all head_groups in this threadgroup
    threadgroup half K_tile[2][TILE_KV_PARALLEL][HEAD_DIM_MAX_PARALLEL];
    threadgroup half V_tile[2][TILE_KV_PARALLEL][HEAD_DIM_MAX_PARALLEL];
    
    // Online softmax state for this query row
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    
    // Output accumulator for this query row
    float o_acc[HEAD_DIM_MAX_PARALLEL / THREADS_PER_HEAD];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o_acc[i] = 0.0f;
    }
    
    const uint num_kv_tiles = (seq_k + TILE_KV_PARALLEL - 1) / TILE_KV_PARALLEL;
    
    // Preload first K/V tile into buffer 0
    {
        const uint tile_start = 0;
        const uint elems_to_load = TILE_KV_PARALLEL * head_dim;
        const uint loads_per_thread = (elems_to_load + THREADS_PER_TG - 1) / THREADS_PER_TG;
        
        for (uint i = 0; i < loads_per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG;
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
        uint next_tile_start = (tile_idx + 1) * TILE_KV_PARALLEL;
        
        // Async load next tile
        if (tile_idx + 1 < num_kv_tiles) {
            const uint elems_to_load = TILE_KV_PARALLEL * head_dim;
            const uint loads_per_thread = (elems_to_load + THREADS_PER_TG - 1) / THREADS_PER_TG;
            
            for (uint i = 0; i < loads_per_thread; ++i) {
                uint idx = tid + i * THREADS_PER_TG;
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
        
        // Compute attention scores for this tile
        uint tile_start = tile_idx * TILE_KV_PARALLEL;
        uint tile_end = min(tile_start + TILE_KV_PARALLEL, seq_k);
        uint tile_len = tile_end - tile_start;
        
        float scores[TILE_KV_PARALLEL];
        
        // Compute scores: only threads in this head_group participate
        for (uint ki = 0; ki < tile_len; ++ki) {
            uint k_pos = tile_start + ki;
            
            // Causal mask
            if (causal && k_pos > q_row) {
                scores[ki] = -INFINITY;
                continue;
            }
            
            // Dot product: each lane computes its slice, then reduce
            float dot = 0.0f;
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                dot += q_reg[i] * float(K_tile[buf_compute][ki][d]);
            }
            dot = simd_sum_parallel(dot);
            scores[ki] = dot * scale;
        }
        
        for (uint ki = tile_len; ki < TILE_KV_PARALLEL; ++ki) {
            scores[ki] = -INFINITY;
        }
        
        // Online softmax: find max in this tile
        float m_tile = -INFINITY;
        for (uint ki = 0; ki < tile_len; ++ki) {
            m_tile = max(m_tile, scores[ki]);
        }
        
        float m_new = max(m_prev, m_tile);
        float correction = exp(m_prev - m_new);
        
        // Update running sum
        float l_new = l_prev * correction;
        for (uint ki = 0; ki < tile_len; ++ki) {
            l_new += exp(scores[ki] - m_new);
        }
        
        // Rescale accumulator
        for (uint i = 0; i < elems_per_lane; ++i) {
            o_acc[i] *= correction;
        }
        
        // Accumulate weighted V values
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
    
    // Normalize and store output
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        if (d < head_dim) {
            O[o_offset + q_row * q_stride_s + d] = half(o_acc[i] * inv_l);
        }
    }
}

// ---------------------------------------------------------------------------
// Variant: Parallel multi-head attention with multiple query rows per threadgroup
//
// This version allows each threadgroup to process HEADS_PER_TG query rows
// from different attention heads simultaneously.
//
// Grid dispatch: [num_heads, 1, batch]
//   Threadgroup processes HEADS_PER_TG query rows from a single attention head
// ---------------------------------------------------------------------------

kernel void parallel_multihead_attention_multirow(
    device const half* Q            [[buffer(0)]],
    device const half* K            [[buffer(1)]],
    device const half* V            [[buffer(2)]],
    device half* O                  [[buffer(3)]],
    constant uint& batch            [[buffer(4)]],
    constant uint& num_heads        [[buffer(5)]],
    constant uint& seq_q            [[buffer(6)]],
    constant uint& seq_k            [[buffer(7)]],
    constant uint& head_dim         [[buffer(8)]],
    constant float& scale           [[buffer(9)]],
    constant uint& causal           [[buffer(10)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint lane_id                   [[thread_index_in_simdgroup]]
) {
    // Each threadgroup processes one attention head
    const uint head = tgid.x;
    const uint b = tgid.z;
    
    if (head >= num_heads || b >= batch) return;
    
    // Each simdgroup (THREADS_PER_HEAD threads) processes one query row
    const uint simd_id = tid / THREADS_PER_HEAD;
    const uint q_row_base = tgid.y * HEADS_PER_TG;
    const uint q_row = q_row_base + simd_id;
    
    if (q_row >= seq_q) return;
    
    const uint q_stride_b = num_heads * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;
    const uint k_stride_b = num_heads * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;
    
    const uint q_offset = b * q_stride_b + head * q_stride_h;
    const uint kv_offset = b * k_stride_b + head * k_stride_h;
    const uint o_offset = b * q_stride_b + head * q_stride_h;
    
    // Load Q vector for this query row
    const uint elems_per_lane = head_dim / THREADS_PER_HEAD;
    float q_reg[HEAD_DIM_MAX_PARALLEL / THREADS_PER_HEAD];
    
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        q_reg[i] = (d < head_dim) ? float(Q[q_offset + q_row * q_stride_s + d]) : 0.0f;
    }
    
    threadgroup half K_tile[2][TILE_KV_PARALLEL][HEAD_DIM_MAX_PARALLEL];
    threadgroup half V_tile[2][TILE_KV_PARALLEL][HEAD_DIM_MAX_PARALLEL];
    
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o_acc[HEAD_DIM_MAX_PARALLEL / THREADS_PER_HEAD];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o_acc[i] = 0.0f;
    }
    
    const uint num_kv_tiles = (seq_k + TILE_KV_PARALLEL - 1) / TILE_KV_PARALLEL;
    
    // Preload first K/V tile
    {
        const uint elems_to_load = TILE_KV_PARALLEL * head_dim;
        const uint loads_per_thread = (elems_to_load + THREADS_PER_TG - 1) / THREADS_PER_TG;
        
        for (uint i = 0; i < loads_per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG;
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
        uint next_start = (tile_idx + 1) * TILE_KV_PARALLEL;
        
        // Load next tile
        if (tile_idx + 1 < num_kv_tiles) {
            const uint elems_to_load = TILE_KV_PARALLEL * head_dim;
            const uint loads_per_thread = (elems_to_load + THREADS_PER_TG - 1) / THREADS_PER_TG;
            
            for (uint i = 0; i < loads_per_thread; ++i) {
                uint idx = tid + i * THREADS_PER_TG;
                if (idx < elems_to_load) {
                    uint kv_row = idx / head_dim;
                    uint kv_col = idx % head_dim;
                    uint src_row = next_start + kv_row;
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
        
        uint tile_start = tile_idx * TILE_KV_PARALLEL;
        uint tile_end = min(tile_start + TILE_KV_PARALLEL, seq_k);
        uint tile_len = tile_end - tile_start;
        
        float scores[TILE_KV_PARALLEL];
        
        for (uint ki = 0; ki < tile_len; ++ki) {
            uint k_pos = tile_start + ki;
            
            if (causal && k_pos > q_row) {
                scores[ki] = -INFINITY;
                continue;
            }
            
            float dot = 0.0f;
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                dot += q_reg[i] * float(K_tile[buf_compute][ki][d]);
            }
            dot = simd_sum_parallel(dot);
            scores[ki] = dot * scale;
        }
        
        for (uint ki = tile_len; ki < TILE_KV_PARALLEL; ++ki) {
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
    
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        if (d < head_dim) {
            O[o_offset + q_row * q_stride_s + d] = half(o_acc[i] * inv_l);
        }
    }
}
