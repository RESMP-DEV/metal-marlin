// flash_attention_v3.metal - Flash Attention V3 with Optimized Causal Masking
//
// Improvements over V2:
//   1. Causal mask applied at both tile and element levels for optimal performance
//   2. Early tile skip for tiles fully beyond causal limit
//   3. Branchless mask computation fused into dot product
//   4. Improved register usage for head_dim=64/128
//
// Causal Mask Strategy:
//   - Compute causal_limit for each query position: q_pos + 1
//   - For threadgroups processing multiple queries, use max causal limit
//   - Skip tiles where tile_start >= causal_limit (no valid positions)
//   - For tiles crossing the boundary, mask elements where k_pos >= q_pos
//
// Memory layout (all row-major, contiguous):
//   Q: [batch, heads_q, seq_q, head_dim]
//   K: [batch, heads_kv, seq_k, head_dim]
//   V: [batch, heads_kv, seq_k, head_dim]
//   O: [batch, heads_q, seq_q, head_dim]

#include <metal_stdlib>
#include "bf16_compat.metal"
using namespace metal;

#ifdef USE_BF16_INPUTS
using input_t = bf16_t;
using output_t = ushort;
#else
using input_t = half;
using output_t = half;
#endif

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

constant constexpr uint TILE_Q = 16;          // Query rows per threadgroup
constant constexpr uint TILE_KV = 24;         // K/V rows per tile (32KB limit)
constant constexpr uint CHUNK_KV = 512;      // Chunk size for long sequences (>4K)
constant constexpr uint HEAD_DIM_64 = 64;
constant constexpr uint HEAD_DIM_128 = 128;

// Padded dimensions for bank conflict avoidance
// Apple Silicon shared memory has 32 banks (128 bytes each for bf16)
// Padding breaks bank alignment when threads access with stride SIMD_SIZE (32)
// Without padding: thread 0 accesses 0,32,64,96 -> bank 0, causing 4-way conflicts
// With +8 padding: thread 0 accesses 0,40,80,120 -> banks 0,1,2,3, no conflicts
constant constexpr uint HEAD_DIM_PADDED = 144;  // 128 + 16 padding (breaks 32 stride)
constant constexpr uint TILE_KV_PADDED = 26;      // 24 + 2 padding (adjusts for memory budget)

constant constexpr uint SIMD_SIZE = 32;
constant constexpr uint NUM_SIMDGROUPS = 4;
constant constexpr uint THREADS_PER_TG = SIMD_SIZE * NUM_SIMDGROUPS;

// ---------------------------------------------------------------------------
// Conversion utilities
// ---------------------------------------------------------------------------

inline float input_to_float(input_t v) {
#ifdef USE_BF16_INPUTS
    return bf16_to_float(v);
#else
    return float(v);
#endif
}

inline void store_output_scalar(device output_t* dst, uint idx, float val) {
#ifdef USE_BF16_INPUTS
    dst[idx] = bf16_from_float_rne(val).bits;
#else
    dst[idx] = half(val);
#endif
}

#ifdef USE_BF16_INPUTS
inline void store_output_bf16_vectorized(device ushort* dst,
                                         uint base,
                                         uint lane_id,
                                         uint elems_per_lane,
                                         thread const float* vals,
                                         uint head_dim) {
    if (elems_per_lane == 4 && ((lane_id & 1u) == 0u)) {
        uint d0 = lane_id * elems_per_lane;
        float4 lo = float4(vals[0], vals[1], vals[2], vals[3]);
        float4 hi = float4(simd_shuffle_xor(vals[0], 1),
                           simd_shuffle_xor(vals[1], 1),
                           simd_shuffle_xor(vals[2], 1),
                           simd_shuffle_xor(vals[3], 1));
        if (d0 + 7 < head_dim) {
            ushort4 lo_packed = float4_to_bf16x4_rne(lo);
            ushort4 hi_packed = float4_to_bf16x4_rne(hi);
            *reinterpret_cast<device ushort4*>(dst + base + d0) = lo_packed;
            *reinterpret_cast<device ushort4*>(dst + base + d0 + 4) = hi_packed;
            return;
        }
    }

    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        if (d < head_dim) {
            store_output_scalar(dst, base + d, vals[i]);
        }
    }
}
#endif

// ---------------------------------------------------------------------------
// AttentionParams
// ---------------------------------------------------------------------------

struct AttentionParams {
    uint batch;
    uint num_heads_q;
    uint num_heads_kv;
    uint seq_q;
    uint seq_k;
    uint head_dim;
    float scale;
    uint gqa_ratio;
    uint is_causal;
};

// ---------------------------------------------------------------------------
// Causal Mask Functions
// ---------------------------------------------------------------------------

// Branchless causal mask: returns -INFINITY if k_pos > q_pos, else score
inline float apply_causal_mask(float score, uint q_pos, uint k_pos) {
    // select(a, b, cond) returns a if cond is false, b if true
    // We want -INFINITY when k_pos > q_pos
    return select(score, -INFINITY, k_pos > q_pos);
}

// ---------------------------------------------------------------------------
// Online Softmax Partial Sum Accumulation for Chunked Attention
// ---------------------------------------------------------------------------

// Partial softmax state for a chunk/tile
struct SoftmaxPartial {
    float m;      // Max value in this partial
    float l;      // Sum of exp(scores - m) in this partial
};

// Merge two partial softmax states into a new combined state
inline SoftmaxPartial merge_softmax_partial(SoftmaxPartial a, SoftmaxPartial b) {
    SoftmaxPartial result;
    // New max is the maximum of both partial maxes
    result.m = max(a.m, b.m);

    // Merge the sum accumulations:
    // l_combined = exp(a.m - new_m) * a.l + exp(b.m - new_m) * b.l
    float exp_a_m = (a.m > -INFINITY) ? exp(a.m - result.m) : 0.0f;
    float exp_b_m = (b.m > -INFINITY) ? exp(b.m - result.m) : 0.0f;
    result.l = exp_a_m * a.l + exp_b_m * b.l;

    return result;
}

// Initialize a partial softmax state from a single score
inline SoftmaxPartial softmax_partial_from_score(float score) {
    SoftmaxPartial result;
    result.m = score;
    result.l = (score > -INFINITY) ? 1.0f : 0.0f;
    return result;
}

// Compute partial softmax statistics for a chunk of scores using SIMDgroup reductions
// Returns: (m, l) where m = max(scores) and l = sum(exp(scores - m))
// Optimized: each thread processes a stride of SIMD_SIZE elements, then reduces
inline SoftmaxPartial compute_partial_softmax(const thread float* scores,
                                                 uint start_idx,
                                                 uint count) {
    SoftmaxPartial result;

    // First pass: compute local max across thread's assigned elements
    // Each thread processes elements: lane_id, lane_id + SIMD_SIZE, lane_id + 2*SIMD_SIZE, ...
    float m_local = -INFINITY;
    for (uint i = 0; i < count; ++i) {
        m_local = max(m_local, scores[start_idx + i]);
    }

    // Reduce max across all threads in simdgroup
    result.m = simd_max(m_local);

    // Second pass: compute sum of exp(scores - m) using local partials
    // Each thread computes its contribution to the sum
    float l_local = 0.0f;
    for (uint i = 0; i < count; ++i) {
        l_local += exp(scores[start_idx + i] - result.m);
    }

    // Reduce sum across all threads in simdgroup
    result.l = simd_sum(l_local);

    return result;
}

// Compute softmax probabilities given partial statistics
// p[i] = exp(scores[i] - m) / l
inline void compute_softmax_probs(thread float* scores,
                                    thread float* probs,
                                    thread const SoftmaxPartial& stats,
                                    uint start_idx,
                                    uint count) {
    float inv_l = (stats.l > 0.0f) ? (1.0f / stats.l) : 0.0f;
    for (uint i = 0; i < count; ++i) {
        probs[start_idx + i] = exp(scores[start_idx + i] - stats.m) * inv_l;
    }
}

// ---------------------------------------------------------------------------
// Chunked K/V Accumulation State for Long Sequences
// ---------------------------------------------------------------------------

// State for accumulating attention across multiple tiles within a chunk
struct ChunkAccumulator {
    float m;              // Running max across the chunk
    float l;              // Running softmax sum across the chunk
    float o[HEAD_DIM_128];  // Running weighted value accumulation
};

// Initialize a chunk accumulator
inline ChunkAccumulator init_chunk_accumulator(uint head_dim) {
    ChunkAccumulator acc;
    acc.m = -INFINITY;
    acc.l = 0.0f;
    for (uint i = 0; i < HEAD_DIM_128; ++i) {
        acc.o[i] = 0.0f;
    }
    return acc;
}

// Update chunk accumulator with a new tile of scores
inline void update_chunk_accumulator(
    thread ChunkAccumulator& acc,
    thread const float* scores,
    thread const float* v_values,  // [TILE_KV * head_dim] accumulated V values
    uint tile_len,
    uint head_dim
) {
    // Find max in this tile
    float m_tile = -INFINITY;
    for (uint i = 0; i < tile_len; ++i) {
        m_tile = max(m_tile, scores[i]);
    }

    // Merge with running state
    float m_new = max(acc.m, m_tile);
    float corr_prev = exp(acc.m - m_new);
    float corr_tile = exp(m_tile - m_new);

    // Update sum and output accumulation
    acc.l = acc.l * corr_prev;
    for (uint i = 0; i < head_dim; ++i) {
        acc.o[i] *= corr_prev;
    }

    // Accumulate contributions from this tile
    float l_tile = 0.0f;
    for (uint ki = 0; ki < tile_len; ++ki) {
        float p = exp(scores[ki] - m_new);
        l_tile += p;
        for (uint d = 0; d < head_dim; ++d) {
            acc.o[d] += p * v_values[ki * head_dim + d];
        }
    }
    acc.l = acc.l + l_tile * corr_tile;
    acc.m = m_new;
}

// ---------------------------------------------------------------------------
// Flash Attention V3 - Causal (Optimized tile and element-level masking)
// ---------------------------------------------------------------------------

kernel void flash_attention_v3_causal(
    device const input_t* Q         [[buffer(0)]],
    device const input_t* K         [[buffer(1)]],
    device const input_t* V         [[buffer(2)]],
    device output_t* O              [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
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

    const uint head_kv = head_q / params.gqa_ratio;

    const uint q_start = q_tile_idx * TILE_Q;
    if (q_start >= seq_q) return;

    const uint q_rows = min(TILE_Q, seq_q - q_start);

    // Strides
    const uint q_stride_b = params.num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;
    const uint k_stride_b = params.num_heads_kv * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;

    const uint q_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;
    const uint kv_base = batch_idx * k_stride_b + head_kv * k_stride_h;

    // Threadgroup memory (use padded dimensions to avoid bank conflicts)
    // Apple Silicon has 32 banks, each 128 bytes for bf16 (64 elements)
    // Padding breaks bank alignment when threads access with stride SIMD_SIZE (32)
    // Without padding: thread 0 accesses 0,32,64,96 -> bank 0 (4-way conflict)
    // With +16 padding: thread 0 accesses 0,32,64,96 -> banks 0,1,2,3 (no conflict)
    threadgroup input_t Q_tile[TILE_Q][HEAD_DIM_PADDED];
    threadgroup input_t K_tile[2][TILE_KV][HEAD_DIM_PADDED];
    threadgroup input_t V_tile[2][TILE_KV][HEAD_DIM_PADDED];

    // Load Q tile
    {
        const uint elems = q_rows * head_dim;
        const uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;
        for (uint i = 0; i < per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG;
            if (idx < elems) {
                uint q_row = idx / head_dim;
                uint q_col = idx % head_dim;
                Q_tile[q_row][q_col] = Q[q_base + q_row * q_stride_s + q_col];
            }
        }
    }

    const uint rows_per_sg = TILE_Q / NUM_SIMDGROUPS;
    const uint sg_q_start = sg_id * rows_per_sg;
    const uint sg_q_rows = min(rows_per_sg, (q_rows > sg_q_start) ? (q_rows - sg_q_start) : 0u);

    const uint elems_per_lane = head_dim / SIMD_SIZE;
    float q_reg[4][HEAD_DIM_128 / SIMD_SIZE];
    float m_prev[4];
    float l_prev[4];
    float o_acc[4][HEAD_DIM_128 / SIMD_SIZE];

    for (uint r = 0; r < rows_per_sg; ++r) {
        m_prev[r] = -INFINITY;
        l_prev[r] = 0.0f;
        for (uint i = 0; i < elems_per_lane; ++i) o_acc[r][i] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = 0; r < sg_q_rows; ++r) {
        uint q_row = sg_q_start + r;
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            q_reg[r][i] = input_to_float(Q_tile[q_row][d]);
        }
    }

    // =========================================================================
    // CAUSAL MASK OPTIMIZATION - TILE LEVEL
    //
    // For each query row, compute its causal limit: q_pos + 1
    // For the threadgroup, use the max causal limit across all rows
    // This allows early skipping of tiles that have no valid positions
    // =========================================================================

    const uint max_q_pos_in_tg = q_start + q_rows - 1;
    const uint causal_limit_tg = min(max_q_pos_in_tg + 1, seq_k);
    
    // =========================================================================
    // CHUNKED QK CALCULATION FOR LONG SEQUENCES (>4K)
    //
    // For sequences >4K tokens, process K tiles in chunks of CHUNK_KV (512)
    // to reduce memory pressure and improve cache locality.
    // Each chunk maintains independent partial softmax statistics that are
    // merged into the running accumulator using online softmax updates.
    // =========================================================================
    
    const bool use_chunking = seq_k > 4096;
    const uint chunk_size = use_chunking ? CHUNK_KV : seq_k;
    const uint num_chunks = use_chunking ? ((causal_limit_tg + chunk_size - 1) / chunk_size) : 1;
    const uint tiles_per_chunk = use_chunking ? ((chunk_size + TILE_KV - 1) / TILE_KV) : ((causal_limit_tg + TILE_KV - 1) / TILE_KV);
    const uint num_kv_tiles = (causal_limit_tg + TILE_KV - 1) / TILE_KV;

    // Preload first tile
    {
        uint tile_len = min(uint(TILE_KV), seq_k);
        uint elems = tile_len * head_dim;
        uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;
        for (uint i = 0; i < per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG;
            if (idx < elems) {
                uint kv_row = idx / head_dim;
                uint kv_col = idx % head_dim;
                K_tile[0][kv_row][kv_col] = K[kv_base + kv_row * k_stride_s + kv_col];
                V_tile[0][kv_row][kv_col] = V[kv_base + kv_row * k_stride_s + kv_col];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf = 0;

    // =========================================================================
    // MAIN LOOP: Chunked K/V processing with causal mask
    //
    // Process K/V in chunks for long sequences:
    //   - Each chunk processes up to CHUNK_KV positions (512 tokens)
    //   - Within each chunk, process TILE_KV positions at a time (24 tokens)
    //   - Accumulate partial softmax statistics across tiles within chunk
    //   - Merge chunk results into running accumulator
    // =========================================================================

    uint global_tile_idx = 0;
    
    for (uint chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        uint chunk_start = chunk_idx * chunk_size;
        uint chunk_end = min(chunk_start + chunk_size, causal_limit_tg);
        
        // Process tiles within this chunk
        while (global_tile_idx * TILE_KV < chunk_end) {
            uint buf_load = 1 - buf;
            uint tile_start = global_tile_idx * TILE_KV;
            
            // Stop if we've moved past this chunk
            if (tile_start >= chunk_end) break;
            
            uint tile_len = min(uint(TILE_KV), chunk_end - tile_start);

        // Skip tiles that are fully masked for this threadgroup
        if (tile_len == 0 || tile_start >= chunk_end) {
            buf = buf_load;
            global_tile_idx++;
            continue;
        }

        // Load next tile (with bounds check for entire sequence)
        uint next_tile_start = (global_tile_idx + 1) * TILE_KV;
        if (next_tile_start < causal_limit_tg) {
            uint next_len = min(uint(TILE_KV), causal_limit_tg - next_tile_start);
            uint elems = next_len * head_dim;
            uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;
            for (uint i = 0; i < per_thread; ++i) {
                uint idx = tid + i * THREADS_PER_TG;
                if (idx < elems) {
                    uint kv_row = idx / head_dim;
                    uint kv_col = idx % head_dim;
                    K_tile[buf_load][kv_row][kv_col] = K[kv_base + (next_tile_start + kv_row) * k_stride_s + kv_col];
                    V_tile[buf_load][kv_row][kv_col] = V[kv_base + (next_tile_start + kv_row) * k_stride_s + kv_col];
                }
            }
        }

        // =====================================================================
        // CAUSAL MASK APPLICATION WITHIN TILED LOOP
        // =====================================================================
        //
        // For each query row in the simdgroup:
        //   1. Compute causal limit for this specific row: q_pos + 1
        //   2. For each key position in tile:
        //      - Compute dot product Q @ K^T
        //      - Apply scale
        //      - Apply causal mask: if k_pos > q_pos, score = -INFINITY
        //
        // The mask is applied per-element within the tile using branchless
        // select() operation, avoiding conditional branches.
        // =====================================================================

        for (uint r = 0; r < sg_q_rows; ++r) {
            uint global_q_pos = q_start + sg_q_start + r;
            uint causal_limit_row = min(global_q_pos + 1, seq_k);

            // Determine per-row tile start and end
            uint row_tile_start = max(tile_start, causal_limit_row);
            uint row_tile_end = min(tile_start + tile_len, causal_limit_row);
            uint row_tile_len = (row_tile_end > row_tile_start) ? (row_tile_end - row_tile_start) : 0u;

            float scores[TILE_KV];

            // Compute scores with causal mask
            for (uint ki = 0; ki < tile_len; ++ki) {
                uint k_pos = tile_start + ki;

                // Compute dot product
                float dot = 0.0f;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    dot += q_reg[r][i] * input_to_float(K_tile[buf][ki][d]);
                }
                dot = simd_sum(dot);

                // Scale and apply causal mask (branchless)
                // select(score, -INFINITY, k_pos > q_pos) masks future positions
                float score = dot * scale;
                scores[ki] = select(score, -INFINITY, k_pos > global_q_pos);
            }

            for (uint ki = tile_len; ki < TILE_KV; ++ki) {
                scores[ki] = -INFINITY;
            }

            // =====================================================================
            // ONLINE SOFTMAX WITH PARTIAL SUM ACCUMULATION
            // =====================================================================
            //
            // For chunked attention blocks, we compute partial softmax statistics
            // per tile and merge them into the running accumulator.
            //
            // The update formula:
            //   m_new = max(m_prev, m_tile)
            //   l_new = l_prev * exp(m_prev - m_new) + l_tile * exp(m_tile - m_new)
            //   o_new = o_prev * exp(m_prev - m_new) + o_tile * exp(m_tile - m_new)
            //
            // This is numerically stable and handles arbitrary chunk sizes.
            // =====================================================================

            // Compute partial statistics for this tile chunk
            SoftmaxPartial partial = compute_partial_softmax(scores, 0, tile_len);

            // Merge with running accumulator
            float m_new = max(m_prev[r], partial.m);
            float corr_prev = exp(m_prev[r] - m_new);
            float corr_tile = exp(partial.m - m_new);

            // Update sum and output accumulation
            l_prev[r] = l_prev[r] * corr_prev + partial.l * corr_tile;
            for (uint i = 0; i < elems_per_lane; ++i) {
                o_acc[r][i] = o_acc[r][i] * corr_prev;  // Scale existing output
            }

            // Accumulate weighted values using the tile's contribution
            // Use SIMDgroup-level reduction for faster normalization
            float p_accum[TILE_KV];
            float inv_l_tile = (partial.l > 0.0f) ? (1.0f / partial.l) : 0.0f;
            for (uint ki = 0; ki < tile_len; ++ki) {
                float exp_score = exp(scores[ki] - partial.m);
                p_accum[ki] = simd_sum(exp_score) * inv_l_tile;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    o_acc[r][i] += p_accum[ki] * input_to_float(V_tile[buf][ki][d]) * corr_tile;
                }
            }

            m_prev[r] = m_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf = buf_load;
        global_tile_idx++;
        } // end tiles within chunk
    } // end chunks loop

    // Store output
    const uint o_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;

    for (uint r = 0; r < sg_q_rows; ++r) {
        uint global_q = q_start + sg_q_start + r;
        if (global_q >= seq_q) continue;

        // SIMDgroup-level reduction for consistent normalization across all lanes
        float l_sum = simd_sum(l_prev[r]);
        float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;

        const uint o_row_base = o_base + (sg_q_start + r) * q_stride_s;
#ifdef USE_BF16_INPUTS
        float out_vals[HEAD_DIM_128 / SIMD_SIZE];
        for (uint i = 0; i < elems_per_lane; ++i) {
            out_vals[i] = o_acc[r][i] * inv_l;
        }
        store_output_bf16_vectorized(O, o_row_base, lane_id, elems_per_lane, out_vals, head_dim);
#else
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                store_output_scalar(O, o_row_base + d, o_acc[r][i] * inv_l);
            }
        }
#endif
    }
}

// ---------------------------------------------------------------------------
// Flash Attention V3 - Decode (Optimized for seq_q=1)
//
// Specialized kernel for autoregressive decoding where we have a single
// query token attending to a long KV cache. Uses all threads for K/V
// processing rather than distributing across Q rows.
//
// For causal decode (autoregressive), the query is at the latest position
// (seq_k-1), so it can attend to ALL positions 0..seq_k-1. No causal
// mask computation needed - eliminates one comparison per attention score.
//
// Dispatch: [num_heads_q, 1, batch] threadgroups
// Each threadgroup handles one head of one batch
// ---------------------------------------------------------------------------

kernel void flash_attention_v3_decode(
    device const input_t* Q         [[buffer(0)]],
    device const input_t* K         [[buffer(1)]],
    device const input_t* V         [[buffer(2)]],
    device output_t* O              [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint lane_id                    [[thread_index_in_simdgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    const uint head_q = tgid.x;
    const uint batch_idx = tgid.z;

    if (head_q >= params.num_heads_q) return;

    const uint gqa_ratio = params.gqa_ratio;
    const uint head_kv = head_q / gqa_ratio;

    const uint head_dim = params.head_dim;
    const uint seq_k = params.seq_k;
    const float scale = params.scale;

    // Strides: [batch, heads, seq, head_dim]
    const uint q_stride_b = params.num_heads_q * head_dim;  // seq_q=1
    const uint k_stride_b = params.num_heads_kv * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;
    const uint o_stride_b = params.num_heads_q * head_dim;  // seq_q=1

    // Base offsets - seq_q=1 so single query position
    const uint q_offset = batch_idx * q_stride_b + head_q * head_dim;
    const uint kv_base = batch_idx * k_stride_b + head_kv * k_stride_h;
    const uint o_offset = batch_idx * o_stride_b + head_q * head_dim;

    // =========================================================================
    // FAST PATH: seq_k == 1 (first token decode)
    // Softmax of single element is 1.0, output = V[0]
    // =========================================================================
    if (seq_k == 1 && sg_id == 0) {
        const uint elems_per_lane = head_dim / SIMD_SIZE;
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                float v_val = input_to_float(V[kv_base + d]);
                store_output_scalar(O, o_offset + d, v_val);
            }
        }
        return;
    }

    // =========================================================================
    // Load Q vector into registers (distributed across simdgroup 0)
    // Each lane holds elems_per_lane consecutive elements
    // =========================================================================
    const uint elems_per_lane = head_dim / SIMD_SIZE;
    float q_reg[HEAD_DIM_128 / SIMD_SIZE];

    if (sg_id == 0) {
        if (elems_per_lane == 4) {
            uint d = lane_id * 4;
            if (d + 3 < head_dim) {
                q_reg[0] = input_to_float(Q[q_offset + d + 0]);
                q_reg[1] = input_to_float(Q[q_offset + d + 1]);
                q_reg[2] = input_to_float(Q[q_offset + d + 2]);
                q_reg[3] = input_to_float(Q[q_offset + d + 3]);
            } else {
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d_tail = d + i;
                    q_reg[i] = (d_tail < head_dim) ? input_to_float(Q[q_offset + d_tail]) : 0.0f;
                }
            }
        } else {
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                q_reg[i] = (d < head_dim) ? input_to_float(Q[q_offset + d]) : 0.0f;
            }
        }
    }

    // =========================================================================
    // Threadgroup memory for K/V tiles (double-buffered)
    // Use HEAD_DIM_PADDED to avoid bank conflicts: 128 + 16 padding ensures
    // threads with stride SIMD_SIZE (32) access different banks.
    // =========================================================================
    threadgroup input_t K_tile[2][TILE_KV][HEAD_DIM_PADDED];
    threadgroup input_t V_tile[2][TILE_KV][HEAD_DIM_PADDED];

    // =========================================================================
    // Online softmax state (simdgroup 0 only for single Q)
    // =========================================================================
    if (sg_id != 0) return;  // Only simdgroup 0 computes for decode

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o_acc[HEAD_DIM_128 / SIMD_SIZE] = {0.0f};

    const uint num_tiles = (seq_k + TILE_KV - 1) / TILE_KV;

    // =========================================================================
    // Preload first K/V tile
    // =========================================================================
    {
        uint tile_len = min(uint(TILE_KV), seq_k);
        uint elems = tile_len * head_dim;
        uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;

        for (uint i = 0; i < per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG;
            if (idx < elems) {
                uint kv_row = idx / head_dim;
                uint kv_col = idx % head_dim;
                K_tile[0][kv_row][kv_col] = K[kv_base + kv_row * k_stride_s + kv_col];
                V_tile[0][kv_row][kv_col] = V[kv_base + kv_row * k_stride_s + kv_col];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf = 0;

    // =========================================================================
    // MAIN LOOP: Process K/V tiles with online softmax
    //
    // For causal decode (seq_q=1, query at seq_k-1 position):
    //   - No causal mask needed - query attends to ALL positions
    //   - All threads collaborate on single query instead of dividing across rows
    // =========================================================================

    for (uint tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        uint buf_load = 1 - buf;
        uint tile_start = tile_idx * TILE_KV;
        uint tile_len = min(uint(TILE_KV), seq_k - tile_start);

        // Async load next tile (all threads participate)
        if (tile_idx + 1 < num_tiles) {
            uint next_start = (tile_idx + 1) * TILE_KV;
            uint next_len = min(uint(TILE_KV), seq_k - next_start);
            uint elems = next_len * head_dim;
            uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;

            for (uint i = 0; i < per_thread; ++i) {
                uint idx = tid + i * THREADS_PER_TG;
                if (idx < elems) {
                    uint kv_row = idx / head_dim;
                    uint kv_col = idx % head_dim;
                    K_tile[buf_load][kv_row][kv_col] = K[kv_base + (next_start + kv_row) * k_stride_s + kv_col];
                    V_tile[buf_load][kv_row][kv_col] = V[kv_base + (next_start + kv_row) * k_stride_s + kv_col];
                }
            }
        }

        // =====================================================================
        // Compute Q @ K^T for this tile (simdgroup 0 only for single Q)
        //
        // NO CAUSAL MASK: decode query at position seq_k-1 attends to all
        // positions 0..seq_k-1. This eliminates one comparison per score.
        // =====================================================================

        float scores[TILE_KV];

        for (uint ki = 0; ki < tile_len; ++ki) {
            // Compute dot product Q @ K[ki]^T using simd-wide reduction
            float dot = 0.0f;
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                dot += q_reg[i] * input_to_float(K_tile[buf][ki][d]);
            }
            dot = simd_sum(dot);

            // Scale - no causal masking needed for decode
            scores[ki] = dot * scale;
        }

        // Pad invalid tile positions with -INFINITY
        for (uint ki = tile_len; ki < TILE_KV; ++ki) {
            scores[ki] = -INFINITY;
        }

        // =====================================================================
        // ONLINE SOFTMAX UPDATE
        //
        // Maintain running max (m_prev) and sum (l_prev) while accumulating
        // weighted values. When a new max is found, rescale accumulator.
        // =====================================================================

        // Find max in this tile
        float m_tile = -INFINITY;
        for (uint ki = 0; ki < tile_len; ++ki) {
            m_tile = max(m_tile, scores[ki]);
        }

        // Merge with running state
        float m_new = max(m_prev, m_tile);

        // Correction factor for previous accumulator
        float corr = exp(m_prev - m_new);

        // Rescale previous sum and output accumulator
        l_prev *= corr;
        for (uint i = 0; i < elems_per_lane; ++i) {
            o_acc[i] *= corr;
        }

        // Accumulate contributions from this tile
        for (uint ki = 0; ki < tile_len; ++ki) {
            float p = exp(scores[ki] - m_new);  // softmax probability
            l_prev += p;
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                o_acc[i] += p * input_to_float(V_tile[buf][ki][d]);
            }
        }

        m_prev = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf = buf_load;
    }

    // =========================================================================
    // Normalize and store output with SIMDgroup-level reduction
    // =========================================================================
    // Use simd_sum to ensure all lanes have consistent normalization factor
    // This provides better numerical stability and allows compiler optimizations
    float l_sum = simd_sum(l_prev);
    float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;

#ifdef USE_BF16_INPUTS
    float out_vals[HEAD_DIM_128 / SIMD_SIZE];
    for (uint i = 0; i < elems_per_lane; ++i) {
        out_vals[i] = o_acc[i] * inv_l;
    }
    store_output_bf16_vectorized(O, o_offset, lane_id, elems_per_lane, out_vals, head_dim);
#else
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        if (d < head_dim) {
            store_output_scalar(O, o_offset + d, o_acc[i] * inv_l);
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// Flash Attention V3 - Decode GQA variant
//
// Optimized for GQA during decode (seq_q=1).
// Multiple Q heads share the same K/V tiles.
// ---------------------------------------------------------------------------

kernel void flash_attention_v3_decode_gqa(
    device const input_t* Q         [[buffer(0)]],
    device const input_t* K         [[buffer(1)]],
    device const input_t* V         [[buffer(2)]],
    device output_t* O              [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint lane_id                    [[thread_index_in_simdgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    const uint head_kv = tgid.x;
    const uint batch_idx = tgid.z;

    const uint gqa_ratio = params.gqa_ratio;
    const uint head_dim = params.head_dim;
    const uint seq_k = params.seq_k;
    const float scale = params.scale;

    // K/V base offset
    const uint k_stride_b = params.num_heads_kv * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;
    const uint o_stride_b = params.num_heads_q * head_dim;
    const uint kv_base = batch_idx * k_stride_b + head_kv * k_stride_h;

    // Process multiple Q heads sharing this KV head
    const uint heads_per_pass = min(gqa_ratio, uint(NUM_SIMDGROUPS));
    const uint num_head_passes = (gqa_ratio + heads_per_pass - 1) / heads_per_pass;

    // Threadgroup memory for K/V (shared across all Q heads in group)
    // Use HEAD_DIM_PADDED to avoid bank conflicts when threads access
    // with stride SIMD_SIZE (32)
    threadgroup input_t K_tile[2][TILE_KV][HEAD_DIM_PADDED];
    threadgroup input_t V_tile[2][TILE_KV][HEAD_DIM_PADDED];

    const uint num_tiles = (seq_k + TILE_KV - 1) / TILE_KV;

    // Preload first K/V tile (all threads cooperate)
    {
        uint tile_len = min(uint(TILE_KV), seq_k);
        uint elems = tile_len * head_dim;
        uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;

        for (uint i = 0; i < per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG;
            if (idx < elems) {
                uint kv_row = idx / head_dim;
                uint kv_col = idx % head_dim;
                K_tile[0][kv_row][kv_col] = K[kv_base + kv_row * k_stride_s + kv_col];
                V_tile[0][kv_row][kv_col] = V[kv_base + kv_row * k_stride_s + kv_col];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf = 0;

    // Process each Q head pass (each simdgroup handles one head)
    for (uint head_pass = 0; head_pass < num_head_passes; ++head_pass) {
        uint head_offset = sg_id;
        if (head_offset >= heads_per_pass) continue;

        uint head_q = head_kv * gqa_ratio + head_offset;
        if (head_q >= params.num_heads_q) continue;

        // Q offset for this head
        const uint q_offset = batch_idx * params.num_heads_q * head_dim + head_q * head_dim;
        const uint o_offset = batch_idx * o_stride_b + head_q * head_dim;

        const uint elems_per_lane = head_dim / SIMD_SIZE;
        float q_reg[HEAD_DIM_128 / SIMD_SIZE];

        // Load Q into registers
        if (elems_per_lane == 4) {
            uint d = lane_id * 4;
            if (d + 3 < head_dim) {
                q_reg[0] = input_to_float(Q[q_offset + d + 0]);
                q_reg[1] = input_to_float(Q[q_offset + d + 1]);
                q_reg[2] = input_to_float(Q[q_offset + d + 2]);
                q_reg[3] = input_to_float(Q[q_offset + d + 3]);
            } else {
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d_tail = d + i;
                    q_reg[i] = (d_tail < head_dim) ? input_to_float(Q[q_offset + d_tail]) : 0.0f;
                }
            }
        } else {
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                q_reg[i] = (d < head_dim) ? input_to_float(Q[q_offset + d]) : 0.0f;
            }
        }

        // Fast path for seq_k == 1
        if (seq_k == 1) {
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                if (d < head_dim) {
                    float v_val = input_to_float(V[kv_base + d]);
                    store_output_scalar(O, o_offset + d, v_val);
                }
            }
            continue;
        }

        // Online softmax state
        float m_prev = -INFINITY;
        float l_prev = 0.0f;
        float o_acc[HEAD_DIM_128 / SIMD_SIZE] = {0.0f};

        // Process tiles
        for (uint tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            uint buf_load = 1 - buf;
            uint tile_start = tile_idx * TILE_KV;
            uint tile_len = min(uint(TILE_KV), seq_k - tile_start);

            // Load next tile
            if (tile_idx + 1 < num_tiles) {
                uint next_start = (tile_idx + 1) * TILE_KV;
                uint next_len = min(uint(TILE_KV), seq_k - next_start);
                uint elems = next_len * head_dim;
                uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;

                for (uint i = 0; i < per_thread; ++i) {
                    uint idx = tid + i * THREADS_PER_TG;
                    if (idx < elems) {
                        uint kv_row = idx / head_dim;
                        uint kv_col = idx % head_dim;
                        K_tile[buf_load][kv_row][kv_col] = K[kv_base + (next_start + kv_row) * k_stride_s + kv_col];
                        V_tile[buf_load][kv_row][kv_col] = V[kv_base + (next_start + kv_row) * k_stride_s + kv_col];
                    }
                }
            }

            // Compute scores (no causal mask for decode)
            float scores[TILE_KV];
            for (uint ki = 0; ki < tile_len; ++ki) {
                float dot = 0.0f;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    dot += q_reg[i] * input_to_float(K_tile[buf][ki][d]);
                }
                dot = simd_sum(dot);
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
                    o_acc[i] += p * input_to_float(V_tile[buf][ki][d]);
                }
            }

            m_prev = m_new;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf = buf_load;
        }

        // Normalize and store with SIMDgroup-level reduction
        // Use simd_sum to ensure consistent normalization across all lanes
        float l_sum = simd_sum(l_prev);
        float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
#ifdef USE_BF16_INPUTS
        float out_vals[HEAD_DIM_128 / SIMD_SIZE];
        for (uint i = 0; i < elems_per_lane; ++i) {
            out_vals[i] = o_acc[i] * inv_l;
        }
        store_output_bf16_vectorized(O, o_offset, lane_id, elems_per_lane, out_vals, head_dim);
#else
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                store_output_scalar(O, o_offset + d, o_acc[i] * inv_l);
            }
        }
#endif
    }
}

// ---------------------------------------------------------------------------
// Flash Attention V3 - Causal GQA variant
// ---------------------------------------------------------------------------

kernel void flash_attention_v3_causal_gqa(
    device const input_t* Q         [[buffer(0)]],
    device const input_t* K         [[buffer(1)]],
    device const input_t* V         [[buffer(2)]],
    device output_t* O              [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint lane_id                    [[thread_index_in_simdgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    const uint head_kv = tgid.x;
    const uint q_tile_idx = tgid.y;
    const uint batch_idx = tgid.z;

    const uint gqa_ratio = params.gqa_ratio;
    const uint head_dim = params.head_dim;
    const uint seq_q = params.seq_q;
    const uint seq_k = params.seq_k;
    const float scale = params.scale;

    const uint q_start = q_tile_idx * TILE_Q;
    if (q_start >= seq_q) return;

    const uint q_rows = min(TILE_Q, seq_q - q_start);

    // Strides
    const uint q_stride_b = params.num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;
    const uint k_stride_b = params.num_heads_kv * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;

    const uint kv_base = batch_idx * k_stride_b + head_kv * k_stride_h;

    threadgroup input_t K_tile[2][TILE_KV][HEAD_DIM_PADDED];
    threadgroup input_t V_tile[2][TILE_KV][HEAD_DIM_PADDED];

    const uint heads_per_pass = min(gqa_ratio, uint(NUM_SIMDGROUPS));
    const uint num_head_passes = (gqa_ratio + heads_per_pass - 1) / heads_per_pass;

    const uint max_q_pos_in_tg = q_start + q_rows - 1;
    const uint causal_limit_tg = min(max_q_pos_in_tg + 1, seq_k);
    const uint num_kv_tiles = (causal_limit_tg + TILE_KV - 1) / TILE_KV;

    // Process each Q head pass
    for (uint head_pass = 0; head_pass < num_head_passes; ++head_pass) {
        uint head_offset = head_pass * heads_per_pass + sg_id;
        if (head_offset >= gqa_ratio) continue;

        uint head_q = head_kv * gqa_ratio + head_offset;
        if (head_q >= params.num_heads_q) continue;

        const uint q_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;

        // Load Q into registers
        const uint elems_per_lane = head_dim / SIMD_SIZE;
        float q_reg[4][HEAD_DIM_128 / SIMD_SIZE];
        float m_prev[4];
        float l_prev[4];
        float o_acc[4][HEAD_DIM_128 / SIMD_SIZE];

        for (uint r = 0; r < q_rows; ++r) {
            m_prev[r] = -INFINITY;
            l_prev[r] = 0.0f;
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                q_reg[r][i] = input_to_float(Q[q_base + r * q_stride_s + d]);
                o_acc[r][i] = 0.0f;
            }
        }

        // Preload first K/V tile
        {
            uint tile_len = min(uint(TILE_KV), seq_k);
            uint elems = tile_len * head_dim;
            uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;
            for (uint i = 0; i < per_thread; ++i) {
                uint idx = tid + i * THREADS_PER_TG;
                if (idx < elems) {
                    uint kv_row = idx / head_dim;
                    uint kv_col = idx % head_dim;
                    K_tile[0][kv_row][kv_col] = K[kv_base + kv_row * k_stride_s + kv_col];
                    V_tile[0][kv_row][kv_col] = V[kv_base + kv_row * k_stride_s + kv_col];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint buf = 0;

        for (uint tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
            uint buf_load = 1 - buf;
            uint tile_start = tile_idx * TILE_KV;
            uint tile_len = min(uint(TILE_KV), causal_limit_tg - tile_start);

            // Skip empty tiles
            if (tile_len == 0) {
                buf = buf_load;
                continue;
            }

            // Load next tile
            if (tile_idx + 1 < num_kv_tiles) {
                uint next_start = (tile_idx + 1) * TILE_KV;
                uint next_len = min(uint(TILE_KV), causal_limit_tg - next_start);
                uint elems = next_len * head_dim;
                uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;
                for (uint i = 0; i < per_thread; ++i) {
                    uint idx = tid + i * THREADS_PER_TG;
                    if (idx < elems) {
                        uint kv_row = idx / head_dim;
                        uint kv_col = idx % head_dim;
                        K_tile[buf_load][kv_row][kv_col] = K[kv_base + (next_start + kv_row) * k_stride_s + kv_col];
                        V_tile[buf_load][kv_row][kv_col] = V[kv_base + (next_start + kv_row) * k_stride_s + kv_col];
                    }
                }
            }

            // Compute with causal mask
            for (uint r = 0; r < q_rows; ++r) {
                uint global_q_pos = q_start + r;
                float scores[TILE_KV];

                for (uint ki = 0; ki < tile_len; ++ki) {
                    uint k_pos = tile_start + ki;

                    float dot = 0.0f;
                    for (uint i = 0; i < elems_per_lane; ++i) {
                        uint d = lane_id * elems_per_lane + i;
                        dot += q_reg[r][i] * input_to_float(K_tile[buf][ki][d]);
                    }
                    dot = simd_sum(dot);

                    // Branchless causal mask
                    float score = dot * scale;
                    scores[ki] = select(score, -INFINITY, k_pos > global_q_pos);
                }

                for (uint ki = tile_len; ki < TILE_KV; ++ki) {
                    scores[ki] = -INFINITY;
                }

                // =====================================================================
                // ONLINE SOFTMAX WITH PARTIAL SUM ACCUMULATION
                // =====================================================================
                //
                // For chunked attention blocks, we compute partial softmax statistics
                // per tile and merge them into the running accumulator.
                //
                // The update formula:
                //   m_new = max(m_prev, m_tile)
                //   l_new = l_prev * exp(m_prev - m_new) + l_tile * exp(m_tile - m_new)
                //   o_new = o_prev * exp(m_prev - m_new) + o_tile * exp(m_tile - m_new)
                //
                // This is numerically stable and handles arbitrary chunk sizes.
                // =====================================================================

                // Compute partial statistics for this tile chunk
                SoftmaxPartial partial = compute_partial_softmax(scores, 0, tile_len);

                // Merge with running accumulator
                float m_new = max(m_prev[r], partial.m);
                float corr_prev = exp(m_prev[r] - m_new);
                float corr_tile = exp(partial.m - m_new);

                // Update sum and output accumulation
                l_prev[r] = l_prev[r] * corr_prev + partial.l * corr_tile;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    o_acc[r][i] = o_acc[r][i] * corr_prev;  // Scale existing output
                }

                // Accumulate weighted values using the tile's contribution
                // Use SIMDgroup-level reduction for faster normalization
                float p_accum[TILE_KV];
                float inv_l_tile = (partial.l > 0.0f) ? (1.0f / partial.l) : 0.0f;
                for (uint ki = 0; ki < tile_len; ++ki) {
                    float exp_score = exp(scores[ki] - partial.m);
                    p_accum[ki] = simd_sum(exp_score) * inv_l_tile;
                    for (uint i = 0; i < elems_per_lane; ++i) {
                        uint d = lane_id * elems_per_lane + i;
                        o_acc[r][i] += p_accum[ki] * input_to_float(V_tile[buf][ki][d]) * corr_tile;
                    }
                }

                m_prev[r] = m_new;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf = buf_load;
        }

        // Store output
        const uint o_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;
        for (uint r = 0; r < q_rows; ++r) {
            // SIMDgroup-level reduction for consistent normalization across all lanes
            float l_sum = simd_sum(l_prev[r]);
            float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
            const uint o_row_base = o_base + r * q_stride_s;
#ifdef USE_BF16_INPUTS
            float out_vals[HEAD_DIM_128 / SIMD_SIZE];
            for (uint i = 0; i < elems_per_lane; ++i) {
                out_vals[i] = o_acc[r][i] * inv_l;
            }
            store_output_bf16_vectorized(O, o_row_base, lane_id, elems_per_lane, out_vals, head_dim);
#else
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                if (d < head_dim) {
                    store_output_scalar(O, o_row_base + d, o_acc[r][i] * inv_l);
                }
            }
#endif
        }
    }
}

// ---------------------------------------------------------------------------
// Flash Attention V3 - General GQA variant (non-causal, prefill)
// ---------------------------------------------------------------------------

kernel void flash_attention_v3_gqa(
    device const input_t* Q         [[buffer(0)]],
    device const input_t* K         [[buffer(1)]],
    device const input_t* V         [[buffer(2)]],
    device output_t* O              [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint lane_id                    [[thread_index_in_simdgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    const uint head_kv = tgid.x;
    const uint q_tile_idx = tgid.y;
    const uint batch_idx = tgid.z;

    const uint gqa_ratio = params.gqa_ratio;
    const uint head_dim = params.head_dim;
    const uint seq_q = params.seq_q;
    const uint seq_k = params.seq_k;
    const float scale = params.scale;

    const uint q_start = q_tile_idx * TILE_Q;
    if (q_start >= seq_q) return;

    const uint q_rows = min(TILE_Q, seq_q - q_start);

    // Strides
    const uint q_stride_b = params.num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;
    const uint k_stride_b = params.num_heads_kv * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;

    const uint kv_base = batch_idx * k_stride_b + head_kv * k_stride_h;

    threadgroup input_t K_tile[2][TILE_KV][HEAD_DIM_PADDED];
    threadgroup input_t V_tile[2][TILE_KV][HEAD_DIM_PADDED];

    const uint heads_per_pass = min(gqa_ratio, uint(NUM_SIMDGROUPS));
    const uint num_head_passes = (gqa_ratio + heads_per_pass - 1) / heads_per_pass;

    const uint num_kv_tiles = (seq_k + TILE_KV - 1) / TILE_KV;

    // Preload first K/V tile (shared across all Q heads)
    {
        uint tile_len = min(uint(TILE_KV), seq_k);
        uint elems = tile_len * head_dim;
        uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;

        for (uint i = 0; i < per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG;
            if (idx < elems) {
                uint kv_row = idx / head_dim;
                uint kv_col = idx % head_dim;
                K_tile[0][kv_row][kv_col] = K[kv_base + kv_row * k_stride_s + kv_col];
                V_tile[0][kv_row][kv_col] = V[kv_base + kv_row * k_stride_s + kv_col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Process each Q head pass
    for (uint head_pass = 0; head_pass < num_head_passes; ++head_pass) {
        uint head_offset = head_pass * heads_per_pass + sg_id;
        if (head_offset >= gqa_ratio) continue;

        uint head_q = head_kv * gqa_ratio + head_offset;
        if (head_q >= params.num_heads_q) continue;

        const uint q_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;

        // Load Q into registers
        const uint elems_per_lane = head_dim / SIMD_SIZE;
        float q_reg[4][HEAD_DIM_128 / SIMD_SIZE];
        float m_prev[4];
        float l_prev[4];
        float o_acc[4][HEAD_DIM_128 / SIMD_SIZE];

        for (uint r = 0; r < q_rows; ++r) {
            m_prev[r] = -INFINITY;
            l_prev[r] = 0.0f;
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                q_reg[r][i] = (d < head_dim) ? input_to_float(Q[q_base + r * q_stride_s + d]) : 0.0f;
                o_acc[r][i] = 0.0f;
            }
        }

        // Process K/V tiles
        for (uint tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
            const uint tile_start = tile_idx * TILE_KV;
            const uint tile_len = min(uint(TILE_KV), seq_k - tile_start);
            const uint curr_buf = tile_idx & 1u;
            const uint next_buf = 1u - curr_buf;

            // Prefetch next tile (all threads cooperate, once per tile)
            if (tile_idx + 1 < num_kv_tiles) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg_id == 0) {
                    uint next_tile_start = (tile_idx + 1) * TILE_KV;
                    uint next_tile_len = min(uint(TILE_KV), seq_k - next_tile_start);
                    uint elems = next_tile_len * head_dim;
                    uint per_thread = (elems + SIMD_SIZE - 1) / SIMD_SIZE;

                    for (uint i = 0; i < per_thread; ++i) {
                        uint idx = lane_id + i * SIMD_SIZE;
                        if (idx < elems) {
                            uint kv_row = idx / head_dim;
                            uint kv_col = idx % head_dim;
                            uint src_row = next_tile_start + kv_row;
                            K_tile[next_buf][kv_row][kv_col] = K[kv_base + src_row * k_stride_s + kv_col];
                            V_tile[next_buf][kv_row][kv_col] = V[kv_base + src_row * k_stride_s + kv_col];
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Compute attention scores (Q @ K^T)
            float s_local[4][TILE_KV];
            for (uint r = 0; r < q_rows; ++r) {
                for (uint k_row = 0; k_row < tile_len; ++k_row) {
                    float dot = 0.0f;
                    for (uint i = 0; i < elems_per_lane; ++i) {
                        uint d = lane_id * elems_per_lane + i;
                        if (d < head_dim) {
                            float k_val = input_to_float(K_tile[curr_buf][k_row][d]);
                            dot += q_reg[r][i] * k_val;
                        }
                    }
                    dot = simd_sum(dot) * scale;
                    s_local[r][k_row] = dot;
                }
            }

            // Online softmax update
            for (uint r = 0; r < q_rows; ++r) {
                float m_new = m_prev[r];
                for (uint k_row = 0; k_row < tile_len; ++k_row) {
                    m_new = max(m_new, s_local[r][k_row]);
                }

                float l_new = 0.0f;
                for (uint k_row = 0; k_row < tile_len; ++k_row) {
                    float p = exp(s_local[r][k_row] - m_new);
                    s_local[r][k_row] = p;
                    l_new += p;
                }

                float correction = exp(m_prev[r] - m_new);
                float scale_o = correction;
                float scale_l = correction * l_prev[r] + l_new;

                for (uint i = 0; i < elems_per_lane; ++i) {
                    o_acc[r][i] *= scale_o;
                }

                // Accumulate attention-weighted values
                for (uint k_row = 0; k_row < tile_len; ++k_row) {
                    float p = s_local[r][k_row];
                    for (uint i = 0; i < elems_per_lane; ++i) {
                        uint d = lane_id * elems_per_lane + i;
                        if (d < head_dim) {
                            float v_val = input_to_float(V_tile[curr_buf][k_row][d]);
                            o_acc[r][i] += p * v_val;
                        }
                    }
                }

                m_prev[r] = m_new;
                l_prev[r] = scale_l;
            }
        }

        // Write output
        const uint o_stride_b = params.num_heads_q * seq_q * head_dim;
        const uint o_stride_h = seq_q * head_dim;
        const uint o_stride_s = head_dim;
        const uint o_base = batch_idx * o_stride_b + head_q * o_stride_h + q_start * o_stride_s;

        for (uint r = 0; r < q_rows; ++r) {
            float inv_l = (l_prev[r] > 0.0f) ? (1.0f / l_prev[r]) : 0.0f;
            uint o_row_base = o_base + r * o_stride_s;

#ifdef USE_BF16_INPUTS
            float out_vals[HEAD_DIM_128 / SIMD_SIZE];
            for (uint i = 0; i < elems_per_lane; ++i) {
                out_vals[i] = o_acc[r][i] * inv_l;
            }
            store_output_bf16_vectorized(O, o_row_base, lane_id, elems_per_lane, out_vals, head_dim);
#else
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                if (d < head_dim) {
                    store_output_scalar(O, o_row_base + d, o_acc[r][i] * inv_l);
                }
            }
#endif
        }
    }
}

// ---------------------------------------------------------------------------
// Flash Attention V3 - MQA variant (gqa_ratio = num_heads_q, 1 KV head)
// ---------------------------------------------------------------------------

kernel void flash_attention_v3_mqa(
    device const input_t* Q         [[buffer(0)]],
    device const input_t* K         [[buffer(1)]],
    device const input_t* V         [[buffer(2)]],
    device output_t* O              [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint lane_id                    [[thread_index_in_simdgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    // MQA: All query heads share single K/V head
    // Dispatch: [1, ceil(seq_q / TILE_Q), batch] threadgroups
    
    const uint q_tile_idx = tgid.y;
    const uint batch_idx = tgid.z;

    const uint num_heads_q = params.num_heads_q;
    const uint head_dim = params.head_dim;
    const uint seq_q = params.seq_q;
    const uint seq_k = params.seq_k;
    const float scale = params.scale;

    const uint q_start = q_tile_idx * TILE_Q;
    if (q_start >= seq_q) return;

    const uint q_rows = min(TILE_Q, seq_q - q_start);

    // Strides
    const uint q_stride_b = num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;
    const uint k_stride_b = seq_k * head_dim; // Single KV head
    const uint k_stride_s = head_dim;

    const uint kv_base = batch_idx * k_stride_b;

    threadgroup input_t K_tile[2][TILE_KV][HEAD_DIM_PADDED];
    threadgroup input_t V_tile[2][TILE_KV][HEAD_DIM_PADDED];

    const uint heads_per_pass = min(num_heads_q, uint(NUM_SIMDGROUPS));
    const uint num_head_passes = (num_heads_q + heads_per_pass - 1) / heads_per_pass;

    const uint num_kv_tiles = (seq_k + TILE_KV - 1) / TILE_KV;

    // Preload first K/V tile (shared by all heads)
    {
        uint tile_len = min(uint(TILE_KV), seq_k);
        uint elems = tile_len * head_dim;
        uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;

        for (uint i = 0; i < per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG;
            if (idx < elems) {
                uint kv_row = idx / head_dim;
                uint kv_col = idx % head_dim;
                K_tile[0][kv_row][kv_col] = K[kv_base + kv_row * k_stride_s + kv_col];
                V_tile[0][kv_row][kv_col] = V[kv_base + kv_row * k_stride_s + kv_col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Process each Q head pass
    for (uint head_pass = 0; head_pass < num_head_passes; ++head_pass) {
        uint head_offset = head_pass * heads_per_pass + sg_id;
        if (head_offset >= num_heads_q) continue;

        uint head_q = head_offset;
        const uint q_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;

        // Load Q into registers
        const uint elems_per_lane = head_dim / SIMD_SIZE;
        float q_reg[4][HEAD_DIM_128 / SIMD_SIZE];
        float m_prev[4];
        float l_prev[4];
        float o_acc[4][HEAD_DIM_128 / SIMD_SIZE];

        for (uint r = 0; r < q_rows; ++r) {
            m_prev[r] = -INFINITY;
            l_prev[r] = 0.0f;
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                q_reg[r][i] = (d < head_dim) ? input_to_float(Q[q_base + r * q_stride_s + d]) : 0.0f;
                o_acc[r][i] = 0.0f;
            }
        }

        // Process K/V tiles
        for (uint tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
            const uint tile_start = tile_idx * TILE_KV;
            const uint tile_len = min(uint(TILE_KV), seq_k - tile_start);
            const uint curr_buf = tile_idx & 1u;
            const uint next_buf = 1u - curr_buf;

            // Prefetch next tile (all threads cooperate, once per tile)
            if (tile_idx + 1 < num_kv_tiles) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg_id == 0) {
                    uint next_tile_start = (tile_idx + 1) * TILE_KV;
                    uint next_tile_len = min(uint(TILE_KV), seq_k - next_tile_start);
                    uint elems = next_tile_len * head_dim;
                    uint per_thread = (elems + SIMD_SIZE - 1) / SIMD_SIZE;

                    for (uint i = 0; i < per_thread; ++i) {
                        uint idx = lane_id + i * SIMD_SIZE;
                        if (idx < elems) {
                            uint kv_row = idx / head_dim;
                            uint kv_col = idx % head_dim;
                            uint src_row = next_tile_start + kv_row;
                            K_tile[next_buf][kv_row][kv_col] = K[kv_base + src_row * k_stride_s + kv_col];
                            V_tile[next_buf][kv_row][kv_col] = V[kv_base + src_row * k_stride_s + kv_col];
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Compute attention scores (Q @ K^T)
            float s_local[4][TILE_KV];
            for (uint r = 0; r < q_rows; ++r) {
                for (uint k_row = 0; k_row < tile_len; ++k_row) {
                    float dot = 0.0f;
                    for (uint i = 0; i < elems_per_lane; ++i) {
                        uint d = lane_id * elems_per_lane + i;
                        if (d < head_dim) {
                            float k_val = input_to_float(K_tile[curr_buf][k_row][d]);
                            dot += q_reg[r][i] * k_val;
                        }
                    }
                    dot = simd_sum(dot) * scale;
                    s_local[r][k_row] = dot;
                }
            }

            // Online softmax update
            for (uint r = 0; r < q_rows; ++r) {
                float m_new = m_prev[r];
                for (uint k_row = 0; k_row < tile_len; ++k_row) {
                    m_new = max(m_new, s_local[r][k_row]);
                }

                float l_new = 0.0f;
                for (uint k_row = 0; k_row < tile_len; ++k_row) {
                    float p = exp(s_local[r][k_row] - m_new);
                    s_local[r][k_row] = p;
                    l_new += p;
                }

                float correction = exp(m_prev[r] - m_new);
                float scale_o = correction;
                float scale_l = correction * l_prev[r] + l_new;

                for (uint i = 0; i < elems_per_lane; ++i) {
                    o_acc[r][i] *= scale_o;
                }

                // Accumulate attention-weighted values
                for (uint k_row = 0; k_row < tile_len; ++k_row) {
                    float p = s_local[r][k_row];
                    for (uint i = 0; i < elems_per_lane; ++i) {
                        uint d = lane_id * elems_per_lane + i;
                        if (d < head_dim) {
                            float v_val = input_to_float(V_tile[curr_buf][k_row][d]);
                            o_acc[r][i] += p * v_val;
                        }
                    }
                }

                m_prev[r] = m_new;
                l_prev[r] = scale_l;
            }
        }

        // Write output
        const uint o_stride_b = num_heads_q * seq_q * head_dim;
        const uint o_stride_h = seq_q * head_dim;
        const uint o_stride_s = head_dim;
        const uint o_base = batch_idx * o_stride_b + head_q * o_stride_h + q_start * o_stride_s;

        for (uint r = 0; r < q_rows; ++r) {
            float inv_l = (l_prev[r] > 0.0f) ? (1.0f / l_prev[r]) : 0.0f;
            uint o_row_base = o_base + r * o_stride_s;

#ifdef USE_BF16_INPUTS
            float out_vals[HEAD_DIM_128 / SIMD_SIZE];
            for (uint i = 0; i < elems_per_lane; ++i) {
                out_vals[i] = o_acc[r][i] * inv_l;
            }
            store_output_bf16_vectorized(O, o_row_base, lane_id, elems_per_lane, out_vals, head_dim);
#else
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                if (d < head_dim) {
                    store_output_scalar(O, o_row_base + d, o_acc[r][i] * inv_l);
                }
            }
#endif
        }
    }
}
