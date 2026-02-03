// attention.metal - Fused Q×K^T + scale + mask + softmax kernel
//
// Computes: P = softmax((Q @ K^T) * scale + mask)
// Avoids materializing the full [seq_q, seq_k] attention matrix in global memory.
//
// Algorithm: Online softmax (Milakov & Gimelshein 2018)
//   For each query row, we process K in tiles, maintaining a running max and
//   a running rescaled exponential sum. This gives numerically stable softmax
//   in a single pass over K without storing the full score vector.
//
// Dispatch: one threadgroup per (batch, head, query_row) triple.
//   Grid: [seq_q, num_heads, batch]
//   Threadgroup: [THREADS_PER_TG, 1, 1]
//
// Each thread in the threadgroup handles a strided subset of the seq_k
// dimension, computing partial dot products and participating in the
// online softmax reduction.
//
// Memory layout (all row-major):
//   Q: [batch, num_heads, seq_q, head_dim]
//   K: [batch, num_heads, seq_k, head_dim]
//   mask: [seq_q, seq_k] (broadcast batch/head) or nullptr for causal
//   P: [batch, num_heads, seq_q, seq_k]
//
// Supported head_dim: 32, 64, 128 (compile-time via template constant)
//
// CUDA -> Metal mapping for this kernel:
//   __shfl_xor_sync -> simd_shuffle_xor
//   warpReduceMax   -> simd_max
//   warpReduceSum   -> simd_sum
//   blockReduceMax  -> threadgroup reduction via simd leaders
//   atomicMax(shmem)-> threadgroup atomic or explicit reduction

#include <metal_stdlib>
#include "reduction_helpers.metal"
using namespace metal;

// ---------------------------------------------------------------------------
// Tile and threadgroup configuration
//
// TILE_K_ATT: number of K columns processed per outer loop iteration.
//   Each thread computes dot products for (TILE_K_ATT / THREADS_PER_TG_ATT)
//   K vectors per iteration.
//
// HEAD_DIM_MAX: maximum supported head dimension for threadgroup memory.
//   Actual head_dim is passed at runtime; we allocate for the max.
//
// BLOCK_SPARSE_*: Configuration for block-sparse attention masks.
//   Block-sparse attention represents the mask at block granularity rather
//   than per-element. Each block of size (BLOCK_Q x BLOCK_K) is either fully
//   computed or fully skipped, reducing memory and computation for sparse
//   attention patterns (e.g., BigBird, Longformer, sliding window).
//
//   The mask is stored as a compact bitset where each bit corresponds to one
//   block: mask_bit_idx = q_block_idx * num_k_blocks + k_block_idx
// ---------------------------------------------------------------------------

constant constexpr uint TILE_K_ATT = 128;      // K vectors loaded per tile
constant constexpr uint THREADS_PER_TG_ATT = 128;  // 4 simdgroups
constant constexpr uint SIMDGROUPS_ATT = THREADS_PER_TG_ATT / 32;
constant constexpr uint HEAD_DIM_MAX = 128;

// Tile sizes for the tiled and fused variants
constant constexpr uint K_TILE_TILED = 32;    // K vectors per tile in tiled variant
constant constexpr uint KV_TILE_FUSED = 16;   // KV vectors per tile in fused variant
constant constexpr uint O_PER_THREAD = (HEAD_DIM_MAX + THREADS_PER_TG_ATT - 1) / THREADS_PER_TG_ATT;

// Block-sparse configuration constants
constant constexpr uint BLOCK_Q_DEFAULT = 16;  // Default block size in Q dimension
constant constexpr uint BLOCK_K_DEFAULT = 16;  // Default block size in K dimension
constant constexpr uint BLOCKS_PER_TG = 8;    // Number of K blocks processed per threadgroup iteration

// ---------------------------------------------------------------------------
// Utility: threadgroup-wide max and sum reductions
//
// Pattern: Helper functions write to threadgroup scratch but do NOT call
// threadgroup_barrier. Kernels call barriers before reading results.
//
// Usage in kernel:
//   parallel_reduce_max_phase1(scratch, value, tid);
//   threadgroup_barrier(mem_flags::mem_threadgroup);
//   parallel_reduce_max_phase2(scratch, tid);
//   threadgroup_barrier(mem_flags::mem_threadgroup);
//   float result = scratch[0];
// ---------------------------------------------------------------------------

// Phase 1: Each simdgroup reduces locally and leader writes to scratch
inline void parallel_reduce_max_phase1(
    threadgroup float* scratch,
    float value,
    uint tid
) {
    float sg_max = simd_max(value);
    uint sg_id = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) {
        scratch[sg_id] = sg_max;
    }
    // Caller must call threadgroup_barrier after this
}

// Phase 2: First simdgroup reduces across all simdgroups
inline void parallel_reduce_max_phase2(
    threadgroup float* scratch,
    uint tid
) {
    uint sg_id = tid / 32;
    uint lane = tid % 32;
    if (sg_id == 0) {
        float v = (lane < SIMDGROUPS_ATT) ? scratch[lane] : -INFINITY;
        float result = simd_max(v);
        if (lane == 0) {
            scratch[0] = result;
        }
    }
    // Caller must call threadgroup_barrier after this
}

// Phase 1: Each simdgroup reduces locally and leader writes to scratch
inline void parallel_reduce_sum_phase1(
    threadgroup float* scratch,
    float value,
    uint tid
) {
    float sg_sum = simd_sum(value);
    uint sg_id = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) {
        scratch[sg_id] = sg_sum;
    }
    // Caller must call threadgroup_barrier after this
}

// Phase 2: First simdgroup reduces across all simdgroups
inline void parallel_reduce_sum_phase2(
    threadgroup float* scratch,
    uint tid
) {
    uint sg_id = tid / 32;
    uint lane = tid % 32;
    if (sg_id == 0) {
        float v = (lane < SIMDGROUPS_ATT) ? scratch[lane] : 0.0f;
        float result = simd_sum(v);
        if (lane == 0) {
            scratch[0] = result;
        }
    }
    // Caller must call threadgroup_barrier after this
}

// ---------------------------------------------------------------------------
// Main attention kernel: fused QK^T + scale + mask + online softmax
// ---------------------------------------------------------------------------

kernel void attention_qk_softmax(
    device const half* Q         [[buffer(0)]],   // [batch, heads, seq_q, head_dim]
    device const half* K         [[buffer(1)]],   // [batch, heads, seq_k, head_dim]
    device const half* mask      [[buffer(2)]],   // [seq_q, seq_k] or nullptr
    device half* P               [[buffer(3)]],   // [batch, heads, seq_q, seq_k]
    constant uint& batch         [[buffer(4)]],
    constant uint& num_heads     [[buffer(5)]],
    constant uint& seq_q         [[buffer(6)]],
    constant uint& seq_k         [[buffer(7)]],
    constant uint& head_dim      [[buffer(8)]],
    constant float& scale        [[buffer(9)]],   // 1/sqrt(head_dim), float for precision
    constant uint& mask_stride_q [[buffer(10)]],  // stride along query dim (== seq_k for dense)
    constant uint& causal        [[buffer(11)]],  // 1 = apply causal mask, 0 = use explicit mask
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]]
) {
    // Decode grid position: (query_row, head, batch_idx)
    uint q_row = tgid.x;
    uint head = tgid.y;
    uint batch_idx = tgid.z;

    if (q_row >= seq_q || head >= num_heads || batch_idx >= batch) return;

    // Pointer to this query vector: Q[batch_idx, head, q_row, :]
    uint q_offset = ((batch_idx * num_heads + head) * seq_q + q_row) * head_dim;
    device const half* q_ptr = Q + q_offset;

    // Base pointer for K[batch_idx, head, :, :]
    uint k_base = (batch_idx * num_heads + head) * seq_k * head_dim;
    device const half* k_base_ptr = K + k_base;

    // Output pointer: P[batch_idx, head, q_row, :]
    uint p_offset = ((batch_idx * num_heads + head) * seq_q + q_row) * seq_k;
    device half* p_ptr = P + p_offset;

    // Mask pointer for this query row (if not causal)
    device const half* mask_row = nullptr;
    if (!causal && mask != nullptr) {
        mask_row = mask + q_row * mask_stride_q;
    }

    // Threadgroup memory for Q vector cache and reductions
    threadgroup half Q_cache[HEAD_DIM_MAX];
    threadgroup float reduction_scratch[SIMDGROUPS_ATT];

    // Cooperatively load Q[q_row] into threadgroup memory
    for (uint i = tid; i < head_dim; i += THREADS_PER_TG_ATT) {
        Q_cache[i] = q_ptr[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Online softmax: process K in tiles, maintaining running max and sum
    //
    // For numerical stability we use the "online" trick:
    //   After processing tile t, we have:
    //     m_t = max of all scores seen so far
    //     d_t = sum of exp(score_i - m_t) for all i seen so far
    //
    //   When a new tile gives a new max m_{t+1} > m_t:
    //     d_{t+1} = d_t * exp(m_t - m_{t+1}) + sum_new_tile
    //
    //   Final: P[i] = exp(score_i - m_final) / d_final
    //
    // Since we write to global memory, we do two passes:
    //   Pass 1: compute all scores, find global max and sum
    //   Pass 2: normalize and write probabilities
    //
    // For seq_k <= TILE_K_ATT * 32 (4096), scores fit in registers.
    // For larger seq_k, we write intermediate scores to P buffer then normalize.
    // -------------------------------------------------------------------------

    // Determine if we can keep all scores in registers
    constexpr uint MAX_SCORES_REG = 64; // Supports seq_k <= 8192
    float local_scores[MAX_SCORES_REG];
    bool use_registers = (seq_k <= MAX_SCORES_REG * THREADS_PER_TG_ATT);
    uint score_idx = 0;

    // Pass 1: compute scores, find max and sum
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;

    // Total number of K tiles
    uint num_k_tiles = (seq_k + TILE_K_ATT - 1) / TILE_K_ATT;

    for (uint tile = 0; tile < num_k_tiles; ++tile) {
        uint k_start = tile * TILE_K_ATT;

        // Each thread processes one K vector per tile (K_PER_THREAD=1)
        uint k_idx = k_start + tid;

        float score = -INFINITY;
        if (k_idx < seq_k) {
            // Apply causal mask: skip future positions
            if (causal && k_idx > q_row) {
                score = -INFINITY;
            } else {
                // Compute dot product: Q_cache[:] dot K[k_idx, :]
                device const half* k_vec = k_base_ptr + k_idx * head_dim;
                float dot = 0.0f;

                // Vectorized dot product over head_dim
                // Process 4 elements at a time for better throughput
                uint d = 0;
                for (; d + 3 < head_dim; d += 4) {
                    float q0 = float(Q_cache[d]);
                    float q1 = float(Q_cache[d + 1]);
                    float q2 = float(Q_cache[d + 2]);
                    float q3 = float(Q_cache[d + 3]);
                    float k0 = float(k_vec[d]);
                    float k1 = float(k_vec[d + 1]);
                    float k2 = float(k_vec[d + 2]);
                    float k3 = float(k_vec[d + 3]);
                    dot += q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
                }
                // Handle remainder
                for (; d < head_dim; ++d) {
                    dot += float(Q_cache[d]) * float(k_vec[d]);
                }

                // Apply scale
                score = dot * scale;

                // Apply mask (additive)
                if (!causal && mask_row != nullptr) {
                    score += float(mask_row[k_idx]);
                }
            }
        }

        // Online softmax update for this thread's score
        if (score > thread_max) {
            // Rescale previous sum to new max
            thread_sum *= exp(thread_max - score);
            thread_max = score;
        }
        if (score > -INFINITY) {
            thread_sum += exp(score - thread_max);
        }

        // Store score temporarily in P for final normalization
        if (use_registers) {
             if (score_idx < MAX_SCORES_REG) {
                 local_scores[score_idx++] = score;
             }
        } else {
             if (k_idx < seq_k) {
                 p_ptr[k_idx] = half(score);
             }
        }
    }

    // Threadgroup-wide reduction of max
    parallel_reduce_max_phase1(reduction_scratch, thread_max, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    parallel_reduce_max_phase2(reduction_scratch, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = reduction_scratch[0];

    // Each thread rescales its partial sum to the global max
    float rescaled_sum = thread_sum * exp(thread_max - global_max);

    // Threadgroup-wide reduction of sum
    parallel_reduce_sum_phase1(reduction_scratch, rescaled_sum, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    parallel_reduce_sum_phase2(reduction_scratch, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_sum = reduction_scratch[0];

    // Avoid division by zero for fully masked rows
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    // Single-pass: compute and write normalized values directly
    if (use_registers) {
        score_idx = 0;
        for (uint k_idx = tid; k_idx < seq_k; k_idx += THREADS_PER_TG_ATT) {
             float score = local_scores[score_idx++];
             float prob = exp(score - global_max) * inv_sum;
             p_ptr[k_idx] = half(prob);
        }
    } else {
        for (uint k_idx = tid; k_idx < seq_k; k_idx += THREADS_PER_TG_ATT) {
            float score = float(p_ptr[k_idx]);
            float prob = exp(score - global_max) * inv_sum;
            p_ptr[k_idx] = half(prob);
        }
    }
}

// ---------------------------------------------------------------------------
// Variant: fused attention with K tile in threadgroup memory
//
// For smaller seq_k and larger head_dim, loading K tiles into threadgroup
// memory reduces global memory bandwidth by allowing all threads to share
// the loaded K data. Trades threadgroup memory for bandwidth.
// ---------------------------------------------------------------------------

kernel void attention_qk_softmax_tiled(
    device const half* Q         [[buffer(0)]],
    device const half* K         [[buffer(1)]],
    device const half* mask      [[buffer(2)]],
    device half* P               [[buffer(3)]],
    constant uint& batch         [[buffer(4)]],
    constant uint& num_heads     [[buffer(5)]],
    constant uint& seq_q         [[buffer(6)]],
    constant uint& seq_k         [[buffer(7)]],
    constant uint& head_dim      [[buffer(8)]],
    constant float& scale        [[buffer(9)]],
    constant uint& mask_stride_q [[buffer(10)]],
    constant uint& causal        [[buffer(11)]],
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]]
) {
    uint q_row = tgid.x;
    uint head = tgid.y;
    uint batch_idx = tgid.z;

    if (q_row >= seq_q || head >= num_heads || batch_idx >= batch) return;

    // Tile size for K vectors loaded to threadgroup memory
    // With head_dim=128: 32 * 128 * 2B = 8192 bytes for K tile
    //                     128 * 2B = 256 bytes for Q cache
    //                     Total ~8.5KB, well within 32KB budget

    uint q_offset = ((batch_idx * num_heads + head) * seq_q + q_row) * head_dim;
    device const half* q_ptr = Q + q_offset;
    uint k_base = (batch_idx * num_heads + head) * seq_k * head_dim;
    device const half* k_base_ptr = K + k_base;
    uint p_offset = ((batch_idx * num_heads + head) * seq_q + q_row) * seq_k;
    device half* p_ptr = P + p_offset;

    device const half* mask_row = nullptr;
    if (!causal && mask != nullptr) {
        mask_row = mask + q_row * mask_stride_q;
    }

    // Threadgroup memory
    threadgroup half Q_cache_t[HEAD_DIM_MAX];
    threadgroup half K_tile[K_TILE_TILED][HEAD_DIM_MAX];
    threadgroup float reduction_scratch_t[SIMDGROUPS_ATT];

    // Load Q vector cooperatively
    for (uint i = tid; i < head_dim; i += THREADS_PER_TG_ATT) {
        Q_cache_t[i] = q_ptr[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float thread_max = -INFINITY;
    float thread_sum = 0.0f;

    uint num_k_tiles = (seq_k + K_TILE_TILED - 1) / K_TILE_TILED;

    for (uint tile = 0; tile < num_k_tiles; ++tile) {
        uint k_start = tile * K_TILE_TILED;
        uint k_end = min(k_start + K_TILE_TILED, seq_k);
        uint tile_len = k_end - k_start;

        // Cooperatively load K tile into threadgroup memory
        // Each thread loads multiple elements to fill [tile_len][head_dim]
        uint total_elements = tile_len * head_dim;
        for (uint i = tid; i < total_elements; i += THREADS_PER_TG_ATT) {
            uint row = i / head_dim;
            uint col = i % head_dim;
            uint global_k_idx = k_start + row;
            if (global_k_idx < seq_k) {
                K_tile[row][col] = k_base_ptr[global_k_idx * head_dim + col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each thread computes dot products for a subset of K vectors in this tile
        for (uint local_k = tid; local_k < tile_len; local_k += THREADS_PER_TG_ATT) {
            uint k_idx = k_start + local_k;

            float score;
            if (causal && k_idx > q_row) {
                score = -INFINITY;
            } else {
                // Dot product from threadgroup memory (no global loads)
                float dot = 0.0f;
                uint d = 0;
                for (; d + 3 < head_dim; d += 4) {
                    dot += float(Q_cache_t[d])     * float(K_tile[local_k][d])
                         + float(Q_cache_t[d + 1]) * float(K_tile[local_k][d + 1])
                         + float(Q_cache_t[d + 2]) * float(K_tile[local_k][d + 2])
                         + float(Q_cache_t[d + 3]) * float(K_tile[local_k][d + 3]);
                }
                for (; d < head_dim; ++d) {
                    dot += float(Q_cache_t[d]) * float(K_tile[local_k][d]);
                }

                score = dot * scale;

                if (!causal && mask_row != nullptr) {
                    score += float(mask_row[k_idx]);
                }
            }

            // Online softmax accumulation
            if (score > thread_max) {
                thread_sum *= exp(thread_max - score);
                thread_max = score;
            }
            if (score > -INFINITY) {
                thread_sum += exp(score - thread_max);
            }

            // Store raw score for normalization pass
            p_ptr[k_idx] = half(score);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Reduce max and sum across threadgroup
    parallel_reduce_max_phase1(reduction_scratch_t, thread_max, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    parallel_reduce_max_phase2(reduction_scratch_t, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = reduction_scratch_t[0];

    float rescaled_sum = thread_sum * exp(thread_max - global_max);

    parallel_reduce_sum_phase1(reduction_scratch_t, rescaled_sum, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    parallel_reduce_sum_phase2(reduction_scratch_t, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_sum = reduction_scratch_t[0];
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    // Normalize and write probabilities
    for (uint k_idx = tid; k_idx < seq_k; k_idx += THREADS_PER_TG_ATT) {
        float score = float(p_ptr[k_idx]);
        float prob = exp(score - global_max) * inv_sum;
        p_ptr[k_idx] = half(prob);
    }
}

// ---------------------------------------------------------------------------
// Attention output kernel: O = P @ V
//
// P: [batch, heads, seq_q, seq_k]
// V: [batch, heads, seq_k, head_dim]
// O: [batch, heads, seq_q, head_dim]
//
// Dispatch: one threadgroup per (batch, head, query_row)
// Grid: [seq_q, num_heads, batch]
// Threadgroup: [THREADS_PER_TG_ATT, 1, 1]
// ---------------------------------------------------------------------------

kernel void attention_pv(
    device const half* P         [[buffer(0)]],
    device const half* V         [[buffer(1)]],
    device half* O               [[buffer(2)]],
    constant uint& batch         [[buffer(3)]],
    constant uint& num_heads     [[buffer(4)]],
    constant uint& seq_q         [[buffer(5)]],
    constant uint& seq_k         [[buffer(6)]],
    constant uint& head_dim      [[buffer(7)]],
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]]
) {
    uint q_row = tgid.x;
    uint head = tgid.y;
    uint batch_idx = tgid.z;

    if (q_row >= seq_q || head >= num_heads || batch_idx >= batch) return;

    uint p_offset = ((batch_idx * num_heads + head) * seq_q + q_row) * seq_k;
    uint v_base = (batch_idx * num_heads + head) * seq_k * head_dim;
    uint o_offset = ((batch_idx * num_heads + head) * seq_q + q_row) * head_dim;

    device const half* p_ptr = P + p_offset;
    device const half* v_ptr = V + v_base;
    device half* o_ptr = O + o_offset;

    // Collaborative reduction over seq_k for each output element d
    // Each element d is handled by (THREADS_PER_TG_ATT / head_dim) threads
    uint threads_per_d = THREADS_PER_TG_ATT / head_dim;
    if (threads_per_d > 1) {
        uint d = tid / threads_per_d;
        uint lane_in_d = tid % threads_per_d;
        
        if (d < head_dim) {
            float acc = 0.0f;
            for (uint k = lane_in_d; k < seq_k; k += threads_per_d) {
                float p = float(p_ptr[k]);
                float v = float(v_ptr[k * head_dim + d]);
                acc += p * v;
            }
            
            // Logarithmic reduction within SIMD group for this element d
            for (uint offset = threads_per_d / 2; offset > 0; offset /= 2) {
                acc += simd_shuffle_down(acc, offset);
            }
            
            if (lane_in_d == 0) {
                o_ptr[d] = half(acc);
            }
        }
    } else {
        for (uint i = 0; i < O_PER_THREAD; ++i) {
            uint d = tid + i * THREADS_PER_TG_ATT;
            if (d >= head_dim) continue;

            float acc = 0.0f;
            for (uint k = 0; k < seq_k; ++k) {
                float p = float(p_ptr[k]);
                float v = float(v_ptr[k * head_dim + d]);
                acc += p * v;
            }
            o_ptr[d] = half(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// Single-pass fused Q@K^T + softmax kernel
//
// Computes: P = softmax((Q @ K^T) * scale + mask)
// in a single pass without writing intermediate scores to global memory.
//
// Key insight: Keep scores in registers during the online softmax pass,
// then write final probabilities in a single cooperative store.
//
// For seq_k <= TILE_K_ATT (128), each thread handles 1 score in registers.
// For longer sequences, we process tiles and accumulate statistics, then
// write the normalized probabilities directly without intermediate storage.
//
// Memory traffic (vs two-pass):
//   Two-pass: read Q, read K, write scores, read scores, write probs
//   Single-pass: read Q, read K, write probs
//   Savings: 2 * seq_k * sizeof(half) per query row = 256 bytes for seq_k=128
//
// Dispatch: one threadgroup per (batch, head, query_row) triple.
// ---------------------------------------------------------------------------

kernel void attention_qk_softmax_fused(
    device const half* Q         [[buffer(0)]],   // [batch, heads, seq_q, head_dim]
    device const half* K         [[buffer(1)]],   // [batch, heads, seq_k, head_dim]
    device const half* mask      [[buffer(2)]],   // [seq_q, seq_k] or nullptr
    device half* P               [[buffer(3)]],   // [batch, heads, seq_q, seq_k]
    constant uint& batch         [[buffer(4)]],
    constant uint& num_heads     [[buffer(5)]],
    constant uint& seq_q         [[buffer(6)]],
    constant uint& seq_k         [[buffer(7)]],
    constant uint& head_dim      [[buffer(8)]],
    constant float& scale        [[buffer(9)]],
    constant uint& mask_stride_q [[buffer(10)]],
    constant uint& causal        [[buffer(11)]],
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]]
) {
    uint q_row = tgid.x;
    uint head = tgid.y;
    uint batch_idx = tgid.z;

    if (q_row >= seq_q || head >= num_heads || batch_idx >= batch) return;

    // Each thread computes scores for its assigned K indices
    // Thread tid handles k_idx = tid, tid + THREADS_PER_TG_ATT, tid + 2*THREADS_PER_TG_ATT, ...
    // Max scores per thread: ceil(seq_k / THREADS_PER_TG_ATT)
    // For seq_k=4096, threads=128: 32 scores per thread (128 bytes of registers)

    uint q_offset = ((batch_idx * num_heads + head) * seq_q + q_row) * head_dim;
    device const half* q_ptr = Q + q_offset;
    uint k_base = (batch_idx * num_heads + head) * seq_k * head_dim;
    device const half* k_base_ptr = K + k_base;
    uint p_offset = ((batch_idx * num_heads + head) * seq_q + q_row) * seq_k;
    device half* p_ptr = P + p_offset;

    device const half* mask_row = nullptr;
    if (!causal && mask != nullptr) {
        mask_row = mask + q_row * mask_stride_q;
    }

    // Threadgroup memory for Q cache and reductions
    threadgroup half Q_cache_fused[HEAD_DIM_MAX];
    threadgroup float reduction_scratch_fused[SIMDGROUPS_ATT];

    // Load Q vector cooperatively
    for (uint i = tid; i < head_dim; i += THREADS_PER_TG_ATT) {
        Q_cache_fused[i] = q_ptr[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Single-pass approach:
    //
    // Phase 1: Each thread computes its scores into registers and accumulates
    //          thread-local online softmax statistics (max, sum).
    //
    // Phase 2: Threadgroup-wide reduction to get global max and sum.
    //
    // Phase 3: Each thread normalizes its scores and writes directly to P.
    //
    // Register budget: max 32 scores per thread for seq_k=4096
    // -------------------------------------------------------------------------

    // Maximum scores per thread (compile-time constant)
    constexpr uint MAX_SCORES_PER_THREAD = 64;  // Support seq_k up to 8192

    // Thread-local score storage
    float thread_scores[MAX_SCORES_PER_THREAD];
    uint num_scores = 0;

    // Online softmax state
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;

    // Compute all scores for this thread's assigned K indices
    for (uint k_idx = tid; k_idx < seq_k; k_idx += THREADS_PER_TG_ATT) {
        float score;

        if (causal && k_idx > q_row) {
            score = -INFINITY;
        } else {
            // Dot product: Q_cache[:] · K[k_idx, :]
            device const half* k_vec = k_base_ptr + k_idx * head_dim;
            float dot = 0.0f;

            // Unrolled dot product
            uint d = 0;
            for (; d + 3 < head_dim; d += 4) {
                float q0 = float(Q_cache_fused[d]);
                float q1 = float(Q_cache_fused[d + 1]);
                float q2 = float(Q_cache_fused[d + 2]);
                float q3 = float(Q_cache_fused[d + 3]);
                float k0 = float(k_vec[d]);
                float k1 = float(k_vec[d + 1]);
                float k2 = float(k_vec[d + 2]);
                float k3 = float(k_vec[d + 3]);
                dot += q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
            }
            for (; d < head_dim; ++d) {
                dot += float(Q_cache_fused[d]) * float(k_vec[d]);
            }

            score = dot * scale;

            // Apply mask
            if (!causal && mask_row != nullptr) {
                score += float(mask_row[k_idx]);
            }
        }

        // Store in thread-local array
        if (num_scores < MAX_SCORES_PER_THREAD) {
            thread_scores[num_scores++] = score;
        }

        // Update online softmax statistics
        if (score > thread_max) {
            thread_sum *= exp(thread_max - score);
            thread_max = score;
        }
        if (score > -INFINITY) {
            thread_sum += exp(score - thread_max);
        }
    }

    // Threadgroup reduction for global max
    parallel_reduce_max_phase1(reduction_scratch_fused, thread_max, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    parallel_reduce_max_phase2(reduction_scratch_fused, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = reduction_scratch_fused[0];

    // Rescale thread sum to global max
    float rescaled_sum = thread_sum * exp(thread_max - global_max);

    // Threadgroup reduction for global sum
    parallel_reduce_sum_phase1(reduction_scratch_fused, rescaled_sum, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    parallel_reduce_sum_phase2(reduction_scratch_fused, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_sum = reduction_scratch_fused[0];

    // Compute inverse sum for normalization
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    // Write normalized probabilities directly to P (single pass over P buffer)
    uint score_idx = 0;
    for (uint k_idx = tid; k_idx < seq_k; k_idx += THREADS_PER_TG_ATT) {
        float score = thread_scores[score_idx++];
        float prob = exp(score - global_max) * inv_sum;
        p_ptr[k_idx] = half(prob);
    }
}

// ---------------------------------------------------------------------------
// Variant: attention with output accumulation (fused QK softmax @ V)
//
// Computes: O = softmax(Q @ K^T * scale + mask) @ V
// in a single kernel, avoiding writing the full P matrix to memory.
//
// Uses the online softmax trick with rescaling of partial V accumulations:
//   As max and sum are updated per K-tile, the running O accumulator
//   is rescaled to account for the new normalization constant.
//
// Output: O [batch, heads, seq_q, head_dim]
// ---------------------------------------------------------------------------

kernel void attention_fused_qkv(
    device const half* Q         [[buffer(0)]],   // [batch, heads, seq_q, head_dim]
    device const half* K         [[buffer(1)]],   // [batch, heads, seq_k, head_dim]
    device const half* V         [[buffer(2)]],   // [batch, heads, seq_k, head_dim]
    device const half* mask      [[buffer(3)]],   // [seq_q, seq_k] or nullptr
    device half* O               [[buffer(4)]],   // [batch, heads, seq_q, head_dim]
    constant uint& batch         [[buffer(5)]],
    constant uint& num_heads     [[buffer(6)]],
    constant uint& seq_q         [[buffer(7)]],
    constant uint& seq_k         [[buffer(8)]],
    constant uint& head_dim      [[buffer(9)]],
    constant float& scale        [[buffer(10)]],
    constant uint& mask_stride_q [[buffer(11)]],
    constant uint& causal        [[buffer(12)]],
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]]
) {
    uint q_row = tgid.x;
    uint head = tgid.y;
    uint batch_idx = tgid.z;

    if (q_row >= seq_q || head >= num_heads || batch_idx >= batch) return;

    // K tile size for threadgroup loading
    // Budget: Q_cache (128*2=256B) + K_tile (16*128*2=4096B) + V_tile (16*128*2=4096B)
    //       + O_accum (128*4=512B per thread... too much for threadgroup)
    //       Total threadgroup: ~8.9KB, fine
    // O accumulator lives in thread-private registers (head_dim floats per thread)

    uint q_offset = ((batch_idx * num_heads + head) * seq_q + q_row) * head_dim;
    device const half* q_ptr = Q + q_offset;
    uint kv_base = (batch_idx * num_heads + head) * seq_k * head_dim;
    device const half* k_base_ptr = K + kv_base;
    device const half* v_base_ptr = V + kv_base;
    uint o_offset = ((batch_idx * num_heads + head) * seq_q + q_row) * head_dim;
    device half* o_ptr = O + o_offset;

    device const half* mask_row = nullptr;
    if (!causal && mask != nullptr) {
        mask_row = mask + q_row * mask_stride_q;
    }

    // Threadgroup memory
    threadgroup half Q_cache_f[HEAD_DIM_MAX];
    threadgroup half K_tile_f[KV_TILE_FUSED][HEAD_DIM_MAX];
    threadgroup half V_tile_f[KV_TILE_FUSED][HEAD_DIM_MAX];
    // Shared scores for V accumulation (all threads need all scores in tile)
    threadgroup float tile_scores[KV_TILE_FUSED];

    // Load Q vector
    for (uint i = tid; i < head_dim; i += THREADS_PER_TG_ATT) {
        Q_cache_f[i] = q_ptr[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-thread O accumulator (in registers, one float per head_dim element)
    // Each thread accumulates a strided subset of head_dim
    // Thread tid handles dimensions: tid, tid+THREADS, tid+2*THREADS, ...
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // O accumulator: each thread stores head_dim / THREADS_PER_TG_ATT values
    // For head_dim=128, threads=128: 1 value per thread
    // For head_dim=64, threads=128: some threads idle for O (but still do scores)
    float o_accum[O_PER_THREAD] = {0.0f};

    uint num_kv_tiles = (seq_k + KV_TILE_FUSED - 1) / KV_TILE_FUSED;

    for (uint tile = 0; tile < num_kv_tiles; ++tile) {
        uint k_start = tile * KV_TILE_FUSED;
        uint k_end = min(k_start + KV_TILE_FUSED, seq_k);
        uint tile_len = k_end - k_start;

        // Cooperatively load K and V tiles
        uint total_k = tile_len * head_dim;
        for (uint i = tid; i < total_k; i += THREADS_PER_TG_ATT) {
            uint row = i / head_dim;
            uint col = i % head_dim;
            uint gk = k_start + row;
            K_tile_f[row][col] = (gk < seq_k) ? k_base_ptr[gk * head_dim + col] : half(0);
            V_tile_f[row][col] = (gk < seq_k) ? v_base_ptr[gk * head_dim + col] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute scores for all K vectors in this tile
        // We need all threads to agree on the scores, so we cooperatively
        // compute them and store in threadgroup memory
        for (uint local_k = tid; local_k < tile_len; local_k += THREADS_PER_TG_ATT) {
            uint k_idx = k_start + local_k;
            float score;
            if (causal && k_idx > q_row) {
                score = -INFINITY;
            } else {
                float dot = 0.0f;
                for (uint d = 0; d < head_dim; ++d) {
                    dot += float(Q_cache_f[d]) * float(K_tile_f[local_k][d]);
                }
                score = dot * scale;
                if (!causal && mask_row != nullptr) {
                    score += float(mask_row[k_idx]);
                }
            }
            tile_scores[local_k] = score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Efficiently find max in this tile using SIMD group functions
        float tile_max = (tid < tile_len) ? tile_scores[tid] : -INFINITY;
        tile_max = simd_max(tile_max);
        
        // Broadcast from first simdgroup to all others using a single shared location
        if (tid == 0) tile_scores[0] = tile_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_max = tile_scores[0];

        // Update running max and rescale previous accumulator
        float prev_max = running_max;
        running_max = max(running_max, tile_max);

        if (prev_max > -INFINITY) {
            float rescale = exp(prev_max - running_max);
            running_sum *= rescale;
            for (uint i = 0; i < O_PER_THREAD; ++i) {
                o_accum[i] *= rescale;
            }
        }

        // Accumulate: for each K vector in tile, compute weight and add weighted V
        for (uint local_k = 0; local_k < tile_len; ++local_k) {
            float s = tile_scores[local_k];
            if (s <= -INFINITY) continue;
            float w = exp(s - running_max);
            running_sum += w;

            // Each thread accumulates its portion of head_dim
            for (uint i = 0; i < O_PER_THREAD; ++i) {
                uint d = tid + i * THREADS_PER_TG_ATT;
                if (d < head_dim) {
                    o_accum[i] += w * float(V_tile_f[local_k][d]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write output
    float inv_sum_f = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    for (uint i = 0; i < O_PER_THREAD; ++i) {
        uint d = tid + i * THREADS_PER_TG_ATT;
        if (d < head_dim) {
            o_ptr[d] = half(o_accum[i] * inv_sum_f);
        }
    }
}

// ---------------------------------------------------------------------------
// Block-sparse attention kernel: Q @ K^T + scale + block-sparse mask + softmax
//
// Block-sparse attention computes attention only for active blocks in a
// sparse pattern. The mask is represented as a compact bitset where each bit
// corresponds to one (BLOCK_Q x BLOCK_K) block.
//
// Memory layout:
//   Q, K, V: same as dense attention
//   mask_bits: [num_q_blocks] where each element is a uint64_t bitset
//              Bit j in mask_bits[i] is set if block (i, j) is active
//   P: output attention weights (sparse structure)
//
// Active block detection:
//   uint64_t mask_block = mask_bits[q_block_idx];
//   bool is_active = (mask_block & (1ULL << k_block_idx)) != 0;
//
// This pattern enables:
//   - Sliding window attention: diagonal blocks only
//   - BigBird: combination of local, global, and random blocks
//   - BigBird/Longformer-style sparse patterns
//
// Parameters:
//   block_q: block size in query dimension (rows per block)
//   block_k: block size in key dimension (columns per block)
//   num_q_blocks: total number of query blocks = (seq_q + block_q - 1) / block_q
//   num_k_blocks: total number of key blocks = (seq_k + block_k - 1) / block_k
//   causal: if 1, apply causal masking (only attend to past positions)
//
// Dispatch: one threadgroup per (batch, head, q_block) pair.
//   Grid: [num_q_blocks, num_heads, batch]
//   Each threadgroup processes all K blocks for one Q block.
//
// Implementation notes:
//   - The kernel uses a two-pass approach: compute raw scores into P buffer,
//     then normalize. This avoids complex score bookkeeping.
//   - Scores for inactive blocks are set to -INFINITY, which becomes 0 after softmax.
//   - For block-sparse patterns, we skip dot products for inactive blocks.
// ---------------------------------------------------------------------------

kernel void attention_block_sparse_qk_softmax(
    device const half* Q          [[buffer(0)]],   // [batch, heads, seq_q, head_dim]
    device const half* K          [[buffer(1)]],   // [batch, heads, seq_k, head_dim]
    device const uint64_t* mask_bits [[buffer(2)]], // [num_q_blocks] - bitset of active K blocks per Q block
    device half* P                [[buffer(3)]],   // [batch, heads, seq_q, seq_k]
    constant uint& batch          [[buffer(4)]],
    constant uint& num_heads      [[buffer(5)]],
    constant uint& seq_q          [[buffer(6)]],
    constant uint& seq_k          [[buffer(7)]],
    constant uint& head_dim       [[buffer(8)]],
    constant float& scale         [[buffer(9)]],   // 1/sqrt(head_dim)
    constant uint& block_q        [[buffer(10)]],  // Block size in Q dimension
    constant uint& block_k        [[buffer(11)]],  // Block size in K dimension
    constant uint& num_q_blocks   [[buffer(12)]],  // Total Q blocks
    constant uint& num_k_blocks   [[buffer(13)]],  // Total K blocks
    constant uint& causal          [[buffer(14)]],  // Causal masking flag
    uint3 tgid                    [[threadgroup_position_in_grid]],
    uint tid                      [[thread_index_in_threadgroup]]
) {
    // Decode grid position: (q_block_idx, head, batch_idx)
    uint q_block_idx = tgid.x;
    uint head = tgid.y;
    uint batch_idx = tgid.z;

    if (q_block_idx >= num_q_blocks || head >= num_heads || batch_idx >= batch) return;

    // Compute query row range for this block
    uint q_start = q_block_idx * block_q;
    uint q_end = min(q_start + block_q, seq_q);
    uint q_block_len = q_end - q_start;

    // Get mask bits for this Q block (bitset of which K blocks are active)
    uint64_t active_mask = mask_bits[q_block_idx];

    // Base pointers for this batch and head
    uint q_base = (batch_idx * num_heads + head) * seq_q * head_dim;
    uint k_base = (batch_idx * num_heads + head) * seq_k * head_dim;

    // Threadgroup memory
    threadgroup half Q_cache[BLOCK_Q_DEFAULT][HEAD_DIM_MAX];
    threadgroup half K_cache[BLOCK_K_DEFAULT][HEAD_DIM_MAX];
    threadgroup float reduction_scratch[SIMDGROUPS_ATT];

    // -------------------------------------------------------------------------
    // Phase 1: Load Q vectors for all rows in this block into threadgroup memory
    // -------------------------------------------------------------------------
    for (uint q_local = 0; q_local < q_block_len; ++q_local) {
        uint q_row = q_start + q_local;
        uint q_offset = q_base + q_row * head_dim;
        for (uint d = tid; d < head_dim; d += THREADS_PER_TG_ATT) {
            Q_cache[q_local][d] = Q[q_offset + d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Phase 2: Process each query row - compute scores, find max and sum
    // -------------------------------------------------------------------------
    // We process each query row separately for cleaner online softmax logic
    for (uint q_local = 0; q_local < q_block_len; ++q_local) {
        uint q_row = q_start + q_local;
        uint p_offset = ((batch_idx * num_heads + head) * seq_q + q_row) * seq_k;

        // Initialize thread-local online softmax state
        float thread_max = -INFINITY;
        float thread_sum = 0.0f;

        // Compute scores for active K blocks
        for (uint k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
            // Check if this K block is active via bitset
            bool is_active = (active_mask & (1ULL << k_block_idx)) != 0;

            if (!is_active) {
                // Write -INFINITY for inactive blocks (will become 0 after softmax)
                uint k_start = k_block_idx * block_k;
                uint k_end = min(k_start + block_k, seq_k);
                for (uint k_idx = k_start + tid; k_idx < k_end; k_idx += THREADS_PER_TG_ATT) {
                    P[p_offset + k_idx] = half(-INFINITY);
                }
                continue;
            }

            // This is an active block - compute scores
            uint k_start = k_block_idx * block_k;
            uint k_end = min(k_start + block_k, seq_k);
            uint k_block_len = k_end - k_start;

            // Load K vectors for this block into threadgroup memory
            for (uint k_local = 0; k_local < k_block_len; ++k_local) {
                uint k_idx = k_start + k_local;
                uint k_offset = k_base + k_idx * head_dim;
                for (uint d = tid; d < head_dim; d += THREADS_PER_TG_ATT) {
                    K_cache[k_local][d] = K[k_offset + d];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute scores for all K positions in this block
            for (uint k_local = tid; k_local < k_block_len; k_local += THREADS_PER_TG_ATT) {
                uint k_idx = k_start + k_local;
                float score = -INFINITY;

                // Apply causal mask if enabled
                if (causal && k_idx > q_row) {
                    score = -INFINITY;
                } else {
                    // Dot product
                    float dot = 0.0f;
                    for (uint d = 0; d < head_dim; ++d) {
                        dot += float(Q_cache[q_local][d]) * float(K_cache[k_local][d]);
                    }
                    score = dot * scale;
                }

                // Store raw score in P buffer
                P[p_offset + k_idx] = half(score);

                // Online softmax update
                if (score > thread_max) {
                    thread_sum *= exp(thread_max - score);
                    thread_max = score;
                }
                if (score > -INFINITY) {
                    thread_sum += exp(score - thread_max);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // -------------------------------------------------------------------------
        // Phase 3: Threadgroup reduction for global max and sum
        // -------------------------------------------------------------------------
        parallel_reduce_max_phase1(reduction_scratch, thread_max, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        parallel_reduce_max_phase2(reduction_scratch, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float global_max = reduction_scratch[0];

        // Rescale thread sum to global max
        float rescaled_sum = thread_sum * exp(thread_max - global_max);

        parallel_reduce_sum_phase1(reduction_scratch, rescaled_sum, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        parallel_reduce_sum_phase2(reduction_scratch, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float global_sum = reduction_scratch[0];
        float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

        // -------------------------------------------------------------------------
        // Phase 4: Normalize scores and write final probabilities
        // -------------------------------------------------------------------------
        for (uint k_idx = tid; k_idx < seq_k; k_idx += THREADS_PER_TG_ATT) {
            float score = float(P[p_offset + k_idx]);
            float prob = exp(score - global_max) * inv_sum;
            P[p_offset + k_idx] = half(prob);
        }
    }
}

// ---------------------------------------------------------------------------
// Block-sparse fused attention: Q @ K^T + softmax @ V with block-sparse mask
//
// Full attention in a single kernel with block-sparse attention pattern.
// Computes: O = softmax_block_sparse(Q @ K^T * scale, mask) @ V
//
// This avoids materializing the full P matrix, only computing attention
// for active blocks. The V accumulation is also block-sparse.
//
// Memory layout:
//   mask_bits: same as attention_block_sparse_qk_softmax
//   O: output attention values (dense) [batch, heads, seq_q, head_dim]
//
// Parameters:
//   causal: if 1, apply causal masking (only attend to past positions)
//
// Dispatch: one threadgroup per (batch, head, q_block) pair.
//
// Implementation notes:
//   - Uses online softmax with rescaling of V accumulators.
//   - Inactive blocks are skipped entirely.
//   - Supports causal masking for autoregressive generation.
// ---------------------------------------------------------------------------

kernel void attention_block_sparse_fused_qkv(
    device const half* Q          [[buffer(0)]],   // [batch, heads, seq_q, head_dim]
    device const half* K          [[buffer(1)]],   // [batch, heads, seq_k, head_dim]
    device const half* V          [[buffer(2)]],   // [batch, heads, seq_k, head_dim]
    device const uint64_t* mask_bits [[buffer(3)]], // [num_q_blocks] - bitset of active K blocks per Q block
    device half* O                [[buffer(4)]],   // [batch, heads, seq_q, head_dim]
    constant uint& batch          [[buffer(5)]],
    constant uint& num_heads      [[buffer(6)]],
    constant uint& seq_q          [[buffer(7)]],
    constant uint& seq_k          [[buffer(8)]],
    constant uint& head_dim       [[buffer(9)]],
    constant float& scale         [[buffer(10)]],
    constant uint& block_q        [[buffer(11)]],
    constant uint& block_k        [[buffer(12)]],
    constant uint& num_q_blocks   [[buffer(13)]],
    constant uint& num_k_blocks   [[buffer(14)]],
    constant uint& causal         [[buffer(15)]],   // Causal masking flag
    uint3 tgid                    [[threadgroup_position_in_grid]],
    uint tid                      [[thread_index_in_threadgroup]]
) {
    uint q_block_idx = tgid.x;
    uint head = tgid.y;
    uint batch_idx = tgid.z;

    if (q_block_idx >= num_q_blocks || head >= num_heads || batch_idx >= batch) return;

    uint q_start = q_block_idx * block_q;
    uint q_end = min(q_start + block_q, seq_q);
    uint q_block_len = q_end - q_start;

    uint64_t active_mask = mask_bits[q_block_idx];

    // Initialize output to zero (all query rows in this block)
    for (uint q_local = 0; q_local < q_block_len; ++q_local) {
        uint q_row = q_start + q_local;
        uint o_offset = ((batch_idx * num_heads + head) * seq_q + q_row) * head_dim;
        for (uint d = tid; d < head_dim; d += THREADS_PER_TG_ATT) {
            O[o_offset + d] = half(0.0f);
        }
    }

    if (active_mask == 0) return;

    uint q_base = (batch_idx * num_heads + head) * seq_q * head_dim;
    uint k_base = (batch_idx * num_heads + head) * seq_k * head_dim;
    uint v_base = (batch_idx * num_heads + head) * seq_k * head_dim;

    threadgroup half Q_cache[BLOCK_Q_DEFAULT][HEAD_DIM_MAX];
    threadgroup half K_cache[BLOCK_K_DEFAULT][HEAD_DIM_MAX];
    threadgroup half V_cache[BLOCK_K_DEFAULT][HEAD_DIM_MAX];
    threadgroup float reduction_scratch[SIMDGROUPS_ATT];
    threadgroup float tile_scores[BLOCK_K_DEFAULT];  // Shared scores for V accumulation

    // Load Q vectors
    for (uint q_local = 0; q_local < q_block_len; ++q_local) {
        uint q_row = q_start + q_local;
        uint q_offset = q_base + q_row * head_dim;
        for (uint d = tid; d < head_dim; d += THREADS_PER_TG_ATT) {
            Q_cache[q_local][d] = Q[q_offset + d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-query accumulators for online softmax and V accumulation
    float running_max[BLOCK_Q_DEFAULT];
    float running_sum[BLOCK_Q_DEFAULT];
    float o_accum[BLOCK_Q_DEFAULT][O_PER_THREAD];

    for (uint i = 0; i < q_block_len; ++i) {
        running_max[i] = -INFINITY;
        running_sum[i] = 0.0f;
        for (uint j = 0; j < O_PER_THREAD; ++j) {
            o_accum[i][j] = 0.0f;
        }
    }

    // Process each K block
    for (uint k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
        bool is_active = (active_mask & (1ULL << k_block_idx)) != 0;

        if (!is_active) continue;

        uint k_start = k_block_idx * block_k;
        uint k_end = min(k_start + block_k, seq_k);
        uint k_block_len = k_end - k_start;

        // Load K and V tiles
        for (uint k_local = 0; k_local < k_block_len; ++k_local) {
            uint k_idx = k_start + k_local;
            uint k_offset = k_base + k_idx * head_dim;
            uint v_offset = v_base + k_idx * head_dim;
            for (uint d = tid; d < head_dim; d += THREADS_PER_TG_ATT) {
                K_cache[k_local][d] = K[k_offset + d];
                V_cache[k_local][d] = V[v_offset + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Process each query row in this block
        for (uint q_local = 0; q_local < q_block_len; ++q_local) {
            uint q_row = q_start + q_local;

            // Compute scores for all K in this block
            // All threads need all scores for V accumulation, so we cooperatively
            // compute and store in threadgroup memory
            float tile_max = -INFINITY;

            for (uint k_local = tid; k_local < k_block_len; k_local += THREADS_PER_TG_ATT) {
                uint k_idx = k_start + k_local;
                float score = -INFINITY;

                // Apply causal mask if enabled
                if (causal && k_idx > q_row) {
                    score = -INFINITY;
                } else {
                    float dot = 0.0f;
                    for (uint d = 0; d < head_dim; ++d) {
                        dot += float(Q_cache[q_local][d]) * float(K_cache[k_local][d]);
                    }
                    score = dot * scale;
                }

                tile_scores[k_local] = score;
                if (score > tile_max) {
                    tile_max = score;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Efficiently find max in this tile using SIMD group functions
            float local_max = (tid < k_block_len) ? tile_scores[tid] : -INFINITY;
            float global_tile_max = simd_max(local_max);

            // Broadcast tile max to all threads
            if (tid == 0) {
                tile_scores[0] = global_tile_max;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            tile_max = tile_scores[0];

            // Update running max and rescale
            float prev_max = running_max[q_local];
            running_max[q_local] = max(running_max[q_local], tile_max);

            if (prev_max > -INFINITY) {
                float rescale = exp(prev_max - running_max[q_local]);
                running_sum[q_local] *= rescale;
                for (uint i = 0; i < O_PER_THREAD; ++i) {
                    o_accum[q_local][i] *= rescale;
                }
            }

            // Accumulate weighted V for all K in this block
            for (uint k_local = 0; k_local < k_block_len; ++k_local) {
                uint k_idx = k_start + k_local;
                float score = tile_scores[k_local];

                if (score <= -INFINITY) continue;

                float w = exp(score - running_max[q_local]);
                running_sum[q_local] += w;

                for (uint i = 0; i < O_PER_THREAD; ++i) {
                    uint d = tid + i * THREADS_PER_TG_ATT;
                    if (d < head_dim) {
                        o_accum[q_local][i] += w * float(V_cache[k_local][d]);
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write output
    for (uint q_local = 0; q_local < q_block_len; ++q_local) {
        float inv_sum = (running_sum[q_local] > 0.0f) ? (1.0f / running_sum[q_local]) : 0.0f;

        uint q_row = q_start + q_local;
        uint o_offset = ((batch_idx * num_heads + head) * seq_q + q_row) * head_dim;

        for (uint i = 0; i < O_PER_THREAD; ++i) {
            uint d = tid + i * THREADS_PER_TG_ATT;
            if (d < head_dim) {
                O[o_offset + d] = half(o_accum[q_local][i] * inv_sum);
            }
        }
    }
}
