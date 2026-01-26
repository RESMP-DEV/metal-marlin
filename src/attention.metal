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
// ---------------------------------------------------------------------------

constant constexpr uint TILE_K_ATT = 128;      // K vectors loaded per tile
constant constexpr uint THREADS_PER_TG_ATT = 128;  // 4 simdgroups
constant constexpr uint SIMDGROUPS_ATT = THREADS_PER_TG_ATT / 32;
constant constexpr uint HEAD_DIM_MAX = 128;

// Tile sizes for the tiled and fused variants
constant constexpr uint K_TILE_TILED = 32;    // K vectors per tile in tiled variant
constant constexpr uint KV_TILE_FUSED = 16;   // KV vectors per tile in fused variant
constant constexpr uint O_PER_THREAD = (HEAD_DIM_MAX + THREADS_PER_TG_ATT - 1) / THREADS_PER_TG_ATT;

// ---------------------------------------------------------------------------
// Utility: threadgroup-wide max and sum reductions
// ---------------------------------------------------------------------------

// Reduce max across entire threadgroup using simdgroup primitives + shared mem
inline float threadgroup_reduce_max(
    float val,
    uint tid,
    threadgroup float* scratch  // size >= SIMDGROUPS_ATT
) {
    // Step 1: simdgroup-level reduction
    float sg_max = simd_max(val);

    // Step 2: simd leaders write to shared memory
    uint sg_id = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) {
        scratch[sg_id] = sg_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: first simdgroup reduces across simdgroup leaders
    float result;
    if (sg_id == 0) {
        float v = (lane < SIMDGROUPS_ATT) ? scratch[lane] : -INFINITY;
        result = simd_max(v);
    }

    // Broadcast result from thread 0 to all threads
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        scratch[0] = result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

// Reduce sum across entire threadgroup
inline float threadgroup_reduce_sum(
    float val,
    uint tid,
    threadgroup float* scratch  // size >= SIMDGROUPS_ATT
) {
    float sg_sum = simd_sum(val);

    uint sg_id = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) {
        scratch[sg_id] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float result;
    if (sg_id == 0) {
        float v = (lane < SIMDGROUPS_ATT) ? scratch[lane] : 0.0f;
        result = simd_sum(v);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        scratch[0] = result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
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
    // Each thread handles (seq_k / THREADS_PER_TG_ATT) scores max
    // With 128 threads, for seq_k=4096 that's 32 scores per thread - feasible
    // For larger sequences we spill to global memory (P buffer)

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

        // Write raw score to P buffer for pass 2 normalization
        if (k_idx < seq_k) {
            p_ptr[k_idx] = half(score);
        }
    }

    // Threadgroup-wide reduction of max
    float global_max = threadgroup_reduce_max(thread_max, tid, reduction_scratch);

    // Each thread rescales its partial sum to the global max
    float rescaled_sum = thread_sum * exp(thread_max - global_max);

    // Threadgroup-wide reduction of sum
    float global_sum = threadgroup_reduce_sum(rescaled_sum, tid, reduction_scratch);

    // Avoid division by zero for fully masked rows
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    // Pass 2: normalize scores and write softmax probabilities
    for (uint k_idx = tid; k_idx < seq_k; k_idx += THREADS_PER_TG_ATT) {
        float score = float(p_ptr[k_idx]);  // Read back raw score
        float prob = exp(score - global_max) * inv_sum;
        p_ptr[k_idx] = half(prob);
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
    float global_max = threadgroup_reduce_max(thread_max, tid, reduction_scratch_t);
    float rescaled_sum = thread_sum * exp(thread_max - global_max);
    float global_sum = threadgroup_reduce_sum(rescaled_sum, tid, reduction_scratch_t);
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

        // Find max in this tile (all threads participate for speed)
        float tile_max = -INFINITY;
        for (uint i = 0; i < tile_len; ++i) {
            tile_max = max(tile_max, tile_scores[i]);
        }

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
