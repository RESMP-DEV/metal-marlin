// moe_router_sparse.metal - Sparse MoE router with learned candidate selection
//
// Optimization over moe_router.metal: Instead of computing router logits for ALL
// experts, compute only for a subset of CANDIDATE experts that a lightweight
// predictor has identified as likely to be selected.
//
// Computation savings:
//   - Standard router: O(batch * hidden_dim * num_experts)
//   - Sparse router: O(batch * hidden_dim * num_candidates)
//   - Typical reduction: 64 experts -> 16 candidates = 4x faster router GEMM
//
// The candidate_ids are provided by the SparseExpertPredictor (Python-side MLP)
// which learns token-to-expert mapping patterns during calibration.
//
// Input:
//   - hidden: [batch, hidden_dim] token hidden states
//   - router_weights: [hidden_dim, num_experts] full router weights
//   - candidate_ids: [batch, num_candidates] predicted candidate expert indices
//   - num_candidates: number of candidates per token (typically 8-16)
//
// Output:
//   - expert_ids: [batch, top_k] final selected expert indices
//   - expert_probs: [batch, top_k] renormalized routing weights
//
// Key optimization: The GEMM only touches num_candidates columns of router_weights,
// reducing memory bandwidth and compute proportionally.

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Maximum candidates to consider (power of 2 for efficient ops)
constant constexpr uint MAX_CANDIDATES = 32;

// Maximum top-k (must be <= MAX_CANDIDATES)
constant constexpr uint MAX_TOP_K = 16;

// Threads per threadgroup
constant constexpr uint SPARSE_ROUTER_THREADS = 64;  // Smaller since less work

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

inline uint div_ceil(uint a, uint b) {
    return (a + b - 1) / b;
}

inline float safe_exp(float x, float max_val) {
    float shifted = x - max_val;
    shifted = clamp(shifted, -88.0f, 88.0f);
    return exp(shifted);
}

// Simdgroup reductions
inline float simd_max_32(float val) {
    val = max(val, simd_shuffle_xor(val, 16));
    val = max(val, simd_shuffle_xor(val, 8));
    val = max(val, simd_shuffle_xor(val, 4));
    val = max(val, simd_shuffle_xor(val, 2));
    val = max(val, simd_shuffle_xor(val, 1));
    return val;
}

inline float simd_sum_32(float val) {
    val += simd_shuffle_xor(val, 16);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);
    return val;
}

// ---------------------------------------------------------------------------
// Sparse Router Kernel
// ---------------------------------------------------------------------------
// Each threadgroup handles one token.
// Computes router logits ONLY for candidate experts, not all experts.
//
// Grid: (batch_size, 1, 1)
// Threadgroup: (SPARSE_ROUTER_THREADS, 1, 1)
// ---------------------------------------------------------------------------

kernel void moe_router_sparse(
    device const half* hidden          [[buffer(0)]],   // [batch, hidden_dim]
    device const half* router_weights  [[buffer(1)]],   // [hidden_dim, num_experts]
    device const uint* candidate_ids   [[buffer(2)]],   // [batch, num_candidates]
    device uint* expert_ids            [[buffer(3)]],   // [batch, top_k] output
    device half* expert_probs          [[buffer(4)]],   // [batch, top_k] output
    constant uint& batch_size          [[buffer(5)]],
    constant uint& hidden_dim          [[buffer(6)]],
    constant uint& num_experts         [[buffer(7)]],
    constant uint& num_candidates      [[buffer(8)]],
    constant uint& top_k               [[buffer(9)]],
    uint tgid                          [[threadgroup_position_in_grid]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    uint batch_idx = tgid;
    if (batch_idx >= batch_size) return;

    // Pointers for this token
    device const half* h = hidden + batch_idx * hidden_dim;
    device const uint* candidates = candidate_ids + batch_idx * num_candidates;

    // Threadgroup memory for candidate logits and reductions
    threadgroup float logits_shared[MAX_CANDIDATES];
    threadgroup float max_shared[2];  // 2 simdgroups
    threadgroup float sum_shared[2];

    const uint num_simdgroups = SPARSE_ROUTER_THREADS / 32;  // 2

    // -------------------------------------------------------------------------
    // Step 1: Compute logits for CANDIDATE experts only
    // -------------------------------------------------------------------------
    // Each thread handles ceil(num_candidates / THREADS) candidates
    // This is the key optimization: we only compute num_candidates dot products
    // instead of num_experts dot products.

    uint candidates_per_thread = div_ceil(num_candidates, SPARSE_ROUTER_THREADS);

    for (uint c_iter = 0; c_iter < candidates_per_thread; ++c_iter) {
        uint c_idx = tid + c_iter * SPARSE_ROUTER_THREADS;
        if (c_idx >= num_candidates) break;

        // Get the actual expert ID for this candidate
        uint expert_id = candidates[c_idx];

        // Compute dot product: h @ router_weights[:, expert_id]
        // router_weights layout: [hidden_dim, num_experts] (column-major per expert)
        device const half* w_col = router_weights + expert_id;

        float acc = 0.0f;
        for (uint d = 0; d < hidden_dim; ++d) {
            acc += float(h[d]) * float(w_col[d * num_experts]);
        }

        logits_shared[c_idx] = acc;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Step 2: Softmax over candidates
    // -------------------------------------------------------------------------

    // Find max
    float local_max = -INFINITY;
    for (uint c = tid; c < num_candidates; c += SPARSE_ROUTER_THREADS) {
        local_max = max(local_max, logits_shared[c]);
    }

    local_max = simd_max_32(local_max);
    if (simd_lane == 0) {
        max_shared[simd_id] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max;
    if (tid == 0) {
        global_max = max_shared[0];
        for (uint s = 1; s < num_simdgroups; ++s) {
            global_max = max(global_max, max_shared[s]);
        }
        max_shared[0] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = max_shared[0];

    // Compute exp and sum
    float local_sum = 0.0f;
    for (uint c = tid; c < num_candidates; c += SPARSE_ROUTER_THREADS) {
        float exp_val = safe_exp(logits_shared[c], global_max);
        logits_shared[c] = exp_val;
        local_sum += exp_val;
    }

    local_sum = simd_sum_32(local_sum);
    if (simd_lane == 0) {
        sum_shared[simd_id] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum;
    if (tid == 0) {
        global_sum = sum_shared[0];
        for (uint s = 1; s < num_simdgroups; ++s) {
            global_sum += sum_shared[s];
        }
        sum_shared[0] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = sum_shared[0];

    // Normalize
    float inv_sum = 1.0f / global_sum;
    for (uint c = tid; c < num_candidates; c += SPARSE_ROUTER_THREADS) {
        logits_shared[c] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Step 3: Top-k selection from candidates
    // -------------------------------------------------------------------------
    // Thread 0 does serial top-k since k is small

    if (tid == 0) {
        float topk_vals[MAX_TOP_K];
        uint topk_candidate_indices[MAX_TOP_K];  // Index into candidates array

        for (uint i = 0; i < top_k; ++i) {
            topk_vals[i] = -INFINITY;
            topk_candidate_indices[i] = 0;
        }

        // Scan candidates
        for (uint c = 0; c < num_candidates; ++c) {
            float prob = logits_shared[c];

            if (prob > topk_vals[top_k - 1]) {
                // Find insertion point
                uint insert_pos = top_k;
                for (uint i = 0; i < top_k; ++i) {
                    if (prob > topk_vals[i]) {
                        insert_pos = i;
                        break;
                    }
                }

                // Shift down
                for (uint i = top_k - 1; i > insert_pos; --i) {
                    topk_vals[i] = topk_vals[i - 1];
                    topk_candidate_indices[i] = topk_candidate_indices[i - 1];
                }

                // Insert
                topk_vals[insert_pos] = prob;
                topk_candidate_indices[insert_pos] = c;
            }
        }

        // Renormalize and write outputs
        // Map candidate indices back to expert IDs
        float selected_sum = 0.0f;
        for (uint i = 0; i < top_k; ++i) {
            selected_sum += topk_vals[i];
        }
        float inv_selected_sum = 1.0f / max(selected_sum, 1e-8f);

        device uint* out_ids = expert_ids + batch_idx * top_k;
        device half* out_probs = expert_probs + batch_idx * top_k;

        for (uint i = 0; i < top_k; ++i) {
            // Convert candidate index to expert ID
            out_ids[i] = candidates[topk_candidate_indices[i]];
            out_probs[i] = half(topk_vals[i] * inv_selected_sum);
        }
    }
}

// ---------------------------------------------------------------------------
// Vectorized sparse router for better memory bandwidth
// ---------------------------------------------------------------------------
// Uses float4 loads for hidden state and accumulation.
// Requires hidden_dim to be divisible by 4.

kernel void moe_router_sparse_vec4(
    device const half4* hidden         [[buffer(0)]],   // [batch, hidden_dim/4]
    device const half* router_weights  [[buffer(1)]],   // [hidden_dim, num_experts]
    device const uint* candidate_ids   [[buffer(2)]],   // [batch, num_candidates]
    device uint* expert_ids            [[buffer(3)]],   // [batch, top_k] output
    device half* expert_probs          [[buffer(4)]],   // [batch, top_k] output
    constant uint& batch_size          [[buffer(5)]],
    constant uint& hidden_dim          [[buffer(6)]],
    constant uint& num_experts         [[buffer(7)]],
    constant uint& num_candidates      [[buffer(8)]],
    constant uint& top_k               [[buffer(9)]],
    uint tgid                          [[threadgroup_position_in_grid]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    uint batch_idx = tgid;
    if (batch_idx >= batch_size) return;

    const uint hidden_dim_vec4 = hidden_dim / 4;

    device const half4* h = hidden + batch_idx * hidden_dim_vec4;
    device const uint* candidates = candidate_ids + batch_idx * num_candidates;

    threadgroup float logits_shared[MAX_CANDIDATES];
    threadgroup float max_shared[2];
    threadgroup float sum_shared[2];

    const uint num_simdgroups = SPARSE_ROUTER_THREADS / 32;

    // Step 1: Vectorized GEMM for candidates
    uint candidates_per_thread = div_ceil(num_candidates, SPARSE_ROUTER_THREADS);

    for (uint c_iter = 0; c_iter < candidates_per_thread; ++c_iter) {
        uint c_idx = tid + c_iter * SPARSE_ROUTER_THREADS;
        if (c_idx >= num_candidates) break;

        uint expert_id = candidates[c_idx];
        device const half* w_col = router_weights + expert_id;

        float4 acc4 = float4(0.0f);

        for (uint d = 0; d < hidden_dim_vec4; ++d) {
            float4 h_vec = float4(h[d]);

            // Load 4 weights from non-contiguous locations
            float4 w_vec = float4(
                float(w_col[(d * 4 + 0) * num_experts]),
                float(w_col[(d * 4 + 1) * num_experts]),
                float(w_col[(d * 4 + 2) * num_experts]),
                float(w_col[(d * 4 + 3) * num_experts])
            );

            acc4 += h_vec * w_vec;
        }

        logits_shared[c_idx] = acc4.x + acc4.y + acc4.z + acc4.w;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Steps 2-3: Same softmax and top-k as non-vectorized version
    // (Copy from above for brevity, or refactor into helper)

    // Find max
    float local_max = -INFINITY;
    for (uint c = tid; c < num_candidates; c += SPARSE_ROUTER_THREADS) {
        local_max = max(local_max, logits_shared[c]);
    }
    local_max = simd_max_32(local_max);
    if (simd_lane == 0) max_shared[simd_id] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max;
    if (tid == 0) {
        global_max = max_shared[0];
        for (uint s = 1; s < num_simdgroups; ++s) {
            global_max = max(global_max, max_shared[s]);
        }
        max_shared[0] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = max_shared[0];

    // Exp and sum
    float local_sum = 0.0f;
    for (uint c = tid; c < num_candidates; c += SPARSE_ROUTER_THREADS) {
        float exp_val = safe_exp(logits_shared[c], global_max);
        logits_shared[c] = exp_val;
        local_sum += exp_val;
    }
    local_sum = simd_sum_32(local_sum);
    if (simd_lane == 0) sum_shared[simd_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum;
    if (tid == 0) {
        global_sum = sum_shared[0];
        for (uint s = 1; s < num_simdgroups; ++s) global_sum += sum_shared[s];
        sum_shared[0] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = sum_shared[0];

    // Normalize
    float inv_sum = 1.0f / global_sum;
    for (uint c = tid; c < num_candidates; c += SPARSE_ROUTER_THREADS) {
        logits_shared[c] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Top-k
    if (tid == 0) {
        float topk_vals[MAX_TOP_K];
        uint topk_candidate_indices[MAX_TOP_K];

        for (uint i = 0; i < top_k; ++i) {
            topk_vals[i] = -INFINITY;
            topk_candidate_indices[i] = 0;
        }

        for (uint c = 0; c < num_candidates; ++c) {
            float prob = logits_shared[c];
            if (prob > topk_vals[top_k - 1]) {
                uint insert_pos = top_k;
                for (uint i = 0; i < top_k; ++i) {
                    if (prob > topk_vals[i]) {
                        insert_pos = i;
                        break;
                    }
                }
                for (uint i = top_k - 1; i > insert_pos; --i) {
                    topk_vals[i] = topk_vals[i - 1];
                    topk_candidate_indices[i] = topk_candidate_indices[i - 1];
                }
                topk_vals[insert_pos] = prob;
                topk_candidate_indices[insert_pos] = c;
            }
        }

        float selected_sum = 0.0f;
        for (uint i = 0; i < top_k; ++i) selected_sum += topk_vals[i];
        float inv_selected_sum = 1.0f / max(selected_sum, 1e-8f);

        device uint* out_ids = expert_ids + batch_idx * top_k;
        device half* out_probs = expert_probs + batch_idx * top_k;

        for (uint i = 0; i < top_k; ++i) {
            out_ids[i] = candidates[topk_candidate_indices[i]];
            out_probs[i] = half(topk_vals[i] * inv_selected_sum);
        }
    }
}

// ---------------------------------------------------------------------------
// Batched sparse router for multiple sequences
// ---------------------------------------------------------------------------
// Processes tokens from multiple sequences with variable lengths.
// Uses 2D grid dispatch: (max_tokens_per_seq, batch_size).

kernel void moe_router_sparse_batched(
    device const half* hidden          [[buffer(0)]],   // [total_tokens, hidden_dim]
    device const half* router_weights  [[buffer(1)]],   // [hidden_dim, num_experts]
    device const uint* candidate_ids   [[buffer(2)]],   // [total_tokens, num_candidates]
    device uint* expert_ids            [[buffer(3)]],   // [total_tokens, top_k] output
    device half* expert_probs          [[buffer(4)]],   // [total_tokens, top_k] output
    device const uint* token_offsets   [[buffer(5)]],   // [batch_size + 1] prefix sums
    constant uint& batch_size          [[buffer(6)]],
    constant uint& hidden_dim          [[buffer(7)]],
    constant uint& num_experts         [[buffer(8)]],
    constant uint& num_candidates      [[buffer(9)]],
    constant uint& top_k               [[buffer(10)]],
    uint2 tgid                         [[threadgroup_position_in_grid]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    // tgid.y = sequence index, tgid.x = token within sequence
    uint seq_idx = tgid.y;
    if (seq_idx >= batch_size) return;

    uint seq_start = token_offsets[seq_idx];
    uint seq_end = token_offsets[seq_idx + 1];
    uint seq_len = seq_end - seq_start;

    uint token_in_seq = tgid.x;
    if (token_in_seq >= seq_len) return;

    uint global_token_idx = seq_start + token_in_seq;

    device const half* h = hidden + global_token_idx * hidden_dim;
    device const uint* candidates = candidate_ids + global_token_idx * num_candidates;

    threadgroup float logits_shared[MAX_CANDIDATES];
    threadgroup float max_shared[2];
    threadgroup float sum_shared[2];

    const uint num_simdgroups = SPARSE_ROUTER_THREADS / 32;

    // GEMM for candidates
    uint candidates_per_thread = div_ceil(num_candidates, SPARSE_ROUTER_THREADS);

    for (uint c_iter = 0; c_iter < candidates_per_thread; ++c_iter) {
        uint c_idx = tid + c_iter * SPARSE_ROUTER_THREADS;
        if (c_idx >= num_candidates) break;

        uint expert_id = candidates[c_idx];
        device const half* w_col = router_weights + expert_id;

        float acc = 0.0f;
        for (uint d = 0; d < hidden_dim; ++d) {
            acc += float(h[d]) * float(w_col[d * num_experts]);
        }
        logits_shared[c_idx] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax
    float local_max = -INFINITY;
    for (uint c = tid; c < num_candidates; c += SPARSE_ROUTER_THREADS) {
        local_max = max(local_max, logits_shared[c]);
    }
    local_max = simd_max_32(local_max);
    if (simd_lane == 0) max_shared[simd_id] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max;
    if (tid == 0) {
        global_max = max_shared[0];
        for (uint s = 1; s < num_simdgroups; ++s) {
            global_max = max(global_max, max_shared[s]);
        }
        max_shared[0] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = max_shared[0];

    float local_sum = 0.0f;
    for (uint c = tid; c < num_candidates; c += SPARSE_ROUTER_THREADS) {
        float exp_val = safe_exp(logits_shared[c], global_max);
        logits_shared[c] = exp_val;
        local_sum += exp_val;
    }
    local_sum = simd_sum_32(local_sum);
    if (simd_lane == 0) sum_shared[simd_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum;
    if (tid == 0) {
        global_sum = sum_shared[0];
        for (uint s = 1; s < num_simdgroups; ++s) global_sum += sum_shared[s];
        sum_shared[0] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = sum_shared[0];

    float inv_sum = 1.0f / global_sum;
    for (uint c = tid; c < num_candidates; c += SPARSE_ROUTER_THREADS) {
        logits_shared[c] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Top-k
    if (tid == 0) {
        float topk_vals[MAX_TOP_K];
        uint topk_candidate_indices[MAX_TOP_K];

        for (uint i = 0; i < top_k; ++i) {
            topk_vals[i] = -INFINITY;
            topk_candidate_indices[i] = 0;
        }

        for (uint c = 0; c < num_candidates; ++c) {
            float prob = logits_shared[c];
            if (prob > topk_vals[top_k - 1]) {
                uint insert_pos = top_k;
                for (uint i = 0; i < top_k; ++i) {
                    if (prob > topk_vals[i]) {
                        insert_pos = i;
                        break;
                    }
                }
                for (uint i = top_k - 1; i > insert_pos; --i) {
                    topk_vals[i] = topk_vals[i - 1];
                    topk_candidate_indices[i] = topk_candidate_indices[i - 1];
                }
                topk_vals[insert_pos] = prob;
                topk_candidate_indices[insert_pos] = c;
            }
        }

        float selected_sum = 0.0f;
        for (uint i = 0; i < top_k; ++i) selected_sum += topk_vals[i];
        float inv_selected_sum = 1.0f / max(selected_sum, 1e-8f);

        device uint* out_ids = expert_ids + global_token_idx * top_k;
        device half* out_probs = expert_probs + global_token_idx * top_k;

        for (uint i = 0; i < top_k; ++i) {
            out_ids[i] = candidates[topk_candidate_indices[i]];
            out_probs[i] = half(topk_vals[i] * inv_selected_sum);
        }
    }
}
