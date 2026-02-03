// moe_fused_router.metal - Fused Router Selection with Sorted Expert Indices
//
// Single kernel that performs:
//   1. Router GEMV: hidden @ router_weights -> logits
//   2. Softmax normalization
//   3. Top-k expert selection per token
//   4. Grouping: sort token-expert pairs by expert ID
//
// This eliminates CPU-side sorting and grouping, providing sorted indices
// directly for efficient batched expert GEMM execution.
//
// Output format:
//   sorted_indices: [total_assignments] - indices into expert_ids, grouped by expert
//   expert_offsets:  [num_experts + 1] - start/end indices for each expert
//   topk_expert_ids: [batch, top_k] - top-k expert selections (for reference)
//   topk_probs:      [batch, top_k] - normalized probabilities
//
// Memory layout for sorted output:
//   expert 0: sorted_indices[offsets[0]:offsets[1]]
//   expert 1: sorted_indices[offsets[1]:offsets[2]]
//   ...
//
// This matches the output of group_tokens_by_expert() in moe_dispatch.py,
// but computed entirely on GPU for zero CPU overhead.
//
// Key optimization: Uses global atomic operations for counting and writing,
// enabling single-pass token grouping across all threadgroups.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_atomic>

using namespace metal;

// ---------------------------------------------------------------------------
// Configuration constants
// ---------------------------------------------------------------------------

// Maximum supported experts
constant constexpr uint ROUTER_MAX_EXPERTS = 256;

// Maximum top-k per token
constant constexpr uint ROUTER_MAX_TOP_K = 16;

// Threads per threadgroup
constant constexpr uint ROUTER_TG_SIZE = 256;

// Simdgroup size
constant constexpr uint SIMD_WIDTH = 32;

// ---------------------------------------------------------------------------
// Router parameters
// ---------------------------------------------------------------------------

struct RouterParams {
    uint batch_size;      // Number of tokens
    uint hidden_dim;      // Hidden dimension
    uint num_experts;     // Total experts
    uint top_k;           // Experts per token
};

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

inline uint div_ceil(uint a, uint b) {
    return (a + b - 1) / b;
}

// ---------------------------------------------------------------------------
// Atomic operations for global counters
// ---------------------------------------------------------------------------

// Atomic add for expert assignment counts
inline void atomic_add_uint(device uint* ptr, uint value) {
    atomic_fetch_add_explicit((device atomic_uint*)ptr, value, memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Numerically stable softmax
// ---------------------------------------------------------------------------

inline float safe_exp(float x, float max_val) {
    float shifted = x - max_val;
    shifted = clamp(shifted, -88.0f, 88.0f);
    return exp(shifted);
}

// ---------------------------------------------------------------------------
// Simdgroup reductions
// ---------------------------------------------------------------------------

inline float simd_max_reduce(float val) {
    val = max(val, simd_shuffle_xor(val, 16));
    val = max(val, simd_shuffle_xor(val, 8));
    val = max(val, simd_shuffle_xor(val, 4));
    val = max(val, simd_shuffle_xor(val, 2));
    val = max(val, simd_shuffle_xor(val, 1));
    return val;
}

inline float simd_sum_reduce(float val) {
    val += simd_shuffle_xor(val, 16);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);
    return val;
}

// ---------------------------------------------------------------------------
// Top-k selection insertion (small k, optimized)
// ---------------------------------------------------------------------------

template <uint K>
inline bool insert_topk(
    thread float (&values)[K],
    thread uint (&indices)[K],
    float val,
    uint idx
) {
    if (val <= values[K - 1]) return false;

    // Find insertion point
    uint insert_pos = K;
    for (uint i = 0; i < K; ++i) {
        if (val > values[i]) {
            insert_pos = i;
            break;
        }
    }

    // Shift elements down
    for (uint i = K - 1; i > insert_pos; --i) {
        values[i] = values[i - 1];
        indices[i] = indices[i - 1];
    }

    // Insert new element
    values[insert_pos] = val;
    indices[insert_pos] = idx;
    return true;
}

// ---------------------------------------------------------------------------
// Main kernel: Router + Top-k + Global Grouping
//
// Each threadgroup handles one token, computing its top-k experts
// and then contributing to global sorted token-expert arrays.
//
// Grid dispatch: threadgroups = batch_size
//   1 threadgroup per token
//   1 SIMD group per token (32 threads)
//   Multiple SIMD groups per threadgroup for larger hidden_dim
//
// Global atomic counters ensure thread-safe writing to sorted output.
// ---------------------------------------------------------------------------

kernel void moe_fused_router_sorted(
    // Inputs
    device const half* hidden          [[buffer(0)]],   // [batch, hidden_dim]
    device const half* router_weights  [[buffer(1)]],   // [hidden_dim, num_experts]
    device uint* expert_offsets     [[buffer(2)]],   // [num_experts + 1] output (atomic counters)
    device uint* sorted_indices     [[buffer(3)]],   // [batch * top_k] output
    device uint* topk_expert_ids   [[buffer(4)]],   // [batch, top_k] output
    device half* topk_probs         [[buffer(5)]],   // [batch, top_k] output

    // Parameters
    constant RouterParams& params [[buffer(6)]],

    // Thread indexing
    uint tgid  [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]]
) {
    const uint batch_idx = tgid;
    if (batch_idx >= params.batch_size) return;

    const uint num_simdgroups = ROUTER_TG_SIZE / SIMD_WIDTH;
    const uint simd_id = tid / SIMD_WIDTH;
    const uint simd_tid = tid % SIMD_WIDTH;

    // =========================================================================
    // Step 1: Router GEMV - compute logits for this token
    // =========================================================================

    // Threadgroup memory for logits and reduction
    threadgroup float tg_logits[ROUTER_MAX_EXPERTS];
    threadgroup float tg_max[num_simdgroups];
    threadgroup float tg_sum[num_simdgroups];
    threadgroup float tg_topk_vals[ROUTER_MAX_TOP_K];
    threadgroup uint tg_topk_ids[ROUTER_MAX_TOP_K];

    device const half* h = hidden + batch_idx * params.hidden_dim;

    // Each thread computes a subset of expert logits
    uint experts_per_thread = div_ceil(params.num_experts, ROUTER_TG_SIZE);

    for (uint e_iter = 0; e_iter < experts_per_thread; ++e_iter) {
        uint expert_idx = tid + e_iter * ROUTER_TG_SIZE;

        if (expert_idx < params.num_experts) {
            // Dot product: h[0:hidden_dim] . router_weights[:, expert_idx]
            // router_weights is [hidden_dim, num_experts], column-major for expert
            float acc = 0.0f;
            device const half* w_col = router_weights + expert_idx;

            // Vectorized accumulation (8 at a time)
            uint d = 0;
            uint hidden_vec = params.hidden_dim & ~7u;  // Round down to multiple of 8

            for (; d < hidden_vec; d += 8) {
                acc += float(h[d + 0]) * float(w_col[(d + 0) * params.num_experts]);
                acc += float(h[d + 1]) * float(w_col[(d + 1) * params.num_experts]);
                acc += float(h[d + 2]) * float(w_col[(d + 2) * params.num_experts]);
                acc += float(h[d + 3]) * float(w_col[(d + 3) * params.num_experts]);
                acc += float(h[d + 4]) * float(w_col[(d + 4) * params.num_experts]);
                acc += float(h[d + 5]) * float(w_col[(d + 5) * params.num_experts]);
                acc += float(h[d + 6]) * float(w_col[(d + 6) * params.num_experts]);
                acc += float(h[d + 7]) * float(w_col[(d + 7) * params.num_experts]);
            }

            // Handle remainder
            for (; d < params.hidden_dim; ++d) {
                acc += float(h[d]) * float(w_col[d * params.num_experts]);
            }

            tg_logits[expert_idx] = acc;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // Step 2: Softmax (numerically stable)
    // =========================================================================

    // 2a. Find max logit
    float local_max = -INFINITY;
    for (uint e = tid; e < params.num_experts; e += ROUTER_TG_SIZE) {
        local_max = max(local_max, tg_logits[e]);
    }

    local_max = simd_max_reduce(local_max);

    if (lane == 0) {
        tg_max[simd_id] = local_max;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = tg_max[0];
    for (uint s = 1; s < num_simdgroups; ++s) {
        global_max = max(global_max, tg_max[s]);
    }

    // Broadcast max to all threads
    if (simd_id == 0 && lane == 0) {
        tg_max[0] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = tg_max[0];

    // 2b. Compute exp(logit - max) and accumulate sum
    float local_sum = 0.0f;
    for (uint e = tid; e < params.num_experts; e += ROUTER_TG_SIZE) {
        float exp_val = safe_exp(tg_logits[e], global_max);
        tg_logits[e] = exp_val;  // Store in place
        local_sum += exp_val;
    }

    local_sum = simd_sum_reduce(local_sum);

    if (lane == 0) {
        tg_sum[simd_id] = local_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum = tg_sum[0];
    for (uint s = 1; s < num_simdgroups; ++s) {
        global_sum += tg_sum[s];
    }

    // Broadcast sum
    if (simd_id == 0 && lane == 0) {
        tg_sum[0] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = tg_sum[0];

    // 2c. Normalize probabilities
    float inv_sum = 1.0f / global_sum;
    for (uint e = tid; e < params.num_experts; e += ROUTER_TG_SIZE) {
        tg_logits[e] *= inv_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // Step 3: Top-k selection (thread 0 does this)
    // =========================================================================

    if (tid == 0) {
        float local_topk_vals[ROUTER_MAX_TOP_K];
        uint local_topk_ids[ROUTER_MAX_TOP_K];

        // Initialize
        for (uint i = 0; i < params.top_k; ++i) {
            local_topk_vals[i] = -INFINITY;
            local_topk_ids[i] = 0;
        }

        // Scan and maintain top-k
        for (uint e = 0; e < params.num_experts; ++e) {
            float prob = tg_logits[e];
            insert_topk<ROUTER_MAX_TOP_K>(local_topk_vals, local_topk_ids, prob, e);
        }

        // Store to shared memory for other threads to access
        for (uint i = 0; i < params.top_k; ++i) {
            tg_topk_vals[i] = local_topk_vals[i];
            tg_topk_ids[i] = local_topk_ids[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // Step 4: Global grouping using atomics
    //
    // For each selected expert (top_k), we need to:
    //   1. Atomically claim a slot in the sorted array for that expert
    //   2. Write our token-expert pair to that slot
    //
    // expert_offsets[e] stores the start index for expert e's assignments.
    // We use it as an atomic counter to claim positions.
    // =========================================================================

    // Make sure all threads have access to top-k results
    uint selected_expert_ids[ROUTER_MAX_TOP_K];
    float selected_probs[ROUTER_MAX_TOP_K];

    for (uint k = 0; k < params.top_k; ++k) {
        selected_expert_ids[k] = tg_topk_ids[k];
        selected_probs[k] = tg_topk_vals[k];
    }

    // Each of our top_k selections claims a slot atomically
    for (uint k = 0; k < params.top_k; ++k) {
        uint expert_id = selected_expert_ids[k];

        if (expert_id >= params.num_experts) continue;

        // Atomically get and increment the position for this expert
        // expert_offsets[expert_id] points to next available slot
        uint position = atomic_fetch_add_explicit(
            (device atomic_uint*)(expert_offsets + expert_id),
            1u,
            memory_order_relaxed
        );

        // Write our token-index-expert-index pair to sorted array
        // We need to encode: which token, and which expert slot (0..top_k-1)
        // Sort key: expert_id * batch_size * top_k + token_idx * top_k + k
        // This ensures tokens group by expert, preserving order within each expert
        uint flat_idx = batch_idx * params.top_k + k;
        sorted_indices[position] = flat_idx;
    }

    // =========================================================================
    // Step 5: Write top-k outputs for reference
    // =========================================================================

    if (tid == 0) {
        // Renormalize top-k probabilities (they're already from softmax, so just renormalize)
        float selected_sum = 0.0f;
        for (uint k = 0; k < params.top_k; ++k) {
            selected_sum += selected_probs[k];
        }

        float inv_selected_sum = 1.0f / max(selected_sum, 1e-8f);

        device uint* out_ids = topk_expert_ids + batch_idx * params.top_k;
        device half* out_probs = topk_probs + batch_idx * params.top_k;

        for (uint k = 0; k < params.top_k; ++k) {
            out_ids[k] = selected_expert_ids[k];
            out_probs[k] = half(selected_probs[k] * inv_selected_sum);
        }
    }
}

// ---------------------------------------------------------------------------
// Batched variant: Process multiple tokens per threadgroup
//
// For large batches, this variant processes multiple tokens per threadgroup
// to better utilize hardware while still producing sorted output.
// ---------------------------------------------------------------------------

kernel void moe_fused_router_sorted_batched(
    // Inputs
    device const half* hidden          [[buffer(0)]],   // [batch, hidden_dim]
    device const half* router_weights  [[buffer(1)]],   // [hidden_dim, num_experts]
    device uint* expert_offsets     [[buffer(2)]],   // [num_experts + 1] output
    device uint* sorted_indices     [[buffer(3)]],   // [batch * top_k] output
    device uint* topk_expert_ids   [[buffer(4)]],   // [batch, top_k] output
    device half* topk_probs         [[buffer(5)]],   // [batch, top_k] output

    // Parameters
    constant RouterParams& params [[buffer(6)]],

    // Thread indexing
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    // 2D grid: tgid.x = token batch index, tgid.y = expert slot (0..top_k-1)
    uint batch_idx = tgid.x;
    uint k_slot = tgid.y;

    if (batch_idx >= params.batch_size || k_slot >= params.top_k) return;

    // =========================================================================
    // Step 1: Compute logits for all experts (thread 0 only per token)
    // =========================================================================

    if (tid == 0) {
        device const half* h = hidden + batch_idx * params.hidden_dim;
        float local_logits[ROUTER_MAX_EXPERTS];
        float max_val = -INFINITY;

        // Compute all logits
        for (uint e = 0; e < params.num_experts; ++e) {
            float acc = 0.0f;
            device const half* w_col = router_weights + e;

            for (uint d = 0; d < params.hidden_dim; ++d) {
                acc += float(h[d]) * float(w_col[d * params.num_experts]);
            }

            local_logits[e] = acc;
            max_val = max(max_val, acc);
        }

        // Softmax: compute exp values
        float sum = 0.0f;
        for (uint e = 0; e < params.num_experts; ++e) {
            local_logits[e] = safe_exp(local_logits[e], max_val);
            sum += local_logits[e];
        }

        // Normalize
        float inv_sum = 1.0f / sum;
        for (uint e = 0; e < params.num_experts; ++e) {
            local_logits[e] *= inv_sum;
        }

        // Top-k selection
        float topk_vals[ROUTER_MAX_TOP_K];
        uint topk_ids[ROUTER_MAX_TOP_K];

        for (uint k = 0; k < params.top_k; ++k) {
            topk_vals[k] = -INFINITY;
            topk_ids[k] = 0;
        }

        for (uint e = 0; e < params.num_experts; ++e) {
            insert_topk<ROUTER_MAX_TOP_K>(topk_vals, topk_ids, local_logits[e], e);
        }

        // Write top-k outputs
        device uint* out_ids = topk_expert_ids + batch_idx * params.top_k;
        device half* out_probs = topk_probs + batch_idx * params.top_k;

        float selected_sum = 0.0f;
        for (uint k = 0; k < params.top_k; ++k) selected_sum += topk_vals[k];
        float inv_selected_sum = 1.0f / max(selected_sum, 1e-8f);

        for (uint k = 0; k < params.top_k; ++k) {
            out_ids[k] = topk_ids[k];
            out_probs[k] = half(topk_vals[k] * inv_selected_sum);
        }

        // Store my expert assignment for this slot
        uint my_expert = topk_ids[k_slot];
        float my_prob = topk_vals[k_slot] * inv_selected_sum;

        // =========================================================================
        // Step 2: Atomic write to sorted arrays
        // =========================================================================

        uint position = atomic_fetch_add_explicit(
            (device atomic_uint*)(expert_offsets + my_expert),
            1u,
            memory_order_relaxed
        );

        uint flat_idx = batch_idx * params.top_k + k_slot;
        sorted_indices[position] = flat_idx;
    }
}

// ---------------------------------------------------------------------------
// Coalesced variant: Transposed weights for better memory access
//
// Expects router_weights in [num_experts, hidden_dim] layout instead of
// [hidden_dim, num_experts]. This eliminates strided access patterns.
// ---------------------------------------------------------------------------

kernel void moe_fused_router_sorted_coalesced(
    // Inputs
    device const half* hidden          [[buffer(0)]],   // [batch, hidden_dim]
    device const half* router_weights  [[buffer(1)]],   // [num_experts, hidden_dim] TRANSPOSED
    device uint* expert_offsets     [[buffer(2)]],   // [num_experts + 1] output
    device uint* sorted_indices     [[buffer(3)]],   // [batch * top_k] output
    device uint* topk_expert_ids   [[buffer(4)]],   // [batch, top_k] output
    device half* topk_probs         [[buffer(5)]],   // [batch, top_k] output

    // Parameters
    constant RouterParams& params [[buffer(6)]],

    // Thread indexing
    uint tgid  [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]]
) {
    const uint batch_idx = tgid;
    if (batch_idx >= params.batch_size) return;

    const uint num_simdgroups = ROUTER_TG_SIZE / SIMD_WIDTH;
    const uint simd_id = tid / SIMD_WIDTH;
    const uint simd_tid = tid % SIMD_WIDTH;

    // Threadgroup memory
    threadgroup float tg_logits[ROUTER_MAX_EXPERTS];
    threadgroup float tg_max[num_simdgroups];
    threadgroup float tg_sum[num_simdgroups];
    threadgroup float tg_topk_vals[ROUTER_MAX_TOP_K];
    threadgroup uint tg_topk_ids[ROUTER_MAX_TOP_K];

    device const half* h = hidden + batch_idx * params.hidden_dim;

    // =========================================================================
    // Step 1: Router GEMV with coalesced access
    // =========================================================================

    uint experts_per_thread = div_ceil(params.num_experts, ROUTER_TG_SIZE);

    for (uint e_iter = 0; e_iter < experts_per_thread; ++e_iter) {
        uint expert_idx = tid + e_iter * ROUTER_TG_SIZE;

        if (expert_idx < params.num_experts) {
            // With transposed weights [num_experts, hidden_dim]:
            // router_weights[expert_idx] is a contiguous row
            device const half* w_row = router_weights + expert_idx * params.hidden_dim;

            float acc = 0.0f;

            // Vectorized loads with half4 (4 at a time)
            uint d = 0;
            uint hidden_vec = params.hidden_dim & ~3u;

            for (; d < hidden_vec; d += 4) {
                half4 h_vec = *reinterpret_cast<device const half4*>(h + d);
                half4 w_vec = *reinterpret_cast<device const half4*>(w_row + d);

                acc += float(h_vec.x) * float(w_vec.x);
                acc += float(h_vec.y) * float(w_vec.y);
                acc += float(h_vec.z) * float(w_vec.z);
                acc += float(h_vec.w) * float(w_vec.w);
            }

            // Handle remainder
            for (; d < params.hidden_dim; ++d) {
                acc += float(h[d]) * float(w_row[d]);
            }

            tg_logits[expert_idx] = acc;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // Steps 2-5: Same as main kernel
    // =========================================================================

    // Softmax
    float local_max = -INFINITY;
    for (uint e = tid; e < params.num_experts; e += ROUTER_TG_SIZE) {
        local_max = max(local_max, tg_logits[e]);
    }

    local_max = simd_max_reduce(local_max);
    if (lane == 0) tg_max[simd_id] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = tg_max[0];
    for (uint s = 1; s < num_simdgroups; ++s) {
        global_max = max(global_max, tg_max[s]);
    }
    if (simd_id == 0 && lane == 0) tg_max[0] = global_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = tg_max[0];

    float local_sum = 0.0f;
    for (uint e = tid; e < params.num_experts; e += ROUTER_TG_SIZE) {
        float exp_val = safe_exp(tg_logits[e], global_max);
        tg_logits[e] = exp_val;
        local_sum += exp_val;
    }

    local_sum = simd_sum_reduce(local_sum);
    if (lane == 0) tg_sum[simd_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum = tg_sum[0];
    for (uint s = 1; s < num_simdgroups; ++s) {
        global_sum += tg_sum[s];
    }
    if (simd_id == 0 && lane == 0) tg_sum[0] = global_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = tg_sum[0];

    float inv_sum = 1.0f / global_sum;
    for (uint e = tid; e < params.num_experts; e += ROUTER_TG_SIZE) {
        tg_logits[e] *= inv_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Top-k
    if (tid == 0) {
        float local_topk_vals[ROUTER_MAX_TOP_K];
        uint local_topk_ids[ROUTER_MAX_TOP_K];

        for (uint i = 0; i < params.top_k; ++i) {
            local_topk_vals[i] = -INFINITY;
            local_topk_ids[i] = 0;
        }

        for (uint e = 0; e < params.num_experts; ++e) {
            insert_topk<ROUTER_MAX_TOP_K>(local_topk_vals, local_topk_ids, tg_logits[e], e);
        }

        for (uint i = 0; i < params.top_k; ++i) {
            tg_topk_vals[i] = local_topk_vals[i];
            tg_topk_ids[i] = local_topk_ids[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Global grouping
    uint selected_expert_ids[ROUTER_MAX_TOP_K];
    float selected_probs[ROUTER_MAX_TOP_K];

    for (uint k = 0; k < params.top_k; ++k) {
        selected_expert_ids[k] = tg_topk_ids[k];
        selected_probs[k] = tg_topk_vals[k];
    }

    for (uint k = 0; k < params.top_k; ++k) {
        uint expert_id = selected_expert_ids[k];

        if (expert_id >= params.num_experts) continue;

        uint position = atomic_fetch_add_explicit(
            (device atomic_uint*)(expert_offsets + expert_id),
            1u,
            memory_order_relaxed
        );

        uint flat_idx = batch_idx * params.top_k + k;
        sorted_indices[position] = flat_idx;
    }

    // Write outputs
    if (tid == 0) {
        float selected_sum = 0.0f;
        for (uint k = 0; k < params.top_k; ++k) {
            selected_sum += selected_probs[k];
        }

        float inv_selected_sum = 1.0f / max(selected_sum, 1e-8f);

        device uint* out_ids = topk_expert_ids + batch_idx * params.top_k;
        device half* out_probs = topk_probs + batch_idx * params.top_k;

        for (uint k = 0; k < params.top_k; ++k) {
            out_ids[k] = selected_expert_ids[k];
            out_probs[k] = half(selected_probs[k] * inv_selected_sum);
        }
    }
}
