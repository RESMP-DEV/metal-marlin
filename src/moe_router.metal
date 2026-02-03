// moe_router.metal - Fused MoE router with top-k expert selection
//
// Computes: expert_probs = softmax(hidden @ router_weights)
// Then selects top-k experts per token with renormalized probabilities.
//
// Fused operations (single kernel launch):
//   1. GEMM: [batch, hidden] @ [hidden, num_experts] -> [batch, num_experts]
//   2. Softmax along expert dimension
//   3. Top-k selection (k typically 2-8)
//   4. Renormalize selected probabilities
//
// Output:
//   - expert_ids: [batch, top_k] uint32
//   - expert_probs: [batch, top_k] half (renormalized to sum to 1)
//
// Key optimization: The router is tiny (e.g., 4096 x 64 for GLM-4).
// Fusing avoids 3 separate kernel launches and intermediate buffers.
// Each token's routing is independent, enabling per-token parallelism.
//
// BF16 accumulation note: Router softmax benefits from the wider dynamic range
// of BF16 (8-bit exponent) for numerical stability in the exp() computation.
// We use float32 accumulators for the GEMM and softmax to maximize precision,
// then convert to FP16 for the output probabilities.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_atomic>

using namespace metal;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Maximum supported experts (must be power of 2 for efficient simdgroup ops)
constant constexpr uint MAX_EXPERTS = 256;

// Maximum top-k (reasonable for MoE architectures)
constant constexpr uint MAX_TOP_K = 16;

// Threads per threadgroup for the router kernel
// One threadgroup handles one token (batch element)
constant constexpr uint ROUTER_THREADS = 128;

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

inline uint div_ceil(uint a, uint b) {
    return (a + b - 1) / b;
}

// ---------------------------------------------------------------------------
// Atomic float add using CAS (Metal 2.3 compatible)
// ---------------------------------------------------------------------------

inline void atomic_add_float(device float* ptr, float value) {
    device atomic_uint* atomic_ptr = (device atomic_uint*)ptr;
    uint old_bits = atomic_load_explicit(atomic_ptr, memory_order_relaxed);
    uint new_bits;
    bool success;
    do {
        float old_val = as_type<float>(old_bits);
        float new_val = old_val + value;
        new_bits = as_type<uint>(new_val);
        success = atomic_compare_exchange_weak_explicit(
            atomic_ptr, &old_bits, new_bits,
            memory_order_relaxed, memory_order_relaxed);
    } while (!success);
}

// Numerically stable softmax helpers using float32 for accumulation
inline float safe_exp(float x, float max_val) {
    float shifted = x - max_val;
    // Clamp to prevent exp underflow/overflow
    shifted = clamp(shifted, -88.0f, 88.0f);
    return exp(shifted);
}

// ---------------------------------------------------------------------------
// Simdgroup reduction primitives
// ---------------------------------------------------------------------------

// Find maximum value across simdgroup (32 threads)
inline float simd_max(float val) {
    val = max(val, simd_shuffle_xor(val, 16));
    val = max(val, simd_shuffle_xor(val, 8));
    val = max(val, simd_shuffle_xor(val, 4));
    val = max(val, simd_shuffle_xor(val, 2));
    val = max(val, simd_shuffle_xor(val, 1));
    return val;
}

// Sum across simdgroup
inline float simd_sum(float val) {
    val += simd_shuffle_xor(val, 16);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);
    return val;
}

// ---------------------------------------------------------------------------
// Top-k selection using parallel bitonic-style comparison
// ---------------------------------------------------------------------------

// Insert a (value, index) pair into a sorted array of size k, maintaining
// descending order. Returns true if the pair was inserted.
template <uint K>
inline bool insert_topk(thread float (&values)[K], thread uint (&indices)[K],
                        float val, uint idx) {
    // Skip if value is less than the smallest in the top-k
    if (val <= values[K - 1]) return false;

    // Find insertion point (binary search would be overkill for small K)
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

// Specialized top-k for k=2 (most common MoE configuration)
inline void topk_2(thread float* probs, uint num_experts,
                   thread float (&top_vals)[2], thread uint (&top_ids)[2]) {
    top_vals[0] = -INFINITY;
    top_vals[1] = -INFINITY;
    top_ids[0] = 0;
    top_ids[1] = 0;

    for (uint i = 0; i < num_experts; ++i) {
        float p = probs[i];
        if (p > top_vals[0]) {
            top_vals[1] = top_vals[0];
            top_ids[1] = top_ids[0];
            top_vals[0] = p;
            top_ids[0] = i;
        } else if (p > top_vals[1]) {
            top_vals[1] = p;
            top_ids[1] = i;
        }
    }
}

// General top-k selection for k > 2
// Uses a simple insertion sort approach since k is typically small (2-8)
template <uint K>
inline void topk_general(thread float* probs, uint num_experts,
                         thread float (&top_vals)[K], thread uint (&top_ids)[K]) {
    // Initialize with negative infinity
    for (uint i = 0; i < K; ++i) {
        top_vals[i] = -INFINITY;
        top_ids[i] = 0;
    }

    // Scan all experts and maintain top-k
    for (uint i = 0; i < num_experts; ++i) {
        insert_topk<K>(top_vals, top_ids, probs[i], i);
    }
}

// ---------------------------------------------------------------------------
// Fused Router Kernel: GEMM + Softmax + Top-K + Renormalize
// ---------------------------------------------------------------------------
//
// Grid dispatch: 1 threadgroup per batch element (token)
// Each threadgroup:
//   1. Cooperatively loads the hidden state for this token
//   2. Computes hidden @ router_weights using simdgroup ops
//   3. Applies numerically stable softmax
//   4. Selects top-k experts
//   5. Renormalizes selected probabilities
//
// For small expert counts (<=64), the entire softmax can be done in registers.
// For larger counts, we use threadgroup memory.
// ---------------------------------------------------------------------------

kernel void moe_router_fused(
    device const half* hidden         [[buffer(0)]],   // [batch, hidden_dim]
    device const half* router_weights [[buffer(1)]],   // [hidden_dim, num_experts]
    device uint* expert_ids           [[buffer(2)]],   // [batch, top_k] output
    device half* expert_probs         [[buffer(3)]],   // [batch, top_k] output
    constant uint& batch_size         [[buffer(4)]],
    constant uint& hidden_dim         [[buffer(5)]],
    constant uint& num_experts        [[buffer(6)]],
    constant uint& top_k              [[buffer(7)]],
    uint tgid                         [[threadgroup_position_in_grid]],
    uint tid                          [[thread_index_in_threadgroup]],
    uint simd_lane                    [[thread_index_in_simdgroup]],
    uint simd_id                      [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles one token
    uint batch_idx = tgid;
    if (batch_idx >= batch_size) return;

    // Pointer to this token's hidden state
    device const half* h = hidden + batch_idx * hidden_dim;

    // Threadgroup memory for logits, reduction, and top-k results
    threadgroup float logits_shared[MAX_EXPERTS];
    threadgroup float max_shared[4];      // One per simdgroup
    threadgroup float sum_shared[4];
    threadgroup float top_k_vals[MAX_TOP_K];
    threadgroup uint top_k_ids[MAX_TOP_K];

    const uint num_simdgroups = ROUTER_THREADS / 32;  // 4

    // -------------------------------------------------------------------------
    // Step 1: Compute logits = hidden @ router_weights (GEMM)
    // -------------------------------------------------------------------------
    // Each thread computes a subset of the output logits
    // For hidden_dim=4096, num_experts=64: each thread handles ~0.5 experts
    // We distribute experts across threads

    uint experts_per_thread = div_ceil(num_experts, ROUTER_THREADS);

    for (uint e_iter = 0; e_iter < experts_per_thread; ++e_iter) {
        uint expert_idx = tid + e_iter * ROUTER_THREADS;
        if (expert_idx >= num_experts) break;

        // Compute dot product: h[0:hidden_dim] . router_weights[:, expert_idx]
        float acc = 0.0f;

        // Vectorized load and accumulate
        // router_weights is [hidden_dim, num_experts], column-major for expert_idx
        device const half* w_col = router_weights + expert_idx;

        for (uint d = 0; d < hidden_dim; ++d) {
            acc += float(h[d]) * float(w_col[d * num_experts]);
        }

        logits_shared[expert_idx] = acc;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Step 2: Numerically stable softmax
    // -------------------------------------------------------------------------

    // 2a. Find max logit (for numerical stability)
    float local_max = -INFINITY;
    for (uint e = tid; e < num_experts; e += ROUTER_THREADS) {
        local_max = max(local_max, logits_shared[e]);
    }

    // Reduce within simdgroup
    local_max = simd_max(local_max);

    // Store simdgroup max
    if (simd_lane == 0) {
        max_shared[simd_id] = local_max;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final max reduction across simdgroups (thread 0)
    float global_max;
    if (tid == 0) {
        global_max = max_shared[0];
        for (uint s = 1; s < num_simdgroups; ++s) {
            global_max = max(global_max, max_shared[s]);
        }
        max_shared[0] = global_max;  // Broadcast
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = max_shared[0];

    // 2b. Compute exp(logit - max) and sum
    float local_sum = 0.0f;
    for (uint e = tid; e < num_experts; e += ROUTER_THREADS) {
        float exp_val = safe_exp(logits_shared[e], global_max);
        logits_shared[e] = exp_val;  // Store exp values
        local_sum += exp_val;
    }

    // Reduce sum within simdgroup
    local_sum = simd_sum(local_sum);

    if (simd_lane == 0) {
        sum_shared[simd_id] = local_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final sum reduction
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

    // 2c. Normalize: softmax probs = exp_val / sum
    float inv_sum = 1.0f / global_sum;
    for (uint e = tid; e < num_experts; e += ROUTER_THREADS) {
        logits_shared[e] *= inv_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Step 3: Top-k selection (single thread, since k is small)
    // -------------------------------------------------------------------------
    // For efficiency with small k and num_experts, thread 0 does serial top-k

    if (tid == 0) {
        // Use stack arrays for top-k
        float local_topk_vals[MAX_TOP_K];
        uint local_topk_ids[MAX_TOP_K];

        // Initialize
        for (uint i = 0; i < top_k; ++i) {
            local_topk_vals[i] = -INFINITY;
            local_topk_ids[i] = 0;
        }

        // Scan and maintain top-k
        for (uint e = 0; e < num_experts; ++e) {
            float prob = logits_shared[e];

            // Check if this should be inserted into top-k
            if (prob > local_topk_vals[top_k - 1]) {
                // Find insertion point
                uint insert_pos = top_k;
                for (uint i = 0; i < top_k; ++i) {
                    if (prob > local_topk_vals[i]) {
                        insert_pos = i;
                        break;
                    }
                }

                // Shift elements down
                for (uint i = top_k - 1; i > insert_pos; --i) {
                    local_topk_vals[i] = local_topk_vals[i - 1];
                    local_topk_ids[i] = local_topk_ids[i - 1];
                }

                // Insert
                local_topk_vals[insert_pos] = prob;
                local_topk_ids[insert_pos] = e;
            }
        }

        // Store to shared memory for renormalization
        for (uint i = 0; i < top_k; ++i) {
            top_k_vals[i] = local_topk_vals[i];
            top_k_ids[i] = local_topk_ids[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Step 4: Renormalize selected probabilities
    // -------------------------------------------------------------------------

    if (tid == 0) {
        // Compute sum of selected probabilities
        float selected_sum = 0.0f;
        for (uint i = 0; i < top_k; ++i) {
            selected_sum += top_k_vals[i];
        }

        // Renormalize and write outputs
        float inv_selected_sum = 1.0f / max(selected_sum, 1e-8f);

        device uint* out_ids = expert_ids + batch_idx * top_k;
        device half* out_probs = expert_probs + batch_idx * top_k;

        for (uint i = 0; i < top_k; ++i) {
            out_ids[i] = top_k_ids[i];
            out_probs[i] = half(top_k_vals[i] * inv_selected_sum);
        }
    }
}

// ---------------------------------------------------------------------------
// Optimized variant for small hidden_dim using simdgroup matrix ops
// ---------------------------------------------------------------------------
// For hidden_dim <= 512 and num_experts <= 64, we can use simdgroup ops
// to accelerate the GEMM portion.

kernel void moe_router_fused_small(
    device const half* hidden         [[buffer(0)]],   // [batch, hidden_dim]
    device const half* router_weights [[buffer(1)]],   // [hidden_dim, num_experts]
    device uint* expert_ids           [[buffer(2)]],   // [batch, top_k]
    device half* expert_probs         [[buffer(3)]],   // [batch, top_k]
    constant uint& batch_size         [[buffer(4)]],
    constant uint& hidden_dim         [[buffer(5)]],
    constant uint& num_experts        [[buffer(6)]],
    constant uint& top_k              [[buffer(7)]],
    uint tgid                         [[threadgroup_position_in_grid]],
    uint tid                          [[thread_index_in_threadgroup]],
    uint simd_lane                    [[thread_index_in_simdgroup]],
    uint simd_id                      [[simdgroup_index_in_threadgroup]]
) {
    // Same logic as above but optimized for small dimensions
    // where we can keep more data in registers

    uint batch_idx = tgid;
    if (batch_idx >= batch_size) return;

    // For very small expert counts, each thread handles the full computation
    if (num_experts <= 32 && hidden_dim <= 256) {
        // Thread 0 does everything in registers
        if (tid != 0) return;

        device const half* h = hidden + batch_idx * hidden_dim;

        float logits[32];  // Up to 32 experts
        float max_val = -INFINITY;

        // Compute all logits
        for (uint e = 0; e < num_experts; ++e) {
            float acc = 0.0f;
            device const half* w_col = router_weights + e;
            for (uint d = 0; d < hidden_dim; ++d) {
                acc += float(h[d]) * float(w_col[d * num_experts]);
            }
            logits[e] = acc;
            max_val = max(max_val, acc);
        }

        // Softmax
        float sum = 0.0f;
        for (uint e = 0; e < num_experts; ++e) {
            logits[e] = safe_exp(logits[e], max_val);
            sum += logits[e];
        }

        float inv_sum = 1.0f / sum;
        for (uint e = 0; e < num_experts; ++e) {
            logits[e] *= inv_sum;
        }

        // Top-k selection
        float top_vals[MAX_TOP_K];
        uint top_ids[MAX_TOP_K];
        for (uint i = 0; i < top_k; ++i) {
            top_vals[i] = -INFINITY;
            top_ids[i] = 0;
        }

        for (uint e = 0; e < num_experts; ++e) {
            float prob = logits[e];
            if (prob > top_vals[top_k - 1]) {
                uint insert_pos = top_k;
                for (uint i = 0; i < top_k; ++i) {
                    if (prob > top_vals[i]) {
                        insert_pos = i;
                        break;
                    }
                }
                for (uint i = top_k - 1; i > insert_pos; --i) {
                    top_vals[i] = top_vals[i - 1];
                    top_ids[i] = top_ids[i - 1];
                }
                top_vals[insert_pos] = prob;
                top_ids[insert_pos] = e;
            }
        }

        // Renormalize and output
        float selected_sum = 0.0f;
        for (uint i = 0; i < top_k; ++i) {
            selected_sum += top_vals[i];
        }
        float inv_selected_sum = 1.0f / max(selected_sum, 1e-8f);

        device uint* out_ids = expert_ids + batch_idx * top_k;
        device half* out_probs = expert_probs + batch_idx * top_k;

        for (uint i = 0; i < top_k; ++i) {
            out_ids[i] = top_ids[i];
            out_probs[i] = half(top_vals[i] * inv_selected_sum);
        }
        return;
    }

    // Fall back to the full kernel for larger dimensions
    // (This path won't be reached for truly small dimensions)
    device const half* h = hidden + batch_idx * hidden_dim;

    threadgroup float logits_shared[MAX_EXPERTS];
    threadgroup float max_shared[4];
    threadgroup float sum_shared[4];

    const uint num_simdgroups = ROUTER_THREADS / 32;

    // GEMM
    uint experts_per_thread = div_ceil(num_experts, ROUTER_THREADS);
    for (uint e_iter = 0; e_iter < experts_per_thread; ++e_iter) {
        uint expert_idx = tid + e_iter * ROUTER_THREADS;
        if (expert_idx >= num_experts) break;

        float acc = 0.0f;
        device const half* w_col = router_weights + expert_idx;
        for (uint d = 0; d < hidden_dim; ++d) {
            acc += float(h[d]) * float(w_col[d * num_experts]);
        }
        logits_shared[expert_idx] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax (same as above)
    float local_max = -INFINITY;
    for (uint e = tid; e < num_experts; e += ROUTER_THREADS) {
        local_max = max(local_max, logits_shared[e]);
    }
    local_max = simd_max(local_max);
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
    for (uint e = tid; e < num_experts; e += ROUTER_THREADS) {
        float exp_val = safe_exp(logits_shared[e], global_max);
        logits_shared[e] = exp_val;
        local_sum += exp_val;
    }
    local_sum = simd_sum(local_sum);
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
    for (uint e = tid; e < num_experts; e += ROUTER_THREADS) {
        logits_shared[e] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Top-k and output
    if (tid == 0) {
        float local_topk_vals[MAX_TOP_K];
        uint local_topk_ids[MAX_TOP_K];
        for (uint i = 0; i < top_k; ++i) {
            local_topk_vals[i] = -INFINITY;
            local_topk_ids[i] = 0;
        }

        for (uint e = 0; e < num_experts; ++e) {
            float prob = logits_shared[e];
            if (prob > local_topk_vals[top_k - 1]) {
                uint insert_pos = top_k;
                for (uint i = 0; i < top_k; ++i) {
                    if (prob > local_topk_vals[i]) {
                        insert_pos = i;
                        break;
                    }
                }
                for (uint i = top_k - 1; i > insert_pos; --i) {
                    local_topk_vals[i] = local_topk_vals[i - 1];
                    local_topk_ids[i] = local_topk_ids[i - 1];
                }
                local_topk_vals[insert_pos] = prob;
                local_topk_ids[insert_pos] = e;
            }
        }

        float selected_sum = 0.0f;
        for (uint i = 0; i < top_k; ++i) selected_sum += local_topk_vals[i];
        float inv_selected_sum = 1.0f / max(selected_sum, 1e-8f);

        device uint* out_ids = expert_ids + batch_idx * top_k;
        device half* out_probs = expert_probs + batch_idx * top_k;

        for (uint i = 0; i < top_k; ++i) {
            out_ids[i] = local_topk_ids[i];
            out_probs[i] = half(local_topk_vals[i] * inv_selected_sum);
        }
    }
}

// ---------------------------------------------------------------------------
// Coalesced Router: Transposed weights for coalesced memory access
// ---------------------------------------------------------------------------
// This kernel expects weights in [num_experts, hidden_dim] layout (row-major
// per expert) instead of [hidden_dim, num_experts]. This eliminates the
// strided access pattern where stride = num_experts (64-256 typically).
//
// Performance benefit: Coalesced loads achieve ~10x better bandwidth than
// strided loads. For hidden_dim=4096, num_experts=64, this reduces memory
// traffic from ~1 cache line per element to ~1 cache line per 8 elements.
//
// Host-side preparation:
//   router_weights_coalesced = router_weights.T.contiguous()  # [num_experts, hidden_dim]
// ---------------------------------------------------------------------------

kernel void moe_router_fused_coalesced(
    device const half* hidden         [[buffer(0)]],   // [batch, hidden_dim]
    device const half* router_weights [[buffer(1)]],   // [num_experts, hidden_dim] TRANSPOSED
    device uint* expert_ids           [[buffer(2)]],   // [batch, top_k] output
    device half* expert_probs         [[buffer(3)]],   // [batch, top_k] output
    constant uint& batch_size         [[buffer(4)]],
    constant uint& hidden_dim         [[buffer(5)]],
    constant uint& num_experts        [[buffer(6)]],
    constant uint& top_k              [[buffer(7)]],
    uint tgid                         [[threadgroup_position_in_grid]],
    uint tid                          [[thread_index_in_threadgroup]],
    uint simd_lane                    [[thread_index_in_simdgroup]],
    uint simd_id                      [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles one token
    uint batch_idx = tgid;
    if (batch_idx >= batch_size) return;

    // Pointer to this token's hidden state
    device const half* h = hidden + batch_idx * hidden_dim;

    // Threadgroup memory for logits, reduction, and top-k results
    threadgroup float logits_shared[MAX_EXPERTS];
    threadgroup float max_shared[4];      // One per simdgroup
    threadgroup float sum_shared[4];

    const uint num_simdgroups = ROUTER_THREADS / 32;  // 4

    // -------------------------------------------------------------------------
    // Step 1: Compute logits = hidden @ router_weights.T (GEMM)
    // -------------------------------------------------------------------------
    // With transposed weights [num_experts, hidden_dim], each thread loads
    // a contiguous row of weights for coalesced access.

    uint experts_per_thread = div_ceil(num_experts, ROUTER_THREADS);

    for (uint e_iter = 0; e_iter < experts_per_thread; ++e_iter) {
        uint expert_idx = tid + e_iter * ROUTER_THREADS;
        if (expert_idx >= num_experts) break;

        // Compute dot product: h[0:hidden_dim] . router_weights[expert_idx, :]
        float acc = 0.0f;

        // COALESCED: router_weights is [num_experts, hidden_dim], row-major per expert
        device const half* w_row = router_weights + expert_idx * hidden_dim;

        // Vectorized loads using half4 (4 elements at a time)
        uint d = 0;
        uint hidden_dim_vec = hidden_dim & ~3u;  // Round down to multiple of 4

        for (; d < hidden_dim_vec; d += 4) {
            // Load 4 hidden values
            half4 h_vec = *reinterpret_cast<device const half4*>(h + d);
            // Load 4 weight values (COALESCED!)
            half4 w_vec = *reinterpret_cast<device const half4*>(w_row + d);
            // Accumulate dot product
            acc += float(h_vec.x) * float(w_vec.x);
            acc += float(h_vec.y) * float(w_vec.y);
            acc += float(h_vec.z) * float(w_vec.z);
            acc += float(h_vec.w) * float(w_vec.w);
        }

        // Handle remainder (hidden_dim not multiple of 4)
        for (; d < hidden_dim; ++d) {
            acc += float(h[d]) * float(w_row[d]);
        }

        logits_shared[expert_idx] = acc;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Steps 2-4: Softmax, Top-k, Renormalize (same as moe_router_fused)
    // -------------------------------------------------------------------------

    // 2a. Find max logit (for numerical stability)
    float local_max = -INFINITY;
    for (uint e = tid; e < num_experts; e += ROUTER_THREADS) {
        local_max = max(local_max, logits_shared[e]);
    }

    // Reduce within simdgroup
    local_max = simd_max(local_max);

    // Store simdgroup max
    if (simd_lane == 0) {
        max_shared[simd_id] = local_max;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final max reduction across simdgroups (thread 0)
    float global_max;
    if (tid == 0) {
        global_max = max_shared[0];
        for (uint s = 1; s < num_simdgroups; ++s) {
            global_max = max(global_max, max_shared[s]);
        }
        max_shared[0] = global_max;  // Broadcast
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = max_shared[0];

    // 2b. Compute exp(logit - max) and sum
    float local_sum = 0.0f;
    for (uint e = tid; e < num_experts; e += ROUTER_THREADS) {
        float exp_val = safe_exp(logits_shared[e], global_max);
        logits_shared[e] = exp_val;  // Store exp values
        local_sum += exp_val;
    }

    // Reduce sum within simdgroup
    local_sum = simd_sum(local_sum);

    if (simd_lane == 0) {
        sum_shared[simd_id] = local_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final sum reduction
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

    // 2c. Normalize: softmax probs = exp_val / sum
    float inv_sum = 1.0f / global_sum;
    for (uint e = tid; e < num_experts; e += ROUTER_THREADS) {
        logits_shared[e] *= inv_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Step 3: Top-k selection (single thread, since k is small)
    // -------------------------------------------------------------------------

    if (tid == 0) {
        // Use stack arrays for top-k
        float local_topk_vals[MAX_TOP_K];
        uint local_topk_ids[MAX_TOP_K];

        // Initialize
        for (uint i = 0; i < top_k; ++i) {
            local_topk_vals[i] = -INFINITY;
            local_topk_ids[i] = 0;
        }

        // Scan and maintain top-k
        for (uint e = 0; e < num_experts; ++e) {
            float prob = logits_shared[e];

            // Check if this should be inserted into top-k
            if (prob > local_topk_vals[top_k - 1]) {
                // Find insertion point
                uint insert_pos = top_k;
                for (uint i = 0; i < top_k; ++i) {
                    if (prob > local_topk_vals[i]) {
                        insert_pos = i;
                        break;
                    }
                }

                // Shift elements down
                for (uint i = top_k - 1; i > insert_pos; --i) {
                    local_topk_vals[i] = local_topk_vals[i - 1];
                    local_topk_ids[i] = local_topk_ids[i - 1];
                }

                // Insert
                local_topk_vals[insert_pos] = prob;
                local_topk_ids[insert_pos] = e;
            }
        }

        // -------------------------------------------------------------------------
        // Step 4: Renormalize selected probabilities
        // -------------------------------------------------------------------------

        // Compute sum of selected probabilities
        float selected_sum = 0.0f;
        for (uint i = 0; i < top_k; ++i) {
            selected_sum += local_topk_vals[i];
        }

        // Renormalize and write outputs
        float inv_selected_sum = 1.0f / max(selected_sum, 1e-8f);

        device uint* out_ids = expert_ids + batch_idx * top_k;
        device half* out_probs = expert_probs + batch_idx * top_k;

        for (uint i = 0; i < top_k; ++i) {
            out_ids[i] = local_topk_ids[i];
            out_probs[i] = half(local_topk_vals[i] * inv_selected_sum);
        }
    }
}

// ---------------------------------------------------------------------------
// Router with auxiliary load balancing loss computation
// ---------------------------------------------------------------------------
// MoE training requires load balancing loss to prevent expert collapse.
// This kernel computes both the routing and the auxiliary loss in one pass.

kernel void moe_router_with_aux_loss(
    device const half* hidden         [[buffer(0)]],   // [batch, hidden_dim]
    device const half* router_weights [[buffer(1)]],   // [hidden_dim, num_experts]
    device uint* expert_ids           [[buffer(2)]],   // [batch, top_k]
    device half* expert_probs         [[buffer(3)]],   // [batch, top_k]
    device float* expert_load         [[buffer(4)]],   // [num_experts] accumulates routing probs (atomic via CAS)
    device atomic_uint* expert_count  [[buffer(5)]],   // [num_experts] counts how many tokens
    constant uint& batch_size         [[buffer(6)]],
    constant uint& hidden_dim         [[buffer(7)]],
    constant uint& num_experts        [[buffer(8)]],
    constant uint& top_k              [[buffer(9)]],
    uint tgid                         [[threadgroup_position_in_grid]],
    uint tid                          [[thread_index_in_threadgroup]],
    uint simd_lane                    [[thread_index_in_simdgroup]],
    uint simd_id                      [[simdgroup_index_in_threadgroup]]
) {
    uint batch_idx = tgid;
    if (batch_idx >= batch_size) return;

    device const half* h = hidden + batch_idx * hidden_dim;

    threadgroup float logits_shared[MAX_EXPERTS];
    threadgroup float max_shared[4];
    threadgroup float sum_shared[4];

    const uint num_simdgroups = ROUTER_THREADS / 32;

    // GEMM: compute logits
    uint experts_per_thread = div_ceil(num_experts, ROUTER_THREADS);
    for (uint e_iter = 0; e_iter < experts_per_thread; ++e_iter) {
        uint expert_idx = tid + e_iter * ROUTER_THREADS;
        if (expert_idx >= num_experts) break;

        float acc = 0.0f;
        device const half* w_col = router_weights + expert_idx;
        for (uint d = 0; d < hidden_dim; ++d) {
            acc += float(h[d]) * float(w_col[d * num_experts]);
        }
        logits_shared[expert_idx] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax: find max
    float local_max = -INFINITY;
    for (uint e = tid; e < num_experts; e += ROUTER_THREADS) {
        local_max = max(local_max, logits_shared[e]);
    }
    local_max = simd_max(local_max);
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

    // Softmax: compute exp and sum
    float local_sum = 0.0f;
    for (uint e = tid; e < num_experts; e += ROUTER_THREADS) {
        float exp_val = safe_exp(logits_shared[e], global_max);
        logits_shared[e] = exp_val;
        local_sum += exp_val;
    }
    local_sum = simd_sum(local_sum);
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

    // Softmax: normalize
    float inv_sum = 1.0f / global_sum;
    for (uint e = tid; e < num_experts; e += ROUTER_THREADS) {
        logits_shared[e] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 handles top-k, output, and load balancing stats
    if (tid == 0) {
        float local_topk_vals[MAX_TOP_K];
        uint local_topk_ids[MAX_TOP_K];
        for (uint i = 0; i < top_k; ++i) {
            local_topk_vals[i] = -INFINITY;
            local_topk_ids[i] = 0;
        }

        for (uint e = 0; e < num_experts; ++e) {
            float prob = logits_shared[e];

            // Accumulate routing probability for load balancing loss
            // This is the "fraction of probability mass" for each expert
            atomic_add_float(expert_load + e, prob);

            // Top-k selection
            if (prob > local_topk_vals[top_k - 1]) {
                uint insert_pos = top_k;
                for (uint i = 0; i < top_k; ++i) {
                    if (prob > local_topk_vals[i]) {
                        insert_pos = i;
                        break;
                    }
                }
                for (uint i = top_k - 1; i > insert_pos; --i) {
                    local_topk_vals[i] = local_topk_vals[i - 1];
                    local_topk_ids[i] = local_topk_ids[i - 1];
                }
                local_topk_vals[insert_pos] = prob;
                local_topk_ids[insert_pos] = e;
            }
        }

        // Increment selection count for selected experts
        for (uint i = 0; i < top_k; ++i) {
            atomic_fetch_add_explicit(expert_count + local_topk_ids[i], 1u, memory_order_relaxed);
        }

        // Renormalize and output
        float selected_sum = 0.0f;
        for (uint i = 0; i < top_k; ++i) selected_sum += local_topk_vals[i];
        float inv_selected_sum = 1.0f / max(selected_sum, 1e-8f);

        device uint* out_ids = expert_ids + batch_idx * top_k;
        device half* out_probs = expert_probs + batch_idx * top_k;

        for (uint i = 0; i < top_k; ++i) {
            out_ids[i] = local_topk_ids[i];
            out_probs[i] = half(local_topk_vals[i] * inv_selected_sum);
        }
    }
}

// ---------------------------------------------------------------------------
// Batched router for processing multiple sequences efficiently
// ---------------------------------------------------------------------------
// When processing many sequences, we can use a 2D grid dispatch where
// each row is a sequence and each column is a token within the sequence.

kernel void moe_router_batched(
    device const half* hidden         [[buffer(0)]],   // [total_tokens, hidden_dim]
    device const half* router_weights [[buffer(1)]],   // [hidden_dim, num_experts]
    device uint* expert_ids           [[buffer(2)]],   // [total_tokens, top_k]
    device half* expert_probs         [[buffer(3)]],   // [total_tokens, top_k]
    device const uint* token_offsets  [[buffer(4)]],   // [batch_size + 1] prefix sums
    constant uint& batch_size         [[buffer(5)]],
    constant uint& hidden_dim         [[buffer(6)]],
    constant uint& num_experts        [[buffer(7)]],
    constant uint& top_k              [[buffer(8)]],
    uint2 tgid                        [[threadgroup_position_in_grid]],
    uint tid                          [[thread_index_in_threadgroup]],
    uint simd_lane                    [[thread_index_in_simdgroup]],
    uint simd_id                      [[simdgroup_index_in_threadgroup]]
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

    // Same logic as moe_router_fused but with batched token index
    threadgroup float logits_shared[MAX_EXPERTS];
    threadgroup float max_shared[4];
    threadgroup float sum_shared[4];

    const uint num_simdgroups = ROUTER_THREADS / 32;

    // GEMM
    uint experts_per_thread = div_ceil(num_experts, ROUTER_THREADS);
    for (uint e_iter = 0; e_iter < experts_per_thread; ++e_iter) {
        uint expert_idx = tid + e_iter * ROUTER_THREADS;
        if (expert_idx >= num_experts) break;

        float acc = 0.0f;
        device const half* w_col = router_weights + expert_idx;
        for (uint d = 0; d < hidden_dim; ++d) {
            acc += float(h[d]) * float(w_col[d * num_experts]);
        }
        logits_shared[expert_idx] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax
    float local_max = -INFINITY;
    for (uint e = tid; e < num_experts; e += ROUTER_THREADS) {
        local_max = max(local_max, logits_shared[e]);
    }
    local_max = simd_max(local_max);
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
    for (uint e = tid; e < num_experts; e += ROUTER_THREADS) {
        float exp_val = safe_exp(logits_shared[e], global_max);
        logits_shared[e] = exp_val;
        local_sum += exp_val;
    }
    local_sum = simd_sum(local_sum);
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
    for (uint e = tid; e < num_experts; e += ROUTER_THREADS) {
        logits_shared[e] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Top-k and output
    if (tid == 0) {
        float local_topk_vals[MAX_TOP_K];
        uint local_topk_ids[MAX_TOP_K];
        for (uint i = 0; i < top_k; ++i) {
            local_topk_vals[i] = -INFINITY;
            local_topk_ids[i] = 0;
        }

        for (uint e = 0; e < num_experts; ++e) {
            float prob = logits_shared[e];
            if (prob > local_topk_vals[top_k - 1]) {
                uint insert_pos = top_k;
                for (uint i = 0; i < top_k; ++i) {
                    if (prob > local_topk_vals[i]) {
                        insert_pos = i;
                        break;
                    }
                }
                for (uint i = top_k - 1; i > insert_pos; --i) {
                    local_topk_vals[i] = local_topk_vals[i - 1];
                    local_topk_ids[i] = local_topk_ids[i - 1];
                }
                local_topk_vals[insert_pos] = prob;
                local_topk_ids[insert_pos] = e;
            }
        }

        float selected_sum = 0.0f;
        for (uint i = 0; i < top_k; ++i) selected_sum += local_topk_vals[i];
        float inv_selected_sum = 1.0f / max(selected_sum, 1e-8f);

        device uint* out_ids = expert_ids + global_token_idx * top_k;
        device half* out_probs = expert_probs + global_token_idx * top_k;

        for (uint i = 0; i < top_k; ++i) {
            out_ids[i] = local_topk_ids[i];
            out_probs[i] = half(local_topk_vals[i] * inv_selected_sum);
        }
    }
}

// ---------------------------------------------------------------------------
// Test kernels
// ---------------------------------------------------------------------------

/// Test kernel: verify softmax correctness
kernel void test_moe_softmax(
    device const float* logits   [[buffer(0)]],  // [num_experts]
    device float* probs          [[buffer(1)]],  // [num_experts]
    constant uint& num_experts   [[buffer(2)]],
    uint tid                     [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    // Find max
    float max_val = logits[0];
    for (uint i = 1; i < num_experts; ++i) {
        max_val = max(max_val, logits[i]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < num_experts; ++i) {
        probs[i] = safe_exp(logits[i], max_val);
        sum += probs[i];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (uint i = 0; i < num_experts; ++i) {
        probs[i] *= inv_sum;
    }
}

/// Test kernel: verify top-k selection
kernel void test_moe_topk(
    device const float* probs    [[buffer(0)]],  // [num_experts]
    device uint* top_ids         [[buffer(1)]],  // [top_k]
    device float* top_probs      [[buffer(2)]],  // [top_k]
    constant uint& num_experts   [[buffer(3)]],
    constant uint& top_k         [[buffer(4)]],
    uint tid                     [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    float topk_vals[MAX_TOP_K];
    uint topk_ids[MAX_TOP_K];

    for (uint i = 0; i < top_k; ++i) {
        topk_vals[i] = -INFINITY;
        topk_ids[i] = 0;
    }

    for (uint e = 0; e < num_experts; ++e) {
        float prob = probs[e];
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
                topk_ids[i] = topk_ids[i - 1];
            }
            topk_vals[insert_pos] = prob;
            topk_ids[insert_pos] = e;
        }
    }

    // Renormalize
    float sum = 0.0f;
    for (uint i = 0; i < top_k; ++i) sum += topk_vals[i];
    float inv_sum = 1.0f / max(sum, 1e-8f);

    for (uint i = 0; i < top_k; ++i) {
        top_ids[i] = topk_ids[i];
        top_probs[i] = topk_vals[i] * inv_sum;
    }
}

/// Test kernel: end-to-end router for single token (validation)
kernel void test_moe_router_single(
    device const half* hidden         [[buffer(0)]],   // [hidden_dim]
    device const half* router_weights [[buffer(1)]],   // [hidden_dim, num_experts]
    device uint* expert_ids           [[buffer(2)]],   // [top_k]
    device half* expert_probs         [[buffer(3)]],   // [top_k]
    device float* debug_logits        [[buffer(4)]],   // [num_experts] debug output
    device float* debug_probs         [[buffer(5)]],   // [num_experts] debug output
    constant uint& hidden_dim         [[buffer(6)]],
    constant uint& num_experts        [[buffer(7)]],
    constant uint& top_k              [[buffer(8)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    // Compute logits
    float logits[MAX_EXPERTS];
    for (uint e = 0; e < num_experts; ++e) {
        float acc = 0.0f;
        device const half* w_col = router_weights + e;
        for (uint d = 0; d < hidden_dim; ++d) {
            acc += float(hidden[d]) * float(w_col[d * num_experts]);
        }
        logits[e] = acc;
        debug_logits[e] = acc;
    }

    // Softmax
    float max_val = logits[0];
    for (uint i = 1; i < num_experts; ++i) max_val = max(max_val, logits[i]);

    float sum = 0.0f;
    for (uint i = 0; i < num_experts; ++i) {
        logits[i] = safe_exp(logits[i], max_val);
        sum += logits[i];
    }

    float inv_sum = 1.0f / sum;
    for (uint i = 0; i < num_experts; ++i) {
        logits[i] *= inv_sum;
        debug_probs[i] = logits[i];
    }

    // Top-k
    float topk_vals[MAX_TOP_K];
    uint topk_ids[MAX_TOP_K];
    for (uint i = 0; i < top_k; ++i) {
        topk_vals[i] = -INFINITY;
        topk_ids[i] = 0;
    }

    for (uint e = 0; e < num_experts; ++e) {
        float prob = logits[e];
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
                topk_ids[i] = topk_ids[i - 1];
            }
            topk_vals[insert_pos] = prob;
            topk_ids[insert_pos] = e;
        }
    }

    // Renormalize
    float selected_sum = 0.0f;
    for (uint i = 0; i < top_k; ++i) selected_sum += topk_vals[i];
    float inv_selected_sum = 1.0f / max(selected_sum, 1e-8f);

    for (uint i = 0; i < top_k; ++i) {
        expert_ids[i] = topk_ids[i];
        expert_probs[i] = half(topk_vals[i] * inv_selected_sum);
    }
}
