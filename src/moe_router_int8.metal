// moe_router_int8.metal - INT8 quantized MoE router kernel
//
// Optimized router for MoE with INT8-quantized weights.
// The router is small (hidden_dim -> num_experts, e.g., 2048 -> 64)
// so INT8 quantization provides:
//   - 4x memory reduction (int8 vs fp32)
//   - 4x bandwidth improvement
//   - Minimal accuracy loss for routing decisions
//
// Quantization scheme:
//   - Per-channel symmetric: W_int8[e, h] = round(W[e, h] / scale[e])
//   - scale[e] = max(abs(W[e, :])) / 127
//   - Dequant: W[e, h] = W_int8[e, h] * scale[e]
//
// Kernel fuses:
//   1. INT8 GEMV: logits[e] = sum_h(input[h] * W_int8[e, h]) * scale[e]
//   2. Softmax over experts
//   3. Top-k selection with renormalization

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

constant constexpr uint MAX_EXPERTS = 256;
constant constexpr uint MAX_TOP_K = 16;
constant constexpr uint ROUTER_THREADS = 128;

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

inline float simd_max(float val) {
    val = max(val, simd_shuffle_xor(val, 16));
    val = max(val, simd_shuffle_xor(val, 8));
    val = max(val, simd_shuffle_xor(val, 4));
    val = max(val, simd_shuffle_xor(val, 2));
    val = max(val, simd_shuffle_xor(val, 1));
    return val;
}

inline float simd_sum(float val) {
    val += simd_shuffle_xor(val, 16);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);
    return val;
}

// ---------------------------------------------------------------------------
// INT8 GEMV helper: compute dot product using int8 weights
// ---------------------------------------------------------------------------
// Uses int32 accumulator for precision, then scales at the end.
// This is more efficient than dequantizing weights first.

inline float int8_dot_product(
    device const half* input,       // [hidden_dim] input activations
    device const char* weights,     // [hidden_dim] int8 weights for one expert
    uint hidden_dim
) {
    // Accumulate in int32 for precision
    int acc = 0;

    // Process 4 elements at a time for SIMD efficiency
    uint d = 0;
    for (; d + 3 < hidden_dim; d += 4) {
        // Load input as half and convert to int for accumulation
        // This is approximate but fast - for exact results, convert input to int16
        int4 w = int4(
            int(weights[d]),
            int(weights[d + 1]),
            int(weights[d + 2]),
            int(weights[d + 3])
        );

        // For mixed int8 x fp16, we scale input to int16 range then accumulate
        // This avoids fp->int conversion per element
        half4 h = half4(input[d], input[d + 1], input[d + 2], input[d + 3]);

        // Direct fp accumulation with int8 weights (convert weights to float)
        acc += w.x * int(round(float(h.x) * 128.0f));
        acc += w.y * int(round(float(h.y) * 128.0f));
        acc += w.z * int(round(float(h.z) * 128.0f));
        acc += w.w * int(round(float(h.w) * 128.0f));
    }

    // Handle remaining elements
    for (; d < hidden_dim; ++d) {
        acc += int(weights[d]) * int(round(float(input[d]) * 128.0f));
    }

    // Convert back to float (scale factor 128 for input normalization)
    return float(acc) / 128.0f;
}

// Alternative: Direct float accumulation (simpler, uses fp32 for accumulation)
inline float int8_dot_product_fp32(
    device const half* input,       // [hidden_dim] input activations
    device const char* weights,     // [hidden_dim] int8 weights for one expert
    uint hidden_dim
) {
    float acc = 0.0f;

    // Process 4 elements at a time
    uint d = 0;
    for (; d + 3 < hidden_dim; d += 4) {
        float4 w = float4(
            float(weights[d]),
            float(weights[d + 1]),
            float(weights[d + 2]),
            float(weights[d + 3])
        );
        float4 h = float4(
            float(input[d]),
            float(input[d + 1]),
            float(input[d + 2]),
            float(input[d + 3])
        );

        acc += dot(w, h);
    }

    // Remaining elements
    for (; d < hidden_dim; ++d) {
        acc += float(weights[d]) * float(input[d]);
    }

    return acc;
}

// ---------------------------------------------------------------------------
// INT8 Router Kernel: GEMV + Softmax + Top-K
// ---------------------------------------------------------------------------
//
// Grid dispatch: 1 threadgroup per batch element (token)
// Each threadgroup cooperatively computes routing for one token.

kernel void moe_router_int8_fused(
    device const half* input          [[buffer(0)]],   // [batch, hidden_dim]
    device const char* weights_int8   [[buffer(1)]],   // [num_experts, hidden_dim] int8
    device const float* scales        [[buffer(2)]],   // [num_experts] per-channel scales
    device uint* expert_ids           [[buffer(3)]],   // [batch, top_k] output
    device half* expert_probs         [[buffer(4)]],   // [batch, top_k] output
    constant uint& batch_size         [[buffer(5)]],
    constant uint& hidden_dim         [[buffer(6)]],
    constant uint& num_experts        [[buffer(7)]],
    constant uint& top_k              [[buffer(8)]],
    uint tgid                         [[threadgroup_position_in_grid]],
    uint tid                          [[thread_index_in_threadgroup]],
    uint simd_lane                    [[thread_index_in_simdgroup]],
    uint simd_id                      [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles one token
    uint batch_idx = tgid;
    if (batch_idx >= batch_size) return;

    // Pointer to this token's input
    device const half* h = input + batch_idx * hidden_dim;

    // Shared memory for logits and reductions
    threadgroup float logits_shared[MAX_EXPERTS];
    threadgroup float max_shared[4];
    threadgroup float sum_shared[4];

    const uint num_simdgroups = ROUTER_THREADS / 32;  // 4

    // -------------------------------------------------------------------------
    // Step 1: INT8 GEMV - compute logits
    // -------------------------------------------------------------------------
    // Each thread handles a subset of experts

    uint experts_per_thread = div_ceil(num_experts, ROUTER_THREADS);

    for (uint e_iter = 0; e_iter < experts_per_thread; ++e_iter) {
        uint expert_idx = tid + e_iter * ROUTER_THREADS;
        if (expert_idx >= num_experts) break;

        // Pointer to this expert's int8 weights
        device const char* w_expert = weights_int8 + expert_idx * hidden_dim;

        // Compute int8 dot product
        float acc = int8_dot_product_fp32(h, w_expert, hidden_dim);

        // Apply per-channel scale
        float logit = acc * scales[expert_idx];

        logits_shared[expert_idx] = logit;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Step 2: Numerically stable softmax
    // -------------------------------------------------------------------------

    // Find max logit
    float local_max = -INFINITY;
    for (uint e = tid; e < num_experts; e += ROUTER_THREADS) {
        local_max = max(local_max, logits_shared[e]);
    }
    local_max = simd_max(local_max);

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
    for (uint e = tid; e < num_experts; e += ROUTER_THREADS) {
        float exp_val = safe_exp(logits_shared[e], global_max);
        logits_shared[e] = exp_val;
        local_sum += exp_val;
    }
    local_sum = simd_sum(local_sum);

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
    for (uint e = tid; e < num_experts; e += ROUTER_THREADS) {
        logits_shared[e] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Step 3: Top-k selection
    // -------------------------------------------------------------------------

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

                // Insert new element
                local_topk_vals[insert_pos] = prob;
                local_topk_ids[insert_pos] = e;
            }
        }

        // Renormalize and output
        float selected_sum = 0.0f;
        for (uint i = 0; i < top_k; ++i) {
            selected_sum += local_topk_vals[i];
        }
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
// Optimized variant for small expert counts (<=64)
// ---------------------------------------------------------------------------
// For GLM-4.7 with 64 experts, each thread can handle one expert's dot product.

kernel void moe_router_int8_small(
    device const half* input          [[buffer(0)]],   // [batch, hidden_dim]
    device const char* weights_int8   [[buffer(1)]],   // [num_experts, hidden_dim] int8
    device const float* scales        [[buffer(2)]],   // [num_experts]
    device uint* expert_ids           [[buffer(3)]],   // [batch, top_k]
    device half* expert_probs         [[buffer(4)]],   // [batch, top_k]
    constant uint& batch_size         [[buffer(5)]],
    constant uint& hidden_dim         [[buffer(6)]],
    constant uint& num_experts        [[buffer(7)]],
    constant uint& top_k              [[buffer(8)]],
    uint tgid                         [[threadgroup_position_in_grid]],
    uint tid                          [[thread_index_in_threadgroup]],
    uint simd_lane                    [[thread_index_in_simdgroup]],
    uint simd_id                      [[simdgroup_index_in_threadgroup]]
) {
    uint batch_idx = tgid;
    if (batch_idx >= batch_size) return;

    device const half* h = input + batch_idx * hidden_dim;

    // For small expert counts, we can keep everything in registers
    // Each thread computes one expert's logit

    float my_logit = -INFINITY;
    uint my_expert = 0;

    if (tid < num_experts) {
        device const char* w_expert = weights_int8 + tid * hidden_dim;
        float acc = int8_dot_product_fp32(h, w_expert, hidden_dim);
        my_logit = acc * scales[tid];
        my_expert = tid;
    }

    // Share logits through threadgroup memory
    threadgroup float logits_shared[MAX_EXPERTS];
    threadgroup float max_shared[4];
    threadgroup float sum_shared[4];

    if (tid < num_experts) {
        logits_shared[tid] = my_logit;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax
    const uint num_simdgroups = ROUTER_THREADS / 32;

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

    // Top-k (thread 0)
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
// Decode-optimized kernel: batch=1 with minimal overhead
// ---------------------------------------------------------------------------
// For single-token decode, we want absolute minimum latency.
// One simdgroup processes all experts cooperatively.

kernel void moe_router_int8_decode(
    device const half* input          [[buffer(0)]],   // [hidden_dim]
    device const char* weights_int8   [[buffer(1)]],   // [num_experts, hidden_dim] int8
    device const float* scales        [[buffer(2)]],   // [num_experts]
    device uint* expert_ids           [[buffer(3)]],   // [top_k]
    device half* expert_probs         [[buffer(4)]],   // [top_k]
    constant uint& hidden_dim         [[buffer(5)]],
    constant uint& num_experts        [[buffer(6)]],
    constant uint& top_k              [[buffer(7)]],
    uint tid                          [[thread_position_in_threadgroup]],
    uint simd_lane                    [[thread_index_in_simdgroup]]
) {
    // Use 64 threads (2 simdgroups) for decode
    // Each thread handles ~1 expert for 64-expert models

    threadgroup float logits_shared[MAX_EXPERTS];

    // Step 1: Each thread computes one expert's logit
    if (tid < num_experts) {
        device const char* w = weights_int8 + tid * hidden_dim;
        float acc = 0.0f;

        // Unrolled dot product for speed
        uint d = 0;
        for (; d + 7 < hidden_dim; d += 8) {
            acc += float(w[d]) * float(input[d]);
            acc += float(w[d+1]) * float(input[d+1]);
            acc += float(w[d+2]) * float(input[d+2]);
            acc += float(w[d+3]) * float(input[d+3]);
            acc += float(w[d+4]) * float(input[d+4]);
            acc += float(w[d+5]) * float(input[d+5]);
            acc += float(w[d+6]) * float(input[d+6]);
            acc += float(w[d+7]) * float(input[d+7]);
        }
        for (; d < hidden_dim; ++d) {
            acc += float(w[d]) * float(input[d]);
        }

        logits_shared[tid] = acc * scales[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Softmax + Top-k (single thread)
    if (tid == 0) {
        // Find max
        float max_val = logits_shared[0];
        for (uint e = 1; e < num_experts; ++e) {
            max_val = max(max_val, logits_shared[e]);
        }

        // Exp and sum
        float sum = 0.0f;
        for (uint e = 0; e < num_experts; ++e) {
            float exp_val = safe_exp(logits_shared[e], max_val);
            logits_shared[e] = exp_val;
            sum += exp_val;
        }

        // Normalize
        float inv_sum = 1.0f / sum;
        for (uint e = 0; e < num_experts; ++e) {
            logits_shared[e] *= inv_sum;
        }

        // Top-k
        float topk_vals[MAX_TOP_K];
        uint topk_ids[MAX_TOP_K];
        for (uint i = 0; i < top_k; ++i) {
            topk_vals[i] = -INFINITY;
            topk_ids[i] = 0;
        }

        for (uint e = 0; e < num_experts; ++e) {
            float prob = logits_shared[e];
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

        // Renormalize and output
        float selected_sum = 0.0f;
        for (uint i = 0; i < top_k; ++i) selected_sum += topk_vals[i];
        float inv_selected_sum = 1.0f / max(selected_sum, 1e-8f);

        for (uint i = 0; i < top_k; ++i) {
            expert_ids[i] = topk_ids[i];
            expert_probs[i] = half(topk_vals[i] * inv_selected_sum);
        }
    }
}
