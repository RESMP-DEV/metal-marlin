/*
 * Fused MoE Router + Argsort Kernel
 * 
 * Eliminates intermediate buffer by fusing:
 * 1. Router logits computation (hidden_states @ router_weights.T)
 * 2. Softmax over experts
 * 3. Top-k selection with argsort
 * 
 * Outputs:
 * - expert_ids: [num_tokens, topk] selected expert indices
 * - sort_order: [num_tokens, topk] local sort indices for reordering
 * - expert_probs: [num_tokens, topk] softmax probabilities
 */

#include <metal_stdlib>
using namespace metal;

#ifndef MAX_EXPERTS
#define MAX_EXPERTS 64
#endif

#ifndef MAX_TOPK
#define MAX_TOPK 8
#endif

struct RouterParams {
    uint num_tokens;
    uint hidden_dim;      // Must be multiple of 4 for float4 loads
    uint num_experts;
    uint topk;
    float temperature;    // Softmax temperature (usually 1.0)
};

// Pair for sorting: (expert_id, probability)
struct ExpertProb {
    uint expert_id;
    float prob;
};

// Compare for descending order by probability
inline bool greater_than(ExpertProb a, ExpertProb b) {
    return a.prob > b.prob;
}

// Simple insertion sort for small topk (more efficient than bitonic for k <= 8)
template<uint N>
inline void insertion_sort_descending(thread ExpertProb* arr, thread uint* order) {
    for (uint i = 1; i < N; i++) {
        ExpertProb key = arr[i];
        uint key_order = order[i];
        int j = int(i) - 1;
        while (j >= 0 && arr[j].prob < key.prob) {
            arr[j + 1] = arr[j];
            order[j + 1] = order[j];
            j--;
        }
        arr[j + 1] = key;
        order[j + 1] = key_order;
    }
}

kernel void moe_router_argsort_fused(
    device const float4* hidden_states [[buffer(0)]],     // [num_tokens, hidden_dim/4]
    device const float4* router_weights [[buffer(1)]],    // [num_experts, hidden_dim/4]
    device uint* expert_ids [[buffer(2)]],                // [num_tokens, topk]
    device uint* sort_order [[buffer(3)]],                // [num_tokens, topk]
    device float* expert_probs [[buffer(4)]],             // [num_tokens, topk]
    constant RouterParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
)
{
    // Bounds check
    if (tid >= params.num_tokens) {
        return;
    }
    
    const uint num_experts = params.num_experts;
    const uint topk = params.topk;
    const uint hidden_dim_4 = params.hidden_dim / 4;
    const float temperature = params.temperature;
    
    // Thread-local storage for logits
    thread float logits[MAX_EXPERTS];
    thread ExpertProb expert_probs_local[MAX_TOPK];
    thread uint local_sort_order[MAX_TOPK];
    
    // =========================================================================
    // Step 1: Compute router logits for this token
    // =========================================================================
    // logits[e] = dot(hidden_states[tid], router_weights[e])
    
    for (uint e = 0; e < num_experts; e++) {
        float4 accum = float4(0.0);
        
        // Vectorized dot product: hidden_dim is multiple of 4
        for (uint h = 0; h < hidden_dim_4; h++) {
            float4 h_vec = hidden_states[tid * hidden_dim_4 + h];
            float4 w_vec = router_weights[e * hidden_dim_4 + h];
            accum += h_vec * w_vec;
        }
        
        // Horizontal sum of float4
        logits[e] = accum.x + accum.y + accum.z + accum.w;
    }
    
    // =========================================================================
    // Step 2: Softmax computation with SIMD reduction
    // =========================================================================
    
    // Find max logit for numerical stability
    float max_logit = logits[0];
    for (uint e = 1; e < num_experts; e++) {
        max_logit = max(max_logit, logits[e]);
    }
    
    // Compute exp(logit - max) and sum
    float exp_sum = 0.0;
    for (uint e = 0; e < num_experts; e++) {
        logits[e] = exp((logits[e] - max_logit) / temperature);
        exp_sum += logits[e];
    }
    
    // Normalize to get probabilities
    float inv_sum = 1.0 / exp_sum;
    for (uint e = 0; e < num_experts; e++) {
        logits[e] *= inv_sum;
    }
    
    // =========================================================================
    // Step 3: Top-k selection with argsort
    // =========================================================================
    
    // Initialize with first topk experts
    for (uint k = 0; k < topk; k++) {
        expert_probs_local[k].expert_id = k;
        expert_probs_local[k].prob = logits[k];
        local_sort_order[k] = k;
    }
    
    // Sort first topk in descending order
    insertion_sort_descending<MAX_TOPK>(expert_probs_local, local_sort_order);
    
    // Iterate through remaining experts, maintaining topk
    for (uint e = topk; e < num_experts; e++) {
        // Check if current expert has higher probability than the smallest in topk
        if (logits[e] > expert_probs_local[topk - 1].prob) {
            // Insert at the end and bubble up
            expert_probs_local[topk - 1].expert_id = e;
            expert_probs_local[topk - 1].prob = logits[e];
            local_sort_order[topk - 1] = e;
            
            // Bubble up to maintain sorted order
            for (int i = int(topk) - 1; i > 0; i--) {
                if (expert_probs_local[i].prob > expert_probs_local[i - 1].prob) {
                    // Swap
                    ExpertProb temp_prob = expert_probs_local[i];
                    expert_probs_local[i] = expert_probs_local[i - 1];
                    expert_probs_local[i - 1] = temp_prob;
                    
                    uint temp_order = local_sort_order[i];
                    local_sort_order[i] = local_sort_order[i - 1];
                    local_sort_order[i - 1] = temp_order;
                } else {
                    break;
                }
            }
        }
    }
    
    // =========================================================================
    // Step 4: Renormalize topk probabilities (common MoE practice)
    // =========================================================================
    
    float topk_sum = 0.0;
    for (uint k = 0; k < topk; k++) {
        topk_sum += expert_probs_local[k].prob;
    }
    
    float topk_inv_sum = 1.0 / topk_sum;
    for (uint k = 0; k < topk; k++) {
        expert_probs_local[k].prob *= topk_inv_sum;
    }
    
    // =========================================================================
    // Step 5: Write outputs
    // =========================================================================
    
    const uint out_offset = tid * topk;
    for (uint k = 0; k < topk; k++) {
        expert_ids[out_offset + k] = expert_probs_local[k].expert_id;
        expert_probs[out_offset + k] = expert_probs_local[k].prob;
        sort_order[out_offset + k] = local_sort_order[k];
    }
}

// ============================================================================
// Optimized variant: Pre-grouped experts for expert-parallel dispatch
// ============================================================================

struct DispatchParams {
    uint num_tokens;
    uint num_experts;
    uint topk;
    uint num_groups;      // Number of expert groups for parallel dispatch
};

// Alternative kernel that outputs expert-wise token lists for parallel dispatch
kernel void moe_router_argsort_fused_grouped(
    device const float4* hidden_states [[buffer(0)]],
    device const float4* router_weights [[buffer(1)]],
    device uint* expert_token_counts [[buffer(2)]],      // [num_experts] atomic counter
    device uint* expert_token_indices [[buffer(3)]],     // [num_experts, max_tokens_per_expert]
    device float* expert_token_weights [[buffer(4)]],    // [num_experts, max_tokens_per_expert]
    device uint* token_expert_ids [[buffer(5)]],         // [num_tokens, topk]
    constant DispatchParams& params [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
)
{
    if (tid >= params.num_tokens) {
        return;
    }
    
    // Reuse the same computation logic
    const uint num_experts = params.num_experts;
    const uint topk = params.topk;
    const uint hidden_dim_4 = 4096 / 4;  // Default, should be parameterized
    
    thread float logits[MAX_EXPERTS];
    thread ExpertProb top_experts[MAX_TOPK];
    
    // Compute logits (simplified - assumes fixed hidden_dim)
    for (uint e = 0; e < num_experts; e++) {
        float4 accum = float4(0.0);
        for (uint h = 0; h < hidden_dim_4; h++) {
            accum += hidden_states[tid * hidden_dim_4 + h] * 
                     router_weights[e * hidden_dim_4 + h];
        }
        logits[e] = accum.x + accum.y + accum.z + accum.w;
    }
    
    // Softmax
    float max_logit = logits[0];
    for (uint e = 1; e < num_experts; e++) {
        max_logit = max(max_logit, logits[e]);
    }
    
    float exp_sum = 0.0;
    for (uint e = 0; e < num_experts; e++) {
        logits[e] = exp(logits[e] - max_logit);
        exp_sum += logits[e];
    }
    
    float inv_sum = 1.0 / exp_sum;
    for (uint e = 0; e < num_experts; e++) {
        logits[e] *= inv_sum;
    }
    
    // Top-k selection
    for (uint k = 0; k < topk; k++) {
        top_experts[k].expert_id = k;
        top_experts[k].prob = logits[k];
    }
    
    for (uint e = topk; e < num_experts; e++) {
        if (logits[e] > top_experts[topk - 1].prob) {
            top_experts[topk - 1].expert_id = e;
            top_experts[topk - 1].prob = logits[e];
            
            for (int i = int(topk) - 1; i > 0; i--) {
                if (top_experts[i].prob > top_experts[i - 1].prob) {
                    ExpertProb temp = top_experts[i];
                    top_experts[i] = top_experts[i - 1];
                    top_experts[i - 1] = temp;
                } else {
                    break;
                }
            }
        }
    }
    
    // Renormalize
    float topk_sum = 0.0;
    for (uint k = 0; k < topk; k++) {
        topk_sum += top_experts[k].prob;
    }
    float renorm = 1.0 / topk_sum;
    
    // Output: record token for each selected expert
    for (uint k = 0; k < topk; k++) {
        uint expert_id = top_experts[k].expert_id;
        float weight = top_experts[k].prob * renorm;
        
        token_expert_ids[tid * topk + k] = expert_id;
        
        // Atomically increment count and get slot
        uint slot = atomic_fetch_add_explicit(
            (device atomic_uint*)&expert_token_counts[expert_id], 
            1, 
            memory_order_relaxed
        );
        
        // Record token index and weight
        expert_token_indices[expert_id * params.num_tokens + slot] = tid;
        expert_token_weights[expert_id * params.num_tokens + slot] = weight;
    }
}
