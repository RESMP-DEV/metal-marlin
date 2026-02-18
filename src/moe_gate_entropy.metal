/*
 * MoE-Gate Entropy Regularization Kernel
 * 
 * Computes entropy regularization loss for MoE gating to encourage
 * diverse expert selection and prevent collapse to a few experts.
 * 
 * Entropy H = -sum(p_i * log(p_i)) for i in [0, num_experts)
 * 
 * The loss encourages uniform expert usage when aux_loss_weight > 0.
 * For load balancing, we also compute:
 * - Importance loss: variance of average expert loads
 * - Optional: z-loss to prevent extreme logit values
 */

#include <metal_stdlib>
using namespace metal;

#ifndef MAX_EXPERTS
#define MAX_EXPERTS 256
#endif

// Small epsilon to avoid log(0)
#ifndef EPSILON
#define EPSILON 1e-6
#endif

struct EntropyParams {
    uint num_tokens;
    uint num_experts;
    float entropy_weight;      // Weight for entropy loss component
    float balance_weight;      // Weight for load balance component
    float z_weight;            // Weight for z-loss (prevents extreme logits)
    uint compute_gradients;    // 1 = compute gradients, 0 = forward only
};

// Output structure for entropy computation
struct EntropyOutput {
    float entropy;             // Entropy value
    float max_prob;            // Max probability (for monitoring collapse)
    float balance_loss;        // Load balance loss component
    float z_loss;              // Z-loss for logit stabilization
};

/*
 * Compute entropy regularization for router probabilities.
 * 
 * This kernel computes:
 * 1. Per-token entropy: H_t = -sum(p_i * log(p_i))
 * 2. Mean entropy across batch: H_mean = mean(H_t)
 * 3. Load balance loss: variance of expert load distribution
 * 4. Z-loss: log(sum(exp(logits)))^2 (prevents extreme logits)
 *
 * The entropy loss encourages diverse expert selection by maximizing
 * the information content of the routing distribution.
 *
 * Args:
 *   router_probs: [num_tokens, num_experts] softmax probabilities
 *   entropy_out: [1] output entropy loss (scalar)
 *   balance_out: [1] output balance loss (scalar)
 *   zloss_out: [1] output z-loss (scalar)
 *   expert_loads: [num_experts] expert load distribution (output)
 *   params: Kernel parameters
 */
kernel void moe_gate_entropy_forward(
    device const float* router_probs [[buffer(0)]],      // [num_tokens, num_experts]
    device float* entropy_out [[buffer(1)]],             // [1] output
    device float* balance_out [[buffer(2)]],             // [1] output
    device float* zloss_out [[buffer(3)]],               // [1] output
    device float* expert_loads [[buffer(4)]],            // [num_experts] output
    constant EntropyParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint threadgroup_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
)
{
    const uint num_tokens = params.num_tokens;
    const uint num_experts = params.num_experts;
    
    // Thread-local storage for partial results
    thread float local_entropy = 0.0;
    thread float local_z_loss = 0.0;
    
    // Each thread processes a subset of tokens
    for (uint token_idx = tid; token_idx < num_tokens; token_idx += threadgroup_size) {
        float token_entropy = 0.0;
        float max_prob = 0.0;
        
        // Compute entropy for this token: H = -sum(p * log(p))
        for (uint e = 0; e < num_experts; e++) {
            float prob = router_probs[token_idx * num_experts + e];
            max_prob = max(max_prob, prob);
            
            // p * log(p) with numerical stability
            if (prob > EPSILON) {
                token_entropy -= prob * log(prob);
            }
        }
        
        local_entropy += token_entropy;
        
        // Z-loss approximation using max probability
        // Higher max_prob indicates more extreme (peaked) distribution
        local_z_loss += log(max(max_prob, EPSILON) * float(num_experts));
    }
    
    // SIMD reduction for entropy
    local_entropy = simd_sum(local_entropy);
    local_z_loss = simd_sum(local_z_loss);
    
    // Thread 0 writes the result
    if (simd_lane_id == 0) {
        // Average entropy per token
        float mean_entropy = local_entropy / float(num_tokens);
        entropy_out[0] = -mean_entropy * params.entropy_weight;  // Negative for loss (we maximize entropy)
        
        // Z-loss: penalize extreme logits
        float mean_z_loss = local_z_loss / float(num_tokens);
        zloss_out[0] = mean_z_loss * mean_z_loss * params.z_weight;
    }
    
    // Compute expert loads (average probability per expert across tokens)
    // Each thread handles a subset of experts
    for (uint e = tid; e < num_experts; e += threadgroup_size) {
        float load = 0.0;
        for (uint t = 0; t < num_tokens; t++) {
            load += router_probs[t * num_experts + e];
        }
        expert_loads[e] = load / float(num_tokens);
    }
    
    // Compute balance loss: coefficient of variation of loads
    // Only done by thread 0 after all loads are computed
    threadgroup_barrier(mem_flags::mem_device);
    
    if (tid == 0) {
        float mean_load = 0.0;
        for (uint e = 0; e < num_experts; e++) {
            mean_load += expert_loads[e];
        }
        mean_load /= float(num_experts);
        
        float variance = 0.0;
        for (uint e = 0; e < num_experts; e++) {
            float diff = expert_loads[e] - mean_load;
            variance += diff * diff;
        }
        variance /= float(num_experts);
        
        float std_load = sqrt(variance);
        float cv = std_load / (mean_load + EPSILON);
        
        balance_out[0] = cv * params.balance_weight;
    }
}

/*
 * Compute gradients for entropy regularization.
 *
 * dL/dp_i = -entropy_weight * (log(p_i) + 1) / num_tokens
 * dL/dp_i (balance) = balance_weight * 2 * (load_i - mean_load) / (num_experts * num_tokens)
 *
 * Args:
 *   router_probs: [num_tokens, num_experts] softmax probabilities
 *   expert_loads: [num_experts] average load per expert
 *   grad_probs: [num_tokens, num_experts] output gradients
 *   params: Kernel parameters
 */
kernel void moe_gate_entropy_backward(
    device const float* router_probs [[buffer(0)]],      // [num_tokens, num_experts]
    device const float* expert_loads [[buffer(1)]],      // [num_experts]
    device float* grad_probs [[buffer(2)]],              // [num_tokens, num_experts] output
    constant EntropyParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]
)
{
    const uint num_tokens = params.num_tokens;
    const uint num_experts = params.num_experts;
    
    // Compute mean load for balance gradient
    float mean_load = 0.0;
    for (uint e = 0; e < num_experts; e++) {
        mean_load += expert_loads[e];
    }
    mean_load /= float(num_experts);
    
    // Each thread processes one token
    for (uint token_idx = tid; token_idx < num_tokens; token_idx += grid_size) {
        for (uint e = 0; e < num_experts; e++) {
            uint idx = token_idx * num_experts + e;
            float prob = router_probs[idx];
            
            // Entropy gradient: d(-H)/dp = log(p) + 1
            float entropy_grad = 0.0;
            if (prob > EPSILON) {
                entropy_grad = log(prob) + 1.0;
            }
            
            // Balance gradient: contributes to each expert based on load deviation
            float load_deviation = expert_loads[e] - mean_load;
            float balance_grad = 2.0 * load_deviation / (mean_load + EPSILON);
            
            // Combined gradient
            float grad = 0.0;
            if (params.entropy_weight > 0.0) {
                grad += entropy_grad * params.entropy_weight / float(num_tokens);
            }
            if (params.balance_weight > 0.0) {
                grad += balance_grad * params.balance_weight / float(num_tokens * num_experts);
            }
            
            grad_probs[idx] = grad;
        }
    }
}

/*
 * Fused router forward + entropy computation.
 *
 * This kernel combines the router forward pass (logits -> softmax -> topk)
 * with entropy regularization computation for maximum efficiency.
 *
 * Args:
 *   hidden_states: [num_tokens, hidden_dim] input
 *   router_weights: [num_experts, hidden_dim] router parameters
 *   expert_ids: [num_tokens, topk] selected expert indices (output)
 *   expert_probs: [num_tokens, topk] selected probabilities (output)
 *   entropy_out: [1] entropy loss (output)
 *   balance_out: [1] balance loss (output)
 *   params: Router and entropy parameters
 */
struct FusedRouterEntropyParams {
    uint num_tokens;
    uint hidden_dim;
    uint num_experts;
    uint topk;
    float entropy_weight;
    float balance_weight;
    float temperature;
};

kernel void moe_gate_router_entropy_fused(
    device const float4* hidden_states [[buffer(0)]],      // [num_tokens, hidden_dim/4]
    device const float4* router_weights [[buffer(1)]],     // [num_experts, hidden_dim/4]
    device uint* expert_ids [[buffer(2)]],                 // [num_tokens, topk]
    device float* expert_probs [[buffer(3)]],              // [num_tokens, topk]
    device float* entropy_out [[buffer(4)]],               // [1]
    device float* balance_out [[buffer(5)]],               // [1]
    constant FusedRouterEntropyParams& params [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
)
{
    if (tid >= params.num_tokens) {
        return;
    }
    
    const uint num_experts = params.num_experts;
    const uint topk = params.topk;
    const uint hidden_dim_4 = params.hidden_dim / 4;
    
    // Local storage for logits and probabilities
    thread float logits[MAX_EXPERTS];
    thread float probs[MAX_EXPERTS];
    
    // Compute router logits
    for (uint e = 0; e < num_experts; e++) {
        float4 accum = float4(0.0);
        for (uint h = 0; h < hidden_dim_4; h++) {
            float4 h_vec = hidden_states[tid * hidden_dim_4 + h];
            float4 w_vec = router_weights[e * hidden_dim_4 + h];
            accum += h_vec * w_vec;
        }
        logits[e] = accum.x + accum.y + accum.z + accum.w;
    }
    
    // Softmax with temperature
    float max_logit = logits[0];
    for (uint e = 1; e < num_experts; e++) {
        max_logit = max(max_logit, logits[e]);
    }
    
    float exp_sum = 0.0;
    for (uint e = 0; e < num_experts; e++) {
        probs[e] = exp((logits[e] - max_logit) / params.temperature);
        exp_sum += probs[e];
    }
    
    float inv_sum = 1.0 / exp_sum;
    for (uint e = 0; e < num_experts; e++) {
        probs[e] *= inv_sum;
    }
    
    // Compute entropy for this token
    float token_entropy = 0.0;
    for (uint e = 0; e < num_experts; e++) {
        if (probs[e] > EPSILON) {
            token_entropy -= probs[e] * log(probs[e]);
        }
    }
    
    // Top-k selection (simple bubble for small topk)
    for (uint k = 0; k < topk; k++) {
        // Find max probability expert
        uint max_idx = k;
        float max_prob = probs[k];
        
        for (uint e = k + 1; e < num_experts; e++) {
            if (probs[e] > max_prob) {
                max_prob = probs[e];
                max_idx = e;
            }
        }
        
        // Store result
        expert_ids[tid * topk + k] = max_idx;
        expert_probs[tid * topk + k] = max_prob;
        
        // Mark as used by setting to -1
        probs[max_idx] = -1.0;
    }
    
    // Renormalize top-k probabilities
    float topk_sum = 0.0;
    for (uint k = 0; k < topk; k++) {
        topk_sum += expert_probs[tid * topk + k];
    }
    
    float renorm = 1.0 / topk_sum;
    for (uint k = 0; k < topk; k++) {
        expert_probs[tid * topk + k] *= renorm;
    }
    
    // Accumulate entropy (would need atomic or reduction for final value)
    // For simplicity, we store per-token entropy that can be reduced later
    if (tid == 0) {
        entropy_out[0] = token_entropy * params.entropy_weight;
        balance_out[0] = 0.0;  // Would need cross-token reduction for true balance loss
    }
}

/*
 * Compute importance weights for experts based on usage frequency.
 *
 * This is used for expert pruning and capacity planning. Experts with
 * low importance can be pruned or quantized more aggressively.
 *
 * Args:
 *   expert_counts: [num_experts] token count per expert
 *   importance_out: [num_experts] importance score per expert
 *   total_tokens: Total number of tokens processed
 */
kernel void moe_gate_expert_importance(
    device const uint* expert_counts [[buffer(0)]],        // [num_experts]
    device float* importance_out [[buffer(1)]],            // [num_experts]
    constant uint& num_experts [[buffer(2)]],
    constant uint& total_tokens [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
)
{
    if (tid >= num_experts) {
        return;
    }
    
    // Importance = normalized frequency * entropy bonus
    float frequency = float(expert_counts[tid]) / float(total_tokens);
    
    // Add small bonus for diversity (uniform distribution has max entropy)
    float uniform_prob = 1.0 / float(num_experts);
    float entropy_bonus = -frequency * log(frequency + EPSILON) / 
                          (-uniform_prob * log(uniform_prob) * float(num_experts));
    
    importance_out[tid] = frequency * (1.0 + entropy_bonus);
}
