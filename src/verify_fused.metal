#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------------
// Parallel Reduction Helpers
// -----------------------------------------------------------------------------

template<typename T>
inline T warp_reduce_max(T val) {
    val = max(val, simd_shuffle_down(val, 16));
    val = max(val, simd_shuffle_down(val, 8));
    val = max(val, simd_shuffle_down(val, 4));
    val = max(val, simd_shuffle_down(val, 2));
    val = max(val, simd_shuffle_down(val, 1));
    return val;
}

template<typename T>
inline T warp_reduce_sum(T val) {
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);
    return val;
}

// -----------------------------------------------------------------------------
// Fused Verification Kernel
// -----------------------------------------------------------------------------

kernel void verify_draft_tokens_fused(
    device const float* target_logits [[buffer(0)]],   // [Batch, K+1, V]
    device const long* draft_tokens [[buffer(1)]],      // [Batch, K]
    device const float* rand_uniform [[buffer(2)]],    // [Batch, K]
    device int* output_count [[buffer(3)]],            // [Batch]
    device uchar* output_mask [[buffer(4)]],           // [Batch, K]
    constant uint& V [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant float& temperature [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint batch_idx = tgid.x;
    const uint num_threads = 1024;
    const uint num_warps = 32; // 1024 / 32
    
    // Shared memory for warp reductions
    threadgroup float shared_max[32];
    threadgroup float shared_sum[32];
    
    // State for acceptance (local to thread 0)
    bool sequence_accepted = true;
    int accepted_count = 0;
    
    // Stride for logits
    // target_logits is [Batch, K+1, V]
    // We access row k for batch_idx
    const uint row_stride = V;
    const uint batch_stride = (K + 1) * V;
    
    // Iterate over draft tokens (K is small, usually 4-8)
    for (uint k = 0; k < K; ++k) {
        // --- Step 1: Compute Max ---
        float local_max = -INFINITY;
        
        // Pointer to start of this logit row
        uint logit_offset = batch_idx * batch_stride + k * row_stride;
        
        for (uint i = tid.x; i < V; i += num_threads) {
            float val = target_logits[logit_offset + i];
            // Apply temperature during load to avoid repeated divisions
            if (temperature > 0) {
                val /= temperature;
            }
            local_max = max(local_max, val);
        }
        
        // Warp reduction
        local_max = warp_reduce_max(local_max);
        
        // Store warp result
        if (simd_lane_id == 0) {
            shared_max[simd_group_id] = local_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Final reduction by first warp
        float global_max = -INFINITY;
        if (simd_group_id == 0) {
            float val = (tid.x < num_warps) ? shared_max[tid.x] : -INFINITY;
            val = warp_reduce_max(val);
            if (tid.x == 0) {
                shared_max[0] = val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        global_max = shared_max[0];
        
        // --- Step 2: Compute Sum Exp ---
        float local_sum = 0.0f;
        for (uint i = tid.x; i < V; i += num_threads) {
            float val = target_logits[logit_offset + i];
            if (temperature > 0) {
                val /= temperature;
            }
            local_sum += exp(val - global_max);
        }
        
        local_sum = warp_reduce_sum(local_sum);
        
        if (simd_lane_id == 0) {
            shared_sum[simd_group_id] = local_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        float global_sum = 0.0f;
        if (simd_group_id == 0) {
            float val = (tid.x < num_warps) ? shared_sum[tid.x] : 0.0f;
            val = warp_reduce_sum(val);
            if (tid.x == 0) {
                shared_sum[0] = val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        global_sum = shared_sum[0];
        
        // --- Step 3: Verify ---
        if (tid.x == 0) {
            if (sequence_accepted) {
                long draft_token_id = draft_tokens[batch_idx * K + k];
                
                // Get logit for draft token
                // Check bounds just in case? V is trusted.
                if (draft_token_id >= 0 && draft_token_id < V) {
                    float draft_logit = target_logits[logit_offset + draft_token_id];
                    if (temperature > 0) {
                        draft_logit /= temperature;
                    }
                    
                    // P(target) = exp(logit - max) / sum
                    float p_target = exp(draft_logit - global_max) / global_sum;
                    
                    // Greedy draft: p_draft = 1.0
                    // ratio = p_target / 1.0 = p_target
                    float ratio = p_target; 
                    
                    float r = rand_uniform[batch_idx * K + k];
                    
                    if (r < ratio) {
                        output_mask[batch_idx * K + k] = 1;
                        accepted_count++;
                    } else {
                        output_mask[batch_idx * K + k] = 0;
                        sequence_accepted = false;
                    }
                } else {
                    // Invalid token ID
                    output_mask[batch_idx * K + k] = 0;
                    sequence_accepted = false;
                }
            } else {
                output_mask[batch_idx * K + k] = 0;
            }
        }
        // Need barrier before next iteration to reuse shared memory?
        // Yes, because shared_max/shared_sum are reused.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid.x == 0) {
        output_count[batch_idx] = accepted_count;
    }
}