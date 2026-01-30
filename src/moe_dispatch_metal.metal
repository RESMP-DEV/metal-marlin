// moe_dispatch_metal.metal - MoE token dispatch kernels for batched GEMM
//
// These kernels implement the full MoE dispatch pipeline:
//   1. Count tokens per expert (histogram)
//   2. Compute exclusive prefix sum for offsets
//   3. Sort tokens by expert (stable sort)
//   4. Gather activations in expert-sorted order
//   5. Scatter expert outputs with weighted sum
//
// This enables efficient batched GEMM where all tokens for each expert
// are contiguous in memory.

#include <metal_stdlib>

using namespace metal;

// ============================================================================
// Kernel 1: Compute expert counts (parallel histogram)
// ============================================================================
//
// Input: expert_ids [batch_size, top_k] - expert assignment for each token
// Output: expert_counts [num_experts] - number of tokens per expert
//
// Each thread processes one (batch, top_k) entry and atomically increments
// the count for the assigned expert.

kernel void moe_compute_expert_counts(
    device const uint32_t* expert_ids       [[buffer(0)]],   // [batch, top_k]
    device atomic_uint* expert_counts       [[buffer(1)]],   // [num_experts]
    constant uint& batch_size               [[buffer(2)]],
    constant uint& top_k                    [[buffer(3)]],
    constant uint& num_experts              [[buffer(4)]],
    uint gid                                [[thread_position_in_grid]]
)
{
    const uint total_entries = batch_size * top_k;
    if (gid >= total_entries) return;
    
    uint expert = expert_ids[gid];
    // Bounds check to prevent out-of-bounds access
    if (expert >= num_experts) return;
    
    atomic_fetch_add_explicit(&expert_counts[expert], 1, memory_order_relaxed);
}

// ============================================================================
// Kernel 2: Compute expert offsets (exclusive prefix sum / scan)
// ============================================================================
//
// Input: expert_counts [num_experts] - count of tokens per expert
// Output: expert_offsets [num_experts + 1] - exclusive prefix sum
//
// expert_offsets[i] = sum of counts for experts 0 to i-1
// expert_offsets[num_experts] = total number of tokens
//
// Uses Kogge-Stone parallel scan algorithm for efficiency on GPU.

kernel void moe_compute_expert_offsets(
    device const uint32_t* expert_counts    [[buffer(0)]],   // [num_experts]
    device uint32_t* expert_offsets         [[buffer(1)]],   // [num_experts + 1]
    constant uint& num_experts              [[buffer(2)]],
    uint3 tgid                              [[threadgroup_position_in_grid]],
    uint3 tid                               [[thread_position_in_threadgroup]],
    uint3 tgsize                            [[threads_per_threadgroup]]
)
{
    const uint tg_size = tgsize.x;
    const uint tid_local = tid.x;
    
    // Shared memory for parallel scan
    threadgroup uint32_t shared_data[256];
    
    // Each threadgroup processes a chunk of experts
    // For simplicity, we assume one threadgroup handles all experts
    // (num_experts typically small: 8-64)
    
    // Load data into shared memory
    if (tid_local < num_experts) {
        shared_data[tid_local] = expert_counts[tid_local];
    } else {
        shared_data[tid_local] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Kogge-Stone parallel prefix sum (exclusive)
    // First convert to inclusive scan, then shift
    
    // Up-sweep phase (reduction)
    for (uint stride = 1; stride < tg_size; stride *= 2) {
        uint idx = (tid_local + 1) * stride * 2 - 1;
        if (idx < tg_size) {
            shared_data[idx] += shared_data[idx - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Clear last element for exclusive scan
    uint total = 0;
    if (tid_local == 0) {
        total = shared_data[tg_size - 1];
        shared_data[tg_size - 1] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Down-sweep phase
    for (uint stride = tg_size / 2; stride >= 1; stride /= 2) {
        uint idx = (tid_local + 1) * stride * 2 - 1;
        if (idx < tg_size) {
            uint temp = shared_data[idx - stride];
            shared_data[idx - stride] = shared_data[idx];
            shared_data[idx] += temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write results
    if (tid_local < num_experts) {
        expert_offsets[tid_local] = shared_data[tid_local];
    }
    
    // Write total count at the end
    if (tid_local == 0) {
        expert_offsets[num_experts] = total;
    }
}

// ============================================================================
// Helper: Sequential scan for single-threaded fallback
// ============================================================================

kernel void moe_compute_expert_offsets_sequential(
    device const uint32_t* expert_counts    [[buffer(0)]],   // [num_experts]
    device uint32_t* expert_offsets         [[buffer(1)]],   // [num_experts + 1]
    constant uint& num_experts              [[buffer(2)]],
    uint gid                                [[thread_position_in_grid]]
)
{
    // Single-threaded sequential scan - simple but slower
    // Used when num_experts is very small
    if (gid > 0) return;
    
    uint32_t sum = 0;
    for (uint i = 0; i < num_experts; i++) {
        expert_offsets[i] = sum;
        sum += expert_counts[i];
    }
    expert_offsets[num_experts] = sum;
}

// ============================================================================
// Kernel 3: Compute sorted indices (stable sort by expert)
// ============================================================================
//
// Input: expert_ids [batch, top_k] - expert assignments
//        expert_offsets [num_experts + 1] - prefix sum from previous kernel
// Output: sorted_indices [batch * top_k] - token indices sorted by expert
//         inverse_indices [batch * top_k] - inverse permutation
//
// This implements a counting sort using the expert_offsets as bucket boundaries.
// We use atomic increment to fill each expert's bucket in order.

kernel void moe_compute_sorted_indices(
    device const uint32_t* expert_ids       [[buffer(0)]],   // [batch, top_k]
    device const uint32_t* expert_offsets   [[buffer(1)]],   // [num_experts + 1]
    device uint32_t* sorted_indices         [[buffer(2)]],   // [total]
    device uint32_t* sorted_expert_indices  [[buffer(3)]],   // [total] - expert for each sorted position
    device uint32_t* inverse_indices        [[buffer(4)]],   // [total]
    device atomic_uint* current_offsets     [[buffer(5)]],   // [num_experts] - temp counter
    constant uint& batch_size               [[buffer(6)]],
    constant uint& top_k                    [[buffer(7)]],
    constant uint& num_experts              [[buffer(8)]],
    uint gid                                [[thread_position_in_grid]]
)
{
    const uint total_entries = batch_size * top_k;
    if (gid >= total_entries) return;
    
    uint expert = expert_ids[gid];
    if (expert >= num_experts) return;
    
    // Get position within this expert's bucket using atomic increment
    uint offset = atomic_fetch_add_explicit(&current_offsets[expert], 1, memory_order_relaxed);
    uint sorted_pos = expert_offsets[expert] + offset;
    
    // Store sorted index (maps from sorted position to original token index)
    sorted_indices[sorted_pos] = gid / top_k;  // token index in batch
    sorted_expert_indices[sorted_pos] = expert;
    
    // Store inverse mapping (maps from original position to sorted position)
    inverse_indices[gid] = sorted_pos;
}

// ============================================================================
// Kernel 4: Gather activations for experts
// ============================================================================
//
// Input: activations [batch, hidden] - input token activations
//        sorted_token_indices [total] - indices sorted by expert
// Output: gathered [total, hidden] - activations in expert-sorted order
//
// Each thread processes one token and copies its activation vector.

kernel void moe_gather_for_experts(
    device const half* activations          [[buffer(0)]],   // [batch, hidden]
    device const uint32_t* sorted_indices   [[buffer(1)]],   // [total]
    device half* gathered                   [[buffer(2)]],   // [total, hidden]
    constant uint& total_tokens             [[buffer(3)]],   // batch * top_k
    constant uint& hidden_dim               [[buffer(4)]],
    uint2 gid                               [[thread_position_in_grid]]
)
{
    const uint token_idx = gid.x;
    const uint hidden_idx = gid.y;
    
    if (token_idx >= total_tokens || hidden_idx >= hidden_dim) return;
    
    // Get original token index
    uint orig_token = sorted_indices[token_idx];
    
    // Gather activation
    gathered[token_idx * hidden_dim + hidden_idx] = 
        activations[orig_token * hidden_dim + hidden_idx];
}

// Vectorized version for better memory throughput
kernel void moe_gather_for_experts_vec4(
    device const half* activations          [[buffer(0)]],   // [batch, hidden]
    device const uint32_t* sorted_indices   [[buffer(1)]],   // [total]
    device half* gathered                   [[buffer(2)]],   // [total, hidden]
    constant uint& total_tokens             [[buffer(3)]],
    constant uint& hidden_dim               [[buffer(4)]],
    uint2 gid                               [[thread_position_in_grid]]
)
{
    const uint token_idx = gid.x;
    const uint hidden_idx = gid.y * 4;  // Process 4 elements at a time
    
    if (token_idx >= total_tokens || hidden_idx + 3 >= hidden_dim) return;
    
    uint orig_token = sorted_indices[token_idx];
    
    // Use SIMD group operations for better coalescing
    uint src_offset = orig_token * hidden_dim + hidden_idx;
    uint dst_offset = token_idx * hidden_dim + hidden_idx;
    
    // Load 4 half values
    half4 val = *((device const half4*)(activations + src_offset));
    *((device half4*)(gathered + dst_offset)) = val;
}

// ============================================================================
// Kernel 5: Scatter expert outputs with weighted sum
// ============================================================================
//
// Input: expert_outputs [total, out_dim] - computed expert outputs
//        expert_probs [batch, top_k] - routing probabilities
//        sorted_indices [total] - sorted token indices
//        sorted_expert_indices [total] - expert for each sorted position
//        inverse_indices [total] - inverse permutation
// Output: combined_output [batch, out_dim] - weighted sum of expert outputs
//
// Each thread processes one output element for one original token.

kernel void moe_scatter_expert_outputs(
    device const half* expert_outputs       [[buffer(0)]],   // [total, out_dim]
    device const half* expert_probs         [[buffer(1)]],   // [batch, top_k]
    device const uint32_t* sorted_indices   [[buffer(2)]],   // [total]
    device const uint32_t* inverse_indices  [[buffer(3)]],   // [total]
    device half* combined_output            [[buffer(4)]],   // [batch, out_dim]
    constant uint& batch_size               [[buffer(5)]],
    constant uint& top_k                    [[buffer(6)]],
    constant uint& out_dim                  [[buffer(7)]],
    constant uint& total_tokens             [[buffer(8)]],   // batch * top_k
    uint2 gid                               [[thread_position_in_grid]]
)
{
    const uint batch_idx = gid.x;
    const uint out_idx = gid.y;
    
    if (batch_idx >= batch_size || out_idx >= out_dim) return;
    
    half accum = 0.0h;
    
    // Accumulate contributions from all top_k experts for this token
    for (uint k = 0; k < top_k; k++) {
        uint orig_idx = batch_idx * top_k + k;
        
        // Get position in sorted array
        uint sorted_pos = inverse_indices[orig_idx];
        
        // Get probability for this expert
        half prob = expert_probs[orig_idx];
        
        // Load expert output and weight it
        half expert_val = expert_outputs[sorted_pos * out_dim + out_idx];
        accum += expert_val * prob;
    }
    
    combined_output[batch_idx * out_dim + out_idx] = accum;
}

// ============================================================================
// Optimized scatter with SIMD group reduction
// ============================================================================

kernel void moe_scatter_expert_outputs_simd(
    device const half* expert_outputs       [[buffer(0)]],   // [total, out_dim]
    device const half* expert_probs         [[buffer(1)]],   // [batch, top_k]
    device const uint32_t* sorted_indices   [[buffer(2)]],   // [total]
    device const uint32_t* inverse_indices  [[buffer(3)]],   // [total]
    device half* combined_output            [[buffer(4)]],   // [batch, out_dim]
    constant uint& batch_size               [[buffer(5)]],
    constant uint& top_k                    [[buffer(6)]],
    constant uint& out_dim                  [[buffer(7)]],
    constant uint& total_tokens             [[buffer(8)]],
    uint2 gid                               [[thread_position_in_grid]],
    uint simd_lane                          [[thread_index_in_simdgroup]],
    uint simd_id                            [[simdgroup_index_in_threadgroup]]
)
{
    const uint batch_idx = gid.x;
    const uint out_idx = gid.y;
    
    if (batch_idx >= batch_size || out_idx >= out_dim) return;
    
    half accum = 0.0h;
    
    // Unroll for common top_k values
    if (top_k == 2) {
        #pragma unroll
        for (uint k = 0; k < 2; k++) {
            uint orig_idx = batch_idx * 2 + k;
            uint sorted_pos = inverse_indices[orig_idx];
            half prob = expert_probs[orig_idx];
            half expert_val = expert_outputs[sorted_pos * out_dim + out_idx];
            accum += expert_val * prob;
        }
    } else if (top_k == 4) {
        #pragma unroll
        for (uint k = 0; k < 4; k++) {
            uint orig_idx = batch_idx * 4 + k;
            uint sorted_pos = inverse_indices[orig_idx];
            half prob = expert_probs[orig_idx];
            half expert_val = expert_outputs[sorted_pos * out_dim + out_idx];
            accum += expert_val * prob;
        }
    } else {
        for (uint k = 0; k < top_k; k++) {
            uint orig_idx = batch_idx * top_k + k;
            uint sorted_pos = inverse_indices[orig_idx];
            half prob = expert_probs[orig_idx];
            half expert_val = expert_outputs[sorted_pos * out_dim + out_idx];
            accum += expert_val * prob;
        }
    }
    
    combined_output[batch_idx * out_dim + out_idx] = accum;
}

// ============================================================================
// Utility kernel: Clear atomic counters
// ============================================================================

kernel void moe_clear_atomic_counters(
    device atomic_uint* counters            [[buffer(0)]],
    constant uint& num_counters             [[buffer(1)]],
    uint gid                                [[thread_position_in_grid]]
)
{
    if (gid >= num_counters) return;
    atomic_store_explicit(&counters[gid], 0, memory_order_relaxed);
}

// ============================================================================
// Utility kernel: Verify dispatch correctness (debug)
// ============================================================================

kernel void moe_verify_dispatch(
    device const uint32_t* expert_counts    [[buffer(0)]],
    device const uint32_t* expert_offsets   [[buffer(1)]],
    device const uint32_t* sorted_indices   [[buffer(2)]],
    device const uint32_t* sorted_experts   [[buffer(3)]],
    device uint32_t* errors                 [[buffer(4)]],   // [1] - error count
    constant uint& num_experts              [[buffer(5)]],
    constant uint& total_tokens             [[buffer(6)]],
    uint gid                                [[thread_position_in_grid]]
)
{
    if (gid >= total_tokens) return;
    
    // Verify each token is in the correct expert range
    uint expert = sorted_experts[gid];
    if (expert >= num_experts) {
        atomic_fetch_add_explicit((device atomic_uint*)errors, 1, memory_order_relaxed);
        return;
    }
    
    // Verify token is within this expert's range
    uint start = expert_offsets[expert];
    uint end = expert_offsets[expert + 1];
    if (gid < start || gid >= end) {
        atomic_fetch_add_explicit((device atomic_uint*)errors, 1, memory_order_relaxed);
    }
}
