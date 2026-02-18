// Paged attention kernel for handling non-contiguous KV blocks
// Used for efficient KV cache management in inference

#include <metal_stdlib>
using namespace metal;

// Constants for block-based KV cache
constexpr constant int BLOCK_SIZE = 16;
constexpr constant int NUM_HEADS = 32;

struct PagedAttentionParams {
    const device float* q;           // Query: [batch, seq_len, num_heads, head_dim]
    const device float* k_cache;     // Key cache: [max_blocks, num_heads, block_size, head_dim]
    const device float* v_cache;     // Value cache: [max_blocks, num_heads, block_size, head_dim]
    const device int32_t* block_table;  // Block table: [batch, max_seq_blocks]
    
    device float* output;            // Output: [batch, seq_len, num_heads, head_dim]
    
    int batch_size;
    int seq_len;
    int num_heads;
    int head_dim;
    int max_blocks;
    int max_seq_blocks;
    
    float scale;                     // Attention scaling factor
};

// Get physical block index for a given logical position
inline int get_physical_block(const device int32_t* block_table,
                               int batch_idx,
                               int logical_block_idx,
                               int max_seq_blocks) {
    return block_table[batch_idx * max_seq_blocks + logical_block_idx];
}

// Get block offset for a given sequence position
inline int get_block_offset(int seq_pos) {
    return seq_pos % BLOCK_SIZE;
}

// Get logical block index for a given sequence position
inline int get_logical_block_idx(int seq_pos) {
    return seq_pos / BLOCK_SIZE;
}

// Load key from non-contiguous KV cache
inline float4 load_key_block(const device float* k_cache,
                             int physical_block,
                             int head_idx,
                             int block_offset,
                             int head_dim,
                             int max_blocks) {
    int base_idx = (physical_block * NUM_HEADS + head_idx) * BLOCK_SIZE * head_dim
                   + block_offset * head_dim;
    return float4(k_cache[base_idx],
                  k_cache[base_idx + 1],
                  k_cache[base_idx + 2],
                  k_cache[base_idx + 3]);
}

// Load value from non-contiguous KV cache
inline float4 load_value_block(const device float* v_cache,
                               int physical_block,
                               int head_idx,
                               int block_offset,
                               int head_dim,
                               int max_blocks) {
    int base_idx = (physical_block * NUM_HEADS + head_idx) * BLOCK_SIZE * head_dim
                   + block_offset * head_dim;
    return float4(v_cache[base_idx],
                  v_cache[base_idx + 1],
                  v_cache[base_idx + 2],
                  v_cache[base_idx + 3]);
}

// Softmax reduction
kernel void paged_attention_softmax(
    device float* attn_scores,
    constant PagedAttentionParams& params,
    uint2 gid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_index_in_threadgroup]]) {
    
    int batch_idx = gid.x;
    int head_idx = gid.y % params.num_heads;
    int seq_pos = gid.y / params.num_heads;
    
    if (batch_idx >= params.batch_size || seq_pos >= params.seq_len) {
        return;
    }
    
    // Load attention scores for current sequence position
    threadgroup float shared_scores[256];
    
    int num_kv_blocks = get_logical_block_idx(seq_pos) + 1;
    
    // Compute max for numerical stability
    float max_score = -INFINITY;
    for (int i = ltid; i < num_kv_blocks * BLOCK_SIZE; i += 256) {
        int kv_pos = i;
        int kv_block_offset = get_block_offset(kv_pos);
        int logical_kv_block = get_logical_block_idx(kv_pos);
        int physical_kv_block = get_physical_block(params.block_table, batch_idx,
                                                    logical_kv_block, params.max_seq_blocks);
        
        if (kv_pos < seq_pos) {
            float score = attn_scores[(batch_idx * params.seq_len * params.num_heads + 
                                       seq_pos * params.num_heads + head_idx) * params.seq_len + kv_pos];
            max_score = max(max_score, score);
        }
    }
    
    shared_scores[ltid] = max_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce to find global max
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (ltid < stride) {
            shared_scores[ltid] = max(shared_scores[ltid], shared_scores[ltid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    max_score = shared_scores[0];
    
    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int i = ltid; i < num_kv_blocks * BLOCK_SIZE; i += 256) {
        int kv_pos = i;
        if (kv_pos < seq_pos) {
            int base_idx = (batch_idx * params.seq_len * params.num_heads + 
                           seq_pos * params.num_heads + head_idx) * params.seq_len + kv_pos;
            float score = attn_scores[base_idx];
            attn_scores[base_idx] = exp(score - max_score);
            sum_exp += attn_scores[base_idx];
        }
    }
    
    shared_scores[ltid] = sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce to compute sum
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (ltid < stride) {
            shared_scores[ltid] += shared_scores[ltid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    sum_exp = shared_scores[0] + 1e-6f;  // Avoid division by zero
    
    // Normalize
    for (int i = ltid; i < num_kv_blocks * BLOCK_SIZE; i += 256) {
        int kv_pos = i;
        if (kv_pos < seq_pos) {
            int base_idx = (batch_idx * params.seq_len * params.num_heads + 
                           seq_pos * params.num_heads + head_idx) * params.seq_len + kv_pos;
            attn_scores[base_idx] /= sum_exp;
        }
    }
}

// Main paged attention kernel
kernel void paged_attention_forward(
    constant PagedAttentionParams& params [[buffer(0)]],
    device float* attn_scores [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_index_in_threadgroup]]) {
    
    int batch_idx = gid.x;
    int head_idx = gid.y % params.num_heads;
    int seq_pos = gid.y / params.num_heads;
    
    if (batch_idx >= params.batch_size || seq_pos >= params.seq_len) {
        return;
    }
    
    // Load query vector
    int q_base_idx = (batch_idx * params.seq_len + seq_pos) * params.num_heads * params.head_dim
                     + head_idx * params.head_dim;
    
    // Compute attention scores with non-contiguous KV blocks
    int num_kv_blocks = get_logical_block_idx(seq_pos) + 1;
    
    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
        int physical_kv_block = get_physical_block(params.block_table, batch_idx,
                                                     kv_block_idx, params.max_seq_blocks);
        
        if (physical_kv_block < 0) continue;  // Invalid block
        
        // Process each position within the block
        for (int offset = 0; offset < BLOCK_SIZE; offset++) {
            int kv_pos = kv_block_idx * BLOCK_SIZE + offset;
            if (kv_pos >= seq_pos) break;
            
            // Compute dot product between Q and K
            float dot_product = 0.0f;
            for (int d = ltid; d < params.head_dim; d += 256) {
                float q_val = params.q[q_base_idx + d];
                
                int k_base_idx = (physical_kv_block * params.num_heads + head_idx) 
                                * BLOCK_SIZE * params.head_dim + offset * params.head_dim + d;
                float k_val = params.k_cache[k_base_idx];
                
                dot_product += q_val * k_val;
            }
            
            threadgroup float shared_dot[256];
            shared_dot[ltid] = dot_product;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Reduce dot product
            for (int stride = 128; stride > 0; stride >>= 1) {
                if (ltid < stride) {
                    shared_dot[ltid] += shared_dot[ltid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            
            // Store scaled attention score
            int score_idx = (batch_idx * params.seq_len * params.num_heads + 
                           seq_pos * params.num_heads + head_idx) * params.seq_len + kv_pos;
            attn_scores[score_idx] = shared_dot[0] * params.scale;
        }
    }
}

// Compute output by weighted sum of values
kernel void paged_attention_output(
    constant PagedAttentionParams& params [[buffer(0)]],
    const device float* attn_scores [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_index_in_threadgroup]]) {
    
    int batch_idx = gid.x;
    int head_idx = gid.y % params.num_heads;
    int seq_pos = gid.y / params.num_heads;
    
    if (batch_idx >= params.batch_size || seq_pos >= params.seq_len) {
        return;
    }
    
    // Compute weighted sum of values
    int num_kv_blocks = get_logical_block_idx(seq_pos) + 1;
    int output_base = (batch_idx * params.seq_len + seq_pos) * params.num_heads * params.head_dim
                      + head_idx * params.head_dim;
    
    for (int d = ltid; d < params.head_dim; d += 256) {
        float sum = 0.0f;
        
        for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
            int physical_kv_block = get_physical_block(params.block_table, batch_idx,
                                                         kv_block_idx, params.max_seq_blocks);
            
            if (physical_kv_block < 0) continue;
            
            for (int offset = 0; offset < BLOCK_SIZE; offset++) {
                int kv_pos = kv_block_idx * BLOCK_SIZE + offset;
                if (kv_pos >= seq_pos) break;
                
                int score_idx = (batch_idx * params.seq_len * params.num_heads + 
                               seq_pos * params.num_heads + head_idx) * params.seq_len + kv_pos;
                float attn_weight = attn_scores[score_idx];
                
                int v_base_idx = (physical_kv_block * params.num_heads + head_idx) 
                                * BLOCK_SIZE * params.head_dim + offset * params.head_dim + d;
                sum += attn_weight * params.v_cache[v_base_idx];
            }
        }
        
        params.output[output_base + d] = sum;
    }
}

// Batched paged attention for multiple sequences
kernel void paged_attention_batched(
    constant PagedAttentionParams& params [[buffer(0)]],
    device float* attn_scores [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]) {
    
    int batch_idx = gid.x;
    int head_idx = gid.y;
    int seq_pos = gid.z;
    
    if (batch_idx >= params.batch_size || 
        head_idx >= params.num_heads || 
        seq_pos >= params.seq_len) {
        return;
    }
    
    // Load query
    int q_base = (batch_idx * params.seq_len + seq_pos) * params.num_heads * params.head_dim
                 + head_idx * params.head_dim;
    
    // Get number of KV blocks for this sequence
    int num_kv_blocks = get_logical_block_idx(seq_pos) + 1;
    
    // Compute attention for each KV position
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int physical_block = get_physical_block(params.block_table, batch_idx,
                                                 kv_block, params.max_seq_blocks);
        if (physical_block < 0) continue;
        
        for (int offset = 0; offset < BLOCK_SIZE; offset++) {
            int kv_pos = kv_block * BLOCK_SIZE + offset;
            if (kv_pos >= seq_pos) break;
            
            // QK dot product
            float score = 0.0f;
            for (int d = 0; d < params.head_dim; d++) {
                float q = params.q[q_base + d];
                int k_base = (physical_block * params.num_heads + head_idx) 
                            * BLOCK_SIZE * params.head_dim + offset * params.head_dim + d;
                float k = params.k_cache[k_base];
                score += q * k;
            }
            
            // Store score
            int score_idx = (batch_idx * params.seq_len * params.num_heads + 
                           seq_pos * params.num_heads + head_idx) * params.seq_len + kv_pos;
            attn_scores[score_idx] = score * params.scale;
        }
    }
}
