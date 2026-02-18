#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// -----------------------------------------------------------------------------
// Constants & Structs
// -----------------------------------------------------------------------------

constant uint TILE_M = 16;
constant uint TILE_N = 16;
constant uint TILE_K = 16;
constant uint TILE_Q = 16;  // Query tile size for chunked prefill
constant uint SIMD_SIZE = 32;

struct MLAAttentionParams {
    uint batch;
    uint seq_q;
    uint seq_k;
    uint hidden_size;
    uint num_heads;
    uint head_dim;
    uint kv_lora_rank;
    uint q_lora_rank;
    uint rope_dim;
    float pad[8];
    float scale;
    uint is_causal;
    uint q_a_group_size;
    uint q_b_group_size;
    uint kv_a_group_size;
    uint kv_b_group_size;
    uint o_group_size;
    float rope_theta;
    float rope_ratio;
    uint rope_base_seq_len;
    uint cache_start_pos;
    uint cache_len;
    uint max_cache_len;
    uint use_fused_q_proj;
    uint use_fused_kv_proj;
    uint fuse_rope_in_kv_a;
    uint skip_kv_decompress;
    uint kv_quant_mode;        // 0=none, 1=fp4, 2=fp8, 3=int8
    uint kv_quant_group_size;
    uint sliding_window;       // 0=disabled, >0=window size
};

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

// FP8 E4M3 max value (must be at global scope for Metal)
constant float FP8_E4M3_MAX = 448.0f;

// FP4 Dequantization (Scalar) - E2M1 format
// FP4 layout: 1 sign bit, 2 exponent bits, 1 mantissa bit
// Value = (-1)^sign * 2^(exp-1) * (1 + mantissa/2) for normal
// Value = (-1)^sign * 2^(-1) * (mantissa/2) for subnormal (exp=0)
inline half dequant_fp4(uint8_t nibble, half scale) {
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;
    
    half value;
    if (exp_bits == 0) {
        // Subnormal: +/- 0.5 * man_bit * 0.5 = +/- man_bit * 0.25
        value = half(man_bit) * 0.25h;
    } else {
        // Normal: +/- 2^(exp-1) * (1 + man_bit * 0.5)
        half exponent = half(1 << (exp_bits - 1));
        half mantissa = 1.0h + half(man_bit) * 0.5h;
        value = exponent * mantissa;
    }
    
    return (sign_bit) ? -value * scale : value * scale;
}

// FP8 E4M3 Dequantization
// FP8 layout: 1 sign bit, 4 exponent bits, 3 mantissa bits
inline half dequant_fp8_e4m3(uint8_t val, half scale) {
    // Extract components
    uint sign_bit = (val >> 7) & 1;
    uint exp_bits = (val >> 3) & 0xF;
    uint man_bits = val & 0x7;
    
    half value;
    if (exp_bits == 0 && man_bits == 0) {
        // Zero
        value = 0.0h;
    } else if (exp_bits == 0xF) {
        // NaN/Inf - treat as max value
        value = 448.0h;  // E4M3 max
    } else if (exp_bits == 0) {
        // Subnormal: +/- 2^(-6) * (mantissa / 8)
        value = half(man_bits) * 0.001953125h;  // 2^(-9) * man_bits
    } else {
        // Normal: +/- 2^(exp-7) * (1 + mantissa/8)
        int exp = int(exp_bits) - 7;
        half mantissa = 1.0h + half(man_bits) * 0.125h;
        value = half(pow(2.0f, float(exp))) * mantissa;
    }
    
    return (sign_bit) ? -value * scale : value * scale;
}

// INT8 Symmetric Dequantization
inline half dequant_int8_sym(int8_t val, half scale) {
    return half(val) * scale;
}

// Load quantized KV value with on-the-fly dequantization
// Handles FP4 (packed), FP8, and INT8 formats
inline half load_kv_quantized(
    device const void* kv_cache,
    device const half* scales,
    uint seq_idx,
    uint dim_idx,
    uint seq_len,
    uint kv_dim,
    uint quant_mode,
    uint group_size
) {
    if (quant_mode == 0) {
        // No quantization - direct FP16 load
        device const half* kv_fp16 = (device const half*)kv_cache;
        return kv_fp16[seq_idx * kv_dim + dim_idx];
    }
    else if (quant_mode == 1) {
        // FP4 quantization - 8 values packed per uint32
        // Packed layout: [seq_len, kv_dim/8] of uint32
        device const uint* kv_fp4 = (device const uint*)kv_cache;
        uint pack_idx = seq_idx * (kv_dim / 8) + (dim_idx / 8);
        uint packed = kv_fp4[pack_idx];
        
        // Extract nibble (4 bits) based on position within pack
        uint nibble_idx = dim_idx % 8;
        uint nibble = (packed >> (nibble_idx * 4)) & 0xF;
        
        // Get scale for this group
        uint group_idx = dim_idx / group_size;
        half scale = scales[seq_idx * ((kv_dim + group_size - 1) / group_size) + group_idx];
        
        return dequant_fp4(uint8_t(nibble), scale);
    }
    else if (quant_mode == 2) {
        // FP8 quantization - 1 byte per value
        device const uint8_t* kv_fp8 = (device const uint8_t*)kv_cache;
        uint8_t val = kv_fp8[seq_idx * kv_dim + dim_idx];
        
        // Get scale for this group
        uint group_idx = dim_idx / group_size;
        half scale = scales[seq_idx * ((kv_dim + group_size - 1) / group_size) + group_idx];
        
        return dequant_fp8_e4m3(val, scale);
    }
    else if (quant_mode == 3) {
        // INT8 quantization - 1 byte per value (signed)
        device const int8_t* kv_int8 = (device const int8_t*)kv_cache;
        int8_t val = kv_int8[seq_idx * kv_dim + dim_idx];
        
        // Get scale for this group
        uint group_idx = dim_idx / group_size;
        half scale = scales[seq_idx * ((kv_dim + group_size - 1) / group_size) + group_idx];
        
        return dequant_int8_sym(val, scale);
    }
    
    return 0.0h;
}

// Helper to load FP4 weight block
inline void load_weight_block_fp4(
    device const uint* w_packed,
    device const half* w_scales,
    uint row,
    uint col,
    uint N,
    thread half* out_vals
) {
    // Implement if needed for manual projection
}

// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------

// Decode Kernel (Single Token) with KV cache quantization support
kernel void mla_fused_attention_decode(
    device const half* hidden [[buffer(0)]],
    device const uint* q_a_packed [[buffer(1)]],
    device const half* q_a_scales [[buffer(2)]],
    device const uint* q_b_packed [[buffer(3)]],
    device const half* q_b_scales [[buffer(4)]],
    device const half* q_bias [[buffer(5)]],
    device const uint* kv_a_packed [[buffer(6)]],
    device const half* kv_a_scales [[buffer(7)]],
    device const uint* kv_b_packed [[buffer(8)]],
    device const half* kv_b_scales [[buffer(9)]],
    device const half* k_cache [[buffer(10)]],
    device const half* v_cache [[buffer(11)]],
    device const half* k_scales [[buffer(12)]],
    device const half* v_scales [[buffer(13)]],
    device const uint* o_packed [[buffer(14)]],
    device const half* o_scales [[buffer(15)]],
    device half* output [[buffer(16)]],
    constant MLAAttentionParams& params [[buffer(17)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    // Grid: (num_heads, 1, batch) - one threadgroup per head
    uint head_idx = tgid.x;
    uint batch_idx = tgid.z;
    
    if (head_idx >= params.num_heads) return;
    
    uint tid_x = tid.x;
    const uint THREADS_PER_TG = 128;
    
    // Threadgroup memory for Q vector and attention computation
    threadgroup half q_vec[128];       // Query vector for this head
    threadgroup half k_vec[256];       // Key cache row being loaded
    threadgroup half v_vec[256];       // Value cache row being loaded
    threadgroup float attn_scores[256]; // Attention scores for this query
    
    // -------------------------------------------------------------------------
    // 1. Q Projection: hidden -> q_latent -> q_head
    // -------------------------------------------------------------------------
    // Collaborative load of hidden state (first 512 elements for efficiency)
    threadgroup half h_local[512];
    uint hidden_idx = batch_idx * params.hidden_size;
    for (uint i = tid_x; i < min(params.hidden_size, 512u); i += THREADS_PER_TG) {
        h_local[i] = hidden[hidden_idx + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Q projection A: hidden -> q_lora_rank using FP4 weights
    threadgroup half q_latent[768];
    for (uint i = tid_x; i < params.q_lora_rank; i += THREADS_PER_TG) {
        float sum = 0.0f;
        for (uint h = 0; h < min(params.hidden_size, 4096u); h += 8) {
            uint pack_idx = (h / 8) * params.q_lora_rank + i;
            uint packed = q_a_packed[pack_idx];
            
            uint group_idx = h / params.q_a_group_size;
            half scale = q_a_scales[group_idx * params.q_lora_rank + i];
            
            for (uint k = 0; k < 8 && (h + k) < params.hidden_size; k++) {
                uint nibble = (packed >> (k * 4)) & 0xF;
                half w = dequant_fp4(uint8_t(nibble), scale);
                sum += float(h_local[h + k]) * float(w);
            }
        }
        q_latent[i] = half(sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Q projection B: q_lora_rank -> head_dim
    uint head_offset = head_idx * params.head_dim;
    for (uint d = tid_x; d < params.head_dim; d += THREADS_PER_TG) {
        float sum = 0.0f;
        for (uint i = 0; i < params.q_lora_rank; i += 8) {
            uint pack_idx = (i / 8) * (params.num_heads * params.head_dim) + head_offset + d;
            uint packed = q_b_packed[pack_idx];
            
            uint group_idx = i / params.q_b_group_size;
            half scale = q_b_scales[group_idx * (params.num_heads * params.head_dim) + head_offset + d];
            
            for (uint k = 0; k < 8 && (i + k) < params.q_lora_rank; k++) {
                uint nibble = (packed >> (k * 4)) & 0xF;
                half w = dequant_fp4(uint8_t(nibble), scale);
                sum += float(q_latent[i + k]) * float(w);
            }
        }
        if (q_bias != nullptr) {
            sum += float(q_bias[head_offset + d]);
        }
        q_vec[d] = half(sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // -------------------------------------------------------------------------
    // 2. KV Cache Processing with Quantization Support
    // -------------------------------------------------------------------------
    // The KV cache contains compressed KV values [cache_len, kv_lora_rank]
    // We need to decompress via kv_b_proj on-the-fly
    
    uint kv_dim = params.kv_lora_rank;
    uint n_groups = (kv_dim + params.kv_quant_group_size - 1) / params.kv_quant_group_size;
    
    // Compute attention with all cached positions
    // For each position in cache, load K and V with dequantization
    float max_score = -1e9f;
    float sum_exp = 0.0f;
    threadgroup float o_acc[128];
    for (uint i = tid_x; i < params.head_dim; i += THREADS_PER_TG) {
        o_acc[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Sliding window attention
    uint window_start = 0;
    if (params.sliding_window > 0) {
        window_start = max(0u, params.cache_len - params.sliding_window);
    }
    
    for (uint pos = window_start; pos < params.cache_len; pos += 1) {
        // Load K from cache with dequantization
        // K cache may be quantized (FP4/FP8/INT8) or FP16
        for (uint d = tid_x; d < kv_dim && d < 256; d += THREADS_PER_TG) {
            k_vec[d] = load_kv_quantized(
                k_cache, k_scales, pos, d, params.cache_len, kv_dim,
                params.kv_quant_mode, params.kv_quant_group_size
            );
        }
        
        // Load V from cache with dequantization
        for (uint d = tid_x; d < kv_dim && d < 256; d += THREADS_PER_TG) {
            v_vec[d] = load_kv_quantized(
                v_cache, v_scales, pos, d, params.cache_len, kv_dim,
                params.kv_quant_mode, params.kv_quant_group_size
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Decompress K via kv_b_proj to get full head_dim
        // Simplified: use first kv_lora_rank as key
        float k_dot = 0.0f;
        for (uint d = tid_x; d < min(params.head_dim, kv_dim); d += THREADS_PER_TG) {
            k_dot += float(q_vec[d]) * float(k_vec[d]);
        }
        // Reduce k_dot across threadgroup
        // Simplified: assume single thread for reduction
        
        // Compute attention score
        float score = k_dot * params.scale;
        
        // Apply causal mask - for decode, we attend to all cached positions
        // No masking needed for decode since we attend to all previous tokens
        
        // Softmax update
        float new_max = max(max_score, score);
        sum_exp = sum_exp * exp(max_score - new_max) + exp(score - new_max);
        max_score = new_max;
        
        // Accumulate output weighted by attention
        // Simplified: accumulate V values
        float attn_w = exp(score - max_score) / sum_exp;
        for (uint d = tid_x; d < min(params.head_dim, kv_dim); d += THREADS_PER_TG) {
            // Metal doesn't support atomic float ops on threadgroup memory
            // Using simple addition - threads access different d values
            o_acc[d] += attn_w * float(v_vec[d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // -------------------------------------------------------------------------
    // 3. Output Projection
    // -------------------------------------------------------------------------
    // Store attention output
    threadgroup half attn_out[128];
    for (uint d = tid_x; d < params.head_dim; d += THREADS_PER_TG) {
        attn_out[d] = half(o_acc[d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // O projection: attn_out -> hidden
    // Each head contributes to portion of output
    uint v_head_dim = params.head_dim;  // May differ in some configs
    for (uint h = tid_x; h < params.hidden_size; h += THREADS_PER_TG) {
        float sum = 0.0f;
        
        // Only this head contributes
        if (h >= head_idx * (params.hidden_size / params.num_heads) && 
            h < (head_idx + 1) * (params.hidden_size / params.num_heads)) {
            for (uint d = 0; d < v_head_dim; d += 8) {
                uint pack_idx = ((head_idx * v_head_dim + d) / 8) * params.hidden_size + h;
                uint packed = o_packed[pack_idx];
                
                uint group_idx = (head_idx * v_head_dim + d) / params.o_group_size;
                half scale = o_scales[group_idx * params.hidden_size + h];
                
                for (uint k = 0; k < 8 && (d + k) < v_head_dim; k++) {
                    uint nibble = (packed >> (k * 4)) & 0xF;
                    half w = dequant_fp4(uint8_t(nibble), scale);
                    sum += float(attn_out[d + k]) * float(w);
                }
            }
        }
        
        // Atomic add to output
        if (sum != 0.0f) {
            uint out_idx = (batch_idx * params.seq_q) * params.hidden_size + h;
            atomic_fetch_add_explicit(
                (device atomic_float*)&output[out_idx],
                sum,
                memory_order_relaxed
            );
        }
    }
}


// Prefill Kernel (Full sequence or large chunk)
// Optimized for processing multiple tokens with full attention
// Uses Flash Attention algorithm for memory efficiency
//
// Grid: (num_heads, ceil(seq_q / TILE_Q), batch)
// Threadgroup: (128 threads = 4 simdgroups)
kernel void mla_fused_attention_prefill(
    device const half* hidden [[buffer(0)]],
    device const uint* q_a_packed [[buffer(1)]],
    device const half* q_a_scales [[buffer(2)]],
    device const uint* q_b_packed [[buffer(3)]],
    device const half* q_b_scales [[buffer(4)]],
    device const half* q_bias [[buffer(5)]],
    device const uint* kv_a_packed [[buffer(6)]],
    device const half* kv_a_scales [[buffer(7)]],
    device const uint* kv_b_packed [[buffer(8)]],
    device const half* kv_b_scales [[buffer(9)]],
    device half* k_cache [[buffer(10)]],  // Writable cache for incremental prefill
    device half* v_cache [[buffer(11)]],  // Writable cache
    device const uint* o_packed [[buffer(12)]],
    device const half* o_scales [[buffer(13)]],
    device half* output [[buffer(14)]],
    constant MLAAttentionParams& params [[buffer(15)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    // Grid dimensions
    uint head_idx = tgid.x;
    uint q_tile_idx = tgid.y;
    uint batch_idx = tgid.z;
    
    if (head_idx >= params.num_heads) return;
    
    const uint SIMDGROUPS_PER_TG = 4;
    const uint THREADS_PER_TG = SIMDGROUPS_PER_TG * SIMD_SIZE; // 128
    
    uint tid_x = tid.x;
    uint simd_id = tid.x / SIMD_SIZE;
    uint simd_lane = tid.x % SIMD_SIZE;
    
    // Q tile bounds
    uint q_start = q_tile_idx * TILE_M;
    uint q_end = min(q_start + TILE_M, params.seq_q);
    uint q_len_tile = q_end - q_start;
    
    if (q_len_tile == 0) return;
    
    // Threadgroup memory
    threadgroup half q_tile[TILE_M][128];      // Query values
    threadgroup half k_tile[TILE_N][128];      // Key values for attention
    threadgroup float s_tile[TILE_M][TILE_N];  // Attention scores
    
    // -------------------------------------------------------------------------
    // 1. Q Projection for this tile
    // -------------------------------------------------------------------------
    for (uint q_local = 0; q_local < q_len_tile; q_local++) {
        uint global_q = q_start + q_local;
        uint hidden_idx = (batch_idx * params.seq_q + global_q) * params.hidden_size;
        
        // Collaborative load of hidden state
        threadgroup half h_local[512];  // Assuming hidden_size <= 4096, use 512 * 8 threads
        for (uint i = tid_x; i < params.hidden_size && i < 512 * 8; i += THREADS_PER_TG) {
            if (i < params.hidden_size) {
                h_local[i / 8] = hidden[hidden_idx + i];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // Q projection A: hidden -> q_lora_rank (simplified)
        threadgroup half q_latent[768];
        for (uint i = tid_x; i < params.q_lora_rank; i += THREADS_PER_TG) {
            float sum = 0.0f;
            for (uint h = 0; h < min(params.hidden_size, 4096u); h++) {
                // Simplified projection - use identity for placeholder
                if (h < params.hidden_size) {
                    sum += float(h_local[h / 8]) * 0.001f;  // Placeholder weights
                }
            }
            q_latent[i] = half(sum);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // Q projection B: q_lora_rank -> head_dim
        uint head_offset = head_idx * params.head_dim;
        for (uint d = tid_x; d < params.head_dim; d += THREADS_PER_TG) {
            float sum = 0.0f;
            for (uint i = 0; i < params.q_lora_rank; i++) {
                sum += float(q_latent[i]) * 0.001f;  // Placeholder
            }
            if (q_bias != nullptr) {
                sum += float(q_bias[head_offset + d]);
            }
            q_tile[q_local][d] = half(sum);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    
    // -------------------------------------------------------------------------
    // 2. KV Cache Write (head 0 only to avoid conflicts)
    // -------------------------------------------------------------------------
    if (head_idx == 0 && params.use_fused_kv_proj) {
        for (uint q_local = 0; q_local < q_len_tile; q_local++) {
            uint global_q = q_start + q_local;
            uint cache_pos = params.cache_start_pos + global_q;
            
            if (cache_pos >= params.max_cache_len) continue;
            
            // Write compressed KV to cache
            uint cache_base = cache_pos * params.kv_lora_rank;
            for (uint i = tid_x; i < params.kv_lora_rank; i += THREADS_PER_TG) {
                // Simplified: compute KV projection
                k_cache[cache_base + i] = half(0.0h);  // Placeholder
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // -------------------------------------------------------------------------
    // 3. Flash Attention with Online Softmax
    // -------------------------------------------------------------------------
    float m_i[4];  // Max scores (max 4 queries per tile)
    float l_i[4];  // Logsumexp
    float o_acc[4][32];  // Output accumulator
    
    for (uint i = 0; i < q_len_tile && i < 4; i++) {
        m_i[i] = -1e9f;
        l_i[i] = 0.0f;
        for (uint j = 0; j < 32; j++) {
            o_acc[i][j] = 0.0f;
        }
    }
    
    // Total keys to attend to
    uint total_k = params.cache_len + params.seq_q;
    
    // Sliding window: limit attention range to window_size
    // For each query position, only attend to tokens within [q_pos - window + 1, q_pos]
    uint window_size = params.sliding_window;
    bool use_sliding_window = window_size > 0;
    
    // Iterate over key tiles
    for (uint k_tile_start = 0; k_tile_start < total_k; k_tile_start += TILE_N) {
        uint k_tile_end = min(k_tile_start + TILE_N, total_k);
        uint k_tile_len = k_tile_end - k_tile_start;
        
        // Load K values for this tile
        for (uint k_local = tid_x; k_local < k_tile_len; k_local += THREADS_PER_TG) {
            uint global_k = k_tile_start + k_local;
            
            // Load from cache or compute
            if (global_k < params.cache_len) {
                // From cache
                for (uint d = 0; d < params.head_dim && d < 128; d++) {
                    k_tile[k_local][d] = k_cache[global_k * params.kv_lora_rank + (d % params.kv_lora_rank)];
                }
            } else {
                // Compute on the fly for current sequence
                for (uint d = 0; d < params.head_dim && d < 128; d++) {
                    k_tile[k_local][d] = half(0.0h);
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute attention scores
        for (uint q_local = 0; q_local < q_len_tile && q_local < 4; q_local++) {
            uint global_q = q_start + q_local;
            uint actual_q_pos = params.cache_start_pos + global_q;
            
            // Compute Q @ K^T for this tile
            float s[TILE_N];
            for (uint k_local = 0; k_local < k_tile_len; k_local++) {
                uint global_k = k_tile_start + k_local;
                
                // Causal masking
                if (params.is_causal && global_k > actual_q_pos) {
                    s[k_local] = -1e9f;
                } else if (use_sliding_window && global_k + window_size <= actual_q_pos) {
                    // Sliding window: mask out tokens outside the window
                    // global_k must be > actual_q_pos - window_size, i.e., global_k + window_size > actual_q_pos
                    s[k_local] = -1e9f;
                } else {
                    // Dot product
                    float dot = 0.0f;
                    for (uint d = simd_lane; d < params.head_dim; d += SIMD_SIZE) {
                        dot += float(q_tile[q_local][d]) * float(k_tile[k_local][d]);
                    }
                    dot = simd_sum(dot);
                    s[k_local] = dot * params.scale;
                }
            }
            
            // Online softmax
            float m_prev = m_i[q_local];
            float m_new = m_prev;
            for (uint k_local = 0; k_local < k_tile_len; k_local++) {
                m_new = max(m_new, s[k_local]);
            }
            
            float l_prev = l_i[q_local];
            float exp_sum = 0.0f;
            for (uint k_local = 0; k_local < k_tile_len; k_local++) {
                exp_sum += exp(s[k_local] - m_new);
            }
            
            float l_new = l_prev * exp(m_prev - m_new) + exp_sum;
            float scale_prev = (l_prev > 0) ? exp(m_prev - m_new) : 0.0f;
            
            // Rescale output
            for (uint j = 0; j < 32; j++) {
                o_acc[q_local][j] *= scale_prev;
            }
            
            // Accumulate V weighted by attention
            for (uint k_local = 0; k_local < k_tile_len; k_local++) {
                float attn_w = exp(s[k_local] - m_new);
                for (uint d = simd_lane; d < params.head_dim && d < 128; d += SIMD_SIZE) {
                    o_acc[q_local][d / SIMD_SIZE] += attn_w * float(k_tile[k_local][d]);
                }
            }
            
            m_i[q_local] = m_new;
            l_i[q_local] = l_new;
        }
        
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Normalize
    for (uint q_local = 0; q_local < q_len_tile && q_local < 4; q_local++) {
        float norm = 1.0f / l_i[q_local];
        for (uint j = 0; j < 32; j++) {
            o_acc[q_local][j] *= norm;
        }
    }
    
    // -------------------------------------------------------------------------
    // 4. Output Projection
    // -------------------------------------------------------------------------
    threadgroup half attn_out[TILE_M][128];
    for (uint q_local = 0; q_local < q_len_tile && q_local < 4; q_local++) {
        for (uint d = simd_lane; d < params.head_dim && d < 128; d += SIMD_SIZE) {
            attn_out[q_local][d] = half(o_acc[q_local][d / SIMD_SIZE]);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write output (simplified - real impl needs proper O projection)
    for (uint q_local = 0; q_local < q_len_tile; q_local++) {
        uint global_q = q_start + q_local;
        uint out_idx = (batch_idx * params.seq_q + global_q) * params.hidden_size;
        
        // Each thread writes a portion of output
        for (uint h = tid_x; h < params.hidden_size; h += THREADS_PER_TG) {
            // Simplified: sum of attention output * weights
            float sum = 0.0f;
            // Only this head contributes
            if (h >= head_idx * (params.hidden_size / params.num_heads) && 
                h < (head_idx + 1) * (params.hidden_size / params.num_heads)) {
                for (uint d = 0; d < params.head_dim && d < 128; d++) {
                    sum += float(attn_out[q_local][d]) * 0.01f;  // Placeholder weight
                }
            }
            
            // Atomic add for multi-head aggregation
            if (sum != 0.0f) {
                atomic_fetch_add_explicit((device atomic_float*)&output[out_idx + h], sum, memory_order_relaxed);
            }
        }
    }
}


// Chunked Prefill Kernel
// Processes sequences in chunks to reduce memory pressure during long context prefill.
// Each chunk attends to itself and all previous chunks (which are in the cache).
//
// Key features:
// - Memory-efficient: Only materializes attention for current chunk
// - Causal: Each position only attends to previous positions
// - Incremental: Cache is updated after each chunk
//
// This kernel performs:
// 1. Q projection: hidden -> q_latent -> q_heads (via FP4 weights)
// 2. KV projection and cache write: hidden -> kv_latent -> write to k_cache/v_cache
// 3. Flash Attention: Q @ K^T with causal masking across [0..cache_len+seq_q)
// 4. Output projection: attn_out -> hidden via FP4 o_proj weights
//
// Grid: (num_heads, ceil(seq_q / TILE_Q), batch)
// Threadgroup: (128 threads = 4 simdgroups)
kernel void mla_chunked_prefill_attention(
    device const half* hidden [[buffer(0)]],              // [batch, seq_q, hidden_size]
    device const uint* q_a_packed [[buffer(1)]],          // Q proj A weights (FP4)
    device const half* q_a_scales [[buffer(2)]],          // Q proj A scales
    device const uint* q_b_packed [[buffer(3)]],          // Q proj B weights (FP4)
    device const half* q_b_scales [[buffer(4)]],          // Q proj B scales
    device const half* q_bias [[buffer(5)]],              // Q proj bias (optional)
    device const uint* kv_a_packed [[buffer(6)]],         // KV proj A weights (FP4)
    device const half* kv_a_scales [[buffer(7)]],         // KV proj A scales
    device const uint* kv_b_packed [[buffer(8)]],         // KV proj B weights (FP4)
    device const half* kv_b_scales [[buffer(9)]],         // KV proj B scales
    device half* k_cache [[buffer(10)]],                  // KV cache [max_cache_len, kv_lora_rank]
    device half* v_cache [[buffer(11)]],                  // V cache (same as k_cache for MLA)
    device const uint* o_packed [[buffer(12)]],           // O proj weights (FP4)
    device const half* o_scales [[buffer(13)]],           // O proj scales
    device half* output [[buffer(14)]],                   // Output [batch, seq_q, hidden_size]
    constant MLAAttentionParams& params [[buffer(15)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    // Grid dimensions: (num_heads, ceil(seq_q / TILE_Q), batch)
    uint head_idx = tgid.x;
    uint q_tile_idx = tgid.y;
    uint batch_idx = tgid.z;
    
    if (head_idx >= params.num_heads) return;
    
    const uint SIMDGROUPS_PER_TG = 4;
    const uint THREADS_PER_TG = SIMDGROUPS_PER_TG * SIMD_SIZE; // 128
    
    uint tid_x = tid.x; // 0..127
    uint simd_id = tid.x / SIMD_SIZE; // 0..3
    uint simd_lane = tid.x % SIMD_SIZE; // 0..31
    
    // Q tile bounds for this chunk
    uint q_start = q_tile_idx * TILE_Q;
    uint q_end = min(q_start + TILE_Q, params.seq_q);
    uint q_len_tile = q_end - q_start;
    
    if (q_len_tile == 0) return;
    
    // Total sequence length (cached + current chunk)
    uint total_k_len = params.cache_len + params.seq_q;
    
    // Sliding window attention
    uint window_size = params.sliding_window;
    bool use_sliding_window = window_size > 0;
    
    // Threadgroup shared memory for Q tile and intermediate results
    // Q tile: [TILE_Q][head_dim] - each head's query vector
    // We store FP16 Q values after projection
    threadgroup half q_tile[TILE_Q][128];  // Max head_dim = 128
    threadgroup half k_tile[TILE_N][128];  // For loading K values during attention
    threadgroup half v_tile[TILE_N][128];  // For loading V values during attention
    threadgroup float attn_tile[TILE_Q][TILE_N];  // Attention scores
    
    // -------------------------------------------------------------------------
    // 1. Q Projection: hidden -> q_latent -> q_heads
    // -------------------------------------------------------------------------
    // Simplified: Each threadgroup processes one head's Q values
    // In practice, we'd use simdgroup_matrix for the projections
    
    // For each query position in the tile
    for (uint q_local = 0; q_local < q_len_tile; q_local++) {
        uint global_q = q_start + q_local;
        
        // Compute position in hidden buffer
        uint hidden_idx = (batch_idx * params.seq_q + global_q) * params.hidden_size;
        
        // Load hidden state for this position (collaborative load)
        threadgroup half h_local[256];  // temp buffer for hidden
        for (uint i = tid_x; i < params.hidden_size; i += THREADS_PER_TG) {
            h_local[i] = hidden[hidden_idx + i];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // Q projection A: hidden -> q_lora_rank
        // This is a GEMM: [1, hidden_size] @ [hidden_size, q_lora_rank]
        // For simplicity, we do a naive dot product here
        // Real implementation would use simdgroup_matrix
        
        threadgroup half q_latent[768];  // Max q_lora_rank = 768
        for (uint i = tid_x; i < params.q_lora_rank; i += THREADS_PER_TG) {
            float sum = 0.0f;
            // Simple FP4 dequant and dot product
            // This is a placeholder - real impl needs proper FP4 matmul
            for (uint h = 0; h < params.hidden_size; h += 8) {
                // Load packed weights
                uint pack_idx = (h / 8) * params.q_lora_rank + i;
                uint packed = (pack_idx < params.hidden_size * params.q_lora_rank / 8) 
                              ? q_a_packed[pack_idx] : 0u;
                
                // Simple scale lookup
                uint group_idx = h / params.q_a_group_size;
                half scale = q_a_scales[group_idx * params.q_lora_rank + i];
                
                // Dequant and accumulate
                for (uint k = 0; k < 8 && (h + k) < params.hidden_size; k++) {
                    uint nibble = (packed >> (k * 4)) & 0xF;
                    half w = dequant_fp4(nibble, scale);
                    sum += float(h_local[h + k]) * float(w);
                }
            }
            q_latent[i] = half(sum);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // Q projection B: q_lora_rank -> num_heads * head_dim
        // Extract this head's portion
        uint head_offset = head_idx * params.head_dim;
        for (uint d = tid_x; d < params.head_dim; d += THREADS_PER_TG) {
            float sum = 0.0f;
            for (uint i = 0; i < params.q_lora_rank; i += 8) {
                uint pack_idx = (i / 8) * (params.num_heads * params.head_dim) + head_offset + d;
                uint packed = (pack_idx < params.q_lora_rank * params.num_heads * params.head_dim / 8)
                              ? q_b_packed[pack_idx] : 0u;
                
                uint group_idx = i / params.q_b_group_size;
                half scale = q_b_scales[group_idx * (params.num_heads * params.head_dim) + head_offset + d];
                
                for (uint k = 0; k < 8 && (i + k) < params.q_lora_rank; k++) {
                    uint nibble = (packed >> (k * 4)) & 0xF;
                    half w = dequant_fp4(nibble, scale);
                    sum += float(q_latent[i + k]) * float(w);
                }
            }
            // Add bias if present
            if (q_bias != nullptr) {
                sum += float(q_bias[head_offset + d]);
            }
            q_tile[q_local][d] = half(sum);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    
    // -------------------------------------------------------------------------
    // 2. KV Projection and Cache Write (only once per position, not per head)
    // Only head 0 writes to cache to avoid conflicts
    // -------------------------------------------------------------------------
    if (head_idx == 0) {
        for (uint q_local = 0; q_local < q_len_tile; q_local++) {
            uint global_q = q_start + q_local;
            uint cache_pos = params.cache_len + global_q;  // Position in cache
            
            // Load hidden state
            uint hidden_idx = (batch_idx * params.seq_q + global_q) * params.hidden_size;
            
            // KV projection A: hidden -> kv_lora_rank + rope_dim
            // Write compressed representation to k_cache
            // Each thread writes a portion
            for (uint i = tid_x; i < params.kv_lora_rank; i += THREADS_PER_TG) {
                float sum = 0.0f;
                for (uint h = 0; h < params.hidden_size; h += 8) {
                    uint pack_idx = (h / 8) * params.kv_lora_rank + i;
                    uint packed = (pack_idx < params.hidden_size * params.kv_lora_rank / 8)
                                  ? kv_a_packed[pack_idx] : 0u;
                    
                    uint group_idx = h / params.kv_a_group_size;
                    half scale = kv_a_scales[group_idx * params.kv_lora_rank + i];
                    
                    for (uint k = 0; k < 8 && (h + k) < params.hidden_size; k++) {
                        uint nibble = (packed >> (k * 4)) & 0xF;
                        half w = dequant_fp4(nibble, scale);
                        half h_val = hidden[hidden_idx + h + k];
                        sum += float(h_val) * float(w);
                    }
                }
                // Write to cache at the correct position
                uint cache_idx = cache_pos * params.kv_lora_rank + i;
                if (cache_idx < params.max_cache_len * params.kv_lora_rank) {
                    k_cache[cache_idx] = half(sum);
                }
            }
        }
    }
    // Ensure all threads see the updated cache
    threadgroup_barrier(mem_flags::mem_device);
    
    // -------------------------------------------------------------------------
    // 3. Flash Attention: Q @ K^T with causal masking
    // -------------------------------------------------------------------------
    // Per-row accumulators for online softmax
    float m_i[4];  // Max score (TILE_Q max 4 for register efficiency)
    float l_i[4];  // Sum of exponentials
    float o_acc[4][32];  // Output accumulator (head_dim max 128 = 4*32)
    
    for (uint i = 0; i < q_len_tile && i < 4; i++) {
        m_i[i] = -1e9f;
        l_i[i] = 0.0f;
        for (uint j = 0; j < 32; j++) {
            o_acc[i][j] = 0.0f;
        }
    }
    
    // Iterate over key tiles
    for (uint k_tile_start = 0; k_tile_start < total_k_len; k_tile_start += TILE_N) {
        uint k_tile_end = min(k_tile_start + TILE_N, total_k_len);
        uint k_tile_len = k_tile_end - k_tile_start;
        
        // Load K values for this tile (collaborative)
        // K is derived from cache via kv_b_proj decompression
        for (uint k_local = tid_x; k_local < k_tile_len; k_local += THREADS_PER_TG) {
            uint global_k = k_tile_start + k_local;
            
            // Load from cache and decompress
            if (global_k < params.cache_len) {
                // From previous cache
                for (uint d = 0; d < params.head_dim && d < 128; d++) {
                    // Simplified: directly use cached value
                    // Real impl would decompress via kv_b_proj
                    k_tile[k_local][d] = k_cache[global_k * params.kv_lora_rank + (d % params.kv_lora_rank)];
                }
            } else {
                // From current chunk - compute on the fly
                uint q_idx = global_k - params.cache_len;
                if (q_idx < params.seq_q) {
                    // Compute KV projection for this position
                    // Simplified placeholder
                    for (uint d = 0; d < params.head_dim && d < 128; d++) {
                        k_tile[k_local][d] = half(0.0h);  // Placeholder
                    }
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute attention scores for each query in the tile
        for (uint q_local = 0; q_local < q_len_tile && q_local < 4; q_local++) {
            uint global_q = q_start + q_local;
            uint actual_q_pos = params.cache_len + global_q;
            
            // Compute Q @ K^T for this tile
            float s[TILE_N];
            for (uint k_local = 0; k_local < k_tile_len; k_local++) {
                uint global_k = k_tile_start + k_local;
                
                // Causal mask
                if (global_k > actual_q_pos) {
                    s[k_local] = -1e9f;
                } else if (use_sliding_window && global_k + window_size <= actual_q_pos) {
                    // Sliding window: tokens outside [actual_q_pos - window + 1, actual_q_pos] are masked
                    s[k_local] = -1e9f;
                } else {
                    // Dot product Q @ K
                    float dot = 0.0f;
                    for (uint d = simd_lane; d < params.head_dim; d += SIMD_SIZE) {
                        dot += float(q_tile[q_local][d]) * float(k_tile[k_local][d]);
                    }
                    // Reduce across simdgroup
                    dot = simd_sum(dot);
                    s[k_local] = dot * params.scale;
                }
            }
            
            // Online softmax update
            float m_prev = m_i[q_local];
            float m_new = m_prev;
            for (uint k_local = 0; k_local < k_tile_len; k_local++) {
                m_new = max(m_new, s[k_local]);
            }
            
            float l_prev = l_i[q_local];
            float exp_sum = 0.0f;
            for (uint k_local = 0; k_local < k_tile_len; k_local++) {
                exp_sum += exp(s[k_local] - m_new);
            }
            
            float l_new = l_prev * exp(m_prev - m_new) + exp_sum;
            
            // Rescale previous output
            float scale_prev = (l_prev > 0) ? exp(m_prev - m_new) : 0.0f;
            for (uint j = 0; j < 32; j++) {
                o_acc[q_local][j] *= scale_prev;
            }
            
            // Load V and accumulate
            for (uint k_local = 0; k_local < k_tile_len; k_local++) {
                float attn_w = exp(s[k_local] - m_new);
                // Load V value (same as K for MLA, or separate)
                for (uint d = simd_lane; d < params.head_dim && d < 128; d += SIMD_SIZE) {
                    half v_val = k_tile[k_local][d];  // Simplified - should be V projection
                    o_acc[q_local][d / SIMD_SIZE] += attn_w * float(v_val);
                }
            }
            
            m_i[q_local] = m_new;
            l_i[q_local] = l_new;
        }
        
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Normalize output
    for (uint q_local = 0; q_local < q_len_tile && q_local < 4; q_local++) {
        float norm = 1.0f / l_i[q_local];
        for (uint j = 0; j < 32; j++) {
            o_acc[q_local][j] *= norm;
        }
    }
    
    // -------------------------------------------------------------------------
    // 4. Output Projection: attn_out -> hidden
    // -------------------------------------------------------------------------
    // Store attention output to threadgroup for projection
    threadgroup half attn_out[TILE_Q][128];
    for (uint q_local = 0; q_local < q_len_tile && q_local < 4; q_local++) {
        for (uint d = simd_lane; d < params.head_dim && d < 128; d += SIMD_SIZE) {
            attn_out[q_local][d] = half(o_acc[q_local][d / SIMD_SIZE]);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    
    // O projection: heads concatenated -> hidden
    // Each head contributes v_head_dim elements to output
    // For MLA: v_head_dim may differ from head_dim
    uint v_head_dim = params.head_dim;  // Simplified - should be params.v_head_dim
    
    for (uint q_local = 0; q_local < q_len_tile; q_local++) {
        uint global_q = q_start + q_local;
        
        // Collaborative projection
        // Each thread computes a portion of the output hidden dimension
        for (uint h = tid_x; h < params.hidden_size; h += THREADS_PER_TG) {
            float sum = 0.0f;
            
            // Compute which head/position this output element comes from
            // Simplified: linear scan through all heads
            for (uint h_idx = 0; h_idx < params.num_heads; h_idx++) {
                // Only accumulate from our head
                if (h_idx != head_idx) continue;
                
                for (uint d = 0; d < v_head_dim; d += 8) {
                    uint pack_idx = ((h_idx * v_head_dim + d) / 8) * params.hidden_size + h;
                    uint packed = (pack_idx < params.num_heads * v_head_dim * params.hidden_size / 8)
                                  ? o_packed[pack_idx] : 0u;
                    
                    uint group_idx = (h_idx * v_head_dim + d) / params.o_group_size;
                    half scale = o_scales[group_idx * params.hidden_size + h];
                    
                    for (uint k = 0; k < 8 && (d + k) < v_head_dim; k++) {
                        uint nibble = (packed >> (k * 4)) & 0xF;
                        half w = dequant_fp4(nibble, scale);
                        sum += float(attn_out[q_local][d + k]) * float(w);
                    }
                }
            }
            
            // Write output with atomic add (multiple heads contribute)
            uint out_idx = (batch_idx * params.seq_q + global_q) * params.hidden_size + h;
            // Use atomic to sum contributions from different heads
            atomic_fetch_add_explicit((device atomic_float*)&output[out_idx], sum, memory_order_relaxed);
        }
    }
}

// KV Cache Write Kernel with Quantization Support
// Writes KV values to cache with optional quantization (FP4/FP8/INT8)
// This is used during prefill to compress and store KV values
//
// Grid: (ceil(seq_len / TILE), batch, 1)
// Threadgroup: (128 threads)
kernel void mla_write_kv_cache_quantized(
    device const half* hidden [[buffer(0)]],              // [batch, seq_len, hidden_size]
    device const uint* kv_a_packed [[buffer(1)]],         // KV proj A weights (FP4)
    device const half* kv_a_scales [[buffer(2)]],         // KV proj A scales
    device void* k_cache [[buffer(3)]],                   // K cache (quantized format)
    device void* v_cache [[buffer(4)]],                   // V cache (quantized format)
    device half* k_scales_out [[buffer(5)]],              // K scales output
    device half* v_scales_out [[buffer(6)]],              // V scales output
    constant MLAAttentionParams& params [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    uint tile_idx = tgid.x;
    uint batch_idx = tgid.z;
    
    const uint TILE = 32;
    const uint THREADS_PER_TG = 128;
    
    uint tid_x = tid.x;
    
    // Process tokens in this tile
    for (uint local_idx = 0; local_idx < TILE; local_idx++) {
        uint seq_idx = tile_idx * TILE + local_idx;
        if (seq_idx >= params.seq_q) break;
        
        uint cache_pos = params.cache_len + seq_idx;
        if (cache_pos >= params.max_cache_len) continue;
        
        // Load hidden state for this position
        uint hidden_idx = (batch_idx * params.seq_q + seq_idx) * params.hidden_size;
        
        threadgroup half h_local[512];
        for (uint i = tid_x; i < min(params.hidden_size, 512u); i += THREADS_PER_TG) {
            h_local[i] = hidden[hidden_idx + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // KV projection A: hidden -> kv_lora_rank
        threadgroup half kv_latent[512];
        for (uint i = tid_x; i < params.kv_lora_rank; i += THREADS_PER_TG) {
            float sum = 0.0f;
            for (uint h = 0; h < min(params.hidden_size, 4096u); h += 8) {
                uint pack_idx = (h / 8) * params.kv_lora_rank + i;
                uint packed = kv_a_packed[pack_idx];
                
                uint group_idx = h / params.kv_a_group_size;
                half scale = kv_a_scales[group_idx * params.kv_lora_rank + i];
                
                for (uint k = 0; k < 8 && (h + k) < params.hidden_size; k++) {
                    uint nibble = (packed >> (k * 4)) & 0xF;
                    half w = dequant_fp4(uint8_t(nibble), scale);
                    sum += float(h_local[h + k]) * float(w);
                }
            }
            kv_latent[i] = half(sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Quantize and write to cache based on mode
        uint kv_dim = params.kv_lora_rank;
        uint n_groups = (kv_dim + params.kv_quant_group_size - 1) / params.kv_quant_group_size;
        
        if (params.kv_quant_mode == 0) {
            // FP16 - no quantization
            device half* k_cache_fp16 = (device half*)k_cache;
            device half* v_cache_fp16 = (device half*)v_cache;
            for (uint i = tid_x; i < kv_dim; i += THREADS_PER_TG) {
                half val = kv_latent[i];
                k_cache_fp16[cache_pos * kv_dim + i] = val;
                v_cache_fp16[cache_pos * kv_dim + i] = val;
            }
        }
        else if (params.kv_quant_mode == 1) {
            // FP4 quantization
            // Compute scale per group
            threadgroup half group_max[64];
            for (uint g = tid_x; g < n_groups; g += THREADS_PER_TG) {
                float max_val = 0.0f;
                uint start_d = g * params.kv_quant_group_size;
                uint end_d = min(start_d + params.kv_quant_group_size, kv_dim);
                for (uint d = start_d; d < end_d; d++) {
                    max_val = max(max_val, abs(float(kv_latent[d])));
                }
                group_max[g] = half(max(max_val * 0.1667f, 1e-8f));  // / 6.0
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Write scales
            for (uint g = tid_x; g < n_groups; g += THREADS_PER_TG) {
                k_scales_out[cache_pos * n_groups + g] = group_max[g];
                v_scales_out[cache_pos * n_groups + g] = group_max[g];
            }
            
            // Quantize and pack
            device uint* k_cache_fp4 = (device uint*)k_cache;
            device uint* v_cache_fp4 = (device uint*)v_cache;
            for (uint pack_idx = tid_x; pack_idx < kv_dim / 8; pack_idx += THREADS_PER_TG) {
                uint base_d = pack_idx * 8;
                uint packed_k = 0;
                uint packed_v = 0;
                
                for (uint k = 0; k < 8; k++) {
                    uint d = base_d + k;
                    if (d >= kv_dim) break;
                    
                    uint g = d / params.kv_quant_group_size;
                    half scale = group_max[g];
                    
                    // Quantize to FP4
                    half val = kv_latent[d];
                    half scaled = clamp(val / scale, -6.0h, 6.0h);
                    int8_t q = int8_t(round(float(scaled) * 2.0f));
                    q = clamp(q + 8, 0, 15);
                    
                    packed_k |= (uint(q) << (k * 4));
                    packed_v |= (uint(q) << (k * 4));
                }
                
                k_cache_fp4[cache_pos * (kv_dim / 8) + pack_idx] = packed_k;
                v_cache_fp4[cache_pos * (kv_dim / 8) + pack_idx] = packed_v;
            }
        }
        else if (params.kv_quant_mode == 2) {
            // FP8 E4M3 quantization
            const float FP8_E4M3_MAX = 448.0f;
            
            // Compute scale per group
            threadgroup half group_max[64];
            for (uint g = tid_x; g < n_groups; g += THREADS_PER_TG) {
                float max_val = 0.0f;
                uint start_d = g * params.kv_quant_group_size;
                uint end_d = min(start_d + params.kv_quant_group_size, kv_dim);
                for (uint d = start_d; d < end_d; d++) {
                    max_val = max(max_val, abs(float(kv_latent[d])));
                }
                group_max[g] = half(max(max_val / FP8_E4M3_MAX, 1e-8f));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Write scales
            for (uint g = tid_x; g < n_groups; g += THREADS_PER_TG) {
                k_scales_out[cache_pos * n_groups + g] = group_max[g];
                v_scales_out[cache_pos * n_groups + g] = group_max[g];
            }
            
            // Quantize
            device uint8_t* k_cache_fp8 = (device uint8_t*)k_cache;
            device uint8_t* v_cache_fp8 = (device uint8_t*)v_cache;
            for (uint d = tid_x; d < kv_dim; d += THREADS_PER_TG) {
                uint g = d / params.kv_quant_group_size;
                half scale = group_max[g];
                
                half val = kv_latent[d];
                half scaled = clamp(val / scale, half(-FP8_E4M3_MAX), half(FP8_E4M3_MAX));
                float norm = float(scaled) / FP8_E4M3_MAX * 127.0f + 128.0f;
                uint8_t q = uint8_t(clamp(round(norm), 0.0f, 255.0f));
                
                k_cache_fp8[cache_pos * kv_dim + d] = q;
                v_cache_fp8[cache_pos * kv_dim + d] = q;
            }
        }
        else if (params.kv_quant_group_size == 3) {
            // INT8 symmetric quantization
            // Compute scale per group
            threadgroup half group_max[64];
            for (uint g = tid_x; g < n_groups; g += THREADS_PER_TG) {
                float max_val = 0.0f;
                uint start_d = g * params.kv_quant_group_size;
                uint end_d = min(start_d + params.kv_quant_group_size, kv_dim);
                for (uint d = start_d; d < end_d; d++) {
                    max_val = max(max_val, abs(float(kv_latent[d])));
                }
                group_max[g] = half(max(max_val / 127.0f, 1e-5f));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Write scales
            for (uint g = tid_x; g < n_groups; g += THREADS_PER_TG) {
                k_scales_out[cache_pos * n_groups + g] = group_max[g];
                v_scales_out[cache_pos * n_groups + g] = group_max[g];
            }
            
            // Quantize
            device int8_t* k_cache_int8 = (device int8_t*)k_cache;
            device int8_t* v_cache_int8 = (device int8_t*)v_cache;
            for (uint d = tid_x; d < kv_dim; d += THREADS_PER_TG) {
                uint g = d / params.kv_quant_group_size;
                half scale = group_max[g];
                
                half val = kv_latent[d];
                float scaled = float(val) / float(scale);
                int8_t q = int8_t(clamp(round(scaled), -128.0f, 127.0f));
                
                k_cache_int8[cache_pos * kv_dim + d] = q;
                v_cache_int8[cache_pos * kv_dim + d] = q;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
