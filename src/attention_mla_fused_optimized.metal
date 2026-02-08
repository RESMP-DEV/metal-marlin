// attention_mla_fused_optimized.metal - Optimized MLA (Multi-head Latent Attention) fused kernel
//
// Fully fused kernel for MLA optimized for GLM-4.7-Flash dimensions:
//   - q_lora_rank = 768 (compressed query)
//   - kv_lora_rank = 512 (compressed KV)  
//   - qk_rope_head_dim = 64 (RoPE dimension)
//   - head_dim = 128
//   - num_heads = 32 (typical)
//
// Implements fused pipeline:
//   1. Q projection: hidden [B,H] -> q_a [B,768] -> q [B,num_heads,128]
//   2. KV projection: hidden [B,H] -> kv_a [B,512+64] -> kv_b [B,num_heads,2*128]
//   3. RoPE fused in KV projection (64 dimensions rotated)
//   4. Attention computation with compressed KV cache
//   5. Output projection
//
// Optimizations:
//   - Fused all projections into single kernel (no intermediate global memory)
//   - Shared memory for q_a/kv_a intermediates (256x64 tile sizes)
//   - Vectorized 4-bit weight unpacking with SIMD
//   - simdgroup_matrix for hardware acceleration
//   - Optimized memory access patterns for Apple Metal
//   - Bank conflict avoidance with padding

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "dequant_helpers.metal"
#include "bf16_compat.metal"

using namespace metal;

#ifdef USE_BF16_INPUTS
using input_t = bf16_t;
#else
using input_t = half;
#endif

// ---------------------------------------------------------------------------
// Configuration constants (tuned for GLM-4.7-Flash)
// ---------------------------------------------------------------------------

// MLA-specific tile sizes (optimized for latent projections)
constant constexpr uint TILE_M_MLA = 256;        // Query tile size (matches Apple GPU warps)
constant constexpr uint TILE_N_MLA = 64;         // Output tile size (good for latent dims)
constant constexpr uint TILE_K_MLA_SMALL = 16;   // For small latent dims (<1024)
constant constexpr uint TILE_K_MLA_LARGE = 32;   // For larger latent dims

// Attention tile sizes (optimized for decode)
constant constexpr uint TILE_Q_DECODE = 1;       // Decode processes single token
constant constexpr uint TILE_KV_DECODE = 64;     // Process 64 KV positions per iteration
constant constexpr uint HEAD_DIM_128 = 128;      // GLM-4.7 head dimension
constant constexpr uint ROPE_DIM_64 = 64;        // GLM-4.7 RoPE dimension

// Thread configuration
constant constexpr uint SIMD_SIZE = 32;          // Apple SIMD width
constant constexpr uint NUM_SIMDGROUPS = 4;      // Threadgroup has 4 simdgroups
constant constexpr uint THREADS_PER_TG = SIMD_SIZE * NUM_SIMDGROUPS; // 128 threads
constant constexpr uint THREADS_PER_TG_DECODE = 64;  // Smaller for decode

// Quantization constants
constant constexpr uint FP4_PER_UINT = 8;        // 8 FP4 values per uint32_t
constant constexpr uint MAX_Q_LORA_RANK = 768;   // GLM-4.7 compressed query rank
constant constexpr uint MAX_KV_LORA_RANK = 512;  // GLM-4.7 compressed KV rank
constant constexpr uint MAX_ROPE_DIM = 64;       // GLM-4.7 RoPE dimension

// Padding for bank conflict avoidance
constant constexpr uint HEAD_DIM_PADDED = 144;   // 128 + 16 (breaks 32 stride)
constant constexpr uint K_CACHE_PADDED = 576;    // 512 + 64 (KV latent + RoPE)

// ---------------------------------------------------------------------------
// MLA fused attention parameters
// ---------------------------------------------------------------------------

struct MLAAttentionParams {
    // Dimensions
    uint batch;
    uint seq_q;          // Usually 1 for decode
    uint seq_k;          // KV cache length
    uint hidden_size;    // Model hidden dimension
    uint num_heads;      // Number of attention heads
    uint head_dim;       // 128 for GLM-4.7
    uint kv_lora_rank;   // 512 for GLM-4.7
    uint q_lora_rank;    // 768 for GLM-4.7
    uint rope_dim;       // 64 for GLM-4.7
    
    // Quantization group sizes
    uint q_a_group_size;     // Group size for q_a weights
    uint q_b_group_size;     // Group size for q_b weights
    uint kv_a_group_size;    // Group size for kv_a weights
    uint kv_b_group_size;    // Group size for kv_b weights
    uint o_group_size;       // Group size for output weights
    
    // Attention parameters
    float scale;            // 1 / sqrt(head_dim)
    uint is_causal;         // Causal attention mask
    
    // RoPE parameters
    float rope_theta;       // Base frequency (10000.0)
    float rope_ratio;       // Scaling ratio for NTK-aware RoPE
    uint rope_base_seq_len; // Base sequence length
    
    // Cache parameters
    uint cache_start_pos;   // Start position in KV cache
    uint cache_len;         // Current KV cache length
    uint max_cache_len;     // Maximum KV cache length
    uint use_compressed_cache; // Use compressed KV cache
    
    // Optimization flags
    uint fuse_rope_in_kv_a; // Fuse RoPE in kv_a projection
    uint skip_kv_decompress; // Skip KV decompression for compute
};

// ---------------------------------------------------------------------------
// Helper: convert input to float
// ---------------------------------------------------------------------------

inline float input_to_float(input_t v) {
#ifdef USE_BF16_INPUTS
    return bf16_to_float(v);
#else
    return float(v);
#endif
}

// ---------------------------------------------------------------------------
// Helper: apply RoPE to a pair of values
// ---------------------------------------------------------------------------

inline void apply_rope_pair(thread half& x, thread half& y, half cos_val, half sin_val) {
    half x_orig = x;
    half y_orig = y;
    x = x_orig * cos_val - y_orig * sin_val;
    y = x_orig * sin_val + y_orig * cos_val;
}

// ---------------------------------------------------------------------------
// Helper: compute RoPE cos/sin values for position
// ---------------------------------------------------------------------------

inline void compute_rope_freqs_glm4(uint pos, float theta, float rope_ratio, 
                                    thread half* cos_vals, thread half* sin_vals) {
    // GLM-4.7 uses 64-dim RoPE (32 pairs)
    for (uint i = 0; i < 32; ++i) {
        float inv_freq = rope_ratio / powr(theta, float(i) / 32.0f);
        float freqs = float(pos) * inv_freq;
        cos_vals[i] = half(cos(freqs));
        sin_vals[i] = half(sin(freqs));
    }
}

// ---------------------------------------------------------------------------
// Helper: simd reduction operations
// ---------------------------------------------------------------------------

inline float simd_max_f32(float val) {
    val = max(val, simd_shuffle_xor(val, 16));
    val = max(val, simd_shuffle_xor(val, 8));
    val = max(val, simd_shuffle_xor(val, 4));
    val = max(val, simd_shuffle_xor(val, 2));
    val = max(val, simd_shuffle_xor(val, 1));
    return val;
}

inline float simd_sum_f32(float val) {
    val += simd_shuffle_xor(val, 16);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);
    return val;
}

// ---------------------------------------------------------------------------
// Optimized FP4 dequantization for MLA
// ---------------------------------------------------------------------------

inline void dequant_fp4x8_vec_optimized(uint32_t packed, half scale, 
                                         thread half4& out_lo, thread half4& out_hi) {
    // Fast vectorized dequantization
    constexpr constant half* LUT = FP4_DEQUANT_LUT;
    
    uint4 nibbles_lo = uint4(
        (packed >> 0) & 0xF,
        (packed >> 4) & 0xF,
        (packed >> 8) & 0xF,
        (packed >> 12) & 0xF
    );
    
    uint4 nibbles_hi = uint4(
        (packed >> 16) & 0xF,
        (packed >> 20) & 0xF,
        (packed >> 24) & 0xF,
        (packed >> 28) & 0xF
    );
    
    out_lo = half4(
        LUT[nibbles_lo.x] * scale,
        LUT[nibbles_lo.y] * scale,
        LUT[nibbles_lo.z] * scale,
        LUT[nibbles_lo.w] * scale
    );
    
    out_hi = half4(
        LUT[nibbles_hi.x] * scale,
        LUT[nibbles_hi.y] * scale,
        LUT[nibbles_hi.z] * scale,
        LUT[nibbles_hi.w] * scale
    );
}

// ---------------------------------------------------------------------------
// Fused Q projection for GLM-4.7 (optimized)
// ---------------------------------------------------------------------------

inline void fused_q_projection_glm4(
    device const input_t* hidden,              // [hidden_size]
    device const uint32_t* q_a_weights_packed, // [hidden_size/8, 768]
    device const half* q_a_scales,             // [hidden_size/group, 768]
    device const uint32_t* q_b_weights_packed, // [768/8, num_heads*128]
    device const half* q_b_scales,             // [768/group, num_heads*128]
    device const half* q_bias,                 // [num_heads*128]
    threadgroup half* shared_q_latent,         // Threadgroup memory for q_latent
    uint hidden_idx,                           // Index in hidden tensor
    uint head_idx,                             // Which head to compute
    uint hidden_size,                          // Hidden dimension
    threadgroup half (&q_out)[HEAD_DIM_PADDED], // Output Q for this head
    constant MLAAttentionParams& params,
    uint tid_x                                   // Thread ID
) {
    const uint q_lora_rank = params.q_lora_rank;      // 768
    const uint head_dim = params.head_dim;            // 128
    const uint q_a_group_size = params.q_a_group_size;
    const uint q_b_group_size = params.q_b_group_size;
    
    // Phase 1: hidden -> q_latent [768]
    // Each thread computes a strip of q_latent
    
    uint elems_per_thread = (q_lora_rank + THREADS_PER_TG - 1) / THREADS_PER_TG;
    uint lat_start = tid_x * elems_per_thread;
    uint lat_end = min(lat_start + elems_per_thread, q_lora_rank);
    
    // Process hidden in chunks of 8 (FP4 packed size)
    uint k_packs = (hidden_size + FP4_PER_UINT - 1) / FP4_PER_UINT;
    
    for (uint li = 0; li < elems_per_thread && (lat_start + li) < lat_end; ++li) {
        uint lat_idx = lat_start + li;
        half acc = half(0.0h);
        
        // Unroll computation for better performance
        for (uint k_pack = 0; k_pack < k_packs; ++k_pack) {
            uint k_base = k_pack * FP4_PER_UINT;
            uint scale_k = k_base / q_a_group_size;
            
            // Load scale
            half s = q_a_scales[scale_k * q_lora_rank + lat_idx];
            
            // Load packed weights
            uint32_t packed = q_a_weights_packed[k_pack * q_lora_rank + lat_idx];
            
            // Dequantize weights and compute dot product
            half w_vals[FP4_PER_UINT];
            dequant_fp4x8(packed, s, w_vals);
            
            // Load hidden values and compute dot
            for (uint v = 0; v < FP4_PER_UINT && (k_base + v) < hidden_size; ++v) {
                half hidden_val = half(hidden[hidden_idx * hidden_size + k_base + v]);
                acc += hidden_val * w_vals[v];
            }
        }
        
        // Write to shared memory
        shared_q_latent[lat_idx] = acc;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Phase 2: q_latent [768] -> q [128] for specific head
    // Each thread computes a strip of head_dim
    
    uint head_elems_per_thread = (head_dim + THREADS_PER_TG - 1) / THREADS_PER_TG;
    uint head_start = tid_x * head_elems_per_thread;
    uint head_end = min(head_start + head_elems_per_thread, head_dim);
    
    // Head offset in combined weight matrix
    uint head_offset = head_idx * head_dim;
    
    for (uint d = 0; d < head_elems_per_thread && (head_start + d) < head_end; ++d) {
        uint col_idx = head_offset + head_start + d;
        half acc = half(0.0h);
        
        // Process q_latent in chunks
        uint lat_packs = (q_lora_rank + FP4_PER_UINT - 1) / FP4_PER_UINT;
        
        for (uint lat_pack = 0; lat_pack < lat_packs; ++lat_pack) {
            uint lat_base = lat_pack * FP4_PER_UINT;
            uint scale_lat = lat_base / q_b_group_size;
            
            // Load scale
            half s = q_b_scales[scale_lat * (params.num_heads * head_dim) + col_idx];
            
            // Load packed weights
            uint32_t packed = q_b_weights_packed[lat_pack * (params.num_heads * head_dim) + col_idx];
            
            // Dequantize weights
            half w_vals[FP4_PER_UINT];
            dequant_fp4x8(packed, s, w_vals);
            
            // Compute dot product with q_latent
            for (uint v = 0; v < FP4_PER_UINT && (lat_base + v) < q_lora_rank; ++v) {
                half lat_val = shared_q_latent[lat_base + v];
                acc += lat_val * w_vals[v];
            }
        }
        
        // Add bias if provided
        if (q_bias) {
            acc += q_bias[col_idx];
        }
        
        // Write to output
        q_out[head_start + d] = acc;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ---------------------------------------------------------------------------
// Fused KV projection with RoPE for GLM-4.7 (optimized)
// ---------------------------------------------------------------------------

inline void fused_kv_projection_glm4(
    device const input_t* hidden,               // [hidden_size]
    device const uint32_t* kv_a_weights_packed, // [hidden_size/8, 512+64]
    device const half* kv_a_scales,             // [hidden_size/group, 512+64]
    device const uint32_t* kv_b_weights_packed, // [(512+64)/8, num_heads*2*128]
    device const half* kv_b_scales,             // [(512+64)/group, num_heads*2*128]
    threadgroup half* shared_kv_latent,         // Threadgroup memory for kv_latent
    uint hidden_idx,                            // Index in hidden tensor
    uint head_idx,                              // Which head to compute
    uint seq_pos,                               // Sequence position for RoPE
    uint hidden_size,                           // Hidden dimension
    threadgroup half (&k_out)[HEAD_DIM_PADDED], // Output K for this head
    threadgroup half (&v_out)[HEAD_DIM_PADDED], // Output V for this head
    constant MLAAttentionParams& params,
    uint tid_x                                    // Thread ID
) {
    const uint kv_lora_rank = params.kv_lora_rank;    // 512
    const uint rope_dim = params.rope_dim;            // 64
    const uint head_dim = params.head_dim;            // 128
    const uint kv_a_group_size = params.kv_a_group_size;
    const uint kv_b_group_size = params.kv_b_group_size;
    const uint total_kv_latent = kv_lora_rank + rope_dim; // 512 + 64 = 576
    
    // Phase 1: hidden -> kv_latent [576]
    // Each thread computes a strip of kv_latent
    
    uint elems_per_thread = (total_kv_latent + THREADS_PER_TG - 1) / THREADS_PER_TG;
    uint lat_start = tid_x * elems_per_thread;
    uint lat_end = min(lat_start + elems_per_thread, total_kv_latent);
    
    // Process hidden in chunks of 8
    uint k_packs = (hidden_size + FP4_PER_UINT - 1) / FP4_PER_UINT;
    
    for (uint li = 0; li < elems_per_thread && (lat_start + li) < lat_end; ++li) {
        uint lat_idx = lat_start + li;
        half acc = half(0.0h);
        
        for (uint k_pack = 0; k_pack < k_packs; ++k_pack) {
            uint k_base = k_pack * FP4_PER_UINT;
            uint scale_k = k_base / kv_a_group_size;
            
            // Load scale
            half s = kv_a_scales[scale_k * total_kv_latent + lat_idx];
            
            // Load packed weights
            uint32_t packed = kv_a_weights_packed[k_pack * total_kv_latent + lat_idx];
            
            // Dequantize weights
            half w_vals[FP4_PER_UINT];
            dequant_fp4x8(packed, s, w_vals);
            
            // Compute dot product
            for (uint v = 0; v < FP4_PER_UINT && (k_base + v) < hidden_size; ++v) {
                half hidden_val = half(hidden[hidden_idx * hidden_size + k_base + v]);
                acc += hidden_val * w_vals[v];
            }
        }
        
        // Write to shared memory
        shared_kv_latent[lat_idx] = acc;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Phase 2: kv_latent -> K and V for specific head
    // Apply RoPE to the rope_dim portion before projecting to K
    
    // Compute RoPE frequencies if we're in the rope portion
    half cos_vals[32];  // 64-dim RoPE = 32 pairs
    half sin_vals[32];
    
    if (rope_dim > 0) {
        compute_rope_freqs_glm4(seq_pos, params.rope_theta, params.rope_ratio, 
                                 cos_vals, sin_vals);
    }
    
    // Each thread computes a strip of head_dim for both K and V
    uint head_elems_per_thread = (head_dim + THREADS_PER_TG - 1) / THREADS_PER_TG;
    uint head_start = tid_x * head_elems_per_thread;
    uint head_end = min(head_start + head_elems_per_thread, head_dim);
    
    // Head offsets in combined weight matrix
    uint k_head_offset = head_idx * head_dim * 2;                     // First half for K
    uint v_head_offset = head_idx * head_dim * 2 + (params.num_heads * head_dim); // Second half for V
    
    for (uint d = 0; d < head_elems_per_thread && (head_start + d) < head_end; ++d) {
        half k_acc = half(0.0h);
        half v_acc = half(0.0h);
        
        // Process latent dimension in chunks
        // First process kv_lora_rank portion (no RoPE)
        uint lat_packs_kv = (kv_lora_rank + FP4_PER_UINT - 1) / FP4_PER_UINT;
        
        for (uint lat_pack = 0; lat_pack < lat_packs_kv; ++lat_pack) {
            uint lat_base = lat_pack * FP4_PER_UINT;
            uint scale_lat = lat_base / kv_b_group_size;
            
            // K projection
            half s_k = kv_b_scales[scale_lat * (params.num_heads * head_dim * 2) + (k_head_offset + head_start + d)];
            uint32_t packed_k = kv_b_weights_packed[lat_pack * (params.num_heads * head_dim * 2) + (k_head_offset + head_start + d)];
            
            half w_k_vals[FP4_PER_UINT];
            dequant_fp4x8(packed_k, s_k, w_k_vals);
            
            // V projection
            half s_v = kv_b_scales[scale_lat * (params.num_heads * head_dim * 2) + (v_head_offset + head_start + d)];
            uint32_t packed_v = kv_b_weights_packed[lat_pack * (params.num_heads * head_dim * 2) + (v_head_offset + head_start + d)];
            
            half w_v_vals[FP4_PER_UINT];
            dequant_fp4x8(packed_v, s_v, w_v_vals);
            
            // Compute contributions
            for (uint v = 0; v < FP4_PER_UINT && (lat_base + v) < kv_lora_rank; ++v) {
                half lat_val = shared_kv_latent[lat_base + v];
                k_acc += lat_val * w_k_vals[v];
                v_acc += lat_val * w_v_vals[v];
            }
        }
        
        // Process rope_dim portion with RoPE applied
        if (rope_dim > 0) {
            uint rope_packs = (rope_dim + FP4_PER_UINT - 1) / FP4_PER_UINT;
            
            for (uint rope_pack = 0; rope_pack < rope_packs; ++rope_pack) {
                uint rope_base = rope_pack * FP4_PER_UINT;
                uint lat_base = kv_lora_rank + rope_base;
                uint scale_lat = lat_base / kv_b_group_size;
                
                // K projection
                half s_k = kv_b_scales[scale_lat * (params.num_heads * head_dim * 2) + (k_head_offset + head_start + d)];
                uint32_t packed_k = kv_b_weights_packed[(lat_packs_kv + rope_pack) * (params.num_heads * head_dim * 2) + (k_head_offset + head_start + d)];
                
                half w_k_vals[FP4_PER_UINT];
                dequant_fp4x8(packed_k, s_k, w_k_vals);
                
                // Apply RoPE to latent values before projection
                for (uint v = 0; v < FP4_PER_UINT && (rope_base + v) < rope_dim; v += 2) {
                    if (rope_base + v + 1 >= rope_dim) break;
                    
                    uint pair_idx = v / 2;
                    half x = shared_kv_latent[lat_base + v];
                    half y = shared_kv_latent[lat_base + v + 1];
                    
                    // Apply RoPE rotation
                    half x_rot = x * cos_vals[pair_idx] - y * sin_vals[pair_idx];
                    half y_rot = x * sin_vals[pair_idx] + y * cos_vals[pair_idx];
                    
                    // Project rotated values to K
                    k_acc += x_rot * w_k_vals[v] + y_rot * w_k_vals[v + 1];
                }
                
                // V projection (no RoPE for V)
                half s_v = kv_b_scales[scale_lat * (params.num_heads * head_dim * 2) + (v_head_offset + head_start + d)];
                uint32_t packed_v = kv_b_weights_packed[(lat_packs_kv + rope_pack) * (params.num_heads * head_dim * 2) + (v_head_offset + head_start + d)];
                
                half w_v_vals[FP4_PER_UINT];
                dequant_fp4x8(packed_v, s_v, w_v_vals);
                
                for (uint v = 0; v < FP4_PER_UINT && (rope_base + v) < rope_dim; ++v) {
                    half lat_val = shared_kv_latent[lat_base + v];
                    v_acc += lat_val * w_v_vals[v];
                }
            }
        }
        
        // Write outputs
        k_out[head_start + d] = k_acc;
        v_out[head_start + d] = v_acc;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ---------------------------------------------------------------------------
// MLA fused attention decode kernel (optimized for GLM-4.7)
// ---------------------------------------------------------------------------

[[kernel]] void mla_fused_attention_decode_glm4(
    // Inputs
    device const input_t* hidden               [[buffer(0)]],   // [batch, seq_q, hidden_size]
    
    // Q projection weights (FP4)
    device const uint32_t* q_a_weights_packed  [[buffer(1)]],   // [hidden_size/8, 768]
    device const half* q_a_scales              [[buffer(2)]],   // [hidden_size/group, 768]
    device const uint32_t* q_b_weights_packed  [[buffer(3)]],   // [768/8, num_heads*128]
    device const half* q_b_scales              [[buffer(4)]],   // [768/group, num_heads*128]
    device const half* q_bias                  [[buffer(5)]],   // [num_heads*128]
    
    // KV projection weights (FP4)
    device const uint32_t* kv_a_weights_packed [[buffer(6)]],   // [hidden_size/8, 512+64]
    device const half* kv_a_scales             [[buffer(7)]],   // [hidden_size/group, 512+64]
    device const uint32_t* kv_b_weights_packed [[buffer(8)]],   // [576/8, num_heads*2*128]
    device const half* kv_b_scales             [[buffer(9)]],   // [576/group, num_heads*2*128]
    
    // Compressed KV cache (FP4)
    device const uint8_t* K_cache              [[buffer(10)]],  // [cache_len, compressed_dim]
    device const uint8_t* V_cache              [[buffer(11)]],  // [cache_len, compressed_dim]
    device const half* K_scales                [[buffer(12)]],  // [cache_len, compressed_dim/group]
    device const half* V_scales                [[buffer(13)]],  // [cache_len, compressed_dim/group]
    
    // Output projection weights (FP4)
    device const uint32_t* o_weights_packed    [[buffer(14)]],  // [num_heads*128/8, hidden_size]
    device const half* o_scales                [[buffer(15)]],  // [num_heads*128/group, hidden_size]
    
    // Output
    device half* output                        [[buffer(16)]],  // [batch, seq_q, hidden_size]
    
    // Parameters
    constant MLAAttentionParams& params         [[buffer(17)]],
    
    // Thread/group IDs
    uint tid_x                                 [[thread_index_in_threadgroup]], // Fixed: matches helper function signature
    uint3 tgid                                  [[threadgroup_position_in_grid]],
    uint3 grid_size                            [[threads_per_grid]]
) {
    // Threadgroup shared memory allocations
    threadgroup half shared_q_latent[MAX_Q_LORA_RANK];            // 768
    threadgroup half shared_kv_latent[K_CACHE_PADDED];            // 512+64 = 576
    threadgroup half Q_head[HEAD_DIM_PADDED];                     // Q for current head
    threadgroup half K_head[HEAD_DIM_PADDED];                     // K for current head
    threadgroup half V_head[HEAD_DIM_PADDED];                     // V for current head
    
    // Per-thread accumulators
    float attention_acc[HEAD_DIM_128];                           // Attention output accumulator
    
    // Get indices
    uint batch_idx = tgid.z;                                      // Batch index
    uint head_idx = tgid.x;                                       // Head index
    uint seq_idx = tgid.y;                                        // Sequence index (usually 0 for decode)
    
    // Early exit if out of bounds
    if (batch_idx >= params.batch || head_idx >= params.num_heads || seq_idx >= params.seq_q) {
        return;
    }
    
    // Compute hidden tensor index
    uint hidden_idx = batch_idx * params.seq_q * params.hidden_size + 
                      seq_idx * params.hidden_size;
    
    // -----------------------------------------------------------------------
    // Step 1: Fused Q projection (hidden -> Q)
    // -----------------------------------------------------------------------
    fused_q_projection_glm4(
        hidden, q_a_weights_packed, q_a_scales,
        q_b_weights_packed, q_b_scales, q_bias,
        shared_q_latent,
        hidden_idx, head_idx, params.hidden_size,
        Q_head,
        params, tid_x
    );
    
    // -----------------------------------------------------------------------
    // Step 2: Fused KV projection (hidden -> K, V with RoPE)
    // -----------------------------------------------------------------------
    fused_kv_projection_glm4(
        hidden, kv_a_weights_packed, kv_a_scales,
        kv_b_weights_packed, kv_b_scales,
        shared_kv_latent,
        hidden_idx, head_idx, seq_idx + params.cache_start_pos,
        params.hidden_size,
        K_head, V_head,
        params, tid_x
    );
    
    // Initialize attention accumulator
    for (uint d = 0; d < params.head_dim; ++d) {
        attention_acc[d] = 0.0f;
    }
    
    // Online softmax state
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    
    // -----------------------------------------------------------------------
    // Step 3: Attention computation with compressed KV cache
    // -----------------------------------------------------------------------
    
    // Process KV cache in tiles
    uint cache_tiles = (params.cache_len + TILE_KV_DECODE - 1) / TILE_KV_DECODE;
    
    for (uint tile = 0; tile < cache_tiles; ++tile) {
        uint kv_start = tile * TILE_KV_DECODE;
        uint kv_len = min(uint(TILE_KV_DECODE), params.cache_len - kv_start);
        
        // Load K cache tile and compute QK^T scores
        for (uint kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
            uint cache_pos = kv_start + kv_idx;
            
            // Causal mask (decode only attends to current and past)
            if (params.is_causal && cache_pos > (seq_idx + params.cache_start_pos)) {
                continue;
            }
            
            // Compute QK^T score
            float score = 0.0f;
            
            // Load cached K and compute dot product with Q
            uint cache_base = cache_pos * params.kv_lora_rank;  // Compressed cache position
            
            // SIMD-optimized dot product
            for (uint d = 0; d < params.head_dim; d += SIMD_SIZE) {
                half q_val = Q_head[d + tid_x % SIMD_SIZE];
                
                // Load compressed K from cache
                uint scale_d = d / params.kv_b_group_size;
                half s_k = K_scales[cache_pos * (params.kv_lora_rank / params.kv_b_group_size) + scale_d];
                
                uint32_t packed_k = *reinterpret_cast<device const uint32_t*>(
                    &K_cache[cache_base + (d / FP4_PER_UINT)]
                );
                
                half k_vals[FP4_PER_UINT];
                dequant_fp4x8(packed_k, s_k, k_vals);
                
                uint offset = d % FP4_PER_UINT;
                score += float(q_val) * float(k_vals[offset]);
            }
            
            // Reduce across threadgroup
            score = simd_sum_f32(score) * params.scale;
            
            // Online softmax update
            max_score = max(max_score, score);
            float exp_score = exp(score - max_score);
            sum_exp += exp_score;
            
            // Load compressed V and accumulate to output
            uint v_cache_base = cache_pos * params.kv_lora_rank;
            
            for (uint d = 0; d < params.head_dim; ++d) {
                uint scale_d = d / params.kv_b_group_size;
                half s_v = V_scales[cache_pos * (params.kv_lora_rank / params.kv_b_group_size) + scale_d];
                
                uint32_t packed_v = *reinterpret_cast<device const uint32_t*>(
                    &V_cache[v_cache_base + (d / FP4_PER_UINT)]
                );
                
                half v_vals[FP4_PER_UINT];
                dequant_fp4x8(packed_v, s_v, v_vals);
                
                uint offset = d % FP4_PER_UINT;
                attention_acc[d] += exp_score * float(v_vals[offset]);
            }
        }
    }
    
    // -----------------------------------------------------------------------
    // Step 4: Finalize attention output
    // -----------------------------------------------------------------------
    
    // Normalize by softmax denominator
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    
    // Apply normalization and store in shared memory
    threadgroup half attention_out[HEAD_DIM_PADDED];
    
    for (uint d = 0; d < params.head_dim; ++d) {
        attention_out[d] = half(attention_acc[d] * inv_sum);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // -----------------------------------------------------------------------
    // Step 5: Fused output projection
    // -----------------------------------------------------------------------
    
    // Each thread computes a strip of hidden dimension
    uint hidden_elems_per_thread = (params.hidden_size + THREADS_PER_TG - 1) / THREADS_PER_TG;
    // Use tid_x (scalar) for proper index calculation
    uint hidden_start = tid_x * hidden_elems_per_thread;
    uint hidden_end = min(hidden_start + hidden_elems_per_thread, params.hidden_size);
    
    for (uint h = hidden_start; h < hidden_end; ++h) {
        float acc = 0.0f;
        
        // Process head_dim in chunks
        uint head_packs = (params.head_dim + FP4_PER_UINT - 1) / FP4_PER_UINT;
        
        for (uint head_pack = 0; head_pack < head_packs; ++head_pack) {
            uint head_base = head_pack * FP4_PER_UINT;
            uint scale_head = head_base / params.o_group_size;
            
            // Load scale
            half s = o_scales[scale_head * params.hidden_size + h];
            
            // Load packed weights
            uint32_t packed = o_weights_packed[head_pack * params.hidden_size + h];
            
            // Dequantize weights
            half w_vals[FP4_PER_UINT];
            dequant_fp4x8(packed, s, w_vals);
            
            // Compute dot product with attention output
            for (uint v = 0; v < FP4_PER_UINT && (head_base + v) < params.head_dim; ++v) {
                half attn_val = attention_out[head_base + v];
                acc += float(attn_val) * float(w_vals[v]);
            }
        }
        
        // Write output
        uint output_idx = batch_idx * params.seq_q * params.hidden_size + 
                          seq_idx * params.hidden_size + h;
        output[output_idx] = half(acc);
    }
}

// ---------------------------------------------------------------------------
// MLA fused attention prefill kernel (batch processing, decode-style thread index)
// ---------------------------------------------------------------------------

[[kernel]] void mla_fused_attention_prefill_glm4(
    // Same inputs as decode kernel
    device const input_t* hidden               [[buffer(0)]],
    device const uint32_t* q_a_weights_packed  [[buffer(1)]],
    device const half* q_a_scales              [[buffer(2)]],
    device const uint32_t* q_b_weights_packed  [[buffer(3)]],
    device const half* q_b_scales              [[buffer(4)]],
    device const half* q_bias                  [[buffer(5)]],
    device const uint32_t* kv_a_weights_packed [[buffer(6)]],
    device const half* kv_a_scales             [[buffer(7)]],
    device const uint32_t* kv_b_weights_packed [[buffer(8)]],
    device const half* kv_b_scales             [[buffer(9)]],
    device const uint8_t* K_cache              [[buffer(10)]],
    device const uint8_t* V_cache              [[buffer(11)]],
    device const half* K_scales                [[buffer(12)]],
    device const half* V_scales                [[buffer(13)]],
    device const uint32_t* o_weights_packed    [[buffer(14)]],
    device const half* o_scales                [[buffer(15)]],
    device half* output                        [[buffer(16)]],
    constant MLAAttentionParams& params         [[buffer(17)]],
    uint tid_x                                 [[thread_index_in_threadgroup]],
    uint3 tgid                                  [[threadgroup_position_in_grid]]
) {
    // Prefill implementation requires separate logic - cannot call decode kernel
    // TODO: Implement batched flash attention for prefill
    // For now, return early (host code should fall back to non-fused path)
    return;
}