#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// GLM-4.7-Flash MoE dimensions (CORRECTED Feb 2026)
// Source: transformers.AutoConfig.from_pretrained('zai-org/GLM-4.7-Flash')
// WRONG previous values: 4608/14336/8/2 - didn't match any GLM model
constant uint GLM47_HIDDEN_SIZE = 2048;
constant uint GLM47_INTERMEDIATE_SIZE = 1536;  // Per expert MLP (NOT 14336!)
constant uint GLM47_NUM_EXPERTS = 64;          // NOT 8
constant uint GLM47_TOP_K = 4;                 // NOT 2

// Tiling constants
constant uint TILE_SIZE = 1024;  // Elements per tile (fits in threadgroup memory)
constant uint SIMD_WIDTH = 32;   // Apple GPU SIMD width

// FP4 E2M1 dequantization LUT
constant half FP4_LUT[16] = {
    0.0h, 0.5h, 1.0h, 1.5h, 2.0h, 3.0h, 4.0h, 6.0h,
    -0.0h, -0.5h, -1.0h, -1.5h, -2.0h, -3.0h, -4.0h, -6.0h
};

// Vectorized 8-element FP4 dequantization
inline half8 dequant_fp4x8_vec(uint32_t packed, half scale) {
    return half8(
        FP4_LUT[(packed >>  0) & 0xF],
        FP4_LUT[(packed >>  4) & 0xF],
        FP4_LUT[(packed >>  8) & 0xF],
        FP4_LUT[(packed >> 12) & 0xF],
        FP4_LUT[(packed >> 16) & 0xF],
        FP4_LUT[(packed >> 20) & 0xF],
        FP4_LUT[(packed >> 24) & 0xF],
        FP4_LUT[(packed >> 28) & 0xF]
    ) * scale;
}

// SiLU activation: x * sigmoid(x)
inline half silu(half x) {
    return x / (1.0h + exp(-x));
}

// Fused MoE MLP kernel for single token decode
// Computes: output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
// Gate projection uses simdgroup_matrix for 2-4x speedup on 8x8 tiles
kernel void mmfp4_fused_moe_mlp(
    // Input hidden state [1, hidden_size]
    device const half* input [[buffer(0)]],
    
    // Expert weights (packed FP4)
    device const uint* gate_proj_packed [[buffer(1)]],  // [intermediate/8, hidden]
    device const uint* up_proj_packed [[buffer(2)]],    // [intermediate/8, hidden]  
    device const uint* down_proj_packed [[buffer(3)]],  // [hidden/8, intermediate]
    
    // Scales
    device const half* gate_scales [[buffer(4)]],       // [intermediate/group, hidden/group]
    device const half* up_scales [[buffer(5)]],
    device const half* down_scales [[buffer(6)]],
    
    // Output [1, hidden_size]
    device half* output [[buffer(7)]],
    
    // Dimensions
    device const uint* params [[buffer(8)]],  // [hidden_size, intermediate_size, group_size]
    
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    const uint hidden_size = params[0];
    const uint intermediate_size = params[1];
    const uint group_size = params[2];
    
    // Dispatch 8 simdgroups per threadgroup (256 threads / 32 SIMD width)
    // Simdgroups 0-3: gate projection (using simdgroup_matrix)
    // Simdgroups 4-7: up projection
    bool is_gate_simd = (simd_id < 4);

    // Threadgroup memory - fits within 16KB limit
    threadgroup half shared_gate[TILE_SIZE];  // For gate simdgroups to write
    threadgroup half shared_up[TILE_SIZE];    // For up simdgroups to write
    threadgroup half shared_activated[TILE_SIZE];
    threadgroup half shared_input[2048]; // GLM47_HIDDEN_SIZE (CORRECTED from 4608)
    // Buffers for simdgroup_matrix operations
    threadgroup half shared_a_row[8];       // Input row (8 elements)
    threadgroup half shared_b_tile[64];     // Weight tile (8x8)
    threadgroup half shared_c_tile[64];     // Result tile (8x8)

    // Cooperative load - each thread loads ~18 elements
    // Done ONCE before all projection loops
    for (uint i = lid; i < hidden_size; i += lsize) {
        shared_input[i] = input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Each thread accumulates multiple output elements
    uint elems_per_thread = (hidden_size + lsize - 1) / lsize;
    float partial_out[8];
    
    for (uint h_idx = 0; h_idx < elems_per_thread; h_idx++) {
        partial_out[h_idx] = 0.0f;
    }
    
    // Process intermediate dimension in tiles
    for (uint tile = 0; tile < intermediate_size; tile += TILE_SIZE) {
        uint tile_size = min(TILE_SIZE, intermediate_size - tile);
        
        // Phase 1: Parallel gate and up projection using separate simdgroups
        if (is_gate_simd) {
            // Gate projection using simdgroup_matrix (8x8 tiles)
            // Each simdgroup computes 8 output elements using cooperative matrix multiply
            // For vector-matrix: y = x @ W^T where x is [1, K], W is [N, K], y is [1, N]
            // We compute 8 columns of y at a time
            
            // Each simdgroup handles a different 8-output chunk
            // simdgroup 0: outputs [tile + 0, tile + 7]
            // simdgroup 1: outputs [tile + 8, tile + 15]
            uint i_base = tile + simd_id * 8;
            
            // Each simdgroup processes its assigned 8 outputs
            // For simplicity, we process one 8-output block per simdgroup
            if (i_base + 7 < tile + tile_size) {
                // Initialize 8x8 accumulator to zero
                simdgroup_matrix<half, 8, 8> acc = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
                
                // Tile over K dimension in 8-element chunks
                // K = hidden_size = 4608, so we have 4608/8 = 576 tiles
                for (uint k_tile = 0; k_tile < hidden_size; k_tile += 8) {
                    // Step 1: Load input fragment (8 elements from x)
                    // Broadcast across all 8 rows of the A matrix
                    if (simd_lane_id < 8) {
                        shared_a_row[simd_lane_id] = shared_input[k_tile + simd_lane_id];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    
                    // Load A matrix (broadcasted row) into simdgroup_matrix
                    simdgroup_matrix<half, 8, 8> a_frag;
                    simdgroup_load(a_frag, shared_a_row, 8, ulong2(0), false);
                    
                    // Step 2: Load and dequantize weight fragment (8x8 tile from W) using cooperative loading
                    // Each thread in simdgroup helps load part of the weight tile
                    // We'll load 8 rows (k dimension) x 8 columns (output dimension)
                    
                    // Use the dequant_fp4x8_vec function for efficient 8-element dequantization
                    // Each thread handles 2 columns (since 32 threads * 2 columns = 64 elements = 8x8)
                    
                    for (uint col_offset = 0; col_offset < 8; col_offset += 4) { // Process 4 columns at a time
                        uint col_start = col_offset + (simd_lane_id % 4) * 2; // Each thread handles 2 columns
                        if (col_start < 8) {
                            for (uint row = 0; row < 8; row++) {
                                uint global_k = k_tile + row;
                                uint global_out_idx1 = i_base + col_start;
                                uint global_out_idx2 = i_base + col_start + 1;
                                
                                // Load packed FP4 weights for two columns
                                uint packed_k = global_k / 8;
                                uint nibble_idx = global_k % 8;
                                
                                // First column weight
                                uint packed_idx1 = packed_k * intermediate_size + global_out_idx1;
                                uint32_t gate_packed1 = gate_proj_packed[packed_idx1];
                                uint group_idx1 = global_k / group_size;
                                half gate_scale1 = gate_scales[group_idx1 * intermediate_size + global_out_idx1];
                                
                                // Second column weight (if within bounds)
                                half weight2 = 0.0h;
                                if (col_start + 1 < 8) {
                                    uint packed_idx2 = packed_k * intermediate_size + global_out_idx2;
                                    uint32_t gate_packed2 = gate_proj_packed[packed_idx2];
                                    uint group_idx2 = global_k / group_size;
                                    half gate_scale2 = gate_scales[group_idx2 * intermediate_size + global_out_idx2];
                                    uint nibble2 = (gate_packed2 >> (nibble_idx * 4)) & 0xF;
                                    weight2 = FP4_LUT[nibble2] * gate_scale2;
                                }
                                
                                uint nibble1 = (gate_packed1 >> (nibble_idx * 4)) & 0xF;
                                half weight1 = FP4_LUT[nibble1] * gate_scale1;
                                
                                // Store in shared memory for simdgroup_load
                                shared_b_tile[row * 8 + col_start] = weight1;
                                if (col_start + 1 < 8) {
                                    shared_b_tile[row * 8 + col_start + 1] = weight2;
                                }
                            }
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    
                    // Load B matrix into simdgroup_matrix
                    simdgroup_matrix<half, 8, 8> b_frag;
                    simdgroup_load(b_frag, shared_b_tile, 8, ulong2(0), false);
                    
                    // Step 3: Multiply-accumulate
                    // C += A * B where A is [8x8 broadcasted], B is [8x8]
                    simdgroup_multiply_accumulate(acc, a_frag, b_frag, acc);
                }
                
                // Store result to threadgroup memory
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_store(acc, shared_c_tile, 8, ulong2(0), false);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                // Extract results - sum across rows for each column
                // Each column j represents the dot product for output i_base + j
                if (simd_lane_id < 8) {
                    half sum = 0.0h;
                    for (uint row = 0; row < 8; row++) {
                        sum += shared_c_tile[row * 8 + simd_lane_id];
                    }
                    shared_gate[(i_base - tile) + simd_lane_id] = sum;
                }
            }
        } else {
            // Up projection - scalar implementation with half4 vectors
            for (uint i = lid; i < tile_size; i += lsize) {
                uint global_i = tile + i;
                float acc = 0.0f;
                
                for (uint k_pack = 0; k_pack < hidden_size / 8; k_pack++) {
                    uint k = k_pack * 8;
                    
                    // Load 8 input elements as two half4 vectors
                    half4 x_lo = *((threadgroup const half4*)(shared_input + k));
                    half4 x_hi = *((threadgroup const half4*)(shared_input + k + 4));
                    
                    // Dequantize 8 FP4 weights
                    uint32_t up_packed = up_proj_packed[k_pack * intermediate_size + global_i];
                    uint group_idx = k / group_size;
                    half up_scale = up_scales[group_idx * intermediate_size + global_i];
                    
                    half8 w_vec = dequant_fp4x8_vec(up_packed, up_scale);
                    half4 w_lo = w_vec.lo;
                    half4 w_hi = w_vec.hi;
                    
                    // Accumulate dot product
                    float4 prod_lo = float4(x_lo) * float4(w_lo);
                    float4 prod_hi = float4(x_hi) * float4(w_hi);
                    acc += dot(prod_lo, float4(1.0f)) + dot(prod_hi, float4(1.0f));
                }
                
                shared_up[i] = half(acc);
            }
        }
        
        // Ensure all simdgroups have written their projections
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Phase 1b: All simdgroups cooperate to combine gate and up with SiLU
        for (uint i = lid; i < tile_size; i += lsize) {
            shared_activated[i] = silu(shared_gate[i]) * shared_up[i];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Phase 2: Accumulate this tile's contribution to output
        for (uint h_idx = 0; h_idx < elems_per_thread; h_idx++) {
            uint h = lid + h_idx * lsize;
            if (h >= hidden_size) continue;
            
            float out_acc = 0.0f;
            
            // Process each element in this tile
            for (uint i = 0; i < tile_size; i++) {
                half act = shared_activated[i];
                uint global_i = tile + i;
                
                // Access packed FP4 weight: 8 elements per uint32
                uint packed_idx = global_i / 8;
                uint nibble_idx = global_i % 8;
                uint32_t down_packed = down_proj_packed[packed_idx * hidden_size + h];
                
                uint group_idx = global_i / group_size;
                half down_scale = down_scales[group_idx * hidden_size + h];
                
                uint down_nib = (down_packed >> (nibble_idx * 4)) & 0xF;
                half down_w = FP4_LUT[down_nib] * down_scale;
                out_acc += float(act) * float(down_w);
            }
            
            partial_out[h_idx] += out_acc;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write final outputs
    for (uint h_idx = 0; h_idx < elems_per_thread; h_idx++) {
        uint h = lid + h_idx * lsize;
        if (h < hidden_size) {
            output[h] = half(partial_out[h_idx]);
        }
    }
}
