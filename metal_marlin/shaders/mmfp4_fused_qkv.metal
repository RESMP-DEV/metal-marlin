#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// FP4 E2M1 branchless dequant (pure ALU, no LUT)
__attribute__((always_inline))
inline half dequant_fp4_scalar(uint nibble) {
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;
    half sub_mag = half(man_bit) * half(0.5h);
    half norm_mag = half(1u << (exp_bits - 1)) * (half(1.0h) + half(man_bit) * half(0.5h));
    half magnitude = select(norm_mag, sub_mag, exp_bits == 0);
    return select(magnitude, -magnitude, bool(sign_bit));
}

// Vectorized FP4x8 dequantization
__attribute__((always_inline))
inline void dequant_fp4x8(uint32_t packed, half scale, thread half *out) {
    for (uint i = 0; i < 8; i++) {
        uint nibble = (packed >> (i * 4)) & 0xF;
        out[i] = dequant_fp4_scalar(nibble) * scale;
    }
}

// Fused Q/K/V projection kernel for decode phase (M=1)
// Computes: Q = A @ Q_W^T, K = A @ K_W^T, V = A @ V_W^T
// All three projections in a single kernel launch
kernel void mmfp4_fused_qkv(
    // Input [1, K]
    device const half* A [[buffer(0)]],
    
    // Q Projection weights [K/8, Nq]
    device const uint* Q_packed [[buffer(1)]],
    device const half* Q_scales [[buffer(2)]],
    
    // K Projection weights [K/8, Nk]
    device const uint* K_packed [[buffer(3)]],
    device const half* K_scales [[buffer(4)]],
    
    // V Projection weights [K/8, Nv]
    device const uint* V_packed [[buffer(5)]],
    device const half* V_scales [[buffer(6)]],
    
    // Outputs
    device half* OutQ [[buffer(7)]],
    device half* OutK [[buffer(8)]],
    device half* OutV [[buffer(9)]],
    
    // Dimensions: [K, Nq, Nk, Nv, group_size]
    device const uint* params [[buffer(10)]],
    
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    const uint K = params[0];
    const uint Nq = params[1];
    const uint Nk = params[2];
    const uint Nv = params[3];
    const uint GROUP_SIZE = params[4];
    
    const uint FP4_PER_UINT = 8;
    const uint TILE_N = 64;
    
    uint thread_idx = tid.x;
    uint simd_id = thread_idx / 32;
    uint simd_lane = thread_idx % 32;
    
    // Each threadgroup processes a tile of N dimension
    // Grid X dimension covers all three outputs: Q, K, V
    uint total_n = Nq + Nk + Nv;
    uint global_col = tgid.x * TILE_N + thread_idx;
    
    if (global_col >= total_n) return;
    
    // Determine which projection this column belongs to
    device const uint* W_packed;
    device const half* W_scales;
    device half* Out;
    uint proj_N;
    
    if (global_col < Nq) {
        W_packed = Q_packed;
        W_scales = Q_scales;
        Out = OutQ;
        proj_N = Nq;
    } else if (global_col < Nq + Nk) {
        W_packed = K_packed;
        W_scales = K_scales;
        Out = OutK;
        proj_N = Nk;
        global_col -= Nq;
    } else {
        W_packed = V_packed;
        W_scales = V_scales;
        Out = OutV;
        proj_N = Nv;
        global_col -= (Nq + Nk);
    }
    
    if (global_col >= proj_N) return;
    
    // Accumulate over K dimension
    float acc = 0.0f;
    uint k_blocks = K / FP4_PER_UINT;
    
    for (uint kb = 0; kb < k_blocks; kb++) {
        uint k_base = kb * FP4_PER_UINT;
        uint group_idx = k_base / GROUP_SIZE;
        
        // Load packed weights and scale
        uint packed = W_packed[kb * proj_N + global_col];
        half scale = W_scales[group_idx * proj_N + global_col];
        
        // Dequantize and accumulate
        half vals[8];
        dequant_fp4x8(packed, scale, vals);
        
        for (uint i = 0; i < 8; i++) {
            acc += (float)A[k_base + i] * (float)vals[i];
        }
    }
    
    Out[global_col] = (half)acc;
}

// Optimized M=1 decode version with threadgroup reduction
kernel void mmfp4_fused_qkv_decode(
    // Input [1, K]
    device const half* A [[buffer(0)]],
    
    // Q Projection weights [K/8, Nq]
    device const uint* Q_packed [[buffer(1)]],
    device const half* Q_scales [[buffer(2)]],
    
    // K Projection weights [K/8, Nk]
    device const uint* K_packed [[buffer(3)]],
    device const half* K_scales [[buffer(4)]],
    
    // V Projection weights [K/8, Nv]
    device const uint* V_packed [[buffer(5)]],
    device const half* V_scales [[buffer(6)]],
    
    // Outputs
    device half* OutQ [[buffer(7)]],
    device half* OutK [[buffer(8)]],
    device half* OutV [[buffer(9)]],
    
    // Dimensions: [K, Nq, Nk, Nv, group_size]
    device const uint* params [[buffer(10)]],
    
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    const uint K = params[0];
    const uint Nq = params[1];
    const uint Nk = params[2];
    const uint Nv = params[3];
    const uint GROUP_SIZE = params[4];
    
    const uint FP4_PER_UINT = 8;
    const uint THREADS_PER_TG = 128;
    
    uint thread_idx = tid.x;
    
    // Process K in chunks, each thread handles partial sum
    uint k_per_thread = (K + THREADS_PER_TG - 1) / THREADS_PER_TG;
    uint k_start = thread_idx * k_per_thread;
    uint k_end = min(k_start + k_per_thread, K);
    
    // Each threadgroup processes all three projections
    // Use threadgroup memory for partial sums
    threadgroup float partial_q[128];
    threadgroup float partial_k[128];
    threadgroup float partial_v[128];
    
    // Initialize partial sums
    partial_q[thread_idx] = 0.0f;
    partial_k[thread_idx] = 0.0f;
    partial_v[thread_idx] = 0.0f;
    
    // Each thread accumulates its slice of K for all outputs
    // This is done cooperatively across the threadgroup
    for (uint col = tgid.x; col < max(Nq, max(Nk, Nv)); col += 256) {
        float acc_q = 0.0f, acc_k = 0.0f, acc_v = 0.0f;
        
        for (uint k = k_start; k < k_end; k += FP4_PER_UINT) {
            if (k >= K) break;
            
            uint kb = k / FP4_PER_UINT;
            uint group_idx = k / GROUP_SIZE;
            
            half a_vals[8];
            for (uint i = 0; i < 8 && (k + i) < K; i++) {
                a_vals[i] = A[k + i];
            }
            
            // Q projection
            if (col < Nq) {
                uint packed = Q_packed[kb * Nq + col];
                half scale = Q_scales[group_idx * Nq + col];
                half vals[8];
                dequant_fp4x8(packed, scale, vals);
                for (uint i = 0; i < 8 && (k + i) < K; i++) {
                    acc_q += (float)a_vals[i] * (float)vals[i];
                }
            }
            
            // K projection
            if (col < Nk) {
                uint packed = K_packed[kb * Nk + col];
                half scale = K_scales[group_idx * Nk + col];
                half vals[8];
                dequant_fp4x8(packed, scale, vals);
                for (uint i = 0; i < 8 && (k + i) < K; i++) {
                    acc_k += (float)a_vals[i] * (float)vals[i];
                }
            }
            
            // V projection
            if (col < Nv) {
                uint packed = V_packed[kb * Nv + col];
                half scale = V_scales[group_idx * Nv + col];
                half vals[8];
                dequant_fp4x8(packed, scale, vals);
                for (uint i = 0; i < 8 && (k + i) < K; i++) {
                    acc_v += (float)a_vals[i] * (float)vals[i];
                }
            }
        }
        
        partial_q[thread_idx] = acc_q;
        partial_k[thread_idx] = acc_k;
        partial_v[thread_idx] = acc_v;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Reduction within threadgroup
        if (thread_idx < 64) {
            partial_q[thread_idx] += partial_q[thread_idx + 64];
            partial_k[thread_idx] += partial_k[thread_idx + 64];
            partial_v[thread_idx] += partial_v[thread_idx + 64];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (thread_idx < 32) {
            partial_q[thread_idx] += partial_q[thread_idx + 32];
            partial_k[thread_idx] += partial_k[thread_idx + 32];
            partial_v[thread_idx] += partial_v[thread_idx + 32];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // simdgroup reduction for final 32
        if (thread_idx < 32) {
            float q_val = partial_q[thread_idx];
            float k_val = partial_k[thread_idx];
            float v_val = partial_v[thread_idx];
            
            q_val += simd_shuffle_xor(q_val, 16);
            q_val += simd_shuffle_xor(q_val, 8);
            q_val += simd_shuffle_xor(q_val, 4);
            q_val += simd_shuffle_xor(q_val, 2);
            q_val += simd_shuffle_xor(q_val, 1);
            
            k_val += simd_shuffle_xor(k_val, 16);
            k_val += simd_shuffle_xor(k_val, 8);
            k_val += simd_shuffle_xor(k_val, 4);
            k_val += simd_shuffle_xor(k_val, 2);
            k_val += simd_shuffle_xor(k_val, 1);
            
            v_val += simd_shuffle_xor(v_val, 16);
            v_val += simd_shuffle_xor(v_val, 8);
            v_val += simd_shuffle_xor(v_val, 4);
            v_val += simd_shuffle_xor(v_val, 2);
            v_val += simd_shuffle_xor(v_val, 1);
            
            if (thread_idx == 0) {
                if (col < Nq) OutQ[col] = (half)q_val;
                if (col < Nk) OutK[col] = (half)k_val;
                if (col < Nv) OutV[col] = (half)v_val;
            }
        }
    }
}
