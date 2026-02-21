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

__attribute__((always_inline))
inline float silu_f32(float x) {
    return x / (1.0f + exp(-x));
}

// Fused gate+up projection kernel
// Computes: gate = x @ gate_W.T, up = x @ up_W.T
// Both projections in a single kernel launch, output concatenated as [gate | up]
kernel void fused_gate_up_gemm(
    // Input [M, K]
    device const half* x [[buffer(0)]],
    
    // Gate Projection weights [K/8, N]
    device const uint* gate_packed [[buffer(1)]],
    device const half* gate_scales [[buffer(2)]],
    
    // Up Projection weights [K/8, N]
    device const uint* up_packed [[buffer(3)]],
    device const half* up_scales [[buffer(4)]],
    
    // Output [M, 2*N] - gate and up concatenated
    device half* output [[buffer(5)]],
    
    // Dimensions: [M, K, N, group_size]
    device const uint* params [[buffer(6)]],
    
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    const uint M = params[0];
    const uint K = params[1];
    const uint N = params[2];
    const uint GROUP_SIZE = params[3];
    
    const uint FP4_PER_UINT = 8;
    const uint TILE_M = 64;
    const uint TILE_N = 64;
    const uint THREADS_PER_TG = 128;
    
    uint thread_idx = tid.x;
    uint simd_id = thread_idx / 32;
    uint simd_lane = thread_idx % 32;
    
    // Tile coordinates
    uint tile_m = tgid.y;
    uint tile_n = tgid.x;
    
    uint row_start = tile_m * TILE_M;
    uint col_start = tile_n * TILE_N;
    
    // Each thread processes one (or more) output elements
    uint local_row = thread_idx / 2;  // 64 rows, 2 columns per thread
    uint local_col = (thread_idx % 2) * 32 + simd_lane;
    
    uint global_row = row_start + local_row;
    uint global_col = col_start + local_col;
    
    if (global_row >= M || global_col >= N) return;
    
    // Accumulators for gate and up projections
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    
    uint k_blocks = K / FP4_PER_UINT;
    
    // Main accumulation loop over K dimension
    for (uint kb = 0; kb < k_blocks; kb++) {
        uint k_base = kb * FP4_PER_UINT;
        uint group_idx = k_base / GROUP_SIZE;
        
        // Load input x values (shared across gate and up)
        half x_vals[8];
        for (uint i = 0; i < 8; i++) {
            x_vals[i] = x[global_row * K + k_base + i];
        }
        
        // Load and dequantize gate weights
        uint gate_packed_val = gate_packed[kb * N + global_col];
        half gate_scale = gate_scales[group_idx * N + global_col];
        half gate_vals[8];
        dequant_fp4x8(gate_packed_val, gate_scale, gate_vals);
        
        // Load and dequantize up weights
        uint up_packed_val = up_packed[kb * N + global_col];
        half up_scale = up_scales[group_idx * N + global_col];
        half up_vals[8];
        dequant_fp4x8(up_packed_val, up_scale, up_vals);
        
        // Accumulate
        for (uint i = 0; i < 8; i++) {
            gate_acc += (float)x_vals[i] * (float)gate_vals[i];
            up_acc += (float)x_vals[i] * (float)up_vals[i];
        }
    }
    
    // Write outputs: gate first half, up second half
    output[global_row * (2 * N) + global_col] = (half)gate_acc;
    output[global_row * (2 * N) + N + global_col] = (half)up_acc;
}

// Fused gate+up+activation kernel.
// Computes: output = silu(x @ gate_W.T) * (x @ up_W.T)
//
// Layouts:
//   x:            [M, K]
//   gate_packed:  [N, K/8]  (row-packed)
//   gate_scales:  [K/group_size, N]
//   up_packed:    [N, K/8]  (row-packed)
//   up_scales:    [K/group_size, N]
//   output:       [M, N]
kernel void mmfp4_fused_gate_up(
    device const half* x [[buffer(0)]],
    device const uint* gate_packed [[buffer(1)]],
    device const half* gate_scales [[buffer(2)]],
    device const uint* up_packed [[buffer(3)]],
    device const half* up_scales [[buffer(4)]],
    device half* output [[buffer(5)]],
    device const uint* params [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    const uint M = params[0];
    const uint K = params[1];
    const uint N = params[2];
    const uint GROUP_SIZE = params[3];

    const uint FP4_PER_UINT = 8;
    const uint TILE_M = 64;
    const uint TILE_N = 64;

    uint thread_idx = tid.x;
    uint simd_lane = thread_idx % 32;

    // Tile coordinates
    uint tile_m = tgid.y;
    uint tile_n = tgid.x;

    uint row_start = tile_m * TILE_M;
    uint col_start = tile_n * TILE_N;

    // 128 threads -> 64 rows x 2 column-halves
    uint local_row = thread_idx / 2;
    uint local_col = (thread_idx % 2) * 32 + simd_lane;

    uint global_row = row_start + local_row;
    uint global_col = col_start + local_col;

    if (global_row >= M || global_col >= N) {
        return;
    }

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    const uint k_blocks = K / FP4_PER_UINT;

    for (uint kb = 0; kb < k_blocks; kb++) {
        uint k_base = kb * FP4_PER_UINT;
        uint group_idx = k_base / GROUP_SIZE;

        half x_vals[8];
        for (uint i = 0; i < 8; i++) {
            x_vals[i] = x[global_row * K + k_base + i];
        }

        // Row-packed [N, K/8]
        uint gate_packed_val = gate_packed[global_col * k_blocks + kb];
        uint up_packed_val = up_packed[global_col * k_blocks + kb];

        half gate_scale = gate_scales[group_idx * N + global_col];
        half up_scale = up_scales[group_idx * N + global_col];

        half gate_vals[8];
        half up_vals[8];
        dequant_fp4x8(gate_packed_val, gate_scale, gate_vals);
        dequant_fp4x8(up_packed_val, up_scale, up_vals);

        for (uint i = 0; i < 8; i++) {
            gate_acc += (float)x_vals[i] * (float)gate_vals[i];
            up_acc += (float)x_vals[i] * (float)up_vals[i];
        }
    }

    output[global_row * N + global_col] = (half)(silu_f32(gate_acc) * up_acc);
}
