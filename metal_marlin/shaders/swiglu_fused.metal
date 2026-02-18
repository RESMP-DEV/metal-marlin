#include <metal_stdlib>
using namespace metal;

// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
inline half silu(half x) {
    return x / (1.0h + exp(-x));
}

// Vectorized SiLU for 4 elements
inline half4 silu_vec(half4 x) {
    return x / (half4(1.0h) + exp(-x));
}

// SwiGLU fused kernel: gate * SiLU(up)
// This kernel fuses the element-wise multiplication with SiLU activation
// to reduce memory bandwidth and kernel launch overhead.
//
// Args:
//   gate: Gate tensor [M, N]
//   up: Up tensor [M, N]  
//   output: Output tensor [M, N] = gate * SiLU(up)
//   dims: [M, N] dimensions
kernel void swiglu_fused(
    device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    device half* output [[buffer(2)]],
    device const uint* dims [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tgsize [[threads_per_threadgroup]]
) {
    uint M = dims[0];
    uint N = dims[1];
    
    // Flatten thread index within threadgroup
    uint thread_idx = tid.y * tgsize.x + tid.x;
    uint total_threads = tgsize.x * tgsize.y;
    
    // Calculate global row (M dimension)
    uint row = tgid.y;
    if (row >= M) return;
    
    // Each threadgroup processes one row
    // Process 4 elements at a time using vectorized loads
    uint col_start = tgid.x * tgsize.x * 4 + thread_idx * 4;
    
    // Base pointers for this row
    device const half* gate_row = gate + row * N;
    device const half* up_row = up + row * N;
    device half* out_row = output + row * N;
    
    // Vectorized path - process 4 elements per thread
    for (uint col = col_start; col + 4 <= N; col += total_threads * 4) {
        // Load 4 elements at once
        half4 gate_vec = *((device const half4*)(gate_row + col));
        half4 up_vec = *((device const half4*)(up_row + col));
        
        // Compute gate * SiLU(up)
        half4 result = gate_vec * silu_vec(up_vec);
        
        // Store 4 elements at once
        *((device half4*)(out_row + col)) = result;
    }
    
    // Scalar tail processing
    uint col_scalar_start = (N / 4) * 4;
    for (uint col = col_scalar_start + thread_idx; col < N; col += total_threads) {
        half g = gate_row[col];
        half u = up_row[col];
        out_row[col] = g * silu(u);
    }
}

// SwiGLU fused kernel for 1D tensors
// Optimized for the common case of [N] shaped inputs
kernel void swiglu_fused_1d(
    device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    device half* output [[buffer(2)]],
    device const uint* N [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tgsize [[threads_per_threadgroup]]
) {
    uint n = N[0];
    uint global_idx = tgid.x * tgsize.x + tid.x;
    uint stride = tgsize.x;
    
    // Vectorized path
    uint vec_end = (n / 4) * 4;
    for (uint i = global_idx * 4; i < vec_end; i += stride * 4) {
        half4 gate_vec = *((device const half4*)(gate + i));
        half4 up_vec = *((device const half4*)(up + i));
        half4 result = gate_vec * silu_vec(up_vec);
        *((device half4*)(output + i)) = result;
    }
    
    // Scalar tail
    for (uint i = vec_end + global_idx; i < n; i += stride) {
        output[i] = gate[i] * silu(up[i]);
    }
}
