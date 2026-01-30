// scatter_gather.metal - Index-based tensor operations for MoE routing and attention
//
// Provides efficient GPU-accelerated scatter/gather operations:
//   - gather_rows: Extract rows from matrix by index (torch.index_select dim=0)
//   - scatter_add: Atomically accumulate values at indices (torch.scatter_add)
//   - index_select: General index selection along dimension 0
//   - gather_2d: 2D gather for attention patterns
//
// These operations are fundamental for:
//   - MoE expert routing (gathering expert weights by expert_idx)
//   - KV cache access patterns (gathering past tokens by position)
//   - Attention sparse patterns (index-based selection)
//
// Performance notes:
//   - Use float4 vectorized variants when inner dimension is divisible by 4
//   - Memory access is coalesced by having adjacent threads read adjacent addresses
//   - For large index arrays, consider sorting indices first for cache efficiency

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// gather_rows: Extract rows from a matrix by index
// ============================================================================
// Equivalent to: dst[i, :] = src[indices[i], :]
// PyTorch: torch.index_select(src, dim=0, index=indices)
//
// Optimized for row-major layout with coalesced memory access.
// Each thread handles one element within a row.
//
// Parameters:
//   src: Source matrix [N, D] - float array
//   indices: Row indices to gather [M] - uint array
//   dst: Output matrix [M, D] - float array
//   num_indices: M - number of rows to gather
//   row_stride: D - elements per row (inner dimension)

kernel void gather_rows(
    device const float* src [[buffer(0)]],          // Source matrix [N, D]
    device const uint* indices [[buffer(1)]],       // Row indices [M]
    device float* dst [[buffer(2)]],                // Output [M, D]
    constant uint& num_indices [[buffer(3)]],       // M
    constant uint& row_stride [[buffer(4)]],        // D (elements per row)
    uint2 gid [[thread_position_in_grid]])          // (col_idx, row_idx)
{
    // gid.x = column within row (0 to row_stride-1)
    // gid.y = which index we're processing (0 to num_indices-1)
    
    if (gid.y >= num_indices || gid.x >= row_stride) return;
    
    // Get the source row index for this output row
    uint src_row = indices[gid.y];
    
    // Coalesced read: adjacent threads read adjacent elements in source
    // Coalesced write: adjacent threads write adjacent elements in destination
    dst[gid.y * row_stride + gid.x] = src[src_row * row_stride + gid.x];
}

// FP16 variant for memory bandwidth optimization
kernel void gather_rows_fp16(
    device const half* src [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device half* dst [[buffer(2)]],
    constant uint& num_indices [[buffer(3)]],
    constant uint& row_stride [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= num_indices || gid.x >= row_stride) return;
    
    uint src_row = indices[gid.y];
    dst[gid.y * row_stride + gid.x] = src[src_row * row_stride + gid.x];
}

// ============================================================================
// gather_rows_vec4: Vectorized gather (4x throughput)
// ============================================================================
// Use when row_stride is divisible by 4 for 4x memory bandwidth efficiency.
// Processes 4 floats per thread using float4 vector loads/stores.
//
// Parameters:
//   src: Source matrix as float4 array [N, D/4]
//   indices: Row indices [M]
//   dst: Output as float4 array [M, D/4]
//   num_indices: M
//   row_stride_vec4: D/4 (number of float4 elements per row)

kernel void gather_rows_vec4(
    device const float4* src [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device float4* dst [[buffer(2)]],
    constant uint& num_indices [[buffer(3)]],
    constant uint& row_stride_vec4 [[buffer(4)]],   // D/4
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= num_indices || gid.x >= row_stride_vec4) return;
    
    uint src_row = indices[gid.y];
    dst[gid.y * row_stride_vec4 + gid.x] = src[src_row * row_stride_vec4 + gid.x];
}

kernel void gather_rows_vec4_fp16(
    device const half4* src [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device half4* dst [[buffer(2)]],
    constant uint& num_indices [[buffer(3)]],
    constant uint& row_stride_vec4 [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= num_indices || gid.x >= row_stride_vec4) return;
    
    uint src_row = indices[gid.y];
    dst[gid.y * row_stride_vec4 + gid.x] = src[src_row * row_stride_vec4 + gid.x];
}

// ============================================================================
// scatter_add: Atomically accumulate values at indices
// ============================================================================
// Equivalent to: dst[indices[i]] += src[i]
// PyTorch: torch.scatter_add(dst, dim=0, index=indices, src=src)
//
// Uses atomic operations for thread-safe accumulation. Multiple threads may
// write to the same output location, so atomic_float is required.
//
// Note: This is the 1D version. For multi-dimensional tensors, flatten
// the outer dimensions and divide inner_size into the stride.
//
// Parameters:
//   src: Source values [N] - float array
//   indices: Destination indices [N] - uint array
//   dst: Output array (atomic) - values accumulated here
//   count: N - number of elements to scatter

kernel void scatter_add(
    device const float* src [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device atomic_float* dst [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    
    uint dst_idx = indices[tid];
    float value = src[tid];
    
    // Atomic addition for race-condition-free accumulation
    atomic_fetch_add_explicit(&dst[dst_idx], value, memory_order_relaxed);
}

// ============================================================================
// scatter_add_2d: 2D scatter add with inner dimension
// ============================================================================
// Equivalent to: dst[indices[i], j] += src[i, j]
// For each element in the batch dimension, scatter-add the entire inner row.
//
// Parameters:
//   src: Source matrix [N, inner_size]
//   indices: Row indices for destination [N]
//   dst: Output matrix [M, inner_size] (M = max index + 1)
//   count: N - number of rows to scatter
//   inner_size: Size of inner dimension

kernel void scatter_add_2d(
    device const float* src [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device atomic_float* dst [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& inner_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    // gid.x = position within inner dimension
    // gid.y = which source row
    if (gid.y >= count || gid.x >= inner_size) return;
    
    uint dst_row = indices[gid.y];
    float value = src[gid.y * inner_size + gid.x];
    
    // Atomic add to destination
    atomic_fetch_add_explicit(&dst[dst_row * inner_size + gid.x], value, memory_order_relaxed);
}

// ============================================================================
// index_select: General index selection along dimension 0
// ============================================================================
// Equivalent to: dst[i, ...] = src[indices[i], ...]
// PyTorch: torch.index_select(src, dim=0, index=indices)
//
// More flexible than gather_rows - handles arbitrary inner dimensions.
// The inner_size parameter accounts for all dimensions after dim 0.
//
// Example: src [N, H, W], indices [M], inner_size = H*W
//          dst [M, H, W]
//
// Parameters:
//   src: Source tensor [N, inner_size] - flattened after dim 0
//   indices: Selection indices [M]
//   dst: Output tensor [M, inner_size]
//   num_indices: M
//   inner_size: Product of all dimensions after dim 0

kernel void index_select(
    device const float* src [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant uint& num_indices [[buffer(3)]],
    constant uint& inner_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= num_indices || gid.x >= inner_size) return;
    
    uint src_idx = indices[gid.y];
    dst[gid.y * inner_size + gid.x] = src[src_idx * inner_size + gid.x];
}

kernel void index_select_fp16(
    device const half* src [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device half* dst [[buffer(2)]],
    constant uint& num_indices [[buffer(3)]],
    constant uint& inner_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= num_indices || gid.x >= inner_size) return;
    
    uint src_idx = indices[gid.y];
    dst[gid.y * inner_size + gid.x] = src[src_idx * inner_size + gid.x];
}

// ============================================================================
// index_select_vec4: Vectorized index selection
// ============================================================================
// Use when inner_size is divisible by 4.

kernel void index_select_vec4(
    device const float4* src [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device float4* dst [[buffer(2)]],
    constant uint& num_indices [[buffer(3)]],
    constant uint& inner_size_vec4 [[buffer(4)]],   // inner_size / 4
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= num_indices || gid.x >= inner_size_vec4) return;
    
    uint src_idx = indices[gid.y];
    dst[gid.y * inner_size_vec4 + gid.x] = src[src_idx * inner_size_vec4 + gid.x];
}

// ============================================================================
// gather_2d: 2D gather for attention patterns
// ============================================================================
// Equivalent to: dst[i] = src[row_indices[i], col_indices[i]]
//
// Gathers elements at specific (row, col) coordinate pairs.
// Useful for attention sparse patterns and non-contiguous access.
//
// Parameters:
//   src: Source matrix [N, M] in row-major order
//   row_indices: Row coordinates [K]
//   col_indices: Column coordinates [K]
//   dst: Output values [K]
//   count: K - number of elements to gather
//   src_cols: M - number of columns in source (row stride)

kernel void gather_2d(
    device const float* src [[buffer(0)]],          // [N, M] matrix
    device const uint* row_indices [[buffer(1)]],   // [K]
    device const uint* col_indices [[buffer(2)]],   // [K]
    device float* dst [[buffer(3)]],                // [K] output
    constant uint& count [[buffer(4)]],
    constant uint& src_cols [[buffer(5)]],          // M (stride for row)
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    
    uint row = row_indices[tid];
    uint col = col_indices[tid];
    
    // Row-major indexing: element at (row, col) is at row * src_cols + col
    dst[tid] = src[row * src_cols + col];
}

kernel void gather_2d_fp16(
    device const half* src [[buffer(0)]],
    device const uint* row_indices [[buffer(1)]],
    device const uint* col_indices [[buffer(2)]],
    device half* dst [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& src_cols [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    
    uint row = row_indices[tid];
    uint col = col_indices[tid];
    
    dst[tid] = src[row * src_cols + col];
}

// ============================================================================
// scatter_2d: 2D scatter for attention patterns
// ============================================================================
// Equivalent to: dst[row_indices[i], col_indices[i]] = src[i]
// (Non-atomic version - assumes no collisions)
//
// Parameters:
//   src: Source values [K]
//   row_indices: Row coordinates [K]
//   col_indices: Column coordinates [K]
//   dst: Destination matrix [N, M]
//   count: K
//   dst_cols: M (row stride for destination)

kernel void scatter_2d(
    device const float* src [[buffer(0)]],
    device const uint* row_indices [[buffer(1)]],
    device const uint* col_indices [[buffer(2)]],
    device float* dst [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& dst_cols [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    
    uint row = row_indices[tid];
    uint col = col_indices[tid];
    
    dst[row * dst_cols + col] = src[tid];
}

// ============================================================================
// gather_batch: Batched gather for 3D tensors
// ============================================================================
// Equivalent to: dst[b, i, :] = src[b, indices[b, i], :]
//
// Batch dimension is preserved, gather happens along second dimension.
// Common in batched attention and batched MoE routing.
//
// Parameters:
//   src: Source tensor [B, N, D]
//   indices: Indices [B, M]
//   dst: Output [B, M, D]
//   batch_size: B
//   num_indices: M (per batch)
//   row_stride: D (inner dimension)

kernel void gather_batch(
    device const float* src [[buffer(0)]],          // [B, N, D]
    device const uint* indices [[buffer(1)]],       // [B, M]
    device float* dst [[buffer(2)]],                // [B, M, D]
    constant uint& batch_size [[buffer(3)]],
    constant uint& num_indices [[buffer(4)]],       // M
    constant uint& src_rows [[buffer(5)]],          // N
    constant uint& row_stride [[buffer(6)]],        // D
    uint3 gid [[thread_position_in_grid]])          // (col, idx, batch)
{
    // gid.x = column within row (0 to row_stride-1)
    // gid.y = which index (0 to num_indices-1)
    // gid.z = batch index (0 to batch_size-1)
    
    if (gid.z >= batch_size || gid.y >= num_indices || gid.x >= row_stride) return;
    
    uint batch_offset = gid.z * src_rows * row_stride;
    uint indices_offset = gid.z * num_indices;
    uint dst_offset = gid.z * num_indices * row_stride;
    
    uint src_row = indices[indices_offset + gid.y];
    
    dst[dst_offset + gid.y * row_stride + gid.x] = 
        src[batch_offset + src_row * row_stride + gid.x];
}

// ============================================================================
// Utility kernels
// ============================================================================

// Clear atomic float buffer (prepare for scatter_add)
// Must be called before scatter_add to initialize accumulation targets to 0
kernel void clear_atomic_float(
    device atomic_float* buffer [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    atomic_store_explicit(&buffer[tid], 0.0f, memory_order_relaxed);
}

// Clear float buffer (non-atomic version)
kernel void clear_float(
    device float* buffer [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    buffer[tid] = 0.0f;
}

// Copy with index validation (safety check for invalid indices)
kernel void gather_rows_safe(
    device const float* src [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant uint& num_indices [[buffer(3)]],
    constant uint& row_stride [[buffer(4)]],
    constant uint& num_src_rows [[buffer(5)]],      // Bounds for validation
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= num_indices || gid.x >= row_stride) return;
    
    uint src_row = indices[gid.y];
    
    // Bounds check - clamp to valid range or skip
    if (src_row >= num_src_rows) return;
    
    dst[gid.y * row_stride + gid.x] = src[src_row * row_stride + gid.x];
}
