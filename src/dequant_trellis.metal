// dequant_trellis.metal - EXL3 Trellis Dequantization for Apple Metal
// ============================================================================
//
// Metal kernel for dequantizing EXL3 trellis-quantized weights back to FP16.
//
// Trellis quantization uses Viterbi algorithm to find optimal quantization
// paths through a codebook grid. Each 16x16 tile (256 elements) is quantized
// together with shared scale factors.
//
// Data layout:
//   - Indices: [tiles_k, tiles_n, 256] int16 - grid indices for each element
//   - Scales:  [n_groups, N] float32 - per-group scale factors
//   - Grid:    [n_levels] float32 - codebook quantization centers
//   - Output:  [K, N] half - dequantized FP16 weights
//
// Reference: trellis_codebook.py for grid generation, viterbi_quant.metal
// for the tile structure used during quantization.
//
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants
// ============================================================================

constant constexpr uint TILE_DIM = 16;        // 16x16 tile dimension
constant constexpr uint TILE_SIZE = 256;      // Total elements per tile

// ============================================================================
// Core Dequantization Primitives
// ============================================================================

/// Dequantize a single element using grid lookup and scale.
///
/// The trellis index maps directly to a grid value, which is then
/// scaled to recover the original weight magnitude.
///
/// @param idx     Trellis index into the codebook grid [0, n_levels-1]
/// @param scale   Per-group scale factor
/// @param grid    Codebook grid values [n_levels]
/// @return        Dequantized FP16 value
inline half dequant_trellis_element(short idx, float scale, device const float* grid) {
    // Grid lookup: index -> normalized value
    float normalized = grid[idx];
    // Apply scale to denormalize
    return half(normalized * scale);
}

/// Dequantize a tile of 256 elements using vectorized operations.
///
/// Processes 8 elements at a time using half4 vectors for efficient
// memory access and ALU utilization on Apple Silicon.
///
/// @param indices     Tile indices [256] - each element is a grid index
/// @param scale       Per-tile scale factor
/// @param grid        Codebook grid values [n_levels]
/// @param out         Output buffer for 256 dequantized half values
/// @param lane_id     Thread index within threadgroup [0, 255]
inline void dequant_trellis_tile(
    device const short* indices,
    float scale,
    device const float* grid,
    threadgroup half* out,
    uint lane_id
) {
    // Each thread processes one element
    if (lane_id < TILE_SIZE) {
        short idx = indices[lane_id];
        out[lane_id] = dequant_trellis_element(idx, scale, grid);
    }
}

// ============================================================================
// Main Dequantization Kernel
// ============================================================================

/// Dequantize trellis-quantized weights to FP16.
///
/// Input Layout:
///   - indices: [tiles_k, tiles_n, 256] int16
///     Each tile contains 256 grid indices for a 16x16 block of weights.
///   - scales: [n_groups, N] float32
///     Per-group scale factors. Groups are contiguous along K dimension.
///   - grid: [n_levels] float32
///     Codebook quantization centers from trellis_codebook.py.
///
/// Output Layout:
///   - weights: [K, N] half in row-major order (K rows, N columns)
///
/// Thread Mapping:
///   - Each thread processes one output element
///   - gid.x = column index (0 to N-1)
///   - gid.y = row index (0 to K-1)
///
/// @param indices     Trellis indices [tiles_k, tiles_n, 256]
/// @param scales      Per-group scales [n_groups, N]
/// @param grid        Codebook grid values [n_levels]
/// @param output      Dequantized output [K, N]
/// @param K           Number of rows (input features)
/// @param N           Number of columns (output features)
/// @param n_levels    Number of quantization levels (2^bits)
/// @param group_size  Number of elements per quantization group
kernel void dequant_trellis(
    device const short* indices [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* grid [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& n_levels [[buffer(6)]],
    constant uint& group_size [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Grid position: gid.x = column (N), gid.y = row (K)
    uint n_idx = gid.x;
    uint k_idx = gid.y;
    
    // Bounds check
    if (n_idx >= N || k_idx >= K) return;
    
    // Compute tile coordinates
    // Tiles are 16x16 blocks covering the K x N matrix
    uint tile_k = k_idx / TILE_DIM;   // Which tile row
    uint tile_n = n_idx / TILE_DIM;   // Which tile column
    uint local_k = k_idx % TILE_DIM;  // Position within tile (0-15)
    uint local_n = n_idx % TILE_DIM;  // Position within tile (0-15)
    
    // Compute tiles_n (number of tiles along N dimension)
    uint tiles_n = (N + TILE_DIM - 1) / TILE_DIM;
    
    // Index into the indices buffer: [tiles_k, tiles_n, 256]
    // Linear index within tile: local_k * 16 + local_n (row-major within tile)
    uint tile_offset = tile_k * tiles_n + tile_n;
    uint local_offset = local_k * TILE_DIM + local_n;
    uint idx = tile_offset * TILE_SIZE + local_offset;
    
    // Load trellis index and bounds check against n_levels
    short trellis_idx = indices[idx];
    
    // Clamp index to valid range (safety check)
    if (trellis_idx < 0 || uint(trellis_idx) >= n_levels) {
        trellis_idx = 0;
    }
    
    // Compute quantization group index
    // Groups are contiguous along K, spanning all N columns
    uint group_idx = k_idx / group_size;
    
    // Load scale for this group and column
    // scales layout: [n_groups, N] row-major
    uint n_groups = (K + group_size - 1) / group_size;
    float scale = scales[group_idx * N + n_idx];
    
    // Dequantize: grid[idx] * scale
    half dequantized = dequant_trellis_element(trellis_idx, scale, grid);
    
    // Write output in row-major order: [K, N]
    output[k_idx * N + n_idx] = dequantized;
}

// ============================================================================
// Optimized Variant: Tile-based Processing
// ============================================================================

/// Tile-optimized trellis dequantization kernel.
///
/// This variant processes entire 16x16 tiles using threadgroups,
/// enabling better memory coalescing and shared memory utilization.
///
/// Threadgroup Layout:
///   - 256 threads per threadgroup (one per tile element)
///   - Each threadgroup processes one tile
///
/// @param indices     Trellis indices [tiles_k, tiles_n, 256]
/// @param scales      Per-group scales [n_groups, N]
/// @param grid        Codebook grid values [n_levels]
/// @param output      Dequantized output [K, N]
/// @param K           Number of rows
/// @param N           Number of columns
/// @param n_levels    Number of quantization levels
/// @param group_size  Elements per quantization group
/// @param tile_id     Threadgroup index (which tile to process)
/// @param lane_id     Thread index within threadgroup [0, 255]
kernel void dequant_trellis_tiled(
    device const short* indices [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* grid [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& n_levels [[buffer(6)]],
    constant uint& group_size [[buffer(7)]],
    uint tile_id [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_threadgroup]]
) {
    // Compute tile coordinates from tile_id
    uint tiles_n = (N + TILE_DIM - 1) / TILE_DIM;
    uint tile_k = tile_id / tiles_n;
    uint tile_n = tile_id % tiles_n;
    
    // Compute global K, N indices for this thread
    uint local_k = lane_id / TILE_DIM;  // 0-15
    uint local_n = lane_id % TILE_DIM;  // 0-15
    uint k_idx = tile_k * TILE_DIM + local_k;
    uint n_idx = tile_n * TILE_DIM + local_n;
    
    // Bounds check
    if (k_idx >= K || n_idx >= N) return;
    
    // Load trellis index
    uint tile_offset = tile_id;
    uint local_offset = lane_id;
    uint idx = tile_offset * TILE_SIZE + local_offset;
    
    short trellis_idx = indices[idx];
    
    // Clamp to valid range
    if (trellis_idx < 0 || uint(trellis_idx) >= n_levels) {
        trellis_idx = 0;
    }
    
    // Load scale
    uint group_idx = k_idx / group_size;
    float scale = scales[group_idx * N + n_idx];
    
    // Dequantize and write output
    output[k_idx * N + n_idx] = dequant_trellis_element(trellis_idx, scale, grid);
}

// ============================================================================
// Bulk Dequantization Helper
// ============================================================================

/// Bulk trellis dequant with explicit tile dimensions.
///
/// Used when tile dimensions are known at compile time or for
/// debugging/validation purposes.
///
/// @param indices     Trellis indices [n_tiles, 256]
/// @param scales      Per-tile scales [n_tiles]
/// @param grid        Codebook grid values [n_levels]
/// @param output      Dequantized output [n_tiles, 256]
/// @param n_tiles     Number of tiles to process
/// @param n_levels    Number of quantization levels
kernel void dequant_trellis_bulk(
    device const short* indices [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* grid [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& n_tiles [[buffer(4)]],
    constant uint& n_levels [[buffer(5)]],
    uint tile_id [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_threadgroup]]
) {
    if (tile_id >= n_tiles || lane_id >= TILE_SIZE) return;
    
    // Load trellis index
    uint idx = tile_id * TILE_SIZE + lane_id;
    short trellis_idx = indices[idx];
    
    // Clamp to valid range
    if (trellis_idx < 0 || uint(trellis_idx) >= n_levels) {
        trellis_idx = 0;
    }
    
    // Load per-tile scale
    float scale = scales[tile_id];
    
    // Dequantize and write
    output[idx] = dequant_trellis_element(trellis_idx, scale, grid);
}

// ============================================================================
// Vectorized Dequantization (half4 stores)
// ============================================================================

/// High-throughput variant using half4 vectorized stores.
///
/// Requires that N is a multiple of 4 for aligned half4 access.
/// Each thread processes 4 consecutive elements along N.
///
/// @param indices     Trellis indices [tiles_k, tiles_n, 256]
/// @param scales      Per-group scales [n_groups, N]
/// @param grid        Codebook grid values [n_levels]
/// @param output      Dequantized output [K, N] as half4 array
/// @param K           Number of rows
/// @param N           Number of columns (must be multiple of 4)
/// @param n_levels    Number of quantization levels
/// @param group_size  Elements per quantization group
kernel void dequant_trellis_aligned(
    device const short* indices [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* grid [[buffer(2)]],
    device half4* output [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& n_levels [[buffer(6)]],
    constant uint& group_size [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Each thread processes 4 consecutive N elements
    uint n_base = gid.x * 4;
    uint k_idx = gid.y;
    
    if (n_base + 3 >= N || k_idx >= K) return;
    
    // Compute tile coordinates (same tile for all 4 elements)
    uint tile_k = k_idx / TILE_DIM;
    uint tile_n_base = n_base / TILE_DIM;
    uint local_k = k_idx % TILE_DIM;
    
    uint tiles_n = (N + TILE_DIM - 1) / TILE_DIM;
    uint group_idx = k_idx / group_size;
    
    half4 result;
    
    // Process 4 consecutive elements
    for (uint i = 0; i < 4; i++) {
        uint n_idx = n_base + i;
        uint tile_n = n_idx / TILE_DIM;
        uint local_n = n_idx % TILE_DIM;
        
        // Load trellis index
        uint tile_offset = tile_k * tiles_n + tile_n;
        uint local_offset = local_k * TILE_DIM + local_n;
        uint idx = tile_offset * TILE_SIZE + local_offset;
        
        short trellis_idx = indices[idx];
        if (trellis_idx < 0 || uint(trellis_idx) >= n_levels) {
            trellis_idx = 0;
        }
        
        // Load scale and dequantize
        float scale = scales[group_idx * N + n_idx];
        result[i] = dequant_trellis_element(trellis_idx, scale, grid);
    }
    
    // Vectorized store
    output[k_idx * (N / 4) + gid.x] = result;
}

// ============================================================================
// Inline Helper for Fused GEMM Kernels
// ============================================================================

/// Inline helper for trellis dequant inside GEMM tile loops.
///
/// This function can be called inline within a GEMM kernel to
/// dequantize trellis-quantized weights on-the-fly without a
/// separate dequantization pass.
///
/// @param tile_indices  256 trellis indices for this tile
/// @param scale         Per-tile scale factor
/// @param grid          Codebook grid values
/// @param n_levels      Number of quantization levels
/// @param lane_id       Thread index [0, 255]
/// @return              Dequantized FP16 value for this lane
inline half dequant_trellis_fused(
    threadgroup const short* tile_indices,
    float scale,
    threadgroup const float* grid,
    uint n_levels,
    uint lane_id
) {
    short idx = tile_indices[lane_id];
    if (idx < 0 || uint(idx) >= n_levels) {
        idx = 0;
    }
    return half(grid[idx] * scale);
}

// ============================================================================
// Sign Flip Kernels for Hadamard Inverse
// ============================================================================

/// Apply sign flips from su (row signs) and sv (column signs).
///
/// The trellis quantization uses Hadamard rotation with sign absorbing:
///   W_rotated = diag(su) @ Had @ W_orig @ Had.T @ diag(sv)
///
/// To recover the original weight space (or get a usable approximation):
///   W_dequant = diag(su) @ W_trellis_decoded @ diag(sv)
///
/// This kernel applies the sign flip: out[k,n] = su[k] * w[k,n] * sv[n]
/// where su and sv contain ±1 values.
///
/// @param weights  Dequantized weights [K, N] half, modified in-place
/// @param su       Row sign vector [K] float32 (±1 values)
/// @param sv       Column sign vector [N] float32 (±1 values)
/// @param K        Number of rows
/// @param N        Number of columns
kernel void apply_sign_flips(
    device half* weights [[buffer(0)]],
    device const float* su [[buffer(1)]],
    device const float* sv [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint n_idx = gid.x;
    uint k_idx = gid.y;
    
    if (n_idx >= N || k_idx >= K) return;
    
    // Load signs (±1 values stored as float32)
    float sign_k = su[k_idx];
    float sign_n = sv[n_idx];
    float combined_sign = sign_k * sign_n;
    
    // Apply combined sign flip
    uint idx = k_idx * N + n_idx;
    weights[idx] = half(float(weights[idx]) * combined_sign);
}

/// Vectorized sign flip kernel using half4 for better throughput.
///
/// Requires N to be a multiple of 4.
///
/// @param weights  Dequantized weights [K, N] as half4 array
/// @param su       Row sign vector [K] float32
/// @param sv       Column sign vector [N] float32  
/// @param K        Number of rows
/// @param N        Number of columns (must be multiple of 4)
kernel void apply_sign_flips_vec4(
    device half4* weights [[buffer(0)]],
    device const float* su [[buffer(1)]],
    device const float* sv [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint n_base = gid.x * 4;
    uint k_idx = gid.y;
    
    if (n_base + 3 >= N || k_idx >= K) return;
    
    float sign_k = su[k_idx];
    
    // Load 4 column signs
    float4 sign_n = float4(sv[n_base], sv[n_base + 1], sv[n_base + 2], sv[n_base + 3]);
    float4 combined = sign_k * sign_n;
    
    // Load, apply, store
    uint idx = k_idx * (N / 4) + gid.x;
    half4 w = weights[idx];
    weights[idx] = half4(float4(w) * combined);
}

// ============================================================================
// Combined Dequant + Sign Flip Kernel
// ============================================================================

/// Fused trellis dequantization with sign flip application.
///
/// Combines dequant_trellis and apply_sign_flips into one kernel pass
/// for better memory efficiency.
///
/// @param indices     Trellis indices [tiles_k, tiles_n, 256] int16
/// @param scales      Per-group scales [n_groups, N] float32
/// @param grid        Codebook grid [n_levels] float32
/// @param su          Row sign vector [K] float32
/// @param sv          Column sign vector [N] float32
/// @param output      Dequantized output [K, N] half
/// @param K           Number of rows
/// @param N           Number of columns
/// @param n_levels    Number of quantization levels
/// @param group_size  Elements per quantization group
kernel void dequant_trellis_with_signs(
    device const short* indices [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* grid [[buffer(2)]],
    device const float* su [[buffer(3)]],
    device const float* sv [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& n_levels [[buffer(8)]],
    constant uint& group_size [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint n_idx = gid.x;
    uint k_idx = gid.y;
    
    if (n_idx >= N || k_idx >= K) return;
    
    // Compute tile coordinates
    uint tile_k = k_idx / TILE_DIM;
    uint tile_n = n_idx / TILE_DIM;
    uint local_k = k_idx % TILE_DIM;
    uint local_n = n_idx % TILE_DIM;
    
    uint tiles_n = (N + TILE_DIM - 1) / TILE_DIM;
    
    // Load trellis index
    uint tile_offset = tile_k * tiles_n + tile_n;
    uint local_offset = local_k * TILE_DIM + local_n;
    uint idx = tile_offset * TILE_SIZE + local_offset;
    
    short trellis_idx = indices[idx];
    if (trellis_idx < 0 || uint(trellis_idx) >= n_levels) {
        trellis_idx = 0;
    }
    
    // Load scale
    uint group_idx = k_idx / group_size;
    float scale = scales[group_idx * N + n_idx];
    
    // Dequantize
    float dequant_val = grid[trellis_idx] * scale;
    
    // Apply sign flips
    float sign_k = su[k_idx];
    float sign_n = sv[n_idx];
    dequant_val *= sign_k * sign_n;
    
    // Write output
    output[k_idx * N + n_idx] = half(dequant_val);
}

// ============================================================================
// Packed Index Dequantization Kernel
// ============================================================================

/// Unpack a single trellis index from packed byte array.
///
/// Supports 2-bit, 3-bit, and 4-bit packing schemes.
/// For 3-bit: 256 indices * 3 bits = 768 bits = 96 bytes/tile
/// For 4-bit: 256 indices * 4 bits = 1024 bits = 128 bytes/tile
/// For 2-bit: 256 indices * 2 bits = 512 bits = 64 bytes/tile
///
/// @param packed      Packed indices for this tile
/// @param idx_in_tile Index within tile [0, 255]
/// @param bits        Bit width (2, 3, or 4)
/// @return            Unpacked codebook index
inline uint unpack_index(device const uchar* packed, uint idx_in_tile, uint bits) {
    uint bit_offset = idx_in_tile * bits;
    uint byte_idx = bit_offset / 8;
    uint bit_in_byte = bit_offset % 8;
    uint mask = (1u << bits) - 1;
    
    // Read 1-2 bytes depending on whether we span a boundary
    uint packed_val = packed[byte_idx];
    if (bit_in_byte + bits > 8) {
        packed_val |= (uint(packed[byte_idx + 1]) << 8);
    }
    
    return (packed_val >> bit_in_byte) & mask;
}

/// Dequantize trellis weights from packed uint8 indices.
///
/// This kernel operates directly on the packed byte format, unpacking
/// indices on-the-fly to avoid the 5x memory inflation of pre-unpacking.
///
/// Input Layout:
///   - packed_indices: [tiles_k, tiles_n, packed_bytes] uint8
///     packed_bytes = ceil(256 * bits / 8)
///   - scales: [n_groups, N] float32
///   - grid: [n_levels] float32 codebook
///   - su: [K] float32 row signs
///   - sv: [N] float32 column signs
///
/// @param packed_indices  Packed byte array [tiles_k, tiles_n, packed_bytes]
/// @param scales          Per-group scales [n_groups, N]
/// @param grid            Codebook grid [n_levels]
/// @param su              Row sign vector [K]
/// @param sv              Column sign vector [N]
/// @param output          Dequantized output [K, N]
/// @param K               Number of rows
/// @param N               Number of columns
/// @param n_levels        Number of quantization levels (2^bits)
/// @param bits            Quantization bit width
kernel void dequant_trellis_packed(
    device const uchar* packed_indices [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* grid [[buffer(2)]],
    device const float* su [[buffer(3)]],
    device const float* sv [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& n_levels [[buffer(8)]],
    constant uint& bits [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint n_idx = gid.x;
    uint k_idx = gid.y;
    
    if (n_idx >= N || k_idx >= K) return;
    
    // Compute tile coordinates
    uint tile_k = k_idx / TILE_DIM;
    uint tile_n = n_idx / TILE_DIM;
    uint local_k = k_idx % TILE_DIM;
    uint local_n = n_idx % TILE_DIM;
    
    uint tiles_n = (N + TILE_DIM - 1) / TILE_DIM;
    uint idx_in_tile = local_k * TILE_DIM + local_n;
    
    // Packed bytes per tile: ceil(256 * bits / 8)
    uint packed_bytes_per_tile = (TILE_SIZE * bits + 7) / 8;
    
    // Offset into packed_indices
    uint tile_offset = (tile_k * tiles_n + tile_n) * packed_bytes_per_tile;
    
    // Unpack the trellis index
    uint trellis_idx = unpack_index(packed_indices + tile_offset, idx_in_tile, bits);
    
    // Clamp to valid range
    if (trellis_idx >= n_levels) {
        trellis_idx = 0;
    }
    
    // Load scale (group size = 128)
    uint group_size = 128;
    uint group_idx = k_idx / group_size;
    float scale = scales[group_idx * N + n_idx];
    
    // Dequantize: grid[idx] * scale * su * sv
    float dequant_val = grid[trellis_idx] * scale;
    dequant_val *= su[k_idx] * sv[n_idx];
    
    output[k_idx * N + n_idx] = half(dequant_val);
}
