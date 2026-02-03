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
// Texture Optimization:
//   The codebook grid is small (4/8/16 values for 2/3/4 bit) but accessed
//   repeatedly. Using texture1d<float> provides:
//   - Hardware texture cache optimized for repeated small lookups
//   - Better cache line utilization than device buffer for tiny arrays
//   - Automatic bounds checking with clamp-to-edge addressing
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

/// Unpack a single trellis index from packed byte array.
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

// ============================================================================
// Threadgroup-Cached Codebook Primitives
// ============================================================================

/// Maximum codebook size (16 levels for 4-bit quantization)
constant constexpr uint MAX_GRID_LEVELS = 16;

/// Load codebook grid into threadgroup memory cooperatively.
inline void load_grid_to_threadgroup(
    constant float* grid_device,
    threadgroup float* grid_tg,
    uint n_levels,
    uint thread_idx
) {
    if (thread_idx < n_levels) {
        grid_tg[thread_idx] = grid_device[thread_idx];
    }
}

// ============================================================================
// Threadgroup-Cached Scales Primitives
// ============================================================================

/// Load tile scales into threadgroup memory cooperatively.
/// Template supports both device and constant address space pointers.
template <typename T>
inline void load_scales_to_threadgroup(
    T scales_device,
    threadgroup float* scales_tg,
    uint N,
    uint n_tile_base,
    uint k_tile_base,
    uint group_size,
    uint K,
    uint lane_id,
    uint stride
) {
    // Compute groups spanned by this tile along K dimension
    uint group_start = k_tile_base / group_size;
    uint k_tile_end = min(k_tile_base + TILE_DIM, K);
    uint group_end = (k_tile_end + group_size - 1) / group_size;
    uint groups_in_tile = group_end - group_start;

    // Total scale values to load: groups_in_tile * 16 columns
    uint total_scales = groups_in_tile * TILE_DIM;

    // Cooperatively load: each thread loads one or more values
    #pragma unroll(4)
    for (uint i = lane_id; i < total_scales; i += stride) {
        uint group_rel = i / TILE_DIM;          // Group offset within tile (0 to groups_in_tile-1)
        uint col_in_tile = i % TILE_DIM;        // Column within tile (0 to 15)
        uint group_abs = group_start + group_rel;  // Absolute group index
        uint n_idx = n_tile_base + col_in_tile;    // Absolute column index

        // Load from device: scales[group * N + n]
        if (n_idx < N) {
            scales_tg[i] = scales_device[group_abs * N + n_idx];
        }
    }
}

// ============================================================================
// Texture Sampler Configuration
// ============================================================================

/// Constexpr sampler for codebook texture lookup.
/// Uses nearest filtering (no interpolation needed) and clamp-to-edge addressing.
constexpr sampler grid_sampler(coord::pixel,          // Use pixel coordinates (0-N)
                               address::clamp_to_edge, // Clamp OOB accesses
                               filter::nearest);       // No interpolation

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
inline half dequant_trellis_element(short idx, float scale, constant const float* grid) {
    // Grid lookup: index -> normalized value
    float normalized = grid[idx];
    // Apply scale to denormalize
    return half(normalized * scale);
}

/// Dequantize a single element using texture-based grid lookup.
///
/// Uses hardware texture sampling for better cache utilization.
/// The texture cache is optimized for repeated small lookups which
/// matches the access pattern of codebook lookups.
///
/// @param idx          Trellis index into the codebook grid [0, n_levels-1]
/// @param scale        Per-group scale factor
/// @param grid_tex     Codebook grid as 1D texture
/// @return             Dequantized FP16 value
inline half dequant_trellis_element_tex(uint idx, float scale, texture1d<float, access::sample> grid_tex) {
    // Texture sample returns float4, we only need .x component
    // Using pixel coordinates (0, 1, 2, ...) not normalized (0.0-1.0)
    float normalized = grid_tex.sample(grid_sampler, float(idx)).x;
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
    constant const float* grid,
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
    constant float* scales [[buffer(1)]],
    constant float* grid [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& n_levels [[buffer(6)]],
    constant uint& group_size [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_threadgroup]]
) {
    // Grid position: gid.x = column (N), gid.y = row (K)
    uint n_idx = gid.x;
    uint k_idx = gid.y;
    
    // Threadgroup memory for scales cache
    threadgroup float scales_cache[256];

    // Compute tile base coordinates
    uint tile_k = tgid.y;
    uint tile_n = tgid.x;
    uint k_tile_base = tile_k * TILE_DIM;
    uint n_tile_base = tile_n * TILE_DIM;

    // Cooperative scales load (256 threads)
    load_scales_to_threadgroup(scales, scales_cache, N, n_tile_base, k_tile_base, group_size, K, lane_id, 256);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Bounds check
    if (n_idx >= N || k_idx >= K) return;
    
    // Compute tile coordinates
    // Tiles are 16x16 blocks covering the K x N matrix
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
    uint group_idx = k_idx / group_size;
    
    // Load scale from threadgroup cache
    uint group_start = k_tile_base / group_size;
    uint group_rel = group_idx - group_start;
    float scale = scales_cache[group_rel * TILE_DIM + local_n];
    
    // Dequantize: grid[idx] * scale
    half dequantized = dequant_trellis_element(trellis_idx, scale, grid);
    
    // Write output in row-major order: [K, N]
    output[k_idx * N + n_idx] = dequantized;
}

// ============================================================================
// Threadgroup-Cached Scales Primitives
// ============================================================================
//
// OPTIMIZATION: Scale factors are accessed repeatedly by all threads in a
// tile (16x16 = 256 threads). Caching scales in threadgroup memory:
//   - Reduces global memory bandwidth by up to 256x
//   - Enables coalesced loads at tile boundaries
//   - Scale memory cost: 16 * n_groups_in_tile floats (typically 16-64 floats)
//
// Scales layout: [n_groups, N] where:
//   - n_groups = K / group_size
//   - N = number of columns (one scale per column per group)
//
// For a 16x16 tile, we need scales for 16 consecutive columns across all
// groups spanned by the tile's K rows.
// ============================================================================

// Moved to top of file

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
    constant float* scales [[buffer(1)]],
    constant float* grid [[buffer(2)]],
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

    // Global K, N base for this tile
    uint k_tile_base = tile_k * TILE_DIM;
    uint n_tile_base = tile_n * TILE_DIM;

    // Threadgroup memory for scales cache
    threadgroup float scales_cache[256];

    // Cooperative scales load
    load_scales_to_threadgroup(scales, scales_cache, N, n_tile_base, k_tile_base, group_size, K, lane_id, 256);
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

    // Load scale from threadgroup cache
    uint group_idx = k_idx / group_size;
    uint group_start = k_tile_base / group_size;
    uint group_rel = group_idx - group_start;
    float scale = scales_cache[group_rel * TILE_DIM + local_n];

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
    constant float* scales [[buffer(1)]],
    constant float* grid [[buffer(2)]],
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
    constant float* scales [[buffer(1)]],
    constant float* grid [[buffer(2)]],
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
    uint local_k = k_idx % TILE_DIM;
    
    uint tiles_n = (N + TILE_DIM - 1) / TILE_DIM;
    uint group_idx = k_idx / group_size;
    
    half4 result;
    
    // Process 4 consecutive elements
    #pragma unroll(4)
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

/// Fully fused dequantization into GEMM inner loop.
/// Dequantizes values on-the-fly and keeps them in registers only.
///
/// @param packed_indices  Packed byte array (pointer to tile start)
/// @param idx_in_tile     Index within the 16x16 tile
/// @param bits            Bit width
/// @param combined_scale  Precomputed scale * su * sv
/// @param grid            Codebook grid values
/// @param n_levels        Number of quantization levels
inline half dequant_trellis_fused_reg(
    device const uchar* packed_indices,
    uint idx_in_tile,
    uint bits,
    float combined_scale,
    constant float* grid,
    uint n_levels
) {
    uint trellis_idx = unpack_index(packed_indices, idx_in_tile, bits);
    if (trellis_idx >= n_levels) trellis_idx = 0;
    return half(grid[trellis_idx] * combined_scale);
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
    constant float* su [[buffer(1)]],
    constant float* sv [[buffer(2)]],
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
    constant float* su [[buffer(1)]],
    constant float* sv [[buffer(2)]],
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
    constant float* scales [[buffer(1)]],
    constant float* grid [[buffer(2)]],
    constant float* su [[buffer(3)]],
    constant float* sv [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& n_levels [[buffer(8)]],
    constant uint& group_size [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_threadgroup]]
) {
    // Threadgroup memory for scales cache
    threadgroup float scales_cache[256];

    // Compute tile base coordinates
    uint tile_k = tgid.y;
    uint tile_n = tgid.x;
    uint k_tile_base = tile_k * TILE_DIM;
    uint n_tile_base = tile_n * TILE_DIM;

    // Cooperative scales load (256 threads)
    load_scales_to_threadgroup(scales, scales_cache, N, n_tile_base, k_tile_base, group_size, K, lane_id, 256);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_idx = gid.x;
    uint k_idx = gid.y;
    
    if (n_idx >= N || k_idx >= K) return;
    
    // Compute tile coordinates
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
    
    // Load scale from threadgroup cache
    uint group_idx = k_idx / group_size;
    uint group_start = k_tile_base / group_size;
    uint group_rel = group_idx - group_start;
    float scale = scales_cache[group_rel * TILE_DIM + local_n];

    // Fused dequantize: grid[idx] * (scale * su * sv)
    // Single combined multiply reduces intermediate fp32 temporaries
    float combined_scale = scale * su[k_idx] * sv[n_idx];
    output[k_idx * N + n_idx] = half(grid[trellis_idx] * combined_scale);
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
// ============================================================================
// Vectorized Index Unpacking (SIMD)
// ============================================================================
//
// These functions unpack 4 indices at once using SIMD vector operations.
// Load a uint32 containing packed indices and extract using bit shifts.
// Provides ~4x throughput improvement for index extraction.
//
// ============================================================================

/// Unpack 4 consecutive 2-bit indices from packed bytes using SIMD.
/// Loads a uint32 containing 4 indices and extracts all 4 at once.
/// @param packed   Pointer to packed byte data
/// @param base_idx Starting index in tile (must be aligned to 4)
/// @return uint4 containing 4 unpacked indices (values 0-3)
inline uint4 unpack_2bit_x4(device const uchar* packed, uint base_idx) {
    // 4 indices per byte, so base_idx/4 gives the byte index
    uint byte_idx = base_idx >> 2;
    // Load 32 bits to get the byte we need (Metal supports unaligned loads)
    uint packed_val = *((device const uint*)(packed + byte_idx));
    
    return uint4(
        (packed_val >> 0) & 0x3,
        (packed_val >> 2) & 0x3,
        (packed_val >> 4) & 0x3,
        (packed_val >> 6) & 0x3
    );
}

/// Unpack 4 consecutive 3-bit indices from packed bytes.
/// Loads a uint32 to cover 12 bits (4 × 3-bit indices) and extracts.
/// @param packed   Pointer to packed byte data
/// @param base_idx Starting index in tile (must be aligned to 4)
/// @return uint4 containing 4 unpacked indices (values 0-7)
inline uint4 unpack_3bit_x4(device const uchar* packed, uint base_idx) {
    uint bit_offset = base_idx * 3;
    uint byte_idx = bit_offset >> 3;
    uint bit_in_byte = bit_offset & 7;

    // Load 32 bits (4 bytes) at once
    uint packed_val = *((device const uint*)(packed + byte_idx));
    
    // Shift to align first bit to 0
    uint val_shifted = packed_val >> bit_in_byte;

    return uint4(
        (val_shifted >> 0) & 0x7,
        (val_shifted >> 3) & 0x7,
        (val_shifted >> 6) & 0x7,
        (val_shifted >> 9) & 0x7
    );
}

/// Unpack 4 consecutive 4-bit indices from packed bytes using SIMD.
/// Loads a uint32 (containing 4 indices) and extracts.
/// @param packed   Pointer to packed byte data
/// @param base_idx Starting index in tile (must be aligned to 4)
/// @return uint4 containing 4 unpacked indices (values 0-15)
inline uint4 unpack_4bit_x4(device const uchar* packed, uint base_idx) {
    // 2 indices per byte, so we need 2 bytes for 4 indices
    uint byte_idx = base_idx >> 1;
    
    // Load 32 bits
    uint packed_val = *((device const uint*)(packed + byte_idx));
    
    return uint4(
        (packed_val >> 0) & 0xF,
        (packed_val >> 4) & 0xF,
        (packed_val >> 8) & 0xF,
        (packed_val >> 12) & 0xF
    );
}

/// Generic vectorized unpack dispatcher for 4 consecutive indices.
/// Routes to the appropriate vectorized unpack function based on bit width.
/// @param packed   Pointer to packed byte data
/// @param base_idx Starting index in tile (must be aligned to 4)
/// @param bits     Bit width (2, 3, or 4)
/// @return uint4 containing 4 unpacked indices
inline uint4 unpack_index_x4(device const uchar* packed, uint base_idx, uint bits) {
    switch (bits) {
        case 2: return unpack_2bit_x4(packed, base_idx);
        case 3: return unpack_3bit_x4(packed, base_idx);
        case 4: return unpack_4bit_x4(packed, base_idx);
        default: return unpack_3bit_x4(packed, base_idx);
    }
}

/// Dequantize 4 consecutive elements using vectorized unpacking.
/// @param packed         Pointer to packed byte data for this tile
/// @param base_idx       Starting index in tile (must be aligned to 4)
/// @param scale          Scale factor for dequantization
/// @param sign           Combined sign factor (su * sv)
/// @param grid           Codebook grid values
/// @param bits           Bit width (2, 3, or 4)
/// @param n_levels       Number of quantization levels
/// @return half4 containing 4 dequantized values
inline half4 dequant_trellis_x4(
    device const uchar* packed,
    uint base_idx,
    float scale,
    float sign,
    constant float* grid,
    uint bits,
    uint n_levels
) {
    // Unpack 4 indices at once using SIMD
    uint4 indices = unpack_index_x4(packed, base_idx, bits);

    // Clamp to valid range
    indices = min(indices, uint4(n_levels - 1));

    // Combined scale factor
    float combined = scale * sign;

    // Dequantize all 4 values
    return half4(
        half(grid[indices.x] * combined),
        half(grid[indices.y] * combined),
        half(grid[indices.z] * combined),
        half(grid[indices.w] * combined)
    );
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
    constant float* scales [[buffer(1)]],
    constant float* grid [[buffer(2)]],
    constant float* su [[buffer(3)]],
    constant float* sv [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& n_levels [[buffer(8)]],
    constant uint& bits [[buffer(9)]],
    constant uint& group_size [[buffer(10)]],
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
    uint idx_in_tile = local_n * TILE_DIM + local_k;  // Transposed weight
    
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
    
    // Load scale
    uint group_idx = k_idx / group_size;
    float scale = scales[group_idx * N + n_idx];

    // Fused dequantize: grid[idx] * (scale * su * sv)
    // Single combined multiply reduces intermediate fp32 temporaries
    float combined_scale = scale * su[k_idx] * sv[n_idx];
    output[k_idx * N + n_idx] = half(grid[trellis_idx] * combined_scale);
}

// ============================================================================
// Texture-Optimized Packed Index Dequantization Kernel
// ============================================================================

/// Dequantize trellis weights using texture-based codebook lookup.
///
/// Uses Metal texture sampling for the codebook grid lookup, providing:
/// - Hardware texture cache optimized for repeated small lookups
/// - Automatic bounds checking with clamp-to-edge addressing
/// - Potentially better cache line utilization for tiny grids (4-16 values)
///
/// Input Layout:
///   - packed_indices: [tiles_k, tiles_n, packed_bytes] uint8
///     packed_bytes = ceil(256 * bits / 8)
///   - scales: [n_groups, N] float32
///   - grid_tex: texture1d [n_levels] float32 - codebook as 1D texture
///   - su: [K] float32 row signs
///   - sv: [N] float32 column signs
///
/// @param packed_indices  Packed byte array [tiles_k, tiles_n, packed_bytes]
/// @param scales          Per-group scales [n_groups, N]
/// @param grid_tex        Codebook grid as 1D texture [n_levels]
/// @param su              Row sign vector [K]
/// @param sv              Column sign vector [N]
/// @param output          Dequantized output [K, N]
/// @param K               Number of rows
/// @param N               Number of columns
/// @param n_levels        Number of quantization levels (2^bits)
/// @param bits            Quantization bit width
/// @param group_size      Quantization group size
kernel void dequant_trellis_packed_tex(
    device const uchar* packed_indices [[buffer(0)]],
    constant float* scales [[buffer(1)]],
    texture1d<float, access::sample> grid_tex [[texture(0)]],
    constant float* su [[buffer(2)]],
    constant float* sv [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& n_levels [[buffer(7)]],
    constant uint& bits [[buffer(8)]],
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
    uint idx_in_tile = local_n * TILE_DIM + local_k;  // Transposed weight

    // Packed bytes per tile: ceil(256 * bits / 8)
    uint packed_bytes_per_tile = (TILE_SIZE * bits + 7) / 8;

    // Offset into packed_indices
    uint tile_offset = (tile_k * tiles_n + tile_n) * packed_bytes_per_tile;

    // Unpack the trellis index
    uint trellis_idx = unpack_index(packed_indices + tile_offset, idx_in_tile, bits);

    // Clamp to valid range (texture also handles this, but belt-and-suspenders)
    if (trellis_idx >= n_levels) {
        trellis_idx = 0;
    }

    // Load scale
    uint group_idx = k_idx / group_size;
    float scale = scales[group_idx * N + n_idx];

    // Fused dequantize: grid_tex[idx] * (scale * su * sv)
    // Precompute combined scale, then use texture lookup with single multiply
    float combined_scale = scale * su[k_idx] * sv[n_idx];

    // Texture sample returns float4, we only need .x component
    float normalized = grid_tex.sample(grid_sampler, float(trellis_idx)).x;
    output[k_idx * N + n_idx] = half(normalized * combined_scale);
}

// ============================================================================
// Vectorized Packed Index Dequantization Kernel (4x SIMD)
// ============================================================================

/// Vectorized trellis dequantization using SIMD index unpacking.
///
/// Each thread processes 4 consecutive elements along K using vectorized
/// unpack_index_x4 operations. This provides ~4x throughput improvement
/// for index extraction by loading uint32 values and extracting indices
/// via bit shifts rather than individual byte loads.
///
/// Grid Configuration:
///   - gid.x = column index n ∈ [0, N)
///   - gid.y = base row index k_base ∈ [0, K/4) - each thread handles k_base*4 to k_base*4+3
///
/// Alignment Requirements:
///   - K should ideally be a multiple of 4 for best efficiency
///   - Handles non-aligned K with bounds checking
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
/// @param bits            Quantization bit width (2, 3, or 4)
/// @param group_size      Quantization group size
kernel void dequant_trellis_packed_vec4(
    device const uchar* packed_indices [[buffer(0)]],
    constant float* scales [[buffer(1)]],
    constant float* grid [[buffer(2)]],
    constant float* su [[buffer(3)]],
    constant float* sv [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& n_levels [[buffer(8)]],
    constant uint& bits [[buffer(9)]],
    constant uint& group_size [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint n_idx = gid.x;
    uint k_base = gid.y * 4;  // Each thread handles 4 consecutive K values

    if (n_idx >= N || k_base >= K) return;

    // Compute tile coordinates for this column
    uint tile_n = n_idx / TILE_DIM;
    uint local_n = n_idx % TILE_DIM;
    uint tiles_n = (N + TILE_DIM - 1) / TILE_DIM;

    // Packed bytes per tile: ceil(256 * bits / 8)
    uint packed_bytes_per_tile = (TILE_SIZE * bits + 7) / 8;

    // Load column sign once (shared across all 4 K values)
    float sv_val = sv[n_idx];

    // Process 4 consecutive K values
    // Check if all 4 values are within the same trellis tile
    uint tile_k = k_base / TILE_DIM;
    uint local_k_base = k_base % TILE_DIM;

    // Check if we can use vectorized path (all 4 within same tile)
    bool same_tile = (local_k_base + 3) < TILE_DIM;

    if (same_tile && k_base + 3 < K) {
        // Fast path: all 4 indices in same trellis tile, use SIMD unpack
        uint tile_offset = (tile_k * tiles_n + tile_n) * packed_bytes_per_tile;
        device const uchar* tile_packed = packed_indices + tile_offset;

        // Compute base index in tile for these 4 elements
        // Index layout: local_n * 16 + local_k (transposed weight)
        uint base_idx_in_tile = local_n * TILE_DIM + local_k_base;

        // Unpack 4 indices at once using SIMD
        uint4 indices = unpack_index_x4(tile_packed, base_idx_in_tile, bits);

        // Clamp to valid range
        indices = min(indices, uint4(n_levels - 1));

        // Load 4 row signs
        float4 su_vals = float4(su[k_base], su[k_base + 1], su[k_base + 2], su[k_base + 3]);

        // Compute group indices for scales (may span groups)
        uint group0 = k_base / group_size;
        uint group1 = (k_base + 1) / group_size;
        uint group2 = (k_base + 2) / group_size;
        uint group3 = (k_base + 3) / group_size;

        // Load scales (potentially from different groups)
        float4 scale_vals = float4(
            scales[group0 * N + n_idx],
            scales[group1 * N + n_idx],
            scales[group2 * N + n_idx],
            scales[group3 * N + n_idx]
        );

        // Dequantize: grid[idx] * scale * su * sv
        float4 combined = scale_vals * su_vals * sv_val;
        half4 results = half4(
            half(grid[indices.x] * combined.x),
            half(grid[indices.y] * combined.y),
            half(grid[indices.z] * combined.z),
            half(grid[indices.w] * combined.w)
        );

        // Store 4 results
        output[k_base * N + n_idx] = results.x;
        output[(k_base + 1) * N + n_idx] = results.y;
        output[(k_base + 2) * N + n_idx] = results.z;
        output[(k_base + 3) * N + n_idx] = results.w;
    } else {
        // Slow path: indices span tile boundary or near K edge
        // Fall back to scalar unpacking for correctness
        #pragma unroll(4)
        for (uint i = 0; i < 4; ++i) {
            uint k_idx = k_base + i;
            if (k_idx >= K) break;

            uint cur_tile_k = k_idx / TILE_DIM;
            uint cur_local_k = k_idx % TILE_DIM;

            uint tile_offset = (cur_tile_k * tiles_n + tile_n) * packed_bytes_per_tile;
            uint idx_in_tile = local_n * TILE_DIM + cur_local_k;

            uint trellis_idx = unpack_index(packed_indices + tile_offset, idx_in_tile, bits);
            if (trellis_idx >= n_levels) trellis_idx = 0;

            uint group_idx = k_idx / group_size;
            float scale = scales[group_idx * N + n_idx];

            float dequant_val = grid[trellis_idx] * scale * su[k_idx] * sv_val;
            output[k_idx * N + n_idx] = half(dequant_val);
        }
    }
}

// ============================================================================
// Tiled Vectorized Dequantization Kernel
// ============================================================================

/// Tile-based vectorized dequantization using threadgroup cooperation.
///
/// Each threadgroup processes a 16x16 trellis tile (256 elements).
/// Threads cooperatively load packed indices and dequantize using
/// SIMD vector operations for maximum throughput.
///
/// Threadgroup Configuration:
///   - 64 threads per threadgroup (each handles 4 elements)
///   - Thread i handles elements [i*4, i*4+3] within the tile
///
/// @param packed_indices  Packed byte array [tiles_k, tiles_n, packed_bytes]
/// @param scales          Per-group scales [n_groups, N]
/// @param grid            Codebook grid [n_levels]
/// @param su              Row sign vector [K]
/// @param sv              Column sign vector [N]
/// @param output          Dequantized output [K, N]
/// @param K               Number of rows
/// @param N               Number of columns
/// @param n_levels        Number of quantization levels
/// @param bits            Quantization bit width
/// @param group_size      Quantization group size
kernel void dequant_trellis_packed_tiled_vec4(
    device const uchar* packed_indices [[buffer(0)]],
    constant float* scales [[buffer(1)]],
    constant float* grid [[buffer(2)]],
    constant float* su [[buffer(3)]],
    constant float* sv [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& n_levels [[buffer(8)]],
    constant uint& bits [[buffer(9)]],
    constant uint& group_size [[buffer(10)]],
    uint tile_id [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_threadgroup]]
) {
    // Compute tile coordinates from tile_id
    uint tiles_n = (N + TILE_DIM - 1) / TILE_DIM;
    uint tile_k = tile_id / tiles_n;
    uint tile_n = tile_id % tiles_n;

    // Global K, N base for this tile
    uint k_tile_base = tile_k * TILE_DIM;
    uint n_tile_base = tile_n * TILE_DIM;

    // Bounds check for entire tile
    if (k_tile_base >= K) return;

    // Threadgroup memory for scales cache
    threadgroup float scales_cache[256];

    // Cooperative scales load
    load_scales_to_threadgroup(scales, scales_cache, N, n_tile_base, k_tile_base, group_size, K, lane_id, 256);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute groups spanned by this tile for cache lookup
    uint group_start = k_tile_base / group_size;

    // Packed bytes per tile
    uint packed_bytes_per_tile = (TILE_SIZE * bits + 7) / 8;
    uint tile_offset = tile_id * packed_bytes_per_tile;
    device const uchar* tile_packed = packed_indices + tile_offset;

    // Each thread handles 4 consecutive elements within the 256-element tile
    // lane_id ∈ [0, 63], each handles elements [lane_id*4, lane_id*4+3]
    uint base_idx = lane_id * 4;
    if (base_idx >= TILE_SIZE) return;

    // Unpack 4 indices at once using SIMD
    uint4 indices = unpack_index_x4(tile_packed, base_idx, bits);
    indices = min(indices, uint4(n_levels - 1));

    // Convert linear index to (local_k, local_n) for each of 4 elements
    // Layout: idx_in_tile = local_n * 16 + local_k (transposed)
    // So local_k = idx % 16, local_n = idx / 16
    uint4 local_k = uint4(base_idx % TILE_DIM, (base_idx + 1) % TILE_DIM,
                          (base_idx + 2) % TILE_DIM, (base_idx + 3) % TILE_DIM);
    uint4 local_n = uint4(base_idx / TILE_DIM, (base_idx + 1) / TILE_DIM,
                          (base_idx + 2) % TILE_DIM, (base_idx + 3) % TILE_DIM);

    // Global coordinates
    uint4 k_idx = k_tile_base + local_k;
    uint4 n_idx = n_tile_base + local_n;

    // Process each of the 4 elements
    #pragma unroll(4)
    for (uint i = 0; i < 4; ++i) {
        uint k = (i == 0) ? k_idx.x : (i == 1) ? k_idx.y : (i == 2) ? k_idx.z : k_idx.w;
        uint n = (i == 0) ? n_idx.x : (i == 1) ? n_idx.y : (i == 2) ? n_idx.z : n_idx.w;
        uint idx = (i == 0) ? indices.x : (i == 1) ? indices.y : (i == 2) ? indices.z : indices.w;

        if (k >= K || n >= N) continue;

        uint group_idx = k / group_size;
        uint group_rel = group_idx - group_start;
        uint cur_local_n = (i == 0) ? local_n.x : (i == 1) ? local_n.y : (i == 2) ? local_n.z : local_n.w;

        // Load scale from threadgroup cache
        float scale = scales_cache[group_rel * TILE_DIM + cur_local_n];
        float dequant_val = grid[idx] * scale * su[k] * sv[n];
        output[k * N + n] = half(dequant_val);
    }
}

// ============================================================================
// Threadgroup-Cached Codebook Primitives
// ============================================================================
//
// OPTIMIZATION: The codebook grid is small (4/8/16 values for 2/3/4 bit) but
// accessed by every thread in the threadgroup. Caching in threadgroup memory:
//   - Reduces global memory traffic by (threads_per_tg / 1) = 128-256x
//   - Enables coalesced single load instead of random scattered accesses
//   - Grid fits in ~64 bytes (16 floats) - trivial threadgroup memory cost
//
// Access pattern: All threads read grid[idx] where idx varies per-thread but
// the grid base address is identical. This is a broadcast-friendly pattern
// that benefits from L1/L2 cache on device memory, but threadgroup memory
// provides guaranteed low-latency access without cache pressure.
// ============================================================================

// Moved to top of file

/// Dequantize using threadgroup-cached codebook.
/// @param idx         Trellis index into codebook [0, n_levels-1]
/// @param scale       Per-group scale factor
/// @param grid_tg     Threadgroup-cached codebook [n_levels]
/// @return            Dequantized FP16 value
inline half dequant_trellis_element_cached(
    uint idx,
    float scale,
    threadgroup const float* grid_tg
) {
    return half(grid_tg[idx] * scale);
}

// ============================================================================
// Vectorized Grid Lookup from Threadgroup Cache (4 values at once)
// ============================================================================
//
// OPTIMIZATION: When processing 4 consecutive elements that share the same
// scale factor, we can batch the grid lookups and dequantization into a
// single vectorized operation. This provides:
//   - Better ALU utilization (SIMD lanes process in parallel)
//   - Reduced loop overhead (1 iteration vs 4)
//   - Compiler can better optimize register allocation
//
// Access pattern: For 4 consecutive elements in a tile row/column, the
// indices differ but scale is often shared (within same quantization group).
// ============================================================================

/// Dequantize 4 elements using vectorized grid lookup from threadgroup cache.
/// @param idx4        Four trellis indices packed as uint4
/// @param scale       Shared scale factor for all 4 elements
/// @param grid_tg     Threadgroup-cached codebook [n_levels]
/// @return            Four dequantized FP16 values as half4
inline half4 dequant_trellis_vec4_cached(
    uint4 idx4,
    float scale,
    threadgroup const float* grid_tg
) {
    // Batch lookup: grid[idx0], grid[idx1], grid[idx2], grid[idx3]
    float4 grid_vals = float4(
        grid_tg[idx4.x],
        grid_tg[idx4.y],
        grid_tg[idx4.z],
        grid_tg[idx4.w]
    );
    // Vectorized multiply: all 4 elements scaled in parallel
    return half4(grid_vals * scale);
}

/// Dequantize 4 elements with individual scales using vectorized grid lookup.
/// @param idx4        Four trellis indices packed as uint4
/// @param scale4      Four scale factors as float4
/// @param grid_tg     Threadgroup-cached codebook [n_levels]
/// @return            Four dequantized FP16 values as half4
inline half4 dequant_trellis_vec4_cached_scales(
    uint4 idx4,
    float4 scale4,
    threadgroup const float* grid_tg
) {
    float4 grid_vals = float4(
        grid_tg[idx4.x],
        grid_tg[idx4.y],
        grid_tg[idx4.z],
        grid_tg[idx4.w]
    );
    return half4(grid_vals * scale4);
}

/// Dequantize 4 elements with sign flips using vectorized operations.
/// Full dequant formula: grid[idx] * scale * su * sv
/// @param idx4        Four trellis indices packed as uint4
/// @param scale       Shared scale factor
/// @param su4         Four row signs as float4
/// @param sv4         Four column signs as float4
/// @param grid_tg     Threadgroup-cached codebook [n_levels]
/// @return            Four dequantized FP16 values as half4
inline half4 dequant_trellis_vec4_cached_full(
    uint4 idx4,
    float scale,
    float4 su4,
    float4 sv4,
    threadgroup const float* grid_tg
) {
    float4 grid_vals = float4(
        grid_tg[idx4.x],
        grid_tg[idx4.y],
        grid_tg[idx4.z],
        grid_tg[idx4.w]
    );
    // Fused multiply: grid * scale * su * sv
    // Compiler should optimize to FMA chain
    return half4(grid_vals * scale * su4 * sv4);
}



// ============================================================================
// Threadgroup-Cached Dequantization Kernels
// ============================================================================

/// Tile-optimized trellis dequantization with threadgroup-cached codebook.
///
/// This variant caches the codebook grid in threadgroup memory at tile
/// boundaries, reducing global memory traffic for the frequent grid lookups.
///
/// Memory Usage:
///   - Grid cache: 64 bytes (16 floats for 4-bit max)
///   - Well within threadgroup memory budget
///
/// @param indices     Trellis indices [tiles_k, tiles_n, 256]
/// @param scales      Per-group scales [n_groups, N]
/// @param grid        Codebook grid values [n_levels]
/// @param output      Dequantized output [K, N]
/// @param K           Number of rows
/// @param N           Number of columns
/// @param n_levels    Number of quantization levels
/// @param group_size  Elements per quantization group
kernel void dequant_trellis_tiled_cached(
    device const short* indices [[buffer(0)]],
    constant float* scales [[buffer(1)]],
    constant float* grid [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& n_levels [[buffer(6)]],
    constant uint& group_size [[buffer(7)]],
    uint tile_id [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_threadgroup]]
) {
    // Threadgroup memory for codebook cache
    threadgroup float grid_cache[MAX_GRID_LEVELS];

    // Cooperative grid load: first n_levels threads load the codebook
    load_grid_to_threadgroup(grid, grid_cache, n_levels, lane_id);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute tile coordinates from tile_id
    uint tiles_n = (N + TILE_DIM - 1) / TILE_DIM;
    uint tile_k = tile_id / tiles_n;
    uint tile_n = tile_id % tiles_n;

    // Global K, N base for this tile
    uint k_tile_base = tile_k * TILE_DIM;
    uint n_tile_base = tile_n * TILE_DIM;

    // Threadgroup memory for scales cache
    // Max groups per tile = ceil(16 / group_size) * tile coverage
    // Worst case: group_size=1, tile spans 16 groups -> 16*16=256 floats (1KB)
    threadgroup float scales_cache[256];

    // Cooperative scales load
    load_scales_to_threadgroup(scales, scales_cache, N, n_tile_base, k_tile_base, group_size, K, lane_id, 256);
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

    // Load scale from threadgroup cache
    uint group_idx = k_idx / group_size;
    uint group_start = k_tile_base / group_size;
    uint group_rel = group_idx - group_start;
    float scale = scales_cache[group_rel * TILE_DIM + local_n];

    // Dequantize using cached grid and cached scales
    output[k_idx * N + n_idx] = dequant_trellis_element_cached(uint(trellis_idx), scale, grid_cache);
}

/// Packed index dequantization with threadgroup-cached codebook.
///
/// Combines threadgroup-cached codebook for reduced global memory traffic
/// with packed index support for memory-efficient storage.
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
/// @param group_size      Quantization group size
kernel void dequant_trellis_packed_cached(
    device const uchar* packed_indices [[buffer(0)]],
    constant float* scales [[buffer(1)]],
    constant float* grid [[buffer(2)]],
    constant float* su [[buffer(3)]],
    constant float* sv [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& n_levels [[buffer(8)]],
    constant uint& bits [[buffer(9)]],
    constant uint& group_size [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_threadgroup]]
) {
    // Threadgroup memory for codebook cache
    threadgroup float grid_cache[MAX_GRID_LEVELS];

    // Cooperative grid load: first n_levels threads load the codebook
    load_grid_to_threadgroup(grid, grid_cache, n_levels, lane_id);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_idx = gid.x;
    uint k_idx = gid.y;

    if (n_idx >= N || k_idx >= K) return;

    // Compute tile coordinates
    uint tile_k = k_idx / TILE_DIM;
    uint tile_n = n_idx / TILE_DIM;
    uint local_k = k_idx % TILE_DIM;
    uint local_n = n_idx % TILE_DIM;

    uint tiles_n = (N + TILE_DIM - 1) / TILE_DIM;
    uint idx_in_tile = local_n * TILE_DIM + local_k;  // Transposed weight

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

    // Threadgroup memory for scales cache
    // Compute tile base coordinates
    uint k_tile_base = tile_k * TILE_DIM;
    uint n_tile_base = tile_n * TILE_DIM;
    threadgroup float scales_cache[256];

    // Compute groups spanned by this tile along K dimension
    uint group_start = k_tile_base / group_size;
    uint k_tile_end = min(k_tile_base + TILE_DIM, K);
    uint group_end = (k_tile_end + group_size - 1) / group_size;
    uint groups_in_tile = group_end - group_start;

    // Total scale values to load: groups_in_tile * 16 columns
    uint total_scales = groups_in_tile * TILE_DIM;

    // Cooperatively load: each thread loads one or more values
    for (uint i = lane_id; i < total_scales; i += 256) {
        uint group_rel = i / TILE_DIM;
        uint col_in_tile = i % TILE_DIM;
        uint group_abs = group_start + group_rel;
        uint n_idx_load = n_tile_base + col_in_tile;

        if (n_idx_load < N) {
            scales_cache[i] = scales[group_abs * N + n_idx_load];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load scale from threadgroup cache
    uint group_idx = k_idx / group_size;
    uint group_rel = group_idx - group_start;
    float scale = scales_cache[group_rel * TILE_DIM + local_n];

    float combined_scale = scale * su[k_idx] * sv[n_idx];

    // Dequantize using cached grid: grid[idx] * combined_scale
    output[k_idx * N + n_idx] = half(grid_cache[trellis_idx] * combined_scale);
}

/// Vectorized packed dequantization with threadgroup-cached codebook (4x SIMD).
///
/// Combines three optimizations:
///   1. Threadgroup-cached codebook for reduced global memory traffic
///   2. Vectorized 4-element index unpacking using SIMD
///   3. Vectorized grid lookups for better ALU utilization
///
/// Grid Configuration:
///   - gid.x = column index n ∈ [0, N)
///   - gid.y = base row index k_base ∈ [0, K/4) - each thread handles k_base*4 to k_base*4+3
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
/// @param bits            Quantization bit width (2, 3, or 4)
/// @param group_size      Quantization group size
kernel void dequant_trellis_packed_cached_vec4(
    device const uchar* packed_indices [[buffer(0)]],
    constant float* scales [[buffer(1)]],
    constant float* grid [[buffer(2)]],
    constant float* su [[buffer(3)]],
    constant float* sv [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& n_levels [[buffer(8)]],
    constant uint& bits [[buffer(9)]],
    constant uint& group_size [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_threadgroup]]
) {
    // Threadgroup memory for codebook cache
    threadgroup float grid_cache[MAX_GRID_LEVELS];

    // Cooperative grid load: first n_levels threads load the codebook
    load_grid_to_threadgroup(grid, grid_cache, n_levels, lane_id);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_idx = gid.x;
    uint k_base = gid.y * 4;  // Each thread handles 4 consecutive K values

    if (n_idx >= N || k_base >= K) return;

    // Compute tile coordinates for this column
    uint tile_n = n_idx / TILE_DIM;
    uint local_n = n_idx % TILE_DIM;
    uint tiles_n = (N + TILE_DIM - 1) / TILE_DIM;

    // Packed bytes per tile: ceil(256 * bits / 8)
    uint packed_bytes_per_tile = (TILE_SIZE * bits + 7) / 8;

    // Load column sign once (shared across all 4 K values)
    float sv_val = sv[n_idx];

    // Process 4 consecutive K values
    uint tile_k = k_base / TILE_DIM;
    uint local_k_base = k_base % TILE_DIM;

    // Threadgroup memory for scales cache
    uint k_tile_base = tile_k * TILE_DIM;
    uint n_tile_base = tile_n * TILE_DIM;
    threadgroup float scales_cache[256];

    // Compute groups spanned by this tile along K dimension
    uint group_start = k_tile_base / group_size;
    uint k_tile_end = min(k_tile_base + TILE_DIM, K);
    uint group_end = (k_tile_end + group_size - 1) / group_size;
    uint groups_in_tile = group_end - group_start;

    // Total scale values to load: groups_in_tile * 16 columns
    uint total_scales = groups_in_tile * TILE_DIM;

    // Cooperatively load: each thread loads one or more values
    #pragma unroll(4)
    for (uint i = lane_id; i < total_scales; i += 256) {
        uint group_rel = i / TILE_DIM;
        uint col_in_tile = i % TILE_DIM;
        uint group_abs = group_start + group_rel;
        uint n_idx_load = n_tile_base + col_in_tile;

        if (n_idx_load < N) {
            scales_cache[i] = scales[group_abs * N + n_idx_load];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Check if we can use vectorized path (all 4 within same tile)
    bool same_tile = (local_k_base + 3) < TILE_DIM;

    if (same_tile && k_base + 3 < K) {
        // Fast path: all 4 indices in same trellis tile, use SIMD unpack + cached grid
        uint tile_offset = (tile_k * tiles_n + tile_n) * packed_bytes_per_tile;
        device const uchar* tile_packed = packed_indices + tile_offset;

        // Compute base index in tile for these 4 elements
        // Index layout: local_n * 16 + local_k (transposed weight)
        uint base_idx_in_tile = local_n * TILE_DIM + local_k_base;

        // Unpack 4 indices at once using SIMD
        uint4 indices = unpack_index_x4(tile_packed, base_idx_in_tile, bits);

        // Clamp to valid range
        indices = min(indices, uint4(n_levels - 1));

        // Load 4 row signs
        float4 su_vals = float4(su[k_base], su[k_base + 1], su[k_base + 2], su[k_base + 3]);

        // Compute group indices for scales (may span groups)
        uint group0 = k_base / group_size;
        uint group1 = (k_base + 1) / group_size;
        uint group2 = (k_base + 2) / group_size;
        uint group3 = (k_base + 3) / group_size;

        // Load scales from threadgroup cache
        uint group0_rel = group0 - group_start;
        uint group1_rel = group1 - group_start;
        uint group2_rel = group2 - group_start;
        uint group3_rel = group3 - group_start;
        float4 scale_vals = float4(
            scales_cache[group0_rel * TILE_DIM + local_n],
            scales_cache[group1_rel * TILE_DIM + local_n],
            scales_cache[group2_rel * TILE_DIM + local_n],
            scales_cache[group3_rel * TILE_DIM + local_n]
        );

        // Vectorized dequantization using cached grid
        float4 grid_vals = float4(
            grid_cache[indices.x],
            grid_cache[indices.y],
            grid_cache[indices.z],
            grid_cache[indices.w]
        );
        float4 combined = scale_vals * su_vals * sv_val;
        half4 results = half4(grid_vals * combined);

        // Store 4 results
        output[k_base * N + n_idx] = results.x;
        output[(k_base + 1) * N + n_idx] = results.y;
        output[(k_base + 2) * N + n_idx] = results.z;
        output[(k_base + 3) * N + n_idx] = results.w;
    } else {
        // Slow path: indices span tile boundary or near K edge
        // Fall back to scalar unpacking but still use cached grid and scales
        #pragma unroll(4)
        for (uint i = 0; i < 4; ++i) {
            uint k_idx = k_base + i;
            if (k_idx >= K) break;

            uint cur_tile_k = k_idx / TILE_DIM;
            uint cur_local_k = k_idx % TILE_DIM;

            uint tile_offset = (cur_tile_k * tiles_n + tile_n) * packed_bytes_per_tile;
            uint idx_in_tile = local_n * TILE_DIM + cur_local_k;

            uint trellis_idx = unpack_index(packed_indices + tile_offset, idx_in_tile, bits);
            if (trellis_idx >= n_levels) trellis_idx = 0;

            uint group_idx = k_idx / group_size;
            uint group_rel = group_idx - group_start;

            // Load scale from threadgroup cache if available
            float scale;
            if (group_rel < groups_in_tile) {
                scale = scales_cache[group_rel * TILE_DIM + local_n];
            } else {
                // Fallback to global memory if spanned multiple tile groups
                scale = scales[group_idx * N + n_idx];
            }

            // Use cached grid lookup
            float dequant_val = grid_cache[trellis_idx] * scale * su[k_idx] * sv_val;
            output[k_idx * N + n_idx] = half(dequant_val);
        }
    }
}

/// Tile-based dequantization with threadgroup-cached codebook and vectorized lookups.
///
/// Each threadgroup processes a 16x16 trellis tile (256 elements) with:
///   1. Codebook cached in threadgroup memory (64 bytes)
///   2. Threads cooperatively load and dequant using SIMD vector operations
///
/// Memory Usage:
///   - Grid cache: 64 bytes (16 floats)
///   - Total threadgroup memory: ~64 bytes (well under 32KB limit)
///
/// Threadgroup Configuration:
///   - 64 threads per threadgroup (each handles 4 elements)
///   - Thread i handles elements [i*4, i*4+3] within the tile
///
/// @param packed_indices  Packed byte array [tiles_k, tiles_n, packed_bytes]
/// @param scales          Per-group scales [n_groups, N]
/// @param grid            Codebook grid [n_levels]
/// @param su              Row sign vector [K]
/// @param sv              Column sign vector [N]
/// @param output          Dequantized output [K, N]
/// @param K               Number of rows
/// @param N               Number of columns
/// @param n_levels        Number of quantization levels
/// @param bits            Quantization bit width
/// @param group_size      Quantization group size
kernel void dequant_trellis_packed_tiled_cached_vec4(
    device const uchar* packed_indices [[buffer(0)]],
    constant float* scales [[buffer(1)]],
    constant float* grid [[buffer(2)]],
    constant float* su [[buffer(3)]],
    constant float* sv [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& n_levels [[buffer(8)]],
    constant uint& bits [[buffer(9)]],
    constant uint& group_size [[buffer(10)]],
    uint tile_id [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_threadgroup]]
) {
    // Threadgroup memory for codebook cache
    threadgroup float grid_cache[MAX_GRID_LEVELS];

    // Cooperative grid load: first n_levels threads load the codebook
    load_grid_to_threadgroup(grid, grid_cache, n_levels, lane_id);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute tile coordinates from tile_id
    uint tiles_n = (N + TILE_DIM - 1) / TILE_DIM;
    uint tile_k = tile_id / tiles_n;
    uint tile_n = tile_id % tiles_n;

    // Global K, N base for this tile
    uint k_tile_base = tile_k * TILE_DIM;
    uint n_tile_base = tile_n * TILE_DIM;

    // Bounds check for entire tile
    if (k_tile_base >= K) return;

    // Threadgroup memory for scales cache
    threadgroup float scales_cache[256];

    // Cooperative scales load at kernel start
    load_scales_to_threadgroup(scales, scales_cache, N, n_tile_base, k_tile_base, group_size, K, lane_id, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute groups spanned by this tile for cache lookup
    uint group_start = k_tile_base / group_size;

    // Packed bytes per tile
    uint packed_bytes_per_tile = (TILE_SIZE * bits + 7) / 8;
    uint tile_offset = tile_id * packed_bytes_per_tile;
    device const uchar* tile_packed = packed_indices + tile_offset;

    // Each thread handles 4 consecutive elements within the 256-element tile
    // lane_id ∈ [0, 63], each handles elements [lane_id*4, lane_id*4+3]
    uint base_idx = lane_id * 4;
    if (base_idx >= TILE_SIZE) return;

    // Unpack 4 indices at once using SIMD
    uint4 indices = unpack_index_x4(tile_packed, base_idx, bits);
    indices = min(indices, uint4(n_levels - 1));

    // Convert linear index to (local_k, local_n) for each of 4 elements
    // Layout: idx_in_tile = local_n * 16 + local_k (transposed)
    // So local_k = idx % 16, local_n = idx / 16
    uint4 local_k = uint4(base_idx % TILE_DIM, (base_idx + 1) % TILE_DIM,
                          (base_idx + 2) % TILE_DIM, (base_idx + 3) % TILE_DIM);
    uint4 local_n = uint4(base_idx / TILE_DIM, (base_idx + 1) / TILE_DIM,
                          (base_idx + 2) / TILE_DIM, (base_idx + 3) / TILE_DIM);

    // Global coordinates
    uint4 k_idx = k_tile_base + local_k;
    uint4 n_idx = n_tile_base + local_n;

    // Vectorized grid lookup from cache
    float4 grid_vals = float4(
        grid_cache[indices.x],
        grid_cache[indices.y],
        grid_cache[indices.z],
        grid_cache[indices.w]
    );

    // Process each of the 4 elements
    #pragma unroll(4)
    for (uint i = 0; i < 4; ++i) {
        uint k = (i == 0) ? k_idx.x : (i == 1) ? k_idx.y : (i == 2) ? k_idx.z : k_idx.w;
        uint n = (i == 0) ? n_idx.x : (i == 1) ? n_idx.y : (i == 2) ? n_idx.z : n_idx.w;
        float gv = (i == 0) ? grid_vals.x : (i == 1) ? grid_vals.y : (i == 2) ? grid_vals.z : grid_vals.w;

        if (k >= K || n >= N) continue;

        // Load scale from threadgroup cache instead of global memory
        uint group_idx = k / group_size;
        uint group_rel = group_idx - group_start;
        uint cur_local_n = (i == 0) ? local_n.x : (i == 1) ? local_n.y : (i == 2) ? local_n.z : local_n.w;
        float scale = scales_cache[group_rel * TILE_DIM + cur_local_n];

        float dequant_val = gv * scale * su[k] * sv[n];
        output[k * N + n] = half(dequant_val);
    }
}
