// gemm_trellis.metal - Fused Dequant+GEMM Kernel for Trellis-Quantized Weights
// ============================================================================
//
// High-performance fused kernel for trellis-quantized GEMM on Apple Metal.
// Performs on-the-fly dequantization of packed trellis indices during the
// GEMM computation without materializing the full FP16 weight matrix.
//
// Trellis Quantization Background:
//   - Weights are quantized using Viterbi algorithm through a codebook grid
//   - Each 16x16 tile (256 elements) shares quantization parameters
//   - Indices are packed into uint8: 2-bit (64B), 3-bit (96B), 4-bit (128B), or 8-bit (256B)
//   - Dequant formula: w[k,n] = grid[idx] * scale * su[k] * sv[n]
//
// INT8 Quantization Support (NEW):
//   - Added support for INT8 quantization (W8A16 mode)
//   - Symmetric quantization: w = int8_val * scale * su * sv
//   - Asymmetric quantization: w = (int8_val - zero_point) * scale * su * sv
//   - Per-group scale factors (typically group_size = 32 or 128)
//   - Optional per-group zero points for asymmetric mode
//   - Optional Hadamard sign vectors (su, sv) for preprocessing
//   - Kernel: gemm_trellis_int8 (buffer slots 0-11)
//
// Kernel Architecture:
//   - Tile dimensions: TILE_M=64, TILE_N=64, TILE_K=32
//   - Threadgroup: 4 simdgroups (128 threads), each handling 32x32 output
//   - Double-buffered A tiles in threadgroup memory (8KB per buffer)
//   - On-the-fly B dequantization with per-simdgroup staging (512B)
//   - Never materializes full B matrix - dequant happens in registers
//
// Memory Layout (Trellis):
//   - A: [M, K] half - input activations (row-major)
//   - packed_indices: [tiles_k, tiles_n, packed_bytes] uint8 - packed trellis
//   - scales: [K/group_size, N] float32 - per-group scale factors
//   - grid: [n_levels] float32 - codebook quantization centers
//   - su: [K] float32 - row signs for Hadamard inverse
//   - sv: [N] float32 - column signs for Hadamard inverse
//   - C: [M, N] half - output matrix (row-major)
//
// Memory Layout (INT8):
//   - A: [M, K] half - input activations (row-major)
//   - int8_weights: [K, N] int8 (packed as uchar, row-major)
//   - scales: [K/group_size, N] float32 - per-group scale factors
//   - zeros: [K/group_size, N] float32 - per-group zero points (nullable)
//   - su: [K] float32 - row signs (nullable)
//   - sv: [N] float32 - column signs (nullable)
//   - C: [M, N] half - output matrix (row-major)
//
// ============================================================================

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// ============================================================================
// Texture Sampler Configuration
// ============================================================================
//
// The codebook grid is small (4/8/16 values for 2/3/4 bit) but accessed
// repeatedly by every thread. Using texture1d<float> provides:
// - Hardware texture cache optimized for repeated small lookups
// - Better cache line utilization than device buffer for tiny arrays
// - Automatic bounds checking with clamp-to-edge addressing
//
// ============================================================================

/// Constexpr sampler for codebook texture lookup.
/// Uses nearest filtering (no interpolation needed) and clamp-to-edge addressing.
constexpr sampler grid_sampler(coord::pixel,          // Use pixel coordinates (0-N)
                               address::clamp_to_edge, // Clamp OOB accesses
                               filter::nearest);       // No interpolation

// ============================================================================
// Tile Dimensions - Optimized for Apple Silicon
// ============================================================================

// Main GEMM tile dimensions
constant constexpr uint TILE_M = 128;     // Output rows per threadgroup
constant constexpr uint TILE_N = 128;     // Output cols per threadgroup
constant constexpr uint TILE_K = 32;      // K-reduction per mainloop iteration
constant constexpr uint K_TILES = TILE_K / 8;  // 4 simdgroup MMA ops per K-block

// Simdgroup configuration: 8 simdgroups tile the 128x128 output
// Each simdgroup handles 32x64 output (4x8 blocks of 8x8 tiles)
constant constexpr uint SIMDGROUPS_PER_TG = 8;
constant constexpr uint THREADS_PER_TG = SIMDGROUPS_PER_TG * 32;  // 256 threads
constant constexpr uint SG_M_TILES = 4;   // 4 rows of 8x8 = 32 rows per simdgroup
constant constexpr uint SG_N_TILES = 8;   // 8 cols of 8x8 = 64 cols per simdgroup

// Trellis tile dimensions (weights are stored in 16x16 tiles)
constant constexpr uint TRELLIS_TILE_DIM = 16;   // 16x16 tile dimension
constant constexpr uint TRELLIS_TILE_SIZE = 256; // Total elements per tile

// ============================================================================
// Packed Index Unpacking Primitives
// ============================================================================

/// Unpack a 2-bit trellis index from packed bytes.
/// Layout: 4 indices per byte (bits [0:1], [2:3], [4:5], [6:7])
inline uint unpack_2bit_index(device const uchar* packed, uint idx_in_tile) {
    uint byte_idx = idx_in_tile >> 2;        // Divide by 4
    uint bit_offset = (idx_in_tile & 3) << 1; // (idx % 4) * 2
    return (packed[byte_idx] >> bit_offset) & 0x3;
}

/// Unpack a 3-bit trellis index from packed bytes.
/// Layout: 8 indices per 3 bytes (packed across byte boundaries)
/// This is the most complex packing due to non-byte alignment.
inline uint unpack_3bit_index(device const uchar* packed, uint idx_in_tile) {
    uint bit_offset = idx_in_tile * 3;
    uint byte_idx = bit_offset >> 3;         // Divide by 8
    uint bit_in_byte = bit_offset & 7;       // Modulo 8
    
    // Read up to 2 bytes (indices may span byte boundary)
    uint packed_val = uint(packed[byte_idx]);
    if (bit_in_byte + 3 > 8) {
        packed_val |= uint(packed[byte_idx + 1]) << 8;
    }
    return (packed_val >> bit_in_byte) & 0x7;
}

/// Unpack a 4-bit trellis index from packed bytes.
/// Layout: 2 indices per byte (bits [0:3], [4:7])
inline uint unpack_4bit_index(device const uchar* packed, uint idx_in_tile) {
    uint byte_idx = idx_in_tile >> 1;        // Divide by 2
    uint shift = (idx_in_tile & 1) << 2;      // 0 or 4
    return (packed[byte_idx] >> shift) & 0xF;
}

/// Unpack an 8-bit trellis index from packed bytes.
/// Layout: 1 index per byte (direct read - no unpacking needed).
/// For 8-bit quantization, each element is stored as a single byte.
inline uint unpack_8bit_index(device const uchar* packed, uint idx_in_tile) {
    return uint(packed[idx_in_tile]);
}

// ============================================================================
// INT8 Dequantization Primitives
// ============================================================================
//
// These functions handle signed INT8 weight dequantization for W8A16 GEMM.
// Unlike trellis quantization (codebook indices), INT8 weights are stored
// as signed bytes directly representing quantized values.
//
// INT8 dequantization formula:
//   Symmetric:   w[k,n] = int8_val * scale * su[k] * sv[n]
//   Asymmetric:  w[k,n] = (int8_val - zero) * scale * su[k] * sv[n]
//
// ============================================================================

/// Extract signed INT8 value from packed byte data.
/// Metal's int8_t cast handles sign extension automatically.
inline int8_t extract_s8(device const uchar* packed, uint idx) {
    return int8_t(packed[idx]);
}

/// Dequantize a single INT8 weight with symmetric quantization.
/// Formula: result = int8_value * scale * su * sv
inline half dequant_int8_element_sym(
    int8_t int8_val,
    float scale,
    float su,
    float sv
) {
    return half(float(int8_val) * scale * su * sv);
}

/// Dequantize a single INT8 weight with symmetric quantization (fused scale).
/// Formula: result = int8_value * combined_scale
/// where combined_scale = scale * su * sv (precomputed)
inline half dequant_int8_element_fused(
    int8_t int8_val,
    float combined_scale
) {
    return half(float(int8_val) * combined_scale);
}

/// Dequantize a single INT8 weight with asymmetric quantization.
/// Formula: result = (int8_value - zero_point) * scale * su * sv
inline half dequant_int8_element_asym(
    int8_t int8_val,
    float scale,
    float zero_point,
    float su,
    float sv
) {
    return half((float(int8_val) - zero_point) * scale * su * sv);
}

/// Dequantize a single INT8 weight with asymmetric quantization (fused scale).
/// Formula: result = (int8_value - zero_point) * combined_scale
/// where combined_scale = scale * su * sv (precomputed)
inline half dequant_int8_element_asym_fused(
    int8_t int8_val,
    float combined_scale,
    float zero_point
) {
    return half((float(int8_val) - zero_point) * combined_scale);
}

/// Dequantize 4 consecutive INT8 values with symmetric quantization.
/// Optimized for SIMD processing of weight rows.
inline void dequant_int8_x4_sym(
    device const uchar* packed,
    uint base_idx,
    float combined_scale,
    thread half4& out
) {
    int8_t b0 = extract_s8(packed, base_idx + 0);
    int8_t b1 = extract_s8(packed, base_idx + 1);
    int8_t b2 = extract_s8(packed, base_idx + 2);
    int8_t b3 = extract_s8(packed, base_idx + 3);
    
    out = half4(
        float(b0) * combined_scale,
        float(b1) * combined_scale,
        float(b2) * combined_scale,
        float(b3) * combined_scale
    );
}

/// Dequantize 4 consecutive INT8 values with asymmetric quantization.
/// Optimized for SIMD processing of weight rows.
inline void dequant_int8_x4_asym(
    device const uchar* packed,
    uint base_idx,
    float combined_scale,
    float zero_point,
    thread half4& out
) {
    int8_t b0 = extract_s8(packed, base_idx + 0);
    int8_t b1 = extract_s8(packed, base_idx + 1);
    int8_t b2 = extract_s8(packed, base_idx + 2);
    int8_t b3 = extract_s8(packed, base_idx + 3);
    
    float fz = zero_point;
    out = half4(
        (float(b0) - fz) * combined_scale,
        (float(b1) - fz) * combined_scale,
        (float(b2) - fz) * combined_scale,
        (float(b3) - fz) * combined_scale
    );
}

/// Dequantize 8 consecutive INT8 values with symmetric quantization.
/// Used for loading 8-element weight columns for MMA operations.
inline void dequant_int8_x8_sym(
    device const uchar* packed,
    uint base_idx,
    float combined_scale,
    thread half* out
) {
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        int8_t val = extract_s8(packed, base_idx + i);
        out[i] = half(float(val) * combined_scale);
    }
}

/// Dequantize 8 consecutive INT8 values with asymmetric quantization.
/// Used for loading 8-element weight columns for MMA operations.
inline void dequant_int8_x8_asym(
    device const uchar* packed,
    uint base_idx,
    float combined_scale,
    float zero_point,
    thread half* out
) {
    float fz = zero_point;
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        int8_t val = extract_s8(packed, base_idx + i);
        out[i] = half((float(val) - fz) * combined_scale);
    }
}

/// Generic index unpack dispatcher.
/// Routes to the appropriate unpack function based on bit width.
inline uint unpack_trellis_index(device const uchar* packed, uint idx_in_tile, uint bits) {
    switch (bits) {
        case 2: return unpack_2bit_index(packed, idx_in_tile);
        case 3: return unpack_3bit_index(packed, idx_in_tile);
        case 4: return unpack_4bit_index(packed, idx_in_tile);
        case 8: return unpack_8bit_index(packed, idx_in_tile);
        default: return unpack_3bit_index(packed, idx_in_tile);  // Default to 3-bit
    }
}

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
/// Loads a single byte (containing 4 indices) and extracts all 4 at once.
/// @param packed   Pointer to packed byte data
/// @param base_idx Starting index in tile (must be aligned to 4)
/// @return uint4 containing 4 unpacked indices (values 0-3)
inline uint4 unpack_2bit_x4(device const uchar* packed, uint base_idx) {
    // 4 indices per byte, so base_idx/4 gives the byte index
    uchar byte_val = packed[base_idx >> 2];
    return uint4(
        (byte_val >> 0) & 0x3,
        (byte_val >> 2) & 0x3,
        (byte_val >> 4) & 0x3,
        (byte_val >> 6) & 0x3
    );
}

/// Unpack 4 consecutive 3-bit indices from a 32-bit packed value.
/// Extracts indices at bit offsets [base, base+3, base+6, base+9].
/// @param packed32       32-bit value containing packed indices
/// @param bit_offset_base Starting bit offset within packed32
/// @return uint4 containing 4 unpacked indices (values 0-7)
inline uint4 unpack_3bit_x4(uint packed32, uint bit_offset_base) {
    return uint4(
        (packed32 >> bit_offset_base) & 0x7,
        (packed32 >> (bit_offset_base + 3)) & 0x7,
        (packed32 >> (bit_offset_base + 6)) & 0x7,
        (packed32 >> (bit_offset_base + 9)) & 0x7
    );
}

/// Unpack 4 consecutive 3-bit indices from packed bytes.
/// Loads enough bytes to cover 12 bits (4 × 3-bit indices) and extracts.
/// @param packed   Pointer to packed byte data
/// @param base_idx Starting index in tile (must be aligned to 4)
/// @return uint4 containing 4 unpacked indices (values 0-7)
inline uint4 unpack_3bit_x4_from_bytes(device const uchar* packed, uint base_idx) {
    uint bit_offset = base_idx * 3;
    uint byte_idx = bit_offset >> 3;
    uint bit_in_byte = bit_offset & 7;

    // Load 3 bytes to cover 12+ bits (4 indices × 3 bits = 12 bits)
    uint packed32 = uint(packed[byte_idx]) |
                    (uint(packed[byte_idx + 1]) << 8) |
                    (uint(packed[byte_idx + 2]) << 16);

    return unpack_3bit_x4(packed32, bit_in_byte);
}

/// Unpack 8 consecutive 3-bit indices from packed bytes.
/// Loads 4 bytes to cover the 24-bit span of 8 × 3-bit indices.
/// @param byte_ptr         Pointer to start of packed data for this tile
/// @param base_bit_in_byte Starting bit offset within first byte
/// @param indices_lo       Output: indices 0-3
/// @param indices_hi       Output: indices 4-7
inline void unpack_3bit_x8(
    device const uchar* byte_ptr,
    uint base_bit_in_byte,
    thread uint4& indices_lo,
    thread uint4& indices_hi
) {
    // Load 4 bytes to cover 24+ bits
    uint packed32 = uint(byte_ptr[0]) |
                    (uint(byte_ptr[1]) << 8) |
                    (uint(byte_ptr[2]) << 16) |
                    (uint(byte_ptr[3]) << 24);

    // Extract 8 indices at 3-bit intervals from base offset
    uint bit = base_bit_in_byte;
    indices_lo = uint4(
        (packed32 >> bit) & 0x7,
        (packed32 >> (bit + 3)) & 0x7,
        (packed32 >> (bit + 6)) & 0x7,
        (packed32 >> (bit + 9)) & 0x7
    );
    indices_hi = uint4(
        (packed32 >> (bit + 12)) & 0x7,
        (packed32 >> (bit + 15)) & 0x7,
        (packed32 >> (bit + 18)) & 0x7,
        (packed32 >> (bit + 21)) & 0x7
    );
}

/// Unpack 4 consecutive 4-bit indices from packed bytes using SIMD.
/// Loads 2 bytes (containing 4 indices) and extracts using uchar2.
/// @param packed   Pointer to packed byte data
/// @param base_idx Starting index in tile (must be aligned to 4)
/// @return uint4 containing 4 unpacked indices (values 0-15)
inline uint4 unpack_4bit_x4(device const uchar* packed, uint base_idx) {
    // 2 indices per byte, so we need 2 bytes for 4 indices
    uint byte_idx = base_idx >> 1;
    uchar2 bytes = uchar2(packed[byte_idx], packed[byte_idx + 1]);
    return uint4(
        (bytes.x >> 0) & 0xF,
        (bytes.x >> 4) & 0xF,
        (bytes.y >> 0) & 0xF,
        (bytes.y >> 4) & 0xF
    );
}

/// Unpack 8 consecutive 4-bit indices from packed bytes using SIMD.
/// Loads 4 bytes (containing 8 indices) and extracts all at once.
/// @param packed   Pointer to packed byte data
/// @param base_idx Starting index in tile (must be aligned to 8)
/// @param indices_lo Output: indices 0-3
/// @param indices_hi Output: indices 4-7
inline void unpack_4bit_x8(
    device const uchar* packed,
    uint base_idx,
    thread uint4& indices_lo,
    thread uint4& indices_hi
) {
    // 2 indices per byte, so we need 4 bytes for 8 indices
    uint byte_idx = base_idx >> 1;
    uchar4 bytes = uchar4(
        packed[byte_idx],
        packed[byte_idx + 1],
        packed[byte_idx + 2],
        packed[byte_idx + 3]
    );
    indices_lo = uint4(
        (bytes.x >> 0) & 0xF,
        (bytes.x >> 4) & 0xF,
        (bytes.y >> 0) & 0xF,
        (bytes.y >> 4) & 0xF
    );
    indices_hi = uint4(
        (bytes.z >> 0) & 0xF,
        (bytes.z >> 4) & 0xF,
        (bytes.w >> 0) & 0xF,
        (bytes.w >> 4) & 0xF
    );
}

/// Unpack 4 consecutive 8-bit indices from packed bytes using SIMD.
/// For 8-bit quantization, each index is a direct byte read.
/// @param packed   Pointer to packed byte data
/// @param base_idx Starting index in tile (must be aligned to 4)
/// @return uint4 containing 4 unpacked indices (values 0-255)
inline uint4 unpack_8bit_x4(device const uchar* packed, uint base_idx) {
    return uint4(
        packed[base_idx],
        packed[base_idx + 1],
        packed[base_idx + 2],
        packed[base_idx + 3]
    );
}

/// Unpack 8 consecutive 8-bit indices from packed bytes using SIMD.
/// For 8-bit quantization, each index is a direct byte read.
/// @param packed   Pointer to packed byte data
/// @param base_idx Starting index in tile (must be aligned to 8)
/// @param indices_lo Output: indices 0-3
/// @param indices_hi Output: indices 4-7
inline void unpack_8bit_x8(
    device const uchar* packed,
    uint base_idx,
    thread uint4& indices_lo,
    thread uint4& indices_hi
) {
    indices_lo = uint4(
        packed[base_idx],
        packed[base_idx + 1],
        packed[base_idx + 2],
        packed[base_idx + 3]
    );
    indices_hi = uint4(
        packed[base_idx + 4],
        packed[base_idx + 5],
        packed[base_idx + 6],
        packed[base_idx + 7]
    );
}

/// Generic vectorized unpack dispatcher for 4 consecutive indices.
/// Routes to the appropriate vectorized unpack function based on bit width.
/// @param packed   Pointer to packed byte data
/// @param base_idx Starting index in tile (must be aligned to 4)
/// @param bits     Bit width (2, 3, 4, or 8)
/// @return uint4 containing 4 unpacked indices
inline uint4 unpack_trellis_x4(device const uchar* packed, uint base_idx, uint bits) {
    switch (bits) {
        case 2: return unpack_2bit_x4(packed, base_idx);
        case 3: return unpack_3bit_x4_from_bytes(packed, base_idx);
        case 4: return unpack_4bit_x4(packed, base_idx);
        case 8: return unpack_8bit_x4(packed, base_idx);
        default: return unpack_3bit_x4_from_bytes(packed, base_idx);
    }
}

/// Compute packed bytes per trellis tile based on bit width.
inline uint packed_bytes_per_trellis_tile(uint bits) {
    return (TRELLIS_TILE_SIZE * bits + 7) / 8;  // ceil(256 * bits / 8)
}

// ============================================================================
// Software Prefetching for Weight Data
// ============================================================================
//
// Prefetch next K-tile's packed weight indices into threadgroup memory
// while computing on current tile. This hides memory latency for weight loads.
//
// Strategy:
//   - Use a threadgroup buffer to stage prefetched packed indices
//   - Each thread prefetches a portion of the packed data cooperatively
//   - Double-buffer approach: prefetch tile K+1 while computing tile K
//   - Prefetched data is read from threadgroup memory instead of device memory
//
// Memory hierarchy benefit:
//   - Device memory: ~200-400 cycles latency
//   - Threadgroup memory: ~2-4 cycles latency
//   - By prefetching, we overlap device memory latency with compute
//
// ============================================================================

// Prefetch buffer sizing:
// - TILE_K=32 spans 2 trellis tiles (32/16 = 2 tiles in K dimension)
// - Each N-column requires one trellis tile per K-block
// - We prefetch for the N-columns this simdgroup will process
// - 8-bit worst case: 256 bytes per tile
// - We prefetch tiles for the next K-iteration
constant constexpr uint PREFETCH_TILES_K = (TILE_K + TRELLIS_TILE_DIM - 1) / TRELLIS_TILE_DIM;  // 2
constant constexpr uint MAX_PACKED_BYTES = 256;  // 8-bit: ceil(256 * 8 / 8) = 256

// Prefetch buffer: one buffer per simdgroup, covering N-tiles it processes
// Each simdgroup processes SG_N_TILES=8 sub-tiles of 8 columns = 64 columns
// That spans ceil(64/16)=4 trellis tiles in N dimension
constant constexpr uint PREFETCH_N_TILES = (SG_N_TILES * 8 + TRELLIS_TILE_DIM - 1) / TRELLIS_TILE_DIM;  // 4
constant constexpr uint PREFETCH_BUF_SIZE_PER_SG = PREFETCH_TILES_K * PREFETCH_N_TILES * MAX_PACKED_BYTES;  // 2*4*256=2048 bytes
constant constexpr uint PREFETCH_MAX_BYTES_PER_LANE = (PREFETCH_BUF_SIZE_PER_SG + 31) / 32;  // Max bytes per lane

/// Cooperative prefetch of packed indices for next K-block.
/// All threads in a simdgroup participate in loading data to threadgroup memory.
///
/// @param packed_indices   Global packed index buffer
/// @param prefetch_buf     Threadgroup staging buffer for this simdgroup
/// @param k_block          K offset for the block to prefetch
/// @param tg_col           Threadgroup column offset
/// @param sg_col_offset    Simdgroup column offset within threadgroup
/// @param tiles_n          Total number of N tiles in matrix
/// @param packed_bytes     Bytes per trellis tile
/// @param simd_lane_id     Lane within simdgroup [0, 31]
/// @param K                Total K dimension
inline void prefetch_weights_cooperative(
    device const uchar* packed_indices,
    threadgroup uchar* prefetch_buf,
    uint k_block,
    uint tg_col,
    uint sg_col_offset,
    uint tiles_n,
    uint packed_bytes,
    uint simd_lane_id,
    uint K
) {
    // Skip if beyond K bounds
    if (k_block >= K) return;

    uint trellis_tile_k_base = k_block / TRELLIS_TILE_DIM;
    uint n_col_base = tg_col + sg_col_offset;
    uint trellis_tile_n_base = n_col_base / TRELLIS_TILE_DIM;

    // Calculate total bytes to prefetch and distribute across 32 lanes
    uint total_bytes = PREFETCH_TILES_K * PREFETCH_N_TILES * packed_bytes;
    uint bytes_per_lane = (total_bytes + 31) / 32;

    #pragma unroll
    for (uint i = 0; i < PREFETCH_MAX_BYTES_PER_LANE; ++i) {
        uint buf_idx = simd_lane_id * bytes_per_lane + i;
        if (i >= bytes_per_lane || buf_idx >= total_bytes) break;

        // Decode which tile and byte this maps to
        uint tile_idx = buf_idx / packed_bytes;
        uint byte_in_tile = buf_idx % packed_bytes;

        uint tile_k_offset = tile_idx / PREFETCH_N_TILES;
        uint tile_n_offset = tile_idx % PREFETCH_N_TILES;

        uint actual_tile_k = trellis_tile_k_base + tile_k_offset;
        uint actual_tile_n = trellis_tile_n_base + tile_n_offset;

        // Bounds check and load
        if (actual_tile_k * TRELLIS_TILE_DIM < K && actual_tile_n < tiles_n) {
            uint tile_offset = (actual_tile_k * tiles_n + actual_tile_n) * packed_bytes;
            prefetch_buf[buf_idx] = packed_indices[tile_offset + byte_in_tile];
        } else {
            prefetch_buf[buf_idx] = 0;
        }
    }
}

/// Unpack trellis index from prefetched threadgroup memory.
/// Same logic as device memory version but reads from threadgroup buffer.
/// @param packed        Threadgroup buffer containing packed indices
/// @param base_offset   Byte offset into the buffer for this tile
/// @param idx_in_tile   Index within the 256-element trellis tile
/// @param bits          Quantization bit width (2, 3, or 4)
inline uint unpack_trellis_index_tg(
    threadgroup uchar* packed,
    uint base_offset,
    uint idx_in_tile,
    uint bits
) {
    switch (bits) {
        case 2: {
            uint byte_idx = idx_in_tile >> 2;
            uint bit_offset = (idx_in_tile & 3) << 1;
            return (packed[base_offset + byte_idx] >> bit_offset) & 0x3;
        }
        case 3: {
            uint bit_offset = idx_in_tile * 3;
            uint byte_idx = bit_offset >> 3;
            uint bit_in_byte = bit_offset & 7;
            uint packed_val = uint(packed[base_offset + byte_idx]);
            if (bit_in_byte + 3 > 8) {
                packed_val |= uint(packed[base_offset + byte_idx + 1]) << 8;
            }
            return (packed_val >> bit_in_byte) & 0x7;
        }
        case 4: {
            uint byte_idx = idx_in_tile >> 1;
            uint shift = (idx_in_tile & 1) << 2;
            return (packed[base_offset + byte_idx] >> shift) & 0xF;
        }
        default: {
            // Default to 3-bit
            uint bit_offset = idx_in_tile * 3;
            uint byte_idx = bit_offset >> 3;
            uint bit_in_byte = bit_offset & 7;
            uint packed_val = uint(packed[base_offset + byte_idx]);
            if (bit_in_byte + 3 > 8) {
                packed_val |= uint(packed[base_offset + byte_idx + 1]) << 8;
            }
            return (packed_val >> bit_in_byte) & 0x7;
        }
    }
}

/// Dequantize using prefetched data from threadgroup memory.
/// @param prefetch_buf     Threadgroup buffer containing prefetched packed indices
/// @param tile_k_offset    Which K-tile within the prefetch buffer [0, PREFETCH_TILES_K)
/// @param tile_n_offset    Which N-tile within the prefetch buffer [0, PREFETCH_N_TILES)
/// @param local_k          K index within the 16x16 trellis tile [0, 15]
/// @param local_n          N index within the 16x16 trellis tile [0, 15]
/// @param combined_scale   Precomputed scale * su * sv
/// @param grid             Codebook grid values
/// @param packed_bytes     Bytes per trellis tile
/// @param bits             Quantization bit width
/// @param n_levels         Number of codebook levels
inline half dequant_from_prefetch(
    threadgroup uchar* prefetch_buf,
    uint tile_k_offset,
    uint tile_n_offset,
    uint local_k,
    uint local_n,
    float combined_scale,
    constant float* grid,
    uint packed_bytes,
    uint bits,
    uint n_levels
) {
    // Calculate offset into prefetch buffer
    uint tile_idx = tile_k_offset * PREFETCH_N_TILES + tile_n_offset;
    uint buf_offset = tile_idx * packed_bytes;

    // Index within the 16x16 trellis tile (transposed: column-major storage)
    uint idx_in_tile = local_n * TRELLIS_TILE_DIM + local_k;

    uint trellis_idx = unpack_trellis_index_tg(prefetch_buf, buf_offset, idx_in_tile, bits);
    if (trellis_idx >= n_levels) trellis_idx = 0;

    return half(grid[trellis_idx] * combined_scale);
}

// ============================================================================
// Dequantization Primitives
// ============================================================================

/// Dequantize a single trellis element using precomputed combined scale.
/// Formula: dequant = grid[idx] * combined_scale
/// where combined_scale = scale * su * sv (precomputed by caller)
///
/// This fused variant reduces 3 multiplications to 1 multiply instruction.
/// The caller precomputes combined_scale once per element/row, avoiding
/// redundant su*sv computation when scale is shared across elements.
///
/// Note: We use a simple multiply here since the add term is 0. The FMA
/// instruction (fused multiply-add) is appropriate when accumulating:
///   result = fma(a, b, accumulator)
/// For dequant where we just need a*b, a plain multiply is more efficient.
inline half dequant_trellis_element_fused(
    uint trellis_idx,
    float combined_scale,
    constant float* grid
) {
    // Single multiply - combined_scale already incorporates scale * su * sv
    return half(grid[trellis_idx] * combined_scale);
}

/// Dequantize a single trellis element (legacy interface).
/// Formula: dequant = grid[idx] * scale * su * sv
/// Prefer dequant_trellis_element_fused when scale factors can be precomputed.
inline half dequant_trellis_element(
    uint idx,
    float scale,
    float su,
    float sv,
    constant float* grid
) {
    // Combine all scale factors in one expression
    // Compiler will optimize the multiplication chain
    return half(grid[idx] * (scale * su * sv));
}

// ============================================================================
// Texture-Based Dequantization Primitives
// ============================================================================
//
// These variants use Metal texture sampling for codebook lookup. Benefits:
// - Texture cache is optimized for repeated small lookups (perfect for 4-16 value codebook)
// - Hardware bounds checking via clamp-to-edge addressing
// - Potentially better L1 cache utilization for tiny arrays
//
// ============================================================================

/// Dequantize a single trellis element using texture-based grid lookup.
/// Formula: dequant = grid_tex[idx] * combined_scale
/// where combined_scale = scale * su * sv (precomputed by caller)
///
/// Uses hardware texture sampling for better cache utilization on small codebooks.
///
/// @param idx             Codebook index [0, n_levels-1]
/// @param combined_scale  Precomputed scale * su * sv
/// @param grid_tex        Codebook as 1D texture [n_levels] float32
/// @return                Dequantized FP16 value
inline half dequant_trellis_element_tex(
    uint idx,
    float combined_scale,
    texture1d<float, access::sample> grid_tex
) {
    // Texture sample returns float4, we only use .x component
    // Using pixel coordinates (0, 1, 2, ...) not normalized (0.0-1.0)
    float grid_val = grid_tex.sample(grid_sampler, float(idx)).x;
    return half(grid_val * combined_scale);
}

/// Dequantize 8 consecutive trellis elements for a single column.
/// Used during GEMM to load and dequant a column of B on-the-fly.
///
/// @param packed        Packed indices for the trellis tile
/// @param local_n       Column index within the 16x16 trellis tile [0, 15]
/// @param scale         Per-column scale factor
/// @param su_vec        Row signs for 8 consecutive K values
/// @param sv            Column sign
/// @param grid          Codebook grid
/// @param out           Output buffer for 8 dequantized values
/// @param bits          Quantization bit width (2, 3, or 4)
inline void dequant_trellis_column_8(
    device const uchar* packed,
    uint local_n,
    float scale,
    thread const float* su_vec,
    float sv,
    constant float* grid,
    thread half* out,
    uint bits
) {
    // OPTIMIZATION: Precompute scale * sv once per column
    // This reduces the inner loop from 3 multiplies to 1 multiply per element
    float scale_sv = scale * sv;

    // Each column has 16 elements in the trellis tile
    // We process 8 at a time (called twice per tile)
    #pragma unroll
    for (uint row = 0; row < 8; ++row) {
        uint idx_in_tile = row * TRELLIS_TILE_DIM + local_n;
        uint trellis_idx = unpack_trellis_index(packed, idx_in_tile, bits);
        // Fused: grid[idx] * (scale * sv * su[row])
        float combined = scale_sv * su_vec[row];
        out[row] = dequant_trellis_element_fused(trellis_idx, combined, grid);
    }
}

// ============================================================================
// Cooperative A Tile Loader (all 128 threads)
// ============================================================================

/// Load A tile from global memory to threadgroup memory.
/// Each thread loads TILE_M * TILE_K / THREADS_PER_TG elements.
inline void load_A_tile_cooperative(
    device const half* A,
    threadgroup half (&A_buf)[TILE_M][TILE_K],
    uint M, uint K,
    uint tg_row, uint k_block,
    uint thread_idx
) {
    constexpr uint ELEMS_PER_THREAD = (TILE_M * TILE_K) / THREADS_PER_TG;  // 32
    
    #pragma unroll
    for (uint i = 0; i < ELEMS_PER_THREAD; ++i) {
        uint flat_idx = thread_idx * ELEMS_PER_THREAD + i;
        uint row = flat_idx / TILE_K;
        uint col = flat_idx % TILE_K;
        uint global_row = tg_row + row;
        uint global_col = k_block + col;
        
        half val = half(0.0h);
        if (global_row < M && global_col < K) {
            val = A[global_row * K + global_col];
        }
        A_buf[row][col] = val;
    }
}

// ============================================================================
// Main Fused GEMM Kernel: gemm_trellis_packed
// ============================================================================

/// Fused dequant + GEMM for trellis-quantized weights.
///
/// Computes C[M,N] = A[M,K] @ dequant(W[K,N]) where W is trellis-quantized.
/// Weights are dequantized on-the-fly during the GEMM computation.
///
/// @param A               Input activations [M, K] half (row-major)
/// @param packed_indices  Packed trellis indices [tiles_k, tiles_n, packed_bytes] uint8
/// @param scales          Per-group scales [K/group_size, N] float32
/// @param grid            Codebook grid values [n_levels] float32
/// @param su              Row signs [K] float32
/// @param sv              Column signs [N] float32
/// @param C               Output matrix [M, N] half (row-major)
/// @param M               Number of rows in A and C
/// @param K               Number of columns in A / rows in W
/// @param N               Number of columns in W and C
/// @param bits            Quantization bit width (2, 3, or 4)
/// @param n_levels        Number of codebook levels (2^bits)
/// @param group_size      Quantization group size (typically 32 or 64)
kernel void gemm_trellis_packed(
    device const half* A               [[buffer(0)]],
    device const uchar* packed_indices [[buffer(1)]],
    constant float* scales         [[buffer(2)]],
    constant float* grid           [[buffer(3)]],
    constant float* su             [[buffer(4)]],
    constant float* sv             [[buffer(5)]],
    device half* C                     [[buffer(6)]],
    constant uint& M                   [[buffer(7)]],
    constant uint& K                   [[buffer(8)]],
    constant uint& N                   [[buffer(9)]],
    constant uint& bits                [[buffer(10)]],
    constant uint& n_levels            [[buffer(11)]],
    constant uint& group_size          [[buffer(12)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint3 tid                          [[thread_position_in_threadgroup]],
    uint simd_lane_id                  [[thread_index_in_simdgroup]],
    uint simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    // -------------------------------------------------------------------------
    // Threadgroup memory allocation
    // -------------------------------------------------------------------------
    // Double-buffered A tiles: 2 * 128 * 32 * 2 bytes = 16KB
    // Per-simdgroup B staging: 4 * 8 * 8 * 2 bytes = 512B
    // Double-buffered weight prefetch: 2 * 4 * 512 bytes = 4KB (per simdgroup: 2*2*128=512B)
    // Codebook cache: 16 * 4 bytes = 64B (tiny overhead for huge cache hit benefit)
    // Total: ~20.6KB (well within 192KB budget on M3 Max)
    threadgroup half A_tiles[2][TILE_M][TILE_K];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];
    threadgroup uchar weight_prefetch[2][SIMDGROUPS_PER_TG][PREFETCH_BUF_SIZE_PER_SG];
    threadgroup float grid_cache[16];  // Max 16 levels for 4-bit quantization
    
    // -------------------------------------------------------------------------
    // Tile assignment
    // -------------------------------------------------------------------------
    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;
    
    // Early exit if this threadgroup is outside M bounds
    if (tg_row >= M) return;
    
    // Simdgroup layout: 2x2 grid covering the 64x64 output tile
    // SG 0: rows [0,31],  cols [0,31]   (simd_id=0)
    // SG 1: rows [0,31],  cols [32,63]  (simd_id=1)
    // SG 2: rows [32,63], cols [0,31]   (simd_id=2)
    // SG 3: rows [32,63], cols [32,63]  (simd_id=3)
    const uint sg_row_offset = (simd_group_id / 2) * 32;
    const uint sg_col_offset = (simd_group_id % 2) * 32;
    
    // -------------------------------------------------------------------------
    // Initialize accumulators
    // Each simdgroup accumulates a 32x32 output tile using 8x8 MMA blocks
    // -------------------------------------------------------------------------
    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0h));
        }
    }
    
    const uint thread_idx = simd_group_id * 32 + simd_lane_id;
    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;
    const uint tiles_n = (N + TRELLIS_TILE_DIM - 1) / TRELLIS_TILE_DIM;
    const uint packed_bytes = packed_bytes_per_trellis_tile(bits);
    
    uint buf_compute = 0;
    uint weight_buf_compute = 0;
    
    // -------------------------------------------------------------------------
    // Prologue: Load codebook, first A tile, and prefetch first weight tile
    // -------------------------------------------------------------------------

    // Cooperative codebook load: first n_levels threads load the grid
    // This is a one-time cost amortized over all K-tiles
    if (thread_idx < n_levels) {
        grid_cache[thread_idx] = grid[thread_idx];
    }

    load_A_tile_cooperative(A, A_tiles[0], M, K, tg_row, 0, thread_idx);

    // Prefetch weights for first K-block
    prefetch_weights_cooperative(
        packed_indices,
        weight_prefetch[0][simd_group_id],
        0,  // k_block = 0
        tg_col,
        sg_col_offset,
        tiles_n,
        packed_bytes,
        simd_lane_id,
        K
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // -------------------------------------------------------------------------
    // Main pipeline loop - Double buffered (A tiles and weights)
    // -------------------------------------------------------------------------
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;
        uint next_k = k_block + TILE_K;
        uint buf_load = 1 - buf_compute;
        uint weight_buf_load = 1 - weight_buf_compute;
        
        // --- Async load next A tile and prefetch next weights while computing ---
        if (next_k < K) {
            load_A_tile_cooperative(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
            
            // Prefetch weights for next K-block
            prefetch_weights_cooperative(
                packed_indices,
                weight_prefetch[weight_buf_load][simd_group_id],
                next_k,
                tg_col,
                sg_col_offset,
                tiles_n,
                packed_bytes,
                simd_lane_id,
                K
            );
        }
        
        // --- Compute: Fused B dequant + MMA ---
        // For each K sub-tile (8 elements), we:
        //   1. Load A fragments from threadgroup memory (reused across N)
        //   2. Dequant B on-the-fly for each N sub-tile
        //   3. Perform MMA accumulation
        
        #pragma unroll
        for (uint kk = 0; kk < K_TILES; ++kk) {
            uint k_sub_base = k_block + kk * 8;
            uint group_idx = k_sub_base / group_size;
            
            // Calculate trellis tile coordinates for this K sub-tile
            uint trellis_tile_k = k_sub_base / TRELLIS_TILE_DIM;
            uint local_k_base = k_sub_base % TRELLIS_TILE_DIM;
            
            // Load A fragments for this K sub-tile (reused across all N tiles)
            simdgroup_matrix<half, 8, 8> a_frag[SG_M_TILES];
            #pragma unroll
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                simdgroup_load(
                    a_frag[mi],
                    &A_tiles[buf_compute][sg_row_offset + mi * 8][kk * 8],
                    TILE_K
                );
            }
            
            // ILP OPTIMIZATION: Prefetch sv values before the N-tile loop
            // These are reused across all N tiles, so load once and cache in registers
            float sv_cached[8];
            #pragma unroll
            for (uint row = 0; row < 8; ++row) {
                uint k_idx = k_sub_base + row;
                sv_cached[row] = (k_idx < K) ? sv[k_idx] : 0.0f;
            }

            // For each N sub-tile, dequant B and accumulate
            #pragma unroll
            for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                uint b_col_base = tg_col + sg_col_offset + ni * 8;

                // --- Fused B dequantization with ILP optimizations ---
                // Lanes 0-7 each handle one column of the 8x8 B tile
                if (simd_lane_id < 8) {
                    uint b_col = b_col_base + simd_lane_id;
                    half dequant_vals[8];

                    if (b_col < N) {
                        // ILP OPTIMIZATION: Load all memory-bound values first
                        // This allows memory ops to start while we compute indices
                        uint trellis_tile_n = b_col / TRELLIS_TILE_DIM;
                        uint local_n = b_col % TRELLIS_TILE_DIM;
                        uint scale_idx = group_idx * N + b_col;

                        // Start memory loads early
                        float scale = scales[scale_idx];
                        float row_sign = su[b_col];
                        float scale_row = scale * row_sign;

                        // ILP OPTIMIZATION: Batch compute all combined scales using cached sv
                        float combined_scale[8];
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            combined_scale[row] = scale_row * sv_cached[row];
                        }

                        // ILP OPTIMIZATION: Unpack indices in pairs to interleave with grid lookups
                        // Process 2 rows at a time to hide memory latency
                        #pragma unroll(4)
                        for (uint row = 0; row < 8; row += 2) {
                            uint k_idx_0 = k_sub_base + row;
                            uint k_idx_1 = k_sub_base + row + 1;

                            // Compute tile offsets for both rows
                            uint actual_tile_k_0 = k_idx_0 / TRELLIS_TILE_DIM;
                            uint actual_tile_k_1 = k_idx_1 / TRELLIS_TILE_DIM;
                            uint local_k_0 = (local_k_base + row) % TRELLIS_TILE_DIM;
                            uint local_k_1 = (local_k_base + row + 1) % TRELLIS_TILE_DIM;

                            uint actual_tile_offset_0 = (actual_tile_k_0 * tiles_n + trellis_tile_n) * packed_bytes;
                            uint actual_tile_offset_1 = (actual_tile_k_1 * tiles_n + trellis_tile_n) * packed_bytes;

                            uint idx_in_tile_0 = local_n * TRELLIS_TILE_DIM + local_k_0;
                            uint idx_in_tile_1 = local_n * TRELLIS_TILE_DIM + local_k_1;

                            // Unpack both indices (interleaved memory access)
                            uint trellis_idx_0 = (k_idx_0 < K) ? unpack_trellis_index(
                                packed_indices + actual_tile_offset_0, idx_in_tile_0, bits) : 0;
                            uint trellis_idx_1 = (k_idx_1 < K) ? unpack_trellis_index(
                                packed_indices + actual_tile_offset_1, idx_in_tile_1, bits) : 0;

                            // Bounds check
                            if (trellis_idx_0 >= n_levels) trellis_idx_0 = 0;
                            if (trellis_idx_1 >= n_levels) trellis_idx_1 = 0;

                            // Grid lookups from threadgroup cache - interleaved
                            // Using grid_cache avoids repeated device memory accesses
                            dequant_vals[row] = (k_idx_0 < K) ?
                                half(grid_cache[trellis_idx_0] * combined_scale[row]) : half(0.0h);
                            dequant_vals[row + 1] = (k_idx_1 < K) ?
                                half(grid_cache[trellis_idx_1] * combined_scale[row + 1]) : half(0.0h);
                        }
                    } else {
                        // Out of bounds - fill with zeros
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            dequant_vals[row] = half(0.0h);
                        }
                    }

                    // Write dequantized values to staging buffer
                    // Layout: B_staging[simd_id][k][n] where k,n in [0,7]
                    #pragma unroll
                    for (uint row = 0; row < 8; ++row) {
                        B_staging[simd_group_id][row][simd_lane_id] = dequant_vals[row];
                    }
                }

                // Synchronize within simdgroup (lightweight barrier)
                simdgroup_barrier(mem_flags::mem_threadgroup);

                // Load B fragment and perform MMA
                simdgroup_matrix<half, 8, 8> b_frag;
                simdgroup_load(b_frag, &B_staging[simd_group_id][0][0], 8);

                #pragma unroll
                for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                    // simdgroup_multiply_accumulate is the core SIMD intrinsic
                    // that maps 8x8 tiles to Apple Silicon's matrix pipeline.
                    // It performs: acc += A * B where A,B are 8x8 and acc accumulates.
                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag[mi], b_frag, acc[mi][ni]);
                }
            }
        }
        
        // Wait for next tile load before swapping buffers
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }
    
    // -------------------------------------------------------------------------
    // Epilogue: Store results to global memory
    // -------------------------------------------------------------------------
    // Per-simdgroup staging to avoid race conditions between simdgroups
    threadgroup half epilogue_staging[SIMDGROUPS_PER_TG][8][8];
    
    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        uint out_row = tg_row + sg_row_offset + mi * 8;
        if (out_row >= M) continue;
        
        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_col = tg_col + sg_col_offset + ni * 8;
            
            // Fast path: full 8x8 tile within bounds -> direct simdgroup_store
            if (out_row + 8 <= M && out_col + 8 <= N) {
                simdgroup_store(acc[mi][ni], C + out_row * N + out_col, N);
                continue;
            }
            
            // Slow path: partial tile or bounds checking needed
            simdgroup_store(acc[mi][ni], &epilogue_staging[simd_group_id][0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            
            // Each thread handles 2 elements (32 threads * 2 = 64 elements)
            for (uint elem = simd_lane_id; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                uint gr = out_row + r;
                uint gc = out_col + c;
                
                if (gr < M && gc < N) {
                    C[gr * N + gc] = epilogue_staging[simd_group_id][r][c];
                }
            }
            
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ============================================================================
// FP32 Accumulation Variant (gemm_trellis_packed_fp32acc)
// ============================================================================

/// FP32 accumulation variant for numerical stability with large K.
/// Same algorithm as above but uses FP32 accumulators internally.
kernel void gemm_trellis_packed_fp32acc(
    device const half* A               [[buffer(0)]],
    device const uchar* packed_indices [[buffer(1)]],
    constant float* scales         [[buffer(2)]],
    constant float* grid           [[buffer(3)]],
    constant float* su             [[buffer(4)]],
    constant float* sv             [[buffer(5)]],
    device half* C                     [[buffer(6)]],
    constant uint& M                   [[buffer(7)]],
    constant uint& K                   [[buffer(8)]],
    constant uint& N                   [[buffer(9)]],
    constant uint& bits                [[buffer(10)]],
    constant uint& n_levels            [[buffer(11)]],
    constant uint& group_size          [[buffer(12)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint3 tid                          [[thread_position_in_threadgroup]],
    uint simd_lane_id                  [[thread_index_in_simdgroup]],
    uint simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half A_tiles[2][TILE_M][TILE_K];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];
    threadgroup float epilogue_staging_fp32[8][8];
    
    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;
    
    if (tg_row >= M) return;
    
    const uint sg_row_offset = (simd_group_id / 2) * 32;
    const uint sg_col_offset = (simd_group_id % 2) * 32;
    
    // FP32 accumulators for numerical stability
    simdgroup_matrix<float, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    
    const uint thread_idx = simd_group_id * 32 + simd_lane_id;
    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;
    const uint tiles_n = (N + TRELLIS_TILE_DIM - 1) / TRELLIS_TILE_DIM;
    const uint packed_bytes = packed_bytes_per_trellis_tile(bits);
    
    uint buf_compute = 0;
    
    load_A_tile_cooperative(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;
        uint next_k = k_block + TILE_K;
        uint buf_load = 1 - buf_compute;
        
        if (next_k < K) {
            load_A_tile_cooperative(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
        }
        
        #pragma unroll
        for (uint kk = 0; kk < K_TILES; ++kk) {
            uint k_sub_base = k_block + kk * 8;
            uint group_idx = k_sub_base / group_size;
            uint trellis_tile_k = k_sub_base / TRELLIS_TILE_DIM;
            uint local_k_base = k_sub_base % TRELLIS_TILE_DIM;

            // ILP OPTIMIZATION: Prefetch su values before the N-tile loop
            float su_cached[8];
            #pragma unroll
            for (uint row = 0; row < 8; ++row) {
                uint k_idx = k_sub_base + row;
                su_cached[row] = (k_idx < K) ? su[k_idx] : 0.0f;
            }

            simdgroup_matrix<half, 8, 8> a_frag[SG_M_TILES];
            #pragma unroll
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                simdgroup_load(
                    a_frag[mi],
                    &A_tiles[buf_compute][sg_row_offset + mi * 8][kk * 8],
                    TILE_K
                );
            }

            #pragma unroll
            for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                uint b_col_base = tg_col + sg_col_offset + ni * 8;

                if (simd_lane_id < 8) {
                    uint b_col = b_col_base + simd_lane_id;
                    half dequant_vals[8];

                    if (b_col < N) {
                        // ILP OPTIMIZATION: Load memory-bound values first
                        uint trellis_tile_n = b_col / TRELLIS_TILE_DIM;
                        uint local_n = b_col % TRELLIS_TILE_DIM;
                        uint scale_idx = group_idx * N + b_col;

                        float scale = scales[scale_idx];
                        float sign_n = sv[b_col];
                        float scale_sign_n = scale * sign_n;

                        // ILP OPTIMIZATION: Batch compute combined scales using cached su
                        float combined_scale[8];
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            combined_scale[row] = scale_sign_n * su_cached[row];
                        }

                        // ILP OPTIMIZATION: Process 2 rows at a time for interleaved memory access
                        #pragma unroll(4)
                        for (uint row = 0; row < 8; row += 2) {
                            uint k_idx_0 = k_sub_base + row;
                            uint k_idx_1 = k_sub_base + row + 1;
                            uint actual_tile_k_0 = k_idx_0 / TRELLIS_TILE_DIM;
                            uint actual_tile_k_1 = k_idx_1 / TRELLIS_TILE_DIM;
                            uint local_k_0 = (local_k_base + row) % TRELLIS_TILE_DIM;
                            uint local_k_1 = (local_k_base + row + 1) % TRELLIS_TILE_DIM;

                            uint actual_tile_offset_0 = (actual_tile_k_0 * tiles_n + trellis_tile_n) * packed_bytes;
                            uint actual_tile_offset_1 = (actual_tile_k_1 * tiles_n + trellis_tile_n) * packed_bytes;
                            uint idx_in_tile_0 = local_n * TRELLIS_TILE_DIM + local_k_0;
                            uint idx_in_tile_1 = local_n * TRELLIS_TILE_DIM + local_k_1;

                            uint trellis_idx_0 = (k_idx_0 < K) ? unpack_trellis_index(
                                packed_indices + actual_tile_offset_0, idx_in_tile_0, bits) : 0;
                            uint trellis_idx_1 = (k_idx_1 < K) ? unpack_trellis_index(
                                packed_indices + actual_tile_offset_1, idx_in_tile_1, bits) : 0;

                            if (trellis_idx_0 >= n_levels) trellis_idx_0 = 0;
                            if (trellis_idx_1 >= n_levels) trellis_idx_1 = 0;

                            dequant_vals[row] = (k_idx_0 < K) ?
                                dequant_trellis_element_fused(trellis_idx_0, combined_scale[row], grid) : half(0.0h);
                            dequant_vals[row + 1] = (k_idx_1 < K) ?
                                dequant_trellis_element_fused(trellis_idx_1, combined_scale[row + 1], grid) : half(0.0h);
                        }
                    } else {
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            dequant_vals[row] = half(0.0h);
                        }
                    }

                    #pragma unroll
                    for (uint row = 0; row < 8; ++row) {
                        B_staging[simd_group_id][row][simd_lane_id] = dequant_vals[row];
                    }
                }

                simdgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_matrix<half, 8, 8> b_frag;
                simdgroup_load(b_frag, &B_staging[simd_group_id][0][0], 8);

                // FP32 accumulation (mixed precision)
                #pragma unroll
                for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag[mi], b_frag, acc[mi][ni]);
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }
    
    // Store with FP32 -> FP16 conversion
    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        uint out_row = tg_row + sg_row_offset + mi * 8;
        if (out_row >= M) continue;
        
        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_col = tg_col + sg_col_offset + ni * 8;
            
            simdgroup_store(acc[mi][ni], &epilogue_staging_fp32[0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint elem = simd_lane_id; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                uint gr = out_row + r;
                uint gc = out_col + c;
                
                if (gr < M && gc < N) {
                    C[gr * N + gc] = half(epilogue_staging_fp32[r][c]);
                }
            }
            
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ============================================================================
// Decode-Optimized Variant (gemm_trellis_packed_decode)
// ============================================================================

/// Optimized variant for small M (autoregressive decode, M=1-16).
/// Uses 32x128 tiles to maximize N coverage per threadgroup.

constant constexpr uint DECODE_TILE_M = 32;
constant constexpr uint DECODE_TILE_N = 128;
constant constexpr uint DECODE_K_TILES = TILE_K / 8;  // 4

kernel void gemm_trellis_packed_decode(
    device const half* A               [[buffer(0)]],
    device const uchar* packed_indices [[buffer(1)]],
    constant float* scales         [[buffer(2)]],
    constant float* grid           [[buffer(3)]],
    constant float* su             [[buffer(4)]],
    constant float* sv             [[buffer(5)]],
    device half* C                     [[buffer(6)]],
    constant uint& M                   [[buffer(7)]],
    constant uint& K                   [[buffer(8)]],
    constant uint& N                   [[buffer(9)]],
    constant uint& bits                [[buffer(10)]],
    constant uint& n_levels            [[buffer(11)]],
    constant uint& group_size          [[buffer(12)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint3 tid                          [[thread_position_in_threadgroup]],
    uint simd_lane_id                  [[thread_index_in_simdgroup]],
    uint simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    // 32x128x32 tiles: A = 2*32*32*2 = 4KB, B_staging = 512B
    threadgroup half A_tiles[2][DECODE_TILE_M][TILE_K];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];
    
    const uint tg_row = tgid.y * DECODE_TILE_M;
    const uint tg_col = tgid.x * DECODE_TILE_N;
    
    if (tg_row >= M) return;
    
    // 4 simdgroups tile 32x128 as 1x4: each handles 32x32
    const uint sg_row_offset = 0;
    const uint sg_col_offset = simd_group_id * 32;
    
    // Each simdgroup handles 4x4 = 16 sub-tiles of 8x8 within its 32x32 region
    constexpr uint DECODE_SG_M_TILES = 4;   // 32 rows / 8 = 4
    constexpr uint DECODE_SG_N_TILES = 4;   // 32 cols / 8 = 4
    
    simdgroup_matrix<half, 8, 8> acc[DECODE_SG_M_TILES][DECODE_SG_N_TILES];
    #pragma unroll
    for (uint mi = 0; mi < DECODE_SG_M_TILES; ++mi) {
        #pragma unroll
        for (uint ni = 0; ni < DECODE_SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0h));
        }
    }
    
    const uint thread_idx = simd_group_id * 32 + simd_lane_id;
    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;
    const uint tiles_n = (N + TRELLIS_TILE_DIM - 1) / TRELLIS_TILE_DIM;
    const uint packed_bytes = packed_bytes_per_trellis_tile(bits);
    
    // A tile load: 32 * 32 = 1024 / 128 = 8 elements per thread
    constexpr uint A_ELEMS = (DECODE_TILE_M * TILE_K) / THREADS_PER_TG;
    
    // Prologue: Load first A tile
    #pragma unroll
    for (uint i = 0; i < A_ELEMS; ++i) {
        uint flat_idx = thread_idx * A_ELEMS + i;
        uint row = flat_idx / TILE_K;
        uint col = flat_idx % TILE_K;
        uint global_row = tg_row + row;
        half val = (global_row < M && col < K) ? A[global_row * K + col] : half(0.0h);
        A_tiles[0][row][col] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint buf_compute = 0;
    
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;
        uint next_k = k_block + TILE_K;
        uint buf_load = 1 - buf_compute;
        
        // Load next A tile
        if (next_k < K) {
            #pragma unroll
            for (uint i = 0; i < A_ELEMS; ++i) {
                uint flat_idx = thread_idx * A_ELEMS + i;
                uint row = flat_idx / TILE_K;
                uint col = flat_idx % TILE_K;
                uint global_row = tg_row + row;
                uint global_col = next_k + col;
                half val = (global_row < M && global_col < K) ? A[global_row * K + global_col] : half(0.0h);
                A_tiles[buf_load][row][col] = val;
            }
        }
        
        // Compute
        #pragma unroll
        for (uint kk = 0; kk < DECODE_K_TILES; ++kk) {
            uint k_sub_base = k_block + kk * 8;
            uint group_idx = k_sub_base / group_size;
            
            simdgroup_matrix<half, 8, 8> a_frag[DECODE_SG_M_TILES];
            #pragma unroll
            for (uint mi = 0; mi < DECODE_SG_M_TILES; ++mi) {
                simdgroup_load(a_frag[mi], &A_tiles[buf_compute][mi * 8][kk * 8], TILE_K);
            }
            
            #pragma unroll
            for (uint ni = 0; ni < DECODE_SG_N_TILES; ++ni) {
                uint b_col_base = tg_col + sg_col_offset + ni * 8;
                
                if (simd_lane_id < 8) {
                    uint b_col = b_col_base + simd_lane_id;
                    half dequant_vals[8];
                    
                    if (b_col < N) {
                        uint trellis_tile_k = k_sub_base / TRELLIS_TILE_DIM;
                        uint local_k_base = k_sub_base % TRELLIS_TILE_DIM;
                        uint trellis_tile_n = b_col / TRELLIS_TILE_DIM;
                        uint local_n = b_col % TRELLIS_TILE_DIM;

                        uint tile_offset = (trellis_tile_k * tiles_n + trellis_tile_n) * packed_bytes;

                        uint scale_idx = group_idx * N + b_col;
                        float scale = scales[scale_idx];
                        float sign_n = sv[b_col];

                        // OPTIMIZATION: Precompute scale * sign_n once per column
                        // This reduces the inner loop from 3 multiplies to 1 multiply
                        float scale_sign_n = scale * sign_n;

                        // Precompute combined scale factors for all 8 rows
                        // combined_scale[i] = scale * sign_n * su[k_sub_base + i]
                        float combined_scale[8];
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            uint k_idx = k_sub_base + row;
                            float su_val = (k_idx < K) ? su[k_idx] : 0.0f;
                            combined_scale[row] = scale_sign_n * su_val;
                        }

                        // Dequantize 8 elements using fused single multiply
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            uint k_idx = k_sub_base + row;
                            uint local_k = (local_k_base + row) % TRELLIS_TILE_DIM;

                            if (k_idx < K) {
                                uint actual_tile_k = k_idx / TRELLIS_TILE_DIM;
                                uint actual_tile_offset = (actual_tile_k * tiles_n + trellis_tile_n) * packed_bytes;
                                uint idx_in_tile = local_n * TRELLIS_TILE_DIM + local_k;  // Transposed weight

                                uint trellis_idx = unpack_trellis_index(
                                    packed_indices + actual_tile_offset, idx_in_tile, bits);
                                if (trellis_idx >= n_levels) trellis_idx = 0;

                                // Use fused dequant with precomputed combined scale
                                dequant_vals[row] = dequant_trellis_element_fused(
                                    trellis_idx, combined_scale[row], grid);
                            } else {
                                dequant_vals[row] = half(0.0h);
                            }
                        }
                    } else {
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            dequant_vals[row] = half(0.0h);
                        }
                    }
                    
                    #pragma unroll
                    for (uint row = 0; row < 8; ++row) {
                        B_staging[simd_group_id][row][simd_lane_id] = dequant_vals[row];
                    }
                }
                
                simdgroup_barrier(mem_flags::mem_threadgroup);
                
                simdgroup_matrix<half, 8, 8> b_frag;
                simdgroup_load(b_frag, &B_staging[simd_group_id][0][0], 8);
                
                #pragma unroll
                for (uint mi = 0; mi < DECODE_SG_M_TILES; ++mi) {
                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag[mi], b_frag, acc[mi][ni]);
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }
    
    // Store
    threadgroup half epilogue_staging[8][8];
    
    #pragma unroll
    for (uint mi = 0; mi < DECODE_SG_M_TILES; ++mi) {
        uint out_row = tg_row + mi * 8;
        if (out_row >= M) continue;
        
        #pragma unroll
        for (uint ni = 0; ni < DECODE_SG_N_TILES; ++ni) {
            uint out_col = tg_col + sg_col_offset + ni * 8;
            
            if (out_row + 8 <= M && out_col + 8 <= N) {
                simdgroup_store(acc[mi][ni], C + out_row * N + out_col, N);
                continue;
            }
            
            simdgroup_store(acc[mi][ni], &epilogue_staging[0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint elem = simd_lane_id; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                uint gr = out_row + r;
                uint gc = out_col + c;
                
                if (gr < M && gc < N) {
                    C[gr * N + gc] = epilogue_staging[r][c];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ============================================================================
// Fully-Fused Dequant+GEMM (gemm_trellis_fused_reg)
// ============================================================================
//
// This kernel fully fuses dequantization into GEMM inner loop.
// Dequantized values are kept in registers only - no staging buffer materialization.
//
// Key optimizations:
//   1. All 32 threads dequant in parallel (each handles 2 elements)
//   2. Dequantized values NEVER written to threadgroup memory
//   3. Manual scalar reduction for 8x8 tile multiplication
//   4. Register-only accumulation (simdgroup_matrix used for final storage)
//
// Thread-to-element mapping for 8x8 B tile (64 elements, 32 threads × 2):
//   Lane i computes partial results for elements [i] and [i+32]
//   Each element is dot product over K dimension
//
// Performance characteristics:
//   - Dequant throughput: 4x (32 threads vs 8)
//   - Memory bandwidth: Reduced (no staging buffer reads/writes)
//   - Register pressure: Higher (full tile in registers per thread)
//
// ============================================================================

/// Fully-fused dequant+accumulate: dequant and accumulate in registers.
/// No staging buffer - values kept in registers throughout.
inline void dequant_and_accumulate_2x(
    device const uchar* packed_indices,
    constant float* scales,
    constant float* grid,
    constant float* su,
    thread const float* sv_cached,
    uint tiles_n,
    uint packed_bytes,
    uint n_levels,
    uint bits,
    uint K,
    uint N,
    uint group_idx,
    uint k_sub_base,
    uint b_col_base,
    uint simd_lane_id,
    half a_vals[2][8],  // A[k][0-7] for this thread's rows
    thread float acc[8]     // Accumulators for 8 output columns
) {
    // For each output column, compute C[row][col] += sum_k(A[row][k] * B[k][col])
    #pragma unroll
    for (uint col = 0; col < 8; ++col) {
        uint b_col = b_col_base + col;
        if (b_col >= N) {
            acc[col] = 0.0f;
            continue;
        }

        float sum0 = 0.0f;
        float sum1 = 0.0f;

        // Inner K-loop: dequant B[k][col] on-the-fly for each k
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            uint k_idx = k_sub_base + k;
            if (k_idx >= K) continue;

            // Dequantize B[k][col] in registers
            uint trellis_tile_k = k_idx / TRELLIS_TILE_DIM;
            uint trellis_tile_n = b_col / TRELLIS_TILE_DIM;
            uint local_k = k_idx % TRELLIS_TILE_DIM;
            uint local_n = b_col % TRELLIS_TILE_DIM;
            uint tile_offset = (trellis_tile_k * tiles_n + trellis_tile_n) * packed_bytes;
            uint idx_in_tile = local_n * TRELLIS_TILE_DIM + local_k;
            uint trellis_idx = unpack_trellis_index(packed_indices + tile_offset, idx_in_tile, bits);
            if (trellis_idx >= n_levels) trellis_idx = 0;

            float scale = scales[group_idx * N + b_col];
            float row_sign = su[b_col];
            float col_sign = sv_cached[k];
            float b_val = grid[trellis_idx] * scale * row_sign * col_sign;

            // Accumulate immediately - B value stays in register only
            sum0 += float(a_vals[0][k]) * b_val;
            sum1 += float(a_vals[1][k]) * b_val;
        }

        acc[col] = sum0 + sum1;
    }
}



kernel void gemm_trellis_fused_reg(
    device const half* A               [[buffer(0)]],
    device const uchar* packed_indices [[buffer(1)]],
    constant float* scales         [[buffer(2)]],
    constant float* grid           [[buffer(3)]],
    constant float* su             [[buffer(4)]],
    constant float* sv             [[buffer(5)]],
    device half* C                     [[buffer(6)]],
    constant uint& M                   [[buffer(7)]],
    constant uint& K                   [[buffer(8)]],
    constant uint& N                   [[buffer(9)]],
    constant uint& bits                [[buffer(10)]],
    constant uint& n_levels            [[buffer(11)]],
    constant uint& group_size          [[buffer(12)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint3 tid                          [[thread_position_in_threadgroup]],
    uint simd_lane_id                  [[thread_index_in_simdgroup]],
    uint simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;

    if (tg_row >= M) return;

    const uint sg_row_offset = (simd_group_id / 2) * 32;
    const uint sg_col_offset = (simd_group_id % 2) * 32;

    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;
    const uint tiles_n = (N + TRELLIS_TILE_DIM - 1) / TRELLIS_TILE_DIM;
    const uint packed_bytes = packed_bytes_per_trellis_tile(bits);

    // -------------------------------------------------------------------------
    // Initialize accumulators in registers (no simdgroup_matrix yet)
    // Each thread accumulates for its assigned output tile
    // -------------------------------------------------------------------------
    thread float tile_acc[SG_M_TILES][SG_N_TILES][8];

    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            #pragma unroll
            for (uint col = 0; col < 8; ++col) {
                tile_acc[mi][ni][col] = 0.0f;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Main loop: Load A, dequant+accumulate B, all in registers
    // -------------------------------------------------------------------------
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;

        #pragma unroll
        for (uint kk = 0; kk < K_TILES; ++kk) {
            uint k_sub_base = k_block + kk * 8;
            uint group_idx = k_sub_base / group_size;

            // Preload sv values for this K sub-tile
            float sv_cached[8];
            {
                float my_sv = 0.0f;
                if (simd_lane_id < 8) {
                    uint k_idx = k_sub_base + simd_lane_id;
                    my_sv = (k_idx < K) ? sv[k_idx] : 0.0f;
                }
                #pragma unroll
                for (uint i = 0; i < 8; ++i) {
                    sv_cached[i] = simd_shuffle(my_sv, i);
                }
            }

            // Load A fragments for each M sub-tile
            #pragma unroll
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                uint a_row_base = sg_row_offset + mi * 8;

                half a_frag[2][8];
                #pragma unroll
                for (uint k = 0; k < 8; ++k) {
                    uint k_idx = k_sub_base + k;
                    #pragma unroll
                    for (uint row = 0; row < 2; ++row) {
                        uint global_row = tg_row + a_row_base + row;
                        if (global_row < M && k_idx < K) {
                            a_frag[row][k] = A[global_row * K + k_idx];
                        } else {
                            a_frag[row][k] = half(0.0h);
                        }
                    }
                }

                // Process each N sub-tile with fused dequant+accumulate
                #pragma unroll
                for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                    uint b_col_base = tg_col + sg_col_offset + ni * 8;

                    thread float col_acc[8];
                    dequant_and_accumulate_2x(
                        packed_indices, scales, grid, su, sv_cached,
                        tiles_n, packed_bytes, n_levels, bits, K, N,
                        group_idx, k_sub_base, b_col_base, simd_lane_id,
                        a_frag, col_acc
                    );

                    #pragma unroll
                    for (uint col = 0; col < 8; ++col) {
                        tile_acc[mi][ni][col] += col_acc[col];
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Epilogue: Store accumulated results to global memory
    // -------------------------------------------------------------------------
    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        uint out_row_base = tg_row + sg_row_offset + mi * 8;

        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_col_base = tg_col + sg_col_offset + ni * 8;

            // Write 8x8 output tile
            for (uint elem = simd_lane_id; elem < 64; elem += 32) {
                uint r = elem >> 3;
                uint c = elem & 7;
                uint global_row = out_row_base + r;
                uint global_col = out_col_base + c;

                if (global_row < M && global_col < N) {
                    C[global_row * N + global_col] = half(tile_acc[mi][ni][c]);
                }
            }
        }
    }
}

// ============================================================================
// INT8 Quantization Support for TrellisLinear
// ============================================================================

/// Fused INT8 dequant + GEMM kernel for TrellisLinear (W8A16 mode).
///
/// Computes C[M,N] = A[M,K] @ dequant_int8(W[K,N]) where W is INT8-quantized.
/// Weights are dequantized on-the-fly during the GEMM computation.
///
/// This kernel uses the same architecture as gemm_trellis_fused_reg but replaces
/// trellis codebook lookups with direct INT8 dequantization.
///
/// @param A           Input activations [M, K] half (row-major)
/// @param int8_weights  INT8 quantized weights [K, N] int8 (row-major, packed as uchar)
/// @param scales      Per-group scales [K/group_size, N] float32
/// @param zeros       Per-group zero points [K/group_size, N] float32 (nullable for symmetric)
/// @param su          Row signs [K] float32 (optional, pass nullptr to ignore)
/// @param sv          Column signs [N] float32 (optional, pass nullptr to ignore)
/// @param C           Output matrix [M, N] half (row-major)
/// @param M           Number of rows in A and C
/// @param K           Number of columns in A / rows in W
/// @param N           Number of columns in W and C
/// @param group_size  Quantization group size (typically 32 or 128)
/// @param asymmetric  0=symmetric quantization, 1=asymmetric (use zeros)
kernel void gemm_trellis_int8(
    device const half* A               [[buffer(0)]],
    device const uchar* int8_weights   [[buffer(1)]],
    constant float* scales             [[buffer(2)]],
    constant float* zeros              [[buffer(3)]],
    constant float* su                 [[buffer(4)]],
    constant float* sv                 [[buffer(5)]],
    device half* C                     [[buffer(6)]],
    constant uint& M                   [[buffer(7)]],
    constant uint& K                   [[buffer(8)]],
    constant uint& N                   [[buffer(9)]],
    constant uint& group_size          [[buffer(10)]],
    constant uint& asymmetric          [[buffer(11)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint simd_lane_id                  [[thread_index_in_simdgroup]],
    uint simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    // -------------------------------------------------------------------------
    // Tile assignment
    // -------------------------------------------------------------------------
    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;
    
    // Early exit if this threadgroup is outside M bounds
    if (tg_row >= M) return;
    
    // Simdgroup layout: 8 simdgroups tile the 128x128 output
    // SG 0-1: rows [0,31],   cols [0,63] and [64,127]
    // SG 2-3: rows [32,63],  cols [0,63] and [64,127]
    // SG 4-5: rows [64,95],  cols [0,63] and [64,127]
    // SG 6-7: rows [96,127], cols [0,63] and [64,127]
    const uint sg_row_offset = (simd_group_id / 2) * 32;
    const uint sg_col_offset = (simd_group_id % 2) * 64;
    
    // -------------------------------------------------------------------------
    // Initialize accumulators (FP32 for precision)
    // Each simdgroup accumulates 4x8 = 32 output columns
    // -------------------------------------------------------------------------
    thread float tile_acc[SG_M_TILES][SG_N_TILES][8];
    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            #pragma unroll
            for (uint col = 0; col < 8; ++col) {
                tile_acc[mi][ni][col] = 0.0f;
            }
        }
    }
    
    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;
    
    // -------------------------------------------------------------------------
    // Main K-loop: Process TILE_K elements at a time
    // -------------------------------------------------------------------------
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;
        
        #pragma unroll
        for (uint kk = 0; kk < K_TILES; ++kk) {
            uint k_sub_base = k_block + kk * 8;
            uint group_idx = k_sub_base / group_size;
            
            // Preload su values for this K sub-tile (row signs)
            float su_cached[8];
            {
                float my_su = 1.0f;  // Default to 1.0 if su is null
                if (simd_lane_id < 8 && su != nullptr) {
                    uint k_idx = k_sub_base + simd_lane_id;
                    my_su = (k_idx < K) ? su[k_idx] : 1.0f;
                }
                #pragma unroll
                for (uint i = 0; i < 8; ++i) {
                    su_cached[i] = simd_shuffle(my_su, i);
                }
            }
            
            // Load A fragments for each M sub-tile
            #pragma unroll
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                uint a_row_base = sg_row_offset + mi * 8;
                
                half a_frag[2][8];
                #pragma unroll
                for (uint k = 0; k < 8; ++k) {
                    uint k_idx = k_sub_base + k;
                    #pragma unroll
                    for (uint row = 0; row < 2; ++row) {
                        uint global_row = tg_row + a_row_base + row;
                        if (global_row < M && k_idx < K) {
                            a_frag[row][k] = A[global_row * K + k_idx];
                        } else {
                            a_frag[row][k] = half(0.0h);
                        }
                    }
                }
                
                // Process each N sub-tile with fused INT8 dequant+accumulate
                #pragma unroll
                for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                    uint b_col_base = tg_col + sg_col_offset + ni * 8;
                    
                    // Compute combined scale and zero point for each column
                    // combined_scale[col] = scale[group_idx, col] * sv[col] * su[k]
                    // We precompute scale[group_idx, col] * sv[col] per column
                    float scale_sv[8];
                    float zero_sv[8];  // Combined zero point (zero * sv for asymmetric)
                    #pragma unroll
                    for (uint col = 0; col < 8; ++col) {
                        uint global_col = b_col_base + col;
                        if (global_col < N) {
                            float scale = scales[group_idx * N + global_col];
                            float sv_val = (sv != nullptr) ? sv[global_col] : 1.0f;
                            scale_sv[col] = scale * sv_val;
                            
                            if (asymmetric && zeros != nullptr) {
                                float zero = zeros[group_idx * N + global_col];
                                zero_sv[col] = zero;
                            } else {
                                zero_sv[col] = 0.0f;
                            }
                        } else {
                            scale_sv[col] = 0.0f;
                            zero_sv[col] = 0.0f;
                        }
                    }
                    
                    // Dequantize and accumulate for each k in [0,8)
                    #pragma unroll
                    for (uint k = 0; k < 8; ++k) {
                        uint k_idx = k_sub_base + k;
                        float su_val = su_cached[k];
                        
                        // Load and dequantize 8 INT8 weights for this K row
                        half b_dequant[8];
                        #pragma unroll
                        for (uint col = 0; col < 8; ++col) {
                            uint global_col = b_col_base + col;
                            if (k_idx < K && global_col < N) {
                                int8_t w_int8 = extract_s8(int8_weights, k_idx * N + global_col);
                                float combined_scale = scale_sv[col] * su_val;
                                
                                if (asymmetric) {
                                    b_dequant[col] = dequant_int8_element_asym_fused(w_int8, combined_scale, zero_sv[col]);
                                } else {
                                    b_dequant[col] = dequant_int8_element_fused(w_int8, combined_scale);
                                }
                            } else {
                                b_dequant[col] = half(0.0h);
                            }
                        }
                        
                        // Accumulate: C += A[row,k] * B_dequant[k,col]
                        #pragma unroll
                        for (uint row = 0; row < 2; ++row) {
                            float a_val = float(a_frag[row][k]);
                            #pragma unroll
                            for (uint col = 0; col < 8; ++col) {
                                tile_acc[mi][ni][col] += a_val * float(b_dequant[col]);
                            }
                        }
                    }
                }
            }
        }
    }
    
    // -------------------------------------------------------------------------
    // Epilogue: Store accumulated results to global memory
    // -------------------------------------------------------------------------
    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        uint out_row_base = tg_row + sg_row_offset + mi * 8;
        
        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_col_base = tg_col + sg_col_offset + ni * 8;
            
            // Write 8x8 output tile
            for (uint elem = simd_lane_id; elem < 64; elem += 32) {
                uint r = elem >> 3;
                uint c = elem & 7;
                uint global_row = out_row_base + r;
                uint global_col = out_col_base + c;
                
                if (global_row < M && global_col < N) {
                    C[global_row * N + global_col] = half(tile_acc[mi][ni][c]);
                }
            }
        }
    }
}

// ============================================================================
// INT8 Quantization Test Kernels
// ============================================================================

/// Test kernel for INT8 symmetric dequantization (single element).
/// Validates the dequant_int8_element_fused function.
kernel void test_int8_dequant_sym_fused(
    device const uchar* packed_input  [[buffer(0)]],  // INT8 weights
    constant float* scale             [[buffer(1)]],  // Scale factor
    device half* output               [[buffer(2)]],  // Output
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid > 0u) return;
    
    int8_t val = extract_s8(packed_input, 0);
    output[0] = dequant_int8_element_fused(val, scale[0]);
}

/// Test kernel for INT8 asymmetric dequantization (single element).
/// Validates the dequant_int8_element_asym_fused function.
kernel void test_int8_dequant_asym_fused(
    device const uchar* packed_input  [[buffer(0)]],  // INT8 weights
    constant float* scale             [[buffer(1)]],  // Scale factor
    constant float* zero_point        [[buffer(2)]],  // Zero point
    device half* output               [[buffer(3)]],  // Output
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid > 0u) return;
    
    int8_t val = extract_s8(packed_input, 0);
    output[0] = dequant_int8_element_asym_fused(val, scale[0], zero_point[0]);
}

/// Test kernel for INT8 symmetric dequantization (x4 vectorized).
/// Validates the dequant_int8_x4_sym function.
kernel void test_int8_dequant_x4_sym(
    device const uchar* packed_input  [[buffer(0)]],  // INT8 weights (4 bytes)
    constant float* scale             [[buffer(1)]],  // Scale factor
    device half* output               [[buffer(2)]],  // Output (4 halfs)
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid > 0u) return;
    
    half4 result;
    dequant_int8_x4_sym(packed_input, 0, scale[0], result);
    
    output[0] = result.x;
    output[1] = result.y;
    output[2] = result.z;
    output[3] = result.w;
}

/// Test kernel for INT8 asymmetric dequantization (x4 vectorized).
/// Validates the dequant_int8_x4_asym function.
kernel void test_int8_dequant_x4_asym(
    device const uchar* packed_input  [[buffer(0)]],  // INT8 weights (4 bytes)
    constant float* scale             [[buffer(1)]],  // Scale factor
    constant float* zero_point        [[buffer(2)]],  // Zero point
    device half* output               [[buffer(3)]],  // Output (4 halfs)
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid > 0u) return;
    
    half4 result;
    dequant_int8_x4_asym(packed_input, 0, scale[0], zero_point[0], result);
    
    output[0] = result.x;
    output[1] = result.y;
    output[2] = result.z;
    output[3] = result.w;
}

/// Test kernel for INT8 symmetric dequantization (x8 vectorized).
/// Validates the dequant_int8_x8_sym function.
kernel void test_int8_dequant_x8_sym(
    device const uchar* packed_input  [[buffer(0)]],  // INT8 weights (8 bytes)
    constant float* scale             [[buffer(1)]],  // Scale factor
    device half* output               [[buffer(2)]],  // Output (8 halfs)
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid > 0u) return;
    
    half result[8];
    dequant_int8_x8_sym(packed_input, 0, scale[0], result);
    
    for (uint i = 0; i < 8; ++i) {
        output[i] = result[i];
    }
}

/// Test kernel for INT8 asymmetric dequantization (x8 vectorized).
/// Validates the dequant_int8_x8_asym function.
kernel void test_int8_dequant_x8_asym(
    device const uchar* packed_input  [[buffer(0)]],  // INT8 weights (8 bytes)
    constant float* scale             [[buffer(1)]],  // Scale factor
    constant float* zero_point        [[buffer(2)]],  // Zero point
    device half* output               [[buffer(3)]],  // Output (8 halfs)
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid > 0u) return;
    
    half result[8];
    dequant_int8_x8_asym(packed_input, 0, scale[0], zero_point[0], result);
    
    for (uint i = 0; i < 8; ++i) {
        output[i] = result[i];
    }
}
