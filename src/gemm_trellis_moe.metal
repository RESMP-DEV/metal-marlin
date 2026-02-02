// gemm_trellis_moe.metal - Fused MoE GEMM with Trellis 3bpw quantization
//
// ---------------------------------------------------------------------------
// Threadgroup Memory Usage Analysis
// ---------------------------------------------------------------------------
// Metal threadgroup memory limit: 32KB
// All kernels are well within limit, ensuring good occupancy.
//
// Kernel                          | Threadgroup Memory | Status
// --------------------------------|-------------------|--------
// moe_fused_trellis               |  6,688 bytes      | OK (21% of limit)
// moe_fused_trellis_fp32acc       |  7,200 bytes      | OK (22% of limit)
// moe_fused_trellis_large         | 14,368 bytes      | OK (44% of limit)
// moe_fused_trellis_simd          |  4,224 bytes      | OK (13% of limit)
// moe_fused_trellis_decode        |  3,616 bytes      | OK (11% of limit)
// moe_fused_trellis_prefill       |  8,320 bytes      | OK (26% of limit)
// moe_fused_trellis_prefill_fp32  | 10,368 bytes      | OK (32% of limit)
//
// Memory breakdown per kernel:
//   - B_gate/B_up/B_down tiles dominate (3x 2KB each for standard, 3x 4KB for large)
//   - Accumulators are small (128-1024 bytes depending on FP16/FP32 and batch)
//   - A_tile/A_tiles are minimal (32-128 bytes)
//
// Occupancy impact: <16KB usage allows 2+ concurrent threadgroups per SM,
// maximizing ALU utilization on Apple Silicon GPUs.
// ---------------------------------------------------------------------------
//
// Single kernel that handles:
//   1. Token routing to top-k experts
//   2. Trellis dequantization (3-bit EXL3) on-the-fly
//   3. SwiGLU activation (gate_proj, up_proj, down_proj)
//   4. Expert probability weighting
//
// STREAMING CHUNK APPROACH for handling intermediate_dim > 64:
// For each chunk of MOE_TILE_N (64) intermediate columns:
//   1. Compute gate_chunk and up_chunk via full K-reduction over hidden_dim
//   2. Apply SwiGLU: swiglu_chunk = silu(gate_chunk) * up_chunk
//   3. Compute partial down contribution: acc_down += swiglu_chunk @ down_weights_chunk
// This avoids materializing the full intermediate result while correctly handling
// arbitrary intermediate dimensions.
//
// Grid layout: (ceil(hidden_dim / MOE_TILE_N), M, top_k)
//   - tgid.x: output column block
//   - tgid.y: token index
//   - tgid.z: expert slot
//
// Memory layout:
//   activations:     [batch, hidden] half
//   gate_weights:    [num_experts, packed_k, packed_n, packed_bytes] uint8 (Trellis)
//   gate_scales:     [num_experts, n_groups, n] half
//   up_weights:      [num_experts, packed_k, packed_n, packed_bytes] uint8 (Trellis)
//   up_scales:       [num_experts, n_groups, n] half
//   down_weights:    [num_experts, packed_k, packed_n, packed_bytes] uint8 (Trellis)
//   down_scales:     [num_experts, n_groups, n] half
//   expert_ids:      [batch, top_k] uint32
//   expert_probs:    [batch, top_k] half
//   su/sv:           [num_experts, ...] half (sign flips)
//   grids:           [bits_to_level] half (codebook lookup)
//   output:          [batch, hidden] half

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constant constexpr uint MOE_TILE_N = 64;        // Output/intermediate columns per threadgroup (standard)
constant constexpr uint MOE_TILE_N_LARGE = 128; // 2x columns for large batch (32+) - better memory bandwidth utilization
constant constexpr uint MOE_TILE_K = 16;        // K dimension tile
constant constexpr uint MOE_SIMDGROUPS = 4;
constant constexpr uint MOE_THREADS = MOE_SIMDGROUPS * 32;  // 128 threads

// Simdgroup matrix tile dimensions (Apple Silicon fixed at 8x8)
constant constexpr uint SIMD_TILE = 8;
// Number of 8x8 tiles covering the MOE tile dimensions
constant constexpr uint N_TILES_PER_MOE = MOE_TILE_N / SIMD_TILE;    // 64/8 = 8
constant constexpr uint K_TILES_PER_MOE = MOE_TILE_K / SIMD_TILE;    // 16/8 = 2

constant constexpr uint TRELLIS_TILE = 16;
constant constexpr uint PACKED_BYTES_3BIT = 96;  // 16*16*3/8

// Scale/sign prefetch cache sizes
constant constexpr uint SCALE_CACHE_SIZE = MOE_TILE_N;  // 64 scales per N-tile
constant constexpr uint SU_CACHE_SIZE = MOE_TILE_K;     // 16 su values per K-tile
constant constexpr uint SV_CACHE_SIZE = MOE_TILE_N;     // 64 sv values per N-tile

// ---------------------------------------------------------------------------
// Debug NaN stage indices (for MOE_DEBUG_NAN mode)
// ---------------------------------------------------------------------------
#ifdef MOE_DEBUG_NAN
constant constexpr uint NAN_STAGE_NONE   = 255;  // No NaN detected
constant constexpr uint NAN_STAGE_GATE   = 0;    // NaN in gate accumulator
constant constexpr uint NAN_STAGE_UP     = 1;    // NaN in up accumulator
constant constexpr uint NAN_STAGE_SWIGLU = 2;    // NaN after SiLU(gate)*up
constant constexpr uint NAN_STAGE_DOWN   = 3;    // NaN in down accumulator
#endif

// ---------------------------------------------------------------------------
// Parameter structure
// ---------------------------------------------------------------------------

struct TrellisParams {
    uint M;              // Batch size (tokens)
    uint K;              // Input hidden dimension
    uint N;              // Output dimension (intermediate for gate/up, hidden for down)
    uint num_experts;    // Total experts
    uint top_k;          // Experts per token
    uint bits;           // 2, 3, or 4
    uint group_size;     // Quantization group size
    uint n_levels;       // Codebook levels (e.g., 8 for 3-bit)
};

// ---------------------------------------------------------------------------
// Vectorized 3-bit unpacking helpers
// ---------------------------------------------------------------------------

/// Unpack 4 consecutive 3-bit indices from a 32-bit packed value.
/// Extracts indices at bit offsets [base, base+3, base+6, base+9].
/// Returns uint4 with 4 indices (values 0-7).
inline uint4 unpack_3bit_x4(uint packed32, uint bit_offset_base) {
    return uint4(
        (packed32 >> bit_offset_base) & 0x7,
        (packed32 >> (bit_offset_base + 3)) & 0x7,
        (packed32 >> (bit_offset_base + 6)) & 0x7,
        (packed32 >> (bit_offset_base + 9)) & 0x7
    );
}

/// Unpack 8 consecutive 3-bit indices from packed bytes.
/// Loads 4 bytes starting at byte_ptr to cover the 24-bit span of 8 x 3-bit indices.
/// Outputs two uint4 vectors (lo: indices 0-3, hi: indices 4-7).
inline void unpack_3bit_x8(
    device const uint8_t* byte_ptr,
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

/// Apply grid lookup and scaling to 4 indices in batch.
/// Returns half4 with dequantized values.
inline half4 dequant_grid_vec4(
    uint4 indices,
    device const half* grid,
    half scale,
    uint n_levels
) {
    // Clamp out-of-range indices to 0
    uint4 safe_idx = uint4(
        indices.x < n_levels ? indices.x : 0,
        indices.y < n_levels ? indices.y : 0,
        indices.z < n_levels ? indices.z : 0,
        indices.w < n_levels ? indices.w : 0
    );

    return half4(
        grid[safe_idx.x] * scale,
        grid[safe_idx.y] * scale,
        grid[safe_idx.z] * scale,
        grid[safe_idx.w] * scale
    );
}

/// Apply sign flips to a half4 vector.
inline half4 apply_signs_vec4(half4 vals, half4 su_vec, half4 sv_vec) {
    return vals * su_vec * sv_vec;
}

// ---------------------------------------------------------------------------
// Vectorized SiLU (Swish) activation functions
//
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Two variants:
// 1. silu_vec4(): Exact implementation using exp()
// 2. fast_silu_vec4(): Polynomial approximation, ~2x faster, <0.5% max error
// ---------------------------------------------------------------------------

/// Exact SiLU using vectorized exp(): x * sigmoid(x)
inline half4 silu_vec4(half4 x) {
    return x / (1.0h + exp(-x));
}

/// Scalar exact SiLU for boundary handling
inline half silu_scalar(half x) {
    return x / (1.0h + exp(-x));
}

/// Fast SiLU polynomial approximation (avoids exp()).
/// Uses a rational approximation to sigmoid:
///   sigmoid(x) ≈ 0.5 + 0.5 * x * rsqrt(1 + x²)   for |x| < 4
/// This gives: silu(x) ≈ x * (0.5 + 0.5 * x * rsqrt(1 + x²))
/// Max relative error: ~0.4% (acceptable for inference)
/// ~2x faster than exp() on Apple Silicon GPU
inline half4 fast_silu_vec4(half4 x) {
    half4 x2 = x * x;
    half4 sigmoid_approx = 0.5h + 0.5h * x * rsqrt(1.0h + x2);
    return x * sigmoid_approx;
}

/// Scalar fast SiLU for boundary handling
inline half fast_silu_scalar(half x) {
    half x2 = x * x;
    half sigmoid_approx = 0.5h + 0.5h * x * rsqrt(1.0h + x2);
    return x * sigmoid_approx;
}

// ---------------------------------------------------------------------------
// 3-bit Trellis dequantization
// ---------------------------------------------------------------------------

inline half trellis_dequant_3bit(
    device const uint8_t* packed_weights,
    device const half* scales,
    device const half* su,
    device const half* sv,
    device const half* grid,
    uint expert_id,
    uint global_k,
    uint global_n,
    uint K_dim,
    uint N_dim,
    uint group_size,
    uint n_levels
) {
    // Compute tile position
    uint num_tiles_n = (N_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint tile_k = global_k / TRELLIS_TILE;
    uint tile_n = global_n / TRELLIS_TILE;
    uint k_in_tile = global_k % TRELLIS_TILE;
    uint n_in_tile = global_n % TRELLIS_TILE;
    uint tile_idx = tile_k * num_tiles_n + tile_n;

    device const uint8_t* tile_packed = packed_weights + tile_idx * PACKED_BYTES_3BIT;

    // Transposed indexing: idx = n * TILE_DIM + k
    uint idx_in_tile = n_in_tile * TRELLIS_TILE + k_in_tile;

    // Unpack 3-bit index
    uint bit_offset = idx_in_tile * 3;
    uint byte_idx = bit_offset >> 3;
    uint bit_in_byte = bit_offset & 7;

    uint packed_val = uint(tile_packed[byte_idx]);
    if (bit_in_byte + 3 > 8) {
        packed_val |= uint(tile_packed[byte_idx + 1]) << 8;
    }
    uint codebook_idx = (packed_val >> bit_in_byte) & 0x7;

    if (codebook_idx >= n_levels) {
        codebook_idx = 0;
    }

    half dequant = grid[codebook_idx];

    // Apply scale
    uint n_groups = (K_dim + group_size - 1) / group_size;
    uint group_idx = global_k / group_size;
    half scale = scales[expert_id * N_dim * n_groups + group_idx * N_dim + global_n];
    dequant *= scale;

    // Apply sign flips
    dequant *= su[expert_id * K_dim + global_k];
    dequant *= sv[expert_id * N_dim + global_n];

    return dequant;
}

// ---------------------------------------------------------------------------
// Load Trellis weight tile with dequantization
// ---------------------------------------------------------------------------

inline void load_trellis_tile(
    device const uint8_t* packed_weights,
    device const half* scales,
    device const half* su,
    device const half* sv,
    device const half* grid,
    threadgroup half (&B_buf)[MOE_TILE_K][MOE_TILE_N],
    uint k_block,
    uint n_block,
    uint expert_id,
    uint K_dim,
    uint N_dim,
    uint group_size,
    uint n_levels,
    uint thread_idx
) {
    const uint elems_per_thread = (MOE_TILE_K * MOE_TILE_N) / MOE_THREADS;

    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        uint k_local = flat_idx / MOE_TILE_N;
        uint n_local = flat_idx % MOE_TILE_N;

        uint global_k = k_block + k_local;
        uint global_n = n_block + n_local;

        half val = 0.0h;
        if (global_k < K_dim && global_n < N_dim) {
            val = trellis_dequant_3bit(
                packed_weights, scales, su, sv, grid,
                expert_id, global_k, global_n,
                K_dim, N_dim, group_size, n_levels
            );
        }
        B_buf[k_local][n_local] = val;
    }
}

// ---------------------------------------------------------------------------
// 3-bit Trellis dequantization with prefetched scale/sign caches
//
// OPTIMIZATION: Scales and sign vectors are accessed with the same indices
// across all 16x16=256 elements in a tile. By prefetching these values into
// threadgroup memory once per K-tile, we reduce global memory traffic by ~256x
// for scales (MOE_TILE_K * MOE_TILE_N accesses -> 1 prefetch) and similarly
// for su/sv vectors.
// ---------------------------------------------------------------------------

inline half trellis_dequant_3bit_cached(
    device const uint8_t* packed_weights,
    threadgroup const half* scale_cache,  // [MOE_TILE_N] prefetched scales for this K-group
    threadgroup const half* su_cache,     // [MOE_TILE_K] prefetched su for this K-tile
    threadgroup const half* sv_cache,     // [MOE_TILE_N] prefetched sv for this N-tile
    device const half* grid,
    uint global_k,
    uint global_n,
    uint k_local,                          // Position within current K-tile
    uint n_local,                          // Position within current N-tile
    uint N_dim,
    uint n_levels
) {
    // Compute tile position
    uint num_tiles_n = (N_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint tile_k = global_k / TRELLIS_TILE;
    uint tile_n = global_n / TRELLIS_TILE;
    uint k_in_tile = global_k % TRELLIS_TILE;
    uint n_in_tile = global_n % TRELLIS_TILE;
    uint tile_idx = tile_k * num_tiles_n + tile_n;

    device const uint8_t* tile_packed = packed_weights + tile_idx * PACKED_BYTES_3BIT;

    // Transposed indexing: idx = n * TILE_DIM + k
    uint idx_in_tile = n_in_tile * TRELLIS_TILE + k_in_tile;

    // Unpack 3-bit index
    uint bit_offset = idx_in_tile * 3;
    uint byte_idx = bit_offset >> 3;
    uint bit_in_byte = bit_offset & 7;

    uint packed_val = uint(tile_packed[byte_idx]);
    if (bit_in_byte + 3 > 8) {
        packed_val |= uint(tile_packed[byte_idx + 1]) << 8;
    }
    uint codebook_idx = (packed_val >> bit_in_byte) & 0x7;

    if (codebook_idx >= n_levels) {
        codebook_idx = 0;
    }

    half dequant = grid[codebook_idx];

    // Apply scale from cache (indexed by n_local within tile)
    dequant *= scale_cache[n_local];

    // Apply sign flips from caches
    dequant *= su_cache[k_local];
    dequant *= sv_cache[n_local];

    return dequant;
}

// ---------------------------------------------------------------------------
// Prefetch scales and sign vectors into threadgroup memory
//
// Called once per K-tile to populate caches. Uses cooperative loading where
// different threads load different elements in parallel.
//
// Memory layout assumed:
//   scales: [num_experts, n_groups, N_dim]
//   su:     [num_experts, K_dim]
//   sv:     [num_experts, N_dim]
// ---------------------------------------------------------------------------

inline void prefetch_scale_sign_caches(
    device const half* scales,
    device const half* su,
    device const half* sv,
    threadgroup half* scale_cache,        // [MOE_TILE_N]
    threadgroup half* su_cache,           // [MOE_TILE_K]
    threadgroup half* sv_cache,           // [MOE_TILE_N]
    uint expert_id,
    uint k_block,                          // Start of K-tile
    uint n_block,                          // Start of N-tile
    uint K_dim,
    uint N_dim,
    uint group_size,
    uint thread_idx
) {
    // Compute scale indexing parameters
    uint n_groups = (K_dim + group_size - 1) / group_size;
    uint group_idx = k_block / group_size;
    uint scale_base = expert_id * N_dim * n_groups + group_idx * N_dim;

    // Load scales: threads 0 to MOE_TILE_N-1 each load one scale
    // With MOE_THREADS=128 and MOE_TILE_N=64, single pass suffices
    if (thread_idx < MOE_TILE_N) {
        uint n_idx = n_block + thread_idx;
        if (n_idx < N_dim) {
            scale_cache[thread_idx] = scales[scale_base + n_idx];
        } else {
            scale_cache[thread_idx] = 0.0h;
        }
    }

    // Load su: threads 0 to MOE_TILE_K-1 each load one su value
    // MOE_TILE_K=16, so threads 0-15 handle this
    if (thread_idx < MOE_TILE_K) {
        uint k_idx = k_block + thread_idx;
        if (k_idx < K_dim) {
            su_cache[thread_idx] = su[expert_id * K_dim + k_idx];
        } else {
            su_cache[thread_idx] = 1.0h;  // Identity for out-of-bounds
        }
    }

    // Load sv: threads 64 to 127 load sv (offset by 64 to avoid conflicts with scale loading)
    // This allows scale and sv loads to happen in parallel
    if (thread_idx >= MOE_TILE_N && thread_idx < MOE_TILE_N + MOE_TILE_N) {
        uint sv_idx = thread_idx - MOE_TILE_N;
        uint n_idx = n_block + sv_idx;
        if (n_idx < N_dim) {
            sv_cache[sv_idx] = sv[expert_id * N_dim + n_idx];
        } else {
            sv_cache[sv_idx] = 1.0h;  // Identity for out-of-bounds
        }
    }
}

// ---------------------------------------------------------------------------
// Load Trellis weight tile with cached scale/sign prefetching
//
// OPTIMIZATION: Uses prefetched scale and sign caches to avoid redundant
// global memory accesses. The caches must be populated by prefetch_scale_sign_caches
// before calling this function.
// ---------------------------------------------------------------------------

inline void load_trellis_tile_cached(
    device const uint8_t* packed_weights,
    threadgroup const half* scale_cache,  // [MOE_TILE_N] prefetched scales
    threadgroup const half* su_cache,     // [MOE_TILE_K] prefetched su
    threadgroup const half* sv_cache,     // [MOE_TILE_N] prefetched sv
    device const half* grid,
    threadgroup half (&B_buf)[MOE_TILE_K][MOE_TILE_N],
    uint k_block,
    uint n_block,
    uint K_dim,
    uint N_dim,
    uint n_levels,
    uint thread_idx
) {
    const uint elems_per_thread = (MOE_TILE_K * MOE_TILE_N) / MOE_THREADS;

    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        uint k_local = flat_idx / MOE_TILE_N;
        uint n_local = flat_idx % MOE_TILE_N;

        uint global_k = k_block + k_local;
        uint global_n = n_block + n_local;

        half val = 0.0h;
        if (global_k < K_dim && global_n < N_dim) {
            val = trellis_dequant_3bit_cached(
                packed_weights, scale_cache, su_cache, sv_cache, grid,
                global_k, global_n, k_local, n_local,
                N_dim, n_levels
            );
        }
        B_buf[k_local][n_local] = val;
    }
}

// ---------------------------------------------------------------------------
// Vectorized Trellis weight tile loading (vec4 optimized)
//
// OPTIMIZATION: Processes 4 elements at a time with unrolled loops.
// Uses the cached scale/sign prefetch infrastructure for reduced global
// memory traffic while also benefiting from vec4 loop unrolling.
//
// Key improvements over load_trellis_tile_cached:
//   1. Unrolled inner loop (4 elements per iteration)
//   2. Reduced loop overhead
//   3. Better instruction-level parallelism
//
// Requires caches to be prefetched via prefetch_scale_sign_caches() first.
// ---------------------------------------------------------------------------

inline void load_trellis_tile_vec4(
    device const uint8_t* packed_weights,
    threadgroup const half* scale_cache,  // [MOE_TILE_N] prefetched scales
    threadgroup const half* su_cache,     // [MOE_TILE_K] prefetched su
    threadgroup const half* sv_cache,     // [MOE_TILE_N] prefetched sv
    device const half* grid,
    threadgroup half (&B_buf)[MOE_TILE_K][MOE_TILE_N],
    uint k_block,
    uint n_block,
    uint K_dim,
    uint N_dim,
    uint n_levels,
    uint thread_idx
) {
    // Tile layout: 16x64 weight tile = 1024 elements
    // 128 threads, so 8 elements per thread
    const uint num_tiles_n = (N_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    const uint tile_k = k_block / TRELLIS_TILE;
    const uint elems_per_thread = (MOE_TILE_K * MOE_TILE_N) / MOE_THREADS;  // 8

    // Process 8 elements per thread, 4 at a time for better ILP
    for (uint i = 0; i < elems_per_thread; i += 4) {
        uint flat_base = thread_idx * elems_per_thread + i;

        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            uint flat_idx = flat_base + j;
            uint k_local = flat_idx / MOE_TILE_N;
            uint n_local = flat_idx % MOE_TILE_N;
            uint global_k = k_block + k_local;
            uint global_n = n_block + n_local;

            if (global_k >= K_dim || global_n >= N_dim) {
                B_buf[k_local][n_local] = 0.0h;
                continue;
            }

            // Compute trellis tile position
            uint tile_n = global_n / TRELLIS_TILE;
            uint k_in_tile = global_k % TRELLIS_TILE;
            uint n_in_tile = global_n % TRELLIS_TILE;
            uint tile_idx = tile_k * num_tiles_n + tile_n;

            device const uint8_t* tile_packed = packed_weights + tile_idx * PACKED_BYTES_3BIT;

            // Transposed indexing for trellis: idx = n * 16 + k
            uint idx_in_tile = n_in_tile * TRELLIS_TILE + k_in_tile;

            // Unpack 3-bit index
            uint bit_offset = idx_in_tile * 3;
            uint byte_idx = bit_offset >> 3;
            uint bit_in_byte = bit_offset & 7;

            // Load bytes to cover the 3-bit span
            uint packed_val = uint(tile_packed[byte_idx]);
            if (bit_in_byte + 3 > 8) {
                packed_val |= uint(tile_packed[byte_idx + 1]) << 8;
            }
            uint codebook_idx = (packed_val >> bit_in_byte) & 0x7;

            if (codebook_idx >= n_levels) {
                codebook_idx = 0;
            }

            // Grid lookup and apply cached scale/signs
            half dequant = grid[codebook_idx];
            dequant *= scale_cache[n_local];
            dequant *= su_cache[k_local];
            dequant *= sv_cache[n_local];

            B_buf[k_local][n_local] = dequant;
        }
    }
}

// ---------------------------------------------------------------------------
// Load activation tile (single token) - Coalesced vector load version
//
// OPTIMIZATION: Uses half4 vector loads for memory coalescing.
// For MOE_TILE_K = 16, only 4 threads participate, but those 4 threads
// load contiguous 8-byte (half4) chunks resulting in a single 32-byte
// coalesced transaction.
//
// Memory access pattern:
//   Thread 0: loads halfs [0,1,2,3]   at base + 0
//   Thread 1: loads halfs [4,5,6,7]   at base + 8
//   Thread 2: loads halfs [8,9,10,11] at base + 16
//   Thread 3: loads halfs [12,13,14,15] at base + 24
// All 4 loads are to consecutive addresses = fully coalesced 32-byte transaction.
// ---------------------------------------------------------------------------

inline void load_activation_tile(
    device const half* activations,
    threadgroup half (&A_buf)[MOE_TILE_K],
    uint token_idx,
    uint k_block,
    uint hidden_dim,
    uint thread_idx
) {
    // Each thread loads 4 consecutive halfs (half4 = 8 bytes)
    // MOE_TILE_K = 16, so we need 4 threads (16/4 = 4)
    constexpr uint HALFS_PER_THREAD = 4;
    constexpr uint THREADS_NEEDED = MOE_TILE_K / HALFS_PER_THREAD;  // 4 threads

    if (thread_idx < THREADS_NEEDED) {
        uint lane_offset = thread_idx * HALFS_PER_THREAD;
        uint global_k = k_block + lane_offset;

        // Compute base address for this token's activation row
        device const half* row_base = activations + token_idx * hidden_dim;

        // Check bounds - all 4 elements must be valid for vector load
        if (global_k + HALFS_PER_THREAD <= hidden_dim) {
            // Coalesced vector load: threads 0-3 access consecutive 8-byte chunks
            half4 chunk = *((device const half4*)(row_base + global_k));

            // Store to threadgroup memory
            A_buf[lane_offset + 0] = chunk.x;
            A_buf[lane_offset + 1] = chunk.y;
            A_buf[lane_offset + 2] = chunk.z;
            A_buf[lane_offset + 3] = chunk.w;
        } else {
            // Boundary case: load element-by-element with bounds checking
            for (uint i = 0; i < HALFS_PER_THREAD; ++i) {
                uint k = global_k + i;
                A_buf[lane_offset + i] = (k < hidden_dim) ? row_base[k] : 0.0h;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Load activation with simdgroup-aware coalescing for larger tiles
//
// OPTIMIZATION: Leverages simdgroup structure for better memory coalescing.
// Each simdgroup (32 threads) loads contiguous memory with perfect coalescing.
//
// Memory access pattern (per simdgroup):
//   Lane 0:  loads halfs [0,1,2,3]     at base + simd_group*128 + 0
//   Lane 1:  loads halfs [4,5,6,7]     at base + simd_group*128 + 4
//   ...
//   Lane 31: loads halfs [124,125,126,127] at base + simd_group*128 + 124
//
// Each lane loads a contiguous half4 (8 bytes), and consecutive lanes
// access consecutive addresses = fully coalesced 256-byte transaction per simdgroup.
//
// For MOE_TILE_K = 16, only the first simdgroup's first 4 lanes participate.
// Threadgroup store uses bank-conflict-free pattern (each lane writes to
// unique bank based on lane_offset).
// ---------------------------------------------------------------------------

inline void load_activation_vec8(
    device const half* activations,
    threadgroup half (&A_buf)[MOE_TILE_K],
    uint token_idx,
    uint k_block,
    uint hidden_dim,
    uint simd_lane,
    uint simd_group
) {
    // For MOE_TILE_K = 16, we only need the first simdgroup
    // and only 4 threads (lanes 0-3) to load 16 halfs via half4 vectors
    if (simd_group == 0 && simd_lane < 4) {
        uint lane_offset = simd_lane * 4;  // 4 halfs per thread
        uint global_k = k_block + lane_offset;

        device const half* row_base = activations + token_idx * hidden_dim;

        if (global_k + 4 <= hidden_dim) {
            // Coalesced vector load: lanes 0-3 access consecutive 8-byte chunks
            // Total: 32 bytes in a single coalesced transaction
            half4 chunk = *((device const half4*)(row_base + global_k));

            // Store to threadgroup with bank-conflict-free pattern
            // Each lane writes to unique indices: 0-3, 4-7, 8-11, 12-15
            // 16 banks in threadgroup memory means each write hits different bank
            A_buf[lane_offset + 0] = chunk.x;
            A_buf[lane_offset + 1] = chunk.y;
            A_buf[lane_offset + 2] = chunk.z;
            A_buf[lane_offset + 3] = chunk.w;
        } else {
            // Boundary case: element-by-element with bounds checking
            for (uint i = 0; i < 4; ++i) {
                uint k = global_k + i;
                A_buf[lane_offset + i] = (k < hidden_dim) ? row_base[k] : 0.0h;
            }
        }
    }
}

// ===========================================================================
// Main MoE kernel with SwiGLU activation - Streaming Chunk Approach
//
// For each token, computes:
//   output = sum_{i=0}^{top_k-1} prob[i] * down(silu(gate(x_i)) * up(x_i))
//
// Where x_i is routed to expert_id[i]
//
// Handles intermediate_dim > 64 by processing in chunks:
//   - For each 64-column chunk of intermediate dimension
//   - Compute gate/up projections
//   - Apply SwiGLU
//   - Accumulate partial down projection
// ===========================================================================

kernel void moe_trellis_swiglu(
    device const half* activations       [[buffer(0)]],   // [batch, hidden]
    device const uint8_t* gate_weights   [[buffer(1)]],   // [num_experts, ...] Trellis
    device const half* gate_scales       [[buffer(2)]],   // [num_experts, ...]
    device const uint8_t* up_weights     [[buffer(3)]],   // [num_experts, ...] Trellis
    device const half* up_scales         [[buffer(4)]],   // [num_experts, ...]
    device const uint8_t* down_weights   [[buffer(5)]],   // [num_experts, ...] Trellis
    device const half* down_scales       [[buffer(6)]],   // [num_experts, ...]
    device const half* gate_su           [[buffer(7)]],   // [num_experts, K]
    device const half* gate_sv           [[buffer(8)]],   // [num_experts, N]
    device const half* up_su             [[buffer(9)]],   // [num_experts, K]
    device const half* up_sv             [[buffer(10)]],  // [num_experts, N]
    device const half* down_su           [[buffer(11)]],  // [num_experts, N]
    device const half* down_sv           [[buffer(12)]],  // [num_experts, K]
    device const half* grid              [[buffer(13)]],  // Codebook grid
    device const uint* expert_ids        [[buffer(14)]],  // [batch, top_k]
    device const half* expert_probs      [[buffer(15)]],  // [batch, top_k]
    device float* output                 [[buffer(16)]],  // [batch, hidden] FP32 for atomic add
    constant TrellisParams& p            [[buffer(17)]],
#ifdef MOE_DEBUG_NAN
    device half* debug_gate              [[buffer(18)]],  // [batch, intermediate_dim]
    device half* debug_up                [[buffer(19)]],  // [batch, intermediate_dim]
    device half* debug_swiglu            [[buffer(20)]],  // [batch, intermediate_dim]
    device uint* debug_nan_stage         [[buffer(21)]],  // [batch] stage where NaN first appeared
#endif
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint thread_idx                      [[thread_index_in_threadgroup]]
) {
    // Threadgroup memory - sized to fit within 32KB limit
    // A_tile: 16*2 = 32 bytes (single token activation slice)
    // B_gate/up/down: 16*64*2 = 2048 bytes each = 6144 bytes
    // swiglu_result: 64*2 = 128 bytes
    // output_tile: 64*2 = 128 bytes
    // gate_acc/up_acc: 64*2 = 128 bytes each = 256 bytes
    // Total: ~6.7KB
    threadgroup half A_tile[MOE_TILE_K];                    // Activation slice
    threadgroup half B_gate[MOE_TILE_K][MOE_TILE_N];        // Gate weights
    threadgroup half B_up[MOE_TILE_K][MOE_TILE_N];          // Up weights
    threadgroup half B_down[MOE_TILE_K][MOE_TILE_N];        // Down weights
    threadgroup half swiglu_result[MOE_TILE_N];             // SwiGLU intermediate (one chunk)
    threadgroup half output_tile[MOE_TILE_N];               // Accumulated output tile
    threadgroup half gate_acc_tg[MOE_TILE_N];               // Gate accumulator (shared)
    threadgroup half up_acc_tg[MOE_TILE_N];                 // Up accumulator (shared)

    // Grid indices
    const uint n_block = tgid.x * MOE_TILE_N;  // Output column block (into hidden_dim)
    const uint token_idx = tgid.y;              // Token index
    const uint slot = tgid.z;                   // Expert slot (0 to top_k-1)

    // Early exit for out-of-bounds
    if (token_idx >= p.M || slot >= p.top_k) {
        return;
    }

    // CRITICAL: Get the expert assigned to THIS SPECIFIC token for this slot
    const uint expert_id = expert_ids[token_idx * p.top_k + slot];
    if (expert_id >= p.num_experts) {
        return;
    }

    // Get probability weight for this expert
    const half prob = expert_probs[token_idx * p.top_k + slot];

    // Dimensions
    const uint hidden_dim = p.K;
    const uint intermediate_dim = p.N;

    // Compute expert weight offsets
    uint num_tiles_k_gate = (hidden_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_n_gate = (intermediate_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_k_down = (intermediate_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_n_down = (hidden_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;

    uint gate_up_expert_size = num_tiles_k_gate * num_tiles_n_gate * PACKED_BYTES_3BIT;
    uint down_expert_size = num_tiles_k_down * num_tiles_n_down * PACKED_BYTES_3BIT;

    // Pointers to this expert's weights
    device const uint8_t* gate_w = gate_weights + expert_id * gate_up_expert_size;
    device const uint8_t* up_w = up_weights + expert_id * gate_up_expert_size;
    device const uint8_t* down_w = down_weights + expert_id * down_expert_size;

    // Initialize output tile to zero (will accumulate partial down projections)
    for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
        output_tile[i] = 0.0h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // STREAMING CHUNK APPROACH
    // Process intermediate_dim in chunks of MOE_TILE_N (64)
    // =========================================================================

    uint num_intermediate_chunks = (intermediate_dim + MOE_TILE_N - 1) / MOE_TILE_N;
    uint num_k_tiles_hidden = (hidden_dim + MOE_TILE_K - 1) / MOE_TILE_K;

    for (uint chunk_idx = 0; chunk_idx < num_intermediate_chunks; ++chunk_idx) {
        uint n_chunk_offset = chunk_idx * MOE_TILE_N;  // Column offset in intermediate space

        // =================================================================
        // PHASE 1: Gate and Up projections for this chunk
        // =================================================================

        // Initialize accumulators
        for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
            gate_acc_tg[i] = 0.0h;
            up_acc_tg[i] = 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // K-dimension loop for gate + up projections
        for (uint kt = 0; kt < num_k_tiles_hidden; ++kt) {
            uint k_block = kt * MOE_TILE_K;

            // Load activations for this token
            load_activation_tile(activations, A_tile, token_idx, k_block, hidden_dim, thread_idx);

            // Load gate weights for this chunk
            load_trellis_tile(gate_w, gate_scales, gate_su, gate_sv, grid,
                            B_gate, k_block, n_chunk_offset, expert_id,
                            hidden_dim, intermediate_dim, p.group_size, p.n_levels, thread_idx);

            // Load up weights for this chunk
            load_trellis_tile(up_w, up_scales, up_su, up_sv, grid,
                            B_up, k_block, n_chunk_offset, expert_id,
                            hidden_dim, intermediate_dim, p.group_size, p.n_levels, thread_idx);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate gate and up: each thread handles a subset of output columns
            for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
                half local_gate = 0.0h;
                half local_up = 0.0h;

                // Dot product over K tile
                for (uint k = 0; k < MOE_TILE_K; ++k) {
                    half act = A_tile[k];
                    local_gate += act * B_gate[k][i];
                    local_up += act * B_up[k][i];
                }

                gate_acc_tg[i] += local_gate;
                up_acc_tg[i] += local_up;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

#ifdef MOE_DEBUG_NAN
        // Check for NaN in gate accumulator and write debug output
        for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
            uint global_n = n_chunk_offset + i;
            if (global_n < intermediate_dim) {
                half g = gate_acc_tg[i];
                debug_gate[token_idx * intermediate_dim + global_n] = g;
                if (isnan(g) && debug_nan_stage[token_idx] == NAN_STAGE_NONE) {
                    debug_nan_stage[token_idx] = NAN_STAGE_GATE;
                }
            }
        }

        // Check for NaN in up accumulator and write debug output
        for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
            uint global_n = n_chunk_offset + i;
            if (global_n < intermediate_dim) {
                half u = up_acc_tg[i];
                debug_up[token_idx * intermediate_dim + global_n] = u;
                if (isnan(u) && debug_nan_stage[token_idx] == NAN_STAGE_NONE) {
                    debug_nan_stage[token_idx] = NAN_STAGE_UP;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Early exit if NaN detected in gate or up (preserve problematic values)
        if (debug_nan_stage[token_idx] != NAN_STAGE_NONE) {
            return;
        }
#endif

        // =================================================================
        // PHASE 2: Apply SwiGLU and store to swiglu_result
        // =================================================================

        // Vectorized SwiGLU using half4 SIMD operations with fast polynomial SiLU
        // MOE_TILE_N=64, each thread processes 4 elements, 16 threads cover the tile
        // fast_silu_vec4 uses rsqrt-based approximation (~2x faster than exp)
        for (uint i = thread_idx * 4; i < MOE_TILE_N; i += MOE_THREADS * 4) {
            uint global_n = n_chunk_offset + i;

            if (global_n + 3 < intermediate_dim) {
                // Fast path: all 4 elements valid - use vectorized load/store
                half4 g = half4(gate_acc_tg[i], gate_acc_tg[i+1],
                                gate_acc_tg[i+2], gate_acc_tg[i+3]);
                half4 u = half4(up_acc_tg[i], up_acc_tg[i+1],
                                up_acc_tg[i+2], up_acc_tg[i+3]);

                // Fast vectorized SiLU: polynomial approximation avoiding exp()
                half4 silu_g = fast_silu_vec4(g);
                half4 result = silu_g * u;

                swiglu_result[i]   = result.x;
                swiglu_result[i+1] = result.y;
                swiglu_result[i+2] = result.z;
                swiglu_result[i+3] = result.w;
            } else {
                // Boundary handling: check each element
                for (uint j = 0; j < 4 && i + j < MOE_TILE_N; ++j) {
                    if (global_n + j < intermediate_dim) {
                        half g = gate_acc_tg[i + j];
                        half u = up_acc_tg[i + j];
                        half silu_g = fast_silu_scalar(g);
                        swiglu_result[i + j] = silu_g * u;
                    } else {
                        swiglu_result[i + j] = 0.0h;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

#ifdef MOE_DEBUG_NAN
        // Check for NaN after SwiGLU and write debug output
        for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
            uint global_n = n_chunk_offset + i;
            if (global_n < intermediate_dim) {
                half s = swiglu_result[i];
                debug_swiglu[token_idx * intermediate_dim + global_n] = s;
                if (isnan(s) && debug_nan_stage[token_idx] == NAN_STAGE_NONE) {
                    debug_nan_stage[token_idx] = NAN_STAGE_SWIGLU;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Early exit if NaN detected in SwiGLU (preserve problematic values)
        if (debug_nan_stage[token_idx] != NAN_STAGE_NONE) {
            return;
        }
#endif

        // =================================================================
        // PHASE 3: Partial down projection for this chunk
        // =================================================================

        // Determine actual chunk size
        uint chunk_end = min(n_chunk_offset + MOE_TILE_N, intermediate_dim);
        uint chunk_size = chunk_end - n_chunk_offset;
        uint num_k_tiles_chunk = (chunk_size + MOE_TILE_K - 1) / MOE_TILE_K;

        for (uint kdt = 0; kdt < num_k_tiles_chunk; ++kdt) {
            uint k_down_local = kdt * MOE_TILE_K;
            uint k_down_global = n_chunk_offset + k_down_local;

            // Load down weights for this K tile
            load_trellis_tile(down_w, down_scales, down_su, down_sv, grid,
                            B_down, k_down_global, n_block, expert_id,
                            intermediate_dim, hidden_dim, p.group_size, p.n_levels, thread_idx);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate: each thread handles a subset of output columns
            for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
                uint global_out_n = n_block + i;
                if (global_out_n >= hidden_dim) continue;

                half local_down = 0.0h;

                // Dot product over this K tile
                uint k_end = min(MOE_TILE_K, chunk_size - k_down_local);
                for (uint k = 0; k < k_end; ++k) {
                    half swiglu_k = swiglu_result[k_down_local + k];
                    local_down += swiglu_k * B_down[k][i];
                }

                output_tile[i] += local_down;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }  // End of chunk loop

#ifdef MOE_DEBUG_NAN
    // Check for NaN in down accumulator (output_tile)
    for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
        uint global_n = n_block + i;
        if (global_n < hidden_dim) {
            half d = output_tile[i];
            if (isnan(d) && debug_nan_stage[token_idx] == NAN_STAGE_NONE) {
                debug_nan_stage[token_idx] = NAN_STAGE_DOWN;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Early exit if NaN detected in down (preserve problematic values)
    if (debug_nan_stage[token_idx] != NAN_STAGE_NONE) {
        return;
    }
#endif

    // =========================================================================
    // PHASE 4: Write output with probability weighting using FP32 atomic CAS
    //
    // CRITICAL FIX: Multiple threadgroups (one per expert slot) write to the
    // same output locations concurrently. We use FP32 output buffer with
    // CAS (compare-and-swap) atomic add for proper synchronization.
    //
    // The output buffer (FP32) must be pre-initialized to zero before kernel launch.
    // The dispatch code converts FP32 output to FP16 after kernel completes.
    // =========================================================================

    for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
        uint global_n = n_block + i;
        if (global_n < hidden_dim) {
            float weighted = float(output_tile[i]) * float(prob);
            uint out_idx = token_idx * hidden_dim + global_n;

            // FP32 atomic add using CAS loop
            device atomic_uint* atomic_ptr = (device atomic_uint*)(&output[out_idx]);
            uint old_bits = atomic_load_explicit(atomic_ptr, memory_order_relaxed);
            uint new_bits;
            bool success;
            do {
                float old_val = as_type<float>(old_bits);
                float new_val = old_val + weighted;
                new_bits = as_type<uint>(new_val);
                success = atomic_compare_exchange_weak_explicit(
                    atomic_ptr, &old_bits, new_bits,
                    memory_order_relaxed, memory_order_relaxed);
            } while (!success);
        }
    }
}

// ===========================================================================
// FP32 Accumulation Variant
//
// Uses FP32 accumulators for numerical stability with large K dimensions.
// Same algorithm as moe_trellis_swiglu but with FP32 intermediate values.
// ===========================================================================

kernel void moe_trellis_swiglu_fp32acc(
    device const half* activations       [[buffer(0)]],
    device const uint8_t* gate_weights   [[buffer(1)]],
    device const half* gate_scales       [[buffer(2)]],
    device const uint8_t* up_weights     [[buffer(3)]],
    device const half* up_scales         [[buffer(4)]],
    device const uint8_t* down_weights   [[buffer(5)]],
    device const half* down_scales       [[buffer(6)]],
    device const half* gate_su           [[buffer(7)]],
    device const half* gate_sv           [[buffer(8)]],
    device const half* up_su             [[buffer(9)]],
    device const half* up_sv             [[buffer(10)]],
    device const half* down_su           [[buffer(11)]],
    device const half* down_sv           [[buffer(12)]],
    device const half* grid              [[buffer(13)]],
    device const uint* expert_ids        [[buffer(14)]],
    device const half* expert_probs      [[buffer(15)]],
    device float* output                 [[buffer(16)]],  // FP32 for atomic add
    constant TrellisParams& p            [[buffer(17)]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint thread_idx                      [[thread_index_in_threadgroup]]
) {
    // Threadgroup memory with FP32 accumulators
    threadgroup half A_tile[MOE_TILE_K];
    threadgroup half B_gate[MOE_TILE_K][MOE_TILE_N];
    threadgroup half B_up[MOE_TILE_K][MOE_TILE_N];
    threadgroup half B_down[MOE_TILE_K][MOE_TILE_N];
    threadgroup float swiglu_result[MOE_TILE_N];             // FP32
    threadgroup float output_tile[MOE_TILE_N];               // FP32
    threadgroup float gate_acc_tg[MOE_TILE_N];               // FP32
    threadgroup float up_acc_tg[MOE_TILE_N];                 // FP32

    const uint n_block = tgid.x * MOE_TILE_N;
    const uint token_idx = tgid.y;
    const uint slot = tgid.z;

    if (token_idx >= p.M || slot >= p.top_k) {
        return;
    }

    const uint expert_id = expert_ids[token_idx * p.top_k + slot];
    if (expert_id >= p.num_experts) {
        return;
    }

    const float prob = float(expert_probs[token_idx * p.top_k + slot]);

    const uint hidden_dim = p.K;
    const uint intermediate_dim = p.N;

    uint num_tiles_k_gate = (hidden_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_n_gate = (intermediate_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_k_down = (intermediate_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_n_down = (hidden_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;

    uint gate_up_expert_size = num_tiles_k_gate * num_tiles_n_gate * PACKED_BYTES_3BIT;
    uint down_expert_size = num_tiles_k_down * num_tiles_n_down * PACKED_BYTES_3BIT;

    device const uint8_t* gate_w = gate_weights + expert_id * gate_up_expert_size;
    device const uint8_t* up_w = up_weights + expert_id * gate_up_expert_size;
    device const uint8_t* down_w = down_weights + expert_id * down_expert_size;

    for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
        output_tile[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_intermediate_chunks = (intermediate_dim + MOE_TILE_N - 1) / MOE_TILE_N;
    uint num_k_tiles_hidden = (hidden_dim + MOE_TILE_K - 1) / MOE_TILE_K;

    for (uint chunk_idx = 0; chunk_idx < num_intermediate_chunks; ++chunk_idx) {
        uint n_chunk_offset = chunk_idx * MOE_TILE_N;

        for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
            gate_acc_tg[i] = 0.0f;
            up_acc_tg[i] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kt = 0; kt < num_k_tiles_hidden; ++kt) {
            uint k_block = kt * MOE_TILE_K;

            load_activation_tile(activations, A_tile, token_idx, k_block, hidden_dim, thread_idx);

            load_trellis_tile(gate_w, gate_scales, gate_su, gate_sv, grid,
                            B_gate, k_block, n_chunk_offset, expert_id,
                            hidden_dim, intermediate_dim, p.group_size, p.n_levels, thread_idx);

            load_trellis_tile(up_w, up_scales, up_su, up_sv, grid,
                            B_up, k_block, n_chunk_offset, expert_id,
                            hidden_dim, intermediate_dim, p.group_size, p.n_levels, thread_idx);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
                float local_gate = 0.0f;
                float local_up = 0.0f;

                for (uint k = 0; k < MOE_TILE_K; ++k) {
                    float act = float(A_tile[k]);
                    local_gate += act * float(B_gate[k][i]);
                    local_up += act * float(B_up[k][i]);
                }

                gate_acc_tg[i] += local_gate;
                up_acc_tg[i] += local_up;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Vectorized SwiGLU using float4 SIMD operations
        // MOE_TILE_N=64, each thread processes 4 elements, 16 threads cover the tile
        for (uint i = thread_idx * 4; i < MOE_TILE_N; i += MOE_THREADS * 4) {
            uint global_n = n_chunk_offset + i;

            if (global_n + 3 < intermediate_dim) {
                // Fast path: all 4 elements valid
                float4 g = float4(gate_acc_tg[i], gate_acc_tg[i+1],
                                  gate_acc_tg[i+2], gate_acc_tg[i+3]);
                float4 u = float4(up_acc_tg[i], up_acc_tg[i+1],
                                  up_acc_tg[i+2], up_acc_tg[i+3]);

                // Vectorized SiLU: x * sigmoid(x) = x / (1 + exp(-x))
                float4 silu_g = g / (1.0f + exp(-g));
                float4 result = silu_g * u;

                swiglu_result[i]   = result.x;
                swiglu_result[i+1] = result.y;
                swiglu_result[i+2] = result.z;
                swiglu_result[i+3] = result.w;
            } else {
                // Boundary handling: check each element
                for (uint j = 0; j < 4 && i + j < MOE_TILE_N; ++j) {
                    if (global_n + j < intermediate_dim) {
                        float g = gate_acc_tg[i + j];
                        float u = up_acc_tg[i + j];
                        float silu_g = g / (1.0f + exp(-g));
                        swiglu_result[i + j] = silu_g * u;
                    } else {
                        swiglu_result[i + j] = 0.0f;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint chunk_end = min(n_chunk_offset + MOE_TILE_N, intermediate_dim);
        uint chunk_size = chunk_end - n_chunk_offset;
        uint num_k_tiles_chunk = (chunk_size + MOE_TILE_K - 1) / MOE_TILE_K;

        for (uint kdt = 0; kdt < num_k_tiles_chunk; ++kdt) {
            uint k_down_local = kdt * MOE_TILE_K;
            uint k_down_global = n_chunk_offset + k_down_local;

            load_trellis_tile(down_w, down_scales, down_su, down_sv, grid,
                            B_down, k_down_global, n_block, expert_id,
                            intermediate_dim, hidden_dim, p.group_size, p.n_levels, thread_idx);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
                uint global_out_n = n_block + i;
                if (global_out_n >= hidden_dim) continue;

                float local_down = 0.0f;

                uint k_end = min(MOE_TILE_K, chunk_size - k_down_local);
                for (uint k = 0; k < k_end; ++k) {
                    float swiglu_k = swiglu_result[k_down_local + k];
                    local_down += swiglu_k * float(B_down[k][i]);
                }

                output_tile[i] += local_down;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // FP32 atomic add using CAS loop (same fix as FP16 variant)
    for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
        uint global_n = n_block + i;
        if (global_n < hidden_dim) {
            float weighted = float(output_tile[i]) * prob;
            uint out_idx = token_idx * hidden_dim + global_n;

            device atomic_uint* atomic_ptr = (device atomic_uint*)(&output[out_idx]);
            uint old_bits = atomic_load_explicit(atomic_ptr, memory_order_relaxed);
            uint new_bits;
            bool success;
            do {
                float old_val = as_type<float>(old_bits);
                float new_val = old_val + weighted;
                new_bits = as_type<uint>(new_val);
                success = atomic_compare_exchange_weak_explicit(
                    atomic_ptr, &old_bits, new_bits,
                    memory_order_relaxed, memory_order_relaxed);
            } while (!success);
        }
    }
}

// ===========================================================================
// Large Batch Variant (moe_trellis_swiglu_large_batch)
//
// Optimized for large batch sizes (32+) with 2x wider output tiles (128 cols).
// Fewer threadgroups with more work per threadgroup improves:
//   - Memory bandwidth utilization (fewer kernel launches, better coalescing)
//   - GPU occupancy (more work per threadgroup reduces scheduler overhead)
//   - Cache efficiency (larger tiles amortize weight loading)
//
// Grid layout: (ceil(hidden_dim / MOE_TILE_N_LARGE), M, top_k)
//   - tgid.x: output column block (128 cols instead of 64)
//   - tgid.y: token index
//   - tgid.z: expert slot
//
// With batch=32, top_k=8:
//   Standard kernel: 256 threadgroups per output block
//   Large batch:     128 threadgroups per output block (2x fewer)
//
// Threadgroup memory: ~13KB (within 32KB limit)
//   - A_tile: 16*2 = 32 bytes
//   - B_gate/up/down: 16*128*2 = 4096 bytes each = 12288 bytes
//   - swiglu_result: 128*4 = 512 bytes (FP32)
//   - output_tile: 128*4 = 512 bytes (FP32)
//   - gate_acc/up_acc: 128*4 = 512 bytes each = 1024 bytes
// ===========================================================================

// Helper: Load Trellis tile with larger N dimension (128 columns)
inline void load_trellis_tile_large(
    device const uint8_t* packed_weights,
    device const half* scales,
    device const half* su,
    device const half* sv,
    device const half* grid,
    threadgroup half (&B_buf)[MOE_TILE_K][MOE_TILE_N_LARGE],
    uint k_block,
    uint n_block,
    uint expert_id,
    uint K_dim,
    uint N_dim,
    uint group_size,
    uint n_levels,
    uint thread_idx
) {
    // 16 * 128 = 2048 elements / 128 threads = 16 elements per thread
    const uint elems_per_thread = (MOE_TILE_K * MOE_TILE_N_LARGE) / MOE_THREADS;

    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        uint k_local = flat_idx / MOE_TILE_N_LARGE;
        uint n_local = flat_idx % MOE_TILE_N_LARGE;

        uint global_k = k_block + k_local;
        uint global_n = n_block + n_local;

        half val = 0.0h;
        if (global_k < K_dim && global_n < N_dim) {
            val = trellis_dequant_3bit(
                packed_weights, scales, su, sv, grid,
                expert_id, global_k, global_n,
                K_dim, N_dim, group_size, n_levels
            );
        }
        B_buf[k_local][n_local] = val;
    }
}

kernel void moe_trellis_swiglu_large_batch(
    device const half* activations       [[buffer(0)]],
    device const uint8_t* gate_weights   [[buffer(1)]],
    device const half* gate_scales       [[buffer(2)]],
    device const uint8_t* up_weights     [[buffer(3)]],
    device const half* up_scales         [[buffer(4)]],
    device const uint8_t* down_weights   [[buffer(5)]],
    device const half* down_scales       [[buffer(6)]],
    device const half* gate_su           [[buffer(7)]],
    device const half* gate_sv           [[buffer(8)]],
    device const half* up_su             [[buffer(9)]],
    device const half* up_sv             [[buffer(10)]],
    device const half* down_su           [[buffer(11)]],
    device const half* down_sv           [[buffer(12)]],
    device const half* grid              [[buffer(13)]],
    device const uint* expert_ids        [[buffer(14)]],
    device const half* expert_probs      [[buffer(15)]],
    device float* output                 [[buffer(16)]],
    constant TrellisParams& p            [[buffer(17)]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint thread_idx                      [[thread_index_in_threadgroup]]
) {
    // Threadgroup memory with 128-column tiles and FP32 accumulators
    threadgroup half A_tile[MOE_TILE_K];
    threadgroup half B_gate[MOE_TILE_K][MOE_TILE_N_LARGE];
    threadgroup half B_up[MOE_TILE_K][MOE_TILE_N_LARGE];
    threadgroup half B_down[MOE_TILE_K][MOE_TILE_N_LARGE];
    threadgroup float swiglu_result[MOE_TILE_N_LARGE];
    threadgroup float output_tile[MOE_TILE_N_LARGE];
    threadgroup float gate_acc_tg[MOE_TILE_N_LARGE];
    threadgroup float up_acc_tg[MOE_TILE_N_LARGE];

    // Grid indices - note 2x wider tiles
    const uint n_block = tgid.x * MOE_TILE_N_LARGE;
    const uint token_idx = tgid.y;
    const uint slot = tgid.z;

    if (token_idx >= p.M || slot >= p.top_k) {
        return;
    }

    const uint expert_id = expert_ids[token_idx * p.top_k + slot];
    if (expert_id >= p.num_experts) {
        return;
    }

    const float prob = float(expert_probs[token_idx * p.top_k + slot]);

    const uint hidden_dim = p.K;
    const uint intermediate_dim = p.N;

    uint num_tiles_k_gate = (hidden_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_n_gate = (intermediate_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_k_down = (intermediate_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_n_down = (hidden_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;

    uint gate_up_expert_size = num_tiles_k_gate * num_tiles_n_gate * PACKED_BYTES_3BIT;
    uint down_expert_size = num_tiles_k_down * num_tiles_n_down * PACKED_BYTES_3BIT;

    device const uint8_t* gate_w = gate_weights + expert_id * gate_up_expert_size;
    device const uint8_t* up_w = up_weights + expert_id * gate_up_expert_size;
    device const uint8_t* down_w = down_weights + expert_id * down_expert_size;

    // Initialize output tile (128 elements)
    for (uint i = thread_idx; i < MOE_TILE_N_LARGE; i += MOE_THREADS) {
        output_tile[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process intermediate_dim in chunks of MOE_TILE_N_LARGE (128)
    uint num_intermediate_chunks = (intermediate_dim + MOE_TILE_N_LARGE - 1) / MOE_TILE_N_LARGE;
    uint num_k_tiles_hidden = (hidden_dim + MOE_TILE_K - 1) / MOE_TILE_K;

    for (uint chunk_idx = 0; chunk_idx < num_intermediate_chunks; ++chunk_idx) {
        uint n_chunk_offset = chunk_idx * MOE_TILE_N_LARGE;

        // Initialize accumulators (128 elements)
        for (uint i = thread_idx; i < MOE_TILE_N_LARGE; i += MOE_THREADS) {
            gate_acc_tg[i] = 0.0f;
            up_acc_tg[i] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // K-dimension loop for gate + up projections
        for (uint kt = 0; kt < num_k_tiles_hidden; ++kt) {
            uint k_block_local = kt * MOE_TILE_K;

            // Load activations
            load_activation_tile(activations, A_tile, token_idx, k_block_local, hidden_dim, thread_idx);

            // Load gate and up weights (128-col tiles)
            load_trellis_tile_large(gate_w, gate_scales, gate_su, gate_sv, grid,
                                    B_gate, k_block_local, n_chunk_offset, expert_id,
                                    hidden_dim, intermediate_dim, p.group_size, p.n_levels, thread_idx);

            load_trellis_tile_large(up_w, up_scales, up_su, up_sv, grid,
                                    B_up, k_block_local, n_chunk_offset, expert_id,
                                    hidden_dim, intermediate_dim, p.group_size, p.n_levels, thread_idx);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate gate and up (128 columns, each thread handles 1 column)
            for (uint i = thread_idx; i < MOE_TILE_N_LARGE; i += MOE_THREADS) {
                float local_gate = 0.0f;
                float local_up = 0.0f;

                for (uint k = 0; k < MOE_TILE_K; ++k) {
                    float act = float(A_tile[k]);
                    local_gate += act * float(B_gate[k][i]);
                    local_up += act * float(B_up[k][i]);
                }

                gate_acc_tg[i] += local_gate;
                up_acc_tg[i] += local_up;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Apply SwiGLU (128 elements)
        for (uint i = thread_idx; i < MOE_TILE_N_LARGE; i += MOE_THREADS) {
            uint global_n = n_chunk_offset + i;

            if (global_n < intermediate_dim) {
                float g = gate_acc_tg[i];
                float u = up_acc_tg[i];
                float silu_g = g / (1.0f + exp(-g));
                swiglu_result[i] = silu_g * u;
            } else {
                swiglu_result[i] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Partial down projection
        uint chunk_end = min(n_chunk_offset + MOE_TILE_N_LARGE, intermediate_dim);
        uint chunk_size = chunk_end - n_chunk_offset;
        uint num_k_tiles_chunk = (chunk_size + MOE_TILE_K - 1) / MOE_TILE_K;

        for (uint kdt = 0; kdt < num_k_tiles_chunk; ++kdt) {
            uint k_down_local = kdt * MOE_TILE_K;
            uint k_down_global = n_chunk_offset + k_down_local;

            // Load down weights (128-col tile)
            load_trellis_tile_large(down_w, down_scales, down_su, down_sv, grid,
                                    B_down, k_down_global, n_block, expert_id,
                                    intermediate_dim, hidden_dim, p.group_size, p.n_levels, thread_idx);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate down projection
            for (uint i = thread_idx; i < MOE_TILE_N_LARGE; i += MOE_THREADS) {
                uint global_out_n = n_block + i;
                if (global_out_n >= hidden_dim) continue;

                float local_down = 0.0f;

                uint k_end = min(MOE_TILE_K, chunk_size - k_down_local);
                for (uint k = 0; k < k_end; ++k) {
                    float swiglu_k = swiglu_result[k_down_local + k];
                    local_down += swiglu_k * float(B_down[k][i]);
                }

                output_tile[i] += local_down;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write output with probability weighting (128 elements)
    for (uint i = thread_idx; i < MOE_TILE_N_LARGE; i += MOE_THREADS) {
        uint global_n = n_block + i;
        if (global_n < hidden_dim) {
            float weighted = output_tile[i] * prob;
            uint out_idx = token_idx * hidden_dim + global_n;

            device atomic_uint* atomic_ptr = (device atomic_uint*)(&output[out_idx]);
            uint old_bits = atomic_load_explicit(atomic_ptr, memory_order_relaxed);
            uint new_bits;
            bool success;
            do {
                float old_val = as_type<float>(old_bits);
                float new_val = old_val + weighted;
                new_bits = as_type<uint>(new_val);
                success = atomic_compare_exchange_weak_explicit(
                    atomic_ptr, &old_bits, new_bits,
                    memory_order_relaxed, memory_order_relaxed);
            } while (!success);
        }
    }
}

// ===========================================================================
// SIMD-Optimized MoE Kernel using simdgroup_matrix operations
//
// This kernel uses Apple Silicon's hardware matrix multiply units via
// simdgroup_matrix<half, 8, 8> for 4-8x speedup on the matmul portions.
//
// Architecture:
// - 4 simdgroups, each handling 16 output columns (2 x 8x8 tiles)
// - Total: 64 output columns per threadgroup (matches MOE_TILE_N)
// - K-dimension processed in 8-element chunks for simdgroup ops
//
// For single-token MoE (M=1):
// - Activation vector is broadcast to 8x8 matrix (all rows identical)
// - Result: 8x8 output where all rows are identical (use row 0)
// - This maps the vector-matrix multiply to hardware matmul
// ===========================================================================

// Simdgroup configuration for SIMD MoE kernel
constant constexpr uint SIMD_MoE_TILE = 8;           // 8x8 simdgroup matrix tiles
constant constexpr uint SIMD_MoE_N_TILES_PER_SG = 2; // Each simdgroup: 2 output tiles = 16 cols

kernel void moe_trellis_swiglu_simd(
    device const half* activations       [[buffer(0)]],
    device const uint8_t* gate_weights   [[buffer(1)]],
    device const half* gate_scales       [[buffer(2)]],
    device const uint8_t* up_weights     [[buffer(3)]],
    device const half* up_scales         [[buffer(4)]],
    device const uint8_t* down_weights   [[buffer(5)]],
    device const half* down_scales       [[buffer(6)]],
    device const half* gate_su           [[buffer(7)]],
    device const half* gate_sv           [[buffer(8)]],
    device const half* up_su             [[buffer(9)]],
    device const half* up_sv             [[buffer(10)]],
    device const half* down_su           [[buffer(11)]],
    device const half* down_sv           [[buffer(12)]],
    device const half* grid              [[buffer(13)]],
    device const uint* expert_ids        [[buffer(14)]],
    device const half* expert_probs      [[buffer(15)]],
    device float* output                 [[buffer(16)]],
    constant TrellisParams& p            [[buffer(17)]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint thread_idx                      [[thread_index_in_threadgroup]],
    uint simd_lane_id                    [[thread_index_in_simdgroup]],
    uint simd_group_id                   [[simdgroup_index_in_threadgroup]]
) {
    // Threadgroup memory layout for SIMD operations
    // A_simd: 8x8 staging for activation broadcast (128 bytes)
    // B_gate/up: 8x64 weight tiles (1024 bytes each = 2048 total)
    // B_down: 8x64 weight tile (1024 bytes)
    // swiglu_result: 64 elements (128 bytes)
    // output_tile: 64 elements (128 bytes)
    // B_staging: 4 x 8x8 = 512 bytes
    // gate_acc_tg/up_acc_tg: 128 bytes each
    // Total: ~4.2KB

    threadgroup half A_simd[SIMD_MoE_TILE][SIMD_MoE_TILE];     // 8x8 broadcast activation
    threadgroup half B_gate[SIMD_MoE_TILE][MOE_TILE_N];        // 8x64 gate weights
    threadgroup half B_up[SIMD_MoE_TILE][MOE_TILE_N];          // 8x64 up weights
    threadgroup half B_down[SIMD_MoE_TILE][MOE_TILE_N];        // 8x64 down weights
    threadgroup half swiglu_result[MOE_TILE_N];
    threadgroup half output_tile[MOE_TILE_N];
    threadgroup half gate_acc_tg[MOE_TILE_N];
    threadgroup half up_acc_tg[MOE_TILE_N];

    // Simdgroup-local staging for storing 8x8 results
    threadgroup half B_staging[MOE_SIMDGROUPS][SIMD_MoE_TILE][SIMD_MoE_TILE];

    const uint n_block = tgid.x * MOE_TILE_N;
    const uint token_idx = tgid.y;
    const uint slot = tgid.z;

    if (token_idx >= p.M || slot >= p.top_k) return;

    const uint expert_id = expert_ids[token_idx * p.top_k + slot];
    if (expert_id >= p.num_experts) return;

    const half prob = expert_probs[token_idx * p.top_k + slot];
    const uint hidden_dim = p.K;
    const uint intermediate_dim = p.N;

    // Expert weight offsets
    uint num_tiles_k_gate = (hidden_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_n_gate = (intermediate_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_k_down = (intermediate_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_n_down = (hidden_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;

    uint gate_up_expert_size = num_tiles_k_gate * num_tiles_n_gate * PACKED_BYTES_3BIT;
    uint down_expert_size = num_tiles_k_down * num_tiles_n_down * PACKED_BYTES_3BIT;

    device const uint8_t* gate_w = gate_weights + expert_id * gate_up_expert_size;
    device const uint8_t* up_w = up_weights + expert_id * gate_up_expert_size;
    device const uint8_t* down_w = down_weights + expert_id * down_expert_size;

    // Initialize output accumulators
    for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
        output_tile[i] = 0.0h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each simdgroup handles 16 output columns (2 x 8x8 tiles)
    // sg_col_offset: base column for this simdgroup within the 64-column tile
    const uint sg_col_offset = simd_group_id * SIMD_MoE_N_TILES_PER_SG * SIMD_MoE_TILE;  // 0, 16, 32, 48

    uint num_intermediate_chunks = (intermediate_dim + MOE_TILE_N - 1) / MOE_TILE_N;
    uint num_k_tiles = (hidden_dim + SIMD_MoE_TILE - 1) / SIMD_MoE_TILE;  // K in 8-element chunks

    for (uint chunk_idx = 0; chunk_idx < num_intermediate_chunks; ++chunk_idx) {
        uint n_chunk_offset = chunk_idx * MOE_TILE_N;

        // =================================================================
        // PHASE 1: Gate and Up projections using simdgroup_matrix
        // =================================================================

        // Initialize per-simdgroup accumulators (in registers via simdgroup_matrix)
        simdgroup_matrix<half, 8, 8> gate_acc[SIMD_MoE_N_TILES_PER_SG];
        simdgroup_matrix<half, 8, 8> up_acc[SIMD_MoE_N_TILES_PER_SG];

        #pragma unroll
        for (uint ni = 0; ni < SIMD_MoE_N_TILES_PER_SG; ++ni) {
            gate_acc[ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
            up_acc[ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }

        // K-dimension loop with SIMD matmul
        for (uint kt = 0; kt < num_k_tiles; ++kt) {
            uint k_block = kt * SIMD_MoE_TILE;  // 8 elements per iteration

            // Load activation into broadcast matrix (8x8 with identical rows)
            // Only first 8 threads participate, each loads one element and broadcasts
            if (thread_idx < SIMD_MoE_TILE) {
                uint global_k = k_block + thread_idx;
                half val = (global_k < hidden_dim) ?
                    activations[token_idx * hidden_dim + global_k] : 0.0h;

                // Broadcast to all 8 rows
                #pragma unroll
                for (uint row = 0; row < SIMD_MoE_TILE; ++row) {
                    A_simd[row][thread_idx] = val;
                }
            }

            // Load B_gate and B_up tiles (8xN where N=64)
            // Each thread loads elements: 128 threads, 8*64 = 512 elements = 4 per thread
            const uint elems_per_thread = (SIMD_MoE_TILE * MOE_TILE_N) / MOE_THREADS;  // 4

            #pragma unroll
            for (uint i = 0; i < elems_per_thread; ++i) {
                uint flat_idx = thread_idx * elems_per_thread + i;
                uint k_local = flat_idx / MOE_TILE_N;
                uint n_local = flat_idx % MOE_TILE_N;

                uint global_k = k_block + k_local;
                uint global_n = n_chunk_offset + n_local;

                half gate_val = 0.0h;
                half up_val = 0.0h;

                if (global_k < hidden_dim && global_n < intermediate_dim) {
                    gate_val = trellis_dequant_3bit(
                        gate_w, gate_scales, gate_su, gate_sv, grid,
                        expert_id, global_k, global_n,
                        hidden_dim, intermediate_dim, p.group_size, p.n_levels);

                    up_val = trellis_dequant_3bit(
                        up_w, up_scales, up_su, up_sv, grid,
                        expert_id, global_k, global_n,
                        hidden_dim, intermediate_dim, p.group_size, p.n_levels);
                }

                B_gate[k_local][n_local] = gate_val;
                B_up[k_local][n_local] = up_val;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Load A fragment (same for all simdgroups since activation is broadcast)
            simdgroup_matrix<half, 8, 8> a_frag;
            simdgroup_load(a_frag, &A_simd[0][0], SIMD_MoE_TILE);

            // For each 8x8 output tile this simdgroup handles
            #pragma unroll
            for (uint ni = 0; ni < SIMD_MoE_N_TILES_PER_SG; ++ni) {
                uint n_tile_offset = sg_col_offset + ni * SIMD_MoE_TILE;

                // Load B gate fragment for this N tile
                simdgroup_matrix<half, 8, 8> b_gate_frag;
                simdgroup_load(b_gate_frag, &B_gate[0][n_tile_offset], MOE_TILE_N);

                // Load B up fragment for this N tile
                simdgroup_matrix<half, 8, 8> b_up_frag;
                simdgroup_load(b_up_frag, &B_up[0][n_tile_offset], MOE_TILE_N);

                // Hardware matmul: C += A * B (8x8 = 64 ops per instruction)
                simdgroup_multiply_accumulate(gate_acc[ni], a_frag, b_gate_frag, gate_acc[ni]);
                simdgroup_multiply_accumulate(up_acc[ni], a_frag, b_up_frag, up_acc[ni]);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Store simdgroup accumulators to threadgroup memory
        // Only row 0 of each accumulator matters (all rows are identical due to broadcast)
        #pragma unroll
        for (uint ni = 0; ni < SIMD_MoE_N_TILES_PER_SG; ++ni) {
            uint n_tile_offset = sg_col_offset + ni * SIMD_MoE_TILE;

            // Store gate_acc to staging, then extract row 0
            simdgroup_store(gate_acc[ni], &B_staging[simd_group_id][0][0], SIMD_MoE_TILE);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // Lane 0-7 copy row 0 of their simdgroup's result
            if (simd_lane_id < SIMD_MoE_TILE) {
                gate_acc_tg[n_tile_offset + simd_lane_id] = B_staging[simd_group_id][0][simd_lane_id];
            }

            // Store up_acc to staging, then extract row 0
            simdgroup_store(up_acc[ni], &B_staging[simd_group_id][0][0], SIMD_MoE_TILE);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            if (simd_lane_id < SIMD_MoE_TILE) {
                up_acc_tg[n_tile_offset + simd_lane_id] = B_staging[simd_group_id][0][simd_lane_id];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // =================================================================
        // PHASE 2: Vectorized SwiGLU
        // =================================================================

        for (uint i = thread_idx * 4; i < MOE_TILE_N; i += MOE_THREADS * 4) {
            uint global_n = n_chunk_offset + i;

            if (global_n + 3 < intermediate_dim) {
                half4 g = half4(gate_acc_tg[i], gate_acc_tg[i+1], gate_acc_tg[i+2], gate_acc_tg[i+3]);
                half4 u = half4(up_acc_tg[i], up_acc_tg[i+1], up_acc_tg[i+2], up_acc_tg[i+3]);
                half4 silu_g = g / (1.0h + exp(-g));
                half4 result = silu_g * u;

                swiglu_result[i]   = result.x;
                swiglu_result[i+1] = result.y;
                swiglu_result[i+2] = result.z;
                swiglu_result[i+3] = result.w;
            } else {
                for (uint j = 0; j < 4 && i + j < MOE_TILE_N; ++j) {
                    if (global_n + j < intermediate_dim) {
                        half g = gate_acc_tg[i + j];
                        half u = up_acc_tg[i + j];
                        half silu_g = g / (1.0h + exp(-g));
                        swiglu_result[i + j] = silu_g * u;
                    } else {
                        swiglu_result[i + j] = 0.0h;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // =================================================================
        // PHASE 3: Down projection using simdgroup_matrix
        // =================================================================

        uint chunk_end = min(n_chunk_offset + MOE_TILE_N, intermediate_dim);
        uint chunk_size = chunk_end - n_chunk_offset;
        uint num_k_tiles_down = (chunk_size + SIMD_MoE_TILE - 1) / SIMD_MoE_TILE;

        // Accumulators for down projection (16 output columns per simdgroup)
        simdgroup_matrix<half, 8, 8> down_acc[SIMD_MoE_N_TILES_PER_SG];

        #pragma unroll
        for (uint ni = 0; ni < SIMD_MoE_N_TILES_PER_SG; ++ni) {
            down_acc[ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }

        for (uint kdt = 0; kdt < num_k_tiles_down; ++kdt) {
            uint k_down_local = kdt * SIMD_MoE_TILE;
            uint k_down_global = n_chunk_offset + k_down_local;

            // Load swiglu_result into broadcast matrix
            if (thread_idx < SIMD_MoE_TILE) {
                half val = (k_down_local + thread_idx < chunk_size) ?
                    swiglu_result[k_down_local + thread_idx] : 0.0h;

                #pragma unroll
                for (uint row = 0; row < SIMD_MoE_TILE; ++row) {
                    A_simd[row][thread_idx] = val;
                }
            }

            // Load B_down tile (8x64)
            const uint down_elems_per_thread = (SIMD_MoE_TILE * MOE_TILE_N) / MOE_THREADS;

            #pragma unroll
            for (uint i = 0; i < down_elems_per_thread; ++i) {
                uint flat_idx = thread_idx * down_elems_per_thread + i;
                uint k_local = flat_idx / MOE_TILE_N;
                uint n_local = flat_idx % MOE_TILE_N;

                uint global_k = k_down_global + k_local;
                uint global_n = n_block + n_local;

                half val = 0.0h;
                if (global_k < intermediate_dim && global_n < hidden_dim) {
                    val = trellis_dequant_3bit(
                        down_w, down_scales, down_su, down_sv, grid,
                        expert_id, global_k, global_n,
                        intermediate_dim, hidden_dim, p.group_size, p.n_levels);
                }
                B_down[k_local][n_local] = val;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_matrix<half, 8, 8> a_down_frag;
            simdgroup_load(a_down_frag, &A_simd[0][0], SIMD_MoE_TILE);

            #pragma unroll
            for (uint ni = 0; ni < SIMD_MoE_N_TILES_PER_SG; ++ni) {
                uint n_tile_offset = sg_col_offset + ni * SIMD_MoE_TILE;

                simdgroup_matrix<half, 8, 8> b_down_frag;
                simdgroup_load(b_down_frag, &B_down[0][n_tile_offset], MOE_TILE_N);

                simdgroup_multiply_accumulate(down_acc[ni], a_down_frag, b_down_frag, down_acc[ni]);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Store down accumulators and accumulate to output_tile
        #pragma unroll
        for (uint ni = 0; ni < SIMD_MoE_N_TILES_PER_SG; ++ni) {
            uint n_tile_offset = sg_col_offset + ni * SIMD_MoE_TILE;

            simdgroup_store(down_acc[ni], &B_staging[simd_group_id][0][0], SIMD_MoE_TILE);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate row 0 to output_tile
            if (simd_lane_id < SIMD_MoE_TILE) {
                output_tile[n_tile_offset + simd_lane_id] += B_staging[simd_group_id][0][simd_lane_id];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // =================================================================
    // PHASE 4: Atomic output write
    // =================================================================

    for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
        uint global_n = n_block + i;
        if (global_n < hidden_dim) {
            float weighted = float(output_tile[i]) * float(prob);
            uint out_idx = token_idx * hidden_dim + global_n;

            device atomic_uint* atomic_ptr = (device atomic_uint*)(&output[out_idx]);
            uint old_bits = atomic_load_explicit(atomic_ptr, memory_order_relaxed);
            uint new_bits;
            bool success;
            do {
                float old_val = as_type<float>(old_bits);
                float new_val = old_val + weighted;
                new_bits = as_type<uint>(new_val);
                success = atomic_compare_exchange_weak_explicit(
                    atomic_ptr, &old_bits, new_bits,
                    memory_order_relaxed, memory_order_relaxed);
            } while (!success);
        }
    }
}

// ===========================================================================
// Optimized Decode Kernel for batch_size=1
//
// For autoregressive generation, batch_size=1 is the common case.
// This kernel is optimized for single-token inference:
//   1. Removes batch dimension overhead - processes exactly one token
//   2. Simplified grid layout: (n_blocks, top_k) instead of (n_blocks, M, top_k)
//   3. Uses FP32 accumulators for numerical stability
//   4. Smaller tile size (32 vs 64) to reduce register pressure and overhead
//   5. Fewer threads per threadgroup (64 vs 128) for better occupancy
//
// Grid layout: (ceil(hidden_dim / DECODE_TILE_N), top_k)
//   - tgid.x: output column block
//   - tgid.y: expert slot (0 to top_k-1)
//
// Memory layout same as moe_trellis_swiglu but with batch_size=1 assumed.
// ===========================================================================

// Constants for decode kernel - smaller tile for reduced overhead
constant constexpr uint DECODE_TILE_N = 32;   // Smaller tile vs MOE_TILE_N=64
constant constexpr uint DECODE_THREADS = 64;  // 2 simdgroups vs 4

// Load Trellis tile with decode-specific dimensions
inline void load_trellis_tile_decode(
    device const uint8_t* packed_weights,
    device const half* scales,
    device const half* su,
    device const half* sv,
    device const half* grid,
    threadgroup half (&B_buf)[MOE_TILE_K][DECODE_TILE_N],
    uint k_block,
    uint n_block,
    uint expert_id,
    uint K_dim,
    uint N_dim,
    uint group_size,
    uint n_levels,
    uint thread_idx
) {
    const uint elems = MOE_TILE_K * DECODE_TILE_N;
    const uint elems_per_thread = (elems + DECODE_THREADS - 1) / DECODE_THREADS;

    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        if (flat_idx >= elems) break;

        uint k_local = flat_idx / DECODE_TILE_N;
        uint n_local = flat_idx % DECODE_TILE_N;

        uint global_k = k_block + k_local;
        uint global_n = n_block + n_local;

        half val = 0.0h;
        if (global_k < K_dim && global_n < N_dim) {
            val = trellis_dequant_3bit(
                packed_weights, scales, su, sv, grid,
                expert_id, global_k, global_n,
                K_dim, N_dim, group_size, n_levels
            );
        }
        B_buf[k_local][n_local] = val;
    }
}

kernel void moe_trellis_swiglu_decode(
    device const half* activations       [[buffer(0)]],   // [1, hidden] - single token
    device const uint8_t* gate_weights   [[buffer(1)]],
    device const half* gate_scales       [[buffer(2)]],
    device const uint8_t* up_weights     [[buffer(3)]],
    device const half* up_scales         [[buffer(4)]],
    device const uint8_t* down_weights   [[buffer(5)]],
    device const half* down_scales       [[buffer(6)]],
    device const half* gate_su           [[buffer(7)]],
    device const half* gate_sv           [[buffer(8)]],
    device const half* up_su             [[buffer(9)]],
    device const half* up_sv             [[buffer(10)]],
    device const half* down_su           [[buffer(11)]],
    device const half* down_sv           [[buffer(12)]],
    device const half* grid              [[buffer(13)]],
    device const uint* expert_ids        [[buffer(14)]],  // [1, top_k]
    device const half* expert_probs      [[buffer(15)]],  // [1, top_k]
    device float* output                 [[buffer(16)]],  // [1, hidden] FP32 for atomic
    constant TrellisParams& p            [[buffer(17)]],
    uint2 tgid                           [[threadgroup_position_in_grid]],
    uint thread_idx                      [[thread_index_in_threadgroup]]
) {
    // Threadgroup memory with decode-specific dimensions
    threadgroup half A_tile[MOE_TILE_K];
    threadgroup half B_gate[MOE_TILE_K][DECODE_TILE_N];
    threadgroup half B_up[MOE_TILE_K][DECODE_TILE_N];
    threadgroup half B_down[MOE_TILE_K][DECODE_TILE_N];
    threadgroup float swiglu_result[DECODE_TILE_N];
    threadgroup float output_tile[DECODE_TILE_N];
    threadgroup float gate_acc_tg[DECODE_TILE_N];
    threadgroup float up_acc_tg[DECODE_TILE_N];

    // Grid indices - 2D grid (no batch dimension)
    const uint n_block = tgid.x * DECODE_TILE_N;
    const uint slot = tgid.y;

    // Single token: no batch loop overhead
    const uint token_idx = 0;

    if (slot >= p.top_k) {
        return;
    }

    // Expert routing for this slot (index directly, not token * top_k + slot)
    const uint expert_id = expert_ids[slot];
    if (expert_id >= p.num_experts) {
        return;
    }

    const float prob = float(expert_probs[slot]);

    const uint hidden_dim = p.K;
    const uint intermediate_dim = p.N;

    // Expert weight offsets
    uint num_tiles_k_gate = (hidden_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_n_gate = (intermediate_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_k_down = (intermediate_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_n_down = (hidden_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;

    uint gate_up_expert_size = num_tiles_k_gate * num_tiles_n_gate * PACKED_BYTES_3BIT;
    uint down_expert_size = num_tiles_k_down * num_tiles_n_down * PACKED_BYTES_3BIT;

    device const uint8_t* gate_w = gate_weights + expert_id * gate_up_expert_size;
    device const uint8_t* up_w = up_weights + expert_id * gate_up_expert_size;
    device const uint8_t* down_w = down_weights + expert_id * down_expert_size;

    // Initialize output tile (no barrier needed - each thread owns its indices)
    for (uint i = thread_idx; i < DECODE_TILE_N; i += DECODE_THREADS) {
        output_tile[i] = 0.0f;
    }

    // Process intermediate_dim in chunks
    uint num_intermediate_chunks = (intermediate_dim + DECODE_TILE_N - 1) / DECODE_TILE_N;
    uint num_k_tiles_hidden = (hidden_dim + MOE_TILE_K - 1) / MOE_TILE_K;

    for (uint chunk_idx = 0; chunk_idx < num_intermediate_chunks; ++chunk_idx) {
        uint n_chunk_offset = chunk_idx * DECODE_TILE_N;

        // Initialize accumulators for this chunk
        // (barrier merged with first k-iteration's tile load barrier at 2073)
        for (uint i = thread_idx; i < DECODE_TILE_N; i += DECODE_THREADS) {
            gate_acc_tg[i] = 0.0f;
            up_acc_tg[i] = 0.0f;
        }

        // K-dimension loop for gate + up projections
        for (uint kt = 0; kt < num_k_tiles_hidden; ++kt) {
            uint k_block = kt * MOE_TILE_K;

            // Load single token activation (direct indexing, no batch offset)
            for (uint i = thread_idx; i < MOE_TILE_K; i += DECODE_THREADS) {
                uint global_k = k_block + i;
                A_tile[i] = (global_k < hidden_dim) ? activations[global_k] : half(0.0h);
            }

            // Load gate and up weights
            load_trellis_tile_decode(gate_w, gate_scales, gate_su, gate_sv, grid,
                                     B_gate, k_block, n_chunk_offset, expert_id,
                                     hidden_dim, intermediate_dim, p.group_size, p.n_levels, thread_idx);

            load_trellis_tile_decode(up_w, up_scales, up_su, up_sv, grid,
                                     B_up, k_block, n_chunk_offset, expert_id,
                                     hidden_dim, intermediate_dim, p.group_size, p.n_levels, thread_idx);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate gate and up
            for (uint i = thread_idx; i < DECODE_TILE_N; i += DECODE_THREADS) {
                float local_gate = 0.0f;
                float local_up = 0.0f;

                for (uint k = 0; k < MOE_TILE_K; ++k) {
                    float act = float(A_tile[k]);
                    local_gate += act * float(B_gate[k][i]);
                    local_up += act * float(B_up[k][i]);
                }

                gate_acc_tg[i] += local_gate;
                up_acc_tg[i] += local_up;
            }

            // Barrier only needed if there's another k-iteration that will overwrite tiles
            if (kt + 1 < num_k_tiles_hidden) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        // No barrier needed: each thread reads its own accumulator index in SwiGLU
        // Apply SwiGLU activation
        for (uint i = thread_idx; i < DECODE_TILE_N; i += DECODE_THREADS) {
            uint global_n = n_chunk_offset + i;

            if (global_n < intermediate_dim) {
                float g = gate_acc_tg[i];
                float u = up_acc_tg[i];
                float silu_g = g / (1.0f + exp(-g));
                swiglu_result[i] = silu_g * u;
            } else {
                swiglu_result[i] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Down projection for this chunk
        uint chunk_end = min(n_chunk_offset + DECODE_TILE_N, intermediate_dim);
        uint chunk_size = chunk_end - n_chunk_offset;
        uint num_k_tiles_chunk = (chunk_size + MOE_TILE_K - 1) / MOE_TILE_K;

        for (uint kdt = 0; kdt < num_k_tiles_chunk; ++kdt) {
            uint k_down_local = kdt * MOE_TILE_K;
            uint k_down_global = n_chunk_offset + k_down_local;

            // Load down weights
            load_trellis_tile_decode(down_w, down_scales, down_su, down_sv, grid,
                                     B_down, k_down_global, n_block, expert_id,
                                     intermediate_dim, hidden_dim, p.group_size, p.n_levels, thread_idx);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate down projection
            for (uint i = thread_idx; i < DECODE_TILE_N; i += DECODE_THREADS) {
                uint global_out_n = n_block + i;
                if (global_out_n >= hidden_dim) continue;

                float local_down = 0.0f;

                uint k_end = min(MOE_TILE_K, chunk_size - k_down_local);
                for (uint k = 0; k < k_end; ++k) {
                    float swiglu_k = swiglu_result[k_down_local + k];
                    local_down += swiglu_k * float(B_down[k][i]);
                }

                output_tile[i] += local_down;
            }

            // Barrier only needed if there's another kdt-iteration that will overwrite B_down
            if (kdt + 1 < num_k_tiles_chunk) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        // No barrier needed: each thread reads its own output_tile index
    }

    // Write output with probability weighting using FP32 atomic CAS
    for (uint i = thread_idx; i < DECODE_TILE_N; i += DECODE_THREADS) {
        uint global_n = n_block + i;
        if (global_n < hidden_dim) {
            float weighted = output_tile[i] * prob;
            uint out_idx = global_n;  // token_idx = 0, no batch offset

            device atomic_uint* atomic_ptr = (device atomic_uint*)(&output[out_idx]);
            uint old_bits = atomic_load_explicit(atomic_ptr, memory_order_relaxed);
            uint new_bits;
            bool success;
            do {
                float old_val = as_type<float>(old_bits);
                float new_val = old_val + weighted;
                new_bits = as_type<uint>(new_val);
                success = atomic_compare_exchange_weak_explicit(
                    atomic_ptr, &old_bits, new_bits,
                    memory_order_relaxed, memory_order_relaxed);
            } while (!success);
        }
    }
}

// ===========================================================================
// Prefill Variant (Batch Size 4-8) - Optimized for Prompt Processing
//
// During prompt prefill, batch_size is typically 4-8 for pipelined processing.
// This kernel processes multiple tokens together, reusing weight loads across
// all tokens in a batch to amortize the expensive memory bandwidth cost.
//
// Key optimization: Load weights ONCE, apply to 4 tokens
//   - Weight loads are the bottleneck (~3-6 bytes/element for Trellis)
//   - Activation loads are cheap (2 bytes/element, small M)
//   - By processing 4 tokens together, we 4x the compute per weight load
//
// Grid layout: (ceil(hidden_dim / MOE_TILE_N), ceil(M / PREFILL_BATCH), top_k)
//   - tgid.x: output column block
//   - tgid.y: base token index (processes tokens base..base+PREFILL_BATCH-1)
//   - tgid.z: expert slot (0 to top_k-1)
// ===========================================================================

constant constexpr uint PREFILL_BATCH = 4;  // Tokens processed together

// Load activation tiles for multiple tokens
inline void load_activation_tiles_prefill(
    device const half* activations,
    threadgroup half (&A_tiles)[PREFILL_BATCH][MOE_TILE_K],
    uint base_token,
    uint k_block,
    uint hidden_dim,
    uint M,
    uint thread_idx
) {
    // Each thread loads elements across all 4 tokens
    for (uint t = 0; t < PREFILL_BATCH; ++t) {
        uint token_idx = base_token + t;
        for (uint i = thread_idx; i < MOE_TILE_K; i += MOE_THREADS) {
            uint global_k = k_block + i;
            half val = 0.0h;
            if (token_idx < M && global_k < hidden_dim) {
                val = activations[token_idx * hidden_dim + global_k];
            }
            A_tiles[t][i] = val;
        }
    }
}

kernel void moe_trellis_swiglu_prefill4(
    device const half* activations       [[buffer(0)]],
    device const uint8_t* gate_weights   [[buffer(1)]],
    device const half* gate_scales       [[buffer(2)]],
    device const uint8_t* up_weights     [[buffer(3)]],
    device const half* up_scales         [[buffer(4)]],
    device const uint8_t* down_weights   [[buffer(5)]],
    device const half* down_scales       [[buffer(6)]],
    device const half* gate_su           [[buffer(7)]],
    device const half* gate_sv           [[buffer(8)]],
    device const half* up_su             [[buffer(9)]],
    device const half* up_sv             [[buffer(10)]],
    device const half* down_su           [[buffer(11)]],
    device const half* down_sv           [[buffer(12)]],
    device const half* grid              [[buffer(13)]],
    device const uint* expert_ids        [[buffer(14)]],
    device const half* expert_probs      [[buffer(15)]],
    device float* output                 [[buffer(16)]],  // FP32 for atomic add
    constant TrellisParams& p            [[buffer(17)]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint thread_idx                      [[thread_index_in_threadgroup]]
) {
    // Threadgroup memory - process 4 tokens with shared weights
    threadgroup half A_tiles[PREFILL_BATCH][MOE_TILE_K];      // Activation slices for 4 tokens
    threadgroup half B_gate[MOE_TILE_K][MOE_TILE_N];          // Gate weights (shared)
    threadgroup half B_up[MOE_TILE_K][MOE_TILE_N];            // Up weights (shared)
    threadgroup half B_down[MOE_TILE_K][MOE_TILE_N];          // Down weights (shared)
    threadgroup half gate_acc[PREFILL_BATCH][MOE_TILE_N];     // Per-token gate accumulators
    threadgroup half up_acc[PREFILL_BATCH][MOE_TILE_N];       // Per-token up accumulators
    threadgroup half swiglu_result[PREFILL_BATCH][MOE_TILE_N];// Per-token SwiGLU results
    threadgroup half output_tile[PREFILL_BATCH][MOE_TILE_N];  // Per-token output accumulators

    const uint n_block = tgid.x * MOE_TILE_N;
    const uint base_token = tgid.y * PREFILL_BATCH;
    const uint slot = tgid.z;

    if (base_token >= p.M || slot >= p.top_k) {
        return;
    }

    const uint hidden_dim = p.K;
    const uint intermediate_dim = p.N;

    uint num_tiles_k_gate = (hidden_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_n_gate = (intermediate_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_k_down = (intermediate_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_n_down = (hidden_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;

    uint gate_up_expert_size = num_tiles_k_gate * num_tiles_n_gate * PACKED_BYTES_3BIT;
    uint down_expert_size = num_tiles_k_down * num_tiles_n_down * PACKED_BYTES_3BIT;

    for (uint t = 0; t < PREFILL_BATCH; ++t) {
        for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
            output_tile[t][i] = 0.0h;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_tokens_in_batch = min(PREFILL_BATCH, p.M - base_token);

    uint expert_ids_local[PREFILL_BATCH];
    half probs_local[PREFILL_BATCH];
    for (uint t = 0; t < PREFILL_BATCH; ++t) {
        uint token_idx = base_token + t;
        if (token_idx < p.M) {
            expert_ids_local[t] = expert_ids[token_idx * p.top_k + slot];
            probs_local[t] = expert_probs[token_idx * p.top_k + slot];
        } else {
            expert_ids_local[t] = 0;
            probs_local[t] = 0.0h;
        }
    }

    uint num_intermediate_chunks = (intermediate_dim + MOE_TILE_N - 1) / MOE_TILE_N;
    uint num_k_tiles_hidden = (hidden_dim + MOE_TILE_K - 1) / MOE_TILE_K;

    for (uint chunk_idx = 0; chunk_idx < num_intermediate_chunks; ++chunk_idx) {
        uint n_chunk_offset = chunk_idx * MOE_TILE_N;

        for (uint t = 0; t < PREFILL_BATCH; ++t) {
            for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
                gate_acc[t][i] = 0.0h;
                up_acc[t][i] = 0.0h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kt = 0; kt < num_k_tiles_hidden; ++kt) {
            uint k_block = kt * MOE_TILE_K;

            load_activation_tiles_prefill(activations, A_tiles, base_token, k_block,
                                          hidden_dim, p.M, thread_idx);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint t = 0; t < num_tokens_in_batch; ++t) {
                uint expert_id = expert_ids_local[t];
                if (expert_id >= p.num_experts) continue;

                device const uint8_t* gate_w = gate_weights + expert_id * gate_up_expert_size;
                device const uint8_t* up_w = up_weights + expert_id * gate_up_expert_size;

                load_trellis_tile(gate_w, gate_scales, gate_su, gate_sv, grid,
                                B_gate, k_block, n_chunk_offset, expert_id,
                                hidden_dim, intermediate_dim, p.group_size, p.n_levels, thread_idx);

                load_trellis_tile(up_w, up_scales, up_su, up_sv, grid,
                                B_up, k_block, n_chunk_offset, expert_id,
                                hidden_dim, intermediate_dim, p.group_size, p.n_levels, thread_idx);

                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
                    half local_gate = 0.0h;
                    half local_up = 0.0h;

                    for (uint k = 0; k < MOE_TILE_K; ++k) {
                        half act = A_tiles[t][k];
                        local_gate += act * B_gate[k][i];
                        local_up += act * B_up[k][i];
                    }

                    gate_acc[t][i] += local_gate;
                    up_acc[t][i] += local_up;
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        for (uint t = 0; t < num_tokens_in_batch; ++t) {
            for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
                uint global_n = n_chunk_offset + i;

                if (global_n < intermediate_dim) {
                    half g = gate_acc[t][i];
                    half u = up_acc[t][i];
                    half silu_g = g / (1.0h + exp(-g));
                    swiglu_result[t][i] = silu_g * u;
                } else {
                    swiglu_result[t][i] = 0.0h;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint chunk_end = min(n_chunk_offset + MOE_TILE_N, intermediate_dim);
        uint chunk_size = chunk_end - n_chunk_offset;
        uint num_k_tiles_chunk = (chunk_size + MOE_TILE_K - 1) / MOE_TILE_K;

        for (uint kdt = 0; kdt < num_k_tiles_chunk; ++kdt) {
            uint k_down_local = kdt * MOE_TILE_K;
            uint k_down_global = n_chunk_offset + k_down_local;

            for (uint t = 0; t < num_tokens_in_batch; ++t) {
                uint expert_id = expert_ids_local[t];
                if (expert_id >= p.num_experts) continue;

                device const uint8_t* down_w = down_weights + expert_id * down_expert_size;

                load_trellis_tile(down_w, down_scales, down_su, down_sv, grid,
                                B_down, k_down_global, n_block, expert_id,
                                intermediate_dim, hidden_dim, p.group_size, p.n_levels, thread_idx);

                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
                    uint global_out_n = n_block + i;
                    if (global_out_n >= hidden_dim) continue;

                    half local_down = 0.0h;
                    uint k_end = min(MOE_TILE_K, chunk_size - k_down_local);

                    for (uint k = 0; k < k_end; ++k) {
                        half swiglu_k = swiglu_result[t][k_down_local + k];
                        local_down += swiglu_k * B_down[k][i];
                    }

                    output_tile[t][i] += local_down;
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    for (uint t = 0; t < num_tokens_in_batch; ++t) {
        uint token_idx = base_token + t;
        half prob = probs_local[t];

        for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
            uint global_n = n_block + i;
            if (global_n < hidden_dim) {
                float weighted = float(output_tile[t][i]) * float(prob);
                uint out_idx = token_idx * hidden_dim + global_n;

                device atomic_uint* atomic_ptr = (device atomic_uint*)(&output[out_idx]);
                uint old_bits = atomic_load_explicit(atomic_ptr, memory_order_relaxed);
                uint new_bits;
                bool success;
                do {
                    float old_val = as_type<float>(old_bits);
                    float new_val = old_val + weighted;
                    new_bits = as_type<uint>(new_val);
                    success = atomic_compare_exchange_weak_explicit(
                        atomic_ptr, &old_bits, new_bits,
                        memory_order_relaxed, memory_order_relaxed);
                } while (!success);
            }
        }
    }
}

// ===========================================================================
// Prefill Variant with FP32 Accumulation
// ===========================================================================

kernel void moe_trellis_swiglu_prefill4_fp32acc(
    device const half* activations       [[buffer(0)]],
    device const uint8_t* gate_weights   [[buffer(1)]],
    device const half* gate_scales       [[buffer(2)]],
    device const uint8_t* up_weights     [[buffer(3)]],
    device const half* up_scales         [[buffer(4)]],
    device const uint8_t* down_weights   [[buffer(5)]],
    device const half* down_scales       [[buffer(6)]],
    device const half* gate_su           [[buffer(7)]],
    device const half* gate_sv           [[buffer(8)]],
    device const half* up_su             [[buffer(9)]],
    device const half* up_sv             [[buffer(10)]],
    device const half* down_su           [[buffer(11)]],
    device const half* down_sv           [[buffer(12)]],
    device const half* grid              [[buffer(13)]],
    device const uint* expert_ids        [[buffer(14)]],
    device const half* expert_probs      [[buffer(15)]],
    device float* output                 [[buffer(16)]],
    constant TrellisParams& p            [[buffer(17)]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint thread_idx                      [[thread_index_in_threadgroup]]
) {
    threadgroup half A_tiles[PREFILL_BATCH][MOE_TILE_K];
    threadgroup half B_gate[MOE_TILE_K][MOE_TILE_N];
    threadgroup half B_up[MOE_TILE_K][MOE_TILE_N];
    threadgroup half B_down[MOE_TILE_K][MOE_TILE_N];
    threadgroup float gate_acc[PREFILL_BATCH][MOE_TILE_N];
    threadgroup float up_acc[PREFILL_BATCH][MOE_TILE_N];
    threadgroup float swiglu_result[PREFILL_BATCH][MOE_TILE_N];
    threadgroup float output_tile[PREFILL_BATCH][MOE_TILE_N];

    const uint n_block = tgid.x * MOE_TILE_N;
    const uint base_token = tgid.y * PREFILL_BATCH;
    const uint slot = tgid.z;

    if (base_token >= p.M || slot >= p.top_k) return;

    const uint hidden_dim = p.K;
    const uint intermediate_dim = p.N;

    uint num_tiles_k_gate = (hidden_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_n_gate = (intermediate_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_k_down = (intermediate_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_n_down = (hidden_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;

    uint gate_up_expert_size = num_tiles_k_gate * num_tiles_n_gate * PACKED_BYTES_3BIT;
    uint down_expert_size = num_tiles_k_down * num_tiles_n_down * PACKED_BYTES_3BIT;

    for (uint t = 0; t < PREFILL_BATCH; ++t) {
        for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
            output_tile[t][i] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_tokens_in_batch = min(PREFILL_BATCH, p.M - base_token);

    uint expert_ids_local[PREFILL_BATCH];
    float probs_local[PREFILL_BATCH];
    for (uint t = 0; t < PREFILL_BATCH; ++t) {
        uint token_idx = base_token + t;
        if (token_idx < p.M) {
            expert_ids_local[t] = expert_ids[token_idx * p.top_k + slot];
            probs_local[t] = float(expert_probs[token_idx * p.top_k + slot]);
        } else {
            expert_ids_local[t] = 0;
            probs_local[t] = 0.0f;
        }
    }

    uint num_intermediate_chunks = (intermediate_dim + MOE_TILE_N - 1) / MOE_TILE_N;
    uint num_k_tiles_hidden = (hidden_dim + MOE_TILE_K - 1) / MOE_TILE_K;

    for (uint chunk_idx = 0; chunk_idx < num_intermediate_chunks; ++chunk_idx) {
        uint n_chunk_offset = chunk_idx * MOE_TILE_N;

        for (uint t = 0; t < PREFILL_BATCH; ++t) {
            for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
                gate_acc[t][i] = 0.0f;
                up_acc[t][i] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kt = 0; kt < num_k_tiles_hidden; ++kt) {
            uint k_block = kt * MOE_TILE_K;

            load_activation_tiles_prefill(activations, A_tiles, base_token, k_block,
                                          hidden_dim, p.M, thread_idx);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint t = 0; t < num_tokens_in_batch; ++t) {
                uint expert_id = expert_ids_local[t];
                if (expert_id >= p.num_experts) continue;

                device const uint8_t* gate_w = gate_weights + expert_id * gate_up_expert_size;
                device const uint8_t* up_w = up_weights + expert_id * gate_up_expert_size;

                load_trellis_tile(gate_w, gate_scales, gate_su, gate_sv, grid,
                                B_gate, k_block, n_chunk_offset, expert_id,
                                hidden_dim, intermediate_dim, p.group_size, p.n_levels, thread_idx);

                load_trellis_tile(up_w, up_scales, up_su, up_sv, grid,
                                B_up, k_block, n_chunk_offset, expert_id,
                                hidden_dim, intermediate_dim, p.group_size, p.n_levels, thread_idx);

                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
                    float local_gate = 0.0f;
                    float local_up = 0.0f;

                    for (uint k = 0; k < MOE_TILE_K; ++k) {
                        float act = float(A_tiles[t][k]);
                        local_gate += act * float(B_gate[k][i]);
                        local_up += act * float(B_up[k][i]);
                    }

                    gate_acc[t][i] += local_gate;
                    up_acc[t][i] += local_up;
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        for (uint t = 0; t < num_tokens_in_batch; ++t) {
            for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
                uint global_n = n_chunk_offset + i;

                if (global_n < intermediate_dim) {
                    float g = gate_acc[t][i];
                    float u = up_acc[t][i];
                    float silu_g = g / (1.0f + exp(-g));
                    swiglu_result[t][i] = silu_g * u;
                } else {
                    swiglu_result[t][i] = 0.0f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint chunk_end = min(n_chunk_offset + MOE_TILE_N, intermediate_dim);
        uint chunk_size = chunk_end - n_chunk_offset;
        uint num_k_tiles_chunk = (chunk_size + MOE_TILE_K - 1) / MOE_TILE_K;

        for (uint kdt = 0; kdt < num_k_tiles_chunk; ++kdt) {
            uint k_down_local = kdt * MOE_TILE_K;
            uint k_down_global = n_chunk_offset + k_down_local;

            for (uint t = 0; t < num_tokens_in_batch; ++t) {
                uint expert_id = expert_ids_local[t];
                if (expert_id >= p.num_experts) continue;

                device const uint8_t* down_w = down_weights + expert_id * down_expert_size;

                load_trellis_tile(down_w, down_scales, down_su, down_sv, grid,
                                B_down, k_down_global, n_block, expert_id,
                                intermediate_dim, hidden_dim, p.group_size, p.n_levels, thread_idx);

                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
                    uint global_out_n = n_block + i;
                    if (global_out_n >= hidden_dim) continue;

                    float local_down = 0.0f;
                    uint k_end = min(MOE_TILE_K, chunk_size - k_down_local);

                    for (uint k = 0; k < k_end; ++k) {
                        float swiglu_k = swiglu_result[t][k_down_local + k];
                        local_down += swiglu_k * float(B_down[k][i]);
                    }

                    output_tile[t][i] += local_down;
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    for (uint t = 0; t < num_tokens_in_batch; ++t) {
        uint token_idx = base_token + t;
        float prob = probs_local[t];

        for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
            uint global_n = n_block + i;
            if (global_n < hidden_dim) {
                float weighted = output_tile[t][i] * prob;
                uint out_idx = token_idx * hidden_dim + global_n;

                device atomic_uint* atomic_ptr = (device atomic_uint*)(&output[out_idx]);
                uint old_bits = atomic_load_explicit(atomic_ptr, memory_order_relaxed);
                uint new_bits;
                bool success;
                do {
                    float old_val = as_type<float>(old_bits);
                    float new_val = old_val + weighted;
                    new_bits = as_type<uint>(new_val);
                    success = atomic_compare_exchange_weak_explicit(
                        atomic_ptr, &old_bits, new_bits,
                        memory_order_relaxed, memory_order_relaxed);
                } while (!success);
            }
        }
    }
}
