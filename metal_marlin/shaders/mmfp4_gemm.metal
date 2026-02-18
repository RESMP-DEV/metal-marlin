#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Aggressive inlining for prefill performance
#define METAL_M_ALWAYS_INLINE __attribute__((always_inline))

// Scale cache size for register prefetching - reduces device memory bandwidth
// Each thread caches scales in registers for its assigned columns
// A fixed size of 4 is used, which corresponds to the worst-case scenario
// for TILE_K=32 (e.g., group_size=8 requires 4 scales). This allows the
// compiler to fully unroll the prefetching loop.
//
// OPTIMIZATION: Using uint (32-bit) to store pairs of half-precision scales
// enables vectorized loads (2 scales per load) and better register packing.
// This reduces global memory instructions by 50% for scale fetching.
constant constexpr uint SCALE_CACHE_SIZE = 4;

// --- FP4 E2M1 branchless dequant (pure ALU, no LUT) ---
METAL_M_ALWAYS_INLINE
inline half dequant_fp4_scalar(uint nibble) {
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;

    half sub_mag = half(man_bit) * half(0.5h);
    half norm_mag = half(1u << (exp_bits - 1)) * (half(1.0h) + half(man_bit) * half(0.5h));
    half magnitude = select(norm_mag, sub_mag, exp_bits == 0);
    return select(magnitude, -magnitude, bool(sign_bit));
}

// NOTE: Uses float scale to work around Metal compiler bug where
// half parameters in inline functions have fractional parts rounded.
METAL_M_ALWAYS_INLINE
inline void dequant_fp4x8(uint32_t packed, half scale, thread half *out) {
    float fscale = (float)scale;
    out[0] = (half)((float)dequant_fp4_scalar((packed >>  0) & 0xF) * fscale);
    out[1] = (half)((float)dequant_fp4_scalar((packed >>  4) & 0xF) * fscale);
    out[2] = (half)((float)dequant_fp4_scalar((packed >>  8) & 0xF) * fscale);
    out[3] = (half)((float)dequant_fp4_scalar((packed >> 12) & 0xF) * fscale);
    out[4] = (half)((float)dequant_fp4_scalar((packed >> 16) & 0xF) * fscale);
    out[5] = (half)((float)dequant_fp4_scalar((packed >> 20) & 0xF) * fscale);
    out[6] = (half)((float)dequant_fp4_scalar((packed >> 24) & 0xF) * fscale);
    out[7] = (half)((float)dequant_fp4_scalar((packed >> 28) & 0xF) * fscale);
}

// Cooperative load helper for A_tile (64x32)
// Uses vectorized loads (ulong2 = 16 bytes = 8 halfs) for improved bandwidth
METAL_M_ALWAYS_INLINE
inline void threadgroup_load_A(
    device const half* A,
    threadgroup half (*A_tile)[32],
    uint tgid_y,
    uint k_tile,
    uint thread_idx,
    uint m_dim,
    uint k_dim
) {
    // 128 threads load 64x32 = 2048 elements. Each thread loads 16 elements.
    uint flat_idx = thread_idx * 16;
    uint r = flat_idx / 32;
    uint c = flat_idx % 32; // 0 or 16

    uint global_row = tgid_y * 64 + r;

    // Boundary check for M dimension
    // When M is small (e.g., M=4), most threads will be beyond valid rows
    if (global_row >= m_dim) {
        // Zero out the 16 elements this thread is responsible for.
        threadgroup ulong2* dst_ptr = (threadgroup ulong2*)(&A_tile[r][c]);
        dst_ptr[0] = 0;
        dst_ptr[1] = 0;
        return;
    }

    uint global_col = k_tile + c;

    // Vectorized load path - only when fully in bounds for both M and K
    // Check: global_row is valid (checked above), global_col+15 is valid in K,
    // and we're not at risk of reading beyond the row in a way that causes issues
    if (global_col + 16 <= k_dim && global_row < m_dim) {
        const device ulong2* src_ptr = (const device ulong2*)(A + global_row * k_dim + global_col);
        threadgroup ulong2* dst_ptr = (threadgroup ulong2*)(&A_tile[r][c]);
        dst_ptr[0] = src_ptr[0];
        dst_ptr[1] = src_ptr[1];
    } else {
        // Scalar fallback for K dimension boundaries or when near M boundary
        // Zero-pad if out of bounds
        for (uint i = 0; i < 16; ++i) {
            uint curr_c = c + i;
            uint curr_global_col = k_tile + curr_c;
            half val = half(0.0h);
            if (global_row < m_dim && curr_global_col < k_dim) {
                val = A[global_row * k_dim + curr_global_col];
            }
            A_tile[r][curr_c] = val;
        }
    }
}



kernel void mmfp4_gemm(
    device const half* A [[buffer(0)]],
    device const uint* B_packed [[buffer(1)]],
    device const half* B_scales [[buffer(2)]],
    device half* C [[buffer(3)]],
    device const uint* M_p [[buffer(4)]],
    device const uint* K_p [[buffer(5)]],
    device const uint* N_p [[buffer(6)]],
    device const uint* group_size_p [[buffer(7)]],
    device const uint* n_groups_p [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]])
{
#if M == 1
    // M=1 decode specialization (GEMV)
    // Optimized for single-token decode where M=1.
    // This path avoids threadgroup memory and simdgroup matrix operations, using a
    // streaming approach for higher performance. Each thread computes 2 output values.

    const uint K = K_p[0];
    const uint N = N_p[0];
    const uint GROUP_SIZE = group_size_p[0];
    const uint N_GROUPS = n_groups_p[0];

    const uint TILE_N = 256;
    const uint COLS_PER_THREAD = 2;
    const uint FP4_PER_UINT = 8;

    uint tgid_x = tgid.x;
    uint tid_x = tid.x;

    uint col_base = tgid_x * TILE_N + tid_x * COLS_PER_THREAD;
    uint col0 = col_base;
    uint col1 = col_base + 1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    uint k_packs = K / FP4_PER_UINT;

    // OPTIMIZATION: Scale cache with register prefetching
    // Keeps scales in registers to reduce global memory bandwidth
    uint cached_group_idx = 0xFFFFFFFFu;
    half cached_scale0 = 0.0h;
    half cached_scale1 = 0.0h;

    // Stream through K dimension
    for (uint k_base = 0; k_base < K; k_base += FP4_PER_UINT) {
        uint pack_idx = k_base / FP4_PER_UINT;
        uint group_idx = k_base / GROUP_SIZE;

        // Load 8 A values (shared across all columns)
        // Since M=1, A is just a vector of size K.
        half a_vals[8];
        #pragma unroll
        for (uint i = 0; i < 8; i++) {
            uint k_idx = k_base + i;
            a_vals[i] = (k_idx < K) ? A[k_idx] : half(0.0h);
        }

        // OPTIMIZATION: Update cached scales only when group changes
        // This amortizes scale memory access across the K dimension
        if (group_idx != cached_group_idx) {
            if (col0 < N) cached_scale0 = B_scales[col0 * N_GROUPS + group_idx];
            if (col1 < N) cached_scale1 = B_scales[col1 * N_GROUPS + group_idx];
            cached_group_idx = group_idx;
        }

        // Process column 0 with fully unrolled dot product
        if (col0 < N && pack_idx < k_packs) {
            uint packed0 = B_packed[pack_idx * N + col0];
            
            if (packed0 != 0) {
                float fscale0 = (float)cached_scale0;

                #pragma unroll
                for (uint i = 0; i < 8; i++) {
                    uint nibble = (packed0 >> (i * 4)) & 0xF;
                    half w_val = (half)((float)dequant_fp4_scalar(nibble) * fscale0);
                    acc0 += float(a_vals[i]) * float(w_val);
                }
            }
        }

        // Process column 1 with fully unrolled dot product
        if (col1 < N && pack_idx < k_packs) {
            uint packed1 = B_packed[pack_idx * N + col1];

            if (packed1 != 0) {
                float fscale1 = (float)cached_scale1;

                #pragma unroll
                for (uint i = 0; i < 8; i++) {
                    uint nibble = (packed1 >> (i * 4)) & 0xF;
                    half w_val = (half)((float)dequant_fp4_scalar(nibble) * fscale1);
                    acc1 += float(a_vals[i]) * float(w_val);
                }
            }
        }
    }

    // Store results
    if (col0 < N) {
        C[col0] = half(acc0);
    }
    if (col1 < N) {
        C[col1] = half(acc1);
    }
#elif defined(M) && M > 16
    // M>16 prefill specialization - OPTIMIZED for high throughput on Apple Silicon
    // Compile-time constants enable aggressive loop unrolling and SIMD optimization
    const uint K = K_p[0];
    const uint N = N_p[0];
    const uint GROUP_SIZE = group_size_p[0];

    const uint TILE_M = 64;
    const uint TILE_N = 64;
    const uint TILE_K = 32;
    const uint SIMDGROUPS_PER_TG = 4;
    const uint THREADS_PER_TG = SIMDGROUPS_PER_TG * 32;  // 128 threads
    const uint FP4_PER_UINT = 8;
    
    // Precompute constants for compile-time optimization
    const uint K_BLOCKS_PER_TILE = TILE_K / FP4_PER_UINT;  // 4 blocks

    uint simd_lane = tid.x % 32;
    uint simd_id = tid.x / 32;
    uint thread_idx = tid.x;
    
    // TILE_M EDGE CASE HANDLING: Early exit for tiles completely outside valid M range
    // This prevents unnecessary computation and potential Inf/NaN in edge tiles
    // when M is not a multiple of TILE_M (e.g., M=17..63, 65..127, etc.)
    uint tile_start_row = tgid.y * TILE_M;
    if (tile_start_row >= M) {
        // Entire tile is outside valid M range - nothing to compute
        return;
    }
    
    // PARTIAL TILE HANDLING: Check if this simdgroup has any valid rows to compute
    // SIMD group layout: 2x2 grid covering 64x64 tile
    // simd_id 0: rows 0-31, simd_id 1: rows 0-31, simd_id 2: rows 32-63, simd_id 3: rows 32-63
    uint sg_row_base = (simd_id / 2) * 32;
    uint simd_start_row = tile_start_row + sg_row_base;
    if (simd_start_row >= M) {
        // This entire simdgroup is beyond valid M range - nothing to compute
        return;
    }
    // Track if this simdgroup is partially valid (some rows valid, some not)
    bool simd_is_partial = (simd_start_row + 32 > M);

    // Aligned threadgroup memory for optimal SIMD access patterns
    threadgroup half A_tile[TILE_M][TILE_K] __attribute__((aligned(128)));
    threadgroup half B_tile[TILE_K][TILE_N] __attribute__((aligned(128)));

    // simd_id 1: top-right (0,32), 3: bottom-right (32,32)
    uint sg_col_base = (simd_id % 2) * 32;

    // Use float accumulators to prevent overflow during accumulation
    // Each simdgroup computes a 32x32 sub-tile using 4x4 8x8 matrices
    simdgroup_matrix<float, 8, 8> acc[4][4];
    #pragma unroll
    for (uint mi = 0; mi < 4; ++mi)
        #pragma unroll
        for (uint ni = 0; ni < 4; ++ni)
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // Main K-loop: Process TILE_K elements at a time
    for (uint k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Load A_tile from device to threadgroup memory
        // Uses vectorized loads (ulong2) for improved bandwidth
        threadgroup_load_A(A, A_tile, tgid.y, k_tile, thread_idx, M, K);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- B-Tile Population with FP4 Dequantization ---
        // Each thread processes columns in strided fashion for coalesced memory access
        #pragma unroll 2
        for (uint b_col = thread_idx; b_col < TILE_N; b_col += THREADS_PER_TG) {
            uint global_col = tgid.x * TILE_N + b_col;

            // Prefetch scales for the current column into registers
            // Scale cache eliminates redundant global memory loads
            // OPTIMIZED: Preload scales into registers before the K-loop to maximize ILP
            // Pack 2 halves into 1 uint for better register utilization
            // Use explicit variables to force register allocation (arrays can spill)
            uint scale_cache_0 = 0;
            uint scale_cache_1 = 0;
            
            // PREFETCH: Aggressive register prefetching of scales with cooperative loading
            // Load all required scales upfront before entering the dequantization loop
            // Cooperative load: Even threads load 2 scales (uint), share with Odd threads
            // This reduces global memory instructions by 50%
            {
                uint group_base = k_tile / GROUP_SIZE;
                const uint N_GROUPS = n_groups_p[0];
                
                #pragma unroll
                for (uint i = 0; i < SCALE_CACHE_SIZE; i += 2) {
                    half s0 = 0.0h;
                    half s1 = 0.0h;

                    // Load s0 (group i)
                    if (SCALE_CACHE_SIZE > i && (group_base + i) < N_GROUPS) {
                         uint raw = 0;
                         if ((simd_lane & 1) == 0 && global_col < N) {
                             // Even threads load pairs [col, col+1] packed in uint
                             raw = *((device const uint*)(B_scales + (group_base + i) * N + global_col));
                         }
                         // Shuffle to neighbor
                         uint neighbor = simd_shuffle_xor(raw, 1);
                         if ((simd_lane & 1) != 0) raw = neighbor;
                         
                         // Unpack: Even thread gets low, Odd thread gets high
                         half2 up = as_type<half2>(raw);
                         s0 = ((simd_lane & 1) == 0) ? up.x : up.y;
                    }

                    // Load s1 (group i+1)
                    if (SCALE_CACHE_SIZE > i + 1 && (group_base + i + 1) < N_GROUPS) {
                         uint raw = 0;
                         if ((simd_lane & 1) == 0 && global_col < N) {
                             raw = *((device const uint*)(B_scales + (group_base + i + 1) * N + global_col));
                         }
                         uint neighbor = simd_shuffle_xor(raw, 1);
                         if ((simd_lane & 1) != 0) raw = neighbor;
                         half2 up = as_type<half2>(raw);
                         s1 = ((simd_lane & 1) == 0) ? up.x : up.y;
                    }

                    // Pack into cache
                    uint packed = as_type<uint>(half2(s0, s1));
                    if (i == 0) scale_cache_0 = packed;
                    else scale_cache_1 = packed;
                }
            }

            // Dequantize FP4 weights and write to B_tile
            // Process 8 elements at a time (one uint32 pack)
            #pragma unroll 4
            for (uint k_sub = 0; k_sub < TILE_K; k_sub += 8) {
                uint k_global = k_tile + k_sub;
                uint k_pack_idx = k_global / FP4_PER_UINT;
                uint sub_group_idx = k_sub / GROUP_SIZE;

                // Bounds check: use k_global < K for proper edge case handling
                if (global_col < N && k_global < K) {
                    uint32_t packed = B_packed[k_pack_idx * N + global_col];

                    // Fast path: zero-packed weights skip dequantization
                    if (packed == 0) {
                        #pragma unroll
                        for (uint k_off = 0; k_off < 8; ++k_off) {
                            B_tile[k_sub + k_off][b_col] = half(0.0h);
                        }
                        continue;
                    }

                    // Dequantize FP4 nibbles to FP16 using cached scale
                    uint idx = sub_group_idx / 2;
                    uint packed_scale = (idx == 0) ? scale_cache_0 : scale_cache_1;
                    half2 unpacked_scale = as_type<half2>(packed_scale);
                    half scale = (sub_group_idx & 1) ? unpacked_scale.y : unpacked_scale.x;
                    
                    half dequant_vals[8];
                    dequant_fp4x8(packed, scale, dequant_vals);
                    #pragma unroll
                    for (uint k_off = 0; k_off < 8; ++k_off) {
                        B_tile[k_sub + k_off][b_col] = dequant_vals[k_off];
                    }
                } else {
                    // Zero-pad for out-of-bounds access
                    #pragma unroll
                    for (uint k_off = 0; k_off < 8; ++k_off) {
                        B_tile[k_sub + k_off][b_col] = half(0.0h);
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Matrix Multiplication using SIMD Group Matrix Operations ---
        // Compute 4x4 8x8 matrix tiles: full 32x32 sub-tile per simdgroup
        // Fully unrolled K-loop for maximum instruction-level parallelism
        #pragma unroll 4
        for (uint k_sub = 0; k_sub < TILE_K; k_sub += 8) {
            #pragma unroll
            for (uint mi = 0; mi < 4; ++mi) {
                uint row_idx = sg_row_base + mi * 8;
                #pragma unroll
                for (uint ni = 0; ni < 4; ++ni) {
                    uint col_idx = sg_col_base + ni * 8;
                    
                    // CRITICAL: Skip entire 8x8 block if row_idx is beyond valid M
                    // This prevents loading invalid data when M < TILE_M (e.g., M=4, seq_len=4)
                    uint global_row_idx = tile_start_row + row_idx;
                    if (global_row_idx >= M) {
                        continue;
                    }
                    
                    // Load A fragment from threadgroup memory
                    simdgroup_matrix<half, 8, 8> a_frag;
                    simdgroup_load(a_frag, &A_tile[row_idx][k_sub], TILE_K);
                    
                    // Load B fragment from threadgroup memory
                    simdgroup_matrix<half, 8, 8> b_frag;
                    simdgroup_load(b_frag, &B_tile[k_sub][col_idx], TILE_N);
                    
                    // FMA: C += A * B
                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Store Results ---
    // Convert from float accumulator to half output
    // Per-simdgroup staging avoids race conditions between 4 simdgroups
    threadgroup float out_staging[SIMDGROUPS_PER_TG][8][8] __attribute__((aligned(128)));

    #pragma unroll
    for (uint mi = 0; mi < 4; ++mi) {
        #pragma unroll
        for (uint ni = 0; ni < 4; ++ni) {
            uint out_row = tgid.y * TILE_M + sg_row_base + mi * 8;
            uint out_col = tgid.x * TILE_N + sg_col_base + ni * 8;

            // Skip entire 8x8 sub-tile if it's completely outside valid M range
            // Critical for small M (e.g., M=4) where most sub-tiles are empty
            if (out_row >= M) {
                continue;
            }
            
            // PARTIAL TILE HANDLING: Zero out accumulator rows beyond valid M range
            // This is critical when M is not a multiple of 8 (e.g., M=4, 12, 20, etc.)
            // For partial tiles (last tile in M dimension), rows beyond M must be zeroed
            if (out_row + 8 > M) {
                // Create a masked accumulator: valid rows keep their values, invalid rows are zero
                // First, store to staging
                simdgroup_store(acc[mi][ni], &out_staging[simd_id][0][0], 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                
                // Zero out rows beyond M in staging (each thread handles 2 rows)
                #pragma unroll
                for (uint row = simd_lane; row < 8; row += 32) {
                    if (out_row + row >= M) {
                        // This entire row is beyond valid M range - zero it
                        out_staging[simd_id][row][0] = 0.0f;
                        out_staging[simd_id][row][1] = 0.0f;
                        out_staging[simd_id][row][2] = 0.0f;
                        out_staging[simd_id][row][3] = 0.0f;
                        out_staging[simd_id][row][4] = 0.0f;
                        out_staging[simd_id][row][5] = 0.0f;
                        out_staging[simd_id][row][6] = 0.0f;
                        out_staging[simd_id][row][7] = 0.0f;
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                
                // Reload back to accumulator
                simdgroup_load(acc[mi][ni], &out_staging[simd_id][0][0], 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Each simdgroup uses its own staging slot
            simdgroup_store(acc[mi][ni], &out_staging[simd_id][0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // Convert float to half during write to output
            // Process 64 elements in 2 iterations (32 threads per simdgroup)
            // Only write elements within valid M and N bounds
            #pragma unroll 2
            for (uint elem = simd_lane; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                if (out_row + r < M && out_col + c < N) {
                    C[(out_row + r) * N + out_col + c] = half(out_staging[simd_id][r][c]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
#else
#if !defined(M)
    const uint M = M_p[0];
#endif
    const uint K = K_p[0];
    const uint N = N_p[0];
    const uint GROUP_SIZE = group_size_p[0];

    const uint TILE_M = 64;
    const uint TILE_N = 64;
    const uint TILE_K = 32;
    const uint SIMDGROUPS_PER_TG = 4;
    const uint THREADS_PER_TG = SIMDGROUPS_PER_TG * 32;
    const uint FP4_PER_UINT = 8;

    uint simd_lane = tid.x % 32;
    uint simd_id = tid.x / 32;
    uint thread_idx = tid.x;
    
    // TILE_M EDGE CASE HANDLING: Early exit for tiles completely outside valid M range
    // Prevents unnecessary computation when M is not a multiple of TILE_M
    uint tile_start_row = tgid.y * TILE_M;
    if (tile_start_row >= M) {
        // Entire tile is outside valid M range - nothing to compute
        return;
    }
    
    // PARTIAL TILE HANDLING: Check if this simdgroup has any valid rows to compute
    // SIMD group layout: 2x2 grid covering 64x64 tile
    // simd_id 0: rows 0-31, simd_id 1: rows 0-31, simd_id 2: rows 32-63, simd_id 3: rows 32-63
    uint sg_row_base = (simd_id / 2) * 32;
    uint simd_start_row = tile_start_row + sg_row_base;
    if (simd_start_row >= M) {
        // This entire simdgroup is beyond valid M range - nothing to compute
        return;
    }

    threadgroup half A_tile[TILE_M][TILE_K] __attribute__((aligned(128)));
    threadgroup half B_tile[TILE_K][TILE_N] __attribute__((aligned(128)));

    uint sg_col_base = (simd_id % 2) * 32;

    // Use float accumulators to prevent overflow during accumulation
    simdgroup_matrix<float, 8, 8> acc[4][4];
    for (uint mi = 0; mi < 4; ++mi)
        for (uint ni = 0; ni < 4; ++ni)
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Optimized cooperative load
        threadgroup_load_A(A, A_tile, tgid.y, k_tile, thread_idx, M, K);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- B-Tile Population ---
        // Threads collaborate to fill B_tile. Each thread processes one or more columns.
        for (uint b_col = thread_idx; b_col < TILE_N; b_col += THREADS_PER_TG) {
            uint global_col = tgid.x * TILE_N + b_col;

            // Prefetch scales for the current column into registers.
            // This is done once per column for the entire TILE_K block.
            // Pack 2 halves into 1 uint for better register utilization
            uint scale_reg_cache_packed[SCALE_CACHE_SIZE / 2];
            #pragma unroll
            for (uint i = 0; i < SCALE_CACHE_SIZE / 2; i++) {
                scale_reg_cache_packed[i] = 0;
            }

            if (global_col < N) {
                uint group_base = k_tile / GROUP_SIZE;
                const uint N_GROUPS = n_groups_p[0];
                
                #pragma unroll
                for (uint i = 0; i < SCALE_CACHE_SIZE; i += 2) {
                    half s0 = 0.0h;
                    half s1 = 0.0h;

                    // Load s0 (group i)
                    if (SCALE_CACHE_SIZE > i && (group_base + i) < N_GROUPS) {
                         uint raw = 0;
                         if ((simd_lane & 1) == 0) {
                             if (global_col + 1 < N) {
                                 raw = *((device const uint*)(B_scales + (group_base + i) * N + global_col));
                             } else {
                                 half val = B_scales[(group_base + i) * N + global_col];
                                 raw = as_type<uint>(half2(val, 0.0h));
                             }
                         }
                         uint neighbor = simd_shuffle_xor(raw, 1);
                         if ((simd_lane & 1) != 0) raw = neighbor;
                         
                         half2 up = as_type<half2>(raw);
                         s0 = ((simd_lane & 1) == 0) ? up.x : up.y;
                    }

                    // Load s1 (group i+1)
                    if (SCALE_CACHE_SIZE > i + 1 && (group_base + i + 1) < N_GROUPS) {
                         uint raw = 0;
                         if ((simd_lane & 1) == 0) {
                             if (global_col + 1 < N) {
                                 raw = *((device const uint*)(B_scales + (group_base + i + 1) * N + global_col));
                             } else {
                                 half val = B_scales[(group_base + i + 1) * N + global_col];
                                 raw = as_type<uint>(half2(val, 0.0h));
                             }
                         }
                         uint neighbor = simd_shuffle_xor(raw, 1);
                         if ((simd_lane & 1) != 0) raw = neighbor;
                         half2 up = as_type<half2>(raw);
                         s1 = ((simd_lane & 1) == 0) ? up.x : up.y;
                    }

                    // Pack into cache
                    scale_reg_cache_packed[i/2] = as_type<uint>(half2(s0, s1));
                }
            }

            // Iterate through K-dimension blocks to dequantize and fill the column in B_tile.
            for (uint k_sub = 0; k_sub < TILE_K; k_sub += 8) {
                uint k_global = k_tile + k_sub;
                uint k_pack_idx = k_global / FP4_PER_UINT;
                uint sub_group_idx = (k_sub / GROUP_SIZE);

                // Fix: Use k_global < K for proper edge case handling
                // When K is not a multiple of 8, the old check (k_pack_idx < K/8)
                // could allow out-of-bounds reads, causing Inf in prefill
                if (global_col < N && k_global < K) {
                    uint32_t packed = B_packed[k_pack_idx * N + global_col];

                    if (packed == 0) {
                        for (uint k_off = 0; k_off < 8; ++k_off) {
                            B_tile[k_sub + k_off][b_col] = half(0.0h);
                        }
                        continue;
                    }

                    uint packed_scale = scale_reg_cache_packed[sub_group_idx / 2];
                    half2 unpacked_scale = as_type<half2>(packed_scale);
                    half scale = (sub_group_idx & 1) ? unpacked_scale.y : unpacked_scale.x;
                    
                    half dequant_vals[8];
                    dequant_fp4x8(packed, scale, dequant_vals);
                    for (uint k_off = 0; k_off < 8; ++k_off) {
                        B_tile[k_sub + k_off][b_col] = dequant_vals[k_off];
                    }
                } else {
                    // Zero-pad if out of bounds
                    for (uint k_off = 0; k_off < 8; ++k_off) {
                        B_tile[k_sub + k_off][b_col] = half(0.0h);
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Matrix Multiplication ---
        // B_tile is now fully populated for this k_tile. Proceed with multiplication.
        // K-loop unrolled by 4 to maximize SIMD group utilization and hide latency
        #pragma unroll 4
        for (uint k_sub = 0; k_sub < TILE_K; k_sub += 8) {
            for (uint mi = 0; mi < 4; ++mi) {
                uint row_idx = sg_row_base + mi * 8;
                for (uint ni = 0; ni < 4; ++ni) {
                    uint col_idx = sg_col_base + ni * 8;
                    
                    // CRITICAL: Skip entire 8x8 block if row_idx is beyond valid M
                    // This prevents loading invalid data when M < TILE_M (e.g., M=4)
                    if (tile_start_row + row_idx >= M) {
                        continue;
                    }

                    simdgroup_matrix<half, 8, 8> a_frag;
                    simdgroup_load(a_frag, &A_tile[row_idx][k_sub], TILE_K);
                    simdgroup_matrix<half, 8, 8> b_frag;
                    simdgroup_load(b_frag, &B_tile[k_sub][col_idx], TILE_N);
                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results - convert from float accumulator to half output
    // Per-simdgroup staging to avoid race conditions between 4 simdgroups
    threadgroup float out_staging[SIMDGROUPS_PER_TG][8][8] __attribute__((aligned(128)));

    for (uint mi = 0; mi < 4; ++mi) {
        for (uint ni = 0; ni < 4; ++ni) {
            uint out_row = tgid.y * TILE_M + sg_row_base + mi * 8;
            uint out_col = tgid.x * TILE_N + sg_col_base + ni * 8;

            // Skip entire 8x8 sub-tile if it's completely outside valid M range
            // This is critical for small M (e.g., M=4) where most tiles are empty
            if (out_row >= M) {
                continue;
            }
            
            // PARTIAL TILE HANDLING: Zero out accumulator rows beyond valid M range
            // This is critical when M is not a multiple of 8 (e.g., M=4, 12, 20, etc.)
            if (out_row + 8 > M) {
                // Store to staging first
                simdgroup_store(acc[mi][ni], &out_staging[simd_id][0][0], 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                
                // Zero out rows beyond M in staging (each thread handles rows strided)
                for (uint row = simd_lane; row < 8; row += 32) {
                    if (out_row + row >= M) {
                        // This entire row is beyond valid M range - zero it
                        out_staging[simd_id][row][0] = 0.0f;
                        out_staging[simd_id][row][1] = 0.0f;
                        out_staging[simd_id][row][2] = 0.0f;
                        out_staging[simd_id][row][3] = 0.0f;
                        out_staging[simd_id][row][4] = 0.0f;
                        out_staging[simd_id][row][5] = 0.0f;
                        out_staging[simd_id][row][6] = 0.0f;
                        out_staging[simd_id][row][7] = 0.0f;
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                
                // Reload back to accumulator
                simdgroup_load(acc[mi][ni], &out_staging[simd_id][0][0], 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Each simdgroup uses its own staging slot
            simdgroup_store(acc[mi][ni], &out_staging[simd_id][0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // Convert float to half during write to output
            // Only process elements that are within valid M and N bounds
            for (uint elem = simd_lane; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                if (out_row + r < M && out_col + c < N) {
                    C[(out_row + r) * N + out_col + c] = half(out_staging[simd_id][r][c]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
#endif
}
