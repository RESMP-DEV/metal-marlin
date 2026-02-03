// gemm_trellis_int8.metal - Fused INT8 Dequant+GEMM for TrellisLinear (W8A16)
// ============================================================================
//
// High-performance INT8 quantized GEMM kernel for Apple Metal.
// Performs on-the-fly INT8 dequantization during GEMM computation.
//
// INT8 Quantization Format:
//   - Weights packed as signed INT8 (4 values per uint32)
//   - Per-group scale factors (group_size typically 32 or 128)
//   - Optional zero points for asymmetric quantization
//   - Optional sign vectors (su, sv) for Hadamard preprocessing
//
// Dequantization formula:
//   Symmetric:   w[k,n] = int8_val * scale[group][n] * su[k] * sv[n]
//   Asymmetric:  w[k,n] = (int8_val - zero) * scale[group][n] * su[k] * sv[n]
//
// Memory Layout:
//   - A: [M, K] half - input activations (row-major)
//   - B: [K/4, N] uint32 - packed INT8 weights (4 values per uint32)
//   - scales: [K/group_size, N] half - per-group scale factors
//   - zeros: [K/group_size, N] half - optional zero points (asymmetric mode)
//   - su: [K] half - optional row signs for Hadamard
//   - sv: [N] half - optional column signs for Hadamard
//   - C: [M, N] half - output matrix (row-major)
//
// Kernel Architecture:
//   - Tile dimensions: 128x128x32 (large tiles), 32x128x32 (decode)
//   - 4 simdgroups (128 threads) per threadgroup
//   - Double-buffered A tiles in threadgroup memory
//   - On-the-fly INT8 dequant with per-simdgroup staging
//
// ============================================================================

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// ============================================================================
// Tile Dimensions
// ============================================================================

// Main GEMM tiles (prefill path: M > 16)
constant constexpr uint TILE_M = 128;
constant constexpr uint TILE_N = 128;
constant constexpr uint TILE_K = 32;
constant constexpr uint K_TILES = TILE_K / 8;  // 4 MMA ops per K-block

// Decode tiles (M <= 16)
constant constexpr uint DECODE_TILE_M = 32;
constant constexpr uint DECODE_TILE_N = 128;
constant constexpr uint DECODE_K_TILES = TILE_K / 8;

// Simdgroup configuration
constant constexpr uint SIMDGROUPS_PER_TG = 4;
constant constexpr uint THREADS_PER_TG = SIMDGROUPS_PER_TG * 32;  // 128 threads
constant constexpr uint SG_M_TILES = 4;   // 32 rows per simdgroup (4 x 8)
constant constexpr uint SG_N_TILES = 4;   // 32 cols per simdgroup (4 x 8)

// INT8 packing
constant constexpr uint INT8_PER_UINT = 4;

// ============================================================================
// INT8 Dequantization Primitives
// ============================================================================

/// Extract signed INT8 byte at position [0-3] from packed uint32.
/// Metal's int8_t cast handles sign extension automatically.
inline int8_t extract_s8(uint32_t packed, uint pos) {
    return int8_t((packed >> (pos * 8u)) & 0xFFu);
}

/// Dequantize 4 INT8 values with symmetric quantization.
/// result = int8_value * scale
inline void dequant_s8x4_sym(uint32_t packed, half scale, thread half4 &out) {
    int8_t b0 = extract_s8(packed, 0);
    int8_t b1 = extract_s8(packed, 1);
    int8_t b2 = extract_s8(packed, 2);
    int8_t b3 = extract_s8(packed, 3);

    float fscale = float(scale);
    out = half4(float4(float(b0), float(b1), float(b2), float(b3)) * fscale);
}

/// Dequantize 4 INT8 values with asymmetric quantization.
/// result = (int8_value - zero_point) * scale
inline void dequant_s8x4_asym(uint32_t packed, half scale, half zero_point, thread half4 &out) {
    int8_t b0 = extract_s8(packed, 0);
    int8_t b1 = extract_s8(packed, 1);
    int8_t b2 = extract_s8(packed, 2);
    int8_t b3 = extract_s8(packed, 3);

    float fscale = float(scale);
    float fzero = float(zero_point);
    out = half4((float4(float(b0), float(b1), float(b2), float(b3)) - fzero) * fscale);
}

/// Dequantize 4 INT8 values with symmetric quantization and sign flip.
/// result = int8_value * scale * sign
inline void dequant_s8x4_sym_sign(uint32_t packed, half scale, half sign, thread half4 &out) {
    dequant_s8x4_sym(packed, scale * sign, out);
}

/// Dequantize 8 INT8 values (2 uint32s) with symmetric quantization.
inline void dequant_s8x8_sym(uint32_t lo, uint32_t hi, half scale,
                              thread half4 &out_lo, thread half4 &out_hi) {
    dequant_s8x4_sym(lo, scale, out_lo);
    dequant_s8x4_sym(hi, scale, out_hi);
}

/// Dequantize 8 INT8 values with asymmetric quantization.
inline void dequant_s8x8_asym(uint32_t lo, uint32_t hi, half scale, half zero,
                               thread half4 &out_lo, thread half4 &out_hi) {
    dequant_s8x4_asym(lo, scale, zero, out_lo);
    dequant_s8x4_asym(hi, scale, zero, out_hi);
}

// ============================================================================
// Cooperative A Tile Loader
// ============================================================================

inline void load_A_tile_int8(
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

// Decode tile loader (smaller tiles)
inline void load_A_tile_decode_int8(
    device const half* A,
    threadgroup half (&A_buf)[DECODE_TILE_M][TILE_K],
    uint M, uint K,
    uint tg_row, uint k_block,
    uint thread_idx
) {
    constexpr uint A_ELEMS = (DECODE_TILE_M * TILE_K) / THREADS_PER_TG;  // 8

    #pragma unroll
    for (uint i = 0; i < A_ELEMS; ++i) {
        uint flat_idx = thread_idx * A_ELEMS + i;
        uint row = flat_idx / TILE_K;
        uint col = flat_idx % TILE_K;
        uint global_row = tg_row + row;
        uint global_col = k_block + col;

        half val = (global_row < M && global_col < K) ? A[global_row * K + global_col] : half(0.0h);
        A_buf[row][col] = val;
    }
}

// ============================================================================
// Main Kernel: Fused INT8 Dequant + GEMM (Prefill Path)
// ============================================================================
//
// For batch sizes M > 16, uses 128x128 output tiles for high throughput.
// Supports both symmetric and asymmetric INT8 quantization.
//
// ============================================================================

kernel void gemm_trellis_int8_packed(
    device const half* A               [[buffer(0)]],   // [M, K] activations
    device const uint32_t* B           [[buffer(1)]],   // [K/4, N] packed INT8
    device const half* scales          [[buffer(2)]],   // [K/group_size, N] scales
    device const half* zeros           [[buffer(3)]],   // [K/group_size, N] zeros (or nullptr)
    device const half* su              [[buffer(4)]],   // [K] row signs (or nullptr)
    device const half* sv              [[buffer(5)]],   // [N] col signs (or nullptr)
    device half* C                     [[buffer(6)]],   // [M, N] output
    constant uint& M                   [[buffer(7)]],
    constant uint& K                   [[buffer(8)]],
    constant uint& N                   [[buffer(9)]],
    constant uint& group_size          [[buffer(10)]],
    constant uint& asymmetric          [[buffer(11)]],  // 0=symmetric, 1=asymmetric
    constant uint& use_signs           [[buffer(12)]],  // 0=no signs, 1=use su/sv
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint simd_lane_id                  [[thread_index_in_simdgroup]],
    uint simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    // -------------------------------------------------------------------------
    // Threadgroup memory
    // -------------------------------------------------------------------------
    threadgroup half A_tiles[2][TILE_M][TILE_K];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];

    // -------------------------------------------------------------------------
    // Tile assignment
    // -------------------------------------------------------------------------
    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;

    if (tg_row >= M) return;

    // Simdgroup layout: 2x2 grid, each handles 64x64 of 128x128 output
    const uint sg_row_offset = (simd_group_id / 2) * 64;
    const uint sg_col_offset = (simd_group_id % 2) * 64;

    // -------------------------------------------------------------------------
    // Initialize accumulators (8 x 8 subtiles within 64x64 region)
    // -------------------------------------------------------------------------
    constexpr uint SG_M_SUBTILES = 8;  // 64/8 = 8 subtiles in M
    constexpr uint SG_N_SUBTILES = 8;  // 64/8 = 8 subtiles in N

    simdgroup_matrix<half, 8, 8> acc[SG_M_SUBTILES][SG_N_SUBTILES];
    #pragma unroll
    for (uint mi = 0; mi < SG_M_SUBTILES; ++mi) {
        #pragma unroll
        for (uint ni = 0; ni < SG_N_SUBTILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0h));
        }
    }

    const uint thread_idx = simd_group_id * 32 + simd_lane_id;
    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;
    const uint k_packs = (K + INT8_PER_UINT - 1) / INT8_PER_UINT;

    uint buf_compute = 0;

    // -------------------------------------------------------------------------
    // Prologue: Load first A tile
    // -------------------------------------------------------------------------
    load_A_tile_int8(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Main K-loop
    // -------------------------------------------------------------------------
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;
        uint next_k = k_block + TILE_K;
        uint buf_load = 1 - buf_compute;

        // Load next A tile (async with compute)
        if (next_k < K) {
            load_A_tile_int8(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
        }

        // Process K-subtiles (8 elements per subtile)
        #pragma unroll
        for (uint kk = 0; kk < K_TILES; ++kk) {
            uint k_sub_base = k_block + kk * 8;
            uint group_idx = k_sub_base / group_size;
            uint k_pack_base = k_sub_base / INT8_PER_UINT;

            // Load A fragments for all M subtiles
            simdgroup_matrix<half, 8, 8> a_frag[SG_M_SUBTILES];
            #pragma unroll
            for (uint mi = 0; mi < SG_M_SUBTILES; ++mi) {
                simdgroup_load(a_frag[mi],
                              &A_tiles[buf_compute][sg_row_offset + mi * 8][kk * 8],
                              TILE_K);
            }

            // Process each N subtile
            #pragma unroll
            for (uint ni = 0; ni < SG_N_SUBTILES; ++ni) {
                uint b_col_base = tg_col + sg_col_offset + ni * 8;

                // Dequantize B tile (lanes 0-7 each handle one column)
                if (simd_lane_id < 8) {
                    uint b_col = b_col_base + simd_lane_id;
                    half dequant_vals[8];

                    if (b_col < N) {
                        // Get scale and optional zero point
                        half scale = scales[group_idx * N + b_col];
                        half zero_point = asymmetric ? zeros[group_idx * N + b_col] : half(0.0h);

                        // Get optional sign
                        half col_sign = use_signs ? sv[b_col] : half(1.0h);
                        half scaled = scale * col_sign;

                        // Dequantize 8 values (2 uint32s = 8 INT8)
                        #pragma unroll
                        for (uint row = 0; row < 8; row += 4) {
                            uint k_idx = k_sub_base + row;
                            if (k_idx < K) {
                                uint pack_idx = k_idx / INT8_PER_UINT;
                                uint32_t packed = B[pack_idx * N + b_col];

                                half row_sign = use_signs ? su[k_idx] : half(1.0h);
                                half combined_scale = scaled * row_sign;

                                half4 dq;
                                if (asymmetric) {
                                    dequant_s8x4_asym(packed, combined_scale, zero_point * col_sign * row_sign, dq);
                                } else {
                                    dequant_s8x4_sym(packed, combined_scale, dq);
                                }

                                dequant_vals[row + 0] = dq.x;
                                dequant_vals[row + 1] = dq.y;
                                dequant_vals[row + 2] = dq.z;
                                dequant_vals[row + 3] = dq.w;
                            } else {
                                dequant_vals[row + 0] = half(0.0h);
                                dequant_vals[row + 1] = half(0.0h);
                                dequant_vals[row + 2] = half(0.0h);
                                dequant_vals[row + 3] = half(0.0h);
                            }
                        }
                    } else {
                        #pragma unroll
                        for (uint i = 0; i < 8; ++i) {
                            dequant_vals[i] = half(0.0h);
                        }
                    }

                    // Write to staging
                    #pragma unroll
                    for (uint row = 0; row < 8; ++row) {
                        B_staging[simd_group_id][row][simd_lane_id] = dequant_vals[row];
                    }
                }

                simdgroup_barrier(mem_flags::mem_threadgroup);

                // Load B fragment and accumulate
                simdgroup_matrix<half, 8, 8> b_frag;
                simdgroup_load(b_frag, &B_staging[simd_group_id][0][0], 8);

                #pragma unroll
                for (uint mi = 0; mi < SG_M_SUBTILES; ++mi) {
                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag[mi], b_frag, acc[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // -------------------------------------------------------------------------
    // Epilogue: Store results
    // -------------------------------------------------------------------------
    threadgroup half epilogue_staging[8][8];

    #pragma unroll
    for (uint mi = 0; mi < SG_M_SUBTILES; ++mi) {
        uint out_row = tg_row + sg_row_offset + mi * 8;
        if (out_row >= M) continue;

        #pragma unroll
        for (uint ni = 0; ni < SG_N_SUBTILES; ++ni) {
            uint out_col = tg_col + sg_col_offset + ni * 8;

            // Fast path: full tile within bounds
            if (out_row + 8 <= M && out_col + 8 <= N) {
                simdgroup_store(acc[mi][ni], C + out_row * N + out_col, N);
                continue;
            }

            // Slow path: bounds checking
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
// Decode Kernel: Fused INT8 Dequant + GEMM (Small M)
// ============================================================================
//
// Optimized for M <= 16 (autoregressive decode). Uses 32x128 tiles.
//
// ============================================================================

kernel void gemm_trellis_int8_decode(
    device const half* A               [[buffer(0)]],
    device const uint32_t* B           [[buffer(1)]],
    device const half* scales          [[buffer(2)]],
    device const half* zeros           [[buffer(3)]],
    device const half* su              [[buffer(4)]],
    device const half* sv              [[buffer(5)]],
    device half* C                     [[buffer(6)]],
    constant uint& M                   [[buffer(7)]],
    constant uint& K                   [[buffer(8)]],
    constant uint& N                   [[buffer(9)]],
    constant uint& group_size          [[buffer(10)]],
    constant uint& asymmetric          [[buffer(11)]],
    constant uint& use_signs           [[buffer(12)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint simd_lane_id                  [[thread_index_in_simdgroup]],
    uint simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half A_tiles[2][DECODE_TILE_M][TILE_K];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];

    const uint tg_row = tgid.y * DECODE_TILE_M;
    const uint tg_col = tgid.x * DECODE_TILE_N;

    if (tg_row >= M) return;

    // 4 simdgroups tile 32x128: each handles 32x32
    const uint sg_row_offset = 0;
    const uint sg_col_offset = simd_group_id * 32;

    constexpr uint DECODE_SG_M_TILES = 4;  // 32/8
    constexpr uint DECODE_SG_N_TILES = 4;  // 32/8

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

    uint buf_compute = 0;

    // Load first A tile
    load_A_tile_decode_int8(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main K-loop
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;
        uint next_k = k_block + TILE_K;
        uint buf_load = 1 - buf_compute;

        if (next_k < K) {
            load_A_tile_decode_int8(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
        }

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
                        half scale = scales[group_idx * N + b_col];
                        half zero_point = asymmetric ? zeros[group_idx * N + b_col] : half(0.0h);
                        half col_sign = use_signs ? sv[b_col] : half(1.0h);
                        half scaled = scale * col_sign;

                        #pragma unroll
                        for (uint row = 0; row < 8; row += 4) {
                            uint k_idx = k_sub_base + row;
                            if (k_idx < K) {
                                uint pack_idx = k_idx / INT8_PER_UINT;
                                uint32_t packed = B[pack_idx * N + b_col];

                                half row_sign = use_signs ? su[k_idx] : half(1.0h);
                                half combined_scale = scaled * row_sign;

                                half4 dq;
                                if (asymmetric) {
                                    dequant_s8x4_asym(packed, combined_scale, zero_point * col_sign * row_sign, dq);
                                } else {
                                    dequant_s8x4_sym(packed, combined_scale, dq);
                                }

                                dequant_vals[row + 0] = dq.x;
                                dequant_vals[row + 1] = dq.y;
                                dequant_vals[row + 2] = dq.z;
                                dequant_vals[row + 3] = dq.w;
                            } else {
                                dequant_vals[row + 0] = half(0.0h);
                                dequant_vals[row + 1] = half(0.0h);
                                dequant_vals[row + 2] = half(0.0h);
                                dequant_vals[row + 3] = half(0.0h);
                            }
                        }
                    } else {
                        #pragma unroll
                        for (uint i = 0; i < 8; ++i) {
                            dequant_vals[i] = half(0.0h);
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

    // Store results
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
// Standalone INT8 Dequantization Kernel
// ============================================================================
//
// For cases where separate dequant + matmul is preferred (debugging, etc.)
//
// ============================================================================

kernel void dequant_trellis_int8(
    device const uint32_t* packed_weights [[buffer(0)]],  // [K/4, N] packed INT8
    device const half* scales            [[buffer(1)]],   // [K/group_size, N]
    device const half* zeros             [[buffer(2)]],   // [K/group_size, N] or nullptr
    device const half* su                [[buffer(3)]],   // [K] or nullptr
    device const half* sv                [[buffer(4)]],   // [N] or nullptr
    device half* output                  [[buffer(5)]],   // [K, N]
    constant uint& K                     [[buffer(6)]],
    constant uint& N                     [[buffer(7)]],
    constant uint& group_size            [[buffer(8)]],
    constant uint& asymmetric            [[buffer(9)]],
    constant uint& use_signs             [[buffer(10)]],
    uint2 gid                            [[thread_position_in_grid]]
) {
    // gid.x = column index (n)
    // gid.y = K block (each thread handles 4 K values)
    uint n_idx = gid.x;
    uint k_block = gid.y;
    uint k_base = k_block * 4;

    if (n_idx >= N || k_base >= K) return;

    // Load packed INT8
    uint pack_idx = k_block * N + n_idx;
    uint32_t packed = packed_weights[pack_idx];

    // Determine group
    uint group_idx = k_base / group_size;

    // Load scale and zero point
    half scale = scales[group_idx * N + n_idx];
    half zero_point = asymmetric ? zeros[group_idx * N + n_idx] : half(0.0h);

    // Get column sign
    half col_sign = use_signs ? sv[n_idx] : half(1.0h);

    // Dequantize 4 values
    half4 dequant;
    if (asymmetric) {
        dequant_s8x4_asym(packed, scale * col_sign, zero_point, dequant);
    } else {
        dequant_s8x4_sym(packed, scale * col_sign, dequant);
    }

    // Apply row signs and write output
    uint k_remain = min(4u, K - k_base);
    for (uint i = 0; i < k_remain; ++i) {
        uint k_idx = k_base + i;
        half row_sign = use_signs ? su[k_idx] : half(1.0h);
        output[k_idx * N + n_idx] = dequant[i] * row_sign;
    }
}

// ============================================================================
// Test Kernels for Validation
// ============================================================================

kernel void test_int8_dequant_trellis(
    device const uint32_t* packed_input [[buffer(0)]],
    device const half* scale           [[buffer(1)]],
    device half* output                [[buffer(2)]],
    uint tid                           [[thread_position_in_grid]]
) {
    if (tid > 0u) return;

    half4 result;
    dequant_s8x4_sym(packed_input[0], scale[0], result);

    output[0] = result.x;
    output[1] = result.y;
    output[2] = result.z;
    output[3] = result.w;
}

kernel void test_int8_dequant_asym_trellis(
    device const uint32_t* packed_input [[buffer(0)]],
    device const half* scale           [[buffer(1)]],
    device const half* zero_point      [[buffer(2)]],
    device half* output                [[buffer(3)]],
    uint tid                           [[thread_position_in_grid]]
) {
    if (tid > 0u) return;

    half4 result;
    dequant_s8x4_asym(packed_input[0], scale[0], zero_point[0], result);

    output[0] = result.x;
    output[1] = result.y;
    output[2] = result.z;
    output[3] = result.w;
}
