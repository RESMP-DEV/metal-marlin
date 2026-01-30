// gemm_epilogue.metal - Fused epilogue operations for quantized GEMM kernels
//
// Provides bias addition and activation functions that execute between the
// GEMM accumulation and global memory store. Since simdgroup_matrix<half,8,8>
// does not support per-element access, the epilogue spills through a per-tile
// threadgroup staging buffer to apply element-wise operations.
//
// Epilogue modes (selected via constant buffer):
//   0: None (identity store)
//   1: Bias only
//   2: GELU (fast approximation)
//   3: SiLU (Swish: x * sigmoid(x))
//   4: Bias + GELU
//   5: Bias + SiLU
//   6: ReLU
//   7: Bias + ReLU
//
// Design rationale:
//   Metal lacks kernel templates, so epilogue selection uses a runtime constant
//   (promoted to compile-time by the Metal compiler's function constant folding).
//   The staging buffer is already required for boundary-safe stores in the
//   existing GEMM kernels, so the epilogue adds zero extra memory cost for
//   boundary tiles. For interior tiles (no boundary check), the epilogue path
//   introduces one threadgroup staging round-trip per 8x8 tile, which is
//   ~16 cycles on M4 Max - negligible relative to the GEMM compute.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// Re-declare tile constants (must match marlin_gemm.metal)
constant constexpr uint EP_SG_M_TILES = 8;
constant constexpr uint EP_SG_N_TILES = 2;

// ---------------------------------------------------------------------------
// Epilogue mode enum (matches buffer constant)
// ---------------------------------------------------------------------------

enum EpilogueMode : uint {
    EPILOGUE_NONE      = 0,
    EPILOGUE_BIAS      = 1,
    EPILOGUE_GELU      = 2,
    EPILOGUE_SILU      = 3,
    EPILOGUE_BIAS_GELU = 4,
    EPILOGUE_BIAS_SILU = 5,
    EPILOGUE_RELU      = 6,
    EPILOGUE_BIAS_RELU = 7,
};

// ---------------------------------------------------------------------------
// Activation functions
//
// All operate on half precision. The GELU uses the sigmoid approximation
// (x * sigmoid(1.702 * x)) which is faster than the tanh form and accurate
// to ~1e-3 relative error for transformer inference.
// ---------------------------------------------------------------------------

/// Fast GELU: x * sigmoid(1.702 * x)
/// Max relative error vs exact GELU: ~0.4% (acceptable for inference)
inline half gelu_approx(half x) {
    half kx = half(1.702h) * x;
    return x * (half(1.0h) / (half(1.0h) + exp(-kx)));
}

/// SiLU (Swish): x * sigmoid(x) = x / (1 + exp(-x))
inline half silu(half x) {
    return x / (half(1.0h) + exp(-x));
}

/// ReLU: max(0, x)
inline half relu(half x) {
    return max(x, half(0.0h));
}

/// Apply activation function to a single value
inline half apply_activation(half x, uint mode) {
    switch (mode) {
        case EPILOGUE_GELU:
        case EPILOGUE_BIAS_GELU:
            return gelu_approx(x);
        case EPILOGUE_SILU:
        case EPILOGUE_BIAS_SILU:
            return silu(x);
        case EPILOGUE_RELU:
        case EPILOGUE_BIAS_RELU:
            return relu(x);
        default:
            return x;
    }
}

/// Check if mode includes bias addition
inline bool has_bias(uint mode) {
    return mode == EPILOGUE_BIAS ||
           mode == EPILOGUE_BIAS_GELU ||
           mode == EPILOGUE_BIAS_SILU ||
           mode == EPILOGUE_BIAS_RELU;
}

/// Check if mode includes any activation
inline bool has_activation(uint mode) {
    return mode >= EPILOGUE_GELU;
}

// ---------------------------------------------------------------------------
// epilogue_apply: Apply bias + activation to an 8x8 staging tile in-place
//
// bias: pointer to bias vector of length N (one value per output column)
// col_offset: global column index of this tile's left edge
// mode: EpilogueMode constant
//
// Each thread in the simdgroup processes 2 elements (64 total / 32 threads).
// The staging buffer must be filled via simdgroup_store before calling this.
// ---------------------------------------------------------------------------

inline void epilogue_apply(
    threadgroup half (&staging)[8][8],
    device const half* bias,
    uint col_offset,
    uint mode,
    uint simd_lane
) {
    for (uint elem = simd_lane; elem < 64; elem += 32) {
        uint r = elem / 8;
        uint c = elem % 8;

        half val = staging[r][c];

        // Bias addition (broadcast across rows)
        if (has_bias(mode)) {
            val += bias[col_offset + c];
        }

        // Activation
        if (has_activation(mode)) {
            val = apply_activation(val, mode);
        }

        staging[r][c] = val;
    }
}

// ---------------------------------------------------------------------------
// Epilogue helpers operating directly on simdgroup accumulators
//
// These are convenience wrappers for kernels that want a simple bias or
// bias+activation fuse without calling store_results_epilogue.
// ---------------------------------------------------------------------------

inline void epilogue_bias(
    thread simdgroup_matrix<half, 8, 8>& acc,
    device const half* bias,
    uint col_offset,
    uint simd_lane,
    threadgroup half (&staging)[8][8]
) {
    simdgroup_store(acc, &staging[0][0], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    epilogue_apply(staging, bias, col_offset, EPILOGUE_BIAS, simd_lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_load(acc, &staging[0][0], 8);
}

inline void epilogue_bias_gelu(
    thread simdgroup_matrix<half, 8, 8>& acc,
    device const half* bias,
    uint col_offset,
    uint simd_lane,
    threadgroup half (&staging)[8][8]
) {
    simdgroup_store(acc, &staging[0][0], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    epilogue_apply(staging, bias, col_offset, EPILOGUE_BIAS_GELU, simd_lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_load(acc, &staging[0][0], 8);
}

// ---------------------------------------------------------------------------
// store_results_epilogue: Store accumulator tiles with fused epilogue
//
// When mode == EPILOGUE_NONE, this is equivalent to the original store_results.
// For non-trivial epilogues, all tiles go through staging to apply bias/act.
//
// This function replaces store_results in epilogue-enabled kernel variants.
// ---------------------------------------------------------------------------

inline void store_results_epilogue(
    thread simdgroup_matrix<half, 8, 8> acc[EP_SG_M_TILES][EP_SG_N_TILES],
    device half* C,
    device const half* bias,  // nullable when mode has no bias
    uint M, uint N,
    uint tg_row, uint tg_col,
    uint sg_row_offset, uint sg_col_offset,
    uint simd_lane,
    uint mode,
    threadgroup half (&staging)[8][8]
) {
    // Fast path: no epilogue, use direct simdgroup_store for interior tiles
    if (mode == EPILOGUE_NONE) {
        for (uint mi = 0; mi < EP_SG_M_TILES; ++mi) {
            for (uint ni = 0; ni < EP_SG_N_TILES; ++ni) {
                uint out_row = tg_row + sg_row_offset + mi * 8;
                uint out_col = tg_col + sg_col_offset + ni * 8;

                if (out_row + 8 <= M && out_col + 8 <= N) {
                    simdgroup_store(acc[mi][ni], C + out_row * N + out_col, N);
                } else {
                    simdgroup_store(acc[mi][ni], &staging[0][0], 8);
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint elem = simd_lane; elem < 64; elem += 32) {
                        uint r = elem / 8;
                        uint c = elem % 8;
                        uint gr = out_row + r;
                        uint gc = out_col + c;
                        if (gr < M && gc < N) {
                            C[gr * N + gc] = staging[r][c];
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
        return;
    }

    // Epilogue path: all tiles go through staging for element-wise ops
    for (uint mi = 0; mi < EP_SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < EP_SG_N_TILES; ++ni) {
            uint out_row = tg_row + sg_row_offset + mi * 8;
            uint out_col = tg_col + sg_col_offset + ni * 8;

            simdgroup_store(acc[mi][ni], &staging[0][0], 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Apply epilogue in-place
            epilogue_apply(staging, bias, out_col, mode, simd_lane);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Store to global memory with boundary check
            if (out_row + 8 <= M && out_col + 8 <= N) {
                // Interior tile: vectorized store via simdgroup_load + store
                simdgroup_matrix<half, 8, 8> result;
                simdgroup_load(result, &staging[0][0], 8);
                simdgroup_store(result, C + out_row * N + out_col, N);
            } else {
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    uint gr = out_row + r;
                    uint gc = out_col + c;
                    if (gr < M && gc < N) {
                        C[gr * N + gc] = staging[r][c];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Vectorized epilogue for half2 operations (2x throughput on Apple GPU)
//
// Processes staging buffer with half2 operations where possible. This doubles
// the effective bandwidth of the epilogue since Apple GPUs execute half2
// operations at the same throughput as scalar half.
// ---------------------------------------------------------------------------

inline void epilogue_apply_vec2(
    threadgroup half (&staging)[8][8],
    device const half* bias,
    uint col_offset,
    uint mode,
    uint simd_lane
) {
    // Process 4 half2 pairs per thread (32 half2 = 64 elements across simdgroup)
    for (uint elem = simd_lane; elem < 32; elem += 32) {
        uint flat = elem * 2;
        uint r = flat / 8;
        uint c = flat % 8;

        // Load pair
        half2 val = half2(staging[r][c], staging[r][c + 1]);

        // Bias: load pair from bias vector
        if (has_bias(mode)) {
            half2 b = half2(bias[col_offset + c], bias[col_offset + c + 1]);
            val += b;
        }

        // Activation (element-wise, since GELU/SiLU aren't trivially vectorizable)
        if (has_activation(mode)) {
            val.x = apply_activation(val.x, mode);
            val.y = apply_activation(val.y, mode);
        }

        staging[r][c]     = val.x;
        staging[r][c + 1] = val.y;
    }

    // Handle remaining elements (threads 16-31 get the second half)
    for (uint elem = simd_lane + 32; elem < 64; elem += 32) {
        uint r = elem / 8;
        uint c = elem % 8;

        half val = staging[r][c];
        if (has_bias(mode)) {
            val += bias[col_offset + c];
        }
        if (has_activation(mode)) {
            val = apply_activation(val, mode);
        }
        staging[r][c] = val;
    }
}

// ===========================================================================
// Kernel: Fused FP4 GEMM with epilogue (bias + activation)
//
// C[M,N] = activation(A[M,K] @ dequant(B_packed[K/8,N], scales[...]) + bias)
//
// Same structure as marlin_gemm_fused_fp4 but adds:
//   - buffer(5): bias vector (half[N]) - may be nullptr if mode has no bias
//   - buffer(6): epilogue_mode constant (uint)
//
// The epilogue is applied per-tile after accumulation completes, before the
// global store. This fuses what would otherwise be a separate bias+activation
// kernel launch, eliminating one global memory round-trip of the full output.
// ===========================================================================

// ---------------------------------------------------------------------------
// FP4 E2M1 dequantization (duplicated from marlin_gemm.metal for standalone compilation)
// ---------------------------------------------------------------------------

/// Guard a dequantized half value: replace NaN/Inf with 0.
inline half guard_finite(half val) {
    return select(val, half(0.0h), !isfinite(val));
}

/// Safe dequant with float intermediates to work around Metal compiler bug.
inline half safe_dequant(half raw, half scale) {
    float result = (float)raw * (float)scale;
    if (!isfinite(result)) {
        result = 0.0f;
    }
    return (half)result;
}

/// FP4 (E2M1) dequant: branchless scalar, select() for subnormal handling.
inline half fused_dequant_fp4_scalar(uint nibble) {
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;

    // Subnormal (exp=0): 0.0 or 0.25
    half sub_mag = half(man_bit) * half(0.25h);
    // Normal (exp>0): 2^(exp-1) * (1 + mantissa*0.5)
    half norm_mag = half(1u << (exp_bits - 1)) * (half(1.0h) + half(man_bit) * half(0.5h));

    half magnitude = select(norm_mag, sub_mag, exp_bits == 0);
    return select(magnitude, -magnitude, bool(sign_bit));
}

/// FP4 dequant: 8 values from packed uint32, scale applied.
inline void fused_dequant_fp4x8(uint32_t packed, half scale, thread half *out) {
    out[0] = safe_dequant(fused_dequant_fp4_scalar((packed >>  0) & 0xF), scale);
    out[1] = safe_dequant(fused_dequant_fp4_scalar((packed >>  4) & 0xF), scale);
    out[2] = safe_dequant(fused_dequant_fp4_scalar((packed >>  8) & 0xF), scale);
    out[3] = safe_dequant(fused_dequant_fp4_scalar((packed >> 12) & 0xF), scale);
    out[4] = safe_dequant(fused_dequant_fp4_scalar((packed >> 16) & 0xF), scale);
    out[5] = safe_dequant(fused_dequant_fp4_scalar((packed >> 20) & 0xF), scale);
    out[6] = safe_dequant(fused_dequant_fp4_scalar((packed >> 24) & 0xF), scale);
    out[7] = safe_dequant(fused_dequant_fp4_scalar((packed >> 28) & 0xF), scale);
}

// Tile constants for the kernel (same as marlin_gemm.metal)
constant constexpr uint EP_TILE_M = 64;
constant constexpr uint EP_TILE_N = 64;
constant constexpr uint EP_TILE_K = 32;
constant constexpr uint EP_K_TILES = EP_TILE_K / 8;
constant constexpr uint EP_SIMDGROUPS_PER_TG = 4;
constant constexpr uint EP_THREADS_PER_TG = EP_SIMDGROUPS_PER_TG * 32;
constant constexpr uint EP_FP4_PER_UINT = 8;

inline void marlin_gemm_fused_fp4_epilogue_impl(
    device const half* A,
    device const uint32_t* B_packed,
    device const half* scales,
    device half* C,
    constant uint4& dims,  // {M, N, K, group_size}
    device const half* bias,
    uint epilogue_mode,
    uint2 tg_pos,
    uint simd_id,
    uint simd_lane,
    threadgroup half (&A_tile)[EP_TILE_M][EP_TILE_K],
    threadgroup half (&B_staging)[EP_SIMDGROUPS_PER_TG][8][8],
    threadgroup half (&epilogue_staging)[8][8]
) {
    const uint M = dims.x;
    const uint N = dims.y;
    const uint K = dims.z;
    const uint group_size = dims.w;

    const uint tg_row = tg_pos.y * EP_TILE_M;
    const uint tg_col = tg_pos.x * EP_TILE_N;

    if (tg_row >= M) return;

    const uint sg_row_offset = 0;  // All simdgroups cover all rows
    const uint sg_col_offset = simd_id * (EP_SG_N_TILES * 8);

    // --- Accumulators ---
    simdgroup_matrix<half, 8, 8> acc[EP_SG_M_TILES][EP_SG_N_TILES];
    for (uint mi = 0; mi < EP_SG_M_TILES; ++mi)
        for (uint ni = 0; ni < EP_SG_N_TILES; ++ni)
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint k_packs = K / EP_FP4_PER_UINT;
    (void)k_packs;  // Suppress unused variable warning

    // --- Main loop over K dimension ---
    for (uint k_block = 0; k_block < K; k_block += EP_TILE_K) {
        // Cooperative load of A tile
        for (uint idx = thread_idx; idx < EP_TILE_M * EP_TILE_K; idx += EP_THREADS_PER_TG) {
            uint row = idx / EP_TILE_K;
            uint col = idx % EP_TILE_K;
            uint global_row = tg_row + row;
            uint global_col = k_block + col;
            A_tile[row][col] = (global_row < M && global_col < K)
                ? A[global_row * K + global_col]
                : half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-simdgroup: dequant B sub-tiles and accumulate
        for (uint kt = 0; kt < EP_K_TILES; ++kt) {
            for (uint ni = 0; ni < EP_SG_N_TILES; ++ni) {
                uint b_col = tg_col + sg_col_offset + ni * 8;

                // Each thread dequants one row of the 8x8 B sub-tile
                // (simd_lane % 8 selects which of the 8 rows, wraps for 32 threads)
                uint b_row_in_tile = simd_lane % 8;
                uint global_k = k_block + kt * 8 + b_row_in_tile;

                if (global_k < K && b_col < N) {
                    uint pack_idx = (global_k / EP_FP4_PER_UINT) * N + b_col;
                    uint32_t packed = B_packed[pack_idx];

                    // Determine scale for this group
                    uint group_idx = global_k / group_size;
                    half scale = scales[group_idx * N + b_col];

                    // Dequant 8 FP4 values to staging row
                    half vals[8];
                    fused_dequant_fp4x8(packed, scale, vals);

                    // Write to staging (column-major for simdgroup_load)
                    for (uint i = 0; i < 8; ++i) {
                        B_staging[simd_id][b_row_in_tile][i] = vals[i];
                    }
                } else {
                    for (uint i = 0; i < 8; ++i) {
                        B_staging[simd_id][b_row_in_tile][i] = half(0.0h);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);

                // Load A fragment and B fragment, accumulate
                for (uint mi = 0; mi < EP_SG_M_TILES; ++mi) {
                    simdgroup_matrix<half, 8, 8> a_frag;
                    simdgroup_load(a_frag,
                                   &A_tile[sg_row_offset + mi * 8][kt * 8],
                                   EP_TILE_K);

                    simdgroup_matrix<half, 8, 8> b_frag;
                    simdgroup_load(b_frag, &B_staging[simd_id][0][0], 8);

                    simdgroup_multiply_accumulate(acc[mi][ni],
                                                  a_frag, b_frag, acc[mi][ni]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Epilogue: Apply bias + activation, then store ---
    store_results_epilogue(acc, C, bias, M, N, tg_row, tg_col,
                           sg_row_offset, sg_col_offset, simd_lane,
                           epilogue_mode, epilogue_staging);
}

kernel void marlin_gemm_fused_fp4_epilogue(
    device const half* A              [[buffer(0)]],
    device const uint32_t* B_packed   [[buffer(1)]],
    device const half* scales         [[buffer(2)]],
    device half* C                    [[buffer(3)]],
    constant uint4& dims              [[buffer(4)]],  // {M, N, K, group_size}
    device const half* bias           [[buffer(5)]],
    constant uint& epilogue_mode      [[buffer(6)]],
    uint2 tg_pos                      [[threadgroup_position_in_grid]],
    uint simd_id                      [[simdgroup_index_in_threadgroup]],
    uint simd_lane                    [[thread_index_in_simdgroup]]
) {
    // Threadgroup memory must be declared in kernel, passed to helpers
    threadgroup half A_tile[EP_TILE_M][EP_TILE_K];
    threadgroup half B_staging[EP_SIMDGROUPS_PER_TG][8][8];
    threadgroup half epilogue_staging[8][8];

    marlin_gemm_fused_fp4_epilogue_impl(A, B_packed, scales, C, dims, bias,
                                        epilogue_mode, tg_pos, simd_id,
                                        simd_lane, A_tile, B_staging,
                                        epilogue_staging);
}

constant uint k_epilogue_mode_fc [[function_constant(0)]];

kernel void marlin_gemm_fused_fp4_epilogue_fc(
    device const half* A              [[buffer(0)]],
    device const uint32_t* B_packed   [[buffer(1)]],
    device const half* scales         [[buffer(2)]],
    device half* C                    [[buffer(3)]],
    constant uint4& dims              [[buffer(4)]],  // {M, N, K, group_size}
    device const half* bias           [[buffer(5)]],
    uint2 tg_pos                      [[threadgroup_position_in_grid]],
    uint simd_id                      [[simdgroup_index_in_threadgroup]],
    uint simd_lane                    [[thread_index_in_simdgroup]]
) {
    // Threadgroup memory must be declared in kernel, passed to helpers
    threadgroup half A_tile[EP_TILE_M][EP_TILE_K];
    threadgroup half B_staging[EP_SIMDGROUPS_PER_TG][8][8];
    threadgroup half epilogue_staging[8][8];

    marlin_gemm_fused_fp4_epilogue_impl(A, B_packed, scales, C, dims, bias,
                                        k_epilogue_mode_fc, tg_pos, simd_id,
                                        simd_lane, A_tile, B_staging,
                                        epilogue_staging);
}

// ===========================================================================
// Standalone epilogue kernel (for use after a GEMM that doesn't fuse)
//
// Applies bias + activation to an existing output matrix in-place.
// Use when the GEMM kernel doesn't support fused epilogues, or when the
// same output needs different activations in different contexts.
//
// Grid: ceil(M/8) x ceil(N/64) threadgroups, 128 threads each.
// ===========================================================================

kernel void epilogue_standalone(
    device half* C                    [[buffer(0)]],
    device const half* bias           [[buffer(1)]],
    constant uint3& dims              [[buffer(2)]],  // {M, N, _unused}
    constant uint& epilogue_mode      [[buffer(3)]],
    uint2 tg_pos                      [[threadgroup_position_in_grid]],
    uint tid                          [[thread_index_in_threadgroup]]
) {
    const uint M = dims.x;
    const uint N = dims.y;

    const uint row = tg_pos.y * 8 + (tid / 8);
    const uint col_base = tg_pos.x * 64 + (tid % 8) * 8;

    if (row >= M) return;

    // Each thread processes 8 consecutive columns
    for (uint i = 0; i < 8; ++i) {
        uint col = col_base + i;
        if (col >= N) break;

        half val = C[row * N + col];

        if (has_bias(epilogue_mode)) {
            val += bias[col];
        }
        if (has_activation(epilogue_mode)) {
            val = apply_activation(val, epilogue_mode);
        }

        C[row * N + col] = val;
    }
}
