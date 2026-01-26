// kernels_autotune.metal - Parameterized GEMM kernel variants for auto-tuning
//
// Generates multiple tile-size configurations of the double-buffered pipelined
// FP4 GEMM kernel. Each variant uses a fixed tile geometry, allowing the Python
// autotuner to benchmark and select the optimal configuration per (M, N, K) shape.
//
// Metal lacks C++ templates for kernel functions, so we use preprocessor macros
// to stamp out fully specialized kernel bodies. Each variant has its own
// threadgroup memory allocation sized exactly for its tile dimensions.
//
// Tile geometry constraints:
//   - TILE_M, TILE_N: multiples of 8 (simdgroup_matrix is 8x8)
//   - TILE_K: multiple of 8 (K_TILES = TILE_K / 8)
//   - Threadgroup memory: 2 * (TILE_M*TILE_K + TILE_K*TILE_N) * 2 bytes <= 32KB
//   - Simdgroups per TG: TILE_M/8 * TILE_N/8 / (SG_M*SG_N) where SG_M*SG_N tiles
//     are handled per simdgroup
//
// Available variants (sorted by output tile area):
//   32x32x32:   Small tiles, high occupancy, good for small M
//   64x32x32:   Tall tiles, good for M >> N shapes
//   32x64x32:   Wide tiles, good for N >> M shapes
//   64x64x16:   Large output, shallow K per iter (high L/S ratio)
//   64x64x32:   Default (matches marlin_gemm.metal)
//   64x64x64:   Large K step, fewer main loop iters (memory-bound shapes)
//   128x64x32:  Very tall, good for large M with moderate N
//   64x128x32:  Very wide, good for large N with moderate M
//   128x128x16: Maximum output tile, minimum K per iter

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// ---------------------------------------------------------------------------
// Shared dequantization primitive (same as marlin_gemm.metal)
// ---------------------------------------------------------------------------

inline half dequant_fp4_at(uint nibble, half scale) {
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;

    half magnitude;
    if (exp_bits == 0) {
        magnitude = half(man_bit) * half(0.25h);
    } else {
        half power = half(1u << (exp_bits - 1));
        half mantissa = half(1.0h) + half(man_bit) * half(0.5h);
        magnitude = power * mantissa;
    }

    half result = sign_bit ? -magnitude : magnitude;
    return result * scale;
}

// ---------------------------------------------------------------------------
// Macro-generated kernel variant
//
// Parameters:
//   TM, TN, TK: tile dimensions
//   SGM, SGN:   simdgroup tile counts (each SG handles SGM x SGN 8x8 blocks)
//   NUM_SG:     simdgroups per threadgroup (TM/8/SGM_ROWS * TN/8/SGN_COLS)
//   SUFFIX:     kernel name suffix for disambiguation
//
// The macro generates:
//   1. Cooperative A tile loader
//   2. Cooperative B tile loader with fused FP4 dequant
//   3. Simdgroup MMA compute
//   4. Result store with boundary handling
//   5. Double-buffered pipelined mainloop
// ---------------------------------------------------------------------------

#define DEFINE_AUTOTUNE_GEMM(TM, TN, TK, SGM, SGN, NUM_SG, SUFFIX) \
\
inline void load_A_tile_##SUFFIX( \
    device const half* A, \
    threadgroup half (&A_buf)[TM][TK], \
    uint M, uint K, \
    uint tg_row, uint k_block, \
    uint thread_idx \
) { \
    const uint threads_per_tg = NUM_SG * 32; \
    const uint total_elems = TM * TK; \
    const uint elems_per_thread = total_elems / threads_per_tg; \
    const uint remainder = total_elems - elems_per_thread * threads_per_tg; \
    const uint my_count = elems_per_thread + (thread_idx < remainder ? 1 : 0); \
    const uint my_start = thread_idx * elems_per_thread + min(thread_idx, remainder); \
    for (uint i = 0; i < my_count; ++i) { \
        uint flat_idx = my_start + i; \
        uint row = flat_idx / TK; \
        uint col = flat_idx % TK; \
        uint global_row = tg_row + row; \
        uint global_col = k_block + col; \
        half val = 0.0h; \
        if (global_row < M && global_col < K) { \
            val = A[global_row * K + global_col]; \
        } \
        A_buf[row][col] = val; \
    } \
} \
\
inline void load_B_tile_dequant_##SUFFIX( \
    device const uint* B, \
    device const half* scales, \
    threadgroup half (&B_buf)[TK][TN], \
    uint K, uint N, \
    uint tg_col, uint k_block, \
    uint group_size, \
    uint thread_idx \
) { \
    const uint FP4_PER_UINT = 8; \
    const uint threads_per_tg = NUM_SG * 32; \
    const uint total_packed = (TK * TN) / FP4_PER_UINT; \
    const uint packed_per_thread = total_packed / threads_per_tg; \
    const uint remainder = total_packed - packed_per_thread * threads_per_tg; \
    const uint scale_tiles = (K + group_size - 1) / group_size; \
    const uint k_packs = (K + FP4_PER_UINT - 1) / FP4_PER_UINT; \
    const uint my_count = packed_per_thread + (thread_idx < remainder ? 1 : 0); \
    const uint my_start = thread_idx * packed_per_thread + min(thread_idx, remainder); \
    for (uint i = 0; i < my_count; ++i) { \
        uint flat_packed_idx = my_start + i; \
        uint n_idx = flat_packed_idx / (TK / FP4_PER_UINT); \
        uint k_group_in_tile = flat_packed_idx % (TK / FP4_PER_UINT); \
        uint global_n = tg_col + n_idx; \
        uint global_k_base = k_block + k_group_in_tile * FP4_PER_UINT; \
        uint scale_k = global_k_base / group_size; \
        half s = 1.0h; \
        if (global_n < N && global_k_base < K && scale_k < scale_tiles) { \
            s = scales[scale_k * N + global_n]; \
        } \
        uint packed = 0; \
        uint b_row = global_k_base / FP4_PER_UINT; \
        if (global_n < N && b_row < k_packs && global_k_base < K) { \
            packed = B[b_row * N + global_n]; \
        } \
        uint tile_k_base = k_group_in_tile * FP4_PER_UINT; \
        for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < TK; ++v) { \
            uint nibble = (packed >> (v * 4)) & 0xF; \
            if (n_idx < TN) { \
                B_buf[tile_k_base + v][n_idx] = dequant_fp4_at(nibble, s); \
            } \
        } \
    } \
} \
\
kernel void marlin_gemm_fp4_##SUFFIX( \
    device const half* A         [[buffer(0)]], \
    device const uint* B         [[buffer(1)]], \
    device const half* scales    [[buffer(2)]], \
    device half* C               [[buffer(3)]], \
    constant uint& M             [[buffer(4)]], \
    constant uint& N             [[buffer(5)]], \
    constant uint& K             [[buffer(6)]], \
    constant uint& group_size    [[buffer(7)]], \
    uint3 tgid                   [[threadgroup_position_in_grid]], \
    uint simd_lane               [[thread_index_in_simdgroup]], \
    uint simd_id                 [[simdgroup_index_in_threadgroup]] \
) { \
    const uint K_TILES_L = TK / 8; \
    const uint SG_ROWS = (TM / 8) / ((TM / 8 + 1) / 2); /* row groups */ \
    const uint SG_COLS = (TN / 8) / ((TN / 8 + 1) / 2); /* col groups */ \
    \
    threadgroup half A_tiles[2][TM][TK]; \
    threadgroup half B_tiles[2][TK][TN]; \
    \
    const uint tg_row = tgid.y * TM; \
    const uint tg_col = tgid.x * TN; \
    \
    /* Map simdgroups to output tile regions */ \
    const uint sg_rows = (TM / 8) / SGM; /* how many SG rows */ \
    const uint sg_cols = (TN / 8) / SGN; /* how many SG cols */ \
    const uint sg_row_id = simd_id / sg_cols; \
    const uint sg_col_id = simd_id % sg_cols; \
    const uint sg_row_offset = sg_row_id * (SGM * 8); \
    const uint sg_col_offset = sg_col_id * (SGN * 8); \
    \
    simdgroup_matrix<half, 8, 8> acc[SGM][SGN]; \
    for (uint mi = 0; mi < SGM; ++mi) { \
        for (uint ni = 0; ni < SGN; ++ni) { \
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h); \
        } \
    } \
    \
    const uint thread_idx = simd_id * 32 + simd_lane; \
    const uint num_k_tiles = (K + TK - 1) / TK; \
    uint buf_compute = 0; \
    \
    /* Prologue: load first K-tile */ \
    load_A_tile_##SUFFIX(A, A_tiles[0], M, K, tg_row, 0, thread_idx); \
    load_B_tile_dequant_##SUFFIX(B, scales, B_tiles[0], K, N, tg_col, 0, \
                                  group_size, thread_idx); \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    /* Main pipeline loop */ \
    for (uint kt = 0; kt < num_k_tiles; ++kt) { \
        uint k_offset = kt * TK; \
        uint next_k = k_offset + TK; \
        uint buf_load = 1 - buf_compute; \
        \
        if (next_k < K) { \
            load_A_tile_##SUFFIX(A, A_tiles[buf_load], M, K, \
                                  tg_row, next_k, thread_idx); \
            load_B_tile_dequant_##SUFFIX(B, scales, B_tiles[buf_load], K, N, \
                                          tg_col, next_k, group_size, thread_idx); \
        } \
        \
        /* Compute: simdgroup MMA over K sub-tiles */ \
        for (uint kk = 0; kk < K_TILES_L; ++kk) { \
            for (uint mi = 0; mi < SGM; ++mi) { \
                simdgroup_matrix<half, 8, 8> a_frag; \
                simdgroup_load(a_frag, \
                               &A_tiles[buf_compute][sg_row_offset + mi * 8][kk * 8], \
                               TK); \
                for (uint ni = 0; ni < SGN; ++ni) { \
                    simdgroup_matrix<half, 8, 8> b_frag; \
                    simdgroup_load(b_frag, \
                                   &B_tiles[buf_compute][kk * 8][sg_col_offset + ni * 8], \
                                   TN); \
                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag, b_frag, \
                                                  acc[mi][ni]); \
                } \
            } \
        } \
        \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        buf_compute = buf_load; \
    } \
    \
    /* Epilogue: store results */ \
    for (uint mi = 0; mi < SGM; ++mi) { \
        for (uint ni = 0; ni < SGN; ++ni) { \
            uint out_row = tg_row + sg_row_offset + mi * 8; \
            uint out_col = tg_col + sg_col_offset + ni * 8; \
            if (out_row + 8 <= M && out_col + 8 <= N) { \
                simdgroup_store(acc[mi][ni], C + out_row * N + out_col, N); \
            } else { \
                threadgroup half staging[8][8]; \
                simdgroup_store(acc[mi][ni], &staging[0][0], 8); \
                threadgroup_barrier(mem_flags::mem_threadgroup); \
                for (uint elem = simd_lane; elem < 64; elem += 32) { \
                    uint r = elem / 8; \
                    uint c = elem % 8; \
                    uint gr = out_row + r; \
                    uint gc = out_col + c; \
                    if (gr < M && gc < N) { \
                        C[gr * N + gc] = staging[r][c]; \
                    } \
                } \
                threadgroup_barrier(mem_flags::mem_threadgroup); \
            } \
        } \
    } \
}

// ---------------------------------------------------------------------------
// Instantiate kernel variants
//
// DEFINE_AUTOTUNE_GEMM(TILE_M, TILE_N, TILE_K, SG_M_tiles, SG_N_tiles, NUM_SG, suffix)
//
// Memory budget per variant (double-buffered):
//   bytes = 2 * (TM*TK + TK*TN) * 2
//
// Variant         TG Memory   Threads   Output/TG    Best For
// 32x32x32        8 KB        128       32x32=1024   Small M, high occupancy
// 64x32x32        12 KB       128       64x32=2048   M > N
// 32x64x32        12 KB       128       32x64=2048   N > M
// 64x64x16        16 KB       128       64x64=4096   Memory-bound (large tile, small K step)
// 64x64x32        16 KB       128       64x64=4096   Default balanced
// 64x64x64        32 KB       128       64x64=4096   Compute-bound (few iters, deep K)
// 128x64x32       24 KB       256       128x64=8192  Very large M
// 64x128x32       24 KB       256       64x128=8192  Very large N
// 128x128x16      32 KB       256       128x128=16K  Maximum tile (shallow K)
// ---------------------------------------------------------------------------

// --- 32x32x32: 4 SG, each handles 2x2 block of 8x8 (SGM=2, SGN=2) ---
// SG layout: 2x2 grid, each SG covers 16x16 of 32x32 output
DEFINE_AUTOTUNE_GEMM(32, 32, 32, 2, 2, 4, t32x32x32)

// --- 64x32x32: 4 SG, each handles 2x2 block of 8x8 ---
// SG layout: 4x1 -> each covers 16x32 of 64x32
// Actually 2x2 with SGM=4,SGN=2: each covers 32x16.. no.
// 64/8=8 M-blocks, 32/8=4 N-blocks. 4 SGs. Each gets 2x1: SGM=2, SGN=4
// Wait: 8 M-blocks / 4 SG_rows = 2 SG rows if sg_rows=4.
// Simpler: 4 SGs, each handles (64/8)/(64/8/SGM) M-tiles...
// With SGM=2, SGN=1, sg_rows = 8/2=4, sg_cols = 4/1=4 -> needs 16 SGs. Wrong.
// Let's use SGM=4, SGN=4, 1 SG: each covers 32x32. Only 1 SG for 64x32? No.
// The correct approach: TILE_M/8 = 8, TILE_N/8 = 4, total 8*4=32 sub-tiles.
// With 4 SGs and SGM=2, SGN=4: each SG handles 2*4=8 sub-tiles, need 32/8=4 SGs. Good.
// sg_rows = 8/2 = 4, sg_cols = 4/4 = 1 -> simd_id / 1 = simd_id (0..3 row), simd_id % 1 = 0
// sg_row_offset = simd_id * 16, sg_col_offset = 0. Each SG covers 16 rows x 32 cols.
DEFINE_AUTOTUNE_GEMM(64, 32, 32, 2, 4, 4, t64x32x32)

// --- 32x64x32: 4 SG, SGM=2, SGN=2 ---
// 32/8=4 M-blocks, 64/8=8 N-blocks. SGM=2,SGN=2: each SG = 4 sub-tiles.
// sg_rows = 4/2=2, sg_cols=8/2=4 -> needs 8 SGs. Too many.
// SGM=4, SGN=2: each SG = 8 sub-tiles. sg_rows=4/4=1, sg_cols=8/2=4 -> 4 SGs.
// sg_row_offset = 0, sg_col_offset = simd_id * 16. Each covers 32 rows x 16 cols.
DEFINE_AUTOTUNE_GEMM(32, 64, 32, 4, 2, 4, t32x64x32)

// --- 64x64x16: 4 SG, SGM=2, SGN=4 (same layout as default) ---
DEFINE_AUTOTUNE_GEMM(64, 64, 16, 2, 4, 4, t64x64x16)

// --- 64x64x32: 4 SG, SGM=2, SGN=4 (matches default kernel) ---
DEFINE_AUTOTUNE_GEMM(64, 64, 32, 2, 4, 4, t64x64x32)

// --- 64x64x64: 4 SG, SGM=2, SGN=4 ---
// Double-buffered: 2*(64*64 + 64*64)*2 = 32768 bytes. Exactly 32KB limit.
DEFINE_AUTOTUNE_GEMM(64, 64, 64, 2, 4, 4, t64x64x64)

// --- 128x64x32: 8 SG (256 threads), SGM=2, SGN=4 ---
// 128/8=16 M-blocks, 64/8=8 N-blocks. SGM=2, SGN=4: each SG = 8 sub-tiles.
// sg_rows = 16/2=8, sg_cols=8/4=2 -> needs 16 SGs. Too many.
// SGM=4, SGN=4: each SG = 16 sub-tiles. sg_rows=16/4=4, sg_cols=8/4=2 -> 8 SGs.
// 8 SGs * 32 = 256 threads.
DEFINE_AUTOTUNE_GEMM(128, 64, 32, 4, 4, 8, t128x64x32)

// --- 64x128x32: 8 SG (256 threads), SGM=4, SGN=4 ---
// 64/8=8 M-blocks, 128/8=16 N-blocks. SGM=4,SGN=4: each = 16 sub-tiles.
// sg_rows=8/4=2, sg_cols=16/4=4 -> 8 SGs. Good.
DEFINE_AUTOTUNE_GEMM(64, 128, 32, 4, 4, 8, t64x128x32)

// --- 128x128x16: 8 SG (256 threads), SGM=4, SGN=8 ---
// 128/8=16 M-blocks, 128/8=16 N-blocks. SGM=4,SGN=8: each = 32 sub-tiles.
// sg_rows=16/4=4, sg_cols=16/8=2 -> 8 SGs. Good.
// Memory: 2*(128*16 + 16*128)*2 = 2*4096*2 = 16384 bytes. Well within budget.
DEFINE_AUTOTUNE_GEMM(128, 128, 16, 4, 8, 8, t128x128x16)

// ---------------------------------------------------------------------------
// Simple macro-style variants (MxNxK suffix) for external autotuning lookup.
// These keep stable kernel names like marlin_gemm_fp4_64x64x32.
// ---------------------------------------------------------------------------

#define GEMM_SGM_32x32x16 2
#define GEMM_SGN_32x32x16 2
#define GEMM_NUM_SG_32x32x16 4

#define GEMM_SGM_64x64x32 2
#define GEMM_SGN_64x64x32 4
#define GEMM_NUM_SG_64x64x32 4

#define GEMM_SGM_128x64x32 4
#define GEMM_SGN_128x64x32 4
#define GEMM_NUM_SG_128x64x32 8

#define GEMM_SGM_64x128x32 4
#define GEMM_SGN_64x128x32 4
#define GEMM_NUM_SG_64x128x32 8

#define DEFINE_GEMM_VARIANT(TILE_M_SIZE, TILE_N_SIZE, TILE_K_SIZE, SUFFIX) \
    DEFINE_AUTOTUNE_GEMM(TILE_M_SIZE, TILE_N_SIZE, TILE_K_SIZE, \
        GEMM_SGM_##SUFFIX, GEMM_SGN_##SUFFIX, GEMM_NUM_SG_##SUFFIX, SUFFIX)

DEFINE_GEMM_VARIANT(32, 32, 16, 32x32x16)
DEFINE_GEMM_VARIANT(64, 64, 32, 64x64x32)
DEFINE_GEMM_VARIANT(128, 64, 32, 128x64x32)
DEFINE_GEMM_VARIANT(64, 128, 32, 64x128x32)
