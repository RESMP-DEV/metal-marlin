// sparse_gemm.metal - Sparse 2:4 structured sparsity GEMM with FP4 weights
//
// Computes C = A @ sparse_dequant(B_sparse, B_meta, scales)
// where B has 2:4 structured sparsity: for every 4 elements along K,
// exactly 2 are non-zero and stored in compressed form.
//
// Sparsity format (NVIDIA 2:4):
//   - B_sparse[K/2/8, N]: packed FP4, only the 2 non-zero values per 4-group
//     are stored, halving the K dimension storage
//   - B_meta[K/16, N]: 2-bit metadata indices per 4-group, indicating which
//     2 of the 4 positions hold non-zero values
//     Each uint32 encodes metadata for 16 K-positions (4 groups of 4, each
//     group uses 4 bits = 2x 2-bit indices)
//
// Kernel variants:
//   1. marlin_gemm_sparse_fp4          - Double-buffered, reconstructs dense tile
//   2. marlin_gemm_sparse_fp4_fused    - Fused dequant, register-resident scatter
//
// The sparse kernel uses the same tile dimensions as the dense variant
// (64x64x32) but the B tile loader scatters values to their correct K
// positions based on metadata, inserting zeros for pruned positions.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// ---------------------------------------------------------------------------
// Tile dimensions (same as dense GEMM for compute compatibility)
// ---------------------------------------------------------------------------

constant constexpr uint SP_TILE_M = 64;
constant constexpr uint SP_TILE_N = 64;
constant constexpr uint SP_TILE_K = 32;

constant constexpr uint SP_K_TILES = SP_TILE_K / 8;  // 4
constant constexpr uint SP_SIMDGROUPS_PER_TG = 4;
constant constexpr uint SP_THREADS_PER_TG = SP_SIMDGROUPS_PER_TG * 32;  // 128

constant constexpr uint SP_SG_M_TILES = 8;
constant constexpr uint SP_SG_N_TILES = 2;

constant constexpr uint SP_FP4_PER_UINT = 8;
constant constexpr uint SP_NUM_BUFFERS = 2;

// 2:4 sparsity constants
// Every group of 4 dense K elements has exactly 2 non-zero values.
// The compressed representation stores only those 2 values.
constant constexpr uint SPARSE_GROUP = 4;       // Dense elements per sparsity group
constant constexpr uint SPARSE_NNZ = 2;         // Non-zeros per group
constant constexpr uint SPARSE_RATIO = 2;       // Compression ratio (K_dense / K_sparse)

// Metadata encoding:
// Each sparsity group (4 dense positions) uses 4 metadata bits:
//   - 2 bits for index of first non-zero (0-3)
//   - 2 bits for index of second non-zero (0-3)
// A uint32 of metadata covers 8 sparsity groups = 32 dense K positions.
// (8 groups * 4 bits/group = 32 bits)
constant constexpr uint META_BITS_PER_GROUP = 4;
constant constexpr uint META_GROUPS_PER_UINT = 8;  // 32 bits / 4 bits
constant constexpr uint META_DENSE_K_PER_UINT = META_GROUPS_PER_UINT * SPARSE_GROUP;  // 32

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

inline uint sp_div_ceil(uint a, uint b) {
    return (a + b - 1) / b;
}

inline uint div_ceil(uint a, uint b) {
    return (a + b - 1) / b;
}

// ---------------------------------------------------------------------------
// FP4 E2M1 dequantization (same as dense kernel)
// ---------------------------------------------------------------------------

inline half sparse_dequant_fp4(uint nibble) {
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;

    half sub_mag = half(man_bit) * half(0.5h);
    half norm_mag = half(1u << (exp_bits - 1)) * (half(1.0h) + half(man_bit) * half(0.5h));

    half magnitude = select(norm_mag, sub_mag, exp_bits == 0);
    return select(magnitude, -magnitude, bool(sign_bit));
}

// ---------------------------------------------------------------------------
// Sparse B tile loader with metadata-driven scatter
//
// This function reconstructs a dense B_buf[TILE_K][TILE_N] tile from the
// compressed sparse representation. For each column n:
//   1. Read the metadata to determine which K positions are non-zero
//   2. Read the compressed FP4 values (only K/2 stored)
//   3. Dequantize and scatter to the correct positions, zeroing the rest
//
// Memory layout:
//   B_sparse is [K_sparse/8, N] where K_sparse = K/2 (half the dense K)
//   B_meta is [K/META_DENSE_K_PER_UINT, N] (one uint32 covers 32 dense K positions)
//   scales is [K/group_size, N]
// ---------------------------------------------------------------------------

inline void load_B_tile_sparse_dequant(
    device const uint* B_sparse,     // [K_sparse/8, N] packed FP4 (compressed)
    device const uint* B_meta,       // [K/32, N] metadata
    device const half* scales,       // [K/group_size, N]
    threadgroup half (&B_buf)[SP_TILE_K][SP_TILE_N],
    uint K,                          // Dense K dimension
    uint N,
    uint tg_col,                     // Starting column of this threadgroup's tile
    uint k_block,                    // Starting dense K position of this tile
    uint group_size,
    uint thread_idx
) {
    // Zero the entire B tile first. Sparse tiles have many zeros, and
    // zeroing upfront lets the scatter loop skip explicit zero writes.
    {
        const uint elems_per_thread = (SP_TILE_K * SP_TILE_N) / SP_THREADS_PER_TG;  // 16
        for (uint i = 0; i < elems_per_thread; ++i) {
            uint flat_idx = thread_idx * elems_per_thread + i;
            uint row = flat_idx / SP_TILE_N;
            uint col = flat_idx % SP_TILE_N;
            B_buf[row][col] = half(0.0h);
        }
    }

    // Each thread processes a subset of (sparsity_group, column) pairs.
    // Within the tile, we have SP_TILE_K/SPARSE_GROUP = 8 sparsity groups
    // per column, and SP_TILE_N = 64 columns. Total work items = 512.
    // With 128 threads, each thread handles 4 work items.

    const uint groups_in_tile = SP_TILE_K / SPARSE_GROUP;  // 8
    const uint total_work = groups_in_tile * SP_TILE_N;     // 512
    const uint work_per_thread = total_work / SP_THREADS_PER_TG;  // 4

    const uint K_sparse = K / SPARSE_RATIO;

    for (uint w = 0; w < work_per_thread; ++w) {
        uint work_idx = thread_idx * work_per_thread + w;
        uint n_local = work_idx / groups_in_tile;   // Column within tile [0..63]
        uint g_local = work_idx % groups_in_tile;   // Sparsity group within tile [0..7]

        uint global_n = tg_col + n_local;
        if (global_n >= N) continue;

        // Dense K range for this sparsity group
        uint dense_k_base = k_block + g_local * SPARSE_GROUP;
        if (dense_k_base >= K) continue;

        // --- Read metadata ---
        // B_meta layout: [K/META_DENSE_K_PER_UINT, N]
        // Each uint32 covers META_DENSE_K_PER_UINT=32 dense K positions.
        // Within that uint32, each sparsity group uses 4 bits.
        uint meta_row = dense_k_base / META_DENSE_K_PER_UINT;
        uint group_in_meta_word = (dense_k_base % META_DENSE_K_PER_UINT) / SPARSE_GROUP;
        uint meta_word = B_meta[meta_row * N + global_n];
        uint meta_bits = (meta_word >> (group_in_meta_word * META_BITS_PER_GROUP)) & 0xF;

        // Extract the two 2-bit indices (positions of non-zero elements)
        uint idx0 = meta_bits & 0x3;         // First non-zero position [0..3]
        uint idx1 = (meta_bits >> 2) & 0x3;  // Second non-zero position [0..3]

        // --- Read compressed FP4 values ---
        // In the compressed representation, each sparsity group of 4 contributes
        // 2 FP4 values (1 byte = 2 nibbles). The sparse K dimension = K/2.
        // Two consecutive FP4 values for this group are at sparse_k_base.
        uint sparse_k_base = (dense_k_base / SPARSE_GROUP) * SPARSE_NNZ;  // Position in compressed stream

        // Read scale for this group
        uint scale_group = dense_k_base / group_size;
        half s = scales[scale_group * N + global_n];

        // The FP4 values are packed 8 per uint32.
        // sparse_k_base gives us the position of the first of the 2 non-zero values.
        uint pack_idx = sparse_k_base / SP_FP4_PER_UINT;
        uint nibble_offset = sparse_k_base % SP_FP4_PER_UINT;

        uint packed = 0;
        if (pack_idx < (K_sparse / SP_FP4_PER_UINT)) {
            packed = B_sparse[pack_idx * N + global_n];
        }

        // Extract the two FP4 nibbles
        uint nibble0 = (packed >> (nibble_offset * 4)) & 0xF;
        uint nibble1 = (packed >> ((nibble_offset + 1) * 4)) & 0xF;

        // Dequantize
        half val0 = sparse_dequant_fp4(nibble0) * s;
        half val1 = sparse_dequant_fp4(nibble1) * s;

        // --- Scatter to dense positions ---
        uint tile_k0 = g_local * SPARSE_GROUP + idx0;
        uint tile_k1 = g_local * SPARSE_GROUP + idx1;

        if (tile_k0 < SP_TILE_K) {
            B_buf[tile_k0][n_local] = val0;
        }
        if (tile_k1 < SP_TILE_K) {
            B_buf[tile_k1][n_local] = val1;
        }
    }
}

// ---------------------------------------------------------------------------
// Cooperative A tile loader (identical to dense kernel)
// ---------------------------------------------------------------------------

inline void sparse_load_A_tile(
    device const half* A,
    threadgroup half (&A_buf)[SP_TILE_M][SP_TILE_K],
    uint M, uint K,
    uint tg_row, uint k_block,
    uint thread_idx
) {
    const uint elems_per_thread = (SP_TILE_M * SP_TILE_K) / SP_THREADS_PER_TG;  // 16
    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        uint row = flat_idx / SP_TILE_K;
        uint col = flat_idx % SP_TILE_K;
        uint global_row = tg_row + row;
        uint global_col = k_block + col;

        half val = (global_row < M && global_col < K)
                   ? A[global_row * K + global_col]
                   : half(0.0h);
        A_buf[row][col] = val;
    }
}

// ---------------------------------------------------------------------------
// Simdgroup compute (identical to dense kernel)
// ---------------------------------------------------------------------------

inline void sparse_compute_from_tiles(
    threadgroup const half (&A_buf)[SP_TILE_M][SP_TILE_K],
    threadgroup const half (&B_buf)[SP_TILE_K][SP_TILE_N],
    thread simdgroup_matrix<half, 8, 8> acc[SP_SG_M_TILES][SP_SG_N_TILES],
    uint sg_row_offset,
    uint sg_col_offset
) {
    for (uint kt = 0; kt < SP_K_TILES; ++kt) {
        for (uint mi = 0; mi < SP_SG_M_TILES; ++mi) {
            simdgroup_matrix<half, 8, 8> a_frag;
            simdgroup_load(a_frag,
                           &A_buf[sg_row_offset + mi * 8][kt * 8],
                           SP_TILE_K);

            for (uint ni = 0; ni < SP_SG_N_TILES; ++ni) {
                simdgroup_matrix<half, 8, 8> b_frag;
                simdgroup_load(b_frag,
                               &B_buf[kt * 8][sg_col_offset + ni * 8],
                               SP_TILE_N);

                simdgroup_multiply_accumulate(acc[mi][ni],
                                              a_frag, b_frag, acc[mi][ni]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Store results (identical to dense kernel)
// ---------------------------------------------------------------------------

inline void sparse_store_results(
    thread simdgroup_matrix<half, 8, 8> acc[SP_SG_M_TILES][SP_SG_N_TILES],
    device half* C,
    threadgroup half (&staging)[8][8],
    uint M, uint N,
    uint tg_row, uint tg_col,
    uint sg_row_offset, uint sg_col_offset,
    uint simd_lane
) {
    for (uint mi = 0; mi < SP_SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SP_SG_N_TILES; ++ni) {
            uint out_row = tg_row + sg_row_offset + mi * 8;
            uint out_col = tg_col + sg_col_offset + ni * 8;

            if (out_row + 8 <= M && out_col + 8 <= N) {
                simdgroup_store(acc[mi][ni],
                                C + out_row * N + out_col,
                                N);
            } else {
                // Edge case: partial tile - use staging buffer
                // Only threads with valid output coordinates participate
                bool needs_partial_store = (out_row < M && out_col < N);
                if (needs_partial_store) {
                    simdgroup_store(acc[mi][ni], &staging[0][0], 8);
                }
                // Barrier must be hit by all threads in the simdgroup uniformly
                simdgroup_barrier(mem_flags::mem_threadgroup);

                if (needs_partial_store) {
                    for (uint elem = simd_lane; elem < 64; elem += 32) {
                        uint r = elem / 8;
                        uint c = elem % 8;
                        if (out_row + r < M && out_col + c < N) {
                            C[(out_row + r) * N + out_col + c] = staging[r][c];
                        }
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

// ===========================================================================
// Kernel 1: Double-buffered pipelined sparse 2:4 FP4 GEMM
//
// C[M,N] = A[M,K] @ sparse_dequant(B_sparse[K/2/8,N], B_meta[K/32,N], scales)
//
// The B tile loader reads compressed weights and metadata, then scatters the
// non-zero values to their correct positions in a zeroed dense tile. The
// compute and store phases are identical to the dense GEMM.
//
// Dispatch: Grid ceil(N/64) x ceil(M/64), threadgroup 128 threads.
//
// Compared to the dense kernel:
//   - B reads are halved (only K/2 values stored)
//   - Additional metadata reads (1 uint32 per 32 dense K positions per column)
//   - Tile zero-initialization adds ~16 stores per thread per iteration
//   - The simdgroup MMA compute is identical (same tile dimensions)
//
// The net effect: bandwidth-bound workloads see ~1.8x speedup from halved B
// reads, while compute-bound workloads see the full 2x theoretical gain since
// the reconstructed tile has 50% zeros that contribute zero to the dot product
// but the hardware MMA doesn't skip them.
// ===========================================================================

kernel void marlin_gemm_sparse_fp4(
    device const half* A             [[buffer(0)]],   // [M, K] dense activations
    device const uint* B_sparse      [[buffer(1)]],   // [K/2/8, N] compressed FP4
    device const uint* B_meta        [[buffer(2)]],   // [K/32, N] sparsity metadata
    device const half* scales        [[buffer(3)]],   // [K/group_size, N] per-group scales
    device half* C                   [[buffer(4)]],   // [M, N] output
    constant uint& M                 [[buffer(5)]],
    constant uint& N                 [[buffer(6)]],
    constant uint& K                 [[buffer(7)]],   // Dense K dimension
    constant uint& group_size        [[buffer(8)]],
    uint3 tgid                       [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]]
) {
    // Double-buffered threadgroup memory
    threadgroup half A_tiles[SP_NUM_BUFFERS][SP_TILE_M][SP_TILE_K];
    threadgroup half B_tiles[SP_NUM_BUFFERS][SP_TILE_K][SP_TILE_N];
    threadgroup half store_staging[8][8];  // Staging buffer for edge-case stores

    const uint tg_row = tgid.y * SP_TILE_M;
    const uint tg_col = tgid.x * SP_TILE_N;

    const uint sg_row_offset = 0;  // All simdgroups cover all rows
    const uint sg_col_offset = simd_id * (SP_SG_N_TILES * 8);

    simdgroup_matrix<half, 8, 8> acc[SP_SG_M_TILES][SP_SG_N_TILES];
    for (uint mi = 0; mi < SP_SG_M_TILES; ++mi)
        for (uint ni = 0; ni < SP_SG_N_TILES; ++ni)
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = sp_div_ceil(K, SP_TILE_K);
    uint buf_compute = 0;

    // --- Prologue: Load first K-tile into buffer 0 ---
    sparse_load_A_tile(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
    load_B_tile_sparse_dequant(B_sparse, B_meta, scales, B_tiles[0],
                               K, N, tg_col, 0, group_size, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Main pipeline loop ---
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_offset = kt * SP_TILE_K;
        uint next_k = k_offset + SP_TILE_K;
        uint buf_load = 1 - buf_compute;

        // Load NEXT K-tile into alternate buffer (overlapped with compute)
        if (next_k < K) {
            sparse_load_A_tile(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
            load_B_tile_sparse_dequant(B_sparse, B_meta, scales, B_tiles[buf_load],
                                       K, N, tg_col, next_k, group_size, thread_idx);
        }

        // Compute on current buffer
        sparse_compute_from_tiles(A_tiles[buf_compute], B_tiles[buf_compute],
                                  acc, sg_row_offset, sg_col_offset);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // --- Store results ---
    sparse_store_results(acc, C, store_staging, M, N, tg_row, tg_col,
                         sg_row_offset, sg_col_offset, simd_lane);
}

// ===========================================================================
// Kernel 2: Fused sparse 2:4 FP4 GEMM (register-resident dequant + scatter)
//
// KEY INNOVATION: Combine the fused dequant approach with sparse scatter.
// Each simdgroup independently reconstructs its 8x8 B sub-tile by reading
// compressed values + metadata, dequantizing in registers, and scattering
// to a per-simdgroup staging buffer.
//
// Benefits over the double-buffered sparse kernel:
//   - No full threadgroup barrier for B tile load
//   - Reduced threadgroup memory (no B_tiles[2][32][64])
//   - Per-simdgroup metadata decode avoids load imbalance
//   - Simdgroup_barrier (32 threads) instead of threadgroup_barrier (128)
//
// Trade-off:
//   - Each metadata word and packed value may be loaded multiple times across
//     K sub-tiles (L2 cache absorbs this on M4 Max with 4MB L2)
//   - More complex per-lane logic for sparse indexing
//
// Dispatch: Grid ceil(N/64) x ceil(M/64), threadgroup 128 threads.
// ===========================================================================

kernel void marlin_gemm_sparse_fp4_fused(
    device const half* A             [[buffer(0)]],   // [M, K] dense activations
    device const uint* B_sparse      [[buffer(1)]],   // [K/2/8, N] compressed FP4
    device const uint* B_meta        [[buffer(2)]],   // [K/32, N] sparsity metadata
    device const half* scales        [[buffer(3)]],   // [K/group_size, N] per-group scales
    device half* C                   [[buffer(4)]],   // [M, N] output
    constant uint& M                 [[buffer(5)]],
    constant uint& N                 [[buffer(6)]],
    constant uint& K                 [[buffer(7)]],   // Dense K dimension
    constant uint& group_size        [[buffer(8)]],
    uint3 tgid                       [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]]
) {
    // Double-buffered A tile in threadgroup memory (shared across simdgroups)
    threadgroup half A_tiles[2][SP_TILE_M][SP_TILE_K];
    // Per-simdgroup B staging for the current 8x8 sub-tile
    threadgroup half B_staging[SP_SIMDGROUPS_PER_TG][8][8];

    const uint tg_row = tgid.y * SP_TILE_M;
    const uint tg_col = tgid.x * SP_TILE_N;
    const uint sg_row_offset = 0;  // All simdgroups cover all rows
    const uint sg_col_offset = simd_id * (SP_SG_N_TILES * 8);

    simdgroup_matrix<half, 8, 8> acc[SP_SG_M_TILES][SP_SG_N_TILES];
    for (uint mi = 0; mi < SP_SG_M_TILES; ++mi)
        for (uint ni = 0; ni < SP_SG_N_TILES; ++ni)
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint K_sparse = K / SPARSE_RATIO;

    // =========================================================================
    // Main K-reduction loop
    // =========================================================================

    const uint k_tiles = div_ceil(K, SP_TILE_K);
    uint buf_idx = 0;

    if (k_tiles > 0) {
        // Initial A tile load
        {
            const uint elems_per_thread = (SP_TILE_M * SP_TILE_K) / SP_THREADS_PER_TG;
            for (uint i = 0; i < elems_per_thread; ++i) {
                uint flat_idx = thread_idx * elems_per_thread + i;
                uint row = flat_idx / SP_TILE_K;
                uint col = flat_idx % SP_TILE_K;
                uint global_row = tg_row + row;
                uint global_col = col;

                half val = (global_row < M && global_col < K)
                           ? A[global_row * K + global_col]
                           : half(0.0h);
                A_tiles[0][row][col] = val;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // =====================================================================
        // Inner loop: fused sparse dequant + scatter + compute
        //
        // For each K sub-tile (kt = 0..3, covering 8 dense K values):
        //   The 8 dense K positions contain 4 sparsity groups of 2 values each
        //   (since group_size=4 for 2:4, 8/4 = 2 sparsity groups per sub-tile).
        //   Each simdgroup reconstructs the 8x8 sub-tile for its output columns
        //   by reading metadata + compressed values and scattering.
        // =====================================================================

        for (uint k_tile = 0; k_tile < k_tiles; ++k_tile) {
            uint k_block = k_tile * SP_TILE_K;
            uint next_k = k_block + SP_TILE_K;
            uint buf_load = 1 - buf_idx;

            if (next_k < K) {
                const uint elems_per_thread = (SP_TILE_M * SP_TILE_K) / SP_THREADS_PER_TG;
                for (uint i = 0; i < elems_per_thread; ++i) {
                    uint flat_idx = thread_idx * elems_per_thread + i;
                    uint row = flat_idx / SP_TILE_K;
                    uint col = flat_idx % SP_TILE_K;
                    uint global_row = tg_row + row;
                    uint global_col = next_k + col;

                    half val = (global_row < M && global_col < K)
                               ? A[global_row * K + global_col]
                               : half(0.0h);
                    A_tiles[buf_load][row][col] = val;
                }
            }

            for (uint kt = 0; kt < SP_K_TILES; ++kt) {
                uint k_sub_base = k_block + kt * 8;  // Dense K start for this sub-tile

            for (uint mi = 0; mi < SP_SG_M_TILES; ++mi) {
                simdgroup_matrix<half, 8, 8> a_frag;
                simdgroup_load(a_frag,
                               &A_tiles[buf_idx][sg_row_offset + mi * 8][kt * 8],
                               SP_TILE_K);

                for (uint ni = 0; ni < SP_SG_N_TILES; ++ni) {
                    uint b_col_base = tg_col + sg_col_offset + ni * 8;

                    // --- Fused sparse B dequant: lanes 0-7 each handle one column ---
                    if (simd_lane < 8) {
                        uint b_col = b_col_base + simd_lane;

                        // Zero the staging column (8 rows) for this lane
                        for (uint r = 0; r < 8; ++r)
                            B_staging[simd_id][r][simd_lane] = half(0.0h);

                        if (b_col < N && k_sub_base < K) {
                            // This 8-element K sub-tile contains 2 sparsity groups.
                            // Decode metadata once and load the 4 FP4 values in one packed word.
                            uint dense_k0 = k_sub_base;
                            uint meta_row = dense_k0 / META_DENSE_K_PER_UINT;
                            uint group_in_word0 = (dense_k0 % META_DENSE_K_PER_UINT) / SPARSE_GROUP;
                            uint meta_word = B_meta[meta_row * N + b_col];

                            uint meta_bits0 = (meta_word >> (group_in_word0 * META_BITS_PER_GROUP)) & 0xF;
                            uint idx0 = meta_bits0 & 0x3;
                            uint idx1 = (meta_bits0 >> 2) & 0x3;

                            uint sparse_k = (dense_k0 / SPARSE_GROUP) * SPARSE_NNZ;
                            uint pack_idx = sparse_k / SP_FP4_PER_UINT;
                            uint nibble_off = sparse_k % SP_FP4_PER_UINT;

                            uint packed = 0;
                            if (pack_idx < (K_sparse / SP_FP4_PER_UINT)) {
                                packed = B_sparse[pack_idx * N + b_col];
                            }

                            uint nib0 = (packed >> (nibble_off * 4)) & 0xF;
                            uint nib1 = (packed >> ((nibble_off + 1) * 4)) & 0xF;
                            uint nib2 = (packed >> ((nibble_off + 2) * 4)) & 0xF;
                            uint nib3 = (packed >> ((nibble_off + 3) * 4)) & 0xF;

                            uint scale_group0 = dense_k0 / group_size;
                            half s0 = scales[scale_group0 * N + b_col];
                            half v0 = sparse_dequant_fp4(nib0) * s0;
                            half v1 = sparse_dequant_fp4(nib1) * s0;

                            B_staging[simd_id][idx0][simd_lane] = v0;
                            B_staging[simd_id][idx1][simd_lane] = v1;

                            uint dense_k1 = dense_k0 + SPARSE_GROUP;
                            if (dense_k1 < K) {
                                uint meta_bits1 = (meta_word >> ((group_in_word0 + 1) * META_BITS_PER_GROUP)) & 0xF;
                                uint idx2 = meta_bits1 & 0x3;
                                uint idx3 = (meta_bits1 >> 2) & 0x3;

                                uint scale_group1 = dense_k1 / group_size;
                                half s1 = scales[scale_group1 * N + b_col];
                                half v2 = sparse_dequant_fp4(nib2) * s1;
                                half v3 = sparse_dequant_fp4(nib3) * s1;

                                B_staging[simd_id][SPARSE_GROUP + idx2][simd_lane] = v2;
                                B_staging[simd_id][SPARSE_GROUP + idx3][simd_lane] = v3;
                            }
                        }
                    }

                    simdgroup_barrier(mem_flags::mem_threadgroup);

                    // Load B fragment from staging and compute
                    simdgroup_matrix<half, 8, 8> b_frag;
                    simdgroup_load(b_frag, &B_staging[simd_id][0][0], 8);

                    simdgroup_multiply_accumulate(acc[mi][ni],
                                                  a_frag, b_frag, acc[mi][ni]);
                }
            }
        }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf_idx = buf_load;
        }
    }

    // =========================================================================
    // Store results
    // =========================================================================

    for (uint mi = 0; mi < SP_SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SP_SG_N_TILES; ++ni) {
            uint out_row = tg_row + sg_row_offset + mi * 8;
            uint out_col = tg_col + sg_col_offset + ni * 8;

            if (out_row + 8 <= M && out_col + 8 <= N) {
                simdgroup_store(acc[mi][ni], C + out_row * N + out_col, N);
            } else {
                // Edge case: partial tile - use staging buffer
                // Only threads with valid output coordinates participate
                threadgroup half out_staging[8][8];
                bool needs_partial_store = (out_row < M && out_col < N);
                if (needs_partial_store) {
                    simdgroup_store(acc[mi][ni], &out_staging[0][0], 8);
                }
                // Barrier must be hit by all threads in the simdgroup uniformly
                simdgroup_barrier(mem_flags::mem_threadgroup);
                if (needs_partial_store) {
                    for (uint elem = simd_lane; elem < 64; elem += 32) {
                        uint r = elem / 8;
                        uint c = elem % 8;
                        if (out_row + r < M && out_col + c < N) {
                            C[(out_row + r) * N + out_col + c] = out_staging[r][c];
                        }
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}
