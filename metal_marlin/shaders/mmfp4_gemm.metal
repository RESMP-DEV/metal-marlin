#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// --- FP4 E2M1 branchless dequant (pure ALU, no LUT) ---
__attribute__((always_inline))
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
__attribute__((always_inline))
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

kernel void mmfp4_gemm(
    device const half* A [[buffer(0)]],
    device const uint* B_packed [[buffer(1)]],
    device const half* B_scales [[buffer(2)]],
    device half* C [[buffer(3)]],
    device const uint* M_p [[buffer(4)]],
    device const uint* K_p [[buffer(5)]],
    device const uint* N_p [[buffer(6)]],
    device const uint* group_size_p [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]])
{
    const uint M = M_p[0];
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

    threadgroup half A_tile[TILE_M][TILE_K];
    threadgroup half B_tile[TILE_K][TILE_N];

    uint sg_row_base = (simd_id / 2) * 32;
    uint sg_col_base = (simd_id % 2) * 32;

    // Use float accumulators to prevent overflow during accumulation
    simdgroup_matrix<float, 8, 8> acc[4][4];
    for (uint mi = 0; mi < 4; ++mi)
        for (uint ni = 0; ni < 4; ++ni)
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k_tile = 0; k_tile < K; k_tile += TILE_K) {
        uint elems_per_thread = (TILE_M * TILE_K + THREADS_PER_TG - 1) / THREADS_PER_TG;
        for (uint e = 0; e < elems_per_thread; ++e) {
            uint flat_idx = thread_idx * elems_per_thread + e;
            if (flat_idx < TILE_M * TILE_K) {
                uint a_row = flat_idx / TILE_K;
                uint a_col = flat_idx % TILE_K;
                uint global_row = tgid.y * TILE_M + a_row;
                uint global_col = k_tile + a_col;
                half val = (global_row < M && global_col < K) ? A[global_row * K + global_col] : half(0.0h);
                A_tile[a_row][a_col] = val;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k_sub = 0; k_sub < TILE_K; k_sub += 8) {
            uint k_global = k_tile + k_sub;
            uint k_pack_idx = k_global / FP4_PER_UINT;
            uint group_idx = k_global / GROUP_SIZE;

            for (uint b_col = thread_idx; b_col < TILE_N; b_col += THREADS_PER_TG) {
                uint global_col = tgid.x * TILE_N + b_col;

                if (global_col < N && k_pack_idx < K / FP4_PER_UINT) {
                    uint32_t packed = B_packed[k_pack_idx * N + global_col];
                    half scale = B_scales[group_idx * N + global_col];
                    half dequant_vals[8];
                    dequant_fp4x8(packed, scale, dequant_vals);
                    for (uint k_off = 0; k_off < 8; ++k_off) {
                        B_tile[k_sub + k_off][b_col] = dequant_vals[k_off];
                    }
                } else {
                    for (uint k_off = 0; k_off < 8; ++k_off) {
                        B_tile[k_sub + k_off][b_col] = half(0.0h);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint mi = 0; mi < 4; ++mi) {
                uint row_idx = sg_row_base + mi * 8;
                for (uint ni = 0; ni < 4; ++ni) {
                    uint col_idx = sg_col_base + ni * 8;
                    simdgroup_matrix<half, 8, 8> a_frag;
                    simdgroup_load(a_frag, &A_tile[row_idx][k_sub], TILE_K);
                    simdgroup_matrix<half, 8, 8> b_frag;
                    simdgroup_load(b_frag, &B_tile[k_sub][col_idx], TILE_N);
                    // half * half + float -> float accumulation
                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results - convert from float accumulator to half output
    // Per-simdgroup staging to avoid race conditions between 4 simdgroups
    threadgroup float out_staging[SIMDGROUPS_PER_TG][8][8];

    for (uint mi = 0; mi < 4; ++mi) {
        for (uint ni = 0; ni < 4; ++ni) {
            uint out_row = tgid.y * TILE_M + sg_row_base + mi * 8;
            uint out_col = tgid.x * TILE_N + sg_col_base + ni * 8;

            // Each simdgroup uses its own staging slot
            simdgroup_store(acc[mi][ni], &out_staging[simd_id][0][0], 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Convert float to half during write to output
            for (uint elem = simd_lane; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                if (out_row + r < M && out_col + c < N) {
                    C[(out_row + r) * N + out_col + c] = half(out_staging[simd_id][r][c]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}