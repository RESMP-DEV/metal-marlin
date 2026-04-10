import re

with open('./metal_marlin/shaders/mmfp4_gemm.metal', 'r') as f:
    code = f.read()

# Replace the inner multiplication loop
old_loop = """        #pragma unroll 4
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
        }"""

new_loop = """        #pragma unroll 4
        for (uint k_sub = 0; k_sub < TILE_K; k_sub += 8) {
            simdgroup_matrix<half, 8, 8> a_frag[4];
            simdgroup_matrix<half, 8, 8> b_frag[4];

            // Hoist A_tile loads: 4 loads per k_sub instead of 16
            for (uint mi = 0; mi < 4; ++mi) {
                uint row_idx = sg_row_base + mi * 8;
                // Only load if within valid M range, otherwise zero-initialize
                if (tile_start_row + row_idx < M) {
                    simdgroup_load(a_frag[mi], &A_tile[row_idx][k_sub], TILE_K);
                } else {
                    a_frag[mi] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
                }
            }

            // Hoist B_tile loads: 4 loads per k_sub instead of 16
            for (uint ni = 0; ni < 4; ++ni) {
                uint col_idx = sg_col_base + ni * 8;
                simdgroup_load(b_frag[ni], &B_tile[k_sub][col_idx], TILE_N);
            }

            // Compute 4x4 = 16 MMAs back-to-back without LDS or branches
            for (uint mi = 0; mi < 4; ++mi) {
                for (uint ni = 0; ni < 4; ++ni) {
                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
                }
            }
        }"""

if old_loop in code:
    code = code.replace(old_loop, new_loop)
    with open('./metal_marlin/shaders/mmfp4_gemm.metal', 'w') as f:
        f.write(code)
    print("Successfully replaced inner loop!")
else:
    print("Could not find old loop.")
