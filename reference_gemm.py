import numpy as np


def reference_tiled_gemm(A, B, TILE_M=64, TILE_N=64, TILE_K=32):
    """Simulate the Metal kernel's computation step by step."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = np.zeros((M, N), dtype=np.float16)

    for tg_row in range(0, M, TILE_M):
        for tg_col in range(0, N, TILE_N):
            # Simulate simdgroup layout
            for simd_id in range(4):
                sg_row = (simd_id // 2) * 16
                sg_col = (simd_id % 2) * 32

                # acc[2][4] for each simdgroup
                acc = np.zeros((2, 4, 8, 8), dtype=np.float32)

                for k_tile in range(0, K, TILE_K):
                    for mi in range(2):  # SG_M_TILES
                        for ni in range(4):  # SG_N_TILES
                            # 8x8 tile computation
                            row_start = tg_row + sg_row + mi * 8
                            col_start = tg_col + sg_col + ni * 8

                            # Simulate simdgroup MMA
                            for kt in range(4):  # K_TILES
                                k_base = k_tile + kt * 8
                                # matmul 8x8 @ 8x8
                                a_tile = A[row_start:row_start + 8, k_base:k_base + 8]
                                b_tile = B[k_base:k_base + 8, col_start:col_start + 8]
                                acc[mi, ni] += a_tile @ b_tile

                # Store acc to C
                for mi in range(2):
                    for ni in range(4):
                        row_start = tg_row + sg_row + mi * 8
                        col_start = tg_col + sg_col + ni * 8
                        C[row_start:row_start + 8, col_start:col_start + 8] = acc[mi, ni]

    return C
