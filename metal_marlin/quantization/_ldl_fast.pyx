# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""Fast block LDL decomposition using Accelerate LAPACK."""

import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef extern from "Accelerate/Accelerate.h" nogil:
    void dpotrf_(char* uplo, int* n, double* a, int* lda, int* info)
    void dpotri_(char* uplo, int* n, double* a, int* lda, int* info)


cpdef tuple block_ldl_fast(cnp.ndarray[cnp.float64_t, ndim=2] H, int block_size=16):
    """Fast block LDL decomposition.
    
    Args:
        H: Symmetric PSD matrix [N, N]
        block_size: Block size (default 16)
    
    Returns:
        (L, D) where H = L @ D @ L.T
    """
    cdef int n = H.shape[0]
    cdef int m = n // block_size
    cdef int info = 0
    cdef char uplo = b'L'
    cdef int i, j, k, kk, b_start, b_end
    
    # Cholesky: H = C @ C.T
    cdef cnp.ndarray[cnp.float64_t, ndim=2] C = H.copy()
    dpotrf_(&uplo, &n, &C[0, 0], &n, &info)
    if info != 0:
        raise ValueError(f"Cholesky failed: {info}")
    
    # Zero upper triangle
    for i in range(n):
        for j in range(i + 1, n):
            C[i, j] = 0.0
    
    cdef cnp.ndarray[cnp.float64_t, ndim=2] L = C.copy()
    cdef cnp.ndarray[cnp.float64_t, ndim=2] D = np.zeros((n, n), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] diag_block = np.zeros((block_size, block_size), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] diag_inv = np.zeros((block_size, block_size), dtype=np.float64)
    
    for k in range(m):
        b_start = k * block_size
        b_end = (k + 1) * block_size
        
        for i in range(block_size):
            for j in range(block_size):
                diag_block[i, j] = C[b_start + i, b_start + j]
        
        for i in range(block_size):
            for j in range(block_size):
                D[b_start + i, b_start + j] = 0.0
                for kk in range(block_size):
                    D[b_start + i, b_start + j] += diag_block[i, kk] * diag_block[j, kk]
        
        diag_inv[:, :] = diag_block[:, :]
        dpotri_(&uplo, &block_size, &diag_inv[0, 0], &block_size, &info)
        if info != 0:
            diag_inv = np.linalg.inv(diag_block)
        
        for i in range(n):
            for j in range(block_size):
                L[i, b_start + j] = 0.0
                for kk in range(block_size):
                    L[i, b_start + j] += C[i, b_start + kk] * diag_inv[kk, j]
    
    return L, D
