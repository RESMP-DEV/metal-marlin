# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Fast eigendecomposition using Accelerate LAPACK dsyevd."""

import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef extern from "Accelerate/Accelerate.h" nogil:
    void dsyevd_(char* jobz, char* uplo, int* n, double* a, int* lda,
                 double* w, double* work, int* lwork, int* iwork, int* liwork,
                 int* info)


cpdef tuple eigh_psd_fast(cnp.ndarray[cnp.float64_t, ndim=2] H, double sigma_reg=0.01):
    """Fast eigendecomposition with PSD enforcement.
    
    Returns H_psd = V @ diag(max(eigenvalues, sigma_reg)) @ V.T
    """
    cdef int n = H.shape[0]
    cdef int info = 0
    cdef char jobz = b'V'
    cdef char uplo = b'L'
    cdef int lwork = -1
    cdef int liwork = -1
    cdef double work_query
    cdef int iwork_query
    cdef int i, j
    cdef double scale
    
    cdef cnp.ndarray[cnp.float64_t, ndim=2] A = H.copy()
    cdef cnp.ndarray[cnp.float64_t, ndim=1] eigenvalues = np.zeros(n, dtype=np.float64)
    
    # Query workspace
    dsyevd_(&jobz, &uplo, &n, &A[0, 0], &n, &eigenvalues[0],
            &work_query, &lwork, &iwork_query, &liwork, &info)
    
    lwork = int(work_query)
    liwork = iwork_query
    
    cdef cnp.ndarray[cnp.float64_t, ndim=1] work = np.zeros(lwork, dtype=np.float64)
    cdef int[::1] iwork = np.zeros(liwork, dtype=np.intc)
    
    # Compute eigendecomposition
    dsyevd_(&jobz, &uplo, &n, &A[0, 0], &n, &eigenvalues[0],
            &work[0], &lwork, &iwork[0], &liwork, &info)
    
    if info != 0:
        raise ValueError(f"dsyevd failed: {info}")
    
    # Clamp eigenvalues
    for i in range(n):
        if eigenvalues[i] < sigma_reg:
            eigenvalues[i] = sigma_reg
    
    # Reconstruct H_psd
    cdef cnp.ndarray[cnp.float64_t, ndim=2] scaled = A.copy()
    for j in range(n):
        scale = eigenvalues[j] ** 0.5
        for i in range(n):
            scaled[i, j] *= scale
    
    cdef cnp.ndarray[cnp.float64_t, ndim=2] H_psd = scaled @ scaled.T
    
    return H_psd, eigenvalues
