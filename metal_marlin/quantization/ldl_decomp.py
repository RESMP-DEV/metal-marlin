"""Block LDL decomposition for GPTQ quantization.

This module implements block LDL decomposition which is faster and more numerically
stable than Cholesky decomposition for GPTQ quantization. The block processing
aligns with GPU tensor core tile sizes (16x16) for efficient execution.

The decomposition computes L and D such that H = L @ D @ L.T, where:
- L is lower triangular with unit diagonal blocks
- D is block diagonal (stored as full matrix for convenience)

Reference:
    - ExllamaV3: https://github.com/turboderp-org/exllamav3
    - LDL decomposition: https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def block_ldl(
    H: NDArray[np.float64],
    block_size: int = 16,
    verbose: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Block LDL decomposition of Hessian.
    
    Returns (L, D) where H = L @ D @ L.T
    
    Block processing is GPU-friendly and numerically stable.
    Uses 16x16 blocks to match trellis tile size.
    
    The algorithm:
    1. Perform Cholesky decomposition: H = C @ C.T
    2. Extract diagonal blocks from C
    3. Normalize to get L with identity diagonal blocks
    4. Reconstruct D from the diagonal blocks
    
    Args:
        H: Symmetric positive definite Hessian [K, K]
        block_size: Block size for decomposition (16 matches tiles)
        verbose: Whether to print progress information
        
    Returns:
        L: Lower triangular factor with unit diagonal blocks [K, K]
        D: Diagonal factor (stored as full matrix for convenience) [K, K]
        
    Raises:
        ValueError: If H is not square or block_size doesn't divide dimensions
        
    Example:
        >>> import numpy as np
        >>> H = np.array([[4.0, 2.0], [2.0, 5.0]])
        >>> L, D = block_ldl(H, block_size=1)
        >>> np.allclose(H, L @ D @ L.T)
        True
    """
    H = np.asarray(H, dtype=np.float64)
    n = H.shape[0]

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be square matrix, got shape {H.shape}")

    if n % block_size != 0:
        raise ValueError(f"Matrix dimension {n} must be divisible by block_size {block_size}")

    m = n // block_size  # Number of blocks along each dimension

    if verbose:
        print(f"Block LDL: {n}x{n} matrix with {block_size}x{block_size} blocks ({m}x{m} grid)")

    # Step 1: Cholesky decomposition H = C @ C.T
    # C is lower triangular
    try:
        C = np.linalg.cholesky(H)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Hessian is not positive definite: {e}")

    # Step 2: Extract diagonal blocks from C
    # C has shape [n, n]. Reshape to [m, block_size, m, block_size] to access blocks
    C_blocks = C.reshape(m, block_size, m, block_size)

    # Diagonal blocks are at positions [i, :, i, :] for i in range(m)
    # Shape: [m, block_size, block_size]
    diag_blocks = np.zeros((m, block_size, block_size), dtype=np.float64)
    for i in range(m):
        diag_blocks[i] = C_blocks[i, :, i, :].copy()

    # Step 3: Compute inverse of each diagonal block
    # These are small blocks (16x16), so direct inversion is efficient
    diag_blocks_inv = np.zeros_like(diag_blocks)
    for i in range(m):
        try:
            diag_blocks_inv[i] = np.linalg.inv(diag_blocks[i])
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Diagonal block {i} is singular: {e}")

    # Step 4: Construct L by normalizing C
    # L = C with each column block multiplied by the inverse of its diagonal block
    L = C.copy()
    L_reshaped = L.reshape(n, m, block_size)  # [n, m, block_size]

    for i in range(m):
        # Column block i spans rows [:, i*block_size:(i+1)*block_size]
        # Multiply by inverse of diagonal block i
        col_start = i * block_size
        col_end = (i + 1) * block_size
        L[:, col_start:col_end] = L[:, col_start:col_end] @ diag_blocks_inv[i]

    # Step 5: Construct D
    # D is block diagonal with D_ii = diag_blocks[i] @ diag_blocks[i].T
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(m):
        row_start = i * block_size
        row_end = (i + 1) * block_size
        D[row_start:row_end, row_start:row_end] = diag_blocks[i] @ diag_blocks[i].T

    if verbose:
        # Verify reconstruction
        H_reconstructed = L @ D @ L.T
        error = np.linalg.norm(H - H_reconstructed) / np.linalg.norm(H)
        print(f"Block LDL reconstruction error: {error:.2e}")

    return L, D


def ldl_solve(
    L: NDArray[np.float64],
    D: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve L @ D @ L.T @ x = b using LDL factors.
    
    Uses forward/backward substitution optimized for the block structure.
    The system is solved in three steps:
    1. Solve L @ y = b for y (forward substitution)
    2. Solve D @ z = y for z (diagonal solve)
    3. Solve L.T @ x = z for x (backward substitution)
    
    Args:
        L: Lower triangular factor with unit diagonal [N, N]
        D: Block diagonal factor [N, N]
        b: Right-hand side [N] or [N, M]
        
    Returns:
        x: Solution vector/matrix with same shape as b
        
    Example:
        >>> import numpy as np
        >>> H = np.array([[4.0, 2.0], [2.0, 5.0]])
        >>> L, D = block_ldl(H, block_size=1)
        >>> b = np.array([1.0, 2.0])
        >>> x = ldl_solve(L, D, b)
        >>> np.allclose(H @ x, b)
        True
    """
    L = np.asarray(L, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    n = L.shape[0]

    if L.shape != (n, n):
        raise ValueError(f"L must be square, got shape {L.shape}")
    if D.shape != (n, n):
        raise ValueError(f"D must have shape ({n}, {n}), got {D.shape}")

    # Handle both vector and matrix right-hand sides
    b_shape = b.shape
    if b.ndim == 1:
        if b.shape[0] != n:
            raise ValueError(f"b must have length {n}, got {b.shape[0]}")
        b = b.reshape(-1, 1)
    elif b.ndim == 2:
        if b.shape[0] != n:
            raise ValueError(f"b must have first dimension {n}, got {b.shape[0]}")
    else:
        raise ValueError(f"b must be 1D or 2D, got {b.ndim}D")

    # Step 1: Forward substitution - solve L @ y = b
    y = np.zeros_like(b)
    for i in range(n):
        # y[i] = b[i] - sum_{j < i} L[i, j] * y[j]
        if i == 0:
            y[i] = b[i]
        else:
            y[i] = b[i] - L[i, :i] @ y[:i]

    # Step 2: Solve D @ z = y
    # D is block diagonal, so we solve block by block
    z = np.zeros_like(y)
    block_size = _detect_block_size(D)

    if block_size > 1:
        # Block diagonal solve
        for i in range(0, n, block_size):
            block_end = min(i + block_size, n)
            D_block = D[i:block_end, i:block_end]
            z[i:block_end] = np.linalg.solve(D_block, y[i:block_end])
    else:
        # True diagonal (1x1 blocks) - element-wise division
        diag = np.diag(D)
        z = y / diag.reshape(-1, 1)

    # Step 3: Backward substitution - solve L.T @ x = z
    x = np.zeros_like(z)
    for i in range(n - 1, -1, -1):
        # x[i] = z[i] - sum_{j > i} L[j, i] * x[j]
        if i == n - 1:
            x[i] = z[i]
        else:
            x[i] = z[i] - L[i + 1:, i] @ x[i + 1:]

    # Reshape back to original shape if input was 1D
    if len(b_shape) == 1:
        x = x.reshape(-1)

    return x


def _detect_block_size(D: NDArray[np.float64]) -> int:
    """Detect the block size from a block diagonal matrix.
    
    Args:
        D: Block diagonal matrix
        
    Returns:
        Detected block size (1 if truly diagonal)
    """
    n = D.shape[0]

    # Check for 1x1 blocks (true diagonal)
    off_diag_norm = np.linalg.norm(D - np.diag(np.diag(D)))
    if off_diag_norm < 1e-10:
        return 1

    # Try common block sizes
    for block_size in [16, 8, 4, 2]:
        if n % block_size == 0:
            # Check if off-block-diagonal elements are negligible
            is_block_diag = True
            for i in range(0, n, block_size):
                for j in range(0, n, block_size):
                    if i != j:
                        block = D[i:i+block_size, j:j+block_size]
                        if np.linalg.norm(block) > 1e-10:
                            is_block_diag = False
                            break
                if not is_block_diag:
                    break
            if is_block_diag:
                return block_size

    return 1  # Fallback


def ldl_inverse(
    L: NDArray[np.float64],
    D: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute H^{-1} from LDL factors where H = L @ D @ L.T.
    
    Uses the formula: H^{-1} = L^{-T} @ D^{-1} @ L^{-1}
    
    Args:
        L: Lower triangular factor with unit diagonal [N, N]
        D: Block diagonal factor [N, N]
        
    Returns:
        H_inv: Inverse of H [N, N]
        
    Example:
        >>> import numpy as np
        >>> H = np.array([[4.0, 2.0], [2.0, 5.0]])
        >>> L, D = block_ldl(H, block_size=1)
        >>> H_inv = ldl_inverse(L, D)
        >>> np.allclose(H_inv, np.linalg.inv(H))
        True
    """
    L = np.asarray(L, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    n = L.shape[0]

    # Compute L^{-1} (forward substitution for identity columns)
    L_inv = np.eye(n, dtype=np.float64)
    for j in range(n):
        for i in range(j + 1, n):
            L_inv[i, j] = -L[i, :i] @ L_inv[:i, j]

    # Compute D^{-1} (invert diagonal blocks)
    D_inv = np.zeros_like(D)
    block_size = _detect_block_size(D)

    for i in range(0, n, block_size):
        block_end = min(i + block_size, n)
        D_block = D[i:block_end, i:block_end]
        D_inv[i:block_end, i:block_end] = np.linalg.inv(D_block)

    # H^{-1} = L^{-T} @ D^{-1} @ L^{-1}
    H_inv = L_inv.T @ D_inv @ L_inv

    return H_inv


def chol2ldl(
    C: NDArray[np.float64],
    block_size: int = 16,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert Cholesky factor to LDL factors.
    
    Given C from H = C @ C.T (Cholesky), compute L and D such that H = L @ D @ L.T.
    
    This is useful when you already have a Cholesky decomposition and want to
    convert it to the block LDL form for use with error compensation algorithms.
    
    Args:
        C: Lower triangular Cholesky factor [N, N]
        block_size: Block size for decomposition
        
    Returns:
        L: Lower triangular factor with unit diagonal blocks [N, N]
        D: Block diagonal factor [N, N]
    """
    C = np.asarray(C, dtype=np.float64)
    n = C.shape[0]

    if n % block_size != 0:
        raise ValueError(f"Matrix dimension {n} must be divisible by block_size {block_size}")

    m = n // block_size

    # Extract diagonal blocks
    C_blocks = C.reshape(m, block_size, m, block_size)
    diag_blocks = np.zeros((m, block_size, block_size), dtype=np.float64)
    for i in range(m):
        diag_blocks[i] = C_blocks[i, :, i, :].copy()

    # Compute inverse of diagonal blocks
    diag_blocks_inv = np.zeros_like(diag_blocks)
    for i in range(m):
        diag_blocks_inv[i] = np.linalg.inv(diag_blocks[i])

    # Construct L by normalizing C
    L = C.copy()
    for i in range(m):
        col_start = i * block_size
        col_end = (i + 1) * block_size
        L[:, col_start:col_end] = L[:, col_start:col_end] @ diag_blocks_inv[i]

    # Construct D
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(m):
        row_start = i * block_size
        row_end = (i + 1) * block_size
        D[row_start:row_end, row_start:row_end] = diag_blocks[i] @ diag_blocks[i].T

    return L, D
