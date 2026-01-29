"""EXL3-style Hadamard preprocessing for Hessian matrices.

Implements the Hadamard rotation scheme from ExllamaV3 for improved
quantization quality through channel decorrelation.
"""

import numpy as np
from numpy.typing import NDArray

from metal_marlin.hadamard import hadamard_matrix


def preprocess_hessian_exl3(
    H: NDArray[np.float64],
    had_k: int = 128,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Apply EXL3-style Hadamard rotation to Hessian.

    From ExllamaV3 finalize_capture_H:
    1. Random sign flips (su) for input channels
    2. Block Hadamard rotation (128-dim blocks)
    3. Returns rotated H and sign vectors for later use

    The transformation is:
        H_rotated = Had @ diag(su) @ H @ diag(su) @ Had.T

    This decorrelates the Hessian channels, making quantization
    more effective by spreading error uniformly.

    Args:
        H: Raw Hessian [K, K]
        had_k: Hadamard block size (128 for EXL3)

    Returns:
        H_rotated: Rotated Hessian [K, K]
        su: Random sign flips [K] for input reconstruction
        Had: Normalized Hadamard matrix [had_k, had_k] (cached for weight rotation)

    Example:
        >>> H = np.random.randn(256, 256).astype(np.float64)
        >>> H = H @ H.T  # Make positive semi-definite
        >>> H_rot, su, Had = preprocess_hessian_exl3(H, had_k=128)
        >>> H_rot.shape
        (256, 256)
        >>> su.shape
        (256,)
        >>> Had.shape
        (128, 128)
    """
    k = H.shape[0]

    # Random sign flips - adds randomness to break symmetry
    # Small epsilon ensures deterministic sign for near-zero values in testing
    rng = np.random.default_rng()
    su = np.sign(rng.standard_normal(k) + 1e-5)

    # Apply to H: H_rot = Had @ diag(su) @ H @ diag(su) @ Had.T
    # Right multiply by diag(su)
    H = H * su[:, None]
    # Right Hadamard
    H = blockwise_hadamard(H, had_k, axis=1)
    # Left multiply by diag(su)
    H = H * su[None, :]
    # Left Hadamard
    H_rotated = blockwise_hadamard(H, had_k, axis=0)

    # Get normalized Hadamard matrix for weight rotation
    Had = hadamard_matrix(had_k)

    return H_rotated, su, Had


def blockwise_hadamard(
    X: NDArray[np.float64],
    block_size: int,
    axis: int,
) -> NDArray[np.float64]:
    """Apply Hadamard transform in blocks along specified axis.

    For EXL3, applies block-diagonal Hadamard transformation where
    each block of size `block_size` is transformed independently.

    Args:
        X: Input array [..., N, ...] where N is the size along `axis`
        block_size: Size of Hadamard blocks (must divide N evenly)
        axis: Axis along which to apply transform (0 or 1)

    Returns:
        Transformed array with same shape as input

    Raises:
        ValueError: If block_size doesn't divide the axis size evenly
        or if block_size is not a power of 2

    Example:
        >>> X = np.random.randn(256, 256).astype(np.float64)
        >>> Y = blockwise_hadamard(X, block_size=128, axis=0)
        >>> Y.shape
        (256, 256)
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    n = X.shape[axis]

    if n % block_size != 0:
        raise ValueError(f"block_size ({block_size}) must divide axis size ({n}) evenly")

    # Get normalized Hadamard matrix (hadamard_matrix returns normalized)
    Had = hadamard_matrix(block_size)

    num_blocks = n // block_size

    if axis == 0:
        # Process along rows: reshape to [num_blocks, block_size, ...]
        # X shape: [n, m] -> [num_blocks, block_size, m]
        other_axis = 1
        m = X.shape[other_axis]

        # Reshape: [n, m] -> [num_blocks, block_size, m]
        X_reshaped = X.reshape(num_blocks, block_size, m)

        # Apply Hadamard: Had @ X_block for each block
        # Had: [block_size, block_size], X_reshaped: [num_blocks, block_size, m]
        # Result: [num_blocks, block_size, m]
        Y_blocks = Had @ X_reshaped

        # Reshape back: [num_blocks, block_size, m] -> [n, m]
        Y = Y_blocks.reshape(n, m)
    else:
        # Process along columns: reshape to [..., num_blocks, block_size]
        # X shape: [m, n] -> [m, num_blocks, block_size]
        other_axis = 0
        m = X.shape[other_axis]

        # Reshape: [m, n] -> [m, num_blocks, block_size]
        X_reshaped = X.reshape(m, num_blocks, block_size)

        # Apply Hadamard: X_block @ Had.T for each block
        # X_reshaped: [m, num_blocks, block_size], Had.T: [block_size, block_size]
        # Result: [m, num_blocks, block_size]
        Y_blocks = X_reshaped @ Had.T

        # Reshape back: [m, num_blocks, block_size] -> [m, n]
        Y = Y_blocks.reshape(m, n)

    return Y


def rotate_weights_exl3(
    W: NDArray[np.float32],
    su: NDArray[np.float64],
    sv: NDArray[np.float64] | None = None,
    had_k: int = 128,
    had_n: int = 128,
) -> NDArray[np.float32]:
    """Apply Hadamard rotation to weights before quantization.

    Must use same su (sign flips) as Hessian preprocessing.

    From EXL3, the rotation applies sign flips and Hadamard transforms
    along each dimension to disperse outliers and improve quantization.

    For full rotation (with sv):
        1. Apply su sign flips to in_features (axis=1)
        2. Apply Hadamard along in_features (axis=1)
        3. Apply sv sign flips to out_features (axis=0)
        4. Apply Hadamard along out_features (axis=0)

    For partial rotation (sv=None, typical for quantization input):
        1. Apply su sign flips to in_features (axis=1)
        2. Apply Hadamard along in_features (axis=1)

    Args:
        W: Weight matrix [out_features, in_features]
        su: Sign flips from preprocess_hessian_exl3 (length in_features)
        sv: Sign flips for output channels (length out_features), or None
        had_k: Hadamard block size for input dimension
        had_n: Hadamard block size for output dimension

    Returns:
        W_rotated: Rotated weights ready for trellis quantization
    """
    W = W.astype(np.float64)

    # Step 1: Apply sign flips for input channels (su)
    # su is applied to in_features (axis=1), so multiply column-wise
    W = W * su[None, :]

    # Step 2: Apply Hadamard along in_features (axis=1) - "right" Hadamard
    W = blockwise_hadamard(W, had_k, axis=1)

    # Step 3 & 4: Apply output dimension rotation if sv provided
    if sv is not None:
        # Apply sign flips for output channels (sv)
        W = W * sv[:, None]

        # Apply Hadamard along out_features (axis=0) - "left" Hadamard
        W = blockwise_hadamard(W, had_n, axis=0)

    return W.astype(np.float32)


def unrotate_weights_exl3(
    W_q: NDArray[np.float32],
    su: NDArray[np.float64],
    sv: NDArray[np.float64],
    had_k: int = 128,
    had_n: int = 128,
) -> NDArray[np.float32]:
    """Reconstruct original weight space from rotated quantized weights.

    Used for quality validation only (inference uses fused rotation).

    The inverse of the EXL3 rotation applies Hadamard transforms in reverse
    order (they are self-inverse) and sign flips twice (su*su=1 reverses).

    Inverse transformation:
        1. Apply Hadamard along out_features (axis=0) - reverses sv Hadamard
        2. Apply sv sign flips (sv * sv = 1 reverses)
        3. Apply Hadamard along in_features (axis=1) - reverses su Hadamard
        4. Apply su sign flips (su * su = 1 reverses)

    Args:
        W_q: Quantized rotated weight matrix [out_features, in_features]
        su: Sign flips for input channels (in_features)
        sv: Sign flips for output channels (out_features)
        had_k: Hadamard block size for input dimension
        had_n: Hadamard block size for output dimension

    Returns:
        W_unrotated: Reconstructed weights in original space
    """
    W = W_q.astype(np.float64)

    # Step 1: Apply Hadamard along out_features (axis=0) - inverse of sv rotation
    # This reverses the final Hadamard (H @ H = I for normalized H)
    W = blockwise_hadamard(W, had_n, axis=0)

    # Step 2: Apply sign flips for output channels (sv * sv = 1, reverses)
    W = W * sv[:, None]

    # Step 3: Apply Hadamard along in_features (axis=1) - inverse of su rotation
    W = blockwise_hadamard(W, had_k, axis=1)

    # Step 4: Apply sign flips for input channels (su * su = 1, reverses)
    W = W * su[None, :]

    return W.astype(np.float32)
