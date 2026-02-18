"""
Batched GEMM with variable sequence lengths support.

This module provides Metal-accelerated batched GEMM operations where each batch
element can have a different sequence length (M dimension). This is essential for:

- Transformer prefill batches where prompts have different lengths
- Continuous batching during inference
- Any scenario where sequence lengths vary across batch elements

Key improvements over standard batched GEMM:
    - Per-batch M dimensions (no need to pad to max length)
    - Per-batch stride values (support non-contiguous batch data)
    - Reduced memory usage (no padding overhead)
    - Better GPU utilization (avoids wasted computation on padding tokens)

Usage:
    >>> from metal_marlin.batched_gemm_variable_seq import dispatch_batched_gemm_variable_seq
    >>>
    >>> # Three sequences with different lengths: [64, 128, 256]
    >>> seq_lengths = torch.tensor([64, 128, 256], dtype=torch.int32, device="mps")
    >>> A = torch.randn(3, 256, 4096, dtype=torch.float16, device="mps")  # [batch, max_seq, K]
    >>> B = ...  # [batch, K, N] packed FP4
    >>> scales = ...  # [batch, K/8, N] scales
    >>>
    >>> # Compute per-batch strides
    >>> A_strides = torch.tensor([seq_len * K for seq_len in seq_lengths], dtype=torch.int32, device="mps")
    >>> C_strides = torch.tensor([seq_len * N for seq_len in seq_lengths], dtype=torch.int32, device="mps")
    >>>
    >>> # Dispatch with variable sequence lengths
    >>> C = dispatch_batched_gemm_variable_seq(
    >>>     lib, A, B, scales,
    >>>     N=4096, K=4096, batch_count=3,
    >>>     M_per_batch=seq_lengths,
    >>>     A_batch_strides=A_strides,
    >>>     B_batch_stride=K * N,
    >>>     C_batch_strides=C_strides,
    >>>     group_size=64
    >>> )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ._compat import HAS_MPS, Metal, require_metal, require_mps
from .metal_dispatch import (
    MetalKernelLibrary,
    dispatch_kernel,
    mps_tensor_to_metal_buffer,
)

if TYPE_CHECKING:
    pass


def dispatch_batched_gemm_variable_seq(
    lib: MetalKernelLibrary,
    A: torch.Tensor,
    B: torch.Tensor,
    scales: torch.Tensor,
    N: int,
    K: int,
    batch_count: int,
    M_per_batch: torch.Tensor,
    A_batch_strides: torch.Tensor,
    B_batch_stride: int,
    C_batch_strides: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dispatch batched GEMM with variable sequence lengths.

    Computes C[b] = A[b] @ dequant(B[b]) for each batch element b,
    where each batch can have a different M dimension (sequence length).

    Args:
        lib: MetalKernelLibrary instance with compiled kernel.
        A: Input activations [batch, max_M, K] or packed with strides.
           dtype: float16 (half), device: mps.
        B: Quantized weights [batch * (K * N / 8)] packed FP4.
           dtype: uint32, device: mps.
        scales: Per-group scales [batch, K/group_size, N].
           dtype: float16, device: mps.
        N: Output dimension (shared across batches).
        K: Shared dimension (must match across all batches).
        batch_count: Number of batch elements.
        M_per_batch: Sequence length for each batch [batch_count].
           dtype: int32, device: mps.
        A_batch_strides: Stride (in elements) for each A[b] [batch_count].
           dtype: int32, device: mps.
        B_batch_stride: Stride (in elements) for B[b].
        C_batch_strides: Stride (in elements) for each C[b] [batch_count].
           dtype: int32, device: mps.
        group_size: Scale sharing group size (typically 64 or 128).

    Returns:
        C: Output activations [batch, max_M, N] or packed with strides.
           dtype: float16, device: mps.

    Note:
        - The kernel internally uses per-batch M values for boundary checks.
        - Strides allow non-contiguous batch data (e.g., from KV cache).
        - Output C is pre-allocated by caller to match batch layout.

    Example:
        >>> # Continuous batching scenario
        >>> seq_lengths = torch.tensor([64, 128, 256], dtype=torch.int32, device="mps")
        >>> A_strides = seq_lengths * K  # [64K, 128K, 256K]
        >>> C_strides = seq_lengths * N  # [64N, 128N, 256N]
        >>>
        >>> C = dispatch_batched_gemm_variable_seq(
        >>>     lib, A, B, scales,
        >>>     N=4096, K=4096, batch_count=3,
        >>>     M_per_batch=seq_lengths,
        >>>     A_batch_strides=A_strides,
        >>>     B_batch_stride=K * N,
        >>>     C_batch_strides=C_strides,
        >>>     group_size=64
        >>> )
    """
    require_mps()
    require_metal()

    if not HAS_MPS or Metal is None:
        raise RuntimeError("MPS device and Metal framework not available")

    device = lib.device

    # Validate inputs
    if not A.is_mps or not B.is_mps or not scales.is_mps:
        raise ValueError("All inputs must be on MPS device")

    if A.dtype != torch.float16 or scales.dtype != torch.float16:
        raise ValueError("A and scales must be float16")

    if B.dtype != torch.uint32:
        raise ValueError("B must be uint32 (packed FP4)")

    if M_per_batch.dtype != torch.int32 or A_batch_strides.dtype != torch.int32 or C_batch_strides.dtype != torch.int32:
        raise ValueError("M_per_batch, A_batch_strides, C_batch_strides must be int32")

    if M_per_batch.dim() != 1 or M_per_batch.shape[0] != batch_count:
        raise ValueError(f"M_per_batch must be [batch_count], got {M_per_batch.shape}")

    if A_batch_strides.dim() != 1 or A_batch_strides.shape[0] != batch_count:
        raise ValueError(f"A_batch_strides must be [batch_count], got {A_batch_strides.shape}")

    if C_batch_strides.dim() != 1 or C_batch_strides.shape[0] != batch_count:
        raise ValueError(f"C_batch_strides must be [batch_count], got {C_batch_strides.shape}")

    # Ensure contiguous tensors
    A = A.contiguous()
    B = B.contiguous()
    scales = scales.contiguous()
    M_per_batch = M_per_batch.contiguous()
    A_batch_strides = A_batch_strides.contiguous()
    C_batch_strides = C_batch_strides.contiguous()

    # Allocate output if not provided
    max_seq_len = M_per_batch.max().item()
    C = torch.zeros(
        batch_count * max_seq_len * N,
        dtype=torch.float16,
        device="mps"
    )

    # Convert to Metal buffers
    A_buf = mps_tensor_to_metal_buffer(A, device)
    B_buf = mps_tensor_to_metal_buffer(B, device)
    scales_buf = mps_tensor_to_metal_buffer(scales, device)
    C_buf = mps_tensor_to_metal_buffer(C, device, copy_back=True)
    M_per_batch_buf = mps_tensor_to_metal_buffer(M_per_batch, device)
    A_batch_strides_buf = mps_tensor_to_metal_buffer(A_batch_strides, device)
    C_batch_strides_buf = mps_tensor_to_metal_buffer(C_batch_strides, device)

    # Constant buffers
    import numpy as np

    n_buf = device.newBufferWithBytes_length_options_(
        np.array([N], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    k_buf = device.newBufferWithBytes_length_options_(
        np.array([K], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    batch_count_buf = device.newBufferWithBytes_length_options_(
        np.array([batch_count], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    b_batch_stride_buf = device.newBufferWithBytes_length_options_(
        np.array([B_batch_stride], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    group_size_buf = device.newBufferWithBytes_length_options_(
        np.array([group_size], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )

    # Compute grid dimensions
    # Grid: (n_tiles, m_tiles, batch_count)
    n_tiles = (N + TILE_N - 1) // TILE_N
    m_tiles = (max_seq_len + TILE_M - 1) // TILE_M

    # Dispatch kernel
    dispatch_kernel(
        lib,
        function_name="marlin_gemm_batched_variable_seq",
        grid=(n_tiles, m_tiles, batch_count),
        threadgroup=(128, 1, 1),  # 4 simdgroups of 32 threads each
        buffers=[
            A_buf,
            B_buf,
            scales_buf,
            C_buf,
            M_per_batch_buf,
            n_buf,
            k_buf,
            batch_count_buf,
            A_batch_strides_buf,
            b_batch_stride_buf,
            C_batch_strides_buf,
            group_size_buf,
        ],
        wait=True,
    )

    return C


def dispatch_grouped_attention_variable_seq(
    lib: MetalKernelLibrary,
    Q: torch.Tensor,
    K: torch.Tensor,
    scores: torch.Tensor,
    seq_lengths: torch.Tensor,
    max_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    batch_count: int,
) -> torch.Tensor:
    """Dispatch grouped attention GEMM with variable sequence lengths.

    Computes Q @ K^T for GQA (grouped-query attention) where each batch
    element can have a different sequence length.

    Args:
        lib: MetalKernelLibrary instance with compiled kernel.
        Q: Query projections [batch, num_q_heads, max_seq, head_dim].
           dtype: float16, device: mps.
        K: Key projections [batch, num_kv_heads, max_seq, head_dim].
           dtype: float16, device: mps.
        scores: Output scores tensor [batch, num_q_heads, max_seq, max_seq].
           dtype: float16, device: mps (pre-allocated).
        seq_lengths: Sequence length for each batch [batch_count].
           dtype: int32, device: mps.
        max_seq_len: Maximum sequence length (for tensor allocation).
        num_q_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (for GQA).
        head_dim: Head dimension.
        batch_count: Number of batch elements.

    Returns:
        scores: Attention scores [batch, num_q_heads, max_seq, max_seq].
           dtype: float16, device: mps.

    Note:
        - Supports GQA where num_q_heads > num_kv_heads.
        - Each batch uses its own seq_len for boundary checks.
        - Output is packed with max_seq_len for all batches.

    Example:
        >>> # Batch with variable prompt lengths
        >>> seq_lengths = torch.tensor([64, 128, 256], dtype=torch.int32, device="mps")
        >>> max_seq = 256
        >>>
        >>> scores = dispatch_grouped_attention_variable_seq(
        >>>     lib, Q, K, scores, seq_lengths,
        >>>     max_seq_len=max_seq,
        >>>     num_q_heads=32, num_kv_heads=8,
        >>>     head_dim=128, batch_count=3
        >>> )
    """
    require_mps()
    require_metal()

    if not HAS_MPS or Metal is None:
        raise RuntimeError("MPS device and Metal framework not available")

    device = lib.device

    # Validate inputs
    if not Q.is_mps or not K.is_mps or not scores.is_mps:
        raise ValueError("All inputs must be on MPS device")

    if Q.dtype != torch.float16 or K.dtype != torch.float16 or scores.dtype != torch.float16:
        raise ValueError("Q, K, and scores must be float16")

    if seq_lengths.dtype != torch.int32:
        raise ValueError("seq_lengths must be int32")

    if seq_lengths.dim() != 1 or seq_lengths.shape[0] != batch_count:
        raise ValueError(f"seq_lengths must be [batch_count], got {seq_lengths.shape}")

    # Ensure contiguous tensors
    Q = Q.contiguous()
    K = K.contiguous()
    seq_lengths = seq_lengths.contiguous()

    # Convert to Metal buffers
    Q_buf = mps_tensor_to_metal_buffer(Q, device)
    K_buf = mps_tensor_to_metal_buffer(K, device)
    scores_buf = mps_tensor_to_metal_buffer(scores, device, copy_back=True)
    seq_lengths_buf = mps_tensor_to_metal_buffer(seq_lengths, device)

    # Constant buffers
    import numpy as np

    max_seq_len_buf = device.newBufferWithBytes_length_options_(
        np.array([max_seq_len], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    num_q_heads_buf = device.newBufferWithBytes_length_options_(
        np.array([num_q_heads], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    num_kv_heads_buf = device.newBufferWithBytes_length_options_(
        np.array([num_kv_heads], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    head_dim_buf = device.newBufferWithBytes_length_options_(
        np.array([head_dim], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )

    # Compute grid dimensions
    seq_tiles = (max_seq_len + TILE_N - 1) // TILE_N

    # Dispatch kernel
    dispatch_kernel(
        lib,
        function_name="marlin_gemm_grouped_attention_variable_seq",
        grid=(seq_tiles, seq_tiles, batch_count * num_q_heads),
        threadgroup=(128, 1, 1),
        buffers=[
            Q_buf,
            K_buf,
            scores_buf,
            seq_lengths_buf,
            max_seq_len_buf,
            num_q_heads_buf,
            num_kv_heads_buf,
            head_dim_buf,
        ],
        wait=True,
    )

    return scores


# Tile dimensions (must match batched_gemm_variable_seq.metal)
TILE_M = 64
TILE_N = 32
TILE_K = 48
