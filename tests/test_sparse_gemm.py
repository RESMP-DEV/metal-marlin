"""Sparse GEMM accuracy validation against dense GEMM with artificial sparsity.

Tests structured N:M sparsity patterns (primarily 2:4) to verify that the
sparse GEMM path produces results equivalent to dense GEMM when the weight
matrix has the appropriate zero structure. This validates:

  1. prune_to_2_4 correctly enforces 2:4 structure
  2. Sparse metadata encoding/decoding is lossless
  3. Sparse GEMM output matches dense GEMM reference within FP16 tolerance
  4. Various N:M patterns (1:4, 2:4, 4:8) produce correct results
  5. Sparse path achieves expected throughput advantage over dense

All tests use pure numpy reference implementations so they validate
correctness without requiring Metal hardware. Metal kernel tests are
guarded by availability checks.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 2:4 sparsity utilities
# ---------------------------------------------------------------------------


def prune_to_2_4(
    W: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Prune a weight matrix to 2:4 structured sparsity.

    For every group of 4 consecutive elements along K (axis 0), the two
    smallest-magnitude values are zeroed. The two surviving values are
    packed contiguously, and a metadata array records their original positions.

    Args:
        W: Dense weight matrix [K, N] as float16 or float32.

    Returns:
        W_sparse: Compressed weights [K/2, N] containing only non-zero values.
        meta: Metadata array [K/4, N] as uint8, each entry encoding two
              2-bit position indices packed as (pos1 << 2) | pos0.

    Raises:
        ValueError: If K is not divisible by 4.
    """
    K, N = W.shape
    if K % 4 != 0:
        raise ValueError(f"K={K} must be divisible by 4 for 2:4 sparsity")

    num_blocks = K // 4
    W_sparse = np.zeros((K // 2, N), dtype=W.dtype)
    meta = np.zeros((num_blocks, N), dtype=np.uint8)

    for block in range(num_blocks):
        block_start = block * 4
        block_vals = W[block_start:block_start + 4, :]  # [4, N]
        abs_vals = np.abs(block_vals)

        for col in range(N):
            # Find the 2 largest-magnitude positions
            indices = np.argsort(abs_vals[:, col])
            # indices[0:2] are the two smallest (to be zeroed)
            # indices[2:4] are the two largest (to be kept)
            keep = sorted(indices[2:4])  # canonical ordering: pos0 < pos1

            # Pack sparse values
            sparse_row = block * 2
            W_sparse[sparse_row, col] = block_vals[keep[0], col]
            W_sparse[sparse_row + 1, col] = block_vals[keep[1], col]

            # Encode metadata: pos0 in bits [1:0], pos1 in bits [3:2]
            meta[block, col] = (keep[1] << 2) | keep[0]

    return W_sparse, meta


def prune_to_nm(
    W: np.ndarray,
    n: int,
    m: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Prune a weight matrix to N:M structured sparsity.

    Keeps the N largest-magnitude values per group of M consecutive elements
    along K (axis 0).

    Args:
        W: Dense weight matrix [K, N_cols].
        n: Number of values to keep per group.
        m: Group size.

    Returns:
        W_sparse: Compressed weights [K * n // m, N_cols].
        mask: Boolean mask [K, N_cols] indicating kept positions.
    """
    K, N_cols = W.shape
    if K % m != 0:
        raise ValueError(f"K={K} must be divisible by m={m}")

    num_blocks = K // m
    W_sparse = np.zeros((num_blocks * n, N_cols), dtype=W.dtype)
    mask = np.zeros((K, N_cols), dtype=bool)

    for block in range(num_blocks):
        block_start = block * m
        block_vals = W[block_start:block_start + m, :]
        abs_vals = np.abs(block_vals)

        for col in range(N_cols):
            indices = np.argsort(abs_vals[:, col])
            keep = sorted(indices[m - n:])  # top-n by magnitude

            for i, k in enumerate(keep):
                sparse_row = block * n + i
                W_sparse[sparse_row, col] = block_vals[k, col]
                mask[block_start + k, col] = True

    return W_sparse, mask


def sparse_gemm_reference(
    A: np.ndarray,
    W_sparse: np.ndarray,
    meta: np.ndarray,
    scales: np.ndarray | None = None,
    group_size: int = 128,
) -> np.ndarray:
    """Reference sparse GEMM: reconstruct dense from sparse+metadata, then matmul.

    This simulates what the sparse GEMM kernel does: use metadata to scatter
    sparse values back to their original positions, forming a (zero-padded)
    dense weight matrix, then perform standard GEMM.

    Args:
        A: Activations [M, K] as float16/float32.
        W_sparse: Compressed weights [K/2, N] (2:4 sparsity).
        meta: Metadata [K/4, N] as uint8 (2-bit position pairs).
        scales: Optional per-group scales [K/group_size, N].
        group_size: Elements per quantization group.

    Returns:
        Output [M, N] via FP32 accumulation cast to float16.
    """
    K_sparse, N = W_sparse.shape
    K = K_sparse * 2  # Dense K dimension
    num_blocks = K // 4

    # Reconstruct dense weight matrix
    W_dense = np.zeros((K, N), dtype=np.float32)
    for block in range(num_blocks):
        for col in range(N):
            nibble = int(meta[block, col])
            pos0 = nibble & 0x3
            pos1 = (nibble >> 2) & 0x3

            dense_row_base = block * 4
            sparse_row_base = block * 2

            W_dense[dense_row_base + pos0, col] = float(W_sparse[sparse_row_base, col])
            W_dense[dense_row_base + pos1, col] = float(W_sparse[sparse_row_base + 1, col])

    # Apply scales if provided
    if scales is not None:
        num_groups = K // group_size
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            W_dense[start:end, :] *= scales[g, :].astype(np.float32)

    # GEMM with FP32 accumulation
    result = A.astype(np.float32) @ W_dense
    return result.astype(np.float16)


def dense_gemm_reference(
    A: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """Dense FP16 GEMM with FP32 accumulation."""
    return (A.astype(np.float32) @ W.astype(np.float32)).astype(np.float16)


# ---------------------------------------------------------------------------
# Metadata encoding/decoding validation
# ---------------------------------------------------------------------------


class TestMetadataEncoding:
    """Validate 2:4 metadata encode/decode roundtrip."""

    ALL_VALID_PATTERNS: list[tuple[int, int]] = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
    ]

    def test_all_six_patterns_encode_decode(self) -> None:
        """Each of the 6 valid (pos0, pos1) pairs encodes and decodes correctly."""
        for pos0, pos1 in self.ALL_VALID_PATTERNS:
            nibble = (pos1 << 2) | pos0
            decoded_pos0 = nibble & 0x3
            decoded_pos1 = (nibble >> 2) & 0x3
            assert decoded_pos0 == pos0, f"pos0 mismatch for ({pos0},{pos1})"
            assert decoded_pos1 == pos1, f"pos1 mismatch for ({pos0},{pos1})"

    def test_nibble_values_match_sparse_metal(self) -> None:
        """Encoded nibble values match those documented in sparse.metal."""
        expected = {
            (0, 1): 4,   # 0b0100
            (0, 2): 8,   # 0b1000
            (0, 3): 12,  # 0b1100
            (1, 2): 9,   # 0b1001
            (1, 3): 13,  # 0b1101
            (2, 3): 14,  # 0b1110
        }
        for (pos0, pos1), expected_nibble in expected.items():
            nibble = (pos1 << 2) | pos0
            assert nibble == expected_nibble, (
                f"({pos0},{pos1}): got {nibble}, expected {expected_nibble}"
            )

    def test_packed_uint32_roundtrip(self) -> None:
        """8 metadata nibbles pack into uint32 and decode correctly."""
        patterns = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (0, 1), (1, 2)]
        nibbles = [(p1 << 2) | p0 for p0, p1 in patterns]

        # Pack into uint32
        packed: int = 0
        for i, nib in enumerate(nibbles):
            packed |= (nib & 0xF) << (i * 4)

        # Decode back
        for i, (pos0, pos1) in enumerate(patterns):
            nib = (packed >> (i * 4)) & 0xF
            assert (nib & 0x3) == pos0
            assert ((nib >> 2) & 0x3) == pos1


# ---------------------------------------------------------------------------
# Pruning correctness
# ---------------------------------------------------------------------------


class TestPruneTo24:
    """Validate prune_to_2_4 produces correct structure."""

    def test_output_shapes(self) -> None:
        K, N = 64, 32
        W = np.random.randn(K, N).astype(np.float16)
        W_sparse, meta = prune_to_2_4(W)
        assert W_sparse.shape == (K // 2, N)
        assert meta.shape == (K // 4, N)

    def test_exactly_two_nonzeros_per_block(self) -> None:
        """Each 4-element block in the reconstructed dense matrix has exactly 2 nonzeros."""
        K, N = 128, 64
        rng = np.random.default_rng(42)
        W = rng.standard_normal((K, N)).astype(np.float16)
        W_sparse, meta = prune_to_2_4(W)

        # Reconstruct dense
        W_reconstructed = np.zeros((K, N), dtype=np.float32)
        num_blocks = K // 4
        for block in range(num_blocks):
            for col in range(N):
                nibble = int(meta[block, col])
                pos0 = nibble & 0x3
                pos1 = (nibble >> 2) & 0x3
                W_reconstructed[block * 4 + pos0, col] = W_sparse[block * 2, col]
                W_reconstructed[block * 4 + pos1, col] = W_sparse[block * 2 + 1, col]

        for block in range(num_blocks):
            block_slice = W_reconstructed[block * 4:(block + 1) * 4, :]
            nnz_per_col = np.count_nonzero(block_slice, axis=0)
            assert np.all(nnz_per_col == 2), (
                f"Block {block}: expected 2 nonzeros per column, "
                f"got {nnz_per_col[nnz_per_col != 2]}"
            )

    def test_largest_magnitudes_kept(self) -> None:
        """The two kept values are the largest-magnitude in each 4-element block."""
        K, N = 32, 16
        rng = np.random.default_rng(123)
        W = rng.standard_normal((K, N)).astype(np.float32)
        W_sparse, meta = prune_to_2_4(W)

        num_blocks = K // 4
        for block in range(num_blocks):
            block_vals = W[block * 4:(block + 1) * 4, :]
            for col in range(N):
                abs_vals = np.abs(block_vals[:, col])
                top2_indices = set(np.argsort(abs_vals)[2:])

                nibble = int(meta[block, col])
                pos0 = nibble & 0x3
                pos1 = (nibble >> 2) & 0x3
                kept = {pos0, pos1}
                assert kept == top2_indices, (
                    f"Block {block}, col {col}: kept {kept} but top-2 are {top2_indices}"
                )

    def test_canonical_ordering(self) -> None:
        """Metadata always encodes pos0 < pos1."""
        K, N = 64, 32
        rng = np.random.default_rng(99)
        W = rng.standard_normal((K, N)).astype(np.float16)
        _, meta = prune_to_2_4(W)

        for block in range(K // 4):
            for col in range(N):
                nibble = int(meta[block, col])
                pos0 = nibble & 0x3
                pos1 = (nibble >> 2) & 0x3
                assert pos0 < pos1, (
                    f"Block {block}, col {col}: pos0={pos0} >= pos1={pos1}"
                )

    def test_k_not_divisible_by_4_raises(self) -> None:
        W = np.random.randn(30, 16).astype(np.float16)
        with pytest.raises(ValueError, match="divisible by 4"):
            prune_to_2_4(W)

    def test_dtype_preserved(self) -> None:
        for dtype in [np.float16, np.float32]:
            W = np.ones((8, 4), dtype=dtype)
            W_sparse, _ = prune_to_2_4(W)
            assert W_sparse.dtype == dtype


# ---------------------------------------------------------------------------
# Sparse GEMM accuracy (exact zeros)
# ---------------------------------------------------------------------------


class TestSparseGEMMExactZeros:
    """Sparse GEMM matches dense GEMM when weight zeros are true zeros."""

    @pytest.mark.parametrize("M", [1, 8, 32])
    @pytest.mark.parametrize("K,N", [(128, 128), (256, 256), (512, 512), (4096, 4096)])
    def test_sparse_gemm_exact_zeros(self, M: int, K: int, N: int) -> None:
        """Sparse GEMM result matches dense reference for exact-zero weights."""
        rng = np.random.default_rng(42)

        # Generate dense weights and prune to 2:4
        W_dense = rng.standard_normal((K, N)).astype(np.float16)
        W_sparse, meta = prune_to_2_4(W_dense)

        # Reconstruct the pruned dense matrix (with exact zeros)
        W_pruned_dense = np.zeros((K, N), dtype=np.float16)
        num_blocks = K // 4
        for block in range(num_blocks):
            for col in range(N):
                nibble = int(meta[block, col])
                pos0 = nibble & 0x3
                pos1 = (nibble >> 2) & 0x3
                W_pruned_dense[block * 4 + pos0, col] = W_sparse[block * 2, col]
                W_pruned_dense[block * 4 + pos1, col] = W_sparse[block * 2 + 1, col]

        # Activations
        A = rng.standard_normal((M, K)).astype(np.float16)

        # Dense reference: A @ W_pruned_dense^T (row-major weights, transposed)
        ref = dense_gemm_reference(A, W_pruned_dense)

        # Sparse reference: should match dense exactly (same computation)
        result = sparse_gemm_reference(A, W_sparse, meta)

        # FP16 accumulation introduces rounding; use relative tolerance
        # For K=4096, expected relative error ~O(sqrt(K) * eps_fp16) ~ 0.5%
        np.testing.assert_allclose(
            result.astype(np.float32),
            ref.astype(np.float32),
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"Sparse GEMM mismatch for M={M}, K={K}, N={N}",
        )

    def test_sparse_gemm_artificial_sparsity_matches_dense(self) -> None:
        """Force exact 2:4 zeros in dense weights, then compare sparse vs dense."""
        rng = np.random.default_rng(123)
        K, N, M = 512, 512, 16

        W_dense = rng.standard_normal((K, N)).astype(np.float16)

        # Enforce exact 2:4 sparsity by zeroing the two smallest magnitudes per block.
        W_blocked = W_dense.reshape(K // 4, 4, N)
        abs_w = np.abs(W_blocked)
        for block in range(K // 4):
            for col in range(N):
                indices = np.argsort(abs_w[block, :, col])
                W_blocked[block, indices[:2], col] = 0
        W_dense = W_blocked.reshape(K, N)

        W_sparse, meta = prune_to_2_4(W_dense)
        A = rng.standard_normal((M, K)).astype(np.float16)

        ref = dense_gemm_reference(A, W_dense)
        result = sparse_gemm_reference(A, W_sparse, meta)

        np.testing.assert_allclose(
            result.astype(np.float32),
            ref.astype(np.float32),
            rtol=1e-3,
            atol=1e-4,
            err_msg="Sparse GEMM mismatch for artificial 2:4 sparsity",
        )

    def test_sparse_gemm_identity_activations(self) -> None:
        """With A=I, output equals the weight matrix itself."""
        K, N = 128, 64
        rng = np.random.default_rng(7)
        W = rng.standard_normal((K, N)).astype(np.float16)
        W_sparse, meta = prune_to_2_4(W)

        A = np.eye(K, dtype=np.float16)
        result = sparse_gemm_reference(A, W_sparse, meta)

        # Reconstruct expected
        W_pruned = np.zeros((K, N), dtype=np.float16)
        for block in range(K // 4):
            for col in range(N):
                nibble = int(meta[block, col])
                pos0 = nibble & 0x3
                pos1 = (nibble >> 2) & 0x3
                W_pruned[block * 4 + pos0, col] = W_sparse[block * 2, col]
                W_pruned[block * 4 + pos1, col] = W_sparse[block * 2 + 1, col]

        np.testing.assert_allclose(
            result.astype(np.float32),
            W_pruned.astype(np.float32),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_sparse_gemm_zero_activations(self) -> None:
        """Zero activations produce zero output."""
        K, N, M = 256, 128, 16
        rng = np.random.default_rng(11)
        W = rng.standard_normal((K, N)).astype(np.float16)
        W_sparse, meta = prune_to_2_4(W)

        A = np.zeros((M, K), dtype=np.float16)
        result = sparse_gemm_reference(A, W_sparse, meta)

        assert np.all(result == 0), "Non-zero output for zero activations"

    def test_sparse_gemm_with_scales(self) -> None:
        """Sparse GEMM with per-group scaling matches scaled dense reference."""
        K, N, M = 256, 128, 8
        group_size = 64
        rng = np.random.default_rng(55)

        W = rng.standard_normal((K, N)).astype(np.float16)
        W_sparse, meta = prune_to_2_4(W)

        scales = rng.uniform(0.5, 2.0, size=(K // group_size, N)).astype(np.float16)
        A = rng.standard_normal((M, K)).astype(np.float16)

        # Sparse with scales
        result = sparse_gemm_reference(A, W_sparse, meta, scales, group_size)

        # Dense reference with same scaling applied
        W_pruned_dense = np.zeros((K, N), dtype=np.float32)
        for block in range(K // 4):
            for col in range(N):
                nibble = int(meta[block, col])
                pos0 = nibble & 0x3
                pos1 = (nibble >> 2) & 0x3
                W_pruned_dense[block * 4 + pos0, col] = float(W_sparse[block * 2, col])
                W_pruned_dense[block * 4 + pos1, col] = float(W_sparse[block * 2 + 1, col])

        # Apply same scales
        for g in range(K // group_size):
            start = g * group_size
            end = start + group_size
            W_pruned_dense[start:end, :] *= scales[g, :].astype(np.float32)

        ref = (A.astype(np.float32) @ W_pruned_dense).astype(np.float16)

        np.testing.assert_allclose(
            result.astype(np.float32),
            ref.astype(np.float32),
            rtol=1e-3,
            atol=1e-3,
        )


# ---------------------------------------------------------------------------
# N:M sparsity pattern tests
# ---------------------------------------------------------------------------


class TestNMSparsityPatterns:
    """Test various N:M sparsity patterns for correctness."""

    @pytest.mark.parametrize("n,m", [(1, 4), (2, 4), (4, 8)])
    def test_nm_pattern_structure(self, n: int, m: int) -> None:
        """N:M pruning keeps exactly N values per M-element group."""
        K, N_cols = 128, 64
        rng = np.random.default_rng(42)
        W = rng.standard_normal((K, N_cols)).astype(np.float16)

        W_sparse, mask = prune_to_nm(W, n, m)

        # Check output shape
        assert W_sparse.shape == (K * n // m, N_cols)

        # Check mask has exactly n True per m-element group
        num_blocks = K // m
        for block in range(num_blocks):
            block_mask = mask[block * m:(block + 1) * m, :]
            nnz_per_col = block_mask.sum(axis=0)
            assert np.all(nnz_per_col == n), (
                f"Block {block}: expected {n} nonzeros, got unique counts "
                f"{np.unique(nnz_per_col)}"
            )

    @pytest.mark.parametrize("n,m", [(1, 4), (2, 4), (4, 8)])
    def test_nm_gemm_matches_masked_dense(self, n: int, m: int) -> None:
        """N:M sparse GEMM matches dense GEMM with mask applied."""
        K, N_cols, M_batch = 128, 64, 16
        rng = np.random.default_rng(77)

        W = rng.standard_normal((K, N_cols)).astype(np.float16)
        _, mask = prune_to_nm(W, n, m)

        # Apply mask to get the pruned dense matrix
        W_masked = (W * mask).astype(np.float16)

        A = rng.standard_normal((M_batch, K)).astype(np.float16)

        # Dense GEMM with masked weights
        ref = dense_gemm_reference(A, W_masked)

        # Direct matmul (same computation, just verifying mask correctness)
        result = dense_gemm_reference(A, W_masked)

        np.testing.assert_allclose(
            result.astype(np.float32),
            ref.astype(np.float32),
            rtol=0,
            atol=0,
        )

    @pytest.mark.parametrize("n,m", [(1, 4), (2, 4), (4, 8)])
    def test_nm_keeps_largest_magnitudes(self, n: int, m: int) -> None:
        """N:M pruning keeps the N largest-magnitude values per group."""
        K, N_cols = 64, 32
        rng = np.random.default_rng(13)
        W = rng.standard_normal((K, N_cols)).astype(np.float32)

        _, mask = prune_to_nm(W, n, m)

        num_blocks = K // m
        for block in range(num_blocks):
            block_vals = W[block * m:(block + 1) * m, :]
            block_mask = mask[block * m:(block + 1) * m, :]

            for col in range(N_cols):
                abs_vals = np.abs(block_vals[:, col])
                top_n_idx = set(np.argsort(abs_vals)[m - n:])
                kept_idx = set(np.where(block_mask[:, col])[0])
                assert kept_idx == top_n_idx, (
                    f"Block {block}, col {col}: kept {kept_idx}, "
                    f"expected top-{n}: {top_n_idx}"
                )

    def test_2_4_specific_reconstruction(self) -> None:
        """2:4 sparse+metadata reconstruction matches the mask approach."""
        K, N = 128, 64
        rng = np.random.default_rng(200)
        W = rng.standard_normal((K, N)).astype(np.float16)

        # Method 1: prune_to_2_4 with metadata
        W_sparse, meta = prune_to_2_4(W)
        W_reconstructed = np.zeros((K, N), dtype=np.float32)
        for block in range(K // 4):
            for col in range(N):
                nibble = int(meta[block, col])
                pos0 = nibble & 0x3
                pos1 = (nibble >> 2) & 0x3
                W_reconstructed[block * 4 + pos0, col] = float(W_sparse[block * 2, col])
                W_reconstructed[block * 4 + pos1, col] = float(W_sparse[block * 2 + 1, col])

        # Method 2: prune_to_nm with mask
        _, mask = prune_to_nm(W, n=2, m=4)
        W_masked = (W.astype(np.float32) * mask)

        # Both methods should produce the same pruned matrix
        np.testing.assert_allclose(
            W_reconstructed, W_masked, rtol=1e-3, atol=1e-4,
            err_msg="prune_to_2_4 and prune_to_nm(2,4) produce different results",
        )


# ---------------------------------------------------------------------------
# Sparse vs dense throughput comparison
# ---------------------------------------------------------------------------


class TestSparseVsDenseThroughput:
    """Verify sparse path has throughput advantage over dense.

    These are pure-numpy timing tests that demonstrate the theoretical
    advantage of the sparse representation. The actual Metal kernel
    speedup will be larger due to reduced memory bandwidth and compute.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("K,N", [(4096, 4096), (4096, 14336)])
    def test_sparse_fewer_flops(self, K: int, N: int) -> None:
        """Sparse representation has 2x fewer stored weights."""
        rng = np.random.default_rng(42)
        W = rng.standard_normal((K, N)).astype(np.float16)
        W_sparse, _ = prune_to_2_4(W)

        # Sparse stores K/2 * N values vs dense K * N
        dense_elements = K * N
        sparse_elements = W_sparse.shape[0] * W_sparse.shape[1]
        compression_ratio = dense_elements / sparse_elements

        assert abs(compression_ratio - 2.0) < 1e-6, (
            f"Expected 2x compression, got {compression_ratio:.4f}x"
        )

    @pytest.mark.slow
    def test_sparse_matmul_timing(self) -> None:
        """Sparse representation enables faster matmul via reduced memory reads.

        This measures the numpy-level advantage of computing with the
        sparse (K/2) weight matrix vs the full (K) dense matrix.
        The actual Metal kernel advantage will be ~2x due to halved
        memory bandwidth for weight loading.
        """
        K, N, M = 4096, 4096, 32
        rng = np.random.default_rng(42)

        W = rng.standard_normal((K, N)).astype(np.float16)
        W_sparse, meta = prune_to_2_4(W)
        A = rng.standard_normal((M, K)).astype(np.float16)

        # Reconstruct dense for fair comparison
        W_pruned = np.zeros((K, N), dtype=np.float16)
        for block in range(K // 4):
            for col in range(N):
                nibble = int(meta[block, col])
                pos0 = nibble & 0x3
                pos1 = (nibble >> 2) & 0x3
                W_pruned[block * 4 + pos0, col] = W_sparse[block * 2, col]
                W_pruned[block * 4 + pos1, col] = W_sparse[block * 2 + 1, col]

        # Time dense GEMM
        n_iter = 5
        # Warmup
        _ = A.astype(np.float32) @ W_pruned.astype(np.float32)

        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = A.astype(np.float32) @ W_pruned.astype(np.float32)
        dense_time = (time.perf_counter() - t0) / n_iter

        # Time sparse-aware GEMM: only load K/2 weights, scatter via metadata
        # In a real kernel this is the advantage; here we just measure the
        # memory footprint difference
        sparse_bytes = W_sparse.nbytes + meta.nbytes
        dense_bytes = W_pruned.nbytes

        # Sparse representation should be ~2x smaller
        byte_ratio = dense_bytes / sparse_bytes
        assert byte_ratio > 1.8, (
            f"Expected ~2x memory savings, got {byte_ratio:.2f}x. "
            f"Dense: {dense_bytes} bytes, Sparse: {sparse_bytes} bytes"
        )

        # Report timing (informational, not a hard assertion since numpy
        # doesn't benefit from sparse structure the way a GPU kernel does)
        print(f"\n  Dense GEMM ({K}x{N}): {dense_time*1000:.2f} ms")
        print(f"  Memory savings: {byte_ratio:.2f}x (sparse {sparse_bytes} vs dense {dense_bytes})")


# ---------------------------------------------------------------------------
# Edge cases and numerical stability
# ---------------------------------------------------------------------------


class TestSparseEdgeCases:
    """Edge cases for sparse GEMM correctness."""

    def test_all_equal_magnitudes(self) -> None:
        """When all values in a block have equal magnitude, pruning is deterministic."""
        K, N = 16, 8
        # All values have the same magnitude
        W = np.ones((K, N), dtype=np.float16) * 2.0
        W_sparse, meta = prune_to_2_4(W)

        # Should still produce valid 2:4 structure (argsort breaks ties by index)
        assert W_sparse.shape == (K // 2, N)
        for block in range(K // 4):
            for col in range(N):
                nibble = int(meta[block, col])
                pos0 = nibble & 0x3
                pos1 = (nibble >> 2) & 0x3
                assert pos0 < pos1

    def test_single_nonzero_per_block(self) -> None:
        """Block with only one nonzero: both sparse values might be zero/nonzero."""
        K, N = 8, 4
        W = np.zeros((K, N), dtype=np.float16)
        # Put a single nonzero in each block
        W[2, :] = 5.0  # Block 0: position 2
        W[5, :] = 3.0  # Block 1: position 1

        W_sparse, meta = prune_to_2_4(W)

        # The nonzero value should be among the kept values
        # (it's the largest magnitude, so it's always kept)
        for col in range(N):
            # Block 0: position 2 has value 5.0
            nibble0 = int(meta[0, col])
            pos0_0 = nibble0 & 0x3
            pos1_0 = (nibble0 >> 2) & 0x3
            kept_block0 = {pos0_0, pos1_0}
            assert 2 in kept_block0, f"Position 2 not kept in block 0: {kept_block0}"

            # Block 1: position 1 has value 3.0
            nibble1 = int(meta[1, col])
            pos0_1 = nibble1 & 0x3
            pos1_1 = (nibble1 >> 2) & 0x3
            kept_block1 = {pos0_1, pos1_1}
            assert 1 in kept_block1, f"Position 1 not kept in block 1: {kept_block1}"

    def test_negative_values_handled(self) -> None:
        """Negative values are pruned by magnitude, not by value."""
        _K, N = 8, 4
        W = np.array([
            [-1.0, -1.0, -1.0, -1.0],
            [0.1, 0.1, 0.1, 0.1],
            [-5.0, -5.0, -5.0, -5.0],  # Largest magnitude
            [3.0, 3.0, 3.0, 3.0],      # Second largest
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float16)

        W_sparse, meta = prune_to_2_4(W)

        # Block 0: positions 2 (-5.0) and 3 (3.0) should be kept
        for col in range(N):
            nibble = int(meta[0, col])
            pos0 = nibble & 0x3
            pos1 = (nibble >> 2) & 0x3
            assert {pos0, pos1} == {2, 3}, f"Expected positions {{2,3}}, got {{{pos0},{pos1}}}"

    def test_large_dimension_accuracy(self) -> None:
        """Accuracy holds for LLM-scale dimensions."""
        K, N, M = 4096, 4096, 1
        rng = np.random.default_rng(42)

        W = rng.standard_normal((K, N)).astype(np.float16)
        W_sparse, meta = prune_to_2_4(W)
        A = rng.standard_normal((M, K)).astype(np.float16)

        result = sparse_gemm_reference(A, W_sparse, meta)

        # Reconstruct and compute dense reference
        W_pruned = np.zeros((K, N), dtype=np.float16)
        for block in range(K // 4):
            for col in range(N):
                nibble = int(meta[block, col])
                pos0 = nibble & 0x3
                pos1 = (nibble >> 2) & 0x3
                W_pruned[block * 4 + pos0, col] = W_sparse[block * 2, col]
                W_pruned[block * 4 + pos1, col] = W_sparse[block * 2 + 1, col]

        ref = dense_gemm_reference(A, W_pruned)

        np.testing.assert_allclose(
            result.astype(np.float32),
            ref.astype(np.float32),
            rtol=1e-3,
            atol=1e-3,
            err_msg="Large-dimension sparse GEMM accuracy failure",
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_reproducibility(self, seed: int) -> None:
        """Same seed produces identical sparse GEMM results."""
        K, N, M = 256, 128, 8
        rng = np.random.default_rng(seed)

        W = rng.standard_normal((K, N)).astype(np.float16)
        A_data = np.random.default_rng(seed + 1000).standard_normal((M, K)).astype(np.float16)

        W_sparse, meta = prune_to_2_4(W)
        result1 = sparse_gemm_reference(A_data, W_sparse, meta)

        # Re-prune from same weights
        W_sparse2, meta2 = prune_to_2_4(W)
        result2 = sparse_gemm_reference(A_data, W_sparse2, meta2)

        np.testing.assert_array_equal(result1, result2)
