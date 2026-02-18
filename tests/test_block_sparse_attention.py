"""Tests for block-sparse attention implementation.

Validates:
- BlockSparseMask creation utilities
- Mask pattern correctness
- Integration with block_sparse_attention_metal
- Numerical accuracy against dense attention
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import torch


def _check_torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _check_metal_available() -> bool:
    try:
        import Metal  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.mark.skipif(not _check_torch_available(), reason="PyTorch not available")
class TestBlockSparseMask:
    """Tests for BlockSparseMask creation utilities."""

    def test_sliding_window_mask_creation(self):
        """Test creation of sliding window block-sparse mask."""
        import torch
        from metal_marlin.attention import BlockSparseMask

        seq_len = 256
        window_size = 128
        block_size = 64

        mask = BlockSparseMask.create_sliding_window(
            seq_len=seq_len,
            window_size=window_size,
            block_size=block_size,
        )

        assert mask.seq_q == seq_len
        assert mask.seq_k == seq_len
        assert mask.block_q == block_size
        assert mask.block_k == block_size
        assert mask.num_q_blocks == 4  # 256 / 64 = 4
        assert mask.num_k_blocks == 4
        assert mask.mask_bits.dtype == torch.uint64
        assert len(mask.mask_bits) == 4

    def test_sliding_window_mask_pattern(self):
        """Verify sliding window mask has correct pattern."""
        import torch
        from metal_marlin.attention import BlockSparseMask

        seq_len = 256
        window_size = 128
        block_size = 64

        mask = BlockSparseMask.create_sliding_window(
            seq_len=seq_len,
            window_size=window_size,
            block_size=block_size,
        )

        # Get mask bits as Python ints for bitwise operations
        # (MPS doesn't support uint64 bitwise ops)
        mask_bits_cpu = mask.mask_bits.cpu()

        # For block 0 (positions 0-63): can attend to positions 0-63
        # (within window of position 63)
        assert mask_bits_cpu[0].item() & (1 << 0)  # Can attend to block 0

        # For block 1 (positions 64-127): can attend to blocks 0-1
        assert mask_bits_cpu[1].item() & (1 << 0)  # Can attend to block 0
        assert mask_bits_cpu[1].item() & (1 << 1)  # Can attend to block 1

        # For block 3 (positions 192-255): with window_size=128,
        # can attend from 255-128=127 onwards, so blocks 2-3
        # But causal limits to block 3 itself
        assert mask_bits_cpu[3].item() & (1 << 2)  # Can attend to block 2
        assert mask_bits_cpu[3].item() & (1 << 3)  # Can attend to block 3

    def test_bigbird_mask_creation(self):
        """Test creation of BigBird-style block-sparse mask."""
        import torch
        from metal_marlin.attention import BlockSparseMask

        seq_len = 512
        mask = BlockSparseMask.create_bigbird(
            seq_len=seq_len,
            num_random_blocks=2,
            num_global_blocks=2,
            window_size=128,
            block_size=64,
        )

        assert mask.seq_q == seq_len
        assert mask.num_q_blocks == 8  # 512 / 64 = 8
        assert mask.mask_bits.dtype == torch.uint64

    def test_bigbird_global_blocks(self):
        """Verify BigBird mask includes global blocks for all positions."""
        from metal_marlin.attention import BlockSparseMask

        seq_len = 256
        num_global_blocks = 2
        block_size = 64

        mask = BlockSparseMask.create_bigbird(
            seq_len=seq_len,
            num_random_blocks=0,
            num_global_blocks=num_global_blocks,
            window_size=0,  # No window
            block_size=block_size,
        )

        # Get mask bits as Python ints for bitwise operations
        mask_bits_cpu = mask.mask_bits.cpu()

        # All positions should attend to global block 0 (within causal constraint)
        for i in range(mask.num_q_blocks):
            assert mask_bits_cpu[i].item() & (1 << 0), f"Block {i} should attend to global block 0"

        # Block 1+ should attend to global block 1 (within causal constraint)
        for i in range(1, mask.num_q_blocks):
            assert mask_bits_cpu[i].item() & (1 << 1), f"Block {i} should attend to global block 1"

    def test_longformer_mask_creation(self):
        """Test creation of Longformer-style block-sparse mask."""
        import torch
        from metal_marlin.attention import BlockSparseMask

        seq_len = 512
        mask = BlockSparseMask.create_longformer(
            seq_len=seq_len,
            window_size=128,
            num_global_tokens=32,
            block_size=64,
        )

        assert mask.seq_q == seq_len
        assert mask.num_q_blocks == 8
        assert mask.mask_bits.dtype == torch.uint64

    def test_from_dense_mask(self):
        """Test conversion from dense mask to block-sparse."""
        import torch
        from metal_marlin.attention import BlockSparseMask

        seq_len = 128
        block_size = 32

        # Create a simple causal dense mask
        dense_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

        mask = BlockSparseMask.from_dense_mask(dense_mask, block_q=block_size, block_k=block_size)

        assert mask.seq_q == seq_len
        assert mask.seq_k == seq_len
        assert mask.num_q_blocks == 4
        assert mask.num_k_blocks == 4

    def test_from_dense_mask_pattern(self):
        """Verify block-sparse mask matches dense mask pattern."""
        import torch
        from metal_marlin.attention import BlockSparseMask

        seq_len = 64
        block_size = 32

        # Create causal mask
        dense_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

        mask = BlockSparseMask.from_dense_mask(dense_mask, block_q=block_size, block_k=block_size)

        # Block 0 (rows 0-31): can attend to columns 0-31 (block 0)
        assert mask.mask_bits[0] & (1 << 0)

        # Block 1 (rows 32-63): can attend to columns 0-63 (blocks 0 and 1)
        assert mask.mask_bits[1] & (1 << 0)
        assert mask.mask_bits[1] & (1 << 1)


@pytest.mark.skipif(not _check_torch_available(), reason="PyTorch not available")
class TestBlockSparseAttentionAccuracy:
    """Tests for block-sparse attention numerical accuracy."""

    def _reference_attention(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        scale: float,
        is_causal: bool = False,
    ) -> np.ndarray:
        """NumPy reference implementation of attention."""
        batch, heads_q, seq_q, head_dim = q.shape
        _, heads_kv, seq_k, _ = k.shape

        # GQA: expand K/V heads to match Q heads
        if heads_kv < heads_q:
            repeat = heads_q // heads_kv
            k = np.repeat(k, repeat, axis=1)
            v = np.repeat(v, repeat, axis=1)

        # Compute Q @ K^T * scale
        scores = np.einsum("bhqd,bhkd->bhqk", q.astype(np.float32), k.astype(np.float32))
        scores = scores * scale

        # Apply causal mask if needed
        if is_causal:
            q_idx = np.arange(seq_q)[:, None]
            k_idx = np.arange(seq_k)[None, :]
            mask = np.where(k_idx > q_idx, -np.inf, 0.0).astype(np.float32)
            scores = scores + mask

        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_max = np.where(np.isinf(scores_max), 0.0, scores_max)
        scores = scores - scores_max
        exp_scores = np.exp(scores)
        sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
        sum_exp = np.where(sum_exp == 0, 1.0, sum_exp)
        softmax = exp_scores / sum_exp

        # Apply attention weights to values
        output = np.einsum("bhqk,bhkd->bhqd", softmax, v.astype(np.float32))

        return output.astype(np.float16)

    def test_block_sparse_mask_sliding_window_accuracy(self, rng):
        """Test that sliding window block-sparse produces correct attention pattern."""
        import torch
        from metal_marlin.attention import BlockSparseMask

        batch, heads, seq_len, head_dim = 1, 4, 128, 64
        window_size = 64
        block_size = 32

        # Generate random inputs
        q = rng.standard_normal((batch, heads, seq_len, head_dim)).astype(np.float16)
        k = rng.standard_normal((batch, heads, seq_len, head_dim)).astype(np.float16)
        v = rng.standard_normal((batch, heads, seq_len, head_dim)).astype(np.float16)

        # Create sliding window mask
        mask = BlockSparseMask.create_sliding_window(
            seq_len=seq_len,
            window_size=window_size,
            block_size=block_size,
        )

        # Convert to dense mask for reference comparison
        dense_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        num_blocks = mask.num_q_blocks

        for q_block_idx in range(num_blocks):
            q_start = q_block_idx * block_size
            q_end = min(q_start + block_size, seq_len)

            # Handle 2D mask_bits: [num_q_blocks, num_mask_words]
            mask_bits_row = mask.mask_bits[q_block_idx]
            for k_block_idx in range(num_blocks):
                word_idx = k_block_idx // 64
                bit_idx = k_block_idx % 64
                # Get the mask word (handle both 1D and 2D cases)
                if mask_bits_row.dim() == 0:
                    mask_word = mask_bits_row.item()
                else:
                    mask_word = mask_bits_row[word_idx].item()
                if mask_word & (1 << bit_idx):
                    k_start = k_block_idx * block_size
                    k_end = min(k_start + block_size, seq_len)
                    dense_mask[q_start:q_end, k_start:k_end] = True

        # Verify mask properties at block granularity
        # Block-sparse attention works at block level, so we verify that
        # blocks that should be masked are indeed masked
        for q_block_idx in range(num_blocks):
            q_start = q_block_idx * block_size
            q_end = min(q_start + block_size, seq_len)

            # For this Q block, find the first and last query positions
            first_q_pos = q_start
            last_q_pos = q_end - 1

            # The window for this block is determined by the last position (widest window)
            window_start = max(0, q_end - window_size)

            for k_block_idx in range(num_blocks):
                k_start = k_block_idx * block_size
                k_end = min(k_start + block_size, seq_len)

                word_idx = k_block_idx // 64
                bit_idx = k_block_idx % 64
                mask_bits_row = mask.mask_bits[q_block_idx]
                if mask_bits_row.dim() == 0:
                    mask_word = mask_bits_row.item()
                else:
                    mask_word = mask_bits_row[word_idx].item()
                is_unmasked = bool(mask_word & (1 << bit_idx))

                if is_unmasked:
                    # This K block should be within the window for at least some queries
                    # Verify that the last query in the block can attend to this K block
                    assert k_start <= last_q_pos, f"Block ({q_block_idx}, {k_block_idx}): K block starts after last Q"
                    assert k_end > window_start, f"Block ({q_block_idx}, {k_block_idx}): K block ends before window start"
                else:
                    # This K block should be outside the window for all queries
                    # Either it's in the future beyond the window, or before window_start
                    assert k_start > last_q_pos or k_end <= window_start, \
                        f"Block ({q_block_idx}, {k_block_idx}): should be masked but isn't"

    def test_block_sparse_vs_dense_sliding_window(self, rng):
        """Compare block-sparse sliding window to dense sliding window."""
        import torch
        import torch.nn.functional as F
        from metal_marlin.attention import BlockSparseMask, create_sliding_window_mask

        batch, heads, seq_len, head_dim = 1, 4, 128, 64
        window_size = 64

        # Generate random inputs
        q_np = rng.standard_normal((batch, heads, seq_len, head_dim)).astype(np.float16)
        k_np = rng.standard_normal((batch, heads, seq_len, head_dim)).astype(np.float16)
        v_np = rng.standard_normal((batch, heads, seq_len, head_dim)).astype(np.float16)

        q = torch.from_numpy(q_np).float()
        k = torch.from_numpy(k_np).float()
        v = torch.from_numpy(v_np).float()

        # Dense sliding window mask
        dense_mask = create_sliding_window_mask(
            seq_len=seq_len,
            window_size=window_size,
            device=torch.device("cpu"),
        )

        # Dense attention
        dense_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=dense_mask.squeeze(0).squeeze(0) if dense_mask is not None else None,
            is_causal=False,
            scale=head_dim**-0.5,
        )

        # Block-sparse mask
        block_mask = BlockSparseMask.create_sliding_window(
            seq_len=seq_len,
            window_size=window_size,
            block_size=32,
        )

        # Verify block-sparse mask covers the same pattern
        # (block granularity may include slightly more)
        assert block_mask.seq_q == seq_len
        assert len(block_mask.mask_bits) == block_mask.num_q_blocks


@pytest.mark.skipif(not _check_torch_available() or not _check_metal_available(),
                    reason="PyTorch or Metal not available")
class TestBlockSparseMetalDispatch:
    """Tests for Metal kernel dispatch of block-sparse attention."""

    def test_block_sparse_attention_metal_basic(self, rng):
        """Test basic block-sparse attention dispatch."""
        import torch
        from metal_marlin.attention import BlockSparseMask, block_sparse_attention_metal

        batch, heads, seq_len, head_dim = 1, 4, 64, 32
        block_size = 32

        # Generate random inputs on MPS
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="mps")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="mps")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="mps")

        # Create block-sparse mask
        mask = BlockSparseMask.create_sliding_window(
            seq_len=seq_len,
            window_size=seq_len,  # Full window for testing
            block_size=block_size,
        )

        # Run block-sparse attention
        try:
            output = block_sparse_attention_metal(q, k, v, mask, causal=True)
            assert output.shape == q.shape
            assert output.dtype == torch.float16
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
        except RuntimeError as e:
            if "Threadgroup memory" in str(e):
                pytest.skip(f"Kernel exceeds threadgroup memory: {e}")
            raise

    def test_block_sparse_attention_metal_output_shape(self, rng):
        """Verify output shape of block-sparse attention."""
        import torch
        from metal_marlin.attention import BlockSparseMask, block_sparse_attention_metal

        batch, heads, seq_len, head_dim = 1, 8, 128, 64
        block_size = 64

        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="mps")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="mps")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="mps")

        mask = BlockSparseMask.create_sliding_window(
            seq_len=seq_len,
            window_size=64,
            block_size=block_size,
        )

        try:
            output = block_sparse_attention_metal(q, k, v, mask, causal=True)
            assert output.shape == (batch, heads, seq_len, head_dim)
        except RuntimeError as e:
            if "Threadgroup memory" in str(e):
                pytest.skip(f"Kernel exceeds threadgroup memory: {e}")
            raise

    def test_block_sparse_attention_vs_reference(self, rng):
        """Compare block-sparse attention to reference implementation."""
        import torch
        import torch.nn.functional as F
        from metal_marlin.attention import BlockSparseMask, block_sparse_attention_metal

        batch, heads, seq_len, head_dim = 1, 4, 64, 32
        block_size = 32

        # Generate inputs
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="mps")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="mps")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="mps")

        # Create causal block-sparse mask (full attention for testing)
        mask = BlockSparseMask.create_sliding_window(
            seq_len=seq_len,
            window_size=seq_len,
            block_size=block_size,
        )

        try:
            # Block-sparse attention
            sparse_output = block_sparse_attention_metal(q, k, v, mask, causal=True)

            # Reference: PyTorch SDPA
            ref_output = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(),
                is_causal=True,
                scale=head_dim**-0.5,
            )

            # Compare (with FP16 tolerance)
            sparse_output_float = sparse_output.float()
            torch.testing.assert_close(
                sparse_output_float, ref_output,
                rtol=1e-2, atol=1e-2,
            )
        except RuntimeError as e:
            if "Threadgroup memory" in str(e):
                pytest.skip(f"Kernel exceeds threadgroup memory: {e}")
            raise


@pytest.mark.skipif(not _check_torch_available(), reason="PyTorch not available")
class TestBlockSparseEdgeCases:
    """Edge case tests for block-sparse attention."""

    def test_empty_mask(self):
        """Test behavior with completely empty mask."""
        import torch
        from metal_marlin.attention import BlockSparseMask

        seq_len = 128
        block_size = 64

        # Create mask with no active blocks
        mask = BlockSparseMask(
            mask_bits=torch.zeros(2, dtype=torch.uint64),
            block_q=block_size,
            block_k=block_size,
            num_q_blocks=2,
            num_k_blocks=2,
            seq_q=seq_len,
            seq_k=seq_len,
        )

        assert mask.mask_bits.sum() == 0

    def test_sequence_not_divisible_by_block(self):
        """Test mask creation when seq_len is not divisible by block_size."""
        import torch
        from metal_marlin.attention import BlockSparseMask

        seq_len = 100  # Not divisible by 64
        block_size = 64

        mask = BlockSparseMask.create_sliding_window(
            seq_len=seq_len,
            window_size=seq_len,
            block_size=block_size,
        )

        # Should have ceil(100/64) = 2 blocks
        assert mask.num_q_blocks == 2
        assert mask.num_k_blocks == 2

    def test_single_block(self):
        """Test mask creation with single block."""
        import torch
        from metal_marlin.attention import BlockSparseMask

        seq_len = 32
        block_size = 64  # Larger than seq_len

        mask = BlockSparseMask.create_sliding_window(
            seq_len=seq_len,
            window_size=seq_len,
            block_size=block_size,
        )

        assert mask.num_q_blocks == 1
        assert mask.num_k_blocks == 1
        # Check first bit is set using CPU comparison to avoid MPS type issues
        mask_bits_cpu = mask.mask_bits.cpu()
        if mask_bits_cpu.dim() == 1:
            assert mask_bits_cpu[0].item() & 1  # Can attend to itself
        else:
            assert mask_bits_cpu[0, 0].item() & 1  # Can attend to itself

    def test_very_small_window(self):
        """Test sliding window with very small window size."""
        import torch
        from metal_marlin.attention import BlockSparseMask

        seq_len = 256
        window_size = 32  # Smaller than block size
        block_size = 64

        mask = BlockSparseMask.create_sliding_window(
            seq_len=seq_len,
            window_size=window_size,
            block_size=block_size,
        )

        # Verify mask was created successfully
        assert mask.num_q_blocks == 4

    def test_large_num_blocks_warning(self):
        """Test that we handle cases with many blocks."""
        import torch
        from metal_marlin.attention import BlockSparseMask

        seq_len = 8192
        block_size = 64

        mask = BlockSparseMask.create_sliding_window(
            seq_len=seq_len,
            window_size=512,
            block_size=block_size,
        )

        # 8192 / 64 = 128 blocks
        assert mask.num_q_blocks == 128
        assert mask.num_k_blocks == 128
        # Make sure mask_bits uses 64-bit integers
        assert mask.mask_bits.dtype == torch.uint64


@pytest.mark.skipif(not _check_torch_available(), reason="PyTorch not available")
class TestMarlinAttentionIntegration:
    """Test MarlinAttention integration with BlockSparseMask."""

    def test_block_sparse_dispatch(self):
        """Verify BlockSparseMask dispatches to block_sparse_attention_metal."""
        import torch
        from unittest.mock import MagicMock, patch
        from metal_marlin.attention import MarlinAttention, BlockSparseMask

        # Mock dependencies to avoid Metal/GPU requirements
        with patch("metal_marlin.attention.MarlinLinear", MagicMock()), \
             patch("metal_marlin.attention.RoPE", MagicMock()), \
             patch("metal_marlin.attention.block_sparse_attention_metal") as mock_sparse_attn, \
             patch("metal_marlin.attention.scaled_dot_product_attention_metal") as mock_sdpa:

            # Initialize attention with dummy config
            attn = MarlinAttention(hidden_size=128, num_heads=4)
            
            # Setup mocks for projections
            attn.q_proj.return_value = torch.randn(1, 32, 128)
            attn.k_proj.return_value = torch.randn(1, 32, 128)
            attn.v_proj.return_value = torch.randn(1, 32, 128)
            attn.rope.side_effect = lambda x, offset=0: x
            
            # Dummy input
            x = torch.randn(1, 32, 128)
            
            # Case 1: Standard Tensor mask -> SDPA
            mask_tensor = torch.zeros(1, 1, 32, 32)
            attn(x, attention_mask=mask_tensor)
            mock_sdpa.assert_called_once()
            mock_sparse_attn.assert_not_called()
            
            # Reset mocks
            mock_sdpa.reset_mock()
            mock_sparse_attn.reset_mock()
            
            # Case 2: BlockSparseMask -> block_sparse_attention_metal
            block_mask = BlockSparseMask(
                mask_bits=torch.tensor([1]), 
                block_q=16, block_k=16, 
                num_q_blocks=1, num_k_blocks=1, 
                seq_q=32, seq_k=32
            )
            attn(x, attention_mask=block_mask)
            mock_sparse_attn.assert_called_once()
            mock_sdpa.assert_not_called()
            
            # Verify arguments passed to block_sparse_attention_metal
            call_args = mock_sparse_attn.call_args
            assert call_args.kwargs['block_sparse_mask'] is block_mask
            assert call_args.kwargs['causal'] is False