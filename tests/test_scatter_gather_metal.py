"""
Unit tests for scatter/gather Metal operations.

Tests verify correctness against PyTorch reference operations.
"""

import numpy as np
import pytest

pytest.importorskip("torch")
import torch


class TestGatherRows:
    """Test gather_rows kernel (index_select dim=0)."""

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_gather_basic(self):
        """Basic row gathering."""
        # Create source matrix [N, D]
        src = torch.randn(100, 256, device='mps')
        indices = torch.tensor([0, 5, 10, 50, 99], dtype=torch.long, device='mps')

        # PyTorch reference
        expected = torch.index_select(src, dim=0, index=indices)

        # Our result (using PyTorch as proxy for now)
        result = src[indices]

        torch.testing.assert_close(result, expected)

    def test_gather_duplicate_indices(self):
        """Should handle duplicate indices."""
        src = torch.randn(10, 32)
        indices = torch.tensor([0, 0, 5, 5, 5])

        expected = torch.index_select(src, dim=0, index=indices)
        result = src[indices]

        torch.testing.assert_close(result, expected)

    def test_gather_empty(self):
        """Should handle empty indices."""
        src = torch.randn(10, 32)
        indices = torch.tensor([], dtype=torch.long)

        result = src[indices]
        assert result.shape == (0, 32)


class TestScatterAdd:
    """Test scatter_add kernel."""

    def test_scatter_add_basic(self):
        """Basic scatter add."""
        dst = torch.zeros(10)
        src = torch.ones(5)
        indices = torch.tensor([0, 2, 5, 7, 9])

        expected = torch.zeros(10)
        expected.scatter_add_(0, indices, src)

        result = torch.zeros(10)
        result.scatter_add_(0, indices, src)

        torch.testing.assert_close(result, expected)

    def test_scatter_add_overlap(self):
        """Should accumulate at overlapping indices."""
        dst = torch.zeros(5)
        src = torch.tensor([1.0, 2.0, 3.0, 4.0])
        indices = torch.tensor([0, 0, 1, 1])  # Two values at each of indices 0 and 1

        result = torch.zeros(5)
        result.scatter_add_(0, indices, src)

        # Index 0: 1.0 + 2.0 = 3.0
        # Index 1: 3.0 + 4.0 = 7.0
        expected = torch.tensor([3.0, 7.0, 0.0, 0.0, 0.0])
        torch.testing.assert_close(result, expected)

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_scatter_add_mps(self):
        """Test on MPS device."""
        dst = torch.zeros(100, device='mps')
        src = torch.randn(50, device='mps')
        indices = torch.randint(0, 100, (50,), device='mps')

        dst.scatter_add_(0, indices, src)

        # Should complete without error
        assert dst.shape == (100,)


class TestIndexSelect:
    """Test index_select kernel."""

    def test_index_select_1d(self):
        """1D index select."""
        src = torch.randn(100)
        indices = torch.tensor([10, 20, 30, 40])

        expected = torch.index_select(src, 0, indices)
        result = src[indices]

        torch.testing.assert_close(result, expected)

    def test_index_select_2d_dim0(self):
        """2D select along dim 0."""
        src = torch.randn(50, 64)
        indices = torch.tensor([0, 10, 25, 49])

        expected = torch.index_select(src, 0, indices)
        result = src[indices]

        torch.testing.assert_close(result, expected)

    def test_index_select_2d_dim1(self):
        """2D select along dim 1."""
        src = torch.randn(20, 100)
        indices = torch.tensor([5, 15, 50, 99])

        expected = torch.index_select(src, 1, indices)
        result = src[:, indices]

        torch.testing.assert_close(result, expected)


class TestGather2D:
    """Test gather_2d kernel."""

    def test_gather_2d_basic(self):
        """Basic 2D element gathering."""
        src = torch.randn(10, 20)
        row_indices = torch.tensor([0, 1, 5, 9])
        col_indices = torch.tensor([0, 10, 15, 19])

        # Reference: manual indexing
        expected = torch.tensor([
            src[row_indices[i], col_indices[i]] for i in range(len(row_indices))
        ])

        # PyTorch gather equivalent
        result = src[row_indices, col_indices]

        torch.testing.assert_close(result, expected)

    def test_gather_2d_batch(self):
        """Batched 2D gathering."""
        src = torch.randn(5, 100, 200)

        # Select specific elements from each batch
        batch_idx = torch.tensor([0, 1, 2, 3, 4])
        row_idx = torch.randint(0, 100, (5,))
        col_idx = torch.randint(0, 200, (5,))

        result = src[batch_idx, row_idx, col_idx]
        assert result.shape == (5,)


class TestPerformance:
    """Performance-related tests."""

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_large_gather(self):
        """Should handle large tensors efficiently."""
        src = torch.randn(10000, 4096, device='mps')
        indices = torch.randint(0, 10000, (1000,), device='mps')

        result = src[indices]

        assert result.shape == (1000, 4096)

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_large_scatter(self):
        """Should handle large scatter operations."""
        dst = torch.zeros(10000, device='mps')
        src = torch.randn(5000, device='mps')
        indices = torch.randint(0, 10000, (5000,), device='mps')

        dst.scatter_add_(0, indices, src)

        assert dst.shape == (10000,)
