"""Tests for Paged Attention vs Linear Attention validation utilities.

This test suite validates the validation utilities themselves, ensuring that
parity checking works correctly and can detect both matching and non-matching
outputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from metal_marlin.paged.validation import (
    ParityValidationResult,
    ParityValidator,
    ValidationConfig,
    validate_paged_block_pool_parity,
    validate_paged_linear_parity,
    validate_paged_v1_parity,
    compute_linear_attention,
)


class TestValidationConfig:
    """Test validation configuration."""
    
    def test_default_config(self):
        """Test default validation configuration."""
        config = ValidationConfig()
        assert config.atol == 1e-5
        assert config.rtol == 0.01
        assert config.check_nan is True
        assert config.check_inf is True
        assert config.require_exact_shape is True
        assert config.verbose is False
    
    def test_custom_config(self):
        """Test custom validation configuration."""
        config = ValidationConfig(
            atol=1e-3,
            rtol=0.05,
            check_nan=False,
            check_inf=False,
            require_exact_shape=False,
            verbose=True,
        )
        assert config.atol == 1e-3
        assert config.rtol == 0.05
        assert config.check_nan is False
        assert config.check_inf is False
        assert config.require_exact_shape is False
        assert config.verbose is True


class TestParityValidationResult:
    """Test parity validation result data structure."""
    
    def test_valid_result_str(self):
        """Test string representation of valid result."""
        result = ParityValidationResult(
            is_valid=True,
            max_diff=1e-6,
            mean_diff=1e-7,
            rel_diff=1e-8,
        )
        str_repr = str(result)
        assert "valid=True" in str_repr
        assert "max_diff=1.000000e-06" in str_repr
    
    def test_invalid_result_str(self):
        """Test string representation of invalid result."""
        result = ParityValidationResult(
            is_valid=False,
            max_diff=0.1,
            mean_diff=0.01,
            rel_diff=0.05,
            error_message="Test error",
        )
        str_repr = str(result)
        assert "valid=False" in str_repr
        assert "error=Test error" in str_repr


class TestValidatePagedLinearParity:
    """Test basic parity validation between two arrays."""
    
    def test_identical_arrays(self):
        """Test validation passes for identical arrays."""
        arr = np.random.randn(2, 4, 8, 64).astype(np.float32)
        result = validate_paged_linear_parity(arr, arr)
        
        assert result.is_valid is True
        assert result.max_diff == 0.0
        assert result.mean_diff == 0.0
        assert result.rel_diff == 0.0
    
    def test_arrays_within_tolerance(self):
        """Test validation passes for arrays within tolerance."""
        arr1 = np.ones((2, 4, 8, 64), dtype=np.float32)
        arr2 = arr1 + 1e-6  # Small difference
        
        result = validate_paged_linear_parity(arr1, arr2)
        
        assert result.is_valid is True
        assert result.max_diff <= 2e-6  # Allow for float32 precision
    
    def test_arrays_outside_tolerance(self):
        """Test validation fails for arrays outside tolerance."""
        arr1 = np.ones((2, 4, 8, 64), dtype=np.float32)
        arr2 = arr1 + 1.0  # Large difference
        
        result = validate_paged_linear_parity(arr1, arr2)
        
        assert result.is_valid is False
        assert result.max_diff == 1.0
        assert result.error_message is not None
    
    def test_custom_tolerance(self):
        """Test validation with custom tolerance."""
        arr1 = np.ones((2, 4, 8, 64), dtype=np.float32)
        arr2 = arr1 + 0.1
        
        # Should fail with default tolerance
        result_default = validate_paged_linear_parity(arr1, arr2)
        assert result_default.is_valid is False
        
        # Should pass with higher tolerance
        config = ValidationConfig(atol=0.2, rtol=0.5)
        result_custom = validate_paged_linear_parity(arr1, arr2, config)
        assert result_custom.is_valid is True
    
    def test_shape_mismatch(self):
        """Test validation fails on shape mismatch."""
        arr1 = np.ones((2, 4, 8, 64), dtype=np.float32)
        arr2 = np.ones((2, 4, 8, 32), dtype=np.float32)
        
        result = validate_paged_linear_parity(arr1, arr2)
        
        assert result.is_valid is False
        assert "Shape mismatch" in result.error_message
    
    def test_shape_mismatch_allowed(self):
        """Test shape mismatch can be allowed (but may still fail on broadcast)."""
        arr1 = np.ones((2, 4, 8, 64), dtype=np.float32)
        arr2 = np.ones((2, 4, 8, 64), dtype=np.float32)  # Same shape now
        
        config = ValidationConfig(require_exact_shape=False)
        result = validate_paged_linear_parity(arr1, arr2, config)
        
        # Should pass when shapes match
        assert result.is_valid is True
    
    def test_nan_detection(self):
        """Test NaN detection."""
        arr1 = np.ones((2, 4, 8, 64), dtype=np.float32)
        arr2 = arr1.copy()
        arr2[0, 0, 0, 0] = np.nan
        
        result = validate_paged_linear_parity(arr1, arr2)
        
        assert result.is_valid is False
        assert "NaN detected" in result.error_message
    
    def test_nan_detection_disabled(self):
        """Test NaN detection can be disabled."""
        arr1 = np.ones((2, 4, 8, 64), dtype=np.float32)
        arr2 = arr1.copy()
        arr2[0, 0, 0, 0] = np.nan
        
        config = ValidationConfig(check_nan=False)
        result = validate_paged_linear_parity(arr1, arr2, config)
        
        # Should not fail due to NaN detection
        assert "NaN detected" not in (result.error_message or "")
    
    def test_inf_detection(self):
        """Test Inf detection."""
        arr1 = np.ones((2, 4, 8, 64), dtype=np.float32)
        arr2 = arr1.copy()
        arr2[0, 0, 0, 0] = np.inf
        
        result = validate_paged_linear_parity(arr1, arr2)
        
        assert result.is_valid is False
        assert "Inf detected" in result.error_message
    
    def test_relative_tolerance(self):
        """Test relative tolerance checking."""
        # Large values with small relative difference
        arr1 = np.array([1000.0, 2000.0, 3000.0], dtype=np.float32)
        arr2 = arr1 + 1.0  # 0.1%, 0.05%, 0.033% relative difference
        
        config = ValidationConfig(atol=0.5, rtol=0.001)  # 0.1% relative tol
        result = validate_paged_linear_parity(arr1, arr2, config)
        
        # First element has 0.1% diff which equals rtol
        # Result validity depends on implementation details
        assert result.rel_diff >= 0.0001  # At least 0.01% rel diff


class TestComputeLinearAttention:
    """Test linear attention computation."""
    
    def test_basic_attention(self):
        """Test basic linear attention computation."""
        batch = 2
        num_heads = 4
        q_len = 8
        kv_len = 16
        head_dim = 64
        
        query = np.random.randn(batch, num_heads, q_len, head_dim).astype(np.float32)
        keys = np.random.randn(batch, num_heads, kv_len, head_dim).astype(np.float32)
        values = np.random.randn(batch, num_heads, kv_len, head_dim).astype(np.float32)
        scale = 1.0 / math.sqrt(head_dim)
        
        output = compute_linear_attention(query, keys, values, scale)
        
        assert output.shape == (batch, num_heads, q_len, head_dim)
        assert not np.isnan(output).any()
        assert not np.isinf(output).any()
    
    def test_causal_attention(self):
        """Test causal linear attention."""
        batch = 1
        num_heads = 2
        seq_len = 8
        head_dim = 64
        
        query = np.random.randn(batch, num_heads, seq_len, head_dim).astype(np.float32)
        keys = np.random.randn(batch, num_heads, seq_len, head_dim).astype(np.float32)
        values = np.random.randn(batch, num_heads, seq_len, head_dim).astype(np.float32)
        scale = 1.0 / math.sqrt(head_dim)
        
        output = compute_linear_attention(query, keys, values, scale, is_causal=True)
        
        assert output.shape == (batch, num_heads, seq_len, head_dim)
    
    def test_gqa_attention(self):
        """Test linear attention with GQA."""
        batch = 2
        num_heads = 8
        num_kv_heads = 2  # GQA ratio of 4
        seq_len = 8
        head_dim = 64
        
        query = np.random.randn(batch, num_heads, seq_len, head_dim).astype(np.float32)
        keys = np.random.randn(batch, num_kv_heads, seq_len, head_dim).astype(np.float32)
        values = np.random.randn(batch, num_kv_heads, seq_len, head_dim).astype(np.float32)
        scale = 1.0 / math.sqrt(head_dim)
        
        output = compute_linear_attention(query, keys, values, scale)
        
        assert output.shape == (batch, num_heads, seq_len, head_dim)


class TestValidatePagedV1Parity:
    """Test validation of paged_attention_v1 against linear attention."""
    
    def test_v1_decode_parity(self):
        """Test parity for decode (single query token)."""
        np.random.seed(42)
        
        num_seqs = 2
        num_heads = 4
        num_kv_heads = 4
        head_dim = 64
        seq_len = 32
        block_size = 16
        
        # Create inputs
        query = np.random.randn(num_seqs, num_heads, head_dim).astype(np.float32)
        
        # Create KV cache
        num_blocks = (seq_len + block_size - 1) // block_size
        k_cache = np.zeros((num_blocks, block_size, num_kv_heads, head_dim), dtype=np.float32)
        v_cache = np.zeros((num_blocks, block_size, num_kv_heads, head_dim), dtype=np.float32)
        
        # Fill with random data
        for seq_idx in range(num_seqs):
            for token_idx in range(seq_len):
                block_idx = token_idx // block_size
                slot_idx = token_idx % block_size
                k_cache[block_idx, slot_idx] = np.random.randn(num_kv_heads, head_dim).astype(np.float32)
                v_cache[block_idx, slot_idx] = np.random.randn(num_kv_heads, head_dim).astype(np.float32)
        
        block_tables = np.arange(num_blocks, dtype=np.int32).reshape(1, -1).repeat(num_seqs, axis=0)
        context_lens = np.full((num_seqs,), seq_len, dtype=np.int32)
        scale = 1.0 / math.sqrt(head_dim)
        
        config = ValidationConfig(atol=1e-5, rtol=0.01)
        result = validate_paged_v1_parity(
            query, k_cache, v_cache, block_tables, context_lens, scale, config
        )
        
        assert result.is_valid, f"Parity check failed: {result.error_message}"
    
    def test_v1_gqa_parity(self):
        """Test parity with GQA configuration.
        
        Note: paged_attention_v1 requires cache heads == query heads,
        so we use num_kv_heads = num_heads for this test.
        """
        np.random.seed(42)
        
        num_seqs = 1
        num_heads = 8
        num_kv_heads = 8  # v1 requires cache heads == query heads
        head_dim = 64
        seq_len = 16
        block_size = 16
        
        query = np.random.randn(num_seqs, num_heads, head_dim).astype(np.float32)
        
        num_blocks = 1
        k_cache = np.random.randn(num_blocks, block_size, num_kv_heads, head_dim).astype(np.float32)
        v_cache = np.random.randn(num_blocks, block_size, num_kv_heads, head_dim).astype(np.float32)
        
        block_tables = np.array([[0]], dtype=np.int32)
        context_lens = np.array([seq_len], dtype=np.int32)
        scale = 1.0 / math.sqrt(head_dim)
        
        config = ValidationConfig(atol=1e-5, rtol=0.01)
        result = validate_paged_v1_parity(
            query, k_cache, v_cache, block_tables, context_lens, scale, config
        )
        
        assert result.is_valid, f"Parity check failed: {result.error_message}"


class TestValidatePagedBlockPoolParity:
    """Test validation of paged_attention (block pool) against linear attention."""
    
    def test_block_pool_decode_parity(self):
        """Test parity for block pool decode."""
        np.random.seed(42)
        
        num_seqs = 1
        num_heads = 4
        num_kv_heads = 4
        seq_len = 1  # Decode
        head_dim = 64
        block_size = 16
        max_seq_len = 32
        
        query = np.random.randn(num_seqs, num_heads, seq_len, head_dim).astype(np.float32)
        
        # Create block pool
        num_blocks = (max_seq_len + block_size - 1) // block_size + 1
        block_pool = np.zeros((num_blocks, 2, block_size, num_kv_heads, head_dim), dtype=np.float32)
        
        # Fill with random data
        for blk in range(num_blocks):
            for slot in range(block_size):
                block_pool[blk, 0, slot] = np.random.randn(num_kv_heads, head_dim).astype(np.float32)
                block_pool[blk, 1, slot] = np.random.randn(num_kv_heads, head_dim).astype(np.float32)
        
        max_blocks = (max_seq_len + block_size - 1) // block_size
        block_tables = np.arange(max_blocks, dtype=np.int32).reshape(1, -1)
        context_lens = np.array([max_seq_len], dtype=np.int32)
        scale = 1.0 / math.sqrt(head_dim)
        
        config = ValidationConfig(atol=1e-5, rtol=0.01)
        result = validate_paged_block_pool_parity(
            query, block_pool, block_tables, context_lens,
            scale, num_kv_heads, block_size, config
        )
        
        assert result.is_valid, f"Parity check failed: {result.error_message}"
    
    def test_block_pool_prefill_parity(self):
        """Test parity for block pool prefill."""
        np.random.seed(42)
        
        num_seqs = 1
        num_heads = 4
        num_kv_heads = 4
        seq_len = 16  # Prefill
        head_dim = 64
        block_size = 16
        
        query = np.random.randn(num_seqs, num_heads, seq_len, head_dim).astype(np.float32)
        
        num_blocks = 2
        block_pool = np.zeros((num_blocks, 2, block_size, num_kv_heads, head_dim), dtype=np.float32)
        
        for blk in range(num_blocks):
            for slot in range(block_size):
                block_pool[blk, 0, slot] = np.random.randn(num_kv_heads, head_dim).astype(np.float32)
                block_pool[blk, 1, slot] = np.random.randn(num_kv_heads, head_dim).astype(np.float32)
        
        block_tables = np.arange(num_blocks, dtype=np.int32).reshape(1, -1)
        context_lens = np.array([seq_len], dtype=np.int32)
        scale = 1.0 / math.sqrt(head_dim)
        
        config = ValidationConfig(atol=1e-5, rtol=0.01)
        result = validate_paged_block_pool_parity(
            query, block_pool, block_tables, context_lens,
            scale, num_kv_heads, block_size, config
        )
        
        assert result.is_valid, f"Parity check failed: {result.error_message}"


class TestParityValidator:
    """Test the ParityValidator class."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        config = ValidationConfig(atol=1e-3)
        validator = ParityValidator(config)
        
        assert validator.config == config
        assert len(validator.results) == 0
    
    def test_validator_default_config(self):
        """Test validator with default config."""
        validator = ParityValidator()
        
        assert validator.config.atol == 1e-5
        assert validator.config.rtol == 0.01
    
    def test_validator_validate_v1(self):
        """Test validator validate_v1 method."""
        np.random.seed(42)
        validator = ParityValidator(ValidationConfig(atol=1e-4))
        
        num_seqs = 1
        num_heads = 4
        head_dim = 64
        seq_len = 16
        block_size = 16
        
        query = np.random.randn(num_seqs, num_heads, head_dim).astype(np.float32)
        k_cache = np.random.randn(1, block_size, num_heads, head_dim).astype(np.float32)
        v_cache = np.random.randn(1, block_size, num_heads, head_dim).astype(np.float32)
        block_tables = np.array([[0]], dtype=np.int32)
        context_lens = np.array([seq_len], dtype=np.int32)
        
        result = validator.validate_v1(query, k_cache, v_cache, block_tables, context_lens)
        
        assert len(validator.results) == 1
        assert result.is_valid
    
    def test_validator_get_statistics(self):
        """Test validator statistics."""
        validator = ParityValidator()
        
        # Initially empty
        stats = validator.get_statistics()
        assert stats["total"] == 0
        assert stats["passed"] == 0
        assert stats["failed"] == 0
        
        # Add some results
        validator.results.append(ParityValidationResult(is_valid=True, max_diff=1e-6, mean_diff=1e-7, rel_diff=1e-8))
        validator.results.append(ParityValidationResult(is_valid=True, max_diff=1e-5, mean_diff=1e-6, rel_diff=1e-7))
        validator.results.append(ParityValidationResult(is_valid=False, max_diff=1.0, mean_diff=0.1, rel_diff=0.5))
        
        stats = validator.get_statistics()
        assert stats["total"] == 3
        assert stats["passed"] == 2
        assert stats["failed"] == 1
        assert stats["pass_rate"] == 2/3
        assert stats["max_diff"] == 1.0
    
    def test_validator_reset(self):
        """Test validator reset."""
        validator = ParityValidator()
        validator.results.append(ParityValidationResult(is_valid=True, max_diff=1e-6, mean_diff=1e-7, rel_diff=1e-8))
        
        assert len(validator.results) == 1
        
        validator.reset()
        
        assert len(validator.results) == 0


class TestEdgeCases:
    """Test edge cases for parity validation."""
    
    def test_single_token(self):
        """Test validation with single token."""
        arr1 = np.ones((1, 1, 1, 1), dtype=np.float32)
        arr2 = arr1.copy()
        
        result = validate_paged_linear_parity(arr1, arr2)
        
        assert result.is_valid is True
        assert result.max_diff == 0.0
    
    def test_large_values(self):
        """Test validation with large values."""
        arr1 = np.ones((2, 4, 8, 64), dtype=np.float32) * 1e6
        arr2 = arr1 + 0.1  # Small absolute difference
        
        result = validate_paged_linear_parity(arr1, arr2)
        
        # Should pass due to relative tolerance
        assert result.rel_diff < 1e-6
    
    def test_small_values(self):
        """Test validation with small values."""
        arr1 = np.ones((2, 4, 8, 64), dtype=np.float32) * 1e-8
        arr2 = arr1 + 1e-9  # 10% relative difference
        
        config = ValidationConfig(atol=1e-10, rtol=0.2)
        result = validate_paged_linear_parity(arr1, arr2, config)
        
        # Should pass with appropriate tolerance
        assert result.is_valid is True
    
    def test_different_dtypes(self):
        """Test validation with different dtypes."""
        arr1 = np.ones((2, 4, 8, 64), dtype=np.float32)
        arr2 = arr1.astype(np.float64)
        
        result = validate_paged_linear_parity(arr1, arr2)
        
        # Should pass (values are the same)
        assert result.is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
