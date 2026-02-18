"""Tests for enhanced Paged Attention vs Linear Attention parity validation.

This test suite uses the enhanced parity_validation module with dtype-aware
tolerances to handle FP16 precision differences.
"""

from __future__ import annotations

import numpy as np
import pytest

from metal_marlin.paged.parity_validation import (
    ParityConfig,
    ParityResult,
    run_paged_v1_parity_test,
    run_comprehensive_parity_suite,
    validate_parity,
)


class TestParityConfig:
    """Test parity configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ParityConfig()
        assert config.fp16_atol == 5e-3
        assert config.fp16_rtol == 5e-2
        assert config.fp32_atol == 1e-5
        assert config.fp32_rtol == 1e-2
    
    def test_fp16_tolerances(self):
        """Test FP16 tolerance selection."""
        config = ParityConfig()
        atol, rtol = config.get_tolerances(np.float16)
        assert atol == 5e-3
        assert rtol == 5e-2
    
    def test_fp32_tolerances(self):
        """Test FP32 tolerance selection."""
        config = ParityConfig()
        atol, rtol = config.get_tolerances(np.float32)
        assert atol == 1e-5
        assert rtol == 1e-2
    
    def test_custom_tolerances(self):
        """Test custom tolerance overrides."""
        config = ParityConfig(atol=1e-4, rtol=1e-3)
        atol, rtol = config.get_tolerances(np.float16)
        assert atol == 1e-4
        assert rtol == 1e-3


class TestValidateParity:
    """Test parity validation function."""
    
    def test_identical_arrays_pass(self):
        """Test identical arrays pass validation."""
        arr = np.random.randn(2, 4, 64).astype(np.float32)
        result = validate_parity(arr, arr)
        
        assert result.passed is True
        assert result.max_abs_diff == 0.0
        assert result.mean_abs_diff == 0.0
    
    def test_arrays_within_tolerance_pass(self):
        """Test arrays within tolerance pass."""
        arr1 = np.ones((2, 4, 64), dtype=np.float32)
        arr2 = arr1 + 1e-6
        
        result = validate_parity(arr1, arr2)
        
        assert result.passed is True
        assert result.max_abs_diff <= 2e-6
    
    def test_arrays_outside_tolerance_fail(self):
        """Test arrays outside tolerance fail."""
        arr1 = np.ones((2, 4, 64), dtype=np.float32)
        arr2 = arr1 + 1.0
        
        result = validate_parity(arr1, arr2)
        
        assert result.passed is False
        assert result.max_abs_diff == 1.0
    
    def test_nan_detection(self):
        """Test NaN detection."""
        arr1 = np.ones((2, 4, 64), dtype=np.float32)
        arr2 = arr1.copy()
        arr2[0, 0, 0] = np.nan
        
        result = validate_parity(arr1, arr2)
        
        assert result.passed is False
        assert "NaN detected" in result.message
    
    def test_inf_detection(self):
        """Test Inf detection."""
        arr1 = np.ones((2, 4, 64), dtype=np.float32)
        arr2 = arr1.copy()
        arr2[0, 0, 0] = np.inf
        
        result = validate_parity(arr1, arr2)
        
        assert result.passed is False
        assert "Inf detected" in result.message
    
    def test_fp16_precision_tolerance(self):
        """Test FP16 arrays with realistic precision differences."""
        # Simulate FP16 precision difference (~0.002)
        arr1 = np.random.randn(2, 4, 64).astype(np.float16)
        arr2 = arr1 + np.random.randn(2, 4, 64).astype(np.float16) * 0.001
        
        config = ParityConfig()
        result = validate_parity(arr1, arr2, config)
        
        # Should pass with FP16 tolerances
        assert result.passed is True or result.max_abs_diff < config.fp16_atol


class TestRunPagedV1ParityTest:
    """Test the paged v1 parity test runner."""
    
    def test_fp16_single_sequence(self):
        """Test FP16 single sequence parity."""
        result = run_paged_v1_parity_test(
            num_seqs=1,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            seq_len=16,
            dtype=np.float16,
            seed=42,
        )
        
        assert result.passed is True, f"Parity failed: {result.message}"
    
    def test_fp16_multi_sequence(self):
        """Test FP16 multi-sequence parity."""
        result = run_paged_v1_parity_test(
            num_seqs=2,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            seq_len=16,
            dtype=np.float16,
            seed=42,
        )
        
        assert result.passed is True, f"Parity failed: {result.message}"
    
    def test_fp32_single_sequence(self):
        """Test FP32 single sequence parity."""
        result = run_paged_v1_parity_test(
            num_seqs=1,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            seq_len=16,
            dtype=np.float32,
            seed=42,
        )
        
        assert result.passed is True, f"Parity failed: {result.message}"
    
    def test_fp32_multi_sequence(self):
        """Test FP32 multi-sequence parity."""
        result = run_paged_v1_parity_test(
            num_seqs=4,
            num_heads=8,
            num_kv_heads=8,
            head_dim=64,
            seq_len=32,
            dtype=np.float32,
            seed=42,
        )
        
        assert result.passed is True, f"Parity failed: {result.message}"
    
    def test_different_head_dims(self):
        """Test different head dimensions."""
        for head_dim in [32, 64, 96, 128]:
            result = run_paged_v1_parity_test(
                num_seqs=1,
                num_heads=4,
                num_kv_heads=4,
                head_dim=head_dim,
                seq_len=16,
                dtype=np.float32,
                seed=42,
            )
            
            assert result.passed is True, f"Parity failed for head_dim={head_dim}: {result.message}"
    
    def test_different_seq_lengths(self):
        """Test different sequence lengths."""
        for seq_len in [1, 8, 16, 32, 64]:
            result = run_paged_v1_parity_test(
                num_seqs=1,
                num_heads=4,
                num_kv_heads=4,
                head_dim=64,
                seq_len=seq_len,
                dtype=np.float32,
                seed=42,
            )
            
            assert result.passed is True, f"Parity failed for seq_len={seq_len}: {result.message}"
    
    def test_different_block_sizes(self):
        """Test different block sizes."""
        for block_size in [8, 16, 32]:
            result = run_paged_v1_parity_test(
                num_seqs=1,
                num_heads=4,
                num_kv_heads=4,
                head_dim=64,
                seq_len=32,
                block_size=block_size,
                dtype=np.float32,
                seed=42,
            )
            
            assert result.passed is True, f"Parity failed for block_size={block_size}: {result.message}"


class TestRunComprehensiveParitySuite:
    """Test the comprehensive parity test suite."""
    
    def test_default_suite_runs(self):
        """Test that default suite runs without errors."""
        results = run_comprehensive_parity_suite()
        
        assert "total" in results
        assert "passed" in results
        assert "failed" in results
        assert "results" in results
    
    def test_custom_configs(self):
        """Test with custom configurations."""
        configs = [
            {"num_seqs": 1, "num_heads": 4, "num_kv_heads": 4, "dtype": np.float32, "name": "test1"},
            {"num_seqs": 2, "num_heads": 8, "num_kv_heads": 8, "dtype": np.float32, "name": "test2"},
        ]
        
        results = run_comprehensive_parity_suite(configs)
        
        assert results["total"] == 2
        # Both should pass with FP32
        assert results["passed"] == 2
        assert results["failed"] == 0
    
    def test_suite_result_structure(self):
        """Test that suite results have correct structure."""
        configs = [
            {"num_seqs": 1, "num_heads": 4, "num_kv_heads": 4, "dtype": np.float32, "name": "test1"},
        ]
        
        results = run_comprehensive_parity_suite(configs)
        
        assert len(results["results"]) == 1
        result_entry = results["results"][0]
        assert "name" in result_entry
        assert "config" in result_entry
        assert "result" in result_entry
        assert isinstance(result_entry["result"], ParityResult)


class TestEdgeCases:
    """Test edge cases for parity validation."""
    
    def test_single_token_decode(self):
        """Test single token decode (seq_len=1)."""
        result = run_paged_v1_parity_test(
            num_seqs=1,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            seq_len=1,
            dtype=np.float32,
            seed=42,
        )
        
        assert result.passed is True, f"Single token parity failed: {result.message}"
    
    def test_single_head(self):
        """Test with single head."""
        result = run_paged_v1_parity_test(
            num_seqs=1,
            num_heads=1,
            num_kv_heads=1,
            head_dim=64,
            seq_len=16,
            dtype=np.float32,
            seed=42,
        )
        
        assert result.passed is True, f"Single head parity failed: {result.message}"
    
    def test_small_head_dim(self):
        """Test with small head dimension."""
        result = run_paged_v1_parity_test(
            num_seqs=1,
            num_heads=4,
            num_kv_heads=4,
            head_dim=16,
            seq_len=16,
            dtype=np.float32,
            seed=42,
        )
        
        assert result.passed is True, f"Small head_dim parity failed: {result.message}"
    
    def test_large_head_dim(self):
        """Test with large head dimension."""
        result = run_paged_v1_parity_test(
            num_seqs=1,
            num_heads=4,
            num_kv_heads=4,
            head_dim=256,
            seq_len=16,
            dtype=np.float32,
            seed=42,
        )
        
        assert result.passed is True, f"Large head_dim parity failed: {result.message}"


@pytest.mark.parametrize("num_seqs", [1, 2, 4])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_paged_v1_parity_parametrized(num_seqs, num_heads, head_dim, dtype):
    """Parametrized parity test covering multiple configurations."""
    result = run_paged_v1_parity_test(
        num_seqs=num_seqs,
        num_heads=num_heads,
        num_kv_heads=num_heads,  # No GQA for v1
        head_dim=head_dim,
        seq_len=16,
        dtype=dtype,
        seed=42,
    )
    
    assert result.passed is True, (
        f"Parity failed for num_seqs={num_seqs}, num_heads={num_heads}, "
        f"head_dim={head_dim}, dtype={dtype}: {result.message}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
