"""Tests for KV cache pooling optimization.

This module tests the KV cache buffer pool implementation that enables
efficient buffer reuse across sequence boundaries, avoiding expensive
GPU memory allocations.

Tests cover:
- Pool hit/miss behavior during reset/reallocation
- Buffer reuse in KVCacheTorch
- Buffer reuse in MLAKVCache  
- Pool metrics tracking
- Pool clearing and reset
"""

import pytest

from metal_marlin._compat import HAS_TORCH, torch
from metal_marlin.kv_cache import (
    KVCacheTorch,
    CacheConfigTorch,
    MLAKVCache,
    clear_pool,
    get_pool_stats,
    reset_pool_metrics,
)

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
requires_mps = pytest.mark.skipif(
    not HAS_TORCH or not torch.backends.mps.is_available(),
    reason="Requires MPS backend"
)


# =============================================================================
# Pool Utility Tests
# =============================================================================


class TestPoolUtilities:
    """Test pool utility functions."""

    def test_clear_pool(self):
        """Test that clear_pool resets the pool."""
        clear_pool()
        stats = get_pool_stats()
        assert stats["pooled_tensors"] == 0

    def test_reset_pool_metrics(self):
        """Test that reset_pool_metrics clears counters."""
        reset_pool_metrics()
        stats = get_pool_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["returns"] == 0
        assert stats["evictions"] == 0

    def test_get_pool_stats_structure(self):
        """Test that get_pool_stats returns expected keys."""
        clear_pool()
        reset_pool_metrics()
        stats = get_pool_stats()
        
        expected_keys = ["hits", "misses", "returns", "evictions", "pooled_tensors", "hit_rate"]
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))


# =============================================================================
# KVCacheTorch Pooling Tests
# =============================================================================


@pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
class TestKVCacheTorchPooling:
    """Test KVCacheTorch buffer pooling."""

    def setup_method(self):
        """Reset pool before each test."""
        clear_pool()
        reset_pool_metrics()

    def test_cache_creation_allocates_buffers(self):
        """Test that cache creation allocates new buffers (pool miss)."""
        config = CacheConfigTorch(
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=128,
        )
        
        # Create cache
        cache = KVCacheTorch(config, batch_size=1, device="cpu")
        
        # Should have pool misses (new allocation)
        stats = get_pool_stats()
        assert stats["misses"] >= 2  # k_cache and v_cache
        assert len(cache.k_cache) == 2
        assert len(cache.v_cache) == 2

    def test_reset_uses_pool_for_reallocation(self):
        """Test that reset uses pool for immediate reallocation.
        
        With the pooling optimization, reset() returns tensors to the pool
        and immediately reallocates from the pool. This should result in
        pool hits for the reallocation.
        """
        config = CacheConfigTorch(
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=128,
        )
        
        cache = KVCacheTorch(config, batch_size=1, device="cpu")
        
        # Reset metrics after creation
        reset_pool_metrics()
        
        # Reset should return buffers to pool and reallocate immediately
        cache.reset()
        
        stats = get_pool_stats()
        # Should have returns (old buffers) and hits (new allocations from pool)
        assert stats["returns"] >= 2
        # Reallocation uses pooled buffers
        assert stats["hits"] >= 2

    def test_reset_maintains_pool_efficiency(self):
        """Test that reset maintains cache usability while using pool.
        
        After reset, the cache should still be fully functional with
        new tensors allocated from the pool.
        """
        config = CacheConfigTorch(
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=128,
        )
        
        cache = KVCacheTorch(config, batch_size=1, device="cpu")
        
        # Use the cache
        k = torch.randn(1, 4, 10, 64, device="cpu")
        v = torch.randn(1, 4, 10, 64, device="cpu")
        cache.update(0, k, v)
        cache.advance(10)
        
        # Reset metrics
        reset_pool_metrics()
        
        # Reset - should use pool
        cache.reset()
        
        stats = get_pool_stats()
        assert stats["hits"] >= 2  # Reallocation from pool
        assert stats["returns"] >= 2  # Old buffers returned
        
        # Cache should still work
        k2 = torch.randn(1, 4, 5, 64, device="cpu")
        v2 = torch.randn(1, 4, 5, 64, device="cpu")
        k_full, v_full = cache.update(0, k2, v2)
        
        assert k_full is not None
        assert v_full is not None

    def test_quantized_cache_pooling(self):
        """Test that quantized caches also use pooling."""
        config = CacheConfigTorch(
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=128,
            cache_dtype="fp8",
        )
        
        cache = KVCacheTorch(config, batch_size=1, device="cpu")
        reset_pool_metrics()
        
        cache.reset()
        
        stats = get_pool_stats()
        # Should return and reallocate k_cache, v_cache, k_scales, v_scales
        assert stats["returns"] >= 4
        assert stats["hits"] >= 4


# =============================================================================
# MLAKVCache Pooling Tests
# =============================================================================


@pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
class TestMLAKVCachePooling:
    """Test MLAKVCache buffer pooling."""

    def setup_method(self):
        """Reset pool before each test."""
        clear_pool()
        reset_pool_metrics()

    def test_mla_cache_creation_uses_pool(self):
        """Test that MLAKVCache creation uses the pool."""
        cache = MLAKVCache(
            num_layers=2,
            batch_size=1,
            max_seq_len=128,
            kv_lora_rank=512,
            device="cpu",
            quantize_mode="none",
        )
        
        # Should have allocated via pool
        stats = get_pool_stats()
        # Pool miss because pool was empty
        assert stats["misses"] >= 1

    def test_mla_reset_uses_pool(self):
        """Test that MLAKVCache reset uses pool for reallocation."""
        cache = MLAKVCache(
            num_layers=2,
            batch_size=1,
            max_seq_len=128,
            kv_lora_rank=512,
            device="cpu",
            quantize_mode="none",
        )
        
        reset_pool_metrics()
        
        cache.reset()
        
        stats = get_pool_stats()
        # Should return buffers and reallocate from pool
        assert stats["returns"] >= 1
        assert stats["hits"] >= 1

    def test_mla_reset_maintains_usability(self):
        """Test that MLAKVCache remains usable after reset."""
        cache = MLAKVCache(
            num_layers=2,
            batch_size=1,
            max_seq_len=128,
            kv_lora_rank=512,
            device="cpu",
            quantize_mode="none",
        )
        
        # Use the cache
        compressed = torch.randn(1, 10, cache.cache_dim, device="cpu")
        cache.update(0, compressed_kv=compressed)
        
        reset_pool_metrics()
        
        # Reset - should use pool
        cache.reset()
        
        stats = get_pool_stats()
        assert stats["hits"] >= 1
        
        # Cache should still work
        compressed2 = torch.randn(1, 5, cache.cache_dim, device="cpu")
        result = cache.update(0, compressed_kv=compressed2)
        
        assert result is not None

    def test_mla_quantized_pooling(self):
        """Test MLAKVCache pooling with quantization."""
        cache = MLAKVCache(
            num_layers=2,
            batch_size=1,
            max_seq_len=128,
            kv_lora_rank=512,
            device="cpu",
            quantize_mode="fp8",
        )
        
        reset_pool_metrics()
        
        cache.reset()
        
        stats = get_pool_stats()
        # Should return and reallocate kv_cache and kv_scales
        assert stats["returns"] >= 2
        assert stats["hits"] >= 2


# =============================================================================
# Pool Metrics Tests
# =============================================================================


class TestPoolMetrics:
    """Test pool metrics tracking."""

    def setup_method(self):
        """Reset pool before each test."""
        clear_pool()
        reset_pool_metrics()

    @pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
    def test_hit_rate_during_reset(self):
        """Test that hit rate reflects pool usage during reset."""
        config = CacheConfigTorch(
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=128,
        )
        
        # First creation: all misses
        cache = KVCacheTorch(config, batch_size=1, device="cpu")
        
        reset_pool_metrics()
        
        # Reset: returns and reallocates (hits)
        cache.reset()
        
        stats = get_pool_stats()
        # Hit rate should be 1.0 for reallocation (all from pool)
        assert stats["hits"] > 0
        # Either we have perfect hit rate or only hits
        assert stats["hit_rate"] == 1.0 or stats["misses"] == 0

    def test_empty_pool_stats(self):
        """Test stats when pool is empty."""
        clear_pool()
        reset_pool_metrics()
        
        stats = get_pool_stats()
        assert stats["pooled_tensors"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["returns"] == 0
        assert stats["hit_rate"] == 0.0


# =============================================================================
# End-to-End Pooling Integration Tests
# =============================================================================


@pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
class TestPoolingIntegration:
    """End-to-end tests for KV cache pooling."""

    def setup_method(self):
        """Reset pool before each test."""
        clear_pool()
        reset_pool_metrics()

    def test_multiple_reset_cycles(self):
        """Test multiple reset cycles maintain pool efficiency."""
        config = CacheConfigTorch(
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=128,
        )
        
        cache = KVCacheTorch(config, batch_size=1, device="cpu")
        
        # Multiple reset cycles
        reset_pool_metrics()
        for _ in range(3):
            cache.reset()
        
        stats = get_pool_stats()
        # Should have hits from reusing pooled buffers each cycle
        assert stats["hits"] > 0

    def test_cache_remains_usable_after_reset(self):
        """Test that cache can be used after reset."""
        config = CacheConfigTorch(
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=128,
        )
        
        cache = KVCacheTorch(config, batch_size=1, device="cpu")
        
        # Use the cache
        k = torch.randn(1, 4, 10, 64, device="cpu")
        v = torch.randn(1, 4, 10, 64, device="cpu")
        cache.update(0, k, v)
        cache.advance(10)
        
        # Reset
        cache.reset()
        
        # Should be usable again
        k2 = torch.randn(1, 4, 5, 64, device="cpu")
        v2 = torch.randn(1, 4, 5, 64, device="cpu")
        k_full, v_full = cache.update(0, k2, v2)
        
        assert k_full is not None
        assert v_full is not None
        assert k_full.shape[2] == 5  # Only new tokens

    def test_mixed_quantization_modes(self):
        """Test that different quantization modes don't interfere."""
        config_fp16 = CacheConfigTorch(
            num_layers=1,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=64,
            cache_dtype="fp16",
        )
        config_fp8 = CacheConfigTorch(
            num_layers=1,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=64,
            cache_dtype="fp8",
        )
        
        # Create both types
        cache_fp16 = KVCacheTorch(config_fp16, batch_size=1, device="cpu")
        cache_fp8 = KVCacheTorch(config_fp8, batch_size=1, device="cpu")
        
        # Both should work after reset
        reset_pool_metrics()
        
        cache_fp16.reset()
        cache_fp8.reset()
        
        stats = get_pool_stats()
        # Both should have used pool for reallocation
        assert stats["hits"] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
