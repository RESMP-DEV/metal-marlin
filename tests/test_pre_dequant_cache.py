"""Tests for KV cache pre-dequantization optimization.

This module tests the pre-dequantization cache that avoids redundant
dequantization operations on repeated access to the same K/V values.

Tests cover:
- Pre-dequantization cache hit behavior
- Cache invalidation on update
- Cache validity across multiple get_kv calls
- All quantization modes (FP4, FP8, INT8)
"""

import pytest

from metal_marlin._compat import HAS_TORCH, torch
from metal_marlin.kv_cache import (
    KVCacheTorch,
    CacheConfigTorch,
    clear_pool,
    reset_pool_metrics,
)

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
requires_mps = pytest.mark.skipif(
    not HAS_TORCH or not torch.backends.mps.is_available(),
    reason="Requires MPS backend"
)


@pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
class TestPreDequantCache:
    """Test pre-dequantization cache behavior."""

    def setup_method(self):
        """Reset pool before each test."""
        clear_pool()
        reset_pool_metrics()

    def test_pre_dequant_cache_fp8(self):
        """Test pre-dequantization cache with FP8 quantization."""
        config = CacheConfigTorch(
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=128,
            cache_dtype="fp8",
        )
        
        cache = KVCacheTorch(config, batch_size=1, device="cpu")
        
        # Update cache with some tokens
        k = torch.randn(1, 4, 10, 64, device="cpu")
        v = torch.randn(1, 4, 10, 64, device="cpu")
        cache.update(0, k, v)
        cache.advance(10)
        
        # First get_kv should populate the cache
        k1, v1 = cache.get_kv(0)
        assert k1 is not None
        assert v1 is not None
        assert 0 in cache._pre_dequant_cache
        
        # Second get_kv should return cached values (same objects)
        k2, v2 = cache.get_kv(0)
        assert k1 is k2  # Should be same tensor (cached)
        assert v1 is v2

    def test_pre_dequant_cache_int8(self):
        """Test pre-dequantization cache with INT8 quantization."""
        config = CacheConfigTorch(
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=128,
            cache_dtype="int8",
        )
        
        cache = KVCacheTorch(config, batch_size=1, device="cpu")
        
        # Update cache with some tokens
        k = torch.randn(1, 4, 10, 64, device="cpu")
        v = torch.randn(1, 4, 10, 64, device="cpu")
        cache.update(0, k, v)
        cache.advance(10)
        
        # First get_kv should populate the cache
        k1, v1 = cache.get_kv(0)
        assert k1 is not None
        assert v1 is not None
        assert 0 in cache._pre_dequant_cache
        
        # Second get_kv should return cached values
        k2, v2 = cache.get_kv(0)
        assert k1 is k2
        assert v1 is v2

    def test_pre_dequant_cache_fp4(self):
        """Test pre-dequantization cache with FP4 quantization."""
        config = CacheConfigTorch(
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=128,
            cache_dtype="fp4",
        )
        
        cache = KVCacheTorch(config, batch_size=1, device="cpu")
        
        # Update cache with some tokens
        k = torch.randn(1, 4, 10, 64, device="cpu")
        v = torch.randn(1, 4, 10, 64, device="cpu")
        cache.update(0, k, v)
        cache.advance(10)
        
        # First get_kv should populate the cache
        k1, v1 = cache.get_kv(0)
        assert k1 is not None
        assert v1 is not None
        assert 0 in cache._pre_dequant_cache
        
        # Second get_kv should return cached values
        k2, v2 = cache.get_kv(0)
        assert k1 is k2
        assert v1 is v2

    def test_pre_dequant_cache_invalidation_on_update(self):
        """Test that cache is invalidated when underlying data changes."""
        config = CacheConfigTorch(
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=128,
            cache_dtype="fp8",
        )
        
        cache = KVCacheTorch(config, batch_size=1, device="cpu")
        
        # First update
        k1 = torch.randn(1, 4, 10, 64, device="cpu")
        v1 = torch.randn(1, 4, 10, 64, device="cpu")
        cache.update(0, k1, v1)
        cache.advance(10)
        
        # Populate cache
        k_cached1, v_cached1 = cache.get_kv(0)
        assert 0 in cache._pre_dequant_cache
        
        # Second update to same layer should invalidate cache
        k2 = torch.randn(1, 4, 5, 64, device="cpu")
        v2 = torch.randn(1, 4, 5, 64, device="cpu")
        cache.update(0, k2, v2)
        cache.advance(5)
        
        # Cache for layer 0 should be updated with new length
        assert 0 in cache._pre_dequant_cache
        cached_k, cached_v = cache._pre_dequant_cache[0]
        assert cached_k.shape[2] == 15  # 10 + 5 tokens
        
        # get_kv should return new values
        k_cached2, v_cached2 = cache.get_kv(0)
        assert k_cached2.shape[2] == 15  # 10 + 5 tokens

    def test_pre_dequant_cache_per_layer(self):
        """Test that cache is maintained per-layer."""
        config = CacheConfigTorch(
            num_layers=3,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=128,
            cache_dtype="fp8",
        )
        
        cache = KVCacheTorch(config, batch_size=1, device="cpu")
        
        # Update layer 0
        k = torch.randn(1, 4, 10, 64, device="cpu")
        v = torch.randn(1, 4, 10, 64, device="cpu")
        cache.update(0, k, v)
        cache.advance(10)
        
        # Populate cache for layer 0
        cache.get_kv(0)
        assert 0 in cache._pre_dequant_cache
        assert 1 not in cache._pre_dequant_cache
        
        # Update layer 1
        cache.update(1, k, v)
        
        # Populate cache for layer 1
        cache.get_kv(1)
        assert 0 in cache._pre_dequant_cache
        assert 1 in cache._pre_dequant_cache

    def test_pre_dequant_cache_cleared_on_reset(self):
        """Test that cache is cleared on reset."""
        config = CacheConfigTorch(
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=128,
            cache_dtype="fp8",
        )
        
        cache = KVCacheTorch(config, batch_size=1, device="cpu")
        
        # Update and populate cache
        k = torch.randn(1, 4, 10, 64, device="cpu")
        v = torch.randn(1, 4, 10, 64, device="cpu")
        cache.update(0, k, v)
        cache.advance(10)
        cache.get_kv(0)
        
        assert 0 in cache._pre_dequant_cache
        
        # Reset should clear cache
        cache.reset()
        assert len(cache._pre_dequant_cache) == 0

    def test_pre_dequant_cache_update_returns_cached(self):
        """Test that update() also uses and returns cached dequantized values."""
        config = CacheConfigTorch(
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=128,
            cache_dtype="fp8",
        )
        
        cache = KVCacheTorch(config, batch_size=1, device="cpu")
        
        # First update - should not have cache yet
        k1 = torch.randn(1, 4, 10, 64, device="cpu")
        v1 = torch.randn(1, 4, 10, 64, device="cpu")
        k_full1, v_full1 = cache.update(0, k1, v1)
        cache.advance(10)
        
        # After update, cache should be populated
        assert 0 in cache._pre_dequant_cache
        
        # get_kv should return the same cached values
        k_cached, v_cached = cache.get_kv(0)
        assert k_full1 is k_cached
        assert v_full1 is v_cached

    def test_pre_dequant_cache_fp4_update_uses_cache(self):
        """Test that FP4 update() also populates and uses the cache."""
        config = CacheConfigTorch(
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=128,
            cache_dtype="fp4",
        )
        
        cache = KVCacheTorch(config, batch_size=1, device="cpu")
        
        # Update should populate cache
        k = torch.randn(1, 4, 10, 64, device="cpu")
        v = torch.randn(1, 4, 10, 64, device="cpu")
        k_full, v_full = cache.update(0, k, v)
        cache.advance(10)
        
        # Cache should be populated
        assert 0 in cache._pre_dequant_cache
        
        # get_kv should return cached values
        k_cached, v_cached = cache.get_kv(0)
        assert k_full is k_cached
        assert v_full is v_cached


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
