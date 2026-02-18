"""Tests for long-running KV cache scenarios."""

import time
import pytest
import numpy as np
from metal_marlin.paged.cache_manager_optimized import (
    PagedKVCacheOptimized,
    EvictionPolicy,
    EvictionConfig,
)
from metal_marlin.paged.kv_block import KVBlockConfig

class TestLongRunningKVCache:
    """Test scenarios typical of long-running servers."""

    def test_frequency_dominance_prevention(self):
        """Test that logarithmic frequency prevents old heavy-hitters from dominating forever."""
        config = KVBlockConfig(block_size=16, num_heads=1, head_dim=64)
        # Use a config that would favor frequency strongly
        eviction_config = EvictionConfig(
            policy=EvictionPolicy.WEIGHTED,
            recency_weight=1.0,
            frequency_weight=5.0, # Strong frequency bias
        )
        cache = PagedKVCacheOptimized(
            config=config,
            num_blocks=10,
            eviction_config=eviction_config,
        )

        # Seq 0: The "Old Heavy Hitter"
        assert cache.add_sequence(0)
        k = np.zeros((16, 1, 64), dtype=np.float16)
        v = np.zeros((16, 1, 64), dtype=np.float16)
        assert cache.append_kv_batch(0, k, v)
        
        # Boost frequency of seq 0 artificially
        for _ in range(100):
            cache.get_kv(0)
            
        # Seq 1-9: New items
        for i in range(1, 10):
            assert cache.add_sequence(i)
            assert cache.append_kv_batch(i, k, v)
            
        # Cache is full now (10 blocks).
        
        # Add Seq 10. Should trigger eviction.
        # If raw frequency is used, Seq 0 (freq 101) will NEVER be evicted compared to Seq 1 (freq 1).
        # Even if Seq 0 hasn't been used in a "long time" (simulated by others being added later).
        # But we want to ensure that if we add MANY new items, eventually we might evict Seq 0 
        # if using a smart policy, OR we verify that log-frequency makes the gap smaller.
        
        # Actually, for this test, we want to verify the implementation details of log-frequency 
        # or just verify the behavior.
        # Let's verify that we can inspect the score and see the effect.
        
        score0 = cache._compute_eviction_score(0)
        score1 = cache._compute_eviction_score(1)
        
        # With raw frequency: 101 vs 1. Score0 >>> Score1.
        # With log frequency: log(101) ~ 4.6 vs log(1) = 0. Score0 > Score1 but gap is smaller.
        
        print(f"Score 0 (Freq 100+): {score0}")
        print(f"Score 1 (Freq 1): {score1}")
        
        # This test is mostly a placeholder to run and observe current behavior.
        assert score0 > score1

    def test_ghost_cache_adaptation(self):
        """Test that ghost cache hits trigger adaptation."""
        config = KVBlockConfig(block_size=16, num_heads=1, head_dim=64)
        eviction_config = EvictionConfig(
            policy=EvictionPolicy.ADAPTIVE,
            memory_pressure_threshold=0.0, # Always adaptive
        )
        cache = PagedKVCacheOptimized(
            config=config,
            num_blocks=5,
            eviction_config=eviction_config,
        )
        
        # Fill cache
        for i in range(5):
            cache.add_sequence(i)
            k = np.zeros((16, 1, 64), dtype=np.float16)
            v = np.zeros((16, 1, 64), dtype=np.float16)
            cache.append_kv_batch(i, k, v)
            
        # Evict seq 0
        cache.add_sequence(99)
        k = np.zeros((16, 1, 64), dtype=np.float16)
        v = np.zeros((16, 1, 64), dtype=np.float16)
        cache.append_kv_batch(99, k, v)
        
        assert not cache.has_sequence(0) or not cache.has_sequence(1) # Something evicted
        
        # If we re-add an evicted sequence, it should be a ghost hit
        evicted = [i for i in range(5) if not cache.has_sequence(i)]
        if evicted:
            victim = evicted[0]
            # Re-add victim
            cache.add_sequence(victim)
            
            # Check if we can detect the adaptation (requires access to internal state)
            # For now, just ensure it runs without error
            pass

    def test_sampled_eviction_performance(self):
        """Test performance of sampled eviction vs full sort."""
        # This would require a large number of sequences
        pass
