"""Tests for optimized KV cache eviction policy in PagedKVCache.

This test suite validates the production-ready eviction optimizations
for long-running servers, including weighted scoring, fragmentation
awareness, and adaptive policies.
"""

import time
import pytest
import numpy as np
from metal_marlin.paged.cache_manager_optimized import (
    PagedKVCacheOptimized,
    EvictionPolicy,
    EvictionConfig,
    EvictionStats,
    WeightedScore,
)
from metal_marlin.paged.kv_block import KVBlockConfig


class TestOptimizedKVCacheEviction:
    """Test optimized eviction policies for long-running servers."""

    def test_weighted_score_calculation(self):
        """Test that weighted scores consider recency, frequency, and size."""
        config = KVBlockConfig(block_size=16, num_heads=1, head_dim=64)
        eviction_config = EvictionConfig(
            policy=EvictionPolicy.WEIGHTED,
            recency_weight=1.0,
            frequency_weight=2.0,
            size_weight=0.5,
        )
        cache = PagedKVCacheOptimized(
            config=config,
            num_blocks=10,
            eviction_config=eviction_config,
        )

        # Add sequence 0: low frequency, small size
        assert cache.add_sequence(0)
        k = np.zeros((16, 1, 64), dtype=np.float16)
        v = np.zeros((16, 1, 64), dtype=np.float16)
        assert cache.append_kv_batch(0, k, v)

        # Add sequence 1: will have higher frequency
        assert cache.add_sequence(1)
        assert cache.append_kv_batch(1, k, v)

        # Access sequence 1 multiple times to increase frequency
        for _ in range(5):
            cache.get_kv(1)
            time.sleep(0.001)

        # Access sequence 0 once
        cache.get_kv(0)

        # Sequence 1 should have higher score (more valuable) due to frequency
        score0 = cache._compute_eviction_score(0)
        score1 = cache._compute_eviction_score(1)

        # Higher frequency = higher score = less likely to evict
        assert score1 > score0, "High frequency sequence should have higher score"

    def test_fragmentation_aware_eviction(self):
        """Test that eviction considers fragmentation impact."""
        config = KVBlockConfig(block_size=16, num_heads=1, head_dim=64)
        eviction_config = EvictionConfig(
            policy=EvictionPolicy.WEIGHTED,
            fragmentation_weight=1.0,  # High fragmentation awareness
        )
        cache = PagedKVCacheOptimized(
            config=config,
            num_blocks=8,
            eviction_config=eviction_config,
        )

        # Create fragmented layout:
        # Seq 0: blocks [0, 2] (fragmented)
        # Seq 1: block [1] (contiguous with gap)
        # Seq 2: blocks [4, 5, 6, 7] (large contiguous)

        assert cache.add_sequence(0)
        k = np.zeros((32, 1, 64), dtype=np.float16)  # 2 blocks
        v = np.zeros((32, 1, 64), dtype=np.float16)
        assert cache.append_kv_batch(0, k, v)

        # Manually fragment by adding/removing sequences
        assert cache.add_sequence(99)
        k_small = np.zeros((16, 1, 64), dtype=np.float16)
        v_small = np.zeros((16, 1, 64), dtype=np.float16)
        assert cache.append_kv_batch(99, k_small, v_small)

        # Add sequence 1 that will be between sequences
        assert cache.add_sequence(1)
        assert cache.append_kv_batch(1, k_small, v_small)

        # Remove seq 99 to create a gap
        cache.remove_sequence(99)

        # Add sequence 2 with more blocks
        assert cache.add_sequence(2)
        k_large = np.zeros((64, 1, 64), dtype=np.float16)  # 4 blocks
        v_large = np.zeros((64, 1, 64), dtype=np.float16)
        assert cache.append_kv_batch(2, k_large, v_large)

        # Force update of fragmentation info
        cache._update_fragmentation_info()

        # Check that fragmentation info is tracked
        frag_info = cache._fragmentation_info
        assert frag_info is not None
        assert frag_info['fragmentation_ratio'] >= 0.0

    def test_adaptive_policy_switching(self):
        """Test that adaptive policy switches based on workload patterns."""
        config = KVBlockConfig(block_size=16, num_heads=1, head_dim=64)
        eviction_config = EvictionConfig(
            policy=EvictionPolicy.ADAPTIVE,
            window_size=5,  # Small window for testing
        )
        cache = PagedKVCacheOptimized(
            config=config,
            num_blocks=10,
            eviction_config=eviction_config,
        )

        # Simulate scan-heavy workload (repeated sequential access)
        for i in range(5):
            assert cache.add_sequence(i)
            k = np.zeros((16, 1, 64), dtype=np.float16)
            v = np.zeros((16, 1, 64), dtype=np.float16)
            assert cache.append_kv_batch(i, k, v)

        # Access sequences in scan pattern (0, 1, 2, 3, 4, 0, 1, 2...)
        for _ in range(3):
            for seq_id in range(5):
                cache.get_kv(seq_id)

        # Check that workload is detected
        pattern = cache._detect_workload_pattern()
        assert pattern is not None

        # Verify adaptive policy is being used
        assert cache.eviction_config.policy == EvictionPolicy.ADAPTIVE

    def test_ttl_based_eviction(self):
        """Test TTL-based eviction for time-sensitive sequences."""
        config = KVBlockConfig(block_size=16, num_heads=1, head_dim=64)
        eviction_config = EvictionConfig(
            policy=EvictionPolicy.WEIGHTED,
            enable_ttl=True,
            default_ttl_seconds=0.1,  # Short TTL for testing
        )
        cache = PagedKVCacheOptimized(
            config=config,
            num_blocks=10,
            eviction_config=eviction_config,
        )

        # Add sequence with short TTL
        assert cache.add_sequence(0)
        k = np.zeros((16, 1, 64), dtype=np.float16)
        v = np.zeros((16, 1, 64), dtype=np.float16)
        assert cache.append_kv_batch(0, k, v)

        # Verify sequence exists
        assert cache.has_sequence(0)

        # Wait for TTL to expire
        time.sleep(0.15)

        # Run cleanup
        expired = cache.cleanup_expired()
        assert expired >= 1, "Expired sequence should be cleaned up"
        assert not cache.has_sequence(0), "Expired sequence should be removed"

    def test_priority_based_eviction(self):
        """Test that high-priority sequences are protected."""
        config = KVBlockConfig(block_size=16, num_heads=1, head_dim=64)
        eviction_config = EvictionConfig(
            policy=EvictionPolicy.WEIGHTED,
        )
        cache = PagedKVCacheOptimized(
            config=config,
            num_blocks=4,
            eviction_config=eviction_config,
        )

        # Add low priority sequence
        assert cache.add_sequence(0, priority=0)
        k = np.zeros((16, 1, 64), dtype=np.float16)
        v = np.zeros((16, 1, 64), dtype=np.float16)
        assert cache.append_kv_batch(0, k, v)

        # Add high priority sequence
        assert cache.add_sequence(1, priority=10)
        assert cache.append_kv_batch(1, k, v)

        # Add more sequences to trigger eviction
        assert cache.add_sequence(2)
        assert cache.append_kv_batch(2, k, v)

        assert cache.add_sequence(3)
        assert cache.append_kv_batch(3, k, v)

        # Try to add another sequence (should trigger eviction)
        assert cache.add_sequence(4)
        assert cache.append_kv_batch(4, k, v)

        # High priority sequence should still exist
        assert cache.has_sequence(1), "High priority sequence should survive eviction"

    def test_batch_eviction(self):
        """Test batch eviction for efficiency."""
        config = KVBlockConfig(block_size=16, num_heads=1, head_dim=64)
        eviction_config = EvictionConfig(
            policy=EvictionPolicy.WEIGHTED,
            batch_eviction=True,
            batch_size=1,
        )
        cache = PagedKVCacheOptimized(
            config=config,
            num_blocks=9,
            eviction_config=eviction_config,
        )

        # Fill cache with multiple sequences
        for i in range(6):
            assert cache.add_sequence(i)
            k = np.zeros((16, 1, 64), dtype=np.float16)
            v = np.zeros((16, 1, 64), dtype=np.float16)
            assert cache.append_kv_batch(i, k, v)
            time.sleep(0.01)  # Ensure time differences

        # Access seq 0 and 1 to make them recent
        cache.get_kv(0)
        cache.get_kv(1)

        # Try to add a large sequence requiring multiple evictions
        assert cache.add_sequence(100)
        k_large = np.zeros((64, 1, 64), dtype=np.float16)  # 4 blocks
        v_large = np.zeros((64, 1, 64), dtype=np.float16)
        assert cache.append_kv_batch(100, k_large, v_large)

        # Verify recent sequences still exist
        assert cache.has_sequence(0)
        assert cache.has_sequence(1)

        # Verify eviction stats were updated
        stats = cache.get_eviction_stats()
        assert stats.num_evictions >= 1

    def test_eviction_stats_tracking(self):
        """Test that eviction statistics are accurately tracked."""
        config = KVBlockConfig(block_size=16, num_heads=1, head_dim=64)
        eviction_config = EvictionConfig(
            policy=EvictionPolicy.LRU,
            track_stats=True,
        )
        cache = PagedKVCacheOptimized(
            config=config,
            num_blocks=4,
            eviction_config=eviction_config,
        )

        # Add and evict sequences
        for i in range(6):
            assert cache.add_sequence(i)
            k = np.zeros((16, 1, 64), dtype=np.float16)
            v = np.zeros((16, 1, 64), dtype=np.float16)
            assert cache.append_kv_batch(i, k, v)

        # Get stats
        stats = cache.get_eviction_stats()
        assert isinstance(stats, EvictionStats)
        assert stats.num_evictions > 0
        assert stats.sequences_evicted > 0
        assert stats.blocks_evicted > 0

        # Check detailed stats if enabled
        if eviction_config.track_detailed_stats:
            detailed = cache.get_detailed_stats()
            assert 'avg_score_evicted' in detailed
            assert 'eviction_reasons' in detailed

    def test_memory_pressure_eviction(self):
        """Test proactive eviction under memory pressure."""
        config = KVBlockConfig(block_size=16, num_heads=1, head_dim=64)
        eviction_config = EvictionConfig(
            policy=EvictionPolicy.WEIGHTED,
            memory_pressure_threshold=0.7,
            proactive_eviction=True,
        )
        cache = PagedKVCacheOptimized(
            config=config,
            num_blocks=10,
            eviction_config=eviction_config,
        )

        # Fill to near capacity
        for i in range(5):
            assert cache.add_sequence(i)
            k = np.zeros((16, 1, 64), dtype=np.float16)
            v = np.zeros((16, 1, 64), dtype=np.float16)
            assert cache.append_kv_batch(i, k, v)

        # Check memory pressure
        pressure = cache.get_memory_pressure()
        assert 0.0 <= pressure <= 1.0

        # Proactive eviction should trigger if over threshold
        if pressure > eviction_config.memory_pressure_threshold:
            freed = cache.do_proactive_eviction(target_pressure=0.5)
            assert freed > 0, "Proactive eviction should free blocks"

    def test_concurrent_access_safety(self):
        """Test thread safety of eviction operations."""
        import threading

        config = KVBlockConfig(block_size=16, num_heads=1, head_dim=64)
        eviction_config = EvictionConfig(
            policy=EvictionPolicy.WEIGHTED,
        )
        cache = PagedKVCacheOptimized(
            config=config,
            num_blocks=20,
            eviction_config=eviction_config,
        )

        errors = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(10):
                    seq_id = thread_id * 100 + i
                    if cache.add_sequence(seq_id):
                        k = np.zeros((16, 1, 64), dtype=np.float16)
                        v = np.zeros((16, 1, 64), dtype=np.float16)
                        try:
                            cache.append_kv_batch(seq_id, k, v)
                        except ValueError:
                            pass  # May have been evicted before append

                    # Randomly access sequences
                    if i > 0:
                        access_id = thread_id * 100 + (i % 5)
                        try:
                            cache.get_kv(access_id)
                        except ValueError:
                            pass  # May have been evicted
            except Exception as e:
                errors.append(e)

        # Run concurrent threads
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"

    def test_lru_compatibility(self):
        """Test that optimized cache maintains LRU behavior as baseline."""
        config = KVBlockConfig(block_size=16, num_heads=1, head_dim=64)
        eviction_config = EvictionConfig(
            policy=EvictionPolicy.LRU,
        )
        cache = PagedKVCacheOptimized(
            config=config,
            num_blocks=4,
            eviction_config=eviction_config,
        )

        # Add sequence 0 and 1
        assert cache.add_sequence(0)
        k = np.zeros((16, 1, 64), dtype=np.float16)
        v = np.zeros((16, 1, 64), dtype=np.float16)
        assert cache.append_kv_batch(0, k, v)

        assert cache.add_sequence(1)
        assert cache.append_kv_batch(1, k, v)

        time.sleep(0.01)

        # Access seq 0 to make it MRU
        cache.get_kv(0)
        time.sleep(0.01)

        # Access seq 1 to make it MRU (seq 0 becomes LRU)
        cache.get_kv(1)
        time.sleep(0.01)

        # Add seq 2 and 3
        assert cache.add_sequence(2)
        assert cache.append_kv_batch(2, k, v)

        assert cache.add_sequence(3)
        assert cache.append_kv_batch(3, k, v)

        # Add seq 4, should evict LRU (seq 0)
        assert cache.add_sequence(4)
        assert cache.append_kv_batch(4, k, v)

        # Seq 0 should be evicted, seq 1 should remain
        assert not cache.has_sequence(0), "LRU sequence should be evicted"
        assert cache.has_sequence(1), "MRU sequence should remain"


class TestEvictionConfig:
    """Test EvictionConfig dataclass."""

    def test_default_config(self):
        """Test default eviction configuration."""
        config = EvictionConfig()
        assert config.policy == EvictionPolicy.WEIGHTED
        assert config.recency_weight == 1.0
        assert config.frequency_weight == 1.0
        assert config.size_weight == 0.5

    def test_config_validation(self):
        """Test that invalid config values are rejected."""
        with pytest.raises(ValueError):
            EvictionConfig(recency_weight=-1.0)

        with pytest.raises(ValueError):
            EvictionConfig(memory_pressure_threshold=1.5)


class TestWeightedScore:
    """Test WeightedScore helper class."""

    def test_score_calculation(self):
        """Test weighted score calculation."""
        score = WeightedScore.calculate(
            recency=1.0,
            frequency=5.0,
            size=10.0,
            fragmentation=0.2,
            config=EvictionConfig(
                recency_weight=1.0,
                frequency_weight=2.0,
                size_weight=0.5,
                fragmentation_weight=1.0
            )
        )
        # Expected: 1.0*1.0 + 5.0*2.0 + 10.0*0.5 - 0.2*1.0 = 1 + 10 + 5 - 0.2 = 15.8
        expected = 1.0 + 10.0 + 5.0 - 0.2
        assert abs(score - expected) < 0.001

    def test_score_normalization(self):
        """Test score normalization for fair comparison."""
        scores = [10.0, 20.0, 30.0]
        normalized = WeightedScore.normalize_scores(scores)
        assert len(normalized) == len(scores)
        assert normalized[0] <= normalized[1] <= normalized[2]
