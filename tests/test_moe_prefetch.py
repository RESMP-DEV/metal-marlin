"""Tests for MoE predictive expert prefetching."""

from __future__ import annotations

import time
from unittest.mock import Mock

import mlx.core as mx
import numpy as np
import pytest

from metal_marlin.moe.prefetch import (
    AsyncExpertLoader,
    ExpertLRUCache,
    ExpertPrefetcher,
    PrefetchConfig,
    PrefetchStats,
    PrefetchStrategy,
    RoutingHistory,
    async_load_experts,
    predict_next_experts,
)


class TestRoutingHistory:
    """Tests for RoutingHistory tracking."""

    def test_record_and_retrieve(self):
        """Test basic recording and retrieval."""
        history = RoutingHistory(layer_idx=0, window_size=10)

        history.record([3, 7])
        history.record([2, 5])
        history.record([3, 8])

        assert len(history.history) == 3
        assert history.get_last_experts() == [3, 8]

    def test_window_trimming(self):
        """Test that history is trimmed to window size."""
        history = RoutingHistory(layer_idx=0, window_size=3)

        for i in range(10):
            history.record([i])

        assert len(history.history) == 3
        # Should contain last 3 entries: [7], [8], [9]
        assert history.get_last_experts() == [9]

    def test_frequency_counts_no_decay(self):
        """Test frequency counting without decay."""
        history = RoutingHistory(layer_idx=0, window_size=10)

        history.record([1, 2])
        history.record([1, 3])
        history.record([1, 2])

        counts = history.get_frequency_counts(decay=1.0)

        assert counts[1] == 3  # Appears in all 3
        assert counts[2] == 2  # Appears in 2
        assert counts[3] == 1  # Appears in 1

    def test_frequency_counts_with_decay(self):
        """Test frequency counting with temporal decay."""
        history = RoutingHistory(layer_idx=0, window_size=10)

        # Record with expert 5 early, expert 1 recent
        history.record([5])  # Oldest, lowest weight
        history.record([1])
        history.record([1])  # Most recent, highest weight

        counts = history.get_frequency_counts(decay=0.5)

        # Expert 1 appears at positions 1 and 2 (most recent)
        # Expert 5 appears at position 0 (oldest)
        # With decay=0.5: weights are [0.25, 0.5, 1.0]
        assert counts[1] > counts[5]  # Recent experts weighted higher

    def test_empty_history(self):
        """Test behavior with empty history."""
        history = RoutingHistory(layer_idx=0, window_size=10)

        assert history.get_last_experts() == []
        assert history.get_frequency_counts() == {}


class TestPredictNextExperts:
    """Tests for predict_next_experts heuristic."""

    def test_repeat_strategy(self):
        """Test REPEAT strategy returns same experts."""
        current = mx.array([3, 7])

        predicted = predict_next_experts(
            current_routing=current,
            strategy=PrefetchStrategy.REPEAT,
            prefetch_k=2,
        )

        assert predicted == [3, 7]

    def test_repeat_strategy_truncates(self):
        """Test REPEAT strategy respects prefetch_k."""
        current = mx.array([1, 2, 3, 4])

        predicted = predict_next_experts(
            current_routing=current,
            strategy=PrefetchStrategy.REPEAT,
            prefetch_k=2,
        )

        assert len(predicted) == 2
        assert predicted == [1, 2]

    def test_history_strategy(self):
        """Test HISTORY strategy uses frequency."""
        history = RoutingHistory(layer_idx=0, window_size=10)
        history.record([1, 5])
        history.record([2, 5])
        history.record([3, 5])  # Expert 5 most frequent

        predicted = predict_next_experts(
            current_routing=[3, 5],
            history=history,
            strategy=PrefetchStrategy.HISTORY,
            prefetch_k=2,
        )

        # Expert 5 appears in all 3, should be first
        assert 5 in predicted

    def test_top_k_recency_strategy(self):
        """Test TOP_K_RECENCY blends current and history."""
        history = RoutingHistory(layer_idx=0, window_size=10)
        history.record([10, 20])
        history.record([10, 30])

        current = [1, 2]  # New experts

        predicted = predict_next_experts(
            current_routing=current,
            history=history,
            strategy=PrefetchStrategy.TOP_K_RECENCY,
            prefetch_k=4,
            decay_factor=0.9,
        )

        # Current routing should be boosted
        assert 1 in predicted or 2 in predicted

    def test_fallback_with_no_history(self):
        """Test strategies fall back gracefully without history."""
        current = mx.array([5, 10])

        for strategy in PrefetchStrategy:
            predicted = predict_next_experts(
                current_routing=current,
                history=None,
                strategy=strategy,
                prefetch_k=2,
            )
            # Should return something, not crash
            assert len(predicted) <= 2


class TestPrefetchStats:
    """Tests for prefetch statistics tracking."""

    def test_prediction_accuracy(self):
        """Test prediction accuracy calculation."""
        stats = PrefetchStats(layer_idx=0)

        # 3 predictions, 2 correct (at least one hit)
        stats.record_prediction([1, 2], [1, 3])  # Hit on 1
        stats.record_prediction([4, 5], [6, 7])  # Miss
        stats.record_prediction([8, 9], [9, 10])  # Hit on 9

        assert stats.predictions_made == 3
        assert stats.predictions_correct == 2
        assert stats.prediction_accuracy == pytest.approx(2 / 3)

    def test_expert_hit_rate(self):
        """Test expert-level hit rate."""
        stats = PrefetchStats(layer_idx=0)

        # Prefetch [1, 2, 3], actual [1, 2] -> 2/3 hit rate
        stats.record_prediction([1, 2, 3], [1, 2])

        assert stats.experts_prefetched == 3
        assert stats.experts_hit == 2
        assert stats.expert_hit_rate == pytest.approx(2 / 3)

    def test_latency_tracking(self):
        """Test latency statistics."""
        stats = PrefetchStats(layer_idx=0)

        stats.record_latency(1.0)
        stats.record_latency(2.0)
        stats.record_latency(3.0)

        assert stats.prefetch_latency_ms == pytest.approx(2.0)


class TestAsyncExpertLoader:
    """Tests for async expert loading."""

    def test_loader_lifecycle(self):
        """Test start/stop lifecycle."""
        loader = AsyncExpertLoader(num_threads=1)

        assert loader._executor is None
        loader.start()
        assert loader._executor is not None
        loader.stop()
        assert loader._executor is None

    def test_context_manager(self):
        """Test context manager usage."""
        with AsyncExpertLoader(num_threads=1) as loader:
            assert loader._executor is not None
        assert loader._executor is None

    def test_submit_and_wait(self):
        """Test submitting and waiting for load.

        NOTE: This test uses numpy arrays to avoid MLX thread-safety issues.
        MLX operations should only be evaluated on the main thread.
        """
        loaded = []

        def mock_load():
            time.sleep(0.01)
            loaded.append(True)
            # Return numpy array to avoid MLX threading issues in tests
            return np.zeros((10, 10))

        with AsyncExpertLoader(num_threads=2) as loader:
            loader.submit(layer_idx=0, expert_id=1, load_fn=mock_load)
            loader.submit(layer_idx=0, expert_id=2, load_fn=mock_load)
            loader.wait_all()

        assert len(loaded) == 2

    def test_duplicate_submit_ignored(self):
        """Test that duplicate submits are ignored while pending."""
        call_count = [0]

        def mock_load():
            call_count[0] += 1
            time.sleep(0.05)
            # Return numpy array to avoid MLX threading issues
            return np.zeros((10, 10))

        with AsyncExpertLoader(num_threads=1) as loader:
            # Submit same expert twice quickly
            loader.submit(layer_idx=0, expert_id=1, load_fn=mock_load)
            loader.submit(layer_idx=0, expert_id=1, load_fn=mock_load)  # Should be ignored
            loader.wait_all()

        assert call_count[0] == 1


class TestAsyncLoadExperts:
    """Tests for async_load_experts convenience function."""

    def test_creates_loader_if_none(self):
        """Test that function creates loader if not provided."""
        calls = []

        def load_fn(layer, expert):
            calls.append((layer, expert))
            # Return numpy to avoid MLX threading issues
            return np.zeros((10, 10))

        loader = async_load_experts(
            layer_idx=0,
            expert_ids=[1, 2],
            load_fn=load_fn,
            loader=None,
            num_threads=2,
        )

        loader.wait_all()
        loader.stop()

        assert len(calls) == 2

    def test_reuses_existing_loader(self):
        """Test that function reuses provided loader."""
        with AsyncExpertLoader(num_threads=1) as existing_loader:
            calls = []

            def load_fn(layer, expert):
                calls.append((layer, expert))
                # Return numpy to avoid MLX threading issues
                return np.zeros((10, 10))

            returned = async_load_experts(
                layer_idx=0,
                expert_ids=[5],
                load_fn=load_fn,
                loader=existing_loader,
            )

            assert returned is existing_loader
            returned.wait_all()


class TestExpertPrefetcher:
    """Tests for high-level ExpertPrefetcher."""

    def test_record_and_predict(self):
        """Test recording routing and predicting."""
        prefetcher = ExpertPrefetcher(
            num_experts=64,
            num_layers=2,
            config=PrefetchConfig(
                strategy=PrefetchStrategy.TOP_K_RECENCY,
                prefetch_k=2,
            ),
        )

        # Record some routing decisions
        prefetcher.record_routing(layer_idx=0, expert_ids=[3, 7])
        prefetcher.record_routing(layer_idx=0, expert_ids=[3, 8])

        # Predict should use history
        predicted = prefetcher.predict_next_experts(layer_idx=0)

        # Expert 3 appears twice, should be predicted
        assert 3 in predicted

    def test_step_combined_operation(self):
        """Test step() does record + predict + prefetch."""
        loads = []

        def mock_load(layer, expert):
            loads.append((layer, expert))
            # Return numpy to avoid MLX threading issues
            return np.zeros((10, 10))

        prefetcher = ExpertPrefetcher(
            num_experts=64,
            num_layers=1,
            load_fn=mock_load,
            config=PrefetchConfig(prefetch_k=2),
        )

        with prefetcher:
            predicted = prefetcher.step(layer_idx=0, expert_ids=[1, 2])
            prefetcher.wait_prefetch()

        assert len(predicted) <= 2
        assert len(loads) > 0  # Something was prefetched

    def test_stats_collection(self):
        """Test statistics collection."""
        prefetcher = ExpertPrefetcher(
            num_experts=64,
            num_layers=2,
            config=PrefetchConfig(enable_stats=True),
        )

        # Simulate some predictions
        prefetcher.record_routing(0, [1, 2])
        prefetcher.predict_next_experts(0)
        prefetcher.record_routing(0, [1, 3])  # Check prediction

        stats = prefetcher.get_stats()

        assert "config" in stats
        assert "per_layer" in stats
        assert "summary" in stats

    def test_clear_history(self):
        """Test clearing history."""
        prefetcher = ExpertPrefetcher(num_experts=64, num_layers=2)

        prefetcher.record_routing(0, [1, 2])
        prefetcher.record_routing(1, [3, 4])

        # Clear single layer
        prefetcher.clear_history(layer_idx=0)
        assert len(prefetcher._history[0].history) == 0
        assert len(prefetcher._history[1].history) == 1

        # Clear all
        prefetcher.clear_history()
        assert len(prefetcher._history[0].history) == 0
        assert len(prefetcher._history[1].history) == 0


class TestExpertLRUCache:
    """Tests for expert weight LRU cache."""

    def test_put_and_get(self):
        """Test basic put/get."""
        cache = ExpertLRUCache(max_size_mb=10)

        weights = mx.ones((100, 100))
        cache.put(layer_idx=0, expert_id=5, weights=weights)

        retrieved = cache.get(layer_idx=0, expert_id=5)
        assert retrieved is not None
        assert mx.array_equal(retrieved, weights)

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = ExpertLRUCache(max_size_mb=10)

        assert cache.get(layer_idx=0, expert_id=99) is None

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        # Small cache that can hold ~1 entry
        cache = ExpertLRUCache(max_size_mb=1)

        # Create weights that are ~0.5MB each (500KB)
        small_weights = mx.zeros((128, 1024), dtype=mx.float32)  # 512KB

        cache.put(0, 1, small_weights)
        cache.put(0, 2, small_weights)
        cache.put(0, 3, small_weights)  # Should evict oldest

        # Expert 1 should be evicted (LRU)
        assert cache.get(0, 1) is None
        # Expert 3 should still be there
        assert cache.get(0, 3) is not None

    def test_get_or_load(self):
        """Test get_or_load caches on miss."""
        cache = ExpertLRUCache(max_size_mb=10)
        call_count = [0]

        def loader():
            call_count[0] += 1
            return mx.ones((10, 10))

        # First call loads
        w1 = cache.get_or_load(0, 5, loader)
        assert call_count[0] == 1

        # Second call uses cache
        w2 = cache.get_or_load(0, 5, loader)
        assert call_count[0] == 1  # No additional load
        assert mx.array_equal(w1, w2)

    def test_invalidate_single(self):
        """Test invalidating single expert."""
        cache = ExpertLRUCache(max_size_mb=10)
        weights = mx.ones((10, 10))

        cache.put(0, 1, weights)
        cache.put(0, 2, weights)

        evicted = cache.invalidate(layer_idx=0, expert_id=1)

        assert evicted == 1
        assert cache.get(0, 1) is None
        assert cache.get(0, 2) is not None

    def test_invalidate_layer(self):
        """Test invalidating entire layer."""
        cache = ExpertLRUCache(max_size_mb=10)
        weights = mx.ones((10, 10))

        cache.put(0, 1, weights)
        cache.put(0, 2, weights)
        cache.put(1, 1, weights)

        evicted = cache.invalidate(layer_idx=0)

        assert evicted == 2
        assert cache.get(0, 1) is None
        assert cache.get(0, 2) is None
        assert cache.get(1, 1) is not None  # Different layer preserved

    def test_hit_rate(self):
        """Test hit rate calculation."""
        cache = ExpertLRUCache(max_size_mb=10)
        cache.put(0, 1, mx.ones((10, 10)))

        cache.get(0, 1)  # Hit
        cache.get(0, 1)  # Hit
        cache.get(0, 99)  # Miss

        assert cache.hit_rate == pytest.approx(2 / 3)


class TestIntegration:
    """Integration tests for the prefetch system."""

    def test_end_to_end_prefetch(self):
        """Test complete prefetch workflow."""
        num_experts = 8
        num_layers = 2

        # Track which experts were loaded
        load_log: list[tuple[int, int]] = []

        def load_expert(layer_idx: int, expert_id: int) -> np.ndarray:
            load_log.append((layer_idx, expert_id))
            # Simulate dequantization (return numpy to avoid MLX threading issues)
            time.sleep(0.001)
            return np.zeros((256, 256))

        with ExpertPrefetcher(
            num_experts=num_experts,
            num_layers=num_layers,
            load_fn=load_expert,
            config=PrefetchConfig(
                strategy=PrefetchStrategy.TOP_K_RECENCY,
                prefetch_k=2,
                history_window=8,
            ),
        ) as prefetcher:
            # Simulate 10 tokens of generation
            for token_idx in range(10):
                for layer_idx in range(num_layers):
                    # Simulate router output (biased toward certain experts)
                    base_expert = (token_idx + layer_idx) % num_experts
                    expert_ids = [base_expert, (base_expert + 1) % num_experts]

                    # Step: record, predict, prefetch
                    prefetcher.step(layer_idx, expert_ids)

            prefetcher.wait_prefetch()

        # Some experts should have been loaded
        assert len(load_log) > 0

        # Get stats
        stats = prefetcher.get_stats()
        assert stats["summary"]["total_predictions"] > 0
