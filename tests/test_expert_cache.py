"""Tests for the ExpertCache module."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import pytest
from metal_marlin.expert_cache import (
    CacheEntry,
    ExpertCache,
    ExpertStats,
    LayerStats,
    TileCoordinator,
    TileKey,
    create_moe_cache,
)


class TestTileKey:
    """Tests for TileKey dataclass."""

    def test_creation(self):
        key = TileKey(layer_idx=0, expert_id=5, tile_idx=10)
        assert key.layer_idx == 0
        assert key.expert_id == 5
        assert key.tile_idx == 10

    def test_equality(self):
        key1 = TileKey(0, 5, 10)
        key2 = TileKey(0, 5, 10)
        key3 = TileKey(0, 5, 11)

        assert key1 == key2
        assert key1 != key3

    def test_hash(self):
        key1 = TileKey(0, 5, 10)
        key2 = TileKey(0, 5, 10)

        # Equal keys should have equal hashes
        assert hash(key1) == hash(key2)

        # Can be used as dict key
        d = {key1: "value"}
        assert d[key2] == "value"


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_creation(self):
        key = TileKey(0, 0, 0)
        data = mx.zeros((64, 64))

        entry = CacheEntry(key=key, data=data, size_bytes=data.nbytes)

        assert entry.key == key
        assert entry.access_count == 0
        assert entry.size_bytes == data.nbytes

    def test_touch(self):
        key = TileKey(0, 0, 0)
        data = mx.zeros((64, 64))
        entry = CacheEntry(key=key, data=data, size_bytes=data.nbytes)

        initial_time = entry.last_access
        initial_count = entry.access_count

        time.sleep(0.01)
        entry.touch()

        assert entry.access_count == initial_count + 1
        assert entry.last_access > initial_time


class TestExpertStats:
    """Tests for ExpertStats dataclass."""

    def test_activation_rate(self):
        stats = ExpertStats(expert_id=0)

        # Initially zero
        assert stats.activation_rate == 0.0

        # Record some activations
        stats.record_batch(100)
        stats.record_activation(10)

        assert stats.activation_rate == pytest.approx(0.1, rel=1e-3)

    def test_recent_rate(self):
        stats = ExpertStats(expert_id=0)

        # Initially zero
        assert stats.recent_rate == 0.0

        # Record activations
        for _ in range(10):
            stats.record_activation(5)

        assert stats.recent_rate == pytest.approx(5.0, rel=1e-3)

    def test_recent_window_limit(self):
        stats = ExpertStats(expert_id=0)

        # Record more than window size
        for i in range(150):
            stats.record_activation(1)

        # Window should be limited to 100
        assert len(stats.recent_window) == 100


class TestLayerStats:
    """Tests for LayerStats dataclass."""

    def test_hit_rate(self):
        stats = LayerStats(layer_idx=0)

        # Initially zero
        assert stats.hit_rate == 0.0

        # Add some hits and misses
        stats.cache_hits = 70
        stats.cache_misses = 30

        assert stats.hit_rate == pytest.approx(0.7, rel=1e-3)

    def test_get_expert_stats(self):
        stats = LayerStats(layer_idx=0)

        # Should create new stats
        expert_stats = stats.get_expert_stats(5)
        assert expert_stats.expert_id == 5

        # Should return same stats
        expert_stats2 = stats.get_expert_stats(5)
        assert expert_stats2 is expert_stats

    def test_get_hot_experts(self):
        stats = LayerStats(layer_idx=0)

        # Set up some expert stats with varying recent_rates
        # recent_rate is the average of recent_window values
        # To create different rates, we record different activation counts per call
        for eid in range(10):
            expert_stats = stats.get_expert_stats(eid)
            # Record a single activation with count = eid
            # So expert 9 has recent_rate=9, expert 8 has recent_rate=8, etc.
            expert_stats.record_activation(eid)

        # Get hot experts with threshold > 0 to filter out expert 0 (rate=0)
        hot = stats.get_hot_experts(threshold=0.5, top_k=3)

        # Should get top 3 by recent_rate: experts 9, 8, 7
        assert len(hot) == 3
        assert hot == [9, 8, 7]  # Sorted by rate descending


class TestExpertCache:
    """Tests for ExpertCache class."""

    def test_basic_get_and_cache(self):
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        call_count = 0

        def dequant_fn():
            nonlocal call_count
            call_count += 1
            return mx.ones((64, 64), dtype=mx.float16)

        # First call should invoke dequant_fn
        tile1 = cache.get_expert_tile(0, 0, 0, dequant_fn)
        assert call_count == 1
        assert tile1.shape == (64, 64)

        # Second call should use cache
        tile2 = cache.get_expert_tile(0, 0, 0, dequant_fn)
        assert call_count == 1  # Not called again
        assert mx.array_equal(tile1, tile2)

    def test_lru_eviction(self):
        # Small cache that can only hold ~2 tiles
        tile_size = 64 * 64 * 2  # float16
        cache_size_mb = (tile_size * 2.5) / (1024 * 1024)
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=int(cache_size_mb * 1000) / 1000)

        def make_tile(expert_id):
            return mx.full((64, 64), expert_id, dtype=mx.float16)

        # Add 3 tiles - should evict first one
        cache.get_expert_tile(0, 0, 0, lambda: make_tile(0))
        cache.get_expert_tile(0, 1, 0, lambda: make_tile(1))
        cache.get_expert_tile(0, 2, 0, lambda: make_tile(2))

        # First tile should be evicted (LRU)
        assert cache._total_evictions > 0

    def test_hit_rate(self):
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        def dequant_fn():
            return mx.ones((64, 64), dtype=mx.float16)

        # Generate some hits and misses
        cache.get_expert_tile(0, 0, 0, dequant_fn)  # Miss
        cache.get_expert_tile(0, 0, 0, dequant_fn)  # Hit
        cache.get_expert_tile(0, 0, 0, dequant_fn)  # Hit
        cache.get_expert_tile(0, 1, 0, dequant_fn)  # Miss
        cache.get_expert_tile(0, 0, 0, dequant_fn)  # Hit

        # 3 hits, 2 misses = 60% hit rate
        assert cache.hit_rate == pytest.approx(0.6, rel=1e-3)

    def test_invalidate_expert(self):
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        def dequant_fn():
            return mx.ones((64, 64), dtype=mx.float16)

        # Cache some tiles
        cache.get_expert_tile(0, 0, 0, dequant_fn)
        cache.get_expert_tile(0, 0, 1, dequant_fn)
        cache.get_expert_tile(0, 1, 0, dequant_fn)

        assert cache.num_entries == 3

        # Invalidate expert 0
        evicted = cache.invalidate_expert(layer_idx=0, expert_id=0)
        assert evicted == 2  # Two tiles for expert 0
        assert cache.num_entries == 1

    def test_invalidate_layer(self):
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        def dequant_fn():
            return mx.ones((64, 64), dtype=mx.float16)

        # Cache tiles in multiple layers
        cache.get_expert_tile(0, 0, 0, dequant_fn)
        cache.get_expert_tile(0, 1, 0, dequant_fn)
        cache.get_expert_tile(1, 0, 0, dequant_fn)

        assert cache.num_entries == 3

        # Invalidate layer 0
        evicted = cache.invalidate_layer(layer_idx=0)
        assert evicted == 2
        assert cache.num_entries == 1

    def test_clear(self):
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        def dequant_fn():
            return mx.ones((64, 64), dtype=mx.float16)

        cache.get_expert_tile(0, 0, 0, dequant_fn)
        cache.get_expert_tile(0, 1, 0, dequant_fn)

        assert cache.num_entries == 2

        cache.clear()

        assert cache.num_entries == 0
        assert cache.size_mb == 0.0

    def test_resize(self):
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        def dequant_fn():
            return mx.ones((64, 64), dtype=mx.float16)

        # Fill cache
        for i in range(10):
            cache.get_expert_tile(0, i, 0, dequant_fn)

        initial_entries = cache.num_entries

        # Shrink cache - should evict some entries
        evicted = cache.resize(0)
        assert evicted == initial_entries
        assert cache.num_entries == 0

    def test_record_expert_activation(self):
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        # Record some activations
        expert_ids = mx.array([[0, 1], [0, 2], [1, 3]], dtype=mx.uint32)
        cache.record_expert_activation(layer_idx=0, expert_ids=expert_ids)

        # Check stats
        stats = cache._layer_stats[0]
        assert stats.get_expert_stats(0).activation_count == 2
        assert stats.get_expert_stats(1).activation_count == 2
        assert stats.get_expert_stats(2).activation_count == 1
        assert stats.get_expert_stats(3).activation_count == 1

    def test_get_prefetch_candidates(self):
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1, prefetch_k=3)

        # Record activations with different frequencies
        for _ in range(10):
            expert_ids = mx.array([[0, 1], [0, 2]], dtype=mx.uint32)
            cache.record_expert_activation(0, expert_ids)

        # Expert 0 should be hottest (activated every batch)
        candidates = cache.get_prefetch_candidates(0)
        assert 0 in candidates

    def test_get_stats(self):
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        def dequant_fn():
            return mx.ones((64, 64), dtype=mx.float16)

        cache.get_expert_tile(0, 0, 0, dequant_fn)
        cache.get_expert_tile(0, 0, 0, dequant_fn)  # Hit

        stats = cache.get_stats()

        assert "global" in stats
        assert "memory" in stats
        assert "config" in stats
        assert "per_layer" in stats

        assert stats["global"]["total_hits"] == 1
        assert stats["global"]["total_misses"] == 1
        assert stats["config"]["num_experts"] == 8

    def test_thread_safety_cache_operations(self):
        """Test that cache internal operations are thread-safe.

        Note: MLX array creation is not thread-safe, so we pre-create tiles
        and only test the cache operations (get/put/evict) for thread safety.
        """
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        # Pre-create tiles in the main thread (MLX not thread-safe for array creation)
        precomputed_tiles = {
            i: mx.full((64, 64), i, dtype=mx.float16) for i in range(4)
        }
        for tile in precomputed_tiles.values():
            mx.eval(tile)

        results = []
        lock = threading.Lock()

        def worker(expert_id):
            # Use pre-computed tile
            tile = precomputed_tiles[expert_id]

            def dequant_fn():
                return tile

            for _ in range(10):
                cached_tile = cache.get_expert_tile(0, expert_id, 0, dequant_fn)
                # Read the value (this should be safe as the tile is already evaluated)
                with lock:
                    results.append((expert_id, cached_tile[0, 0].item()))

        # Run concurrent access to cache operations
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            for f in futures:
                f.result()

        # All results should be consistent
        for expert_id, value in results:
            assert value == expert_id

    def test_repr(self):
        cache = ExpertCache(num_experts=64, num_layers=28, cache_size_mb=512)
        repr_str = repr(cache)

        assert "ExpertCache" in repr_str
        assert "num_experts=64" in repr_str
        assert "num_layers=28" in repr_str


class TestTileCoordinator:
    """Tests for TileCoordinator class."""

    def test_basic_properties(self):
        coord = TileCoordinator(
            weight_shape=(256, 128),
            tile_shape=(64, 64),
        )

        assert coord.num_row_tiles == 4  # 256/64
        assert coord.num_col_tiles == 2  # 128/64
        assert coord.num_tiles == 8

    def test_tile_to_coords(self):
        coord = TileCoordinator(
            weight_shape=(256, 128),
            tile_shape=(64, 64),
        )

        assert coord.tile_to_coords(0) == (0, 0)
        assert coord.tile_to_coords(1) == (0, 1)
        assert coord.tile_to_coords(2) == (1, 0)
        assert coord.tile_to_coords(3) == (1, 1)

    def test_coords_to_tile(self):
        coord = TileCoordinator(
            weight_shape=(256, 128),
            tile_shape=(64, 64),
        )

        assert coord.coords_to_tile(0, 0) == 0
        assert coord.coords_to_tile(0, 1) == 1
        assert coord.coords_to_tile(1, 0) == 2
        assert coord.coords_to_tile(1, 1) == 3

    def test_roundtrip(self):
        coord = TileCoordinator(
            weight_shape=(256, 128),
            tile_shape=(64, 64),
        )

        for tile_idx in range(coord.num_tiles):
            row, col = coord.tile_to_coords(tile_idx)
            assert coord.coords_to_tile(row, col) == tile_idx

    def test_tile_bounds(self):
        coord = TileCoordinator(
            weight_shape=(256, 128),
            tile_shape=(64, 64),
        )

        # First tile
        r_start, r_end, c_start, c_end = coord.tile_bounds(0)
        assert (r_start, r_end, c_start, c_end) == (0, 64, 0, 64)

        # Second tile
        r_start, r_end, c_start, c_end = coord.tile_bounds(1)
        assert (r_start, r_end, c_start, c_end) == (0, 64, 64, 128)

    def test_tile_bounds_with_padding(self):
        # Non-aligned dimensions
        coord = TileCoordinator(
            weight_shape=(100, 100),
            tile_shape=(64, 64),
        )

        # Last tile should be clipped
        last_tile = coord.num_tiles - 1
        r_start, r_end, c_start, c_end = coord.tile_bounds(last_tile)
        assert r_end == 100  # Clipped to actual size
        assert c_end == 100

    def test_all_tile_indices(self):
        coord = TileCoordinator(
            weight_shape=(256, 128),
            tile_shape=(64, 64),
        )

        indices = coord.all_tile_indices()
        assert indices == list(range(8))

    def test_tiles_for_output_range(self):
        coord = TileCoordinator(
            weight_shape=(256, 128),
            tile_shape=(64, 64),
        )

        # Need outputs 0-64 (first row of tiles)
        tiles = coord.tiles_for_output_range(0, 64)
        assert tiles == [0, 1]  # First row of tiles

        # Need outputs 64-128 (second row of tiles)
        tiles = coord.tiles_for_output_range(64, 128)
        assert tiles == [2, 3]  # Second row of tiles


class TestCreateMoeCache:
    """Tests for create_moe_cache helper function."""

    def test_glm4_config(self):
        config = {
            "num_hidden_layers": 28,
            "num_experts": 64,
        }

        cache = create_moe_cache(config, cache_size_mb=256)

        assert cache.num_layers == 28
        assert cache.num_experts == 64
        assert cache.cache_size_bytes == 256 * 1024 * 1024

    def test_mixtral_config(self):
        config = {
            "n_layer": 32,
            "num_local_experts": 8,
        }

        cache = create_moe_cache(config, cache_size_mb=512)

        assert cache.num_layers == 32
        assert cache.num_experts == 8

    def test_default_values(self):
        config = {}  # Empty config

        cache = create_moe_cache(config)

        assert cache.num_layers == 32  # Default
        assert cache.num_experts == 64  # Default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
