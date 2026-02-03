"""Tests for buffer pool efficiency metrics."""

import pytest


class TestBufferPoolMetrics:
    """Tests for BufferPoolMetrics class."""

    def test_metrics_initialization(self):
        """Test that metrics initialize with correct defaults."""
        from metal_marlin._buffer_pool import BufferPoolMetrics

        metrics = BufferPoolMetrics()

        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
        assert metrics.hit_rate == 0.0
        assert metrics.fragmentation_ratio == 0.0
        assert metrics.current_allocated == 0
        assert metrics.current_pooled == 0
        assert metrics.avg_buffer_lifetime == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        from metal_marlin._buffer_pool import BufferPoolMetrics

        metrics = BufferPoolMetrics()

        # No requests yet
        assert metrics.hit_rate == 0.0

        # 50% hit rate
        metrics.cache_hits = 5
        metrics.cache_misses = 5
        assert metrics.hit_rate == 0.5

        # 100% hit rate
        metrics.cache_misses = 0
        assert metrics.hit_rate == 1.0

    def test_smoothed_hit_rate(self):
        """Test exponentially smoothed hit rate."""
        from metal_marlin._buffer_pool import BufferPoolMetrics

        metrics = BufferPoolMetrics()

        # Start with a hit
        metrics.update_smoothed_hit_rate(True, alpha=0.5)
        assert metrics.hit_rate_smoothed == 0.5

        # Another hit
        metrics.update_smoothed_hit_rate(True, alpha=0.5)
        assert metrics.hit_rate_smoothed == 0.75

        # A miss
        metrics.update_smoothed_hit_rate(False, alpha=0.5)
        assert metrics.hit_rate_smoothed == 0.375

    def test_buffer_lifetime_tracking(self):
        """Test buffer lifetime tracking."""
        from metal_marlin._buffer_pool import BufferPoolMetrics

        metrics = BufferPoolMetrics()

        # Record some lifetimes
        metrics.record_buffer_lifetime(0.1)
        metrics.record_buffer_lifetime(0.3)
        metrics.record_buffer_lifetime(0.2)

        assert metrics.buffer_lifetime_count == 3
        assert metrics.avg_buffer_lifetime == pytest.approx(0.2, abs=0.001)
        assert metrics.min_buffer_lifetime == 0.1
        assert metrics.max_buffer_lifetime == 0.3

    def test_pool_efficiency_score(self):
        """Test pool efficiency score calculation."""
        from metal_marlin._buffer_pool import BufferPoolMetrics

        metrics = BufferPoolMetrics()

        # Not enough data
        assert metrics.pool_efficiency == 0.0

        # Good efficiency (high hit rate, low fragmentation)
        metrics.cache_hits = 90
        metrics.cache_misses = 10
        metrics.fragmentation_ratio = 0.1

        # Efficiency = 0.9 * 0.6 + 0.9 * 0.4 = 0.9
        assert metrics.pool_efficiency == pytest.approx(0.9, abs=0.01)

    def test_tuning_recommendations_low_hit_rate(self):
        """Test tuning recommendations for low hit rate."""
        from metal_marlin._buffer_pool import BufferPoolMetrics

        metrics = BufferPoolMetrics()
        metrics.cache_hits = 30
        metrics.cache_misses = 70
        metrics.fragmentation_ratio = 0.1

        recs = metrics.get_tuning_recommendations()

        assert any("Low hit rate" in r for r in recs)

    def test_tuning_recommendations_high_fragmentation(self):
        """Test tuning recommendations for high fragmentation."""
        from metal_marlin._buffer_pool import BufferPoolMetrics

        metrics = BufferPoolMetrics()
        metrics.cache_hits = 80
        metrics.cache_misses = 20
        metrics.fragmentation_ratio = 0.6

        recs = metrics.get_tuning_recommendations()

        assert any("fragmentation" in r.lower() for r in recs)

    def test_to_dict(self):
        """Test metrics serialization to dict."""
        from metal_marlin._buffer_pool import BufferPoolMetrics

        metrics = BufferPoolMetrics()
        metrics.cache_hits = 50
        metrics.cache_misses = 50
        metrics.fragmentation_ratio = 0.2
        metrics.current_allocated = 1000
        metrics.current_pooled = 500
        metrics.record_buffer_lifetime(0.1)

        d = metrics.to_dict()

        assert d["cache_hits"] == 50
        assert d["cache_misses"] == 50
        assert d["hit_rate"] == 0.5
        assert d["fragmentation_ratio"] == 0.2
        assert d["current_allocated_bytes"] == 1000
        assert d["current_pooled_bytes"] == 500
        assert d["avg_buffer_lifetime_sec"] == 0.1


class TestMetalBufferPoolMetrics:
    """Tests for MetalBufferPool metrics integration."""

    def test_metrics_property(self):
        """Test that metrics property returns current metrics."""
        # Skip if Metal not available
        pytest.importorskip("Metal")

        import Metal as Metal

        from metal_marlin._buffer_pool import MetalBufferPool

        device = Metal.MTLCreateSystemDefaultDevice()
        pool = MetalBufferPool(device)

        metrics = pool.metrics

        assert metrics is not None
        assert hasattr(metrics, 'cache_hits')
        assert hasattr(metrics, 'hit_rate')

    def test_stats_includes_metrics(self):
        """Test that stats() includes new metrics."""
        # Skip if Metal not available
        pytest.importorskip("Metal")

        import Metal as Metal

        from metal_marlin._buffer_pool import MetalBufferPool

        device = Metal.MTLCreateSystemDefaultDevice()
        pool = MetalBufferPool(device)

        stats = pool.stats()

        # Check new metrics are present
        assert "hit_rate" in stats
        assert "hit_rate_smoothed" in stats
        assert "peak_memory_bytes" in stats
        assert "avg_buffer_lifetime_sec" in stats
        assert "pool_efficiency" in stats
        assert "total_requests" in stats
        assert "tuning_recommendations" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
