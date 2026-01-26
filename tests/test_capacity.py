"""Tests for MoE capacity factor adjustment."""

import math

import mlx.core as mx
import numpy as np
import pytest

from metal_marlin.capacity import (
    CapacityAnalyzer,
    DynamicCapacity,
    DynamicCapacityConfig,
    analyze_overflow_rate,
    auto_tune_capacity,
    compute_expert_capacity,
    count_expert_assignments,
    dynamic_capacity,
)


class TestComputeExpertCapacity:
    """Tests for compute_expert_capacity function."""

    def test_basic_capacity(self):
        """Test basic capacity calculation."""
        # 64 tokens, 8 experts, factor 1.0, top_k=1
        # Expected: ceil(64 * 1 * 1.0 / 8) = 8
        capacity = compute_expert_capacity(64, 8, 1.0, top_k=1)
        assert capacity == 8

    def test_capacity_with_top_k(self):
        """Test capacity with top_k > 1."""
        # 64 tokens, 8 experts, factor 1.0, top_k=2
        # Expected: ceil(64 * 2 * 1.0 / 8) = 16
        capacity = compute_expert_capacity(64, 8, 1.0, top_k=2)
        assert capacity == 16

    def test_capacity_with_factor(self):
        """Test capacity with non-unity factor."""
        # 64 tokens, 8 experts, factor 1.5, top_k=1
        # Expected: ceil(64 * 1 * 1.5 / 8) = ceil(12) = 12
        capacity = compute_expert_capacity(64, 8, 1.5, top_k=1)
        assert capacity == 12

    def test_capacity_rounds_up(self):
        """Test that capacity rounds up."""
        # 10 tokens, 8 experts, factor 1.0, top_k=1
        # Expected: ceil(10 * 1 * 1.0 / 8) = ceil(1.25) = 2
        capacity = compute_expert_capacity(10, 8, 1.0, top_k=1)
        assert capacity == 2

    def test_minimum_capacity(self):
        """Test minimum capacity is 1."""
        # Very small batch
        capacity = compute_expert_capacity(1, 64, 1.0, top_k=1)
        assert capacity >= 1

    def test_invalid_factor_raises(self):
        """Test that invalid capacity factor raises."""
        with pytest.raises(ValueError):
            compute_expert_capacity(64, 8, 0.0, top_k=1)
        with pytest.raises(ValueError):
            compute_expert_capacity(64, 8, -1.0, top_k=1)


class TestCountExpertAssignments:
    """Tests for count_expert_assignments function."""

    def test_uniform_distribution(self):
        """Test counting with uniform distribution."""
        # Each expert gets exactly 2 assignments
        expert_ids = mx.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=mx.int32)
        counts = count_expert_assignments(expert_ids, num_experts=4)
        mx.eval(counts)

        expected = np.array([2, 2, 2, 2], dtype=np.int32)
        np.testing.assert_array_equal(np.array(counts), expected)

    def test_skewed_distribution(self):
        """Test counting with skewed distribution."""
        # Expert 0 gets most assignments
        expert_ids = mx.array([[0, 0], [0, 1], [0, 0]], dtype=mx.int32)
        counts = count_expert_assignments(expert_ids, num_experts=4)
        mx.eval(counts)

        expected = np.array([5, 1, 0, 0], dtype=np.int32)
        np.testing.assert_array_equal(np.array(counts), expected)


class TestCapacityAnalyzer:
    """Tests for CapacityAnalyzer class."""

    def test_no_overflow(self):
        """Test analyzer with no overflow."""
        analyzer = CapacityAnalyzer(num_experts=4, capacity_factor=2.0, top_k=2)

        # Uniform distribution, well under capacity
        expert_ids = mx.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=mx.int32)
        info = analyzer.record_batch(expert_ids)

        assert info.dropped_tokens == 0
        assert info.drop_rate == 0.0
        assert len(info.overflow_experts) == 0

        stats = analyzer.get_stats()
        assert stats.overall_drop_rate == 0.0
        assert stats.batches_with_overflow == 0

    def test_with_overflow(self):
        """Test analyzer detecting overflow."""
        analyzer = CapacityAnalyzer(num_experts=4, capacity_factor=1.0, top_k=2)

        # Expert 0 gets 6 assignments, capacity is ceil(4*2*1.0/4)=2
        expert_ids = mx.array([[0, 0], [0, 0], [0, 0], [1, 2]], dtype=mx.int32)
        info = analyzer.record_batch(expert_ids)

        # Capacity = 2, expert 0 has 6 assignments -> 4 overflow
        assert info.dropped_tokens == 4
        assert info.drop_rate == 4 / 8  # 4 dropped out of 8 total
        assert 0 in info.overflow_experts

        stats = analyzer.get_stats()
        assert stats.overall_drop_rate > 0.0
        assert stats.batches_with_overflow == 1

    def test_multiple_batches(self):
        """Test analyzer across multiple batches."""
        analyzer = CapacityAnalyzer(num_experts=4, capacity_factor=1.0, top_k=1)

        # Batch 1: no overflow
        expert_ids1 = mx.array([[0], [1], [2], [3]], dtype=mx.int32)
        info1 = analyzer.record_batch(expert_ids1)
        assert info1.dropped_tokens == 0

        # Batch 2: overflow
        expert_ids2 = mx.array([[0], [0], [0], [0]], dtype=mx.int32)
        info2 = analyzer.record_batch(expert_ids2)
        assert info2.dropped_tokens > 0

        stats = analyzer.get_stats()
        assert stats.total_batches == 2
        assert stats.batches_with_overflow == 1

    def test_reset(self):
        """Test analyzer reset."""
        analyzer = CapacityAnalyzer(num_experts=4, capacity_factor=1.0, top_k=1)

        expert_ids = mx.array([[0], [0], [0], [0]], dtype=mx.int32)
        analyzer.record_batch(expert_ids)

        stats_before = analyzer.get_stats()
        assert stats_before.total_batches > 0

        analyzer.reset()
        stats_after = analyzer.get_stats()
        assert stats_after.total_batches == 0


class TestAnalyzeOverflowRate:
    """Tests for analyze_overflow_rate function."""

    def test_single_batch(self):
        """Test analysis of single batch."""
        expert_ids = mx.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=mx.int32)
        stats = analyze_overflow_rate(expert_ids, num_experts=4, capacity_factor=2.0)

        assert stats.total_batches == 1
        assert stats.overall_drop_rate == 0.0

    def test_batch_list(self):
        """Test analysis of batch list."""
        batches = [
            mx.array([[0, 1], [2, 3]], dtype=mx.int32),
            mx.array([[0, 0], [0, 0]], dtype=mx.int32),
        ]
        stats = analyze_overflow_rate(batches, num_experts=4, capacity_factor=1.0)

        assert stats.total_batches == 2


class TestAutoTuneCapacity:
    """Tests for auto_tune_capacity function."""

    def test_finds_sufficient_factor(self):
        """Test that auto-tune finds a factor with acceptable drop rate."""
        # Create history with varying load
        np.random.seed(42)
        batches = []
        for _ in range(20):
            # Skewed distribution - some experts get more load
            expert_ids = np.random.choice(8, size=(32, 2), p=[0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.05])
            batches.append(mx.array(expert_ids, dtype=mx.int32))

        factor = auto_tune_capacity(
            batches, num_experts=8, target_drop_rate=0.01, min_factor=1.0, max_factor=4.0
        )

        # Verify the found factor achieves the target
        stats = analyze_overflow_rate(batches, num_experts=8, capacity_factor=factor)
        assert stats.overall_drop_rate <= 0.01 or factor >= 4.0

    def test_empty_history(self):
        """Test with empty history returns min_factor."""
        factor = auto_tune_capacity([], num_experts=8, target_drop_rate=0.01, min_factor=1.5)
        assert factor == 1.5

    def test_already_satisfies_target(self):
        """Test when min_factor already satisfies target."""
        # Uniform distribution - low factor should work
        batches = [
            mx.array([[i % 8, (i + 1) % 8] for i in range(8)], dtype=mx.int32)
            for _ in range(10)
        ]

        factor = auto_tune_capacity(
            batches, num_experts=8, target_drop_rate=0.5, min_factor=1.0, max_factor=4.0
        )

        # Should return minimum since target is easily achieved
        assert factor == 1.0


class TestDynamicCapacity:
    """Tests for DynamicCapacity class."""

    def test_initial_capacity(self):
        """Test initial capacity uses base factor."""
        config = DynamicCapacityConfig(base_factor=1.5)
        dyn = DynamicCapacity(num_experts=8, top_k=2, config=config)

        capacity = dyn.get_capacity(batch_size=16)
        expected = compute_expert_capacity(16, 8, 1.5, 2)
        assert capacity == expected

    def test_capacity_increases_on_overflow(self):
        """Test factor increases when drops exceed target."""
        config = DynamicCapacityConfig(
            base_factor=1.0,
            target_drop_rate=0.01,
            increase_step=0.25,
            window_size=3,
            cooldown_batches=2,
        )
        dyn = DynamicCapacity(num_experts=4, top_k=2, config=config)

        # Create batches with heavy overflow
        high_overflow_batch = mx.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=mx.int32)

        initial_factor = dyn.current_factor
        for _ in range(10):
            dyn.update(high_overflow_batch)

        # Factor should have increased
        assert dyn.current_factor > initial_factor

    def test_capacity_decreases_when_low_drops(self):
        """Test factor decreases when drops are very low."""
        config = DynamicCapacityConfig(
            base_factor=2.0,
            target_drop_rate=0.01,
            decrease_threshold=0.001,
            decrease_step=0.1,
            window_size=3,
            cooldown_batches=2,
        )
        dyn = DynamicCapacity(num_experts=8, top_k=2, config=config)

        # Create batches with no overflow (uniform distribution)
        uniform_batch = mx.array([[i % 8, (i + 1) % 8] for i in range(8)], dtype=mx.int32)

        initial_factor = dyn.current_factor
        for _ in range(15):
            dyn.update(uniform_batch)

        # Factor should have decreased
        assert dyn.current_factor < initial_factor

    def test_get_stats(self):
        """Test get_stats returns expected fields."""
        dyn = DynamicCapacity(num_experts=8, top_k=2)

        expert_ids = mx.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=mx.int32)
        dyn.update(expert_ids)

        stats = dyn.get_stats()
        assert "current_factor" in stats
        assert "recent_drop_rate" in stats
        assert "batches_recorded" in stats
        assert stats["batches_recorded"] == 1

    def test_reset(self):
        """Test reset restores initial state."""
        config = DynamicCapacityConfig(base_factor=1.5)
        dyn = DynamicCapacity(num_experts=8, top_k=2, config=config)

        # Run some updates
        expert_ids = mx.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=mx.int32)
        for _ in range(10):
            dyn.update(expert_ids)

        # Reset
        dyn.reset()

        assert dyn.current_factor == config.base_factor
        assert dyn.get_stats()["batches_recorded"] == 0


class TestDynamicCapacityFunction:
    """Tests for dynamic_capacity per-batch function."""

    def test_adapts_to_load(self):
        """Test that capacity adapts to actual load."""
        # Uniform load - same batch size for fair comparison
        # 4 tokens, each routes to 2 different experts -> each expert gets 2 assignments
        uniform_ids = mx.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=mx.int32)
        capacity_uniform = dynamic_capacity(uniform_ids, num_experts=4)

        # Skewed load (all to one expert) - same batch size
        # 4 tokens, all to expert 0 -> expert 0 gets 8 assignments
        skewed_ids = mx.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=mx.int32)
        capacity_skewed = dynamic_capacity(skewed_ids, num_experts=4)

        # With same batch size, skewed requires higher capacity because max_load is 8 vs 2
        # uniform: max expert load = 2 (each expert gets 2)
        # skewed: max expert load = 8 (expert 0 gets all 8)
        assert capacity_skewed > capacity_uniform

    def test_respects_bounds(self):
        """Test that capacity respects min/max bounds."""
        # Very skewed load
        expert_ids = mx.array([[0, 0] for _ in range(64)], dtype=mx.int32)

        capacity = dynamic_capacity(
            expert_ids, num_experts=8, min_factor=1.0, max_factor=2.0
        )

        min_capacity = compute_expert_capacity(64, 8, 1.0, 2)
        max_capacity = compute_expert_capacity(64, 8, 2.0, 2)

        assert min_capacity <= capacity <= max_capacity

    def test_target_utilization(self):
        """Test that target utilization adds headroom."""
        expert_ids = mx.array([[i % 4, (i + 1) % 4] for i in range(8)], dtype=mx.int32)

        # Lower utilization target means more headroom
        capacity_90 = dynamic_capacity(expert_ids, num_experts=4, target_utilization=0.9)
        capacity_50 = dynamic_capacity(expert_ids, num_experts=4, target_utilization=0.5)

        # 50% target should give more capacity
        assert capacity_50 >= capacity_90


class TestCapacityStatsProperties:
    """Tests for CapacityStats computed properties."""

    def test_overflow_rate(self):
        """Test overflow_rate computation."""
        analyzer = CapacityAnalyzer(num_experts=4, capacity_factor=1.0, top_k=1)

        # First batch: no overflow
        expert_ids1 = mx.array([[0], [1], [2], [3]], dtype=mx.int32)
        analyzer.record_batch(expert_ids1)

        # Second batch: overflow
        expert_ids2 = mx.array([[0], [0], [0], [0]], dtype=mx.int32)
        analyzer.record_batch(expert_ids2)

        stats = analyzer.get_stats()
        assert stats.overflow_rate == 0.5  # 1 of 2 batches had overflow

    def test_top_overflow_experts(self):
        """Test top_overflow_experts method."""
        analyzer = CapacityAnalyzer(num_experts=8, capacity_factor=1.0, top_k=1)

        # Expert 0 overflows 3 times, expert 1 overflows 2 times
        for _ in range(3):
            analyzer.record_batch(mx.array([[0], [0], [0], [0]], dtype=mx.int32))
        for _ in range(2):
            analyzer.record_batch(mx.array([[1], [1], [1], [1]], dtype=mx.int32))

        stats = analyzer.get_stats()
        top = stats.top_overflow_experts(k=2)

        assert top[0][0] == 0  # Expert 0 has most overflows
        assert top[1][0] == 1  # Expert 1 is second
