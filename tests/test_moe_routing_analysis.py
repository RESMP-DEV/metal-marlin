"""Tests for MoE routing analysis module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from metal_marlin.analysis.moe_routing import (
    ExpertCooccurrence,
    ExpertLoadStats,
    LayerRoutingProfile,
    MoERoutingProfiler,
    RoutingPredictability,
    simulate_routing_for_model,
)


class TestExpertLoadStats:
    """Tests for ExpertLoadStats dataclass."""

    def test_default_values(self) -> None:
        stats = ExpertLoadStats(expert_id=0)
        assert stats.expert_id == 0
        assert stats.total_activations == 0
        assert stats.activation_rate == 0.0
        assert stats.is_hot is False
        assert stats.is_cold is False
        assert stats.is_dead is False

    def test_primary_selection_rate_zero_activations(self) -> None:
        stats = ExpertLoadStats(expert_id=0)
        assert stats.primary_selection_rate == 0.0

    def test_primary_selection_rate_with_data(self) -> None:
        stats = ExpertLoadStats(
            expert_id=0,
            total_activations=100,
            rank_distribution={0: 60, 1: 40},
        )
        assert stats.primary_selection_rate == 0.6


class TestExpertCooccurrence:
    """Tests for ExpertCooccurrence dataclass."""

    def test_initialization(self) -> None:
        cooc = ExpertCooccurrence(num_experts=8)
        assert cooc.num_experts == 8
        assert cooc.cooccurrence_matrix.shape == (8, 8)
        assert cooc.conditional_probs is None
        assert cooc.top_pairs == []


class TestLayerRoutingProfile:
    """Tests for LayerRoutingProfile dataclass."""

    def test_empty_profile(self) -> None:
        profile = LayerRoutingProfile(layer_idx=0)
        assert profile.layer_idx == 0
        assert profile.total_tokens == 0
        assert profile.get_hot_experts() == []
        assert profile.get_cold_experts() == []
        assert profile.get_dead_experts() == []


class TestMoERoutingProfiler:
    """Tests for the main MoERoutingProfiler class."""

    def test_initialization(self) -> None:
        profiler = MoERoutingProfiler(
            num_experts=8,
            num_layers=4,
            top_k=2,
        )
        assert profiler.num_experts == 8
        assert profiler.num_layers == 4
        assert profiler.top_k == 2

    def test_record_routing_basic(self) -> None:
        profiler = MoERoutingProfiler(num_experts=8, num_layers=4, top_k=2)

        # Record some routing decisions
        expert_ids = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
        expert_probs = np.array([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.8, 0.2]])

        profiler.record_routing(0, expert_ids, expert_probs)

        # Check that data was recorded
        assert len(profiler._expert_ids[0]) == 1
        assert len(profiler._expert_probs[0]) == 1

    def test_record_routing_invalid_layer(self) -> None:
        profiler = MoERoutingProfiler(num_experts=8, num_layers=4, top_k=2)

        expert_ids = np.array([[0, 1], [2, 3]])

        with pytest.raises(ValueError):
            profiler.record_routing(-1, expert_ids)

        with pytest.raises(ValueError):
            profiler.record_routing(4, expert_ids)

    def test_layer_profiles_computation(self) -> None:
        profiler = MoERoutingProfiler(num_experts=8, num_layers=2, top_k=2)

        # Create data where expert 0 is heavily used, expert 7 is rarely used
        rng = np.random.default_rng(42)

        for layer in range(2):
            # Skewed distribution toward low-numbered experts
            probs = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.08, 0.05, 0.02])
            probs /= probs.sum()

            expert_ids = []
            for _ in range(100):
                selected = rng.choice(8, size=2, replace=False, p=probs)
                expert_ids.append(selected)

            profiler.record_routing(layer, np.array(expert_ids))

        # Access layer profiles to trigger computation
        profiles = profiler.layer_profiles

        assert len(profiles) == 2
        assert profiles[0].total_tokens == 100
        assert profiles[0].active_experts > 0

    def test_cooccurrence_computation(self) -> None:
        profiler = MoERoutingProfiler(num_experts=4, num_layers=1, top_k=2)

        # Create data where experts 0 and 1 frequently appear together
        expert_ids = np.array([
            [0, 1], [0, 1], [0, 1],  # 0 and 1 together 3 times
            [2, 3], [2, 3],  # 2 and 3 together 2 times
            [0, 2], [1, 3],  # Mixed pairs
        ])
        profiler.record_routing(0, expert_ids)

        cooc = profiler.cooccurrence

        # Check that 0,1 pair has high co-occurrence
        assert cooc.cooccurrence_matrix[0, 1] >= 3
        assert cooc.cooccurrence_matrix[1, 0] >= 3  # Symmetric

        # Check conditional probabilities exist
        assert cooc.conditional_probs is not None

    def test_hot_cold_dead_experts(self) -> None:
        profiler = MoERoutingProfiler(
            num_experts=8, num_layers=1, top_k=2, hot_threshold=1.5, cold_threshold=0.5
        )

        # Create highly skewed data
        # Experts 0, 1: heavily used (hot)
        # Experts 6, 7: never used (dead)
        expert_ids = []
        for _ in range(100):
            expert_ids.append([0, 1])
        for _ in range(50):
            expert_ids.append([2, 3])
        for _ in range(20):
            expert_ids.append([4, 5])

        profiler.record_routing(0, np.array(expert_ids))

        dead = profiler.get_dead_experts()
        assert 6 in dead
        assert 7 in dead

    def test_prefetch_recommendations(self) -> None:
        profiler = MoERoutingProfiler(num_experts=8, num_layers=1, top_k=2)

        # Create data where some experts are more popular
        expert_ids = []
        for _ in range(50):
            expert_ids.append([0, 1])
        for _ in range(30):
            expert_ids.append([2, 3])
        for _ in range(10):
            expert_ids.append([4, 5])
        for _ in range(5):
            expert_ids.append([6, 7])

        profiler.record_routing(0, np.array(expert_ids))

        recs = profiler.get_prefetch_recommendations(0, num_experts=4)

        # Most popular experts should be recommended first
        assert len(recs) == 4
        # Expert 0 or 1 should be in top recommendations
        assert 0 in recs or 1 in recs

    def test_generate_report(self) -> None:
        profiler = MoERoutingProfiler(num_experts=8, num_layers=2, top_k=2)

        rng = np.random.default_rng(42)
        for layer in range(2):
            expert_ids = rng.integers(0, 8, size=(50, 2))
            profiler.record_routing(layer, expert_ids)

        report = profiler.generate_report()

        # Check report structure
        assert "summary" in report
        assert "load_balance" in report
        assert "cooccurrence" in report
        assert "predictability" in report
        assert "per_layer" in report

        # Check summary values
        assert report["summary"]["num_experts"] == 8
        assert report["summary"]["num_layers"] == 2
        assert report["summary"]["total_tokens_profiled"] == 100

    def test_save_report_json_serializable(self) -> None:
        """Ensure report can be serialized to JSON without numpy type issues."""
        profiler = MoERoutingProfiler(num_experts=8, num_layers=2, top_k=2)

        rng = np.random.default_rng(42)
        for layer in range(2):
            expert_ids = rng.integers(0, 8, size=(50, 2))
            expert_probs = rng.random((50, 2))
            expert_probs /= expert_probs.sum(axis=1, keepdims=True)
            profiler.record_routing(layer, expert_ids, expert_probs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            profiler.save_report(path)

            # Verify the file can be read back
            with open(path) as f:
                loaded = json.load(f)

            assert loaded["summary"]["num_experts"] == 8


class TestRoutingPredictability:
    """Tests for routing predictability analysis."""

    def test_correlated_layers(self) -> None:
        """Test that correlated routing is detected."""
        profiler = MoERoutingProfiler(num_experts=8, num_layers=4, top_k=2)

        # Create identical routing across all layers (perfect correlation)
        rng = np.random.default_rng(42)
        base_routing = rng.integers(0, 8, size=(100, 2))

        for layer in range(4):
            profiler.record_routing(layer, base_routing.copy())

        pred = profiler.predictability

        # All layers should have high correlation
        assert pred.layer_correlations[0, 1] > 0.9
        assert pred.layer_correlations[0, 3] > 0.9


class TestSimulateRouting:
    """Tests for the routing simulation function."""

    @pytest.mark.parametrize(
        "model_name,expected_experts,expected_layers,expected_top_k",
        [
            ("mixtral", 8, 32, 2),
            ("glm47", 64, 40, 2),
            ("qwen3_30b", 128, 48, 8),
        ],
    )
    def test_simulate_known_models(
        self,
        model_name: str,
        expected_experts: int,
        expected_layers: int,
        expected_top_k: int,
    ) -> None:
        profiler = simulate_routing_for_model(model_name, num_samples=100)

        assert profiler.num_experts == expected_experts
        assert profiler.num_layers == expected_layers
        assert profiler.top_k == expected_top_k

        # Verify data was generated for all layers
        for layer_idx in range(expected_layers):
            assert len(profiler._expert_ids[layer_idx]) > 0

    def test_simulate_deterministic_with_seed(self) -> None:
        """Same seed should produce same results."""
        profiler1 = simulate_routing_for_model("mixtral", num_samples=50, seed=123)
        profiler2 = simulate_routing_for_model("mixtral", num_samples=50, seed=123)

        report1 = profiler1.generate_report()
        report2 = profiler2.generate_report()

        assert report1["summary"] == report2["summary"]

    def test_simulate_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            simulate_routing_for_model("nonexistent_model")
