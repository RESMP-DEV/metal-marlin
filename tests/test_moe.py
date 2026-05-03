"""Consolidated tests for active MoE functionality.

This module focuses on the current live surface:
- Token-to-expert grouping and dispatch
- Routing analysis and profiling
- Expert weight caching
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from metal_marlin.analysis.moe_routing import (
    ExpertCooccurrence,
    ExpertLoadStats,
    LayerRoutingProfile,
    MoERoutingProfiler,
    RoutingPredictability,
    simulate_routing_for_model,
)
from metal_marlin.expert_cache import (
    CacheEntry,
    ExpertCache,
    ExpertStats,
    LayerStats,
    TileCoordinator,
    TileKey,
    create_moe_cache,
)
from metal_marlin.kernels import moe_expert_gemm_fp4, moe_router_topk
from metal_marlin.moe_dispatch import compute_expert_load as torch_compute_expert_load
from metal_marlin.moe_dispatch import (
    compute_load_balancing_loss as torch_compute_load_balancing_loss,
)
from metal_marlin.moe_dispatch import (
    gather_for_experts,
    group_tokens_by_expert,
    group_tokens_by_expert_full,
    scatter_expert_outputs,
)

if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)

# ============================================================================
# PyTorch/Device Setup
# ============================================================================

TORCH_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ============================================================================
# Module Imports
# ============================================================================


# ============================================================================
# Dispatch Tests (PyTorch/MPS)
# ============================================================================


class TestGroupTokensByExpertTorch:
    """Tests for the core group_tokens_by_expert function (PyTorch)."""

    def test_basic_grouping(self):
        """Test basic token grouping with simple input."""
        logger.info("running test_basic_grouping")
        expert_ids = torch.tensor([[0, 2], [1, 2], [0, 1]], dtype=torch.int32, device=TORCH_DEVICE)
        num_experts = 3

        sorted_idx, offsets, inverse = group_tokens_by_expert(expert_ids, num_experts)

        total = expert_ids.numel()
        assert offsets[-1].item() == total

        expert_counts = np.diff(offsets.cpu().numpy())
        assert expert_counts[0] == 2
        assert expert_counts[1] == 2
        assert expert_counts[2] == 2

        expert_ids_flat = expert_ids.reshape(-1).cpu().numpy()
        sorted_experts = expert_ids_flat[sorted_idx.cpu().numpy()]
        assert np.all(sorted_experts[:-1] <= sorted_experts[1:])

        perm = sorted_idx.cpu().numpy()[inverse.cpu().numpy()]
        assert np.array_equal(perm, np.arange(total))

    def test_single_expert_per_token(self):
        """Test with top_k=1 (single expert per token)."""
        logger.info("running test_single_expert_per_token")
        expert_ids = torch.tensor([[0], [1], [0], [1]], dtype=torch.int32, device=TORCH_DEVICE)
        num_experts = 2

        sorted_idx, offsets, inverse = group_tokens_by_expert(expert_ids, num_experts)

        assert offsets[0].item() == 0
        assert offsets[1].item() == 2
        assert offsets[2].item() == 4

    def test_uneven_expert_distribution(self):
        """Test when experts have unequal load."""
        logger.info("running test_uneven_expert_distribution")
        expert_ids = torch.tensor(
            [[0, 1], [0, 2], [0, 1], [0, 2]], dtype=torch.int32, device=TORCH_DEVICE
        )
        num_experts = 3

        sorted_idx, offsets, inverse = group_tokens_by_expert(expert_ids, num_experts)

        expert_counts = np.diff(offsets.cpu().numpy())
        assert expert_counts[0] == 4
        assert expert_counts[1] == 2
        assert expert_counts[2] == 2

    def test_empty_experts(self):
        """Test when some experts receive no tokens."""
        logger.info("running test_empty_experts")
        expert_ids = torch.tensor([[0], [2]], dtype=torch.int32, device=TORCH_DEVICE)
        num_experts = 4

        sorted_idx, offsets, inverse = group_tokens_by_expert(expert_ids, num_experts)

        expert_counts = np.diff(offsets.cpu().numpy())
        assert expert_counts[0] == 1
        assert expert_counts[1] == 0
        assert expert_counts[2] == 1
        assert expert_counts[3] == 0

    def test_large_batch(self):
        """Test with larger batch size."""
        logger.info("running test_large_batch")
        batch_size = 128
        top_k = 4
        num_experts = 16

        np.random.seed(42)
        expert_ids_np = np.random.randint(0, num_experts, size=(batch_size, top_k))
        expert_ids = torch.tensor(expert_ids_np, dtype=torch.int32, device=TORCH_DEVICE)

        sorted_idx, offsets, inverse = group_tokens_by_expert(expert_ids, num_experts)

        total = batch_size * top_k
        assert offsets[-1].item() == total
        assert sorted_idx.shape[0] == total
        assert inverse.shape[0] == total

        expert_ids_flat = expert_ids.reshape(-1).cpu().numpy()
        sorted_experts = expert_ids_flat[sorted_idx.cpu().numpy()]
        assert np.all(sorted_experts[:-1] <= sorted_experts[1:])

        perm = sorted_idx.cpu().numpy()[inverse.cpu().numpy()]
        assert np.array_equal(perm, np.arange(total))


class TestMoEDispatchInfoTorch:
    """Tests for the full dispatch info structure (PyTorch)."""

    def test_dispatch_info_structure(self):
        """Test MoEDispatchInfo fields are correct."""
        logger.info("running test_dispatch_info_structure")
        expert_ids = torch.tensor([[0, 2], [1, 0], [2, 1]], dtype=torch.int32, device=TORCH_DEVICE)
        num_experts = 3

        info = group_tokens_by_expert_full(expert_ids, num_experts)

        assert info.num_tokens == 3
        assert info.top_k == 2
        assert info.num_experts == 3
        assert info.total_assignments == 6

        assert info.sorted_token_indices.shape == (6,)
        assert info.sorted_expert_indices.shape == (6,)
        assert info.expert_offsets.shape == (4,)
        assert info.inverse_indices.shape == (6,)

    def test_token_and_expert_indices_consistency(self):
        """Verify sorted_token_indices and sorted_expert_indices are consistent."""
        logger.info("running test_token_and_expert_indices_consistency")
        expert_ids = torch.tensor([[1, 0], [0, 2], [2, 1]], dtype=torch.int32, device=TORCH_DEVICE)
        num_experts = 3

        info = group_tokens_by_expert_full(expert_ids, num_experts)

        sorted_token_idx = info.sorted_token_indices.cpu().numpy()
        sorted_expert_slot = info.sorted_expert_indices.cpu().numpy()
        expert_ids_np = expert_ids.cpu().numpy()

        for i in range(info.total_assignments):
            token_idx = sorted_token_idx[i]
            expert_slot = sorted_expert_slot[i]
            expected_expert = expert_ids_np[token_idx, expert_slot]

            for e in range(num_experts):
                start = info.expert_offsets[e].item()
                end = info.expert_offsets[e + 1].item()
                if start <= i < end:
                    assert expected_expert == e
                    break


class TestGatherAndScatterTorch:
    """Tests for gather_for_experts and scatter_expert_outputs (PyTorch)."""

    def test_gather_for_experts(self):
        """Test activation gathering in expert-sorted order."""
        logger.info("running test_gather_for_experts")
        batch_size = 4
        hidden_dim = 8
        top_k = 2
        num_experts = 3

        activations = torch.arange(
            batch_size * hidden_dim, dtype=torch.float32, device=TORCH_DEVICE
        ).reshape(batch_size, hidden_dim)
        expert_ids = torch.tensor(
            [[0, 1], [1, 2], [0, 2], [1, 0]], dtype=torch.int32, device=TORCH_DEVICE
        )

        info = group_tokens_by_expert_full(expert_ids, num_experts)
        gathered = gather_for_experts(activations, info)

        assert gathered.shape == (batch_size * top_k, hidden_dim)

        sorted_token_idx = info.sorted_token_indices.cpu().numpy()
        activations_np = activations.cpu().numpy()
        gathered_np = gathered.cpu().numpy()

        for i in range(batch_size * top_k):
            expected_token = sorted_token_idx[i]
            np.testing.assert_array_equal(gathered_np[i], activations_np[expected_token])

    def test_scatter_expert_outputs(self):
        """Test output scattering and weighted combination."""
        logger.info("running test_scatter_expert_outputs")
        batch_size = 3
        out_dim = 4
        top_k = 2
        num_experts = 3

        expert_ids = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.int32, device=TORCH_DEVICE)
        expert_probs = torch.tensor(
            [[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]], dtype=torch.float32, device=TORCH_DEVICE
        )

        info = group_tokens_by_expert_full(expert_ids, num_experts)
        expert_outputs = torch.ones(
            (batch_size * top_k, out_dim), dtype=torch.float32, device=TORCH_DEVICE
        )

        result = scatter_expert_outputs(expert_outputs, expert_probs, info)

        expected = np.ones((batch_size, out_dim), dtype=np.float32)
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-5)

    def test_scatter_with_varying_outputs(self):
        """Test scatter with different expert outputs."""
        logger.info("running test_scatter_with_varying_outputs")
        batch_size = 2
        top_k = 2
        num_experts = 2

        expert_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=TORCH_DEVICE)
        expert_probs = torch.tensor(
            [[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32, device=TORCH_DEVICE
        )

        info = group_tokens_by_expert_full(expert_ids, num_experts)

        expert_outputs_list = []
        for i in range(batch_size * top_k):
            for e in range(num_experts):
                start = info.expert_offsets[e].item()
                end = info.expert_offsets[e + 1].item()
                if start <= i < end:
                    if e == 0:
                        expert_outputs_list.append([1.0, 0.0])
                    else:
                        expert_outputs_list.append([0.0, 1.0])
                    break

        expert_outputs = torch.tensor(expert_outputs_list, dtype=torch.float32, device=TORCH_DEVICE)
        result = scatter_expert_outputs(expert_outputs, expert_probs, info)

        expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-5)


class TestExpertLoadTorch:
    """Tests for expert load computation (PyTorch)."""

    def test_compute_expert_load(self):
        """Test expert load counting."""
        logger.info("running test_compute_expert_load")
        expert_ids = torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.int32, device=TORCH_DEVICE)
        num_experts = 3

        load = torch_compute_expert_load(expert_ids, num_experts)

        expected = np.array([2, 2, 2], dtype=np.int64)
        np.testing.assert_array_equal(load.cpu().numpy(), expected)

    def test_compute_expert_load_uneven(self):
        """Test with uneven expert distribution."""
        logger.info("running test_compute_expert_load_uneven")
        expert_ids = torch.tensor([[0, 0], [0, 1], [0, 0]], dtype=torch.int32, device=TORCH_DEVICE)
        num_experts = 3

        load = torch_compute_expert_load(expert_ids, num_experts)

        expected = np.array([5, 1, 0], dtype=np.int64)
        np.testing.assert_array_equal(load.cpu().numpy(), expected)


class TestLoadBalancingLossTorch:
    """Tests for auxiliary load balancing loss (PyTorch)."""

    def test_perfect_balance(self):
        """Test loss when load is perfectly balanced."""
        logger.info("running test_perfect_balance")
        batch_size = 4
        num_experts = 2

        expert_ids = torch.tensor([[0], [1], [0], [1]], dtype=torch.int32, device=TORCH_DEVICE)
        expert_probs_pre_topk = (
            torch.ones((batch_size, num_experts), dtype=torch.float32, device=TORCH_DEVICE)
            / num_experts
        )

        loss = torch_compute_load_balancing_loss(expert_probs_pre_topk, expert_ids, num_experts)

        assert abs(loss.item() - 1.0) < 1e-5

    def test_loss_with_skewed_probs(self):
        """Test loss is higher when probs match skewed routing."""
        logger.info("running test_loss_with_skewed_probs")
        num_experts = 2

        expert_ids = torch.tensor([[0], [0], [0], [0]], dtype=torch.int32, device=TORCH_DEVICE)
        expert_probs_pre_topk = torch.tensor(
            [[0.9, 0.1], [0.8, 0.2], [0.9, 0.1], [0.85, 0.15]],
            dtype=torch.float32,
            device=TORCH_DEVICE,
        )

        loss = torch_compute_load_balancing_loss(expert_probs_pre_topk, expert_ids, num_experts)

        assert loss.item() > 1.0


class TestEndToEndTorch:
    """End-to-end integration tests (PyTorch)."""

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
    def test_moe_router_topk_function(self):
        """Test moe_router_topk kernel function directly."""
        logger.info("running test_moe_router_topk_function")
        batch, hidden_dim, num_experts = 4, 128, 8
        hidden = torch.randn(batch, hidden_dim, device="mps")
        router_weights = torch.randn(hidden_dim, num_experts, device="mps")

        for top_k in [1, 2, 4]:
            ids, probs = moe_router_topk(hidden, router_weights, top_k=top_k)
            assert ids.shape == (batch, top_k)
            assert probs.shape == (batch, top_k)

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
    def test_moe_router_expert_gemm_pipeline(self):
        """Test full router -> expert GEMM pipeline."""
        logger.info("running test_moe_router_expert_gemm_pipeline")
        batch, hidden_dim, out_dim = 8, 256, 512
        num_experts, top_k = 8, 2

        # Input
        hidden = torch.randn(batch, hidden_dim, device="mps", dtype=torch.float16)
        router_weights = torch.randn(hidden_dim, num_experts, device="mps", dtype=torch.float16)

        # Simulated FP4 expert weights
        expert_weights = torch.randint(
            0,
            255,
            (num_experts, hidden_dim // 8, out_dim),
            dtype=torch.uint8,
            device="mps",
        )
        scales = torch.randn(
            num_experts,
            hidden_dim // 128,
            out_dim,
            device="mps",
            dtype=torch.float16,
        ).abs()

        # Router
        expert_ids, expert_probs = moe_router_topk(
            hidden.float(), router_weights.float(), top_k=top_k
        )

        assert expert_ids.shape == (batch, top_k)
        assert expert_probs.shape == (batch, top_k)
        assert (expert_ids >= 0).all() and (expert_ids < num_experts).all()
        assert torch.allclose(
            expert_probs.sum(dim=-1),
            torch.ones(batch, device="mps"),
            atol=1e-5,
        )

        # Expert GEMM
        output = moe_expert_gemm_fp4(hidden, expert_weights, scales, expert_ids, expert_probs)

        assert output.shape == (batch, out_dim)
        assert not output.isnan().any()

    def test_full_moe_dispatch_flow(self):
        """Test complete dispatch -> gather -> compute -> scatter flow."""
        logger.info("running test_full_moe_dispatch_flow")
        batch_size = 8
        hidden_dim = 16
        out_dim = 16
        top_k = 2
        num_experts = 4

        np.random.seed(123)
        expert_ids_np = np.random.randint(0, num_experts, size=(batch_size, top_k))
        expert_ids = torch.tensor(expert_ids_np, dtype=torch.int32, device=TORCH_DEVICE)

        expert_probs_np = np.random.rand(batch_size, top_k).astype(np.float32)
        expert_probs_np /= expert_probs_np.sum(axis=1, keepdims=True)
        expert_probs = torch.tensor(expert_probs_np, dtype=torch.float32, device=TORCH_DEVICE)

        torch.manual_seed(456)
        activations = torch.randn(
            (batch_size, hidden_dim), dtype=torch.float32, device=TORCH_DEVICE
        )

        info = group_tokens_by_expert_full(expert_ids, num_experts)
        gathered = gather_for_experts(activations, info)
        expert_outputs = gathered[:, :out_dim]
        result = scatter_expert_outputs(expert_outputs, expert_probs, info)

        assert result.shape == (batch_size, out_dim)

        expected = activations[:, :out_dim].cpu().numpy()
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)


# ============================================================================
# Routing Analysis Tests
# ============================================================================


class TestExpertLoadStatsAnalysis:
    """Tests for ExpertLoadStats dataclass."""

    def test_default_values(self) -> None:
        logger.info("running test_default_values")
        stats = ExpertLoadStats(expert_id=0)
        assert stats.expert_id == 0
        assert stats.total_activations == 0
        assert stats.activation_rate == 0.0
        assert stats.is_hot is False
        assert stats.is_cold is False
        assert stats.is_dead is False

    def test_primary_selection_rate_zero_activations(self) -> None:
        logger.info("running test_primary_selection_rate_zero_activations")
        stats = ExpertLoadStats(expert_id=0)
        assert stats.primary_selection_rate == 0.0

    def test_primary_selection_rate_with_data(self) -> None:
        logger.info("running test_primary_selection_rate_with_data")
        stats = ExpertLoadStats(
            expert_id=0,
            total_activations=100,
            rank_distribution={0: 60, 1: 40},
        )
        assert stats.primary_selection_rate == 0.6


class TestExpertCooccurrence:
    """Tests for ExpertCooccurrence dataclass."""

    def test_initialization(self) -> None:
        logger.info("running test_initialization")
        cooc = ExpertCooccurrence(num_experts=8)
        assert cooc.num_experts == 8
        assert cooc.cooccurrence_matrix.shape == (8, 8)
        assert cooc.conditional_probs is None
        assert cooc.top_pairs == []


class TestLayerRoutingProfile:
    """Tests for LayerRoutingProfile dataclass."""

    def test_empty_profile(self) -> None:
        logger.info("running test_empty_profile")
        profile = LayerRoutingProfile(layer_idx=0)
        assert profile.layer_idx == 0
        assert profile.total_tokens == 0
        assert profile.get_hot_experts() == []
        assert profile.get_cold_experts() == []
        assert profile.get_dead_experts() == []


class TestMoERoutingProfiler:
    """Tests for the main MoERoutingProfiler class."""

    def test_initialization(self) -> None:
        logger.info("running test_initialization")
        profiler = MoERoutingProfiler(
            num_experts=8,
            num_layers=4,
            top_k=2,
        )
        assert profiler.num_experts == 8
        assert profiler.num_layers == 4
        assert profiler.top_k == 2

    def test_record_routing_basic(self) -> None:
        logger.info("running test_record_routing_basic")
        profiler = MoERoutingProfiler(num_experts=8, num_layers=4, top_k=2)

        expert_ids = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
        expert_probs = np.array([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.8, 0.2]])

        profiler.record_routing(0, expert_ids, expert_probs)

        assert len(profiler._expert_ids[0]) == 1
        assert len(profiler._expert_probs[0]) == 1

    def test_record_routing_invalid_layer(self) -> None:
        logger.info("running test_record_routing_invalid_layer")
        profiler = MoERoutingProfiler(num_experts=8, num_layers=4, top_k=2)

        expert_ids = np.array([[0, 1], [2, 3]])

        with pytest.raises(ValueError):
            profiler.record_routing(-1, expert_ids)

        with pytest.raises(ValueError):
            profiler.record_routing(4, expert_ids)

    def test_layer_profiles_computation(self) -> None:
        logger.info("running test_layer_profiles_computation")
        profiler = MoERoutingProfiler(num_experts=8, num_layers=2, top_k=2)

        rng = np.random.default_rng(42)

        for layer in range(2):
            probs = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.08, 0.05, 0.02])
            probs /= probs.sum()

            expert_ids = []
            for _ in range(100):
                selected = rng.choice(8, size=2, replace=False, p=probs)
                expert_ids.append(selected)

            profiler.record_routing(layer, np.array(expert_ids))

        profiles = profiler.layer_profiles

        assert len(profiles) == 2
        assert profiles[0].total_tokens == 100
        assert profiles[0].active_experts > 0

    def test_cooccurrence_computation(self) -> None:
        logger.info("running test_cooccurrence_computation")
        profiler = MoERoutingProfiler(num_experts=4, num_layers=1, top_k=2)

        expert_ids = np.array(
            [
                [0, 1],
                [0, 1],
                [0, 1],
                [2, 3],
                [2, 3],
                [0, 2],
                [1, 3],
            ]
        )
        profiler.record_routing(0, expert_ids)

        cooc = profiler.cooccurrence

        assert cooc.cooccurrence_matrix[0, 1] >= 3
        assert cooc.cooccurrence_matrix[1, 0] >= 3
        assert cooc.conditional_probs is not None

    def test_hot_cold_dead_experts(self) -> None:
        logger.info("running test_hot_cold_dead_experts")
        profiler = MoERoutingProfiler(
            num_experts=8, num_layers=1, top_k=2, hot_threshold=1.5, cold_threshold=0.5
        )

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
        logger.info("running test_prefetch_recommendations")
        profiler = MoERoutingProfiler(num_experts=8, num_layers=1, top_k=2)

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

        assert len(recs) == 4
        assert 0 in recs or 1 in recs

    def test_generate_report(self) -> None:
        logger.info("running test_generate_report")
        profiler = MoERoutingProfiler(num_experts=8, num_layers=2, top_k=2)

        rng = np.random.default_rng(42)
        for layer in range(2):
            expert_ids = rng.integers(0, 8, size=(50, 2))
            profiler.record_routing(layer, expert_ids)

        report = profiler.generate_report()

        assert "summary" in report
        assert "load_balance" in report
        assert "cooccurrence" in report
        assert "predictability" in report
        assert "per_layer" in report

        assert report["summary"]["num_experts"] == 8
        assert report["summary"]["num_layers"] == 2
        assert report["summary"]["total_tokens_profiled"] == 100

    def test_save_report_json_serializable(self) -> None:
        """Ensure report can be serialized to JSON without numpy type issues."""
        logger.info("running test_save_report_json_serializable")
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

            with open(path) as f:
                loaded = json.load(f)

            assert loaded["summary"]["num_experts"] == 8


class TestRoutingPredictability:
    """Tests for routing predictability analysis."""

    def test_correlated_layers(self) -> None:
        """Test that correlated routing is detected."""
        logger.info("running test_correlated_layers")
        profiler = MoERoutingProfiler(num_experts=8, num_layers=4, top_k=2)

        rng = np.random.default_rng(42)
        base_routing = rng.integers(0, 8, size=(100, 2))

        for layer in range(4):
            profiler.record_routing(layer, base_routing.copy())

        pred = profiler.predictability

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
        logger.info("running test_simulate_known_models")
        profiler = simulate_routing_for_model(model_name, num_samples=100)

        assert profiler.num_experts == expected_experts
        assert profiler.num_layers == expected_layers
        assert profiler.top_k == expected_top_k

        for layer_idx in range(expected_layers):
            assert len(profiler._expert_ids[layer_idx]) > 0

    def test_simulate_deterministic_with_seed(self) -> None:
        """Same seed should produce same results."""
        logger.info("running test_simulate_deterministic_with_seed")
        profiler1 = simulate_routing_for_model("mixtral", num_samples=50, seed=123)
        profiler2 = simulate_routing_for_model("mixtral", num_samples=50, seed=123)

        report1 = profiler1.generate_report()
        report2 = profiler2.generate_report()

        assert report1["summary"] == report2["summary"]

    def test_simulate_unknown_model_raises(self) -> None:
        logger.info("running test_simulate_unknown_model_raises")
        with pytest.raises(ValueError, match="Unknown model"):
            simulate_routing_for_model("nonexistent_model")


# Expert Cache Tests (PyTorch)
# ============================================================================


class TestTileKey:
    """Tests for TileKey dataclass."""

    def test_creation(self):
        logger.info("running test_creation")
        key = TileKey(layer_idx=0, expert_id=5, tile_idx=10)
        assert key.layer_idx == 0
        assert key.expert_id == 5
        assert key.tile_idx == 10

    def test_equality(self):
        logger.info("running test_equality")
        key1 = TileKey(0, 5, 10)
        key2 = TileKey(0, 5, 10)
        key3 = TileKey(0, 5, 11)

        assert key1 == key2
        assert key1 != key3

    def test_hash(self):
        logger.info("running test_hash")
        key1 = TileKey(0, 5, 10)
        key2 = TileKey(0, 5, 10)

        assert hash(key1) == hash(key2)

        d = {key1: "value"}
        assert d[key2] == "value"


class TestCacheEntryClass:
    """Tests for CacheEntry dataclass."""

    def test_creation(self):
        logger.info("running test_creation")
        key = TileKey(0, 0, 0)
        data = torch.zeros((64, 64), device=TORCH_DEVICE)

        entry = CacheEntry(key=key, data=data, size_bytes=data.numel() * data.element_size())

        assert entry.key == key
        assert entry.access_count == 0
        assert entry.size_bytes == data.numel() * data.element_size()

    def test_touch(self):
        logger.info("running test_touch")
        key = TileKey(0, 0, 0)
        data = torch.zeros((64, 64), device=TORCH_DEVICE)
        entry = CacheEntry(key=key, data=data, size_bytes=data.numel() * data.element_size())

        initial_time = entry.last_access
        initial_count = entry.access_count

        time.sleep(0.01)
        entry.touch()

        assert entry.access_count == initial_count + 1
        assert entry.last_access > initial_time


class TestExpertStatsCache:
    """Tests for ExpertStats dataclass (cache module)."""

    def test_activation_rate(self):
        logger.info("running test_activation_rate")
        stats = ExpertStats(expert_id=0)

        assert stats.activation_rate == 0.0

        stats.record_batch(100)
        stats.record_activation(10)

        assert stats.activation_rate == pytest.approx(0.1, rel=1e-3)

    def test_recent_rate(self):
        logger.info("running test_recent_rate")
        stats = ExpertStats(expert_id=0)

        assert stats.recent_rate == 0.0

        for _ in range(10):
            stats.record_activation(5)

        assert stats.recent_rate == pytest.approx(5.0, rel=1e-3)

    def test_recent_window_limit(self):
        logger.info("running test_recent_window_limit")
        stats = ExpertStats(expert_id=0)

        for i in range(150):
            stats.record_activation(1)

        assert len(stats.recent_window) == 100


class TestLayerStatsCache:
    """Tests for LayerStats dataclass (cache module)."""

    def test_hit_rate(self):
        logger.info("running test_hit_rate")
        stats = LayerStats(layer_idx=0)

        assert stats.hit_rate == 0.0

        stats.cache_hits = 70
        stats.cache_misses = 30

        assert stats.hit_rate == pytest.approx(0.7, rel=1e-3)

    def test_get_expert_stats(self):
        logger.info("running test_get_expert_stats")
        stats = LayerStats(layer_idx=0)

        expert_stats = stats.get_expert_stats(5)
        assert expert_stats.expert_id == 5

        expert_stats2 = stats.get_expert_stats(5)
        assert expert_stats2 is expert_stats

    def test_get_hot_experts(self):
        logger.info("running test_get_hot_experts")
        stats = LayerStats(layer_idx=0)

        for eid in range(10):
            expert_stats = stats.get_expert_stats(eid)
            expert_stats.record_activation(eid)

        hot = stats.get_hot_experts(threshold=0.5, top_k=3)

        assert len(hot) == 3
        assert hot == [9, 8, 7]


class TestExpertCacheModule:
    """Tests for ExpertCache class."""

    def test_basic_get_and_cache(self):
        logger.info("running test_basic_get_and_cache")
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        call_count = 0

        def dequant_fn():
            logger.info("dequant_fn called")
            nonlocal call_count
            call_count += 1
            return torch.ones((64, 64), dtype=torch.float16, device=TORCH_DEVICE)

        tile1 = cache.get_expert_tile(0, 0, 0, dequant_fn)
        assert call_count == 1
        assert tile1.shape == (64, 64)

        tile2 = cache.get_expert_tile(0, 0, 0, dequant_fn)
        assert call_count == 1
        assert torch.equal(tile1, tile2)

    def test_lru_eviction(self):
        """Test LRU eviction."""
        logger.info("running test_lru_eviction")
        tile_size = 64 * 64 * 2
        cache_size_mb = (tile_size * 2.5) / (1024 * 1024)
        cache = ExpertCache(
            num_experts=8, num_layers=2, cache_size_mb=int(cache_size_mb * 1000) / 1000
        )

        def make_tile(expert_id):
            logger.debug("make_tile called with expert_id=%s", expert_id)
            return torch.full((64, 64), expert_id, dtype=torch.float16, device=TORCH_DEVICE)

        cache.get_expert_tile(0, 0, 0, lambda: make_tile(0))
        cache.get_expert_tile(0, 1, 0, lambda: make_tile(1))
        cache.get_expert_tile(0, 2, 0, lambda: make_tile(2))

        assert cache._total_evictions > 0

    def test_hit_rate(self):
        logger.info("running test_hit_rate")
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        def dequant_fn():
            logger.info("dequant_fn called")
            return torch.ones((64, 64), dtype=torch.float16, device=TORCH_DEVICE)

        cache.get_expert_tile(0, 0, 0, dequant_fn)
        cache.get_expert_tile(0, 0, 0, dequant_fn)
        cache.get_expert_tile(0, 0, 0, dequant_fn)
        cache.get_expert_tile(0, 1, 0, dequant_fn)
        cache.get_expert_tile(0, 0, 0, dequant_fn)

        assert cache.hit_rate == pytest.approx(0.6, rel=1e-3)

    def test_invalidate_expert(self):
        logger.info("running test_invalidate_expert")
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        def dequant_fn():
            logger.info("dequant_fn called")
            return torch.ones((64, 64), dtype=torch.float16, device=TORCH_DEVICE)

        cache.get_expert_tile(0, 0, 0, dequant_fn)
        cache.get_expert_tile(0, 0, 1, dequant_fn)
        cache.get_expert_tile(0, 1, 0, dequant_fn)

        assert cache.num_entries == 3

        evicted = cache.invalidate_expert(layer_idx=0, expert_id=0)
        assert evicted == 2
        assert cache.num_entries == 1

    def test_invalidate_layer(self):
        logger.info("running test_invalidate_layer")
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        def dequant_fn():
            logger.info("dequant_fn called")
            return torch.ones((64, 64), dtype=torch.float16, device=TORCH_DEVICE)

        cache.get_expert_tile(0, 0, 0, dequant_fn)
        cache.get_expert_tile(0, 1, 0, dequant_fn)
        cache.get_expert_tile(1, 0, 0, dequant_fn)

        assert cache.num_entries == 3

        evicted = cache.invalidate_layer(layer_idx=0)
        assert evicted == 2
        assert cache.num_entries == 1

    def test_clear(self):
        logger.info("running test_clear")
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        def dequant_fn():
            logger.info("dequant_fn called")
            return torch.ones((64, 64), dtype=torch.float16, device=TORCH_DEVICE)

        cache.get_expert_tile(0, 0, 0, dequant_fn)
        cache.get_expert_tile(0, 1, 0, dequant_fn)

        assert cache.num_entries == 2

        cache.clear()

        assert cache.num_entries == 0
        assert cache.size_mb == 0.0

    def test_resize(self):
        logger.info("running test_resize")
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        def dequant_fn():
            logger.info("dequant_fn called")
            return torch.ones((64, 64), dtype=torch.float16, device=TORCH_DEVICE)

        for i in range(10):
            cache.get_expert_tile(0, i, 0, dequant_fn)

        initial_entries = cache.num_entries

        evicted = cache.resize(0)
        assert evicted == initial_entries
        assert cache.num_entries == 0

    def test_record_expert_activation(self):
        logger.info("running test_record_expert_activation")
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        expert_ids = torch.tensor([[0, 1], [0, 2], [1, 3]], dtype=torch.int32, device=TORCH_DEVICE)
        cache.record_expert_activation(layer_idx=0, expert_ids=expert_ids)

        stats = cache._layer_stats[0]
        assert stats.get_expert_stats(0).activation_count == 2
        assert stats.get_expert_stats(1).activation_count == 2
        assert stats.get_expert_stats(2).activation_count == 1
        assert stats.get_expert_stats(3).activation_count == 1

    def test_get_prefetch_candidates(self):
        logger.info("running test_get_prefetch_candidates")
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1, prefetch_k=3)

        for _ in range(10):
            expert_ids = torch.tensor([[0, 1], [0, 2]], dtype=torch.int32, device=TORCH_DEVICE)
            cache.record_expert_activation(0, expert_ids)

        candidates = cache.get_prefetch_candidates(0)
        assert 0 in candidates

    def test_get_stats(self):
        logger.info("running test_get_stats")
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        def dequant_fn():
            logger.info("dequant_fn called")
            return torch.ones((64, 64), dtype=torch.float16, device=TORCH_DEVICE)

        cache.get_expert_tile(0, 0, 0, dequant_fn)
        cache.get_expert_tile(0, 0, 0, dequant_fn)

        stats = cache.get_stats()

        assert "global" in stats
        assert "memory" in stats
        assert "config" in stats
        assert "per_layer" in stats

        assert stats["global"]["total_hits"] == 1
        assert stats["global"]["total_misses"] == 1
        assert stats["config"]["num_experts"] == 8

    def test_thread_safety_cache_operations(self):
        """Test that cache internal operations are thread-safe."""
        logger.info("running test_thread_safety_cache_operations")
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        # Use CPU for thread safety test - MPS doesn't handle concurrent
        # .item() calls well across threads
        precomputed_tiles = {
            i: torch.full((64, 64), i, dtype=torch.float16, device="cpu") for i in range(4)
        }

        results = []
        lock = threading.Lock()

        def worker(expert_id):
            logger.debug("worker called with expert_id=%s", expert_id)
            tile = precomputed_tiles[expert_id]

            def dequant_fn():
                logger.info("dequant_fn called")
                return tile

            for _ in range(10):
                cached_tile = cache.get_expert_tile(0, expert_id, 0, dequant_fn)
                with lock:
                    results.append((expert_id, cached_tile[0, 0].item()))

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            for f in futures:
                f.result()

        for expert_id, value in results:
            assert value == expert_id

    def test_repr(self):
        logger.info("running test_repr")
        cache = ExpertCache(num_experts=64, num_layers=28, cache_size_mb=512)
        repr_str = repr(cache)

        assert "ExpertCache" in repr_str
        assert "num_experts=64" in repr_str
        assert "num_layers=28" in repr_str


class TestTileCoordinator:
    """Tests for TileCoordinator class."""

    def test_basic_properties(self):
        logger.info("running test_basic_properties")
        coord = TileCoordinator(
            weight_shape=(256, 128),
            tile_shape=(64, 64),
        )

        assert coord.num_row_tiles == 4
        assert coord.num_col_tiles == 2
        assert coord.num_tiles == 8

    def test_tile_to_coords(self):
        logger.info("running test_tile_to_coords")
        coord = TileCoordinator(
            weight_shape=(256, 128),
            tile_shape=(64, 64),
        )

        assert coord.tile_to_coords(0) == (0, 0)
        assert coord.tile_to_coords(1) == (0, 1)
        assert coord.tile_to_coords(2) == (1, 0)
        assert coord.tile_to_coords(3) == (1, 1)

    def test_coords_to_tile(self):
        logger.info("running test_coords_to_tile")
        coord = TileCoordinator(
            weight_shape=(256, 128),
            tile_shape=(64, 64),
        )

        assert coord.coords_to_tile(0, 0) == 0
        assert coord.coords_to_tile(0, 1) == 1
        assert coord.coords_to_tile(1, 0) == 2
        assert coord.coords_to_tile(1, 1) == 3

    def test_roundtrip(self):
        logger.info("running test_roundtrip")
        coord = TileCoordinator(
            weight_shape=(256, 128),
            tile_shape=(64, 64),
        )

        for tile_idx in range(coord.num_tiles):
            row, col = coord.tile_to_coords(tile_idx)
            assert coord.coords_to_tile(row, col) == tile_idx

    def test_tile_bounds(self):
        logger.info("running test_tile_bounds")
        coord = TileCoordinator(
            weight_shape=(256, 128),
            tile_shape=(64, 64),
        )

        r_start, r_end, c_start, c_end = coord.tile_bounds(0)
        assert (r_start, r_end, c_start, c_end) == (0, 64, 0, 64)

        r_start, r_end, c_start, c_end = coord.tile_bounds(1)
        assert (r_start, r_end, c_start, c_end) == (0, 64, 64, 128)

    def test_tile_bounds_with_padding(self):
        """Test non-aligned dimensions."""
        logger.info("running test_tile_bounds_with_padding")
        coord = TileCoordinator(
            weight_shape=(100, 100),
            tile_shape=(64, 64),
        )

        last_tile = coord.num_tiles - 1
        r_start, r_end, c_start, c_end = coord.tile_bounds(last_tile)
        assert r_end == 100
        assert c_end == 100

    def test_all_tile_indices(self):
        logger.info("running test_all_tile_indices")
        coord = TileCoordinator(
            weight_shape=(256, 128),
            tile_shape=(64, 64),
        )

        indices = coord.all_tile_indices()
        assert indices == list(range(8))

    def test_tiles_for_output_range(self):
        logger.info("running test_tiles_for_output_range")
        coord = TileCoordinator(
            weight_shape=(256, 128),
            tile_shape=(64, 64),
        )

        tiles = coord.tiles_for_output_range(0, 64)
        assert tiles == [0, 1]

        tiles = coord.tiles_for_output_range(64, 128)
        assert tiles == [2, 3]


class TestCreateMoeCache:
    """Tests for create_moe_cache helper function."""

    def test_glm4_config(self):
        logger.info("running test_glm4_config")
        config = {
            "num_hidden_layers": 28,
            "num_experts": 64,
        }

        cache = create_moe_cache(config, cache_size_mb=256)

        assert cache.num_layers == 28
        assert cache.num_experts == 64
        assert cache.cache_size_bytes == 256 * 1024 * 1024

    def test_mixtral_config(self):
        logger.info("running test_mixtral_config")
        config = {
            "n_layer": 32,
            "num_local_experts": 8,
        }

        cache = create_moe_cache(config, cache_size_mb=512)

        assert cache.num_layers == 32
        assert cache.num_experts == 8

    def test_default_values(self):
        logger.info("running test_default_values")
        config = {}

        cache = create_moe_cache(config)

        assert cache.num_layers == 32
        assert cache.num_experts == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
