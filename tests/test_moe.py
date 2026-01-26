"""Consolidated tests for MoE (Mixture of Experts) functionality.

This module contains tests for:
- Token-to-expert grouping and dispatch
- Predictive expert prefetching
- Routing analysis and profiling
- Token dispatcher
- Expert weight caching
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

# ============================================================================
# PyTorch/Device Setup
# ============================================================================

TORCH_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ============================================================================
# Module Imports
# ============================================================================

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
from metal_marlin.moe.token_dispatcher import (
    DispatchInfo,
    DispatchStats,
    TokenDispatcher,
    combine_expert_outputs,
    dispatch_to_experts,
    gather_tokens_for_expert,
)
from metal_marlin.moe.token_dispatcher import (
    compute_expert_load as dispatcher_compute_expert_load,
)
from metal_marlin.moe.token_dispatcher import (
    compute_load_balancing_loss as dispatcher_compute_load_balancing_loss,
)
from metal_marlin.moe.token_dispatcher import (
    group_tokens_by_expert as dispatcher_group_tokens_by_expert,
)
from metal_marlin.moe_dispatch import (
    compute_expert_load as torch_compute_expert_load,
)
from metal_marlin.moe_dispatch import (
    compute_load_balancing_loss as torch_compute_load_balancing_loss,
)
from metal_marlin.moe_dispatch import (
    gather_for_experts,
    group_tokens_by_expert,
    group_tokens_by_expert_full,
    scatter_expert_outputs,
)

# ============================================================================
# Dispatch Tests (PyTorch/MPS)
# ============================================================================


class TestGroupTokensByExpertTorch:
    """Tests for the core group_tokens_by_expert function (PyTorch)."""

    def test_basic_grouping(self):
        """Test basic token grouping with simple input."""
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
        expert_ids = torch.tensor([[0], [1], [0], [1]], dtype=torch.int32, device=TORCH_DEVICE)
        num_experts = 2

        sorted_idx, offsets, inverse = group_tokens_by_expert(expert_ids, num_experts)

        assert offsets[0].item() == 0
        assert offsets[1].item() == 2
        assert offsets[2].item() == 4

    def test_uneven_expert_distribution(self):
        """Test when experts have unequal load."""
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
        expert_ids = torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.int32, device=TORCH_DEVICE)
        num_experts = 3

        load = torch_compute_expert_load(expert_ids, num_experts)

        expected = np.array([2, 2, 2], dtype=np.int64)
        np.testing.assert_array_equal(load.cpu().numpy(), expected)

    def test_compute_expert_load_uneven(self):
        """Test with uneven expert distribution."""
        expert_ids = torch.tensor([[0, 0], [0, 1], [0, 0]], dtype=torch.int32, device=TORCH_DEVICE)
        num_experts = 3

        load = torch_compute_expert_load(expert_ids, num_experts)

        expected = np.array([5, 1, 0], dtype=np.int64)
        np.testing.assert_array_equal(load.cpu().numpy(), expected)


class TestLoadBalancingLossTorch:
    """Tests for auxiliary load balancing loss (PyTorch)."""

    def test_perfect_balance(self):
        """Test loss when load is perfectly balanced."""
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

    def test_full_moe_dispatch_flow(self):
        """Test complete dispatch -> gather -> compute -> scatter flow."""
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
# Prefetch Tests (PyTorch)
# ============================================================================


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
        assert history.get_last_experts() == [9]

    def test_frequency_counts_no_decay(self):
        """Test frequency counting without decay."""
        history = RoutingHistory(layer_idx=0, window_size=10)

        history.record([1, 2])
        history.record([1, 3])
        history.record([1, 2])

        counts = history.get_frequency_counts(decay=1.0)

        assert counts[1] == 3
        assert counts[2] == 2
        assert counts[3] == 1

    def test_frequency_counts_with_decay(self):
        """Test frequency counting with temporal decay."""
        history = RoutingHistory(layer_idx=0, window_size=10)

        history.record([5])
        history.record([1])
        history.record([1])

        counts = history.get_frequency_counts(decay=0.5)

        assert counts[1] > counts[5]

    def test_empty_history(self):
        """Test behavior with empty history."""
        history = RoutingHistory(layer_idx=0, window_size=10)

        assert history.get_last_experts() == []
        assert history.get_frequency_counts() == {}


class TestPredictNextExperts:
    """Tests for predict_next_experts heuristic."""

    def test_repeat_strategy(self):
        """Test REPEAT strategy returns same experts."""
        current = torch.tensor([3, 7], device=TORCH_DEVICE)

        predicted = predict_next_experts(
            current_routing=current,
            strategy=PrefetchStrategy.REPEAT,
            prefetch_k=2,
        )

        assert predicted == [3, 7]

    def test_repeat_strategy_truncates(self):
        """Test REPEAT strategy respects prefetch_k."""
        current = torch.tensor([1, 2, 3, 4], device=TORCH_DEVICE)

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
        history.record([3, 5])

        predicted = predict_next_experts(
            current_routing=[3, 5],
            history=history,
            strategy=PrefetchStrategy.HISTORY,
            prefetch_k=2,
        )

        assert 5 in predicted

    def test_top_k_recency_strategy(self):
        """Test TOP_K_RECENCY blends current and history."""
        history = RoutingHistory(layer_idx=0, window_size=10)
        history.record([10, 20])
        history.record([10, 30])

        current = [1, 2]

        predicted = predict_next_experts(
            current_routing=current,
            history=history,
            strategy=PrefetchStrategy.TOP_K_RECENCY,
            prefetch_k=4,
            decay_factor=0.9,
        )

        assert 1 in predicted or 2 in predicted

    def test_fallback_with_no_history(self):
        """Test strategies fall back gracefully without history."""
        current = torch.tensor([5, 10], device=TORCH_DEVICE)

        for strategy in PrefetchStrategy:
            predicted = predict_next_experts(
                current_routing=current,
                history=None,
                strategy=strategy,
                prefetch_k=2,
            )
            assert len(predicted) <= 2


class TestPrefetchStats:
    """Tests for prefetch statistics tracking."""

    def test_prediction_accuracy(self):
        """Test prediction accuracy calculation."""
        stats = PrefetchStats(layer_idx=0)

        stats.record_prediction([1, 2], [1, 3])
        stats.record_prediction([4, 5], [6, 7])
        stats.record_prediction([8, 9], [9, 10])

        assert stats.predictions_made == 3
        assert stats.predictions_correct == 2
        assert stats.prediction_accuracy == pytest.approx(2 / 3)

    def test_expert_hit_rate(self):
        """Test expert-level hit rate."""
        stats = PrefetchStats(layer_idx=0)

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
        """Test submitting and waiting for load."""
        loaded = []

        def mock_load():
            time.sleep(0.01)
            loaded.append(True)
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
            return np.zeros((10, 10))

        with AsyncExpertLoader(num_threads=1) as loader:
            loader.submit(layer_idx=0, expert_id=1, load_fn=mock_load)
            loader.submit(layer_idx=0, expert_id=1, load_fn=mock_load)
            loader.wait_all()

        assert call_count[0] == 1


class TestAsyncLoadExperts:
    """Tests for async_load_experts convenience function."""

    def test_creates_loader_if_none(self):
        """Test that function creates loader if not provided."""
        calls = []

        def load_fn(layer, expert):
            calls.append((layer, expert))
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

        prefetcher.record_routing(layer_idx=0, expert_ids=[3, 7])
        prefetcher.record_routing(layer_idx=0, expert_ids=[3, 8])

        predicted = prefetcher.predict_next_experts(layer_idx=0)

        assert 3 in predicted

    def test_step_combined_operation(self):
        """Test step() does record + predict + prefetch."""
        loads = []

        def mock_load(layer, expert):
            loads.append((layer, expert))
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
        assert len(loads) > 0

    def test_stats_collection(self):
        """Test statistics collection."""
        prefetcher = ExpertPrefetcher(
            num_experts=64,
            num_layers=2,
            config=PrefetchConfig(enable_stats=True),
        )

        prefetcher.record_routing(0, [1, 2])
        prefetcher.predict_next_experts(0)
        prefetcher.record_routing(0, [1, 3])

        stats = prefetcher.get_stats()

        assert "config" in stats
        assert "per_layer" in stats
        assert "summary" in stats

    def test_clear_history(self):
        """Test clearing history."""
        prefetcher = ExpertPrefetcher(num_experts=64, num_layers=2)

        prefetcher.record_routing(0, [1, 2])
        prefetcher.record_routing(1, [3, 4])

        prefetcher.clear_history(layer_idx=0)
        assert len(prefetcher._history[0].history) == 0
        assert len(prefetcher._history[1].history) == 1

        prefetcher.clear_history()
        assert len(prefetcher._history[0].history) == 0
        assert len(prefetcher._history[1].history) == 0


class TestExpertLRUCachePrefetch:
    """Tests for expert weight LRU cache (from prefetch module)."""

    def test_put_and_get(self):
        """Test basic put/get."""
        cache = ExpertLRUCache(max_size_mb=10)

        weights = torch.ones((100, 100), device=TORCH_DEVICE)
        cache.put(layer_idx=0, expert_id=5, weights=weights)

        retrieved = cache.get(layer_idx=0, expert_id=5)
        assert retrieved is not None
        assert torch.equal(retrieved, weights)

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = ExpertLRUCache(max_size_mb=10)

        assert cache.get(layer_idx=0, expert_id=99) is None

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = ExpertLRUCache(max_size_mb=1)

        small_weights = torch.zeros((128, 1024), dtype=torch.float32, device=TORCH_DEVICE)

        cache.put(0, 1, small_weights)
        cache.put(0, 2, small_weights)
        cache.put(0, 3, small_weights)

        assert cache.get(0, 1) is None
        assert cache.get(0, 3) is not None

    def test_get_or_load(self):
        """Test get_or_load caches on miss."""
        cache = ExpertLRUCache(max_size_mb=10)
        call_count = [0]

        def loader():
            call_count[0] += 1
            return torch.ones((10, 10), device=TORCH_DEVICE)

        w1 = cache.get_or_load(0, 5, loader)
        assert call_count[0] == 1

        w2 = cache.get_or_load(0, 5, loader)
        assert call_count[0] == 1
        assert torch.equal(w1, w2)

    def test_invalidate_single(self):
        """Test invalidating single expert."""
        cache = ExpertLRUCache(max_size_mb=10)
        weights = torch.ones((10, 10), device=TORCH_DEVICE)

        cache.put(0, 1, weights)
        cache.put(0, 2, weights)

        evicted = cache.invalidate(layer_idx=0, expert_id=1)

        assert evicted == 1
        assert cache.get(0, 1) is None
        assert cache.get(0, 2) is not None

    def test_invalidate_layer(self):
        """Test invalidating entire layer."""
        cache = ExpertLRUCache(max_size_mb=10)
        weights = torch.ones((10, 10), device=TORCH_DEVICE)

        cache.put(0, 1, weights)
        cache.put(0, 2, weights)
        cache.put(1, 1, weights)

        evicted = cache.invalidate(layer_idx=0)

        assert evicted == 2
        assert cache.get(0, 1) is None
        assert cache.get(0, 2) is None
        assert cache.get(1, 1) is not None

    def test_hit_rate(self):
        """Test hit rate calculation."""
        cache = ExpertLRUCache(max_size_mb=10)
        cache.put(0, 1, torch.ones((10, 10), device=TORCH_DEVICE))

        cache.get(0, 1)
        cache.get(0, 1)
        cache.get(0, 99)

        assert cache.hit_rate == pytest.approx(2 / 3)


class TestPrefetchIntegration:
    """Integration tests for the prefetch system."""

    def test_end_to_end_prefetch(self):
        """Test complete prefetch workflow."""
        num_experts = 8
        num_layers = 2

        load_log: list[tuple[int, int]] = []

        def load_expert(layer_idx: int, expert_id: int) -> np.ndarray:
            load_log.append((layer_idx, expert_id))
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
            for token_idx in range(10):
                for layer_idx in range(num_layers):
                    base_expert = (token_idx + layer_idx) % num_experts
                    expert_ids = [base_expert, (base_expert + 1) % num_experts]
                    prefetcher.step(layer_idx, expert_ids)

            prefetcher.wait_prefetch()

        assert len(load_log) > 0

        stats = prefetcher.get_stats()
        assert stats["summary"]["total_predictions"] > 0


# ============================================================================
# Routing Analysis Tests
# ============================================================================


class TestExpertLoadStatsAnalysis:
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

        expert_ids = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
        expert_probs = np.array([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.8, 0.2]])

        profiler.record_routing(0, expert_ids, expert_probs)

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
        profiler = simulate_routing_for_model(model_name, num_samples=100)

        assert profiler.num_experts == expected_experts
        assert profiler.num_layers == expected_layers
        assert profiler.top_k == expected_top_k

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


# ============================================================================
# Token Dispatcher Tests (PyTorch)
# ============================================================================


class TestGroupTokensByExpertDispatcher:
    """Tests for the group_tokens_by_expert function (dispatcher module)."""

    def test_basic_grouping(self):
        """Test basic token grouping with simple input."""
        expert_ids = torch.tensor([[0, 2], [1, 2], [0, 1]], dtype=torch.int32, device=TORCH_DEVICE)
        num_experts = 3

        info = dispatcher_group_tokens_by_expert(expert_ids, num_experts)

        assert info.num_tokens == 3
        assert info.top_k == 2
        assert info.num_experts == 3
        assert info.total_assignments == 6

        assert info.sorted_token_indices.shape == (6,)
        assert info.sorted_expert_slots.shape == (6,)
        assert info.expert_offsets.shape == (4,)
        assert info.inverse_indices.shape == (6,)

        assert info.expert_offsets[-1].item() == 6

        expert_counts = np.diff(info.expert_offsets.cpu().numpy())
        assert expert_counts[0] == 2
        assert expert_counts[1] == 2
        assert expert_counts[2] == 2

    def test_single_expert_per_token(self):
        """Test with top_k=1 (single expert per token)."""
        expert_ids = torch.tensor([[0], [1], [0], [1]], dtype=torch.int32, device=TORCH_DEVICE)
        num_experts = 2

        info = dispatcher_group_tokens_by_expert(expert_ids, num_experts)

        assert info.expert_offsets[0].item() == 0
        assert info.expert_offsets[1].item() == 2
        assert info.expert_offsets[2].item() == 4

    def test_uneven_distribution(self):
        """Test when experts have unequal load."""
        expert_ids = torch.tensor(
            [[0, 1], [0, 2], [0, 1], [0, 2]], dtype=torch.int32, device=TORCH_DEVICE
        )
        num_experts = 3

        info = dispatcher_group_tokens_by_expert(expert_ids, num_experts)

        expert_counts = np.diff(info.expert_offsets.cpu().numpy())
        assert expert_counts[0] == 4
        assert expert_counts[1] == 2
        assert expert_counts[2] == 2

    def test_empty_experts(self):
        """Test when some experts receive no tokens."""
        expert_ids = torch.tensor([[0], [2]], dtype=torch.int32, device=TORCH_DEVICE)
        num_experts = 4

        info = dispatcher_group_tokens_by_expert(expert_ids, num_experts)

        expert_counts = np.diff(info.expert_offsets.cpu().numpy())
        assert expert_counts[0] == 1
        assert expert_counts[1] == 0
        assert expert_counts[2] == 1
        assert expert_counts[3] == 0

    def test_inverse_indices_correctness(self):
        """Verify inverse indices correctly restore order."""
        expert_ids = torch.tensor([[1, 0], [0, 2], [2, 1]], dtype=torch.int32, device=TORCH_DEVICE)
        num_experts = 3

        info = dispatcher_group_tokens_by_expert(expert_ids, num_experts)

        total = info.total_assignments
        assert len(np.unique(info.inverse_indices.cpu().numpy())) == total

    def test_large_batch(self):
        """Test with larger batch size."""
        batch_size = 128
        top_k = 4
        num_experts = 16

        np.random.seed(42)
        expert_ids_np = np.random.randint(0, num_experts, size=(batch_size, top_k))
        expert_ids = torch.tensor(expert_ids_np, dtype=torch.int32, device=TORCH_DEVICE)

        info = dispatcher_group_tokens_by_expert(expert_ids, num_experts)

        total = batch_size * top_k
        assert info.expert_offsets[-1].item() == total
        assert info.sorted_token_indices.shape[0] == total
        assert info.inverse_indices.shape[0] == total


class TestGatherTokensForExpertDispatcher:
    """Tests for gathering activations for specific experts."""

    def test_gather_for_single_expert(self):
        """Test gathering activations for one expert."""
        batch_size = 4
        hidden_dim = 8
        num_experts = 3

        activations = torch.arange(
            batch_size * hidden_dim, dtype=torch.float32, device=TORCH_DEVICE
        ).reshape(batch_size, hidden_dim)
        expert_ids = torch.tensor(
            [[0, 1], [1, 2], [0, 2], [1, 0]], dtype=torch.int32, device=TORCH_DEVICE
        )

        info = dispatcher_group_tokens_by_expert(expert_ids, num_experts)

        gathered = gather_tokens_for_expert(activations, info, expert_id=0)

        assert gathered.shape[0] == 3
        assert gathered.shape[1] == hidden_dim

    def test_gather_empty_expert(self):
        """Test gathering for expert with no tokens."""
        activations = torch.ones((4, 8), dtype=torch.float32, device=TORCH_DEVICE)
        expert_ids = torch.tensor([[0], [0], [0], [0]], dtype=torch.int32, device=TORCH_DEVICE)
        num_experts = 3

        info = dispatcher_group_tokens_by_expert(expert_ids, num_experts)

        gathered = gather_tokens_for_expert(activations, info, expert_id=1)

        assert gathered.shape[0] == 0
        assert gathered.shape[1] == 8


class TestDispatchToExperts:
    """Tests for batched expert dispatch."""

    def test_dispatch_simple(self):
        """Test basic dispatch with identity experts."""
        batch_size = 4
        hidden_dim = 8
        out_dim = 8
        top_k = 2
        num_experts = 3

        activations = torch.ones((batch_size, hidden_dim), dtype=torch.float32, device=TORCH_DEVICE)
        expert_ids = torch.tensor(
            [[0, 1], [1, 2], [0, 2], [1, 0]], dtype=torch.int32, device=TORCH_DEVICE
        )

        info = dispatcher_group_tokens_by_expert(expert_ids, num_experts)

        def expert_forward(x: torch.Tensor, expert_id: int) -> torch.Tensor:
            return x[:, :out_dim]

        outputs = dispatch_to_experts(activations, info, expert_forward)

        assert outputs.shape == (batch_size * top_k, out_dim)
        np.testing.assert_allclose(
            outputs.cpu().numpy(), np.ones((batch_size * top_k, out_dim)), rtol=1e-5
        )

    def test_dispatch_with_expert_specific_output(self):
        """Test dispatch where each expert produces different output."""
        batch_size = 2
        hidden_dim = 4
        num_experts = 2

        activations = torch.ones((batch_size, hidden_dim), dtype=torch.float32, device=TORCH_DEVICE)
        expert_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=TORCH_DEVICE)

        info = dispatcher_group_tokens_by_expert(expert_ids, num_experts)

        def expert_forward(x: torch.Tensor, expert_id: int) -> torch.Tensor:
            return torch.full(x.shape, float(expert_id + 1), dtype=torch.float32, device=x.device)

        outputs = dispatch_to_experts(activations, info, expert_forward)

        offsets = info.expert_offsets.cpu().numpy()
        for e in range(num_experts):
            start, end = int(offsets[e]), int(offsets[e + 1])
            if start < end:
                expected_val = float(e + 1)
                actual = outputs[start:end].cpu().numpy()
                np.testing.assert_allclose(actual, expected_val, rtol=1e-5)


class TestCombineExpertOutputs:
    """Tests for combining expert outputs with probability weighting."""

    def test_combine_with_equal_probs(self):
        """Test combination with equal routing probabilities."""
        batch_size = 2
        out_dim = 4
        top_k = 2
        num_experts = 2

        expert_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=TORCH_DEVICE)
        expert_probs = torch.tensor(
            [[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32, device=TORCH_DEVICE
        )

        info = dispatcher_group_tokens_by_expert(expert_ids, num_experts)

        expert_outputs = torch.ones(
            (batch_size * top_k, out_dim), dtype=torch.float32, device=TORCH_DEVICE
        )

        result = combine_expert_outputs(expert_outputs, expert_probs, info)

        expected = np.ones((batch_size, out_dim), dtype=np.float32)
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-5)

    def test_combine_with_varying_probs(self):
        """Test combination with different routing probabilities."""
        batch_size = 2
        top_k = 2
        num_experts = 2

        expert_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=TORCH_DEVICE)
        expert_probs = torch.tensor(
            [[0.7, 0.3], [0.6, 0.4]], dtype=torch.float32, device=TORCH_DEVICE
        )

        info = dispatcher_group_tokens_by_expert(expert_ids, num_experts)

        outputs_list = []
        for i in range(batch_size * top_k):
            for e in range(num_experts):
                start = int(info.expert_offsets[e].item())
                end = int(info.expert_offsets[e + 1].item())
                if start <= i < end:
                    if e == 0:
                        outputs_list.append([1.0, 0.0])
                    else:
                        outputs_list.append([0.0, 1.0])
                    break

        expert_outputs = torch.tensor(outputs_list, dtype=torch.float32, device=TORCH_DEVICE)

        result = combine_expert_outputs(expert_outputs, expert_probs, info)

        expected = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float32)
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-5)


class TestTokenDispatcherClass:
    """Tests for the high-level TokenDispatcher class."""

    def test_dispatcher_creation(self):
        """Test creating a TokenDispatcher."""
        dispatcher = TokenDispatcher(
            num_experts=64,
            hidden_dim=4096,
            intermediate_dim=14336,
            top_k=2,
        )

        assert dispatcher.num_experts == 64
        assert dispatcher.hidden_dim == 4096
        assert dispatcher.top_k == 2

    def test_dispatcher_simple_dispatch(self):
        """Test simple dispatch through TokenDispatcher."""
        dispatcher = TokenDispatcher(
            num_experts=4,
            hidden_dim=8,
            intermediate_dim=16,
            top_k=2,
            enable_stats=True,
        )

        batch_size = 4
        hidden_dim = 8

        hidden_states = torch.ones(
            (batch_size, hidden_dim), dtype=torch.float32, device=TORCH_DEVICE
        )
        expert_ids = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [0, 3]], dtype=torch.int32, device=TORCH_DEVICE
        )
        expert_probs = torch.tensor(
            [[0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.8, 0.2]],
            dtype=torch.float32,
            device=TORCH_DEVICE,
        )

        def expert_forward(x: torch.Tensor, expert_id: int) -> torch.Tensor:
            return x

        output = dispatcher.dispatch(hidden_states, expert_ids, expert_probs, expert_forward)

        np.testing.assert_allclose(output.cpu().numpy(), hidden_states.cpu().numpy(), rtol=1e-5)

        assert dispatcher.last_stats is not None
        assert dispatcher.last_stats.num_tokens == 4
        assert dispatcher.last_stats.top_k == 2
        assert dispatcher.last_stats.active_experts == 4

    def test_dispatcher_with_shared_expert(self):
        """Test dispatch with shared expert."""
        dispatcher = TokenDispatcher(
            num_experts=4,
            hidden_dim=8,
            intermediate_dim=16,
            top_k=2,
        )

        batch_size = 4
        hidden_dim = 8

        hidden_states = torch.ones(
            (batch_size, hidden_dim), dtype=torch.float32, device=TORCH_DEVICE
        )
        expert_ids = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [0, 3]], dtype=torch.int32, device=TORCH_DEVICE
        )
        expert_probs = torch.tensor(
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
            dtype=torch.float32,
            device=TORCH_DEVICE,
        )

        def expert_forward(x: torch.Tensor, expert_id: int) -> torch.Tensor:
            return x

        def shared_expert_forward(x: torch.Tensor, expert_id: int) -> torch.Tensor:
            return x * 2.0

        output = dispatcher.dispatch_with_shared_expert(
            hidden_states,
            expert_ids,
            expert_probs,
            expert_forward,
            shared_expert_forward,
            shared_expert_weight=0.5,
        )

        expected = np.full((batch_size, hidden_dim), 2.0, dtype=np.float32)
        np.testing.assert_allclose(output.cpu().numpy(), expected, rtol=1e-5)


class TestComputeExpertLoadDispatcher:
    """Tests for expert load computation (dispatcher module)."""

    def test_balanced_load(self):
        """Test with balanced expert distribution."""
        expert_ids = torch.tensor(
            [[0, 1], [0, 1], [0, 1], [0, 1]], dtype=torch.int32, device=TORCH_DEVICE
        )
        num_experts = 2

        load = dispatcher_compute_expert_load(expert_ids, num_experts)

        np.testing.assert_array_equal(load.cpu().numpy(), [4, 4])

    def test_unbalanced_load(self):
        """Test with unbalanced expert distribution."""
        expert_ids = torch.tensor([[0, 0], [0, 1], [0, 0]], dtype=torch.int32, device=TORCH_DEVICE)
        num_experts = 3

        load = dispatcher_compute_expert_load(expert_ids, num_experts)

        np.testing.assert_array_equal(load.cpu().numpy(), [5, 1, 0])


class TestLoadBalancingLossDispatcher:
    """Tests for auxiliary load balancing loss (dispatcher module)."""

    def test_perfect_balance(self):
        """Test loss with perfectly balanced routing."""
        batch_size = 4
        num_experts = 2

        expert_ids = torch.tensor([[0], [1], [0], [1]], dtype=torch.int32, device=TORCH_DEVICE)
        router_probs = (
            torch.ones((batch_size, num_experts), dtype=torch.float32, device=TORCH_DEVICE)
            / num_experts
        )

        loss = dispatcher_compute_load_balancing_loss(router_probs, expert_ids, num_experts)

        assert abs(loss.item() - 1.0) < 1e-5

    def test_skewed_routing(self):
        """Test loss increases with skewed routing and probs."""
        num_experts = 2

        expert_ids = torch.tensor([[0], [0], [0], [0]], dtype=torch.int32, device=TORCH_DEVICE)

        router_probs = torch.tensor(
            [[0.9, 0.1], [0.8, 0.2], [0.9, 0.1], [0.85, 0.15]],
            dtype=torch.float32,
            device=TORCH_DEVICE,
        )

        loss = dispatcher_compute_load_balancing_loss(router_probs, expert_ids, num_experts)

        assert loss.item() > 1.0


class TestTokenDispatcherEndToEnd:
    """End-to-end integration tests for token dispatcher."""

    def test_full_dispatch_flow(self):
        """Test complete dispatch flow from grouping to output."""
        batch_size = 8
        hidden_dim = 16
        out_dim = 16
        top_k = 2
        num_experts = 4

        np.random.seed(42)

        expert_ids_np = np.random.randint(0, num_experts, size=(batch_size, top_k))
        expert_ids = torch.tensor(expert_ids_np, dtype=torch.int32, device=TORCH_DEVICE)

        expert_probs_np = np.random.rand(batch_size, top_k).astype(np.float32)
        expert_probs_np /= expert_probs_np.sum(axis=1, keepdims=True)
        expert_probs = torch.tensor(expert_probs_np, dtype=torch.float32, device=TORCH_DEVICE)

        torch.manual_seed(42)
        activations = torch.randn(
            (batch_size, hidden_dim), dtype=torch.float32, device=TORCH_DEVICE
        )

        info = dispatcher_group_tokens_by_expert(expert_ids, num_experts)

        def expert_forward(x: torch.Tensor, expert_id: int) -> torch.Tensor:
            return x[:, :out_dim]

        expert_outputs = dispatch_to_experts(activations, info, expert_forward)
        result = combine_expert_outputs(expert_outputs, expert_probs, info)

        expected = activations[:, :out_dim].cpu().numpy()
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

    def test_dispatch_info_expert_batch_size(self):
        """Test DispatchInfo.expert_batch_size method."""
        expert_ids = torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.int32, device=TORCH_DEVICE)
        num_experts = 3

        info = dispatcher_group_tokens_by_expert(expert_ids, num_experts)

        assert info.expert_batch_size(0) == 2
        assert info.expert_batch_size(1) == 2
        assert info.expert_batch_size(2) == 2

    def test_dispatch_stats(self):
        """Test DispatchStats calculation."""
        dispatcher = TokenDispatcher(
            num_experts=4,
            hidden_dim=8,
            intermediate_dim=16,
            top_k=2,
            enable_stats=True,
        )

        expert_ids = torch.tensor(
            [[0, 1], [0, 1], [0, 1], [0, 1]], dtype=torch.int32, device=TORCH_DEVICE
        )
        expert_probs = torch.tensor(
            [[0.6, 0.4], [0.6, 0.4], [0.6, 0.4], [0.6, 0.4]],
            dtype=torch.float32,
            device=TORCH_DEVICE,
        )
        hidden_states = torch.ones((4, 8), dtype=torch.float32, device=TORCH_DEVICE)

        def expert_forward(x: torch.Tensor, expert_id: int) -> torch.Tensor:
            return x

        dispatcher.dispatch(hidden_states, expert_ids, expert_probs, expert_forward)

        stats = dispatcher.last_stats
        assert stats is not None
        assert stats.active_experts == 2
        assert stats.max_expert_load == 4
        assert stats.load_imbalance == 0.0


# ============================================================================
# Expert Cache Tests (PyTorch)
# ============================================================================


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

        assert hash(key1) == hash(key2)

        d = {key1: "value"}
        assert d[key2] == "value"


class TestCacheEntryClass:
    """Tests for CacheEntry dataclass."""

    def test_creation(self):
        key = TileKey(0, 0, 0)
        data = torch.zeros((64, 64), device=TORCH_DEVICE)

        entry = CacheEntry(key=key, data=data, size_bytes=data.numel() * data.element_size())

        assert entry.key == key
        assert entry.access_count == 0
        assert entry.size_bytes == data.numel() * data.element_size()

    def test_touch(self):
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
        stats = ExpertStats(expert_id=0)

        assert stats.activation_rate == 0.0

        stats.record_batch(100)
        stats.record_activation(10)

        assert stats.activation_rate == pytest.approx(0.1, rel=1e-3)

    def test_recent_rate(self):
        stats = ExpertStats(expert_id=0)

        assert stats.recent_rate == 0.0

        for _ in range(10):
            stats.record_activation(5)

        assert stats.recent_rate == pytest.approx(5.0, rel=1e-3)

    def test_recent_window_limit(self):
        stats = ExpertStats(expert_id=0)

        for i in range(150):
            stats.record_activation(1)

        assert len(stats.recent_window) == 100


class TestLayerStatsCache:
    """Tests for LayerStats dataclass (cache module)."""

    def test_hit_rate(self):
        stats = LayerStats(layer_idx=0)

        assert stats.hit_rate == 0.0

        stats.cache_hits = 70
        stats.cache_misses = 30

        assert stats.hit_rate == pytest.approx(0.7, rel=1e-3)

    def test_get_expert_stats(self):
        stats = LayerStats(layer_idx=0)

        expert_stats = stats.get_expert_stats(5)
        assert expert_stats.expert_id == 5

        expert_stats2 = stats.get_expert_stats(5)
        assert expert_stats2 is expert_stats

    def test_get_hot_experts(self):
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
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        call_count = 0

        def dequant_fn():
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
        tile_size = 64 * 64 * 2
        cache_size_mb = (tile_size * 2.5) / (1024 * 1024)
        cache = ExpertCache(
            num_experts=8, num_layers=2, cache_size_mb=int(cache_size_mb * 1000) / 1000
        )

        def make_tile(expert_id):
            return torch.full((64, 64), expert_id, dtype=torch.float16, device=TORCH_DEVICE)

        cache.get_expert_tile(0, 0, 0, lambda: make_tile(0))
        cache.get_expert_tile(0, 1, 0, lambda: make_tile(1))
        cache.get_expert_tile(0, 2, 0, lambda: make_tile(2))

        assert cache._total_evictions > 0

    def test_hit_rate(self):
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        def dequant_fn():
            return torch.ones((64, 64), dtype=torch.float16, device=TORCH_DEVICE)

        cache.get_expert_tile(0, 0, 0, dequant_fn)
        cache.get_expert_tile(0, 0, 0, dequant_fn)
        cache.get_expert_tile(0, 0, 0, dequant_fn)
        cache.get_expert_tile(0, 1, 0, dequant_fn)
        cache.get_expert_tile(0, 0, 0, dequant_fn)

        assert cache.hit_rate == pytest.approx(0.6, rel=1e-3)

    def test_invalidate_expert(self):
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        def dequant_fn():
            return torch.ones((64, 64), dtype=torch.float16, device=TORCH_DEVICE)

        cache.get_expert_tile(0, 0, 0, dequant_fn)
        cache.get_expert_tile(0, 0, 1, dequant_fn)
        cache.get_expert_tile(0, 1, 0, dequant_fn)

        assert cache.num_entries == 3

        evicted = cache.invalidate_expert(layer_idx=0, expert_id=0)
        assert evicted == 2
        assert cache.num_entries == 1

    def test_invalidate_layer(self):
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        def dequant_fn():
            return torch.ones((64, 64), dtype=torch.float16, device=TORCH_DEVICE)

        cache.get_expert_tile(0, 0, 0, dequant_fn)
        cache.get_expert_tile(0, 1, 0, dequant_fn)
        cache.get_expert_tile(1, 0, 0, dequant_fn)

        assert cache.num_entries == 3

        evicted = cache.invalidate_layer(layer_idx=0)
        assert evicted == 2
        assert cache.num_entries == 1

    def test_clear(self):
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        def dequant_fn():
            return torch.ones((64, 64), dtype=torch.float16, device=TORCH_DEVICE)

        cache.get_expert_tile(0, 0, 0, dequant_fn)
        cache.get_expert_tile(0, 1, 0, dequant_fn)

        assert cache.num_entries == 2

        cache.clear()

        assert cache.num_entries == 0
        assert cache.size_mb == 0.0

    def test_resize(self):
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        def dequant_fn():
            return torch.ones((64, 64), dtype=torch.float16, device=TORCH_DEVICE)

        for i in range(10):
            cache.get_expert_tile(0, i, 0, dequant_fn)

        initial_entries = cache.num_entries

        evicted = cache.resize(0)
        assert evicted == initial_entries
        assert cache.num_entries == 0

    def test_record_expert_activation(self):
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        expert_ids = torch.tensor([[0, 1], [0, 2], [1, 3]], dtype=torch.int32, device=TORCH_DEVICE)
        cache.record_expert_activation(layer_idx=0, expert_ids=expert_ids)

        stats = cache._layer_stats[0]
        assert stats.get_expert_stats(0).activation_count == 2
        assert stats.get_expert_stats(1).activation_count == 2
        assert stats.get_expert_stats(2).activation_count == 1
        assert stats.get_expert_stats(3).activation_count == 1

    def test_get_prefetch_candidates(self):
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1, prefetch_k=3)

        for _ in range(10):
            expert_ids = torch.tensor([[0, 1], [0, 2]], dtype=torch.int32, device=TORCH_DEVICE)
            cache.record_expert_activation(0, expert_ids)

        candidates = cache.get_prefetch_candidates(0)
        assert 0 in candidates

    def test_get_stats(self):
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        def dequant_fn():
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
        cache = ExpertCache(num_experts=8, num_layers=2, cache_size_mb=1)

        # Use CPU for thread safety test - MPS doesn't handle concurrent
        # .item() calls well across threads
        precomputed_tiles = {
            i: torch.full((64, 64), i, dtype=torch.float16, device="cpu") for i in range(4)
        }

        results = []
        lock = threading.Lock()

        def worker(expert_id):
            tile = precomputed_tiles[expert_id]

            def dequant_fn():
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

        assert coord.num_row_tiles == 4
        assert coord.num_col_tiles == 2
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

        r_start, r_end, c_start, c_end = coord.tile_bounds(0)
        assert (r_start, r_end, c_start, c_end) == (0, 64, 0, 64)

        r_start, r_end, c_start, c_end = coord.tile_bounds(1)
        assert (r_start, r_end, c_start, c_end) == (0, 64, 64, 128)

    def test_tile_bounds_with_padding(self):
        """Test non-aligned dimensions."""
        coord = TileCoordinator(
            weight_shape=(100, 100),
            tile_shape=(64, 64),
        )

        last_tile = coord.num_tiles - 1
        r_start, r_end, c_start, c_end = coord.tile_bounds(last_tile)
        assert r_end == 100
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

        tiles = coord.tiles_for_output_range(0, 64)
        assert tiles == [0, 1]

        tiles = coord.tiles_for_output_range(64, 128)
        assert tiles == [2, 3]


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
        config = {}

        cache = create_moe_cache(config)

        assert cache.num_layers == 32
        assert cache.num_experts == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
