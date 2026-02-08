"""Tests for ExpertSelectionCache routing fast path and speculative prefetch."""

from __future__ import annotations

import torch

from metal_marlin.moe.expert_selection_cache import (
    ExpertSelectionCache,
    ExpertSelectionCacheConfig,
)


def test_hash_lookup_fast_path_skips_router_forward() -> None:
    """Repeated hidden states should hit cache and bypass router forward."""
    calls = {"count": 0}

    def router_fn(_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        calls["count"] += 1
        return (
            torch.tensor([3, 7], dtype=torch.int64),
            torch.tensor([0.72, 0.28], dtype=torch.float32),
        )

    hidden = torch.randn(4096, dtype=torch.float32)

    with ExpertSelectionCache(num_experts=64, top_k=2) as cache:
        first = cache.get_or_route(hidden, router_fn=router_fn, layer_idx=0)
        second = cache.get_or_route(hidden.clone(), router_fn=router_fn, layer_idx=0)

    assert first.cache_hit is False
    assert first.hit_type == "miss"
    assert second.cache_hit is True
    assert second.hit_type == "exact"
    assert second.expert_ids == (3, 7)
    assert calls["count"] == 1


def test_router_cache_lru_eviction() -> None:
    """LRU policy should evict oldest routing key when over capacity."""
    calls = {"count": 0}

    def router_fn(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        calls["count"] += 1
        # Map to two deterministic expert ids.
        idx = int(torch.argmax(hidden).item())
        first = idx % 64
        second = (idx + 1) % 64
        return (
            torch.tensor([first, second], dtype=torch.int64),
            torch.tensor([0.8, 0.2], dtype=torch.float32),
        )

    config = ExpertSelectionCacheConfig(
        max_routing_entries=2,
        enable_locality_fallback=False,
    )

    h1 = torch.zeros(32, dtype=torch.float32)
    h2 = torch.zeros(32, dtype=torch.float32)
    h3 = torch.zeros(32, dtype=torch.float32)
    h1[0] = 1.0
    h2[1] = 1.0
    h3[2] = 1.0

    with ExpertSelectionCache(num_experts=64, top_k=2, config=config) as cache:
        cache.get_or_route(h1, router_fn=router_fn, layer_idx=0)
        cache.get_or_route(h2, router_fn=router_fn, layer_idx=0)
        cache.get_or_route(h3, router_fn=router_fn, layer_idx=0)
        # h1 should be evicted after third insert.
        cache.get_or_route(h1, router_fn=router_fn, layer_idx=0)
        stats = cache.get_stats()

    assert calls["count"] == 4
    assert stats["routing_cache_size"] == 2
    assert stats["stats"]["lru_evictions"] >= 1


def test_locality_hit_reuses_similar_hidden_states() -> None:
    """Near-identical hidden states should hit exact/locality cache path."""
    calls = {"count": 0}

    def router_fn(_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        calls["count"] += 1
        return (
            torch.tensor([11, 23], dtype=torch.int64),
            torch.tensor([0.61, 0.39], dtype=torch.float32),
        )

    config = ExpertSelectionCacheConfig(
        max_routing_entries=32,
        sketch_dim=48,
        locality_search_window=16,
        locality_similarity_threshold=0.995,
        enable_locality_fallback=True,
    )

    base = torch.randn(3072, dtype=torch.float32)
    similar = base + (1e-3 * torch.randn_like(base))

    with ExpertSelectionCache(num_experts=64, top_k=2, config=config) as cache:
        first = cache.get_or_route(base, router_fn=router_fn, layer_idx=0)
        second = cache.get_or_route(similar, router_fn=router_fn, layer_idx=0)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.hit_type in {"exact", "locality"}
    assert calls["count"] == 1


def test_locality_hit_promotes_exact_hash_entry() -> None:
    """A locality reuse should be promoted so subsequent lookups hit exact hash cache."""
    calls = {"count": 0}

    def router_fn(_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        calls["count"] += 1
        return (
            torch.tensor([5, 41], dtype=torch.int64),
            torch.tensor([0.58, 0.42], dtype=torch.float32),
        )

    config = ExpertSelectionCacheConfig(
        max_routing_entries=32,
        sketch_dim=64,
        locality_search_window=32,
        locality_similarity_threshold=0.95,
        enable_locality_fallback=True,
    )

    with ExpertSelectionCache(num_experts=64, top_k=2, config=config) as cache:
        base = torch.randn(4096, dtype=torch.float32)
        base_hash = cache.hidden_state_hash(base)
        similar = None
        for _ in range(512):
            candidate = base + (2e-3 * torch.randn_like(base))
            if cache.hidden_state_hash(candidate) != base_hash:
                similar = candidate
                break
        assert similar is not None

        first = cache.get_or_route(base, router_fn=router_fn, layer_idx=0)
        second = cache.get_or_route(similar, router_fn=router_fn, layer_idx=0)
        third = cache.get_or_route(
            similar.clone(),
            router_fn=router_fn,
            layer_idx=0,
            allow_locality=False,
        )

        stats = cache.get_stats()

    assert first.hit_type == "miss"
    assert second.hit_type == "locality"
    assert third.hit_type == "exact"
    assert calls["count"] == 1
    assert stats["stats"]["locality_promotions"] >= 1


def test_speculative_prefetch_populates_weight_cache() -> None:
    """Prefetch should asynchronously load and cache predicted experts."""
    loaded: list[tuple[int, int]] = []

    def load_weight(layer_idx: int, expert_id: int) -> torch.Tensor:
        loaded.append((layer_idx, expert_id))
        return torch.full((4, 4), float(expert_id), dtype=torch.float32)

    config = ExpertSelectionCacheConfig(
        max_routing_entries=64,
        max_prefetched_weights=16,
        prefetch_k=4,
        prefetch_threads=2,
        enable_speculative_prefetch=True,
    )

    h0 = torch.randn(256, dtype=torch.float32)
    h1 = torch.randn(256, dtype=torch.float32)
    h2 = torch.randn(256, dtype=torch.float32)
    h3 = torch.randn(256, dtype=torch.float32)

    with ExpertSelectionCache(
        num_experts=64,
        top_k=2,
        load_expert_weight=load_weight,
        config=config,
        device="cpu",
    ) as cache:
        # Build short transition history for context pair (1,2).
        cache.cache_routing(h0, [1, 2], [0.7, 0.3])
        cache.cache_routing(h1, [7, 8], [0.6, 0.4])
        cache.cache_routing(h2, [1, 2], [0.65, 0.35])
        cache.cache_routing(h3, [7, 9], [0.55, 0.45])

        predicted = cache.prefetch_likely_experts(layer_idx=3, context_expert_ids=[1, 2], k=4)
        cache.wait_for_prefetch(timeout=2.0)

        prefetched = cache.get_prefetched_weights(layer_idx=3, expert_ids=predicted)

    assert predicted
    assert 7 in predicted
    assert prefetched
    for tensor in prefetched.values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"
    assert loaded
