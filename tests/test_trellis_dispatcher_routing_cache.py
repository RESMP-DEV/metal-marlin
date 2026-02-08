import torch

from metal_marlin.trellis.optimizations import (
    ExpertSelectionCache,
    MixedBPWMoEDispatcher,
)


def test_expert_selection_cache_multi_entry_lookup_and_eviction() -> None:
    cache = ExpertSelectionCache(max_entries=2, similarity_threshold=0.95)

    x0 = torch.randn(1, 128, dtype=torch.float16)
    x1 = torch.randn(1, 128, dtype=torch.float16)
    x2 = torch.randn(1, 128, dtype=torch.float16)
    experts0 = torch.tensor([[1, 2]], dtype=torch.long)
    experts1 = torch.tensor([[2, 3]], dtype=torch.long)
    experts2 = torch.tensor([[3, 0]], dtype=torch.long)
    weights = torch.tensor([[0.7, 0.3]], dtype=torch.float16)

    assert cache.lookup(x0) is None
    cache.store(x0, experts0, weights)
    cache.store(x1, experts1, weights)

    cached = cache.lookup(x0)
    assert cached is not None
    cached_experts, cached_weights = cached
    assert torch.equal(cached_experts, experts0)
    assert torch.equal(cached_weights, weights)

    # Add a third entry to force eviction of the oldest one (x0).
    cache.store(x2, experts2, weights)
    assert cache.lookup(x0) is None


class _DummyWorkspacePool:
    def get_output_buffer(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, 128, dtype=torch.float16)


class _DummyCmdManager:
    def begin_batch(self) -> None:
        return None

    def commit_and_wait(self) -> None:
        return None


class _DummyRouter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        # [batch, num_experts]
        return torch.tensor([[1.0, 3.0, 2.0, 0.5]], dtype=torch.float16)


class _DummyLayer:
    def __init__(self) -> None:
        self.router = _DummyRouter()
        self.num_experts_per_tok = 2
        self.hidden_dim = 128
        self.experts = [object(), object(), object(), object()]
        self.expert_selection_cache = ExpertSelectionCache(max_entries=4)
        self._bit_group_buffers = {}

    def _get_workspace_buffer_pool(self) -> _DummyWorkspacePool:
        return _DummyWorkspacePool()

    def _get_async_cmd_manager(self) -> _DummyCmdManager:
        return _DummyCmdManager()


def test_mixed_bpw_dispatcher_uses_routing_cache_before_router() -> None:
    layer = _DummyLayer()
    dispatcher = MixedBPWMoEDispatcher(layer)
    x = torch.randn(1, 128, dtype=torch.float16)

    # First call: cache miss, runs router.
    dispatcher.dispatch(x, lib=object(), buffer_pool=object())
    assert layer.router.calls == 1

    # Second call with same hidden state: cache hit, no router call.
    dispatcher.dispatch(x, lib=object(), buffer_pool=object())
    assert layer.router.calls == 1
    assert layer.expert_selection_cache.get_stats()["hits"] == 1
