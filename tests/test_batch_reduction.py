"""Verify that layer batching reduces command buffer commits."""

import math
import os
from types import SimpleNamespace

from metal_marlin.trellis import async_dispatch
from metal_marlin.trellis.async_dispatch import LayerBatchContext

os.environ["METAL_MARLIN_TRACE_BATCH"] = "1"


class _FakeCommandManager:
    """Tracks synthetic dispatch/commit activity for batch tracing."""

    def __init__(self) -> None:
        self.commits = 0
        self.dispatches = 0
        self.begin_calls = 0
        self._batch_active = False

    def start_batch(self) -> None:
        self.begin_calls += 1
        self._batch_active = True

    def has_active_batch(self) -> bool:
        return self._batch_active

    def commit_and_wait(self) -> None:
        self.commits += 1
        self._batch_active = False
        if os.environ.get("METAL_MARLIN_TRACE_BATCH") == "1":
            print("BATCH COMMIT")

    def dispatch_layer(self, layer_idx: int) -> None:
        self.dispatches += 1
        if os.environ.get("METAL_MARLIN_TRACE_BATCH") == "1":
            print(f"BATCH DISPATCH layer={layer_idx}")


class _FakeMoELayer:
    """Minimal stand-in for TrellisMoEMLP used by LayerBatchContext."""

    def __init__(self, cmd_manager: _FakeCommandManager) -> None:
        self._cmd_manager = cmd_manager

    def _get_async_cmd_manager(self) -> _FakeCommandManager:
        return self._cmd_manager

    def dispatch_layer(self, layer_idx: int) -> None:
        self._cmd_manager.dispatch_layer(layer_idx)


def _run_forward(model: SimpleNamespace, batch_ctx: LayerBatchContext) -> None:
    """Run a synthetic forward loop and mark each layer complete."""
    for layer_idx, layer in enumerate(model.layers):
        layer.mlp.dispatch_layer(layer_idx)
        batch_ctx.layer_complete()


def test_layer_batching_reduces_commits(monkeypatch, capsys) -> None:
    """46 layers batched by 4 should commit far fewer than per-layer dispatch."""
    # Force the batching context to run in environments without Metal.
    monkeypatch.setattr(async_dispatch, "HAS_METAL", True)

    # LayerBatchContext checks isinstance(..., TrellisMoEMLP) at runtime.
    import metal_marlin.trellis.model as trellis_model

    monkeypatch.setattr(trellis_model, "TrellisMoEMLP", _FakeMoELayer, raising=False)

    total_layers = 46
    layer_batch_size = 4
    cmd_manager = _FakeCommandManager()
    model = SimpleNamespace(
        layers=[SimpleNamespace(mlp=_FakeMoELayer(cmd_manager)) for _ in range(total_layers)]
    )

    with LayerBatchContext(model, batch_size=layer_batch_size) as batch_ctx:
        _run_forward(model, batch_ctx)

    output = capsys.readouterr().out
    commit_count = output.count("BATCH COMMIT")
    dispatch_count = output.count("BATCH DISPATCH")

    # Expected: 46 layers with batch_size=4 should produce ~12 commits.
    expected_commits = math.ceil(total_layers / layer_batch_size)
    assert commit_count == expected_commits

    # Without batching: 46+ commits (one per layer minimum).
    assert commit_count < 20

    # Dispatches should still happen for every layer.
    assert dispatch_count >= 40
