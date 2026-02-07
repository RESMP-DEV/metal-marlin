"""Focused performance-path regression tests for MR-GPTQ.

These tests validate control-flow decisions that affect quantization throughput,
without requiring model downloads or heavy calibration runs.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

import metal_marlin.gptq_accelerated as gptq_accelerated
from metal_marlin.mr_gptq import HessianCollector, gptq_quantize_layer


class _TinyLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def test_hessian_hook_avoids_per_hook_numpy_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hook path should keep tensor-native accumulation after first conversion."""
    collector = HessianCollector(accumulator_dtype="float32")
    model = _TinyLinearModel()
    collector.register_hooks(model)

    layer_name = "proj"
    assert layer_name in collector._layer_dims

    # Simulate checkpoint-restored Hessian as NumPy; hook should convert once,
    # then stay tensor-native for subsequent hooks.
    collector._hessians[layer_name] = (np.zeros((4, 4), dtype=np.float32), 0)

    as_tensor_spy = MagicMock(wraps=torch.as_tensor)
    monkeypatch.setattr(torch, "as_tensor", as_tensor_spy)

    model(torch.ones(2, 4, dtype=torch.float32))
    model(torch.full((3, 4), 2.0, dtype=torch.float32))

    h_sum, n_samples = collector._hessians[layer_name]
    assert isinstance(h_sum, torch.Tensor)
    assert n_samples == 5

    # Critical performance behavior: no per-hook NumPy -> tensor conversion.
    assert as_tensor_spy.call_count == 1

    hessians = collector.get_hessians(apply_damping=False)
    assert isinstance(hessians[layer_name].hessian, np.ndarray)

    collector.remove_hooks()


def test_gptq_quantize_layer_runs_vectorized_propagation_path() -> None:
    """GPTQ path should include and exercise vectorized in-group propagation."""
    source = inspect.getsource(gptq_quantize_layer)
    assert "W[:, i + 1 : g_end] -= err[:, None] * h_ratio[None, :]" in source

    # Deterministic tiny case where Hessian coupling changes a later-column
    # quantization choice due to propagated error.
    weights = np.array([[0.49, 0.51, 1.0]], dtype=np.float32)
    coupled_hessian = np.array(
        [
            [1.0, -0.99, 0.0],
            [-0.99, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    identity_hessian = np.eye(3, dtype=np.float32)
    grid = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    _, _, idx_coupled = gptq_quantize_layer(
        weights,
        coupled_hessian,
        grid,
        group_size=3,
        actorder=False,
        percdamp=0.0,
    )
    _, _, idx_identity = gptq_quantize_layer(
        weights,
        identity_hessian,
        grid,
        group_size=3,
        actorder=False,
        percdamp=0.0,
    )

    # With coupling + propagation, middle column crosses quantization boundary.
    assert idx_coupled[0, 1] == 1  # -> quantized to 0.0
    assert idx_identity[0, 1] == 2  # -> quantized to 1.0


@pytest.mark.parametrize(
    ("selected_backend", "backend_ctor_name"),
    [
        (gptq_accelerated.Backend.CUDA, "CUDABackend"),
        (gptq_accelerated.Backend.MPS, "MPSBackend"),
        (gptq_accelerated.Backend.NUMPY, "NumPyBackend"),
    ],
)
def test_accelerated_auto_selects_available_backend(
    monkeypatch: pytest.MonkeyPatch,
    selected_backend: gptq_accelerated.Backend,
    backend_ctor_name: str,
) -> None:
    """AUTO backend should dispatch to the backend marked available."""
    sentinel_backend = object()

    monkeypatch.setattr(
        gptq_accelerated,
        "detect_best_backend",
        lambda: selected_backend,
    )

    selected_ctor = MagicMock(return_value=sentinel_backend)
    monkeypatch.setattr(gptq_accelerated, backend_ctor_name, selected_ctor)

    for other_ctor in {"CUDABackend", "MPSBackend", "NumPyBackend"} - {backend_ctor_name}:
        monkeypatch.setattr(
            gptq_accelerated,
            other_ctor,
            MagicMock(side_effect=AssertionError(f"Unexpected constructor call: {other_ctor}")),
        )

    quantizer = gptq_accelerated.GPTQAccelerated.create(
        backend=gptq_accelerated.Backend.AUTO
    )

    selected_ctor.assert_called_once_with()
    assert quantizer._backend is sentinel_backend

