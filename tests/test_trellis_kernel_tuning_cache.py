"""Tests for persistent kernel auto-tuning cache in TrellisMoEMLP."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

try:
    from metal_marlin.trellis.model import TrellisMoEMLP
    from metal_marlin.trellis.testing import create_mock_moe_mlp

    HAS_TRELLIS = True
except ImportError:
    HAS_TRELLIS = False


requires_trellis = pytest.mark.skipif(
    not HAS_TRELLIS, reason="Trellis modules required"
)


def _make_mock_moe_layer(
    *,
    hidden_dim: int = 32,
    intermediate_dim: int = 64,
    num_experts: int = 4,
    bits: int = 3,
) -> TrellisMoEMLP:
    return create_mock_moe_mlp(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        num_experts_per_tok=min(2, num_experts),
        bits=bits,
        device="cpu",
        eager_buffers=False,
    )


@requires_trellis
def test_tuning_cache_key_includes_model_dimensions() -> None:
    """Cache path hash should vary with hidden/intermediate/experts/bits."""
    base = _make_mock_moe_layer(
        hidden_dim=32, intermediate_dim=64, num_experts=4, bits=3
    )
    hidden_changed = _make_mock_moe_layer(
        hidden_dim=48, intermediate_dim=64, num_experts=4, bits=3
    )
    intermediate_changed = _make_mock_moe_layer(
        hidden_dim=32, intermediate_dim=96, num_experts=4, bits=3
    )
    experts_changed = _make_mock_moe_layer(
        hidden_dim=32, intermediate_dim=64, num_experts=6, bits=3
    )
    bits_changed = _make_mock_moe_layer(
        hidden_dim=32, intermediate_dim=64, num_experts=4, bits=4
    )

    base_path = base._get_tuning_cache_path()
    assert hidden_changed._get_tuning_cache_path() != base_path
    assert intermediate_changed._get_tuning_cache_path() != base_path
    assert experts_changed._get_tuning_cache_path() != base_path
    assert bits_changed._get_tuning_cache_path() != base_path


@requires_trellis
def test_second_load_uses_cached_tuning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Second layer load should skip auto-tuning when cache file exists."""
    cache_path = tmp_path / "kernel_tuning_test.json"
    monkeypatch.setattr(
        TrellisMoEMLP, "_get_tuning_cache_path", lambda self: cache_path
    )

    first = _make_mock_moe_layer()
    first._kernel_config = {
        "use_fp32_False": {"time_ms": 1.25, "use_fp32_acc": False},
        "optimal": {"time_ms": 1.25, "use_fp32_acc": False},
    }
    first._save_tuning_cache()
    assert cache_path.exists()

    second = _make_mock_moe_layer()
    x = torch.randn(2, second.hidden_dim, dtype=torch.float16)

    caplog.clear()
    with caplog.at_level(logging.INFO):
        second._auto_tune_kernels(x)

    assert second._kernel_auto_tuned is True
    assert second._kernel_config == first._kernel_config
    assert "Starting kernel auto-tuning..." not in caplog.text
    assert "Kernel auto-tuning complete" not in caplog.text


@requires_trellis
def test_optimal_use_fp32_acc_respects_cache() -> None:
    """_get_optimal_use_fp32_acc should prefer cached config over heuristic."""
    # Small hidden dim (<1024), default heuristic is False
    layer = _make_mock_moe_layer(hidden_dim=32)
    
    # By default (no cache), it should be False
    assert layer._get_optimal_use_fp32_acc() is False

    # Inject cache saying True is optimal (e.g. maybe better on some hardware)
    layer._kernel_config = {
        "optimal": {"use_fp32_acc": True}
    }
    assert layer._get_optimal_use_fp32_acc() is True

    # Inject cache saying False is optimal
    layer._kernel_config = {
        "optimal": {"use_fp32_acc": False}
    }
    assert layer._get_optimal_use_fp32_acc() is False