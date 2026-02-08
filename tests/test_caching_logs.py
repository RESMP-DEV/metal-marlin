"""Tests for log messages in persistent kernel auto-tuning cache."""

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

def _make_mock_moe_layer() -> TrellisMoEMLP:
    return create_mock_moe_mlp(
        hidden_dim=32,
        intermediate_dim=64,
        num_experts=4,
        bits=3,
        device="cpu",
        eager_buffers=False,
    )

@requires_trellis
def test_caching_log_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify correct log messages for cache miss (tuning) and cache hit (loading)."""
    
    # 1. Setup mock cache path
    cache_path = tmp_path / "kernel_tuning_logs_test.json"
    monkeypatch.setattr(
        TrellisMoEMLP, "_get_tuning_cache_path", lambda self: cache_path
    )
    
    # Reset the static flag so we can test the INFO message if we want, 
    # but more importantly to have a clean state.
    TrellisMoEMLP._tuning_message_printed = False

    # 2. First Run: Cache Miss -> Should Tune
    first = _make_mock_moe_layer()
    x = torch.randn(2, first.hidden_dim, dtype=torch.float16)
    
    # We need to mock _load_cached_tuning to return None (simulating miss)
    # The real implementation will verify the file doesn't exist, so we don't strictly need to mock it
    # if we ensure the file doesn't exist.
    if cache_path.exists():
        cache_path.unlink()

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        first._auto_tune_kernels(x)
        
    # Verification for First Run
    assert "Starting kernel auto-tuning for MoE layer..." in caplog.text
    # It should NOT say it loaded from cache
    assert "Loaded kernel config from cache" not in caplog.text
    # It SHOULD say it saved the config (my new log message)
    assert f"Saved kernel config to {cache_path}" in caplog.text
    
    # Verify file was actually created
    assert cache_path.exists()

    # 3. Second Run: Cache Hit -> Should Load
    second = _make_mock_moe_layer()
    
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        second._auto_tune_kernels(x)
        
    # Verification for Second Run
    # Should log the success message
    assert "Loaded kernel config from cache" in caplog.text
    # Should NOT log tuning start messages
    assert "Starting kernel auto-tuning for MoE layer..." not in caplog.text
    assert "Optimizing kernel for MoE layer..." not in caplog.text
