"""Tests for ``metal_marlin.launch_tracing``.

Two tiers:
- *Unit tests* (no Metal / MPS required) – always run.
- *MPS smoke test* – only runs when PyTorch MPS is available.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

# Import the module under test – it has no Metal dependency.
from metal_marlin.launch_tracing import (
    all_events,
    commit_count,
    copy_count,
    dispatch_count,
    enable_for_testing,
    is_enabled,
    kernel_names,
    record_commit,
    record_copy,
    record_dispatch,
    record_kernel,
    record_wait,
    reset,
    total_elapsed_ms,
    trace_region,
    wait_count,
    write_json,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_state():
    """Ensure each test starts with a clean tracing state."""
    # Force-enable so we can exercise the recording paths.
    enable_for_testing()
    reset()
    yield
    reset()


# ---------------------------------------------------------------------------
# Unit tests – no Metal required
# ---------------------------------------------------------------------------

class TestDisabledIsNoOp:
    """When tracing is disabled, every API is a cheap no-op."""

    def test_dispatch_no_op_when_disabled(self) -> None:
        import metal_marlin.launch_tracing as lt
        lt._ENABLED = False  # type: ignore[attr-defined]
        lt.reset()
        record_dispatch("k")
        assert dispatch_count() == 0

    def test_commit_no_op_when_disabled(self) -> None:
        import metal_marlin.launch_tracing as lt
        lt._ENABLED = False  # type: ignore[attr-defined]
        lt.reset()
        record_commit("c")
        assert commit_count() == 0

    def test_wait_no_op_when_disabled(self) -> None:
        import metal_marlin.launch_tracing as lt
        lt._ENABLED = False  # type: ignore[attr-defined]
        lt.reset()
        record_wait("w")
        assert wait_count() == 0

    def test_copy_no_op_when_disabled(self) -> None:
        import metal_marlin.launch_tracing as lt
        lt._ENABLED = False  # type: ignore[attr-defined]
        lt.reset()
        record_copy("a", "b")
        assert copy_count() == 0

    def test_region_no_op_when_disabled(self) -> None:
        import metal_marlin.launch_tracing as lt
        lt._ENABLED = False  # type: ignore[attr-defined]
        lt.reset()
        with trace_region("x"):
            pass
        assert all_events() == []

    def test_write_json_returns_none_when_disabled(self) -> None:
        import metal_marlin.launch_tracing as lt
        lt._ENABLED = False  # type: ignore[attr-defined]
        lt.reset()
        assert write_json() is None


class TestRecordDispatch:
    """record_dispatch accumulates events correctly."""

    def test_single_dispatch(self) -> None:
        record_dispatch("marlin_fp4_gemm", M=1, N=4096, K=4096)
        assert dispatch_count() == 1
        evts = all_events()
        assert len(evts) == 1
        assert evts[0]["type"] == "dispatch"
        assert evts[0]["kernel"] == "marlin_fp4_gemm"
        assert evts[0]["M"] == 1
        assert "elapsed_ms" in evts[0]

    def test_multiple_dispatches(self) -> None:
        for name in ("k1", "k2", "k3"):
            record_dispatch(name)
        assert dispatch_count() == 3
        assert kernel_names() == ["k1", "k2", "k3"]

    def test_dispatch_does_not_pollute_other_counters(self) -> None:
        record_dispatch("k")
        assert commit_count() == 0
        assert wait_count() == 0
        assert copy_count() == 0


class TestRecordCommit:
    def test_single_commit(self) -> None:
        record_commit("cb0")
        assert commit_count() == 1
        assert all_events()[0]["label"] == "cb0"

    def test_commit_with_metadata(self) -> None:
        record_commit("cb1", queue="main")
        evts = all_events()
        assert evts[0]["queue"] == "main"


class TestRecordWait:
    def test_single_wait(self) -> None:
        record_wait("wait0")
        assert wait_count() == 1


class TestRecordCopy:
    def test_single_copy(self) -> None:
        record_copy("src_buf", "dst_buf", size_bytes=1024)
        assert copy_count() == 1
        evts = all_events()
        assert evts[0]["src"] == "src_buf"
        assert evts[0]["dst"] == "dst_buf"
        assert evts[0]["size_bytes"] == 1024


class TestRecordKernel:
    def test_kernel_recorded(self) -> None:
        record_kernel("my_kernel", variant="fp4")
        evts = all_events()
        assert len(evts) == 1
        assert evts[0]["type"] == "kernel"
        assert evts[0]["variant"] == "fp4"


class TestTraceRegion:
    def test_region_begin_end(self) -> None:
        with trace_region("forward", layer=3):
            record_dispatch("inner_kernel")
        evts = all_events()
        types = [e["type"] for e in evts]
        assert types == ["region_begin", "dispatch", "region_end"]
        assert evts[0]["name"] == "forward"
        assert evts[0]["layer"] == 3
        assert evts[2]["duration_ms"] >= 0.0

    def test_nested_regions(self) -> None:
        with trace_region("outer"):
            with trace_region("inner"):
                record_dispatch("k")
        evts = all_events()
        types = [e["type"] for e in evts]
        assert types == [
            "region_begin",  # outer
            "region_begin",  # inner
            "dispatch",
            "region_end",    # inner
            "region_end",    # outer
        ]


class TestQueryHelpers:
    def test_counts_mixed_events(self) -> None:
        record_dispatch("k1")
        record_commit("c1")
        record_wait("w1")
        record_copy("s", "d", size_bytes=8)
        record_dispatch("k2")
        assert dispatch_count() == 2
        assert commit_count() == 1
        assert wait_count() == 1
        assert copy_count() == 1

    def test_kernel_names_order(self) -> None:
        record_dispatch("first")
        record_commit("mid")
        record_dispatch("second")
        assert kernel_names() == ["first", "second"]

    def test_all_events_is_copy(self) -> None:
        record_dispatch("k")
        snap = all_events()
        assert snap is not _raw_events()
        assert len(snap) == 1

    def test_total_elapsed_ms_nonzero(self) -> None:
        record_dispatch("k")
        assert total_elapsed_ms() >= 0.0

    def test_total_elapsed_ms_zero_when_empty(self) -> None:
        assert total_elapsed_ms() == 0.0


class TestReset:
    def test_reset_clears_everything(self) -> None:
        record_dispatch("k")
        record_commit("c")
        reset()
        assert dispatch_count() == 0
        assert commit_count() == 0
        assert all_events() == []


class TestWriteJson:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        record_dispatch("k1", M=1, N=4096)
        record_commit("cb0")
        result = write_json(output_dir=str(tmp_path), filename="trace.json")
        assert result is not None
        assert Path(result).exists()

        data = json.loads(Path(result).read_text())
        assert data["version"] == "launch_tracing_v1"
        assert data["dispatch_count"] == 1
        assert data["commit_count"] == 1
        assert len(data["events"]) == 2

    def test_default_output_dir(self) -> None:
        """write_json with no args writes under agent_workspace/qwen36_27b."""
        record_dispatch("k")
        result = write_json()
        assert result is not None
        assert "qwen36_27b" in result
        # Clean up
        Path(result).unlink(missing_ok=True)

    def test_returns_none_when_no_events(self, tmp_path: Path) -> None:
        # Enabled but no events → None
        result = write_json(output_dir=str(tmp_path))
        assert result is None

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "dir"
        record_dispatch("k")
        result = write_json(output_dir=str(nested))
        assert result is not None
        assert nested.is_dir()


class TestIsEnabled:
    def test_enable_for_testing_sets_true(self) -> None:
        enable_for_testing()
        assert is_enabled() is True


# ---------------------------------------------------------------------------
# MPS-gated smoke test
# ---------------------------------------------------------------------------

try:
    from metal_marlin._compat import HAS_MPS
except Exception:
    HAS_MPS = False


@pytest.mark.skipif(not HAS_MPS, reason="Requires MPS (Apple Silicon)")
class TestMPSmoke:
    """Minimal smoke test: record a fake dispatch lifecycle alongside an MPS
    tensor operation to ensure the tracing helper doesn't interfere with
    actual Metal workloads."""

    def test_trace_alongside_mps_tensor(self) -> None:
        import torch

        device = "mps"
        x = torch.randn(4, 4, device=device)

        with trace_region("matmul_smoke", batch=4):
            record_dispatch("torch_matmul", shape=(4, 4))
            _ = x @ x
            record_commit("cb_smoke")
            record_wait("wait_smoke")

        assert dispatch_count() == 1
        assert commit_count() == 1
        assert wait_count() == 1
        assert len(kernel_names()) == 1


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _raw_events() -> list[dict[str, Any]]:
    """Access the module-level _events list for identity checks."""
    import metal_marlin.launch_tracing as lt
    return lt._events  # type: ignore[attr-defined]
