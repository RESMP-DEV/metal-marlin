"""Reusable dispatch tracing helper for Metal Marlin.

Records kernel dispatches, command-buffer commits, waits, buffer copies,
kernel names, and elapsed wall-clock time so launch-count regressions
are detectable in CI.

Enable with the ``MM_LAUNCH_TRACE=1`` environment variable.
When disabled every public function is a near-zero-cost no-op (an
``if not _ENABLED: return`` guard on a module-level bool).
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Opt-in gate – set MM_LAUNCH_TRACE=1 to enable.
# When False every public call returns immediately.
# ---------------------------------------------------------------------------
_ENABLED: bool = os.getenv("MM_LAUNCH_TRACE") == "1"

# Default output directory (relative to repo root).
_DEFAULT_OUTPUT_DIR = "agent_workspace/qwen36_27b"

# Module-level accumulator – only mutated when tracing is active.
_events: list[dict[str, Any]] = []
_t0: float = 0.0  # epoch recorded on first use


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_t0() -> None:
    """Lazily record the trace epoch so import-time cost is zero."""
    global _t0
    if _t0 == 0.0:
        _t0 = time.perf_counter()


def _elapsed_ms() -> float:
    _ensure_t0()
    return (time.perf_counter() - _t0) * 1_000.0


def _append(event: dict[str, Any]) -> None:
    """Central append – single mutation point for easy auditing."""
    _events.append(event)


# ---------------------------------------------------------------------------
# Public API – each function is a trivial early-return when disabled.
# ---------------------------------------------------------------------------

def record_dispatch(kernel_name: str, **metadata: Any) -> None:
    """Record a compute-kernel dispatch.

    Parameters
    ----------
    kernel_name:
        Human-readable kernel identifier (e.g. ``"marlin_fp4_gemm"``).
    metadata:
        Arbitrary extra key/value pairs (tensor shapes, group sizes …).
    """
    if not _ENABLED:
        return
    _append({
        "type": "dispatch",
        "kernel": kernel_name,
        "elapsed_ms": _elapsed_ms(),
        **metadata,
    })
    logger.debug("launch_tracing: dispatch %s", kernel_name)


def record_commit(label: str = "", **metadata: Any) -> None:
    """Record a command-buffer commit."""
    if not _ENABLED:
        return
    _append({
        "type": "commit",
        "label": label,
        "elapsed_ms": _elapsed_ms(),
        **metadata,
    })
    logger.debug("launch_tracing: commit %s", label)


def record_wait(label: str = "", **metadata: Any) -> None:
    """Record a waitUntilCompleted / future.wait() call."""
    if not _ENABLED:
        return
    _append({
        "type": "wait",
        "label": label,
        "elapsed_ms": _elapsed_ms(),
        **metadata,
    })
    logger.debug("launch_tracing: wait %s", label)


def record_copy(src_label: str = "", dst_label: str = "",
                size_bytes: int = 0, **metadata: Any) -> None:
    """Record a buffer copy / blit operation."""
    if not _ENABLED:
        return
    _append({
        "type": "copy",
        "src": src_label,
        "dst": dst_label,
        "size_bytes": size_bytes,
        "elapsed_ms": _elapsed_ms(),
        **metadata,
    })
    logger.debug("launch_tracing: copy %s -> %s (%d bytes)",
                 src_label, dst_label, size_bytes)


def record_kernel(kernel_name: str, **metadata: Any) -> None:
    """Record a kernel-name registration (distinct from dispatch).

    Useful for pre-launch bookkeeping when the kernel name is computed
    before the actual dispatch call.
    """
    if not _ENABLED:
        return
    _append({
        "type": "kernel",
        "kernel": kernel_name,
        "elapsed_ms": _elapsed_ms(),
        **metadata,
    })
    logger.debug("launch_tracing: kernel %s", kernel_name)


@contextmanager
def trace_region(name: str, **metadata: Any) -> Generator[None, None, None]:
    """Context manager that records begin / end events with elapsed time.

    Example::

        with trace_region("forward_pass", layer=0):
            model(x)
    """
    if not _ENABLED:
        yield
        return
    start_ms = _elapsed_ms()
    _append({
        "type": "region_begin",
        "name": name,
        "elapsed_ms": start_ms,
        **metadata,
    })
    try:
        yield
    finally:
        end_ms = _elapsed_ms()
        _append({
            "type": "region_end",
            "name": name,
            "elapsed_ms": end_ms,
            "duration_ms": round(end_ms - start_ms, 6),
            **metadata,
        })


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def dispatch_count() -> int:
    """Return the number of recorded dispatch events."""
    return sum(1 for e in _events if e.get("type") == "dispatch")


def commit_count() -> int:
    """Return the number of recorded commit events."""
    return sum(1 for e in _events if e.get("type") == "commit")


def wait_count() -> int:
    """Return the number of recorded wait events."""
    return sum(1 for e in _events if e.get("type") == "wait")


def copy_count() -> int:
    """Return the number of recorded copy events."""
    return sum(1 for e in _events if e.get("type") == "copy")


def kernel_names() -> list[str]:
    """Return ordered list of dispatched kernel names."""
    return [e["kernel"] for e in _events if e.get("type") == "dispatch"]


def all_events() -> list[dict[str, Any]]:
    """Return a shallow copy of the recorded event list."""
    return list(_events)


def total_elapsed_ms() -> float:
    """Return ms since the first event was recorded (0 if no events)."""
    if not _events:
        return 0.0
    return _elapsed_ms()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def write_json(
    output_dir: str | None = None,
    filename: str = "launch_trace.json",
) -> str | None:
    """Write accumulated events to a compact JSON file.

    Parameters
    ----------
    output_dir:
        Target directory.  Defaults to ``agent_workspace/qwen36_27b``
        relative to the package root (``contrib/metal_marlin/``).
    filename:
        JSON file name inside *output_dir*.

    Returns
    -------
    The written file path, or ``None`` if tracing is disabled / no events.
    """
    if not _ENABLED or not _events:
        return None

    if output_dir is None:
        # Resolve relative to the contrib/metal_marlin/ directory.
        output_dir = str(
            Path(__file__).resolve().parent.parent / _DEFAULT_OUTPUT_DIR
        )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / filename

    payload: dict[str, Any] = {
        "version": "launch_tracing_v1",
        "total_elapsed_ms": round(total_elapsed_ms(), 6),
        "dispatch_count": dispatch_count(),
        "commit_count": commit_count(),
        "wait_count": wait_count(),
        "copy_count": copy_count(),
        "events": _events,
    }

    target.write_text(json.dumps(payload, indent=2))
    logger.info("launch_tracing: wrote %s (%d events)", target, len(_events))
    return str(target)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def reset() -> None:
    """Clear all accumulated events and reset the clock."""
    global _events, _t0
    _events = []
    _t0 = 0.0


def is_enabled() -> bool:
    """Return whether launch tracing is currently active."""
    return _ENABLED


def enable_for_testing() -> None:
    """Programmatically enable tracing **for unit tests only**.

    Real production code should use the ``MM_LAUNCH_TRACE=1`` env var.
    """
    global _ENABLED
    _ENABLED = True
