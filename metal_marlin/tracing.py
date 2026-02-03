"""Chrome tracing for Metal Marlin profiling.

Enable with MM_TRACE=1 environment variable.
Output chrome://tracing JSON format.
"""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_ENABLED = os.getenv("MM_TRACE") == "1"
_EVENTS: list[dict[str, Any]] = []
_START_TIME = time.perf_counter()


def _timestamp_us() -> int:
    """Get timestamp in microseconds since tracing started."""
    return int((time.perf_counter() - _START_TIME) * 1_000_000)


def record_event(
    name: str,
    cat: str,
    ph: str,
    ts: int | None = None,
    dur: int | None = None,
    args: dict[str, Any] | None = None,
) -> None:
    """Record a Chrome trace event.
    
    Args:
        name: Event name
        cat: Category (e.g., 'kernel', 'cpu')
        ph: Phase ('B'=begin, 'E'=end, 'X'=complete, 'i'=instant)
        ts: Timestamp in microseconds (auto-generated if None)
        dur: Duration in microseconds (for 'X' phase)
        args: Additional metadata
    """
    if not _ENABLED:
        return

    event = {
        "name": name,
        "cat": cat,
        "ph": ph,
        "ts": ts if ts is not None else _timestamp_us(),
        "pid": os.getpid(),
        "tid": 0,
    }

    if dur is not None:
        event["dur"] = dur

    if args:
        event["args"] = args

    _EVENTS.append(event)


@contextmanager
def trace_scope(name: str, cat: str = "cpu", **args: Any):
    """Context manager for tracing a code block.
    
    Example:
        with trace_scope("matmul", cat="kernel", M=4096, N=4096):
            kernel.launch()
    """
    if not _ENABLED:
        yield
        return

    ts = _timestamp_us()
    record_event(name, cat, "B", ts=ts, args=args or None)
    try:
        yield
    finally:
        record_event(name, cat, "E", ts=_timestamp_us())


def trace_kernel(name: str, **args: Any):
    """Record a kernel launch event.
    
    Example:
        trace_kernel("marlin_kernel", M=4096, N=4096, K=4096)
    """
    record_event(name, "kernel", "i", args=args or None)


def trace_instant(name: str, cat: str = "cpu", **args: Any):
    """Record an instant event.
    
    Example:
        trace_instant("checkpoint", cat="memory", bytes=1024)
    """
    record_event(name, cat, "i", args=args or None)


def write_trace(output_path: str | None = None) -> None:
    """Write trace events to JSON file.
    
    Args:
        output_path: Output file path (default: trace_<timestamp>.json)
    """
    if not _ENABLED or not _EVENTS:
        return

    if output_path is None:
        output_path = f"trace_{int(time.time())}.json"

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    trace_data = {
        "traceEvents": _EVENTS,
        "displayTimeUnit": "ms",
        "otherData": {
            "version": "metal_marlin_tracer_v1"
        }
    }

    with open(output_path_obj, "w") as f:
        json.dump(trace_data, f, indent=2)

    print(f"Trace written to {output_path_obj.absolute()}")
    print("Open in chrome://tracing")


def clear_trace() -> None:
    """Clear all recorded events."""
    global _EVENTS, _START_TIME
    _EVENTS.clear()
    _START_TIME = time.perf_counter()


def is_enabled() -> bool:
    """Check if tracing is enabled."""
    return _ENABLED
