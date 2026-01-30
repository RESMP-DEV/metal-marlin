"""GPU+ANE pipeline tracing for Chrome trace viewer visualization.

This module provides tracing capabilities to visualize GPU and ANE (Apple Neural Engine)
event overlap during inference, enabling performance analysis via chrome://tracing.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    """A single trace event representing execution on a device.
    
    Attributes:
        name: Event name/label (e.g., "layer_0_forward")
        device: Target device ("gpu" or "ane")
        start_us: Start timestamp in microseconds
        end_us: End timestamp in microseconds
        metadata: Optional additional metadata for the event
    """
    name: str
    device: str  # "gpu" or "ane"
    start_us: int
    end_us: int
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineTracer:
    """Record GPU/ANE events for Chrome trace viewer.
    
    This tracer captures execution events on GPU and ANE devices and exports them
    to the Chrome Trace Event format for visualization in chrome://tracing.
    
    Usage:
        tracer = PipelineTracer()
        
        # Record events
        tracer.record("gpu_layer_0", "gpu", duration_us=500)
        tracer.record("ane_layer_0", "ane", duration_us=300)
        
        # Export for visualization
        tracer.export_chrome_trace(Path("trace.json"))
    """

    # Device to track mapping for Chrome tracing
    _DEVICE_PID_MAP = {"gpu": 1, "ane": 2}
    _DEVICE_TID_MAP = {"gpu": 1, "ane": 2}

    def __init__(self) -> None:
        """Initialize an empty pipeline tracer."""
        self.events: list[TraceEvent] = []
        self._start_time: float | None = None

    def start(self) -> None:
        """Start the tracing session.
        
        Records the reference start time for subsequent events.
        """
        self._start_time = time.perf_counter()

    def record(
        self,
        name: str,
        device: str,
        duration_us: int,
        start_us: int | None = None,
        **metadata: Any
    ) -> None:
        """Record an execution event.
        
        Args:
            name: Event name/label
            device: Target device ("gpu" or "ane")
            duration_us: Duration in microseconds
            start_us: Optional explicit start time in microseconds. If not provided,
                     uses time elapsed since start() was called.
            **metadata: Optional additional metadata
        
        Raises:
            ValueError: If device is not "gpu" or "ane"
            RuntimeError: If start() was not called and start_us is not provided
        """
        if device not in ("gpu", "ane"):
            raise ValueError(f"Device must be 'gpu' or 'ane', got: {device}")

        if start_us is None:
            if self._start_time is None:
                raise RuntimeError(
                    "Must call start() before recording events without explicit start_us"
                )
            # Calculate start time relative to tracing start
            elapsed_us = int((time.perf_counter() - self._start_time) * 1_000_000)
            start_us = elapsed_us

        event = TraceEvent(
            name=name,
            device=device,
            start_us=start_us,
            end_us=start_us + duration_us,
            metadata=metadata
        )
        self.events.append(event)

    def record_block(
        self,
        name: str,
        device: str,
        **metadata: Any
    ) -> "TraceBlock":
        """Create a context manager for recording a timed block.
        
        Args:
            name: Event name/label
            device: Target device ("gpu" or "ane")
            **metadata: Optional additional metadata
            
        Returns:
            A context manager that records the block's execution time
            
        Example:
            with tracer.record_block("forward_pass", "gpu"):
                # GPU computation here
                pass
        """
        return TraceBlock(self, name, device, **metadata)

    def export_chrome_trace(self, path: Path) -> None:
        """Export events to Chrome Trace Event format.
        
        Generates a JSON file compatible with chrome://tracing for visualization.
        Events are organized by device on separate tracks.
        
        Args:
            path: Output file path for the trace JSON
            
        Raises:
            ValueError: If no events have been recorded
        """
        if not self.events:
            raise ValueError("No events to export")

        trace_events: list[dict[str, Any]] = []

        # Add process/thread metadata events
        trace_events.extend([
            {
                "name": "process_name",
                "ph": "M",
                "pid": 1,
                "tid": 1,
                "args": {"name": "GPU"}
            },
            {
                "name": "process_name",
                "ph": "M",
                "pid": 2,
                "tid": 2,
                "args": {"name": "ANE"}
            },
            {
                "name": "thread_name",
                "ph": "M",
                "pid": 1,
                "tid": 1,
                "args": {"name": "GPU Execution"}
            },
            {
                "name": "thread_name",
                "ph": "M",
                "pid": 2,
                "tid": 2,
                "args": {"name": "ANE Execution"}
            }
        ])

        # Convert trace events to Chrome format
        for event in self.events:
            pid = self._DEVICE_PID_MAP.get(event.device, 1)
            tid = self._DEVICE_TID_MAP.get(event.device, 1)

            # Duration event (complete event type 'X')
            chrome_event: dict[str, Any] = {
                "name": event.name,
                "ph": "X",  # Complete event (start + duration)
                "ts": event.start_us,
                "dur": event.end_us - event.start_us,
                "pid": pid,
                "tid": tid,
                "args": event.metadata if event.metadata else {}
            }
            trace_events.append(chrome_event)

        trace = {"traceEvents": trace_events}

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(trace, f, indent=2)

    def clear(self) -> None:
        """Clear all recorded events."""
        self.events.clear()
        self._start_time = None

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of recorded events.
        
        Returns:
            Dictionary with event counts and total time per device
        """
        gpu_events = [e for e in self.events if e.device == "gpu"]
        ane_events = [e for e in self.events if e.device == "ane"]

        gpu_time = sum(e.end_us - e.start_us for e in gpu_events)
        ane_time = sum(e.end_us - e.start_us for e in ane_events)

        # Calculate overlap
        all_intervals = [(e.start_us, e.end_us, e.device) for e in self.events]
        overlap_us = self._calculate_overlap(all_intervals)

        return {
            "total_events": len(self.events),
            "gpu_events": len(gpu_events),
            "ane_events": len(ane_events),
            "gpu_time_us": gpu_time,
            "ane_time_us": ane_time,
            "overlap_time_us": overlap_us,
            "utilization": {
                "gpu_ms": gpu_time / 1000,
                "ane_ms": ane_time / 1000,
                "overlap_ms": overlap_us / 1000
            }
        }

    def _calculate_overlap(
        self,
        intervals: list[tuple[int, int, str]]
    ) -> int:
        """Calculate time where both GPU and ANE are active.
        
        Args:
            intervals: List of (start, end, device) tuples
            
        Returns:
            Total overlap time in microseconds
        """
        gpu_intervals = [(s, e) for s, e, d in intervals if d == "gpu"]
        ane_intervals = [(s, e) for s, e, d in intervals if d == "ane"]

        overlap_us = 0
        for g_start, g_end in gpu_intervals:
            for a_start, a_end in ane_intervals:
                # Calculate intersection
                overlap_start = max(g_start, a_start)
                overlap_end = min(g_end, a_end)
                if overlap_start < overlap_end:
                    overlap_us += overlap_end - overlap_start

        return overlap_us


class TraceBlock:
    """Context manager for timing a block of code.
    
    Usage:
        tracer = PipelineTracer()
        tracer.start()
        
        with tracer.record_block("inference", "gpu"):
            # Code to time
            result = model.forward(input)
    """

    def __init__(
        self,
        tracer: PipelineTracer,
        name: str,
        device: str,
        **metadata: Any
    ):
        """Initialize the trace block.
        
        Args:
            tracer: Parent PipelineTracer instance
            name: Event name
            device: Target device ("gpu" or "ane")
            **metadata: Optional metadata
        """
        self._tracer = tracer
        self._name = name
        self._device = device
        self._metadata = metadata
        self._start_us: int = 0

    def __enter__(self) -> "TraceBlock":
        """Start timing the block."""
        self._start_us = int((time.perf_counter() - self._tracer._start_time) * 1_000_000) if self._tracer._start_time else 0
        return self

    def __exit__(self, *args: Any) -> None:
        """End timing and record the event."""
        if self._tracer._start_time:
            end_us = int((time.perf_counter() - self._tracer._start_time) * 1_000_000)
            duration_us = end_us - self._start_us
            self._tracer.record(
                self._name,
                self._device,
                duration_us,
                start_us=self._start_us,
                **self._metadata
            )


# Convenience function for quick tracing
def create_tracer() -> PipelineTracer:
    """Create and start a new PipelineTracer.
    
    Returns:
        A started PipelineTracer ready to record events
        
    Example:
        tracer = create_tracer()
        
        tracer.record("init", "gpu", 100)
        tracer.record("compute", "ane", 200)
        
        tracer.export_chrome_trace(Path("trace.json"))
    """
    tracer = PipelineTracer()
    tracer.start()
    return tracer
