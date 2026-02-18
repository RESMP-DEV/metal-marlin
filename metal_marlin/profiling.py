"""Metal GPU Profiling utilities.

Provides Chrome Tracing export for Metal command buffer execution times.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from ._compat import HAS_PYOBJC_METAL

logger = logging.getLogger(__name__)


class MetalProfiler:
    """Metal GPU Profiler for capturing kernel execution times.

    Wraps command buffers to record GPU start/end timestamps and exports
    trace data in Chrome Tracing format.
    """

    def __init__(self, enabled: bool = True):
        """Initialize the profiler.

        Args:
            enabled: whether to enable profiling.
        """
        self.enabled = enabled and HAS_PYOBJC_METAL
        self.events: List[Dict[str, Any]] = []
        self._region_stack: List[str] = []
        self._start_time = time.time()  # CPU start time

        if enabled and not HAS_PYOBJC_METAL:
            logger.warning(
                "MetalProfiler enabled but PyObjC Metal not found. Profiling disabled."
            )

    def start_region(self, name: str) -> None:
        """Start a named profiling region.
        
        Any command buffers profiled while this region is active will be
        tagged with this name.
        """
        if not self.enabled:
            return
        self._region_stack.append(name)

    def end_region(self) -> None:
        """End the current profiling region."""
        if not self.enabled:
            return
        if self._region_stack:
            self._region_stack.pop()

    def profile_buffer(self, buffer: Any, name: Optional[str] = None) -> None:
        """Attach profiling handler to a command buffer.

        Must be called before the command buffer is committed.

        Args:
            buffer: The MTLCommandBuffer to profile.
            name: Optional override for region name.
        """
        if not self.enabled:
            return

        # Use current region if available, otherwise "Unknown"
        region_name = name or (self._region_stack[-1] if self._region_stack else "Global")

        def handler(cmd_buf: Any) -> None:
            # This callback runs on a background thread when GPU work completes
            if cmd_buf.status() == 5:  # MTLCommandBufferStatusError
                return

            # Get timestamps (seconds)
            start = cmd_buf.GPUStartTime()
            end = cmd_buf.GPUEndTime()

            # Filter invalid timestamps (0.0 means not recorded)
            if start == 0.0 or end == 0.0:
                return

            # Convert to microseconds for Chrome Trace
            self.events.append({
                "name": region_name,
                "ph": "X",  # Complete event
                "ts": start * 1e6,
                "dur": (end - start) * 1e6,
                "pid": 1,
                "tid": 1,  # GPU "thread"
                "cat": "gpu",
                "args": {
                    "desc": str(cmd_buf)
                }
            })

        buffer.addCompletedHandler_(handler)

    def export_trace(self, path: str) -> None:
        """Write Chrome trace JSON to file.

        Args:
            path: Output path for the JSON trace file.
        """
        if not self.events:
            logger.warning("No profiling events to export")
            return

        data = {
            "traceEvents": self.events,
            "displayTimeUnit": "ms",
            "otherData": {
                "version": "MetalProfiler v1.0"
            }
        }

        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Exported Metal trace to {path}")
        except Exception as e:
            logger.error(f"Failed to export trace: {e}")
