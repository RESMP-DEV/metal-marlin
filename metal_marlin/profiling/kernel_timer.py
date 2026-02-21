"""Simple kernel timing profiler."""
from __future__ import annotations
import time
import logging
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List

_logger = logging.getLogger(__name__)
_global_timer: "KernelTimer | None" = None

class KernelTimer:
    """Tracks kernel execution times."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self._depth = 0
    
    @contextmanager
    def time(self, name: str):
        """Context manager to time a kernel/operation."""
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.timings[name].append(elapsed_ms)
    
    def report(self, top_n: int = 20) -> str:
        """Generate timing report."""
        if not self.timings:
            return "No timings recorded"
        
        lines = ["Kernel Timing Report", "=" * 60]
        stats = []
        for name, times in self.timings.items():
            total = sum(times)
            count = len(times)
            avg = total / count
            stats.append((total, name, count, avg, min(times), max(times)))
        
        stats.sort(reverse=True)  # By total time
        
        for total, name, count, avg, min_t, max_t in stats[:top_n]:
            lines.append(f"{name:40} total={total:8.2f}ms  calls={count:5d}  "
                         f"avg={avg:6.2f}ms  min={min_t:6.2f}ms  max={max_t:6.2f}ms")
        
        return "\n".join(lines)
    
    def reset(self):
        self.timings.clear()

def get_timer() -> KernelTimer:
    global _global_timer
    if _global_timer is None:
        _global_timer = KernelTimer()
    return _global_timer
