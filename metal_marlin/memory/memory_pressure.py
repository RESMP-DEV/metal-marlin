import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psutil
import torch


@dataclass
class MemoryPressureConfig:
    """Configuration for memory pressure detection.

    Attributes:
        warning_threshold_mb: Free memory MB to trigger warning state
        critical_threshold_mb: Free memory MB to trigger critical state
        check_interval_seconds: How often to check system memory
        enable_monitoring: Enable background monitoring
        pressure_history_window: Number of checks to keep for trend analysis
    """
    warning_threshold_mb: int = 4096    # 4GB warning
    critical_threshold_mb: int = 2048   # 2GB critical
    check_interval_seconds: float = 0.5
    enable_monitoring: bool = True
    pressure_history_window: int = 10


@dataclass
class MemoryPressureStats:
    """Statistics for memory pressure monitoring."""
    total_memory_mb: int = 0
    available_memory_mb: int = 0
    pressure_level: str = "normal"  # normal, warning, critical
    is_warning: bool = False
    is_critical: bool = False
    trend: str = "stable"  # increasing, decreasing, stable
    last_check_time: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_memory_mb": self.total_memory_mb,
            "available_memory_mb": self.available_memory_mb,
            "pressure_level": self.pressure_level,
            "is_warning": self.is_warning,
            "is_critical": self.is_critical,
            "trend": self.trend,
            "last_check_time": self.last_check_time,
        }


class MemoryPressureMonitor:
    """Monitors system memory pressure to prevent OOM errors.
    
    Features:
    - Periodic system memory checking via psutil
    - Warning/Critical state detection
    - Pressure trend analysis (increasing/decreasing usage)
    - Thread-safe status queries
    
    This allows the system to proactively reduce memory usage (e.g., clear caches,
    stop prefetching) before hitting OS limits or causing swap thrashing.
    """
    
    def __init__(self, config: MemoryPressureConfig | None = None) -> None:
        self.config = config or MemoryPressureConfig()
        self._stats = MemoryPressureStats()
        self._lock = threading.RLock()
        
        self._history: list[int] = []  # History of available memory
        self._last_check_time = 0.0
        
        # Initial check
        self._update_stats()
        
    def check_pressure(self) -> tuple[bool, bool]:
        """Check current memory pressure state.
        
        Returns:
            Tuple of (is_warning, is_critical)
        """
        self._update_stats_if_needed()
        with self._lock:
            return self._stats.is_warning, self._stats.is_critical
            
    def get_stats(self) -> dict[str, Any]:
        """Get current memory pressure statistics."""
        self._update_stats_if_needed()
        with self._lock:
            return self._stats.to_dict()
            
    def _update_stats_if_needed(self) -> None:
        """Update stats if check interval has passed."""
        now = time.time()
        if now - self._last_check_time >= self.config.check_interval_seconds:
            self._update_stats()
            
    def _update_stats(self) -> None:
        """Perform memory check and update statistics."""
        if not self.config.enable_monitoring:
            return

        try:
            available_bytes = self._get_available_memory()
            available_mb = available_bytes // (1024 * 1024)
            total_mb = psutil.virtual_memory().total // (1024 * 1024)

            with self._lock:
                self._last_check_time = time.time()

                # Update history for trend analysis
                self._history.append(available_mb)
                if len(self._history) > self.config.pressure_history_window:
                    self._history.pop(0)

                # Determine state
                is_critical = available_mb < self.config.critical_threshold_mb
                is_warning = available_mb < self.config.warning_threshold_mb

                pressure_level = "normal"
                if is_critical:
                    pressure_level = "critical"
                elif is_warning:
                    pressure_level = "warning"

                # Analyze trend (decreasing available memory = increasing pressure)
                trend = "stable"
                if len(self._history) >= 3:
                    recent = self._history[-3:]
                    if recent[0] > recent[1] > recent[2]:
                        trend = "increasing"  # Pressure increasing (memory decreasing)
                    elif recent[0] < recent[1] < recent[2]:
                        trend = "decreasing"  # Pressure decreasing (memory freeing)

                self._stats = MemoryPressureStats(
                    total_memory_mb=total_mb,
                    available_memory_mb=available_mb,
                    pressure_level=pressure_level,
                    is_warning=is_warning,
                    is_critical=is_critical,
                    trend=trend,
                    last_check_time=self._last_check_time,
                )
        except Exception:
            # Fallback if psutil fails
            pass

    def _get_available_memory(self) -> int:
        """Get available memory in bytes, platform-aware.

        Returns:
            Available memory in bytes for the current platform.
            - MPS: Uses system available memory (unified memory architecture)
            - CUDA: Uses GPU free memory
            - CPU: Uses system available memory
        """
        if torch.backends.mps.is_available():
            # MPS: use system available memory (unified)
            return psutil.virtual_memory().available
        elif torch.cuda.is_available():
            # CUDA: use GPU free memory
            free, _ = torch.cuda.mem_get_info()
            return free
        else:
            return psutil.virtual_memory().available

    @property
    def critical_threshold_bytes(self) -> int:
        """Get critical threshold in bytes."""
        return self.config.critical_threshold_mb * 1024 * 1024

    def check_headroom(self, required_bytes: int) -> bool:
        """Check if required_bytes can be allocated without hitting critical threshold.

        Args:
            required_bytes: Number of bytes that will be allocated.

        Returns:
            True if the allocation can proceed without hitting critical threshold,
            False otherwise.
        """
        available = self._get_available_memory()
        return (available - required_bytes) > self.critical_threshold_bytes

    def ensure_headroom(
        self,
        required_bytes: int,
        eviction_callback: Callable[[], int] | None = None
    ) -> bool:
        """Ensure headroom exists, evicting cached buffers if needed.

        This method proactively attempts to free memory by calling the eviction
        callback until either sufficient headroom is achieved or the callback
        can no longer free memory.

        Args:
            required_bytes: Number of bytes that need to be allocated.
            eviction_callback: Optional callback that evicts cached buffers and
                returns the number of bytes freed. Called repeatedly until
                sufficient headroom is achieved or no more memory can be freed.

        Returns:
            True if headroom was achieved (either already available or through
            eviction), False if eviction was insufficient.
        """
        # First, check if we already have enough headroom
        if self.check_headroom(required_bytes):
            return True

        # If no callback provided, we can't do anything
        if eviction_callback is None:
            return False

        # Attempt eviction until we have enough headroom or can't free more
        max_iterations = 10  # Prevent infinite loops
        for _ in range(max_iterations):
            freed = eviction_callback()
            if freed == 0:
                # No more memory could be freed
                break
            if self.check_headroom(required_bytes):
                return True

        return False


# Global memory pressure monitor instance
_global_pressure_monitor: MemoryPressureMonitor | None = None
_global_monitor_lock = threading.Lock()


def get_global_memory_pressure_monitor(
    config: MemoryPressureConfig | None = None
) -> MemoryPressureMonitor:
    """Get or create the global memory pressure monitor.
    
    Args:
        config: Configuration for the monitor
        
    Returns:
        Global MemoryPressureMonitor instance
    """
    global _global_pressure_monitor
    with _global_monitor_lock:
        if _global_pressure_monitor is None:
            _global_pressure_monitor = MemoryPressureMonitor(config)
        return _global_pressure_monitor
