#!/usr/bin/env python3
"""
Metal Marlin Real-Time Performance Dashboard

Tracks and displays:
- Tokens per second (generation throughput)
- GPU utilization % (simulated/estimated)
- Memory usage (RAM and Metal)

Usage:
    python perf_dashboard.py
    python perf_dashboard.py --web --port 8080
    python perf_dashboard.py --refresh 100

Press 'q' or Ctrl+C to exit.
"""

from __future__ import annotations

import argparse
import curses
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

# Optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Check for Torch/MPS
try:
    import torch
    HAS_TORCH = True
    HAS_MPS = torch.backends.mps.is_available() and torch.backends.mps.is_built()
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False


@dataclass
class MetricPoint:
    """Single metric data point with timestamp."""
    timestamp: float
    value: float


class MetricHistory:
    """Time-series history for a metric."""

    def __init__(self, name: str, unit: str, max_points: int = 1000):
        self.name = name
        self.unit = unit
        self.data: deque[MetricPoint] = deque(maxlen=max_points)

    def add(self, value: float) -> None:
        """Add a new data point."""
        self.data.append(MetricPoint(time.time(), value))

    def latest(self) -> float:
        """Get the most recent value."""
        return self.data[-1].value if self.data else 0.0

    def average(self, seconds: int = 5) -> float:
        """Get average value over the last N seconds."""
        if not self.data:
            return 0.0
        cutoff = time.time() - seconds
        values = [p.value for p in self.data if p.timestamp >= cutoff]
        return sum(values) / len(values) if values else 0.0


class PerformanceCollector:
    """Collects real-time performance metrics."""

    def __init__(self, max_history: int = 1000):
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._demo_mode = False

        # Metrics
        self.tokens_per_second = MetricHistory("Tokens/Second", "tok/s", max_history)
        self.gpu_utilization = MetricHistory("GPU Utilization", "%", max_history)
        self.memory_metal = MetricHistory("Metal Memory", "MB", max_history)
        self.memory_ram = MetricHistory("RAM Usage", "MB", max_history)

        # Internal state
        self._last_mps_memory = 0
        self._mps_memory_samples: list[float] = []

    def _get_mps_memory(self) -> float:
        """Get current Metal memory usage in MB."""
        if not HAS_TORCH or not HAS_MPS:
            return 0.0
        try:
            # current_allocated_memory returns bytes
            return torch.mps.current_allocated_memory() / 1024.0 / 1024.0
        except Exception:
            return 0.0

    def _get_ram_usage(self) -> float:
        """Get current RAM usage in MB."""
        if not HAS_PSUTIL:
            return 0.0
        try:
            return psutil.Process().memory_info().rss / 1024.0 / 1024.0
        except Exception:
            return 0.0

    def _estimate_gpu_util(self, current_mem: float) -> float:
        """Estimate GPU utilization based on memory churn (heuristic)."""
        # In real scenario, we'd use powermetrics or Metal performance HUD
        # Here we use memory allocation changes as a proxy for activity
        if not hasattr(self, '_last_mem_check'):
            self._last_mem_check = current_mem
            return 0.0

        delta = abs(current_mem - self._last_mem_check)
        self._last_mem_check = current_mem

        # Heuristic: 10MB change per 100ms = ~10% util (tuned for demo)
        util = min(100.0, delta * 10.0)

        # Smooth it
        self._mps_memory_samples.append(util)
        if len(self._mps_memory_samples) > 5:
            self._mps_memory_samples.pop(0)

        return sum(self._mps_memory_samples) / len(self._mps_memory_samples)

    def _collect_loop(self, refresh_ms: int) -> None:
        """Background collection loop."""
        import random
        interval = refresh_ms / 1000.0

        while self._running:
            with self._lock:
                if self._demo_mode:
                    # Simulated data
                    tps = 45.0 + random.uniform(-5, 10)
                    gpu = 30.0 + random.uniform(-5, 20)
                    mem = 1500.0 + random.uniform(-50, 100)
                    ram = 800.0 + random.uniform(0, 10)

                    self.tokens_per_second.add(tps)
                    self.gpu_utilization.add(gpu)
                    self.memory_metal.add(mem)
                    self.memory_ram.add(ram)
                else:
                    # Real data collection
                    # Note: Tokens/sec needs to be fed from outside or queried from server
                    # For this standalone example, we'll default to 0 unless set

                    metal_mem = self._get_mps_memory()
                    self.memory_metal.add(metal_mem)

                    ram_mem = self._get_ram_usage()
                    self.memory_ram.add(ram_mem)

                    gpu_util = self._estimate_gpu_util(metal_mem)
                    self.gpu_utilization.add(gpu_util)

                    # If we can't get TPS, we add 0 (or keep last if implemented)
                    if not self.tokens_per_second.data:
                        self.tokens_per_second.add(0.0)
                    else:
                        # Just replicate last value if no update, or 0?
                        # Let's add 0 to show "idle"
                        self.tokens_per_second.add(0.0)

            time.sleep(interval)

    def start(self, refresh_ms: int = 100, demo: bool = False) -> None:
        if self._running:
            return
        self._running = True
        self._demo_mode = demo
        self._thread = threading.Thread(target=self._collect_loop, args=(refresh_ms,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "tokens_per_sec": {
                        "current": self.tokens_per_second.latest(),
                        "avg_5s": self.tokens_per_second.average(5)
                    },
                    "gpu_util": {
                        "current": self.gpu_utilization.latest(),
                        "avg_5s": self.gpu_utilization.average(5)
                    },
                    "metal_memory_mb": {
                        "current": self.memory_metal.latest(),
                    },
                    "ram_usage_mb": {
                        "current": self.memory_ram.latest(),
                    }
                }
            }


class DashboardHandler(BaseHTTPRequestHandler):
    """Simple JSON API handler."""

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"""
            <html><head><meta http-equiv=\"refresh\" content=\"0.5\"></head>
            <body style=\"font-family: monospace; background: #111; color: #0f0; padding: 20px;\">
            <h1>Metal Marlin Dashboard</h1>
            <p>Access <a href=\"/api/metrics\" style=\"color: #0ff\">/api/metrics</a> for JSON data.</p>
            </body></html>
            """)
        elif self.path == '/api/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            snapshot = collector.get_snapshot()
            self.wfile.write(json.dumps(snapshot, indent=2).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Quiet logging


def run_web_server(port: int):
    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    server.serve_forever()


def draw_tui(stdscr, collector: PerformanceCollector, refresh_ms: int):
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_CYAN, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)

    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        # Header
        title = " Metal Marlin Performance Dashboard "
        stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD | curses.color_pair(2))
        stdscr.addstr(1, 0, "=" * (width - 1), curses.color_pair(2))

        # Snapshot
        data = collector.get_snapshot()["metrics"]

        # Layout
        col_w = width // 2

        # Tokens/Sec
        tps = data["tokens_per_sec"]["current"]
        tps_avg = data["tokens_per_sec"]["avg_5s"]
        stdscr.addstr(3, 2, "Tokens / Second", curses.A_BOLD)
        stdscr.addstr(4, 2, f"{tps:>6.1f} tok/s (curr)", curses.color_pair(1))
        stdscr.addstr(5, 2, f"{tps_avg:>6.1f} tok/s (avg)", curses.color_pair(3))

        # GPU Util
        gpu = data["gpu_util"]["current"]
        stdscr.addstr(3, col_w + 2, "GPU Utilization", curses.A_BOLD)
        bar_len = 20
        filled = int((gpu / 100.0) * bar_len)
        bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
        stdscr.addstr(4, col_w + 2, f"{bar} {gpu:>5.1f}%", curses.color_pair(1))

        # Memory
        mem_metal = data["metal_memory_mb"]["current"]
        mem_ram = data["ram_usage_mb"]["current"]

        stdscr.addstr(8, 2, "Memory Usage", curses.A_BOLD)
        stdscr.addstr(9, 2, f"Metal: {mem_metal:>6.1f} MB", curses.color_pair(2))
        stdscr.addstr(10, 2, f"RAM:   {mem_ram:>6.1f} MB", curses.color_pair(2))

        # Footer
        footer = "Press 'q' to quit"
        stdscr.addstr(height - 2, (width - len(footer)) // 2, footer, curses.A_DIM)

        stdscr.refresh()

        # Input handling
        stdscr.timeout(refresh_ms)
        c = stdscr.getch()
        if c == ord('q'):
            break

# Global for web server access
collector = None

def main():
    global collector

    parser = argparse.ArgumentParser(description="Metal Marlin Performance Dashboard")
    parser.add_argument("--web", action="store_true", help="Enable web dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Web port (default: 8080)")
    parser.add_argument("--refresh", type=int, default=100, help="Refresh interval (ms)")
    parser.add_argument("--demo", action="store_true", help="Run with simulated data")

    args = parser.parse_args()

    collector = PerformanceCollector(max_history=1000)
    collector.start(refresh_ms=args.refresh, demo=args.demo)

    try:
        if args.web:
            web_thread = threading.Thread(target=run_web_server, args=(args.port,), daemon=True)
            web_thread.start()
            print(f"Web dashboard running at http://localhost:{args.port}")

        # Run TUI
        try:
            curses.wrapper(draw_tui, collector, args.refresh)
        except curses.error:
            # Fallback if terminal issues
            print("Error initializing TUI. Running in text mode.")
            while True:
                data = collector.get_snapshot()["metrics"]
                print(f"\rTPS: {data['tokens_per_sec']['current']:.1f} | GPU: {data['gpu_util']['current']:.1f}% | Mem: {data['metal_memory_mb']['current']:.1f}MB", end="")
                time.sleep(args.refresh / 1000.0)

    except KeyboardInterrupt:
        pass
    finally:
        collector.stop()
        print("\nExiting.")

if __name__ == "__main__":
    main()
