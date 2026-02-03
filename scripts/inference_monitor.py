#!/usr/bin/env python3
"""
Terminal-based Inference Monitor Dashboard for Metal Marlin.

Real-time monitoring of inference performance metrics including:
- Tokens per second (TPS)
- Memory usage (GPU/CPU)
- Batch throughput
- Token latency distribution
- Expert routing (for MoE models)
- KV cache utilization

Usage:
    python inference_monitor.py --model-path /path/to/model --port 8888
    python inference_monitor.py --log-file inference.log
    python inference_monitor.py --refresh-rate 0.5
"""

from __future__ import annotations

import argparse
import curses
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import psutil
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class InferenceMetrics:
    """Container for inference performance metrics."""
    
    timestamp: float = 0.0
    tokens_generated: int = 0
    time_elapsed: float = 0.0
    prefill_tokens: int = 0
    prefill_time: float = 0.0
    decode_tokens: int = 0
    decode_time: float = 0.0
    
    # Memory metrics
    gpu_memory_used: float = 0.0  # GB
    gpu_memory_total: float = 0.0  # GB
    cpu_memory_used: float = 0.0  # GB
    cpu_memory_percent: float = 0.0
    
    # Latency metrics
    token_latencies: list[float] = field(default_factory=list)
    
    # MoE metrics (if applicable)
    expert_loads: dict[int, int] = field(default_factory=dict)
    
    # KV cache metrics
    kv_cache_size_mb: float = 0.0
    kv_cache_utilization: float = 0.0
    
    @property
    def tokens_per_second(self) -> float:
        """Calculate overall TPS."""
        if self.time_elapsed > 0:
            return self.tokens_generated / self.time_elapsed
        return 0.0
    
    @property
    def prefill_tps(self) -> float:
        """Calculate prefill TPS."""
        if self.prefill_time > 0:
            return self.prefill_tokens / self.prefill_time
        return 0.0
    
    @property
    def decode_tps(self) -> float:
        """Calculate decode TPS."""
        if self.decode_time > 0:
            return self.decode_tokens / self.decode_time
        return 0.0
    
    @property
    def mean_latency(self) -> float:
        """Average token latency in ms."""
        if self.token_latencies:
            return sum(self.token_latencies) / len(self.token_latencies) * 1000
        return 0.0
    
    @property
    def p50_latency(self) -> float:
        """Median token latency in ms."""
        if self.token_latencies:
            sorted_lat = sorted(self.token_latencies)
            idx = len(sorted_lat) // 2
            return sorted_lat[idx] * 1000
        return 0.0
    
    @property
    def p99_latency(self) -> float:
        """99th percentile token latency in ms."""
        if self.token_latencies:
            sorted_lat = sorted(self.token_latencies)
            idx = int(len(sorted_lat) * 0.99)
            return sorted_lat[idx] * 1000
        return 0.0


class MetricsCollector:
    """Collects inference metrics from various sources."""
    
    def __init__(self, history_size: int = 100):
        self.history: deque[InferenceMetrics] = deque(maxlen=history_size)
        self.current = InferenceMetrics()
        self.start_time = time.time()
        
    def update_from_log(self, log_line: str) -> bool:
        """Parse and update metrics from log line."""
        try:
            # Try to parse as JSON
            if log_line.strip().startswith("{"):
                data = json.loads(log_line)
                self._update_from_dict(data)
                return True
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Parse text log patterns
        if "tokens/sec" in log_line or "TPS" in log_line:
            self._parse_tps_line(log_line)
            return True
        elif "memory" in log_line.lower():
            self._parse_memory_line(log_line)
            return True
            
        return False
    
    def _update_from_dict(self, data: dict[str, Any]) -> None:
        """Update metrics from dictionary."""
        self.current.timestamp = data.get("timestamp", time.time())
        self.current.tokens_generated = data.get("tokens_generated", 0)
        self.current.time_elapsed = data.get("time_elapsed", 0.0)
        self.current.prefill_tokens = data.get("prefill_tokens", 0)
        self.current.prefill_time = data.get("prefill_time", 0.0)
        self.current.decode_tokens = data.get("decode_tokens", 0)
        self.current.decode_time = data.get("decode_time", 0.0)
        
        if "gpu_memory_used" in data:
            self.current.gpu_memory_used = data["gpu_memory_used"]
        if "gpu_memory_total" in data:
            self.current.gpu_memory_total = data["gpu_memory_total"]
            
        if "token_latencies" in data:
            self.current.token_latencies = data["token_latencies"]
            
        if "expert_loads" in data:
            self.current.expert_loads = data["expert_loads"]
            
        self.history.append(self.current)
        self.current = InferenceMetrics()
    
    def _parse_tps_line(self, line: str) -> None:
        """Parse TPS from text log line."""
        # Example: "Generated 128 tokens in 2.5s (51.2 tokens/sec)"
        import re
        match = re.search(r"(\d+)\s+tokens.*?([\d.]+)s.*?([\d.]+)\s+tokens", line)
        if match:
            tokens = int(match.group(1))
            elapsed = float(match.group(2))
            self.current.tokens_generated = tokens
            self.current.time_elapsed = elapsed
            self.history.append(self.current)
            self.current = InferenceMetrics()
    
    def _parse_memory_line(self, line: str) -> None:
        """Parse memory usage from text log line."""
        # Example: "GPU Memory: 4.2 / 16.0 GB"
        import re
        match = re.search(r"([\d.]+)\s*/\s*([\d.]+)\s*GB", line)
        if match:
            self.current.gpu_memory_used = float(match.group(1))
            self.current.gpu_memory_total = float(match.group(2))
    
    def collect_system_metrics(self) -> None:
        """Collect current system metrics."""
        if HAS_TORCH and torch.backends.mps.is_available():
            try:
                # MPS memory stats
                allocated = torch.mps.current_allocated_memory() / 1e9
                self.current.gpu_memory_used = allocated
            except AttributeError:
                pass
        
        # CPU memory
        mem = psutil.virtual_memory()
        self.current.cpu_memory_used = mem.used / 1e9
        self.current.cpu_memory_percent = mem.percent
        
        self.current.timestamp = time.time()


class InferenceMonitor:
    """Terminal-based dashboard for monitoring inference."""
    
    def __init__(
        self,
        log_file: Path | None = None,
        refresh_rate: float = 1.0,
        history_size: int = 100,
    ):
        self.log_file = log_file
        self.refresh_rate = refresh_rate
        self.collector = MetricsCollector(history_size=history_size)
        self.running = False
        self.log_position = 0
        
    def read_log_updates(self) -> None:
        """Read new lines from log file."""
        if not self.log_file or not self.log_file.exists():
            return
            
        try:
            with open(self.log_file) as f:
                f.seek(self.log_position)
                for line in f:
                    self.collector.update_from_log(line)
                self.log_position = f.tell()
        except OSError:
            pass
    
    def run(self, stdscr: Any) -> None:
        """Main monitoring loop."""
        self.running = True
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(True)  # Non-blocking input
        
        # Color pairs
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        
        while self.running:
            # Update metrics
            self.read_log_updates()
            self.collector.collect_system_metrics()
            
            # Render dashboard
            self.render(stdscr)
            
            # Check for quit key
            key = stdscr.getch()
            if key in (ord('q'), ord('Q'), 27):  # q, Q, or ESC
                self.running = False
                break
            
            # Refresh rate
            time.sleep(self.refresh_rate)
    
    def render(self, stdscr: Any) -> None:
        """Render dashboard to terminal."""
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        
        # Title bar
        title = "═══ Metal Marlin Inference Monitor ═══"
        stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)
        
        row = 2
        
        # Latest metrics
        if self.collector.history:
            latest = self.collector.history[-1]
            
            # Performance metrics
            stdscr.addstr(row, 2, "Performance:", curses.A_BOLD | curses.color_pair(4))
            row += 1
            
            tps_color = self._get_tps_color(latest.tokens_per_second)
            stdscr.addstr(row, 4, f"Overall TPS:     {latest.tokens_per_second:>8.2f} tok/s", curses.color_pair(tps_color))
            row += 1
            
            if latest.prefill_tokens > 0:
                stdscr.addstr(row, 4, f"Prefill TPS:     {latest.prefill_tps:>8.2f} tok/s", curses.color_pair(tps_color))
                row += 1
            
            if latest.decode_tokens > 0:
                stdscr.addstr(row, 4, f"Decode TPS:      {latest.decode_tps:>8.2f} tok/s", curses.color_pair(tps_color))
                row += 1
            
            stdscr.addstr(row, 4, f"Tokens Generated: {latest.tokens_generated:>7}")
            row += 2
            
            # Memory metrics
            stdscr.addstr(row, 2, "Memory:", curses.A_BOLD | curses.color_pair(4))
            row += 1
            
            if latest.gpu_memory_total > 0:
                mem_pct = (latest.gpu_memory_used / latest.gpu_memory_total) * 100
                mem_color = self._get_memory_color(mem_pct)
                mem_bar = self._render_bar(mem_pct, width=30)
                stdscr.addstr(row, 4, f"GPU: {latest.gpu_memory_used:>5.2f} / {latest.gpu_memory_total:.2f} GB  {mem_bar}", curses.color_pair(mem_color))
                row += 1
            
            if latest.cpu_memory_percent > 0:
                cpu_color = self._get_memory_color(latest.cpu_memory_percent)
                cpu_bar = self._render_bar(latest.cpu_memory_percent, width=30)
                stdscr.addstr(row, 4, f"CPU: {latest.cpu_memory_used:>5.2f} GB ({latest.cpu_memory_percent:>5.1f}%)  {cpu_bar}", curses.color_pair(cpu_color))
                row += 1
            
            if latest.kv_cache_size_mb > 0:
                stdscr.addstr(row, 4, f"KV Cache: {latest.kv_cache_size_mb:.1f} MB ({latest.kv_cache_utilization:.1f}% full)")
                row += 1
            
            row += 1
            
            # Latency metrics
            if latest.token_latencies:
                stdscr.addstr(row, 2, "Token Latency:", curses.A_BOLD | curses.color_pair(4))
                row += 1
                stdscr.addstr(row, 4, f"Mean: {latest.mean_latency:>6.2f} ms")
                row += 1
                stdscr.addstr(row, 4, f"P50:  {latest.p50_latency:>6.2f} ms")
                row += 1
                stdscr.addstr(row, 4, f"P99:  {latest.p99_latency:>6.2f} ms")
                row += 2
            
            # Expert routing (for MoE models)
            if latest.expert_loads:
                stdscr.addstr(row, 2, "Expert Routing:", curses.A_BOLD | curses.color_pair(4))
                row += 1
                
                total_loads = sum(latest.expert_loads.values())
                for expert_id in sorted(latest.expert_loads.keys())[:10]:  # Show first 10
                    load = latest.expert_loads[expert_id]
                    pct = (load / total_loads * 100) if total_loads > 0 else 0
                    bar = self._render_bar(pct, width=20)
                    stdscr.addstr(row, 4, f"Expert {expert_id:>2}: {bar} {pct:>5.1f}%")
                    row += 1
                    if row >= height - 3:
                        break
                
                row += 1
        
        # TPS history sparkline
        if len(self.collector.history) > 1 and row < height - 5:
            stdscr.addstr(row, 2, "TPS History:", curses.A_BOLD | curses.color_pair(4))
            row += 1
            tps_values = [m.tokens_per_second for m in self.collector.history if m.tokens_per_second > 0]
            if tps_values:
                sparkline = self._render_sparkline(tps_values, width - 6)
                stdscr.addstr(row, 4, sparkline, curses.color_pair(1))
                row += 1
                
                # Min/Max
                min_tps = min(tps_values)
                max_tps = max(tps_values)
                stdscr.addstr(row, 4, f"Min: {min_tps:.1f}  Max: {max_tps:.1f}  Avg: {sum(tps_values)/len(tps_values):.1f}")
                row += 1
        
        # Footer
        footer = "Press 'q' or ESC to quit"
        stdscr.addstr(height - 1, (width - len(footer)) // 2, footer, curses.A_DIM)
        
        stdscr.refresh()
    
    def _get_tps_color(self, tps: float) -> int:
        """Get color based on TPS value."""
        if tps > 50:
            return 1  # Green
        elif tps > 20:
            return 2  # Yellow
        else:
            return 3  # Red
    
    def _get_memory_color(self, percent: float) -> int:
        """Get color based on memory usage."""
        if percent < 70:
            return 1  # Green
        elif percent < 85:
            return 2  # Yellow
        else:
            return 3  # Red
    
    def _render_bar(self, percent: float, width: int = 20) -> str:
        """Render a progress bar."""
        filled = int((percent / 100) * width)
        return "█" * filled + "░" * (width - filled)
    
    def _render_sparkline(self, values: list[float], width: int) -> str:
        """Render a sparkline chart."""
        if not values:
            return ""
        
        # Sample values to fit width
        if len(values) > width:
            step = len(values) / width
            sampled = [values[int(i * step)] for i in range(width)]
        else:
            sampled = values
        
        # Normalize to sparkline chars
        min_val = min(sampled)
        max_val = max(sampled)
        range_val = max_val - min_val if max_val > min_val else 1
        
        chars = "▁▂▃▄▅▆▇█"
        sparkline = ""
        for val in sampled:
            normalized = (val - min_val) / range_val
            idx = int(normalized * (len(chars) - 1))
            sparkline += chars[idx]
        
        return sparkline


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Terminal-based inference monitor for Metal Marlin"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to inference log file to monitor",
    )
    parser.add_argument(
        "--refresh-rate",
        type=float,
        default=1.0,
        help="Dashboard refresh rate in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=100,
        help="Number of metrics samples to keep in history (default: 100)",
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not HAS_TORCH:
        print("Warning: PyTorch not found. GPU metrics will be unavailable.", file=sys.stderr)
    
    # Create monitor
    monitor = InferenceMonitor(
        log_file=args.log_file,
        refresh_rate=args.refresh_rate,
        history_size=args.history_size,
    )
    
    # Run with curses
    try:
        curses.wrapper(monitor.run)
    except KeyboardInterrupt:
        pass
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
