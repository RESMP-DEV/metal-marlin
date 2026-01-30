import threading
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median
from time import time


@dataclass
class RequestMetrics:
    start_time: float = field(default_factory=time)
    end_time: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    endpoint: str = ""


class MetricsCollector:
    def __init__(self):
        self._lock = threading.Lock()
        self._requests: list[RequestMetrics] = []
        self._active_count = 0
        self._endpoint_counts: defaultdict[str, int] = defaultdict(int)
        self._queue_depth_callback = None

    def set_queue_depth_callback(self, callback):
        self._queue_depth_callback = callback

    def start_request(self, endpoint: str = "unknown") -> RequestMetrics:
        with self._lock:
            self._active_count += 1
            self._endpoint_counts[endpoint] += 1
        return RequestMetrics(endpoint=endpoint)

    def end_request(self, metrics: RequestMetrics):
        metrics.end_time = time()
        with self._lock:
            self._active_count -= 1
            self._requests.append(metrics)
            if len(self._requests) > 1000:
                self._requests = self._requests[-1000:]

    def end_request_by_endpoint(
        self, endpoint: str, prompt_tokens: int = 0, completion_tokens: int = 0
    ):
        req_metrics = RequestMetrics(
            endpoint=endpoint, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        self.end_request(req_metrics)

    def get_stats(self) -> dict:
        queue_depth = 0
        if self._queue_depth_callback:
            try:
                queue_depth = self._queue_depth_callback()
            except Exception:
                pass

        with self._lock:
            if not self._requests:
                return {
                    "total_requests": 0,
                    "active_requests": self._active_count,
                    "queue_depth": queue_depth,
                    "latency_ms": {},
                    "tokens": {},
                    "by_endpoint": dict(self._endpoint_counts),
                }

            latencies = [
                (r.end_time - r.start_time) * 1000 for r in self._requests if r.end_time is not None
            ]
            prompt_tokens = [r.prompt_tokens for r in self._requests]
            completion_tokens = [r.completion_tokens for r in self._requests]

            latencies_sorted = sorted(latencies)
            p50 = median(latencies)
            p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)] if latencies_sorted else 0
            p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)] if latencies_sorted else 0

            return {
                "total_requests": len(self._requests),
                "active_requests": self._active_count,
                "queue_depth": queue_depth,
                "latency_ms": {
                    "p50": round(p50, 2),
                    "p95": round(p95, 2),
                    "p99": round(p99, 2),
                },
                "tokens": {
                    "total_prompt": sum(prompt_tokens),
                    "total_completion": sum(completion_tokens),
                    "avg_per_request": round(
                        (sum(prompt_tokens) + sum(completion_tokens)) / len(self._requests),
                        2,
                    ),
                },
                "by_endpoint": dict(self._endpoint_counts),
            }

    def to_prometheus(self) -> str:
        stats = self.get_stats()
        lines = [
            "# HELP metal_marlin_requests_total Total number of requests",
            "# TYPE metal_marlin_requests_total counter",
            f"metal_marlin_requests_total {stats['total_requests']}",
            "",
            "# HELP metal_marlin_active_requests Number of currently active requests",
            "# TYPE metal_marlin_active_requests gauge",
            f"metal_marlin_active_requests {stats['active_requests']}",
        ]

        if stats["latency_ms"]:
            lines.extend(
                [
                    "",
                    "# HELP metal_marlin_latency_ms Request latency in milliseconds",
                    "# TYPE metal_marlin_latency_ms gauge",
                    f'metal_marlin_latency_ms{{quantile="0.5"}} {stats["latency_ms"]["p50"]}',
                    f'metal_marlin_latency_ms{{quantile="0.95"}} {stats["latency_ms"]["p95"]}',
                    f'metal_marlin_latency_ms{{quantile="0.99"}} {stats["latency_ms"]["p99"]}',
                ]
            )

        if stats["tokens"]:
            lines.extend(
                [
                    "",
                    "# HELP metal_marlin_tokens_total Total tokens processed",
                    "# TYPE metal_marlin_tokens_total counter",
                    f'metal_marlin_tokens_total{{type="prompt"}} {stats["tokens"]["total_prompt"]}',
                    f'metal_marlin_tokens_total{{type="completion"}} {stats["tokens"]["total_completion"]}',
                    "",
                    "# HELP metal_marlin_avg_tokens_per_request Average tokens per request",
                    "# TYPE metal_marlin_avg_tokens_per_request gauge",
                    f"metal_marlin_avg_tokens_per_request {stats['tokens']['avg_per_request']}",
                ]
            )

        for endpoint, count in stats.get("by_endpoint", {}).items():
            safe_endpoint = endpoint.replace("/", "_").replace("{", "").replace("}", "")
            lines.extend(
                [
                    "",
                    f"# HELP metal_marlin_requests_by_endpoint Requests to {endpoint}",
                    "# TYPE metal_marlin_requests_by_endpoint counter",
                    f'metal_marlin_requests_by_endpoint{{endpoint="{safe_endpoint}"}} {count}',
                ]
            )

        gpu_mem = self._get_gpu_memory_mb()
        if gpu_mem:
            lines.extend(
                [
                    "",
                    "# HELP metal_marlin_gpu_memory_mb GPU memory usage in MB",
                    "# TYPE metal_marlin_gpu_memory_mb gauge",
                    f"metal_marlin_gpu_memory_mb {gpu_mem}",
                ]
            )

        lines.extend(
            [
                "",
                "# HELP metal_marlin_queue_depth Current queue depth",
                "# TYPE metal_marlin_queue_depth gauge",
                f"metal_marlin_queue_depth {stats['queue_depth']}",
            ]
        )

        return "\n".join(lines)

    def _get_gpu_memory_mb(self) -> int | None:
        try:
            import torch

            if torch.backends.mps.is_available():
                return torch.mps.current_allocated_memory() // (1024 * 1024)
        except Exception:
            pass
        return None
