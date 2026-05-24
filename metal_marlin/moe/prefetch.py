"""Predictive expert prefetch helpers for MoE layers."""

from __future__ import annotations

import logging
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Callable

logger = logging.getLogger(__name__)


class PrefetchStrategy(str, Enum):
    """Strategies for predicting which experts should be prefetched."""

    TOP_K_RECENCY = "top_k_recency"
    FREQUENCY = "frequency"
    MARKOV = "markov"
    ATTENTION_AWARE = "attention_aware"


@dataclass
class PrefetchConfig:
    """Configuration for predictive expert prefetching."""

    strategy: PrefetchStrategy = PrefetchStrategy.TOP_K_RECENCY
    prefetch_k: int = 4
    history_window: int = 32
    decay_factor: float = 0.9
    min_confidence: float = 0.05
    async_threads: int = 2
    enable_stats: bool = True
    max_prefetch_queue: int = 16
    prediction_window: int = 1
    enable_attention_aware: bool = False


class ExpertPrefetcher:
    """Small asynchronous expert prefetcher used by MMFP4 MoE layers.

    The layer owns the authoritative expert modules. This helper predicts a
    short list of likely next experts, calls the provided load function in a
    background pool, and keeps the most recent loaded weights reachable for
    diagnostics. It is deliberately conservative: failures are counted and
    logged, but never interrupt inference.
    """

    def __init__(
        self,
        num_experts: int,
        num_layers: int,
        cache: Any,
        load_fn: Callable[[int, int], dict[str, Any]],
        config: PrefetchConfig | None = None,
    ) -> None:
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.cache = cache
        self.load_fn = load_fn
        self.config = config or PrefetchConfig()
        self._history: deque[tuple[int, tuple[int, ...]]] = deque(
            maxlen=max(1, self.config.history_window)
        )
        self._futures: list[Future[dict[str, Any]]] = []
        self._loaded: dict[tuple[int, int], dict[str, Any]] = {}
        self._lock = Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._started = False
        self._requests = 0
        self._hits = 0
        self._misses = 0
        self._failures = 0

    def start(self) -> None:
        """Start the background prefetch executor."""
        if self._started:
            return
        max_workers = max(1, self.config.async_threads)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="expert-prefetch",
        )
        self._started = True

    def stop(self) -> None:
        """Stop the background executor after draining outstanding work."""
        executor = self._executor
        self._executor = None
        self._started = False
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def step(
        self,
        layer_idx: int,
        current_indices: Any,
        attention_pattern: Any | None = None,
    ) -> None:
        """Record current routing and prefetch likely next-token experts."""
        del attention_pattern
        expert_ids = self._normalize_expert_ids(current_indices)
        if not expert_ids:
            return

        with self._lock:
            self._history.append((layer_idx, tuple(expert_ids)))

        predicted = self._predict(layer_idx, set(expert_ids))
        self.prefetch_experts(layer_idx, predicted)

    def prefetch_experts(self, layer_idx: int, expert_ids: list[int]) -> None:
        """Queue expert prefetches for a layer."""
        if not expert_ids:
            return
        if not self._started:
            self.start()

        assert self._executor is not None
        with self._lock:
            self._prune_done_locked()
            queued = {
                getattr(future, "_prefetch_key", None)
                for future in self._futures
                if not future.done()
            }
            capacity = max(0, self.config.max_prefetch_queue - len(self._futures))

            for expert_id in expert_ids[:capacity]:
                if not 0 <= expert_id < self.num_experts:
                    continue
                key = (layer_idx, expert_id)
                self._requests += 1
                if key in self._loaded:
                    self._hits += 1
                    continue
                self._misses += 1
                if key in queued:
                    continue
                future = self._executor.submit(self._load_one, layer_idx, expert_id)
                setattr(future, "_prefetch_key", key)
                self._futures.append(future)

    def wait_prefetch(self, timeout: float | None = None) -> None:
        """Wait briefly for queued prefetches and collect completed results."""
        with self._lock:
            futures = list(self._futures)
        if not futures:
            return

        done, _ = wait(futures, timeout=timeout)
        with self._lock:
            for future in done:
                self._collect_future_locked(future)
            self._prune_done_locked()

    def get_stats(self) -> dict[str, Any]:
        """Return prefetch counters for diagnostics and tests."""
        with self._lock:
            outstanding = sum(1 for future in self._futures if not future.done())
            loaded = len(self._loaded)
            history_len = len(self._history)
        total = self._hits + self._misses
        hit_rate = self._hits / total if total else 0.0
        return {
            "requests": self._requests,
            "hits": self._hits,
            "misses": self._misses,
            "failures": self._failures,
            "hit_rate": hit_rate,
            "outstanding": outstanding,
            "loaded": loaded,
            "history_len": history_len,
            "strategy": self.config.strategy.value,
        }

    def clear_history(self) -> None:
        """Clear routing history and cached prefetch results."""
        with self._lock:
            self._history.clear()
            self._loaded.clear()
            self._futures.clear()
            self._requests = 0
            self._hits = 0
            self._misses = 0
            self._failures = 0

    def _predict(self, layer_idx: int, current: set[int]) -> list[int]:
        with self._lock:
            history = [ids for layer, ids in self._history if layer == layer_idx]

        if not history:
            return sorted(current)[: self.config.prefetch_k]

        counts: Counter[int] = Counter()
        weight = 1.0
        for ids in reversed(history):
            for expert_id in ids:
                counts[expert_id] += weight
            weight *= self.config.decay_factor

        ranked = [expert for expert, _ in counts.most_common()]
        if self.config.strategy == PrefetchStrategy.TOP_K_RECENCY:
            recent = list(history[-1])
            ranked = recent + [expert for expert in ranked if expert not in recent]

        return [expert for expert in ranked if 0 <= expert < self.num_experts][
            : self.config.prefetch_k
        ]

    def _load_one(self, layer_idx: int, expert_id: int) -> dict[str, Any]:
        try:
            weights = self.load_fn(layer_idx, expert_id)
        except Exception:
            logger.exception("expert prefetch failed for layer=%s expert=%s", layer_idx, expert_id)
            with self._lock:
                self._failures += 1
            return {}

        with self._lock:
            self._loaded[(layer_idx, expert_id)] = weights
        return weights

    def _collect_future_locked(self, future: Future[dict[str, Any]]) -> None:
        try:
            future.result()
        except Exception:
            self._failures += 1

    def _prune_done_locked(self) -> None:
        retained = []
        for future in self._futures:
            if future.done():
                self._collect_future_locked(future)
            else:
                retained.append(future)
        self._futures = retained

    @staticmethod
    def _normalize_expert_ids(indices: Any) -> list[int]:
        if hasattr(indices, "detach"):
            indices = indices.detach().cpu().reshape(-1).tolist()
        elif hasattr(indices, "reshape") and hasattr(indices, "tolist"):
            indices = indices.reshape(-1).tolist()
        elif isinstance(indices, int):
            indices = [indices]

        return [int(expert_id) for expert_id in indices]
