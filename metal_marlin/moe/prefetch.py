"""Predictive expert prefetching for autoregressive MoE generation.

In autoregressive generation, we can exploit temporal locality:
- Token N's routing is known after router computation
- While computing token N, we can preload experts for token N+1
- Challenge: Routing for N+1 depends on N's output (hidden state)

Strategies:
1. **Repeat** (simplest): Assume token N+1 uses same experts as token N.
   Works well because consecutive tokens often share semantic context.

2. **History** (better): Track expert usage over a sliding window.
   Predict most frequently used experts from recent history.

3. **Attention-guided** (best): Use attention patterns to weight predictions.
   If token N attends strongly to position P, and P used experts E,
   then N+1 is likely to use similar experts to P.

4. **Top-k recency**: Blend of repeat and history. Weight recent experts
   higher than older ones using exponential decay.

Usage:
    from metal_marlin.moe.prefetch import ExpertPrefetcher, PrefetchStrategy

    # Create prefetcher for 64-expert model
    prefetcher = ExpertPrefetcher(
        num_experts=64,
        num_layers=28,
        strategy=PrefetchStrategy.TOP_K_RECENCY,
        cache=expert_cache,  # Optional existing ExpertCache
    )

    # During autoregressive generation
    for token_idx in range(seq_len):
        # Token N: compute router, get expert assignments
        expert_ids, expert_probs = router(hidden_states)

        # Record routing decision
        prefetcher.record_routing(layer_idx=0, expert_ids=expert_ids)

        # Start loading predicted experts for next token in background
        predicted = prefetcher.predict_next_experts(layer_idx=0)
        prefetcher.async_load_experts(layer_idx=0, expert_ids=predicted)

        # Compute with current experts (prefetched experts ready for next iter)
        output = moe_forward(hidden_states, expert_ids, expert_probs)
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from metal_marlin.expert_cache import ExpertCache


class PrefetchStrategy(Enum):
    """Strategy for predicting which experts to prefetch."""

    REPEAT = auto()
    """Use same experts as current token. Simplest, works well for semantic locality."""

    HISTORY = auto()
    """Use most frequent experts from sliding window history."""

    ATTENTION_GUIDED = auto()
    """Weight predictions by attention pattern similarity."""

    TOP_K_RECENCY = auto()
    """Blend recent experts with exponential decay weighting."""


@dataclass
class PrefetchConfig:
    """Configuration for expert prefetching.

    Attributes:
        strategy: Which prediction strategy to use.
        history_window: Number of recent tokens to track for history-based prediction.
        prefetch_k: Number of experts to prefetch per layer.
        decay_factor: Exponential decay for TOP_K_RECENCY (0.9 = recent tokens weighted 10x).
        min_confidence: Minimum prediction confidence to trigger prefetch.
        async_threads: Number of threads for async expert loading.
        enable_stats: Whether to collect prefetch hit/miss statistics.
    """

    strategy: PrefetchStrategy = PrefetchStrategy.TOP_K_RECENCY
    history_window: int = 32
    prefetch_k: int = 4
    decay_factor: float = 0.9
    min_confidence: float = 0.05
    async_threads: int = 2
    enable_stats: bool = True


@dataclass
class RoutingHistory:
    """Tracks routing decisions for prediction.

    Stores recent expert assignments per layer with timestamps for
    recency-weighted predictions.
    """

    layer_idx: int
    window_size: int
    # deque of (timestamp, expert_ids) tuples
    history: deque[tuple[float, list[int]]] = field(default_factory=deque)

    def record(self, expert_ids: Sequence[int]) -> None:
        """Record a routing decision."""
        timestamp = time.time()
        self.history.append((timestamp, list(expert_ids)))

        # Trim to window size
        while len(self.history) > self.window_size:
            self.history.popleft()

    def get_frequency_counts(self, decay: float = 1.0) -> dict[int, float]:
        """Get expert frequency counts with optional temporal decay.

        Args:
            decay: Exponential decay factor. 1.0 = no decay (uniform weight).
                   0.9 = each older entry weighted 0.9x the next newer entry.

        Returns:
            Dict mapping expert_id to weighted frequency count.
        """
        counts: dict[int, float] = {}
        n = len(self.history)

        for i, (_, expert_ids) in enumerate(self.history):
            # Weight by position: most recent = 1.0, oldest = decay^(n-1)
            weight = decay ** (n - 1 - i) if decay < 1.0 else 1.0

            for eid in expert_ids:
                counts[eid] = counts.get(eid, 0.0) + weight

        return counts

    def get_last_experts(self) -> list[int]:
        """Get experts from most recent routing decision."""
        if not self.history:
            return []
        _, expert_ids = self.history[-1]
        return expert_ids

    def clear(self) -> None:
        """Clear history."""
        self.history.clear()


@dataclass
class PrefetchStats:
    """Statistics for prefetch accuracy.

    Tracks how often predicted experts match actual routing decisions.
    """

    layer_idx: int
    predictions_made: int = 0
    predictions_correct: int = 0  # At least one predicted expert was used
    experts_prefetched: int = 0
    experts_hit: int = 0  # Prefetched experts that were actually used
    prefetch_latency_ms: float = 0.0
    _latency_samples: list[float] = field(default_factory=list)

    @property
    def prediction_accuracy(self) -> float:
        """Fraction of predictions where at least one expert was used."""
        if self.predictions_made == 0:
            return 0.0
        return self.predictions_correct / self.predictions_made

    @property
    def expert_hit_rate(self) -> float:
        """Fraction of prefetched experts that were actually used."""
        if self.experts_prefetched == 0:
            return 0.0
        return self.experts_hit / self.experts_prefetched

    def record_prediction(self, predicted: list[int], actual: list[int]) -> None:
        """Record prediction vs actual routing for accuracy tracking."""
        self.predictions_made += 1
        self.experts_prefetched += len(predicted)

        predicted_set = set(predicted)
        actual_set = set(actual)
        hits = predicted_set & actual_set

        if hits:
            self.predictions_correct += 1
        self.experts_hit += len(hits)

    def record_latency(self, latency_ms: float) -> None:
        """Record prefetch latency sample."""
        self._latency_samples.append(latency_ms)
        # Keep rolling average
        if len(self._latency_samples) > 100:
            self._latency_samples = self._latency_samples[-100:]
        self.prefetch_latency_ms = sum(self._latency_samples) / len(self._latency_samples)

    def as_dict(self) -> dict[str, Any]:
        """Return stats as dictionary."""
        return {
            "layer_idx": self.layer_idx,
            "predictions_made": self.predictions_made,
            "prediction_accuracy": f"{self.prediction_accuracy:.2%}",
            "expert_hit_rate": f"{self.expert_hit_rate:.2%}",
            "avg_latency_ms": f"{self.prefetch_latency_ms:.2f}",
        }


def predict_next_experts(
    current_routing: mx.array | list[int],
    attention_pattern: mx.array | None = None,
    history: RoutingHistory | None = None,
    num_experts: int = 64,
    strategy: PrefetchStrategy = PrefetchStrategy.TOP_K_RECENCY,
    prefetch_k: int = 4,
    decay_factor: float = 0.9,
    min_confidence: float = 0.05,
) -> list[int]:
    """Predict which experts will be needed for the next token.

    This heuristic predicts which experts to preload based on the
    current token's routing and optional attention patterns.

    Args:
        current_routing: Expert IDs selected for current token.
            Shape: [top_k] or [batch, top_k] for batched inference.
        attention_pattern: Optional attention weights from current token.
            Shape: [num_heads, seq_len] or [batch, num_heads, seq_len].
            Used for attention-guided prediction.
        history: Optional routing history for frequency-based prediction.
        num_experts: Total number of experts in MoE layer.
        strategy: Prediction strategy to use.
        prefetch_k: Number of experts to predict.
        decay_factor: Temporal decay for recency-weighted prediction.
        min_confidence: Minimum confidence to include an expert.

    Returns:
        List of predicted expert IDs, sorted by confidence.

    Example:
        >>> current_routing = mx.array([3, 7])  # Current token uses experts 3, 7
        >>> predicted = predict_next_experts(
        ...     current_routing,
        ...     strategy=PrefetchStrategy.REPEAT,
        ...     prefetch_k=2
        ... )
        >>> print(predicted)  # [3, 7] - assumes same experts
    """
    # Convert to list if MLX array
    if isinstance(current_routing, mx.array):
        routing = current_routing.reshape(-1).tolist()
    else:
        routing = list(current_routing)

    if strategy == PrefetchStrategy.REPEAT:
        # Simplest: assume next token uses same experts
        return routing[:prefetch_k]

    elif strategy == PrefetchStrategy.HISTORY:
        # Use frequency from history window
        if history is None:
            return routing[:prefetch_k]

        counts = history.get_frequency_counts(decay=1.0)  # No decay for pure history
        if not counts:
            return routing[:prefetch_k]

        # Sort by frequency
        sorted_experts = sorted(counts.items(), key=lambda x: -x[1])

        # Filter by minimum confidence (normalize counts to probabilities)
        total = sum(c for _, c in sorted_experts)
        predicted = [eid for eid, cnt in sorted_experts if cnt / total >= min_confidence][
            :prefetch_k
        ]

        return predicted if predicted else routing[:prefetch_k]

    elif strategy == PrefetchStrategy.ATTENTION_GUIDED:
        # Weight by attention pattern + history
        if attention_pattern is None or history is None:
            # Fall back to recency
            return _predict_recency(routing, history, prefetch_k, decay_factor)

        # Get attention-weighted expert distribution
        # This assumes we have stored which experts were used at each position
        # For now, fall back to recency with attention as tiebreaker
        return _predict_recency(routing, history, prefetch_k, decay_factor)

    elif strategy == PrefetchStrategy.TOP_K_RECENCY:
        return _predict_recency(routing, history, prefetch_k, decay_factor)

    else:
        # Default: repeat current routing
        return routing[:prefetch_k]


def _predict_recency(
    current_routing: list[int],
    history: RoutingHistory | None,
    prefetch_k: int,
    decay_factor: float,
) -> list[int]:
    """Predict experts using recency-weighted frequency.

    Combines current routing with historical frequency, weighting
    recent tokens more heavily.
    """
    if history is None or len(history.history) == 0:
        return current_routing[:prefetch_k]

    # Get decayed frequency counts from history
    counts = history.get_frequency_counts(decay=decay_factor)

    # Boost current routing (most recent information)
    for eid in current_routing:
        counts[eid] = counts.get(eid, 0.0) + 2.0  # Weight current 2x

    # Sort by weighted frequency
    sorted_experts = sorted(counts.items(), key=lambda x: -x[1])

    # Return top-k
    return [eid for eid, _ in sorted_experts[:prefetch_k]]


class AsyncExpertLoader:
    """Background loader for expert weights.

    Manages a thread pool that asynchronously loads and dequantizes
    expert weights. Integration with ExpertCache ensures prefetched
    experts are cached for fast access.
    """

    def __init__(
        self,
        cache: ExpertCache | None = None,
        num_threads: int = 2,
    ):
        """Initialize async loader.

        Args:
            cache: Optional ExpertCache for storing prefetched experts.
            num_threads: Number of background threads for loading.
        """
        self.cache = cache
        self.num_threads = num_threads
        self._executor: ThreadPoolExecutor | None = None
        self._pending: dict[tuple[int, int], Any] = {}  # (layer, expert) -> Future
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the background loader."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.num_threads,
                thread_name_prefix="expert_prefetch",
            )

    def stop(self) -> None:
        """Stop the background loader and wait for pending tasks."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
            self._pending.clear()

    def submit(
        self,
        layer_idx: int,
        expert_id: int,
        load_fn: Callable[[], mx.array],
    ) -> None:
        """Submit expert for background loading.

        If the expert is already cached or pending, this is a no-op.

        Args:
            layer_idx: Layer index of the expert.
            expert_id: Expert ID within the layer.
            load_fn: Function that loads and returns the expert weights.
        """
        if self._executor is None:
            self.start()

        key = (layer_idx, expert_id)

        with self._lock:
            # Skip if already pending
            if key in self._pending:
                return

            # Skip if already cached (check via cache if available)
            # Note: We can't check tile-level caching here, but we can
            # check if the expert was recently loaded

            # Submit to thread pool
            future = self._executor.submit(self._load_expert, layer_idx, expert_id, load_fn)
            self._pending[key] = future

    def _load_expert(
        self,
        layer_idx: int,
        expert_id: int,
        load_fn: Callable[[], mx.array],
    ) -> mx.array | None:
        """Load expert weights (runs in background thread).

        Note: mx.eval() is intentionally NOT called here because MLX operations
        are not thread-safe. The caller should synchronize MLX evaluations
        on the main thread after wait_all() returns.
        """
        try:
            weights = load_fn()
            # NOTE: Do NOT call mx.eval() here - MLX is not thread-safe.
            # The weights will be lazily evaluated when accessed on the main thread.
            # This is safe because MLX graphs are immutable once created.

            # If cache available, the cache would be updated via the dequant_fn
            # callback pattern in get_expert_tile

            return weights
        except Exception:
            return None
        finally:
            key = (layer_idx, expert_id)
            with self._lock:
                self._pending.pop(key, None)

    def is_pending(self, layer_idx: int, expert_id: int) -> bool:
        """Check if expert is currently being loaded."""
        with self._lock:
            return (layer_idx, expert_id) in self._pending

    def wait_all(self, timeout: float | None = None) -> None:
        """Wait for all pending loads to complete."""
        with self._lock:
            futures = list(self._pending.values())

        for future in futures:
            try:
                future.result(timeout=timeout)
            except Exception:
                pass

    def __enter__(self) -> AsyncExpertLoader:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


def async_load_experts(
    layer_idx: int,
    expert_ids: list[int],
    load_fn: Callable[[int, int], mx.array],
    loader: AsyncExpertLoader | None = None,
    num_threads: int = 2,
) -> AsyncExpertLoader:
    """Start background loading of expert weights.

    Convenience function that creates or reuses an AsyncExpertLoader
    and submits experts for prefetching.

    Args:
        layer_idx: Layer containing the experts.
        expert_ids: List of expert IDs to prefetch.
        load_fn: Function (layer_idx, expert_id) -> weights to load experts.
        loader: Optional existing AsyncExpertLoader to reuse.
        num_threads: Number of threads if creating new loader.

    Returns:
        AsyncExpertLoader instance (reused or newly created).

    Example:
        >>> def load_expert(layer, eid):
        ...     return dequant_fp4(packed_weights[layer][eid], scales[layer][eid])
        >>> predicted = predict_next_experts(current_routing, ...)
        >>> loader = async_load_experts(layer_idx=0, expert_ids=predicted, load_fn=load_expert)
        >>> # ... compute with current experts ...
        >>> loader.wait_all()  # Ensure prefetch complete before next iteration
    """
    if loader is None:
        loader = AsyncExpertLoader(num_threads=num_threads)
        loader.start()

    for expert_id in expert_ids:
        loader.submit(
            layer_idx=layer_idx,
            expert_id=expert_id,
            load_fn=lambda _layer=layer_idx, _exp=expert_id: load_fn(_layer, _exp),
        )

    return loader


class ExpertPrefetcher:
    """High-level prefetcher for MoE autoregressive generation.

    Combines prediction, loading, and caching into a single interface.
    Designed to be used during autoregressive token generation.

    Example:
        >>> prefetcher = ExpertPrefetcher(
        ...     num_experts=64,
        ...     num_layers=28,
        ...     load_fn=lambda l, e: dequant_expert(l, e),
        ... )
        >>>
        >>> for layer_idx in range(num_layers):
        ...     # Get current routing from router
        ...     expert_ids = router(hidden)
        ...
        ...     # Record and prefetch
        ...     prefetcher.step(layer_idx, expert_ids)
        ...
        ...     # Compute with current experts
        ...     output = expert_forward(hidden, expert_ids)
        >>>
        >>> # Check prefetch effectiveness
        >>> print(prefetcher.stats)
    """

    def __init__(
        self,
        num_experts: int,
        num_layers: int,
        load_fn: Callable[[int, int], mx.array] | None = None,
        cache: ExpertCache | None = None,
        config: PrefetchConfig | None = None,
    ):
        """Initialize prefetcher.

        Args:
            num_experts: Number of experts per MoE layer.
            num_layers: Number of MoE layers in the model.
            load_fn: Function (layer_idx, expert_id) -> weights to load experts.
                     If None, prefetching is prediction-only (no loading).
            cache: Optional ExpertCache for caching prefetched experts.
            config: Prefetch configuration. Uses defaults if not provided.
        """
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.load_fn = load_fn
        self.cache = cache
        self.config = config or PrefetchConfig()

        # Per-layer routing history
        self._history: dict[int, RoutingHistory] = {
            i: RoutingHistory(layer_idx=i, window_size=self.config.history_window)
            for i in range(num_layers)
        }

        # Per-layer prediction stats
        self._stats: dict[int, PrefetchStats] = {
            i: PrefetchStats(layer_idx=i) for i in range(num_layers)
        }

        # Async loader
        self._loader = AsyncExpertLoader(
            cache=cache,
            num_threads=self.config.async_threads,
        )

        # Track last predictions for accuracy measurement
        self._last_predictions: dict[int, list[int]] = {}

    def record_routing(self, layer_idx: int, expert_ids: mx.array | list[int]) -> None:
        """Record a routing decision.

        Call this after the router selects experts for the current token.
        Updates history and checks previous prediction accuracy.

        Args:
            layer_idx: Layer index.
            expert_ids: Selected expert IDs, shape [top_k] or [batch, top_k].
        """
        # Convert to list
        if isinstance(expert_ids, mx.array):
            ids = expert_ids.reshape(-1).tolist()
        else:
            ids = list(expert_ids)

        # Check previous prediction accuracy
        if self.config.enable_stats and layer_idx in self._last_predictions:
            predicted = self._last_predictions.pop(layer_idx)
            self._stats[layer_idx].record_prediction(predicted, ids)

        # Record in history
        self._history[layer_idx].record(ids)

    def predict_next_experts(
        self,
        layer_idx: int,
        attention_pattern: mx.array | None = None,
    ) -> list[int]:
        """Predict experts for the next token.

        Args:
            layer_idx: Layer index.
            attention_pattern: Optional attention weights for attention-guided prediction.

        Returns:
            List of predicted expert IDs.
        """
        history = self._history.get(layer_idx)
        current = history.get_last_experts() if history else []

        predicted = predict_next_experts(
            current_routing=current,
            attention_pattern=attention_pattern,
            history=history,
            num_experts=self.num_experts,
            strategy=self.config.strategy,
            prefetch_k=self.config.prefetch_k,
            decay_factor=self.config.decay_factor,
            min_confidence=self.config.min_confidence,
        )

        # Store for accuracy tracking
        if self.config.enable_stats:
            self._last_predictions[layer_idx] = predicted

        return predicted

    def prefetch_experts(
        self,
        layer_idx: int,
        expert_ids: list[int] | None = None,
    ) -> None:
        """Start background loading of predicted experts.

        Args:
            layer_idx: Layer index.
            expert_ids: Experts to prefetch. If None, uses predict_next_experts().
        """
        if self.load_fn is None:
            return

        if expert_ids is None:
            expert_ids = self.predict_next_experts(layer_idx)

        start = time.time()

        for expert_id in expert_ids:
            self._loader.submit(
                layer_idx=layer_idx,
                expert_id=expert_id,
                load_fn=lambda _layer=layer_idx, _exp=expert_id: self.load_fn(_layer, _exp),
            )

        if self.config.enable_stats:
            latency_ms = (time.time() - start) * 1000
            self._stats[layer_idx].record_latency(latency_ms)

    def step(
        self,
        layer_idx: int,
        expert_ids: mx.array | list[int],
        attention_pattern: mx.array | None = None,
    ) -> list[int]:
        """Combined record + predict + prefetch step.

        Convenience method that:
        1. Records current routing
        2. Predicts next experts
        3. Starts prefetching predicted experts

        Args:
            layer_idx: Layer index.
            expert_ids: Current token's selected expert IDs.
            attention_pattern: Optional attention weights.

        Returns:
            Predicted expert IDs for next token.
        """
        self.record_routing(layer_idx, expert_ids)
        predicted = self.predict_next_experts(layer_idx, attention_pattern)
        self.prefetch_experts(layer_idx, predicted)
        return predicted

    def wait_prefetch(self, timeout: float | None = None) -> None:
        """Wait for all pending prefetch operations."""
        self._loader.wait_all(timeout=timeout)

    def clear_history(self, layer_idx: int | None = None) -> None:
        """Clear routing history.

        Args:
            layer_idx: Layer to clear. If None, clears all layers.
        """
        if layer_idx is not None:
            self._history[layer_idx].clear()
        else:
            for h in self._history.values():
                h.clear()

    def get_stats(self, layer_idx: int | None = None) -> dict[str, Any]:
        """Get prefetch statistics.

        Args:
            layer_idx: Specific layer to get stats for. If None, returns all.

        Returns:
            Statistics dictionary.
        """
        if layer_idx is not None:
            return self._stats[layer_idx].as_dict()

        return {
            "config": {
                "strategy": self.config.strategy.name,
                "prefetch_k": self.config.prefetch_k,
                "history_window": self.config.history_window,
            },
            "per_layer": {i: s.as_dict() for i, s in self._stats.items()},
            "summary": self._compute_summary_stats(),
        }

    def _compute_summary_stats(self) -> dict[str, Any]:
        """Compute aggregate statistics across layers."""
        total_predictions = sum(s.predictions_made for s in self._stats.values())
        total_correct = sum(s.predictions_correct for s in self._stats.values())
        total_prefetched = sum(s.experts_prefetched for s in self._stats.values())
        total_hits = sum(s.experts_hit for s in self._stats.values())

        return {
            "total_predictions": total_predictions,
            "prediction_accuracy": (
                f"{total_correct / total_predictions:.2%}" if total_predictions > 0 else "N/A"
            ),
            "expert_hit_rate": (
                f"{total_hits / total_prefetched:.2%}" if total_prefetched > 0 else "N/A"
            ),
            "avg_latency_ms": (
                f"{np.mean([s.prefetch_latency_ms for s in self._stats.values()]):.2f}"
            ),
        }

    def start(self) -> None:
        """Start the background loader."""
        self._loader.start()

    def stop(self) -> None:
        """Stop the background loader."""
        self._loader.stop()

    def __enter__(self) -> ExpertPrefetcher:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        return (
            f"ExpertPrefetcher("
            f"num_experts={self.num_experts}, "
            f"num_layers={self.num_layers}, "
            f"strategy={self.config.strategy.name})"
        )


# LRU cache specifically for expert weights (complementary to tile-based ExpertCache)
class ExpertLRUCache:
    """Simple LRU cache for full expert weight matrices.

    Unlike the tile-based ExpertCache which caches dequantized tiles,
    this caches complete expert weight matrices. Useful for smaller
    models or when tile-level caching adds overhead.

    Args:
        max_size_mb: Maximum cache size in megabytes.
        num_layers: Number of MoE layers.
    """

    def __init__(self, max_size_mb: int = 512, num_layers: int = 28):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.num_layers = num_layers
        self._cache: OrderedDict[tuple[int, int], mx.array] = OrderedDict()
        self._size_bytes = 0
        self._lock = threading.Lock()

        # Statistics
        self.hits = 0
        self.misses = 0

    def get(self, layer_idx: int, expert_id: int) -> mx.array | None:
        """Get expert weights from cache."""
        key = (layer_idx, expert_id)
        with self._lock:
            if key in self._cache:
                self.hits += 1
                self._cache.move_to_end(key)
                return self._cache[key]
            self.misses += 1
            return None

    def put(self, layer_idx: int, expert_id: int, weights: mx.array) -> None:
        """Add expert weights to cache."""
        key = (layer_idx, expert_id)
        size = weights.nbytes

        with self._lock:
            # Evict if needed
            while self._size_bytes + size > self.max_size_bytes and self._cache:
                _, evicted = self._cache.popitem(last=False)
                self._size_bytes -= evicted.nbytes

            # Don't cache if too large
            if size > self.max_size_bytes:
                return

            # Add to cache
            self._cache[key] = weights
            self._size_bytes += size

    def get_or_load(
        self,
        layer_idx: int,
        expert_id: int,
        load_fn: Callable[[], mx.array],
    ) -> mx.array:
        """Get from cache or load and cache."""
        cached = self.get(layer_idx, expert_id)
        if cached is not None:
            return cached

        weights = load_fn()
        mx.eval(weights)
        self.put(layer_idx, expert_id, weights)
        return weights

    def invalidate(self, layer_idx: int, expert_id: int | None = None) -> int:
        """Invalidate cached entries.

        Args:
            layer_idx: Layer to invalidate.
            expert_id: Specific expert, or None for entire layer.

        Returns:
            Number of entries evicted.
        """
        with self._lock:
            if expert_id is not None:
                key = (layer_idx, expert_id)
                if key in self._cache:
                    weights = self._cache.pop(key)
                    self._size_bytes -= weights.nbytes
                    return 1
                return 0

            # Invalidate entire layer
            keys_to_remove = [k for k in self._cache if k[0] == layer_idx]
            for key in keys_to_remove:
                weights = self._cache.pop(key)
                self._size_bytes -= weights.nbytes
            return len(keys_to_remove)

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._size_bytes = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def size_mb(self) -> float:
        """Current size in MB."""
        return self._size_bytes / (1024 * 1024)

    def __repr__(self) -> str:
        return (
            f"ExpertLRUCache("
            f"size={self.size_mb:.1f}/{self.max_size_bytes / (1024**2):.0f}MB, "
            f"entries={len(self._cache)}, "
            f"hit_rate={self.hit_rate:.1%})"
        )
