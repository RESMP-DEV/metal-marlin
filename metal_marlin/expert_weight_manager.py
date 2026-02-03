"""On-demand expert weight loading for memory-efficient MoE inference.

For 64-expert models, keeping only 32 hot experts in GPU memory saves ~50%
GPU memory while maintaining high hit rates for typical workloads.

Architecture:
- Hot experts: Weights resident in GPU memory as Metal buffers
- Cold experts: Weights on CPU (or memory-mapped from disk)
- LRU eviction: When a cold expert is activated, evict least-recently-used

Key insight: Expert activation follows power-law distribution. With top-8
routing, typically only 16-24 experts see significant traffic. Caching
top-32 covers >95% of activations for most workloads.

Usage:
    from metal_marlin.expert_weight_manager import ExpertWeightManager

    manager = ExpertWeightManager(
        num_experts=64,
        num_layers=45,
        resident_experts=32,
    )

    # During model loading
    for layer_idx in range(num_layers):
        for expert_idx in range(num_experts):
            manager.register_expert(layer_idx, expert_idx, cpu_weights)

    # Initialize GPU buffers for hot experts
    manager.warmup_hot_experts(device)

    # During inference
    expert_buffers = manager.get_expert_buffers(layer_idx, selected_experts)
"""

from __future__ import annotations

import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ._compat import HAS_TORCH, require_torch

if HAS_TORCH:
    import torch
elif TYPE_CHECKING:
    import torch


@dataclass(slots=True)
class ExpertWeightEntry:
    """Metadata and weights for a single expert."""

    layer_idx: int
    expert_idx: int

    # CPU storage (always resident)
    cpu_weights: dict[str, torch.Tensor]

    # GPU storage (None if cold)
    gpu_buffers: dict[str, Any] | None = None

    # Usage tracking
    access_count: int = 0
    last_access: float = field(default_factory=time.monotonic)
    load_count: int = 0  # Times loaded from CPU to GPU

    @property
    def is_hot(self) -> bool:
        """Check if expert weights are in GPU memory."""
        return self.gpu_buffers is not None

    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_access = time.monotonic()


@dataclass
class LayerExpertStats:
    """Per-layer statistics for expert activation patterns."""

    layer_idx: int
    activation_counts: dict[int, int] = field(default_factory=lambda: {})
    total_activations: int = 0

    def record_activation(self, expert_idx: int) -> None:
        """Record an expert activation."""
        self.activation_counts[expert_idx] = self.activation_counts.get(expert_idx, 0) + 1
        self.total_activations += 1

    def get_hot_experts(self, top_k: int) -> list[int]:
        """Get the top-k most frequently activated experts."""
        sorted_experts = sorted(
            self.activation_counts.items(),
            key=lambda x: -x[1],
        )
        return [expert_idx for expert_idx, _ in sorted_experts[:top_k]]

    def get_activation_rate(self, expert_idx: int) -> float:
        """Get activation rate for an expert."""
        if self.total_activations == 0:
            return 0.0
        return self.activation_counts.get(expert_idx, 0) / self.total_activations


@dataclass
class PrefetchRequest:
    """Request to prefetch an expert to GPU."""

    layer_idx: int
    expert_idx: int
    priority: float = 1.0  # Higher = more urgent


class ExpertWeightManager:
    """Manages on-demand loading of expert weights with LRU caching.

    Keeps a fixed number of "hot" experts in GPU memory and loads
    cold experts on-demand when they are selected by the router.

    Thread-safe for concurrent access from multiple inference threads.

    Args:
        num_experts: Total number of experts per layer.
        num_layers: Number of MoE layers in the model.
        resident_experts: Number of experts to keep in GPU memory per layer.
        prefetch_k: Number of experts to prefetch based on routing history.
        enable_stats: Track activation statistics for adaptive caching.
        enable_prefetch: Enable background prefetch thread for overlap.
        prefetch_window: Number of previous layers to track for prediction.
    """

    def __init__(
        self,
        num_experts: int,
        num_layers: int,
        resident_experts: int = 32,
        prefetch_k: int = 4,
        enable_stats: bool = True,
        enable_prefetch: bool = True,
        prefetch_window: int = 3,
    ):
        require_torch("ExpertWeightManager")

        self.num_experts = num_experts
        self.num_layers = num_layers
        self.resident_experts = min(resident_experts, num_experts)
        self.prefetch_k = prefetch_k
        self.enable_stats = enable_stats
        self.enable_prefetch = enable_prefetch
        self.prefetch_window = prefetch_window

        # Per-layer expert storage: layer_idx -> OrderedDict[expert_idx, ExpertWeightEntry]
        # OrderedDict maintains LRU order (most recent at end)
        self._experts: dict[int, OrderedDict[int, ExpertWeightEntry]] = {
            layer_idx: OrderedDict() for layer_idx in range(num_layers)
        }

        # Per-layer GPU memory tracking
        self._hot_count: dict[int, int] = {layer_idx: 0 for layer_idx in range(num_layers)}

        # Per-layer activation statistics
        self._layer_stats: dict[int, LayerExpertStats] = {
            layer_idx: LayerExpertStats(layer_idx=layer_idx) for layer_idx in range(num_layers)
        }

        # Global statistics
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._evictions: int = 0
        self._loads: int = 0
        self._prefetch_hits: int = 0  # Experts loaded by prefetch before use
        self._prefetch_misses: int = 0  # Prefetched experts evicted before use

        # Thread safety
        self._lock = threading.RLock()

        # Metal device reference (set during warmup)
        self._device: Any = None

        # Buffer creation callback (set by model)
        self._create_gpu_buffers_fn: Callable[[dict[str, torch.Tensor], Any], dict[str, Any]] | None = None

        # Prefetch background thread
        self._prefetch_queue: queue.PriorityQueue[tuple[float, PrefetchRequest]] = queue.PriorityQueue()
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_stop: threading.Event = threading.Event()

        # Routing history for prediction (sliding window)
        self._routing_history: dict[int, list[list[int]]] = {
            layer_idx: [] for layer_idx in range(num_layers)
        }

    def set_device(self, device: Any) -> None:
        """Set the Metal device for GPU buffer creation.

        Args:
            device: MTLDevice instance.
        """
        self._device = device

    def set_buffer_creator(
        self,
        fn: Callable[[dict[str, torch.Tensor], Any], dict[str, Any]],
    ) -> None:
        """Set the callback for creating GPU buffers from CPU weights.

        Args:
            fn: Function that takes (cpu_weights_dict, device) and returns
                dict of Metal buffers.
        """
        self._create_gpu_buffers_fn = fn

    def register_expert(
        self,
        layer_idx: int,
        expert_idx: int,
        cpu_weights: dict[str, torch.Tensor],
    ) -> None:
        """Register an expert's weights (stored on CPU).

        Call this during model loading for each expert. Weights are stored
        on CPU and loaded to GPU on-demand.

        Args:
            layer_idx: Layer index.
            expert_idx: Expert index within the layer.
            cpu_weights: Dictionary of weight tensors on CPU. Expected keys:
                - gate_weights: Packed indices
                - gate_scales: Quantization scales
                - gate_su, gate_sv: Singular vectors
                - up_weights, up_scales, up_su, up_sv
                - down_weights, down_scales, down_su, down_sv
        """
        with self._lock:
            entry = ExpertWeightEntry(
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                cpu_weights=cpu_weights,
            )
            self._experts[layer_idx][expert_idx] = entry

    def warmup_hot_experts(
        self,
        device: Any | None = None,
        initial_hot: list[list[int]] | None = None,
    ) -> None:
        """Load initial hot experts to GPU memory.

        Call after all experts are registered. By default, loads the first
        `resident_experts` experts per layer. Override with `initial_hot`
        to specify which experts to preload (e.g., based on profiling data).

        Args:
            device: Optional MTLDevice. If not provided, uses previously set device.
            initial_hot: Optional list of expert indices per layer to preload.
                        Shape: [num_layers, resident_experts]
        """
        if device is not None:
            self._device = device

        if self._device is None:
            raise RuntimeError("Metal device not set. Call set_device() first.")

        if self._create_gpu_buffers_fn is None:
            raise RuntimeError(
                "Buffer creator not set. Call set_buffer_creator() first."
            )

        with self._lock:
            for layer_idx in range(self.num_layers):
                # Determine which experts to preload
                if initial_hot is not None and layer_idx < len(initial_hot):
                    hot_experts = initial_hot[layer_idx][: self.resident_experts]
                else:
                    # Default: first N experts
                    hot_experts = list(range(min(self.resident_experts, self.num_experts)))

                # Load to GPU
                for expert_idx in hot_experts:
                    if expert_idx in self._experts[layer_idx]:
                        self._load_to_gpu(layer_idx, expert_idx)

    def _load_to_gpu(self, layer_idx: int, expert_idx: int) -> None:
        """Load expert weights from CPU to GPU.

        Must be called with lock held.
        """
        entry = self._experts[layer_idx][expert_idx]
        if entry.is_hot:
            return  # Already loaded

        if self._create_gpu_buffers_fn is None:
            raise RuntimeError("Buffer creator not set. Call set_buffer_creator() first.")

        # Create GPU buffers
        entry.gpu_buffers = self._create_gpu_buffers_fn(entry.cpu_weights, self._device)
        entry.load_count += 1
        self._hot_count[layer_idx] += 1
        self._loads += 1

    def _evict_from_gpu(self, layer_idx: int, expert_idx: int) -> None:
        """Evict expert weights from GPU memory.

        Must be called with lock held.
        """
        entry = self._experts[layer_idx][expert_idx]
        if not entry.is_hot:
            return  # Already cold

        # Release GPU buffers
        entry.gpu_buffers = None
        self._hot_count[layer_idx] -= 1
        self._evictions += 1

    def _evict_lru(self, layer_idx: int) -> int | None:
        """Evict the least-recently-used hot expert from a layer.

        Must be called with lock held.

        Returns:
            Expert index that was evicted, or None if no hot experts.
        """
        layer_experts = self._experts[layer_idx]

        # Find LRU hot expert (first in OrderedDict that is hot)
        for expert_idx, entry in layer_experts.items():
            if entry.is_hot:
                self._evict_from_gpu(layer_idx, expert_idx)
                return expert_idx

        return None

    def get_expert_buffers(
        self,
        layer_idx: int,
        expert_indices: torch.Tensor | list[int],
    ) -> list[dict[str, Any]]:
        """Get GPU buffers for the specified experts, loading on-demand.

        This is the main inference-time API. For each requested expert:
        - If hot: return cached GPU buffers
        - If cold: load to GPU, possibly evicting LRU expert first

        Args:
            layer_idx: Layer index.
            expert_indices: Tensor or list of expert indices to retrieve.

        Returns:
            List of GPU buffer dictionaries, one per requested expert.
        """
        expert_list: list[int]
        if isinstance(expert_indices, torch.Tensor):
            # Tensor.tolist() returns list[Unknown] per type checker
            expert_list = [int(x) for x in expert_indices.flatten().tolist()]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        else:
            expert_list = expert_indices

        # Deduplicate while preserving order
        unique_experts: list[int] = list(dict.fromkeys(expert_list))

        results: list[dict[str, Any]] = []

        with self._lock:
            layer_experts = self._experts[layer_idx]

            for expert_idx in unique_experts:
                entry = layer_experts.get(expert_idx)
                if entry is None:
                    raise ValueError(
                        f"Expert {expert_idx} not registered for layer {layer_idx}"
                    )

                if entry.is_hot:
                    # Cache hit
                    self._cache_hits += 1
                    entry.touch()
                    # Move to end for LRU
                    layer_experts.move_to_end(expert_idx)
                else:
                    # Cache miss - need to load
                    self._cache_misses += 1

                    # Evict if at capacity
                    if self._hot_count[layer_idx] >= self.resident_experts:
                        self._evict_lru(layer_idx)

                    # Load to GPU
                    self._load_to_gpu(layer_idx, expert_idx)
                    entry.touch()
                    layer_experts.move_to_end(expert_idx)

                # Record activation for stats
                if self.enable_stats:
                    self._layer_stats[layer_idx].record_activation(expert_idx)

                if entry.gpu_buffers is not None:
                    results.append(entry.gpu_buffers)

            # Record routing for prefetch prediction
            if self.enable_prefetch:
                self.record_routing(layer_idx, unique_experts)

                # Trigger prefetch for next layer (overlapped with current computation)
                if layer_idx + 1 < self.num_layers:
                    self.prefetch_next_layer(layer_idx)

        return results

    def get_expert_buffers_batch(
        self,
        layer_idx: int,
        expert_ids: torch.Tensor,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Get stacked GPU buffers for a batch of expert selections.

        Optimized for batch inference where multiple tokens select different
        experts. Returns a single dict of stacked buffers covering all unique
        experts, plus a remapping tensor.

        Args:
            layer_idx: Layer index.
            expert_ids: Expert selection tensor [batch, top_k].

        Returns:
            Tuple of:
            - Dict of stacked GPU buffers for unique experts
            - Remap tensor to index into stacked buffers
        """
        # Get unique experts in selection order
        # Tensor.tolist() returns list[Unknown] per type checker
        unique_list: list[int] = [int(x) for x in expert_ids.flatten().unique().tolist()]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType, reportUnknownMemberType]

        # Get individual buffers
        buffers = self.get_expert_buffers(layer_idx, unique_list)

        # Create remap: original expert_id -> index in stacked buffers
        expert_to_idx: dict[int, int] = {eid: i for i, eid in enumerate(unique_list)}
        remap = expert_ids.clone()
        for orig_id, new_idx in expert_to_idx.items():
            remap[expert_ids == orig_id] = new_idx

        # Stack buffers
        # For now, return list; actual stacking depends on kernel requirements
        stacked = {
            key: [b[key] for b in buffers]
            for key in buffers[0].keys()
        }

        return stacked, remap

    def prefetch_experts(
        self,
        layer_idx: int,
        expert_indices: list[int],
    ) -> int:
        """Speculatively prefetch experts to GPU memory.

        Call this with predicted expert indices (e.g., from routing history)
        to reduce cache misses. Non-blocking if called from a separate thread.

        Args:
            layer_idx: Layer index.
            expert_indices: Expert indices to prefetch.

        Returns:
            Number of experts actually loaded (excludes already-hot experts).
        """
        loaded = 0

        with self._lock:
            for expert_idx in expert_indices:
                if expert_idx not in self._experts[layer_idx]:
                    continue

                entry = self._experts[layer_idx][expert_idx]
                if entry.is_hot:
                    continue

                # Evict if at capacity
                if self._hot_count[layer_idx] >= self.resident_experts:
                    self._evict_lru(layer_idx)

                self._load_to_gpu(layer_idx, expert_idx)
                loaded += 1

        return loaded

    def start_prefetch_thread(self) -> None:
        """Start background thread for async expert prefetching.

        The prefetch thread continuously processes prefetch requests from
        a priority queue, loading experts to GPU while computation happens
        on already-loaded experts.

        Call this after warmup_hot_experts() to enable overlapped loading.
        """
        if not self.enable_prefetch:
            return

        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            return  # Already running

        self._prefetch_stop.clear()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            name="expert-prefetch",
            daemon=True,
        )
        self._prefetch_thread.start()

    def stop_prefetch_thread(self) -> None:
        """Stop the background prefetch thread gracefully."""
        if self._prefetch_thread is None:
            return

        self._prefetch_stop.set()
        # Wake up thread if it's waiting
        try:
            self._prefetch_queue.put((-999.0, PrefetchRequest(-1, -1)), timeout=0.1)
        except queue.Full:
            pass
        self._prefetch_thread.join(timeout=2.0)
        self._prefetch_thread = None

    def _prefetch_worker(self) -> None:
        """Background worker that processes prefetch requests.

        Runs in a separate thread, continuously loading experts from CPU
        to GPU based on predicted need. Overlaps I/O with computation.
        """
        while not self._prefetch_stop.is_set():
            try:
                # Wait for prefetch request (with timeout to check stop flag)
                priority, request = self._prefetch_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Check if this is a sentinel value to wake up for shutdown
            if request.layer_idx < 0:
                continue

            # Try to load the expert (non-blocking if already hot)
            with self._lock:
                if request.layer_idx >= self.num_layers:
                    continue
                if request.expert_idx not in self._experts[request.layer_idx]:
                    continue

                entry = self._experts[request.layer_idx][request.expert_idx]
                if entry.is_hot:
                    # Already loaded, mark as prefetch hit
                    self._prefetch_hits += 1
                    continue

                # Evict if at capacity
                if self._hot_count[request.layer_idx] >= self.resident_experts:
                    self._evict_lru(request.layer_idx)

                # Load to GPU (this is the I/O overlap opportunity)
                self._load_to_gpu(request.layer_idx, request.expert_idx)

    def async_prefetch_experts(
        self,
        layer_idx: int,
        expert_indices: list[int],
        priority: float = 0.5,
    ) -> None:
        """Queue experts for asynchronous prefetch by background thread.

        Non-blocking. Experts will be loaded in background, overlapping with
        current computation. Use this to preload experts for upcoming layers.

        Args:
            layer_idx: Layer index.
            expert_indices: Expert indices to prefetch.
            priority: Prefetch priority (0.0 = low, 1.0 = high). Higher
                     priority requests processed first.
        """
        if not self.enable_prefetch:
            return

        if self._prefetch_thread is None or not self._prefetch_thread.is_alive():
            # Fallback to synchronous prefetch if thread not running
            self.prefetch_experts(layer_idx, expert_indices)
            return

        # Add to queue (negative priority for min-heap behavior)
        for expert_idx in expert_indices:
            request = PrefetchRequest(
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                priority=priority,
            )
            try:
                self._prefetch_queue.put((-priority, request), block=False)
            except queue.Full:
                # Queue full, skip this request
                pass

    def record_routing(self, layer_idx: int, expert_indices: torch.Tensor | list[int]) -> None:
        """Record routing decision for prefetch prediction.

        Call this after router computation to track which experts were
        selected. Used to predict experts for next token/layer.

        Args:
            layer_idx: Layer index.
            expert_indices: Selected expert indices (can be [batch, top_k] tensor).
        """
        if not self.enable_prefetch:
            return

        # Convert to flat list
        if isinstance(expert_indices, torch.Tensor):
            expert_list: list[int] = [int(x) for x in expert_indices.flatten().unique().tolist()]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType, reportUnknownMemberType]
        else:
            expert_list = list(set(expert_indices))

        with self._lock:
            history = self._routing_history[layer_idx]
            history.append(expert_list)

            # Keep sliding window
            if len(history) > self.prefetch_window:
                history.pop(0)

    def predict_next_experts(self, layer_idx: int, k: int | None = None) -> list[int]:
        """Predict which experts will be needed for next token/batch.

        Uses routing history with recency weighting:
        - Most recent routing: weight 1.0
        - Previous routing: weight 0.5
        - Earlier routing: weight 0.25

        Args:
            layer_idx: Layer index.
            k: Number of experts to predict (default: self.prefetch_k).

        Returns:
            List of predicted expert indices, sorted by likelihood.
        """
        if k is None:
            k = self.prefetch_k

        with self._lock:
            history = self._routing_history[layer_idx]
            if not history:
                # No history, use activation stats
                return self.get_prefetch_candidates(layer_idx)[:k]

            # Weight recent routing decisions more heavily
            expert_scores: dict[int, float] = {}
            decay = 0.5  # Exponential decay factor

            for i, expert_list in enumerate(reversed(history)):
                weight = decay ** i  # Most recent = 1.0, previous = 0.5, etc.
                for expert_idx in expert_list:
                    expert_scores[expert_idx] = expert_scores.get(expert_idx, 0.0) + weight

            # Sort by score (descending)
            sorted_experts = sorted(
                expert_scores.items(),
                key=lambda x: -x[1],
            )

            # Return top-k, excluding already-hot experts
            candidates: list[int] = []
            for expert_idx, _ in sorted_experts:
                if expert_idx in self._experts[layer_idx]:
                    entry = self._experts[layer_idx][expert_idx]
                    if not entry.is_hot:
                        candidates.append(expert_idx)
                        if len(candidates) >= k:
                            break

            return candidates

    def prefetch_next_layer(self, current_layer_idx: int) -> None:
        """Predict and prefetch experts for the next layer.

        Call this during computation of current_layer_idx to overlap
        loading for the next layer. Combines history-based prediction
        with same-layer repeat heuristic.

        Args:
            current_layer_idx: Current layer being computed.
        """
        if not self.enable_prefetch:
            return

        next_layer = current_layer_idx + 1
        if next_layer >= self.num_layers:
            return

        # Strategy: Use current layer's routing as initial guess for next layer
        # (consecutive layers often have similar routing patterns)
        with self._lock:
            current_history = self._routing_history[current_layer_idx]
            if current_history:
                # Use most recent routing from current layer
                predicted = current_history[-1][:self.prefetch_k]
            else:
                # Fallback to next layer's own history
                predicted = self.predict_next_experts(next_layer)

        # Queue for async loading
        self.async_prefetch_experts(next_layer, predicted, priority=0.8)

    def get_prefetch_candidates(self, layer_idx: int) -> list[int]:
        """Get expert indices predicted to be needed, based on activation history.

        Args:
            layer_idx: Layer index.

        Returns:
            List of expert indices to prefetch, sorted by likelihood.
        """
        if not self.enable_stats:
            return []

        with self._lock:
            stats = self._layer_stats[layer_idx]

            # Get hot experts that aren't already loaded
            hot_experts = stats.get_hot_experts(self.prefetch_k + self.resident_experts)

            candidates: list[int] = []
            for expert_idx in hot_experts:
                if expert_idx in self._experts[layer_idx]:
                    entry = self._experts[layer_idx][expert_idx]
                    if not entry.is_hot:
                        candidates.append(expert_idx)
                        if len(candidates) >= self.prefetch_k:
                            break

            return candidates

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics.

        Returns:
            Dictionary with cache performance metrics.
        """
        with self._lock:
            total_requests = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

            per_layer_stats = {}
            for layer_idx in range(self.num_layers):
                layer_experts = self._experts[layer_idx]
                hot_count = self._hot_count[layer_idx]
                cold_count = len(layer_experts) - hot_count

                layer_stats = self._layer_stats[layer_idx]
                top_experts = layer_stats.get_hot_experts(10)

                per_layer_stats[layer_idx] = {
                    "hot_experts": hot_count,
                    "cold_experts": cold_count,
                    "total_activations": layer_stats.total_activations,
                    "top_10_experts": top_experts,
                }

            return {
                "global": {
                    "cache_hits": self._cache_hits,
                    "cache_misses": self._cache_misses,
                    "hit_rate": hit_rate,
                    "evictions": self._evictions,
                    "loads": self._loads,
                },
                "config": {
                    "num_experts": self.num_experts,
                    "num_layers": self.num_layers,
                    "resident_experts": self.resident_experts,
                    "prefetch_k": self.prefetch_k,
                },
                "per_layer": per_layer_stats,
            }

    def clear_layer(self, layer_idx: int) -> None:
        """Clear all GPU buffers for a layer.

        Args:
            layer_idx: Layer index to clear.
        """
        with self._lock:
            for expert_idx, entry in self._experts[layer_idx].items():
                if entry.is_hot:
                    self._evict_from_gpu(layer_idx, expert_idx)

    def clear_all(self) -> None:
        """Clear all GPU buffers across all layers."""
        with self._lock:
            for layer_idx in range(self.num_layers):
                self.clear_layer(layer_idx)

    def get_memory_usage(self) -> dict[str, Any]:
        """Estimate memory usage of cached expert weights.

        Returns:
            Dictionary with memory usage estimates.
        """
        with self._lock:
            total_hot = 0
            total_cold = 0
            hot_bytes = 0
            cold_bytes = 0

            for layer_idx in range(self.num_layers):
                for entry in self._experts[layer_idx].values():
                    # Estimate size from CPU weights
                    entry_bytes = sum(
                        t.numel() * t.element_size() for t in entry.cpu_weights.values()
                    )

                    if entry.is_hot:
                        total_hot += 1
                        hot_bytes += entry_bytes
                    else:
                        total_cold += 1
                        cold_bytes += entry_bytes

            return {
                "hot_experts": total_hot,
                "cold_experts": total_cold,
                "hot_memory_mb": hot_bytes / (1024 * 1024),
                "cold_memory_mb": cold_bytes / (1024 * 1024),
                "total_memory_mb": (hot_bytes + cold_bytes) / (1024 * 1024),
                "savings_percent": (cold_bytes / (hot_bytes + cold_bytes) * 100)
                if (hot_bytes + cold_bytes) > 0
                else 0,
            }

    def __repr__(self) -> str:
        total_hot = sum(self._hot_count.values())
        total_experts = self.num_experts * self.num_layers
        return (
            f"ExpertWeightManager("
            f"experts={total_experts}, "
            f"hot={total_hot}/{self.resident_experts * self.num_layers}, "
            f"hit_rate={self._cache_hits / max(1, self._cache_hits + self._cache_misses):.1%})"
        )


def create_expert_weight_manager_from_config(
    config: dict[str, Any],
    resident_fraction: float = 0.5,
) -> ExpertWeightManager:
    """Create ExpertWeightManager from model configuration.

    Args:
        config: Model configuration dictionary.
        resident_fraction: Fraction of experts to keep resident (default: 0.5).

    Returns:
        Configured ExpertWeightManager instance.
    """
    num_experts = (
        config.get("num_experts")
        or config.get("num_local_experts")
        or config.get("n_experts")
        or 64
    )

    num_layers = (
        config.get("num_hidden_layers")
        or config.get("n_layer")
        or config.get("num_layers")
        or 32
    )

    # Adjust for first_moe_layer (some dense layers at the start)
    first_moe_layer = config.get("first_moe_layer", 0)
    moe_layers = num_layers - first_moe_layer

    resident_experts = max(1, int(num_experts * resident_fraction))

    return ExpertWeightManager(
        num_experts=num_experts,
        num_layers=moe_layers,
        resident_experts=resident_experts,
    )
