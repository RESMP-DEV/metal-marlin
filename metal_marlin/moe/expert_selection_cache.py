"""Expert-selection cache for fast Top-K MoE routing reuse.

This module targets decode-time MoE routing workloads such as GLM-4.7-Flash
(64 experts, Top-2). It implements:

1. Routing decision cache:
   - Hash lookup: hidden_state_hash -> (expert_ids, expert_probs)
   - LRU eviction for recent-token locality
   - Optional locality fallback for "similar hidden state" reuse

2. Fast path:
   - Cache hit skips router forward pass
   - Cache hit updates LRU state and access stats

3. Speculative expert prefetch:
   - Predict likely experts from short-term context/transition history
   - Asynchronously preload expert weights to target device memory
"""

from __future__ import annotations

import hashlib
import struct
import threading
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._compat import HAS_MPS, HAS_TORCH, torch

if TYPE_CHECKING:
    import torch as torch_typing  # noqa: F401 - used for TensorType alias

TensorType = "torch_typing.Tensor" if TYPE_CHECKING else Any
HitType = Literal["exact", "locality", "miss"]


def create_glm47_flash_config() -> ExpertSelectionCacheConfig:
    """Create optimized configuration for GLM-4.7-Flash (64 experts, Top-2).
    
    GLM-4.7-Flash has specific characteristics that this configuration optimizes for:
    - 64 experts with top-2 routing: only 3.125% of experts active per token
    - High expert locality in decode: consecutive tokens often use same experts
    - Large hidden dimension: requires efficient sketching to reduce memory overhead
    
    The optimized settings:
    - Larger routing cache (4096 entries): captures more decode patterns
    - Larger prefetch cache (256): stores more expert weights to hide latency
    - Higher similarity threshold (0.99): ensures quality for exact matches
    - Context window of 256: captures longer-range expert transitions
    
    Returns:
        Configuration optimized for GLM-4.7-Flash decode workloads.
    
    Example:
        >>> config = create_glm47_flash_config()
        >>> cache = ExpertSelectionCache(
        ...     num_experts=64,
        ...     top_k=2,
        ...     config=config,
        ...     device="mps",
        ... )
    """
    return ExpertSelectionCacheConfig(
        max_routing_entries=4096,
        max_prefetched_weights=256,
        sketch_dim=64,  # Higher for large hidden dimensions
        locality_search_window=512,
        locality_similarity_threshold=0.99,
        prefetch_k=6,  # Prefetch more experts for 64-expert model
        context_window=256,
        prefetch_threads=2,
        enable_locality_fallback=True,
        enable_speculative_prefetch=True,
        min_probability_floor=1e-5,
    )


def create_expert_selection_cache_for_glm47_flash(
    load_expert_weight: Callable[[int, int], TensorType] | None = None,
    device: str | None = None,
    num_experts: int = 64,
    top_k: int = 2,
) -> ExpertSelectionCache:
    """Factory function to create expert selection cache optimized for GLM-4.7-Flash.
    
    This is a convenience function that creates an ExpertSelectionCache with
    GLM-4.7-Flash specific optimizations pre-configured.
    
    Args:
        load_expert_weight: Optional callback to load expert weights.
            Signature: load_expert_weight(layer_idx: int, expert_id: int) -> Tensor
            If provided, enables speculative prefetching.
        device: Target device for prefetched weights (default: auto-detect).
            Typically "mps" for Apple Silicon, "cuda" for NVIDIA GPUs, or "cpu".
        num_experts: Number of experts in MoE layer (default: 64 for GLM-4.7-Flash).
        top_k: Number of experts selected per token (default: 2 for GLM-4.7-Flash).
    
    Returns:
        Configured ExpertSelectionCache instance.
    
    Example:
        >>> # Basic usage for decode caching
        >>> cache = create_expert_selection_cache_for_glm47_flash()
        >>> 
        >>> # With expert weight loading for prefetch
        >>> def load_expert(layer_idx, expert_id):
        ...     return model.layers[layer_idx].mlp.experts[expert_id]
        >>> cache = create_expert_selection_cache_for_glm47_flash(
        ...     load_expert_weight=load_expert,
        ...     device="mps",
        ... )
        >>> 
        >>> # Use in forward pass
        >>> def route_with_cache(hidden_state, layer_idx):
        ...     result = cache.get_or_route(
        ...         hidden_state=hidden_state,
        ...         router_fn=lambda x: router(x),  # Your router function
        ...         layer_idx=layer_idx,
        ...     )
        ...     return result.expert_ids, result.expert_probs
    """
    config = create_glm47_flash_config()
    return ExpertSelectionCache(
        num_experts=num_experts,
        top_k=top_k,
        load_expert_weight=load_expert_weight,
        config=config,
        device=device,
    )


@dataclass
class ExpertSelectionCacheConfig:
    """Configuration for expert-selection caching."""

    max_routing_entries: int = 2048
    max_prefetched_weights: int = 128
    sketch_dim: int = 32
    locality_search_window: int = 256
    locality_similarity_threshold: float = 0.985
    prefetch_k: int = 4
    context_window: int = 128
    prefetch_threads: int = 2
    enable_locality_fallback: bool = True
    enable_speculative_prefetch: bool = True
    min_probability_floor: float = 1e-5


@dataclass
class ExpertSelectionCacheStats:
    """Runtime statistics for cache effectiveness and prefetch behavior."""

    exact_hits: int = 0
    locality_hits: int = 0
    misses: int = 0
    router_forwards: int = 0
    lru_evictions: int = 0
    locality_promotions: int = 0
    prefetch_submitted: int = 0
    prefetch_completed: int = 0
    prefetch_errors: int = 0
    weight_cache_hits: int = 0
    weight_cache_misses: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Overall routing cache hit rate (exact + locality)."""
        total = self.exact_hits + self.locality_hits + self.misses
        if total == 0:
            return 0.0
        return (self.exact_hits + self.locality_hits) / total

    def as_dict(self) -> dict[str, float | int]:
        """Convert stats to a serializable dictionary."""
        return {
            "exact_hits": self.exact_hits,
            "locality_hits": self.locality_hits,
            "misses": self.misses,
            "router_forwards": self.router_forwards,
            "lru_evictions": self.lru_evictions,
            "locality_promotions": self.locality_promotions,
            "prefetch_submitted": self.prefetch_submitted,
            "prefetch_completed": self.prefetch_completed,
            "prefetch_errors": self.prefetch_errors,
            "weight_cache_hits": self.weight_cache_hits,
            "weight_cache_misses": self.weight_cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
        }


@dataclass
class CacheLookupResult:
    """Result from cache lookup or router fallback."""

    hidden_state_hash: int
    expert_ids: tuple[int, ...]
    expert_probs: tuple[float, ...]
    cache_hit: bool
    hit_type: HitType


@dataclass
class _RoutingCacheEntry:
    """Stored routing decision entry."""

    hidden_state_hash: int
    sketch: np.ndarray
    expert_ids: tuple[int, ...]
    expert_probs: tuple[float, ...]
    access_count: int = 0
    last_access: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        """Refresh recency/access counters for LRU bookkeeping."""
        self.access_count += 1
        self.last_access = time.monotonic()


class ExpertSelectionCache:
    """Caches MoE router decisions and speculatively prefetches likely experts.

    The primary fast path is:
    1. Compute hidden_state_hash from a compact hidden-state sketch.
    2. Lookup hash in routing cache.
    3. On hit, reuse `(expert_ids, expert_probs)` and skip router forward.
    4. On miss, run router, then populate cache and update history.
    """

    def __init__(
        self,
        num_experts: int = 64,
        top_k: int = 2,
        load_expert_weight: Callable[[int, int], TensorType] | None = None,
        config: ExpertSelectionCacheConfig | None = None,
        device: str | None = None,
    ):
        self.num_experts = max(1, int(num_experts))
        self.top_k = max(1, int(top_k))
        self.config = config or ExpertSelectionCacheConfig()
        self.device = device or self._infer_device()
        self.load_expert_weight = load_expert_weight

        self._routing_cache: OrderedDict[int, _RoutingCacheEntry] = OrderedDict()
        self._prefetched_weights: OrderedDict[tuple[int, int], TensorType] = OrderedDict()
        self._inflight_prefetch: dict[tuple[int, int], Future[None]] = {}
        self._sketch_index_cache: dict[tuple[int, int, str, int], TensorType] = {}

        self._recent_pairs: deque[tuple[int, ...]] = deque(maxlen=max(1, self.config.context_window))
        self._transition_counts: dict[tuple[int, ...], Counter[int]] = {}
        self._expert_popularity: Counter[int] = Counter()

        self._stats = ExpertSelectionCacheStats()
        self._lock = threading.RLock()

        self._executor: ThreadPoolExecutor | None = None
        self._prefetch_enabled = (
            self.config.enable_speculative_prefetch and self.load_expert_weight is not None
        )
        if self._prefetch_enabled:
            self._executor = ThreadPoolExecutor(
                max_workers=max(1, int(self.config.prefetch_threads)),
                thread_name_prefix="expert_selection_prefetch",
            )

    def _infer_device(self) -> str:
        """Infer default destination for prefetched weights."""
        if HAS_TORCH and torch is not None:
            if HAS_MPS:
                try:
                    if torch.backends.mps.is_available():
                        return "mps"
                except AttributeError:
                    pass
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                return "cuda"
        return "cpu"

    def _to_numpy_vector(self, hidden_state: TensorType | Sequence[float]) -> np.ndarray:
        """Convert hidden state to a flat float32 numpy vector."""
        np_vec = np.asarray(hidden_state, dtype=np.float32).reshape(-1)

        if np_vec.size == 0:
            raise ValueError("hidden_state must contain at least one element")

        return np_vec

    def _build_torch_sketch(self, hidden_state: TensorType, sketch_dim: int) -> np.ndarray:
        """Build sketch directly from a torch tensor, moving only sketch-sized data to CPU."""
        vector = hidden_state.detach().reshape(-1)
        if vector.numel() == 0:
            raise ValueError("hidden_state must contain at least one element")

        dtype = vector.dtype
        if str(dtype) == "torch.bfloat16":
            vector = vector.float()

        if sketch_dim >= vector.numel():
            return self._to_numpy_vector(vector)

        indices = torch.linspace(0, vector.numel() - 1, sketch_dim, device=vector.device).long()
        sampled = vector.index_select(0, indices)
        return sampled.cpu().numpy()

    def _build_sketch(self, hidden_state: TensorType | Sequence[float], sketch_dim: int) -> np.ndarray:
        """Build a compact sketch of the hidden state for hashing."""
        np_vec = self._to_numpy_vector(hidden_state)
        vec_len = np_vec.size

        if sketch_dim >= vec_len:
            return np_vec

        step = vec_len // sketch_dim
        sketch = np.zeros(sketch_dim, dtype=np.float32)

        for i in range(sketch_dim):
            start_idx = i * step
            end_idx = min((i + 1) * step, vec_len)
            if start_idx < end_idx:
                sketch[i] = np.mean(np_vec[start_idx:end_idx])

        return sketch

    def hidden_state_hash(self, hidden_state: TensorType | Sequence[float]) -> int:
        """Compute hash of hidden state using sketch for fast cache lookup.

        Args:
            hidden_state: Input hidden state tensor or array.

        Returns:
            64-bit integer hash suitable for dictionary keys.
        """
        if HAS_TORCH and torch is not None and isinstance(hidden_state, torch.Tensor):
            sketch = self._build_torch_sketch(hidden_state, self.config.sketch_dim)
        else:
            sketch = self._build_sketch(hidden_state, self.config.sketch_dim)

        hash_input = sketch.tobytes() + struct.pack("<II", self.num_experts, self.top_k)
        hash_bytes = hashlib.sha256(hash_input).digest()[:8]
        return int.from_bytes(hash_bytes, byteorder="little", signed=False)

    def _sketch_similarity(self, sketch1: np.ndarray, sketch2: np.ndarray) -> float:
        """Compute cosine similarity between two sketches."""
        norm1 = np.linalg.norm(sketch1) + 1e-8
        norm2 = np.linalg.norm(sketch2) + 1e-8
        return float(np.dot(sketch1, sketch2) / (norm1 * norm2))

    def _find_locality_match(
        self, target_sketch: np.ndarray, exclude_hash: int
    ) -> tuple[int, _RoutingCacheEntry] | None:
        """Find cache entry with similar hidden state (locality fallback)."""
        if not self.config.enable_locality_fallback:
            return None

        best_hash = -1
        best_entry = None
        best_similarity = self.config.locality_similarity_threshold

        window = 0
        for cached_hash, entry in reversed(self._routing_cache.items()):
            if cached_hash == exclude_hash:
                continue

            similarity = self._sketch_similarity(target_sketch, entry.sketch)
            if similarity > best_similarity:
                best_similarity = similarity
                best_hash = cached_hash
                best_entry = entry

            window += 1
            if window >= self.config.locality_search_window:
                break

        if best_entry is not None:
            self._stats.locality_hits += 1
            self._stats.locality_promotions += 1

            promoted_entry = _RoutingCacheEntry(
                hidden_state_hash=best_hash,
                sketch=target_sketch,
                expert_ids=best_entry.expert_ids,
                expert_probs=best_entry.expert_probs,
                access_count=best_entry.access_count + 1,
            )
            return best_hash, promoted_entry

        return None

    def cache_routing(
        self,
        hidden_state: TensorType | Sequence[float],
        expert_ids: Sequence[int] | np.ndarray,
        expert_probs: Sequence[float] | np.ndarray,
    ) -> None:
        """Manually add a routing decision to the cache.

        Args:
            hidden_state: Input hidden state.
            expert_ids: Selected expert indices.
            expert_probs: Expert routing probabilities.
        """
        hash_key = self.hidden_state_hash(hidden_state)

        if HAS_TORCH and torch is not None and isinstance(hidden_state, torch.Tensor):
            sketch = self._build_torch_sketch(hidden_state, self.config.sketch_dim)
        else:
            sketch = self._build_sketch(hidden_state, self.config.sketch_dim)

        expert_ids_tuple = tuple(int(e) for e in expert_ids[: self.top_k])
        expert_probs_tuple = tuple(float(p) for p in expert_probs[: self.top_k])

        entry = _RoutingCacheEntry(
            hidden_state_hash=hash_key,
            sketch=sketch,
            expert_ids=expert_ids_tuple,
            expert_probs=expert_probs_tuple,
        )

        with self._lock:
            self._add_routing_entry_locked(hash_key, entry)
            self._update_history_locked(expert_ids_tuple)

    def get_or_route(
        self,
        hidden_state: TensorType | Sequence[float],
        router_fn: Callable[[TensorType | Sequence[float]], tuple[Sequence[int], Sequence[float]]],
        layer_idx: int = 0,
        allow_locality: bool = True,
    ) -> CacheLookupResult:
        """Get routing from cache or run router.

        Args:
            hidden_state: Input hidden state.
            router_fn: Router function returning (expert_ids, expert_probs).
            layer_idx: Layer index for prefetch tracking.
            allow_locality: Whether to use locality fallback on miss.

        Returns:
            CacheLookupResult with routing decision and hit status.
        """
        hash_key = self.hidden_state_hash(hidden_state)

        if HAS_TORCH and torch is not None and isinstance(hidden_state, torch.Tensor):
            sketch = self._build_torch_sketch(hidden_state, self.config.sketch_dim)
        else:
            sketch = self._build_sketch(hidden_state, self.config.sketch_dim)

        with self._lock:
            cached = self._routing_cache.get(hash_key)
            if cached is not None:
                cached.touch()
                self._routing_cache.move_to_end(hash_key)
                self._stats.exact_hits += 1

                return CacheLookupResult(
                    hidden_state_hash=hash_key,
                    expert_ids=cached.expert_ids,
                    expert_probs=cached.expert_probs,
                    cache_hit=True,
                    hit_type="exact",
                )

            if allow_locality:
                locality_match = self._find_locality_match(sketch, hash_key)
                if locality_match is not None:
                    local_hash, promoted_entry = locality_match

                    # Update LRU for the source entry
                    self._add_routing_entry_locked(local_hash, promoted_entry)

                    # Create a NEW entry for the current hash to enable exact hits later
                    new_entry = _RoutingCacheEntry(
                        hidden_state_hash=hash_key,
                        sketch=sketch,
                        expert_ids=promoted_entry.expert_ids,
                        expert_probs=promoted_entry.expert_probs,
                        access_count=1,
                    )
                    self._add_routing_entry_locked(hash_key, new_entry)
                    self._update_history_locked(promoted_entry.expert_ids)

                    return CacheLookupResult(
                        hidden_state_hash=hash_key,
                        expert_ids=promoted_entry.expert_ids,
                        expert_probs=promoted_entry.expert_probs,
                        cache_hit=True,
                        hit_type="locality",
                    )

        self._stats.misses += 1

        expert_ids_seq, expert_probs_seq = router_fn(hidden_state)
        expert_ids = tuple(int(e) for e in expert_ids_seq[: self.top_k])
        expert_probs = tuple(float(p) for p in expert_probs_seq[: self.top_k])

        entry = _RoutingCacheEntry(
            hidden_state_hash=hash_key,
            sketch=sketch,
            expert_ids=expert_ids,
            expert_probs=expert_probs,
        )

        with self._lock:
            self._add_routing_entry_locked(hash_key, entry)
            self._update_history_locked(expert_ids)

        self._stats.router_forwards += 1

        return CacheLookupResult(
            hidden_state_hash=hash_key,
            expert_ids=expert_ids,
            expert_probs=expert_probs,
            cache_hit=False,
            hit_type="miss",
        )

    def _add_routing_entry_locked(self, hash_key: int, entry: _RoutingCacheEntry) -> None:
        """Add routing entry with LRU eviction."""
        self._routing_cache[hash_key] = entry
        self._routing_cache.move_to_end(hash_key)

        while len(self._routing_cache) > self.config.max_routing_entries:
            evicted_key, _ = self._routing_cache.popitem(last=False)
            self._stats.lru_evictions += 1

    def _update_history_locked(self, expert_ids: tuple[int, ...]) -> None:
        """Update transition history and expert popularity."""
        self._recent_pairs.append(expert_ids)
        for expert_id in expert_ids:
            self._expert_popularity[expert_id] += 1

        if len(self._recent_pairs) >= 2:
            prev_pair = self._recent_pairs[-2]
            curr_pair = expert_ids

            for prev_expert in prev_pair:
                transition_key = (prev_expert,)
                if transition_key not in self._transition_counts:
                    self._transition_counts[transition_key] = Counter()

                for curr_expert in curr_pair:
                    self._transition_counts[transition_key][curr_expert] += 1

    def predict_likely_experts(
        self,
        context_expert_ids: Sequence[int] | None = None,
        k: int | None = None,
    ) -> list[int]:
        """Predict likely next experts from context.

        Args:
            context_expert_ids: Recent expert IDs to base prediction on.
            k: Number of experts to predict (default: config.prefetch_k).

        Returns:
            List of predicted expert IDs sorted by likelihood.
        """
        if k is None:
            k = self.config.prefetch_k

        predictions: Counter[int] = Counter()

        if context_expert_ids is None and len(self._recent_pairs) > 0:
            context_expert_ids = self._recent_pairs[-1]

        if context_expert_ids is None:
            context_expert_ids = []

        for expert_id in context_expert_ids:
            transition_key = (expert_id,)
            if transition_key in self._transition_counts:
                transitions = self._transition_counts[transition_key]
                for next_expert, count in transitions.most_common(k * 2):
                    predictions[next_expert] += count

        if not predictions:
            popular = [eid for eid, _ in self._expert_popularity.most_common(k)]
            return popular

        return [eid for eid, _ in predictions.most_common(k)]

    def _add_prefetched_weight_locked(
        self, key: tuple[int, int], weight: TensorType
    ) -> None:
        """Add prefetched weight with LRU eviction."""
        self._prefetched_weights[key] = weight
        self._prefetched_weights.move_to_end(key)

        while len(self._prefetched_weights) > self.config.max_prefetched_weights:
            _, evicted = self._prefetched_weights.popitem(last=False)

    def _place_weight_on_device(self, weight: TensorType) -> TensorType:
        """Move weight to target device."""
        if HAS_TORCH and torch is not None and isinstance(weight, torch.Tensor):
            target_device = self.device
            if target_device != "cpu" and weight.device.type != target_device:
                return weight.to(self.device)

        return weight

    def _prefetch_one(self, layer_idx: int, expert_id: int) -> None:
        """Background prefetch worker."""
        key = (layer_idx, expert_id)
        try:
            if self.load_expert_weight is None:
                return

            weight = self.load_expert_weight(layer_idx, expert_id)
            weight = self._place_weight_on_device(weight)

            with self._lock:
                self._add_prefetched_weight_locked(key, weight)
                self._stats.prefetch_completed += 1
        except Exception:
            with self._lock:
                self._stats.prefetch_errors += 1
        finally:
            with self._lock:
                self._inflight_prefetch.pop(key, None)

    def _submit_prefetch(self, layer_idx: int, expert_id: int) -> None:
        """Submit a speculative prefetch task if not already cached/in-flight."""
        key = (layer_idx, expert_id)

        with self._lock:
            if key in self._prefetched_weights:
                self._prefetched_weights.move_to_end(key)
                return
            if key in self._inflight_prefetch:
                return

            if self._executor is None:
                return

            future = self._executor.submit(self._prefetch_one, layer_idx, expert_id)
            self._inflight_prefetch[key] = future
            self._stats.prefetch_submitted += 1

    def prefetch_likely_experts(
        self,
        layer_idx: int,
        context_expert_ids: Sequence[int] | None = None,
        k: int | None = None,
    ) -> list[int]:
        """Speculatively prefetch experts predicted from context."""
        predicted = self.predict_likely_experts(context_expert_ids=context_expert_ids, k=k)
        if not predicted:
            return []

        if not self._prefetch_enabled:
            return predicted

        for expert_id in predicted:
            self._submit_prefetch(layer_idx=layer_idx, expert_id=expert_id)

        return predicted

    def get_prefetched_weight(self, layer_idx: int, expert_id: int) -> TensorType | None:
        """Return prefetched expert weight if available in cache."""
        key = (layer_idx, expert_id)
        with self._lock:
            weight = self._prefetched_weights.get(key)
            if weight is None:
                self._stats.weight_cache_misses += 1
                return None

            self._prefetched_weights.move_to_end(key)
            self._stats.weight_cache_hits += 1
            return weight

    def get_prefetched_weights(
        self,
        layer_idx: int,
        expert_ids: Sequence[int],
    ) -> dict[int, TensorType]:
        """Return all currently cached prefetched weights for requested experts."""
        found: dict[int, TensorType] = {}
        for expert_id in expert_ids:
            weight = self.get_prefetched_weight(layer_idx, int(expert_id))
            if weight is not None:
                found[int(expert_id)] = weight
        return found

    def store_prefetched_weight(self, layer_idx: int, expert_id: int, weight: TensorType) -> None:
        """Store externally loaded weight in the prefetch cache."""
        key = (layer_idx, int(expert_id))
        prepared = self._place_weight_on_device(weight)
        with self._lock:
            self._add_prefetched_weight_locked(key, prepared)

    def wait_for_prefetch(self, timeout: float | None = None) -> None:
        """Wait for currently in-flight prefetch jobs."""
        with self._lock:
            futures = list(self._inflight_prefetch.values())

        if not futures:
            return

        done, _ = wait(futures, timeout=timeout)
        for future in done:
            try:
                future.result()
            except Exception:
                pass

    def clear(self) -> None:
        """Clear routing cache, prefetch cache, and context history."""
        with self._lock:
            self._routing_cache.clear()
            self._prefetched_weights.clear()
            self._recent_pairs.clear()
            self._transition_counts.clear()
            self._expert_popularity.clear()
            self._inflight_prefetch.clear()
            self._sketch_index_cache.clear()

    def get_cached_experts(self, hidden_state: TensorType | Sequence[float]) -> tuple[tuple[int, ...], tuple[float, ...]] | None:
        """Get cached expert selection for hidden state if exists.
        
        This is a simpler version of get_or_route that only checks cache without routing fallback.
        Returns (expert_ids, expert_probs) on cache hit, None on miss.
        """
        hash_val = self.hidden_state_hash(hidden_state)
        with self._lock:
            if hash_val in self._routing_cache:
                entry = self._routing_cache[hash_val]
                entry.touch()
                self._routing_cache.move_to_end(hash_val)
                return entry.expert_ids, entry.expert_probs
        return None

    def get_stats(self) -> dict[str, Any]:
        """Return cache and prefetch metrics."""
        with self._lock:
            return {
                "device": self.device,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "routing_cache_size": len(self._routing_cache),
                "routing_cache_capacity": max(1, int(self.config.max_routing_entries)),
                "prefetched_weight_count": len(self._prefetched_weights),
                "prefetched_weight_capacity": max(1, int(self.config.max_prefetched_weights)),
                "stats": self._stats.as_dict(),
            }

    def close(self) -> None:
        """Release background resources."""
        self.wait_for_prefetch()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> ExpertSelectionCache:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        stats = self._stats
        return (
            "ExpertSelectionCache("
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"routing_entries={len(self._routing_cache)}, "
            f"prefetched_weights={len(self._prefetched_weights)}, "
            f"hit_rate={stats.cache_hit_rate:.1%})"
        )
