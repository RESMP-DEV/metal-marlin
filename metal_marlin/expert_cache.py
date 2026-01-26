"""Intelligent expert caching for MoE models on Unified Memory.

On Apple Silicon's Unified Memory, all experts are always resident in memory.
The bottleneck is dequantization, not memory access. This module provides:

1. LRU cache for dequantized expert tiles - avoids redundant dequantization
2. Expert activation frequency tracking - informs prefetch decisions
3. History-based prefetching - speculatively dequantize likely-needed experts
4. Per-layer statistics - enables adaptive cache sizing

Key insight: With 64 experts and top-2 routing, only ~3% of experts are "hot"
at any given time. Caching their dequantized weights provides significant
speedup without the memory pressure of caching all experts.

Usage:
    from metal_marlin.expert_cache import ExpertCache

    # Create cache for a model with 64 experts
    cache = ExpertCache(
        num_experts=64,
        num_layers=28,
        cache_size_mb=512,
        tile_shape=(64, 64),  # Match GEMM tile dimensions
    )

    # During forward pass
    tile = cache.get_expert_tile(
        layer_idx=0,
        expert_id=5,
        tile_idx=0,
        dequant_fn=lambda: dequant_fp4(packed, scales, K, N, group_size),
    )
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    pass


@dataclass
class TileKey:
    """Unique identifier for a cached expert tile.

    A tile is identified by:
    - layer_idx: Which transformer layer (0 to num_layers-1)
    - expert_id: Which expert (0 to num_experts-1)
    - tile_idx: Which tile within the expert's weight matrix

    The tile_idx encodes both row and column tile indices packed into one int:
        tile_idx = row_tile * num_col_tiles + col_tile
    """
    layer_idx: int
    expert_id: int
    tile_idx: int

    def __hash__(self) -> int:
        return hash((self.layer_idx, self.expert_id, self.tile_idx))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TileKey):
            return NotImplemented
        return (
            self.layer_idx == other.layer_idx
            and self.expert_id == other.expert_id
            and self.tile_idx == other.tile_idx
        )


@dataclass
class CacheEntry:
    """A cached dequantized tile with metadata."""
    key: TileKey
    data: mx.array
    size_bytes: int
    access_count: int = 0
    last_access: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_access = time.time()


@dataclass
class ExpertStats:
    """Activation statistics for a single expert."""
    expert_id: int
    activation_count: int = 0
    total_tokens: int = 0
    recent_window: list[int] = field(default_factory=list)  # Recent activation counts

    @property
    def activation_rate(self) -> float:
        """Fraction of tokens that activated this expert."""
        if self.total_tokens == 0:
            return 0.0
        return self.activation_count / self.total_tokens

    def record_activation(self, count: int = 1) -> None:
        """Record expert activation(s)."""
        self.activation_count += count
        self.recent_window.append(count)
        # Keep only last 100 windows
        if len(self.recent_window) > 100:
            self.recent_window.pop(0)

    def record_batch(self, batch_size: int) -> None:
        """Record a batch of tokens for rate calculation."""
        self.total_tokens += batch_size

    @property
    def recent_rate(self) -> float:
        """Recent activation rate (last 100 windows)."""
        if not self.recent_window:
            return 0.0
        return sum(self.recent_window) / len(self.recent_window)


@dataclass
class LayerStats:
    """Per-layer statistics for cache tuning."""
    layer_idx: int
    expert_stats: dict[int, ExpertStats] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    prefetch_hits: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate for this layer."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def get_expert_stats(self, expert_id: int) -> ExpertStats:
        """Get or create stats for an expert."""
        if expert_id not in self.expert_stats:
            self.expert_stats[expert_id] = ExpertStats(expert_id=expert_id)
        return self.expert_stats[expert_id]

    def get_hot_experts(self, threshold: float = 0.1, top_k: int | None = None) -> list[int]:
        """Get experts with activation rate above threshold.

        Args:
            threshold: Minimum activation rate to be considered "hot"
            top_k: If provided, return at most top_k experts sorted by rate

        Returns:
            List of expert IDs sorted by activation rate (descending)
        """
        hot = [
            (eid, stats.recent_rate)
            for eid, stats in self.expert_stats.items()
            if stats.recent_rate >= threshold
        ]
        hot.sort(key=lambda x: -x[1])

        if top_k is not None:
            hot = hot[:top_k]

        return [eid for eid, _ in hot]


class ExpertCache:
    """LRU cache for dequantized expert tiles with intelligent prefetching.

    The cache stores dequantized tiles from MoE expert weight matrices. When a
    tile is requested, the cache either returns the cached version or triggers
    dequantization and caches the result.

    Cache eviction uses LRU policy with size-based limits. Statistics are
    maintained per-layer to enable adaptive sizing and prefetch decisions.

    Thread-safe: All operations are protected by a lock for concurrent access.

    Args:
        num_experts: Number of experts in the MoE layer
        num_layers: Number of transformer layers with MoE
        cache_size_mb: Maximum cache size in megabytes
        tile_shape: Shape of each cached tile (rows, cols)
        enable_prefetch: Whether to enable speculative prefetching
        prefetch_k: Number of likely experts to prefetch per layer
    """

    def __init__(
        self,
        num_experts: int,
        num_layers: int,
        cache_size_mb: int = 512,
        tile_shape: tuple[int, int] = (64, 64),
        enable_prefetch: bool = True,
        prefetch_k: int = 4,
    ):
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.cache_size_bytes = cache_size_mb * 1024 * 1024
        self.tile_shape = tile_shape
        self.enable_prefetch = enable_prefetch
        self.prefetch_k = prefetch_k

        # LRU cache: OrderedDict maintains insertion order, move_to_end() for LRU
        self._cache: OrderedDict[TileKey, CacheEntry] = OrderedDict()
        self._current_size: int = 0

        # Per-layer statistics
        self._layer_stats: dict[int, LayerStats] = {
            i: LayerStats(layer_idx=i) for i in range(num_layers)
        }

        # Global statistics
        self._total_hits: int = 0
        self._total_misses: int = 0
        self._total_evictions: int = 0

        # Thread safety
        self._lock = threading.RLock()

        # Prefetch queue: tiles to dequantize in background
        self._prefetch_queue: list[tuple[TileKey, Callable[[], mx.array]]] = []

    def get_expert_tile(
        self,
        layer_idx: int,
        expert_id: int,
        tile_idx: int,
        dequant_fn: Callable[[], mx.array],
    ) -> mx.array:
        """Get a dequantized expert tile, using cache if available.

        If the tile is cached, returns immediately. Otherwise, calls dequant_fn
        to compute the tile, caches it, and returns the result.

        Args:
            layer_idx: Transformer layer index
            expert_id: Expert index within the layer
            tile_idx: Tile index within the expert's weight matrix
            dequant_fn: Callable that returns the dequantized tile if not cached

        Returns:
            Dequantized tile as mx.array
        """
        key = TileKey(layer_idx, expert_id, tile_idx)

        with self._lock:
            # Check cache first
            if key in self._cache:
                entry = self._cache[key]
                entry.touch()
                # Move to end for LRU
                self._cache.move_to_end(key)

                # Update stats
                self._total_hits += 1
                self._layer_stats[layer_idx].cache_hits += 1

                return entry.data

            # Cache miss - dequantize
            self._total_misses += 1
            self._layer_stats[layer_idx].cache_misses += 1

        # Dequantization outside lock (can be slow)
        tile_data = dequant_fn()
        mx.eval(tile_data)  # Ensure computation completes

        with self._lock:
            # Add to cache
            self._put(key, tile_data)

            return tile_data

    def _put(self, key: TileKey, data: mx.array) -> None:
        """Add a tile to the cache, evicting if necessary.

        Must be called with lock held.
        """
        # Calculate size
        size_bytes = data.nbytes

        # Evict if necessary
        while self._current_size + size_bytes > self.cache_size_bytes and self._cache:
            self._evict_one()

        # Don't cache tiles larger than entire cache
        if size_bytes > self.cache_size_bytes:
            return

        # Add entry
        entry = CacheEntry(
            key=key,
            data=data,
            size_bytes=size_bytes,
        )
        entry.touch()

        self._cache[key] = entry
        self._current_size += size_bytes

    def _evict_one(self) -> None:
        """Evict the least recently used entry.

        Must be called with lock held.
        """
        if not self._cache:
            return

        # OrderedDict: first item is LRU
        key, entry = next(iter(self._cache.items()))
        del self._cache[key]
        self._current_size -= entry.size_bytes
        self._total_evictions += 1

    def record_expert_activation(
        self,
        layer_idx: int,
        expert_ids: mx.array,
    ) -> None:
        """Record which experts were activated for a batch.

        This information is used to track activation frequency and inform
        prefetch decisions. Should be called during forward pass after
        router selects experts.

        Args:
            layer_idx: Transformer layer index
            expert_ids: Array of activated expert IDs, shape [batch, top_k]
        """
        with self._lock:
            layer_stats = self._layer_stats[layer_idx]
            batch_size = expert_ids.shape[0]

            # Convert to numpy for iteration
            ids = list(expert_ids.reshape(-1).tolist())

            # Count activations per expert
            counts: dict[int, int] = {}
            for eid in ids:
                counts[eid] = counts.get(eid, 0) + 1

            # Update expert stats
            for eid, count in counts.items():
                expert_stats = layer_stats.get_expert_stats(eid)
                expert_stats.record_activation(count)

            # Record batch for all experts
            for expert_stats in layer_stats.expert_stats.values():
                expert_stats.record_batch(batch_size)

    def get_prefetch_candidates(self, layer_idx: int) -> list[int]:
        """Get list of experts likely to be needed, based on history.

        Uses recent activation rates to predict which experts should be
        prefetched. Returns the top-k "hottest" experts for the layer.

        Args:
            layer_idx: Transformer layer index

        Returns:
            List of expert IDs to prefetch, sorted by likelihood
        """
        with self._lock:
            layer_stats = self._layer_stats[layer_idx]
            return layer_stats.get_hot_experts(
                threshold=0.01,  # At least 1% activation rate
                top_k=self.prefetch_k,
            )

    def prefetch_expert_tiles(
        self,
        layer_idx: int,
        expert_id: int,
        tile_indices: list[int],
        dequant_fn: Callable[[int], mx.array],
    ) -> None:
        """Speculatively prefetch tiles for an expert.

        Tiles that are already cached are skipped. This should be called
        in a background thread or when there's idle time.

        Args:
            layer_idx: Transformer layer index
            expert_id: Expert index to prefetch
            tile_indices: List of tile indices to prefetch
            dequant_fn: Function that takes tile_idx and returns dequantized data
        """
        if not self.enable_prefetch:
            return

        with self._lock:
            for tile_idx in tile_indices:
                key = TileKey(layer_idx, expert_id, tile_idx)

                # Skip if already cached
                if key in self._cache:
                    continue

                # Queue for prefetch
                self._prefetch_queue.append((key, lambda t=tile_idx: dequant_fn(t)))

        # Process prefetch queue (could be done async)
        self._process_prefetch_queue()

    def _process_prefetch_queue(self, max_items: int = 8) -> None:
        """Process pending prefetch requests.

        Processes up to max_items from the queue to avoid blocking too long.
        """
        items_to_process = []

        with self._lock:
            while self._prefetch_queue and len(items_to_process) < max_items:
                key, fn = self._prefetch_queue.pop(0)
                # Skip if already cached (might have been filled by a real request)
                if key not in self._cache:
                    items_to_process.append((key, fn))

        # Dequantize outside lock
        for key, fn in items_to_process:
            data = fn()
            mx.eval(data)

            with self._lock:
                # Double-check before adding
                if key not in self._cache:
                    self._put(key, data)
                    self._layer_stats[key.layer_idx].prefetch_hits += 1

    def invalidate_expert(self, layer_idx: int, expert_id: int) -> int:
        """Remove all cached tiles for a specific expert.

        Useful when expert weights are updated (e.g., fine-tuning).

        Args:
            layer_idx: Transformer layer index
            expert_id: Expert index to invalidate

        Returns:
            Number of tiles evicted
        """
        with self._lock:
            keys_to_remove = [
                key for key in self._cache
                if key.layer_idx == layer_idx and key.expert_id == expert_id
            ]

            for key in keys_to_remove:
                entry = self._cache.pop(key)
                self._current_size -= entry.size_bytes

            return len(keys_to_remove)

    def invalidate_layer(self, layer_idx: int) -> int:
        """Remove all cached tiles for a specific layer.

        Args:
            layer_idx: Transformer layer index

        Returns:
            Number of tiles evicted
        """
        with self._lock:
            keys_to_remove = [
                key for key in self._cache if key.layer_idx == layer_idx
            ]

            for key in keys_to_remove:
                entry = self._cache.pop(key)
                self._current_size -= entry.size_bytes

            return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._current_size = 0
            self._prefetch_queue.clear()

    def resize(self, new_size_mb: int) -> int:
        """Resize the cache, evicting entries if necessary.

        Args:
            new_size_mb: New maximum cache size in megabytes

        Returns:
            Number of entries evicted
        """
        new_size_bytes = new_size_mb * 1024 * 1024
        evicted = 0

        with self._lock:
            self.cache_size_bytes = new_size_bytes

            while self._current_size > self.cache_size_bytes and self._cache:
                self._evict_one()
                evicted += 1

        return evicted

    @property
    def hit_rate(self) -> float:
        """Overall cache hit rate."""
        total = self._total_hits + self._total_misses
        return self._total_hits / total if total > 0 else 0.0

    @property
    def size_mb(self) -> float:
        """Current cache size in megabytes."""
        return self._current_size / (1024 * 1024)

    @property
    def num_entries(self) -> int:
        """Number of cached entries."""
        return len(self._cache)

    def get_stats(self) -> dict:
        """Get comprehensive cache statistics.

        Returns:
            Dictionary with cache statistics including:
            - Global hit/miss rates
            - Per-layer statistics
            - Memory usage
            - Expert activation patterns
        """
        with self._lock:
            layer_stats = {}
            for layer_idx, stats in self._layer_stats.items():
                hot_experts = stats.get_hot_experts(threshold=0.05, top_k=10)
                layer_stats[layer_idx] = {
                    "hit_rate": stats.hit_rate,
                    "cache_hits": stats.cache_hits,
                    "cache_misses": stats.cache_misses,
                    "prefetch_hits": stats.prefetch_hits,
                    "hot_experts": hot_experts,
                    "num_experts_tracked": len(stats.expert_stats),
                }

            return {
                "global": {
                    "hit_rate": self.hit_rate,
                    "total_hits": self._total_hits,
                    "total_misses": self._total_misses,
                    "total_evictions": self._total_evictions,
                },
                "memory": {
                    "size_mb": self.size_mb,
                    "max_size_mb": self.cache_size_bytes / (1024 * 1024),
                    "num_entries": self.num_entries,
                    "utilization": self._current_size / self.cache_size_bytes if self.cache_size_bytes > 0 else 0.0,
                },
                "config": {
                    "num_experts": self.num_experts,
                    "num_layers": self.num_layers,
                    "tile_shape": self.tile_shape,
                    "enable_prefetch": self.enable_prefetch,
                    "prefetch_k": self.prefetch_k,
                },
                "per_layer": layer_stats,
            }

    def __repr__(self) -> str:
        return (
            f"ExpertCache("
            f"num_experts={self.num_experts}, "
            f"num_layers={self.num_layers}, "
            f"size={self.size_mb:.1f}/{self.cache_size_bytes / (1024 * 1024):.0f}MB, "
            f"entries={self.num_entries}, "
            f"hit_rate={self.hit_rate:.1%})"
        )


class TileCoordinator:
    """Helper class to manage tile indexing for expert weight matrices.

    Translates between linear tile indices and (row, col) tile coordinates.
    Useful for iterating over tiles or computing prefetch patterns.

    Args:
        weight_shape: Shape of expert weight matrix (out_features, in_features)
        tile_shape: Shape of each tile (tile_rows, tile_cols)
    """

    def __init__(
        self,
        weight_shape: tuple[int, int],
        tile_shape: tuple[int, int] = (64, 64),
    ):
        self.weight_shape = weight_shape
        self.tile_shape = tile_shape

        self.num_row_tiles = (weight_shape[0] + tile_shape[0] - 1) // tile_shape[0]
        self.num_col_tiles = (weight_shape[1] + tile_shape[1] - 1) // tile_shape[1]
        self.num_tiles = self.num_row_tiles * self.num_col_tiles

    def tile_to_coords(self, tile_idx: int) -> tuple[int, int]:
        """Convert linear tile index to (row_tile, col_tile) coordinates."""
        row_tile = tile_idx // self.num_col_tiles
        col_tile = tile_idx % self.num_col_tiles
        return row_tile, col_tile

    def coords_to_tile(self, row_tile: int, col_tile: int) -> int:
        """Convert (row_tile, col_tile) coordinates to linear tile index."""
        return row_tile * self.num_col_tiles + col_tile

    def tile_bounds(self, tile_idx: int) -> tuple[int, int, int, int]:
        """Get row/col bounds for a tile.

        Returns:
            (row_start, row_end, col_start, col_end) - half-open intervals
        """
        row_tile, col_tile = self.tile_to_coords(tile_idx)

        row_start = row_tile * self.tile_shape[0]
        row_end = min(row_start + self.tile_shape[0], self.weight_shape[0])

        col_start = col_tile * self.tile_shape[1]
        col_end = min(col_start + self.tile_shape[1], self.weight_shape[1])

        return row_start, row_end, col_start, col_end

    def all_tile_indices(self) -> list[int]:
        """Get list of all tile indices."""
        return list(range(self.num_tiles))

    def tiles_for_output_range(
        self,
        out_start: int,
        out_end: int,
    ) -> list[int]:
        """Get tile indices that cover a range of output rows.

        Useful for selective prefetching when you know which outputs are needed.

        Args:
            out_start: First output row needed
            out_end: Last output row needed (exclusive)

        Returns:
            List of tile indices that cover the output range
        """
        start_tile_row = out_start // self.tile_shape[0]
        end_tile_row = (out_end - 1) // self.tile_shape[0] + 1

        tiles = []
        for row_tile in range(start_tile_row, end_tile_row):
            for col_tile in range(self.num_col_tiles):
                tiles.append(self.coords_to_tile(row_tile, col_tile))

        return tiles


def create_moe_cache(
    config: dict,
    cache_size_mb: int = 512,
) -> ExpertCache:
    """Create an ExpertCache from model configuration.

    Automatically extracts num_experts and num_layers from model config dict.
    Supports common MoE architectures (GLM-4, Mixtral, Qwen-MoE).

    Args:
        config: Model configuration dictionary with keys like:
            - num_hidden_layers / n_layer / num_layers
            - num_experts / num_local_experts / n_experts
        cache_size_mb: Maximum cache size in megabytes

    Returns:
        Configured ExpertCache instance
    """
    # Extract num_layers
    num_layers = (
        config.get("num_hidden_layers")
        or config.get("n_layer")
        or config.get("num_layers")
        or 32  # Reasonable default
    )

    # Extract num_experts
    num_experts = (
        config.get("num_experts")
        or config.get("num_local_experts")
        or config.get("n_experts")
        or 64  # Reasonable default for modern MoE
    )

    # Determine tile shape based on typical GEMM tile sizes
    # Match the Metal kernel tile dimensions
    tile_shape = (64, 64)

    return ExpertCache(
        num_experts=num_experts,
        num_layers=num_layers,
        cache_size_mb=cache_size_mb,
        tile_shape=tile_shape,
    )
