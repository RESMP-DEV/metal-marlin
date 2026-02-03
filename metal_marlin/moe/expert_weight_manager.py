"""Expert weight manager with predictive prefetching.

Manages expert weight loading/unloading with prefetch to overlap I/O with computation.
Integrates ExpertPrefetcher with an LRU cache and async weight loading.

Usage:
    from metal_marlin.moe.expert_weight_manager import ExpertWeightManager
    
    # Initialize with model config
    manager = ExpertWeightManager(
        num_experts=64,
        num_layers=28,
        expert_size_mb=50,
        cache_capacity_mb=1024,
        device="mps",
    )
    
    # During forward pass
    for layer_idx in range(num_layers):
        # Get experts for current token (blocks until ready)
        weights = manager.get_experts(layer_idx, expert_ids)
        
        # Compute with experts
        output = moe_forward(hidden, weights)
        
        # Prefetch next experts in background (non-blocking)
        manager.prefetch_next(layer_idx, expert_ids, attention_pattern)
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .prefetch import ExpertPrefetcher, PrefetchConfig

if TYPE_CHECKING:

    TensorType = Any
else:
    TensorType = Any


@dataclass
class WeightManagerConfig:
    """Configuration for expert weight manager.
    
    Attributes:
        cache_capacity_mb: Maximum cache size in megabytes.
        prefetch_config: Configuration for expert prefetching.
        enable_lru: Use LRU eviction policy (vs. no eviction).
        load_timeout_sec: Timeout for weight loading operations.
        enable_metrics: Track cache hit/miss metrics.
    """
    cache_capacity_mb: int = 1024
    prefetch_config: PrefetchConfig = field(default_factory=PrefetchConfig)
    enable_lru: bool = True
    load_timeout_sec: float = 5.0
    enable_metrics: bool = True


class ExpertWeightManager:
    """Manages expert weights with predictive prefetching.
    
    Coordinates three components:
    1. LRU cache for loaded weights
    2. Predictive prefetcher for next-token expert prediction
    3. Async loader for background weight loading
    
    Thread-safe for concurrent get/prefetch operations.
    """
    
    def __init__(
        self,
        num_experts: int,
        num_layers: int,
        expert_size_mb: float,
        config: WeightManagerConfig | None = None,
        device: str = "mps",
        weights_dir: Path | str | None = None,
    ):
        """Initialize expert weight manager.
        
        Args:
            num_experts: Number of experts per layer.
            num_layers: Number of MoE layers.
            expert_size_mb: Size of each expert in megabytes.
            config: Manager configuration.
            device: Target device for weights ("mps", "cpu", "cuda").
            weights_dir: Directory containing expert checkpoint files.
        """
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.expert_size_mb = expert_size_mb
        self.config = config or WeightManagerConfig()
        self.device = device
        self.weights_dir = Path(weights_dir) if weights_dir else None
        
        # LRU cache for loaded weights
        self._cache: OrderedDict[tuple[int, int], TensorType] = OrderedDict()
        self._cache_lock = threading.RLock()
        self._max_cache_size = int(self.config.cache_capacity_mb / expert_size_mb)
        
        # Prefetcher for predicting next experts
        self._prefetcher = ExpertPrefetcher(
            num_experts=num_experts,
            num_layers=num_layers,
            config=self.config.prefetch_config,
            load_fn=self._load_expert_async,
        )
        
        # Metrics
        self._metrics = {
            "hits": 0,
            "misses": 0,
            "prefetch_hits": 0,
            "load_time_ms": [],
        }
        self._metrics_lock = threading.Lock()
    
    def get_experts(
        self,
        layer_idx: int,
        expert_ids: list[int] | TensorType,
    ) -> list[TensorType]:
        """Get expert weights, blocking until loaded.
        
        Args:
            layer_idx: Layer index.
            expert_ids: Expert IDs to retrieve.
            
        Returns:
            List of expert weight tensors.
        """
        if not isinstance(expert_ids, list):
            expert_ids = self._tensor_to_list(expert_ids)
        
        weights = []
        for expert_id in expert_ids:
            weight = self._get_cached_or_load(layer_idx, expert_id)
            weights.append(weight)
        
        return weights
    
    def prefetch_next(
        self,
        layer_idx: int,
        current_expert_ids: list[int] | TensorType,
        attention_pattern: TensorType | None = None,
    ) -> list[int]:
        """Prefetch predicted experts for next token (non-blocking).
        
        Args:
            layer_idx: Layer index.
            current_expert_ids: Current token's expert IDs.
            attention_pattern: Optional attention weights for prediction.
            
        Returns:
            Predicted expert IDs for next token.
        """
        # Record current routing and predict next experts
        predicted = self._prefetcher.step(
            layer_idx=layer_idx,
            expert_ids=current_expert_ids,
            attention_pattern=attention_pattern,
        )
        
        return predicted
    
    def wait_prefetch(self, timeout: float | None = None) -> None:
        """Wait for all pending prefetch operations to complete.
        
        Args:
            timeout: Maximum time to wait in seconds.
        """
        self._prefetcher.wait_prefetch(timeout=timeout)
    
    def _get_cached_or_load(self, layer_idx: int, expert_id: int) -> TensorType:
        """Get expert from cache or load synchronously."""
        key = (layer_idx, expert_id)
        
        with self._cache_lock:
            # Check cache
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._record_hit()
                return self._cache[key]
            
            # Cache miss - load synchronously
            self._record_miss()
        
        # Load outside lock to avoid blocking cache access
        start = time.time()
        weight = self._load_expert_sync(layer_idx, expert_id)
        load_time_ms = (time.time() - start) * 1000
        
        # Add to cache
        with self._cache_lock:
            self._add_to_cache(key, weight)
            
        if self.config.enable_metrics:
            with self._metrics_lock:
                self._metrics["load_time_ms"].append(load_time_ms)
        
        return weight
    
    def _add_to_cache(self, key: tuple[int, int], weight: TensorType) -> None:
        """Add weight to cache with LRU eviction (must hold cache lock)."""
        # Evict if at capacity
        if self.config.enable_lru:
            while len(self._cache) >= self._max_cache_size:
                self._cache.popitem(last=False)  # Remove oldest
        
        self._cache[key] = weight
        self._cache.move_to_end(key)  # Mark as most recent
    
    def _load_expert_sync(self, layer_idx: int, expert_id: int) -> TensorType:
        """Load expert weights synchronously from disk/network.
        
        Override this method to implement custom loading logic.
        """
        # Placeholder - implement actual loading logic
        # In real usage, this would load from checkpoint files
        if self.weights_dir:
            weight_path = self.weights_dir / f"layer_{layer_idx}" / f"expert_{expert_id}.pt"
            # Load from file (requires torch)
            try:
                import torch
                return torch.load(weight_path, map_location=self.device)
            except ImportError:
                pass
        
        # Dummy weight for testing
        return {"dummy": f"layer{layer_idx}_expert{expert_id}"}
    
    def _load_expert_async(self, layer_idx: int, expert_id: int) -> None:
        """Background load for prefetching (called by ExpertPrefetcher)."""
        key = (layer_idx, expert_id)
        
        # Skip if already cached
        with self._cache_lock:
            if key in self._cache:
                if self.config.enable_metrics:
                    with self._metrics_lock:
                        self._metrics["prefetch_hits"] += 1
                return
        
        # Load weight
        weight = self._load_expert_sync(layer_idx, expert_id)
        
        # Add to cache
        with self._cache_lock:
            self._add_to_cache(key, weight)
    
    def _record_hit(self) -> None:
        """Record cache hit."""
        if self.config.enable_metrics:
            with self._metrics_lock:
                self._metrics["hits"] += 1
    
    def _record_miss(self) -> None:
        """Record cache miss."""
        if self.config.enable_metrics:
            with self._metrics_lock:
                self._metrics["misses"] += 1
    
    def get_metrics(self) -> dict[str, Any]:
        """Get cache and prefetch metrics.
        
        Returns:
            Dictionary with hit rates, load times, and prefetch stats.
        """
        with self._metrics_lock:
            hits = self._metrics["hits"]
            misses = self._metrics["misses"]
            total = hits + misses
            hit_rate = hits / total if total > 0 else 0.0
            
            load_times = self._metrics["load_time_ms"]
            avg_load_ms = sum(load_times) / len(load_times) if load_times else 0.0
            
            metrics = {
                "cache": {
                    "hits": hits,
                    "misses": misses,
                    "hit_rate": hit_rate,
                    "size": len(self._cache),
                    "capacity": self._max_cache_size,
                },
                "loading": {
                    "avg_load_time_ms": avg_load_ms,
                    "num_loads": len(load_times),
                },
                "prefetch": self._prefetcher.get_stats(),
            }
        
        return metrics
    
    def clear_cache(self) -> None:
        """Clear all cached weights."""
        with self._cache_lock:
            self._cache.clear()
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._metrics_lock:
            self._metrics = {
                "hits": 0,
                "misses": 0,
                "prefetch_hits": 0,
                "load_time_ms": [],
            }
    
    @staticmethod
    def _tensor_to_list(tensor: TensorType) -> list[int]:
        """Convert tensor to list of ints."""
        if isinstance(tensor, list):
            return tensor
        
        # Try torch tensor conversion
        try:
            if hasattr(tensor, "cpu"):
                tensor = tensor.cpu()
            if hasattr(tensor, "numpy"):
                return tensor.numpy().flatten().tolist()
            if hasattr(tensor, "tolist"):
                return tensor.tolist()
        except Exception:
            pass
        
        # Fallback: assume it's iterable
        return list(tensor)
