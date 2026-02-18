"""Optimized KV cache manager with advanced eviction policies.

This module provides PagedKVCacheOptimized, an extension of PagedKVCache
that implements advanced eviction policies including Weighted, Adaptive,
and Priority-based eviction for long-running servers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Optional, Dict, List, Tuple

import numpy as np

from .cache_manager import PagedKVCache, EvictionStats, SequenceEvictionMetadata
from .kv_block import KVBlockConfig

class EvictionPolicy(Enum):
    """Cache eviction policy."""
    LRU = "lru"
    FIFO = "fifo"
    LFU = "lfu"
    WEIGHTED = "weighted"
    ADAPTIVE = "adaptive"

@dataclass
class EvictionConfig:
    """Configuration for eviction policies."""
    policy: EvictionPolicy = EvictionPolicy.WEIGHTED
    max_size: Optional[int] = None
    
    # Weights for WEIGHTED policy
    recency_weight: float = 1.0
    frequency_weight: float = 1.0
    size_weight: float = 0.5
    fragmentation_weight: float = 0.0
    
    # Adaptive policy settings
    window_size: int = 10
    
    # TTL settings
    enable_ttl: bool = False
    default_ttl_seconds: float = 3600.0
    
    # Batch eviction
    batch_eviction: bool = False
    batch_size: int = 1
    
    # Stats
    track_stats: bool = False
    track_detailed_stats: bool = False
    
    # Memory pressure
    memory_pressure_threshold: float = 0.9
    proactive_eviction: bool = False

    def __post_init__(self):
        if self.recency_weight < 0:
            raise ValueError("recency_weight must be non-negative")
        if self.memory_pressure_threshold > 1.0:
            raise ValueError("memory_pressure_threshold must be <= 1.0")

class WeightedScore:
    """Helper for weighted score calculation."""
    @staticmethod
    def calculate(recency: float, frequency: float, size: float, fragmentation: float, config: EvictionConfig) -> float:
        return (
            recency * config.recency_weight +
            frequency * config.frequency_weight +
            size * config.size_weight -
            fragmentation * config.fragmentation_weight
        )

    @staticmethod
    def normalize_scores(scores: list[float]) -> list[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

@dataclass
class OptimizedSequenceEvictionMetadata(SequenceEvictionMetadata):
    """Extended metadata for optimized eviction."""
    priority: int = 0
    ttl: Optional[float] = None
    
class PagedKVCacheOptimized(PagedKVCache):
    """Paged KV cache with optimized eviction policies."""

    def __init__(
        self,
        config: KVBlockConfig | None = None,
        num_blocks: int = 1024,
        use_multimodal: bool = False,
        eviction_config: EvictionConfig | None = None,
        eviction_policy: Any = None, 
    ) -> None:
        # Initialize base
        super().__init__(
            config=config,
            num_blocks=num_blocks,
            use_multimodal=use_multimodal,
        )
        
        self.eviction_config = eviction_config or EvictionConfig()
        
        # Override metadata storage with optimized version
        self._eviction_metadata: dict[int, OptimizedSequenceEvictionMetadata] = {}
        
        self._fragmentation_info = {'fragmentation_ratio': 0.0}
        self._workload_history: List[str] = []
        self._detailed_stats = {
            'avg_score_evicted': 0.0,
            'eviction_reasons': {}
        }

    def add_sequence(self, seq_id: int, priority: int = 0, ttl: Optional[float] = None) -> bool:
        """Add a new sequence with priority and TTL."""
        # Ensure we have at least one block available
        if self.allocator.num_free == 0:
            if not self._evict(num_blocks_needed=1):
                return False

        success = self.page_table.add_sequence(seq_id)
        if success:
            if self.use_multimodal and hasattr(self.allocator, 'register_sequence'):
                self.allocator.register_sequence(seq_id)
            
            # Initialize optimized eviction metadata
            now = time.time()
            self._eviction_metadata[seq_id] = OptimizedSequenceEvictionMetadata(
                seq_id=seq_id,
                last_access_time=now,
                access_count=1,
                creation_time=now,
                priority=priority,
                ttl=ttl or (self.eviction_config.default_ttl_seconds if self.eviction_config.enable_ttl else None)
            )

        return success

    def fork_sequence(self, src_id: int, dst_id: int) -> bool:
        """Fork sequence with zero-copy COW."""
        success = super().fork_sequence(src_id, dst_id)
        if success:
            # Update metadata to be OptimizedSequenceEvictionMetadata
            # super().fork_sequence initializes it as SequenceEvictionMetadata
            # We need to replace it.
            
            # Inherit priority/ttl from parent? For now use defaults/parent values
            src_meta = self._eviction_metadata.get(src_id)
            priority = src_meta.priority if src_meta else 0
            ttl = src_meta.ttl if src_meta else None
            
            now = time.time()
            self._eviction_metadata[dst_id] = OptimizedSequenceEvictionMetadata(
                seq_id=dst_id,
                last_access_time=now,
                access_count=1,
                creation_time=now,
                priority=priority,
                ttl=ttl
            )
        return success

    def _compute_eviction_score(self, seq_id: int, override_config: EvictionConfig | None = None) -> float:
        """Compute eviction score for a sequence. Higher score = LESS likely to evict."""
        if seq_id not in self._eviction_metadata:
            return 0.0
            
        meta = self._eviction_metadata[seq_id]
        
        # Calculate raw components
        recency = meta.last_access_time
        frequency = float(meta.access_count)
        
        # Get sequence size and fragmentation
        state = self.page_table.sequences.get(seq_id)
        size = 0.0
        fragmentation = 0.0
        
        if state:
            size = float(len(state.block_indices))
            if size > 0:
                total_capacity = size * self.config.block_size
                if total_capacity > 0:
                    utilization = state.logical_len / total_capacity
                    # Calculate wasted blocks (internal fragmentation)
                    fragmentation_ratio = 1.0 - utilization
                    fragmentation = fragmentation_ratio * size

        # Apply adaptive policy adjustments if not overridden
        config = override_config or self.eviction_config
        if override_config is None and config.policy == EvictionPolicy.ADAPTIVE:
            pressure = self.get_memory_pressure()
            if pressure > config.memory_pressure_threshold:
                # Under pressure, favor LRU (recency) and penalize large/fragmented items more
                config = replace(
                    config,
                    recency_weight=config.recency_weight * 2.0,
                    size_weight=config.size_weight * 0.5,
                    # Strongly penalize wasted space (more than size gain)
                    fragmentation_weight=max(config.fragmentation_weight, config.size_weight + 1.0)
                )
        
        return WeightedScore.calculate(
            recency=recency,
            frequency=frequency,
            size=size,
            fragmentation=fragmentation,
            config=config
        )

    def _evict(self, num_blocks_needed: int) -> bool:
        """Optimized eviction."""
        # Optimization: cleanup expired first to free space without forced eviction
        if self.eviction_config.enable_ttl:
            self.cleanup_expired()
            if self.allocator.num_free >= num_blocks_needed:
                return True

        if not self._eviction_metadata:
            return False
            
        candidates = list(self._eviction_metadata.values())
        
        # Sort based on policy
        policy = self.eviction_config.policy
        
        if policy == EvictionPolicy.LRU:
            candidates.sort(key=lambda m: m.last_access_time)
        elif policy == EvictionPolicy.FIFO:
            candidates.sort(key=lambda m: m.creation_time)
        elif policy == EvictionPolicy.LFU:
            candidates.sort(key=lambda m: m.access_count)
        elif policy == EvictionPolicy.WEIGHTED or policy == EvictionPolicy.ADAPTIVE:
            # Prepare config
            config = self.eviction_config
            pattern = self._detect_workload_pattern()
            
            if policy == EvictionPolicy.ADAPTIVE:
                pressure = self.get_memory_pressure()
                
                # Base adjustments
                new_recency = config.recency_weight
                new_frequency = config.frequency_weight
                new_size = config.size_weight
                new_frag = config.fragmentation_weight
                
                if pressure > config.memory_pressure_threshold:
                    # Under high pressure, prioritize fragmentation reduction
                    new_frag = max(new_frag, 2.0)
                    
                if pattern == "scan":
                    # Scan workload: Recency is noise, Frequency is signal.
                    # Protect frequent items, evict one-hit wonders.
                    new_recency *= 0.1
                    new_frequency = max(new_frequency, 2.0)
                else:
                    # Working set: Recency is strong signal (LRU)
                    new_recency = max(new_recency, 1.0)

                config = replace(
                    config,
                    recency_weight=new_recency,
                    frequency_weight=new_frequency,
                    size_weight=new_size,
                    fragmentation_weight=new_frag
                )

            # Collect raw metrics for normalization
            metrics = []
            now = time.time()
            for m in candidates:
                state = self.page_table.sequences.get(m.seq_id)
                size = 0.0
                fragmentation = 0.0
                if state:
                    size = float(len(state.block_indices))
                    if size > 0:
                        total_capacity = size * self.config.block_size
                        if total_capacity > 0:
                            utilization = state.logical_len / total_capacity
                            fragmentation = (1.0 - utilization) * size
                
                metrics.append({
                    'meta': m,
                    'recency': m.last_access_time,
                    'frequency': float(m.access_count),
                    'size': size,
                    'fragmentation': fragmentation
                })
            
            # Normalize metrics
            if metrics:
                recencies = [x['recency'] for x in metrics]
                frequencies = [x['frequency'] for x in metrics]
                sizes = [x['size'] for x in metrics]
                frags = [x['fragmentation'] for x in metrics]
                
                norm_recencies = WeightedScore.normalize_scores(recencies)
                norm_frequencies = WeightedScore.normalize_scores(frequencies)
                norm_sizes = WeightedScore.normalize_scores(sizes)
                norm_frags = WeightedScore.normalize_scores(frags)
                
                # Compute final scores
                scores = []
                for i, x in enumerate(metrics):
                    score = WeightedScore.calculate(
                        recency=norm_recencies[i],
                        frequency=norm_frequencies[i],
                        size=norm_sizes[i],
                        fragmentation=norm_frags[i],
                        config=config
                    )
                    scores.append((x['meta'], score))
                
                # Sort by (priority, score) ascending
                scores.sort(key=lambda x: (x[0].priority, x[1]))
                candidates = [x[0] for x in scores]
            
        blocks_freed = 0
        sequences_evicted = 0
        
        # Batch eviction setting
        target_freed = num_blocks_needed
        if self.eviction_config.batch_eviction and self.eviction_config.batch_size > 1:
            target_freed = max(num_blocks_needed, self.eviction_config.batch_size * 2) # Heuristic

        for meta in candidates:
            # Free the sequence
            blocks_before = self.allocator.num_free
            self.remove_sequence(meta.seq_id)
            blocks_after = self.allocator.num_free
            
            freed = blocks_after - blocks_before
            blocks_freed += freed
            sequences_evicted += 1
            
            # Update stats
            self._eviction_stats.num_evictions += 1
            self._eviction_stats.blocks_evicted += freed
            self._eviction_stats.sequences_evicted += 1
            
            # Track detailed stats
            if self.eviction_config.track_detailed_stats:
                self._detailed_stats['eviction_reasons']['pressure'] = self._detailed_stats['eviction_reasons'].get('pressure', 0) + 1
            
            if blocks_freed >= target_freed:
                break
                    
        return blocks_freed >= num_blocks_needed

    def _update_fragmentation_info(self):
        """Update fragmentation stats."""
        stats = self.get_stats()
        self._fragmentation_info['fragmentation_ratio'] = stats.fragmentation_ratio

    def _detect_workload_pattern(self):
        """Detect workload pattern for adaptive policy."""
        if not self._eviction_metadata:
            return "working_set"
            
        # Analyze access patterns
        total_access = sum(m.access_count for m in self._eviction_metadata.values())
        count = len(self._eviction_metadata)
        avg_access = total_access / count if count > 0 else 0
        
        # Low average access count (< 1.5) implies many items read once (scan)
        if avg_access < 1.5:
            return "scan"
        return "working_set"
        
    def cleanup_expired(self) -> int:
        """Cleanup expired sequences."""
        if not self.eviction_config.enable_ttl:
            return 0
        
        now = time.time()
        expired = []
        for seq_id, meta in self._eviction_metadata.items():
            if meta.ttl and (now - meta.last_access_time > meta.ttl):
                 expired.append(seq_id)
                 
        for seq_id in expired:
            self.remove_sequence(seq_id)
            
        return len(expired)

    def get_eviction_stats(self) -> EvictionStats:
        return self._eviction_stats
        
    def get_detailed_stats(self) -> dict:
        return self._detailed_stats
        
    def get_memory_pressure(self) -> float:
        return 1.0 - (self.allocator.num_free / self.num_blocks)
        
    def do_proactive_eviction(self, target_pressure: float) -> int:
        """Evict until pressure <= target_pressure."""
        current_free = self.allocator.num_free
        target_free = int((1.0 - target_pressure) * self.num_blocks)
        needed = target_free - current_free
        if needed <= 0:
            return 0
            
        start_free = self.allocator.num_free
        self._evict(needed)
        return self.allocator.num_free - start_free
