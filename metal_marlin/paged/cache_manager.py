"""Copy-on-Write cache manager for prompt sharing.

This module implements COW (Copy-on-Write) semantics for KV cache prompt sharing:
- Sequences with common prefixes share physical blocks via reference counting
- First write to a shared block triggers automatic COW (copy-on-write)
- Zero-copy sharing until divergence occurs

The COW mechanism is implemented via:
1. BlockAllocator: Manages refcounts and copy_on_write() operations
2. PageTable: Handles fork_sequence() for sharing blocks
3. PagedKVCache: Orchestrates COW operations during append_kv()
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .allocator import BlockAllocator, MultimodalBlockAllocator
from .kv_block import KVBlock, KVBlockConfig
from .page_table import PageTable


def _compute_prompt_hash(tokens: list[int]) -> str:
    """Compute hash for prompt prefix matching."""
    token_bytes = b"".join(t.to_bytes(4, "little") for t in tokens)
    return hashlib.sha256(token_bytes).hexdigest()[:16]


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
    policy: EvictionPolicy = EvictionPolicy.LRU
    max_size: int | None = None
    
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
class SequenceEvictionMetadata:
    """Eviction metadata for sequences."""
    seq_id: int
    last_access_time: float = 0.0
    access_count: int = 0
    creation_time: float = 0.0
    priority: int = 0
    ttl: float | None = None


@dataclass
class EvictionStats:
    """Eviction statistics."""
    num_evictions: int = 0
    blocks_evicted: int = 0
    sequences_evicted: int = 0


@dataclass
class CacheStats:
    """Cache statistics including COW metrics."""
    num_blocks_total: int
    num_blocks_free: int
    num_blocks_allocated: int
    num_sequences: int
    total_tokens: int
    memory_used_bytes: int
    memory_total_bytes: int
    fragmentation_ratio: float
    shared_prompt_blocks: int = 0  # Blocks shared via COW
    cow_operations: int = 0  # Number of COW triggers
    prompt_cache_hits: int = 0  # Prompt prefix cache hits
    evictions: int = 0  # Number of evictions triggered
    sequences_evicted: int = 0  # Number of sequences evicted


class PagedKVCache:
    """Paged KV cache with copy-on-write prompt sharing.
    
    Features:
    - Block-level reference counting for zero-copy sharing
    - Automatic COW on first write to shared blocks
    - Prompt prefix caching for efficient batch inference
    - Per-sequence memory tracking
    
    COW Workflow:
        1. Create parent sequence, fill with tokens
        2. Fork child sequences via fork_sequence() (zero-copy)
        3. Child writes trigger COW automatically in append_kv()
        4. Parent remains unchanged, child gets private copy
    
    Example:
        >>> cache = PagedKVCache(num_blocks=100)
        >>> cache.add_sequence(0)  # Parent
        >>> # ... fill parent with prompt tokens ...
        >>> cache.fork_sequence(0, 1)  # Child shares blocks
        >>> cache.append_kv(1, k, v)  # Triggers COW on shared block
    """

    def __init__(
        self,
        config: KVBlockConfig | None = None,
        num_blocks: int = 1024,
        use_multimodal: bool = False,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        eviction_config: EvictionConfig | None = None,
    ) -> None:
        self.config = config or KVBlockConfig()
        self.num_blocks = num_blocks
        self.use_multimodal = use_multimodal
        
        # Handle eviction configuration
        if eviction_config:
            self.eviction_config = eviction_config
            # Sync policy if config provided
            self.eviction_policy = eviction_config.policy
        else:
            self.eviction_config = EvictionConfig(policy=eviction_policy)
            self.eviction_policy = eviction_policy

        if use_multimodal:
            self.allocator = MultimodalBlockAllocator(
                num_blocks=num_blocks,
                block_size=self.config.block_size,
            )
        else:
            self.allocator = BlockAllocator(num_blocks=num_blocks)

        self.page_table = PageTable(
            allocator=self.allocator,
            block_size=self.config.block_size,
        )

        # Flat block storage
        block_shape = (self.num_blocks, self.config.block_size, self.config.num_heads, self.config.head_dim)
        self.k_blocks = np.zeros(block_shape, dtype=self.config.dtype)
        self.v_blocks = np.zeros(block_shape, dtype=self.config.dtype)
        
        # Prompt sharing cache
        self._prompt_cache: dict[str, tuple[list[int], list[int]]] = {}
        self._cow_count = 0
        self._prompt_hits = 0
        self._shared_block_count = 0
        
        # Eviction tracking
        self._eviction_metadata: dict[int, SequenceEvictionMetadata] = {}
        self._eviction_stats = EvictionStats()
        self._detailed_stats = {
            'avg_score_evicted': 0.0,
            'eviction_reasons': {}
        }
        self._fragmentation_info = {'fragmentation_ratio': 0.0}

    def add_sequence(
        self, 
        seq_id: int, 
        priority: int = 0, 
        ttl: float | None = None
    ) -> bool:
        """Add a new sequence."""
        # Ensure we have at least one block available for the new sequence
        if self.allocator.num_free == 0:
            if not self._evict(num_blocks_needed=1):
                return False

        success = self.page_table.add_sequence(seq_id)
        if success:
            if self.use_multimodal and isinstance(
                self.allocator, MultimodalBlockAllocator
            ):
                self.allocator.register_sequence(seq_id)
            
            # Initialize eviction metadata
            self._eviction_metadata[seq_id] = SequenceEvictionMetadata(
                seq_id=seq_id,
                last_access_time=time.time(),
                access_count=1,
                creation_time=time.time(),
                priority=priority,
                ttl=ttl or (self.eviction_config.default_ttl_seconds if self.eviction_config.enable_ttl else None)
            )

        return success

    def remove_sequence(self, seq_id: int) -> None:
        """Remove a sequence."""
        if self.use_multimodal and isinstance(self.allocator, MultimodalBlockAllocator):
            self.allocator.unregister_sequence(seq_id)
        self.page_table.remove_sequence(seq_id)
        self._eviction_metadata.pop(seq_id, None)

    def append_kv(
        self,
        seq_id: int,
        key: NDArray[Any],
        value: NDArray[Any],
    ) -> bool:
        """Append KV with automatic COW on shared blocks."""
        if not self.page_table.has_sequence(seq_id):
            raise ValueError(f"Sequence {seq_id} not registered")

        # Update access stats
        if seq_id in self._eviction_metadata:
            meta = self._eviction_metadata[seq_id]
            meta.last_access_time = time.time()
            meta.access_count += 1

        state = self.page_table.sequences[seq_id]
        
        # Determine which block to write to
        token_pos = state.logical_len
        block_offset = token_pos // self.config.block_size
        token_offset = token_pos % self.config.block_size
        
        # Extend blocks if needed
        if block_offset >= len(state.block_indices):
            block_idx = self.allocator.allocate()
            if block_idx is None:
                # Try eviction
                if self._evict(num_blocks_needed=1):
                    block_idx = self.allocator.allocate()
                
                if block_idx is None:
                    return False

            state.block_indices.append(block_idx)
        
        tail_block_idx = state.block_indices[block_offset]
        
        # Check refcount for COW
        # Access allocator block state directly
        if self.use_multimodal and isinstance(self.allocator, MultimodalBlockAllocator):
             current_ref_count = self.allocator.blocks[tail_block_idx].ref_count
        else:
             current_ref_count = self.allocator.blocks[tail_block_idx].ref_count

        # COW if block is shared (refcount > 1)
        if current_ref_count > 1:
            self._cow_count += 1
            
            # Check if we have enough blocks for COW (requires 1 new block)
            if self.allocator.num_free == 0:
                if not self._evict(num_blocks_needed=1):
                    return False

            new_idx = self.page_table.cow_block(seq_id, block_offset)
            if new_idx is None:
                return False
                
            # Copy content from old block to new block
            self.k_blocks[new_idx] = self.k_blocks[tail_block_idx].copy()
            self.v_blocks[new_idx] = self.v_blocks[tail_block_idx].copy()
            
            tail_block_idx = new_idx

        # Write data to flat storage
        self.k_blocks[tail_block_idx, token_offset] = key
        self.v_blocks[tail_block_idx, token_offset] = value
        
        state.logical_len += 1
        return True

    def append_kv_batch(
        self,
        seq_id: int,
        keys: NDArray[Any],
        values: NDArray[Any],
    ) -> bool:
        """Append multiple KV pairs with automatic COW on shared blocks.

        Args:
            seq_id: Sequence identifier.
            keys: Key tensor of shape [num_tokens, num_heads, head_dim].
            values: Value tensor of shape [num_tokens, num_heads, head_dim].

        Returns:
            True if successful, False on OOM.
        """
        if not self.page_table.has_sequence(seq_id):
            raise ValueError(f"Sequence {seq_id} not registered")

        # Update access stats
        if seq_id in self._eviction_metadata:
            meta = self._eviction_metadata[seq_id]
            meta.last_access_time = time.time()
            meta.access_count += 1

        state = self.page_table.sequences[seq_id]
        num_tokens = keys.shape[0]
        token_pos = state.logical_len

        # Process tokens block by block
        tokens_remaining = num_tokens
        token_idx = 0

        while tokens_remaining > 0:
            block_offset = token_pos // self.config.block_size
            token_offset = token_pos % self.config.block_size

            # Extend blocks if needed
            if block_offset >= len(state.block_indices):
                block_idx = self.allocator.allocate()
                if block_idx is None:
                    # Try eviction
                    if self._evict(num_blocks_needed=1):
                        block_idx = self.allocator.allocate()
                    
                    if block_idx is None:
                        return False

                state.block_indices.append(block_idx)

            tail_block_idx = state.block_indices[block_offset]

            # Check refcount for COW
            if self.use_multimodal and isinstance(self.allocator, MultimodalBlockAllocator):
                 current_ref_count = self.allocator.blocks[tail_block_idx].ref_count
            else:
                 current_ref_count = self.allocator.blocks[tail_block_idx].ref_count

            # COW if block is shared
            if current_ref_count > 1:
                self._cow_count += 1
                
                # Check if we have enough blocks for COW
                if self.allocator.num_free == 0:
                    if not self._evict(num_blocks_needed=1):
                        return False

                new_idx = self.page_table.cow_block(seq_id, block_offset)
                if new_idx is None:
                    return False
                
                # Copy content
                self.k_blocks[new_idx] = self.k_blocks[tail_block_idx].copy()
                self.v_blocks[new_idx] = self.v_blocks[tail_block_idx].copy()
                
                tail_block_idx = new_idx

            # Calculate how many tokens fit in this block
            space_in_block = self.config.block_size - token_offset
            tokens_to_append = min(tokens_remaining, space_in_block)

            # Append batch slice
            end_idx = token_idx + tokens_to_append
            
            self.k_blocks[tail_block_idx, token_offset : token_offset + tokens_to_append] = keys[token_idx:end_idx]
            self.v_blocks[tail_block_idx, token_offset : token_offset + tokens_to_append] = values[token_idx:end_idx]

            tokens_remaining -= tokens_to_append
            token_idx = end_idx
            token_pos += tokens_to_append
            state.logical_len += tokens_to_append

        return True

    def get_kv(self, seq_id: int) -> tuple[NDArray[Any], NDArray[Any]]:
        """Get all KV for a sequence."""
        if not self.page_table.has_sequence(seq_id):
            raise ValueError(f"Sequence {seq_id} not registered")

        # Update access stats
        if seq_id in self._eviction_metadata:
            meta = self._eviction_metadata[seq_id]
            meta.last_access_time = time.time()
            meta.access_count += 1

        state = self.page_table.sequences[seq_id]
        if state.logical_len == 0:
            return (
                np.zeros((0, self.config.num_heads, self.config.head_dim), dtype=self.config.dtype),
                np.zeros((0, self.config.num_heads, self.config.head_dim), dtype=self.config.dtype),
            )

        # Gather blocks
        # We need to handle partial last block
        # Optimization: preallocate result array? Or use list of arrays.
        # Concatenating list of arrays is usually fine for this.
        
        all_keys = []
        all_values = []
        
        # Calculate full blocks
        num_full_blocks = state.logical_len // self.config.block_size
        remainder = state.logical_len % self.config.block_size
        
        for i, block_idx in enumerate(state.block_indices):
            if i < len(state.block_indices) - 1:
                # Full block
                all_keys.append(self.k_blocks[block_idx])
                all_values.append(self.v_blocks[block_idx])
            else:
                # Last block (might be full or partial)
                valid_len = remainder if remainder > 0 else self.config.block_size
                # If logical_len is exact multiple, remainder is 0 but block is full.
                # Logic: logical_len 16 -> 1 block, full. 16 // 16 = 1. remainder = 0.
                # If blocks=1, i=0. i < 0 is false. Else branch.
                # valid_len needs to be correct.
                # If remainder == 0 and logical_len > 0, it means the last block is full.
                # But wait, if logical_len is exact multiple, do we have an extra empty block?
                # No, append_kv only adds block if needed.
                # So if remainder == 0, the last block is actually full (size 16).
                # UNLESS the sequence is empty (handled at start).
                
                if remainder == 0:
                    valid_len = self.config.block_size
                else:
                    valid_len = remainder
                
                all_keys.append(self.k_blocks[block_idx, :valid_len])
                all_values.append(self.v_blocks[block_idx, :valid_len])

        keys = np.concatenate(all_keys, axis=0)
        values = np.concatenate(all_values, axis=0)
        return keys, values

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

    def get_memory_pressure(self) -> float:
        return 1.0 - (self.allocator.num_free / self.num_blocks)

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

    def _evict(self, num_blocks_needed: int) -> bool:
        """Evict sequences to free up blocks.
        
        Args:
            num_blocks_needed: Minimum number of blocks to free.
            
        Returns:
            True if enough blocks were freed, False otherwise.
        """
        # Optimization: cleanup expired first to free space without forced eviction
        if self.eviction_config.enable_ttl:
            self.cleanup_expired()
            if self.allocator.num_free >= num_blocks_needed:
                return True

        if not self._eviction_metadata:
            return False
            
        # Get candidates for eviction
        candidates = list(self._eviction_metadata.values())
        
        # Sort based on policy
        if self.eviction_policy == EvictionPolicy.LRU:
            candidates.sort(key=lambda m: m.last_access_time)
        elif self.eviction_policy == EvictionPolicy.FIFO:
            candidates.sort(key=lambda m: m.creation_time)
        elif self.eviction_policy == EvictionPolicy.LFU:
            candidates.sort(key=lambda m: m.access_count)
        elif self.eviction_policy == EvictionPolicy.WEIGHTED or self.eviction_policy == EvictionPolicy.ADAPTIVE:
             # Prepare config
            config = self.eviction_config
            pattern = self._detect_workload_pattern()
            
            if self.eviction_policy == EvictionPolicy.ADAPTIVE:
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
                # Priority: higher priority should be evicted LAST (so higher score)
                # But here we want to sort candidates to pick *victims*.
                # Score: Higher score = LESS likely to evict (better to keep).
                # So sort by score ASCENDING (lowest score = worst = evict first).
                # Priority: Higher priority = better to keep.
                # So sort key should include priority.
                # If priority is high, we want it at the END of the list (safe).
                # If score is high, we want it at the END of the list.
                
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
            state = self.page_table.sequences.get(meta.seq_id)
            if state:
                # Count blocks owned by this sequence (approximate, doesn't account for sharing perfectly)
                # But remove_sequence will free blocks, so allocator.num_free will increase.
                # We can check allocator.num_free improvement.
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
                
                if self.eviction_config.track_detailed_stats:
                    self._detailed_stats['eviction_reasons']['pressure'] = self._detailed_stats['eviction_reasons'].get('pressure', 0) + 1

                if blocks_freed >= target_freed:
                    break
                    
        return blocks_freed >= num_blocks_needed

    def fork_sequence(self, src_id: int, dst_id: int) -> bool:
        """Fork sequence with zero-copy COW."""
        success = self.page_table.fork_sequence(src_id, dst_id)
        if success:
            src_state = self.page_table.sequences.get(src_id)
            if src_state:
                self._shared_block_count += len(src_state.block_indices)
            
            if self.use_multimodal and isinstance(
                self.allocator, MultimodalBlockAllocator
            ):
                self.allocator.register_sequence(dst_id)
                
            # Initialize eviction metadata for new sequence
            # Inherit priority/ttl from parent? For now use defaults/parent values
            src_meta = self._eviction_metadata.get(src_id)
            priority = src_meta.priority if src_meta else 0
            ttl = src_meta.ttl if src_meta else None
            
            self._eviction_metadata[dst_id] = SequenceEvictionMetadata(
                seq_id=dst_id,
                last_access_time=time.time(),
                access_count=1,
                creation_time=time.time(),
                priority=priority,
                ttl=ttl
            )
            
            # Update source access stats
            if src_id in self._eviction_metadata:
                self._eviction_metadata[src_id].last_access_time = time.time()
                self._eviction_metadata[src_id].access_count += 1
                
        return success
    
    def add_sequence_with_prompt(
        self,
        seq_id: int,
        prompt_tokens: list[int],
        reuse_prefix: bool = True,
    ) -> bool:
        """Add sequence with prompt prefix sharing."""
        if not reuse_prefix or not prompt_tokens:
            return self.add_sequence(seq_id)
        
        prompt_hash = _compute_prompt_hash(prompt_tokens)
        
        # Check cache for matching prompt
        if prompt_hash in self._prompt_cache:
            cached_blocks, cached_seq_ids = self._prompt_cache[prompt_hash]
            
            valid_src = None
            for src_id in cached_seq_ids:
                if self.page_table.has_sequence(src_id):
                    valid_src = src_id
                    break
            
            if valid_src is not None:
                # Fork from cached sequence (COW sharing)
                if self.fork_sequence(valid_src, seq_id):
                    self._prompt_hits += 1
                    cached_seq_ids.append(seq_id)
                    return True
            else:
                # Cache stale
                del self._prompt_cache[prompt_hash]
        
        # Fresh sequence
        if self.add_sequence(seq_id):
            state = self.page_table.sequences[seq_id]
            self._prompt_cache[prompt_hash] = (
                state.block_indices.copy(),
                [seq_id],
            )
            return True
        
        return False
    
    def share_prompt_blocks(
        self,
        src_seq_id: int,
        dst_seq_ids: list[int],
        num_prefix_tokens: int | None = None,
    ) -> list[bool]:
        """Share prompt prefix across multiple sequences."""
        if not self.page_table.has_sequence(src_seq_id):
            return [False] * len(dst_seq_ids)
        
        src_state = self.page_table.sequences[src_seq_id]
        
        if num_prefix_tokens is None:
            shared_blocks = src_state.block_indices
        else:
            num_shared_blocks = (
                num_prefix_tokens + self.config.block_size - 1
            ) // self.config.block_size
            shared_blocks = src_state.block_indices[:num_shared_blocks]
        
        results = []
        for dst_id in dst_seq_ids:
            success = self.fork_sequence(src_seq_id, dst_id)
            results.append(success)
            
        if any(results):
            self._shared_block_count += len(shared_blocks) * sum(results)
            
        return results
    
    def get_shared_blocks(self, seq_id: int) -> list[int]:
        """Get indices of shared blocks (refcount > 1)."""
        if not self.page_table.has_sequence(seq_id):
            return []
        
        state = self.page_table.sequences[seq_id]
        shared = []
        
        for block_idx in state.block_indices:
            block_state = self.allocator.blocks[block_idx]
            if block_state.ref_count > 1:
                shared.append(block_idx)
                
        return shared

    def get_block_table(self, seq_id: int) -> list[int]:
        """Get block table for a sequence."""
        return self.page_table.get_block_table(seq_id)

    def get_eviction_stats(self) -> EvictionStats:
        """Get eviction statistics."""
        return self._eviction_stats

    def get_detailed_stats(self) -> dict:
        """Get detailed eviction statistics."""
        return self._detailed_stats

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        total_tokens = sum(
            state.logical_len for state in self.page_table.sequences.values()
        )

        total_allocated_blocks = self.allocator.num_allocated
        total_blocks = self.num_blocks
        blocks_free = self.allocator.num_free

        memory_per_block = self.config.memory_bytes
        memory_used = total_allocated_blocks * memory_per_block
        memory_total = total_blocks * memory_per_block

        if total_allocated_blocks > 0:
            ideal_blocks = (total_tokens + self.config.block_size - 1) // self.config.block_size
            fragmentation = ideal_blocks / total_allocated_blocks
        else:
            fragmentation = 1.0

        return CacheStats(
            num_blocks_total=total_blocks,
            num_blocks_free=blocks_free,
            num_blocks_allocated=total_allocated_blocks,
            num_sequences=self.page_table.num_sequences,
            total_tokens=total_tokens,
            memory_used_bytes=memory_used,
            memory_total_bytes=memory_total,
            fragmentation_ratio=fragmentation,
            shared_prompt_blocks=self._shared_block_count,
            cow_operations=self._cow_count,
            prompt_cache_hits=self._prompt_hits,
            evictions=self._eviction_stats.num_evictions,
            sequences_evicted=self._eviction_stats.sequences_evicted,
        )

    def has_sequence(self, seq_id: int) -> bool:
        """Check if sequence registered."""
        return self.page_table.has_sequence(seq_id)

    @property
    def num_sequences(self) -> int:
        """Number of sequences."""
        return self.page_table.num_sequences

    def sequence_ids(self) -> list[int]:
        """Get sequence IDs."""
        return self.page_table.sequence_ids()

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"PagedKVCache("
            f"blocks={stats.num_blocks_allocated}/{stats.num_blocks_total}, "
            f"sequences={stats.num_sequences}, "
            f"tokens={stats.total_tokens}, "
            f"cow={stats.cow_operations}, "
            f"shared={stats.shared_prompt_blocks})"
        )
