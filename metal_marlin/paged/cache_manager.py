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
from dataclasses import dataclass
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


@dataclass
class SequenceEvictionMetadata:
    """Eviction metadata for sequences."""
    seq_id: int
    last_access_time: float = 0.0
    access_count: int = 0
    creation_time: float = 0.0


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
    ) -> None:
        self.config = config or KVBlockConfig()
        self.num_blocks = num_blocks
        self.use_multimodal = use_multimodal

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

        self.blocks: list[KVBlock] = [
            KVBlock(config=self.config) for _ in range(num_blocks)
        ]
        
        # Prompt sharing cache
        self._prompt_cache: dict[str, tuple[list[int], list[int]]] = {}
        self._cow_count = 0
        self._prompt_hits = 0
        self._shared_block_count = 0

    def add_sequence(self, seq_id: int) -> bool:
        """Add a new sequence."""
        success = self.page_table.add_sequence(seq_id)
        if success:
            block_idx = self.page_table.sequences[seq_id].block_indices[0]
            self.blocks[block_idx].allocate()

            if self.use_multimodal and isinstance(
                self.allocator, MultimodalBlockAllocator
            ):
                self.allocator.register_sequence(seq_id)

        return success

    def remove_sequence(self, seq_id: int) -> None:
        """Remove a sequence."""
        if self.use_multimodal and isinstance(self.allocator, MultimodalBlockAllocator):
            self.allocator.unregister_sequence(seq_id)
        self.page_table.remove_sequence(seq_id)

    def append_kv(
        self,
        seq_id: int,
        key: NDArray[Any],
        value: NDArray[Any],
    ) -> bool:
        """Append KV with automatic COW on shared blocks."""
        if not self.page_table.has_sequence(seq_id):
            raise ValueError(f"Sequence {seq_id} not registered")

        state = self.page_table.sequences[seq_id]
        
        # Determine which block to write to
        token_pos = state.logical_len
        block_offset = token_pos // self.config.block_size
        
        # Extend blocks if needed
        while block_offset >= len(state.block_indices):
            block_idx = self.allocator.allocate()
            if block_idx is None:
                return False
            state.block_indices.append(block_idx)
            self.blocks[block_idx].allocate()
        
        tail_block_idx = state.block_indices[block_offset]
        tail_block = self.blocks[tail_block_idx]

        # COW if block is shared (refcount > 1)
        if tail_block.ref_count > 1:
            self._cow_count += 1
            new_idx = self.page_table.cow_block(seq_id, block_offset)
            if new_idx is None:
                return False
            # Copy block content to new private copy
            old_block = tail_block
            tail_block = self.blocks[new_idx]
            if old_block._data is not None:
                tail_block.allocate()
                tail_block._data[:] = old_block._data
                tail_block._token_count = old_block._token_count

        tail_block.append_kv(key, value)
        state.logical_len += 1
        return True

    def get_kv(self, seq_id: int) -> tuple[NDArray[Any], NDArray[Any]]:
        """Get all KV for a sequence."""
        if not self.page_table.has_sequence(seq_id):
            raise ValueError(f"Sequence {seq_id} not registered")

        state = self.page_table.sequences[seq_id]
        if state.logical_len == 0:
            return (
                np.zeros((0, self.config.num_heads, self.config.head_dim), dtype=self.config.dtype),
                np.zeros((0, self.config.num_heads, self.config.head_dim), dtype=self.config.dtype),
            )

        all_keys = []
        all_values = []
        for block_idx in state.block_indices:
            block = self.blocks[block_idx]
            if block.token_count > 0:
                k, v = block.get_kv()
                all_keys.append(k)
                all_values.append(v)

        keys = np.concatenate(all_keys, axis=0)[: state.logical_len]
        values = np.concatenate(all_values, axis=0)[: state.logical_len]
        return keys, values

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
            f"seqs={stats.num_sequences}, "
            f"tokens={stats.total_tokens}, "
            f"cow={stats.cow_operations}, "
            f"shared={stats.shared_prompt_blocks})"
        )
