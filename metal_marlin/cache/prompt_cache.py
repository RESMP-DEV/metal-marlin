"""Prompt/prefix caching for efficient transformer inference.

Implements automatic prefix caching inspired by vLLM's approach:
1. Hashes token sequences at block granularity for efficient matching
2. Reuses KV cache blocks for common prefixes (system prompts, few-shot examples)
3. Hierarchical storage: GPU (hot) / RAM (warm) / disk (cold)

Common use cases:
- Same system prompt for all requests: ~3-4 blocks cached and reused
- Few-shot examples repeated across requests: shared block chain
- Chat history growing incrementally: prefix blocks already computed

Key design decisions:
- Block-aligned hashing: Match existing PagedAttention block boundaries (16 tokens)
- Content-addressed storage: Same tokens → same hash → same blocks
- Reference counting: Blocks freed only when all users release them
- LRU eviction: Least recently used blocks evicted first under memory pressure

Usage:
    from metal_marlin.cache.prompt_cache import PrefixCache, PrefixCacheConfig

    config = PrefixCacheConfig(
        num_layers=32,
        num_heads=32,
        head_dim=128,
        block_size=16,
        max_cached_blocks=1024,  # 1GB for 32-layer model
    )
    cache = PrefixCache(config)

    # Encode prompt to tokens
    tokens = tokenizer.encode("You are a helpful assistant...")

    # Try to find cached prefix
    match = cache.match_prefix(tokens)
    if match.num_matched_tokens > 0:
        # Reuse cached KV for matched prefix
        kv_blocks = cache.get_blocks(match.block_ids)
        # Only compute KV for remaining tokens
        new_tokens = tokens[match.num_matched_tokens:]
    else:
        # Compute all KV from scratch
        new_tokens = tokens

    # After computing new KV, cache it
    cache.store_prefix(tokens, new_kv_blocks)
"""

from __future__ import annotations

import hashlib
import pickle
import struct
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class StorageTier(Enum):
    """Hierarchical storage tiers for cached blocks."""

    GPU = "gpu"  # Fastest, limited capacity (unified memory on Apple Silicon)
    RAM = "ram"  # Fast, larger capacity
    DISK = "disk"  # Slow, unlimited capacity


@dataclass
class PrefixCacheConfig:
    """Configuration for prefix cache.

    Attributes:
        num_layers: Number of transformer layers.
        num_heads: Number of KV heads (for GQA, use num_kv_heads).
        head_dim: Dimension per head.
        block_size: Tokens per cache block (should match PagedAttention block size).
        max_gpu_blocks: Maximum blocks to keep in GPU memory (hot tier).
        max_ram_blocks: Maximum blocks to keep in RAM (warm tier).
        disk_cache_path: Path for disk cache. None disables disk caching.
        eviction_policy: "lru" (default), "lfu", or "fifo".
        hash_algorithm: Hash algorithm for prefix matching ("xxhash", "md5", "sha256").
        enable_metrics: Track hit/miss statistics.
    """

    num_layers: int
    num_heads: int
    head_dim: int
    block_size: int = 16
    max_gpu_blocks: int = 512  # ~500MB for typical 32-layer model
    max_ram_blocks: int = 2048  # ~2GB in RAM
    disk_cache_path: Path | None = None
    eviction_policy: str = "lru"
    hash_algorithm: str = "xxhash"  # Fast hashing, falls back to md5
    enable_metrics: bool = True

    @property
    def bytes_per_block(self) -> int:
        """Memory footprint of one block across all layers (K+V)."""
        # 2 for K+V, 2 bytes per element (fp16/bf16)
        return 2 * self.block_size * self.num_heads * self.head_dim * 2 * self.num_layers

    @property
    def max_gpu_bytes(self) -> int:
        """Maximum GPU memory for cached blocks."""
        return self.max_gpu_blocks * self.bytes_per_block

    @property
    def max_ram_bytes(self) -> int:
        """Maximum RAM for cached blocks."""
        return self.max_ram_blocks * self.bytes_per_block


@dataclass
class PrefixMatch:
    """Result of prefix matching.

    Attributes:
        num_matched_tokens: Number of tokens matched from the start.
        num_matched_blocks: Number of complete blocks matched.
        block_hashes: Hashes of matched blocks (for retrieval).
        partial_tokens: Tokens in the last partial block (if any).
        cache_tier: Where the matched blocks are stored.
    """

    num_matched_tokens: int
    num_matched_blocks: int
    block_hashes: list[str]
    partial_tokens: int = 0
    cache_tier: StorageTier = StorageTier.GPU


@dataclass
class CachedBlock:
    """Metadata and storage for a cached KV block.

    The actual KV data is stored per-layer as a list of arrays.
    Shape per layer: [2, block_size, num_heads, head_dim] (K=0, V=1).
    """

    block_hash: str
    token_count: int  # Actual tokens stored (may be < block_size for partial)
    ref_count: int = 1
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    tier: StorageTier = StorageTier.GPU
    # KV data: list of arrays per layer, or None if on disk
    kv_data: list[Any] | None = None
    # For disk storage
    disk_path: Path | None = None

    def touch(self) -> None:
        """Update access time and count."""
        self.last_access = time.time()
        self.access_count += 1


@dataclass
class CacheMetrics:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    partial_hits: int = 0  # Prefix partially matched
    evictions: int = 0
    gpu_to_ram_moves: int = 0
    ram_to_disk_moves: int = 0
    disk_loads: int = 0
    bytes_saved: int = 0  # Estimated bytes not recomputed

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self.hits + self.misses + self.partial_hits
        return self.hits / total if total > 0 else 0.0

    def reset(self) -> None:
        """Reset all metrics."""
        self.hits = 0
        self.misses = 0
        self.partial_hits = 0
        self.evictions = 0
        self.gpu_to_ram_moves = 0
        self.ram_to_disk_moves = 0
        self.disk_loads = 0
        self.bytes_saved = 0


# Try to import xxhash for fast hashing, fall back to hashlib
try:
    import xxhash

    HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False


def hash_tokens(tokens: Sequence[int], algorithm: str = "xxhash") -> str:
    """Hash a sequence of tokens to a content-addressed identifier.

    Uses xxhash for speed when available, falls back to MD5.

    Args:
        tokens: Sequence of token IDs.
        algorithm: "xxhash" (default, fast), "md5", or "sha256".

    Returns:
        Hex digest string uniquely identifying this token sequence.
    """
    # Pack tokens as bytes for hashing
    token_bytes = struct.pack(f">{len(tokens)}I", *tokens)

    if algorithm == "xxhash" and HAS_XXHASH:
        return xxhash.xxh64(token_bytes).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(token_bytes).hexdigest()
    else:
        # Default to MD5 (fast, collision-resistant enough for cache keys)
        return hashlib.md5(token_bytes).hexdigest()


def hash_prefix(tokens: Sequence[int], block_size: int, algorithm: str = "xxhash") -> list[str]:
    """Hash tokens at block boundaries for prefix matching.

    This is the key operation for efficient prefix caching: tokens are
    hashed in block-aligned chunks so that common prefixes produce the
    same hash chain regardless of what follows.

    Args:
        tokens: Full token sequence.
        block_size: Number of tokens per block.
        algorithm: Hash algorithm to use.

    Returns:
        List of hashes, one per complete block. The hash of block N
        depends on blocks 0..N (chain hashing for uniqueness).
    """
    hashes = []
    num_blocks = len(tokens) // block_size

    # Chain hashing: each block's hash includes previous blocks' hash
    # This ensures [A, B] and [A, C] have same hash for block 0 but differ for block 1
    prev_hash = ""
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        block_tokens = tokens[start:end]

        # Include previous hash in the input for chain uniqueness
        if prev_hash:
            # Combine previous hash + block tokens
            combined = prev_hash.encode() + struct.pack(f">{len(block_tokens)}I", *block_tokens)
            if algorithm == "xxhash" and HAS_XXHASH:
                block_hash = xxhash.xxh64(combined).hexdigest()
            elif algorithm == "sha256":
                block_hash = hashlib.sha256(combined).hexdigest()
            else:
                block_hash = hashlib.md5(combined).hexdigest()
        else:
            block_hash = hash_tokens(block_tokens, algorithm)

        hashes.append(block_hash)
        prev_hash = block_hash

    return hashes


class PrefixCache:
    """Hierarchical prefix cache with automatic block management.

    Implements content-addressed storage of KV cache blocks for prefix reuse.
    Supports three-tier storage (GPU/RAM/disk) with automatic eviction.

    Thread-safe for concurrent access.
    """

    def __init__(self, config: PrefixCacheConfig) -> None:
        self.config = config
        self._lock = threading.RLock()

        # Block storage by tier (hash -> CachedBlock)
        # Using OrderedDict for LRU ordering
        self._gpu_blocks: OrderedDict[str, CachedBlock] = OrderedDict()
        self._ram_blocks: OrderedDict[str, CachedBlock] = OrderedDict()
        self._disk_index: dict[str, CachedBlock] = {}  # Metadata only, data on disk

        # Metrics
        self.metrics = CacheMetrics()

        # Ensure disk cache directory exists
        if config.disk_cache_path:
            config.disk_cache_path.mkdir(parents=True, exist_ok=True)

    def match_prefix(self, tokens: Sequence[int]) -> PrefixMatch:
        """Find the longest cached prefix for a token sequence.

        This is the main entry point for cache lookups. It returns information
        about how many tokens can be reused from cache.

        Args:
            tokens: Full token sequence to match.

        Returns:
            PrefixMatch describing the longest matched prefix.
        """
        if len(tokens) < self.config.block_size:
            # Too short for any complete block
            if self.config.enable_metrics:
                self.metrics.misses += 1
            return PrefixMatch(
                num_matched_tokens=0,
                num_matched_blocks=0,
                block_hashes=[],
            )

        # Hash tokens at block boundaries
        block_hashes = hash_prefix(tokens, self.config.block_size, self.config.hash_algorithm)

        # Find longest matching prefix
        matched_hashes = []
        match_tier = StorageTier.GPU

        with self._lock:
            for block_hash in block_hashes:
                # Check each tier in order
                if block_hash in self._gpu_blocks:
                    matched_hashes.append(block_hash)
                    self._gpu_blocks[block_hash].touch()
                    # Move to end for LRU
                    self._gpu_blocks.move_to_end(block_hash)
                elif block_hash in self._ram_blocks:
                    matched_hashes.append(block_hash)
                    self._ram_blocks[block_hash].touch()
                    self._ram_blocks.move_to_end(block_hash)
                    match_tier = StorageTier.RAM
                elif block_hash in self._disk_index:
                    matched_hashes.append(block_hash)
                    self._disk_index[block_hash].touch()
                    match_tier = StorageTier.DISK
                else:
                    # Prefix chain broken
                    break

        num_matched = len(matched_hashes)

        # Update metrics
        if self.config.enable_metrics:
            if num_matched == 0:
                self.metrics.misses += 1
            elif num_matched == len(block_hashes):
                self.metrics.hits += 1
                self.metrics.bytes_saved += num_matched * self.config.bytes_per_block
            else:
                self.metrics.partial_hits += 1
                self.metrics.bytes_saved += num_matched * self.config.bytes_per_block

        return PrefixMatch(
            num_matched_tokens=num_matched * self.config.block_size,
            num_matched_blocks=num_matched,
            block_hashes=matched_hashes,
            partial_tokens=len(tokens) % self.config.block_size,
            cache_tier=match_tier,
        )

    def get_blocks(
        self,
        block_hashes: list[str],
        promote_to_gpu: bool = True,
    ) -> list[list[Any]]:
        """Retrieve KV data for cached blocks.

        Args:
            block_hashes: List of block hashes to retrieve.
            promote_to_gpu: If True, promote RAM/disk blocks to GPU tier.

        Returns:
            List of KV data per block. Each block is a list of arrays
            (one per layer), shape [2, block_size, num_heads, head_dim].
        """
        result = []

        with self._lock:
            for block_hash in block_hashes:
                block = self._get_block(block_hash)
                if block is None:
                    raise KeyError(f"Block {block_hash} not found in cache")

                # Load data if on disk
                if block.tier == StorageTier.DISK:
                    self._load_from_disk(block)
                    if self.config.enable_metrics:
                        self.metrics.disk_loads += 1

                # Promote to GPU if requested
                if promote_to_gpu and block.tier != StorageTier.GPU:
                    self._promote_to_gpu(block)

                result.append(block.kv_data)
                block.touch()

        return result

    def store_prefix(
        self,
        tokens: Sequence[int],
        kv_blocks: list[list[Any]],
        start_block: int = 0,
    ) -> list[str]:
        """Store KV blocks for a token prefix.

        Args:
            tokens: Token sequence that produced this KV.
            kv_blocks: List of KV data per block. Each block is a list
                of arrays (one per layer).
            start_block: Which block index to start storing at (for
                incremental caching when prefix was partially matched).

        Returns:
            List of block hashes for the stored blocks.
        """
        # Compute hashes for all blocks
        all_hashes = hash_prefix(tokens, self.config.block_size, self.config.hash_algorithm)

        # Only store new blocks
        hashes_to_store = all_hashes[start_block : start_block + len(kv_blocks)]

        with self._lock:
            for i, block_hash in enumerate(hashes_to_store):
                # Skip if already cached
                if self._has_block(block_hash):
                    # Increment ref count
                    block = self._get_block(block_hash)
                    if block:
                        block.ref_count += 1
                    continue

                # Create new cached block
                block = CachedBlock(
                    block_hash=block_hash,
                    token_count=self.config.block_size,
                    kv_data=kv_blocks[i],
                    tier=StorageTier.GPU,
                )

                # Add to GPU tier (evict if necessary)
                self._add_to_gpu(block)

        return hashes_to_store

    def extend_from_cache(
        self,
        prefix_kv: list[list[Any]],
        new_kv: list[list[Any]],
        concat_fn: Callable[[Any, Any], Any] | None = None,
    ) -> list[list[Any]]:
        """Concatenate cached prefix KV with newly computed KV.

        This is the main entry point for using cached prefixes in inference.
        It efficiently combines cached and new KV tensors.

        Args:
            prefix_kv: Cached KV blocks from get_blocks().
            new_kv: Newly computed KV for tokens after the prefix.
            concat_fn: Optional function to concatenate two arrays.
                If None, uses MLX or numpy concatenation.

        Returns:
            Combined KV for the full sequence. Structure matches input:
            list of arrays per layer, shape [2, seq_len, num_heads, head_dim].
        """
        if not prefix_kv:
            return new_kv

        if concat_fn is None:
            concat_fn = lambda a, b: np.concatenate([a, b], axis=1)  # noqa: E731

        # Concatenate block-wise first (within prefix)
        num_layers = len(prefix_kv[0])
        combined_prefix = []

        for layer_idx in range(num_layers):
            layer_kv_list = [block[layer_idx] for block in prefix_kv]
            if len(layer_kv_list) == 1:
                combined_prefix.append(layer_kv_list[0])
            else:
                # Concatenate all prefix blocks for this layer
                combined = layer_kv_list[0]
                for block_kv in layer_kv_list[1:]:
                    combined = concat_fn(combined, block_kv)
                combined_prefix.append(combined)

        # Now concatenate prefix with new KV
        result = []
        for layer_idx in range(num_layers):
            if new_kv and new_kv[layer_idx] is not None:
                result.append(concat_fn(combined_prefix[layer_idx], new_kv[layer_idx]))
            else:
                result.append(combined_prefix[layer_idx])

        return result

    def release_blocks(self, block_hashes: list[str]) -> None:
        """Decrement reference counts for blocks.

        Call this when a sequence using these blocks is complete.
        Blocks with ref_count 0 become eligible for eviction.
        """
        with self._lock:
            for block_hash in block_hashes:
                block = self._get_block(block_hash)
                if block:
                    block.ref_count = max(0, block.ref_count - 1)

    def clear(self) -> None:
        """Clear all cached blocks."""
        with self._lock:
            self._gpu_blocks.clear()
            self._ram_blocks.clear()
            self._disk_index.clear()

            # Clear disk cache
            if self.config.disk_cache_path:
                for f in self.config.disk_cache_path.glob("*.pkl"):
                    f.unlink()

    def memory_usage(self) -> dict[str, int]:
        """Get current memory usage per tier in bytes."""
        with self._lock:
            return {
                "gpu": len(self._gpu_blocks) * self.config.bytes_per_block,
                "ram": len(self._ram_blocks) * self.config.bytes_per_block,
                "disk_blocks": len(self._disk_index),
            }

    # Internal methods

    def _has_block(self, block_hash: str) -> bool:
        """Check if block exists in any tier."""
        return (
            block_hash in self._gpu_blocks
            or block_hash in self._ram_blocks
            or block_hash in self._disk_index
        )

    def _get_block(self, block_hash: str) -> CachedBlock | None:
        """Get block from any tier."""
        if block_hash in self._gpu_blocks:
            return self._gpu_blocks[block_hash]
        if block_hash in self._ram_blocks:
            return self._ram_blocks[block_hash]
        if block_hash in self._disk_index:
            return self._disk_index[block_hash]
        return None

    def _add_to_gpu(self, block: CachedBlock) -> None:
        """Add block to GPU tier, evicting if necessary."""
        # Evict if at capacity
        while len(self._gpu_blocks) >= self.config.max_gpu_blocks:
            self._evict_from_gpu()

        self._gpu_blocks[block.block_hash] = block
        block.tier = StorageTier.GPU

    def _evict_from_gpu(self) -> None:
        """Evict least recently used block from GPU to RAM."""
        if not self._gpu_blocks:
            return

        # Find LRU block with ref_count == 0
        evict_hash = None
        for block_hash, block in self._gpu_blocks.items():
            if block.ref_count == 0:
                evict_hash = block_hash
                break

        if evict_hash is None:
            # All blocks in use, evict oldest anyway (with warning)
            evict_hash = next(iter(self._gpu_blocks))

        block = self._gpu_blocks.pop(evict_hash)
        block.tier = StorageTier.RAM

        # Add to RAM tier (may cascade eviction)
        while len(self._ram_blocks) >= self.config.max_ram_blocks:
            self._evict_from_ram()

        self._ram_blocks[block.block_hash] = block

        if self.config.enable_metrics:
            self.metrics.evictions += 1
            self.metrics.gpu_to_ram_moves += 1

    def _evict_from_ram(self) -> None:
        """Evict least recently used block from RAM to disk."""
        if not self._ram_blocks:
            return

        # Find LRU block with ref_count == 0
        evict_hash = None
        for block_hash, block in self._ram_blocks.items():
            if block.ref_count == 0:
                evict_hash = block_hash
                break

        if evict_hash is None:
            evict_hash = next(iter(self._ram_blocks))

        block = self._ram_blocks.pop(evict_hash)

        if self.config.disk_cache_path:
            # Save to disk
            self._save_to_disk(block)
            self._disk_index[block.block_hash] = block
            block.kv_data = None  # Free memory
        # else: just drop it

        if self.config.enable_metrics:
            self.metrics.evictions += 1
            self.metrics.ram_to_disk_moves += 1

    def _promote_to_gpu(self, block: CachedBlock) -> None:
        """Promote block from RAM/disk to GPU tier."""
        # Remove from current tier
        if block.block_hash in self._ram_blocks:
            del self._ram_blocks[block.block_hash]
        elif block.block_hash in self._disk_index:
            del self._disk_index[block.block_hash]

        # Add to GPU
        self._add_to_gpu(block)

    def _save_to_disk(self, block: CachedBlock) -> None:
        """Save block data to disk."""
        if not self.config.disk_cache_path:
            return

        disk_path = self.config.disk_cache_path / f"{block.block_hash}.pkl"
        block.disk_path = disk_path
        block.tier = StorageTier.DISK

        # Save numpy arrays
        data_to_save = block.kv_data if block.kv_data else []

        with open(disk_path, "wb") as f:
            pickle.dump(data_to_save, f)

    def _load_from_disk(self, block: CachedBlock) -> None:
        """Load block data from disk."""
        if block.disk_path is None or not block.disk_path.exists():
            raise ValueError(f"Disk cache file not found for block {block.block_hash}")

        with open(block.disk_path, "rb") as f:
            block.kv_data = pickle.load(f)  # noqa: S301 - trusted cache file

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"PrefixCache("
                f"gpu={len(self._gpu_blocks)}/{self.config.max_gpu_blocks}, "
                f"ram={len(self._ram_blocks)}/{self.config.max_ram_blocks}, "
                f"disk={len(self._disk_index)}, "
                f"hit_rate={self.metrics.hit_rate:.2%})"
            )


class RadixPrefixCache(PrefixCache):
    """Prefix cache with radix tree for O(log n) prefix matching.

    Uses a radix tree (compressed trie) indexed by token blocks to enable
    faster prefix matching for large cache sizes. The standard PrefixCache
    uses O(n) linear hash chain matching; this uses O(log n) tree traversal.

    Particularly beneficial when:
    - Many distinct prefixes are cached (>1000 blocks)
    - Prefixes have significant shared structure
    - Frequent partial matches expected
    """

    def __init__(self, config: PrefixCacheConfig) -> None:
        super().__init__(config)
        self._radix_root: dict[str, Any] = {}  # Token block hash -> subtree

    def match_prefix(self, tokens: Sequence[int]) -> PrefixMatch:
        """Find longest cached prefix using radix tree traversal."""
        if len(tokens) < self.config.block_size:
            if self.config.enable_metrics:
                self.metrics.misses += 1
            return PrefixMatch(
                num_matched_tokens=0,
                num_matched_blocks=0,
                block_hashes=[],
            )

        block_hashes = hash_prefix(tokens, self.config.block_size, self.config.hash_algorithm)
        matched_hashes = []
        match_tier = StorageTier.GPU

        with self._lock:
            current = self._radix_root
            for block_hash in block_hashes:
                if block_hash not in current:
                    break

                # Found match, descend tree
                node = current[block_hash]
                matched_hashes.append(block_hash)

                # Update LRU in storage tiers
                block = self._get_block(block_hash)
                if block:
                    block.touch()
                    if block.tier == StorageTier.GPU and block_hash in self._gpu_blocks:
                        self._gpu_blocks.move_to_end(block_hash)
                    elif block.tier == StorageTier.RAM and block_hash in self._ram_blocks:
                        self._ram_blocks.move_to_end(block_hash)
                    match_tier = block.tier

                # Move to subtree for next block
                current = node.get("_children", {})

        num_matched = len(matched_hashes)

        if self.config.enable_metrics:
            if num_matched == 0:
                self.metrics.misses += 1
            elif num_matched == len(block_hashes):
                self.metrics.hits += 1
                self.metrics.bytes_saved += num_matched * self.config.bytes_per_block
            else:
                self.metrics.partial_hits += 1
                self.metrics.bytes_saved += num_matched * self.config.bytes_per_block

        return PrefixMatch(
            num_matched_tokens=num_matched * self.config.block_size,
            num_matched_blocks=num_matched,
            block_hashes=matched_hashes,
            partial_tokens=len(tokens) % self.config.block_size,
            cache_tier=match_tier,
        )

    def store_prefix(
        self,
        tokens: Sequence[int],
        kv_blocks: list[list[Any]],
        start_block: int = 0,
    ) -> list[str]:
        """Store KV blocks and update radix tree."""
        all_hashes = hash_prefix(tokens, self.config.block_size, self.config.hash_algorithm)
        hashes_to_store = all_hashes[start_block : start_block + len(kv_blocks)]

        with self._lock:
            # Build radix tree path
            current = self._radix_root
            for i, block_hash in enumerate(all_hashes[:start_block]):
                if block_hash not in current:
                    # Path should exist for already-cached prefix
                    current[block_hash] = {"_children": {}}
                current = current[block_hash].get("_children", {})

            # Add new blocks
            for i, block_hash in enumerate(hashes_to_store):
                if block_hash not in current:
                    current[block_hash] = {"_children": {}}

                # Store actual data if not already cached
                if not self._has_block(block_hash):
                    block = CachedBlock(
                        block_hash=block_hash,
                        token_count=self.config.block_size,
                        kv_data=kv_blocks[i],
                        tier=StorageTier.GPU,
                    )
                    self._add_to_gpu(block)
                else:
                    block = self._get_block(block_hash)
                    if block:
                        block.ref_count += 1

                current = current[block_hash]["_children"]

        return hashes_to_store

    def clear(self) -> None:
        """Clear all cached blocks and radix tree."""
        super().clear()
        with self._lock:
            self._radix_root.clear()
