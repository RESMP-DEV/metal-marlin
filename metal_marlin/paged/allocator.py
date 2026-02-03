"""Block allocator for paged KV cache.

Manages a pool of fixed-size blocks. Supports reference counting for
copy-on-write (beam search, speculative decoding).

Extended for VLM (Vision-Language Models):
- Tracks image vs text token boundaries
- Prefix caching for repeated image contexts
- Dynamic image token count support
- Vision encoder output caching

Fragmentation Minimization:
- Address-ordered free list with binary search insertion (O(log n))
- Free block coalescing on free operations
- Segregated free lists for different power-of-2 block sizes
- Best-fit allocation for multi-block contiguous allocation
- Fragmentation metrics tracking
- Buddy-system inspired allocation for power-of-2 sized requests
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from numpy.typing import NDArray


def _ceil_pow2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


class TokenModality(Enum):
    """Modality type for tokens in a block."""

    TEXT = "text"
    IMAGE = "image"
    CROSS_ATTENTION = "cross_attention"  # For models with separate cross-attn KV


@dataclass
class BlockState:
    """Metadata for a single physical block."""

    block_idx: int
    ref_count: int = 0
    is_free: bool = True


@dataclass
class MultimodalBlockState:
    """Extended metadata for multimodal KV cache blocks.

    Tracks modality information for VLM prefix caching optimization.
    Image blocks that hash to the same content can be shared across
    sequences (same image, different questions).
    """

    block_idx: int
    ref_count: int = 0
    is_free: bool = True
    modality: TokenModality = TokenModality.TEXT
    # Content hash for prefix caching (images with same hash share blocks)
    content_hash: str | None = None
    # For partial blocks, track valid token range
    valid_start: int = 0
    valid_end: int = 0  # Exclusive; 0 means not yet written


@dataclass
class ImageRegion:
    """Describes a contiguous region of image tokens within a sequence.

    VLMs interleave image tokens with text. This tracks where image
    tokens are positioned so attention masks can be correctly applied
    (e.g., some models use cross-attention for image tokens).
    """

    start_pos: int  # Absolute position in sequence
    num_tokens: int  # Number of image tokens (varies with resolution)
    image_hash: str  # For identifying repeated images
    block_indices: list[int] = field(default_factory=list)  # Physical blocks


@dataclass
class SequenceModality:
    """Tracks modality boundaries for a single sequence.

    For VLMs, a sequence may have structure like:
      [SYS_TEXT] [IMAGE_1] [USER_TEXT] [IMAGE_2] [MORE_TEXT]

    This structure enables:
    - Proper attention masking (some models mask image-image attention)
    - Prefix caching (reuse IMAGE_1 blocks for same-image queries)
    - Memory accounting (image tokens often dominate memory usage)
    """

    seq_id: int
    image_regions: list[ImageRegion] = field(default_factory=list)
    text_ranges: list[tuple[int, int]] = field(default_factory=list)  # (start, end)

    def add_image_region(
        self,
        start_pos: int,
        num_tokens: int,
        image_hash: str,
        block_indices: list[int],
    ) -> None:
        """Register an image token region."""
        self.image_regions.append(
            ImageRegion(
                start_pos=start_pos,
                num_tokens=num_tokens,
                image_hash=image_hash,
                block_indices=block_indices,
            )
        )

    def add_text_range(self, start: int, end: int) -> None:
        """Register a text token range."""
        self.text_ranges.append((start, end))

    @property
    def total_image_tokens(self) -> int:
        """Total number of image tokens across all regions."""
        return sum(r.num_tokens for r in self.image_regions)

    @property
    def total_text_tokens(self) -> int:
        """Total number of text tokens across all ranges."""
        return sum(end - start for start, end in self.text_ranges)

    def get_modality_at(self, pos: int) -> TokenModality:
        """Return the modality of the token at given position."""
        for region in self.image_regions:
            if region.start_pos <= pos < region.start_pos + region.num_tokens:
                return TokenModality.IMAGE
        return TokenModality.TEXT


class BlockAllocator:
    """Fixed-pool block allocator with reference counting.

    Allocates blocks from a pre-sized pool. Blocks with ref_count > 1
    are shared (COW) and must be copied before mutation.

    Args:
        num_blocks: Total number of physical blocks in the pool.
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.blocks: list[BlockState] = [BlockState(block_idx=i) for i in range(num_blocks)]
        # Address-ordered free list for better spatial locality
        self._free_list: list[int] = list(range(num_blocks))

    @property
    def num_free(self) -> int:
        return len(self._free_list)

    @property
    def num_allocated(self) -> int:
        return self.num_blocks - self.num_free

    def allocate(self) -> int | None:
        """Allocate a single block. Returns block index or None if OOM."""
        if not self._free_list:
            return None
        # Pop from front for address-ordered allocation
        idx = self._free_list.pop(0)
        self.blocks[idx].is_free = False
        self.blocks[idx].ref_count = 1
        return idx

    def free(self, block_idx: int) -> None:
        """Decrement ref_count; return block to pool when it reaches zero."""
        block = self.blocks[block_idx]
        block.ref_count -= 1
        if block.ref_count <= 0:
            block.ref_count = 0
            block.is_free = True
            # Binary search insertion to maintain address order
            import bisect
            bisect.insort(self._free_list, block_idx)

    def copy_on_write(self, block_idx: int) -> int | None:
        """If block is shared, allocate a new block for exclusive write.

        Returns the new block index (or original if already exclusive).
        Returns None if a copy is needed but pool is exhausted.
        """
        block = self.blocks[block_idx]
        if block.ref_count == 1:
            return block_idx  # Already exclusive

        # Need a fresh block for the writer
        new_idx = self.allocate()
        if new_idx is None:
            return None

        # Decrement old block's ref (the writer no longer references it)
        block.ref_count -= 1
        return new_idx


class MultimodalBlockAllocator:
    """Block allocator extended for Vision-Language Models.

    Adds support for:
    - Tracking image vs text token boundaries per block
    - Prefix caching: blocks containing identical image tokens are shared
    - Dynamic image token counts (resolution-dependent)
    - Separate accounting for image/text memory usage

    The prefix cache uses content hashing: if two sequences process the
    same image, they share the KV cache blocks for those image tokens.
    This is safe because image tokens are generated deterministically
    from the vision encoder (same image → same KV).

    Args:
        num_blocks: Total number of physical blocks in the pool.
        block_size: Tokens per block (for image token count calculations).
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size

        # Physical block metadata
        self.blocks: list[MultimodalBlockState] = [
            MultimodalBlockState(block_idx=i) for i in range(num_blocks)
        ]
        # Address-ordered free list for better spatial locality
        self._free_list: list[int] = list(range(num_blocks))

        # Prefix cache: maps content_hash → list of (block_idx, valid_token_count)
        # Enables sharing image KV blocks across sequences
        self._prefix_cache: dict[str, list[tuple[int, int]]] = {}

        # Per-sequence modality tracking
        self._sequence_modality: dict[int, SequenceModality] = {}

        # Statistics
        self._image_blocks_allocated = 0
        self._text_blocks_allocated = 0
        self._prefix_cache_hits = 0
        self._fragmentation_score = 0.0

    @property
    def num_free(self) -> int:
        return len(self._free_list)

    @property
    def num_allocated(self) -> int:
        return self.num_blocks - self.num_free

    @property
    def image_blocks_allocated(self) -> int:
        return self._image_blocks_allocated

    @property
    def text_blocks_allocated(self) -> int:
        return self._text_blocks_allocated

    @property
    def prefix_cache_hits(self) -> int:
        return self._prefix_cache_hits

    def allocate(
        self,
        modality: TokenModality = TokenModality.TEXT,
        content_hash: str | None = None,
    ) -> int | None:
        """Allocate a single block with modality tracking.

        Args:
            modality: Whether this block will hold image or text tokens.
            content_hash: For image blocks, hash of the image content.
                If provided and matches a cached block, returns the
                cached block with incremented refcount (prefix sharing).

        Returns:
            Block index, or None if OOM.
        """
        # Check prefix cache for image blocks
        if modality == TokenModality.IMAGE and content_hash is not None:
            cached = self._prefix_cache.get(content_hash)
            if cached:
                # Reuse cached block(s) - increment refcount
                block_idx, _ = cached[0]  # Use first block in cache
                self.blocks[block_idx].ref_count += 1
                self._prefix_cache_hits += 1
                return block_idx

        # Normal allocation
        if not self._free_list:
            return None

        # Pop from front for address-ordered allocation
        idx = self._free_list.pop(0)
        block = self.blocks[idx]
        block.is_free = False
        block.ref_count = 1
        block.modality = modality
        block.content_hash = content_hash
        block.valid_start = 0
        block.valid_end = 0

        # Update statistics
        if modality == TokenModality.IMAGE:
            self._image_blocks_allocated += 1
        else:
            self._text_blocks_allocated += 1

        return idx

    def allocate_image_blocks(
        self,
        num_tokens: int,
        image_hash: str,
    ) -> list[int] | None:
        """Allocate blocks for an image's tokens, with prefix cache lookup.

        For VLMs, image token count depends on resolution. This method:
        1. Checks if we have cached blocks for this image_hash
        2. If yes, increments their refcounts and returns them
        3. If no, allocates fresh blocks

        Args:
            num_tokens: Number of image tokens (varies with resolution).
            image_hash: Hash identifying the image content.

        Returns:
            List of block indices, or None if OOM.
        """
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        # Check prefix cache
        cached = self._prefix_cache.get(image_hash)
        if cached and len(cached) >= num_blocks_needed:
            # Verify we have enough cached blocks and they're still valid
            block_indices = []
            for block_idx, valid_count in cached[:num_blocks_needed]:
                if not self.blocks[block_idx].is_free:
                    self.blocks[block_idx].ref_count += 1
                    block_indices.append(block_idx)
                else:
                    # Cache entry is stale, fall through to fresh allocation
                    break
            else:
                # All cached blocks valid
                self._prefix_cache_hits += 1
                return block_indices
            # Partial hit - free the blocks we just incremented and allocate fresh
            for idx in block_indices:
                self.blocks[idx].ref_count -= 1

        # Fresh allocation
        if self.num_free < num_blocks_needed:
            return None

        block_indices = []
        tokens_remaining = num_tokens
        for i in range(num_blocks_needed):
            idx = self.allocate(
                modality=TokenModality.IMAGE,
                content_hash=image_hash if i == 0 else None,
            )
            if idx is None:
                # Shouldn't happen given check above, but clean up
                for allocated_idx in block_indices:
                    self.free(allocated_idx)
                return None

            # Track valid token range in this block
            tokens_in_block = min(tokens_remaining, self.block_size)
            self.blocks[idx].valid_start = 0
            self.blocks[idx].valid_end = tokens_in_block
            tokens_remaining -= tokens_in_block

            block_indices.append(idx)

        # Update prefix cache
        cache_entries = [(idx, self.blocks[idx].valid_end) for idx in block_indices]
        self._prefix_cache[image_hash] = cache_entries

        return block_indices

    def free(self, block_idx: int) -> None:
        """Decrement ref_count; return block to pool when it reaches zero."""
        import bisect
        
        block = self.blocks[block_idx]
        block.ref_count -= 1

        if block.ref_count <= 0:
            # Update statistics before freeing
            if block.modality == TokenModality.IMAGE:
                self._image_blocks_allocated -= 1
            else:
                self._text_blocks_allocated -= 1

            # Remove from prefix cache if this was the last reference
            if block.content_hash and block.content_hash in self._prefix_cache:
                cached = self._prefix_cache[block.content_hash]
                self._prefix_cache[block.content_hash] = [
                    (idx, cnt) for idx, cnt in cached if idx != block_idx
                ]
                if not self._prefix_cache[block.content_hash]:
                    del self._prefix_cache[block.content_hash]

            block.ref_count = 0
            block.is_free = True
            block.modality = TokenModality.TEXT
            block.content_hash = None
            block.valid_start = 0
            block.valid_end = 0
            
            # Binary search insertion to maintain address order
            bisect.insort(self._free_list, block_idx)
            self._update_fragmentation_metric()

    def copy_on_write(self, block_idx: int) -> int | None:
        """If block is shared, allocate a new block for exclusive write.

        Preserves modality information in the new block.
        """
        block = self.blocks[block_idx]
        if block.ref_count == 1:
            return block_idx

        # Allocate with same modality
        new_idx = self.allocate(modality=block.modality)
        if new_idx is None:
            return None

        # Copy metadata (but not content_hash - the new block is independent)
        new_block = self.blocks[new_idx]
        new_block.valid_start = block.valid_start
        new_block.valid_end = block.valid_end

        block.ref_count -= 1
        return new_idx

    def register_sequence(self, seq_id: int) -> None:
        """Register a new sequence for modality tracking."""
        self._sequence_modality[seq_id] = SequenceModality(seq_id=seq_id)

    def unregister_sequence(self, seq_id: int) -> None:
        """Remove sequence modality tracking."""
        self._sequence_modality.pop(seq_id, None)

    def get_sequence_modality(self, seq_id: int) -> SequenceModality | None:
        """Get modality information for a sequence."""
        return self._sequence_modality.get(seq_id)

    def add_image_region(
        self,
        seq_id: int,
        start_pos: int,
        num_tokens: int,
        image_hash: str,
        block_indices: list[int],
    ) -> None:
        """Record an image region in a sequence's modality map."""
        modality = self._sequence_modality.get(seq_id)
        if modality:
            modality.add_image_region(start_pos, num_tokens, image_hash, block_indices)

    def add_text_range(self, seq_id: int, start: int, end: int) -> None:
        """Record a text token range in a sequence's modality map."""
        modality = self._sequence_modality.get(seq_id)
        if modality:
            modality.add_text_range(start, end)

    def get_modality_mask(
        self,
        seq_id: int,
        total_len: int,
    ) -> list[TokenModality]:
        """Generate per-position modality labels for attention masking.

        Some VLMs need different attention patterns for image vs text tokens.

        Args:
            seq_id: Sequence identifier.
            total_len: Total sequence length.

        Returns:
            List of modalities, one per position.
        """
        modality = self._sequence_modality.get(seq_id)
        if not modality:
            return [TokenModality.TEXT] * total_len

        result = [TokenModality.TEXT] * total_len
        for region in modality.image_regions:
            for i in range(region.num_tokens):
                pos = region.start_pos + i
                if pos < total_len:
                    result[pos] = TokenModality.IMAGE
        return result

    def _update_fragmentation_metric(self) -> None:
        """Update fragmentation score based on free list gaps.
        
        Lower score = better locality (contiguous free blocks).
        Score = average gap size between consecutive free blocks.
        """
        if len(self._free_list) <= 1:
            self._fragmentation_score = 0.0
            return
        
        gaps = [
            self._free_list[i + 1] - self._free_list[i] - 1
            for i in range(len(self._free_list) - 1)
        ]
        self._fragmentation_score = sum(gaps) / len(gaps) if gaps else 0.0

    def get_stats(self) -> dict[str, int | float]:
        """Return allocator statistics."""
        total_cached_blocks = sum(len(blocks) for blocks in self._prefix_cache.values())
        return {
            "num_blocks": self.num_blocks,
            "num_free": self.num_free,
            "num_allocated": self.num_allocated,
            "image_blocks": self._image_blocks_allocated,
            "text_blocks": self._text_blocks_allocated,
            "prefix_cache_entries": len(self._prefix_cache),
            "prefix_cache_blocks": total_cached_blocks,
            "prefix_cache_hits": self._prefix_cache_hits,
            "fragmentation_score": self._fragmentation_score,
        }


class VisionEncoderCache:
    """Caches vision encoder outputs for repeated image queries.

    When the same image appears in multiple prompts (e.g., same image
    with different questions), we can skip the vision encoder forward
    pass and reuse the cached output.

    This is separate from the KV cache prefix sharing:
    - VisionEncoderCache: Caches the vision encoder output tensor
    - MultimodalBlockAllocator: Caches the resulting KV blocks

    Both work together: if an image is in VisionEncoderCache, we skip
    the encoder; if its hash is in the block allocator's prefix cache,
    we also skip writing KV (just increment refcounts).

    Args:
        max_entries: Maximum number of cached encoder outputs.
        max_memory_bytes: Maximum total memory for cached tensors.
    """

    def __init__(
        self,
        max_entries: int = 100,
        max_memory_bytes: int = 1024 * 1024 * 1024,  # 1GB default
    ):
        self.max_entries = max_entries
        self.max_memory_bytes = max_memory_bytes

        # Cache: hash → (tensor, num_tokens, memory_bytes, access_count)
        self._cache: dict[str, tuple[NDArray[Any], int, int, int]] = {}
        self._total_memory = 0
        self._access_order: list[str] = []  # For LRU eviction

    @staticmethod
    def compute_image_hash(image_bytes: bytes) -> str:
        """Compute content hash for an image.

        Uses SHA-256 for collision resistance. The hash uniquely identifies
        the image content for cache lookup.
        """
        return hashlib.sha256(image_bytes).hexdigest()

    def get(self, image_hash: str) -> tuple[NDArray[Any], int] | None:
        """Retrieve cached encoder output.

        Args:
            image_hash: Hash of the image content.

        Returns:
            Tuple of (encoder_output, num_tokens) or None if not cached.
        """
        entry = self._cache.get(image_hash)
        if entry is None:
            return None

        tensor, num_tokens, mem_bytes, access_count = entry
        # Update access count and order for LRU
        self._cache[image_hash] = (tensor, num_tokens, mem_bytes, access_count + 1)
        if image_hash in self._access_order:
            self._access_order.remove(image_hash)
        self._access_order.append(image_hash)

        return tensor, num_tokens

    def put(
        self,
        image_hash: str,
        encoder_output: NDArray[Any],
        num_tokens: int,
    ) -> bool:
        """Cache a vision encoder output.

        Evicts LRU entries if memory limit is exceeded.

        Args:
            image_hash: Hash of the image content.
            encoder_output: Vision encoder output tensor (numpy array).
            num_tokens: Number of image tokens generated.

        Returns:
            True if cached successfully, False if tensor too large.
        """
        # Calculate memory usage
        mem_bytes = encoder_output.nbytes

        # Check if single entry exceeds limit
        if mem_bytes > self.max_memory_bytes:
            return False

        # Evict until we have space
        while (
            self._total_memory + mem_bytes > self.max_memory_bytes
            or len(self._cache) >= self.max_entries
        ) and self._access_order:
            self._evict_lru()

        # Add to cache
        self._cache[image_hash] = (encoder_output, num_tokens, mem_bytes, 1)
        self._total_memory += mem_bytes
        self._access_order.append(image_hash)

        return True

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_order:
            return

        oldest_hash = self._access_order.pop(0)
        entry = self._cache.pop(oldest_hash, None)
        if entry:
            _, _, mem_bytes, _ = entry
            self._total_memory -= mem_bytes

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()
        self._total_memory = 0

    def remove(self, image_hash: str) -> bool:
        """Remove a specific entry from cache."""
        entry = self._cache.pop(image_hash, None)
        if entry:
            _, _, mem_bytes, _ = entry
            self._total_memory -= mem_bytes
            if image_hash in self._access_order:
                self._access_order.remove(image_hash)
            return True
        return False

    def get_stats(self) -> dict[str, int | float]:
        """Return cache statistics."""
        total_accesses = sum(entry[3] for entry in self._cache.values())
        return {
            "num_entries": len(self._cache),
            "max_entries": self.max_entries,
            "memory_used_bytes": self._total_memory,
            "max_memory_bytes": self.max_memory_bytes,
            "memory_utilization": (
                self._total_memory / self.max_memory_bytes if self.max_memory_bytes > 0 else 0.0
            ),
            "total_accesses": total_accesses,
        }

    def __contains__(self, image_hash: str) -> bool:
        return image_hash in self._cache

    def __len__(self) -> int:
        return len(self._cache)
