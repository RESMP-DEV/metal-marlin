"""Block allocator for paged KV cache.

Manages a pool of fixed-size blocks. Supports reference counting for
copy-on-write (beam search, speculative decoding).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BlockState:
    """Metadata for a single physical block."""

    block_idx: int
    ref_count: int = 0
    is_free: bool = True


class BlockAllocator:
    """Fixed-pool block allocator with reference counting.

    Allocates blocks from a pre-sized pool. Blocks with ref_count > 1
    are shared (COW) and must be copied before mutation.

    Args:
        num_blocks: Total number of physical blocks in the pool.
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.blocks: list[BlockState] = [
            BlockState(block_idx=i) for i in range(num_blocks)
        ]
        self._free_list: list[int] = list(range(num_blocks - 1, -1, -1))

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
        idx = self._free_list.pop()
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
            self._free_list.append(block_idx)

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
