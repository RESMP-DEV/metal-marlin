"""Sequence-to-block mapping (page table) for paged KV cache.

Maps logical sequence positions to physical block indices. Handles
block allocation on append, deallocation on sequence removal, and
copy-on-write forking for beam search / speculative decoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .allocator import BlockAllocator


@dataclass
class SequenceState:
    """Track block mapping for a single sequence."""

    seq_id: int
    block_indices: list[int] = field(default_factory=list)
    logical_len: int = 0  # Total tokens in sequence

    @property
    def num_blocks(self) -> int:
        return len(self.block_indices)


class PageTable:
    """Maps sequences to KV cache blocks.

    Handles:
    - Block allocation for new sequences
    - Block extension when sequence grows
    - Block sharing for beam search / speculative decoding
    - Copy-on-write for shared blocks on mutation

    Args:
        allocator: BlockAllocator instance managing the physical block pool.
        block_size: Number of tokens per block.
    """

    def __init__(self, allocator: BlockAllocator, block_size: int = 16):
        self.allocator = allocator
        self.block_size = block_size
        self.sequences: dict[int, SequenceState] = {}

    def add_sequence(self, seq_id: int) -> bool:
        """Register new sequence with an initial block.

        Returns False if the allocator is out of memory.
        """
        block_idx = self.allocator.allocate()
        if block_idx is None:
            return False

        self.sequences[seq_id] = SequenceState(
            seq_id=seq_id,
            block_indices=[block_idx],
            logical_len=0,
        )
        return True

    def append_token(self, seq_id: int) -> bool:
        """Record that sequence grew by one token.

        Allocates a new block if the current tail block is full.
        Returns False if out of memory.
        """
        state = self.sequences[seq_id]
        state.logical_len += 1

        # Check if need new block
        current_capacity = len(state.block_indices) * self.block_size
        if state.logical_len > current_capacity:
            block_idx = self.allocator.allocate()
            if block_idx is None:
                state.logical_len -= 1  # Rollback
                return False
            state.block_indices.append(block_idx)

        return True

    def append_tokens(self, seq_id: int, num_tokens: int) -> bool:
        """Append multiple tokens, allocating blocks as needed.

        Atomic: either all tokens are appended or none are (on OOM).
        """
        state = self.sequences[seq_id]
        new_len = state.logical_len + num_tokens
        blocks_needed = (new_len + self.block_size - 1) // self.block_size
        blocks_to_alloc = blocks_needed - len(state.block_indices)

        if blocks_to_alloc > 0:
            if self.allocator.num_free < blocks_to_alloc:
                return False
            for _ in range(blocks_to_alloc):
                block_idx = self.allocator.allocate()
                if block_idx is None:
                    # Shouldn't happen given the check above, but be safe
                    return False
                state.block_indices.append(block_idx)

        state.logical_len = new_len
        return True

    def remove_sequence(self, seq_id: int) -> None:
        """Free all blocks for sequence."""
        state = self.sequences.pop(seq_id, None)
        if state:
            for block_idx in state.block_indices:
                self.allocator.free(block_idx)

    def fork_sequence(self, src_id: int, dst_id: int) -> bool:
        """Create COW copy for beam search.

        Shares all blocks via reference counting. The first write to a
        shared block triggers copy_on_write in the allocator.
        """
        src = self.sequences.get(src_id)
        if not src:
            return False

        # Share blocks via refcount
        for block_idx in src.block_indices:
            self.allocator.blocks[block_idx].ref_count += 1

        self.sequences[dst_id] = SequenceState(
            seq_id=dst_id,
            block_indices=src.block_indices.copy(),
            logical_len=src.logical_len,
        )
        return True

    def cow_block(self, seq_id: int, block_offset: int) -> int | None:
        """Copy-on-write for a specific block in a sequence.

        If the block is shared (ref_count > 1), allocates a fresh copy.
        Returns the (possibly new) block index, or None on OOM.
        """
        state = self.sequences[seq_id]
        old_idx = state.block_indices[block_offset]
        new_idx = self.allocator.copy_on_write(old_idx)
        if new_idx is None:
            return None
        state.block_indices[block_offset] = new_idx
        return new_idx

    def get_block_table(self, seq_id: int) -> list[int]:
        """Get block indices for attention kernel dispatch."""
        return self.sequences[seq_id].block_indices

    def get_slot_mapping(self, seq_id: int) -> tuple[int, int]:
        """Get (block_idx, offset_within_block) for the current write position.

        Useful for the attention kernel to know where to write new KV entries.
        """
        state = self.sequences[seq_id]
        if state.logical_len == 0:
            return state.block_indices[0], 0
        # Position of the next write slot
        pos = state.logical_len
        block_offset = pos // self.block_size
        slot_offset = pos % self.block_size
        # If we're at the start of a new block that hasn't been allocated yet,
        # point to the last allocated block's end
        if block_offset >= len(state.block_indices):
            block_offset = len(state.block_indices) - 1
            slot_offset = self.block_size
        return state.block_indices[block_offset], slot_offset

    def has_sequence(self, seq_id: int) -> bool:
        """Check if a sequence is registered in the page table."""
        return seq_id in self.sequences

    @property
    def num_sequences(self) -> int:
        return len(self.sequences)

    def sequence_ids(self) -> list[int]:
        return list(self.sequences.keys())
