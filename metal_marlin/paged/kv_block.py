"""Fixed-size KV cache block for paged attention.

Based on vLLM's PagedAttention (Kwon et al., "Efficient Memory Management
for Large Language Model Serving with PagedAttention", SOSP 2023).

Block size chosen to balance:
- Memory efficiency (smaller blocks = less fragmentation)
- Compute efficiency (larger blocks = better GPU utilization)

vLLM default: 16 tokens per block.
For M4 with simdgroup, 16 aligns well with 32-wide SIMD.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx


@dataclass(frozen=True)
class KVBlockConfig:
    """Immutable configuration for KV blocks."""

    block_size: int = 16  # Tokens per block
    num_heads: int = 32
    head_dim: int = 128  # Common for Llama models
    dtype: mx.Dtype = field(default=mx.float16)

    @property
    def memory_bytes(self) -> int:
        """Memory footprint of one block in bytes."""
        elem_size = 2 if self.dtype == mx.float16 else 4
        return 2 * self.block_size * self.num_heads * self.head_dim * elem_size

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Storage shape: [2, block_size, num_heads, head_dim]."""
        return (2, self.block_size, self.num_heads, self.head_dim)


class KVBlock:
    """Fixed-size KV cache block for paged attention.

    MLX arrays are immutable, so append operations produce new arrays.
    The block tracks how many token slots are filled and manages
    copy-on-write semantics via reference counting.

    Storage layout: [2, block_size, num_heads, head_dim]
    Index 0 = K, Index 1 = V.
    """

    __slots__ = ("config", "_data", "_token_count", "_ref_count")

    def __init__(self, config: KVBlockConfig | None = None) -> None:
        self.config = config or KVBlockConfig()
        self._data: mx.array | None = None
        self._token_count: int = 0
        self._ref_count: int = 0

    def allocate(self) -> None:
        """Allocate block memory."""
        self._data = mx.zeros(self.config.shape, dtype=self.config.dtype)
        self._token_count = 0

    @property
    def data(self) -> mx.array | None:
        """The underlying storage array."""
        return self._data

    @property
    def token_count(self) -> int:
        """Number of tokens currently stored."""
        return self._token_count

    @property
    def ref_count(self) -> int:
        """Reference count for copy-on-write."""
        return self._ref_count

    def acquire(self) -> None:
        """Increment reference count."""
        self._ref_count += 1

    def release(self) -> int:
        """Decrement reference count. Returns new count."""
        self._ref_count = max(0, self._ref_count - 1)
        return self._ref_count

    def append_kv(self, k: mx.array, v: mx.array) -> int:
        """Append K,V for a single token.

        Args:
            k: Key tensor of shape [num_heads, head_dim].
            v: Value tensor of shape [num_heads, head_dim].

        Returns:
            Number of slots remaining in block.

        Raises:
            RuntimeError: If block is full or not allocated.
        """
        if self._data is None:
            raise RuntimeError("Block not allocated")
        if self._token_count >= self.config.block_size:
            raise RuntimeError("Block is full")

        # MLX is functional; we build updated slices and reassemble.
        idx = self._token_count
        k_plane = self._data[0]
        v_plane = self._data[1]

        # Replace row `idx` in each plane.
        k_plane = mx.concatenate([
            k_plane[:idx],
            mx.expand_dims(k, axis=0),
            k_plane[idx + 1:],
        ], axis=0)
        v_plane = mx.concatenate([
            v_plane[:idx],
            mx.expand_dims(v, axis=0),
            v_plane[idx + 1:],
        ], axis=0)

        self._data = mx.stack([k_plane, v_plane], axis=0)
        self._token_count += 1
        return self.config.block_size - self._token_count

    def append_kv_batch(self, keys: mx.array, values: mx.array) -> int:
        """Append multiple token KV pairs at once.

        More efficient than repeated single appends since it performs
        one array reconstruction instead of N.

        Args:
            keys: Key tensor of shape [num_tokens, num_heads, head_dim].
            values: Value tensor of shape [num_tokens, num_heads, head_dim].

        Returns:
            Number of slots remaining in block.

        Raises:
            RuntimeError: If block is full/unallocated or batch exceeds capacity.
        """
        if self._data is None:
            raise RuntimeError("Block not allocated")

        num_tokens = keys.shape[0]
        if self._token_count + num_tokens > self.config.block_size:
            raise RuntimeError(
                f"Batch of {num_tokens} tokens exceeds remaining "
                f"{self.config.block_size - self._token_count} slots"
            )

        idx = self._token_count
        end = idx + num_tokens
        k_plane = self._data[0]
        v_plane = self._data[1]

        k_plane = mx.concatenate([
            k_plane[:idx],
            keys,
            k_plane[end:],
        ], axis=0)
        v_plane = mx.concatenate([
            v_plane[:idx],
            values,
            v_plane[end:],
        ], axis=0)

        self._data = mx.stack([k_plane, v_plane], axis=0)
        self._token_count += num_tokens
        return self.config.block_size - self._token_count

    def get_kv(self) -> tuple[mx.array, mx.array]:
        """Get the filled portion of K and V.

        Returns:
            Tuple of (keys, values), each of shape [token_count, num_heads, head_dim].

        Raises:
            RuntimeError: If block is not allocated.
        """
        if self._data is None:
            raise RuntimeError("Block not allocated")
        return self._data[0, :self._token_count], self._data[1, :self._token_count]

    @property
    def is_full(self) -> bool:
        """Whether the block has no remaining slots."""
        return self._token_count >= self.config.block_size

    @property
    def is_empty(self) -> bool:
        """Whether the block has no tokens stored."""
        return self._token_count == 0

    @property
    def remaining(self) -> int:
        """Number of empty slots."""
        return self.config.block_size - self._token_count

    @property
    def memory_bytes(self) -> int:
        """Memory footprint of this block in bytes."""
        return self.config.memory_bytes

    def reset(self) -> None:
        """Clear block contents without deallocating."""
        if self._data is not None:
            self._data = mx.zeros(self.config.shape, dtype=self.config.dtype)
        self._token_count = 0
        self._ref_count = 0

    def copy(self) -> KVBlock:
        """Create an independent copy of this block (for CoW)."""
        new_block = KVBlock(config=self.config)
        if self._data is not None:
            # MLX arrays are already immutable/refcounted, but we want
            # logical independence for the mutable wrapper state.
            new_block._data = self._data
            new_block._token_count = self._token_count
        return new_block

    def __repr__(self) -> str:
        return (
            f"KVBlock(tokens={self._token_count}/{self.config.block_size}, "
            f"heads={self.config.num_heads}, dim={self.config.head_dim}, "
            f"refs={self._ref_count}, "
            f"allocated={'yes' if self._data is not None else 'no'})"
        )
