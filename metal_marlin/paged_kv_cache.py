"""Paged KV cache wrapper with vLLM-style 16-token blocks.

This module provides a lightweight block allocator and block-table interface
for paged attention dispatch. Storage is preallocated in fixed-size blocks and
logical token positions are mapped to physical blocks via per-sequence tables.

Supported storage formats:
- ``fp16``: full-precision KV values.
- ``fp8``: byte KV values plus FP16 scales.
- ``int4``: packed INT4 KV values (8 values per uint32) plus FP16 scales.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

KVCacheDType = Literal["fp16", "fp8", "int4"]


@dataclass(frozen=True)
class _StorageLayout:
    """Resolved storage layout for a cache dtype."""

    value_shape: tuple[int, ...]
    value_dtype: np.dtype
    scale_shape: tuple[int, ...] | None
    scale_dtype: np.dtype | None


class PagedKVCache:
    """Paged KV cache using fixed-size blocks.

    The cache follows the vLLM paged-attention pattern with 16 tokens per block.
    Each sequence has a block table that maps logical block offsets to physical
    block indices in the global block pool.

    Args:
        num_blocks: Total number of physical blocks in the cache pool.
        num_kv_heads: Number of KV heads per token.
        head_dim: KV head dimension.
        dtype: Storage format. One of ``"fp16"``, ``"fp8"``, ``"int4"``.
        block_size: Tokens per block. Must be 16.
    """

    BLOCK_SIZE = 16

    def __init__(
        self,
        num_blocks: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: str = "fp16",
        block_size: int = BLOCK_SIZE,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be > 0, got {num_kv_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {head_dim}")
        if block_size != self.BLOCK_SIZE:
            raise ValueError(
                f"PagedKVCache uses fixed block_size={self.BLOCK_SIZE}, got {block_size}"
            )

        self.num_blocks = num_blocks
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype: KVCacheDType = self._normalize_dtype(dtype)

        # Free physical block pool (address ordered).
        self._free_blocks: deque[int] = deque(range(num_blocks))

        # Sequence metadata.
        self._next_seq_id = 0
        self._block_tables: dict[int, list[int]] = {}
        self._context_lens: dict[int, int] = {}

        layout = self._resolve_layout()
        self.k_blocks = np.zeros(layout.value_shape, dtype=layout.value_dtype)
        self.v_blocks = np.zeros(layout.value_shape, dtype=layout.value_dtype)
        self.k_scales = (
            np.ones(layout.scale_shape, dtype=layout.scale_dtype)
            if layout.scale_shape is not None and layout.scale_dtype is not None
            else None
        )
        self.v_scales = (
            np.ones(layout.scale_shape, dtype=layout.scale_dtype)
            if layout.scale_shape is not None and layout.scale_dtype is not None
            else None
        )

    @staticmethod
    def _normalize_dtype(dtype: str) -> KVCacheDType:
        key = dtype.strip().lower()
        aliases = {
            "float16": "fp16",
            "half": "fp16",
            "e4m3": "fp8",
            "int4_sym": "int4",
        }
        key = aliases.get(key, key)
        if key not in {"fp16", "fp8", "int4"}:
            raise ValueError("dtype must be one of: fp16, fp8, int4")
        return key  # type: ignore[return-value]

    def _resolve_layout(self) -> _StorageLayout:
        # Base dense shape: [num_blocks, block_size, num_kv_heads, head_dim]
        dense_shape = (
            self.num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
        )

        if self.dtype == "fp16":
            return _StorageLayout(
                value_shape=dense_shape,
                value_dtype=np.dtype(np.float16),
                scale_shape=None,
                scale_dtype=None,
            )

        if self.dtype == "fp8":
            scale_shape = (
                self.num_blocks,
                self.block_size,
                self.num_kv_heads,
                1,
            )
            return _StorageLayout(
                value_shape=dense_shape,
                value_dtype=np.dtype(np.uint8),
                scale_shape=scale_shape,
                scale_dtype=np.dtype(np.float16),
            )

        # INT4 packed as 8 values per uint32 to match common kernel expectations.
        packed_head_dim = (self.head_dim + 7) // 8
        packed_shape = (
            self.num_blocks,
            self.block_size,
            self.num_kv_heads,
            packed_head_dim,
        )
        scale_shape = (
            self.num_blocks,
            self.block_size,
            self.num_kv_heads,
            1,
        )
        return _StorageLayout(
            value_shape=packed_shape,
            value_dtype=np.dtype(np.uint32),
            scale_shape=scale_shape,
            scale_dtype=np.dtype(np.float16),
        )

    def allocate_blocks(self, num_tokens: int, seq_id: int | None = None) -> list[int]:
        """Allocate blocks for ``num_tokens`` and attach them to a sequence.

        If ``seq_id`` is omitted, a new sequence is created automatically.

        Args:
            num_tokens: Number of additional logical tokens for the sequence.
            seq_id: Optional sequence identifier.

        Returns:
            List of newly allocated physical block indices.

        Raises:
            RuntimeError: If insufficient free blocks are available.
        """
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be >= 0, got {num_tokens}")

        if seq_id is None:
            seq_id = self._next_seq_id
            self._next_seq_id += 1

        if seq_id not in self._block_tables:
            self._block_tables[seq_id] = []
            self._context_lens[seq_id] = 0

        current_tokens = self._context_lens[seq_id]
        target_tokens = current_tokens + num_tokens
        blocks_required = (target_tokens + self.block_size - 1) // self.block_size
        blocks_missing = blocks_required - len(self._block_tables[seq_id])

        if blocks_missing > len(self._free_blocks):
            raise RuntimeError(
                f"Out of KV blocks: need {blocks_missing}, have {len(self._free_blocks)} free"
            )

        newly_allocated: list[int] = []
        for _ in range(blocks_missing):
            block_idx = self._free_blocks.popleft()
            self._block_tables[seq_id].append(block_idx)
            newly_allocated.append(block_idx)

        self._context_lens[seq_id] = target_tokens
        return newly_allocated

    def get_block_tables(self, pad_value: int = -1) -> NDArray[np.int32]:
        """Return padded block tables for kernel dispatch.

        The output has shape ``[num_seqs, max_blocks_per_seq]`` and dtype int32.
        Sequence rows are ordered by ascending ``seq_id``.
        """
        if not self._block_tables:
            return np.empty((0, 0), dtype=np.int32)

        seq_ids = sorted(self._block_tables)
        max_blocks = max(len(self._block_tables[seq_id]) for seq_id in seq_ids)
        tables = np.full((len(seq_ids), max_blocks), pad_value, dtype=np.int32)
        for row, seq_id in enumerate(seq_ids):
            blocks = self._block_tables[seq_id]
            if blocks:
                tables[row, : len(blocks)] = blocks
        return tables

    def get_context_lens(self) -> NDArray[np.int32]:
        """Return context lengths aligned with ``get_block_tables()`` row order."""
        if not self._context_lens:
            return np.empty((0,), dtype=np.int32)
        seq_ids = sorted(self._context_lens)
        return np.asarray([self._context_lens[seq_id] for seq_id in seq_ids], dtype=np.int32)

    def get_sequence_ids(self) -> NDArray[np.int32]:
        """Return sequence IDs aligned with ``get_block_tables()`` row order."""
        if not self._block_tables:
            return np.empty((0,), dtype=np.int32)
        return np.asarray(sorted(self._block_tables), dtype=np.int32)

    @property
    def num_free_blocks(self) -> int:
        """Number of currently free physical blocks."""
        return len(self._free_blocks)

    @property
    def num_allocated_blocks(self) -> int:
        """Number of currently allocated physical blocks."""
        return self.num_blocks - len(self._free_blocks)

    def quantize_kv(
        self,
        k: NDArray,
        v: NDArray,
        slot_mapping: NDArray[np.int32],
    ) -> None:
        """Quantize and store K/V tensors into the paged cache.

        Args:
            k: Key tensor of shape ``[num_tokens, num_heads, head_dim]``.
            v: Value tensor of shape ``[num_tokens, num_heads, head_dim]``.
            slot_mapping: Flat slot indices for each token.
        """
        if k.shape != v.shape:
            raise ValueError(f"k and v shapes must match, got {k.shape} vs {v.shape}")
        if k.shape[0] != slot_mapping.shape[0]:
            raise ValueError(
                f"k/v batch size {k.shape[0]} does not match slot_mapping {slot_mapping.shape[0]}"
            )

        block_indices = slot_mapping // self.block_size
        block_offsets = slot_mapping % self.block_size

        if self.dtype == "fp16":
            self.k_blocks[block_indices, block_offsets] = k
            self.v_blocks[block_indices, block_offsets] = v

        elif self.dtype == "fp8":
            # E4M3 quantization (max value 448.0)
            # We store per-token scales.
            # Scale = max(abs(x)) / 448.0
            MAX_E4M3 = 448.0

            # Compute scales [num_tokens, num_heads, 1]
            k_abs_max = np.max(np.abs(k), axis=-1, keepdims=True)
            v_abs_max = np.max(np.abs(v), axis=-1, keepdims=True)

            k_scales = k_abs_max / MAX_E4M3
            v_scales = v_abs_max / MAX_E4M3

            # Avoid division by zero
            k_scales[k_scales == 0] = 1.0
            v_scales[v_scales == 0] = 1.0

            # Quantize
            # Map [-448, 448] to [0, 255] centered at 128
            scaled_k = k / k_scales
            k_q = np.round(scaled_k / MAX_E4M3 * 127.0 + 128.0)
            k_q = np.clip(k_q, 0, 255).astype(np.uint8)

            scaled_v = v / v_scales
            v_q = np.round(scaled_v / MAX_E4M3 * 127.0 + 128.0)
            v_q = np.clip(v_q, 0, 255).astype(np.uint8)

            # Store quantized values
            self.k_blocks[block_indices, block_offsets] = k_q
            self.v_blocks[block_indices, block_offsets] = v_q

            # Store scales
            if self.k_scales is not None:
                self.k_scales[block_indices, block_offsets] = k_scales.astype(np.float16)
            if self.v_scales is not None:
                self.v_scales[block_indices, block_offsets] = v_scales.astype(np.float16)

        elif self.dtype == "int4":
            raise NotImplementedError("int4 quantization not implemented in Python yet")

    def dequantize_kv(
        self,
        slot_mapping: NDArray[np.int32],
    ) -> tuple[NDArray, NDArray]:
        """Retrieve and dequantize K/V tensors from the paged cache.

        Args:
            slot_mapping: Flat slot indices to retrieve.

        Returns:
            Tuple of (k, v) tensors of shape ``[num_tokens, num_heads, head_dim]``.
        """
        block_indices = slot_mapping // self.block_size
        block_offsets = slot_mapping % self.block_size

        if self.dtype == "fp16":
            k = self.k_blocks[block_indices, block_offsets].astype(np.float16)
            v = self.v_blocks[block_indices, block_offsets].astype(np.float16)
            return k, v

        elif self.dtype == "fp8":
            k_stored = self.k_blocks[block_indices, block_offsets]
            v_stored = self.v_blocks[block_indices, block_offsets]
            
            # Retrieve scales
            if self.k_scales is None or self.v_scales is None:
                raise RuntimeError("FP8 cache missing scales")
                
            k_scales = self.k_scales[block_indices, block_offsets].astype(np.float32)
            v_scales = self.v_scales[block_indices, block_offsets].astype(np.float32)

            # Dequantize (simulated)
            MAX_E4M3 = 448.0
            
            # Map [0, 255] back to [-448, 448]
            k_reconstructed_scaled = (k_stored.astype(np.float32) - 128.0) / 127.0 * MAX_E4M3
            k = k_reconstructed_scaled * k_scales
            
            v_reconstructed_scaled = (v_stored.astype(np.float32) - 128.0) / 127.0 * MAX_E4M3
            v = v_reconstructed_scaled * v_scales
            
            return k, v

        elif self.dtype == "int4":
            raise NotImplementedError("int4 dequantization not implemented in Python yet")
            
        return np.array([]), np.array([])
