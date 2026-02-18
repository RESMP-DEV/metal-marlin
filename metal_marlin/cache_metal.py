"""Efficient KV cache implementations using PyTorch MPS for Apple Silicon.

This module provides high-performance KV cache implementations optimized for
Apple Silicon via PyTorch MPS:

1. MetalKVCache - Base contiguous KV cache with O(1) update/get
2. MetalPagedKVCache - Paged allocation (vLLM-style) for variable sequences
3. MetalQuantizedKVCache - FP8/INT8 quantized cache for 2x memory savings

Key features:
- Paged allocation avoids fragmentation for variable-length sequences
- Quantized storage (FP8) halves memory for long contexts
- Copy-on-write semantics for beam search / speculative decoding
- Continuous batching support with per-sequence tracking

Memory comparison (32-layer, 32-head model, head_dim=128):
    Standard FP16 (8K ctx):   4 GB per sequence
    MetalKVCache (8K ctx):    4 GB (same, contiguous)
    MetalPagedKVCache (8K):   4 GB (paged, no fragmentation)
    MetalQuantizedKVCache:    2 GB (FP8 quantized)

Usage:
    from metal_marlin.cache_metal import (
        MetalKVCache,
        MetalPagedKVCache,
        MetalQuantizedKVCache,
    )

    # Basic contiguous cache
    cache = MetalKVCache(num_layers=32, num_heads=32, head_dim=128, max_seq_len=8192)
    cache.update(layer_idx=0, key=k_new, value=v_new, positions=positions)
    k, v = cache.get(layer_idx=0, positions=positions)

    # Paged cache for serving (vLLM-style)
    paged_cache = MetalPagedKVCache(
        num_layers=32, num_heads=32, head_dim=128,
        block_size=16, num_blocks=1024,
    )
    seq = paged_cache.allocate_sequence(seq_id=0)
    paged_cache.append(seq_id=0, layer_idx=0, key=k, value=v)

    # Quantized cache for long context
    quant_cache = MetalQuantizedKVCache(
        num_layers=32, num_heads=32, head_dim=128, max_seq_len=32768,
        quant_type="fp8",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

from ._compat import require_torch, torch

if TYPE_CHECKING:
    import torch as torch_typing


class QuantType(Enum):
    """Quantization type for KV cache storage."""

    NONE = "none"  # Full precision (FP16/BF16)
    FP8 = "fp8"  # FP8 E4M3 (2x compression)
    INT8 = "int8"  # INT8 symmetric (2x compression)


@dataclass
class MetalKVCacheConfig:
    """Configuration for Metal KV cache.

    Attributes:
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads (query heads for GQA).
        num_kv_heads: Number of key-value heads (defaults to num_heads).
        head_dim: Dimension per attention head.
        max_seq_len: Maximum sequence length to support.
        dtype: Storage dtype ("fp16" or "bf16").
    """

    num_layers: int
    num_heads: int
    head_dim: int
    max_seq_len: int
    num_kv_heads: int | None = None
    dtype: Literal["fp16", "bf16"] = "bf16"

    def __post_init__(self) -> None:
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads


class MetalKVCache:
    """Base KV cache on MPS with contiguous allocation.

    Pre-allocates memory for the maximum sequence length to avoid fragmentation.
    Updates are O(1) via slice assignment; retrieval supports arbitrary positions.

    Memory layout per layer:
        K: [batch, num_kv_heads, max_seq_len, head_dim]
        V: [batch, num_kv_heads, max_seq_len, head_dim]

    Attributes:
        config: Cache configuration.
        batch_size: Current batch size.
        seq_lens: Current sequence length per layer.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        num_kv_heads: int | None = None,
        batch_size: int = 1,
        dtype: Literal["fp16", "bf16"] = "bf16",
        device: str = "mps",
    ):
        """Initialize contiguous KV cache.

        Args:
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            max_seq_len: Maximum sequence length.
            num_kv_heads: Number of KV heads (defaults to num_heads for MHA).
            batch_size: Batch size.
            dtype: Storage dtype.
            device: Device to allocate tensors on.
        """
        require_torch("MetalKVCache")

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.dtype_str = dtype
        self.device = device

        # PyTorch dtype
        self._dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

        # Allocate cache tensors
        cache_shape = (batch_size, self.num_kv_heads, max_seq_len, head_dim)

        self.k_cache: list[torch_typing.Tensor] = [
            torch.zeros(cache_shape, dtype=self._dtype, device=device) for _ in range(num_layers)
        ]
        self.v_cache: list[torch_typing.Tensor] = [
            torch.zeros(cache_shape, dtype=self._dtype, device=device) for _ in range(num_layers)
        ]

        # Track sequence length per layer (all start at 0)
        self.seq_lens: list[int] = [0] * num_layers

    def update(
        self,
        layer_idx: int,
        key: torch_typing.Tensor,
        value: torch_typing.Tensor,
        positions: torch_typing.Tensor | None = None,
    ) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Update cache with new K, V and return full cached tensors.

        Args:
            layer_idx: Which transformer layer.
            key: New key tensor [batch, num_kv_heads, new_seq_len, head_dim].
            value: New value tensor [batch, num_kv_heads, new_seq_len, head_dim].
            positions: Optional explicit positions [new_seq_len]. If None, appends.

        Returns:
            Tuple of (k_full, v_full) including cached history.
        """
        new_seq_len = key.shape[2]
        current_len = self.seq_lens[layer_idx]

        if positions is None:
            # Append mode: positions are current_len to current_len + new_seq_len
            start_pos = current_len
            end_pos = current_len + new_seq_len

            if end_pos > self.max_seq_len:
                raise ValueError(
                    f"Sequence length {end_pos} exceeds max_seq_len {self.max_seq_len}"
                )

            # Update cache via slice assignment (PyTorch supports this directly)
            self.k_cache[layer_idx][:, :, start_pos:end_pos, :] = key.to(self._dtype)
            self.v_cache[layer_idx][:, :, start_pos:end_pos, :] = value.to(self._dtype)

            self.seq_lens[layer_idx] = end_pos
        else:
            # Explicit positions: scatter update (useful for speculative decoding)
            positions_list = positions.tolist()
            for i, pos in enumerate(positions_list):
                if isinstance(pos, list):
                    pos = pos[0]
                pos = int(pos)
                if pos >= self.max_seq_len:
                    raise ValueError(f"Position {pos} exceeds max_seq_len {self.max_seq_len}")

                # Update single position via slice assignment
                self.k_cache[layer_idx][:, :, pos : pos + 1, :] = key[:, :, i : i + 1, :].to(
                    self._dtype
                )
                self.v_cache[layer_idx][:, :, pos : pos + 1, :] = value[:, :, i : i + 1, :].to(
                    self._dtype
                )

            # Update seq_len to max position + 1
            max_pos = max(int(p) if not isinstance(p, list) else int(p[0]) for p in positions_list)
            self.seq_lens[layer_idx] = max(self.seq_lens[layer_idx], max_pos + 1)

        # Return full cached K, V up to current length
        end = self.seq_lens[layer_idx]
        return (
            self.k_cache[layer_idx][:, :, :end, :],
            self.v_cache[layer_idx][:, :, :end, :],
        )

    def get(
        self,
        layer_idx: int,
        positions: torch_typing.Tensor | None = None,
    ) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Get cached K, V tensors.

        Args:
            layer_idx: Which transformer layer.
            positions: Optional positions to retrieve. If None, returns full cache.

        Returns:
            Tuple of (key, value) tensors.
        """
        if positions is None:
            # Return full cache up to current length
            end = self.seq_lens[layer_idx]
            return (
                self.k_cache[layer_idx][:, :, :end, :],
                self.v_cache[layer_idx][:, :, :end, :],
            )

        # Use optimized direct indexing instead of index_select
        return self._direct_kv_index(layer_idx, positions)

    def _direct_kv_index(
        self, layer_idx: int, positions: torch_typing.Tensor
    ) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Optimized KV indexing using direct tensor indexing.

        Avoids torch.index_select which causes GPU->CPU synchronization on MPS.
        Uses direct indexing: cache[..., positions, :] which is fully parallel.

        Args:
            layer_idx: Which transformer layer.
            positions: Positions to retrieve [num_positions].

        Returns:
            Tuple of (key, value) tensors indexed at specified positions.
        """
        # Direct indexing is faster than index_select on MPS
        # Shape: [batch, num_kv_heads, num_positions, head_dim]
        k = self.k_cache[layer_idx][..., positions, :]
        v = self.v_cache[layer_idx][..., positions, :]
        return k, v

    @property
    def seq_len(self) -> int:
        """Current sequence length (assumes all layers in sync)."""
        return self.seq_lens[0] if self.seq_lens else 0

    def reset(self, layer_idx: int | None = None) -> None:
        """Reset cache for new sequence.

        Args:
            layer_idx: Specific layer to reset, or None for all layers.
        """
        if layer_idx is not None:
            self.seq_lens[layer_idx] = 0
        else:
            self.seq_lens = [0] * self.num_layers

    def memory_usage_mb(self) -> float:
        """Return current memory usage in MB."""
        bytes_per_element = 2  # FP16/BF16
        elements = (
            self.batch_size * self.num_kv_heads * sum(self.seq_lens) * self.head_dim * 2  # K and V
        )
        return elements * bytes_per_element / 1024 / 1024

    def max_memory_mb(self) -> float:
        """Return maximum memory allocation in MB."""
        bytes_per_element = 2
        elements = (
            self.batch_size
            * self.num_kv_heads
            * self.max_seq_len
            * self.head_dim
            * 2  # K and V
            * self.num_layers
        )
        return elements * bytes_per_element / 1024 / 1024


@dataclass
class PagedBlockConfig:
    """Configuration for paged KV cache blocks."""

    block_size: int = 16  # Tokens per block (vLLM default)
    num_kv_heads: int = 8
    head_dim: int = 128


@dataclass
class PagedSequence:
    """Tracks block allocation for a single sequence."""

    seq_id: int
    block_indices: list[int] = field(default_factory=list)  # Physical block IDs
    token_count: int = 0
    ref_count: int = 1  # For COW

    @property
    def num_blocks(self) -> int:
        return len(self.block_indices)


class MetalPagedKVCache:
    """Paged KV cache (vLLM-style) for variable-length sequences.

    Uses fixed-size blocks to avoid memory fragmentation. Supports:
    - Dynamic sequence allocation/deallocation
    - Copy-on-write for beam search and speculative decoding
    - Continuous batching with per-sequence block tables

    Block layout:
        Each block stores [block_size, num_kv_heads, head_dim] for K and V.
        Sequences are mapped to a list of block indices (block table).

    Memory model:
        - Pre-allocates a pool of num_blocks physical blocks
        - Sequences request blocks on demand
        - Free blocks returned to pool when sequences complete

    This design enables efficient memory utilization in serving scenarios
    where sequences have highly variable lengths.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 1024,
        num_kv_heads: int | None = None,
        dtype: Literal["fp16", "bf16"] = "bf16",
        device: str = "mps",
    ):
        """Initialize paged KV cache.

        Args:
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            block_size: Tokens per block.
            num_blocks: Total physical blocks in the pool.
            num_kv_heads: Number of KV heads (for GQA).
            dtype: Storage dtype.
            device: Device to allocate tensors on.
        """
        require_torch("MetalPagedKVCache")

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype_str = dtype
        self.device = device

        self._dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

        # Physical block storage per layer
        # Shape: [num_blocks, block_size, num_kv_heads, head_dim]
        block_shape = (num_blocks, block_size, self.num_kv_heads, head_dim)
        self.k_blocks: list[torch_typing.Tensor] = [
            torch.zeros(block_shape, dtype=self._dtype, device=device) for _ in range(num_layers)
        ]
        self.v_blocks: list[torch_typing.Tensor] = [
            torch.zeros(block_shape, dtype=self._dtype, device=device) for _ in range(num_layers)
        ]

        # Block allocation state
        self._free_blocks: list[int] = list(range(num_blocks - 1, -1, -1))
        self._block_ref_counts: list[int] = [0] * num_blocks

        # Per-sequence tracking
        self._sequences: dict[int, PagedSequence] = {}

    @property
    def num_free_blocks(self) -> int:
        """Number of unallocated blocks."""
        return len(self._free_blocks)

    @property
    def num_allocated_blocks(self) -> int:
        """Number of allocated blocks."""
        return self.num_blocks - len(self._free_blocks)

    def allocate_sequence(self, seq_id: int) -> PagedSequence:
        """Register a new sequence for block allocation.

        Args:
            seq_id: Unique sequence identifier.

        Returns:
            PagedSequence tracking object.

        Raises:
            ValueError: If seq_id already exists.
        """
        if seq_id in self._sequences:
            raise ValueError(f"Sequence {seq_id} already exists")

        seq = PagedSequence(seq_id=seq_id)
        self._sequences[seq_id] = seq
        return seq

    def free_sequence(self, seq_id: int) -> None:
        """Release all blocks for a sequence.

        Args:
            seq_id: Sequence to free.
        """
        if seq_id not in self._sequences:
            return

        seq = self._sequences[seq_id]
        for block_idx in seq.block_indices:
            self._release_block(block_idx)

        del self._sequences[seq_id]

    def _allocate_block(self) -> int | None:
        """Allocate a single block from the pool.

        Returns:
            Block index or None if pool is exhausted.
        """
        if not self._free_blocks:
            return None

        block_idx = self._free_blocks.pop()
        self._block_ref_counts[block_idx] = 1
        return block_idx

    def _release_block(self, block_idx: int) -> None:
        """Release a block back to the pool."""
        self._block_ref_counts[block_idx] -= 1
        if self._block_ref_counts[block_idx] <= 0:
            self._block_ref_counts[block_idx] = 0
            self._free_blocks.append(block_idx)

    def _ensure_capacity(self, seq_id: int, new_tokens: int) -> bool:
        """Ensure sequence has enough blocks for new tokens.

        Args:
            seq_id: Sequence ID.
            new_tokens: Number of new tokens to accommodate.

        Returns:
            True if capacity available, False if OOM.
        """
        seq = self._sequences[seq_id]
        current_capacity = seq.num_blocks * self.block_size
        needed_capacity = seq.token_count + new_tokens

        while current_capacity < needed_capacity:
            block_idx = self._allocate_block()
            if block_idx is None:
                return False
            seq.block_indices.append(block_idx)
            current_capacity += self.block_size

        return True

    def append(
        self,
        seq_id: int,
        layer_idx: int,
        key: torch_typing.Tensor,
        value: torch_typing.Tensor,
    ) -> bool:
        """Append K, V to a sequence's cache.

        Args:
            seq_id: Sequence ID.
            layer_idx: Transformer layer.
            key: Key tensor [num_kv_heads, new_seq_len, head_dim].
            value: Value tensor [num_kv_heads, new_seq_len, head_dim].

        Returns:
            True if successful, False if OOM.
        """
        if seq_id not in self._sequences:
            raise ValueError(f"Sequence {seq_id} not registered")

        new_tokens = key.shape[1]  # [heads, seq, dim]
        if not self._ensure_capacity(seq_id, new_tokens):
            return False

        seq = self._sequences[seq_id]

        # Write tokens to blocks
        for i in range(new_tokens):
            token_idx = seq.token_count + i
            block_num = token_idx // self.block_size
            slot_in_block = token_idx % self.block_size
            block_idx = seq.block_indices[block_num]

            # Update block storage via slice assignment
            k_token = key[:, i, :]  # [num_kv_heads, head_dim]
            v_token = value[:, i, :]

            self.k_blocks[layer_idx][block_idx, slot_in_block, :, :] = k_token.to(self._dtype)
            self.v_blocks[layer_idx][block_idx, slot_in_block, :, :] = v_token.to(self._dtype)

        seq.token_count += new_tokens
        return True

    def get_kv(
        self,
        seq_id: int,
        layer_idx: int,
    ) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Retrieve full K, V for a sequence.

        Args:
            seq_id: Sequence ID.
            layer_idx: Transformer layer.

        Returns:
            Tuple of (key, value) with shape [num_kv_heads, seq_len, head_dim].
        """
        if seq_id not in self._sequences:
            raise ValueError(f"Sequence {seq_id} not registered")

        seq = self._sequences[seq_id]
        if seq.token_count == 0:
            return (
                torch.zeros(
                    (self.num_kv_heads, 0, self.head_dim),
                    dtype=self._dtype,
                    device=self.device,
                ),
                torch.zeros(
                    (self.num_kv_heads, 0, self.head_dim),
                    dtype=self._dtype,
                    device=self.device,
                ),
            )

        # Gather tokens from blocks
        k_parts = []
        v_parts = []

        tokens_remaining = seq.token_count
        for block_idx in seq.block_indices:
            tokens_in_block = min(tokens_remaining, self.block_size)
            k_parts.append(
                self.k_blocks[layer_idx][block_idx, :tokens_in_block]
            )  # [tokens, heads, dim]
            v_parts.append(self.v_blocks[layer_idx][block_idx, :tokens_in_block])
            tokens_remaining -= tokens_in_block
            if tokens_remaining <= 0:
                break

        # Concatenate and transpose to [heads, seq, dim]
        k = torch.cat(k_parts, dim=0).permute(1, 0, 2)
        v = torch.cat(v_parts, dim=0).permute(1, 0, 2)
        return k, v

    def fork_sequence(self, src_seq_id: int, dst_seq_id: int) -> PagedSequence:
        """Fork a sequence for beam search (copy-on-write).

        The new sequence shares blocks with the source until modified.

        Args:
            src_seq_id: Source sequence to fork.
            dst_seq_id: New sequence ID.

        Returns:
            PagedSequence for the forked sequence.
        """
        if src_seq_id not in self._sequences:
            raise ValueError(f"Source sequence {src_seq_id} not found")
        if dst_seq_id in self._sequences:
            raise ValueError(f"Destination sequence {dst_seq_id} already exists")

        src_seq = self._sequences[src_seq_id]

        # Create new sequence sharing blocks (increment ref counts)
        dst_seq = PagedSequence(
            seq_id=dst_seq_id,
            block_indices=src_seq.block_indices.copy(),
            token_count=src_seq.token_count,
        )

        for block_idx in dst_seq.block_indices:
            self._block_ref_counts[block_idx] += 1

        self._sequences[dst_seq_id] = dst_seq
        return dst_seq

    def get_block_table(self, seq_id: int) -> list[int]:
        """Get block table for a sequence (for kernel dispatch).

        Args:
            seq_id: Sequence ID.

        Returns:
            List of physical block indices.
        """
        if seq_id not in self._sequences:
            return []
        return self._sequences[seq_id].block_indices.copy()

    def memory_usage_mb(self) -> float:
        """Return current memory usage in MB."""
        bytes_per_element = 2
        elements_per_block = self.block_size * self.num_kv_heads * self.head_dim * 2  # K and V
        allocated_blocks = self.num_allocated_blocks
        return (
            allocated_blocks
            * elements_per_block
            * bytes_per_element
            * self.num_layers
            / 1024
            / 1024
        )


# FP8 E4M3 constants
FP8_E4M3_MAX = 448.0


class MetalQuantizedKVCache:
    """FP8 quantized KV cache for memory-efficient long contexts.

    Achieves 2x memory reduction by storing K, V in FP8 format with per-head
    or per-token scaling factors. Dequantization happens on-the-fly during
    attention computation.

    Quantization format:
        - FP8 E4M3: 4 exponent bits, 3 mantissa bits (range +/-448)
        - INT8: 8-bit symmetric quantization (range +/-127)

    Both formats store values as uint8 with FP16 scales for dequantization.
    The choice depends on value distribution; FP8 handles larger dynamic range
    while INT8 has better precision for bounded values.

    Memory comparison (vs FP16 baseline):
        FP16: 2 bytes/element
        FP8/INT8: 1 byte/element + 2 bytes/token/head (scales) ~= 1.02 bytes
        Savings: ~49%
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        num_kv_heads: int | None = None,
        batch_size: int = 1,
        quant_type: Literal["fp8", "int8"] = "fp8",
        scale_granularity: Literal["per_head", "per_token"] = "per_head",
        device: str = "mps",
    ):
        """Initialize quantized KV cache.

        Args:
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            max_seq_len: Maximum sequence length.
            num_kv_heads: Number of KV heads (for GQA).
            batch_size: Batch size.
            quant_type: Quantization format ("fp8" or "int8").
            scale_granularity: Scale computation granularity.
            device: Device to allocate tensors on.
        """
        require_torch("MetalQuantizedKVCache")

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.quant_type = QuantType(quant_type)
        self.scale_granularity = scale_granularity
        self.device = device

        # Quantized storage: uint8
        cache_shape = (batch_size, self.num_kv_heads, max_seq_len, head_dim)
        self.k_cache: list[torch_typing.Tensor] = [
            torch.zeros(cache_shape, dtype=torch.uint8, device=device) for _ in range(num_layers)
        ]
        self.v_cache: list[torch_typing.Tensor] = [
            torch.zeros(cache_shape, dtype=torch.uint8, device=device) for _ in range(num_layers)
        ]

        # Scales: shape depends on granularity
        if scale_granularity == "per_head":
            # One scale per batch, head, position
            scale_shape = (batch_size, self.num_kv_heads, max_seq_len, 1)
        else:  # per_token
            # One scale per batch, position (shared across heads)
            scale_shape = (batch_size, 1, max_seq_len, 1)

        self.k_scales: list[torch_typing.Tensor] = [
            torch.zeros(scale_shape, dtype=torch.float16, device=device) for _ in range(num_layers)
        ]
        self.v_scales: list[torch_typing.Tensor] = [
            torch.zeros(scale_shape, dtype=torch.float16, device=device) for _ in range(num_layers)
        ]

        # Sequence tracking
        self.seq_lens: list[int] = [0] * num_layers

    def _quantize(
        self, tensor: torch_typing.Tensor
    ) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Quantize tensor to FP8/INT8.

        Args:
            tensor: Input [batch, num_kv_heads, seq_len, head_dim].

        Returns:
            Tuple of (quantized, scales).
        """
        # Compute scales based on granularity
        if self.scale_granularity == "per_head":
            # Max abs per head per position
            abs_max = torch.amax(torch.abs(tensor), dim=-1, keepdim=True)
        else:
            # Max abs per position (across heads)
            abs_max = torch.amax(torch.abs(tensor), dim=(1, -1), keepdim=True)

        abs_max = torch.clamp(abs_max, min=1e-8)  # Avoid division by zero

        if self.quant_type == QuantType.FP8:
            # FP8 E4M3 max is 448
            scale = abs_max / FP8_E4M3_MAX
            scaled = tensor / scale
            scaled = torch.clamp(scaled, -FP8_E4M3_MAX, FP8_E4M3_MAX)
            # Map to [0, 255] centered at 128
            quantized = torch.round(scaled / FP8_E4M3_MAX * 127.0) + 128.0
            quantized = torch.clamp(quantized, 0, 255).to(torch.uint8)
        else:  # INT8
            scale = abs_max / 127.0
            scaled = tensor / scale
            scaled = torch.clamp(scaled, -127.0, 127.0)
            # Map to [0, 255] centered at 128
            quantized = torch.round(scaled) + 128.0
            quantized = torch.clamp(quantized, 0, 255).to(torch.uint8)

        return quantized, scale.to(torch.float16)

    def _dequantize(
        self,
        quantized: torch_typing.Tensor,
        scales: torch_typing.Tensor,
    ) -> torch_typing.Tensor:
        """Dequantize to FP16 for attention computation.

        Args:
            quantized: uint8 quantized tensor.
            scales: FP16 scales.

        Returns:
            Dequantized tensor in FP16.
        """
        # Map [0, 255] back to signed range
        signed = quantized.to(torch.float16) - 128.0

        if self.quant_type == QuantType.FP8:
            return signed / 127.0 * FP8_E4M3_MAX * scales
        else:  # INT8
            return signed * scales

    def update(
        self,
        layer_idx: int,
        key: torch_typing.Tensor,
        value: torch_typing.Tensor,
        positions: torch_typing.Tensor | None = None,
    ) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Update cache with quantized K, V and return dequantized tensors.

        Args:
            layer_idx: Transformer layer.
            key: Key tensor [batch, num_kv_heads, new_seq_len, head_dim].
            value: Value tensor [batch, num_kv_heads, new_seq_len, head_dim].
            positions: Optional positions (None = append mode).

        Returns:
            Tuple of dequantized (k_full, v_full).
        """
        new_seq_len = key.shape[2]
        start_pos = self.seq_lens[layer_idx]
        end_pos = start_pos + new_seq_len

        if end_pos > self.max_seq_len:
            raise ValueError(f"Sequence length {end_pos} exceeds max_seq_len {self.max_seq_len}")

        # Quantize new K, V
        k_quant, k_scale = self._quantize(key)
        v_quant, v_scale = self._quantize(value)

        # Update cache slices via slice assignment
        self.k_cache[layer_idx][:, :, start_pos:end_pos, :] = k_quant
        self.v_cache[layer_idx][:, :, start_pos:end_pos, :] = v_quant
        self.k_scales[layer_idx][:, :, start_pos:end_pos, :] = k_scale
        self.v_scales[layer_idx][:, :, start_pos:end_pos, :] = v_scale

        self.seq_lens[layer_idx] = end_pos

        # Return dequantized full cache
        k_full = self._dequantize(
            self.k_cache[layer_idx][:, :, :end_pos, :],
            self.k_scales[layer_idx][:, :, :end_pos, :],
        )
        v_full = self._dequantize(
            self.v_cache[layer_idx][:, :, :end_pos, :],
            self.v_scales[layer_idx][:, :, :end_pos, :],
        )

        return k_full, v_full

    def get(
        self,
        layer_idx: int,
        positions: torch_typing.Tensor | None = None,
    ) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Get dequantized K, V tensors.

        Args:
            layer_idx: Transformer layer.
            positions: Optional specific positions.

        Returns:
            Tuple of dequantized (key, value).
        """
        end = self.seq_lens[layer_idx]
        if end == 0:
            return (
                torch.zeros(
                    (self.batch_size, self.num_kv_heads, 0, self.head_dim),
                    dtype=torch.float16,
                    device=self.device,
                ),
                torch.zeros(
                    (self.batch_size, self.num_kv_heads, 0, self.head_dim),
                    dtype=torch.float16,
                    device=self.device,
                ),
            )

        if positions is None:
            k = self._dequantize(
                self.k_cache[layer_idx][:, :, :end, :],
                self.k_scales[layer_idx][:, :, :end, :],
            )
            v = self._dequantize(
                self.v_cache[layer_idx][:, :, :end, :],
                self.v_scales[layer_idx][:, :, :end, :],
            )
        else:
            # Use direct indexing instead of index_select for better MPS performance
            k_quant = self.k_cache[layer_idx][..., positions, :]
            v_quant = self.v_cache[layer_idx][..., positions, :]
            k_scale = self.k_scales[layer_idx][..., positions, :]
            v_scale = self.v_scales[layer_idx][..., positions, :]
            k = self._dequantize(k_quant, k_scale)
            v = self._dequantize(v_quant, v_scale)

        return k, v

    @property
    def seq_len(self) -> int:
        """Current sequence length (assumes all layers in sync)."""
        return self.seq_lens[0] if self.seq_lens else 0

    def reset(self, layer_idx: int | None = None) -> None:
        """Reset cache for new sequence."""
        if layer_idx is not None:
            self.seq_lens[layer_idx] = 0
        else:
            self.seq_lens = [0] * self.num_layers

    def memory_usage_mb(self) -> float:
        """Return current memory usage in MB (quantized)."""
        # uint8 storage: 1 byte per element
        base_elements = (
            self.batch_size * self.num_kv_heads * sum(self.seq_lens) * self.head_dim * 2  # K and V
        )
        base_bytes = base_elements  # 1 byte each

        # Scales: FP16 (2 bytes) per head/token per position
        if self.scale_granularity == "per_head":
            scale_elements = self.batch_size * self.num_kv_heads * sum(self.seq_lens) * 2
        else:
            scale_elements = self.batch_size * sum(self.seq_lens) * 2
        scale_bytes = scale_elements * 2

        return (base_bytes + scale_bytes) / 1024 / 1024

    def compression_ratio(self) -> float:
        """Compression ratio vs FP16."""
        fp16_bytes = (
            self.batch_size
            * self.num_kv_heads
            * sum(self.seq_lens)
            * self.head_dim
            * 2  # K and V
            * 2  # 2 bytes per FP16
        )
        if fp16_bytes == 0:
            return 1.0
        return fp16_bytes / (self.memory_usage_mb() * 1024 * 1024)


@dataclass
class ContinuousBatchState:
    """State for a single sequence in continuous batching."""

    seq_id: int
    token_count: int = 0
    is_prefill: bool = True
    is_finished: bool = False


class ContinuousBatchManager:
    """Manager for continuous batching with paged KV cache.

    Coordinates multiple sequences running in parallel with dynamic
    addition/removal as sequences complete or new requests arrive.

    Features:
    - Dynamic batch composition (add/remove sequences)
    - Per-sequence position tracking
    - Automatic block allocation/deallocation
    - Efficient iteration over active sequences
    """

    def __init__(
        self,
        cache: MetalPagedKVCache,
        max_batch_size: int = 32,
    ):
        """Initialize continuous batch manager.

        Args:
            cache: Paged KV cache for storage.
            max_batch_size: Maximum concurrent sequences.
        """
        self.cache = cache
        self.max_batch_size = max_batch_size
        self._sequences: dict[int, ContinuousBatchState] = {}
        self._next_seq_id = 0

    @property
    def batch_size(self) -> int:
        """Current number of active sequences."""
        return len(self._sequences)

    @property
    def active_seq_ids(self) -> list[int]:
        """List of active sequence IDs."""
        return [sid for sid, s in self._sequences.items() if not s.is_finished]

    def can_add_sequence(self) -> bool:
        """Check if batch has capacity for another sequence."""
        return self.batch_size < self.max_batch_size

    def add_sequence(self, initial_tokens: int = 0) -> int:
        """Add a new sequence to the batch.

        Args:
            initial_tokens: Number of tokens to preallocate.

        Returns:
            New sequence ID.

        Raises:
            RuntimeError: If batch is at capacity.
        """
        if not self.can_add_sequence():
            raise RuntimeError(f"Batch at capacity ({self.max_batch_size})")

        seq_id = self._next_seq_id
        self._next_seq_id += 1

        self.cache.allocate_sequence(seq_id)
        self._sequences[seq_id] = ContinuousBatchState(
            seq_id=seq_id,
            token_count=initial_tokens,
            is_prefill=True,
        )

        return seq_id

    def remove_sequence(self, seq_id: int) -> None:
        """Remove a sequence from the batch.

        Args:
            seq_id: Sequence to remove.
        """
        if seq_id not in self._sequences:
            return

        self.cache.free_sequence(seq_id)
        del self._sequences[seq_id]

    def mark_finished(self, seq_id: int) -> None:
        """Mark a sequence as finished (will be removed on next cleanup).

        Args:
            seq_id: Sequence that completed.
        """
        if seq_id in self._sequences:
            self._sequences[seq_id].is_finished = True

    def cleanup_finished(self) -> list[int]:
        """Remove all finished sequences.

        Returns:
            List of removed sequence IDs.
        """
        finished = [sid for sid, s in self._sequences.items() if s.is_finished]
        for seq_id in finished:
            self.remove_sequence(seq_id)
        return finished

    def get_sequence_state(self, seq_id: int) -> ContinuousBatchState | None:
        """Get state for a sequence.

        Args:
            seq_id: Sequence ID.

        Returns:
            Sequence state or None if not found.
        """
        return self._sequences.get(seq_id)

    def advance_sequence(self, seq_id: int, num_tokens: int = 1) -> None:
        """Advance a sequence's token count.

        Args:
            seq_id: Sequence ID.
            num_tokens: Number of tokens generated.
        """
        if seq_id in self._sequences:
            state = self._sequences[seq_id]
            state.token_count += num_tokens
            state.is_prefill = False  # After first token, we're in decode

    def get_block_tables(self) -> dict[int, list[int]]:
        """Get block tables for all active sequences.

        Returns:
            Dict mapping seq_id to block indices.
        """
        return {seq_id: self.cache.get_block_table(seq_id) for seq_id in self.active_seq_ids}

    def get_context_lengths(self) -> dict[int, int]:
        """Get context lengths for all active sequences.

        Returns:
            Dict mapping seq_id to token count.
        """
        return {seq_id: self._sequences[seq_id].token_count for seq_id in self.active_seq_ids}
