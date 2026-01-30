"""Specialized KV cache for Multi-Head Latent Attention (MLA) models.

MLA (used by DeepSeek-V2, GLM-4) compresses KV cache representations:
- Standard MHA: KV cache = O(n_layers * seq_len * n_heads * head_dim)
- MLA: KV cache = O(n_layers * seq_len * kv_lora_rank)

Key insight: Store compressed latents directly and decompress on-demand during
attention. This provides ~8x memory reduction compared to standard KV cache.

Memory comparison (seq_len=4096, n_heads=32, head_dim=128, dtype=fp16):
- Standard: 4096 * 32 * 128 * 2 = 32 MB per layer
- MLA (rank=512): 4096 * 512 * 2 = 4 MB per layer

Components:
- `kv_a_proj_with_mqa`: Compress hidden states to latent space
- `kv_b_proj`: Decompress from latent for attention (on-demand)

Usage:
    from metal_marlin.paged.mla_cache import (
        MLACacheConfig,
        MLABlock,
        MLABlockAllocator,
        mla_attention,
    )

    config = MLACacheConfig(
        block_size=16,
        kv_lora_rank=512,  # Latent dimension
        num_kv_heads=8,    # Final KV heads after decompression
        head_dim=128,
    )

    block = MLABlock(config)
    block.allocate()
    block.append_latent(compressed_kv)  # [kv_lora_rank]

    # During attention: decompress on-demand
    k, v = block.decompress(kv_b_proj_weight)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax


@dataclass(frozen=True)
class MLACacheConfig:
    """Immutable configuration for MLA KV blocks.

    MLA stores compressed latent representations instead of full KV pairs.
    The latent dimension (kv_lora_rank) is typically much smaller than
    num_kv_heads * head_dim.

    Attributes:
        block_size: Tokens per block (default 16, same as vLLM).
        kv_lora_rank: Latent/compressed dimension. Common values:
            - GLM-4-9B: 512
            - DeepSeek-V2-Lite: 512
            - DeepSeek-V2: 1536
        num_kv_heads: Number of KV heads after decompression.
        head_dim: Dimension per head after decompression.
        rope_head_dim: Optional RoPE dimension applied in latent space.
            Some MLA variants apply RoPE before decompression.
        quantize_mode: Compression for latent storage.
            - "none": Full precision (bf16/fp16)
            - "fp8": 2x compression
            - "fp4": 4x compression (experimental)
        dtype: Storage dtype (ignored if quantize_mode != "none").
    """

    block_size: int = 16
    kv_lora_rank: int = 512
    num_kv_heads: int = 8
    head_dim: int = 128
    rope_head_dim: int = 0  # 0 means no RoPE in latent space
    quantize_mode: Literal["none", "fp8", "fp4"] = "none"
    # Use field with factory to avoid mutable default
    _dtype_str: str = field(default="bf16", repr=False)

    @property
    def dtype(self) -> np.dtype:
        """Get numpy dtype for storage."""
        dtype_map = {"fp16": np.float16, "bf16": np.float16, "fp32": np.float32}
        return np.dtype(dtype_map.get(self._dtype_str, np.float16))

    @property
    def latent_bytes_per_token(self) -> int:
        """Bytes per token in latent space."""
        if self.quantize_mode == "fp4":
            return self.kv_lora_rank // 2  # 4 bits per value
        elif self.quantize_mode == "fp8":
            return self.kv_lora_rank  # 8 bits per value
        else:
            return self.kv_lora_rank * 2  # 16 bits per value

    @property
    def standard_bytes_per_token(self) -> int:
        """Bytes per token if using standard KV cache (for comparison)."""
        # K and V, each [num_kv_heads, head_dim], 16-bit
        return 2 * self.num_kv_heads * self.head_dim * 2

    @property
    def memory_savings_ratio(self) -> float:
        """Memory savings compared to standard KV cache."""
        return self.standard_bytes_per_token / self.latent_bytes_per_token

    @property
    def memory_bytes(self) -> int:
        """Memory footprint of one block in bytes."""
        return self.block_size * self.latent_bytes_per_token

    @property
    def latent_shape(self) -> tuple[int, int]:
        """Storage shape per block: [block_size, kv_lora_rank]."""
        return (self.block_size, self.kv_lora_rank)

    @property
    def decompressed_shape(self) -> tuple[int, int, int, int]:
        """Shape after decompression: [2, block_size, num_kv_heads, head_dim]."""
        return (2, self.block_size, self.num_kv_heads, self.head_dim)


class MLABlock:
    """Fixed-size block for MLA compressed latent storage.

    Stores compressed KV representations (latents) and decompresses on-demand
    during attention computation. This enables 4-8x memory reduction compared
    to standard KV cache.

    Storage layout: [block_size, kv_lora_rank]
    - Each row is the compressed representation for one token
    - Decompression uses kv_b_proj to produce K and V

    The block supports copy-on-write semantics for beam search and
    speculative decoding via reference counting.
    """

    __slots__ = (
        "config",
        "_latents",
        "_scales",  # For quantized modes
        "_token_count",
        "_ref_count",
        "_prefix_hash",  # For prefix caching
    )

    def __init__(self, config: MLACacheConfig | None = None) -> None:
        self.config = config or MLACacheConfig()
        self._latents: NDArray[Any] | None = None
        self._scales: NDArray[Any] | None = None
        self._token_count: int = 0
        self._ref_count: int = 0
        self._prefix_hash: int | None = None

    def allocate(self) -> None:
        """Allocate block memory for latent storage."""
        if self.config.quantize_mode == "fp4":
            # Pack 8 FP4 values per uint32
            packed_dim = self.config.kv_lora_rank // 8
            self._latents = np.zeros((self.config.block_size, packed_dim), dtype=np.uint32)
            # Per-token scales
            self._scales = np.zeros((self.config.block_size, 1), dtype=np.float16)
        elif self.config.quantize_mode == "fp8":
            self._latents = np.zeros(self.config.latent_shape, dtype=np.uint8)
            self._scales = np.zeros((self.config.block_size, 1), dtype=np.float16)
        else:
            self._latents = np.zeros(self.config.latent_shape, dtype=self.config.dtype)
        self._token_count = 0

    @property
    def latents(self) -> NDArray[Any] | None:
        """The underlying latent storage array."""
        return self._latents

    @property
    def token_count(self) -> int:
        """Number of tokens currently stored."""
        return self._token_count

    @property
    def ref_count(self) -> int:
        """Reference count for copy-on-write."""
        return self._ref_count

    @property
    def prefix_hash(self) -> int | None:
        """Hash of the prefix for prefix caching."""
        return self._prefix_hash

    def set_prefix_hash(self, h: int) -> None:
        """Set the prefix hash for this block."""
        self._prefix_hash = h

    def acquire(self) -> None:
        """Increment reference count."""
        self._ref_count += 1

    def release(self) -> int:
        """Decrement reference count. Returns new count."""
        self._ref_count = max(0, self._ref_count - 1)
        return self._ref_count

    def append_latent(self, latent: NDArray[Any]) -> int:
        """Append compressed latent for a single token.

        Args:
            latent: Compressed representation of shape [kv_lora_rank].
                This is the output of kv_a_proj_with_mqa.

        Returns:
            Number of slots remaining in block.

        Raises:
            RuntimeError: If block is full or not allocated.
        """
        if self._latents is None:
            raise RuntimeError("Block not allocated")
        if self._token_count >= self.config.block_size:
            raise RuntimeError("Block is full")

        idx = self._token_count

        if self.config.quantize_mode == "fp4":
            packed, scale = self._quantize_fp4(latent)
            self._latents[idx] = packed
            self._scales[idx] = scale
        elif self.config.quantize_mode == "fp8":
            quant, scale = self._quantize_fp8(latent)
            self._latents[idx] = quant
            self._scales[idx] = scale
        else:
            # Full precision: direct storage
            self._latents[idx] = latent.astype(self.config.dtype)

        self._token_count += 1
        return self.config.block_size - self._token_count

    def append_latent_batch(self, latents: NDArray[Any]) -> int:
        """Append multiple token latents at once.

        More efficient than repeated single appends.

        Args:
            latents: Compressed representations [num_tokens, kv_lora_rank].

        Returns:
            Number of slots remaining in block.

        Raises:
            RuntimeError: If batch exceeds remaining capacity.
        """
        if self._latents is None:
            raise RuntimeError("Block not allocated")

        num_tokens = latents.shape[0]
        if self._token_count + num_tokens > self.config.block_size:
            raise RuntimeError(
                f"Batch of {num_tokens} tokens exceeds remaining "
                f"{self.config.block_size - self._token_count} slots"
            )

        idx = self._token_count
        end = idx + num_tokens

        if self.config.quantize_mode == "fp4":
            packed, scales = self._quantize_fp4_batch(latents)
            self._latents[idx:end] = packed
            self._scales[idx:end] = scales
        elif self.config.quantize_mode == "fp8":
            quant, scales = self._quantize_fp8_batch(latents)
            self._latents[idx:end] = quant
            self._scales[idx:end] = scales
        else:
            self._latents[idx:end] = latents.astype(self.config.dtype)

        self._token_count += num_tokens
        return self.config.block_size - self._token_count

    def get_latents(self) -> NDArray[Any]:
        """Get the filled portion of latents (dequantized if needed).

        Returns:
            Latents of shape [token_count, kv_lora_rank].

        Raises:
            RuntimeError: If block is not allocated.
        """
        if self._latents is None:
            raise RuntimeError("Block not allocated")

        raw = self._latents[: self._token_count]

        if self.config.quantize_mode == "fp4":
            return self._dequant_fp4(raw, self._scales[: self._token_count])
        elif self.config.quantize_mode == "fp8":
            return self._dequant_fp8(raw, self._scales[: self._token_count])
        else:
            return raw

    def decompress(self, kv_b_proj: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        """Decompress latents to full K, V using the decompression projection.

        This is the key MLA operation: latents @ kv_b_proj.T -> [K, V]

        Args:
            kv_b_proj: Decompression weight matrix of shape
                [2 * num_kv_heads * head_dim, kv_lora_rank].
                The first half produces K, second half produces V.

        Returns:
            Tuple of (keys, values):
                - keys: [token_count, num_kv_heads, head_dim]
                - values: [token_count, num_kv_heads, head_dim]

        Raises:
            RuntimeError: If block is not allocated.
        """
        if self._latents is None:
            raise RuntimeError("Block not allocated")

        latents = self.get_latents()  # [token_count, kv_lora_rank]

        # Decompress: latents @ kv_b_proj.T
        # kv_b_proj: [2 * num_kv_heads * head_dim, kv_lora_rank]
        # Result: [token_count, 2 * num_kv_heads * head_dim]
        kv_decompressed = latents @ kv_b_proj.T

        # Split into K and V
        split_dim = self.config.num_kv_heads * self.config.head_dim
        k_flat = kv_decompressed[:, :split_dim]
        v_flat = kv_decompressed[:, split_dim:]

        # Reshape to [token_count, num_kv_heads, head_dim]
        k = k_flat.reshape(self._token_count, self.config.num_kv_heads, self.config.head_dim)
        v = v_flat.reshape(self._token_count, self.config.num_kv_heads, self.config.head_dim)

        return k, v

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
        if self._latents is not None:
            self._latents.fill(0)
            if self._scales is not None:
                self._scales.fill(0)
        self._token_count = 0
        self._ref_count = 0
        self._prefix_hash = None

    def copy(self) -> MLABlock:
        """Create an independent copy of this block (for CoW)."""
        new_block = MLABlock(config=self.config)
        if self._latents is not None:
            new_block._latents = self._latents.copy()
            if self._scales is not None:
                new_block._scales = self._scales.copy()
            new_block._token_count = self._token_count
            new_block._prefix_hash = self._prefix_hash
        return new_block

    # --------------------------------------------------------------------------
    # Quantization helpers
    # --------------------------------------------------------------------------

    def _quantize_fp4(self, latent: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        """Quantize single latent vector to FP4."""
        abs_max = np.max(np.abs(latent))
        abs_max = max(abs_max, 1e-8)
        scale = abs_max / 6.0  # FP4 E2M1 max is 6.0

        scaled = latent / scale
        scaled = np.clip(scaled, -6.0, 6.0)
        quantized = np.round(scaled * 2.0).astype(np.int8)
        quantized = np.clip(quantized + 8, 0, 15).astype(np.uint8)

        # Pack 8 values per uint32
        packed_dim = self.config.kv_lora_rank // 8
        reshaped = quantized.reshape(packed_dim, 8)
        packed = np.zeros((packed_dim,), dtype=np.uint32)
        for i in range(8):
            packed = packed | (reshaped[:, i].astype(np.uint32) << (i * 4))

        return packed, np.array([scale], dtype=np.float16)

    def _quantize_fp4_batch(self, latents: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        """Quantize batch of latents to FP4."""
        num_tokens = latents.shape[0]
        abs_max = np.max(np.abs(latents), axis=-1, keepdims=True)
        abs_max = np.maximum(abs_max, 1e-8)
        scales = abs_max / 6.0

        scaled = latents / scales
        scaled = np.clip(scaled, -6.0, 6.0)
        quantized = np.round(scaled * 2.0).astype(np.int8)
        quantized = np.clip(quantized + 8, 0, 15).astype(np.uint8)

        # Pack 8 values per uint32
        packed_dim = self.config.kv_lora_rank // 8
        reshaped = quantized.reshape(num_tokens, packed_dim, 8)
        packed = np.zeros((num_tokens, packed_dim), dtype=np.uint32)
        for i in range(8):
            packed = packed | (reshaped[:, :, i].astype(np.uint32) << (i * 4))

        return packed, scales.astype(np.float16)

    def _dequant_fp4(self, packed: NDArray[Any], scales: NDArray[Any]) -> NDArray[Any]:
        """Dequantize FP4 packed latents."""
        batch_size = packed.shape[0]
        packed_dim = packed.shape[1]
        full_dim = packed_dim * 8

        # Unpack
        unpacked = []
        for i in range(8):
            nibble = (packed >> (i * 4)) & 0xF
            signed = nibble.astype(np.float16) - 8.0
            unpacked.append(signed)

        # Interleave properly
        result = np.stack(unpacked, axis=-1).reshape(batch_size, full_dim)
        return result * scales / 2.0

    def _quantize_fp8(self, latent: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        """Quantize single latent vector to FP8 (simulated)."""
        abs_max = np.max(np.abs(latent))
        abs_max = max(abs_max, 1e-8)
        scale = abs_max / 448.0  # E4M3 max

        scaled = latent / scale
        scaled = np.clip(scaled, -448.0, 448.0)
        quantized = np.round(scaled / 448.0 * 127.0) + 128.0
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)

        return quantized, np.array([scale], dtype=np.float16)

    def _quantize_fp8_batch(self, latents: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        """Quantize batch of latents to FP8."""
        abs_max = np.max(np.abs(latents), axis=-1, keepdims=True)
        abs_max = np.maximum(abs_max, 1e-8)
        scales = abs_max / 448.0

        scaled = latents / scales
        scaled = np.clip(scaled, -448.0, 448.0)
        quantized = np.round(scaled / 448.0 * 127.0) + 128.0
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)

        return quantized, scales.astype(np.float16)

    def _dequant_fp8(self, quantized: NDArray[Any], scales: NDArray[Any]) -> NDArray[Any]:
        """Dequantize FP8 latents."""
        signed = quantized.astype(np.float16) - 128.0
        return signed / 127.0 * 448.0 * scales

    def __repr__(self) -> str:
        return (
            f"MLABlock(tokens={self._token_count}/{self.config.block_size}, "
            f"rank={self.config.kv_lora_rank}, "
            f"refs={self._ref_count}, "
            f"mode={self.config.quantize_mode}, "
            f"allocated={'yes' if self._latents is not None else 'no'})"
        )


@dataclass
class MLABlockState:
    """Metadata for a single physical MLA block."""

    block_idx: int
    ref_count: int = 0
    is_free: bool = True
    prefix_hash: int | None = None


class MLABlockAllocator:
    """Block allocator for MLA paged cache with prefix caching support.

    Manages a pool of fixed-size MLA blocks. Supports:
    - Reference counting for copy-on-write (beam search, speculative decoding)
    - Prefix caching: reuse blocks with matching prefix hashes

    Args:
        num_blocks: Total number of physical blocks in the pool.
        config: MLACacheConfig for block parameters.
    """

    def __init__(self, num_blocks: int, config: MLACacheConfig | None = None):
        self.num_blocks = num_blocks
        self.config = config or MLACacheConfig()

        self.blocks: list[MLABlockState] = [MLABlockState(block_idx=i) for i in range(num_blocks)]
        self._free_list: list[int] = list(range(num_blocks - 1, -1, -1))

        # Prefix cache: hash -> block_idx
        # When a sequence completes, its blocks can be added to prefix cache
        # for reuse by sequences with matching prefixes
        self._prefix_cache: dict[int, int] = {}

        # Actual storage for blocks (allocated lazily)
        self._storage: list[MLABlock | None] = [None] * num_blocks

    @property
    def num_free(self) -> int:
        return len(self._free_list)

    @property
    def num_allocated(self) -> int:
        return self.num_blocks - self.num_free

    @property
    def num_cached_prefixes(self) -> int:
        return len(self._prefix_cache)

    def allocate(self) -> int | None:
        """Allocate a single block. Returns block index or None if OOM."""
        if not self._free_list:
            return None
        idx = self._free_list.pop()
        self.blocks[idx].is_free = False
        self.blocks[idx].ref_count = 1
        self.blocks[idx].prefix_hash = None

        # Lazy allocation of actual storage
        if self._storage[idx] is None:
            block = MLABlock(self.config)
            block.allocate()
            self._storage[idx] = block
        else:
            self._storage[idx].reset()

        return idx

    def free(self, block_idx: int) -> None:
        """Decrement ref_count; return block to pool when it reaches zero."""
        state = self.blocks[block_idx]
        state.ref_count -= 1
        if state.ref_count <= 0:
            state.ref_count = 0
            state.is_free = True

            # Remove from prefix cache if present
            if state.prefix_hash is not None and state.prefix_hash in self._prefix_cache:
                if self._prefix_cache[state.prefix_hash] == block_idx:
                    del self._prefix_cache[state.prefix_hash]
            state.prefix_hash = None

            self._free_list.append(block_idx)

    def get_block(self, block_idx: int) -> MLABlock | None:
        """Get the MLABlock at the given index."""
        return self._storage[block_idx]

    def copy_on_write(self, block_idx: int) -> int | None:
        """If block is shared, allocate a new block for exclusive write.

        Returns the new block index (or original if already exclusive).
        Returns None if a copy is needed but pool is exhausted.
        """
        state = self.blocks[block_idx]
        if state.ref_count == 1:
            return block_idx  # Already exclusive

        new_idx = self.allocate()
        if new_idx is None:
            return None

        # Copy block contents
        old_block = self._storage[block_idx]
        new_block = self._storage[new_idx]
        if old_block is not None and new_block is not None:
            new_block._latents = (
                old_block._latents.copy() if old_block._latents is not None else None
            )
            if old_block._scales is not None:
                new_block._scales = old_block._scales.copy()
            new_block._token_count = old_block._token_count

        # Decrement old block's ref count
        state.ref_count -= 1
        return new_idx

    # --------------------------------------------------------------------------
    # Prefix caching
    # --------------------------------------------------------------------------

    def register_prefix(self, block_idx: int, prefix_hash: int) -> None:
        """Register a block as cacheable with the given prefix hash.

        Call this when a block is complete and its prefix can be reused.
        """
        state = self.blocks[block_idx]
        state.prefix_hash = prefix_hash
        self._prefix_cache[prefix_hash] = block_idx

        # Store hash in the block itself for verification
        block = self._storage[block_idx]
        if block is not None:
            block.set_prefix_hash(prefix_hash)

    def lookup_prefix(self, prefix_hash: int) -> int | None:
        """Look up a block by prefix hash.

        Returns block_idx if found (and acquires it), None otherwise.
        """
        if prefix_hash not in self._prefix_cache:
            return None

        block_idx = self._prefix_cache[prefix_hash]
        state = self.blocks[block_idx]

        # Verify block is still valid and has matching content
        if state.is_free:
            # Block was freed, remove stale entry
            del self._prefix_cache[prefix_hash]
            return None

        # Acquire the block (increment ref count)
        state.ref_count += 1
        return block_idx

    def memory_usage_mb(self) -> float:
        """Return current memory usage in MB."""
        allocated = sum(1 for block in self._storage if block is not None)
        return allocated * self.config.memory_bytes / 1024 / 1024

    def memory_usage_stats(self) -> dict[str, Any]:
        """Return detailed memory usage statistics."""
        standard_cache_bytes = (
            self.num_allocated * self.config.block_size * self.config.standard_bytes_per_token
        )
        mla_cache_bytes = (
            self.num_allocated * self.config.block_size * self.config.latent_bytes_per_token
        )

        return {
            "allocated_blocks": self.num_allocated,
            "free_blocks": self.num_free,
            "cached_prefixes": self.num_cached_prefixes,
            "mla_cache_mb": mla_cache_bytes / 1024 / 1024,
            "standard_cache_mb": standard_cache_bytes / 1024 / 1024,
            "memory_savings_ratio": self.config.memory_savings_ratio,
            "bytes_saved_mb": (standard_cache_bytes - mla_cache_bytes) / 1024 / 1024,
        }


def mla_attention(
    query: NDArray[Any],
    latent_pool: NDArray[Any],
    block_tables: NDArray[Any],
    context_lens: NDArray[Any],
    kv_b_proj: NDArray[Any],
    scale: float | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    block_size: int = 16,
) -> NDArray[Any]:
    """Compute scaled dot-product attention with MLA paged latent cache.

    This performs on-demand decompression: latents are decompressed to K, V
    only for the tokens needed in the attention computation.

    Args:
        query: Query tensor [num_seqs, num_heads, seq_len, head_dim].
        latent_pool: Pre-allocated latent storage
            [num_blocks, block_size, kv_lora_rank].
        block_tables: Block indices per sequence [num_seqs, max_blocks_per_seq].
        context_lens: Number of valid KV tokens per sequence [num_seqs].
        kv_b_proj: Decompression weight [2 * num_kv_heads * head_dim, kv_lora_rank].
        scale: Attention scale factor (default: head_dim ** -0.5).
        num_kv_heads: Number of KV heads (inferred from kv_b_proj if not provided).
        head_dim: Head dimension (inferred from query if not provided).
        block_size: Tokens per block.

    Returns:
        Attention output [num_seqs, num_heads, seq_len, head_dim].
    """
    num_seqs = query.shape[0]
    num_heads = query.shape[1]
    seq_len = query.shape[2]
    if head_dim is None:
        head_dim = query.shape[3]

    # Infer num_kv_heads from projection shape
    kv_lora_rank = kv_b_proj.shape[1]
    if num_kv_heads is None:
        num_kv_heads = kv_b_proj.shape[0] // (2 * head_dim)

    max_blocks = block_tables.shape[1]
    max_context = max_blocks * block_size

    if scale is None:
        scale = 1.0 / (head_dim**0.5)

    # Gather latents from pool for each sequence
    # latent_pool: [num_blocks, block_size, kv_lora_rank]
    flat_indices = block_tables.reshape(-1)  # [num_seqs * max_blocks]
    gathered = latent_pool[flat_indices]  # [num_seqs * max_blocks, block_size, kv_lora_rank]

    # Reshape to [num_seqs, max_context, kv_lora_rank]
    gathered = gathered.reshape(num_seqs, max_blocks * block_size, kv_lora_rank)

    # Decompress: [num_seqs, max_context, 2 * num_kv_heads * head_dim]
    kv_decompressed = gathered @ kv_b_proj.T

    # Split into K and V
    split_dim = num_kv_heads * head_dim
    k_flat = kv_decompressed[:, :, :split_dim]  # [num_seqs, max_context, num_kv_heads * head_dim]
    v_flat = kv_decompressed[:, :, split_dim:]

    # Reshape to [num_seqs, max_context, num_kv_heads, head_dim]
    keys = k_flat.reshape(num_seqs, max_context, num_kv_heads, head_dim)
    values = v_flat.reshape(num_seqs, max_context, num_kv_heads, head_dim)

    # Transpose to [num_seqs, num_kv_heads, max_context, head_dim]
    keys = np.transpose(keys, (0, 2, 1, 3))
    values = np.transpose(values, (0, 2, 1, 3))

    # GQA expansion: repeat KV heads to match query heads
    if num_kv_heads < num_heads:
        repeat_factor = num_heads // num_kv_heads
        keys = np.repeat(keys, repeat_factor, axis=1)
        values = np.repeat(values, repeat_factor, axis=1)

    # Compute attention scores: [num_seqs, num_heads, seq_len, max_context]
    attn_weights = (query @ np.transpose(keys, (0, 1, 3, 2))) * scale

    # Build validity mask from context_lens
    kv_positions = np.arange(max_context)[None, :]  # [1, max_context]
    context_lens_2d = context_lens[:, None]  # [num_seqs, 1]
    valid_mask = kv_positions < context_lens_2d

    # Expand for broadcasting: [num_seqs, 1, 1, max_context]
    valid_mask = valid_mask[:, None, None, :]
    attn_weights = np.where(valid_mask, attn_weights, float("-inf"))

    # Causal mask for prefill
    if seq_len > 1:
        q_positions = np.arange(seq_len)[None, None, :, None]
        kv_pos_expanded = kv_positions[None, None, None, :]
        offsets = context_lens[:, None, None, None] - seq_len + q_positions
        causal_mask = kv_pos_expanded <= offsets
        attn_weights = np.where(causal_mask, attn_weights, float("-inf"))

    attn_weights = softmax(attn_weights, axis=-1)

    # Compute output: [num_seqs, num_heads, seq_len, head_dim]
    output = attn_weights @ values

    return output


def compare_memory_usage(
    seq_len: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    kv_lora_rank: int,
    dtype_bytes: int = 2,
) -> dict[str, float]:
    """Compare memory usage between standard and MLA KV cache.

    Args:
        seq_len: Sequence length.
        num_layers: Number of transformer layers.
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads.
        head_dim: Dimension per head.
        kv_lora_rank: MLA latent dimension.
        dtype_bytes: Bytes per element (2 for fp16/bf16).

    Returns:
        Dictionary with memory comparison statistics.
    """
    # Standard KV cache: 2 (K and V) * seq_len * num_kv_heads * head_dim * dtype_bytes * num_layers
    standard_bytes_per_layer = 2 * seq_len * num_kv_heads * head_dim * dtype_bytes
    standard_total_bytes = standard_bytes_per_layer * num_layers

    # MLA cache: seq_len * kv_lora_rank * dtype_bytes * num_layers
    mla_bytes_per_layer = seq_len * kv_lora_rank * dtype_bytes
    mla_total_bytes = mla_bytes_per_layer * num_layers

    return {
        "standard_per_layer_mb": standard_bytes_per_layer / 1024 / 1024,
        "standard_total_mb": standard_total_bytes / 1024 / 1024,
        "mla_per_layer_mb": mla_bytes_per_layer / 1024 / 1024,
        "mla_total_mb": mla_total_bytes / 1024 / 1024,
        "savings_ratio": standard_total_bytes / mla_total_bytes
        if mla_total_bytes > 0
        else float("inf"),
        "bytes_saved_total_mb": (standard_total_bytes - mla_total_bytes) / 1024 / 1024,
    }
