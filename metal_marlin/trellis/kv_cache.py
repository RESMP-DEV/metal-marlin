"""Trellis KV Cache with optional paged attention support.

This module provides a TrellisKVCache implementation that can use either:
1. Standard contiguous tensor storage (default)
2. PagedKVCache for memory-efficient block-based storage

The paged attention support enables:
- Memory-efficient KV cache management via block tables
- Copy-on-write semantics for prompt sharing
- Better memory utilization for long sequences
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    from metal_marlin.paged.mla_paged_adapter import (
        paged_attention_mla_int4,
        quantize_kv_int4,
    )
    HAS_PAGED_INT4 = True
except ImportError:
    HAS_PAGED_INT4 = False


class TrellisKVCache:
    """Trellis-oriented KV cache with optional paged attention support.

    This class wraps either standard contiguous tensor storage or a PagedKVCache
    for memory-efficient block-based storage. When use_paged=True, it creates
    and manages a PagedKVCache internally.

    Features:
    - Standard contiguous tensor storage (use_paged=False)
    - Paged block-based storage with block tables (use_paged=True)
    - Compatible with paged_attention_v1 for decode workloads
    - MLA-aware: stores compressed KV (kv_lora_rank + qk_rope_head_dim)

    Example:
        # Standard cache
        cache = TrellisKVCache(
            num_layers=32,
            max_seq_len=8192,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )

        # Paged cache
        cache = TrellisKVCache(
            num_layers=32,
            max_seq_len=8192,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            use_paged=True,
            block_size=16,
        )
    """

    def __init__(
        self,
        num_layers: int,
        max_seq_len: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "mps",
        use_paged: bool = False,
        block_size: int = 16,
    ):
        """Initialize TrellisKVCache.

        Args:
            num_layers: Number of transformer layers.
            max_seq_len: Maximum sequence length.
            kv_lora_rank: Compressed KV dimension (latent rank).
            qk_rope_head_dim: Dimension of rotary positional embeddings.
            dtype: Data type for cache tensors.
            device: Device to store cache on.
            use_paged: Whether to use paged (block-based) storage.
            block_size: Number of tokens per block (when use_paged=True).
        """
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.dtype = dtype
        self.device = device
        self.use_paged = use_paged
        self.block_size = block_size

        # Cache dimension = latent + RoPE components
        self.cache_dim = kv_lora_rank + qk_rope_head_dim

        # Internal cache storage
        self._kv_cache: torch.Tensor | None = None
        self._paged_cache: Any = None
        self._seq_lens: torch.Tensor | None = None
        self._compressed: bool = False

        if use_paged:
            self._init_paged_cache()
        else:
            self._init_standard_cache()

    def _init_standard_cache(self) -> None:
        """Initialize standard contiguous tensor cache."""
        self._kv_cache = torch.zeros(
            (self.num_layers, self.max_seq_len, self.cache_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self._seq_lens = torch.zeros(
            self.num_layers, dtype=torch.long, device=self.device
        )

    def _init_paged_cache(self) -> None:
        """Initialize paged block-based cache."""
        from ..paged.attention import paged_attention_v1, PagedKVCache
        from ..paged.kv_block import KVBlockConfig

        # Store the paged_attention_v1 function for later use
        self._paged_attention_fn = paged_attention_v1

        # Create KV block config for paged cache
        config = KVBlockConfig(
            block_size=self.block_size,
            num_heads=1,  # MLA stores compressed representation
            head_dim=self.cache_dim,
            dtype=self.dtype,
        )

        # Calculate number of blocks needed
        num_blocks = (self.max_seq_len + self.block_size - 1) // self.block_size
        num_blocks *= self.num_layers  # Blocks per layer

        # Create the paged cache
        self._paged_cache = PagedKVCache(
            config=config,
            num_blocks=num_blocks,
        )

        # Track sequence lengths per layer
        self._seq_lens = torch.zeros(
            self.num_layers, dtype=torch.long, device="cpu"
        )

    def update(
        self,
        layer_idx: int,
        compressed_kv: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Update cache with new compressed KV tokens.

        Args:
            layer_idx: Layer index.
            compressed_kv: New compressed KV tokens [batch, seq, cache_dim].

        Returns:
            Full cached sequence for the layer [batch, total_seq, cache_dim].
        """
        if compressed_kv is None:
            raise ValueError("compressed_kv is required")

        if self.use_paged:
            return self._update_paged(layer_idx, compressed_kv)
        else:
            return self._update_standard(layer_idx, compressed_kv)

    def _update_standard(
        self, layer_idx: int, compressed_kv: torch.Tensor
    ) -> torch.Tensor:
        """Update standard contiguous cache."""
        assert self._kv_cache is not None
        assert self._seq_lens is not None

        batch_size, seq_len, dim = compressed_kv.shape
        start_pos = int(self._seq_lens[layer_idx].item())
        end_pos = start_pos + seq_len

        if end_pos > self.max_seq_len:
            raise ValueError(
                f"Sequence length {end_pos} exceeds max_seq_len {self.max_seq_len}"
            )

        # Update cache
        self._kv_cache[layer_idx, start_pos:end_pos] = compressed_kv[0]
        self._seq_lens[layer_idx] = end_pos

        # Return full sequence
        return self._kv_cache[layer_idx, :end_pos].unsqueeze(0)

    def _update_paged(
        self, layer_idx: int, compressed_kv: torch.Tensor
    ) -> torch.Tensor:
        """Update paged block-based cache."""
        assert self._paged_cache is not None
        assert self._seq_lens is not None

        import numpy as np

        batch_size, seq_len, dim = compressed_kv.shape

        # Ensure sequence exists in paged cache
        seq_id = layer_idx  # Use layer_idx as sequence ID
        if not self._paged_cache.has_sequence(seq_id):
            self._paged_cache.add_sequence(seq_id)

        # Convert to numpy for paged cache
        compressed_np = compressed_kv[0].detach().cpu().numpy()

        # Append each token to paged cache
        for i in range(seq_len):
            # Create dummy key/value from compressed representation
            # In practice, these would be decompressed properly
            key = compressed_np[i : i + 1, np.newaxis, :]  # [1, 1, cache_dim]
            value = key.copy()

            success = self._paged_cache.append_kv(seq_id, key, value)
            if not success:
                raise RuntimeError(f"Failed to append KV to paged cache for layer {layer_idx}")

        self._seq_lens[layer_idx] += seq_len

        # Return the updated sequence
        # For paged cache, we gather from blocks
        return self.get(layer_idx) or compressed_kv

    def get(self, layer_idx: int) -> torch.Tensor | None:
        """Get full cached sequence for a layer.

        Args:
            layer_idx: Layer index.

        Returns:
            Cached sequence [batch, seq_len, cache_dim] or None if empty.
        """
        if self.use_paged:
            return self._get_paged(layer_idx)
        else:
            return self._get_standard(layer_idx)

    def _get_standard(self, layer_idx: int) -> torch.Tensor | None:
        """Get from standard cache."""
        assert self._kv_cache is not None
        assert self._seq_lens is not None

        seq_len = int(self._seq_lens[layer_idx].item())
        if seq_len == 0:
            return None

        return self._kv_cache[layer_idx, :seq_len].unsqueeze(0)

    def _get_paged(self, layer_idx: int) -> torch.Tensor | None:
        """Get from paged cache."""
        assert self._paged_cache is not None
        assert self._seq_lens is not None

        seq_len = int(self._seq_lens[layer_idx].item())
        if seq_len == 0:
            return None

        # Retrieve from paged cache
        seq_id = layer_idx
        try:
            keys, values = self._paged_cache.get_kv(seq_id)
            # Convert back to torch tensor
            import torch

            result = torch.from_numpy(keys[:, 0, :]).to(
                device=self.device, dtype=self.dtype
            )
            return result.unsqueeze(0)
        except ValueError:
            return None

    def get_seq_len(self) -> int:
        """Get current sequence length (last layer)."""
        assert self._seq_lens is not None
        return int(self._seq_lens[-1].item())

    @property
    def seq_len(self) -> int:
        """Current sequence length (last layer)."""
        return self.get_seq_len()

    def reset(self) -> None:
        """Reset cache state."""
        if self._seq_lens is not None:
            self._seq_lens.zero_()

        if self.use_paged and self._paged_cache is not None:
            # Reset paged cache by removing and re-adding sequences
            for seq_id in list(self._paged_cache.sequence_ids()):
                self._paged_cache.remove_sequence(seq_id)
        elif self._kv_cache is not None:
            self._kv_cache.zero_()

    def memory_usage_mb(self) -> float:
        """Calculate memory usage in MB."""
        if self.use_paged:
            return self._paged_memory_usage_mb()
        else:
            return self._standard_memory_usage_mb()

    def _standard_memory_usage_mb(self) -> float:
        """Calculate standard cache memory usage."""
        if self._kv_cache is None:
            return 0.0

        elements = self.num_layers * self.max_seq_len * self.cache_dim
        bytes_per_element = 2 if self.dtype == torch.float16 else 4
        return elements * bytes_per_element / (1024 * 1024)

    def _paged_memory_usage_mb(self) -> float:
        """Calculate paged cache memory usage."""
        if self._paged_cache is None:
            return 0.0

        stats = self._paged_cache.get_stats()
        return stats.memory_used_bytes / (1024 * 1024)

    def get_paged_attention_fn(self):
        """Get the paged attention function (when use_paged=True).

        Returns:
            paged_attention_v1 function if use_paged=True, else None.
        """
        if not self.use_paged:
            return None
        return getattr(self, "_paged_attention_fn", None)

    def get_paged_cache(self):
        """Get the underlying PagedKVCache (when use_paged=True).

        Returns:
            PagedKVCache instance if use_paged=True, else None.
        """
        return self._paged_cache

    def get_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get K and V tensors for a layer.

        Args:
            layer_idx: Layer index.

        Returns:
            Tuple of (k, v) tensors.
        """
        kv = self.get(layer_idx)
        if kv is None:
            raise ValueError(f"No KV cache for layer {layer_idx}")
        # Split the compressed KV into K and V components
        # For MLA: kv_lora_rank + qk_rope_head_dim = cache_dim
        k = kv[..., : self.kv_lora_rank]
        v = kv[..., self.kv_lora_rank :]
        return k, v

    def compress_to_int4(self, threshold_seq_len: int = 256) -> None:
        """Compress KV cache to INT4 when sequence exceeds threshold.

        MLA already compresses KV 8.9× via low-rank projection.
        INT4 reduces storage another 4×, giving 35.6× total compression.
        """
        # Import HAS_PAGED_INT4 and quantize_kv_int4
        try:
            from ..quant.int4 import HAS_PAGED_INT4, quantize_kv_int4
        except ImportError:
            return

        if not HAS_PAGED_INT4:
            return
        if self.seq_len < threshold_seq_len:
            return
        if self._compressed:
            return

        # Quantize existing KV tensors
        for layer_idx in range(self.num_layers):
            k, v = self.get_kv(layer_idx)
            self.k_cache[layer_idx] = quantize_kv_int4(k)
            self.v_cache[layer_idx] = quantize_kv_int4(v)
        self._compressed = True
