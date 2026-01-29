"""
KV cache for Multi-head Latent Attention (MLA) that stores compressed representations.

MLA caches the compressed KV representation (after kv_a_proj) rather than the full KV,
reducing cache memory by ~8x for long sequences compared to standard KV caches.

Usage:
    from metal_marlin.trellis_kv_cache import TrellisKVCache

    cache = TrellisKVCache(
        num_layers=32,
        batch_size=1,
        max_seq_len=4096,
        kv_lora_rank=512,
    )

    # During forward pass with compressed KV from kv_a_proj
    compressed_kv_full = cache.update(compressed_kv, layer_idx=0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._compat import require_torch, torch

if TYPE_CHECKING:
    import torch as torch_typing


class TrellisKVCache:
    """KV cache for MLA that stores compressed representations.

    MLA stores the compressed KV (after kv_a_proj) rather than full KV,
    reducing cache memory by ~8x for long sequences.

    The cache shape is [num_layers, batch, max_seq, kv_rank] where kv_rank
    is typically much smaller than num_heads * head_dim (e.g., 512 vs 4096).

    Attributes:
        num_layers: Number of transformer layers.
        batch_size: Batch size for generation.
        max_seq_len: Maximum sequence length to allocate.
        kv_lora_rank: Compressed KV dimension (rank of the latent space).
        cache: The underlying tensor [num_layers, batch, max_seq, kv_rank].
        seq_lens: Current sequence lengths per batch item.
    """

    def __init__(  # noqa: PLR0913
        self,
        num_layers: int,
        batch_size: int,
        max_seq_len: int,
        kv_lora_rank: int,
        device: str = "mps",
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize the Trellis KV cache.

        Args:
            num_layers: Number of transformer layers.
            batch_size: Batch size for generation.
            max_seq_len: Maximum sequence length to allocate.
            kv_lora_rank: Compressed KV dimension (rank of the latent space).
            device: Device to store cache on (default: 'mps' for Metal).
            dtype: Data type for cache tensors (default: float16).
        """
        require_torch()

        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.kv_lora_rank = kv_lora_rank
        self.device = device
        self.dtype = dtype

        # Store compressed KV: [num_layers, batch, max_seq, kv_rank]
        self.cache: torch_typing.Tensor = torch.zeros(
            num_layers,
            batch_size,
            max_seq_len,
            kv_lora_rank,
            dtype=dtype,
            device=device,
        )
        self.seq_lens: torch_typing.Tensor = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )

    def update(
        self,
        compressed_kv: torch_typing.Tensor,
        layer_idx: int,
    ) -> torch_typing.Tensor:
        """Update cache and return full compressed KV for this layer.

        Args:
            compressed_kv: New compressed KV tokens [batch, seq, kv_rank].
            layer_idx: Which transformer layer this is for.

        Returns:
            Full compressed KV for this layer [batch, total_seq, kv_rank]
            including all previously cached tokens plus new ones.

        Raises:
            ValueError: If sequence length exceeds max_seq_len.
        """
        batch, seq_len, _ = compressed_kv.shape

        # Get current position (assume same seq_len for batch items)
        start = self.seq_lens[0].item()
        end = start + seq_len

        if end > self.max_seq_len:
            raise ValueError(
                f"Sequence length {end} exceeds max_seq_len {self.max_seq_len}"
            )

        # Append new tokens
        self.cache[layer_idx, :batch, start:end] = compressed_kv

        # Update sequence length on last layer
        if layer_idx == self.num_layers - 1:
            self.seq_lens[:batch] = end

        # Return all cached KV for this layer
        return self.cache[layer_idx, :batch, :end]

    def get_seq_len(self) -> int:
        """Current sequence length in cache.

        Returns:
            The current sequence length (assumes same length for all batch items).
        """
        return self.seq_lens[0].item()

    def reset(self) -> None:
        """Clear cache for new generation.

        Resets sequence lengths to zero. The underlying cache tensor
        is not zeroed to avoid unnecessary memory operations.
        """
        self.seq_lens.zero_()

    def memory_usage_mb(self) -> float:
        """Calculate memory usage of the cache in MB.

        Returns:
            Memory usage in megabytes.
        """
        bytes_per_element = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        elements = (
            self.num_layers
            * self.batch_size
            * self.max_seq_len
            * self.kv_lora_rank
        )
        return elements * bytes_per_element / 1024 / 1024
