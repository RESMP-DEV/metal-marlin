"""
KV cache for Multi-head Latent Attention (MLA) that stores compressed representations.

MLA caches the compressed KV representation (after kv_a_proj) rather than the full KV,
reducing cache memory by ~8x for long sequences compared to standard KV caches.

The compressed KV has shape: [batch, seq_len, kv_lora_rank + num_kv_heads * head_dim_for_mqa]

This consists of:
- c_kv: Compressed latent representation [batch, seq_len, kv_lora_rank]
- k_pe: Positional embeddings for MQA [batch, seq_len, num_kv_heads * head_dim_for_mqa]

Usage:
    from metal_marlin.trellis.kv_cache import TrellisKVCache

    cache = TrellisKVCache(
        num_layers=32,
        batch_size=1,
        max_seq_len=4096,
        kv_lora_rank=512,
        num_kv_heads=4,
        head_dim=64,
    )

    # During forward pass with compressed KV from kv_a_proj
    # compressed_kv shape: [batch, seq, kv_lora_rank + num_kv_heads * head_dim]
    compressed_kv_full = cache.update(layer_idx=0, compressed_kv=compressed_kv_new)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._compat import require_torch, torch

if TYPE_CHECKING:
    import torch as torch_typing


class TrellisKVCache:
    """KV cache for Multi-head Latent Attention (MLA) that stores compressed representations.

    MLA (Multi-head Latent Attention) caches the compressed KV representation after
    kv_a_proj (the first projection in the two-step KV compression) rather than the
    full decompressed KV tensors. This reduces KV cache memory by approximately 8x
    for long sequences compared to standard KV caches.

    The cache stores:
    - c_kv: Compressed latent vectors of shape [num_layers, batch, max_seq, kv_lora_rank]
    - k_pe: Positional embeddings of shape [num_layers, batch, max_seq, num_kv_heads * head_dim]

    Memory Layout:
    - Stored as separate tensors for MPS optimization (contiguous, half precision)
    - c_kv is the main compressed representation (small, e.g., 512 dims)
    - k_pe is the positional embedding component for multi-query attention

    MLA KV Caching Flow:
    1. During prefill: hidden_states -> kv_a_proj -> compressed_kv -> [CACHE STORED]
    2. During generation: [CACHE RETRIEVED] -> compressed_kv -> kv_b_proj -> full_kv

    The cache is updated incrementally during autoregressive generation, with
    position_ids determining where new tokens are written.

    Attributes:
        num_layers: Number of transformer layers.
        batch_size: Batch size for generation.
        max_seq_len: Maximum sequence length to allocate.
        kv_lora_rank: Compressed KV dimension (rank of latent space, e.g., 512).
        num_kv_heads: Number of KV heads for multi-query attention.
        head_dim: Dimension per head.
        cache_dim: Total dimension stored (kv_lora_rank + num_kv_heads * head_dim).
        device: Device to store cache on.
        dtype: Data type for cache tensors (default: float16 for MPS optimization).
        c_kv: Compressed KV cache [num_layers, batch, max_seq, kv_lora_rank].
        k_pe: Positional embeddings [num_layers, batch, max_seq, num_kv_heads * head_dim].
        seq_lens: Current sequence lengths per batch item.
    """

    def __init__(  # noqa: PLR0913
        self,
        num_layers: int,
        batch_size: int,
        max_seq_len: int,
        kv_lora_rank: int,
        num_kv_heads: int = 4,
        head_dim: int = 64,
        device: str = "mps",
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize the Trellis KV cache for MLA.

        Args:
            num_layers: Number of transformer layers.
            batch_size: Batch size for generation.
            max_seq_len: Maximum sequence length to allocate.
            kv_lora_rank: Compressed KV dimension (rank of latent space).
            num_kv_heads: Number of KV heads for multi-query attention.
            head_dim: Dimension per head.
            device: Device to store cache on (default: 'mps' for Metal).
            dtype: Data type for cache tensors (default: float16).
        """
        require_torch()

        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.kv_lora_rank = kv_lora_rank
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.cache_dim = kv_lora_rank + num_kv_heads * head_dim
        self.device = device
        self.dtype = dtype

        # MPS-optimal: contiguous, half precision storage
        # c_kv: compressed latent representation [num_layers, batch, max_seq, kv_lora_rank]
        self.c_kv: torch_typing.Tensor = torch.zeros(
            num_layers,
            batch_size,
            max_seq_len,
            kv_lora_rank,
            dtype=dtype,
            device=device,
        )

        # k_pe: positional embeddings for MQA [num_layers, batch, max_seq, num_kv_heads * head_dim]
        self.k_pe: torch_typing.Tensor = torch.zeros(
            num_layers,
            batch_size,
            max_seq_len,
            num_kv_heads * head_dim,
            dtype=dtype,
            device=device,
        )

        # Track sequence lengths per batch item
        self.seq_lens: torch_typing.Tensor = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )

    def update(
        self,
        layer_idx: int,
        compressed_kv: torch_typing.Tensor,
    ) -> torch_typing.Tensor:
        """Update cache with new compressed KV and return full cached sequence.

        This is the primary interface for MLA KV caching. The compressed_kv tensor
        contains both the latent representation and positional embeddings concatenated
        along the last dimension.

        Args:
            layer_idx: Which transformer layer this is for.
            compressed_kv: New compressed KV tokens [batch, seq_len, cache_dim]
                where cache_dim = kv_lora_rank + num_kv_heads * head_dim.
                The first kv_lora_rank dimensions are c_kv (latent),
                the remaining are k_pe (positional embeddings).

        Returns:
            Full compressed KV for this layer [batch, total_seq, cache_dim]
            including all previously cached tokens plus new ones.
            Format is same as input: [c_kv | k_pe] concatenated.

        Raises:
            ValueError: If sequence length exceeds max_seq_len.
        """
        batch, seq_len, input_dim = compressed_kv.shape

        if input_dim != self.cache_dim:
            raise ValueError(
                f"Input dimension {input_dim} does not match expected cache_dim "
                f"{self.cache_dim} (kv_lora_rank={self.kv_lora_rank} + "
                f"num_kv_heads*head_dim={self.num_kv_heads * self.head_dim})"
            )

        # Get current position (assume same seq_len for batch items during generation)
        start = self.seq_lens[0].item()
        end = start + seq_len

        if end > self.max_seq_len:
            raise ValueError(
                f"Sequence length {end} exceeds max_seq_len {self.max_seq_len}"
            )

        # Split compressed_kv into c_kv and k_pe components
        # c_kv: [batch, seq_len, kv_lora_rank]
        # k_pe: [batch, seq_len, num_kv_heads * head_dim]
        c_kv_new = compressed_kv[..., : self.kv_lora_rank]
        k_pe_new = compressed_kv[..., self.kv_lora_rank :]

        # Ensure contiguous memory layout for MPS optimization
        if not c_kv_new.is_contiguous():
            c_kv_new = c_kv_new.contiguous()
        if not k_pe_new.is_contiguous():
            k_pe_new = k_pe_new.contiguous()

        # Store in cache - slice assignment maintains contiguous layout
        self.c_kv[layer_idx, :batch, start:end] = c_kv_new.to(self.dtype)
        self.k_pe[layer_idx, :batch, start:end] = k_pe_new.to(self.dtype)

        # Update sequence length on last layer (all layers processed for this token)
        if layer_idx == self.num_layers - 1:
            self.seq_lens[:batch] = end

        # Return all cached KV for this layer, concatenated format
        c_kv_full = self.c_kv[layer_idx, :batch, :end]
        k_pe_full = self.k_pe[layer_idx, :batch, :end]

        # Concatenate for output format [batch, total_seq, cache_dim]
        return torch.cat([c_kv_full, k_pe_full], dim=-1)

    def get(
        self,
        layer_idx: int,
    ) -> torch_typing.Tensor | None:
        """Get the full cached compressed KV for a layer.

        This retrieves the cached state in the format suitable for decompression
        via kv_b_proj. Returns None if no tokens are cached for this layer.

        Args:
            layer_idx: Which transformer layer to retrieve.

        Returns:
            Compressed KV tensor [batch, seq_len, cache_dim] where
            cache_dim = kv_lora_rank + num_kv_heads * head_dim, or
            None if cache is empty for this layer.
            The tensor is in MPS-optimal format (contiguous, half precision).
        """
        seq_len = self.seq_lens[0].item()
        if seq_len == 0:
            return None

        c_kv = self.c_kv[layer_idx, :, :seq_len]
        k_pe = self.k_pe[layer_idx, :, :seq_len]

        # Ensure contiguous output for MPS efficiency
        if not c_kv.is_contiguous():
            c_kv = c_kv.contiguous()
        if not k_pe.is_contiguous():
            k_pe = k_pe.contiguous()

        return torch.cat([c_kv, k_pe], dim=-1)

    def get_seq_len(self) -> int:
        """Current sequence length in cache.

        Returns:
            The current sequence length (assumes same length for all batch items).
        """
        return self.seq_lens[0].item()

    def advance(self, num_tokens: int = 1) -> None:
        """Advance sequence position (for compatibility with standard KV cache).

        In TrellisKVCache, sequence lengths are updated during update() calls.
        This method is provided for interface compatibility.

        Args:
            num_tokens: Number of tokens to advance (ignored, seq_len already updated).
        """
        # Seq lengths are already updated in update() when layer_idx == num_layers - 1
        pass

    def reset(self) -> None:
        """Clear cache for new generation.

        Resets sequence lengths to zero. The underlying cache tensors
        are not zeroed to avoid unnecessary memory operations.
        """
        self.seq_lens.zero_()

    def memory_usage_mb(self) -> float:
        """Calculate memory usage of the cache in MB.

        Returns:
            Memory usage in megabytes.
        """
        bytes_per_element = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4

        # c_kv: [num_layers, batch, max_seq, kv_lora_rank]
        c_kv_elements = (
            self.num_layers * self.batch_size * self.max_seq_len * self.kv_lora_rank
        )

        # k_pe: [num_layers, batch, max_seq, num_kv_heads * head_dim]
        k_pe_elements = (
            self.num_layers
            * self.batch_size
            * self.max_seq_len
            * self.num_kv_heads
            * self.head_dim
        )

        total_elements = c_kv_elements + k_pe_elements
        return total_elements * bytes_per_element / 1024 / 1024
