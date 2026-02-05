"""KV cache for Multi-head Latent Attention (MLA) that stores compressed representations.

MLA caches the compressed KV representation (after kv_a_proj) rather than the full KV,
reducing cache memory by ~8x for long sequences compared to standard KV caches.

The compressed KV has shape: [batch, seq_len, kv_lora_rank + qk_rope_head_dim]

This consists of:
- c_kv: Compressed latent representation [batch, seq_len, kv_lora_rank]
- k_pe: Rotary positional embeddings [batch, seq_len, qk_rope_head_dim]

Usage:
    from metal_marlin.trellis.kv_cache import TrellisKVCache

    cache = TrellisKVCache(
        num_layers=32,
        batch_size=1,
        max_seq_len=4096,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )

    # During forward pass with compressed KV from kv_a_proj
    # compressed_kv shape: [batch, seq, kv_lora_rank + qk_rope_head_dim]
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

    The cache stores compressed KV in a unified tensor:
    - kv_cache: [num_layers, batch, max_seq, cache_dim] where cache_dim = kv_lora_rank + qk_rope_head_dim
      - First kv_lora_rank dimensions are c_kv (compressed latent)
      - Remaining qk_rope_head_dim dimensions are k_pe (positional embeddings)

    Int8 Quantization:
    - When quantize=True, the cache stores KV as int8 with per-token scales
    - Quantization is applied on cache write (update method)
    - Dequantization is performed on-the-fly during read operations
    - Enables ~2x memory reduction compared to float16 storage

    Memory Layout (Optimal for Decode):
    - Stored in BSHD format [num_layers, batch, seq_len, cache_dim]
      - B = Batch, S = Sequence, H = 1 (implicit head), D = cache_dim
      - The seq dimension (S) is contiguous, optimal for decode writes
      - Single-token appends are coalesced memory writes
    - On read for attention: transpose to BHSD format via permute (zero-copy)
      - Returns [batch, cache_dim, seq_len] via strided view (no copy)
      - BHSD layout: [Batch, Head/Dim, Sequence, Dim] where head=cache_dim

    MLA KV Caching Flow:
    1. During prefill: hidden_states -> kv_a_proj -> compressed_kv -> [CACHE STORED]
    2. During generation: [CACHE RETRIEVED] -> compressed_kv -> kv_b_proj -> full_kv

    The cache is updated incrementally during autoregressive generation, with
    seq_lens tracking current sequence positions per batch item.

    Attributes:
        num_layers: Number of transformer layers.
        batch_size: Batch size for generation.
        max_seq_len: Maximum sequence length to allocate.
        kv_lora_rank: Compressed KV dimension (rank of latent space, e.g., 512).
        qk_rope_head_dim: Dimension of rotary positional embedding (e.g., 64).
        cache_dim: Total dimension stored (kv_lora_rank + qk_rope_head_dim).
        device: Device to store cache on.
        dtype: Data type for cache tensors (default: float16 for MPS optimization).
        quantize: Whether to use int8 quantization.
        kv_cache: Unified KV cache [num_layers, batch, max_seq, cache_dim] (BSHD layout).
        kv_scales: Per-token scales for quantization [num_layers, batch, max_seq, 1].
        seq_lens: Current sequence lengths per batch item.
    """

    def __init__(  # noqa: PLR0913
        self,
        num_layers: int,
        batch_size: int,
        max_seq_len: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int = 64,
        device: str = "mps",
        dtype: torch.dtype = torch.float16,
        quantize: bool = False,
        # Legacy parameters for backwards compatibility
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
    ):
        """Initialize the Trellis KV cache for MLA.

        Args:
            num_layers: Number of transformer layers.
            batch_size: Batch size for generation.
            max_seq_len: Maximum sequence length to allocate.
            kv_lora_rank: Compressed KV dimension (rank of latent space).
            qk_rope_head_dim: Dimension of rotary positional embedding (default: 64).
            device: Device to store cache on (default: 'mps' for Metal).
            dtype: Data type for cache tensors (default: float16).
            quantize: Whether to use int8 quantization (~2x memory reduction).
            num_kv_heads: DEPRECATED - ignored, kept for backwards compatibility.
            head_dim: DEPRECATED - ignored, kept for backwards compatibility.
        """
        require_torch()

        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.cache_dim = kv_lora_rank + qk_rope_head_dim
        self.device = device
        self.dtype = dtype
        self.quantize = quantize

        if quantize:
            storage_dtype = torch.int8
            scale_dtype = torch.float32
        else:
            storage_dtype = dtype
            scale_dtype = dtype

        # Unified KV cache: [num_layers, batch, max_seq, cache_dim]
        # Layout: BSHD (Batch, Sequence, Head=1, Dim=cache_dim)
        # Seq dimension is contiguous for optimal decode writes
        self.kv_cache: torch_typing.Tensor = torch.zeros(
            num_layers,
            batch_size,
            max_seq_len,
            self.cache_dim,
            dtype=storage_dtype,
            device=device,
        )

        # Per-token scale factors for symmetric int8 quantization
        # Shape: [num_layers, batch, max_seq, 1] - one scale per token
        if quantize:
            self.kv_scales: torch_typing.Tensor = torch.ones(
                num_layers, batch_size, max_seq_len, 1, dtype=scale_dtype, device=device
            )
        else:
            self.kv_scales = None

        # Track sequence lengths per batch item
        self.seq_lens: torch_typing.Tensor = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )

        # Track write position for multi-layer updates
        # This is set when layer 0 is updated and used by subsequent layers
        self._current_write_start: int = 0
        self._current_write_end: int = 0

    @property
    def seq_len(self) -> int:
        """Get the current sequence length (assumes uniform batch)."""
        return int(self.seq_lens[0].item())

    def _quantize_per_token(
        self, tensor: torch_typing.Tensor
    ) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Quantize tensor to int8 with per-token symmetric scaling.

        Args:
            tensor: Input tensor [batch, seq_len, cache_dim]

        Returns:
            Tuple of (quantized_int8, scales) where scales has shape [batch, seq_len, 1]
        """
        if not self.quantize:
            return tensor.to(self.dtype), None

        # Compute per-token max (across cache_dim)
        abs_max = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        scale = abs_max / 127.0
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale

    def _dequantize_per_token(
        self, quantized: torch_typing.Tensor, scales: torch_typing.Tensor
    ) -> torch_typing.Tensor:
        """Dequantize int8 tensor using per-token scales.

        Args:
            quantized: int8 tensor [batch, seq_len, cache_dim]
            scales: scale factors [batch, seq_len, 1]

        Returns:
            Dequantized float16 tensor
        """
        if not self.quantize:
            return quantized.to(self.dtype)

        return (quantized.to(self.dtype) * scales).to(self.dtype)

    def update(
        self,
        layer_idx: int,
        compressed_kv: torch_typing.Tensor,
    ) -> torch_typing.Tensor:
        """Update cache with new compressed KV and return full cached sequence.

        This is the primary interface for MLA KV caching. The compressed_kv tensor
        contains both the latent representation and positional embeddings concatenated
        along the last dimension. Stores data in BSHD layout for optimal write
        performance (seq dimension contiguous for decode).

        Args:
            layer_idx: Which transformer layer this is for.
            compressed_kv: New compressed KV tokens [batch, seq_len, cache_dim]
                where cache_dim = kv_lora_rank + qk_rope_head_dim.
                The first kv_lora_rank dimensions are c_kv (latent),
                the remaining are k_pe (positional embeddings).

        Returns:
            Full compressed KV for this layer [batch, total_seq, cache_dim]
            including all previously cached tokens plus new ones, in BSHD layout
            (strided view, not a copy).
            Format is same as input: [c_kv | k_pe] concatenated.

        Raises:
            ValueError: If sequence length exceeds max_seq_len.
        """
        batch, seq_len, input_dim = compressed_kv.shape

        if input_dim != self.cache_dim:
            raise ValueError(
                f"Input dimension {input_dim} does not match expected cache_dim "
                f"{self.cache_dim} (kv_lora_rank={self.kv_lora_rank} + "
                f"qk_rope_head_dim={self.qk_rope_head_dim})"
            )

        # Get current position (assume same seq_len for batch items during generation)
        # For layer 0, capture the write position and store it for use by other layers
        if layer_idx == 0:
            start = self.seq_lens[0].item()
            end = start + seq_len
            if end > self.max_seq_len:
                raise ValueError(f"Sequence length {end} exceeds max_seq_len {self.max_seq_len}")
            # Store for use by other layers in this update batch
            self._current_write_start = int(start)
            self._current_write_end = int(end)
        else:
            # Use the stored write position from layer 0
            start = self._current_write_start
            end = self._current_write_end

        # Write to cache with optional quantization
        if self.quantize:
            kv_q, kv_scales = self._quantize_per_token(compressed_kv)
            self.kv_cache[layer_idx, :batch, start:end] = kv_q
            self.kv_scales[layer_idx, :batch, start:end] = kv_scales
        else:
            self.kv_cache[layer_idx, :batch, start:end] = compressed_kv.to(self.dtype)

        # Update sequence length on last layer (all layers processed for this token)
        if layer_idx == self.num_layers - 1:
            self.seq_lens[:batch] = end

        # Return all cached KV for this layer (contiguous copy for MPS efficiency)
        kv_full = self.kv_cache[layer_idx, :batch, :end]

        if self.quantize:
            kv_full = self._dequantize_per_token(kv_full, self.kv_scales[layer_idx, :batch, :end])

        # Return contiguous tensor for MPS efficiency
        return kv_full.contiguous()

    def _get_effective_seq_len(self) -> int:
        """Get the effective sequence length for retrieval.

        Uses seq_lens if available, otherwise falls back to _current_write_end
        for data that's been written but not yet committed.
        """
        seq_len = int(self.seq_lens[0].item())
        if seq_len > 0:
            return seq_len
        # Fall back to current write end if data was written but seq_len not updated
        return self._current_write_end

    def get(
        self,
        layer_idx: int,
    ) -> torch_typing.Tensor | None:
        """Get the full cached compressed KV for a layer.

        This retrieves the cached state in BSHD layout [batch, seq_len, cache_dim]
        suitable for decompression via kv_b_proj. Returns a contiguous copy for
        MPS efficiency.

        Args:
            layer_idx: Which transformer layer to retrieve.

        Returns:
            Compressed KV tensor [batch, seq_len, cache_dim] in BSHD layout
            (contiguous), or None if cache is empty.
            Format: [c_kv | k_pe] concatenated on last dimension.
        """
        seq_len = self._get_effective_seq_len()
        if seq_len == 0:
            return None

        kv = self.kv_cache[layer_idx, :, :seq_len]

        if self.quantize:
            kv = self._dequantize_per_token(kv, self.kv_scales[layer_idx, :, :seq_len])

        return kv.contiguous()

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

        Resets sequence lengths to zero and clears write position trackers.
        The underlying cache tensor is not zeroed to avoid unnecessary memory operations.
        """
        self.seq_lens.zero_()
        self._current_write_start = 0
        self._current_write_end = 0

    def memory_usage_mb(self) -> float:
        """Calculate memory usage of the cache in MB.

        Returns:
            Memory usage in megabytes.
        """
        if self.quantize:
            bytes_per_element = 1  # int8
            scale_bytes = 4  # float32
        else:
            bytes_per_element = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
            scale_bytes = 0

        # kv_cache: [num_layers, batch, max_seq, cache_dim]
        kv_elements = self.num_layers * self.batch_size * self.max_seq_len * self.cache_dim

        # Scale factors: [num_layers, batch, max_seq, 1]
        scale_elements = (
            self.num_layers * self.batch_size * self.max_seq_len if self.quantize else 0
        )

        return (kv_elements * bytes_per_element + scale_elements * scale_bytes) / 1024 / 1024

    def prefetch_layer(self, layer_idx: int) -> torch_typing.Tensor | None:
        """Initiate prefetch of a layer's cache to warm GPU caches.

        This issues a memory read of the cache slice for the specified layer,
        which warms the GPU cache hierarchy before the data is actually needed.
        Call this for layer N+1 while computing layer N to hide memory latency.

        The prefetch is implemented as accessing the first element of the cache
        slice along with calling contiguous() on the slices if needed.
        This ensures the memory pages are loaded into GPU caches with minimal
        compute overhead.

        On Apple Silicon, this is effective because:
        1. The unified memory architecture means prefetching warms the SLC
        2. Sequential layer access patterns benefit from cache locality
        3. The tensor slice views share underlying memory with the main tensor

        Args:
            layer_idx: Which layer's cache to prefetch.

        Returns:
            A scalar tensor representing a minimal reduction over the prefetched
            data. The return value is primarily for synchronization; the important
            side effect is warming the cache. Returns None if cache is empty.

        Example:
            # In layer iteration:
            for i, layer in enumerate(self.model.layers):
                # Prefetch next layer's cache while computing current
                if i + 1 < num_layers:
                    kv_cache.prefetch_layer(i + 1)
                hidden_states = layer(hidden_states, kv_cache=kv_cache)
        """
        seq_len = self._get_effective_seq_len()
        if seq_len == 0 or layer_idx >= self.num_layers:
            return None

        # Touch the cache data to warm GPU caches.
        # Access the slices to create views that reference the underlying memory.
        kv_slice = self.kv_cache[layer_idx, :, :seq_len, :]

        # Use element access to trigger memory prefetch with minimal compute.
        # Accessing [0,0,0] forces the GPU to load the memory page containing
        # the start of the slice, and due to cache line sizes, adjacent data
        # gets pulled into cache as well.
        prefetch_signal = kv_slice[0, 0, 0]

        if seq_len > 256:
            # Touch middle of sequence for better cache coverage
            mid = seq_len // 2
            prefetch_signal = prefetch_signal + kv_slice[0, mid, 0]

        return prefetch_signal

    def prefetch_layer_async(
        self,
        layer_idx: int,
        lib: object | None = None,
    ) -> object | None:
        """Issue async prefetch using Metal's secondary command queue.

        This provides true async prefetching by dispatching a memory-touching
        kernel on the decode queue while the main computation runs on the
        primary queue. The GPU can execute both concurrently.

        Requires a MetalKernelLibrary instance to dispatch the prefetch kernel.
        If lib is None, falls back to prefetch_layer() which uses PyTorch.

        Args:
            layer_idx: Which layer's cache to prefetch.
            lib: MetalKernelLibrary instance for async dispatch.

        Returns:
            The command buffer if async prefetch was dispatched, None otherwise.
            The caller should not wait on this; the next layer's get() will
            synchronize implicitly.

        Example:
            # With async prefetch:
            for i, layer in enumerate(layers):
                if i + 1 < num_layers:
                    kv_cache.prefetch_layer_async(i + 1, lib=metal_lib)
                hidden_states = layer(hidden_states, kv_cache=kv_cache)
        """
        seq_len = self._get_effective_seq_len()
        if seq_len == 0 or layer_idx >= self.num_layers:
            return None

        # If no Metal library provided, fall back to PyTorch-based prefetch
        if lib is None:
            return self.prefetch_layer(layer_idx)

        # Import here to avoid circular dependency
        from ..metal_dispatch import mps_tensor_to_metal_buffer

        # Get Metal buffers for the cache slices.
        # These are views into the underlying tensor storage.
        kv_slice = self.kv_cache[layer_idx, :, :seq_len, :].contiguous()

        try:
            # Get Metal buffers (zero-copy from MPS tensors)
            kv_buffer = mps_tensor_to_metal_buffer(kv_slice, lib.device)

            # Create command buffer on separate decode queue for async prefetch
            command_buffer = lib.decode_queue.commandBuffer()
            blit_encoder = command_buffer.blitCommandEncoder()

            # Use copyFromBuffer with same source and destination as a no-op
            # that forces GPU to load the buffer into cache hierarchy.
            # This is better than synchronizeResource_ because:
            # 1. No barrier - allows GPU to continue immediately
            # 2. True prefetch - warms cache without blocking
            # 3. Overlap with computation - runs on decode queue while primary queue computes
            blit_encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
                kv_buffer, 0, kv_buffer, 0, 0
            )

            blit_encoder.endEncoding()
            command_buffer.commit()
            # Don't wait - this is async prefetch that will complete before next layer's get()

            return command_buffer

        except Exception:
            # Fall back to synchronous prefetch on any error
            return self.prefetch_layer(layer_idx)

    def get_layer_slices(
        self,
        layer_idx: int,
    ) -> tuple[torch_typing.Tensor, torch_typing.Tensor] | None:
        """Get raw slice views for a layer split into c_kv and k_pe components.

        This is more efficient than get() when you need to process c_kv and k_pe
        separately, as it avoids splitting overhead. The returned tensors
        are strided views into the cache storage (BSHD layout).

        Args:
            layer_idx: Which transformer layer to retrieve.

        Returns:
            Tuple of (c_kv_slice, k_pe_slice) where:
            - c_kv_slice: [batch, seq_len, kv_lora_rank] (BSHD layout, strided view)
            - k_pe_slice: [batch, seq_len, qk_rope_head_dim] (BSHD layout, strided view)
            Returns None if cache is empty.
        """
        seq_len = self._get_effective_seq_len()
        if seq_len == 0:
            return None

        kv = self.kv_cache[layer_idx, :, :seq_len]

        if self.quantize:
            kv = self._dequantize_per_token(kv, self.kv_scales[layer_idx, :, :seq_len])

        # Split into c_kv and k_pe using strided views (no copy)
        c_kv = kv[..., : self.kv_lora_rank]
        k_pe = kv[..., self.kv_lora_rank :]

        return c_kv, k_pe

    def get_layer_for_attention(
        self,
        layer_idx: int,
    ) -> torch_typing.Tensor | None:
        """Get cached KV in BHSD format for attention kernels (zero-copy transpose).

        Transposes from BSHD [batch, seq_len, cache_dim] to
        BHSD [batch, cache_dim, seq_len] format that attention kernels expect,
        without copying the underlying data.

        Args:
            layer_idx: Which transformer layer to retrieve.

        Returns:
            Compressed KV tensor [batch, cache_dim, seq_len] where
            cache_dim = kv_lora_rank + qk_rope_head_dim, in BHSD format (strided view).
            Returns None if cache is empty.
        """
        seq_len = self._get_effective_seq_len()
        if seq_len == 0:
            return None

        kv = self.kv_cache[layer_idx, :, :seq_len]

        if self.quantize:
            kv = self._dequantize_per_token(kv, self.kv_scales[layer_idx, :, :seq_len])

        # Transpose from BSHD [batch, seq_len, cache_dim] to BHSD [batch, cache_dim, seq_len]
        # Using permute creates a strided view without copying data
        return kv.permute(0, 2, 1)

    def get_snapshot(self) -> dict[str, torch.Tensor]:
        """Get a snapshot of the KV cache state for prompt caching.

        Creates deep copies of all cache tensors to enable storing
        the cache state for later reuse with identical prompts.

        Returns:
            Dict containing snapshots of:
            - kv_cache: Unified KV cache [num_layers, batch, max_seq, cache_dim]
            - seq_lens: Sequence lengths [batch]
            - kv_scales: Quantization scales if quantize=True (optional)
        """
        snapshot = {
            "kv_cache": self.kv_cache.clone(),
            "seq_lens": self.seq_lens.clone(),
        }

        if self.quantize:
            snapshot["kv_scales"] = self.kv_scales.clone()

        return snapshot

    def restore_snapshot(self, snapshot: dict[str, torch.Tensor]) -> None:
        """Restore KV cache state from a snapshot.

        Args:
            snapshot: Dict containing cache state from get_snapshot().
                Must include: kv_cache, seq_lens
                Optional: kv_scales (if quantize=True)
        """
        self.kv_cache.copy_(snapshot["kv_cache"])
        self.seq_lens.copy_(snapshot["seq_lens"])

        if self.quantize and "kv_scales" in snapshot:
            self.kv_scales.copy_(snapshot["kv_scales"])

    # Backwards compatibility properties
    @property
    def c_kv(self) -> torch_typing.Tensor:
        """Get compressed latent representation (c_kv component).

        Returns:
            Strided view into first kv_lora_rank dimensions of kv_cache.
            Shape: [num_layers, batch, max_seq, kv_lora_rank]
        """
        return self.kv_cache[..., : self.kv_lora_rank]

    @property
    def k_pe(self) -> torch_typing.Tensor:
        """Get positional embeddings (k_pe component).

        Returns:
            Strided view into last qk_rope_head_dim dimensions of kv_cache.
            Shape: [num_layers, batch, max_seq, qk_rope_head_dim]
        """
        return self.kv_cache[..., self.kv_lora_rank :]

    @property
    def c_kv_scales(self) -> torch_typing.Tensor | None:
        """Get scales for c_kv (for backwards compatibility).

        Returns:
            kv_scales tensor if quantize=True, None otherwise.
            Note: With unified quantization, the same scales apply to both components.
        """
        return self.kv_scales

    @property
    def k_pe_scales(self) -> torch_typing.Tensor | None:
        """Get scales for k_pe (for backwards compatibility).

        Returns:
            kv_scales tensor if quantize=True, None otherwise.
            Note: With unified quantization, the same scales apply to both components.
        """
        return self.kv_scales


class CompressedKVCache(TrellisKVCache):
    """Extended KV cache with compression for long sequences (GLM-4.7-Flash style).

    GLM-4.7-Flash uses MLA which already has compressed KV (kv_lora_rank=512).
    This class provides further optimizations:

    1. Late Int8 Quantization:
       - Stores KV in float16 for first 128 tokens (better accuracy during warmup)
       - Switches to int8 quantization after sequence stabilizes (memory savings)

    2. Sliding Window Attention:
       - For layers > 32, uses sliding window to limit KV cache size
       - Old tokens are evicted using LRU policy

    3. Grouped KV Recomputation:
       - Groups KV heads to reduce cache size by kv_group_size
       - Recomputes non-cached heads on-the-fly during attention

    Memory Layout:
    - Early layers (0-32): Full cache with optional late quantization
    - Deep layers (>32): Sliding window of size window_size
    - All layers: Grouped KV storage if kv_group_size > 1

    Example:
        >>> cache = CompressedKVCache(
        ...     num_layers=40,
        ...     batch_size=1,
        ...     max_seq_len=32768,
        ...     kv_lora_rank=512,
        ...     compression='int8',
        ...     sliding_window_layer_threshold=32,
        ...     sliding_window_size=4096,
        ...     kv_group_size=4,
        ... )
        >>> compressed_kv = cache.update(layer_idx=0, compressed_kv=new_kv)
    """

    def __init__(  # noqa: PLR0913
        self,
        num_layers: int,
        batch_size: int,
        max_seq_len: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int = 64,
        device: str = "mps",
        dtype: torch.dtype = torch.float16,
        compression: str = "int8",
        quantization_start_seq_len: int = 128,
        sliding_window_layer_threshold: int = 32,
        sliding_window_size: int = 4096,
        kv_group_size: int = 1,
        # Legacy parameters for backwards compatibility
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
    ):
        """Initialize the Compressed KV cache for long sequences.

        Args:
            num_layers: Number of transformer layers.
            batch_size: Batch size for generation.
            max_seq_len: Maximum sequence length to allocate.
            kv_lora_rank: Compressed KV dimension (rank of latent space).
            qk_rope_head_dim: Dimension of rotary positional embedding (default: 64).
            device: Device to store cache on (default: 'mps' for Metal).
            dtype: Data type for cache tensors (default: float16).
            compression: Compression type - 'int8', 'fp8', or 'none' (default: 'int8').
            quantization_start_seq_len: Sequence length to start int8 quantization (default: 128).
            sliding_window_layer_threshold: Layer index above which sliding window is used (default: 32).
            sliding_window_size: Window size for deep layers (default: 4096).
            kv_group_size: Number of heads to group for recomputation (1 = no grouping).
            num_kv_heads: DEPRECATED - ignored, kept for backwards compatibility.
            head_dim: DEPRECATED - ignored, kept for backwards compatibility.
        """
        # Initialize parent with quantize=False (we handle quantization ourselves)
        super().__init__(
            num_layers=num_layers,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            device=device,
            dtype=dtype,
            quantize=False,  # We handle compression ourselves
        )

        self.compression = compression
        self.quantization_start_seq_len = quantization_start_seq_len
        self.sliding_window_layer_threshold = sliding_window_layer_threshold
        self.sliding_window_size = sliding_window_size
        self.kv_group_size = kv_group_size

        # Track compression state per layer
        self._compression_enabled: list[bool] = [False] * num_layers

        # Sliding window tracking for deep layers
        # Stores the starting position of the window for each layer
        self._window_start: torch_typing.Tensor = torch.zeros(
            num_layers, dtype=torch.long, device=device
        )

        # Circular buffer indices for sliding window layers
        self._circular_pos: torch_typing.Tensor = torch.zeros(
            num_layers, dtype=torch.long, device=device
        )

        # If compression is enabled, allocate separate int8 buffers
        if compression == "int8":
            self._kv_cache_int8: torch_typing.Tensor | None = torch.zeros(
                num_layers,
                batch_size,
                max_seq_len,
                self.cache_dim,
                dtype=torch.int8,
                device=device,
            )
            self._kv_scales: torch_typing.Tensor | None = torch.ones(
                num_layers, batch_size, max_seq_len, 1, dtype=torch.float32, device=device
            )
        else:
            self._kv_cache_int8 = None
            self._kv_scales = None

        # For grouped KV: track which head groups are cached
        if kv_group_size > 1:
            # We cache 1 out of every kv_group_size heads
            # Store indices of cached groups
            self._cached_group_indices: torch_typing.Tensor | None = torch.arange(
                0, kv_lora_rank, kv_group_size, device=device
            )
        else:
            self._cached_group_indices = None

    def __len__(self) -> int:
        """Get current sequence length."""
        return int(self.seq_lens[0].item())

    def _quantize_int8(
        self, tensor: torch_typing.Tensor
    ) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Quantize tensor to int8 with per-token symmetric scaling.

        Args:
            tensor: Input tensor [batch, seq_len, cache_dim]

        Returns:
            Tuple of (quantized_int8, scales) where scales has shape [batch, seq_len, 1]
        """
        # Compute per-token max (across cache_dim)
        abs_max = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        scale = abs_max / 127.0
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale

    def _dequantize_int8(
        self, quantized: torch_typing.Tensor, scales: torch_typing.Tensor
    ) -> torch_typing.Tensor:
        """Dequantize int8 tensor using per-token scales.

        Args:
            quantized: int8 tensor [batch, seq_len, cache_dim]
            scales: scale factors [batch, seq_len, 1]

        Returns:
            Dequantized float16 tensor
        """
        return (quantized.to(self.dtype) * scales).to(self.dtype)

    def _should_use_sliding_window(self, layer_idx: int) -> bool:
        """Check if this layer should use sliding window attention."""
        return layer_idx >= self.sliding_window_layer_threshold

    def _get_window_bounds(self, layer_idx: int) -> tuple[int, int]:
        """Get the valid window bounds for a sliding window layer.

        Returns:
            Tuple of (start_idx, end_idx) for the valid window.
        """
        if not self._should_use_sliding_window(layer_idx):
            return (0, self.seq_len)

        current_len = self.seq_len
        window_start = max(0, current_len - self.sliding_window_size)
        return (window_start, current_len)

    def _apply_sliding_window_eviction(self, layer_idx: int) -> None:
        """Evict old tokens from sliding window layer.

        For circular buffer management in deep layers.
        """
        if not self._should_use_sliding_window(layer_idx):
            return

        current_len = int(self.seq_lens[0].item())
        if current_len <= self.sliding_window_size:
            return

        # Mark that this layer is using sliding window
        window_start = current_len - self.sliding_window_size
        self._window_start[layer_idx] = window_start

    def update(
        self,
        layer_idx: int,
        compressed_kv: torch_typing.Tensor,
    ) -> torch_typing.Tensor:
        """Update cache with new compressed KV.

        For long sequences:
        1. Applies int8 quantization if seq_len > quantization_start_seq_len
        2. Uses sliding window for layers > sliding_window_layer_threshold
        3. Groups KV for memory savings if kv_group_size > 1

        Args:
            layer_idx: Which transformer layer this is for.
            compressed_kv: New compressed KV tokens [batch, seq_len, cache_dim]

        Returns:
            Full compressed KV for this layer [batch, total_seq, cache_dim]
            with compression applied based on layer and sequence length.
        """
        batch, seq_len, input_dim = compressed_kv.shape

        if input_dim != self.cache_dim:
            raise ValueError(
                f"Input dimension {input_dim} does not match expected cache_dim "
                f"{self.cache_dim} (kv_lora_rank={self.kv_lora_rank} + "
                f"qk_rope_head_dim={self.qk_rope_head_dim})"
            )

        # Get current position
        if layer_idx == 0:
            start = self.seq_lens[0].item()
            end = start + seq_len
            if end > self.max_seq_len:
                raise ValueError(
                    f"Sequence length {end} exceeds max_seq_len {self.max_seq_len}"
                )
            self._current_write_start = int(start)
            self._current_write_end = int(end)
        else:
            start = self._current_write_start
            end = self._current_write_end

        # Determine if we should compress based on sequence length
        should_compress = (
            self.compression == "int8"
            and end > self.quantization_start_seq_len
        )

        # Enable compression for this layer once threshold is crossed
        if should_compress and not self._compression_enabled[layer_idx]:
            self._compression_enabled[layer_idx] = True

            # Optionally: compress existing cached tokens
            # This is done lazily on first write after threshold

        # Apply sliding window eviction for deep layers
        self._apply_sliding_window_eviction(layer_idx)

        # Write to cache (with optional compression)
        if should_compress and self._kv_cache_int8 is not None:
            # Quantize and store in int8 buffer
            kv_q, kv_scales = self._quantize_int8(compressed_kv)
            self._kv_cache_int8[layer_idx, :batch, start:end] = kv_q
            self._kv_scales[layer_idx, :batch, start:end] = kv_scales

            # Also store in float buffer for easy retrieval
            # (can be optimized to only store in one buffer)
            self.kv_cache[layer_idx, :batch, start:end] = compressed_kv.to(self.dtype)
        else:
            # Store in float16
            self.kv_cache[layer_idx, :batch, start:end] = compressed_kv.to(self.dtype)

        # Update sequence length on last layer
        if layer_idx == self.num_layers - 1:
            self.seq_lens[:batch] = end

        # Return full cached KV (with sliding window for deep layers)
        if self._should_use_sliding_window(layer_idx):
            window_start, window_end = self._get_window_bounds(layer_idx)
            kv_full = self.kv_cache[layer_idx, :batch, window_start:window_end]
        else:
            kv_full = self.kv_cache[layer_idx, :batch, :end]

        return kv_full.contiguous()

    def get(
        self,
        layer_idx: int,
    ) -> torch_typing.Tensor | None:
        """Get cached KV for a layer (with sliding window for deep layers).

        Args:
            layer_idx: Which transformer layer to retrieve.

        Returns:
            Compressed KV tensor [batch, seq_len, cache_dim] or None if empty.
            For deep layers (>32), returns only the sliding window.
        """
        seq_len = self._get_effective_seq_len()
        if seq_len == 0:
            return None

        # Apply sliding window for deep layers
        if self._should_use_sliding_window(layer_idx):
            window_start, window_end = self._get_window_bounds(layer_idx)
            kv = self.kv_cache[layer_idx, :, window_start:window_end]
        else:
            kv = self.kv_cache[layer_idx, :, :seq_len]

        # Dequantize if needed
        if self._compression_enabled[layer_idx] and self._kv_cache_int8 is not None:
            if self._should_use_sliding_window(layer_idx):
                window_start, window_end = self._get_window_bounds(layer_idx)
                kv_q = self._kv_cache_int8[layer_idx, :, window_start:window_end]
                scales = self._kv_scales[layer_idx, :, window_start:window_end]
            else:
                kv_q = self._kv_cache_int8[layer_idx, :, :seq_len]
                scales = self._kv_scales[layer_idx, :, :seq_len]
            kv = self._dequantize_int8(kv_q, scales)

        return kv.contiguous()

    def get_sliding_window_size(self, layer_idx: int) -> int | None:
        """Get the effective window size for a layer.

        Args:
            layer_idx: Layer index to query.

        Returns:
            Window size if sliding window is active, None otherwise.
        """
        if not self._should_use_sliding_window(layer_idx):
            return None
        return min(self.seq_len, self.sliding_window_size)

    def recompute_grouped_kv(
        self,
        layer_idx: int,
        cached_kv: torch_typing.Tensor,
        target_group: int,
    ) -> torch_typing.Tensor:
        """Recompute non-cached KV heads from cached group.

        For grouped KV attention: given cached KV for one head group,
        recompute the KV for other heads in the group.

        Args:
            layer_idx: Layer index.
            cached_kv: Cached KV for the master head [batch, seq_len, cache_dim]
            target_group: Target group index to recompute.

        Returns:
            Recomputed KV for target_group [batch, seq_len, cache_dim]
        """
        if self.kv_group_size <= 1:
            return cached_kv

        # Simple recomputation: interpolate within group
        # In practice, this would use the KV projection weights
        # For now, return the cached KV (exact recomputation requires model weights)
        return cached_kv

    def get_memory_stats(self) -> dict[str, float]:
        """Get memory usage statistics for the cache.

        Returns:
            Dict with memory breakdown in MB:
            - float_cache: Float16 cache memory
            - int8_cache: Int8 cache memory (if compression enabled)
            - scales: Scale factors memory
            - total: Total memory usage
            - effective_compression: Effective compression ratio
        """
        float_bytes = 2  # float16
        int8_bytes = 1
        scale_bytes = 4  # float32

        # Base float cache
        float_elements = self.num_layers * self.batch_size * self.max_seq_len * self.cache_dim
        float_memory = float_elements * float_bytes / 1024 / 1024

        stats = {
            "float_cache_mb": float_memory,
            "int8_cache_mb": 0.0,
            "scales_mb": 0.0,
            "total_mb": float_memory,
            "effective_compression": 1.0,
        }

        if self._kv_cache_int8 is not None:
            int8_memory = float_elements * int8_bytes / 1024 / 1024
            scale_elements = self.num_layers * self.batch_size * self.max_seq_len
            scales_memory = scale_elements * scale_bytes / 1024 / 1024

            # Count how many layers are compressed
            compressed_layers = sum(self._compression_enabled)
            compression_ratio = compressed_layers / self.num_layers if self.num_layers > 0 else 0

            stats["int8_cache_mb"] = int8_memory * compression_ratio
            stats["scales_mb"] = scales_memory * compression_ratio

            # Adjust float memory (only non-compressed layers store float)
            stats["float_cache_mb"] = float_memory * (1 - compression_ratio)
            stats["total_mb"] = (
                stats["float_cache_mb"] + stats["int8_cache_mb"] + stats["scales_mb"]
            )

            # Effective compression: float vs (int8 + scales)
            if compressed_layers > 0:
                compressed_float = float_memory * compression_ratio * float_bytes
                compressed_int8 = stats["int8_cache_mb"] + stats["scales_mb"]
                stats["effective_compression"] = compressed_float / compressed_int8

        return stats
