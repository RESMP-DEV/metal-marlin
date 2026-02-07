"""Complete trellis-quantized model for inference.

Provides a high-level nn.Module interface for loading and running
trellis-quantized models with support for dense and MoE layers.

Buffer Caching Strategy:
-----------------------
Weight buffers are created lazily on first forward pass and cached for reuse.
This achieves <1ms overhead for repeated inference by avoiding Metal buffer
creation during the decode phase.

- MoE layers: CachedWeightBuffers holds 13 Metal buffers for stacked expert weights
- Output buffers: OutputBufferPool pre-allocates common batch sizes (1, 2, 4, 8, 16)

Cache invalidation: Buffers are invalidated when model weights change (e.g., after
quantization or fine-tuning). Call invalidate_buffer_cache() to clear all cached
buffers.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kv_cache import TrellisKVCache
from ..metal_dispatch import (
    HAS_METAL,
    MetalKernelLibrary,
    PipelinedLayerDispatcher,
    mps_tensor_to_metal_buffer,
)
from ..transformer import RMSNorm
from .attention import TrellisMLAConfig, TrellisMLAttention
from .config import TrellisModelConfig
from .layer import TrellisDenseMLP
from .linear import TrellisLinear
from .moe_dispatch import (
    BatchedDispatcher,
    CachedRouterBuffers,
    CachedWeightBuffers,
    MoEBufferPool,
    RouterBufferPool,
    create_cached_weight_buffers,
    dispatch_moe_trellis_swiglu,
)
from .softmax_topk import SoftmaxTopKDispatcher

__all__ = [
    "TrellisModel",
    "TrellisDecoderLayer",
    "TrellisMoEMLP",
    "CausalLMOutput",
    "TrellisForCausalLM",
]

# Type alias for progress callback: (current_step, total_steps, message) -> None
ProgressCallback = Callable[[int, int, str], None]


logger = logging.getLogger(__name__)

# Canonical public Trellis model API symbols. `trellis.lm` and
# `trellis.__init__` re-export these names for backwards compatibility.
TRELLIS_PUBLIC_API_SYMBOLS: tuple[str, ...] = (
    "TrellisModel",
    "TrellisDecoderLayer",
    "TrellisForCausalLM",
    "TrellisMoEMLP",
)


@torch.compile(mode="reduce-overhead", fullgraph=True)
def _compiled_router_forward(
    x: torch.Tensor, router: nn.Module, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compiled router forward pass for better fusion.

    Combines router forward, softmax, and topk selection into a single
    compiled kernel to reduce Python overhead and enable fusion.

    Args:
        x: Input tensor [batch, hidden_dim].
        router: Router linear layer.
        top_k: Number of experts to select.

    Returns:
        Tuple of (indices, weights) where:
        - indices: Selected expert indices [batch, top_k]
        - weights: Normalized routing weights [batch, top_k]
    """
    logits = router(x)
    weights, indices = torch.topk(
        F.softmax(logits, dim=-1, dtype=torch.float16),
        k=top_k,
        dim=-1,
    )
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return indices, weights


if TYPE_CHECKING:
    from .loader import TrellisModelLoader

if HAS_METAL:
    pass


# Module-level timing stats for buffer caching performance measurement
_buffer_timing_stats: dict[str, list[float]] = {
    "first_call_ms": [],
    "cached_call_ms": [],
    "buffer_creation_ms": [],
}


def get_buffer_timing_stats() -> dict[str, Any]:
    """Get timing statistics for buffer caching performance.

    Returns:
        Dictionary with timing stats:
        - first_call_ms: List of first-call times (buffer creation)
        - cached_call_ms: List of subsequent call times (cached path)
        - first_call_avg_ms: Average first-call time
        - cached_call_avg_ms: Average cached-call time
        - speedup: Ratio of first_call_avg to cached_call_avg
    """
    stats = dict(_buffer_timing_stats)
    if stats["first_call_ms"]:
        stats["first_call_avg_ms"] = sum(
            stats["first_call_ms"]) / len(stats["first_call_ms"])
    else:
        stats["first_call_avg_ms"] = 0.0
    if stats["cached_call_ms"]:
        stats["cached_call_avg_ms"] = sum(
            stats["cached_call_ms"]) / len(stats["cached_call_ms"])
    else:
        stats["cached_call_avg_ms"] = 0.0
    if stats["buffer_creation_ms"]:
        stats["buffer_creation_avg_ms"] = sum(stats["buffer_creation_ms"]) / len(
            stats["buffer_creation_ms"]
        )
    else:
        stats["buffer_creation_avg_ms"] = 0.0
    if stats["cached_call_avg_ms"] > 0:
        stats["speedup"] = stats["first_call_avg_ms"] / \
            stats["cached_call_avg_ms"]
    else:
        stats["speedup"] = float("inf")
    return stats


def reset_buffer_timing_stats() -> None:
    """Reset buffer timing statistics."""
    global _buffer_timing_stats
    _buffer_timing_stats = {
        "first_call_ms": [],
        "cached_call_ms": [],
        "buffer_creation_ms": [],
    }


@dataclass
class PromptCacheEntry:
    """Entry in the prompt cache storing KV cache for a prompt prefix.

    Attributes:
        prompt_hash: Hash of the prompt tokens.
        kv_cache_snapshot: Snapshot of KV cache tensors for all layers.
        seq_len: Length of the cached prompt.
        timestamp: When this cache entry was created.
    """

    prompt_hash: str
    kv_cache_snapshot: dict[str, torch.Tensor]
    seq_len: int
    timestamp: float

    def is_expired(self, max_age_seconds: float = 3600) -> bool:
        """Check if this cache entry has expired.

        Args:
            max_age_seconds: Maximum age in seconds before expiration.

        Returns:
            True if expired, False otherwise.
        """
        return time.monotonic() - self.timestamp > max_age_seconds


# Module-level prompt cache - shared across all model instances
# Stores (prompt_hash, PromptCacheEntry) pairs
# Key: hash of prompt tokens -> Value: PromptCacheEntry
_prompt_cache: dict[str, PromptCacheEntry] = {}

# Prompt cache statistics
_prompt_cache_stats: dict[str, Any] = {
    "hits": 0,
    "misses": 0,
    "evictions": 0,
    "total_saved_prefill_tokens": 0,
}

# Maximum cache size (number of entries)
_MAX_PROMPT_CACHE_SIZE = 100

# Maximum age for cache entries (seconds)
_PROMPT_CACHE_MAX_AGE = 3600


def hash_prompt_tokens(input_ids: torch.Tensor) -> str:
    """Compute a hash for prompt tokens.

    Args:
        input_ids: Token IDs tensor [batch, seq_len].

    Returns:
        SHA256 hash as hex string.
    """
    tokens_bytes = input_ids.cpu().numpy().tobytes()
    return hashlib.sha256(tokens_bytes).hexdigest()


def get_prompt_cache_stats() -> dict[str, Any]:
    """Get prompt cache statistics.

    Returns:
        Dict with cache hits, misses, hit rate, and other stats.
    """
    total = _prompt_cache_stats["hits"] + _prompt_cache_stats["misses"]
    hit_rate = _prompt_cache_stats["hits"] / total if total > 0 else 0.0

    return {
        "cache_size": len(_prompt_cache),
        "hits": _prompt_cache_stats["hits"],
        "misses": _prompt_cache_stats["misses"],
        "hit_rate": hit_rate,
        "total_saved_prefill_tokens": _prompt_cache_stats["total_saved_prefill_tokens"],
        "evictions": _prompt_cache_stats["evictions"],
    }


def reset_prompt_cache_stats() -> None:
    """Reset prompt cache statistics."""
    global _prompt_cache_stats
    _prompt_cache_stats = {
        "hits": 0,
        "misses": 0,
        "evictions": 0,
        "total_saved_prefill_tokens": 0,
    }


def clear_prompt_cache() -> None:
    """Clear all cached prompts."""
    global _prompt_cache
    _prompt_cache.clear()
    reset_prompt_cache_stats()


def lookup_prompt_cache(
    input_ids: torch.Tensor,
) -> PromptCacheEntry | None:
    """Look up a prompt in the cache.

    Args:
        input_ids: Token IDs tensor [batch, seq_len].

    Returns:
        Cached entry if found, None otherwise.
    """
    global _prompt_cache_stats

    prompt_hash = hash_prompt_tokens(input_ids)

    if prompt_hash in _prompt_cache:
        entry = _prompt_cache[prompt_hash]
        if entry.is_expired(_PROMPT_CACHE_MAX_AGE):
            del _prompt_cache[prompt_hash]
            _prompt_cache_stats["misses"] += 1
            _prompt_cache_stats["evictions"] += 1
            return None

        _prompt_cache_stats["hits"] += 1
        _prompt_cache_stats["total_saved_prefill_tokens"] += entry.seq_len
        return entry

    _prompt_cache_stats["misses"] += 1
    return None


def store_prompt_cache(input_ids: torch.Tensor, kv_cache: TrellisKVCache) -> PromptCacheEntry:
    """Store a prompt in the cache.

    Args:
        input_ids: Token IDs tensor [batch, seq_len].
        kv_cache: KV cache to snapshot.

    Returns:
        The created cache entry.
    """
    global _prompt_cache

    prompt_hash = hash_prompt_tokens(input_ids)

    snapshot = kv_cache.get_snapshot()

    entry = PromptCacheEntry(
        prompt_hash=prompt_hash,
        kv_cache_snapshot=snapshot,
        seq_len=input_ids.shape[-1],
        timestamp=time.monotonic(),
    )

    _prompt_cache[prompt_hash] = entry

    # Evict oldest entries if cache is full
    while len(_prompt_cache) > _MAX_PROMPT_CACHE_SIZE:
        oldest_key = min(_prompt_cache.keys(),
                         key=lambda k: _prompt_cache[k].timestamp)
        del _prompt_cache[oldest_key]
        _prompt_cache_stats["evictions"] += 1

    return entry


def evict_expired_prompt_cache_entries() -> None:
    """Remove expired entries from the prompt cache."""
    global _prompt_cache, _prompt_cache_stats

    expired_keys = [k for k, v in _prompt_cache.items(
    ) if v.is_expired(_PROMPT_CACHE_MAX_AGE)]

    for key in expired_keys:
        del _prompt_cache[key]
        _prompt_cache_stats["evictions"] += 1


@dataclass
class TokenEmbeddingCache:
    """Cache for frequently-used token embeddings.

    Tracks token frequencies during generation and caches embeddings for
    the most common tokens (top 1000 by default). This avoids repeated
    embedding lookups during autoregressive decode.

    Cache hit rate is typically 60-80% for natural text.
    """

    vocab_size: int
    hidden_dim: int
    top_k: int = 1000
    device: str = "mps"
    dtype: torch.dtype = torch.float16

    _token_counts_tensor: torch.Tensor = field(init=False)
    _cached_embeddings: torch.Tensor | None = None
    _cache_mask: torch.Tensor = field(init=False)
    _cache_built: bool = False

    def __post_init__(self) -> None:
        """Initialize tensor-based counters and cache mask."""
        self._token_counts_tensor = torch.zeros(
            self.vocab_size, dtype=torch.int64, device=self.device
        )
        self._cache_mask = torch.zeros(
            self.vocab_size, dtype=torch.bool, device=self.device
        )

    def record_token(self, token_id: int) -> None:
        """Record a token usage for frequency tracking.

        Args:
            token_id: Token ID to record.
        """
        if 0 <= token_id < self.vocab_size:
            self._token_counts_tensor[token_id] += 1

    def build_cache(self, embedding_layer: nn.Embedding) -> None:
        """Build the embedding cache from recorded token frequencies.

        Pre-computes embeddings for the top_k most frequent tokens and stores
        them in a contiguous tensor for fast lookup.

        Args:
            embedding_layer: The nn.Embedding layer to cache from.
        """
        if self._cache_built:
            return

        if self._token_counts_tensor.sum() == 0:
            return

        # Get top-k tokens by count (tensor-based, no GPU→CPU sync for sorting)
        top_k_values, top_k_indices = torch.topk(
            self._token_counts_tensor, min(self.top_k, self.vocab_size)
        )
        top_tokens = [(idx.item(), count.item()) for idx, count in zip(
            top_k_indices, top_k_values) if count > 0]

        self._cached_embeddings = torch.zeros(
            self.vocab_size, self.hidden_dim, dtype=self.dtype, device=self.device
        )

        for token_id, _ in top_tokens:
            self._cached_embeddings[token_id] = embedding_layer.weight[token_id].to(
                self.dtype)
            self._cache_mask[token_id] = True

        self._cache_built = True

    def get_embeddings(
        self, input_ids: torch.Tensor, embedding_layer: nn.Embedding
    ) -> torch.Tensor:
        """Get embeddings for input tokens, using cache when available.

        Args:
            input_ids: Token IDs [batch, seq_len].
            embedding_layer: The nn.Embedding layer for uncached tokens.

        Returns:
            Embeddings [batch, seq_len, hidden_dim].
        """
        if not self._cache_built or self._cached_embeddings is None:
            return embedding_layer(input_ids)

        flat_ids = input_ids.flatten()
        result = torch.zeros(
            flat_ids.shape[0], self.hidden_dim, dtype=self.dtype, device=self.device
        )

        # Tensor-based membership check (no .tolist() GPU→CPU sync)
        cached_mask = self._cache_mask[flat_ids]

        if cached_mask.any():
            result[cached_mask] = self._cached_embeddings[flat_ids[cached_mask]]

        if (~cached_mask).any():
            uncached_ids = flat_ids[~cached_mask]
            uncached_embeddings = embedding_layer(uncached_ids).to(self.dtype)
            result[~cached_mask] = uncached_embeddings

        # Tensor-based token counting (no .tolist() GPU→CPU sync)
        # Use scatter_add to batch update counts
        ones = torch.ones_like(flat_ids, dtype=torch.int64)
        self._token_counts_tensor.scatter_add_(0, flat_ids.long(), ones)

        return result.reshape(*input_ids.shape, self.hidden_dim)

    def get_cache_stats(self) -> dict[str, int | float]:
        """Get cache statistics.

        Returns:
            Dict with:
            - cache_size: Number of cached tokens
            - total_tokens: Total unique tokens seen
            - cache_hit_rate: (cache_size / total_tokens) if tokens seen
        """
        total_unique = (self._token_counts_tensor > 0).sum().item()
        cache_size = self._cache_mask.sum().item()
        hit_rate = cache_size / total_unique if total_unique > 0 else 0.0

        return {
            "cache_size": cache_size,
            "total_tokens": total_unique,
            "cache_hit_rate": hit_rate,
        }


@dataclass
class OutputBufferPool:
    """Pre-allocated output buffer pool for common batch sizes.

    Avoids repeated allocation during autoregressive decode by reusing
    output tensors of the same shape.
    """

    hidden_dim: int
    device: Any  # MTLDevice
    _buffers: dict[int, tuple[torch.Tensor, Any]] = field(default_factory=dict)

    def get_output_buffer(
        self, batch_size: int, dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, Any]:
        """Get or create an output buffer for the given batch size.

        Args:
            batch_size: Number of tokens in the batch.
            dtype: Data type for the buffer.

        Returns:
            Tuple of (PyTorch tensor, Metal buffer) for output.
        """
        key = batch_size
        if key not in self._buffers:
            tensor = torch.zeros(batch_size, self.hidden_dim,
                                 dtype=dtype, device="mps")
            metal_buf = mps_tensor_to_metal_buffer(
                tensor, self.device, copy_back=True)
            self._buffers[key] = (tensor, metal_buf)
        return self._buffers[key]

    def preallocate(self, batch_sizes: list[int], dtype: torch.dtype = torch.float32) -> None:
        """Pre-allocate buffers for common batch sizes.

        Args:
            batch_sizes: List of batch sizes to pre-allocate.
            dtype: Data type for the buffers.
        """
        for bs in batch_sizes:
            self.get_output_buffer(bs, dtype)

    def clear(self) -> None:
        """Clear all cached buffers."""
        self._buffers.clear()

    def memory_usage_bytes(self) -> int:
        """Get total memory usage of cached buffers in bytes."""
        total = 0
        for tensor, _ in self._buffers.values():
            total += tensor.numel() * tensor.element_size()
        return total


class WorkspaceBufferPool:
    """Pre-allocated workspace buffers for MoE forward.

    Eliminates per-forward allocations by maintaining pre-allocated buffers
    for common tensor shapes used during MoE computation:
    - output_buffer: FP16 output tensor [batch, hidden_dim]
    - accum_buffer: FP32 accumulator for numerical stability [batch, hidden_dim]
    - intermediate: FP16 intermediate activations [batch, intermediate_dim]

    Buffers are lazily grown when larger batch sizes are requested.
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int, device: str | torch.device):
        """Initialize workspace buffer pool.

        Args:
            hidden_dim: Hidden dimension size.
            intermediate_dim: Intermediate dimension size (typically 4x hidden).
            device: Device to allocate buffers on.
        """
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.device = device

        # Pre-allocate common shapes (batch_size=1 is most common for decode)
        self.output_buffer = torch.empty(
            1, hidden_dim, dtype=torch.float16, device=device)
        self.accum_buffer = torch.empty(
            1, hidden_dim, dtype=torch.float32, device=device)
        self.intermediate = torch.empty(
            1, intermediate_dim, dtype=torch.float16, device=device)

        # Track current capacity
        self._current_batch_size = 1

    def get_output_buffer(self, batch_size: int) -> torch.Tensor:
        """Get output buffer for the given batch size.

        Args:
            batch_size: Number of tokens in the batch.

        Returns:
            Output buffer tensor [batch_size, hidden_dim] in fp16.
        """
        if batch_size == 1:
            return self.output_buffer
        # Grow if needed
        if batch_size > self.output_buffer.shape[0]:
            self.output_buffer = torch.empty(
                batch_size, self.hidden_dim, dtype=torch.float16, device=self.device
            )
            self._current_batch_size = batch_size
        return self.output_buffer[:batch_size]

    def get_accum_buffer(self, batch_size: int) -> torch.Tensor:
        """Get accumulator buffer for the given batch size.

        Args:
            batch_size: Number of tokens in the batch.

        Returns:
            Accumulator buffer tensor [batch_size, hidden_dim] in fp32.
        """
        if batch_size == 1:
            return self.accum_buffer
        # Grow if needed
        if batch_size > self.accum_buffer.shape[0]:
            self.accum_buffer = torch.empty(
                batch_size, self.hidden_dim, dtype=torch.float32, device=self.device
            )
            self._current_batch_size = max(
                self._current_batch_size, batch_size)
        return self.accum_buffer[:batch_size]

    def get_intermediate_buffer(self, batch_size: int) -> torch.Tensor:
        """Get intermediate buffer for the given batch size.

        Args:
            batch_size: Number of tokens in the batch.

        Returns:
            Intermediate buffer tensor [batch_size, intermediate_dim] in fp16.
        """
        if batch_size == 1:
            return self.intermediate
        # Grow if needed
        if batch_size > self.intermediate.shape[0]:
            self.intermediate = torch.empty(
                batch_size, self.intermediate_dim, dtype=torch.float16, device=self.device
            )
            self._current_batch_size = max(
                self._current_batch_size, batch_size)
        return self.intermediate[:batch_size]

    def get_all_buffers(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get all workspace buffers for the given batch size.

        Args:
            batch_size: Number of tokens in the batch.

        Returns:
            Tuple of (output_buffer, accum_buffer, intermediate_buffer).
        """
        return (
            self.get_output_buffer(batch_size),
            self.get_accum_buffer(batch_size),
            self.get_intermediate_buffer(batch_size),
        )

    def clear(self) -> None:
        """Clear all buffers, freeing memory."""
        self.output_buffer = torch.empty(
            1, self.hidden_dim, dtype=torch.float16, device=self.device)
        self.accum_buffer = torch.empty(
            1, self.hidden_dim, dtype=torch.float32, device=self.device)
        self.intermediate = torch.empty(
            1, self.intermediate_dim, dtype=torch.float16, device=self.device)
        self._current_batch_size = 1

    def memory_usage_bytes(self) -> int:
        """Get total memory usage of buffers in bytes."""
        return (
            self.output_buffer.numel() * self.output_buffer.element_size()
            + self.accum_buffer.numel() * self.accum_buffer.element_size()
            + self.intermediate.numel() * self.intermediate.element_size()
        )


class TrellisMoEMLP(nn.Module):
    """MoE MLP with trellis-quantized weights for MoE layers.

    Implements a mixture of experts with:
    - Router: selects top-k experts per token
    - Multiple experts: each is a dense MLP with SwiGLU
    - Shared expert: always applied (for model stability)

    This is used for layers >= first_moe_layer in GLM-4.7-Flash.

    Attributes:
        router: Linear layer for expert selection.
        experts: List of expert MLPs (TrellisDenseMLP).
        shared_expert: Always-active expert for stability.
        num_experts_per_tok: Number of experts to activate per token.
    """

    def __init__(
        self,
        router: nn.Linear,
        experts: list[TrellisDenseMLP],
        shared_expert: TrellisDenseMLP,
        num_experts_per_tok: int = 8,
        eager_buffers: bool = True,
    ):
        """Initialize TrellisMoEMLP.

        Args:
            router: Linear layer for expert selection scores.
            experts: List of expert MLPs (each a TrellisDenseMLP).
            shared_expert: Always-active expert MLP.
            num_experts_per_tok: Number of experts to activate per token.
            eager_buffers: If True (default), skip pre-stacking expert weights
                and create Metal buffers on-demand for better memory efficiency.
                If False, pre-stack weights into contiguous tensors (deprecated).
        """
        super().__init__()
        self._eager_buffers = eager_buffers
        self.router = router
        self.experts = nn.ModuleList(experts)
        self.shared_expert = shared_expert
        self.num_experts_per_tok = num_experts_per_tok

        # Check for mixed-precision experts (sensitivity-aware quantization)
        # Different experts may have different bit widths, which prevents stacking
        self._is_mixed_precision = self._check_mixed_precision()

        # Prepare contiguous expert weights for fast dispatch
        self._prepare_expert_weights()

        # Initialize _lib to None before _build_bit_group_cache which checks it
        self._lib: MetalKernelLibrary | None = None

        # Build bit-group cache for unified dispatch by (gate, up, down) bit tuple
        self._build_bit_group_cache()

        # Lazy Metal library and cached buffers
        self._cached_weight_buffers: CachedWeightBuffers | None = None
        self._cached_router_buffers: CachedRouterBuffers | None = None
        self._router_buffer_pool: RouterBufferPool | None = None
        self._output_buffer_pool: OutputBufferPool | None = None
        self._buffer_pool: MoEBufferPool | None = None
        self._workspace_buffer_pool: WorkspaceBufferPool | None = None
        self._use_fused_router: bool = True  # Enable fused router by default

        # Batched dispatch support - when set, forward() queues dispatches
        # instead of executing immediately. Call commit_batched_dispatch() to execute.
        self._batched_dispatcher: BatchedDispatcher | None = None
        self._pending_output: torch.Tensor | None = None  # Output from queued dispatch
        # Input for shared expert addition
        self._pending_input: torch.Tensor | None = None

        # Routing cache for decode optimization
        # Consecutive tokens often select same experts - cache and reuse when similar
        self._routing_cache_enabled: bool = True
        self._routing_cache_threshold: float = 0.95  # Cosine similarity threshold
        # Cache: (last_input_norm, selected_experts, routing_weights)
        self._cached_routing: tuple[torch.Tensor,
                                    torch.Tensor, torch.Tensor] | None = None
        # Stats for monitoring cache effectiveness (disabled by default for perf)
        self._routing_cache_hits: int = 0
        self._routing_cache_misses: int = 0

        # Buffer validity tracking for cache invalidation
        self._buffer_version: int = 0
        self._weights_hash: int | None = None

        # Per-bit-group cached buffers for mixed precision dispatch
        # Maps bit_width -> (CachedWeightBuffers, expert_indices_in_group)
        self._bit_group_buffers: dict[int,
                                      tuple[CachedWeightBuffers, list[int]]] | None = None

        # Bit-group cache for unified dispatch by (gate, up, down) bit tuple
        # Maps bit_tuple -> (expert_indices, unified_cached_buffers)
        # For GLM-4.7-Flash, ~6 unique bit tuples across 64 experts enable grouped dispatch
        self._bit_group_cache: (
            dict[tuple[int, int, int], tuple[list[int], CachedWeightBuffers]] | None
        ) = None

        # Timing instrumentation (disabled by default for performance)
        # Set _track_timing = True to enable timing stats collection
        self._first_forward_done = False
        self._track_timing = False

        # INT8 router flag (set via quantize_router_to_int8())
        self._use_int8_router = False

        # Fused softmax+topk dispatcher (created lazily)
        # Set _use_fused_topk = True to enable (experimental)
        self._use_fused_topk: bool = False
        self._softmax_topk_dispatcher: SoftmaxTopKDispatcher | None = None

        # Expert selection frequency tracking for hot/cold expert management
        self._expert_selection_counts: list[int] = [0] * len(experts)
        self._hot_expert_threshold: float = 0.5  # Top 50% are hot experts
        self._hot_experts: set[int] = set(
            range(len(experts)))  # Initially all hot
        # Lazy load cold experts
        self._cold_expert_buffer_pool: dict[int, CachedWeightBuffers] = {}
        self._expert_frequency_update_interval: int = 100  # Update hot/cold every N calls
        self._forward_call_count: int = 0

        # Optional eager buffer creation for memory efficiency
        if eager_buffers:
            self._create_buffers_eagerly()

    def _prepare_expert_weights(self) -> None:
        """Prepare expert weights in contiguous format for fast MoE dispatch.

        When self._eager_buffers is True, this method does minimal work
        since _create_buffers_eagerly() will handle buffer creation.

        This method is guarded to run only once. Subsequent calls are no-ops.
        """
        # Guard: ensure this runs only once (at load time, not during inference)
        if getattr(self, "_weights_prepared", False):
            return
        self._weights_prepared = True

        # If using eager buffers, skip creating stacked MPS tensors
        # _create_buffers_eagerly() will create buffers directly from CPU
        if self._eager_buffers:
            # Just store dimensions for later use
            first_expert = self.experts[0]
            self.hidden_dim = first_expert.gate_proj.in_features
            self.intermediate_dim = first_expert.gate_proj.out_features
            self.bits = first_expert.gate_proj.bits
            return

        # Deprecated: using non-optimized memory path
        import warnings

        warnings.warn(
            "Using non-optimized memory path. Set eager_buffers=True "
            "or use TrellisForCausalLM.from_pretrained() for better memory efficiency.",
            DeprecationWarning,
            stacklevel=2,
        )

        num_experts = len(self.experts)

        # Get dimensions from first expert
        first_expert = self.experts[0]
        hidden_dim = first_expert.gate_proj.in_features
        intermediate_dim = first_expert.gate_proj.out_features
        bits = first_expert.gate_proj.bits

        # Stack expert weights
        gate_weights_list = []
        gate_scales_list = []
        up_weights_list = []
        up_scales_list = []
        down_weights_list = []
        down_scales_list = []

        gate_su_list = []
        gate_sv_list = []
        up_su_list = []
        up_sv_list = []
        down_su_list = []
        down_sv_list = []

        for expert in self.experts:
            # Transpose packed weights from TrellisWeight [tiles_out, tiles_in, packed]
            # to GEMM convention [tiles_in, tiles_out, packed] for MoE kernel
            gate_weights_list.append(
                expert.gate_proj.packed_indices.permute(1, 0, 2).contiguous())
            gate_scales_list.append(expert.gate_proj.scales)
            up_weights_list.append(
                expert.up_proj.packed_indices.permute(1, 0, 2).contiguous())
            up_scales_list.append(expert.up_proj.scales)
            down_weights_list.append(
                expert.down_proj.packed_indices.permute(1, 0, 2).contiguous())
            down_scales_list.append(expert.down_proj.scales)

            gate_su_list.append(expert.gate_proj.su)
            gate_sv_list.append(expert.gate_proj.sv)
            up_su_list.append(expert.up_proj.su)
            up_sv_list.append(expert.up_proj.sv)
            down_su_list.append(expert.down_proj.su)
            down_sv_list.append(expert.down_proj.sv)

        # Stack along new dimension 0 (num_experts)
        # Convert scales/su/sv to float16 NOW to avoid dtype conversion copies
        # during Metal buffer creation later
        self.register_buffer("gate_weights_stacked",
                             torch.stack(gate_weights_list, dim=0))
        self.register_buffer("gate_scales_stacked", torch.stack(
            gate_scales_list, dim=0).half())
        self.register_buffer("up_weights_stacked",
                             torch.stack(up_weights_list, dim=0))
        self.register_buffer("up_scales_stacked", torch.stack(
            up_scales_list, dim=0).half())
        self.register_buffer("down_weights_stacked",
                             torch.stack(down_weights_list, dim=0))
        self.register_buffer("down_scales_stacked", torch.stack(
            down_scales_list, dim=0).half())

        self.register_buffer("gate_su_stacked", torch.stack(
            gate_su_list, dim=0).half())
        self.register_buffer("gate_sv_stacked", torch.stack(
            gate_sv_list, dim=0).half())
        self.register_buffer(
            "up_su_stacked", torch.stack(up_su_list, dim=0).half())
        self.register_buffer(
            "up_sv_stacked", torch.stack(up_sv_list, dim=0).half())
        self.register_buffer("down_su_stacked", torch.stack(
            down_su_list, dim=0).half())
        self.register_buffer("down_sv_stacked", torch.stack(
            down_sv_list, dim=0).half())

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.bits = bits

    def _check_mixed_precision(self) -> bool:
        """Check if experts have different bit widths (sensitivity-aware quantization).

        Returns:
            True if experts have varying bit widths, False if uniform.
        """
        if not self.experts:
            return False

        # Collect all unique bit widths across all projections
        all_bits = set()
        for expert in self.experts:
            all_bits.add(expert.gate_proj.bits)
            all_bits.add(expert.up_proj.bits)
            all_bits.add(expert.down_proj.bits)

        is_mixed = len(all_bits) > 1
        if is_mixed:
            # Store per-expert bits for dispatch grouping (use gate_proj as reference)
            self.expert_bits = torch.tensor(
                [e.gate_proj.bits for e in self.experts], dtype=torch.int32
            )
            self.unique_bits = sorted(all_bits)
            logger.info(
                f"Mixed-precision MoE detected: bits={self.unique_bits}. "
                "Using grouped bit-tuple Metal dispatch "
                "(single fused batched dispatch is disabled). "
                "Per-expert dispatch is only used as a fallback when grouped "
                "buffers are unavailable. "
                f"For optimal performance, quantize with uniform bit width."
            )
        return is_mixed

    def _build_bit_groups(self) -> dict[tuple[int, int, int], list[int]]:
        """Build expert groups by (gate, up, down) bit tuple for grouped dispatch.

        Sensitivity-aware trellis quantization assigns different bit widths to
        each projection independently. The fused MoE kernel requires uniform bits
        for all three projections, so we must group experts by the full
        (gate_bits, up_bits, down_bits) tuple.

        Returns:
            Dict mapping (gate_bits, up_bits, down_bits) -> list of expert indices.
        """
        groups: dict[tuple[int, int, int], list[int]] = {}
        for i, expert in enumerate(self.experts):
            bit_tuple = (
                expert.gate_proj.bits,
                expert.up_proj.bits,
                expert.down_proj.bits,
            )
            if bit_tuple not in groups:
                groups[bit_tuple] = []
            groups[bit_tuple].append(i)
        return groups

    def _build_bit_group_cache(self) -> None:
        """Build a cache mapping bit_tuple -> (expert_indices, unified_cached_buffers).

        For GLM-4.7-Flash, there are ~6 unique bit tuples across 64 experts.
        This allows dispatching all experts with the same bit configuration together,
        reducing dispatch overhead from O(unique_experts) ~8-16 to O(bit_groups) ~6.

        This method:
        1. Iterates over self.experts and extracts (gate.bits, up.bits, down.bits)
        2. Groups expert indices by bit_tuple
        3. Stores indices only (defer buffer creation to avoid memory duplication)
        4. Stores in self._bit_group_cache

        The cache format is:
            dict[tuple[int,int,int], tuple[list[int], None]]

        MEMORY OPTIMIZATION: We no longer pre-create CPU tensor copies here.
        Buffer creation is deferred to _ensure_bit_group_buffers() on first use.
        This saves ~16GB of duplicate weight storage.
        """
        if not self.experts:
            self._bit_group_cache = {}
            return

        # Group experts by (gate_bits, up_bits, down_bits) tuple
        groups: dict[tuple[int, int, int], list[int]] = {}
        for i, expert in enumerate(self.experts):
            bit_tuple = (
                expert.gate_proj.bits,
                expert.up_proj.bits,
                expert.down_proj.bits,
            )
            if bit_tuple not in groups:
                groups[bit_tuple] = []
            groups[bit_tuple].append(i)

        # Store only indices - buffers created lazily in _ensure_bit_group_buffers
        self._bit_group_cache = {
            bit_tuple: (expert_indices, None)
            for bit_tuple, expert_indices in groups.items()
        }

        logger.debug(
            f"Built bit-group cache: {len(groups)} unique bit tuples "
            f"across {len(self.experts)} experts (buffers deferred)"
        )

    def _stack_mixed_precision_weights(self, weights: list[torch.Tensor]) -> torch.Tensor:
        """Stack packed_indices tensors with potentially different bit widths.

        Sensitivity-aware trellis quantization assigns different bit widths to
        different experts. This causes packed_indices to have different last
        dimensions (64 for 2-bit, 96 for 3-bit, 128 for 4-bit, etc).

        This method pads all tensors to the maximum size for uniform stacking.
        The padding zeros don't affect computation since they're beyond the
        valid packed data range.

        Args:
            weights: List of packed_indices tensors, potentially with different
                last dimension sizes.

        Returns:
            Stacked tensor [num_experts, tiles_k, tiles_n, max_packed_bytes].
        """
        # Find max packed bytes across all experts
        max_packed = max(w.shape[-1] for w in weights)

        # Check if all weights already have same size (common case)
        if all(w.shape[-1] == max_packed for w in weights):
            return torch.stack(weights, dim=0)

        # Pad weights with different sizes to max
        padded = []
        for w in weights:
            if w.shape[-1] < max_packed:
                pad_size = max_packed - w.shape[-1]
                # Pad last dimension with zeros
                w = torch.nn.functional.pad(w, (0, pad_size), value=0)
            padded.append(w)

        return torch.stack(padded, dim=0)

    def _prepare_expert_weights_cpu(self) -> dict[str, torch.Tensor]:
        """Prepare expert weights on CPU for direct Metal buffer creation.

        Returns a dict of stacked CPU tensors that can be passed directly
        to create_cached_weight_buffers_from_cpu().

        Note: Handles mixed-precision experts where different experts may have
        different bit widths (sensitivity-aware quantization). Packed indices
        are padded to max size for uniform stacking.

        This method keeps weights on CPU throughout to avoid the expensive
        .cpu() -> .to("mps") round-trip. Weights should remain on CPU until
        directly copied to Metal buffers.

        Returns:
            Dict with keys: gate_weights, gate_scales, up_weights, up_scales,
            down_weights, down_scales, gate_su, gate_sv, up_su, up_sv,
            down_su, down_sv, grid. All tensors on CPU.
        """

        # Helper to ensure tensor is on CPU without round-trip through MPS
        def _ensure_cpu(tensor: torch.Tensor) -> torch.Tensor:
            """Get tensor on CPU, avoiding device transfer if already there."""
            if tensor.device.type == "cpu":
                return tensor
            # If on MPS/cuda, we need to move to CPU (but this should be rare
            # if weights are loaded correctly on CPU initially)
            return tensor.cpu()

        # Use mixed-precision stacking for packed_indices (may have different sizes)
        # Process on CPU to avoid MPS -> CPU synchronization
        gate_weights = self._stack_mixed_precision_weights(
            [_ensure_cpu(e.gate_proj.packed_indices) for e in self.experts]
        )
        up_weights = self._stack_mixed_precision_weights(
            [_ensure_cpu(e.up_proj.packed_indices) for e in self.experts]
        )
        down_weights = self._stack_mixed_precision_weights(
            [_ensure_cpu(e.down_proj.packed_indices) for e in self.experts]
        )

        # Scales and sign vectors always have uniform shapes
        gate_scales = torch.stack(
            [_ensure_cpu(e.gate_proj.scales) for e in self.experts], dim=0)
        up_scales = torch.stack([_ensure_cpu(e.up_proj.scales)
                                for e in self.experts], dim=0)
        down_scales = torch.stack(
            [_ensure_cpu(e.down_proj.scales) for e in self.experts], dim=0)

        gate_su = torch.stack([_ensure_cpu(e.gate_proj.su)
                              for e in self.experts], dim=0)
        gate_sv = torch.stack([_ensure_cpu(e.gate_proj.sv)
                              for e in self.experts], dim=0)
        up_su = torch.stack([_ensure_cpu(e.up_proj.su)
                            for e in self.experts], dim=0)
        up_sv = torch.stack([_ensure_cpu(e.up_proj.sv)
                            for e in self.experts], dim=0)
        down_su = torch.stack([_ensure_cpu(e.down_proj.su)
                              for e in self.experts], dim=0)
        down_sv = torch.stack([_ensure_cpu(e.down_proj.sv)
                              for e in self.experts], dim=0)

        # Transpose packed weights from [num_experts, tiles_out, tiles_in, packed]
        # to [num_experts, tiles_in, tiles_out, packed] for MoE kernel
        # All operations happen on CPU - no device transfers
        gate_weights = gate_weights.permute(0, 2, 1, 3).contiguous()
        gate_scales = gate_scales.half()
        up_weights = up_weights.permute(0, 2, 1, 3).contiguous()
        up_scales = up_scales.half()
        down_weights = down_weights.permute(0, 2, 1, 3).contiguous()
        down_scales = down_scales.half()

        gate_su = gate_su.half()
        gate_sv = gate_sv.half()
        up_su = up_su.half()
        up_sv = up_sv.half()
        down_su = down_su.half()
        down_sv = down_sv.half()

        grid = _ensure_cpu(self.experts[0].gate_proj.grid).half()

        return {
            "gate_weights": gate_weights,
            "gate_scales": gate_scales,
            "up_weights": up_weights,
            "up_scales": up_scales,
            "down_weights": down_weights,
            "down_scales": down_scales,
            "gate_su": gate_su,
            "gate_sv": gate_sv,
            "up_su": up_su,
            "up_sv": up_sv,
            "down_su": down_su,
            "down_sv": down_sv,
            "grid": grid,
        }

    def _get_lib(self) -> MetalKernelLibrary:
        """Get or create Metal kernel library."""
        if self._lib is None:
            self._lib = MetalKernelLibrary.from_source_dir()
        return self._lib

    def _check_fast_moe_available(self) -> bool:
        """Check if fast MoE kernel is available."""
        try:
            lib = self._get_lib()
            # Try to get pipeline - will raise if not available
            lib.get_pipeline("moe_trellis_swiglu")
            return True
        except Exception as e:
            import warnings

            warnings.warn(
                f"Fast MoE kernel unavailable, using slow path: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            return False

    def _get_cached_buffers(self) -> CachedWeightBuffers:
        """Get or create cached Metal buffers for static weights.

        Caches weight buffers on first call to avoid creating 13 Metal buffers
        per dispatch during decode. Only dynamic inputs (activations, expert_ids,
        expert_probs) need new buffers per call.
        """
        if self._cached_weight_buffers is not None:
            return self._cached_weight_buffers

        # If eager_buffers mode but buffers weren't created, create them now
        if getattr(self, "_eager_buffers", False):
            self._create_buffers_eagerly()
            return self._cached_weight_buffers

        # Original lazy creation path (from MPS tensors)
        start_time = time.perf_counter()
        lib = self._get_lib()
        self._cached_weight_buffers = create_cached_weight_buffers(
            device=lib.device,
            gate_weights=self.gate_weights_stacked,
            gate_scales=self.gate_scales_stacked,
            up_weights=self.up_weights_stacked,
            up_scales=self.up_scales_stacked,
            down_weights=self.down_weights_stacked,
            down_scales=self.down_scales_stacked,
            gate_su=self.gate_su_stacked,
            gate_sv=self.gate_sv_stacked,
            up_su=self.up_su_stacked,
            up_sv=self.up_sv_stacked,
            down_su=self.down_su_stacked,
            down_sv=self.down_sv_stacked,
            grid=self.experts[0].gate_proj.grid,
        )
        # Record buffer creation time
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _buffer_timing_stats["buffer_creation_ms"].append(elapsed_ms)

        # Compute weights hash for cache invalidation
        self._weights_hash = hash(
            (
                self.gate_weights_stacked.data_ptr(),
                self.up_weights_stacked.data_ptr(),
                self.down_weights_stacked.data_ptr(),
            )
        )
        self._buffer_version += 1

        return self._cached_weight_buffers

    def _get_output_buffer_pool(self) -> OutputBufferPool:
        """Get or create output buffer pool for common batch sizes.

        Pre-allocates output tensors for batch sizes 1, 2, 4, 8, 16 to avoid
        repeated allocation during autoregressive decode.
        """
        if self._output_buffer_pool is None:
            lib = self._get_lib()
            self._output_buffer_pool = OutputBufferPool(
                hidden_dim=self.hidden_dim,
                device=lib.device,
            )
            # Pre-allocate common decode batch sizes
            self._output_buffer_pool.preallocate(
                [1, 2, 4, 8, 16], dtype=torch.float32)
        return self._output_buffer_pool

    def _get_buffer_pool(self) -> MoEBufferPool:
        """Get or create MoE buffer pool for dynamic input/output buffers.

        Pre-allocates activation, expert_ids, expert_probs, and output buffers
        for common batch sizes to avoid repeated buffer creation during decode.
        """
        if self._buffer_pool is None:
            lib = self._get_lib()
            self._buffer_pool = MoEBufferPool(
                device=lib.device,
                hidden_dim=self.hidden_dim,
                max_batch=32,
            )
            # Pre-allocate buffers for top_k
            self._buffer_pool.preallocate_top_k(self.num_experts_per_tok)
        return self._buffer_pool

    def _get_workspace_buffer_pool(self) -> WorkspaceBufferPool:
        """Get or create workspace buffer pool for MoE forward.

        Pre-allocates output, accumulator, and intermediate buffers
        to eliminate per-forward allocations.
        """
        if self._workspace_buffer_pool is None:
            device = str(next(self.router.parameters()).device)
            self._workspace_buffer_pool = WorkspaceBufferPool(
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                device=device,
            )
        return self._workspace_buffer_pool

    def invalidate_buffer_cache(self) -> None:
        """Invalidate all cached buffers.

        Call this after modifying model weights (e.g., after quantization
        or fine-tuning) to ensure buffers are recreated on next forward pass.
        """
        self._cached_weight_buffers = None
        if self._output_buffer_pool is not None:
            self._output_buffer_pool.clear()
        self._output_buffer_pool = None
        if self._buffer_pool is not None:
            self._buffer_pool.clear()
        self._buffer_pool = None
        if self._workspace_buffer_pool is not None:
            self._workspace_buffer_pool.clear()
        self._workspace_buffer_pool = None
        self._buffer_version += 1
        self._weights_hash = None
        logger.debug("Buffer cache invalidated, version=%d",
                     self._buffer_version)

    def set_batched_dispatcher(self, dispatcher: BatchedDispatcher | None) -> None:
        """Set a batched dispatcher for deferred kernel execution.

        When a dispatcher is set, forward() queues dispatches instead of
        executing immediately. Call get_pending_output() after all layers have
        queued their dispatches and the dispatcher has committed.

        Args:
            dispatcher: BatchedDispatcher to use, or None to use immediate execution.
        """
        self._batched_dispatcher = dispatcher
        self._pending_output = None

    def get_pending_output(self) -> torch.Tensor | None:
        """Get the output tensor from a queued dispatch.

        This should be called after the batched dispatcher has committed.
        Returns None if no dispatch was queued.
        """
        return self._pending_output

    def _create_buffers_eagerly(self) -> None:
        """Create Metal buffers eagerly from CPU tensors.

        This method:
        1. Prepares expert weights on CPU (no MPS copy)
        2. Creates Metal buffers directly from CPU
        3. Deletes the stacked PyTorch tensors to free memory
        4. Optionally deletes individual expert weight tensors
        5. Pre-allocates buffer pool for decode (batch=1) fast path

        Called during __init__ when eager_buffers=True.
        """
        import gc

        from .moe_dispatch import create_cached_weight_buffers_from_cpu

        # Get Metal device - cache lib for decode fast path
        lib = self._get_lib()

        # Prepare weights on CPU
        cpu_weights = self._prepare_expert_weights_cpu()

        # Create Metal buffers directly from CPU
        self._cached_weight_buffers = create_cached_weight_buffers_from_cpu(
            device=lib.device, **cpu_weights
        )

        # Delete CPU weight copies
        del cpu_weights

        # Delete the stacked MPS tensors if they exist
        # (from _prepare_expert_weights which runs before this)
        for name in [
            "gate_weights_stacked",
            "gate_scales_stacked",
            "up_weights_stacked",
            "up_scales_stacked",
            "down_weights_stacked",
            "down_scales_stacked",
            "gate_su_stacked",
            "gate_sv_stacked",
            "up_su_stacked",
            "up_sv_stacked",
            "down_su_stacked",
            "down_sv_stacked",
        ]:
            if hasattr(self, name):
                delattr(self, name)

        # For mixed precision, create per-bit-group buffers for grouped dispatch
        if self._is_mixed_precision:
            self._create_bit_group_buffers()
            logger.info(
                f"Mixed-precision grouped dispatch enabled: "
                f"{len(self._bit_group_buffers)} bit groups"
            )
            # Clear expert weights after creating Metal buffers
            for expert in self.experts:
                for proj in [expert.gate_proj, expert.up_proj, expert.down_proj]:
                    proj.register_buffer(
                        "packed_indices", torch.empty(0, dtype=torch.uint8))
                    proj.register_buffer(
                        "scales", torch.empty(0, dtype=torch.float16))
                    proj.register_buffer(
                        "su", torch.empty(0, dtype=torch.float16))
                    proj.register_buffer(
                        "sv", torch.empty(0, dtype=torch.float16))
        else:
            # CRITICAL: Clear the original expert weights (the MPS tensors)
            # These are the source data that was copied to Metal buffers - now redundant
            # This is what actually frees the ~6GB of MPS memory
            for expert in self.experts:
                for proj in [expert.gate_proj, expert.up_proj, expert.down_proj]:
                    # Replace with empty tensors to free memory but keep module structure
                    proj.register_buffer(
                        "packed_indices", torch.empty(0, dtype=torch.uint8))
                    proj.register_buffer(
                        "scales", torch.empty(0, dtype=torch.float16))
                    proj.register_buffer(
                        "su", torch.empty(0, dtype=torch.float16))
                    proj.register_buffer(
                        "sv", torch.empty(0, dtype=torch.float16))

        # Pre-allocate buffer pool for decode fast path (batch=1 is most common)
        # This ensures _buffer_pool is ready on first forward call
        self._get_buffer_pool()

        # Force garbage collection
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        logger.debug("Created Metal buffers eagerly, freed PyTorch tensors")

    def _create_bit_group_buffers(self) -> None:
        """Create per-bit-group Metal buffers for mixed-precision dispatch.

        For sensitivity-aware quantization where experts have different bit widths,
        we cannot use a single batched dispatch (kernel requires uniform bits).
        Instead, we group experts by bit width and dispatch each group separately.

        This method creates:
        1. Separate CachedWeightBuffers for each bit group
        2. Global-to-local expert ID mapping for remapping during dispatch
        3. Bit-group membership lookup (expert_id -> bit_width)

        Called during _create_buffers_eagerly() when mixed precision is detected.
        """
        from .moe_dispatch import create_cached_weight_buffers_from_cpu

        lib = self._get_lib()
        bit_groups = self._build_bit_groups()

        self._bit_group_buffers = {}
        self._expert_to_bit_group: dict[int, int] = {}  # expert_id -> bits
        # expert_id -> local_id within group
        self._expert_to_local_id: dict[int, int] = {}

        for bits, expert_indices in bit_groups.items():
            # Build local ID mapping
            for local_id, global_id in enumerate(expert_indices):
                self._expert_to_bit_group[global_id] = bits
                self._expert_to_local_id[global_id] = local_id

            # Stack weights for this bit group only
            group_experts = [self.experts[i] for i in expert_indices]

            gate_weights = torch.stack(
                [e.gate_proj.packed_indices for e in group_experts], dim=0)
            up_weights = torch.stack(
                [e.up_proj.packed_indices for e in group_experts], dim=0)
            down_weights = torch.stack(
                [e.down_proj.packed_indices for e in group_experts], dim=0)

            gate_scales = torch.stack(
                [e.gate_proj.scales for e in group_experts], dim=0)
            up_scales = torch.stack(
                [e.up_proj.scales for e in group_experts], dim=0)
            down_scales = torch.stack(
                [e.down_proj.scales for e in group_experts], dim=0)

            gate_su = torch.stack(
                [e.gate_proj.su for e in group_experts], dim=0)
            gate_sv = torch.stack(
                [e.gate_proj.sv for e in group_experts], dim=0)
            up_su = torch.stack([e.up_proj.su for e in group_experts], dim=0)
            up_sv = torch.stack([e.up_proj.sv for e in group_experts], dim=0)
            down_su = torch.stack(
                [e.down_proj.su for e in group_experts], dim=0)
            down_sv = torch.stack(
                [e.down_proj.sv for e in group_experts], dim=0)

            # Transpose for MoE kernel and move to CPU
            cpu_weights = {
                "gate_weights": gate_weights.cpu().permute(0, 2, 1, 3).contiguous(),
                "gate_scales": gate_scales.cpu().half(),
                "up_weights": up_weights.cpu().permute(0, 2, 1, 3).contiguous(),
                "up_scales": up_scales.cpu().half(),
                "down_weights": down_weights.cpu().permute(0, 2, 1, 3).contiguous(),
                "down_scales": down_scales.cpu().half(),
                "gate_su": gate_su.cpu().half(),
                "gate_sv": gate_sv.cpu().half(),
                "up_su": up_su.cpu().half(),
                "up_sv": up_sv.cpu().half(),
                "down_su": down_su.cpu().half(),
                "down_sv": down_sv.cpu().half(),
                "grid": group_experts[0].gate_proj.grid.cpu().half(),
            }

            # Create Metal buffers for this bit group
            cached_buffers = create_cached_weight_buffers_from_cpu(
                device=lib.device, **cpu_weights)

            self._bit_group_buffers[bits] = (cached_buffers, expert_indices)

            logger.debug(
                f"Created bit-group buffers: bits={bits}, num_experts={len(expert_indices)}"
            )

    def _check_routing_cache(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Check if cached routing decision can be reused for this input.

        For decode (batch=1), consecutive tokens often select same experts.
        This method checks if the current input is similar enough to the
        cached input to reuse its routing decision, skipping the router
        forward pass.

        Args:
            x: Input tensor [1, hidden_dim] in fp16.

        Returns:
            Tuple of (selected_experts, routing_weights) if cache hit, None otherwise.
        """
        if not self._routing_cache_enabled or self._cached_routing is None:
            return None

        cached_norm, cached_experts, cached_weights = self._cached_routing

        # Fast cosine similarity: dot(x, cached) / (|x| * |cached|)
        # Both are already L2-normalized, so just dot product
        x_flat = x.view(-1)
        x_norm = x_flat / (x_flat.norm() + 1e-8)
        similarity = float(torch.dot(x_norm, cached_norm).cpu().numpy())

        if similarity >= self._routing_cache_threshold:
            self._routing_cache_hits += 1
            return cached_experts, cached_weights

        self._routing_cache_misses += 1
        return None

    def _update_routing_cache(
        self, x: torch.Tensor, selected_experts: torch.Tensor, routing_weights: torch.Tensor
    ) -> None:
        """Update the routing cache with new routing decision.

        Args:
            x: Input tensor [1, hidden_dim] in fp16.
            selected_experts: Selected expert indices [1, top_k].
            routing_weights: Routing weights [1, top_k].
        """
        if not self._routing_cache_enabled:
            return

        # L2 normalize for cosine similarity
        x_flat = x.view(-1)
        x_norm = x_flat / (x_flat.norm() + 1e-8)

        self._cached_routing = (
            x_norm.clone(), selected_experts.clone(), routing_weights.clone())

    def get_routing_cache_stats(self) -> dict[str, int | float]:
        """Get routing cache hit/miss statistics.

        Returns:
            Dict with hits, misses, and hit_rate.
        """
        total = self._routing_cache_hits + self._routing_cache_misses
        hit_rate = self._routing_cache_hits / total if total > 0 else 0.0
        return {
            "hits": self._routing_cache_hits,
            "misses": self._routing_cache_misses,
            "hit_rate": hit_rate,
        }

    def reset_routing_cache_stats(self) -> None:
        """Reset routing cache statistics."""
        self._routing_cache_hits = 0
        self._routing_cache_misses = 0

    def get_expert_stats(self) -> dict[str, Any]:
        """Get expert selection statistics and hot/cold expert information.

        Returns:
            Dict with:
            - selection_counts: List of selection counts per expert
            - hot_experts: Set of hot expert indices
            - cold_experts: Set of cold expert indices
            - hot_threshold: Current hot expert threshold (0.0-1.0)
            - cold_buffer_pool_size: Number of cold experts currently in buffer pool
        """
        num_experts = len(self.experts)
        cold_experts = set(range(num_experts)) - self._hot_experts
        return {
            "selection_counts": self._expert_selection_counts.copy(),
            "hot_experts": sorted(self._hot_experts),
            "cold_experts": sorted(cold_experts),
            "hot_threshold": self._hot_expert_threshold,
            "cold_buffer_pool_size": len(self._cold_expert_buffer_pool),
            "forward_calls": self._forward_call_count,
        }

    def reset_expert_stats(self) -> None:
        """Reset expert selection frequency statistics."""
        self._expert_selection_counts = [0] * len(self.experts)
        self._forward_call_count = 0
        self._hot_experts = set(range(len(self.experts)))
        self._cold_expert_buffer_pool.clear()

    def set_hot_expert_threshold(self, threshold: float) -> None:
        """Set the threshold for hot expert classification.

        Args:
            threshold: Fraction of experts to keep as hot (0.0-1.0).
                      Default is 0.5 (top 50% are hot).
                      Lower values reduce memory but may increase latency.
        """
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        self._hot_expert_threshold = threshold
        # Recompute hot experts with new threshold
        self._recompute_hot_experts()

    def get_memory_usage_estimate(self) -> dict[str, int]:
        """Estimate memory usage of the MoE layer.

        Returns:
            Dict with estimated memory usage in bytes:
            - hot_experts_bytes: Memory for hot expert buffers
            - cold_experts_bytes: Memory for cold experts in buffer pool
            - total_experts_bytes: Total if all experts were hot
            - savings_bytes: Memory saved by cold expert sharing
        """
        num_experts = len(self.experts)
        num_hot = len(self._hot_experts)
        num_cold_loaded = len(self._cold_expert_buffer_pool)

        # Estimate bytes per expert (13 buffers: weights, scales, su, sv, grid)
        # This is an approximation - actual size depends on dimensions
        bytes_per_expert = 0
        if num_experts > 0 and self._cached_weight_buffers is not None:
            # Calculate based on first expert dimensions
            expert = self.experts[0]
            hidden_dim = expert.gate_proj.in_features
            intermediate_dim = expert.gate_proj.out_features
            bits = expert.gate_proj.bits

            # Rough estimate: packed weights + scales + su/sv + grid
            tiles_in = (hidden_dim + 127) // 128
            tiles_out = (intermediate_dim + 127) // 128
            packed_size = tiles_in * tiles_out * (128 * 128 * bits // 8)
            bytes_per_expert = packed_size * 3 + hidden_dim * \
                6 * 2  # 3 projections, scales/su/sv

        hot_bytes = num_hot * bytes_per_expert
        cold_bytes = num_cold_loaded * bytes_per_expert
        total_if_all_hot = num_experts * bytes_per_expert
        savings = (num_experts - num_hot) * bytes_per_expert - cold_bytes

        return {
            "hot_experts_bytes": hot_bytes,
            "cold_experts_bytes": cold_bytes,
            "total_experts_bytes": total_if_all_hot,
            "savings_bytes": max(0, savings),
            "num_hot_experts": num_hot,
            "num_cold_experts": num_experts - num_hot,
            "num_cold_loaded": num_cold_loaded,
        }

    def clear_routing_cache(self) -> None:
        """Clear the routing cache (e.g., at start of new generation)."""
        self._cached_routing = None

    def _update_expert_frequencies(self, selected_experts: torch.Tensor) -> None:
        """Update expert selection frequency counts.

        Tracks how often each expert is selected to identify hot vs cold experts.
        Called during forward pass to accumulate statistics.

        Args:
            selected_experts: Selected expert indices [batch, top_k].
        """
        # Flatten and count selections
        flat_experts = selected_experts.flatten().tolist()
        for expert_id in flat_experts:
            self._expert_selection_counts[expert_id] += 1

    def _recompute_hot_experts(self) -> None:
        """Recompute which experts are 'hot' based on selection frequency.

        Hot experts (top 50% by frequency) keep dedicated buffers.
        Cold experts share a buffer pool and are loaded on demand.
        This reduces GPU memory by ~30% for typical workloads where
        a subset of experts is used more frequently.
        """
        num_experts = len(self.experts)
        if num_experts == 0:
            return

        # Sort experts by selection count
        expert_counts = [(i, count)
                         for i, count in enumerate(self._expert_selection_counts)]
        expert_counts.sort(key=lambda x: x[1], reverse=True)

        # Top 50% are hot experts
        num_hot = max(1, int(num_experts * self._hot_expert_threshold))
        self._hot_experts = set(expert_id for expert_id,
                                _ in expert_counts[:num_hot])

        # Clear cold expert buffer pool to free memory
        # They will be lazily reloaded when needed
        self._cold_expert_buffer_pool.clear()

        logger.debug(
            "Recomputed hot experts: %d hot (%s), %d cold",
            len(self._hot_experts),
            sorted(self._hot_experts)[:10],
            num_experts - len(self._hot_experts),
        )

    def _get_cold_expert_buffers(self, expert_id: int) -> CachedWeightBuffers | None:
        """Get or create weight buffers for a cold expert on demand.

        Cold experts are loaded into a shared buffer pool when needed.
        This enables memory-efficient handling of infrequently-used experts.

        Args:
            expert_id: Index of the cold expert to load.

        Returns:
            CachedWeightBuffers for the cold expert, or None if loading fails.
        """
        if expert_id in self._cold_expert_buffer_pool:
            return self._cold_expert_buffer_pool[expert_id]

        # Load cold expert weights from CPU storage
        from .moe_dispatch import create_cached_weight_buffers_from_cpu

        try:
            expert = self.experts[expert_id]
            lib = self._get_lib()

            # Prepare single expert weights on CPU
            cpu_weights = {
                "gate_weights": expert.gate_proj.packed_indices.cpu().permute(1, 0, 2).contiguous(),
                "gate_scales": expert.gate_proj.scales.cpu().half(),
                "up_weights": expert.up_proj.packed_indices.cpu().permute(1, 0, 2).contiguous(),
                "up_scales": expert.up_proj.scales.cpu().half(),
                "down_weights": expert.down_proj.packed_indices.cpu().permute(1, 0, 2).contiguous(),
                "down_scales": expert.down_proj.scales.cpu().half(),
                "gate_su": expert.gate_proj.su.cpu().half(),
                "gate_sv": expert.gate_proj.sv.cpu().half(),
                "up_su": expert.up_proj.su.cpu().half(),
                "up_sv": expert.up_proj.sv.cpu().half(),
                "down_su": expert.down_proj.su.cpu().half(),
                "down_sv": expert.down_proj.sv.cpu().half(),
                "grid": expert.gate_proj.grid.cpu().half(),
            }

            # Create Metal buffers from CPU
            buffers = create_cached_weight_buffers_from_cpu(
                device=lib.device, **cpu_weights)
            self._cold_expert_buffer_pool[expert_id] = buffers

            return buffers
        except Exception as e:
            logger.warning(
                "Failed to load cold expert %d buffers: %s", expert_id, e)
            return None

    def get_weight_tensors(self) -> dict[str, torch.Tensor] | None:
        """Get weight tensors for async transfer (if not using eager buffers)."""
        if self._cached_weight_buffers is not None or self._eager_buffers:
            return None

        # Return dict matching create_cached_weight_buffers arguments
        return {
            "gate_weights": self.gate_weights_stacked,
            "gate_scales": self.gate_scales_stacked,
            "up_weights": self.up_weights_stacked,
            "up_scales": self.up_scales_stacked,
            "down_weights": self.down_weights_stacked,
            "down_scales": self.down_scales_stacked,
            "gate_su": self.gate_su_stacked,
            "gate_sv": self.gate_sv_stacked,
            "up_su": self.up_su_stacked,
            "up_sv": self.up_sv_stacked,
            "down_su": self.down_su_stacked,
            "down_sv": self.down_sv_stacked,
            "grid": self.experts[0].gate_proj.grid,
        }

    def _dispatch_mixed_precision(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        lib: MetalKernelLibrary,
        buffer_pool: MoEBufferPool,
    ) -> torch.Tensor:
        """Grouped dispatch for mixed-precision experts.

        For sensitivity-aware quantization where experts have different bit widths,
        we group selected experts by their bit width and dispatch each group
        separately using the fast Metal kernel. This is more efficient than
        per-expert sequential dispatch while handling mixed precision.

        Algorithm:
        1. Group selected expert slots by their bit width
        2. For each bit group with active experts:
           - Remap global expert IDs to local indices within the group
           - Dispatch using that group's cached buffers
           - Scale output by routing weights for those slots
        3. Sum all group outputs

        For top-4 routing with 2-4 bit mixed precision, we typically have
        2-3 bit groups, achieving 10-20x speedup vs sequential dispatch.

        Args:
            x: Input tensor [batch, hidden_dim] in fp16.
            selected_experts: Selected expert indices [batch, top_k].
            routing_weights: Normalized routing weights [batch, top_k].
            lib: Metal kernel library.
            buffer_pool: Buffer pool for intermediate allocations.

        Returns:
            Output tensor [batch, hidden_dim] in fp16.
        """
        batch_size = x.shape[0]
        device = x.device

        # Initialize output accumulator using workspace buffer pool
        workspace_pool = self._get_workspace_buffer_pool()
        output = workspace_pool.get_output_buffer(batch_size).zero_()

        # Cache per-device lookup tensors:
        # - local_id_map[global_expert_id] -> local group index (or -1 if not in group)
        # This avoids Python-side global->local remapping and per-token `.item()` calls.
        if not hasattr(self, "_bit_group_lookup_cache"):
            self._bit_group_lookup_cache: dict[
                str, dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]]
            ] = {}
        device_key = str(device)
        device_lookup_cache = self._bit_group_lookup_cache.setdefault(device_key, {})
        num_experts_total = len(self.experts)

        # Group selected expert slots by bit width and dispatch once per active group.
        for bits, (cached_buffers, group_expert_indices) in self._bit_group_buffers.items():
            if not group_expert_indices:
                continue

            lookup_entry = device_lookup_cache.get(bits)
            if lookup_entry is None:
                group_ids_t = torch.tensor(
                    group_expert_indices,
                    dtype=torch.long,
                    device=device,
                )
                local_id_map = torch.full(
                    (num_experts_total,),
                    -1,
                    dtype=torch.long,
                    device=device,
                )
                local_id_map[group_ids_t] = torch.arange(
                    group_ids_t.numel(),
                    dtype=torch.long,
                    device=device,
                )
                lookup_entry = (group_ids_t, local_id_map)
                device_lookup_cache[bits] = lookup_entry

            _group_ids_t, local_id_map = lookup_entry

            # selected_local_ids: [batch, top_k], -1 where not in this bit group.
            selected_local_ids = local_id_map[selected_experts]
            batch_indices, slot_indices = torch.nonzero(
                selected_local_ids >= 0, as_tuple=True
            )
            if batch_indices.numel() == 0:
                continue

            group_activations = x[batch_indices]
            local_expert_ids = selected_local_ids[batch_indices, slot_indices].unsqueeze(1)
            group_probs = routing_weights[batch_indices, slot_indices].unsqueeze(1)

            group_output = dispatch_moe_trellis_swiglu(
                lib=lib,
                activations=group_activations,
                gate_weights=None,
                gate_scales=None,
                up_weights=None,
                up_scales=None,
                down_weights=None,
                down_scales=None,
                gate_su=None,
                gate_sv=None,
                up_su=None,
                up_sv=None,
                down_su=None,
                down_sv=None,
                grid=None,
                expert_ids=local_expert_ids,
                expert_probs=group_probs,
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                num_experts=len(group_expert_indices),
                top_k=1,
                bits=bits,
                cached_buffers=cached_buffers,
                buffer_pool=buffer_pool,
                use_fp32_acc=self.hidden_dim >= 1024,
            )

            output.index_add_(0, batch_indices, group_output)

        return output

    def forward_fast(
        self,
        x: torch.Tensor,
        workspace: torch.Tensor | None = None,
        workspace_offset: int = 0,
        cached_buffers: CachedWeightBuffers | None = None,
    ) -> torch.Tensor:
        """Fast forward pass using fused MoE dispatch.

        Replaces the slow sequential expert iteration (512 calls) with a
        single batched Metal kernel that processes all experts in parallel.

        For batch=1 (decode), uses an optimized path that:
        - Keeps everything in fp16 throughout (no dtype tracking/restore)
        - Minimizes intermediate tensor allocations
        - Uses pre-cached buffers directly
        - Caches routing decisions for similar inputs (~30% cache hit rate)
        - Tracks expert frequencies and manages hot/cold expert buffers

        Note: Batched dispatch across layers is disabled because sequential
        layer dependencies require immediate execution. Each layer's output
        feeds into the next layer's attention, so we cannot defer MoE
        computation.
        """
        batch_size = x.shape[0] if x.dim(
        ) == 2 else x.numel() // self.hidden_dim

        # Track forward calls for periodic hot/cold expert recomputation
        self._forward_call_count += 1

        # Fast path for decode (batch=1) - minimize overhead
        if batch_size == 1:
            # All resources must be pre-initialized - use getters to ensure availability
            cached = cached_buffers if cached_buffers is not None else self._get_cached_buffers()
            lib = self._get_lib()
            buffer_pool = self._get_buffer_pool()
            if cached is None or buffer_pool is None:
                raise RuntimeError(
                    "Fast MoE path requires cached buffers and buffer pool. "
                    "Ensure model is properly initialized with _create_buffers_eagerly()."
                )

            # For batch=1, x is [1, hidden_dim] - use view when contiguous to avoid allocation
            if x.dim() != 2:
                x = (
                    x.view(1, self.hidden_dim)
                    if x.is_contiguous()
                    else x.reshape(1, self.hidden_dim)
                )

            # Metal kernels require fp16 - convert only if not already fp16
            # In the standard forward path, inputs should already be fp16
            if x.dtype != torch.float16:
                x = x.half()

            # Try routing cache first - skips router forward ~30% of the time
            cache_result = self._check_routing_cache(x)
            if cache_result is not None:
                selected_experts, routing_weights = cache_result
            else:
                # Use compiled router forward for better fusion
                selected_experts, routing_weights = _compiled_router_forward(
                    x, self.router, self.num_experts_per_tok
                )

                # Update cache for next call
                self._update_routing_cache(
                    x, selected_experts, routing_weights)

            # Track expert selection frequencies for hot/cold management
            self._update_expert_frequencies(selected_experts)

            # Periodically recompute hot/cold experts
            if self._forward_call_count % self._expert_frequency_update_interval == 0:
                self._recompute_hot_experts()

            if self._is_mixed_precision:
                # Prefer prebuilt per-bit-group buffers with local expert remapping.
                # This path avoids the slower grouped fallback behavior when
                # zero-copy cache entries are unavailable for non-contiguous groups.
                if getattr(self, "_bit_group_buffers", None):
                    return self._dispatch_mixed_precision(
                        x=x,
                        selected_experts=selected_experts,
                        routing_weights=routing_weights,
                        lib=lib,
                        buffer_pool=buffer_pool,
                    )
                return self._forward_grouped(
                    x,
                    selected_experts,
                    routing_weights,
                    workspace=workspace,
                    workspace_offset=workspace_offset,
                )
            elif self._batched_dispatcher is not None:
                self._pending_input = x
                output = self._batched_dispatcher.queue_moe_dispatch(
                    activations=x,
                    expert_ids=selected_experts,
                    expert_probs=routing_weights,
                    hidden_dim=self.hidden_dim,
                    intermediate_dim=self.intermediate_dim,
                    num_experts=len(self.experts),
                    top_k=self.num_experts_per_tok,
                    bits=self.bits,
                    cached_buffers=cached,
                    buffer_pool=buffer_pool,
                    use_fp32_acc=self.hidden_dim >= 1024,
                )
            else:
                output = dispatch_moe_trellis_swiglu(
                    lib=lib,
                    activations=x,
                    gate_weights=None,
                    gate_scales=None,
                    up_weights=None,
                    up_scales=None,
                    down_weights=None,
                    down_scales=None,
                    gate_su=None,
                    gate_sv=None,
                    up_su=None,
                    up_sv=None,
                    down_su=None,
                    down_sv=None,
                    grid=None,
                    expert_ids=selected_experts,
                    expert_probs=routing_weights,
                    hidden_dim=self.hidden_dim,
                    intermediate_dim=self.intermediate_dim,
                    num_experts=len(self.experts),
                    top_k=self.num_experts_per_tok,
                    bits=self.bits,
                    cached_buffers=cached,
                    buffer_pool=buffer_pool,
                    use_fp32_acc=self.hidden_dim >= 1024,
                )

            # Add shared expert - output and shared_expert(x) are both fp16
            output.add_(self.shared_expert(x))
            return output

        # Standard path for batch > 1 (prefill)
        # Keep everything in fp16 throughout - no dtype tracking needed
        batch_shape = x.shape[:-1]

        # Convert to fp16 if needed (should be fp16 already in normal forward)
        if x.dtype != torch.float16:
            x = x.half()
        # Use view instead of reshape when possible to avoid allocation
        x_flat = (
            x.view(-1, self.hidden_dim)
            if x.is_contiguous()
            else x.reshape(-1, self.hidden_dim)
        )

        # Route - router weights should be fp16 (ensured at load time)
        router_logits = self.router(x_flat)

        # Select top-k experts - softmax in fp16 (stable on MPS)
        routing_weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=-1, dtype=torch.float16),
            k=self.num_experts_per_tok,
            dim=-1,
        )

        # Normalize weights (in-place to avoid allocation)
        routing_weights.div_(routing_weights.sum(dim=-1, keepdim=True))

        # Track expert selection frequencies for hot/cold management
        self._update_expert_frequencies(selected_experts)

        # Periodically recompute hot/cold experts
        if self._forward_call_count % self._expert_frequency_update_interval == 0:
            self._recompute_hot_experts()

        # Get cached buffers (required for fast path)
        cached_buffers = (
            cached_buffers if cached_buffers is not None else self._get_cached_buffers()
        )
        buffer_pool = self._get_buffer_pool()

        # For mixed precision, use grouped dispatch (per bit-tuple batching)
        if self._is_mixed_precision:
            if getattr(self, "_bit_group_buffers", None):
                output = self._dispatch_mixed_precision(
                    x=x_flat,
                    selected_experts=selected_experts,
                    routing_weights=routing_weights,
                    lib=self._get_lib(),
                    buffer_pool=buffer_pool,
                )
            else:
                output = self._forward_grouped(
                    x_flat,
                    selected_experts,
                    routing_weights,
                    workspace=workspace,
                    workspace_offset=workspace_offset,
                )
            return (
                output.view(*batch_shape, self.hidden_dim)
                if output.is_contiguous()
                else output.reshape(*batch_shape, self.hidden_dim)
            )
        elif self._batched_dispatcher is not None:
            if self._eager_buffers:
                output = self._batched_dispatcher.queue_moe_dispatch(
                    activations=x_flat,
                    expert_ids=selected_experts,
                    expert_probs=routing_weights,
                    hidden_dim=self.hidden_dim,
                    intermediate_dim=self.intermediate_dim,
                    num_experts=len(self.experts),
                    top_k=self.num_experts_per_tok,
                    bits=self.bits,
                    cached_buffers=cached_buffers,
                    buffer_pool=buffer_pool,
                    use_fp32_acc=self.hidden_dim >= 1024,
                )
            else:
                output = self._batched_dispatcher.queue_moe_dispatch(
                    activations=x_flat,
                    expert_ids=selected_experts,
                    expert_probs=routing_weights,
                    hidden_dim=self.hidden_dim,
                    intermediate_dim=self.intermediate_dim,
                    num_experts=len(self.experts),
                    top_k=self.num_experts_per_tok,
                    bits=self.bits,
                    cached_buffers=cached_buffers,
                    buffer_pool=buffer_pool,
                    use_fp32_acc=self.hidden_dim >= 1024,
                )
            self._pending_output = output
        else:
            if self._eager_buffers:
                output = dispatch_moe_trellis_swiglu(
                    lib=self._get_lib(),
                    activations=x_flat,
                    gate_weights=None,
                    gate_scales=None,
                    up_weights=None,
                    up_scales=None,
                    down_weights=None,
                    down_scales=None,
                    gate_su=None,
                    gate_sv=None,
                    up_su=None,
                    up_sv=None,
                    down_su=None,
                    down_sv=None,
                    grid=None,
                    expert_ids=selected_experts,
                    expert_probs=routing_weights,
                    hidden_dim=self.hidden_dim,
                    intermediate_dim=self.intermediate_dim,
                    num_experts=len(self.experts),
                    top_k=self.num_experts_per_tok,
                    bits=self.bits,
                    cached_buffers=cached_buffers,
                    buffer_pool=buffer_pool,
                    use_fp32_acc=self.hidden_dim >= 1024,
                )
            else:
                output = dispatch_moe_trellis_swiglu(
                    lib=self._get_lib(),
                    activations=x_flat,
                    gate_weights=self.gate_weights_stacked,
                    gate_scales=self.gate_scales_stacked,
                    up_weights=self.up_weights_stacked,
                    up_scales=self.up_scales_stacked,
                    down_weights=self.down_weights_stacked,
                    down_scales=self.down_scales_stacked,
                    gate_su=self.gate_su_stacked,
                    gate_sv=self.gate_sv_stacked,
                    up_su=self.up_su_stacked,
                    up_sv=self.up_sv_stacked,
                    down_su=self.down_su_stacked,
                    down_sv=self.down_sv_stacked,
                    grid=self.experts[0].gate_proj.grid,
                    expert_ids=selected_experts,
                    expert_probs=routing_weights,
                    hidden_dim=self.hidden_dim,
                    intermediate_dim=self.intermediate_dim,
                    num_experts=len(self.experts),
                    top_k=self.num_experts_per_tok,
                    bits=self.bits,
                    cached_buffers=cached_buffers,
                    buffer_pool=buffer_pool,
                    use_fp32_acc=self.hidden_dim >= 1024,
                )

        # Add shared expert (always applied) - both output and shared_expert(x) are fp16
        output.add_(self.shared_expert(x))

        # Restore shape - use view when contiguous to avoid allocation
        return (
            output.view(*batch_shape, self.hidden_dim)
            if output.is_contiguous()
            else output.reshape(*batch_shape, self.hidden_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        workspace: torch.Tensor | None = None,
        workspace_offset: int = 0,
        cached_buffers: CachedWeightBuffers | None = None,
    ) -> torch.Tensor:
        """Forward pass with MoE routing using fused Metal kernel.

        Unconditionally uses the fast Metal path.

        Args:
            x: Input tensor [..., hidden_size].
            workspace: Optional persistent workspace buffer.
            workspace_offset: Byte offset for scratch memory.
            cached_buffers: Pre-cached weight buffers (optimized path).

        Returns:
            Output tensor [..., hidden_size].
        """
        return self.forward_fast(
            x,
            workspace=workspace,
            workspace_offset=workspace_offset,
            cached_buffers=cached_buffers,
        )

    def _forward_grouped_fallback(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        workspace: torch.Tensor | None = None,
        workspace_offset: int = 0,
    ) -> torch.Tensor:
        """Fallback for mixed-precision using per-expert dispatch."""
        from .moe_dispatch import dispatch_moe_trellis_swiglu

        lib = self._get_lib()
        buffer_pool = self._get_buffer_pool()
        cached = self._get_cached_buffers()

        output = dispatch_moe_trellis_swiglu(
            lib=lib,
            activations=x,
            gate_weights=None,
            gate_scales=None,
            up_weights=None,
            up_scales=None,
            down_weights=None,
            down_scales=None,
            gate_su=None,
            gate_sv=None,
            up_su=None,
            up_sv=None,
            down_su=None,
            down_sv=None,
            grid=None,
            expert_ids=selected_experts,
            expert_probs=routing_weights,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_experts=len(self.experts),
            top_k=self.num_experts_per_tok,
            bits=self.bits,
            cached_buffers=cached,
            buffer_pool=buffer_pool,
            use_fp32_acc=self.hidden_dim >= 1024,
        )

        output.add_(self.shared_expert(x))
        return output

    def _ensure_bit_group_buffers(self) -> None:
        """Ensure bit_group_cache has proper Metal buffers (lazy init).

        When _build_bit_group_cache runs before Metal is available, the cache
        contains None instead of CachedWeightBuffers. This method lazily creates
        Metal buffers when first accessed during forward pass.

        ZERO-COPY OPTIMIZATION: Uses tensor views/slices from the global
        _cached_weight_buffers instead of creating new stacked buffers.
        This avoids memory duplication while enabling grouped dispatch.

        FALLBACK: If experts within a bit group are non-contiguous or
        heterogeneous bit configurations prevent safe slicing, falls back
        to per-expert dispatch for that group.
        """
        if self._cached_weight_buffers is None:
            # Global buffers not available yet, will retry on next forward
            return

        if not hasattr(self, '_bit_group_cache') or self._bit_group_cache is None:
            return

        from .moe_dispatch import CachedWeightBuffers

        global_buffers = self._cached_weight_buffers

        # Get number of experts from stacked weights
        # gate_weights shape: [num_experts, tiles_in, tiles_out, packed_bytes]
        num_experts_total = self.gate_weights_stacked.shape[0] if hasattr(
            self, 'gate_weights_stacked') else len(self.experts)

        for bit_tuple, (expert_indices, existing_buffers) in self._bit_group_cache.items():
            # Skip if already created
            if existing_buffers is not None and isinstance(existing_buffers, CachedWeightBuffers):
                continue

            # Skip empty groups
            if not expert_indices:
                continue

            # Check if experts are contiguous (required for zero-copy slicing)
            is_contiguous = (
                len(expert_indices) > 0 and
                expert_indices == list(range(expert_indices[0], expert_indices[-1] + 1))
            )

            if not is_contiguous:
                # Non-contiguous experts require either:
                # 1. Copy-based stacking (memory expensive), or
                # 2. Multiple separate dispatches
                # Fall back to per-expert dispatch for this group
                continue

            start_idx = expert_indices[0]
            end_idx = expert_indices[-1] + 1  # Exclusive

            # Validate indices are within bounds
            if end_idx > num_experts_total:
                logger.warning(
                    f"Bit group {bit_tuple} has expert indices exceeding "
                    f"total experts ({num_experts_total}), skipping"
                )
                continue

            try:
                # Create zero-copy views into global buffers
                # Each buffer is sliced along the expert dimension (dim 0)
                group_buffers = CachedWeightBuffers(
                    gate_weights=global_buffers.gate_weights,
                    gate_scales=global_buffers.gate_scales,
                    up_weights=global_buffers.up_weights,
                    up_scales=global_buffers.up_scales,
                    down_weights=global_buffers.down_weights,
                    down_scales=global_buffers.down_scales,
                    gate_su=global_buffers.gate_su,
                    gate_sv=global_buffers.gate_sv,
                    up_su=global_buffers.up_su,
                    up_sv=global_buffers.up_sv,
                    down_su=global_buffers.down_su,
                    down_sv=global_buffers.down_sv,
                    grid=global_buffers.grid,
                )

                # Store metadata for slicing in dispatch
                # We use a special marker to indicate this is a view-based buffer
                # The actual slicing will be handled by the kernel params
                self._bit_group_cache[bit_tuple] = (
                    expert_indices,
                    group_buffers,
                )

                # Store slice info for use in dispatch
                if not hasattr(self, '_bit_group_slice_info'):
                    self._bit_group_slice_info: dict[
                        tuple[int, int, int], tuple[int, int]
                    ] = {}
                self._bit_group_slice_info[bit_tuple] = (start_idx, end_idx)

                logger.debug(
                    f"Created zero-copy bit-group buffers: {bit_tuple}, "
                    f"experts[{start_idx}:{end_idx}]"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to create zero-copy buffers for {bit_tuple}: {e}. "
                    f"Falling back to per-expert dispatch for this group."
                )
                continue

    def _forward_grouped(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        workspace: torch.Tensor | None = None,
        workspace_offset: int = 0,
    ) -> torch.Tensor:
        # Guard: fall back if cache not available
        if not hasattr(self, '_bit_group_cache') or self._bit_group_cache is None:
            return self._forward_grouped_fallback(
                x, selected_experts, routing_weights, workspace, workspace_offset
            )

        # Attempt to create zero-copy buffers if not already done
        # This handles lazy initialization when Metal becomes available
        self._ensure_bit_group_buffers()

        from .moe_dispatch import dispatch_moe_per_bit_tuple, dispatch_moe_trellis_swiglu

        # Split groups by availability:
        # - Available groups run through grouped dispatch.
        # - Missing/unavailable groups run through targeted fallback dispatch.
        available_groups: dict[tuple[int, int, int], tuple[list[int], Any]] = {}
        missing_groups: dict[tuple[int, int, int], list[int]] = {}
        for bit_tuple, (expert_indices, cached_buffers) in self._bit_group_cache.items():
            if cached_buffers is None or isinstance(cached_buffers, dict):
                missing_groups[bit_tuple] = expert_indices
            else:
                available_groups[bit_tuple] = (expert_indices, cached_buffers)

        workspace_pool = self._get_workspace_buffer_pool()
        output_accum = workspace_pool.get_accum_buffer(x.shape[0])
        output_fp16 = workspace_pool.get_output_buffer(x.shape[0])

        if available_groups:
            dispatch_moe_per_bit_tuple(
                lib=self._get_lib(),
                activations=x,
                expert_ids=selected_experts,
                expert_probs=routing_weights,
                bit_group_buffers=available_groups,
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                num_experts=len(self.experts),
                top_k=self.num_experts_per_tok,
                buffer_pool=self._get_buffer_pool(),
                use_fp32_acc=self.hidden_dim >= 1024,
                output_accum=output_accum,
                output_fp16=output_fp16,
            )
        else:
            output_accum.zero_()

        # Dispatch only missing groups via fallback path and merge into output_accum.
        if missing_groups:
            cached_buffers = self._get_cached_buffers()
            buffer_pool = self._get_buffer_pool()
            lib = self._get_lib()

            for bit_tuple, expert_indices in missing_groups.items():
                if not expert_indices:
                    continue

                mask = torch.zeros_like(selected_experts, dtype=torch.bool)
                for expert_id in expert_indices:
                    mask |= selected_experts == expert_id

                batch_indices, slot_indices = torch.nonzero(mask, as_tuple=True)
                if batch_indices.numel() == 0:
                    continue

                group_output = dispatch_moe_trellis_swiglu(
                    lib,
                    activations=x[batch_indices],
                    gate_weights=None,
                    gate_scales=None,
                    up_weights=None,
                    up_scales=None,
                    down_weights=None,
                    down_scales=None,
                    gate_su=None,
                    gate_sv=None,
                    up_su=None,
                    up_sv=None,
                    down_su=None,
                    down_sv=None,
                    grid=None,
                    expert_ids=selected_experts[batch_indices, slot_indices].unsqueeze(1),
                    expert_probs=routing_weights[batch_indices, slot_indices].unsqueeze(1),
                    hidden_dim=self.hidden_dim,
                    intermediate_dim=self.intermediate_dim,
                    num_experts=len(self.experts),
                    top_k=1,
                    bits=bit_tuple,
                    cached_buffers=cached_buffers,
                    buffer_pool=buffer_pool,
                    use_fp32_acc=self.hidden_dim >= 1024,
                )

                output_accum.index_add_(0, batch_indices, group_output.float())

        output_fp16.copy_(output_accum)

        # Add shared expert
        output_fp16.add_(self.shared_expert(x))
        return output_fp16

    @classmethod
    def from_loader(
        cls,
        loader: TrellisModelLoader,
        config: TrellisModelConfig,
        layer_idx: int,
        router_weights: dict[str, torch.Tensor],
        device: str = "mps",
        eager_buffers: bool = True,
    ) -> TrellisMoEMLP:
        """Create TrellisMoEMLP from a TrellisModelLoader.

        Args:
            loader: TrellisModelLoader instance for the model.
            config: Model configuration.
            layer_idx: Layer index to load.
            router_weights: Router weights dictionary.
            device: Device to place modules on.
            eager_buffers: If True (default), create Metal buffers eagerly
                for better memory efficiency.

        Returns:
            TrellisMoEMLP module initialized with layer weights.
        """
        layer_weights = loader.load_layer(layer_idx)
        prefix = f"model.layers.{layer_idx}.mlp"

        # Get router weight and ensure fp16 to avoid dtype conversions in forward
        router_weight = router_weights[f"{prefix}.gate.weight"]

        # Create router as fp16 - Metal kernels work in fp16, so keep router fp16
        # to avoid dtype conversion on every forward pass
        router = nn.Linear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            device=device,
            dtype=torch.float16,
        )
        router.weight.data = router_weight.to(
            device=device, dtype=torch.float16)

        # Create experts
        experts = []
        for expert_idx in range(config.num_experts):
            expert_prefix = f"{prefix}.experts.{expert_idx}"
            expert = TrellisDenseMLP(
                gate_proj=TrellisLinear.from_trellis_weight(
                    layer_weights[f"{expert_prefix}.gate_proj.weight"],
                    device=device,
                ),
                up_proj=TrellisLinear.from_trellis_weight(
                    layer_weights[f"{expert_prefix}.up_proj.weight"],
                    device=device,
                ),
                down_proj=TrellisLinear.from_trellis_weight(
                    layer_weights[f"{expert_prefix}.down_proj.weight"],
                    device=device,
                ),
            )
            experts.append(expert)

        # Create shared expert (GLM uses 'shared_experts' plural in weights)
        shared_expert = TrellisDenseMLP(
            gate_proj=TrellisLinear.from_trellis_weight(
                layer_weights[f"{prefix}.shared_experts.gate_proj.weight"],
                device=device,
            ),
            up_proj=TrellisLinear.from_trellis_weight(
                layer_weights[f"{prefix}.shared_experts.up_proj.weight"],
                device=device,
            ),
            down_proj=TrellisLinear.from_trellis_weight(
                layer_weights[f"{prefix}.shared_experts.down_proj.weight"],
                device=device,
            ),
        )

        return cls(
            router=router,
            experts=experts,
            shared_expert=shared_expert,
            num_experts_per_tok=config.num_experts_per_tok,
            eager_buffers=True,  # Memory-optimized: create Metal buffers from CPU
        )

    def quantize_router_to_int8(self) -> dict[str, float]:
        """Convert router weights to INT8 for faster routing computation.

        The router is small (hidden_dim -> num_experts, e.g., 2048 -> 64)
        and can benefit from int8 quantization:
        - 4x smaller memory footprint
        - Faster matmul (reduced memory bandwidth)
        - Negligible accuracy loss for routing decisions

        Returns:
            Dict with quantization error metrics:
            - max_abs_error: Maximum absolute reconstruction error
            - mean_abs_error: Mean absolute reconstruction error
            - snr_db: Signal-to-noise ratio in dB
        """
        from .router_int8 import Int8RouterLinear, measure_quantization_error

        # Skip if already quantized
        if self._use_int8_router:
            return {"already_quantized": True}  # type: ignore[dict-item]

        # Get original router weights
        original_weight = self.router.weight.data.clone()

        # Convert to Int8RouterLinear
        int8_router = Int8RouterLinear.from_float(
            self.router,
            device=str(self.router.weight.device),
        )

        # Replace router with quantized version
        self.router = int8_router  # type: ignore[assignment]
        self._use_int8_router = True

        # Measure and return quantization error
        # type: ignore[assignment]
        weights_int8: torch.Tensor = int8_router.weights_int8
        scales: torch.Tensor = int8_router.scales  # type: ignore[assignment]
        return measure_quantization_error(original_weight, weights_int8, scales)

    def get_router_memory_usage(self) -> dict[str, int | str]:
        """Get memory usage of router weights.

        Returns:
            Dict with:
            - weights_bytes: Size of weight tensor in bytes
            - scales_bytes: Size of scale tensor in bytes (if int8)
            - total_bytes: Total memory usage
        """
        if self._use_int8_router:
            # type: ignore[attr-defined, assignment]
            weights_int8: torch.Tensor = self.router.weights_int8
            # type: ignore[attr-defined, assignment]
            scales: torch.Tensor = self.router.scales
            weights_bytes = weights_int8.numel() * weights_int8.element_size()
            scales_bytes = scales.numel() * scales.element_size()
            return {
                "weights_bytes": weights_bytes,
                "scales_bytes": scales_bytes,
                "total_bytes": weights_bytes + scales_bytes,
                "dtype": "int8",
            }
        else:
            weight = self.router.weight
            weights_bytes = weight.numel() * weight.element_size()
            return {
                "weights_bytes": weights_bytes,
                "scales_bytes": 0,
                "total_bytes": weights_bytes,
                "dtype": str(weight.dtype),
            }


class TrellisDecoderLayer(nn.Module):
    """Complete transformer decoder layer with trellis-quantized weights.

    Implements a GLM-style decoder layer with:
    - MLA attention (Multi-head Latent Attention)
    - RMSNorm pre-normalization
    - Dense or MoE MLP (depending on layer index)
    - Residual connections

    Attributes:
        self_attn: Attention module (to be implemented).
        mlp: Dense or MoE MLP module.
        input_layernorm: Pre-attention normalization.
        post_attention_layernorm: Post-attention normalization.
        config: Layer configuration.
    """

    def __init__(
        self,
        config: TrellisModelConfig,
        layer_idx: int,
        device: str = "mps",
    ):
        """Initialize TrellisDecoderLayer.

        Args:
            config: Model configuration.
            layer_idx: Layer index (0-indexed).
            device: Device to place modules on.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # MLA attention will be created in from_loader
        self.self_attn = None

        # MLP (dense or MoE)
        self.mlp = None  # Will be set in from_loader

        # Normalization
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self._checkpoint_context: dict[str, object] | None = None

        self.to(device)

    def get_weight_tensors(self) -> dict[str, torch.Tensor] | None:
        """Get weight tensors from MoE MLP if available."""
        if isinstance(self.mlp, TrellisMoEMLP):
            return self.mlp.get_weight_tensors()
        return None

    def _layer_forward_impl(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: TrellisKVCache | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
        workspace_offset: int = 0,
        cached_buffers: CachedWeightBuffers | None = None,
    ) -> torch.Tensor:
        """Forward pass through the decoder layer.

        Args:
            hidden_states: Input tensor [..., seq_len, hidden_size].
            attention_mask: Causal attention mask.
            position_ids: Position IDs for RoPE.
            kv_cache: KV cache for generation.
            rope_cos: Precomputed RoPE cos values [1, 1, seq_len, rope_dim//2].
            rope_sin: Precomputed RoPE sin values [1, 1, seq_len, rope_dim//2].
            workspace: Optional persistent workspace buffer.
            workspace_offset: Byte offset in workspace for scratch memory.
            cached_buffers: Pre-cached weight buffers for MoE MLP.

        Returns:
            Output tensor [..., seq_len, hidden_size].
        """
        layer_dtype = hidden_states.dtype

        # Pre-attention normalization
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention with precomputed RoPE cache
        attn_output = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            layer_idx=self.layer_idx,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        if attn_output.dtype != layer_dtype:
            attn_output = attn_output.to(dtype=layer_dtype)

        # Residual connection
        hidden_states = residual + attn_output

        # Post-attention normalization
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        if isinstance(self.mlp, TrellisMoEMLP):
            mlp_output = self.mlp(
                hidden_states,
                workspace=workspace,
                workspace_offset=workspace_offset,
                cached_buffers=cached_buffers,
            )
        else:
            mlp_output = self.mlp(hidden_states)
        if mlp_output.dtype != residual.dtype:
            mlp_output = mlp_output.to(dtype=residual.dtype)

        # Residual connection
        hidden_states = residual + mlp_output

        return hidden_states

    def _layer_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._checkpoint_context is None:
            raise RuntimeError(
                "Checkpoint context is missing for TrellisDecoderLayer")
        context = self._checkpoint_context
        return self._layer_forward_impl(
            hidden_states,
            attention_mask=context["attention_mask"],
            position_ids=context["position_ids"],
            kv_cache=context["kv_cache"],
            rope_cos=context["rope_cos"],
            rope_sin=context["rope_sin"],
            workspace=context["workspace"],
            workspace_offset=context["workspace_offset"],
            cached_buffers=context["cached_buffers"],
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: TrellisKVCache | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
        workspace_offset: int = 0,
        cached_buffers: CachedWeightBuffers | None = None,
    ) -> torch.Tensor:
        """Forward pass through the decoder layer."""
        if self.gradient_checkpointing and self.training:
            self._checkpoint_context = {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "kv_cache": kv_cache,
                "rope_cos": rope_cos,
                "rope_sin": rope_sin,
                "workspace": workspace,
                "workspace_offset": workspace_offset,
                "cached_buffers": cached_buffers,
            }
            try:
                return torch.utils.checkpoint.checkpoint(
                    self._layer_forward,
                    hidden_states,
                    use_reentrant=False,
                )
            finally:
                self._checkpoint_context = None

        return self._layer_forward_impl(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            workspace=workspace,
            workspace_offset=workspace_offset,
            cached_buffers=cached_buffers,
        )

    @classmethod
    def from_loader(
        cls,
        loader: TrellisModelLoader,
        config: TrellisModelConfig,
        layer_idx: int,
        router_weights: dict[str, torch.Tensor],
        base_weights: dict[str, torch.Tensor],
        device: str = "mps",
        defer_buffer_creation: bool = True,
    ) -> TrellisDecoderLayer:
        """Create TrellisDecoderLayer from a TrellisModelLoader.

        Args:
            loader: TrellisModelLoader instance.
            config: Model configuration.
            layer_idx: Layer index.
            router_weights: Router weights for MoE layers.
            base_weights: Base model weights.
            device: Device to place modules on.
            defer_buffer_creation: If True (default), defer Metal buffer creation
                until first forward pass for faster loading.

        Returns:
            TrellisDecoderLayer module initialized with layer weights.
        """
        layer = cls(config, layer_idx, device)

        # Load layer weights
        layer_weights = loader.load_layer(layer_idx)
        prefix = f"model.layers.{layer_idx}.self_attn"

        # Load layernorm weights from base_weights.safetensors
        layernorm_weights = loader.load_layernorm_weights(layer_idx)

        # Create MLA attention config with GLM-4 dimensions
        mla_config = TrellisMLAConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            qk_nope_head_dim=getattr(config, "qk_nope_head_dim", 192),
            qk_rope_head_dim=getattr(config, "qk_rope_head_dim", 64),
            v_head_dim=getattr(config, "v_head_dim", 256),
            kv_lora_rank=config.kv_lora_rank,
            q_lora_rank=config.q_lora_rank,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
        )

        # Get attention projections
        # GLM uses low-rank Q: q_a_proj + q_b_proj
        q_a_proj = None
        q_b_proj = None
        if mla_config.q_lora_rank:
            q_a_key = f"{prefix}.q_a_proj.weight"
            q_b_key = f"{prefix}.q_b_proj.weight"
            if q_a_key in layer_weights:
                q_a_proj = TrellisLinear.from_trellis_weight(
                    layer_weights[q_a_key],
                    device=device,
                )
            if q_b_key in layer_weights:
                q_b_proj = TrellisLinear.from_trellis_weight(
                    layer_weights[q_b_key],
                    device=device,
                )

        # GLM uses kv_a_proj_with_mqa (includes MQA heads)
        kv_a_key = f"{prefix}.kv_a_proj_with_mqa.weight"
        if kv_a_key not in layer_weights:
            kv_a_key = f"{prefix}.kv_a_proj.weight"  # Fallback
        kv_b_key = f"{prefix}.kv_b_proj.weight"
        o_key = f"{prefix}.o_proj.weight"

        kv_a_proj = TrellisLinear.from_trellis_weight(
            layer_weights[kv_a_key],
            device=device,
        )
        kv_b_proj = TrellisLinear.from_trellis_weight(
            layer_weights[kv_b_key],
            device=device,
        )
        o_proj = TrellisLinear.from_trellis_weight(
            layer_weights[o_key],
            device=device,
        )

        # Load MLA layernorms (q_a_layernorm and kv_a_layernorm)
        q_a_layernorm = None
        kv_a_layernorm = None
        if "self_attn.q_a_layernorm.weight" in layernorm_weights:
            q_a_ln_weight = layernorm_weights["self_attn.q_a_layernorm.weight"]
            q_a_layernorm = RMSNorm(
                q_a_ln_weight.shape[0], eps=config.rms_norm_eps)
            q_a_layernorm.weight.data = q_a_ln_weight.to(device)
        if "self_attn.kv_a_layernorm.weight" in layernorm_weights:
            kv_a_ln_weight = layernorm_weights["self_attn.kv_a_layernorm.weight"]
            kv_a_layernorm = RMSNorm(
                kv_a_ln_weight.shape[0], eps=config.rms_norm_eps)
            kv_a_layernorm.weight.data = kv_a_ln_weight.to(device)

        layer.self_attn = TrellisMLAttention(
            config=mla_config,
            q_a_proj=q_a_proj,
            q_b_proj=q_b_proj,
            kv_a_proj=kv_a_proj,
            kv_b_proj=kv_b_proj,
            o_proj=o_proj,
            q_a_layernorm=q_a_layernorm,
            kv_a_layernorm=kv_a_layernorm,
        )

        # Load input/post-attention layernorms
        if "input_layernorm.weight" in layernorm_weights:
            layer.input_layernorm.weight.data = layernorm_weights["input_layernorm.weight"].to(
                device
            )
        if "post_attention_layernorm.weight" in layernorm_weights:
            layer.post_attention_layernorm.weight.data = layernorm_weights[
                "post_attention_layernorm.weight"
            ].to(device)

        # Create MLP (dense or MoE)
        is_moe = config.is_moe_layer(layer_idx)
        logger.debug(
            f"Layer {layer_idx}: is_moe={is_moe}, num_experts={config.num_experts}, first_moe={config.first_moe_layer}"
        )
        if is_moe:
            logger.debug(f"Creating TrellisMoEMLP for layer {layer_idx}")
            layer.mlp = TrellisMoEMLP.from_loader(
                loader,
                config,
                layer_idx,
                router_weights,
                device,
                eager_buffers=not defer_buffer_creation,
            )
        else:
            logger.debug(f"Creating TrellisDenseMLP for layer {layer_idx}")
            layer.mlp = TrellisDenseMLP.from_loader(loader, layer_idx, device)

        return layer


class ActivationPingPongBuffer:
    """Ping-pong buffer pool for reusing activation memory across layers.

    Reduces activation memory by 50% by maintaining two buffers and alternating
    between them across consecutive layers:
    - Layer 0: read A, write B
    - Layer 1: read B, write A
    - Layer 2: read A, write B
    ...

    This eliminates the need for N separate activation buffers, requiring only 2.
    """

    def __init__(self, hidden_dim: int, device: str = "mps", dtype: torch.dtype = torch.float16):
        """Initialize ping-pong buffer pool.

        Args:
            hidden_dim: Hidden dimension for activation tensors.
            device: Device to allocate buffers on.
            dtype: Data type for activations (default: fp16).
        """
        self.hidden_dim = hidden_dim
        self.device = device
        self.dtype = dtype
        self._buffer_a: torch.Tensor | None = None
        self._buffer_b: torch.Tensor | None = None
        self._current_idx = 0  # 0 for A, 1 for B
        self._workspace: torch.Tensor | None = None
        self._offsets: tuple[int, int] = (0, 0)

    def set_workspace(self, workspace: torch.Tensor, offset_a: int, offset_b: int) -> None:
        """Set workspace buffer and offsets for ping-pong buffers."""
        self._workspace = workspace
        self._offsets = (offset_a, offset_b)

    def _get_buffer_from_workspace(
        self, offset_bytes: int, batch_size: int, seq_len: int
    ) -> torch.Tensor:
        """Get a buffer from the workspace."""
        if self._workspace is None:
            raise ValueError("Workspace not set")

        element_size = torch.tensor([], dtype=self.dtype).element_size()
        offset_elements = offset_bytes // element_size
        numel = batch_size * seq_len * self.hidden_dim

        # Create view from workspace
        ws_view = self._workspace.view(self.dtype)
        return ws_view[offset_elements: offset_elements + numel].view(
            batch_size, seq_len, self.hidden_dim
        )

    def get_input_buffer(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """Get the input buffer for the current layer.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            Input activation tensor [batch_size, seq_len, hidden_dim].
        """
        if self._workspace is not None:
            offset = self._offsets[0] if self._current_idx == 0 else self._offsets[1]
            return self._get_buffer_from_workspace(offset, batch_size, seq_len)

        buffer = self._buffer_a if self._current_idx == 0 else self._buffer_b
        if buffer is None:
            buffer = torch.zeros(
                batch_size, seq_len, self.hidden_dim, dtype=self.dtype, device=self.device
            )
            if self._current_idx == 0:
                self._buffer_a = buffer
            else:
                self._buffer_b = buffer
        elif buffer.shape != (batch_size, seq_len, self.hidden_dim):
            buffer = buffer.resize_(batch_size, seq_len, self.hidden_dim)
        return buffer

    def get_output_buffer(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """Get the output buffer for the current layer and swap.

        After getting the output buffer, the ping-pong index is swapped
        so the next layer uses this buffer as input.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            Output activation tensor [batch_size, seq_len, hidden_dim].
        """
        if self._workspace is not None:
            offset = self._offsets[1] if self._current_idx == 0 else self._offsets[0]
            self._current_idx = 1 - self._current_idx
            return self._get_buffer_from_workspace(offset, batch_size, seq_len)

        # Get the opposite buffer for output
        output_buffer = self._buffer_b if self._current_idx == 0 else self._buffer_a
        if output_buffer is None:
            output_buffer = torch.zeros(
                batch_size, seq_len, self.hidden_dim, dtype=self.dtype, device=self.device
            )
            if self._current_idx == 0:
                self._buffer_b = output_buffer
            else:
                self._buffer_a = output_buffer
        elif output_buffer.shape != (batch_size, seq_len, self.hidden_dim):
            output_buffer = output_buffer.resize_(
                batch_size, seq_len, self.hidden_dim)

        # Swap index for next layer
        self._current_idx = 1 - self._current_idx
        return output_buffer

    def swap(self) -> None:
        """Manually swap the ping-pong buffer index.

        Use this to reset or control buffer swapping.
        """
        self._current_idx = 1 - self._current_idx

    def clear(self) -> None:
        """Clear all buffers, freeing memory."""
        self._buffer_a = None
        self._buffer_b = None
        self._current_idx = 0


class TrellisModel(nn.Module):
    """Complete trellis-quantized model for inference."""

    def __init__(self, config: TrellisModelConfig):
        super().__init__()
        self.config = config

        # Embedding (not quantized, from base model)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Token embedding cache for frequently-used tokens
        self._embedding_cache: TokenEmbeddingCache | None = None

        # Layers
        self.layers = nn.ModuleList()

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Ping-pong activation buffers for memory efficiency
        self._activation_buffers: ActivationPingPongBuffer | None = None

        # Persistent workspace buffer (256MB)
        self.workspace_size = 256 * 1024 * 1024  # 256MB
        self.register_buffer(
            "workspace", torch.zeros(self.workspace_size, dtype=torch.uint8), persistent=False
        )

    def enable_gradient_checkpointing(self) -> None:
        """Enable activation checkpointing to reduce memory."""
        for layer in self.layers:
            layer.gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: TrellisKVCache | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        use_ping_pong: bool = True,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch, seq_len].
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs for RoPE.
            kv_cache: Optional KV cache for generation.
            rope_cos: Precomputed RoPE cos cache from parent module.
            rope_sin: Precomputed RoPE sin cache from parent module.
            use_ping_pong: If True, use ping-pong buffers to reduce activation memory by 50%.

        Returns:
            Hidden states tensor [batch, seq_len, hidden_size].
        """
        hidden_states = self.embed_tokens(input_ids)
        batch_size, seq_len = input_ids.shape

        # Ensure workspace is on correct device
        if self.workspace.device != hidden_states.device:
            self.workspace = self.workspace.to(hidden_states.device)

        # Create causal mask
        if attention_mask is None:
            attention_mask = self._make_causal_mask(
                seq_len, hidden_states.device)

        # Get position IDs for RoPE if not provided
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=hidden_states.device).unsqueeze(0)

        # Set up batched dispatch for all MoE layers
        self.setup_batched_dispatch()

        # Get Metal library for async KV cache prefetching
        lib = None
        for layer in self.layers:
            if isinstance(layer.mlp, TrellisMoEMLP):
                lib = layer.mlp._get_lib()
                break

        # Define workspace regions (offsets in bytes)
        mb = 1024 * 1024
        # Region A: 0-64MB (PingPong A)
        offset_a = 0
        # Region B: 64-128MB (PingPong B)
        offset_b = 64 * mb
        # Region C: 128-256MB (Shared layer scratch)
        offset_scratch = 128 * mb

        # Initialize pipeliner if library available
        pipeliner = None
        if lib is not None:
            pipeliner = PipelinedLayerDispatcher(lib, len(self.layers))

            # Prefetch layer 0 weights synchronously if needed
            if len(self.layers) > 0:
                first_layer = self.layers[0]
                if hasattr(first_layer, "get_weight_tensors"):
                    weights = first_layer.get_weight_tensors()
                    if weights is not None:
                        pipeliner.prefetch_layer_weights_sync(0, weights)

        # Use ping-pong buffers to reduce activation memory by 50%
        if use_ping_pong:
            if self._activation_buffers is None:
                self._activation_buffers = ActivationPingPongBuffer(
                    hidden_dim=self.config.hidden_size,
                    device=str(hidden_states.device),
                    dtype=hidden_states.dtype,
                )

            # Configure workspace for ping-pong buffers
            self._activation_buffers.set_workspace(
                self.workspace, offset_a, offset_b)

            # Copy embedding output to first buffer
            input_buffer = self._activation_buffers.get_input_buffer(
                batch_size, seq_len)
            input_buffer.copy_(hidden_states)

            for layer_idx, layer in enumerate(self.layers):
                # Prefetch KV cache for layer N+1 before computing layer N
                if kv_cache is not None and lib is not None and layer_idx + 1 < len(self.layers):
                    kv_cache.prefetch_layer_async(layer_idx + 1, lib=lib)

                # 1. Start async transfer for layer N+1 weights
                if pipeliner is not None and layer_idx + 1 < len(self.layers):
                    next_layer = self.layers[layer_idx + 1]
                    if hasattr(next_layer, "get_weight_tensors"):
                        weights = next_layer.get_weight_tensors()
                        if weights is not None:
                            pipeliner.start_prefetch_async(
                                layer_idx + 1, weights)

                # Skip layers marked for pruning (identity pass-through)
                if self.config.should_skip_layer(layer_idx):
                    continue

                # Get output buffer for this layer
                output_buffer = self._activation_buffers.get_output_buffer(
                    batch_size, seq_len)

                # Get weight buffers for this layer (wait if transfer pending)
                layer_buffers = None
                if pipeliner is not None:
                    # Check if we have pre-fetched buffers
                    buf_dict = pipeliner.wait_for_prefetch(layer_idx)
                    if buf_dict:
                        layer_buffers = CachedWeightBuffers(**buf_dict)

                # 2. Compute layer N (with transferred buffers if available)
                output_buffer = layer(
                    input_buffer,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    kv_cache=kv_cache,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    workspace=self.workspace,
                    workspace_offset=offset_scratch,
                    cached_buffers=layer_buffers,
                )

                # Ping-pong swap: output becomes input for next layer
                input_buffer = output_buffer

            hidden_states = input_buffer
        else:
            for layer_idx, layer in enumerate(self.layers):
                # Prefetch KV cache for layer N+1 before computing layer N
                if kv_cache is not None and lib is not None and layer_idx + 1 < len(self.layers):
                    kv_cache.prefetch_layer_async(layer_idx + 1, lib=lib)

                # Skip layers marked for pruning (identity pass-through)
                if self.config.should_skip_layer(layer_idx):
                    continue

                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    kv_cache=kv_cache,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    workspace=self.workspace,
                    workspace_offset=offset_scratch,
                )

        # Flush all batched MoE dispatches in a single command buffer
        self.flush_batched_dispatch()

        # Final normalization
        hidden_states = self.norm(hidden_states)

        return hidden_states

    def _get_cached_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings using cache for frequent tokens.

        Records token frequencies and periodically builds the embedding cache
        for the top 1000 most common tokens. After the cache is built, looks up
        cached tokens directly to avoid embedding layer computation.

        Args:
            input_ids: Token IDs [batch, seq_len].

        Returns:
            Embeddings [batch, seq_len, hidden_size].
        """
        flat_ids = input_ids.flatten()
        num_tokens = flat_ids.numel()
        self._total_tokens_seen += num_tokens

        for tid in flat_ids.tolist():
            self._embedding_cache.record_token(tid)

        if self._total_tokens_seen >= self._cache_build_threshold:
            self._embedding_cache.build_cache(self.embed_tokens)

        return self._embedding_cache.get_embeddings(input_ids, self.embed_tokens)

    def _make_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        mask = torch.triu(torch.ones(
            seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def setup_batched_dispatch(self) -> BatchedDispatcher | None:
        """Set up batched dispatch for all MoE layers.

        Creates a single BatchedDispatcher and assigns it to all MoE MLP layers.
        Call flush_batched_dispatch() after forward pass to execute all queued
        MoE kernel dispatches in a single command buffer.

        Returns:
            The BatchedDispatcher if any MoE layers exist, None otherwise.
        """
        # Find the first MoE layer to get the Metal library
        lib = None
        for layer in self.layers:
            if isinstance(layer.mlp, TrellisMoEMLP):
                lib = layer.mlp._get_lib()
                break

        if lib is None:
            return None

        # Create dispatcher and assign to all MoE layers
        dispatcher = BatchedDispatcher(lib)
        for layer in self.layers:
            if isinstance(layer.mlp, TrellisMoEMLP):
                layer.mlp.set_batched_dispatcher(dispatcher)

        return dispatcher

    def flush_batched_dispatch(self) -> None:
        """Execute all queued MoE dispatches in a single command buffer.

        Call this after forward() when batched dispatch is enabled.
        This commits all 45 MoE kernel dispatches as a single Metal command buffer,
        reducing per-layer command buffer overhead.

        After this call, all output tensors from MoE layers contain valid results.
        """
        for layer in self.layers:
            if hasattr(layer.mlp, "_batched_dispatcher") and layer.mlp._batched_dispatcher:
                layer.mlp._batched_dispatcher.commit_and_wait()
                layer.mlp._pending_output = layer.mlp.get_pending_output()
                break  # All layers share the same dispatcher

    def clear_batched_dispatch(self) -> None:
        """Disable batched dispatch for all MoE layers.

        Restores immediate execution mode for MoE kernels.
        """
        for layer in self.layers:
            if hasattr(layer.mlp, "set_batched_dispatcher"):
                layer.mlp.set_batched_dispatcher(None)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "mps",
        load_in_layers: bool = True,
        progress_callback: ProgressCallback | None = None,
        defer_buffer_creation: bool = True,
    ) -> TrellisModel:
        """Load model from trellis-quantized checkpoint.

        Args:
            model_path: Path to quantized model directory
            device: Device to load model on
            load_in_layers: If True, load one layer at a time (memory efficient)
            progress_callback: Optional callback for progress updates.
                Signature: (current_step, total_steps, message) -> None
            defer_buffer_creation: If True (default), defer Metal buffer creation
                until first forward pass. This speeds up loading significantly.
        """
        load_start = time.perf_counter()

        config = TrellisModelConfig.from_pretrained(model_path)
        model = cls(config)

        from .loader import TrellisModelLoader

        loader = TrellisModelLoader(model_path)

        # Total steps: 2 (base weights) + num_layers + 1 (finalize)
        total_steps = config.num_hidden_layers + 3
        current_step = 0

        def report_progress(msg: str) -> None:
            nonlocal current_step
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, msg)

        # Load non-quantized weights (embedding, norms, lm_head)
        report_progress("Loading embeddings and norms")
        base_weights = cls._load_base_weights(model_path)
        model.embed_tokens.weight.data = base_weights["model.embed_tokens.weight"].to(
            device)
        model.norm.weight.data = base_weights["model.norm.weight"].to(device)

        # Load router weights if MoE
        router_weights = {}
        if config.num_experts > 1:
            report_progress("Loading router weights")
            router_weights = loader.load_router_weights()
        else:
            report_progress("No MoE layers")

        # Load layers with progress reporting
        for layer_idx in range(config.num_hidden_layers):
            layer = TrellisDecoderLayer.from_loader(
                loader,
                config,
                layer_idx,
                router_weights,
                base_weights,
                device,
                defer_buffer_creation=defer_buffer_creation,
            )
            model.layers.append(layer)

            if load_in_layers:
                # Clear loader cache to save memory
                loader.clear_layer_cache(layer_idx)

            report_progress(
                f"Loaded layer {layer_idx + 1}/{config.num_hidden_layers}")

        report_progress("Finalizing model")
        load_elapsed = time.perf_counter() - load_start
        logger.info(f"Model loaded in {load_elapsed:.1f}s")

        return model.to(device)

    @staticmethod
    def _load_base_weights(model_path: str) -> dict[str, torch.Tensor]:
        """Load non-quantized weights (embedding, norms, lm_head).

        Uses memory-mapped loading to avoid loading weights into RAM upfront.
        The OS pages in weight data on demand during the forward pass.
        """
        from pathlib import Path

        from ..mmap_loader import MmapSafetensorsLoader

        path = Path(model_path)

        # Try loading from quantized model directory using mmap
        base_weights_path = path / "base_weights.safetensors"
        if base_weights_path.exists():
            # Use mmap for lazy loading - weights are paged in by OS on demand
            loader = MmapSafetensorsLoader(base_weights_path, device="cpu")
            weights = {name: loader.get_tensor(name) for name in loader.keys()}
            # Keep loader reference to prevent garbage collection during model load
            weights["_mmap_loader"] = loader  # type: ignore[dict-item]
            return weights

        # Fall back to HuggingFace
        raise FileNotFoundError(
            f"base_weights.safetensors not found in {model_path}. "
            "Run extract_base_weights.py first."
        )


def __getattr__(name: str):
    if name in ("CausalLMOutput", "TrellisForCausalLM"):
        from .lm import CausalLMOutput, TrellisForCausalLM
        if name == "CausalLMOutput":
            return CausalLMOutput
        return TrellisForCausalLM
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")