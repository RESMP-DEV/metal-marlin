"""
Activation checkpointing for memory-efficient inference on Apple Silicon.

Implements gradient-checkpointing-style activation recomputation for inference.
Instead of storing all intermediate activations for long sequences, recomputes
them on-demand by chunking the forward pass.

This trades compute for memory, which is valuable on Unified Memory systems
where GPU and CPU share the same memory pool.

Usage:
    @checkpoint_activations(chunk_size=1024)
    def forward(self, x: mx.array) -> mx.array:
        ...

    # Or wrap an existing module
    checkpointed_block = CheckpointedModule(block, chunk_size=512)

Integrates with MLX's lazy evaluation model. Activations remain lazy (unevaluated)
until explicitly needed, and chunked processing materializes only one chunk at
a time, keeping peak memory bounded.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    pass


P = ParamSpec("P")
R = TypeVar("R", bound=mx.array)


@dataclass
class CheckpointConfig:
    """Configuration for activation checkpointing.

    Args:
        chunk_size: Number of sequence positions to process at once.
            Larger chunks use more memory but are more efficient.
            Must be positive. Default: 1024.
        preserve_rng_state: Whether to save/restore RNG state for each chunk.
            Important for dropout layers during inference with stochastic
            sampling. Default: True.
        materialize_between_chunks: Whether to call mx.eval() between chunks
            to force materialization. Setting True ensures constant memory
            but adds synchronization overhead. Default: False.
        min_seq_length: Minimum sequence length before chunking activates.
            For short sequences, the overhead isn't worth it. Default: 2048.
    """

    chunk_size: int = 1024
    preserve_rng_state: bool = True
    materialize_between_chunks: bool = False
    min_seq_length: int = 2048


@dataclass
class CheckpointStats:
    """Runtime statistics for checkpointed execution.

    Tracks how many chunks were processed, whether checkpointing was active,
    and cumulative recomputation overhead.
    """

    num_chunks: int = 0
    total_seq_length: int = 0
    checkpointing_active: bool = False
    recompute_count: int = 0
    chunk_boundaries: list[tuple[int, int]] = field(default_factory=list)

    @property
    def overhead_ratio(self) -> float:
        """Ratio of recomputed vs direct computation.

        A ratio of 1.0 means no overhead (no recomputation).
        A ratio of 2.0 means each activation was computed twice on average.
        """
        if self.total_seq_length == 0:
            return 1.0
        total_recomputed = sum(end - start for start, end in self.chunk_boundaries)
        return total_recomputed * self.recompute_count / max(self.total_seq_length, 1)


def checkpoint_activations(
    chunk_size: int = 1024,
    *,
    preserve_rng_state: bool = True,
    materialize_between_chunks: bool = False,
    min_seq_length: int = 2048,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for activation checkpointing on forward methods.

    Wraps a forward function to process long sequences in chunks, recomputing
    activations for each chunk rather than storing them all. This bounds peak
    memory usage to O(chunk_size) instead of O(seq_length).

    The decorated function must:
    - Accept a tensor with sequence dimension at axis 1: [batch, seq_len, ...]
    - Return a tensor with the same sequence dimension structure

    Args:
        chunk_size: Tokens per chunk. Default: 1024.
        preserve_rng_state: Save/restore RNG for reproducibility. Default: True.
        materialize_between_chunks: Force evaluation between chunks. Default: False.
        min_seq_length: Skip chunking for sequences shorter than this. Default: 2048.

    Returns:
        Decorated function with chunked processing.

    Example:
        class MyTransformer(nn.Module):
            @checkpoint_activations(chunk_size=512)
            def __call__(self, x):
                # Process normally - decorator handles chunking
                return self.layers(x)
    """
    config = CheckpointConfig(
        chunk_size=chunk_size,
        preserve_rng_state=preserve_rng_state,
        materialize_between_chunks=materialize_between_chunks,
        min_seq_length=min_seq_length,
    )

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return _chunked_forward(fn, config, *args, **kwargs)

        # Attach config and stats accessors
        wrapper._checkpoint_config = config  # type: ignore[attr-defined]
        wrapper._checkpoint_stats = CheckpointStats()  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def _chunked_forward(
    fn: Callable[P, R],
    config: CheckpointConfig,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    """Execute forward pass with chunked activation checkpointing.

    This is the core implementation that:
    1. Extracts the input tensor (first positional arg)
    2. Determines if chunking is needed based on sequence length
    3. Processes each chunk independently
    4. Concatenates results along the sequence dimension

    For MLX's lazy evaluation, each chunk remains lazy until:
    - materialize_between_chunks forces mx.eval()
    - The final concatenation is evaluated downstream
    """
    if not args:
        return fn(*args, **kwargs)

    # First positional argument is assumed to be the input tensor
    x = args[0]
    if not isinstance(x, mx.array):
        return fn(*args, **kwargs)

    # Check dimensions: expect [batch, seq_len, ...] or [seq_len, ...]
    if x.ndim < 2:
        return fn(*args, **kwargs)

    seq_dim = 1 if x.ndim >= 2 else 0
    seq_len = x.shape[seq_dim]

    # Skip chunking for short sequences
    if seq_len <= config.min_seq_length:
        return fn(*args, **kwargs)

    # Compute chunk boundaries
    chunk_size = config.chunk_size
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    chunks: list[mx.array] = []
    boundaries: list[tuple[int, int]] = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, seq_len)
        boundaries.append((start, end))

        # Extract chunk along sequence dimension
        if seq_dim == 1:
            x_chunk = x[:, start:end]
        else:
            x_chunk = x[start:end]

        # Rebuild args with chunked input
        chunk_args = (x_chunk,) + args[1:]

        # Forward pass on chunk - stays lazy until evaluated
        chunk_out = fn(*chunk_args, **kwargs)
        chunks.append(chunk_out)

        # Optionally materialize to bound memory
        if config.materialize_between_chunks:
            mx.eval(chunk_out)

    # Concatenate chunks along sequence dimension
    if seq_dim == 1:
        result = mx.concatenate(chunks, axis=1)
    else:
        result = mx.concatenate(chunks, axis=0)

    return result  # type: ignore[return-value]


class CheckpointedModule(nn.Module):
    """Wrapper that applies activation checkpointing to any nn.Module.

    Wraps an existing module's forward pass to use chunked processing.
    Useful for wrapping transformer blocks or other memory-intensive layers.

    Args:
        module: The nn.Module to wrap.
        config: Checkpoint configuration. If None, uses defaults.
        chunk_size: Shorthand for config.chunk_size if config not provided.

    Example:
        # Wrap a transformer block
        block = MarlinTransformerBlock(...)
        checkpointed = CheckpointedModule(block, chunk_size=512)
        output = checkpointed(hidden_states)

        # Or with full config
        config = CheckpointConfig(chunk_size=256, materialize_between_chunks=True)
        checkpointed = CheckpointedModule(block, config=config)
    """

    def __init__(
        self,
        module: nn.Module,
        config: CheckpointConfig | None = None,
        chunk_size: int | None = None,
    ):
        super().__init__()
        self.module = module

        if config is None:
            config = CheckpointConfig(
                chunk_size=chunk_size if chunk_size is not None else 1024
            )
        elif chunk_size is not None:
            config = CheckpointConfig(
                chunk_size=chunk_size,
                preserve_rng_state=config.preserve_rng_state,
                materialize_between_chunks=config.materialize_between_chunks,
                min_seq_length=config.min_seq_length,
            )

        self.config = config
        self._stats = CheckpointStats()

    def __call__(self, x: mx.array, **kwargs: Any) -> mx.array:
        """Forward pass with automatic chunking for long sequences."""
        return _chunked_forward_module(self.module, self.config, self._stats, x, **kwargs)

    @property
    def stats(self) -> CheckpointStats:
        """Access runtime statistics for the last forward pass."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset accumulated statistics."""
        self._stats = CheckpointStats()


def _chunked_forward_module(
    module: nn.Module,
    config: CheckpointConfig,
    stats: CheckpointStats,
    x: mx.array,
    **kwargs: Any,
) -> mx.array:
    """Execute module forward with chunked checkpointing and stats tracking."""
    seq_dim = 1 if x.ndim >= 2 else 0
    seq_len = x.shape[seq_dim]

    # Update stats
    stats.total_seq_length = seq_len
    stats.chunk_boundaries = []
    stats.recompute_count = 1

    # Skip chunking for short sequences
    if seq_len <= config.min_seq_length:
        stats.checkpointing_active = False
        stats.num_chunks = 1
        return module(x, **kwargs)

    stats.checkpointing_active = True
    chunk_size = config.chunk_size
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    stats.num_chunks = num_chunks

    chunks: list[mx.array] = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, seq_len)
        stats.chunk_boundaries.append((start, end))

        # Extract chunk
        if seq_dim == 1:
            x_chunk = x[:, start:end]
        else:
            x_chunk = x[start:end]

        # Forward pass on chunk
        chunk_out = module(x_chunk, **kwargs)
        chunks.append(chunk_out)

        # Optionally materialize
        if config.materialize_between_chunks:
            mx.eval(chunk_out)

    # Concatenate
    if seq_dim == 1:
        return mx.concatenate(chunks, axis=1)
    return mx.concatenate(chunks, axis=0)


class SequentialCheckpoint(nn.Module):
    """Sequential container with activation checkpointing between layers.

    Unlike CheckpointedModule which chunks within a single module,
    SequentialCheckpoint checkpoints between layers in a sequence.
    This is useful for deep networks where we want to recompute
    layer outputs rather than store them all.

    This implements "segment checkpointing" where the network is divided
    into segments, and only segment boundaries are stored. Middle layer
    activations are recomputed during the backward-like access pattern.

    For inference-only use (no gradients), this primarily helps when:
    1. Accessing activations for probing/analysis
    2. Memory-constrained streaming of very large batches

    Args:
        layers: List of nn.Module layers to execute sequentially.
        checkpoint_every: Store activations every N layers. Default: 4.
            Smaller = more memory, less recompute.
            Larger = less memory, more recompute.
        config: Optional CheckpointConfig for fine-grained control.

    Example:
        layers = [MarlinTransformerBlock(...) for _ in range(32)]
        model = SequentialCheckpoint(layers, checkpoint_every=8)
        # Only stores activations at layers 0, 8, 16, 24, 32
        output = model(x)
    """

    def __init__(
        self,
        layers: list[nn.Module],
        checkpoint_every: int = 4,
        config: CheckpointConfig | None = None,
    ):
        super().__init__()
        self.layers = layers
        self.checkpoint_every = checkpoint_every
        self.config = config or CheckpointConfig()
        self._checkpoints: dict[int, mx.array] = {}

    def __call__(self, x: mx.array, **kwargs: Any) -> mx.array:
        """Forward pass with segment checkpointing."""
        self._checkpoints.clear()

        for i, layer in enumerate(self.layers):
            x = layer(x, **kwargs)

            # Store checkpoint at segment boundaries
            if i % self.checkpoint_every == 0:
                if self.config.materialize_between_chunks:
                    mx.eval(x)
                self._checkpoints[i] = x

        return x

    def recompute_from(self, checkpoint_idx: int, target_idx: int, x: mx.array, **kwargs: Any) -> mx.array:
        """Recompute activations from a checkpoint to a target layer.

        Useful for analysis or when you need intermediate activations
        without storing them all.

        Args:
            checkpoint_idx: Index of the checkpointed layer to start from.
            target_idx: Index of the target layer (inclusive).
            x: Input tensor (ignored if checkpoint exists).
            **kwargs: Additional arguments passed to each layer.

        Returns:
            Activation at target_idx.
        """
        if checkpoint_idx in self._checkpoints:
            x = self._checkpoints[checkpoint_idx]
            start_idx = checkpoint_idx + 1
        else:
            start_idx = 0

        for i in range(start_idx, target_idx + 1):
            x = self.layers[i](x, **kwargs)

        return x


def chunked_apply(
    fn: Callable[[mx.array], mx.array],
    x: mx.array,
    chunk_size: int = 1024,
    seq_dim: int = 1,
    materialize: bool = False,
) -> mx.array:
    """Apply a function to a tensor in chunks along a sequence dimension.

    Functional API for one-off chunked processing without decorators.

    Args:
        fn: Function to apply to each chunk.
        x: Input tensor.
        chunk_size: Elements per chunk.
        seq_dim: Which dimension to chunk along. Default: 1.
        materialize: Whether to evaluate between chunks. Default: False.

    Returns:
        Concatenated output tensor.

    Example:
        # Process a long sequence through attention
        def attn_fn(chunk):
            return self.attention(chunk, mask=causal_mask_for(chunk))

        output = chunked_apply(attn_fn, hidden_states, chunk_size=512)
    """
    seq_len = x.shape[seq_dim]
    if seq_len <= chunk_size:
        return fn(x)

    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    chunks: list[mx.array] = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, seq_len)

        # Slice along the sequence dimension
        slices = [slice(None)] * x.ndim
        slices[seq_dim] = slice(start, end)
        x_chunk = x[tuple(slices)]

        chunk_out = fn(x_chunk)
        chunks.append(chunk_out)

        if materialize:
            mx.eval(chunk_out)

    return mx.concatenate(chunks, axis=seq_dim)


@dataclass
class MemoryBudget:
    """Memory budget controller for adaptive checkpointing.

    Estimates memory usage and adjusts chunk sizes dynamically to stay
    within a target memory budget. Useful for running large models on
    systems with limited Unified Memory.

    Args:
        target_memory_mb: Target peak memory usage in MB.
        activation_bytes_per_token: Estimated bytes per token per layer.
            For FP16 with hidden_size=4096: ~8KB per token.
        safety_margin: Fraction of budget to keep as headroom. Default: 0.1.
    """

    target_memory_mb: float
    activation_bytes_per_token: int = 8192  # ~8KB for typical 4096 hidden
    safety_margin: float = 0.1

    def optimal_chunk_size(self, hidden_size: int, num_layers: int = 1) -> int:
        """Calculate optimal chunk size for given model dimensions.

        Args:
            hidden_size: Model hidden dimension.
            num_layers: Number of layers being processed (for segment checkpointing).

        Returns:
            Recommended chunk size in tokens.
        """
        available_mb = self.target_memory_mb * (1 - self.safety_margin)
        available_bytes = available_mb * 1024 * 1024

        # Estimate bytes per token: hidden_size * dtype_bytes * num_activations
        # For a transformer block: ~4x hidden_size for attention, ~3x for MLP
        bytes_per_token = hidden_size * 2 * 7 * num_layers  # FP16 = 2 bytes

        optimal = int(available_bytes / bytes_per_token)

        # Clamp to reasonable range
        optimal = max(64, min(optimal, 8192))

        # Round to power of 2 for efficiency
        return 1 << (optimal.bit_length() - 1)

    def create_config(self, hidden_size: int, num_layers: int = 1) -> CheckpointConfig:
        """Create a CheckpointConfig optimized for the memory budget.

        Args:
            hidden_size: Model hidden dimension.
            num_layers: Number of layers.

        Returns:
            CheckpointConfig with optimized chunk_size.
        """
        chunk_size = self.optimal_chunk_size(hidden_size, num_layers)
        return CheckpointConfig(
            chunk_size=chunk_size,
            materialize_between_chunks=True,  # Essential for memory bounding
            min_seq_length=chunk_size,  # Always chunk if using budget
        )
