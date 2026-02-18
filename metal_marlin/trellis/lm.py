"""TrellisForCausalLM: Language model wrapper for trellis-quantized models.

This module provides a high-level interface for trellis-quantized models
with language modeling head for text generation tasks.

Usage:
    from metal_marlin.trellis.lm import TrellisForCausalLM

    model = TrellisForCausalLM.from_pretrained("model_path")
    logits = model(input_ids)

    # Generate text
    generated = model.generate(input_ids, max_new_tokens=100, temperature=0.8)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kv_cache import TrellisKVCache
from ..metal_dispatch import MetalKernelLibrary
from .config import TrellisModelConfig

# Import from model for backward compatibility and usage
from .model import TrellisDecoderLayer, TrellisModel, TrellisMoEMLP, WorkspaceBufferPool
from .moe_dispatch import MoEBufferPool

# Import LayerBatchContext for batching command buffers across layers
try:
    from .async_dispatch import HAS_METAL, LayerBatchContext
except ImportError:
    LayerBatchContext = None
    HAS_METAL = False

if TYPE_CHECKING:
    # Avoid circular import at runtime if possible, though these are re-exported
    pass

logger = logging.getLogger(__name__)


@dataclass
class CausalLMOutput:
    """Output from TrellisForCausalLM compatible with HuggingFace interface."""

    logits: torch.Tensor
    """Logits tensor [batch, seq_len, vocab_size]."""


class TrellisForCausalLM(nn.Module):
    """Trellis model with language modeling head for text generation.

    Wraps TrellisModel with an LM head projection for generating logits.
    Supports autoregressive generation with temperature, top-k, and top-p sampling.

    Attributes:
        model: The underlying TrellisModel.
        config: Model configuration.
        lm_head: Linear projection from hidden_size to vocab_size.
        rope_cos_cache: Precomputed RoPE cos table [max_seq_len, rope_dim//2].
        rope_sin_cache: Precomputed RoPE sin table [max_seq_len, rope_dim//2].
    """

    def __init__(self, config: TrellisModelConfig):
        """Initialize TrellisForCausalLM.

        Precomputes RoPE sin/cos tables for fast lookup during forward pass,
        avoiding redundant computation on every layer and every token.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.model = TrellisModel(config)

        # LM head (not quantized, tied to embedding or separate)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        # Shared buffer pool for all MoE layers - reduces mps_tensor_to_metal_buffer calls
        # from O(layers * batch_sizes) to O(batch_sizes) by reusing buffers across layers
        self._shared_buffer_pool: MoEBufferPool | None = None

        # Shared workspace buffer pool for MoE forward - eliminates per-forward allocations
        self._workspace_buffer_pool: WorkspaceBufferPool | None = None

        # Shared Metal library for all MoE layers - CRITICAL for batch_dispatch to work!
        # Without this, each layer has its own lib and only one lib's batch_mode is set.
        self._shared_lib: MetalKernelLibrary | None = None

        # Precompute RoPE sin/cos tables for fast lookup during forward
        # This saves computation on every layer, every forward pass
        self._build_rope_cache()

    def _build_rope_cache(self) -> None:
        """Precompute RoPE sin/cos tables for fast lookup during forward pass.

        Computes sin/cos for all positions up to max_seq_len and stores as a tuple.
        During forward, layers receive position indices and lookup the cached values.

        Cache dimensions:
        - self._rope_cache: Tuple of (sin, cos) tensors [max_seq_len, rope_dim//2]

        where rope_dim = qk_rope_head_dim (typically 64 for GLM-4 MLA)
        """
        # Determine RoPE dimension (use MLA rope dim if available)
        rope_dim = getattr(self.config, "qk_rope_head_dim", 64)
        max_seq_len = self.config.max_position_embeddings
        rope_theta = self.config.rope_theta

        # Compute inverse frequencies for full rope_dim
        # GLM's RoPE implementation needs one frequency per dimension
        inv_freq = 1.0 / \
            (rope_theta ** (torch.arange(0, rope_dim, dtype=torch.float32) / rope_dim))

        # Position indices [0, max_seq_len)
        positions = torch.arange(max_seq_len, dtype=torch.float32)

        # Compute angles: freqs[pos, freq] = pos * inv_freq[freq]
        freqs = torch.outer(positions, inv_freq)  # [max_seq_len, rope_dim]

        # Precompute sin/cos with proper shape for attention
        # Add batch and head dimensions: [1, 1, max_seq_len, rope_dim]
        sin_cache = torch.sin(freqs).unsqueeze(0).unsqueeze(0)
        cos_cache = torch.cos(freqs).unsqueeze(0).unsqueeze(0)

        self.register_buffer("rope_sin_cache", sin_cache)
        self.register_buffer("rope_cos_cache", cos_cache)

        logger.debug(
            "Precomputed RoPE cache: seq_len=%d, rope_dim=%d, cache_shape=%s",
            max_seq_len,
            rope_dim,
            list(self.rope_cos_cache.shape),
        )

    def get_rope_cache(
        self, position_ids: torch.Tensor | None = None, seq_len: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get precomputed RoPE sin/cos values for given positions.

        Args:
            position_ids: Position indices [batch, seq_len] or [seq_len].
                         If None, uses seq_len to generate range [0, seq_len).
            seq_len: Sequence length (used if position_ids is None).

        Returns:
            Tuple of (cos, sin) tensors shaped for broadcasting:
            - cos: [1, 1, seq_len, rope_dim//2] or [seq_len, rope_dim//2]
            - sin: [1, 1, seq_len, rope_dim//2] or [seq_len, rope_dim//2]

        Usage:
            cos, sin = model.get_rope_cache(position_ids)
            q_rotated = apply_rotary_pos_emb(q, cos, sin)
        """
        if position_ids is not None:
            # Gather sin/cos for specific positions
            # position_ids can be [batch, seq] or [seq]
            if position_ids.dim() == 2:
                # [batch, seq_len] -> use first batch for lookup (typical case)
                positions = position_ids[0]
            else:
                positions = position_ids

            cos = self.rope_cos_cache[positions]  # [seq_len, rope_dim//2]
            sin = self.rope_sin_cache[positions]
        else:
            # Use first seq_len positions
            if seq_len is None:
                raise ValueError(
                    "Either position_ids or seq_len must be provided")
            cos = self.rope_cos_cache[:seq_len]
            sin = self.rope_sin_cache[:seq_len]

        # Add broadcast dimensions for attention: [seq, dim] -> [1, 1, seq, dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        return cos, sin

    def _get_shared_buffer_pool(self) -> MoEBufferPool | None:
        """Get or create a shared buffer pool for all MoE layers.

        Creates a single MoEBufferPool that can be reused across all MoE layers,
        reducing mps_tensor_to_metal_buffer calls from O(layers * batch_sizes) to
        O(batch_sizes).

        Returns:
            Shared MoEBufferPool, or None if no MoE layers exist.
        """
        if self._shared_buffer_pool is not None:
            return self._shared_buffer_pool

        # Find first MoE layer to get Metal device and hidden_dim
        for layer in self.model.layers:
            if isinstance(layer.mlp, TrellisMoEMLP) and hasattr(layer.mlp, "_get_lib"):
                lib = layer.mlp._get_lib()
                self._shared_buffer_pool = MoEBufferPool(
                    device=lib.device,
                    hidden_dim=self.config.hidden_size,
                    max_batch=32,
                )
                # Pre-allocate for common top_k values
                if hasattr(layer.mlp, "num_experts_per_tok"):
                    self._shared_buffer_pool.preallocate_top_k(
                        layer.mlp.num_experts_per_tok)
                break

        return self._shared_buffer_pool

    def _setup_shared_buffer_pool(self) -> None:
        """Initialize shared buffer pool and assign to all MoE layers.

        Call this after model loading to enable buffer sharing across layers.
        This reduces mps_tensor_to_metal_buffer overhead by ~45x for models
        with 45 MoE layers.
        """
        shared_pool = self._get_shared_buffer_pool()
        if shared_pool is None:
            return

        # Assign shared pool to all MoE layers, replacing their per-layer pools
        for layer in self.model.layers:
            if isinstance(layer.mlp, TrellisMoEMLP):
                # Replace per-layer pool with shared pool
                layer.mlp._buffer_pool = shared_pool

    def _get_workspace_buffer_pool(self) -> WorkspaceBufferPool | None:
        """Get or create a shared workspace buffer pool for all MoE layers.

        Creates a single WorkspaceBufferPool that can be reused across all MoE layers,
        eliminating per-forward allocations for output, accumulator, and intermediate
        buffers.

        Returns:
            Shared WorkspaceBufferPool, or None if no MoE layers exist.
        """
        if self._workspace_buffer_pool is not None:
            return self._workspace_buffer_pool

        # Find first MoE layer to get dimensions
        for layer in self.model.layers:
            if isinstance(layer.mlp, TrellisMoEMLP):
                device = str(next(layer.mlp.router.parameters()).device)
                self._workspace_buffer_pool = WorkspaceBufferPool(
                    hidden_dim=self.config.hidden_size,
                    intermediate_dim=getattr(
                        layer.mlp, "intermediate_dim", self.config.hidden_size * 4),
                    device=device,
                )
                break

        return self._workspace_buffer_pool

    def _setup_workspace_buffer_pool(self) -> None:
        """Initialize workspace buffer pool and assign to all MoE layers.

        Call this after model loading to enable workspace buffer sharing across layers.
        This eliminates per-forward allocations for output, accumulator, and intermediate
        buffers.
        """
        workspace_pool = self._get_workspace_buffer_pool()
        if workspace_pool is None:
            return

        # Assign shared workspace pool to all MoE layers
        for layer in self.model.layers:
            if isinstance(layer.mlp, TrellisMoEMLP):
                layer.mlp._workspace_buffer_pool = workspace_pool

    def _setup_shared_lib(self) -> MetalKernelLibrary | None:
        """Initialize a shared Metal library for ALL layers that use Metal dispatch.

        CRITICAL for batch_dispatch to work! Without a shared lib, each layer
        has its own MetalKernelLibrary instance, and only ONE lib's _batch_mode
        is set True during batch_dispatch(). This means most layers still
        create separate command buffers.

        With a shared lib, ALL layers use the same lib, so when batch_mode is
        set, ALL dispatches go into the same command buffer.

        Sets shared lib on:
        - TrellisMoEMLP layers (MoE fused kernels)
        - TrellisLinear modules (attention projections, shared_expert, dense MLP)

        Returns:
            The shared MetalKernelLibrary, or None if no modules need it.
        """
        if self._shared_lib is not None:
            return self._shared_lib

        # Create ONE lib and share across ALL modules that use Metal
        self._shared_lib = MetalKernelLibrary.from_source_dir()

        # Import here to avoid circular dependency
        from .linear import TrellisLinear

        # Assign to ALL modules that use Metal dispatch
        count = 0
        for module in self.modules():
            if isinstance(module, TrellisMoEMLP):
                module._lib = self._shared_lib
                count += 1
            elif isinstance(module, TrellisLinear):
                module._lib = self._shared_lib
                count += 1

        return self._shared_lib

    def clear_routing_caches(self) -> None:
        """Clear routing caches in all MoE layers.

        Call this at the start of a new generation to ensure fresh routing
        decisions. The cache is per-layer and accumulates during decode,
        so clearing between generations prevents stale routing decisions.
        """
        for layer in self.model.layers:
            if isinstance(layer.mlp, TrellisMoEMLP):
                layer.mlp.clear_routing_cache()

    def get_routing_cache_stats(self) -> dict[str, int | float]:
        """Get aggregate routing cache statistics across all MoE layers.

        Returns:
            Dict with total hits, misses, hit_rate, and per-layer stats.
        """
        total_hits = 0
        total_misses = 0
        per_layer_stats = []

        for i, layer in enumerate(self.model.layers):
            if isinstance(layer.mlp, TrellisMoEMLP):
                stats = layer.mlp.get_routing_cache_stats()
                total_hits += stats["hits"]
                total_misses += stats["misses"]
                per_layer_stats.append({"layer": i, **stats})

        total = total_hits + total_misses
        return {
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": total_hits / total if total > 0 else 0.0,
            "per_layer": per_layer_stats,
        }

    def reset_routing_cache_stats(self) -> None:
        """Reset routing cache statistics in all MoE layers."""
        for layer in self.model.layers:
            if isinstance(layer.mlp, TrellisMoEMLP):
                layer.mlp.reset_routing_cache_stats()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: TrellisKVCache | None = None,
        prefetch_kv: bool = True,
    ) -> CausalLMOutput:
        """Forward pass returning logits.

        Uses batch dispatch to encode all MoE layer kernel dispatches into a
        single Metal command buffer, reducing per-layer command buffer overhead.

        For decode (single token with KV cache), enables KV cache prefetching:
        while computing layer N, prefetches layer N+1's cache to hide memory latency.

        RoPE sin/cos tables are precomputed on model load and sliced for the
        current positions, avoiding redundant computation on every layer.

        Args:
            input_ids: Input token IDs [batch, seq_len].
            attention_mask: Optional attention mask [batch, seq_len].
            position_ids: Optional position IDs [batch, seq_len].
            kv_cache: Optional KV cache for generation.
            prefetch_kv: Whether to prefetch next layer's KV cache during decode.
                        Only effective when kv_cache is provided and seq_len=1.

        Returns:
            CausalLMOutput with logits tensor [batch, seq_len, vocab_size].
        """
        hidden_states = self.model.embed_tokens(input_ids)
        # Keep model hidden states on a single fast inference dtype to avoid
        # bf16/fp16 residual promotion to fp32 across layers.
        if hidden_states.dtype != torch.float16:
            hidden_states = hidden_states.to(dtype=torch.float16)
        batch_size, seq_len = input_ids.shape
        device = hidden_states.device

        if self._workspace_buffer_pool is None:
            self._setup_workspace_buffer_pool()

        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = self.model._make_causal_mask(seq_len, device)

        # Get rope cache (sin, cos) for position-based lookup
        # Use the registered buffers created by _build_rope_cache()
        sin_cache = self.rope_sin_cache
        cos_cache = self.rope_cos_cache

        # Move caches to same device as hidden states
        sin_cache = sin_cache.to(device=device, dtype=hidden_states.dtype)
        cos_cache = cos_cache.to(device=device, dtype=hidden_states.dtype)

        # Pass position_ids and full caches - attention will lookup as needed
        # This avoids slicing sin/cos at every layer
        if position_ids is None:
            if kv_cache is not None and seq_len == 1 and kv_cache.seq_len > 0:
                # Decode step: position should continue from cached length.
                position_ids = torch.full(
                    (batch_size, 1),
                    kv_cache.seq_len,
                    device=device,
                    dtype=torch.long,
                )
            else:
                position_ids = torch.arange(
                    seq_len, device=device, dtype=torch.long
                ).unsqueeze(0).expand(batch_size, -1)

        # Get SHARED library for batch context - all layers must use same lib!
        # Without this, only one layer's lib has batch_mode=True.
        lib = self._setup_shared_lib()

        # Determine if we should use KV prefetching.
        # Prefetch is beneficial for decode (seq_len=1) where memory bandwidth dominates.
        # For prefill (seq_len>1), compute dominates and prefetching adds overhead.
        use_prefetch = (
            prefetch_kv
            and kv_cache is not None
            and seq_len == 1
            and kv_cache.seq_len > 0  # Cache has content to prefetch
        )

        num_layers = len(self.model.layers)

        # Use LayerBatchContext for MoE dispatch batching (reduces commits from ~46 to ~6)
        # This is separate from lib.batch_dispatch() which handles attention kernel batching.
        from contextlib import nullcontext

        from .model import LAYERS_PER_BATCH
        use_layer_batch = (
            batch_size == 1 and HAS_METAL and LayerBatchContext is not None)
        layer_batch_mgr = (
            LayerBatchContext(
                self.model, batch_size=LAYERS_PER_BATCH) if use_layer_batch else nullcontext()
        )

        # Batch all layer dispatches into single command buffer
        if lib is not None:
            with lib.batch_dispatch(), layer_batch_mgr as batch_ctx:
                for i, layer in enumerate(self.model.layers):
                    # Prefetch next layer's KV cache while computing current layer.
                    # This warms GPU caches for the next iteration's memory reads.
                    if use_prefetch and i + 1 < num_layers:
                        prefetch_async = getattr(
                            kv_cache, "prefetch_layer_async", None)
                        if callable(prefetch_async):
                            prefetch_async(i + 1, lib=lib)
                        else:
                            kv_cache.prefetch_layer(i + 1)

                    hidden_states = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        kv_cache=kv_cache,
                        rope_cos=cos_cache,
                        rope_sin=sin_cache,
                        _batch_ctx=batch_ctx,
                    )
                    # NOTE: No sync inside batch_dispatch! It would conflict with batched
                    # command buffer. The batch_dispatch context manager handles the sync.
        else:
            with layer_batch_mgr as batch_ctx:
                for i, layer in enumerate(self.model.layers):
                    # Prefetch next layer's KV cache
                    if use_prefetch and i + 1 < num_layers:
                        kv_cache.prefetch_layer(i + 1)

                    hidden_states = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        kv_cache=kv_cache,
                        rope_cos=cos_cache,
                        rope_sin=sin_cache,
                        _batch_ctx=batch_ctx,
                    )

        # Final normalization - keep in fp16 for memory efficiency
        hidden_states = self.model.norm(hidden_states)

        # Cast to fp32 only for final output to halve memory usage
        # Intermediate activations remain in fp16 throughout the model
        logits = self.lm_head(hidden_states.to(torch.float32))
        return CausalLMOutput(logits=logits)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Autoregressive generation with KV cache.

        Generates tokens autoregressively using the model with efficient
        KV caching for improved performance on long sequences.

        Args:
            input_ids: Initial token IDs [batch, seq_len].
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (0.0 = greedy/argmax, <1.0 = focused, >1.0 = random).
            top_k: Number of highest probability tokens to keep for top-k sampling.
            top_p: Cumulative probability threshold for nucleus (top-p) sampling.

        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens].
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initialize MLA KV cache for efficient generation
        # MLA caches compressed representation (kv_lora_rank + qk_rope_head_dim)
        # instead of full K,V tensors, reducing cache size by ~8x
        # Note: TrellisKVCache now defaults to use_paged=True for vLLM-style blocks
        kv_cache = TrellisKVCache(
            num_layers=self.config.num_hidden_layers,
            max_seq_len=seq_len + max_new_tokens,
            kv_lora_rank=self.config.kv_lora_rank,
            qk_rope_head_dim=self.config.qk_rope_head_dim,
            device=str(device),
        )

        # Track which sequences are finished (for batched generation)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Initial forward pass to fill KV cache with prompt
        _ = self.forward(input_ids, kv_cache=kv_cache)

        # Get current sequence length from cache
        current_len = kv_cache.seq_len

        # Generate tokens one at a time
        for _ in range(max_new_tokens):
            # Get logits for the last position only
            output = self.forward(
                input_ids[:, -1:],
                kv_cache=kv_cache,
            )
            next_token_logits = output.logits[:, -1, :]  # [batch, vocab_size]

            # Apply temperature
            if temperature != 1.0 and temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k, dim=-1)[0][..., -1, None]
                )
                next_token_logits = next_token_logits.masked_fill(
                    indices_to_remove, float("-inf"))

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[...,
                                         1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits = next_token_logits.masked_fill(
                    indices_to_remove, float("-inf"))

            # Sample from the filtered distribution
            if temperature == 0.0:
                # Greedy decoding: pick argmax directly (no softmax needed)
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_token_logits, dim=-1)
                # Clamp to avoid nan/inf issues in multinomial
                probs = probs.clamp(min=0.0)
                prob_sum = probs.sum(dim=-1, keepdim=True)
                prob_sum = prob_sum.clamp(min=1e-8)
                probs = probs / prob_sum
                next_token = torch.multinomial(
                    probs, num_samples=1)  # [batch, 1]

            # Mark sequences as finished if EOS token is generated
            if hasattr(self.config, "eos_token_id") and self.config.eos_token_id is not None:
                finished = finished | (
                    next_token.squeeze(-1) == self.config.eos_token_id)
            else:
                # Default EOS token ID (commonly 2 for many models)
                finished = finished | (next_token.squeeze(-1) == 2)

            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Stop if all sequences are finished
            if finished.all():
                break

        return input_ids

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        device: str = "mps",
        mmap: bool = True,
        optimize_memory: bool = True,
    ) -> TrellisForCausalLM:
        """Load a TrellisForCausalLM model from path.

        Loads the model configuration, base model weights, and LM head weights
        from the specified path. Supports both tied and separate LM heads.

        Args:
            model_path: Path to the model directory containing config.json
                and base_weights.safetensors.
            device: Device to load the model on (default: "mps").
            mmap: Enable memory-mapped loading when reading PyTorch
                checkpoints with torch.load (default: True).
            optimize_memory: Enable post-load memory optimizations.

        Returns:
            Loaded TrellisForCausalLM instance.
        """
        model_path = Path(model_path)
        config = TrellisModelConfig.from_pretrained(str(model_path))
        model = cls(config)

        # CRITICAL: Move model to device FIRST, then load weights
        # This ensures weights are loaded directly to GPU, not CPU then copied
        model = model.to(device)

        # Load base model
        model.model = TrellisModel.from_pretrained(str(model_path), device)

        # Load lm_head weight. Prefer memory-mapped torch checkpoints when present.
        # This avoids eagerly copying large checkpoint files into RAM.
        checkpoint_path: Path | None = None
        for candidate in ("model.pt", "pytorch_model.bin", "checkpoint.pt"):
            candidate_path = model_path / candidate
            if candidate_path.exists():
                checkpoint_path = candidate_path
                break

        lm_head_weight: torch.Tensor | None = None
        if checkpoint_path is not None:
            try:
                state_dict = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    mmap=mmap,
                )
            except TypeError:
                # Backward compatibility for older torch versions without mmap arg.
                state_dict = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                )
            if (
                isinstance(state_dict, dict)
                and "state_dict" in state_dict
                and isinstance(state_dict["state_dict"], dict)
            ):
                state_dict = state_dict["state_dict"]

            if isinstance(state_dict, dict) and "lm_head.weight" in state_dict:
                maybe_weight = state_dict["lm_head.weight"]
                if isinstance(maybe_weight, torch.Tensor):
                    lm_head_weight = maybe_weight

        if lm_head_weight is None:
            # Fall back to trellis base_weights.safetensors.
            base_weights = TrellisModel._load_base_weights(str(model_path))
            if "lm_head.weight" in base_weights:
                lm_head_weight = base_weights["lm_head.weight"]

        if lm_head_weight is not None:
            model.lm_head.weight.data = lm_head_weight.to(device)
        else:
            # Tied embeddings - share weight with embed_tokens
            model.lm_head.weight = model.model.embed_tokens.weight

        # Optimize memory if requested
        if optimize_memory:
            model.optimize_memory(verbose=False)

        # Setup shared buffer pool for all MoE layers to reduce buffer creation overhead
        model._setup_shared_buffer_pool()

        # Setup shared workspace buffer pool for all MoE layers to eliminate per-forward allocations
        model._setup_workspace_buffer_pool()

        # Setup shared Metal library for ALL layers to enable batch dispatch
        # CRITICAL: Without this, each layer creates its own lib and batch_mode doesn't work
        model._setup_shared_lib()

        # Model already moved to device earlier, no need to move again
        return model

    def optimize_memory(self, verbose: bool = False) -> dict:
        """Optimize memory by creating Metal buffers and freeing tensors.

        Call after loading the model to minimize memory footprint.

        Args:
            verbose: If True, print memory stats during optimization.

        Returns:
            Dict with memory stats before/after optimization.
        """
        import gc

        stats = {"layers_optimized": 0}

        if verbose:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            stats["before_rss_gb"] = process.memory_info().rss / 1e9

        # Optimize each MoE layer
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer.mlp, "_cached_weight_buffers"):
                # Force eager buffer creation if not already done
                if layer.mlp._cached_weight_buffers is None:
                    layer.mlp._get_cached_buffers()
                    stats["layers_optimized"] += 1

                    if verbose:
                        print(f"  Layer {i}: created Metal buffers")

        # Force garbage collection
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()  # Second pass to catch released MPS memory

        if verbose:
            stats["after_rss_gb"] = process.memory_info().rss / 1e9
            stats["freed_gb"] = stats["before_rss_gb"] - stats["after_rss_gb"]
            print(f"  Memory freed: {stats['freed_gb']:.2f} GB")

        return stats

    def quantize_routers_to_int8(self, verbose: bool = False) -> dict[str, Any]:
        """Quantize all MoE router weights to INT8.

        The router is small (hidden_dim -> num_experts, e.g., 2048 -> 64)
        and can benefit from int8 quantization:
        - 4x smaller memory footprint
        - Faster matmul (reduced memory bandwidth)
        - Negligible accuracy loss for routing decisions

        For GLM-4.7 with 45 MoE layers:
        - FP16 routers: 45 * 2048 * 64 * 2 = 11.5 MB
        - INT8 routers: 45 * 2048 * 64 * 1 + 45 * 64 * 4 = 5.9 MB
        - Savings: ~50%

        Args:
            verbose: If True, print per-layer quantization stats.

        Returns:
            Dict with:
            - num_layers_quantized: Number of MoE layers quantized
            - total_memory_saved_bytes: Total memory saved
            - avg_snr_db: Average signal-to-noise ratio across layers
            - per_layer: Per-layer quantization stats (if verbose)
        """
        stats: dict[str, Any] = {
            "num_layers_quantized": 0,
            "total_memory_saved_bytes": 0,
            "snr_db_values": [],
        }

        if verbose:
            stats["per_layer"] = []

        for i, layer in enumerate(self.model.layers):
            if isinstance(layer.mlp, TrellisMoEMLP):
                # Get memory before
                before = layer.mlp.get_router_memory_usage()

                # Quantize
                quant_stats = layer.mlp.quantize_router_to_int8()

                # Get memory after
                after = layer.mlp.get_router_memory_usage()

                # Update stats
                if "already_quantized" not in quant_stats:
                    stats["num_layers_quantized"] += 1
                    memory_saved = int(
                        before["total_bytes"]) - int(after["total_bytes"])
                    stats["total_memory_saved_bytes"] += memory_saved
                    stats["snr_db_values"].append(quant_stats.get("snr_db", 0))

                    if verbose:
                        print(
                            f"  Layer {i}: SNR={quant_stats.get('snr_db', 0):.1f}dB, "
                            f"saved {memory_saved / 1024:.1f}KB"
                        )
                        stats["per_layer"].append(
                            {
                                "layer": i,
                                "snr_db": quant_stats.get("snr_db", 0),
                                "memory_saved_bytes": memory_saved,
                            }
                        )

        # Compute average SNR
        if stats["snr_db_values"]:
            stats["avg_snr_db"] = sum(
                stats["snr_db_values"]) / len(stats["snr_db_values"])
        else:
            stats["avg_snr_db"] = 0.0

        del stats["snr_db_values"]  # Remove intermediate list

        if verbose:
            print(
                f"Quantized {stats['num_layers_quantized']} routers, "
                f"saved {stats['total_memory_saved_bytes'] / 1024:.1f}KB total, "
                f"avg SNR={stats['avg_snr_db']:.1f}dB"
            )

        return stats


# Public compatibility aliases.
#
# `TrellisForCausalLM` and related core model classes are now canonicalized in
# `trellis.model`. Keep these names available from `trellis.lm` so existing
# import sites continue working without caller-side rewrites.

# CausalLMOutput is defined in this module
# TrellisForCausalLM is defined in this module
# TrellisModel, TrellisDecoderLayer, TrellisMoEMLP are imported from .model

__all__ = [
    "CausalLMOutput",
    "TrellisForCausalLM",
    "TrellisModel",
    "TrellisDecoderLayer",
    "TrellisMoEMLP",
    "WorkspaceBufferPool",
]
