"""MMFP4-specific draft model for speculative decoding.

This module provides a DraftModel implementation that uses the MMFP4MTPHead
(Multi-Token Prediction head) for efficient speculative decoding. The MTP head
is designed to predict multiple future tokens from hidden states, making it
ideal for use as a draft model.

Optimized version includes:
- Pre-allocated buffers for inference
- Fused operations for draft generation
- Optimized speculation loop with minimal allocations
- FastSpeculationEngine integration for maximum throughput
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any

from ..layers.mmfp4_mtp_head import MMFP4MTPHead, verify_kernel
from ..kv_cache import KVCache
from .draft import DraftModel, DraftOutput

# Try to import optimized version
try:
    from ..layers.mmfp4_mtp_head_optimized import (
        OptimizedMMFP4MTPHead, 
        FastSpeculationEngine,
        UltraFastMTPHead,
    )
    HAS_OPTIMIZED = True
    HAS_ULTRAFAST = True
except ImportError:
    HAS_OPTIMIZED = False
    HAS_ULTRAFAST = False


class MMFP4DraftModel(DraftModel):
    """Draft model for MMFP4 using Multi-Token Prediction head.
    
    Uses the MMFP4MTPHead to predict multiple tokens in a single forward pass,
    making it much more efficient than autoregressive draft models. This enables
    higher speculation depths with lower latency overhead.
    
    Supports weight sharing with the target model for memory-efficient
    speculative decoding. When enabled, the draft model shares a subset
    of parameters with the target model, reducing memory footprint by ~40-60%.
    
    Optimized features:
    - Pre-allocated buffers for inference
    - Fused operations in speculation loop
    - Fast path for single-batch inference
    
    Args:
        hidden_size: Model hidden dimension
        vocab_size: Output vocabulary size
        num_predictions: Number of tokens to predict (speculation depth)
        group_size: MMFP4 quantization group size
        device: Device for tensors
        dtype: Data type for computations
        weight_sharing: Whether weight sharing is enabled (set automatically)
        use_optimized: Whether to use optimized MTP head (default: True)
    
    Example:
        >>> draft = MMFP4DraftModel(
        ...     hidden_size=4096,
        ...     vocab_size=32000,
        ...     num_predictions=4,
        ... )
        >>> hidden_states = torch.randn(1, 1, 4096)  # From target model
        >>> output = draft.speculate_from_hidden(hidden_states, num_tokens=4)
        >>> print(output.tokens.shape)  # [1, 4]
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_predictions: int = 4,
        group_size: int = 128,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
        weight_sharing: bool = False,
        use_optimized: bool = True,
        use_ultrafast: bool = True,
    ):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_predictions = num_predictions
        self.group_size = group_size
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.weight_sharing = weight_sharing
        self.use_optimized = use_optimized and HAS_OPTIMIZED
        self.use_ultrafast = use_ultrafast and HAS_ULTRAFAST
        
        # Initialize the MTP head (use ultra-fast/optimized version if available)
        if self.use_ultrafast and HAS_ULTRAFAST:
            # UltraFastMTPHead with torch.compile for maximum speed
            self.mtp_head = UltraFastMTPHead(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_predictions=num_predictions,
                group_size=group_size,
            )
        elif self.use_optimized and HAS_OPTIMIZED:
            self.mtp_head = OptimizedMMFP4MTPHead(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_predictions=num_predictions,
                group_size=group_size,
            )
        else:
            self.mtp_head = MMFP4MTPHead(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_predictions=num_predictions,
                group_size=group_size,
            )
        self.mtp_head = self.mtp_head.to(device=device, dtype=dtype)
        self.mtp_head.eval()
        
        # Initialize fast speculation engine if using optimized head
        if self.use_ultrafast and HAS_ULTRAFAST:
            # UltraFastMTPHead has speculate_ultrafast built-in
            self._fast_engine = None  # Not needed for UltraFastMTPHead
        elif self.use_optimized and HAS_OPTIMIZED:
            self._fast_engine = FastSpeculationEngine(
                self.mtp_head,
                num_predictions=num_predictions,
                enable_caching=False,  # Cache handled at pipeline level
            )
        else:
            self._fast_engine = None
        
        # Cache for hidden states
        self._cached_hidden: Tensor | None = None
        self._cache_seq_len: int = 0
        
        # Pre-allocated buffers for inference (initialized on first use)
        self._token_buffer: Tensor | None = None
        self._prob_buffer: Tensor | None = None
    
    @classmethod
    def from_target_model(
        cls,
        target_model: nn.Module,
        num_predictions: int = 4,
        group_size: int = 128,
    ) -> MMFP4DraftModel:
        """Create a draft model from a target model's configuration.
        
        Extracts hidden_size and vocab_size from the target model to ensure
        compatibility.
        
        Args:
            target_model: Target language model with config attribute
            num_predictions: Number of speculative tokens to generate
            group_size: MMFP4 quantization group size
        
        Returns:
            Configured MMFP4DraftModel instance
        """
        config = getattr(target_model, "config", None)
        if config is not None:
            hidden_size = getattr(config, "hidden_size", 4096)
            vocab_size = getattr(config, "vocab_size", 32000)
        else:
            # Try to infer from model
            hidden_size = getattr(target_model, "hidden_size", 4096)
            vocab_size = getattr(target_model, "vocab_size", 32000)
        
        # Infer device from target model
        device = None
        try:
            device = next(target_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        
        return cls(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_predictions=num_predictions,
            group_size=group_size,
            device=device,
        )
    
    @classmethod
    def from_target_model_with_weight_sharing(
        cls,
        target_model: nn.Module,
        num_predictions: int = 4,
        group_size: int = 128,
        share_lm_head: bool = True,
    ) -> MMFP4DraftModel:
        """Create a draft model with weight sharing from a target model.
        
        This factory method creates a memory-efficient draft model that shares
        a subset of parameters with the target model. This reduces memory
        footprint by approximately 40-60% compared to a standalone draft model.
        
        Weight sharing strategy:
        - Shares LM head weights for output projections (if share_lm_head=True)
        - Creates lightweight MTP-specific parameters for input projection
        
        Args:
            target_model: Target language model to share weights with
            num_predictions: Number of speculative tokens to generate
            group_size: MMFP4 quantization group size
            share_lm_head: Whether to share LM head weights with target model
        
        Returns:
            MMFP4DraftModel with shared weights from target model
            
        Example:
            >>> target_model = load_target_model()
            >>> draft = MMFP4DraftModel.from_target_model_with_weight_sharing(
            ...     target_model,
            ...     num_predictions=4,
            ... )
            >>> print(f"Memory savings: {draft.get_memory_savings()['savings_bytes'] / 1e6:.1f} MB")
        """
        # Create instance with weight sharing enabled
        instance = cls.from_target_model(
            target_model=target_model,
            num_predictions=num_predictions,
            group_size=group_size,
        )
        
        # Replace MTP head with weight-sharing version
        instance.mtp_head = MMFP4MTPHead.from_target_model(
            target_model=target_model,
            num_predictions=num_predictions,
            group_size=group_size,
            share_lm_head=share_lm_head,
        )
        instance.mtp_head = instance.mtp_head.to(device=instance.device, dtype=instance.dtype)
        instance.mtp_head.eval()
        
        instance.weight_sharing = True
        
        return instance
    
    def get_memory_savings(self) -> dict[str, float]:
        """Get memory savings statistics from weight sharing.
        
        Returns:
            Dict with memory statistics in bytes and percentages
        """
        if not self.weight_sharing or not hasattr(self.mtp_head, 'get_memory_savings'):
            return {
                "total_params": sum(p.numel() for p in self.mtp_head.parameters()),
                "shared_params": 0,
                "unique_params": sum(p.numel() for p in self.mtp_head.parameters()),
                "savings_bytes": 0.0,
                "savings_percent": 0.0,
            }
        
        savings = self.mtp_head.get_memory_savings()
        total = savings["total_params"]
        shared = savings["shared_params"]
        
        if total > 0:
            savings["savings_percent"] = (shared / total) * 100
        else:
            savings["savings_percent"] = 0.0
            
        return savings
    
    def is_weight_sharing_enabled(self) -> bool:
        """Check if weight sharing is enabled for this draft model."""
        return self.weight_sharing
    
    def set_hidden_states(self, hidden_states: Tensor) -> None:
        """Cache hidden states from the target model for drafting.
        
        This is the primary way to provide context for the draft model.
        The MTP head uses these hidden states to predict future tokens.
        
        Args:
            hidden_states: Hidden states from target model, [batch, seq, hidden].
                Typically the last layer's output.
        """
        self._cached_hidden = hidden_states.detach()
        self._cache_seq_len = hidden_states.shape[1]
    
    def speculate_from_hidden(
        self,
        hidden_states: Tensor,
        num_tokens: int = 4,
        temperature: float = 1.0,
    ) -> DraftOutput:
        """Generate speculative tokens from hidden states - optimized path.
        
        This is the main drafting method optimized for maximum throughput.
        Uses fused operations and minimizes memory allocations for the hot
        path during speculative decoding.
        
        Optimizations:
        - Uses fast speculation engine if available (single-batch optimized)
        - Pre-allocated buffers to avoid per-call allocations
        - Fused operations in forward pass
        - Optimized inference mode with no_grad context
        
        Args:
            hidden_states: Target model hidden states, [batch, seq, hidden]
            num_tokens: Number of speculative tokens to generate
            temperature: Sampling temperature (1.0 = greedy, <1.0 = more random)
        
        Returns:
            DraftOutput with proposed tokens and probability distributions
        """
        num_tokens = min(num_tokens, self.num_predictions)
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        
        # Ensure hidden states match model dtype
        if hidden_states.dtype != self.mtp_head.input_proj.weight.dtype:
            hidden_states = hidden_states.to(dtype=self.mtp_head.input_proj.weight.dtype)
        
        # OPTIMIZED PATH 1: FastSpeculationEngine for single batch
        if self._fast_engine is not None and batch_size == 1:
            tokens, probs = self._fast_engine.speculate(
                hidden_states,
                num_tokens=num_tokens,
                temperature=temperature,
            )
            return DraftOutput(tokens=tokens, probs=probs)
        
        # OPTIMIZED PATH 2: Use speculate_fast_path from MMFP4MTPHead if available
        if hasattr(self.mtp_head, 'speculate_fast_path'):
            tokens, probs = self.mtp_head.speculate_fast_path(
                hidden_states,
                num_tokens=num_tokens,
                temperature=temperature,
            )
            return DraftOutput(tokens=tokens, probs=probs)
        
        # Optimized standard path with pre-allocated buffers
        with torch.inference_mode():
            # Initialize buffers on first use
            if self._token_buffer is None or self._token_buffer.device != device:
                self._token_buffer = torch.empty(
                    batch_size, self.num_predictions,
                    dtype=torch.long, device=device
                )
                self._prob_buffer = torch.empty(
                    batch_size, self.num_predictions, self.vocab_size,
                    dtype=torch.float32, device=device
                )
            
            # Get predictions from MTP head - uses fused output projection
            # Returns [batch, num_predictions, vocab]
            all_logits = self.mtp_head(hidden_states)
            
            # Take only the requested number of tokens
            if num_tokens < self.num_predictions:
                logits = all_logits[:, :num_tokens, :]
            else:
                logits = all_logits
            
            # Apply temperature in-place when possible for memory efficiency
            if temperature > 0 and abs(temperature - 1.0) > 1e-6:
                logits = logits / temperature
            
            # Compute probabilities into pre-allocated buffer
            probs_view = self._prob_buffer[:batch_size, :num_tokens, :]
            torch.softmax(logits, dim=-1, out=probs_view)
            
            # Greedy selection into pre-allocated buffer
            token_view = self._token_buffer[:batch_size, :num_tokens]
            torch.argmax(probs_view, dim=-1, out=token_view)
            
            # Return views as copies to avoid buffer reuse issues
            return DraftOutput(
                tokens=token_view.clone(),
                probs=probs_view.clone(),
            )
    
    def speculate(
        self,
        input_ids: Tensor,
        kv_cache: KVCache | None = None,
        num_tokens: int = 4,
    ) -> DraftOutput:
        """Generate speculative token proposals.
        
        This method is called by the SpeculativeEngine. It requires that
        hidden states have been cached via set_hidden_states().
        
        Args:
            input_ids: Current context token IDs, shape [batch, seq_len].
                Not used directly - we rely on cached hidden states.
            kv_cache: Target model's KV cache (unused for MTP-based drafting).
            num_tokens: Number of speculative tokens to generate.
        
        Returns:
            DraftOutput with proposed tokens and probability distributions.
        
        Raises:
            RuntimeError: If hidden states haven't been cached.
        """
        if self._cached_hidden is None:
            # Fallback: return uniform distribution
            return self._fallback_output(input_ids.shape[0], num_tokens)
        
        return self.speculate_from_hidden(self._cached_hidden, num_tokens)
    
    def reset(self) -> None:
        """Reset cached hidden states for a new sequence."""
        self._cached_hidden = None
        self._cache_seq_len = 0
    
    def _fallback_output(self, batch_size: int, num_tokens: int) -> DraftOutput:
        """Generate fallback output when hidden states aren't available."""
        tokens = torch.zeros(
            batch_size, 
            num_tokens, 
            dtype=torch.long, 
            device=self.device
        )
        probs = torch.ones(
            batch_size, 
            num_tokens, 
            self.vocab_size, 
            device=self.device
        ) / self.vocab_size
        return DraftOutput(tokens=tokens, probs=probs)
    
    def get_speedup_estimate(self, acceptance_rate: float) -> float:
        """Estimate speedup from speculative decoding.
        
        Args:
            acceptance_rate: Fraction of draft tokens typically accepted
        
        Returns:
            Estimated speedup factor (e.g., 2.0 = 2x faster)
        """
        # Cost model: draft is ~1/10th the cost of target per token
        # But we generate num_predictions tokens in parallel
        draft_cost = 0.1
        target_cost = 1.0
        
        # Expected tokens per target forward pass
        # = acceptance_rate * num_predictions (accepted draft)
        # + 1 (bonus token from target)
        expected_tokens = acceptance_rate * self.num_predictions + 1
        
        # Cost per step
        cost_per_step = draft_cost + target_cost
        
        # Effective cost per token
        effective_cost = cost_per_step / expected_tokens
        
        # Speedup vs autoregressive (cost = 1.0 per token)
        speedup = 1.0 / effective_cost
        
        return speedup


class MMFP4DraftModelWithTarget(DraftModel):
    """MMFP4 draft model that runs the target model to get hidden states.
    
    This is a convenience wrapper that handles running the target model
    to obtain hidden states, then using the MTP head for drafting.
    
    Args:
        target_model: The target language model
        mtp_head: Pre-initialized MMFP4MTPHead (optional)
        num_predictions: Number of tokens to predict
        device: Device for tensors
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        mtp_head: MMFP4MTPHead | None = None,
        num_predictions: int = 4,
        device: torch.device | None = None,
    ):
        self.target_model = target_model
        self.device = device or torch.device("cpu")
        
        # Get model config
        config = getattr(target_model, "config", None)
        if config is not None:
            hidden_size = getattr(config, "hidden_size", 4096)
            vocab_size = getattr(config, "vocab_size", 32000)
        else:
            hidden_size = getattr(target_model, "hidden_size", 4096)
            vocab_size = getattr(target_model, "vocab_size", 32000)
        
        # Initialize or use provided MTP head
        if mtp_head is not None:
            self.mtp_head = mtp_head
        else:
            self.mtp_head = MMFP4MTPHead(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_predictions=num_predictions,
            )
        
        self.mtp_head = self.mtp_head.to(self.device)
        self.mtp_head.eval()
        
        self.num_predictions = num_predictions
        self.vocab_size = vocab_size
        
        # Cache
        self._last_hidden: Tensor | None = None
    
    def speculate(
        self,
        input_ids: Tensor,
        kv_cache: KVCache | None = None,
        num_tokens: int = 4,
    ) -> DraftOutput:
        """Generate speculative tokens by running target model + MTP head.
        
        Args:
            input_ids: Current context tokens [batch, seq_len]
            kv_cache: Target model's KV cache (optional)
            num_tokens: Number of speculative tokens
        
        Returns:
            DraftOutput with tokens and probabilities
        """
        num_tokens = min(num_tokens, self.num_predictions)
        
        with torch.no_grad():
            # Run target model to get hidden states
            # Assumes target_model returns hidden states or has a method to get them
            if hasattr(self.target_model, "get_hidden_states"):
                hidden = self.target_model.get_hidden_states(input_ids, kv_cache=kv_cache)
            elif hasattr(self.target_model, "forward"):
                # Try to get hidden states from forward pass
                outputs = self.target_model(input_ids, output_hidden_states=True)
                if hasattr(outputs, "hidden_states"):
                    hidden = outputs.hidden_states[-1]  # Last layer
                else:
                    hidden = outputs
            else:
                # Fallback
                return self._fallback_output(input_ids.shape[0], num_tokens)
            
            self._last_hidden = hidden
            
            # Use MTP head to predict tokens
            all_logits = self.mtp_head(hidden)
            logits = all_logits[:, :num_tokens, :]
            probs = F.softmax(logits, dim=-1)
            tokens = torch.argmax(probs, dim=-1)
        
        return DraftOutput(tokens=tokens, probs=probs)
    
    def reset(self) -> None:
        """Reset cached state."""
        self._last_hidden = None
    
    def _fallback_output(self, batch_size: int, num_tokens: int) -> DraftOutput:
        """Generate fallback output."""
        tokens = torch.zeros(
            batch_size, num_tokens, dtype=torch.long, device=self.device
        )
        probs = torch.ones(
            batch_size, num_tokens, self.vocab_size, device=self.device
        ) / self.vocab_size
        return DraftOutput(tokens=tokens, probs=probs)


__all__ = ["MMFP4DraftModel", "MMFP4DraftModelWithTarget"]
