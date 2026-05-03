"""MTP-based draft model adapter for speculative decoding."""

from __future__ import annotations
import logging

import torch
from torch import Tensor

from ..kv_cache import KVCache
from ..layers.mmfp4_mtp_head import MMFP4MTPHead
from .draft import DraftModel, DraftOutput



logger = logging.getLogger(__name__)

class MTPDraft(DraftModel):
    """Draft model using Multi-Token Prediction head.
    
    This adapter wraps an MTP head to provide the DraftModel interface
    for the speculative decoding engine. It's ~10x cheaper than a
    separate draft model because it only runs the MTP head, not a
    full transformer.
    
    Requirements:
        - Target model must have hidden states accessible
        - MTP head must be trained on target model's distribution
    """
    
    def __init__(
        self,
        mtp_head: MMFP4MTPHead,
        device: torch.device | None = None,
    ):
        logger.debug("initializing %s with mtp_head=%s, device=%s", type(self).__name__, mtp_head, device)
        self.head = mtp_head
        self.device = device or torch.device("mps")
        self._cached_hidden: Tensor | None = None
    
    def set_hidden_states(self, hidden_states: Tensor) -> None:
        """Cache hidden states from the target model."""
        logger.debug("set_hidden_states called with hidden_states=%s", hidden_states)
        self._cached_hidden = hidden_states
    
    def speculate(
        self,
        input_ids: Tensor,
        kv_cache: KVCache | None = None,
        num_tokens: int = 4,
    ) -> DraftOutput:
        """Generate speculative tokens using MTP head."""
        logger.debug("speculate called with input_ids=%s, kv_cache=%s, num_tokens=%s", input_ids, kv_cache, num_tokens)
        if self._cached_hidden is None:
            # No hidden states - return uniform fallback
            batch_size = input_ids.shape[0]
            return self._fallback(batch_size, num_tokens)
        
        # Limit to MTP head's capacity
        num_tokens = min(num_tokens, self.head.num_predictions)
        
        # Get predictions from MTP head
        tokens, probs = self.head.speculate(
            self._cached_hidden,
            temperature=1.0,
        )
        
        # Truncate to requested length
        tokens = tokens[:, :num_tokens]
        probs = probs[:, :num_tokens, :]
        
        return DraftOutput(tokens=tokens, probs=probs)
    
    def reset(self) -> None:
        """Reset cached state."""
        logger.debug("reset called")
        self._cached_hidden = None
    
    def _fallback(self, batch_size: int, num_tokens: int) -> DraftOutput:
        """Return uniform distribution when no hidden states."""
        logger.debug("_fallback called with batch_size=%s, num_tokens=%s", batch_size, num_tokens)
        vocab_size = self.head.vocab_size
        tokens = torch.zeros(
            batch_size, num_tokens,
            dtype=torch.long, device=self.device
        )
        probs = torch.ones(
            batch_size, num_tokens, vocab_size,
            dtype=torch.float32, device=self.device
        ) / vocab_size
        return DraftOutput(tokens=tokens, probs=probs)


__all__ = ["MTPDraft"]
