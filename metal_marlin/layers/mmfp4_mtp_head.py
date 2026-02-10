"""Multi-Token Prediction head for MMFP4 models.

MTP heads predict N future tokens from hidden states,
enabling speculative decoding without a separate draft model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mmfp4_linear import MMFP4Linear


class MMFP4MTPHead(nn.Module):
    """Multi-Token Prediction head using MMFP4 quantization.
    
    Predicts next N tokens from current hidden state.
    Each prediction head shares the input projection but has
    separate output projections for different lookahead distances.
    
    Args:
        hidden_size: Model hidden dimension
        vocab_size: Output vocabulary size
        num_predictions: Number of tokens to predict (default 4)
        group_size: MMFP4 quantization group size
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_predictions: int = 4,
        group_size: int = 128,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_predictions = num_predictions
        
        # Shared input projection (hidden -> intermediate)
        intermediate_size = hidden_size // 2  # Smaller for efficiency
        self.input_proj = nn.Linear(hidden_size, intermediate_size)
        self.norm = nn.RMSNorm(intermediate_size, eps=1e-5)
        
        # Separate output heads for each prediction position
        self.output_heads = nn.ModuleList([
            nn.Linear(intermediate_size, vocab_size, bias=False)
            for _ in range(num_predictions)
        ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next N tokens.
        
        Args:
            hidden_states: [batch, seq, hidden]
        
        Returns:
            logits: [batch, seq, num_predictions, vocab]
        """
        # Take last position for decode
        h = hidden_states[:, -1:, :]  # [batch, 1, hidden]
        
        # Shared transformation
        h = self.input_proj(h)  # [batch, 1, intermediate]
        h = F.silu(h)
        h = self.norm(h)
        
        # Predict each position
        predictions = []
        for head in self.output_heads:
            logits = head(h)  # [batch, 1, vocab]
            predictions.append(logits)
        
        # Stack: [batch, 1, num_predictions, vocab]
        return torch.stack(predictions, dim=2).squeeze(1)
    
    def speculate(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate speculative tokens with probabilities.
        
        Args:
            hidden_states: [batch, seq, hidden]
            temperature: Sampling temperature
        
        Returns:
            tokens: [batch, num_predictions] predicted token IDs
            probs: [batch, num_predictions, vocab] probability distributions
        """
        logits = self.forward(hidden_states)  # [batch, num_predictions, vocab]
        
        if temperature > 0:
            scaled_logits = logits / temperature
        else:
            scaled_logits = logits
        
        probs = F.softmax(scaled_logits, dim=-1)
        tokens = probs.argmax(dim=-1)  # Greedy for max acceptance
        
        return tokens, probs


__all__ = ["MMFP4MTPHead"]
