"""Conformer attention module for ASR models.

Implements multi-head self-attention with relative positional embeddings
as used in the Conformer architecture.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer_config import ConformerConfig


class ConformerAttention(nn.Module):
    """Conformer multi-head self-attention module with relative positional encoding.

    Implements the MHSA component used in Conformer architecture with
    relative positional embeddings as described in:
    "Conformer: Convolution-augmented Transformer for Speech Recognition"
    (https://arxiv.org/abs/2005.08100)

    The attention mechanism incorporates relative positional information through
    learned biases per attention head, allowing the model to better capture
    sequential dependencies in speech data.
    """

    def __init__(self, config: ConformerConfig):
        """Initialize ConformerAttention module.

        Args:
            config: ConformerConfig containing model parameters
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        assert self.head_dim * self.num_heads == self.hidden_size, (
            "hidden_size must be divisible by num_heads"
        )

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        # Relative positional bias projection: projects pos embeddings to per-head biases
        self.pos_bias = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Scale factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of ConformerAttention module.

        Implements relative position attention where positional embeddings are
        used to compute head-specific bias terms that are added to attention scores.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            pos_emb: Positional embeddings of shape (batch_size, seq_len, hidden_size)
            mask: Attention mask of shape (batch_size, seq_len, seq_len) or
                  broadcastable to (batch_size, 1, seq_len, seq_len)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (B, T, H*D)
        k = self.k_proj(x)  # (B, T, H*D)
        v = self.v_proj(x)  # (B, T, H*D)

        # Reshape for multi-head attention: (B, T, H, D) -> (B, H, T, D)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute relative positional bias: (B, T, H) -> (B, H, T)
        pos_bias = self.pos_bias(pos_emb).transpose(1, 2)  # (B, H, T)

        # Compute attention scores: (B, H, T, T)
        # Standard scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # Add relative positional bias to attention scores
        # pos_bias is (B, H, T), we add it to every column (key position) of scores
        scores = scores + pos_bias.unsqueeze(2)  # (B, H, T, T) + (B, H, 1, T)

        # Apply mask if provided
        if mask is not None:
            # mask should be broadcastable to (B, H, T, T)
            # For causal/padding masks: mask == 0 means masked positions
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (B, H, T, D)

        # Reshape back: (B, H, T, D) -> (B, T, H*D)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        )

        # Final projection
        output = self.out_proj(attn_output)

        return output
