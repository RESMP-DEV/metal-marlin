"""
Flash Attention wrapper for Metal.

This module provides a high-level FlashAttention class that integrates
with the optimized Metal kernels while providing fallback to PyTorch's
scaled_dot_product_attention for compatibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._compat import HAS_MPS
from .attention import scaled_dot_product_attention_metal
from .flash_attention_v2 import flash_attention_v2


class FlashAttention(nn.Module):
    """
    Flash Attention layer with input/output reshaping and Metal optimization.

    This class wraps the optimized flash_attention_v2 kernel for Apple Silicon,
    handling the necessary data layout requirements and providing a robust
    fallback to standard PyTorch attention when Metal is not available or
    when input conditions (like arbitrary attention masks) are not supported.

    Args:
        causal (bool): Whether to apply causal masking. Default: True.
        scale (float | None): Attention scale factor. If None, uses 1/sqrt(head_dim).
        use_flash_attention (bool): whether to use flash attention kernel.
    """

    def __init__(
        self,
        causal: bool = True,
        scale: float | None = None,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        self.causal = causal
        self.scale = scale
        self.use_flash_attention = use_flash_attention

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for Flash Attention.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_kv_heads, kv_seq_len, head_dim]
            v: Value tensor [batch, num_kv_heads, kv_seq_len, head_dim]
            attention_mask: Optional attention mask [batch, num_heads, seq_len, kv_seq_len]

        Returns:
            Output tensor [batch, seq_len, num_heads * v_head_dim]
            Note: The output is flattened (reshaped) for linear projection.
        """
        batch_size, num_heads, seq_len, _ = q.shape
        _, _, _, v_head_dim = v.shape

        # Check if we can use the optimized Metal kernel
        # 1. User enabled it
        # 2. MPS is available and input is on MPS
        # 3. No arbitrary attention mask (kernel only supports causal/none)
        can_use_flash = (
            self.use_flash_attention
            and HAS_MPS
            and q.is_mps
            and attention_mask is None
        )

        if can_use_flash:
            # Use optimized Flash Attention V2 kernel
            # The kernel handles GQA (num_kv_heads < num_heads) internally
            scale = self.scale if self.scale is not None else q.shape[-1] ** -0.5

            # flash_attention_v2 returns [batch, num_heads, seq_len, head_dim]
            attn_output = flash_attention_v2(
                q, k, v, scale=scale, causal=self.causal
            )
        else:
            # Fallback to PyTorch's scaled_dot_product_attention
            # This handles arbitrary masks and CPU/CUDA devices

            # If standard SDPA is used, we need to handle GQA expansion manually if needed
            # (Note: scaled_dot_product_attention_metal handles GQA expansion internally in some versions,
            # but standard F.scaled_dot_product_attention might typically expect matching heads or broadcasting.
            # However, PyTorch 2.0+ SDPA supports GQA broadcasting.
            # Let's rely on scaled_dot_product_attention_metal from .attention module
            # which likely has the correct logic.)

            attn_output = scaled_dot_product_attention_metal(
                q,
                k,
                v,
                attn_mask=attention_mask,
                scale=self.scale,
                is_causal=self.causal and attention_mask is None,
            )

        # Output reshaping: [batch, num_heads, seq_len, v_head_dim] -> [batch, seq_len, num_heads * v_head_dim]
        # This prepares the tensor for the output linear projection
        output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, num_heads * v_head_dim)
        )

        return output
