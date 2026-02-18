"""
Multi-head attention layer with Marlin-quantized projections.

Uses MarlinLinear for Q/K/V/O projections with FP4 quantized weights.
Supports Grouped Query Attention (GQA) where num_kv_heads < num_heads.

This implementation uses PyTorch with MPS device for Apple Silicon,
and delegates attention computation to attention_metal functions.

Usage:
    from metal_marlin.python.attention import MarlinAttention

    attn = MarlinAttention(
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,  # GQA
    )
    output = attn(hidden_states, kv_cache=cache, layer_idx=0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kv_cache import KVCache
from .layers import MarlinLinear


@dataclass
class BlockSparseMask:
    """
    Block-sparse attention mask configuration.

    Block-sparse attention divides the attention matrix into blocks of size
    (block_q x block_k) and only computes attention for active blocks.
    This reduces both memory and computation for sparse attention patterns
    like those used in BigBird, Longformer, and sliding window attention.

    The mask is stored as a list of uint64 bitsets, where each bit represents
    one K block. This compact representation enables efficient GPU processing.

    Attributes:
        mask_bits: List of uint64 bitsets, one per Q block. Bit j in mask_bits[i]
                   is 1 if Q block i can attend to K block j.
        block_q: Block size in query dimension (rows per block)
        block_k: Block size in key dimension (columns per block)
        num_q_blocks: Total number of query blocks
        num_k_blocks: Total number of key blocks
        seq_q: Query sequence length
        seq_k: Key sequence length

    Example:
        # Create a block-sparse mask for sliding window attention
        mask = BlockSparseMask.create_sliding_window(
            seq_len=4096,
            window_size=1024,
            block_size=64,
        )
    """

    mask_bits: torch.Tensor  # [num_q_blocks, num_mask_words] uint64 tensor
    block_q: int
    block_k: int
    num_q_blocks: int
    num_k_blocks: int
    seq_q: int
    seq_k: int

    @staticmethod
    def create_sliding_window(
        seq_len: int,
        window_size: int,
        block_size: int = 64,
        device: torch.device | None = None,
    ) -> BlockSparseMask:
        """
        Create a block-sparse mask for sliding window + causal attention.

        Each query position can only attend to positions within the window
        and before it (causal constraint).

        Args:
            seq_len: Sequence length (assumes seq_q == seq_k)
            window_size: Maximum distance to attend to (in tokens)
            block_size: Block size for both Q and K dimensions
            device: Device to create mask on

        Returns:
            BlockSparseMask configured for sliding window attention
        """
        if device is None:
            device = _get_device()

        num_blocks = (seq_len + block_size - 1) // block_size
        num_mask_words = (num_blocks + 63) // 64
        
        # Create mask on CPU (MPS doesn't support uint64 bitwise ops)
        mask_bits_cpu = torch.zeros((num_blocks, num_mask_words), dtype=torch.uint64)

        for q_block_idx in range(num_blocks):
            q_start = q_block_idx * block_size
            q_end = min(q_start + block_size, seq_len)

            # For each query position in this block, find valid K blocks
            # The window extends from max(0, q_end - window_size) to q_end
            window_start = max(0, q_end - window_size)
            k_block_start = window_start // block_size
            k_block_end = q_block_idx + 1  # Causal: can only attend to <= q_block

            # Set bits for all valid K blocks
            for k_block_idx in range(k_block_start, min(k_block_end, num_blocks)):
                word_idx = k_block_idx // 64
                bit_idx = k_block_idx % 64
                mask_bits_cpu[q_block_idx, word_idx] |= (1 << bit_idx)

        # Move to target device
        mask_bits = mask_bits_cpu.to(device)

        return BlockSparseMask(
            mask_bits=mask_bits,
            block_q=block_size,
            block_k=block_size,
            num_q_blocks=num_blocks,
            num_k_blocks=num_blocks,
            seq_q=seq_len,
            seq_k=seq_len,
        )

    @staticmethod
    def create_bigbird(
        seq_len: int,
        num_random_blocks: int = 3,
        num_global_blocks: int = 2,
        window_size: int = 256,
        block_size: int = 64,
        device: torch.device | None = None,
    ) -> BlockSparseMask:
        """
        Create a block-sparse mask for BigBird-style attention.

        BigBird combines:
        - Local sliding window attention
        - Random global blocks (visible to all positions)
        - Fixed global blocks at sequence start

        Args:
            seq_len: Sequence length
            num_random_blocks: Number of random global blocks
            num_global_blocks: Number of fixed global blocks at start
            window_size: Sliding window size
            block_size: Block size for both Q and K dimensions
            device: Device to create mask on

        Returns:
            BlockSparseMask configured for BigBird attention
        """
        if device is None:
            device = _get_device()

        num_blocks = (seq_len + block_size - 1) // block_size
        num_mask_words = (num_blocks + 63) // 64
        
        # Create mask on CPU (MPS doesn't support uint64 bitwise ops)
        mask_bits_cpu = torch.zeros((num_blocks, num_mask_words), dtype=torch.uint64)

        # Fixed global blocks (first num_global_blocks blocks)
        global_mask_bits = [0] * num_mask_words
        for g in range(min(num_global_blocks, num_blocks)):
            word_idx = g // 64
            bit_idx = g % 64
            global_mask_bits[word_idx] |= (1 << bit_idx)

        # Random global blocks
        if num_random_blocks > 0 and num_blocks > num_global_blocks:
            torch.manual_seed(42)  # For reproducibility
            random_blocks = torch.randperm(num_blocks - num_global_blocks)[:num_random_blocks]
            for r in random_blocks:
                idx = r.item() + num_global_blocks
                word_idx = idx // 64
                bit_idx = idx % 64
                global_mask_bits[word_idx] |= (1 << bit_idx)

        # Sliding window blocks
        window_blocks = window_size // block_size

        for q_block_idx in range(num_blocks):
            # Start with global mask
            for w in range(num_mask_words):
                mask_bits_cpu[q_block_idx, w] = global_mask_bits[w]

            # Add sliding window blocks
            window_start = max(0, q_block_idx - window_blocks)
            window_end = min(num_blocks, q_block_idx + window_blocks + 1)

            for k_block_idx in range(window_start, window_end):
                word_idx = k_block_idx // 64
                bit_idx = k_block_idx % 64
                mask_bits_cpu[q_block_idx, word_idx] |= (1 << bit_idx)

            # Apply causal constraint (only attend to past)
            # Mask out any bits > q_block_idx
            q_word_idx = q_block_idx // 64
            q_bit_idx = q_block_idx % 64
            
            # For words completely after q_block, clear all bits
            for w in range(q_word_idx + 1, num_mask_words):
                mask_bits_cpu[q_block_idx, w] = 0
            
            # For the word containing q_block, mask out higher bits
            causal_mask = (1 << (q_bit_idx + 1)) - 1
            mask_bits_cpu[q_block_idx, q_word_idx] &= causal_mask

        # Move to target device
        mask_bits = mask_bits_cpu.to(device)

        return BlockSparseMask(
            mask_bits=mask_bits,
            block_q=block_size,
            block_k=block_size,
            num_q_blocks=num_blocks,
            num_k_blocks=num_blocks,
            seq_q=seq_len,
            seq_k=seq_len,
        )

    @staticmethod
    def create_longformer(
        seq_len: int,
        window_size: int = 256,
        num_global_tokens: int = 16,
        block_size: int = 64,
        device: torch.device | None = None,
    ) -> BlockSparseMask:
        """
        Create a block-sparse mask for Longformer-style attention.

        Longformer uses:
        - Fixed-size sliding window (dilated)
        - Global attention on specific tokens (e.g., [CLS])

        Args:
            seq_len: Sequence length
            window_size: Sliding window size on each side
            num_global_tokens: Number of tokens with global attention
            block_size: Block size for both Q and K dimensions
            device: Device to create mask on

        Returns:
            BlockSparseMask configured for Longformer attention
        """
        if device is None:
            device = _get_device()

        num_blocks = (seq_len + block_size - 1) // block_size
        num_mask_words = (num_blocks + 63) // 64
        global_blocks = (num_global_tokens + block_size - 1) // block_size

        # Create mask on CPU using numpy to avoid overflow issues
        mask_bits_np = np.zeros((num_blocks, num_mask_words), dtype=np.uint64)

        window_blocks = window_size // block_size

        for q_block_idx in range(num_blocks):
            # Global blocks (visible to all, within causal constraint)
            for g in range(min(global_blocks, num_blocks)):
                if q_block_idx >= g:  # Causal check
                    word_idx = g // 64
                    bit_idx = g % 64
                    mask_bits_np[q_block_idx, word_idx] |= (np.uint64(1) << bit_idx)

            # Sliding window (within causal constraint)
            window_start = max(global_blocks, q_block_idx - window_blocks)
            window_end = min(num_blocks, q_block_idx + 1)  # Causal: only up to q_block_idx

            for k_block_idx in range(window_start, window_end):
                word_idx = k_block_idx // 64
                bit_idx = k_block_idx % 64
                mask_bits_np[q_block_idx, word_idx] |= (np.uint64(1) << bit_idx)

        # Convert to torch tensor
        mask_bits = torch.from_numpy(mask_bits_np).to(device)

        return BlockSparseMask(
            mask_bits=mask_bits,
            block_q=block_size,
            block_k=block_size,
            num_q_blocks=num_blocks,
            num_k_blocks=num_blocks,
            seq_q=seq_len,
            seq_k=seq_len,
        )

    @staticmethod
    def from_dense_mask(
        dense_mask: torch.Tensor,
        block_q: int = 64,
        block_k: int = 64,
    ) -> BlockSparseMask:
        """
        Convert a dense attention mask to block-sparse format.

        Args:
            dense_mask: Dense mask tensor [seq_q, seq_k] where True/0 means attend
            block_q: Block size in query dimension
            block_k: Block size in key dimension

        Returns:
            BlockSparseMask representing the same mask pattern
        """
        seq_q, seq_k = dense_mask.shape
        device = dense_mask.device

        num_q_blocks = (seq_q + block_q - 1) // block_q
        num_k_blocks = (seq_k + block_k - 1) // block_k
        num_mask_words = (num_k_blocks + 63) // 64

        # Create mask on CPU using numpy to avoid overflow issues
        mask_bits_np = np.zeros((num_q_blocks, num_mask_words), dtype=np.uint64)

        # Move dense mask to CPU for processing
        dense_mask_cpu = dense_mask.cpu()

        for q_block_idx in range(num_q_blocks):
            q_start = q_block_idx * block_q
            q_end = min(q_start + block_q, seq_q)

            for k_block_idx in range(num_k_blocks):
                k_start = k_block_idx * block_k
                k_end = min(k_start + block_k, seq_k)

                # Check if any position in this block is unmasked
                block = dense_mask_cpu[q_start:q_end, k_start:k_end]
                if block.any():
                    word_idx = k_block_idx // 64
                    bit_idx = k_block_idx % 64
                    mask_bits_np[q_block_idx, word_idx] |= (np.uint64(1) << bit_idx)

        # Convert to torch tensor and move to device
        mask_bits = torch.from_numpy(mask_bits_np).to(device)

        return BlockSparseMask(
            mask_bits=mask_bits,
            block_q=block_q,
            block_k=block_k,
            num_q_blocks=num_q_blocks,
            num_k_blocks=num_k_blocks,
            seq_q=seq_q,
            seq_k=seq_k,
        )

if TYPE_CHECKING:
    pass


def _get_device() -> torch.device:
    """Get the appropriate device (MPS on Apple Silicon, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RoPE(nn.Module):
    """Rotary Position Embedding."""

    def __init__(self, dims: int, traditional: bool = False, base: float = 10000.0):
        super().__init__()
        self.dims = dims
        self.traditional = traditional
        self.base = base

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0,
    ) -> torch.Tensor:
        """Apply RoPE to input tensor.

        Args:
            x: Input tensor [batch, num_heads, seq_len, head_dim]
            offset: Position offset for KV cache

        Returns:
            Tensor with RoPE applied
        """
        shape = x.shape
        seq_len = shape[2]
        head_dim = shape[3]
        device = x.device
        dtype = x.dtype

        # Compute position indices
        positions = torch.arange(
            offset, offset + seq_len, dtype=torch.float32, device=device)

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dims, 2, dtype=torch.float32, device=device) / self.dims)
        )

        # Compute angles: [seq_len, dims/2]
        freqs = torch.outer(positions, inv_freq)

        # Compute cos and sin
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # Expand dims for broadcasting: [1, 1, seq_len, dims/2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Split into even and odd indices
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # Apply rotation
        if self.traditional:
            # Traditional RoPE
            x_rotated_even = x_even * cos - x_odd * sin
            x_rotated_odd = x_even * sin + x_odd * cos
        else:
            # Llama-style RoPE (half-rotation)
            x_rotated_even = x_even * cos - x_odd * sin
            x_rotated_odd = x_odd * cos + x_even * sin

        # Interleave back
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        x_rotated = x_rotated.view(*shape[:-1], head_dim)

        return x_rotated.to(dtype)


class MarlinAttention(nn.Module):
    """
    Multi-head attention with Marlin-quantized projections.

    Supports:
    - Standard MHA (num_heads == num_kv_heads)
    - Grouped Query Attention (num_kv_heads < num_heads)
    - KV caching for efficient autoregressive generation
    - RoPE position embeddings
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        quant_type: str = "fp4",
        group_size: int = 128,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 4096,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = self.head_dim**-0.5

        # Quantized projections
        self.q_proj = MarlinLinear(
            hidden_size,
            num_heads * self.head_dim,
            bias=bias,
            quant_type=quant_type,
            group_size=group_size,
        )
        self.k_proj = MarlinLinear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=bias,
            quant_type=quant_type,
            group_size=group_size,
        )
        self.v_proj = MarlinLinear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=bias,
            quant_type=quant_type,
            group_size=group_size,
        )
        self.o_proj = MarlinLinear(
            num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            quant_type=quant_type,
            group_size=group_size,
        )

        # RoPE embeddings
        self.rope = RoPE(self.head_dim, base=rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, hidden_size]
        attention_mask: torch.Tensor | BlockSparseMask | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask (Tensor or BlockSparseMask)
            position_ids: Optional position IDs for RoPE
            kv_cache: Optional KV cache for autoregressive generation
            layer_idx: Layer index for KV cache

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V projections using Marlin kernels
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads,
                   self.head_dim).transpose(1, 2)

        # Apply RoPE
        position_offset = kv_cache.seq_len if kv_cache else 0
        q = self.rope(q, offset=position_offset)
        k = self.rope(k, offset=position_offset)

        # Update KV cache
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        # Expand K, V for GQA
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # Handle Block-Sparse Attention
        if isinstance(attention_mask, BlockSparseMask):
            attn_output = block_sparse_attention_metal(
                q,
                k,
                v,
                block_sparse_mask=attention_mask,
                scale=self.scale,
                causal=False,  # Mask handles causal logic if constructed that way
            )
        else:
            # Optimized causal mask handling
            # Decode (seq_len == 1): Skip mask computation, hardcode causal constraint
            # Prefill (seq_len > 1): Use efficient bitmask or standard causal mask
            if attention_mask is None:
                if seq_len == 1:
                    # Decode: No mask needed, rely on is_causal=True
                    is_causal = True
                    attn_mask = None
                else:
                    # Prefill: Create causal mask if none provided
                    kv_seq_len = k.shape[-2]
                    attn_mask = create_causal_mask(
                        seq_len, kv_seq_len, device=q.device)
                    is_causal = False
            else:
                attn_mask = attention_mask
                is_causal = False

            # Use Metal-optimized attention via scaled_dot_product_attention
            # This will use the MPS backend's optimized attention implementation
            attn_output = scaled_dot_product_attention_metal(
                q,
                k,
                v,
                attn_mask=attn_mask,
                scale=self.scale,
                is_causal=is_causal,
            )

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.head_dim)
        )
        output = self.o_proj(attn_output)

        return output


def scaled_dot_product_attention_metal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Scaled dot-product attention with Metal-optimized backends.

    Dispatches to fused_attention() which tries in order:
    1. fused_scaled_dot_product_attention (MPSGraph)
    2. flash_attention_v2 (Metal kernels)
    3. F.scaled_dot_product_attention (PyTorch fallback)

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, kv_seq_len, head_dim]
        v: Value tensor [batch, num_heads, kv_seq_len, head_dim]
        attn_mask: Optional attention mask [batch, num_heads, seq_len, kv_seq_len]
        scale: Optional scale factor. If None, uses 1/sqrt(head_dim)
        is_causal: If True, applies causal masking

    Returns:
        Attention output [batch, num_heads, seq_len, head_dim]
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    # Use fused_attention dispatcher which tries optimized backends
    try:
        from .fused_attention_mps import fused_attention
        return fused_attention(
            q,
            k,
            v,
            mask=attn_mask,
            scale=scale,
            causal=is_causal,
        )
    except Exception:
        # Fallback to PyTorch's SDPA
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
        )


def flash_attention_metal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    causal: bool = True,
    num_kv_heads: int | None = None,
) -> torch.Tensor:
    """
    Flash attention implementation for Metal/MPS devices.

    This function provides a unified API for attention computation that
    can be called from ONNX converters and other components.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_kv_heads, kv_seq_len, head_dim]
        v: Value tensor [batch, num_kv_heads, kv_seq_len, head_dim]
        scale: Optional scale factor. If None, uses 1/sqrt(head_dim)
        causal: Whether to apply causal masking
        num_kv_heads: Number of KV heads for GQA. If provided and different
            from num_heads, K and V will be repeated accordingly.

    Returns:
        Attention output [batch, num_heads, seq_len, head_dim]
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    num_heads = q.shape[1]

    # Handle GQA by repeating K/V heads
    if num_kv_heads is not None and num_kv_heads < num_heads:
        repeat_factor = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

    return scaled_dot_product_attention_metal(
        q,
        k,
        v,
        scale=scale,
        is_causal=causal,
    )


def sliding_window_attention_metal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    scale: float | None = None,
    causal: bool = True,
    num_kv_heads: int | None = None,
) -> torch.Tensor:
    """
    Sliding window attention for Metal/MPS devices.

    Each token only attends to the most recent window_size tokens, providing
    O(seq * window) memory complexity instead of O(seq^2). This is the
    attention pattern used by Mistral and similar models.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_kv_heads, kv_seq_len, head_dim]
        v: Value tensor [batch, num_kv_heads, kv_seq_len, head_dim]
        window_size: Sliding window size (e.g., 4096 for Mistral)
        scale: Optional scale factor. If None, uses 1/sqrt(head_dim)
        causal: Whether to apply causal masking within window
        num_kv_heads: Number of KV heads for GQA. If provided and different
            from num_heads, K and V will be repeated accordingly.

    Returns:
        Attention output [batch, num_heads, seq_len, head_dim]
    """
    # Import here to avoid circular dependency
    from .sliding_window_attention import sliding_window_attention

    if scale is None:
        scale = q.shape[-1] ** -0.5

    num_heads = q.shape[1]

    # Handle GQA by repeating K/V heads
    # Note: The sliding_window_attention kernel handles GQA natively,
    # but for compatibility we can also expand here
    if num_kv_heads is not None and num_kv_heads < num_heads:
        repeat_factor = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

    return sliding_window_attention(
        q,
        k,
        v,
        window_size=window_size,
        scale=scale,
        causal=causal,
    )


def create_causal_bitmask(
    seq_len: int,
    kv_seq_len: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    """
    Create causal attention mask as bitmask for efficient prefill.

    Uses 64-bit integers (uint64) to pack mask bits, enabling
    fast bitwise operations in Metal kernels instead of per-element
    comparisons.

    Args:
        seq_len: Query sequence length
        kv_seq_len: Key/value sequence length (defaults to seq_len)
        device: Device to create the mask on

    Returns:
        Bitmask tensor [num_words, seq_len] where num_words = ceil(kv_seq_len / 64)
        Each uint64 contains 64 mask bits: bit k = 0 (unmasked) if k <= q_pos
        Returns None for single-token decode
    """
    kv_seq_len = kv_seq_len or seq_len

    # Single token decode - no masking needed
    if seq_len == 1:
        return None

    if device is None:
        device = _get_device()

    # Compute number of 64-bit words needed
    num_words = (kv_seq_len + 63) // 64

    # Vectorized bitmask creation
    # Create query position indices: [seq_len]
    q_pos = torch.arange(seq_len, device=device)

    # Compute start bit for each query: [seq_len]
    start_bits = (q_pos + 1) % 64

    # Compute which word each mask starts in: [seq_len]
    start_words = (q_pos + 1) // 64

    # Initialize bitmask: [num_words, seq_len]
    bitmask = torch.zeros((num_words, seq_len),
                          dtype=torch.uint64, device=device)

    # All ones mask (0xFFFFFFFFFFFFFFFF)
    all_ones = torch.tensor(
        0xFFFFFFFFFFFFFFFF, dtype=torch.uint64, device=device)

    # For each query position, set the appropriate words
    # Create a mask of which queries affect which words
    word_range = torch.arange(
        num_words, device=device).unsqueeze(1)  # [num_words, 1]

    # Masked positions: words >= start_words
    mask_mask = word_range >= start_words.unsqueeze(0)  # [num_words, seq_len]

    # Set full words (after the first partial word)
    full_word_mask = mask_mask & (word_range > start_words.unsqueeze(0))
    bitmask[full_word_mask] = all_ones

    # Set partial words (first masked word for each query)
    partial_mask = mask_mask & (word_range == start_words.unsqueeze(0))
    # Compute partial mask values: all_ones << start_bit
    partial_values = torch.zeros(
        (num_words, seq_len), dtype=torch.uint64, device=device)
    partial_values[partial_mask] = all_ones.unsqueeze(
        0) << start_bits.unsqueeze(0)[partial_mask]
    bitmask = bitmask | partial_values

    # Handle cases where mask_start >= kv_seq_len (no masking needed)
    valid_queries = q_pos < kv_seq_len - 1
    bitmask = bitmask[:, valid_queries]

    return bitmask.unsqueeze(0)  # [1, num_words, seq_len_valid]


def create_causal_mask(
    seq_len: int,
    kv_seq_len: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    """
    Create causal attention mask.

    Optimized for prefill (seq_len > 1) with efficient triangular mask creation.
    For decode (seq_len == 1), returns None to rely on kernel's built-in causal handling.

    Args:
        seq_len: Query sequence length
        kv_seq_len: Key/value sequence length (defaults to seq_len)
        device: Device to create the mask on

    Returns:
        Causal mask with -inf for masked positions, or None for single-token decode
    """
    kv_seq_len = kv_seq_len or seq_len

    # Single token decode - no masking needed
    # The attention kernel will handle causal constraint via is_causal parameter
    if seq_len == 1:
        return None

    if device is None:
        device = _get_device()

    # Prefill: efficient triangular mask
    # Use tril with zero filling, then negate to get -inf pattern
    # This is more efficient than triu with -inf
    mask = torch.zeros((seq_len, kv_seq_len), device=device)
    mask = torch.tril(mask, diagonal=0)
    mask = torch.where(mask == 0, torch.tensor(
        float("-inf"), device=device), mask)

    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, kv_seq_len]


def block_sparse_attention_metal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_sparse_mask: BlockSparseMask,
    scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Block-sparse attention computation using optimized Metal kernels.

    Computes attention only for active blocks specified by the block_sparse_mask,
    significantly reducing memory bandwidth and computation for sparse patterns.

    Args:
        q: Query tensor [batch, num_heads, seq_q, head_dim], MPS device
        k: Key tensor [batch, num_heads, seq_k, head_dim], MPS device
        v: Value tensor [batch, num_heads, seq_k, head_dim], MPS device
        block_sparse_mask: BlockSparseMask defining active attention blocks
        scale: Optional attention scale factor (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking within active blocks

    Returns:
        Output tensor [batch, num_heads, seq_q, head_dim], MPS device

    Example:
        # Create block-sparse mask for sliding window attention
        mask = BlockSparseMask.create_sliding_window(
            seq_len=4096,
            window_size=1024,
            block_size=64,
        )

        # Compute block-sparse attention
        output = block_sparse_attention_metal(q, k, v, mask, causal=True)

    Note:
        This function dispatches to the attention_block_sparse_fused_qkv Metal kernel
        which provides optimized memory access patterns for sparse attention.
    """
    from ._compat import HAS_MPS, HAS_PYOBJC_METAL, HAS_TORCH

    if not HAS_TORCH or torch is None:
        raise RuntimeError("Block-sparse attention requires PyTorch")
    if not HAS_MPS:
        raise RuntimeError("Block-sparse attention requires MPS backend (Apple Silicon)")
    if not HAS_PYOBJC_METAL:
        raise RuntimeError(
            "Block-sparse attention requires PyObjC Metal. Install with:\n"
            "  pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders"
        )

    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    if scale is None:
        scale = head_dim**-0.5

    # Validate dimensions match mask
    if seq_q != block_sparse_mask.seq_q or seq_k != block_sparse_mask.seq_k:
        raise ValueError(
            f"Sequence length mismatch: q={seq_q}, k={seq_k}, "
            f"mask=({block_sparse_mask.seq_q}, {block_sparse_mask.seq_k})"
        )

    # Ensure tensors are on MPS and contiguous
    q = q.to(device="mps", dtype=torch.float16).contiguous()
    k = k.to(device="mps", dtype=torch.float16).contiguous()
    v = v.to(device="mps", dtype=torch.float16).contiguous()

    # Import Metal libraries
    import Metal
    import numpy as np
    from pathlib import Path

    # Load and compile the attention kernel
    kernel_path = Path(__file__).parent.parent / "src" / "attention.metal"
    if not kernel_path.exists():
        raise FileNotFoundError(f"Metal kernel not found: {kernel_path}")

    # Create Metal device and compile kernel
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("No Metal device available")

    # Read kernel source with includes
    source = kernel_path.read_text()
    include_path = kernel_path.parent / "reduction_helpers.metal"
    if include_path.exists():
        include_source = include_path.read_text()
        source = source.replace('#include "reduction_helpers.metal"', include_source)

    # Compile options
    options = Metal.MTLCompileOptions.new()
    options.setLanguageVersion_(Metal.MTLLanguageVersion3_0)

    library, err = device.newLibraryWithSource_options_error_(source, options, None)
    if err is not None:
        raise RuntimeError(f"Metal compilation error: {err}")

    # Get the kernel function
    func = library.newFunctionWithName_("attention_block_sparse_fused_qkv")
    if func is None:
        raise RuntimeError("Kernel 'attention_block_sparse_fused_qkv' not found")

    pipeline, err = device.newComputePipelineStateWithFunction_error_(func, None)
    if err is not None:
        raise RuntimeError(f"Pipeline creation error: {err}")

    # Allocate output tensor
    output = torch.empty(
        (batch, num_heads, seq_q, head_dim), dtype=torch.float16, device="mps"
    )

    # Helper to create Metal buffers from tensors
    def _tensor_to_buffer(tensor: torch.Tensor) -> Any:
        storage = tensor.untyped_storage()
        ptr = storage.data_ptr()
        size = storage.nbytes()

        # Try zero-copy first
        try:
            buffer = device.newBufferWithBytesNoCopy_length_options_deallocator_(
                ptr, size, Metal.MTLResourceStorageModeShared, None
            )
            if buffer is not None:
                return buffer
        except (TypeError, ValueError):
            pass

        # Fallback to copy-based approach
        import ctypes

        arr_type = ctypes.c_uint8 * size
        arr = arr_type.from_address(ptr)
        data = bytes(arr)
        return device.newBufferWithBytes_length_options_(
            data, len(data), Metal.MTLResourceStorageModeShared
        )

    def _make_uint32_buffer(val: int):
        data = np.array([val], dtype=np.uint32)
        return device.newBufferWithBytes_length_options_(
            data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared
        )

    def _make_float_buffer(val: float):
        data = np.array([val], dtype=np.float32)
        return device.newBufferWithBytes_length_options_(
            data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared
        )

    # Create buffers
    q_buf = _tensor_to_buffer(q)
    k_buf = _tensor_to_buffer(k)
    v_buf = _tensor_to_buffer(v)
    o_buf = _tensor_to_buffer(output)

    # Mask bits buffer (uint64)
    mask_bits_np = block_sparse_mask.mask_bits.cpu().numpy().astype(np.uint64)
    mask_buf = device.newBufferWithBytes_length_options_(
        mask_bits_np.tobytes(),
        mask_bits_np.nbytes,
        Metal.MTLResourceStorageModeShared,
    )

    # Calculate number of mask words per Q block (ceil(num_k_blocks / 64))
    num_mask_words = (block_sparse_mask.num_k_blocks + 63) // 64

    # Constant buffers
    batch_buf = _make_uint32_buffer(batch)
    num_heads_buf = _make_uint32_buffer(num_heads)
    seq_q_buf = _make_uint32_buffer(seq_q)
    seq_k_buf = _make_uint32_buffer(seq_k)
    head_dim_buf = _make_uint32_buffer(head_dim)
    scale_buf = _make_float_buffer(scale)
    block_q_buf = _make_uint32_buffer(block_sparse_mask.block_q)
    block_k_buf = _make_uint32_buffer(block_sparse_mask.block_k)
    num_q_blocks_buf = _make_uint32_buffer(block_sparse_mask.num_q_blocks)
    num_k_blocks_buf = _make_uint32_buffer(block_sparse_mask.num_k_blocks)
    num_mask_words_buf = _make_uint32_buffer(num_mask_words)
    causal_buf = _make_uint32_buffer(1 if causal else 0)

    # Create command queue and buffer
    command_queue = device.newCommandQueue()
    command_buffer = command_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()

    # Set pipeline and buffers
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(q_buf, 0, 0)
    encoder.setBuffer_offset_atIndex_(k_buf, 0, 1)
    encoder.setBuffer_offset_atIndex_(v_buf, 0, 2)
    encoder.setBuffer_offset_atIndex_(mask_buf, 0, 3)
    encoder.setBuffer_offset_atIndex_(o_buf, 0, 4)
    encoder.setBuffer_offset_atIndex_(batch_buf, 0, 5)
    encoder.setBuffer_offset_atIndex_(num_heads_buf, 0, 6)
    encoder.setBuffer_offset_atIndex_(seq_q_buf, 0, 7)
    encoder.setBuffer_offset_atIndex_(seq_k_buf, 0, 8)
    encoder.setBuffer_offset_atIndex_(head_dim_buf, 0, 9)
    encoder.setBuffer_offset_atIndex_(scale_buf, 0, 10)
    encoder.setBuffer_offset_atIndex_(block_q_buf, 0, 11)
    encoder.setBuffer_offset_atIndex_(block_k_buf, 0, 12)
    encoder.setBuffer_offset_atIndex_(num_q_blocks_buf, 0, 13)
    encoder.setBuffer_offset_atIndex_(num_k_blocks_buf, 0, 14)
    encoder.setBuffer_offset_atIndex_(num_mask_words_buf, 0, 15)
    encoder.setBuffer_offset_atIndex_(causal_buf, 0, 16)

    # Dispatch: one threadgroup per (batch, head, q_block)
    grid_x = block_sparse_mask.num_q_blocks
    grid_y = num_heads
    grid_z = batch

    threadgroup_size = 128  # THREADS_PER_TG_ATT from shader

    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(grid_x, grid_y, grid_z),
        Metal.MTLSizeMake(threadgroup_size, 1, 1),
    )

    encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()

    return output


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    kv_seq_len: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    """
    Create a sliding window + causal attention mask.

    For use with standard attention implementations that don't have native
    sliding window support. The mask allows each position to attend only to
    positions within the window and before it (causal).

    Note: For optimal performance, use sliding_window_attention_metal() instead
    which uses a specialized kernel that doesn't materialize the full mask.

    Args:
        seq_len: Query sequence length
        window_size: Sliding window size
        kv_seq_len: Key/value sequence length (defaults to seq_len)
        device: Device to create the mask on

    Returns:
        Mask with -inf for masked positions, or None for single-token decode
    """
    kv_seq_len = kv_seq_len or seq_len

    # Single token decode - no masking needed
    if seq_len == 1:
        return None

    if device is None:
        device = _get_device()

    # Create position indices
    q_positions = torch.arange(seq_len, device=device).unsqueeze(1)
    k_positions = torch.arange(kv_seq_len, device=device).unsqueeze(0)

    # Causal mask: k_pos <= q_pos
    causal_mask = k_positions <= q_positions

    # Window mask: q_pos - k_pos < window_size
    window_mask = (q_positions - k_positions) < window_size

    # Combine: must satisfy both conditions
    combined_mask = causal_mask & window_mask

    # Convert to attention mask format (-inf for masked positions)
    mask = torch.where(
        combined_mask,
        torch.tensor(0.0, device=device),
        torch.tensor(float("-inf"), device=device),
    )

    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, kv_seq_len]


__all__ = [
    "MarlinAttention",
    "RoPE",
    "BlockSparseMask",
    "scaled_dot_product_attention_metal",
    "flash_attention_metal",
    "sliding_window_attention_metal",
    "block_sparse_attention_metal",
    "create_causal_mask",
    "create_causal_bitmask",
    "create_sliding_window_mask",
]
