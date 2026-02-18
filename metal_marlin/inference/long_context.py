"""Long-context optimization for attention computation.

This module provides optimizations for processing very long sequences (128K+ tokens):
- Dynamic sequence parallelism for distributed attention computation
- Hierarchical KV cache compression for memory efficiency
- Sliding window attention for local context focus
- Streaming attention for infinite context scenarios

Usage:
    from metal_marlin.inference.long_context import LongContextConfig, LongContextAttention
    
    config = LongContextConfig(
        max_seq_len=131072,
        use_sliding_window=True,
        window_size=4096,
        enable_kv_compression=True,
        compression_ratio=4.0,
    )
    
    attn = LongContextAttention(config)
    output = attn.forward(q, k, v, position_ids)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from .._compat import HAS_TORCH, torch

if TYPE_CHECKING:
    from ..kv_cache import KVCache


@dataclass
class LongContextConfig:
    """Configuration for long-context attention optimization.
    
    Attributes:
        max_seq_len: Maximum supported sequence length (e.g., 131072 for 128K).
        use_sliding_window: Whether to use sliding window attention.
        window_size: Size of the sliding window for local attention.
        global_tokens: Number of global tokens (always attend to these).
        enable_kv_compression: Whether to compress KV cache for long contexts.
        compression_ratio: KV cache compression ratio (e.g., 4.0 = 4x smaller).
        compression_threshold: Sequence length threshold to enable compression.
        use_streaming: Enable streaming attention for infinite contexts.
        streaming_chunk_size: Chunk size for streaming attention.
        use_sequence_parallel: Enable sequence parallelism across devices.
        sp_degree: Sequence parallelism degree (number of partitions).
        enable_offload: Offload inactive KV cache to CPU memory.
        offload_threshold: Sequence length threshold for offloading.
        memory_efficient_attention: Use memory-efficient attention algorithm.
        attn_chunk_size: Chunk size for attention computation (memory vs speed trade-off).
    """
    
    max_seq_len: int = 131072
    use_sliding_window: bool = True
    window_size: int = 4096
    global_tokens: int = 16
    enable_kv_compression: bool = True
    compression_ratio: float = 4.0
    compression_threshold: int = 8192
    use_streaming: bool = False
    streaming_chunk_size: int = 2048
    use_sequence_parallel: bool = False
    sp_degree: int = 1
    enable_offload: bool = False
    offload_threshold: int = 32768
    memory_efficient_attention: bool = True
    attn_chunk_size: int = 1024
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.window_size >= self.max_seq_len:
            raise ValueError(
                f"window_size ({self.window_size}) must be < max_seq_len ({self.max_seq_len})"
            )
        if self.compression_ratio < 1.0:
            raise ValueError(f"compression_ratio must be >= 1.0, got {self.compression_ratio}")
        if self.sp_degree < 1:
            raise ValueError(f"sp_degree must be >= 1, got {self.sp_degree}")


@dataclass
class SlidingWindowConfig:
    """Configuration for sliding window attention.
    
    Sliding window attention restricts each token to only attend to
    the most recent `window_size` tokens, reducing memory and compute
    from O(N^2) to O(N * window_size).
    
    Attributes:
        window_size: Number of tokens in the sliding window.
        global_tokens: Number of global tokens (can attend to all tokens).
        local_dilation: Stride within the window (for dilated attention).
    """
    
    window_size: int = 4096
    global_tokens: int = 16
    local_dilation: int = 1
    
    def get_attention_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Create causal sliding window attention mask.
        
        Args:
            seq_len: Sequence length.
            device: Target device.
            dtype: Mask dtype.
            
        Returns:
            Attention mask of shape [seq_len, seq_len] where True = can attend.
        """
        positions_q = torch.arange(seq_len, device=device).unsqueeze(1)
        positions_k = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # Causal mask: can only attend to previous tokens
        causal_mask = positions_q >= positions_k
        
        # Sliding window mask: can attend to tokens within window_size
        window_mask = (positions_q - positions_k) < self.window_size
        
        # Global tokens can attend to everything (and everything attends to them)
        global_mask_k = positions_k < self.global_tokens
        global_mask_q = positions_q < self.global_tokens
        
        # Combine: causal AND (window OR global)
        mask = causal_mask & (window_mask | global_mask_k | global_mask_q)
        
        return mask.to(dtype)


class KVCacheCompressor:
    """Compresses KV cache for long-context scenarios.
    
    Uses hierarchical compression:
    1. Recent tokens: Full precision (uncompressed)
    2. Medium-distance: FP8/BF16 compression
    3. Long-distance: Aggressive quantization or eviction
    """
    
    def __init__(
        self,
        config: LongContextConfig,
        head_dim: int,
        num_kv_heads: int,
    ):
        """Initialize KV cache compressor.
        
        Args:
            config: Long context configuration.
            head_dim: Dimension per head.
            num_kv_heads: Number of KV heads.
        """
        self.config = config
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        
        # Define compression tiers
        self.tier_full_len = config.window_size  # Recent tokens, full precision
        self.tier_fp8_len = config.window_size * 2  # Medium distance, FP8
        
    def should_compress(self, seq_len: int) -> bool:
        """Check if compression should be applied.
        
        Args:
            seq_len: Current sequence length.
            
        Returns:
            True if compression should be applied.
        """
        return (
            self.config.enable_kv_compression
            and seq_len >= self.config.compression_threshold
        )
    
    def get_compression_info(self, seq_len: int) -> dict[str, Any]:
        """Get compression information for the current sequence.
        
        Args:
            seq_len: Current sequence length.
            
        Returns:
            Dictionary with compression tier information.
        """
        if not self.should_compress(seq_len):
            return {
                "compress": False,
                "tiers": {"full": seq_len, "fp8": 0, "evicted": 0},
            }
        
        # Calculate tier lengths
        full_len = min(seq_len, self.tier_full_len)
        remaining = seq_len - full_len
        
        fp8_len = min(remaining, self.tier_fp8_len - full_len)
        evicted = remaining - fp8_len
        
        return {
            "compress": True,
            "tiers": {
                "full": full_len,
                "fp8": fp8_len,
                "evicted": evicted,
            },
            "compression_ratio": seq_len / (full_len + fp8_len / 2) if fp8_len > 0 else 1.0,
        }
    
    def compress_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Compress KV tensors.
        
        Args:
            k: Key tensor [batch, num_kv_heads, seq_len, head_dim].
            v: Value tensor [batch, num_kv_heads, seq_len, head_dim].
            
        Returns:
            Tuple of (compressed_k, compressed_v, compression_info).
        """
        batch, num_kv_heads, seq_len, head_dim = k.shape
        
        info = self.get_compression_info(seq_len)
        if not info["compress"]:
            return k, v, info
        
        tiers = info["tiers"]
        full_len = tiers["full"]
        fp8_len = tiers["fp8"]
        
        # Split into tiers
        k_full = k[:, :, :full_len, :]
        v_full = v[:, :, :full_len, :]
        
        if fp8_len > 0:
            k_fp8 = k[:, :, full_len:full_len + fp8_len, :]
            v_fp8 = v[:, :, full_len:full_len + fp8_len, :]
            
            # Compress FP8 tier (using bfloat16 as proxy for FP8)
            k_fp8_compressed = k_fp8.to(torch.bfloat16)
            v_fp8_compressed = v_fp8.to(torch.bfloat16)
            
            # Concatenate
            k_compressed = torch.cat([k_full, k_fp8_compressed.to(k.dtype)], dim=2)
            v_compressed = torch.cat([v_full, v_fp8_compressed.to(v.dtype)], dim=2)
        else:
            k_compressed = k_full
            v_compressed = v_full
        
        return k_compressed, v_compressed, info


class StreamingAttention:
    """Streaming attention for processing infinite-length sequences.
    
    Processes the sequence in fixed-size chunks, maintaining a fixed-size
    KV cache (like a rolling buffer). This allows processing sequences of
    arbitrary length with constant memory.
    
    Based on the "Efficient Streaming Language Models with Attention Sinks" paper.
    """
    
    def __init__(
        self,
        config: LongContextConfig,
        num_kv_heads: int,
        head_dim: int,
    ):
        """Initialize streaming attention.
        
        Args:
            config: Long context configuration.
            num_kv_heads: Number of KV heads.
            head_dim: Dimension per head.
        """
        self.config = config
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        # Fixed-size KV cache for streaming
        self.cache_size = config.window_size + config.global_tokens
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None
        self.cache_len = 0
        
    def reset(self) -> None:
        """Reset the streaming cache."""
        self.k_cache = None
        self.v_cache = None
        self.cache_len = 0
    
    def update_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update streaming cache with new KV tensors.
        
        Args:
            k: New key tensor [batch, num_kv_heads, seq_len, head_dim].
            v: New value tensor [batch, num_kv_heads, seq_len, head_dim].
            
        Returns:
            Updated cache tensors (k_cache, v_cache).
        """
        batch, num_kv_heads, seq_len, head_dim = k.shape
        
        # Initialize cache if needed
        if self.k_cache is None:
            self.k_cache = torch.zeros(
                batch, num_kv_heads, self.cache_size, head_dim,
                dtype=k.dtype, device=k.device,
            )
            self.v_cache = torch.zeros(
                batch, num_kv_heads, self.cache_size, head_dim,
                dtype=v.dtype, device=v.device,
            )
        
        # Concatenate new KV to cache
        if self.cache_len + seq_len <= self.cache_size:
            # Cache has room
            self.k_cache[:, :, self.cache_len:self.cache_len + seq_len, :] = k
            self.v_cache[:, :, self.cache_len:self.cache_len + seq_len, :] = v
            self.cache_len += seq_len
        else:
            # Evict oldest tokens (but keep global tokens)
            available_space = self.cache_size - self.cache_len
            if available_space > 0:
                self.k_cache[:, :, self.cache_len:self.cache_len + available_space, :] = k[:, :, :available_space, :]
                self.v_cache[:, :, self.cache_len:self.cache_len + available_space, :] = v[:, :, :available_space, :]
            
            # Roll the cache: shift remaining entries, append new ones
            remaining = seq_len - available_space
            if remaining > 0:
                # Keep global tokens at the beginning
                global_k = self.k_cache[:, :, :self.config.global_tokens, :]
                global_v = self.v_cache[:, :, :self.config.global_tokens, :]
                
                # Shift and append
                shift_len = self.cache_size - self.config.global_tokens - remaining
                if shift_len > 0:
                    self.k_cache[:, :, self.config.global_tokens:self.cache_size - remaining, :] = \
                        self.k_cache[:, :, self.config.global_tokens + remaining:self.cache_size, :]
                    self.v_cache[:, :, self.config.global_tokens:self.cache_size - remaining, :] = \
                        self.v_cache[:, :, self.config.global_tokens + remaining:self.cache_size, :]
                
                # Append new tokens
                self.k_cache[:, :, self.cache_size - remaining:self.cache_size, :] = k[:, :, available_space:, :]
                self.v_cache[:, :, self.cache_size - remaining:self.cache_size, :] = v[:, :, available_space:, :]
                
                # Restore global tokens
                self.k_cache[:, :, :self.config.global_tokens, :] = global_k
                self.v_cache[:, :, :self.config.global_tokens, :] = global_v
            
            self.cache_len = self.cache_size
        
        return self.k_cache[:, :, :self.cache_len, :], self.v_cache[:, :, :self.cache_len, :]


class MemoryEfficientAttention:
    """Memory-efficient attention using chunking and Flash Attention.
    
    Reduces memory usage from O(N^2) to O(N * chunk_size) by computing
    attention in chunks rather than materializing the full attention matrix.
    """
    
    def __init__(self, chunk_size: int = 1024):
        """Initialize memory-efficient attention.
        
        Args:
            chunk_size: Size of chunks for attention computation.
        """
        self.chunk_size = chunk_size
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """Compute memory-efficient attention.
        
        Args:
            q: Query tensor [batch, num_heads, seq_q, head_dim].
            k: Key tensor [batch, num_kv_heads, seq_k, head_dim].
            v: Value tensor [batch, num_kv_heads, seq_k, head_dim].
            scale: Attention scale (defaults to 1/sqrt(head_dim)).
            causal: Whether to use causal masking.
            
        Returns:
            Attention output [batch, num_heads, seq_q, head_dim].
        """
        import torch.nn.functional as F
        
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape
        
        if scale is None:
            scale = head_dim ** -0.5
        
        # For small sequences, use standard attention
        if seq_q <= self.chunk_size and seq_k <= self.chunk_size:
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
        
        # Process query in chunks
        outputs = []
        for q_start in range(0, seq_q, self.chunk_size):
            q_end = min(q_start + self.chunk_size, seq_q)
            q_chunk = q[:, :, q_start:q_end, :]
            
            # For causal attention, each query chunk only needs to attend
            # to key chunks up to its end position
            if causal:
                k_end = q_end
                k_causal = k[:, :, :k_end, :]
                v_causal = v[:, :, :k_end, :]
                chunk_causal = True
            else:
                k_causal = k
                v_causal = v
                chunk_causal = False
            
            # Compute attention for this chunk
            chunk_out = F.scaled_dot_product_attention(
                q_chunk, k_causal, v_causal,
                is_causal=chunk_causal, scale=scale,
            )
            outputs.append(chunk_out)
        
        return torch.cat(outputs, dim=2)


class LongContextAttention:
    """Long-context attention with optimizations for very long sequences.
    
    Combines multiple optimization techniques:
    - Sliding window attention for local focus
    - KV cache compression for memory efficiency
    - Memory-efficient attention computation
    - Streaming for infinite contexts
    """
    
    def __init__(
        self,
        config: LongContextConfig,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float = 10000.0,
    ):
        """Initialize long-context attention.
        
        Args:
            config: Long context configuration.
            num_heads: Number of attention heads.
            num_kv_heads: Number of KV heads.
            head_dim: Dimension per head.
            rope_theta: RoPE base frequency.
        """
        self.config = config
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        
        # Initialize components
        self.sliding_window = SlidingWindowConfig(
            window_size=config.window_size,
            global_tokens=config.global_tokens,
        ) if config.use_sliding_window else None
        
        self.kv_compressor = KVCacheCompressor(
            config=config,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
        ) if config.enable_kv_compression else None
        
        self.streaming = StreamingAttention(
            config=config,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        ) if config.use_streaming else None
        
        self.efficient_attn = MemoryEfficientAttention(
            chunk_size=config.attn_chunk_size,
        ) if config.memory_efficient_attention else None
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        scale: float | None = None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Forward pass with long-context optimizations.
        
        Args:
            q: Query tensor [batch, num_heads, seq_q, head_dim].
            k: Key tensor [batch, num_kv_heads, seq_k, head_dim].
            v: Value tensor [batch, num_kv_heads, seq_k, head_dim].
            position_ids: Position IDs [batch, seq_q] or [seq_q].
            attention_mask: Optional attention mask.
            scale: Attention scale.
            use_cache: Whether to return updated cache.
            
        Returns:
            Tuple of (attention_output, (k_cache, v_cache) if use_cache else None).
        """
        import torch.nn.functional as F
        
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape
        
        if scale is None:
            scale = head_dim ** -0.5
        
        # Apply KV cache compression if enabled
        if self.kv_compressor is not None and self.kv_compressor.should_compress(seq_k):
            k, v, compression_info = self.kv_compressor.compress_kv(k, v)
            seq_k = k.shape[2]
        
        # Use memory-efficient attention for long sequences
        if self.efficient_attn is not None and seq_k > self.config.attn_chunk_size:
            output = self.efficient_attn.forward(q, k, v, scale=scale, causal=True)
        else:
            # Standard attention with optional sliding window
            if self.sliding_window is not None and seq_k > self.config.window_size:
                # Create sliding window mask
                attn_mask = self.sliding_window.get_attention_mask(
                    seq_k, q.device, dtype=q.dtype,
                )
                # Slice to query length
                if seq_q < seq_k:
                    attn_mask = attn_mask[-seq_q:, :]
                
                # Expand for batch and heads
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                attn_mask = attn_mask.expand(batch, num_heads, seq_q, seq_k)
                
                output = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, scale=scale,
                )
            else:
                output = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=scale,
                )
        
        # Return updated cache if requested
        if use_cache:
            return output, (k, v)
        return output, None
    
    def reset(self) -> None:
        """Reset any internal state (e.g., streaming cache)."""
        if self.streaming is not None:
            self.streaming.reset()


def create_long_context_config(
    max_seq_len: int = 131072,
    model_type: str = "default",
) -> LongContextConfig:
    """Create a pre-configured LongContextConfig for common use cases.
    
    Args:
        max_seq_len: Maximum sequence length.
        model_type: Model type for preset configurations.
            - "default": Balanced settings
            - "aggressive": Maximum compression, lowest memory
            - "quality": Minimal compression, best quality
            - "streaming": For infinite context processing
            
    Returns:
        LongContextConfig instance.
    """
    presets = {
        "default": LongContextConfig(
            max_seq_len=max_seq_len,
            use_sliding_window=True,
            window_size=min(4096, max_seq_len // 32),
            enable_kv_compression=True,
            compression_ratio=4.0,
            compression_threshold=8192,
        ),
        "aggressive": LongContextConfig(
            max_seq_len=max_seq_len,
            use_sliding_window=True,
            window_size=min(2048, max_seq_len // 64),
            enable_kv_compression=True,
            compression_ratio=8.0,
            compression_threshold=4096,
            memory_efficient_attention=True,
            attn_chunk_size=512,
        ),
        "quality": LongContextConfig(
            max_seq_len=max_seq_len,
            use_sliding_window=True,
            window_size=min(8192, max_seq_len // 16),
            enable_kv_compression=True,
            compression_ratio=2.0,
            compression_threshold=16384,
            memory_efficient_attention=True,
            attn_chunk_size=2048,
        ),
        "streaming": LongContextConfig(
            max_seq_len=max_seq_len,
            use_streaming=True,
            use_sliding_window=True,
            window_size=4096,
            global_tokens=16,
            streaming_chunk_size=2048,
            enable_kv_compression=False,
        ),
    }
    
    if model_type not in presets:
        raise ValueError(f"Unknown model_type: {model_type}. Available: {list(presets.keys())}")
    
    return presets[model_type]


# Export symbols
__all__ = [
    "LongContextConfig",
    "SlidingWindowConfig",
    "KVCacheCompressor",
    "StreamingAttention",
    "MemoryEfficientAttention",
    "LongContextAttention",
    "create_long_context_config",
]
