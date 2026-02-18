"""C++ MLA Attention for high-performance Metal operations.

This module provides Python bindings for the C++ MLA attention operations,
which implement optimized Metal-based projection kernels for FP4 quantized
Multi-head Latent Attention.

The C++ implementation is significantly faster than pure Python/PyTorch for:
- MLA projections with FP4 quantized weights (prefill phase)
- Decode projections with FP4 quantized weights (single token)
- Fused KV projections combining kv_a and kv_b

Example:
    >>> from metal_marlin.mla_attention_cpp import MLAAttentionCpp, is_available
    >>> 
    >>> if is_available():
    ...     # Create C++ MLA attention handler
    ...     mla = MLAAttentionCpp()
    ...     
    ...     # Run projection (buffers must be pre-allocated)
    ...     mla.mla_proj_fp4(ctx, A, B_packed, scales, C, M, N, K, group_size)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# Import C++ extension if available
cpp_ext: Any = None
MetalContext: Any = None
mla_proj_fp4: Any = None
mla_decode_proj_fp4: Any = None
mla_fused_kv_proj_fp4: Any = None

_has_cpp_ext = False

try:
    import metal_marlin._cpp_ext as cpp_ext
    from metal_marlin._cpp_ext import (
        MetalContext,
        mla_proj_fp4,
        mla_decode_proj_fp4,
        mla_fused_kv_proj_fp4,
    )
    _has_cpp_ext = True
except ImportError:
    cpp_ext = None
    MetalContext = None
    mla_proj_fp4 = None
    mla_decode_proj_fp4 = None
    mla_fused_kv_proj_fp4 = None


def is_available() -> bool:
    """Check if C++ MLA attention is available.
    
    Returns:
        True if the C++ extension is built and available.
    """
    return _has_cpp_ext and mla_proj_fp4 is not None


def require_cpp_ext() -> None:
    """Raise RuntimeError if C++ extension is not available."""
    if not is_available():
        raise RuntimeError(
            "C++ MLA attention not available. "
            "Build the extension with: cd contrib/metal_marlin && uv pip install -e ."
        )


class MLAAttentionCpp:
    """High-performance C++ MLA attention operations.
    
    This class wraps the C++ MLA attention kernels for efficient
    FP4 quantized projections on Metal.
    
    The C++ implementation provides:
    - mla_proj_fp4: General projection for prefill (batch tokens)
    - mla_decode_proj_fp4: Optimized single-token projection for decode
    - mla_fused_kv_proj_fp4: Fused kv_a + kv_b projection
    
    Example:
        >>> from metal_marlin.mla_attention_cpp import MLAAttentionCpp
        >>> from metal_marlin._cpp_ext import MetalContext
        >>> 
        >>> mla = MLAAttentionCpp()
        >>> ctx = MetalContext()
        >>> 
        >>> # Run projection (data as bytes)
        >>> mla.mla_proj_fp4(ctx, A_bytes, B_bytes, S_bytes, C_bytes, 
        ...                    M=16, N=512, K=4096, group_size=128)
    """
    
    def __init__(self) -> None:
        """Initialize C++ MLA attention wrapper.
        
        Raises:
            RuntimeError: If C++ extension is not available.
        """
        require_cpp_ext()
    
    def mla_proj_fp4(
        self,
        ctx: Any,
        A: bytes,
        B_packed: bytes,
        scales: bytes,
        C: bytes,
        M: int,
        N: int,
        K: int,
        group_size: int,
        wait: bool = True,
    ) -> None:
        """MLA projection with FP4 quantized weights (prefill phase).
        
        Computes: C = A @ B^T where B is FP4 quantized
        
        Args:
            ctx: MetalContext instance
            A: Input matrix as bytes [M, K] float16
            B_packed: Packed FP4 weight matrix [K/2, N] uint8 (2x4bit values per byte)
            scales: Scale factors for dequantization
            C: Output buffer as bytes [M, N] float16
            M: Batch size (number of tokens)
            N: Output dimension
            K: Input dimension
            group_size: Quantization group size
            wait: Whether to wait for kernel completion
            
        Raises:
            RuntimeError: If C++ extension is not available.
        """
        require_cpp_ext()
        mla_proj_fp4(ctx, A, B_packed, scales, C, M, N, K, group_size, wait)
    
    def mla_decode_proj_fp4(
        self,
        ctx: Any,
        x: bytes,
        W_packed: bytes,
        scales: bytes,
        out: bytes,
        K: int,
        N: int,
        group_size: int,
        wait: bool = True,
    ) -> None:
        """MLA decode projection with FP4 quantized weights (single token).
        
        Optimized for single-token decode phase:
        out = x @ W^T where W is FP4 quantized
        
        Args:
            ctx: MetalContext instance
            x: Input vector as bytes [K] float16
            W_packed: Packed FP4 weight matrix [K/2, N] uint8
            scales: Scale factors for dequantization
            out: Output buffer as bytes [N] float16
            K: Input dimension
            N: Output dimension
            group_size: Quantization group size
            wait: Whether to wait for kernel completion
            
        Raises:
            RuntimeError: If C++ extension is not available.
        """
        require_cpp_ext()
        mla_decode_proj_fp4(ctx, x, W_packed, scales, out, K, N, group_size, wait)
    
    def mla_fused_kv_proj_fp4(
        self,
        ctx: Any,
        hidden: bytes,
        W_a_packed: bytes,
        scales_a: bytes,
        W_b_packed: bytes,
        scales_b: bytes,
        out: bytes,
        M: int,
        K_hidden: int,
        K_latent: int,
        N_out: int,
        group_size_a: int,
        group_size_b: int,
        wait: bool = True,
    ) -> None:
        """Fused MLA kv_a + kv_b projection with FP4 quantized weights.
        
        Computes fused: out = (hidden @ W_a^T) @ W_b^T
        where both W_a and W_b are FP4 quantized.
        
        This avoids intermediate buffer allocation between kv_a and kv_b.
        
        Args:
            ctx: MetalContext instance
            hidden: Input hidden states as bytes [M, K_hidden] float16
            W_a_packed: Packed FP4 weights for kv_a [K_hidden/2, K_latent] uint8
            scales_a: Scale factors for W_a
            W_b_packed: Packed FP4 weights for kv_b [K_latent/2, N_out] uint8
            scales_b: Scale factors for W_b
            out: Output buffer as bytes [M, N_out] float16
            M: Batch size
            K_hidden: Hidden dimension
            K_latent: Latent dimension (compressed)
            N_out: Output dimension
            group_size_a: Quantization group size for W_a
            group_size_b: Quantization group size for W_b
            wait: Whether to wait for kernel completion
            
        Raises:
            RuntimeError: If C++ extension is not available.
        """
        require_cpp_ext()
        mla_fused_kv_proj_fp4(
            ctx, hidden, W_a_packed, scales_a, W_b_packed, scales_b, out,
            M, K_hidden, K_latent, N_out, group_size_a, group_size_b, wait
        )


# Export public API
__all__ = [
    "MLAAttentionCpp",
    "is_available",
    "require_cpp_ext",
    # Also export the raw functions for advanced users
    "mla_proj_fp4",
    "mla_decode_proj_fp4",
    "mla_fused_kv_proj_fp4",
    "MetalContext",
]
