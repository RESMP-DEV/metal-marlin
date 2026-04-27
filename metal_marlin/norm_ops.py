"""Norm operations (LayerNorm, RMSNorm) using C++ extension.

This module provides Python bindings for the C++ norm operations,
which implement optimized CPU-based LayerNorm and RMSNorm.

Example:
    >>> from metal_marlin.norm_ops import LayerNormOp, RMSNormOp
    >>> import numpy as np
    >>> 
    >>> # LayerNorm
    >>> config = LayerNormConfig(num_tokens=2, hidden_dim=4)
    >>> op = LayerNormOp(config)
    >>> input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    >>> gamma = np.ones(4, dtype=np.float32)
    >>> beta = np.zeros(4, dtype=np.float32)
    >>> result = op.forward(input_data.tolist(), gamma.tolist(), beta.tolist())
    >>> 
    >>> # RMSNorm
    >>> rms_config = RMSNormConfig(num_tokens=2, hidden_dim=4)
    >>> rms_op = RMSNormOp(rms_config)
    >>> rms_result = rms_op.forward(input_data.tolist(), gamma.tolist())
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Try to import from C++ extension
try:
    from metal_marlin._cpp_ext import (
        LayerNormConfig,
        NormResult,
        RMSNormConfig,
    )
    from metal_marlin._cpp_ext import (
        LayerNormOp as _LayerNormOp,
    )
    from metal_marlin._cpp_ext import (
        RMSNormOp as _RMSNormOp,
    )
    from metal_marlin._cpp_ext import norm_utils as _norm_utils
    _HAS_CPP_EXT = True
except ImportError:
    _HAS_CPP_EXT = False



logger = logging.getLogger(__name__)

class LayerNormOp:
    """Python wrapper for C++ LayerNorm operation.
    
    Layer Normalization: output = (x - mean) / sqrt(var + eps) * gamma + beta
    
    Args:
        num_tokens: Number of tokens (batch_size * seq_len)
        hidden_dim: Hidden dimension size
        eps: Small constant for numerical stability (default: 1e-5)
    """
    
    def __init__(self, num_tokens: int, hidden_dim: int, eps: float = 1e-5):
        logger.debug("initializing %s with num_tokens=%s, hidden_dim=%s, eps=%s", type(self).__name__, num_tokens, hidden_dim, eps)
        if not _HAS_CPP_EXT:
            raise ImportError(
                "C++ extension not available. "
                "Build with: cd contrib/metal_marlin && uv run python setup.py build_ext --inplace"
            )
        
        config = LayerNormConfig()
        config.num_tokens = num_tokens
        config.hidden_dim = hidden_dim
        config.eps = eps
        
        self._op = _LayerNormOp(config)
        self._num_tokens = num_tokens
        self._hidden_dim = hidden_dim
    
    def forward(
        self,
        input: list[float],
        gamma: list[float],
        beta: list[float] | None = None,
    ) -> NormResult:
        """Apply LayerNorm.
        
        Args:
            input: Flattened input array [num_tokens * hidden_dim]
            gamma: Scale weights [hidden_dim]
            beta: Bias weights [hidden_dim], or None for no bias
            
        Returns:
            NormResult with normalized output
        """
        logger.debug("forward: input shape=%s dtype=%s", input.shape if hasattr(input, "shape") else type(input).__name__, input.dtype if hasattr(input, "dtype") else "N/A")
        if beta is None:
            beta = []
        return self._op.forward(input, gamma, beta)
    
    @property
    def num_tokens(self) -> int:
        logger.debug("num_tokens called")
        return self._num_tokens
    
    @property
    def hidden_dim(self) -> int:
        logger.debug("hidden_dim called")
        return self._hidden_dim


class RMSNormOp:
    """Python wrapper for C++ RMSNorm operation.
    
    RMS Normalization: output = x / sqrt(mean(x^2) + eps) * gamma
    
    Args:
        num_tokens: Number of tokens (batch_size * seq_len)
        hidden_dim: Hidden dimension size
        eps: Small constant for numerical stability (default: 1e-6)
    """
    
    def __init__(self, num_tokens: int, hidden_dim: int, eps: float = 1e-6):
        logger.debug("initializing %s with num_tokens=%s, hidden_dim=%s, eps=%s", type(self).__name__, num_tokens, hidden_dim, eps)
        if not _HAS_CPP_EXT:
            raise ImportError(
                "C++ extension not available. "
                "Build with: cd contrib/metal_marlin && uv run python setup.py build_ext --inplace"
            )
        
        config = RMSNormConfig()
        config.num_tokens = num_tokens
        config.hidden_dim = hidden_dim
        config.eps = eps
        
        self._op = _RMSNormOp(config)
        self._num_tokens = num_tokens
        self._hidden_dim = hidden_dim
    
    def forward(
        self,
        input: list[float],
        gamma: list[float],
    ) -> NormResult:
        """Apply RMSNorm.
        
        Args:
            input: Flattened input array [num_tokens * hidden_dim]
            gamma: Scale weights [hidden_dim]
            
        Returns:
            NormResult with normalized output
        """
        logger.debug("forward: input shape=%s dtype=%s", input.shape if hasattr(input, "shape") else type(input).__name__, input.dtype if hasattr(input, "dtype") else "N/A")
        return self._op.forward(input, gamma)
    
    def forward_fused(
        self,
        input: list[float],
        residual: list[float],
        gamma: list[float],
    ) -> NormResult:
        """Apply fused residual add + RMSNorm.
        
        Computes: residual_out = input + residual, then RMSNorm(residual_out)
        
        Args:
            input: Input tensor [num_tokens * hidden_dim]
            residual: Residual tensor [num_tokens * hidden_dim]
            gamma: Scale weights [hidden_dim]
            
        Returns:
            NormResult with normalized output and residual output
        """
        logger.debug("forward_fused called with input=%s, residual=%s, gamma=%s", input, residual, gamma)
        return self._op.forward_fused(input, residual, gamma)
    
    @property
    def num_tokens(self) -> int:
        logger.debug("num_tokens called")
        return self._num_tokens
    
    @property
    def hidden_dim(self) -> int:
        logger.debug("hidden_dim called")
        return self._hidden_dim


def compute_mean(data: list[float]) -> float:
    """Compute mean of a float array."""
    logger.debug("compute_mean called with data=%s", data)
    if not _HAS_CPP_EXT:
        raise ImportError("C++ extension not available")
    return _norm_utils.compute_mean(data, len(data))


def compute_variance(data: list[float], mean: float) -> float:
    """Compute variance of a float array."""
    logger.debug("compute_variance called with data=%s, mean=%s", data, mean)
    if not _HAS_CPP_EXT:
        raise ImportError("C++ extension not available")
    return _norm_utils.compute_variance(data, len(data), mean)


def compute_rms(data: list[float]) -> float:
    """Compute RMS (root mean square) of a float array."""
    logger.debug("compute_rms called with data=%s", data)
    if not _HAS_CPP_EXT:
        raise ImportError("C++ extension not available")
    return _norm_utils.compute_rms(data, len(data))


def is_hidden_dim_supported(hidden_dim: int) -> bool:
    """Check if hidden dimension is supported by optimized kernels."""
    logger.debug("is_hidden_dim_supported called with hidden_dim=%s", hidden_dim)
    if not _HAS_CPP_EXT:
        raise ImportError("C++ extension not available")
    return _norm_utils.is_hidden_dim_supported(hidden_dim)


def get_recommended_threadgroup_size(hidden_dim: int) -> int:
    """Get recommended threadgroup size for given hidden dimension."""
    logger.debug("get_recommended_threadgroup_size called with hidden_dim=%s", hidden_dim)
    if not _HAS_CPP_EXT:
        raise ImportError("C++ extension not available")
    return _norm_utils.get_recommended_threadgroup_size(hidden_dim)


__all__ = [
    "LayerNormOp",
    "RMSNormOp",
    "NormResult",
    "compute_mean",
    "compute_variance",
    "compute_rms",
    "is_hidden_dim_supported",
    "get_recommended_threadgroup_size",
]
