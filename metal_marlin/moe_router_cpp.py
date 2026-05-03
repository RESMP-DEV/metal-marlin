"""C++ MoE Router for high-performance MoE token routing.

This module provides a Python wrapper around the C++ FastRouterDispatcher,
which handles the computationally intensive task of routing tokens to experts
in a Mixture of Experts model.

The C++ implementation is optimized for performance using SIMD instructions
and a cache for frequently co-occurring expert pairs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from metal_marlin._compat import HAS_CPP_EXT

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Import C++ extension if available
cpp_ext: Any = None
FastRouterDispatcher: Any = None
RouterBatchOutput: Any = None

if HAS_CPP_EXT:
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    
    try:
        import metal_marlin._cpp_ext as cpp_ext
        from metal_marlin._cpp_ext import FastRouterDispatcher, RouterBatchOutput
    except ImportError:
        cpp_ext = None
        FastRouterDispatcher = None
        RouterBatchOutput = None


logger = logging.getLogger(__name__)

def is_available() -> bool:
    """Check if C++ MoE router is available."""
    logger.debug("is_available called")
    return FastRouterDispatcher is not None

class MoERouterCpp:
    """
    High-performance C++ MoE router.

    This class wraps the C++ FastRouterDispatcher to provide an efficient
    implementation of the MoE routing mechanism.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        top_k: int,
        max_batch_tokens: int = 128,
        hot_pair_cache_capacity: int = 256,
        hot_pair_threshold: int = 8,
    ):
        logger.debug("initializing %s with num_experts=%s, hidden_dim=%s, top_k=%s, max_batch_tokens=%s, hot_pair_cache_capacity=%s", type(self).__name__, num_experts, hidden_dim, top_k, max_batch_tokens, hot_pair_cache_capacity)
        if not is_available():
            raise RuntimeError(
                "C++ MoE router is not available. "
                "Build the extension with: cd contrib/metal_marlin && uv pip install -e ."
            )

        self._dispatcher = FastRouterDispatcher(
            num_experts,
            hidden_dim,
            top_k,
            max_batch_tokens,
            hot_pair_cache_capacity,
            hot_pair_threshold,
        )

    @property
    def num_experts(self) -> int:
        logger.debug("num_experts called")
        return self._dispatcher.num_experts()

    @property
    def hidden_dim(self) -> int:
        logger.debug("hidden_dim called")
        return self._dispatcher.hidden_dim()

    @property
    def top_k(self) -> int:
        logger.debug("top_k called")
        return self._dispatcher.top_k()

    @property
    def hot_pair_count(self) -> int:
        logger.debug("hot_pair_count called")
        return self._dispatcher.hot_pair_count()

    def reset_hot_pair_cache(self):
        logger.debug("reset_hot_pair_cache called")
        self._dispatcher.reset_hot_pair_cache()

    def route_batch(
        self,
        token_activations: NDArray,
        router_weights: NDArray[np.float32],
        router_bias: NDArray[np.float32] | None = None,
    ) -> RouterBatchOutput:
        """
        Perform routing for a batch of tokens.

        Args:
            token_activations: A numpy array of shape (num_tokens, hidden_dim) with dtype=bfloat16 (represented as uint16).
            router_weights: A numpy array of shape (num_experts, hidden_dim) with dtype=float32.
            router_bias: An optional numpy array of shape (num_experts,) with dtype=float32.

        Returns:
            A RouterBatchOutput object containing the routing information.
        """
        logger.debug("route_batch called with token_activations=%s, router_weights=%s, router_bias=%s", token_activations, router_weights, router_bias)
        if token_activations.dtype.name != 'uint16':
             raise TypeError(f"token_activations must be uint16 (representing bfloat16), but got {token_activations.dtype}")

        num_tokens = token_activations.shape[0]
        
        activations_bytes = token_activations.tobytes()

        return self._dispatcher.route_batch(
            activations_bytes,
            num_tokens,
            np.ascontiguousarray(router_weights, dtype=np.float32),
            np.ascontiguousarray(router_bias, dtype=np.float32) if router_bias is not None else None,
        )

__all__ = ["MoERouterCpp", "is_available"]
