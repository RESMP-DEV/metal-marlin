"""C++ Expert Manager for high-performance MoE token dispatch.

This module provides a Python wrapper around the C++ TokenGroupManager,
which handles token grouping and dispatch for Mixture of Experts (MoE)
operations with minimal overhead.

The C++ implementation is significantly faster than pure Python for:
- Token-to-expert grouping (via sorting)
- Activation gathering/scattering
- Load balancing diagnostics

Example:
    >>> from metal_marlin.expert_manager_cpp import ExpertManagerCpp
    >>> manager = ExpertManagerCpp(num_experts=8, max_tokens=1024)
    >>> 
    >>> # Group tokens by expert assignment
    >>> expert_ids = [[0, 3], [1, 5], [2, 7]]  # [batch, top_k]
    >>> info = manager.group_tokens(expert_ids, batch_size=3, top_k=2)
    >>> 
    >>> # Check expert loads
    >>> loads = manager.compute_expert_loads(info)
    >>> print(f"Active experts: {info.active_expert_count()}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from metal_marlin._compat import HAS_CPP_EXT

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Import C++ extension if available
cpp_ext: Any = None
TokenGroupManager: Any = None
DispatchInfo: Any = None
TokenGroup: Any = None
ExpertBufferPool: Any = None

if HAS_CPP_EXT:
    try:
        # Try to import torch first (required for _cpp_ext due to torch symbols)
        try:
            import torch  # noqa: F401
        except ImportError:
            pass  # torch not available, _cpp_ext might still work
        
        import metal_marlin._cpp_ext as cpp_ext
        from metal_marlin._cpp_ext import TokenGroupManager, DispatchInfo, TokenGroup
        
        # Import ExpertBufferPool for expert weight management
        try:
            from metal_marlin._cpp_ext import ExpertBufferPool
        except ImportError:
            ExpertBufferPool = None
    except ImportError:
        cpp_ext = None
        TokenGroupManager = None
        DispatchInfo = None
        TokenGroup = None
        ExpertBufferPool = None


def is_available() -> bool:
    """Check if C++ expert manager is available.
    
    Returns:
        True if the C++ extension is built and available.
    """
    return TokenGroupManager is not None


def is_expert_buffer_pool_available() -> bool:
    """Check if C++ ExpertBufferPool is available.
    
    Returns:
        True if the ExpertBufferPool is available in the C++ extension.
    """
    return ExpertBufferPool is not None


def initialize_expert_buffer_pool(device_ptr: int, heap_size: int = 1024 * 1024 * 1024) -> bool:
    """Initialize the global ExpertBufferPool singleton.
    
    This should be called once during model loading to set up the buffer pool
    for expert weight management. The pool provides PINNED priority buffers
    that are never evicted, making them ideal for expert weights.
    
    Args:
        device_ptr: Integer pointer to the MTLDevice (from device.raw())
        heap_size: Size of the heap in bytes (default 1GB)
        
    Returns:
        True if initialization succeeded, False otherwise.
        
    Example:
        >>> from metal_marlin.expert_manager_cpp import initialize_expert_buffer_pool
        >>> from metal_marlin._cpp_ext import MetalDevice
        >>> device = MetalDevice.default_device()
        >>> initialize_expert_buffer_pool(device.raw())
        True
    """
    if ExpertBufferPool is None:
        return False
    
    try:
        pool = ExpertBufferPool.instance()
        if not pool.is_initialized():
            pool.initialize(device_ptr, heap_size)
        return pool.is_initialized()
    except Exception:
        return False


def get_expert_buffer_pool() -> Any:
    """Get the ExpertBufferPool singleton instance.
    
    Returns:
        The ExpertBufferPool singleton, or None if not available.
        
    Raises:
        RuntimeError: If the pool is not initialized.
    """
    if ExpertBufferPool is None:
        raise RuntimeError("ExpertBufferPool not available. Build C++ extension with: uv pip install -e .")
    
    pool = ExpertBufferPool.instance()
    if not pool.is_initialized():
        raise RuntimeError("ExpertBufferPool not initialized. Call initialize_expert_buffer_pool() first.")
    
    return pool


def allocate_expert_weight_buffer(size: int) -> Any:
    """Allocate a pinned buffer for expert weights from the pool.
    
    Args:
        size: Size in bytes to allocate.
        
    Returns:
        BufferHandle object, or None if allocation failed.
        
    Example:
        >>> handle = allocate_expert_weight_buffer(1024 * 1024)  # 1MB
        >>> if handle:
        ...     # Use handle.buffer() to get MTLBuffer
        ...     pass
    """
    pool = get_expert_buffer_pool()
    return pool.allocate_weight(size)


def clear_expert_buffer_pool() -> None:
    """Clear all buffers from the ExpertBufferPool.
    
    This releases all pooled memory back to the system.
    """
    if ExpertBufferPool is not None:
        pool = ExpertBufferPool.instance()
        if pool.is_initialized():
            pool.clear()


class ExpertManagerCpp:
    """High-performance C++ expert manager for MoE dispatch.
    
    This class wraps the C++ TokenGroupManager to provide efficient
    token grouping and dispatch operations for MoE layers.
    
    The C++ implementation avoids Python GIL contention and uses
    optimized sorting for token grouping.
    
    Args:
        num_experts: Total number of experts in the MoE layer
        max_tokens: Maximum number of tokens to support (for pre-allocation)
        max_top_k: Maximum top-k value (for pre-allocation)
        
    Raises:
        RuntimeError: If C++ extension is not available.
        ValueError: If num_experts is not positive.
        
    Example:
        >>> manager = ExpertManagerCpp(num_experts=8, max_tokens=1024, max_top_k=2)
        >>> 
        >>> # Expert assignments [batch, top_k]
        >>> expert_ids = np.array([[0, 3], [1, 5], [2, 7]], dtype=np.int32)
        >>> info = manager.group_tokens(expert_ids.flatten(), batch_size=3, top_k=2)
        >>> 
        >>> # Get load statistics
        >>> loads = manager.compute_expert_loads(info)
        >>> imbalance = manager.compute_load_imbalance(info)
    """
    
    def __init__(
        self,
        num_experts: int,
        max_tokens: int = 8192,
        max_top_k: int = 8,
    ) -> None:
        if TokenGroupManager is None:
            raise RuntimeError(
                "C++ extension not available. "
                "Build with: cd contrib/metal_marlin && uv pip install -e ."
            )
        
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        
        self._num_experts = num_experts
        self._max_tokens = max_tokens
        self._max_top_k = max_top_k
        
        # Create the underlying C++ manager
        self._manager = TokenGroupManager(num_experts, max_tokens, max_top_k)
    
    @property
    def num_experts(self) -> int:
        """Number of experts."""
        return self._num_experts
    
    @property
    def max_tokens(self) -> int:
        """Maximum number of tokens supported."""
        return self._max_tokens
    
    @property
    def max_top_k(self) -> int:
        """Maximum top-k value supported."""
        return self._max_top_k
    
    def group_tokens(
        self,
        expert_ids: NDArray[np.int32],
        batch_size: int,
        top_k: int,
    ) -> Any:
        """Group tokens by their assigned expert.
        
        Given expert assignments [batch, top_k], produces a DispatchInfo
        that reorders tokens so all tokens assigned to the same expert
        are contiguous.
        
        Args:
            expert_ids: Flattened [batch, top_k] expert assignments
            batch_size: Number of tokens in batch
            top_k: Number of experts per token
            
        Returns:
            DispatchInfo object with grouping information
            
        Raises:
            ValueError: If inputs are invalid
            
        Example:
            >>> expert_ids = np.array([0, 3, 1, 5, 2, 7], dtype=np.int32)
            >>> info = manager.group_tokens(expert_ids, batch_size=3, top_k=2)
            >>> print(f"Total assignments: {info.total_assignments()}")
        """
        if expert_ids.dtype != np.int32:
            expert_ids = expert_ids.astype(np.int32)
        
        # Ensure contiguous array
        expert_ids = np.ascontiguousarray(expert_ids)
        
        return self._manager.group_tokens(expert_ids, batch_size, top_k)
    
    def group_tokens_2d(
        self,
        expert_ids: NDArray[np.int32],
        batch_size: int,
        top_k: int,
    ) -> Any:
        """Group tokens from 2D array layout.
        
        Same as group_tokens but accepts a 2D array directly.
        
        Args:
            expert_ids: 2D array of shape [batch, top_k]
            batch_size: Number of tokens (rows)
            top_k: Number of experts per token (columns)
            
        Returns:
            DispatchInfo object with grouping information
        """
        if expert_ids.dtype != np.int32:
            expert_ids = expert_ids.astype(np.int32)
        
        # Flatten to 1D
        expert_ids = np.ascontiguousarray(expert_ids.ravel())
        
        return self._manager.group_tokens(expert_ids, batch_size, top_k)
    
    def compute_expert_loads(self, info: Any) -> list[int]:
        """Compute token count per expert.
        
        Args:
            info: DispatchInfo from group_tokens
            
        Returns:
            List of token counts per expert
        """
        return self._manager.compute_expert_loads(info)
    
    def compute_load_imbalance(self, info: Any) -> float:
        """Compute load imbalance metric.
        
        Returns the coefficient of variation (std/mean) of non-zero
        expert loads. Lower values indicate better balance.
        
        Args:
            info: DispatchInfo from group_tokens
            
        Returns:
            Load imbalance ratio (0.0 = perfectly balanced)
        """
        return self._manager.compute_load_imbalance(info)
    
    def is_load_balanced(self, info: Any, threshold: float = 2.0) -> bool:
        """Check if dispatch has acceptable load balance.
        
        Args:
            info: DispatchInfo from group_tokens
            threshold: Maximum acceptable imbalance (default 2.0)
            
        Returns:
            True if load is balanced within threshold
        """
        return self._manager.is_load_balanced(info, threshold)
    
    def get_token_group(self, info: Any, expert_id: int) -> Any:
        """Get TokenGroup for a specific expert.
        
        Args:
            info: DispatchInfo from group_tokens
            expert_id: Expert index
            
        Returns:
            TokenGroup view into the dispatch info
        """
        return self._manager.get_token_group(expert_id, info)


class CppDispatchInfo:
    """Python wrapper for C++ DispatchInfo.
    
    This class provides convenient access to dispatch information
    with numpy array conversion where needed.
    """
    
    def __init__(self, info: Any) -> None:
        """Wrap a C++ DispatchInfo object.
        
        Args:
            info: C++ DispatchInfo object
        """
        self._info = info
    
    @property
    def sorted_token_indices(self) -> NDArray[np.int32]:
        """[total_assignments] indices into original batch."""
        return np.array(self._info.sorted_token_indices, dtype=np.int32)
    
    @property
    def sorted_expert_indices(self) -> NDArray[np.int32]:
        """[total_assignments] which expert slot (0 to top_k-1)."""
        return np.array(self._info.sorted_expert_indices, dtype=np.int32)
    
    @property
    def expert_offsets(self) -> NDArray[np.int32]:
        """[num_experts + 1] start index for each expert's assignments."""
        return np.array(self._info.expert_offsets, dtype=np.int32)
    
    @property
    def inverse_indices(self) -> NDArray[np.int32]:
        """[total_assignments] indices to scatter expert outputs back."""
        return np.array(self._info.inverse_indices, dtype=np.int32)
    
    @property
    def num_tokens(self) -> int:
        """Number of tokens."""
        return self._info.num_tokens
    
    @property
    def top_k(self) -> int:
        """Top-k value."""
        return self._info.top_k
    
    @property
    def num_experts(self) -> int:
        """Number of experts."""
        return self._info.num_experts
    
    def total_assignments(self) -> int:
        """Total number of token-expert assignments."""
        return self._info.total_assignments()
    
    def expert_batch_size(self, expert_id: int) -> int:
        """Get number of tokens assigned to a specific expert."""
        return self._info.expert_batch_size(expert_id)
    
    def is_expert_active(self, expert_id: int) -> bool:
        """Check if an expert has any assigned tokens."""
        return self._info.is_expert_active(expert_id)
    
    def active_expert_count(self) -> int:
        """Get number of active experts."""
        return self._info.active_expert_count()


def create_expert_manager(
    num_experts: int,
    max_tokens: int = 8192,
    max_top_k: int = 8,
) -> ExpertManagerCpp:
    """Create a C++ expert manager.
    
    Factory function that creates an ExpertManagerCpp instance.
    Falls back to Python implementation if C++ extension is not available.
    
    Args:
        num_experts: Total number of experts
        max_tokens: Maximum tokens to support
        max_top_k: Maximum top-k value
        
    Returns:
        ExpertManagerCpp instance
        
    Raises:
        RuntimeError: If C++ extension is not available.
    """
    if not is_available():
        raise RuntimeError(
            "C++ expert manager not available. "
            "Build the extension with: cd contrib/metal_marlin && uv pip install -e ."
        )
    
    return ExpertManagerCpp(num_experts, max_tokens, max_top_k)


# Export public API
__all__ = [
    "ExpertManagerCpp",
    "CppDispatchInfo",
    "create_expert_manager",
    "is_available",
    # ExpertBufferPool exports
    "ExpertBufferPool",
    "is_expert_buffer_pool_available",
    "initialize_expert_buffer_pool",
    "get_expert_buffer_pool",
    "allocate_expert_weight_buffer",
    "clear_expert_buffer_pool",
]
