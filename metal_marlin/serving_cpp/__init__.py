"""C++ serving mode integration for Metal Marlin.

This module provides the high-performance C++ path for serving mode inference.
It wraps the C++ ServingContext and integrates with the ServingEngine for
low-latency decode and high-throughput prefill.

Usage:
    from metal_marlin._compat import HAS_CPP_EXT
    from metal_marlin.serving_cpp import ServingCppDispatcher
    
    if HAS_CPP_EXT:
        dispatcher = ServingCppDispatcher()
        # Use in serving hot path
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from metal_marlin._compat import HAS_CPP_EXT, _metal_dispatch_ext

if TYPE_CHECKING:
    import torch


class ServingCppDispatcher:
    """High-performance C++ dispatcher for serving mode.
    
    This class wraps the C++ ServingContext and provides:
    - Synchronous dispatch for low-latency decode
    - Asynchronous dispatch for throughput-optimized prefill
    - Metrics tracking for performance monitoring
    - Automatic fallback to Python path if C++ extension unavailable
    
    The C++ path eliminates PyObjC bridge overhead (~50μs per call)
    and provides more predictable latency for the serving hot path.
    
    Attributes:
        available: True if the C++ fast path is available.
        
    Example:
        >>> dispatcher = ServingCppDispatcher()
        >>> if dispatcher.available:
        ...     # Synchronous decode (low latency)
        ...     dispatcher.dispatch_sync(
        ...         "mmfp4_gemm", grid, threadgroup, buffers
        ...     )
        ...     
        ...     # Asynchronous prefill (high throughput)
        ...     handle = dispatcher.dispatch_async(
        ...         "attention_kernel", grid, threadgroup, buffers
        ...     )
        ...     dispatcher.wait(handle)
    """
    
    def __init__(self) -> None:
        """Initialize the C++ serving dispatcher."""
        self._ctx: Any = None
        self._serving_ctx: Any = None
        self._available = False
        
        if HAS_CPP_EXT and _metal_dispatch_ext is not None:
            try:
                # Check if ServingContext is available in the C++ extension
                if hasattr(_metal_dispatch_ext, 'ServingContext'):
                    self._ctx = _metal_dispatch_ext.MetalContext()
                    self._serving_ctx = _metal_dispatch_ext.ServingContext(self._ctx)
                    self._available = True
            except Exception:
                self._available = False
    
    @property
    def available(self) -> bool:
        """Return True if the C++ serving path is available."""
        return self._available
    
    def dispatch_sync(
        self,
        kernel_name: str,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        buffers: list[int],
    ) -> None:
        """Synchronous kernel dispatch for low-latency decode.
        
        This is the hot path for token generation. The kernel is dispatched
        synchronously, blocking until completion for minimal latency.
        
        Args:
            kernel_name: Name of the Metal kernel function.
            grid: Grid dimensions (x, y, z).
            threadgroup: Threadgroup dimensions (x, y, z).
            buffers: List of buffer pointers (from tensor.data_ptr()).
            
        Raises:
            RuntimeError: If C++ serving path is not available.
            
        Example:
            >>> buffers = [A.data_ptr(), B.data_ptr(), C.data_ptr()]
            >>> dispatcher.dispatch_sync(
            ...     "mmfp4_gemm", (8, 8, 1), (128, 1, 1), buffers
            ... )
        """
        if not self._available:
            raise RuntimeError("C++ serving path not available")
        
        self._serving_ctx.dispatch_sync(kernel_name, grid, threadgroup, buffers)
    
    def dispatch_async(
        self,
        kernel_name: str,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        buffers: list[int],
    ) -> Any:
        """Asynchronous kernel dispatch for throughput-optimized prefill.
        
        This is the path for prompt processing. The kernel is dispatched
        asynchronously, allowing CPU to continue while GPU works.
        
        Args:
            kernel_name: Name of the Metal kernel function.
            grid: Grid dimensions (x, y, z).
            threadgroup: Threadgroup dimensions (x, y, z).
            buffers: List of buffer pointers (from tensor.data_ptr()).
            
        Returns:
            A handle that can be passed to wait() or is_complete().
            
        Raises:
            RuntimeError: If C++ serving path is not available.
            
        Example:
            >>> buffers = [Q.data_ptr(), K.data_ptr(), V.data_ptr()]
            >>> handle = dispatcher.dispatch_async(
            ...     "flash_attention", (16, 1, 1), (128, 1, 1), buffers
            ... )
            >>> # Do other work...
            >>> dispatcher.wait(handle)
        """
        if not self._available:
            raise RuntimeError("C++ serving path not available")
        
        return self._serving_ctx.dispatch_async(kernel_name, grid, threadgroup, buffers)
    
    def wait(self, handle: Any) -> None:
        """Wait for an async dispatch to complete.
        
        Args:
            handle: The handle returned by dispatch_async().
        """
        if self._available and handle is not None:
            self._serving_ctx.wait(handle)
    
    def is_complete(self, handle: Any) -> bool:
        """Check if an async dispatch is complete (non-blocking).
        
        Args:
            handle: The handle returned by dispatch_async().
            
        Returns:
            True if the dispatch is complete, False otherwise.
        """
        if not self._available or handle is None:
            return True
        return self._serving_ctx.is_complete(handle)
    
    def get_metrics(self) -> dict[str, Any]:
        """Get serving metrics from the C++ layer.
        
        Returns:
            Dictionary with metrics:
            - dispatch_count: Total number of dispatches
            - total_dispatch_us: Total dispatch time in microseconds
            - avg_dispatch_us: Average dispatch time in microseconds
        """
        if not self._available:
            return {"dispatch_count": 0, "total_dispatch_us": 0, "avg_dispatch_us": 0}
        
        return dict(self._serving_ctx.metrics())
    
    def reset_metrics(self) -> None:
        """Reset serving metrics."""
        if self._available:
            self._serving_ctx.reset_metrics()


class ServingCppEngineWrapper:
    """Wrapper to integrate C++ serving dispatcher with ServingEngine.
    
    This class provides a drop-in replacement for the standard inference
    path that uses the C++ serving mode for dispatch.
    
    Attributes:
        available: True if the C++ fast path is available.
        
    Example:
        >>> from metal_marlin.serving.engine import ServingEngine
        >>> from metal_marlin.serving_cpp import ServingCppEngineWrapper
        >>> 
        >>> engine = ServingEngine(config)
        >>> cpp_wrapper = ServingCppEngineWrapper(engine)
        >>> 
        >>> # Use cpp_wrapper for C++ accelerated inference
        >>> if cpp_wrapper.available:
        ...     result = cpp_wrapper.generate(prompt, max_tokens=100)
    """
    
    def __init__(self, engine: Any) -> None:
        """Initialize the C++ serving engine wrapper.
        
        Args:
            engine: The ServingEngine instance to wrap.
        """
        self._engine = engine
        self._dispatcher = ServingCppDispatcher()
    
    @property
    def available(self) -> bool:
        """Return True if the C++ serving path is available."""
        return self._dispatcher.available
    
    def dispatch_for_decode(
        self,
        kernel_name: str,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        tensors: list[torch.Tensor],
    ) -> None:
        """Dispatch a kernel for decode phase using C++ fast path.
        
        Args:
            kernel_name: Name of the Metal kernel.
            grid: Grid dimensions.
            threadgroup: Threadgroup dimensions.
            tensors: List of PyTorch MPS tensors.
        """
        if not self._dispatcher.available:
            raise RuntimeError("C++ serving path not available")
        
        # Extract buffer pointers from tensors
        buffer_ptrs = [t.data_ptr() for t in tensors]
        
        # Synchronous dispatch for decode (low latency)
        self._dispatcher.dispatch_sync(kernel_name, grid, threadgroup, buffer_ptrs)
    
    def dispatch_for_prefill(
        self,
        kernel_name: str,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        tensors: list[torch.Tensor],
    ) -> Any:
        """Dispatch a kernel for prefill phase using C++ fast path.
        
        Args:
            kernel_name: Name of the Metal kernel.
            grid: Grid dimensions.
            threadgroup: Threadgroup dimensions.
            tensors: List of PyTorch MPS tensors.
            
        Returns:
            Handle for the async dispatch.
        """
        if not self._dispatcher.available:
            raise RuntimeError("C++ serving path not available")
        
        # Extract buffer pointers from tensors
        buffer_ptrs = [t.data_ptr() for t in tensors]
        
        # Asynchronous dispatch for prefill (high throughput)
        return self._dispatcher.dispatch_async(
            kernel_name, grid, threadgroup, buffer_ptrs
        )
    
    def get_metrics(self) -> dict[str, Any]:
        """Get C++ serving metrics."""
        return self._dispatcher.get_metrics()


def create_serving_cpp_dispatcher() -> ServingCppDispatcher | None:
    """Factory function to create a C++ serving dispatcher if available.
    
    Returns:
        ServingCppDispatcher instance if available, None otherwise.
        
    Example:
        >>> dispatcher = create_serving_cpp_dispatcher()
        >>> if dispatcher:
        ...     dispatcher.dispatch_sync("kernel", grid, tg, buffers)
    """
    dispatcher = ServingCppDispatcher()
    return dispatcher if dispatcher.available else None


__all__ = [
    "ServingCppDispatcher",
    "ServingCppEngineWrapper",
    "create_serving_cpp_dispatcher",
]
