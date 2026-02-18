"""Async command buffer completion handling for Metal.

This module provides asynchronous command buffer execution using Metal's
addCompletedHandler_ callback mechanism instead of blocking waitUntilCompleted().

This enables overlapping CPU work with GPU execution:
    - Commit work to GPU
    - Continue CPU processing
    - Check completion status or wait later when results are needed

Example:
    from metal_marlin.async_dispatch import AsyncCommandBuffer
    
    # Wrap an existing command buffer
    async_cb = AsyncCommandBuffer(command_buffer)
    
    # Submit to GPU and get future
    future = async_cb.commit()
    
    # Do other CPU work while GPU executes
    process_other_data()
    
    # Wait for GPU completion when needed
    future.wait()
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

# Check PyObjC Metal availability
try:
    import Foundation
    import Metal

    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    Metal = None
    Foundation = None


class AsyncFuture:
    """Future/promise for async command buffer completion.
    
    Provides a simple synchronization primitive for checking GPU completion
    status without blocking. The future is resolved when the Metal command
    buffer's completion handler is invoked.
    
    Attributes:
        _completed_event: Threading event that signals completion
        _error: Any error that occurred during execution
        _callback: Optional user callback to invoke on completion
    """
    
    __slots__ = ("_completed_event", "_error", "_callback", "_command_buffer")
    
    def __init__(self, command_buffer: Any | None = None) -> None:
        """Initialize an async future.
        
        Args:
            command_buffer: The MTLCommandBuffer being tracked (optional)
        """
        self._completed_event = threading.Event()
        self._error: Any | None = None
        self._callback: Callable[[], None] | None = None
        self._command_buffer: Any | None = command_buffer
    
    def mark_completed(self, error: Any | None = None) -> None:
        """Mark the future as completed.
        
        This is called internally by the completion handler.
        
        Args:
            error: Any error that occurred during execution
        """
        self._error = error
        self._completed_event.set()
        
        # Invoke user callback if registered
        if self._callback is not None:
            try:
                self._callback()
            except Exception:
                pass  # User callbacks shouldn't break our state
    
    def wait(self) -> None:
        """Block until GPU completes.
        
        This method blocks the calling thread until the command buffer
        has finished executing on the GPU.
        
        Raises:
            RuntimeError: If an error occurred during GPU execution
        """
        self._completed_event.wait()
        
        if self._error is not None:
            raise RuntimeError(f"Command buffer execution failed: {self._error}")
    
    def is_ready(self) -> bool:
        """Check if GPU finished.
        
        Returns:
            True if the command buffer has completed execution,
            False if still in progress.
        """
        return self._completed_event.is_set()
    
    def on_complete(self, callback: Callable[[], None]) -> None:
        """Register a callback to be invoked on completion.
        
        If already completed, the callback is invoked immediately.
        
        Args:
            callback: Function to call when command buffer completes
        """
        if self.is_ready():
            try:
                callback()
            except Exception:
                pass
        else:
            self._callback = callback
    
    @property
    def error(self) -> Any | None:
        """Any error that occurred during execution, or None."""
        return self._error


class AsyncCommandBuffer:
    """Wrapper for async command buffer completion handling.
    
    This class wraps an MTLCommandBuffer and provides a future-based
    interface for async completion using addCompletedHandler_.
    
    Example:
        # Create or obtain a command buffer
        command_buffer = command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        # ... encode work ...
        encoder.endEncoding()
        
        # Wrap for async execution
        async_cb = AsyncCommandBuffer(command_buffer)
        future = async_cb.commit()
        
        # Continue CPU work
        do_other_work()
        
        # Wait for completion
        future.wait()
    
    Attributes:
        _command_buffer: The underlying MTLCommandBuffer
        _future: The AsyncFuture for completion tracking
        _committed: Whether the buffer has been committed
    """
    
    __slots__ = ("_command_buffer", "_future", "_committed")
    
    def __init__(self, command_buffer: Any) -> None:
        """Initialize with a command buffer.
        
        Args:
            command_buffer: An MTLCommandBuffer instance
            
        Raises:
            RuntimeError: If PyObjC Metal is not available
        """
        if not HAS_METAL:
            raise RuntimeError(
                "AsyncCommandBuffer requires PyObjC. Install with:\n"
                "  pip install pyobjc-framework-Metal"
            )
        
        self._command_buffer = command_buffer
        self._future: AsyncFuture | None = None
        self._committed = False
    
    def commit(self) -> AsyncFuture:
        """Submit to GPU, return future.
        
        Commits the command buffer for execution and returns an AsyncFuture
        that can be used to track completion status.
        
        Returns:
            AsyncFuture for checking completion status
            
        Raises:
            RuntimeError: If the buffer has already been committed
        """
        if self._committed:
            raise RuntimeError("Command buffer already committed")
        
        self._committed = True
        self._future = AsyncFuture(self._command_buffer)
        
        # Create completion handler using PyObjC block
        # The handler is called when the command buffer completes
        def completion_handler(cmd_buffer: Any) -> None:
            """Called by Metal when command buffer completes."""
            status = cmd_buffer.status()
            error = None
            
            # MTLCommandBufferStatus:
            # 0 = notEnqueued, 1 = enqueued, 2 = committed
            # 3 = scheduled, 4 = completed, 5 = error
            if status == 5:
                error = cmd_buffer.error()
            
            if self._future is not None:
                self._future.mark_completed(error)
        
        # Add the completion handler using addCompletedHandler_
        # This is the key API that enables async notification
        self._command_buffer.addCompletedHandler_(completion_handler)
        
        # Commit the command buffer to start GPU execution
        self._command_buffer.commit()
        
        return self._future
    
    @property
    def future(self) -> AsyncFuture | None:
        """The AsyncFuture for this command buffer, or None if not committed."""
        return self._future
    
    @property
    def command_buffer(self) -> Any:
        """The underlying MTLCommandBuffer."""
        return self._command_buffer
    
    @property
    def is_committed(self) -> bool:
        """Whether the command buffer has been committed."""
        return self._committed


def create_async_future() -> AsyncFuture:
    """Create a standalone AsyncFuture for manual completion management.
    
    This is useful when you need to coordinate multiple async operations
    or implement custom completion logic.
    
    Returns:
        A new AsyncFuture instance
    """
    return AsyncFuture()


def commit_async(command_buffer: Any) -> AsyncFuture:
    """Commit a command buffer asynchronously and return a future.
    
    This is a convenience function equivalent to:
        async_cb = AsyncCommandBuffer(command_buffer)
        return async_cb.commit()
    
    Args:
        command_buffer: An MTLCommandBuffer instance
        
    Returns:
        AsyncFuture for tracking completion
    """
    async_cb = AsyncCommandBuffer(command_buffer)
    return async_cb.commit()


class AsyncCommandQueue:
    """High-level async command queue wrapper.
    
    Provides a convenient interface for submitting multiple command buffers
    asynchronously and tracking their completion.
    
    Example:
        queue = AsyncCommandQueue(command_queue)
        
        # Submit multiple buffers
        future1 = queue.submit(buffer1)
        future2 = queue.submit(buffer2)
        
        # Wait for all to complete
        future1.wait()
        future2.wait()
    """
    
    __slots__ = ("_command_queue",)
    
    def __init__(self, command_queue: Any) -> None:
        """Initialize with a command queue.
        
        Args:
            command_queue: An MTLCommandQueue instance
        """
        self._command_queue = command_queue
    
    def submit(self, command_buffer: Any) -> AsyncFuture:
        """Submit a command buffer asynchronously.
        
        Args:
            command_buffer: An MTLCommandBuffer instance
            
        Returns:
            AsyncFuture for tracking completion
        """
        return commit_async(command_buffer)
    
    def create_buffer(self) -> Any:
        """Create a new command buffer from the queue.
        
        Returns:
            A new MTLCommandBuffer instance
        """
        return self._command_queue.commandBuffer()


# -----------------------------------------------------------------------------
# Synchronization helpers for multiple futures
# -----------------------------------------------------------------------------

def wait_all(futures: list[AsyncFuture]) -> None:
    """Wait for all futures to complete.
    
    Args:
        futures: List of AsyncFuture instances to wait for
    """
    for future in futures:
        future.wait()


def wait_any(futures: list[AsyncFuture]) -> int:
    """Wait for any future to complete.
    
    Args:
        futures: List of AsyncFuture instances to monitor
        
    Returns:
        Index of the first completed future
    """
    # Poll with a small timeout to avoid busy waiting
    import time
    
    while True:
        for i, future in enumerate(futures):
            if future.is_ready():
                return i
        time.sleep(0.001)  # 1ms sleep to avoid busy waiting


def all_ready(futures: list[AsyncFuture]) -> bool:
    """Check if all futures are ready.
    
    Args:
        futures: List of AsyncFuture instances to check
        
    Returns:
        True if all futures have completed
    """
    return all(f.is_ready() for f in futures)


def any_ready(futures: list[AsyncFuture]) -> bool:
    """Check if any future is ready.
    
    Args:
        futures: List of AsyncFuture instances to check
        
    Returns:
        True if at least one future has completed
    """
    return any(f.is_ready() for f in futures)
