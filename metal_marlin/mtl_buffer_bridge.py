"""
Optimized MTLBuffer direct pointer access bridge.

This module provides the fastest possible access to Metal buffer contents by using:
- Cached IMP (method implementation) pointers for Objective-C dispatch
- Direct memory access without C++ wrapper overhead  
- SIMD-aligned memory operations
- Zero-copy MPS tensor wrapping

Usage:
    from metal_marlin.mtl_buffer_bridge import DirectBufferPtr, get_buffer_ptr

    # Fast one-shot pointer access
    ptr, length = get_buffer_ptr(mtl_buffer)
    
    # Reusable pointer wrapper
    dbp = DirectBufferPtr(mtl_buffer)
    mv = dbp.as_float32()  # numpy array view
    
    # Zero-copy MPS tensor wrapping
    from metal_marlin.mtl_buffer_bridge import MPSTensorWrapper
    wrapper = MPSTensorWrapper.wrap(mps_buffer)
    ptr = wrapper.ptr  # Direct pointer
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    import numpy as np

# Try to import the native extension
try:
    from metal_marlin._mtl_buffer_bridge import (
        DirectBufferPtr as _NativeDirectBufferPtr,
        MPSTensorWrapper as _NativeMPSTensorWrapper,
        align_buffer_size,
        align_to_cache_line,
        fast_copy,
        fast_zero,
        get_buffer_ptr as _native_get_buffer_ptr,
        get_buffer_ptr_batch as _native_get_buffer_ptr_batch,
        prefetch_read,
        prefetch_write,
        CACHE_LINE_SIZE,
        PAGE_SIZE,
        LARGE_BUFFER_THRESHOLD,
    )
    _HAS_NATIVE = True
except ImportError as _import_err:
    _HAS_NATIVE = False
    _NativeDirectBufferPtr = None
    _NativeMPSTensorWrapper = None
    
    # Define fallback constants
    CACHE_LINE_SIZE = 128
    PAGE_SIZE = 16384
    LARGE_BUFFER_THRESHOLD = 65536


class DirectBufferPtr:
    """
    Zero-overhead direct pointer access to Metal buffer contents.
    
    This class provides the fastest possible access to MTLBuffer memory by:
    1. Caching IMP (method implementation) pointers for fast ObjC dispatch
    2. Storing direct pointer to buffer contents
    3. Providing SIMD-aligned memory operations
    
    Performance: ~5ns per access (vs ~50ns through PyObjC)
    
    Example:
        dbp = DirectBufferPtr(mtl_buffer)
        
        # Get raw pointer
        ptr = dbp.ptr
        
        # Get numpy view (zero-copy)
        arr = dbp.as_float32()
        arr[0:100] = some_data
        
        # Fast copy operations
        dbp.copy_from(src_ptr, size, offset=0)
        dbp.copy_to(dst_ptr, size, offset=0)
        
        # Zero regions
        dbp.zero(offset=0, size=1024)
        dbp.zero_all()
    """
    
    def __init__(self, buffer: Any) -> None:
        """
        Create direct pointer wrapper for MTLBuffer.
        
        Args:
            buffer: PyObjC MTLBuffer or capsule pointer
            
        Raises:
            TypeError: If buffer type is not recognized
            RuntimeError: If native extension is not available
        """
        if not _HAS_NATIVE:
            raise RuntimeError(
                "Native extension not available. "
                "Build with: cd contrib/metal_marlin && python setup.py build_ext --inplace"
            )
        
        self._native = _NativeDirectBufferPtr(buffer)
        
    @property
    def is_valid(self) -> bool:
        """Check if pointer is valid."""
        return self._native.is_valid
    
    @property
    def ptr(self) -> int:
        """Get raw pointer as integer."""
        return self._native.ptr
    
    @property
    def length(self) -> int:
        """Get buffer length in bytes."""
        return self._native.length
    
    def as_float32(self) -> "np.ndarray":
        """Get numpy float32 array view (zero-copy)."""
        return self._native.as_float32()
    
    def as_float16(self) -> "np.ndarray":
        """Get numpy float16 (uint16) array view (zero-copy)."""
        return self._native.as_float16()
    
    def as_int32(self) -> "np.ndarray":
        """Get numpy int32 array view (zero-copy)."""
        return self._native.as_int32()
    
    def as_uint8(self) -> "np.ndarray":
        """Get numpy uint8 array view (zero-copy)."""
        return self._native.as_uint8()
    
    def copy_from(self, src: int, size: int, offset: int = 0) -> None:
        """
        Copy data from source pointer to buffer.
        
        Args:
            src: Source pointer (as integer)
            size: Number of bytes to copy
            offset: Byte offset into destination buffer
        """
        self._native.copy_from(src, size, offset)
    
    def copy_to(self, dst: int, size: int, offset: int = 0) -> None:
        """
        Copy data from buffer to destination pointer.
        
        Args:
            dst: Destination pointer (as integer)
            size: Number of bytes to copy
            offset: Byte offset into source buffer
        """
        self._native.copy_to(dst, size, offset)
    
    def zero(self, offset: int, size: int) -> None:
        """Zero a region of the buffer."""
        self._native.zero(offset, size)
    
    def zero_all(self) -> None:
        """Zero entire buffer."""
        self._native.zero_all()
    
    def prefetch_read(self) -> None:
        """Prefetch buffer for reading (performance hint)."""
        self._native.prefetch_read()
    
    def prefetch_write(self) -> None:
        """Prefetch buffer for writing (performance hint)."""
        self._native.prefetch_write()
    
    def __len__(self) -> int:
        return self.length
    
    def __bool__(self) -> bool:
        return self.is_valid


class MPSTensorWrapper:
    """
    Zero-copy wrapper for MPS tensor data.
    
    Wraps existing MPS tensor memory without copying, providing
    direct pointer access for interop with C/C++ code.
    
    Example:
        # Wrap MPS tensor buffer
        wrapper = MPSTensorWrapper.wrap(mps_tensor_buffer)
        
        # Access pointer
        ptr = wrapper.ptr
        
        # Use with ctypes/cffi
        c_func(ptr, wrapper.size)
    """
    
    def __init__(self, native_wrapper: Any):
        """Private constructor - use MPSTensorWrapper.wrap() instead."""
        self._native = native_wrapper
    
    @staticmethod
    def wrap(buffer: Any) -> "MPSTensorWrapper":
        """
        Wrap existing MPS tensor buffer (no copy).
        
        Args:
            buffer: PyObjC MTLBuffer from MPS tensor
            
        Returns:
            MPSTensorWrapper with direct pointer access
        """
        if not _HAS_NATIVE:
            raise RuntimeError("Native extension not available")
        
        return MPSTensorWrapper(_NativeMPSTensorWrapper.wrap(buffer))
    
    @property
    def is_valid(self) -> bool:
        """Check if wrapper is valid."""
        return self._native.is_valid
    
    @property
    def ptr(self) -> int:
        """Get data pointer as integer."""
        return self._native.ptr
    
    @property
    def size(self) -> int:
        """Get tensor size in bytes."""
        return self._native.size
    
    def __len__(self) -> int:
        return self.size
    
    def __bool__(self) -> bool:
        return self.is_valid


def get_buffer_ptr(buffer: Any) -> Tuple[int, int]:
    """
    Get direct pointer and length from MTLBuffer (fast one-shot access).
    
    This is the fastest way to get buffer pointer for FFI calls.
    
    Args:
        buffer: PyObjC MTLBuffer or capsule
        
    Returns:
        Tuple of (pointer_as_int, length_in_bytes)
        
    Example:
        ptr, length = get_buffer_ptr(mtl_buffer)
        ctypes_func(ptr, length)
    """
    if not _HAS_NATIVE:
        raise RuntimeError("Native extension not available")
    
    return _native_get_buffer_ptr(buffer)


def get_buffer_ptr_batch(buffers: list) -> list[Tuple[int, int]]:
    """
    Get pointers for multiple buffers efficiently.
    
    Minimizes Python overhead when accessing many buffers.
    
    Args:
        buffers: List of PyObjC MTLBuffer objects
        
    Returns:
        List of (pointer, length) tuples
    """
    if not _HAS_NATIVE:
        raise RuntimeError("Native extension not available")
    
    return _native_get_buffer_ptr_batch(buffers)


def aligned_copy(dst: int, src: int, size: int) -> None:
    """
    Fast aligned memory copy.
    
    Args:
        dst: Destination pointer (as integer)
        src: Source pointer (as integer)
        size: Number of bytes to copy
    """
    if _HAS_NATIVE:
        fast_copy(dst, src, size)
    else:
        import ctypes
        ctypes.memmove(dst, src, size)


def aligned_zero(ptr: int, size: int) -> None:
    """
    Fast aligned zero fill.
    
    Args:
        ptr: Pointer to zero (as integer)
        size: Number of bytes to zero
    """
    if _HAS_NATIVE:
        fast_zero(ptr, size)
    else:
        import ctypes
        ctypes.memset(ptr, 0, size)


def is_native_available() -> bool:
    """Check if native optimized extension is available."""
    return _HAS_NATIVE


# Export symbols
__all__ = [
    "DirectBufferPtr",
    "MPSTensorWrapper",
    "get_buffer_ptr",
    "get_buffer_ptr_batch",
    "aligned_copy",
    "aligned_zero",
    "align_buffer_size",
    "align_to_cache_line",
    "prefetch_read",
    "prefetch_write",
    "CACHE_LINE_SIZE",
    "PAGE_SIZE",
    "LARGE_BUFFER_THRESHOLD",
    "is_native_available",
]
