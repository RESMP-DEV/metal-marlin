"""
MTLBuffer direct pointer access bridge.

Provides fast direct access to Metal buffer contents by bypassing PyObjC's
method dispatch overhead. Falls back to PyObjC when extension is unavailable.

Usage:
    from metal_marlin.buffer_bridge import get_buffer_view, get_buffer_memoryview

    # Option 1: Get reusable view object
    view = get_buffer_view(mtl_buffer)
    mv = memoryview(view)
    mv[:100] = data

    # Option 2: One-shot memoryview
    mv = get_buffer_memoryview(mtl_buffer)
    mv[:100] = data

    # Option 3: Direct pointer for FFI
    ptr, length = get_buffer_ptr(mtl_buffer)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import Foundation

if TYPE_CHECKING:
    pass

# Try to import Cython extension
_HAS_NATIVE_BRIDGE = False
try:
    from metal_marlin._metal_buffer_bridge import (
        MetalBufferView as _NativeBufferView,
    )
    from metal_marlin._metal_buffer_bridge import (
        aligned_copy as _native_aligned_copy,
    )
    from metal_marlin._metal_buffer_bridge import (
        get_buffer_contents_ptr as _native_get_ptr,
    )
    from metal_marlin._metal_buffer_bridge import (
        get_buffer_memoryview as _native_get_memoryview,
    )
    from metal_marlin._metal_buffer_bridge import (
        get_buffer_pointers_batch as _native_get_batch,
    )
    _HAS_NATIVE_BRIDGE = True
except ImportError:
    _NativeBufferView = None
    _native_aligned_copy = None
    _native_get_ptr = None
    _native_get_memoryview = None
    _native_get_batch = None


class _PyObjCBufferView:
    """Fallback buffer view using PyObjC.

    Used when Cython extension is not available. Provides the same interface
    but with higher overhead.
    """

    __slots__ = ("_buffer", "_contents", "_length", "_view")

    def __init__(self, buffer: Any):
        self._buffer = buffer
        self._contents = buffer.contents()
        self._length = buffer.length()
        self._view: memoryview | None = None

    def __getbuffer__(self, view, flags):
        raise NotImplementedError("PyObjC fallback does not support buffer protocol directly")

    @property
    def ptr(self) -> int:
        """Raw pointer as integer."""
        # PyObjC doesn't expose raw pointer directly
        return id(self._contents)

    @property
    def length(self) -> int:
        """Buffer length in bytes."""
        return self._length

    def as_memoryview(self) -> memoryview:
        """Get memoryview of buffer contents."""
        if self._view is None:
            self._view = memoryview(self._contents.as_buffer(self._length))
        return self._view

    def write(self, offset: int, data: bytes) -> None:
        """Write bytes to buffer at offset."""
        mv = self.as_memoryview()
        mv[offset:offset + len(data)] = data
        self._buffer.didModifyRange_(Foundation.NSMakeRange(offset, len(data)))

    def read(self, offset: int, size: int) -> bytes:
        """Read bytes from buffer at offset."""
        mv = self.as_memoryview()
        return bytes(mv[offset:offset + size])

    def zero(self, offset: int, size: int) -> None:
        """Zero a region of the buffer."""
        mv = self.as_memoryview()
        mv[offset:offset + size] = b"\x00" * size
        self._buffer.didModifyRange_(Foundation.NSMakeRange(offset, size))


def get_buffer_view(buffer: Any) -> Any:
    """Get a view object for MTLBuffer contents.

    Returns a MetalBufferView (native) or _PyObjCBufferView (fallback).
    The view provides:
    - memoryview access via as_memoryview() or memoryview(view)
    - Direct read/write methods
    - Pointer access for FFI

    Args:
        buffer: PyObjC MTLBuffer object

    Returns:
        Buffer view object with read/write methods
    """
    if _HAS_NATIVE_BRIDGE and _NativeBufferView is not None:
        return _NativeBufferView(buffer)
    return _PyObjCBufferView(buffer)


def get_buffer_memoryview(buffer: Any) -> memoryview:
    """Get memoryview of MTLBuffer contents.

    For one-shot access. For repeated access, use get_buffer_view().

    Args:
        buffer: PyObjC MTLBuffer object

    Returns:
        memoryview of buffer contents
    """
    if _HAS_NATIVE_BRIDGE and _native_get_memoryview is not None:
        return _native_get_memoryview(buffer)
    return _PyObjCBufferView(buffer).as_memoryview()


def get_buffer_ptr(buffer: Any) -> tuple[int, int]:
    """Get raw pointer and length from MTLBuffer.

    For use with ctypes/cffi FFI or other native code.

    Args:
        buffer: PyObjC MTLBuffer object

    Returns:
        Tuple of (pointer_as_int, length_in_bytes)
    """
    if _HAS_NATIVE_BRIDGE and _native_get_ptr is not None:
        return _native_get_ptr(buffer)
    # Fallback: can't get actual pointer, return id as approximation
    contents = buffer.contents()
    return (id(contents), buffer.length())


def get_buffer_pointers_batch(buffers: list[Any]) -> list[tuple[int, int]]:
    """Get pointers for multiple buffers efficiently.

    Minimizes Python overhead when accessing many buffers.

    Args:
        buffers: List of PyObjC MTLBuffer objects

    Returns:
        List of (pointer, length) tuples
    """
    if _HAS_NATIVE_BRIDGE and _native_get_batch is not None:
        return _native_get_batch(buffers)
    return [get_buffer_ptr(buf) for buf in buffers]


def aligned_copy(
    dst_ptr: int,
    src_ptr: int,
    size: int,
    dst_offset: int = 0,
    src_offset: int = 0,
) -> None:
    """Copy data between raw pointers.

    Only available with native extension. Raises if extension not loaded.

    Args:
        dst_ptr: Destination pointer as integer
        src_ptr: Source pointer as integer
        size: Bytes to copy
        dst_offset: Offset into destination
        src_offset: Offset into source
    """
    if _HAS_NATIVE_BRIDGE and _native_aligned_copy is not None:
        _native_aligned_copy(dst_ptr, src_ptr, size, dst_offset, src_offset)
    else:
        raise RuntimeError(
            "aligned_copy requires native extension. "
            "Build with: cd contrib/metal_marlin && python setup.py build_ext --inplace"
        )


def is_native_bridge_available() -> bool:
    """Check if native Cython extension is available."""
    return _HAS_NATIVE_BRIDGE


__all__ = [
    "get_buffer_view",
    "get_buffer_memoryview",
    "get_buffer_ptr",
    "get_buffer_pointers_batch",
    "aligned_copy",
    "is_native_bridge_available",
]
