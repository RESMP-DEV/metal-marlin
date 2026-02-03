# cython: language_level=3
"""
Declaration file for _metal_buffer_bridge extension.
"""

from libc.stdint cimport uintptr_t

cdef class MetalBufferView:
    cdef void* _ptr
    cdef Py_ssize_t _length
    cdef object _buffer_ref

    cpdef write(self, Py_ssize_t offset, bytes data)
    cpdef bytes read(self, Py_ssize_t offset, Py_ssize_t size)
    cpdef zero(self, Py_ssize_t offset, Py_ssize_t size)

cpdef object get_buffer_contents_ptr(buffer)
cpdef memoryview get_buffer_memoryview(buffer)
cpdef list get_buffer_pointers_batch(list buffers)
cpdef void aligned_copy(
    Py_ssize_t dst_ptr,
    Py_ssize_t src_ptr,
    Py_ssize_t size,
    Py_ssize_t dst_offset=*,
    Py_ssize_t src_offset=*
) noexcept nogil
