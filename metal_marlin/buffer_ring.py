"""Ring buffer for transient allocations.

Pre-allocates a large contiguous buffer and hands out sequential slices during
forward passes. Calling reset() returns the offset to zero, enabling zero-cost
buffer reuse without per-allocation overhead.

Key optimizations:
- Single large allocation (default 256MB) instead of many small allocations
- Sequential slice allocation with cache-line alignment
- O(1) reset between forward passes
- No per-allocation overhead (just offset tracking)

Usage:
    from metal_marlin.buffer_ring import RingBuffer

    ring = RingBuffer(device, capacity=256 * 1024 * 1024)

    def forward_pass(...):
        ring.reset()

        buf1, offset1 = ring.alloc(batch * hidden * 2)
        buf2, offset2 = ring.alloc(batch * hidden * 4)

    No cleanup needed - reset on next forward pass
"""

from __future__ import annotations

from typing import Any

try:
    import Metal

    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    Metal = None


CACHE_LINE_BYTES = 128
PAGE_SIZE_BYTES = 16384
LARGE_BUFFER_THRESHOLD = 65536


def _round_up(value: int, alignment: int = CACHE_LINE_BYTES) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _align_buffer_size(size: int) -> int:
    if size >= LARGE_BUFFER_THRESHOLD:
        return _round_up(size, PAGE_SIZE_BYTES)
    return _round_up(size, CACHE_LINE_BYTES)


class RingBuffer:
    """Ring buffer for per-forward-pass transient allocations.

    Pre-allocates a large contiguous Metal buffer and hands out sequential
    slices during a forward pass. Calling reset() at the start of each forward
    pass returns the offset to zero, enabling zero-cost buffer reuse without
    any per-allocation overhead.

    Args:
        device: MTLDevice for buffer allocation
        capacity: Total capacity in bytes (default 256MB)
        storage_mode: Metal storage mode (default MTLResourceStorageModeShared)

    Usage:
        ring = RingBuffer(device, capacity=256 * 1024 * 1024)

        def forward_pass(...):
            ring.reset()

            buf1, offset1 = ring.alloc(batch * hidden * 2)
            buf2, offset2 = ring.alloc(batch * hidden * 4)

        The ring buffer returns the same underlying Metal buffer for all
        allocations, with different byte offsets. Callers must use the offset
        when dispatching Metal kernels to access the correct region.
    """

    DEFAULT_CAPACITY = 256 * 1024 * 1024

    def __init__(
        self,
        device: Any,
        capacity: int = DEFAULT_CAPACITY,
        *,
        storage_mode: int | None = None,
    ) -> None:
        if not HAS_METAL:
            raise RuntimeError("Metal required. Install with: pip install pyobjc-framework-Metal")

        if storage_mode is None:
            storage_mode = Metal.MTLResourceStorageModeShared

        self._device = device
        self._capacity = _align_buffer_size(capacity)
        self._storage_mode = storage_mode
        self._offset = 0

        self._backing = bytearray(self._capacity)
        self._buffer = device.newBufferWithBytesNoCopy_length_options_deallocator_(
            self._backing,
            self._capacity,
            storage_mode,
            None,
        )
        if self._buffer is None:
            raise RuntimeError(f"Failed to allocate ring buffer of {self._capacity} bytes")

        self._high_water_mark = 0
        self._allocation_count = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def used(self) -> int:
        return self._offset

    @property
    def available(self) -> int:
        return self._capacity - self._offset

    @property
    def high_water_mark(self) -> int:
        return self._high_water_mark

    @property
    def buffer(self) -> Any:
        return self._buffer

    def reset(self) -> None:
        self._offset = 0
        self._allocation_count = 0

    def can_alloc(self, size: int) -> bool:
        aligned_size = _round_up(size, CACHE_LINE_BYTES)
        return self._offset + aligned_size <= self._capacity

    def alloc(self, size: int) -> tuple[Any, int]:
        aligned_size = _round_up(size, CACHE_LINE_BYTES)

        if self._offset + aligned_size > self._capacity:
            raise RuntimeError(
                f"Ring buffer overflow: requested {size} bytes "
                f"(aligned to {aligned_size}), "
                f"only {self._capacity - self._offset} available "
                f"of {self._capacity} total capacity. "
                f"Consider increasing ring buffer size or calling reset()."
            )

        offset = self._offset
        self._offset += aligned_size
        self._allocation_count += 1

        if self._offset > self._high_water_mark:
            self._high_water_mark = self._offset

        return self._buffer, offset

    def alloc_bytes(self, size: int) -> tuple[memoryview, int]:
        aligned_size = _round_up(size, CACHE_LINE_BYTES)

        if self._offset + aligned_size > self._capacity:
            raise RuntimeError(
                f"Ring buffer overflow: requested {size} bytes "
                f"(aligned to {aligned_size}), "
                f"only {self._capacity - self._offset} available"
            )

        offset = self._offset
        self._offset += aligned_size
        self._allocation_count += 1

        if self._offset > self._high_water_mark:
            self._high_water_mark = self._offset

        view = memoryview(self._backing)[offset : offset + size]
        return view, offset

    def stats(self) -> dict[str, Any]:
        return {
            "capacity_bytes": self._capacity,
            "used_bytes": self._offset,
            "available_bytes": self._capacity - self._offset,
            "high_water_mark_bytes": self._high_water_mark,
            "allocation_count": self._allocation_count,
            "utilization": self._offset / self._capacity if self._capacity > 0 else 0,
            "peak_utilization": (
                self._high_water_mark / self._capacity if self._capacity > 0 else 0
            ),
        }


_ring_buffer: RingBuffer | None = None
_ring_buffer_device_id: int | None = None


def get_ring_buffer(
    device: Any,
    capacity: int = RingBuffer.DEFAULT_CAPACITY,
) -> RingBuffer:
    global _ring_buffer, _ring_buffer_device_id

    device_id = id(device)
    if _ring_buffer is None or _ring_buffer_device_id != device_id:
        _ring_buffer = RingBuffer(device, capacity)
        _ring_buffer_device_id = device_id

    return _ring_buffer


def reset_ring_buffer() -> None:
    global _ring_buffer
    if _ring_buffer is not None:
        _ring_buffer.reset()


def ring_buffer_stats() -> dict[str, Any] | None:
    global _ring_buffer
    if _ring_buffer is not None:
        return _ring_buffer.stats()
    return None
