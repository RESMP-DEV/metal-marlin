"""Metal buffer pooling for shared CPU/GPU buffers."""

from __future__ import annotations

from typing import Any

import Metal


def _round_up(value: int, alignment: int = 256) -> int:
    return ((value + alignment - 1) // alignment) * alignment


class MetalBufferPool:
    def __init__(
        self,
        device: Any,
        *,
        storage_mode: int = Metal.MTLResourceStorageModeShared,
    ) -> None:
        self._device = device
        self._storage_mode = storage_mode
        self._pools: dict[int, list[Any]] = {}
        self._backing: dict[int, bytearray] = {}
        self._sizes: dict[int, int] = {}

    def get(self, size: int) -> Any:
        """Get or create buffer of at least `size` bytes."""
        for pool_size in sorted(self._pools.keys()):
            if pool_size >= size and self._pools[pool_size]:
                return self._pools[pool_size].pop()

        alloc_size = _round_up(size)
        if self._storage_mode == Metal.MTLResourceStorageModeShared:
            backing = bytearray(alloc_size)
            buffer = self._device.newBufferWithBytesNoCopy_length_options_deallocator_(
                backing,
                alloc_size,
                Metal.MTLResourceStorageModeShared,
                None,
            )
            self._backing[id(buffer)] = backing
        else:
            buffer = self._device.newBufferWithLength_options_(
                alloc_size, self._storage_mode
            )
        if buffer is None:
            raise RuntimeError("Failed to allocate Metal buffer")

        self._sizes[id(buffer)] = alloc_size
        return buffer

    def release(self, buf: Any) -> None:
        """Return buffer to pool for reuse."""
        buf_id = id(buf)
        size = self._sizes.get(buf_id)
        if size is None:
            return
        self._pools.setdefault(size, []).append(buf)
