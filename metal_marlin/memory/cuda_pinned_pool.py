"""Pinned memory pool for CUDA weight loading.

Pre-allocates pinned CPU buffers to amortize the cost of cudaHostAlloc
(~100us per call). Buffers are reused across load_tensor() calls via a
semaphore-guarded pool, eliminating per-tensor pin/unpin overhead.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager

import torch


class CUDAPinnedPool:
    """Pool of pre-allocated pinned CPU buffers for fast H2D DMA transfers.

    Usage:
        pool = CUDAPinnedPool(num_buffers=4, buffer_size_mb=64)
        with pool.get_buffer(size) as buf:
            # buf is a pinned CPU tensor, ready for non_blocking .to(device)
            data = load_from_file_into(buf)
            gpu_tensor = buf[:actual_size].to("cuda:0", non_blocking=True)
    """

    def __init__(self, num_buffers: int = 4, buffer_size_mb: float = 64.0) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDAPinnedPool requires CUDA. torch.cuda.is_available() returned False."
            )

        self._buffer_size = int(buffer_size_mb * 1024 * 1024)
        self._num_buffers = num_buffers

        # Pre-allocate pinned buffers
        self._pool: deque[torch.Tensor] = deque()
        for _ in range(num_buffers):
            self._pool.append(
                torch.empty(self._buffer_size, dtype=torch.uint8, pin_memory=True)
            )

        # Semaphore counts available buffers; blocks get_buffer when exhausted
        self._semaphore = threading.Semaphore(num_buffers)
        self._lock = threading.Lock()

    @contextmanager
    def get_buffer(self, size_bytes: int) -> Generator[torch.Tensor, None, None]:
        """Get a pinned buffer from the pool. Blocks if all in use.

        If *size_bytes* exceeds the pre-allocated buffer size, a one-off
        pinned tensor is allocated and freed on exit (the pool is not touched).

        Args:
            size_bytes: Minimum usable bytes required.

        Yields:
            A pinned CPU ``torch.uint8`` tensor with at least *size_bytes* elements.
        """
        if size_bytes > self._buffer_size:
            # Oversized request: allocate a one-off pinned tensor.
            yield torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
            return

        # Wait for a buffer to become available
        self._semaphore.acquire()
        buf: torch.Tensor | None = None
        try:
            with self._lock:
                buf = self._pool.popleft()
            yield buf
        finally:
            if buf is not None:
                with self._lock:
                    self._pool.append(buf)
            self._semaphore.release()
