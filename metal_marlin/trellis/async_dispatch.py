"""Async Metal command buffer manager for batched kernel dispatch.

Eliminates per-kernel waitUntilCompleted() by batching dispatches
into a single command buffer per layer.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._compat import Metal
from ..metal_dispatch import (HAS_METAL, MetalKernelLibrary,
                              _copy_buffer_to_tensor, _CopyBackBuffer)

if TYPE_CHECKING:
    from .model import TrellisModel

_BATCH_TRACE_LOGGER = logging.getLogger("metal_marlin.batch_trace")
_BATCH_TRACE_ENV_VAR = "METAL_MARLIN_TRACE_BATCH"


@dataclass
class _PendingDispatch:
    """A pending kernel dispatch to be encoded."""
    pipeline: Any
    grid: tuple[int, int, int]
    threadgroup: tuple[int, int, int]
    buffers: list[Any]


class AsyncCommandBufferManager:
    """Manages async command buffer for batched MoE dispatch.

    Notes:
        - All dispatches in a batch share one command buffer.
        - Each dispatch gets its own encoder (Metal requirement for different pipelines).
        - Designed for single-threaded decode usage (no locks).
    """

    __slots__ = (
        "lib",
        "_current_cmd_buf",
        "_pending_dispatches",
        "_last_committed_cmd_buf",
        "_committed_current_batch",
        "_batch_active",
        "_current_batch_id",
        "_post_commit_callbacks",
        "_copy_back_buffers",
    )

    def __init__(self, lib: MetalKernelLibrary):
        self.lib = lib
        self._current_cmd_buf: Any | None = None
        self._pending_dispatches: list[_PendingDispatch] = []
        self._last_committed_cmd_buf: Any | None = None
        self._committed_current_batch = False
        self._batch_active = False
        self._current_batch_id: int | None = None
        self._post_commit_callbacks: list[Callable[[], None]] = []
        self._copy_back_buffers: list[_CopyBackBuffer] = []

    def begin_batch(self) -> None:
        """Backward-compatible alias for start_batch()."""
        self.start_batch()

    def start_batch(self) -> None:
        """Begin a new batch of kernel dispatches."""
        if not HAS_METAL or Metal is None:
            raise RuntimeError(
                "Metal is required for AsyncCommandBufferManager.")

        if self._current_cmd_buf is not None:
            raise RuntimeError(
                "A batch is already active. Commit it before beginning a new batch."
            )

        self._current_cmd_buf = self.lib.command_queue.commandBuffer()
        self._pending_dispatches = []
        self._committed_current_batch = False
        self._batch_active = True
        self._post_commit_callbacks.clear()
        if self._current_batch_id is None:
            self._current_batch_id = 1
        else:
            self._current_batch_id += 1

        if os.environ.get(_BATCH_TRACE_ENV_VAR):
            _BATCH_TRACE_LOGGER.info(
                "BATCH START: id=%s", self._current_batch_id)

    def has_active_batch(self) -> bool:
        """Check if a batch is currently active."""
        return self._batch_active

    def dispatch_kernel(
        self,
        function_name: str,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        buffers: list[Any],
    ) -> None:
        """Backward-compatible alias for dispatch()."""
        self.dispatch(function_name, grid, threadgroup, buffers)

    def dispatch(
        self,
        function_name: str,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        buffers: list[Any],
    ) -> None:
        """Queue a kernel dispatch (non-blocking)."""
        if os.environ.get(_BATCH_TRACE_ENV_VAR):
            _BATCH_TRACE_LOGGER.info(
                "DISPATCH: batch_active=%s, id=%s",
                self._batch_active,
                self._current_batch_id,
            )

        if self._current_cmd_buf is None:
            raise RuntimeError("No active batch. Call begin_batch() first.")
        if self._committed_current_batch:
            raise RuntimeError("Current batch is already committed.")

        pipeline = self.lib.get_pipeline(function_name)
        self._pending_dispatches.append(_PendingDispatch(
            pipeline=pipeline,
            grid=grid,
            threadgroup=threadgroup,
            buffers=buffers,
        ))

    def dispatch_immediate(
        self,
        pipeline: Any,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        buffers: list[Any],
    ) -> None:
        """Dispatch kernel and commit immediately (no batching)."""
        if not HAS_METAL or Metal is None:
            raise RuntimeError(
                "Metal is required for AsyncCommandBufferManager.")

        cmd_buffer = self.lib.command_queue.commandBuffer()
        encoder = cmd_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(*grid),
            Metal.MTLSizeMake(*threadgroup),
        )
        encoder.endEncoding()
        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()

    def commit_and_wait(self) -> None:
        """Commit all pending dispatches and wait for completion."""
        if self._current_cmd_buf is not None:
            self.commit_async()

        if self._last_committed_cmd_buf is not None:
            self._last_committed_cmd_buf.waitUntilCompleted()
            self._last_committed_cmd_buf = None

        # Copy back results to tensors
        copy_backs = getattr(self, "_copy_back_buffers", [])
        for cb in copy_backs:
            _copy_buffer_to_tensor(cb.buffer, cb.tensor)
        self._copy_back_buffers = []

        callbacks = tuple(self._post_commit_callbacks)
        self._post_commit_callbacks.clear()
        for callback in callbacks:
            callback()
        self._committed_current_batch = False

    def register_post_commit(self, callback: Callable[[], None]) -> None:
        """Register a callback that runs after commit_and_wait() completes."""
        self._post_commit_callbacks.append(callback)

    def commit_async(self) -> None:
        """Backward-compatible alias for commit_batch()."""
        self.commit_batch()

    def commit_batch(self) -> None:
        """Commit without waiting (for prefetch/pipelining)."""
        if self._current_cmd_buf is None:
            raise RuntimeError("No active batch. Call begin_batch() first.")
        if self._committed_current_batch:
            raise RuntimeError("Current batch is already committed.")
        dispatch_count = len(self._pending_dispatches)

        if os.environ.get(_BATCH_TRACE_ENV_VAR):
            _BATCH_TRACE_LOGGER.info(
                "BATCH COMMIT: id=%s, dispatches=%s",
                self._current_batch_id,
                dispatch_count,
            )

        # Track copy-back buffers for post-wait
        copy_back_buffers: list[_CopyBackBuffer] = []

        # Encode all pending dispatches - each gets its own encoder
        for dispatch in self._pending_dispatches:
            encoder = self._current_cmd_buf.computeCommandEncoder()
            encoder.setComputePipelineState_(dispatch.pipeline)

            for idx, buffer in enumerate(dispatch.buffers):
                # Unwrap _CopyBackBuffer if present
                if isinstance(buffer, _CopyBackBuffer):
                    encoder.setBuffer_offset_atIndex_(buffer.buffer, 0, idx)
                    copy_back_buffers.append(buffer)
                else:
                    encoder.setBuffer_offset_atIndex_(buffer, 0, idx)

            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(*dispatch.grid),
                Metal.MTLSizeMake(*dispatch.threadgroup),
            )
            encoder.endEncoding()

        # Store copy-back buffers for commit_and_wait
        self._copy_back_buffers = copy_back_buffers

        # Commit the command buffer
        if self._pending_dispatches:
            self._current_cmd_buf.commit()
            self._last_committed_cmd_buf = self._current_cmd_buf
        else:
            self._last_committed_cmd_buf = None

        self._current_cmd_buf = None
        self._pending_dispatches = []
        self._committed_current_batch = True
        self._batch_active = False


class LayerBatchContext:
    """Batch command buffers across multiple MoE layers."""

    def __init__(self, model: "TrellisModel", batch_size: int = 8):
        if not HAS_METAL:
            # Return a dummy context manager when Metal is not available
            class DummyContext:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def layer_complete(self): pass
            raise RuntimeError(
                "LayerBatchContext requires Metal (HAS_METAL=True)")

        self.model = model
        self.batch_size = batch_size  # Layers to batch before commit
        self._layer_count = 0
        self._shared_cmd_manager = None
        self._batch_started = False

    def __enter__(self):
        # Get shared command manager from first MoE layer
        # Note: Some models (like GLM-4.7) have dense layers at the beginning
        from .model import TrellisMoEMLP

        shared_manager = None
        for layer in self.model.layers:
            if isinstance(layer.mlp, TrellisMoEMLP):
                shared_manager = layer.mlp._get_async_cmd_manager()
                break

        if shared_manager is None:
            raise RuntimeError("No MoE layers found for batched dispatch")

        self._shared_cmd_manager = shared_manager
        self._shared_cmd_manager.start_batch()
        self._batch_started = True
        return self

    def layer_complete(self):
        """Called after each layer's MoE dispatch."""
        self._layer_count += 1
        if self._layer_count >= self.batch_size:
            self._shared_cmd_manager.commit_and_wait()
            self._shared_cmd_manager.start_batch()
            self._batch_started = True
            self._layer_count = 0

    def ensure_batch_active(self) -> None:
        """Ensure a batch is active, starting one if needed."""
        if self._shared_cmd_manager and not self._shared_cmd_manager.has_active_batch():
            self._shared_cmd_manager.start_batch()
            self._batch_started = True

    def __exit__(self, *args):
        if self._layer_count > 0:
            self._shared_cmd_manager.commit_and_wait()
            self._batch_started = False
