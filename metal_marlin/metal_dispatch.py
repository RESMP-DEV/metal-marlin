"""
Metal kernel dispatcher using PyObjC.

Direct Metal shader dispatch without MLX dependency. Works with PyTorch MPS tensors
by sharing the underlying Metal buffers.

This module provides:
    - MetalKernelLibrary: Compiles and caches Metal shaders from .metal files
    - dispatch_kernel(): Low-level kernel dispatch with argument binding
    - PyTorch MPS tensor <-> Metal buffer interop

Usage:
    from metal_marlin.metal_dispatch import MetalKernelLibrary, dispatch_gemm_fp4

    lib = MetalKernelLibrary.from_source_dir()

    # Dispatch quantized GEMM
    output = dispatch_gemm_fp4(lib, A_mps, B_packed_mps, scales_mps, M, N, K)

Requirements:
    - macOS with Metal support
    - PyObjC: pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders
    - PyTorch with MPS backend

Note:
    This is the preferred path for Apple Silicon inference - no MLX dependency,
    direct control over kernel dispatch and memory management.
"""

from __future__ import annotations

import logging
import os
import weakref
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import _ctypes
import numpy as np

from ._compat import HAS_CPP_EXT, _metal_dispatch_ext
from ._padding import pad_to_multiple, unpad
from .metallib_loader import get_kernel_from_metallib

# Logger for kernel loading diagnostics (metallib vs JIT)
_kernel_logger = logging.getLogger(__name__ + ".kernels")


# GIL release context manager for GPU operations
@contextmanager
def _release_gil():
    """Release the GIL during blocking GPU operations.

    Allows Python threads to run while Metal GPU computation is in progress.
    """
    thread_state = _ctypes.PyEval_SaveThread()
    try:
        yield
    finally:
        _ctypes.PyEval_RestoreThread(thread_state)


# Check PyObjC Metal availability
try:
    import Foundation
    import Metal

    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    Metal = None
    Foundation = None

# Check PyTorch MPS availability
# Use try/except for backends access since torch may be partially loaded during
# circular imports (e.g., pytest importing from _compat before full init)
try:
    import torch

    HAS_TORCH = True
    try:
        HAS_MPS = torch.backends.mps.is_available()
    except AttributeError:
        # torch.backends may not exist during partial imports
        HAS_MPS = False
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False
    torch = None

# Check for C++ extension availability (provides 5-10x dispatch speedup)

# ---------------------------------------------------------------------------
# FastPath: Low-overhead dispatch using C++ extension when available
# ---------------------------------------------------------------------------


class FastPath:
    """Low-overhead Metal kernel dispatch using C++ extension.

    When the C++ extension (_cpp_ext) is available, this class provides
    a fast path that bypasses PyObjC overhead for kernel dispatch. Falls back to
    the standard PyObjC path when the extension is not available.

    The C++ extension eliminates:
    - PyObjC bridge overhead (~50μs per call)
    - Python method dispatch overhead
    - Object allocation for MTLSize creation

    Usage:
        fast_path = FastPath(lib)
        if fast_path.available:
            fast_path.dispatch(pipeline, grid, threadgroup, buffers, wait=True)
        else:
            # Fall back to standard dispatch
            dispatch_kernel(lib, ...)

    Performance:
        - PyObjC path: ~80-150μs per dispatch
        - FastPath: ~5-15μs per dispatch
    """

    __slots__ = ("_lib", "_ctx", "_pipelines", "_available")

    def __init__(self, lib: MetalKernelLibrary):
        """Initialize FastPath for a MetalKernelLibrary.

        Args:
            lib: MetalKernelLibrary to dispatch kernels from.
        """
        self._lib = lib
        self._ctx: Any = None
        self._pipelines: dict[str, Any] = {}
        self._available = HAS_CPP_EXT and _metal_dispatch_ext is not None

        # Create MetalContext and load libraries if available
        if self._available:
            try:
                self._ctx = _metal_dispatch_ext.MetalContext()
                # Load all metallib files that the library knows about
                self._load_metallibs()
            except Exception:
                self._ctx = None
                self._available = False

    def _load_metallibs(self) -> None:
        """Load metallib files into the C++ context."""
        # Get the source directory from the library
        if hasattr(self._lib, "source_dir"):
            source_dir = Path(self._lib.source_dir)
            # Load any .metallib files
            for metallib_file in source_dir.glob("**/*.metallib"):
                try:
                    self._ctx.load_metallib(str(metallib_file))
                except Exception:
                    pass

    @property
    def available(self) -> bool:
        """Return True if the C++ fast path is available."""
        return self._available

    def _get_pipeline(self, kernel_name: str) -> Any:
        """Get or create pipeline from C++ context.

        Args:
            kernel_name: Name of the kernel function.

        Returns:
            MTLComputePipelineState from C++ extension.
        """
        if kernel_name not in self._pipelines:
            # Get the pipeline from C++ extension (searches all loaded libraries)
            self._pipelines[kernel_name] = self._ctx.get_pipeline(kernel_name)
        return self._pipelines[kernel_name]

    def dispatch(
        self,
        kernel_name: str,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        buffers: Sequence[Any],
        offsets: Sequence[int] | None = None,
        wait: bool = True,
    ) -> None:
        """Dispatch a kernel using the fast C++ path.

        Args:
            kernel_name: Name of the kernel function to dispatch.
            grid: Grid dimensions (threadgroups in X, Y, Z).
            threadgroup: Threadgroup dimensions (threads in X, Y, Z).
            buffers: Sequence of ManagedBuffer objects from C++ extension.
            offsets: Optional sequence of byte offsets for each buffer.
            wait: If True, wait for kernel completion.

        Raises:
            RuntimeError: If fast path is not available.
        """
        if not self._available:
            raise RuntimeError(
                "FastPath not available - use standard dispatch")

        pipeline = self._get_pipeline(kernel_name)
        _metal_dispatch_ext.dispatch_kernel(
            self._ctx,
            pipeline,
            grid,
            threadgroup,
            list(buffers),
            wait,
            list(offsets) if offsets else [],
        )

    def dispatch_batched(
        self,
        dispatches: Sequence[tuple[str, tuple[int, int, int], tuple[int, int, int], Sequence[Any]]],
        wait: bool = True,
    ) -> None:
        """Dispatch multiple kernels in a single command buffer.

        Args:
            dispatches: Sequence of (kernel_name, grid, threadgroup, buffers) tuples.
            wait: If True, wait for all kernels to complete.

        Raises:
            RuntimeError: If fast path is not available.
        """
        if not self._available:
            raise RuntimeError(
                "FastPath not available - use standard dispatch")

        batch = _metal_dispatch_ext.BatchDispatch(self._ctx)

        for kernel_name, grid, threadgroup, buffers in dispatches:
            pipeline = self._get_pipeline(kernel_name)
            batch.add_kernel(pipeline, grid, threadgroup, list(buffers))

        batch.commit(wait)

    def batch_mmfp4_gemm(
        self,
        ops: Sequence[tuple[str, Any, Any, Any, Any, int, int, int, int]],
        wait: bool = True,
    ) -> None:
        """Dispatch multiple MMFP4 GEMM kernels in a single command buffer.

        Args:
            ops: Sequence of (kernel_name, A, B, S, C, M, N, K, group_size) tuples.
            wait: If True, wait for all kernels to complete.
        """
        if not self._available:
             raise RuntimeError("FastPath not available")
        
        batch = _metal_dispatch_ext.BatchDispatch(self._ctx)
        
        def to_mb(obj: Any) -> Any:
            if isinstance(obj, _metal_dispatch_ext.ManagedBuffer):
                return obj
            if hasattr(obj, "data_ptr"):
                return self.create_buffer_from_ptr(obj.data_ptr(), obj.nbytes)
            raise TypeError(f"FastPath requires ManagedBuffer or Tensor, got {type(obj)}")

        for op in ops:
            kernel_name, A, B, S, C, M, N, K, group_size = op
            pipeline = self._get_pipeline(kernel_name)
            batch.add_mmfp4_gemm(
                pipeline,
                to_mb(A), to_mb(B), to_mb(S), to_mb(C),
                M, N, K, group_size
            )
            
        batch.commit(wait)

    def batch_int4_gemm(
        self,
        ops: Sequence[tuple[str, Any, Any, Any, Any, Any, int, int, int, int]],
        wait: bool = True,
    ) -> None:
        """Dispatch multiple INT4 GEMM kernels in a single command buffer.

        Args:
            ops: Sequence of (kernel_name, A, B, S, Z, C, M, N, K, group_size) tuples.
            wait: If True, wait for all kernels to complete.
        """
        if not self._available:
             raise RuntimeError("FastPath not available")
        
        batch = _metal_dispatch_ext.BatchDispatch(self._ctx)
        
        def to_mb(obj: Any) -> Any:
            if isinstance(obj, _metal_dispatch_ext.ManagedBuffer):
                return obj
            if hasattr(obj, "data_ptr"):
                return self.create_buffer_from_ptr(obj.data_ptr(), obj.nbytes)
            raise TypeError(f"FastPath requires ManagedBuffer or Tensor, got {type(obj)}")

        for op in ops:
            kernel_name, A, B, S, Z, C, M, N, K, group_size = op
            pipeline = self._get_pipeline(kernel_name)
            batch.add_int4_gemm(
                pipeline,
                to_mb(A), to_mb(B), to_mb(S), to_mb(Z), to_mb(C),
                M, N, K, group_size
            )
            
        batch.commit(wait)

    def mmfp4_gemm(
        self,
        kernel_name: str,
        A: Any,
        B: Any,
        S: Any,
        C: Any,
        M: int,
        N: int,
        K: int,
        group_size: int,
        wait: bool = True,
    ) -> None:
        """Dispatch MMFP4 GEMM using C++ extension."""
        if not self._available:
             raise RuntimeError("FastPath not available")
        
        pipeline = self._get_pipeline(kernel_name)
        
        def to_mb(obj: Any) -> Any:
            if isinstance(obj, _metal_dispatch_ext.ManagedBuffer):
                return obj
            if hasattr(obj, "data_ptr"):
                # Zero-copy wrap of tensor memory
                return self.create_buffer_from_ptr(obj.data_ptr(), obj.nbytes)
            raise TypeError(f"FastPath requires ManagedBuffer or Tensor, got {type(obj)}")

        _metal_dispatch_ext.mmfp4_gemm(
            self._ctx,
            pipeline,
            to_mb(A), to_mb(B), to_mb(S), to_mb(C),
            M, N, K, group_size,
            wait
        )

    def int4_gemm(
        self,
        kernel_name: str,
        A: Any,
        B: Any,
        S: Any,
        Z: Any,
        C: Any,
        M: int,
        N: int,
        K: int,
        group_size: int,
        wait: bool = True,
    ) -> None:
        """Dispatch INT4 GEMM using C++ extension."""
        if not self._available:
             raise RuntimeError("FastPath not available")
        
        pipeline = self._get_pipeline(kernel_name)
        
        def to_mb(obj: Any) -> Any:
            if isinstance(obj, _metal_dispatch_ext.ManagedBuffer):
                return obj
            if hasattr(obj, "data_ptr"):
                # Zero-copy wrap of tensor memory
                return self.create_buffer_from_ptr(obj.data_ptr(), obj.nbytes)
            raise TypeError(f"FastPath requires ManagedBuffer or Tensor, got {type(obj)}")

        _metal_dispatch_ext.int4_gemm(
            self._ctx,
            pipeline,
            to_mb(A), to_mb(B), to_mb(S), to_mb(Z), to_mb(C),
            M, N, K, group_size,
            wait
        )

    def int2_gemm(
        self,
        kernel_name: str,
        A: Any,
        B: Any,
        S: Any,
        C: Any,
        M: int,
        N: int,
        K: int,
        group_size: int,
        wait: bool = True,
    ) -> None:
        """Dispatch INT2 GEMM using C++ extension."""
        if not self._available:
             raise RuntimeError("FastPath not available")
        
        pipeline = self._get_pipeline(kernel_name)
        
        def to_mb(obj: Any) -> Any:
            if isinstance(obj, _metal_dispatch_ext.ManagedBuffer):
                return obj
            if hasattr(obj, "data_ptr"):
                # Zero-copy wrap of tensor memory
                return self.create_buffer_from_ptr(obj.data_ptr(), obj.nbytes)
            raise TypeError(f"FastPath requires ManagedBuffer or Tensor, got {type(obj)}")

        _metal_dispatch_ext.int2_gemm(
            self._ctx,
            pipeline,
            to_mb(A), to_mb(B), to_mb(S), to_mb(C),
            M, N, K, group_size,
            wait
        )

    def create_buffer(self, size: int, use_pool: bool = True) -> Any:
        """Create a Metal buffer using the C++ extension.

        Args:
            size: Buffer size in bytes.
            use_pool: If True, use buffer pool for reuse.

        Returns:
            ManagedBuffer from C++ extension.
        """
        if not self._available:
            raise RuntimeError("FastPath not available")
        return _metal_dispatch_ext.create_buffer(self._ctx, size, use_pool)

    def create_buffer_from_ptr(self, ptr: int, size: int) -> Any:
        """Create a zero-copy Metal buffer from a memory pointer.

        Useful for wrapping MPS tensor memory without copying.

        Args:
            ptr: Memory pointer as integer (e.g., from tensor.data_ptr()).
            size: Buffer size in bytes.

        Returns:
            ManagedBuffer from C++ extension.
        """
        if not self._available:
            raise RuntimeError("FastPath not available")
        return _metal_dispatch_ext.create_buffer_from_ptr(self._ctx, ptr, size)


# Singleton cache for FastPath instances per library
_fast_path_cache: dict[int, FastPath] = {}


def get_fast_path(lib: MetalKernelLibrary) -> FastPath:
    """Get or create a FastPath instance for a MetalKernelLibrary.

    Args:
        lib: MetalKernelLibrary to get fast path for.

    Returns:
        FastPath instance (may or may not be available).
    """
    lib_id = id(lib)
    if lib_id not in _fast_path_cache:
        _fast_path_cache[lib_id] = FastPath(lib)
    return _fast_path_cache[lib_id]


if HAS_METAL:
    from ._buffer_pool import MetalBufferPool, _align_buffer_size

    _STAGING_POOLS: dict[int, MetalBufferPool] = {}
    # Cache Metal buffers by tensor Python id.
    # Store (weakref, buffer) so we can verify the tensor is still alive.
    # This avoids PyTorch tensor __eq__ issues with WeakKeyDictionary.
    _WEIGHT_BUFFER_CACHE: dict[int, tuple[weakref.ref, Any]] = {}
    # Threshold for using async blit (1MB)
    _ASYNC_TRANSFER_THRESHOLD = 1024 * 1024

    def _get_staging_pool(device: Any) -> MetalBufferPool:
        pool = _STAGING_POOLS.get(id(device))
        if pool is None:
            pool = MetalBufferPool(
                device, storage_mode=Metal.MTLResourceStorageModeShared)
            _STAGING_POOLS[id(device)] = pool
        return pool

    _BATCH_THRESHOLD = 4 * 1024
    _BATCH_BUFFER_SIZE = 256 * 1024

    class SmallTransferBatcher:
        """Accumulates and batches small transfers to reduce overhead.

        Small transfers (<4KB) are accumulated into a staging buffer and
        flushed in a single blit operation, reducing transfer count by ~10x.
        """

        __slots__ = (
            "_lib",
            "_device",
            "_staging_buffer",
            "_offset",
            "_pending",
            "_cmd_buffer",
            "_blit",
        )

        def __init__(self, lib: MetalKernelLibrary, device: Any):
            self._lib = lib
            self._device = device
            self._staging_buffer = _get_staging_pool(
                device).get(_BATCH_BUFFER_SIZE)
            self._offset = 0
            self._pending: list[tuple[int, Any, Any, int]] = []
            self._cmd_buffer: Any = None
            self._blit: Any = None

            def add(self, data: bytes, private_buf: Any) -> int:
                """Add a small transfer to the batch.

                Args:
                    data: Bytes to transfer
                    private_buf: Destination private buffer

                Returns:
                    Offset in batch where data was placed
                """
                size = len(data)

                if size > _BATCH_THRESHOLD:
                    raise ValueError(
                        f"Transfer size {size} exceeds batch threshold {_BATCH_THRESHOLD}"
                    )

                # Align offset to cache line to prevent false sharing
                # M3 Max cache line is 128 bytes
                aligned_offset = _round_up(self._offset, 128)

                if aligned_offset + size > _BATCH_BUFFER_SIZE:
                    self.flush()
                    aligned_offset = 0

                contents = self._staging_buffer.contents()
                view = memoryview(contents.as_buffer(
                    self._staging_buffer.length()))
                view[aligned_offset: aligned_offset + size] = data
                self._staging_buffer.didModifyRange_(
                    Foundation.NSMakeRange(aligned_offset, size))

                offset = aligned_offset
                self._pending.append(
                    (offset, self._staging_buffer, private_buf, size))
                self._offset = offset + size
                return offset

        def flush(self) -> None:
            """Flush all pending transfers in a single blit operation."""
            if not self._pending:
                return

            self._cmd_buffer = self._lib.command_queue.commandBuffer()
            self._blit = self._cmd_buffer.blitCommandEncoder()

            for src_offset, staging, dest, size in self._pending:
                self._blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
                    staging, src_offset, dest, 0, size
                )

            self._blit.endEncoding()
            self._cmd_buffer.commit()
            self._cmd_buffer.waitUntilCompleted()

            self._pending.clear()
            self._offset = 0
            self._cmd_buffer = None
            self._blit = None

        def __del__(self) -> None:
            self.flush()
            _get_staging_pool(self._device).release(self._staging_buffer)

    class StagingTransferHandle:
        """Handle for in-flight staging buffer transfers.

        Allows the CPU to continue while the GPU handles the blit copy
        from shared/managed staging buffer to private GPU memory.
        """

        __slots__ = ("_command_buffer", "_staging_buffer",
                     "_private_buffer", "_completed")

        def __init__(self, command_buffer: Any, staging_buffer: Any, private_buffer: Any):
            self._command_buffer = command_buffer
            self._staging_buffer = staging_buffer
            self._private_buffer = private_buffer
            self._completed = False

        @property
        def destination_buffer(self) -> Any:
            """The private GPU buffer being written to."""
            return self._private_buffer

        @property
        def staging_buffer(self) -> Any:
            """The staging buffer (managed/shared)."""
            return self._staging_buffer

        def is_complete(self) -> bool:
            """Check if transfer has completed without blocking."""
            if self._completed:
                return True
            # MTLCommandBuffer.status: 0=notEnqueued, 1=enqueued, 2=committed,
            # 3=scheduled, 4=completed, 5=error
            status = self._command_buffer.status()
            if status >= 4:
                self._completed = True
                return True
            return False

        def wait(self) -> None:
            """Block until transfer completes."""
            if not self._completed:
                self._command_buffer.waitUntilCompleted()
                self._completed = True

    def _blit_copy(lib: MetalKernelLibrary, source: Any, destination: Any, size: int) -> None:
        command_buffer = lib.command_queue.commandBuffer()
        blit = command_buffer.blitCommandEncoder()
        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
            source, 0, destination, 0, size
        )
        blit.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

    def _blit_copy_async(
        lib: MetalKernelLibrary,
        source: Any,
        destination: Any,
        size: int,
        staging_buffer: Any,
    ) -> StagingTransferHandle:
        """Asynchronous blit copy from staging to private buffer.

        Returns a handle that can be waited on later, allowing CPU to continue
        while GPU handles the transfer.

        Args:
            lib: MetalKernelLibrary for command queue access
            source: Source buffer (staging buffer)
            destination: Destination buffer (private GPU memory)
            size: Number of bytes to copy
            staging_buffer: The staging buffer (needed for cleanup)

        Returns:
            StagingTransferHandle for synchronization
        """
        command_buffer = lib.command_queue.commandBuffer()
        blit = command_buffer.blitCommandEncoder()
        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
            source, 0, destination, 0, size
        )
        blit.endEncoding()
        command_buffer.commit()
        return StagingTransferHandle(command_buffer, staging_buffer, destination)

    def _private_buffer_from_bytes(
        lib: MetalKernelLibrary,
        device: Any,
        data: bytes,
    ) -> Any | StagingTransferHandle:
        """Create GPU buffer from CPU bytes with staging for large transfers.

        Small transfers (<1MB): Use shared storage mode for zero-copy access.
        Large transfers (>=1MB): Write to shared staging buffer, then async blit
        to private GPU memory. Returns StagingTransferHandle for async transfers.

        Args:
            lib: MetalKernelLibrary for command queue
            device: MTLDevice
            data: Bytes to transfer

        Returns:
            Shared MTLBuffer (for small transfers) or StagingTransferHandle (for large)
        """
        size = len(data)

        if size < _ASYNC_TRANSFER_THRESHOLD:
            # Align size to 128 bytes
            aligned_size = _round_up(size, 128)
            if aligned_size > size:
                data = data + b"\0" * (aligned_size - size)

            shared_buf = device.newBufferWithBytes_length_options_(
                data, aligned_size, Metal.MTLResourceStorageModeShared
            )
            return shared_buf

        staging_pool = _get_staging_pool(device)
        # Staging pool buffers are already aligned
        staging_buffer = staging_pool.get(size)
        contents = staging_buffer.contents()
        view = memoryview(contents.as_buffer(staging_buffer.length()))
        view[:size] = data
        staging_buffer.didModifyRange_(Foundation.NSMakeRange(0, size))

        # Align shared buffer allocation for zero-copy access
        aligned_size = _align_buffer_size(size)
        private_buffer = device.newBufferWithLength_options_(
            aligned_size, Metal.MTLResourceStorageModeShared
        )
        if private_buffer is None:
            staging_pool.release(staging_buffer)
            raise RuntimeError(
                f"Failed to allocate shared buffer of {aligned_size} bytes")

        return _blit_copy_async(lib, staging_buffer, private_buffer, size, staging_buffer)

    def _private_buffer_from_tensor(
        tensor: torch.Tensor,
        lib: MetalKernelLibrary,
        device: Any,
        *,
        cache: bool,
        async_transfer: bool = True,
    ) -> Any | StagingTransferHandle:
        """Create GPU buffer from PyTorch tensor with staging for large transfers.

        MPS tensors: Always use shared storage (already on GPU).
        CPU tensors: Small transfers use shared storage, large transfers (>=1MB)
        use staging buffer + async blit to private GPU memory.

        Args:
            tensor: PyTorch tensor to convert
            lib: MetalKernelLibrary for command queue
            device: MTLDevice
            cache: Whether to cache the resulting buffer
            async_transfer: Whether to use async transfers for large CPU tensors

        Returns:
            Shared MTLBuffer or StagingTransferHandle for async transfers
        """
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        tensor_id = id(tensor)
        if cache and tensor_id in _WEIGHT_BUFFER_CACHE:
            ref, buf = _WEIGHT_BUFFER_CACHE[tensor_id]
            if ref() is tensor:
                return buf
            del _WEIGHT_BUFFER_CACHE[tensor_id]

        if tensor.is_mps:
            shared_buf = mps_tensor_to_metal_buffer(tensor, device)
            result = shared_buf
        else:
            data = tensor.detach().cpu().numpy().tobytes()
            size = len(data)

            if size < _ASYNC_TRANSFER_THRESHOLD or not async_transfer:
                result = device.newBufferWithBytes_length_options_(
                    data, size, Metal.MTLResourceStorageModeShared
                )
            else:
                staging_pool = _get_staging_pool(device)
                staging_buffer = staging_pool.get(size)
                contents = staging_buffer.contents()
                view = memoryview(contents.as_buffer(staging_buffer.length()))
                view[:size] = data
                staging_buffer.didModifyRange_(Foundation.NSMakeRange(0, size))

                private_buffer = device.newBufferWithLength_options_(
                    size, Metal.MTLResourceStorageModeShared
                )
                if private_buffer is None:
                    staging_pool.release(staging_buffer)
                    raise RuntimeError(
                        f"Failed to allocate shared buffer of {size} bytes")

                result = _blit_copy_async(
                    lib, staging_buffer, private_buffer, size, staging_buffer)

        if cache and not isinstance(result, StagingTransferHandle):
            _WEIGHT_BUFFER_CACHE[tensor_id] = (weakref.ref(tensor), result)

        return result


def require_metal() -> None:
    """Raise if Metal/PyObjC is not available."""
    if not HAS_METAL:
        raise RuntimeError(
            "Metal dispatch requires PyObjC. Install with:\n"
            "  pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders"
        )


def require_mps() -> None:
    """Raise if PyTorch MPS is not available."""
    if not HAS_MPS:
        raise RuntimeError(
            "Metal dispatch requires PyTorch with MPS backend.\n"
            "Ensure you're on Apple Silicon with PyTorch >= 2.0"
        )


def get_gpu_family(device: Any) -> int:
    """Return Apple GPU family (7=M1, 8=M2, 9=M3+)."""
    require_metal()
    apple9 = getattr(Metal, "MTLGPUFamilyApple9", None)
    apple8 = getattr(Metal, "MTLGPUFamilyApple8", None)
    if apple9 is not None and device.supportsFamily_(apple9):
        return 9
    if apple8 is not None and device.supportsFamily_(apple8):
        return 8
    return 7


# ---------------------------------------------------------------------------
# Metal shader source directory
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).parent.parent / "src"


def get_shader_source(name: str) -> str:
    """Load Metal shader source from src/ directory.

    Args:
        name: Shader filename without extension (e.g., 'marlin_gemm')

    Returns:
        Shader source code as string.
    """
    path = _SRC_DIR / f"{name}.metal"
    if not path.exists():
        raise FileNotFoundError(f"Metal shader not found: {path}")
    return path.read_text()


def load_metallib(metallib_path: str | Path | None = None) -> Any:
    """Load precompiled Metal library (.metallib) file.

    This is 100-1000x faster than runtime compilation for kernel dispatch.

    Args:
        metallib_path: Path to .metallib file. If None, uses default location.

    Returns:
        MTLLibrary object with all precompiled kernels.
    """
    require_metal()

    if metallib_path is None:
        # Default location: metal_marlin/lib/metal_marlin.metallib
        metallib_path = Path(__file__).parent / "lib" / "metal_marlin.metallib"

    metallib_path = Path(metallib_path)
    if not metallib_path.exists():
        raise FileNotFoundError(
            f"Precompiled metallib not found: {metallib_path}\n"
            f"Run: ./scripts/build_metallib.sh to generate it."
        )

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("No Metal device available")

    url = Foundation.NSURL.fileURLWithPath_(str(metallib_path))
    library, error = device.newLibraryWithURL_error_(url, None)

    if library is None:
        error_msg = error.localizedDescription() if error else "Unknown error"
        raise RuntimeError(f"Failed to load metallib: {error_msg}")

    return library


# Global cached metallib for load_metallib()
_loaded_metallib: Any = None


def get_loaded_metallib() -> Any:
    """Get cached metallib loaded via load_metallib(), loading if needed.

    Returns None if metallib file doesn't exist (falls back to runtime compilation).
    """
    global _loaded_metallib
    if _loaded_metallib is None:
        try:
            _loaded_metallib = load_metallib()
        except FileNotFoundError:
            return None  # Fall back to runtime compilation
    return _loaded_metallib


# ---------------------------------------------------------------------------
# Metal Kernel Library
# ---------------------------------------------------------------------------


class MetalKernelLibrary:
    """Compiled Metal shader library with kernel caching.

    Manages compilation of .metal source files and provides access to
    compute pipeline states for kernel dispatch.

    Example:
        lib = MetalKernelLibrary.from_source_dir()
        pipeline = lib.get_pipeline("marlin_gemm_fp4")
        # Use pipeline for dispatch...
    """

    def __init__(self, device: Any = None):
        """Initialize with Metal device.

        Args:
            device: MTLDevice instance. If None, uses default system device.
        """
        require_metal()

        if device is None:
            device = Metal.MTLCreateSystemDefaultDevice()
            if device is None:
                raise RuntimeError("No Metal device available")

        self._device = device
        self._libraries: dict[str, Any] = {}  # source_name -> MTLLibrary
        # function_name -> MTLComputePipelineState
        self._pipelines: dict[str, Any] = {}
        self._command_queue = device.newCommandQueue()

        # Secondary command queue for pipeline overlap (prefill/decode)
        # Using separate queues allows GPU to interleave work from both streams
        self._decode_queue = device.newCommandQueue()

        # Small transfer batcher for batching <4KB transfers
        self._small_transfer_batcher: SmallTransferBatcher | None = None

        # Batch dispatch state
        self._batch_mode = False
        self._batch_encoder: Any = None
        self._batch_command_buffer: Any = None
        self._batch_copy_backs: list[Any] = []

        # Pipeline overlap state - tracks in-flight command buffers
        self._inflight_prefill: Any = None
        self._inflight_decode: Any = None

        # Pipelined dispatch state - for async prefill/decode overlap
        self._prefill_encoder: Any = None
        self._prefill_buffer: Any = None
        self._decode_encoder: Any = None
        self._decode_buffer: Any = None

    @classmethod
    def from_source_dir(cls, src_dir: Path | None = None) -> MetalKernelLibrary:
        """Create library and compile all shaders from source directory.

        Args:
            src_dir: Path to Metal source files. Defaults to metal_marlin/src/

        Returns:
            MetalKernelLibrary with all shaders compiled.
        """
        lib = cls()

        if src_dir is None:
            src_dir = _SRC_DIR
        src_dir = Path(src_dir)

        # Compile all .metal files
        for metal_file in sorted(src_dir.glob("*.metal")):
            try:
                lib.compile_source(metal_file.stem, metal_file.read_text())
            except Exception as e:
                print(f"Warning: Failed to compile {metal_file.name}: {e}")

        return lib

    @property
    def device(self) -> Any:
        """The MTLDevice used for compilation and dispatch."""
        return self._device

    @property
    def command_queue(self) -> Any:
        """The MTLCommandQueue for kernel dispatch (primary/prefill)."""
        return self._command_queue

    @property
    def decode_queue(self) -> Any:
        """The MTLCommandQueue for decode operations (secondary stream)."""
        return self._decode_queue

    @contextmanager
    def batch_dispatch(self, wait: bool = True) -> Generator["BatchDispatchState", None, None]:
        """Context manager for batching multiple kernel dispatches.

        Reuses a single command buffer for all dispatches within the context,
        reducing per-dispatch overhead from ~80-150μs to ~5-15μs.

        Args:
            wait: If True, waits for completion on exit. If False, returns
                  a BatchDispatchState that can be waited on later.

        Usage:
            # Synchronous - waits for all kernels on exit
            with lib.batch_dispatch():
                dispatch_kernel(lib, ...)  # Encoded but not committed
                dispatch_kernel(lib, ...)  # Encoded but not committed
            # All kernels committed and waited on exit

            # Asynchronous - returns immediately, wait later
            with lib.batch_dispatch(wait=False) as batch:
                dispatch_kernel(lib, ...)
                dispatch_kernel(lib, ...)
            # ... do other work ...
            batch.wait()  # Block until complete
        """
        self._batch_mode = True
        self._batch_command_buffer = self.command_queue.commandBuffer()
        self._batch_encoder = self._batch_command_buffer.computeCommandEncoder()
        self._batch_copy_backs = []
        
        # Create state object for async waiting
        batch_state = BatchDispatchState(self._batch_command_buffer)
        
        try:
            yield batch_state
        finally:
            self._batch_encoder.endEncoding()
            self._batch_command_buffer.commit()
            
            if wait:
                self._batch_command_buffer.waitUntilCompleted()
                batch_state._completed = True
                
                # Copy back results
                for item in self._batch_copy_backs:
                    _copy_buffer_to_tensor(item.buffer, item.tensor)
                self._batch_copy_backs = []
            
            self._batch_mode = False
            self._batch_encoder = None
            self._batch_command_buffer = None

    # -------------------------------------------------------------------------
    # Pipelined Prefill/Decode Dispatch
    # -------------------------------------------------------------------------
    # These methods enable overlapping prefill and decode work on separate
    # command queues. The GPU can execute decode kernels while encoding
    # prefill work, improving throughput by ~50%.
    #
    # Usage:
    #     # Start prefill for request A (async)
    #     lib.begin_prefill()
    #     lib.dispatch_prefill(kernel, grid, threadgroup, buffers)
    #     lib.commit_prefill()  # Returns immediately, GPU starts working
    #
    #     # Meanwhile, start decode for request B (async)
    #     lib.begin_decode()
    #     lib.dispatch_decode(kernel, grid, threadgroup, buffers)
    #     lib.commit_decode()  # Returns immediately
    #
    #     # Wait for both when needed
    #     lib.wait_prefill()  # Block until prefill done
    #     lib.wait_decode()   # Block until decode done

    def begin_prefill(self) -> None:
        """Begin encoding prefill commands on the primary queue.

        Creates a new command buffer and encoder for prefill work. Call
        dispatch_prefill() to add kernels, then commit_prefill() to submit.
        """
        if self._prefill_buffer is not None:
            raise RuntimeError(
                "Prefill already in progress. Call commit_prefill() first.")
        self._prefill_buffer = self._command_queue.commandBuffer()
        self._prefill_encoder = self._prefill_buffer.computeCommandEncoder()

    def dispatch_prefill(
        self,
        pipeline: Any,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        buffers: Sequence[Any],
    ) -> None:
        """Dispatch a kernel on the prefill stream.

        Args:
            pipeline: MTLComputePipelineState from get_pipeline().
            grid: Threadgroup grid dimensions (x, y, z).
            threadgroup: Threads per threadgroup (x, y, z).
            buffers: Metal buffers to bind as kernel arguments.
        """
        if self._prefill_encoder is None:
            raise RuntimeError("Must call begin_prefill() first")

        self._prefill_encoder.setComputePipelineState_(pipeline)
        for i, buf in enumerate(buffers):
            self._prefill_encoder.setBuffer_offset_atIndex_(buf, 0, i)

        self._prefill_encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(*grid),
            Metal.MTLSizeMake(*threadgroup),
        )

    def commit_prefill(self) -> None:
        """Submit prefill command buffer for execution (non-blocking).

        The GPU will begin executing prefill work immediately. Call
        wait_prefill() to block until completion.
        """
        if self._prefill_encoder is None:
            raise RuntimeError("Must call begin_prefill() first")

        self._prefill_encoder.endEncoding()
        self._prefill_buffer.commit()
        self._inflight_prefill = self._prefill_buffer
        self._prefill_encoder = None
        self._prefill_buffer = None

    def wait_prefill(self) -> None:
        """Block until in-flight prefill work completes."""
        if self._inflight_prefill is not None:
            self._inflight_prefill.waitUntilCompleted()
            self._inflight_prefill = None

    def begin_decode(self) -> None:
        """Begin encoding decode commands on the secondary queue.

        Creates a new command buffer and encoder for decode work. Call
        dispatch_decode() to add kernels, then commit_decode() to submit.
        """
        if self._decode_buffer is not None:
            raise RuntimeError(
                "Decode already in progress. Call commit_decode() first.")
        self._decode_buffer = self._decode_queue.commandBuffer()
        self._decode_encoder = self._decode_buffer.computeCommandEncoder()

    def dispatch_decode(
        self,
        pipeline: Any,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        buffers: Sequence[Any],
    ) -> None:
        """Dispatch a kernel on the decode stream.

        Args:
            pipeline: MTLComputePipelineState from get_pipeline().
            grid: Threadgroup grid dimensions (x, y, z).
            threadgroup: Threads per threadgroup (x, y, z).
            buffers: Metal buffers to bind as kernel arguments.
        """
        if self._decode_encoder is None:
            raise RuntimeError("Must call begin_decode() first")

        self._decode_encoder.setComputePipelineState_(pipeline)
        for i, buf in enumerate(buffers):
            self._decode_encoder.setBuffer_offset_atIndex_(buf, 0, i)

        self._decode_encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(*grid),
            Metal.MTLSizeMake(*threadgroup),
        )

    def commit_decode(self) -> None:
        """Submit decode command buffer for execution (non-blocking).

        The GPU will begin executing decode work immediately. Call
        wait_decode() to block until completion.
        """
        if self._decode_encoder is None:
            raise RuntimeError("Must call begin_decode() first")

        self._decode_encoder.endEncoding()
        self._decode_buffer.commit()
        self._inflight_decode = self._decode_buffer
        self._decode_encoder = None
        self._decode_buffer = None

    def wait_decode(self) -> None:
        """Block until in-flight decode work completes."""
        if self._inflight_decode is not None:
            self._inflight_decode.waitUntilCompleted()
            self._inflight_decode = None

    def has_inflight_prefill(self) -> bool:
        """Check if prefill work is currently in flight."""
        return self._inflight_prefill is not None

    def has_inflight_decode(self) -> bool:
        """Check if decode work is currently in flight."""
        return self._inflight_decode is not None

    def wait_all(self) -> None:
        """Block until all in-flight work completes."""
        self.wait_prefill()
        self.wait_decode()

    def _create_compile_options(self) -> Any:
        """Create MTLCompileOptions with aggressive fast-math flags.

        Environment variables:
            MM_FAST_MATH: Set to "0" to disable fast-math for correctness testing.
                Any other value or unset enables fast-math.
            METAL_MARLIN_DISABLE_FAST_MATH: Set to "1", "true", or "yes" to disable
                all fast-math optimizations for correctness testing.
            METAL_MARLIN_SAFE_MATH: Set to "1", "true", or "yes" to enable a safer
                mode that keeps fast-math but preserves invariance for consistent
                position calculations.

        Fast-math enables:
            - Reciprocal approximations (faster division via rcp ops)
            - Fused multiply-add operations (FMA)
            - Relaxed NaN/Inf handling (assumes no NaN/Inf in hot paths)
            - Reassociation of floating-point operations
            - Denormal flushing to zero (ftz)
            - No signed zeros (nsz)
            - Unsafe math optimizations

        Returns:
            Configured MTLCompileOptions instance.
        """
        options = Metal.MTLCompileOptions.new()

        # Check environment for correctness testing modes
        disable_fast_math = os.getenv("MM_FAST_MATH", "") == "0"
        if not disable_fast_math:
            disable_fast_math = os.getenv("METAL_MARLIN_DISABLE_FAST_MATH", "").lower() in {
                "1",
                "true",
                "yes",
            }
        safe_math = os.getenv("METAL_MARLIN_SAFE_MATH", "").lower() in {
            "1",
            "true",
            "yes",
        }

        fast_math_enabled = not disable_fast_math

        # Fast math: enables reciprocal approximations, FMA, relaxed NaN/Inf handling
        options.setFastMathEnabled_(fast_math_enabled)

        # Preserve invariance: when False, allows aggressive optimizations that may
        # produce different results for the same inputs in different contexts.
        # Disable for maximum performance in inference (no shadow volumes needed).
        # Safe mode keeps invariance for debugging position-sensitive issues.
        if hasattr(options, "setPreserveInvariance_"):
            options.setPreserveInvariance_(safe_math)

        # Optimization level: prioritize performance over binary size.
        # MTLLibraryOptimizationLevelDefault = 0 (balanced)
        # MTLLibraryOptimizationLevelSize = 1 (smaller binaries)
        # MTLLibraryOptimizationLevelPerformance = 2 (faster execution, added in macOS 14+)
        if hasattr(Metal, "MTLLibraryOptimizationLevelPerformance"):
            options.setOptimizationLevel_(
                Metal.MTLLibraryOptimizationLevelPerformance)

        # Metal 3.0 for simdgroup_matrix support
        options.setLanguageVersion_(Metal.MTLLanguageVersion3_0)

        # Set preprocessor macros for shader-level fast-math hints
        # These allow shaders to conditionally compile NaN/Inf checks
        if hasattr(options, "setPreprocessorMacros_"):
            macros = {}
            if fast_math_enabled:
                # Shaders can use #ifdef METAL_FAST_MATH to skip checks
                macros["METAL_FAST_MATH"] = "1"
                # Explicitly disable NaN/Inf checks in shader hot paths
                macros["METAL_DISABLE_NAN_CHECKS"] = "1"
                macros["METAL_DISABLE_INF_CHECKS"] = "1"
                # Enable reciprocal approximation hints
                macros["METAL_USE_RCP_APPROX"] = "1"
                # Flush denormals to zero
                macros["METAL_FLUSH_DENORMALS"] = "1"
            else:
                # Strict mode: enable all checks for correctness testing
                macros["METAL_FAST_MATH"] = "0"
                macros["METAL_STRICT_MATH"] = "1"
            options.setPreprocessorMacros_(macros)

        return options

    def compile_source(self, name: str, source: str) -> Any:
        """Compile Metal source code into a library.

        Args:
            name: Identifier for this source (e.g., 'marlin_gemm')
            source: Metal shader source code

        Returns:
            MTLLibrary instance.
        """
        # Preprocess includes before compilation (PyObjC Metal doesn't resolve #include).
        source = self._preprocess_includes(source)

        options = self._create_compile_options()

        library, error = self._device.newLibraryWithSource_options_error_(
            source, options, None)

        if library is None:
            # Try to get error message
            error_msg = str(error) if error else "Unknown error"
            raise RuntimeError(
                f"Failed to compile Metal source '{name}': {error_msg}")

        self._libraries[name] = library
        return library

    def compile_source_with_defines(
        self, name: str, source: str, defines: dict[str, int | str]
    ) -> Any:
        """Compile Metal source code with preprocessor defines.

        Args:
            name: Identifier for this source (e.g., 'gemm_trellis_moe_256')
            source: Metal shader source code
            defines: Dict of preprocessor macros to define (e.g., {'MOE_SIMDGROUPS_CONFIG': 8})

        Returns:
            MTLLibrary instance.
        """
        # Build preprocessor header from defines
        define_lines = [f"#define {k} {v}" for k, v in defines.items()]
        define_header = "\n".join(define_lines) + "\n" if define_lines else ""

        # Preprocess includes before compilation
        source = self._preprocess_includes(source)

        # Insert defines after metal_stdlib include but before the rest
        # Find the end of the initial #include block
        import re

        include_end = 0
        for match in re.finditer(r"#include\s*<[^>]+>", source):
            include_end = max(include_end, match.end())

        if include_end > 0:
            # Insert defines after includes
            source = source[:include_end] + "\n" + \
                define_header + source[include_end:]
        else:
            # No includes found, prepend defines
            source = define_header + source

        options = self._create_compile_options()

        library, error = self._device.newLibraryWithSource_options_error_(
            source, options, None)

        if library is None:
            error_msg = str(error) if error else "Unknown error"
            raise RuntimeError(
                f"Failed to compile Metal source '{name}': {error_msg}")

        self._libraries[name] = library
        return library

    def _preprocess_includes(self, source: str) -> str:
        """Resolve #include directives by inlining referenced files.

        System includes (<metal_stdlib>, <simd/simd.h>, etc.) are preserved.
        Local includes ("file.metal") are inlined.
        """
        import re

        # Only match local includes with double quotes, not system includes with angle brackets
        local_include_pattern = re.compile(r'#include\s*"([^"]+)"')
        processed: set[str] = set()

        def resolve(src: str, depth: int = 0) -> str:
            if depth > 10:
                raise RuntimeError("Include depth exceeded")

            def replacer(match: re.Match[str]) -> str:
                filename = match.group(1)
                if filename in processed:
                    return ""
                processed.add(filename)

                include_path = _SRC_DIR / filename
                if not include_path.exists():
                    include_path = _SRC_DIR / "fusion" / filename
                if not include_path.exists():
                    raise FileNotFoundError(f"Include not found: {filename}")

                content = include_path.read_text()
                return resolve(content, depth + 1)

            return local_include_pattern.sub(replacer, src)

        return resolve(source)

    def get_pipeline(
        self,
        function_name: str,
        library_name: str | None = None,
    ) -> Any:
        """Get or create compute pipeline for a kernel function.

        Args:
            function_name: Name of the kernel function in the Metal source.
            library_name: Which compiled library contains this function.
                         If None, searches all libraries.

        Returns:
            MTLComputePipelineState for dispatching.
        """
        cache_key = f"{library_name or '*'}::{function_name}"

        if cache_key in self._pipelines:
            return self._pipelines[cache_key]

        # Find the function
        function = None
        if library_name is not None:
            lib = self._libraries.get(library_name)
            if lib is None:
                raise KeyError(f"Library '{library_name}' not compiled")
            function = lib.newFunctionWithName_(function_name)
        else:
            # Search all libraries
            for lib in self._libraries.values():
                function = lib.newFunctionWithName_(function_name)
                if function is not None:
                    break

        if function is None:
            raise KeyError(
                f"Function '{function_name}' not found in any library")

        # Create pipeline state
        pipeline, error = self._device.newComputePipelineStateWithFunction_error_(
            function, None)

        if pipeline is None:
            raise RuntimeError(
                f"Failed to create pipeline for '{function_name}'")

        self._pipelines[cache_key] = pipeline
        return pipeline

    def list_functions(self, library_name: str) -> list[str]:
        """List all function names in a compiled library."""
        lib = self._libraries.get(library_name)
        if lib is None:
            return []
        return list(lib.functionNames())

    def get_kernel(self, library_name: str, function_name: str) -> Any:
        """Get Metal kernel function.

        Tries precompiled metallib first for 100x faster dispatch,
        falling back to JIT compilation if not found.
        """
        cache_key = f"precompiled::{function_name}"

        # Check cache first
        if cache_key in self._pipelines:
            return self._pipelines[cache_key]

        # Try precompiled library first (100x faster)
        kernel_fn = get_kernel_from_metallib(function_name)
        if kernel_fn is not None:
            # Create pipeline state from precompiled function
            pipeline, error = self._device.newComputePipelineStateWithFunction_error_(
                kernel_fn, None
            )
            if pipeline is not None:
                self._pipelines[cache_key] = pipeline
                _kernel_logger.debug(f"[metallib] {function_name}")
                return pipeline

        # Fall back to JIT compilation
        _kernel_logger.debug(f"[jit] {function_name}")
        return self.get_pipeline(function_name, library_name)

    def _get_metal_buffer(self, tensor: torch.Tensor) -> Any:
        """Get MTLBuffer from MPS tensor (zero-copy)."""
        require_mps()

        if not tensor.is_mps:
            raise ValueError("Tensor must be on MPS device")

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Use tensor.data_ptr() to get buffer address for zero-copy interop.
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()

        buffer = self._device.newBufferWithBytesNoCopy_length_options_deallocator_(
            ptr, size, Metal.MTLResourceStorageModeShared, None
        )

        if buffer is None:
            raise RuntimeError("Failed to create Metal buffer from tensor")

        return buffer

    def _dispatch(
        self,
        kernel: Any,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        *args: Any,
    ) -> None:
        """Dispatch a Metal kernel with buffer/constant arguments."""
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(kernel)

        buffer_idx = 0
        texture_idx = 0

        buffers: list[Any] = []
        for arg in args:
            if isinstance(arg, (int, np.integer)):
                const = np.array([int(arg)], dtype=np.uint32)
                buf = self._device.newBufferWithBytes_length_options_(
                    const.tobytes(), const.nbytes, Metal.MTLResourceStorageModeShared
                )
                encoder.setBuffer_offset_atIndex_(buf, 0, buffer_idx)
                buffer_idx += 1
            elif hasattr(arg, "textureType"):
                encoder.setTexture_atIndex_(arg, texture_idx)
                texture_idx += 1
            elif hasattr(arg, "buffer"):
                # Handle _CopyBackBuffer and similar wrappers
                encoder.setBuffer_offset_atIndex_(arg.buffer, 0, buffer_idx)
                buffer_idx += 1
            else:
                encoder.setBuffer_offset_atIndex_(arg, 0, buffer_idx)
                buffer_idx += 1

        grid_size = Metal.MTLSizeMake(*grid)
        tg_size = Metal.MTLSizeMake(*threadgroup)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)

        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

    def fp4_gemm(
        self,
        input: torch.Tensor,  # [M, K] input activations
        weight: torch.Tensor,  # Packed FP4 weights
        scales: torch.Tensor,  # Per-group scales
        N: int,  # Output features
        K: int,  # Input features
        group_size: int = 128,
    ) -> torch.Tensor:
        """Fused FP4 dequantize + GEMM using marlin_gemm kernel."""
        M = input.shape[0]
        orig_N = N

        pad_m = 0
        pad_n = 0
        if _padding_enabled(None):
            packed_k = weight.shape[0] * 8
            packed_n = weight.shape[1]
            scales_k = scales.shape[0] * group_size
            scales_n = scales.shape[1]

            k_target = _round_up(max(K, packed_k, scales_k),
                                 max(_PAD_MULTIPLE, group_size))
            n_target = _round_up(max(N, packed_n, scales_n), _PAD_MULTIPLE)

            input, pad_m = pad_to_multiple(input, 0, _PAD_MULTIPLE)
            input, _ = _pad_tensor_to_size(input, 1, k_target)
            weight = _pad_packed_fp4(weight, k_target, n_target)
            scales = _pad_scales(scales, k_target, n_target, group_size)

            M = input.shape[0]
            N = n_target
            K = k_target
            pad_n = N - orig_N if N >= orig_N else 0

        # Allocate output
        output = torch.empty((M, N), dtype=input.dtype, device=input.device)

        # Get kernel
        kernel = self.get_kernel("marlin_gemm", "marlin_gemm_fp4")

        # Get Metal buffers from MPS tensors
        input_buf = self._get_metal_buffer(input)
        weight_buf = self._get_metal_buffer(weight)
        scales_buf = self._get_metal_buffer(scales)
        output_buf = self._get_metal_buffer(output)

        # Compute grid dimensions (match marlin_gemm.metal)
        grid_m = (M + TILE_M - 1) // TILE_M
        grid_n = (N + TILE_N - 1) // TILE_N

        # Dispatch
        self._dispatch(
            kernel,
            (grid_n, grid_m, 1),
            (THREADS_PER_TG, 1, 1),
            input_buf,
            weight_buf,
            scales_buf,
            output_buf,
            M,
            N,
            K,
            group_size,
        )

        if pad_m or pad_n:
            output = unpad(output, 0, pad_m)
            output = unpad(output, 1, pad_n)
        return output

    def int4_gemm(
        self,
        input: torch.Tensor,  # [M, K] input activations
        weight: torch.Tensor,  # Packed INT4 weights
        scales: torch.Tensor,  # Per-group scales
        N: int,  # Output features
        K: int,  # Input features
        group_size: int = 128,
    ) -> torch.Tensor:
        """Fused INT4 dequantize + GEMM using marlin_gemm kernel."""
        M = input.shape[0]
        orig_N = N

        pad_m = 0
        pad_n = 0
        if _padding_enabled(None):
            packed_k = weight.shape[0] * 8
            packed_n = weight.shape[1]
            scales_k = scales.shape[0] * group_size
            scales_n = scales.shape[1]

            k_target = _round_up(max(K, packed_k, scales_k),
                                 max(_PAD_MULTIPLE, group_size))
            n_target = _round_up(max(N, packed_n, scales_n), _PAD_MULTIPLE)

            input, pad_m = pad_to_multiple(input, 0, _PAD_MULTIPLE)
            input, _ = _pad_tensor_to_size(input, 1, k_target)
            weight = _pad_packed_fp4(weight, k_target, n_target)
            scales = _pad_scales(scales, k_target, n_target, group_size)

            M = input.shape[0]
            N = n_target
            K = k_target
            pad_n = N - orig_N if N >= orig_N else 0

        zeros = torch.full(
            scales.shape,
            8.0,
            dtype=scales.dtype,
            device=scales.device,
        )

        output = torch.empty((M, N), dtype=input.dtype, device=input.device)

        kernel = self.get_kernel("marlin_gemm", "marlin_gemm_fused_u4")

        input_buf = self._get_metal_buffer(input)
        weight_buf = self._get_metal_buffer(weight)
        scales_buf = self._get_metal_buffer(scales)
        zeros_buf = self._get_metal_buffer(zeros)
        output_buf = self._get_metal_buffer(output)

        grid_m = (M + TILE_M - 1) // TILE_M
        grid_n = (N + TILE_N - 1) // TILE_N

        self._dispatch(
            kernel,
            (grid_n, grid_m, 1),
            (THREADS_PER_TG, 1, 1),
            input_buf,
            weight_buf,
            scales_buf,
            zeros_buf,
            output_buf,
            M,
            N,
            K,
            group_size,
        )

        if pad_m or pad_n:
            output = unpad(output, 0, pad_m)
            output = unpad(output, 1, pad_n)
        return output

    def int2_gemm(
        self,
        input: torch.Tensor,  # [M, K] input activations
        weight: torch.Tensor,  # Packed INT2 weights
        scales: torch.Tensor,  # Per-group scales
        N: int,  # Output features
        K: int,  # Input features
        group_size: int = 128,
    ) -> torch.Tensor:
        """Fused INT2 dequantize + GEMM using marlin_gemm kernel."""
        M = input.shape[0]
        orig_N = N

        pad_m = 0
        pad_n = 0
        if _padding_enabled(None):
            packed_k = weight.shape[0]
            packed_n = weight.shape[1] * 16
            scales_k = scales.shape[0] * group_size
            scales_n = scales.shape[1]

            k_target = _round_up(max(K, packed_k, scales_k),
                                 max(_PAD_MULTIPLE, group_size))
            n_target = _round_up(max(N, packed_n, scales_n),
                                 max(_PAD_MULTIPLE, 16))

            input, pad_m = pad_to_multiple(input, 0, _PAD_MULTIPLE)
            input, _ = _pad_tensor_to_size(input, 1, k_target)
            weight = _pad_packed_n(weight, k_target, n_target, 16)
            scales = _pad_scales(scales, k_target, n_target, group_size)

            M = input.shape[0]
            N = n_target
            K = k_target
            pad_n = N - orig_N if N >= orig_N else 0

        output = torch.empty((M, N), dtype=input.dtype, device=input.device)

        kernel = self.get_kernel("marlin_gemm", "marlin_gemm_int2")

        input_buf = self._get_metal_buffer(input)
        weight_buf = self._get_metal_buffer(weight)
        scales_buf = self._get_metal_buffer(scales)
        output_buf = self._get_metal_buffer(output)

        grid_m = (M + TILE_M - 1) // TILE_M
        grid_n = (N + TILE_N - 1) // TILE_N

        self._dispatch(
            kernel,
            (grid_n, grid_m, 1),
            (THREADS_PER_TG, 1, 1),
            input_buf,
            weight_buf,
            scales_buf,
            output_buf,
            M,
            N,
            K,
            group_size,
        )

        if pad_m or pad_n:
            output = unpad(output, 0, pad_m)
            output = unpad(output, 1, pad_n)
        return output

    def get_small_transfer_batcher(self) -> SmallTransferBatcher:
        """Get or create the small transfer batcher.

        Returns:
            SmallTransferBatcher instance for batching <4KB transfers.
        """
        if self._small_transfer_batcher is None:
            self._small_transfer_batcher = SmallTransferBatcher(
                self, self._device)
        return self._small_transfer_batcher

    def flush_small_transfers(self) -> None:
        """Flush pending small transfers from the batcher."""
        if self._small_transfer_batcher is not None:
            self._small_transfer_batcher.flush()

    def create_batched_dispatcher(self) -> BatchedDispatcher:
        """Create a BatchedDispatcher for batching multiple kernel dispatches.

        Returns:
            BatchedDispatcher instance bound to this library.
        """
        return BatchedDispatcher(self)


# ---------------------------------------------------------------------------
# Batched Command Encoding
# ---------------------------------------------------------------------------


class BatchedDispatcher:
    """Batches multiple kernel dispatches into a single command buffer.

    Instead of:
      dispatch(kernel1) -> commit -> wait
      dispatch(kernel2) -> commit -> wait

    Does:
      encoder.dispatch(kernel1)
      encoder.dispatch(kernel2)
      commit -> wait (once)

    Reduces per-dispatch overhead from ~0.05ms to ~0.001ms.

    Example:
        lib = MetalKernelLibrary.from_source_dir()
        bd = lib.create_batched_dispatcher()

        bd.begin()
        bd.dispatch("kernel_a", (8, 8, 1), (32, 1, 1), [buf_a, buf_b])
        bd.dispatch("kernel_b", (4, 4, 1), (64, 1, 1), [buf_c, buf_d])
        count = bd.commit_and_wait()  # Executes both kernels with single commit
    """

    def __init__(self, lib: MetalKernelLibrary):
        """Initialize BatchedDispatcher.

        Args:
            lib: MetalKernelLibrary instance to dispatch kernels from.
        """
        self.lib = lib
        self.queue = lib.command_queue
        self._cmd_buffer: Any = None
        self._encoder: Any = None
        self._dispatches = 0

    def begin(self) -> None:
        """Start a new batched dispatch session.

        Must be called before dispatch(). Creates a new command buffer
        and compute encoder.
        """
        self._cmd_buffer = self.queue.commandBuffer()
        self._encoder = self._cmd_buffer.computeCommandEncoder()
        self._dispatches = 0

    def dispatch(
        self,
        function_name: str,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        buffers: Sequence[Any],
        offsets: Sequence[int] | None = None,
    ) -> None:
        """Queue a kernel dispatch (doesn't execute until commit).

        Args:
            function_name: Name of the kernel function to dispatch.
            grid: Grid dimensions (x, y, z) for threadgroups.
            threadgroup: Threadgroup dimensions (x, y, z).
            buffers: List of Metal buffers to bind to kernel arguments.
            offsets: Optional sequence of byte offsets for each buffer.
        """
        if self._encoder is None:
            raise RuntimeError("Must call begin() before dispatch()")

        pipeline = self.lib.get_pipeline(function_name)
        self._encoder.setComputePipelineState_(pipeline)

        for i, buf in enumerate(buffers):
            offset = offsets[i] if offsets is not None else 0
            self._encoder.setBuffer_offset_atIndex_(buf, offset, i)

        self._encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(*grid),
            Metal.MTLSizeMake(*threadgroup),
        )
        self._dispatches += 1

    def commit_and_wait(self) -> int:
        """Execute all queued dispatches.

        Returns:
            Number of dispatches that were executed.

        Raises:
            RuntimeError: If begin() was not called.
        """
        if self._encoder is None:
            raise RuntimeError("Must call begin() before commit_and_wait()")

        self._encoder.endEncoding()
        self._cmd_buffer.commit()
        self._cmd_buffer.waitUntilCompleted()

        count = self._dispatches
        self._cmd_buffer = None
        self._encoder = None
        self._dispatches = 0
        return count


class BatchDispatchState:
    """State object for async batch dispatch operations.
    
    Allows waiting on a batch of dispatched kernels after the context exits,
    enabling CPU work to overlap with GPU execution.
    
    Example:
        with lib.batch_dispatch(wait=False) as batch:
            dispatch_kernel(lib, "kernel_a", ...)
            dispatch_kernel(lib, "kernel_b", ...)
        
        # GPU is executing while CPU continues here
        do_other_work()
        
        # Wait for GPU to finish when needed
        batch.wait()
    """
    
    __slots__ = ("_command_buffer", "_completed")
    
    def __init__(self, command_buffer: Any):
        self._command_buffer = command_buffer
        self._completed = False
    
    @property
    def completed(self) -> bool:
        """Check if the batch has completed without blocking."""
        if self._completed:
            return True
        # MTLCommandBuffer.status: 0=notEnqueued, 1=enqueued, 2=committed,
        # 3=scheduled, 4=completed, 5=error
        status = self._command_buffer.status()
        if status >= 4:
            self._completed = True
            return True
        return False
    
    def wait(self) -> None:
        """Block until the batch dispatch completes."""
        if not self._completed:
            self._command_buffer.waitUntilCompleted()
            self._completed = True


# ---------------------------------------------------------------------------
# Async Transfer / Compute Overlap
# ---------------------------------------------------------------------------


class AsyncTransferHandle:
    """Handle for an in-flight async transfer operation.

    Provides synchronization and status checking for transfers running
    on a secondary command queue while compute runs on the primary queue.
    """

    __slots__ = ("_command_buffer", "_destination", "_completed")

    def __init__(self, command_buffer: Any, destination: Any):
        self._command_buffer = command_buffer
        self._destination = destination
        self._completed = False

    @property
    def destination_buffer(self) -> Any:
        """The destination Metal buffer being written to."""
        return self._destination

    def is_complete(self) -> bool:
        """Check if the transfer has completed without blocking."""
        if self._completed:
            return True
        # MTLCommandBuffer.status: 0=notEnqueued, 1=enqueued, 2=committed, 3=scheduled, 4=completed, 5=error
        status = self._command_buffer.status()
        if status >= 4:  # completed or error
            self._completed = True
            return True
        return False

    def wait(self) -> None:
        """Block until the transfer completes."""
        if not self._completed:
            self._command_buffer.waitUntilCompleted()
            self._completed = True


class AsyncTransferManager:
    """Manages async data transfers overlapped with GPU compute.

    Uses a secondary command queue for blit (transfer) operations while
    compute kernels run on the primary queue. This enables the pipeline:

        Layer N:   [Compute] ─────────────────────
        Layer N+1:            [Transfer weights] ─[Compute]

    Metal allows concurrent command buffers on different queues to execute
    in parallel when there are no data dependencies.

    Example:
        manager = AsyncTransferManager(lib)

        # Start async transfer for layer N+1 weights
        handle = manager.start_transfer_async(source_buf, size)

        # Execute layer N compute on primary queue
        dispatch_kernel(lib, "layer_n_kernel", ...)

        # Wait for layer N+1 weights to be ready before using them
        handle.wait()
        layer_n_plus_1_weights = handle.destination_buffer
    """

    def __init__(self, lib: MetalKernelLibrary):
        """Initialize AsyncTransferManager.

        Args:
            lib: MetalKernelLibrary providing device and queues.
        """
        self._lib = lib
        self._device = lib.device
        # Use the secondary decode_queue for transfers to overlap with compute
        self._transfer_queue = lib.decode_queue
        # Pool of shared GPU buffers for zero-copy access
        self._buffer_pool: dict[int, list[Any]] = {}

    def _get_private_buffer(self, size: int) -> Any:
        """Get or create a shared GPU buffer of the given size (zero-copy).

        Uses MTLResourceStorageModeShared for zero-copy access between
        CPU and GPU. Uses a pool to avoid allocation overhead for common sizes.
        Aligns to cache line boundary (128 bytes) to prevent false sharing.
        """
        # Align to 128-byte cache line for M3 Max optimization
        aligned_size = _align_buffer_size(size)

        if aligned_size in self._buffer_pool and self._buffer_pool[aligned_size]:
            return self._buffer_pool[aligned_size].pop()

        buffer = self._device.newBufferWithLength_options_(
            aligned_size, Metal.MTLResourceStorageModeShared
        )
        if buffer is None:
            raise RuntimeError(
                f"Failed to allocate shared buffer of size {aligned_size}")
        return buffer

    def _return_buffer(self, buffer: Any, size: int) -> None:
        """Return a buffer to the pool for reuse.

        Size is aligned to cache line boundary (128 bytes) to prevent
        false sharing between adjacent buffers.
        """
        aligned_size = _align_buffer_size(size)
        if aligned_size not in self._buffer_pool:
            self._buffer_pool[aligned_size] = []
        # Limit pool size to avoid memory bloat
        if len(self._buffer_pool[aligned_size]) < 8:
            self._buffer_pool[aligned_size].append(buffer)

    def start_transfer_async(
        self,
        source: Any,
        size: int,
        destination: Any | None = None,
    ) -> AsyncTransferHandle:
        """Start an async blit copy from source to destination buffer.

        The transfer executes on the secondary queue while the primary
        queue continues with compute work.

        Args:
            source: Source MTLBuffer (typically shared/managed storage)
            size: Number of bytes to copy
            destination: Optional destination MTLBuffer. If None, allocates
                        a private buffer from the pool.

        Returns:
            AsyncTransferHandle for synchronization and buffer access.
        """
        if destination is None:
            destination = self._get_private_buffer(size)

        command_buffer = self._transfer_queue.commandBuffer()
        blit = command_buffer.blitCommandEncoder()
        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
            source, 0, destination, 0, size
        )
        blit.endEncoding()
        command_buffer.commit()

        return AsyncTransferHandle(command_buffer, destination)

    def transfer_sync(self, source: Any, size: int, destination: Any | None = None) -> Any:
        """Synchronous transfer (for comparison/fallback).

        Args:
            source: Source MTLBuffer
            size: Bytes to copy
            destination: Optional destination buffer

        Returns:
            Destination MTLBuffer with copied data.
        """
        handle = self.start_transfer_async(source, size, destination)
        handle.wait()
        return handle.destination_buffer


class PipelinedLayerDispatcher:
    """Coordinates pipelined execution across transformer layers.

    Overlaps weight transfers for layer N+1 with compute for layer N:

        Layer 0: [Transfer] [Compute]
        Layer 1:            [Transfer] [Compute]
        Layer 2:                       [Transfer] [Compute]
                 ← hidden state dependency →

    For models with weights already in GPU memory (cached), this has minimal
    benefit. The primary use cases are:
    1. First forward pass with cold weight cache
    2. Layer offloading scenarios
    3. Very large batch prefill where transfer time is significant

    Usage:
        pipeliner = PipelinedLayerDispatcher(lib, num_layers=60)

        # Prefetch layer 0 weights synchronously (need them immediately)
        layer_0_buffers = pipeliner.prefetch_layer_weights(layer_0_tensors)

        for i in range(num_layers):
            # Start async prefetch for next layer while computing current
            if i + 1 < num_layers:
                pipeliner.start_prefetch_async(i + 1, layer_tensors[i + 1])

            # Compute current layer
            output = compute_layer(input, pipeliner.get_layer_buffers(i))

            # Ensure next layer weights are ready before proceeding
            if i + 1 < num_layers:
                pipeliner.wait_for_prefetch(i + 1)
    """

    def __init__(self, lib: MetalKernelLibrary, num_layers: int):
        """Initialize PipelinedLayerDispatcher.

        Args:
            lib: MetalKernelLibrary for buffer allocation and transfers.
            num_layers: Number of transformer layers (for buffer management).
        """
        self._lib = lib
        self._transfer_manager = AsyncTransferManager(lib)
        self._num_layers = num_layers

        # Track in-flight prefetches: layer_idx -> AsyncTransferHandle
        self._inflight: dict[int, list[AsyncTransferHandle]] = {}

        # Cached weight buffers per layer (populated on first prefetch)
        self._layer_buffers: dict[int, dict[str, Any]] = {}

    def prefetch_layer_weights_sync(
        self,
        layer_idx: int,
        weight_tensors: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Synchronously transfer layer weights to private GPU buffers.

        Use this for the first layer that needs to execute immediately.

        Args:
            layer_idx: Layer index for tracking.
            weight_tensors: Dict of weight name -> tensor to transfer.

        Returns:
            Dict of weight name -> Metal buffer ready for compute.
        """
        if layer_idx in self._layer_buffers:
            return self._layer_buffers[layer_idx]

        buffers: dict[str, Any] = {}
        for name, tensor in weight_tensors.items():
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            # Create shared buffer from tensor
            source_buf = mps_tensor_to_metal_buffer(tensor, self._lib.device)
            size = tensor.numel() * tensor.element_size()
            # Transfer to private buffer synchronously
            dest_buf = self._transfer_manager.transfer_sync(source_buf, size)
            buffers[name] = dest_buf

        self._layer_buffers[layer_idx] = buffers
        return buffers

    def start_prefetch_async(
        self,
        layer_idx: int,
        weight_tensors: dict[str, torch.Tensor],
    ) -> None:
        """Start async prefetch of layer weights.

        Call this while the previous layer is computing. The transfers
        run on a secondary queue overlapped with compute.

        Args:
            layer_idx: Layer index being prefetched.
            weight_tensors: Dict of weight name -> tensor to transfer.
        """
        if layer_idx in self._layer_buffers:
            # Already cached, nothing to do
            return

        handles: list[AsyncTransferHandle] = []
        buffers: dict[str, Any] = {}

        for name, tensor in weight_tensors.items():
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            source_buf = mps_tensor_to_metal_buffer(tensor, self._lib.device)
            size = tensor.numel() * tensor.element_size()
            handle = self._transfer_manager.start_transfer_async(
                source_buf, size)
            handles.append(handle)
            buffers[name] = handle.destination_buffer

        self._inflight[layer_idx] = handles
        self._layer_buffers[layer_idx] = buffers

    def wait_for_prefetch(self, layer_idx: int) -> dict[str, Any]:
        """Wait for async prefetch to complete and return buffers.

        Call this before using a layer's weights.

        Args:
            layer_idx: Layer index to wait for.

        Returns:
            Dict of weight name -> Metal buffer ready for compute.
        """
        if layer_idx in self._inflight:
            for handle in self._inflight[layer_idx]:
                handle.wait()
            del self._inflight[layer_idx]

        return self._layer_buffers.get(layer_idx, {})

    def get_layer_buffers(self, layer_idx: int) -> dict[str, Any]:
        """Get cached buffers for a layer (must be prefetched first).

        Args:
            layer_idx: Layer index.

        Returns:
            Dict of weight name -> Metal buffer.

        Raises:
            KeyError: If layer has not been prefetched.
        """
        if layer_idx not in self._layer_buffers:
            raise KeyError(f"Layer {layer_idx} has not been prefetched")
        return self._layer_buffers[layer_idx]

    def clear_layer_cache(self, layer_idx: int) -> None:
        """Clear cached buffers for a layer to free memory.

        Use for layer offloading scenarios where layers are cycled.

        Args:
            layer_idx: Layer index to clear.
        """
        if layer_idx in self._layer_buffers:
            del self._layer_buffers[layer_idx]
        if layer_idx in self._inflight:
            # Wait for any in-flight transfers first
            for handle in self._inflight[layer_idx]:
                handle.wait()
            del self._inflight[layer_idx]

    def clear_all(self) -> None:
        """Clear all cached buffers and pending transfers."""
        # Wait for all in-flight transfers
        for layer_idx in list(self._inflight.keys()):
            for handle in self._inflight[layer_idx]:
                handle.wait()
        self._inflight.clear()
        self._layer_buffers.clear()


# ---------------------------------------------------------------------------
# PyTorch MPS <-> Metal buffer interop
# ---------------------------------------------------------------------------


class _CopyBackBuffer:
    """Wrapper for buffers that need to be copied back into a tensor."""

    __slots__ = ("buffer", "tensor")

    def __init__(self, buffer: Any, tensor: torch.Tensor) -> None:
        self.buffer = buffer
        self.tensor = tensor


def _torch_dtype_to_numpy(dtype: torch.dtype) -> np.dtype:
    mapping = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.int8: np.int8,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
    }
    np_dtype = mapping.get(dtype)
    if np_dtype is None:
        # Fallback for less common dtypes (e.g., bfloat16) via CPU tensor.
        np_dtype = torch.empty((), dtype=dtype, device="cpu").numpy().dtype
    return np_dtype


def _copy_buffer_to_tensor(buffer: Any, tensor: torch.Tensor) -> None:
    contents = buffer.contents()
    length = buffer.length()
    np_dtype = _torch_dtype_to_numpy(tensor.dtype)
    arr = np.frombuffer(contents.as_buffer(length), dtype=np_dtype)
    # Truncate to tensor size (buffer may be page-aligned and larger)
    expected_size = tensor.numel()
    if arr.size > expected_size:
        arr = arr[:expected_size]
    arr = arr.reshape(tuple(tensor.shape)).copy()
    tensor.copy_(torch.from_numpy(arr).to(device=tensor.device))


def mps_tensor_to_metal_buffer(
    tensor: torch.Tensor, device: Any, *, copy_back: bool = False
) -> Any:
    """Get Metal buffer from PyTorch MPS tensor.

    Prefers zero-copy interop; falls back to a shared buffer copy when PyObjC
    cannot wrap the MPS device pointer. Set copy_back=True for output tensors.

    Args:
        tensor: PyTorch tensor on MPS device
        device: MTLDevice (must match MPS device)

    Returns:
        MTLBuffer or a copy-back wrapper for output tensors.
    """
    require_mps()

    if not tensor.is_mps:
        raise ValueError("Tensor must be on MPS device")

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    size = tensor.numel() * tensor.element_size()

    # Ensure pending MPS ops are done before we access tensor data.
    torch.mps.synchronize()

    try:
        ptr = tensor.data_ptr()
        buffer = device.newBufferWithBytesNoCopy_length_options_deallocator_(
            ptr, size, Metal.MTLResourceStorageModeShared, None
        )
        if buffer is not None:
            return buffer
    except Exception:
        buffer = None

    # PyObjC cannot reliably wrap the MPS device pointer. Fall back to a shared buffer.
    if copy_back:
        aligned_size = _align_buffer_size(size)
        buffer = device.newBufferWithLength_options_(
            aligned_size, Metal.MTLResourceStorageModeShared
        )
        if buffer is None:
            raise RuntimeError(
                "Failed to create Metal buffer for output tensor")
        return _CopyBackBuffer(buffer, tensor)

    # Handle dtype conversion for numpy compatibility
    # BFloat16 is not supported by numpy, convert to float16
    cpu_tensor = tensor.detach().cpu()
    if cpu_tensor.dtype == torch.bfloat16:
        cpu_tensor = cpu_tensor.to(torch.float16)
    arr = cpu_tensor.numpy()

    aligned_size = _align_buffer_size(arr.nbytes)
    data = arr.tobytes()
    if aligned_size > len(data):
        data = data + b"\0" * (aligned_size - len(data))

    buffer = device.newBufferWithBytes_length_options_(
        data, aligned_size, Metal.MTLResourceStorageModeShared
    )
    if buffer is None:
        raise RuntimeError("Failed to create Metal buffer from tensor data")
    return buffer


def numpy_array_to_metal_buffer(arr: np.ndarray, device: Any) -> Any:
    """Create Metal buffer from numpy array.

    Lowest-level buffer creation - use when you already have numpy data.
    Aligns allocation to 128 bytes.

    Args:
        arr: Numpy array (must be contiguous)
        device: MTLDevice to create buffer on

    Returns:
        MTLBuffer containing array data
    """
    require_mps()

    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    aligned_size = _align_buffer_size(arr.nbytes)
    data = arr.tobytes()
    if aligned_size > len(data):
        data = data + b"\0" * (aligned_size - len(data))

    buffer = device.newBufferWithBytes_length_options_(
        data, aligned_size, Metal.MTLResourceStorageModeShared
    )
    if buffer is None:
        raise RuntimeError(
            f"Failed to create Metal buffer from numpy array shape={arr.shape}")
    return buffer


def cpu_tensor_to_metal_buffer(tensor: torch.Tensor, device: Any) -> Any:
    """Create Metal buffer directly from CPU tensor.

    Bypasses MPS entirely to avoid double-copy when MPS zero-copy fails.
    More memory efficient for large static weights.

    Args:
        tensor: PyTorch tensor on CPU (NOT MPS)
        device: MTLDevice to create buffer on

    Returns:
        MTLBuffer containing tensor data
    """
    require_mps()

    if tensor.is_mps or tensor.is_cuda:
        raise ValueError("Tensor must be on CPU, not GPU")

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

        # Convert to numpy and create buffer
        # BFloat16 not supported by numpy, convert to float16
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float16)

        arr = tensor.numpy()
        aligned_size = _align_buffer_size(arr.nbytes)
        data = arr.tobytes()
        if aligned_size > len(data):
            data = data + b"\0" * (aligned_size - len(data))

        buffer = device.newBufferWithBytes_length_options_(
            data, aligned_size, Metal.MTLResourceStorageModeShared
        )
        if buffer is None:
            raise RuntimeError(
                f"Failed to create Metal buffer from CPU tensor shape={tensor.shape}"
            )
        return buffer


def cpu_tensor_to_metal_texture(tensor: torch.Tensor, device: Any) -> Any:
    """Create Metal texture 2D (height=1) directly from CPU tensor.

    Args:
        tensor: PyTorch tensor on CPU (NOT MPS). Must be float32 or float16.
        device: MTLDevice to create texture on.

    Returns:
        MTLTexture containing tensor data.
    """
    require_mps()

    if tensor.is_mps or tensor.is_cuda:
        raise ValueError("Tensor must be on CPU, not GPU")

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # Determine pixel format
    if tensor.dtype == torch.float32:
        pixel_format = Metal.MTLPixelFormatR32Float
    elif tensor.dtype == torch.float16:
        pixel_format = Metal.MTLPixelFormatR16Float
    else:
        raise ValueError(f"Unsupported dtype for texture: {tensor.dtype}")

    width = tensor.numel()

    # Use Texture2D with height 1 as requested
    descriptor = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
        pixel_format, width, 1, False
    )

    texture = device.newTextureWithDescriptor_(descriptor)
    if texture is None:
        raise RuntimeError(
            f"Failed to create Metal texture from tensor shape={tensor.shape}")

    # Copy data to texture
    region = Metal.MTLRegionMake2D(0, 0, width, 1)

    # BFloat16 not supported by numpy
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float16)

    arr = tensor.numpy()
    data = arr.tobytes()

    # Bytes per row is just total bytes since height is 1
    bytes_per_row = len(data)

    texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
        region, 0, data, bytes_per_row
    )

    return texture


def mps_tensors_to_metal_buffers(
    tensors: list[torch.Tensor], device: Any, *, copy_back: bool = False
) -> list[Any]:
    """Batch create Metal buffers from multiple PyTorch MPS tensors.

    Reduces Metal API overhead by batching buffer creation operations.
    Instead of:
        buf_a = mps_tensor_to_metal_buffer(a, device)
        buf_b = mps_tensor_to_metal_buffer(b, device)
        buf_c = mps_tensor_to_metal_buffer(c, device)

    Use:
        bufs = mps_tensors_to_metal_buffers([a, b, c], device)

    Args:
        tensors: List of PyTorch tensors on MPS device
        device: MTLDevice (must match MPS device)
        copy_back: If True, create buffers that support copy-back for outputs

    Returns:
        List of MTLBuffer objects corresponding to input tensors
    """
    require_mps()

    # Synchronize once for all tensors
    if tensors:
        torch.mps.synchronize()

    buffers: list[Any] = []
    for tensor in tensors:
        if not tensor.is_mps:
            raise ValueError("All tensors must be on MPS device")

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        size = tensor.numel() * tensor.element_size()
        buffer: Any = None

        try:
            ptr = tensor.data_ptr()
            buffer = device.newBufferWithBytesNoCopy_length_options_deallocator_(
                ptr, size, Metal.MTLResourceStorageModeShared, None
            )
        except Exception:
            buffer = None

        if buffer is None:
            # Fall back to buffer copy
            if copy_back:
                aligned_size = _align_buffer_size(size)
                buffer = device.newBufferWithLength_options_(
                    aligned_size, Metal.MTLResourceStorageModeShared
                )
                if buffer is None:
                    raise RuntimeError("Failed to create Metal buffer")
                buffer = _CopyBackBuffer(buffer, tensor)
            else:
                cpu_tensor = tensor.detach().cpu()
                if cpu_tensor.dtype == torch.bfloat16:
                    cpu_tensor = cpu_tensor.to(torch.float16)
                arr = cpu_tensor.numpy()
                aligned_size = _align_buffer_size(arr.nbytes)
                data = arr.tobytes()
                if aligned_size > len(data):
                    data = data + b"\0" * (aligned_size - len(data))

                buffer = device.newBufferWithBytes_length_options_(
                    data, aligned_size, Metal.MTLResourceStorageModeShared
                )
                if buffer is None:
                    raise RuntimeError(
                        "Failed to create Metal buffer from tensor data")

        buffers.append(buffer)

    return buffers


def cpu_tensors_to_metal_buffers(tensors: list[torch.Tensor], device: Any) -> list[Any]:
    """Batch create Metal buffers from multiple CPU tensors.

    Reduces Metal API overhead by batching buffer creation operations.
    More memory efficient for large static weights.

    Args:
        tensors: List of PyTorch tensors on CPU (NOT MPS or CUDA)
        device: MTLDevice to create buffer on

    Returns:
        List of MTLBuffer objects corresponding to input tensors
    """
    require_mps()

    buffers: list[Any] = []
    for tensor in tensors:
        if tensor.is_mps or tensor.is_cuda:
            raise ValueError("All tensors must be on CPU, not GPU")

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # BFloat16 not supported by numpy, convert to float16
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float16)

        arr = tensor.numpy()
        buffer = device.newBufferWithBytes_length_options_(
            arr.tobytes(), arr.nbytes, Metal.MTLResourceStorageModeShared
        )
        if buffer is None:
            raise RuntimeError(
                f"Failed to create Metal buffer from CPU tensor shape={tensor.shape}"
            )
        buffers.append(buffer)

    return buffers


def metal_buffer_to_numpy(buffer: Any, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    """Read Metal buffer contents to numpy array.

    Args:
        buffer: MTLBuffer
        dtype: numpy dtype for interpretation
        shape: desired array shape

    Returns:
        numpy array (copy of buffer data)
    """
    require_metal()

    # Get raw bytes
    contents = buffer.contents()
    length = buffer.length()

    # Create numpy array from buffer
    arr = np.frombuffer(contents.as_buffer(length), dtype=dtype)
    return arr.reshape(shape).copy()


# ---------------------------------------------------------------------------
# Kernel dispatch helpers
# ---------------------------------------------------------------------------


def dispatch_kernel(
    lib: MetalKernelLibrary,
    function_name: str,
    grid: tuple[int, int, int],
    threadgroup: tuple[int, int, int],
    buffers: Sequence[Any],
    wait: bool = False,
    offsets: Sequence[int] | None = None,
    textures: Sequence[Any] | None = None,
) -> Any:
    """Dispatch a Metal compute kernel.

    Uses FastPath (C++ extension) when available for 5-10x lower dispatch overhead.
    Falls back to PyObjC path when extension is not available or in batch mode.

    Args:
        lib: MetalKernelLibrary with compiled shaders
        function_name: Kernel function to dispatch
        grid: Grid dimensions (threadgroups in X, Y, Z)
        threadgroup: Threadgroup dimensions (threads in X, Y, Z)
        buffers: Sequence of MTLBuffer arguments (in order)
        wait: If True, wait for kernel completion
        offsets: Optional sequence of byte offsets for each buffer.
        textures: Optional sequence of MTLTexture arguments (in order).

    Returns:
        None if wait=True or in batch mode, otherwise the command buffer.
    """
    pipeline = lib.get_pipeline(function_name)

    # Check for _CopyBackBuffer in buffers (requires Python path for copy-back)
    copy_back: list[_CopyBackBuffer] = []
    has_copy_back = False
    for buf in buffers:
        if isinstance(buf, _CopyBackBuffer):
            has_copy_back = True
            copy_back.append(buf)

    # Try FastPath when:
    # - Not in batch mode (batch mode uses shared encoder)
    # - No copy-back buffers (requires Python-side copy after wait)
    # - No textures (FastPath currently doesn't support textures)
    # - C++ extension available
    if not lib._batch_mode and not has_copy_back and not textures:
        fast_path = get_fast_path(lib)
        if fast_path.available:
            try:
                return fast_path.dispatch(
                    function_name, grid, threadgroup, buffers, offsets, wait
                )
            except Exception:
                # Fall back to Python path on any error
                pass

    # Standard PyObjC dispatch path

    # Use batch encoder if in batch mode
    if lib._batch_mode and lib._batch_encoder is not None:
        encoder = lib._batch_encoder
        # Don't create new command buffer - use the batched one
        in_batch = True
    else:
        command_buffer = lib.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        in_batch = False

    encoder.setComputePipelineState_(pipeline)

    # Bind buffers
    for i, buf in enumerate(buffers):
        offset = offsets[i] if offsets is not None else 0
        if isinstance(buf, _CopyBackBuffer):
            encoder.setBuffer_offset_atIndex_(buf.buffer, offset, i)
        else:
            encoder.setBuffer_offset_atIndex_(buf, offset, i)

    # Bind textures
    if textures:
        for i, tex in enumerate(textures):
            encoder.setTexture_atIndex_(tex, i)

    # Dispatch
    grid_size = Metal.MTLSizeMake(*grid)
    tg_size = Metal.MTLSizeMake(*threadgroup)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)

    # If in batch mode, don't end/commit - the context manager handles it
    if lib._batch_mode:
        if copy_back:
            lib._batch_copy_backs.extend(copy_back)
        return None

    encoder.endEncoding()
    command_buffer.commit()

    if wait:
        command_buffer.waitUntilCompleted()
        for item in copy_back:
            _copy_buffer_to_tensor(item.buffer, item.tensor)
        return None
    return command_buffer


def dispatch_kernel_indirect(
    lib: MetalKernelLibrary,
    function_name: str,
    expert_counts: torch.Tensor,
    threadgroup: tuple[int, int, int],
    buffers: Sequence[Any],
    wait: bool = False,
    offsets: Sequence[int] | None = None,
) -> Any:
    """Dispatch Metal kernel with indirect command buffer for dynamic expert selection.

    Instead of CPU deciding grid dimensions, GPU reads expert counts from buffer
    and determines how many threadgroups to launch per expert dynamically.

    Args:
        lib: MetalKernelLibrary with compiled shaders
        function_name: Kernel function to dispatch
        expert_counts: [num_experts] tensor with token count per expert (int32)
        threadgroup: Threadgroup dimensions (threads in X, Y, Z)
        buffers: Sequence of MTLBuffer arguments (in order)
        wait: If True, wait for kernel completion
        offsets: Optional sequence of byte offsets for each buffer.

    Returns:
        None if wait=True or in batch mode, otherwise the command buffer.

    Note:
        The expert_counts buffer format for MTLDispatchThreadgroupsIndirectArguments:
        struct {
            uint32_t threadgroupsPerGrid[3];  // X, Y, Z threadgroups
        };
        For MoE: threadgroupsPerGrid[0] = (expert_tokens + threads_per_tg - 1) / threads_per_tg
    """
    require_mps()
    pipeline = lib.get_pipeline(function_name)
    device = lib.device

    # Convert expert counts to indirect dispatch args
    # MTLDispatchThreadgroupsIndirectArguments is 3 uint32s (12 bytes per expert)
    num_experts = expert_counts.shape[0]
    threads_per_tg = threadgroup[0] * threadgroup[1] * threadgroup[2]

    # Create indirect command buffer: for each expert, compute threadgroup count
    # Format: [threadgroups_x, threadgroups_y, threadgroups_z] per expert
    indirect_args = torch.zeros(
        num_experts, 3, dtype=torch.uint32, device="mps")

    # Calculate threadgroups per expert: ceil(expert_count / threads_per_tg)
    expert_counts_uint = expert_counts.to(torch.uint32)
    indirect_args[:, 0] = (expert_counts_uint +
                           threads_per_tg - 1) // threads_per_tg
    indirect_args[:, 1] = 1
    indirect_args[:, 2] = 1

    # Convert to Metal buffer
    indirect_buf = mps_tensor_to_metal_buffer(
        indirect_args.contiguous(), device)

    # Use batch encoder if in batch mode
    if lib._batch_mode and lib._batch_encoder is not None:
        encoder = lib._batch_encoder
        in_batch = True
    else:
        command_buffer = lib.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        in_batch = False

    encoder.setComputePipelineState_(pipeline)

    # Bind buffers
    copy_back: list[_CopyBackBuffer] = []
    for i, buf in enumerate(buffers):
        offset = offsets[i] if offsets is not None else 0
        if isinstance(buf, _CopyBackBuffer):
            encoder.setBuffer_offset_atIndex_(buf.buffer, offset, i)
            copy_back.append(buf)
        else:
            encoder.setBuffer_offset_atIndex_(buf, offset, i)

    # Dispatch with indirect buffer - GPU reads threadgroup counts
    tg_size = Metal.MTLSizeMake(*threadgroup)

    # Dispatch each expert with its own threadgroup count
    for expert_id in range(num_experts):
        offset_bytes = expert_id * 12  # 3 uint32s per expert
        encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup_(
            indirect_buf, offset_bytes, tg_size
        )

    # If in batch mode, don't end/commit - the context manager handles it
    if lib._batch_mode:
        if copy_back:
            lib._batch_copy_backs.extend(copy_back)
        return None

    encoder.endEncoding()
    command_buffer.commit()

    if wait:
        command_buffer.waitUntilCompleted()
        for item in copy_back:
            _copy_buffer_to_tensor(item.buffer, item.tensor)
        return None
    return command_buffer


# ---------------------------------------------------------------------------
# High-level GEMM dispatch
# ---------------------------------------------------------------------------

# Tile dimensions (must match marlin_gemm.metal)
TILE_M = 64
TILE_N = 64
TILE_K = 32
THREADS_PER_TG = 128
FP32_ACCUM_K_THRESHOLD = 256

# Padding config (set METAL_MARLIN_GEMM_PADDING=0 to disable)
_ENABLE_GEMM_PADDING = os.getenv("METAL_MARLIN_GEMM_PADDING", "1").lower() not in (
    "0",
    "false",
    "no",
)
_PAD_MULTIPLE = 8


def _padding_enabled(override: bool | None) -> bool:
    if override is None:
        return _ENABLE_GEMM_PADDING
    return override


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _pad_tensor_to_size(tensor: torch.Tensor, dim: int, size: int) -> tuple[torch.Tensor, int]:
    dim = dim % tensor.dim()
    current = tensor.size(dim)
    if current == size:
        return tensor, 0
    if current > size:
        raise ValueError(
            f"Cannot pad dim {dim} from {current} to smaller size {size}")
    pad_size = size - current
    new_shape = list(tensor.shape)
    new_shape[dim] = size
    padded = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
    slices = [slice(None)] * tensor.dim()
    slices[dim] = slice(0, current)
    padded[tuple(slices)] = tensor
    return padded, pad_size


def _pad_scales(
    scales: torch.Tensor,
    k_target: int,
    n_target: int,
    group_size: int,
) -> torch.Tensor:
    if k_target % group_size != 0:
        raise ValueError(
            f"k_target={k_target} must be divisible by group_size={group_size} for scales"
        )
    k_groups_target = k_target // group_size
    if scales.shape[0] > k_groups_target or scales.shape[1] > n_target:
        raise ValueError(
            "scales shape is larger than requested padded dimensions: "
            f"{scales.shape} vs ({k_groups_target}, {n_target})"
        )
    padded = torch.zeros(
        (k_groups_target, n_target),
        dtype=scales.dtype,
        device=scales.device,
    )
    padded[: scales.shape[0], : scales.shape[1]] = scales
    return padded


def _pad_packed_fp4(
    packed: torch.Tensor,
    k_target: int,
    n_target: int,
) -> torch.Tensor:
    k_packs_target = k_target // 8
    if packed.shape[0] > k_packs_target or packed.shape[1] > n_target:
        raise ValueError(
            "packed FP4 shape is larger than requested padded dimensions: "
            f"{packed.shape} vs ({k_packs_target}, {n_target})"
        )
    padded = torch.zeros(
        (k_packs_target, n_target),
        dtype=packed.dtype,
        device=packed.device,
    )
    padded[: packed.shape[0], : packed.shape[1]] = packed
    return padded


def _pad_packed_n(
    packed: torch.Tensor,
    k_target: int,
    n_target: int,
    pack_factor: int,
) -> torch.Tensor:
    n_packed_target = n_target // pack_factor
    if packed.shape[0] > k_target or packed.shape[1] > n_packed_target:
        raise ValueError(
            "packed shape is larger than requested padded dimensions: "
            f"{packed.shape} vs ({k_target}, {n_packed_target})"
        )
    padded = torch.zeros(
        (k_target, n_packed_target),
        dtype=packed.dtype,
        device=packed.device,
    )
    padded[: packed.shape[0], : packed.shape[1]] = packed
    return padded


def _dispatch_decode_gemv_fp4(
    lib: MetalKernelLibrary,
    A: torch.Tensor,
    B_packed: torch.Tensor,
    scales: torch.Tensor,
    M: int,
    N: int,
    K: int,
    group_size: int = 32,
    enable_padding: bool | None = None,
) -> torch.Tensor:
    """Dispatch FP4 decode GEMV (M=1) using optimized decode kernel.

    The decode_gemv_fp4_wide kernel is optimized for single-token inference:
    - TILE_N = 512 columns per threadgroup (vs 64 for standard GEMM)
    - 4 columns per thread for better instruction-level parallelism
    - No wasted compute on M-padding (M is always 1)

    Expected speedup vs marlin_gemm_fp4 for M=1: ~3-4x

    Kernel signature:
        decode_gemv_fp4_wide(
            device const half* A,      // [1, K] - buffer 0
            device const uint* B,      // [K/8, N] - buffer 1
            device const half* scales, // [K/group_size, N] - buffer 2
            device half* C,            // [1, N] - buffer 3
            constant uint& M,          // buffer 4
            constant uint& N,          // buffer 5
            constant uint& K,          // buffer 6
            constant uint& group_size  // buffer 7
        )
    """
    device = lib.device
    orig_N = N
    pad_n = 0

    # Decode kernel uses TILE_N = 512
    DECODE_TILE_N = 512

    if _padding_enabled(enable_padding):
        packed_k = B_packed.shape[0] * 8
        packed_n = B_packed.shape[1]
        scales_k = scales.shape[0] * group_size
        scales_n = scales.shape[1]

        k_target = _round_up(max(K, packed_k, scales_k),
                             max(_PAD_MULTIPLE, group_size))
        n_target = _round_up(max(N, packed_n, scales_n), DECODE_TILE_N)

        A, _ = _pad_tensor_to_size(A, 1, k_target)
        B_packed = _pad_packed_fp4(B_packed, k_target, n_target)
        scales = _pad_scales(scales, k_target, n_target, group_size)

        N = n_target
        K = k_target
        pad_n = N - orig_N if N >= orig_N else 0

    # Allocate output
    C = torch.empty((M, N), dtype=torch.float16, device="mps")

    # Convert tensors to Metal buffers
    A_half = A.half().contiguous()
    A_buf = _private_buffer_from_tensor(A_half, lib, device, cache=False)
    B_packed_contig = B_packed.contiguous()
    B_buf = _private_buffer_from_tensor(
        B_packed_contig, lib, device, cache=True)
    scales_half = scales if scales.dtype == torch.float16 else scales.half()
    scales_half = scales_half.contiguous()
    S_buf = _private_buffer_from_tensor(scales_half, lib, device, cache=True)
    C_buf = mps_tensor_to_metal_buffer(C, device, copy_back=True)

    # Create param buffers for constant values
    M_buf = _private_buffer_from_bytes(
        lib, device, np.array([M], dtype=np.uint32).tobytes())
    N_buf = _private_buffer_from_bytes(
        lib, device, np.array([N], dtype=np.uint32).tobytes())
    K_buf = _private_buffer_from_bytes(
        lib, device, np.array([K], dtype=np.uint32).tobytes())
    gs_buf = _private_buffer_from_bytes(
        lib, device, np.array([group_size], dtype=np.uint32).tobytes()
    )

    # Compute grid: each threadgroup handles DECODE_TILE_N columns
    grid_n = (N + DECODE_TILE_N - 1) // DECODE_TILE_N

    # Dispatch decode kernel
    dispatch_kernel(
        lib,
        function_name="decode_gemv_fp4_wide",
        grid=(grid_n, 1, 1),
        threadgroup=(128, 1, 1),
        buffers=[A_buf, B_buf, S_buf, C_buf, M_buf, N_buf, K_buf, gs_buf],
        wait=True,
    )

    if pad_n:
        C = unpad(C, 1, pad_n)
    return C


def dispatch_gemm_fp4(
    lib: MetalKernelLibrary,
    A: torch.Tensor,
    B_packed: torch.Tensor,
    scales: torch.Tensor,
    M: int,
    N: int,
    K: int,
    group_size: int = 32,
    enable_padding: bool | None = None,
) -> torch.Tensor:
    """Dispatch FP4 quantized GEMM: C = A @ dequant(B).

    Args:
        lib: MetalKernelLibrary with marlin_gemm compiled
        A: Input activations [M, K], fp16/bf16, MPS tensor
        B_packed: Packed FP4 weights [(K+pad)//8, N+pad], uint32, MPS
        scales: Per-group scales [K//group_size, N], fp16, MPS
        M: Rows of A (batch * seq for transformers)
        N: Columns of B (output features)
        K: Inner dimension (input features)
        group_size: Quantization group size
        enable_padding: Override padding config (None uses env default)

    Returns:
        Output tensor [M, N], fp16, MPS
    """
    require_mps()

    device = lib.device

    # === DECODE OPTIMIZATION: Use GEMV kernel for M=1 ===
    # The standard marlin_gemm_fp4 kernel uses 64x64 tiles which is catastrophically
    # inefficient for decode (M=1): 98.4% of compute is wasted on zero padding.
    # The decode_gemv_fp4_wide kernel uses TILE_N=512 with 4 cols/thread for ~3-4x speedup.
    if M == 1:
        return _dispatch_decode_gemv_fp4(
            lib, A, B_packed, scales, M, N, K, group_size, enable_padding
        )

    gpu_family = get_gpu_family(device)
    if K > FP32_ACCUM_K_THRESHOLD:
        kernel_name = (
            "marlin_gemm_fused_fp4_fp32acc" if gpu_family >= 9 else "marlin_gemm_fp4_fp32acc"
        )
    else:
        kernel_name = "marlin_gemm_fused_fp4" if gpu_family >= 9 else "marlin_gemm_fp4"

    pad_m_multiple = _PAD_MULTIPLE
    pad_n_multiple = _PAD_MULTIPLE
    if kernel_name == "marlin_gemm_fused_fp4":
        # Avoid partial-tile stores in fused kernel until boundary handling is fully verified.
        pad_m_multiple = max(_PAD_MULTIPLE, TILE_M)
        pad_n_multiple = max(_PAD_MULTIPLE, TILE_N)

    orig_N = N
    pad_m = 0
    pad_n = 0
    if _padding_enabled(enable_padding):
        packed_k = B_packed.shape[0] * 8
        packed_n = B_packed.shape[1]
        scales_k = scales.shape[0] * group_size
        scales_n = scales.shape[1]

        k_target = _round_up(max(K, packed_k, scales_k),
                             max(_PAD_MULTIPLE, group_size))
        n_target = _round_up(max(N, packed_n, scales_n), pad_n_multiple)

        A, pad_m = pad_to_multiple(A, 0, pad_m_multiple)
        A, _ = _pad_tensor_to_size(A, 1, k_target)
        B_packed = _pad_packed_fp4(B_packed, k_target, n_target)
        scales = _pad_scales(scales, k_target, n_target, group_size)

        M = A.shape[0]
        N = n_target
        K = k_target
        pad_n = N - orig_N if N >= orig_N else 0

    # Allocate output
    C = torch.empty((M, N), dtype=torch.float16, device="mps")

    # Convert tensors to Metal buffers
    A_half = A.half().contiguous()
    A_buf = _private_buffer_from_tensor(A_half, lib, device, cache=False)
    B_packed_contig = B_packed.contiguous()
    B_buf = _private_buffer_from_tensor(
        B_packed_contig, lib, device, cache=True)
    scales_half = scales if scales.dtype == torch.float16 else scales.half()
    scales_half = scales_half.contiguous()
    S_buf = _private_buffer_from_tensor(scales_half, lib, device, cache=True)
    C_buf = mps_tensor_to_metal_buffer(C, device, copy_back=True)

    # Create separate param buffers (kernel expects bufffers at indices 4, 5, 6, 7)
    M_buf = _private_buffer_from_bytes(
        lib, device, np.array([M], dtype=np.uint32).tobytes())
    N_buf = _private_buffer_from_bytes(
        lib, device, np.array([N], dtype=np.uint32).tobytes())
    K_buf = _private_buffer_from_bytes(
        lib, device, np.array([K], dtype=np.uint32).tobytes())
    gs_buf = _private_buffer_from_bytes(
        lib, device, np.array([group_size], dtype=np.uint32).tobytes()
    )

    # Compute grid
    grid_m = (M + TILE_M - 1) // TILE_M
    grid_n = (N + TILE_N - 1) // TILE_N

    # Dispatch
    fast_path = get_fast_path(lib)
    if fast_path.available and hasattr(fast_path, "mmfp4_gemm"):
        fast_path.mmfp4_gemm(
            kernel_name,
            A_buf, B_buf, S_buf, C_buf,
            M, N, K, group_size,
            wait=True
        )
    else:
        dispatch_kernel(
            lib,
            function_name=kernel_name,
            grid=(grid_n, grid_m, 1),
            threadgroup=(THREADS_PER_TG, 1, 1),
            buffers=[A_buf, B_buf, S_buf, C_buf, M_buf, N_buf, K_buf, gs_buf],
            wait=True,
        )

    if pad_m or pad_n:
        C = unpad(C, 0, pad_m)
        C = unpad(C, 1, pad_n)
    return C


def dispatch_gemm_fp8(
    lib: MetalKernelLibrary,
    A: torch.Tensor,
    B_packed: torch.Tensor,
    scales: torch.Tensor,
    M: int,
    N: int,
    K: int,
    group_size: int = 128,
    enable_padding: bool | None = None,
) -> torch.Tensor:
    """Dispatch FP8 E4M3 quantized GEMM: C = A @ dequant(B).

    Similar to dispatch_gemm_fp4 but for FP8 weights.
    """
    require_mps()

    device = lib.device

    orig_N = N
    pad_m = 0
    pad_n = 0
    if _padding_enabled(enable_padding):
        packed_k = B_packed.shape[0]
        packed_n = B_packed.shape[1] * 4
        scales_k = scales.shape[0] * group_size
        scales_n = scales.shape[1]

        k_target = _round_up(max(K, packed_k, scales_k),
                             max(_PAD_MULTIPLE, group_size))
        n_target = _round_up(max(N, packed_n, scales_n), max(_PAD_MULTIPLE, 4))

        A, pad_m = pad_to_multiple(A, 0, _PAD_MULTIPLE)
        A, _ = _pad_tensor_to_size(A, 1, k_target)
        B_packed = _pad_packed_n(B_packed, k_target, n_target, 4)
        scales = _pad_scales(scales, k_target, n_target, group_size)

        M = A.shape[0]
        N = n_target
        K = k_target
        pad_n = N - orig_N if N >= orig_N else 0

    C = torch.empty((M, N), dtype=torch.float16, device="mps")

    A_half = A.half().contiguous()
    A_buf = _private_buffer_from_tensor(A_half, lib, device, cache=False)
    B_packed_contig = B_packed.contiguous()
    B_buf = _private_buffer_from_tensor(
        B_packed_contig, lib, device, cache=True)
    scales_half = scales if scales.dtype == torch.float16 else scales.half()
    scales_half = scales_half.contiguous()
    S_buf = _private_buffer_from_tensor(scales_half, lib, device, cache=True)
    C_buf = mps_tensor_to_metal_buffer(C, device, copy_back=True)

    params = np.array([M, N, K, group_size], dtype=np.uint32)
    params_buf = _private_buffer_from_bytes(lib, device, params.tobytes())

    grid_m = (M + TILE_M - 1) // TILE_M
    grid_n = (N + TILE_N - 1) // TILE_N

    dispatch_kernel(
        lib,
        function_name="marlin_gemm_fp8_e4m3",
        grid=(grid_m, grid_n, 1),
        threadgroup=(THREADS_PER_TG, 1, 1),
        buffers=[A_buf, B_buf, S_buf, C_buf, params_buf],
        wait=True,
    )

    if pad_m or pad_n:
        C = unpad(C, 0, pad_m)
        C = unpad(C, 1, pad_n)
    return C


def dispatch_gemm_int2(
    lib: MetalKernelLibrary,
    A: torch.Tensor,
    B_packed: torch.Tensor,
    scales: torch.Tensor,
    M: int,
    N: int,
    K: int,
    group_size: int = 128,
    enable_padding: bool | None = None,
) -> torch.Tensor:
    """Dispatch INT2 quantized GEMM: C = A @ dequant(B).

    For cold MoE experts with extreme compression.
    Uses PyTorch fallback since Metal INT2 kernel is not yet implemented.
    """
    require_mps()

    # PyTorch fallback for INT2 GEMM (Metal kernel not yet implemented)
    # INT2 packs 16 values per uint32 along N dimension: [K, N//16]
    A_2d = A.reshape(M, K).to(torch.float32)
    device = A.device
    K_actual, N_packed = B_packed.shape
    N_full = N_packed * 16  # Each uint32 holds 16 INT2 values

    # Unpack INT2 to FP32: [K, N]
    B_full = torch.empty((K_actual, N_full),
                         device=device, dtype=torch.float32)
    for j in range(16):  # 16 values per uint32, along N dimension
        nibbles = ((B_packed >> (j * 2)) & 0x3).to(torch.int32)
        # INT2 codebook: 0->-1.5, 1->-0.5, 2->0.5, 3->1.5 (symmetric)
        vals = nibbles.float() - 1.5  # Map to [-1.5, 1.5] range
        B_full[:, j::16] = vals

    # Trim to actual N if padded
    if N_full > N:
        B_full = B_full[:, :N]

    # Apply scales: [K//group_size, N] -> [K, N]
    scales_f = scales.to(torch.float32)
    scales_exp = scales_f.repeat_interleave(group_size, dim=0)
    if scales_exp.shape[0] > K_actual:
        scales_exp = scales_exp[:K_actual, :]
    if scales_exp.shape[1] > N:
        scales_exp = scales_exp[:, :N]
    B_full = B_full[:K_actual, :N] * scales_exp[:K_actual, :N]

    out = (A_2d[:, :K_actual] @ B_full).to(torch.float16)
    return out


def _dispatch_gemm_int2_metal(
    lib: MetalKernelLibrary,
    A: torch.Tensor,
    B_packed: torch.Tensor,
    scales: torch.Tensor,
    M: int,
    N: int,
    K: int,
    group_size: int = 128,
    enable_padding: bool | None = None,
) -> torch.Tensor:
    """Metal kernel dispatch for INT2 GEMM (not yet implemented)."""
    require_mps()

    device = lib.device

    orig_N = N
    pad_m = 0
    pad_n = 0
    if _padding_enabled(enable_padding):
        packed_k = B_packed.shape[0]
        packed_n = B_packed.shape[1] * 16
        scales_k = scales.shape[0] * group_size
        scales_n = scales.shape[1]

        k_target = _round_up(max(K, packed_k, scales_k),
                             max(_PAD_MULTIPLE, group_size))
        n_target = _round_up(max(N, packed_n, scales_n),
                             max(_PAD_MULTIPLE, 16))

        A, pad_m = pad_to_multiple(A, 0, _PAD_MULTIPLE)
        A, _ = _pad_tensor_to_size(A, 1, k_target)
        B_packed = _pad_packed_n(B_packed, k_target, n_target, 16)
        scales = _pad_scales(scales, k_target, n_target, group_size)

        M = A.shape[0]
        N = n_target
        K = k_target
        pad_n = N - orig_N if N >= orig_N else 0

    C = torch.empty((M, N), dtype=torch.float16, device="mps")

    A_half = A.half().contiguous()
    A_buf = _private_buffer_from_tensor(A_half, lib, device, cache=False)
    B_packed_contig = B_packed.contiguous()
    B_buf = _private_buffer_from_tensor(
        B_packed_contig, lib, device, cache=True)
    scales_half = scales if scales.dtype == torch.float16 else scales.half()
    scales_half = scales_half.contiguous()
    S_buf = _private_buffer_from_tensor(scales_half, lib, device, cache=True)
    C_buf = mps_tensor_to_metal_buffer(C, device, copy_back=True)

    params = np.array([M, N, K, group_size], dtype=np.uint32)
    params_buf = _private_buffer_from_bytes(lib, device, params.tobytes())

    grid_m = (M + TILE_M - 1) // TILE_M
    grid_n = (N + TILE_N - 1) // TILE_N

    # Use INT2 dequant kernel
    # Note: This may need a fused GEMM variant if not available
    dispatch_kernel(
        lib,
        function_name="marlin_gemm_int2",  # Would need to add this to .metal
        grid=(grid_m, grid_n, 1),
        threadgroup=(THREADS_PER_TG, 1, 1),
        buffers=[A_buf, B_buf, S_buf, C_buf, params_buf],
        wait=True,
    )

    if pad_m or pad_n:
        C = unpad(C, 0, pad_m)
        C = unpad(C, 1, pad_n)
    return C


# ---------------------------------------------------------------------------
# FP4 Dequantization dispatch
# ---------------------------------------------------------------------------

# Optimal dequant kernel parameters (must match dequant.metal)
DEQUANT_OPT_THREADS = 128
DEQUANT_OPT_PACKS_PER_THREAD = 4
DEQUANT_OPT_PACKS_PER_TG = DEQUANT_OPT_THREADS * DEQUANT_OPT_PACKS_PER_THREAD


def dispatch_dequant_fp4(
    lib: MetalKernelLibrary,
    packed: torch.Tensor,
    scales: torch.Tensor,
    K: int,
    N: int,
    group_size: int = 32,
) -> torch.Tensor:
    """Dispatch FP4 dequantization: output = dequant(packed, scales).

    Uses the row-major optimized kernel for best performance on [K, N] output.

    Args:
        lib: MetalKernelLibrary with dequant compiled
        packed: Packed FP4 weights [K/8, N], uint32, MPS tensor
        scales: Per-group scales [K/group_size, N], fp16, MPS
        K: Number of elements in reduction dimension
        N: Number of output columns
        group_size: Quantization group size

    Returns:
        Dequantized tensor [K, N], fp16, MPS
    """
    require_mps()

    device = lib.device

    # Allocate output
    output = torch.empty((K, N), dtype=torch.float16, device="mps")

    # Convert tensors to Metal buffers
    packed_buf = mps_tensor_to_metal_buffer(packed.contiguous(), device)
    scales_buf = mps_tensor_to_metal_buffer(scales.half().contiguous(), device)
    output_buf = mps_tensor_to_metal_buffer(output, device, copy_back=True)

    # Create constant buffers
    K_buf = device.newBufferWithBytes_length_options_(
        np.array([K], dtype=np.uint32).tobytes(
        ), 4, Metal.MTLResourceStorageModeShared
    )
    N_buf = device.newBufferWithBytes_length_options_(
        np.array([N], dtype=np.uint32).tobytes(
        ), 4, Metal.MTLResourceStorageModeShared
    )
    gs_buf = device.newBufferWithBytes_length_options_(
        np.array([group_size], dtype=np.uint32).tobytes(
        ), 4, Metal.MTLResourceStorageModeShared
    )

    # Grid dimensions for row-major kernel
    # Each threadgroup handles ROWMAJOR_N_TILE (128) columns and 8 K-rows
    ROWMAJOR_N_TILE = 128
    grid_n = (N + ROWMAJOR_N_TILE - 1) // ROWMAJOR_N_TILE
    grid_k = (K + 7) // 8  # K/8 blocks

    dispatch_kernel(
        lib,
        function_name="dequant_fp4_optimal_rowmajor",
        grid=(grid_n, grid_k, 1),
        threadgroup=(128, 1, 1),
        buffers=[packed_buf, scales_buf, output_buf, K_buf, N_buf, gs_buf],
        wait=True,
    )

    return output


def dispatch_dequant_fp4_linear(
    lib: MetalKernelLibrary,
    packed: torch.Tensor,
    scales: torch.Tensor,
    num_packed: int,
    group_size: int = 32,
) -> torch.Tensor:
    """Dispatch linear FP4 dequantization for bandwidth benchmarking.

    Uses the simdgroup-optimal kernel designed for maximum throughput.

    Args:
        lib: MetalKernelLibrary with dequant compiled
        packed: Packed FP4 weights [num_packed], uint32, MPS tensor
        scales: Per-group scales [num_groups], fp16, MPS
        num_packed: Number of packed uint32s
        group_size: Quantization group size

    Returns:
        Dequantized tensor [num_packed * 8], fp16, MPS
    """
    require_mps()

    device = lib.device

    # Allocate output (8 FP16 values per packed uint32)
    num_elements = num_packed * 8
    output = torch.empty(num_elements, dtype=torch.float16, device="mps")

    # Convert tensors to Metal buffers
    packed_buf = mps_tensor_to_metal_buffer(packed.contiguous(), device)
    scales_buf = mps_tensor_to_metal_buffer(scales.half().contiguous(), device)
    output_buf = mps_tensor_to_metal_buffer(output, device, copy_back=True)

    # Create constant buffers
    np_buf = device.newBufferWithBytes_length_options_(
        np.array([num_packed], dtype=np.uint32).tobytes(
        ), 4, Metal.MTLResourceStorageModeShared
    )
    gs_buf = device.newBufferWithBytes_length_options_(
        np.array([group_size], dtype=np.uint32).tobytes(
        ), 4, Metal.MTLResourceStorageModeShared
    )

    # Grid: 1D, each threadgroup processes DEQUANT_OPT_PACKS_PER_TG packed words
    num_threadgroups = (
        num_packed + DEQUANT_OPT_PACKS_PER_TG - 1) // DEQUANT_OPT_PACKS_PER_TG

    dispatch_kernel(
        lib,
        function_name="dequant_fp4_simdgroup_optimal",
        grid=(num_threadgroups, 1, 1),
        threadgroup=(DEQUANT_OPT_THREADS, 1, 1),
        buffers=[packed_buf, scales_buf, output_buf, np_buf, gs_buf],
        wait=True,
    )

    return output


def dispatch_dequant_fp4_bandwidth_max(
    lib: MetalKernelLibrary,
    packed: torch.Tensor,
    scales: torch.Tensor,
    num_packed: int,
    group_size: int = 32,
) -> torch.Tensor:
    """Dispatch maximum-bandwidth FP4 dequantization kernel.

    Requirements:
        - num_packed must be divisible by 4 (for uint4 loads)
        - All buffers should be 16-byte aligned

    Args:
        lib: MetalKernelLibrary with dequant compiled
        packed: Packed FP4 weights [num_packed], uint32, MPS tensor
        scales: Per-group scales [num_groups], fp16, MPS
        num_packed: Number of packed uint32s (must be divisible by 4)
        group_size: Quantization group size

    Returns:
        Dequantized tensor [num_packed * 8], fp16, MPS
    """
    require_mps()

    if num_packed % 4 != 0:
        raise ValueError(
            "num_packed must be divisible by 4 for bandwidth_max kernel")

    device = lib.device

    # Allocate output
    num_elements = num_packed * 8
    output = torch.empty(num_elements, dtype=torch.float16, device="mps")

    # Convert tensors to Metal buffers
    packed_buf = mps_tensor_to_metal_buffer(packed.contiguous(), device)
    scales_buf = mps_tensor_to_metal_buffer(scales.half().contiguous(), device)
    output_buf = mps_tensor_to_metal_buffer(output, device, copy_back=True)

    # Kernel expects num_packed / 4
    num_packed_div4 = num_packed // 4
    np_buf = device.newBufferWithBytes_length_options_(
        np.array([num_packed_div4], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    gs_buf = device.newBufferWithBytes_length_options_(
        np.array([group_size], dtype=np.uint32).tobytes(
        ), 4, Metal.MTLResourceStorageModeShared
    )

    # Grid: 1D, each thread processes one uint4 (4 packed words)
    num_threadgroups = (
        num_packed_div4 + DEQUANT_OPT_THREADS - 1) // DEQUANT_OPT_THREADS

    dispatch_kernel(
        lib,
        function_name="dequant_fp4_bandwidth_max",
        grid=(num_threadgroups, 1, 1),
        threadgroup=(DEQUANT_OPT_THREADS, 1, 1),
        buffers=[packed_buf, scales_buf, output_buf, np_buf, gs_buf],
        wait=True,
    )

    return output


def benchmark_dequant_fp4(
    lib: MetalKernelLibrary,
    # 16M packed = 128M FP16 values = 256 MB output
    num_packed: int = 1024 * 1024 * 16,
    group_size: int = 32,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
) -> dict[str, float]:
    """Benchmark FP4 dequantization bandwidth.

    Measures effective memory bandwidth for the optimal dequant kernels.

    Args:
        lib: MetalKernelLibrary with dequant compiled
        num_packed: Number of packed uint32s to process
        group_size: Quantization group size
        warmup_iters: Number of warmup iterations
        benchmark_iters: Number of timed iterations

    Returns:
        Dictionary with:
            - 'time_ms': Average kernel time in milliseconds
            - 'bandwidth_gb_s': Effective memory bandwidth in GB/s
            - 'throughput_gop_s': Dequant throughput in billion ops/sec
    """
    require_mps()
    import time

    # Round up to multiple of 512 for clean tile alignment
    num_packed = ((num_packed + 511) // 512) * 512
    num_groups = (num_packed * 8 + group_size - 1) // group_size

    # Create test data
    packed = torch.randint(0, 2**32, (num_packed,), dtype=torch.int32, device="mps").view(
        torch.uint32
    )
    scales = torch.randn(num_groups, dtype=torch.float16,
                         device="mps") * 0.1 + 0.5

    # Warmup
    for _ in range(warmup_iters):
        _ = dispatch_dequant_fp4_bandwidth_max(
            lib, packed, scales, num_packed, group_size)
    torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(benchmark_iters):
        _ = dispatch_dequant_fp4_bandwidth_max(
            lib, packed, scales, num_packed, group_size)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / benchmark_iters) * 1000

    # Calculate bandwidth
    # Read: packed (4 bytes/pack) + scales (2 bytes/group, amortized)
    # Write: output (8 * 2 bytes/pack = 16 bytes/pack)
    # Total: ~20 bytes effective per pack (dominated by writes)
    read_bytes = num_packed * 4 + num_groups * 2
    write_bytes = num_packed * 8 * 2
    total_bytes = read_bytes + write_bytes
    bandwidth_gb_s = (total_bytes / (avg_time_ms / 1000)) / 1e9

    # Throughput: 8 FP4 values dequantized per packed word
    num_values = num_packed * 8
    throughput_gop_s = (num_values / (avg_time_ms / 1000)) / 1e9

    return {
        "time_ms": avg_time_ms,
        "bandwidth_gb_s": bandwidth_gb_s,
        "throughput_gop_s": throughput_gop_s,
        "num_values": num_values,
        "output_mb": write_bytes / 1e6,
    }


# ---------------------------------------------------------------------------
# MoE Dispatch Functions
# ---------------------------------------------------------------------------

# MoE tile dimensions (must match moe_dispatch_optimized.metal)
MOE_TILE_M = 16
MOE_TILE_N = 64
MOE_THREADS_PER_TG = 64


def dispatch_moe_optimized(
    lib: MetalKernelLibrary,
    activations: torch.Tensor,
    router_weights: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_scales: torch.Tensor,
    shared_weights: torch.Tensor | None,
    shared_scales: torch.Tensor | None,
    batch_size: int,
    hidden_dim: int,
    out_dim: int,
    num_experts: int = 64,
    top_k: int = 4,
    group_size: int = 128,
) -> torch.Tensor:
    """Dispatch ultra-optimized MoE forward pass.

    Single-kernel fused routing + GEMM + combination for minimal overhead.
    Optimized for GLM-4.7-Flash: 64 experts, top-k=4, shared expert.

    Args:
        lib: MetalKernelLibrary with moe_dispatch_optimized compiled
        activations: Input [batch, hidden_dim], fp16, MPS tensor
        router_weights: Router weights [hidden_dim, num_experts], fp16, MPS
        expert_weights: Packed FP4 expert weights [num_experts, K/8, N], uint32, MPS
        expert_scales: Expert scales [num_experts, K/group_size, N], fp16, MPS
        shared_weights: Optional shared expert weights [K/8, N], uint32, MPS
        shared_scales: Optional shared expert scales [K/group_size, N], fp16, MPS
        batch_size: Number of tokens
        hidden_dim: Input dimension
        out_dim: Output dimension
        num_experts: Total number of experts (default 64)
        top_k: Experts per token (default 4)
        group_size: Quantization group size (default 128)

    Returns:
        Output tensor [batch, out_dim], fp16, MPS
    """
    require_mps()

    device = lib.device

    # Allocate output
    output = torch.empty((batch_size, out_dim),
                         dtype=torch.float16, device="mps")

    # Convert tensors to Metal buffers
    act_buf = mps_tensor_to_metal_buffer(
        activations.half().contiguous(), device)
    router_buf = mps_tensor_to_metal_buffer(
        router_weights.half().contiguous(), device)
    expert_w_buf = mps_tensor_to_metal_buffer(
        expert_weights.contiguous(), device)
    expert_s_buf = mps_tensor_to_metal_buffer(
        expert_scales.half().contiguous(), device)

    has_shared = 1 if shared_weights is not None else 0
    if shared_weights is not None:
        shared_w_buf = mps_tensor_to_metal_buffer(
            shared_weights.contiguous(), device)
        shared_s_buf = mps_tensor_to_metal_buffer(
            shared_scales.half().contiguous(), device)
    else:
        # Create dummy buffers for unused shared expert
        shared_w_buf = device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared)
        shared_s_buf = device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared)

    out_buf = mps_tensor_to_metal_buffer(output, device, copy_back=True)

    # Create params buffer
    # struct MoEOptParams { batch_size, hidden_dim, out_dim, num_experts, top_k, group_size, has_shared }
    params = np.array(
        [batch_size, hidden_dim, out_dim, num_experts,
            top_k, group_size, has_shared],
        dtype=np.uint32,
    )
    params_buf = device.newBufferWithBytes_length_options_(
        params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
    )

    # Compute grid
    grid_n = (out_dim + MOE_TILE_N - 1) // MOE_TILE_N
    grid_m = (batch_size + MOE_TILE_M - 1) // MOE_TILE_M

    # Dispatch
    dispatch_kernel(
        lib,
        function_name="moe_dispatch_optimized",
        grid=(grid_n, grid_m, 1),
        threadgroup=(MOE_THREADS_PER_TG, 1, 1),
        buffers=[
            act_buf,
            router_buf,
            expert_w_buf,
            expert_s_buf,
            shared_w_buf,
            shared_s_buf,
            out_buf,
            params_buf,
        ],
        wait=True,
    )

    return output


def dispatch_moe_prerouted(
    lib: MetalKernelLibrary,
    activations: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_scales: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_probs: torch.Tensor,
    shared_weights: torch.Tensor | None,
    shared_scales: torch.Tensor | None,
    batch_size: int,
    hidden_dim: int,
    out_dim: int,
    num_experts: int = 64,
    top_k: int = 4,
    group_size: int = 128,
) -> torch.Tensor:
    """Dispatch MoE with pre-computed routing decisions.

    Use when routing is computed separately (e.g., for profiling or debugging).

    Args:
        lib: MetalKernelLibrary with moe_dispatch_optimized compiled
        activations: Input [batch, hidden_dim], fp16, MPS tensor
        expert_weights: Packed FP4 expert weights [num_experts, K/8, N], uint32, MPS
        expert_scales: Expert scales [num_experts, K/group_size, N], fp16, MPS
        expert_ids: Pre-computed expert assignments [batch, top_k], uint32, MPS
        expert_probs: Pre-computed expert probabilities [batch, top_k], fp16, MPS
        shared_weights: Optional shared expert weights [K/8, N], uint32, MPS
        shared_scales: Optional shared expert scales [K/group_size, N], fp16, MPS
        batch_size: Number of tokens
        hidden_dim: Input dimension
        out_dim: Output dimension
        num_experts: Total number of experts (default 64)
        top_k: Experts per token (default 4)
        group_size: Quantization group size (default 128)

    Returns:
        Output tensor [batch, out_dim], fp16, MPS
    """
    require_mps()

    device = lib.device

    output = torch.empty((batch_size, out_dim),
                         dtype=torch.float16, device="mps")

    act_buf = mps_tensor_to_metal_buffer(
        activations.half().contiguous(), device)
    expert_w_buf = mps_tensor_to_metal_buffer(
        expert_weights.contiguous(), device)
    expert_s_buf = mps_tensor_to_metal_buffer(
        expert_scales.half().contiguous(), device)
    ids_buf = mps_tensor_to_metal_buffer(expert_ids.int().contiguous(), device)
    probs_buf = mps_tensor_to_metal_buffer(
        expert_probs.half().contiguous(), device)

    has_shared = 1 if shared_weights is not None else 0
    if shared_weights is not None:
        shared_w_buf = mps_tensor_to_metal_buffer(
            shared_weights.contiguous(), device)
        shared_s_buf = mps_tensor_to_metal_buffer(
            shared_scales.half().contiguous(), device)
    else:
        shared_w_buf = device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared)
        shared_s_buf = device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared)

    out_buf = mps_tensor_to_metal_buffer(output, device, copy_back=True)

    params = np.array(
        [batch_size, hidden_dim, out_dim, num_experts,
            top_k, group_size, has_shared],
        dtype=np.uint32,
    )
    params_buf = device.newBufferWithBytes_length_options_(
        params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
    )

    grid_n = (out_dim + MOE_TILE_N - 1) // MOE_TILE_N
    grid_m = (batch_size + MOE_TILE_M - 1) // MOE_TILE_M

    dispatch_kernel(
        lib,
        function_name="moe_dispatch_optimized_prerouted",
        grid=(grid_n, grid_m, 1),
        threadgroup=(MOE_THREADS_PER_TG, 1, 1),
        buffers=[
            act_buf,
            expert_w_buf,
            expert_s_buf,
            ids_buf,
            probs_buf,
            shared_w_buf,
            shared_s_buf,
            out_buf,
            params_buf,
        ],
        wait=True,
    )

    return output


def dispatch_moe_decode(
    lib: MetalKernelLibrary,
    activations: torch.Tensor,
    router_weights: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_scales: torch.Tensor,
    shared_weights: torch.Tensor | None,
    shared_scales: torch.Tensor | None,
    hidden_dim: int,
    out_dim: int,
    num_experts: int = 64,
    top_k: int = 4,
    group_size: int = 128,
) -> torch.Tensor:
    """Dispatch single-token MoE decode (batch_size=1).

    Optimized for minimal latency in autoregressive generation.

    Args:
        lib: MetalKernelLibrary with moe_dispatch_optimized compiled
        activations: Input [hidden_dim], fp16, MPS tensor (single token)
        router_weights: Router weights [hidden_dim, num_experts], fp16, MPS
        expert_weights: Packed FP4 expert weights [num_experts, K/8, N], uint32, MPS
        expert_scales: Expert scales [num_experts, K/group_size, N], fp16, MPS
        shared_weights: Optional shared expert weights [K/8, N], uint32, MPS
        shared_scales: Optional shared expert scales [K/group_size, N], fp16, MPS
        hidden_dim: Input dimension
        out_dim: Output dimension
        num_experts: Total number of experts (default 64)
        top_k: Experts per token (default 4)
        group_size: Quantization group size (default 128)

    Returns:
        Output tensor [out_dim], fp16, MPS
    """
    require_mps()

    device = lib.device

    output = torch.empty(out_dim, dtype=torch.float16, device="mps")

    act_buf = mps_tensor_to_metal_buffer(
        activations.half().contiguous(), device)
    router_buf = mps_tensor_to_metal_buffer(
        router_weights.half().contiguous(), device)
    expert_w_buf = mps_tensor_to_metal_buffer(
        expert_weights.contiguous(), device)
    expert_s_buf = mps_tensor_to_metal_buffer(
        expert_scales.half().contiguous(), device)

    has_shared = 1 if shared_weights is not None else 0
    if shared_weights is not None:
        shared_w_buf = mps_tensor_to_metal_buffer(
            shared_weights.contiguous(), device)
        shared_s_buf = mps_tensor_to_metal_buffer(
            shared_scales.half().contiguous(), device)
    else:
        shared_w_buf = device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared)
        shared_s_buf = device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared)

    out_buf = mps_tensor_to_metal_buffer(output, device, copy_back=True)

    params = np.array(
        [1, hidden_dim, out_dim, num_experts, top_k, group_size, has_shared], dtype=np.uint32
    )
    params_buf = device.newBufferWithBytes_length_options_(
        params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
    )

    # Decode kernel: 1D grid over output dimension
    dispatch_kernel(
        lib,
        function_name="moe_dispatch_decode",
        grid=(out_dim, 1, 1),
        threadgroup=(128, 1, 1),
        buffers=[
            act_buf,
            router_buf,
            expert_w_buf,
            expert_s_buf,
            shared_w_buf,
            shared_s_buf,
            out_buf,
            params_buf,
        ],
        wait=True,
    )

    return output


def benchmark_moe_dispatch(
    lib: MetalKernelLibrary,
    batch_size: int = 32,
    hidden_dim: int = 4096,
    out_dim: int = 14336,
    num_experts: int = 64,
    top_k: int = 4,
    group_size: int = 128,
    has_shared: bool = True,
    warmup_iters: int = 10,
    benchmark_iters: int = 50,
) -> dict[str, float]:
    """Benchmark MoE dispatch kernel performance.

    Measures time for routing + GEMM + combination.

    Args:
        lib: MetalKernelLibrary with moe_dispatch_optimized compiled
        batch_size: Number of tokens
        hidden_dim: Input dimension
        out_dim: Output dimension
        num_experts: Total number of experts
        top_k: Experts per token
        group_size: Quantization group size
        has_shared: Whether to include shared expert
        warmup_iters: Number of warmup iterations
        benchmark_iters: Number of timed iterations

    Returns:
        Dictionary with performance metrics.
    """
    require_mps()
    import time

    k_packed = hidden_dim // 8
    num_groups = (hidden_dim + group_size - 1) // group_size

    # Create test data
    activations = torch.randn(batch_size, hidden_dim,
                              dtype=torch.float16, device="mps")
    router_weights = torch.randn(
        hidden_dim, num_experts, dtype=torch.float16, device="mps") * 0.01
    expert_weights = torch.randint(
        0, 2**32, (num_experts, k_packed, out_dim), dtype=torch.int32, device="mps"
    ).view(torch.uint32)
    expert_scales = (
        torch.randn(num_experts, num_groups, out_dim,
                    dtype=torch.float16, device="mps") * 0.1 + 0.5
    ).abs()

    if has_shared:
        shared_weights = torch.randint(
            0, 2**32, (k_packed, out_dim), dtype=torch.int32, device="mps"
        ).view(torch.uint32)
        shared_scales = (
            torch.randn(num_groups, out_dim, dtype=torch.float16,
                        device="mps") * 0.1 + 0.5
        ).abs()
    else:
        shared_weights = None
        shared_scales = None

    # Warmup
    for _ in range(warmup_iters):
        _ = dispatch_moe_optimized(
            lib,
            activations,
            router_weights,
            expert_weights,
            expert_scales,
            shared_weights,
            shared_scales,
            batch_size,
            hidden_dim,
            out_dim,
            num_experts,
            top_k,
            group_size,
        )
    torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(benchmark_iters):
        _ = dispatch_moe_optimized(
            lib,
            activations,
            router_weights,
            expert_weights,
            expert_scales,
            shared_weights,
            shared_scales,
            batch_size,
            hidden_dim,
            out_dim,
            num_experts,
            top_k,
            group_size,
        )
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / benchmark_iters) * 1000

    # Compute FLOPs
    # Active params per token: (top_k + has_shared) * (hidden_dim * out_dim)
    active_experts = top_k + (1 if has_shared else 0)
    flops_per_token = 2 * active_experts * \
        hidden_dim * out_dim  # 2 for multiply-add
    total_flops = flops_per_token * batch_size

    # Add routing cost: batch_size * hidden_dim * num_experts
    routing_flops = 2 * batch_size * hidden_dim * num_experts
    total_flops += routing_flops

    tflops = total_flops / 1e12
    tflops_per_sec = tflops / (avg_time_ms / 1000)

    # Memory bandwidth estimate
    # Read: activations + router weights + (top_k + shared) expert weights
    read_bytes = (
        batch_size * hidden_dim * 2  # activations
        + hidden_dim * num_experts * 2  # router weights
        + active_experts
        # expert weights + scales
        * (k_packed * out_dim * 4 + num_groups * out_dim * 2)
    )
    write_bytes = batch_size * out_dim * 2  # output
    total_bytes = read_bytes + write_bytes
    bandwidth_gb_s = (total_bytes / (avg_time_ms / 1000)) / 1e9

    # Equivalent dense model time estimate
    # Dense model with same active params: batch_size * hidden_dim * out_dim * active_experts
    dense_flops = 2 * batch_size * hidden_dim * out_dim * active_experts
    # Theoretical peak (M4 Max ~200 TFLOPs FP16)
    theoretical_peak_tflops = 200
    dense_time_ms = (dense_flops / 1e12) / theoretical_peak_tflops * 1000

    overhead_percent = ((avg_time_ms / dense_time_ms) - 1) * 100

    return {
        "batch_size": batch_size,
        "time_ms": avg_time_ms,
        "tflops_per_sec": tflops_per_sec,
        "bandwidth_gb_s": bandwidth_gb_s,
        "active_experts": active_experts,
        "flops_billion": total_flops / 1e9,
        "dense_time_ms_estimate": dense_time_ms,
        "moe_overhead_percent": overhead_percent,
    }


# ---------------------------------------------------------------------------
# Hessian Computation Dispatch (Metal-accelerated H = X^T @ X)
# ---------------------------------------------------------------------------


def dispatch_hessian_compute(
    lib: MetalKernelLibrary,
    X: torch.Tensor,
    sigma_reg: float = 0.01,
) -> torch.Tensor:
    """Dispatch Metal Hessian computation kernel.

    Computes H = 2 * X^T @ X using optimized simdgroup matrix operations.
    Much faster than PyTorch MPS for large matrices (4K+ hidden dim).

    Args:
        lib: MetalKernelLibrary with hessian.metal compiled
        X: Activation matrix [n_samples, hidden_dim], float16/bf16/float32, MPS tensor
        sigma_reg: Regularization as fraction of diagonal mean (default 0.01)

    Returns:
        H: Hessian matrix [hidden_dim, hidden_dim], float32, MPS tensor
    """
    require_mps()

    device = lib.device
    n_samples, hidden_dim = X.shape

    # Convert to float16 for Metal kernel (uses FP32 accumulation internally)
    X_fp16 = X.half().contiguous()

    # Allocate output Hessian
    H = torch.zeros(hidden_dim, hidden_dim, dtype=torch.float32, device="mps")

    # Create Metal buffers
    X_buf = mps_tensor_to_metal_buffer(X_fp16, device)
    H_buf = mps_tensor_to_metal_buffer(H, device, copy_back=True)

    # Create parameter buffers
    n_samples_buf = device.newBufferWithBytes_length_options_(
        np.array([n_samples], dtype=np.uint32).tobytes(
        ), 4, Metal.MTLResourceStorageModeShared
    )
    hidden_dim_buf = device.newBufferWithBytes_length_options_(
        np.array([hidden_dim], dtype=np.uint32).tobytes(
        ), 4, Metal.MTLResourceStorageModeShared
    )

    # Tile dimensions from hessian.metal (HESSIAN_TILE_DIM = 64)
    TILE_DIM = 64
    THREADS_PER_TG = 128  # 4 simdgroups * 32 threads

    grid_x = (hidden_dim + TILE_DIM - 1) // TILE_DIM
    grid_y = (hidden_dim + TILE_DIM - 1) // TILE_DIM

    # Dispatch hessian_compute_fp16 kernel
    dispatch_kernel(
        lib,
        function_name="hessian_compute_fp16",
        grid=(grid_x, grid_y, 1),
        threadgroup=(THREADS_PER_TG, 1, 1),
        buffers=[X_buf, H_buf, n_samples_buf, hidden_dim_buf],
        wait=True,
    )

    # Normalize first: kernel computes 2 * X^T @ X, we want X^T @ X / n_samples
    H /= 2 * n_samples

    # Then apply regularization: H += sigma_reg * mean(diag(H)) * I
    diag_mean = H.diagonal().mean()
    H += sigma_reg * diag_mean * \
        torch.eye(hidden_dim, device="mps", dtype=torch.float32)

    return H


def dispatch_hessian_accumulate(
    lib: MetalKernelLibrary,
    X: torch.Tensor,
    H: torch.Tensor,
) -> torch.Tensor:
    """Accumulate into existing Hessian: H += 2 * X^T @ X.

    For streaming/batched Hessian collection.

    Args:
        lib: MetalKernelLibrary with hessian.metal compiled
        X: Activation matrix [n_samples, hidden_dim], MPS tensor
        H: Existing Hessian [hidden_dim, hidden_dim], float32, MPS tensor (modified in-place)

    Returns:
        H: Updated Hessian (same tensor, modified in-place)
    """
    require_mps()

    device = lib.device
    n_samples, hidden_dim = X.shape

    X_fp16 = X.half().contiguous()
    H = H.contiguous()

    X_buf = mps_tensor_to_metal_buffer(X_fp16, device)
    H_buf = mps_tensor_to_metal_buffer(H, device, copy_back=True)

    n_samples_buf = device.newBufferWithBytes_length_options_(
        np.array([n_samples], dtype=np.uint32).tobytes(
        ), 4, Metal.MTLResourceStorageModeShared
    )
    hidden_dim_buf = device.newBufferWithBytes_length_options_(
        np.array([hidden_dim], dtype=np.uint32).tobytes(
        ), 4, Metal.MTLResourceStorageModeShared
    )

    TILE_DIM = 64
    THREADS_PER_TG = 128

    grid_x = (hidden_dim + TILE_DIM - 1) // TILE_DIM
    grid_y = (hidden_dim + TILE_DIM - 1) // TILE_DIM

    dispatch_kernel(
        lib,
        function_name="hessian_accumulate",
        grid=(grid_x, grid_y, 1),
        threadgroup=(THREADS_PER_TG, 1, 1),
        buffers=[X_buf, H_buf, n_samples_buf, hidden_dim_buf],
        wait=True,
    )

    return H


# ---------------------------------------------------------------------------
# Viterbi Quantization Dispatch
# ---------------------------------------------------------------------------


def dispatch_viterbi_quantize(
    lib: MetalKernelLibrary,
    tiles: torch.Tensor,
    scales: torch.Tensor,
    grid: torch.Tensor,
    use_u4_kernel: bool = True,
    return_dequant: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Dispatch Viterbi trellis quantization on Metal GPU.

    Uses the viterbi_quant.metal shader for parallel tile quantization.
    Each tile is 16x16 (256 elements) processed by one threadgroup.

    Args:
        lib: MetalKernelLibrary with viterbi_quant compiled
        tiles: Input tiles [n_tiles, 256], float32, MPS tensor
        scales: Per-tile scale factors [n_tiles], float32, MPS tensor
        grid: Quantization grid values [n_states], float32, MPS tensor
        use_u4_kernel: Use optimized 4-bit kernel (16 states) if True
        return_dequant: If False, skip dequant copy-back (faster)

    Returns:
        indices: Quantized state indices [n_tiles, 256], uint8, MPS tensor
        dequantized: Reconstructed values [n_tiles, 256], float32, or None
    """
    require_mps()

    device = lib.device
    n_tiles = tiles.shape[0]
    n_states = grid.shape[0]

    # Ensure contiguous float32 tensors
    tiles = tiles.float().contiguous()
    scales = scales.float().contiguous()
    grid = grid.float().contiguous()

    # Allocate output buffers
    indices = torch.zeros(n_tiles, 256, dtype=torch.uint8, device="mps")
    dequantized = torch.zeros(n_tiles, 256, dtype=torch.float32, device="mps")

    # Create Metal buffers
    tiles_buf = mps_tensor_to_metal_buffer(tiles, device)
    scales_buf = mps_tensor_to_metal_buffer(scales, device)
    grid_buf = mps_tensor_to_metal_buffer(grid, device)
    indices_buf = mps_tensor_to_metal_buffer(indices, device, copy_back=True)
    # Only copy-back dequant if needed (saves 84MB GPU->CPU transfer per dispatch)
    dequant_buf = mps_tensor_to_metal_buffer(
        dequantized, device, copy_back=return_dequant)

    # Parameters buffer
    params = np.array([n_tiles, n_states], dtype=np.uint32)
    params_buf = device.newBufferWithBytes_length_options_(
        params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
    )

    # Select kernel
    if use_u4_kernel and n_states == 16:
        kernel_name = "quantize_tiles_viterbi_u4"
        # U4 kernel only needs n_tiles param (n_states fixed at 16)
        params = np.array([n_tiles], dtype=np.uint32)
        params_buf = device.newBufferWithBytes_length_options_(
            params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
        )
        buffers = [tiles_buf, scales_buf, grid_buf,
                   indices_buf, dequant_buf, params_buf]
    else:
        kernel_name = "quantize_tiles_viterbi"
        buffers = [
            tiles_buf,
            scales_buf,
            grid_buf,
            indices_buf,
            dequant_buf,
            params_buf,
            params_buf,
        ]
        # Note: second params_buf is for n_states

    # Dispatch: one threadgroup per tile, 256 threads per group
    dispatch_kernel(
        lib,
        function_name=kernel_name,
        grid=(n_tiles, 1, 1),
        threadgroup=(256, 1, 1),
        buffers=buffers,
        wait=True,
    )

    return indices, dequantized


def dispatch_viterbi_quantize_naive(
    lib: MetalKernelLibrary,
    tiles: torch.Tensor,
    scales: torch.Tensor,
    grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch naive (greedy) quantization on Metal GPU.

    Faster but lower quality than Viterbi - useful for comparison.

    Args:
        lib: MetalKernelLibrary with viterbi_quant compiled
        tiles: Input tiles [n_tiles, 256], float32, MPS tensor
        scales: Per-tile scale factors [n_tiles], float32, MPS tensor
        grid: Quantization grid values [n_states], float32, MPS tensor

    Returns:
        indices: Quantized state indices [n_tiles, 256], uint8, MPS tensor
        dequantized: Reconstructed values [n_tiles, 256], float32, MPS tensor
    """
    require_mps()

    device = lib.device
    n_tiles = tiles.shape[0]
    n_states = grid.shape[0]

    tiles = tiles.float().contiguous()
    scales = scales.float().contiguous()
    grid = grid.float().contiguous()

    indices = torch.zeros(n_tiles, 256, dtype=torch.uint8, device="mps")
    dequantized = torch.zeros(n_tiles, 256, dtype=torch.float32, device="mps")

    tiles_buf = mps_tensor_to_metal_buffer(tiles, device)
    scales_buf = mps_tensor_to_metal_buffer(scales, device)
    grid_buf = mps_tensor_to_metal_buffer(grid, device)
    indices_buf = mps_tensor_to_metal_buffer(indices, device, copy_back=True)
    dequant_buf = mps_tensor_to_metal_buffer(
        dequantized, device, copy_back=True)

    params = np.array([n_tiles, n_states], dtype=np.uint32)
    params_buf = device.newBufferWithBytes_length_options_(
        params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
    )

    dispatch_kernel(
        lib,
        function_name="quantize_tiles_naive",
        grid=(n_tiles, 1, 1),
        threadgroup=(256, 1, 1),
        buffers=[tiles_buf, scales_buf, grid_buf,
                 indices_buf, dequant_buf, params_buf, params_buf],
        wait=True,
    )

    return indices, dequantized


# ---------------------------------------------------------------------------
# Module-level singleton for convenience
# ---------------------------------------------------------------------------

_default_library: MetalKernelLibrary | None = None


def get_default_library() -> MetalKernelLibrary:
    """Get or create the default kernel library."""
    global _default_library
    if _default_library is None:
        _default_library = MetalKernelLibrary.from_source_dir()
    return _default_library


def get_kernel(kernel_name: str) -> Any | None:
    """Get kernel function from metallib (module-level convenience function).

    This is a thin wrapper around get_kernel_from_metallib() for convenience.
    Returns None if the kernel is not found in the metallib.

    Args:
        kernel_name: Name of kernel function.

    Returns:
        MTLFunction or None if not found.
    """
    return get_kernel_from_metallib(kernel_name)


# ---------------------------------------------------------------------------
# Test / verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick test of Metal availability
    print(f"PyObjC Metal: {HAS_METAL}")
    print(f"PyTorch MPS: {HAS_MPS}")

    if HAS_METAL:
        lib = MetalKernelLibrary.from_source_dir()
        print(f"\nCompiled libraries: {list(lib._libraries.keys())}")

        # List functions from a library
        for name in lib._libraries:
            funcs = lib.list_functions(name)
            print(f"  {name}: {len(funcs)} functions")

        # Run dequant bandwidth benchmark if MPS available
        if HAS_MPS:
            print("\n--- FP4 Dequant Bandwidth Benchmark ---")
            results = benchmark_dequant_fp4(lib)
            print(f"  Time per iteration: {results['time_ms']:.3f} ms")
            print(
                f"  Effective bandwidth: {results['bandwidth_gb_s']:.1f} GB/s")
            print(
                f"  Dequant throughput: {results['throughput_gop_s']:.1f} GOP/s")
            print(f"  Output size: {results['output_mb']:.1f} MB")
