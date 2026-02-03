"""Fast inference path using C++ extension for 5-10x lower dispatch overhead.

This module provides:
    - FastInferenceContext: Manages C++ extension context and buffer pools
    - fast_dispatch_gemm_trellis: Direct GEMM dispatch without PyObjC overhead
    
Usage:
    from metal_marlin.fast_inference import FastInferenceContext
    
    ctx = FastInferenceContext()
    if ctx.available:
        output = ctx.dispatch_gemm_trellis(weights, input_tensor, scales)
    else:
        # Fall back to regular dispatch
        output = dispatch_gemm_trellis_packed(lib, weights, input_tensor, ...)

Performance:
    - PyObjC dispatch: ~80-150μs per kernel call
    - C++ extension: ~5-15μs per kernel call
    - For MoE with 2-4 experts, saves ~200-600μs per layer
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._compat import HAS_CPP_EXT, HAS_TORCH, _metal_dispatch_ext, torch

if TYPE_CHECKING:
    import torch

_logger = logging.getLogger(__name__)

# Check if extension and torch are available
_FAST_PATH_AVAILABLE = HAS_CPP_EXT and HAS_TORCH and _metal_dispatch_ext is not None


class FastInferenceContext:
    """Context for fast Metal kernel dispatch using C++ extension.

    This class provides a high-level interface for using the C++ extension's
    low-overhead dispatch path. It manages:
    - MetalContext initialization
    - Metallib loading
    - Buffer pool for intermediate tensors
    - Kernel pipeline caching

    Example:
        >>> ctx = FastInferenceContext()
        >>> if ctx.available:
        ...     # Use C++ extension for fast dispatch
        ...     output = ctx.gemm_trellis_packed(A, packed_indices, scales, grid, su, sv, K, N, bits)
        ... else:
        ...     # Fall back to PyObjC
        ...     output = dispatch_gemm_trellis_packed(lib, A, packed_indices, scales, grid, su, sv, K, N, bits)
    """

    __slots__ = ("_ctx", "_available", "_pipelines", "_buffers")

    def __init__(self, metallib_paths: list[str] | None = None):
        """Initialize fast inference context.

        Args:
            metallib_paths: Optional list of paths to .metallib files to load.
                           If None, searches default locations.
        """
        self._ctx: Any = None
        self._available = False
        self._pipelines: dict[str, Any] = {}
        self._buffers: dict[int, Any] = {}  # size -> buffer cache

        if not _FAST_PATH_AVAILABLE:
            _logger.debug("C++ extension not available, fast path disabled")
            return

        try:
            self._ctx = _metal_dispatch_ext.MetalContext()

            # Load metallibs
            metallib_paths = metallib_paths or self._find_metallibs()
            for path in metallib_paths:
                try:
                    self._ctx.load_metallib(path)
                    _logger.debug(f"Loaded metallib: {path}")
                except Exception as e:
                    _logger.warning(f"Failed to load metallib {path}: {e}")

            self._available = True
            _logger.info("Fast inference context initialized")
        except Exception as e:
            _logger.warning(
                f"Failed to initialize fast inference context: {e}")
            self._ctx = None
            self._available = False

    def _find_metallibs(self) -> list[str]:
        """Find metallib files in standard locations."""
        paths = []

        # Check relative to this file
        module_dir = Path(__file__).parent
        candidates = [
            module_dir / "lib" / "metal_marlin.metallib",
            module_dir / "shaders" / "metal_marlin.metallib",
            module_dir.parent / "build" / "metal_marlin.metallib",
        ]

        for candidate in candidates:
            if candidate.exists():
                paths.append(str(candidate))

        return paths

    @property
    def available(self) -> bool:
        """Return True if the fast path is available."""
        return self._available

    def get_pipeline(self, kernel_name: str) -> Any:
        """Get a compute pipeline for a kernel.

        Args:
            kernel_name: Name of the kernel function.

        Returns:
            Pipeline capsule from C++ extension.
        """
        if not self._available:
            raise RuntimeError("Fast inference context not available")

        if kernel_name not in self._pipelines:
            self._pipelines[kernel_name] = self._ctx.get_pipeline(kernel_name)
        return self._pipelines[kernel_name]

    def create_buffer(self, size: int) -> Any:
        """Create or reuse a Metal buffer.

        Args:
            size: Buffer size in bytes.

        Returns:
            ManagedBuffer from C++ extension.
        """
        if not self._available:
            raise RuntimeError("Fast inference context not available")

        return _metal_dispatch_ext.create_buffer(self._ctx, size, True)

    def wrap_tensor(self, tensor: torch.Tensor) -> Any:
        """Wrap a PyTorch MPS tensor as a ManagedBuffer.

        Args:
            tensor: MPS tensor to wrap.

        Returns:
            ManagedBuffer pointing to the tensor's data.
        """
        if not self._available:
            raise RuntimeError("Fast inference context not available")

        if not tensor.is_mps:
            raise ValueError("Tensor must be on MPS device")

        # Get data pointer from MPS tensor
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()

        return _metal_dispatch_ext.create_buffer_from_ptr(self._ctx, ptr, size)

    def dispatch(
        self,
        kernel_name: str,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        buffers: list[Any],
        wait: bool = True,
    ) -> None:
        """Dispatch a kernel.

        Args:
            kernel_name: Name of the kernel function.
            grid: Grid dimensions (threadgroups).
            threadgroup: Threadgroup dimensions (threads).
            buffers: List of ManagedBuffer objects.
            wait: If True, wait for kernel completion.
        """
        if not self._available:
            raise RuntimeError("Fast inference context not available")

        pipeline = self.get_pipeline(kernel_name)
        _metal_dispatch_ext.dispatch_kernel(
            self._ctx,
            pipeline,
            grid,
            threadgroup,
            buffers,
            wait,
        )

    def gemm_trellis_packed(
        self,
        A: torch.Tensor,
        packed_indices: torch.Tensor,
        scales: torch.Tensor,
        grid: torch.Tensor,
        su: torch.Tensor,
        sv: torch.Tensor,
        K: int,
        N: int,
        bits: int,
        group_size: int = 32,
    ) -> torch.Tensor:
        """Dispatch fused trellis GEMM kernel.

        Computes C[M,N] = A[M,K] @ dequant(W[K,N]) where W is trellis-quantized.
        Weights are dequantized on-the-fly during the GEMM computation without
        materializing the full FP16 weight matrix.

        Args:
            A: Input activations [M, K] float16, MPS tensor
            packed_indices: Packed trellis indices [tiles_k, tiles_n, packed_bytes] uint8
            scales: Per-group scales [n_groups, N] float32, MPS tensor
            grid: Codebook grid [n_levels] float32, MPS tensor
            su: Row signs [K] float32, MPS tensor
            sv: Column signs [N] float32, MPS tensor
            K: Number of columns in A / rows in W
            N: Number of columns in W and C
            bits: Quantization bit width (2, 3, or 4)
            group_size: Quantization group size (default 32)

        Returns:
            Output matrix [M, N] float16, MPS tensor
        """
        if not self._available:
            raise RuntimeError("Fast inference context not available")

        M = A.shape[0]
        n_levels = grid.shape[0]

        # Ensure proper types and contiguity
        A = A.contiguous()
        packed_indices = packed_indices.contiguous()
        scales = scales.float().contiguous()
        grid = grid.float().contiguous()
        su = su.float().contiguous()
        sv = sv.float().contiguous()

        # Allocate output
        output = torch.zeros(M, N, dtype=torch.float16, device="mps")

        # Wrap tensors as ManagedBuffers
        A_buf = self.wrap_tensor(A)
        packed_indices_buf = self.wrap_tensor(packed_indices)
        scales_buf = self.wrap_tensor(scales)
        grid_buf = self.wrap_tensor(grid)
        su_buf = self.wrap_tensor(su)
        sv_buf = self.wrap_tensor(sv)
        output_buf = self.wrap_tensor(output)

        # Create separate buffers for each constant parameter
        def make_uint_buffer(val: int) -> Any:
            data = np.array([val], dtype=np.uint32)
            return _metal_dispatch_ext.create_buffer_from_bytes(
                self._ctx, data.tobytes(), False
            )

        M_buf = make_uint_buffer(M)
        K_buf = make_uint_buffer(K)
        N_buf = make_uint_buffer(N)
        bits_buf = make_uint_buffer(bits)
        n_levels_buf = make_uint_buffer(n_levels)
        group_size_buf = make_uint_buffer(group_size)

        # Select kernel based on shape (matching dispatch.py behavior)
        if M <= 16:
            kernel_name = "gemm_trellis_packed_decode"
            TILE_M = 32
            TILE_N = 128
        else:
            kernel_name = "gemm_trellis_packed"
            TILE_M = 64
            TILE_N = 64

        # Compute grid
        grid_x = (N + TILE_N - 1) // TILE_N
        grid_y = (M + TILE_M - 1) // TILE_M
        threads_per_tg = 128  # 4 simdgroups * 32 threads

        # Create buffer list matching exact order from dispatch.py
        bufs = [
            A_buf,
            packed_indices_buf,
            scales_buf,
            grid_buf,
            su_buf,
            sv_buf,
            output_buf,
            M_buf,
            K_buf,
            N_buf,
            bits_buf,
            n_levels_buf,
            group_size_buf,
        ]

        # Dispatch
        self.dispatch(
            kernel_name,
            grid=(grid_x, grid_y, 1),
            threadgroup=(threads_per_tg, 1, 1),
            buffers=bufs,
            wait=True,
        )

        return output

    def gemm_trellis_decode(
        self,
        A: torch.Tensor,
        packed_indices: torch.Tensor,
        scales: torch.Tensor,
        grid: torch.Tensor,
        su: torch.Tensor,
        sv: torch.Tensor,
        K: int,
        N: int,
        bits: int,
        group_size: int = 32,
    ) -> torch.Tensor:
        """Dispatch decode-optimized trellis GEMM kernel.

        Optimized for small batch sizes (M <= 16).
        Uses TILE_M=32, TILE_N=128 for better decode throughput.
        """
        if not self._available:
            raise RuntimeError("Fast inference context not available")

        M = A.shape[0]
        n_levels = grid.shape[0]

        # Allocate output
        output = torch.empty(M, N, dtype=torch.float16, device="mps")

        # Wrap tensors as ManagedBuffers
        bufs = [
            self.wrap_tensor(A.contiguous()),
            self.wrap_tensor(packed_indices.contiguous()),
            self.wrap_tensor(scales.contiguous()),
            self.wrap_tensor(grid.contiguous()),
            self.wrap_tensor(su.contiguous()),
            self.wrap_tensor(sv.contiguous()),
            self.wrap_tensor(output),
        ]

        # Decode kernel uses different tile sizes
        TILE_M, TILE_N = 32, 128
        threads_per_tg = 128

        grid_x = (N + TILE_N - 1) // TILE_N
        grid_y = (M + TILE_M - 1) // TILE_M

        self.dispatch(
            "gemm_trellis_packed_decode",
            grid=(grid_x, grid_y, 1),
            threadgroup=(threads_per_tg, 1, 1),
            buffers=bufs,
            wait=True,
        )

        return output


# Module-level singleton for convenience
_global_ctx: FastInferenceContext | None = None


def get_fast_context() -> FastInferenceContext:
    """Get the global fast inference context.

    Creates the context lazily on first call.

    Returns:
        FastInferenceContext singleton.
    """
    global _global_ctx
    if _global_ctx is None:
        _global_ctx = FastInferenceContext()
    return _global_ctx


def fast_dispatch_available() -> bool:
    """Check if fast dispatch is available.

    Returns:
        True if C++ extension is available and initialized.
    """
    return get_fast_context().available
