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

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

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
try:
    import torch

    HAS_TORCH = True
    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False
    torch = None


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
        self._pipelines: dict[str, Any] = {}  # function_name -> MTLComputePipelineState
        self._command_queue = device.newCommandQueue()

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
        """The MTLCommandQueue for kernel dispatch."""
        return self._command_queue

    def compile_source(self, name: str, source: str) -> Any:
        """Compile Metal source code into a library.

        Args:
            name: Identifier for this source (e.g., 'marlin_gemm')
            source: Metal shader source code

        Returns:
            MTLLibrary instance.
        """
        options = Metal.MTLCompileOptions.new()
        # Enable fast math for performance
        options.setFastMathEnabled_(True)
        # Metal 3.0 for simdgroup_matrix
        options.setLanguageVersion_(Metal.MTLLanguageVersion3_0)

        library, error = self._device.newLibraryWithSource_options_error_(source, options, None)

        if library is None:
            # Try to get error message
            error_msg = str(error) if error else "Unknown error"
            raise RuntimeError(f"Failed to compile Metal source '{name}': {error_msg}")

        self._libraries[name] = library
        return library

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
            raise KeyError(f"Function '{function_name}' not found in any library")

        # Create pipeline state
        pipeline, error = self._device.newComputePipelineStateWithFunction_error_(function, None)

        if pipeline is None:
            raise RuntimeError(f"Failed to create pipeline for '{function_name}'")

        self._pipelines[cache_key] = pipeline
        return pipeline

    def list_functions(self, library_name: str) -> list[str]:
        """List all function names in a compiled library."""
        lib = self._libraries.get(library_name)
        if lib is None:
            return []
        return list(lib.functionNames())

    def get_kernel(self, library_name: str, function_name: str) -> Any:
        """Get a compute pipeline from a specific library."""
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

        buffers: list[Any] = []
        for arg in args:
            if isinstance(arg, (int, np.integer)):
                const = np.array([int(arg)], dtype=np.uint32)
                buf = self._device.newBufferWithBytes_length_options_(
                    const.tobytes(), const.nbytes, Metal.MTLResourceStorageModeShared
                )
                buffers.append(buf)
            else:
                buffers.append(arg)

        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)

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
            (grid_m, grid_n, 1),
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

        return output

    def int4_gemm(
        self,
        input: torch.Tensor,  # [M, K] input activations
        weight: torch.Tensor,  # Packed INT4 weights
        scales: torch.Tensor,  # Per-group scales
        zeros: torch.Tensor,  # Per-group zero points
        N: int,  # Output features
        K: int,  # Input features
        group_size: int = 128,
    ) -> torch.Tensor:
        """Fused INT4 dequantize + GEMM using marlin_gemm kernel."""
        M = input.shape[0]

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
            (grid_m, grid_n, 1),
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

        return output


# ---------------------------------------------------------------------------
# PyTorch MPS <-> Metal buffer interop
# ---------------------------------------------------------------------------


def mps_tensor_to_metal_buffer(tensor: torch.Tensor, device: Any) -> Any:
    """Get Metal buffer from PyTorch MPS tensor.

    This creates a shared buffer - no copy is made.

    Args:
        tensor: PyTorch tensor on MPS device
        device: MTLDevice (must match MPS device)

    Returns:
        MTLBuffer sharing the tensor's memory.
    """
    require_mps()

    if not tensor.is_mps:
        raise ValueError("Tensor must be on MPS device")

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # Get the raw data pointer via storage
    # PyTorch MPS tensors use Metal buffers internally
    storage = tensor.untyped_storage()

    # Create a Metal buffer from the same memory
    # This uses Metal's buffer creation from existing memory
    ptr = storage.data_ptr()
    size = storage.nbytes()

    buffer = device.newBufferWithBytesNoCopy_length_options_deallocator_(
        ptr, size, Metal.MTLResourceStorageModeShared, None
    )

    if buffer is None:
        raise RuntimeError("Failed to create Metal buffer from tensor")

    return buffer


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
    wait: bool = True,
) -> None:
    """Dispatch a Metal compute kernel.

    Args:
        lib: MetalKernelLibrary with compiled shaders
        function_name: Kernel function to dispatch
        grid: Grid dimensions (threadgroups in X, Y, Z)
        threadgroup: Threadgroup dimensions (threads in X, Y, Z)
        buffers: Sequence of MTLBuffer arguments (in order)
        wait: If True, wait for kernel completion
    """
    pipeline = lib.get_pipeline(function_name)

    command_buffer = lib.command_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()

    encoder.setComputePipelineState_(pipeline)

    # Bind buffers
    for i, buf in enumerate(buffers):
        encoder.setBuffer_offset_atIndex_(buf, 0, i)

    # Dispatch
    grid_size = Metal.MTLSizeMake(*grid)
    tg_size = Metal.MTLSizeMake(*threadgroup)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)

    encoder.endEncoding()
    command_buffer.commit()

    if wait:
        command_buffer.waitUntilCompleted()


# ---------------------------------------------------------------------------
# High-level GEMM dispatch
# ---------------------------------------------------------------------------

# Tile dimensions (must match marlin_gemm.metal)
TILE_M = 64
TILE_N = 64
TILE_K = 32
THREADS_PER_TG = 128


def dispatch_gemm_fp4(
    lib: MetalKernelLibrary,
    A: torch.Tensor,
    B_packed: torch.Tensor,
    scales: torch.Tensor,
    M: int,
    N: int,
    K: int,
    group_size: int = 32,
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

    Returns:
        Output tensor [M, N], fp16, MPS
    """
    require_mps()

    device = lib.device

    # Allocate output
    C = torch.empty((M, N), dtype=torch.float16, device="mps")

    # Convert tensors to Metal buffers
    A_buf = mps_tensor_to_metal_buffer(A.half().contiguous(), device)
    B_buf = mps_tensor_to_metal_buffer(B_packed.contiguous(), device)
    S_buf = mps_tensor_to_metal_buffer(scales.half().contiguous(), device)
    C_buf = mps_tensor_to_metal_buffer(C, device)

    # Create params buffer (struct matching kernel expectations)
    # struct GemmParams { uint M, N, K, group_size; }
    params = np.array([M, N, K, group_size], dtype=np.uint32)
    params_buf = device.newBufferWithBytes_length_options_(
        params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
    )

    # Compute grid
    grid_m = (M + TILE_M - 1) // TILE_M
    grid_n = (N + TILE_N - 1) // TILE_N

    # Dispatch
    dispatch_kernel(
        lib,
        function_name="marlin_gemm_fp4",
        grid=(grid_m, grid_n, 1),
        threadgroup=(THREADS_PER_TG, 1, 1),
        buffers=[A_buf, B_buf, S_buf, C_buf, params_buf],
        wait=True,
    )

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
) -> torch.Tensor:
    """Dispatch FP8 E4M3 quantized GEMM: C = A @ dequant(B).

    Similar to dispatch_gemm_fp4 but for FP8 weights.
    """
    require_mps()

    device = lib.device

    C = torch.empty((M, N), dtype=torch.float16, device="mps")

    A_buf = mps_tensor_to_metal_buffer(A.half().contiguous(), device)
    B_buf = mps_tensor_to_metal_buffer(B_packed.contiguous(), device)
    S_buf = mps_tensor_to_metal_buffer(scales.half().contiguous(), device)
    C_buf = mps_tensor_to_metal_buffer(C, device)

    params = np.array([M, N, K, group_size], dtype=np.uint32)
    params_buf = device.newBufferWithBytes_length_options_(
        params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
    )

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
) -> torch.Tensor:
    """Dispatch INT2 quantized GEMM: C = A @ dequant(B).

    For cold MoE experts with extreme compression.
    """
    require_mps()

    device = lib.device

    C = torch.empty((M, N), dtype=torch.float16, device="mps")

    A_buf = mps_tensor_to_metal_buffer(A.half().contiguous(), device)
    B_buf = mps_tensor_to_metal_buffer(B_packed.contiguous(), device)
    S_buf = mps_tensor_to_metal_buffer(scales.half().contiguous(), device)
    C_buf = mps_tensor_to_metal_buffer(C, device)

    params = np.array([M, N, K, group_size], dtype=np.uint32)
    params_buf = device.newBufferWithBytes_length_options_(
        params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
    )

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
    output_buf = mps_tensor_to_metal_buffer(output, device)

    # Create constant buffers
    K_buf = device.newBufferWithBytes_length_options_(
        np.array([K], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    N_buf = device.newBufferWithBytes_length_options_(
        np.array([N], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    gs_buf = device.newBufferWithBytes_length_options_(
        np.array([group_size], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
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
    output_buf = mps_tensor_to_metal_buffer(output, device)

    # Create constant buffers
    np_buf = device.newBufferWithBytes_length_options_(
        np.array([num_packed], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    gs_buf = device.newBufferWithBytes_length_options_(
        np.array([group_size], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )

    # Grid: 1D, each threadgroup processes DEQUANT_OPT_PACKS_PER_TG packed words
    num_threadgroups = (num_packed + DEQUANT_OPT_PACKS_PER_TG - 1) // DEQUANT_OPT_PACKS_PER_TG

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
        raise ValueError("num_packed must be divisible by 4 for bandwidth_max kernel")

    device = lib.device

    # Allocate output
    num_elements = num_packed * 8
    output = torch.empty(num_elements, dtype=torch.float16, device="mps")

    # Convert tensors to Metal buffers
    packed_buf = mps_tensor_to_metal_buffer(packed.contiguous(), device)
    scales_buf = mps_tensor_to_metal_buffer(scales.half().contiguous(), device)
    output_buf = mps_tensor_to_metal_buffer(output, device)

    # Kernel expects num_packed / 4
    num_packed_div4 = num_packed // 4
    np_buf = device.newBufferWithBytes_length_options_(
        np.array([num_packed_div4], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    gs_buf = device.newBufferWithBytes_length_options_(
        np.array([group_size], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )

    # Grid: 1D, each thread processes one uint4 (4 packed words)
    num_threadgroups = (num_packed_div4 + DEQUANT_OPT_THREADS - 1) // DEQUANT_OPT_THREADS

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
    num_packed: int = 1024 * 1024 * 16,  # 16M packed = 128M FP16 values = 256 MB output
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
    scales = torch.randn(num_groups, dtype=torch.float16, device="mps") * 0.1 + 0.5

    # Warmup
    for _ in range(warmup_iters):
        _ = dispatch_dequant_fp4_bandwidth_max(lib, packed, scales, num_packed, group_size)
    torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(benchmark_iters):
        _ = dispatch_dequant_fp4_bandwidth_max(lib, packed, scales, num_packed, group_size)
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
MOE_TILE_M = 32
MOE_TILE_N = 128
MOE_THREADS_PER_TG = 128


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
    output = torch.empty((batch_size, out_dim), dtype=torch.float16, device="mps")

    # Convert tensors to Metal buffers
    act_buf = mps_tensor_to_metal_buffer(activations.half().contiguous(), device)
    router_buf = mps_tensor_to_metal_buffer(router_weights.half().contiguous(), device)
    expert_w_buf = mps_tensor_to_metal_buffer(expert_weights.contiguous(), device)
    expert_s_buf = mps_tensor_to_metal_buffer(expert_scales.half().contiguous(), device)

    has_shared = 1 if shared_weights is not None else 0
    if shared_weights is not None:
        shared_w_buf = mps_tensor_to_metal_buffer(shared_weights.contiguous(), device)
        shared_s_buf = mps_tensor_to_metal_buffer(shared_scales.half().contiguous(), device)
    else:
        # Create dummy buffers for unused shared expert
        shared_w_buf = device.newBufferWithLength_options_(4, Metal.MTLResourceStorageModeShared)
        shared_s_buf = device.newBufferWithLength_options_(4, Metal.MTLResourceStorageModeShared)

    out_buf = mps_tensor_to_metal_buffer(output, device)

    # Create params buffer
    # struct MoEOptParams { batch_size, hidden_dim, out_dim, num_experts, top_k, group_size, has_shared }
    params = np.array(
        [batch_size, hidden_dim, out_dim, num_experts, top_k, group_size, has_shared],
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
        function_name="moe_dispatch_ultra_optimized",
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

    output = torch.empty((batch_size, out_dim), dtype=torch.float16, device="mps")

    act_buf = mps_tensor_to_metal_buffer(activations.half().contiguous(), device)
    expert_w_buf = mps_tensor_to_metal_buffer(expert_weights.contiguous(), device)
    expert_s_buf = mps_tensor_to_metal_buffer(expert_scales.half().contiguous(), device)
    ids_buf = mps_tensor_to_metal_buffer(expert_ids.int().contiguous(), device)
    probs_buf = mps_tensor_to_metal_buffer(expert_probs.half().contiguous(), device)

    has_shared = 1 if shared_weights is not None else 0
    if shared_weights is not None:
        shared_w_buf = mps_tensor_to_metal_buffer(shared_weights.contiguous(), device)
        shared_s_buf = mps_tensor_to_metal_buffer(shared_scales.half().contiguous(), device)
    else:
        shared_w_buf = device.newBufferWithLength_options_(4, Metal.MTLResourceStorageModeShared)
        shared_s_buf = device.newBufferWithLength_options_(4, Metal.MTLResourceStorageModeShared)

    out_buf = mps_tensor_to_metal_buffer(output, device)

    params = np.array(
        [batch_size, hidden_dim, out_dim, num_experts, top_k, group_size, has_shared],
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

    act_buf = mps_tensor_to_metal_buffer(activations.half().contiguous(), device)
    router_buf = mps_tensor_to_metal_buffer(router_weights.half().contiguous(), device)
    expert_w_buf = mps_tensor_to_metal_buffer(expert_weights.contiguous(), device)
    expert_s_buf = mps_tensor_to_metal_buffer(expert_scales.half().contiguous(), device)

    has_shared = 1 if shared_weights is not None else 0
    if shared_weights is not None:
        shared_w_buf = mps_tensor_to_metal_buffer(shared_weights.contiguous(), device)
        shared_s_buf = mps_tensor_to_metal_buffer(shared_scales.half().contiguous(), device)
    else:
        shared_w_buf = device.newBufferWithLength_options_(4, Metal.MTLResourceStorageModeShared)
        shared_s_buf = device.newBufferWithLength_options_(4, Metal.MTLResourceStorageModeShared)

    out_buf = mps_tensor_to_metal_buffer(output, device)

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
    activations = torch.randn(batch_size, hidden_dim, dtype=torch.float16, device="mps")
    router_weights = torch.randn(hidden_dim, num_experts, dtype=torch.float16, device="mps") * 0.01
    expert_weights = torch.randint(
        0, 2**32, (num_experts, k_packed, out_dim), dtype=torch.int32, device="mps"
    ).view(torch.uint32)
    expert_scales = (
        torch.randn(num_experts, num_groups, out_dim, dtype=torch.float16, device="mps") * 0.1 + 0.5
    ).abs()

    if has_shared:
        shared_weights = torch.randint(
            0, 2**32, (k_packed, out_dim), dtype=torch.int32, device="mps"
        ).view(torch.uint32)
        shared_scales = (
            torch.randn(num_groups, out_dim, dtype=torch.float16, device="mps") * 0.1 + 0.5
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
    flops_per_token = 2 * active_experts * hidden_dim * out_dim  # 2 for multiply-add
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
        * (k_packed * out_dim * 4 + num_groups * out_dim * 2)  # expert weights + scales
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
# Module-level singleton for convenience
# ---------------------------------------------------------------------------

_default_library: MetalKernelLibrary | None = None


def get_default_library() -> MetalKernelLibrary:
    """Get or create the default kernel library."""
    global _default_library
    if _default_library is None:
        _default_library = MetalKernelLibrary.from_source_dir()
    return _default_library


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
            print(f"  Effective bandwidth: {results['bandwidth_gb_s']:.1f} GB/s")
            print(f"  Dequant throughput: {results['throughput_gop_s']:.1f} GOP/s")
            print(f"  Output size: {results['output_mb']:.1f} MB")
