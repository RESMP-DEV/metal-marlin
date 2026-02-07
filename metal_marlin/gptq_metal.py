"""
Metal-accelerated GPTQ quantization operations.

Accelerates the compute-intensive parts of GPTQ:
- Hessian computation (H = 2 * X^T @ X)
- Cholesky decomposition for solving update equations
- Weight update computation

Usage:
    from metal_marlin.gptq_metal import GPTQMetal

    gptq = GPTQMetal()
    H = gptq.compute_hessian(activations)  # [in_features, in_features]
    L = gptq.cholesky_decompose(H)          # Lower triangular
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Dependency checks
try:
    import torch

    HAS_TORCH = True
    try:
        HAS_MPS = torch.backends.mps.is_available()
    except AttributeError:
        HAS_MPS = False
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False
    torch = None  # type: ignore[assignment]

try:
    import Metal

    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    Metal = None

logger = logging.getLogger(__name__)

_SHADER_DIR = Path(__file__).parent.parent / "src"
_FORCE_TORCH_MATMUL_ENV = "METAL_MARLIN_GPTQ_FORCE_TORCH_MATMUL"
_HESSIAN_MIN_FEATURES = 32
_HESSIAN_FEATURE_ALIGNMENT = 32
_HESSIAN_TILE_DIM = 64
_HESSIAN_THREADS_PER_TG = 128
_UINT32_MAX = (1 << 32) - 1
_DEFAULT_MAX_BUFFER_BYTES = 2 * 1024 * 1024 * 1024


def _resolve_includes(source: str, src_dir: Path) -> str:
    """Resolve #include directives in Metal shader source.

    Args:
        source: Metal shader source code
        src_dir: Directory to search for include files

    Returns:
        Source code with includes inlined
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

            include_path = src_dir / filename
            if not include_path.exists():
                raise FileNotFoundError(f"Include not found: {filename}")

            content = include_path.read_text()
            return resolve(content, depth + 1)

        return local_include_pattern.sub(replacer, src)

    return resolve(source)


def _preprocess_shader(source: str) -> str:
    """Preprocess Metal shader source by resolving includes.

    Args:
        source: Metal shader source code

    Returns:
        Preprocessed source code
    """
    return _resolve_includes(source, _SHADER_DIR)


class GPTQMetal:
    """Metal-accelerated GPTQ operations."""

    def __init__(self, device: torch.device | None = None):
        """Initialize GPTQ Metal dispatcher.

        Args:
            device: Metal device to use. If None, uses default system device.

        Raises:
            RuntimeError: If Metal or PyTorch MPS is not available.
        """
        if not HAS_METAL:
            raise RuntimeError(
                "GPTQMetal requires PyObjC Metal. Install with:\n"
                "  pip install pyobjc-framework-Metal"
            )
        if not HAS_MPS:
            raise RuntimeError("GPTQMetal requires PyTorch MPS backend (Apple Silicon).")

        self._device = device or Metal.MTLCreateSystemDefaultDevice()
        if self._device is None:
            raise RuntimeError("Failed to create Metal system default device.")
        self._command_queue = self._device.newCommandQueue()
        if self._command_queue is None:
            raise RuntimeError("Failed to create Metal command queue.")

        # Load and preprocess shader sources
        try:
            shader_path = _SHADER_DIR / "hessian.metal"
            if shader_path.exists():
                self._hessian_source = _preprocess_shader(shader_path.read_text())
            else:
                logger.warning(f"Hessian shader not found at {shader_path}")
                self._hessian_source = ""
        except Exception as e:
            logger.warning(f"Failed to load Hessian shader: {e}")
            self._hessian_source = ""
            
        try:
            cholesky_path = _SHADER_DIR / "cholesky.metal"
            if cholesky_path.exists():
                self._cholesky_source = _preprocess_shader(cholesky_path.read_text())
            else:
                self._cholesky_source = ""
        except Exception as e:
            logger.debug(f"Failed to load Cholesky shader (optional): {e}")
            self._cholesky_source = ""

        # Compile pipelines lazily
        self._pipelines: dict[str, Any] = {}
        self._uint32_constant_buffer_cache: dict[int, Any] = {}

    def _metal_max_buffer_length(self) -> int:
        """Best-effort query for maximum Metal buffer size."""
        if self._device is None:
            return _DEFAULT_MAX_BUFFER_BYTES
        max_len_attr = getattr(self._device, "maxBufferLength", None)
        try:
            max_len = max_len_attr() if callable(max_len_attr) else max_len_attr
            if max_len is not None:
                max_len_int = int(max_len)
                if max_len_int > 0:
                    return max_len_int
        except Exception:
            pass
        return _DEFAULT_MAX_BUFFER_BYTES

    def _allocate_shared_buffer(self, size: int, *, name: str) -> Any:
        """Allocate a shared Metal buffer with size validation."""
        if self._device is None:
            raise RuntimeError(f"Metal device unavailable while allocating {name}")
        if size <= 0:
            raise RuntimeError(f"Cannot allocate {name} with non-positive size: {size}")

        max_buffer = self._metal_max_buffer_length()
        if size > max_buffer:
            raise RuntimeError(
                f"{name} size {size} exceeds Metal max buffer length {max_buffer}"
            )

        buffer = self._device.newBufferWithLength_options_(
            size,
            Metal.MTLResourceStorageModeShared,
        )
        if buffer is None:
            raise RuntimeError(f"Failed to allocate shared Metal buffer for {name}")
        return buffer

    def _write_bytes_to_shared_buffer(
        self,
        buffer: Any,
        payload: bytes | bytearray | memoryview,
        *,
        name: str,
    ) -> None:
        """Copy payload bytes into a shared Metal buffer."""
        if buffer is None:
            raise RuntimeError(f"Cannot write {name}: destination buffer is None")

        payload_view = memoryview(payload)
        if payload_view.format != "B":
            payload_view = payload_view.cast("B")
        payload_len = payload_view.nbytes
        if payload_len <= 0:
            raise RuntimeError(f"Cannot write empty payload for {name}")

        try:
            buffer_len = int(buffer.length())
        except Exception as e:
            raise RuntimeError(f"Failed to query buffer length for {name}: {e}") from e
        if buffer_len < payload_len:
            raise RuntimeError(
                f"Buffer too small for {name}: buffer={buffer_len}, payload={payload_len}"
            )

        contents = buffer.contents()
        if contents is None:
            raise RuntimeError(f"Failed to get buffer contents for {name}")

        # Use memoryview for safe and efficient copy
        view = memoryview(contents.as_buffer(buffer_len))
        view[:payload_len] = payload_view

    def _get_uint32_constant_buffer(self, value: int) -> Any:
        """Get a cached Metal buffer containing a single uint32 constant."""
        if value in self._uint32_constant_buffer_cache:
            return self._uint32_constant_buffer_cache[value]
        if not (0 <= value <= _UINT32_MAX):
            raise RuntimeError(f"uint32 constant out of range: {value}")

        import struct

        data = struct.pack("I", value)
        buffer = self._allocate_shared_buffer(4, name=f"uint32_constant[{value}]")
        self._write_bytes_to_shared_buffer(buffer, data, name=f"uint32_constant[{value}]")

        self._uint32_constant_buffer_cache[value] = buffer
        return buffer

    def _should_force_torch_matmul(self) -> bool:
        return os.environ.get(_FORCE_TORCH_MATMUL_ENV, "0").strip() == "1"

    def _torch_hessian_matmul(self, X: torch.Tensor, normalize: bool) -> torch.Tensor:
        """Safe fallback path that keeps output on MPS with expected contract."""
        # Use float32 for accumulation to ensure numerical stability and MPS compatibility
        X_fp32 = X.to(device="mps", dtype=torch.float32)
        H = (2.0 * (X_fp32.T @ X_fp32)).to(device="mps", dtype=torch.float32).contiguous()
        if normalize:
            n_samples = int(X_fp32.shape[0])
            if n_samples <= 0:
                raise ValueError(f"n_samples must be positive for normalization, got {n_samples}")
            H /= float(n_samples)
        # Detach to prevent gradient tracking on Hessian output
        return H.detach()

    def _is_hessian_kernel_eligible(
        self,
        X: torch.Tensor,
        n_samples: int,
        in_features: int,
    ) -> bool:
        """Check if Metal Hessian kernel should be attempted.

        Returns False (fallback) on any suspicious condition to avoid crashes.
        """
        try:
            # Force torch matmul via environment variable override
            if self._should_force_torch_matmul():
                logger.debug("GPTQMetal: torch fallback forced via env var")
                return False

            # Metal device state
            if self._device is None or self._command_queue is None:
                logger.debug("GPTQMetal: Metal device or queue not available")
                return False
            if not callable(getattr(self._command_queue, "commandBuffer", None)):
                logger.debug("GPTQMetal: Metal command queue missing commandBuffer()")
                return False

            # Shader source availability
            if not self._hessian_source:
                logger.debug("GPTQMetal: Hessian shader source not available")
                return False

            if not isinstance(X, torch.Tensor):
                logger.debug("GPTQMetal: Input is not a torch.Tensor")
                return False

            # Tensor device checks
            if not hasattr(X, "is_mps") or not X.is_mps:
                logger.debug(f"GPTQMetal: Input not on MPS (is_mps={getattr(X, 'is_mps', False)})")
                return False
            if X.device.type != "mps":
                logger.debug(f"GPTQMetal: Input device not mps ({X.device})")
                return False
            if X.layout != torch.strided:
                logger.debug(f"GPTQMetal: Unsupported tensor layout {X.layout}")
                return False

            # Shape validity
            if n_samples <= 0 or in_features <= 0:
                return False
            if X.ndim != 2:
                return False
            if tuple(X.shape) != (n_samples, in_features):
                logger.debug(
                    "GPTQMetal: Input shape mismatch: tensor=%s expected=(%s, %s)",
                    tuple(X.shape),
                    n_samples,
                    in_features,
                )
                return False
            if int(X.numel()) <= 0:
                logger.debug("GPTQMetal: Input tensor has no elements")
                return False
            if n_samples > _UINT32_MAX or in_features > _UINT32_MAX:
                logger.debug(f"GPTQMetal: Dimensions exceed uint32 max ({n_samples}, {in_features})")
                return False
            
            # Element count check (prevent 32-bit overflow in shader indexing)
            num_elements = int(n_samples) * int(in_features)
            if num_elements > _UINT32_MAX:
                logger.debug(f"GPTQMetal: Element count {num_elements} exceeds uint32 limit")
                return False
            
            hessian_elements = int(in_features) * int(in_features)
            if hessian_elements > _UINT32_MAX:
                logger.debug(f"GPTQMetal: Hessian elements {hessian_elements} exceeds uint32 limit")
                return False

            # Conservative shape policy: keep the custom kernel for aligned
            # production-size cases and route edge shapes through torch fallback.
            if in_features < _HESSIAN_MIN_FEATURES:
                logger.debug(f"GPTQMetal: in_features {in_features} < min {_HESSIAN_MIN_FEATURES}")
                return False
            if in_features % _HESSIAN_FEATURE_ALIGNMENT != 0:
                logger.debug(f"GPTQMetal: in_features {in_features} not aligned to {_HESSIAN_FEATURE_ALIGNMENT}")
                return False

            # Dtype support checks
            if X.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                logger.debug(f"GPTQMetal: Unsupported dtype {X.dtype}")
                return False
            if not X.is_floating_point():
                logger.debug("GPTQMetal: Input tensor must be floating point")
                return False
            try:
                if not bool(torch.isfinite(X).all().item()):
                    logger.debug("GPTQMetal: Input contains NaN/Inf")
                    return False
            except Exception:
                logger.debug("GPTQMetal: Failed finite-value check", exc_info=True)
                return False

            # Range check for FP16 compatibility
            # If using FP16 kernel (for FP16/FP32 inputs), values must fit in FP16.
            # If input is BF16, we use BF16 kernel, so full BF16 range is fine.
            # If input is FP32, we must ensure it fits in FP16, OR we could convert to BF16?
            # Current policy: FP32 -> FP16 kernel. So check range.
            if X.dtype == torch.float32 or X.dtype == torch.float16:
                # Quick check on max absolute value to prevent Inf/NaN in kernel
                # This syncs with CPU, but is necessary for correctness.
                try:
                    max_val = X.abs().max().item()
                    if max_val > 65504:
                        logger.debug(f"GPTQMetal: Input range {max_val} exceeds fp16 max")
                        return False
                except Exception:
                    return False

            # Conservative memory-layout checks for safety
            if X.storage_offset() != 0:
                logger.debug("GPTQMetal: Input has non-zero storage offset")
                return False
            if not X.is_contiguous():
                logger.debug("GPTQMetal: Input is not contiguous")
                return False
            if X.stride(-1) != 1:
                return False

            # Shared staging buffers must fit device limits.
            max_buffer_size = self._metal_max_buffer_length()
            x_bytes = num_elements * int(X.element_size())
            h_bytes = hessian_elements * 4  # float32 output
            if x_bytes <= 0 or h_bytes <= 0:
                return False
            if x_bytes > max_buffer_size or h_bytes > max_buffer_size:
                logger.debug(f"GPTQMetal: Buffer size exceeds limit ({x_bytes}, {h_bytes} vs {max_buffer_size})")
                return False

            if not hasattr(torch, "mps") or not callable(getattr(torch.mps, "synchronize", None)):
                logger.debug("GPTQMetal: torch.mps.synchronize unavailable")
                return False

            # All checks passed
            return True

        except Exception as e:
            # Any unexpected error during eligibility -> fallback
            logger.debug(f"GPTQMetal: eligibility check exception: {e}")
            return False

    @staticmethod
    def _torch_dtype_to_numpy_dtype(dtype: torch.dtype) -> np.dtype:
        mapping: dict[torch.dtype, np.dtype] = {
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.uint8: np.uint8,
            torch.bool: np.bool_,
            # BFloat16 not supported directly in numpy, handled via view
        }
        np_dtype = mapping.get(dtype)
        if np_dtype is None:
            raise RuntimeError(f"Unsupported dtype for buffer copy-back: {dtype}")
        return np_dtype

    def _bridge_mps_tensor_copy(
        self,
        tensor: torch.Tensor,
        *,
        name: str,
        output_only: bool = False,
    ) -> tuple[torch.Tensor, Any, list[Any]]:
        """Bridge an MPS tensor to Metal via explicit staging-buffer copy.

        Returns:
            Tuple of (tensor, buffer, lifetime_holders). The holders list ensures
            intermediate CPU tensors stay alive during Metal dispatch.
        """
        if tensor is None:
            raise RuntimeError(f"{name} is None")
        if not hasattr(tensor, "is_mps"):
            raise RuntimeError(f"{name} does not have 'is_mps' attribute")
        if not tensor.is_mps:
            raise RuntimeError(f"{name} must be on MPS device")
        if self._device is None:
            raise RuntimeError("Metal device is not available")

        try:
            tensor = tensor.contiguous()
        except Exception as e:
            raise RuntimeError(f"Failed to make {name} contiguous: {e}") from e

        numel = int(tensor.numel())
        elem_size = int(tensor.element_size())
        size = numel * elem_size
        
        buffer = self._allocate_shared_buffer(size, name=name)

        if output_only:
            return tensor, buffer, []

        lifetime_holders: list[Any] = []
        try:
            # Detach and move to CPU for staging.
            cpu_tensor = tensor.detach().cpu().contiguous()
            
            # Handle BF16 by viewing as int16 (since numpy lacks bf16 support)
            # This preserves the bits exactly for the Metal kernel.
            if cpu_tensor.dtype == torch.bfloat16:
                cpu_tensor_view = cpu_tensor.view(torch.int16)
            else:
                cpu_tensor_view = cpu_tensor

            numpy_array = cpu_tensor_view.numpy()
            if not numpy_array.flags["C_CONTIGUOUS"]:
                numpy_array = np.ascontiguousarray(numpy_array)

            # Copy using memoryview to avoid huge .tobytes() allocation
            # self._write_bytes_to_shared_buffer supports memoryview/buffer protocol
            self._write_bytes_to_shared_buffer(buffer, numpy_array, name=name)

            # Keep host references alive until dispatch finishes.
            lifetime_holders = [cpu_tensor, cpu_tensor_view, numpy_array]

        except Exception as e:
            raise RuntimeError(f"Failed to stage tensor data for {name}: {e}") from e

        return tensor, buffer, lifetime_holders

    @staticmethod
    def _copy_shared_buffer_bytes_to_host(buffer: Any, size: int, *, name: str) -> bytes:
        """Copy bytes out of a shared Metal buffer into host-owned memory."""
        # Note: This returns a copy (bytes object). For large buffers, direct numpy read is preferred.
        if size <= 0:
            raise RuntimeError(f"{name}: Requested copy size must be positive, got {size}")
        try:
            buffer_length = int(buffer.length())
            if buffer_length < size:
                raise RuntimeError(
                    f"{name}: Buffer too small for host copy: buffer={buffer_length}, size={size}"
                )
            contents = buffer.contents()
            if contents is None:
                raise RuntimeError(f"{name}: Buffer contents is None")
            view = memoryview(contents.as_buffer(buffer_length))
            return bytes(view[:size])
        except Exception as e:
            raise RuntimeError(f"{name}: Failed copying shared buffer to host bytes: {e}") from e

    def _validate_buffer_before_copy(self, buffer: Any, tensor: torch.Tensor, *, name: str) -> None:
        """Validate Metal buffer is compatible with tensor before copying.

        Args:
            buffer: Metal buffer to validate
            tensor: Target tensor
            name: Name for error messages

        Raises:
            RuntimeError: If buffer is incompatible
        """
        if buffer is None:
            raise RuntimeError(f"{name}: Metal buffer is None")
        
        try:
            buffer_length = int(buffer.length())
            if buffer_length <= 0:
                raise RuntimeError(f"{name}: Buffer length {buffer_length} must be positive")
            
            expected_size = int(tensor.numel()) * int(tensor.element_size())
            if buffer_length < expected_size:
                raise RuntimeError(
                    f"{name}: Buffer too small for tensor: "
                    f"buffer={buffer_length}, tensor={expected_size}"
                )
            
            contents = buffer.contents()
            if contents is None:
                raise RuntimeError(f"{name}: Buffer contents is None")
        except Exception as e:
            raise RuntimeError(f"{name}: Failed to validate Metal buffer: {e}") from e

    def _copy_metal_buffer_to_mps_tensor(self, buffer: Any, tensor: torch.Tensor, *, name: str) -> None:
        """Copy Metal shared-buffer output back into an MPS tensor.

        Args:
            buffer: Metal buffer to copy from
            tensor: Target MPS tensor
            name: Name for error messages

        Raises:
            RuntimeError: If copy fails
        """
        if tensor is None:
            raise RuntimeError(f"{name}: Target tensor is None")
        
        # Validate buffer before attempting copy
        self._validate_buffer_before_copy(buffer, tensor, name=name)

        try:
            expected_size = int(tensor.numel()) * int(tensor.element_size())
            host_bytes = self._copy_shared_buffer_bytes_to_host(
                buffer,
                expected_size,
                name=f"{name}_copy",
            )

            np_dtype = self._torch_dtype_to_numpy_dtype(tensor.dtype)
            
            # Create a host-owned copy before moving back to MPS.
            # This avoids lifetime hazards from views tied to Metal shared buffers.
            raw_array = np.frombuffer(host_bytes, dtype=np_dtype, count=tensor.numel())
            host_array = raw_array.reshape(tuple(tensor.shape)).copy()

            # Copy back to MPS
            tensor.copy_(torch.from_numpy(host_array).to(device=tensor.device, dtype=tensor.dtype))
        except Exception as e:
            raise RuntimeError(f"{name}: Failed to copy buffer to MPS tensor: {e}") from e

    def _validate_command_buffer(self, command_buffer: Any, *, kernel_name: str) -> None:
        """Raise if the command buffer did not complete successfully."""
        if command_buffer is None:
            raise RuntimeError(f"Command buffer is None for {kernel_name}")

        try:
            status = command_buffer.status()
        except Exception as e:
            raise RuntimeError(f"Failed to get command buffer status for {kernel_name}: {e}") from e

        completed = getattr(Metal, "MTLCommandBufferStatusCompleted", 4)
        if status != completed:
            error_obj = None
            try:
                error_obj = command_buffer.error()
            except Exception:
                pass
            raise RuntimeError(
                f"Metal command buffer failed for {kernel_name}: status={status}, error={error_obj}"
            )

    @staticmethod
    def _validate_hessian_output_contract(H: torch.Tensor, *, in_features: int) -> None:
        """Validate that Hessian output meets the expected contract.

        Args:
            H: Hessian matrix to validate
            in_features: Expected feature dimension

        Raises:
            RuntimeError: If output contract is violated
        """
        expected_shape = (in_features, in_features)
        if tuple(H.shape) != expected_shape:
            raise RuntimeError(
                f"Hessian shape mismatch: expected {expected_shape}, got {tuple(H.shape)}"
            )
        if H.device.type != "mps":
            raise RuntimeError(f"Hessian must be on MPS device, got {H.device}")
        if not H.is_contiguous():
            raise RuntimeError("Hessian must be contiguous")
        if H.dtype not in (torch.float16, torch.float32):
            raise RuntimeError(
                f"Hessian must be float16 or float32, got {H.dtype}"
            )
        if not torch.isfinite(H).all():
            raise RuntimeError("Hessian contains NaN or Inf values")

    def _validate_hessian_numerical_correctness(
        self,
        H_metal: torch.Tensor,
        X: torch.Tensor,
        n_samples: int,
        normalize: bool,
    ) -> bool:
        """Validate that Metal Hessian output is numerically correct.

        Performs lightweight checks to detect Metal kernel bugs or buffer corruption
        without requiring full recomputation.

        Args:
            H_metal: Hessian matrix from Metal kernel
            X: Input activation matrix [n_samples, in_features]
            n_samples: Number of samples
            normalize: Whether normalization was applied

        Returns:
            True if Metal output appears correct, False otherwise
        """
        try:
            # Check 1: Symmetry (Hessian should be symmetric)
            # Allow some numerical tolerance for FP16
            atol = 1e-2 if H_metal.dtype == torch.float16 else 1e-4
            asymmetry = (H_metal - H_metal.T).abs().max().item()
            if asymmetry > atol:
                logger.debug(
                    f"Metal Hessian symmetry check failed: max asymmetry={asymmetry} > {atol}"
                )
                return False

            # Check 2: Diagonal should be non-negative (H = X^T @ X is PSD)
            diagonal = torch.diagonal(H_metal)
            if (diagonal < -1e-3).any():
                logger.debug(
                    f"Metal Hessian has negative diagonal values: {diagonal.min().item()}"
                )
                return False

            # Check 3: Spot check with torch computation.
            # Use ALL samples for the probe features so normalization matches exactly.
            check_size = min(8, int(H_metal.shape[0]))
            if check_size <= 0:
                return False
            X_check = X[:, :check_size].to(dtype=torch.float32)
            H_ref_check = (2.0 * (X_check.T @ X_check)).to(device="mps", dtype=torch.float32)
            if normalize:
                H_ref_check /= float(n_samples)

            # Compare the submatrix
            H_metal_check = H_metal[:check_size, :check_size].to(dtype=torch.float32)
            # Use tolerance based on dtype
            rtol = 8e-2 if H_metal.dtype == torch.float16 else 2e-2
            if not torch.allclose(H_metal_check, H_ref_check, atol=atol, rtol=rtol):
                max_diff = (H_metal_check - H_ref_check).abs().max().item()
                logger.debug(
                    f"Metal Hessian spot check failed: max diff={max_diff} > {atol} (rtol={rtol})"
                )
                return False

            # All checks passed
            return True
        except Exception as e:
            logger.debug(f"Numerical validation raised exception: {e}")
            return False

    def _get_pipeline(self, source: str, function_name: str):
        """Compile and cache a compute pipeline."""
        if function_name in self._pipelines:
            return self._pipelines[function_name]

        options = Metal.MTLCompileOptions.new()
        library, error = self._device.newLibraryWithSource_options_error_(source, options, None)
        if library is None:
            raise RuntimeError(f"Failed to compile Metal shader for {function_name}: {error}")

        function = library.newFunctionWithName_(function_name)
        if function is None:
            raise RuntimeError(f"Function {function_name} not found in shader")

        pipeline, error = self._device.newComputePipelineStateWithFunction_error_(function, None)
        if pipeline is None:
            raise RuntimeError(f"Failed to create pipeline for {function_name}: {error}")

        self._pipelines[function_name] = pipeline
        return pipeline

    @staticmethod
    def _pipeline_max_threads(pipeline: Any) -> int | None:
        """Best-effort query for pipeline max threads per threadgroup."""
        max_threads_attr = getattr(pipeline, "maxTotalThreadsPerThreadgroup", None)
        try:
            if callable(max_threads_attr):
                max_threads = int(max_threads_attr())
            elif max_threads_attr is None:
                return None
            else:
                max_threads = int(max_threads_attr)
        except Exception:
            return None
        return max_threads if max_threads > 0 else None

    def _validate_dispatch_config(
        self,
        *,
        kernel_name: str,
        grid_x: int,
        grid_y: int,
        tg_threads: int,
        pipeline: Any,
    ) -> None:
        """Validate dispatch geometry before touching encoder state."""
        if grid_x <= 0 or grid_y <= 0:
            raise RuntimeError(f"{kernel_name}: invalid dispatch grid ({grid_x}, {grid_y})")
        if grid_x > _UINT32_MAX or grid_y > _UINT32_MAX:
            raise RuntimeError(f"{kernel_name}: dispatch grid exceeds uint32 ({grid_x}, {grid_y})")
        if tg_threads <= 0:
            raise RuntimeError(f"{kernel_name}: invalid threadgroup size {tg_threads}")

        max_threads = self._pipeline_max_threads(pipeline)
        if max_threads is not None and tg_threads > max_threads:
            raise RuntimeError(
                f"{kernel_name}: threadgroup size {tg_threads} exceeds pipeline max {max_threads}"
            )

    def _new_command_encoder(self, *, kernel_name: str) -> tuple[Any, Any]:
        """Create a command buffer and compute encoder with validation."""
        if self._command_queue is None:
            raise RuntimeError(f"{kernel_name}: Metal command queue is unavailable")
        command_buffer = self._command_queue.commandBuffer()
        if command_buffer is None:
            raise RuntimeError(f"{kernel_name}: failed to create command buffer")
        encoder = command_buffer.computeCommandEncoder()
        if encoder is None:
            raise RuntimeError(f"{kernel_name}: failed to create compute encoder")
        return command_buffer, encoder

    @staticmethod
    def _end_encoding(encoder: Any, *, kernel_name: str) -> None:
        """End Metal command encoding if encoder was created."""
        if encoder is None:
            return
        try:
            encoder.endEncoding()
        except Exception as e:
            raise RuntimeError(f"{kernel_name}: failed to end encoding: {e}") from e

    def _dispatch_hessian_compute(
        self,
        X: torch.Tensor,
        n_samples: int,
        in_features: int,
    ) -> torch.Tensor:
        """Dispatch hessian_compute kernel with guarded execution."""
        tile_dim = _HESSIAN_TILE_DIM
        tg_threads = _HESSIAN_THREADS_PER_TG
        lifetime_holders: list[Any] = []

        # Synchronize MPS before custom Metal execution
        torch.mps.synchronize()

        try:
            # Determine correct kernel based on dtype
            # If BFloat16, use 'hessian_compute' (native BF16 support)
            # If Float16, use 'hessian_compute_fp16'
            # If Float32, convert to Float16 (if range permits) and use 'hessian_compute_fp16'
            
            if X.dtype == torch.bfloat16:
                kernel_name = "hessian_compute"
                X_work = X  # Pass as-is (bridge handles view as int16)
            elif X.dtype == torch.float32:
                # We already checked range in eligibility, so FP16 is safe and offers better precision than BF16
                kernel_name = "hessian_compute_fp16"
                X_work = X.to(torch.float16)
            else:
                kernel_name = "hessian_compute_fp16"
                X_work = X
            
            # Ensure contiguous
            if not X_work.is_contiguous():
                X_work = X_work.contiguous()

            H = torch.zeros((in_features, in_features), dtype=torch.float32, device=X.device).contiguous()
            
            # Bridge tensors
            _, X_buffer, X_holders = self._bridge_mps_tensor_copy(X_work, name="X_work")
            H, H_buffer, H_holders = self._bridge_mps_tensor_copy(H, name="H", output_only=True)
            lifetime_holders.extend(X_holders)
            lifetime_holders.extend(H_holders)

            n_samples_buf = self._get_uint32_constant_buffer(n_samples)
            in_features_buf = self._get_uint32_constant_buffer(in_features)

            pipeline = self._get_pipeline(self._hessian_source, kernel_name)

            grid_x = (in_features + tile_dim - 1) // tile_dim
            grid_y = (in_features + tile_dim - 1) // tile_dim
            self._validate_dispatch_config(
                kernel_name=kernel_name,
                grid_x=grid_x,
                grid_y=grid_y,
                tg_threads=tg_threads,
                pipeline=pipeline,
            )

            command_buffer, encoder = self._new_command_encoder(kernel_name=kernel_name)
            
            try:
                encoder.setComputePipelineState_(pipeline)
                encoder.setBuffer_offset_atIndex_(X_buffer, 0, 0)
                encoder.setBuffer_offset_atIndex_(H_buffer, 0, 1)
                encoder.setBuffer_offset_atIndex_(n_samples_buf, 0, 2)
                encoder.setBuffer_offset_atIndex_(in_features_buf, 0, 3)

                grid_size = Metal.MTLSizeMake(grid_x, grid_y, 1)
                tg_size = Metal.MTLSizeMake(tg_threads, 1, 1)
                encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)
            finally:
                self._end_encoding(encoder, kernel_name=kernel_name)

            command_buffer.commit()
            command_buffer.waitUntilCompleted()
            self._validate_command_buffer(command_buffer, kernel_name=kernel_name)

            # Sync after Metal work
            torch.mps.synchronize()

            # Copy result back
            self._copy_metal_buffer_to_mps_tensor(H_buffer, H, name="H")
            
            # Validate output meets expected contract before returning
            self._validate_hessian_output_contract(H, in_features=in_features)
            
            return H
        except Exception as dispatch_error:
            # Re-raise with context
            raise RuntimeError(
                f"Metal Hessian dispatch failed: {dispatch_error}"
            ) from dispatch_error
        finally:
            # Explicitly keep references alive until this point
            _ = lifetime_holders

    def _dispatch_hessian_normalize(
        self,
        H: torch.Tensor,
        in_features: int,
        n_samples: int,
    ) -> None:
        """Dispatch hessian_normalize kernel with guarded execution."""
        tile_dim = _HESSIAN_TILE_DIM
        tg_threads = _HESSIAN_THREADS_PER_TG
        lifetime_holders: list[Any] = []

        torch.mps.synchronize()

        try:
            H_work = H if H.is_contiguous() else H.contiguous()
            _, H_buffer, H_holders = self._bridge_mps_tensor_copy(H_work, name="H_work")
            lifetime_holders.extend(H_holders)

            in_features_buf = self._get_uint32_constant_buffer(in_features)
            n_samples_buf = self._get_uint32_constant_buffer(n_samples)

            kernel_name = "hessian_normalize"
            pipeline = self._get_pipeline(self._hessian_source, kernel_name)

            grid_x = (in_features + tile_dim - 1) // tile_dim
            grid_y = (in_features + tile_dim - 1) // tile_dim
            self._validate_dispatch_config(
                kernel_name=kernel_name,
                grid_x=grid_x,
                grid_y=grid_y,
                tg_threads=tg_threads,
                pipeline=pipeline,
            )

            command_buffer, encoder = self._new_command_encoder(kernel_name=kernel_name)
            
            try:
                encoder.setComputePipelineState_(pipeline)
                encoder.setBuffer_offset_atIndex_(H_buffer, 0, 0)
                encoder.setBuffer_offset_atIndex_(in_features_buf, 0, 1)
                encoder.setBuffer_offset_atIndex_(n_samples_buf, 0, 2)

                grid_size = Metal.MTLSizeMake(grid_x, grid_y, 1)
                tg_size = Metal.MTLSizeMake(tg_threads, 1, 1)
                encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)
            finally:
                self._end_encoding(encoder, kernel_name=kernel_name)

            command_buffer.commit()
            command_buffer.waitUntilCompleted()
            self._validate_command_buffer(command_buffer, kernel_name=kernel_name)

            torch.mps.synchronize()
            self._copy_metal_buffer_to_mps_tensor(H_buffer, H_work, name="H_work")

            if H_work is not H:
                H.copy_(H_work)
            
            # Validate the normalized Hessian still meets contract
            self._validate_hessian_output_contract(H, in_features=in_features)
        except Exception as dispatch_error:
            # Re-raise with context
            raise RuntimeError(
                f"Metal Hessian normalization failed: {dispatch_error}"
            ) from dispatch_error
        finally:
            # Explicitly keep references alive
            _ = lifetime_holders

    def _guarded_metal_hessian_dispatch(
        self, X: torch.Tensor, n_samples: int, in_features: int, normalize: bool
    ) -> torch.Tensor | None:
        """
        A wrapper for the Metal Hessian dispatch that catches all exceptions
        and returns None on failure, preventing crashes from propagating.
        
        Also performs numerical validation to detect incorrect Metal output.
        """
        try:
            H = self._dispatch_hessian_compute(X, n_samples, in_features)
            if normalize:
                self._dispatch_hessian_normalize(H, in_features, n_samples)
            
            # Validate basic output contract
            self._validate_hessian_output_contract(H, in_features=in_features)
            
            # Perform numerical correctness checks to detect Metal kernel bugs
            # or buffer corruption issues
            if not self._validate_hessian_numerical_correctness(H, X, n_samples, normalize):
                logger.warning(
                    "Metal Hessian failed numerical validation, falling back to torch. "
                    "This indicates Metal kernel bug or buffer corruption."
                )
                return None
            
            return H.detach()
        except Exception as e:
            logger.warning(
                "Metal Hessian kernel failed with an exception, falling back to torch. "
                f"Error: {e}",
                exc_info=True
            )
            return None

    def compute_hessian(
        self,
        activations: torch.Tensor | NDArray,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Compute Hessian matrix H = 2 * X^T @ X on Metal GPU.

        Includes extensive safety checks and robust fallback to torch.matmul.
        Preserves environment variable override METAL_MARLIN_GPTQ_FORCE_TORCH_MATMUL=1.
        """
        if isinstance(activations, np.ndarray):
            X = torch.from_numpy(activations)
        else:
            X = activations

        if X.ndim != 2:
            raise ValueError(
                f"compute_hessian expects a 2D activation matrix, got shape {tuple(X.shape)}"
            )

        # Ensure we are on MPS and in compatible dtype for initial processing
        if X.dtype == torch.float64:
            X = X.to(torch.float32)
        X = X.to("mps")

        n_samples, in_features = X.shape

        # 1. Environment variable force fallback (highest priority)
        if self._should_force_torch_matmul():
            logger.debug("GPTQMetal: Force fallback via METAL_MARLIN_GPTQ_FORCE_TORCH_MATMUL=1")
            H_forced = self._torch_hessian_matmul(X, normalize=normalize)
            self._validate_hessian_output_contract(H_forced, in_features=in_features)
            return H_forced

        # 2. Attempt Metal kernel path if eligible
        if self._is_hessian_kernel_eligible(X, n_samples, in_features):
            # Guarded dispatch prevents kernel crashes from taking down the interpreter
            result = self._guarded_metal_hessian_dispatch(X, n_samples, in_features, normalize)
            if result is not None:
                return result
            # If result is None, it means the guarded dispatch failed, and we'll use the fallback.
            logger.debug("GPTQMetal: Metal dispatch failed, proceeding with torch fallback.")

        # 3. Fallback path for ineligibility or Metal kernel failure
        H_fallback = self._torch_hessian_matmul(X, normalize=normalize)
        self._validate_hessian_output_contract(H_fallback, in_features=in_features)
        return H_fallback

    def cholesky_decompose(
        self,
        H: torch.Tensor,
        regularization: float = 1e-6,
    ) -> torch.Tensor:
        """Compute Cholesky decomposition H = L @ L^T."""
        H_reg = H + regularization * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
        return torch.linalg.cholesky(H_reg)

    def quantize_weight_gptq(
        self,
        weight: torch.Tensor,
        H: torch.Tensor,
        bits: int = 4,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a weight matrix using GPTQ algorithm."""
        from .gptq import GPTQQuantizer

        _, in_features = weight.shape
        original_device = weight.device
        output_dtype = weight.dtype if weight.dtype.is_floating_point else torch.float32

        weight_np = weight.detach().cpu().to(dtype=torch.float64).numpy()
        H_np = H.detach().cpu().to(dtype=torch.float64).numpy()

        group_size = blocksize if in_features % blocksize == 0 else -1
        quantizer = GPTQQuantizer(
            bits=bits,
            group_size=group_size,
            sym=True,
            actorder=True,
            damp=percdamp,
            block_size=blocksize,
        )

        result = quantizer.quantize_weight(weight_np, H_np)

        quantized_weight = torch.from_numpy(result.Q).to(device=original_device, dtype=output_dtype).contiguous()
        scales = torch.from_numpy(result.scales).to(device=original_device, dtype=output_dtype).contiguous()
        zeros = (
            torch.from_numpy(result.zeros).to(device=original_device, dtype=output_dtype).contiguous()
            if result.zeros is not None else torch.zeros_like(scales)
        )

        return quantized_weight, scales, zeros


# Convenience functions
def compute_hessian_metal(activations: torch.Tensor | NDArray, normalize: bool = True) -> torch.Tensor:
    return GPTQMetal().compute_hessian(activations, normalize)


def cholesky_decompose(H: torch.Tensor | NDArray, regularization: float = 1e-6) -> NDArray[np.float64]:
    gptq = GPTQMetal()
    if isinstance(H, np.ndarray):
        H = torch.from_numpy(H.astype(np.float32)).to("mps")
    return gptq.cholesky_decompose(H, regularization).cpu().numpy().astype(np.float64)
