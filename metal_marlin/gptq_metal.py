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

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Dependency checks
try:
    import torch

    HAS_TORCH = True
    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False
    torch = None  # type: ignore[assignment]

try:
    import Foundation
    import Metal

    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    Metal = None
    Foundation = None  # type: ignore[assignment]

_SHADER_DIR = Path(__file__).parent.parent / "src"


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
        self._command_queue = self._device.newCommandQueue()

        # Load and preprocess shader sources
        self._hessian_source = _preprocess_shader((_SHADER_DIR / "hessian.metal").read_text())
        self._cholesky_source = _preprocess_shader(
            (_SHADER_DIR / "cholesky.metal").read_text()
            if (_SHADER_DIR / "cholesky.metal").exists()
            else ""
        )

        # Compile pipelines lazily
        self._pipelines: dict[str, Any] = {}

    def _get_pipeline(self, source: str, function_name: str):
        """Compile and cache a compute pipeline."""
        # Check cache
        if function_name in self._pipelines:
            return self._pipelines[function_name]

        options = Metal.MTLCompileOptions.new()
        library, error = self._device.newLibraryWithSource_options_error_(source, options, None)
        if library is None:
            error_msg = str(error) if error else "Unknown error"
            raise RuntimeError(f"Failed to compile Metal shader for {function_name}: {error_msg}")

        function = library.newFunctionWithName_(function_name)
        if function is None:
            raise RuntimeError(f"Function {function_name} not found in shader")

        pipeline, error = self._device.newComputePipelineStateWithFunction_error_(function, None)
        if pipeline is None:
            error_msg = str(error) if error else "Unknown error"
            raise RuntimeError(f"Failed to create pipeline for {function_name}: {error_msg}")

        # Cache the pipeline
        self._pipelines[function_name] = pipeline
        return pipeline

    def _dispatch_hessian_compute(
        self,
        X: torch.Tensor,
        n_samples: int,
        in_features: int,
    ) -> torch.Tensor:
        """Dispatch hessian_compute kernel.

        Args:
            X: Input activations [n_samples, in_features] on MPS
            n_samples: Number of samples (rows)
            in_features: Hidden dimension (columns)

        Returns:
            Hessian matrix [in_features, in_features] on MPS
        """
        TILE_DIM = 64
        TILE_K = 16

        H = torch.zeros((in_features, in_features), dtype=torch.float32, device=X.device)

        # Ensure contiguous and synchronize MPS
        X = X.contiguous()
        H = H.contiguous()

        # Synchronize MPS before accessing data
        torch.mps.synchronize()

        # Create Metal buffer for X and H
        # Convert X to BF16 bytes (shader expects BF16 input)
        if X.dtype == torch.float16:
            X_np = X.detach().cpu().numpy().astype(np.float16).view(np.uint16)
        else:
            X_np = X.detach().cpu().numpy().astype(np.float16).view(np.uint16)
        H_np = H.detach().cpu().numpy()

        X_buffer = self._device.newBufferWithBytes_length_options_(
            X_np.tobytes(),
            X_np.nbytes,
            Metal.MTLResourceStorageModeShared,
        )
        H_buffer = self._device.newBufferWithBytes_length_options_(
            H_np.tobytes(),
            H_np.nbytes,
            Metal.MTLResourceStorageModeShared,
        )

        # Create constant buffers using struct pack
        import struct

        n_samples_bytes = struct.pack("I", n_samples)
        in_features_bytes = struct.pack("I", in_features)

        n_samples_buf = self._device.newBufferWithBytes_length_options_(
            n_samples_bytes,
            4,
            Metal.MTLResourceStorageModeShared,
        )
        in_features_buf = self._device.newBufferWithBytes_length_options_(
            in_features_bytes,
            4,
            Metal.MTLResourceStorageModeShared,
        )

        # Always use BF16 kernel (handles both BF16 and FP16 by converting internally)
        # TODO: Fix hessian_compute_fp16 kernel compilation issues
        kernel_name = "hessian_compute"

        # Compile pipeline
        pipeline = self._get_pipeline(self._hessian_source, kernel_name)

        # Grid: ceil(in_features / TILE_DIM) × ceil(in_features / TILE_DIM)
        grid_x = (in_features + TILE_DIM - 1) // TILE_DIM
        grid_y = (in_features + TILE_DIM - 1) // TILE_DIM

        # Threadgroup: 128 threads (4 simdgroups × 32)
        tg_threads = 128

        # Dispatch
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(X_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(H_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(n_samples_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(in_features_buf, 0, 3)

        grid_size = Metal.MTLSizeMake(grid_x, grid_y, 1)
        tg_size = Metal.MTLSizeMake(tg_threads, 1, 1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)

        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        # Copy result back to torch tensor
        result_bytes = H_buffer.contents().as_buffer(H_np.nbytes)
        H_np[:] = np.frombuffer(result_bytes, dtype=np.float32).reshape(H_np.shape)
        H = torch.from_numpy(H_np).to(X.device)

        return H

    def _dispatch_hessian_normalize(
        self,
        H: torch.Tensor,
        in_features: int,
        n_samples: int,
    ) -> None:
        """Dispatch hessian_normalize kernel (in-place division by n_samples).

        Args:
            H: Hessian matrix [in_features, in_features] on MPS (modified in-place)
            in_features: Hidden dimension
            n_samples: Number of samples to divide by
        """
        TILE_DIM = 64

        H = H.contiguous()
        H_np = H.detach().cpu().numpy()

        H_buffer = self._device.newBufferWithBytes_length_options_(
            H_np.tobytes(),
            H_np.nbytes,
            Metal.MTLResourceStorageModeShared,
        )

        import struct

        in_features_bytes = struct.pack("I", in_features)
        n_samples_bytes = struct.pack("I", n_samples)

        in_features_buf = self._device.newBufferWithBytes_length_options_(
            in_features_bytes,
            4,
            Metal.MTLResourceStorageModeShared,
        )
        n_samples_buf = self._device.newBufferWithBytes_length_options_(
            n_samples_bytes,
            4,
            Metal.MTLResourceStorageModeShared,
        )

        pipeline = self._get_pipeline(self._hessian_source, "hessian_normalize")

        grid_x = (in_features + TILE_DIM - 1) // TILE_DIM
        grid_y = (in_features + TILE_DIM - 1) // TILE_DIM
        tg_threads = 128

        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(H_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(in_features_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(n_samples_buf, 0, 2)

        grid_size = Metal.MTLSizeMake(grid_x, grid_y, 1)
        tg_size = Metal.MTLSizeMake(tg_threads, 1, 1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)

        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        # Copy result back
        result_bytes = H_buffer.contents().as_buffer(H_np.nbytes)
        H_np[:] = np.frombuffer(result_bytes, dtype=np.float32).reshape(H_np.shape)
        H.copy_(torch.from_numpy(H_np).to(H.device))

    def compute_hessian(
        self,
        activations: torch.Tensor | NDArray,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Compute Hessian matrix H = 2 * X^T @ X on Metal GPU.

        Args:
            activations: Input activations [n_samples, in_features]
            normalize: If True, divide by n_samples

        Returns:
            Hessian matrix [in_features, in_features] on MPS device
        """
        # Convert to torch tensor on MPS
        if isinstance(activations, np.ndarray):
            X = torch.from_numpy(activations).to("mps")
        else:
            X = activations.to("mps")

        if X.ndim != 2:
            raise ValueError(f"activations must be 2D, got {X.shape}")

        n_samples, in_features = X.shape

        # Temporarily use PyTorch MPS matmul for correctness
        # The Metal shader integration needs further debugging
        H = 2.0 * (X.T @ X)

        if normalize:
            H = H / n_samples

        return H

    def cholesky_decompose(
        self,
        H: torch.Tensor,
        regularization: float = 1e-6,
    ) -> torch.Tensor:
        """Compute Cholesky decomposition H = L @ L^T.

        Args:
            H: Symmetric positive definite matrix [n, n]
            regularization: Small value added to diagonal for numerical stability

        Returns:
            Lower triangular matrix L [n, n]
        """
        # Add regularization to diagonal
        H_reg = H + regularization * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)

        # Use torch.linalg.cholesky (MPS accelerated in PyTorch 2.0+)
        # Custom cholesky.metal provides finer control for very large matrices
        L = torch.linalg.cholesky(H_reg)

        return L

    def quantize_weight_gptq(
        self,
        weight: torch.Tensor,
        H: torch.Tensor,
        bits: int = 4,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a weight matrix using GPTQ algorithm.

        Args:
            weight: Weight matrix [out_features, in_features]
            H: Hessian matrix [in_features, in_features]
            bits: Quantization bits (4 or 8)
            blocksize: Block size for sequential quantization
            percdamp: Percentage of mean diagonal for damping

        Returns:
            Tuple of (quantized_weight, scales, zeros)
        """
        # Full GPTQ implementation - see gptq.py for reference
        # This is a placeholder for the Metal-accelerated version
        raise NotImplementedError(
            "Full GPTQ quantization not yet implemented in Metal. "
            "Use gptq.quantize_layer_gptq() for CPU version."
        )


# Module-level convenience functions
def compute_hessian_metal(
    activations: torch.Tensor | NDArray,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute Hessian on Metal GPU (convenience function)."""
    gptq = GPTQMetal()
    return gptq.compute_hessian(activations, normalize)


def cholesky_decompose(
    H: torch.Tensor | NDArray,
    regularization: float = 1e-6,
) -> NDArray[np.float64]:
    """Compute Cholesky decomposition on Metal GPU (convenience function).

    Args:
        H: Symmetric positive definite matrix [n, n] (numpy array or torch tensor)
        regularization: Small value added to diagonal for numerical stability

    Returns:
        Lower triangular matrix L [n, n] as numpy array (float64)
    """
    gptq = GPTQMetal()

    # Convert numpy array to torch tensor if needed
    # MPS doesn't support float64, so convert to float32
    if isinstance(H, np.ndarray):
        H = torch.from_numpy(H.astype(np.float32)).to("mps")

    L_torch = gptq.cholesky_decompose(H, regularization)

    # Return as float64 numpy array for compatibility with gptq.py
    return L_torch.cpu().numpy().astype(np.float64)
