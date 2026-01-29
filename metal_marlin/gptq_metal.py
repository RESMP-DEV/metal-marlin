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
from typing import TYPE_CHECKING

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
    import Metal

    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    Metal = None  # type: ignore[assignment]

_SHADER_DIR = Path(__file__).parent.parent / "src"


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
            raise RuntimeError(
                "GPTQMetal requires PyTorch MPS backend (Apple Silicon)."
            )

        self._device = device or Metal.MTLCreateSystemDefaultDevice()
        self._command_queue = self._device.newCommandQueue()

        # Load shader sources
        self._hessian_source = (_SHADER_DIR / "hessian.metal").read_text()
        self._cholesky_source = (_SHADER_DIR / "cholesky.metal").read_text()

        # Compile pipelines lazily
        self._hessian_pipeline = None
        self._cholesky_pipeline = None

    def _get_pipeline(self, source: str, function_name: str):
        """Compile and cache a compute pipeline."""
        options = Metal.MTLCompileOptions.new()
        library = self._device.newLibraryWithSource_options_error_(
            source, options, None
        )[0]
        if library is None:
            raise RuntimeError(f"Failed to compile Metal shader for {function_name}")

        function = library.newFunctionWithName_(function_name)
        if function is None:
            raise RuntimeError(f"Function {function_name} not found in shader")

        pipeline = self._device.newComputePipelineStateWithFunction_error_(
            function, None
        )[0]
        return pipeline

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

        # For now, use PyTorch MPS matmul (hessian.metal integration TBD)
        # The Metal shader provides fine-grained control for very large matrices
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
        H_reg = H + regularization * torch.eye(
            H.shape[0], device=H.device, dtype=H.dtype
        )

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
