"""Hessian collection for GPTQ quantization.

This module provides pure-numpy implementations for collecting the Hessian
approximation H = X^T @ X from layer activations during calibration passes.
The Hessian captures second-order information about input distributions that
GPTQ uses for optimal quantization error compensation.

Memory Efficiency:
    Instead of storing all activations, we accumulate X^T @ X in a streaming
    fashion. For a layer with in_features=4096, this requires only 4096^2 * 8
    = 128MB (float64) regardless of calibration dataset size.

Numerical Precision:
    Uses float64 for accumulation to prevent precision loss when summing
    many outer products. The final Hessian can be cast to float32 after
    damping is applied.

Usage:
    # Single layer
    collector = HessianCollector(in_features=4096)
    for batch in calibration_data:
        x = get_layer_input(batch)  # [batch, seq, 4096]
        collector.accumulate(x)
    H = collector.get_hessian(damp=0.01)

    # Multi-layer with hooks
    manager = HessianManager()
    manager.register_layer("q_proj", in_features=4096)
    manager.register_layer("k_proj", in_features=4096)
    # ... run forward passes with hooks calling manager.accumulate()
    hessians = manager.get_all_hessians(damp=0.01)

Reference:
    GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
    https://arxiv.org/abs/2210.17323
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:

    from numpy.typing import NDArray


@dataclass
class HessianCollector:
    """Collect Hessian approximation H = X^T X from activations.

    Provides memory-efficient streaming accumulation of the Hessian matrix
    H = X^T @ X where X contains all calibration activations [samples, features].

    Instead of storing all activations, this class accumulates the outer
    product X^T @ X incrementally, requiring only O(in_features^2) memory
    regardless of calibration dataset size.

    Attributes:
        in_features: Dimension of input features.
        H: Running sum of X^T @ X [in_features, in_features].
        n_samples: Total number of samples (tokens) accumulated.
        dtype: Numpy dtype for accumulation (default: float64 for precision).

    Example:
        >>> collector = HessianCollector(in_features=4096)
        >>> for batch in calibration_data:
        ...     x = model.layer_input(batch)  # [batch, seq_len, 4096]
        ...     collector.accumulate(x)
        >>> H = collector.get_hessian(damp=0.01)
        >>> print(f"Collected from {collector.n_samples} tokens")
    """

    in_features: int
    H: NDArray[np.float64] = field(init=False)
    n_samples: int = field(default=0, init=False)
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float64))

    def __post_init__(self) -> None:
        """Initialize the Hessian accumulator."""
        self.H = np.zeros((self.in_features, self.in_features), dtype=self.dtype)

    def accumulate(self, x: NDArray[np.floating]) -> None:
        """Add batch of activations to Hessian accumulator.

        The input x can have shape:
        - [batch, seq_len, in_features] - 3D transformer activations
        - [batch, in_features] - 2D activations
        - [n_samples, in_features] - pre-flattened 2D

        All shapes are flattened to [n_samples, in_features] before
        computing and adding the contribution X^T @ X to H.

        Uses float64 accumulation for numerical stability when summing
        many outer products.

        Args:
            x: Input activations with last dimension = in_features.

        Raises:
            ValueError: If x.shape[-1] != in_features.
        """
        x = np.asarray(x)

        # Validate feature dimension
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, "
                f"got shape {x.shape}"
            )

        # Flatten to 2D: [n_samples, in_features]
        x_flat = x.reshape(-1, self.in_features).astype(self.dtype)
        n_samples = x_flat.shape[0]

        # Accumulate X^T @ X: [in_features, in_features]
        # This is the key memory-efficient step: we don't store x_flat,
        # only the running sum of outer products.
        self.H += x_flat.T @ x_flat
        self.n_samples += n_samples

    def get_hessian(self, damp: float = 0.01) -> NDArray[np.float32]:
        """Return damped Hessian for GPTQ quantization.

        The Hessian is normalized by sample count and regularized:
            H_damped = H / n_samples + damp * mean(diag(H)) * I

        This damping ensures:
        1. Positive definiteness for stable Cholesky decomposition
        2. Numerical stability when Hessian is near-singular
        3. Regularization to handle rank-deficient activations

        Args:
            damp: Damping factor as fraction of mean diagonal.
                Default 0.01 (1% of mean diagonal added to diagonal).
                Use higher values (0.05-0.1) if Cholesky fails.

        Returns:
            Damped Hessian as float32 [in_features, in_features].

        Raises:
            ValueError: If no samples have been accumulated.
        """
        if self.n_samples == 0:
            raise ValueError(
                "No samples accumulated. Call accumulate() with calibration data first."
            )

        # Normalize by sample count
        H = self.H / self.n_samples

        # Compute damping: λ = damp * mean(diag(H))
        diag_mean = np.mean(np.diag(H))
        lambda_damp = damp * diag_mean

        # Add damping: H_damped = H + λI
        H += lambda_damp * np.eye(self.in_features, dtype=self.dtype)

        return H.astype(np.float32)

    def get_raw_hessian(self) -> NDArray[np.float64]:
        """Return unnormalized, undamped Hessian (H = sum of X^T @ X).

        Useful for manual normalization or combining Hessians from
        multiple collectors (e.g., distributed calibration).

        Returns:
            Raw Hessian accumulator as float64.
        """
        return self.H.copy()

    def merge(self, other: HessianCollector) -> None:
        """Merge another HessianCollector's accumulation into this one.

        Useful for distributed calibration where different workers
        collect Hessians on different data subsets.

        Args:
            other: Another HessianCollector with same in_features.

        Raises:
            ValueError: If in_features don't match.
        """
        if other.in_features != self.in_features:
            raise ValueError(
                f"Cannot merge collectors with different in_features: "
                f"{self.in_features} vs {other.in_features}"
            )

        self.H += other.H
        self.n_samples += other.n_samples

    def reset(self) -> None:
        """Clear accumulated Hessian and sample count."""
        self.H.fill(0)
        self.n_samples = 0

    @property
    def memory_bytes(self) -> int:
        """Memory used by Hessian accumulator in bytes."""
        return self.H.nbytes

    @property
    def memory_mb(self) -> float:
        """Memory used by Hessian accumulator in megabytes."""
        return self.memory_bytes / (1024 * 1024)


class HessianManager:
    """Coordinate Hessian collection across multiple model layers.

    Manages HessianCollector instances for each layer and provides
    methods for registering layers, accumulating activations, and
    retrieving final Hessians.

    Designed to work with model instrumentation (hooks) that call
    accumulate_layer() during forward passes.

    Example with hooks:
        manager = HessianManager()

        # Register layers to track
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                manager.register_layer(name, module.in_features)

        # Install hooks (framework-specific)
        def make_hook(layer_name):
            def hook(module, input, output):
                x = input[0].detach().cpu().numpy()
                manager.accumulate_layer(layer_name, x)
            return hook

        for name, module in model.named_modules():
            if name in manager.layer_names:
                module.register_forward_hook(make_hook(name))

        # Run calibration
        for batch in calibration_data:
            _ = model(batch)

        # Get Hessians
        hessians = manager.get_all_hessians(damp=0.01)

    Attributes:
        collectors: Dict mapping layer names to HessianCollector instances.
        default_damp: Default damping factor for get_hessian().
    """

    def __init__(self, default_damp: float = 0.01) -> None:
        """Initialize HessianManager.

        Args:
            default_damp: Default damping factor for Hessian retrieval.
        """
        self.collectors: dict[str, HessianCollector] = {}
        self.default_damp = default_damp
        self._hooks: list[tuple[Any, Any]] = []  # (module, hook_handle)

    def register_layer(self, name: str, in_features: int) -> None:
        """Register a layer for Hessian collection.

        Args:
            name: Unique layer identifier (e.g., "model.layers.0.q_proj").
            in_features: Input dimension of the layer.

        Raises:
            ValueError: If layer with this name already registered.
        """
        if name in self.collectors:
            raise ValueError(
                f"Layer {name!r} already registered. "
                f"Call reset_layer() or unregister_layer() first."
            )

        self.collectors[name] = HessianCollector(in_features=in_features)

    def unregister_layer(self, name: str) -> None:
        """Remove a layer from tracking.

        Args:
            name: Layer name to remove.
        """
        self.collectors.pop(name, None)

    def accumulate_layer(self, name: str, x: NDArray[np.floating]) -> None:
        """Add activations to a specific layer's Hessian.

        Args:
            name: Layer name (must be registered).
            x: Input activations for this layer.

        Raises:
            KeyError: If layer not registered.
        """
        if name not in self.collectors:
            raise KeyError(
                f"Layer {name!r} not registered. "
                f"Call register_layer() first. "
                f"Registered layers: {list(self.collectors.keys())}"
            )

        self.collectors[name].accumulate(x)

    def get_hessian(
        self, name: str, damp: float | None = None
    ) -> NDArray[np.float32]:
        """Get damped Hessian for a specific layer.

        Args:
            name: Layer name.
            damp: Damping factor (uses default_damp if None).

        Returns:
            Damped Hessian as float32.

        Raises:
            KeyError: If layer not registered.
        """
        if name not in self.collectors:
            raise KeyError(f"Layer {name!r} not registered.")

        damp = damp if damp is not None else self.default_damp
        return self.collectors[name].get_hessian(damp=damp)

    def get_all_hessians(
        self, damp: float | None = None
    ) -> dict[str, NDArray[np.float32]]:
        """Get damped Hessians for all registered layers.

        Args:
            damp: Damping factor (uses default_damp if None).

        Returns:
            Dict mapping layer names to damped Hessians.
        """
        damp = damp if damp is not None else self.default_damp
        return {
            name: collector.get_hessian(damp=damp)
            for name, collector in self.collectors.items()
        }

    def get_sample_counts(self) -> dict[str, int]:
        """Get sample counts for all layers.

        Returns:
            Dict mapping layer names to sample counts.
        """
        return {name: c.n_samples for name, c in self.collectors.items()}

    def reset_layer(self, name: str) -> None:
        """Reset a specific layer's Hessian accumulator.

        Args:
            name: Layer name to reset.
        """
        if name in self.collectors:
            self.collectors[name].reset()

    def reset_all(self) -> None:
        """Reset all Hessian accumulators."""
        for collector in self.collectors.values():
            collector.reset()

    def remove_hooks(self) -> None:
        """Remove any registered hooks and clear hook list.

        Call this after calibration is complete to restore original
        model behavior.
        """
        for module, hook_handle in self._hooks:
            try:
                hook_handle.remove()
            except Exception:
                pass
        self._hooks.clear()

    @property
    def layer_names(self) -> list[str]:
        """List of registered layer names."""
        return list(self.collectors.keys())

    @property
    def num_layers(self) -> int:
        """Number of registered layers."""
        return len(self.collectors)

    @property
    def total_memory_mb(self) -> float:
        """Total memory used by all Hessian accumulators in MB."""
        return sum(c.memory_mb for c in self.collectors.values())

    def summary(self) -> str:
        """Return summary string of collection state."""
        lines = [
            f"HessianManager: {self.num_layers} layers, {self.total_memory_mb:.1f} MB",
            "-" * 60,
        ]
        for name, collector in self.collectors.items():
            lines.append(
                f"  {name}: {collector.in_features}x{collector.in_features}, "
                f"{collector.n_samples} samples, {collector.memory_mb:.1f} MB"
            )
        return "\n".join(lines)


# =============================================================================
# Utility functions
# =============================================================================


def compute_hessian_from_activations(
    activations: list[NDArray[np.floating]] | NDArray[np.floating],
    damp: float = 0.01,
) -> NDArray[np.float32]:
    """Convenience function to compute damped Hessian from activations.

    Args:
        activations: Single array [n_samples, features] or list of arrays
            that will be concatenated. Each array can be 2D or 3D.
        damp: Damping factor.

    Returns:
        Damped Hessian as float32.
    """
    # Handle single array vs list
    if isinstance(activations, np.ndarray):
        activations = [activations]

    # Determine in_features from first activation
    first = np.asarray(activations[0])
    in_features = first.shape[-1]

    # Create collector and accumulate
    collector = HessianCollector(in_features=in_features)
    for x in activations:
        collector.accumulate(x)

    return collector.get_hessian(damp=damp)


def validate_hessian(H: NDArray[np.floating], rtol: float = 1e-5) -> dict[str, Any]:
    """Validate Hessian matrix properties for GPTQ.

    Checks:
    1. Symmetry: H == H^T
    2. Positive semi-definiteness: all eigenvalues >= 0
    3. Finite values: no NaN or Inf
    4. Conditioning: condition number not too extreme

    Args:
        H: Hessian matrix to validate.
        rtol: Relative tolerance for symmetry check.

    Returns:
        Dict with validation results and diagnostics.
    """
    H = np.asarray(H)
    n = H.shape[0]

    results: dict[str, Any] = {
        "shape": H.shape,
        "dtype": str(H.dtype),
    }

    # Check symmetry
    sym_error = np.max(np.abs(H - H.T))
    results["is_symmetric"] = bool(sym_error < rtol * np.max(np.abs(H)))
    results["symmetry_error"] = float(sym_error)

    # Check for NaN/Inf
    results["has_nan"] = bool(np.any(np.isnan(H)))
    results["has_inf"] = bool(np.any(np.isinf(H)))
    results["is_finite"] = not (results["has_nan"] or results["has_inf"])

    # Check eigenvalues for positive semi-definiteness
    if results["is_finite"]:
        eigenvalues = np.linalg.eigvalsh(H)
        results["min_eigenvalue"] = float(np.min(eigenvalues))
        results["max_eigenvalue"] = float(np.max(eigenvalues))
        results["is_positive_semidefinite"] = bool(np.min(eigenvalues) >= -1e-6 * n)

        # Condition number (ratio of max to min non-zero eigenvalue)
        nonzero_eigs = eigenvalues[np.abs(eigenvalues) > 1e-10]
        if len(nonzero_eigs) > 0:
            results["condition_number"] = float(
                np.max(np.abs(nonzero_eigs)) / np.min(np.abs(nonzero_eigs))
            )
        else:
            results["condition_number"] = float("inf")
    else:
        results["is_positive_semidefinite"] = False
        results["condition_number"] = float("nan")

    # Diagonal statistics
    diag = np.diag(H)
    results["diag_mean"] = float(np.mean(diag))
    results["diag_std"] = float(np.std(diag))
    results["diag_min"] = float(np.min(diag))
    results["diag_max"] = float(np.max(diag))

    # Overall validation
    results["is_valid"] = (
        results["is_symmetric"]
        and results["is_finite"]
        and results["is_positive_semidefinite"]
    )

    return results
