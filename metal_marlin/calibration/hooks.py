"""Forward hooks for Hessian collection during calibration passes.

This module provides a unified interface for registering forward hooks on
PyTorch and MLX models to collect Hessian matrices (X^T @ X) for GPTQ
quantization. The hooks capture input activations to linear layers and
accumulate the Hessian incrementally for memory efficiency.

Supports:
- PyTorch models via register_forward_hook
- MLX models via __call__ wrapper (MLX lacks native hook support)
- GQA/MQA attention patterns (different Q/K/V dimensions)

Usage:
    from metal_marlin.calibration.hooks import CalibrationHooks, HessianCollector

    hooks = CalibrationHooks()
    hooks.register_linear_hooks(model)

    for batch in calibration_data:
        _ = model(batch)  # Forward passes accumulate Hessians

    hessians = hooks.get_hessians()
    hooks.remove_hooks()

Reference:
    GPTQ Paper: https://arxiv.org/abs/2210.17323
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .._compat import HAS_MLX, HAS_TORCH, mx, torch

if TYPE_CHECKING:
    from collections.abc import Callable

    if HAS_TORCH:
        import torch.utils.hooks


@dataclass
class HessianCollector:
    """Accumulates Hessian matrix H = X^T @ X for a single layer.

    The GPTQ algorithm requires the Hessian to estimate the importance of
    each input feature for quantization. This collector accumulates the
    Hessian incrementally using a running sum, which is memory-efficient
    compared to storing all activations.

    Memory usage: O(in_features^2) regardless of calibration set size.
    For in_features=4096: 4096^2 * 8 bytes = 128MB (float64 accumulator).

    Attributes:
        in_features: Number of input features for the layer.
        hessian: Accumulated Hessian matrix [in_features, in_features].
        n_samples: Total number of samples (tokens) accumulated.
        damping: Damping factor for numerical stability.
    """

    in_features: int
    hessian: NDArray[np.float64] = field(init=False)
    n_samples: int = field(default=0, init=False)
    damping: float = 0.01

    def __post_init__(self) -> None:
        self.hessian = np.zeros(
            (self.in_features, self.in_features), dtype=np.float64
        )

    def update(self, activations: NDArray[np.floating]) -> None:
        """Update Hessian with a batch of activations.

        Args:
            activations: Input activations [batch, seq_len, in_features] or
                        [n_samples, in_features] if already flattened.
                        Accepts numpy arrays, torch tensors, or MLX arrays.
        """
        # Convert to numpy float64 for accumulation stability
        x = _to_numpy_f64(activations)

        # Flatten to 2D: [*, in_features] -> [n_samples, in_features]
        if x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])
        elif x.ndim == 1:
            x = x.reshape(1, -1)

        n_samples, in_feat = x.shape
        if in_feat != self.in_features:
            raise ValueError(
                f"Expected in_features={self.in_features}, got {in_feat}"
            )

        # Accumulate H = X^T @ X
        self.hessian += x.T @ x
        self.n_samples += n_samples

    def get_damped_hessian(self) -> NDArray[np.float64]:
        """Return the damped Hessian H + λI.

        Damping improves numerical stability for Cholesky decomposition
        and prevents singular Hessians when some features have low variance.

        The damping factor λ = damping * mean(diag(H)).

        Returns:
            Damped Hessian [in_features, in_features] as float64.
        """
        H = self.hessian.copy()
        diag_mean = np.mean(np.diag(H))
        if diag_mean > 0:
            lambda_damp = self.damping * diag_mean
            H[np.diag_indices_from(H)] += lambda_damp
        return H

    def reset(self) -> None:
        """Clear accumulated Hessian and sample count."""
        self.hessian.fill(0.0)
        self.n_samples = 0


class CalibrationHooks:
    """Manage forward hooks for Hessian collection across model layers.

    This class provides a unified interface for instrumenting both PyTorch
    and MLX models with hooks that capture input activations to linear
    layers for Hessian computation.

    PyTorch models use the native register_forward_hook API.
    MLX models use __call__ wrapping since MLX lacks native hooks.

    Supports GQA/MQA attention patterns where Q has different dimensions
    than K/V by maintaining per-layer HessianCollectors with appropriate
    in_features dimensions.

    Example:
        >>> hooks = CalibrationHooks()
        >>> hooks.register_linear_hooks(model)
        >>> for batch in calibration_data:
        ...     _ = model(batch)
        >>> hessians = hooks.get_hessians()
        >>> hooks.remove_hooks()
    """

    def __init__(self, damping: float = 0.01):
        """Initialize calibration hooks manager.

        Args:
            damping: Damping factor for Hessian regularization (default: 0.01).
                    Higher values improve stability but may affect accuracy.
        """
        self.collectors: dict[str, HessianCollector] = {}
        self.handles: list[Any] = []  # torch RemovableHandle or (module, original_call)
        self._mlx_hooks: list[tuple[Any, Any]] = []  # (module, original_call)
        self.damping = damping

    def register_linear_hooks(
        self,
        model: Any,
        layer_filter: Callable[[str, Any], bool] | None = None,
    ) -> int:
        """Register hooks on all Linear layers in the model.

        Detects the model framework (PyTorch or MLX) and registers appropriate
        hooks. For GQA/MQA attention, each projection (Q/K/V) gets its own
        collector with the correct in_features dimension.

        Args:
            model: PyTorch nn.Module or MLX nn.Module to instrument.
            layer_filter: Optional filter function (name, module) -> bool.
                         If provided, only layers where filter returns True
                         will be instrumented. Default: all Linear layers.

        Returns:
            Number of layers instrumented.

        Raises:
            RuntimeError: If neither PyTorch nor MLX is available.
        """
        if HAS_TORCH and _is_torch_module(model):
            return self._register_torch_hooks(model, layer_filter)
        elif HAS_MLX and _is_mlx_module(model):
            return self._register_mlx_hooks(model, layer_filter)
        else:
            raise RuntimeError(
                "Model framework not detected. Install PyTorch or MLX."
            )

    def _register_torch_hooks(
        self,
        model: Any,
        layer_filter: Callable[[str, Any], bool] | None,
    ) -> int:
        """Register forward hooks on PyTorch model."""
        import torch.nn as tnn

        count = 0
        for name, module in model.named_modules():
            if not isinstance(module, tnn.Linear):
                continue
            if layer_filter is not None and not layer_filter(name, module):
                continue

            in_features = module.in_features
            self.collectors[name] = HessianCollector(
                in_features=in_features, damping=self.damping
            )

            hook = module.register_forward_hook(
                self._make_torch_hook(name)
            )
            self.handles.append(hook)
            count += 1

        return count

    def _make_torch_hook(self, name: str) -> Callable:
        """Create a forward hook closure for a named layer."""

        def hook(module: Any, input: tuple, output: Any) -> None:
            # input is a tuple, first element is the activation tensor
            if isinstance(input, tuple) and len(input) > 0:
                x = input[0]
            else:
                x = input

            collector = self.collectors.get(name)
            if collector is not None:
                x_np = _to_numpy_f64(x)
                collector.update(x_np)

        return hook

    def _register_mlx_hooks(
        self,
        model: Any,
        layer_filter: Callable[[str, Any], bool] | None,
    ) -> int:
        """Register hooks on MLX model via __call__ wrapping."""
        if mx is None:
            raise RuntimeError("MLX not available")

        import mlx.nn as mnn

        count = 0
        for name, module in _named_modules_mlx(model):
            if not isinstance(module, mnn.Linear):
                continue
            if layer_filter is not None and not layer_filter(name, module):
                continue

            # Get in_features from weight shape: [out_features, in_features]
            in_features = module.weight.shape[1]
            self.collectors[name] = HessianCollector(
                in_features=in_features, damping=self.damping
            )

            # Wrap __call__ to intercept input
            self._wrap_mlx_module(name, module)
            count += 1

        return count

    def _wrap_mlx_module(self, name: str, module: Any) -> None:
        """Wrap MLX module __call__ to collect activations."""
        original_call = module.__call__

        def wrapped_call(x: Any, *args: Any, **kwargs: Any) -> Any:
            collector = self.collectors.get(name)
            if collector is not None:
                x_np = _to_numpy_f64(x)
                collector.update(x_np)
            return original_call(x, *args, **kwargs)

        module.__call__ = wrapped_call  # type: ignore[method-assign]
        self._mlx_hooks.append((module, original_call))

    def collect_from_module(
        self,
        module_name: str,
        module: Any,
    ) -> HessianCollector:
        """Create and register a collection hook for a specific module.

        Use this method when you need fine-grained control over which
        modules are instrumented, rather than auto-detecting all Linear
        layers.

        Args:
            module_name: Name for this collector (used as key in results).
            module: The module to instrument. Must be a Linear layer.

        Returns:
            The HessianCollector for this module.

        Raises:
            ValueError: If module is not a supported Linear layer type.
        """
        if HAS_TORCH and _is_torch_linear(module):
            in_features = module.in_features
            collector = HessianCollector(in_features=in_features, damping=self.damping)
            self.collectors[module_name] = collector

            hook = module.register_forward_hook(self._make_torch_hook(module_name))
            self.handles.append(hook)
            return collector

        elif HAS_MLX and _is_mlx_linear(module):
            in_features = module.weight.shape[1]
            collector = HessianCollector(in_features=in_features, damping=self.damping)
            self.collectors[module_name] = collector

            self._wrap_mlx_module(module_name, module)
            return collector

        else:
            raise ValueError(
                f"Module {module_name} is not a supported Linear layer type"
            )

    def remove_hooks(self) -> None:
        """Clean up all registered hooks.

        For PyTorch: removes hooks via RemovableHandle.remove()
        For MLX: restores original __call__ methods
        """
        # Remove PyTorch hooks
        for handle in self.handles:
            if hasattr(handle, "remove"):
                handle.remove()
        self.handles.clear()

        # Restore MLX __call__ methods
        for module, original_call in self._mlx_hooks:
            module.__call__ = original_call  # type: ignore[method-assign]
        self._mlx_hooks.clear()

    def get_hessians(self, apply_damping: bool = True) -> dict[str, NDArray[np.float64]]:
        """Return all collected Hessians.

        Args:
            apply_damping: If True, return damped Hessians (H + λI).
                          If False, return raw accumulated Hessians.

        Returns:
            Dictionary mapping layer names to Hessian matrices.
            Each Hessian has shape [in_features, in_features].
        """
        result: dict[str, NDArray[np.float64]] = {}
        for name, collector in self.collectors.items():
            if apply_damping:
                result[name] = collector.get_damped_hessian()
            else:
                result[name] = collector.hessian.copy()
        return result

    def get_sample_counts(self) -> dict[str, int]:
        """Return sample counts for each layer.

        Returns:
            Dictionary mapping layer names to number of samples processed.
        """
        return {name: collector.n_samples for name, collector in self.collectors.items()}

    def reset(self) -> None:
        """Reset all collectors without removing hooks.

        Use this to start fresh calibration on a new dataset while keeping
        the same hooks registered.
        """
        for collector in self.collectors.values():
            collector.reset()

    @property
    def num_layers(self) -> int:
        """Number of instrumented layers."""
        return len(self.collectors)

    @property
    def layer_names(self) -> list[str]:
        """Names of instrumented layers."""
        return list(self.collectors.keys())


class GQACalibrationHooks(CalibrationHooks):
    """Calibration hooks with special handling for GQA/MQA attention.

    Grouped-Query Attention (GQA) and Multi-Query Attention (MQA) use
    different numbers of heads for Q vs K/V projections:
    - Q: [hidden_size, num_heads * head_dim]
    - K: [hidden_size, num_kv_heads * head_dim]
    - V: [hidden_size, num_kv_heads * head_dim]

    This class handles the dimension mismatch by maintaining separate
    collectors for each projection type and allows specifying which
    attention pattern is used.

    Example:
        >>> hooks = GQACalibrationHooks(
        ...     num_heads=32,
        ...     num_kv_heads=8,  # GQA with 4:1 ratio
        ...     head_dim=128,
        ... )
        >>> hooks.register_attention_hooks(model, layer_prefix="model.layers")
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        damping: float = 0.01,
    ):
        """Initialize GQA-aware calibration hooks.

        Args:
            num_heads: Number of query heads.
            num_kv_heads: Number of key/value heads (< num_heads for GQA).
            head_dim: Dimension per attention head.
            damping: Hessian damping factor.
        """
        super().__init__(damping=damping)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Compute expected dimensions
        self.q_out_features = num_heads * head_dim
        self.kv_out_features = num_kv_heads * head_dim

    def register_attention_hooks(
        self,
        model: Any,
        layer_prefix: str = "model.layers",
        q_name: str = "self_attn.q_proj",
        k_name: str = "self_attn.k_proj",
        v_name: str = "self_attn.v_proj",
        o_name: str = "self_attn.o_proj",
    ) -> int:
        """Register hooks on attention projection layers.

        Filters for attention layers matching the specified naming pattern
        and handles dimension differences between Q and K/V projections.

        Args:
            model: Model to instrument.
            layer_prefix: Prefix for transformer layers (e.g., "model.layers").
            q_name: Name pattern for Q projection within attention.
            k_name: Name pattern for K projection within attention.
            v_name: Name pattern for V projection within attention.
            o_name: Name pattern for output projection within attention.

        Returns:
            Number of attention projections instrumented.
        """
        attention_patterns = {q_name, k_name, v_name, o_name}

        def attention_filter(name: str, module: Any) -> bool:
            if not name.startswith(layer_prefix):
                return False
            for pattern in attention_patterns:
                if pattern in name:
                    return True
            return False

        return self.register_linear_hooks(model, layer_filter=attention_filter)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _to_numpy_f64(arr: Any) -> NDArray[np.float64]:
    """Convert any array type to numpy float64.

    Handles PyTorch tensors, MLX arrays, and numpy arrays.
    """
    # PyTorch tensor
    if HAS_TORCH and torch is not None:
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy().astype(np.float64)

    # MLX array
    if HAS_MLX and mx is not None:
        if hasattr(arr, "__module__") and "mlx" in str(type(arr).__module__):
            mx.eval(arr)
            return np.array(arr).astype(np.float64)

    # Already numpy or array-like
    return np.asarray(arr, dtype=np.float64)


def _is_torch_module(obj: Any) -> bool:
    """Check if object is a PyTorch nn.Module."""
    if not HAS_TORCH or torch is None:
        return False
    import torch.nn as tnn
    return isinstance(obj, tnn.Module)


def _is_mlx_module(obj: Any) -> bool:
    """Check if object is an MLX nn.Module."""
    if not HAS_MLX or mx is None:
        return False
    import mlx.nn as mnn
    return isinstance(obj, mnn.Module)


def _is_torch_linear(obj: Any) -> bool:
    """Check if object is a PyTorch Linear layer."""
    if not HAS_TORCH or torch is None:
        return False
    import torch.nn as tnn
    return isinstance(obj, tnn.Linear)


def _is_mlx_linear(obj: Any) -> bool:
    """Check if object is an MLX Linear layer."""
    if not HAS_MLX or mx is None:
        return False
    import mlx.nn as mnn
    return isinstance(obj, mnn.Linear)


def _named_modules_mlx(
    module: Any, prefix: str = ""
) -> list[tuple[str, Any]]:
    """Recursively enumerate all submodules of an MLX module.

    MLX nn.Module exposes children via .children() which returns a dict.
    We handle nested dicts and lists for model.layers patterns.
    """
    result: list[tuple[str, Any]] = []

    if not hasattr(module, "children"):
        return result

    children = module.children()
    if not isinstance(children, dict):
        return result

    for attr_name, child in children.items():
        full_name = f"{prefix}.{attr_name}" if prefix else attr_name

        if HAS_MLX and mx is not None:
            import mlx.nn as mnn
            if isinstance(child, mnn.Module):
                result.append((full_name, child))
                result.extend(_named_modules_mlx(child, full_name))
            elif isinstance(child, list):
                for i, item in enumerate(child):
                    if isinstance(item, mnn.Module):
                        item_name = f"{full_name}.{i}"
                        result.append((item_name, item))
                        result.extend(_named_modules_mlx(item, item_name))
            elif isinstance(child, dict):
                for key, item in child.items():
                    if isinstance(item, mnn.Module):
                        item_name = f"{full_name}.{key}"
                        result.append((item_name, item))
                        result.extend(_named_modules_mlx(item, item_name))

    return result
