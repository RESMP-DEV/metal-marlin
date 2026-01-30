"""Calibration utilities for GPTQ quantization.

This subpackage provides tools for collecting activation statistics
and Hessian approximations needed for high-quality weight quantization.

Key components:
- CalibrationDataset: Multi-domain v3 calibration dataset (~800 samples)
- HessianCollector: Memory-efficient streaming Hessian accumulation (H = X^T @ X)
- HessianManager: Coordinate collection across model layers with hooks
- CalibrationHooks: Unified forward hooks for PyTorch and MLX models
- GQACalibrationHooks: Hooks with GQA/MQA attention pattern support

The Hessian H = X^T @ X captures input activation statistics that GPTQ uses
to make optimal per-column quantization decisions with error compensation.

Example (direct accumulation):
    from metal_marlin.calibration import HessianCollector

    collector = HessianCollector(in_features=4096)
    for batch in calibration_data:
        activations = model.get_layer_input(batch)  # [batch, seq, features]
        collector.accumulate(activations)
    H = collector.get_hessian(damp=0.01)

Example (with hooks for PyTorch/MLX models):
    from metal_marlin.calibration import CalibrationHooks

    hooks = CalibrationHooks()
    hooks.register_linear_hooks(model)

    for batch in calibration_data:
        _ = model(batch)  # Hooks capture inputs automatically

    hessians = hooks.get_hessians()
    hooks.remove_hooks()

Example (GQA/MQA attention):
    from metal_marlin.calibration import GQACalibrationHooks

    hooks = GQACalibrationHooks(
        num_heads=32,
        num_kv_heads=8,  # GQA with 4:1 ratio
        head_dim=128,
    )
    hooks.register_attention_hooks(model, layer_prefix="model.layers")
"""

# Backwards compatibility imports from the parent module (calibration.py)
# These must be imported carefully to avoid circular import issues.
# We use importlib to defer the import until all modules are initialized.
import importlib as _importlib


def _get_legacy_module():
    """Lazy import of parent calibration module."""
    # Import the module directly by its full path to avoid circular reference
    import sys
    from pathlib import Path

    # Get the parent calibration.py file
    _parent = Path(__file__).parent.parent / "calibration.py"
    if _parent.exists():
        import importlib.util

        spec = _importlib.util.spec_from_file_location(
            "metal_marlin._calibration_legacy", str(_parent)
        )
        if spec and spec.loader:
            module = _importlib.util.module_from_spec(spec)
            sys.modules["metal_marlin._calibration_legacy"] = module
            spec.loader.exec_module(module)
            return module
    return None


# Re-export from parent module for backwards compatibility
# These are lazily imported to avoid circular issues
def __getattr__(name):
    """Lazy attribute access for backwards compatibility."""
    _legacy_names = {
        "BartowskiCalibrationLegacy",
        "CalibrationDataset",
        "compute_activation_ranges",
        "load_ranges",
        "ranges_to_scales",
        "save_ranges",
    }
    if name in _legacy_names:
        legacy = _get_legacy_module()
        if legacy:
            actual_name = name.replace("Legacy", "") if name.endswith("Legacy") else name
            return getattr(legacy, actual_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from .adaptive_quant import (
    AdaptiveQuantizer,
    AdaptiveQuantResult,
    LayerBudget,
    ModelBudgetAllocation,
    compute_moe_expert_sensitivity,
)
from .calibration_dataset import BartowskiCalibration, CalibrationDataset
from .hessian_collector import HessianCollector, HessianManager
from .hooks import CalibrationHooks, GQACalibrationHooks
from .sensitivity import (
    LayerSensitivity,
    analyze_layer_sensitivity,
    compute_model_sensitivity_profile,
    load_sensitivity_profile,
    recommend_quantization,
    save_sensitivity_profile,
    sensitivity_to_config,
)

__all__ = [
    # Adaptive quantization
    "AdaptiveQuantizer",
    "AdaptiveQuantResult",
    "LayerBudget",
    "ModelBudgetAllocation",
    "compute_moe_expert_sensitivity",
    # Calibration dataset
    "CalibrationDataset",
    "BartowskiCalibration",  # Backwards compatibility alias
    # Hooks
    "CalibrationHooks",
    "GQACalibrationHooks",
    "HessianCollector",
    "HessianManager",
    "LayerSensitivity",
    "analyze_layer_sensitivity",
    "compute_model_sensitivity_profile",
    "load_sensitivity_profile",
    "recommend_quantization",
    "save_sensitivity_profile",
    "sensitivity_to_config",
]
