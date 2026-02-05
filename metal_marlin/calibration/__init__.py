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
