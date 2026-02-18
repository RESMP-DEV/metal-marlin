"""Per-layer FLOPs calculation for Metal Marlin operations.

This module provides tools to calculate theoretical FLOPs (floating-point
operations) for neural network layers, with specific support for quantized
operations used in Metal Marlin.

Example:
    from metal_marlin.profile import LayerFLOPsCalculator, calculate_layer_flops

    # Calculate FLOPs for a single linear layer
    flops = calculate_layer_flops(
        layer_type="linear",
        in_features=4096,
        out_features=11008,
        batch_size=8,
        seq_len=2048,
        quantized=True
    )
    print(f"Linear layer FLOPs: {flops / 1e12:.2f} TFLOPs")

    # Profile an entire model
    calculator = LayerFLOPsCalculator()
    for name, module in model.named_modules():
        calculator.add_module(name, module, input_shape=(8, 2048, 4096))
    calculator.print_summary()
"""

from __future__ import annotations

# Re-export core FLOPs calculation utilities from utils.profile_ops
from metal_marlin.utils.profile_ops import (
    LayerFLOPs,
    LayerFLOPsCounter,
    TransformerLayerFLOPs,
    calculate_attention_flops,
    calculate_embedding_flops,
    calculate_ffn_flops,
    calculate_layernorm_flops,
    calculate_matmul_flops,
    profile_model_flops,
)

# Import profile module-specific functionality
from .calculator import (
    LayerFLOPsCalculator,
    calculate_layer_flops,
    estimate_marlin_linear_flops,
    profile_model_layers,
)

__all__ = [
    # Core dataclasses
    "LayerFLOPs",
    "LayerFLOPsCounter",
    "TransformerLayerFLOPs",
    # FLOPs calculation functions
    "calculate_attention_flops",
    "calculate_embedding_flops",
    "calculate_ffn_flops",
    "calculate_layernorm_flops",
    "calculate_matmul_flops",
    "calculate_layer_flops",
    "estimate_marlin_linear_flops",
    # High-level utilities
    "LayerFLOPsCalculator",
    "profile_model_flops",
    "profile_model_layers",
]
