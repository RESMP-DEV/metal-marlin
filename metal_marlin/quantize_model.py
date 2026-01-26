"""Model-level quantization API for Metal Marlin.

Provides a single-call interface to quantize all nn.Linear layers in an MLX
model to Marlin FP4 (E2M1) format. Follows the same pattern as mlx.nn.quantize
but replaces with MarlinLinear layers backed by the fused dequant-GEMM Metal
kernel.

Usage:
    import mlx.nn as nn
    from metal_marlin.python.quantize_model import quantize_model

    model = load_my_model()
    quantize_model(model, group_size=128)
    # All nn.Linear layers (except lm_head, embed_tokens, etc.) are now
    # MarlinLinear with FP4 weights and Metal kernel dispatch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import mlx.nn as nn
from mlx.utils import tree_map_with_path

from .layers import MarlinLinear


def quantize_model(
    model: nn.Module,
    quant_type: Literal["fp4"] = "fp4",
    group_size: int = 128,
    skip_layers: set[str] | None = None,
    layer_filter: Callable[[str, nn.Module], bool] | None = None,
) -> nn.Module:
    """Quantize all nn.Linear layers in a model to Marlin FP4 format.

    Traverses the model's leaf modules and replaces qualifying nn.Linear
    layers with MarlinLinear instances backed by the fused dequant-GEMM
    Metal kernel. The model is modified in-place.

    Args:
        model: MLX model to quantize.
        quant_type: Quantization format. Currently only "fp4" (E2M1) is
            supported.
        group_size: Number of elements per quantization group along the
            input dimension. Must divide the layer's in_features. Default: 128.
        skip_layers: Set of name fragments to skip. A layer is skipped if
            any fragment appears anywhere in its dotted path. Defaults to
            {"lm_head", "embed_tokens", "wte", "wpe"}.
        layer_filter: Optional predicate (path, module) -> bool. When
            provided, a layer is only quantized if this returns True.
            Applied after skip_layers filtering.

    Returns:
        The same model instance with qualifying layers replaced in-place.
    """
    if quant_type != "fp4":
        raise NotImplementedError(
            f"Only quant_type='fp4' is currently supported, got {quant_type!r}"
        )

    skip_layers = skip_layers or {"lm_head", "embed_tokens", "wte", "wpe"}

    def _should_quantize(path: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if any(skip in path for skip in skip_layers):
            return False
        if layer_filter is not None and not layer_filter(path, module):
            return False
        # Verify dimensions are compatible with packing
        _, in_features = module.weight.shape
        if in_features % 8 != 0:
            return False
        return True

    def _maybe_quantize(path: str, module: nn.Module) -> nn.Module:
        if _should_quantize(path, module):
            quantized = MarlinLinear.from_linear(module, quant_type="fp4", group_size=group_size)
            return quantized
        return module

    leaves = model.leaf_modules()
    leaves = tree_map_with_path(_maybe_quantize, leaves, is_leaf=nn.Module.is_module)
    model.update_modules(leaves)
    return model


def estimate_model_size(model: nn.Module, group_size: int = 128) -> dict[str, float]:
    """Estimate model memory footprint comparing FP16 vs FP4 quantization.

    Walks the model's parameters and estimates memory usage. For layers
    that are already MarlinLinear, counts packed weight + scales. For
    nn.Linear layers, estimates what quantized size would be.

    Args:
        model: Model to analyze (quantized or unquantized).
        group_size: Group size for estimation of unquantized layers.

    Returns:
        Dict with keys:
            fp16_bytes: Bytes for non-quantized parameters (FP16).
            quantized_bytes: Bytes for quantized weight storage.
            total_bytes: Sum of fp16_bytes + quantized_bytes.
            fp16_mb, quantized_mb, total_mb: Same in MiB.
            num_quantized_layers: Count of MarlinLinear layers.
            num_unquantized_layers: Count of remaining nn.Linear layers.
    """
    fp16_bytes = 0
    quantized_bytes = 0
    num_quantized = 0
    num_unquantized = 0

    def _visit(path: str, module: nn.Module) -> None:
        nonlocal fp16_bytes, quantized_bytes, num_quantized, num_unquantized

        if isinstance(module, MarlinLinear):
            num_quantized += 1
            # Packed weights: [K, N//8] as uint32 (4 bytes each)
            K = module.in_features
            N = module.out_features
            weight_bytes = (K * N) // 2  # 4 bits per weight = 0.5 bytes
            # Scales: [K//group_size, N] as float16
            num_groups = K // module.group_size
            scale_bytes = num_groups * N * 2
            # Bias if present
            bias_bytes = N * 2 if module.bias is not None else 0
            quantized_bytes += weight_bytes + scale_bytes + bias_bytes
        elif isinstance(module, nn.Linear):
            num_unquantized += 1
            out_features, in_features = module.weight.shape
            fp16_bytes += out_features * in_features * 2
            if hasattr(module, "bias") and module.bias is not None:
                fp16_bytes += out_features * 2

    model.apply_to_modules(_visit)

    total = fp16_bytes + quantized_bytes
    return {
        "fp16_bytes": fp16_bytes,
        "quantized_bytes": quantized_bytes,
        "total_bytes": total,
        "fp16_mb": fp16_bytes / (1024 * 1024),
        "quantized_mb": quantized_bytes / (1024 * 1024),
        "total_mb": total / (1024 * 1024),
        "num_quantized_layers": num_quantized,
        "num_unquantized_layers": num_unquantized,
    }
