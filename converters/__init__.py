"""Model format converters and executors.

This package provides format-agnostic model loading and execution:

- `onnx_executor`: Execute ONNX models using Metal Marlin kernels
- `ort_marlin_provider`: ONNX Runtime custom op provider for Metal Marlin
- `safetensors_loader`: Load HuggingFace config.json, map weight names, quantize
- `gguf_to_marlin`: Convert GGUF MXFP4 weights to Marlin format (see metal_marlin.gguf_to_marlin)

The philosophy is to support model formats (ONNX, safetensors, GGUF), not
model architectures (Llama, Mistral). This way, any standard transformer
works automatically.
"""

from .calibration import (
    CalibrationCollector,
    CalibrationDataset,
    CalibrationStats,
    compute_scales,
)
from .onnx_executor import ONNXExecutor, ONNXGraph, load_onnx_model
from .ort_marlin_provider import (
    MarlinFlashAttentionOp,
    MarlinQuantizedLinearOp,
    MarlinQuantizedMatMulOp,
    create_session,
    export_with_marlin_ops,
)
from .quantize import LayerReport, QuantizationReport, quantize_model
from .safetensors_loader import (
    detect_model_type,
    estimate_memory,
    get_supported_architectures,
    load_and_quantize,
    load_config,
    load_mapped_safetensors,
    map_weight_names,
    register_architecture,
)

__all__ = [
    "CalibrationCollector",
    "CalibrationDataset",
    "CalibrationStats",
    "LayerReport",
    "MarlinFlashAttentionOp",
    "MarlinQuantizedLinearOp",
    "MarlinQuantizedMatMulOp",
    "ONNXExecutor",
    "ONNXGraph",
    "QuantizationReport",
    "compute_scales",
    "create_session",
    "detect_model_type",
    "estimate_memory",
    "export_with_marlin_ops",
    "get_supported_architectures",
    "load_and_quantize",
    "load_config",
    "load_mapped_safetensors",
    "load_onnx_model",
    "map_weight_names",
    "quantize_model",
    "register_architecture",
]
