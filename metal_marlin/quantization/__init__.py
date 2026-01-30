"""Quantization utilities for Metal Marlin.

EXL3-style trellis quantization with:
- Layer-wise streaming (one layer in memory at a time)
- Parallel tile quantization
- RAM-aware calibration batching
- LDL decomposition (faster than Cholesky)
- Integrated Hadamard rotation
"""

from __future__ import annotations

from metal_marlin.quantization.calibration_streamer import CalibrationBatch, CalibrationStreamer

# Optional: exl3_pipeline requires transformers
try:
    from metal_marlin.quantization.exl3_pipeline import (
        BartowskiCalibrationV3,
        LayerInfo,
        LayerStreamer,
        QuantizationResult,
        collect_layer_hessian,
        copy_non_linear_tensors,
        quantize_model_exl3,
        save_exl3_layer,
        write_exl3_config,
    )
    from metal_marlin.quantization.exl3_pipeline import EXL3Quantizer as EXL3QuantizerPipeline

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False
    BartowskiCalibrationV3 = None  # type: ignore[assignment, misc]
    LayerInfo = None  # type: ignore[assignment, misc]
    LayerStreamer = None  # type: ignore[assignment, misc]
    QuantizationResult = None  # type: ignore[assignment, misc]
    EXL3QuantizerPipeline = None  # type: ignore[assignment, misc]
    collect_layer_hessian = None  # type: ignore[assignment]
    copy_non_linear_tensors = None  # type: ignore[assignment]
    quantize_model_exl3 = None  # type: ignore[assignment]
    save_exl3_layer = None  # type: ignore[assignment]
    write_exl3_config = None  # type: ignore[assignment]

from metal_marlin.quantization.exl3_quantizer import (
    EXL3Quantizer,
    EXL3QuantResult,
    ldlq_quantize_layer,
)
from metal_marlin.quantization.hadamard_preprocess import (
    blockwise_hadamard,
    preprocess_hessian_exl3,
    rotate_weights_exl3,
    unrotate_weights_exl3,
)

# Optional: these require safetensors
try:
    from metal_marlin.quantization.hessian_streaming import (
        StreamingHessianCollector,
        collect_all_hessians_streaming,
    )
    from metal_marlin.quantization.layer_streamer import LayerStreamer as LayerStreamerBase
    from metal_marlin.quantization.layer_streamer import LayerWeights

    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False
    StreamingHessianCollector = None  # type: ignore[assignment, misc]
    collect_all_hessians_streaming = None  # type: ignore[assignment]
    LayerStreamerBase = None  # type: ignore[assignment, misc]
    LayerWeights = None  # type: ignore[assignment, misc]

from metal_marlin.quantization.ldl_decomp import block_ldl, ldl_solve
from metal_marlin.quantization.ldlq import compute_group_scales, compute_tile_scales, pack_indices
from metal_marlin.quantization.ldlq import ldlq_quantize_layer as ldlq_quantize_layer_core
from metal_marlin.quantization.trellis_codebook import TrellisCodebook
from metal_marlin.quantization.trellis_tile import (
    TrellisTile,
    apply_tensor_core_perm,
    apply_tensor_core_perm_i,
    tensor_core_perm,
    tensor_core_perm_i,
)
from metal_marlin.quantization.viterbi_quant import (
    compute_quantization_error,
    quantize_tile_greedy,
    quantize_tile_viterbi,
    quantize_tiles_parallel,
)

__all__ = [
    # Feature flags
    "_HAS_TRANSFORMERS",
    "_HAS_SAFETENSORS",
    # Calibration
    "BartowskiCalibrationV3",
    "CalibrationBatch",
    "CalibrationStreamer",
    "StreamingHessianCollector",
    "collect_all_hessians_streaming",
    # Quantizers
    "EXL3QuantResult",
    "EXL3Quantizer",
    "EXL3QuantizerPipeline",
    "ldlq_quantize_layer",
    "ldlq_quantize_layer_core",
    # Pipeline
    "LayerInfo",
    "LayerStreamer",
    "LayerStreamerBase",
    "LayerWeights",
    "QuantizationResult",
    "collect_layer_hessian",
    "copy_non_linear_tensors",
    "quantize_model_exl3",
    "save_exl3_layer",
    "write_exl3_config",
    # Hadamard
    "blockwise_hadamard",
    "preprocess_hessian_exl3",
    "rotate_weights_exl3",
    "unrotate_weights_exl3",
    # LDL
    "block_ldl",
    "ldl_solve",
    # LDLQ
    "compute_group_scales",
    "compute_tile_scales",
    "pack_indices",
    # Trellis
    "TrellisCodebook",
    "TrellisTile",
    "apply_tensor_core_perm",
    "apply_tensor_core_perm_i",
    "tensor_core_perm",
    "tensor_core_perm_i",
    # Viterbi
    "compute_quantization_error",
    "quantize_tile_greedy",
    "quantize_tile_viterbi",
    "quantize_tiles_parallel",
]
