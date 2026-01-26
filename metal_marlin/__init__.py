"""Metal Marlin: FP4-quantized GEMM for Apple Silicon via MLX custom kernels.

This package provides quantized GEMM operations optimized for Apple Silicon.
When MLX is available, it uses Metal compute kernels for acceleration.
When MLX is not available, core functionality falls back to numpy.

Key exports:
- MarlinLinear: Quantized linear layer (works with or without MLX)
- MRGPTQQuantizer: Hessian-aware GPTQ quantization with Hadamard rotation
- pack_fp4_weights, quantized_linear: Low-level quantization (requires MLX)
- HAS_MLX, HAS_TORCH: Feature flags for runtime detection
"""

from ._compat import HAS_MLX, HAS_TORCH

# Always-available imports (work without MLX)
from .calibration import (
    AdaptiveQuantizer,
    AdaptiveQuantResult,
    HessianCollector,
    HessianManager,
    LayerBudget,
    ModelBudgetAllocation,
    compute_moe_expert_sensitivity,
)
from .gptq import GPTQQuantizer
from .gptq_fp4 import FP4GPTQQuantizer
from .hadamard import (
    HadamardMetadata,
    apply_hadamard_rotation,
    hadamard_matrix,
    inverse_hadamard_rotation,
)
from .layers import MarlinLinear
from .mixed_precision import (
    LayerPrecisionSelector,
    LayerQuantConfig,
    LayerSensitivity,
    MixedPrecisionConfig,
    MoEPrecisionConfig,
    Precision,
)
from .mr_gptq import MRGPTQQuantizer, QuantizationFormat, QuantizationReport
from .onnx_graph import (
    ONNXGraphInfo,
    ONNXOp,
    detect_architecture,
    parse_onnx_graph,
    parse_onnx_graph_full,
    summarize_graph,
)
from .onnx_loader import (
    extract_onnx_weights,
    get_onnx_config,
    list_onnx_tensors,
    normalize_onnx_name,
)
from .quantize import (  # FP8 quantization  # FP8 quantization
    FP8_E4M3_MAX,
    FP8_E4M3_VALUES,
    FP8_E5M2_MAX,
    FP8_E5M2_VALUES,
    compute_fp8_e4m3_values,
    compute_fp8_e5m2_values,
    dequant_fp8_e4m3,
    dequant_fp8_e5m2,
    pack_fp8_weights,
    pack_int4_weights,
    pack_nf4_weights,
    quantize_to_fp8,
    quantize_to_int4,
    quantize_to_nf4,
    unpack_fp4_weights,
)
from .quantize import pack_fp4_weights as pack_fp4_weights_cpu  # FP4/INT4/NF4 quantization
from .vision import (
    InternVLProjector,
    LLaVAProjector,
    Qwen2VLProjector,
    VisionProjector,
    VisionProjectorConfig,
    detect_projector_type,
)

# MLX-dependent imports - only available when MLX is installed
# MLA attention requires MLX for Metal kernels
if HAS_MLX:
    from .autotune import (
        autotune_gemm,
        autotuned_linear,
        load_autotune_cache,
        save_autotune_cache,
        sweep_problem_sizes,
    )
    from .capacity import (
        CapacityAnalyzer,
        CapacityStats,
        DynamicCapacity,
        DynamicCapacityConfig,
        OverflowInfo,
        analyze_overflow_rate,
        auto_tune_capacity,
        compute_expert_capacity,
        dynamic_capacity,
    )
    from .checkpoint import (
        CheckpointConfig,
        CheckpointedModule,
        CheckpointStats,
        MemoryBudget,
        SequentialCheckpoint,
        checkpoint_activations,
        chunked_apply,
    )
    from .expert_cache import (
        ExpertCache,
        ExpertStats,
        LayerStats,
        TileCoordinator,
        TileKey,
        create_moe_cache,
    )
    from .kernels import moe_expert_gemm_fp4, moe_router_topk
    from .metal_marlin import MarlinLinear as MarlinLinearLegacy
    from .metal_marlin import pack_fp4_weights, quantized_linear
    from .mla_attention import (
        MLAAttention,
        MLAConfig,
        MLAKVCache,
        MLARoPE,
        create_mla_from_hf_config,
    )
    from .mlp import MarlinMLP
    from .moe_dispatch import (
        MoEDispatchInfo,
        compute_expert_load,
        compute_load_balancing_loss,
        gather_for_experts,
        group_tokens_by_expert,
        group_tokens_by_expert_full,
        scatter_expert_outputs,
    )
    from .quantize_model import estimate_model_size, quantize_model
    from .speculative import DraftModel, DraftOutput, NGramDraft, SmallModelDraft

__all__ = [
    # Feature flags
    "HAS_MLX",
    "HAS_TORCH",
    # Capacity management
    "CapacityAnalyzer",
    "CapacityStats",
    "DynamicCapacity",
    "DynamicCapacityConfig",
    "OverflowInfo",
    "analyze_overflow_rate",
    "auto_tune_capacity",
    "compute_expert_capacity",
    "dynamic_capacity",
    # FP8 quantization
    "FP8_E4M3_MAX",
    "FP8_E4M3_VALUES",
    "FP8_E5M2_MAX",
    "FP8_E5M2_VALUES",
    "compute_fp8_e4m3_values",
    "compute_fp8_e5m2_values",
    "dequant_fp8_e4m3",
    "dequant_fp8_e5m2",
    "pack_fp8_weights",
    "quantize_to_fp8",
    # FP4/INT4/NF4 quantization (CPU)
    "pack_fp4_weights_cpu",
    "pack_int4_weights",
    "pack_nf4_weights",
    "quantize_to_int4",
    "quantize_to_nf4",
    "unpack_fp4_weights",
    # Calibration
    "AdaptiveQuantizer",
    "AdaptiveQuantResult",
    "HessianCollector",
    "HessianManager",
    "LayerBudget",
    "ModelBudgetAllocation",
    "compute_moe_expert_sensitivity",
    # MR-GPTQ quantization
    "FP4GPTQQuantizer",
    "GPTQQuantizer",
    "HadamardMetadata",
    "MRGPTQQuantizer",
    "QuantizationFormat",
    "QuantizationReport",
    "apply_hadamard_rotation",
    "hadamard_matrix",
    "inverse_hadamard_rotation",
    # Mixed precision configuration
    "LayerPrecisionSelector",
    "LayerQuantConfig",
    "LayerSensitivity",
    "MixedPrecisionConfig",
    "MoEPrecisionConfig",
    "Precision",
    # Core modules
    "CheckpointConfig",
    "CheckpointStats",
    "CheckpointedModule",
    "DraftModel",
    "DraftOutput",
    "ExpertCache",
    "ExpertStats",
    "LayerStats",
    "MarlinLinear",
    "MarlinLinearLegacy",
    "MarlinMLP",
    "MLAAttention",
    "MLAConfig",
    "MLAKVCache",
    "MLARoPE",
    "MemoryBudget",
    "MoEDispatchInfo",
    "NGramDraft",
    "ONNXGraphInfo",
    "ONNXOp",
    "SequentialCheckpoint",
    "SmallModelDraft",
    "TileCoordinator",
    "TileKey",
    "create_mla_from_hf_config",
    "autotune_gemm",
    "autotuned_linear",
    "checkpoint_activations",
    "chunked_apply",
    "compute_expert_load",
    "compute_load_balancing_loss",
    "create_moe_cache",
    "detect_architecture",
    "estimate_model_size",
    "extract_onnx_weights",
    "gather_for_experts",
    "get_onnx_config",
    "group_tokens_by_expert",
    "group_tokens_by_expert_full",
    "list_onnx_tensors",
    "load_autotune_cache",
    "moe_expert_gemm_fp4",
    "moe_router_topk",
    "normalize_onnx_name",
    "pack_fp4_weights",
    "parse_onnx_graph",
    "parse_onnx_graph_full",
    "quantize_model",
    "quantized_linear",
    "save_autotune_cache",
    "scatter_expert_outputs",
    "summarize_graph",
    "sweep_problem_sizes",
    # Vision projectors
    "VisionProjector",
    "VisionProjectorConfig",
    "LLaVAProjector",
    "Qwen2VLProjector",
    "InternVLProjector",
    "detect_projector_type",
]
