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
from .gptq import GPTQQuantizer
from .gptq_fp4 import FP4GPTQQuantizer
from .hadamard import (
    HadamardMetadata,
    apply_hadamard_rotation,
    hadamard_matrix,
    inverse_hadamard_rotation,
)
from .layers import MarlinLinear
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

# MLX-dependent imports - only available when MLX is installed
if HAS_MLX:
    from .autotune import (
        autotune_gemm,
        autotuned_linear,
        load_autotune_cache,
        save_autotune_cache,
        sweep_problem_sizes,
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
    "MemoryBudget",
    "MoEDispatchInfo",
    "NGramDraft",
    "ONNXGraphInfo",
    "ONNXOp",
    "SequentialCheckpoint",
    "SmallModelDraft",
    "TileCoordinator",
    "TileKey",
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
]
