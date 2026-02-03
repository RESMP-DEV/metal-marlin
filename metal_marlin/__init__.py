"""Metal Marlin: FP4-quantized GEMM for Apple Silicon via Metal and PyTorch MPS.

This package provides quantized GEMM operations optimized for Apple Silicon.
Uses PyTorch MPS for tensor operations and direct Metal kernels for compute.

Key exports:
- MarlinLinear: Quantized linear layer
- MRGPTQQuantizer: Hessian-aware GPTQ quantization with Hadamard rotation
- MoE dispatch: Token-to-expert grouping with PyTorch MPS routing
- HAS_TORCH: Feature flag for runtime detection
"""

from typing import TYPE_CHECKING, Any

# Always-available imports
from ._compat import HAS_MPS, HAS_PYOBJC_METAL, HAS_TORCH

if TYPE_CHECKING:
    import torch

# Attention kernels
try:
    from .attention import (
        MarlinAttention,
        RoPE,
        create_causal_mask,
        create_sliding_window_mask,
        flash_attention_metal,
        scaled_dot_product_attention_metal,
        sliding_window_attention_metal,
    )
except ImportError:
    MarlinAttention = None
    RoPE = None
    create_causal_mask = None
    create_sliding_window_mask = None
    flash_attention_metal = None
    scaled_dot_product_attention_metal = None
    sliding_window_attention_metal = None
from .awq import (
    AWQResult,
    awq_dequantize,
    awq_quantize,
    awq_quantize_model,
    compute_activation_stats,
    compute_salient_scaling,
    find_salient_weights,
)
from .calibration import (
    AdaptiveQuantizer,
    AdaptiveQuantResult,
    HessianCollector,
    HessianManager,
    LayerBudget,
    ModelBudgetAllocation,
    compute_moe_expert_sensitivity,
)

# Expert gather/scatter operations for MoE routing
from .expert_ops import expert_gather, expert_scatter_add

# Metal-accelerated generation (requires PyTorch MPS + PyObjC Metal)
from .generate import GenerationConfig, generate, generate_batch, generate_stream
from .gptq import GPTQQuantizer
from .gptq_fp4 import FP4GPTQQuantizer
from .hadamard import (
    HadamardMetadata,
    apply_hadamard_rotation,
    hadamard_matrix,
    inverse_hadamard_rotation,
)
from .heap_allocator import (
    HeapAllocation,
    HeapAllocatorMetrics,
    HeapBufferPool,
    MetalHeapAllocator,
)
from .kernels import marlin_gemm_fp4, marlin_gemm_int4
from .kv_cache_torch import CacheConfigTorch, KVCacheTorch
from .layer_replacement import (
    MetalQuantizedMoE,
    find_linear_layers,
    find_moe_layers,
    get_parent_module,
    quantize_linear_layer,
    quantize_moe_experts,
    replace_linear_layers,
    replace_moe_layers,
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

# MoE dispatch now uses PyTorch MPS (no MLX dependency)
from .moe_dispatch import (
    MoEDispatchInfo,
    compute_expert_load,
    compute_load_balancing_loss,
    gather_for_experts,
    group_tokens_by_expert,
    group_tokens_by_expert_full,
    scatter_expert_outputs,
)
from .moe_ops import fused_moe_forward
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

# Pipelined prefill/decode for improved throughput
from .pipeline import (
    AsyncPipelineScheduler,
    InferenceRequest,
    PipelineScheduler,
    RequestState,
    generate_pipelined,
)
from .quantize import (  # FP8 quantization
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
from .rope import (
    YaRNConfig,
    YaRNRoPE,
    compute_yarn_cos_sin_cache,
    compute_yarn_inv_freq,
    create_rope_from_config,
    get_yarn_mscale,
)
from .sampler import MetalSampler, SamplingConfig, sample_next_token
from .vision import (
    InternVLProjector,
    LLaVAProjector,
    Qwen2VLProjector,
    VisionProjector,
    VisionProjectorConfig,
    detect_projector_type,
)

# Fast inference path (optional - requires C++ extension)
try:
    from .fast_inference import FastInferenceContext, fast_dispatch_available, get_fast_context
except ImportError:
    FastInferenceContext = None
    fast_dispatch_available = False  # type: ignore[assignment]
    get_fast_context = None  # type: ignore[assignment]

# Async transfer / compute overlap (optional - requires PyObjC Metal)
try:
    from .metal_dispatch import (
        AsyncTransferHandle,
        AsyncTransferManager,
        PipelinedLayerDispatcher,
    )
except ImportError:
    AsyncTransferHandle = None
    AsyncTransferManager = None
    PipelinedLayerDispatcher = None

# Metal GPU dispatchers (optional - require Apple Silicon + PyObjC)
try:
    from .fp4_metal import FP4Metal, dequantize_fp4_metal, quantize_fp4_metal
except ImportError:
    FP4Metal = None
    quantize_fp4_metal = None  # type: ignore[assignment]
    dequantize_fp4_metal = None  # type: ignore[assignment]

try:
    from .gptq_metal import GPTQMetal, compute_hessian_metal
except ImportError:
    GPTQMetal = None
    compute_hessian_metal = None  # type: ignore[assignment]

try:
    from .hadamard_metal import HadamardMetal, hadamard_transform_metal
except ImportError:
    HadamardMetal = None
    hadamard_transform_metal = None  # type: ignore[assignment]

# HAS_MLX is deprecated - kept for backwards compatibility
HAS_MLX = False


# Preload metallib at import time for faster first kernel dispatch
def _preload_metallib() -> None:
    try:
        from .metallib_loader import get_precompiled_library

        lib = get_precompiled_library()
        if lib is not None:
            import logging

            logging.getLogger(__name__).debug("Preloaded metallib")
    except Exception:
        pass  # Silently fall back to JIT


# Uncomment to enable preloading:
# _preload_metallib()

__all__ = [
    # Feature flags
    "HAS_MLX",  # Deprecated, always False
    "HAS_MPS",
    "HAS_PYOBJC_METAL",
    "HAS_TORCH",
    # AWQ quantization
    "AWQResult",
    "awq_dequantize",
    "awq_quantize",
    "awq_quantize_model",
    "compute_activation_stats",
    "compute_salient_scaling",
    "find_salient_weights",
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
    "marlin_gemm_fp4",
    "marlin_gemm_int4",
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
    "HeapAllocation",
    "HeapAllocatorMetrics",
    "HeapBufferPool",
    "MRGPTQQuantizer",
    "MetalHeapAllocator",
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
    "MarlinLinear",
    "find_linear_layers",
    "find_moe_layers",
    "get_parent_module",
    "quantize_linear_layer",
    "quantize_moe_experts",
    "replace_linear_layers",
    "replace_moe_layers",
    "MetalQuantizedMoE",
    "MoEDispatchInfo",
    "ONNXGraphInfo",
    "ONNXOp",
    "compute_expert_load",
    "compute_load_balancing_loss",
    "detect_architecture",
    "extract_onnx_weights",
    "gather_for_experts",
    "get_onnx_config",
    "group_tokens_by_expert",
    "group_tokens_by_expert_full",
    "list_onnx_tensors",
    "normalize_onnx_name",
    "parse_onnx_graph",
    "parse_onnx_graph_full",
    "scatter_expert_outputs",
    "fused_moe_forward",
    "expert_gather",
    "expert_scatter_add",
    "summarize_graph",
    # Vision projectors
    "VisionProjector",
    "VisionProjectorConfig",
    "LLaVAProjector",
    "Qwen2VLProjector",
    "InternVLProjector",
    "detect_projector_type",
    # Metal-accelerated generation
    "GenerationConfig",
    "generate",
    "generate_batch",
    "generate_stream",
    # Pipelined prefill/decode
    "AsyncPipelineScheduler",
    "InferenceRequest",
    "PipelineScheduler",
    "RequestState",
    "generate_pipelined",
    "CacheConfigTorch",
    "KVCacheTorch",
    "MetalSampler",
    "SamplingConfig",
    "sample_next_token",
    # YaRN RoPE
    "YaRNConfig",
    "YaRNRoPE",
    "compute_yarn_cos_sin_cache",
    "compute_yarn_inv_freq",
    "create_rope_from_config",
    "get_yarn_mscale",
    # Attention
    "MarlinAttention",
    "RoPE",
    "create_causal_mask",
    "create_sliding_window_mask",
    "flash_attention_metal",
    "scaled_dot_product_attention_metal",
    "sliding_window_attention_metal",
    # Metal GPU dispatchers
    "FP4Metal",
    "GPTQMetal",
    "HadamardMetal",
    "compute_hessian_metal",
    "dequantize_fp4_metal",
    "hadamard_transform_metal",
    "quantize_fp4_metal",
    # Fast inference path
    "FastInferenceContext",
    "fast_dispatch_available",
    "get_fast_context",
    # Async transfer / compute overlap
    "AsyncTransferHandle",
    "AsyncTransferManager",
    "PipelinedLayerDispatcher",
]
