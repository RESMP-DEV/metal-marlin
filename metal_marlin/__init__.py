"""Metal Marlin: FP4-quantized GEMM for Apple Silicon via Metal and PyTorch MPS.

This package provides quantized GEMM operations optimized for Apple Silicon.
Uses PyTorch MPS for tensor operations and direct Metal kernels for compute.

Key exports:
- MarlinLinear: Quantized linear layer
- MRGPTQQuantizer: Hessian-aware GPTQ quantization with Hadamard rotation
- MoE dispatch: Token-to-expert grouping with PyTorch MPS routing
- HAS_TORCH: Feature flag for runtime detection
"""

from ._compat import HAS_MPS, HAS_PYOBJC_METAL, HAS_TORCH

# HAS_MLX is deprecated - kept for backwards compatibility
HAS_MLX = False

# Always-available imports
# Attention kernels
from .attention import (
    MarlinAttention,
    RoPE,
    create_causal_mask,
    create_sliding_window_mask,
    flash_attention_metal,
    scaled_dot_product_attention_metal,
    sliding_window_attention_metal,
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
from .kv_cache_torch import CacheConfigTorch, KVCacheTorch
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

__all__ = [
    # Feature flags
    "HAS_MLX",  # Deprecated, always False
    "HAS_MPS",
    "HAS_PYOBJC_METAL",
    "HAS_TORCH",
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
    "MarlinLinear",
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
]
