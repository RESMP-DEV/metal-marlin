"""Trellis quantization submodule.

This module provides EXL3-style Trellis quantization for inference:
- TrellisLinear: Quantized linear layer using packed indices
- TrellisForCausalLM: Full causal language model with Trellis quantization
- TrellisModelLoader: Load Trellis-quantized models from safetensors
- TrellisKVCache: KV cache optimized for Trellis models
- CompressedKVCache: Extended KV cache with int8 quantization and sliding window

Example:
    >>> from metal_marlin.trellis import TrellisForCausalLM
    >>> model = TrellisForCausalLM.from_pretrained("model_path", device="mps")
    >>> output = model(input_ids)
"""

from __future__ import annotations

from ..kv_cache import CompressedKVCache, TrellisKVCache
# Attention and KV cache
from .attention import (TrellisMLAConfig, TrellisMLAttention,
                        create_mla_projections)
from .config import GLM4_TOKENIZER_ID, TrellisModelConfig
# Dispatch functions
from .dispatch import (dequantize_trellis_weight, dispatch_gemm_trellis_auto,
                       dispatch_gemm_trellis_decode_auto, dispatch_sign_flips,
                       dispatch_trellis_dequant,
                       dispatch_trellis_dequant_fused,
                       dispatch_trellis_dequant_packed)

# Mixed bit-width MoE dispatch
try:
    from .mixed_bpw_dispatch import (MixedBPWMoEDispatcher, MoEConfig,
                                     dispatch_mixed_bpw_moe,
                                     dispatch_mixed_bpw_moe_with_cpp_fallback,
                                     get_mixed_bpw_stats,
                                     reset_mixed_bpw_stats)
except ImportError:
    pass

# Auto-tuning for mixed bit-width kernels
try:
    from .autotune_mixed_bpw import (AutotuneConfig, BenchmarkResult,
                                     KernelConfig, MixedBPWAutoTuner)
except ImportError:
    pass

# Generation
from .generate import GenerationConfig, TrellisGenerator
# MLP layers
from .layer import TrellisDenseMLP
from .linear import TrellisLinear, TrellisModelWrapper
# Core model classes.
# NOTE: These names are part of the long-standing public API used by call sites
# importing from `metal_marlin.trellis`, `metal_marlin.trellis.model`, and
# `metal_marlin.trellis.lm`.
from .loader import TrellisModelLoader, TrellisWeight
from .model import CausalLMOutput as _CausalLMOutput
from .model import TrellisDecoderLayer as _TrellisDecoderLayer
from .model import TrellisForCausalLM as _TrellisForCausalLM
from .model import TrellisModel as _TrellisModel
from .model import TrellisMoEMLP as _TrellisMoEMLP
from .moe import TrellisExpert, TrellisMoELayer
# Packing utilities
from .packing import (compute_compression_ratio, compute_packed_size,
                      pack_indices, pack_indices_vectorized,
                      pack_trellis_indices, unpack_indices,
                      unpack_indices_vectorized, unpack_trellis_indices)

# Public compatibility aliases.
CausalLMOutput = _CausalLMOutput
TrellisForCausalLM = _TrellisForCausalLM
TrellisModel = _TrellisModel
TrellisMoEMLP = _TrellisMoEMLP
TrellisDecoderLayer = _TrellisDecoderLayer

__all__ = [
    # Models
    "TrellisForCausalLM",
    "TrellisModel",
    "TrellisMoEMLP",
    "TrellisDecoderLayer",
    "CausalLMOutput",
    "TrellisLinear",
    "TrellisModelWrapper",
    "TrellisWeight",
    "TrellisModelLoader",
    "TrellisModelConfig",
    "GLM4_TOKENIZER_ID",
    # Attention
    "TrellisMLAConfig",
    "TrellisMLAttention",
    "create_mla_projections",
    "TrellisKVCache",
    "CompressedKVCache",
    # Layers
    "TrellisDenseMLP",
    "TrellisMoELayer",
    "TrellisExpert",
    # Generation
    "TrellisGenerator",
    "GenerationConfig",
    # Dispatch
    "dispatch_trellis_dequant",
    "dispatch_trellis_dequant_packed",
    "dispatch_trellis_dequant_fused",
    "dispatch_sign_flips",
    "dequantize_trellis_weight",
    "dispatch_gemm_trellis_auto",
    "dispatch_gemm_trellis_decode_auto",
    # Mixed bit-width MoE
    "MixedBPWMoEDispatcher",
    "MoEConfig",
    "dispatch_mixed_bpw_moe",
    "dispatch_mixed_bpw_moe_with_cpp_fallback",
    "get_mixed_bpw_stats",
    "reset_mixed_bpw_stats",
    # Auto-tuning
    "MixedBPWAutoTuner",
    "KernelConfig",
    "BenchmarkResult",
    "AutotuneConfig",
    # Packing
    "pack_indices",
    "unpack_indices",
    "pack_trellis_indices",
    "unpack_trellis_indices",
    "compute_packed_size",
    "compute_compression_ratio",
    "pack_indices_vectorized",
    "unpack_indices_vectorized",
]
