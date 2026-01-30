"""Trellis quantization submodule.

This module provides EXL3-style Trellis quantization for inference:
- TrellisLinear: Quantized linear layer using packed indices
- TrellisForCausalLM: Full causal language model with Trellis quantization
- TrellisModelLoader: Load Trellis-quantized models from safetensors
- TrellisKVCache: KV cache optimized for Trellis models

Example:
    >>> from metal_marlin.trellis import TrellisForCausalLM
    >>> model = TrellisForCausalLM.from_pretrained("model_path", device="mps")
    >>> output = model(input_ids)
"""

from __future__ import annotations

# Attention and KV cache
from .attention import TrellisMLAConfig, TrellisMLAttention, create_mla_projections
from .config import TrellisModelConfig

# Dispatch functions
from .dispatch import (
    dequantize_trellis_weight,
    dispatch_sign_flips,
    dispatch_trellis_dequant,
    dispatch_trellis_dequant_fused,
    dispatch_trellis_dequant_packed,
)

# Generation
from .generate import GenerationConfig, TrellisGenerator
from .kv_cache import TrellisKVCache

# MLP layers
from .layer import TrellisDenseMLP
from .linear import TrellisLinear, TrellisModelWrapper
from .loader import TrellisModelLoader, TrellisWeight

# Core model classes
from .model import TrellisDecoderLayer, TrellisForCausalLM, TrellisModel, TrellisMoEMLP
from .moe import TrellisExpert, TrellisMoEConfig, TrellisMoELayer

# Packing utilities
from .packing import (
    compute_compression_ratio,
    compute_packed_size,
    pack_indices,
    pack_indices_vectorized,
    pack_trellis_indices,
    unpack_indices,
    unpack_indices_vectorized,
    unpack_trellis_indices,
)

__all__ = [
    # Models
    "TrellisForCausalLM",
    "TrellisModel",
    "TrellisMoEMLP",
    "TrellisDecoderLayer",
    "TrellisLinear",
    "TrellisModelWrapper",
    "TrellisWeight",
    "TrellisModelLoader",
    "TrellisModelConfig",
    # Attention
    "TrellisMLAConfig",
    "TrellisMLAttention",
    "create_mla_projections",
    "TrellisKVCache",
    # Layers
    "TrellisDenseMLP",
    "TrellisMoEConfig",
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
