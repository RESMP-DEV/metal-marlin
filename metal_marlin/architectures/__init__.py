"""Architecture-specific implementations for Metal Marlin.

This module provides specialized implementations for various transformer
architectures that require non-standard attention mechanisms or layer types.

Supported architectures:

Differential Transformer (Microsoft, 2024):
    Noise-reducing differential attention that computes:
    Output = softmax(Q1@K1^T/sqrt(d))@V - lambda * softmax(Q2@K2^T/sqrt(d))@V

    - Two attention heads per "logical" head
    - Learnable lambda parameter for noise cancellation
    - Improved long-context and retrieval performance

Hybrid Architectures (Jamba, StripedHyena, Zamba):
    Mixed Attention/SSM layers for combining transformer attention with
    state space models, with per-layer routing and state management.
"""

# Differential Transformer (always available)
from .diff_transformer import (
    DifferentialAttention,
    DifferentialAttentionConfig,
    DifferentialMarlinAttention,
    create_causal_mask,
    parse_diff_transformer_config,
)
from .rwkv import (
    RWKVBlock,
    RWKVChannelMixing,
    RWKVLayerState,
    RWKVTimeMixing,
    rwkv_wkv_batched,
    rwkv_wkv_single_token,
)

__all__ = [
    # Differential Transformer
    "DifferentialAttention",
    "DifferentialAttentionConfig",
    "DifferentialMarlinAttention",
    "create_causal_mask",
    "parse_diff_transformer_config",
    # Hybrid architecture support
    "HybridLayerType",
    "LayerProtocol",
    "LayerState",
    "StateType",
    "AttentionLayerConfig",
    "HybridArchitectureConfig",
    "HybridLayerConfig",
    "MambaLayerConfig",
    "build_hybrid_config",
    "HybridBlock",
    "HybridModel",
    "HybridModelBuilder",
    "MambaBlock",
    "MambaState",
    "MarlinMambaBlock",
    "SelectiveScanConfig",
    # RWKV
    "RWKVBlock",
    "RWKVChannelMixing",
    "RWKVLayerState",
    "RWKVTimeMixing",
    "rwkv_wkv_batched",
    "rwkv_wkv_single_token",
]

from .base import (
    HybridLayerType,
    LayerProtocol,
    LayerState,
    StateType,
)
from .config import (
    AttentionLayerConfig,
    HybridArchitectureConfig,
    HybridLayerConfig,
    MambaLayerConfig,
)
from .hybrid import (
    HybridBlock,
    HybridModel,
    HybridModelBuilder,
    build_hybrid_config,
)
from .mamba import (
    MambaBlock,
    MambaState,
    MarlinMambaBlock,
    SelectiveScanConfig,
)
