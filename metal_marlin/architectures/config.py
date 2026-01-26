"""Configuration dataclasses for hybrid architectures.

Provides flexible configuration for models that mix different layer types:
- Per-layer type specification (attention, mamba, etc.)
- Per-layer quantization configuration
- Architecture detection from HuggingFace configs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..mixed_precision import (
    LayerQuantConfig,
    MixedPrecisionConfig,
    Precision,
)
from .base import HybridLayerType


@dataclass
class AttentionLayerConfig:
    """Configuration for attention layers in hybrid models.

    Attributes:
        num_heads: Number of attention heads.
        num_kv_heads: Number of KV heads (for GQA). None = MHA.
        head_dim: Dimension per head. If None, computed as hidden_size // num_heads.
        rope_theta: Base frequency for RoPE.
        max_position_embeddings: Maximum sequence length for RoPE precomputation.
        sliding_window: Sliding window size for local attention. None = full attention.
        use_flash_attention: Whether to use flash attention kernels.
    """

    num_heads: int = 32
    num_kv_heads: int | None = None
    head_dim: int | None = None
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096
    sliding_window: int | None = None
    use_flash_attention: bool = True


@dataclass
class MambaLayerConfig:
    """Configuration for Mamba/SSM layers.

    Attributes:
        d_state: State dimension (N in paper). Higher = more expressive.
        d_conv: Convolutional kernel width. Standard is 4.
        expand: Expansion factor for inner dimension. Standard is 2.
        dt_rank: Rank for delta projection. "auto" = ceil(hidden_size / 16).
        dt_min: Minimum value for delta (time step).
        dt_max: Maximum value for delta.
        dt_init: Initialization method for delta projection.
        dt_scale: Scaling factor for delta initialization.
        dt_init_floor: Minimum value for dt initialization.
        conv_bias: Whether to use bias in conv projection.
        bias: Whether to use bias in linear projections.
        use_fast_path: Use optimized fused kernel when available.
    """

    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | str = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False
    use_fast_path: bool = True


@dataclass
class HybridLayerConfig:
    """Configuration for a single layer in a hybrid model.

    Combines layer type, type-specific config, and quantization settings.

    Attributes:
        layer_type: What kind of layer (attention, mamba, etc.).
        attention_config: Config if layer_type is ATTENTION.
        mamba_config: Config if layer_type is MAMBA/MAMBA2.
        quant_config: Per-layer quantization settings.
        shared_layer_idx: For SHARED_ATTENTION, which layer's weights to use.
        mlp_intermediate_size: MLP intermediate dimension (None = 4 * hidden).
        use_gated_mlp: Whether to use gated MLP (SwiGLU/GeGLU).
        mlp_activation: Activation function for MLP.
        norm_eps: Epsilon for layer normalization.
    """

    layer_type: HybridLayerType = HybridLayerType.ATTENTION
    attention_config: AttentionLayerConfig | None = None
    mamba_config: MambaLayerConfig | None = None
    quant_config: LayerQuantConfig | None = None
    shared_layer_idx: int | None = None  # For Zamba-style shared attention
    mlp_intermediate_size: int | None = None
    use_gated_mlp: bool = True
    mlp_activation: str = "silu"
    norm_eps: float = 1e-6


@dataclass
class HybridArchitectureConfig:
    """Full configuration for a hybrid architecture model.

    Supports heterogeneous layer stacks with per-layer-type quantization.

    Example Jamba config:
        config = HybridArchitectureConfig(
            hidden_size=4096,
            num_layers=32,
            layer_types=["mamba", "mamba", "attention", "mamba", ...],
            attention_config=AttentionLayerConfig(num_heads=32),
            mamba_config=MambaLayerConfig(d_state=16),
            attention_quant=LayerQuantConfig(Precision.FP4_E2M1, 64),
            mamba_quant=LayerQuantConfig(Precision.FP4_E2M1, 128),
        )

    Attributes:
        hidden_size: Model hidden dimension.
        num_layers: Total number of layers.
        vocab_size: Vocabulary size for embeddings.
        layer_types: List of layer type names or HybridLayerType values.
        layer_configs: Optional per-layer override configs.
        attention_config: Default config for attention layers.
        mamba_config: Default config for mamba layers.
        attention_quant: Default quantization for attention layers.
        mamba_quant: Default quantization for mamba layers.
        mlp_quant: Default quantization for MLP layers.
        embed_quant: Quantization for embedding layer.
        norm_quant: Quantization for normalization layers.
        rms_norm_eps: Epsilon for RMSNorm.
        tie_word_embeddings: Whether to tie input/output embeddings.
        use_cache: Whether to use KV cache during inference.
        architecture_name: Name of the architecture (e.g., "jamba", "zamba").
    """

    hidden_size: int = 4096
    num_layers: int = 32
    vocab_size: int = 32000
    layer_types: list[str | HybridLayerType] = field(default_factory=list)
    layer_configs: list[HybridLayerConfig] | None = None

    # Default configs for each layer type
    attention_config: AttentionLayerConfig = field(default_factory=AttentionLayerConfig)
    mamba_config: MambaLayerConfig = field(default_factory=MambaLayerConfig)

    # Per-layer-type quantization
    attention_quant: LayerQuantConfig = field(
        default_factory=lambda: LayerQuantConfig(Precision.FP4_E2M1, 64)
    )
    mamba_quant: LayerQuantConfig = field(
        default_factory=lambda: LayerQuantConfig(Precision.FP4_E2M1, 128)
    )
    mlp_quant: LayerQuantConfig = field(
        default_factory=lambda: LayerQuantConfig(Precision.FP4_E2M1, 128)
    )
    embed_quant: LayerQuantConfig = field(
        default_factory=lambda: LayerQuantConfig(Precision.BF16)
    )
    norm_quant: LayerQuantConfig = field(
        default_factory=lambda: LayerQuantConfig(Precision.BF16)
    )

    # General model settings
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    use_cache: bool = True
    architecture_name: str = ""

    def __post_init__(self) -> None:
        """Normalize layer_types to HybridLayerType enum values."""
        normalized = []
        for lt in self.layer_types:
            if isinstance(lt, str):
                normalized.append(_parse_layer_type(lt))
            else:
                normalized.append(lt)
        self.layer_types = normalized

        # Auto-fill layer_configs if not provided
        if self.layer_configs is None and self.layer_types:
            self.layer_configs = [
                self._create_layer_config(lt, idx)
                for idx, lt in enumerate(self.layer_types)
            ]

    def _create_layer_config(
        self, layer_type: HybridLayerType, layer_idx: int
    ) -> HybridLayerConfig:
        """Create default config for a layer based on its type."""
        if layer_type == HybridLayerType.ATTENTION:
            return HybridLayerConfig(
                layer_type=layer_type,
                attention_config=self.attention_config,
                quant_config=self.attention_quant,
                norm_eps=self.rms_norm_eps,
            )
        elif layer_type in (HybridLayerType.MAMBA, HybridLayerType.MAMBA2):
            return HybridLayerConfig(
                layer_type=layer_type,
                mamba_config=self.mamba_config,
                quant_config=self.mamba_quant,
                norm_eps=self.rms_norm_eps,
            )
        elif layer_type == HybridLayerType.SHARED_ATTENTION:
            # Zamba-style: find nearest attention layer to share with
            shared_idx = self._find_shared_attention_layer(layer_idx)
            return HybridLayerConfig(
                layer_type=layer_type,
                attention_config=self.attention_config,
                quant_config=self.attention_quant,
                shared_layer_idx=shared_idx,
                norm_eps=self.rms_norm_eps,
            )
        else:
            return HybridLayerConfig(
                layer_type=layer_type,
                quant_config=self.mlp_quant,
                norm_eps=self.rms_norm_eps,
            )

    def _find_shared_attention_layer(self, current_idx: int) -> int:
        """Find the nearest preceding attention layer for weight sharing."""
        for idx in range(current_idx - 1, -1, -1):
            if self.layer_types[idx] == HybridLayerType.ATTENTION:
                return idx
        # Default to layer 0 if no attention found
        return 0

    def get_layer_quant(self, layer_idx: int) -> LayerQuantConfig:
        """Get quantization config for a specific layer."""
        if self.layer_configs and layer_idx < len(self.layer_configs):
            cfg = self.layer_configs[layer_idx]
            if cfg.quant_config is not None:
                return cfg.quant_config

        # Fall back to type-based defaults
        if layer_idx < len(self.layer_types):
            lt = self.layer_types[layer_idx]
            if lt == HybridLayerType.ATTENTION:
                return self.attention_quant
            elif lt in (HybridLayerType.MAMBA, HybridLayerType.MAMBA2):
                return self.mamba_quant

        return self.mlp_quant

    def to_mixed_precision_config(self) -> MixedPrecisionConfig:
        """Convert to MixedPrecisionConfig for quantization pipeline."""
        return MixedPrecisionConfig(
            embeddings=self.embed_quant,
            lm_head=self.embed_quant,
            norms=self.norm_quant,
            attention_qkv=self.attention_quant,
            attention_out=self.attention_quant,
            mlp_gate=self.mlp_quant,
            mlp_up=self.mlp_quant,
            mlp_down=self.mlp_quant,
        )

    @classmethod
    def from_hf_config(
        cls,
        config_dict: dict[str, Any],
        architecture_name: str | None = None,
    ) -> HybridArchitectureConfig:
        """Create config from HuggingFace model config dict.

        Supports:
        - Jamba (AI21)
        - Zamba (Zyphra)
        - StripedHyena (Together)
        - Mamba-based models

        Args:
            config_dict: HuggingFace config.json contents.
            architecture_name: Override architecture detection.

        Returns:
            HybridArchitectureConfig for the model.
        """
        arch = architecture_name or detect_hybrid_architecture(config_dict)

        # Common fields
        hidden_size = config_dict.get("hidden_size", 4096)
        num_layers = config_dict.get("num_hidden_layers", 32)
        vocab_size = config_dict.get("vocab_size", 32000)

        # Build layer type list based on architecture
        layer_types = _get_layer_types_for_architecture(arch, num_layers, config_dict)

        # Build attention config
        attention_config = AttentionLayerConfig(
            num_heads=config_dict.get("num_attention_heads", 32),
            num_kv_heads=config_dict.get("num_key_value_heads"),
            rope_theta=config_dict.get("rope_theta", 10000.0),
            max_position_embeddings=config_dict.get("max_position_embeddings", 4096),
        )

        # Build mamba config
        mamba_config = MambaLayerConfig(
            d_state=config_dict.get("mamba_d_state", config_dict.get("state_size", 16)),
            d_conv=config_dict.get("mamba_d_conv", config_dict.get("conv_kernel", 4)),
            expand=config_dict.get("mamba_expand", config_dict.get("expand", 2)),
        )

        return cls(
            hidden_size=hidden_size,
            num_layers=num_layers,
            vocab_size=vocab_size,
            layer_types=layer_types,
            attention_config=attention_config,
            mamba_config=mamba_config,
            rms_norm_eps=config_dict.get("rms_norm_eps", 1e-6),
            tie_word_embeddings=config_dict.get("tie_word_embeddings", False),
            architecture_name=arch,
        )

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> HybridArchitectureConfig:
        """Load config from a model directory.

        Args:
            model_path: Path to model directory containing config.json.

        Returns:
            HybridArchitectureConfig for the model.
        """
        import json

        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            config_dict = json.load(f)

        return cls.from_hf_config(config_dict)


def _parse_layer_type(name: str) -> HybridLayerType:
    """Parse layer type from string name."""
    name_lower = name.lower().strip()
    mapping = {
        "attention": HybridLayerType.ATTENTION,
        "attn": HybridLayerType.ATTENTION,
        "self_attention": HybridLayerType.ATTENTION,
        "mamba": HybridLayerType.MAMBA,
        "ssm": HybridLayerType.MAMBA,
        "mamba2": HybridLayerType.MAMBA2,
        "ssd": HybridLayerType.MAMBA2,
        "hyena": HybridLayerType.HYENA,
        "linear_attention": HybridLayerType.LINEAR_ATTENTION,
        "rwkv": HybridLayerType.LINEAR_ATTENTION,
        "eagle": HybridLayerType.LINEAR_ATTENTION,
        "retnet": HybridLayerType.LINEAR_ATTENTION,
        "mlp": HybridLayerType.MLP_ONLY,
        "mlp_only": HybridLayerType.MLP_ONLY,
        "moe": HybridLayerType.MOE,
        "mixture_of_experts": HybridLayerType.MOE,
        "shared_attention": HybridLayerType.SHARED_ATTENTION,
        "shared": HybridLayerType.SHARED_ATTENTION,
    }
    if name_lower not in mapping:
        raise ValueError(
            f"Unknown layer type: {name}. "
            f"Supported: {list(mapping.keys())}"
        )
    return mapping[name_lower]


def detect_hybrid_architecture(config_dict: dict[str, Any]) -> str:
    """Detect hybrid architecture from HuggingFace config.

    Args:
        config_dict: Model config.json contents.

    Returns:
        Architecture name: "jamba", "zamba", "stripedhyena", "mamba", or "dense".
    """
    # Check model_type field
    model_type = config_dict.get("model_type", "").lower()

    if "jamba" in model_type:
        return "jamba"
    if "zamba" in model_type:
        return "zamba"
    if "rwkv" in model_type or "eagle" in model_type:
        return "rwkv"
    if "stripedhyena" in model_type or "striped_hyena" in model_type:
        return "stripedhyena"
    if "mamba" in model_type:
        return "mamba"

    # Check architectures list
    architectures = config_dict.get("architectures", [])
    for arch in architectures:
        arch_lower = arch.lower()
        if "jamba" in arch_lower:
            return "jamba"
        if "zamba" in arch_lower:
            return "zamba"
        if "rwkv" in arch_lower or "eagle" in arch_lower:
            return "rwkv"
        if "hyena" in arch_lower:
            return "stripedhyena"
        if "mamba" in arch_lower:
            return "mamba"

    # Check for hybrid indicators
    if "mamba_d_state" in config_dict or "state_size" in config_dict:
        # Has mamba config - check for attention too
        if config_dict.get("num_attention_heads"):
            return "hybrid"  # Generic hybrid
        return "mamba"

    # Check for explicit layer type list
    if "layer_types" in config_dict:
        return "hybrid"

    # Check for attn_layer_idx (Jamba pattern)
    if "attn_layer_idx" in config_dict:
        return "jamba"

    # Default to dense transformer
    return "dense"


def _get_layer_types_for_architecture(
    arch: str, num_layers: int, config_dict: dict[str, Any]
) -> list[HybridLayerType]:
    """Generate layer type list for known architectures."""

    if arch == "jamba":
        # Jamba: Mamba layers with periodic attention
        # Default pattern: attention every 8 layers
        attn_indices = set(config_dict.get("attn_layer_idx", list(range(7, num_layers, 8))))
        return [
            HybridLayerType.ATTENTION if i in attn_indices else HybridLayerType.MAMBA
            for i in range(num_layers)
        ]

    elif arch == "zamba":
        # Zamba: SSM layers with shared attention
        # Pattern: SSM-SSM-SharedAttn repeated, with one true attention at layer 0
        types = []
        for i in range(num_layers):
            if i == 0:
                types.append(HybridLayerType.ATTENTION)
            elif (i + 1) % 3 == 0:  # Every 3rd layer is shared attention
                types.append(HybridLayerType.SHARED_ATTENTION)
            else:
                types.append(HybridLayerType.MAMBA)
        return types

    elif arch == "stripedhyena":
        # StripedHyena: Alternating Hyena and Attention
        return [
            HybridLayerType.ATTENTION if i % 2 == 0 else HybridLayerType.HYENA
            for i in range(num_layers)
        ]

    elif arch == "mamba":
        # Pure Mamba model
        return [HybridLayerType.MAMBA] * num_layers

    elif arch == "rwkv":
        # Pure RWKV model (linear attention)
        return [HybridLayerType.LINEAR_ATTENTION] * num_layers

    elif arch == "hybrid":
        # Try to extract from config
        if "layer_types" in config_dict:
            return [_parse_layer_type(lt) for lt in config_dict["layer_types"]]
        # Default to alternating
        return [
            HybridLayerType.ATTENTION if i % 2 == 0 else HybridLayerType.MAMBA
            for i in range(num_layers)
        ]

    else:
        # Dense transformer
        return [HybridLayerType.ATTENTION] * num_layers


# Convenience functions for common hybrid patterns
def create_jamba_config(
    num_layers: int = 32,
    hidden_size: int = 4096,
    attn_every: int = 8,
    **kwargs: Any,
) -> HybridArchitectureConfig:
    """Create a Jamba-style config with periodic attention layers.

    Args:
        num_layers: Total number of layers.
        hidden_size: Model hidden dimension.
        attn_every: Insert attention layer every N layers.
        **kwargs: Additional config overrides.

    Returns:
        HybridArchitectureConfig for Jamba-style model.
    """
    layer_types = [
        HybridLayerType.ATTENTION if (i + 1) % attn_every == 0 else HybridLayerType.MAMBA
        for i in range(num_layers)
    ]

    return HybridArchitectureConfig(
        hidden_size=hidden_size,
        num_layers=num_layers,
        layer_types=layer_types,
        architecture_name="jamba",
        **kwargs,
    )


def create_zamba_config(
    num_layers: int = 24,
    hidden_size: int = 3072,
    **kwargs: Any,
) -> HybridArchitectureConfig:
    """Create a Zamba-style config with shared attention.

    Zamba uses a single set of attention weights shared across multiple layers,
    interleaved with Mamba blocks.

    Args:
        num_layers: Total number of layers.
        hidden_size: Model hidden dimension.
        **kwargs: Additional config overrides.

    Returns:
        HybridArchitectureConfig for Zamba-style model.
    """
    # Zamba pattern: one attention layer, rest alternate Mamba + shared attention
    layer_types = []
    for i in range(num_layers):
        if i == 0:
            layer_types.append(HybridLayerType.ATTENTION)
        elif (i + 1) % 3 == 0:
            layer_types.append(HybridLayerType.SHARED_ATTENTION)
        else:
            layer_types.append(HybridLayerType.MAMBA)

    return HybridArchitectureConfig(
        hidden_size=hidden_size,
        num_layers=num_layers,
        layer_types=layer_types,
        architecture_name="zamba",
        **kwargs,
    )


def create_interleaved_config(
    num_layers: int = 32,
    hidden_size: int = 4096,
    layer_a: HybridLayerType = HybridLayerType.ATTENTION,
    layer_b: HybridLayerType = HybridLayerType.MAMBA,
    **kwargs: Any,
) -> HybridArchitectureConfig:
    """Create config with alternating layer types.

    Useful for StripedHyena-style architectures.

    Args:
        num_layers: Total number of layers.
        hidden_size: Model hidden dimension.
        layer_a: First layer type (even indices).
        layer_b: Second layer type (odd indices).
        **kwargs: Additional config overrides.

    Returns:
        HybridArchitectureConfig with alternating layers.
    """
    layer_types = [layer_a if i % 2 == 0 else layer_b for i in range(num_layers)]

    return HybridArchitectureConfig(
        hidden_size=hidden_size,
        num_layers=num_layers,
        layer_types=layer_types,
        architecture_name="interleaved",
        **kwargs,
    )
