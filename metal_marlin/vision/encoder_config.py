"""Vision encoder configuration for VLM quantization.

This module defines architecture-specific configurations for vision encoders
in popular Vision-Language Models. Each VLM uses different vision backbones:

- Qwen2-VL: Custom ViT with 2D RoPE (Rotary Position Embedding)
- LLaVA-1.5/1.6: OpenAI CLIP ViT-L/14 or SigLIP
- InternVL2: InternViT-6B (custom 6B parameter ViT)
- Pixtral: Custom vision encoder for Mistral VLM

Vision encoders have unique quantization challenges:
1. Patch embeddings convert raw pixels to token representations
2. Position encodings (learned, sinusoidal, or RoPE-2D) encode spatial information
3. Self-attention captures spatial relationships between patches
4. Cross-attention (if present) bridges vision features to LLM space

This module provides:
- VisionArchitecture enum for architecture detection
- VisionEncoderConfig dataclass with model-specific settings
- Factory methods for common VLM configurations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class VisionArchitecture(str, Enum):
    """Supported vision encoder architectures."""

    # OpenAI CLIP variants
    CLIP_VIT_L_14 = "clip-vit-l-14"  # LLaVA-1.5
    CLIP_VIT_L_14_336 = "clip-vit-l-14-336"  # LLaVA-1.5 high-res

    # SigLIP variants (LLaVA-1.6, LLaVA-NeXT)
    SIGLIP_SO400M = "siglip-so400m"

    # Qwen2-VL custom ViT
    QWEN2_VL_VIT = "qwen2-vl-vit"

    # InternVL variants
    INTERN_VIT_300M = "intern-vit-300m"
    INTERN_VIT_6B = "intern-vit-6b"

    # Pixtral (Mistral VLM)
    PIXTRAL_VIT = "pixtral-vit"

    # Generic ViT fallback
    GENERIC_VIT = "generic-vit"


class PositionEmbeddingType(str, Enum):
    """Types of position embeddings used in vision encoders."""

    LEARNED = "learned"  # Learnable position embeddings (CLIP)
    SINUSOIDAL_2D = "sinusoidal-2d"  # 2D sine-cosine (ViT original)
    ROPE_2D = "rope-2d"  # 2D Rotary Position Embedding (Qwen2-VL)
    ALIBI = "alibi"  # Attention with Linear Biases


@dataclass
class CrossAttentionConfig:
    """Configuration for vision-language cross-attention layers.

    Cross-attention layers are critical precision points in VLMs:
    - They bridge vision features (patches) and language features (tokens)
    - Errors here propagate to all downstream language generation
    - Generally should use higher precision than pure vision layers

    Attributes:
        num_layers: Number of cross-attention layers (0 = no cross-attention).
        hidden_dim: Hidden dimension for cross-attention.
        num_heads: Number of attention heads.
        num_kv_heads: Number of KV heads (for GQA). None = same as num_heads.
        use_gqa: Whether to use Grouped Query Attention.
        precision: Recommended precision ("fp8", "fp4", "bf16").
        group_size: Quantization group size if quantized.
    """

    num_layers: int = 0
    hidden_dim: int = 4096
    num_heads: int = 32
    num_kv_heads: int | None = None
    use_gqa: bool = False
    precision: str = "fp8"  # Higher precision recommended
    group_size: int = 64


@dataclass
class VisionEncoderConfig:
    """Configuration for vision encoder quantization.

    Provides model-specific settings for vision encoder architectures,
    including layer naming patterns, dimension sizes, and recommended
    precision settings.

    Attributes:
        architecture: Vision encoder architecture type.
        model_name: Human-readable model name.
        hidden_size: Vision encoder hidden dimension.
        intermediate_size: MLP intermediate dimension.
        num_attention_heads: Number of attention heads.
        num_hidden_layers: Number of transformer layers.
        patch_size: Patch size for image tokenization.
        image_size: Expected input image size (can be tuple for non-square).
        num_channels: Number of input channels (3 for RGB).
        position_embedding_type: Type of position embedding.
        cls_token: Whether architecture uses CLS token.
        pooling_type: Pooling strategy ("cls", "mean", "none").

        # Layer patterns for name matching
        layer_prefix: Prefix for encoder layers (e.g., "vision_tower.encoder.layers").
        qkv_pattern: Pattern for Q/K/V projection weights.
        mlp_pattern: Pattern for MLP weights.
        embed_pattern: Pattern for patch embedding weights.
        norm_pattern: Pattern for layer norm weights.

        # Precision settings
        patch_embed_precision: Precision for patch embeddings.
        attention_precision: Precision for attention layers.
        mlp_precision: Precision for MLP layers.
        norm_precision: Precision for layer norms.
        default_group_size: Default quantization group size.

        # Cross-attention (for architectures that use it)
        cross_attention: Cross-attention configuration or None.

        # Additional metadata
        metadata: Extra architecture-specific metadata.
    """

    # Architecture identification
    architecture: VisionArchitecture = VisionArchitecture.GENERIC_VIT
    model_name: str = "Generic ViT"

    # Model dimensions
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    patch_size: int = 14
    image_size: int | tuple[int, int] = 224
    num_channels: int = 3

    # Position embedding
    position_embedding_type: PositionEmbeddingType = PositionEmbeddingType.LEARNED
    cls_token: bool = True
    pooling_type: str = "cls"

    # Layer name patterns (for matching tensor names)
    layer_prefix: str = "vision_model.encoder.layers"
    qkv_pattern: str = "self_attn.{qkv}_proj"
    mlp_pattern: str = "mlp.{fc}"
    embed_pattern: str = "embeddings.patch_embedding"
    norm_pattern: str = "{layer_norm}"
    pos_embed_pattern: str = "embeddings.position_embedding"

    # Precision settings (per layer type)
    patch_embed_precision: str = "bf16"  # Critical, keep high
    attention_precision: str = "fp8"  # Sensitive to spatial
    mlp_precision: str = "fp4"  # Generally robust
    norm_precision: str = "bf16"  # Always keep high
    output_proj_precision: str = "fp8"  # Output projection

    # Quantization settings
    default_group_size: int = 64
    attention_group_size: int = 64
    mlp_group_size: int = 128

    # Cross-attention (optional)
    cross_attention: CrossAttentionConfig | None = None

    # Extra metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_patches(self) -> int:
        """Number of patches per image (excluding CLS token)."""
        if isinstance(self.image_size, tuple):
            h, w = self.image_size
        else:
            h = w = self.image_size
        return (h // self.patch_size) * (w // self.patch_size)

    @property
    def sequence_length(self) -> int:
        """Sequence length including CLS token if present."""
        return self.num_patches + (1 if self.cls_token else 0)

    @property
    def patch_embedding_dim(self) -> int:
        """Dimension of patch embedding projection."""
        return self.num_channels * self.patch_size * self.patch_size

    @classmethod
    def clip_vit_l_14(cls) -> VisionEncoderConfig:
        """Configuration for OpenAI CLIP ViT-L/14 (used in LLaVA-1.5)."""
        return cls(
            architecture=VisionArchitecture.CLIP_VIT_L_14,
            model_name="CLIP ViT-L/14",
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            num_hidden_layers=24,
            patch_size=14,
            image_size=224,
            position_embedding_type=PositionEmbeddingType.LEARNED,
            cls_token=True,
            pooling_type="cls",
            layer_prefix="vision_model.encoder.layers",
            qkv_pattern="self_attn.{qkv}_proj",
            mlp_pattern="mlp.{fc}",
            embed_pattern="vision_model.embeddings.patch_embedding",
            norm_pattern="layer_norm{idx}",
            # CLIP uses learned position embeddings - sensitive
            patch_embed_precision="bf16",
            attention_precision="fp8",
            mlp_precision="fp4",
            attention_group_size=64,
            mlp_group_size=128,
        )

    @classmethod
    def clip_vit_l_14_336(cls) -> VisionEncoderConfig:
        """Configuration for CLIP ViT-L/14 with 336px input (LLaVA-1.5)."""
        config = cls.clip_vit_l_14()
        config.architecture = VisionArchitecture.CLIP_VIT_L_14_336
        config.model_name = "CLIP ViT-L/14@336px"
        config.image_size = 336
        return config

    @classmethod
    def siglip_so400m(cls) -> VisionEncoderConfig:
        """Configuration for SigLIP SO400M (used in LLaVA-1.6, LLaVA-NeXT)."""
        return cls(
            architecture=VisionArchitecture.SIGLIP_SO400M,
            model_name="SigLIP SO400M",
            hidden_size=1152,
            intermediate_size=4304,
            num_attention_heads=16,
            num_hidden_layers=27,
            patch_size=14,
            image_size=384,  # SigLIP uses 384px
            position_embedding_type=PositionEmbeddingType.LEARNED,
            cls_token=False,  # SigLIP uses mean pooling, no CLS
            pooling_type="mean",
            layer_prefix="vision_tower.vision_model.encoder.layers",
            attention_precision="fp8",
            mlp_precision="fp4",
            attention_group_size=64,
            mlp_group_size=128,
        )

    @classmethod
    def qwen2_vl(cls) -> VisionEncoderConfig:
        """Configuration for Qwen2-VL vision encoder.

        Qwen2-VL uses a custom ViT with:
        - 2D RoPE (Rotary Position Embedding) for position encoding
        - Spatial merge for variable resolution support
        - 14x14 patches with dynamic resolution
        """
        return cls(
            architecture=VisionArchitecture.QWEN2_VL_VIT,
            model_name="Qwen2-VL ViT",
            hidden_size=1280,
            intermediate_size=5120,
            num_attention_heads=16,
            num_hidden_layers=32,
            patch_size=14,
            image_size=(980, 980),  # Max resolution, supports dynamic
            position_embedding_type=PositionEmbeddingType.ROPE_2D,
            cls_token=False,
            pooling_type="none",  # Features passed directly to perceiver
            layer_prefix="visual.blocks",
            qkv_pattern="attn.{qkv}",
            mlp_pattern="mlp.{fc}",
            embed_pattern="visual.patch_embed",
            norm_pattern="norm{idx}",
            pos_embed_pattern="visual.rope",  # RoPE parameters
            # Qwen2-VL attention is position-critical due to RoPE-2D
            patch_embed_precision="bf16",
            attention_precision="fp8",  # RoPE requires higher precision
            mlp_precision="fp4",
            attention_group_size=64,
            mlp_group_size=128,
            # Cross-attention resampler
            cross_attention=CrossAttentionConfig(
                num_layers=4,
                hidden_dim=1280,
                num_heads=16,
                precision="fp8",
                group_size=64,
            ),
            metadata={
                "rope_theta": 10000.0,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,  # For video
            },
        )

    @classmethod
    def intern_vit_6b(cls) -> VisionEncoderConfig:
        """Configuration for InternViT-6B (used in InternVL2).

        InternViT-6B is a 6-billion parameter vision encoder with:
        - 48 transformer layers
        - 6144 hidden dimension
        - Large MLP expansion (4x)
        """
        return cls(
            architecture=VisionArchitecture.INTERN_VIT_6B,
            model_name="InternViT-6B",
            hidden_size=3200,
            intermediate_size=12800,
            num_attention_heads=25,
            num_hidden_layers=48,
            patch_size=14,
            image_size=448,
            position_embedding_type=PositionEmbeddingType.LEARNED,
            cls_token=True,
            pooling_type="cls",
            layer_prefix="vision_model.encoder.layers",
            # InternViT uses custom naming
            qkv_pattern="attention.{attention_type}",
            mlp_pattern="mlp.{fc}",
            embed_pattern="vision_model.embeddings",
            # Large model benefits from aggressive quantization
            patch_embed_precision="bf16",
            attention_precision="fp8",
            mlp_precision="fp4",
            attention_group_size=64,
            mlp_group_size=128,
            # Cross-attention QLLaMA projector
            cross_attention=CrossAttentionConfig(
                num_layers=6,
                hidden_dim=3200,
                num_heads=25,
                precision="fp8",
                group_size=64,
            ),
            metadata={
                "use_flash_attention": True,
                "layer_scale": True,
            },
        )

    @classmethod
    def intern_vit_300m(cls) -> VisionEncoderConfig:
        """Configuration for InternViT-300M (smaller InternVL variant)."""
        return cls(
            architecture=VisionArchitecture.INTERN_VIT_300M,
            model_name="InternViT-300M",
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            num_hidden_layers=24,
            patch_size=14,
            image_size=224,
            position_embedding_type=PositionEmbeddingType.LEARNED,
            cls_token=True,
            pooling_type="cls",
            layer_prefix="vision_model.encoder.layers",
            patch_embed_precision="bf16",
            attention_precision="fp8",
            mlp_precision="fp4",
            attention_group_size=64,
            mlp_group_size=128,
        )

    @classmethod
    def pixtral(cls) -> VisionEncoderConfig:
        """Configuration for Pixtral vision encoder (Mistral VLM).

        Pixtral uses a custom vision encoder designed for efficiency:
        - Efficient attention patterns
        - Lightweight cross-attention
        """
        return cls(
            architecture=VisionArchitecture.PIXTRAL_VIT,
            model_name="Pixtral ViT",
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            num_hidden_layers=24,
            patch_size=16,  # Pixtral uses 16x16 patches
            image_size=384,
            position_embedding_type=PositionEmbeddingType.ROPE_2D,
            cls_token=False,
            pooling_type="none",
            layer_prefix="vision_encoder.transformer.layers",
            patch_embed_precision="bf16",
            attention_precision="fp8",
            mlp_precision="fp4",
            attention_group_size=64,
            mlp_group_size=128,
            cross_attention=CrossAttentionConfig(
                num_layers=2,
                hidden_dim=1024,
                num_heads=16,
                precision="fp8",
                group_size=64,
            ),
            metadata={
                "rope_theta": 10000.0,
                "use_gated_mlp": True,
            },
        )

    @classmethod
    def from_hf_config(
        cls,
        config_path: str | Path | None = None,
        config_dict: dict[str, Any] | None = None,
    ) -> VisionEncoderConfig:
        """Create VisionEncoderConfig from HuggingFace config.

        Attempts to detect architecture from config and return appropriate
        preset. Falls back to generic ViT if architecture is unknown.

        Args:
            config_path: Path to config.json file.
            config_dict: Pre-loaded config dictionary.

        Returns:
            VisionEncoderConfig for the detected architecture.
        """
        import json

        if config_dict is None:
            if config_path is None:
                raise ValueError("Either config_path or config_dict required")
            config_path = Path(config_path)
            with open(config_path) as f:
                config_dict = json.load(f)

        # Extract vision config if nested
        vision_config = config_dict.get("vision_config", config_dict)

        # Try to detect architecture
        model_type = config_dict.get("model_type", "").lower()
        vision_model_type = vision_config.get("model_type", "").lower()

        # Architecture detection
        if "qwen2_vl" in model_type or "qwen2-vl" in model_type:
            return cls.qwen2_vl()
        elif "internvl" in model_type or "intern_vit" in vision_model_type:
            hidden = vision_config.get("hidden_size", 1024)
            if hidden >= 3000:
                return cls.intern_vit_6b()
            return cls.intern_vit_300m()
        elif "siglip" in vision_model_type:
            return cls.siglip_so400m()
        elif "clip" in vision_model_type or "llava" in model_type:
            image_size = vision_config.get("image_size", 224)
            if image_size >= 336:
                return cls.clip_vit_l_14_336()
            return cls.clip_vit_l_14()
        elif "pixtral" in model_type:
            return cls.pixtral()

        # Generic fallback with config values
        return cls(
            architecture=VisionArchitecture.GENERIC_VIT,
            model_name=f"Generic ViT ({model_type})",
            hidden_size=vision_config.get("hidden_size", 1024),
            intermediate_size=vision_config.get("intermediate_size", 4096),
            num_attention_heads=vision_config.get("num_attention_heads", 16),
            num_hidden_layers=vision_config.get("num_hidden_layers", 24),
            patch_size=vision_config.get("patch_size", 14),
            image_size=vision_config.get("image_size", 224),
        )

    def get_layer_precision(self, layer_name: str) -> tuple[str, int]:
        """Get recommended precision and group size for a layer.

        Args:
            layer_name: Full layer name from model state dict.

        Returns:
            (precision, group_size) tuple.
        """
        name_lower = layer_name.lower()

        # Patch embeddings - always high precision
        if "patch" in name_lower and "embed" in name_lower:
            return self.patch_embed_precision, 0

        # Position embeddings - keep high precision
        if "position" in name_lower or "pos_embed" in name_lower:
            return "bf16", 0

        # Layer norms - always high precision
        if "norm" in name_lower or "layernorm" in name_lower:
            return self.norm_precision, 0

        # CLS token - keep high precision
        if "cls" in name_lower:
            return "bf16", 0

        # Attention Q/K/V projections
        if any(
            p in name_lower for p in ["q_proj", "k_proj", "v_proj", "qkv", "query", "key", "value"]
        ):
            return self.attention_precision, self.attention_group_size

        # Attention output projection
        if "o_proj" in name_lower or "out_proj" in name_lower:
            return self.output_proj_precision, self.attention_group_size

        # MLP layers
        if any(p in name_lower for p in ["mlp", "fc", "gate", "up_proj", "down_proj"]):
            return self.mlp_precision, self.mlp_group_size

        # Cross-attention (if present)
        if self.cross_attention and "cross" in name_lower:
            return self.cross_attention.precision, self.cross_attention.group_size

        # Default
        return self.mlp_precision, self.default_group_size

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "architecture": self.architecture.value,
            "model_name": self.model_name,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "patch_size": self.patch_size,
            "image_size": self.image_size,
            "num_channels": self.num_channels,
            "position_embedding_type": self.position_embedding_type.value,
            "cls_token": self.cls_token,
            "pooling_type": self.pooling_type,
            "layer_prefix": self.layer_prefix,
            "patch_embed_precision": self.patch_embed_precision,
            "attention_precision": self.attention_precision,
            "mlp_precision": self.mlp_precision,
            "norm_precision": self.norm_precision,
            "attention_group_size": self.attention_group_size,
            "mlp_group_size": self.mlp_group_size,
            "cross_attention": (
                {
                    "num_layers": self.cross_attention.num_layers,
                    "hidden_dim": self.cross_attention.hidden_dim,
                    "num_heads": self.cross_attention.num_heads,
                    "precision": self.cross_attention.precision,
                    "group_size": self.cross_attention.group_size,
                }
                if self.cross_attention
                else None
            ),
            "metadata": self.metadata,
        }


def detect_vision_architecture(model_path: str | Path) -> VisionArchitecture:
    """Detect vision encoder architecture from model directory.

    Args:
        model_path: Path to model directory containing config.json.

    Returns:
        Detected VisionArchitecture enum value.
    """
    import json

    model_path = Path(model_path)
    config_file = model_path / "config.json"

    if not config_file.exists():
        return VisionArchitecture.GENERIC_VIT

    with open(config_file) as f:
        config = json.load(f)

    model_type = config.get("model_type", "").lower()
    vision_config = config.get("vision_config", {})
    vision_type = vision_config.get("model_type", "").lower()

    # Detection logic
    if "qwen2_vl" in model_type or "qwen2-vl" in model_type:
        return VisionArchitecture.QWEN2_VL_VIT
    elif "internvl" in model_type or "intern_vit" in vision_type:
        hidden = vision_config.get("hidden_size", 1024)
        return (
            VisionArchitecture.INTERN_VIT_6B
            if hidden >= 3000
            else VisionArchitecture.INTERN_VIT_300M
        )
    elif "siglip" in vision_type:
        return VisionArchitecture.SIGLIP_SO400M
    elif "clip" in vision_type:
        image_size = vision_config.get("image_size", 224)
        return (
            VisionArchitecture.CLIP_VIT_L_14_336
            if image_size >= 336
            else VisionArchitecture.CLIP_VIT_L_14
        )
    elif "pixtral" in model_type:
        return VisionArchitecture.PIXTRAL_VIT

    return VisionArchitecture.GENERIC_VIT
