"""Vision-language projector implementations.

This module provides vision-to-LLM projector layers for multimodal models.
Projectors are kept at higher precision (FP8 or FP16) to preserve visual
information quality, as they are critical bottleneck components.

Supported architectures:
- LLaVA: Simple 2-layer MLP with GELU activation
- Qwen2-VL: Perceiver resampler with cross-attention and learned queries
- InternVL: QLLaMA-style projector similar to Perceiver but with position encodings

Usage:
    from metal_marlin.vision.projector import (
        VisionProjectorConfig,
        LLaVAProjector,
        detect_projector_type,
    )

    config = VisionProjectorConfig.from_hf_config(hf_config)
    projector = LLaVAProjector(config)
    llm_embeds = projector(vision_features)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._compat import HAS_MLX, from_numpy, mx, nn, to_numpy
from ..dtypes import DTypeConfig, get_default_config

if TYPE_CHECKING:
    import mlx.core as mx
    import mlx.nn as nn


class ProjectorType(Enum):
    """Supported vision projector architectures."""

    LLAVA_MLP = "llava_mlp"  # LLaVA-style 2-layer MLP
    PERCEIVER = "perceiver"  # Qwen2-VL Perceiver resampler
    QLLAMA = "qllama"  # InternVL QLLaMA-style projector
    LINEAR = "linear"  # Simple linear projection
    IDENTITY = "identity"  # No projection (dimensions must match)


@dataclass
class VisionProjectorConfig:
    """Configuration for vision-language projectors.

    This configuration supports multiple projector architectures used in
    vision-language models. The projector maps vision encoder outputs to
    the LLM's embedding space.

    Attributes:
        projector_type: Architecture type (llava_mlp, perceiver, qllama, linear).
        vision_hidden_size: Input dimension from vision encoder.
        llm_hidden_size: Output dimension matching LLM embedding size.
        intermediate_size: Hidden dimension for MLP projectors (default: 4x vision).
        num_query_tokens: Number of learned queries for Perceiver/QLLaMA projectors.
        num_attention_heads: Number of attention heads for cross-attention projectors.
        num_resampler_layers: Number of resampler layers (Perceiver/QLLaMA).
        activation: Activation function (gelu, silu, relu).
        precision: Weight precision ("fp16", "fp8", "bf16"). Higher precision
            recommended for projectors as they are information bottlenecks.
        use_bias: Whether to use bias in linear layers.
        dropout: Dropout probability (0.0 for inference).
        max_image_tokens: Maximum number of image tokens per image.
        supports_video: Whether the projector handles video frame sequences.
        dtype_config: Optional dtype configuration for precision control.
    """

    projector_type: ProjectorType = ProjectorType.LLAVA_MLP
    vision_hidden_size: int = 1024
    llm_hidden_size: int = 4096
    intermediate_size: int | None = None  # Default: 4x vision_hidden_size
    num_query_tokens: int = 64  # For Perceiver/QLLaMA
    num_attention_heads: int = 16
    num_resampler_layers: int = 2
    activation: Literal["gelu", "silu", "relu"] = "gelu"
    precision: Literal["fp16", "fp8", "bf16"] = "fp16"  # Keep projectors at higher precision
    use_bias: bool = True
    dropout: float = 0.0
    max_image_tokens: int = 576  # 24x24 patches
    supports_video: bool = False
    dtype_config: DTypeConfig | None = None

    def __post_init__(self) -> None:
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.vision_hidden_size
        if self.dtype_config is None:
            self.dtype_config = get_default_config()

    @classmethod
    def from_hf_config(cls, config: dict[str, Any]) -> VisionProjectorConfig:
        """Create config from HuggingFace model config.json.

        Parses configuration from various VLM architectures:
        - LLaVA: llava-hf/llava-v1.6-*
        - Qwen2-VL: Qwen/Qwen2-VL-*
        - InternVL: OpenGVLab/InternVL2-*

        Args:
            config: HuggingFace config dictionary from config.json

        Returns:
            VisionProjectorConfig instance
        """
        projector_type = detect_projector_type(config)

        # Vision encoder hidden size
        vision_hidden = config.get("vision_config", {}).get(
            "hidden_size",
            config.get("visual_hidden_size", config.get("vision_hidden_size", 1024)),
        )

        # LLM hidden size
        llm_hidden = config.get("hidden_size", config.get("text_config", {}).get("hidden_size", 4096))

        # Intermediate size for MLP projectors
        intermediate = config.get(
            "projector_hidden_size",
            config.get("mm_projector_hidden_size", 4 * vision_hidden),
        )

        # Perceiver/QLLaMA specific
        num_queries = config.get(
            "num_query_tokens",
            config.get("num_image_tokens", config.get("resampler_num_queries", 64)),
        )
        num_heads = config.get(
            "resampler_num_heads",
            config.get("num_attention_heads", 16),
        )
        num_layers = config.get(
            "resampler_num_layers",
            config.get("num_resampler_layers", 2),
        )

        # Activation
        activation = config.get(
            "projector_activation",
            config.get("mm_projector_activation", "gelu"),
        )
        if activation not in ("gelu", "silu", "relu"):
            activation = "gelu"

        # Video support
        supports_video = config.get("supports_video", False) or "video" in config.get("model_type", "").lower()

        return cls(
            projector_type=projector_type,
            vision_hidden_size=vision_hidden,
            llm_hidden_size=llm_hidden,
            intermediate_size=intermediate,
            num_query_tokens=num_queries,
            num_attention_heads=num_heads,
            num_resampler_layers=num_layers,
            activation=activation,
            supports_video=supports_video,
        )


def detect_projector_type(config: dict[str, Any]) -> ProjectorType:
    """Detect projector type from HuggingFace config.

    Examines model_type and architecture-specific keys to determine
    the appropriate projector architecture.

    Args:
        config: HuggingFace config dictionary

    Returns:
        Detected ProjectorType
    """
    model_type = config.get("model_type", "").lower()
    arch = config.get("architectures", [""])[0].lower() if config.get("architectures") else ""

    # Qwen2-VL uses Perceiver resampler
    if "qwen2_vl" in model_type or "qwen2-vl" in model_type or "qwen2vl" in arch:
        return ProjectorType.PERCEIVER

    # InternVL uses QLLaMA-style projector
    if "internvl" in model_type or "internlm" in model_type:
        if config.get("use_qllama_projector", False) or config.get("projector_type") == "qllama":
            return ProjectorType.QLLAMA
        # Newer InternVL versions may use different projectors
        return ProjectorType.QLLAMA

    # LLaVA and most other VLMs use MLP projector
    if "llava" in model_type or "llava" in arch:
        return ProjectorType.LLAVA_MLP

    # Check explicit projector_type in config
    explicit_type = config.get("projector_type", config.get("mm_projector_type", ""))
    if explicit_type:
        type_mapping = {
            "mlp": ProjectorType.LLAVA_MLP,
            "mlp2x_gelu": ProjectorType.LLAVA_MLP,
            "linear": ProjectorType.LINEAR,
            "perceiver": ProjectorType.PERCEIVER,
            "qllama": ProjectorType.QLLAMA,
            "identity": ProjectorType.IDENTITY,
        }
        return type_mapping.get(explicit_type.lower(), ProjectorType.LLAVA_MLP)

    # Default to LLaVA MLP
    return ProjectorType.LLAVA_MLP


def _get_activation(name: str) -> Any:
    """Get activation function by name."""
    if not HAS_MLX or nn is None:
        raise RuntimeError("MLX required for activation functions")

    if name == "gelu":
        return nn.gelu
    elif name == "silu":
        return nn.silu
    elif name == "relu":
        return nn.relu
    else:
        raise ValueError(f"Unknown activation: {name}")


class LLaVAProjector:
    """LLaVA-style 2-layer MLP projector.

    Maps vision encoder outputs to LLM embedding space through:
        vision_features -> linear1 -> GELU -> linear2 -> llm_embeddings

    This is a simple but effective projector used in LLaVA, LLaVA-1.5, and
    many derivative models.

    Args:
        config: VisionProjectorConfig with projector parameters
    """

    def __init__(self, config: VisionProjectorConfig):
        if HAS_MLX and nn is not None:
            nn.Module.__init__(self)

        self.config = config
        self.vision_hidden_size = config.vision_hidden_size
        self.llm_hidden_size = config.llm_hidden_size
        self.intermediate_size = config.intermediate_size or (4 * config.vision_hidden_size)

        if HAS_MLX and mx is not None:
            # MLX path: use standard nn.Linear (kept at higher precision)
            self.linear1 = nn.Linear(
                self.vision_hidden_size,
                self.intermediate_size,
                bias=config.use_bias,
            )
            self.linear2 = nn.Linear(
                self.intermediate_size,
                self.llm_hidden_size,
                bias=config.use_bias,
            )
            self._activation = _get_activation(config.activation)
        else:
            # Numpy fallback: store weights
            self.weight1 = np.zeros(
                (self.intermediate_size, self.vision_hidden_size), dtype=np.float16
            )
            self.weight2 = np.zeros(
                (self.llm_hidden_size, self.intermediate_size), dtype=np.float16
            )
            if config.use_bias:
                self.bias1 = np.zeros((self.intermediate_size,), dtype=np.float16)
                self.bias2 = np.zeros((self.llm_hidden_size,), dtype=np.float16)
            else:
                self.bias1 = None
                self.bias2 = None

    def __call__(self, x: Any) -> Any:
        """Forward pass through the projector.

        Args:
            x: Vision features [batch, num_patches, vision_hidden_size]
               or [num_patches, vision_hidden_size] for single image

        Returns:
            LLM embeddings [batch, num_patches, llm_hidden_size]
        """
        if HAS_MLX and mx is not None:
            return self._forward_mlx(x)
        return self._forward_numpy(x)

    def _forward_mlx(self, x: Any) -> Any:
        """MLX forward pass."""
        # Convert numpy to MLX if needed
        if isinstance(x, np.ndarray):
            x = mx.array(x)
        h = self.linear1(x)
        h = self._activation(h)
        return self.linear2(h)

    def _forward_numpy(self, x: Any) -> Any:
        """Numpy fallback forward pass."""
        x_np = to_numpy(x).astype(np.float32)
        orig_shape = x_np.shape

        # Flatten batch dimensions
        if x_np.ndim > 2:
            x_np = x_np.reshape(-1, x_np.shape[-1])

        # Linear 1
        h = x_np @ self.weight1.T
        if self.bias1 is not None:
            h = h + self.bias1

        # Activation (GELU approximation)
        if self.config.activation == "gelu":
            h = 0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h**3)))
        elif self.config.activation == "silu":
            h = h * (1 / (1 + np.exp(-h)))
        elif self.config.activation == "relu":
            h = np.maximum(h, 0)

        # Linear 2
        out = h @ self.weight2.T
        if self.bias2 is not None:
            out = out + self.bias2

        # Restore shape
        out_shape = list(orig_shape[:-1]) + [self.llm_hidden_size]
        out = out.reshape(out_shape)

        if HAS_MLX and mx is not None:
            return from_numpy(out, backend="mlx")
        return out

    def extra_repr(self) -> str:
        return (
            f"vision_hidden={self.vision_hidden_size}, "
            f"llm_hidden={self.llm_hidden_size}, "
            f"intermediate={self.intermediate_size}, "
            f"activation={self.config.activation}"
        )


class Qwen2VLProjector:
    """Qwen2-VL Perceiver resampler projector.

    Uses cross-attention with learned query tokens to resample vision
    features to a fixed number of tokens. This provides:
    - Flexible input resolution handling
    - Reduced sequence length for efficiency
    - Better handling of multi-image inputs

    Architecture:
        learned_queries + vision_features -> cross_attention -> output

    Args:
        config: VisionProjectorConfig with perceiver parameters
    """

    def __init__(self, config: VisionProjectorConfig):
        if HAS_MLX and nn is not None:
            nn.Module.__init__(self)

        self.config = config
        self.vision_hidden_size = config.vision_hidden_size
        self.llm_hidden_size = config.llm_hidden_size
        self.num_query_tokens = config.num_query_tokens
        self.num_heads = config.num_attention_heads
        self.num_layers = config.num_resampler_layers
        self.head_dim = config.llm_hidden_size // config.num_attention_heads

        if HAS_MLX and mx is not None:
            # Learned query tokens
            self.query_tokens = mx.zeros((1, self.num_query_tokens, self.llm_hidden_size))

            # Project vision features to match query dimension
            self.vision_projection = nn.Linear(
                self.vision_hidden_size,
                self.llm_hidden_size,
                bias=config.use_bias,
            )

            # Cross-attention layers
            self.layers = []
            for _ in range(self.num_layers):
                layer = _PerceiverResamplerLayer(
                    hidden_size=self.llm_hidden_size,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    use_bias=config.use_bias,
                    activation=config.activation,
                )
                self.layers.append(layer)

            # Output projection (optional, for dimension alignment)
            self.output_projection = nn.Linear(
                self.llm_hidden_size,
                self.llm_hidden_size,
                bias=config.use_bias,
            )
        else:
            # Numpy fallback - simplified
            self.query_tokens = np.zeros(
                (1, self.num_query_tokens, self.llm_hidden_size), dtype=np.float16
            )
            self.vision_proj_weight = np.zeros(
                (self.llm_hidden_size, self.vision_hidden_size), dtype=np.float16
            )
            self.output_proj_weight = np.zeros(
                (self.llm_hidden_size, self.llm_hidden_size), dtype=np.float16
            )

    def __call__(
        self,
        vision_features: Any,
        attention_mask: Any | None = None,
    ) -> Any:
        """Forward pass through Perceiver resampler.

        Args:
            vision_features: Vision encoder outputs [batch, num_patches, vision_hidden]
            attention_mask: Optional mask for variable-length sequences

        Returns:
            Resampled features [batch, num_query_tokens, llm_hidden]
        """
        if HAS_MLX and mx is not None:
            return self._forward_mlx(vision_features, attention_mask)
        return self._forward_numpy(vision_features)

    def _forward_mlx(self, vision_features: Any, attention_mask: Any | None) -> Any:
        """MLX forward pass with cross-attention."""
        # Convert numpy to MLX if needed
        if isinstance(vision_features, np.ndarray):
            vision_features = mx.array(vision_features)
        batch_size = vision_features.shape[0]

        # Project vision features
        vision_embeds = self.vision_projection(vision_features)

        # Expand query tokens for batch
        queries = mx.broadcast_to(
            self.query_tokens,
            (batch_size, self.num_query_tokens, self.llm_hidden_size),
        )

        # Apply cross-attention layers
        hidden_states = queries
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states=vision_embeds,
                attention_mask=attention_mask,
            )

        # Final projection
        output = self.output_projection(hidden_states)
        return output

    def _forward_numpy(self, vision_features: Any) -> Any:
        """Simplified numpy fallback (no cross-attention, just projection)."""
        x_np = to_numpy(vision_features).astype(np.float32)
        batch_size = x_np.shape[0]

        # Project vision features
        x_flat = x_np.reshape(-1, x_np.shape[-1])
        projected = x_flat @ self.vision_proj_weight.T
        projected = projected.reshape(batch_size, -1, self.llm_hidden_size)

        # Simple average pooling to num_query_tokens (simplified fallback)
        num_patches = projected.shape[1]
        if num_patches > self.num_query_tokens:
            # Chunk and average
            chunk_size = num_patches // self.num_query_tokens
            output = np.zeros(
                (batch_size, self.num_query_tokens, self.llm_hidden_size),
                dtype=np.float32,
            )
            for i in range(self.num_query_tokens):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, num_patches)
                output[:, i, :] = projected[:, start:end, :].mean(axis=1)
        else:
            # Pad with zeros
            output = np.zeros(
                (batch_size, self.num_query_tokens, self.llm_hidden_size),
                dtype=np.float32,
            )
            output[:, :num_patches, :] = projected

        if HAS_MLX and mx is not None:
            return from_numpy(output, backend="mlx")
        return output

    def extra_repr(self) -> str:
        return (
            f"vision_hidden={self.vision_hidden_size}, "
            f"llm_hidden={self.llm_hidden_size}, "
            f"num_queries={self.num_query_tokens}, "
            f"num_heads={self.num_heads}, "
            f"num_layers={self.num_layers}"
        )


class _PerceiverResamplerLayer:
    """Single layer of Perceiver resampler with cross-attention and FFN."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        use_bias: bool = True,
        activation: str = "gelu",
    ):
        if HAS_MLX and nn is not None:
            nn.Module.__init__(self)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        if HAS_MLX and nn is not None:
            # Cross-attention components
            self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=use_bias)
            self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=use_bias)
            self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=use_bias)
            self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=use_bias)

            # Layer norms
            self.attn_ln = nn.LayerNorm(hidden_size)
            self.ffn_ln = nn.LayerNorm(hidden_size)

            # FFN
            self.ffn_up = nn.Linear(hidden_size, hidden_size * 4, bias=use_bias)
            self.ffn_down = nn.Linear(hidden_size * 4, hidden_size, bias=use_bias)
            self._activation = _get_activation(activation)

    def __call__(
        self,
        hidden_states: Any,
        encoder_hidden_states: Any,
        attention_mask: Any | None = None,
    ) -> Any:
        """Forward pass with cross-attention and FFN.

        Args:
            hidden_states: Query tokens [batch, num_queries, hidden_size]
            encoder_hidden_states: Vision features [batch, num_patches, hidden_size]
            attention_mask: Optional attention mask

        Returns:
            Updated hidden states [batch, num_queries, hidden_size]
        """
        if not HAS_MLX or mx is None:
            # Simplified fallback: just return hidden states
            return hidden_states

        # Pre-norm for attention
        residual = hidden_states
        hidden_states = self.attn_ln(hidden_states)

        # Cross-attention
        batch_size, num_queries, _ = hidden_states.shape
        _, num_patches, _ = encoder_hidden_states.shape

        # Q from queries, K/V from vision features
        q = self.q_proj(hidden_states)
        k = self.k_proj(encoder_hidden_states)
        v = self.v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, num_queries, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, num_patches, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, num_patches, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Attention scores
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = mx.softmax(scores, axis=-1)
        attn_output = attn_weights @ v

        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, num_queries, self.num_heads * self.head_dim
        )
        attn_output = self.o_proj(attn_output)

        # Residual
        hidden_states = residual + attn_output

        # FFN with pre-norm
        residual = hidden_states
        hidden_states = self.ffn_ln(hidden_states)
        hidden_states = self.ffn_up(hidden_states)
        hidden_states = self._activation(hidden_states)
        hidden_states = self.ffn_down(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class InternVLProjector:
    """InternVL QLLaMA-style projector.

    Similar to Perceiver resampler but with:
    - Sinusoidal position encodings for vision features
    - Optional bidirectional attention in early layers
    - Supports larger query counts for high-resolution images

    Args:
        config: VisionProjectorConfig with QLLaMA parameters
    """

    def __init__(self, config: VisionProjectorConfig):
        if HAS_MLX and nn is not None:
            nn.Module.__init__(self)

        self.config = config
        self.vision_hidden_size = config.vision_hidden_size
        self.llm_hidden_size = config.llm_hidden_size
        self.num_query_tokens = config.num_query_tokens
        self.num_heads = config.num_attention_heads
        self.num_layers = config.num_resampler_layers
        self.head_dim = config.llm_hidden_size // config.num_attention_heads

        if HAS_MLX and mx is not None:
            # Learned query tokens
            self.query_tokens = mx.zeros((1, self.num_query_tokens, self.llm_hidden_size))

            # Vision input projection
            self.vision_projection = nn.Linear(
                self.vision_hidden_size,
                self.llm_hidden_size,
                bias=config.use_bias,
            )

            # Position encoding for vision features
            self.max_patches = config.max_image_tokens
            self._init_position_encoding()

            # Cross-attention layers
            self.layers = []
            for _ in range(self.num_layers):
                layer = _PerceiverResamplerLayer(
                    hidden_size=self.llm_hidden_size,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    use_bias=config.use_bias,
                    activation=config.activation,
                )
                self.layers.append(layer)

            # Output layer norm
            self.output_ln = nn.LayerNorm(self.llm_hidden_size)
        else:
            # Numpy fallback
            self.query_tokens = np.zeros(
                (1, self.num_query_tokens, self.llm_hidden_size), dtype=np.float16
            )
            self.vision_proj_weight = np.zeros(
                (self.llm_hidden_size, self.vision_hidden_size), dtype=np.float16
            )
            self.position_encoding = self._create_sinusoidal_positions_np(
                config.max_image_tokens, self.llm_hidden_size
            )

    def _init_position_encoding(self) -> None:
        """Initialize sinusoidal position encodings."""
        if not HAS_MLX or mx is None:
            return

        positions = mx.arange(self.max_patches, dtype=mx.float32)
        dim = self.llm_hidden_size
        inv_freq = 1.0 / (10000 ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

        # [max_patches, dim/2]
        sincos = mx.outer(positions, inv_freq)
        # [max_patches, dim]
        pos_enc = mx.concatenate([mx.sin(sincos), mx.cos(sincos)], axis=-1)
        self.position_encoding = pos_enc[None, :, :]  # [1, max_patches, dim]

    def _create_sinusoidal_positions_np(self, max_len: int, dim: int) -> np.ndarray:
        """Create sinusoidal position encodings (numpy)."""
        positions = np.arange(max_len, dtype=np.float32)
        inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        sincos = np.outer(positions, inv_freq)
        return np.concatenate([np.sin(sincos), np.cos(sincos)], axis=-1)[None, :, :]

    def __call__(
        self,
        vision_features: Any,
        attention_mask: Any | None = None,
    ) -> Any:
        """Forward pass through QLLaMA projector.

        Args:
            vision_features: Vision encoder outputs [batch, num_patches, vision_hidden]
            attention_mask: Optional mask for variable-length sequences

        Returns:
            Projected features [batch, num_query_tokens, llm_hidden]
        """
        if HAS_MLX and mx is not None:
            return self._forward_mlx(vision_features, attention_mask)
        return self._forward_numpy(vision_features)

    def _forward_mlx(self, vision_features: Any, attention_mask: Any | None) -> Any:
        """MLX forward pass."""
        # Convert numpy to MLX if needed
        if isinstance(vision_features, np.ndarray):
            vision_features = mx.array(vision_features)
        batch_size, num_patches, _ = vision_features.shape

        # Project vision features
        vision_embeds = self.vision_projection(vision_features)

        # Add position encoding
        if num_patches <= self.max_patches:
            pos_enc = self.position_encoding[:, :num_patches, :]
            vision_embeds = vision_embeds + pos_enc

        # Expand query tokens for batch
        queries = mx.broadcast_to(
            self.query_tokens,
            (batch_size, self.num_query_tokens, self.llm_hidden_size),
        )

        # Apply cross-attention layers
        hidden_states = queries
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states=vision_embeds,
                attention_mask=attention_mask,
            )

        # Final layer norm
        output = self.output_ln(hidden_states)
        return output

    def _forward_numpy(self, vision_features: Any) -> Any:
        """Numpy fallback."""
        x_np = to_numpy(vision_features).astype(np.float32)
        batch_size, num_patches, _ = x_np.shape

        # Project and add position encoding
        x_flat = x_np.reshape(-1, x_np.shape[-1])
        projected = x_flat @ self.vision_proj_weight.T
        projected = projected.reshape(batch_size, num_patches, self.llm_hidden_size)

        if num_patches <= self.max_patches:
            projected = projected + self.position_encoding[:, :num_patches, :]

        # Simple pooling fallback
        if num_patches > self.num_query_tokens:
            chunk_size = num_patches // self.num_query_tokens
            output = np.zeros(
                (batch_size, self.num_query_tokens, self.llm_hidden_size),
                dtype=np.float32,
            )
            for i in range(self.num_query_tokens):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, num_patches)
                output[:, i, :] = projected[:, start:end, :].mean(axis=1)
        else:
            output = np.zeros(
                (batch_size, self.num_query_tokens, self.llm_hidden_size),
                dtype=np.float32,
            )
            output[:, :num_patches, :] = projected

        if HAS_MLX and mx is not None:
            return from_numpy(output, backend="mlx")
        return output

    def extra_repr(self) -> str:
        return (
            f"vision_hidden={self.vision_hidden_size}, "
            f"llm_hidden={self.llm_hidden_size}, "
            f"num_queries={self.num_query_tokens}, "
            f"num_heads={self.num_heads}, "
            f"num_layers={self.num_layers}, "
            f"max_patches={self.max_patches}"
        )


class LinearProjector:
    """Simple linear projection for dimension matching.

    Used when vision and LLM dimensions differ but no complex
    transformation is needed.

    Args:
        config: VisionProjectorConfig
    """

    def __init__(self, config: VisionProjectorConfig):
        if HAS_MLX and nn is not None:
            nn.Module.__init__(self)

        self.config = config
        self.vision_hidden_size = config.vision_hidden_size
        self.llm_hidden_size = config.llm_hidden_size

        if HAS_MLX and nn is not None:
            self.projection = nn.Linear(
                self.vision_hidden_size,
                self.llm_hidden_size,
                bias=config.use_bias,
            )
        else:
            self.weight = np.zeros(
                (self.llm_hidden_size, self.vision_hidden_size), dtype=np.float16
            )
            self.bias = np.zeros((self.llm_hidden_size,), dtype=np.float16) if config.use_bias else None

    def __call__(self, x: Any) -> Any:
        if HAS_MLX and mx is not None:
            return self.projection(x)

        x_np = to_numpy(x).astype(np.float32)
        orig_shape = x_np.shape
        x_flat = x_np.reshape(-1, x_np.shape[-1])

        out = x_flat @ self.weight.T
        if self.bias is not None:
            out = out + self.bias

        out_shape = list(orig_shape[:-1]) + [self.llm_hidden_size]
        out = out.reshape(out_shape)

        if HAS_MLX and mx is not None:
            return from_numpy(out, backend="mlx")
        return out


class IdentityProjector:
    """Identity projector when dimensions already match."""

    def __init__(self, config: VisionProjectorConfig):
        self.config = config
        if config.vision_hidden_size != config.llm_hidden_size:
            raise ValueError(
                f"IdentityProjector requires matching dimensions, got "
                f"vision={config.vision_hidden_size}, llm={config.llm_hidden_size}"
            )

    def __call__(self, x: Any) -> Any:
        return x


class VisionProjector:
    """Factory class for creating vision projectors.

    Automatically selects the appropriate projector based on configuration.

    Usage:
        config = VisionProjectorConfig.from_hf_config(hf_config)
        projector = VisionProjector.from_config(config)
        output = projector(vision_features)
    """

    @staticmethod
    def from_config(config: VisionProjectorConfig) -> Any:
        """Create projector instance from configuration.

        Args:
            config: VisionProjectorConfig specifying projector type and parameters

        Returns:
            Projector instance (LLaVAProjector, Qwen2VLProjector, etc.)
        """
        projector_map = {
            ProjectorType.LLAVA_MLP: LLaVAProjector,
            ProjectorType.PERCEIVER: Qwen2VLProjector,
            ProjectorType.QLLAMA: InternVLProjector,
            ProjectorType.LINEAR: LinearProjector,
            ProjectorType.IDENTITY: IdentityProjector,
        }

        projector_cls = projector_map.get(config.projector_type)
        if projector_cls is None:
            raise ValueError(f"Unknown projector type: {config.projector_type}")

        return projector_cls(config)

    @staticmethod
    def from_hf_config(hf_config: dict[str, Any]) -> Any:
        """Create projector directly from HuggingFace config.

        Args:
            hf_config: HuggingFace config.json dictionary

        Returns:
            Projector instance
        """
        config = VisionProjectorConfig.from_hf_config(hf_config)
        return VisionProjector.from_config(config)


# Make classes inherit from nn.Module when MLX is available
if HAS_MLX and nn is not None:
    _OrigLLaVA = LLaVAProjector
    _OrigQwen2VL = Qwen2VLProjector
    _OrigInternVL = InternVLProjector
    _OrigLinear = LinearProjector
    _OrigLayer = _PerceiverResamplerLayer

    class LLaVAProjector(nn.Module):  # type: ignore[no-redef]
        """LLaVA-style 2-layer MLP projector."""

        __doc__ = _OrigLLaVA.__doc__

        def __init__(self, config: VisionProjectorConfig):
            super().__init__()
            _OrigLLaVA.__init__(self, config)

        __call__ = _OrigLLaVA.__call__
        _forward_mlx = _OrigLLaVA._forward_mlx
        _forward_numpy = _OrigLLaVA._forward_numpy
        extra_repr = _OrigLLaVA.extra_repr

    class Qwen2VLProjector(nn.Module):  # type: ignore[no-redef]
        """Qwen2-VL Perceiver resampler projector."""

        __doc__ = _OrigQwen2VL.__doc__

        def __init__(self, config: VisionProjectorConfig):
            super().__init__()
            _OrigQwen2VL.__init__(self, config)

        __call__ = _OrigQwen2VL.__call__
        _forward_mlx = _OrigQwen2VL._forward_mlx
        _forward_numpy = _OrigQwen2VL._forward_numpy
        extra_repr = _OrigQwen2VL.extra_repr

    class InternVLProjector(nn.Module):  # type: ignore[no-redef]
        """InternVL QLLaMA-style projector."""

        __doc__ = _OrigInternVL.__doc__

        def __init__(self, config: VisionProjectorConfig):
            super().__init__()
            _OrigInternVL.__init__(self, config)

        __call__ = _OrigInternVL.__call__
        _forward_mlx = _OrigInternVL._forward_mlx
        _forward_numpy = _OrigInternVL._forward_numpy
        _init_position_encoding = _OrigInternVL._init_position_encoding
        _create_sinusoidal_positions_np = _OrigInternVL._create_sinusoidal_positions_np
        extra_repr = _OrigInternVL.extra_repr

    class LinearProjector(nn.Module):  # type: ignore[no-redef]
        """Simple linear projection."""

        __doc__ = _OrigLinear.__doc__

        def __init__(self, config: VisionProjectorConfig):
            super().__init__()
            _OrigLinear.__init__(self, config)

        __call__ = _OrigLinear.__call__

    class _PerceiverResamplerLayer(nn.Module):  # type: ignore[no-redef]
        """Perceiver resampler layer with cross-attention."""

        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__()
            _OrigLayer.__init__(self, *args, **kwargs)

        __call__ = _OrigLayer.__call__
