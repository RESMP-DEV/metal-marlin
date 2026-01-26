"""Hybrid model implementation for mixed Attention/SSM architectures.

Provides:
- HybridBlock: Routes to appropriate layer implementation based on type
- HybridModel: Full model with heterogeneous layer stack
- HybridModelBuilder: Factory for constructing models from config
- build_hybrid_config: Build architecture config from a simple dict format
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._compat import HAS_MLX, require_mlx
from ..mixed_precision import LayerQuantConfig, Precision
from .base import HybridLayerType, LayerState, StateType
from .config import (
    AttentionLayerConfig,
    HybridArchitectureConfig,
    HybridLayerConfig,
    MambaLayerConfig,
)

if HAS_MLX:
    import mlx.core as mx
    import mlx.nn as nn
else:
    mx = None
    nn = None


class HybridBlock(nn.Module if HAS_MLX else object):
    """Unified block that dispatches to appropriate layer implementation.

    This is the core building block for hybrid models. Each block wraps
    a specific layer implementation (attention, mamba, etc.) and provides
    a consistent interface for the model.

    The block handles:
    - Layer normalization (pre-norm architecture)
    - Routing to the appropriate layer implementation
    - State management for different layer types
    - Residual connections

    Attributes:
        layer_type: What kind of layer this block contains.
        layer: The actual layer implementation (MarlinAttention, MambaBlock, etc.).
        hidden_size: Model hidden dimension.
    """

    def __init__(
        self,
        layer_config: HybridLayerConfig,
        hidden_size: int,
        intermediate_size: int | None = None,
        shared_attention: Any | None = None,  # For Zamba-style weight sharing
        layer_idx: int = 0,
        num_layers: int | None = None,
    ):
        """Initialize hybrid block.

        Args:
            layer_config: Configuration for this layer.
            hidden_size: Model hidden dimension.
            intermediate_size: MLP intermediate size. Default 4 * hidden_size.
            shared_attention: Optional attention layer for SHARED_ATTENTION type.
        """
        require_mlx("HybridBlock")
        super().__init__()

        self.layer_type = layer_config.layer_type
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or hidden_size * 4
        self.config = layer_config
        self.layer_idx = layer_idx
        self.num_layers = num_layers

        # Build the appropriate layer
        self.layer = self._build_layer(layer_config, shared_attention)

        # Determine state type
        if self.layer_type == HybridLayerType.ATTENTION:
            self.state_type = StateType.KV_CACHE
        elif self.layer_type in (HybridLayerType.MAMBA, HybridLayerType.MAMBA2):
            self.state_type = StateType.HYBRID
        elif self.layer_type == HybridLayerType.SHARED_ATTENTION:
            self.state_type = StateType.KV_CACHE
        elif self.layer_type == HybridLayerType.LINEAR_ATTENTION:
            self.state_type = StateType.SSM_STATE
        else:
            self.state_type = StateType.NONE

    def _build_layer(
        self,
        config: HybridLayerConfig,
        shared_attention: Any | None = None,
    ) -> Any:
        """Build the layer implementation based on type."""
        if config.layer_type == HybridLayerType.ATTENTION:
            return self._build_attention(config)

        elif config.layer_type == HybridLayerType.SHARED_ATTENTION:
            if shared_attention is not None:
                return self._build_shared_attention(config, shared_attention)
            return self._build_attention(config)

        elif config.layer_type in (HybridLayerType.MAMBA, HybridLayerType.MAMBA2):
            return self._build_mamba(config)

        elif config.layer_type == HybridLayerType.MLP_ONLY:
            return self._build_mlp_only(config)

        elif config.layer_type == HybridLayerType.MOE:
            return self._build_moe(config)

        elif config.layer_type == HybridLayerType.HYENA:
            # Hyena not implemented - fall back to attention with warning
            import warnings
            warnings.warn(
                "Hyena layers not yet implemented, using attention fallback. "
                "This will not have the same compute characteristics.",
                UserWarning,
            )
            return self._build_attention(config)

        elif config.layer_type == HybridLayerType.LINEAR_ATTENTION:
            return self._build_linear_attention(config)

        else:
            raise ValueError(f"Unknown layer type: {config.layer_type}")

    def _build_attention(self, config: HybridLayerConfig) -> Any:
        """Build attention-based transformer block."""
        from ..transformer import MarlinTransformerBlock

        attn_cfg = config.attention_config or AttentionLayerConfig()
        quant_cfg = config.quant_config or LayerQuantConfig(Precision.FP4_E2M1, 64)

        quant_type = _precision_to_str(quant_cfg.precision)

        return MarlinTransformerBlock(
            hidden_size=self.hidden_size,
            num_heads=attn_cfg.num_heads,
            intermediate_size=config.mlp_intermediate_size or self.intermediate_size,
            num_kv_heads=attn_cfg.num_kv_heads,
            quant_type=quant_type,
            group_size=quant_cfg.group_size,
            rms_norm_eps=config.norm_eps,
            rope_theta=attn_cfg.rope_theta,
            max_position_embeddings=attn_cfg.max_position_embeddings,
        )

    def _build_shared_attention(
        self,
        config: HybridLayerConfig,
        shared_layer: Any,
    ) -> Any:
        """Build shared attention block (Zamba-style).

        For shared attention, we wrap the shared layer to use its weights
        but maintain separate state.
        """
        return SharedAttentionWrapper(
            shared_layer=shared_layer,
            norm_eps=config.norm_eps,
            hidden_size=self.hidden_size,
            intermediate_size=config.mlp_intermediate_size or self.intermediate_size,
            quant_config=config.quant_config,
        )

    def _build_mamba(self, config: HybridLayerConfig) -> Any:
        """Build Mamba block."""
        from .mamba import MarlinMambaBlock

        mamba_cfg = config.mamba_config or MambaLayerConfig()
        quant_cfg = config.quant_config or LayerQuantConfig(Precision.FP4_E2M1, 128)

        return MarlinMambaBlock(
            hidden_size=self.hidden_size,
            intermediate_size=config.mlp_intermediate_size or self.intermediate_size,
            d_state=mamba_cfg.d_state,
            d_conv=mamba_cfg.d_conv,
            expand=mamba_cfg.expand,
            quant_config=quant_cfg,
            rms_norm_eps=config.norm_eps,
            use_gated_mlp=config.use_gated_mlp,
            mlp_activation=config.mlp_activation,
        )

    def _build_mlp_only(self, config: HybridLayerConfig) -> Any:
        """Build MLP-only block (no attention/SSM)."""

        quant_cfg = config.quant_config or LayerQuantConfig(Precision.FP4_E2M1, 128)
        quant_type = _precision_to_str(quant_cfg.precision)

        return MLPOnlyBlock(
            hidden_size=self.hidden_size,
            intermediate_size=config.mlp_intermediate_size or self.intermediate_size,
            quant_type=quant_type,
            group_size=quant_cfg.group_size,
            norm_eps=config.norm_eps,
            gated=config.use_gated_mlp,
            activation=config.mlp_activation,
        )

    def _build_linear_attention(self, config: HybridLayerConfig) -> Any:
        """Build RWKV block for linear attention."""
        from .rwkv import RWKVBlock

        attn_cfg = config.attention_config or AttentionLayerConfig()
        quant_cfg = config.quant_config or LayerQuantConfig(Precision.FP4_E2M1, 128)
        quant_type = _precision_to_str(quant_cfg.precision)

        return RWKVBlock(
            hidden_size=self.hidden_size,
            num_heads=attn_cfg.num_heads,
            intermediate_size=config.mlp_intermediate_size or self.intermediate_size,
            quant_type=quant_type,
            group_size=quant_cfg.group_size,
            layer_id=self.layer_idx,
            num_layers=self.num_layers or 24,
            layer_norm_eps=config.norm_eps,
        )

    def _build_moe(self, config: HybridLayerConfig) -> Any:
        """Build MoE block.

        MoE is complex enough that it should use existing infrastructure.
        This is a placeholder for integration with MoE dispatch.
        """
        raise NotImplementedError(
            "MoE blocks should use dedicated MoE infrastructure. "
            "See moe_dispatch.py for token routing."
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        state: LayerState | None = None,
        layer_idx: int = 0,
    ) -> tuple[mx.array, LayerState | None]:
        """Forward pass through the hybrid block.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional mask for attention layers.
            position_ids: Optional position IDs for RoPE.
            state: Optional state from previous timestep.
            layer_idx: Layer index for state addressing.

        Returns:
            (output, new_state) where output is [batch, seq_len, hidden_size]
        """
        if self.layer_type in (HybridLayerType.ATTENTION, HybridLayerType.SHARED_ATTENTION):
            # Attention layers use transformer block interface
            kv_cache = state.kv_cache if state else None
            output = self.layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
            )
            # Transformer block returns just output, state is in kv_cache
            new_state = state  # KV cache is updated in-place
            return output, new_state

        elif self.layer_type in (HybridLayerType.MAMBA, HybridLayerType.MAMBA2):
            # Mamba layers return (output, new_state)
            return self.layer(
                hidden_states,
                position_ids=position_ids,
                state=state,
                attention_mask=attention_mask,
            )

        elif self.layer_type == HybridLayerType.LINEAR_ATTENTION:
            return self.layer(
                hidden_states,
                state=state,
                layer_idx=layer_idx,
            )

        else:
            # Stateless layers
            output = self.layer(hidden_states)
            return output, None

    def init_state(self, batch_size: int, layer_idx: int) -> LayerState | None:
        """Initialize state for this layer."""
        if hasattr(self.layer, "init_state"):
            return self.layer.init_state(batch_size, layer_idx)
        return None


class SharedAttentionWrapper(nn.Module if HAS_MLX else object):
    """Wrapper for shared attention weights (Zamba-style).

    In Zamba, one set of attention weights is shared across multiple layers.
    This wrapper provides the attention computation with its own normalization
    and MLP, but uses shared Q/K/V/O weights.
    """

    def __init__(
        self,
        shared_layer: Any,
        norm_eps: float,
        hidden_size: int,
        intermediate_size: int,
        quant_config: LayerQuantConfig | None = None,
    ):
        require_mlx("SharedAttentionWrapper")
        super().__init__()

        self.shared_attention = shared_layer.self_attn  # Extract attention module
        self.hidden_size = hidden_size

        # Own normalization
        self.input_layernorm = RMSNorm(hidden_size, norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, norm_eps)

        # Own MLP
        from ..mlp import MarlinMLP

        quant_cfg = quant_config or LayerQuantConfig(Precision.FP4_E2M1, 128)
        quant_type = _precision_to_str(quant_cfg.precision)

        self.mlp = MarlinMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_type=quant_type,
            group_size=quant_cfg.group_size,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        kv_cache: Any | None = None,
        layer_idx: int = 0,
    ) -> mx.array:
        """Forward with shared attention weights."""
        # Attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.shared_attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MLPOnlyBlock(nn.Module if HAS_MLX else object):
    """Block with only MLP (no attention or SSM).

    Useful for some hybrid architectures that have dedicated MLP-only layers.
    """

    layer_type = HybridLayerType.MLP_ONLY
    state_type = StateType.NONE

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_type: str = "fp4",
        group_size: int = 128,
        norm_eps: float = 1e-6,
        gated: bool = True,
        activation: str = "silu",
    ):
        require_mlx("MLPOnlyBlock")
        super().__init__()

        from ..mlp import MarlinMLP

        self.hidden_size = hidden_size
        self.norm = RMSNorm(hidden_size, norm_eps)
        self.mlp = MarlinMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_type=quant_type,
            group_size=group_size,
            activation=activation,
            gated=gated,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class RMSNorm(nn.Module if HAS_MLX else object):
    """RMSNorm for hybrid blocks."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        if not HAS_MLX:
            raise RuntimeError("MLX required for RMSNorm")
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        variance = mx.mean(x ** 2, axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance + self.eps)
        return self.weight * x


class HybridModel(nn.Module if HAS_MLX else object):
    """Full hybrid model with heterogeneous layer stack.

    Combines embeddings, hybrid blocks, and output projection into
    a complete language model.
    """

    def __init__(
        self,
        config: HybridArchitectureConfig,
    ):
        """Initialize hybrid model from config.

        Args:
            config: Full architecture configuration.
        """
        require_mlx("HybridModel")
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Build layers
        self.layers = self._build_layers(config)

        # Final norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Output projection
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            from ..layers import MarlinLinear
            quant_type = _precision_to_str(config.embed_quant.precision)
            self.lm_head = MarlinLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                quant_type=quant_type,
                group_size=config.embed_quant.group_size,
            )

    def _build_layers(
        self, config: HybridArchitectureConfig
    ) -> list[HybridBlock]:
        """Build the layer stack respecting weight sharing."""
        layers: list[HybridBlock] = []

        for i, layer_config in enumerate(config.layer_configs or []):
            if layer_config.layer_type == HybridLayerType.SHARED_ATTENTION:
                shared_attention = _resolve_shared_attention(
                    layers, layer_config, i
                )
                block = HybridBlock(
                    layer_config,
                    config.hidden_size,
                    intermediate_size=layer_config.mlp_intermediate_size,
                    shared_attention=shared_attention,
                    layer_idx=i,
                    num_layers=config.num_layers,
                )
            else:
                block = HybridBlock(
                    layer_config,
                    config.hidden_size,
                    intermediate_size=layer_config.mlp_intermediate_size,
                    layer_idx=i,
                    num_layers=config.num_layers,
                )
            layers.append(block)

        return layers

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        states: list[LayerState] | None = None,
    ) -> tuple[mx.array, list[LayerState | None]]:
        """Forward pass through the model.

        Args:
            input_ids: [batch, seq_len] input token IDs.
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs.
            states: Optional list of layer states for generation.

        Returns:
            (logits, new_states) where logits is [batch, seq_len, vocab_size]
        """
        # Embed
        hidden_states = self.embed_tokens(input_ids)

        # Process through layers
        new_states = []
        for i, layer in enumerate(self.layers):
            state = states[i] if states else None
            hidden_states, new_state = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                state=state,
                layer_idx=i,
            )
            new_states.append(new_state)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # Output projection
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            # Tied embeddings
            logits = hidden_states @ self.embed_tokens.weight.T

        return logits, new_states

    def init_states(self, batch_size: int) -> list[LayerState | None]:
        """Initialize states for all layers."""
        return [
            layer.init_state(batch_size, i)
            for i, layer in enumerate(self.layers)
        ]

    def get_layer_types(self) -> list[HybridLayerType]:
        """Get list of layer types in order."""
        return [layer.layer_type for layer in self.layers]


class HybridModelBuilder:
    """Factory for building hybrid models from configs.

    Supports:
    - Building from HybridArchitectureConfig
    - Loading from HuggingFace model directories
    - Converting pretrained weights
    """

    @staticmethod
    def from_config(config: HybridArchitectureConfig) -> HybridModel:
        """Build model from config.

        Args:
            config: Architecture configuration.

        Returns:
            Initialized HybridModel (weights are random).
        """
        return HybridModel(config)

    @staticmethod
    def from_dict(config_dict: dict[str, Any]) -> HybridModel:
        """Build model from a simple config dict.

        Expected format:
            {
              "layer_types": ["attention", "mamba", ...],
              "attention_config": {...},
              "mamba_config": {...},
              "quantization": {
                 "attention": {...},
                 "mamba": {...},
                 "mlp": {...},
                 "embed": {...},
                 "norm": {...},
              }
            }
        """
        config = build_hybrid_config(config_dict)
        return HybridModel(config)

    @staticmethod
    def from_pretrained(
        model_path: str | Path,
        quant_type: str = "fp4",
        group_size: int = 128,
    ) -> HybridModel:
        """Load model from pretrained directory.

        Args:
            model_path: Path to model directory.
            quant_type: Quantization type for weights.
            group_size: Quantization group size.

        Returns:
            HybridModel with loaded weights.
        """
        # Load config
        config = HybridArchitectureConfig.from_pretrained(model_path)

        # Override quantization settings
        quant_precision = _str_to_precision(quant_type)
        config.attention_quant = LayerQuantConfig(quant_precision, group_size)
        config.mamba_quant = LayerQuantConfig(quant_precision, group_size)
        config.mlp_quant = LayerQuantConfig(quant_precision, group_size)

        # Build model
        model = HybridModel(config)

        # Load weights
        model = _load_weights(model, Path(model_path))

        return model

    @staticmethod
    def estimate_memory(
        config: HybridArchitectureConfig,
        batch_size: int = 1,
        max_seq_len: int = 2048,
    ) -> dict[str, int]:
        """Estimate memory requirements for a hybrid model.

        Args:
            config: Architecture configuration.
            batch_size: Batch size for inference.
            max_seq_len: Maximum sequence length.

        Returns:
            Dict with memory estimates in bytes.
        """
        from .base import estimate_hybrid_state_memory

        # State memory
        state_mem = estimate_hybrid_state_memory(
            layer_types=config.layer_types,
            hidden_size=config.hidden_size,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )

        # Weight memory (rough estimate)
        weight_bytes = 0
        for layer_config in config.layer_configs or []:
            quant_cfg = layer_config.quant_config or LayerQuantConfig(Precision.FP4_E2M1, 128)
            bits = _precision_to_bits(quant_cfg.precision)

            if layer_config.layer_type == HybridLayerType.ATTENTION:
                # Q, K, V, O projections
                attn_params = 4 * config.hidden_size * config.hidden_size
                weight_bytes += attn_params * bits // 8
            elif layer_config.layer_type in (HybridLayerType.MAMBA, HybridLayerType.MAMBA2):
                mamba_cfg = layer_config.mamba_config or MambaLayerConfig()
                d_inner = config.hidden_size * mamba_cfg.expand
                # in_proj, x_proj, dt_proj, out_proj + conv
                mamba_params = (
                    config.hidden_size * 2 * d_inner  # in_proj
                    + d_inner * (mamba_cfg.d_state * 2 + 16)  # x_proj (approx)
                    + 16 * d_inner  # dt_proj
                    + d_inner * config.hidden_size  # out_proj
                    + d_inner * mamba_cfg.d_conv  # conv
                )
                weight_bytes += mamba_params * bits // 8

            # MLP for all layer types
            intermediate = layer_config.mlp_intermediate_size or config.hidden_size * 4
            mlp_params = 3 * config.hidden_size * intermediate  # gate, up, down
            weight_bytes += mlp_params * bits // 8

        # Embeddings
        embed_bits = _precision_to_bits(config.embed_quant.precision)
        embed_bytes = config.vocab_size * config.hidden_size * embed_bits // 8
        if not config.tie_word_embeddings:
            embed_bytes *= 2  # lm_head separate

        return {
            "weight_bytes": weight_bytes,
            "embedding_bytes": embed_bytes,
            "state_bytes": state_mem["total_bytes"],
            "total_bytes": weight_bytes + embed_bytes + state_mem["total_bytes"],
            **state_mem,
        }


def build_hybrid_config(config_dict: dict[str, Any]) -> HybridArchitectureConfig:
    """Build HybridArchitectureConfig from a simple config dict."""
    layer_types = config_dict.get("layer_types") or []
    if not layer_types:
        raise ValueError("Hybrid config requires non-empty 'layer_types'.")

    num_layers = config_dict.get("num_layers", len(layer_types))
    if num_layers != len(layer_types):
        raise ValueError(
            f"num_layers ({num_layers}) must match len(layer_types) ({len(layer_types)})."
        )

    attention_config = AttentionLayerConfig(
        **config_dict.get("attention_config", {})
    )
    mamba_config = MambaLayerConfig(**config_dict.get("mamba_config", {}))

    defaults = HybridArchitectureConfig()
    quantization = config_dict.get("quantization", {})

    attention_quant = _parse_quant_config(
        config_dict.get("attention_quant") or quantization.get("attention"),
        defaults.attention_quant,
    )
    mamba_quant = _parse_quant_config(
        config_dict.get("mamba_quant") or quantization.get("mamba"),
        defaults.mamba_quant,
    )
    mlp_quant = _parse_quant_config(
        config_dict.get("mlp_quant") or quantization.get("mlp"),
        defaults.mlp_quant,
    )
    embed_quant = _parse_quant_config(
        config_dict.get("embed_quant") or quantization.get("embed"),
        defaults.embed_quant,
    )
    norm_quant = _parse_quant_config(
        config_dict.get("norm_quant") or quantization.get("norm"),
        defaults.norm_quant,
    )

    architecture_name = config_dict.get("architecture_name") or _detect_architecture_from_layer_types(
        layer_types
    )

    return HybridArchitectureConfig(
        hidden_size=config_dict.get("hidden_size", defaults.hidden_size),
        num_layers=num_layers,
        vocab_size=config_dict.get("vocab_size", defaults.vocab_size),
        layer_types=layer_types,
        attention_config=attention_config,
        mamba_config=mamba_config,
        attention_quant=attention_quant,
        mamba_quant=mamba_quant,
        mlp_quant=mlp_quant,
        embed_quant=embed_quant,
        norm_quant=norm_quant,
        rms_norm_eps=config_dict.get("rms_norm_eps", defaults.rms_norm_eps),
        tie_word_embeddings=config_dict.get(
            "tie_word_embeddings", defaults.tie_word_embeddings
        ),
        use_cache=config_dict.get("use_cache", defaults.use_cache),
        architecture_name=architecture_name,
    )


def _resolve_shared_attention(
    layers: list[HybridBlock],
    layer_config: HybridLayerConfig,
    layer_idx: int,
) -> Any | None:
    """Resolve which attention layer to share weights from."""
    if layer_config.shared_layer_idx is not None:
        if layer_config.shared_layer_idx >= len(layers):
            raise ValueError(
                f"Shared attention index {layer_config.shared_layer_idx} "
                f"is not available at layer {layer_idx}."
            )
        return layers[layer_config.shared_layer_idx].layer

    for prior in layers:
        if prior.layer_type == HybridLayerType.ATTENTION:
            return prior.layer

    if layers:
        import warnings

        warnings.warn(
            "Shared attention requested but no attention layer found; "
            "falling back to non-shared attention.",
            UserWarning,
        )
    return None


def _detect_architecture_from_layer_types(
    layer_types: list[str | HybridLayerType],
) -> str:
    """Infer architecture name from layer type list."""
    normalized = []
    for lt in layer_types:
        if isinstance(lt, HybridLayerType):
            normalized.append(lt)
        else:
            normalized.append(str(lt).lower().strip())

    if any(
        lt == HybridLayerType.SHARED_ATTENTION or lt == "shared_attention"
        for lt in normalized
    ):
        return "zamba"
    if any(lt == HybridLayerType.HYENA or lt == "hyena" for lt in normalized):
        return "stripedhyena"

    has_attention = any(
        lt == HybridLayerType.ATTENTION or lt == "attention" for lt in normalized
    )
    has_mamba = any(
        lt in (HybridLayerType.MAMBA, HybridLayerType.MAMBA2)
        or lt in ("mamba", "mamba2", "ssm", "ssd")
        for lt in normalized
    )

    if has_attention and has_mamba:
        return "jamba"
    if has_mamba:
        return "mamba"
    if has_attention:
        return "dense"
    return "hybrid"


def _parse_quant_config(
    raw: LayerQuantConfig | dict[str, Any] | str | None,
    default: LayerQuantConfig,
) -> LayerQuantConfig:
    """Parse a quantization config dictionary or shorthand."""
    if raw is None:
        return default
    if isinstance(raw, LayerQuantConfig):
        return raw
    if isinstance(raw, str):
        return LayerQuantConfig(_str_to_precision(raw), default.group_size, default.symmetric)
    if isinstance(raw, dict):
        precision = raw.get("precision", default.precision)
        if isinstance(precision, str):
            precision = _str_to_precision(precision)
        return LayerQuantConfig(
            precision=precision,
            group_size=raw.get("group_size", default.group_size),
            symmetric=raw.get("symmetric", default.symmetric),
        )
    raise TypeError(f"Unsupported quant config type: {type(raw)}")


def _precision_to_str(precision: Precision) -> str:
    """Convert Precision enum to string for MarlinLinear."""
    mapping = {
        Precision.FP4_E2M1: "fp4",
        Precision.INT4: "int4",
        Precision.FP8_E4M3: "fp8",
        Precision.INT8: "int8",
        Precision.FP16: "fp16",
        Precision.BF16: "bf16",
    }
    return mapping.get(precision, "fp4")


def _str_to_precision(quant_type: str) -> Precision:
    """Convert string to Precision enum."""
    mapping = {
        "fp4": Precision.FP4_E2M1,
        "int4": Precision.INT4,
        "fp8": Precision.FP8_E4M3,
        "int8": Precision.INT8,
        "fp16": Precision.FP16,
        "bf16": Precision.BF16,
    }
    return mapping.get(quant_type.lower(), Precision.FP4_E2M1)


def _precision_to_bits(precision: Precision) -> int:
    """Get bit width for precision."""
    mapping = {
        Precision.FP16: 16,
        Precision.BF16: 16,
        Precision.FP8_E4M3: 8,
        Precision.INT8: 8,
        Precision.FP4_E2M1: 4,
        Precision.INT4: 4,
        Precision.INT3: 3,
        Precision.INT2: 2,
        Precision.NF3: 3,
        Precision.NF2: 2,
    }
    return mapping.get(precision, 4)


def _load_weights(model: HybridModel, model_path: Path) -> HybridModel:
    """Load weights from safetensors files.

    This is a placeholder - actual implementation would map HF weights
    to the hybrid model structure.
    """
    import glob

    safetensor_files = glob.glob(str(model_path / "*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    # TODO: Implement weight loading with proper name mapping
    # This requires understanding the source model's weight naming convention
    # and mapping to our hybrid model structure

    import warnings
    warnings.warn(
        "Weight loading not fully implemented. "
        "Model initialized with random weights.",
        UserWarning,
    )

    return model
