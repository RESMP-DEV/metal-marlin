from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)

# Tokenizer ID for GLM-4.7 models
GLM4_TOKENIZER_ID = "zai-org/GLM-4.7-Flash"


def _resolve_full_attention_interval(config: object) -> int | None:
    """Extract ``full_attention_interval`` from a HF config-like object.

    Qwen3NextConfig may store this in ``__dict__`` / ``to_dict()`` rather
    than as a plain attribute.  This helper checks both paths.
    """
    # Direct attribute
    logger.debug("_resolve_full_attention_interval called with config=%s", config)
    val = getattr(config, "full_attention_interval", None)
    if val is not None:
        try:
            return int(val)
        except (TypeError, ValueError):
            pass

    # Fallback: raw config dict
    as_dict: dict = {}
    if hasattr(config, "to_dict"):
        try:
            as_dict = config.to_dict()  # type: ignore[union-attr]
        except Exception:
            pass
    elif hasattr(config, "__dict__"):
        as_dict = config.__dict__  # type: ignore[attr-defined]

    val = as_dict.get("full_attention_interval")
    if val is not None:
        try:
            return int(val)
        except (TypeError, ValueError):
            pass

    return None


def _resolve_full_attention_interval_from_dict(data: dict) -> int | None:
    """Extract ``full_attention_interval`` from a config.json dict.

    Handles the case where this field may be nested inside ``text_config``
    or stored at the top level.
    """
    # Check top-level first
    logger.debug("_resolve_full_attention_interval_from_dict called with data=%s", data)
    val = data.get("full_attention_interval")
    if val is not None:
        try:
            return int(val)
        except (TypeError, ValueError):
            pass

    # Check nested text_config
    text_cfg = data.get("text_config")
    if isinstance(text_cfg, dict):
        val = text_cfg.get("full_attention_interval")
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass

    return None


@dataclass
class TrellisModelConfig:
    """Configuration for trellis-quantized model.

    This config supports multiple model architectures. Defaults are generic
    placeholders - use from_pretrained() to load model-specific values.

    Supported models:
    - GLM-4.7-Flash: MLA attention, 64 experts, first_moe_layer=1
    - Qwen3-30B-A3B: GQA attention, 128 experts, first_moe_layer=0 (all MoE)
    - DeepSeek-V3: MLA attention, 256 experts, first_moe_layer varies
    - Qwen3.5/3.6 DeltaNet hybrids: interleaved linear + full attention,
      optional MoE with shared experts
    - Qwen VL / VL-MoE multimodal: nested text_config support

    Qwen Hybrid DeltaNet Support:
    - layer_types: Per-layer attention type (linear_attention/full_attention)
    - full_attention_interval: Frequency of full-attention layers in hybrids
    - linear_* fields: DeltaNet linear attention dimensions
    - shared_expert_intermediate_size: Shared expert FFN size for hybrid MoE
    - Nested text_config handling for multimodal Qwen VL models
    """

    # Model architecture (generic defaults, override via from_pretrained)
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_kv_heads: int = 8  # GQA default; MLA models may differ
    head_dim: int = 128
    intermediate_size: int = 14336  # Dense MLP hidden size
    vocab_size: int = 151936

    # MLA (Multi-head Latent Attention) - set to None for standard GQA
    # When set, enables compressed KV cache via low-rank projections
    kv_lora_rank: int | None = None  # None = standard GQA, 512 = GLM MLA
    q_lora_rank: int | None = None  # None = no Q compression, 768 = GLM MLA
    # KV head dim if different from Q (GLM: 1120)
    kv_head_dim: int | None = None
    kv_rope_dim: int = 0  # Position-dependent component in kv_a_proj (GLM: 64)

    # GLM-4 MLA specific dimensions (used when kv_lora_rank is set)
    qk_nope_head_dim: int = 192  # Non-positional Q/K dimension
    qk_rope_head_dim: int = 64  # Rotary Q/K dimension (same as kv_rope_dim)
    v_head_dim: int = 256  # Value head dimension

    # MoE configuration
    num_experts: int = 1  # 1 = dense model (no MoE)
    num_shared_experts: int = 0  # Shared expert count (GLM has 1)
    num_experts_per_tok: int = 1  # Top-k experts per token
    moe_intermediate_size: int | None = None  # Per-expert hidden size
    shared_expert_intermediate_size: int | None = None  # Qwen hybrid MoE
    first_moe_layer: int = 0  # First layer using MoE (0 = all layers)

    # Qwen hybrid DeltaNet layer configuration
    # Per-layer attention type list, e.g.
    # ["linear_attention", "full_attention", "linear_attention", ...]
    layer_types: list[str] | None = None
    # How often a full-attention layer appears in DeltaNet hybrids
    full_attention_interval: int | None = None

    # DeltaNet linear-attention dimensions (Qwen3.5/3.6 hybrids)
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_conv_kernel_dim: int | None = None

    # RoPE configuration
    rope_theta: float = 10000.0
    max_position_embeddings: int = 32768
    rope_scaling: dict | None = None  # YaRN, NTK, etc.

    # Normalization
    rms_norm_eps: float = 1e-6

    # Quantization info (set during quantization)
    quantization_bits: int = 4  # Average bits per weight

    # Layer pruning - layers to skip during inference
    # Set via prune_layers() or from_importance_analysis()
    skip_layers: list[int] | None = None

    # Optimizations
    use_mixed_bpw_optimizations: bool = True  # Enable MixedBPWMoEDispatcher logic
    # Enable automatic kernel tuning on first run
    enable_kernel_autotune: bool = True

    @classmethod
    def from_pretrained(cls, model_path: str) -> "TrellisModelConfig":
        """Load config from model directory or HuggingFace.

        Reads config.json and maps architecture-specific field names to
        our unified config format.
        """
        logger.debug("from_pretrained called with model_path=%s", model_path)
        import json
        from pathlib import Path

        path = Path(model_path)
        config_path = path / "config.json"

        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)
            return cls._from_dict(data)

        # Try loading from HuggingFace
        try:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(model_path)
            return cls._from_hf_config(hf_config)
        except Exception:
            raise ValueError(
                f"Could not load config from {model_path}. "
                "Provide config.json or a valid HuggingFace model ID."
            )

    @classmethod
    def _from_dict(cls, data: dict) -> "TrellisModelConfig":
        """Create config from a dictionary (config.json contents).

        Handles nested ``text_config`` dicts (Qwen VL / VL-MoE style
        multimodal configs) by merging text-level fields into the top-level
        mapping before constructing the dataclass.
        """
        # If a nested text_config dict is present, merge its fields so
        # architecture-specific names are resolved uniformly.
        logger.debug("_from_dict called with data=%s", data)
        if "text_config" in data and isinstance(data["text_config"], dict):
            data = {**data, **data["text_config"]}

        # Direct field mapping for fields we recognize
        kwargs = {k: v for k, v in data.items() if hasattr(cls, k)}

        # Handle architecture-specific field name variations
        field_mappings = {
            # Qwen/Llama style
            "num_key_value_heads": "num_kv_heads",
            "num_local_experts": "num_experts",
            "num_experts_per_tok": "num_experts_per_tok",
            "moe_intermediate_size": "moe_intermediate_size",
            # DeepSeek style
            "n_routed_experts": "num_experts",
            "num_experts_per_token": "num_experts_per_tok",
            "moe_layer_freq": None,  # Handle specially
            "first_k_dense_replace": "first_moe_layer",
            # GLM style MLA
            "kv_lora_rank": "kv_lora_rank",
            "q_lora_rank": "q_lora_rank",
            "qk_nope_head_dim": "qk_nope_head_dim",
            "qk_rope_head_dim": "qk_rope_head_dim",
            "v_head_dim": "v_head_dim",
            # Qwen hybrid MoE shared expert
            "shared_expert_intermediate_size": "shared_expert_intermediate_size",
            "num_shared_experts": "num_shared_experts",
            # Qwen hybrid DeltaNet layer configuration
            "layer_types": "layer_types",
            "full_attention_interval": "full_attention_interval",
            # DeltaNet linear-attention dimensions
            "linear_key_head_dim": "linear_key_head_dim",
            "linear_value_head_dim": "linear_value_head_dim",
            "linear_num_key_heads": "linear_num_key_heads",
            "linear_num_value_heads": "linear_num_value_heads",
            "linear_conv_kernel_dim": "linear_conv_kernel_dim",
        }

        for src_key, dst_key in field_mappings.items():
            if src_key in data and dst_key and dst_key not in kwargs:
                kwargs[dst_key] = data[src_key]

        # Handle full_attention_interval with fallback for nested dicts
        if "full_attention_interval" not in kwargs:
            fai = _resolve_full_attention_interval_from_dict(data)
            if fai is not None:
                kwargs["full_attention_interval"] = fai
        elif isinstance(kwargs["full_attention_interval"], str):
            # Coerce string to int if possible
            try:
                kwargs["full_attention_interval"] = int(kwargs["full_attention_interval"])
            except (TypeError, ValueError):
                kwargs["full_attention_interval"] = None

        # Validate layer_types is a list if present
        if "layer_types" in kwargs and not isinstance(kwargs["layer_types"], list):
            kwargs["layer_types"] = None

        # Compute head_dim if not provided
        if "head_dim" not in kwargs and "hidden_size" in kwargs and "num_attention_heads" in kwargs:
            kwargs["head_dim"] = kwargs["hidden_size"] // kwargs["num_attention_heads"]

        # Detect MoE vs dense
        if kwargs.get("num_experts", 1) <= 1:
            kwargs["num_experts"] = 1
            kwargs["first_moe_layer"] = kwargs.get(
                "num_hidden_layers", 32)  # No MoE layers

        return cls(**kwargs)

    @classmethod
    def _from_hf_config(cls, hf_config) -> "TrellisModelConfig":
        """Create config from HuggingFace AutoConfig.

        Handles nested ``text_config`` (multimodal Qwen VL models) by
        resolving the effective text config first, then extracting fields.
        """
        logger.debug("_from_hf_config called with hf_config=%s", hf_config)
        kwargs = {}

        # Resolve effective text config for multimodal wrappers
        text_cfg = hf_config
        if hasattr(hf_config, "text_config"):
            nested = getattr(hf_config, "text_config", None)
            if nested is not None and hasattr(nested, "vocab_size"):
                text_cfg = nested

        # Standard LLM fields (read from effective text config)
        for attr in [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "vocab_size",
            "intermediate_size",
            "max_position_embeddings",
            "rms_norm_eps",
            "rope_theta",
        ]:
            if hasattr(text_cfg, attr):
                kwargs[attr] = getattr(text_cfg, attr)

        # KV heads (various names)
        for attr in ["num_key_value_heads", "num_kv_heads"]:
            if hasattr(text_cfg, attr):
                kwargs["num_kv_heads"] = getattr(text_cfg, attr)
                break

        # MoE fields
        for attr in ["num_local_experts", "n_routed_experts", "num_experts"]:
            if hasattr(text_cfg, attr):
                kwargs["num_experts"] = getattr(text_cfg, attr)
                break

        if hasattr(text_cfg, "num_experts_per_tok"):
            kwargs["num_experts_per_tok"] = text_cfg.num_experts_per_tok

        if hasattr(text_cfg, "num_shared_experts"):
            kwargs["num_shared_experts"] = text_cfg.num_shared_experts

        if hasattr(text_cfg, "moe_intermediate_size"):
            kwargs["moe_intermediate_size"] = text_cfg.moe_intermediate_size

        if hasattr(text_cfg, "shared_expert_intermediate_size"):
            kwargs["shared_expert_intermediate_size"] = (
                text_cfg.shared_expert_intermediate_size
            )

        # MLA fields (GLM/DeepSeek)
        if hasattr(text_cfg, "kv_lora_rank"):
            kwargs["kv_lora_rank"] = text_cfg.kv_lora_rank
        if hasattr(text_cfg, "q_lora_rank"):
            kwargs["q_lora_rank"] = text_cfg.q_lora_rank

        # GLM-4 MLA specific dimensions
        if hasattr(text_cfg, "qk_nope_head_dim"):
            kwargs["qk_nope_head_dim"] = text_cfg.qk_nope_head_dim
        if hasattr(text_cfg, "qk_rope_head_dim"):
            kwargs["qk_rope_head_dim"] = text_cfg.qk_rope_head_dim
        if hasattr(text_cfg, "v_head_dim"):
            kwargs["v_head_dim"] = text_cfg.v_head_dim

        # Qwen hybrid DeltaNet fields
        if hasattr(text_cfg, "layer_types") and isinstance(
            text_cfg.layer_types, list
        ):
            kwargs["layer_types"] = text_cfg.layer_types

        # full_attention_interval may live in kwargs dict rather than attr
        fai = _resolve_full_attention_interval(text_cfg)
        if fai is not None:
            kwargs["full_attention_interval"] = fai

        # DeltaNet linear-attention dimensions
        for attr in (
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_conv_kernel_dim",
        ):
            if hasattr(text_cfg, attr):
                kwargs[attr] = getattr(text_cfg, attr)

        # RoPE scaling
        if hasattr(text_cfg, "rope_scaling") and text_cfg.rope_scaling:
            kwargs["rope_scaling"] = text_cfg.rope_scaling

        if hasattr(hf_config, "use_mixed_bpw_optimizations"):
            kwargs["use_mixed_bpw_optimizations"] = hf_config.use_mixed_bpw_optimizations

        if hasattr(hf_config, "enable_kernel_autotune"):
            kwargs["enable_kernel_autotune"] = hf_config.enable_kernel_autotune

        return cls(**kwargs)

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Check if layer uses MoE (vs dense MLP)."""
        logger.debug("is_moe_layer called with layer_idx=%s", layer_idx)
        if self.num_experts <= 1:
            return False
        return layer_idx >= self.first_moe_layer

    def is_mla_model(self) -> bool:
        """Check if model uses Multi-head Latent Attention."""
        logger.debug("is_mla_model called")
        return self.kv_lora_rank is not None

    def should_skip_layer(self, layer_idx: int) -> bool:
        """Check if layer should be skipped during inference."""
        logger.debug("should_skip_layer called with layer_idx=%s", layer_idx)
        if self.skip_layers is None:
            return False
        return layer_idx in self.skip_layers

    def prune_layers(self, layers_to_skip: list[int]) -> "TrellisModelConfig":
        """Create a new config with specified layers marked for skipping.

        Args:
            layers_to_skip: Layer indices to skip during inference.

        Returns:
            New config with skip_layers set.
        """
        logger.debug("prune_layers called with layers_to_skip=%s", layers_to_skip)
        import copy

        new_config = copy.copy(self)
        new_config.skip_layers = sorted(layers_to_skip)
        return new_config

    @classmethod
    def from_importance_analysis(
        cls,
        base_config: "TrellisModelConfig",
        analysis_path: str,
        threshold_pct: float | None = None,
    ) -> "TrellisModelConfig":
        """Create config with pruned layers from importance analysis.

        Args:
            base_config: Base model configuration.
            analysis_path: Path to layer importance analysis JSON.
            threshold_pct: Override threshold (uses analysis threshold if None).

        Returns:
            Config with skip_layers set based on importance analysis.
        """
        logger.debug("from_importance_analysis called with base_config=%s, analysis_path=%s, threshold_pct=%s", base_config, analysis_path, threshold_pct)
        import json
        from pathlib import Path

        with open(Path(analysis_path)) as f:
            analysis = json.load(f)

        if threshold_pct is None:
            threshold_pct = analysis.get("threshold_pct", 0.5)

        # Find prunable layers
        skip_layers = []
        for layer_data in analysis.get("layers", []):
            if layer_data.get("perplexity_delta_pct", float("inf")) < threshold_pct:
                skip_layers.append(layer_data["layer_idx"])

        return base_config.prune_layers(skip_layers)
