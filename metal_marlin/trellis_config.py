from dataclasses import dataclass


@dataclass
class TrellisModelConfig:
    """Configuration for trellis-quantized model.

    This config supports multiple model architectures. Defaults are generic
    placeholders - use from_pretrained() to load model-specific values.

    Supported models:
    - GLM-4.7-Flash: MLA attention, 64 experts, first_moe_layer=1
    - Qwen3-30B-A3B: GQA attention, 128 experts, first_moe_layer=0 (all MoE)
    - DeepSeek-V3: MLA attention, 256 experts, first_moe_layer varies
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
    kv_head_dim: int | None = None  # KV head dim if different from Q (GLM: 1120)
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
    first_moe_layer: int = 0  # First layer using MoE (0 = all layers)

    # RoPE configuration
    rope_theta: float = 10000.0
    max_position_embeddings: int = 32768
    rope_scaling: dict | None = None  # YaRN, NTK, etc.

    # Normalization
    rms_norm_eps: float = 1e-6

    # Quantization info (set during quantization)
    quantization_bits: int = 4  # Average bits per weight

    @classmethod
    def from_pretrained(cls, model_path: str) -> "TrellisModelConfig":
        """Load config from model directory or HuggingFace.

        Reads config.json and maps architecture-specific field names to
        our unified config format.
        """
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
        """Create config from a dictionary (config.json contents)."""
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
        }

        for src_key, dst_key in field_mappings.items():
            if src_key in data and dst_key and dst_key not in kwargs:
                kwargs[dst_key] = data[src_key]

        # Compute head_dim if not provided
        if "head_dim" not in kwargs and "hidden_size" in kwargs and "num_attention_heads" in kwargs:
            kwargs["head_dim"] = kwargs["hidden_size"] // kwargs["num_attention_heads"]

        # Detect MoE vs dense
        if kwargs.get("num_experts", 1) <= 1:
            kwargs["num_experts"] = 1
            kwargs["first_moe_layer"] = kwargs.get("num_hidden_layers", 32)  # No MoE layers

        return cls(**kwargs)

    @classmethod
    def _from_hf_config(cls, hf_config) -> "TrellisModelConfig":
        """Create config from HuggingFace AutoConfig."""
        kwargs = {}

        # Standard LLM fields
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
            if hasattr(hf_config, attr):
                kwargs[attr] = getattr(hf_config, attr)

        # KV heads (various names)
        for attr in ["num_key_value_heads", "num_kv_heads"]:
            if hasattr(hf_config, attr):
                kwargs["num_kv_heads"] = getattr(hf_config, attr)
                break

        # MoE fields
        for attr in ["num_local_experts", "n_routed_experts", "num_experts"]:
            if hasattr(hf_config, attr):
                kwargs["num_experts"] = getattr(hf_config, attr)
                break

        if hasattr(hf_config, "num_experts_per_tok"):
            kwargs["num_experts_per_tok"] = hf_config.num_experts_per_tok

        # MLA fields (GLM/DeepSeek)
        if hasattr(hf_config, "kv_lora_rank"):
            kwargs["kv_lora_rank"] = hf_config.kv_lora_rank
        if hasattr(hf_config, "q_lora_rank"):
            kwargs["q_lora_rank"] = hf_config.q_lora_rank

        # GLM-4 MLA specific dimensions
        if hasattr(hf_config, "qk_nope_head_dim"):
            kwargs["qk_nope_head_dim"] = hf_config.qk_nope_head_dim
        if hasattr(hf_config, "qk_rope_head_dim"):
            kwargs["qk_rope_head_dim"] = hf_config.qk_rope_head_dim
        if hasattr(hf_config, "v_head_dim"):
            kwargs["v_head_dim"] = hf_config.v_head_dim

        # RoPE scaling
        if hasattr(hf_config, "rope_scaling") and hf_config.rope_scaling:
            kwargs["rope_scaling"] = hf_config.rope_scaling

        return cls(**kwargs)

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Check if layer uses MoE (vs dense MLP)."""
        if self.num_experts <= 1:
            return False
        return layer_idx >= self.first_moe_layer

    def is_mla_model(self) -> bool:
        """Check if model uses Multi-head Latent Attention."""
        return self.kv_lora_rank is not None
