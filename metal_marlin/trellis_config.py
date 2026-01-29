from dataclasses import dataclass, field


@dataclass
class TrellisModelConfig:
    """Configuration for trellis-quantized model."""

    # Model architecture
    hidden_size: int = 2048
    num_hidden_layers: int = 47
    num_attention_heads: int = 32
    num_kv_heads: int = 4
    head_dim: int = 64
    intermediate_size: int = 10240  # Dense layers
    vocab_size: int = 152064

    # MLA (Multi-head Latent Attention)
    kv_lora_rank: int = 512
    q_lora_rank: int | None = None

    # MoE configuration
    num_experts: int = 64
    num_shared_experts: int = 1
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 1536  # Per expert
    first_moe_layer: int = 2  # Layers 0,1 are dense

    # RoPE configuration
    rope_theta: float = 10000.0
    max_position_embeddings: int = 131072
    rope_scaling: dict | None = field(default_factory=lambda: {
        "type": "yarn",
        "factor": 32.0,
        "original_max_position_embeddings": 4096,
    })

    # Normalization
    rms_norm_eps: float = 1e-6

    # Quantization info
    quantization_bits: int = 3  # Average bits per weight

    @classmethod
    def from_pretrained(cls, model_path: str) -> "TrellisModelConfig":
        """Load config from model directory or HuggingFace."""
        import json
        from pathlib import Path

        path = Path(model_path)
        config_path = path / "config.json"

        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)
            return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

        # Try loading from HuggingFace
        try:
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(model_path)
            return cls(
                hidden_size=hf_config.hidden_size,
                num_hidden_layers=hf_config.num_hidden_layers,
                # ... map other fields
            )
        except Exception:
            return cls()  # Use defaults

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Check if layer uses MoE (vs dense MLP)."""
        return layer_idx >= self.first_moe_layer
