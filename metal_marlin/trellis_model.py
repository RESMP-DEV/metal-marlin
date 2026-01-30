"""Complete trellis-quantized model for inference.

Provides a high-level nn.Module interface for loading and running
trellis-quantized models with support for dense and MoE layers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import RMSNorm
from .trellis_attention import TrellisMLAConfig, TrellisMLAttention
from .trellis_config import TrellisModelConfig
from .trellis_kv_cache import TrellisKVCache
from .trellis_layer import TrellisDenseMLP
from .trellis_linear import TrellisLinear

if TYPE_CHECKING:
    from .trellis_loader import TrellisModelLoader


class TrellisMoEMLP(nn.Module):
    """MoE MLP with trellis-quantized weights for MoE layers.

    Implements a mixture of experts with:
    - Router: selects top-k experts per token
    - Multiple experts: each is a dense MLP with SwiGLU
    - Shared expert: always applied (for model stability)

    This is used for layers >= first_moe_layer in GLM-4.7-Flash.

    Attributes:
        router: Linear layer for expert selection.
        experts: List of expert MLPs (TrellisDenseMLP).
        shared_expert: Always-active expert for stability.
        num_experts_per_tok: Number of experts to activate per token.
    """

    def __init__(
        self,
        router: nn.Linear,
        experts: list[TrellisDenseMLP],
        shared_expert: TrellisDenseMLP,
        num_experts_per_tok: int = 8,
    ):
        """Initialize TrellisMoEMLP.

        Args:
            router: Linear layer for expert selection scores.
            experts: List of expert MLPs (each a TrellisDenseMLP).
            shared_expert: Always-active expert MLP.
            num_experts_per_tok: Number of experts to activate per token.
        """
        super().__init__()
        self.router = router
        self.experts = nn.ModuleList(experts)
        self.shared_expert = shared_expert
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with MoE routing.

        Args:
            x: Input tensor [..., hidden_size].

        Returns:
            Output tensor [..., hidden_size].
        """
        # Get router scores
        router_logits = self.router(x)  # [..., num_experts]

        # Select top-k experts
        routing_weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=-1, dtype=torch.float),
            k=self.num_experts_per_tok,
            dim=-1,
        )

        # Normalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Initialize output
        final_hidden_states = torch.zeros_like(x)

        # Process each expert in the top-k
        for i in range(self.num_experts_per_tok):
            expert_idx = selected_experts[..., i]
            expert_weight = routing_weights[..., i]

            # Gather tokens for this expert
            expert_mask = F.one_hot(expert_idx, num_classes=len(self.experts)).float()
            expert_mask = expert_mask * expert_weight.unsqueeze(-1)

            # Apply expert
            for expert_id, expert_module in enumerate(self.experts):
                # Get tokens assigned to this expert
                mask = expert_mask[..., expert_id]
                if mask.sum() > 0:
                    expert_input = x * mask.unsqueeze(-1)
                    expert_output = expert_module(expert_input)
                    final_hidden_states += expert_output * mask.unsqueeze(-1)

        # Add shared expert (always applied)
        shared_output = self.shared_expert(x)
        final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states

    @classmethod
    def from_loader(
        cls,
        loader: TrellisModelLoader,
        config: TrellisModelConfig,
        layer_idx: int,
        router_weights: dict[str, torch.Tensor],
        device: str = "mps",
    ) -> TrellisMoEMLP:
        """Create TrellisMoEMLP from a TrellisModelLoader.

        Args:
            loader: TrellisModelLoader instance for the model.
            config: Model configuration.
            layer_idx: Layer index to load.
            router_weights: Router weights dictionary.
            device: Device to place modules on.

        Returns:
            TrellisMoEMLP module initialized with layer weights.
        """
        layer_weights = loader.load_layer(layer_idx)
        prefix = f"model.layers.{layer_idx}.mlp"

        # Create router
        router = nn.Linear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            device=device,
            dtype=torch.float16,
        )
        router.weight.data = router_weights[f"{prefix}.gate.weight"].to(device)

        # Create experts
        experts = []
        for expert_idx in range(config.num_experts):
            expert_prefix = f"{prefix}.experts.{expert_idx}"
            expert = TrellisDenseMLP(
                gate_proj=TrellisLinear.from_trellis_weight(
                    layer_weights[f"{expert_prefix}.gate_proj.weight"],
                    device=device,
                ),
                up_proj=TrellisLinear.from_trellis_weight(
                    layer_weights[f"{expert_prefix}.up_proj.weight"],
                    device=device,
                ),
                down_proj=TrellisLinear.from_trellis_weight(
                    layer_weights[f"{expert_prefix}.down_proj.weight"],
                    device=device,
                ),
            )
            experts.append(expert)

        # Create shared expert (GLM uses 'shared_experts' plural in weights)
        shared_expert = TrellisDenseMLP(
            gate_proj=TrellisLinear.from_trellis_weight(
                layer_weights[f"{prefix}.shared_experts.gate_proj.weight"],
                device=device,
            ),
            up_proj=TrellisLinear.from_trellis_weight(
                layer_weights[f"{prefix}.shared_experts.up_proj.weight"],
                device=device,
            ),
            down_proj=TrellisLinear.from_trellis_weight(
                layer_weights[f"{prefix}.shared_experts.down_proj.weight"],
                device=device,
            ),
        )

        return cls(
            router=router,
            experts=experts,
            shared_expert=shared_expert,
            num_experts_per_tok=config.num_experts_per_tok,
        )


class TrellisDecoderLayer(nn.Module):
    """Complete transformer decoder layer with trellis-quantized weights.

    Implements a GLM-style decoder layer with:
    - MLA attention (Multi-head Latent Attention)
    - RMSNorm pre-normalization
    - Dense or MoE MLP (depending on layer index)
    - Residual connections

    Attributes:
        self_attn: Attention module (to be implemented).
        mlp: Dense or MoE MLP module.
        input_layernorm: Pre-attention normalization.
        post_attention_layernorm: Post-attention normalization.
        config: Layer configuration.
    """

    def __init__(
        self,
        config: TrellisModelConfig,
        layer_idx: int,
        device: str = "mps",
    ):
        """Initialize TrellisDecoderLayer.

        Args:
            config: Model configuration.
            layer_idx: Layer index (0-indexed).
            device: Device to place modules on.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # MLA attention will be created in from_loader
        self.self_attn = None

        # MLP (dense or MoE)
        self.mlp = None  # Will be set in from_loader

        # Normalization
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.to(device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: TrellisKVCache | None = None,
    ) -> torch.Tensor:
        """Forward pass through the decoder layer.

        Args:
            hidden_states: Input tensor [..., seq_len, hidden_size].
            attention_mask: Causal attention mask.
            position_ids: Position IDs for RoPE.
            kv_cache: KV cache for generation.

        Returns:
            Output tensor [..., seq_len, hidden_size].
        """
        # Pre-attention normalization
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        attn_output = self.self_attn(
            hidden_states,
            kv_cache=kv_cache,
            layer_idx=self.layer_idx,
        )

        # Residual connection
        hidden_states = residual + attn_output

        # Post-attention normalization
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        mlp_output = self.mlp(hidden_states)

        # Residual connection
        hidden_states = residual + mlp_output

        return hidden_states

    @classmethod
    def from_loader(
        cls,
        loader: TrellisModelLoader,
        config: TrellisModelConfig,
        layer_idx: int,
        router_weights: dict[str, torch.Tensor],
        base_weights: dict[str, torch.Tensor],
        device: str = "mps",
    ) -> TrellisDecoderLayer:
        """Create TrellisDecoderLayer from a TrellisModelLoader.

        Args:
            loader: TrellisModelLoader instance.
            config: Model configuration.
            layer_idx: Layer index.
            router_weights: Router weights for MoE layers.
            base_weights: Base model weights.
            device: Device to place modules on.

        Returns:
            TrellisDecoderLayer module initialized with layer weights.
        """
        layer = cls(config, layer_idx, device)

        # Load layer weights
        layer_weights = loader.load_layer(layer_idx)
        prefix = f"model.layers.{layer_idx}.self_attn"

        # Create MLA attention
        mla_config = TrellisMLAConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            kv_lora_rank=config.kv_lora_rank,
            q_lora_rank=config.q_lora_rank,
            kv_rope_dim=getattr(config, "kv_rope_dim", 0),
            kv_head_dim=getattr(config, "kv_head_dim", None),
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
        )

        # Get attention projections
        # GLM uses low-rank Q: q_a_proj + q_b_proj
        q_a_proj = None
        q_b_proj = None
        if mla_config.q_lora_rank:
            q_a_key = f"{prefix}.q_a_proj.weight"
            q_b_key = f"{prefix}.q_b_proj.weight"
            if q_a_key in layer_weights:
                q_a_proj = TrellisLinear.from_trellis_weight(
                    layer_weights[q_a_key],
                    device=device,
                )
            if q_b_key in layer_weights:
                q_b_proj = TrellisLinear.from_trellis_weight(
                    layer_weights[q_b_key],
                    device=device,
                )

        # GLM uses kv_a_proj_with_mqa (includes MQA heads)
        kv_a_key = f"{prefix}.kv_a_proj_with_mqa.weight"
        if kv_a_key not in layer_weights:
            kv_a_key = f"{prefix}.kv_a_proj.weight"  # Fallback
        kv_b_key = f"{prefix}.kv_b_proj.weight"
        o_key = f"{prefix}.o_proj.weight"

        kv_a_proj = TrellisLinear.from_trellis_weight(
            layer_weights[kv_a_key],
            device=device,
        )
        kv_b_proj = TrellisLinear.from_trellis_weight(
            layer_weights[kv_b_key],
            device=device,
        )
        o_proj = TrellisLinear.from_trellis_weight(
            layer_weights[o_key],
            device=device,
        )

        layer.self_attn = TrellisMLAttention(
            config=mla_config,
            q_a_proj=q_a_proj,
            q_b_proj=q_b_proj,
            kv_a_proj=kv_a_proj,
            kv_b_proj=kv_b_proj,
            o_proj=o_proj,
        )

        # Create MLP (dense or MoE)
        if config.is_moe_layer(layer_idx):
            layer.mlp = TrellisMoEMLP.from_loader(loader, config, layer_idx, router_weights, device)
        else:
            layer.mlp = TrellisDenseMLP.from_loader(loader, layer_idx, device)

        return layer


class TrellisModel(nn.Module):
    """Complete trellis-quantized model for inference."""

    def __init__(self, config: TrellisModelConfig):
        super().__init__()
        self.config = config

        # Embedding (not quantized, from base model)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Layers
        self.layers = nn.ModuleList()

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: TrellisKVCache | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        # Create causal mask
        if attention_mask is None:
            seq_len = input_ids.shape[1]
            attention_mask = self._make_causal_mask(seq_len, hidden_states.device)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        return self.norm(hidden_states)

    def _make_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "mps",
        load_in_layers: bool = True,
    ) -> TrellisModel:
        """Load model from trellis-quantized checkpoint.

        Args:
            model_path: Path to quantized model directory
            device: Device to load model on
            load_in_layers: If True, load one layer at a time (memory efficient)
        """

        config = TrellisModelConfig.from_pretrained(model_path)
        model = cls(config)

        from .trellis_loader import TrellisModelLoader

        loader = TrellisModelLoader(model_path)

        # Load non-quantized weights (embedding, norms, lm_head)
        base_weights = cls._load_base_weights(model_path)
        model.embed_tokens.weight.data = base_weights["model.embed_tokens.weight"].to(device)
        model.norm.weight.data = base_weights["model.norm.weight"].to(device)

        # Load router weights if MoE
        router_weights = {}
        if config.num_experts > 1:
            router_weights = loader.load_router_weights()

        # Load layers
        for layer_idx in range(config.num_hidden_layers):
            layer = TrellisDecoderLayer.from_loader(
                loader, config, layer_idx, router_weights, base_weights, device
            )
            model.layers.append(layer)

            if load_in_layers:
                # Clear loader cache to save memory
                loader.clear_layer_cache(layer_idx)

        return model.to(device)

    @staticmethod
    def _load_base_weights(model_path: str) -> dict[str, torch.Tensor]:
        """Load non-quantized weights (embedding, norms, lm_head)."""
        from pathlib import Path

        from safetensors.torch import load_file

        path = Path(model_path)

        # Try loading from quantized model directory
        base_weights_path = path / "base_weights.safetensors"
        if base_weights_path.exists():
            return load_file(base_weights_path)

        # Fall back to HuggingFace
        raise FileNotFoundError(
            f"base_weights.safetensors not found in {model_path}. "
            "Run extract_base_weights.py first."
        )


class TrellisForCausalLM(nn.Module):
    """Trellis model with language modeling head for text generation.

    Wraps TrellisModel with an LM head projection for generating logits.
    Supports autoregressive generation with temperature, top-k, and top-p sampling.

    Attributes:
        model: The underlying TrellisModel.
        config: Model configuration.
        lm_head: Linear projection from hidden_size to vocab_size.
    """

    def __init__(self, config: TrellisModelConfig):
        """Initialize TrellisForCausalLM.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.model = TrellisModel(config)

        # LM head (not quantized, tied to embedding or separate)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: TrellisKVCache | None = None,
    ) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            input_ids: Input token IDs [batch, seq_len].
            attention_mask: Optional attention mask [batch, seq_len].
            position_ids: Optional position IDs [batch, seq_len].
            kv_cache: Optional KV cache for generation.

        Returns:
            Logits tensor [batch, seq_len, vocab_size].
        """
        hidden_states = self.model(input_ids, attention_mask, position_ids, kv_cache)
        logits = self.lm_head(hidden_states)
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Autoregressive generation with KV cache.

        Generates tokens autoregressively using the model with efficient
        KV caching for improved performance on long sequences.

        Args:
            input_ids: Initial token IDs [batch, seq_len].
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (1.0 = greedy, <1.0 = focused, >1.0 = random).
            top_k: Number of highest probability tokens to keep for top-k sampling.
            top_p: Cumulative probability threshold for nucleus (top-p) sampling.

        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens].
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initialize KV cache for efficient generation
        kv_cache = TrellisKVCache(
            num_layers=self.config.num_hidden_layers,
            batch_size=batch_size,
            max_seq_len=seq_len + max_new_tokens,
            kv_lora_rank=self.config.kv_lora_rank,
            device=str(device),
        )

        # Track which sequences are finished (for batched generation)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Initial forward pass to fill KV cache with prompt
        _ = self.forward(input_ids, kv_cache=kv_cache)

        # Get current sequence length from cache
        current_len = kv_cache.get_seq_len()

        # Generate tokens one at a time
        for _ in range(max_new_tokens):
            # Get logits for the last position only
            logits = self.forward(
                input_ids[:, -1:],
                kv_cache=kv_cache,
            )  # [batch, 1, vocab_size]
            next_token_logits = logits[:, -1, :]  # [batch, vocab_size]

            # Apply temperature
            if temperature != 1.0 and temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k, dim=-1)[0][..., -1, None]
                )
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float("-inf"))

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float("-inf"))

            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]

            # Mark sequences as finished if EOS token is generated
            if hasattr(self.config, "eos_token_id") and self.config.eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == self.config.eos_token_id)
            else:
                # Default EOS token ID (commonly 2 for many models)
                finished = finished | (next_token.squeeze(-1) == 2)

            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Stop if all sequences are finished
            if finished.all():
                break

        return input_ids

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "mps",
    ) -> TrellisForCausalLM:
        """Load a TrellisForCausalLM model from path.

        Loads the model configuration, base model weights, and LM head weights
        from the specified path. Supports both tied and separate LM heads.

        Args:
            model_path: Path to the model directory containing config.json
                and base_weights.safetensors.
            device: Device to load the model on (default: "mps").

        Returns:
            Loaded TrellisForCausalLM instance.
        """
        config = TrellisModelConfig.from_pretrained(model_path)
        model = cls(config)

        # Load base model
        model.model = TrellisModel.from_pretrained(model_path, device)

        # Load lm_head weight from base_weights.safetensors (may be tied to embed_tokens)
        base_weights = TrellisModel._load_base_weights(model_path)
        if "lm_head.weight" in base_weights:
            model.lm_head.weight.data = base_weights["lm_head.weight"].to(device)
        else:
            # Tied embeddings - share weight with embed_tokens
            model.lm_head.weight = model.model.embed_tokens.weight

        return model.to(device)
