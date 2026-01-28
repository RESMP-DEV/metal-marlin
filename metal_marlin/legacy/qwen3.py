"""Legacy Qwen3 model implementations (deprecated)."""

from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attention import RoPE
from ..quantized_linear import QuantizedLinear
from ..quantized_loader import QuantizedModel
from ..rope import YaRNConfig, YaRNRoPE, get_yarn_mscale
from ..transformer import RMSNorm


def _get_config_value(config: dict, key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def _build_linear_from_weight(weight: torch.Tensor, bias: torch.Tensor | None = None) -> nn.Linear:
    out_features, in_features = weight.shape
    layer = nn.Linear(in_features, out_features, bias=bias is not None)
    layer.weight.data.copy_(weight)
    if bias is not None:
        layer.bias.data.copy_(bias)
    return layer


def _get_weight(quantized_model: QuantizedModel, name: str) -> Any:
    if name in quantized_model.weights:
        return quantized_model.weights[name]
    if name in quantized_model.bf16_weights:
        return quantized_model.bf16_weights[name]
    raise KeyError(f"Missing weight: {name}")


def _build_quantized_linear(quantized_model: QuantizedModel, weight_name: str) -> nn.Module:
    weight = _get_weight(quantized_model, weight_name)
    if isinstance(weight, torch.Tensor):
        return _build_linear_from_weight(weight)
    return QuantizedLinear(weight)


class QuantizedQwen3Attention(nn.Module):
    """DEPRECATED: Use Transformers + replace_linear_layers() instead."""

    def __init__(
        self,
        quantized_model: QuantizedModel,
        layer_idx: int,
        warn_if_standalone: bool = True,
    ):
        if warn_if_standalone:
            warnings.warn(
                "QuantizedQwen3Attention is deprecated. "
                "Use transformers.Qwen3ForCausalLM + replace_linear_layers().",
                DeprecationWarning,
                stacklevel=2,
            )
        super().__init__()
        config = quantized_model.config
        hidden_size = _get_config_value(config, "hidden_size", 4096)
        num_heads = _get_config_value(config, "num_attention_heads", 32)
        num_kv_heads = _get_config_value(config, "num_key_value_heads", num_heads)
        head_dim = hidden_size // num_heads

        prefix = f"model.layers.{layer_idx}.self_attn."
        self.q_proj = _build_quantized_linear(quantized_model, f"{prefix}q_proj.weight")
        self.k_proj = _build_quantized_linear(quantized_model, f"{prefix}k_proj.weight")
        self.v_proj = _build_quantized_linear(quantized_model, f"{prefix}v_proj.weight")
        self.o_proj = _build_quantized_linear(quantized_model, f"{prefix}o_proj.weight")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        yarn_config = YaRNConfig.from_hf_config(config)
        if yarn_config:
            self.rope = YaRNRoPE(dim=head_dim, config=yarn_config)
        else:
            self.rope = RoPE(head_dim, base=_get_config_value(config, "rope_theta", 10000.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: Any | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        _ = position_ids, kv_cache, layer_idx
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        if attention_mask is None:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=False)

        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn)


class QuantizedQwen3MLP(nn.Module):
    """DEPRECATED: Use Transformers + replace_linear_layers() instead."""

    def __init__(
        self,
        quantized_model: QuantizedModel,
        layer_idx: int,
        warn_if_standalone: bool = True,
    ):
        if warn_if_standalone:
            warnings.warn(
                "QuantizedQwen3MLP is deprecated. "
                "Use transformers.Qwen3ForCausalLM + replace_linear_layers().",
                DeprecationWarning,
                stacklevel=2,
            )
        super().__init__()
        prefix = f"model.layers.{layer_idx}.mlp."
        self.gate_proj = _build_quantized_linear(quantized_model, f"{prefix}gate_proj.weight")
        self.up_proj = _build_quantized_linear(quantized_model, f"{prefix}up_proj.weight")
        self.down_proj = _build_quantized_linear(quantized_model, f"{prefix}down_proj.weight")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class QuantizedQwen3Layer(nn.Module):
    """DEPRECATED: Use Transformers + replace_linear_layers() instead."""

    def __init__(
        self,
        quantized_model: QuantizedModel,
        layer_idx: int,
        warn_if_standalone: bool = True,
    ):
        if warn_if_standalone:
            warnings.warn(
                "QuantizedQwen3Layer is deprecated. "
                "Use transformers.Qwen3ForCausalLM + replace_linear_layers().",
                DeprecationWarning,
                stacklevel=2,
            )
        super().__init__()
        config = quantized_model.config
        hidden_size = config.get("hidden_size", 4096)
        eps = config.get("rms_norm_eps", 1e-6)

        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.self_attn = QuantizedQwen3Attention(
            quantized_model,
            layer_idx,
            warn_if_standalone=False,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.mlp = QuantizedQwen3MLP(
            quantized_model,
            layer_idx,
            warn_if_standalone=False,
        )

        prefix = f"model.layers.{layer_idx}"
        input_norm = quantized_model.bf16_weights.get(f"{prefix}.input_layernorm.weight")
        if input_norm is not None:
            self.input_layernorm.weight.data.copy_(input_norm)

        post_norm = quantized_model.bf16_weights.get(f"{prefix}.post_attention_layernorm.weight")
        if post_norm is not None:
            self.post_attention_layernorm.weight.data.copy_(post_norm)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class QuantizedQwen3MoE(nn.Module):
    """Quantized Qwen3 MoE for inference.

    Qwen3-30B-A3B reference:
        - 48 layers
        - 64 experts per layer
        - 8 experts active per token
        - Shared expert
    """

    def __init__(self, quantized_model: QuantizedModel):
        super().__init__()
        self.config = quantized_model.config

        # Qwen3 uses YaRN RoPE scaling for extended context.
        self.yarn_config = YaRNConfig.from_hf_config(self.config)
        if self.yarn_config is None:
            self.attention_scale = 1.0
        elif self.yarn_config.attention_factor is not None:
            self.attention_scale = self.yarn_config.attention_factor
        else:
            self.attention_scale = get_yarn_mscale(
                self.yarn_config.scale_factor, self.yarn_config.mscale_all_dim
            )

        # Build model structure matching HF architecture
        self.embed_tokens = nn.Embedding.from_pretrained(
            quantized_model.bf16_weights["model.embed_tokens.weight"]
        )

        self.layers = nn.ModuleList()
        for i in range(self.config["num_hidden_layers"]):
            # Qwen3 attention details are handled inside the layer implementation.
            layer = QuantizedQwen3Layer(
                quantized_model,
                i,
                warn_if_standalone=False,
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.config["hidden_size"])
        self.lm_head = QuantizedLinear(quantized_model.weights["lm_head.weight"])

    def forward(self, input_ids, attention_mask=None, past_key_values=None):
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, past_key_values)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    @classmethod
    def from_quantized(cls, path: str) -> QuantizedQwen3MoE:
        quantized = QuantizedModel.load(path)
        return cls(quantized)


__all__ = [
    "QuantizedQwen3Attention",
    "QuantizedQwen3Layer",
    "QuantizedQwen3MLP",
    "QuantizedQwen3MoE",
]
