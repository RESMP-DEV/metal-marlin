"""Legacy Llama model implementations (deprecated)."""

from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attention import RoPE
from ..quantized_linear import QuantizedLinear
from ..quantized_loader import QuantizedModel
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


def _get_bias(quantized_model: QuantizedModel, name: str) -> torch.Tensor | None:
    return quantized_model.bf16_weights.get(name)


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


class QuantizedLlamaAttention(nn.Module):
    """DEPRECATED: legacy custom Llama attention.

    Prefer Transformers + replace_linear_layers() instead.
    """

    def __init__(
        self,
        quantized_model: QuantizedModel,
        layer_idx: int,
        warn_if_standalone: bool = True,
    ):
        if warn_if_standalone:
            warnings.warn(
                "QuantizedLlamaAttention is deprecated. "
                "Use Transformers + replace_linear_layers() instead. "
                "See STATUS.md for migration guide.",
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


class QuantizedLlamaMLP(nn.Module):
    """DEPRECATED: legacy custom Llama MLP.

    Prefer Transformers + replace_linear_layers() instead.
    """

    def __init__(
        self,
        quantized_model: QuantizedModel,
        layer_idx: int,
        warn_if_standalone: bool = True,
    ):
        if warn_if_standalone:
            warnings.warn(
                "QuantizedLlamaMLP is deprecated. "
                "Use Transformers + replace_linear_layers() instead. "
                "See STATUS.md for migration guide.",
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


class QuantizedLlamaLayer(nn.Module):
    """DEPRECATED: legacy custom Llama layer.

    Prefer Transformers + replace_linear_layers() instead.
    """

    def __init__(
        self,
        quantized_model: QuantizedModel,
        layer_idx: int,
        warn_if_standalone: bool = True,
    ):
        if warn_if_standalone:
            warnings.warn(
                "QuantizedLlamaLayer is deprecated. "
                "Use Transformers + replace_linear_layers() instead. "
                "See STATUS.md for migration guide.",
                DeprecationWarning,
                stacklevel=2,
            )
        super().__init__()
        config = quantized_model.config
        hidden_size = _get_config_value(config, "hidden_size", 4096)
        eps = _get_config_value(config, "rms_norm_eps", 1e-6)

        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.self_attn = QuantizedLlamaAttention(
            quantized_model,
            layer_idx,
            warn_if_standalone=False,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.mlp = QuantizedLlamaMLP(
            quantized_model,
            layer_idx,
            warn_if_standalone=False,
        )

        input_norm_weight = _get_bias(quantized_model, f"model.layers.{layer_idx}.input_layernorm.weight")
        if input_norm_weight is not None:
            self.input_layernorm.weight.data.copy_(input_norm_weight)

        post_norm_weight = _get_bias(
            quantized_model, f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        )
        if post_norm_weight is not None:
            self.post_attention_layernorm.weight.data.copy_(post_norm_weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: Any | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QuantizedLlama(nn.Module):
    """Quantized Llama for inference."""

    def __init__(self, quantized_model: QuantizedModel):
        super().__init__()
        self.config = quantized_model.config
        hidden_size = _get_config_value(self.config, "hidden_size", 4096)
        num_layers = _get_config_value(self.config, "num_hidden_layers", 32)
        vocab_size = _get_config_value(self.config, "vocab_size", 32000)
        eps = _get_config_value(self.config, "rms_norm_eps", 1e-6)

        embed_weight = quantized_model.bf16_weights.get("model.embed_tokens.weight")
        if embed_weight is None:
            raise KeyError("Missing embedding weight: model.embed_tokens.weight")
        self.embed_tokens = nn.Embedding.from_pretrained(embed_weight)

        self.layers = nn.ModuleList(
            [
                QuantizedLlamaLayer(
                    quantized_model,
                    i,
                    warn_if_standalone=False,
                )
                for i in range(num_layers)
            ]
        )

        self.norm = RMSNorm(hidden_size, eps=eps)
        norm_weight = quantized_model.bf16_weights.get("model.norm.weight")
        if norm_weight is not None:
            self.norm.weight.data.copy_(norm_weight)

        if "lm_head.weight" in quantized_model.weights:
            self.lm_head = _build_quantized_linear(quantized_model, "lm_head.weight")
        elif "lm_head.weight" in quantized_model.bf16_weights:
            self.lm_head = _build_linear_from_weight(quantized_model.bf16_weights["lm_head.weight"])
        else:
            self.lm_head = None
            self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=past_key_values,
                layer_idx=layer_idx,
            )

        hidden_states = self.norm(hidden_states)
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        return F.linear(hidden_states, self.embed_tokens.weight)

    @classmethod
    def from_quantized(cls, path: str) -> QuantizedLlama:
        quantized = QuantizedModel.load(path)
        return cls(quantized)


__all__ = [
    "QuantizedLlama",
    "QuantizedLlamaAttention",
    "QuantizedLlamaLayer",
    "QuantizedLlamaMLP",
]
