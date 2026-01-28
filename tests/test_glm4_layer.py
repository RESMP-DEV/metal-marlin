"""Legacy GLM-4 layer component tests.

Tests deprecated MetalMLAAttention and MetalMLP classes.
Primary GLM-4.7 validation: test_glm47_transformers.py
Kept for regression testing of layer internals.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from metal_marlin._compat import HAS_TORCH
from metal_marlin.models.glm4 import (
    QuantizedGLM4Attention,
    QuantizedGLM4Layer,
    QuantizedGLM4MLP,
)

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")


def create_mock_glm4_model(overrides: dict[str, Any] | None = None):
    model = MagicMock()
    model.config = {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 64,
        "num_hidden_layers": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "rope_ratio": 1.0,
    }
    if overrides:
        model.config.update(overrides)
    model.weights = {}
    model.bf16_weights = {}
    return model


def _small_glm4_overrides() -> dict[str, Any]:
    return {
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "kv_lora_rank": 16,
        "q_lora_rank": 32,
        "qk_rope_head_dim": 8,
        "num_hidden_layers": 1,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "rope_ratio": 1.0,
        "group_size": 32,
    }


def _populate_glm4_attention_weights(model, layer_idx: int = 0) -> None:
    config = model.config
    hidden_size = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    head_dim = hidden_size // num_heads
    qk_rope_head_dim = config["qk_rope_head_dim"]
    qk_nope_head_dim = config.get("qk_nope_head_dim", head_dim - qk_rope_head_dim)
    v_head_dim = config.get("v_head_dim", head_dim)
    q_lora_rank = config.get("q_lora_rank")
    kv_lora_rank = config["kv_lora_rank"]

    prefix = f"model.layers.{layer_idx}.self_attn"

    if q_lora_rank is not None:
        model.weights[f"{prefix}.q_a_proj.weight"] = torch.randn(q_lora_rank, hidden_size)
        model.weights[f"{prefix}.q_b_proj.weight"] = torch.randn(
            num_heads * (qk_nope_head_dim + qk_rope_head_dim), q_lora_rank
        )
    else:
        model.weights[f"{prefix}.q_proj.weight"] = torch.randn(
            num_heads * (qk_nope_head_dim + qk_rope_head_dim), hidden_size
        )

    model.weights[f"{prefix}.kv_a_proj.weight"] = torch.randn(
        kv_lora_rank + qk_rope_head_dim, hidden_size
    )
    model.weights[f"{prefix}.kv_b_proj.weight"] = torch.randn(
        num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank
    )
    model.weights[f"{prefix}.o_proj.weight"] = torch.randn(hidden_size, num_heads * v_head_dim)


def _populate_glm4_mlp_weights(model, layer_idx: int = 0, intermediate_size: int = 128) -> None:
    hidden_size = model.config["hidden_size"]
    prefix = f"model.layers.{layer_idx}.mlp"

    model.weights[f"{prefix}.gate_proj.weight"] = torch.randn(intermediate_size, hidden_size)
    model.weights[f"{prefix}.up_proj.weight"] = torch.randn(intermediate_size, hidden_size)
    model.weights[f"{prefix}.down_proj.weight"] = torch.randn(hidden_size, intermediate_size)


def _populate_glm4_norm_weights(model, layer_idx: int = 0) -> None:
    hidden_size = model.config["hidden_size"]
    model.bf16_weights[f"model.layers.{layer_idx}.input_layernorm.weight"] = torch.ones(hidden_size)
    model.bf16_weights[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = torch.ones(
        hidden_size
    )


def test_glm4_mla_config():
    model = create_mock_glm4_model(_small_glm4_overrides())
    _populate_glm4_attention_weights(model)

    attention = QuantizedGLM4Attention(model, 0)
    mla = attention.mla

    assert mla.hidden_size == model.config["hidden_size"]
    assert mla.num_heads == model.config["num_attention_heads"]
    assert mla.kv_lora_rank == model.config["kv_lora_rank"]
    assert mla.q_lora_rank == model.config["q_lora_rank"]
    assert mla.qk_rope_head_dim == model.config["qk_rope_head_dim"]
    assert mla.rope_ratio == model.config["rope_ratio"]
    assert mla.rope_q.base == model.config["rope_theta"]


def test_glm4_attention_shapes():
    model = create_mock_glm4_model(_small_glm4_overrides())
    _populate_glm4_attention_weights(model)

    attention = QuantizedGLM4Attention(model, 0)

    hidden_size = model.config["hidden_size"]
    hidden_states = torch.randn(2, 3, hidden_size)
    out = attention(hidden_states)

    assert out.shape == hidden_states.shape


def test_glm4_mlp_forward():
    model = create_mock_glm4_model(_small_glm4_overrides())
    _populate_glm4_mlp_weights(model)

    mlp = QuantizedGLM4MLP(model, 0)

    hidden_size = model.config["hidden_size"]
    hidden_states = torch.randn(2, 3, hidden_size)
    out = mlp(hidden_states)

    assert out.shape == hidden_states.shape


def test_glm4_layer_forward():
    model = create_mock_glm4_model(_small_glm4_overrides())
    _populate_glm4_attention_weights(model)
    _populate_glm4_mlp_weights(model)
    _populate_glm4_norm_weights(model)

    try:
        layer = QuantizedGLM4Layer(model, 0)
    except NotImplementedError:
        pytest.xfail("QuantizedGLM4Layer not yet implemented")

    hidden_size = model.config["hidden_size"]
    hidden_states = torch.randn(2, 3, hidden_size)
    out = layer(hidden_states)

    assert out.shape == hidden_states.shape
