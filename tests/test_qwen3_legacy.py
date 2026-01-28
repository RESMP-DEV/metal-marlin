"""Legacy Qwen3 component tests.

Tests deprecated QuantizedQwen3Attention/MLP/Layer classes.
Primary Qwen3 validation:
- test_qwen3_dense_transformers.py (dense models)
- test_qwen3_moe_transformers.py (MoE models)

Kept for regression testing of layer internals.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch

if HAS_TORCH:
    import torch.nn as nn
    import torch.nn.functional as F

    from metal_marlin.legacy.qwen3 import (
        QuantizedQwen3Attention,
        QuantizedQwen3Layer,
        QuantizedQwen3MLP,
    )

# Try to import inference engine (may not be available)
_HAS_INFERENCE_ENGINE = False
try:
    from metal_marlin.inference import MetalInferenceEngine, load_quantized_model

    _HAS_INFERENCE_ENGINE = True
except Exception:
    pass


pytestmark = [
    pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch"),
    pytest.mark.skip(reason="Legacy components - use Transformers integration tests instead"),
]


def create_mock_quantized_model(hidden_size: int = 2048, num_heads: int = 16, num_layers: int = 2):
    model = MagicMock()
    model.config = {
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_heads // 4,  # GQA
        "intermediate_size": hidden_size * 4,
        "num_hidden_layers": num_layers,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
    }
    model.weights = {}
    model.bf16_weights = {}
    return model


@pytest.fixture
def mock_model():
    return create_mock_quantized_model()


@pytest.fixture
def torch_seed():
    torch.manual_seed(0)


def _populate_qwen3_weights(model: MagicMock, layer_idx: int = 0) -> None:
    config = model.config
    hidden_size = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = hidden_size // num_heads
    intermediate_size = config["intermediate_size"]

    prefix = f"model.layers.{layer_idx}."
    attn_prefix = f"{prefix}self_attn."
    mlp_prefix = f"{prefix}mlp."

    def rand_weight(out_features: int, in_features: int) -> torch.Tensor:
        return torch.randn(out_features, in_features, dtype=torch.float32)

    model.weights[f"{attn_prefix}q_proj.weight"] = rand_weight(hidden_size, hidden_size)
    model.weights[f"{attn_prefix}k_proj.weight"] = rand_weight(num_kv_heads * head_dim, hidden_size)
    model.weights[f"{attn_prefix}v_proj.weight"] = rand_weight(num_kv_heads * head_dim, hidden_size)
    model.weights[f"{attn_prefix}o_proj.weight"] = rand_weight(hidden_size, hidden_size)

    model.weights[f"{mlp_prefix}gate_proj.weight"] = rand_weight(intermediate_size, hidden_size)
    model.weights[f"{mlp_prefix}up_proj.weight"] = rand_weight(intermediate_size, hidden_size)
    model.weights[f"{mlp_prefix}down_proj.weight"] = rand_weight(hidden_size, intermediate_size)

    model.bf16_weights[f"{prefix}input_layernorm.weight"] = torch.ones(
        hidden_size, dtype=torch.float32
    )
    model.bf16_weights[f"{prefix}post_attention_layernorm.weight"] = torch.ones(
        hidden_size, dtype=torch.float32
    )


if HAS_TORCH:

    class ZeroModule(nn.Module):
        """Module that returns zeros with the same shape as input."""

        def forward(self, x, *args, **kwargs):
            return torch.zeros_like(x)


# ==============================================================================
# Tests
# ==============================================================================


def test_qwen3_attention_shapes(mock_model, torch_seed):
    """Verify Q/K/V/O dimensions for Qwen3 attention."""
    _populate_qwen3_weights(mock_model)

    attention = QuantizedQwen3Attention(mock_model, layer_idx=0)
    hidden_size = mock_model.config["hidden_size"]
    num_heads = mock_model.config["num_attention_heads"]
    num_kv_heads = mock_model.config["num_key_value_heads"]
    head_dim = hidden_size // num_heads

    batch_size, seq_len = 2, 5
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    q = attention.q_proj(hidden_states)
    k = attention.k_proj(hidden_states)
    v = attention.v_proj(hidden_states)

    assert q.shape == (batch_size, seq_len, hidden_size)
    assert k.shape == (batch_size, seq_len, num_kv_heads * head_dim)
    assert v.shape == (batch_size, seq_len, num_kv_heads * head_dim)

    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    assert q.shape == (batch_size, num_heads, seq_len, head_dim)
    assert k.shape == (batch_size, num_kv_heads, seq_len, head_dim)
    assert v.shape == (batch_size, num_kv_heads, seq_len, head_dim)

    out = attention(hidden_states)
    assert out.shape == (batch_size, seq_len, hidden_size)


def test_qwen3_mlp_forward(mock_model, torch_seed):
    """Test SwiGLU activation in Qwen3 MLP."""
    _populate_qwen3_weights(mock_model)

    mlp = QuantizedQwen3MLP(mock_model, layer_idx=0)
    hidden_size = mock_model.config["hidden_size"]

    hidden_states = torch.randn(2, 4, hidden_size)

    expected = mlp.down_proj(F.silu(mlp.gate_proj(hidden_states)) * mlp.up_proj(hidden_states))
    actual = mlp(hidden_states)

    torch.testing.assert_close(actual, expected)


def test_qwen3_layer_forward(mock_model, torch_seed):
    """Full forward pass shapes for QuantizedQwen3Layer."""
    _populate_qwen3_weights(mock_model)

    try:
        layer = QuantizedQwen3Layer(mock_model, layer_idx=0)
    except NotImplementedError:
        pytest.xfail("QuantizedQwen3Layer not implemented yet")

    hidden_size = mock_model.config["hidden_size"]
    hidden_states = torch.randn(2, 3, hidden_size)

    out = layer(hidden_states, attention_mask=None, past_key_values=None)
    assert out.shape == hidden_states.shape


def test_qwen3_layer_residual(mock_model, torch_seed):
    """Verify residual connections are applied in the Qwen3 layer."""
    _populate_qwen3_weights(mock_model)

    try:
        layer = QuantizedQwen3Layer(mock_model, layer_idx=0)
    except NotImplementedError:
        pytest.xfail("QuantizedQwen3Layer not implemented yet")

    if not hasattr(layer, "self_attn") or not hasattr(layer, "mlp"):
        pytest.skip("Qwen3 layer missing expected modules for residual test")

    if hasattr(layer, "input_layernorm"):
        layer.input_layernorm = nn.Identity()
    if hasattr(layer, "post_attention_layernorm"):
        layer.post_attention_layernorm = nn.Identity()

    layer.self_attn = ZeroModule()
    layer.mlp = ZeroModule()

    hidden_size = mock_model.config["hidden_size"]
    hidden_states = torch.randn(2, 3, hidden_size)

    out = layer(hidden_states, attention_mask=None, past_key_values=None)
    torch.testing.assert_close(out, hidden_states)


# === Merged from test_qwen3_inference.py ===

# NOTE: imports already at top of file from test_qwen3_layer.py

ROOT = Path(__file__).resolve().parents[1]


def _resolve_model_path(model_path: str) -> Path:
    resolved = ROOT / model_path
    if not resolved.exists():
        pytest.skip(f"Model artifacts not found: {resolved}")
    return resolved


@pytest.mark.parametrize(
    "model_path,expected_layers",
    [
        ("benchmarks/results/qwen3_4b_fp4", 36),
        ("benchmarks/results/qwen3_32b_fp4", 64),
        ("benchmarks/results/qwen3_30b_fp8_int2", 48),
    ],
)
def test_qwen3_loads(model_path: str, expected_layers: int) -> None:
    model, _tokenizer = load_quantized_model(_resolve_model_path(model_path))
    if hasattr(model, "layers"):
        assert len(model.layers) == expected_layers
    else:
        assert getattr(model, "num_layers", None) == expected_layers


@pytest.mark.smoke
@pytest.mark.skipif(not HAS_MPS, reason="Requires MPS (Apple Silicon)")
def test_qwen3_4b_generates() -> None:
    """Quick smoke test with small model."""
    engine = MetalInferenceEngine(str(_resolve_model_path("benchmarks/results/qwen3_4b_fp4")))
    output = engine.generate("What is 2+2?", max_tokens=20)
    assert "4" in output or "four" in output.lower()
