import pytest
import torch
import torch.nn as nn

from metal_marlin.early_exit import EarlyExitModel
from metal_marlin.trellis.config import TrellisModelConfig
from metal_marlin.trellis.model import TrellisDecoderLayer


class MockAttention(nn.Module):
    def forward(self, hidden_states, **kwargs):
        # Return zeros so residual connection preserves input
        return torch.zeros_like(hidden_states)

class MockMLP(nn.Module):
    def forward(self, hidden_states, **kwargs):
        # Return zeros so residual connection preserves input
        return torch.zeros_like(hidden_states)

def create_model(exit_threshold=0.5):
    config = TrellisModelConfig(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        vocab_size=100,
        num_experts=1
    )
    model = EarlyExitModel(config, exit_threshold=exit_threshold)

    # Manually populate layers with mocks
    for i in range(config.num_hidden_layers):
        layer = TrellisDecoderLayer(config, i, device="cpu")
        layer.self_attn = MockAttention()
        layer.mlp = MockMLP()
        model.model.layers.append(layer)

    return model

def test_early_exit_initialization():
    model = create_model()
    assert len(model.exit_heads) == 4
    assert isinstance(model.exit_heads[0], nn.Linear)

def test_early_exit_forward_no_exit():
    model = create_model(exit_threshold=0.99) # Very high threshold
    input_ids = torch.randint(0, 100, (1, 10))
    output = model(input_ids)

    assert output.logits.shape == (1, 10, 100)
    # Should run all layers (0, 1, 2, 3) and return exit_layer_idx=3 (0-based index of last layer)
    assert output.exit_layer_idx == 3

def test_early_exit_trigger():
    model = create_model(exit_threshold=0.1) # Low threshold

    # Force high confidence in layer 0 head
    # Set weights to output high value for index 0 or 1 depending on input sign
    with torch.no_grad():
        model.exit_heads[0].weight.fill_(0.0)
        model.exit_heads[0].weight[0, 0] = 1000.0
        model.exit_heads[0].weight[1, 0] = -1000.0

    input_ids = torch.randint(0, 100, (1, 10))
    output = model(input_ids)

    # Should exit at layer 0
    assert output.exit_layer_idx == 0
    assert output.logits.shape == (1, 10, 100)

def test_early_exit_trigger_layer_1():
    model = create_model(exit_threshold=0.1)

    # Layer 0: low confidence
    with torch.no_grad():
        model.exit_heads[0].weight.fill_(0.0) # all zeros -> uniform probs

        # Layer 1: high confidence
        model.exit_heads[1].weight.fill_(0.0)
        model.exit_heads[1].weight[0, 0] = 1000.0
        model.exit_heads[1].weight[1, 0] = -1000.0

    input_ids = torch.randint(0, 100, (1, 10))
    output = model(input_ids)

    # Should exit at layer 1
    assert output.exit_layer_idx == 1
