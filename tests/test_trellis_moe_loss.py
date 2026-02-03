"""Tests for MoE auxiliary balance loss in TrellisMoELayer."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from metal_marlin.trellis.config import TrellisModelConfig
from metal_marlin.trellis.moe import TrellisMoELayer
from metal_marlin.trellis.testing import create_mock_trellis_linear

HAS_MPS = torch.backends.mps.is_available()
requires_mps = pytest.mark.skipif(not HAS_MPS, reason="MPS required (Apple Silicon)")

def create_mock_trellis_moe_layer(
    hidden_dim=64,
    intermediate_dim=128,
    num_experts=4,
    num_experts_per_tok=2,
    bits=3,
    device="mps",
    aux_loss_weight=0.0
):
    """Create a TrellisMoELayer with mock weights."""
    config = TrellisModelConfig(
        hidden_size=hidden_dim,
        intermediate_size=intermediate_dim,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        quantization_bits=bits,
    )

    router_weight = torch.randn(num_experts, hidden_dim, device=device, dtype=torch.float32)

    # Create weight dicts for each expert
    layer_weights = {}
    for i in range(num_experts):
        # Create mock linear layers
        gate = create_mock_trellis_linear(hidden_dim, intermediate_dim, bits, device)
        up = create_mock_trellis_linear(hidden_dim, intermediate_dim, bits, device)
        down = create_mock_trellis_linear(intermediate_dim, hidden_dim, bits, device)

        # Helper to extract dict from TrellisLinear
        def get_linear_dict(linear):
            return {
                "indices": linear.packed_indices,
                "scales": linear.scales,
                "su": linear.su,
                "sv": linear.sv,
                "bits": linear.bits,
                "original_shape": (linear.out_features, linear.in_features)
            }

        layer_weights[f"experts.{i}.gate_proj"] = get_linear_dict(gate)
        layer_weights[f"experts.{i}.up_proj"] = get_linear_dict(up)
        layer_weights[f"experts.{i}.down_proj"] = get_linear_dict(down)

    layer = TrellisMoELayer(
        config=config,
        layer_weights=layer_weights,
        router_weight=router_weight,
        layer_idx=0,
        device=device,
        aux_loss_weight=aux_loss_weight,
    )

    return layer

@requires_mps
class TestMoEBalanceLoss:
    """Test 7: Auxiliary balance loss functionality."""

    def test_loss_computation(self):
        """Verify balance loss is computed and stored."""
        torch.manual_seed(42)
        device = "mps"

        # Create MoE with balance loss enabled
        layer = create_mock_trellis_moe_layer(
            hidden_dim=64,
            intermediate_dim=128,
            num_experts=4,
            num_experts_per_tok=2,
            bits=3,
            device=device,
            aux_loss_weight=0.1,
        )
        layer.train()  # Loss only computed in training mode

        x = torch.randn(4, 64, dtype=torch.float16, device=device)

        # Initial loss should be 0 (from buffer init)
        assert layer.get_aux_loss().item() == 0.0

        # Forward pass
        _ = layer(x)

        # Loss should be updated
        loss = layer.get_aux_loss()
        assert loss.item() > 0.0, "Balance loss should be positive for random initialization"

        # Verify loss tensor is on correct device
        assert loss.device.type == device

    def test_loss_gradients(self):
        """Verify gradients flow from balance loss to router weights."""
        torch.manual_seed(42)
        device = "mps"

        layer = create_mock_trellis_moe_layer(
            hidden_dim=64,
            intermediate_dim=128,
            num_experts=4,
            num_experts_per_tok=2,
            bits=3,
            device=device,
            aux_loss_weight=1.0,
        )
        layer.train()

        # Ensure router weights require gradients
        layer.router.weight.requires_grad_(True)
        # Clear existing gradients
        if layer.router.weight.grad is not None:
            layer.router.weight.grad.zero_()

        x = torch.randn(4, 64, dtype=torch.float16, device=device)

        # Forward pass
        _ = layer(x)

        # Get loss
        loss = layer.get_aux_loss()

        # Backward pass on loss only
        loss.backward()

        # Check gradients on router
        assert layer.router.weight.grad is not None, "Router weights should have gradients"
        # Since using differentiable loss now, gradient should be non-zero
        assert layer.router.weight.grad.abs().sum() > 0, "Router gradients should be non-zero"

    def test_loss_disabled_inference(self):
        """Verify loss is not computed in eval mode or when disabled."""
        torch.manual_seed(42)
        device = "mps"

        # 1. Enabled but in eval mode
        layer = create_mock_trellis_moe_layer(
            hidden_dim=64,
            intermediate_dim=128,
            num_experts=4,
            num_experts_per_tok=2,
            bits=3,
            device=device,
            aux_loss_weight=0.1,
        )
        layer.eval()

        x = torch.randn(4, 64, dtype=torch.float16, device=device)
        _ = layer(x)

        assert layer.get_aux_loss().item() == 0.0, "Loss should be 0 in eval mode"

        # 2. Disabled via weight=0.0
        layer = create_mock_trellis_moe_layer(
            hidden_dim=64,
            intermediate_dim=128,
            num_experts=4,
            num_experts_per_tok=2,
            bits=3,
            device=device,
            aux_loss_weight=0.0,
        )
        layer.train()

        _ = layer(x)
        assert layer.get_aux_loss().item() == 0.0, "Loss should be 0 when weight is 0.0"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
