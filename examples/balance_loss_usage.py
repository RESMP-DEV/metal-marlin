#!/usr/bin/env python3
"""Example: Using auxiliary balance loss for expert utilization.

The balance loss encourages even expert utilization during fine-tuning
by penalizing high variance in expert loads.
"""

import torch

from metal_marlin.trellis.config import TrellisModelConfig
from metal_marlin.trellis.moe import TrellisMoELayer


def train_with_balance_loss():
    """Example training loop with balance loss."""

    # Create config
    config = TrellisModelConfig(
        hidden_size=128,
        intermediate_size=256,
        num_experts=8,
        num_experts_per_tok=2,
    )

    # Initialize MoE layer with balance loss
    layer = TrellisMoELayer(
        config=config,
        layer_weights={},  # Load actual weights here
        router_weight=torch.randn(config.num_experts, config.hidden_size),
        layer_idx=0,
        device="cpu",
        aux_loss_weight=0.01,  # Enable balance loss with weight 0.01
    )

    # Optimizer
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-4)

    # Training loop
    layer.train()
    for step in range(10):
        # Forward pass
        x = torch.randn(32, config.hidden_size)
        output = layer(x)

        # Compute task loss (e.g., language modeling)
        task_loss = output.pow(2).mean()  # Placeholder

        # Get auxiliary balance loss
        aux_loss = layer.get_aux_loss()

        # Total loss = task loss + aux loss
        total_loss = task_loss + aux_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Step {step}: task_loss={task_loss:.4f}, aux_loss={aux_loss:.4f}")


def inference_without_balance_loss():
    """Example inference without balance loss overhead."""

    config = TrellisModelConfig(
        hidden_size=128,
        intermediate_size=256,
        num_experts=8,
        num_experts_per_tok=2,
    )

    # Initialize MoE layer without balance loss for inference
    layer = TrellisMoELayer(
        config=config,
        layer_weights={},
        router_weight=torch.randn(config.num_experts, config.hidden_size),
        layer_idx=0,
        device="cpu",
        aux_loss_weight=0.0,  # Disabled for inference
    )

    layer.eval()
    with torch.no_grad():
        x = torch.randn(32, config.hidden_size)
        output = layer(x)
        # aux_loss will be 0.0
        print(f"Inference output shape: {output.shape}")


if __name__ == "__main__":
    print("=== Training with balance loss ===")
    train_with_balance_loss()

    print("\n=== Inference without balance loss ===")
    inference_without_balance_loss()
