"""Check for NaN at tile boundaries in MoE kernel."""

import torch

from metal_marlin.trellis.testing import create_mock_moe_mlp

# Test with different intermediate_dim values
# 512 = 8 tiles (no boundary issues expected)
# 520 = 8 tiles + partial (potential boundary issue)

for intermediate_dim in [512, 520, 576, 600]:
    moe = create_mock_moe_mlp(
        hidden_dim=256,
        intermediate_dim=intermediate_dim,
        num_experts=4,
        device='mps'
    )
    x = torch.randn(1, 256, dtype=torch.float16, device='mps')

    with torch.inference_mode():
        output = moe(x)

    nan_count = output.isnan().sum().item()
    print(f"intermediate_dim={intermediate_dim}: NaN count={nan_count}")
