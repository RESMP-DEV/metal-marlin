#!/usr/bin/env python3
"""Isolate which dimension causes NaN in MoE kernel."""

import torch

from metal_marlin.trellis.testing import create_mock_moe_mlp

# Test seed 0 with varying configs to isolate issue
configs = [
    (256, 512, 4, 2),  # Original mock size
    (256, 512, 64, 8),  # More experts
    (2048, 512, 4, 2),  # Larger hidden
    (256, 1536, 4, 2),  # Larger intermediate
    (2048, 1536, 4, 2),  # Real dims, fewer experts
    (2048, 1536, 64, 8),  # Full real config
]

for hidden, inter, experts, topk in configs:
    torch.manual_seed(0)
    moe = create_mock_moe_mlp(
        hidden_dim=hidden,
        intermediate_dim=inter,
        num_experts=experts,
        num_experts_per_tok=topk,
        bits=3,
        device="mps",
    )
    moe._use_fast_moe = True

    torch.manual_seed(100)
    x = torch.randn(4, hidden, dtype=torch.float16, device="mps")

    with torch.inference_mode():
        out = moe(x)

    has_nan = out.isnan().any().item()
    status = "NaN!" if has_nan else "OK"
    print(f"hidden={hidden}, inter={inter}, experts={experts}, topk={topk}: {status}")
