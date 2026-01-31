#!/usr/bin/env python3
"""Check weight buffer sizes and indexing."""

import torch

from metal_marlin.trellis.testing import create_mock_moe_mlp

torch.manual_seed(0)
moe = create_mock_moe_mlp(
    hidden_dim=2048,
    intermediate_dim=1536,
    num_experts=64,
    num_experts_per_tok=8,
    bits=3,
    device="mps",
)

print("Weight buffer shapes:")
print(f"  gate_weights_stacked: {moe.gate_weights_stacked.shape}")
print(f"  gate_scales_stacked: {moe.gate_scales_stacked.shape}")
print(f"  gate_su_stacked: {moe.gate_su_stacked.shape}")
print(f"  gate_sv_stacked: {moe.gate_sv_stacked.shape}")

# Check if shapes match expected
num_experts = 64
hidden_dim = 2048
intermediate_dim = 1536
TRELLIS_TILE = 16
PACKED_BYTES = 96

num_tiles_k = (hidden_dim + TRELLIS_TILE - 1) // TRELLIS_TILE  # 128
num_tiles_n = (intermediate_dim + TRELLIS_TILE - 1) // TRELLIS_TILE  # 96

print()
print(f"Expected gate weight shape: [{num_experts}, {num_tiles_k}, {num_tiles_n}, {PACKED_BYTES}]")
print(f"Actual gate weight shape: {list(moe.gate_weights_stacked.shape)}")

# Check if expert 32 access would be valid
expert_32_offset = 32 * num_tiles_k * num_tiles_n * PACKED_BYTES
total_size = num_experts * num_tiles_k * num_tiles_n * PACKED_BYTES
print()
print(f"Expert 32 offset: {expert_32_offset} bytes")
print(f"Total buffer size: {total_size} bytes")
print(f"Buffer within bounds: {expert_32_offset < total_size}")

# Check scale buffer layout
group_size = 128
n_groups = (hidden_dim + group_size - 1) // group_size
print()
print(f"Expected scale shape: [{num_experts}, {n_groups}, {intermediate_dim}]")
print(f"Actual scale shape: {list(moe.gate_scales_stacked.shape)}")
