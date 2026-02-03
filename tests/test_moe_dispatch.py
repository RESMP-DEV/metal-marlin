import os
import sys

import pytest
import torch


def _has_metal() -> bool:
    try:
        from metal_marlin.metal_dispatch import HAS_METAL, HAS_MPS
        return HAS_METAL and HAS_MPS
    except ImportError:
        return False

pytestmark = pytest.mark.skipif(not _has_metal(), reason="Metal not available")

@pytest.fixture
def mock_moe_weights():
    num_experts = 4
    hidden = 256
    intermediate = 192
    top_k = 2
    bits = 4  # Matches Trellis
    group_size = 128
    n_levels = 1 << bits
    tile_size = 16
    packed_bytes_per_tile = (tile_size * tile_size * bits + 7) // 8

    num_tiles_k_gate = (hidden + tile_size - 1) // tile_size
    num_tiles_n_gate = (intermediate + tile_size - 1) // tile_size
    num_tiles_k_down = (intermediate + tile_size - 1) // tile_size
    num_tiles_n_down = (hidden + tile_size - 1) // tile_size

    n_groups_gate = (hidden + group_size - 1) // group_size
    n_groups_down = (intermediate + group_size - 1) // group_size

    device = "mps"

    return {
        "num_experts": num_experts,
        "hidden_dim": hidden,
        "intermediate_dim": intermediate,
        "top_k": top_k,
        "bits": bits,
        "gate_weights": torch.randint(0, 256, (num_experts, num_tiles_k_gate, num_tiles_n_gate, packed_bytes_per_tile), dtype=torch.uint8, device=device),
        "gate_scales": torch.randn(num_experts, n_groups_gate, intermediate, dtype=torch.float16, device=device),
        "gate_su": torch.randn(num_experts, hidden, dtype=torch.float16, device=device),
        "gate_sv": torch.randn(num_experts, intermediate, dtype=torch.float16, device=device),
        "up_weights": torch.randint(0, 256, (num_experts, num_tiles_k_gate, num_tiles_n_gate, packed_bytes_per_tile), dtype=torch.uint8, device=device),
        "up_scales": torch.randn(num_experts, n_groups_gate, intermediate, dtype=torch.float16, device=device),
        "up_su": torch.randn(num_experts, hidden, dtype=torch.float16, device=device),
        "up_sv": torch.randn(num_experts, intermediate, dtype=torch.float16, device=device),
        "down_weights": torch.randint(0, 256, (num_experts, num_tiles_k_down, num_tiles_n_down, packed_bytes_per_tile), dtype=torch.uint8, device=device),
        "down_scales": torch.randn(num_experts, n_groups_down, hidden, dtype=torch.float16, device=device),
        "down_su": torch.randn(num_experts, intermediate, dtype=torch.float16, device=device),
        "down_sv": torch.randn(num_experts, hidden, dtype=torch.float16, device=device),
        "grid": torch.randn(n_levels, dtype=torch.float16, device=device),
    }

def test_dispatch_moe_trellis_swiglu_batched_executes(mock_moe_weights):
    from metal_marlin.metal_dispatch import MetalKernelLibrary
    from metal_marlin.trellis.moe_dispatch import dispatch_moe_trellis_swiglu_batched

    lib = MetalKernelLibrary.from_source_dir()

    hidden_dim = mock_moe_weights["hidden_dim"]
    intermediate_dim = mock_moe_weights["intermediate_dim"]
    num_experts = mock_moe_weights["num_experts"]
    top_k = mock_moe_weights["top_k"]
    bits = mock_moe_weights["bits"]

    batch_size = 8
    device = "mps"

    activations = torch.randn(batch_size, hidden_dim, dtype=torch.float16, device=device)
    expert_ids = torch.randint(0, num_experts, (batch_size, top_k), dtype=torch.int32, device=device)
    expert_probs = torch.rand(batch_size, top_k, dtype=torch.float16, device=device)
    expert_probs = expert_probs / expert_probs.sum(dim=-1, keepdim=True)

    # Run the batched dispatch
    # Note: we pass weights directly, cached_buffers=None
    output = dispatch_moe_trellis_swiglu_batched(
        lib=lib,
        activations=activations,
        gate_weights=mock_moe_weights["gate_weights"],
        gate_scales=mock_moe_weights["gate_scales"],
        up_weights=mock_moe_weights["up_weights"],
        up_scales=mock_moe_weights["up_scales"],
        down_weights=mock_moe_weights["down_weights"],
        down_scales=mock_moe_weights["down_scales"],
        gate_su=mock_moe_weights["gate_su"],
        gate_sv=mock_moe_weights["gate_sv"],
        up_su=mock_moe_weights["up_su"],
        up_sv=mock_moe_weights["up_sv"],
        down_su=mock_moe_weights["down_su"],
        down_sv=mock_moe_weights["down_sv"],
        grid=mock_moe_weights["grid"],
        expert_ids=expert_ids,
        expert_probs=expert_probs,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        top_k=top_k,
        bits=bits,
        cached_buffers=None,
        buffer_pool=None
    )

    assert output.shape == (batch_size, hidden_dim)
    assert output.dtype == torch.float16
    assert not output.isnan().any()
