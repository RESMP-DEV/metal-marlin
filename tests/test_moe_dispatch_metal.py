import pytest
import torch

from metal_marlin.moe_dispatch import _USE_METAL

pytestmark = pytest.mark.skipif(not _USE_METAL, reason="Metal not available")

@pytest.mark.parametrize("batch,top_k,num_experts", [
    (32, 2, 8),
    (128, 2, 16),
    (256, 4, 64),
])
def test_group_tokens_matches_pytorch(batch, top_k, num_experts):
    import metal_marlin.moe_dispatch as mod
    from metal_marlin.moe_dispatch_metal import group_tokens_by_expert_metal

    torch.manual_seed(42)
    expert_ids = torch.randint(0, num_experts, (batch, top_k), device="mps")

    # PyTorch path
    mod._USE_METAL = False
    sorted_pt, offsets_pt, inverse_pt = mod.group_tokens_by_expert(expert_ids, num_experts)

    # Metal path
    sorted_metal, offsets_metal, inverse_metal = group_tokens_by_expert_metal(
        expert_ids, num_experts
    )

    # Offsets should match exactly
    torch.testing.assert_close(offsets_metal, offsets_pt)

    # sorted_indices should produce same grouping
    # (exact order within expert group may differ)
    expert_ids_flat = expert_ids.reshape(-1)
    for e in range(num_experts):
        start, end = int(offsets_pt[e]), int(offsets_pt[e+1])
        pt_experts = expert_ids_flat[sorted_pt[start:end]]
        metal_experts = expert_ids_flat[sorted_metal[start:end]]
        assert (pt_experts == e).all()
        assert (metal_experts == e).all()

def test_gather_scatter_roundtrip():
    """Test gather + expert compute + scatter gives correct result."""
    from metal_marlin.moe_dispatch import (
        gather_for_experts,
        group_tokens_by_expert_full,
        scatter_expert_outputs,
    )

    torch.manual_seed(42)
    batch, top_k, num_experts, hidden = 64, 2, 8, 256

    expert_ids = torch.randint(0, num_experts, (batch, top_k), device="mps")
    expert_probs = torch.softmax(torch.randn(batch, top_k, device="mps"), dim=-1)
    activations = torch.randn(batch, hidden, device="mps")

    dispatch_info = group_tokens_by_expert_full(expert_ids, num_experts)
    gathered = gather_for_experts(activations, dispatch_info)

    # Simulate expert computation (identity for testing)
    expert_outputs = gathered

    combined = scatter_expert_outputs(expert_outputs, expert_probs, dispatch_info)

    # Manually compute expected output
    expected = torch.zeros(batch, hidden, device="mps")
    for b in range(batch):
        for k in range(top_k):
            expected[b] += expert_probs[b, k] * activations[b]

    torch.testing.assert_close(combined, expected, rtol=1e-4, atol=1e-5)
