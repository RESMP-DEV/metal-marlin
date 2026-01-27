"""Test MoE integration with implemented kernels."""
import pytest
import torch

from metal_marlin.kernels import moe_expert_gemm_fp4, moe_router_topk


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
class TestMoEIntegration:
    def test_router_expert_gemm_pipeline(self):
        """Test full router -> expert GEMM pipeline."""
        batch, hidden_dim, out_dim = 8, 256, 512
        num_experts, top_k = 8, 2

        # Input
        hidden = torch.randn(batch, hidden_dim, device="mps", dtype=torch.float16)
        router_weights = torch.randn(hidden_dim, num_experts, device="mps", dtype=torch.float16)

        # Simulated FP4 expert weights
        expert_weights = torch.randint(
            0,
            255,
            (num_experts, hidden_dim // 8, out_dim),
            dtype=torch.uint8,
            device="mps",
        )
        scales = torch.randn(
            num_experts,
            hidden_dim // 128,
            out_dim,
            device="mps",
            dtype=torch.float16,
        ).abs()

        # Router
        expert_ids, expert_probs = moe_router_topk(
            hidden.float(), router_weights.float(), top_k=top_k
        )

        assert expert_ids.shape == (batch, top_k)
        assert expert_probs.shape == (batch, top_k)
        assert (expert_ids >= 0).all() and (expert_ids < num_experts).all()
        assert torch.allclose(
            expert_probs.sum(dim=-1),
            torch.ones(batch, device="mps"),
            atol=1e-5,
        )

        # Expert GEMM
        output = moe_expert_gemm_fp4(hidden, expert_weights, scales, expert_ids, expert_probs)

        assert output.shape == (batch, out_dim)
        assert not output.isnan().any()

    def test_moe_with_different_top_k(self):
        """Test router with various top_k values."""
        batch, hidden_dim, num_experts = 4, 128, 8
        hidden = torch.randn(batch, hidden_dim, device="mps")
        router_weights = torch.randn(hidden_dim, num_experts, device="mps")

        for top_k in [1, 2, 4]:
            ids, probs = moe_router_topk(hidden, router_weights, top_k=top_k)
            assert ids.shape == (batch, top_k)
            assert probs.shape == (batch, top_k)
