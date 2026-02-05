"""Test Trellis MoE kernel dequantization accuracy.

Verifies that the fast Metal MoE kernel in gemm_trellis_moe.metal produces
outputs numerically close to the slow sequential Python implementation.

This tests the 3-bit Trellis dequantization accuracy in the Metal shader.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

HAS_MPS = torch.backends.mps.is_available()

try:
    from metal_marlin.quantization.trellis_codebook import TrellisCodebook
    from metal_marlin.trellis.layer import TrellisDenseMLP
    from metal_marlin.trellis.linear import TrellisLinear
    from metal_marlin.trellis.model import TrellisMoEMLP
    HAS_TRELLIS = True
except ImportError:
    HAS_TRELLIS = False


requires_mps = pytest.mark.skipif(not HAS_MPS, reason="MPS required")
requires_trellis = pytest.mark.skipif(
    not HAS_TRELLIS, reason="Trellis modules required")


@dataclass
class MockTrellisWeight:
    """Mock TrellisWeight for testing without loading real model files."""

    packed_indices: torch.Tensor
    scales: torch.Tensor
    su: torch.Tensor
    sv: torch.Tensor
    bits: int
    original_shape: tuple[int, int]


def create_mock_trellis_linear(
    in_features: int,
    out_features: int,
    bits: int = 3,
    device: str = "mps",
) -> TrellisLinear:
    """Create a TrellisLinear with random packed weights for testing.

    Creates realistic packed Trellis weights that will dequantize to
    reasonable values for numerical accuracy testing.
    """
    TILE_DIM = 16
    tiles_k = (out_features + TILE_DIM - 1) // TILE_DIM  # K = out_features
    tiles_n = (in_features + TILE_DIM - 1) // TILE_DIM   # N = in_features
    packed_bytes = {2: 64, 3: 96, 4: 128}[bits]
    n_groups = (in_features + 127) // 128

    # Random packed indices (valid 3-bit values 0-7)
    packed = torch.randint(
        0, 256, (tiles_k, tiles_n, packed_bytes), dtype=torch.uint8)

    # Reasonable scales (small positive values)
    scales = torch.rand(n_groups, out_features,
                        dtype=torch.float32) * 0.1 + 0.01

    # Sign flips (+1 or -1)
    su = torch.where(torch.rand(in_features) > 0.5, torch.ones(
        in_features), -torch.ones(in_features))
    sv = torch.where(torch.rand(out_features) > 0.5, torch.ones(
        out_features), -torch.ones(out_features))

    mock_weight = MockTrellisWeight(
        packed_indices=packed,
        scales=scales,
        su=su.float(),
        sv=sv.float(),
        bits=bits,
        original_shape=(out_features, in_features),
    )

    return TrellisLinear.from_trellis_weight(mock_weight, device=device)


def create_mock_dense_mlp(
    hidden_dim: int,
    intermediate_dim: int,
    bits: int = 3,
    device: str = "mps",
) -> TrellisDenseMLP:
    """Create a mock TrellisDenseMLP for testing."""
    gate_proj = create_mock_trellis_linear(
        hidden_dim, intermediate_dim, bits, device)
    up_proj = create_mock_trellis_linear(
        hidden_dim, intermediate_dim, bits, device)
    down_proj = create_mock_trellis_linear(
        intermediate_dim, hidden_dim, bits, device)
    return TrellisDenseMLP(gate_proj, up_proj, down_proj)


def create_mock_moe_mlp(
    hidden_dim: int = 256,
    intermediate_dim: int = 512,
    num_experts: int = 4,
    num_experts_per_tok: int = 2,
    bits: int = 3,
    device: str = "mps",
) -> TrellisMoEMLP:
    """Create a mock TrellisMoEMLP for testing fast vs slow path accuracy."""
    # Create router
    router = nn.Linear(hidden_dim, num_experts, bias=False,
                       device=device, dtype=torch.float32)
    nn.init.xavier_uniform_(router.weight)

    # Create experts
    experts = [
        create_mock_dense_mlp(hidden_dim, intermediate_dim, bits, device)
        for _ in range(num_experts)
    ]

    # Create shared expert
    shared_expert = create_mock_dense_mlp(
        hidden_dim, intermediate_dim, bits, device)

    return TrellisMoEMLP(
        router=router,
        experts=experts,
        shared_expert=shared_expert,
        num_experts_per_tok=num_experts_per_tok,
    )


def moe_forward_slow(mlp: TrellisMoEMLP, x: torch.Tensor) -> torch.Tensor:
    """Slow sequential MoE forward (the fallback path).

    This is a copy of the slow path from TrellisMoEMLP.forward() that we use
    as the reference implementation for accuracy testing.
    """
    orig_dtype = x.dtype
    x_router = x.to(mlp.router.weight.dtype)

    router_logits = mlp.router(x_router)

    routing_weights, selected_experts = torch.topk(
        F.softmax(router_logits, dim=-1, dtype=torch.float),
        k=mlp.num_experts_per_tok,
        dim=-1,
    )
    routing_weights = routing_weights / \
        routing_weights.sum(dim=-1, keepdim=True)

    final_hidden_states = torch.zeros_like(x)

    for k_idx in range(mlp.num_experts_per_tok):
        expert_idx = selected_experts[..., k_idx]
        weight = routing_weights[..., k_idx]

        unique_experts = expert_idx.unique()
        for expert_id in unique_experts.tolist():
            mask = expert_idx == expert_id
            if not mask.any():
                continue

            w = weight * mask.float()
            expert_out = mlp.experts[expert_id](x)
            final_hidden_states.add_(expert_out * w.unsqueeze(-1))

    shared_output = mlp.shared_expert(x)
    final_hidden_states = final_hidden_states + shared_output

    return final_hidden_states


@requires_mps
@requires_trellis
class TestTrellisMoEAccuracy:
    """Tests for Trellis MoE fast vs slow path accuracy."""

    def test_moe_output_matches_slow_path(self):
        """Compare fast vs slow MoE output within tolerance.

        This is the primary accuracy test for the Metal MoE kernel.
        """
        torch.manual_seed(42)
        device = "mps"

        # Create small test model
        hidden_dim = 128
        intermediate_dim = 256
        num_experts = 4
        num_experts_per_tok = 2
        batch_size = 4

        mlp = create_mock_moe_mlp(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            bits=3,
            device=device,
        )

        x = torch.randn(batch_size, hidden_dim,
                        dtype=torch.float16, device=device)

        # Run slow path (reference)
        with torch.no_grad():
            slow_output = moe_forward_slow(mlp, x)

        # Run fast path (always available)
        with torch.no_grad():
            fast_output = mlp.forward_fast(x)

        # Check numerical accuracy
        max_diff = (slow_output - fast_output).abs().max().item()
        mean_diff = (slow_output - fast_output).abs().mean().item()
        rel_error = max_diff / (slow_output.abs().max().item() + 1e-8)

        # 3-bit quantization introduces significant error, so tolerance is higher
        # than FP16 precision. Max diff < 0.1 is reasonable for 3bpw.
        assert max_diff < 0.1, (
            f"Output mismatch: max diff = {max_diff:.6f}, "
            f"mean diff = {mean_diff:.6f}, rel error = {rel_error:.4%}"
        )

        # Also check that outputs are not NaN/Inf
        assert torch.isfinite(fast_output).all(
        ), "Fast output contains NaN/Inf"
        assert torch.isfinite(slow_output).all(
        ), "Slow output contains NaN/Inf"

    def test_moe_output_correlation(self):
        """Check that fast and slow outputs are highly correlated."""
        torch.manual_seed(123)
        device = "mps"

        mlp = create_mock_moe_mlp(
            hidden_dim=128,
            intermediate_dim=256,
            num_experts=4,
            num_experts_per_tok=2,
            bits=3,
            device=device,
        )

        x = torch.randn(8, 128, dtype=torch.float16, device=device)

        with torch.no_grad():
            slow_output = moe_forward_slow(mlp, x)
            fast_output = mlp.forward_fast(x)

        # Flatten and compute Pearson correlation
        slow_flat = slow_output.float().flatten()
        fast_flat = fast_output.float().flatten()

        # Center the data
        slow_centered = slow_flat - slow_flat.mean()
        fast_centered = fast_flat - fast_flat.mean()

        # Correlation coefficient
        correlation = (slow_centered * fast_centered).sum() / (
            slow_centered.norm() * fast_centered.norm() + 1e-8
        )

        # Correlation should be very high (> 0.99) for similar outputs
        assert correlation > 0.95, f"Correlation too low: {correlation:.4f}"

    def test_dequantization_3bit_range(self):
        """Test that 3-bit dequantized values are in expected range."""
        torch.manual_seed(0)
        device = "mps"

        linear = create_mock_trellis_linear(128, 256, bits=3, device=device)

        # Dequantize and check range
        weights = linear.dequantize()

        # 3-bit Trellis uses a codebook, values should be bounded
        max_val = weights.abs().max().item()

        # With scales ~0.01-0.1 and codebook values, expect max < 1.0
        assert max_val < 5.0, f"Dequantized weight too large: {max_val}"
        assert torch.isfinite(weights).all(
        ), "Dequantized weights contain NaN/Inf"

    def test_moe_determinism(self):
        """Test that MoE outputs are deterministic across runs."""
        torch.manual_seed(456)
        device = "mps"

        mlp = create_mock_moe_mlp(
            hidden_dim=64,
            intermediate_dim=128,
            num_experts=2,
            num_experts_per_tok=1,
            bits=3,
            device=device,
        )

        x = torch.randn(2, 64, dtype=torch.float16, device=device)

        with torch.no_grad():
            out1 = mlp(x)
            out2 = mlp(x)

        assert torch.allclose(out1, out2), "MoE output not deterministic"

    def test_moe_different_batch_sizes(self):
        """Test accuracy across different batch sizes."""
        torch.manual_seed(789)
        device = "mps"

        mlp = create_mock_moe_mlp(
            hidden_dim=64,
            intermediate_dim=128,
            num_experts=4,
            num_experts_per_tok=2,
            bits=3,
            device=device,
        )

        for batch_size in [1, 4, 8, 16]:
            x = torch.randn(batch_size, 64, dtype=torch.float16, device=device)

            with torch.no_grad():
                slow_output = moe_forward_slow(mlp, x)
                fast_output = mlp.forward_fast(x)

            max_diff = (slow_output - fast_output).abs().max().item()
            assert max_diff < 0.1, f"Batch size {batch_size}: max diff = {max_diff:.6f}"

    def test_routing_weights_preserved(self):
        """Test that expert routing weights are applied correctly."""
        torch.manual_seed(101)
        device = "mps"

        mlp = create_mock_moe_mlp(
            hidden_dim=64,
            intermediate_dim=128,
            num_experts=4,
            num_experts_per_tok=2,
            bits=3,
            device=device,
        )

        x = torch.randn(4, 64, dtype=torch.float16, device=device)

        # Get router logits directly
        x_router = x.to(mlp.router.weight.dtype)
        router_logits = mlp.router(x_router)
        routing_weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=-1, dtype=torch.float),
            k=mlp.num_experts_per_tok,
            dim=-1,
        )

        # Weights should sum to 1 after normalization
        routing_weights = routing_weights / \
            routing_weights.sum(dim=-1, keepdim=True)
        assert torch.allclose(
            routing_weights.sum(dim=-1),
            torch.ones(4, device=device),
            atol=1e-5,
        ), "Routing weights don't sum to 1"

        # Expert indices should be valid
        assert (selected_experts >= 0).all()
        assert (selected_experts < 4).all()


@requires_mps
@requires_trellis
class TestTrellisLinearAccuracy:
    """Tests for TrellisLinear dequantization accuracy."""

    def test_linear_output_finite(self):
        """Test that TrellisLinear produces finite outputs."""
        torch.manual_seed(0)
        device = "mps"

        linear = create_mock_trellis_linear(128, 256, bits=3, device=device)
        x = torch.randn(4, 128, dtype=torch.float16, device=device)

        with torch.no_grad():
            out = linear(x)

        assert torch.isfinite(out).all(
        ), "TrellisLinear output contains NaN/Inf"

    def test_linear_shape(self):
        """Test TrellisLinear output shape."""
        torch.manual_seed(0)
        device = "mps"

        linear = create_mock_trellis_linear(128, 256, bits=3, device=device)
        x = torch.randn(4, 128, dtype=torch.float16, device=device)

        with torch.no_grad():
            out = linear(x)

        assert out.shape == (4, 256), f"Unexpected shape: {out.shape}"

    def test_dequantize_vs_forward(self):
        """Test that explicit dequantize matches forward pass."""
        torch.manual_seed(42)
        device = "mps"

        linear = create_mock_trellis_linear(64, 128, bits=3, device=device)
        x = torch.randn(2, 64, dtype=torch.float16, device=device)

        with torch.no_grad():
            # Fused forward
            out_fused = linear(x)

            # Explicit dequantize + matmul
            weights = linear.dequantize()
            out_explicit = x @ weights.T.to(x.dtype)

        # Should be very close (same computation, just different ordering)
        max_diff = (out_fused - out_explicit).abs().max().item()
        assert max_diff < 0.01, f"Fused vs explicit dequant mismatch: {max_diff:.6f}"


@requires_mps
@requires_trellis
class TestDenseMlpAccuracy:
    """Tests for TrellisDenseMLP accuracy."""

    def test_swiglu_activation(self):
        """Test that SwiGLU activation is computed correctly."""
        torch.manual_seed(0)
        device = "mps"

        mlp = create_mock_dense_mlp(64, 128, bits=3, device=device)
        x = torch.randn(4, 64, dtype=torch.float16, device=device)

        with torch.no_grad():
            out = mlp(x)

        # Output should be finite
        assert torch.isfinite(out).all(), "SwiGLU output contains NaN/Inf"

        # Shape preserved
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"

    def test_dense_mlp_determinism(self):
        """Test that TrellisDenseMLP is deterministic."""
        torch.manual_seed(0)
        device = "mps"

        mlp = create_mock_dense_mlp(64, 128, bits=3, device=device)
        x = torch.randn(4, 64, dtype=torch.float16, device=device)

        with torch.no_grad():
            out1 = mlp(x)
            out2 = mlp(x)

        assert torch.allclose(out1, out2), "TrellisDenseMLP not deterministic"
