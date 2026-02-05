"""Tests for grouped dispatch using per-bit-tuple caching."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

HAS_MPS = torch.backends.mps.is_available()

try:
    from metal_marlin.trellis.layer import TrellisDenseMLP
    from metal_marlin.trellis.linear import TrellisLinear
    from metal_marlin.trellis.model import TrellisMoEMLP
    from metal_marlin.trellis.moe_dispatch import CachedWeightBuffers

    HAS_TRELLIS = True
except ImportError:
    HAS_TRELLIS = False


requires_mps = pytest.mark.skipif(not HAS_MPS, reason="MPS required")
requires_trellis = pytest.mark.skipif(
    not HAS_TRELLIS, reason="Trellis modules required"
)


@dataclass
class MockTrellisWeight:
    """Mock TrellisWeight for testing without loading model files."""

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
    """Create a TrellisLinear with random packed weights for testing."""
    tile_dim = 16
    tiles_k = (out_features + tile_dim - 1) // tile_dim
    tiles_n = (in_features + tile_dim - 1) // tile_dim
    packed_bytes = {2: 64, 3: 96, 4: 128}[bits]
    n_groups = (in_features + 127) // 128

    packed = torch.randint(
        0, 256, (tiles_k, tiles_n, packed_bytes), dtype=torch.uint8
    )
    scales = torch.rand(n_groups, out_features, dtype=torch.float32) * 0.1 + 0.01
    su = torch.where(
        torch.rand(in_features) > 0.5, torch.ones(in_features), -torch.ones(in_features)
    )
    sv = torch.where(
        torch.rand(out_features) > 0.5, torch.ones(out_features), -torch.ones(out_features)
    )

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
    gate_proj = create_mock_trellis_linear(hidden_dim, intermediate_dim, bits, device)
    up_proj = create_mock_trellis_linear(hidden_dim, intermediate_dim, bits, device)
    down_proj = create_mock_trellis_linear(intermediate_dim, hidden_dim, bits, device)
    return TrellisDenseMLP(gate_proj, up_proj, down_proj)


def create_mock_moe_mlp_mixed(
    hidden_dim: int = 64,
    intermediate_dim: int = 128,
    bits_per_expert: tuple[int, ...] = (2, 3, 2, 3),
    num_experts_per_tok: int = 2,
    device: str = "mps",
) -> TrellisMoEMLP:
    """Create a mock mixed-precision TrellisMoEMLP for grouped dispatch tests."""
    num_experts = len(bits_per_expert)
    router = nn.Linear(
        hidden_dim, num_experts, bias=False, device=device, dtype=torch.float32
    )
    nn.init.xavier_uniform_(router.weight)

    experts = [
        create_mock_dense_mlp(hidden_dim, intermediate_dim, bits, device)
        for bits in bits_per_expert
    ]
    shared_expert = create_mock_dense_mlp(
        hidden_dim, intermediate_dim, bits_per_expert[0], device
    )

    return TrellisMoEMLP(
        router=router,
        experts=experts,
        shared_expert=shared_expert,
        num_experts_per_tok=num_experts_per_tok,
    )


@requires_mps
@requires_trellis
class TestBitGroupDispatch:
    def test_bit_group_cache_creation(self):
        """Test that _bit_group_cache is created with expected structure."""
        torch.manual_seed(0)
        mlp = create_mock_moe_mlp_mixed()

        cache = mlp._bit_group_cache
        assert cache is not None

        expected = {(2, 2, 2): [0, 2], (3, 3, 3): [1, 3]}
        assert set(cache.keys()) == set(expected.keys())

        for bit_tuple, (expert_indices, cached_buffers) in cache.items():
            assert expert_indices == expected[bit_tuple]
            assert isinstance(cached_buffers, dict)
            assert "gate_weights" in cached_buffers
            assert "grid" in cached_buffers

    def test_lazy_buffer_conversion(self):
        """Test that CPU tensors convert to Metal buffers on first use."""
        torch.manual_seed(0)
        mlp = create_mock_moe_mlp_mixed()

        cache = mlp._bit_group_cache
        assert cache is not None
        assert any(
            isinstance(cached_buffers, dict)
            for _, (_, cached_buffers) in cache.items()
        )

        mlp._ensure_bit_group_buffers()

        for _, (_, cached_buffers) in mlp._bit_group_cache.items():
            assert isinstance(cached_buffers, CachedWeightBuffers)

    def test_forward_grouped_matches_fallback(self):
        """Test that grouped dispatch matches fallback output."""
        torch.manual_seed(0)
        mlp = create_mock_moe_mlp_mixed()
        x = torch.randn(2, mlp.hidden_dim, dtype=torch.float16, device="mps")

        with torch.no_grad():
            router_logits = mlp.router(x)
            routing_weights, selected_experts = torch.topk(
                torch.softmax(router_logits, dim=-1, dtype=torch.float16),
                k=mlp.num_experts_per_tok,
                dim=-1,
            )
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )

            grouped_out = mlp._forward_grouped(x, selected_experts, routing_weights)
            fallback_out = mlp._forward_grouped_fallback(
                x, selected_experts, routing_weights
            )

        torch.testing.assert_close(grouped_out, fallback_out, rtol=1e-2, atol=1e-2)
