"""Regression test for mixed-BPW grouped bit-tuple dispatch in decode."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

try:
    from metal_marlin.trellis.layer import TrellisDenseMLP
    from metal_marlin.trellis.linear import TrellisLinear
    from metal_marlin.trellis.model import TrellisMoEMLP

    HAS_TRELLIS = True
except ImportError:
    HAS_TRELLIS = False


@dataclass
class _MockTrellisWeight:
    packed_indices: torch.Tensor
    scales: torch.Tensor
    su: torch.Tensor
    sv: torch.Tensor
    bits: int
    original_shape: tuple[int, int]


def _create_mock_trellis_linear(
    in_features: int,
    out_features: int,
    bits: int,
    device: str,
) -> TrellisLinear:
    tile_dim = 16
    tiles_k = (out_features + tile_dim - 1) // tile_dim
    tiles_n = (in_features + tile_dim - 1) // tile_dim
    packed_bytes = {2: 64, 3: 96, 4: 128}[bits]
    n_groups = (in_features + 127) // 128

    packed = torch.randint(
        0, 256, (tiles_k, tiles_n, packed_bytes), dtype=torch.uint8, device=device
    )
    scales = torch.rand(n_groups, out_features, dtype=torch.float32, device=device)
    su = torch.where(
        torch.rand(in_features, device=device) > 0.5,
        torch.ones(in_features, device=device),
        -torch.ones(in_features, device=device),
    )
    sv = torch.where(
        torch.rand(out_features, device=device) > 0.5,
        torch.ones(out_features, device=device),
        -torch.ones(out_features, device=device),
    )

    mock_weight = _MockTrellisWeight(
        packed_indices=packed,
        scales=scales,
        su=su.float(),
        sv=sv.float(),
        bits=bits,
        original_shape=(out_features, in_features),
    )
    return TrellisLinear.from_trellis_weight(mock_weight, device=device)


def _create_mock_dense_mlp(
    hidden_dim: int,
    intermediate_dim: int,
    bits: int,
    device: str,
) -> TrellisDenseMLP:
    gate_proj = _create_mock_trellis_linear(hidden_dim, intermediate_dim, bits, device)
    up_proj = _create_mock_trellis_linear(hidden_dim, intermediate_dim, bits, device)
    down_proj = _create_mock_trellis_linear(intermediate_dim, hidden_dim, bits, device)
    return TrellisDenseMLP(gate_proj, up_proj, down_proj)


def _create_mock_moe_mlp_mixed(
    hidden_dim: int = 32,
    intermediate_dim: int = 64,
    bits_per_expert: tuple[int, ...] = (2, 3, 4, 2),
    num_experts_per_tok: int = 3,
    device: str = "cpu",
) -> TrellisMoEMLP:
    num_experts = len(bits_per_expert)
    router = nn.Linear(
        hidden_dim, num_experts, bias=False, device=device, dtype=torch.float32
    )

    experts = [
        _create_mock_dense_mlp(hidden_dim, intermediate_dim, bits, device)
        for bits in bits_per_expert
    ]
    shared_expert = _create_mock_dense_mlp(
        hidden_dim, intermediate_dim, bits_per_expert[0], device
    )

    return TrellisMoEMLP(
        router=router,
        experts=experts,
        shared_expert=shared_expert,
        num_experts_per_tok=num_experts_per_tok,
    )


@pytest.mark.skipif(not HAS_TRELLIS, reason="Trellis modules required")
def test_mixed_bpw_decode_keeps_grouped_dispatch_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mixed-BPW decode should stay on grouped bit-tuple dispatch and be deterministic."""
    import metal_marlin.trellis.moe_dispatch as moe_dispatch_mod

    torch.manual_seed(1234)

    # Avoid real Metal buffer creation; this test validates routing/dispatch control flow.
    monkeypatch.setattr(TrellisMoEMLP, "_create_buffers_eagerly", lambda self: None)

    mlp = _create_mock_moe_mlp_mixed()
    mlp.shared_expert = nn.Identity()

    # Ensure bit-group cache exists and is "ready" (non-None, non-dict buffers).
    mlp._build_bit_group_cache()
    assert mlp._bit_group_cache is not None
    mlp._bit_group_cache = {
        bit_tuple: (expert_indices, object())
        for bit_tuple, (expert_indices, _cached_buffers) in mlp._bit_group_cache.items()
    }

    # Keep test CPU-only; grouped path should still be exercised.
    monkeypatch.setattr(mlp, "_get_lib", lambda: object())
    monkeypatch.setattr(mlp, "_get_buffer_pool", lambda: object())

    fallback_calls = 0

    def _fail_fallback(*_args: object, **_kwargs: object) -> torch.Tensor:
        nonlocal fallback_calls
        fallback_calls += 1
        raise AssertionError("Regressed to mixed-BPW fallback path.")

    monkeypatch.setattr(mlp, "_forward_grouped_fallback", _fail_fallback)

    dispatch_calls = 0
    active_tuple_counts: list[int] = []

    def _fake_dispatch_moe_per_bit_tuple(
        *,
        activations: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_probs: torch.Tensor,
        bit_group_buffers: dict[tuple[int, int, int], tuple[list[int], object]],
        **_kwargs: object,
    ) -> torch.Tensor:
        nonlocal dispatch_calls
        dispatch_calls += 1

        selected = [int(v) for v in expert_ids.reshape(-1).tolist()]
        active_tuples = {
            bit_tuple
            for bit_tuple, (expert_list, _cached) in bit_group_buffers.items()
            if any(expert_id in expert_list for expert_id in selected)
        }
        active_tuple_counts.append(len(active_tuples))

        # Deterministic stand-in for grouped kernel output.
        token_scale = (expert_probs.float() * (expert_ids.float() + 1.0)).sum(
            dim=-1, keepdim=True
        )
        return (activations.float() * (1.0 + token_scale)).to(dtype=activations.dtype)

    monkeypatch.setattr(
        moe_dispatch_mod, "dispatch_moe_per_bit_tuple", _fake_dispatch_moe_per_bit_tuple
    )

    with torch.no_grad():
        mlp.router.weight.zero_()
        mlp.router.weight[:, 0] = torch.tensor([9.0, 7.0, 5.0, 3.0], dtype=torch.float32)

    x = torch.zeros(1, mlp.hidden_dim, dtype=torch.float16)
    x[0, 0] = 1.0

    def _route_decode(
        input_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = mlp.router(input_x.to(dtype=mlp.router.weight.dtype))
        routing_weights, selected_experts = torch.topk(
            torch.softmax(router_logits, dim=-1, dtype=torch.float16),
            k=mlp.num_experts_per_tok,
            dim=-1,
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        return selected_experts, routing_weights

    selected_experts_1, routing_weights_1 = _route_decode(x)
    selected_experts_2, routing_weights_2 = _route_decode(x)

    selected_tuples = {
        (
            mlp.experts[idx].gate_proj.bits,
            mlp.experts[idx].up_proj.bits,
            mlp.experts[idx].down_proj.bits,
        )
        for idx in selected_experts_1[0].tolist()
    }

    assert len(selected_tuples) >= 2

    output_1 = mlp._forward_grouped(x, selected_experts_1, routing_weights_1)
    output_2 = mlp._forward_grouped(x, selected_experts_2, routing_weights_2)

    # 1) No fallback regression.
    assert fallback_calls == 0
    # 2) Grouped dispatch remains active for mixed tuples.
    assert dispatch_calls == 2
    assert min(active_tuple_counts) >= 2
    # 3) Shape correctness + determinism for fixed seed/input.
    assert output_1.shape == (1, mlp.hidden_dim)
    assert output_2.shape == (1, mlp.hidden_dim)
    torch.testing.assert_close(selected_experts_1, selected_experts_2)
    torch.testing.assert_close(routing_weights_1, routing_weights_2)
    torch.testing.assert_close(output_1, output_2)


@pytest.mark.skipif(not HAS_TRELLIS, reason="Trellis modules required")
def test_mixed_bpw_grouped_partial_fallback_merges_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grouped path should run for available tuples and fallback only missing tuples."""
    import metal_marlin.trellis.moe_dispatch as moe_dispatch_mod

    monkeypatch.setattr(TrellisMoEMLP, "_create_buffers_eagerly", lambda self: None)
    mlp = _create_mock_moe_mlp_mixed(
        bits_per_expert=(2, 2, 3, 3),
        num_experts_per_tok=3,
        device="cpu",
    )
    mlp.shared_expert = nn.Identity()

    # Keep the cache layout stable for this control-flow test:
    # tuple (2,2,2) is available, tuple (3,3,3) is unavailable.
    mlp._bit_group_cache = {
        (2, 2, 2): ([0, 1], object()),
        (3, 3, 3): ([2, 3], None),
    }
    monkeypatch.setattr(mlp, "_ensure_bit_group_buffers", lambda: None)

    class _FakeWorkspacePool:
        def __init__(self, hidden_dim: int) -> None:
            self.hidden_dim = hidden_dim

        def get_accum_buffer(self, batch_size: int) -> torch.Tensor:
            return torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32)

        def get_output_buffer(self, batch_size: int) -> torch.Tensor:
            return torch.zeros(batch_size, self.hidden_dim, dtype=torch.float16)

    monkeypatch.setattr(
        mlp, "_get_workspace_buffer_pool", lambda: _FakeWorkspacePool(mlp.hidden_dim)
    )
    monkeypatch.setattr(mlp, "_get_lib", lambda: object())
    monkeypatch.setattr(mlp, "_get_buffer_pool", lambda: object())
    monkeypatch.setattr(mlp, "_get_cached_buffers", lambda: object())

    def _fail_full_fallback(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("Should not use full grouped fallback when some groups are ready.")

    monkeypatch.setattr(mlp, "_forward_grouped_fallback", _fail_full_fallback)

    grouped_calls: list[set[tuple[int, int, int]]] = []
    fallback_bits: list[tuple[int, int, int]] = []
    fallback_expert_ids: list[torch.Tensor] = []

    def _fake_dispatch_moe_per_bit_tuple(*_args: object, **kwargs: object) -> torch.Tensor:
        bit_group_buffers = kwargs["bit_group_buffers"]
        output_accum = kwargs["output_accum"]
        output_fp16 = kwargs["output_fp16"]

        assert isinstance(bit_group_buffers, dict)
        assert isinstance(output_accum, torch.Tensor)
        assert isinstance(output_fp16, torch.Tensor)

        grouped_calls.append(set(bit_group_buffers.keys()))
        output_accum.zero_()
        output_accum.add_(2.0)
        output_fp16.copy_(output_accum)
        return output_fp16

    def _fake_dispatch_moe_trellis_swiglu(*_args: object, **kwargs: object) -> torch.Tensor:
        activations = kwargs["activations"]
        bits = kwargs["bits"]
        expert_ids = kwargs["expert_ids"]

        assert isinstance(activations, torch.Tensor)
        assert isinstance(bits, tuple)
        assert isinstance(expert_ids, torch.Tensor)

        fallback_bits.append(bits)
        fallback_expert_ids.append(expert_ids.clone())
        return torch.full_like(activations, 3.0, dtype=torch.float16)

    monkeypatch.setattr(
        moe_dispatch_mod, "dispatch_moe_per_bit_tuple", _fake_dispatch_moe_per_bit_tuple
    )
    monkeypatch.setattr(
        moe_dispatch_mod, "dispatch_moe_trellis_swiglu", _fake_dispatch_moe_trellis_swiglu
    )

    x = torch.ones(1, mlp.hidden_dim, dtype=torch.float16)
    selected_experts = torch.tensor([[0, 2, 1]], dtype=torch.long)
    routing_weights = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float16)

    output = mlp._forward_grouped(x, selected_experts, routing_weights)

    assert grouped_calls == [{(2, 2, 2)}]
    assert fallback_bits == [(3, 3, 3)]
    assert len(fallback_expert_ids) == 1
    assert torch.equal(fallback_expert_ids[0], torch.tensor([[2]], dtype=torch.long))

    expected = torch.full_like(output, 6.0, dtype=torch.float16)
    torch.testing.assert_close(output, expected)
