"""Regression test for mixed-BPW partial grouped-buffer fallback behavior."""

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
def test_mixed_bpw_partial_group_fallback_is_tuple_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unavailable bit tuples should fallback locally, not force global fallback."""
    import metal_marlin.trellis.moe_dispatch as moe_dispatch_mod

    torch.manual_seed(2026)

    monkeypatch.setattr(TrellisMoEMLP, "_create_buffers_eagerly", lambda self: None)

    mlp = _create_mock_moe_mlp_mixed()
    mlp.shared_expert = nn.Identity()

    mlp._build_bit_group_cache()
    assert mlp._bit_group_cache is not None

    unavailable_tuple = (3, 3, 3)
    unavailable_expert_ids = {1}

    mlp._bit_group_cache = {
        bit_tuple: (
            expert_indices,
            {} if bit_tuple == unavailable_tuple else object(),
        )
        for bit_tuple, (expert_indices, _cached_buffers) in mlp._bit_group_cache.items()
    }

    monkeypatch.setattr(mlp, "_ensure_bit_group_buffers", lambda: None)
    monkeypatch.setattr(mlp, "_get_lib", lambda: object())
    monkeypatch.setattr(mlp, "_get_buffer_pool", lambda: object())
    monkeypatch.setattr(mlp, "_get_cached_buffers", lambda: object())

    def _fail_global_fallback(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("Global fallback should not run for partial tuple unavailability.")

    monkeypatch.setattr(mlp, "_forward_grouped_fallback", _fail_global_fallback)

    grouped_calls = 0
    grouped_active_tuples: list[set[tuple[int, int, int]]] = []
    fallback_calls = 0
    fallback_tuples: list[tuple[int, int, int]] = []
    fallback_expert_ids: set[int] = set()

    def _fake_grouped_dispatch(
        *,
        activations: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_probs: torch.Tensor,
        bit_group_buffers: dict[tuple[int, int, int], tuple[list[int], object]],
        **_kwargs: object,
    ) -> torch.Tensor:
        nonlocal grouped_calls
        grouped_calls += 1

        selected = [int(v) for v in expert_ids.reshape(-1).tolist()]
        available_experts = set()
        active_tuples = set()
        for bit_tuple, (expert_list, _cached) in bit_group_buffers.items():
            expert_set = set(expert_list)
            available_experts |= expert_set
            if any(expert_id in expert_set for expert_id in selected):
                active_tuples.add(bit_tuple)
        grouped_active_tuples.append(active_tuples)

        output = torch.zeros_like(activations, dtype=torch.float32)
        for row in range(expert_ids.shape[0]):
            contrib = 0.0
            for slot in range(expert_ids.shape[1]):
                expert_id = int(expert_ids[row, slot])
                if expert_id in available_experts:
                    contrib += float(expert_probs[row, slot]) * (expert_id + 1) * 0.1
            output[row] = activations[row].float() * (1.0 + contrib)
        return output.to(dtype=activations.dtype)

    def _fake_tuple_fallback(
        _lib: object,
        *,
        activations: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_probs: torch.Tensor,
        bits: tuple[int, int, int],
        **_kwargs: object,
    ) -> torch.Tensor:
        nonlocal fallback_calls
        fallback_calls += 1
        fallback_tuples.append(bits)

        nonzero = expert_probs.reshape(-1) > 0
        used_expert_ids = [int(v) for v in expert_ids.reshape(-1)[nonzero].tolist()]
        assert used_expert_ids
        assert all(expert_id in unavailable_expert_ids for expert_id in used_expert_ids)
        fallback_expert_ids.update(used_expert_ids)

        output = torch.zeros_like(activations, dtype=torch.float32)
        for row in range(expert_ids.shape[0]):
            weight = float(expert_probs[row, 0])
            expert_id = int(expert_ids[row, 0])
            output[row] = activations[row].float() * (weight * (expert_id + 1) * 0.2)
        return output.to(dtype=activations.dtype)

    monkeypatch.setattr(
        moe_dispatch_mod, "dispatch_moe_per_bit_tuple", _fake_grouped_dispatch
    )
    monkeypatch.setattr(
        moe_dispatch_mod, "dispatch_moe_trellis_swiglu", _fake_tuple_fallback
    )

    x = torch.zeros(2, mlp.hidden_dim, dtype=torch.float16)
    x[:, 0] = torch.tensor([1.0, 0.5], dtype=torch.float16)
    x[:, 1] = torch.tensor([0.25, -0.75], dtype=torch.float16)

    selected_experts = torch.tensor([[0, 1, 2], [3, 1, 2]], dtype=torch.long)
    routing_weights = torch.tensor(
        [[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]], dtype=torch.float16
    )

    output_1 = mlp._forward_grouped(x, selected_experts, routing_weights)
    output_2 = mlp._forward_grouped(x, selected_experts, routing_weights)

    available_tuples = {(2, 2, 2), (4, 4, 4)}

    # Grouped dispatch still runs for the tuples with available grouped buffers.
    assert grouped_calls == 2
    assert grouped_active_tuples[0] == available_tuples
    assert grouped_active_tuples[1] == available_tuples
    assert all(unavailable_tuple not in active for active in grouped_active_tuples)

    # Fallback dispatch runs only for the unavailable tuple, not globally.
    assert fallback_calls == 2
    assert fallback_tuples == [unavailable_tuple, unavailable_tuple]
    assert fallback_expert_ids == unavailable_expert_ids

    # Output remains shape-correct and deterministic for fixed seed/input.
    assert output_1.shape == (2, mlp.hidden_dim)
    assert output_2.shape == (2, mlp.hidden_dim)
    torch.testing.assert_close(output_1, output_2)
