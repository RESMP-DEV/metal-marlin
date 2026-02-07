"""Regression tests for grouped routing in dispatch_moe_trellis_swiglu hot path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from metal_marlin import moe_dispatch as parent_moe_dispatch
from metal_marlin.trellis import moe_dispatch as trellis_moe_dispatch


class _FakeCommandBuffer:
    def waitUntilCompleted(self) -> None:
        return None


class _FakeBufferPool:
    def __init__(self, hidden_dim: int) -> None:
        self.hidden_dim = hidden_dim
        self.last_activations: torch.Tensor | None = None
        self.last_expert_ids: torch.Tensor | None = None
        self.last_expert_probs: torch.Tensor | None = None

    def get_activation_buffer(self, batch_size: int, activations: torch.Tensor) -> object:
        assert activations.shape[0] == batch_size
        self.last_activations = activations.clone()
        return object()

    def get_expert_ids_buffer(
        self,
        batch_size: int,
        top_k: int,
        expert_ids: torch.Tensor,
    ) -> object:
        assert expert_ids.shape == (batch_size, top_k)
        self.last_expert_ids = expert_ids.clone()
        return object()

    def get_expert_probs_buffer(
        self,
        batch_size: int,
        top_k: int,
        expert_probs: torch.Tensor,
    ) -> object:
        assert expert_probs.shape == (batch_size, top_k)
        self.last_expert_probs = expert_probs.clone()
        return object()

    def get_output_buffer(self, batch_size: int) -> tuple[torch.Tensor, object]:
        row_values = torch.arange(1, batch_size + 1, dtype=torch.float32).unsqueeze(1)
        output_fp32 = row_values.repeat(1, self.hidden_dim)
        return output_fp32, object()

    def get_params_buffer(
        self,
        batch_size: int,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int,
        top_k: int,
        bits: int,
    ) -> object:
        assert hidden_dim == self.hidden_dim
        assert batch_size > 0
        assert intermediate_dim > 0
        assert num_experts > 0
        assert top_k == 1
        assert bits > 0
        return object()

    def get_output_fp16(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, dtype=torch.float16)


def _build_dispatch_info_no_argsort(
    expert_ids: torch.Tensor,
    num_experts: int,
) -> parent_moe_dispatch.MoEDispatchInfo:
    """Build MoEDispatchInfo grouped by expert without using torch.argsort."""
    batch_size, top_k = expert_ids.shape
    total = batch_size * top_k
    flat_expert_ids = expert_ids.reshape(-1).tolist()

    sorted_positions: list[int] = []
    expert_offsets = [0]
    for expert_id in range(num_experts):
        for flat_idx, assigned_expert in enumerate(flat_expert_ids):
            if assigned_expert == expert_id:
                sorted_positions.append(flat_idx)
        expert_offsets.append(len(sorted_positions))

    sorted_indices = torch.tensor(sorted_positions, dtype=torch.int64)
    sorted_token_indices = sorted_indices // top_k
    sorted_expert_indices = sorted_indices % top_k
    expert_offsets_tensor = torch.tensor(expert_offsets, dtype=torch.int64)

    inverse_indices = torch.empty(total, dtype=torch.int64)
    inverse_indices[sorted_indices] = torch.arange(total, dtype=torch.int64)

    return parent_moe_dispatch.MoEDispatchInfo(
        sorted_token_indices=sorted_token_indices,
        sorted_expert_indices=sorted_expert_indices,
        expert_offsets=expert_offsets_tensor,
        inverse_indices=inverse_indices,
        num_tokens=batch_size,
        top_k=top_k,
        num_experts=num_experts,
    )


@pytest.mark.parametrize("top_k", [4, 8])
def test_dispatch_moe_trellis_swiglu_hotpath_uses_grouped_routing(
    monkeypatch: pytest.MonkeyPatch,
    top_k: int,
) -> None:
    """Hot path should use grouped routing prep (no direct torch.argsort)."""
    batch_size = 3
    hidden_dim = 6
    intermediate_dim = 12
    num_experts = 16
    bits = 4
    total_assignments = batch_size * top_k

    activations = torch.arange(
        batch_size * hidden_dim, dtype=torch.float16
    ).reshape(batch_size, hidden_dim)
    expert_ids = (
        (torch.arange(total_assignments, dtype=torch.int64).reshape(batch_size, top_k) * 5) + 3
    ) % num_experts
    raw_probs = torch.arange(1, total_assignments + 1, dtype=torch.float32).reshape(
        batch_size, top_k
    )
    expert_probs = (raw_probs / raw_probs.sum(dim=-1, keepdim=True)).to(torch.float16)

    grouped_call_count = [0]

    def _fake_group_tokens_mixed_bpw_primary_gpu(
        *,
        expert_ids: torch.Tensor,
        expert_probs: torch.Tensor,
        num_experts: int,
    ) -> parent_moe_dispatch.MoEDispatchInfo:
        del expert_probs
        grouped_call_count[0] += 1
        return _build_dispatch_info_no_argsort(expert_ids.to(torch.int64), num_experts)

    monkeypatch.setattr(
        trellis_moe_dispatch,
        "_group_tokens_mixed_bpw_primary_gpu",
        _fake_group_tokens_mixed_bpw_primary_gpu,
    )
    monkeypatch.setattr(
        trellis_moe_dispatch,
        "_get_available_moe_kernels_from_inventory",
        lambda _lib: None,
    )
    monkeypatch.setattr(
        trellis_moe_dispatch,
        "select_moe_kernel",
        lambda *args, **kwargs: ("moe_trellis_swiglu", 64),
    )
    monkeypatch.setattr(
        trellis_moe_dispatch,
        "dispatch_kernel",
        lambda *args, **kwargs: _FakeCommandBuffer(),
    )
    monkeypatch.setattr(
        torch,
        "argsort",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dispatch_moe_trellis_swiglu hot path should not call torch.argsort")
        ),
    )

    buffer_pool = _FakeBufferPool(hidden_dim=hidden_dim)
    cached_buffers = trellis_moe_dispatch.CachedWeightBuffers(*([object()] * 13))
    fake_lib = SimpleNamespace(device=object(), _libraries=None, _pipelines=None)

    output = trellis_moe_dispatch.dispatch_moe_trellis_swiglu(
        lib=fake_lib,
        activations=activations,
        gate_weights=None,
        gate_scales=None,
        up_weights=None,
        up_scales=None,
        down_weights=None,
        down_scales=None,
        gate_su=None,
        gate_sv=None,
        up_su=None,
        up_sv=None,
        down_su=None,
        down_sv=None,
        grid=None,
        expert_ids=expert_ids,
        expert_probs=expert_probs,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        top_k=top_k,
        bits=bits,
        cached_buffers=cached_buffers,
        buffer_pool=buffer_pool,
    )

    assert grouped_call_count[0] == 1
    assert buffer_pool.last_activations is not None
    assert buffer_pool.last_expert_ids is not None
    assert buffer_pool.last_expert_probs is not None

    dispatch_info = _build_dispatch_info_no_argsort(expert_ids, num_experts)
    expected_sorted_activations = activations[dispatch_info.sorted_token_indices]
    expected_sorted_ids = expert_ids[
        dispatch_info.sorted_token_indices, dispatch_info.sorted_expert_indices
    ]
    expected_sorted_probs = expert_probs[
        dispatch_info.sorted_token_indices, dispatch_info.sorted_expert_indices
    ]

    assert torch.equal(buffer_pool.last_activations, expected_sorted_activations)
    assert torch.equal(buffer_pool.last_expert_ids.reshape(-1), expected_sorted_ids)
    assert torch.equal(buffer_pool.last_expert_probs.reshape(-1), expected_sorted_probs)

    expected_output = torch.zeros(batch_size, hidden_dim, dtype=torch.float16)
    row_values = torch.arange(1, total_assignments + 1, dtype=torch.float16).unsqueeze(1)
    row_values = row_values.repeat(1, hidden_dim)
    for row_idx, token_idx in enumerate(dispatch_info.sorted_token_indices.tolist()):
        expected_output[token_idx] += row_values[row_idx]

    assert output.shape == (batch_size, hidden_dim)
    assert output.dtype == torch.float16
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
