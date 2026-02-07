"""Tests for fairway grouped-input preparation in Trellis MoE dispatch."""

from __future__ import annotations

import torch

from metal_marlin import moe_dispatch as parent_moe_dispatch
from metal_marlin.trellis import moe_dispatch as trellis_moe_dispatch


def test_prepare_fairway_grouped_inputs_cpu_fallback_outputs() -> None:
    """CPU fallback should still produce grouped-kernel-ready tensors."""
    trellis_moe_dispatch.reset_mixed_bpw_grouping_fallback_counters()

    expert_ids = torch.tensor(
        [
            [2, 0],
            [1, 2],
            [0, 1],
        ],
        dtype=torch.int64,
    )
    expert_probs = torch.tensor(
        [
            [0.7, 0.3],
            [0.8, 0.2],
            [0.6, 0.4],
        ],
        dtype=torch.float32,
    )
    num_experts = 4

    sorted_token_ids, expert_offsets, sorted_probs = (
        trellis_moe_dispatch.prepare_fairway_grouped_inputs(
            expert_ids=expert_ids,
            expert_probs=expert_probs,
            num_experts=num_experts,
        )
    )

    expected_dispatch = parent_moe_dispatch.group_tokens_by_expert_full(
        expert_ids, num_experts
    )
    expected_sorted_probs = expert_probs[
        expected_dispatch.sorted_token_indices, expected_dispatch.sorted_expert_indices
    ].to(torch.float16)

    assert sorted_token_ids.dtype == torch.int32
    assert expert_offsets.dtype == torch.int32
    assert sorted_probs.dtype == torch.float16

    assert sorted_token_ids.shape == (expert_ids.numel(),)
    assert expert_offsets.shape == (num_experts + 1,)
    assert sorted_probs.shape == (expert_ids.numel(),)

    assert torch.equal(sorted_token_ids, expected_dispatch.sorted_token_indices.to(torch.int32))
    assert torch.equal(expert_offsets, expected_dispatch.expert_offsets.to(torch.int32))
    assert torch.equal(sorted_probs, expected_sorted_probs)

    diagnostics = trellis_moe_dispatch.get_mixed_bpw_grouping_fallback_diagnostics()
    counters = diagnostics["counters"]
    assert counters["grouping_calls_total"] == 1
    assert counters["grouping_cpu_fallback_total"] == 1
    assert counters["grouping_cpu_fallback_reason_non_mps_inputs"] == 1
    assert counters.get("grouping_gpu_primary_success_total", 0) == 0
    assert diagnostics["last_fallback"]["reason_code"] == "non_mps_inputs"


def test_prepare_fairway_grouped_inputs_records_gpu_unavailable_reason(
    monkeypatch,
) -> None:
    """Missing GPU grouping entry-point should use the existing fallback reason code."""
    trellis_moe_dispatch.reset_mixed_bpw_grouping_fallback_counters()
    monkeypatch.setattr(trellis_moe_dispatch, "group_tokens_by_expert_full_gpu", None)

    expert_ids = torch.tensor([[0, 1]], dtype=torch.int64)
    expert_probs = torch.tensor([[0.55, 0.45]], dtype=torch.float32)

    trellis_moe_dispatch.prepare_fairway_grouped_inputs(
        expert_ids=expert_ids,
        expert_probs=expert_probs,
        num_experts=2,
    )

    diagnostics = trellis_moe_dispatch.get_mixed_bpw_grouping_fallback_diagnostics()
    counters = diagnostics["counters"]
    assert counters["grouping_calls_total"] == 1
    assert counters["grouping_cpu_fallback_total"] == 1
    assert counters["grouping_cpu_fallback_reason_gpu_grouping_unavailable"] == 1
    assert diagnostics["last_fallback"]["reason_code"] == "gpu_grouping_unavailable"
