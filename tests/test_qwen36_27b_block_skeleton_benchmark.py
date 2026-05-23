from __future__ import annotations

import pytest

from benchmarks.bench_qwen36_27b_block_skeleton import (
    assert_launch_budget,
    expected_dispatches_per_token,
    expected_kernel_breakdown,
    kernel_counts_from_names,
    make_summary,
    wrapper_command_buffers_from_kernel_names,
)
from metal_marlin.kernels.qwen36_27b import (
    KERNEL_ARGMAX,
    KERNEL_ATTENTION_CACHE,
    KERNEL_ATTENTION_DECODE,
    KERNEL_ATTENTION_OUT,
    KERNEL_ATTENTION_QKV,
    KERNEL_DELTANET_UPDATE,
    KERNEL_DENSE_DOWN,
    KERNEL_DENSE_GATE_UP,
    KERNEL_LINEAR_AZ,
    KERNEL_LINEAR_GATED_NORM,
    KERNEL_LINEAR_OUT,
    KERNEL_LM_HEAD,
    KERNEL_QKVB,
    KERNEL_RMSNORM,
)


def test_expected_dispatch_breakdown_matches_qwen36_wrapper_skeleton() -> None:
    assert expected_kernel_breakdown() == {
        KERNEL_RMSNORM: 129,
        KERNEL_QKVB: 48,
        KERNEL_DELTANET_UPDATE: 48,
        KERNEL_LINEAR_AZ: 48,
        KERNEL_LINEAR_GATED_NORM: 48,
        KERNEL_LINEAR_OUT: 48,
        KERNEL_ATTENTION_QKV: 16,
        KERNEL_ATTENTION_CACHE: 16,
        KERNEL_ATTENTION_DECODE: 16,
        KERNEL_ATTENTION_OUT: 16,
        KERNEL_DENSE_GATE_UP: 64,
        KERNEL_DENSE_DOWN: 64,
        KERNEL_LM_HEAD: 1,
        KERNEL_ARGMAX: 1,
    }
    assert expected_dispatches_per_token() == 563


def test_kernel_counts_from_names_is_stable_and_sorted() -> None:
    assert kernel_counts_from_names(["b", "a", "b", "c", "a", "b"]) == {
        "a": 2,
        "b": 3,
        "c": 1,
    }


def test_wrapper_command_buffer_estimate_accounts_for_linear_attention_batch() -> None:
    assert wrapper_command_buffers_from_kernel_names(
        [
            KERNEL_QKVB,
            KERNEL_DELTANET_UPDATE,
            KERNEL_LINEAR_AZ,
            KERNEL_QKVB,
            KERNEL_DELTANET_UPDATE,
            KERNEL_RMSNORM,
        ]
    ) == 4


def test_benchmark_summary_is_explicitly_not_quality_claim() -> None:
    summary = make_summary(
        manifest_path="agent_workspace/qwen36_27b/artifacts/prototype_block_layer0/manifest.json",
        runner="direct-metal-buffer",
        decode_tokens=2,
        warmup_tokens=1,
        elapsed_ms=25.0,
        kernel_names=[
            KERNEL_RMSNORM,
            KERNEL_QKVB,
            KERNEL_DELTANET_UPDATE,
            KERNEL_RMSNORM,
        ],
        command_buffers=2,
    )

    assert summary.runner == "direct-metal-buffer"
    assert summary.decode_tok_per_s == 80.0
    assert summary.dispatch_count == 4
    assert summary.dispatches_per_token == 2.0
    assert summary.command_buffers == 2
    assert summary.command_buffers_per_token == 1.0
    assert summary.expected_dispatches_per_token == 563
    assert summary.kernel_counts == {
        KERNEL_DELTANET_UPDATE: 1,
        KERNEL_QKVB: 1,
        KERNEL_RMSNORM: 2,
    }
    assert summary.template_weight_reuse is True
    assert summary.quality_claim is False
    assert "perplexity" in summary.notes


def test_launch_budget_gate_accepts_current_direct_skeleton() -> None:
    summary = make_summary(
        manifest_path=None,
        runner="direct-metal-buffer",
        decode_tokens=1,
        warmup_tokens=0,
        elapsed_ms=1.0,
        kernel_names=[KERNEL_RMSNORM, KERNEL_QKVB, KERNEL_DELTANET_UPDATE],
        command_buffers=1,
    )

    assert_launch_budget(
        summary,
        max_dispatches_per_token=3,
        max_command_buffers_per_token=1,
    )


def test_launch_budget_gate_rejects_command_buffer_regression() -> None:
    summary = make_summary(
        manifest_path=None,
        runner="python-wrapper",
        decode_tokens=1,
        warmup_tokens=0,
        elapsed_ms=1.0,
        kernel_names=[KERNEL_RMSNORM, KERNEL_QKVB, KERNEL_DELTANET_UPDATE],
        command_buffers=3,
    )

    with pytest.raises(ValueError, match="command-buffer budget exceeded"):
        assert_launch_budget(summary, max_command_buffers_per_token=1)
