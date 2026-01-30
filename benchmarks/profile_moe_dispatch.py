#!/usr/bin/env python3
"""Profile MoE dispatch/gather time and memory traffic.

Focuses on:
- Router time (softmax + top-k)
- Dispatch time (token grouping + scatter to expert inputs)
- Expert GEMM time
- Gather time (combine expert outputs)
- Estimated memory traffic and dtype conversions
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure metal_marlin is importable from project layout
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from metal_marlin._compat import HAS_TORCH, torch  # noqa: E402
from metal_marlin.moe.token_dispatcher import (  # noqa: E402
    combine_expert_outputs,
    group_tokens_by_expert,
)


@dataclass
class TimeStats:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def _require_torch() -> None:
    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required for MoE profiling")


def _sync_device(device: str) -> None:
    _require_torch()
    assert torch is not None
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _get_device(preferred: str | None) -> str:
    _require_torch()
    assert torch is not None
    if preferred:
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _parse_dtype(name: str) -> Any:
    _require_torch()
    assert torch is not None
    name = name.lower()
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def _time_with_result(device: str, fn):
    _sync_device(device)
    start = time.perf_counter()
    result = fn()
    _sync_device(device)
    return (time.perf_counter() - start), result


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(int(len(sorted_vals) * quantile), len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def _summarize_ms(values_s: list[float]) -> TimeStats:
    if not values_s:
        return TimeStats(0.0, 0.0, 0.0, 0.0, 0.0)
    values_ms = [v * 1000.0 for v in values_s]
    return TimeStats(
        mean_ms=statistics.mean(values_ms),
        p50_ms=_percentile(values_ms, 0.50),
        p95_ms=_percentile(values_ms, 0.95),
        min_ms=min(values_ms),
        max_ms=max(values_ms),
    )


def _format_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def _router_forward(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = hidden_states @ router_weight
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)
    return top_ids, top_probs, logits, probs


def _dispatch_current(
    hidden_states: torch.Tensor,
    dispatch_info: Any,
) -> list[torch.Tensor | None]:
    tokens_per_expert: list[torch.Tensor | None] = []
    for expert_id in range(dispatch_info.num_experts):
        start = int(dispatch_info.expert_offsets[expert_id].item())
        end = int(dispatch_info.expert_offsets[expert_id + 1].item())
        if start == end:
            tokens_per_expert.append(None)
            continue
        token_indices = dispatch_info.sorted_token_indices[start:end]
        tokens_per_expert.append(hidden_states[token_indices])
    return tokens_per_expert


def _dispatch_ideal(
    hidden_states: torch.Tensor,
    dispatch_info: Any,
) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
    grouped_tokens = hidden_states[dispatch_info.sorted_token_indices]
    tokens_per_expert: list[torch.Tensor | None] = []
    for expert_id in range(dispatch_info.num_experts):
        start = int(dispatch_info.expert_offsets[expert_id].item())
        end = int(dispatch_info.expert_offsets[expert_id + 1].item())
        if start == end:
            tokens_per_expert.append(None)
            continue
        tokens_per_expert.append(grouped_tokens[start:end])
    return grouped_tokens, tokens_per_expert


def _run_experts(
    tokens_per_expert: list[torch.Tensor | None],
    weights: torch.Tensor,
    shared_weights: bool,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for expert_id, tokens in enumerate(tokens_per_expert):
        if tokens is None or tokens.numel() == 0:
            continue
        weight = weights if shared_weights else weights[expert_id]
        outputs.append(tokens @ weight)
    if not outputs:
        raise RuntimeError("No expert outputs produced")
    return torch.cat(outputs, dim=0)


def _estimate_dispatch_bytes(
    hidden_states: torch.Tensor,
    dispatch_info: Any,
    expert_ids: torch.Tensor,
) -> dict[str, int]:
    elem_bytes = hidden_states.element_size()
    total_assignments = dispatch_info.total_assignments
    token_bytes = total_assignments * hidden_states.shape[1] * elem_bytes

    # Minimum: read inputs + write gathered expert inputs
    min_bytes = 2 * token_bytes

    # Indexing + sorting overhead (estimated)
    index_bytes = 0
    index_bytes += dispatch_info.sorted_token_indices.numel() * dispatch_info.sorted_token_indices.element_size()
    index_bytes += dispatch_info.sorted_expert_slots.numel() * dispatch_info.sorted_expert_slots.element_size()
    index_bytes += dispatch_info.expert_offsets.numel() * dispatch_info.expert_offsets.element_size()
    index_bytes += dispatch_info.inverse_indices.numel() * dispatch_info.inverse_indices.element_size()

    # Transient tensors in grouping (expert_ids_flat, sort_keys, sorted_indices)
    index_bytes += expert_ids.numel() * 8  # expert_ids_flat int64
    index_bytes += expert_ids.numel() * 8  # sort_keys int64
    index_bytes += expert_ids.numel() * 8  # sorted_indices int64

    return {
        "min_bytes": int(min_bytes),
        "index_bytes_est": int(index_bytes),
        "actual_bytes_est": int(min_bytes + index_bytes),
    }


def _estimate_gather_bytes(
    expert_outputs: torch.Tensor,
    expert_probs: torch.Tensor,
    dispatch_info: Any,
) -> dict[str, int]:
    out_bytes = expert_outputs.numel() * expert_outputs.element_size()
    probs_bytes = expert_probs.numel() * expert_probs.element_size()
    combined_bytes = dispatch_info.num_tokens * expert_outputs.shape[1] * expert_outputs.element_size()

    min_bytes = out_bytes + probs_bytes + combined_bytes

    # weighted_outputs write + read, weighted_original write + read
    weighted_bytes = 2 * out_bytes
    reorder_bytes = 2 * out_bytes

    actual_bytes = out_bytes + probs_bytes + weighted_bytes + reorder_bytes + combined_bytes

    return {
        "min_bytes": int(min_bytes),
        "actual_bytes_est": int(actual_bytes),
    }


def _describe_dtype_conversions(
    hidden_states: torch.Tensor,
    logits: torch.Tensor,
    probs: torch.Tensor,
    expert_ids: torch.Tensor,
    dispatch_info: Any,
) -> list[str]:
    conversions: list[str] = []
    if logits.dtype != hidden_states.dtype:
        conversions.append(f"router logits cast {hidden_states.dtype} -> {logits.dtype}")
    if probs.dtype != logits.dtype:
        conversions.append(f"router probs cast {logits.dtype} -> {probs.dtype}")
    if expert_ids.dtype != torch.int64:
        conversions.append(f"expert_ids cast {expert_ids.dtype} -> torch.int64 for sorting")
    if dispatch_info.sorted_token_indices.dtype != torch.int64:
        conversions.append("dispatch indices stored as non-int64")
    return conversions


def _active_expert_stats(dispatch_info: Any) -> tuple[int, int]:
    counts = dispatch_info.expert_offsets[1:] - dispatch_info.expert_offsets[:-1]
    active = int((counts > 0).sum().item())
    max_load = int(counts.max().item()) if active else 0
    return active, max_load


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    _require_torch()
    assert torch is not None

    device = _get_device(args.device)
    dtype = _parse_dtype(args.dtype)

    torch.manual_seed(args.seed)

    hidden_states = torch.randn(
        args.batch_size,
        args.hidden_dim,
        device=device,
        dtype=dtype,
    )

    router_weight = torch.randn(
        args.hidden_dim,
        args.num_experts,
        device=device,
        dtype=dtype,
    )

    if args.shared_weights:
        expert_weights = torch.randn(
            args.hidden_dim,
            args.out_dim,
            device=device,
            dtype=dtype,
        )
    else:
        expert_weights = torch.randn(
            args.num_experts,
            args.hidden_dim,
            args.out_dim,
            device=device,
            dtype=dtype,
        )

    # Warmup
    for _ in range(args.warmup):
        expert_ids, expert_probs, _, _ = _router_forward(hidden_states, router_weight, args.top_k)
        dispatch_info = group_tokens_by_expert(expert_ids, args.num_experts)
        tokens_current = _dispatch_current(hidden_states, dispatch_info)
        outputs_current = _run_experts(tokens_current, expert_weights, args.shared_weights)
        _ = combine_expert_outputs(outputs_current, (expert_probs, dispatch_info))
        _sync_device(device)

    router_times: list[float] = []
    dispatch_info_times: list[float] = []
    dispatch_times: list[float] = []
    expert_times: list[float] = []
    gather_times: list[float] = []

    ideal_dispatch_times: list[float] = []
    ideal_expert_times: list[float] = []
    ideal_gather_times: list[float] = []

    last_dispatch_info = None
    last_expert_ids = None
    last_expert_probs = None
    last_logits = None
    last_probs = None
    last_expert_outputs = None

    for _ in range(args.iterations):
        router_time, (expert_ids, expert_probs, logits, probs) = _time_with_result(
            device,
            lambda: _router_forward(hidden_states, router_weight, args.top_k),
        )
        dispatch_info_time, dispatch_info = _time_with_result(
            device,
            lambda: group_tokens_by_expert(expert_ids, args.num_experts),
        )
        dispatch_time, tokens_current = _time_with_result(
            device,
            lambda: _dispatch_current(hidden_states, dispatch_info),
        )
        expert_time, expert_outputs = _time_with_result(
            device,
            lambda: _run_experts(tokens_current, expert_weights, args.shared_weights),
        )
        gather_time, combined = _time_with_result(
            device,
            lambda: combine_expert_outputs(expert_outputs, (expert_probs, dispatch_info)),
        )

        router_times.append(router_time)
        dispatch_info_times.append(dispatch_info_time)
        dispatch_times.append(dispatch_time)
        expert_times.append(expert_time)
        gather_times.append(gather_time)

        last_dispatch_info = dispatch_info
        last_expert_ids = expert_ids
        last_expert_probs = expert_probs
        last_logits = logits
        last_probs = probs
        last_expert_outputs = expert_outputs

        if args.measure_ideal:
            ideal_dispatch_time, (grouped_tokens, tokens_ideal) = _time_with_result(
                device,
                lambda: _dispatch_ideal(hidden_states, dispatch_info),
            )
            ideal_expert_time, expert_outputs_ideal = _time_with_result(
                device,
                lambda: _run_experts(tokens_ideal, expert_weights, args.shared_weights),
            )
            ideal_gather_time, _ = _time_with_result(
                device,
                lambda: combine_expert_outputs(expert_outputs_ideal, (expert_probs, dispatch_info)),
            )
            ideal_dispatch_times.append(ideal_dispatch_time)
            ideal_expert_times.append(ideal_expert_time)
            ideal_gather_times.append(ideal_gather_time)

    assert last_dispatch_info is not None
    assert last_expert_ids is not None
    assert last_expert_probs is not None
    assert last_logits is not None
    assert last_probs is not None
    assert last_expert_outputs is not None

    dispatch_bytes = _estimate_dispatch_bytes(hidden_states, last_dispatch_info, last_expert_ids)
    gather_bytes = _estimate_gather_bytes(last_expert_outputs, last_expert_probs, last_dispatch_info)

    dtype_conversions = _describe_dtype_conversions(
        hidden_states,
        last_logits,
        last_probs,
        last_expert_ids,
        last_dispatch_info,
    )

    active_experts, max_load = _active_expert_stats(last_dispatch_info)

    router_stats = _summarize_ms(router_times)
    dispatch_info_stats = _summarize_ms(dispatch_info_times)
    dispatch_stats = _summarize_ms(dispatch_times)
    expert_stats = _summarize_ms(expert_times)
    gather_stats = _summarize_ms(gather_times)

    ideal_dispatch_stats = _summarize_ms(ideal_dispatch_times)
    ideal_expert_stats = _summarize_ms(ideal_expert_times)
    ideal_gather_stats = _summarize_ms(ideal_gather_times)

    total_moe_ms = (
        router_stats.mean_ms
        + dispatch_info_stats.mean_ms
        + dispatch_stats.mean_ms
        + expert_stats.mean_ms
        + gather_stats.mean_ms
    )
    dispatch_total_ms = dispatch_info_stats.mean_ms + dispatch_stats.mean_ms

    dispatch_pct = (dispatch_total_ms / total_moe_ms * 100.0) if total_moe_ms > 0 else 0.0
    gather_pct = (gather_stats.mean_ms / total_moe_ms * 100.0) if total_moe_ms > 0 else 0.0
    expert_pct = (expert_stats.mean_ms / total_moe_ms * 100.0) if total_moe_ms > 0 else 0.0

    bytes_per_token = (
        (dispatch_bytes["actual_bytes_est"] + gather_bytes["actual_bytes_est"]) / args.batch_size
        if args.batch_size > 0
        else 0.0
    )

    return {
        "config": {
            "device": device,
            "dtype": str(dtype),
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "out_dim": args.out_dim,
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "shared_weights": args.shared_weights,
            "active_experts": active_experts,
            "max_expert_load": max_load,
        },
        "timing_ms": {
            "router": router_stats,
            "dispatch_info": dispatch_info_stats,
            "dispatch_scatter": dispatch_stats,
            "expert_gemm": expert_stats,
            "gather": gather_stats,
            "total_moe_ms": total_moe_ms,
            "dispatch_pct": dispatch_pct,
            "gather_pct": gather_pct,
            "expert_pct": expert_pct,
            "ideal_dispatch": ideal_dispatch_stats,
            "ideal_expert": ideal_expert_stats,
            "ideal_gather": ideal_gather_stats,
        },
        "memory_bytes": {
            "dispatch": dispatch_bytes,
            "gather": gather_bytes,
            "bytes_per_token": bytes_per_token,
        },
        "dtype_conversions": dtype_conversions,
    }


def _print_report(report: dict[str, Any]) -> None:
    cfg = report["config"]
    timing = report["timing_ms"]
    memory = report["memory_bytes"]

    router = timing["router"]
    dispatch_info = timing["dispatch_info"]
    dispatch_scatter = timing["dispatch_scatter"]
    expert = timing["expert_gemm"]
    gather = timing["gather"]

    print("MoE DISPATCH/GATHER PROFILE")
    print("============================")
    print(
        f"Config: device={cfg['device']} dtype={cfg['dtype']} batch={cfg['batch_size']} "
        f"hidden={cfg['hidden_dim']} out={cfg['out_dim']} experts={cfg['num_experts']} top_k={cfg['top_k']}"
    )
    print(
        f"Active experts={cfg['active_experts']} max_expert_load={cfg['max_expert_load']} "
        f"shared_weights={cfg['shared_weights']}"
    )

    print("\nTiming (ms, mean/p50/p95):")
    print(
        f"  Router: {router.mean_ms:.3f} / {router.p50_ms:.3f} / {router.p95_ms:.3f}"
    )
    print(
        f"  Dispatch info (sort): {dispatch_info.mean_ms:.3f} / {dispatch_info.p50_ms:.3f} / {dispatch_info.p95_ms:.3f}"
    )
    print(
        f"  Dispatch scatter: {dispatch_scatter.mean_ms:.3f} / {dispatch_scatter.p50_ms:.3f} / {dispatch_scatter.p95_ms:.3f}"
    )
    print(
        f"  Expert GEMM: {expert.mean_ms:.3f} / {expert.p50_ms:.3f} / {expert.p95_ms:.3f}"
    )
    print(
        f"  Gather combine: {gather.mean_ms:.3f} / {gather.p50_ms:.3f} / {gather.p95_ms:.3f}"
    )

    print("\nTime breakdown (% of total MoE time):")
    print(f"  Dispatch (info+scatter): {timing['dispatch_pct']:.2f}%")
    print(f"  Gather: {timing['gather_pct']:.2f}%")
    print(f"  Expert compute: {timing['expert_pct']:.2f}%")
    print(f"  Total MoE time: {timing['total_moe_ms']:.3f} ms")

    if timing["dispatch_pct"] > 10.0:
        print("\nOptimization potential: DISPATCH > 10% of MoE time (high priority)")
    else:
        print("\nOptimization potential: dispatch under 10% of MoE time")

    ideal_dispatch = timing["ideal_dispatch"]
    ideal_expert = timing["ideal_expert"]
    ideal_gather = timing["ideal_gather"]
    if ideal_dispatch.mean_ms > 0.0:
        print("\nCurrent vs Ideal (single gather) dispatch:")
        print(
            f"  Dispatch scatter mean: {dispatch_scatter.mean_ms:.3f} ms vs {ideal_dispatch.mean_ms:.3f} ms"
        )
        print(
            f"  Expert GEMM mean: {expert.mean_ms:.3f} ms vs {ideal_expert.mean_ms:.3f} ms"
        )
        if ideal_gather.mean_ms > 0.0:
            print(
                f"  Gather mean: {gather.mean_ms:.3f} ms vs {ideal_gather.mean_ms:.3f} ms"
            )

    print("\nMemory traffic (estimated):")
    dispatch_mem = memory["dispatch"]
    gather_mem = memory["gather"]
    print(
        f"  Dispatch min: {_format_bytes(dispatch_mem['min_bytes'])}, "
        f"actual: {_format_bytes(dispatch_mem['actual_bytes_est'])}"
    )
    print(
        f"  Dispatch index overhead: {_format_bytes(dispatch_mem['index_bytes_est'])}"
    )
    print(
        f"  Gather min: {_format_bytes(gather_mem['min_bytes'])}, "
        f"actual: {_format_bytes(gather_mem['actual_bytes_est'])}"
    )
    print(f"  Bytes/token (dispatch+gather): {_format_bytes(memory['bytes_per_token'])}")

    conversions = report["dtype_conversions"]
    print("\nDtype conversions:")
    if conversions:
        for item in conversions:
            print(f"  - {item}")
    else:
        print("  - None detected")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile MoE dispatch/gather timing and memory traffic",
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Number of tokens")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Input hidden dimension")
    parser.add_argument("--out-dim", type=int, default=1024, help="Output hidden dimension")
    parser.add_argument("--num-experts", type=int, default=64, help="Number of experts")
    parser.add_argument("--top-k", type=int, default=4, help="Top-k experts per token")
    parser.add_argument("--dtype", type=str, default="fp16", help="fp16|bf16|fp32")
    parser.add_argument("--device", type=str, default=None, help="cpu|mps|cuda")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=10, help="Timed iterations")
    parser.add_argument(
        "--shared-weights",
        action="store_true",
        default=True,
        help="Use a single shared expert weight matrix (default)",
    )
    parser.add_argument(
        "--per-expert-weights",
        action="store_false",
        dest="shared_weights",
        help="Allocate distinct weight matrices per expert",
    )
    parser.add_argument(
        "--measure-ideal",
        action="store_true",
        default=True,
        help="Also time ideal dispatch path (single gather + views)",
    )
    parser.add_argument(
        "--no-measure-ideal",
        action="store_false",
        dest="measure_ideal",
        help="Disable ideal dispatch timing",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")

    args = parser.parse_args()

    report = run_profile(args)
    _print_report(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=lambda o: o.__dict__)
        print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
