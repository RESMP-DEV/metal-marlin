#!/usr/bin/env python3
"""Profile memory bandwidth utilization breakdown during inference.

Runs GLM-4.7-Flash on MPS, instruments key phases, estimates memory traffic,
and exports a JSON report with timing + roofline analysis.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from metal_marlin.inference_metal import (
    MetalAttention,
    MetalGLM47Model,
    MetalMLAAttention,
    MetalMLP,
    MetalMoELayer,
    MetalQuantizedLinear,
)
from metal_marlin.profiling.occupancy import detect_gpu


import os
import sys

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

class MemoryProfiler:
    def __init__(self) -> None:
        self.timings: dict[str, list[int]] = defaultdict(list)

    @contextmanager
    def profile(self, name: str):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            self.timings[name].append(time.perf_counter_ns() - start)


@dataclass
class PhaseTraffic:
    bytes_read: int = 0
    bytes_written: int = 0

    @property
    def total_bytes(self) -> int:
        return self.bytes_read + self.bytes_written

    def add(self, *, read_bytes: int = 0, write_bytes: int = 0) -> None:
        self.bytes_read += int(read_bytes)
        self.bytes_written += int(write_bytes)


def _tensor_bytes(obj: Any) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(item) for item in obj)
    if isinstance(obj, dict):
        return sum(_tensor_bytes(item) for item in obj.values())
    return 0


def _quantized_weight_bytes(module: MetalQuantizedLinear) -> int:
    weight_bytes = module.weight_packed.numel() * module.weight_packed.element_size()
    scale_bytes = module.scales.numel() * module.scales.element_size()
    return int(weight_bytes + scale_bytes)


def _estimate_attention_bytes(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int,
) -> tuple[int, int]:
    input_bytes = batch * seq_len * num_heads * head_dim * dtype_bytes
    qkv_bytes = 3 * input_bytes
    attn_scores = batch * num_heads * seq_len * seq_len * dtype_bytes
    output_bytes = batch * seq_len * num_heads * head_dim * dtype_bytes
    read_bytes = input_bytes + qkv_bytes + attn_scores
    write_bytes = output_bytes
    return read_bytes, write_bytes


def _estimate_mlp_bytes(
    *,
    batch: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    dtype_bytes: int,
) -> tuple[int, int]:
    input_bytes = batch * seq_len * hidden_size * dtype_bytes
    intermediate_bytes = 2 * batch * seq_len * intermediate_size * dtype_bytes
    output_bytes = batch * seq_len * hidden_size * dtype_bytes
    read_bytes = input_bytes + intermediate_bytes
    write_bytes = output_bytes
    return read_bytes, write_bytes


def _estimate_logit_bytes(
    *,
    batch: int,
    seq_len: int,
    hidden_size: int,
    vocab_size: int,
    dtype_bytes: int,
) -> tuple[int, int]:
    input_bytes = batch * seq_len * hidden_size * dtype_bytes
    output_bytes = batch * seq_len * vocab_size * dtype_bytes
    return input_bytes, output_bytes


def _infer_mlp_intermediate(module: torch.nn.Module, hidden_size: int) -> int:
    if hasattr(module, "intermediate_size"):
        return int(getattr(module, "intermediate_size"))
    if hasattr(module, "gate_proj"):
        return int(getattr(module, "gate_proj").out_features)
    if hasattr(module, "experts") and len(getattr(module, "experts")) > 0:
        expert = getattr(module, "experts")[0]
        if hasattr(expert, "gate_proj"):
            return int(getattr(expert, "gate_proj").out_features)
    return int(hidden_size)


def _wrap_module_forward(
    module: torch.nn.Module,
    *,
    phase: str,
    profiler: MemoryProfiler,
    traffic: dict[str, PhaseTraffic],
    traffic_fn: Callable[[tuple[Any, ...], Any], tuple[int, int]] | None = None,
) -> None:
    orig_forward = module.forward

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with profiler.profile(phase):
            result = orig_forward(*args, **kwargs)
        if traffic_fn is not None:
            read_bytes, write_bytes = traffic_fn(args, result)
            traffic[phase].add(read_bytes=read_bytes, write_bytes=write_bytes)
        return result

    module.forward = wrapped  # type: ignore[assignment]


def _make_kv_update_wrapper(
    kv_cache: Any,
    *,
    profiler: MemoryProfiler,
    traffic: dict[str, PhaseTraffic],
) -> None:
    orig_update = kv_cache.update

    def wrapped(layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        with profiler.profile("kv_cache_read_write"):
            k_full, v_full = orig_update(layer_idx, k_new, v_new)
        read_bytes = _tensor_bytes(k_full) + _tensor_bytes(v_full)
        write_bytes = _tensor_bytes(k_new) + _tensor_bytes(v_new)
        traffic["kv_cache_read_write"].add(read_bytes=read_bytes, write_bytes=write_bytes)
        return k_full, v_full

    kv_cache.update = wrapped  # type: ignore[assignment]


def _wrap_quantized_linear(
    module: MetalQuantizedLinear,
    *,
    profiler: MemoryProfiler,
    traffic: dict[str, PhaseTraffic],
    weight_bytes_by_phase: dict[str, int],
) -> None:
    orig_forward = module.forward
    weight_bytes = _quantized_weight_bytes(module)
    phase = getattr(module, "_profile_category", "weight_loading")

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with profiler.profile("weight_loading"):
            result = orig_forward(*args, **kwargs)
        traffic["weight_loading"].add(read_bytes=weight_bytes, write_bytes=0)
        weight_bytes_by_phase[phase] = weight_bytes_by_phase.get(phase, 0) + weight_bytes
        return result

    module.forward = wrapped  # type: ignore[assignment]


def _summarize_timings(timings: dict[str, list[int]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name, values in timings.items():
        if not values:
            continue
        ms_values = [v / 1e6 for v in values]
        sorted_ms = sorted(ms_values)
        count = float(len(ms_values))
        summary[name] = {
            "count": count,
            "total_ms": sum(ms_values),
            "mean_ms": sum(ms_values) / count,
            "p50_ms": sorted_ms[int(0.5 * (count - 1))],
            "p95_ms": sorted_ms[int(0.95 * (count - 1))],
        }
    return summary


def _roofline_bound(
    *,
    flops: float,
    bytes_moved: float,
    elapsed_ms: float,
    peak_tflops: float,
    peak_bw_gbs: float,
) -> dict[str, float | str]:
    if elapsed_ms <= 0 or bytes_moved <= 0:
        return {"arithmetic_intensity": 0.0, "tflops": 0.0, "bound": "unknown"}
    elapsed_s = elapsed_ms / 1000.0
    tflops = (flops / elapsed_s) / 1e12
    intensity = flops / bytes_moved
    ridge = (peak_tflops * 1000.0) / peak_bw_gbs
    bound = "memory" if intensity < ridge else "compute"
    return {"arithmetic_intensity": intensity, "tflops": tflops, "bound": bound}


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile memory bandwidth breakdown.")
    parser.add_argument("model_path", type=str, help="Path to quantized GLM-4.7-Flash model directory")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "results" / "memory_profile.json"),
    )
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS backend is required for this benchmark.")

    torch.manual_seed(args.seed)

    model = MetalGLM47Model.from_quantized(args.model_path)
    model.eval()

    profiler = MemoryProfiler()
    traffic: dict[str, PhaseTraffic] = defaultdict(PhaseTraffic)
    weight_bytes_by_phase: dict[str, int] = {}

    # Attach phase labels to quantized linears.
    for name, module in model.named_modules():
        if isinstance(module, MetalQuantizedLinear):
            if ".self_attn." in name or ".self_attn" in name:
                module._profile_category = "attention"  # type: ignore[attr-defined]
            elif ".mlp." in name or ".mlp" in name:
                module._profile_category = "ffn_moe"  # type: ignore[attr-defined]
            else:
                module._profile_category = "logit_projection_sampling"  # type: ignore[attr-defined]

    # Wrap key modules for timing + activation traffic.
    _wrap_module_forward(
        model.embed_tokens,
        phase="token_embedding",
        profiler=profiler,
        traffic=traffic,
        traffic_fn=lambda args, result: (0, _tensor_bytes(result)),
    )

    for layer in model.layers:
        self_attn = layer["self_attn"]
        if isinstance(self_attn, (MetalAttention, MetalMLAAttention)):
            _wrap_module_forward(
                self_attn,
                phase="attention",
                profiler=profiler,
                traffic=traffic,
                traffic_fn=lambda args, result, _self_attn=self_attn: _estimate_attention_bytes(
                    batch=args[0].shape[0],
                    seq_len=args[0].shape[1],
                    num_heads=_self_attn.num_heads,
                    head_dim=_self_attn.head_dim,
                    dtype_bytes=args[0].element_size(),
                ),
            )

        mlp = layer["mlp"]
        if isinstance(mlp, (MetalMLP, MetalMoELayer)):
            _wrap_module_forward(
                mlp,
                phase="ffn_moe",
                profiler=profiler,
                traffic=traffic,
                traffic_fn=lambda args, result, _mlp=mlp: _estimate_mlp_bytes(
                    batch=args[0].shape[0],
                    seq_len=args[0].shape[1],
                    hidden_size=args[0].shape[2],
                    intermediate_size=_infer_mlp_intermediate(_mlp, args[0].shape[2]),
                    dtype_bytes=args[0].element_size(),
                ),
            )

    _wrap_module_forward(
        model.lm_head,
        phase="logit_projection_sampling",
        profiler=profiler,
        traffic=traffic,
        traffic_fn=lambda args, result: (
            _estimate_logit_bytes(
                batch=args[0].shape[0],
                seq_len=args[0].shape[1],
                hidden_size=args[0].shape[2],
                vocab_size=result.shape[-1],
                dtype_bytes=args[0].element_size(),
            )[0]
            + model.lm_head.weight.numel() * model.lm_head.weight.element_size(),
            _estimate_logit_bytes(
                batch=args[0].shape[0],
                seq_len=args[0].shape[1],
                hidden_size=args[0].shape[2],
                vocab_size=result.shape[-1],
                dtype_bytes=args[0].element_size(),
            )[1],
        ),
    )

    # Wrap quantized linear for weight traffic + timing.
    for module in model.modules():
        if isinstance(module, MetalQuantizedLinear):
            _wrap_quantized_linear(
                module,
                profiler=profiler,
                traffic=traffic,
                weight_bytes_by_phase=weight_bytes_by_phase,
            )

    # Input setup
    input_ids = torch.randint(
        low=0,
        high=model.vocab_size,
        size=(args.batch_size, args.seq_len),
        device="mps",
        dtype=torch.long,
    )
    kv_cache = model.create_kv_cache(batch_size=args.batch_size)
    _make_kv_update_wrapper(kv_cache, profiler=profiler, traffic=traffic)

    # Prefill
    with torch.no_grad():
        logits = model(input_ids, kv_cache=kv_cache)
        kv_cache.advance(input_ids.shape[1])

        # Decode loop with explicit sampling instrumentation
        generated = input_ids.tolist()[0]
        for _ in range(args.max_new_tokens):
            next_logits = logits[:, -1, :]

            with profiler.profile("logit_projection_sampling"):
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            traffic["logit_projection_sampling"].add(
                read_bytes=_tensor_bytes(next_logits),
                write_bytes=_tensor_bytes(probs) + _tensor_bytes(next_token),
            )

            next_token_id = next_token.item()
            generated.append(next_token_id)
            logits = model(next_token, kv_cache=kv_cache)
            kv_cache.advance(1)

    summary = _summarize_timings(profiler.timings)

    base_phases = [
        "token_embedding",
        "attention",
        "ffn_moe",
        "kv_cache_read_write",
        "logit_projection_sampling",
    ]
    base_total_ms = sum(summary.get(phase, {}).get("total_ms", 0.0) for phase in base_phases)
    time_breakdown_pct = {
        phase: (summary.get(phase, {}).get("total_ms", 0.0) / base_total_ms * 100.0)
        if base_total_ms > 0
        else 0.0
        for phase in base_phases
    }

    gpu = detect_gpu()
    peak_bw = gpu.peak_bw_gbs
    peak_tflops = gpu.peak_tflops_fp16

    # Aggregate memory traffic and bandwidth.
    memory_breakdown = {}
    bandwidth_breakdown = {}
    for phase, phase_traffic in traffic.items():
        total_bytes = phase_traffic.total_bytes
        elapsed_ms = summary.get(phase, {}).get("total_ms", 0.0)
        elapsed_s = elapsed_ms / 1000.0 if elapsed_ms > 0 else 0.0
        bandwidth_gbs = (total_bytes / elapsed_s) / 1e9 if elapsed_s > 0 else 0.0
        memory_breakdown[phase] = {
            "bytes_read": phase_traffic.bytes_read,
            "bytes_written": phase_traffic.bytes_written,
            "bytes_total": total_bytes,
            "gb_total": total_bytes / 1e9,
        }
        bandwidth_breakdown[phase] = {
            "bandwidth_gbs": bandwidth_gbs,
            "bandwidth_utilization_pct": (bandwidth_gbs / peak_bw * 100.0) if peak_bw > 0 else 0.0,
        }

    overall_bytes = sum(p.total_bytes for p in traffic.values())
    overall_ms = sum(summary.get(phase, {}).get("total_ms", 0.0) for phase in base_phases)
    overall_bw = (overall_bytes / (overall_ms / 1000.0)) / 1e9 if overall_ms > 0 else 0.0

    # Roofline estimates (approximate).
    roofline = {}
    batch = args.batch_size
    seq_len = args.seq_len
    hidden = model.hidden_size
    heads = model.layers[0]["self_attn"].num_heads
    head_dim = model.layers[0]["self_attn"].head_dim
    mlp0 = model.layers[0]["mlp"]
    intermediate = _infer_mlp_intermediate(mlp0, hidden)
    vocab = model.vocab_size

    attn_flops = 4.0 * batch * seq_len * seq_len * heads * head_dim
    mlp_flops = 6.0 * batch * seq_len * hidden * intermediate
    logit_flops = 2.0 * batch * seq_len * hidden * vocab

    for phase, flops in (
        ("attention", attn_flops),
        ("ffn_moe", mlp_flops),
        ("logit_projection_sampling", logit_flops),
    ):
        phase_bytes = memory_breakdown.get(phase, {}).get("bytes_total", 0.0)
        elapsed_ms = summary.get(phase, {}).get("total_ms", 0.0)
        roofline[phase] = _roofline_bound(
            flops=flops,
            bytes_moved=phase_bytes,
            elapsed_ms=elapsed_ms,
            peak_tflops=peak_tflops,
            peak_bw_gbs=peak_bw,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "metadata": {
            "model_path": args.model_path,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "max_new_tokens": args.max_new_tokens,
            "device": "mps",
            "gpu": gpu.name,
        },
        "timings_ms": summary,
        "time_breakdown_pct": time_breakdown_pct,
        "memory_traffic": memory_breakdown,
        "bandwidth": bandwidth_breakdown,
        "overall": {
            "total_time_ms": overall_ms,
            "total_bytes": overall_bytes,
            "effective_bandwidth_gbs": overall_bw,
            "peak_bandwidth_gbs": peak_bw,
            "bandwidth_utilization_pct": (overall_bw / peak_bw * 100.0) if peak_bw > 0 else 0.0,
        },
        "roofline": roofline,
        "weight_loading": {
            "total_bytes_read": traffic["weight_loading"].bytes_read,
            "bytes_by_phase": weight_bytes_by_phase,
            "note": "Weight loading overlaps with attention/ffn/logit timing (fused kernels).",
        },
    }

    output_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
