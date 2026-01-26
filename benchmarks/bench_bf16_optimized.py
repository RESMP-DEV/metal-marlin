#!/usr/bin/env python3
"""Comprehensive BF16 optimization benchmark suite.

Covers:
1) GEMM path comparison (old vs optimized)
2) Attention path comparison (old vs FP32 attention)
3) MoE dispatch comparison (old vs optimized)
4) End-to-end inference (GLM-4.7-Flash or similar)
5) Baseline comparison with percent improvements
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Ensure metal_marlin is importable from project layout
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch  # noqa: E402

try:
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch optional
    F = None


BASELINE_PATH = Path(__file__).parent / "baseline_bf16.json"
RESULTS_DIR = Path(__file__).parent / "results"


def _require_torch(feature: str) -> None:
    if not HAS_TORCH or torch is None:
        raise RuntimeError(f"PyTorch is required for {feature}.")


def _require_mps(feature: str) -> None:
    _require_torch(feature)
    if not HAS_MPS:
        raise RuntimeError("PyTorch MPS backend is required for this benchmark.")


def _mps_sync() -> None:
    if torch is not None:
        torch.mps.synchronize()


def _mps_memory() -> dict[str, int]:
    if not HAS_TORCH or torch is None:
        return {"current": 0, "driver": 0}
    if not torch.backends.mps.is_available():
        return {"current": 0, "driver": 0}
    current = int(torch.mps.current_allocated_memory()) if hasattr(
        torch.mps, "current_allocated_memory"
    ) else 0
    driver = int(torch.mps.driver_allocated_memory()) if hasattr(
        torch.mps, "driver_allocated_memory"
    ) else current
    if driver == 0:
        driver = current
    return {"current": current, "driver": driver}


def _time_stage(fn) -> tuple[float, Any]:
    _mps_sync()
    start = time.perf_counter()
    out = fn()
    _mps_sync()
    return (time.perf_counter() - start) * 1000.0, out


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _gemm_flops(M: int, N: int, K: int) -> float:
    return 2.0 * M * N * K


def _gemm_bytes_old(M: int, N: int, K: int) -> int:
    a = M * K
    b = K * N
    c = M * N
    # Old path bytes per element: 14 bytes for A/B/C (see README in script)
    return int(14 * (a + b + c))


def _gemm_bytes_new(M: int, N: int, K: int) -> int:
    a = M * K
    b = K * N
    c = M * N
    # New path bytes per element: 10 bytes for A/B/C
    return int(10 * (a + b + c))


def _benchmark_gemm(
    *,
    M: int,
    N: int,
    K: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    _require_mps("GEMM benchmarking")
    assert torch is not None

    A = torch.randn(M, K, dtype=torch.bfloat16, device="mps")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="mps")

    def _run_old() -> dict[str, float]:
        t1, (a_fp32, b_fp32) = _time_stage(lambda: (A.float(), B.float()))
        t2, (a_fp16, b_fp16) = _time_stage(lambda: (a_fp32.half(), b_fp32.half()))
        t3, c_fp16 = _time_stage(lambda: a_fp16 @ b_fp16)
        t4, c_fp32 = _time_stage(lambda: c_fp16.float())
        t5, _ = _time_stage(lambda: c_fp32.to(torch.bfloat16))
        return {
            "bf16_to_fp32_ms": t1,
            "fp32_to_fp16_ms": t2,
            "gemm_fp16_ms": t3,
            "fp16_to_fp32_ms": t4,
            "fp32_to_bf16_ms": t5,
        }

    def _run_new() -> dict[str, float]:
        t1, (a_fp32, b_fp32) = _time_stage(lambda: (A.float(), B.float()))
        t2, c_fp32 = _time_stage(lambda: a_fp32 @ b_fp32)
        t3, _ = _time_stage(lambda: c_fp32.to(torch.bfloat16))
        return {
            "bf16_to_fp32_ms": t1,
            "gemm_fp32_ms": t2,
            "fp32_to_bf16_ms": t3,
        }

    # Warmup
    for _ in range(warmup):
        _run_old()
        _run_new()

    old_samples: list[dict[str, float]] = []
    new_samples: list[dict[str, float]] = []

    for _ in range(iterations):
        old_samples.append(_run_old())
        new_samples.append(_run_new())

    def _summarize(samples: list[dict[str, float]], is_old: bool) -> dict[str, Any]:
        keys = samples[0].keys()
        means = {k: _mean([s[k] for s in samples]) for k in keys}
        total_ms = sum(means.values())
        flops = _gemm_flops(M, N, K)
        tflops = (flops / (total_ms / 1000.0)) / 1e12 if total_ms > 0 else 0.0
        gemm_key = "gemm_fp16_ms" if is_old else "gemm_fp32_ms"
        gemm_ms = means.get(gemm_key, 0.0)
        gemm_tflops = (flops / (gemm_ms / 1000.0)) / 1e12 if gemm_ms > 0 else 0.0
        bytes_moved = _gemm_bytes_old(M, N, K) if is_old else _gemm_bytes_new(M, N, K)
        memory_gb_s = (bytes_moved / (total_ms / 1000.0)) / 1e9 if total_ms > 0 else 0.0
        return {
            "total_ms": total_ms,
            "tflops": tflops,
            "gemm_tflops": gemm_tflops,
            "memory_gb_s": memory_gb_s,
            "breakdown_ms": means,
        }

    return {
        "shape": {"M": M, "N": N, "K": K},
        "old": _summarize(old_samples, is_old=True),
        "new": _summarize(new_samples, is_old=False),
    }


def _benchmark_attention(
    *,
    seq_len: int,
    head_dim: int,
    heads: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    _require_mps("attention benchmarking")
    if F is None:
        raise RuntimeError("torch.nn.functional is required for attention benchmarking.")
    assert torch is not None

    batch = 1
    q_bf16 = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="mps")
    k_bf16 = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="mps")
    v_bf16 = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="mps")
    scale = 1.0 / math.sqrt(head_dim)

    def _run_old() -> float:
        q_fp16 = q_bf16.float().half()
        k_fp16 = k_bf16.float().half()
        v_fp16 = v_bf16.float().half()
        _ = F.scaled_dot_product_attention(q_fp16, k_fp16, v_fp16, scale=scale)
        return 0.0

    def _run_new() -> float:
        q_fp32 = q_bf16.float()
        k_fp32 = k_bf16.float()
        v_fp32 = v_bf16.float()
        _ = F.scaled_dot_product_attention(q_fp32, k_fp32, v_fp32, scale=scale)
        return 0.0

    for _ in range(warmup):
        _run_old()
        _mps_sync()
        _run_new()
        _mps_sync()

    old_times: list[float] = []
    new_times: list[float] = []

    baseline_mem = _mps_memory()["driver"]
    old_peak = baseline_mem
    for _ in range(iterations):
        t0 = time.perf_counter()
        _run_old()
        _mps_sync()
        old_times.append((time.perf_counter() - t0) * 1000.0)
        old_peak = max(old_peak, _mps_memory()["driver"])

    old_mem = max(0, old_peak - baseline_mem)

    baseline_mem = _mps_memory()["driver"]
    new_peak = baseline_mem
    for _ in range(iterations):
        t0 = time.perf_counter()
        _run_new()
        _mps_sync()
        new_times.append((time.perf_counter() - t0) * 1000.0)
        new_peak = max(new_peak, _mps_memory()["driver"])

    new_mem = max(0, new_peak - baseline_mem)

    tokens = batch * seq_len
    old_mean_ms = _mean(old_times)
    new_mean_ms = _mean(new_times)

    return {
        "shape": {"batch": batch, "heads": heads, "seq_len": seq_len, "head_dim": head_dim},
        "old": {
            "time_ms": old_mean_ms,
            "tokens_per_sec": (tokens / (old_mean_ms / 1000.0)) if old_mean_ms > 0 else 0.0,
            "memory_mb": old_mem / (1024 * 1024),
        },
        "new": {
            "time_ms": new_mean_ms,
            "tokens_per_sec": (tokens / (new_mean_ms / 1000.0)) if new_mean_ms > 0 else 0.0,
            "memory_mb": new_mem / (1024 * 1024),
        },
    }


def _benchmark_moe(
    *,
    num_experts: int,
    active_experts: int,
    tokens: int,
    hidden_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    _require_mps("MoE benchmarking")
    assert torch is not None

    tokens_per_expert = tokens // active_experts
    total_tokens = tokens_per_expert * active_experts

    x = torch.randn(total_tokens, hidden_size, dtype=torch.bfloat16, device="mps")
    # Only materialize active expert weights to keep memory bounded
    weights = torch.randn(
        active_experts, hidden_size, hidden_size, dtype=torch.bfloat16, device="mps"
    )

    router = torch.arange(active_experts, device="mps").repeat_interleave(tokens_per_expert)

    def _run_old() -> tuple[float, float]:
        dispatch_ms = 0.0
        compute_ms = 0.0
        for expert in range(active_experts):
            t0 = time.perf_counter()
            idx = (router == expert).nonzero(as_tuple=False).squeeze(-1)
            x_e = x.index_select(0, idx)
            _mps_sync()
            dispatch_ms += (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            _ = x_e @ weights[expert]
            _mps_sync()
            compute_ms += (time.perf_counter() - t0) * 1000.0
        return dispatch_ms, compute_ms

    def _run_new() -> tuple[float, float]:
        t0 = time.perf_counter()
        order = torch.argsort(router)
        x_sorted = x.index_select(0, order).view(active_experts, tokens_per_expert, hidden_size)
        _mps_sync()
        dispatch_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        _ = torch.bmm(x_sorted, weights)
        _mps_sync()
        compute_ms = (time.perf_counter() - t0) * 1000.0
        return dispatch_ms, compute_ms

    for _ in range(warmup):
        _run_old()
        _run_new()

    old_dispatch: list[float] = []
    old_compute: list[float] = []
    new_dispatch: list[float] = []
    new_compute: list[float] = []

    for _ in range(iterations):
        d_ms, c_ms = _run_old()
        old_dispatch.append(d_ms)
        old_compute.append(c_ms)

        d_ms, c_ms = _run_new()
        new_dispatch.append(d_ms)
        new_compute.append(c_ms)

    return {
        "shape": {
            "num_experts": num_experts,
            "active_experts": active_experts,
            "tokens": total_tokens,
            "hidden_size": hidden_size,
        },
        "old": {
            "dispatch_overhead_ms": _mean(old_dispatch),
            "expert_compute_ms": _mean(old_compute),
        },
        "new": {
            "dispatch_overhead_ms": _mean(new_dispatch),
            "expert_compute_ms": _mean(new_compute),
        },
    }


class MemoryTracker:
    def __init__(self) -> None:
        self.peak_bytes = 0

    def update(self) -> None:
        mem = _mps_memory()["driver"]
        self.peak_bytes = max(self.peak_bytes, mem)

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 * 1024)


def _benchmark_inference(
    *,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    runs: int,
) -> dict[str, Any]:
    _require_mps("inference benchmarking")
    assert torch is not None

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("transformers is required for inference benchmark") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="mps",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(model.device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)

    prompt_tokens = int(input_ids.shape[1])
    memory = MemoryTracker()

    # Warmup
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=1, do_sample=False)
    _mps_sync()

    first_token_times: list[float] = []
    decode_times: list[float] = []
    output_lengths: list[int] = []

    with torch.no_grad():
        for _ in range(runs):
            _mps_sync()
            t0 = time.perf_counter()
            _ = model.generate(input_ids, max_new_tokens=1, do_sample=False)
            _mps_sync()
            first_token_times.append((time.perf_counter() - t0) * 1000.0)
            memory.update()

            _mps_sync()
            t0 = time.perf_counter()
            output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
            _mps_sync()
            decode_times.append(time.perf_counter() - t0)
            output_lengths.append(int(output.shape[1] - input_ids.shape[1]))
            memory.update()

    avg_first_token_ms = _mean(first_token_times)
    avg_decode_s = _mean(decode_times)
    avg_output_tokens = int(_mean([float(v) for v in output_lengths]))
    tokens_per_sec = avg_output_tokens / avg_decode_s if avg_decode_s > 0 else 0.0

    return {
        "model_id": model_id,
        "prompt_tokens": prompt_tokens,
        "output_tokens": avg_output_tokens,
        "tokens_per_sec": tokens_per_sec,
        "time_to_first_token_ms": avg_first_token_ms,
        "memory_peak_mb": memory.peak_mb,
    }


def _load_baseline(path: Path) -> dict[str, Any]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _get_path(obj: dict[str, Any], path: str) -> float | None:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def _percent_improvement(new_val: float, base_val: float, higher_is_better: bool) -> float | None:
    if base_val <= 0:
        return None
    if higher_is_better:
        return (new_val - base_val) / base_val * 100.0
    return (base_val - new_val) / base_val * 100.0


def _compare_to_baseline(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    metrics = [
        ("gemm.old.tflops", True),
        ("gemm.new.tflops", True),
        ("attention.old.tokens_per_sec", True),
        ("attention.new.tokens_per_sec", True),
        ("moe.old.expert_compute_ms", False),
        ("moe.old.dispatch_overhead_ms", False),
        ("moe.new.expert_compute_ms", False),
        ("moe.new.dispatch_overhead_ms", False),
        ("inference.tokens_per_sec", True),
        ("inference.time_to_first_token_ms", False),
        ("inference.memory_peak_mb", False),
    ]
    improvements: dict[str, float] = {}
    for key, higher_is_better in metrics:
        cur = _get_path(current, key)
        base = _get_path(baseline, key)
        if cur is None or base is None:
            continue
        delta = _percent_improvement(cur, base, higher_is_better)
        if delta is not None:
            improvements[key] = delta
    return improvements


def _print_report(results: dict[str, Any]) -> None:
    gemm = results["gemm"]
    attention = results["attention"]
    moe = results["moe"]
    inference = results["inference"]

    print("=" * 80)
    print("BF16 Optimization Benchmark")
    print("=" * 80)

    print("\nGEMM 4096x4096x4096")
    print(f"  Old: {gemm['old']['total_ms']:.3f} ms, {gemm['old']['tflops']:.2f} TFLOPS, {gemm['old']['memory_gb_s']:.1f} GB/s")
    print(f"  New: {gemm['new']['total_ms']:.3f} ms, {gemm['new']['tflops']:.2f} TFLOPS, {gemm['new']['memory_gb_s']:.1f} GB/s")
    print("  Breakdown (old):")
    for k, v in gemm["old"]["breakdown_ms"].items():
        print(f"    {k}: {v:.3f} ms")
    print("  Breakdown (new):")
    for k, v in gemm["new"]["breakdown_ms"].items():
        print(f"    {k}: {v:.3f} ms")

    print("\nAttention (seq_len=2048, head_dim=128)")
    print(f"  Old: {attention['old']['tokens_per_sec']:.1f} tok/s, {attention['old']['memory_mb']:.1f} MB")
    print(f"  New: {attention['new']['tokens_per_sec']:.1f} tok/s, {attention['new']['memory_mb']:.1f} MB")

    print("\nMoE (64 experts, 4 active)")
    print(f"  Old: compute={moe['old']['expert_compute_ms']:.2f} ms, dispatch={moe['old']['dispatch_overhead_ms']:.2f} ms")
    print(f"  New: compute={moe['new']['expert_compute_ms']:.2f} ms, dispatch={moe['new']['dispatch_overhead_ms']:.2f} ms")

    print("\nEnd-to-end inference")
    print(f"  Model: {inference['model_id']}")
    print(f"  Tokens/sec: {inference['tokens_per_sec']:.1f}")
    print(f"  Time to first token: {inference['time_to_first_token_ms']:.2f} ms")
    print(f"  Memory peak: {inference['memory_peak_mb']:.1f} MB")

    improvements = results.get("improvements")
    if improvements:
        print("\nImprovements vs baseline")
        for key, delta in improvements.items():
            print(f"  {key}: {delta:+.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="BF16 optimization benchmark suite")
    parser.add_argument("--model", default="zai-org/GLM-4.7-Flash", help="HF model ID")
    parser.add_argument("--prompt", default="Explain transformer attention.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--moe-tokens", type=int, default=2048)
    parser.add_argument("--moe-hidden", type=int, default=2048)
    parser.add_argument("--record-baseline", action="store_true")
    args = parser.parse_args()

    results = {
        "meta": {
            "device": "mps",
            "torch_version": getattr(torch, "__version__", "unknown") if torch is not None else "none",
        },
        "gemm": _benchmark_gemm(
            M=4096,
            N=4096,
            K=4096,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "attention": _benchmark_attention(
            seq_len=2048,
            head_dim=128,
            heads=32,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "moe": _benchmark_moe(
            num_experts=64,
            active_experts=4,
            tokens=args.moe_tokens,
            hidden_size=args.moe_hidden,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "inference": _benchmark_inference(
            model_id=args.model,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            runs=args.runs,
        ),
    }

    baseline = _load_baseline(BASELINE_PATH)
    if baseline:
        results["improvements"] = _compare_to_baseline(results, baseline)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _save_json(RESULTS_DIR / "bf16_optimized_results.json", results)

    if args.record_baseline:
        _save_json(BASELINE_PATH, results)

    _print_report(results)


if __name__ == "__main__":
    main()
