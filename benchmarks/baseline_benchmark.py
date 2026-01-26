#!/usr/bin/env python3
"""Baseline benchmark suite for GLM-4.7-Flash.

Captures end-to-end metrics, component throughput, and optional comparisons
(llama.cpp, PyTorch MPS, MLX) into a single JSON baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure metal_marlin is importable from project layout
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch  # noqa: E402
from metal_marlin.kv_cache import CacheConfig, KVCache  # noqa: E402
from metal_marlin.profiling.memory_bandwidth import (  # noqa: E402
    MemoryBandwidthProfiler,
)
from metal_marlin.profiling.occupancy import detect_gpu  # noqa: E402

try:  # Optional dependency
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None


@dataclass
class EndToEndMetrics:
    tokens_per_second: float
    ttft_ms: float
    memory_peak_gb: float
    bandwidth_util_pct: float | None
    bandwidth_gbs: float | None


class MemoryTracker:
    def __init__(self, device: str):
        self.device = device
        self.peak_bytes = 0
        if self.device == "cuda" and torch is not None:
            torch.cuda.reset_peak_memory_stats()

    def update(self) -> None:
        if torch is None:
            return
        if self.device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            current = int(torch.mps.current_allocated_memory())
        elif self.device == "cuda":
            current = int(torch.cuda.max_memory_allocated())
        else:
            current = _get_process_memory_bytes()
        self.peak_bytes = max(self.peak_bytes, current)

    @property
    def peak_gb(self) -> float:
        return self.peak_bytes / (1024 * 1024 * 1024)


def _get_process_memory_bytes() -> int:
    import resource
    import sys as _sys

    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss
    if _sys.platform == "darwin":
        return int(rss)
    return int(rss * 1024)


def _require_torch(feature: str) -> None:
    if not HAS_TORCH or torch is None:
        raise RuntimeError(f"PyTorch is required for {feature}.")


def _require_mps(feature: str) -> None:
    _require_torch(feature)
    if not HAS_MPS:
        raise RuntimeError("PyTorch MPS backend is required for this benchmark.")


def _mps_sync() -> None:
    if torch is not None and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(int(len(sorted_vals) * quantile), len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def _get_config_value(config: Any, names: list[str], default: int) -> int:
    for name in names:
        if hasattr(config, name):
            value = getattr(config, name)
            if isinstance(value, int):
                return value
    if isinstance(config, dict):
        for name in names:
            value = config.get(name)
            if isinstance(value, int):
                return value
    return default


def _estimate_kv_bytes_per_elem(dtype: str | None) -> float:
    if dtype in ("fp16", "bf16"):
        return 2.0
    if dtype == "fp8":
        return 2.0  # stored as fp16
    if dtype == "fp4":
        return 0.5
    return 2.0


def _estimate_decode_bandwidth(
    fn,
    *,
    seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_elem: float,
) -> tuple[float | None, float | None, float | None]:
    try:
        profiler = MemoryBandwidthProfiler()
    except Exception:
        return None, None, None

    bytes_read = int(2 * num_layers * num_kv_heads * head_dim * seq_len * bytes_per_elem)
    bytes_written = int(2 * num_layers * num_kv_heads * head_dim * bytes_per_elem)

    measurement = profiler.measure(
        name="decode_step",
        fn=fn,
        bytes_read=bytes_read,
        bytes_written=bytes_written,
        warmup=2,
        iterations=5,
    )
    return (
        measurement.efficiency_pct,
        measurement.total_bandwidth_gbs,
        measurement.peak_bandwidth_gbs,
    )


def _build_input_ids(tokenizer: Any, prompt: str, device: str):
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return input_ids.to(device)


def benchmark_metal_marlin(
    model_path: Path,
    prompt: str,
    num_tokens: int,
    num_runs: int,
) -> tuple[EndToEndMetrics, dict[str, Any]]:
    _require_mps("Metal Marlin inference")
    assert torch is not None

    from metal_marlin.inference import load_quantized_model

    device = "mps"
    model, tokenizer = load_quantized_model(model_path, device=device)
    model.eval()

    input_ids = _build_input_ids(tokenizer, prompt, device)
    prompt_tokens = int(input_ids.shape[1])

    memory = MemoryTracker(device)
    prefill_times: list[float] = []
    decode_times: list[float] = []
    decode_latencies_ms: list[float] = []
    first_token_latencies_ms: list[float] = []

    with torch.no_grad():
        for _ in range(num_runs):
            kv_cache = model.create_kv_cache(batch_size=1)

            _mps_sync()
            start = time.perf_counter()
            logits = model(input_ids, kv_cache=kv_cache)
            kv_cache.advance(input_ids.shape[1])
            _mps_sync()
            prefill_time = time.perf_counter() - start
            prefill_times.append(prefill_time)
            memory.update()

            next_token = torch.argmax(logits[:, -1, :], dim=-1)

            decode_start = time.perf_counter()
            for step in range(num_tokens):
                next_input = next_token.view(1, 1)
                _mps_sync()
                step_start = time.perf_counter()
                logits = model(next_input, kv_cache=kv_cache)
                kv_cache.advance(1)
                _mps_sync()
                step_time_ms = (time.perf_counter() - step_start) * 1000.0
                decode_latencies_ms.append(step_time_ms)
                if step == 0:
                    first_token_latencies_ms.append(step_time_ms)
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                memory.update()
            decode_times.append(time.perf_counter() - decode_start)

    prefill_tok_s = prompt_tokens / statistics.mean(prefill_times)
    decode_tok_s = num_tokens / statistics.mean(decode_times)
    first_token_ms = statistics.mean(first_token_latencies_ms) if first_token_latencies_ms else 0.0
    p50_ms = _percentile(decode_latencies_ms, 0.50)
    p99_ms = _percentile(decode_latencies_ms, 0.99)

    num_layers = getattr(model, "num_layers", 0) or 0
    num_kv_heads = getattr(model, "num_kv_heads", 0) or 0
    head_dim = getattr(model, "head_dim", 0) or 0
    kv_bytes_per_elem = 2.0

    def _decode_step() -> None:
        kv_cache = model.create_kv_cache(batch_size=1)
        logits = model(input_ids, kv_cache=kv_cache)
        kv_cache.advance(input_ids.shape[1])
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        _ = model(next_token.view(1, 1), kv_cache=kv_cache)
        kv_cache.advance(1)

    bw_util, bw_gbs, peak_gbs = _estimate_decode_bandwidth(
        _decode_step,
        seq_len=prompt_tokens,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bytes_per_elem=kv_bytes_per_elem,
    )

    metrics = EndToEndMetrics(
        tokens_per_second=float(decode_tok_s),
        ttft_ms=float(first_token_ms),
        memory_peak_gb=float(memory.peak_gb),
        bandwidth_util_pct=float(bw_util) if bw_util is not None else None,
        bandwidth_gbs=float(bw_gbs) if bw_gbs is not None else None,
    )

    details = {
        "prompt_tokens": prompt_tokens,
        "prefill_tok_s": float(prefill_tok_s),
        "decode_tok_s": float(decode_tok_s),
        "first_token_ms": float(first_token_ms),
        "p50_ms": float(p50_ms),
        "p99_ms": float(p99_ms),
        "bandwidth_peak_gbs": float(peak_gbs) if peak_gbs is not None else None,
        "backend": "metal_marlin",
    }
    return metrics, details


def benchmark_torch_mps(
    model_path: Path,
    prompt: str,
    num_tokens: int,
    num_runs: int,
) -> tuple[EndToEndMetrics, dict[str, Any]]:
    _require_mps("PyTorch MPS inference")
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise RuntimeError("transformers is required for PyTorch MPS benchmark.")
    assert torch is not None

    device = "mps"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    input_ids = _build_input_ids(tokenizer, prompt, device)
    prompt_tokens = int(input_ids.shape[1])

    memory = MemoryTracker(device)
    prefill_times: list[float] = []
    decode_times: list[float] = []
    decode_latencies_ms: list[float] = []
    first_token_latencies_ms: list[float] = []

    with torch.no_grad():
        for _ in range(num_runs):
            _mps_sync()
            start = time.perf_counter()
            outputs = model(input_ids, use_cache=True)
            _mps_sync()
            prefill_time = time.perf_counter() - start
            prefill_times.append(prefill_time)
            memory.update()

            past = outputs.past_key_values
            logits = outputs.logits
            next_token = torch.argmax(logits[:, -1, :], dim=-1)

            decode_start = time.perf_counter()
            for step in range(num_tokens):
                _mps_sync()
                step_start = time.perf_counter()
                outputs = model(
                    next_token.view(1, 1),
                    use_cache=True,
                    past_key_values=past,
                )
                _mps_sync()
                step_time_ms = (time.perf_counter() - step_start) * 1000.0
                decode_latencies_ms.append(step_time_ms)
                if step == 0:
                    first_token_latencies_ms.append(step_time_ms)
                past = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                memory.update()
            decode_times.append(time.perf_counter() - decode_start)

    prefill_tok_s = prompt_tokens / statistics.mean(prefill_times)
    decode_tok_s = num_tokens / statistics.mean(decode_times)
    first_token_ms = statistics.mean(first_token_latencies_ms) if first_token_latencies_ms else 0.0
    p50_ms = _percentile(decode_latencies_ms, 0.50)
    p99_ms = _percentile(decode_latencies_ms, 0.99)

    config = getattr(model, "config", None)
    num_layers = _get_config_value(config, ["num_hidden_layers"], 0)
    num_heads = _get_config_value(config, ["num_attention_heads"], 0)
    num_kv_heads = _get_config_value(config, ["num_key_value_heads", "num_kv_heads"], num_heads)
    hidden_size = _get_config_value(config, ["hidden_size"], 0)
    head_dim = hidden_size // num_heads if num_heads else 0

    def _decode_step() -> None:
        outputs = model(input_ids, use_cache=True)
        past = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        _ = model(next_token.view(1, 1), use_cache=True, past_key_values=past)

    bw_util, bw_gbs, peak_gbs = _estimate_decode_bandwidth(
        _decode_step,
        seq_len=prompt_tokens,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bytes_per_elem=2.0,
    )

    metrics = EndToEndMetrics(
        tokens_per_second=float(decode_tok_s),
        ttft_ms=float(first_token_ms),
        memory_peak_gb=float(memory.peak_gb),
        bandwidth_util_pct=float(bw_util) if bw_util is not None else None,
        bandwidth_gbs=float(bw_gbs) if bw_gbs is not None else None,
    )

    details = {
        "prompt_tokens": prompt_tokens,
        "prefill_tok_s": float(prefill_tok_s),
        "decode_tok_s": float(decode_tok_s),
        "first_token_ms": float(first_token_ms),
        "p50_ms": float(p50_ms),
        "p99_ms": float(p99_ms),
        "bandwidth_peak_gbs": float(peak_gbs) if peak_gbs is not None else None,
        "backend": "torch_mps",
    }
    return metrics, details


def benchmark_gemm(
    *,
    warmup: int,
    iterations: int,
) -> list[dict[str, Any]]:
    _require_mps("GEMM benchmarking")
    assert torch is not None

    sys.path.insert(0, str(Path(__file__).parent))
    from framework import Benchmark

    shapes = [
        (1, 4096, 4096, "Llama-7B q/k/v"),
        (1, 4096, 11008, "Llama-7B up_proj"),
        (1, 11008, 4096, "Llama-7B down_proj"),
        (1, 4096, 14336, "Llama-3-8B up_proj"),
        (1, 14336, 4096, "Llama-3-8B down_proj"),
        (32, 4096, 4096, "Batch32 decode"),
        (128, 4096, 4096, "Prefill 128"),
    ]

    bench = Benchmark(warmup=warmup, iterations=iterations)
    results: list[dict[str, Any]] = []

    for M, N, K, label in shapes:
        A = torch.randn(M, K, dtype=torch.float16, device="mps")
        B = torch.randn(K, N, dtype=torch.float16, device="mps")
        _mps_sync()

        def fn(a: Any = A, b: Any = B) -> Any:
            return a @ b

        result = bench.run(label, fn, M, N, K)
        results.append(
            {
                "shape": {"M": M, "N": N, "K": K},
                "label": label,
                "tflops": result.tflops,
                "mean_ms": result.mean_ms,
            }
        )

    return results


def benchmark_attention(
    *,
    seq_len: int,
    head_dim: int,
    heads: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    _require_mps("attention benchmarking")
    if torch is None:
        raise RuntimeError("PyTorch not available.")
    import torch.nn.functional as F

    batch = 1
    q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="mps")
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="mps")
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="mps")
    scale = 1.0 / math.sqrt(head_dim)

    def fn() -> Any:
        return F.scaled_dot_product_attention(q, k, v, scale=scale)

    for _ in range(warmup):
        fn()
        _mps_sync()

    times: list[float] = []
    for _ in range(iterations):
        _mps_sync()
        start = time.perf_counter()
        fn()
        _mps_sync()
        times.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.mean(times)
    tokens = batch * seq_len
    tokens_per_sec = tokens / (mean_ms / 1000.0) if mean_ms > 0 else 0.0

    return {
        "shape": {"batch": batch, "heads": heads, "seq_len": seq_len, "head_dim": head_dim},
        "tokens_per_sec": float(tokens_per_sec),
        "mean_ms": float(mean_ms),
    }


def benchmark_moe(
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

    x = torch.randn(total_tokens, hidden_size, dtype=torch.float16, device="mps")
    weights = torch.randn(
        active_experts, hidden_size, hidden_size, dtype=torch.float16, device="mps"
    )
    router = torch.arange(active_experts, device="mps").repeat_interleave(tokens_per_expert)

    def run_grouped() -> tuple[float, float]:
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
        run_grouped()

    dispatch_times: list[float] = []
    compute_times: list[float] = []

    for _ in range(iterations):
        d_ms, c_ms = run_grouped()
        dispatch_times.append(d_ms)
        compute_times.append(c_ms)

    dispatch_ms = statistics.mean(dispatch_times)
    compute_ms = statistics.mean(compute_times)
    total_ms = dispatch_ms + compute_ms

    tokens_per_sec = total_tokens / (total_ms / 1000.0) if total_ms > 0 else 0.0

    return {
        "shape": {
            "num_experts": num_experts,
            "active_experts": active_experts,
            "tokens": total_tokens,
            "hidden_size": hidden_size,
        },
        "dispatch_ms": float(dispatch_ms),
        "compute_ms": float(compute_ms),
        "tokens_per_sec": float(tokens_per_sec),
    }


def benchmark_kv_cache(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    warmup: int,
    iterations: int,
    cache_dtype: str,
) -> dict[str, Any]:
    _require_mps("KV cache benchmarking")
    assert torch is not None

    config = CacheConfig(
        num_layers=num_layers,
        num_heads=num_kv_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=seq_len,
        cache_dtype=cache_dtype,
    )
    cache = KVCache(config, batch_size=1)

    k_new = torch.randn(1, num_kv_heads, 1, head_dim, dtype=torch.float16, device="mps")
    v_new = torch.randn(1, num_kv_heads, 1, head_dim, dtype=torch.float16, device="mps")

    def write_once() -> None:
        for layer in range(num_layers):
            cache._slice_update(cache.k_cache[layer], k_new, cache.cache_position, 1)
            cache._slice_update(cache.v_cache[layer], v_new, cache.cache_position, 1)

    def read_once() -> None:
        for layer in range(num_layers):
            _ = cache.k_cache[layer][:, :, : cache.cache_position + 1, :].clone()
            _ = cache.v_cache[layer][:, :, : cache.cache_position + 1, :].clone()

    for _ in range(warmup):
        write_once()
        cache.cache_position += 1
        read_once()

    write_times: list[float] = []
    read_times: list[float] = []

    for _ in range(iterations):
        _mps_sync()
        start = time.perf_counter()
        write_once()
        _mps_sync()
        write_times.append((time.perf_counter() - start) * 1000.0)
        cache.cache_position += 1

        _mps_sync()
        start = time.perf_counter()
        read_once()
        _mps_sync()
        read_times.append((time.perf_counter() - start) * 1000.0)

    bytes_per_elem = _estimate_kv_bytes_per_elem(cache_dtype)
    bytes_write = int(2 * num_layers * num_kv_heads * head_dim * bytes_per_elem)
    bytes_read = int(
        2 * num_layers * num_kv_heads * head_dim * (cache.cache_position) * bytes_per_elem
    )

    write_ms = statistics.mean(write_times)
    read_ms = statistics.mean(read_times)

    write_bw_gbs = (bytes_write / (write_ms / 1000.0)) / 1e9 if write_ms > 0 else 0.0
    read_bw_gbs = (bytes_read / (read_ms / 1000.0)) / 1e9 if read_ms > 0 else 0.0

    return {
        "shape": {
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "seq_len": cache.cache_position,
        },
        "write_bandwidth_gbs": float(write_bw_gbs),
        "read_bandwidth_gbs": float(read_bw_gbs),
        "write_mean_ms": float(write_ms),
        "read_mean_ms": float(read_ms),
        "cache_dtype": cache_dtype,
    }


def benchmark_llama_cpp(
    *,
    gguf_path: Path | None,
    prompt: str,
    num_tokens: int,
) -> dict[str, Any]:
    if gguf_path is None:
        return {"status": "skipped", "reason": "no gguf model path provided"}

    try:
        from llama_cpp import Llama  # type: ignore
    except Exception as exc:
        return {"status": "unavailable", "reason": f"llama_cpp import failed: {exc}"}

    try:
        model = Llama(model_path=str(gguf_path), n_gpu_layers=-1)

        start = time.perf_counter()
        output = model(prompt, max_tokens=1)
        ttft_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        output = model(prompt, max_tokens=num_tokens)
        elapsed = time.perf_counter() - start
        usage = output.get("usage", {}) if isinstance(output, dict) else {}
        tokens_generated = int(usage.get("completion_tokens", num_tokens))

        tokens_per_second = tokens_generated / elapsed if elapsed > 0 else 0.0

        return {
            "status": "ok",
            "tokens_per_second": float(tokens_per_second),
            "ttft_ms": float(ttft_ms),
            "tokens_generated": int(tokens_generated),
        }
    except Exception as exc:  # pragma: no cover - optional dependency runtime
        return {"status": "error", "reason": str(exc)}


def benchmark_mlx(
    *,
    model_name: str | None,
    prompt: str,
    num_tokens: int,
) -> dict[str, Any]:
    if model_name is None:
        return {"status": "skipped", "reason": "no MLX model provided"}

    try:
        import mlx.core as mx  # type: ignore
        from mlx_lm import utils as mlx_utils  # type: ignore
    except Exception as exc:
        return {"status": "unavailable", "reason": f"mlx import failed: {exc}"}

    try:
        model, tokenizer = mlx_utils.load(model_name)
        prompt_ids = tokenizer.encode(prompt)

        start = time.perf_counter()
        logits, cache = model(mx.array([prompt_ids]), cache=None)
        _ = logits
        ttft_ms = (time.perf_counter() - start) * 1000.0

        tokens_generated = 0
        start = time.perf_counter()
        ids = list(prompt_ids)
        for _ in range(num_tokens):
            logits, cache = model(mx.array([ids[-1:]]), cache=cache)
            next_token = int(mx.argmax(logits[0, -1]))
            ids.append(next_token)
            tokens_generated += 1
        elapsed = time.perf_counter() - start

        tokens_per_second = tokens_generated / elapsed if elapsed > 0 else 0.0

        return {
            "status": "ok",
            "tokens_per_second": float(tokens_per_second),
            "ttft_ms": float(ttft_ms),
            "tokens_generated": int(tokens_generated),
        }
    except Exception as exc:  # pragma: no cover - optional dependency runtime
        return {"status": "error", "reason": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline benchmark for GLM-4.7-Flash")
    parser.add_argument("--model-path", type=Path, default=_ROOT / "models" / "GLM-4.7-Flash")
    parser.add_argument("--quantized-model-path", type=Path, default=None)
    parser.add_argument("--prompt", type=str, default="Explain transformers in one paragraph.")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--attention-seq", type=int, default=4096)
    parser.add_argument("--attention-heads", type=int, default=32)
    parser.add_argument("--attention-head-dim", type=int, default=128)
    parser.add_argument("--moe-experts", type=int, default=16)
    parser.add_argument("--moe-active-experts", type=int, default=4)
    parser.add_argument("--moe-tokens", type=int, default=2048)
    parser.add_argument("--moe-hidden", type=int, default=2048)
    parser.add_argument("--kv-layers", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--kv-head-dim", type=int, default=128)
    parser.add_argument("--kv-seq-len", type=int, default=4096)
    parser.add_argument("--kv-cache-dtype", type=str, default="bf16")
    parser.add_argument("--llama-gguf", type=Path, default=None)
    parser.add_argument("--mlx-model", type=str, default=None)
    parser.add_argument("--date", type=str, default="2026-01-26")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "baseline_2026_01.json",
    )

    args = parser.parse_args()

    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required to run this benchmark.")
    if not HAS_MPS:
        raise RuntimeError("PyTorch MPS backend is required to run this benchmark.")

    gpu = detect_gpu()
    gpu_name = gpu.name.replace("_", " ").title()
    hardware = f"{gpu_name} {gpu.gpu_cores}-core GPU"

    end_to_end = {}
    primary_metrics: EndToEndMetrics | None = None

    if args.quantized_model_path is not None:
        metrics, details = benchmark_metal_marlin(
            args.quantized_model_path,
            args.prompt,
            args.tokens,
            args.runs,
        )
        end_to_end["metal_marlin"] = details
        primary_metrics = metrics

    metrics, details = benchmark_torch_mps(
        args.model_path,
        args.prompt,
        args.tokens,
        args.runs,
    )
    end_to_end["torch_mps"] = details
    if primary_metrics is None:
        primary_metrics = metrics

    gemm = benchmark_gemm(warmup=args.warmup, iterations=args.iterations)
    attention = benchmark_attention(
        seq_len=args.attention_seq,
        head_dim=args.attention_head_dim,
        heads=args.attention_heads,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    moe = benchmark_moe(
        num_experts=args.moe_experts,
        active_experts=args.moe_active_experts,
        tokens=args.moe_tokens,
        hidden_size=args.moe_hidden,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    kv_cache = benchmark_kv_cache(
        num_layers=args.kv_layers,
        num_kv_heads=args.kv_heads,
        head_dim=args.kv_head_dim,
        seq_len=args.kv_seq_len,
        warmup=args.warmup,
        iterations=args.iterations,
        cache_dtype=args.kv_cache_dtype,
    )

    comparisons = {
        "llama_cpp": benchmark_llama_cpp(
            gguf_path=args.llama_gguf,
            prompt=args.prompt,
            num_tokens=args.tokens,
        ),
        "mlx": benchmark_mlx(
            model_name=args.mlx_model,
            prompt=args.prompt,
            num_tokens=args.tokens,
        ),
    }

    assert primary_metrics is not None

    output = {
        "date": args.date,
        "model": "GLM-4.7-Flash",
        "hardware": hardware,
        "metrics": {
            "tokens_per_second": primary_metrics.tokens_per_second,
            "ttft_ms": primary_metrics.ttft_ms,
            "memory_peak_gb": primary_metrics.memory_peak_gb,
            "bandwidth_util_pct": primary_metrics.bandwidth_util_pct,
            "bandwidth_gbs": primary_metrics.bandwidth_gbs,
        },
        "components": {
            "gemm": gemm,
            "attention": attention,
            "moe": moe,
            "kv_cache": kv_cache,
        },
        "end_to_end": end_to_end,
        "comparisons": comparisons,
    }

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"Baseline written to {output_path}")


if __name__ == "__main__":
    main()
