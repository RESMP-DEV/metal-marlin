#!/usr/bin/env python3
"""Throughput benchmark comparing quantized vs BF16 inference."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Any

from metal_marlin._compat import HAS_TORCH, torch
from metal_marlin.inference import load_quantized_model


@dataclass
class BenchmarkStats:
    prefill_tok_s: float
    decode_tok_s: float
    memory_peak_mb: float
    first_token_ms: float
    p50_ms: float
    p99_ms: float
    prefill_times_s: list[float]
    decode_times_s: list[float]
    decode_latencies_ms: list[float]


def _require_torch() -> None:
    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required for throughput benchmarking.")


def _get_device() -> str:
    _require_torch()
    assert torch is not None
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _sync_device(device: str) -> None:
    _require_torch()
    assert torch is not None
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


class MemoryTracker:
    def __init__(self, device: str):
        _require_torch()
        assert torch is not None
        self.device = device
        self.peak_bytes = 0
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def update(self) -> None:
        _require_torch()
        assert torch is not None
        if self.device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            current = torch.mps.current_allocated_memory()
        elif self.device == "cuda":
            current = torch.cuda.max_memory_allocated()
        else:
            current = _get_process_memory_bytes()
        self.peak_bytes = max(self.peak_bytes, int(current))

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 * 1024)


def _get_process_memory_bytes() -> int:
    import resource
    import sys

    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss * 1024)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(int(len(sorted_vals) * quantile), len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def _build_input_ids(tokenizer: Any, prompt: str, device: str):
    _require_torch()
    assert torch is not None
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


def _benchmark_quantized_model(
    model_path: str,
    prompt: str,
    num_tokens: int,
    num_runs: int,
) -> tuple[BenchmarkStats, int, str]:
    _require_torch()
    assert torch is not None
    device = _get_device()
    model, tokenizer = load_quantized_model(model_path, device=device)
    model.eval()

    input_ids = _build_input_ids(tokenizer, prompt, device)
    prompt_tokens = int(input_ids.shape[1])

    memory = MemoryTracker(device)

    # Warmup
    with torch.no_grad():
        kv_cache = model.create_kv_cache(batch_size=1)
        _ = model(input_ids, kv_cache=kv_cache)
        kv_cache.advance(input_ids.shape[1])
        _sync_device(device)

    prefill_times: list[float] = []
    decode_times: list[float] = []
    decode_latencies_ms: list[float] = []
    first_token_latencies_ms: list[float] = []

    with torch.no_grad():
        for _ in range(num_runs):
            kv_cache = model.create_kv_cache(batch_size=1)

            _sync_device(device)
            start = time.perf_counter()
            logits = model(input_ids, kv_cache=kv_cache)
            kv_cache.advance(input_ids.shape[1])
            _sync_device(device)
            prefill_time = time.perf_counter() - start
            prefill_times.append(prefill_time)
            memory.update()

            next_token = torch.argmax(logits[:, -1, :], dim=-1)

            decode_start = time.perf_counter()
            for step in range(num_tokens):
                next_input = next_token.view(1, 1)
                _sync_device(device)
                step_start = time.perf_counter()
                logits = model(next_input, kv_cache=kv_cache)
                kv_cache.advance(1)
                _sync_device(device)
                step_time_ms = (time.perf_counter() - step_start) * 1000.0
                decode_latencies_ms.append(step_time_ms)
                if step == 0:
                    first_token_latencies_ms.append(step_time_ms)
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                memory.update()
            decode_times.append(time.perf_counter() - decode_start)

    prefill_tok_s = prompt_tokens / statistics.mean(prefill_times)
    decode_tok_s = num_tokens / statistics.mean(decode_times)
    first_token_ms = statistics.mean(first_token_latencies_ms)
    p50_ms = _percentile(decode_latencies_ms, 0.50)
    p99_ms = _percentile(decode_latencies_ms, 0.99)

    stats = BenchmarkStats(
        prefill_tok_s=float(prefill_tok_s),
        decode_tok_s=float(decode_tok_s),
        memory_peak_mb=float(memory.peak_mb),
        first_token_ms=float(first_token_ms),
        p50_ms=float(p50_ms),
        p99_ms=float(p99_ms),
        prefill_times_s=prefill_times,
        decode_times_s=decode_times,
        decode_latencies_ms=decode_latencies_ms,
    )
    return stats, prompt_tokens, device


def _benchmark_bf16_model(
    model_path: str,
    prompt: str,
    num_tokens: int,
    num_runs: int,
) -> tuple[BenchmarkStats, int, str]:
    _require_torch()
    assert torch is not None

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _get_device()
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if device != "cpu":
        model = model.to(device)
    model.eval()

    input_ids = _build_input_ids(tokenizer, prompt, device)
    prompt_tokens = int(input_ids.shape[1])

    memory = MemoryTracker(device)

    # Warmup
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        _sync_device(device)
        _ = outputs.logits

    prefill_times: list[float] = []
    decode_times: list[float] = []
    decode_latencies_ms: list[float] = []
    first_token_latencies_ms: list[float] = []

    with torch.no_grad():
        for _ in range(num_runs):
            _sync_device(device)
            start = time.perf_counter()
            outputs = model(input_ids, use_cache=True)
            _sync_device(device)
            prefill_time = time.perf_counter() - start
            prefill_times.append(prefill_time)
            memory.update()

            past = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)

            decode_start = time.perf_counter()
            for step in range(num_tokens):
                next_input = next_token.view(1, 1)
                _sync_device(device)
                step_start = time.perf_counter()
                outputs = model(next_input, past_key_values=past, use_cache=True)
                _sync_device(device)
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
    first_token_ms = statistics.mean(first_token_latencies_ms)
    p50_ms = _percentile(decode_latencies_ms, 0.50)
    p99_ms = _percentile(decode_latencies_ms, 0.99)

    stats = BenchmarkStats(
        prefill_tok_s=float(prefill_tok_s),
        decode_tok_s=float(decode_tok_s),
        memory_peak_mb=float(memory.peak_mb),
        first_token_ms=float(first_token_ms),
        p50_ms=float(p50_ms),
        p99_ms=float(p99_ms),
        prefill_times_s=prefill_times,
        decode_times_s=decode_times,
        decode_latencies_ms=decode_latencies_ms,
    )
    return stats, prompt_tokens, device


def benchmark_model(
    model_path: str,
    prompt: str = "Explain quantum computing in simple terms.",
    num_tokens: int = 100,
    num_runs: int = 5,
) -> dict:
    """Benchmark tokens/second for quantized model."""
    stats, prompt_tokens, device = _benchmark_quantized_model(
        model_path=model_path,
        prompt=prompt,
        num_tokens=num_tokens,
        num_runs=num_runs,
    )
    return {
        "model_path": model_path,
        "backend": "quantized_marlin",
        "device": device,
        "prompt_tokens": prompt_tokens,
        "num_tokens": num_tokens,
        "num_runs": num_runs,
        "prefill_tok_s": stats.prefill_tok_s,
        "decode_tok_s": stats.decode_tok_s,
        "memory_peak_mb": stats.memory_peak_mb,
        "latency_ms": {
            "first_token": stats.first_token_ms,
            "p50": stats.p50_ms,
            "p99": stats.p99_ms,
        },
    }


def benchmark_bf16_model(
    model_path: str,
    prompt: str = "Explain quantum computing in simple terms.",
    num_tokens: int = 100,
    num_runs: int = 5,
) -> dict:
    """Benchmark tokens/second for BF16 model."""
    stats, prompt_tokens, device = _benchmark_bf16_model(
        model_path=model_path,
        prompt=prompt,
        num_tokens=num_tokens,
        num_runs=num_runs,
    )
    return {
        "model_path": model_path,
        "backend": "bf16_transformers",
        "device": device,
        "prompt_tokens": prompt_tokens,
        "num_tokens": num_tokens,
        "num_runs": num_runs,
        "prefill_tok_s": stats.prefill_tok_s,
        "decode_tok_s": stats.decode_tok_s,
        "memory_peak_mb": stats.memory_peak_mb,
        "latency_ms": {
            "first_token": stats.first_token_ms,
            "p50": stats.p50_ms,
            "p99": stats.p99_ms,
        },
    }


def benchmark_compare(
    quantized_path: str,
    bf16_path: str,
    prompt: str,
    num_tokens: int,
    num_runs: int,
) -> dict[str, dict]:
    quantized = benchmark_model(
        quantized_path,
        prompt=prompt,
        num_tokens=num_tokens,
        num_runs=num_runs,
    )
    bf16 = benchmark_bf16_model(
        bf16_path,
        prompt=prompt,
        num_tokens=num_tokens,
        num_runs=num_runs,
    )
    return {"quantized": quantized, "bf16": bf16}


def _format_report(label: str, result: dict) -> str:
    latency = result["latency_ms"]
    return (
        f"{label} ({result['backend']} on {result['device']}):\n"
        f"  Prefill: {result['prefill_tok_s']:.1f} tok/s\n"
        f"  Decode:  {result['decode_tok_s']:.1f} tok/s\n"
        f"  Memory:  {result['memory_peak_mb']:.1f} MB\n"
        f"  Latency: first={latency['first_token']:.2f} ms, "
        f"p50={latency['p50']:.2f} ms, p99={latency['p99']:.2f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Throughput benchmark (quantized vs BF16).")
    parser.add_argument("--quantized", required=True, help="Path to quantized model directory")
    parser.add_argument("--bf16", required=True, help="Path or HF ID for BF16 model")
    parser.add_argument(
        "--prompt",
        default="Explain quantum computing in simple terms.",
        help="Prompt for benchmarking",
    )
    parser.add_argument("--num-tokens", type=int, default=100, help="Tokens to decode")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of runs")
    args = parser.parse_args()

    results = benchmark_compare(
        quantized_path=args.quantized,
        bf16_path=args.bf16,
        prompt=args.prompt,
        num_tokens=args.num_tokens,
        num_runs=args.num_runs,
    )

    print(_format_report("Quantized", results["quantized"]))
    print()
    print(_format_report("BF16", results["bf16"]))


if __name__ == "__main__":
    main()
