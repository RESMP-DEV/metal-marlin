#!/usr/bin/env python3
"""Throughput benchmark comparing quantized vs BF16 inference."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from metal_marlin._compat import HAS_TORCH, torch
from metal_marlin.hf_loader import load_quantized_weights
from metal_marlin.inference_metal import MetalQuantizedLinear
from metal_marlin.layer_replacement import replace_linear_layers

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None


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

import os

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

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


def _apply_quantized_weights(
    model: Any,
    quantized: dict[str, dict[str, Any]],
    *,
    device: str,
) -> int:
    _require_torch()
    assert torch is not None
    loaded = 0
    for name, module in model.named_modules():
        if not isinstance(module, MetalQuantizedLinear):
            continue
        entry = quantized.get(f"{name}.weight")
        if not entry or "packed" not in entry:
            continue
        packed = torch.from_numpy(entry["packed"]).to(device=device, dtype=torch.uint32)
        scales = torch.from_numpy(entry["scales"]).to(device=device, dtype=torch.float16)
        if module._needs_output_slice:
            pad_cols = module._padded_out_features - module.out_features
            packed = torch.nn.functional.pad(packed, (0, pad_cols, 0, 0))
            scales = torch.nn.functional.pad(scales, (0, pad_cols, 0, 0))
        module.weight_packed.copy_(packed)
        module.scales.copy_(scales)
        loaded += 1
    return loaded


def _load_quantized_transformers_model(
    model_id: str,
    quantized_path: str | None,
    device: str,
    *,
    bits: int = 4,
) -> tuple[Any, Any]:
    _require_torch()
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise RuntimeError("transformers is required for quantized benchmarking.")
    if device != "mps":
        raise RuntimeError("Quantized benchmarks require MPS device.")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="mps",
    )
    replace_linear_layers(model, bits=bits)

    quantized_dir = Path(quantized_path) if quantized_path else None
    if quantized_dir and quantized_dir.exists():
        quantized = load_quantized_weights(quantized_dir)
        loaded = _apply_quantized_weights(model, quantized, device=device)
        if loaded == 0:
            raise RuntimeError(f"No quantized weights matched modules in {quantized_dir}")

    tokenizer_source = str(quantized_dir) if quantized_dir and quantized_dir.exists() else model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def _benchmark_transformers_model(
    model: Any,
    tokenizer: Any,
    prompt: str,
    num_tokens: int,
    num_runs: int,
    device: str,
) -> tuple[BenchmarkStats, int, str]:
    _require_torch()
    assert torch is not None

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


def _benchmark_quantized_model(
    model_id: str,
    quantized_path: str | None,
    prompt: str,
    num_tokens: int,
    num_runs: int,
) -> tuple[BenchmarkStats, int, str]:
    _require_torch()
    assert torch is not None
    device = _get_device()
    model, tokenizer = _load_quantized_transformers_model(
        model_id=model_id,
        quantized_path=quantized_path,
        device=device,
    )
    return _benchmark_transformers_model(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        num_tokens=num_tokens,
        num_runs=num_runs,
        device=device,
    )


def _benchmark_bf16_model(
    model_path: str,
    prompt: str,
    num_tokens: int,
    num_runs: int,
) -> tuple[BenchmarkStats, int, str]:
    _require_torch()
    assert torch is not None

    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise RuntimeError("transformers is required for BF16 benchmarking.")

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

    return _benchmark_transformers_model(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        num_tokens=num_tokens,
        num_runs=num_runs,
        device=device,
    )


def benchmark_model(
    model_id: str,
    quantized_path: str | None,
    prompt: str = "Explain quantum computing in simple terms.",
    num_tokens: int = 100,
    num_runs: int = 5,
) -> dict:
    """Benchmark tokens/second for quantized model."""
    stats, prompt_tokens, device = _benchmark_quantized_model(
        model_id=model_id,
        quantized_path=quantized_path,
        prompt=prompt,
        num_tokens=num_tokens,
        num_runs=num_runs,
    )
    return {
        "model_path": model_id,
        "quantized_path": quantized_path,
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
        bf16_path,
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
