"""
End-to-end Llama inference benchmarks.

Compares FP16 baseline vs Marlin FP4-quantized inference on Apple Silicon,
measuring prefill throughput and autoregressive generation speed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx

# Ensure metal_marlin is importable from project layout
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "python"))

from metal_marlin import MarlinLinear, pack_fp4_weights  # noqa: E402


def load_llama(model_name: str):
    """Load Llama model for benchmarking via mlx-lm."""
    try:
        from mlx_lm import load
        return load(model_name)
    except ImportError:
        print("Install mlx-lm: uv pip install mlx-lm")
        raise


def quantize_model_marlin(model, group_size: int = 128) -> None:
    """Replace all nn.Linear / QuantizedLinear layers with MarlinLinear in-place.

    Walks the model tree and swaps any linear layer whose weight is large enough
    to benefit from FP4 quantization.
    """
    import mlx.nn as nn

    def _swap_linear(module, name: str, child) -> bool:
        if isinstance(child, nn.QuantizedLinear):
            marlin = MarlinLinear.from_quantized_linear(child)
            setattr(module, name, marlin)
            return True
        if isinstance(child, nn.Linear):
            w = child.weight
            mx.eval(w)
            w_packed, scales = pack_fp4_weights(w, group_size=group_size)
            bias = child.bias if hasattr(child, "bias") and child.bias is not None else None
            setattr(module, name, MarlinLinear(w_packed, scales, bias, group_size))
            return True
        return False

    def _recurse(module):
        for name in dir(module):
            if name.startswith("_"):
                continue
            try:
                child = getattr(module, name)
            except Exception:
                continue
            if isinstance(child, (nn.Linear, nn.QuantizedLinear)):
                _swap_linear(module, name, child)
            elif hasattr(child, "__dict__") and isinstance(child, nn.Module):
                _recurse(child)

    _recurse(model)


def benchmark_prefill(model, tokenizer, prompt: str, num_runs: int = 10) -> dict:
    """Benchmark prefill (prompt processing) speed."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Warmup
    _ = model(input_ids)
    mx.eval(_)

    # Benchmark
    times: list[float] = []
    for _ in range(num_runs):
        mx.synchronize()
        start = time.perf_counter()
        out = model(input_ids)
        mx.eval(out)
        mx.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    mean_time = sum(times) / len(times)
    tokens_per_sec = len(tokens) / mean_time

    return {
        "prompt_tokens": len(tokens),
        "mean_time_ms": mean_time * 1000,
        "tokens_per_sec": tokens_per_sec,
    }


def benchmark_generation(
    model, tokenizer, prompt: str, max_tokens: int = 100,
) -> dict:
    """Benchmark autoregressive generation speed using mlx_lm.utils."""
    try:
        from mlx_lm.utils import generate_step
    except ImportError:
        # Fallback: manual greedy decode via model forward
        return _benchmark_generation_manual(model, tokenizer, prompt, max_tokens)

    tokens = tokenizer.encode(prompt)
    prompt_arr = mx.array(tokens)

    generated: list[int] = []
    mx.synchronize()
    start = time.perf_counter()

    for token, _ in generate_step(prompt_arr, model, temp=0.0):
        generated.append(token)
        if token == tokenizer.eos_token_id or len(generated) >= max_tokens:
            break

    mx.synchronize()
    elapsed = time.perf_counter() - start

    tokens_per_sec = len(generated) / elapsed if elapsed > 0 else 0.0

    return {
        "generated_tokens": len(generated),
        "total_time_sec": elapsed,
        "tokens_per_sec": tokens_per_sec,
        "text": tokenizer.decode(generated),
    }


def _benchmark_generation_manual(
    model, tokenizer, prompt: str, max_tokens: int = 100,
) -> dict:
    """Manual greedy generation fallback when generate_step unavailable."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    generated: list[int] = []
    mx.synchronize()
    start = time.perf_counter()

    # Initial prefill
    logits = model(input_ids)
    mx.eval(logits)
    next_token = int(mx.argmax(logits[:, -1, :], axis=-1)[0])
    generated.append(next_token)

    # Autoregressive decode (no KV cache - measures raw model speed)
    for _ in range(max_tokens - 1):
        input_ids = mx.array([[next_token]])
        logits = model(input_ids)
        mx.eval(logits)
        next_token = int(mx.argmax(logits[:, -1, :], axis=-1)[0])
        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break

    mx.synchronize()
    elapsed = time.perf_counter() - start

    tokens_per_sec = len(generated) / elapsed if elapsed > 0 else 0.0

    return {
        "generated_tokens": len(generated),
        "total_time_sec": elapsed,
        "tokens_per_sec": tokens_per_sec,
        "text": tokenizer.decode(generated),
    }


def compare_fp16_vs_marlin(
    model_name: str = "mlx-community/Llama-3.2-1B-Instruct-4bit",
) -> dict:
    """Compare FP16 vs Marlin-quantized inference.

    Default model is a small 4-bit Llama variant from mlx-community that
    fits in unified memory without OOM on most Apple Silicon Macs.
    """
    prompt = "The meaning of life is"

    # FP16 baseline (load 4-bit, dequant would be too large for 7B;
    # use the MLX native quantized path as baseline instead)
    print(f"Loading model: {model_name}")
    model_baseline, tokenizer = load_llama(model_name)

    print("Benchmarking baseline (MLX native quantized)...")
    baseline_prefill = benchmark_prefill(model_baseline, tokenizer, prompt)
    baseline_gen = benchmark_generation(model_baseline, tokenizer, prompt)
    print(f"  Prefill: {baseline_prefill['tokens_per_sec']:.1f} tok/s")
    print(f"  Generation: {baseline_gen['tokens_per_sec']:.1f} tok/s")

    # Marlin FP4
    print("Converting to Marlin FP4...")
    quantize_model_marlin(model_baseline, group_size=128)
    mx.eval(model_baseline.parameters())

    print("Benchmarking Marlin FP4...")
    marlin_prefill = benchmark_prefill(model_baseline, tokenizer, prompt)
    marlin_gen = benchmark_generation(model_baseline, tokenizer, prompt)
    print(f"  Prefill: {marlin_prefill['tokens_per_sec']:.1f} tok/s")
    print(f"  Generation: {marlin_gen['tokens_per_sec']:.1f} tok/s")

    prefill_speedup = marlin_prefill["tokens_per_sec"] / baseline_prefill["tokens_per_sec"]
    gen_speedup = marlin_gen["tokens_per_sec"] / baseline_gen["tokens_per_sec"]
    print("\nSpeedup (Marlin vs baseline):")
    print(f"  Prefill: {prefill_speedup:.2f}x")
    print(f"  Generation: {gen_speedup:.2f}x")

    return {
        "model": model_name,
        "baseline_prefill_tok_s": baseline_prefill["tokens_per_sec"],
        "baseline_gen_tok_s": baseline_gen["tokens_per_sec"],
        "marlin_prefill_tok_s": marlin_prefill["tokens_per_sec"],
        "marlin_gen_tok_s": marlin_gen["tokens_per_sec"],
        "prefill_speedup": prefill_speedup,
        "gen_speedup": gen_speedup,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Llama inference benchmark: baseline vs Marlin FP4")
    parser.add_argument(
        "--model", type=str,
        default="mlx-community/Llama-3.2-1B-Instruct-4bit",
        help="HuggingFace model ID (must be mlx-lm compatible)",
    )
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--prompt", type=str, default="The meaning of life is", help="Prompt text")
    args = parser.parse_args()

    compare_fp16_vs_marlin(model_name=args.model)
