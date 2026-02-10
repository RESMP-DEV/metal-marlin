"""Benchmark MoE kernel performance: fast path vs slow path."""

import time

import torch
from transformers import AutoTokenizer

from metal_marlin.trellis.lm import TrellisForCausalLM


import os
import sys

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

def benchmark_forward(model, input_ids, warmup=5, trials=20):
    # Warmup
    for _ in range(warmup):
        with torch.inference_mode():
            _ = model(input_ids)
        torch.mps.synchronize()

    # Timed runs
    times = []
    for _ in range(trials):
        torch.mps.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            _ = model(input_ids)
        torch.mps.synchronize()
        times.append(time.perf_counter() - start)

    return {
        'mean_ms': sum(times) / len(times) * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000,
    }

def main():
    print("Loading model...")
    model = TrellisForCausalLM.from_pretrained(
        'models/GLM-4.7-Flash-Marlin-MMFP4', device='mps'
    )
    tokenizer = AutoTokenizer.from_pretrained(
        'zai-org/GLM-4.7-Flash', trust_remote_code=True
    )

    # Test at different context lengths
    for ctx_len in [128, 512, 1024, 2048]:
        prompt = "Hello " * (ctx_len // 2)
        inputs = tokenizer(prompt, return_tensors='pt', max_length=ctx_len, truncation=True)
        input_ids = inputs['input_ids'].to('mps')

        stats = benchmark_forward(model, input_ids)
        tokens_per_sec = ctx_len / (stats['mean_ms'] / 1000)

        print(f"Context {ctx_len:4d}: {stats['mean_ms']:.1f}ms "
              f"({tokens_per_sec:.0f} tok/s)")

if __name__ == '__main__':
    main()
