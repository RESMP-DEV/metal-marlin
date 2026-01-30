"""Benchmark trellis generation performance."""

import time

import torch

from metal_marlin.trellis_generate import GenerationConfig, TrellisGenerator
from metal_marlin.trellis_lm import TrellisForCausalLM


def benchmark_generation(
    model_path: str,
    prompt: str = "The quick brown fox",
    max_tokens: int = 100,
    warmup: int = 2,
    iterations: int = 5,
):
    from transformers import AutoTokenizer

    print(f"Loading model from {model_path}...")
    model = TrellisForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash")

    generator = TrellisGenerator(model, tokenizer)
    config = GenerationConfig(max_new_tokens=max_tokens, do_sample=False)

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        generator.generate(prompt, config)
        torch.mps.synchronize()

    # Benchmark
    print(f"Benchmarking ({iterations} iterations)...")
    times = []
    for i in range(iterations):
        torch.mps.synchronize()
        start = time.perf_counter()
        output = generator.generate(prompt, config)
        torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        tokens_generated = len(tokenizer.encode(output)) - len(tokenizer.encode(prompt))
        print(f"  Run {i+1}: {tokens_generated} tokens in {elapsed:.2f}s "
              + f"({tokens_generated/elapsed:.1f} tok/s)")

    avg_time = sum(times) / len(times)
    print(f"\nAverage: {avg_time:.2f}s ({max_tokens/avg_time:.1f} tok/s)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/GLM-4.7-Flash-EXL3-3bpw")
    parser.add_argument("--prompt", default="Explain quantum computing:")
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    benchmark_generation(args.model, args.prompt, args.max_tokens)
