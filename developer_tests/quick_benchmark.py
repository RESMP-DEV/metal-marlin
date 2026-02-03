#!/usr/bin/env python3
"""GLM-4.7-Flash comprehensive benchmark: load, prefill, decode."""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from metal_marlin.trellis.lm import TrellisForCausalLM

MODEL_PATH = Path(__file__).parent / "models" / "GLM-4.7-Flash-Trellis-3bpw"


def benchmark_section(name: str):
    print(f"\n{'='*60}")
    print(f" {name}")
    print('='*60)


def main():
    print("GLM-4.7-Flash Metal Benchmark")
    print(f"Model: {MODEL_PATH}")

    # 1. Model loading
    benchmark_section("1. Model Loading")
    t0 = time.perf_counter()
    model = TrellisForCausalLM.from_pretrained(str(MODEL_PATH), device='mps')
    torch.mps.synchronize()
    load_time = time.perf_counter() - t0
    print(f"Load time: {load_time:.2f}s")

    # Get model config info
    num_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 'unknown'
    print(f"Layers: {num_layers}, Hidden size: {hidden_size}")

    # 2. Prefill 128 tokens
    benchmark_section("2. Prefill (128 tokens)")
    x128 = torch.randint(0, 1000, (1, 128)).to('mps')

    # Warmup
    with torch.no_grad():
        _ = model(x128)
    torch.mps.synchronize()

    # Benchmark
    times_128 = []
    for _ in range(5):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x128)
        torch.mps.synchronize()
        times_128.append(time.perf_counter() - t0)

    avg_128 = sum(times_128) / len(times_128)
    print(f"Time: {avg_128*1000:.1f}ms (min: {min(times_128)*1000:.1f}, max: {max(times_128)*1000:.1f})")
    print(f"Throughput: {128/avg_128:.1f} tok/s")

    # 3. Prefill 512 tokens
    benchmark_section("3. Prefill (512 tokens)")
    x512 = torch.randint(0, 1000, (1, 512)).to('mps')

    # Warmup
    with torch.no_grad():
        _ = model(x512)
    torch.mps.synchronize()

    # Benchmark
    times_512 = []
    for _ in range(5):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x512)
        torch.mps.synchronize()
        times_512.append(time.perf_counter() - t0)

    avg_512 = sum(times_512) / len(times_512)
    print(f"Time: {avg_512*1000:.1f}ms (min: {min(times_512)*1000:.1f}, max: {max(times_512)*1000:.1f})")
    print(f"Throughput: {512/avg_512:.1f} tok/s")

    # 4. Decode throughput (autoregressive generation simulation)
    benchmark_section("4. Decode Throughput")

    # Start with a small context and generate tokens one at a time
    context_len = 32
    decode_steps = 64

    # Initial context
    input_ids = torch.randint(0, 1000, (1, context_len)).to('mps')

    # Warmup with single token generation
    with torch.no_grad():
        _ = model(input_ids)
    torch.mps.synchronize()

    # Benchmark decode (simulating autoregressive generation)
    decode_times = []
    current_ids = input_ids.clone()

    for step in range(decode_steps):
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(current_ids)
            # Get next token (greedy for simplicity)
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        torch.mps.synchronize()
        decode_times.append(time.perf_counter() - t0)

        # Append token for next iteration (growing context)
        current_ids = torch.cat([current_ids, next_token], dim=1)

    # Per-token decode time (excluding the growing context overhead)
    avg_decode = sum(decode_times) / len(decode_times)
    print(f"Generated: {decode_steps} tokens")
    print(f"Avg per-token: {avg_decode*1000:.1f}ms")
    print(f"Decode throughput: {1/avg_decode:.1f} tok/s")

    # Also test single-token decode (fixed context, no growth)
    benchmark_section("4b. Single Token Decode (Fixed Context)")
    single_input = torch.randint(0, 1000, (1, 64)).to('mps')

    # Warmup
    with torch.no_grad():
        _ = model(single_input)
    torch.mps.synchronize()

    single_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(single_input)
        torch.mps.synchronize()
        single_times.append(time.perf_counter() - t0)

    avg_single = sum(single_times) / len(single_times)
    print(f"Fixed 64-token input: {avg_single*1000:.1f}ms")
    print(f"Throughput: {64/avg_single:.1f} tok/s")

    # Summary
    benchmark_section("Summary")
    print(f"Model load:         {load_time:.2f}s")
    print(f"Prefill 128 tok:    {128/avg_128:.1f} tok/s ({avg_128*1000:.1f}ms)")
    print(f"Prefill 512 tok:    {512/avg_512:.1f} tok/s ({avg_512*1000:.1f}ms)")
    print(f"Decode (growing):   {1/avg_decode:.1f} tok/s ({avg_decode*1000:.1f}ms/tok)")
    print(f"Decode (fixed 64):  {64/avg_single:.1f} tok/s ({avg_single*1000:.1f}ms)")


if __name__ == "__main__":
    main()
