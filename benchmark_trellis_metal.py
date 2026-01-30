#!/usr/bin/env python3
"""Benchmark Trellis model inference with Metal shaders.

This script benchmarks the end-to-end inference performance of the Trellis
quantized model using Metal GPU acceleration.

Usage:
    python benchmark_trellis_metal.py --model models/GLM-4.7-Flash-EXL3-3bpw \
                                      --prompt "Hello, world!" \
                                      --max-tokens 128 \
                                      --device mps
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from metal_marlin.metal_dispatch import HAS_METAL, HAS_MPS
from metal_marlin.trellis_lm import TrellisForCausalLM

if TYPE_CHECKING:
    pass


def benchmark_forward_pass(
    model: TrellisForCausalLM,
    input_ids: torch.Tensor,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> dict[str, float]:
    """Benchmark forward pass performance.

    Args:
        model: The Trellis model
        input_ids: Input token IDs
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs

    Returns:
        Dictionary with timing statistics
    """
    device = next(model.parameters()).device
    print(f"Benchmarking on {device}...")

    # Warmup
    print(f"  Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(input_ids)
        if device.type == "mps":
            torch.mps.synchronize()

    # Benchmark
    print(f"  Benchmarking ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids)
        if device.type == "mps":
            torch.mps.synchronize()
        end = time.perf_counter()
        elapsed = end - start
        times.append(elapsed)
        print(f"    Run {i + 1}: {elapsed * 1000:.2f} ms")

    # Statistics
    times_tensor = torch.tensor(times)
    return {
        "mean_ms": times_tensor.mean().item() * 1000,
        "median_ms": times_tensor.median().item() * 1000,
        "min_ms": times_tensor.min().item() * 1000,
        "max_ms": times_tensor.max().item() * 1000,
        "std_ms": times_tensor.std().item() * 1000,
    }


def benchmark_generation(
    model: TrellisForCausalLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Benchmark autoregressive generation.

    Args:
        model: The Trellis model
        prompt_ids: Prompt token IDs
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Dictionary with generation statistics
    """
    device = next(model.parameters()).device
    print(f"\nBenchmarking generation on {device}...")
    print(f"  Prompt length: {prompt_ids.shape[1]} tokens")
    print(f"  Max new tokens: {max_new_tokens}")

    # Generate
    start = time.perf_counter()
    generated_ids = []
    input_ids = prompt_ids.clone()

    for i in range(max_new_tokens):
        token_start = time.perf_counter()

        with torch.no_grad():
            logits = model(input_ids)

        # Sample next token
        next_token_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated_ids.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if device.type == "mps":
            torch.mps.synchronize()

        token_end = time.perf_counter()
        token_time = (token_end - token_start) * 1000

        if (i + 1) % 10 == 0:
            print(f"    Generated {i + 1}/{max_new_tokens} tokens (last: {token_time:.2f} ms)")

    end = time.perf_counter()
    total_time = end - start

    # Statistics
    tokens_per_second = max_new_tokens / total_time
    ms_per_token = (total_time / max_new_tokens) * 1000

    return {
        "total_time_s": total_time,
        "tokens_generated": max_new_tokens,
        "tokens_per_second": tokens_per_second,
        "ms_per_token": ms_per_token,
        "prompt_tokens": prompt_ids.shape[1],
        "total_tokens": prompt_ids.shape[1] + max_new_tokens,
    }


def print_stats(name: str, stats: dict[str, float]) -> None:
    """Print benchmark statistics."""
    print(f"\n{name} Results:")
    print("=" * 50)
    for key, value in stats.items():
        if "ms" in key:
            print(f"  {key}: {value:.2f}")
        elif "s" in key:
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Trellis model with Metal shaders")
    parser.add_argument(
        "--model",
        type=str,
        default="models/GLM-4.7-Flash-EXL3-3bpw",
        help="Path to trellis-quantized model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you today?",
        help="Prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "mps"],
        default="mps" if HAS_MPS else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for forward pass",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length for forward pass benchmark",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="Skip forward pass benchmark",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation benchmark",
    )

    args = parser.parse_args()

    # Print system info
    print("=" * 70)
    print("Trellis Metal Benchmark")
    print("=" * 70)
    print(f"Metal Available: {HAS_METAL}")
    print(f"MPS Available: {HAS_MPS}")
    print(f"Device: {args.device}")
    print(f"Model: {args.model}")
    print()

    # Check device availability
    if args.device == "mps" and not HAS_MPS:
        print("Error: MPS requested but not available")
        return 1

    # Load model
    print("Loading model...")
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model path not found: {model_path}")
        return 1

    try:
        model = TrellisForCausalLM.from_pretrained(model_path, device=args.device)
        model.eval()
        print(f"  Loaded {len(model.model.layers)} layers")
        print(f"  Hidden size: {model.config.hidden_size}")
        print(f"  Attention heads: {model.config.num_attention_heads}")
        print(f"  Experts: {model.config.num_experts}")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Move model to device
    if args.device == "mps":
        model = model.to("mps")
        torch.mps.synchronize()

    # Forward pass benchmark
    if not args.skip_forward:
        print("\n" + "=" * 70)
        print("Forward Pass Benchmark")
        print("=" * 70)

        # Create random input
        input_ids = torch.randint(0, model.config.vocab_size, (args.batch_size, args.seq_len))
        if args.device == "mps":
            input_ids = input_ids.to("mps")

        print(f"Input shape: {input_ids.shape}")
        print(f"Batch size: {args.batch_size}")
        print(f"Sequence length: {args.seq_len}")

        try:
            stats = benchmark_forward_pass(model, input_ids)
            print_stats("Forward Pass", stats)
        except Exception as e:
            print(f"Error in forward pass: {e}")
            import traceback

            traceback.print_exc()

    # Generation benchmark
    if not args.skip_generation:
        print("\n" + "=" * 70)
        print("Generation Benchmark")
        print("=" * 70)

        # Simple tokenization (just use ord values for testing)
        # In production, use the actual tokenizer
        prompt_ids = torch.tensor([[ord(c) % model.config.vocab_size for c in args.prompt[:256]]])
        if args.device == "mps":
            prompt_ids = prompt_ids.to("mps")

        try:
            stats = benchmark_generation(model, prompt_ids, max_new_tokens=args.max_tokens)
            print_stats("Generation", stats)
        except Exception as e:
            print(f"Error in generation: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
