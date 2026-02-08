"""Benchmark all quantized models for comparison."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class BenchmarkResult:
    model_name: str
    compression_ratio: float
    tokens_per_second: float
    memory_gb: float
    perplexity: float | None


def _mps_allocated_gb() -> float:
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1e9
    return 0.0


def benchmark_model(model_path: Path, prompt: str, num_tokens: int = 100) -> BenchmarkResult:
    from metal_marlin.inference import MetalInferenceEngine

import os
import sys

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

    engine = MetalInferenceEngine(str(model_path))

    # Warmup
    _ = engine.generate(prompt, max_tokens=10)

    # Benchmark
    start = time.perf_counter()
    _ = engine.generate(prompt, max_tokens=num_tokens)
    elapsed = time.perf_counter() - start
    tokens_per_second = num_tokens / elapsed if elapsed > 0 else 0.0

    return BenchmarkResult(
        model_name=model_path.name,
        compression_ratio=engine.compression_ratio,
        tokens_per_second=tokens_per_second,
        memory_gb=_mps_allocated_gb(),
        perplexity=None,
    )


def _default_models(benchmarks_dir: Path) -> list[Path]:
    return [
        benchmarks_dir / "results/glm4_flash_fp8_int2",
        benchmarks_dir / "results/qwen3_4b_fp4",
        benchmarks_dir / "results/qwen3_30b_fp8_int2",
        benchmarks_dir / "results/nemotron_30b_fp8_int2",
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model directories to benchmark (defaults to benchmarks/results/*).",
    )
    parser.add_argument(
        "--prompt",
        default="Explain the theory of relativity in simple terms:",
        help="Prompt to use for generation.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=100,
        help="Number of tokens to generate for timing.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    benchmarks_dir = Path(__file__).parent

    if args.models:
        model_paths = [Path(p) for p in args.models]
    else:
        model_paths = _default_models(benchmarks_dir)

    for model_path in model_paths:
        if not model_path.exists():
            print(f"Skipping missing model: {model_path}")
            continue
        result = benchmark_model(model_path, args.prompt, num_tokens=args.num_tokens)
        print(
            f"{result.model_name}: {result.tokens_per_second:.1f} tok/s, "
            f"{result.compression_ratio:.1f}x compression, "
            f"{result.memory_gb:.2f} GB"
        )


if __name__ == "__main__":
    main()
