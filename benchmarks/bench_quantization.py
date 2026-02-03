#!/usr/bin/env python3
"""Benchmark all quantization formats: Trellis 3bpw, GGUF Q4_K, GPTQ 4-bit.

Measures speed, memory, and perplexity for each format.
"""

from __future__ import annotations

import gc
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Add parent to path for metal_marlin imports
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Benchmark Results
# ---------------------------------------------------------------------------


@dataclass
class QuantBenchmarkResult:
    """Result for a single quantization format."""

    name: str
    format_type: str  # "trellis_3bpw", "gguf_q4k", "gptq_4bit"
    load_time_s: float
    memory_gb: float
    throughput_tok_s: float
    perplexity: float
    disk_size_gb: float
    bpw: float  # bits per weight

    def to_dict(self) -> dict:
        """Convert to dictionary for easy display."""
        return {
            "Format": self.name,
            "Type": self.format_type,
            "BPW": f"{self.bpw:.1f}",
            "Disk (GB)": f"{self.disk_size_gb:.2f}",
            "Load Time (s)": f"{self.load_time_s:.1f}",
            "Memory (GB)": f"{self.memory_gb:.2f}",
            "Throughput (tok/s)": f"{self.throughput_tok_s:.0f}",
            "Perplexity": f"{self.perplexity:.2f}",
        }


# ---------------------------------------------------------------------------
# Memory Utilities
# ---------------------------------------------------------------------------


def get_mps_memory() -> float:
    """Get current MPS allocated memory in GB."""
    if HAS_TORCH and torch.backends.mps.is_available():
        torch.mps.synchronize()
        return torch.mps.current_allocated_memory() / 1e9
    return 0.0


# ---------------------------------------------------------------------------
# Throughput Benchmarking
# ---------------------------------------------------------------------------


def benchmark_throughput(
    model: Any,
    tokenizer: Any,
    seq_len: int = 128,
    batch_size: int = 4,
    warmup: int = 3,
    iterations: int = 20,
) -> float:
    """Benchmark model throughput in tokens/second.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        seq_len: Sequence length
        batch_size: Batch size
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations

    Returns:
        Throughput in tokens/second
    """
    if not HAS_TORCH:
        return 0.0

    model.eval()
    device = next(model.parameters()).device

    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)

        torch.mps.synchronize() if device.type == "mps" else None
        start = time.perf_counter()

        for _ in range(iterations):
            _ = model(input_ids)

        torch.mps.synchronize() if device.type == "mps" else None
        elapsed = time.perf_counter() - start

    tokens = batch_size * seq_len * iterations
    return tokens / elapsed


# ---------------------------------------------------------------------------
# Perplexity Computation
# ---------------------------------------------------------------------------


def compute_perplexity(
    model: Any, tokenizer: Any, texts: list[str], max_length: int = 512
) -> float:
    """Compute perplexity of a model on a text dataset.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        texts: List of text strings to evaluate
        max_length: Maximum sequence length per sample

    Returns:
        Perplexity (exp of mean cross-entropy loss)
    """
    if not HAS_TORCH:
        return 0.0

    model.eval()
    device = next(model.parameters()).device

    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            tokens = tokenizer.encode(text, return_tensors="pt")
            if tokens.shape[1] < 2:
                continue
            tokens = tokens[:, :max_length]

            input_ids = tokens[:, :-1].to(device)
            targets = tokens[:, 1:].to(device)

            outputs = model(input_ids)
            logits = outputs.logits

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            batch_size, seq_len, vocab_size = log_probs.shape
            token_log_probs = log_probs.view(-1, vocab_size)[
                torch.arange(seq_len, device=device), targets.view(-1)
            ]

            nll = -token_log_probs.sum().item()
            total_nll += nll
            total_tokens += seq_len

    if total_tokens == 0:
        return float("inf")

    return math.exp(total_nll / total_tokens)


# ---------------------------------------------------------------------------
# Trellis 3bpw Benchmark
# ---------------------------------------------------------------------------


def benchmark_trellis_3bpw(model_path: str, texts: list[str]) -> QuantBenchmarkResult:
    """Benchmark Trellis 3bpw quantization.

    Args:
        model_path: Path to Trellis model
        texts: Test texts for perplexity

    Returns:
        Benchmark results
    """
    print(f"\nBenchmarking Trellis 3bpw: {model_path}")

    from transformers import AutoTokenizer

    from metal_marlin.trellis.lm import TrellisForCausalLM

    gc.collect()
    if HAS_TORCH and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    baseline = get_mps_memory()

    disk_size_gb = (
        sum(f.stat().st_size for f in Path(model_path).rglob("*.safetensors") if f.is_file()) / 1e9
    )

    start_load = time.perf_counter()
    model = TrellisForCausalLM.from_pretrained(model_path, device="mps")
    load_time = time.perf_counter() - start_load

    memory = get_mps_memory() - baseline

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    throughput = benchmark_throughput(model, tokenizer)
    perplexity = compute_perplexity(model, tokenizer, texts)

    return QuantBenchmarkResult(
        name="Trellis 3bpw",
        format_type="trellis_3bpw",
        load_time_s=load_time,
        memory_gb=memory,
        throughput_tok_s=throughput,
        perplexity=perplexity,
        disk_size_gb=disk_size_gb,
        bpw=3.0,
    )


# ---------------------------------------------------------------------------
# GGUF Q4_K Benchmark
# ---------------------------------------------------------------------------


def benchmark_gguf_q4k(model_path: str, texts: list[str]) -> QuantBenchmarkResult:
    """Benchmark GGUF Q4_K quantization.

    Args:
        model_path: Path to GGUF model
        texts: Test texts for perplexity

    Returns:
        Benchmark results
    """
    print(f"\nBenchmarking GGUF Q4_K: {model_path}")

    gc.collect()
    if HAS_TORCH and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    baseline = get_mps_memory()

    disk_size_gb = Path(model_path).stat().st_size / 1e9

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        start_load = time.perf_counter()

        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.float16
        )
        load_time = time.perf_counter() - start_load

        memory = get_mps_memory() - baseline

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        throughput = benchmark_throughput(model, tokenizer)
        perplexity = compute_perplexity(model, tokenizer, texts)

        return QuantBenchmarkResult(
            name="GGUF Q4_K",
            format_type="gguf_q4k",
            load_time_s=load_time,
            memory_gb=memory,
            throughput_tok_s=throughput,
            perplexity=perplexity,
            disk_size_gb=disk_size_gb,
            bpw=4.5,
        )
    except Exception as e:
        print(f"  Error loading GGUF model: {e}")
        print("  GGUF benchmark requires llama.cpp or GGUF-specific loader")
        return QuantBenchmarkResult(
            name="GGUF Q4_K",
            format_type="gguf_q4k",
            load_time_s=0,
            memory_gb=0,
            throughput_tok_s=0,
            perplexity=float("inf"),
            disk_size_gb=disk_size_gb,
            bpw=4.5,
        )


# ---------------------------------------------------------------------------
# GPTQ 4-bit Benchmark
# ---------------------------------------------------------------------------


def benchmark_gptq_4bit(model_path: str, texts: list[str]) -> QuantBenchmarkResult:
    """Benchmark GPTQ 4-bit quantization.

    Args:
        model_path: Path to GPTQ model
        texts: Test texts for perplexity

    Returns:
        Benchmark results
    """
    print(f"\nBenchmarking GPTQ 4-bit: {model_path}")

    gc.collect()
    if HAS_TORCH and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    baseline = get_mps_memory()

    disk_size_gb = (
        sum(f.stat().st_size for f in Path(model_path).rglob("*.safetensors") if f.is_file()) / 1e9
    )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        start_load = time.perf_counter()

        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.float16
        )
        load_time = time.perf_counter() - start_load

        memory = get_mps_memory() - baseline

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        throughput = benchmark_throughput(model, tokenizer)
        perplexity = compute_perplexity(model, tokenizer, texts)

        return QuantBenchmarkResult(
            name="GPTQ 4-bit",
            format_type="gptq_4bit",
            load_time_s=load_time,
            memory_gb=memory,
            throughput_tok_s=throughput,
            perplexity=perplexity,
            disk_size_gb=disk_size_gb,
            bpw=4.0,
        )
    except Exception as e:
        print(f"  Error loading GPTQ model: {e}")
        print("  GPTQ benchmark requires AutoGPTQ or transformers with GPTQ support")
        return QuantBenchmarkResult(
            name="GPTQ 4-bit",
            format_type="gptq_4bit",
            load_time_s=0,
            memory_gb=0,
            throughput_tok_s=0,
            perplexity=float("inf"),
            disk_size_gb=disk_size_gb,
            bpw=4.0,
        )


# ---------------------------------------------------------------------------
# Results Display
# ---------------------------------------------------------------------------


def display_comparison_table(results: list[QuantBenchmarkResult]) -> None:
    """Display results in a formatted comparison table.

    Args:
        results: List of benchmark results
    """
    print("\n" + "=" * 100)
    print("Quantization Format Comparison")
    print("=" * 100)

    if not results:
        print("No results to display")
        return

    headers = list(results[0].to_dict().keys())
    col_widths = [max(len(h), 12) for h in headers]

    for i, header in enumerate(headers):
        print(f"{header:{col_widths[i]}}", end=" ")
    print()

    print("-" * sum(col_widths + len(headers) * 1))

    for result in results:
        row = result.to_dict()
        for i, header in enumerate(headers):
            print(f"{row[header]:{col_widths[i]}}", end=" ")
        print()

    print("=" * 100)

    # Summary statistics
    if len(results) > 1:
        print("\nSummary:")
        best_throughput = max(results, key=lambda r: r.throughput_tok_s)
        best_perplexity = min(results, key=lambda r: r.perplexity)
        best_memory = min(results, key=lambda r: r.memory_gb)
        smallest_disk = min(results, key=lambda r: r.disk_size_gb)

        print(
            f"  Best throughput: {best_throughput.name} ({best_throughput.throughput_tok_s:.0f} tok/s)"
        )
        print(f"  Best perplexity: {best_perplexity.name} ({best_perplexity.perplexity:.2f})")
        print(f"  Lowest memory: {best_memory.name} ({best_memory.memory_gb:.2f} GB)")
        print(f"  Smallest disk: {smallest_disk.name} ({smallest_disk.disk_size_gb:.2f} GB)")


# ---------------------------------------------------------------------------
# Main Benchmark Runner
# ---------------------------------------------------------------------------


def load_wikitext2_samples(max_samples: int = 50) -> list[str]:
    """Load WikiText-2 samples for perplexity evaluation.

    Args:
        max_samples: Maximum number of samples

    Returns:
        List of text samples
    """
    from metal_marlin.eval import load_wikitext2

    try:
        return load_wikitext2(max_samples)
    except Exception as e:
        print(f"Warning: Could not load WikiText-2: {e}")
        return []


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark quantization formats")
    parser.add_argument("--trellis", type=str, help="Path to Trellis 3bpw model")
    parser.add_argument("--gguf", type=str, help="Path to GGUF Q4_K model")
    parser.add_argument("--gptq", type=str, help="Path to GPTQ 4-bit model")
    parser.add_argument("--models-dir", type=str, default="models", help="Default models directory")
    parser.add_argument("--samples", type=int, default=50, help="Number of WikiText-2 samples")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer samples")

    args = parser.parse_args()

    print("Quantization Format Benchmark")
    print("=" * 60)

    samples = args.samples // 2 if args.quick else args.samples
    texts = load_wikitext2_samples(samples)

    print(f"Loaded {len(texts)} WikiText-2 samples for perplexity evaluation")
    print()

    results: list[QuantBenchmarkResult] = []

    models_dir = Path(args.models_dir)

    # Trellis 3bpw
    trellis_path = args.trellis or models_dir / "GLM-4.7-Flash-Trellis-3bpw"
    if Path(trellis_path).exists():
        try:
            result = benchmark_trellis_3bpw(str(trellis_path), texts)
            results.append(result)
        except Exception as e:
            print(f"  Failed to benchmark Trellis: {e}")
    else:
        print(f"  Trellis model not found at {trellis_path}")

    # GGUF Q4_K
    gguf_path = args.gguf or None
    if gguf_path and Path(gguf_path).exists():
        try:
            result = benchmark_gguf_q4k(str(gguf_path), texts)
            results.append(result)
        except Exception as e:
            print(f"  Failed to benchmark GGUF: {e}")
    else:
        print("  GGUF model not specified or not found")

    # GPTQ 4-bit
    gptq_path = args.gptq or None
    if gptq_path and Path(gptq_path).exists():
        try:
            result = benchmark_gptq_4bit(str(gptq_path), texts)
            results.append(result)
        except Exception as e:
            print(f"  Failed to benchmark GPTQ: {e}")
    else:
        print("  GPTQ model not specified or not found")

    display_comparison_table(results)

    if results:
        output_path = _ROOT / "results" / "quantization_benchmark_results.json"
        output_path.parent.mkdir(exist_ok=True)

        import json

        with open(output_path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
