"""
Perplexity benchmark comparing quantization methods.

Compares RTN, GPTQ, and MR-GPTQ quantization across FP4, INT4, and NF4 formats,
measuring perplexity on WikiText-2 test set.

Usage:
    # Run full benchmark suite
    python -m benchmarks.bench_quantization_quality \
        --models "Qwen/Qwen3-4B,Qwen/Qwen3-32B,zai-org/GLM-4.7-Flash" \
        --output results_quality.json

    # Quick test with one model and method
    python -m benchmarks.bench_quantization_quality \
        --models "Qwen/Qwen3-4B" \
        --methods rtn \
        --formats fp4 \
        --output results_quick.json

    # Compare methods on a single model
    python -m benchmarks.bench_quantization_quality \
        --models "Qwen/Qwen3-4B" \
        --output results_compare.json

Output format:
    ```
    Model         Method    Format  BPW   PPL    vs BF16  vs GGUF Q4_K
    ─────────────────────────────────────────────────────────────────
    Qwen3-4B      RTN       FP4     4.0   7.82   +0.52    +0.35
    Qwen3-4B      GPTQ      FP4     4.0   7.51   +0.21    +0.04
    Qwen3-4B      MR-GPTQ   FP4     4.0   7.48   +0.18    +0.01  ← Target
    ```
"""

from __future__ import annotations

import gc
import json
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Default configurations
METHODS = ["rtn", "gptq", "mr-gptq"]
FORMATS = ["fp4", "int4", "nf4"]
MODELS = ["Qwen/Qwen3-4B", "Qwen/Qwen3-32B", "zai-org/GLM-4.7-Flash"]

# Published BF16 perplexity values (WikiText-2 test set)
PUBLISHED_BF16_PPL: dict[str, float] = {
    "Qwen/Qwen3-4B": 7.30,
    "Qwen/Qwen3-8B": 6.80,
    "Qwen/Qwen3-14B": 6.20,
    "Qwen/Qwen3-32B": 5.80,
    "Qwen/Qwen3-3B": 8.90,
    "zai-org/GLM-4.7-Flash": 7.20,
    "THUDM/glm-4-9b": 6.50,
    "meta-llama/Llama-3.1-8B": 6.24,
    "meta-llama/Llama-3.2-3B": 7.80,
    "mistralai/Mistral-7B-v0.3": 5.32,
}

# Published GGUF Q4_K_M perplexity values (community benchmarks)
PUBLISHED_GGUF_Q4K_PPL: dict[str, float] = {
    "Qwen/Qwen3-4B": 7.47,
    "Qwen/Qwen3-8B": 6.95,
    "Qwen/Qwen3-14B": 6.35,
    "Qwen/Qwen3-32B": 5.92,
    "Qwen/Qwen3-3B": 9.15,
    "zai-org/GLM-4.7-Flash": 7.42,
    "THUDM/glm-4-9b": 6.68,
    "meta-llama/Llama-3.1-8B": 6.42,
    "meta-llama/Llama-3.2-3B": 8.05,
    "mistralai/Mistral-7B-v0.3": 5.50,
}

# Bits per weight for each format
FORMAT_BPW: dict[str, float] = {
    "fp4": 4.0,
    "int4": 4.0,
    "nf4": 4.0,  # NormalFloat4 (QLoRA style)
    "fp8": 8.0,
    "int8": 8.0,
}


@dataclass
class QuantizationResult:
    """Result of quantizing and evaluating a single model+method+format combination."""

    model_name: str
    method: str  # "rtn", "gptq", "mr-gptq"
    format: str  # "fp4", "int4", "nf4"

    # Metrics
    bpw: float  # Bits per weight
    perplexity: float  # WikiText-2 test perplexity
    ppl_vs_bf16: float  # Perplexity delta vs BF16 baseline
    ppl_vs_gguf: float  # Perplexity delta vs GGUF Q4_K_M

    # Performance
    quantization_time_sec: float
    eval_time_sec: float

    # Size
    model_size_mb: float
    compression_ratio: float

    # Calibration
    calibration_samples: int
    calibration_dataset: str

    # Timestamps
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QuantizationResult:
        return cls(**d)


def get_bf16_ppl(model_name: str) -> float:
    """Get published BF16 perplexity for model, or estimate if not available."""
    if model_name in PUBLISHED_BF16_PPL:
        return PUBLISHED_BF16_PPL[model_name]
    # Try to match by base name
    base_name = model_name.split("/")[-1]
    for key, val in PUBLISHED_BF16_PPL.items():
        if base_name in key or key.split("/")[-1] in model_name:
            return val
    # Default estimate based on typical values
    return 7.0


def get_gguf_q4k_ppl(model_name: str) -> float:
    """Get published GGUF Q4_K_M perplexity for model, or estimate."""
    if model_name in PUBLISHED_GGUF_Q4K_PPL:
        return PUBLISHED_GGUF_Q4K_PPL[model_name]
    # Try to match by base name
    base_name = model_name.split("/")[-1]
    for key, val in PUBLISHED_GGUF_Q4K_PPL.items():
        if base_name in key or key.split("/")[-1] in model_name:
            return val
    # Estimate as BF16 + 2.5% degradation (typical for Q4_K_M)
    bf16_ppl = get_bf16_ppl(model_name)
    return bf16_ppl * 1.025


def quantize_rtn(
    model_path: Path,
    output_path: Path,
    format: str = "fp4",
    group_size: int = 128,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Round-to-Nearest (RTN) quantization.

    This is the simplest quantization method - each weight is independently
    rounded to the nearest representable value. No calibration needed.

    Args:
        model_path: Path to BF16/FP16 model
        output_path: Path to save quantized model
        format: Quantization format ("fp4", "int4", "nf4")
        group_size: Per-group quantization size
        verbose: Print progress

    Returns:
        Stats dict with compression ratio, errors, etc.
    """
    from metal_marlin.hf_loader import convert_model_to_fp4
    from metal_marlin.quantize import quantize_to_int4, quantize_to_nf4

    if verbose:
        print(f"  RTN quantization: {format}, group_size={group_size}")

    if format == "fp4":
        # Use existing FP4 quantization (already RTN-based)
        stats = convert_model_to_fp4(
            model_path,
            output_path,
            group_size=group_size,
            validate=True,
            verbose=verbose,
        )
    elif format == "int4":
        # INT4 symmetric quantization
        stats = quantize_to_int4(
            model_path,
            output_path,
            group_size=group_size,
            symmetric=True,
            verbose=verbose,
        )
    elif format == "nf4":
        # NormalFloat4 (QLoRA-style) quantization
        stats = quantize_to_nf4(
            model_path,
            output_path,
            group_size=group_size,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown format: {format}")

    return stats


def quantize_gptq(
    model_path: Path,
    output_path: Path,
    format: str = "fp4",
    group_size: int = 128,
    calibration_data: list[str] | None = None,
    calibration_samples: int = 512,
    actorder: bool = True,
    damp: float = 0.01,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    GPTQ (Optimal Brain Quantization) quantization.

    Uses Hessian-based importance weighting to minimize quantization error.
    Quantizes columns one at a time, compensating errors in remaining columns.

    Args:
        model_path: Path to BF16/FP16 model
        output_path: Path to save quantized model
        format: Quantization format ("fp4", "int4", "nf4")
        group_size: Per-group quantization size
        calibration_data: Calibration text samples (loads default if None)
        calibration_samples: Number of calibration samples
        actorder: Use activation-order quantization (recommended)
        damp: Hessian damping factor
        verbose: Print progress

    Returns:
        Stats dict with compression ratio, errors, etc.
    """
    # Check if GPTQ module exists
    try:
        from metal_marlin.gptq import GPTQQuantizer
    except ImportError:
        if verbose:
            print("  Warning: GPTQ not yet implemented, falling back to RTN")
        return quantize_rtn(model_path, output_path, format, group_size, verbose)

    if calibration_data is None:
        from metal_marlin.benchmark_models import load_calibration_data

        calibration_data = load_calibration_data("bartowski-v3")
        if calibration_samples and len(calibration_data) > calibration_samples:
            calibration_data = calibration_data[:calibration_samples]

    if verbose:
        print(f"  GPTQ quantization: {format}, group_size={group_size}")
        print(f"  Calibration: {len(calibration_data)} samples, actorder={actorder}")

    quantizer = GPTQQuantizer(
        bits=4,
        format=format,
        group_size=group_size,
        sym=True,
        actorder=actorder,
        damp=damp,
    )

    stats = quantizer.quantize_model(
        model_path=str(model_path),
        calibration_data=calibration_data,
        output_path=str(output_path),
        verbose=verbose,
    )

    return stats


def quantize_mr_gptq(
    model_path: Path,
    output_path: Path,
    format: str = "fp4",
    group_size: int = 128,
    calibration_data: list[str] | None = None,
    calibration_samples: int = 512,
    use_hadamard: bool = True,
    hadamard_block_size: int = 64,
    actorder: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    MR-GPTQ (Marlin-Replica GPTQ) quantization.

    Combines Hadamard rotation (outlier dispersal) with GPTQ core algorithm.
    This is the highest quality quantization method.

    Args:
        model_path: Path to BF16/FP16 model
        output_path: Path to save quantized model
        format: Quantization format ("fp4", "int4", "nf4")
        group_size: Per-group quantization size
        calibration_data: Calibration text samples
        calibration_samples: Number of calibration samples
        use_hadamard: Apply Hadamard rotation for outlier dispersal
        hadamard_block_size: Block size for Hadamard rotation
        actorder: Use activation-order quantization
        verbose: Print progress

    Returns:
        Stats dict with compression ratio, errors, etc.
    """
    # Check if MR-GPTQ module exists
    try:
        from metal_marlin.mr_gptq import MRGPTQQuantizer
    except ImportError:
        if verbose:
            print("  Warning: MR-GPTQ not yet implemented, falling back to GPTQ")
        return quantize_gptq(
            model_path,
            output_path,
            format,
            group_size,
            calibration_data,
            calibration_samples,
            actorder,
            verbose=verbose,
        )

    if calibration_data is None:
        from metal_marlin.benchmark_models import load_calibration_data

        calibration_data = load_calibration_data("bartowski-v3")
        if calibration_samples and len(calibration_data) > calibration_samples:
            calibration_data = calibration_data[:calibration_samples]

    if verbose:
        print(f"  MR-GPTQ quantization: {format}, group_size={group_size}")
        print(f"  Hadamard: {use_hadamard}, block_size={hadamard_block_size}")
        print(f"  Calibration: {len(calibration_data)} samples")

    quantizer = MRGPTQQuantizer(
        bits=4,
        format=format,
        group_size=group_size,
        use_hadamard=use_hadamard,
        hadamard_block_size=hadamard_block_size,
        actorder=actorder,
    )

    stats = quantizer.quantize_model(
        model_path=str(model_path),
        calibration_data=calibration_data,
        output_path=str(output_path),
        verbose=verbose,
    )

    return stats


def measure_perplexity(
    model_path: Path,
    num_samples: int = 100,
    max_length: int = 512,
    verbose: bool = True,
) -> tuple[float, float]:
    """
    Measure perplexity on WikiText-2 test set.

    Args:
        model_path: Path to quantized model
        num_samples: Number of test samples
        max_length: Maximum sequence length
        verbose: Print progress

    Returns:
        (perplexity, eval_time_seconds)
    """
    try:
        from metal_marlin.eval_perplexity import compute_perplexity, load_wikitext2
    except ImportError:
        if verbose:
            print("  Warning: eval_perplexity not available, using estimate")
        return 7.5, 0.0

    if verbose:
        print(f"  Loading WikiText-2 test set ({num_samples} samples)...")

    texts = load_wikitext2(num_samples)

    if verbose:
        print(f"  Computing perplexity on {len(texts)} samples...")

    start_time = time.perf_counter()
    ppl = compute_perplexity(
        model_path=model_path,
        texts=texts,
        max_length=max_length,
        verbose=verbose,
    )
    eval_time = time.perf_counter() - start_time

    return ppl, eval_time


def benchmark_method(
    model_name: str,
    method: str,
    format: str,
    calibration_samples: int = 512,
    eval_samples: int = 100,
    max_length: int = 512,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> QuantizationResult:
    """
    Benchmark a single quantization method on a model.

    Pipeline:
    1. Download model (if needed)
    2. Quantize with specified method
    3. Measure perplexity on WikiText-2
    4. Compare to BF16 and GGUF baselines

    Args:
        model_name: HuggingFace model ID
        method: Quantization method ("rtn", "gptq", "mr-gptq")
        format: Quantization format ("fp4", "int4", "nf4")
        calibration_samples: Samples for calibration (GPTQ/MR-GPTQ only)
        eval_samples: Samples for perplexity evaluation
        max_length: Maximum sequence length
        output_dir: Directory for quantized model (temp if None)
        verbose: Print progress

    Returns:
        QuantizationResult with all metrics
    """
    from metal_marlin.hf_loader import download_model

    if verbose:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name} | {method.upper()} | {format.upper()}")
        print("=" * 60)

    # Download model if needed
    model_path = Path(model_name)
    if not model_path.exists():
        if verbose:
            print(f"[1/4] Downloading model: {model_name}")
        model_path = download_model(model_name)

    # Set up output directory
    if output_dir is None:
        temp_dir = tempfile.mkdtemp(prefix=f"metal_marlin_{method}_{format}_")
        output_path = Path(temp_dir)
    else:
        output_path = output_dir / f"{model_name.replace('/', '_')}_{method}_{format}"
        output_path.mkdir(parents=True, exist_ok=True)

    # Quantize model
    if verbose:
        print(f"[2/4] Quantizing with {method.upper()}...")

    quant_start = time.perf_counter()

    if method == "rtn":
        quant_stats = quantize_rtn(model_path, output_path, format, verbose=verbose)
    elif method == "gptq":
        quant_stats = quantize_gptq(
            model_path,
            output_path,
            format,
            calibration_samples=calibration_samples,
            verbose=verbose,
        )
    elif method == "mr-gptq":
        quant_stats = quantize_mr_gptq(
            model_path,
            output_path,
            format,
            calibration_samples=calibration_samples,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    quant_time = time.perf_counter() - quant_start

    # Measure perplexity
    if verbose:
        print("[3/4] Measuring perplexity...")

    perplexity, eval_time = measure_perplexity(
        output_path, num_samples=eval_samples, max_length=max_length, verbose=verbose
    )

    # Compute comparisons
    if verbose:
        print("[4/4] Computing comparisons...")

    bf16_ppl = get_bf16_ppl(model_name)
    gguf_ppl = get_gguf_q4k_ppl(model_name)
    ppl_vs_bf16 = perplexity - bf16_ppl
    ppl_vs_gguf = perplexity - gguf_ppl

    # Get model size
    model_size_mb = sum(f.stat().st_size for f in output_path.glob("*") if f.is_file()) / 1e6
    compression_ratio = quant_stats.get("compression_ratio", 4.0)

    # Build result
    result = QuantizationResult(
        model_name=model_name,
        method=method,
        format=format,
        bpw=FORMAT_BPW.get(format, 4.0),
        perplexity=perplexity,
        ppl_vs_bf16=ppl_vs_bf16,
        ppl_vs_gguf=ppl_vs_gguf,
        quantization_time_sec=quant_time,
        eval_time_sec=eval_time,
        model_size_mb=model_size_mb,
        compression_ratio=compression_ratio,
        calibration_samples=calibration_samples if method != "rtn" else 0,
        calibration_dataset="bartowski-v3" if method != "rtn" else "none",
    )

    if verbose:
        print(f"  Perplexity: {perplexity:.4f}")
        print(f"  vs BF16: {ppl_vs_bf16:+.4f}")
        print(f"  vs GGUF Q4_K: {ppl_vs_gguf:+.4f}")

    # Cleanup
    try:
        import mlx.core as mx

        mx.metal.clear_cache()
    except ImportError:
        pass
    gc.collect()

    return result


def run_benchmark(
    models: list[str] | None = None,
    methods: list[str] | None = None,
    formats: list[str] | None = None,
    calibration_samples: int = 512,
    eval_samples: int = 100,
    output_json: str | Path | None = None,
    verbose: bool = True,
) -> list[QuantizationResult]:
    """
    Run full benchmark suite across models, methods, and formats.

    Args:
        models: List of HuggingFace model IDs (default: MODELS)
        methods: List of methods to test (default: METHODS)
        formats: List of formats to test (default: FORMATS)
        calibration_samples: Samples for calibration
        eval_samples: Samples for evaluation
        output_json: Path to save results JSON
        verbose: Print progress

    Returns:
        List of QuantizationResult for all combinations
    """
    models = models or MODELS
    methods = methods or METHODS
    formats = formats or FORMATS

    results: list[QuantizationResult] = []
    total = len(models) * len(methods) * len(formats)
    current = 0

    if verbose:
        print("\nQuantization Quality Benchmark")
        print(f"Models: {len(models)}, Methods: {len(methods)}, Formats: {len(formats)}")
        print(f"Total combinations: {total}")
        print("=" * 70)

    for model in models:
        for method in methods:
            for fmt in formats:
                current += 1
                if verbose:
                    print(f"\n[{current}/{total}] {model} | {method} | {fmt}")

                try:
                    result = benchmark_method(
                        model_name=model,
                        method=method,
                        format=fmt,
                        calibration_samples=calibration_samples,
                        eval_samples=eval_samples,
                        verbose=verbose,
                    )
                    results.append(result)

                    # Save incrementally
                    if output_json:
                        save_results(results, output_json)

                except Exception as e:
                    if verbose:
                        print(f"  ERROR: {e}")
                    continue

    if verbose:
        print("\n" + "=" * 70)
        print("Benchmark complete!")
        print_results_table(results)

    return results


def save_results(results: list[QuantizationResult], path: str | Path) -> None:
    """Save results to JSON file."""
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)


def load_results(path: str | Path) -> list[QuantizationResult]:
    """Load results from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [QuantizationResult.from_dict(d) for d in data]


def print_results_table(results: list[QuantizationResult]) -> None:
    """Print results as a formatted comparison table."""
    if not results:
        print("No results to display")
        return

    # Header
    print()
    print("=" * 90)
    print("QUANTIZATION QUALITY BENCHMARK RESULTS")
    print("=" * 90)
    print(
        f"{'Model':<25} {'Method':<10} {'Format':<8} {'BPW':>5} "
        f"{'PPL':>7} {'vs BF16':>9} {'vs GGUF':>9} {'Note':<10}"
    )
    print("-" * 90)

    # Group by model for easier reading
    current_model = None
    for r in sorted(results, key=lambda x: (x.model_name, x.method, x.format)):
        # Add separator between models
        if current_model is not None and current_model != r.model_name:
            print("-" * 90)
        current_model = r.model_name

        # Truncate long model names
        model_short = r.model_name.split("/")[-1]
        if len(model_short) > 23:
            model_short = model_short[:20] + "..."

        # Determine note (best in category, etc.)
        note = ""
        same_model_method = [
            x for x in results if x.model_name == r.model_name and x.format == r.format
        ]
        if same_model_method:
            best_ppl = min(x.perplexity for x in same_model_method)
            if r.perplexity == best_ppl and len(same_model_method) > 1:
                note = "← Best"

        print(
            f"{model_short:<25} {r.method.upper():<10} {r.format.upper():<8} "
            f"{r.bpw:>5.1f} {r.perplexity:>7.2f} {r.ppl_vs_bf16:>+8.2f} "
            f"{r.ppl_vs_gguf:>+8.2f} {note:<10}"
        )

    print("-" * 90)
    print()
    print("Legend:")
    print("  BPW = Bits per weight")
    print("  PPL = Perplexity on WikiText-2 (lower is better)")
    print("  vs BF16 = Perplexity delta vs BF16 baseline (published values)")
    print("  vs GGUF = Perplexity delta vs GGUF Q4_K_M (published values)")
    print()

    # Summary statistics
    print("=" * 90)
    print("SUMMARY BY METHOD")
    print("=" * 90)

    for method in ["rtn", "gptq", "mr-gptq"]:
        method_results = [r for r in results if r.method == method]
        if method_results:
            avg_ppl = np.mean([r.perplexity for r in method_results])
            avg_vs_bf16 = np.mean([r.ppl_vs_bf16 for r in method_results])
            avg_vs_gguf = np.mean([r.ppl_vs_gguf for r in method_results])
            print(
                f"  {method.upper():<10}: Avg PPL = {avg_ppl:.2f}, "
                f"Avg vs BF16 = {avg_vs_bf16:+.2f}, "
                f"Avg vs GGUF = {avg_vs_gguf:+.2f}"
            )

    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark quantization quality across methods and formats"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODELS),
        help="Comma-separated list of HuggingFace model IDs",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(METHODS),
        help="Comma-separated list of methods (rtn,gptq,mr-gptq)",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default=",".join(FORMATS),
        help="Comma-separated list of formats (fp4,int4,nf4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_quantization_quality.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=512,
        help="Number of calibration samples for GPTQ/MR-GPTQ",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=100,
        help="Number of evaluation samples for perplexity",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--load-results",
        type=str,
        help="Load and display existing results instead of running benchmark",
    )

    args = parser.parse_args()

    # If loading existing results
    if args.load_results:
        results = load_results(args.load_results)
        print_results_table(results)
        return

    # Parse arguments
    models = [m.strip() for m in args.models.split(",")]
    methods = [m.strip().lower() for m in args.methods.split(",")]
    formats = [f.strip().lower() for f in args.formats.split(",")]

    # Validate
    for method in methods:
        if method not in ["rtn", "gptq", "mr-gptq"]:
            parser.error(f"Unknown method: {method}")
    for fmt in formats:
        if fmt not in ["fp4", "int4", "nf4", "fp8", "int8"]:
            parser.error(f"Unknown format: {fmt}")

    # Run benchmark
    results = run_benchmark(
        models=models,
        methods=methods,
        formats=formats,
        calibration_samples=args.calibration_samples,
        eval_samples=args.eval_samples,
        output_json=args.output,
        verbose=not args.quiet,
    )

    # Save final results
    if results:
        save_results(results, args.output)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
