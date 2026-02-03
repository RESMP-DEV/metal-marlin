#!/usr/bin/env python3
"""Apply layer pruning based on importance analysis.

Takes a layer importance analysis JSON and creates a pruned model config.
Can also benchmark the speedup from pruning.

Usage:
    cd contrib/metal_marlin

    # Create pruned config from analysis
    uv run python scripts/prune_model.py apply \\
        --model-path ./glm4-fp4/ \\
        --analysis benchmarks/results/layer_importance_glm4.json

    # Benchmark original vs pruned
    uv run python scripts/prune_model.py benchmark \\
        --model-path ./glm4-fp4/ \\
        --analysis benchmarks/results/layer_importance_glm4.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    import torch
except ImportError as e:
    raise SystemExit(f"PyTorch required: {e}")


def load_analysis(analysis_path: str) -> dict:
    """Load layer importance analysis from JSON."""
    with open(analysis_path) as f:
        return json.load(f)


def get_prunable_layers(analysis: dict, threshold_pct: float | None = None) -> list[int]:
    """Extract prunable layers from analysis."""
    if threshold_pct is None:
        threshold_pct = analysis.get("threshold_pct", 0.5)

    prunable = []
    for layer in analysis.get("layers", []):
        delta_pct = layer.get("perplexity_delta_pct", float("inf"))
        if delta_pct < threshold_pct:
            prunable.append(layer["layer_idx"])

    return sorted(prunable)


def apply_pruning(args: argparse.Namespace) -> int:
    """Create pruned model configuration."""
    from metal_marlin.trellis import TrellisModelConfig

    print(f"Loading analysis from: {args.analysis}")
    analysis = load_analysis(args.analysis)

    threshold = args.threshold or analysis.get("threshold_pct", 0.5)
    prunable = get_prunable_layers(analysis, threshold)

    print("\nAnalysis summary:")
    print(f"  Baseline PPL: {analysis['baseline_perplexity']:.4f}")
    print(f"  Total layers: {analysis['num_layers']}")
    print(f"  Threshold: <{threshold}% PPL increase")
    print(f"  Prunable layers: {len(prunable)}")

    if not prunable:
        print("\nNo layers below threshold. Nothing to prune.")
        return 0

    print(f"  Layers to skip: {prunable}")

    # Load base config
    print(f"\nLoading base config from: {args.model_path}")
    base_config = TrellisModelConfig.from_pretrained(args.model_path)

    # Create pruned config
    pruned_config = base_config.prune_layers(prunable)

    # Save pruned config
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.model_path) / "config_pruned.json"

    # Convert to dict for JSON serialization
    config_dict = {
        "hidden_size": pruned_config.hidden_size,
        "num_hidden_layers": pruned_config.num_hidden_layers,
        "num_attention_heads": pruned_config.num_attention_heads,
        "num_kv_heads": pruned_config.num_kv_heads,
        "head_dim": pruned_config.head_dim,
        "intermediate_size": pruned_config.intermediate_size,
        "vocab_size": pruned_config.vocab_size,
        "kv_lora_rank": pruned_config.kv_lora_rank,
        "q_lora_rank": pruned_config.q_lora_rank,
        "num_experts": pruned_config.num_experts,
        "num_shared_experts": pruned_config.num_shared_experts,
        "num_experts_per_tok": pruned_config.num_experts_per_tok,
        "first_moe_layer": pruned_config.first_moe_layer,
        "rms_norm_eps": pruned_config.rms_norm_eps,
        "rope_theta": pruned_config.rope_theta,
        "max_position_embeddings": pruned_config.max_position_embeddings,
        "skip_layers": pruned_config.skip_layers,
        "_pruning_applied": True,
        "_pruning_threshold_pct": threshold,
        "_original_num_layers": analysis["num_layers"],
    }

    with open(output_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    speedup = len(prunable) / analysis["num_layers"] * 100
    print(f"\nPruned config saved to: {output_path}")
    print(f"Estimated speedup: ~{speedup:.1f}%")
    print(f"\nTo use: Load model with skip_layers={prunable}")

    return 0


def benchmark_pruning(args: argparse.Namespace) -> int:
    """Benchmark original vs pruned model."""
    import numpy as np

    from metal_marlin.eval import load_wikitext2
    from metal_marlin.eval.perplexity import log_softmax
    from metal_marlin.trellis import TrellisForCausalLM, TrellisModelConfig

    print(f"Loading analysis from: {args.analysis}")
    analysis = load_analysis(args.analysis)

    threshold = args.threshold or analysis.get("threshold_pct", 0.5)
    prunable = get_prunable_layers(analysis, threshold)

    print(f"\nPrunable layers ({len(prunable)}): {prunable}")

    # Resolve device
    device = args.device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    start = time.perf_counter()

    # Load with original config
    original_config = TrellisModelConfig.from_pretrained(args.model_path)
    model = TrellisForCausalLM.from_pretrained(args.model_path, device=device)
    load_time = time.perf_counter() - start
    print(f"Loaded in {load_time:.1f}s on {device}")

    # Load tokenizer
    from transformers import AutoTokenizer

    model_dir = Path(args.model_path)
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)
    tokenizer_id = config_data.get("tokenizer_id", str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

    # Load evaluation data
    print(f"\nLoading WikiText-2 ({args.samples} samples)...")
    texts = load_wikitext2(max_samples=args.samples)

    def benchmark_inference(skip_layers: list[int] | None = None) -> tuple[float, float]:
        """Benchmark inference speed and perplexity."""
        # Update config's skip_layers
        model.model.config.skip_layers = skip_layers

        total_tokens = 0
        total_nll = 0.0
        total_time = 0.0

        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            tokens = tokens[: args.max_length]

            input_ids = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
            targets = np.array(tokens[1:])

            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            elapsed = time.perf_counter() - start

            logits = logits.squeeze(0).float().cpu().numpy()
            log_probs = log_softmax(logits, axis=-1)
            token_log_probs = log_probs[np.arange(len(targets)), targets]

            total_nll += -np.sum(token_log_probs)
            total_tokens += len(targets)
            total_time += elapsed

        ppl = np.exp(total_nll / total_tokens)
        tokens_per_sec = total_tokens / total_time

        return ppl, tokens_per_sec

    # Warmup
    print("\nWarming up...")
    _ = benchmark_inference(skip_layers=None)
    gc.collect()

    # Benchmark original
    print("\nBenchmarking original model...")
    orig_ppl, orig_tps = benchmark_inference(skip_layers=None)
    print(f"  PPL: {orig_ppl:.4f}, Throughput: {orig_tps:.1f} tok/s")

    # Benchmark pruned
    print(f"\nBenchmarking pruned model (skip {len(prunable)} layers)...")
    pruned_ppl, pruned_tps = benchmark_inference(skip_layers=prunable)
    print(f"  PPL: {pruned_ppl:.4f}, Throughput: {pruned_tps:.1f} tok/s")

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    ppl_delta = pruned_ppl - orig_ppl
    ppl_delta_pct = (ppl_delta / orig_ppl) * 100
    speedup = (pruned_tps / orig_tps - 1) * 100

    print(f"\nOriginal:  PPL={orig_ppl:.4f}, {orig_tps:.1f} tok/s")
    print(f"Pruned:    PPL={pruned_ppl:.4f}, {pruned_tps:.1f} tok/s")
    print(f"\nPPL change: {ppl_delta:+.4f} ({ppl_delta_pct:+.2f}%)")
    print(f"Speedup: {speedup:+.1f}%")
    print(f"Layers removed: {len(prunable)}/{analysis['num_layers']}")

    # Reset config
    model.model.config.skip_layers = None

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply layer pruning based on importance analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Apply subcommand
    apply_parser = subparsers.add_parser("apply", help="Create pruned model config")
    apply_parser.add_argument("--model-path", required=True, help="Path to model directory")
    apply_parser.add_argument("--analysis", required=True, help="Path to importance analysis JSON")
    apply_parser.add_argument("--threshold", type=float, help="Override pruning threshold %%")
    apply_parser.add_argument("--output", help="Output config path (default: model_path/config_pruned.json)")

    # Benchmark subcommand
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark original vs pruned")
    bench_parser.add_argument("--model-path", required=True, help="Path to model directory")
    bench_parser.add_argument("--analysis", required=True, help="Path to importance analysis JSON")
    bench_parser.add_argument("--threshold", type=float, help="Override pruning threshold %%")
    bench_parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    bench_parser.add_argument("--samples", type=int, default=10, help="Evaluation samples")
    bench_parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")

    args = parser.parse_args()

    if args.command == "apply":
        return apply_pruning(args)
    elif args.command == "benchmark":
        return benchmark_pruning(args)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
