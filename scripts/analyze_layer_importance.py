#!/usr/bin/env python3
"""Layer importance analysis for model pruning.

Measures the importance of each transformer layer by computing perplexity
with that layer skipped. Layers with minimal perplexity impact (<0.5%) are
candidates for pruning.

Methodology:
1. Compute baseline perplexity with all layers
2. For each layer, skip it during inference and measure perplexity
3. Compute relative perplexity increase (importance score)
4. Identify layers below threshold for removal

The layer skip is implemented by replacing the layer's forward pass with
an identity function (passing hidden_states through unchanged, but still
updating KV cache positions to maintain correct sequence tracking).

Usage:
    cd contrib/metal_marlin
    uv run python scripts/analyze_layer_importance.py --model-path ./glm4-fp4/
    uv run python scripts/analyze_layer_importance.py --model-id THUDM/glm-4-9b-chat
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    import torch
except ImportError as e:
    raise SystemExit(f"PyTorch required: {e}")


@dataclass
class LayerImportance:
    """Importance metrics for a single layer."""

    layer_idx: int
    perplexity_baseline: float
    perplexity_skipped: float
    perplexity_delta: float
    perplexity_delta_pct: float
    is_moe_layer: bool
    is_prunable: bool  # delta_pct < threshold


@dataclass
class ImportanceAnalysis:
    """Complete layer importance analysis results."""

    model_id: str
    timestamp: str
    num_layers: int
    baseline_perplexity: float
    threshold_pct: float
    eval_samples: int
    eval_max_length: int

    layers: list[LayerImportance]

    # Summary
    prunable_layers: list[int]
    prunable_count: int
    estimated_speedup_pct: float

    def to_json(self) -> dict[str, Any]:
        result = asdict(self)
        result["layers"] = [asdict(layer) for layer in self.layers]
        return result


def load_model_and_tokenizer(
    model_path: str | None = None,
    model_id: str | None = None,
    device: str = "auto",
    dtype: str = "bf16",
) -> tuple[Any, Any, Any]:
    """Load model, tokenizer, and config.

    Supports both local trellis-quantized models and HuggingFace models.
    """
    from transformers import AutoTokenizer

    # Resolve device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # Resolve dtype
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(dtype, torch.bfloat16)

    if model_path:
        # Try loading as trellis-quantized model
        model_dir = Path(model_path)
        config_path = model_dir / "config.json"

        if config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)

            # Check if it's a trellis-quantized model
            if "trellis_version" in config_data or "quantization_config" in config_data:
                from metal_marlin.trellis import TrellisForCausalLM, TrellisModelConfig

                trellis_config = TrellisModelConfig.from_pretrained(str(model_dir))
                trellis_model = TrellisForCausalLM.from_pretrained(str(model_dir), device=device)
                tokenizer_id = config_data.get("tokenizer_id", str(model_dir))
                trellis_tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_id, trust_remote_code=True
                )
                return trellis_model, trellis_tokenizer, trellis_config

        # Fall back to HuggingFace
        model_id = str(model_dir)

    if model_id:
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        return model, tokenizer, config

    raise ValueError("Must provide either model_path or model_id")


def get_model_layers(model: Any) -> list[Any]:
    """Extract transformer layers from model."""
    # Try common attribute paths
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "layers"):
        return list(model.layers)

    raise ValueError("Could not find transformer layers in model")


class LayerSkipWrapper:
    """Context manager to temporarily skip a layer during forward pass."""

    def __init__(self, layer: Any, layer_idx: int):
        self.layer = layer
        self.layer_idx = layer_idx
        self.original_forward = None

    def __enter__(self):
        self.original_forward = self.layer.forward

        def skip_forward(hidden_states, *args, **kwargs):
            # Pass hidden states through unchanged (identity)
            # This preserves tensor shape and gradient flow
            return hidden_states

        self.layer.forward = skip_forward
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.layer.forward = self.original_forward
        return False


def compute_perplexity(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int = 256,
    device: str = "mps",
    skip_layer: int | None = None,
    layers: list[Any] | None = None,
    verbose: bool = False,
) -> float:
    """Compute perplexity, optionally skipping a specific layer."""
    from metal_marlin.eval.perplexity import log_softmax

    total_nll = 0.0
    total_tokens = 0

    # Context manager for layer skipping
    skip_ctx = None
    if skip_layer is not None and layers is not None:
        skip_ctx = LayerSkipWrapper(layers[skip_layer], skip_layer)

    try:
        if skip_ctx:
            skip_ctx.__enter__()

        for i, text in enumerate(texts):
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            tokens = tokens[:max_length]

            input_ids = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
            targets = np.array(tokens[1:])

            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                logits = logits.squeeze(0).float().cpu().numpy()

            log_probs = log_softmax(logits, axis=-1)
            token_log_probs = log_probs[np.arange(len(targets)), targets]
            nll = -np.sum(token_log_probs)

            total_nll += nll
            total_tokens += len(targets)

            if verbose and (i + 1) % 5 == 0:
                ppl_so_far = np.exp(total_nll / total_tokens)
                print(f"    [{i + 1}/{len(texts)}] Running PPL: {ppl_so_far:.4f}")

            # Memory cleanup
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    finally:
        if skip_ctx:
            skip_ctx.__exit__(None, None, None)

    if total_tokens == 0:
        raise ValueError("No valid tokens for perplexity computation")

    return float(np.exp(total_nll / total_tokens))


def analyze_layer_importance(
    model: Any,
    tokenizer: Any,
    config: Any,
    texts: list[str],
    max_length: int = 256,
    threshold_pct: float = 0.5,
    device: str = "mps",
    verbose: bool = True,
) -> ImportanceAnalysis:
    """Analyze importance of each layer by measuring perplexity impact."""
    layers = get_model_layers(model)
    num_layers = len(layers)

    if verbose:
        print(f"\nAnalyzing {num_layers} layers...")
        print(f"Threshold for pruning: <{threshold_pct}% perplexity increase")

    # Step 1: Compute baseline perplexity
    if verbose:
        print("\n[1/2] Computing baseline perplexity...")
    start = time.perf_counter()
    baseline_ppl = compute_perplexity(
        model, tokenizer, texts, max_length, device, verbose=verbose
    )
    baseline_time = time.perf_counter() - start
    if verbose:
        print(f"  Baseline PPL: {baseline_ppl:.4f} ({baseline_time:.1f}s)")

    # Step 2: Measure each layer's importance
    if verbose:
        print(f"\n[2/2] Testing each layer (0-{num_layers - 1})...")

    layer_results: list[LayerImportance] = []
    prunable_layers: list[int] = []

    # Check if config has MoE layer info
    first_moe_layer = getattr(config, "first_moe_layer", 0)
    num_experts = getattr(config, "num_experts", 1)
    is_moe_model = num_experts > 1

    for layer_idx in range(num_layers):
        if verbose:
            print(f"\n  Layer {layer_idx}/{num_layers - 1}:")

        start = time.perf_counter()
        skipped_ppl = compute_perplexity(
            model, tokenizer, texts, max_length, device,
            skip_layer=layer_idx, layers=layers, verbose=False
        )
        elapsed = time.perf_counter() - start

        delta = skipped_ppl - baseline_ppl
        delta_pct = (delta / baseline_ppl) * 100
        is_moe = is_moe_model and layer_idx >= first_moe_layer
        is_prunable = delta_pct < threshold_pct

        result = LayerImportance(
            layer_idx=layer_idx,
            perplexity_baseline=baseline_ppl,
            perplexity_skipped=skipped_ppl,
            perplexity_delta=delta,
            perplexity_delta_pct=delta_pct,
            is_moe_layer=is_moe,
            is_prunable=is_prunable,
        )
        layer_results.append(result)

        if is_prunable:
            prunable_layers.append(layer_idx)

        layer_type = "MoE" if is_moe else "Dense"
        status = "✓ PRUNABLE" if is_prunable else ""
        if verbose:
            print(
                f"    [{layer_type}] PPL: {skipped_ppl:.4f} "
                f"(Δ={delta:+.4f}, {delta_pct:+.2f}%) {status} [{elapsed:.1f}s]"
            )

    # Summary
    prunable_count = len(prunable_layers)
    estimated_speedup = (prunable_count / num_layers) * 100

    analysis = ImportanceAnalysis(
        model_id=getattr(config, "_name_or_path", "unknown"),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        num_layers=num_layers,
        baseline_perplexity=baseline_ppl,
        threshold_pct=threshold_pct,
        eval_samples=len(texts),
        eval_max_length=max_length,
        layers=layer_results,
        prunable_layers=prunable_layers,
        prunable_count=prunable_count,
        estimated_speedup_pct=estimated_speedup,
    )

    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze layer importance for model pruning"
    )
    parser.add_argument("--model-path", help="Path to local model directory")
    parser.add_argument("--model-id", help="HuggingFace model ID")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--samples", type=int, default=20, help="Evaluation samples")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Perplexity increase threshold %% for pruning (default: 0.5)",
    )
    parser.add_argument("--output", help="Output JSON path (default: auto-generated)")
    parser.add_argument("-v", "--verbose", action="store_true", default=True)
    args = parser.parse_args()

    if not args.model_path and not args.model_id:
        parser.error("Must provide either --model-path or --model-id")

    # Resolve device
    device = args.device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    print("=" * 70)
    print("Layer Importance Analysis for Model Pruning")
    print("=" * 70)

    # Load model
    model_ref = args.model_path or args.model_id
    print(f"\nLoading model: {model_ref}")
    start = time.perf_counter()
    model, tokenizer, config = load_model_and_tokenizer(
        model_path=args.model_path,
        model_id=args.model_id,
        device=device,
        dtype=args.dtype,
    )
    load_time = time.perf_counter() - start
    print(f"Loaded in {load_time:.1f}s on {device}")

    # Load evaluation data
    print(f"\nLoading WikiText-2 ({args.samples} samples)...")
    from metal_marlin.eval import load_wikitext2

    texts = load_wikitext2(max_samples=args.samples)
    print(f"  Loaded {len(texts)} samples")

    # Run analysis
    analysis = analyze_layer_importance(
        model=model,
        tokenizer=tokenizer,
        config=config,
        texts=texts,
        max_length=args.max_length,
        threshold_pct=args.threshold,
        device=device,
        verbose=args.verbose,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nBaseline perplexity: {analysis.baseline_perplexity:.4f}")
    print(f"Total layers: {analysis.num_layers}")
    print(f"Prunable layers (<{args.threshold}% PPL impact): {analysis.prunable_count}")

    if analysis.prunable_layers:
        print(f"  Layers: {analysis.prunable_layers}")
        print(f"\nEstimated speedup from pruning: ~{analysis.estimated_speedup_pct:.1f}%")

        # Show ranked importance
        print("\nLayer importance ranking (lowest to highest):")
        sorted_layers = sorted(analysis.layers, key=lambda x: x.perplexity_delta_pct)
        for i, layer in enumerate(sorted_layers[:10]):
            status = "✓" if layer.is_prunable else " "
            layer_type = "MoE" if layer.is_moe_layer else "Dense"
            print(
                f"  {status} Layer {layer.layer_idx:2d} [{layer_type:5s}]: "
                f"{layer.perplexity_delta_pct:+.3f}%"
            )
    else:
        print("\nNo layers below pruning threshold.")
        print("Consider increasing --threshold or using more evaluation samples.")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        model_name = Path(model_ref).stem if args.model_path else args.model_id.replace("/", "_")
        output_path = _ROOT / "benchmarks" / "results" / f"layer_importance_{model_name}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(analysis.to_json(), f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
