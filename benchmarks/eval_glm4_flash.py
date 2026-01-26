#!/usr/bin/env python3
"""
GLM-4.7-Flash MLA (Multi-head Latent Attention) Layer Sensitivity Evaluation.

GLM-4.7-Flash uses a novel MLA architecture with compressed KV cache via learned
latent projections. This differs from standard MHA/GQA in several ways:

1. **Latent Bottleneck**: Instead of full KV projections, uses compressed latent
   representations with `kv_lora_rank=512` (vs full hidden_size).
2. **Separate Q/K/V with Compression**: Q projection uses `q_lora_rank=1536` for
   intermediate representation before final query projection.
3. **RoPE Scaling**: Uses `rope_ratio` for position embedding scaling with the
   compressed representations.

Key MLA-specific layer patterns:
- `q_a_proj`: Query LoRA down-projection (hidden → q_lora_rank)
- `q_b_proj`: Query LoRA up-projection (q_lora_rank → num_heads * head_dim)
- `kv_a_proj`: KV LoRA down-projection (hidden → kv_lora_rank + qk_rope_head_dim)
- `kv_b_proj`: KV LoRA up-projection (kv_lora_rank → num_heads * 2 * head_dim)
- `o_proj`: Output projection (standard)

This evaluation tests:
1. Which MLA layers are most sensitive to quantization
2. Whether Hadamard rotation benefits latent projections
3. Optimal precision/group_size for KV compression layers

Usage:
    cd contrib/metal_marlin
    uv run python benchmarks/eval_glm4_flash.py --samples 50
    uv run python benchmarks/eval_glm4_flash.py --full  # Comprehensive analysis
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Add metal_marlin to path (standalone project)
# Import directly from submodules to avoid circular imports in __init__.py
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# Direct imports from submodules to avoid package __init__ circular imports
from metal_marlin.hadamard import (
    apply_hadamard_rotation,
    compute_outlier_stats,
)
from metal_marlin.quantize_fp4 import (
    compute_quantization_error,
    quantize_fp4,
)

# Lazy import for hf_loader (may pull in heavy dependencies)
_hf_loader = None


def _get_hf_loader():
    """Lazy import hf_loader to avoid import issues."""
    global _hf_loader
    if _hf_loader is None:
        from metal_marlin import hf_loader as _hf_mod
        _hf_loader = _hf_mod
    return _hf_loader


def download_model(model_id: str) -> Path:
    """Download model from HuggingFace."""
    return _get_hf_loader().download_model(model_id)


def iter_safetensors_weights(model_path):
    """Iterate over safetensors weights."""
    return _get_hf_loader().iter_safetensors_weights(model_path)


def load_model_config(model_path):
    """Load model config."""
    return _get_hf_loader().load_model_config(model_path)

# MLA-specific layer patterns
MLA_LAYER_PATTERNS = {
    # Query latent projections (most sensitive to positional information)
    "q_a_proj": ["q_a_proj", "q_down_proj", "query_a_proj"],
    "q_b_proj": ["q_b_proj", "q_up_proj", "query_b_proj"],
    # KV latent projections (critical for compressed cache)
    "kv_a_proj": ["kv_a_proj", "kv_down_proj", "kv_a_layernorm"],
    "kv_b_proj": ["kv_b_proj", "kv_up_proj"],
    # Standard attention output
    "o_proj": ["o_proj", "out_proj", "dense"],
    # MoE components (GLM-4.7 has MoE too)
    "router": ["router", "gate.weight", "moe_gate"],
    "experts": ["experts.", "expert."],
    "shared_expert": ["shared_expert", "shared_experts"],
}


def classify_mla_layer(name: str) -> str:
    """Classify a layer name into MLA-specific category."""
    name_lower = name.lower()
    for category, patterns in MLA_LAYER_PATTERNS.items():
        for pattern in patterns:
            if pattern in name_lower:
                return category
    return "other"


@dataclass
class MLALayerAnalysis:
    """Analysis results for a single MLA layer."""

    name: str
    category: str
    shape: tuple[int, int]
    params: int

    # Outlier statistics (before/after Hadamard)
    outlier_stats_original: dict[str, float]
    outlier_stats_hadamard: dict[str, float]
    hadamard_benefit: float  # max_mean_ratio reduction

    # Quantization error (FP4 with different group sizes)
    quant_error_g64: dict[str, float]
    quant_error_g128: dict[str, float]
    quant_error_g256: dict[str, float]

    # Quantization error with Hadamard + FP4
    quant_error_hadamard_g64: dict[str, float]
    quant_error_hadamard_g128: dict[str, float]

    # Recommended config
    recommended_group_size: int
    recommended_use_hadamard: bool
    sensitivity_score: float  # Higher = more sensitive


@dataclass
class MLAEvalConfig:
    """Configuration for MLA evaluation."""

    model_id: str = "THUDM/glm-4-9b-chat"  # Fallback model
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    num_samples: int = 50
    test_group_sizes: list[int] = field(default_factory=lambda: [64, 128, 256])
    hadamard_block_size: int = 64


@dataclass
class MLAEvalResults:
    """Complete MLA evaluation results."""

    model_id: str
    model_type: str
    architecture: str
    kv_lora_rank: int
    q_lora_rank: int
    total_params: int

    # Layer-level analysis
    layers: list[dict[str, Any]]

    # Category summaries
    category_summary: dict[str, dict[str, Any]]

    # Overall recommendations
    recommendations: dict[str, Any]

    # Benchmark metrics
    timestamp: str
    eval_duration_s: float


def analyze_layer_sensitivity(
    name: str,
    tensor: np.ndarray,
    config: MLAEvalConfig,
) -> MLALayerAnalysis:
    """
    Analyze quantization sensitivity for a single layer.

    Tests:
    1. Outlier statistics before/after Hadamard
    2. FP4 quantization error at multiple group sizes
    3. Hadamard + FP4 quantization error
    """
    category = classify_mla_layer(name)
    out_feat, in_feat = tensor.shape
    params = tensor.size

    # Ensure tensor is float32 for analysis
    w = tensor.astype(np.float32)

    # 1. Outlier statistics (original)
    outlier_original = compute_outlier_stats(w)

    # 2. Hadamard rotation
    # Pad to multiple of block_size if needed
    block_size = config.hadamard_block_size
    if in_feat % block_size != 0:
        pad_size = block_size - (in_feat % block_size)
        w_padded = np.pad(w, ((0, 0), (0, pad_size)), mode="constant")
    else:
        w_padded = w
        pad_size = 0

    w_hadamard, h_meta = apply_hadamard_rotation(w_padded, block_size=block_size, axis=0)

    # Outlier stats after Hadamard
    outlier_hadamard = compute_outlier_stats(w_hadamard)

    # Hadamard benefit = ratio reduction
    benefit = (
        outlier_original["max_mean_ratio"] - outlier_hadamard["max_mean_ratio"]
    ) / outlier_original["max_mean_ratio"] * 100

    # 3. Quantization error at different group sizes
    quant_errors = {}
    for gs in config.test_group_sizes:
        if in_feat % gs == 0:
            packed, scales = quantize_fp4(w, group_size=gs)
            err = compute_quantization_error(w, packed, scales, gs)
            quant_errors[f"g{gs}"] = err
        else:
            quant_errors[f"g{gs}"] = {"rmse": float("nan"), "max_error": float("nan")}

    # 4. Hadamard + FP4 quantization
    hadamard_quant_errors = {}
    for gs in [64, 128]:
        if w_hadamard.shape[1] % gs == 0:
            packed, scales = quantize_fp4(w_hadamard, group_size=gs)
            err = compute_quantization_error(w_hadamard, packed, scales, gs)
            hadamard_quant_errors[f"g{gs}"] = err
        else:
            hadamard_quant_errors[f"g{gs}"] = {"rmse": float("nan"), "max_error": float("nan")}

    # 5. Determine recommendations
    # Compare RMSE at g128 with and without Hadamard
    rmse_plain_g128 = quant_errors.get("g128", {}).get("rmse", float("inf"))
    rmse_hadamard_g128 = hadamard_quant_errors.get("g128", {}).get("rmse", float("inf"))

    use_hadamard = rmse_hadamard_g128 < rmse_plain_g128 * 0.95  # 5% improvement threshold

    # Choose group size based on RMSE threshold
    if quant_errors.get("g64", {}).get("rmse", float("inf")) < 0.01:
        recommended_gs = 64
    elif quant_errors.get("g128", {}).get("rmse", float("inf")) < 0.02:
        recommended_gs = 128
    else:
        recommended_gs = 64  # Default to tighter for sensitive layers

    # Sensitivity score based on RMSE and outlier ratio
    sensitivity = (
        outlier_original["max_mean_ratio"] * 0.3 +
        quant_errors.get("g128", {}).get("rmse", 0.1) * 100 * 0.7
    )

    return MLALayerAnalysis(
        name=name,
        category=category,
        shape=(out_feat, in_feat),
        params=params,
        outlier_stats_original=outlier_original,
        outlier_stats_hadamard=outlier_hadamard,
        hadamard_benefit=benefit,
        quant_error_g64=quant_errors.get("g64", {}),
        quant_error_g128=quant_errors.get("g128", {}),
        quant_error_g256=quant_errors.get("g256", {}),
        quant_error_hadamard_g64=hadamard_quant_errors.get("g64", {}),
        quant_error_hadamard_g128=hadamard_quant_errors.get("g128", {}),
        recommended_group_size=recommended_gs,
        recommended_use_hadamard=use_hadamard,
        sensitivity_score=sensitivity,
    )


def run_mla_evaluation(config: MLAEvalConfig) -> MLAEvalResults:
    """
    Run comprehensive MLA layer sensitivity evaluation.

    Steps:
    1. Load model weights from HuggingFace
    2. Identify and analyze MLA-specific layers
    3. Test Hadamard benefit on each layer type
    4. Generate recommendations for optimal quantization
    """
    from datetime import datetime

    start_time = time.perf_counter()

    print("=" * 70)
    print("GLM-4.7-Flash MLA Layer Sensitivity Evaluation")
    print("=" * 70)

    # Try to load model
    print(f"\nModel: {config.model_id}")
    try:
        model_path = download_model(config.model_id)
        model_cfg = load_model_config(model_path)
        print(f"  Hidden size: {model_cfg.hidden_size}")
        print(f"  Layers: {model_cfg.num_hidden_layers}")
        print(f"  Model type: {model_cfg.model_type}")

        # Check for MLA-specific config
        if hasattr(model_cfg, "kv_lora_rank"):
            config.kv_lora_rank = model_cfg.kv_lora_rank
            print(f"  KV LoRA rank: {config.kv_lora_rank}")
        if hasattr(model_cfg, "q_lora_rank"):
            config.q_lora_rank = model_cfg.q_lora_rank
            print(f"  Q LoRA rank: {config.q_lora_rank}")

    except Exception as e:
        print(f"  Warning: Could not load model: {e}")
        print("  Using synthetic analysis with expected MLA structure")

        # Create synthetic test tensors matching expected MLA dimensions
        return _run_synthetic_mla_eval(config)

    # Analyze each layer
    print("\n" + "=" * 70)
    print("Step 1: Analyzing Layer Sensitivity")
    print("=" * 70)

    layer_analyses = []
    category_stats = {}

    for name, tensor, _ in iter_safetensors_weights(model_path):
        # Skip non-weight tensors
        if "weight" not in name.lower():
            continue

        # Skip 1D tensors (biases, norms)
        if tensor.ndim != 2:
            continue

        out_feat, in_feat = tensor.shape

        # Skip very small tensors
        if in_feat < 64:
            continue

        category = classify_mla_layer(name)

        # Analyze layer
        print(f"  Analyzing: {name} [{category}] {tensor.shape}")
        try:
            analysis = analyze_layer_sensitivity(name, tensor, config)
            layer_analyses.append(analysis)

            # Update category stats
            if category not in category_stats:
                category_stats[category] = {
                    "count": 0,
                    "total_params": 0,
                    "avg_hadamard_benefit": 0.0,
                    "avg_rmse_g128": 0.0,
                    "avg_sensitivity": 0.0,
                }
            cat_stat = category_stats[category]
            cat_stat["count"] += 1
            cat_stat["total_params"] += analysis.params
            cat_stat["avg_hadamard_benefit"] += analysis.hadamard_benefit
            cat_stat["avg_rmse_g128"] += analysis.quant_error_g128.get("rmse", 0.0)
            cat_stat["avg_sensitivity"] += analysis.sensitivity_score

        except Exception as e:
            print(f"    Warning: Failed to analyze {name}: {e}")

    # Finalize category averages
    for cat_stat in category_stats.values():
        if cat_stat["count"] > 0:
            cat_stat["avg_hadamard_benefit"] /= cat_stat["count"]
            cat_stat["avg_rmse_g128"] /= cat_stat["count"]
            cat_stat["avg_sensitivity"] /= cat_stat["count"]

    # Generate recommendations
    print("\n" + "=" * 70)
    print("Step 2: Generating Recommendations")
    print("=" * 70)

    recommendations = _generate_recommendations(category_stats, layer_analyses)

    # Print summary
    _print_summary(category_stats, recommendations)

    # Build results
    eval_duration = time.perf_counter() - start_time

    results = MLAEvalResults(
        model_id=config.model_id,
        model_type=model_cfg.model_type if 'model_cfg' in dir() else "unknown",
        architecture="MLA (Multi-head Latent Attention)",
        kv_lora_rank=config.kv_lora_rank,
        q_lora_rank=config.q_lora_rank,
        total_params=sum(a.params for a in layer_analyses),
        layers=[asdict(a) for a in layer_analyses],
        category_summary=category_stats,
        recommendations=recommendations,
        timestamp=datetime.now().isoformat(),
        eval_duration_s=eval_duration,
    )

    return results


def _generate_recommendations(
    category_stats: dict[str, dict[str, Any]],
    layer_analyses: list[MLALayerAnalysis],
) -> dict[str, Any]:
    """Generate quantization recommendations based on analysis."""
    recommendations = {
        "by_category": {},
        "summary": {},
    }

    # MLA-specific recommendations based on sensitivity
    mla_categories = ["q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj", "o_proj"]

    for cat in mla_categories:
        if cat not in category_stats:
            continue

        stat = category_stats[cat]
        avg_benefit = stat["avg_hadamard_benefit"]
        avg_rmse = stat["avg_rmse_g128"]
        avg_sensitivity = stat["avg_sensitivity"]

        # Determine recommended config
        if avg_sensitivity > 5.0 or avg_rmse > 0.02:
            # High sensitivity - use FP16 or tight FP4
            rec = {
                "precision": "fp16" if avg_rmse > 0.05 else "fp4",
                "group_size": 32 if avg_rmse > 0.02 else 64,
                "use_hadamard": avg_benefit > 10,
                "rationale": f"High sensitivity (score={avg_sensitivity:.1f}, RMSE={avg_rmse:.4f})",
            }
        elif avg_sensitivity > 2.0:
            # Medium sensitivity
            rec = {
                "precision": "fp4",
                "group_size": 64,
                "use_hadamard": avg_benefit > 5,
                "rationale": f"Medium sensitivity (score={avg_sensitivity:.1f})",
            }
        else:
            # Low sensitivity - aggressive quantization ok
            rec = {
                "precision": "fp4",
                "group_size": 128,
                "use_hadamard": avg_benefit > 15,
                "rationale": f"Low sensitivity (score={avg_sensitivity:.1f})",
            }

        recommendations["by_category"][cat] = rec

    # Latent projection specific recommendations
    recommendations["summary"] = {
        "kv_compression_layers": {
            "description": "kv_a_proj and kv_b_proj form the KV cache compression bottleneck",
            "recommendation": "Use FP4/g64 with Hadamard if benefit > 10%",
            "note": "These layers directly impact KV cache memory and attention quality",
        },
        "query_latent_layers": {
            "description": "q_a_proj and q_b_proj form the query compression path",
            "recommendation": "Generally more sensitive than KV; prefer FP4/g64 or FP16 for q_a_proj",
            "note": "Query projection errors directly propagate to attention scores",
        },
        "output_projection": {
            "description": "Standard o_proj layer",
            "recommendation": "FP4/g128 usually sufficient; less sensitive than Q/KV paths",
        },
        "hadamard_benefit": {
            "description": "Hadamard rotation disperses outliers before quantization",
            "most_beneficial": [
                cat for cat, stat in category_stats.items()
                if stat.get("avg_hadamard_benefit", 0) > 10
            ],
            "least_beneficial": [
                cat for cat, stat in category_stats.items()
                if stat.get("avg_hadamard_benefit", 0) < 5
            ],
        },
    }

    return recommendations


def _print_summary(
    category_stats: dict[str, dict[str, Any]],
    recommendations: dict[str, Any],
) -> None:
    """Print analysis summary."""
    print("\nCategory Summary:")
    print("-" * 70)
    print(f"{'Category':<20} {'Count':>6} {'Params':>10} {'Hadamard':>10} {'RMSE g128':>10} {'Sensitivity':>12}")
    print("-" * 70)

    for cat, stat in sorted(category_stats.items(), key=lambda x: -x[1]["avg_sensitivity"]):
        print(
            f"{cat:<20} {stat['count']:>6} "
            f"{stat['total_params']/1e6:>9.1f}M "
            f"{stat['avg_hadamard_benefit']:>9.1f}% "
            f"{stat['avg_rmse_g128']:>10.4f} "
            f"{stat['avg_sensitivity']:>12.2f}"
        )

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR MLA LAYERS")
    print("=" * 70)

    for cat, rec in recommendations.get("by_category", {}).items():
        print(f"\n{cat}:")
        print(f"  Precision: {rec['precision']}")
        print(f"  Group size: {rec['group_size']}")
        print(f"  Use Hadamard: {rec['use_hadamard']}")
        print(f"  Rationale: {rec['rationale']}")

    # Highlight key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    summary = recommendations.get("summary", {})

    if summary.get("hadamard_benefit", {}).get("most_beneficial"):
        print(f"\nHadamard most beneficial for: {', '.join(summary['hadamard_benefit']['most_beneficial'])}")

    if summary.get("hadamard_benefit", {}).get("least_beneficial"):
        print(f"Hadamard least beneficial for: {', '.join(summary['hadamard_benefit']['least_beneficial'])}")

    print("\nKV Compression Layers:")
    kv_rec = summary.get("kv_compression_layers", {})
    print(f"  {kv_rec.get('recommendation', 'N/A')}")
    print(f"  Note: {kv_rec.get('note', 'N/A')}")

    print("\nQuery Latent Layers:")
    q_rec = summary.get("query_latent_layers", {})
    print(f"  {q_rec.get('recommendation', 'N/A')}")
    print(f"  Note: {q_rec.get('note', 'N/A')}")


def _run_synthetic_mla_eval(config: MLAEvalConfig) -> MLAEvalResults:
    """
    Run evaluation with synthetic tensors matching expected MLA structure.

    Used when model download fails but we still want to demonstrate the analysis.
    """
    from datetime import datetime

    print("\nRunning synthetic MLA analysis...")
    print(f"  KV LoRA rank: {config.kv_lora_rank}")
    print(f"  Q LoRA rank: {config.q_lora_rank}")

    # Expected MLA dimensions for GLM-4.7-Flash
    hidden_size = 4096
    num_heads = 32
    head_dim = 128
    kv_lora_rank = config.kv_lora_rank
    q_lora_rank = config.q_lora_rank
    qk_rope_head_dim = 64  # Typical value

    # Synthetic layer configurations
    synthetic_layers = [
        ("model.layers.0.self_attn.q_a_proj.weight", (q_lora_rank, hidden_size)),
        ("model.layers.0.self_attn.q_b_proj.weight", (num_heads * head_dim, q_lora_rank)),
        ("model.layers.0.self_attn.kv_a_proj.weight", (kv_lora_rank + qk_rope_head_dim, hidden_size)),
        ("model.layers.0.self_attn.kv_b_proj.weight", (num_heads * 2 * head_dim, kv_lora_rank)),
        ("model.layers.0.self_attn.o_proj.weight", (hidden_size, num_heads * head_dim)),
    ]

    layer_analyses = []
    category_stats = {}

    for name, shape in synthetic_layers:
        # Generate synthetic weights with realistic distribution
        np.random.seed(42)  # Reproducibility
        tensor = np.random.randn(*shape).astype(np.float32) * 0.02

        # Add some outliers (typical for transformers)
        outlier_mask = np.random.rand(*shape) < 0.001
        tensor[outlier_mask] *= 10

        category = classify_mla_layer(name)
        print(f"  Analyzing: {name} [{category}] {shape}")

        try:
            analysis = analyze_layer_sensitivity(name, tensor, config)
            layer_analyses.append(analysis)

            # Update category stats
            if category not in category_stats:
                category_stats[category] = {
                    "count": 0,
                    "total_params": 0,
                    "avg_hadamard_benefit": 0.0,
                    "avg_rmse_g128": 0.0,
                    "avg_sensitivity": 0.0,
                }
            cat_stat = category_stats[category]
            cat_stat["count"] += 1
            cat_stat["total_params"] += analysis.params
            cat_stat["avg_hadamard_benefit"] += analysis.hadamard_benefit
            cat_stat["avg_rmse_g128"] += analysis.quant_error_g128.get("rmse", 0.0)
            cat_stat["avg_sensitivity"] += analysis.sensitivity_score

        except Exception as e:
            print(f"    Warning: Failed to analyze {name}: {e}")

    # Finalize averages
    for cat_stat in category_stats.values():
        if cat_stat["count"] > 0:
            cat_stat["avg_hadamard_benefit"] /= cat_stat["count"]
            cat_stat["avg_rmse_g128"] /= cat_stat["count"]
            cat_stat["avg_sensitivity"] /= cat_stat["count"]

    recommendations = _generate_recommendations(category_stats, layer_analyses)
    _print_summary(category_stats, recommendations)

    return MLAEvalResults(
        model_id=config.model_id,
        model_type="synthetic",
        architecture="MLA (Multi-head Latent Attention) - Synthetic",
        kv_lora_rank=config.kv_lora_rank,
        q_lora_rank=config.q_lora_rank,
        total_params=sum(a.params for a in layer_analyses),
        layers=[asdict(a) for a in layer_analyses],
        category_summary=category_stats,
        recommendations=recommendations,
        timestamp=datetime.now().isoformat(),
        eval_duration_s=0.0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GLM-4.7-Flash MLA layer sensitivity for quantization"
    )
    parser.add_argument(
        "--model",
        default="THUDM/glm-4-9b-chat",
        help="HuggingFace model ID (default: THUDM/glm-4-9b-chat)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples for perplexity (if testing inference)",
    )
    parser.add_argument(
        "--kv-lora-rank",
        type=int,
        default=512,
        help="KV LoRA rank for MLA (default: 512)",
    )
    parser.add_argument(
        "--q-lora-rank",
        type=int,
        default=1536,
        help="Q LoRA rank for MLA (default: 1536)",
    )
    parser.add_argument(
        "--hadamard-block",
        type=int,
        default=64,
        help="Hadamard block size (default: 64)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full analysis with all layers (slower)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: benchmarks/results/glm4_flash_mla.json)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic tensors instead of downloading model",
    )

    args = parser.parse_args()

    config = MLAEvalConfig(
        model_id=args.model,
        kv_lora_rank=args.kv_lora_rank,
        q_lora_rank=args.q_lora_rank,
        num_samples=args.samples,
        hadamard_block_size=args.hadamard_block,
    )

    if args.synthetic:
        results = _run_synthetic_mla_eval(config)
    else:
        results = run_mla_evaluation(config)

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = _ROOT / "benchmarks" / "results" / "glm4_flash_mla.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(asdict(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
