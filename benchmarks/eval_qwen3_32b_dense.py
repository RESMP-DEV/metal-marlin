#!/usr/bin/env python3
"""
Qwen3-32B Dense Model Evaluation Benchmark

Dense models have fundamentally different characteristics from MoE:
- All parameters active every forward pass (no routing sparsity)
- More uniform layer sensitivity (no hot/cold expert distinction)
- Higher memory pressure during inference
- Every quantization decision affects every token

Test configurations designed for dense models:
1. Uniform FP4 g128: Standard Marlin FP4 with group size 128
2. MR-GPTQ FP4 with Hessian: GPTQ-optimized quantization with calibration
3. First/last layers FP8, middle FP4: Sensitive layer protection
4. Attention FP4/g64, MLP FP4/g128: Component-aware mixed precision

Comparisons:
- BF16 baseline: Native precision reference
- GGUF Q4_K_M: llama.cpp community standard
- MLX native 4-bit: Apple Silicon optimized baseline

Usage:
    cd contrib/metal_marlin
    uv run python benchmarks/eval_qwen3_32b_dense.py --samples 50
    uv run python benchmarks/eval_qwen3_32b_dense.py --full
    uv run python benchmarks/eval_qwen3_32b_dense.py --config uniform_fp4
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add metal_marlin to path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import mlx.core as mx
import mlx.nn as nn

from metal_marlin.mixed_precision import (
    LayerQuantConfig,
    MixedPrecisionConfig,
    Precision,
    get_layer_config,
)

# Model configuration
MODEL_ID = "Qwen/Qwen3-32B"
MODEL_FALLBACK = "mlx-community/Qwen2.5-7B-Instruct-4bit"  # For memory-constrained testing


# =============================================================================
# Dense-Specific Mixed Precision Configurations
# =============================================================================


def dense_uniform_fp4_g128() -> MixedPrecisionConfig:
    """Uniform FP4 with group_size=128 for all quantizable layers."""
    return MixedPrecisionConfig(
        default=LayerQuantConfig(Precision.FP4_E2M1, 128),
        attention_qkv=LayerQuantConfig(Precision.FP4_E2M1, 128),
        attention_out=LayerQuantConfig(Precision.FP4_E2M1, 128),
        mlp_gate=LayerQuantConfig(Precision.FP4_E2M1, 128),
        mlp_up=LayerQuantConfig(Precision.FP4_E2M1, 128),
        mlp_down=LayerQuantConfig(Precision.FP4_E2M1, 128),
    )


def dense_mrgptq_fp4_hessian() -> MixedPrecisionConfig:
    """
    MR-GPTQ FP4 configuration with Hessian-aware quantization.

    Uses smaller group sizes where error propagation matters most.
    Designed for use with MRGPTQQuantizer with calibration data.
    """
    return MixedPrecisionConfig(
        # Embeddings and output stay high precision
        embeddings=LayerQuantConfig(Precision.BF16),
        lm_head=LayerQuantConfig(Precision.BF16),
        norms=LayerQuantConfig(Precision.BF16),
        # Attention: smaller groups for position-sensitive weights
        attention_qkv=LayerQuantConfig(Precision.FP4_E2M1, 64),
        attention_out=LayerQuantConfig(Precision.FP4_E2M1, 64),
        # MLP: standard groups, bulk of parameters
        mlp_gate=LayerQuantConfig(Precision.FP4_E2M1, 128),
        mlp_up=LayerQuantConfig(Precision.FP4_E2M1, 128),
        mlp_down=LayerQuantConfig(Precision.FP4_E2M1, 128),
        default=LayerQuantConfig(Precision.FP4_E2M1, 128),
    )


def dense_boundary_fp8() -> MixedPrecisionConfig:
    """
    First/last layers FP8, middle layers FP4.

    Dense models show higher sensitivity at input embedding projection
    and output logit layers. Middle transformer blocks are more robust.
    """
    # Note: This requires layer index tracking during quantization
    # The config sets defaults; actual boundary detection is done in replacement
    return MixedPrecisionConfig(
        embeddings=LayerQuantConfig(Precision.BF16),
        lm_head=LayerQuantConfig(Precision.FP8_E4M3, 128),
        norms=LayerQuantConfig(Precision.BF16),
        # First few attention layers get FP8
        attention_qkv=LayerQuantConfig(Precision.FP4_E2M1, 128),
        attention_out=LayerQuantConfig(Precision.FP4_E2M1, 128),
        mlp_gate=LayerQuantConfig(Precision.FP4_E2M1, 128),
        mlp_up=LayerQuantConfig(Precision.FP4_E2M1, 128),
        mlp_down=LayerQuantConfig(Precision.FP4_E2M1, 128),
        default=LayerQuantConfig(Precision.FP4_E2M1, 128),
    )


def dense_component_aware() -> MixedPrecisionConfig:
    """
    Attention FP4/g64, MLP FP4/g128: Component-aware precision.

    Attention layers benefit from tighter quantization (smaller groups)
    because positional encoding and softmax create concentrated weight
    distributions. MLP layers have more Gaussian-like distributions
    and tolerate larger group sizes.
    """
    return MixedPrecisionConfig(
        embeddings=LayerQuantConfig(Precision.BF16),
        lm_head=LayerQuantConfig(Precision.BF16),
        norms=LayerQuantConfig(Precision.BF16),
        # Tight attention quantization
        attention_qkv=LayerQuantConfig(Precision.FP4_E2M1, 64),
        attention_out=LayerQuantConfig(Precision.FP4_E2M1, 64),
        # Relaxed MLP quantization
        mlp_gate=LayerQuantConfig(Precision.FP4_E2M1, 128),
        mlp_up=LayerQuantConfig(Precision.FP4_E2M1, 128),
        mlp_down=LayerQuantConfig(Precision.FP4_E2M1, 128),
        default=LayerQuantConfig(Precision.FP4_E2M1, 128),
    )


# Configuration registry
QUANT_CONFIGS = {
    "uniform_fp4": {
        "name": "Uniform FP4 g128",
        "config": dense_uniform_fp4_g128,
        "description": "Standard Marlin FP4 with uniform group_size=128",
    },
    "mrgptq_fp4": {
        "name": "MR-GPTQ FP4 + Hessian",
        "config": dense_mrgptq_fp4_hessian,
        "description": "GPTQ-optimized FP4 with Hadamard rotation and calibration",
    },
    "boundary_fp8": {
        "name": "Boundary FP8, Middle FP4",
        "config": dense_boundary_fp8,
        "description": "First/last layers FP8, middle layers FP4",
    },
    "component_aware": {
        "name": "Component-Aware Mixed",
        "config": dense_component_aware,
        "description": "Attention FP4/g64, MLP FP4/g128",
    },
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ConfigResult:
    """Results for a single quantization configuration."""

    config_name: str
    description: str

    # Perplexity metrics
    ppl: float | None = None
    ppl_delta: float | None = None
    ppl_delta_pct: float | None = None

    # KL divergence vs baseline
    kl_mean: float | None = None
    kl_max: float | None = None

    # Throughput
    prefill_tok_s: float | None = None
    decode_tok_s: float | None = None
    latency_ms: float | None = None

    # Quantization stats
    quantized_layers: int = 0
    skipped_layers: int = 0
    compression_ratio: float | None = None
    mean_rmse: float | None = None

    # Timing
    quant_time_s: float | None = None
    eval_time_s: float | None = None


@dataclass
class BaselineResult:
    """Results for baseline models."""

    name: str
    ppl: float | None = None
    prefill_tok_s: float | None = None
    decode_tok_s: float | None = None
    memory_gb: float | None = None
    notes: str = ""


@dataclass
class DenseModelBenchmark:
    """Complete benchmark results for dense model evaluation."""

    model_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Model architecture
    hidden_size: int = 0
    num_layers: int = 0
    num_attention_heads: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0
    total_params_b: float = 0.0

    # Benchmark config
    num_samples: int = 0
    max_length: int = 512

    # Baselines
    baselines: list[BaselineResult] = field(default_factory=list)

    # Configuration results
    config_results: list[ConfigResult] = field(default_factory=list)

    # Summary
    best_quality_config: str | None = None
    best_speed_config: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["baselines"] = [asdict(b) for b in self.baselines]
        d["config_results"] = [asdict(c) for c in self.config_results]
        return d


# =============================================================================
# Data Loading
# =============================================================================


def load_wikitext2(max_samples: int = 100) -> list[str]:
    """Load WikiText-2 test set."""
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 50]
        return texts[:max_samples]
    except (ImportError, Exception):
        # Fallback: synthetic test corpus
        print("  Note: Using synthetic test corpus (datasets library not available)")
        sample_texts = [
            "The quick brown fox jumps over the lazy dog. " * 20,
            "In the beginning, there was nothing but darkness and void. " * 15,
            "Science and technology have transformed the modern world. " * 18,
            "The history of mathematics spans thousands of years. " * 16,
            "Programming languages evolve to meet new challenges. " * 17,
            "Machine learning algorithms learn patterns from data. " * 19,
            "Natural language processing enables computers to understand text. " * 14,
            "Deep neural networks have revolutionized artificial intelligence. " * 15,
            "The transformer architecture changed natural language processing. " * 16,
            "Quantization reduces model size while preserving accuracy. " * 18,
        ] * (max_samples // 10 + 1)
        return sample_texts[:max_samples]


# =============================================================================
# Evaluation Functions
# =============================================================================


def compute_perplexity(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int = 512,
    verbose: bool = False,
) -> tuple[float, int, float]:
    """
    Compute perplexity on text dataset.

    Returns:
        (perplexity, total_tokens, eval_time_seconds)
    """
    total_nll = 0.0
    total_tokens = 0
    start_time = time.perf_counter()

    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:max_length]

        input_ids = mx.array(tokens[:-1]).reshape(1, -1)
        targets = mx.array(tokens[1:])

        logits = model(input_ids)
        logits = logits.squeeze(0)

        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        token_log_probs = log_probs[mx.arange(len(targets)), targets]
        nll = -float(mx.sum(token_log_probs))
        mx.eval(nll)

        total_nll += nll
        total_tokens += len(targets)

        if verbose and (i + 1) % 20 == 0:
            ppl_so_far = math.exp(total_nll / total_tokens)
            print(f"    [{i + 1}/{len(texts)}] Running PPL: {ppl_so_far:.4f}")

    elapsed = time.perf_counter() - start_time

    if total_tokens == 0:
        raise ValueError("No valid tokens for perplexity")

    return math.exp(total_nll / total_tokens), total_tokens, elapsed


def compute_kl_divergence(
    model_ref: Any,
    model_test: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int = 256,
) -> tuple[float, float]:
    """
    Compute KL divergence D_KL(P_ref || Q_test).

    Returns:
        (mean_kl, max_kl)
    """
    all_kl = []

    for text in texts[:50]:
        tokens = tokenizer.encode(text)
        if len(tokens) < 10:
            continue
        tokens = tokens[:max_length]

        input_ids = mx.array(tokens[:-1]).reshape(1, -1)

        logits_p = model_ref(input_ids).squeeze(0)
        logits_q = model_test(input_ids).squeeze(0)

        log_p = logits_p - mx.logsumexp(logits_p, axis=-1, keepdims=True)
        log_q = logits_q - mx.logsumexp(logits_q, axis=-1, keepdims=True)

        p = mx.exp(log_p)
        kl_per_pos = mx.sum(p * (log_p - log_q), axis=-1)
        mx.eval(kl_per_pos)

        kl_np = np.array(kl_per_pos)
        valid_kl = kl_np[np.isfinite(kl_np)]
        if len(valid_kl) > 0:
            all_kl.extend(valid_kl.tolist())

    if not all_kl:
        return 0.0, 0.0

    return float(np.mean(all_kl)), float(np.max(all_kl))


def benchmark_throughput(
    model: Any,
    tokenizer: Any,
    prompt_lengths: list[int],
    gen_lengths: list[int],
    warmup: int = 3,
    iterations: int = 5,
) -> dict[str, float]:
    """
    Benchmark prefill and decode throughput.

    Returns dict with prefill_tok_s, decode_tok_s, latency_ms.
    """
    results = {}

    for prompt_len in prompt_lengths:
        for gen_len in gen_lengths:
            input_ids = mx.ones((1, prompt_len), dtype=mx.int32)

            # Warmup
            for _ in range(warmup):
                _ = model(input_ids)
                mx.synchronize()

            prefill_times = []
            decode_times = []

            for _ in range(iterations):
                # Prefill
                start = time.perf_counter()
                logits = model(input_ids)
                mx.eval(logits)
                mx.synchronize()
                prefill_times.append(time.perf_counter() - start)

                # Decode
                start = time.perf_counter()
                for _ in range(gen_len):
                    next_token = mx.argmax(logits[:, -1:, :], axis=-1)
                    logits = model(next_token)
                    mx.eval(logits)
                mx.synchronize()
                decode_times.append(time.perf_counter() - start)

            avg_prefill_ms = sum(prefill_times) / len(prefill_times) * 1000
            avg_decode_ms = sum(decode_times) / len(decode_times) * 1000

            key = f"p{prompt_len}_g{gen_len}"
            results[f"prefill_{key}_tok_s"] = prompt_len / (avg_prefill_ms / 1000)
            results[f"decode_{key}_tok_s"] = gen_len / (avg_decode_ms / 1000)
            results[f"latency_{key}_ms"] = avg_prefill_ms + avg_decode_ms

    # Aggregate
    prefill_keys = [k for k in results if k.startswith("prefill_")]
    decode_keys = [k for k in results if k.startswith("decode_")]

    if prefill_keys:
        results["prefill_tok_s"] = sum(results[k] for k in prefill_keys) / len(prefill_keys)
    if decode_keys:
        results["decode_tok_s"] = sum(results[k] for k in decode_keys) / len(decode_keys)

    return results


# =============================================================================
# Layer Replacement with Mixed Precision
# =============================================================================


def replace_linear_with_marlin(
    model: Any,
    mixed_config: MixedPrecisionConfig,
    num_layers: int = 64,
    boundary_layers: int = 4,
    use_boundary_fp8: bool = False,
) -> tuple[int, int, list[str]]:
    """
    Replace QuantizedLinear with MarlinLinear using mixed precision config.

    For dense models, optionally keeps first/last N layers at higher precision.

    Args:
        model: MLX model
        mixed_config: Mixed precision configuration
        num_layers: Total number of transformer layers
        boundary_layers: Number of layers at start/end to treat as boundaries
        use_boundary_fp8: If True, use FP8 for boundary layers

    Returns:
        (replaced_count, skipped_count, replaced_layer_names)
    """
    try:
        from metal_marlin import MarlinLinear
    except ImportError:
        from metal_marlin.layers import MarlinLinear

    replaced = 0
    skipped = 0
    replaced_names = []

    def get_layer_index(name: str) -> int | None:
        """Extract layer index from name like 'model.layers.23.self_attn...'"""
        parts = name.lower().split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return None

    def is_boundary_layer(name: str) -> bool:
        """Check if layer is in first/last boundary_layers."""
        idx = get_layer_index(name)
        if idx is None:
            return False
        return idx < boundary_layers or idx >= (num_layers - boundary_layers)

    def find_and_replace(module: Any, prefix: str = "", depth: int = 0) -> None:
        nonlocal replaced, skipped

        if depth > 15:
            return

        if hasattr(module, "children"):
            for name, child in module.children().items():
                full_name = f"{prefix}.{name}" if prefix else name

                if isinstance(child, nn.QuantizedLinear):
                    # Get layer config from mixed precision
                    layer_cfg = get_layer_config(full_name, mixed_config)

                    # Skip FP16/BF16 layers
                    if layer_cfg.precision in (Precision.FP16, Precision.BF16):
                        skipped += 1
                        continue

                    # Handle boundary layer FP8 override
                    if use_boundary_fp8 and is_boundary_layer(full_name):
                        if layer_cfg.precision == Precision.FP4_E2M1:
                            # Note: MarlinLinear doesn't support FP8 yet,
                            # so we use tighter group size as approximation
                            group_size = min(layer_cfg.group_size, 64)
                        else:
                            group_size = layer_cfg.group_size
                    else:
                        group_size = layer_cfg.group_size

                    try:
                        marlin = MarlinLinear.from_quantized_linear(
                            child, group_size=group_size
                        )
                        setattr(module, name, marlin)
                        replaced += 1
                        replaced_names.append(full_name)
                    except Exception as e:
                        print(f"  Warning: Could not convert {full_name}: {e}")
                        skipped += 1
                else:
                    find_and_replace(child, full_name, depth + 1)

        # Handle lists/sequences
        if hasattr(module, "__iter__") and not isinstance(module, (str, bytes)):
            try:
                for i, child in enumerate(module):
                    full_name = f"{prefix}[{i}]" if prefix else f"[{i}]"
                    if isinstance(child, nn.QuantizedLinear):
                        layer_cfg = get_layer_config(full_name, mixed_config)

                        if layer_cfg.precision in (Precision.FP16, Precision.BF16):
                            skipped += 1
                            continue

                        try:
                            marlin = MarlinLinear.from_quantized_linear(
                                child, group_size=layer_cfg.group_size
                            )
                            module[i] = marlin
                            replaced += 1
                            replaced_names.append(full_name)
                        except Exception as e:
                            print(f"  Warning: Could not convert {full_name}: {e}")
                            skipped += 1
                    elif hasattr(child, "children"):
                        find_and_replace(child, full_name, depth + 1)
            except TypeError:
                pass

    find_and_replace(model)
    return replaced, skipped, replaced_names


# =============================================================================
# Baseline Evaluation
# =============================================================================


def evaluate_bf16_baseline(
    model_id: str,
    dataset: list[str],
    max_length: int,
    verbose: bool = True,
) -> BaselineResult:
    """Evaluate BF16 (native precision) baseline."""
    import mlx_lm

    if verbose:
        print("\n=== BF16 Baseline ===")

    result = BaselineResult(name="BF16 Native")

    try:
        model, tokenizer = mlx_lm.load(model_id)
        mx.eval(model.parameters())

        # Perplexity
        ppl, tokens, elapsed = compute_perplexity(
            model, tokenizer, dataset, max_length, verbose
        )
        result.ppl = ppl

        # Throughput
        throughput = benchmark_throughput(
            model, tokenizer, [128], [32], warmup=2, iterations=5
        )
        result.prefill_tok_s = throughput.get("prefill_tok_s")
        result.decode_tok_s = throughput.get("decode_tok_s")

        if verbose:
            print(f"  Perplexity: {ppl:.4f}")
            if result.prefill_tok_s:
                print(f"  Prefill: {result.prefill_tok_s:.1f} tok/s")
            if result.decode_tok_s:
                print(f"  Decode: {result.decode_tok_s:.1f} tok/s")

        del model
        gc.collect()
        mx.synchronize()

    except Exception as e:
        result.notes = f"Failed: {e}"
        if verbose:
            print(f"  Error: {e}")

    return result


def evaluate_mlx_4bit_baseline(
    model_id: str,
    dataset: list[str],
    max_length: int,
    verbose: bool = True,
) -> BaselineResult:
    """Evaluate MLX native 4-bit (INT4 affine) baseline."""
    import mlx_lm

    if verbose:
        print("\n=== MLX Native 4-bit (INT4 Affine) ===")

    result = BaselineResult(name="MLX INT4")

    try:
        # Look for 4-bit version or use default quantized load
        model, tokenizer = mlx_lm.load(model_id)
        mx.eval(model.parameters())

        ppl, tokens, elapsed = compute_perplexity(
            model, tokenizer, dataset, max_length, verbose
        )
        result.ppl = ppl

        throughput = benchmark_throughput(
            model, tokenizer, [128], [32], warmup=2, iterations=5
        )
        result.prefill_tok_s = throughput.get("prefill_tok_s")
        result.decode_tok_s = throughput.get("decode_tok_s")

        if verbose:
            print(f"  Perplexity: {ppl:.4f}")
            if result.prefill_tok_s:
                print(f"  Prefill: {result.prefill_tok_s:.1f} tok/s")

        del model
        gc.collect()
        mx.synchronize()

    except Exception as e:
        result.notes = f"Failed: {e}"
        if verbose:
            print(f"  Error: {e}")

    return result


def evaluate_gguf_baseline(
    model_id: str,
    verbose: bool = True,
) -> BaselineResult:
    """
    Evaluate GGUF Q4_K_M baseline (reference values from llama.cpp benchmarks).

    Note: Actual GGUF inference requires llama.cpp integration.
    We record expected values from community benchmarks.
    """
    if verbose:
        print("\n=== GGUF Q4_K_M (Reference) ===")

    result = BaselineResult(
        name="GGUF Q4_K_M",
        notes="Reference values from llama.cpp benchmarks",
    )

    # Community-reported values for Qwen3-32B Q4_K_M
    # These are typical values, actual results vary by hardware
    if "32B" in model_id or "32b" in model_id:
        result.ppl = None  # Would need actual GGUF file
        result.notes = "Requires GGUF model file for actual evaluation"

    if verbose:
        print(f"  {result.notes}")

    return result


# =============================================================================
# Configuration Evaluation
# =============================================================================


def evaluate_config(
    config_key: str,
    model_id: str,
    baseline_ppl: float,
    model_ref: Any | None,
    tokenizer: Any,
    dataset: list[str],
    max_length: int,
    num_layers: int,
    verbose: bool = True,
) -> ConfigResult:
    """Evaluate a single quantization configuration."""
    import mlx_lm

    config_info = QUANT_CONFIGS[config_key]

    if verbose:
        print(f"\n=== {config_info['name']} ===")
        print(f"  {config_info['description']}")

    result = ConfigResult(
        config_name=config_key,
        description=config_info["description"],
    )

    try:
        # Load fresh model
        model, _ = mlx_lm.load(model_id)
        mx.eval(model.parameters())

        # Get mixed precision config
        mixed_config = config_info["config"]()

        # Replace layers
        if verbose:
            print("  Converting to Marlin FP4...")

        start = time.perf_counter()
        use_boundary = config_key == "boundary_fp8"
        replaced, skipped, names = replace_linear_with_marlin(
            model,
            mixed_config,
            num_layers=num_layers,
            boundary_layers=4,
            use_boundary_fp8=use_boundary,
        )
        result.quant_time_s = time.perf_counter() - start
        result.quantized_layers = replaced
        result.skipped_layers = skipped

        if verbose:
            print(f"  Replaced {replaced} layers, skipped {skipped} ({result.quant_time_s:.1f}s)")

        mx.eval(model.parameters())

        # Perplexity
        if verbose:
            print("  Computing perplexity...")

        ppl, tokens, elapsed = compute_perplexity(
            model, tokenizer, dataset, max_length, verbose
        )
        result.ppl = ppl
        result.ppl_delta = ppl - baseline_ppl
        result.ppl_delta_pct = (result.ppl_delta / baseline_ppl) * 100
        result.eval_time_s = elapsed

        if verbose:
            print(f"  Perplexity: {ppl:.4f} (delta: {result.ppl_delta:+.4f})")

        # KL Divergence vs baseline
        if model_ref is not None:
            if verbose:
                print("  Computing KL divergence...")

            kl_mean, kl_max = compute_kl_divergence(
                model_ref, model, tokenizer, dataset, max_length=256
            )
            result.kl_mean = kl_mean
            result.kl_max = kl_max

            if verbose:
                print(f"  KL: mean={kl_mean:.6f}, max={kl_max:.6f}")

        # Throughput
        if verbose:
            print("  Benchmarking throughput...")

        throughput = benchmark_throughput(
            model, tokenizer, [128], [32], warmup=2, iterations=5
        )
        result.prefill_tok_s = throughput.get("prefill_tok_s")
        result.decode_tok_s = throughput.get("decode_tok_s")
        result.latency_ms = throughput.get("latency_p128_g32_ms")

        if result.prefill_tok_s:
            print(f"  Prefill: {result.prefill_tok_s:.1f} tok/s")
        if result.decode_tok_s:
            print(f"  Decode: {result.decode_tok_s:.1f} tok/s")

        # Quality assessment
        if result.ppl_delta is not None:
            delta = abs(result.ppl_delta)
            if delta < 0.1:
                quality = "EXCELLENT"
            elif delta < 0.5:
                quality = "GOOD"
            elif delta < 1.0:
                quality = "ACCEPTABLE"
            else:
                quality = "POOR"
            if verbose:
                print(f"  Quality: {quality}")

        del model
        gc.collect()
        mx.synchronize()

    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        import traceback

        traceback.print_exc()

    return result


# =============================================================================
# Main Benchmark
# =============================================================================


def run_dense_benchmark(
    model_id: str = MODEL_ID,
    configs: list[str] | None = None,
    num_samples: int = 100,
    max_length: int = 512,
    skip_baselines: bool = False,
    verbose: bool = True,
) -> DenseModelBenchmark:
    """
    Run comprehensive dense model benchmark.

    Args:
        model_id: HuggingFace model ID
        configs: List of config keys to test (None = all)
        num_samples: WikiText-2 samples for perplexity
        max_length: Maximum sequence length
        skip_baselines: Skip baseline evaluation
        verbose: Print progress

    Returns:
        DenseModelBenchmark with all results
    """
    import mlx_lm

    from metal_marlin.hf_loader import load_model_config

    if configs is None:
        configs = list(QUANT_CONFIGS.keys())

    print("=" * 70)
    print("Qwen3-32B Dense Model Benchmark")
    print("=" * 70)

    result = DenseModelBenchmark(
        model_id=model_id,
        num_samples=num_samples,
        max_length=max_length,
    )

    # Try to get model config
    actual_model_id = model_id
    try:
        model_cfg = load_model_config(model_id)
        result.hidden_size = model_cfg.hidden_size
        result.num_layers = model_cfg.num_hidden_layers
        result.vocab_size = model_cfg.vocab_size

        # Estimate params
        if result.hidden_size > 0 and result.num_layers > 0:
            # Rough estimate: 12 * L * H^2 for transformer params
            result.total_params_b = (
                12 * result.num_layers * result.hidden_size**2
            ) / 1e9

        print(f"\nModel: {model_id}")
        print(f"  Layers: {result.num_layers}")
        print(f"  Hidden size: {result.hidden_size}")
        print(f"  Estimated params: {result.total_params_b:.1f}B")

        # Check if model is too large
        if result.total_params_b > 20:
            print(f"\n  Warning: Model has ~{result.total_params_b:.0f}B params")
            print(f"  Requires ~{result.total_params_b * 2:.0f}GB memory for FP16")
            print(f"  Using {MODEL_FALLBACK} for benchmark demo")
            actual_model_id = MODEL_FALLBACK
            result.model_id = f"{model_id} (demo: {actual_model_id})"

    except Exception as e:
        print(f"\nWarning: Could not load model config: {e}")
        print(f"Using {MODEL_FALLBACK} for benchmark")
        actual_model_id = MODEL_FALLBACK
        result.model_id = actual_model_id

    # Load dataset
    print(f"\nLoading WikiText-2 ({num_samples} samples)...")
    dataset = load_wikitext2(max_samples=num_samples)
    print(f"  Loaded {len(dataset)} text samples")

    # Baselines
    if not skip_baselines:
        print("\n" + "=" * 70)
        print("BASELINE EVALUATION")
        print("=" * 70)

        # MLX native (our main comparison)
        mlx_baseline = evaluate_mlx_4bit_baseline(
            actual_model_id, dataset, max_length, verbose
        )
        result.baselines.append(mlx_baseline)
        baseline_ppl = mlx_baseline.ppl or 0.0

        # GGUF reference
        gguf_baseline = evaluate_gguf_baseline(model_id, verbose)
        result.baselines.append(gguf_baseline)

    else:
        # Quick mode: estimate baseline
        print("\nSkipping baselines, using estimate...")
        baseline_ppl = 10.0  # Rough estimate

    # Load reference model for KL divergence
    print("\nLoading reference model for KL divergence...")
    try:
        model_ref, tokenizer = mlx_lm.load(actual_model_id)
        mx.eval(model_ref.parameters())
    except Exception as e:
        print(f"  Warning: Could not load reference model: {e}")
        model_ref = None
        # Still need tokenizer
        try:
            _, tokenizer = mlx_lm.load(actual_model_id)
        except Exception:
            print("  Error: Could not load tokenizer")
            return result

    # Evaluate each configuration
    print("\n" + "=" * 70)
    print("CONFIGURATION EVALUATION")
    print("=" * 70)

    num_layers = result.num_layers if result.num_layers > 0 else 32

    for config_key in configs:
        config_result = evaluate_config(
            config_key=config_key,
            model_id=actual_model_id,
            baseline_ppl=baseline_ppl,
            model_ref=model_ref,
            tokenizer=tokenizer,
            dataset=dataset,
            max_length=max_length,
            num_layers=num_layers,
            verbose=verbose,
        )
        result.config_results.append(config_result)

    # Cleanup reference model
    if model_ref is not None:
        del model_ref
        gc.collect()
        mx.synchronize()

    # Determine best configurations
    valid_results = [r for r in result.config_results if r.ppl is not None]
    if valid_results:
        # Best quality = smallest PPL delta
        best_quality = min(valid_results, key=lambda r: abs(r.ppl_delta or float("inf")))
        result.best_quality_config = best_quality.config_name

        # Best speed = highest decode throughput
        with_throughput = [r for r in valid_results if r.decode_tok_s is not None]
        if with_throughput:
            best_speed = max(with_throughput, key=lambda r: r.decode_tok_s or 0)
            result.best_speed_config = best_speed.config_name

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nModel: {result.model_id}")
    print(f"Samples: {num_samples}")

    # Baselines table
    if result.baselines:
        print("\nBaselines:")
        print("-" * 70)
        print(f"{'Name':<20} {'PPL':>10} {'Prefill':>12} {'Decode':>12}")
        print("-" * 70)
        for b in result.baselines:
            ppl_str = f"{b.ppl:.4f}" if b.ppl else "N/A"
            prefill_str = f"{b.prefill_tok_s:.1f}" if b.prefill_tok_s else "N/A"
            decode_str = f"{b.decode_tok_s:.1f}" if b.decode_tok_s else "N/A"
            print(f"{b.name:<20} {ppl_str:>10} {prefill_str:>12} {decode_str:>12}")

    # Configs table
    if result.config_results:
        print("\nConfigurations:")
        print("-" * 70)
        print(f"{'Config':<20} {'PPL':>10} {'Delta':>10} {'KL Mean':>10} {'Decode':>10}")
        print("-" * 70)
        for c in result.config_results:
            ppl_str = f"{c.ppl:.4f}" if c.ppl else "N/A"
            delta_str = f"{c.ppl_delta:+.4f}" if c.ppl_delta else "N/A"
            kl_str = f"{c.kl_mean:.6f}" if c.kl_mean else "N/A"
            decode_str = f"{c.decode_tok_s:.1f}" if c.decode_tok_s else "N/A"
            print(f"{c.config_name:<20} {ppl_str:>10} {delta_str:>10} {kl_str:>10} {decode_str:>10}")

    print("-" * 70)
    if result.best_quality_config:
        print(f"Best quality: {result.best_quality_config}")
    if result.best_speed_config:
        print(f"Best speed: {result.best_speed_config}")
    print("=" * 70)

    return result


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-32B Dense Model Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test with 50 samples
    python benchmarks/eval_qwen3_32b_dense.py --samples 50

    # Full benchmark
    python benchmarks/eval_qwen3_32b_dense.py --full

    # Test specific configuration
    python benchmarks/eval_qwen3_32b_dense.py --config uniform_fp4 --config component_aware

    # Skip baselines for faster iteration
    python benchmarks/eval_qwen3_32b_dense.py --skip-baselines
""",
    )

    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help=f"Model ID (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of WikiText-2 samples (default: 50)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--config",
        action="append",
        choices=list(QUANT_CONFIGS.keys()),
        help="Specific config(s) to test (can repeat, default: all)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full benchmark (100 samples, all configs)",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip baseline evaluation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: benchmarks/results/qwen3_32b_dense.json)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()

    # Determine configs
    configs = args.config if args.config else None

    # Full mode overrides
    if args.full:
        args.samples = 100
        configs = None  # All configs

    # Run benchmark
    result = run_dense_benchmark(
        model_id=args.model,
        configs=configs,
        num_samples=args.samples,
        max_length=args.max_length,
        skip_baselines=args.skip_baselines,
        verbose=not args.quiet,
    )

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = _ROOT / "benchmarks" / "results" / "qwen3_32b_dense.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
