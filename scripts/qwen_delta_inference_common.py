"""Shared inference helpers for Qwen3.5 / Qwen3.6 MoE models on Apple Silicon (MPS).

This module provides:
- Architecture-specific configuration (model paths, tokenizer IDs, MoE params)
- Memory profiling utilities
- Active parameter estimation for MoE models
- Report generation and sanity-gate checking
- Common inference pipeline orchestration
- Shared argument parsing and inference entry point

Usage:
    from qwen_delta_inference_common import (
        QwenInferenceConfig,
        get_arch_config,
        peak_memory_mb,
        reset_peak_memory,
        active_params_per_token,
        run_inference_benchmark,
        print_report,
        check_sanity_gate,
        build_inference_argparser,
        run_inference_main,
    )
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Architecture configurations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QwenInferenceConfig:
    """Architecture-specific defaults for Qwen MoE inference.

    Attributes:
        model_name: Human-readable model name for reports (e.g., "Qwen3.5-35B-A3B").
        default_model_path: Relative path under repo root for quantized checkpoint.
        default_tokenizer: HuggingFace tokenizer ID or local path.
        num_experts: Total number of experts in MoE layers.
        num_experts_per_tok: Number of experts activated per token.
        hidden_size: Hidden dimension size.
        intermediate_size: FFN intermediate dimension.
        num_hidden_layers: Number of transformer layers.
        vocab_size: Vocabulary size (for dense param estimation).
        expected_active_params: Approximate active params per token (for fallback).
        min_decode_speed: Minimum acceptable decode speed in tok/s (M4 sanity gate).
    """
    model_name: str
    default_model_path: str
    default_tokenizer: str
    num_experts: int = 256
    num_experts_per_tok: int = 8
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 48
    vocab_size: int = 152064
    expected_active_params: int = 3_500_000_000
    min_decode_speed: float = 15.0


# Predefined configurations for known Qwen models
QWEN35_35B_A3B_CONFIG = QwenInferenceConfig(
    model_name="Qwen3.5-35B-A3B",
    default_model_path="models/Qwen3.5-35B-A3B-MMFP4",
    default_tokenizer="Qwen/Qwen3.5-35B-A3B",
    num_experts=256,
    num_experts_per_tok=8,
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=48,
    vocab_size=152064,
    expected_active_params=3_500_000_000,
    min_decode_speed=15.0,
)

QWEN36_35B_A3B_CONFIG = QwenInferenceConfig(
    model_name="Qwen3.6-35B-A3B",
    default_model_path="models/Qwen3.6-35B-A3B-MMFP4",
    default_tokenizer="Qwen/Qwen3.6-35B-A3B",
    num_experts=256,
    num_experts_per_tok=8,
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=48,
    vocab_size=152064,
    expected_active_params=3_500_000_000,
    min_decode_speed=15.0,
)


def get_arch_config(model_path: Path, tokenizer: str) -> QwenInferenceConfig:
    """Select architecture config based on model path or tokenizer.

    Falls back to Qwen3.5 config if no match is found.

    Args:
        model_path: Path to the quantized checkpoint.
        tokenizer: Tokenizer ID or path.

    Returns:
        Matching QwenInferenceConfig.
    """
    path_str = str(model_path).lower()
    tokenizer_str = tokenizer.lower()

    if "qwen3.6" in path_str or "qwen3.6" in tokenizer_str or "qwen36" in path_str:
        return QWEN36_35B_A3B_CONFIG
    if "qwen3.5" in path_str or "qwen3.5" in tokenizer_str or "qwen35" in path_str:
        return QWEN35_35B_A3B_CONFIG

    # Default to Qwen3.5 config for unknown variants
    return QWEN35_35B_A3B_CONFIG


# ---------------------------------------------------------------------------
# Memory profiling utilities
# ---------------------------------------------------------------------------


def peak_memory_mb(torch_backend: Any) -> float:
    """Return peak allocated memory in MB, or 0.0 if unavailable.

    Args:
        torch_backend: The torch module (needed for backend checks).

    Returns:
        Peak memory in MB.
    """
    if torch_backend.backends.mps.is_available():
        try:
            return torch_backend.mps.current_allocated_memory() / (1024 * 1024)
        except Exception:
            pass
    if torch_backend.cuda.is_available():
        try:
            return torch_backend.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:
            pass
    return 0.0


def reset_peak_memory(torch_backend: Any) -> None:
    """Reset peak memory counters.

    Args:
        torch_backend: The torch module (needed for backend checks).
    """
    if torch_backend.cuda.is_available():
        try:
            torch_backend.cuda.reset_peak_memory_stats()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Active parameter estimation
# ---------------------------------------------------------------------------


def active_params_per_token(model: Any, config: QwenInferenceConfig | None = None) -> int | None:
    """Estimate active parameter count per decode token for MoE models.

    For Qwen MoE models, computes the active parameters based on:
    - MoE path: num_layers * num_experts_per_tok * (2 * hidden * intermediate)
    - Dense path: 2 * vocab * hidden + num_layers * 4 * hidden^2

    Falls back to config.expected_active_params if config cannot be read.

    Args:
        model: The loaded model.
        config: Optional architecture config for fallback values.

    Returns:
        Estimated active parameters per token, or None if estimation fails.
    """
    model_config = getattr(model, "config", None)
    if model_config is None:
        if config is not None:
            return config.expected_active_params
        return None

    # Try to read from model config first
    num_experts = getattr(model_config, "num_experts", None)
    num_experts_per_tok = getattr(model_config, "num_experts_per_tok", None)
    intermediate_size = getattr(model_config, "intermediate_size", None)
    hidden_size = getattr(model_config, "hidden_size", None)
    num_hidden_layers = getattr(model_config, "num_hidden_layers", None)

    if all(v is not None for v in (
        num_experts, num_experts_per_tok, intermediate_size,
        hidden_size, num_hidden_layers,
    )):
        per_expert = 2 * hidden_size * intermediate_size
        moe_active = num_hidden_layers * num_experts_per_tok * per_expert
        vocab_size = getattr(model_config, "vocab_size", 152064)
        dense_params = (
            2 * vocab_size * hidden_size
            + num_hidden_layers * 4 * hidden_size * hidden_size
        )
        return int(moe_active + dense_params)

    # Fallback: use architecture config if provided
    if config is not None:
        return config.expected_active_params

    # Last resort: heuristic based on model name
    model_name = getattr(model_config, "_name_or_path", "")
    if "35B" in model_name or "35b" in model_name.lower():
        return 3_500_000_000

    return None


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


@dataclass
class InferenceMetrics:
    """Metrics collected during inference benchmark."""
    prompt_tokens: int
    total_new_tokens: int
    prefill_time_ms: float
    decode_time_s: float
    peak_memory_mb: float
    active_params: int | None
    generated_text: str

    @property
    def prefill_speed(self) -> float:
        """Prefill speed in tokens per second."""
        return 1.0 / (self.prefill_time_ms / 1000.0 + 1e-9)

    @property
    def decode_speed(self) -> float:
        """Decode speed in tokens per second."""
        if self.decode_time_s <= 0:
            return 0.0
        return self.total_new_tokens / self.decode_time_s


def print_report(metrics: InferenceMetrics, model_name: str) -> None:
    """Print formatted inference report.

    Args:
        metrics: Collected inference metrics.
        model_name: Human-readable model name for the report header.
    """
    print()
    print("=" * 60)
    print(f"{model_name} MMFP4 Inference Report")
    print("=" * 60)
    print("Generated text:")
    print(f"{metrics.generated_text}")
    print()
    print(f"Prefill time (first token):     {metrics.prefill_time_ms:>8.2f} ms")
    print(f"Prefill speed (first token):    {metrics.prefill_speed:>8.2f} tok/s")
    print(f"Decode tokens generated:        {metrics.total_new_tokens:>8d}")
    print(f"Decode time:                    {metrics.decode_time_s:>8.3f} s")
    print(f"Decode speed:                   {metrics.decode_speed:>8.2f} tok/s")
    print(f"Peak memory usage:              {metrics.peak_memory_mb:>8.2f} MB")
    if metrics.active_params is not None:
        print(
            f"Active parameters per token:    {metrics.active_params:>8,} "
            f"(~{metrics.active_params / 1e9:.1f}B)"
        )
    else:
        print("Active parameters per token:    N/A")
    print("=" * 60)


def check_sanity_gate(metrics: InferenceMetrics, min_decode_speed: float) -> bool:
    """Check if decode speed meets the minimum threshold.

    Args:
        metrics: Collected inference metrics.
        min_decode_speed: Minimum acceptable decode speed in tok/s.

    Returns:
        True if the sanity gate passed, False otherwise.
    """
    if metrics.decode_speed < min_decode_speed:
        print(
            f"\nERROR: Decode speed {metrics.decode_speed:.2f} tok/s is below the "
            f"minimum threshold of {min_decode_speed:.2f} tok/s.",
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------


def run_inference_benchmark(
    pipeline: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: str,
    torch_backend: Any,
) -> InferenceMetrics:
    """Run the inference benchmark and collect metrics.

    This function handles:
    - Tokenization of the prompt
    - Warm-up pass for MPS stability
    - Prefill (first token) timing
    - Decode (remaining tokens) timing using KV cache
    - Memory profiling

    Args:
        pipeline: MMFP4Pipeline with model and tokenizer loaded.
        prompt: Input prompt text.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (<=0 for greedy).
        top_p: Nucleus sampling threshold.
        top_k: Top-k filtering (0 = disabled).
        device: Torch device string.
        torch_backend: The torch module.

    Returns:
        InferenceMetrics with all collected measurements.
    """
    # Tokenize prompt
    inputs = pipeline.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(pipeline.device)
    prompt_tokens = input_ids.shape[1]

    # Warm-up pass
    pad_token_id = getattr(pipeline.tokenizer, "pad_token_id", None)
    with torch_backend.inference_mode():
        _ = pipeline.model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    if torch_backend.backends.mps.is_available():
        torch_backend.mps.synchronize()

    # Reset memory stats before benchmark
    reset_peak_memory(torch_backend)

    # Prefill: generate the first new token
    prefill_start = time.perf_counter()
    with torch_backend.inference_mode():
        outputs = pipeline.model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=pad_token_id,
            return_dict_in_generate=True,
            use_cache=True,
        )
    if torch_backend.backends.mps.is_available():
        torch_backend.mps.synchronize()
    prefill_end = time.perf_counter()

    prefill_time_ms = (prefill_end - prefill_start) * 1000.0

    # Decode: generate remaining tokens using KV cache
    past_key_values = outputs.past_key_values
    generated_ids = outputs.sequences
    decode_tokens = max_new_tokens - 1

    decode_start = time.perf_counter()
    if decode_tokens > 0:
        do_sample = temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": decode_tokens,
            "past_key_values": past_key_values,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
            "use_cache": True,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(float(temperature), 1e-5)
            gen_kwargs["top_p"] = float(top_p)
            if top_k > 0:
                gen_kwargs["top_k"] = int(top_k)

        with torch_backend.inference_mode():
            decode_outputs = pipeline.model.generate(
                input_ids=generated_ids[:, -1:],
                **gen_kwargs,
            )
        if torch_backend.backends.mps.is_available():
            torch_backend.mps.synchronize()
        if decode_outputs.shape[1] > 1:
            generated_ids = torch_backend.cat(
                [generated_ids, decode_outputs[:, 1:]], dim=1,
            )
    decode_end = time.perf_counter()

    decode_time_s = decode_end - decode_start
    total_new_tokens = generated_ids.shape[1] - prompt_tokens

    # Decode generated text
    generated_text = pipeline.tokenizer.decode(
        generated_ids[0, prompt_tokens:], skip_special_tokens=True,
    )

    # Collect memory
    peak_mem = peak_memory_mb(torch_backend)

    # Estimate active parameters
    active_params = active_params_per_token(pipeline.model)

    return InferenceMetrics(
        prompt_tokens=prompt_tokens,
        total_new_tokens=total_new_tokens,
        prefill_time_ms=prefill_time_ms,
        decode_time_s=decode_time_s,
        peak_memory_mb=peak_mem,
        active_params=active_params,
        generated_text=generated_text,
    )


# ---------------------------------------------------------------------------
# Shared argument parsing
# ---------------------------------------------------------------------------


def build_inference_argparser(
    config: QwenInferenceConfig,
    env_prefix: str,
) -> argparse.ArgumentParser:
    """Build a standard inference argument parser for a Qwen variant.

    All architecture-specific defaults (model path, tokenizer, decode speed
    threshold) come from *config*, so callers never hard-code them.

    Args:
        config: Architecture config providing default values.
        env_prefix: Upper-case prefix for the model-path env var
            (e.g. ``"QWEN35"`` → reads ``QWEN35_MODEL_PATH``).

    Returns:
        An ``ArgumentParser`` with all standard inference flags populated.
    """
    # Resolve repo root relative to the scripts/ directory
    repo_root = Path(__file__).resolve().parent.parent
    default_model = repo_root / config.default_model_path

    parser = argparse.ArgumentParser(
        description=(
            f"Run end-to-end {config.model_name} MMFP4 inference on Apple Silicon."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.getenv(f"{env_prefix}_MODEL_PATH", str(default_model))),
        help="Path to the quantized checkpoint directory.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=config.default_tokenizer,
        help="HuggingFace tokenizer identifier or local path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Input prompt for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Torch device (mps, cuda, cpu).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (<=0 for greedy).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k filtering (0 = disabled).",
    )
    parser.add_argument(
        "--min-decode-speed",
        type=float,
        default=config.min_decode_speed,
        help="Minimum acceptable decode speed in tok/s (hard gate).",
    )
    return parser


# ---------------------------------------------------------------------------
# Shared inference entry point
# ---------------------------------------------------------------------------


def run_inference_main(
    argv: list[str] | None,
    config: QwenInferenceConfig,
    env_prefix: str,
) -> int:
    """Common main() for Qwen inference wrapper scripts.

    This function implements the full load → benchmark → report → gate flow
    so that ``run_qwen35_inference.py`` / ``run_qwen36_inference.py`` only need
    to call ``run_inference_main(None, CONFIG, "QWEN35")``.

    Late imports for ``torch`` and ``MMFP4Pipeline`` are performed inside this
    function so the wrapper script's top-level import stays lightweight.

    Args:
        argv: CLI arguments (``None`` → ``sys.argv[1:]``).
        config: Architecture config for the target model variant.
        env_prefix: Upper-case env-var prefix (e.g. ``"QWEN36"``).

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    # -- path setup for metal_marlin package ------------------------------
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from metal_marlin._compat import require_torch, torch  # noqa: E402

    require_torch()
    assert torch is not None

    from metal_marlin.inference.mmfp4_pipeline import MMFP4Pipeline  # noqa: E402

    # -- parse args -------------------------------------------------------
    parser = build_inference_argparser(config, env_prefix)
    args = parser.parse_args(argv)

    if not args.model_path.exists():
        print(
            f"ERROR: Quantized checkpoint not found: {args.model_path}",
            file=sys.stderr,
        )
        return 1

    # -- load pipeline ----------------------------------------------------
    print(f"Loading quantized checkpoint from {args.model_path} ...")
    print(f"Tokenizer: {args.tokenizer}")

    tokenizer_path: Path | str = args.model_path
    tokenizer_files = {"tokenizer.json", "tokenizer_config.json"}
    if not any((Path(tokenizer_path) / f).exists() for f in tokenizer_files):
        tokenizer_path = args.tokenizer

    pipeline = MMFP4Pipeline.from_pretrained(
        str(args.model_path),
        device=args.device,
        enable_persistent_cache=False,
        use_paged_attention=False,
    )

    # If tokenizer was not found in the checkpoint dir, reload from HF
    if tokenizer_path != args.model_path:
        from transformers import AutoTokenizer

        pipeline.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True,
        )
        if (
            getattr(pipeline.tokenizer, "pad_token_id", None) is None
            and getattr(pipeline.tokenizer, "eos_token_id", None) is not None
        ):
            pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token

    print(f"Model loaded on device: {pipeline.device}")

    # -- get architecture config for reports ------------------------------
    arch_config = get_arch_config(args.model_path, args.tokenizer)

    # -- run inference benchmark ------------------------------------------
    metrics: InferenceMetrics = run_inference_benchmark(
        pipeline=pipeline,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        device=args.device,
        torch_backend=torch,
    )

    # -- print report -----------------------------------------------------
    print_report(metrics, arch_config.model_name)

    # -- sanity gate ------------------------------------------------------
    min_speed = args.min_decode_speed
    if not check_sanity_gate(metrics, min_speed):
        print(
            f"ERROR: Decode speed {metrics.decode_speed:.2f} tok/s is below the "
            f"minimum threshold of {min_speed:.2f} tok/s on M4.",
            file=sys.stderr,
        )
        return 1

    print("\nPASS: All sanity gates cleared.")
    return 0
