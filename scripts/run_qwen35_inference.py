#!/usr/bin/env python3
"""End-to-end inference script for Qwen3.5-35B-A3B with MMFP4 quantized weights on M4 (MPS).

Loads a pre-quantized checkpoint, runs a simple generation benchmark,
and prints latency / throughput / memory metrics with a hard sanity gate
on decode speed.

Usage:
    cd contrib/metal_marlin
    uv run python scripts/run_qwen35_inference.py

Environment:
    QWEN35_MODEL_PATH   Override default checkpoint path.
    HF_TOKEN            HuggingFace token (only needed if tokenizer must be downloaded).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup – allow importing from the metal_marlin package one level up.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from metal_marlin._compat import require_torch, torch  # noqa: E402

require_torch()
assert torch is not None

from metal_marlin.inference.mmfp4_pipeline import MMFP4Pipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "Qwen3.5-35B-A3B-MMFP4"
DEFAULT_TOKENIZER = "Qwen/Qwen3.5-35B-A3B"
DEFAULT_PROMPT = "Once upon a time"
DEFAULT_MAX_NEW_TOKENS = 128
MIN_DECODE_SPEED_TOK_S = 15.0  # sanity gate for M4


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end Qwen3.5-35B-A3B MMFP4 inference on Apple Silicon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.getenv("QWEN35_MODEL_PATH", str(DEFAULT_MODEL_PATH))),
        help="Path to the quantized checkpoint directory.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="HuggingFace tokenizer identifier or local path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Input prompt for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
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
        default=MIN_DECODE_SPEED_TOK_S,
        help="Minimum acceptable decode speed in tok/s (hard gate).",
    )
    return parser.parse_args(argv)


def _peak_memory_mb() -> float:
    """Return peak allocated memory in MB, or 0.0 if unavailable."""
    if torch.backends.mps.is_available():
        try:
            return torch.mps.current_allocated_memory() / (1024 * 1024)
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:
            pass
    return 0.0


def _reset_peak_memory() -> None:
    """Reset peak memory counters."""
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def _active_params_per_token(model: Any) -> int | None:
    """Estimate active parameter count per decode token for MoE models.

    For Qwen3.5-35B-A3B (256 experts, top-8), the expected active count is
    approximately 3B parameters per token.
    """
    config = getattr(model, "config", None)
    if config is None:
        return None

    num_experts = getattr(config, "num_experts", None)
    num_experts_per_tok = getattr(config, "num_experts_per_tok", None)
    intermediate_size = getattr(config, "intermediate_size", None)
    hidden_size = getattr(config, "hidden_size", None)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)

    if all(v is not None for v in (
        num_experts, num_experts_per_tok, intermediate_size,
        hidden_size, num_hidden_layers,
    )):
        per_expert = 2 * hidden_size * intermediate_size
        moe_active = num_hidden_layers * num_experts_per_tok * per_expert
        vocab_size = getattr(config, "vocab_size", 152064)
        dense_params = (
            2 * vocab_size * hidden_size
            + num_hidden_layers * 4 * hidden_size * hidden_size
        )
        return int(moe_active + dense_params)

    model_name = getattr(config, "_name_or_path", "")
    if "35B" in model_name or "35b" in model_name.lower():
        return 3_500_000_000
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.model_path.exists():
        print(
            f"ERROR: Quantized checkpoint not found: {args.model_path}",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # Load pipeline
    # ------------------------------------------------------------------
    print(f"Loading quantized checkpoint from {args.model_path} ...")
    print(f"Tokenizer: {args.tokenizer}")

    tokenizer_path = args.model_path
    tokenizer_files = {"tokenizer.json", "tokenizer_config.json"}
    if not any((tokenizer_path / f).exists() for f in tokenizer_files):
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

    # ------------------------------------------------------------------
    # Tokenize prompt and warm-up
    # ------------------------------------------------------------------
    inputs = pipeline.tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(pipeline.device)
    prompt_tokens = input_ids.shape[1]
    print(f"Prompt tokens: {prompt_tokens}")

    # Warm-up pass to stabilise MPS allocations
    pad_token_id = getattr(pipeline.tokenizer, "pad_token_id", None)
    with torch.inference_mode():
        _ = pipeline.model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    # ------------------------------------------------------------------
    # Benchmark: prefill + decode
    # ------------------------------------------------------------------
    _reset_peak_memory()

    # Prefill: generate the first new token
    prefill_start = time.perf_counter()
    with torch.inference_mode():
        outputs = pipeline.model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=pad_token_id,
            return_dict_in_generate=True,
            use_cache=True,
        )
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    prefill_end = time.perf_counter()

    prefill_time_ms = (prefill_end - prefill_start) * 1000.0

    # Decode: generate remaining tokens using the KV cache from prefill
    past_key_values = outputs.past_key_values
    generated_ids = outputs.sequences
    decode_tokens = args.max_new_tokens - 1

    decode_start = time.perf_counter()
    if decode_tokens > 0:
        do_sample = args.temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": decode_tokens,
            "past_key_values": past_key_values,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
            "use_cache": True,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(float(args.temperature), 1e-5)
            gen_kwargs["top_p"] = float(args.top_p)
            if args.top_k > 0:
                gen_kwargs["top_k"] = int(args.top_k)

        with torch.inference_mode():
            decode_outputs = pipeline.model.generate(
                input_ids=generated_ids[:, -1:],
                **gen_kwargs,
            )
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        if decode_outputs.shape[1] > 1:
            generated_ids = torch.cat(
                [generated_ids, decode_outputs[:, 1:]], dim=1,
            )
    decode_end = time.perf_counter()

    decode_time_s = decode_end - decode_start
    total_new_tokens = generated_ids.shape[1] - prompt_tokens
    decode_speed = total_new_tokens / (decode_time_s + 1e-9) if decode_time_s > 0 else 0.0
    prefill_speed = 1.0 / (prefill_time_ms / 1000.0 + 1e-9)

    # ------------------------------------------------------------------
    # Decode output
    # ------------------------------------------------------------------
    generated_text = pipeline.tokenizer.decode(
        generated_ids[0, prompt_tokens:], skip_special_tokens=True,
    )

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------
    peak_mem_mb = _peak_memory_mb()

    # ------------------------------------------------------------------
    # Active parameters
    # ------------------------------------------------------------------
    active_params = _active_params_per_token(pipeline.model)

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Qwen3.5-35B-A3B MMFP4 Inference Report")
    print("=" * 60)
    print(f"Generated text:")
    print(f"{generated_text}")
    print()
    print(f"Prefill time (first token):     {prefill_time_ms:>8.2f} ms")
    print(f"Prefill speed (first token):    {prefill_speed:>8.2f} tok/s")
    print(f"Decode tokens generated:        {total_new_tokens:>8d}")
    print(f"Decode time:                    {decode_time_s:>8.3f} s")
    print(f"Decode speed:                   {decode_speed:>8.2f} tok/s")
    print(f"Peak memory usage:              {peak_mem_mb:>8.2f} MB")
    if active_params is not None:
        print(
            f"Active parameters per token:    {active_params:>8,} "
            f"(~{active_params / 1e9:.1f}B)"
        )
    else:
        print("Active parameters per token:    N/A")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Sanity gate
    # ------------------------------------------------------------------
    if decode_speed < args.min_decode_speed:
        print(
            f"\nERROR: Decode speed {decode_speed:.2f} tok/s is below the "
            f"minimum threshold of {args.min_decode_speed:.2f} tok/s on M4.",
            file=sys.stderr,
        )
        return 1

    print("\nPASS: All sanity gates cleared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
