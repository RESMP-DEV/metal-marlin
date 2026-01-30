#!/usr/bin/env python3
'''
End-to-end GLM-4.7-Flash inference with FP4 quantization.

Usage:
    python glm4_flash_inference.py --prompt "Hello, how are you?"
    python glm4_flash_inference.py --interactive
'''

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch
from metal_marlin.inference.pipeline_v2 import TransformersMarlinPipeline
from metal_marlin.transformers_loader import load_and_quantize


def _mps_memory_bytes() -> dict[str, int]:
    if not HAS_TORCH or torch is None or not HAS_MPS:
        return {}
    stats: dict[str, int] = {}
    if hasattr(torch.mps, "current_allocated_memory"):
        stats["current"] = int(torch.mps.current_allocated_memory())
    if hasattr(torch.mps, "driver_allocated_memory"):
        stats["driver"] = int(torch.mps.driver_allocated_memory())
    return stats


def _format_bytes(value: int) -> str:
    sign = "-" if value < 0 else ""
    abs_value = abs(value)
    if abs_value >= 1024**3:
        return f"{sign}{abs_value / 1024**3:.2f} GB"
    if abs_value >= 1024**2:
        return f"{sign}{abs_value / 1024**2:.1f} MB"
    if abs_value >= 1024:
        return f"{sign}{abs_value / 1024:.1f} KB"
    return f"{sign}{abs_value} B"


def _format_memory_line(prefix: str, stats: dict[str, int] | None) -> str:
    if not stats:
        return f"{prefix}: MPS memory unavailable"
    parts = []
    if "current" in stats:
        parts.append(f"current={_format_bytes(stats['current'])}")
    if "driver" in stats:
        parts.append(f"driver={_format_bytes(stats['driver'])}")
    return f"{prefix}: " + ", ".join(parts)


def _format_memory_delta(before: dict[str, int], after: dict[str, int]) -> str:
    if not before or not after:
        return "MPS memory delta unavailable"
    parts = []
    if "current" in before and "current" in after:
        delta = after["current"] - before["current"]
        parts.append(f"current_delta={_format_bytes(delta)}")
    if "driver" in before and "driver" in after:
        delta = after["driver"] - before["driver"]
        parts.append(f"driver_delta={_format_bytes(delta)}")
    return ", ".join(parts) if parts else "MPS memory delta unavailable"


def _sync_mps() -> None:
    if HAS_TORCH and torch is not None and HAS_MPS and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _stream_chat(
    pipeline: TransformersMarlinPipeline,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[str, int, float, dict[str, int], dict[str, int]]:
    mem_before = _mps_memory_bytes()
    _sync_mps()
    start = time.perf_counter()
    streamer = pipeline.chat(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stream=True,
    )
    response = ""
    for token in streamer:
        print(token, end="", flush=True)
        response += token
    _sync_mps()
    elapsed = time.perf_counter() - start
    mem_after = _mps_memory_bytes()
    token_count = len(tokenizer.encode(response, add_special_tokens=False))
    return response, token_count, elapsed, mem_before, mem_after


def _print_generation_metrics(token_count: int, elapsed: float, mem_before: dict[str, int], mem_after: dict[str, int]) -> None:
    tok_per_s = token_count / elapsed if elapsed > 0 else 0.0
    print()
    print(
        "[metrics] "
        f"new_tokens={token_count} "
        f"elapsed={elapsed:.2f}s "
        f"tok/s={tok_per_s:.2f} "
        f"{_format_memory_delta(mem_before, mem_after)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GLM-4.7-Flash with Marlin FP4 quantization")
    parser.add_argument("--model", default="zai-org/GLM-4.7-Flash")
    parser.add_argument("--prompt", help="Single prompt to run")
    parser.add_argument("--interactive", action="store_true", help="Run an interactive chat session")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--format", default="fp4")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)

    args = parser.parse_args()

    if not args.prompt and not args.interactive:
        parser.error("Provide --prompt or --interactive")

    if args.bits != 4:
        print("Only 4-bit FP4 quantization is supported in this example.", file=sys.stderr)
        return 1

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        print("transformers is required for this script. Install with: pip install transformers", file=sys.stderr)
        raise SystemExit(1) from exc

    print("Loading and quantizing model...")
    model, stats = load_and_quantize(
        args.model,
        bits=args.bits,
        group_size=args.group_size,
        format=args.format,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Quantization summary:")
    if stats.get("quantized_bytes") is not None:
        print(f"  quantized_bytes: {_format_bytes(int(stats['quantized_bytes']))}")
    if stats.get("original_bytes") is not None:
        print(f"  original_bytes: {_format_bytes(int(stats['original_bytes']))}")
    if stats.get("compression_ratio") is not None:
        print(f"  compression_ratio: {stats['compression_ratio']:.2f}x")
    if stats.get("quantized_count") is not None:
        print(f"  quantized_layers: {stats['quantized_count']}")
    if stats.get("skipped_count") is not None:
        print(f"  skipped_layers: {stats['skipped_count']}")

    print(_format_memory_line("MPS memory after load", _mps_memory_bytes()))

    pipeline = TransformersMarlinPipeline(model, tokenizer)

    if args.prompt:
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ]
        print("\nAssistant: ", end="", flush=True)
        response, token_count, elapsed, mem_before, mem_after = _stream_chat(
            pipeline,
            tokenizer,
            messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        _print_generation_metrics(token_count, elapsed, mem_before, mem_after)
        return 0

    history: list[dict[str, str]] = [{"role": "system", "content": args.system}]
    print("Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        history.append({"role": "user", "content": user_input})
        print("Assistant: ", end="", flush=True)
        response, token_count, elapsed, mem_before, mem_after = _stream_chat(
            pipeline,
            tokenizer,
            history,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        history.append({"role": "assistant", "content": response})
        _print_generation_metrics(token_count, elapsed, mem_before, mem_after)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
