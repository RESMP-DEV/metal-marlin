#!/usr/bin/env python3
"""Memory benchmark comparing BF16 vs quantized Metal Marlin models on MPS.

Measures:
- Model weight memory (device bytes)
- KV cache memory (estimated)
- Peak activation memory (sampled during forward)
- Total VRAM usage (MPS driver allocation)
"""

from __future__ import annotations

import argparse
import gc
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch
from metal_marlin.inference_metal import MetalGLM47Model

try:
    from transformers import AutoModelForCausalLM
import os
import sys

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

except ImportError:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None


@dataclass
class MemoryStats:
    """Collected memory metrics for a model run."""

    label: str
    weight_bytes: int
    kv_cache_bytes: int
    activation_peak_bytes: int
    vram_peak_bytes: int
    vram_after_load_bytes: int
    vram_after_forward_bytes: int
    notes: list[str]


def _format_bytes(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def _mps_memory() -> dict[str, int]:
    """Return MPS memory stats if available."""
    if not HAS_TORCH or torch is None or not torch.backends.mps.is_available():
        return {"current": 0, "driver": 0}

    current = 0
    driver = 0
    if hasattr(torch.mps, "current_allocated_memory"):
        current = int(torch.mps.current_allocated_memory())
    if hasattr(torch.mps, "driver_allocated_memory"):
        driver = int(torch.mps.driver_allocated_memory())
    if driver == 0:
        driver = current
    return {"current": current, "driver": driver}


def _sync_and_collect() -> None:
    if HAS_TORCH and torch is not None and torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
    gc.collect()


def _sum_tensor_bytes(tensors: Iterable[Any], device_type: str | None = None) -> int:
    total = 0
    if not HAS_TORCH or torch is None:
        return total

    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        if device_type is not None and tensor.device.type != device_type:
            continue
        total += tensor.numel() * tensor.element_size()
    return total


def _get_attr(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _find_layer_sequence(model: Any) -> list[Any] | None:
    candidates = [
        "model.layers",
        "model.decoder.layers",
        "transformer.h",
        "transformer.layers",
        "gpt_neox.layers",
        "layers",
    ]
    for path in candidates:
        seq = _get_attr(model, path)
        if seq is None:
            continue
        if isinstance(seq, (list, tuple)):
            return list(seq)
        if HAS_TORCH and torch is not None and isinstance(seq, torch.nn.ModuleList):
            return list(seq)
    return None


def _estimate_kv_cache_bytes(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int,
    bytes_per_elem: int,
) -> int:
    return 2 * num_layers * batch_size * num_kv_heads * seq_len * head_dim * bytes_per_elem


def _get_int_config(config: Any, keys: list[str], default: int) -> int:
    for key in keys:
        if hasattr(config, key):
            value = getattr(config, key)
            if isinstance(value, int):
                return value
    return default


def _measure_activation_peak(
    model: Any,
    *,
    input_ids: Any,
    attention_mask: Any | None = None,
    use_cache: bool | None = None,
) -> tuple[int, int]:
    """Return (activation_peak_bytes, vram_peak_bytes)."""
    baseline = _mps_memory()["current"]
    peak_current = baseline
    peak_driver = _mps_memory()["driver"]

    hooks = []
    layers = _find_layer_sequence(model)

    def _sample(_: Any, __: Any, ___: Any) -> None:
        nonlocal peak_current, peak_driver
        mem = _mps_memory()
        peak_current = max(peak_current, mem["current"])
        peak_driver = max(peak_driver, mem["driver"])

    if layers is not None:
        for layer in layers:
            hooks.append(layer.register_forward_hook(_sample))

    try:
        with torch.no_grad():
            torch.mps.synchronize()
            if use_cache is None:
                _ = model(input_ids, attention_mask=attention_mask)
            else:
                _ = model(input_ids, attention_mask=attention_mask, use_cache=use_cache)
            torch.mps.synchronize()
            _sample(None, None, None)
    finally:
        for hook in hooks:
            hook.remove()

    activation_peak = max(0, peak_current - baseline)
    return activation_peak, peak_driver


def _run_bf16(
    *,
    model_id: str,
    batch_size: int,
    seq_len: int,
    kv_seq_len: int,
) -> MemoryStats:
    if AutoModelForCausalLM is None:
        raise RuntimeError("transformers is required for BF16 benchmark")

    _sync_and_collect()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="mps",
        trust_remote_code=True,
    )
    model.eval()
    torch.mps.synchronize()

    weight_bytes = _sum_tensor_bytes(model.parameters(), device_type="mps") + _sum_tensor_bytes(
        model.buffers(), device_type="mps"
    )

    mem_after_load = _mps_memory()

    vocab_size = _get_int_config(model.config, ["vocab_size"], 32000)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="mps")
    attention_mask = torch.ones_like(input_ids)

    activation_peak, vram_peak = _measure_activation_peak(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    mem_after_forward = _mps_memory()

    num_layers = _get_int_config(model.config, ["num_hidden_layers", "n_layer"], 0)
    num_heads = _get_int_config(model.config, ["num_attention_heads", "n_head"], 0)
    num_kv_heads = _get_int_config(model.config, ["num_key_value_heads"], num_heads)
    hidden_size = _get_int_config(model.config, ["hidden_size", "n_embd"], 0)
    head_dim = hidden_size // num_heads if num_heads > 0 else 0

    kv_cache_bytes = _estimate_kv_cache_bytes(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len=kv_seq_len,
        batch_size=batch_size,
        bytes_per_elem=2,
    )

    notes = []
    if num_layers == 0 or num_heads == 0 or head_dim == 0:
        notes.append("KV cache estimate missing config fields; reported as 0.")

    return MemoryStats(
        label="BF16",
        weight_bytes=weight_bytes,
        kv_cache_bytes=kv_cache_bytes,
        activation_peak_bytes=activation_peak,
        vram_peak_bytes=vram_peak,
        vram_after_load_bytes=mem_after_load["driver"],
        vram_after_forward_bytes=mem_after_forward["driver"],
        notes=notes,
    )


def _run_quantized(
    *,
    model_path: Path,
    bits: int,
    batch_size: int,
    seq_len: int,
    kv_seq_len: int,
) -> MemoryStats:
    _sync_and_collect()

    model = MetalGLM47Model.from_quantized(model_path, bits=bits)
    torch.mps.synchronize()

    weight_bytes = _sum_tensor_bytes(model.parameters(), device_type="mps") + _sum_tensor_bytes(
        model.buffers(), device_type="mps"
    )

    mem_after_load = _mps_memory()

    vocab_size = model.config.get("vocab_size", 32000)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="mps")

    activation_peak, vram_peak = _measure_activation_peak(model, input_ids=input_ids)

    mem_after_forward = _mps_memory()

    num_layers = int(model.config.get("num_layers", 0))
    num_heads = int(model.config.get("num_heads", 0))
    hidden_size = int(model.config.get("hidden_size", 0))
    head_dim = hidden_size // num_heads if num_heads > 0 else 0

    kv_cache_bytes = 0
    notes = []

    kv_lora_rank = model.config.get("kv_lora_rank")
    qk_rope_head_dim = model.config.get("qk_rope_head_dim")
    if kv_lora_rank is not None and qk_rope_head_dim is not None:
        latent_dim = int(kv_lora_rank) + int(qk_rope_head_dim)
        kv_cache_bytes = (
            2 * num_layers * batch_size * num_heads * kv_seq_len * latent_dim * 2
        )
        notes.append("KV cache estimate uses MLA latent dims (kv_lora_rank + qk_rope_head_dim).")
    elif num_layers > 0 and num_heads > 0 and head_dim > 0:
        kv_cache_bytes = _estimate_kv_cache_bytes(
            num_layers=num_layers,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            seq_len=kv_seq_len,
            batch_size=batch_size,
            bytes_per_elem=2,
        )
    else:
        notes.append("KV cache estimate missing config fields; reported as 0.")

    return MemoryStats(
        label=f"Quantized (bits={bits})",
        weight_bytes=weight_bytes,
        kv_cache_bytes=kv_cache_bytes,
        activation_peak_bytes=activation_peak,
        vram_peak_bytes=vram_peak,
        vram_after_load_bytes=mem_after_load["driver"],
        vram_after_forward_bytes=mem_after_forward["driver"],
        notes=notes,
    )


def _print_report(stats: list[MemoryStats]) -> None:
    print("Memory Benchmark (MPS)")
    print("=" * 70)

    for item in stats:
        print(f"{item.label}:")
        print(f"  Weights:          {_format_bytes(item.weight_bytes)}")
        print(f"  KV cache:         {_format_bytes(item.kv_cache_bytes)}")
        print(f"  Peak activation:  {_format_bytes(item.activation_peak_bytes)}")
        print(f"  VRAM (load):      {_format_bytes(item.vram_after_load_bytes)}")
        print(f"  VRAM (forward):   {_format_bytes(item.vram_after_forward_bytes)}")
        print(f"  VRAM (peak):      {_format_bytes(item.vram_peak_bytes)}")
        if item.notes:
            for note in item.notes:
                print(f"  Note: {note}")
        print()

    if len(stats) == 2:
        bf16, quant = stats
        if bf16.weight_bytes > 0 and quant.weight_bytes > 0:
            ratio = bf16.weight_bytes / quant.weight_bytes
            print("Weight Compression")
            print("=" * 70)
            print(f"  Reduction ratio: {ratio:.2f}x")
            print("  Expected:        ~9x (FP4 + scales vs BF16)")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="BF16 vs quantized memory benchmark")
    parser.add_argument("--bf16-model", type=str, help="HF model ID or path for BF16")
    parser.add_argument("--quantized-path", type=Path, help="Path to quantized model directory")
    parser.add_argument("--quant-bits", type=int, default=4, choices=[2, 4, 8])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--kv-seq-len", type=int, default=2048)

    args = parser.parse_args()

    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required for this benchmark")
    if not HAS_MPS:
        raise RuntimeError("PyTorch MPS backend is required for this benchmark")

    if not args.bf16_model and not args.quantized_path:
        raise RuntimeError("Provide --bf16-model and/or --quantized-path")

    stats: list[MemoryStats] = []

    if args.bf16_model:
        stats.append(
            _run_bf16(
                model_id=args.bf16_model,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                kv_seq_len=args.kv_seq_len,
            )
        )

    if args.quantized_path:
        stats.append(
            _run_quantized(
                model_path=args.quantized_path,
                bits=args.quant_bits,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                kv_seq_len=args.kv_seq_len,
            )
        )

    _print_report(stats)


if __name__ == "__main__":
    main()
