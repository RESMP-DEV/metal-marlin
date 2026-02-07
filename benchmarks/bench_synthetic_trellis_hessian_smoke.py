#!/usr/bin/env python3
"""Tiny synthetic benchmark for Trellis load/prefill/decode/Hessian smoke paths.

Usage:
    uv run python contrib/metal_marlin/benchmarks/bench_synthetic_trellis_hessian_smoke.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file as save_safetensors

BENCHMARK_NAME = "synthetic_trellis_hessian_smoke"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks" / "results" / f"{BENCHMARK_NAME}.json"

# Ensure local contrib/metal_marlin package resolves when launched from repo root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True, slots=True)
class SyntheticTrellisFixtureMetadata:
    """Metadata for generated synthetic Trellis fixtures."""

    model_path: Path
    vocab_size: int
    hidden_size: int
    num_layers: int


def _has_mps() -> bool:
    return bool(
        hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def _mps_sync() -> None:
    if _has_mps() and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_results(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def _dot_to_mm_key(name: str) -> str:
    return name.replace(".", "__")


def _packed_tile_bytes(bits: int) -> int:
    return (256 * bits + 7) // 8


def _layer_weight_specs(
    *,
    layer_idx: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_kv_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
) -> list[tuple[str, int, int]]:
    prefix = f"model.layers.{layer_idx}"
    qk_total_head_dim = qk_nope_head_dim + qk_rope_head_dim

    return [
        (f"{prefix}.self_attn.q_a_proj.weight", q_lora_rank, hidden_size),
        (
            f"{prefix}.self_attn.q_b_proj.weight",
            num_attention_heads * qk_total_head_dim,
            q_lora_rank,
        ),
        (f"{prefix}.self_attn.kv_a_proj_with_mqa.weight", kv_lora_rank + qk_rope_head_dim, hidden_size),
        (
            f"{prefix}.self_attn.kv_b_proj.weight",
            num_kv_heads * (qk_nope_head_dim + v_head_dim),
            kv_lora_rank,
        ),
        (f"{prefix}.self_attn.o_proj.weight", hidden_size, num_attention_heads * v_head_dim),
        (f"{prefix}.mlp.gate_proj.weight", intermediate_size, hidden_size),
        (f"{prefix}.mlp.up_proj.weight", intermediate_size, hidden_size),
        (f"{prefix}.mlp.down_proj.weight", hidden_size, intermediate_size),
    ]


def build_synthetic_trellis_fixture(
    root_dir: str | Path,
    *,
    model_name: str = "synthetic-trellis-smoke",
    seed: int = 2026,
    vocab_size: int = 128,
    hidden_size: int = 64,
    num_hidden_layers: int = 2,
    intermediate_size: int = 128,
    bits: int = 4,
) -> SyntheticTrellisFixtureMetadata:
    """Build a tiny deterministic Trellis checkpoint on local disk."""
    root_dir = Path(root_dir)
    model_path = root_dir / model_name
    model_path.mkdir(parents=True, exist_ok=True)

    # Keep dimensions small so pre-merge runs stay fast.
    num_attention_heads = 4
    num_kv_heads = 2
    q_lora_rank = 16
    kv_lora_rank = 16
    qk_nope_head_dim = 8
    qk_rope_head_dim = 8
    v_head_dim = 16

    rng = torch.Generator(device="cpu").manual_seed(seed)

    config = {
        "architectures": ["TrellisForCausalLM"],
        "model_type": "trellis",
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": qk_nope_head_dim + qk_rope_head_dim,
        "intermediate_size": intermediate_size,
        "num_experts": 1,
        "num_experts_per_tok": 1,
        "first_moe_layer": num_hidden_layers,
        "kv_lora_rank": kv_lora_rank,
        "q_lora_rank": q_lora_rank,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "quantization_bits": bits,
    }
    (model_path / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    base_weights: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(
            (vocab_size, hidden_size),
            dtype=torch.float32,
            generator=rng,
        )
        * 0.02,
        "model.norm.weight": torch.ones(hidden_size, dtype=torch.float32),
        "lm_head.weight": torch.randn(
            (vocab_size, hidden_size),
            dtype=torch.float32,
            generator=rng,
        )
        * 0.02,
    }
    for layer_idx in range(num_hidden_layers):
        prefix = f"model.layers.{layer_idx}"
        base_weights[f"{prefix}.input_layernorm.weight"] = torch.ones(hidden_size, dtype=torch.float32)
        base_weights[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(
            hidden_size,
            dtype=torch.float32,
        )
        base_weights[f"{prefix}.self_attn.q_a_layernorm.weight"] = torch.ones(
            q_lora_rank,
            dtype=torch.float32,
        )
        base_weights[f"{prefix}.self_attn.kv_a_layernorm.weight"] = torch.ones(
            kv_lora_rank,
            dtype=torch.float32,
        )
    save_safetensors(base_weights, str(model_path / "base_weights.safetensors"))

    shard_name = "model-00001-of-00001.safetensors"
    shard_tensors: dict[str, torch.Tensor] = {}
    quantization_layers: list[dict[str, object]] = []
    per_layer_tensors: dict[int, dict[str, torch.Tensor]] = {}
    per_layer_entries: dict[int, list[dict[str, object]]] = {}

    for layer_idx in range(num_hidden_layers):
        layer_tensors: dict[str, torch.Tensor] = {}
        layer_entries: list[dict[str, object]] = []

        for name, out_features, in_features in _layer_weight_specs(
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
        ):
            tiles_out = (out_features + 15) // 16
            tiles_in = (in_features + 15) // 16
            packed_bytes = _packed_tile_bytes(bits)
            packed_len = tiles_out * tiles_in * packed_bytes

            indices = torch.randint(
                0,
                256,
                (1 + packed_len,),
                dtype=torch.uint8,
                generator=rng,
            )
            indices[0] = bits

            n_groups = max(1, math.ceil(in_features / 128))
            scales = (
                torch.rand((n_groups, out_features), dtype=torch.float32, generator=rng) * 0.05
                + 0.005
            )
            su = torch.randint(0, 2, (in_features,), generator=rng, dtype=torch.int8).to(torch.float32)
            sv = torch.randint(0, 2, (out_features,), generator=rng, dtype=torch.int8).to(torch.float32)
            su = su.mul_(2.0).sub_(1.0)
            sv = sv.mul_(2.0).sub_(1.0)

            base_key = _dot_to_mm_key(name)
            components = {
                f"{base_key}__indices": indices,
                f"{base_key}__scales": scales,
                f"{base_key}__su": su,
                f"{base_key}__sv": sv,
            }
            shard_tensors.update(components)
            layer_tensors.update(components)

            tensor_meta = {
                "name": name,
                "bits": bits,
                "shape": [out_features, in_features],
                "mse": 0.0,
            }
            quantization_layers.append(tensor_meta)
            layer_entries.append(tensor_meta)

        per_layer_tensors[layer_idx] = layer_tensors
        per_layer_entries[layer_idx] = layer_entries

    save_safetensors(shard_tensors, str(model_path / shard_name))

    total_size = sum(tensor.numel() * tensor.element_size() for tensor in shard_tensors.values())
    weight_map = {name: shard_name for name in shard_tensors}
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"format": "trellis_v3_synthetic", "total_size": int(total_size)},
                "weight_map": weight_map,
            },
            indent=2,
            sort_keys=True,
        )
    )
    (model_path / "quantization_index.json").write_text(
        json.dumps({"layers": quantization_layers}, indent=2, sort_keys=True)
    )

    for layer_idx in range(num_hidden_layers):
        layer_dir = model_path / f"layer_{layer_idx:04d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        save_safetensors(per_layer_tensors[layer_idx], str(layer_dir / "tensor_0001.safetensors"))
        (layer_dir / "index.json").write_text(
            json.dumps(
                {
                    "layer_idx": layer_idx,
                    "total_tensors": len(per_layer_entries[layer_idx]),
                    "tensors": per_layer_entries[layer_idx],
                },
                indent=2,
                sort_keys=True,
            )
        )

    return SyntheticTrellisFixtureMetadata(
        model_path=model_path,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_hidden_layers,
    )


def run_benchmark(output_path: Path) -> dict[str, Any]:
    """Run one-pass Trellis synthetic smoke benchmark and write JSON output."""
    output_path = output_path.resolve()

    if not _has_mps():
        skipped = {
            "benchmark": BENCHMARK_NAME,
            "timestamp_utc": _utc_now(),
            "skipped": True,
            "reason": "MPS unavailable (requires Apple Silicon + MPS runtime).",
        }
        _write_results(output_path, skipped)
        return skipped

    from metal_marlin.gptq_metal import GPTQMetal
    from metal_marlin.kv_cache import TrellisKVCache
    from metal_marlin.trellis.lm import TrellisForCausalLM
    from tests.helpers.synthetic_trellis_fixture import (
        get_checked_in_synthetic_trellis_fixture_path,
    )

    torch.set_grad_enabled(False)
    torch.manual_seed(2026)

    prefill_tokens = 4

    fixture_path = get_checked_in_synthetic_trellis_fixture_path()
    if not fixture_path.exists():
        raise FileNotFoundError(
            f"Checked-in synthetic fixture missing: {fixture_path}"
        )

    fixture_config = json.loads((fixture_path / "config.json").read_text())
    fixture = SyntheticTrellisFixtureMetadata(
        model_path=fixture_path,
        vocab_size=int(fixture_config["vocab_size"]),
        hidden_size=int(fixture_config["hidden_size"]),
        num_layers=int(fixture_config["num_hidden_layers"]),
    )

    t0 = time.perf_counter()
    model = TrellisForCausalLM.from_pretrained(
        fixture.model_path,
        device="mps",
        optimize_memory=True,
    )
    model.eval()
    _mps_sync()
    model_load_ms = (time.perf_counter() - t0) * 1000.0

    prefill_ids = torch.randint(
        low=0,
        high=fixture.vocab_size,
        size=(1, prefill_tokens),
        device="mps",
        dtype=torch.long,
    )
    kv_cache = TrellisKVCache(
        num_layers=model.config.num_hidden_layers,
        batch_size=1,
        max_seq_len=prefill_tokens + 4,
        kv_lora_rank=model.config.kv_lora_rank,
        qk_rope_head_dim=model.config.qk_rope_head_dim,
        device="mps",
    )

    with torch.inference_mode():
        t0 = time.perf_counter()
        _ = model(prefill_ids, kv_cache=kv_cache).logits
        _mps_sync()
        prefill_latency_ms = (time.perf_counter() - t0) * 1000.0

    decode_ids = torch.randint(
        low=0,
        high=fixture.vocab_size,
        size=(1, 1),
        device="mps",
        dtype=torch.long,
    )
    with torch.inference_mode():
        t0 = time.perf_counter()
        _ = model(decode_ids, kv_cache=kv_cache).logits
        _mps_sync()
        decode_latency_ms = (time.perf_counter() - t0) * 1000.0

    gptq = GPTQMetal()
    activations = torch.randn(
        32,
        fixture.hidden_size,
        device="mps",
        dtype=torch.float16,
    )
    t0 = time.perf_counter()
    hessian = gptq.compute_hessian(activations, normalize=True)
    _mps_sync()
    hessian_latency_ms = (time.perf_counter() - t0) * 1000.0

    assert tuple(hessian.shape) == (fixture.hidden_size, fixture.hidden_size)

    results = {
        "benchmark": BENCHMARK_NAME,
        "timestamp_utc": _utc_now(),
        "fixture": {
            "hidden_size": fixture.hidden_size,
            "num_layers": fixture.num_layers,
            "vocab_size": fixture.vocab_size,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": 1,
        },
        "metrics": {
            "model_load_time_ms": model_load_ms,
            "prefill_step_latency_ms": prefill_latency_ms,
            "decode_step_latency_ms": decode_latency_ms,
            "hessian_compute_latency_ms": hessian_latency_ms,
        },
    }
    _write_results(output_path, results)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run synthetic Trellis/Hessian smoke benchmark and write JSON results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    results = run_benchmark(args.output)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
