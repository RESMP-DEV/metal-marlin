"""Synthetic Trellis fixture builder used for API + dispatch regressions.

This helper creates a tiny deterministic checkpoint locally (no network access)
that is intentionally sized for fast test/runtime execution.

The generated directory includes the files needed for:
- ``TrellisForCausalLM.from_pretrained(...)``
- ``TrellisModel.from_pretrained(...)``

The fixture is specifically intended to validate Trellis API surface and
runtime dispatch paths, not model quality.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file as save_safetensors


@dataclass(frozen=True, slots=True)
class SyntheticTrellisFixtureMetadata:
    """Metadata returned by synthetic Trellis fixture builders."""

    model_path: str
    vocab_size: int
    hidden_size: int
    layer_count: int
    num_layers: int
    num_experts: int
    config: dict[str, Any]


def _dot_to_safetensors_key(tensor_name: str) -> str:
    """Convert dotted tensor names to metal_marlin trellis key format."""
    return tensor_name.replace(".", "__")


def _packed_tile_bytes(bits: int) -> int:
    return (256 * bits + 7) // 8


def _validate_fixture_shape(
    *,
    hidden_size: int,
    num_hidden_layers: int,
    num_experts: int,
    bits: int,
) -> None:
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if hidden_size > 64:
        raise ValueError(f"hidden_size must be <= 64 for tiny synthetic fixtures, got {hidden_size}")
    if hidden_size % 4 != 0:
        raise ValueError(f"hidden_size must be divisible by 4, got {hidden_size}")

    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    if num_hidden_layers > 2:
        raise ValueError(
            f"num_hidden_layers must be <= 2 for tiny synthetic fixtures, got {num_hidden_layers}"
        )

    if num_experts <= 0:
        raise ValueError("num_experts must be >= 1")

    if bits < 2 or bits > 8:
        raise ValueError(f"bits must be in [2, 8], got {bits}")


def _iter_layer_weight_specs(
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
    num_experts: int,
) -> list[tuple[str, int, int]]:
    """List (tensor_name, out_features, in_features) for one transformer layer."""
    prefix = f"model.layers.{layer_idx}"
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    specs: list[tuple[str, int, int]] = [
        (f"{prefix}.self_attn.q_a_proj.weight", q_lora_rank, hidden_size),
        (f"{prefix}.self_attn.q_b_proj.weight", num_attention_heads * qk_head_dim, q_lora_rank),
        (f"{prefix}.self_attn.kv_a_proj_with_mqa.weight", kv_lora_rank + qk_rope_head_dim, hidden_size),
        (
            f"{prefix}.self_attn.kv_b_proj.weight",
            num_kv_heads * (qk_nope_head_dim + v_head_dim),
            kv_lora_rank,
        ),
        (f"{prefix}.self_attn.o_proj.weight", hidden_size, num_attention_heads * v_head_dim),
    ]

    if num_experts > 1:
        for expert_idx in range(num_experts):
            specs.extend(
                [
                    (
                        f"{prefix}.mlp.experts.{expert_idx}.gate_proj.weight",
                        intermediate_size,
                        hidden_size,
                    ),
                    (
                        f"{prefix}.mlp.experts.{expert_idx}.up_proj.weight",
                        intermediate_size,
                        hidden_size,
                    ),
                    (
                        f"{prefix}.mlp.experts.{expert_idx}.down_proj.weight",
                        hidden_size,
                        intermediate_size,
                    ),
                ]
            )
        # Shared experts are loaded unconditionally by Trellis MoE paths.
        specs.extend(
            [
                (f"{prefix}.mlp.shared_experts.gate_proj.weight", intermediate_size, hidden_size),
                (f"{prefix}.mlp.shared_experts.up_proj.weight", intermediate_size, hidden_size),
                (f"{prefix}.mlp.shared_experts.down_proj.weight", hidden_size, intermediate_size),
            ]
        )
    else:
        specs.extend(
            [
                (f"{prefix}.mlp.gate_proj.weight", intermediate_size, hidden_size),
                (f"{prefix}.mlp.up_proj.weight", intermediate_size, hidden_size),
                (f"{prefix}.mlp.down_proj.weight", hidden_size, intermediate_size),
            ]
        )

    return specs


def _make_quantized_weight_components(
    *,
    tensor_name: str,
    out_features: int,
    in_features: int,
    bits: int,
    rng: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Create deterministic synthetic trellis components for one quantized weight."""
    tiles_out = (out_features + 15) // 16
    tiles_in = (in_features + 15) // 16
    packed_len = tiles_out * tiles_in * _packed_tile_bytes(bits)

    indices = torch.randint(0, 256, (1 + packed_len,), dtype=torch.uint8, generator=rng)
    indices[0] = bits  # Trellis packed header byte

    n_groups = max(1, math.ceil(in_features / 128))
    scales = torch.rand((n_groups, out_features), dtype=torch.float32, generator=rng) * 0.05 + 0.005
    su = torch.randint(0, 2, (in_features,), dtype=torch.int8, generator=rng).to(torch.float32)
    sv = torch.randint(0, 2, (out_features,), dtype=torch.int8, generator=rng).to(torch.float32)
    su = su.mul_(2.0).sub_(1.0)
    sv = sv.mul_(2.0).sub_(1.0)

    base_key = _dot_to_safetensors_key(tensor_name)
    return {
        f"{base_key}__indices": indices,
        f"{base_key}__scales": scales,
        f"{base_key}__su": su,
        f"{base_key}__sv": sv,
    }


def build_synthetic_trellis_fixture(
    tmp_path: Path | str,
    num_experts: int = 1,
    hidden_size: int = 64,
    num_hidden_layers: int = 2,
    vocab_size: int = 128,
    intermediate_size: int | None = None,
    seed: int = 2026,
    bits: int = 4,
    layer_count: int | None = None,
    model_name: str | None = None,
) -> SyntheticTrellisFixtureMetadata:
    """Build a tiny deterministic Trellis checkpoint directory on local disk.

    Notes:
    - This function exists for API + dispatch regression validation.
    - It intentionally keeps dimensions tiny for CI/runtime speed.
    - ``layer_count`` is accepted as an alias for ``num_hidden_layers``.
    """
    if layer_count is not None:
        num_hidden_layers = layer_count

    if intermediate_size is None:
        intermediate_size = hidden_size * 2

    _validate_fixture_shape(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_experts=num_experts,
        bits=bits,
    )

    root_dir = Path(tmp_path)
    fixture_dir_name = model_name or f"synthetic-trellis-{num_experts}expert"
    model_path = root_dir / fixture_dir_name
    model_path.mkdir(parents=True, exist_ok=True)

    num_attention_heads = 4
    num_kv_heads = 2
    q_lora_rank = 16
    kv_lora_rank = 16
    qk_nope_head_dim = 8
    qk_rope_head_dim = 8
    v_head_dim = qk_nope_head_dim + qk_rope_head_dim

    rng = torch.Generator(device="cpu").manual_seed(seed)

    config: dict[str, Any] = {
        "architectures": ["TrellisForCausalLM"],
        "model_type": "trellis",
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_kv_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": qk_nope_head_dim + qk_rope_head_dim,
        "kv_lora_rank": kv_lora_rank,
        "q_lora_rank": q_lora_rank,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "num_experts": num_experts,
        "num_shared_experts": 0,
        "num_experts_per_tok": min(2, num_experts) if num_experts > 1 else 1,
        "first_moe_layer": 0 if num_experts > 1 else num_hidden_layers,
        "max_position_embeddings": 512,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "quantization_bits": bits,
        "torch_dtype": "float32",
    }
    if num_experts > 1:
        config["moe_intermediate_size"] = intermediate_size

    (model_path / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    base_weights: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": (
            torch.randn((vocab_size, hidden_size), dtype=torch.float32, generator=rng) * 0.02
        ),
        "model.norm.weight": torch.ones(hidden_size, dtype=torch.float32),
        "lm_head.weight": (
            torch.randn((vocab_size, hidden_size), dtype=torch.float32, generator=rng) * 0.02
        ),
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
        if num_experts > 1:
            base_weights[f"{prefix}.mlp.gate.weight"] = torch.randn(
                (num_experts, hidden_size),
                dtype=torch.float32,
                generator=rng,
            ) * 0.02

    save_safetensors(base_weights, str(model_path / "base_weights.safetensors"))

    shard_name = "model-00001-of-00001.safetensors"
    shard_tensors: dict[str, torch.Tensor] = {}
    quantization_layers: list[dict[str, Any]] = []

    for layer_idx in range(num_hidden_layers):
        layer_tensors: dict[str, torch.Tensor] = {}
        layer_entries: list[dict[str, Any]] = []

        for tensor_name, out_features, in_features in _iter_layer_weight_specs(
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
            num_experts=num_experts,
        ):
            components = _make_quantized_weight_components(
                tensor_name=tensor_name,
                out_features=out_features,
                in_features=in_features,
                bits=bits,
                rng=rng,
            )
            layer_tensors.update(components)
            shard_tensors.update(components)

            tensor_meta = {
                "name": tensor_name,
                "bits": bits,
                "shape": [out_features, in_features],
                "mse": 0.0,
            }
            layer_entries.append(tensor_meta)
            quantization_layers.append(dict(tensor_meta))

        layer_dir = model_path / f"layer_{layer_idx:04d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        save_safetensors(layer_tensors, str(layer_dir / "tensor_0001.safetensors"))
        (layer_dir / "index.json").write_text(
            json.dumps(
                {
                    "layer_idx": layer_idx,
                    "total_tensors": len(layer_entries),
                    "tensors": layer_entries,
                },
                indent=2,
                sort_keys=True,
            )
        )

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

    tokenizer_config = {
        "model_max_length": 512,
        "tokenizer_class": "LlamaTokenizer",
    }
    (model_path / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, sort_keys=True)
    )
    (model_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {
                    "type": "BPE",
                    "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2},
                    "merges": [],
                }
            },
            indent=2,
            sort_keys=True,
        )
    )
    (model_path / "special_tokens_map.json").write_text(
        json.dumps({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}, indent=2)
    )

    return SyntheticTrellisFixtureMetadata(
        model_path=str(model_path),
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        layer_count=num_hidden_layers,
        num_layers=num_hidden_layers,
        num_experts=num_experts,
        config=config,
    )


def create_synthetic_trellis_fixture(
    tmp_path: Path | str,
    *,
    num_experts: int = 1,
    hidden_size: int = 64,
    num_hidden_layers: int = 2,
    layer_count: int | None = None,
    vocab_size: int = 128,
    seed: int = 2026,
    bits: int = 4,
) -> SyntheticTrellisFixtureMetadata:
    """Convenience wrapper for tests that need a one-call tiny fixture builder."""
    return build_synthetic_trellis_fixture(
        tmp_path=tmp_path,
        num_experts=num_experts,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        layer_count=layer_count,
        vocab_size=vocab_size,
        seed=seed,
        bits=bits,
    )


def get_checked_in_synthetic_trellis_fixture_path() -> Path:
    """Return path to the checked-in tiny synthetic Trellis fixture."""
    return Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_trellis_smoke"


__all__ = [
    "SyntheticTrellisFixtureMetadata",
    "build_synthetic_trellis_fixture",
    "create_synthetic_trellis_fixture",
    "get_checked_in_synthetic_trellis_fixture_path",
]
