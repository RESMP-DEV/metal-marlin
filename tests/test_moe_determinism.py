"""Lightweight MoE determinism test - loads single layer only."""

from __future__ import annotations

import gc
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest
import torch

pytestmark = pytest.mark.requires_mps


def _copy_with_optional_transpose(
    dst: torch.Tensor,
    src: torch.Tensor,
    *,
    device: str,
) -> None:
    if src.shape == dst.shape:
        aligned = src
    elif src.ndim == dst.ndim and src.transpose(-1, -2).shape == dst.shape:
        aligned = src.transpose(-1, -2)
    else:
        raise ValueError(
            f"Shape mismatch: source={tuple(src.shape)} destination={tuple(dst.shape)}"
        )
    dst.copy_(aligned.contiguous().to(device=device, dtype=dst.dtype))


def _load_layer_moe_weights(
    *,
    model_dir: Path,
    layer_idx: int,
    num_experts: int,
    weight_map: dict[str, str],
    experts: Any,
    safe_open: Any,
    device: str,
) -> None:
    def _name(expert_idx: int, proj_name: str, suffix: str) -> str:
        return f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}.{suffix}"

    layer_prefix = f"model.layers.{layer_idx}.mlp.experts."
    layer_keys = {k: v for k, v in weight_map.items() if k.startswith(layer_prefix)}
    needed_shards = sorted(set(layer_keys.values()))

    with ExitStack() as stack:
        shard_handles = {
            shard: stack.enter_context(
                safe_open(str(model_dir / shard), framework="pt")
            )
            for shard in needed_shards
        }

        for expert_idx in range(num_experts):
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                weight_name = _name(expert_idx, proj_name, "weight")
                scale_name = _name(expert_idx, proj_name, "scales")
                if weight_name not in layer_keys or scale_name not in layer_keys:
                    raise KeyError(
                        "Missing MoE tensor(s) for "
                        f"layer={layer_idx} expert={expert_idx} proj={proj_name}"
                    )

                packed_src = shard_handles[layer_keys[weight_name]].get_tensor(weight_name)
                scales_src = shard_handles[layer_keys[scale_name]].get_tensor(scale_name)

                packed_dst = getattr(experts, f"{proj_name}_packed")[expert_idx]
                scales_dst = getattr(experts, f"{proj_name}_scales")[expert_idx]
                _copy_with_optional_transpose(packed_dst, packed_src, device=device)
                _copy_with_optional_transpose(scales_dst, scales_src, device=device)


def test_moe_forward_deterministic() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS backend required for MoE determinism test.")

    safetensors = pytest.importorskip("safetensors")
    from metal_marlin.glm4_moe_experts import QuantizedGlm4MoEExperts

    device = "mps"
    layer_idx = 1
    model_dir = Path(__file__).resolve().parents[1] / "models" / "glm47-flash-mmfp4"
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"

    if not config_path.exists() or not index_path.exists():
        pytest.skip(f"Missing model files under {model_dir}")

    config = json.loads(config_path.read_text())
    weight_map = json.loads(index_path.read_text())["weight_map"]

    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config.get("moe_intermediate_size", config["intermediate_size"]))
    num_experts = int(config.get("n_routed_experts", 64))
    group_size = 128

    layer_prefix = f"model.layers.{layer_idx}.mlp.experts."
    layer_keys = {k: v for k, v in weight_map.items() if k.startswith(layer_prefix)}
    if not layer_keys:
        pytest.skip(f"No layer {layer_idx} MoE expert weights found in model index.")

    missing_shards = sorted(
        shard for shard in set(layer_keys.values()) if not (model_dir / shard).exists()
    )
    if missing_shards:
        pytest.skip(f"Missing required layer-{layer_idx} shard(s): {missing_shards}")

    experts = QuantizedGlm4MoEExperts(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        group_size=group_size,
        device=device,
    )

    hidden: torch.Tensor | None = None
    top_k_idx: torch.Tensor | None = None
    top_k_weights: torch.Tensor | None = None
    results: list[torch.Tensor] = []
    try:
        # Load ONLY layer-1 MoE expert tensors from safetensors shards.
        _load_layer_moe_weights(
            model_dir=model_dir,
            layer_idx=layer_idx,
            num_experts=num_experts,
            weight_map=weight_map,
            experts=experts,
            safe_open=safetensors.safe_open,
            device=device,
        )
        experts.eval()

        torch.manual_seed(0)
        hidden = torch.randn(1, hidden_size, dtype=torch.float16, device=device)
        top_k_idx = torch.tensor([[6, 11]], dtype=torch.long, device=device)
        top_k_weights = torch.tensor([[0.6, 0.4]], dtype=torch.float16, device=device)

        for _ in range(3):
            torch.mps.synchronize()
            out = experts.forward(hidden, top_k_idx, top_k_weights)
            torch.mps.synchronize()
            results.append(out.clone())

        assert torch.allclose(results[0], results[1], atol=1e-5)
        assert torch.allclose(results[1], results[2], atol=1e-5)
    finally:
        del experts
        if hidden is not None:
            del hidden
        if top_k_idx is not None:
            del top_k_idx
        if top_k_weights is not None:
            del top_k_weights
        if results:
            del results
        gc.collect()
        torch.mps.empty_cache()
