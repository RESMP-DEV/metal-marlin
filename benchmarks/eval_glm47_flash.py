#!/usr/bin/env python3
"""GLM-4.7-Flash quality evaluation with MoE + MLA metrics.

Evaluates:
- MoE config (64 routed + 1 shared expert)
- Expert activation frequency and utilization stats
- Shared expert contribution vs routed experts
- MLA cache compression and cache-consistency quality
- Optional perplexity on WikiText-2

Usage:
  cd contrib/metal_marlin
  uv run python benchmarks/eval_glm47_flash.py
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    import torch
except Exception as exc:  # pragma: no cover - runtime guard
    raise SystemExit(f"PyTorch required: {exc}")

try:
    from transformers import AutoTokenizer, Glm4MoeLiteForCausalLM
except Exception as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "Transformers>=5.0.0 required with Glm4MoeLiteForCausalLM available. "
        f"Import error: {exc}"
    )

from metal_marlin.eval_perplexity import (  # noqa: E402
    compute_perplexity_from_logits,
    load_wikitext2,
)
from metal_marlin.paged.mla_cache import compare_memory_usage  # noqa: E402


@dataclass
class ExpertEnergyStats:
    shared_energy_ratio: float
    routed_energy_ratio: float


@dataclass
class CacheQualityStats:
    rmse: float
    max_abs: float


@dataclass
class EvalResults:
    model_id: str
    device: str
    dtype: str
    timestamp: str

    # MoE config
    n_routed_experts: int
    n_shared_experts: int
    top_k: int

    # Expert usage
    experts_used_ratio: float
    experts_used_count: int
    expert_activation_frequency: dict[int, float]

    # Shared vs routed
    shared_vs_routed: ExpertEnergyStats

    # MLA cache
    mla_cache: dict[str, float]
    cache_quality: CacheQualityStats

    # Quality
    perplexity: float | None = None
    ppl_tokens: int | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def _extract_tensor(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    if hasattr(output, "logits"):
        return output.logits  # type: ignore[no-any-return]
    return None


def _infer_topk(indices: torch.Tensor, weights: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    if indices.dtype.is_floating_point and weights is not None and not weights.dtype.is_floating_point:
        return weights, indices
    return indices, weights


def _parse_router_output(output: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    indices = None
    weights = None
    if isinstance(output, (tuple, list)) and output:
        if len(output) >= 1:
            indices = output[0]
        if len(output) >= 2:
            weights = output[1]
    elif isinstance(output, dict):
        indices = output.get("topk_indices") or output.get("indices")
        weights = output.get("topk_weights") or output.get("weights")
    else:
        if hasattr(output, "topk_indices"):
            indices = getattr(output, "topk_indices")
            weights = getattr(output, "topk_weights", None)
        elif hasattr(output, "indices"):
            indices = getattr(output, "indices")
            weights = getattr(output, "weights", None)

    if indices is None and isinstance(output, torch.Tensor):
        if output.ndim >= 2:
            k = min(2, output.shape[-1])
            weights, indices = torch.topk(output, k=k, dim=-1)
            weights = torch.softmax(output, dim=-1).gather(-1, indices)

    if indices is None:
        return None, None

    if weights is not None and isinstance(indices, torch.Tensor) and isinstance(weights, torch.Tensor):
        indices, weights = _infer_topk(indices, weights)

    return indices, weights


def _norm_stats_store() -> dict[int, dict[str, float]]:
    return defaultdict(lambda: {"sum_sq": 0.0, "numel": 0.0})


def analyze_expert_usage(model, input_ids_list: list[torch.Tensor]) -> tuple[
    dict[tuple[int, int], int],
    dict[tuple[int, int], float],
    ExpertEnergyStats,
]:
    """Track which experts are activated during inference."""
    backbone = getattr(model, "model", None)
    if backbone is None or not hasattr(backbone, "layers"):
        raise RuntimeError("Unexpected model structure: missing model.layers")

    expert_counts: dict[tuple[int, int], int] = defaultdict(int)
    expert_weight_sums: dict[tuple[int, int], float] = defaultdict(float)

    total_stats = _norm_stats_store()
    shared_stats = _norm_stats_store()

    handles = []

    def make_router_hook(layer_idx: int):
        def hook(module, _input, output):
            indices, weights = _parse_router_output(output)
            if indices is None:
                return
            flat_indices = indices.reshape(-1).detach().cpu().tolist()
            for idx in flat_indices:
                expert_counts[(layer_idx, int(idx))] += 1
            if weights is not None:
                flat_weights = weights.reshape(-1).detach().cpu().tolist()
                for idx, weight in zip(flat_indices, flat_weights):
                    expert_weight_sums[(layer_idx, int(idx))] += float(weight)
        return hook

    def make_norm_hook(store: dict[int, dict[str, float]], layer_idx: int):
        def hook(module, _input, output):
            tensor = _extract_tensor(output)
            if tensor is None:
                return
            data = tensor.detach().float()
            store[layer_idx]["sum_sq"] += float((data * data).sum().item())
            store[layer_idx]["numel"] += float(data.numel())
        return hook

    for i, layer in enumerate(backbone.layers[1:], 1):  # Skip dense layer 0
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue

        router = None
        for attr in ("gate", "router", "gate_proj", "router_gate"):
            if hasattr(mlp, attr):
                router = getattr(mlp, attr)
                break

        if router is not None:
            handles.append(router.register_forward_hook(make_router_hook(i)))

        shared = getattr(mlp, "shared_experts", None)
        if shared is not None:
            if isinstance(shared, torch.nn.ModuleList):
                for shared_idx, mod in enumerate(shared):
                    handles.append(mod.register_forward_hook(make_norm_hook(shared_stats, i)))
            else:
                handles.append(shared.register_forward_hook(make_norm_hook(shared_stats, i)))

        handles.append(mlp.register_forward_hook(make_norm_hook(total_stats, i)))

    with torch.no_grad():
        for input_ids in input_ids_list:
            _ = model(input_ids)

    for handle in handles:
        handle.remove()

    shared_energy = 0.0
    total_energy = 0.0
    for layer_idx, totals in total_stats.items():
        total_energy += totals["sum_sq"]
        shared_energy += shared_stats.get(layer_idx, {}).get("sum_sq", 0.0)

    if total_energy > 0:
        shared_ratio = shared_energy / total_energy
    else:
        shared_ratio = 0.0

    shared_vs_routed = ExpertEnergyStats(
        shared_energy_ratio=float(shared_ratio),
        routed_energy_ratio=float(max(0.0, 1.0 - shared_ratio)),
    )

    return expert_counts, expert_weight_sums, shared_vs_routed


def summarize_expert_usage(
    expert_counts: dict[tuple[int, int], int],
    n_experts: int,
) -> tuple[float, int, dict[int, float]]:
    per_expert = defaultdict(int)
    for (_layer_idx, expert_idx), count in expert_counts.items():
        per_expert[expert_idx] += count

    used_experts = {idx for idx, count in per_expert.items() if count > 0}
    used_ratio = len(used_experts) / n_experts if n_experts else 0.0

    total_routes = sum(per_expert.values())
    freq = {}
    for idx in range(n_experts):
        freq[idx] = per_expert.get(idx, 0) / total_routes if total_routes else 0.0

    return float(used_ratio), len(used_experts), freq


def evaluate_cache_consistency(model, input_ids: torch.Tensor) -> CacheQualityStats:
    with torch.no_grad():
        full = model(input_ids, use_cache=True)
        full_logits = full.logits.detach()

        past = None
        step_logits = []
        for i in range(input_ids.shape[1]):
            token = input_ids[:, i : i + 1]
            out = model(token, use_cache=True, past_key_values=past)
            past = out.past_key_values
            step_logits.append(out.logits.detach())

        step_logits = torch.cat(step_logits, dim=1)
        diff = (full_logits - step_logits).float()
        rmse = torch.sqrt(torch.mean(diff * diff)).item()
        max_abs = torch.max(torch.abs(diff)).item()

    return CacheQualityStats(rmse=float(rmse), max_abs=float(max_abs))


def main() -> int:
    parser = argparse.ArgumentParser(description="GLM-4.7-Flash quality evaluation")
    parser.add_argument("--model-id", default="zai-org/GLM-4.7-Flash")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--samples", type=int, default=20, help="WikiText-2 samples")
    parser.add_argument("--max-length", type=int, default=256, help="Max length per sample")
    parser.add_argument("--expert-samples", type=int, default=6)
    parser.add_argument("--cache-check-len", type=int, default=128)
    parser.add_argument("--mla-seq-len", type=int, default=4096)
    parser.add_argument(
        "--output",
        default=str(_ROOT / "benchmarks" / "results" / "glm47_flash_eval.json"),
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    torch_dtype = _resolve_dtype(args.dtype)

    print("=" * 70)
    print("GLM-4.7-Flash Quality Evaluation")
    print("=" * 70)

    print(f"\nLoading model: {args.model_id}")
    start = time.perf_counter()
    model = Glm4MoeLiteForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    load_time = time.perf_counter() - start
    print(f"Loaded in {load_time:.1f}s")

    config = model.config
    n_routed = int(getattr(config, "n_routed_experts", getattr(config, "num_experts", 0)) or 0)
    n_shared = int(getattr(config, "n_shared_experts", getattr(config, "shared_experts", 0)) or 0)
    top_k = int(getattr(config, "moe_top_k", getattr(config, "top_k", 0)) or 0)

    if n_routed != 64 or n_shared != 1:
        print(
            f"WARNING: Expected 64 routed + 1 shared expert, got {n_routed} routed + {n_shared} shared"
        )

    print("\n[1/4] MoE config")
    print(f"  Routed experts: {n_routed}")
    print(f"  Shared experts: {n_shared}")
    print(f"  Top-k: {top_k}")

    print("\n[2/4] MLA cache compression")
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
    head_dim = int(config.hidden_size // num_heads)
    kv_lora_rank = int(getattr(config, "kv_lora_rank", 0) or 0)

    mla_cache_stats = compare_memory_usage(
        seq_len=args.mla_seq_len,
        num_layers=int(config.num_hidden_layers),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_lora_rank=kv_lora_rank,
        dtype_bytes=2,
    )

    for key, value in mla_cache_stats.items():
        if "ratio" in key:
            print(f"  {key}: {value:.2f}x")
        else:
            print(f"  {key}: {value:.2f}")

    print("\n[3/4] MLA cache consistency")
    texts = load_wikitext2(max_samples=max(args.samples, args.expert_samples))
    if not texts:
        print("  ERROR: No evaluation texts loaded")
        return 1

    cache_text = texts[0]
    cache_tokens = tokenizer(
        cache_text,
        return_tensors="pt",
        truncation=True,
        max_length=args.cache_check_len,
    ).input_ids.to(device)

    cache_quality = evaluate_cache_consistency(model, cache_tokens)
    print(f"  cache_rmse: {cache_quality.rmse:.6f}")
    print(f"  cache_max_abs: {cache_quality.max_abs:.6f}")

    print("\n[4/4] Expert usage + perplexity")
    expert_texts = texts[: args.expert_samples]
    input_ids_list = [
        tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        ).input_ids.to(device)
        for text in expert_texts
    ]

    expert_counts, expert_weight_sums, shared_vs_routed = analyze_expert_usage(
        model, input_ids_list
    )

    experts_used_ratio, experts_used_count, expert_activation_frequency = summarize_expert_usage(
        expert_counts, n_routed
    )

    print(f"  experts_used: {experts_used_count}/{n_routed} ({experts_used_ratio * 100:.1f}%)")
    print(
        "  shared_energy_ratio: "
        f"{shared_vs_routed.shared_energy_ratio * 100:.2f}%"
    )

    # Optional perplexity (quick estimate)
    def logits_fn(input_ids_np: np.ndarray) -> np.ndarray:
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        input_ids = torch.tensor(input_ids_np, dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits.detach().float().cpu().numpy()
        return logits

    sample_texts = texts[: args.samples]
    perplexity = compute_perplexity_from_logits(
        logits_fn=logits_fn,
        tokenizer=tokenizer,
        texts=sample_texts,
        max_length=args.max_length,
        verbose=True,
    )

    token_count = sum(
        len(tokenizer.encode(text, truncation=True, max_length=args.max_length))
        for text in sample_texts
    )

    print(f"  perplexity: {perplexity:.4f} ({token_count} tokens)")

    results = EvalResults(
        model_id=args.model_id,
        device=device,
        dtype=args.dtype,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        n_routed_experts=n_routed,
        n_shared_experts=n_shared,
        top_k=top_k,
        experts_used_ratio=experts_used_ratio,
        experts_used_count=experts_used_count,
        expert_activation_frequency={
            int(k): float(v) for k, v in expert_activation_frequency.items()
        },
        shared_vs_routed=shared_vs_routed,
        mla_cache=mla_cache_stats,
        cache_quality=cache_quality,
        perplexity=float(perplexity),
        ppl_tokens=int(token_count),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results.to_json(), f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
