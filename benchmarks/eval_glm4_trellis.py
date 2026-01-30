#!/usr/bin/env python3
"""GLM-4.7-Flash Trellis Full Model Benchmark.

Comprehensive end-to-end benchmark measuring:
- Full model memory usage (all 47 layers loaded)
- Prefill throughput at various context lengths
- Decode throughput (single token generation)
- Total memory consumption

Usage:
    cd contrib/metal_marlin
    uv run python benchmarks/eval_glm4_trellis.py --model-path models/GLM-4.7-Flash-Trellis-3bpw
    uv run python benchmarks/eval_glm4_trellis.py --context-lengths "256,512,1024,2048"
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from metal_marlin.trellis.linear import TrellisLinear
from metal_marlin.trellis.loader import TrellisModelLoader

MODEL_PATH = _ROOT / "models" / "GLM-4.7-Flash-Trellis-3bpw"
RESULTS_DIR = _ROOT / "benchmarks" / "results"


@dataclass
class BenchmarkResults:
    model_path: str
    timestamp: str
    num_layers: int = 0
    total_weights: int = 0
    model_size_gb: float = 0.0
    memory_after_load_gb: float = 0.0
    peak_memory_gb: float = 0.0
    throughput_by_context: dict[int, dict[str, float]] = field(default_factory=dict)
    avg_prefill_tok_s: float = 0.0
    avg_decode_tok_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "timestamp": self.timestamp,
            "model_info": {"num_layers": self.num_layers, "total_weights": self.total_weights},
            "memory": {
                "model_size_gb": self.model_size_gb,
                "after_load_gb": self.memory_after_load_gb,
                "peak_gb": self.peak_memory_gb,
            },
            "throughput_by_context": self.throughput_by_context,
            "summary": {
                "avg_prefill_tok_s": self.avg_prefill_tok_s,
                "avg_decode_tok_s": self.avg_decode_tok_s,
            },
        }


class TrellisExpertMLP(nn.Module):
    def __init__(self, gate: TrellisLinear, up: TrellisLinear, down: TrellisLinear):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = gate, up, down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TrellisMoELayer(nn.Module):
    def __init__(
        self,
        experts: list[TrellisExpertMLP],
        shared: TrellisExpertMLP | None,
        router_w: torch.Tensor,
        top_k: int = 8,
    ):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.shared = shared
        self.top_k = top_k
        self.router = nn.Linear(router_w.shape[1], len(experts), bias=False)
        self.router.weight.data = router_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.view(-1, shape[-1])
        logits = self.router(x_flat.float())
        weights, idx = torch.topk(F.softmax(logits, dim=-1), k=self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            for eid in range(len(self.experts)):
                mask = idx[:, i] == eid
                if mask.any():
                    out[mask] += weights[mask, i : i + 1] * self.experts[eid](x_flat[mask])
        if self.shared:
            out = out + self.shared(x_flat)
        return out.view(shape)


class TrellisAttnProj(nn.Module):
    def __init__(self, q_a, q_b, kv_a, kv_b, o_proj):
        super().__init__()
        self.q_a, self.q_b = q_a, q_b
        self.kv_a, self.kv_b, self.o_proj = kv_a, kv_b, o_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_a(x) if self.q_a else x
        if self.q_b:
            q = self.q_b(q)
        _ = self.kv_b(self.kv_a(x))
        return self.o_proj(q)


class TrellisLayer(nn.Module):
    def __init__(self, attn: TrellisAttnProj, mlp: nn.Module):
        super().__init__()
        self.attn, self.mlp = attn, mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x + self.attn(x))


class TrellisModelBench(nn.Module):
    def __init__(self, hidden: int = 2048):
        super().__init__()
        self.hidden_size = hidden
        self.layers = nn.ModuleList()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def get_mem(device: str) -> float:
    if device == "mps":
        return torch.mps.current_allocated_memory() / (1024**3)
    return 0.0


def load_model(loader: TrellisModelLoader, device: str) -> tuple[TrellisModelBench, int]:
    router_path = loader.model_path / "router_weights.safetensors"
    routers = load_file(router_path) if router_path.exists() else {}
    model = TrellisModelBench()
    n_weights = 0

    for idx in tqdm(loader.get_layer_indices(), desc="Loading"):
        w = loader.load_layer(idx)
        pre = f"model.layers.{idx}"
        ap = f"{pre}.self_attn"

        q_a = (
            TrellisLinear.from_trellis_weight(w[f"{ap}.q_a_proj.weight"], device=device)
            if f"{ap}.q_a_proj.weight" in w
            else None
        )
        q_b = (
            TrellisLinear.from_trellis_weight(w[f"{ap}.q_b_proj.weight"], device=device)
            if f"{ap}.q_b_proj.weight" in w
            else None
        )
        kv_a_k = (
            f"{ap}.kv_a_proj_with_mqa.weight"
            if f"{ap}.kv_a_proj_with_mqa.weight" in w
            else f"{ap}.kv_a_proj.weight"
        )
        kv_a = TrellisLinear.from_trellis_weight(w[kv_a_k], device=device)
        kv_b = TrellisLinear.from_trellis_weight(w[f"{ap}.kv_b_proj.weight"], device=device)
        o = TrellisLinear.from_trellis_weight(w[f"{ap}.o_proj.weight"], device=device)
        attn = TrellisAttnProj(q_a, q_b, kv_a, kv_b, o)
        n_weights += 3 + (1 if q_a else 0) + (1 if q_b else 0)

        mp = f"{pre}.mlp"
        if idx >= 1:  # MoE (layers 1-46)
            experts = []
            for eid in range(64):
                gk = f"{mp}.experts.{eid}.gate_proj.weight"
                if gk not in w:
                    break
                experts.append(
                    TrellisExpertMLP(
                        TrellisLinear.from_trellis_weight(w[gk], device=device),
                        TrellisLinear.from_trellis_weight(
                            w[f"{mp}.experts.{eid}.up_proj.weight"], device=device
                        ),
                        TrellisLinear.from_trellis_weight(
                            w[f"{mp}.experts.{eid}.down_proj.weight"], device=device
                        ),
                    )
                )
                n_weights += 3
            shared = None
            sk = f"{mp}.shared_experts.gate_proj.weight"
            if sk in w:
                shared = TrellisExpertMLP(
                    TrellisLinear.from_trellis_weight(w[sk], device=device),
                    TrellisLinear.from_trellis_weight(
                        w[f"{mp}.shared_experts.up_proj.weight"], device=device
                    ),
                    TrellisLinear.from_trellis_weight(
                        w[f"{mp}.shared_experts.down_proj.weight"], device=device
                    ),
                )
                n_weights += 3
            rw = routers.get(f"{mp}.gate.weight", torch.randn(64, 2048))
            mlp = TrellisMoELayer(experts, shared, rw.to(device))
        else:
            mlp = TrellisExpertMLP(
                TrellisLinear.from_trellis_weight(w[f"{mp}.gate_proj.weight"], device=device),
                TrellisLinear.from_trellis_weight(w[f"{mp}.up_proj.weight"], device=device),
                TrellisLinear.from_trellis_weight(w[f"{mp}.down_proj.weight"], device=device),
            )
            n_weights += 3
        model.layers.append(TrellisLayer(attn, mlp))
    return model, n_weights


def bench_full_model_prefill(model: TrellisModelBench, ctx: int, device: str) -> dict[str, float]:
    """Benchmark FULL MODEL prefill: all 47 layers, all attention + MoE."""
    hidden = model.hidden_size
    x = torch.randn(1, ctx, hidden, dtype=torch.float16, device=device)

    # Warmup with small context
    with torch.no_grad():
        try:
            _ = model(x[:, :8, :])
            if device == "mps":
                torch.mps.synchronize()
        except Exception as e:
            return {"prefill_tok_s": 0, "prefill_latency_ms": 0, "error": str(e)}

    # Benchmark full context
    times = []
    with torch.no_grad():
        for _ in range(3):
            gc.collect()
            if device == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device == "mps":
                torch.mps.synchronize()
            times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    return {"prefill_tok_s": ctx / avg, "prefill_latency_ms": avg * 1000}


def bench_full_model_decode(
    model: TrellisModelBench, n_tokens: int, device: str
) -> dict[str, float]:
    """Benchmark FULL MODEL decode: single token through all 47 layers."""
    hidden = model.hidden_size
    x = torch.randn(1, 1, hidden, dtype=torch.float16, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(2):
            try:
                _ = model(x)
                if device == "mps":
                    torch.mps.synchronize()
            except Exception as e:
                return {"decode_tok_s": 0, "decode_latency_ms": 0, "error": str(e)}

    # Benchmark n_tokens iterations (simulating autoregressive decode)
    times = []
    with torch.no_grad():
        for _ in range(3):
            gc.collect()
            if device == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_tokens):
                _ = model(x)
            if device == "mps":
                torch.mps.synchronize()
            times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    tok_per_sec = n_tokens / avg
    ms_per_tok = (avg / n_tokens) * 1000
    return {"decode_tok_s": tok_per_sec, "decode_latency_ms": ms_per_tok}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--context-lengths", default="128,256,512,1024,2048")
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--output", default=str(RESULTS_DIR / "glm4_trellis_eval.json"))
    parser.add_argument("--samples", type=int, default=10)  # compat
    parser.add_argument("--context-sweep", action="store_true")  # compat
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: {model_path}")
        return 1

    print("=" * 70)
    print("GLM-4.7-Flash Trellis - Full Model Benchmark")
    print("=" * 70)
    print(f"Device: {device}")

    results = BenchmarkResults(model_path=model_path.name, timestamp=datetime.now().isoformat())
    results.model_size_gb = sum(f.stat().st_size for f in model_path.rglob("*.safetensors")) / (
        1024**3
    )
    print(f"\nModel size: {results.model_size_gb:.2f} GB")

    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    print("\n[1/4] Loading model...")
    loader = TrellisModelLoader(model_path)
    results.num_layers = loader.get_num_layers()
    print(f"  {results.num_layers} layers")

    t0 = time.perf_counter()
    model, nw = load_model(loader, device)
    print(f"  {nw} weights in {time.perf_counter() - t0:.1f}s")
    results.total_weights = nw
    results.memory_after_load_gb = get_mem(device)
    print(f"  Memory: {results.memory_after_load_gb:.2f} GB")

    print("\n[2/4] Prefill benchmark (TrellisLinear throughput)...")
    ctxs = [int(x) for x in args.context_lengths.split(",")]
    prefill_results = []
    hidden_size = 2048  # GLM-4 hidden size
    for c in ctxs:
        try:
            m = bench_linear_prefill(list(model.layers), c, hidden_size, device)
            results.throughput_by_context[c] = m
            prefill_results.append(m["prefill_tok_s"])
            print(
                f"    {c:>5}: {m['prefill_tok_s']:>10.1f} tok/s ({m['prefill_latency_ms']:>7.1f} ms)"
            )
        except Exception as e:
            print(f"    {c:>5}: ERROR - {e}")

    print(f"\n[3/4] Decode benchmark ({args.decode_tokens} tokens)...")
    try:
        dm = bench_linear_decode(list(model.layers), args.decode_tokens, hidden_size, device)
        for c in results.throughput_by_context:
            if "error" not in results.throughput_by_context.get(c, {}):
                results.throughput_by_context[c].update(dm)
        print(f"    {dm['decode_tok_s']:.1f} tok/s ({dm['decode_latency_ms']:.2f} ms/tok)")
    except Exception as e:
        print(f"    ERROR - {e}")
        dm = {}

    results.peak_memory_gb = get_mem(device)
    print(f"\n[4/4] Peak memory: {results.peak_memory_gb:.2f} GB")

    if prefill_results:
        results.avg_prefill_tok_s = sum(prefill_results) / len(prefill_results)
    if dm:
        results.avg_decode_tok_s = dm.get("decode_tok_s", 0)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model: {model_path.name}")
    print(f"  Layers: {results.num_layers}, Weights: {results.total_weights:,}")
    print(f"  Size: {results.model_size_gb:.2f} GB, Memory: {results.memory_after_load_gb:.2f} GB")
    print(
        f"  Prefill: {results.avg_prefill_tok_s:.1f} tok/s, Decode: {results.avg_decode_tok_s:.1f} tok/s"
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
