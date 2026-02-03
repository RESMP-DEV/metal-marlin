#!/usr/bin/env python3
"""Profile MoE layer components: router, dispatch, experts, combine."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from metal_marlin.trellis.lm import TrellisForCausalLM
from metal_marlin.trellis.model import TrellisMoEMLP


def profile_moe_components(
    model: TrellisForCausalLM,
    hidden_states: torch.Tensor,
    num_warmup: int = 3,
    num_iters: int = 10,
) -> dict[str, float]:
    """Profile individual MoE components.

    Args:
        model: Loaded TrellisForCausalLM model.
        hidden_states: Input hidden states [batch, seq, hidden].
        num_warmup: Number of warmup iterations.
        num_iters: Number of profiling iterations.

    Returns:
        Dictionary mapping component names to average time in ms.
    """
    results: dict[str, list[float]] = {
        "router": [],
        "topk_dispatch": [],
        "experts_kernel": [],
        "shared_expert": [],
        "total_moe": [],
    }

    # Find first MoE layer
    moe_mlp: TrellisMoEMLP | None = None
    moe_layer_idx = -1
    for idx, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, TrellisMoEMLP):
            moe_mlp = layer.mlp
            moe_layer_idx = idx
            break

    if moe_mlp is None:
        print("No MoE layer found!")
        return {}

    print(f"Profiling MoE layer {moe_layer_idx}")
    print(f"  Num experts: {len(moe_mlp.experts)}")
    print(f"  Top-k: {moe_mlp.num_experts_per_tok}")
    print(f"  Hidden dim: {moe_mlp.hidden_dim}")
    print(f"  Intermediate dim: {moe_mlp.intermediate_dim}")
    print()

    # Flatten input for MoE
    batch_shape = hidden_states.shape[:-1]
    hidden_dim = hidden_states.shape[-1]
    x_flat = hidden_states.reshape(-1, hidden_dim).half()
    num_tokens = x_flat.shape[0]

    # Get Metal library and buffers
    lib = moe_mlp._get_lib()
    cached_buffers = moe_mlp._get_cached_buffers()
    buffer_pool = moe_mlp._get_buffer_pool()

    # Import dispatch function
    from metal_marlin.trellis.moe_dispatch import dispatch_moe_trellis_swiglu

    # Warmup
    print(f"Warming up ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = moe_mlp(hidden_states)
        torch.mps.synchronize()

    print(f"Profiling ({num_iters} iterations)...")

    for i in range(num_iters):
        # --- 1. Profile Router ---
        torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            router_logits = moe_mlp.router(x_flat)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        results["router"].append((t1 - t0) * 1000)

        # --- 2. Profile Top-K Dispatch (softmax + topk + normalize) ---
        torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            routing_weights, selected_experts = torch.topk(
                F.softmax(router_logits, dim=-1, dtype=torch.float16),
                k=moe_mlp.num_experts_per_tok,
                dim=-1,
            )
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        results["topk_dispatch"].append((t1 - t0) * 1000)

        # --- 3. Profile Experts Kernel (the main MoE dispatch) ---
        torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            moe_output = dispatch_moe_trellis_swiglu(
                lib=lib,
                activations=x_flat,
                gate_weights=None,
                gate_scales=None,
                up_weights=None,
                up_scales=None,
                down_weights=None,
                down_scales=None,
                gate_su=None,
                gate_sv=None,
                up_su=None,
                up_sv=None,
                down_su=None,
                down_sv=None,
                grid=None,
                expert_ids=selected_experts,
                expert_probs=routing_weights,
                hidden_dim=moe_mlp.hidden_dim,
                intermediate_dim=moe_mlp.intermediate_dim,
                num_experts=len(moe_mlp.experts),
                top_k=moe_mlp.num_experts_per_tok,
                bits=moe_mlp.bits,
                cached_buffers=cached_buffers,
                buffer_pool=buffer_pool,
                use_fp32_acc=moe_mlp.hidden_dim >= 1024,
            )
        torch.mps.synchronize()
        t1 = time.perf_counter()
        results["experts_kernel"].append((t1 - t0) * 1000)

        # --- 4. Profile Shared Expert ---
        torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            shared_out = moe_mlp.shared_expert(hidden_states)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        results["shared_expert"].append((t1 - t0) * 1000)

        # --- 5. Profile Full MoE (for reference) ---
        torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = moe_mlp(hidden_states)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        results["total_moe"].append((t1 - t0) * 1000)

    # Compute averages
    avg_results = {}
    for key, times in results.items():
        avg_results[key] = sum(times) / len(times)

    return avg_results


def main():
    import sys

    model_path = "models/GLM-4.7-Flash-Trellis-3bpw"
    seq_len = 128

    # Check for command line args
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        seq_len = int(sys.argv[2])

    print(f"Loading model from: {model_path}")
    print(f"Sequence length: {seq_len}")
    print("=" * 60)

    try:
        model = TrellisForCausalLM.from_pretrained(model_path, device="mps")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("\nUsage: python profile_moe_detailed.py [model_path] [seq_len]")
        return

    # Create input
    x = torch.randint(0, 1000, (1, seq_len)).to("mps")

    # Get hidden states by running through embedding + first layers
    print("\nGenerating hidden states...")
    with torch.no_grad():
        hidden_states = model.model.embed_tokens(x)
        # Run through dense layers (layer 0, maybe 1)
        for idx, layer in enumerate(model.model.layers):
            if isinstance(layer.mlp, TrellisMoEMLP):
                break
            hidden_states = layer(hidden_states)[0] if isinstance(layer(hidden_states), tuple) else layer(hidden_states)

    print(f"Hidden states shape: {hidden_states.shape}")
    print()

    # Profile components
    results = profile_moe_components(model, hidden_states, num_warmup=5, num_iters=20)

    if not results:
        return

    # Print results
    print("\n" + "=" * 60)
    print("MoE Component Timing (average over 20 iterations)")
    print("=" * 60)

    total_components = (
        results["router"]
        + results["topk_dispatch"]
        + results["experts_kernel"]
        + results["shared_expert"]
    )

    print(f"\n{'Component':<20} {'Time (ms)':<12} {'% of Total':<12}")
    print("-" * 44)

    for name in ["router", "topk_dispatch", "experts_kernel", "shared_expert"]:
        time_ms = results[name]
        pct = (time_ms / total_components) * 100 if total_components > 0 else 0
        print(f"{name:<20} {time_ms:<12.3f} {pct:<12.1f}%")

    print("-" * 44)
    print(f"{'Sum of components':<20} {total_components:<12.3f} 100.0%")
    print(f"{'Measured total':<20} {results['total_moe']:<12.3f}")

    overhead = results["total_moe"] - total_components
    print(f"\nOverhead (coordination): {overhead:.3f}ms ({overhead/results['total_moe']*100:.1f}%)")

    # Identify bottleneck
    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)

    max_component = max(
        [("router", results["router"]),
         ("topk_dispatch", results["topk_dispatch"]),
         ("experts_kernel", results["experts_kernel"]),
         ("shared_expert", results["shared_expert"])],
        key=lambda x: x[1]
    )

    print(f"\nBottleneck: {max_component[0]} ({max_component[1]:.3f}ms, "
          f"{max_component[1]/total_components*100:.1f}% of component time)")

    # Per-expert cost estimate
    num_experts = len(model.model.layers[2].mlp.experts) if hasattr(model.model.layers[2].mlp, 'experts') else 64
    top_k = 8
    print(f"\nExpert kernel ({num_experts} experts, top-{top_k}):")
    print(f"  Total kernel time: {results['experts_kernel']:.3f}ms")
    print(f"  Per-active-expert: {results['experts_kernel']/top_k:.3f}ms")


if __name__ == "__main__":
    main()
