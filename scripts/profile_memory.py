#!/usr/bin/env python3
"""Profile memory allocation during forward pass to identify leak source.

Profiles layer-by-layer memory consumption during a single token forward pass
through a Trellis model to identify which layers cause memory jumps.

Usage:
    cd contrib/metal_marlin && uv run python scripts/profile_memory.py

    # With specific model path
    uv run python scripts/profile_memory.py --model models/GLM-4.7-Flash-EXL3-3bpw

    # With custom sequence length
    uv run python scripts/profile_memory.py --seq-len 128
"""

from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from metal_marlin.trellis.model import TrellisForCausalLM


@dataclass
class MemorySnapshot:
    """Memory snapshot at a point in execution."""

    stage: str
    allocated_mb: float
    delta_mb: float


def get_mps_memory_mb() -> float:
    """Get current MPS allocated memory in MB."""
    if not torch.backends.mps.is_available():
        return 0.0
    torch.mps.synchronize()
    return torch.mps.current_allocated_memory() / (1024 * 1024)


def force_gc() -> None:
    """Force garbage collection and MPS synchronization."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


def profile_model_load(model_path: str) -> tuple[TrellisForCausalLM, list[MemorySnapshot]]:
    """Load model and profile memory during load.

    Args:
        model_path: Path to the Trellis model directory.

    Returns:
        Tuple of (loaded model, list of memory snapshots).
    """
    from metal_marlin.trellis.model import TrellisForCausalLM

    snapshots: list[MemorySnapshot] = []
    prev_mem = get_mps_memory_mb()

    snapshots.append(MemorySnapshot("before_load", prev_mem, 0.0))

    model = TrellisForCausalLM.from_pretrained(model_path, device="mps")

    force_gc()
    curr_mem = get_mps_memory_mb()
    snapshots.append(MemorySnapshot("after_load", curr_mem, curr_mem - prev_mem))

    return model, snapshots


def profile_layer_forward(
    model: TrellisForCausalLM,
    input_ids: torch.Tensor,
    verbose: bool = True,
) -> list[MemorySnapshot]:
    """Profile memory during layer-by-layer forward pass.

    Manually steps through each decoder layer, logging memory before and after.

    Args:
        model: Loaded TrellisForCausalLM model.
        input_ids: Input token IDs [batch, seq_len].
        verbose: If True, print memory info during profiling.

    Returns:
        List of memory snapshots, one per layer plus embedding/norm stages.
    """
    snapshots: list[MemorySnapshot] = []
    device = input_ids.device

    force_gc()
    prev_mem = get_mps_memory_mb()
    snapshots.append(MemorySnapshot("before_forward", prev_mem, 0.0))

    # Embedding
    hidden_states = model.model.embed_tokens(input_ids)
    force_gc()
    curr_mem = get_mps_memory_mb()
    delta = curr_mem - prev_mem
    snapshots.append(MemorySnapshot("after_embed", curr_mem, delta))
    if verbose:
        print(f"  embed_tokens:  {curr_mem:8.2f} MB (delta: {delta:+8.2f} MB)")
    prev_mem = curr_mem

    # Create causal mask
    seq_len = input_ids.shape[1]
    attention_mask = model.model._make_causal_mask(seq_len, device)

    # Forward through each layer
    num_layers = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):
        # Determine if MoE or dense
        is_moe = model.config.is_moe_layer(i)
        layer_type = "MoE" if is_moe else "Dense"

        # Forward through layer
        hidden_states = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=None,
            kv_cache=None,
        )

        # Sync and measure
        force_gc()
        curr_mem = get_mps_memory_mb()
        delta = curr_mem - prev_mem

        stage_name = f"layer_{i:02d}_{layer_type}"
        snapshots.append(MemorySnapshot(stage_name, curr_mem, delta))

        if verbose:
            flag = "***" if abs(delta) > 100 else "   "
            print(f"  {stage_name:20s}: {curr_mem:8.2f} MB (delta: {delta:+8.2f} MB) {flag}")

        prev_mem = curr_mem

        # Sync every 8 layers as the model does
        if (i + 1) % 8 == 0:
            torch.mps.synchronize()

    # Final norm
    hidden_states = model.model.norm(hidden_states)
    force_gc()
    curr_mem = get_mps_memory_mb()
    delta = curr_mem - prev_mem
    snapshots.append(MemorySnapshot("after_norm", curr_mem, delta))
    if verbose:
        print(f"  final_norm:    {curr_mem:8.2f} MB (delta: {delta:+8.2f} MB)")
    prev_mem = curr_mem

    # LM head
    logits = model.lm_head(hidden_states)
    force_gc()
    curr_mem = get_mps_memory_mb()
    delta = curr_mem - prev_mem
    snapshots.append(MemorySnapshot("after_lm_head", curr_mem, delta))
    if verbose:
        print(f"  lm_head:       {curr_mem:8.2f} MB (delta: {delta:+8.2f} MB)")

    # Cleanup intermediate tensors
    del hidden_states, logits, attention_mask
    force_gc()
    curr_mem = get_mps_memory_mb()
    delta = curr_mem - prev_mem
    snapshots.append(MemorySnapshot("after_cleanup", curr_mem, delta))
    if verbose:
        print(f"  after_cleanup: {curr_mem:8.2f} MB (delta: {delta:+8.2f} MB)")

    return snapshots


def profile_moe_detailed(
    model: TrellisForCausalLM,
    layer_idx: int,
    input_ids: torch.Tensor,
    verbose: bool = True,
) -> list[MemorySnapshot]:
    """Profile memory within a single MoE layer in detail.

    Breaks down the MoE forward pass into router, expert dispatch, and shared expert.

    Args:
        model: Loaded TrellisForCausalLM model.
        layer_idx: Index of the MoE layer to profile.
        input_ids: Input token IDs [batch, seq_len].
        verbose: If True, print memory info during profiling.

    Returns:
        List of memory snapshots for sub-operations within the MoE layer.
    """
    import torch.nn.functional as F

    from metal_marlin.trellis.model import TrellisMoEMLP

    snapshots: list[MemorySnapshot] = []

    # Get hidden states up to the target layer
    hidden_states = model.model.embed_tokens(input_ids)
    for i in range(layer_idx):
        hidden_states = model.model.layers[i](hidden_states)

    layer = model.model.layers[layer_idx]
    if not isinstance(layer.mlp, TrellisMoEMLP):
        print(f"Layer {layer_idx} is not MoE, skipping detailed profile")
        return snapshots

    # Pre-attention norm and attention
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    attn_output = layer.self_attn(hidden_states, kv_cache=None, layer_idx=layer_idx)
    hidden_states = residual + attn_output

    # Post-attention norm
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)

    force_gc()
    prev_mem = get_mps_memory_mb()
    snapshots.append(MemorySnapshot("before_moe", prev_mem, 0.0))

    moe = layer.mlp
    x = hidden_states

    # Router computation
    x_flat = x.view(-1, moe.hidden_dim)
    x_router = x_flat.to(moe.router.weight.dtype)
    router_logits = moe.router(x_router)

    force_gc()
    curr_mem = get_mps_memory_mb()
    delta = curr_mem - prev_mem
    snapshots.append(MemorySnapshot("after_router", curr_mem, delta))
    if verbose:
        print(f"    router:        {curr_mem:8.2f} MB (delta: {delta:+8.2f} MB)")
    prev_mem = curr_mem

    # Top-k selection
    routing_weights, selected_experts = torch.topk(
        F.softmax(router_logits, dim=-1, dtype=torch.float),
        k=moe.num_experts_per_tok,
        dim=-1,
    )
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

    force_gc()
    curr_mem = get_mps_memory_mb()
    delta = curr_mem - prev_mem
    snapshots.append(MemorySnapshot("after_topk", curr_mem, delta))
    if verbose:
        print(f"    topk:          {curr_mem:8.2f} MB (delta: {delta:+8.2f} MB)")
    prev_mem = curr_mem

    # Check which path will be used
    uses_fast = moe._use_fast_moe and x.is_mps
    path_name = "fast_dispatch" if uses_fast else "slow_dispatch"

    if uses_fast:
        # Fast fused dispatch
        from metal_marlin.trellis.moe_dispatch import dispatch_moe_trellis_swiglu

        output = dispatch_moe_trellis_swiglu(
            lib=moe._get_lib(),
            activations=x_flat,
            gate_weights=moe.gate_weights_stacked,
            gate_scales=moe.gate_scales_stacked,
            up_weights=moe.up_weights_stacked,
            up_scales=moe.up_scales_stacked,
            down_weights=moe.down_weights_stacked,
            down_scales=moe.down_scales_stacked,
            gate_su=moe.gate_su_stacked,
            gate_sv=moe.gate_sv_stacked,
            up_su=moe.up_su_stacked,
            up_sv=moe.up_sv_stacked,
            down_su=moe.down_su_stacked,
            down_sv=moe.down_sv_stacked,
            grid=moe.experts[0].gate_proj.grid,
            expert_ids=selected_experts,
            expert_probs=routing_weights,
            hidden_dim=moe.hidden_dim,
            intermediate_dim=moe.intermediate_dim,
            num_experts=len(moe.experts),
            top_k=moe.num_experts_per_tok,
            bits=moe.bits,
            cached_buffers=moe._get_cached_buffers(),
        )
    else:
        # Slow sequential path - profile each unique expert
        unique_experts = selected_experts.unique().tolist()
        output = torch.zeros_like(x)

        for j, expert_id in enumerate(unique_experts):
            expert_mask = selected_experts == expert_id
            weights_for_expert = torch.where(
                expert_mask,
                routing_weights,
                torch.zeros_like(routing_weights),
            ).sum(dim=-1)

            expert_output = moe.experts[expert_id](x)
            output += expert_output * weights_for_expert.unsqueeze(-1)

            del expert_output, weights_for_expert, expert_mask

            if j % 8 == 0:
                force_gc()
                curr_mem = get_mps_memory_mb()
                delta = curr_mem - prev_mem
                if verbose and abs(delta) > 10:
                    print(
                        f"    expert_{j:02d}:     {curr_mem:8.2f} MB (delta: {delta:+8.2f} MB)"
                    )
                prev_mem = curr_mem

    force_gc()
    curr_mem = get_mps_memory_mb()
    delta = curr_mem - prev_mem
    snapshots.append(MemorySnapshot(f"after_{path_name}", curr_mem, delta))
    if verbose:
        print(f"    {path_name:14s}: {curr_mem:8.2f} MB (delta: {delta:+8.2f} MB)")
    prev_mem = curr_mem

    # Shared expert
    shared_output = moe.shared_expert(x)
    output = output + shared_output

    force_gc()
    curr_mem = get_mps_memory_mb()
    delta = curr_mem - prev_mem
    snapshots.append(MemorySnapshot("after_shared", curr_mem, delta))
    if verbose:
        print(f"    shared_expert: {curr_mem:8.2f} MB (delta: {delta:+8.2f} MB)")

    return snapshots


def analyze_snapshots(snapshots: list[MemorySnapshot]) -> None:
    """Print analysis of memory snapshots.

    Args:
        snapshots: List of memory snapshots from profiling.
    """
    if not snapshots:
        return

    print("\n" + "=" * 60)
    print("MEMORY ANALYSIS")
    print("=" * 60)

    # Find largest memory jumps
    jumps = [(s.stage, s.delta_mb) for s in snapshots if s.delta_mb > 50]
    jumps.sort(key=lambda x: x[1], reverse=True)

    if jumps:
        print("\nLargest memory jumps (>50 MB):")
        for stage, delta in jumps[:10]:
            print(f"  {stage:30s}: {delta:+8.2f} MB")

    # Final vs expected
    final_mem = snapshots[-1].allocated_mb
    initial_mem = snapshots[0].allocated_mb

    print(f"\nInitial memory:  {initial_mem:8.2f} MB")
    print(f"Final memory:    {final_mem:8.2f} MB")
    print(f"Net increase:    {final_mem - initial_mem:+8.2f} MB")

    # Check against expectations
    expected_base = 4000  # ~4GB after load
    if final_mem > expected_base * 2:
        print(f"\nWARNING: Final memory ({final_mem:.0f} MB) exceeds 2x expected base ({expected_base} MB)")
        print("This suggests a memory leak or excessive allocation during forward pass.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile memory allocation during Trellis model forward pass"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/GLM-4.7-Flash-EXL3-3bpw",
        help="Path to Trellis model directory",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1,
        help="Sequence length for forward pass (default: 1 for single token)",
    )
    parser.add_argument(
        "--profile-moe",
        type=int,
        default=None,
        help="Also profile MoE layer N in detail (e.g., --profile-moe 2)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary, not per-layer stats",
    )
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available. This script requires Apple Silicon.")
        return 1

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model path does not exist: {model_path}")
        return 1

    verbose = not args.quiet

    print(f"Model: {model_path}")
    print(f"Sequence length: {args.seq_len}")
    print()

    # Load model
    print("Loading model...")
    model, load_snapshots = profile_model_load(str(model_path))

    if verbose:
        print(f"  Model loaded: {load_snapshots[-1].allocated_mb:.2f} MB")
        print(f"  Layers: {len(model.model.layers)}")
        print(f"  MoE layers: {sum(1 for i in range(len(model.model.layers)) if model.config.is_moe_layer(i))}")

    # Create input
    input_ids = torch.randint(0, model.config.vocab_size, (1, args.seq_len), device="mps")

    # Profile forward pass
    print("\nProfiling forward pass...")
    forward_snapshots = profile_layer_forward(model, input_ids, verbose=verbose)

    # Optionally profile specific MoE layer
    moe_snapshots: list[MemorySnapshot] = []
    if args.profile_moe is not None:
        layer_idx = args.profile_moe
        if 0 <= layer_idx < len(model.model.layers) and model.config.is_moe_layer(layer_idx):
            print(f"\nProfiling MoE layer {layer_idx} in detail...")
            moe_snapshots = profile_moe_detailed(model, layer_idx, input_ids, verbose=verbose)
        else:
            print(f"\nWARNING: Layer {layer_idx} is not a valid MoE layer")

    # Analysis
    all_snapshots = load_snapshots + forward_snapshots + moe_snapshots
    analyze_snapshots(all_snapshots)

    return 0


if __name__ == "__main__":
    sys.exit(main())
