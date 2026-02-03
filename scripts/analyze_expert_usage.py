#!/usr/bin/env python3
"""Track and prune rarely-used experts by analyzing inference on multiple samples.

This script analyzes MoE routing patterns to identify experts that are rarely selected
and can potentially be pruned to reduce model size and memory usage.

Usage:
    uv run python scripts/analyze_expert_usage.py --model models/GLM-4.7-Flash-Trellis-3bpw --samples 1000
"""

import argparse
import json
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class ExpertTracker:
    """Tracks expert selection statistics across inference samples."""

    def __init__(self, num_layers: int, num_experts: int, top_k: int):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k = top_k

        # Per-layer expert selection counts
        # shape: [num_layers, num_experts]
        self.expert_counts = np.zeros((num_layers, num_experts), dtype=np.int64)

        # Total tokens processed per layer
        self.tokens_per_layer = np.zeros(num_layers, dtype=np.int64)

        # Total assignments (tokens * top_k) per layer
        self.assignments_per_layer = np.zeros(num_layers, dtype=np.int64)

    def record_routing(self, layer_idx: int, expert_ids: torch.Tensor, num_tokens: int) -> None:
        """Record expert selections for a layer.

        Args:
            layer_idx: Layer index (0 to num_layers-1)
            expert_ids: [batch, top_k] or [batch*seq, top_k] expert selections
            num_tokens: Number of tokens in the batch
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return

        # Flatten to [total_assignments]
        flat_expert_ids = expert_ids.reshape(-1).cpu().numpy()

        # Count selections per expert
        counts = np.bincount(flat_expert_ids, minlength=self.num_experts)
        self.expert_counts[layer_idx] += counts

        self.tokens_per_layer[layer_idx] += num_tokens
        self.assignments_per_layer[layer_idx] += len(flat_expert_ids)

    def get_summary(self) -> dict[str, Any]:
        """Generate summary statistics.

        Returns:
            Dictionary with analysis results.
        """
        results = {
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "total_tokens": int(self.tokens_per_layer.sum()),
            "total_assignments": int(self.assignments_per_layer.sum()),
        }

        # Per-layer statistics
        per_layer = []
        for layer_idx in range(self.num_layers):
            total_assignments = self.assignments_per_layer[layer_idx]
            if total_assignments == 0:
                continue

            avg_per_expert = total_assignments / self.num_experts

            # Compute activation rates
            rates = self.expert_counts[layer_idx] / total_assignments

            # Identify categories
            dead_experts = np.where(self.expert_counts[layer_idx] == 0)[0].tolist()
            rare_experts = np.where((rates > 0) & (rates < 0.001))[0].tolist()
            normal_experts = np.where((rates >= 0.001) & (rates < 0.01))[0].tolist()
            hot_experts = np.where(rates >= 0.01)[0].tolist()

            per_layer.append(
                {
                    "layer": layer_idx,
                    "tokens": int(self.tokens_per_layer[layer_idx]),
                    "assignments": int(total_assignments),
                    "dead_experts": dead_experts,
                    "dead_count": len(dead_experts),
                    "rare_experts": rare_experts,
                    "rare_count": len(rare_experts),
                    "normal_experts": normal_experts,
                    "normal_count": len(normal_experts),
                    "hot_experts": hot_experts,
                    "hot_count": len(hot_experts),
                }
            )

        results["per_layer"] = per_layer

        # Global pruning candidates (experts dead in ALL layers)
        if per_layer:
            dead_sets = [set(p["dead_experts"]) for p in per_layer]
            prunable = set(range(self.num_experts)).intersection(*dead_sets)
            results["prunable_experts"] = sorted(prunable)
            results["prunable_count"] = len(prunable)

            # Potential memory savings
            # Each expert has gate_up + down projections
            # Size estimation (adjust based on actual model size)
            avg_hidden = 2048
            avg_intermediate = 1536
            bits_per_param = 3  # trellis quantization
            bytes_per_expert = 3 * avg_intermediate * avg_hidden * bits_per_param / 8
            results["potential_memory_savings_mb"] = (
                len(prunable) * bytes_per_expert / (1024 * 1024)
            )

        return results

    def print_report(self) -> None:
        """Print a human-readable report."""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print("EXPERT USAGE ANALYSIS REPORT")
        print("=" * 70)

        print("\nConfiguration:")
        print(f"  Experts per layer: {summary['num_experts']}")
        print(f"  Layers: {summary['num_layers']}")
        print(f"  Top-k: {summary['top_k']}")
        print(f"  Total tokens processed: {summary['total_tokens']:,}")
        print(f"  Total assignments: {summary['total_assignments']:,}")

        print("\nExpert Categories:")
        print("  Dead: Never selected")
        print("  Rare: Selected in <0.1% of assignments")
        print("  Normal: 0.1% - 1% of assignments")
        print("  Hot: >1% of assignments")

        for layer_info in summary["per_layer"]:
            print(f"\nLayer {layer_info['layer']}:")
            print(f"  Tokens: {layer_info['tokens']:,}")
            print(f"  Dead experts: {layer_info['dead_count']}")
            print(f"  Rare experts: {layer_info['rare_count']}")
            print(f"  Normal experts: {layer_info['normal_count']}")
            print(f"  Hot experts: {layer_info['hot_count']}")

        if summary.get("prunable_count", 0) > 0:
            print("\n" + "=" * 70)
            print("PRUNING RECOMMENDATIONS")
            print("=" * 70)
            print(f"\nPrunable experts (dead in ALL layers): {summary['prunable_count']}")
            print(f"Expert IDs: {summary['prunable_experts']}")
            print(
                f"Potential memory savings: {summary.get('potential_memory_savings_mb', 0):.2f} MB"
            )
            print("\nThese experts can be safely removed without affecting inference quality.")
        else:
            print("\n" + "=" * 70)
            print("NO PRUNING CANDIDATES")
            print("=" * 70)
            print("\nAll experts were selected during analysis.")


def register_hooks(model: torch.nn.Module, tracker: ExpertTracker) -> list:
    """Register forward hooks to track expert selections.

    Args:
        model: The model to track
        tracker: ExpertTracker instance

    Returns:
        List of hook handles (for cleanup)
    """
    handles = []

    # Find all MoE layers and attach hooks
    for name, module in model.named_modules():
        if "moe" in name.lower() and hasattr(module, "forward"):
            # Try to extract layer index from name
            try:
                layer_idx = int(name.split(".layers.")[1].split(".")[0])
            except (IndexError, ValueError):
                continue

            # Create a hook that captures expert_ids
            def make_hook(layer_idx: int):
                def hook(module, args, kwargs):
                    # The MoE forward pass computes expert_ids
                    # We need to intercept after routing but before expert execution
                    input_tensor = args[0]

                    # Call the original forward
                    # Get expert_ids by calling router directly
                    if hasattr(module, "router"):
                        with torch.no_grad():
                            router_logits = input_tensor @ module.router.weight.T
                            routing_weights = F.softmax(router_logits, dim=-1)
                            expert_weights, expert_ids = torch.topk(
                                routing_weights,
                                module.config.num_experts_per_tok,
                                dim=-1,
                                sorted=True,
                            )
                            num_tokens = (
                                input_tensor.shape[0]
                                if input_tensor.dim() == 2
                                else input_tensor.shape[0] * input_tensor.shape[1]
                            )
                            if input_tensor.dim() == 3:
                                num_tokens = input_tensor.shape[0] * input_tensor.shape[1]

                            tracker.record_routing(layer_idx, expert_ids, num_tokens)

                    return None

                return hook

            # Pre-forward hook to intercept before MoE processes
            handles.append(module.register_forward_pre_hook(make_hook(layer_idx)))

    return handles


def run_inference(
    model: torch.nn.Module,
    tracker: ExpertTracker,
    tokenizer,
    prompt: str = "The quick brown fox jumps over the lazy dog.",
    max_new_tokens: int = 50,
    num_samples: int = 100,
) -> None:
    """Run inference on multiple samples to track expert usage.

    Args:
        model: The model to run inference on
        tracker: ExpertTracker instance
        tokenizer: Tokenizer instance
        prompt: Input prompt for generation
        max_new_tokens: Maximum tokens to generate
        num_samples: Number of samples to run
    """
    print(f"\nRunning inference on {num_samples} samples...")
    print(f"Prompt: {prompt}")
    print(f"Generating {max_new_tokens} tokens per sample")

    # Register hooks
    handles = register_hooks(model, tracker)

    try:
        # Run multiple samples
        for i in range(num_samples):
            if i > 0 and i % 10 == 0:
                print(f"  Completed {i}/{num_samples} samples...")

            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )

    finally:
        # Remove hooks
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze expert usage and identify pruning candidates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    # Analyze with 1000 samples
    uv run python scripts/analyze_expert_usage.py --model models/GLM-4.7-Flash-Trellis-3bpw --samples 1000

    # Quick test with 100 samples
    uv run python scripts/analyze_expert_usage.py --model models/GLM-4.7-Flash-Trellis-3bpw --samples 100

    # Save results to file
    uv run python scripts/analyze_expert_usage.py --model models/GLM-4.7-Flash-Trellis-3bpw --samples 1000 --output results.json
""",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/GLM-4.7-Flash-Trellis-3bpw",
        help="Path to model directory",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of inference samples to run (default: 1000)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate per sample (default: 50)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox jumps over the lazy dog.",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to run on (default: mps)",
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    from transformers import AutoTokenizer

    from metal_marlin.trellis.model import TrellisForCausalLM

    model = TrellisForCausalLM.from_pretrained(args.model, device=args.device)
    model.eval()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get MoE configuration
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        first_layer = model.model.layers[0]
        if hasattr(first_layer, "mlp"):
            moe = first_layer.mlp
            num_experts = len(moe.experts) if hasattr(moe, "experts") else 64
            top_k = moe.num_experts_per_tok if hasattr(moe, "num_experts_per_tok") else 2
        else:
            num_experts = 64
            top_k = 2
    else:
        num_experts = 64
        top_k = 2

    num_layers = len(model.model.layers)

    print("MoE Configuration:")
    print(f"  Experts: {num_experts}")
    print(f"  Top-k: {top_k}")
    print(f"  Layers: {num_layers}")

    # Create tracker
    tracker = ExpertTracker(num_layers, num_experts, top_k)

    # Run inference
    run_inference(
        model,
        tracker,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        num_samples=args.samples,
    )

    # Print report
    tracker.print_report()

    # Save results if requested
    if args.output:
        results = tracker.get_summary()
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
