#!/usr/bin/env python3
"""
Batch Metal Kernel Optimization (AlphaHENG-powered)

Runs entropy-first optimization across ALL Metal shaders in the project,
using DIVERSE problem sizes from multiple LLM architectures to find
configurations that generalize well.

Design Philosophy:
==================
Instead of optimizing for a specific model (which creates overfitted shaders),
we optimize for BROAD COVERAGE:

1. Problem sizes are drawn from Qwen3-4B/32B, Qwen3-30B-A3B, GLM-4.7-Flash, DeepSeek-V3.2
2. Optimization scores configurations by WORST-CASE across the diversity
3. Shaders should work well for N=768 (Qwen MoE) AND N=25600 (Qwen3-32B MLP)
4. Flags structural issues when a shader can't handle certain size classes

This prevents:
- Creating N shader variants for N models
- Overfitting tile parameters to one dimension
- Discovering "optimal" configs that only work for benchmarks

Usage:
    # Optimize all kernels with diverse problem sizes (default)
    cd contrib/metal_marlin && uv run python scripts/optimize_all_kernels.py

    # Focus on MoE kernels (tests N=768, N=1536, N=4096 together)
    cd contrib/metal_marlin && uv run python scripts/optimize_all_kernels.py --category moe

    # Higher entropy for thorough search
    cd contrib/metal_marlin && uv run python scripts/optimize_all_kernels.py --num-random 50

    # Use a specific benchmark profile for spawned optimize_kernel runs
    cd contrib/metal_marlin && uv run python scripts/optimize_all_kernels.py --profile mixed_bpw_fairway_glm47

    # List available kernels without running
    cd contrib/metal_marlin && uv run python scripts/optimize_all_kernels.py --list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
METAL_MARLIN_ROOT = _SCRIPT_DIR.parent
SRC_DIR = METAL_MARLIN_ROOT / "src"
FUSION_DIR = SRC_DIR / "fusion"

# ============================================================================
# MODEL ARCHITECTURES (reference, for documentation)
# ============================================================================
# These are used to DERIVE the diverse problem set, not to select one model.
# All configs from HuggingFace as of 2026-01.

MODEL_ARCHITECTURES = {
    # === DENSE MODELS ===
    "qwen3-4b": {
        "type": "dense",
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_attention_heads": 32,
        "num_kv_heads": 8,  # GQA 4:1
        "head_dim": 128,
    },
    "qwen3-32b": {
        "type": "dense",
        "hidden_size": 5120,
        "intermediate_size": 25600,
        "num_attention_heads": 64,
        "num_kv_heads": 8,  # GQA 8:1
        "head_dim": 128,
    },
    # === MoE MODELS ===
    "qwen3-30b-a3b": {
        "type": "moe",
        "hidden_size": 2048,
        "moe_intermediate_size": 768,  # VERY small!
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_attention_heads": 32,
        "num_kv_heads": 4,  # GQA 8:1
        "head_dim": 128,
    },
    "glm47-flash": {
        "type": "moe+mla",
        "hidden_size": 2048,
        "moe_intermediate_size": 1536,
        "dense_intermediate_size": 10240,
        "num_experts": 64,
        "num_experts_per_tok": 4,
        "kv_lora_rank": 512,
        "q_lora_rank": 768,
        "num_attention_heads": 20,
        "v_head_dim": 256,
    },
    "deepseek-v3.2": {
        "type": "moe+mla+dsa",  # MLA + Differentially Sparse Attention
        "hidden_size": 7168,
        "moe_intermediate_size": 2048,
        "dense_intermediate_size": 18432,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "num_attention_heads": 128,
        "num_kv_heads": 128,  # MLA, not GQA
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        # DSA (Differentially Sparse Attention) - NEW in V3.2
        "index_head_dim": 128,
        "index_n_heads": 64,
        "index_topk": 2048,
    },
}

# ============================================================================
# UNIFIED DIVERSE PROBLEM SIZES
# ============================================================================
# These combine shapes from ALL target architectures to ensure shaders
# generalize across the full operating envelope.

# GEMM: (M, N, K) covering decode→prefill, small→large dimensions
DIVERSE_GEMM_SIZES = [
    # === DECODE (M=1-32, memory-bound) ===
    # Qwen3-4B dense MLP
    (1, 9728, 2560),  # up proj
    (1, 2560, 9728),  # down proj
    # Qwen3-32B dense MLP
    (1, 25600, 5120),  # up proj (LARGE)
    (1, 5120, 25600),  # down proj
    # MoE expert shapes (small N)
    (1, 768, 2048),  # Qwen3-30B MoE (VERY small N)
    (1, 1536, 2048),  # GLM-4.7 MoE
    (1, 2048, 7168),  # DeepSeek-V3.2 MoE
    # Batched decode
    (8, 9728, 2560),  # Qwen3-4B batched
    (8, 768, 2048),  # Qwen3-30B MoE batched
    (32, 2048, 7168),  # DeepSeek-V3.2 batched
    # === PREFILL (M=128-2048, compute-bound) ===
    # Small N (MoE experts)
    (128, 768, 2048),  # Qwen3-30B MoE
    (256, 1536, 2048),  # GLM-4.7 MoE
    (256, 2048, 7168),  # DeepSeek-V3.2 MoE
    # Medium N (dense models)
    (256, 9728, 2560),  # Qwen3-4B MLP
    (512, 9728, 2560),  # Qwen3-4B long
    # Large N (Qwen3-32B)
    (256, 25600, 5120),  # Qwen3-32B MLP
    # MLA projections
    (256, 512, 2048),  # kv_lora (GLM/DeepSeek)
    (256, 1536, 7168),  # q_lora (DeepSeek)
    # Very long context
    (2048, 9728, 2560),  # Qwen3-4B max context
]

# MoE Expert GEMM: specifically tests the small-N regime
MOE_DIVERSE_SIZES = [
    # Qwen3-30B-A3B (N=768, extremely small!)
    (1, 768, 2048),
    (8, 768, 2048),
    (32, 768, 2048),
    (256, 768, 2048),
    # GLM-4.7-Flash (N=1536)
    (1, 1536, 2048),
    (32, 1536, 2048),
    (256, 1536, 2048),
    # DeepSeek-V3.2 (N=2048, hidden=7168)
    (1, 2048, 7168),
    (32, 2048, 7168),
    (256, 2048, 7168),
    # Down projections (transposed)
    (1, 2048, 768),  # Qwen3-30B down
    (1, 2048, 1536),  # GLM-4.7 down
    (1, 7168, 2048),  # DeepSeek-V3.2 down
    (256, 2048, 768),
    (256, 7168, 2048),
]

# Attention: (batch, seq_len, num_heads, head_dim)
DIVERSE_ATTENTION_SIZES = [
    # Standard GQA (head_dim=128)
    (1, 1, 32, 128),  # Qwen3-4B decode
    (8, 1, 32, 128),  # batched decode
    (1, 512, 32, 128),  # prefill
    (1, 2048, 32, 128),  # long prefill
    (1, 8192, 32, 128),  # very long (40K context)
    # Qwen3-32B (64 Q heads, 8 KV heads)
    (1, 1, 64, 128),  # decode
    (1, 512, 64, 128),  # prefill
    # GLM-4.7 MLA (v_head_dim=256)
    (1, 1, 20, 256),  # decode
    (1, 512, 20, 256),  # prefill
    (1, 2048, 20, 256),  # long
    # DeepSeek-V3.2 MLA (128 heads, v_head_dim=128)
    (1, 1, 128, 128),  # decode
    (1, 512, 128, 128),  # prefill
    (1, 8192, 128, 128),  # very long (163K context)
    # DeepSeek-V3.2 DSA (Differentially Sparse Attention)
    # index_n_heads=64, index_head_dim=128, index_topk=2048
    (1, 2048, 64, 128),  # DSA index computation
    (1, 8192, 64, 128),  # DSA long context (sparse selection of topk=2048)
    # Few KV heads (extreme GQA)
    (1, 1, 4, 128),  # Qwen3-30B KV (4 heads)
    (1, 512, 4, 128),
]

# Latent projections (MLA: GLM-4.7-Flash and DeepSeek-V3.2)
PROJECTION_SIZES = [
    # GLM-4.7-Flash
    (1, 512, 2048),  # kv_lora decode
    (1, 768, 2048),  # q_lora decode
    (128, 512, 2048),  # kv_lora prefill
    (128, 768, 2048),  # q_lora prefill
    # DeepSeek-V3.2 (larger ranks)
    (1, 512, 7168),  # kv_lora decode
    (1, 1536, 7168),  # q_lora decode
    (256, 512, 7168),  # kv_lora prefill
    (256, 1536, 7168),  # q_lora prefill
    # Long context
    (2048, 512, 2048),  # GLM long
    (2048, 512, 7168),  # DeepSeek long
]

# ============================================================================
# KERNEL CATEGORIZATION
# ============================================================================


def categorize_kernel(kernel_path: Path) -> str:
    """Categorize a kernel by its type."""
    name = kernel_path.stem.lower()

    if "attention" in name or "flash" in name or "mla" in name:
        return "attention"
    if "moe" in name:
        return "moe"
    if "gemm" in name or "marlin" in name or "dense" in name:
        return "gemm"
    if "gemv" in name or "decode" in name:
        return "gemv"
    if "dequant" in name or "quant" in name:
        return "quant"
    if "fusion" in str(kernel_path) or "mlp" in name or "norm" in name:
        return "fusion"
    if "sparse" in name:
        return "sparse"
    return "other"


def get_problem_sizes_for_kernel(kernel_path: Path) -> list[tuple[int, ...]]:
    """Get DIVERSE problem sizes covering multiple model architectures.

    Returns shapes from GLM-4.7, Qwen3, Llama-7B, Llama-70B combined.
    Optimization should find configs that work well ACROSS this diversity.
    """
    category = categorize_kernel(kernel_path)
    name = kernel_path.stem.lower()

    # MoE kernels get the full range of expert intermediate sizes
    if "moe" in name:
        return MOE_DIVERSE_SIZES

    # MLA/projection kernels
    if "mla" in name or "proj" in name:
        return PROJECTION_SIZES

    # Attention kernels need to handle different head configs
    if category == "attention":
        # Convert (batch, seq, heads, head_dim) to (M, N, K) for benchmarking
        return [(b * s, h * d, d) for b, s, h, d in DIVERSE_ATTENTION_SIZES]

    # GEMV (decode-focused) only uses small M
    if category == "gemv":
        return [(m, n, k) for m, n, k in DIVERSE_GEMM_SIZES if m <= 32]

    # Default: full GEMM diversity
    return DIVERSE_GEMM_SIZES


def find_optimizable_kernels() -> list[Path]:
    """Find all kernels with tunable TILE constants."""
    kernels = []

    for metal_file in SRC_DIR.glob("*.metal"):
        content = metal_file.read_text()
        if "constant" in content and "constexpr" in content and "TILE" in content:
            kernels.append(metal_file)

    # Also check fusion subdirectory
    if FUSION_DIR.exists():
        for metal_file in FUSION_DIR.glob("*.metal"):
            content = metal_file.read_text()
            if "constant" in content and "constexpr" in content and "TILE" in content:
                kernels.append(metal_file)

    return sorted(kernels)


def format_problem_sizes(sizes: list[tuple[int, ...]]) -> str:
    """Format problem sizes as CLI argument."""
    return ";".join(",".join(str(x) for x in size) for size in sizes)


def run_optimization(
    kernel_path: Path,
    num_random: int,
    generate_only: bool,
    agents: int,
    profile: str | None = None,
) -> tuple[str, bool]:
    """Run optimization for a single kernel."""

    problem_sizes = get_problem_sizes_for_kernel(kernel_path)

    # Limit problem sizes to avoid explosion (pick representative subset)
    if len(problem_sizes) > 8:
        # Select diverse subset: small, medium, large M + different N/K
        small_m = [s for s in problem_sizes if s[0] <= 32][:2]
        medium_m = [s for s in problem_sizes if 32 < s[0] <= 512][:3]
        large_m = [s for s in problem_sizes if s[0] > 512][:3]
        problem_sizes = small_m + medium_m + large_m

    problem_str = format_problem_sizes(problem_sizes)

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/optimize_kernel.py",
        str(kernel_path.relative_to(METAL_MARLIN_ROOT)),
        "--num-random",
        str(num_random),
        "--problem-sizes",
        problem_str,
    ]

    if generate_only:
        cmd.append("--generate-only")
    else:
        cmd.extend(["--agents", str(agents)])

    if profile:
        cmd.extend(["--profile", profile])

    result = subprocess.run(
        cmd,
        cwd=METAL_MARLIN_ROOT,
        capture_output=True,
        text=True,
    )

    return result.stdout + result.stderr, result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Batch optimize all Metal kernels using entropy-first exploration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Generalized Optimization:
  Problem sizes from MULTIPLE modern architectures (2026):
  
  Dense Models:
    - Qwen3-4B:      hidden=2560, intermediate=9728, GQA 4:1
    - Qwen3-32B:     hidden=5120, intermediate=25600, GQA 8:1
  
  MoE Models:
    - Qwen3-30B-A3B: hidden=2048, moe_intermediate=768 (VERY small!)
    - GLM-4.7-Flash: hidden=2048, moe_intermediate=1536, MLA
    - DeepSeek-V3.2: hidden=7168, moe_intermediate=2048, MLA+DSA

  Shaders are optimized to work well ACROSS this full range.

Categories:
  gemm       - Matrix multiplication (N=768 to N=25600)
  attention  - Attention (head_dim=128/256, 4-128 heads)
  moe        - MoE experts (N=768 Qwen, N=1536 GLM, N=2048 DeepSeek)
  gemv       - Matrix-vector (decode-focused, M<=32)
  quant      - Quantization/dequantization
  fusion     - Fused operation kernels
  sparse     - Sparse computation kernels

Examples:
  # Optimize all kernels with diverse problem sizes
  python scripts/optimize_all_kernels.py

  # Focus on MoE (tests N=768, N=1536, N=2048 together)
  python scripts/optimize_all_kernels.py --category moe

  # Higher entropy for thorough search
  python scripts/optimize_all_kernels.py --num-random 50

  # Use a specific optimize_kernel benchmark profile
  python scripts/optimize_all_kernels.py --profile mixed_bpw_fairway_glm47
""",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["gemm", "attention", "moe", "gemv", "quant", "fusion", "sparse", "all"],
        default="all",
        help="Kernel category to optimize (default: all)",
    )
    parser.add_argument(
        "--num-random",
        type=int,
        default=20,
        help="Number of random parameter combinations per kernel (default: 20)",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=50,
        help="Number of agents for AlphaHENG dispatch (default: 50)",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate task YAMLs without dispatching",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available kernels and exit",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        help="Optimize specific kernel by name (partial match)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        help=(
            "Forward benchmark profile to optimize_kernel.py "
            "(e.g., 'mixed_bpw_fairway_glm47' for GLM-4.7 optimization)"
        ),
    )

    args = parser.parse_args()

    # Find all optimizable kernels
    kernels = find_optimizable_kernels()

    # Filter by category
    if args.category != "all":
        kernels = [k for k in kernels if categorize_kernel(k) == args.category]

    # Filter by name
    if args.kernel:
        kernels = [k for k in kernels if args.kernel.lower() in k.stem.lower()]

    if not kernels:
        print("No matching kernels found.")
        sys.exit(1)

    # List mode
    if args.list:
        print(f"Found {len(kernels)} optimizable kernels:\n")
        by_category: dict[str, list[Path]] = {}
        for k in kernels:
            cat = categorize_kernel(k)
            by_category.setdefault(cat, []).append(k)

        for cat in sorted(by_category.keys()):
            print(f"  [{cat}]")
            for k in by_category[cat]:
                sizes = get_problem_sizes_for_kernel(k)
                print(f"    {k.name} ({len(sizes)} problem sizes)")
            print()
        return

    # Run optimization
    print("=" * 70)
    print("AlphaHENG Batch Kernel Optimization (Generalized)")
    print("=" * 70)
    print("Coverage: Qwen3-4B/32B + Qwen3-30B-A3B + GLM-4.7-Flash + DeepSeek-V3.2")
    print(f"Kernels: {len(kernels)}")
    print(f"Random variants per kernel: {args.num_random}")
    print(f"Mode: {'generate-only' if args.generate_only else f'dispatch ({args.agents} agents)'}")
    if args.profile:
        print(f"Profile: {args.profile}")
    print()

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    for i, kernel in enumerate(kernels, 1):
        category = categorize_kernel(kernel)
        problem_sizes = get_problem_sizes_for_kernel(kernel)

        print(f"[{i}/{len(kernels)}] {kernel.name} ({category})")
        print(f"  Problem sizes: {len(problem_sizes)} configurations")

        output, success = run_optimization(
            kernel,
            args.num_random,
            args.generate_only,
            args.agents,
            args.profile,
        )

        if success:
            print("  ✓ Generated tasks")
        else:
            print("  ✗ Failed")
            print(f"    {output[:200]}...")

        results.append((kernel.name, category, success))
        print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    success_count = sum(1 for _, _, s in results if s)
    print(f"Succeeded: {success_count}/{len(results)}")

    if not args.generate_only:
        print("\nMonitor progress:")
        print("  uv run alphaheng status")
        print("\nCollect results for each kernel:")
        print("  uv run python scripts/optimize_kernel.py --collect-results <SESSION_ID>")


if __name__ == "__main__":
    main()
