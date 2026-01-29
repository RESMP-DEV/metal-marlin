#!/usr/bin/env python3
"""
Structural Metal Kernel Optimization (AlphaHENG-powered)

Unlike optimize_kernel.py which searches for TILE_* constants to tune,
this script generates LLM-driven optimization tasks for ANY kernel.

The LLM analyzes each kernel's structure and proposes optimizations like:
- Adding simdgroup_matrix operations where beneficial
- Introducing async prefetching (simdgroup_async_copy)
- Reducing threadgroup_barrier count
- Loop unrolling opportunities
- Memory access pattern improvements
- Occupancy tuning (threads per threadgroup)

This works on ALL kernels, not just those with tunable constants.

Usage:
    # Generate optimization tasks for all kernels
    cd contrib/metal_marlin && uv run python scripts/optimize_all_structural.py

    # Generate for specific category
    cd contrib/metal_marlin && uv run python scripts/optimize_all_structural.py --category attention

    # Generate for specific kernel
    cd contrib/metal_marlin && uv run python scripts/optimize_all_structural.py --kernel layernorm.metal
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
METAL_MARLIN_ROOT = _SCRIPT_DIR.parent
SRC_DIR = METAL_MARLIN_ROOT / "src"
FUSION_DIR = SRC_DIR / "fusion"

# AlphaHENG root for task dispatch
_POTENTIAL_ALPHAHENG_ROOT = METAL_MARLIN_ROOT.parent.parent
ALPHAHENG_ROOT = (
    _POTENTIAL_ALPHAHENG_ROOT if (_POTENTIAL_ALPHAHENG_ROOT / "alphaheng").is_dir() else None
)


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
    if "layernorm" in name or "rope" in name or "hadamard" in name:
        return "elementwise"
    if "sampling" in name or "router" in name:
        return "routing"
    return "other"


def analyze_kernel_structure(kernel_path: Path) -> dict:
    """Analyze a kernel's structure for optimization opportunities."""
    content = kernel_path.read_text()
    lines = content.split("\n")

    return {
        "path": str(kernel_path),
        "name": kernel_path.name,
        "lines": len(lines),
        "simdgroup_ops": content.count("simdgroup"),
        "simdgroup_matrix": content.count("simdgroup_matrix"),
        "async_copy": content.count("async_copy"),
        "barriers": content.count("threadgroup_barrier"),
        "loops": content.count("for (") + content.count("for("),
        "tile_constants": content.count("TILE_"),
        "has_threadgroup_mem": "threadgroup " in content,
        "has_device_mem": "device " in content,
        "kernel_count": content.count("kernel "),
    }


def get_problem_sizes_for_category(category: str) -> list[tuple[int, int, int]]:
    """Get representative problem sizes for benchmarking.

    These are derived from Qwen3-4B/32B, Qwen3-30B-A3B, GLM-4.7-Flash, DeepSeek-V3.2.
    """
    if category == "attention":
        # (batch*seq, heads*head_dim, head_dim) shapes
        return [
            (1, 4096, 128),  # single token
            (32, 4096, 128),  # batched decode
            (512, 4096, 128),  # prefill
            (2048, 8192, 128),  # long context
        ]
    elif category == "moe":
        # MoE expert sizes: N=768 (Qwen), N=1536 (GLM), N=2048 (DeepSeek)
        return [
            (1, 768, 2048),  # Qwen3-30B MoE decode
            (32, 768, 2048),  # batched
            (256, 1536, 2048),  # GLM-4.7 prefill
            (256, 2048, 7168),  # DeepSeek-V3.2 prefill
        ]
    elif category in ("gemm", "gemv"):
        return [
            (1, 9728, 2560),  # Qwen3-4B up proj decode
            (32, 9728, 2560),  # batched
            (256, 25600, 5120),  # Qwen3-32B prefill
            (2048, 9728, 2560),  # long context
        ]
    elif category == "elementwise":
        # (batch, seq_len, hidden_size) flattened
        return [
            (1, 2560, 1),  # single token Qwen3-4B
            (512, 2560, 1),  # prefill
            (2048, 5120, 1),  # Qwen3-32B long
            (8192, 7168, 1),  # DeepSeek-V3.2 long
        ]
    else:
        # Default GEMM-like sizes
        return [
            (1, 4096, 4096),
            (32, 4096, 4096),
            (256, 4096, 4096),
            (1024, 4096, 4096),
        ]


def generate_structural_task(
    kernel_path: Path,
    analysis: dict,
    session_id: str,
) -> dict:
    """Generate an LLM-driven structural optimization task for a kernel."""

    category = categorize_kernel(kernel_path)
    problem_sizes = get_problem_sizes_for_category(category)
    problem_str = "; ".join(f"({m},{n},{k})" for m, n, k in problem_sizes)

    # Build context about the kernel's current state
    context_lines = [
        f"Kernel: {kernel_path.name}",
        f"Category: {category}",
        f"Lines of code: {analysis['lines']}",
        f"Simdgroup operations: {analysis['simdgroup_ops']}",
        f"Simdgroup matrix ops: {analysis['simdgroup_matrix']}",
        f"Async copy operations: {analysis['async_copy']}",
        f"Threadgroup barriers: {analysis['barriers']}",
        f"Loops: {analysis['loops']}",
        f"TILE constants: {analysis['tile_constants']}",
        f"Uses threadgroup memory: {analysis['has_threadgroup_mem']}",
    ]
    context = "\n".join(context_lines)

    return {
        "name": f"struct-{session_id}-{kernel_path.stem}",
        "prompt": f"""Analyze and optimize this Metal kernel for Apple Silicon (M1/M2/M3/M4).

**Kernel Analysis:**
{context}

**Problem sizes to benchmark:** {problem_str}

**TASK:**
1. Read the kernel source at `contrib/metal_marlin/{kernel_path.relative_to(METAL_MARLIN_ROOT)}`
2. Identify structural optimization opportunities (not just constant tuning):

   **Memory Access Patterns:**
   - Can we add simdgroup_async_copy for prefetching?
   - Are memory accesses coalesced?
   - Can we reduce bank conflicts in threadgroup memory?

   **Compute Patterns:**
   - Can scalar operations be replaced with simdgroup_matrix (8x8)?
   - Are there opportunities for SIMD shuffle operations?
   - Can we reduce the number of threadgroup_barrier calls?

   **Loop Structure:**
   - Are loops amenable to unrolling?
   - Can we pipeline loop iterations?
   - Are there loop-invariant computations to hoist?

   **Occupancy:**
   - Is the threadgroup size optimal for the workload?
   - Can we adjust threads per simdgroup usage?

3. Propose 1-3 specific code modifications with:
   - Clear before/after code snippets
   - Hypothesis for why this improves performance
   - Risk assessment (will it compile? correctness concerns?)

4. For the MOST PROMISING optimization:
   - Create a modified version of the kernel
   - Benchmark baseline vs modified
   - Report speedup

5. Save results to:
   `contrib/metal_marlin/agent_workspace/struct_{session_id}/{kernel_path.stem}.json`

**OUTPUT FORMAT:**
```json
{{
  "kernel": "{kernel_path.name}",
  "baseline_us": {{}},  // timing per problem size
  "optimizations_proposed": [
    {{
      "name": "descriptive_name",
      "hypothesis": "why this should help",
      "applied": true/false,
      "modified_us": {{}},  // if applied
      "speedup": 1.0,  // if applied
      "code_diff": "before -> after summary"
    }}
  ],
  "best_optimization": "name or null",
  "best_speedup": 1.0
}}
```

**IMPORTANT:** 
- Focus on STRUCTURAL changes, not just constant tuning
- If the kernel is already well-optimized, say so
- Correctness is paramount - don't break functionality
""",
        "priority": "P1",
        "dependencies": [],
    }


def generate_apply_task(session_id: str, kernels: list[Path]) -> dict:
    """Generate a task to apply the best optimizations found."""

    kernel_names = [k.stem for k in kernels]

    return {
        "name": f"struct-{session_id}-apply",
        "prompt": f"""Apply the best structural optimizations found in this session.

Session: {session_id}
Results directory: `contrib/metal_marlin/agent_workspace/struct_{session_id}/`

**TASK:**
1. Read all result JSON files from the directory
2. For each kernel with speedup > 1.05 (5% improvement):
   - Review the proposed optimization
   - Verify the code diff is safe
   - Apply it to the source file if confident
3. Generate summary report

**Kernels analyzed:** {", ".join(kernel_names)}

**Commands:**
```bash
cd contrib/metal_marlin && uv run python -c "
import json
from pathlib import Path

results_dir = Path('agent_workspace/struct_{session_id}')
applied = []
skipped = []

for f in sorted(results_dir.glob('*.json')):
    data = json.loads(f.read_text())
    kernel = data.get('kernel', f.stem)
    best = data.get('best_speedup', 1.0)
    opt_name = data.get('best_optimization')
    
    if best > 1.05 and opt_name:
        print(f'{{kernel}}: {{best:.3f}}x ({{opt_name}})')
        applied.append((kernel, best, opt_name))
    else:
        skipped.append(kernel)

print(f'\\nApplied: {{len(applied)}} kernels')
print(f'Skipped: {{len(skipped)}} kernels (no significant improvement)')
"
```
""",
        "priority": "P3",
        "dependencies": [f"struct-{session_id}-{k.stem}" for k in kernels],
    }


def find_all_metal_kernels() -> list[Path]:
    """Find all Metal shader files."""
    kernels = list(SRC_DIR.glob("*.metal"))
    if FUSION_DIR.exists():
        kernels.extend(FUSION_DIR.glob("*.metal"))
    return sorted(kernels)


def generate_task_yaml(
    kernels: list[Path],
    analyses: list[dict],
    session_id: str,
    output_dir: Path,
) -> Path:
    """Generate AlphaHENG task YAML for structural optimization."""

    tasks = []

    # Generate analysis task for each kernel
    for kernel, analysis in zip(kernels, analyses):
        task = generate_structural_task(kernel, analysis, session_id)
        tasks.append(task)

    # Add final apply task
    tasks.append(generate_apply_task(session_id, kernels))

    # Write YAML
    yaml_content = f"""# yaml-language-server: $schema=
# Metal Kernel Structural Optimization Tasks
# Session: {session_id}
# Generated: {datetime.now().isoformat()}
# Kernels: {len(kernels)}
# 
# This uses LLM-driven analysis to find STRUCTURAL optimizations,
# not just constant tuning. Works on ALL kernels.

tasks:
"""

    for task in tasks:
        deps_str = json.dumps(task["dependencies"])
        # Indent the prompt properly
        prompt_lines = task["prompt"].split("\n")
        indented_prompt = "\n".join("      " + line for line in prompt_lines)

        yaml_content += f"""
  - name: {task["name"]}
    prompt: |
{indented_prompt}
    priority: {task["priority"]}
    dependencies: {deps_str}
"""

    output_path = output_dir / f"struct_{session_id}.yaml"
    output_path.write_text(yaml_content)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate structural optimization tasks for ALL Metal kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Unlike optimize_kernel.py which only works on kernels with TILE_* constants,
this script generates LLM-driven optimization tasks for ANY kernel.

The LLM analyzes each kernel's structure and proposes optimizations like:
- Adding simdgroup_matrix operations
- Introducing async prefetching
- Reducing barrier counts
- Loop unrolling
- Memory access improvements

Examples:
  # Generate tasks for all kernels
  python scripts/optimize_all_structural.py

  # Focus on attention kernels
  python scripts/optimize_all_structural.py --category attention

  # Specific kernel
  python scripts/optimize_all_structural.py --kernel layernorm.metal
""",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=[
            "attention",
            "moe",
            "gemm",
            "gemv",
            "quant",
            "fusion",
            "sparse",
            "elementwise",
            "routing",
            "other",
            "all",
        ],
        default="all",
        help="Kernel category to optimize (default: all)",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        help="Specific kernel to optimize (partial name match)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=METAL_MARLIN_ROOT / "tasks",
        help="Output directory for task YAML",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all kernels and their analysis, without generating tasks",
    )

    args = parser.parse_args()

    # Find all kernels
    kernels = find_all_metal_kernels()

    # Filter by category
    if args.category != "all":
        kernels = [k for k in kernels if categorize_kernel(k) == args.category]

    # Filter by name
    if args.kernel:
        kernels = [k for k in kernels if args.kernel.lower() in k.name.lower()]

    if not kernels:
        print("No matching kernels found.")
        return

    # Analyze all kernels
    analyses = [analyze_kernel_structure(k) for k in kernels]

    # List mode
    if args.list:
        print(f"Found {len(kernels)} kernels:\n")

        # Group by category
        by_category: dict[str, list[tuple[Path, dict]]] = {}
        for k, a in zip(kernels, analyses):
            cat = categorize_kernel(k)
            by_category.setdefault(cat, []).append((k, a))

        for cat in sorted(by_category.keys()):
            print(f"  [{cat}]")
            for k, a in by_category[cat]:
                print(
                    f"    {k.name:40s} "
                    f"simd:{a['simdgroup_ops']:3d} "
                    f"loops:{a['loops']:3d} "
                    f"barriers:{a['barriers']:2d} "
                    f"matrix:{a['simdgroup_matrix']:2d}"
                )
            print()
        return

    # Generate session ID
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate YAML
    args.output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = generate_task_yaml(kernels, analyses, session_id, args.output_dir)

    print("=" * 70)
    print("Structural Metal Kernel Optimization")
    print("=" * 70)
    print(f"Session: {session_id}")
    print(f"Kernels: {len(kernels)}")
    print(f"Generated: {yaml_path}")
    print()
    print("This generates LLM-driven analysis tasks that look for:")
    print("  - simdgroup_matrix opportunities")
    print("  - Async prefetch potential")
    print("  - Barrier reduction")
    print("  - Loop unrolling")
    print("  - Memory access patterns")
    print()
    print("To dispatch:")
    if ALPHAHENG_ROOT:
        print(f"  cd {ALPHAHENG_ROOT} && uv run alphaheng tasks add {yaml_path}")
    else:
        print(f"  uv run alphaheng tasks add {yaml_path}")


if __name__ == "__main__":
    main()
