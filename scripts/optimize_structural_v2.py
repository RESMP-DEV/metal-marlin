#!/usr/bin/env python3
"""
Structural Metal Kernel Optimization v2 (Prescriptive Benchmarking)

Unlike v1 which asked LLMs to "analyze and propose", this script generates
tasks with EMBEDDED BENCHMARK CODE that always runs and saves results.

Each task tests a SPECIFIC structural transformation:
1. Async prefetch insertion (simdgroup_async_copy)
2. Loop unrolling pragmas
3. Barrier reduction
4. Threadgroup size adjustments
5. Occupancy attributes

The key difference from v1:
- v1: "Please analyze and save results if you find improvements"
- v2: "Run this benchmark code which ALWAYS saves JSON"

Usage:
    cd contrib/metal_marlin && uv run python scripts/optimize_structural_v2.py
    cd contrib/metal_marlin && uv run python scripts/optimize_structural_v2.py --kernel marlin_gemm.metal
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
METAL_MARLIN_ROOT = _SCRIPT_DIR.parent
SRC_DIR = METAL_MARLIN_ROOT / "src"
FUSION_DIR = SRC_DIR / "fusion"

_POTENTIAL_ALPHAHENG_ROOT = METAL_MARLIN_ROOT.parent.parent
ALPHAHENG_ROOT = (
    _POTENTIAL_ALPHAHENG_ROOT if (_POTENTIAL_ALPHAHENG_ROOT / "alphaheng").is_dir() else None
)

# Structural transformations with regex patterns and replacements
STRUCTURAL_TRANSFORMS: dict[str, dict] = {
    # DISABLED: Loop unrolling causes duplicate pragma errors on most attention kernels
    # Most Metal kernels already have #pragma unroll where needed
    # "unroll_inner_loop": {
    #     "description": "Add #pragma unroll to inner loops",
    #     "pattern": r"...",
    #     "applicable_to": ["attention", "moe"],
    # },
    # Async copy prefetch - wrap device reads in async copy
    "async_prefetch_hint": {
        "description": "Add memory prefetch hint before main loop",
        "pattern": r"(kernel\s+void\s+\w+[^{]+\{)",
        "replacement": r"\1\n    // Prefetch hint: consider simdgroup_async_copy for better memory throughput",
        "max_applications": 1,
        "applicable_to": ["gemm", "attention"],
    },
    # Explicit simdgroup barrier instead of threadgroup barrier
    "simdgroup_barrier": {
        "description": "Use simdgroup_barrier where full threadgroup sync isn't needed",
        "pattern": r"threadgroup_barrier\(mem_flags::mem_threadgroup\)",
        "replacement": "simdgroup_barrier(mem_flags::mem_none)",
        "max_applications": 1,  # Conservative - only first occurrence
        "applicable_to": ["gemm", "attention", "moe"],
    },
    # NOTE: thread_execution_width is NOT a valid Metal attribute.
    # SIMD width is implicitly 32 on Apple Silicon. Removed invalid transform.
    # Max threads per threadgroup attribute - correct Metal syntax
    "max_threads_256": {
        "description": "Limit max threads per threadgroup to 256 for better occupancy",
        "pattern": r"^(kernel\s+void\s+)(\w+\s*\()",
        "replacement": r"[[max_total_threads_per_threadgroup(256)]]\n\1\2",
        "max_applications": 1,
        "applicable_to": ["attention"],
    },
    # Reduce register pressure with explicit casting
    "half_precision_compute": {
        "description": "Use half precision for intermediate computations",
        "pattern": r"(\s+)float\s+(\w+)\s*=\s*([^;]+);(\s*//[^\n]*)?",
        "replacement": r"\1half \2 = half(\3);\4",
        "max_applications": 2,  # Only first few occurrences
        "applicable_to": ["elementwise"],
    },
}


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
    if "layernorm" in name or "rope" in name or "hadamard" in name:
        return "elementwise"
    return "other"


def get_problem_sizes_for_category(category: str) -> list[tuple[int, int, int]]:
    """Get representative problem sizes for benchmarking."""
    if category == "attention":
        return [(1, 4096, 128), (32, 4096, 128), (512, 4096, 128)]
    elif category == "moe":
        return [(1, 768, 2048), (32, 768, 2048), (256, 1536, 2048)]
    elif category in ("gemm", "gemv"):
        return [(1, 9728, 2560), (32, 9728, 2560), (256, 9728, 2560)]
    elif category == "elementwise":
        return [(1, 2560, 1), (512, 2560, 1), (2048, 5120, 1)]
    else:
        return [(1, 4096, 4096), (32, 4096, 4096), (256, 4096, 4096)]


def get_applicable_transforms(kernel_path: Path) -> list[tuple[str, dict]]:
    """Get transforms applicable to this kernel."""
    import re as re_module

    category = categorize_kernel(kernel_path)
    content = kernel_path.read_text()

    applicable = []
    for name, transform in STRUCTURAL_TRANSFORMS.items():
        # Check if category matches
        if category not in transform.get("applicable_to", []):
            continue

        # Check if we should skip (e.g., already has this optimization)
        skip_if = transform.get("skip_if_contains")
        if skip_if and skip_if in content:
            continue

        # Check if pattern exists in kernel (use MULTILINE for ^ patterns)
        pattern = transform["pattern"]
        flags = re_module.MULTILINE if pattern.startswith("^") else 0
        if not re_module.search(pattern, content, flags):
            continue

        applicable.append((name, transform))

    return applicable


def generate_benchmark_code(
    kernel_path: Path,
    transform_name: str,
    transform: dict,
    session_id: str,
    problem_sizes: list[tuple[int, int, int]],
) -> str:
    """Generate Python benchmark code that ALWAYS runs and saves results."""
    problem_str = ", ".join(f"[{m},{n},{k}]" for m, n, k in problem_sizes)
    variant_hash = hashlib.md5(f"{transform_name}:{kernel_path.name}".encode()).hexdigest()[:8]

    return f"""
import json
import re
import time
import sys
sys.path.insert(0, '.')

from pathlib import Path

kernel_path = Path('{kernel_path}')
original = kernel_path.read_text()

# Transform details
transform_name = '{transform_name}'
pattern = r'{transform["pattern"]}'
replacement = r'{transform["replacement"]}'
max_applications = {transform.get("max_applications", 1)}

# Apply transformation
modified = original
count = 0
for _ in range(max_applications):
    new_modified = re.sub(pattern, replacement, modified, count=1)
    if new_modified == modified:
        break
    modified = new_modified
    count += 1

result = {{
    'variant': '{transform_name}',
    'hash': '{variant_hash}',
    'kernel': '{kernel_path.name}',
    'transform_applied': count > 0,
    'transform_count': count,
    'description': '{transform["description"]}',
}}

if count == 0:
    print(f'Pattern did not match - kernel may not need this optimization')
    result['compile_success'] = True
    result['note'] = 'Pattern not found - optimization not applicable'
else:
    # Write modified kernel
    kernel_path.write_text(modified)

    try:
        from metal_marlin._compat import HAS_MPS, torch
        from metal_marlin.metal_dispatch import MetalKernelLibrary, dispatch_gemm_fp4
        from metal_marlin.kernels import pack_fp4_weights

        if not HAS_MPS:
            raise RuntimeError('MPS not available')

        # Recompile with modified source
        lib = MetalKernelLibrary.from_source_dir()

        problem_sizes = {problem_str}
        baseline_results = {{}}
        modified_results = {{}}

        # Benchmark BASELINE first (restore original, run, then re-apply modified)
        kernel_path.write_text(original)
        lib_baseline = MetalKernelLibrary.from_source_dir()

        for M, N, K in problem_sizes:
            A = torch.randn(M, K, dtype=torch.float16, device='mps')
            weight = torch.randn(N, K, dtype=torch.float16, device='mps')
            B_packed, scales = pack_fp4_weights(weight, group_size=32)

            # Warmup
            for _ in range(5):
                _ = dispatch_gemm_fp4(lib_baseline, A, B_packed, scales, M, N, K, 32)
            torch.mps.synchronize()

            # Benchmark baseline
            start = time.perf_counter()
            for _ in range(20):
                _ = dispatch_gemm_fp4(lib_baseline, A, B_packed, scales, M, N, K, 32)
            torch.mps.synchronize()
            elapsed = time.perf_counter() - start
            baseline_results[(M, N, K)] = (elapsed / 20) * 1e6

        # Benchmark MODIFIED
        kernel_path.write_text(modified)
        lib_modified = MetalKernelLibrary.from_source_dir()

        for M, N, K in problem_sizes:
            A = torch.randn(M, K, dtype=torch.float16, device='mps')
            weight = torch.randn(N, K, dtype=torch.float16, device='mps')
            B_packed, scales = pack_fp4_weights(weight, group_size=32)

            # Warmup
            for _ in range(5):
                _ = dispatch_gemm_fp4(lib_modified, A, B_packed, scales, M, N, K, 32)
            torch.mps.synchronize()

            # Benchmark modified
            start = time.perf_counter()
            for _ in range(20):
                _ = dispatch_gemm_fp4(lib_modified, A, B_packed, scales, M, N, K, 32)
            torch.mps.synchronize()
            elapsed = time.perf_counter() - start
            modified_results[(M, N, K)] = (elapsed / 20) * 1e6

        # Calculate speedup
        total_baseline = sum(baseline_results.values())
        total_modified = sum(modified_results.values())
        speedup = total_baseline / total_modified if total_modified > 0 else 1.0

        result['compile_success'] = True
        result['baseline_us'] = {{str(k): v for k, v in baseline_results.items()}}
        result['modified_us'] = {{str(k): v for k, v in modified_results.items()}}
        result['speedup'] = speedup
        result['timestamp'] = time.time()

        print(f'{transform_name}: speedup={{speedup:.3f}}x')
        for sz in problem_sizes:
            print(f'  {{sz}}: {{baseline_results[tuple(sz)]:.2f}} -> {{modified_results[tuple(sz)]:.2f}} us')

    except Exception as e:
        result['compile_success'] = False
        result['error'] = str(e)
        print(f'{transform_name}: FAILED - {{e}}')
    finally:
        # ALWAYS restore original
        kernel_path.write_text(original)

# ALWAYS save result
out_dir = Path('agent_workspace/struct_{session_id}')
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / '{kernel_path.stem}_{transform_name}.json'
out_file.write_text(json.dumps(result, indent=2))
print(f'Saved to {{out_file}}')
"""


def generate_task(
    kernel_path: Path,
    transform_name: str,
    transform: dict,
    session_id: str,
) -> dict:
    """Generate a task with embedded benchmark code."""

    category = categorize_kernel(kernel_path)
    problem_sizes = get_problem_sizes_for_category(category)

    benchmark_code = generate_benchmark_code(
        kernel_path, transform_name, transform, session_id, problem_sizes
    )

    # Escape for YAML embedding
    benchmark_code_escaped = benchmark_code.replace("\\", "\\\\").replace('"', '\\"')

    return {
        "name": f"struct-{session_id}-{kernel_path.stem}-{transform_name}",
        "prompt": f"""Test structural optimization: {transform_name}

Kernel: {kernel_path.name}
Transform: {transform["description"]}

This task has EMBEDDED BENCHMARK CODE that runs automatically.
Your job is to execute it and report the results.

```bash
cd contrib/metal_marlin && uv run python -c "{benchmark_code_escaped}"
```

After running, check the output JSON for speedup results.
If speedup > 1.05 (5% improvement), the optimization is worthwhile.
""",
        "priority": "P1",
        "dependencies": [],
    }


def generate_collect_task(
    session_id: str,
    tasks: list[dict],
) -> dict:
    """Generate a task to collect all results and find the best."""

    task_names = [t["name"] for t in tasks]

    return {
        "name": f"struct-{session_id}-collect",
        "prompt": f"""Collect structural optimization results and identify winners.

Session: {session_id}
Results directory: `contrib/metal_marlin/agent_workspace/struct_{session_id}/`

```bash
cd contrib/metal_marlin && uv run python -c "
import json
from pathlib import Path

results_dir = Path('agent_workspace/struct_{session_id}')
winners = []
all_results = []

for f in sorted(results_dir.glob('*.json')):
    data = json.loads(f.read_text())
    all_results.append(data)

    kernel = data.get('kernel', 'unknown')
    variant = data.get('variant', 'unknown')
    speedup = data.get('speedup', 1.0)
    success = data.get('compile_success', False)

    if success and speedup > 1.05:
        winners.append((kernel, variant, speedup))
        print(f'WINNER: {{kernel}} - {{variant}}: {{speedup:.3f}}x')
    else:
        print(f'  skip: {{kernel}} - {{variant}}: {{speedup:.3f}}x')

print()
print(f'Total results: {{len(all_results)}}')
print(f'Winners (>5% speedup): {{len(winners)}}')

# Save summary
summary = {{
    'session': '{session_id}',
    'total_tests': len(all_results),
    'winners': [
        {{'kernel': k, 'variant': v, 'speedup': s}}
        for k, v, s in winners
    ]
}}
(results_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
print(f'Summary saved to {{results_dir / \"summary.json\"}}')
"
```

Report which optimizations showed improvement.
""",
        "priority": "P3",
        "dependencies": task_names,
    }


def generate_apply_task(session_id: str) -> dict:
    """Generate a task to apply winning optimizations."""

    return {
        "name": f"struct-{session_id}-apply",
        "prompt": f"""Apply winning structural optimizations from session {session_id}.

Read `contrib/metal_marlin/agent_workspace/struct_{session_id}/summary.json` to find winners.

For each winner with speedup > 1.10 (10% improvement):
1. Apply the transformation to the source file
2. Run tests to verify correctness
3. Commit with message describing the optimization

```bash
cd contrib/metal_marlin && uv run python -c "
import json
from pathlib import Path

summary = json.loads(Path('agent_workspace/struct_{session_id}/summary.json').read_text())
for w in summary.get('winners', []):
    if w['speedup'] > 1.10:
        print(f'APPLY: {{w[\"kernel\"]}} - {{w[\"variant\"]}} ({{w[\"speedup\"]:.3f}}x)')
    else:
        print(f'SKIP (too small): {{w[\"kernel\"]}} - {{w[\"variant\"]}} ({{w[\"speedup\"]:.3f}}x)')
"
```

For significant winners, apply the transformation manually by:
1. Reading the result JSON to get the regex pattern and replacement
2. Applying to the source file
3. Running: `uv run pytest tests/ -v -k gemm -x`
""",
        "priority": "P3",
        "dependencies": [f"struct-{session_id}-collect"],
    }


def find_all_metal_kernels() -> list[Path]:
    """Find all Metal shader files."""
    kernels = list(SRC_DIR.glob("*.metal"))
    if FUSION_DIR.exists():
        kernels.extend(FUSION_DIR.glob("*.metal"))
    return sorted(kernels)


def generate_task_yaml(
    tasks: list[dict],
    session_id: str,
    output_dir: Path,
) -> Path:
    """Generate AlphaHENG task YAML."""

    yaml_content = f"""# yaml-language-server: $schema=
# Structural Metal Kernel Optimization v2
# Session: {session_id}
# Generated: {datetime.now().isoformat()}
# Tasks: {len(tasks)}
#
# This version has EMBEDDED BENCHMARK CODE that always runs.
# Each task applies a specific transformation and measures impact.

tasks:
"""

    for task in tasks:
        deps_str = json.dumps(task["dependencies"])
        # Indent prompt properly
        prompt_lines = task["prompt"].split("\n")
        indented_prompt = "\n".join("      " + line for line in prompt_lines)

        yaml_content += f"""
  - name: {task["name"]}
    prompt: |
{indented_prompt}
    priority: {task["priority"]}
    dependencies: {deps_str}
"""

    output_path = output_dir / f"struct_v2_{session_id}.yaml"
    output_path.write_text(yaml_content)
    return output_path


def run_benchmark_directly(
    kernel_path: Path,
    transform_name: str,
    transform: dict,
    session_id: str,
) -> dict:
    """Run benchmark directly without going through task queue."""
    import re
    import time

    try:
        from metal_marlin._compat import HAS_MPS, torch
        from metal_marlin.kernels import pack_fp4_weights
        from metal_marlin.metal_dispatch import MetalKernelLibrary, dispatch_gemm_fp4
    except ImportError as e:
        return {
            "variant": transform_name,
            "kernel": kernel_path.name,
            "compile_success": False,
            "error": f"Import failed: {e}",
        }

    if not HAS_MPS:
        return {
            "variant": transform_name,
            "kernel": kernel_path.name,
            "compile_success": False,
            "error": "MPS not available",
        }

    original = kernel_path.read_text()
    pattern = transform["pattern"]
    replacement = transform["replacement"]
    max_applications = transform.get("max_applications", 1)

    # Use MULTILINE for patterns that use ^ (start of line)
    flags = re.MULTILINE if pattern.startswith("^") else 0

    # Apply transformation
    modified = original
    count = 0
    for _ in range(max_applications):
        new_modified = re.sub(pattern, replacement, modified, count=1, flags=flags)
        if new_modified == modified:
            break
        modified = new_modified
        count += 1

    result = {
        "variant": transform_name,
        "kernel": kernel_path.name,
        "transform_applied": count > 0,
        "transform_count": count,
        "description": transform["description"],
    }

    if count == 0:
        result["compile_success"] = True
        result["note"] = "Pattern not found - optimization not applicable"
        return result

    # Write modified kernel
    kernel_path.write_text(modified)

    try:
        category = categorize_kernel(kernel_path)
        problem_sizes = get_problem_sizes_for_category(category)

        # Benchmark BASELINE first
        kernel_path.write_text(original)
        lib_baseline = MetalKernelLibrary.from_source_dir()

        baseline_results = {}
        for M, N, K in problem_sizes:
            A = torch.randn(M, K, dtype=torch.float16, device="mps")
            weight = torch.randn(N, K, dtype=torch.float16, device="mps")
            B_packed, scales = pack_fp4_weights(weight, group_size=32)

            # Warmup
            for _ in range(5):
                _ = dispatch_gemm_fp4(lib_baseline, A, B_packed, scales, M, N, K, 32)
            torch.mps.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(20):
                _ = dispatch_gemm_fp4(lib_baseline, A, B_packed, scales, M, N, K, 32)
            torch.mps.synchronize()
            elapsed = time.perf_counter() - start
            baseline_results[(M, N, K)] = (elapsed / 20) * 1e6

        # Benchmark MODIFIED
        kernel_path.write_text(modified)
        lib_modified = MetalKernelLibrary.from_source_dir()

        modified_results = {}
        for M, N, K in problem_sizes:
            A = torch.randn(M, K, dtype=torch.float16, device="mps")
            weight = torch.randn(N, K, dtype=torch.float16, device="mps")
            B_packed, scales = pack_fp4_weights(weight, group_size=32)

            # Warmup
            for _ in range(5):
                _ = dispatch_gemm_fp4(lib_modified, A, B_packed, scales, M, N, K, 32)
            torch.mps.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(20):
                _ = dispatch_gemm_fp4(lib_modified, A, B_packed, scales, M, N, K, 32)
            torch.mps.synchronize()
            elapsed = time.perf_counter() - start
            modified_results[(M, N, K)] = (elapsed / 20) * 1e6

        # Calculate speedup
        total_baseline = sum(baseline_results.values())
        total_modified = sum(modified_results.values())
        speedup = total_baseline / total_modified if total_modified > 0 else 1.0

        result["compile_success"] = True
        result["baseline_us"] = {str(k): v for k, v in baseline_results.items()}
        result["modified_us"] = {str(k): v for k, v in modified_results.items()}
        result["speedup"] = speedup
        result["timestamp"] = time.time()

    except Exception as e:
        result["compile_success"] = False
        result["error"] = str(e)
    finally:
        # ALWAYS restore original
        kernel_path.write_text(original)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate structural optimization tasks with embedded benchmarks",
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
        help="List applicable transforms per kernel without generating tasks",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run benchmarks directly instead of generating tasks",
    )

    args = parser.parse_args()

    # Find kernels
    kernels = find_all_metal_kernels()

    if args.kernel:
        kernels = [k for k in kernels if args.kernel.lower() in k.name.lower()]

    if not kernels:
        print("No matching kernels found.")
        return

    # Find applicable transforms for each kernel
    kernel_transforms: list[tuple[Path, str, dict]] = []
    for kernel in kernels:
        transforms = get_applicable_transforms(kernel)
        for name, transform in transforms:
            kernel_transforms.append((kernel, name, transform))

    if args.list:
        print(f"Found {len(kernel_transforms)} applicable transformations:\n")
        for kernel, name, transform in kernel_transforms:
            print(f"  {kernel.name}: {name}")
            print(f"    {transform['description']}")
        return

    if not kernel_transforms:
        print("No applicable transformations found for these kernels.")
        print("Try: --kernel marlin_gemm.metal")
        return

    # Generate session ID
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Direct execution mode
    if args.run:
        print("=" * 70)
        print("Structural Metal Kernel Optimization v2 - DIRECT RUN")
        print("=" * 70)
        print(f"Session: {session_id}")
        print(f"Testing {len(kernel_transforms)} transformations...")
        print()

        results_dir = METAL_MARLIN_ROOT / "agent_workspace" / f"struct_{session_id}"
        results_dir.mkdir(parents=True, exist_ok=True)

        all_results = []
        winners = []

        for i, (kernel, name, transform) in enumerate(kernel_transforms, 1):
            print(f"[{i}/{len(kernel_transforms)}] {kernel.name} - {name}...")

            result = run_benchmark_directly(kernel, name, transform, session_id)
            all_results.append(result)

            # Save individual result
            result_file = results_dir / f"{kernel.stem}_{name}.json"
            result_file.write_text(json.dumps(result, indent=2))

            # Print result
            if result.get("compile_success"):
                speedup = result.get("speedup", 1.0)
                if speedup > 1.05:
                    winners.append((kernel.name, name, speedup))
                    print(f"  WINNER: {speedup:.3f}x speedup")
                elif "note" in result:
                    print(f"  Skip: {result['note']}")
                else:
                    print(f"  {speedup:.3f}x (no significant change)")
            else:
                print(f"  FAILED: {result.get('error', 'Unknown error')}")

        # Summary
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total tests: {len(all_results)}")
        print(f"Winners (>5% speedup): {len(winners)}")
        print()

        if winners:
            print("Winning optimizations:")
            for kernel, variant, speedup in sorted(winners, key=lambda x: -x[2]):
                print(f"  {kernel} - {variant}: {speedup:.3f}x")
        else:
            print("No significant improvements found.")

        # Save summary
        summary = {
            "session": session_id,
            "total_tests": len(all_results),
            "winners": [{"kernel": k, "variant": v, "speedup": s} for k, v, s in winners],
        }
        summary_file = results_dir / "summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        print(f"\nResults saved to: {results_dir}")
        return

    # Generate tasks for AlphaHENG
    tasks = []
    for kernel, name, transform in kernel_transforms:
        task = generate_task(kernel, name, transform, session_id)
        tasks.append(task)

    # Add collect and apply tasks
    tasks.append(generate_collect_task(session_id, list(tasks)))
    tasks.append(generate_apply_task(session_id))

    # Generate YAML
    args.output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = generate_task_yaml(tasks, session_id, args.output_dir)

    print("=" * 70)
    print("Structural Metal Kernel Optimization v2")
    print("=" * 70)
    print(f"Session: {session_id}")
    print(f"Kernels: {len(kernels)}")
    print(f"Transforms: {len(kernel_transforms)}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Generated: {yaml_path}")
    print()
    print("Key difference from v1:")
    print("  - Each task has EMBEDDED benchmark code that ALWAYS runs")
    print("  - Results are ALWAYS saved to JSON (not dependent on LLM)")
    print("  - Specific transformations instead of open-ended analysis")
    print()
    print("To dispatch:")
    if ALPHAHENG_ROOT:
        print(f"  cd {ALPHAHENG_ROOT} && uv run alphaheng tasks add {yaml_path}")
    else:
        print(f"  uv run alphaheng tasks add {yaml_path}")


if __name__ == "__main__":
    main()
