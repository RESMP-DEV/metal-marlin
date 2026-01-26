#!/usr/bin/env python3
"""
Metal Kernel Optimization Loop (AlphaHENG-powered)

Generates optimization tasks and dispatches them through AlphaHENG's
distributed agent swarm for parallel exploration of the optimization space.

Each optimization variant is a separate task that:
1. Applies a transformation to the kernel
2. Benchmarks the modified kernel
3. Reports results back

After all tasks complete, the best variant is selected and applied.

Usage:
    # Generate and dispatch optimization tasks
    uv run python scripts/optimize_kernel.py src/marlin_gemm.metal --agents 20

    # Generate tasks only (don't dispatch)
    uv run python scripts/optimize_kernel.py src/marlin_gemm.metal --generate-only

    # Collect results after swarm completes
    uv run python scripts/optimize_kernel.py src/marlin_gemm.metal --collect-results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Root of AlphaHENG repo
ALPHAHENG_ROOT = Path(__file__).parent.parent.parent.parent
METAL_MARLIN_ROOT = Path(__file__).parent.parent

# Optimization patterns to explore in parallel
OPTIMIZATION_PATTERNS: list[dict[str, Any]] = [
    # Tile size variants (explore in parallel)
    {
        "name": "tile_m_32",
        "description": "Set TILE_M=32",
        "pattern": r"#define\s+TILE_M\s+\d+",
        "replacement": "#define TILE_M 32",
        "applicable_to": ["gemm", "attention"],
    },
    {
        "name": "tile_m_64",
        "description": "Set TILE_M=64",
        "pattern": r"#define\s+TILE_M\s+\d+",
        "replacement": "#define TILE_M 64",
        "applicable_to": ["gemm", "attention"],
    },
    {
        "name": "tile_m_128",
        "description": "Set TILE_M=128",
        "pattern": r"#define\s+TILE_M\s+\d+",
        "replacement": "#define TILE_M 128",
        "applicable_to": ["gemm", "attention"],
    },
    {
        "name": "tile_n_32",
        "description": "Set TILE_N=32",
        "pattern": r"#define\s+TILE_N\s+\d+",
        "replacement": "#define TILE_N 32",
        "applicable_to": ["gemm"],
    },
    {
        "name": "tile_n_64",
        "description": "Set TILE_N=64",
        "pattern": r"#define\s+TILE_N\s+\d+",
        "replacement": "#define TILE_N 64",
        "applicable_to": ["gemm"],
    },
    {
        "name": "tile_n_128",
        "description": "Set TILE_N=128",
        "pattern": r"#define\s+TILE_N\s+\d+",
        "replacement": "#define TILE_N 128",
        "applicable_to": ["gemm"],
    },
    {
        "name": "tile_k_16",
        "description": "Set TILE_K=16",
        "pattern": r"#define\s+TILE_K\s+\d+",
        "replacement": "#define TILE_K 16",
        "applicable_to": ["gemm"],
    },
    {
        "name": "tile_k_32",
        "description": "Set TILE_K=32",
        "pattern": r"#define\s+TILE_K\s+\d+",
        "replacement": "#define TILE_K 32",
        "applicable_to": ["gemm"],
    },
    {
        "name": "tile_k_64",
        "description": "Set TILE_K=64",
        "pattern": r"#define\s+TILE_K\s+\d+",
        "replacement": "#define TILE_K 64",
        "applicable_to": ["gemm"],
    },
    # Threadgroup sizes
    {
        "name": "threads_128",
        "description": "Set THREADS_PER_TG=128",
        "pattern": r"#define\s+THREADS_PER_TG\s+\d+",
        "replacement": "#define THREADS_PER_TG 128",
        "applicable_to": ["gemm", "attention", "moe"],
    },
    {
        "name": "threads_256",
        "description": "Set THREADS_PER_TG=256",
        "pattern": r"#define\s+THREADS_PER_TG\s+\d+",
        "replacement": "#define THREADS_PER_TG 256",
        "applicable_to": ["gemm", "attention", "moe"],
    },
    {
        "name": "threads_512",
        "description": "Set THREADS_PER_TG=512",
        "pattern": r"#define\s+THREADS_PER_TG\s+\d+",
        "replacement": "#define THREADS_PER_TG 512",
        "applicable_to": ["gemm", "attention", "moe"],
    },
    # Simdgroup configurations
    {
        "name": "simd_4",
        "description": "Set SIMD_PER_TG=4",
        "pattern": r"#define\s+SIMD_PER_TG\s+\d+",
        "replacement": "#define SIMD_PER_TG 4",
        "applicable_to": ["gemm", "attention"],
    },
    {
        "name": "simd_8",
        "description": "Set SIMD_PER_TG=8",
        "pattern": r"#define\s+SIMD_PER_TG\s+\d+",
        "replacement": "#define SIMD_PER_TG 8",
        "applicable_to": ["gemm", "attention"],
    },
    {
        "name": "simd_16",
        "description": "Set SIMD_PER_TG=16",
        "pattern": r"#define\s+SIMD_PER_TG\s+\d+",
        "replacement": "#define SIMD_PER_TG 16",
        "applicable_to": ["gemm", "attention"],
    },
    # Accumulator precision
    {
        "name": "fp32_accumulator",
        "description": "Use float accumulator instead of half",
        "pattern": r"half\s+acc\b",
        "replacement": "float acc",
        "applicable_to": ["gemm", "attention"],
    },
    # Unroll factors
    {
        "name": "unroll_2",
        "description": "Add #pragma unroll 2 to main loop",
        "pattern": r"(for\s*\([^)]*k[^)]*\)\s*\{)",
        "replacement": "#pragma unroll 2\n    \\1",
        "applicable_to": ["gemm"],
        "once": True,
    },
    {
        "name": "unroll_4",
        "description": "Add #pragma unroll 4 to main loop",
        "pattern": r"(for\s*\([^)]*k[^)]*\)\s*\{)",
        "replacement": "#pragma unroll 4\n    \\1",
        "applicable_to": ["gemm"],
        "once": True,
    },
    {
        "name": "unroll_8",
        "description": "Add #pragma unroll 8 to main loop",
        "pattern": r"(for\s*\([^)]*k[^)]*\)\s*\{)",
        "replacement": "#pragma unroll 8\n    \\1",
        "applicable_to": ["gemm"],
        "once": True,
    },
]


@dataclass
class OptimizationVariant:
    """A single optimization variant to test."""

    name: str
    description: str
    kernel_path: str
    pattern: str
    replacement: str
    variant_source: str  # The modified source code

    def get_variant_hash(self) -> str:
        """Get a short hash identifying this variant."""
        return hashlib.sha256(self.variant_source.encode()).hexdigest()[:8]


@dataclass
class OptimizationResult:
    """Result from testing an optimization variant."""

    variant_name: str
    variant_hash: str
    compile_success: bool
    benchmark_us: float | None
    speedup_vs_baseline: float | None
    error: str | None = None


@dataclass
class OptimizationSession:
    """Track an optimization session."""

    kernel_path: str
    kernel_type: str
    session_id: str
    start_time: str
    baseline_us: float | None = None
    variants: list[OptimizationVariant] = field(default_factory=lambda: [])
    results: list[OptimizationResult] = field(default_factory=lambda: [])
    best_variant: str | None = None
    best_speedup: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel_path": self.kernel_path,
            "kernel_type": self.kernel_type,
            "session_id": self.session_id,
            "start_time": self.start_time,
            "baseline_us": self.baseline_us,
            "variants_count": len(self.variants),
            "results": [
                {
                    "name": r.variant_name,
                    "hash": r.variant_hash,
                    "compile_success": r.compile_success,
                    "benchmark_us": r.benchmark_us,
                    "speedup": r.speedup_vs_baseline,
                    "error": r.error,
                }
                for r in self.results
            ],
            "best_variant": self.best_variant,
            "best_speedup": self.best_speedup,
        }


def detect_kernel_type(kernel_path: Path) -> str:
    """Detect kernel type from filename."""
    name = kernel_path.stem.lower()
    if "gemm" in name or "marlin" in name:
        return "gemm"
    if "attention" in name or "flash" in name:
        return "attention"
    if "moe" in name:
        return "moe"
    if "dequant" in name:
        return "dequant"
    return "other"


def get_applicable_patterns(kernel_type: str, source: str) -> list[dict[str, Any]]:
    """Get patterns applicable to this kernel type that match the source."""
    patterns: list[dict[str, Any]] = []
    for p in OPTIMIZATION_PATTERNS:
        applicable = p.get("applicable_to", [])
        if kernel_type not in applicable and "all" not in applicable:
            continue
        # Check if pattern matches source
        if not re.search(p["pattern"], source):
            continue
        patterns.append(p)
    return patterns


def generate_variants(
    kernel_path: Path,
    kernel_type: str,
    source: str,
) -> list[OptimizationVariant]:
    """Generate all optimization variants to test."""
    variants: list[OptimizationVariant] = []
    patterns = get_applicable_patterns(kernel_type, source)

    for p in patterns:
        # Apply the transformation
        if p.get("once"):
            new_source = re.sub(p["pattern"], p["replacement"], source, count=1)
        else:
            new_source = re.sub(p["pattern"], p["replacement"], source)

        if new_source != source:
            variants.append(
                OptimizationVariant(
                    name=p["name"],
                    description=p["description"],
                    kernel_path=str(kernel_path),
                    pattern=p["pattern"],
                    replacement=p["replacement"],
                    variant_source=new_source,
                )
            )

    return variants


def generate_task_yaml(
    session: OptimizationSession,
    output_dir: Path,
) -> Path:
    """Generate AlphaHENG task YAML for optimization variants."""

    tasks: list[dict[str, Any]] = []

    # Task 0: Benchmark baseline
    tasks.append(
        {
            "name": f"opt-{session.session_id}-baseline",
            "prompt": f"""Benchmark the baseline kernel for optimization comparison.

Kernel: {session.kernel_path}
Session: {session.session_id}

Steps:
1. Run the benchmark script for this kernel
2. Record the timing in us (microseconds)
3. Save results to: agent_workspace/opt_{session.session_id}/baseline.json

```bash
cd {METAL_MARLIN_ROOT}
python3 -c "
import json
import time
import torch
from pathlib import Path

# Ensure kernel compiles
from metal_marlin.metal_dispatch import MetalKernelLibrary
lib = MetalKernelLibrary.from_source_dir()

# Benchmark (placeholder - real benchmark would dispatch kernel)
# TODO: Add actual kernel dispatch benchmark
benchmark_us = 100.0  # Placeholder

# Save result
out_dir = Path('agent_workspace/opt_{session.session_id}')
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / 'baseline.json').write_text(json.dumps({{
    'kernel': '{session.kernel_path}',
    'benchmark_us': benchmark_us,
    'timestamp': time.time(),
}}))
print(f'Baseline: {{benchmark_us}} us')
"
```
""",
            "priority": "P0",
            "dependencies": [],
        }
    )

    # Generate task for each variant
    for _, variant in enumerate(session.variants):
        variant_hash = variant.get_variant_hash()

        tasks.append(
            {
                "name": f"opt-{session.session_id}-{variant.name}",
                "prompt": f"""Test optimization variant: {variant.name}

Kernel: {session.kernel_path}
Variant: {variant.name} ({variant_hash})
Description: {variant.description}

Steps:
1. Create a temporary copy of the kernel with this modification
2. Apply transformation: `{variant.pattern}` → `{variant.replacement}`
3. Benchmark the modified kernel
4. Save results to: agent_workspace/opt_{session.session_id}/{variant.name}.json

The modified kernel source should have this change applied:
- Pattern: `{variant.pattern}`
- Replacement: `{variant.replacement}`

```bash
cd {METAL_MARLIN_ROOT}
python3 -c "
import json
import re
import time
import torch
from pathlib import Path

kernel_path = Path('{variant.kernel_path}')
original = kernel_path.read_text()

# Apply transformation
pattern = r'''{variant.pattern}'''
replacement = r'''{variant.replacement}'''
modified = re.sub(pattern, replacement, original, count=1)

if modified == original:
    print('ERROR: Pattern did not match')
    exit(1)

# Write modified kernel
kernel_path.write_text(modified)

try:
    # Test compilation
    from metal_marlin.metal_dispatch import MetalKernelLibrary
    lib = MetalKernelLibrary.from_source_dir()

    # Benchmark (placeholder)
    benchmark_us = 100.0  # TODO: Real benchmark
    compile_success = True
    error = None

except Exception as e:
    benchmark_us = None
    compile_success = False
    error = str(e)
finally:
    # Restore original
    kernel_path.write_text(original)

# Save result
out_dir = Path('agent_workspace/opt_{session.session_id}')
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / '{variant.name}.json').write_text(json.dumps({{
    'variant': '{variant.name}',
    'hash': '{variant_hash}',
    'compile_success': compile_success,
    'benchmark_us': benchmark_us,
    'error': error,
    'timestamp': time.time(),
}}))
print(f'{variant.name}: {{benchmark_us}} us' if compile_success else f'{variant.name}: FAILED - {{error}}')
"
```
""",
                "priority": "P1",
                "dependencies": [f"opt-{session.session_id}-baseline"],
            }
        )

    # Final task: Collect results and apply best
    all_variant_names = [f"opt-{session.session_id}-{v.name}" for v in session.variants]
    tasks.append(
        {
            "name": f"opt-{session.session_id}-collect",
            "prompt": f"""Collect optimization results and apply the best variant.

Session: {session.session_id}
Results directory: agent_workspace/opt_{session.session_id}/

Steps:
1. Read all result JSON files from the results directory
2. Find the variant with best speedup vs baseline
3. If best speedup > 1.02 (2% improvement), apply that variant permanently
4. Generate summary report

```bash
cd {METAL_MARLIN_ROOT}
python3 -c "
import json
import re
from pathlib import Path

results_dir = Path('agent_workspace/opt_{session.session_id}')
kernel_path = Path('{session.kernel_path}')

# Load baseline
baseline = json.loads((results_dir / 'baseline.json').read_text())
baseline_us = baseline['benchmark_us']

# Load all variant results
results = []
for f in results_dir.glob('*.json'):
    if f.name == 'baseline.json':
        continue
    data = json.loads(f.read_text())
    if data.get('compile_success') and data.get('benchmark_us'):
        speedup = baseline_us / data['benchmark_us']
        results.append({{
            'name': data['variant'],
            'benchmark_us': data['benchmark_us'],
            'speedup': speedup,
        }})

# Sort by speedup
results.sort(key=lambda x: x['speedup'], reverse=True)

print(f'Baseline: {{baseline_us:.2f}} us')
print()
print('Results (sorted by speedup):')
for r in results[:10]:
    print(f'  {{r[\"name\"]}}: {{r[\"benchmark_us\"]:.2f}} us ({{r[\"speedup\"]:.3f}}x)')

if results and results[0]['speedup'] > 1.02:
    best = results[0]
    print(f'')
    print(f'Best: {{best[\"name\"]}} with {{best[\"speedup\"]:.3f}}x speedup')
    print(f'TODO: Apply this variant permanently')
else:
    print(f'')
    print(f'No significant improvement found (threshold: 1.02x)')
"
```
""",
            "priority": "P0",
            "dependencies": all_variant_names,
        }
    )

    # Write YAML
    yaml_content = f"""# yaml-language-server: $schema=
# Metal Kernel Optimization Tasks
# Session: {session.session_id}
# Kernel: {session.kernel_path}
# Generated: {session.start_time}
# Variants: {len(session.variants)}

tasks:
"""

    for task in tasks:
        deps_str = json.dumps(task["dependencies"])
        yaml_content += f"""
  - name: {task["name"]}
    prompt: |
{_indent(task["prompt"], 6)}
    priority: {task["priority"]}
    dependencies: {deps_str}
"""

    output_path = output_dir / f"opt_{session.session_id}.yaml"
    output_path.write_text(yaml_content)
    return output_path


def _indent(text: str, spaces: int) -> str:
    """Indent all lines of text."""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.split("\n"))


def dispatch_tasks(yaml_path: Path, agents: int) -> None:
    """Dispatch tasks to AlphaHENG."""
    print(f"Adding tasks from: {yaml_path}")

    result = subprocess.run(
        ["uv", "run", "alphaheng", "tasks", "add", str(yaml_path)],
        cwd=ALPHAHENG_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Failed to add tasks: {result.stderr}")
        sys.exit(1)

    print(result.stdout)

    # Check if coordinator is running
    result = subprocess.run(
        ["pgrep", "-f", "alphaheng coordinator"],
        capture_output=True,
    )

    if result.returncode != 0:
        print(f"\nStarting coordinator with {agents} agents...")
        subprocess.Popen(
            ["uv", "run", "alphaheng", "coordinator", "--local-agents", str(agents)],
            cwd=ALPHAHENG_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("Coordinator started in background")
    else:
        print("Coordinator already running")


def collect_results(session_id: str) -> OptimizationSession | None:
    """Collect results from a completed optimization session."""
    results_dir = METAL_MARLIN_ROOT / "agent_workspace" / f"opt_{session_id}"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return None

    # Load baseline
    baseline_path = results_dir / "baseline.json"
    if not baseline_path.exists():
        print("Baseline results not found")
        return None

    baseline = json.loads(baseline_path.read_text())

    # Load variant results
    results: list[OptimizationResult] = []
    for f in results_dir.glob("*.json"):
        if f.name == "baseline.json":
            continue
        data = json.loads(f.read_text())
        speedup = None
        if data.get("compile_success") and data.get("benchmark_us"):
            speedup = baseline["benchmark_us"] / data["benchmark_us"]
        results.append(
            OptimizationResult(
                variant_name=data["variant"],
                variant_hash=data.get("hash", ""),
                compile_success=data.get("compile_success", False),
                benchmark_us=data.get("benchmark_us"),
                speedup_vs_baseline=speedup,
                error=data.get("error"),
            )
        )

    # Find best
    best_result: OptimizationResult | None = max(
        (r for r in results if r.speedup_vs_baseline),
        key=lambda r: r.speedup_vs_baseline or 0,
        default=None,
    )

    best_variant_name: str | None = best_result.variant_name if best_result else None
    best_speedup_value: float = (
        best_result.speedup_vs_baseline if best_result and best_result.speedup_vs_baseline else 1.0
    )

    session = OptimizationSession(
        kernel_path="",  # Would need to store this
        kernel_type="",
        session_id=session_id,
        start_time="",
        baseline_us=baseline["benchmark_us"],
        results=results,
        best_variant=best_variant_name,
        best_speedup=best_speedup_value,
    )

    return session


def main():
    parser = argparse.ArgumentParser(
        description="Optimize Metal kernels using AlphaHENG agent swarm"
    )
    parser.add_argument(
        "kernel",
        type=Path,
        nargs="?",
        help="Path to the Metal kernel file to optimize",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=10,
        help="Number of agents to use (default: 10)",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate task YAML without dispatching",
    )
    parser.add_argument(
        "--collect-results",
        type=str,
        metavar="SESSION_ID",
        help="Collect results from a completed session",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=METAL_MARLIN_ROOT / "tasks",
        help="Output directory for task YAML",
    )

    args = parser.parse_args()

    # Collect mode
    if args.collect_results:
        session = collect_results(args.collect_results)
        if session:
            print(f"\nSession: {session.session_id}")
            print(f"Baseline: {session.baseline_us:.2f} us")
            print(f"Best: {session.best_variant} ({session.best_speedup:.3f}x)")
            print("\nAll results:")
            for r in sorted(
                session.results, key=lambda x: x.speedup_vs_baseline or 0, reverse=True
            ):
                status = "✓" if r.compile_success else "✗"
                speedup = f"{r.speedup_vs_baseline:.3f}x" if r.speedup_vs_baseline else "N/A"
                print(f"  {status} {r.variant_name}: {speedup}")
        return

    # Require kernel for generate/dispatch mode
    if not args.kernel:
        parser.error("kernel argument is required unless using --collect-results")

    # Resolve kernel path
    kernel_path = args.kernel
    if not kernel_path.exists():
        # Try relative to src/
        src_path = METAL_MARLIN_ROOT / "src" / args.kernel
        if src_path.exists():
            kernel_path = src_path
        else:
            print(f"Kernel not found: {args.kernel}")
            sys.exit(1)

    kernel_path = kernel_path.resolve()

    # Read kernel source
    source = kernel_path.read_text()
    kernel_type = detect_kernel_type(kernel_path)

    # Generate session ID
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate variants
    variants = generate_variants(kernel_path, kernel_type, source)

    if not variants:
        print(f"No applicable optimization patterns for {kernel_path.name}")
        sys.exit(1)

    print(f"Kernel: {kernel_path.name}")
    print(f"Type: {kernel_type}")
    print(f"Variants: {len(variants)}")
    print()

    # Create session
    session = OptimizationSession(
        kernel_path=str(kernel_path),
        kernel_type=kernel_type,
        session_id=session_id,
        start_time=datetime.now().isoformat(),
        variants=variants,
    )

    # Generate task YAML
    args.output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = generate_task_yaml(session, args.output_dir)
    print(f"Generated: {yaml_path}")

    if args.generate_only:
        print("\nTask YAML generated. To dispatch:")
        print(f"  uv run alphaheng tasks add {yaml_path}")
        return

    # Dispatch to AlphaHENG
    dispatch_tasks(yaml_path, args.agents)

    print(f"\nOptimization session started: {session_id}")
    print("Monitor progress: uv run alphaheng status")
    print(f"Collect results: python scripts/optimize_kernel.py --collect-results {session_id}")


if __name__ == "__main__":
    main()
