#!/usr/bin/env python3
"""
Automated benchmark suite runner.

Runs all performance benchmarks, saves results with timestamp,
compares against previous run, and reports regressions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


import os

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

def run_benchmark(script_path: Path, args: list[str] = None) -> dict[str, Any] | None:
    """Run a single benchmark script and capture its JSON output."""
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )

        if result.returncode != 0:
            print(f"⚠️  {script_path.name} failed with code {result.returncode}")
            return None

        # Try to parse JSON from stdout
        for line in result.stdout.split("\n"):
            line = line.strip()
            if line.startswith("{") or line.startswith("["):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

        print(f"⚠️  {script_path.name} produced no JSON output")
        return None

    except subprocess.TimeoutExpired:
        print(f"⚠️  {script_path.name} timed out after 300s")
        return None
    except Exception as e:
        print(f"⚠️  {script_path.name} error: {e}")
        return None


def discover_benchmarks(bench_dir: Path, include: list[str] = None, exclude: list[str] = None) -> list[Path]:
    """Discover benchmark scripts in the benchmarks directory."""
    patterns = ["bench_*.py", "benchmark_*.py"]

    scripts = []
    for pattern in patterns:
        scripts.extend(bench_dir.glob(pattern))

    # Filter by include/exclude
    if include:
        scripts = [s for s in scripts if any(inc in s.name for inc in include)]
    if exclude:
        scripts = [s for s in scripts if not any(exc in s.name for exc in exclude)]

    # Exclude this script
    scripts = [s for s in scripts if s.name != "run_all.py"]

    return sorted(scripts)


def load_previous_results(results_file: Path) -> dict[str, Any] | None:
    """Load the most recent benchmark results."""
    if not results_file.exists():
        return None

    try:
        with results_file.open() as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Could not load previous results: {e}")
        return None


def compare_results(current: dict[str, Any], previous: dict[str, Any], threshold: float = 0.1) -> list[dict]:
    """Compare current results against previous run and detect regressions."""
    regressions = []

    current_benchmarks = current.get("benchmarks", {})
    previous_benchmarks = previous.get("benchmarks", {})

    for bench_name, current_data in current_benchmarks.items():
        if bench_name not in previous_benchmarks:
            continue

        previous_data = previous_benchmarks[bench_name]

        # Extract metrics to compare (mean time, throughput, etc.)
        current_metrics = extract_metrics(current_data)
        previous_metrics = extract_metrics(previous_data)

        for metric_name, current_value in current_metrics.items():
            if metric_name not in previous_metrics:
                continue

            previous_value = previous_metrics[metric_name]

            # Skip if either value is invalid
            if current_value <= 0 or previous_value <= 0:
                continue

            # For time metrics, higher is worse; for throughput, lower is worse
            if "time" in metric_name.lower() or "latency" in metric_name.lower():
                change = (current_value - previous_value) / previous_value
                if change > threshold:
                    regressions.append({
                        "benchmark": bench_name,
                        "metric": metric_name,
                        "previous": previous_value,
                        "current": current_value,
                        "change_pct": change * 100,
                        "type": "slowdown",
                    })
            elif "throughput" in metric_name.lower() or "ops" in metric_name.lower():
                change = (previous_value - current_value) / previous_value
                if change > threshold:
                    regressions.append({
                        "benchmark": bench_name,
                        "metric": metric_name,
                        "previous": previous_value,
                        "current": current_value,
                        "change_pct": change * 100,
                        "type": "slowdown",
                    })

    return regressions


def extract_metrics(benchmark_data: Any) -> dict[str, float]:
    """Extract numeric metrics from benchmark data."""
    metrics = {}

    if isinstance(benchmark_data, dict):
        for key, value in benchmark_data.items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
            elif isinstance(value, dict):
                # Recursively extract nested metrics
                nested = extract_metrics(value)
                for nested_key, nested_value in nested.items():
                    metrics[f"{key}.{nested_key}"] = nested_value

    return metrics


def save_results(results: dict[str, Any], output_file: Path):
    """Save benchmark results to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all performance benchmarks and report regressions"
    )
    parser.add_argument(
        "--include",
        nargs="+",
        help="Only run benchmarks matching these patterns",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="Exclude benchmarks matching these patterns",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Regression threshold as fraction (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file (default: results/benchmark_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="Compare against specific previous results file",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip comparison with previous results",
    )
    args = parser.parse_args()

    # Setup paths
    bench_dir = Path(__file__).parent
    results_dir = bench_dir / "results"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or (results_dir / f"benchmark_{timestamp}.json")

    # Discover benchmarks
    benchmarks = discover_benchmarks(bench_dir, args.include, args.exclude)

    if not benchmarks:
        print("No benchmarks found!")
        return 1

    print(f"Running {len(benchmarks)} benchmarks...")
    print()

    # Run benchmarks
    results = {
        "timestamp": timestamp,
        "benchmarks": {},
    }

    for i, bench_path in enumerate(benchmarks, 1):
        print(f"[{i}/{len(benchmarks)}] Running {bench_path.name}...", end=" ", flush=True)

        bench_result = run_benchmark(bench_path)

        if bench_result:
            results["benchmarks"][bench_path.stem] = bench_result
            print("✓")
        else:
            print("✗")

    print()

    # Save results
    save_results(results, output_file)

    # Compare with previous run
    if not args.no_compare:
        if args.compare:
            previous = load_previous_results(args.compare)
        else:
            # Find most recent results file
            existing_results = sorted(results_dir.glob("benchmark_*.json"))
            if len(existing_results) > 1:
                # Get second-to-last (latest is the one we just created)
                previous = load_previous_results(existing_results[-2])
            else:
                previous = None

        if previous:
            print()
            print("Comparing with previous results...")
            regressions = compare_results(results, previous, args.threshold)

            if regressions:
                print()
                print(f"⚠️  Found {len(regressions)} regression(s):")
                print()
                for reg in regressions:
                    print(f"  {reg['benchmark']}.{reg['metric']}:")
                    print(f"    Previous: {reg['previous']:.4f}")
                    print(f"    Current:  {reg['current']:.4f}")
                    print(f"    Change:   {reg['change_pct']:+.2f}%")
                    print()
                return 1
            else:
                print("✓ No regressions detected")
        else:
            print("ℹ️  No previous results to compare against")

    print()
    print("✓ Benchmark suite completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
