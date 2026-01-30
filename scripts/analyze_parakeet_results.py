#!/usr/bin/env python3
"""
Analyze Parakeet-TDT benchmark results and generate comprehensive reports.

This script reads benchmark JSON files and creates:
1. Summary table in markdown format
2. Throughput comparison charts using matplotlib
3. Memory vs quality tradeoff plots
4. M4 Max specific recommendations

Usage:
    python scripts/analyze_parakeet_results.py --help
    python scripts/analyze_parakeet_results.py --output-dir docs/
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    import matplotlib.style as style

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Charts will be skipped.", file=sys.stderr)


@dataclass
class BenchmarkResult:
    """Single benchmark result data structure."""

    config_name: str
    audio_length_sec: float
    transcription_time_sec: float
    memory_peak_gb: float
    realtime_factor: float
    device: str = "mps"


@dataclass
class MpsBaselineResult:
    """MPS baseline benchmark result."""

    transcription_time_sec: float
    memory_peak_gb: float
    realtime_factor: float


@dataclass
class CpuBaselineResult:
    """CPU baseline benchmark result."""

    transcription_time_sec: float
    memory_peak_gb: float
    realtime_factor: float


def load_json(file_path: Path) -> dict[str, Any]:
    """Load and parse JSON file with error handling."""
    if not file_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {file_path}")

    try:
        with open(file_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")


def parse_comprehensive_results(data: dict[str, Any]) -> list[BenchmarkResult]:
    """Parse comprehensive benchmark results."""
    results = []
    for result_data in data.get("results", []):
        result = BenchmarkResult(
            config_name=result_data.get("config_name", "unknown"),
            audio_length_sec=result_data.get("audio_length_sec", 0.0),
            transcription_time_sec=result_data.get("transcription_time_sec", 0.0),
            memory_peak_gb=result_data.get("memory_peak_gb", 0.0),
            realtime_factor=result_data.get("realtime_factor", 0.0),
            device=result_data.get("device", "mps"),
        )
        results.append(result)
    return results


def parse_scaling_results(data: dict[str, Any]) -> dict[str, list[BenchmarkResult]]:
    """Parse audio scaling benchmark results by configuration."""
    config_results = defaultdict(list)

    for result_data in data.get("results", []):
        config_name = result_data.get("config_name", "unknown")
        result = BenchmarkResult(
            config_name=config_name,
            audio_length_sec=result_data.get("audio_length_sec", 0.0),
            transcription_time_sec=result_data.get("transcription_time_sec", 0.0),
            memory_peak_gb=result_data.get("memory_peak_gb", 0.0),
            realtime_factor=result_data.get("realtime_factor", 0.0),
            device=result_data.get("device", "mps"),
        )
        config_results[config_name].append(result)

    return dict(config_results)


def parse_baseline_results(data: dict[str, Any]) -> tuple[MpsBaselineResult, CpuBaselineResult]:
    """Parse MPS baseline benchmark results."""
    mps_data = data.get("mps", {})
    mps_metrics = mps_data.get("metrics", {})
    mps_result = MpsBaselineResult(
        transcription_time_sec=mps_metrics.get("transcription_time_sec", 0.0),
        memory_peak_gb=mps_metrics.get("memory_peak_gb", 0.0),
        realtime_factor=mps_metrics.get("realtime_factor", 0.0),
    )

    cpu_data = data.get("cpu", {})
    cpu_metrics = cpu_data.get("metrics", {})
    cpu_result = CpuBaselineResult(
        transcription_time_sec=cpu_metrics.get("transcription_time_sec", 0.0),
        memory_peak_gb=cpu_metrics.get("memory_peak_gb", 0.0),
        realtime_factor=cpu_metrics.get("realtime_factor", 0.0),
    )

    return mps_result, cpu_result


def generate_summary_table(comprehensive_results: list[BenchmarkResult]) -> str:
    """Generate markdown summary table of benchmark results."""
    if not comprehensive_results:
        return "No comprehensive benchmark results available.\n"

    # Group results by configuration
    config_stats = defaultdict(list)
    for result in comprehensive_results:
        config_stats[result.config_name].append(result)

    # Calculate averages for each configuration
    summary_data = []
    for config_name, results in config_stats.items():
        avg_rt_factor = sum(r.realtime_factor for r in results) / len(results)
        avg_memory = sum(r.memory_peak_gb for r in results) / len(results)
        avg_time = sum(r.transcription_time_sec for r in results) / len(results)

        summary_data.append(
            {
                "config": config_name,
                "avg_realtime_factor": avg_rt_factor,
                "avg_memory_gb": avg_memory,
                "avg_time_sec": avg_time,
                "samples": len(results),
            }
        )

    # Sort by realtime factor (higher is better)
    summary_data.sort(key=lambda x: x["avg_realtime_factor"], reverse=True)

    # Generate markdown table
    table = "## Performance Summary\n\n"
    table += "| Configuration | Avg Real-time Factor | Avg Memory (GB) | Avg Time (s) | Samples |\n"
    table += "|---------------|---------------------|----------------|--------------|---------|\n"

    for data in summary_data:
        table += f"| {data['config']} | {data['avg_realtime_factor']:.1f}x | "
        table += f"{data['avg_memory_gb']:.3f} | {data['avg_time_sec']:.3f} | {data['samples']} |\n"

    table += "\n"
    return table


def generate_scaling_analysis(scaling_results: dict[str, list[BenchmarkResult]]) -> str:
    """Generate analysis of audio scaling performance."""
    if not scaling_results:
        return "No scaling benchmark results available.\n"

    analysis = "## Audio Length Scaling Analysis\n\n"

    # Find best performer for each audio length
    lengths = set()
    for results in scaling_results.values():
        lengths.update(r.audio_length_sec for r in results)

    lengths = sorted(lengths)

    analysis += "### Best Configuration by Audio Length\n\n"
    analysis += "| Audio Length (s) | Best Config | Real-time Factor | Memory (GB) |\n"
    analysis += "|------------------|-------------|------------------|-------------|\n"

    for length in lengths:
        best_result = None
        best_config = None

        for config_name, results in scaling_results.items():
            for result in results:
                if result.audio_length_sec == length:
                    if best_result is None or result.realtime_factor > best_result.realtime_factor:
                        best_result = result
                        best_config = config_name

        if best_result:
            analysis += f"| {length:.1f} | {best_config} | {best_result.realtime_factor:.1f}x | "
            analysis += f"{best_result.memory_peak_gb:.3f} |\n"

    analysis += "\n"

    # Performance trends
    analysis += "### Performance Trends\n\n"
    for config_name, results in scaling_results.items():
        results.sort(key=lambda x: x.audio_length_sec)
        if len(results) >= 2:
            first_rt = results[0].realtime_factor
            last_rt = results[-1].realtime_factor
            improvement = (last_rt - first_rt) / first_rt * 100
            analysis += f"- **{config_name}**: {improvement:+.1f}% improvement from {results[0].audio_length_sec}s to {results[-1].audio_length_sec}s\n"

    analysis += "\n"
    return analysis


def generate_throughput_chart(
    scaling_results: dict[str, list[BenchmarkResult]], output_path: Path
) -> None:
    """Generate throughput comparison chart."""
    if not HAS_MATPLOTLIB or not scaling_results:
        return

    style.use("default")
    plt.figure(figsize=(12, 8))

    # Plot real-time factor vs audio length for each configuration
    for config_name, results in scaling_results.items():
        results.sort(key=lambda x: x.audio_length_sec)
        lengths = [r.audio_length_sec for r in results]
        rt_factors = [r.realtime_factor for r in results]
        plt.plot(lengths, rt_factors, marker="o", label=config_name, linewidth=2, markersize=6)

    plt.xlabel("Audio Length (seconds)", fontsize=12)
    plt.ylabel("Real-time Factor (higher is better)", fontsize=12)
    plt.title("Parakeet-TDT Throughput vs Audio Length", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_memory_chart(
    scaling_results: dict[str, list[BenchmarkResult]], output_path: Path
) -> None:
    """Generate memory usage comparison chart."""
    if not HAS_MATPLOTLIB or not scaling_results:
        return

    style.use("default")
    plt.figure(figsize=(12, 8))

    # Plot memory usage vs audio length for each configuration
    for config_name, results in scaling_results.items():
        results.sort(key=lambda x: x.audio_length_sec)
        lengths = [r.audio_length_sec for r in results]
        memories = [r.memory_peak_gb for r in results]
        plt.plot(lengths, memories, marker="s", label=config_name, linewidth=2, markersize=6)

    plt.xlabel("Audio Length (seconds)", fontsize=12)
    plt.ylabel("Peak Memory Usage (GB)", fontsize=12)
    plt.title("Parakeet-TDT Memory Usage vs Audio Length", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_m4_max_recommendations(
    comprehensive_results: list[BenchmarkResult],
    scaling_results: dict[str, list[BenchmarkResult]],
    mps_baseline: MpsBaselineResult | None = None,
) -> str:
    """Generate M4 Max specific recommendations."""
    recommendations = "## M4 Max Recommendations\n\n"

    if not comprehensive_results and not scaling_results:
        recommendations += "No benchmark data available for recommendations.\n"
        return recommendations

    # Find best overall configuration
    all_results = comprehensive_results.copy()
    for results in scaling_results.values():
        all_results.extend(results)

    if all_results:
        # Group by configuration and calculate average performance
        config_performance = defaultdict(list)
        for result in all_results:
            # Use real-time factor as primary metric
            performance_score = result.realtime_factor
            # Penalize high memory usage (>1GB is concerning for M4 Max)
            if result.memory_peak_gb > 1.0:
                performance_score *= 0.8
            config_performance[result.config_name].append(performance_score)

        # Find best configuration
        best_config = None
        best_score = 0

        for config_name, scores in config_performance.items():
            avg_score = sum(scores) / len(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_config = config_name

        if best_config:
            recommendations += f"### Best Overall Configuration: **{best_config}**\n\n"

            # Get specific results for this config
            config_results = [r for r in all_results if r.config_name == best_config]
            if config_results:
                avg_memory = sum(r.memory_peak_gb for r in config_results) / len(config_results)
                avg_rt_factor = sum(r.realtime_factor for r in config_results) / len(config_results)

                recommendations += f"- **Average Performance**: {avg_rt_factor:.1f}x real-time\n"
                recommendations += f"- **Average Memory**: {avg_memory:.3f} GB\n"
                recommendations += f"- **M4 Max Efficiency**: {'Excellent' if avg_memory < 0.1 else 'Good' if avg_memory < 0.5 else 'Moderate'}\n\n"

    # Specific M4 Max optimizations
    recommendations += "### M4 Max Optimization Tips\n\n"
    recommendations += "1. **Memory Management**: M4 Max has unified memory architecture. "
    recommendations += (
        "Configurations using <100MB memory are optimal for concurrent processing.\n\n"
    )

    recommendations += (
        "2. **Neural Engine**: Consider configurations that leverage Apple's Neural Engine "
    )
    recommendations += "for best power efficiency.\n\n"

    recommendations += (
        "3. **Thermal Considerations**: Sustained workloads benefit from configurations "
    )
    recommendations += "with lower peak memory usage to maintain performance.\n\n"

    # Audio length specific recommendations
    if scaling_results:
        recommendations += "### Audio Length Recommendations\n\n"

        for length in [1.0, 5.0, 10.0, 30.0]:
            best_for_length = None
            best_rt = 0

            for config_name, results in scaling_results.items():
                for result in results:
                    if (
                        abs(result.audio_length_sec - length) < 0.1
                        and result.realtime_factor > best_rt
                    ):
                        best_rt = result.realtime_factor
                        best_for_length = config_name

            if best_for_length:
                recommendations += (
                    f"- **{length}s audio**: Use `{best_for_length}` ({best_rt:.1f}x real-time)\n"
                )

        recommendations += "\n"

    return recommendations


def generate_report(
    comprehensive_path: Path, scaling_path: Path, baseline_path: Path, output_dir: Path
) -> None:
    """Generate complete benchmark report."""
    # Load data
    comprehensive_data = load_json(comprehensive_path)
    scaling_data = load_json(scaling_path)
    baseline_data = load_json(baseline_path) if baseline_path.exists() else None

    # Parse results
    comprehensive_results = parse_comprehensive_results(comprehensive_data)
    scaling_results = parse_scaling_results(scaling_data)

    mps_baseline = None
    if baseline_data:
        mps_baseline, _ = parse_baseline_results(baseline_data)

    # Generate markdown content
    md = "# Parakeet-TDT-0.6B Benchmark Results\n\n"
    md += f"**Generated**: {comprehensive_data.get('date', 'Unknown date')}\n\n"

    md += generate_summary_table(comprehensive_results)
    md += generate_scaling_analysis(scaling_results)

    # Add chart references if matplotlib is available
    if HAS_MATPLOTLIB:
        md += "## Performance Visualizations\n\n"
        md += "![Throughput Comparison](charts/parakeet_throughput_comparison.png)\n\n"
        md += "![Memory Usage](charts/parakeet_memory_usage.png)\n\n"

    md += generate_m4_max_recommendations(comprehensive_results, scaling_results, mps_baseline)

    # Add methodology section
    md += "## Benchmark Methodology\n\n"
    md += "- **Audio Sample Rate**: 16kHz\n"
    md += "- **Test Lengths**: 1s, 5s, 10s, 30s audio segments\n"
    md += "- **Device**: Apple Silicon (MPS backend)\n"
    md += "- **Metrics**: Real-time factor, peak memory, transcription time\n"
    md += "- **Configurations Tested**: FP4 baseline, hybrid conservative/aggressive, mixed precision\n\n"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save markdown report
    report_path = output_dir / "parakeet_benchmark_report.md"
    with open(report_path, "w") as f:
        f.write(md)

    print(f"Generated report: {report_path}")

    # Generate charts if matplotlib is available
    if HAS_MATPLOTLIB:
        charts_dir = output_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        throughput_path = charts_dir / "parakeet_throughput_comparison.png"
        memory_path = charts_dir / "parakeet_memory_usage.png"

        generate_throughput_chart(scaling_results, throughput_path)
        generate_memory_chart(scaling_results, memory_path)

        if throughput_path.exists():
            print(f"Generated throughput chart: {throughput_path}")
        if memory_path.exists():
            print(f"Generated memory chart: {memory_path}")
    else:
        print("Skipping charts (matplotlib not available)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze Parakeet-TDT benchmark results")
    parser.add_argument(
        "--comprehensive",
        default="benchmarks/results/parakeet_comprehensive.json",
        help="Path to comprehensive benchmark results",
    )
    parser.add_argument(
        "--scaling",
        default="benchmarks/results/parakeet_audio_scaling.json",
        help="Path to audio scaling benchmark results",
    )
    parser.add_argument(
        "--baseline",
        default="benchmarks/results/parakeet_mps_baseline.json",
        help="Path to MPS baseline benchmark results",
    )
    parser.add_argument(
        "--output-dir", default="docs", help="Output directory for report and charts"
    )

    args = parser.parse_args()

    # Convert paths to Path objects
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    comprehensive_path = project_root / args.comprehensive
    scaling_path = project_root / args.scaling
    baseline_path = project_root / args.baseline
    output_dir = project_root / args.output_dir

    try:
        generate_report(comprehensive_path, scaling_path, baseline_path, output_dir)
        print("Analysis completed successfully!")
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
