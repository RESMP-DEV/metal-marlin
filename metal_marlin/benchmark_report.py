"""
Benchmark result aggregation and reporting.

Collects benchmark results from multiple JSON files, aggregates statistics,
and generates markdown reports with visualizations.

Usage:
    # Generate report from benchmark results
    python -m metal_marlin.benchmark_report generate ./benchmarks/results/

    # Plot perplexity vs compression
    python -m metal_marlin.benchmark_report plot-ppl ./benchmarks/results/

    # Plot throughput comparison
    python -m metal_marlin.benchmark_report plot-throughput ./benchmarks/results/
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore[assignment]
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    plt = None  # type: ignore[assignment]
    HAS_MATPLOTLIB = False


# Default paths relative to package root
_ROOT = Path(__file__).parent.parent  # metal_marlin/
_BENCHMARKS_DIR = _ROOT / "benchmarks"
_RESULTS_DIR = _BENCHMARKS_DIR / "results"


@dataclass
class BenchmarkEntry:
    """Single benchmark measurement from JSON."""

    model: str
    config: str  # e.g., "fp4-g128", "int4-g64", "uniform", "mixed"
    quant_type: str  # "fp4", "int4", "fp8", "fp16"
    group_size: int
    calibration: str  # "wikitext2", "bartowski_v3", "none"

    # Perplexity metrics
    ppl_base: float  # Baseline FP16 perplexity
    ppl_quant: float  # Quantized perplexity
    ppl_delta_pct: float  # (quant - base) / base * 100

    # Throughput metrics
    tokens_per_sec: float
    tflops: float
    memory_gb_s: float

    # Size metrics
    model_size_gb: float
    compression_ratio: float  # Original / Quantized size

    # Metadata
    timestamp: str = ""
    hardware: str = ""
    notes: str = ""


def aggregate_results(results_dir: str | Path) -> pd.DataFrame:
    """
    Load all benchmark JSONs from a directory into a DataFrame.

    Scans for JSON files matching patterns:
    - *_benchmark.json
    - *_results.json
    - comparison*.json
    - perplexity*.json

    Args:
        results_dir: Directory containing benchmark JSON files.

    Returns:
        DataFrame with columns for all benchmark metrics.
        Returns empty DataFrame if pandas not available or no results found.
    """
    if not HAS_PANDAS:
        raise ImportError(
            "pandas required for aggregate_results. Install with: pip install pandas"
        )

    results_dir = Path(results_dir)
    if not results_dir.exists():
        return pd.DataFrame()

    entries: list[dict[str, Any]] = []

    # Find all JSON files matching patterns
    patterns = ["*_benchmark.json", "*_results.json", "comparison*.json", "perplexity*.json"]
    json_files: list[Path] = []
    for pattern in patterns:
        json_files.extend(results_dir.glob(pattern))

    # Also include bare .json files
    json_files.extend(results_dir.glob("*.json"))
    json_files = list(set(json_files))

    for json_path in sorted(json_files):
        try:
            data = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not load {json_path}: {e}")
            continue

        # Handle different JSON formats
        entries.extend(_parse_benchmark_json(data, json_path.name))

    if not entries:
        return pd.DataFrame()

    df = pd.DataFrame(entries)

    # Ensure numeric columns
    numeric_cols = [
        "ppl_base",
        "ppl_quant",
        "ppl_delta_pct",
        "tokens_per_sec",
        "tflops",
        "memory_gb_s",
        "model_size_gb",
        "compression_ratio",
        "group_size",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _parse_benchmark_json(data: dict[str, Any], filename: str) -> list[dict[str, Any]]:
    """Parse benchmark JSON into list of entry dicts."""
    entries: list[dict[str, Any]] = []

    # Extract timestamp and hardware from config section if present
    config = data.get("config", {})
    timestamp = data.get("timestamp", "")
    hardware = config.get("hardware", config.get("hw_peak_tflops", ""))
    if isinstance(hardware, (int, float)):
        hardware = f"M4 Max ({hardware:.0f} TFLOPS)"

    # Handle "results" array format (from bench_comparison.py)
    if "results" in data and isinstance(data["results"], list):
        for r in data["results"]:
            entry = _extract_entry(r, timestamp, hardware, filename)
            if entry:
                entries.append(entry)

    # Handle flat single-result format
    elif "label" in data or "model" in data:
        entry = _extract_entry(data, timestamp, hardware, filename)
        if entry:
            entries.append(entry)

    # Handle perplexity results format
    elif "perplexity" in data or "ppl" in data:
        entry = _extract_ppl_entry(data, timestamp, hardware, filename)
        if entry:
            entries.append(entry)

    return entries


def _extract_entry(
    r: dict[str, Any], timestamp: str, hardware: str, filename: str
) -> dict[str, Any] | None:
    """Extract a single benchmark entry from result dict."""
    if not r:
        return None

    # Parse label/model name
    label = r.get("label", r.get("name", r.get("model", filename)))
    model = _extract_model_name(label)
    config = _extract_config(label, r)
    quant_type = _extract_quant_type(config, r)
    group_size = r.get("group_size", _extract_group_size(config))
    calibration = r.get("calibration", _extract_calibration(filename, r))

    # Perplexity (may not be present in all benchmarks)
    ppl_base = r.get("ppl_base", r.get("baseline_ppl", 0.0))
    ppl_quant = r.get("ppl_quant", r.get("ppl", r.get("perplexity", 0.0)))
    if ppl_base > 0 and ppl_quant > 0:
        ppl_delta_pct = (ppl_quant - ppl_base) / ppl_base * 100
    else:
        ppl_delta_pct = r.get("ppl_delta_pct", 0.0)

    # Throughput
    tokens_per_sec = r.get("tokens_per_sec", r.get("tok_s", 0.0))
    tflops = r.get("tflops", 0.0)
    memory_gb_s = r.get("memory_gb_s", r.get("bandwidth_util_pct", 0.0) * 5.46)  # Approx M4 Max

    # Size
    model_size_gb = r.get("model_size_gb", r.get("size_gb", 0.0))
    compression_ratio = r.get("compression_ratio", 1.0)

    return {
        "model": model,
        "config": config,
        "quant_type": quant_type,
        "group_size": group_size,
        "calibration": calibration,
        "ppl_base": ppl_base,
        "ppl_quant": ppl_quant,
        "ppl_delta_pct": ppl_delta_pct,
        "tokens_per_sec": tokens_per_sec,
        "tflops": tflops,
        "memory_gb_s": memory_gb_s,
        "model_size_gb": model_size_gb,
        "compression_ratio": compression_ratio,
        "timestamp": timestamp,
        "hardware": hardware,
    }


def _extract_ppl_entry(
    data: dict[str, Any], timestamp: str, hardware: str, filename: str
) -> dict[str, Any] | None:
    """Extract perplexity-focused benchmark entry."""
    model = data.get("model", data.get("model_path", _extract_model_name(filename)))
    ppl = data.get("perplexity", data.get("ppl", 0.0))
    ppl_base = data.get("baseline_ppl", data.get("fp16_ppl", 0.0))

    if ppl_base > 0:
        ppl_delta_pct = (ppl - ppl_base) / ppl_base * 100
    else:
        ppl_delta_pct = 0.0

    return {
        "model": model,
        "config": data.get("config", "default"),
        "quant_type": data.get("quant_type", "fp4"),
        "group_size": data.get("group_size", 128),
        "calibration": data.get("calibration", "wikitext2"),
        "ppl_base": ppl_base,
        "ppl_quant": ppl,
        "ppl_delta_pct": ppl_delta_pct,
        "tokens_per_sec": data.get("tokens_per_sec", 0.0),
        "tflops": data.get("tflops", 0.0),
        "memory_gb_s": data.get("memory_gb_s", 0.0),
        "model_size_gb": data.get("model_size_gb", 0.0),
        "compression_ratio": data.get("compression_ratio", 1.0),
        "timestamp": timestamp,
        "hardware": hardware,
    }


def _extract_model_name(label: str) -> str:
    """Extract model name from label string."""
    # Common patterns: "Llama-2-7B-FP4", "mistral-7b-fp4-g128"
    label = label.lower()
    for model in ["llama-2-7b", "llama-2-13b", "llama-3-8b", "llama-3-70b",
                  "mistral-7b", "mixtral-8x7b", "glm-4", "qwen", "phi-3"]:
        if model in label:
            return model.title()
    return label.split("-")[0].title() if "-" in label else label.title()


def _extract_config(label: str, r: dict[str, Any]) -> str:
    """Extract config identifier (e.g., fp4-g128) from label or result."""
    if "config" in r:
        return str(r["config"])

    label = label.lower()
    parts = []

    # Quant type
    if "fp4" in label:
        parts.append("fp4")
    elif "int4" in label or "u4" in label:
        parts.append("int4")
    elif "fp8" in label:
        parts.append("fp8")
    elif "fp16" in label:
        parts.append("fp16")

    # Group size
    if "g32" in label or "g-32" in label:
        parts.append("g32")
    elif "g64" in label or "g-64" in label:
        parts.append("g64")
    elif "g128" in label or "g-128" in label:
        parts.append("g128")
    elif "g256" in label or "g-256" in label:
        parts.append("g256")

    # Mixed vs uniform
    if "mixed" in label:
        parts.append("mixed")
    elif "uniform" in label:
        parts.append("uniform")

    return "-".join(parts) if parts else "default"


def _extract_quant_type(config: str, r: dict[str, Any]) -> str:
    """Extract quantization type from config string or result."""
    if "quant_type" in r:
        return str(r["quant_type"])

    config = config.lower()
    if "fp4" in config:
        return "fp4"
    elif "int4" in config or "u4" in config:
        return "int4"
    elif "fp8" in config:
        return "fp8"
    elif "fp16" in config:
        return "fp16"
    return "fp4"


def _extract_group_size(config: str) -> int:
    """Extract group size from config string."""
    import re

    match = re.search(r"g(\d+)", config.lower())
    if match:
        return int(match.group(1))
    return 128  # Default


def _extract_calibration(filename: str, r: dict[str, Any]) -> str:
    """Extract calibration dataset from filename or result."""
    if "calibration" in r:
        return str(r["calibration"])

    fname = filename.lower()
    if "bartowski" in fname or "v3" in fname:
        return "bartowski_v3"
    elif "wikitext" in fname or "wiki" in fname:
        return "wikitext2"
    return "none"


def generate_markdown_report(df: pd.DataFrame) -> str:
    """
    Generate markdown report from aggregated benchmark DataFrame.

    Includes:
    - Summary table with best configs per model
    - Per-model detailed breakdown
    - Comparison: uniform vs mixed-precision
    - Comparison: WikiText-2 vs Bartowski v3 calibration

    Args:
        df: DataFrame from aggregate_results()

    Returns:
        Markdown string for RESULTS.md
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required for generate_markdown_report")

    if df.empty:
        return "# Benchmark Results\n\nNo results found.\n"

    lines: list[str] = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append("# Metal Marlin Benchmark Results")
    lines.append("")
    lines.append(f"*Generated: {timestamp}*")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.extend(_generate_summary_table(df))
    lines.append("")

    # Per-model breakdown
    lines.append("## Per-Model Results")
    lines.append("")
    lines.extend(_generate_model_breakdown(df))
    lines.append("")

    # Uniform vs mixed-precision comparison
    lines.append("## Uniform vs Mixed-Precision")
    lines.append("")
    lines.extend(_generate_precision_comparison(df))
    lines.append("")

    # Calibration comparison
    lines.append("## Calibration Dataset Comparison")
    lines.append("")
    lines.extend(_generate_calibration_comparison(df))
    lines.append("")

    # Methodology notes
    lines.append("## Methodology")
    lines.append("")
    lines.append("- **Perplexity**: WikiText-2 test set, 512 token context")
    lines.append("- **Throughput**: Mean of 100 iterations after 15 warmup, outliers removed")
    lines.append("- **Compression ratio**: Original FP16 size / Quantized size")
    lines.append("- **Hardware**: Apple M4 Max (32 TFLOPS, 546 GB/s memory bandwidth)")
    lines.append("")

    return "\n".join(lines)


def _generate_summary_table(df: pd.DataFrame) -> list[str]:
    """Generate summary table with best config per model."""
    lines: list[str] = []

    # Group by model, find best config (lowest PPL delta with good compression)
    if df.empty:
        return ["No data available."]

    # Header
    lines.append("| Model | Best Config | PPL Delta % | Compression | Tokens/s |")
    lines.append("|-------|-------------|-------------|-------------|----------|")

    models = df["model"].unique()
    for model in sorted(models):
        model_df = df[df["model"] == model]
        if model_df.empty:
            continue

        # Find best: lowest PPL delta among configs with compression > 2x
        compressed = model_df[model_df["compression_ratio"] >= 2.0]
        if compressed.empty:
            compressed = model_df

        if "ppl_delta_pct" in compressed.columns:
            best_idx = compressed["ppl_delta_pct"].idxmin()
            if pd.notna(best_idx):
                best = compressed.loc[best_idx]
                lines.append(
                    f"| {model} | {best['config']} | "
                    f"{best['ppl_delta_pct']:.2f}% | "
                    f"{best['compression_ratio']:.1f}x | "
                    f"{best['tokens_per_sec']:.0f} |"
                )

    return lines


def _generate_model_breakdown(df: pd.DataFrame) -> list[str]:
    """Generate detailed per-model tables."""
    lines: list[str] = []

    models = df["model"].unique()
    for model in sorted(models):
        model_df = df[df["model"] == model].copy()
        if model_df.empty:
            continue

        lines.append(f"### {model}")
        lines.append("")
        lines.append("| Config | Quant | Group | PPL Base | PPL Quant | Delta % | Compression | TFLOPS |")
        lines.append("|--------|-------|-------|----------|-----------|---------|-------------|--------|")

        # Sort by PPL delta
        if "ppl_delta_pct" in model_df.columns:
            model_df = model_df.sort_values("ppl_delta_pct")

        for _, row in model_df.iterrows():
            lines.append(
                f"| {row['config']} | {row['quant_type']} | {row['group_size']} | "
                f"{row['ppl_base']:.2f} | {row['ppl_quant']:.2f} | "
                f"{row['ppl_delta_pct']:.2f}% | "
                f"{row['compression_ratio']:.1f}x | {row['tflops']:.2f} |"
            )

        lines.append("")

    return lines


def _generate_precision_comparison(df: pd.DataFrame) -> list[str]:
    """Compare uniform vs mixed-precision configs."""
    lines: list[str] = []

    # Filter to uniform and mixed configs
    uniform = df[df["config"].str.contains("uniform", case=False, na=False)]
    mixed = df[df["config"].str.contains("mixed", case=False, na=False)]

    if uniform.empty and mixed.empty:
        lines.append("No uniform/mixed-precision comparison data available.")
        lines.append("")
        lines.append("This comparison requires benchmark results with configs labeled as 'uniform' or 'mixed'.")
        return lines

    lines.append("| Model | Uniform PPL Δ% | Mixed PPL Δ% | Improvement |")
    lines.append("|-------|----------------|--------------|-------------|")

    models = set(uniform["model"].unique()) & set(mixed["model"].unique())
    for model in sorted(models):
        u_row = uniform[uniform["model"] == model].iloc[0] if not uniform[uniform["model"] == model].empty else None
        m_row = mixed[mixed["model"] == model].iloc[0] if not mixed[mixed["model"] == model].empty else None

        if u_row is not None and m_row is not None:
            improvement = u_row["ppl_delta_pct"] - m_row["ppl_delta_pct"]
            lines.append(
                f"| {model} | {u_row['ppl_delta_pct']:.2f}% | "
                f"{m_row['ppl_delta_pct']:.2f}% | {improvement:+.2f}% |"
            )

    if not models:
        lines.append("| (no matching models) | - | - | - |")

    return lines


def _generate_calibration_comparison(df: pd.DataFrame) -> list[str]:
    """Compare WikiText-2 vs Bartowski v3 calibration."""
    lines: list[str] = []

    wiki = df[df["calibration"].str.contains("wikitext", case=False, na=False)]
    bart = df[df["calibration"].str.contains("bartowski", case=False, na=False)]

    if wiki.empty and bart.empty:
        lines.append("No calibration comparison data available.")
        lines.append("")
        lines.append("This comparison requires benchmark results with different calibration datasets.")
        return lines

    lines.append("| Model | Config | WikiText-2 PPL Δ% | Bartowski v3 PPL Δ% | Difference |")
    lines.append("|-------|--------|-------------------|---------------------|------------|")

    # Find matching model/config pairs
    wiki_configs = set(zip(wiki["model"], wiki["config"]))
    bart_configs = set(zip(bart["model"], bart["config"]))
    common = wiki_configs & bart_configs

    for model, config in sorted(common):
        w_row = wiki[(wiki["model"] == model) & (wiki["config"] == config)].iloc[0]
        b_row = bart[(bart["model"] == model) & (bart["config"] == config)].iloc[0]

        diff = b_row["ppl_delta_pct"] - w_row["ppl_delta_pct"]
        lines.append(
            f"| {model} | {config} | {w_row['ppl_delta_pct']:.2f}% | "
            f"{b_row['ppl_delta_pct']:.2f}% | {diff:+.2f}% |"
        )

    if not common:
        lines.append("| (no matching configs) | - | - | - | - |")

    return lines


def plot_ppl_vs_compression(df: pd.DataFrame, output: str | Path) -> None:
    """
    Create scatter plot of PPL delta % vs compression ratio.

    X-axis: Compression ratio (higher = smaller model)
    Y-axis: PPL delta % (lower = better quality)

    Points colored by quantization type (FP4, INT4, FP8).
    Point size scaled by throughput.

    Args:
        df: DataFrame from aggregate_results()
        output: Path for output PNG file
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib required for plot_ppl_vs_compression. Install with: pip install matplotlib"
        )

    if df.empty:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Color map for quant types
    colors = {"fp4": "#2196F3", "int4": "#4CAF50", "fp8": "#FF9800", "fp16": "#9E9E9E"}

    quant_types = df["quant_type"].unique()
    for qtype in quant_types:
        subset = df[df["quant_type"] == qtype]
        color = colors.get(qtype, "#757575")

        # Scale size by tokens/sec (normalize to 50-200 range)
        sizes = subset["tokens_per_sec"]
        if sizes.max() > 0:
            sizes = 50 + (sizes / sizes.max()) * 150
        else:
            sizes = np.full(len(subset), 100)

        ax.scatter(
            subset["compression_ratio"],
            subset["ppl_delta_pct"],
            c=color,
            s=sizes,
            alpha=0.7,
            label=qtype.upper(),
            edgecolors="white",
            linewidth=0.5,
        )

        # Add model labels for notable points
        for _, row in subset.iterrows():
            if abs(row["ppl_delta_pct"]) > 5 or row["compression_ratio"] > 3.5:
                ax.annotate(
                    row["model"],
                    (row["compression_ratio"], row["ppl_delta_pct"]),
                    fontsize=8,
                    alpha=0.7,
                    xytext=(5, 5),
                    textcoords="offset points",
                )

    ax.set_xlabel("Compression Ratio (×)", fontsize=12)
    ax.set_ylabel("Perplexity Delta (%)", fontsize=12)
    ax.set_title("Quality vs Compression Tradeoff", fontsize=14, fontweight="bold")

    # Ideal zone annotation
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1, color="green", linestyle=":", alpha=0.5, label="1% degradation")
    ax.axhline(y=5, color="orange", linestyle=":", alpha=0.5, label="5% degradation")

    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Set reasonable limits
    ax.set_xlim(left=1)
    y_max = min(df["ppl_delta_pct"].max() * 1.1, 20)
    y_min = max(df["ppl_delta_pct"].min() * 1.1, -2)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {output}")


def plot_throughput_comparison(df: pd.DataFrame, output: str | Path) -> None:
    """
    Create bar chart comparing throughput by model and config.

    Groups bars by model, with different colors for each config/quant type.

    Args:
        df: DataFrame from aggregate_results()
        output: Path for output PNG file
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib required for plot_throughput_comparison. Install with: pip install matplotlib"
        )

    if df.empty:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    models = sorted(df["model"].unique())
    configs = sorted(df["config"].unique())

    x = np.arange(len(models))
    width = 0.8 / max(len(configs), 1)

    colors = plt.cm.Set2(np.linspace(0, 1, len(configs)))

    for i, config in enumerate(configs):
        throughputs = []
        for model in models:
            subset = df[(df["model"] == model) & (df["config"] == config)]
            if not subset.empty:
                throughputs.append(subset["tokens_per_sec"].mean())
            else:
                throughputs.append(0)

        offset = (i - len(configs) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            throughputs,
            width,
            label=config,
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
        )

        # Add value labels on bars
        for bar, val in zip(bars, throughputs):
            if val > 0:
                ax.annotate(
                    f"{val:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Tokens/sec", fontsize=12)
    ax.set_title("Throughput by Model and Configuration", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {output}")


def generate_results_file(results_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    """
    Generate complete RESULTS.md with tables and plots.

    Args:
        results_dir: Directory containing benchmark JSON files
        output_dir: Output directory for RESULTS.md and plots (defaults to benchmarks/)

    Returns:
        Path to generated RESULTS.md
    """
    results_dir = Path(results_dir)
    if output_dir is None:
        output_dir = results_dir.parent if results_dir.name == "results" else results_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate results
    df = aggregate_results(results_dir)

    if df.empty:
        print(f"No benchmark results found in {results_dir}")
        # Still generate empty template
        md_content = "# Metal Marlin Benchmark Results\n\nNo benchmark results found.\n"
        md_content += "\n## How to Generate Results\n\n"
        md_content += "1. Run benchmarks:\n"
        md_content += "   ```bash\n"
        md_content += "   uv run python -m benchmarks.bench_comparison\n"
        md_content += "   ```\n\n"
        md_content += "2. Regenerate this report:\n"
        md_content += "   ```bash\n"
        md_content += "   uv run python -m metal_marlin.benchmark_report generate ./benchmarks/results/\n"
        md_content += "   ```\n"
    else:
        # Generate markdown report
        md_content = generate_markdown_report(df)

        # Generate plots if matplotlib available
        if HAS_MATPLOTLIB:
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            try:
                plot_ppl_vs_compression(df, plots_dir / "ppl_vs_compression.png")
                md_content += "\n## Plots\n\n"
                md_content += "![PPL vs Compression](plots/ppl_vs_compression.png)\n\n"
            except Exception as e:
                print(f"Warning: Could not generate PPL plot: {e}")

            try:
                plot_throughput_comparison(df, plots_dir / "throughput_comparison.png")
                md_content += "![Throughput Comparison](plots/throughput_comparison.png)\n"
            except Exception as e:
                print(f"Warning: Could not generate throughput plot: {e}")

    # Write RESULTS.md
    results_file = output_dir / "RESULTS.md"
    results_file.write_text(md_content)
    print(f"Generated: {results_file}")

    return results_file


# ============================================================================
# CLI
# ============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark result aggregation and reporting")
    subparsers = parser.add_subparsers(dest="command")

    # Generate command
    gen_p = subparsers.add_parser("generate", help="Generate RESULTS.md from benchmark JSONs")
    gen_p.add_argument("results_dir", nargs="?", default=str(_RESULTS_DIR),
                       help="Directory containing benchmark JSON files")
    gen_p.add_argument("-o", "--output", help="Output directory for RESULTS.md")

    # Plot PPL command
    ppl_p = subparsers.add_parser("plot-ppl", help="Generate PPL vs compression plot")
    ppl_p.add_argument("results_dir", nargs="?", default=str(_RESULTS_DIR))
    ppl_p.add_argument("-o", "--output", default="ppl_vs_compression.png")

    # Plot throughput command
    tput_p = subparsers.add_parser("plot-throughput", help="Generate throughput comparison plot")
    tput_p.add_argument("results_dir", nargs="?", default=str(_RESULTS_DIR))
    tput_p.add_argument("-o", "--output", default="throughput_comparison.png")

    # Aggregate command (for debugging/exploration)
    agg_p = subparsers.add_parser("aggregate", help="Aggregate results and print DataFrame")
    agg_p.add_argument("results_dir", nargs="?", default=str(_RESULTS_DIR))
    agg_p.add_argument("--csv", help="Export to CSV file")

    args = parser.parse_args()

    if args.command == "generate":
        generate_results_file(args.results_dir, args.output)

    elif args.command == "plot-ppl":
        df = aggregate_results(args.results_dir)
        plot_ppl_vs_compression(df, args.output)

    elif args.command == "plot-throughput":
        df = aggregate_results(args.results_dir)
        plot_throughput_comparison(df, args.output)

    elif args.command == "aggregate":
        df = aggregate_results(args.results_dir)
        if df.empty:
            print("No results found.")
        else:
            print(df.to_string())
            if args.csv:
                df.to_csv(args.csv, index=False)
                print(f"\nExported to {args.csv}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
