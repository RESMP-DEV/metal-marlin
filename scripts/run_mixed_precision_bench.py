#!/usr/bin/env python3
"""Runner script for mixed-precision MoE benchmarks.

This script provides a CLI for running and comparing different
MoE dispatch strategies on both synthetic and real models.

Usage:
    # Quick synthetic benchmark
    python run_mixed_precision_bench.py
    
    # Full benchmark with all strategies
    python run_mixed_precision_bench.py --full
    
    # Compare specific strategies
    python run_mixed_precision_bench.py --strategies slow_path fast_mixed hybrid fast_uniform
    
    # Save results to file
    python run_mixed_precision_bench.py --output results/bench_$(date +%Y%m%d).json
    
    # Use real GLM-4.7 model
    python run_mixed_precision_bench.py --model glm4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark mixed-precision MoE dispatch strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategies:
  slow_path        Sequential per-expert dispatch (baseline)
  fast_uniform     Batched dispatch with uniform bits (ideal target)
  fast_mixed       Batched dispatch with per-projection bits
  hybrid           Batched for common, sequential for rare
  max_bits_padded  Pad to max bits, single dispatch
  
Examples:
  %(prog)s                    Quick synthetic benchmark
  %(prog)s --full             Full benchmark with all strategies
  %(prog)s --strategies slow_path fast_mixed hybrid fast_uniform
  %(prog)s --model glm4       Benchmark real GLM-4.7 model
""",
    )
    
    # Model selection
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--model",
        choices=["synthetic", "glm4"],
        default="synthetic",
        help="Model to benchmark (default: synthetic)",
    )
    model_group.add_argument(
        "--model-path",
        type=str,
        default="TheLongMind/GLM-4.7-Flash-Trellis-MM",
        help="HuggingFace model path for glm4 (default: %(default)s)",
    )
    
    # Benchmark configuration
    bench_group = parser.add_argument_group("Benchmark Options")
    bench_group.add_argument(
        "--strategies",
        nargs="+",
        choices=["slow_path", "fast_uniform", "fast_mixed", "hybrid", "max_bits_padded"],
        default=["slow_path", "fast_mixed", "hybrid", "fast_uniform", "max_bits_padded"],
        help="Strategies to benchmark (default: slow_path fast_mixed hybrid fast_uniform max_bits_padded)",
    )
    bench_group.add_argument(
        "--full",
        action="store_true",
        help="Run all strategies (overrides --strategies)",
    )
    bench_group.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations (default: 3)",
    )
    bench_group.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Timed iterations (default: 10)",
    )
    bench_group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    bench_group.add_argument(
        "--seq-len",
        type=int,
        default=1,
        help="Sequence length (default: 1)",
    )
    
    # Synthetic model options
    synth_group = parser.add_argument_group("Synthetic Model Options")
    synth_group.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension (default: 512)",
    )
    synth_group.add_argument(
        "--intermediate-dim",
        type=int,
        default=1408,
        help="Intermediate dimension (default: 1408)",
    )
    synth_group.add_argument(
        "--num-experts",
        type=int,
        default=8,
        help="Number of experts (default: 8)",
    )
    synth_group.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Top-k experts (default: 2)",
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output",
        "-o",
        type=str,
        help="Save results to JSON file",
    )
    output_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output",
    )
    output_group.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use (default: mps)",
    )
    
    return parser.parse_args()


def load_synthetic_model(args: argparse.Namespace) -> torch.nn.Module:
    """Load synthetic model for benchmark."""
    from tests.fixtures.synthetic_mixed_moe import (
        SyntheticConfig,
        create_synthetic_model,
    )
    
    config = SyntheticConfig(
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )
    
    return create_synthetic_model(config=config, device=args.device)


def load_glm4_model(args: argparse.Namespace) -> torch.nn.Module:
    """Load real GLM-4.7 model for benchmark."""
    from metal_marlin.metal_core import get_device_info
    from metal_marlin.shards.hf_shard_loader import HFShardLoader
    from metal_marlin.trellis.model import TrellisModel
    
    print(f"Loading GLM-4.7 from: {args.model_path}")
    device_info = get_device_info()
    print(f"Using device: {device_info.get('device_name', 'unknown')}")
    print(f"Available memory: {device_info.get('memory_gb', 0):.1f} GB")
    
    # Load model
    loader = HFShardLoader(args.model_path)
    model = TrellisModel.from_pretrained(loader)
    model.to(args.device)
    
    return model


def run_benchmark(args: argparse.Namespace) -> None:
    """Run benchmark with given configuration."""
    from benchmarks.mixed_precision_bench import (
        BenchmarkConfig,
        MixedPrecisionBenchmark,
    )
    
    # Load model
    if not args.quiet:
        print("=" * 60)
        print("Mixed-Precision MoE Benchmark")
        print("=" * 60)
        print()
        
    if args.model == "synthetic":
        model = load_synthetic_model(args)
        if not args.quiet:
            print(f"Model: Synthetic (hidden={args.hidden_dim}, experts={args.num_experts})")
    else:
        model = load_glm4_model(args)
        if not args.quiet:
            print(f"Model: GLM-4.7 ({args.model_path})")
            
    # Get bit distribution if available
    if hasattr(model, "get_bit_distribution"):
        dist = model.get_bit_distribution()
        if not args.quiet:
            print(f"Bit tuples: {dist['bit_tuple_counts']}")
            
    # Configure benchmark
    strategies = (
        ["slow_path", "fast_uniform", "fast_mixed", "hybrid", "max_bits_padded"]
        if args.full
        else args.strategies
    )
    
    config = BenchmarkConfig(
        warmup=args.warmup,
        iterations=args.iterations,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        strategies=strategies,
    )
    
    if not args.quiet:
        print(f"\nConfig: {args.warmup} warmup, {args.iterations} iterations")
        print(f"Input: batch={args.batch_size}, seq_len={args.seq_len}")
        print(f"Strategies: {', '.join(strategies)}")
        print()
        
    # Run benchmark
    bench = MixedPrecisionBenchmark(model, config, device=args.device)
    
    start_time = time.time()
    results = bench.compare_all()
    elapsed = time.time() - start_time
    
    # Print results
    bench.print_results(results)
    
    if not args.quiet:
        print(f"\nTotal time: {elapsed:.1f}s")
        
    # Save results
    if args.output:
        bench.save_results(results, args.output)
        
    return results


def main() -> int:
    args = parse_args()
    
    try:
        run_benchmark(args)
        return 0
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        if "--" not in sys.argv or args.quiet:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
