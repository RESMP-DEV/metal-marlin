#!/usr/bin/env python3
"""
Metal Kernel Optimization Loop (AlphaHENG-powered)

Entropy-First Optimization Strategy
====================================
Metal/Apple Silicon has sparse representation in LLM training data compared to CUDA.
"Known optimal" patterns from CUDA often hurt Metal performance (e.g., tile_m_32 gave
0.122x on Metal despite being standard CUDA advice for small batches).

This script defaults to ENTROPY INJECTION: random parameter sampling + Metal-specific
exploratory patterns, using hardware benchmarks as the oracle to filter hypotheses.
This approach discovered exp_simdgroups_1 (1.895x speedup) - a pattern that inverts
CUDA wisdom but works because Apple Silicon's architecture differs fundamentally.

Each optimization variant is a separate task dispatched through AlphaHENG's agent swarm:
1. Apply transformation to kernel (random, exploratory, or deterministic)
2. Benchmark on real hardware
3. Let measurements filter what works

Usage:
    # Default: entropy-first with 20 random + exploratory patterns
    cd contrib/metal_marlin && uv run python scripts/optimize_kernel.py src/marlin_gemm.metal

    # Deterministic-only (disable entropy injection)
    cd contrib/metal_marlin && uv run python scripts/optimize_kernel.py src/marlin_gemm.metal --no-explore

    # More entropy for thorough search
    cd contrib/metal_marlin && uv run python scripts/optimize_kernel.py src/marlin_gemm.metal --num-random 50

    # Run local benchmark without AlphaHENG
    cd contrib/metal_marlin && uv run python scripts/optimize_kernel.py --autotune

    # Collect results after swarm completes
    cd contrib/metal_marlin && uv run python scripts/optimize_kernel.py --collect-results SESSION_ID
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Resolve paths relative to this script (standalone-compatible)
_SCRIPT_DIR = Path(__file__).parent
METAL_MARLIN_ROOT = _SCRIPT_DIR.parent

# Only resolve AlphaHENG root if running from within the monorepo
_POTENTIAL_ALPHAHENG_ROOT = METAL_MARLIN_ROOT.parent.parent
ALPHAHENG_ROOT = (
    _POTENTIAL_ALPHAHENG_ROOT if (_POTENTIAL_ALPHAHENG_ROOT / "alphaheng").is_dir() else None
)

# ============================================================================
# DETERMINISTIC PATTERNS - Well-known configurations to test
# ============================================================================
# These are designed to match kernel configuration options in src/*.metal
# Metal uses `constant constexpr uint` syntax for compile-time constants
OPTIMIZATION_PATTERNS: list[dict[str, Any]] = [
    # Tile size variants - aligned with KernelConfig options
    {
        "name": "tile_m_32",
        "description": "Set TILE_M=32 (small batches)",
        "pattern": r"constant\s+constexpr\s+uint\s+TILE_M\s*=\s*\d+",
        "replacement": "constant constexpr uint TILE_M = 32",
        "applicable_to": ["gemm", "attention"],
        "priority": "P1",
    },
    {
        "name": "tile_m_64",
        "description": "Set TILE_M=64 (balanced default)",
        "pattern": r"constant\s+constexpr\s+uint\s+TILE_M\s*=\s*\d+",
        "replacement": "constant constexpr uint TILE_M = 64",
        "applicable_to": ["gemm", "attention"],
        "priority": "P1",
    },
    {
        "name": "tile_m_128",
        "description": "Set TILE_M=128 (large batches)",
        "pattern": r"constant\s+constexpr\s+uint\s+TILE_M\s*=\s*\d+",
        "replacement": "constant constexpr uint TILE_M = 128",
        "applicable_to": ["gemm", "attention"],
        "priority": "P1",
    },
    {
        "name": "tile_n_32",
        "description": "Set TILE_N=32",
        "pattern": r"constant\s+constexpr\s+uint\s+TILE_N\s*=\s*\d+",
        "replacement": "constant constexpr uint TILE_N = 32",
        "applicable_to": ["gemm"],
        "priority": "P1",
    },
    {
        "name": "tile_n_64",
        "description": "Set TILE_N=64 (balanced default)",
        "pattern": r"constant\s+constexpr\s+uint\s+TILE_N\s*=\s*\d+",
        "replacement": "constant constexpr uint TILE_N = 64",
        "applicable_to": ["gemm"],
        "priority": "P1",
    },
    {
        "name": "tile_n_128",
        "description": "Set TILE_N=128 (large output features)",
        "pattern": r"constant\s+constexpr\s+uint\s+TILE_N\s*=\s*\d+",
        "replacement": "constant constexpr uint TILE_N = 128",
        "applicable_to": ["gemm"],
        "priority": "P1",
    },
    {
        "name": "tile_k_16",
        "description": "Set TILE_K=16 (memory bound)",
        "pattern": r"constant\s+constexpr\s+uint\s+TILE_K\s*=\s*\d+",
        "replacement": "constant constexpr uint TILE_K = 16",
        "applicable_to": ["gemm"],
        "priority": "P1",
    },
    {
        "name": "tile_k_32",
        "description": "Set TILE_K=32 (balanced default)",
        "pattern": r"constant\s+constexpr\s+uint\s+TILE_K\s*=\s*\d+",
        "replacement": "constant constexpr uint TILE_K = 32",
        "applicable_to": ["gemm"],
        "priority": "P1",
    },
    {
        "name": "tile_k_64",
        "description": "Set TILE_K=64 (compute bound)",
        "pattern": r"constant\s+constexpr\s+uint\s+TILE_K\s*=\s*\d+",
        "replacement": "constant constexpr uint TILE_K = 64",
        "applicable_to": ["gemm"],
        "priority": "P1",
    },
    # Simdgroup configurations
    {
        "name": "simdgroups_2",
        "description": "Set SIMDGROUPS_PER_TG=2",
        "pattern": r"constant\s+constexpr\s+uint\s+SIMDGROUPS_PER_TG\s*=\s*\d+",
        "replacement": "constant constexpr uint SIMDGROUPS_PER_TG = 2",
        "applicable_to": ["gemm", "attention"],
        "priority": "P1",
    },
    {
        "name": "simdgroups_4",
        "description": "Set SIMDGROUPS_PER_TG=4 (balanced)",
        "pattern": r"constant\s+constexpr\s+uint\s+SIMDGROUPS_PER_TG\s*=\s*\d+",
        "replacement": "constant constexpr uint SIMDGROUPS_PER_TG = 4",
        "applicable_to": ["gemm", "attention"],
        "priority": "P1",
    },
    {
        "name": "simdgroups_8",
        "description": "Set SIMDGROUPS_PER_TG=8",
        "pattern": r"constant\s+constexpr\s+uint\s+SIMDGROUPS_PER_TG\s*=\s*\d+",
        "replacement": "constant constexpr uint SIMDGROUPS_PER_TG = 8",
        "applicable_to": ["gemm", "attention"],
        "priority": "P1",
    },
    # Simd tile decomposition - affects occupancy and register pressure
    {
        "name": "sg_m_tiles_2",
        "description": "Set SG_M_TILES=2 (fewer M rows per simdgroup)",
        "pattern": r"constant\s+constexpr\s+uint\s+SG_M_TILES\s*=\s*\d+",
        "replacement": "constant constexpr uint SG_M_TILES = 2",
        "applicable_to": ["gemm"],
        "priority": "P2",
    },
    {
        "name": "sg_m_tiles_4",
        "description": "Set SG_M_TILES=4 (default)",
        "pattern": r"constant\s+constexpr\s+uint\s+SG_M_TILES\s*=\s*\d+",
        "replacement": "constant constexpr uint SG_M_TILES = 4",
        "applicable_to": ["gemm"],
        "priority": "P2",
    },
    {
        "name": "sg_m_tiles_8",
        "description": "Set SG_M_TILES=8 (more M rows per simdgroup)",
        "pattern": r"constant\s+constexpr\s+uint\s+SG_M_TILES\s*=\s*\d+",
        "replacement": "constant constexpr uint SG_M_TILES = 8",
        "applicable_to": ["gemm"],
        "priority": "P2",
    },
    {
        "name": "sg_n_tiles_2",
        "description": "Set SG_N_TILES=2",
        "pattern": r"constant\s+constexpr\s+uint\s+SG_N_TILES\s*=\s*\d+",
        "replacement": "constant constexpr uint SG_N_TILES = 2",
        "applicable_to": ["gemm"],
        "priority": "P2",
    },
    {
        "name": "sg_n_tiles_4",
        "description": "Set SG_N_TILES=4 (default)",
        "pattern": r"constant\s+constexpr\s+uint\s+SG_N_TILES\s*=\s*\d+",
        "replacement": "constant constexpr uint SG_N_TILES = 4",
        "applicable_to": ["gemm"],
        "priority": "P2",
    },
    {
        "name": "sg_n_tiles_8",
        "description": "Set SG_N_TILES=8",
        "pattern": r"constant\s+constexpr\s+uint\s+SG_N_TILES\s*=\s*\d+",
        "replacement": "constant constexpr uint SG_N_TILES = 8",
        "applicable_to": ["gemm"],
        "priority": "P2",
    },
    # Buffer count for async pipelining
    {
        "name": "num_buffers_2",
        "description": "Use double buffering (NUM_BUFFERS=2)",
        "pattern": r"constant\s+constexpr\s+uint\s+NUM_BUFFERS\s*=\s*\d+",
        "replacement": "constant constexpr uint NUM_BUFFERS = 2",
        "applicable_to": ["gemm"],
        "priority": "P2",
    },
    {
        "name": "num_buffers_3",
        "description": "Use triple buffering (NUM_BUFFERS=3)",
        "pattern": r"constant\s+constexpr\s+uint\s+NUM_BUFFERS\s*=\s*\d+",
        "replacement": "constant constexpr uint NUM_BUFFERS = 3",
        "applicable_to": ["gemm"],
        "priority": "P2",
    },
]

# ============================================================================
# EXPLORATORY PATTERNS - High-variance experiments for Metal-specific discovery
# ============================================================================
# Metal GPUs (Apple Silicon) have VERY different optimal patterns than CUDA:
# - Unified memory architecture (no explicit host/device transfers)
# - Tile-based deferred rendering architecture in GPU cores
# - simdgroup_matrix intrinsics with specific sizing requirements
# - Different occupancy characteristics (threads_per_simdgroup fixed at 32)
# - Register file organization differs from CUDA warps
#
# These patterns explore non-obvious optimizations that may outperform
# "conventional wisdom" from CUDA kernels.

METAL_EXPLORATORY_PATTERNS: list[dict[str, Any]] = [
    # Unconventional tile ratios that exploit Metal's tile memory
    {
        "name": "exp_asymmetric_tiles_wide",
        "description": "TILE_M=32, TILE_N=128 - wide output for memory coalescing",
        "multi_pattern": [
            (
                r"constant\s+constexpr\s+uint\s+TILE_M\s*=\s*\d+",
                "constant constexpr uint TILE_M = 32",
            ),
            (
                r"constant\s+constexpr\s+uint\s+TILE_N\s*=\s*\d+",
                "constant constexpr uint TILE_N = 128",
            ),
        ],
        "applicable_to": ["gemm"],
        "priority": "P2",
        "hypothesis": "Metal's tile memory may favor wide output tiles for better cache line utilization",
    },
    {
        "name": "exp_asymmetric_tiles_tall",
        "description": "TILE_M=128, TILE_N=32 - tall tiles for batch dimension",
        "multi_pattern": [
            (
                r"constant\s+constexpr\s+uint\s+TILE_M\s*=\s*\d+",
                "constant constexpr uint TILE_M = 128",
            ),
            (
                r"constant\s+constexpr\s+uint\s+TILE_N\s*=\s*\d+",
                "constant constexpr uint TILE_N = 32",
            ),
        ],
        "applicable_to": ["gemm"],
        "priority": "P2",
        "hypothesis": "Tall tiles may improve register reuse for batched operations",
    },
    # Non-power-of-2 experiments (Metal doesn't require power-of-2 as strictly as CUDA)
    {
        "name": "exp_tile_k_48",
        "description": "TILE_K=48 - non-power-of-2 K dimension",
        "pattern": r"constant\s+constexpr\s+uint\s+TILE_K\s*=\s*\d+",
        "replacement": "constant constexpr uint TILE_K = 48",
        "applicable_to": ["gemm"],
        "priority": "P2",
        "hypothesis": "Non-power-of-2 may reduce bank conflicts in threadgroup memory",
    },
    {
        "name": "exp_tile_k_24",
        "description": "TILE_K=24 - smaller non-power-of-2 K",
        "pattern": r"constant\s+constexpr\s+uint\s+TILE_K\s*=\s*\d+",
        "replacement": "constant constexpr uint TILE_K = 24",
        "applicable_to": ["gemm"],
        "priority": "P2",
        "hypothesis": "May allow more occupancy with reduced register pressure",
    },
    # Simdgroup density experiments
    {
        "name": "exp_simdgroups_1",
        "description": "Single simdgroup per threadgroup - minimal footprint",
        "pattern": r"constant\s+constexpr\s+uint\s+SIMDGROUPS_PER_TG\s*=\s*\d+",
        "replacement": "constant constexpr uint SIMDGROUPS_PER_TG = 1",
        "applicable_to": ["gemm", "attention"],
        "priority": "P2",
        "hypothesis": "Single simdgroup may maximize occupancy through more concurrent threadgroups",
    },
    {
        "name": "exp_simdgroups_16",
        "description": "16 simdgroups per threadgroup - maximum parallelism",
        "pattern": r"constant\s+constexpr\s+uint\s+SIMDGROUPS_PER_TG\s*=\s*\d+",
        "replacement": "constant constexpr uint SIMDGROUPS_PER_TG = 16",
        "applicable_to": ["gemm", "attention"],
        "priority": "P2",
        "hypothesis": "High simdgroup count may better utilize ALU units on M3 Pro/Max",
    },
    # Apple-specific: simdgroup_matrix 8x8 vs custom tiling
    {
        "name": "exp_sg_tiles_1x1",
        "description": "SG_M_TILES=1, SG_N_TILES=1 - minimal per-simdgroup work",
        "multi_pattern": [
            (
                r"constant\s+constexpr\s+uint\s+SG_M_TILES\s*=\s*\d+",
                "constant constexpr uint SG_M_TILES = 1",
            ),
            (
                r"constant\s+constexpr\s+uint\s+SG_N_TILES\s*=\s*\d+",
                "constant constexpr uint SG_N_TILES = 1",
            ),
        ],
        "applicable_to": ["gemm"],
        "priority": "P2",
        "hypothesis": "Minimal simdgroup work may enable more memory-level parallelism",
    },
    {
        "name": "exp_sg_tiles_8x8",
        "description": "SG_M_TILES=8, SG_N_TILES=8 - maximum per-simdgroup work",
        "multi_pattern": [
            (
                r"constant\s+constexpr\s+uint\s+SG_M_TILES\s*=\s*\d+",
                "constant constexpr uint SG_M_TILES = 8",
            ),
            (
                r"constant\s+constexpr\s+uint\s+SG_N_TILES\s*=\s*\d+",
                "constant constexpr uint SG_N_TILES = 8",
            ),
        ],
        "applicable_to": ["gemm"],
        "priority": "P2",
        "hypothesis": "Maximum register utilization may amortize memory latency",
    },
    # Quad-buffer for aggressive pipelining
    {
        "name": "exp_num_buffers_4",
        "description": "Quad buffering for aggressive async prefetch",
        "pattern": r"constant\s+constexpr\s+uint\s+NUM_BUFFERS\s*=\s*\d+",
        "replacement": "constant constexpr uint NUM_BUFFERS = 4",
        "applicable_to": ["gemm"],
        "priority": "P2",
        "hypothesis": "4-stage pipeline may better hide memory latency on unified memory",
    },
    # Extreme configurations to probe boundaries
    {
        "name": "exp_tiny_tiles",
        "description": "TILE_M=16, TILE_N=16, TILE_K=16 - very small tiles",
        "multi_pattern": [
            (
                r"constant\s+constexpr\s+uint\s+TILE_M\s*=\s*\d+",
                "constant constexpr uint TILE_M = 16",
            ),
            (
                r"constant\s+constexpr\s+uint\s+TILE_N\s*=\s*\d+",
                "constant constexpr uint TILE_N = 16",
            ),
            (
                r"constant\s+constexpr\s+uint\s+TILE_K\s*=\s*\d+",
                "constant constexpr uint TILE_K = 16",
            ),
        ],
        "applicable_to": ["gemm"],
        "priority": "P3",
        "hypothesis": "Tiny tiles = high occupancy, may win for small matrices",
    },
    {
        "name": "exp_large_tiles",
        "description": "TILE_M=128, TILE_N=128, TILE_K=64 - very large tiles",
        "multi_pattern": [
            (
                r"constant\s+constexpr\s+uint\s+TILE_M\s*=\s*\d+",
                "constant constexpr uint TILE_M = 128",
            ),
            (
                r"constant\s+constexpr\s+uint\s+TILE_N\s*=\s*\d+",
                "constant constexpr uint TILE_N = 128",
            ),
            (
                r"constant\s+constexpr\s+uint\s+TILE_K\s*=\s*\d+",
                "constant constexpr uint TILE_K = 64",
            ),
        ],
        "applicable_to": ["gemm"],
        "priority": "P3",
        "hypothesis": "Large tiles maximize compute per memory access for big matrices",
    },
]


def generate_random_variants(
    kernel_type: str,
    source: str,
    num_variants: int = 5,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate random parameter combinations for exploration.

    This injects entropy into the search space - randomly sampling from
    parameter ranges to discover non-obvious optima that wouldn't be found
    by predetermined patterns.
    """
    if seed is not None:
        random.seed(seed)

    # Parameter ranges to sample from
    param_ranges = {
        "TILE_M": [16, 24, 32, 48, 64, 96, 128],
        "TILE_N": [16, 24, 32, 48, 64, 96, 128],
        "TILE_K": [8, 16, 24, 32, 48, 64],
        "SIMDGROUPS_PER_TG": [1, 2, 4, 6, 8, 12, 16],
        "SG_M_TILES": [1, 2, 4, 8],
        "SG_N_TILES": [1, 2, 4, 8],
        "NUM_BUFFERS": [2, 3, 4],
    }

    variants = []
    seen_configs = set()

    for i in range(num_variants * 3):  # Generate extra to account for duplicates
        if len(variants) >= num_variants:
            break

        config = {}
        multi_pattern = []

        # Check which parameters exist in source and sample them
        for param, values in param_ranges.items():
            pattern = rf"constant\s+constexpr\s+uint\s+{param}\s*=\s*\d+"
            if re.search(pattern, source):
                value = random.choice(values)
                config[param] = value
                multi_pattern.append((pattern, f"constant constexpr uint {param} = {value}"))

        if not multi_pattern:
            continue

        # Create unique config key
        config_key = tuple(sorted(config.items()))
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)

        # Format name and description
        name = f"rand_{i:03d}_" + "_".join(
            f"{k[0].lower()}{v}" for k, v in sorted(config.items())[:3]
        )
        desc = ", ".join(f"{k}={v}" for k, v in sorted(config.items()))

        variants.append(
            {
                "name": name[:40],  # Limit name length
                "description": f"Random config: {desc}",
                "multi_pattern": multi_pattern,
                "applicable_to": [kernel_type],
                "priority": "P2",
                "hypothesis": "Randomly sampled to explore parameter space",
            }
        )

    return variants


@dataclass
class OptimizationVariant:
    """A single optimization variant to test."""

    name: str
    description: str
    kernel_path: str
    pattern: str
    replacement: str
    variant_source: str  # The modified source code
    priority: str = "P1"

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
    gflops: float | None = None
    bandwidth_gb_s: float | None = None
    error: str | None = None


@dataclass
class OptimizationSession:
    """Track an optimization session."""

    kernel_path: str
    kernel_type: str
    session_id: str
    start_time: str
    problem_sizes: list[tuple[int, int, int]] = field(default_factory=list)
    baseline_us: float | None = None
    variants: list[OptimizationVariant] = field(default_factory=list)
    results: list[OptimizationResult] = field(default_factory=list)
    best_variant: str | None = None
    best_speedup: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel_path": self.kernel_path,
            "kernel_type": self.kernel_type,
            "session_id": self.session_id,
            "start_time": self.start_time,
            "problem_sizes": self.problem_sizes,
            "baseline_us": self.baseline_us,
            "variants_count": len(self.variants),
            "results": [
                {
                    "name": r.variant_name,
                    "hash": r.variant_hash,
                    "compile_success": r.compile_success,
                    "benchmark_us": r.benchmark_us,
                    "speedup": r.speedup_vs_baseline,
                    "gflops": r.gflops,
                    "bandwidth_gb_s": r.bandwidth_gb_s,
                    "error": r.error,
                }
                for r in self.results
            ],
            "best_variant": self.best_variant,
            "best_speedup": self.best_speedup,
        }


# ---------------------------------------------------------------------------
# Local Benchmarking (without AlphaHENG)
# ---------------------------------------------------------------------------


def run_local_benchmark(
    problem_sizes: list[tuple[int, int, int]],
    warmup: int = 5,
    iters: int = 20,
    group_size: int = 32,
) -> dict[str, Any]:
    """Run local benchmarking using Metal dispatch.

    This provides a standalone way to optimize kernels without AlphaHENG.
    """
    # Add metal_marlin to path
    sys.path.insert(0, str(METAL_MARLIN_ROOT))

    try:
        from metal_marlin._compat import HAS_MPS, torch
        from metal_marlin.kernels import pack_fp4_weights
        from metal_marlin.metal_dispatch import dispatch_gemm_fp4, get_default_library

        if not HAS_MPS:
            print("ERROR: MPS not available. Metal benchmarking requires Apple Silicon.")
            return {"error": "MPS not available"}

        lib = get_default_library()
        results = {}

        for M, N, K in problem_sizes:
            print(f"Benchmarking M={M}, N={N}, K={K}...")

            # Create test data
            A = torch.randn(M, K, dtype=torch.float16, device="mps")
            weight = torch.randn(N, K, dtype=torch.float16, device="mps")
            B_packed, scales = pack_fp4_weights(weight, group_size=group_size)

            # Warmup
            for _ in range(warmup):
                _ = dispatch_gemm_fp4(lib, A, B_packed, scales, M, N, K, group_size)
            torch.mps.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(iters):
                _ = dispatch_gemm_fp4(lib, A, B_packed, scales, M, N, K, group_size)
            torch.mps.synchronize()
            elapsed = time.perf_counter() - start

            avg_time_us = (elapsed / iters) * 1e6

            # Compute FLOPs (2 ops per FMA: multiply + add)
            flops = 2 * M * N * K
            gflops = flops / (avg_time_us * 1e-6) / 1e9

            # Estimate bandwidth
            bytes_read = M * K * 2 + (K // 8) * N * 4 + (K // group_size) * N * 2
            bytes_write = M * N * 2
            bandwidth_gb_s = (bytes_read + bytes_write) / (avg_time_us * 1e-6) / 1e9

            results[(M, N, K)] = {
                "time_us": avg_time_us,
                "gflops": gflops,
                "bandwidth_gb_s": bandwidth_gb_s,
            }
            print(f"  Time: {avg_time_us:.2f} us, {gflops:.1f} GFLOPS, {bandwidth_gb_s:.1f} GB/s")

        return {
            "results": results,
        }
    except ImportError as e:
        print(f"ERROR: Cannot import metal_marlin: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"ERROR: Benchmark failed: {e}")
        return {"error": str(e)}


def benchmark_kernel_variant(
    kernel_source: str,
    problem_size: tuple[int, int, int],
    warmup: int = 3,
    iters: int = 10,
    group_size: int = 32,
) -> dict[str, Any]:
    """Benchmark a kernel variant with modified source code.

    Returns timing and throughput metrics.
    """
    sys.path.insert(0, str(METAL_MARLIN_ROOT))

    try:
        from metal_marlin._compat import HAS_MPS, torch
        from metal_marlin.kernels import pack_fp4_weights
        from metal_marlin.metal_dispatch import MetalKernelLibrary, dispatch_gemm_fp4

        if not HAS_MPS:
            return {"error": "MPS not available", "compile_success": False}

        M, N, K = problem_size

        # Create library with modified source
        lib = MetalKernelLibrary()
        try:
            lib.compile_source("modified_kernel", kernel_source)
        except RuntimeError as e:
            return {"error": str(e), "compile_success": False}

        # Create test data
        A = torch.randn(M, K, dtype=torch.float16, device="mps")
        weight = torch.randn(N, K, dtype=torch.float16, device="mps")
        B_packed, scales = pack_fp4_weights(weight, group_size=group_size)

        # Warmup
        for _ in range(warmup):
            _ = dispatch_gemm_fp4(lib, A, B_packed, scales, M, N, K, group_size)
        torch.mps.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(iters):
            _ = dispatch_gemm_fp4(lib, A, B_packed, scales, M, N, K, group_size)
        torch.mps.synchronize()
        elapsed = time.perf_counter() - start

        avg_time_us = (elapsed / iters) * 1e6

        # Compute FLOPs (2 ops per FMA: multiply + add)
        flops = 2 * M * N * K
        gflops = flops / (avg_time_us * 1e-6) / 1e9

        # Estimate bandwidth
        bytes_read = M * K * 2 + (K // 8) * N * 4 + (K // group_size) * N * 2
        bytes_write = M * N * 2
        bandwidth_gb_s = (bytes_read + bytes_write) / (avg_time_us * 1e-6) / 1e9

        return {
            "compile_success": True,
            "benchmark_us": avg_time_us,
            "gflops": gflops,
            "bandwidth_gb_s": bandwidth_gb_s,
        }

    except Exception as e:
        return {"error": str(e), "compile_success": False}


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


def get_applicable_patterns(
    kernel_type: str,
    source: str,
    include_exploratory: bool = False,
    num_random: int = 0,
    random_seed: int | None = None,
) -> list[dict[str, Any]]:
    """Get patterns applicable to this kernel type that match the source.

    Args:
        kernel_type: Type of kernel (gemm, attention, etc.)
        source: Kernel source code
        include_exploratory: Include Metal-specific exploratory patterns
        num_random: Number of random variants to generate (entropy injection)
        random_seed: Seed for reproducible random variants
    """
    patterns: list[dict[str, Any]] = []

    # Standard deterministic patterns
    for p in OPTIMIZATION_PATTERNS:
        applicable = p.get("applicable_to", [])
        if kernel_type not in applicable and "all" not in applicable:
            continue
        # Check if pattern matches source
        if "multi_pattern" in p:
            # All patterns in multi_pattern must match
            if all(re.search(pat, source) for pat, _ in p["multi_pattern"]):
                patterns.append(p)
        elif re.search(p.get("pattern", ""), source):
            patterns.append(p)

    # Metal-specific exploratory patterns
    if include_exploratory:
        for p in METAL_EXPLORATORY_PATTERNS:
            applicable = p.get("applicable_to", [])
            if kernel_type not in applicable and "all" not in applicable:
                continue
            # Check if patterns match source
            if "multi_pattern" in p:
                if all(re.search(pat, source) for pat, _ in p["multi_pattern"]):
                    patterns.append(p)
            elif re.search(p.get("pattern", ""), source):
                patterns.append(p)

    # Random variants for entropy injection
    if num_random > 0:
        random_patterns = generate_random_variants(kernel_type, source, num_random, random_seed)
        patterns.extend(random_patterns)

    return patterns


def generate_variants(
    kernel_path: Path,
    kernel_type: str,
    source: str,
    include_exploratory: bool = False,
    num_random: int = 0,
    random_seed: int | None = None,
) -> list[OptimizationVariant]:
    """Generate all optimization variants to test.

    Args:
        kernel_path: Path to kernel file
        kernel_type: Type of kernel (gemm, attention, etc.)
        source: Kernel source code
        include_exploratory: Include Metal-specific exploratory patterns
        num_random: Number of random variants to generate
        random_seed: Seed for reproducible random variants
    """
    variants: list[OptimizationVariant] = []
    patterns = get_applicable_patterns(
        kernel_type, source, include_exploratory, num_random, random_seed
    )

    for p in patterns:
        # Apply the transformation(s)
        new_source = source

        if "multi_pattern" in p:
            # Apply multiple pattern replacements
            for pattern, replacement in p["multi_pattern"]:
                new_source = re.sub(pattern, replacement, new_source)
            # Build combined pattern/replacement strings for metadata
            combined_pattern = " && ".join(pat for pat, _ in p["multi_pattern"])
            combined_replacement = " && ".join(repl for _, repl in p["multi_pattern"])
        elif p.get("once"):
            new_source = re.sub(p["pattern"], p["replacement"], source, count=1)
            combined_pattern = p["pattern"]
            combined_replacement = p["replacement"]
        else:
            new_source = re.sub(p["pattern"], p["replacement"], source)
            combined_pattern = p["pattern"]
            combined_replacement = p["replacement"]

        if new_source != source:
            variants.append(
                OptimizationVariant(
                    name=p["name"],
                    description=p["description"],
                    kernel_path=str(kernel_path),
                    pattern=combined_pattern,
                    replacement=combined_replacement,
                    variant_source=new_source,
                    priority=p.get("priority", "P1"),
                )
            )

    return variants


def generate_task_yaml(
    session: OptimizationSession,
    output_dir: Path,
    include_llm_hypotheses: bool = False,
) -> Path:
    """Generate AlphaHENG task YAML for optimization variants.

    Args:
        session: The optimization session
        output_dir: Directory to write the YAML file
        include_llm_hypotheses: If True, add tasks for LLM-driven hypothesis generation
    """

    tasks: list[dict[str, Any]] = []

    # Default problem sizes (M, N, K) for benchmarking
    problem_sizes = session.problem_sizes or [
        (1, 4096, 4096),  # Single token decode
        (32, 4096, 4096),  # Small batch
        (256, 4096, 4096),  # Medium batch
    ]
    # Format problem sizes as Python list of tuples
    problem_str = ";".join(f"{m},{n},{k}" for m, n, k in problem_sizes)
    problem_list_str = ", ".join(f"({m},{n},{k})" for m, n, k in problem_sizes)

    # Task 0: Benchmark baseline
    tasks.append(
        {
            "name": f"opt-{session.session_id}-baseline",
            "prompt": f"""Benchmark the baseline kernel for optimization comparison.

Kernel: {session.kernel_path}
Session: {session.session_id}
Problem sizes: {problem_str}

Steps:
1. Run the benchmark script for this kernel
2. Record the timing in us (microseconds) for each problem size
3. Save results to: `contrib/metal_marlin/agent_workspace/opt_{session.session_id}/baseline.json`

Commands:
```bash
cd contrib/metal_marlin && uv run python -c "
import json
import time
import sys
sys.path.insert(0, '.')

from metal_marlin._compat import HAS_MPS, torch
from metal_marlin.metal_dispatch import get_default_library, dispatch_gemm_fp4
from metal_marlin.kernels import pack_fp4_weights
from pathlib import Path

if not HAS_MPS:
    print('ERROR: MPS not available')
    sys.exit(1)

lib = get_default_library()
problem_sizes = [{problem_list_str}]
results = {{}}

for M, N, K in problem_sizes:
    A = torch.randn(M, K, dtype=torch.float16, device='mps')
    weight = torch.randn(N, K, dtype=torch.float16, device='mps')
    B_packed, scales = pack_fp4_weights(weight, group_size=32)

    # Warmup
    for _ in range(5):
        _ = dispatch_gemm_fp4(lib, A, B_packed, scales, M, N, K, 32)
    torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(20):
        _ = dispatch_gemm_fp4(lib, A, B_packed, scales, M, N, K, 32)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    avg_us = (elapsed / 20) * 1e6
    results[(M, N, K)] = avg_us
    print(f'  {{M}}x{{N}}x{{K}}: {{avg_us:.2f}} us')

# Save result
out_dir = Path('agent_workspace/opt_{session.session_id}')
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / 'baseline.json').write_text(json.dumps({{
    'kernel': '{session.kernel_path}',
    'results': {{str(k): v for k, v in results.items()}},
    'timestamp': time.time(),
}}))
print(f'Baseline saved to {{out_dir / \"baseline.json\"}}')
"
```
""",
            "priority": "P0",
            "dependencies": [],
        }
    )

    # Generate task for each variant
    for variant in session.variants:
        variant_hash = variant.get_variant_hash()

        # Escape the pattern and replacement for embedding in Python code
        pattern_escaped = variant.pattern.replace("\\", "\\\\").replace("'", "\\'")
        replacement_escaped = variant.replacement.replace("\\", "\\\\").replace("'", "\\'")

        tasks.append(
            {
                "name": f"opt-{session.session_id}-{variant.name}",
                "prompt": f"""Test optimization variant: {variant.name}

Kernel: {session.kernel_path}
Variant: {variant.name} ({variant_hash})
Description: {variant.description}
Problem sizes: {problem_str}

Steps:
1. Create a temporary copy of the kernel with this modification
2. Apply transformation and benchmark
3. Save results to: `contrib/metal_marlin/agent_workspace/opt_{session.session_id}/{variant.name}.json`

The modification:
- Pattern: `{variant.pattern}`
- Replacement: `{variant.replacement}`

Commands:
```bash
cd contrib/metal_marlin && uv run python -c "
import json
import re
import time
import sys
sys.path.insert(0, '.')

from pathlib import Path

kernel_path = Path('{variant.kernel_path}')
original = kernel_path.read_text()

# Apply transformation
pattern = r'{pattern_escaped}'
replacement = r'{replacement_escaped}'
modified = re.sub(pattern, replacement, original, count=1)

if modified == original:
    print('ERROR: Pattern did not match')
    result = {{'variant': '{variant.name}', 'hash': '{variant_hash}', 'compile_success': False, 'error': 'Pattern did not match'}}
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

        problem_sizes = [{problem_str.replace(";", "], [")}]
        results = {{}}

        for M, N, K in problem_sizes:
            A = torch.randn(M, K, dtype=torch.float16, device='mps')
            weight = torch.randn(N, K, dtype=torch.float16, device='mps')
            B_packed, scales = pack_fp4_weights(weight, group_size=32)

            # Warmup
            for _ in range(5):
                _ = dispatch_gemm_fp4(lib, A, B_packed, scales, M, N, K, 32)
            torch.mps.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(20):
                _ = dispatch_gemm_fp4(lib, A, B_packed, scales, M, N, K, 32)
            torch.mps.synchronize()
            elapsed = time.perf_counter() - start

            avg_us = (elapsed / 20) * 1e6
            results[(M, N, K)] = avg_us
            print(f'  {{M}}x{{N}}x{{K}}: {{avg_us:.2f}} us')

        result = {{
            'variant': '{variant.name}',
            'hash': '{variant_hash}',
            'compile_success': True,
            'results': {{str(k): v for k, v in results.items()}},
            'timestamp': time.time(),
        }}
        print(f'{variant.name}: SUCCESS')

    except Exception as e:
        result = {{'variant': '{variant.name}', 'hash': '{variant_hash}', 'compile_success': False, 'error': str(e)}}
        print(f'{variant.name}: FAILED - {{e}}')
    finally:
        # Restore original
        kernel_path.write_text(original)

# Save result
out_dir = Path('agent_workspace/opt_{session.session_id}')
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / '{variant.name}.json').write_text(json.dumps(result))
print(f'Saved to {{out_dir / \"{variant.name}.json\"}}')
"
```
""",
                "priority": variant.priority,
                "dependencies": [f"opt-{session.session_id}-baseline"],
            }
        )

    # =========================================================================
    # LLM Hypothesis Generation Tasks (optional)
    # =========================================================================
    # These tasks ask the LLM to analyze the kernel and propose novel
    # optimizations that may not be captured by predetermined patterns.
    # The LLM can leverage its understanding of GPU architectures to
    # suggest non-obvious improvements.

    all_variant_names = [f"opt-{session.session_id}-{v.name}" for v in session.variants]

    if include_llm_hypotheses:
        # Task: Propose novel optimizations based on kernel analysis
        tasks.append(
            {
                "name": f"opt-{session.session_id}-llm-analyze",
                "prompt": f"""Analyze this Metal GEMM kernel and propose novel optimizations.

Kernel: {session.kernel_path}
Session: {session.session_id}
Kernel type: {session.kernel_type}

**CONTEXT: Apple Silicon GPU Architecture**
Metal GPUs differ significantly from CUDA:
- Unified Memory Architecture (UMA): CPU and GPU share memory, no explicit transfers
- Tile-Based Deferred Rendering: GPU processes in tiles, affects memory access patterns
- simdgroup_matrix: 8x8 matrix operations, different from CUDA's tensor cores
- Fixed 32 threads per simdgroup (vs CUDA's 32-thread warps with more flexibility)
- Different optimal occupancy characteristics
- Threadgroup memory is limited (~32KB typical)

**YOUR TASK:**
1. Read the kernel source at `{session.kernel_path}`
2. Identify potential Metal-specific optimizations that are NON-OBVIOUS from CUDA experience
3. Consider:
   - Memory access patterns that exploit UMA
   - Tile sizes that align with Apple GPU tile dimensions
   - Async copy patterns using simdgroup_async_copy
   - Register pressure vs occupancy tradeoffs specific to Apple Silicon
   - Bank conflict avoidance in threadgroup memory (different from CUDA shared memory)
   - simdgroup matrix tiling that maximizes ALU utilization

**OUTPUT:**
Propose 3-5 specific code modifications as regex patterns:

```
PATTERN_1:
Name: descriptive_name
Hypothesis: Why this might improve performance on Metal
Search: <regex pattern to match>
Replace: <replacement>

PATTERN_2:
...
```

Save your analysis to:
`contrib/metal_marlin/agent_workspace/opt_{session.session_id}/llm_hypotheses.json`

Format:
```json
{{
  "hypotheses": [
    {{
      "name": "pattern_name",
      "hypothesis": "Why this should work on Metal",
      "pattern": "regex pattern",
      "replacement": "replacement string",
      "confidence": 0.0-1.0
    }}
  ]
}}
```
""",
                "priority": "P1",
                "dependencies": [f"opt-{session.session_id}-baseline"],
            }
        )

        # Task: Test LLM-proposed hypotheses
        tasks.append(
            {
                "name": f"opt-{session.session_id}-llm-test",
                "prompt": f"""Test the LLM-proposed optimization hypotheses.

Session: {session.session_id}
Hypotheses file: `contrib/metal_marlin/agent_workspace/opt_{session.session_id}/llm_hypotheses.json`
Problem sizes: {problem_str}

**TASK:**
1. Read the hypotheses from the JSON file
2. For each hypothesis with confidence > 0.3:
   a. Apply the pattern/replacement to the kernel
   b. Compile and benchmark
   c. Record results
3. Save results to: `contrib/metal_marlin/agent_workspace/opt_{session.session_id}/llm_results.json`

**IMPORTANT:**
- Always restore the original kernel after testing each hypothesis
- If compilation fails, record the error and move on
- Use the same benchmark methodology as other variants (5 warmup, 20 iterations)

Commands:
```bash
cd contrib/metal_marlin && uv run python -c "
import json
import re
import time
import sys
sys.path.insert(0, '.')

from pathlib import Path

hyp_file = Path('agent_workspace/opt_{session.session_id}/llm_hypotheses.json')
if not hyp_file.exists():
    print('LLM hypotheses file not found, skipping')
    sys.exit(0)

hypotheses = json.loads(hyp_file.read_text()).get('hypotheses', [])
kernel_path = Path('{session.kernel_path}')
original = kernel_path.read_text()

from metal_marlin._compat import HAS_MPS, torch
from metal_marlin.metal_dispatch import MetalKernelLibrary, dispatch_gemm_fp4
from metal_marlin.kernels import pack_fp4_weights

results = []

for hyp in hypotheses:
    if hyp.get('confidence', 0) < 0.3:
        continue

    name = hyp['name']
    print(f'Testing: {{name}}')

    try:
        modified = re.sub(hyp['pattern'], hyp['replacement'], original)
        if modified == original:
            results.append({{'name': name, 'error': 'Pattern did not match'}})
            continue

        kernel_path.write_text(modified)

        lib = MetalKernelLibrary.from_source_dir()
        problem_sizes = [{problem_list_str}]

        timing_results = {{}}
        for M, N, K in problem_sizes:
            A = torch.randn(M, K, dtype=torch.float16, device='mps')
            weight = torch.randn(N, K, dtype=torch.float16, device='mps')
            B_packed, scales = pack_fp4_weights(weight, group_size=32)

            for _ in range(5):
                _ = dispatch_gemm_fp4(lib, A, B_packed, scales, M, N, K, 32)
            torch.mps.synchronize()

            start = time.perf_counter()
            for _ in range(20):
                _ = dispatch_gemm_fp4(lib, A, B_packed, scales, M, N, K, 32)
            torch.mps.synchronize()
            elapsed = time.perf_counter() - start

            timing_results[(M,N,K)] = (elapsed/20)*1e6
            print(f'  {{M}}x{{N}}x{{K}}: {{timing_results[(M,N,K)]:.2f}} us')

        results.append({{
            'name': name,
            'hypothesis': hyp.get('hypothesis', ''),
            'compile_success': True,
            'results': {{str(k): v for k, v in timing_results.items()}},
        }})

    except Exception as e:
        results.append({{'name': name, 'error': str(e)}})
    finally:
        kernel_path.write_text(original)

out_path = Path('agent_workspace/opt_{session.session_id}/llm_results.json')
out_path.write_text(json.dumps(results, indent=2))
print(f'LLM hypothesis results saved to {{out_path}}')
"
```
""",
                "priority": "P1",
                "dependencies": [f"opt-{session.session_id}-llm-analyze"],
            }
        )

        all_variant_names.extend(
            [
                f"opt-{session.session_id}-llm-test",
            ]
        )

    # =========================================================================
    # Final Collection Task
    # =========================================================================
    tasks.append(
        {
            "name": f"opt-{session.session_id}-collect",
            "prompt": f"""Collect optimization results and report the best variant.

Session: {session.session_id}
Results directory: `contrib/metal_marlin/agent_workspace/opt_{session.session_id}/`

Steps:
1. Read all result JSON files from the results directory
2. Find the variant with best speedup vs baseline
3. Generate summary report
4. Save best variant info for the apply task

Commands:
```bash
cd contrib/metal_marlin && uv run python -c "
import json
from pathlib import Path

results_dir = Path('agent_workspace/opt_{session.session_id}')
kernel_path = Path('{session.kernel_path}')

# Load baseline
baseline_file = results_dir / 'baseline.json'
if not baseline_file.exists():
    print('ERROR: Baseline results not found')
    exit(1)

baseline = json.loads(baseline_file.read_text())
baseline_results = baseline.get('results', {{}})

# Load all variant results
variants = []
for f in sorted(results_dir.glob('*.json')):
    if f.name in ['baseline.json', 'best_variant.json', 'llm_hypotheses.json']:
        continue
    data = json.loads(f.read_text())
    if data.get('compile_success') and data.get('results'):
        # Compute average speedup across problem sizes
        speedups = []
        for key, variant_us in data['results'].items():
            if key in baseline_results:
                speedups.append(baseline_results[key] / variant_us)
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            variants.append({{
                'name': data['variant'],
                'speedup': avg_speedup,
                'results': data['results'],
                'file': f.name,
            }})

# Sort by speedup
variants.sort(key=lambda x: x['speedup'], reverse=True)

print('='*60)
print(f'Optimization Session: {session.session_id}')
print(f'Kernel: {session.kernel_path}')
print('='*60)
print()
print('Baseline results:')
for key, us in baseline_results.items():
    print(f'  {{key}}: {{us:.2f}} us')
print()
print('Results (sorted by speedup):')
for v in variants[:15]:
    print(f'  {{v[\"name\"]}}: {{v[\"speedup\"]:.3f}}x')
print()

# Save best variant info for apply task
if variants and variants[0]['speedup'] > 1.02:
    best = variants[0]
    best_info = {{
        'variant_name': best['name'],
        'speedup': best['speedup'],
        'kernel_path': '{session.kernel_path}',
        'session_id': '{session.session_id}',
    }}
    (results_dir / 'best_variant.json').write_text(json.dumps(best_info, indent=2))
    print(f'BEST: {{best[\"name\"]}} ({{best[\"speedup\"]:.3f}}x speedup)')
    print(f'Saved to best_variant.json for auto-apply')
else:
    print('No significant improvement found (threshold: >2%)')
"
```
""",
            "priority": "P3",  # Run after all P1/P2 variants complete
            "dependencies": all_variant_names,
        }
    )

    # =========================================================================
    # Apply Best Variant (Auto-apply if >2% improvement)
    # =========================================================================
    tasks.append(
        {
            "name": f"opt-{session.session_id}-apply",
            "prompt": rf"""Apply the best optimization variant to the kernel source file.

Session: {session.session_id}
Kernel: {session.kernel_path}
Results directory: `contrib/metal_marlin/agent_workspace/opt_{session.session_id}/`

**THIS TASK ACTUALLY MODIFIES THE SOURCE FILE** if improvement > 2%.

Steps:
1. Read best_variant.json to get the winning variant name
2. Look up that variant's transformation pattern
3. Apply the transformation to the actual kernel file
4. Verify the change by running a quick benchmark
5. Commit message with speedup info

Commands:
```bash
cd contrib/metal_marlin && uv run python scripts/optimize_kernel.py --apply-best {session.session_id}
```

If --apply-best is not available, apply manually:
```bash
cd contrib/metal_marlin && uv run python -c "
import json
import re
from pathlib import Path

results_dir = Path('agent_workspace/opt_{session.session_id}')
kernel_path = Path('{session.kernel_path}')

# Read best variant info
best_file = results_dir / 'best_variant.json'
if not best_file.exists():
    print('No best_variant.json found - no improvement to apply')
    exit(0)

best = json.loads(best_file.read_text())
variant_name = best['variant_name']
speedup = best['speedup']

print(f'Applying {{variant_name}} ({{speedup:.3f}}x speedup) to {{kernel_path.name}}')

# Map variant names to transformations
VARIANT_TRANSFORMS = {{
    'tile_m_32': ('TILE_M', 32),
    'tile_m_64': ('TILE_M', 64),
    'tile_m_128': ('TILE_M', 128),
    'tile_n_32': ('TILE_N', 32),
    'tile_n_64': ('TILE_N', 64),
    'tile_n_128': ('TILE_N', 128),
    'tile_k_16': ('TILE_K', 16),
    'tile_k_32': ('TILE_K', 32),
    'tile_k_64': ('TILE_K', 64),
    'simdgroups_2': ('SIMDGROUPS_PER_TG', 2),
    'simdgroups_4': ('SIMDGROUPS_PER_TG', 4),
    'simdgroups_8': ('SIMDGROUPS_PER_TG', 8),
    'num_buffers_2': ('NUM_BUFFERS', 2),
    'num_buffers_3': ('NUM_BUFFERS', 3),
    'exp_simdgroups_1': ('SIMDGROUPS_PER_TG', 1),
    'exp_simdgroups_16': ('SIMDGROUPS_PER_TG', 16),
    'exp_num_buffers_4': ('NUM_BUFFERS', 4),
    'exp_tile_k_48': ('TILE_K', 48),
    'exp_tile_k_24': ('TILE_K', 24),
}}

if variant_name not in VARIANT_TRANSFORMS:
    # Handle multi-pattern or random variants
    print(f'Complex variant {{variant_name}} - manual review needed')
    print('Check the variant JSON file for the exact transformation.')
    exit(1)

param, value = VARIANT_TRANSFORMS[variant_name]
source = kernel_path.read_text()

# Apply transformation
pattern = rf'constant\s+constexpr\s+uint\s+{{param}}\s*=\s*\d+'
replacement = f'constant constexpr uint {{param}} = {{value}}'

new_source, count = re.subn(pattern, replacement, source)
if count == 0:
    print(f'Pattern not found: {{param}}')
    exit(1)

kernel_path.write_text(new_source)
print(f'Applied: {{param}} = {{value}}')
print(f'Modified: {{kernel_path}}')
print()
print('Verify with: cd contrib/metal_marlin && uv run pytest tests/ -v -k gemm')
"
```
""",
            "priority": "P3",  # Run after collect completes
            "dependencies": [f"opt-{session.session_id}-collect"],
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
    if ALPHAHENG_ROOT is None:
        print("ERROR: Not running within AlphaHENG monorepo.")
        print("Cannot dispatch tasks. Use --generate-only to create task YAML.")
        sys.exit(1)

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

    def normalize_key(key: str) -> tuple[int, int, int]:
        """Normalize problem size key to (M, N, K) tuple.

        Handles various formats from different agents:
        - "(1, 4096, 4096)" - tuple string with parens/spaces
        - "1,4096,4096" - comma-separated
        - "1_4096_4096" - underscore-separated
        - "(1,4096,4096)" - tuple string without spaces
        """
        # Remove parentheses and spaces
        clean = key.replace("(", "").replace(")", "").replace(" ", "")
        # Split on comma or underscore
        if "_" in clean:
            parts = clean.split("_")
        else:
            parts = clean.split(",")
        return (int(parts[0]), int(parts[1]), int(parts[2]))

    # Load baseline
    baseline_path = results_dir / "baseline.json"
    if not baseline_path.exists():
        print("Baseline results not found")
        return None

    baseline = json.loads(baseline_path.read_text())
    baseline_raw = baseline.get("results", {})

    # Normalize baseline keys
    baseline_results: dict[tuple[int, int, int], float] = {}
    for key, value in baseline_raw.items():
        try:
            normalized = normalize_key(key)
            baseline_results[normalized] = value
        except (ValueError, IndexError):
            print(f"Warning: Could not parse baseline key: {key}")

    # Load variant results
    results: list[OptimizationResult] = []
    for f in sorted(results_dir.glob("*.json")):
        if f.name == "baseline.json":
            continue
        if f.name == "llm_hypotheses.json":
            # This is the LLM analysis output, not a benchmark result
            continue

        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            print(f"Warning: Skipping {f.name} (invalid JSON: {e})")
            continue

        speedup = None
        if data.get("compile_success") and data.get("results"):
            # Calculate average speedup across problem sizes
            speedups = []
            for key, variant_us in data["results"].items():
                try:
                    normalized = normalize_key(key)
                    if normalized in baseline_results:
                        speedups.append(baseline_results[normalized] / variant_us)
                except (ValueError, IndexError):
                    continue
            if speedups:
                speedup = sum(speedups) / len(speedups)

        results.append(
            OptimizationResult(
                variant_name=data.get("variant", f.stem),
                variant_hash=data.get("hash", ""),
                compile_success=data.get("compile_success", False),
                benchmark_us=None,  # We track per-problem-size now
                speedup_vs_baseline=speedup,
                error=data.get("error"),
            )
        )

    # Find best
    valid_results = [r for r in results if r.speedup_vs_baseline is not None]
    best_result = max(valid_results, key=lambda r: r.speedup_vs_baseline or 0, default=None)

    best_variant_name = best_result.variant_name if best_result else None
    best_speedup_value = (
        best_result.speedup_vs_baseline if best_result and best_result.speedup_vs_baseline else 1.0
    )

    session = OptimizationSession(
        kernel_path=baseline.get("kernel", ""),
        kernel_type="",
        session_id=session_id,
        start_time="",
        baseline_us=None,  # Tracked per-problem-size now
        results=results,
        best_variant=best_variant_name,
        best_speedup=best_speedup_value,
    )

    return session


# Variant name to transformation mapping
VARIANT_TRANSFORMS: dict[str, tuple[str, int] | list[tuple[str, int]]] = {
    # Single parameter variants
    "tile_m_32": ("TILE_M", 32),
    "tile_m_64": ("TILE_M", 64),
    "tile_m_128": ("TILE_M", 128),
    "tile_n_32": ("TILE_N", 32),
    "tile_n_64": ("TILE_N", 64),
    "tile_n_128": ("TILE_N", 128),
    "tile_k_16": ("TILE_K", 16),
    "tile_k_32": ("TILE_K", 32),
    "tile_k_64": ("TILE_K", 64),
    "simdgroups_2": ("SIMDGROUPS_PER_TG", 2),
    "simdgroups_4": ("SIMDGROUPS_PER_TG", 4),
    "simdgroups_8": ("SIMDGROUPS_PER_TG", 8),
    "sg_m_tiles_2": ("SG_M_TILES", 2),
    "sg_m_tiles_4": ("SG_M_TILES", 4),
    "sg_m_tiles_8": ("SG_M_TILES", 8),
    "sg_n_tiles_2": ("SG_N_TILES", 2),
    "sg_n_tiles_4": ("SG_N_TILES", 4),
    "sg_n_tiles_8": ("SG_N_TILES", 8),
    "num_buffers_2": ("NUM_BUFFERS", 2),
    "num_buffers_3": ("NUM_BUFFERS", 3),
    # Exploratory single-param variants
    "exp_simdgroups_1": ("SIMDGROUPS_PER_TG", 1),
    "exp_simdgroups_16": ("SIMDGROUPS_PER_TG", 16),
    "exp_num_buffers_4": ("NUM_BUFFERS", 4),
    "exp_tile_k_48": ("TILE_K", 48),
    "exp_tile_k_24": ("TILE_K", 24),
    # Multi-parameter variants
    "exp_asymmetric_tiles_wide": [("TILE_M", 32), ("TILE_N", 128)],
    "exp_asymmetric_tiles_tall": [("TILE_M", 128), ("TILE_N", 32)],
    "exp_sg_tiles_1x1": [("SG_M_TILES", 1), ("SG_N_TILES", 1)],
    "exp_sg_tiles_8x8": [("SG_M_TILES", 8), ("SG_N_TILES", 8)],
    "exp_tiny_tiles": [("TILE_M", 16), ("TILE_N", 16), ("TILE_K", 16)],
    "exp_large_tiles": [("TILE_M", 128), ("TILE_N", 128), ("TILE_K", 64)],
}


def apply_best_optimization(session_id: str) -> bool:
    """Apply the best optimization from a completed session to the source file.

    Returns True if optimization was applied, False otherwise.
    """
    results_dir = METAL_MARLIN_ROOT / "agent_workspace" / f"opt_{session_id}"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return False

    # Read best variant info
    best_file = results_dir / "best_variant.json"
    if not best_file.exists():
        print("No best_variant.json found - either no improvement found or collect task not run")

        # Try to generate it now by running collect
        session = collect_results(session_id)
        if session and session.best_speedup > 1.02:
            # Save best_variant.json
            best_info = {
                "variant_name": session.best_variant,
                "speedup": session.best_speedup,
                "kernel_path": session.kernel_path,
                "session_id": session_id,
            }
            best_file.write_text(json.dumps(best_info, indent=2))
            print(
                f"Generated best_variant.json: {session.best_variant} ({session.best_speedup:.3f}x)"
            )
        else:
            print("No significant improvement (>2%) found to apply")
            return False

    best = json.loads(best_file.read_text())
    variant_name = best["variant_name"]
    speedup = best["speedup"]
    kernel_path = Path(best["kernel_path"])

    if not kernel_path.exists():
        print(f"Kernel file not found: {kernel_path}")
        return False

    print(f"Applying {variant_name} ({speedup:.3f}x speedup) to {kernel_path.name}")

    # Get transformation for this variant
    if variant_name not in VARIANT_TRANSFORMS:
        # Check if it's a random variant - need to look up the actual parameters
        variant_file = results_dir / f"{variant_name}.json"
        if variant_file.exists():
            variant_data = json.loads(variant_file.read_text())
            # Random variants store their params in the description
            desc = variant_data.get("description", "")
            print(f"Random variant - description: {desc}")
            print("Manual application needed - check the variant JSON for parameters")
        else:
            print(f"Unknown variant: {variant_name}")
            print("Check the variant JSON file for the exact transformation.")
        return False

    transform = VARIANT_TRANSFORMS[variant_name]
    source = kernel_path.read_text()

    # Apply transformation(s)
    if isinstance(transform, tuple):
        # Single parameter
        transforms = [transform]
    else:
        # Multi-parameter
        transforms = transform

    modified_source = source
    for param, value in transforms:
        pattern = rf"constant\s+constexpr\s+uint\s+{param}\s*=\s*\d+"
        replacement = f"constant constexpr uint {param} = {value}"
        new_source, count = re.subn(pattern, replacement, modified_source)

        if count == 0:
            print(f"Warning: Pattern not found for {param} - skipping")
            continue

        modified_source = new_source
        print(f"  Applied: {param} = {value}")

    if modified_source == source:
        print("No changes made - patterns may not have matched")
        return False

    # Write the modified source
    kernel_path.write_text(modified_source)
    print(f"\nModified: {kernel_path}")
    print(f"Speedup: {speedup:.3f}x")
    print("\nVerify with:")
    print("  cd contrib/metal_marlin && uv run pytest tests/ -v -k gemm")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Optimize Metal kernels using entropy-first exploration via AlphaHENG agent swarm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: entropy-first (20 random + exploratory + deterministic patterns)
  cd contrib/metal_marlin && uv run python scripts/optimize_kernel.py src/marlin_gemm.metal

  # Deterministic-only (disable entropy injection for reproducibility)
  cd contrib/metal_marlin && uv run python scripts/optimize_kernel.py src/marlin_gemm.metal --no-explore

  # Higher entropy for thorough search (50 random variants)
  cd contrib/metal_marlin && uv run python scripts/optimize_kernel.py src/marlin_gemm.metal --num-random 50

  # Local benchmarking without AlphaHENG swarm
  cd contrib/metal_marlin && uv run python scripts/optimize_kernel.py --autotune

  # Collect results after swarm completes
  cd contrib/metal_marlin && uv run python scripts/optimize_kernel.py --collect-results SESSION_ID

Why Entropy-First:
  Metal/Apple Silicon has sparse LLM training data. "Known optimal" CUDA patterns can
  actively hurt Metal performance. Random sampling + benchmark-as-oracle discovers
  non-obvious optima (e.g., exp_simdgroups_1 achieved 1.895x by inverting CUDA wisdom).
""",
    )
    parser.add_argument(
        "kernel",
        type=Path,
        nargs="?",
        help="Path to the Metal kernel file to optimize (relative to metal_marlin/src/)",
    )
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="Run local autotuning using existing kernel configurations (no AlphaHENG)",
    )
    parser.add_argument(
        "--problem-sizes",
        type=str,
        default="256,4096,4096;32,4096,4096;1,4096,4096",
        help="Problem sizes as 'M,N,K;M,N,K;...' (default: common transformer shapes)",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=10,
        help="Number of agents to use for AlphaHENG dispatch (default: 10)",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate task YAML without dispatching to AlphaHENG",
    )
    parser.add_argument(
        "--collect-results",
        type=str,
        metavar="SESSION_ID",
        help="Collect results from a completed session",
    )
    parser.add_argument(
        "--apply-best",
        type=str,
        metavar="SESSION_ID",
        help="Apply the best optimization from a completed session to the source file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=METAL_MARLIN_ROOT / "tasks",
        help="Output directory for task YAML (default: tasks/)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations for benchmarking (default: 5)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Benchmark iterations (default: 20)",
    )
    parser.add_argument(
        "--no-explore",
        action="store_true",
        dest="no_explore",
        help="Disable entropy injection (deterministic patterns only, for reproducibility)",
    )
    parser.add_argument(
        "--num-random",
        type=int,
        default=20,
        help="Number of random parameter combinations to test (default: 20, entropy injection)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Seed for reproducible random variants",
    )
    parser.add_argument(
        "--llm-hypotheses",
        action="store_true",
        help="Include LLM-driven hypothesis generation tasks (requires agents capable of analysis)",
    )

    args = parser.parse_args()

    # Parse problem sizes
    problem_sizes = []
    for size_str in args.problem_sizes.split(";"):
        parts = size_str.strip().split(",")
        if len(parts) == 3:
            problem_sizes.append((int(parts[0]), int(parts[1]), int(parts[2])))

    # Local autotune mode
    if args.autotune:
        print("=" * 60)
        print("Metal Kernel Local Autotuning")
        print("=" * 60)
        print(f"Problem sizes: {problem_sizes}")
        print()

        results = run_local_benchmark(
            problem_sizes,
            warmup=args.warmup,
            iters=args.iters,
        )

        if "error" in results:
            print(f"Autotuning failed: {results['error']}")
            sys.exit(1)

        print()
        print("=" * 60)
        print("Autotuning complete!")
        return

    # Collect mode
    if args.collect_results:
        session = collect_results(args.collect_results)
        if session:
            print("=" * 60)
            print(f"Session: {session.session_id}")
            print(f"Kernel: {session.kernel_path}")
            print("=" * 60)
            print(f"\nBest: {session.best_variant} ({session.best_speedup:.3f}x)")
            print("\nAll results:")
            for r in sorted(
                session.results, key=lambda x: x.speedup_vs_baseline or 0, reverse=True
            ):
                status = "✓" if r.compile_success else "✗"
                speedup = f"{r.speedup_vs_baseline:.3f}x" if r.speedup_vs_baseline else "N/A"
                print(f"  {status} {r.variant_name}: {speedup}")

            # Suggest applying if significant improvement
            if session.best_speedup > 1.02:
                print("\nTo apply the best optimization:")
                print(
                    f"  uv run python scripts/optimize_kernel.py --apply-best {args.collect_results}"
                )
        return

    # Apply best mode
    if args.apply_best:
        success = apply_best_optimization(args.apply_best)
        sys.exit(0 if success else 1)

    # Require kernel for generate/dispatch mode
    if not args.kernel:
        parser.error("kernel argument is required unless using --autotune or --collect-results")

    # Resolve kernel path
    kernel_path = args.kernel
    if not kernel_path.exists():
        # Try relative to src/
        src_path = METAL_MARLIN_ROOT / "src" / args.kernel
        if src_path.exists():
            kernel_path = src_path
        else:
            print(f"Kernel not found: {args.kernel}")
            print(f"Tried: {args.kernel} and {src_path}")
            sys.exit(1)

    kernel_path = kernel_path.resolve()

    # Read kernel source
    source = kernel_path.read_text()
    kernel_type = detect_kernel_type(kernel_path)

    # Generate session ID
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine exploration settings (entropy-first by default)
    include_exploratory = not args.no_explore
    num_random = args.num_random if not args.no_explore else 0

    # Generate variants
    variants = generate_variants(
        kernel_path,
        kernel_type,
        source,
        include_exploratory=include_exploratory,
        num_random=num_random,
        random_seed=args.random_seed,
    )

    if not variants:
        print(f"No applicable optimization patterns for {kernel_path.name}")
        print("The kernel may not have the expected #define patterns.")
        print("Consider using --autotune for existing configurations.")
        sys.exit(1)

    # Count variant types
    deterministic_count = sum(
        1 for v in variants if not v.name.startswith("rand_") and not v.name.startswith("exp_")
    )
    exploratory_count = sum(1 for v in variants if v.name.startswith("exp_"))
    random_count = sum(1 for v in variants if v.name.startswith("rand_"))

    print("=" * 60)
    print("Metal Kernel Optimization via AlphaHENG")
    print("=" * 60)
    print(f"Kernel: {kernel_path.name}")
    print(f"Type: {kernel_type}")
    print(f"Total variants: {len(variants)}")
    if include_exploratory or num_random > 0:
        print(f"  - Deterministic: {deterministic_count}")
        print(f"  - Metal-exploratory: {exploratory_count}")
        print(f"  - Random (entropy): {random_count}")
    if args.llm_hypotheses:
        print("  - LLM-proposed: (generated at runtime)")
    print(f"Problem sizes: {problem_sizes}")
    print()

    # Create session
    session = OptimizationSession(
        kernel_path=str(kernel_path),
        kernel_type=kernel_type,
        session_id=session_id,
        start_time=datetime.now().isoformat(),
        problem_sizes=problem_sizes,
        variants=variants,
    )

    # Generate task YAML
    args.output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = generate_task_yaml(
        session,
        args.output_dir,
        include_llm_hypotheses=args.llm_hypotheses,
    )
    print(f"Generated: {yaml_path}")

    if args.generate_only:
        print("\nTask YAML generated. To dispatch:")
        if ALPHAHENG_ROOT:
            print(f"  cd {ALPHAHENG_ROOT} && uv run alphaheng tasks add {yaml_path}")
        else:
            print(f"  uv run alphaheng tasks add {yaml_path}")
        return

    # Dispatch to AlphaHENG
    dispatch_tasks(yaml_path, args.agents)

    print()
    print(f"Optimization session started: {session_id}")
    print("Monitor progress: uv run alphaheng status")
    print(
        f"Collect results: cd contrib/metal_marlin && uv run python scripts/optimize_kernel.py --collect-results {session_id}"
    )


if __name__ == "__main__":
    main()
