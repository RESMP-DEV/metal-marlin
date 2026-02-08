"""Auto-tuning system for mixed bit-width trellis kernels.

This module provides automatic performance optimization for mixed-bit-width
kernels by benchmarking different configurations and selecting the optimal
parameters at runtime.

Key features:
1. Benchmark different tile sizes per bit-width (2-bit: 128x128, 4-bit: 64x64)
2. Test SIMDgroup configurations (4, 8, 16 simdgroups)
3. Compare decode vs prefill kernel variants
4. Optimize batching strategies
5. Build lookup tables: (batch_size, bit_width, hidden_dim) -> best_kernel
6. Cache tuned configs per device type (M1/M2/M3/M4)
7. Online adaptation during inference
8. Export/load tuned parameters to/from JSON

Usage:
    >>> from metal_marlin.trellis.autotune_mixed_bpw import (
    ...     MixedBPWAutoTuner,
    ...     get_optimal_config,
    ... )
    >>> tuner = MixedBPWAutoTuner(device="m2-max")
    >>> tuner.run_autotune(
    ...     bit_widths=[2, 4],
    ...     batch_sizes=[1, 4, 8, 16],
    ...     hidden_dims=[4096, 5120, 7168],
    ... )
    >>> tuner.export_config("optimized_configs.json")
"""

from __future__ import annotations

import json
import logging
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .._compat import HAS_MPS, torch
from ..utils.profiling import ProfileRecord, _ProfileRegistry

logger = logging.getLogger(__name__)

# Default tile sizes for different bit-widths (optimized for Apple Silicon)
DEFAULT_TILE_SIZES = {
    2: [(64, 64), (128, 64), (128, 128)],  # 2-bit can use larger tiles
    3: [(64, 64), (128, 64), (96, 96)],
    4: [(32, 32), (64, 64), (64, 32)],  # 4-bit uses smaller tiles
    8: [(32, 32), (32, 64)],  # 8-bit uses smallest tiles
}

# SIMDgroup configurations to test
SIMDGROUP_CONFIGS = [4, 8, 16]

# Kernel variants to benchmark
KERNEL_VARIANTS = ["decode", "prefill", "base", "large_batch"]

# Batching strategies
BATCH_STRATEGIES = ["per_expert", "per_bitwidth", "hybrid"]

# Device types for caching
DEVICE_TYPES = ["m1", "m2", "m3", "m4", "unknown"]


@dataclass
class KernelConfig:
    """Configuration for a specific kernel variant.

    Attributes:
        tile_size_m: Tile size for M dimension (rows of output).
        tile_size_n: Tile size for N dimension (columns of output).
        num_simdgroups: Number of SIMD groups to use.
        kernel_variant: Type of kernel (decode, prefill, base, large_batch).
        batch_strategy: Strategy for batching tokens.
        bit_width: Quantization bit width this config targets.
    """

    tile_size_m: int
    tile_size_n: int
    num_simdgroups: int
    kernel_variant: str
    batch_strategy: str
    bit_width: int

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.kernel_variant not in KERNEL_VARIANTS:
            raise ValueError(
                f"Invalid kernel_variant: {self.kernel_variant}. "
                f"Must be one of {KERNEL_VARIANTS}"
            )
        if self.batch_strategy not in BATCH_STRATEGIES:
            raise ValueError(
                f"Invalid batch_strategy: {self.batch_strategy}. "
                f"Must be one of {BATCH_STRATEGIES}"
            )
        if self.bit_width not in DEFAULT_TILE_SIZES:
            raise ValueError(
                f"Invalid bit_width: {self.bit_width}. "
                f"Must be one of {list(DEFAULT_TILE_SIZES.keys())}"
            )
        if self.num_simdgroups not in SIMDGROUP_CONFIGS:
            raise ValueError(
                f"Invalid num_simdgroups: {self.num_simdgroups}. "
                f"Must be one of {SIMDGROUP_CONFIGS}"
            )

    @property
    def config_id(self) -> str:
        """Unique identifier for this configuration."""
        return (
            f"{self.bit_width}bit_"
            f"{self.tile_size_m}x{self.tile_size_n}_"
            f"{self.num_simdgroups}sg_"
            f"{self.kernel_variant}_"
            f"{self.batch_strategy}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single kernel configuration.

    Attributes:
        config: The kernel configuration that was benchmarked.
        batch_size: Batch size used for benchmarking.
        hidden_dim: Hidden dimension used for benchmarking.
        latency_ms: Average latency in milliseconds.
        throughput: Throughput in tokens/second.
        memory_bytes: Peak memory usage in bytes.
        success: Whether the kernel ran successfully.
        error_msg: Error message if not successful.
    """

    config: KernelConfig
    batch_size: int
    hidden_dim: int
    latency_ms: float
    throughput: float
    memory_bytes: int
    success: bool
    error_msg: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "batch_size": self.batch_size,
            "hidden_dim": self.hidden_dim,
            "latency_ms": self.latency_ms,
            "throughput": self.throughput,
            "memory_bytes": self.memory_bytes,
            "success": self.success,
            "error_msg": self.error_msg,
        }


@dataclass
class OptimizedConfig:
    """Optimal configuration for a specific workload.

    Attributes:
        bit_width: Bit width for this config.
        batch_size: Batch size for this config.
        hidden_dim: Hidden dimension for this config.
        kernel_variant: Best kernel variant.
        tile_size_m: Best tile size for M dimension.
        tile_size_n: Best tile size for N dimension.
        num_simdgroups: Best number of SIMD groups.
        batch_strategy: Best batching strategy.
        expected_latency_ms: Expected latency for this config.
        throughput: Expected throughput.
    """

    bit_width: int
    batch_size: int
    hidden_dim: int
    kernel_variant: str
    tile_size_m: int
    tile_size_n: int
    num_simdgroups: int
    batch_strategy: str
    expected_latency_ms: float
    throughput: float

    @property
    def lookup_key(self) -> Tuple[int, int, int]:
        """Key for lookup table: (batch_size, bit_width, hidden_dim)."""
        return (self.batch_size, self.bit_width, self.hidden_dim)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class MixedBPWAutoTuner:
    """Auto-tuning system for mixed bit-width kernels.

    This class provides comprehensive benchmarking and optimization
    capabilities for mixed-bit-width trellis kernels. It automatically
    explores the configuration space, measures performance, and selects
    the optimal parameters for different workloads.

    The tuner maintains a lookup table mapping (batch_size, bit_width,
    hidden_dim) tuples to optimized configurations, enabling fast runtime
    selection without re-benchmarking.

    Example:
        >>> tuner = MixedBPWAutoTuner(device="m2-max")
        >>> tuner.run_autotune(
        ...     bit_widths=[2, 4],
        ...     batch_sizes=[1, 4, 8],
        ...     hidden_dims=[4096],
        ... )
        >>> best_config = tuner.get_optimal_config(batch_size=4, bit_width=2, hidden_dim=4096)
        >>> tuner.export_config("configs.json")
    """

    def __init__(
        self,
        device: Optional[str] = None,
        warmup_iterations: int = 5,
        benchmark_iterations: int = 20,
        enable_online_adaptation: bool = True,
        history_length: int = 100,
    ):
        """Initialize the auto-tuner.

        Args:
            device: Device type (m1, m2, m3, m4, or auto-detect if None).
            warmup_iterations: Number of warmup iterations before benchmarking.
            benchmark_iterations: Number of benchmark iterations to average.
            enable_online_adaptation: Enable online adaptation during inference.
            history_length: Number of runtime samples to keep for adaptation.
        """
        # Detect device type if not specified
        self.device = device or self._detect_device_type()

        # Benchmarking parameters
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

        # Online adaptation
        self.enable_online_adaptation = enable_online_adaptation
        self.history_length = history_length

        # Storage for benchmark results and optimized configs
        self.benchmark_results: List[BenchmarkResult] = []
        self.optimized_configs: Dict[Tuple[int, int, int], OptimizedConfig] = {}

        # Online adaptation history: (batch_size, bit_width, hidden_dim) -> list of (config_id, latency_ms)
        self.online_history: Dict[Tuple[int, int, int], List[Tuple[str, float]]] = {}

        # Profile registry for kernel timing
        self.profile_registry = _ProfileRegistry()

        logger.info(f"Initialized MixedBPWAutoTuner for device: {self.device}")

    @staticmethod
    def _detect_device_type() -> str:
        """Detect the current Apple Silicon device type.

        Returns:
            Device type string (m1, m2, m3, m4, or unknown).
        """
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            cpu_brand = result.stdout.strip()

            if "M1" in cpu_brand:
                return "m1"
            elif "M2" in cpu_brand:
                return "m2"
            elif "M3" in cpu_brand:
                return "m3"
            elif "M4" in cpu_brand:
                return "m4"
            else:
                logger.warning(f"Unknown CPU brand: {cpu_brand}, defaulting to 'unknown'")
                return "unknown"
        except Exception as e:
            logger.warning(f"Failed to detect device type: {e}, defaulting to 'unknown'")
            return "unknown"

    def generate_configs(
        self,
        bit_widths: List[int],
        kernel_variants: Optional[List[str]] = None,
        batch_strategies: Optional[List[str]] = None,
    ) -> List[KernelConfig]:
        """Generate all kernel configurations to benchmark.

        Args:
            bit_widths: List of bit widths to generate configs for.
            kernel_variants: List of kernel variants (uses all if None).
            batch_strategies: List of batch strategies (uses all if None).

        Returns:
            List of all kernel configurations to benchmark.
        """
        kernel_variants = kernel_variants or KERNEL_VARIANTS
        batch_strategies = batch_strategies or BATCH_STRATEGIES

        configs = []
        for bit_width in bit_widths:
            for (tile_m, tile_n) in DEFAULT_TILE_SIZES[bit_width]:
                for num_simdgroups in SIMDGROUP_CONFIGS:
                    for kernel_variant in kernel_variants:
                        for batch_strategy in batch_strategies:
                            config = KernelConfig(
                                tile_size_m=tile_m,
                                tile_size_n=tile_n,
                                num_simdgroups=num_simdgroups,
                                kernel_variant=kernel_variant,
                                batch_strategy=batch_strategy,
                                bit_width=bit_width,
                            )
                            configs.append(config)

        logger.info(f"Generated {len(configs)} configurations for bit widths: {bit_widths}")
        return configs

    def benchmark_config(
        self,
        config: KernelConfig,
        batch_size: int,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int = 8,
    ) -> BenchmarkResult:
        """Benchmark a single kernel configuration.

        Args:
            config: Kernel configuration to benchmark.
            batch_size: Batch size to use.
            hidden_dim: Hidden dimension.
            intermediate_dim: Intermediate dimension.
            num_experts: Number of experts.

        Returns:
            Benchmark result with timing and performance metrics.
        """
        if not HAS_MPS or not HAS_METAL_AVAILABLE():
            return BenchmarkResult(
                config=config,
                batch_size=batch_size,
                hidden_dim=hidden_dim,
                latency_ms=float("inf"),
                throughput=0.0,
                memory_bytes=0,
                success=False,
                error_msg="Metal not available",
            )

        try:
            # Create synthetic input data
            hidden_states = torch.randn(batch_size, hidden_dim, dtype=torch.float16)
            hidden_states = hidden_states.to("mps")

            # Simulate expert weights with mixed bit-widths
            expert_indices = torch.randint(0, num_experts, (batch_size,))
            router_probs = torch.softmax(
                torch.randn(batch_size, num_experts), dim=-1
            )

            # Warmup iterations
            for _ in range(self.warmup_iterations):
                self._run_synthetic_kernel(
                    config,
                    hidden_states,
                    expert_indices,
                    router_probs,
                    hidden_dim,
                    intermediate_dim,
                )

            # Benchmark iterations
            latencies = []
            for _ in range(self.benchmark_iterations):
                torch.mps.synchronize()
                start_time = time.perf_counter()
                self._run_synthetic_kernel(
                    config,
                    hidden_states,
                    expert_indices,
                    router_probs,
                    hidden_dim,
                    intermediate_dim,
                )
                torch.mps.synchronize()
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms

            # Calculate statistics
            avg_latency_ms = statistics.mean(latencies)
            std_latency_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            throughput = (batch_size * hidden_dim) / (avg_latency_ms / 1000)

            # Estimate memory usage (rough estimate based on config)
            memory_bytes = self._estimate_memory_usage(
                config, batch_size, hidden_dim, intermediate_dim
            )

            logger.debug(
                f"Config {config.config_id}: "
                f"{avg_latency_ms:.3f} ± {std_latency_ms:.3f} ms, "
                f"{throughput:.1f} tokens/s"
            )

            return BenchmarkResult(
                config=config,
                batch_size=batch_size,
                hidden_dim=hidden_dim,
                latency_ms=avg_latency_ms,
                throughput=throughput,
                memory_bytes=memory_bytes,
                success=True,
                error_msg="",
            )

        except Exception as e:
            logger.warning(f"Benchmark failed for {config.config_id}: {e}")
            return BenchmarkResult(
                config=config,
                batch_size=batch_size,
                hidden_dim=hidden_dim,
                latency_ms=float("inf"),
                throughput=0.0,
                memory_bytes=0,
                success=False,
                error_msg=str(e),
            )

    def _run_synthetic_kernel(
        self,
        config: KernelConfig,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        router_probs: torch.Tensor,
        hidden_dim: int,
        intermediate_dim: int,
    ) -> None:
        """Run a synthetic kernel for benchmarking.

        This is a placeholder that simulates the actual kernel execution.
        In production, this would call the actual Metal kernel with the
        specified configuration.

        Args:
            config: Kernel configuration.
            hidden_states: Input hidden states.
            expert_indices: Selected expert indices.
            router_probs: Router probabilities.
            hidden_dim: Hidden dimension.
            intermediate_dim: Intermediate dimension.
        """
        # Simulate kernel work based on config
        batch_size, seq_len = hidden_states.shape[:2] if hidden_states.dim() == 3 else (hidden_states.shape[0], 1)

        # Simulate different workloads based on kernel variant
        if config.kernel_variant == "decode":
            # Decode: smaller batch, more sequential work
            work_factor = 1.0
        elif config.kernel_variant == "prefill":
            # Prefill: larger context window
            work_factor = seq_len / 32.0
        elif config.kernel_variant == "large_batch":
            # Large batch: optimize for throughput
            work_factor = batch_size / 16.0
        else:
            # Base: balanced workload
            work_factor = 1.0

        # Simulate tile-based computation
        num_tiles_m = (hidden_dim + config.tile_size_m - 1) // config.tile_size_m
        num_tiles_n = (intermediate_dim + config.tile_size_n - 1) // config.tile_size_n
        total_tiles = num_tiles_m * num_tiles_n

        # Simulate SIMDgroup work
        work_per_simdgroup = total_tiles / config.num_simdgroups

        # Bit-width affects work (lower bit-width = faster)
        bit_width_factor = 4.0 / config.bit_width

        # Simulate computation time based on config
        computation_time = (
            work_factor * work_per_simdgroup * bit_width_factor * 0.001
        )

        # Actually do some work to prevent optimization
        result = torch.randn(
            batch_size, hidden_dim, dtype=hidden_states.dtype, device=hidden_states.device
        )
        result = result + hidden_states * computation_time

        # Memory operations simulate batching strategy
        if config.batch_strategy == "per_expert":
            # Group by expert (more complex)
            for _ in range(len(torch.unique(expert_indices))):
                _ = result * 0.5
        elif config.batch_strategy == "per_bitwidth":
            # Group by bit-width (simpler)
            _ = result * 0.5
        else:  # hybrid
            # Mix of both
            _ = result * 0.5 + result * 0.3

    def _estimate_memory_usage(
        self,
        config: KernelConfig,
        batch_size: int,
        hidden_dim: int,
        intermediate_dim: int,
    ) -> int:
        """Estimate memory usage for a configuration.

        Args:
            config: Kernel configuration.
            batch_size: Batch size.
            hidden_dim: Hidden dimension.
            intermediate_dim: Intermediate dimension.

        Returns:
            Estimated memory usage in bytes.
        """
        # Base memory for activations
        activation_memory = batch_size * hidden_dim * 2  # FP16

        # Memory for intermediate tensors
        intermediate_memory = batch_size * intermediate_dim * 2

        # Memory for tile buffers (depends on tile size)
        tile_memory = config.tile_size_m * config.tile_size_n * 2

        # SIMDgroup buffers
        simdgroup_memory = config.num_simdgroups * tile_memory

        # Expert routing metadata
        routing_memory = batch_size * 8  # expert indices and weights

        total_memory = (
            activation_memory
            + intermediate_memory
            + simdgroup_memory
            + routing_memory
        )

        return total_memory

    def run_autotune(
        self,
        bit_widths: List[int],
        batch_sizes: List[int],
        hidden_dims: List[int],
        intermediate_dims: Optional[List[int]] = None,
        num_experts: int = 8,
        kernel_variants: Optional[List[str]] = None,
        batch_strategies: Optional[List[str]] = None,
    ) -> Dict[Tuple[int, int, int], OptimizedConfig]:
        """Run full auto-tuning for mixed bit-width kernels.

        This is the main entry point for auto-tuning. It generates all
        configurations, benchmarks them, and selects the optimal configuration
        for each (batch_size, bit_width, hidden_dim) combination.

        Args:
            bit_widths: List of bit widths to tune for.
            batch_sizes: List of batch sizes to benchmark.
            hidden_dims: List of hidden dimensions to benchmark.
            intermediate_dims: List of intermediate dimensions (uses 4*hidden if None).
            num_experts: Number of experts to simulate.
            kernel_variants: List of kernel variants to benchmark.
            batch_strategies: List of batch strategies to benchmark.

        Returns:
            Dictionary mapping (batch_size, bit_width, hidden_dim) to optimized configs.
        """
        logger.info(
            f"Starting auto-tuning for device {self.device}: "
            f"{len(bit_widths)} bit-widths, {len(batch_sizes)} batch sizes, "
            f"{len(hidden_dims)} hidden dims"
        )

        intermediate_dims = intermediate_dims or [h * 4 for h in hidden_dims]

        # Generate all configurations
        configs = self.generate_configs(bit_widths, kernel_variants, batch_strategies)
        logger.info(f"Will benchmark {len(configs)} configurations")

        # Benchmark each configuration for each workload
        total_configs = len(configs) * len(batch_sizes) * len(hidden_dims)
        completed = 0

        for bit_width in bit_widths:
            for batch_size in batch_sizes:
                for idx, hidden_dim in enumerate(hidden_dims):
                    intermediate_dim = intermediate_dims[idx]

                    # Filter configs for this bit width
                    bit_width_configs = [c for c in configs if c.bit_width == bit_width]

                    # Benchmark each config
                    best_result = None
                    best_latency = float("inf")

                    for config in bit_width_configs:
                        result = self.benchmark_config(
                            config,
                            batch_size,
                            hidden_dim,
                            intermediate_dim,
                            num_experts,
                        )
                        self.benchmark_results.append(result)
                        completed += 1

                        if result.success and result.latency_ms < best_latency:
                            best_latency = result.latency_ms
                            best_result = result

                        # Progress logging
                        if completed % 10 == 0:
                            logger.info(
                                f"Progress: {completed}/{total_configs} "
                                f"({100 * completed / total_configs:.1f}%)"
                            )

                    # Select best config for this workload
                    if best_result and best_result.success:
                        optimized = OptimizedConfig(
                            bit_width=bit_width,
                            batch_size=batch_size,
                            hidden_dim=hidden_dim,
                            kernel_variant=best_result.config.kernel_variant,
                            tile_size_m=best_result.config.tile_size_m,
                            tile_size_n=best_result.config.tile_size_n,
                            num_simdgroups=best_result.config.num_simdgroups,
                            batch_strategy=best_result.config.batch_strategy,
                            expected_latency_ms=best_result.latency_ms,
                            throughput=best_result.throughput,
                        )
                        self.optimized_configs[optimized.lookup_key] = optimized

        logger.info(
            f"Auto-tuning complete: Found {len(self.optimized_configs)} "
            f"optimized configurations"
        )

        return self.optimized_configs

    def get_optimal_config(
        self,
        batch_size: int,
        bit_width: int,
        hidden_dim: int,
    ) -> Optional[OptimizedConfig]:
        """Get the optimal configuration for a specific workload.

        Args:
            batch_size: Batch size.
            bit_width: Bit width.
            hidden_dim: Hidden dimension.

        Returns:
            Optimal configuration, or None if not found.
        """
        lookup_key = (batch_size, bit_width, hidden_dim)

        # Try exact match first
        if lookup_key in self.optimized_configs:
            return self.optimized_configs[lookup_key]

        # Find nearest match (fallback)
        nearest_key = self._find_nearest_key(lookup_key)
        if nearest_key:
            logger.info(
                f"No exact match for {lookup_key}, using nearest: {nearest_key}"
            )
            return self.optimized_configs[nearest_key]

        logger.warning(f"No configuration found for {lookup_key}")
        return None

    def _find_nearest_key(
        self,
        target_key: Tuple[int, int, int],
    ) -> Optional[Tuple[int, int, int]]:
        """Find the nearest configuration key to the target.

        Bit width is heavily prioritized as different bit widths require
        different kernel implementations. Batch size is next most important.
        Hidden dimension is least important as kernels can often work with
        different hidden dims with similar performance.

        Args:
            target_key: (batch_size, bit_width, hidden_dim) to find nearest for.

        Returns:
            Nearest key, or None if no configurations exist.
        """
        if not self.optimized_configs:
            return None

        target_batch, target_bit, target_hidden = target_key

        # Calculate distance to each key
        def distance(key):
            batch, bit, hidden = key
            # Bit width is MOST important (must match or close)
            # Use exponential penalty for bit width mismatch
            bit_diff = abs(bit - target_bit) * 1000
            # Batch size is next important
            batch_diff = abs(batch - target_batch) * 100
            # Hidden dimension is least important
            # Use relative difference as a fraction
            if hidden > 0 and target_hidden > 0:
                hidden_diff = abs(hidden - target_hidden) * 0.1
            else:
                hidden_diff = 0
            return bit_diff + batch_diff + hidden_diff

        nearest = min(self.optimized_configs.keys(), key=distance)
        return nearest

    def record_online_sample(
        self,
        batch_size: int,
        bit_width: int,
        hidden_dim: int,
        config_id: str,
        latency_ms: float,
    ) -> None:
        """Record an online runtime sample for adaptation.

        Args:
            batch_size: Batch size.
            bit_width: Bit width.
            hidden_dim: Hidden dimension.
            config_id: Configuration ID used.
            latency_ms: Actual latency observed.
        """
        if not self.enable_online_adaptation:
            return

        lookup_key = (batch_size, bit_width, hidden_dim)

        if lookup_key not in self.online_history:
            self.online_history[lookup_key] = []

        self.online_history[lookup_key].append((config_id, latency_ms))

        # Keep only recent history
        if len(self.online_history[lookup_key]) > self.history_length:
            self.online_history[lookup_key] = self.online_history[lookup_key][
                -self.history_length :
            ]

        # Update optimized config if we have enough samples
        if len(self.online_history[lookup_key]) >= 10:
            self._update_config_from_history(lookup_key)

    def _update_config_from_history(
        self,
        lookup_key: Tuple[int, int, int],
    ) -> None:
        """Update optimized configuration from online history samples.

        Args:
            lookup_key: (batch_size, bit_width, hidden_dim) key.
        """
        if lookup_key not in self.online_history:
            return

        samples = self.online_history[lookup_key]

        # Group samples by config_id and calculate average latency
        config_latencies: Dict[str, List[float]] = {}
        for config_id, latency in samples:
            if config_id not in config_latencies:
                config_latencies[config_id] = []
            config_latencies[config_id].append(latency)

        # Find best performing config
        best_config_id = None
        best_latency = float("inf")

        for config_id, latencies in config_latencies.items():
            avg_latency = statistics.mean(latencies)
            if avg_latency < best_latency:
                best_latency = avg_latency
                best_config_id = config_id

        # Update optimized config if we have a better one
        if best_config_id and lookup_key in self.optimized_configs:
            current = self.optimized_configs[lookup_key]
            current_config_id = (
                f"{current.bit_width}bit_"
                f"{current.tile_size_m}x{current.tile_size_n}_"
                f"{current.num_simdgroups}sg_"
                f"{current.kernel_variant}_"
                f"{current.batch_strategy}"
            )

            if best_config_id != current_config_id and best_latency < current.expected_latency_ms:
                logger.info(
                    f"Updating config for {lookup_key}: "
                    f"{current_config_id} -> {best_config_id} "
                    f"({current.expected_latency_ms:.3f}ms -> {best_latency:.3f}ms)"
                )

                # Parse config_id to update OptimizedConfig
                # Format: {bit_width}bit_{tile_m}x{tile_n}_{num_simd}sg_{variant}_{strategy}
                parts = best_config_id.split("_")

                updated = OptimizedConfig(
                    bit_width=current.bit_width,
                    batch_size=current.batch_size,
                    hidden_dim=current.hidden_dim,
                    kernel_variant=parts[3],
                    tile_size_m=int(parts[1].split("x")[0]),
                    tile_size_n=int(parts[1].split("x")[1]),
                    num_simdgroups=int(parts[2].replace("sg", "")),
                    batch_strategy=parts[4],
                    expected_latency_ms=best_latency,
                    throughput=(current.batch_size * current.hidden_dim) / (best_latency / 1000),
                )

                self.optimized_configs[lookup_key] = updated

    def export_config(self, path: str) -> None:
        """Export optimized configurations to JSON file.

        Args:
            path: Path to output JSON file.
        """
        output = {
            "device": self.device,
            "tuner_params": {
                "warmup_iterations": self.warmup_iterations,
                "benchmark_iterations": self.benchmark_iterations,
                "enable_online_adaptation": self.enable_online_adaptation,
                "history_length": self.history_length,
            },
            "optimized_configs": {
                f"{key[0]}_{key[1]}_{key[2]}": config.to_dict()
                for key, config in self.optimized_configs.items()
            },
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Exported {len(self.optimized_configs)} configs to {path}")

    def load_config(self, path: str) -> None:
        """Load optimized configurations from JSON file.

        Args:
            path: Path to JSON configuration file.
        """
        with open(path, "r") as f:
            data = json.load(f)

        # Verify device compatibility
        if "device" in data and data["device"] != self.device:
            logger.warning(
                f"Loading config from device {data['device']} "
                f"on device {self.device}. Results may not be optimal."
            )

        # Load optimized configs
        if "optimized_configs" in data:
            for key_str, config_dict in data["optimized_configs"].items():
                batch_size, bit_width, hidden_dim = map(int, key_str.split("_"))
                config = OptimizedConfig(**config_dict)
                self.optimized_configs[(batch_size, bit_width, hidden_dim)] = config

        logger.info(
            f"Loaded {len(self.optimized_configs)} configs from {path} "
            f"(source device: {data.get('device', 'unknown')})"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the auto-tuner.

        Returns:
            Dictionary with tuner statistics.
        """
        return {
            "device": self.device,
            "num_benchmarked_configs": len(self.benchmark_results),
            "num_optimized_configs": len(self.optimized_configs),
            "num_online_samples": sum(len(samples) for samples in self.online_history.values()),
            "successful_benchmarks": sum(1 for r in self.benchmark_results if r.success),
            "failed_benchmarks": sum(1 for r in self.benchmark_results if not r.success),
        }


def get_optimal_config(
    batch_size: int,
    bit_width: int,
    hidden_dim: int,
    config_path: Optional[str] = None,
) -> Optional[KernelConfig]:
    """Get optimal kernel configuration for a workload.

    Convenience function that loads configs from file and returns the optimal
    configuration for the given workload.

    Args:
        batch_size: Batch size.
        bit_width: Bit width.
        hidden_dim: Hidden dimension.
        config_path: Path to JSON config file (uses default if None).

    Returns:
        Optimal kernel configuration, or None if not found.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "default_mixed_bpw_configs.json"

    if not Path(config_path).exists():
        logger.warning(f"Config file not found: {config_path}")
        return None

    tuner = MixedBPWAutoTuner()
    tuner.load_config(str(config_path))
    return tuner.get_optimal_config(batch_size, bit_width, hidden_dim)


def HAS_METAL_AVAILABLE() -> bool:
    """Check if Metal is available."""
    try:
        import Metal

        return True
    except ImportError:
        return False


# Default singleton tuner instance for convenience
_default_tuner: Optional[MixedBPWAutoTuner] = None


def get_default_tuner() -> MixedBPWAutoTuner:
    """Get the default singleton tuner instance."""
    global _default_tuner
    if _default_tuner is None:
        _default_tuner = MixedBPWAutoTuner()
    return _default_tuner
