"""System memory introspection for adaptive batch sizing.

Queries available system memory (RAM and GPU) for dynamic resource allocation
on macOS and Linux. On Apple Silicon, GPU shares Unified Memory with the CPU.

Usage:
    from metal_marlin.utils import get_system_memory, compute_optimal_batch_size

    mem = get_system_memory()
    print(f"Available: {mem.available_ram_gb:.1f} GB RAM, {mem.gpu_available_gb:.1f} GB GPU")

    batch_size = compute_optimal_batch_size(
        tensor_shapes=[(4096, 4096), (4096, 11008)],
        available_memory_gb=mem.gpu_available_gb,
    )
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class MemoryInfo:
    """System memory information for resource allocation.

    Attributes:
        total_ram_gb: Total system RAM in gigabytes.
        available_ram_gb: Currently available (free + reclaimable) RAM in GB.
        gpu_total_gb: Total GPU memory in GB. On Apple Silicon, equals total_ram_gb.
        gpu_available_gb: Available GPU memory in GB. On Apple Silicon, estimated
            from available RAM minus a headroom factor.
        recommended_batch_tensors: Suggested number of tensor batches that fit
            comfortably in available memory with safety margin.
    """

    total_ram_gb: float
    available_ram_gb: float
    gpu_total_gb: float
    gpu_available_gb: float
    recommended_batch_tensors: int


def get_system_memory(*, refresh: bool = False) -> MemoryInfo:
    """Query system memory (macOS/Linux).

    On macOS with Apple Silicon, queries unified memory via sysctl.
    On Linux, uses /proc/meminfo.

    Args:
        refresh: If True, bypass cache and re-query the system.
            Default False for performance; memory stats are cached for 1 call.

    Returns:
        MemoryInfo with current memory statistics.

    Note:
        On Apple Silicon, GPU memory is unified with RAM. The gpu_available_gb
        estimate accounts for typical Metal/MLX allocation overhead.
    """
    if refresh:
        _get_system_memory_impl.cache_clear()
    return _get_system_memory_impl()


@lru_cache(maxsize=1)
def _get_system_memory_impl() -> MemoryInfo:
    """Internal implementation with caching."""
    system = platform.system()

    if system == "Darwin":
        return _query_macos_memory()
    elif system == "Linux":
        return _query_linux_memory()
    else:
        # Fallback: return safe defaults
        return MemoryInfo(
            total_ram_gb=8.0,
            available_ram_gb=4.0,
            gpu_total_gb=4.0,
            gpu_available_gb=2.0,
            recommended_batch_tensors=4,
        )


def _query_macos_memory() -> MemoryInfo:
    """Query memory on macOS using sysctl and vm_stat.

    On Apple Silicon:
    - hw.memsize gives total physical RAM
    - vm_stat gives page-level memory breakdown
    - GPU shares Unified Memory with CPU
    """
    # Get total memory via sysctl
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        total_bytes = int(result.stdout.strip())
        total_gb = total_bytes / (1024**3)
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        total_gb = 8.0  # Safe fallback

    # Get memory pressure via vm_stat
    available_gb = _parse_vm_stat_available(total_gb)

    # On Apple Silicon, GPU uses Unified Memory
    # Reserve headroom for Metal driver overhead and MLX allocator fragmentation
    # Use 15% of available (not total) memory, minimum 1GB
    gpu_total_gb = total_gb
    metal_headroom = max(1.0, available_gb * 0.15)
    gpu_available_gb = max(1.0, available_gb - metal_headroom)

    # Estimate recommended batch count (conservative)
    recommended = _estimate_batch_count(gpu_available_gb)

    return MemoryInfo(
        total_ram_gb=total_gb,
        available_ram_gb=available_gb,
        gpu_total_gb=gpu_total_gb,
        gpu_available_gb=gpu_available_gb,
        recommended_batch_tensors=recommended,
    )


def _parse_vm_stat_available(total_gb: float) -> float:
    """Parse vm_stat output to estimate available memory."""
    try:
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = result.stdout.strip().split("\n")

        # Parse page size from header: "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
        page_size = 16384  # Default for Apple Silicon
        if "page size of" in lines[0]:
            import re

            match = re.search(r"page size of (\d+) bytes", lines[0])
            if match:
                page_size = int(match.group(1))

        # Parse page counts
        stats: dict[str, int] = {}
        for line in lines[1:]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            # Remove trailing period and whitespace
            value = value.strip().rstrip(".")
            try:
                stats[key] = int(value)
            except ValueError:
                continue

        # Available = free + speculative + (some of inactive)
        # "Pages free" + "Pages speculative" are immediately available
        # "Pages purgeable" can be reclaimed
        free_pages = stats.get("pages free", 0)
        speculative_pages = stats.get("pages speculative", 0)
        purgeable_pages = stats.get("pages purgeable", 0)

        available_pages = free_pages + speculative_pages + purgeable_pages
        available_bytes = available_pages * page_size
        available_gb = available_bytes / (1024**3)

        # Sanity check
        if available_gb < 0.5 or available_gb > total_gb:
            return total_gb * 0.3  # Fallback: assume 30% available

        return available_gb

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return total_gb * 0.3


def _query_linux_memory() -> MemoryInfo:
    """Query memory on Linux via /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                # Parse value like "16384 kB"
                parts = value.strip().split()
                if parts:
                    try:
                        meminfo[key] = int(parts[0])  # Value in kB
                    except ValueError:
                        continue

        total_kb = meminfo.get("MemTotal", 8 * 1024 * 1024)
        available_kb = meminfo.get("MemAvailable", total_kb // 2)

        total_gb = total_kb / (1024**2)
        available_gb = available_kb / (1024**2)

        # On Linux, GPU memory is separate. Without NVIDIA tools, assume conservative estimate.
        # This would need nvidia-smi parsing for accurate NVIDIA GPU stats.
        gpu_total_gb = 0.0
        gpu_available_gb = 0.0

        recommended = _estimate_batch_count(available_gb)

        return MemoryInfo(
            total_ram_gb=total_gb,
            available_ram_gb=available_gb,
            gpu_total_gb=gpu_total_gb,
            gpu_available_gb=gpu_available_gb,
            recommended_batch_tensors=recommended,
        )

    except (FileNotFoundError, PermissionError):
        return MemoryInfo(
            total_ram_gb=8.0,
            available_ram_gb=4.0,
            gpu_total_gb=0.0,
            gpu_available_gb=0.0,
            recommended_batch_tensors=4,
        )


def _estimate_batch_count(available_gb: float) -> int:
    """Estimate how many tensor batches fit in available memory.

    Assumes typical transformer weights (~2GB per batch of medium layers).
    Returns a conservative count suitable for prefetching decisions.
    """
    # Rough estimate: 2GB per batch of typical transformer weights
    # Includes model weights, activations, and working memory
    gb_per_batch = 2.0
    count = int(available_gb / gb_per_batch)
    return max(1, min(count, 16))  # Clamp to [1, 16]


def estimate_tensor_memory(
    shape: tuple[int, ...],
    dtype: np.dtype | type | str = np.float16,
) -> int:
    """Estimate memory for a tensor in bytes.

    Args:
        shape: Tensor dimensions (e.g., (4096, 4096) for a weight matrix).
        dtype: NumPy dtype or string. Default: float16.

    Returns:
        Memory footprint in bytes.

    Example:
        >>> estimate_tensor_memory((4096, 4096), np.float16)
        33554432  # 32 MB
    """
    dtype = np.dtype(dtype)
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    return num_elements * dtype.itemsize


def compute_optimal_batch_size(
    tensor_shapes: list[tuple[int, ...]],
    available_memory_gb: float,
    dtype: np.dtype | type | str = np.float16,
    safety_factor: float = 0.7,
) -> int:
    """Compute batch size that fits in memory with safety margin.

    Given a list of tensor shapes that will be allocated per batch, computes
    how many batches can fit in available memory.

    Args:
        tensor_shapes: List of tensor shapes, one per tensor allocated per batch.
            For quantized inference, include weight matrices and activation buffers.
        available_memory_gb: Available GPU/RAM in gigabytes.
        dtype: Data type for tensors. Default: float16.
        safety_factor: Fraction of memory to use (0.0-1.0). Default: 0.7.
            The remaining 30% provides headroom for fragmentation and runtime allocations.

    Returns:
        Maximum batch size that fits, at least 1.

    Example:
        >>> shapes = [(4096, 4096), (4096, 11008), (11008, 4096)]  # Typical LLaMA MLP
        >>> compute_optimal_batch_size(shapes, available_memory_gb=12.0)
        54
    """
    if not tensor_shapes:
        return 1

    # Sum memory per batch
    dtype = np.dtype(dtype)
    bytes_per_batch = sum(estimate_tensor_memory(shape, dtype) for shape in tensor_shapes)

    if bytes_per_batch == 0:
        return 1

    # Available bytes with safety margin
    available_bytes = available_memory_gb * (1024**3) * safety_factor

    batch_size = int(available_bytes / bytes_per_batch)
    return max(1, batch_size)


def get_metal_memory_pressure() -> str | None:
    """Query current Metal memory pressure level on macOS.

    Uses the memory_pressure utility to check system memory state.

    Returns:
        One of "normal", "warn", "critical", or None if unavailable.
    """
    if platform.system() != "Darwin":
        return None

    try:
        result = subprocess.run(
            ["memory_pressure"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stdout.lower()

        if "critical" in output:
            return "critical"
        elif "warn" in output:
            return "warn"
        elif "normal" in output:
            return "normal"
        return None

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def should_reduce_allocation(threshold: str = "warn") -> bool:
    """Check if memory pressure indicates allocation should be reduced.

    Args:
        threshold: Pressure level at which to return True.
            "warn" (default): True if warn or critical.
            "critical": True only if critical.

    Returns:
        True if current pressure meets or exceeds threshold.
    """
    pressure = get_metal_memory_pressure()
    if pressure is None:
        return False

    if threshold == "critical":
        return pressure == "critical"
    else:  # "warn" threshold
        return pressure in ("warn", "critical")
