"""
Kernel profiling utilities.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from ._compat import HAS_TORCH, torch


def _gpu_sync() -> None:
    """Synchronize GPU if torch MPS is available."""
    if HAS_TORCH and torch is not None and torch.backends.mps.is_available():
        torch.mps.synchronize()


@dataclass
class KernelProfile:
    name: str
    gpu_time_ms: float
    cpu_time_ms: float
    memory_allocated_mb: float


_profiles: list[KernelProfile] = []


@contextmanager
def profile_kernel(name: str) -> Iterator[None]:
    """Profile a kernel execution."""
    # Force sync before
    _gpu_sync()

    cpu_start = time.perf_counter()

    yield  # Execute kernel

    # Force sync after
    _gpu_sync()
    cpu_end = time.perf_counter()

    profile = KernelProfile(
        name=name,
        gpu_time_ms=0,  # Would need Metal profiling API
        cpu_time_ms=(cpu_end - cpu_start) * 1000,
        memory_allocated_mb=0,
    )
    _profiles.append(profile)


def print_profiles() -> None:
    """Print collected profiles."""
    print(f"{'Kernel':<40} {'CPU Time (ms)':>15}")
    print("-" * 55)
    for p in _profiles:
        print(f"{p.name:<40} {p.cpu_time_ms:>15.3f}")


def clear_profiles() -> None:
    """Clear collected profiles."""
    _profiles.clear()


# Usage example:
# with profile_kernel("marlin_gemm_fp4"):
#     result = marlin_gemm_fp4(A, B, scales)
# print_profiles()
