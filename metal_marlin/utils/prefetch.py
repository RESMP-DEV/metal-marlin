"""Memory-aware tensor prefetching for quantization.

This module provides adaptive prefetching of tensors from safetensors files,
optimizing memory usage by loading batches that fit within available RAM.

Usage:
    from metal_marlin.utils.prefetch import AdaptivePrefetcher, get_system_memory

    # Check available memory
    mem = get_system_memory()
    print(f"Available: {mem.available_ram_gb:.1f} GB")

    # Iterate over tensors with adaptive prefetching
    st_files = list(Path("model/").glob("*.safetensors"))
    prefetcher = AdaptivePrefetcher(st_files, target_memory_gb=48.0)

    for name, tensor in prefetcher:
        # Process tensor while next batch loads in background
        quantized = quantize_fp4(tensor)
"""

from __future__ import annotations

import platform
import struct
import subprocess
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import numpy as np


@dataclass
class SystemMemoryInfo:
    """System memory information."""

    total_ram_gb: float
    available_ram_gb: float
    used_ram_gb: float
    swap_total_gb: float
    swap_used_gb: float
    # Metal-specific (macOS only)
    metal_available_gb: float | None = None

    @property
    def usage_percent(self) -> float:
        """Memory usage as a percentage."""
        if self.total_ram_gb == 0:
            return 0.0
        return (self.used_ram_gb / self.total_ram_gb) * 100

    @property
    def available_with_pressure(self) -> float:
        """Available memory accounting for memory pressure.

        On macOS, accounts for Metal unified memory.
        On Linux, uses available memory directly.
        """
        if self.metal_available_gb is not None:
            # Unified memory: use minimum of system available and Metal available
            return min(self.available_ram_gb, self.metal_available_gb)
        return self.available_ram_gb


def get_system_memory() -> SystemMemoryInfo:
    """Query current system memory status.

    Returns:
        SystemMemoryInfo with current memory statistics.

    Note:
        On macOS, also queries Metal unified memory availability.
        Falls back to psutil if available, otherwise uses OS-specific methods.
    """
    system = platform.system()

    if system == "Darwin":
        return _get_macos_memory()
    elif system == "Linux":
        return _get_linux_memory()
    else:
        # Fallback to psutil if available
        return _get_psutil_memory()


def _get_macos_memory() -> SystemMemoryInfo:
    """Get memory info on macOS using vm_stat and sysctl."""
    # Get page size and vm_stat
    try:
        page_size = int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"]).strip())
    except (subprocess.CalledProcessError, ValueError):
        page_size = 16384  # Default for Apple Silicon

    try:
        vm_stat = subprocess.check_output(["vm_stat"]).decode()
    except subprocess.CalledProcessError:
        return _get_psutil_memory()

    # Parse vm_stat output
    stats: dict[str, int] = {}
    for line in vm_stat.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            # Remove trailing period and parse
            value = value.strip().rstrip(".")
            try:
                stats[key] = int(value)
            except ValueError:
                pass

    # Calculate memory values (in pages)
    free_pages = stats.get("pages_free", 0)
    active_pages = stats.get("pages_active", 0)
    inactive_pages = stats.get("pages_inactive", 0)
    speculative_pages = stats.get("pages_speculative", 0)
    wired_pages = stats.get("pages_wired_down", 0)
    compressed_pages = stats.get("pages_occupied_by_compressor", 0)
    purgeable_pages = stats.get("pages_purgeable", 0)

    # Total physical memory
    try:
        total_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
    except (subprocess.CalledProcessError, ValueError):
        # Estimate from vm_stat
        total_pages = free_pages + active_pages + inactive_pages + speculative_pages + wired_pages
        total_bytes = total_pages * page_size

    # Available = free + inactive + purgeable (can be reclaimed)
    available_pages = free_pages + inactive_pages + purgeable_pages + speculative_pages
    available_bytes = available_pages * page_size

    # Used = active + wired + compressed
    used_pages = active_pages + wired_pages + compressed_pages
    used_bytes = used_pages * page_size

    # Swap info
    try:
        swap_output = subprocess.check_output(["sysctl", "-n", "vm.swapusage"]).decode()
        # Format: "total = 2048.00M  used = 512.00M  free = 1536.00M"
        swap_total = 0.0
        swap_used = 0.0
        for part in swap_output.split():
            if part.endswith("M"):
                val = float(part[:-1]) / 1024  # Convert MB to GB
                if "total" in swap_output.split()[swap_output.split().index(part) - 2]:
                    swap_total = val
                elif "used" in swap_output.split()[swap_output.split().index(part) - 2]:
                    swap_used = val
    except (subprocess.CalledProcessError, ValueError, IndexError):
        swap_total = 0.0
        swap_used = 0.0

    # Metal unified memory (Apple Silicon)
    metal_available = None
    try:
        # On Apple Silicon, Metal can use unified memory
        # Recommended GPU memory is typically 75% of total for safe operation
        if _is_apple_silicon():
            metal_available = (total_bytes * 0.75) / (1024**3)
    except Exception:
        pass

    return SystemMemoryInfo(
        total_ram_gb=total_bytes / (1024**3),
        available_ram_gb=available_bytes / (1024**3),
        used_ram_gb=used_bytes / (1024**3),
        swap_total_gb=swap_total,
        swap_used_gb=swap_used,
        metal_available_gb=metal_available,
    )


def _get_linux_memory() -> SystemMemoryInfo:
    """Get memory info on Linux from /proc/meminfo."""
    meminfo: dict[str, int] = {}

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    # Values are in kB
                    try:
                        meminfo[key] = int(parts[1])
                    except ValueError:
                        pass
    except OSError:
        return _get_psutil_memory()

    total_kb = meminfo.get("MemTotal", 0)
    available_kb = meminfo.get("MemAvailable", 0)
    free_kb = meminfo.get("MemFree", 0)
    buffers_kb = meminfo.get("Buffers", 0)
    cached_kb = meminfo.get("Cached", 0)
    swap_total_kb = meminfo.get("SwapTotal", 0)
    swap_free_kb = meminfo.get("SwapFree", 0)

    # If MemAvailable not present (older kernels), estimate it
    if available_kb == 0:
        available_kb = free_kb + buffers_kb + cached_kb

    used_kb = total_kb - available_kb

    return SystemMemoryInfo(
        total_ram_gb=total_kb / (1024**2),
        available_ram_gb=available_kb / (1024**2),
        used_ram_gb=used_kb / (1024**2),
        swap_total_gb=swap_total_kb / (1024**2),
        swap_used_gb=(swap_total_kb - swap_free_kb) / (1024**2),
    )


def _get_psutil_memory() -> SystemMemoryInfo:
    """Fallback memory info using psutil."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return SystemMemoryInfo(
            total_ram_gb=vm.total / (1024**3),
            available_ram_gb=vm.available / (1024**3),
            used_ram_gb=vm.used / (1024**3),
            swap_total_gb=swap.total / (1024**3),
            swap_used_gb=swap.used / (1024**3),
        )
    except ImportError:
        # Last resort: estimate from /dev/null or just return zeros
        return SystemMemoryInfo(
            total_ram_gb=0.0,
            available_ram_gb=0.0,
            used_ram_gb=0.0,
            swap_total_gb=0.0,
            swap_used_gb=0.0,
        )


def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    if platform.system() != "Darwin":
        return False
    try:
        arch = subprocess.check_output(["uname", "-m"]).decode().strip()
        return arch == "arm64"
    except subprocess.CalledProcessError:
        return False


@dataclass
class TensorMetadata:
    """Metadata for a tensor in a safetensors file."""

    name: str
    file_path: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    size_bytes: int

    @property
    def numel(self) -> int:
        """Total number of elements."""
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def size_gb(self) -> float:
        """Size in gigabytes."""
        return self.size_bytes / (1024**3)


def _parse_safetensors_metadata(file_path: Path) -> list[TensorMetadata]:
    """Parse tensor metadata from safetensors file without loading tensors.

    Safetensors format:
    - First 8 bytes: header size (uint64 little-endian)
    - Next N bytes: JSON header with tensor metadata
    - Remaining: tensor data

    Returns:
        List of TensorMetadata for all tensors in the file.
    """
    tensors: list[TensorMetadata] = []

    with open(file_path, "rb") as f:
        # Read header size
        header_size_bytes = f.read(8)
        if len(header_size_bytes) < 8:
            return tensors
        header_size = struct.unpack("<Q", header_size_bytes)[0]

        # Read and parse JSON header
        header_bytes = f.read(header_size)
        if len(header_bytes) < header_size:
            return tensors

        import json

        header = json.loads(header_bytes.decode("utf-8"))

        # Data starts after header
        data_offset = 8 + header_size

        # Parse tensor metadata
        for name, info in header.items():
            if name == "__metadata__":
                continue

            dtype = info.get("dtype", "F16")
            shape = tuple(info.get("shape", []))
            offsets = info.get("data_offsets", [0, 0])

            # Calculate byte size
            dtype_sizes = {
                "F16": 2,
                "F32": 4,
                "F64": 8,
                "BF16": 2,
                "I8": 1,
                "I16": 2,
                "I32": 4,
                "I64": 8,
                "U8": 1,
                "U16": 2,
                "U32": 4,
                "U64": 8,
                "BOOL": 1,
            }
            elem_size = dtype_sizes.get(dtype, 2)

            numel = 1
            for dim in shape:
                numel *= dim
            size_bytes = numel * elem_size

            tensors.append(
                TensorMetadata(
                    name=name,
                    file_path=file_path,
                    dtype=dtype,
                    shape=shape,
                    offset=data_offset + offsets[0],
                    size_bytes=size_bytes,
                )
            )

    return tensors


class AdaptivePrefetcher:
    """Prefetch tensors based on available memory.

    Adaptively loads tensors from safetensors files, batching them to fit
    within a memory budget. While you process the current batch, the next
    batch loads in a background thread.

    Args:
        st_files: List of safetensors file paths to iterate over.
        target_memory_gb: Maximum memory to use for prefetching. If None,
            uses 60% of available RAM.
        prefetch_factor: Number of batches to prefetch ahead. Default 1.
        filter_fn: Optional function (name, metadata) -> bool to filter tensors.
            Return True to include the tensor.

    Example:
        >>> st_files = list(Path("model/").glob("*.safetensors"))
        >>> prefetcher = AdaptivePrefetcher(st_files, target_memory_gb=48.0)
        >>> for name, tensor in prefetcher:
        ...     # tensor is a numpy array
        ...     packed, scales = quantize_fp4(tensor)

        # With filtering (only weight tensors):
        >>> prefetcher = AdaptivePrefetcher(
        ...     st_files,
        ...     filter_fn=lambda n, m: "weight" in n and m.shape is not None and len(m.shape) == 2
        ... )
    """

    def __init__(
        self,
        st_files: list[Path],
        target_memory_gb: float | None = None,
        prefetch_factor: int = 1,
        filter_fn: Any | None = None,
    ):
        self.st_files = [Path(f) for f in st_files]
        self.prefetch_factor = max(1, prefetch_factor)
        self.filter_fn = filter_fn

        # Determine memory budget
        if target_memory_gb is None:
            mem_info = get_system_memory()
            target_memory_gb = mem_info.available_with_pressure * 0.6
        self.target_memory_bytes = int(target_memory_gb * 1024**3)

        # Parse all tensor metadata upfront (fast, no data loading)
        self._all_tensors: list[TensorMetadata] = []
        for st_file in self.st_files:
            tensors = _parse_safetensors_metadata(st_file)
            for t in tensors:
                if self.filter_fn is None or self.filter_fn(t.name, t):
                    self._all_tensors.append(t)

        # Sort by file then offset for sequential reads
        self._all_tensors.sort(key=lambda t: (str(t.file_path), t.offset))

        # Prefetch state
        self._current_idx = 0
        self._prefetch_buffer: list[tuple[str, np.ndarray]] = []
        self._buffer_lock = Lock()
        self._prefetch_event = Event()
        self._stop_event = Event()
        self._prefetch_thread: Thread | None = None
        self._executor: ThreadPoolExecutor | None = None

    def __len__(self) -> int:
        """Total number of tensors to iterate."""
        return len(self._all_tensors)

    @property
    def total_size_gb(self) -> float:
        """Total size of all tensors in GB."""
        return sum(t.size_bytes for t in self._all_tensors) / (1024**3)

    def _load_tensor(self, metadata: TensorMetadata) -> tuple[str, np.ndarray]:
        """Load a single tensor from disk."""
        from safetensors import safe_open

        with safe_open(str(metadata.file_path), framework="numpy") as f:
            tensor = f.get_tensor(metadata.name)
        return metadata.name, tensor

    def _compute_batch(self, start_idx: int) -> list[TensorMetadata]:
        """Compute which tensors fit in the next batch."""
        batch: list[TensorMetadata] = []
        batch_bytes = 0

        for i in range(start_idx, len(self._all_tensors)):
            t = self._all_tensors[i]
            # Reserve 2x tensor size for processing overhead
            needed = t.size_bytes * 2
            if batch_bytes + needed > self.target_memory_bytes and batch:
                break
            batch.append(t)
            batch_bytes += needed

        return batch

    def prefetch_batch(self, batch_size: int | None = None) -> list[tuple[str, np.ndarray]]:
        """Load batch of tensors into memory.

        Args:
            batch_size: Number of tensors to load. If None, loads as many
                as fit in the memory budget.

        Returns:
            List of (name, tensor) tuples.
        """
        if batch_size is not None:
            # Load exactly batch_size tensors
            end_idx = min(self._current_idx + batch_size, len(self._all_tensors))
            batch_meta = self._all_tensors[self._current_idx : end_idx]
        else:
            # Load as many as fit in memory
            batch_meta = self._compute_batch(self._current_idx)

        if not batch_meta:
            return []

        # Load tensors (use thread pool for parallel file reads)
        result: list[tuple[str, np.ndarray]] = []

        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=4)

        futures = [self._executor.submit(self._load_tensor, m) for m in batch_meta]
        for future in futures:
            result.append(future.result())

        self._current_idx += len(batch_meta)
        return result

    def _prefetch_worker(self) -> None:
        """Background worker for prefetching."""
        while not self._stop_event.is_set():
            # Wait for signal to prefetch
            self._prefetch_event.wait(timeout=0.1)
            if self._stop_event.is_set():
                break
            self._prefetch_event.clear()

            # Check if we need to prefetch
            with self._buffer_lock:
                if len(self._prefetch_buffer) >= self.prefetch_factor:
                    continue
                current_idx = self._current_idx

            # Load next batch
            batch = self._compute_batch(current_idx)
            if not batch:
                continue

            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=4)

            loaded: list[tuple[str, np.ndarray]] = []
            futures = [self._executor.submit(self._load_tensor, m) for m in batch]
            for future in futures:
                if self._stop_event.is_set():
                    break
                loaded.append(future.result())

            with self._buffer_lock:
                self._prefetch_buffer.extend(loaded)
                self._current_idx += len(loaded)

    def __iter__(self) -> Iterator[tuple[str, np.ndarray]]:
        """Yield tensors with prefetching.

        Each iteration yields (tensor_name, numpy_array). Background prefetching
        keeps the next batch loaded while you process the current one.
        """
        self._current_idx = 0
        self._prefetch_buffer = []
        self._stop_event.clear()

        # Start prefetch thread
        self._prefetch_thread = Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()

        # Trigger initial prefetch
        self._prefetch_event.set()

        try:
            yielded = 0
            total = len(self._all_tensors)

            while yielded < total:
                # Try to get item from buffer
                item: tuple[str, np.ndarray] | None = None

                # Poll with timeout to avoid deadlock
                for _ in range(100):  # 10 seconds max wait (100 * 0.1s)
                    with self._buffer_lock:
                        if self._prefetch_buffer:
                            item = self._prefetch_buffer.pop(0)
                            break

                    # Buffer empty - trigger prefetch and wait briefly
                    self._prefetch_event.set()
                    import time
                    time.sleep(0.1)

                    # Early exit check
                    with self._buffer_lock:
                        if yielded >= total:
                            break

                if item is None:
                    # Timeout or completed - break out
                    break

                name, tensor = item

                # Trigger next prefetch before yielding
                self._prefetch_event.set()

                yield name, tensor
                yielded += 1

        finally:
            self._stop_event.set()
            self._prefetch_event.set()  # Wake up thread to exit
            if self._prefetch_thread is not None:
                self._prefetch_thread.join(timeout=1.0)
            if self._executor is not None:
                self._executor.shutdown(wait=False)
                self._executor = None

    def reset(self) -> None:
        """Reset iterator to beginning."""
        self._current_idx = 0
        with self._buffer_lock:
            self._prefetch_buffer.clear()

    def estimate_batches(self) -> int:
        """Estimate number of batches needed to process all tensors."""
        idx = 0
        count = 0
        while idx < len(self._all_tensors):
            batch = self._compute_batch(idx)
            if not batch:
                break
            idx += len(batch)
            count += 1
        return count

    def __enter__(self) -> AdaptivePrefetcher:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - cleanup resources."""
        self._stop_event.set()
        self._prefetch_event.set()
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=1.0)
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
