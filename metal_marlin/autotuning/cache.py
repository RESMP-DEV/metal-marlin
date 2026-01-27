"""Persistent cache for autotuning results.

Stores optimal tile configurations per (GPU fingerprint, problem_size) combination.
The cache is stored as JSON in a platform-appropriate location and loaded on
first use.

Cache Structure:
    {
        "version": 2,
        "gpu_fingerprint": "Apple M2 Max (38 cores, 96GB)",
        "entries": {
            "4096_4096_4096": {
                "config": {...},
                "perf": {"gflops": 1234.5, "bandwidth_gb_s": 456.7}
            }
        }
    }

Usage:
    from metal_marlin.autotuning import AutotuneCache

    cache = AutotuneCache()  # Auto-detects GPU and loads existing cache

    # Get cached config (or None if not cached)
    config = cache.get(M=4096, N=4096, K=4096)

    # Store a new result
    cache.put(M=4096, N=4096, K=4096, config=tile_config, gflops=1234.5)

    # Save cache to disk
    cache.save()
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tile_search import TileConfig

# Cache version - bump when format changes
CACHE_VERSION = 2


@dataclass(frozen=True)
class GPUFingerprint:
    """Identifies a specific GPU for cache keying.

    Attributes:
        name: GPU model name (e.g., "Apple M2 Max").
        cores: Number of GPU cores (affects parallelism).
        memory_gb: Total GPU memory in GB (affects tile sizing).
        metal_family: Metal GPU family (Apple3, Apple7, etc.).
    """

    name: str
    cores: int
    memory_gb: int
    metal_family: str

    def to_key(self) -> str:
        """Convert to a string key for cache file naming."""
        # Normalize name for filesystem
        safe_name = self.name.replace(" ", "_").replace("(", "").replace(")", "")
        return f"{safe_name}_{self.cores}c_{self.memory_gb}GB"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON storage."""
        return {
            "name": self.name,
            "cores": self.cores,
            "memory_gb": self.memory_gb,
            "metal_family": self.metal_family,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GPUFingerprint:
        """Deserialize from JSON."""
        return cls(
            name=data["name"],
            cores=data["cores"],
            memory_gb=data["memory_gb"],
            metal_family=data.get("metal_family", "unknown"),
        )

    def matches(self, other: GPUFingerprint) -> bool:
        """Check if another fingerprint matches this GPU.

        Allows some flexibility in matching (e.g., ignore minor memory differences)
        to reuse cached results across similar configurations.
        """
        if self.name != other.name:
            return False
        if self.cores != other.cores:
            return False
        # Allow 32GB memory variance (different RAM configs on same chip)
        if abs(self.memory_gb - other.memory_gb) > 32:
            return False
        return True


def detect_gpu() -> GPUFingerprint:
    """Detect the current GPU and return its fingerprint.

    On macOS, queries system_profiler for GPU information.
    On Linux, would query nvidia-smi or similar.

    Returns:
        GPUFingerprint for the current system.
    """
    system = platform.system()

    if system == "Darwin":
        return _detect_macos_gpu()
    else:
        # Linux or other - return generic fingerprint
        return GPUFingerprint(
            name="Unknown GPU",
            cores=0,
            memory_gb=0,
            metal_family="unknown",
        )


def _detect_macos_gpu() -> GPUFingerprint:
    """Detect GPU on macOS using system_profiler and sysctl."""
    name = "Apple Silicon"
    cores = 0
    memory_gb = 0
    metal_family = "unknown"

    # Get chip name from sysctl
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        chip_name = result.stdout.strip()
        # Extract M-series identifier
        if "Apple" in chip_name:
            name = chip_name
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Get GPU cores from system_profiler
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        data = json.loads(result.stdout)
        displays = data.get("SPDisplaysDataType", [])
        for display in displays:
            # Look for GPU core count
            if "sppci_cores" in display:
                cores = int(display["sppci_cores"])
            elif "spdisplays_mtlgpufamilysupport" in display:
                metal_family = display["spdisplays_mtlgpufamilysupport"]
            # Parse GPU model name as fallback
            if "sppci_model" in display:
                model = display["sppci_model"]
                if "Apple" in model:
                    name = model
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass

    # Get total memory (GPU shares unified memory on Apple Silicon)
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        memory_bytes = int(result.stdout.strip())
        memory_gb = memory_bytes // (1024**3)
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        memory_gb = 8  # Fallback

    # Estimate GPU cores from chip name if not found
    if cores == 0:
        cores = _estimate_gpu_cores(name)

    # Determine Metal family from chip generation
    if metal_family == "unknown":
        metal_family = _estimate_metal_family(name)

    return GPUFingerprint(
        name=name,
        cores=cores,
        memory_gb=memory_gb,
        metal_family=metal_family,
    )


def _estimate_gpu_cores(chip_name: str) -> int:
    """Estimate GPU cores from Apple Silicon chip name."""
    chip_name = chip_name.lower()

    # M4 family (2024)
    if "m4 ultra" in chip_name:
        return 80
    if "m4 max" in chip_name:
        return 40
    if "m4 pro" in chip_name:
        return 20
    if "m4" in chip_name:
        return 10

    # M3 family (2023)
    if "m3 ultra" in chip_name:
        return 76
    if "m3 max" in chip_name:
        return 40
    if "m3 pro" in chip_name:
        return 18
    if "m3" in chip_name:
        return 10

    # M2 family (2022)
    if "m2 ultra" in chip_name:
        return 76
    if "m2 max" in chip_name:
        return 38
    if "m2 pro" in chip_name:
        return 19
    if "m2" in chip_name:
        return 10

    # M1 family (2020-2021)
    if "m1 ultra" in chip_name:
        return 64
    if "m1 max" in chip_name:
        return 32
    if "m1 pro" in chip_name:
        return 16
    if "m1" in chip_name:
        return 8

    return 8  # Conservative default


def _estimate_metal_family(chip_name: str) -> str:
    """Estimate Metal GPU family from chip name."""
    chip_name = chip_name.lower()

    if "m4" in chip_name:
        return "Apple9"  # M4 is Apple9 GPU family
    if "m3" in chip_name:
        return "Apple8"  # M3 is Apple8 GPU family
    if "m2" in chip_name:
        return "Apple7"  # M2 is Apple7 GPU family
    if "m1" in chip_name:
        return "Apple7"  # M1 is Apple7 GPU family
    if "a14" in chip_name or "a15" in chip_name or "a16" in chip_name:
        return "Apple7"

    return "Apple3"  # Conservative fallback


def get_cache_dir() -> Path:
    """Get the directory for storing autotuning cache files.

    Uses XDG_CACHE_HOME on Linux, ~/Library/Caches on macOS.
    """
    system = platform.system()

    if system == "Darwin":
        cache_base = Path.home() / "Library" / "Caches"
    elif system == "Linux":
        xdg_cache = Path.home() / ".cache"
        cache_base = Path(platform.os.environ.get("XDG_CACHE_HOME", str(xdg_cache)))
    else:
        cache_base = Path.home() / ".cache"

    cache_dir = cache_base / "metal_marlin" / "autotune"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@dataclass
class CacheEntry:
    """A single cached autotuning result."""

    config: TileConfig
    gflops: float = 0.0
    bandwidth_gb_s: float = 0.0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON storage."""
        return {
            "config": {
                "tile_m": self.config.tile_m,
                "tile_n": self.config.tile_n,
                "tile_k": self.config.tile_k,
                "simdgroups_per_tg": self.config.simdgroups_per_tg,
                "threads_per_tg": self.config.threads_per_tg,
                "sg_m_tiles": self.config.sg_m_tiles,
                "sg_n_tiles": self.config.sg_n_tiles,
                "num_buffers": self.config.num_buffers,
            },
            "perf": {
                "gflops": self.gflops,
                "bandwidth_gb_s": self.bandwidth_gb_s,
                "elapsed_ms": self.elapsed_ms,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntry:
        """Deserialize from JSON."""
        cfg = data["config"]
        perf = data.get("perf", {})
        return cls(
            config=TileConfig(
                tile_m=cfg["tile_m"],
                tile_n=cfg["tile_n"],
                tile_k=cfg["tile_k"],
                simdgroups_per_tg=cfg["simdgroups_per_tg"],
                threads_per_tg=cfg["threads_per_tg"],
                sg_m_tiles=cfg["sg_m_tiles"],
                sg_n_tiles=cfg["sg_n_tiles"],
                num_buffers=cfg.get("num_buffers", 2),
            ),
            gflops=perf.get("gflops", 0.0),
            bandwidth_gb_s=perf.get("bandwidth_gb_s", 0.0),
            elapsed_ms=perf.get("elapsed_ms", 0.0),
        )


class AutotuneCache:
    """Persistent cache for autotuning results.

    Automatically detects the current GPU and loads/saves cached results
    specific to that hardware configuration.

    Example:
        cache = AutotuneCache()

        # Check for cached result
        config = cache.get(4096, 4096, 4096)
        if config is None:
            # Run autotuning
            config = searcher.search(4096, 4096, 4096)
            cache.put(4096, 4096, 4096, config, gflops=result.gflops)

        cache.save()
    """

    def __init__(self, cache_dir: Path | None = None, auto_load: bool = True) -> None:
        """Initialize the cache.

        Args:
            cache_dir: Directory for cache files. Uses default if None.
            auto_load: Whether to automatically load existing cache on init.
        """
        self.cache_dir = cache_dir or get_cache_dir()
        self.gpu = detect_gpu()
        self.entries: dict[str, CacheEntry] = {}
        self._dirty = False

        if auto_load:
            self.load()

    def _problem_key(self, M: int, N: int, K: int, group_size: int = 32) -> str:
        """Generate cache key for a problem size."""
        return f"{M}_{N}_{K}_g{group_size}"

    def _cache_path(self) -> Path:
        """Get the cache file path for the current GPU."""
        filename = f"{self.gpu.to_key()}.json"
        return self.cache_dir / filename

    def get(self, M: int, N: int, K: int, group_size: int = 32) -> TileConfig | None:
        """Look up cached config for a problem size.

        Args:
            M: Number of rows.
            N: Number of columns.
            K: Shared dimension.
            group_size: Quantization group size.

        Returns:
            Cached TileConfig if found, None otherwise.
        """
        key = self._problem_key(M, N, K, group_size)
        entry = self.entries.get(key)
        return entry.config if entry else None

    def get_entry(self, M: int, N: int, K: int, group_size: int = 32) -> CacheEntry | None:
        """Look up full cached entry including performance data."""
        key = self._problem_key(M, N, K, group_size)
        return self.entries.get(key)

    def put(
        self,
        M: int,
        N: int,
        K: int,
        config: TileConfig,
        gflops: float = 0.0,
        bandwidth_gb_s: float = 0.0,
        elapsed_ms: float = 0.0,
        group_size: int = 32,
    ) -> None:
        """Store a tuning result in the cache.

        Args:
            M, N, K: Problem dimensions.
            config: Optimal tile configuration.
            gflops: Achieved GFLOPS (for reference).
            bandwidth_gb_s: Achieved memory bandwidth (for reference).
            elapsed_ms: Kernel execution time in ms.
            group_size: Quantization group size.
        """
        key = self._problem_key(M, N, K, group_size)
        self.entries[key] = CacheEntry(
            config=config,
            gflops=gflops,
            bandwidth_gb_s=bandwidth_gb_s,
            elapsed_ms=elapsed_ms,
        )
        self._dirty = True

    def load(self) -> bool:
        """Load cache from disk.

        Returns:
            True if cache was loaded successfully, False otherwise.
        """
        path = self._cache_path()
        if not path.exists():
            return False

        try:
            data = json.loads(path.read_text())

            # Check version
            if data.get("version") != CACHE_VERSION:
                return False

            # Check GPU fingerprint matches
            stored_gpu = GPUFingerprint.from_dict(data.get("gpu", {}))
            if not self.gpu.matches(stored_gpu):
                return False

            # Load entries
            entries = data.get("entries", {})
            for key, entry_data in entries.items():
                self.entries[key] = CacheEntry.from_dict(entry_data)

            self._dirty = False
            return True

        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    def save(self) -> None:
        """Save cache to disk."""
        if not self._dirty:
            return

        data = {
            "version": CACHE_VERSION,
            "gpu": self.gpu.to_dict(),
            "entries": {key: entry.to_dict() for key, entry in self.entries.items()},
        }

        path = self._cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        self._dirty = False

    def clear(self) -> None:
        """Clear all cached entries."""
        self.entries.clear()
        self._dirty = True

    def invalidate(self, M: int, N: int, K: int, group_size: int = 32) -> bool:
        """Remove a specific entry from the cache.

        Returns:
            True if entry was found and removed.
        """
        key = self._problem_key(M, N, K, group_size)
        if key in self.entries:
            del self.entries[key]
            self._dirty = True
            return True
        return False

    def get_or_tune(
        self,
        M: int,
        N: int,
        K: int,
        group_size: int = 32,
        searcher: Any = None,
    ) -> TileConfig:
        """Get cached config or run autotuning if not cached.

        This is the primary entry point for autotuning-aware code.

        Args:
            M, N, K: Problem dimensions.
            group_size: Quantization group size.
            searcher: Optional TileSearcher instance. If None, creates default.

        Returns:
            Optimal TileConfig for this problem size.
        """
        # Try cache first
        config = self.get(M, N, K, group_size)
        if config is not None:
            return config

        # Run autotuning
        if searcher is None:
            from .tile_search import TileSearcher

            searcher = TileSearcher(group_size=group_size)

        config, results = searcher.search_with_logging(M, N, K)

        # Find best result for perf data
        valid = [r for r in results if r.valid and r.config == config]
        if valid:
            best = valid[0]
            self.put(
                M,
                N,
                K,
                config,
                gflops=best.gflops,
                bandwidth_gb_s=best.bandwidth_gb_s,
                elapsed_ms=best.elapsed_ms,
                group_size=group_size,
            )
        else:
            self.put(M, N, K, config, group_size=group_size)

        return config

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, key: tuple[int, int, int]) -> bool:
        M, N, K = key
        return self._problem_key(M, N, K) in self.entries


# Global cache instance for convenience
_global_cache: AutotuneCache | None = None


def get_global_cache() -> AutotuneCache:
    """Get the global autotuning cache instance.

    Creates and loads the cache on first access.
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = AutotuneCache()
    return _global_cache


def get_tuned_config(M: int, N: int, K: int, group_size: int = 32) -> TileConfig:
    """Convenience function to get a tuned config using global cache.

    If not cached, runs autotuning and caches the result.
    """
    cache = get_global_cache()
    return cache.get_or_tune(M, N, K, group_size)
