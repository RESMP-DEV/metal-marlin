"""Precompiled Metal library (.metallib) loader.

This module handles loading precompiled .metallib files for fast kernel dispatch.
Precompiled libraries are 100-1000x faster than runtime compilation.

Build the metallib with: ./scripts/build_metallib.sh
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default metallib location
DEFAULT_METALLIB = Path(__file__).parent / "lib" / "metal_marlin.metallib"

# Module-level cache
_cached_library: Any = None
_cached_path: Path | None = None


class MetallibNotFoundError(FileNotFoundError):
    """Raised when metallib file does not exist."""

    pass


class MetallibLoadError(RuntimeError):
    """Raised when metallib fails to load."""

    pass


# Check PyObjC Metal availability
try:
    import Foundation
    import Metal

    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    Metal = None
    Foundation = None


def require_metal() -> None:
    """Raise if Metal is not available."""
    if not HAS_METAL:
        raise RuntimeError(
            "PyObjC Metal framework not available. "
            "Install with: pip install pyobjc-framework-Metal"
        )


def load_metallib(path: str | Path | None = None) -> Any:
    """Load a precompiled Metal library (.metallib) file.

    Args:
        path: Path to .metallib file. If None, uses default location.

    Returns:
        MTLLibrary object with precompiled kernels.

    Raises:
        MetallibNotFoundError: If metallib file doesn't exist.
        MetallibLoadError: If metallib fails to load.

    Note:
        This function populates the module cache, so subsequent calls to
        get_precompiled_library() will return the same object.
    """
    global _cached_library, _cached_path

    require_metal()

    if path is None:
        path = DEFAULT_METALLIB
    path = Path(path)

    # Return cached if same path
    if _cached_library is not None and _cached_path == path:
        return _cached_library

    if not path.exists():
        raise MetallibNotFoundError(
            f"Precompiled metallib not found: {path}\n"
            f"Run: ./scripts/build_metallib.sh to generate it."
        )

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise MetallibLoadError("No Metal device available")

    url = Foundation.NSURL.fileURLWithPath_(str(path))
    library, error = device.newLibraryWithURL_error_(url, None)

    if library is None:
        error_msg = error.localizedDescription() if error else "Unknown error"
        raise MetallibLoadError(f"Failed to load metallib: {error_msg}")

    # Cache the loaded library
    _cached_library = library
    _cached_path = path

    return library


def get_metallib_version(path: str | Path | None = None) -> dict[str, Any]:
    """Get version information from metallib.

    Args:
        path: Path to .metallib file. If None, uses default.

    Returns:
        Dict with version info or error message.
    """
    if path is None:
        path = DEFAULT_METALLIB
    path = Path(path)

    if not path.exists():
        return {"error": f"Metallib not found: {path}"}

    try:
        stat = path.stat()
        result: dict[str, Any] = {
            "path": str(path),
            "build_date": stat.st_mtime,
            "size_bytes": stat.st_size,
        }

        # Try to extract embedded version info from metallib
        # Metallib files may contain embedded git hash and shader count
        version_info = _extract_metallib_version_info(path)
        result.update(version_info)

        return result
    except Exception as e:
        return {"error": str(e)}


def _extract_metallib_version_info(path: Path) -> dict[str, Any]:
    """Extract version info embedded in metallib file.

    Some metallib files have embedded metadata strings containing
    git hash and shader count information.

    Args:
        path: Path to .metallib file.

    Returns:
        Dict with extracted version info (may be empty).
    """
    result: dict[str, Any] = {}
    try:
        # Read first 64KB of file to look for embedded metadata
        with open(path, "rb") as f:
            data = f.read(65536)

        # Try to decode as text and look for version patterns
        try:
            text = data.decode("utf-8", errors="ignore")

            # Look for git hash pattern (7-40 hex chars)
            import re
            git_match = re.search(r"git[_-]?hash[=:]([a-f0-9]{7,40})", text, re.IGNORECASE)
            if git_match:
                result["git_hash"] = git_match.group(1)

            # Look for shader count pattern
            shader_match = re.search(r"shader[_-]?count[=:](\d+)", text, re.IGNORECASE)
            if shader_match:
                result["shader_count"] = int(shader_match.group(1))

            # Look for metal version pattern
            metal_match = re.search(r"metal[_-]?version[=:]([\d.]+)", text, re.IGNORECASE)
            if metal_match:
                result["metal_version"] = metal_match.group(1)

        except Exception:
            pass

    except Exception:
        pass

    return result


def get_precompiled_library(path: str | Path | None = None) -> Any | None:
    """Get cached precompiled library, loading if needed.

    Returns None on failure (allows fallback to JIT).
    """
    global _cached_library, _cached_path

    if path is None:
        path = DEFAULT_METALLIB
    path = Path(path)

    # Return cached if same path
    if _cached_library is not None and _cached_path == path:
        return _cached_library

    try:
        _cached_library = load_metallib(path)
        _cached_path = path
        return _cached_library
    except (MetallibNotFoundError, MetallibLoadError) as e:
        logger.warning(f"Metallib not available: {e}")
        return None


def clear_cache() -> None:
    """Clear cached metallib (for hot-reload during development)."""
    global _cached_library, _cached_path
    _cached_library = None
    _cached_path = None


def get_kernel_from_metallib(
    kernel_name: str,
    library: Any | None = None
) -> Any | None:
    """Get kernel function from metallib.

    Args:
        kernel_name: Name of kernel function.
        library: MTLLibrary, or None to use cached.

    Returns:
        MTLFunction or None if not found.
    """
    if library is None:
        library = get_precompiled_library()

    if library is None:
        return None

    kernel_fn = library.newFunctionWithName_(kernel_name)
    # PyObjC may return a falsy wrapper object instead of None
    # Check explicitly if the result is actually None
    if kernel_fn is None or not kernel_fn:
        return None
    return kernel_fn


def _get_metal_source_dirs(metallib_path: Path) -> list[Path]:
    """Get all directories that may contain .metal source files.

    Args:
        metallib_path: Path to the .metallib file.

    Returns:
        List of directories to scan for .metal files.
    """
    dirs = []

    # Main src/ directory (contrib/metal_marlin/src)
    src_dir = metallib_path.parent.parent.parent / "src"
    if src_dir.exists():
        dirs.append(src_dir)

    # metal_marlin subdirectories (distributed, vision, src)
    metal_marlin_dir = metallib_path.parent.parent
    for subdir in ["distributed", "vision", "src"]:
        subdir_path = metal_marlin_dir / subdir
        if subdir_path.exists():
            dirs.append(subdir_path)

    return dirs


def _compute_file_checksum(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file.

    Args:
        file_path: Path to the file.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _collect_metal_files(source_dirs: list[Path]) -> list[Path]:
    """Collect all .metal files from source directories.

    Args:
        source_dirs: List of directories to scan.

    Returns:
        Sorted list of .metal file paths.
    """
    metal_files = []
    for src_dir in source_dirs:
        metal_files.extend(src_dir.glob("**/*.metal"))
    return sorted(set(metal_files))


def compute_source_checksums(
    metallib_path: Path | None = None,
) -> dict[str, str]:
    """Compute checksums for all .metal source files.

    Args:
        metallib_path: Path to metallib (used to locate source dirs).

    Returns:
        Dict mapping relative file paths to their SHA-256 checksums.
    """
    if metallib_path is None:
        metallib_path = DEFAULT_METALLIB
    metallib_path = Path(metallib_path)

    source_dirs = _get_metal_source_dirs(metallib_path)
    if not source_dirs:
        return {}

    # Find a common root for relative paths
    project_root = metallib_path.parent.parent.parent
    metal_files = _collect_metal_files(source_dirs)

    checksums = {}
    for metal_file in metal_files:
        try:
            rel_path = metal_file.relative_to(project_root)
            checksums[str(rel_path)] = _compute_file_checksum(metal_file)
        except (ValueError, OSError) as e:
            logger.warning(f"Failed to checksum {metal_file}: {e}")
            continue

    return checksums


def get_checksum_manifest_path(metallib_path: Path) -> Path:
    """Get path to the checksum manifest file for a metallib.

    Args:
        metallib_path: Path to the .metallib file.

    Returns:
        Path to the corresponding .checksums.json file.
    """
    return metallib_path.with_suffix(".checksums.json")


def save_checksum_manifest(
    metallib_path: Path | None = None,
    checksums: dict[str, str] | None = None,
) -> Path:
    """Save checksum manifest alongside the metallib.

    Args:
        metallib_path: Path to metallib. If None, uses default.
        checksums: Precomputed checksums, or None to compute fresh.

    Returns:
        Path to the saved manifest file.
    """
    if metallib_path is None:
        metallib_path = DEFAULT_METALLIB
    metallib_path = Path(metallib_path)

    if checksums is None:
        checksums = compute_source_checksums(metallib_path)

    manifest_path = get_checksum_manifest_path(metallib_path)

    manifest = {
        "version": 1,
        "metallib": str(metallib_path.name),
        "file_count": len(checksums),
        "checksums": checksums,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    logger.debug(f"Saved checksum manifest: {manifest_path}")
    return manifest_path


def load_checksum_manifest(metallib_path: Path | None = None) -> dict[str, str] | None:
    """Load checksum manifest for a metallib.

    Args:
        metallib_path: Path to metallib. If None, uses default.

    Returns:
        Dict mapping file paths to checksums, or None if manifest missing/invalid.
    """
    if metallib_path is None:
        metallib_path = DEFAULT_METALLIB
    metallib_path = Path(metallib_path)

    manifest_path = get_checksum_manifest_path(metallib_path)
    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)

        if manifest.get("version") != 1:
            logger.warning(f"Unknown manifest version: {manifest.get('version')}")
            return None

        return manifest.get("checksums", {})
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load checksum manifest: {e}")
        return None


def is_metallib_stale(path: str | Path | None = None) -> bool:
    """Check if metallib is stale by comparing source file checksums.

    This is more reliable than mtime-based checks across different filesystems
    and git operations.

    Args:
        path: Path to .metallib file. If None, uses default.

    Returns:
        True if metallib should be rebuilt, False otherwise.
    """
    if path is None:
        path = DEFAULT_METALLIB
    path = Path(path)

    if not path.exists():
        return True

    # Try checksum-based comparison first
    stored_checksums = load_checksum_manifest(path)
    if stored_checksums is not None:
        current_checksums = compute_source_checksums(path)

        # Check for added, removed, or modified files
        stored_files = set(stored_checksums.keys())
        current_files = set(current_checksums.keys())

        if stored_files != current_files:
            added = current_files - stored_files
            removed = stored_files - current_files
            if added:
                logger.debug(f"New .metal files: {added}")
            if removed:
                logger.debug(f"Removed .metal files: {removed}")
            return True

        for file_path, current_hash in current_checksums.items():
            if stored_checksums.get(file_path) != current_hash:
                logger.debug(f"Modified .metal file: {file_path}")
                return True

        return False

    logger.debug("No checksum manifest found; treating metallib as stale")
    return True


def get_staleness_details(path: str | Path | None = None) -> dict[str, Any]:
    """Get detailed staleness information for debugging.

    Args:
        path: Path to .metallib file. If None, uses default.

    Returns:
        Dict with staleness details including added/removed/modified files.
    """
    if path is None:
        path = DEFAULT_METALLIB
    path = Path(path)

    result: dict[str, Any] = {
        "metallib_path": str(path),
        "metallib_exists": path.exists(),
        "is_stale": False,
        "reason": None,
        "added_files": [],
        "removed_files": [],
        "modified_files": [],
        "has_manifest": False,
    }

    if not path.exists():
        result["is_stale"] = True
        result["reason"] = "metallib does not exist"
        return result

    stored_checksums = load_checksum_manifest(path)
    if stored_checksums is None:
        result["reason"] = "no checksum manifest"
        result["is_stale"] = True
        return result

    result["has_manifest"] = True
    current_checksums = compute_source_checksums(path)

    stored_files = set(stored_checksums.keys())
    current_files = set(current_checksums.keys())

    added = sorted(current_files - stored_files)
    removed = sorted(stored_files - current_files)
    modified = sorted(
        f for f in (stored_files & current_files)
        if stored_checksums[f] != current_checksums.get(f)
    )

    result["added_files"] = added
    result["removed_files"] = removed
    result["modified_files"] = modified

    if added or removed or modified:
        result["is_stale"] = True
        reasons = []
        if added:
            reasons.append(f"{len(added)} added")
        if removed:
            reasons.append(f"{len(removed)} removed")
        if modified:
            reasons.append(f"{len(modified)} modified")
        result["reason"] = ", ".join(reasons)
    else:
        result["reason"] = "checksums match"

    return result
