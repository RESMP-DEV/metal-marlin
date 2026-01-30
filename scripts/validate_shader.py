#!/usr/bin/env python3
"""
Validate Metal shader standalone compilation.

This script tests whether Metal shaders can be compiled WITHOUT framework
integration. This isolates whether compilation issues are in our shader
code vs. the PyTorch/MPS integration layer.

Validation approaches:
1. Metal command-line compiler (xcrun -sdk macosx metal)
2. Metal library linking (xcrun metallib)
3. PyTorch MPS device availability check

Usage:
    python validate_shader.py                    # Validate marlin_gemm.metal
    python validate_shader.py --all              # Validate all .metal files
    python validate_shader.py path/to/file.metal # Validate specific file
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Project root
_ROOT = Path(__file__).parent.parent
_SRC_DIR = _ROOT / "src"
_DEFAULT_SHADER = _SRC_DIR / "marlin_gemm.metal"


def validate_with_metal_compiler(shader_path: Path) -> tuple[bool, str]:
    """
    Compile shader using Apple's Metal command-line compiler.

    This is the most direct test: bypasses any framework and uses the
    same compiler that Xcode uses.

    Args:
        shader_path: Path to the .metal file

    Returns:
        (success, output) tuple
    """
    if not shader_path.exists():
        return False, f"Shader file not found: {shader_path}"

    # Create a temp file for the AIR output
    with tempfile.NamedTemporaryFile(suffix=".air", delete=False) as f:
        air_path = Path(f.name)

    try:
        # Compile to AIR (Apple Intermediate Representation)
        # -std=metal3.0 for latest Metal features (simdgroup_matrix, etc.)
        result = subprocess.run(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-std=metal3.0",
                "-c",
                str(shader_path),
                "-o",
                str(air_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            # Also try linking to metallib to catch linker errors
            with tempfile.NamedTemporaryFile(suffix=".metallib", delete=False) as f:
                metallib_path = Path(f.name)

            try:
                link_result = subprocess.run(
                    [
                        "xcrun",
                        "-sdk",
                        "macosx",
                        "metallib",
                        str(air_path),
                        "-o",
                        str(metallib_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if link_result.returncode == 0:
                    return True, "Metal compiler + linker: OK"
                else:
                    return False, f"Metal linker failed:\n{link_result.stderr}"
            finally:
                metallib_path.unlink(missing_ok=True)
        else:
            return False, f"Metal compiler failed:\n{result.stderr}"

    except FileNotFoundError:
        return False, "xcrun not found. Are you on macOS with Xcode installed?"
    except subprocess.TimeoutExpired:
        return False, "Metal compiler timed out (>60s)"
    finally:
        air_path.unlink(missing_ok=True)


def check_duplicate_kernels(shader_path: Path) -> list[str]:
    """
    Scan for duplicate kernel definitions which would cause linker errors.

    Args:
        shader_path: Path to the .metal file

    Returns:
        List of duplicated kernel names.
    """
    if not shader_path.exists():
        return []

    src = shader_path.read_text()

    # Extract kernel names
    kernel_pattern = re.compile(r"^kernel\s+void\s+(\w+)\s*\(", re.MULTILINE)
    kernel_names = kernel_pattern.findall(src)

    # Find duplicates
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for name in kernel_names:
        seen[name] = seen.get(name, 0) + 1
        if seen[name] == 2:
            duplicates.append(name)

    return duplicates


def extract_kernel_names(shader_path: Path) -> list[str]:
    """
    Extract all kernel function names from a Metal shader.

    Args:
        shader_path: Path to the .metal file

    Returns:
        List of kernel names found in the source.
    """
    if not shader_path.exists():
        return []

    src = shader_path.read_text()
    kernel_pattern = re.compile(r"^kernel\s+void\s+(\w+)\s*\(", re.MULTILINE)
    return kernel_pattern.findall(src)


def check_mps_available() -> tuple[bool, str]:
    """
    Check if PyTorch MPS backend is available.

    Returns:
        (available, message) tuple
    """
    try:
        import torch

        if not hasattr(torch.backends, "mps"):
            return False, "PyTorch version does not support MPS"

        if not torch.backends.mps.is_available():
            return False, "MPS not available (no Apple Silicon or macOS < 12.3)"

        if not torch.backends.mps.is_built():
            return False, "PyTorch not built with MPS support"

        return True, f"MPS available (PyTorch {torch.__version__})"

    except ImportError:
        return False, "PyTorch not installed"


def validate_shader(shader_path: Path, verbose: bool = True) -> bool:
    """
    Run all validation checks on a shader file.

    Args:
        shader_path: Path to the .metal file
        verbose: Whether to print detailed output

    Returns:
        True if all validations pass, False otherwise.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Validating: {shader_path.name}")
        print("=" * 60)
        print(f"Path: {shader_path}")
        print(f"Exists: {shader_path.exists()}")

    if not shader_path.exists():
        if verbose:
            print("\nERROR: Shader file not found!")
        return False

    # Check file stats
    stat = shader_path.stat()
    if verbose:
        print(f"Size: {stat.st_size:,} bytes")

    # Check for duplicate kernel definitions
    if verbose:
        print("\n--- Duplicate Kernel Check ---")
    duplicates = check_duplicate_kernels(shader_path)
    if duplicates:
        if verbose:
            print(f"WARNING: DUPLICATES FOUND: {duplicates}")
            print("   This will cause linker errors!")
    else:
        if verbose:
            print("OK: No duplicate kernel definitions")

    # Metal command-line compiler
    if verbose:
        print("\n--- Metal Compiler (xcrun metal) ---")
    success, msg = validate_with_metal_compiler(shader_path)
    if verbose:
        status = "OK" if success else "FAIL"
        print(f"{status}: {msg}")

    # Extract and display kernel names
    if verbose:
        print("\n--- Kernel Functions Found ---")
        kernels = extract_kernel_names(shader_path)
        if kernels:
            for name in kernels:
                print(f"  - {name}")
            print(f"Total: {len(kernels)} kernels")
        else:
            print("  No kernel functions found")

    # Summary
    all_ok = success and not duplicates
    if verbose:
        print("\n--- Result ---")
        if all_ok:
            print("PASS: Shader compiles successfully standalone")
        else:
            print("FAIL: Shader has issues:")
            if duplicates:
                print(f"  - Duplicate kernels: {duplicates}")
            if not success:
                print("  - Compilation failed")

    return all_ok


def validate_all_shaders(src_dir: Path, verbose: bool = True) -> dict[str, bool]:
    """
    Validate all .metal files in a directory.

    Args:
        src_dir: Directory containing .metal files
        verbose: Whether to print detailed output

    Returns:
        dict mapping filename -> success
    """
    results: dict[str, bool] = {}

    metal_files = sorted(src_dir.glob("*.metal"))
    if not metal_files:
        if verbose:
            print(f"No .metal files found in {src_dir}")
        return results

    if verbose:
        print(f"Found {len(metal_files)} Metal shader files")

    for shader_path in metal_files:
        success = validate_shader(shader_path, verbose=verbose)
        results[shader_path.name] = success

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Metal shader compilation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "shader",
        nargs="?",
        type=Path,
        default=None,
        help="Path to .metal file to validate (default: marlin_gemm.metal)",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Validate all .metal files in src/",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only print summary, not per-shader details",
    )
    parser.add_argument(
        "--check-mps",
        action="store_true",
        help="Also check PyTorch MPS availability",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Check MPS if requested
    if args.check_mps:
        print("\n--- PyTorch MPS Backend ---")
        mps_ok, mps_msg = check_mps_available()
        status = "OK" if mps_ok else "INFO"
        print(f"{status}: {mps_msg}")

    # Validate shaders
    if args.all:
        results = validate_all_shaders(_SRC_DIR, verbose=verbose)

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        passed = sum(1 for v in results.values() if v)
        failed = len(results) - passed

        for name, success in sorted(results.items()):
            status = "PASS" if success else "FAIL"
            print(f"  {status}: {name}")

        print(f"\nTotal: {passed} passed, {failed} failed, {len(results)} total")
        return 0 if failed == 0 else 1

    else:
        shader_path = args.shader if args.shader else _DEFAULT_SHADER
        if not shader_path.is_absolute():
            # Try relative to src dir first
            if (_SRC_DIR / shader_path).exists():
                shader_path = _SRC_DIR / shader_path
            # Otherwise assume relative to cwd

        success = validate_shader(shader_path, verbose=verbose)
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
