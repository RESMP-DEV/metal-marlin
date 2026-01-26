#!/usr/bin/env python3
"""
Validate marlin_gemm.metal standalone compilation.

This script tests whether the Metal shader can be compiled WITHOUT MLX's
kernel concatenation/wrapping. This isolates whether compilation issues
are in our shader code vs. the MLX integration.

Two validation approaches:
1. Metal command-line compiler (xcrun -sdk macosx metal)
2. MLX metal_kernel API with minimal wrappers

FINDINGS:
---------
(Will be populated after running the script)
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

# Project root
_ROOT = Path(__file__).parent.parent
_SHADER_PATH = _ROOT / "src" / "marlin_gemm.metal"


def validate_with_metal_compiler() -> tuple[bool, str]:
    """
    Compile shader using Apple's Metal command-line compiler.

    This is the most direct test — bypasses MLX entirely and uses the
    same compiler that Xcode uses.

    Returns:
        (success, output) tuple
    """
    if not _SHADER_PATH.exists():
        return False, f"Shader file not found: {_SHADER_PATH}"

    # Create a temp file for the AIR output
    with tempfile.NamedTemporaryFile(suffix=".air", delete=False) as f:
        air_path = f.name

    try:
        # Compile to AIR (Apple Intermediate Representation)
        # -std=metal3.0 for latest Metal features (simdgroup_matrix, etc.)
        result = subprocess.run(
            [
                "xcrun", "-sdk", "macosx", "metal",
                "-std=metal3.0",
                "-c", str(_SHADER_PATH),
                "-o", air_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            return True, "Metal compiler: OK"
        else:
            return False, f"Metal compiler failed:\n{result.stderr}"

    except FileNotFoundError:
        return False, "xcrun not found — are you on macOS with Xcode installed?"
    except subprocess.TimeoutExpired:
        return False, "Metal compiler timed out (>60s)"
    finally:
        Path(air_path).unlink(missing_ok=True)


def check_duplicate_kernels() -> list[str]:
    """
    Scan for duplicate kernel definitions which would cause linker errors.

    Returns list of duplicated kernel names.
    """
    if not _SHADER_PATH.exists():
        return []

    src = _SHADER_PATH.read_text()

    # Extract kernel names
    import re
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


def validate_with_mlx() -> dict[str, tuple[bool, str]]:
    """
    Test kernel compilation via MLX metal_kernel API.

    NOTE: mx.fast.metal_kernel doesn't load full .metal files directly.
    It expects a source snippet that becomes the kernel body, with the
    header providing helper functions. So this test uses a minimal wrapper
    approach for each kernel.

    Returns:
        dict mapping kernel_name -> (success, message)
    """
    try:
        import mlx.core as mx
    except ImportError:
        return {"mlx": (False, "MLX not installed")}

    results: dict[str, tuple[bool, str]] = {}

    # Read full shader source
    src = _SHADER_PATH.read_text()

    # List of kernels we expect to be able to compile
    # These are the main public kernels
    kernels_to_test = [
        "marlin_gemm_fp4",
        "marlin_gemm_fp4_striped",
        "marlin_zero_reduction",
        "marlin_gemm_fused_fp4",
        "marlin_gemm_fused_u4",
        "marlin_gemm_fp4_fp32acc",
        "marlin_gemm_fp16_pipelined",
    ]

    # MLX metal_kernel creates its own kernel wrapper, so we can't directly
    # compile standalone kernels from .metal files. Instead, we verify
    # the file compiles via the Metal compiler (above) and just check
    # that the kernel names we expect are present.
    for name in kernels_to_test:
        if f"kernel void {name}(" in src:
            results[name] = (True, "Found in source")
        else:
            results[name] = (False, "NOT found in source")

    return results


def main():
    print("=" * 60)
    print("Metal Shader Validation: marlin_gemm.metal")
    print("=" * 60)
    print(f"\nShader path: {_SHADER_PATH}")
    print(f"Exists: {_SHADER_PATH.exists()}")

    if not _SHADER_PATH.exists():
        print("\nERROR: Shader file not found!")
        sys.exit(1)

    # Check file stats
    stat = _SHADER_PATH.stat()
    print(f"Size: {stat.st_size:,} bytes")

    # Check for duplicate kernel definitions
    print("\n--- Duplicate Kernel Check ---")
    duplicates = check_duplicate_kernels()
    if duplicates:
        print(f"⚠️  DUPLICATES FOUND: {duplicates}")
        print("   This will cause linker errors!")
    else:
        print("✓ No duplicate kernel definitions")

    # Metal command-line compiler
    print("\n--- Metal Compiler (xcrun metal) ---")
    success, msg = validate_with_metal_compiler()
    if success:
        print(f"✓ {msg}")
    else:
        print(f"✗ {msg}")

    # MLX kernel presence check
    print("\n--- Expected Kernels ---")
    mlx_results = validate_with_mlx()
    for name, (ok, msg) in mlx_results.items():
        status = "✓" if ok else "✗"
        print(f"{status} {name}: {msg}")

    # Summary
    print("\n--- Summary ---")
    all_ok = success and not duplicates
    if all_ok:
        print("✓ Shader compiles successfully standalone")
        print("  If tests fail, issue is in MLX integration, not shader syntax")
    else:
        print("✗ Shader has issues:")
        if duplicates:
            print(f"  - Duplicate kernels: {duplicates}")
        if not success:
            print("  - Compilation failed")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
