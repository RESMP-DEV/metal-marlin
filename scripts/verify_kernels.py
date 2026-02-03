
from pathlib import Path

from metal_marlin.metal_dispatch import MetalKernelLibrary
from metal_marlin.metallib_loader import (
    DEFAULT_METALLIB,
    get_metallib_version,
    get_staleness_details,
)

DEFAULT_METALLIB_VERSION_FILE = (
    Path(__file__).parent.parent / "metal_marlin" / "lib" / ".metallib_version"
)


def parse_metallib_version_file(version_file: Path | None = None) -> dict[str, str]:
    """Parse the .metallib_version file to get expected version info.

    Args:
        version_file: Path to .metallib_version file. If None, uses default.

    Returns:
        Dict with version info (build_date, git_hash, shader_count, metal_version).
    """
    if version_file is None:
        version_file = DEFAULT_METALLIB_VERSION_FILE

    result: dict[str, str] = {}
    if not version_file.exists():
        return result

    try:
        with open(version_file) as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    key, value = line.split("=", 1)
                    result[key] = value
    except Exception:
        pass

    return result


def compare_metallib_versions(
    expected: dict[str, str],
    actual: dict[str, object],
) -> list[str]:
    """Compare expected version data against actual metallib metadata.

    Returns a list of error strings for any mismatches or missing metadata.
    """
    errors: list[str] = []

    expected_hash = expected.get("git_hash")
    actual_hash = actual.get("git_hash")
    if expected_hash:
        if not actual_hash:
            errors.append("git hash missing from library metadata")
        elif expected_hash != actual_hash:
            errors.append(
                f"git hash mismatch: expected {expected_hash}, got {actual_hash}"
            )

    expected_shaders = expected.get("shader_count")
    actual_shaders = actual.get("shader_count")
    if expected_shaders:
        if actual_shaders is None:
            errors.append("shader count missing from library metadata")
        else:
            try:
                expected_count = int(expected_shaders)
            except ValueError:
                errors.append(
                    f"shader count in version file is not an integer: {expected_shaders}"
                )
            else:
                if expected_count != actual_shaders:
                    errors.append(
                        f"shader count mismatch: expected {expected_shaders}, got {actual_shaders}"
                    )

    expected_metal = expected.get("metal_version")
    actual_metal = actual.get("metal_version")
    if expected_metal:
        if not actual_metal:
            errors.append("metal version missing from library metadata")
        elif expected_metal != actual_metal:
            errors.append(
                f"metal version mismatch: expected {expected_metal}, got {actual_metal}"
            )

    return errors


def validate_precompiled_library() -> tuple[bool, str]:
    """Validate that the precompiled library exists and matches expected version.

    This is the main validation entry point that checks:
    1. The precompiled library file exists
    2. The library version matches the expected version from .metallib_version

    Returns:
        Tuple of (is_valid, message).
    """
    # Step 1: Check if precompiled library exists
    if not DEFAULT_METALLIB.exists():
        return (
            False,
            f"Precompiled library not found: {DEFAULT_METALLIB}\n"
            f"  Run: ./scripts/build_metallib.sh"
        )

    # Step 2: Get expected version info
    if not DEFAULT_METALLIB_VERSION_FILE.exists():
        return (
            False,
            f"Expected version file not found: {DEFAULT_METALLIB_VERSION_FILE}\n"
            f"  Run: ./scripts/build_metallib.sh"
        )

    expected = parse_metallib_version_file(DEFAULT_METALLIB_VERSION_FILE)
    if not expected:
        return (
            False,
            f"Expected version file is empty or unreadable: {DEFAULT_METALLIB_VERSION_FILE}"
        )

    # Step 3: Get actual version from library
    actual = get_metallib_version(DEFAULT_METALLIB)
    if "error" in actual:
        return False, f"Failed to get library version: {actual['error']}"

    # Step 4: Compare versions
    errors = compare_metallib_versions(expected, actual)

    if errors:
        return False, "; ".join(errors)

    # Build success message
    parts = [f"Library: {DEFAULT_METALLIB.name}"]
    expected_hash = expected.get("git_hash")
    expected_shaders = expected.get("shader_count")
    expected_metal = expected.get("metal_version")
    if expected_hash:
        parts.append(f"git: {expected_hash}")
    if expected_shaders:
        parts.append(f"shaders: {expected_shaders}")
    if expected_metal:
        parts.append(f"metal: {expected_metal}")

    return True, f"Precompiled library validated ({', '.join(parts)})"


def check_library_exists() -> tuple[bool, str]:
    """Check that the precompiled library exists and matches expected version."""
    return validate_precompiled_library()


def validate_metallib_version() -> tuple[bool, str]:
    """Validate that the precompiled metallib matches the expected version.

    Returns:
        Tuple of (is_valid, message).
    """
    expected = parse_metallib_version_file()
    if not expected:
        return True, "No expected version file found, skipping version check"

    # First check: metallib file must exist
    if not DEFAULT_METALLIB.exists():
        return False, f"Precompiled library not found: {DEFAULT_METALLIB}"

    actual = get_metallib_version(DEFAULT_METALLIB)
    if "error" in actual:
        return False, f"Failed to get metallib version: {actual['error']}"

    errors = compare_metallib_versions(expected, actual)

    if errors:
        return False, "; ".join(errors)

    # Build summary message
    parts = []
    expected_hash = expected.get("git_hash")
    expected_shaders = expected.get("shader_count")
    expected_metal = expected.get("metal_version")
    if expected_hash:
        parts.append(f"git hash: {expected_hash}")
    if expected_shaders:
        parts.append(f"shaders: {expected_shaders}")
    if expected_metal:
        parts.append(f"metal: {expected_metal}")

    if parts:
        return True, f"Version matches ({', '.join(parts)})"
    return True, "Version check passed"


def validate_metallib() -> bool:
    """Validate that the precompiled metallib exists and is up-to-date.

    Returns:
        True if metallib is valid and current, False if missing or stale.

    Prints detailed status information about the metallib.
    """
    print("\n=== Precompiled Metallib Validation ===")

    if not DEFAULT_METALLIB.exists():
        print(f"✗ Precompiled metallib not found: {DEFAULT_METALLIB}")
        print("  Run: ./scripts/build_metallib.sh")
        return False

    # Get version information
    version_info = get_metallib_version(DEFAULT_METALLIB)
    if "error" in version_info:
        print(f"✗ Failed to get metallib version: {version_info['error']}")
        return False

    print(f"✓ Metallib found: {DEFAULT_METALLIB}")
    print(f"  Size: {version_info['size_bytes']:,} bytes")
    print(f"  Build date: {version_info['build_date']}")

    # Check staleness
    stale_details = get_staleness_details(DEFAULT_METALLIB)

    if stale_details["is_stale"]:
        print(f"\n⚠ Metallib is stale: {stale_details['reason']}")
        if stale_details.get("added_files"):
            print(f"  Added files ({len(stale_details['added_files'])}):")
            for f in stale_details["added_files"][:3]:
                print(f"    - {f}")
            if len(stale_details["added_files"]) > 3:
                print(f"    ... and {len(stale_details['added_files']) - 3} more")
        if stale_details.get("modified_files"):
            print(f"  Modified files ({len(stale_details['modified_files'])}):")
            for f in stale_details["modified_files"][:3]:
                print(f"    - {f}")
            if len(stale_details["modified_files"]) > 3:
                print(f"    ... and {len(stale_details['modified_files']) - 3} more")
        if stale_details.get("removed_files"):
            print(f"  Removed files ({len(stale_details['removed_files'])}):")
            for f in stale_details["removed_files"][:3]:
                print(f"    - {f}")
            if len(stale_details["removed_files"]) > 3:
                print(f"    ... and {len(stale_details['removed_files']) - 3} more")
        print("\n  Run: ./scripts/build_metallib.sh")
        return False

    if stale_details.get("has_manifest"):
        print("✓ Checksums match (verified with manifest)")
    else:
        print("✓ Metallib is current (mtime check)")

    return True


def main() -> int:
    """Run kernel verification.

    Returns:
        0 if all kernels verified successfully, 1 otherwise.
    """
    # Step 1: Validate precompiled library exists and matches expected version
    print("\n=== Precompiled Library Validation ===")
    lib_valid, lib_msg = validate_precompiled_library()
    if lib_valid:
        print(f"✓ {lib_msg}")
    else:
        print(f"✗ {lib_msg}")
        return 1  # Exit early if library validation fails

    # Step 2: Validate metallib freshness/staleness
    metallib_valid = validate_metallib()

    # Step 3: Validate metallib version against expected version (redundant but kept for compatibility)
    print("\n=== Version Validation ===")
    version_valid, version_msg = validate_metallib_version()
    if version_valid:
        print(f"✓ {version_msg}")
    else:
        print(f"✗ {version_msg}")

    print("\n=== Kernel Verification ===")
    print("Loading Metal kernel library...")
    lib = MetalKernelLibrary.from_source_dir()

    required_kernels = [
        "marlin_gemm_fp4",
        "moe_dispatch_optimized",
        "flash_attention_v2",
        "simdgroup_attention",
        "dense_gemm",
    ]

    all_passed = True
    for kernel in required_kernels:
        try:
            pipeline = lib.get_pipeline(kernel)
            print(f"✓ {kernel}")
        except Exception as e:
            print(f"✗ {kernel}: {e}")
            all_passed = False

    # Exit with error code if validation failed
    if not metallib_valid or not version_valid or not all_passed:
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
