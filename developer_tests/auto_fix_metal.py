#!/usr/bin/env python3
"""
Auto-fix Metal validation errors by parsing GPU validation output
and generating patches for kernel buffer access issues.

Usage:
    # Run with default model
    python auto_fix_metal.py

    # Run with existing validation log
    python auto_fix_metal.py --log validation_errors.txt

    # Dry run (don't apply patches)
    python auto_fix_metal.py --dry-run

    # Run specific test script
    python auto_fix_metal.py --script test_attention.py
"""
import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
METAL_DIRS = [
    SCRIPT_DIR / "src",
    SCRIPT_DIR / "src" / "fusion",
    SCRIPT_DIR / "metal_marlin" / "src",
    SCRIPT_DIR / "metal_marlin" / "distributed",
    SCRIPT_DIR / "metal_marlin" / "vision",
]

# Default test script for validation
DEFAULT_TEST_SCRIPT = """
import torch
from metal_marlin.trellis.lm import TrellisForCausalLM

model = TrellisForCausalLM.from_pretrained('models/GLM-4.7-Flash-Trellis-3bpw', device='mps')
x = torch.randint(0, 1000, (1, 64)).to('mps')
with torch.no_grad():
    out = model(x)
print("Inference complete")
"""


@dataclass
class ValidationError:
    """Parsed Metal validation error."""
    kernel_name: str
    buffer_index: int
    error_type: str
    message: str


@dataclass
class Fix:
    """Proposed fix for a validation error."""
    kernel_name: str
    buffer_index: int
    fix_type: str  # "add_const" or "remove_const"
    metal_file: Path | None = None
    line_number: int | None = None


@dataclass
class BufferInfo:
    """Information about a buffer parameter in a kernel."""
    name: str
    index: int
    has_const: bool
    full_match: str
    offset: int


def run_validation(script: str | None = None, timeout: int = 300) -> str:
    """Run model inference with Metal shader validation enabled."""
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env["MTL_SHADER_VALIDATION"] = "1"
    env["MTL_DEBUG_LAYER"] = "1"

    test_code = script or DEFAULT_TEST_SCRIPT

    print("Running inference with Metal validation enabled...")
    print(f"Timeout: {timeout}s")

    try:
        result = subprocess.run(
            ["uv", "run", "python", "-c", test_code],
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_DIR),
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"ERROR: Process timed out after {timeout}s")
        return ""

    if result.returncode == -9:
        print("ERROR: Process killed (likely OOM). Try a smaller model or batch size.")
        return result.stderr or ""

    if result.returncode != 0:
        print(f"Process exited with code {result.returncode}")
        if result.stdout:
            print(f"stdout:\n{result.stdout[:2000]}")

    return result.stderr or ""


def parse_validation_errors(log_text: str) -> list[ValidationError]:
    """Parse Metal validation errors from log output."""
    errors: list[ValidationError] = []

    # Pattern 1: Buffer access with write enabled on read-only
    # "Compute Function(kernel_name): Bytes at index N have read-only access but write access enabled"
    pattern1 = re.compile(
        r"Compute Function\((\w+)\):\s*Bytes at index (\d+)\s+have read-only access but write access enabled",
        re.IGNORECASE,
    )

    # Pattern 2: General buffer binding validation
    # "Compute Function(kernel_name): index N ..."
    pattern2 = re.compile(
        r"Compute Function\((\w+)\):.*?index\s+(\d+).*?write access",
        re.IGNORECASE | re.DOTALL,
    )

    # Pattern 3: Texture/buffer mismatch
    pattern3 = re.compile(
        r"Compute Function\((\w+)\):.*?buffer\s+at\s+index\s+(\d+)",
        re.IGNORECASE,
    )

    # Deduplicate by (kernel, buffer_index) since multiple patterns may match
    seen: set[tuple[str, int]] = set()

    for pattern, error_type in [
        (pattern1, "read_only_write_access"),
        (pattern2, "buffer_write_access"),
        (pattern3, "buffer_binding"),
    ]:
        for match in pattern.finditer(log_text):
            kernel_name = match.group(1)
            buffer_idx = int(match.group(2))
            key = (kernel_name, buffer_idx)

            if key not in seen:
                seen.add(key)
                errors.append(
                    ValidationError(
                        kernel_name=kernel_name,
                        buffer_index=buffer_idx,
                        error_type=error_type,
                        message=match.group(0)[:200],
                    )
                )

    return errors


def find_kernel_in_metal(kernel_name: str) -> tuple[Path | None, int | None]:
    """Find the Metal file and line number where a kernel is defined."""
    kernel_pattern = re.compile(
        rf"kernel\s+void\s+{re.escape(kernel_name)}\s*\(",
        re.MULTILINE,
    )

    for metal_dir in METAL_DIRS:
        if not metal_dir.exists():
            continue
        for metal_file in metal_dir.glob("*.metal"):
            content = metal_file.read_text()
            match = kernel_pattern.search(content)
            if match:
                line_num = content[:match.start()].count("\n") + 1
                return metal_file, line_num

    return None, None


def get_buffer_info(metal_file: Path, kernel_name: str) -> list[BufferInfo]:
    """Get information about all buffer parameters in a kernel."""
    content = metal_file.read_text()

    kernel_pattern = re.compile(
        rf"kernel\s+void\s+{re.escape(kernel_name)}\s*\(",
        re.MULTILINE,
    )

    match = kernel_pattern.search(content)
    if not match:
        return []

    # Find closing paren
    pos = match.end()
    paren_depth = 1
    while pos < len(content) and paren_depth > 0:
        if content[pos] == '(':
            paren_depth += 1
        elif content[pos] == ')':
            paren_depth -= 1
        pos += 1

    params_text = content[match.end():pos - 1]

    # Match buffer parameters
    param_pattern = re.compile(
        r"(device\s+)(const\s+)?([^,\[\]]+\*\s*)(\w+)\s*\[\[buffer\((\d+)\)\]\]",
    )

    buffers: list[BufferInfo] = []
    for param_match in param_pattern.finditer(params_text):
        buffers.append(BufferInfo(
            name=param_match.group(4),
            index=int(param_match.group(5)),
            has_const=param_match.group(2) is not None,
            full_match=param_match.group(0),
            offset=match.end() + param_match.start(),
        ))

    return buffers


def generate_fixes(errors: list[ValidationError]) -> list[Fix]:
    """Generate fixes for validation errors.

    Error types and their meanings:
    - "read_only_write_access": Buffer is bound as read-only at the encoder level,
      but the shader has write access enabled. This usually means the buffer
      should be marked as const in the shader (read-only access) OR the
      encoder binding is wrong (but we can only fix the shader side here).

    Strategy:
    - If buffer doesn't have const and error is about read-only/write mismatch:
      Add const to make the shader match the read-only binding.
    - If buffer has const and error is about write access:
      This is unusual - the shader declares read-only but something is trying
      to write. This might indicate the buffer actually needs write access.
    """
    fixes: list[Fix] = []

    for error in errors:
        metal_file, line_num = find_kernel_in_metal(error.kernel_name)

        # Default: add const to match read-only binding expectation
        fix_type = "add_const"

        if metal_file:
            buffers = get_buffer_info(metal_file, error.kernel_name)
            for buf in buffers:
                if buf.index == error.buffer_index:
                    if buf.has_const:
                        # Buffer already has const - the error suggests the
                        # shader is trying to write despite const declaration.
                        # This is a more complex issue, but we'll suggest
                        # removing const since the kernel apparently needs
                        # write access.
                        fix_type = "remove_const"
                    else:
                        # Buffer doesn't have const - add it to match
                        # read-only binding expectation
                        fix_type = "add_const"
                    break

        fix = Fix(
            kernel_name=error.kernel_name,
            buffer_index=error.buffer_index,
            fix_type=fix_type,
            metal_file=metal_file,
            line_number=line_num,
        )
        fixes.append(fix)

    return fixes


def apply_const_fix(metal_file: Path, kernel_name: str, buffer_index: int) -> bool:
    """
    Apply const qualifier to a buffer parameter in a Metal kernel.

    For buffers that are read-only but missing const, we add 'const' after 'device'.
    Pattern: "device half*" -> "device const half*"

    Returns True if fix was applied.
    """
    content = metal_file.read_text()

    # Find the kernel function signature (may span multiple lines)
    kernel_pattern = re.compile(
        rf"kernel\s+void\s+{re.escape(kernel_name)}\s*\(",
        re.MULTILINE,
    )

    match = kernel_pattern.search(content)
    if not match:
        print(f"  Could not find kernel {kernel_name} in {metal_file.name}")
        return False

    # Find the closing paren of the parameter list
    paren_depth = 1
    pos = match.end()
    while pos < len(content) and paren_depth > 0:
        if content[pos] == '(':
            paren_depth += 1
        elif content[pos] == ')':
            paren_depth -= 1
        pos += 1

    params_text = content[match.end():pos - 1]

    # Pattern to match buffer parameters:
    # "device const half* name [[buffer(N)]]" or "device half* name [[buffer(N)]]"
    # Captures: (device ), (const )?, (type* ), (name), (buffer index)
    param_pattern = re.compile(
        r"(device\s+)(const\s+)?([^,\[\]]+\*\s*)(\w+)\s*\[\[buffer\((\d+)\)\]\]",
    )

    target_match = None
    for param_match in param_pattern.finditer(params_text):
        if int(param_match.group(5)) == buffer_index:
            target_match = param_match
            break

    if not target_match:
        print(f"  Could not find buffer({buffer_index}) in {kernel_name}")
        return False

    # Check if already has const
    if target_match.group(2):
        print(f"  Buffer {buffer_index} already has 'const' qualifier")
        return False

    # Build the fix: insert "const " after "device "
    # The matched text is: "device <type>* name [[buffer(N)]]"
    # We want:             "device const <type>* name [[buffer(N)]]"
    old_param = target_match.group(0)
    new_param = (
        target_match.group(1) +  # "device "
        "const " +
        target_match.group(3) +  # type*
        target_match.group(4) +  # name
        f" [[buffer({buffer_index})]]"
    )

    # Replace in the full content
    param_abs_start = match.end() + target_match.start()
    param_abs_end = match.end() + target_match.end()

    new_content = content[:param_abs_start] + new_param + content[param_abs_end:]

    metal_file.write_text(new_content)
    print(f"  Fixed: {old_param.strip()}")
    print(f"      -> {new_param.strip()}")
    return True


def remove_const_fix(metal_file: Path, kernel_name: str, buffer_index: int) -> bool:
    """
    Remove const qualifier from a buffer parameter in a Metal kernel.

    For buffers that need write access but have const.
    Pattern: "device const half*" -> "device half*"

    Returns True if fix was applied.
    """
    content = metal_file.read_text()

    kernel_pattern = re.compile(
        rf"kernel\s+void\s+{re.escape(kernel_name)}\s*\(",
        re.MULTILINE,
    )

    match = kernel_pattern.search(content)
    if not match:
        print(f"  Could not find kernel {kernel_name} in {metal_file.name}")
        return False

    # Find closing paren
    paren_depth = 1
    pos = match.end()
    while pos < len(content) and paren_depth > 0:
        if content[pos] == '(':
            paren_depth += 1
        elif content[pos] == ')':
            paren_depth -= 1
        pos += 1

    params_text = content[match.end():pos - 1]

    param_pattern = re.compile(
        r"(device\s+)(const\s+)([^,\[\]]+\*\s*)(\w+)\s*\[\[buffer\((\d+)\)\]\]",
    )

    target_match = None
    for param_match in param_pattern.finditer(params_text):
        if int(param_match.group(5)) == buffer_index:
            target_match = param_match
            break

    if not target_match:
        print(f"  Could not find const buffer({buffer_index}) in {kernel_name}")
        return False

    # Build the fix: remove "const " after "device "
    old_param = target_match.group(0)
    new_param = (
        target_match.group(1) +  # "device "
        # skip group(2) which is "const "
        target_match.group(3) +  # type*
        target_match.group(4) +  # name
        f" [[buffer({buffer_index})]]"
    )

    param_abs_start = match.end() + target_match.start()
    param_abs_end = match.end() + target_match.end()

    new_content = content[:param_abs_start] + new_param + content[param_abs_end:]

    metal_file.write_text(new_content)
    print(f"  Fixed: {old_param.strip()}")
    print(f"      -> {new_param.strip()}")
    return True


def apply_fix(fix: Fix) -> bool:
    """Apply a fix based on its type."""
    if not fix.metal_file:
        print(f"  Skipping {fix.kernel_name}[{fix.buffer_index}]: Metal file not found")
        return False

    if fix.fix_type == "add_const":
        return apply_const_fix(fix.metal_file, fix.kernel_name, fix.buffer_index)
    elif fix.fix_type == "remove_const":
        return remove_const_fix(fix.metal_file, fix.kernel_name, fix.buffer_index)
    else:
        print(f"  Unknown fix type: {fix.fix_type}")
        return False


def print_summary(errors: list[ValidationError], fixes: list[Fix]) -> None:
    """Print summary of errors and fixes."""
    print(f"\n{'='*60}")
    print(f"Found {len(errors)} validation errors")
    print(f"{'='*60}\n")

    # Group by kernel
    by_kernel: dict[str, list[ValidationError]] = defaultdict(list)
    for e in errors:
        by_kernel[e.kernel_name].append(e)

    for kernel_name, kernel_errors in sorted(by_kernel.items()):
        print(f"Kernel: {kernel_name}")
        for e in kernel_errors:
            print(f"  buffer[{e.buffer_index}]: {e.error_type}")
        print()

    if fixes:
        print(f"{'='*60}")
        print(f"Proposed {len(fixes)} fixes")
        print(f"{'='*60}\n")

        for fix in fixes:
            loc = f"{fix.metal_file.name}:{fix.line_number}" if fix.metal_file else "unknown"
            print(f"  {fix.kernel_name}[{fix.buffer_index}] -> {fix.fix_type} ({loc})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-fix Metal validation errors")
    parser.add_argument("--log", type=Path, help="Use existing validation log file")
    parser.add_argument("--script", type=Path, help="Python script to run for validation")
    parser.add_argument("--dry-run", action="store_true", help="Don't apply fixes")
    parser.add_argument("--timeout", type=int, default=300, help="Validation timeout (seconds)")
    args = parser.parse_args()

    # Get validation errors
    if args.log:
        print(f"Reading validation log from {args.log}")
        log_text = args.log.read_text()
    else:
        script = args.script.read_text() if args.script else None
        log_text = run_validation(script=script, timeout=args.timeout)

    if not log_text:
        print("No validation output captured.")
        return 1

    # Parse errors
    errors = parse_validation_errors(log_text)

    if not errors:
        print("No validation errors found!")
        print("\nRaw log preview:")
        print(log_text[:2000] if len(log_text) > 2000 else log_text)
        return 0

    # Generate fixes
    fixes = generate_fixes(errors)

    # Print summary
    print_summary(errors, fixes)

    # Apply fixes
    if fixes and not args.dry_run:
        print(f"\n{'='*60}")
        print("Applying fixes...")
        print(f"{'='*60}\n")

        applied = 0
        for fix in fixes:
            print(f"Fixing {fix.kernel_name}[{fix.buffer_index}] ({fix.fix_type})...")
            if apply_fix(fix):
                applied += 1

        print(f"\nApplied {applied}/{len(fixes)} fixes")
    elif args.dry_run:
        print("\n[Dry run - no fixes applied]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
