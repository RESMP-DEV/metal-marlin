#!/usr/bin/env env
"""
GLM-4 Metal Shader Test-and-Fix Pipeline

Runs the GLM-4 forward pass, captures Metal shader errors, and generates
fix tasks for the AlphaHENG swarm.

Usage:
    # Test current state and generate fix tasks if needed
    uv run python scripts/glm4_test_fix.py test
    
    # Run generated fix tasks
    uv run alphaheng tasks add agent_workspace/glm4_shader_fixes.yaml
    uv run alphaheng coordinator --local-workers 10
    
    # Continuous test-and-fix loop
    uv run python scripts/glm4_test_fix.py loop --max-iters 5
"""

import argparse
import hashlib
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class MetalError:
    """A Metal shader compilation error."""
    file: str
    line: int
    column: int
    severity: str
    message: str
    source_line: str | None = None

    @property
    def task_id(self) -> str:
        content = f"{self.file}:{self.line}:{self.message[:50]}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}: {self.severity}: {self.message}"


def run_forward_test(timeout: int = 120) -> tuple[bool, str, list[MetalError]]:
    """Run GLM-4 forward pass test and capture errors.

    Returns:
        (success, output, errors)
    """
    test_script = '''
import torch
import sys
sys.stdout.reconfigure(line_buffering=True)
import warnings
warnings.filterwarnings("ignore")

from metal_marlin.trellis.lm import TrellisForCausalLM

print("Loading model...")
model = TrellisForCausalLM.from_pretrained("models/GLM-4.7-Flash-Trellis-3bpw", device="mps")
print("Model loaded OK")

x = torch.randint(0, 1000, (1, 16)).to("mps")
print("Running forward pass...")
with torch.no_grad():
    out = model(x)
print(f"Output shape: {out.logits.shape}")
print("SUCCESS: Forward pass completed")
'''

    # Write temp script
    script_path = Path(tempfile.gettempdir()) / "glm4_test.py"
    script_path.write_text(test_script)

    # Run from metal_marlin directory
    metal_marlin_dir = Path(__file__).parent.parent
    result = subprocess.run(
        ["uv", "run", "python", str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(metal_marlin_dir),
        timeout=timeout,
        env={
            **dict(__import__("os").environ),
            "PYTHONUNBUFFERED": "1",
        }
    )

    output = result.stdout + result.stderr
    errors = parse_metal_errors(output)
    success = "SUCCESS: Forward pass completed" in output

    return success, output, errors


def parse_metal_errors(output: str) -> list[MetalError]:
    """Parse Metal shader compilation errors from output."""
    errors = []

    # Pattern: program_source:LINE:COL: error: MESSAGE
    pattern = r'program_source:(\d+):(\d+):\s*(error|warning):\s*(.+?)(?=\n|$)'

    for match in re.finditer(pattern, output, re.MULTILINE):
        line = int(match.group(1))
        col = int(match.group(2))
        severity = match.group(3)
        message = match.group(4).strip()

        # Try to extract the source line (usually follows error)
        source_line = None
        pos = match.end()
        remaining = output[pos:pos+200]
        source_match = re.search(r'^\s*(.+?)$', remaining, re.MULTILINE)
        if source_match and not source_match.group(1).startswith('program_source'):
            source_line = source_match.group(1).strip()

        errors.append(MetalError(
            file="program_source",  # Runtime-compiled shader
            line=line,
            column=col,
            severity=severity,
            message=message,
            source_line=source_line
        ))

    return errors


def identify_shader_file(errors: list[MetalError]) -> dict[str, list[MetalError]]:
    """Try to map errors to actual shader files based on error content."""
    shader_mapping = {
        "moe_fused_dispatch": "contrib/metal_marlin/src/moe_fused_dispatch_shared.metal",
        "moe_fused_shared_expert": "contrib/metal_marlin/src/moe_fused_shared_expert.metal",
        "moe_fused_router": "contrib/metal_marlin/src/moe_fused_router.metal",
        "fp4_dequant": "contrib/metal_marlin/src/trellis_fp4_gemm.metal",
        "trellis": "contrib/metal_marlin/src/trellis_fp4_gemm.metal",
    }

    file_errors: dict[str, list[MetalError]] = defaultdict(list)

    for error in errors:
        # Try to identify by error message content
        assigned = False
        for keyword, filepath in shader_mapping.items():
            if keyword in error.message.lower() or (error.source_line and keyword in error.source_line.lower()):
                error.file = filepath
                file_errors[filepath].append(error)
                assigned = True
                break

        if not assigned:
            # Check source line for hints
            if error.source_line:
                if "threadgroup" in error.source_line and ("B_gate" in error.source_line or "B_up" in error.source_line):
                    error.file = shader_mapping["moe_fused_dispatch"]
                    file_errors[error.file].append(error)
                else:
                    file_errors["unknown"].append(error)
            else:
                file_errors["unknown"].append(error)

    return dict(file_errors)


def generate_fix_prompt(filepath: str, errors: list[MetalError]) -> str:
    """Generate a fix task prompt for shader errors."""

    error_list = []
    for i, err in enumerate(errors[:10], 1):
        error_list.append(f"Error {i} (line {err.line}): {err.message}")
        if err.source_line:
            error_list.append(f"  Source: {err.source_line}")

    if len(errors) > 10:
        error_list.append(f"... and {len(errors) - 10} more errors")

    prompt = f"""Fix Metal shader compilation errors in `{filepath}`.

## Errors Found

{chr(10).join(error_list)}

## Common Metal Shader Issues

1. **threadgroup address space in non-kernel function**
   - Move threadgroup declarations to kernel function
   - Pass as threadgroup references/pointers to helper functions

2. **implicitly deleted copy constructor**
   - Don't copy simdgroup_matrix types, use references

3. **zero-length arrays**
   - Check compile-time constants like TILE_N / THREADS
   - Add static_assert or guards

4. **missing device/threadgroup qualifier**
   - Add appropriate address space qualifier to pointers

## Verification

After fixing, the shader should compile at runtime. Test with:
```bash
cd contrib/metal_marlin && uv run python -c "
import torch
torch.set_default_device('mps')
from metal_marlin.trellis.model import TrellisForCausalLM
model = TrellisForCausalLM.from_quantized('models/GLM-4.7-Flash-Trellis-3bpw')
x = torch.randint(0, 1000, (1, 16))
with torch.no_grad():
    out = model(x)
print('SUCCESS')
"
```
"""
    return prompt


def generate_tasks(errors: list[MetalError], output_path: Path) -> int:
    """Generate task YAML from errors."""

    file_errors = identify_shader_file(errors)

    tasks = []
    for filepath, errs in file_errors.items():
        if filepath == "unknown":
            continue

        task_id = f"fix-shader-{hashlib.sha256(filepath.encode()).hexdigest()[:8]}"
        task = {
            "name": task_id,
            "prompt": generate_fix_prompt(filepath, errs),
            "priority": "P0",
            "dependencies": [],
        }
        tasks.append(task)

    # Add verification task
    tasks.append({
        "name": "verify-glm4-metal-forward",
        "prompt": """Verify GLM-4.7-Flash forward pass works after shader fixes.

Run from repo root:
```bash
cd contrib/metal_marlin && uv run python -c "
import torch
torch.set_default_device('mps')
from metal_marlin.trellis.model import TrellisForCausalLM
model = TrellisForCausalLM.from_quantized('models/GLM-4.7-Flash-Trellis-3bpw')
x = torch.randint(0, 1000, (1, 16))
with torch.no_grad():
    out = model(x)
print(f'Output shape: {out.logits.shape}')
print('SUCCESS')
" 2>&1
```

Expected: "SUCCESS" without Metal shader errors.
""",
        "priority": "P1",
        "dependencies": [t["name"] for t in tasks if t["name"].startswith("fix-shader-")],
    })

    task_data = {
        "metadata": {
            "name": "GLM-4 Metal Shader Fixes",
            "description": f"Auto-generated tasks to fix {len(errors)} Metal shader errors",
            "generated_by": "scripts/glm4_test_fix.py",
        },
        "tasks": tasks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(task_data, f, default_flow_style=False, sort_keys=False)

    return len(tasks)


def run_test_command(args) -> None:
    """Run forward test and optionally generate tasks."""
    print("=" * 60)
    print("🧪 GLM-4.7-Flash Forward Pass Test")
    print("=" * 60)

    try:
        success, output, errors = run_forward_test(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print(f"❌ Test timed out after {args.timeout}s")
        print("   This usually indicates extremely slow CPU fallback")
        return
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return

    if success:
        print("✅ SUCCESS: Forward pass completed!")
        print("\nOutput:")
        print(output[-500:] if len(output) > 500 else output)
        return

    print(f"❌ Forward pass failed with {len(errors)} Metal errors")
    print("\nErrors found:")
    for err in errors[:5]:
        print(f"  Line {err.line}: {err.message[:80]}")
        if err.source_line:
            print(f"    Source: {err.source_line[:60]}")

    if len(errors) > 5:
        print(f"  ... and {len(errors) - 5} more")

    if args.generate_tasks:
        output_path = Path(args.output)
        num_tasks = generate_tasks(errors, output_path)
        print(f"\n📝 Generated {num_tasks} fix tasks: {output_path}")
        print("\nTo run:")
        print(f"  uv run alphaheng tasks add {output_path}")
        print("  uv run alphaheng coordinator --local-workers 10")


def run_loop_command(args) -> None:
    """Run continuous test-and-fix loop."""
    print("=" * 60)
    print("🔄 GLM-4 Test-and-Fix Loop")
    print("=" * 60)
    print(f"Max iterations: {args.max_iters}")
    print(f"Wait between iterations: {args.wait}s")
    print()

    for iteration in range(1, args.max_iters + 1):
        print(f"\n{'=' * 40}")
        print(f"Iteration {iteration}/{args.max_iters}")
        print("=" * 40)

        try:
            success, output, errors = run_forward_test(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print("⏱️  Test timed out - likely CPU fallback, continuing...")
            errors = []
            success = False
        except Exception as e:
            print(f"❌ Test error: {e}")
            break

        if success:
            print("✅ SUCCESS! Forward pass works!")
            break

        if not errors:
            print("⚠️  No Metal errors but test failed - may be different issue")
            break

        print(f"Found {len(errors)} Metal errors")

        # Generate and submit tasks
        output_path = Path(f"agent_workspace/glm4_shader_iter{iteration}.yaml")
        num_tasks = generate_tasks(errors, output_path)
        print(f"📝 Generated {num_tasks} tasks: {output_path}")

        # Add to queue
        result = subprocess.run(
            ["uv", "run", "alphaheng", "tasks", "add", str(output_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("📋 Added tasks to queue")
        else:
            print(f"⚠️  Failed to add tasks: {result.stderr}")

        print(f"⏳ Waiting {args.wait}s for swarm to process...")
        time.sleep(args.wait)

    print("\n" + "=" * 60)
    print("Loop completed")


def main():
    parser = argparse.ArgumentParser(
        description="GLM-4 Metal Shader Test-and-Fix Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Test command
    test_parser = subparsers.add_parser("test", help="Run forward test")
    test_parser.add_argument("--timeout", type=int,
                             default=120, help="Test timeout in seconds")
    test_parser.add_argument("--generate-tasks", "-g", action="store_true", default=True,
                             help="Generate fix tasks if errors found")
    test_parser.add_argument("--output", "-o", default="agent_workspace/glm4_shader_fixes.yaml",
                             help="Output YAML file for tasks")

    # Loop command
    loop_parser = subparsers.add_parser(
        "loop", help="Continuous test-and-fix loop")
    loop_parser.add_argument(
        "--max-iters", "-n", type=int, default=5, help="Maximum iterations")
    loop_parser.add_argument(
        "--wait", "-w", type=int, default=60, help="Seconds to wait between iterations")
    loop_parser.add_argument("--timeout", type=int,
                             default=120, help="Test timeout per iteration")

    args = parser.parse_args()

    if args.command == "test":
        run_test_command(args)
    elif args.command == "loop":
        run_loop_command(args)


if __name__ == "__main__":
    main()
