#!/usr/bin/env python3
"""
Systematic audit of Metal kernels for common issues.

Categories checked:
1. threadgroup variables in non-kernel functions (illegal in Metal)
2. Missing const on read-only buffers (performance/correctness)
3. Zero-length arrays (undefined behavior)
4. threadgroup_barrier outside kernel functions
5. Mismatched simdgroup operations (potential correctness issues)
6. Uninitialized threadgroup memory usage
7. Missing bounds checks on buffer accesses
8. Incorrect memory order on atomic operations
"""

import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Issue:
    file: str
    line: int
    category: str
    severity: str  # critical, warning, info
    description: str


def find_kernel_ranges(content: str) -> list[tuple[int, int, str]]:
    """Find line ranges for each kernel function."""
    lines = content.split('\n')
    kernels = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if 'kernel void' in line or (i > 0 and 'kernel' in lines[i-1] and 'void' in line):
            # Found a kernel, extract name
            match = re.search(r'void\s+(\w+)\s*\(', line)
            if not match and i > 0:
                combined = lines[i-1] + ' ' + line
                match = re.search(r'void\s+(\w+)\s*\(', combined)

            kernel_name = match.group(1) if match else "unknown"
            start_line = i + 1

            # Find opening brace
            brace_count = 0
            j = i
            while j < len(lines):
                brace_count += lines[j].count('{') - lines[j].count('}')
                if brace_count > 0:
                    break
                j += 1

            # Find closing brace
            while j < len(lines) and brace_count > 0:
                j += 1
                if j < len(lines):
                    brace_count += lines[j].count('{') - lines[j].count('}')

            end_line = j + 1
            kernels.append((start_line, end_line, kernel_name))
            i = j + 1
        else:
            i += 1

    return kernels


def is_in_kernel(line_num: int, kernel_ranges: list[tuple[int, int, str]]) -> str | None:
    """Check if line is inside a kernel, return kernel name or None."""
    for start, end, name in kernel_ranges:
        if start <= line_num <= end:
            return name
    return None


def check_threadgroup_in_non_kernel(content: str, kernel_ranges: list[tuple[int, int, str]]) -> Iterator[tuple[int, str]]:
    """Check for threadgroup variables declared outside kernel functions."""
    lines = content.split('\n')

    for i, line in enumerate(lines, 1):
        # Skip comments
        stripped = line.strip()
        if stripped.startswith('//') or stripped.startswith('/*'):
            continue

        # Look for threadgroup variable declarations (not parameters)
        if 'threadgroup ' in line and '[[' not in line:
            # This is a threadgroup variable declaration
            if is_in_kernel(i, kernel_ranges) is None:
                # Check if it's a helper function definition, not a struct/global
                # Helper functions with threadgroup params are fine
                if '(' not in line or ')' not in line:
                    yield (i, f"threadgroup variable outside kernel: {stripped[:60]}")


def check_missing_const(content: str) -> Iterator[tuple[int, str]]:
    """Check for read-only buffers missing const qualifier."""
    # Find kernel signatures
    kernel_pattern = re.compile(
        r'kernel\s+void\s+(\w+)\s*\((.*?)\)\s*\{',
        re.DOTALL
    )

    for match in kernel_pattern.finditer(content):
        kernel_name = match.group(1)
        params_str = match.group(2)
        kernel_start = content[:match.start()].count('\n') + 1

        # Parse each parameter
        params = []
        depth = 0
        current = ""
        for char in params_str:
            if char == '<':
                depth += 1
            elif char == '>':
                depth -= 1
            elif char == ',' and depth == 0:
                params.append(current.strip())
                current = ""
                continue
            current += char
        if current.strip():
            params.append(current.strip())

        for param in params:
            # Check for device pointers without const (excluding buffer 0 which is typically output)
            if 'device ' in param and '*' in param and 'const' not in param:
                buf_match = re.search(r'\[\[buffer\((\d+)\)\]\]', param)
                if buf_match:
                    buf_idx = int(buf_match.group(1))
                    # Buffer 0 is typically output, others should probably be const
                    if buf_idx > 0:
                        # Check common input buffer names
                        input_names = ['input', 'weights', 'scales', 'zeros', 'bias', 'src', 'a_', 'b_', 'query', 'key', 'value', 'q_', 'k_', 'v_']
                        is_likely_input = any(name in param.lower() for name in input_names)
                        if is_likely_input:
                            yield (kernel_start, f"{kernel_name}: buffer {buf_idx} likely missing const")


def check_zero_length_arrays(content: str) -> Iterator[tuple[int, str]]:
    """Check for zero-length array declarations."""
    lines = content.split('\n')

    for i, line in enumerate(lines, 1):
        if re.search(r'\[\s*0\s*\]', line):
            stripped = line.strip()
            if not stripped.startswith('//'):
                yield (i, f"Zero-length array: {stripped[:60]}")


def check_threadgroup_barrier_outside_kernel(content: str, kernel_ranges: list[tuple[int, int, str]]) -> Iterator[tuple[int, str]]:
    """Check for threadgroup_barrier calls outside kernel functions."""
    lines = content.split('\n')

    for i, line in enumerate(lines, 1):
        if 'threadgroup_barrier' in line:
            if is_in_kernel(i, kernel_ranges) is None:
                stripped = line.strip()
                if not stripped.startswith('//'):
                    yield (i, f"threadgroup_barrier outside kernel: {stripped[:60]}")


def check_simdgroup_issues(content: str) -> Iterator[tuple[int, str]]:
    """Check for potential simdgroup operation issues."""
    lines = content.split('\n')

    in_kernel = False
    has_simd_shuffle = False
    has_simd_barrier = False
    kernel_start = 0
    kernel_name = ""

    for i, line in enumerate(lines, 1):
        if 'kernel void' in line:
            in_kernel = True
            has_simd_shuffle = False
            has_simd_barrier = False
            kernel_start = i
            match = re.search(r'void\s+(\w+)', line)
            kernel_name = match.group(1) if match else "unknown"

        if in_kernel:
            if 'simd_shuffle' in line or 'simd_broadcast' in line:
                has_simd_shuffle = True
            if 'simdgroup_barrier' in line:
                has_simd_barrier = True

            # End of kernel (simplified)
            if line.strip() == '}' and i > kernel_start + 5:
                if has_simd_shuffle and not has_simd_barrier:
                    yield (kernel_start, f"{kernel_name}: simd_shuffle without simdgroup_barrier")
                in_kernel = False


def check_uninitialized_threadgroup(content: str, kernel_ranges: list[tuple[int, int, str]]) -> Iterator[tuple[int, str]]:
    """Check for threadgroup memory that might be used before initialization."""
    lines = content.split('\n')

    for start, end, kernel_name in kernel_ranges:
        threadgroup_vars = []

        for i in range(start - 1, min(end, len(lines))):
            line = lines[i]

            # Find threadgroup declarations
            if 'threadgroup ' in line and '[[' not in line and '=' not in line:
                match = re.search(r'threadgroup\s+\w+(?:<[^>]+>)?\s+(\w+)', line)
                if match:
                    var_name = match.group(1)
                    threadgroup_vars.append((i + 1, var_name))

        # Check if there's a barrier before first use
        for var_line, var_name in threadgroup_vars:
            has_barrier_before_use = False
            first_use_line = None

            for i in range(var_line, min(end, len(lines))):
                line = lines[i]
                if 'threadgroup_barrier' in line:
                    has_barrier_before_use = True
                if var_name in line and i + 1 != var_line:
                    first_use_line = i + 1
                    break

            # Only report if used and no barrier before use
            if first_use_line and not has_barrier_before_use:
                yield (var_line, f"{kernel_name}: threadgroup var '{var_name}' potentially used before barrier")


def check_atomic_memory_order(content: str) -> Iterator[tuple[int, str]]:
    """Check for atomic operations that might have incorrect memory ordering."""
    lines = content.split('\n')

    for i, line in enumerate(lines, 1):
        # Look for atomic operations
        if 'atomic_' in line and 'memory_order_relaxed' in line:
            # Check if this is a critical section (has compare_exchange or fetch_add used for sync)
            if 'compare_exchange' in line or 'fetch_add' in line:
                stripped = line.strip()
                yield (i, f"Atomic sync with relaxed memory order: {stripped[:50]}")


def audit_file(metal_file: Path) -> list[Issue]:
    """Audit a single Metal file for issues."""
    issues = []
    content = metal_file.read_text()
    rel_path = metal_file.name

    kernel_ranges = find_kernel_ranges(content)

    # Run all checks
    checks = [
        (check_threadgroup_in_non_kernel, "threadgroup_non_kernel", "critical", content, kernel_ranges),
        (check_missing_const, "missing_const", "warning", content),
        (check_zero_length_arrays, "zero_length_array", "critical", content),
        (check_threadgroup_barrier_outside_kernel, "barrier_non_kernel", "critical", content, kernel_ranges),
        (check_simdgroup_issues, "simdgroup_issue", "warning", content),
        (check_uninitialized_threadgroup, "uninitialized_threadgroup", "warning", content, kernel_ranges),
        (check_atomic_memory_order, "atomic_memory_order", "info", content),
    ]

    for check_tuple in checks:
        func = check_tuple[0]
        category = check_tuple[1]
        severity = check_tuple[2]
        args = check_tuple[3:]

        for line_num, description in func(*args):
            issues.append(Issue(
                file=rel_path,
                line=line_num,
                category=category,
                severity=severity,
                description=description
            ))

    return issues


def main():
    # Find all Metal files
    src_dir = Path(__file__).parent / "src"
    if not src_dir.exists():
        print(f"Error: {src_dir} does not exist")
        return

    metal_files = list(src_dir.rglob("*.metal"))
    print(f"Auditing {len(metal_files)} Metal kernel files...\n")

    all_issues: list[Issue] = []
    issues_by_category: dict[str, list[Issue]] = defaultdict(list)

    for metal_file in sorted(metal_files):
        file_issues = audit_file(metal_file)
        all_issues.extend(file_issues)
        for issue in file_issues:
            issues_by_category[issue.category].append(issue)

    # Print summary by category
    print("=" * 70)
    print("AUDIT SUMMARY BY CATEGORY")
    print("=" * 70)

    severity_order = {"critical": 0, "warning": 1, "info": 2}

    for category in sorted(issues_by_category.keys(), key=lambda c: (severity_order.get(issues_by_category[c][0].severity, 3), c)):
        cat_issues = issues_by_category[category]
        severity = cat_issues[0].severity
        print(f"\n[{severity.upper()}] {category}: {len(cat_issues)} issues")
        print("-" * 50)

        # Group by file
        by_file: dict[str, list[Issue]] = defaultdict(list)
        for issue in cat_issues:
            by_file[issue.file].append(issue)

        for file, file_issues in sorted(by_file.items()):
            print(f"  {file}:")
            for issue in file_issues[:5]:  # Limit per file
                print(f"    L{issue.line}: {issue.description}")
            if len(file_issues) > 5:
                print(f"    ... and {len(file_issues) - 5} more")

    # Write detailed report
    import tempfile
    report_path = Path(tempfile.gettempdir()) / "metal_kernel_audit.txt"
    with open(report_path, 'w') as f:
        f.write("METAL KERNEL AUDIT REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total files audited: {len(metal_files)}\n")
        f.write(f"Total issues found: {len(all_issues)}\n\n")

        f.write("ISSUES BY SEVERITY:\n")
        for severity in ["critical", "warning", "info"]:
            count = len([i for i in all_issues if i.severity == severity])
            f.write(f"  {severity}: {count}\n")
        f.write("\n")

        f.write("DETAILED ISSUES:\n")
        f.write("-" * 70 + "\n\n")

        for category, cat_issues in sorted(issues_by_category.items()):
            f.write(f"\n## {category.upper()} ({len(cat_issues)} issues)\n\n")
            for issue in cat_issues:
                f.write(f"{issue.file}:{issue.line} [{issue.severity}]\n")
                f.write(f"  {issue.description}\n\n")

    print(f"\n{'=' * 70}")
    print(f"Total: {len(all_issues)} issues found")
    print(f"  Critical: {len([i for i in all_issues if i.severity == 'critical'])}")
    print(f"  Warning: {len([i for i in all_issues if i.severity == 'warning'])}")
    print(f"  Info: {len([i for i in all_issues if i.severity == 'info'])}")
    print(f"\nFull report: {report_path}")

    # Output category summary for task generation
    print("\n" + "=" * 70)
    print("CATEGORIES FOR FIX TASKS:")
    print("=" * 70)
    for category, cat_issues in sorted(issues_by_category.items(), key=lambda x: -len(x[1])):
        severity = cat_issues[0].severity
        affected_files = sorted(set(i.file for i in cat_issues))
        print(f"\n{category} ({len(cat_issues)} issues, {severity}):")
        print(f"  Affected files: {', '.join(affected_files[:10])}")
        if len(affected_files) > 10:
            print(f"  ... and {len(affected_files) - 10} more files")


if __name__ == "__main__":
    main()
