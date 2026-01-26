"""
Memory access pattern auditing for Metal GPU kernels.

This module provides:
1. Static analysis of Metal shader source to detect non-coalesced access patterns
2. Runtime benchmarks to validate coalescing on real hardware
3. Report generation with optimization suggestions

Key patterns detected:
- Non-coalesced global memory access (strided/scattered)
- Shared memory bank conflicts (32-way banks, 4-byte granularity)
- Sequential scale access in dequantization (cache thrashing)
- Strided Q/K/V access in attention (inefficient for long sequences)
- Scattered expert access in MoE dispatch
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Optional torch/Metal imports for runtime validation
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class AccessPattern(Enum):
    """Classification of memory access patterns."""

    COALESCED = "coalesced"  # Adjacent threads access adjacent addresses
    STRIDED = "strided"  # Regular stride > 1 between threads
    SCATTERED = "scattered"  # Irregular/data-dependent access
    BROADCAST = "broadcast"  # All threads read same address
    SEQUENTIAL = "sequential"  # Single thread sequential access (bad)
    BANK_CONFLICT = "bank_conflict"  # Shared memory bank conflict
    UNKNOWN = "unknown"


@dataclass
class MemoryAccess:
    """Represents a detected memory access in shader source."""

    line_number: int
    code: str
    buffer_name: str
    index_expr: str
    pattern: AccessPattern
    severity: str  # "info", "warning", "critical"
    suggestion: str
    context: str = ""  # Surrounding kernel context


@dataclass
class ShaderAnalysis:
    """Analysis results for a single shader file."""

    file_path: Path
    kernel_names: list[str] = field(default_factory=list)
    accesses: list[MemoryAccess] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    @property
    def critical_count(self) -> int:
        return sum(1 for a in self.accesses if a.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for a in self.accesses if a.severity == "warning")


@dataclass
class MemoryAuditReport:
    """Complete audit report across all analyzed shaders."""

    analyses: list[ShaderAnalysis] = field(default_factory=list)
    runtime_results: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def summary(self) -> str:
        """Generate human-readable summary."""
        total_critical = sum(a.critical_count for a in self.analyses)
        total_warning = sum(a.warning_count for a in self.analyses)
        total_kernels = sum(len(a.kernel_names) for a in self.analyses)

        lines = [
            "=" * 60,
            "Metal Marlin Memory Access Audit Report",
            "=" * 60,
            f"Files analyzed: {len(self.analyses)}",
            f"Kernels found: {total_kernels}",
            f"Critical issues: {total_critical}",
            f"Warnings: {total_warning}",
            "",
        ]

        for analysis in self.analyses:
            if analysis.accesses:
                lines.append(f"\n{analysis.file_path.name}:")
                lines.append("-" * 40)
                for access in analysis.accesses:
                    severity_icon = {"critical": "❌", "warning": "⚠️", "info": "ℹ️"}.get(
                        access.severity, "•"
                    )
                    lines.append(
                        f"  {severity_icon} Line {access.line_number}: {access.pattern.value}"
                    )
                    lines.append(f"     {access.code.strip()[:60]}")
                    lines.append(f"     → {access.suggestion}")

        if self.runtime_results:
            lines.append("\nRuntime Validation:")
            lines.append("-" * 40)
            for name, result in self.runtime_results.items():
                lines.append(f"  {name}: {result}")

        return "\n".join(lines)


class MemoryAuditor:
    """Audits Metal shader source for memory access patterns."""

    # Patterns that indicate non-coalesced access
    STRIDED_PATTERNS = [
        # Access with multiplication by dimension > 1
        r"\[\s*\w+\s*\*\s*(\w+)\s*\+",  # [i * N +
        r"\[\s*\w+\s*\*\s*head_dim\s*\+",
        r"\[\s*\w+\s*\*\s*hidden_dim\s*\+",
        r"\[\s*\w+\s*\*\s*num_experts\s*\+",
        # Column-major access in row-major layout
        r"\[\s*\w+\s*\*\s*N\s*\+\s*\w+\s*\]",  # k * N + n pattern
        # Strided loop access
        r"\[\s*\w+\s*\+\s*\w+\s*\*\s*stride",
    ]

    # Patterns indicating good coalesced access
    COALESCED_PATTERNS = [
        r"\[\s*tid\s*\]",  # Direct thread ID indexing
        r"\[\s*thread_idx\s*\]",
        r"\[\s*gid\.x\s*\]",
        r"\[\s*\w+\s*\+\s*tid\s*\]",  # base + tid
        r"\[\s*base_idx\s*\+\s*i\s*\]",  # Sequential within thread
    ]

    # Bank conflict patterns in threadgroup memory
    BANK_CONFLICT_PATTERNS = [
        # Accessing [row][col] where col has stride 32
        r"threadgroup\s+\w+\s+\w+\s*\[.*\]\s*\[32\]",
        # Power-of-2 stride access
        r"\[\s*\w+\s*\*\s*32\s*\]",
    ]

    # Scale access patterns (often sequential per-group)
    SCALE_ACCESS_PATTERNS = [
        r"scales\s*\[\s*group_idx",
        r"scales\s*\[\s*scale_k\s*\*",
        r"expert_scales\s*\[",
    ]

    def __init__(self, shader_dir: Path | str | None = None):
        """Initialize auditor with optional shader directory."""
        if shader_dir is None:
            # Default to metal_marlin/src
            self.shader_dir = Path(__file__).parent.parent.parent / "src"
        else:
            self.shader_dir = Path(shader_dir)

    def analyze_file(self, file_path: Path | str) -> ShaderAnalysis:
        """Analyze a single Metal shader file for memory access patterns."""
        file_path = Path(file_path)
        analysis = ShaderAnalysis(file_path=file_path)

        if not file_path.exists():
            analysis.warnings.append(f"File not found: {file_path}")
            return analysis

        content = file_path.read_text()
        lines = content.split("\n")

        # Extract kernel names
        kernel_pattern = r"kernel\s+void\s+(\w+)\s*\("
        analysis.kernel_names = re.findall(kernel_pattern, content)

        # Track current kernel context
        current_kernel = None
        brace_depth = 0

        for line_num, line in enumerate(lines, 1):
            # Track kernel boundaries
            kernel_match = re.search(kernel_pattern, line)
            if kernel_match:
                current_kernel = kernel_match.group(1)
                brace_depth = 0

            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0 and current_kernel:
                current_kernel = None

            # Skip comments and preprocessor
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("#"):
                continue

            # Check for buffer accesses
            buffer_accesses = re.findall(r"(\w+)\s*\[(.*?)\]", line)

            for buffer_name, index_expr in buffer_accesses:
                # Skip obviously local variables
                if buffer_name in ("vals", "row", "col", "out", "acc", "staging"):
                    continue

                access = self._classify_access(
                    line_num, line, buffer_name, index_expr, current_kernel
                )
                if access and access.pattern != AccessPattern.COALESCED:
                    analysis.accesses.append(access)

        # Add kernel-specific suggestions
        self._add_kernel_suggestions(analysis)

        return analysis

    def _classify_access(
        self,
        line_num: int,
        code: str,
        buffer_name: str,
        index_expr: str,
        kernel_context: str | None,
    ) -> MemoryAccess | None:
        """Classify a memory access pattern."""

        # Check for known good patterns first
        for pattern in self.COALESCED_PATTERNS:
            if re.search(pattern, f"[{index_expr}]"):
                return None  # Good access, don't report

        # Check for strided access
        for pattern in self.STRIDED_PATTERNS:
            if re.search(pattern, f"[{index_expr}]"):
                return MemoryAccess(
                    line_number=line_num,
                    code=code,
                    buffer_name=buffer_name,
                    index_expr=index_expr,
                    pattern=AccessPattern.STRIDED,
                    severity="warning",
                    suggestion=self._get_strided_suggestion(buffer_name, kernel_context),
                    context=kernel_context or "",
                )

        # Check for scale access patterns (often the bottleneck)
        for pattern in self.SCALE_ACCESS_PATTERNS:
            if re.search(pattern, code):
                return MemoryAccess(
                    line_number=line_num,
                    code=code,
                    buffer_name=buffer_name,
                    index_expr=index_expr,
                    pattern=AccessPattern.SEQUENTIAL,
                    severity="critical"
                    if "dequant" in (kernel_context or "").lower()
                    else "warning",
                    suggestion=(
                        "Scale access is sequential across K dimension. Consider: "
                        "1) Preload scales into threadgroup memory, "
                        "2) Use texture cache for scale buffer, "
                        "3) Tile scales along with weights"
                    ),
                    context=kernel_context or "",
                )

        # Check for scattered access (data-dependent indexing)
        if re.search(r"\[\s*\w+_ids?\s*\[", code) or re.search(r"token_batch\[", code):
            return MemoryAccess(
                line_number=line_num,
                code=code,
                buffer_name=buffer_name,
                index_expr=index_expr,
                pattern=AccessPattern.SCATTERED,
                severity="warning",
                suggestion=(
                    "Scattered access via indirect indexing. Consider: "
                    "1) Sort tokens by expert for better locality, "
                    "2) Use gather instructions if available, "
                    "3) Batch adjacent expert assignments together"
                ),
                context=kernel_context or "",
            )

        return None

    def _get_strided_suggestion(self, buffer_name: str, kernel_context: str | None) -> str:
        """Get context-specific suggestion for strided access."""
        ctx = (kernel_context or "").lower()
        buf = buffer_name.lower()

        if "scale" in buf:
            return (
                "Scale buffer strided access. Consider caching scales in "
                "threadgroup memory or using texture cache for better spatial locality."
            )

        if "k_vec" in buf or ctx and "attention" in ctx:
            return (
                "Key vector strided access (k * head_dim). Consider transposing K "
                "to [head_dim, seq_k] layout or using simdgroup shuffles for "
                "register-based transpose."
            )

        if "v_" in buf or "value" in buf:
            return (
                "Value vector strided access. Consider interleaving V with K for "
                "better cache utilization during softmax @ V computation."
            )

        if "expert" in buf or "router" in buf:
            return (
                "Expert weight strided access. Consider column-major layout for "
                "expert weights when N < K, or use blocked layouts for large experts."
            )

        return (
            "Strided memory access detected. Ensure adjacent threads access "
            "adjacent memory addresses for coalesced loads. Consider tiling "
            "or transposing data layout."
        )

    def _add_kernel_suggestions(self, analysis: ShaderAnalysis) -> None:
        """Add kernel-specific optimization suggestions."""
        for kernel in analysis.kernel_names:
            kernel_lower = kernel.lower()

            if "dequant" in kernel_lower:
                analysis.suggestions.append(
                    f"[{kernel}] Dequantization kernel: Ensure scale buffer uses "
                    "texture cache ([[texture]]) or is tiled into threadgroup memory "
                    "to avoid sequential global loads."
                )

            if "attention" in kernel_lower:
                analysis.suggestions.append(
                    f"[{kernel}] Attention kernel: Consider fused QKV layouts "
                    "([seq, 3*head_dim]) to improve Q/K/V access coalescing. "
                    "Flash attention style tiling reduces memory traffic."
                )

            if "moe" in kernel_lower or "expert" in kernel_lower:
                analysis.suggestions.append(
                    f"[{kernel}] MoE kernel: Token-to-expert sorting before dispatch "
                    "improves memory locality. Consider expert parallelism with "
                    "atomic accumulation for better GPU utilization."
                )

    def audit_all(self) -> MemoryAuditReport:
        """Audit all Metal shader files in the configured directory."""
        report = MemoryAuditReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))

        if not self.shader_dir.exists():
            return report

        for shader_file in sorted(self.shader_dir.glob("*.metal")):
            analysis = self.analyze_file(shader_file)
            report.analyses.append(analysis)

        return report

    def audit_priority_kernels(self) -> MemoryAuditReport:
        """Audit only the priority kernels: dequant, attention, MoE."""
        report = MemoryAuditReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))

        priority_files = [
            "dequant.metal",
            "attention.metal",
            "flash_attention.metal",
            "moe_router.metal",
            "moe_expert_gemm.metal",
            "marlin_gemm.metal",
        ]

        for filename in priority_files:
            shader_file = self.shader_dir / filename
            if shader_file.exists():
                analysis = self.analyze_file(shader_file)
                report.analyses.append(analysis)

        return report


def analyze_shader_file(file_path: str | Path) -> ShaderAnalysis:
    """Convenience function to analyze a single shader file."""
    auditor = MemoryAuditor()
    return auditor.analyze_file(file_path)


# -----------------------------------------------------------------------------
# Runtime validation with synthetic workloads
# -----------------------------------------------------------------------------


def run_coalescing_benchmark(
    pattern: str = "all", sizes: list[int] | None = None, device: str = "mps"
) -> dict[str, Any]:
    """
    Run runtime benchmarks to validate memory coalescing.

    Patterns:
    - "coalesced": Adjacent thread access (baseline)
    - "strided": Strided access (N stride)
    - "scattered": Random access
    - "scale": Sequential scale access pattern
    - "all": Run all patterns

    Returns timing and bandwidth measurements.
    """
    if not HAS_TORCH:
        return {"error": "torch not available for runtime benchmarks"}

    if not torch.backends.mps.is_available():
        return {"error": "MPS not available"}

    if sizes is None:
        sizes = [1024, 4096, 16384, 65536]

    results: dict[str, Any] = {}

    patterns_to_run = (
        ["coalesced", "strided", "scattered", "scale"] if pattern == "all" else [pattern]
    )

    for p in patterns_to_run:
        results[p] = _benchmark_pattern(p, sizes, device)

    return results


def _benchmark_pattern(pattern: str, sizes: list[int], device: str) -> dict[str, float]:
    """Benchmark a specific access pattern."""
    import torch

    results = {}

    for size in sizes:
        # Create test data
        data = torch.randn(size, device=device, dtype=torch.float16)

        # Create index pattern
        if pattern == "coalesced":
            indices = torch.arange(size, device=device)
        elif pattern == "strided":
            stride = 64  # Typical head_dim stride
            indices = torch.arange(0, size, stride, device=device) % size
        elif pattern == "scattered":
            indices = torch.randperm(size, device=device)
        elif pattern == "scale":
            # Simulate group_size=128 scale access
            group_size = 128
            num_groups = size // group_size
            indices = torch.arange(num_groups, device=device).repeat_interleave(group_size)
        else:
            indices = torch.arange(size, device=device)

        # Warmup
        for _ in range(10):
            _ = data[indices].sum()

        torch.mps.synchronize()

        # Benchmark
        iterations = 100
        start = time.perf_counter()

        for _ in range(iterations):
            data[indices].sum()

        torch.mps.synchronize()
        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000 / iterations
        bandwidth_gbps = (size * 2 * 2) / (elapsed_ms * 1e6)  # Read + write, 2 bytes/fp16

        results[f"size_{size}"] = {
            "time_ms": elapsed_ms,
            "bandwidth_gbps": bandwidth_gbps,
        }

    return results


# -----------------------------------------------------------------------------
# Detailed analysis of priority kernels
# -----------------------------------------------------------------------------


def analyze_dequant_patterns() -> dict[str, Any]:
    """
    Detailed analysis of FP4 dequantization memory patterns.

    Key findings from dequant.metal:

    1. Scale access pattern (CRITICAL):
       - scales[group_idx] accessed sequentially per group
       - group_idx = base_idx / group_size
       - When group_size=128, all 8 threads in a packed word share one scale
       - BUT different threadgroups may access different scale addresses

       Problem: Sequential scale access from global memory
       Solution: Cache scales in threadgroup memory before main loop

    2. Packed weight access (GOOD):
       - packed_weights[tid] is coalesced (adjacent threads, adjacent addresses)
       - Each thread loads one uint32 = 8 FP4 values

    3. Output access (GOOD):
       - output[tid * 2] with half4 stores is coalesced
       - 8-byte aligned stores utilize full memory transaction width
    """
    return {
        "kernel": "dequant_int4_kernel / dequant_fp4_kernel",
        "critical_issues": [
            {
                "pattern": "Sequential scale access",
                "location": "dequant.metal:257-259",
                "code": "uint group_idx = base_idx / group_size; half scale = scales[group_idx];",
                "impact": "Each thread independently loads scale from global memory",
                "fix": "Preload scales into threadgroup memory: threadgroup half scale_cache[MAX_GROUPS];",
            }
        ],
        "good_patterns": [
            "Packed weight loads are coalesced (one uint32 per thread)",
            "Output writes use half4 vectorized stores",
            "Magic bias dequantization is ALU-bound (no memory overhead)",
        ],
        "recommendations": [
            "Add texture cache attribute [[texture(N)]] to scales buffer",
            "Consider tiling: load scale once per tile, reuse across 128 elements",
            "For bulk dequant, prefetch next group's scale during compute",
        ],
    }


def analyze_attention_patterns() -> dict[str, Any]:
    """
    Detailed analysis of attention kernel memory patterns.

    Key findings from attention.metal:

    1. Q vector load (GOOD):
       - Cooperatively loaded into Q_cache[HEAD_DIM_MAX]
       - All threads participate, amortized cost

    2. K vector access (WARNING):
       - k_vec = k_base_ptr + k_idx * head_dim
       - Each thread loads k_vec[d] sequentially
       - Strided access pattern: k0[0], k0[1], k0[2]... then k1[0], k1[1]...

       Problem: Sequential load within thread, stride between threads
       Better: Transpose K to [head_dim, seq_k] for coalesced seq_k access

    3. Output P matrix (WARNING):
       - p_ptr[k_idx] written with stride 1 - good for write
       - BUT read back in pass 2 with same pattern - re-reading global memory

       Problem: Two-pass algorithm reads scores back from global
       Better: Keep scores in registers/threadgroup for single-pass

    4. Tiled variant K_tile load (GOOD):
       - K vectors loaded into threadgroup memory
       - Amortizes global memory access across all threads
    """
    return {
        "kernel": "attention_qk_softmax / attention_fused_qkv",
        "critical_issues": [
            {
                "pattern": "K vector strided sequential load",
                "location": "attention.metal:225-245",
                "code": "device const half* k_vec = k_base_ptr + k_idx * head_dim; ... k_vec[d]",
                "impact": "Each thread loads head_dim values sequentially from non-contiguous addresses",
                "fix": "Use tiled variant (attention_qk_softmax_tiled) or transpose K layout",
            },
            {
                "pattern": "Two-pass score materialization",
                "location": "attention.metal:268-290",
                "code": "p_ptr[k_idx] = half(score); ... float(p_ptr[k_idx])",
                "impact": "Writes then reads scores from global memory",
                "fix": "Use register-resident scores for small seq_k, or single-pass online softmax",
            },
        ],
        "good_patterns": [
            "Q vector cooperatively loaded to threadgroup memory",
            "Tiled variant caches K in threadgroup (8KB for 32 vecs @ dim=128)",
            "Simdgroup reductions avoid atomic overhead",
        ],
        "recommendations": [
            "Prefer attention_qk_softmax_tiled for seq_k <= 4096",
            "Consider transposed K layout [head_dim, seq_k] for long sequences",
            "Fuse softmax normalization to avoid score re-read",
        ],
    }


def analyze_moe_patterns() -> dict[str, Any]:
    """
    Detailed analysis of MoE kernel memory patterns.

    Key findings from moe_router.metal and moe_expert_gemm.metal:

    1. Router GEMM (WARNING):
       - w_col = router_weights + expert_idx
       - Accesses w_col[d * num_experts] - column-major in row-major storage

       Problem: Strided access (stride = num_experts)
       Better: Transpose router weights to [num_experts, hidden_dim]

    2. Expert dispatch (WARNING):
       - Assumes all tokens in tile use same expert (approximation)
       - moe_expert_gemm_fp4_grouped uses scattered token_batch[row] access

       Problem: Indirect indexing causes scattered memory access
       Better: Sort tokens by expert for contiguous access

    3. Expert weight loading (GOOD):
       - expert_weights[expert_id * stride] is contiguous per expert
       - Once expert is selected, weight access is coalesced

    4. Output accumulation (WARNING):
       - output[token_id * out_dim + col] += val
       - Multiple experts may write to same token (race condition on Metal)

       Problem: Atomic adds or separate buffers needed
       Better: Use expert-local output then reduce, or sort to avoid conflicts
    """
    return {
        "kernel": "moe_router_fused / moe_expert_gemm_fp4_grouped",
        "critical_issues": [
            {
                "pattern": "Router weight column-major access",
                "location": "moe_router.metal:223-225",
                "code": "device const half* w_col = router_weights + expert_idx; ... w_col[d * num_experts]",
                "impact": "Stride-num_experts access pattern, poor coalescing",
                "fix": "Transpose router_weights to [num_experts, hidden_dim] row-major",
            },
            {
                "pattern": "Scattered token activation gather",
                "location": "moe_expert_gemm.metal:519-534",
                "code": "uint token_id = token_batch[row]; val = activations[token_id * hidden_dim + col];",
                "impact": "Indirect indexing causes non-coalesced global loads",
                "fix": "Pre-sort tokens by expert ID for contiguous access patterns",
            },
        ],
        "good_patterns": [
            "Expert weights are contiguous per expert after selection",
            "Probability weighting is fused into output store",
            "Double-buffered K-loop hides memory latency",
        ],
        "recommendations": [
            "Transpose router weights for coalesced GEMM",
            "Sort tokens by expert before dispatch (host-side)",
            "Use grouped kernel (moe_expert_gemm_fp4_grouped) with sorted inputs",
            "Consider expert-local output buffers to avoid atomic contention",
        ],
    }


def generate_full_report(output_path: str | Path | None = None) -> str:
    """
    Generate comprehensive memory audit report.

    Combines:
    - Static shader analysis
    - Detailed pattern analysis for priority kernels
    - Runtime benchmark results (if available)
    - Actionable recommendations
    """
    auditor = MemoryAuditor()

    # Static analysis
    report = auditor.audit_priority_kernels()

    # Detailed analysis
    dequant_analysis = analyze_dequant_patterns()
    attention_analysis = analyze_attention_patterns()
    moe_analysis = analyze_moe_patterns()

    # Runtime benchmarks (if torch available)
    if HAS_TORCH and torch.backends.mps.is_available():
        report.runtime_results = run_coalescing_benchmark(pattern="all", sizes=[4096, 16384])

    # Build report
    lines = [
        report.summary(),
        "",
        "=" * 60,
        "DETAILED PATTERN ANALYSIS",
        "=" * 60,
        "",
        "## FP4 Dequantization",
        "-" * 40,
    ]

    for issue in dequant_analysis["critical_issues"]:
        lines.append(f"❌ {issue['pattern']}")
        lines.append(f"   Location: {issue['location']}")
        lines.append(f"   Fix: {issue['fix']}")
        lines.append("")

    lines.append("\n## Attention")
    lines.append("-" * 40)

    for issue in attention_analysis["critical_issues"]:
        lines.append(f"⚠️ {issue['pattern']}")
        lines.append(f"   Location: {issue['location']}")
        lines.append(f"   Fix: {issue['fix']}")
        lines.append("")

    lines.append("\n## MoE Dispatch")
    lines.append("-" * 40)

    for issue in moe_analysis["critical_issues"]:
        lines.append(f"⚠️ {issue['pattern']}")
        lines.append(f"   Location: {issue['location']}")
        lines.append(f"   Fix: {issue['fix']}")
        lines.append("")

    lines.append("\n" + "=" * 60)
    lines.append("TOP RECOMMENDATIONS")
    lines.append("=" * 60)
    lines.append("")
    lines.append("1. DEQUANT: Cache scales in threadgroup memory or use texture cache")
    lines.append("2. ATTENTION: Use tiled variant for better K-vector locality")
    lines.append("3. MOE: Transpose router weights and pre-sort tokens by expert")
    lines.append("4. GENERAL: Ensure output stores use vectorized half4/float4 writes")

    report_text = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(report_text)

    return report_text


if __name__ == "__main__":
    # CLI: Run full audit and print report
    print(generate_full_report())
