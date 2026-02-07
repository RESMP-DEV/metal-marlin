#!/usr/bin/env python3
"""Generate atomic mixed-BPW fairway autotune tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import textwrap


@dataclass(frozen=True)
class TaskSpec:
    name: str
    priority: str
    prompt: str
    dependencies: tuple[str, ...] = ()
    verify_command: str | None = None


def _clean(text: str) -> str:
    return textwrap.dedent(text).strip("\n")


def _flow_list(values: tuple[str, ...]) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join(values) + "]"


def _indent_block(text: str, spaces: int = 6) -> str:
    pad = " " * spaces
    return "\n".join(f"{pad}{line}" if line else pad for line in text.splitlines())


def build_tasks() -> list[TaskSpec]:
    work_root = "contrib/metal_marlin/agent_workspace/mixed_bpw_fairway_autotune"
    candidates_dir = f"{work_root}/candidates"
    results_dir = f"{work_root}/results"
    source_kernel = "contrib/metal_marlin/src/gemm_trellis_moe.metal"

    baseline_result = f"{results_dir}/baseline_decode.json"
    async_candidate = f"{candidates_dir}/gemm_trellis_moe__async_prefetch_hint.metal"
    simd_candidate = f"{candidates_dir}/gemm_trellis_moe__simdgroup_barrier.metal"
    threads_candidate = f"{candidates_dir}/gemm_trellis_moe__max_threads_256.metal"

    async_result = f"{results_dir}/async_prefetch_hint_decode.json"
    simd_result = f"{results_dir}/simdgroup_barrier_decode.json"
    threads_result = f"{results_dir}/max_threads_256_decode.json"

    summary_json = f"{results_dir}/post_collect_summary.json"
    summary_md = "contrib/metal_marlin/docs/glm47_mixed_bpw_fairway_autotune_summary.md"

    baseline_task = "mixed-bpw-fairway-baseline-benchmark"
    async_transform_task = "mixed-bpw-fairway-transform-async-prefetch-hint"
    simd_transform_task = "mixed-bpw-fairway-transform-simdgroup-barrier"
    threads_transform_task = "mixed-bpw-fairway-transform-max-threads-256"

    async_bench_task = "mixed-bpw-fairway-benchmark-async-prefetch-hint"
    simd_bench_task = "mixed-bpw-fairway-benchmark-simdgroup-barrier"
    threads_bench_task = "mixed-bpw-fairway-benchmark-max-threads-256"

    return [
        TaskSpec(
            name=baseline_task,
            priority="P0",
            prompt=_clean(
                f"""
                Run one baseline benchmark action for mixed-BPW fairway decode.

                1. Ensure this directory exists:
                   - `{results_dir}`
                2. Run:
                   `uv run python contrib/metal_marlin/benchmarks/benchmark_mixed_bpw_decode.py --prompt-len 256 --decode-tokens 192 --warmup 8 --runs 5 --output {baseline_result}`
                3. If the benchmark is skipped because `ALPHAHENG_TASK_MODE=1`,
                   write `{baseline_result}` manually with keys:
                   `benchmark`, `variant`, `status`, and `timestamp_utc`.
                """
            ),
            verify_command=_clean(
                f"""
                test -f {baseline_result} &&
                rg -q "mixed_bpw_decode|benchmark" {baseline_result} &&
                rg -q "decode_tok_per_s|status" {baseline_result}
                """
            ),
        ),
        TaskSpec(
            name=async_transform_task,
            priority="P1",
            prompt=_clean(
                f"""
                Apply exactly one transform action: `async_prefetch_hint`.

                Source file:
                - `{source_kernel}`

                Output candidate file:
                - `{async_candidate}`

                Rules:
                1. Copy the source to the candidate path (do not edit source in place).
                2. Insert this one comment line near the main kernel entry:
                   `// Prefetch hint: consider simdgroup_async_copy for mixed_bpw_fairway`
                3. Do not apply any other transformations in this task.
                """
            ),
            verify_command=_clean(
                f"""
                test -f {async_candidate} &&
                rg -q "simdgroup_async_copy for mixed_bpw_fairway" {async_candidate}
                """
            ),
        ),
        TaskSpec(
            name=async_bench_task,
            priority="P1",
            dependencies=(baseline_task, async_transform_task),
            prompt=_clean(
                f"""
                Run one benchmark action for the `async_prefetch_hint` candidate.

                Inputs:
                - Candidate: `{async_candidate}`
                - Source to swap temporarily: `{source_kernel}`

                Output:
                - `{async_result}`

                Procedure:
                1. Back up `{source_kernel}` to `{work_root}/gemm_trellis_moe.backup.metal`.
                2. Copy `{async_candidate}` over `{source_kernel}`.
                3. Run:
                   `uv run python contrib/metal_marlin/benchmarks/benchmark_mixed_bpw_decode.py --prompt-len 256 --decode-tokens 192 --warmup 8 --runs 5 --output {async_result}`
                4. Restore `{source_kernel}` from the backup.
                5. If skipped because `ALPHAHENG_TASK_MODE=1`, write `{async_result}`
                   manually with keys `benchmark`, `variant`, `status`, `timestamp_utc`.
                """
            ),
            verify_command=_clean(
                f"""
                test -f {async_result} &&
                rg -q "async_prefetch_hint|variant" {async_result} &&
                rg -q "decode_tok_per_s|status" {async_result}
                """
            ),
        ),
        TaskSpec(
            name=simd_transform_task,
            priority="P1",
            prompt=_clean(
                f"""
                Apply exactly one transform action: `simdgroup_barrier`.

                Source file:
                - `{source_kernel}`

                Output candidate file:
                - `{simd_candidate}`

                Rules:
                1. Copy the source to the candidate path (do not edit source in place).
                2. Replace only the first occurrence of:
                   `threadgroup_barrier(mem_flags::mem_threadgroup)`
                   with:
                   `simdgroup_barrier(mem_flags::mem_none)`
                3. Do not apply any other transformations in this task.
                """
            ),
            verify_command=_clean(
                f"""
                test -f {simd_candidate} &&
                rg -q "simdgroup_barrier\\(mem_flags::mem_none\\)" {simd_candidate}
                """
            ),
        ),
        TaskSpec(
            name=simd_bench_task,
            priority="P1",
            dependencies=(baseline_task, simd_transform_task),
            prompt=_clean(
                f"""
                Run one benchmark action for the `simdgroup_barrier` candidate.

                Inputs:
                - Candidate: `{simd_candidate}`
                - Source to swap temporarily: `{source_kernel}`

                Output:
                - `{simd_result}`

                Procedure:
                1. Back up `{source_kernel}` to `{work_root}/gemm_trellis_moe.backup.metal`.
                2. Copy `{simd_candidate}` over `{source_kernel}`.
                3. Run:
                   `uv run python contrib/metal_marlin/benchmarks/benchmark_mixed_bpw_decode.py --prompt-len 256 --decode-tokens 192 --warmup 8 --runs 5 --output {simd_result}`
                4. Restore `{source_kernel}` from the backup.
                5. If skipped because `ALPHAHENG_TASK_MODE=1`, write `{simd_result}`
                   manually with keys `benchmark`, `variant`, `status`, `timestamp_utc`.
                """
            ),
            verify_command=_clean(
                f"""
                test -f {simd_result} &&
                rg -q "simdgroup_barrier|variant" {simd_result} &&
                rg -q "decode_tok_per_s|status" {simd_result}
                """
            ),
        ),
        TaskSpec(
            name=threads_transform_task,
            priority="P1",
            prompt=_clean(
                f"""
                Apply exactly one transform action: `max_threads_256`.

                Source file:
                - `{source_kernel}`

                Output candidate file:
                - `{threads_candidate}`

                Rules:
                1. Copy the source to the candidate path (do not edit source in place).
                2. Add the attribute `[[max_total_threads_per_threadgroup(256)]]`
                   immediately above the first `kernel void` declaration.
                3. Do not apply any other transformations in this task.
                """
            ),
            verify_command=_clean(
                f"""
                test -f {threads_candidate} &&
                rg -q "\\[\\[max_total_threads_per_threadgroup\\(256\\)\\]\\]" {threads_candidate}
                """
            ),
        ),
        TaskSpec(
            name=threads_bench_task,
            priority="P1",
            dependencies=(baseline_task, threads_transform_task),
            prompt=_clean(
                f"""
                Run one benchmark action for the `max_threads_256` candidate.

                Inputs:
                - Candidate: `{threads_candidate}`
                - Source to swap temporarily: `{source_kernel}`

                Output:
                - `{threads_result}`

                Procedure:
                1. Back up `{source_kernel}` to `{work_root}/gemm_trellis_moe.backup.metal`.
                2. Copy `{threads_candidate}` over `{source_kernel}`.
                3. Run:
                   `uv run python contrib/metal_marlin/benchmarks/benchmark_mixed_bpw_decode.py --prompt-len 256 --decode-tokens 192 --warmup 8 --runs 5 --output {threads_result}`
                4. Restore `{source_kernel}` from the backup.
                5. If skipped because `ALPHAHENG_TASK_MODE=1`, write `{threads_result}`
                   manually with keys `benchmark`, `variant`, `status`, `timestamp_utc`.
                """
            ),
            verify_command=_clean(
                f"""
                test -f {threads_result} &&
                rg -q "max_threads_256|variant" {threads_result} &&
                rg -q "decode_tok_per_s|status" {threads_result}
                """
            ),
        ),
        TaskSpec(
            name="mixed-bpw-fairway-post-collect",
            priority="P2",
            dependencies=(baseline_task, async_bench_task, simd_bench_task, threads_bench_task),
            prompt=_clean(
                f"""
                Run one post-collect action to aggregate fairway autotune results.

                Inputs:
                - `{baseline_result}`
                - `{async_result}`
                - `{simd_result}`
                - `{threads_result}`

                Outputs:
                - `{summary_json}`
                - `{summary_md}`

                Required output content:
                1. Per-variant decode metrics (`decode_ms_per_token`, `decode_tok_per_s`) when present.
                2. `speedup_vs_baseline` for each variant when baseline and variant throughput are available.
                3. Explicit skipped status propagation when benchmark files are skip-only.
                4. A winner section naming the best non-skipped variant if one exists.
                """
            ),
            verify_command=_clean(
                f"""
                test -f {summary_json} &&
                test -f {summary_md} &&
                rg -q "speedup_vs_baseline|winner|skipped" {summary_json}
                """
            ),
        ),
    ]


def validate_tasks(tasks: list[TaskSpec]) -> None:
    has_baseline = False
    has_post_collect = False

    for task in tasks:
        if task.priority in {"P0", "P1"} and not task.verify_command:
            raise ValueError(f"P0/P1 task missing verify_command: {task.name}")

        if "contrib/metal_marlin/" not in task.prompt:
            raise ValueError(f"Prompt must use contrib-scoped paths: {task.name}")

        if "baseline" in task.name:
            has_baseline = True
        if "post-collect" in task.name or "post_collect" in task.name:
            has_post_collect = True

    if not has_baseline:
        raise ValueError("At least one baseline benchmark task is required.")
    if not has_post_collect:
        raise ValueError("At least one post-collect task is required.")


def render_yaml(tasks: list[TaskSpec]) -> str:
    generated = datetime.now(timezone.utc).isoformat()

    lines = [
        "# yaml-language-server: $schema=https://raw.githubusercontent.com/RESMP-DEV/alphaheng/main/configs/task_schema.json",
        "# Mixed-BPW Fairway Autotune",
        "# Scope: contrib/metal_marlin only",
        f"# Generated: {generated}",
        "",
        "tasks:",
    ]

    for task in tasks:
        lines.append(f"  - name: {task.name}")
        lines.append(f"    priority: {task.priority}")
        if task.dependencies:
            lines.append(f"    dependencies: {_flow_list(task.dependencies)}")
        lines.append("    prompt: |")
        lines.append(_indent_block(task.prompt, spaces=6))
        if task.verify_command:
            lines.append("    verify_command: |")
            lines.append(_indent_block(task.verify_command, spaces=6))

    return "\n".join(lines) + "\n"


def main() -> None:
    script_path = Path(__file__).resolve()
    metal_marlin_root = script_path.parents[1]
    output_path = metal_marlin_root / "tasks" / "mixed_bpw_fairway_autotune.yaml"

    tasks = build_tasks()
    validate_tasks(tasks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_yaml(tasks), encoding="utf-8")

    print(f"Generated {len(tasks)} tasks at {output_path}")


if __name__ == "__main__":
    main()
