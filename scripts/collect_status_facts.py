#!/usr/bin/env python3
"""
Collect codebase facts as machine-readable XML.

This script ONLY collects facts - no prose generation.
An LLM agent task consumes this XML to generate STATUS.md.

NO imports of optional packages (no MLX, no torch detection via import).
All detection is done via grep/filesystem inspection.

Usage:
    uv run python scripts/collect_status_facts.py > facts.xml
"""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from xml.dom import minidom

ROOT = Path(__file__).parent.parent
SRC_DIR = ROOT / "src"
TESTS_DIR = ROOT / "tests"
BENCHMARKS_DIR = ROOT / "benchmarks"
METAL_MARLIN_DIR = ROOT / "metal_marlin"


def grep_count(pattern: str, path: str = ".", include: str = "*.py") -> tuple[int, list[str]]:
    """Count files matching grep pattern. Returns (count, file_list)."""
    try:
        result = subprocess.run(
            ["grep", "-rl", pattern, f"--include={include}", path],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        files = [f for f in result.stdout.strip().split("\n") if f]
        return len(files), files
    except Exception:
        return -1, []


def check_command_exists(cmd: str) -> bool:
    """Check if a command/binary exists."""
    try:
        subprocess.run(["which", cmd], capture_output=True, check=True)
        return True
    except Exception:
        return False


def get_pyproject_deps() -> list[str]:
    """Extract dependencies from pyproject.toml."""
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():
        return []

    deps = []
    in_deps = False
    content = pyproject.read_text()
    for line in content.split("\n"):
        if "dependencies" in line and "=" in line:
            in_deps = True
            continue
        if in_deps:
            if line.strip().startswith("]"):
                in_deps = False
            elif line.strip().startswith('"') or line.strip().startswith("'"):
                dep = line.strip().strip("\",'").split("[")[0].split(">=")[0].split("==")[0]
                deps.append(dep)
    return deps


def compile_shader(metal_file: Path) -> dict:
    """Try to compile a Metal shader using xcrun."""
    try:
        result = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", str(metal_file), "-o", "/dev/null"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return {"compiles": True, "error": None}
        else:
            # Extract first error line
            stderr = result.stderr
            for line in stderr.split("\n"):
                if "error:" in line:
                    return {"compiles": False, "error": line.strip()[:200]}
            return {"compiles": False, "error": stderr[:200] if stderr else "Unknown error"}
    except Exception as e:
        return {"compiles": False, "error": str(e)[:100]}


def collect_facts() -> ET.Element:
    """Collect all facts into an XML structure."""
    root = ET.Element("status_facts")
    root.set("timestamp", datetime.now().isoformat())
    root.set("project", "metal_marlin")

    # === MLX References (the thing we're removing) ===
    mlx_elem = ET.SubElement(root, "mlx_references")
    mlx_elem.set("goal", "zero")

    # Build patterns via concatenation to avoid matching this script itself
    # Keys also use concatenation to prevent self-matching on dictionary key literals
    _mlx = "ml" + "x"
    _mx = "m" + "x"
    patterns = {
        "import_" + _mlx: f"import {_mlx}",
        "from_" + _mlx: f"from {_mlx}",
        _mlx + "_lm": f"{_mlx}_lm",
        _mx + "_fast": f"{_mx}.fast",
        _mx + "_core": f"{_mx}.core",
        _mx + "_nn": f"{_mx}.nn",
    }

    all_mlx_files = set()
    for key, pattern in patterns.items():
        count, files = grep_count(pattern)
        ref = ET.SubElement(mlx_elem, "pattern")
        ref.set("name", key)
        ref.set("grep", pattern)
        ref.set("count", str(count))
        all_mlx_files.update(files)
        for f in files[:10]:  # Cap at 10 examples
            file_elem = ET.SubElement(ref, "file")
            file_elem.text = f

    mlx_elem.set("total_files", str(len(all_mlx_files)))

    # === Metal Shaders ===
    shaders_elem = ET.SubElement(root, "metal_shaders")
    shaders_elem.set("directory", "src/")

    if SRC_DIR.exists():
        metal_files = sorted(SRC_DIR.glob("*.metal"))
        compiling = 0
        for mf in metal_files:
            shader = ET.SubElement(shaders_elem, "shader")
            shader.set("name", mf.stem)
            shader.set("path", str(mf.relative_to(ROOT)))

            result = compile_shader(mf)
            shader.set("compiles", str(result["compiles"]).lower())
            if result["compiles"]:
                compiling += 1
            if result["error"]:
                error_elem = ET.SubElement(shader, "error")
                error_elem.text = result["error"]

        shaders_elem.set("total", str(len(metal_files)))
        shaders_elem.set("compiling", str(compiling))

    # === File Counts ===
    counts_elem = ET.SubElement(root, "file_counts")

    def add_count(name: str, pattern: str, path: Path):
        elem = ET.SubElement(counts_elem, "count")
        elem.set("name", name)
        if path.exists():
            files = list(path.rglob(pattern)) if "**" not in pattern else list(path.glob(pattern))
            elem.set("value", str(len(files)))
        else:
            elem.set("value", "0")
            elem.set("missing", "true")

    add_count("python_total", "*.py", ROOT)
    add_count("python_metal_marlin", "*.py", METAL_MARLIN_DIR)
    add_count("python_tests", "*.py", TESTS_DIR)
    add_count("python_benchmarks", "*.py", BENCHMARKS_DIR)
    add_count("metal_shaders", "*.metal", SRC_DIR)

    # === Quantized Models ===
    models_elem = ET.SubElement(root, "quantized_models")
    results_dir = BENCHMARKS_DIR / "results"

    if results_dir.exists():
        for item in results_dir.iterdir():
            if item.is_dir():
                model = ET.SubElement(models_elem, "model")
                model.set("name", item.name)

                safetensors = list(item.glob("*.safetensors"))
                model.set("has_weights", str(len(safetensors) > 0).lower())
                model.set("has_config", str((item / "config.json").exists()).lower())

                if safetensors:
                    size_bytes = sum(f.stat().st_size for f in safetensors)
                    model.set("size_mb", f"{size_bytes / 1024 / 1024:.1f}")

    # === Dependencies (from pyproject.toml, not imports) ===
    deps_elem = ET.SubElement(root, "dependencies")
    deps_elem.set("source", "pyproject.toml")

    pyproject_deps = get_pyproject_deps()
    for dep in pyproject_deps:
        dep_elem = ET.SubElement(deps_elem, "dependency")
        dep_elem.set("name", dep)
        # Flag MLX as unwanted
        if "mlx" in dep.lower():
            dep_elem.set("status", "remove")

    # === Task Queue (if AlphaHENG available) ===
    try:
        result = subprocess.run(
            ["uv", "run", "alphaheng", "status"],
            cwd=ROOT.parent.parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            queue_elem = ET.SubElement(root, "task_queue")
            for line in result.stdout.split("\n"):
                if "Pending:" in line:
                    queue_elem.set("pending", line.split()[-1])
                elif "In Progress:" in line:
                    queue_elem.set("in_progress", line.split()[-1])
                elif "Completed:" in line:
                    queue_elem.set("completed", line.split()[-1])
    except Exception:
        pass

    return root


def main():
    facts = collect_facts()

    # Pretty print XML
    xml_str = ET.tostring(facts, encoding="unicode")
    dom = minidom.parseString(xml_str)
    print(dom.toprettyxml(indent="  "))


if __name__ == "__main__":
    main()
