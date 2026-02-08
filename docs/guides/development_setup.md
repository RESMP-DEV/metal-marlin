# Development Setup Guide

## Virtual Environment Management

Metal Marlin uses **uv** for Python dependency management. This guide explains how to work with virtual environments correctly.

### Quick Start

```bash
# Install dependencies (creates .venv automatically)
uv sync

# Run commands in the project environment
uv run pytest tests/ -v
uv run python scripts/benchmark.py

# Check which environment is active
uv run python -c "import sys; print(sys.prefix)"
```

### Understanding uv's Environment Model

**uv automatically manages a single project environment at `.venv`:**

- `uv sync` creates/updates `.venv` based on `pyproject.toml`
- `uv run` always uses `.venv` (not `VIRTUAL_ENV` or other envs)
- No need for `source .venv/bin/activate` – `uv run` handles it

### Common Mistakes (and How to Avoid Them)

#### ❌ Don't: Create multiple environments

```bash
# DON'T DO THIS
uv venv .venv_test
uv venv .venv_backup
python -m venv .venv_old
```

**Why:** uv ignores these entirely. You'll waste disk space and create confusion.

#### ❌ Don't: Activate environments manually

```bash
# DON'T DO THIS
source .venv/bin/activate
python script.py
```

**Why:** This bypasses uv's dependency resolution and can lead to version mismatches.

#### ✅ Do: Use `uv run` for everything

```bash
# CORRECT
uv run python script.py
uv run pytest tests/
uv run metal-marlin serve model/
```

### Cleanup Existing Environments

If you have leftover `.venv_*` directories from previous experiments:

```bash
# Run the cleanup script
./scripts/cleanup_venvs.sh

# Or manually remove them
rm -rf .venv_run .venv_old .venv_broken .venv_*
```

**Safe deletion criteria:**
- Keep: `.venv` (the active project environment)
- Delete: Everything else (`.venv_*`, old backups, etc.)

### Troubleshooting

#### "Wrong Python version" errors

```bash
# Check .python-version file
cat .python-version  # Should show: 3.12

# Verify uv is using correct version
uv run python --version  # Should show: Python 3.12.x
```

#### "Package not found" errors

```bash
# Resync dependencies
uv sync --extra all

# Force clean rebuild
rm -rf .venv
uv sync --extra all
```

#### Multiple environments detected

```bash
# Check what uv is using
uv run python -c "import sys; print(sys.prefix)"

# Should always be: /path/to/metal_marlin/.venv
# If not, delete other environments and run: uv sync
```

### CI/CD Best Practices

For automated builds:

```bash
# In CI scripts, always use uv run
uv sync --frozen  # Install exact versions from uv.lock
uv run pytest tests/ -v
uv run ruff check .
```

### Integration with AlphaHENG

When running Metal Marlin tasks through AlphaHENG:

```yaml
# tasks/example.yaml
- name: test-metal-marlin
  prompt: |
    cd contrib/metal_marlin
    uv run pytest tests/test_gemm.py -v
  priority: P0
```

**Why `cd contrib/metal_marlin`:** AlphaHENG agents execute from repo root.  
**Why `uv run`:** Ensures correct environment regardless of agent state.

### Further Reading

- [uv documentation](https://github.com/astral-sh/uv)
- [Python 3.12 requirement](../../../../AGENTS.md#python_environment)
- [AlphaHENG task conventions](../../../../AGENTS.md#task_yaml)
