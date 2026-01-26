"""Pytest configuration for Metal Marlin accuracy tests.

Test speed tiers:
- Default: All tests (~1567 tests, ~15 min)
- Fast mode (--fast): Core sanity tests only (~150 tests, ~30s)
- Parallel: Use -n auto for parallel execution

Usage:
    pytest tests/ --fast              # Quick feedback (30s)
    pytest tests/ -n auto             # Parallel full suite (~3 min)
    pytest tests/ -n auto --fast      # Parallel fast (~10s)
    pytest tests/ -m "not slow"       # Skip slow markers
"""

import pytest
from metal_marlin._compat import HAS_MLX

# Skip marker for MLX-dependent tests
requires_mlx = pytest.mark.skipif(not HAS_MLX, reason="Requires MLX (Apple Silicon only)")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (perplexity, large GEMM)",
    )
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Run only essential tests for quick feedback (~150 tests, 30s)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow (requires --run-slow)")
    config.addinivalue_line("markers", "expensive: mark test as computationally expensive")
    config.addinivalue_line("markers", "smoke: essential smoke test (always runs)")
    config.addinivalue_line("markers", "requires_mlx: mark test as requiring MLX (Apple Silicon)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # Skip slow tests unless --run-slow is specified
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Need --run-slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Fast mode: only run first test of each parametrized group + smoke tests
    if config.getoption("--fast"):
        seen_base_names: set[str] = set()
        skip_fast = pytest.mark.skip(reason="--fast mode: skipping redundant parametrized test")

        for item in items:
            # Always run smoke tests
            if "smoke" in item.keywords:
                continue

            # Get base test name (without parametrize suffix)
            # e.g., "test_foo[param1]" -> "test_foo"
            base_name = item.name.split("[")[0]
            full_base = f"{item.parent.name}::{base_name}" if item.parent else base_name

            # Only run first instance of parametrized tests
            if full_base in seen_base_names:
                item.add_marker(skip_fast)
            else:
                seen_base_names.add(full_base)
