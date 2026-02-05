"""Pytest configuration for Metal Marlin tests.

Provides PyTorch MPS fixtures and test configuration for Metal Marlin's
quantized inference engine. All fixtures are designed for Apple Silicon
hardware with MPS acceleration.

Test speed tiers:
- Default: All tests (~1500+ tests)
- Fast mode (--fast): Core sanity tests only (~150 tests, ~30s)
- Parallel: Use -n auto for parallel execution

Usage:
    pytest tests/ --fast              # Quick feedback (30s)
    pytest tests/ -n auto             # Parallel full suite
    pytest tests/ -n auto --fast      # Parallel fast (~10s)
    pytest tests/ -m "not slow"       # Skip slow markers
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

try:
    from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch
except Exception:  # pragma: no cover - fallback for partial installs
    import importlib.util
    import sys
    from types import ModuleType
    from pathlib import Path

    _pkg_root = Path(__file__).resolve().parents[1] / "metal_marlin"
    _pkg_name = "metal_marlin"
    if _pkg_name not in sys.modules:
        _pkg = ModuleType(_pkg_name)
        _pkg.__path__ = [str(_pkg_root)]
        sys.modules[_pkg_name] = _pkg

    _compat_path = _pkg_root / "_compat.py"
    _spec = importlib.util.spec_from_file_location(f"{_pkg_name}._compat", _compat_path)
    if _spec is None or _spec.loader is None:
        raise
    _compat = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _compat
    _spec.loader.exec_module(_compat)

    HAS_MPS = _compat.HAS_MPS
    HAS_TORCH = _compat.HAS_TORCH
    torch = _compat.torch

if TYPE_CHECKING:
    import torch as torch_types

# Skip markers for hardware-dependent tests
requires_mps = pytest.mark.skipif(not HAS_MPS, reason="Requires MPS (Apple Silicon)")
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")

# Legacy marker (deprecated, always skips since MLX is removed)
requires_mlx = pytest.mark.skip(reason="MLX support removed; use PyTorch MPS")


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
    parser.addoption(
        "--device",
        action="store",
        default=None,
        help="Force specific device (mps, cpu). Default: auto-detect MPS",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow (requires --run-slow)")
    config.addinivalue_line("markers", "expensive: mark test as computationally expensive")
    config.addinivalue_line("markers", "smoke: essential smoke test (always runs)")
    config.addinivalue_line("markers", "requires_mps: mark test as requiring MPS (Apple Silicon)")
    config.addinivalue_line("markers", "requires_torch: mark test as requiring PyTorch")
    config.addinivalue_line("markers", "requires_mlx: DEPRECATED - MLX support removed")


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


# ==============================================================================
# Device fixtures
# ==============================================================================


@pytest.fixture(scope="session")
def device(request: pytest.FixtureRequest) -> str:
    """Return the test device: 'mps' if available, otherwise 'cpu'.

    Can be overridden with --device=<device> option.
    """
    forced = request.config.getoption("--device")
    if forced:
        return forced
    if HAS_MPS:
        return "mps"
    return "cpu"


@pytest.fixture(scope="session")
def mps_device() -> str:
    """Return 'mps' device, skipping if MPS is unavailable."""
    if not HAS_MPS:
        pytest.skip("MPS not available")
    return "mps"


@pytest.fixture(scope="session")
def cpu_device() -> str:
    """Return 'cpu' device for reference implementations."""
    return "cpu"


# ==============================================================================
# Dtype fixtures
# ==============================================================================


@pytest.fixture(params=["float16", "bfloat16"])
def half_dtype(request: pytest.FixtureRequest) -> torch_types.dtype:
    """Parametrized fixture for half-precision dtypes (float16, bfloat16)."""
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[request.param]


@pytest.fixture
def float16_dtype() -> torch_types.dtype:
    """Return torch.float16 dtype."""
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")
    return torch.float16


@pytest.fixture
def bfloat16_dtype() -> torch_types.dtype:
    """Return torch.bfloat16 dtype."""
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")
    return torch.bfloat16


@pytest.fixture
def float32_dtype() -> torch_types.dtype:
    """Return torch.float32 dtype."""
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")
    return torch.float32


# ==============================================================================
# Random state fixtures
# ==============================================================================


@pytest.fixture
def seed() -> int:
    """Fixed seed for reproducible tests."""
    return 42


@pytest.fixture
def rng(seed: int) -> np.random.Generator:
    """NumPy random generator with fixed seed."""
    return np.random.default_rng(seed)


@pytest.fixture
def torch_rng(seed: int, device: str) -> None:
    """Set PyTorch random seed for reproducible tests.

    Sets seeds for CPU and MPS backends.
    """
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")
    torch.manual_seed(seed)
    if HAS_MPS:
        # MPS uses the same seed as CPU in PyTorch
        pass


# ==============================================================================
# Tensor factory fixtures
# ==============================================================================


@pytest.fixture
def make_tensor(device: str):
    """Factory fixture for creating tensors on the test device.

    Returns a function: (shape, dtype, requires_grad) -> Tensor
    """
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")

    def _make(
        shape: tuple[int, ...],
        dtype: torch_types.dtype = torch.float16,
        requires_grad: bool = False,
    ) -> torch_types.Tensor:
        return torch.randn(shape, dtype=dtype, device=device, requires_grad=requires_grad)

    return _make


@pytest.fixture
def make_weight_tensor(device: str, rng: np.random.Generator):
    """Factory fixture for creating weight tensors with controlled distribution.

    Returns a function: (shape, dtype, scale) -> Tensor

    Weights are initialized from N(0, scale) and converted to the target dtype.
    """
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")

    def _make(
        shape: tuple[int, ...],
        dtype: torch_types.dtype = torch.float16,
        scale: float = 0.02,
    ) -> torch_types.Tensor:
        data = rng.standard_normal(shape).astype(np.float32) * scale
        return torch.from_numpy(data).to(dtype=dtype, device=device)

    return _make


@pytest.fixture
def make_int_tensor(device: str, rng: np.random.Generator):
    """Factory fixture for creating integer tensors.

    Returns a function: (shape, low, high, dtype) -> Tensor
    """
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")

    def _make(
        shape: tuple[int, ...],
        low: int = 0,
        high: int = 16,
        dtype: torch_types.dtype = torch.int32,
    ) -> torch_types.Tensor:
        data = rng.integers(low, high, size=shape)
        return torch.from_numpy(data).to(dtype=dtype, device=device)

    return _make


# ==============================================================================
# GEMM dimension fixtures
# ==============================================================================


@pytest.fixture(
    params=[
        (1, 4096, 4096),  # Single token
        (8, 4096, 4096),  # Small batch
        (32, 4096, 4096),  # Medium batch
        (1, 4096, 14336),  # LLaMA-style up projection
        (1, 14336, 4096),  # LLaMA-style down projection
    ],
    ids=["1x4096x4096", "8x4096x4096", "32x4096x4096", "up_proj", "down_proj"],
)
def gemm_dims(request: pytest.FixtureRequest) -> tuple[int, int, int]:
    """Parametrized GEMM dimensions (M, N, K) for common inference shapes."""
    return request.param


@pytest.fixture(
    params=[
        (1, 4096, 4096),
        (8, 4096, 4096),
    ],
    ids=["1x4096", "8x4096"],
)
def small_gemm_dims(request: pytest.FixtureRequest) -> tuple[int, int, int]:
    """Smaller GEMM dimensions for fast smoke tests."""
    return request.param


# ==============================================================================
# MoE fixtures
# ==============================================================================


@pytest.fixture(
    params=[2, 4, 8],
    ids=["top2", "top4", "top8"],
)
def top_k(request: pytest.FixtureRequest) -> int:
    """Number of experts per token for MoE routing."""
    return request.param


@pytest.fixture(
    params=[8, 64, 128],
    ids=["8experts", "64experts", "128experts"],
)
def num_experts(request: pytest.FixtureRequest) -> int:
    """Total number of experts in MoE layer."""
    return request.param


# ==============================================================================
# Tolerance fixtures
# ==============================================================================


@pytest.fixture
def fp16_atol() -> float:
    """Absolute tolerance for FP16 comparisons."""
    return 1e-3


@pytest.fixture
def fp16_rtol() -> float:
    """Relative tolerance for FP16 comparisons."""
    return 1e-2


@pytest.fixture
def bf16_atol() -> float:
    """Absolute tolerance for BF16 comparisons."""
    return 5e-3


@pytest.fixture
def bf16_rtol() -> float:
    """Relative tolerance for BF16 comparisons."""
    return 2e-2


@pytest.fixture
def quant_atol() -> float:
    """Absolute tolerance for quantized (FP4/INT4) comparisons."""
    return 0.1


@pytest.fixture
def quant_rtol() -> float:
    """Relative tolerance for quantized (FP4/INT4) comparisons."""
    return 0.05


# ==============================================================================
# Utility functions (not fixtures, but commonly used in tests)
# ==============================================================================


def assert_close(
    actual: torch_types.Tensor,
    expected: torch_types.Tensor,
    atol: float = 1e-3,
    rtol: float = 1e-2,
    msg: str = "",
) -> None:
    """Assert two tensors are close within tolerance.

    Moves tensors to CPU for comparison if needed.
    """
    if torch is None:
        raise RuntimeError("PyTorch required for assert_close")

    actual_cpu = actual.detach().float().cpu()
    expected_cpu = expected.detach().float().cpu()

    if not torch.allclose(actual_cpu, expected_cpu, atol=atol, rtol=rtol):
        diff = (actual_cpu - expected_cpu).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        raise AssertionError(
            f"{msg}\nMax diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}, atol={atol}, rtol={rtol}"
        )


def relative_error(actual: torch_types.Tensor, expected: torch_types.Tensor) -> float:
    """Compute relative error between two tensors."""
    if torch is None:
        raise RuntimeError("PyTorch required for relative_error")

    actual_cpu = actual.detach().float().cpu()
    expected_cpu = expected.detach().float().cpu()

    diff = (actual_cpu - expected_cpu).abs()
    scale = expected_cpu.abs().clamp(min=1e-8)
    return (diff / scale).mean().item()
