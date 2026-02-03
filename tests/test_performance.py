"""End-to-end performance regression tests for GLM-4.7-Flash 3bpw.

These tests enforce performance thresholds to prevent regressions:
- Decode latency: < 1000ms per token (interim target)
- Memory usage: < 4GB for GLM-4.7-Flash 3bpw

Tests are marked as slow and intended to run in CI nightly.

Environment variables:
- PERF_MODEL_PATH: Path to GLM-4.7-Flash-3bpw model (default: auto-detect)
- PERF_LATENCY_THRESHOLD_MS: Override decode latency threshold (default: 1000)
- PERF_MEMORY_THRESHOLD_GB: Override memory threshold (default: 4.0)
- PERF_NUM_TOKENS: Number of tokens to generate (default: 10)

Usage:
    pytest tests/test_performance.py -v --run-slow
    PERF_MODEL_PATH=/path/to/model pytest tests/test_performance.py -v --run-slow
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch

if TYPE_CHECKING:
    import torch as torch_types


DECODE_LATENCY_THRESHOLD_MS = float(os.environ.get("PERF_LATENCY_THRESHOLD_MS", "1000"))
MEMORY_THRESHOLD_GB = float(os.environ.get("PERF_MEMORY_THRESHOLD_GB", "4.0"))
NUM_TOKENS = int(os.environ.get("PERF_NUM_TOKENS", "10"))


@dataclass
class PerfResult:
    metric_name: str
    value: float
    unit: str
    threshold: float
    passed: bool
    details: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.metric_name}: {self.value:.4f} {self.unit} "
            f"(threshold: {self.threshold:.4f} {self.unit})"
        )


def _mps_memory_gb() -> float:
    if not HAS_TORCH or torch is None or not HAS_MPS:
        return 0.0

    try:
        if hasattr(torch.mps, "current_allocated_memory"):
            bytes_used = torch.mps.current_allocated_memory()
        elif hasattr(torch.mps, "driver_allocated_memory"):
            bytes_used = torch.mps.driver_allocated_memory()
        else:
            return 0.0
        return bytes_used / (1024**3)
    except Exception:
        return 0.0


def _sync_gpu() -> None:
    if HAS_TORCH and torch is not None and HAS_MPS:
        torch.mps.synchronize()


def _clear_memory() -> None:
    gc.collect()
    if HAS_TORCH and torch is not None and HAS_MPS:
        torch.mps.synchronize()
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


class LatencyTimer:
    def __init__(self, sync_gpu: bool = True) -> None:
        self.sync_gpu = sync_gpu
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def __enter__(self) -> LatencyTimer:
        if self.sync_gpu:
            _sync_gpu()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        if self.sync_gpu:
            _sync_gpu()
        self.end_time = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class MemoryTracker:
    def __init__(self) -> None:
        self.baseline_gb: float = 0.0
        self.peak_gb: float = 0.0

    def __enter__(self) -> MemoryTracker:
        _clear_memory()
        self.baseline_gb = _mps_memory_gb()
        self.peak_gb = self.baseline_gb
        return self

    def __exit__(self, *args) -> None:
        _sync_gpu()
        final = _mps_memory_gb()
        self.peak_gb = max(self.peak_gb, final)

    def sample(self) -> None:
        current = _mps_memory_gb()
        self.peak_gb = max(self.peak_gb, current)

    @property
    def used_gb(self) -> float:
        return max(0.0, self.peak_gb - self.baseline_gb)


@pytest.fixture
def model_path() -> str:
    """Get model path from environment or use default."""
    from pathlib import Path

    env_path = os.environ.get("PERF_MODEL_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"PERF_MODEL_PATH does not exist: {env_path}")

    # Default search paths: check project-local models, server convention, and user home
    default_paths = [
        Path(__file__).parent.parent / "models" / "GLM-4.7-Flash-Trellis-3bpw",
        Path("/models/GLM-4.7-Flash-Trellis-3bpw"),  # Server convention for model storage
        Path.home() / "models" / "GLM-4.7-Flash-Trellis-3bpw",
    ]

    for path in default_paths:
        if path.exists():
            return str(path)

    pytest.skip(
        "GLM-4.7-Flash-3bpw model not found. Set PERF_MODEL_PATH or place model in /models/ or ~/models/"
    )


@pytest.fixture
def prompt_text() -> str:
    """Sample prompt for generation."""
    return "The future of artificial intelligence is"


@pytest.mark.slow
@pytest.mark.skipif(not HAS_MPS, reason="Requires MPS (Apple Silicon)")
@pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
class TestGLM47FlashPerformance:
    """End-to-end performance tests for GLM-4.7-Flash 3bpw."""

    def test_decode_latency_per_token(self, model_path: str, prompt_text: str) -> None:
        """Measure single-token decode latency.

        Verifies that decoding a single token takes less than the threshold.
        This is critical for interactive applications.
        """
        if not torch:
            pytest.skip("PyTorch not available")

        from transformers import AutoTokenizer

        from metal_marlin.inference import MetalInferenceEngine

        print(f"\nLoading model from: {model_path}")
        engine = MetalInferenceEngine(model_path, device="mps")

        tokenizer = AutoTokenizer.from_pretrained(engine.quantized.config.get("base_model_id"))
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to("mps")

        _clear_memory()
        _sync_gpu()

        latencies_ms = []

        print(f"\nGenerating {NUM_TOKENS} tokens...")
        for i in range(NUM_TOKENS):
            with LatencyTimer() as timer:
                logits = engine.model(input_ids)

            token_id = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, token_id], dim=1)

            latency_ms = timer.elapsed_ms
            latencies_ms.append(latency_ms)
            print(f"  Token {i + 1}: {latency_ms:.2f}ms")

        avg_latency_ms = sum(latencies_ms) / len(latencies_ms)
        max_latency_ms = max(latencies_ms)
        min_latency_ms = min(latencies_ms)

        result = PerfResult(
            metric_name="decode_latency",
            value=avg_latency_ms,
            unit="ms/token",
            threshold=DECODE_LATENCY_THRESHOLD_MS,
            passed=avg_latency_ms < DECODE_LATENCY_THRESHOLD_MS,
            details=(
                f"Generated {NUM_TOKENS} tokens. "
                f"Avg: {avg_latency_ms:.2f}ms, "
                f"Min: {min_latency_ms:.2f}ms, "
                f"Max: {max_latency_ms:.2f}ms"
            ),
        )

        print(f"\n{result}")
        print(f"  {result.details}")

        assert result.passed, (
            f"REGRESSION: Decode latency {result.value:.2f}ms/token exceeds "
            f"threshold {result.threshold:.2f}ms/token"
        )

    def test_memory_usage_during_generation(self, model_path: str, prompt_text: str) -> None:
        """Measure peak memory usage during generation.

        Verifies that peak memory stays below the threshold.
        Critical for running on devices with limited memory.
        """
        if not torch:
            pytest.skip("PyTorch not available")

        from transformers import AutoTokenizer

        from metal_marlin.inference import MetalInferenceEngine

        print(f"\nLoading model from: {model_path}")
        engine = MetalInferenceEngine(model_path, device="mps")

        tokenizer = AutoTokenizer.from_pretrained(engine.quantized.config.get("base_model_id"))
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to("mps")

        with MemoryTracker() as tracker:
            print(f"\nGenerating {NUM_TOKENS} tokens with memory tracking...")
            for i in range(NUM_TOKENS):
                logits = engine.model(input_ids)
                token_id = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, token_id], dim=1)

                tracker.sample()
                print(f"  Token {i + 1}: {tracker.used_gb:.2f}GB")

        result = PerfResult(
            metric_name="peak_memory",
            value=tracker.used_gb,
            unit="GB",
            threshold=MEMORY_THRESHOLD_GB,
            passed=tracker.used_gb < MEMORY_THRESHOLD_GB,
            details=f"Peak memory during {NUM_TOKENS} token generation",
        )

        print(f"\n{result}")
        print(f"  {result.details}")

        assert result.passed, (
            f"REGRESSION: Peak memory {result.value:.2f}GB exceeds "
            f"threshold {result.threshold:.2f}GB"
        )

    def test_latency_and_memory_combined(self, model_path: str, prompt_text: str) -> None:
        """Combined test measuring both latency and memory.

        Verifies that both performance metrics stay within thresholds.
        """
        if not torch:
            pytest.skip("PyTorch not available")

        from transformers import AutoTokenizer

        from metal_marlin.inference import MetalInferenceEngine

        print(f"\nCombined performance test for {NUM_TOKENS} tokens")
        print(f"Latency threshold: {DECODE_LATENCY_THRESHOLD_MS}ms/token")
        print(f"Memory threshold: {MEMORY_THRESHOLD_GB}GB")

        engine = MetalInferenceEngine(model_path, device="mps")

        tokenizer = AutoTokenizer.from_pretrained(engine.quantized.config.get("base_model_id"))
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to("mps")

        latencies_ms = []

        with MemoryTracker() as tracker:
            for i in range(NUM_TOKENS):
                with LatencyTimer() as timer:
                    logits = engine.model(input_ids)

                token_id = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, token_id], dim=1)

                latencies_ms.append(timer.elapsed_ms)
                tracker.sample()

        avg_latency_ms = sum(latencies_ms) / len(latencies_ms)

        latency_result = PerfResult(
            metric_name="decode_latency",
            value=avg_latency_ms,
            unit="ms/token",
            threshold=DECODE_LATENCY_THRESHOLD_MS,
            passed=avg_latency_ms < DECODE_LATENCY_THRESHOLD_MS,
        )

        memory_result = PerfResult(
            metric_name="peak_memory",
            value=tracker.used_gb,
            unit="GB",
            threshold=MEMORY_THRESHOLD_GB,
            passed=tracker.used_gb < MEMORY_THRESHOLD_GB,
        )

        print(f"\n{latency_result}")
        print(f"{memory_result}")

        both_passed = latency_result.passed and memory_result.passed
        assert both_passed, (
            f"REGRESSION: "
            f"Latency {latency_result.value:.2f}ms/token (threshold: {latency_result.threshold:.2f}ms), "
            f"Memory {memory_result.value:.2f}GB (threshold: {memory_result.threshold:.2f}GB)"
        )


class TestPerformanceConfig:
    """Validate performance test configuration."""

    @pytest.mark.smoke
    def test_thresholds_are_reasonable(self) -> None:
        """Verify that configured thresholds are reasonable."""
        print("\nPerformance threshold configuration:")
        print(f"  Decode latency: {DECODE_LATENCY_THRESHOLD_MS}ms/token")
        print(f"  Memory: {MEMORY_THRESHOLD_GB}GB")

        assert DECODE_LATENCY_THRESHOLD_MS > 0, "Latency threshold must be positive"
        assert MEMORY_THRESHOLD_GB > 0, "Memory threshold must be positive"
        assert DECODE_LATENCY_THRESHOLD_MS < 5000, "Latency threshold seems too high"
        assert MEMORY_THRESHOLD_GB < 16, "Memory threshold seems too high"

    @pytest.mark.smoke
    def test_memory_tracking_infrastructure(self) -> None:
        """Verify memory measurement infrastructure works."""
        if not HAS_TORCH or not HAS_MPS:
            pytest.skip("MPS not available")

        with MemoryTracker() as tracker:
            pass

        assert tracker.used_gb >= 0.0, "Memory tracking returned negative value"
        print(f"\nMemory tracking OK: baseline={tracker.baseline_gb:.3f}GB")

    @pytest.mark.smoke
    def test_latency_measurement_infrastructure(self) -> None:
        """Verify latency measurement infrastructure works."""
        with LatencyTimer(sync_gpu=False) as timer:
            time.sleep(0.01)

        elapsed = timer.elapsed_ms
        assert 5.0 < elapsed < 50.0, f"Timer broken: expected ~10ms, got {elapsed:.2f}ms"
        print(f"\nLatency tracking OK: measured {elapsed:.2f}ms for 10ms sleep")
