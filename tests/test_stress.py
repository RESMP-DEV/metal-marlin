"""Stress tests for Metal Marlin FP4 GEMM stability under heavy load."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

# Add metal_marlin package to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import mlx.core as mx
from metal_marlin import pack_fp4_weights
from metal_marlin.metal_marlin import quantized_linear as marlin_gemm_fp4
from metal_marlin.quantize import pack_fp4_weights as pack_fp4_weights_padded


def create_quantized_weights(
    K: int,
    N: int,
    group_size: int = 32,
    *,
    allow_padding: bool = False,
) -> tuple[mx.array, mx.array]:
    """Create quantized weight matrices for testing."""
    weights = mx.random.normal((K, N)) * 2.0

    if allow_padding or K % group_size != 0 or N % 8 != 0:
        packed, scales, _meta = pack_fp4_weights_padded(weights, group_size=group_size, pad_k=True)
        return packed, scales

    packed, scales = pack_fp4_weights(weights, group_size=group_size)
    return packed, scales


class TestStress:
    def test_repeated_calls(self):
        """Many repeated GEMM calls."""
        A = mx.random.normal((32, 4096))
        B, scales = create_quantized_weights(4096, 4096)

        for _ in range(1000):
            _result = marlin_gemm_fp4(A, B, scales)
            mx.synchronize()

    def test_varying_sizes(self):
        """Rapidly varying problem sizes."""
        sizes = [
            (1, 4096, 4096),
            (32, 4096, 4096),
            (1, 11008, 4096),
            (64, 4096, 4096),
            (1, 4096, 11008),
        ]

        for _ in range(100):
            for M, N, K in sizes:
                A = mx.random.normal((M, K))
                B, scales = create_quantized_weights(K, N)
                _result = marlin_gemm_fp4(A, B, scales)
                mx.synchronize()

    def test_memory_pressure(self):
        """Run under memory pressure."""
        pressure = [mx.random.normal((1024, 1024)) for _ in range(100)]

        A = mx.random.normal((32, 4096))
        B, scales = create_quantized_weights(4096, 4096)

        for _ in range(100):
            _result = marlin_gemm_fp4(A, B, scales)
            mx.synchronize()

        del pressure

    @pytest.mark.slow
    def test_long_running(self):
        """Run for extended period."""
        A = mx.random.normal((32, 4096))
        B, scales = create_quantized_weights(4096, 4096)

        start = time.time()
        count = 0

        while time.time() - start < 60:
            _result = marlin_gemm_fp4(A, B, scales)
            mx.synchronize()
            count += 1

        print(f"Completed {count} iterations in 60 seconds")
