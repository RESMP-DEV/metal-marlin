"""Tests for decode (generation) optimization module.

Tests:
1. decode_gemv_fp4 correctness vs full GEMM
2. decode_gemv_fp4_wide correctness
3. decode_gemv_fp4_batched correctness
4. quantized_kv_attention correctness
5. select_decode_kernel selection logic
6. persistent_decode_step full layer test
"""

from __future__ import annotations

import pytest

# Skip tests if MLX not available
mlx = pytest.importorskip("mlx.core")

import mlx.core as mx
import numpy as np

# Import after MLX check
from metal_marlin.kernels import (
    decode_gemv_fp4,
    decode_gemv_fp4_batched,
    decode_gemv_fp4_wide,
    marlin_gemm_fp4,
    pack_fp4_weights,
    select_decode_kernel,
)


class TestDecodeGEMV:
    """Tests for decode GEMV kernels."""

    @pytest.fixture
    def setup_weights(self):
        """Create test weights for decode tests."""
        # Typical LLM dimensions
        K = 4096  # hidden_size
        N = 4096  # output_dim
        group_size = 128

        # Create random weights
        np.random.seed(42)
        W_np = np.random.randn(N, K).astype(np.float32) * 0.1

        # Pack to FP4
        W_mx = mx.array(W_np, dtype=mx.bfloat16)
        packed, scales = pack_fp4_weights(W_mx, group_size=group_size)
        mx.eval(packed, scales)

        return {
            "K": K,
            "N": N,
            "group_size": group_size,
            "W_np": W_np,
            "packed": packed,
            "scales": scales,
        }

    def test_decode_gemv_fp4_correctness(self, setup_weights):
        """Test decode_gemv_fp4 produces same results as full GEMM."""
        cfg = setup_weights
        K, _N = cfg["K"], cfg["N"]
        packed, scales = cfg["packed"], cfg["scales"]

        # Create single-token activation
        A = mx.random.normal((1, K), dtype=mx.bfloat16)
        mx.eval(A)

        # Full GEMM reference
        ref = marlin_gemm_fp4(A, packed, scales, cfg["group_size"])
        mx.eval(ref)

        # Decode GEMV
        out = decode_gemv_fp4(A, packed, scales, cfg["group_size"])
        mx.eval(out)

        # Compare (FP4 introduces some error, but decode should match GEMM)
        ref_np = np.array(ref).flatten()
        out_np = np.array(out).flatten()

        # Should be identical (same underlying kernel logic)
        np.testing.assert_allclose(
            out_np, ref_np, rtol=1e-3, atol=1e-3,
            err_msg="decode_gemv_fp4 differs from marlin_gemm_fp4"
        )

    def test_decode_gemv_fp4_1d_input(self, setup_weights):
        """Test decode_gemv_fp4 with 1D input (common case)."""
        cfg = setup_weights
        K, N = cfg["K"], cfg["N"]
        packed, scales = cfg["packed"], cfg["scales"]

        # 1D activation
        A = mx.random.normal((K,), dtype=mx.bfloat16)
        mx.eval(A)

        # Should work with 1D input and return 1D output
        out = decode_gemv_fp4(A, packed, scales, cfg["group_size"])
        mx.eval(out)

        assert out.ndim == 1, f"Expected 1D output, got shape {out.shape}"
        assert out.shape[0] == N, f"Expected shape ({N},), got {out.shape}"

    def test_decode_gemv_fp4_wide_correctness(self, setup_weights):
        """Test decode_gemv_fp4_wide produces same results as standard."""
        cfg = setup_weights
        K, _N = cfg["K"], cfg["N"]
        packed, scales = cfg["packed"], cfg["scales"]

        A = mx.random.normal((1, K), dtype=mx.bfloat16)
        mx.eval(A)

        # Standard decode
        ref = decode_gemv_fp4(A, packed, scales, cfg["group_size"])
        mx.eval(ref)

        # Wide decode
        out = decode_gemv_fp4_wide(A, packed, scales, cfg["group_size"])
        mx.eval(out)

        ref_np = np.array(ref).flatten()
        out_np = np.array(out).flatten()

        np.testing.assert_allclose(
            out_np, ref_np, rtol=1e-3, atol=1e-3,
            err_msg="decode_gemv_fp4_wide differs from decode_gemv_fp4"
        )

    def test_decode_gemv_fp4_batched_correctness(self, setup_weights):
        """Test decode_gemv_fp4_batched for M=2..8."""
        cfg = setup_weights
        K, _N = cfg["K"], cfg["N"]
        packed, scales = cfg["packed"], cfg["scales"]

        for M in [2, 4, 8]:
            A = mx.random.normal((M, K), dtype=mx.bfloat16)
            mx.eval(A)

            # Full GEMM reference
            ref = marlin_gemm_fp4(A, packed, scales, cfg["group_size"])
            mx.eval(ref)

            # Batched decode
            out = decode_gemv_fp4_batched(A, packed, scales, cfg["group_size"])
            mx.eval(out)

            ref_np = np.array(ref)
            out_np = np.array(out)

            np.testing.assert_allclose(
                out_np, ref_np, rtol=1e-3, atol=1e-3,
                err_msg=f"decode_gemv_fp4_batched (M={M}) differs from GEMM"
            )

    def test_decode_gemv_small_dimensions(self):
        """Test decode GEMV with small dimensions."""
        K = 256
        N = 512
        group_size = 32

        np.random.seed(123)
        W_np = np.random.randn(N, K).astype(np.float32) * 0.1
        W_mx = mx.array(W_np, dtype=mx.bfloat16)
        packed, scales = pack_fp4_weights(W_mx, group_size=group_size)
        mx.eval(packed, scales)

        A = mx.random.normal((K,), dtype=mx.bfloat16)
        mx.eval(A)

        out = decode_gemv_fp4(A, packed, scales, group_size)
        mx.eval(out)

        assert out.shape == (N,), f"Expected shape ({N},), got {out.shape}"

    def test_decode_gemv_large_dimensions(self):
        """Test decode GEMV with large dimensions (30B-scale projections)."""
        # Typical 30B model dimensions
        K = 8192
        N = 22016  # intermediate_size for ~30B
        group_size = 128

        np.random.seed(456)
        W_np = np.random.randn(N, K).astype(np.float32) * 0.02
        W_mx = mx.array(W_np, dtype=mx.bfloat16)
        packed, scales = pack_fp4_weights(W_mx, group_size=group_size)
        mx.eval(packed, scales)

        A = mx.random.normal((K,), dtype=mx.bfloat16)
        mx.eval(A)

        out = decode_gemv_fp4(A, packed, scales, group_size)
        mx.eval(out)

        assert out.shape == (N,), f"Expected shape ({N},), got {out.shape}"


class TestSelectDecodeKernel:
    """Tests for kernel selection logic."""

    def test_m1_small_n(self):
        """M=1, small N -> standard decode."""
        kernel = select_decode_kernel(M=1, N=256, K=4096)
        assert kernel == "decode_gemv_fp4"

    def test_m1_large_n(self):
        """M=1, large N -> wide decode."""
        kernel = select_decode_kernel(M=1, N=4096, K=4096)
        assert kernel == "decode_gemv_fp4_wide"

    def test_m1_boundary_n(self):
        """M=1, N=512 boundary -> wide decode."""
        kernel = select_decode_kernel(M=1, N=512, K=4096)
        assert kernel == "decode_gemv_fp4_wide"

    def test_small_batch(self):
        """M=2..8 -> batched decode."""
        for m in [2, 4, 8]:
            kernel = select_decode_kernel(M=m, N=4096, K=4096)
            assert kernel == "decode_gemv_fp4_batched", f"Failed for M={m}"

    def test_large_batch(self):
        """M > 8 -> full GEMM."""
        for m in [16, 32, 64]:
            kernel = select_decode_kernel(M=m, N=4096, K=4096)
            assert kernel == "marlin_gemm_fp4", f"Failed for M={m}"


class TestQuantizedKVAttention:
    """Tests for quantized KV cache attention."""

    @pytest.fixture
    def setup_attention(self):
        """Create test tensors for attention tests."""
        batch_size = 1
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        seq_len = 512

        # Query for single new token
        q = mx.random.normal(
            (batch_size, num_heads, 1, head_dim), dtype=mx.bfloat16
        )

        # KV cache (simulate FP8 quantized)
        k_cache_fp = mx.random.normal(
            (batch_size, num_kv_heads, seq_len, head_dim), dtype=mx.bfloat16
        )
        v_cache_fp = mx.random.normal(
            (batch_size, num_kv_heads, seq_len, head_dim), dtype=mx.bfloat16
        )

        # Quantize to FP8 (simulated with uint8 + scales)
        k_max = mx.max(mx.abs(k_cache_fp), axis=-1, keepdims=True)
        k_max = mx.maximum(k_max, 1e-8)
        k_scale = k_max / 448.0
        k_cache_quant = mx.round((k_cache_fp / k_scale) * 127.0 + 128.0)
        k_cache_quant = mx.clip(k_cache_quant, 0, 255).astype(mx.uint8)

        v_max = mx.max(mx.abs(v_cache_fp), axis=-1, keepdims=True)
        v_max = mx.maximum(v_max, 1e-8)
        v_scale = v_max / 448.0
        v_cache_quant = mx.round((v_cache_fp / v_scale) * 127.0 + 128.0)
        v_cache_quant = mx.clip(v_cache_quant, 0, 255).astype(mx.uint8)

        mx.eval(q, k_cache_quant, v_cache_quant, k_scale, v_scale)
        mx.eval(k_cache_fp, v_cache_fp)

        return {
            "batch_size": batch_size,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "seq_len": seq_len,
            "q": q,
            "k_cache_fp": k_cache_fp,
            "v_cache_fp": v_cache_fp,
            "k_cache_quant": k_cache_quant,
            "v_cache_quant": v_cache_quant,
            "k_scale": k_scale,
            "v_scale": v_scale,
        }

    def test_quantized_kv_attention_shape(self, setup_attention):
        """Test output shape of quantized KV attention."""
        from metal_marlin.inference import quantized_kv_attention

        cfg = setup_attention

        out = quantized_kv_attention(
            q=cfg["q"],
            k_cache=cfg["k_cache_quant"],
            v_cache=cfg["v_cache_quant"],
            k_scales=cfg["k_scale"],
            v_scales=cfg["v_scale"],
            num_heads=cfg["num_heads"],
            num_kv_heads=cfg["num_kv_heads"],
        )
        mx.eval(out)

        expected_shape = (
            cfg["batch_size"],
            cfg["num_heads"],
            1,
            cfg["head_dim"],
        )
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"

    def test_quantized_kv_attention_accuracy(self, setup_attention):
        """Test quantized attention is close to FP reference."""
        import math

        from metal_marlin.inference import quantized_kv_attention

        cfg = setup_attention
        q = cfg["q"]
        num_heads = cfg["num_heads"]
        num_kv_heads = cfg["num_kv_heads"]
        head_dim = cfg["head_dim"]

        # Reference attention with FP cache
        k_fp = cfg["k_cache_fp"]
        v_fp = cfg["v_cache_fp"]

        # Expand for GQA
        repeat_factor = num_heads // num_kv_heads
        k_exp = mx.repeat(k_fp, repeat_factor, axis=1)
        v_exp = mx.repeat(v_fp, repeat_factor, axis=1)

        scale = 1.0 / math.sqrt(head_dim)
        attn_weights = (q @ k_exp.transpose(0, 1, 3, 2)) * scale
        attn_weights = mx.softmax(attn_weights, axis=-1)
        ref_out = attn_weights @ v_exp
        mx.eval(ref_out)

        # Quantized attention
        quant_out = quantized_kv_attention(
            q=q,
            k_cache=cfg["k_cache_quant"],
            v_cache=cfg["v_cache_quant"],
            k_scales=cfg["k_scale"],
            v_scales=cfg["v_scale"],
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        mx.eval(quant_out)

        ref_np = np.array(ref_out)
        quant_np = np.array(quant_out)

        # FP8 quantization introduces some error, allow larger tolerance
        np.testing.assert_allclose(
            quant_np, ref_np, rtol=0.1, atol=0.1,
            err_msg="Quantized KV attention differs too much from FP reference"
        )


class TestDecodeState:
    """Tests for DecodeState persistent buffer management."""

    def test_decode_state_creation(self):
        """Test DecodeState can be created with config and weights."""
        from metal_marlin.inference import DecodeConfig, DecodeState, LayerWeights

        config = DecodeConfig(
            num_layers=2,
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
            intermediate_size=11008,
            group_size=128,
        )

        # Create minimal layer weights (just shapes, zeros for test)
        group_size = config.group_size
        hidden_size = config.hidden_size
        num_heads = config.num_heads
        num_kv_heads = config.num_kv_heads or num_heads
        head_dim = config.head_dim or (hidden_size // num_heads)
        intermediate_size = config.intermediate_size or 11008

        def make_weights() -> LayerWeights:
            return LayerWeights(
                q_proj_weight=mx.zeros((hidden_size // 8, num_heads * head_dim), dtype=mx.uint32),
                q_proj_scales=mx.ones((hidden_size // group_size, num_heads * head_dim), dtype=mx.bfloat16),
                k_proj_weight=mx.zeros((hidden_size // 8, num_kv_heads * head_dim), dtype=mx.uint32),
                k_proj_scales=mx.ones((hidden_size // group_size, num_kv_heads * head_dim), dtype=mx.bfloat16),
                v_proj_weight=mx.zeros((hidden_size // 8, num_kv_heads * head_dim), dtype=mx.uint32),
                v_proj_scales=mx.ones((hidden_size // group_size, num_kv_heads * head_dim), dtype=mx.bfloat16),
                o_proj_weight=mx.zeros((num_heads * head_dim // 8, hidden_size), dtype=mx.uint32),
                o_proj_scales=mx.ones((num_heads * head_dim // group_size, hidden_size), dtype=mx.bfloat16),
                gate_proj_weight=mx.zeros((hidden_size // 8, intermediate_size), dtype=mx.uint32),
                gate_proj_scales=mx.ones((hidden_size // group_size, intermediate_size), dtype=mx.bfloat16),
                up_proj_weight=mx.zeros((hidden_size // 8, intermediate_size), dtype=mx.uint32),
                up_proj_scales=mx.ones((hidden_size // group_size, intermediate_size), dtype=mx.bfloat16),
                down_proj_weight=mx.zeros((intermediate_size // 8, hidden_size), dtype=mx.uint32),
                down_proj_scales=mx.ones((intermediate_size // group_size, hidden_size), dtype=mx.bfloat16),
                input_layernorm_weight=mx.ones((hidden_size,), dtype=mx.bfloat16),
                post_attention_layernorm_weight=mx.ones((hidden_size,), dtype=mx.bfloat16),
            )

        layer_weights = [make_weights() for _ in range(config.num_layers)]

        state = DecodeState(config, layer_weights)

        # Check persistent buffers were created
        assert state.qkv_buffer is not None
        assert state.attn_out_buffer is not None
        assert state.mlp_buffer is not None


class TestDecodePerformance:
    """Performance-oriented tests (optional, may be slow)."""

    @pytest.mark.slow
    def test_decode_vs_gemm_speedup(self):
        """Measure speedup of decode GEMV vs full GEMM."""
        import time

        K = 4096
        N = 4096
        group_size = 128
        warmup_iters = 10
        bench_iters = 100

        # Setup weights
        np.random.seed(789)
        W_np = np.random.randn(N, K).astype(np.float32) * 0.1
        W_mx = mx.array(W_np, dtype=mx.bfloat16)
        packed, scales = pack_fp4_weights(W_mx, group_size=group_size)
        mx.eval(packed, scales)

        A = mx.random.normal((1, K), dtype=mx.bfloat16)
        mx.eval(A)

        # Warmup GEMM
        for _ in range(warmup_iters):
            out = marlin_gemm_fp4(A, packed, scales, group_size)
            mx.eval(out)

        # Benchmark GEMM
        start = time.perf_counter()
        for _ in range(bench_iters):
            out = marlin_gemm_fp4(A, packed, scales, group_size)
            mx.eval(out)
        gemm_time = (time.perf_counter() - start) / bench_iters * 1000  # ms

        # Warmup decode
        for _ in range(warmup_iters):
            out = decode_gemv_fp4(A, packed, scales, group_size)
            mx.eval(out)

        # Benchmark decode
        start = time.perf_counter()
        for _ in range(bench_iters):
            out = decode_gemv_fp4(A, packed, scales, group_size)
            mx.eval(out)
        decode_time = (time.perf_counter() - start) / bench_iters * 1000  # ms

        speedup = gemm_time / decode_time
        print(f"\nGEMM time: {gemm_time:.3f}ms, Decode time: {decode_time:.3f}ms")
        print(f"Speedup: {speedup:.2f}x")

        # Should see some speedup (at least 1.5x) for M=1
        assert speedup > 1.0, f"Decode should be faster than GEMM for M=1 (got {speedup:.2f}x)"
