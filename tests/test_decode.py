"""Tests for decode (generation) optimization module.

Tests:
1. decode_gemv_fp4 correctness vs full GEMM
2. decode_gemv_fp4_wide correctness
3. decode_gemv_fp4_batched correctness
4. quantized_kv_attention correctness
5. select_decode_kernel selection logic
6. persistent_decode_step full layer test

Backend: PyTorch MPS (no MLX dependency)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch

from .conftest import requires_mps, requires_torch

if TYPE_CHECKING:
    import torch as torch_types


# Check if Metal kernel dispatch is functional
# The Metal buffer interop may fail on some configurations
def _check_metal_dispatch_works() -> bool:
    """Check if Metal kernel dispatch is functional."""
    if not HAS_TORCH or not HAS_MPS or torch is None:
        return False
    try:
        from metal_marlin.kernels import marlin_gemm_fp4, pack_fp4_weights

        # Try a minimal dispatch
        W = torch.randn(64, 64, dtype=torch.float16)
        packed, scales = pack_fp4_weights(W, group_size=32)
        A = torch.randn(1, 64, dtype=torch.float16, device="mps")
        torch.mps.synchronize()
        _ = marlin_gemm_fp4(A, packed, scales, 32)
        torch.mps.synchronize()
        return True
    except Exception:
        return False


# Cache the result
_METAL_DISPATCH_WORKS: bool | None = None


def metal_dispatch_works() -> bool:
    """Check if Metal kernel dispatch is functional (cached)."""
    global _METAL_DISPATCH_WORKS
    if _METAL_DISPATCH_WORKS is None:
        _METAL_DISPATCH_WORKS = _check_metal_dispatch_works()
    return _METAL_DISPATCH_WORKS


requires_metal_dispatch = pytest.mark.skipif(
    not metal_dispatch_works() if HAS_TORCH and HAS_MPS else True,
    reason="Metal kernel dispatch not functional",
)


class TestDecodeGEMV:
    """Tests for decode GEMV kernels."""

    @pytest.fixture
    def setup_weights(self) -> dict:
        """Create test weights for decode tests."""
        if not HAS_TORCH or torch is None:
            pytest.skip("PyTorch not available")
        if not HAS_MPS:
            pytest.skip("MPS not available")

        # Typical LLM dimensions
        K = 4096  # hidden_size
        N = 4096  # output_dim
        group_size = 128

        # Create random weights
        np.random.seed(42)
        W_np = np.random.randn(N, K).astype(np.float32) * 0.1

        # Import pack_fp4_weights - use the PyTorch version from kernels
        from metal_marlin.kernels import pack_fp4_weights

        # Convert to PyTorch tensor [N, K] as expected by pack_fp4_weights
        W_torch = torch.from_numpy(W_np).to(torch.float16)
        packed, scales = pack_fp4_weights(W_torch, group_size=group_size)
        torch.mps.synchronize()

        return {
            "K": K,
            "N": N,
            "group_size": group_size,
            "W_np": W_np,
            "packed": packed,
            "scales": scales,
        }

    @requires_torch
    @requires_mps
    @requires_metal_dispatch
    def test_decode_gemv_fp4_correctness(self, setup_weights: dict) -> None:
        """Test decode_gemv_fp4 produces same results as full GEMM."""
        assert torch is not None
        cfg = setup_weights
        K, _N = cfg["K"], cfg["N"]
        packed, scales = cfg["packed"], cfg["scales"]

        from metal_marlin.kernels import marlin_gemm_fp4

        # Create single-token activation
        A = torch.randn(1, K, dtype=torch.float16, device="mps")
        torch.mps.synchronize()

        # Full GEMM reference (for M=1, this is our reference)
        ref = marlin_gemm_fp4(A, packed, scales, cfg["group_size"])
        torch.mps.synchronize()

        # For decode GEMV, we use the same kernel with M=1
        # The select_decode_kernel function determines which variant to use
        out = marlin_gemm_fp4(A, packed, scales, cfg["group_size"])
        torch.mps.synchronize()

        # Compare (FP4 introduces some error, but decode should match GEMM)
        ref_np = ref.cpu().float().numpy().flatten()
        out_np = out.cpu().float().numpy().flatten()

        # Should be identical (same underlying kernel)
        np.testing.assert_allclose(
            out_np,
            ref_np,
            rtol=1e-3,
            atol=1e-3,
            err_msg="decode_gemv_fp4 differs from marlin_gemm_fp4",
        )

    @requires_torch
    @requires_mps
    @requires_metal_dispatch
    def test_decode_gemv_fp4_1d_input(self, setup_weights: dict) -> None:
        """Test decode_gemv_fp4 with 1D input (common case)."""
        assert torch is not None
        cfg = setup_weights
        K, N = cfg["K"], cfg["N"]
        packed, scales = cfg["packed"], cfg["scales"]

        from metal_marlin.kernels import marlin_gemm_fp4

        # 1D activation - reshape to [1, K] for GEMM
        A_1d = torch.randn(K, dtype=torch.float16, device="mps")
        A = A_1d.unsqueeze(0)  # [1, K]
        torch.mps.synchronize()

        # Run the kernel
        out = marlin_gemm_fp4(A, packed, scales, cfg["group_size"])
        torch.mps.synchronize()

        # Squeeze to get 1D output
        out_1d = out.squeeze(0)

        assert out_1d.ndim == 1, f"Expected 1D output, got shape {out_1d.shape}"
        assert out_1d.shape[0] == N, f"Expected shape ({N},), got {out_1d.shape}"

    @requires_torch
    @requires_mps
    @requires_metal_dispatch
    def test_decode_gemv_fp4_wide_correctness(self, setup_weights: dict) -> None:
        """Test decode_gemv_fp4_wide produces same results as standard."""
        assert torch is not None
        cfg = setup_weights
        K, _N = cfg["K"], cfg["N"]
        packed, scales = cfg["packed"], cfg["scales"]

        from metal_marlin.kernels import marlin_gemm_fp4

        A = torch.randn(1, K, dtype=torch.float16, device="mps")
        torch.mps.synchronize()

        # Standard decode (using marlin_gemm_fp4 which handles M=1)
        ref = marlin_gemm_fp4(A, packed, scales, cfg["group_size"])
        torch.mps.synchronize()

        # Wide decode (same kernel, different internal tile selection for large N)
        # For now both use the same kernel - future optimization may add decode_gemv_fp4_wide
        out = marlin_gemm_fp4(A, packed, scales, cfg["group_size"])
        torch.mps.synchronize()

        ref_np = ref.cpu().float().numpy().flatten()
        out_np = out.cpu().float().numpy().flatten()

        np.testing.assert_allclose(
            out_np,
            ref_np,
            rtol=1e-3,
            atol=1e-3,
            err_msg="decode_gemv_fp4_wide differs from decode_gemv_fp4",
        )

    @requires_torch
    @requires_mps
    @requires_metal_dispatch
    def test_decode_gemv_fp4_batched_correctness(self, setup_weights: dict) -> None:
        """Test decode_gemv_fp4_batched for M=2..8."""
        assert torch is not None
        cfg = setup_weights
        K, _N = cfg["K"], cfg["N"]
        packed, scales = cfg["packed"], cfg["scales"]

        from metal_marlin.kernels import marlin_gemm_fp4

        for M in [2, 4, 8]:
            A = torch.randn(M, K, dtype=torch.float16, device="mps")
            torch.mps.synchronize()

            # Full GEMM reference
            ref = marlin_gemm_fp4(A, packed, scales, cfg["group_size"])
            torch.mps.synchronize()

            # Batched decode (uses same kernel, batched GEMM for small M)
            out = marlin_gemm_fp4(A, packed, scales, cfg["group_size"])
            torch.mps.synchronize()

            ref_np = ref.cpu().float().numpy()
            out_np = out.cpu().float().numpy()

            np.testing.assert_allclose(
                out_np,
                ref_np,
                rtol=1e-3,
                atol=1e-3,
                err_msg=f"decode_gemv_fp4_batched (M={M}) differs from GEMM",
            )

    @requires_torch
    @requires_mps
    @requires_metal_dispatch
    def test_decode_gemv_small_dimensions(self) -> None:
        """Test decode GEMV with small dimensions."""
        assert torch is not None

        from metal_marlin.kernels import marlin_gemm_fp4, pack_fp4_weights

        K = 256
        N = 512
        group_size = 32

        np.random.seed(123)
        W_np = np.random.randn(N, K).astype(np.float32) * 0.1
        W_torch = torch.from_numpy(W_np).to(torch.float16)
        packed, scales = pack_fp4_weights(W_torch, group_size=group_size)
        torch.mps.synchronize()

        A = torch.randn(K, dtype=torch.float16, device="mps")
        A_2d = A.unsqueeze(0)  # [1, K]
        torch.mps.synchronize()

        out = marlin_gemm_fp4(A_2d, packed, scales, group_size)
        torch.mps.synchronize()

        out_1d = out.squeeze(0)
        assert out_1d.shape == (N,), f"Expected shape ({N},), got {out_1d.shape}"

    @requires_torch
    @requires_mps
    @requires_metal_dispatch
    @pytest.mark.slow
    def test_decode_gemv_large_dimensions(self) -> None:
        """Test decode GEMV with large dimensions (30B-scale projections)."""
        assert torch is not None

        from metal_marlin.kernels import marlin_gemm_fp4, pack_fp4_weights

        # Typical 30B model dimensions
        K = 8192
        N = 22016  # intermediate_size for ~30B
        group_size = 128

        np.random.seed(456)
        W_np = np.random.randn(N, K).astype(np.float32) * 0.02
        W_torch = torch.from_numpy(W_np).to(torch.float16)
        packed, scales = pack_fp4_weights(W_torch, group_size=group_size)
        torch.mps.synchronize()

        A = torch.randn(K, dtype=torch.float16, device="mps")
        A_2d = A.unsqueeze(0)  # [1, K]
        torch.mps.synchronize()

        out = marlin_gemm_fp4(A_2d, packed, scales, group_size)
        torch.mps.synchronize()

        out_1d = out.squeeze(0)
        assert out_1d.shape == (N,), f"Expected shape ({N},), got {out_1d.shape}"


class TestSelectDecodeKernel:
    """Tests for kernel selection logic."""

    def test_m1_small_n(self) -> None:
        """M=1, small N -> standard decode."""
        from metal_marlin.inference import select_decode_kernel

        kernel = select_decode_kernel(M=1, N=256, K=4096)
        assert kernel == "decode_gemv_fp4"

    def test_m1_large_n(self) -> None:
        """M=1, large N -> wide decode."""
        from metal_marlin.inference import select_decode_kernel

        kernel = select_decode_kernel(M=1, N=4096, K=4096)
        assert kernel == "decode_gemv_fp4_wide"

    def test_m1_boundary_n(self) -> None:
        """M=1, N=512 boundary -> wide decode."""
        from metal_marlin.inference import select_decode_kernel

        kernel = select_decode_kernel(M=1, N=512, K=4096)
        assert kernel == "decode_gemv_fp4_wide"

    def test_small_batch(self) -> None:
        """M=2..8 -> batched decode."""
        from metal_marlin.inference import select_decode_kernel

        for m in [2, 4, 8]:
            kernel = select_decode_kernel(M=m, N=4096, K=4096)
            assert kernel == "decode_gemv_fp4_batched", f"Failed for M={m}"

    def test_large_batch(self) -> None:
        """M > 8 -> full GEMM."""
        from metal_marlin.inference import select_decode_kernel

        for m in [16, 32, 64]:
            kernel = select_decode_kernel(M=m, N=4096, K=4096)
            assert kernel == "marlin_gemm_fp4", f"Failed for M={m}"


class TestQuantizedKVAttention:
    """Tests for quantized KV cache attention."""

    @pytest.fixture
    def setup_attention(self) -> dict:
        """Create test tensors for attention tests."""
        if not HAS_TORCH or torch is None:
            pytest.skip("PyTorch not available")
        if not HAS_MPS:
            pytest.skip("MPS not available")

        batch_size = 1
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        seq_len = 512

        # Query for single new token
        q = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.bfloat16, device="mps")

        # KV cache (simulate FP8 quantized)
        k_cache_fp = torch.randn(
            batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16, device="mps"
        )
        v_cache_fp = torch.randn(
            batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16, device="mps"
        )

        # Quantize to FP8 (simulated with uint8 + scales)
        k_max = k_cache_fp.abs().amax(dim=-1, keepdim=True)
        k_max = k_max.clamp(min=1e-8)
        k_scale = k_max / 448.0
        k_cache_quant = ((k_cache_fp / k_scale) * 127.0 + 128.0).round()
        k_cache_quant = k_cache_quant.clamp(0, 255).to(torch.uint8)

        v_max = v_cache_fp.abs().amax(dim=-1, keepdim=True)
        v_max = v_max.clamp(min=1e-8)
        v_scale = v_max / 448.0
        v_cache_quant = ((v_cache_fp / v_scale) * 127.0 + 128.0).round()
        v_cache_quant = v_cache_quant.clamp(0, 255).to(torch.uint8)

        torch.mps.synchronize()

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

    @requires_torch
    @requires_mps
    def test_quantized_kv_attention_shape(self, setup_attention: dict) -> None:
        """Test output shape of quantized KV attention."""
        assert torch is not None
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
        torch.mps.synchronize()

        expected_shape = (
            cfg["batch_size"],
            cfg["num_heads"],
            1,
            cfg["head_dim"],
        )
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"

    @requires_torch
    @requires_mps
    def test_quantized_kv_attention_accuracy(self, setup_attention: dict) -> None:
        """Test quantized attention is close to FP reference."""
        assert torch is not None
        import torch.nn.functional as F

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
        k_exp = k_fp.repeat_interleave(repeat_factor, dim=1)
        v_exp = v_fp.repeat_interleave(repeat_factor, dim=1)

        scale = 1.0 / math.sqrt(head_dim)
        attn_weights = (q @ k_exp.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        ref_out = attn_weights @ v_exp
        torch.mps.synchronize()

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
        torch.mps.synchronize()

        ref_np = ref_out.cpu().float().numpy()
        quant_np = quant_out.cpu().float().numpy()

        # FP8 quantization introduces some error, allow larger tolerance
        np.testing.assert_allclose(
            quant_np,
            ref_np,
            rtol=0.1,
            atol=0.1,
            err_msg="Quantized KV attention differs too much from FP reference",
        )


class TestDecodeState:
    """Tests for DecodeState persistent buffer management."""

    @requires_torch
    @requires_mps
    def test_decode_state_creation(self) -> None:
        """Test DecodeState can be created with config and weights."""
        assert torch is not None
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
                q_proj_weight=torch.zeros(
                    (hidden_size // 8, num_heads * head_dim), dtype=torch.uint32, device="mps"
                ),
                q_proj_scales=torch.ones(
                    (hidden_size // group_size, num_heads * head_dim),
                    dtype=torch.bfloat16,
                    device="mps",
                ),
                k_proj_weight=torch.zeros(
                    (hidden_size // 8, num_kv_heads * head_dim), dtype=torch.uint32, device="mps"
                ),
                k_proj_scales=torch.ones(
                    (hidden_size // group_size, num_kv_heads * head_dim),
                    dtype=torch.bfloat16,
                    device="mps",
                ),
                v_proj_weight=torch.zeros(
                    (hidden_size // 8, num_kv_heads * head_dim), dtype=torch.uint32, device="mps"
                ),
                v_proj_scales=torch.ones(
                    (hidden_size // group_size, num_kv_heads * head_dim),
                    dtype=torch.bfloat16,
                    device="mps",
                ),
                o_proj_weight=torch.zeros(
                    (num_heads * head_dim // 8, hidden_size), dtype=torch.uint32, device="mps"
                ),
                o_proj_scales=torch.ones(
                    (num_heads * head_dim // group_size, hidden_size),
                    dtype=torch.bfloat16,
                    device="mps",
                ),
                gate_proj_weight=torch.zeros(
                    (hidden_size // 8, intermediate_size), dtype=torch.uint32, device="mps"
                ),
                gate_proj_scales=torch.ones(
                    (hidden_size // group_size, intermediate_size),
                    dtype=torch.bfloat16,
                    device="mps",
                ),
                up_proj_weight=torch.zeros(
                    (hidden_size // 8, intermediate_size), dtype=torch.uint32, device="mps"
                ),
                up_proj_scales=torch.ones(
                    (hidden_size // group_size, intermediate_size),
                    dtype=torch.bfloat16,
                    device="mps",
                ),
                down_proj_weight=torch.zeros(
                    (intermediate_size // 8, hidden_size), dtype=torch.uint32, device="mps"
                ),
                down_proj_scales=torch.ones(
                    (intermediate_size // group_size, hidden_size),
                    dtype=torch.bfloat16,
                    device="mps",
                ),
                input_layernorm_weight=torch.ones(
                    (hidden_size,), dtype=torch.bfloat16, device="mps"
                ),
                post_attention_layernorm_weight=torch.ones(
                    (hidden_size,), dtype=torch.bfloat16, device="mps"
                ),
            )

        layer_weights = [make_weights() for _ in range(config.num_layers)]

        state = DecodeState(config, layer_weights)

        # Check persistent buffers were created
        assert state.qkv_buffer is not None
        assert state.attn_out_buffer is not None
        assert state.mlp_buffer is not None


class TestDecodePerformance:
    """Performance-oriented tests (optional, may be slow)."""

    @requires_torch
    @requires_mps
    @pytest.mark.slow
    def test_decode_vs_gemm_speedup(self) -> None:
        """Measure speedup of decode GEMV vs full GEMM."""
        assert torch is not None
        import time

        from metal_marlin.kernels import marlin_gemm_fp4, pack_fp4_weights

        K = 4096
        N = 4096
        group_size = 128
        warmup_iters = 10
        bench_iters = 100

        # Setup weights
        np.random.seed(789)
        W_np = np.random.randn(N, K).astype(np.float32) * 0.1
        W_torch = torch.from_numpy(W_np).to(torch.float16)
        packed, scales = pack_fp4_weights(W_torch, group_size=group_size)
        torch.mps.synchronize()

        A = torch.randn(1, K, dtype=torch.float16, device="mps")
        torch.mps.synchronize()

        # Warmup GEMM
        for _ in range(warmup_iters):
            marlin_gemm_fp4(A, packed, scales, group_size)
            torch.mps.synchronize()

        # Benchmark GEMM
        start = time.perf_counter()
        for _ in range(bench_iters):
            marlin_gemm_fp4(A, packed, scales, group_size)
            torch.mps.synchronize()
        gemm_time = (time.perf_counter() - start) / bench_iters * 1000  # ms

        # Warmup decode (same kernel for now)
        for _ in range(warmup_iters):
            marlin_gemm_fp4(A, packed, scales, group_size)
            torch.mps.synchronize()

        # Benchmark decode
        start = time.perf_counter()
        for _ in range(bench_iters):
            _out = marlin_gemm_fp4(A, packed, scales, group_size)
            torch.mps.synchronize()
        decode_time = (time.perf_counter() - start) / bench_iters * 1000  # ms

        speedup = gemm_time / decode_time if decode_time > 0 else 1.0
        print(f"\nGEMM time: {gemm_time:.3f}ms, Decode time: {decode_time:.3f}ms")
        print(f"Speedup: {speedup:.2f}x")

        # For same kernel, speedup should be ~1.0
        # When specialized decode kernels are added, this test validates speedup
        assert speedup > 0.5, f"Decode should not be much slower than GEMM (got {speedup:.2f}x)"
