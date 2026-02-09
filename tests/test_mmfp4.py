"""MMFP4 unit tests for dequantization, GEMM, linear pass, edge cases, and memory.

These tests cover:
1. MMFP4 dequantization correctness against a reference implementation.
2. MMFP4 GEMM correctness against FP16 reference matmul.
3. MMFP4 linear forward/backward behavior.
4. Edge cases (non-aligned sizes and multiple group sizes).
5. Memory growth checks for repeated MMFP4 GEMM calls.
"""

from __future__ import annotations

import gc

import pytest
import torch
import torch.nn as nn

from metal_marlin.kernels import dequant_fp4, marlin_gemm_fp4, pack_fp4_weights
from metal_marlin.layers import MarlinLinear
from metal_marlin.quantize import pack_fp4_weights as pack_fp4_weights_cpu
from metal_marlin.quantize import unpack_fp4_weights


HAS_MPS = torch.backends.mps.is_available()
requires_mps = pytest.mark.skipif(not HAS_MPS, reason="Requires MPS (Apple Silicon)")


def _fp4_e2m1_table(device: torch.device) -> torch.Tensor:
    """Return the FP4 E2M1 lookup table used by MMFP4 paths."""
    return torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=device,
        dtype=torch.float32,
    )


def _reference_dequant_mmfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Reference MMFP4 dequantization for packed [K/8, N] weights."""
    k_packed, n = packed.shape
    k = k_packed * 8
    table = _fp4_e2m1_table(packed.device)

    out = torch.empty((k, n), dtype=torch.float32, device=packed.device)
    for bit_pos in range(8):
        nibbles = ((packed >> (bit_pos * 4)) & 0xF).to(torch.long)
        out[bit_pos::8, :] = table[nibbles]

    scales_expanded = scales.to(torch.float32).repeat_interleave(group_size, dim=0)[:k, :]
    return (out * scales_expanded).to(torch.float16)


@pytest.fixture(scope="module")
def loaded_mmfp4_model():
    """Load a deterministic FP16 model and its MMFP4 packed representation."""
    if not HAS_MPS:
        pytest.skip("MPS not available")

    device = torch.device("mps")
    torch.manual_seed(1234)

    in_features = 256
    out_features = 128
    group_size = 32

    fp16_linear = nn.Linear(
        in_features,
        out_features,
        bias=True,
        device=device,
        dtype=torch.float16,
    ).eval()

    mmfp4_linear = MarlinLinear.from_linear(
        fp16_linear,
        quant_type="fp4",
        group_size=group_size,
    ).to(device)

    packed, scales = pack_fp4_weights(fp16_linear.weight.detach(), group_size=group_size)

    return {
        "device": device,
        "in_features": in_features,
        "out_features": out_features,
        "group_size": group_size,
        "fp16_linear": fp16_linear,
        "mmfp4_linear": mmfp4_linear,
        "packed": packed,
        "scales": scales,
    }


@pytest.fixture(scope="module")
def mmfp4_kernel_case():
    """Create reusable MMFP4 kernel inputs."""
    if not HAS_MPS:
        pytest.skip("MPS not available")

    device = torch.device("mps")
    torch.manual_seed(2026)

    m = 8
    k = 1024
    n = 128
    group_size = 32

    activations = torch.randn((m, k), dtype=torch.float16, device=device)
    weights = torch.randn((k, n), dtype=torch.float16, device=device)
    packed, scales = pack_fp4_weights(weights.T.contiguous(), group_size=group_size)

    return {
        "device": device,
        "m": m,
        "k": k,
        "n": n,
        "group_size": group_size,
        "activations": activations,
        "weights": weights,
        "packed": packed,
        "scales": scales,
    }


@requires_mps
def test_mmfp4_dequantization_correctness(mmfp4_kernel_case):
    """MMFP4 dequantization matches reference implementation."""
    packed = mmfp4_kernel_case["packed"]
    scales = mmfp4_kernel_case["scales"]
    k = mmfp4_kernel_case["k"]
    n = mmfp4_kernel_case["n"]
    group_size = mmfp4_kernel_case["group_size"]

    ref = _reference_dequant_mmfp4(packed, scales, group_size)
    out = dequant_fp4(packed, scales, k, n, group_size=group_size)

    torch.testing.assert_close(out.float(), ref.float(), rtol=0.0, atol=1e-3)


@requires_mps
def test_mmfp4_gemm_correctness_vs_fp16_matmul(mmfp4_kernel_case):
    """MMFP4 GEMM output matches FP16 matmul against reference dequantized weights."""
    activations = mmfp4_kernel_case["activations"]
    packed = mmfp4_kernel_case["packed"]
    scales = mmfp4_kernel_case["scales"]
    group_size = mmfp4_kernel_case["group_size"]
    k = mmfp4_kernel_case["k"]

    dequant_ref = _reference_dequant_mmfp4(packed, scales, group_size)
    ref = activations.float() @ dequant_ref.float()
    out = marlin_gemm_fp4(activations, packed, scales, group_size=group_size)

    ref_scale = ref.abs().max().item() + 1e-7
    atol = max((float(k) ** 0.5) * 5e-3 * ref_scale, 1e-3)
    torch.testing.assert_close(out.float(), ref.float(), rtol=5e-2, atol=atol)


@requires_mps
def test_mmfp4linear_forward_backward_pass(loaded_mmfp4_model):
    """MMFP4 linear layer runs forward and backward with finite gradients."""
    layer = loaded_mmfp4_model["mmfp4_linear"]
    device = loaded_mmfp4_model["device"]
    in_features = loaded_mmfp4_model["in_features"]
    out_features = loaded_mmfp4_model["out_features"]

    x = torch.randn((4, in_features), device=device, dtype=torch.float16, requires_grad=True)
    y = layer(x)

    assert y.shape == (4, out_features)
    assert torch.isfinite(y).all()

    loss = y.float().pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_mmfp4_edge_case_non_aligned_sizes_roundtrip():
    """MMFP4 CPU pack/unpack handles non-aligned K and N via padding metadata."""
    torch.manual_seed(7)

    k = 130  # not divisible by 8 or group_size
    n = 67   # not divisible by common vector widths
    group_size = 32

    weights = torch.randn((k, n), dtype=torch.float32)
    packed, scales, meta = pack_fp4_weights_cpu(
        weights,
        group_size=group_size,
        pad_k=True,
        output_backend="torch",
    )
    recovered = unpack_fp4_weights(
        packed,
        scales,
        meta,
        output_backend="torch",
    )

    assert recovered.shape == weights.shape
    assert meta["orig_K"] == k
    assert meta["orig_N"] == n
    assert meta["padded_K"] % 8 == 0
    assert meta["padded_K"] % group_size == 0


@requires_mps
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_mmfp4_edge_case_group_sizes(group_size: int):
    """MMFP4 GEMM remains numerically stable across supported group sizes."""
    torch.manual_seed(100 + group_size)
    device = torch.device("mps")

    m = 4
    k = 256
    n = 96

    a = torch.randn((m, k), dtype=torch.float16, device=device)
    w = torch.randn((k, n), dtype=torch.float16, device=device)
    packed, scales = pack_fp4_weights(w.T.contiguous(), group_size=group_size)

    ref_w = _reference_dequant_mmfp4(packed, scales, group_size)
    ref = a.float() @ ref_w.float()
    out = marlin_gemm_fp4(a, packed, scales, group_size=group_size)

    ref_scale = ref.abs().max().item() + 1e-7
    atol = max((float(k) ** 0.5) * 5e-3 * ref_scale, 1e-3)
    torch.testing.assert_close(out.float(), ref.float(), rtol=5e-2, atol=atol)


@requires_mps
def test_mmfp4_memory_leak_check(mmfp4_kernel_case):
    """Repeated MMFP4 GEMM calls should not show unbounded memory growth."""
    device = mmfp4_kernel_case["device"]
    packed = mmfp4_kernel_case["packed"]
    scales = mmfp4_kernel_case["scales"]
    group_size = mmfp4_kernel_case["group_size"]
    k = mmfp4_kernel_case["k"]
    n = mmfp4_kernel_case["n"]

    if device.type != "mps":
        pytest.skip("MPS memory metrics require mps device")

    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    gc.collect()
    torch.mps.synchronize()

    # Warm-up to stabilize allocator behavior.
    for _ in range(5):
        x = torch.randn((4, k), dtype=torch.float16, device=device)
        _ = marlin_gemm_fp4(x, packed, scales, group_size=group_size)

    torch.mps.synchronize()
    gc.collect()
    baseline = torch.mps.current_allocated_memory()
    peak = baseline

    for step in range(30):
        x = torch.randn((4, k), dtype=torch.float16, device=device)
        out = marlin_gemm_fp4(x, packed, scales, group_size=group_size)
        assert out.shape == (4, n)
        del x, out

        if step % 5 == 0:
            torch.mps.synchronize()
            gc.collect()
            current = torch.mps.current_allocated_memory()
            peak = max(peak, current)

    torch.mps.synchronize()
    gc.collect()
    final = torch.mps.current_allocated_memory()

    # Allow moderate allocator noise, fail on sustained growth.
    allowed_growth_bytes = 96 * 1024 * 1024
    assert peak - baseline < allowed_growth_bytes, (
        "Potential MMFP4 memory leak detected: "
        f"peak growth={(peak - baseline) / (1024**2):.2f}MB, "
        f"baseline={baseline / (1024**2):.2f}MB"
    )
    assert final - baseline < allowed_growth_bytes, (
        "Potential MMFP4 memory leak detected: "
        f"final growth={(final - baseline) / (1024**2):.2f}MB, "
        f"baseline={baseline / (1024**2):.2f}MB"
    )
