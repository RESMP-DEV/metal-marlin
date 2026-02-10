#!/usr/bin/env python3
"""Benchmark MoE kernel variants using actual dispatch paths.

Tests the practical kernel selection paths available through dispatch_moe_trellis_swiglu
and documents optimal kernel selection for M4 Max.

Usage:
    cd contrib/metal_marlin
    uv run benchmarks/bench_kernel_variants.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from metal_marlin.trellis.model import TrellisForCausalLM
from metal_marlin.trellis.moe_dispatch import (
    dispatch_moe_trellis_swiglu,
    select_moe_kernel,
)

# Batch sizes to test
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
TOP_K = 2

# Test configuration
WARMUP_ITERS = 10
BENCH_ITERS = 50


def load_model() -> TrellisForCausalLM:
    """Load Trellis model for benchmarking."""
    model_path = Path("models/GLM-4.7-Flash-Marlin-MMFP4")
    if not model_path.exists():
        raise RuntimeError(f"Model not found at {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = TrellisForCausalLM.from_pretrained(str(model_path), device="mps")
    return model


def get_moe_layer(model: TrellisForCausalLM):
    """Find first MoE layer in model."""
    for layer in model.model.layers:
        if hasattr(layer.mlp, "_is_mixed_precision"):
            return layer.mlp
    raise RuntimeError("No MoE layer found")


def force_kernel_dispatch(
    moe,
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_probs: torch.Tensor,
    forced_kernel: str | None = None,
    use_fp32_acc: bool = False,
) -> torch.Tensor:
    """Dispatch with optional kernel override for benchmarking.
    
    This bypasses select_moe_kernel to test specific kernel variants.
    """
    from metal_marlin.metal_dispatch import mps_tensor_to_metal_buffer
    import numpy as np
    import Metal
    
    lib = moe._get_lib()
    cached = moe._cached_weight_buffers
    
    if cached is None:
        raise RuntimeError("Cached buffers not initialized")
    
    batch_size = x.shape[0]
    hidden_dim = moe.hidden_dim
    intermediate_dim = moe.intermediate_dim
    num_experts = len(moe.experts)
    bits = moe.bits
    
    # Prepare inputs
    x_contig = x.contiguous()
    expert_ids_int = expert_ids.int().contiguous()
    expert_probs_half = expert_probs.half().contiguous()
    
    # Use standard dispatch if no forced kernel
    if forced_kernel is None:
        return dispatch_moe_trellis_swiglu(
            lib=lib,
            activations=x,
            gate_weights=None,
            gate_scales=None,
            up_weights=None,
            up_scales=None,
            down_weights=None,
            down_scales=None,
            gate_su=None,
            gate_sv=None,
            up_su=None,
            up_sv=None,
            down_su=None,
            down_sv=None,
            grid=None,
            expert_ids=expert_ids,
            expert_probs=expert_probs,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            top_k=TOP_K,
            bits=bits,
            cached_buffers=cached,
            buffer_pool=moe._get_buffer_pool(),
            use_fp32_acc=use_fp32_acc,
        )
    
    # Forced kernel path - manually dispatch
    kernel_name = forced_kernel
    tile_n = 128 if "large_batch" in kernel_name else 64
    
    # Compute grid
    is_decode = kernel_name == "moe_trellis_swiglu_decode"
    is_prefill4 = "prefill4" in kernel_name
    
    grid_x = (hidden_dim + tile_n - 1) // tile_n
    if is_decode:
        grid_y = 1
    elif is_prefill4:
        grid_y = (batch_size + 3) // 4
    else:
        grid_y = batch_size
    grid_z = TOP_K
    
    # Get buffers
    activations_buf = mps_tensor_to_metal_buffer(x_contig, lib.device)
    expert_ids_buf = mps_tensor_to_metal_buffer(expert_ids_int, lib.device)
    expert_probs_buf = mps_tensor_to_metal_buffer(expert_probs_half, lib.device)
    
    # Create output buffer
    output_fp32 = torch.zeros(batch_size, hidden_dim, dtype=torch.float32, device="mps")
    output_buf = mps_tensor_to_metal_buffer(output_fp32, lib.device, copy_back=True)
    
    # Create params buffer
    params_data = np.array(
        [batch_size, hidden_dim, intermediate_dim, num_experts, TOP_K,
         bits, bits, bits, tile_n,
         1 << bits, 1 << bits, 1 << bits],
        dtype=np.uint32,
    )
    params_buf = lib.device.newBufferWithBytes_length_options_(
        params_data.tobytes(), params_data.nbytes, Metal.MTLResourceStorageModeShared
    )
    
    # Get pipeline
    pipeline = lib.get_kernel("gemm_trellis_moe", kernel_name)
    if pipeline is None:
        raise RuntimeError(f"Kernel {kernel_name} not available")
    
    # Dispatch
    command_buffer = lib.command_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    
    # Bind all buffers
    encoder.setBuffer_offset_atIndex_(activations_buf, 0, 0)
    encoder.setBuffer_offset_atIndex_(cached.gate_weights, 0, 1)
    encoder.setBuffer_offset_atIndex_(cached.gate_scales, 0, 2)
    encoder.setBuffer_offset_atIndex_(cached.up_weights, 0, 3)
    encoder.setBuffer_offset_atIndex_(cached.up_scales, 0, 4)
    encoder.setBuffer_offset_atIndex_(cached.down_weights, 0, 5)
    encoder.setBuffer_offset_atIndex_(cached.down_scales, 0, 6)
    encoder.setBuffer_offset_atIndex_(cached.gate_su, 0, 7)
    encoder.setBuffer_offset_atIndex_(cached.gate_sv, 0, 8)
    encoder.setBuffer_offset_atIndex_(cached.up_su, 0, 9)
    encoder.setBuffer_offset_atIndex_(cached.up_sv, 0, 10)
    encoder.setBuffer_offset_atIndex_(cached.down_su, 0, 11)
    encoder.setBuffer_offset_atIndex_(cached.down_sv, 0, 12)
    encoder.setBuffer_offset_atIndex_(cached.grid, 0, 13)
    encoder.setBuffer_offset_atIndex_(expert_ids_buf, 0, 14)
    encoder.setBuffer_offset_atIndex_(expert_probs_buf, 0, 15)
    encoder.setBuffer_offset_atIndex_(output_buf, 0, 16)
    encoder.setBuffer_offset_atIndex_(params_buf, 0, 17)
    
    grid_size = Metal.MTLSizeMake(grid_x, grid_y, grid_z)
    tg_size = Metal.MTLSizeMake(128, 1, 1)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)
    encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()
    
    # Copy output back
    output_fp16 = output_fp32.half()
    return output_fp16


def benchmark_variant(
    moe,
    batch_size: int,
    forced_kernel: str | None = None,
    use_fp32_acc: bool = False,
) -> dict[str, Any] | None:
    """Benchmark a specific kernel variant."""
    
    hidden_dim = moe.hidden_dim
    num_experts = len(moe.experts)
    
    # Create test tensors
    x = torch.randn(batch_size, hidden_dim, device="mps", dtype=torch.float16)
    expert_ids = torch.randint(0, num_experts, (batch_size, TOP_K), device="mps")
    expert_probs = torch.ones(batch_size, TOP_K, device="mps", dtype=torch.float32)
    expert_probs = expert_probs / expert_probs.sum(dim=1, keepdim=True)
    
    # Determine kernel name for result
    if forced_kernel:
        kernel_name = forced_kernel
    else:
        kernel_name, _ = select_moe_kernel(batch_size, use_fp32_acc)
    
    # Warmup
    try:
        for _ in range(WARMUP_ITERS):
            _ = force_kernel_dispatch(
                moe, x, expert_ids, expert_probs,
                forced_kernel=forced_kernel,
                use_fp32_acc=use_fp32_acc
            )
        torch.mps.synchronize()
        
        # Benchmark
        times = []
        for _ in range(BENCH_ITERS):
            torch.mps.synchronize()
            start = time.perf_counter()
            _ = force_kernel_dispatch(
                moe, x, expert_ids, expert_probs,
                forced_kernel=forced_kernel,
                use_fp32_acc=use_fp32_acc
            )
            torch.mps.synchronize()
            times.append((time.perf_counter() - start) * 1000)  # ms
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        p50_time = sorted(times)[len(times) // 2]
        
        tokens_per_sec = (batch_size / avg_time) * 1000 if avg_time > 0 else 0
        
        return {
            "kernel": kernel_name,
            "batch_size": batch_size,
            "avg_ms": round(avg_time, 3),
            "min_ms": round(min_time, 3),
            "p50_ms": round(p50_time, 3),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "forced": forced_kernel is not None,
            "fp32_acc": use_fp32_acc,
        }
    except Exception as e:
        print(f"    Error: {e}")
        return None


def run_benchmark():
    """Run kernel variant benchmark."""
    print("=" * 80)
    print("MoE Kernel Variant Benchmark for M4 Max")
    print("=" * 80)
    print()
    
    model = load_model()
    moe = get_moe_layer(model)
    
    print(f"\nModel: GLM-4.7-Flash-Marlin-MMFP4")
    print(f"Hidden dim: {moe.hidden_dim}, Intermediate: {moe.intermediate_dim}")
    print(f"Experts: {len(moe.experts)}, Top-K: {TOP_K}")
    print(f"\nBatch sizes: {BATCH_SIZES}")
    print(f"Iterations: {BENCH_ITERS} (warmup: {WARMUP_ITERS})")
    print()
    
    results = []
    
    # Test 1: Current select_moe_kernel selection
    print("\n" + "-" * 60)
    print("Testing CURRENT select_moe_kernel selection:")
    print("-" * 60)
    
    for batch_size in BATCH_SIZES:
        print(f"  Batch {batch_size:3d}...", end=" ", flush=True)
        result = benchmark_variant(moe, batch_size, forced_kernel=None)
        if result:
            results.append(result)
            print(f"{result['kernel']:<40} {result['avg_ms']:>7.3f} ms  "
                  f"{result['tokens_per_sec']:>8.1f} tok/s")
    
    # Test 2: Large batch kernel (batch >= 16)
    print("\n" + "-" * 60)
    print("Testing large_batch kernel (tile_n=128):")
    print("-" * 60)
    
    for batch_size in [16, 32, 64, 128]:
        print(f"  Batch {batch_size:3d}...", end=" ", flush=True)
        result = benchmark_variant(
            moe, batch_size,
            forced_kernel="moe_trellis_swiglu_large_batch"
        )
        if result:
            results.append(result)
            print(f"{result['kernel']:<40} {result['avg_ms']:>7.3f} ms  "
                  f"{result['tokens_per_sec']:>8.1f} tok/s")
    
    # Test 3: SIMD kernel for small batches
    print("\n" + "-" * 60)
    print("Testing SIMD kernel:")
    print("-" * 60)
    
    for batch_size in [1, 2, 4, 8, 16]:
        print(f"  Batch {batch_size:3d}...", end=" ", flush=True)
        result = benchmark_variant(
            moe, batch_size,
            forced_kernel="moe_trellis_swiglu_simd"
        )
        if result:
            results.append(result)
            print(f"{result['kernel']:<40} {result['avg_ms']:>7.3f} ms  "
                  f"{result['tokens_per_sec']:>8.1f} tok/s")
    
    # Save results
    output_file = Path("benchmarks/results/kernel_variants_m4max.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump({
            "device": "M4 Max",
            "model": "GLM-4.7-Flash-Marlin-MMFP4",
            "warmup_iters": WARMUP_ITERS,
            "bench_iters": BENCH_ITERS,
            "results": results,
        }, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    
    # Generate recommendations
    print("\n" + "=" * 80)
    print("OPTIMAL KERNEL SELECTION RECOMMENDATIONS")
    print("=" * 80)
    
    # Group by batch and find best
    by_batch: dict[int, list[dict]] = {}
    for r in results:
        b = r["batch_size"]
        if b not in by_batch:
            by_batch[b] = []
        by_batch[b].append(r)
    
    print(f"\n{'Batch':<8} {'Fastest Kernel':<40} {'Time (ms)':<12} {'Tok/s':<10}")
    print("-" * 80)
    
    recommendations = []
    for batch_size in sorted(by_batch.keys()):
        variants = by_batch[batch_size]
        variants.sort(key=lambda x: x["avg_ms"])
        best = variants[0]
        recommendations.append((batch_size, best["kernel"], best["avg_ms"]))
        print(f"{batch_size:<8} {best['kernel']:<40} {best['avg_ms']:<12.3f} {best['tokens_per_sec']:<10.1f}")
    
    # Print optimized select_moe_kernel logic
    print("\n" + "=" * 80)
    print("PROPOSED OPTIMIZED select_moe_kernel LOGIC:")
    print("=" * 80)
    print("""
def select_moe_kernel(
    batch_size: int,
    use_fp32_acc: bool = False,
    gate_bits: int | None = None,
    up_bits: int | None = None,
    down_bits: int | None = None,
) -> tuple[str, int]:
    \"\"\"Select optimal MoE kernel based on batch size and bit widths.
    
    M4 Max Optimized Selection (based on profiling):
    - batch = 1: decode kernel (fused SwiGLU, lowest latency)
    - 2 <= batch <= 8: prefill4 kernel (4-token parallel processing)
    - 9 <= batch <= 16: prefill4 or base kernel depending on fp32_acc
    - batch >= 17: large_batch kernel (128-column tiles, better BW)
    
    Specialized kernels for known bit-width patterns (GLM-4.7-Flash):
    - gate=6, up=2, down=3: optimized decode kernel with compile-time params
    - gate=6, up=3, down=4: alternative optimized decode
    - gate=6, up=2, down=4: alternative optimized decode
    \"\"\"
    # Specialized decode kernels for common bit patterns
    if batch_size == 1 and not use_fp32_acc:
        if gate_bits == 6 and up_bits == 2 and down_bits == 3:
            return "moe_trellis_swiglu_decode_6_2_3", 64
        if gate_bits == 6 and up_bits == 3 and down_bits == 4:
            return "moe_trellis_swiglu_decode_6_3_4", 64
        if gate_bits == 6 and up_bits == 2 and down_bits == 4:
            return "moe_trellis_swiglu_decode_6_2_4", 64

    if batch_size == 1:
        return "moe_trellis_swiglu_decode", 64
    elif batch_size <= 8:
        # Small prefill - use prefill4 for 4-token parallelism
        if use_fp32_acc:
            return "moe_trellis_swiglu_prefill4_fp32acc", 64
        return "moe_trellis_swiglu_prefill4", 64
    elif batch_size <= 16:
        # Medium batch - prefill4 still good
        if use_fp32_acc:
            return "moe_trellis_swiglu_prefill4_fp32acc", 64
        return "moe_trellis_swiglu_prefill4", 64
    else:
        # Large batch (17+) - use large_batch kernel with 128-column tiles
        # for better memory bandwidth utilization
        # Note: large_batch kernel doesn't have fp32acc variant
        return "moe_trellis_swiglu_large_batch", 128
""")


if __name__ == "__main__":
    run_benchmark()
