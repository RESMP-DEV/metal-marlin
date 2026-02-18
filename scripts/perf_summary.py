#!/usr/bin/env python3
"""Print performance summary."""

import sys
from pathlib import Path

# Add project root to path to allow importing metal_marlin
METAL_MARLIN_ROOT = Path(__file__).parent.parent
sys.path.append(str(METAL_MARLIN_ROOT))

try:
    from metal_marlin.profiling.memory_bandwidth import analyze_bandwidth_bottleneck
    from metal_marlin.profiling.occupancy import detect_gpu
except ImportError:
    # Fallback if dependencies are missing
    def detect_gpu():
        class MockGPU:
            peak_bw_gbs = 400
            name = "M4 Max (Fallback)"
            peak_tflops_fp16 = 32.0
        return MockGPU()

    def analyze_bandwidth_bottleneck(achieved, peak, ai, peak_tflops=32.0):
        util = (achieved / peak) * 100
        return {
            "bound": "memory" if ai < 75 else "compute",
            "utilization_pct": util,
            "headroom_pct": 100 - util,
            "arithmetic_intensity": ai
        }

print("=" * 60)
print("GLM-4.7-Flash Performance Summary")
print("=" * 60)

# Old benchmark (from bench_glm47_trellis.py using trellis.lm)
old_decode_ms = 21638
old_tps = 1000 / old_decode_ms

# Actual measured from profile_layers.py
measured_decode_ms = 17249
measured_tps = 1000 / measured_decode_ms

print("")
print("OLD Implementation (trellis.lm - dequant on each forward):")
print(f"  Single token decode: {old_decode_ms:.0f} ms")
print(f"  Throughput: {old_tps:.3f} tok/s")
print("")
print("NEW Implementation (trellis.model - fast MoE kernel):")
print(f"  Single token decode: {measured_decode_ms:.0f} ms")
print(f"  Throughput: {measured_tps:.3f} tok/s")
print("")
print(f"Improvement: {old_decode_ms / measured_decode_ms:.2f}x faster")
print("")
print("Per-layer breakdown (MoE layer):")
print("  MLP (MoE): 325 ms")
print("  Attention: 71 ms")
print("  Total layer: 416 ms")
print("")
pct = 325 / 416 * 100
print("Bottleneck: MoE kernel (64 experts x 8 selected x SwiGLU)")
print(f"  325ms / 416ms = {pct:.0f}% of layer time")
print("")

# Memory bandwidth utilization per kernel
print("=" * 60)
print("Memory Bandwidth Utilization (Per-Kernel)")
print("=" * 60)

# Detect hardware limits
gpu = detect_gpu()
peak_bw = gpu.peak_bw_gbs
peak_tflops = gpu.peak_tflops_fp16
print(f"Hardware: {gpu.name}")
print(f"Peak Memory Bandwidth: {peak_bw:.1f} GB/s")
print(f"Peak Compute (FP16):   {peak_tflops:.1f} TFLOPS")
print("-" * 60)

# MoE kernel: 64 experts x 8 selected, W4A16 matmul + SwiGLU
# Approximate data movement per token: weights (quantized) + activations
moe_data_gb = 0.85  # GB of data transferred
moe_time_s = 0.325  # 325ms
moe_bw_gbs = moe_data_gb / moe_time_s

# Estimate AI: 4-bit weights => 0.5 bytes/param. 
# Params = 0.85GB / 0.5 = 1.7B params. 
# FLOPs = 2 * Params = 3.4 GFLOPs.
moe_ai = (3.4 * 1e9) / (moe_data_gb * 1e9)

moe_analysis = analyze_bandwidth_bottleneck(
    achieved_gbs=moe_bw_gbs,
    peak_gbs=peak_bw,
    arithmetic_intensity=moe_ai,
    peak_tflops=peak_tflops
)

print(f"MoE Kernel (W4A16 MatMul + SwiGLU):")
print(f"  Data transferred: {moe_data_gb:.2f} GB")
print(f"  Time:             {moe_time_s * 1000:.0f} ms")
print(f"  Bandwidth:        {moe_bw_gbs:.1f} GB/s")
print(f"  Utilization:      {moe_analysis['utilization_pct']:.1f}%")
print(f"  Bottleneck:       {moe_analysis['bound'].upper()} (AI={moe_ai:.1f})")
if moe_analysis['utilization_pct'] < 10:
    print("  Analysis:         Latency bound (small batch size)")

print("")

# Attention kernel: QKV proj + attention + output proj
attn_data_gb = 0.12  # GB of data transferred
attn_time_s = 0.071  # 71ms
attn_bw_gbs = attn_data_gb / attn_time_s

# Estimate AI: similar 4-bit weights
attn_ai = 4.0

attn_analysis = analyze_bandwidth_bottleneck(
    achieved_gbs=attn_bw_gbs,
    peak_gbs=peak_bw,
    arithmetic_intensity=attn_ai,
    peak_tflops=peak_tflops
)

print(f"Attention Kernel (QKV + Softmax + Proj):")
print(f"  Data transferred: {attn_data_gb:.2f} GB")
print(f"  Time:             {attn_time_s * 1000:.0f} ms")
print(f"  Bandwidth:        {attn_bw_gbs:.1f} GB/s")
print(f"  Utilization:      {attn_analysis['utilization_pct']:.1f}%")
print(f"  Bottleneck:       {attn_analysis['bound'].upper()} (AI={attn_ai:.1f})")
if attn_analysis['utilization_pct'] < 10:
    print("  Analysis:         Latency bound (small batch size)")

print("")
print("Note: Low utilization with low batch size indicates latency/overhead bottleneck.")
print("      Fused kernels reduce launch overhead but cannot eliminate DRAM latency.")
print("=" * 60)
