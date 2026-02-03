#!/usr/bin/env python3
"""Performance comparison after optimization."""

print("=" * 65)
print("GLM-4.7-Flash Performance Comparison (After Optimization)")
print("=" * 65)

# Before optimization
before_decode_ms = 17249
before_tps = 1000 / before_decode_ms

# After optimization
after_decode_ms = 7533
after_tps = 1000 / after_decode_ms

# GGUF reference
gguf_tps = 20

print("")
print("BEFORE Optimization:")
print(f"  Single token decode: {before_decode_ms:.0f} ms")
print(f"  Throughput: {before_tps:.3f} tok/s")
print("")
print("AFTER Optimization:")
print(f"  Single token decode: {after_decode_ms:.0f} ms")
print(f"  Throughput: {after_tps:.2f} tok/s")
print("")
print(f"Improvement: {before_decode_ms / after_decode_ms:.2f}x faster")
print("")
print("GGUF Reference:")
print(f"  Throughput: ~{gguf_tps} tok/s")
print(f"  Gap remaining: {gguf_tps / after_tps:.0f}x")
print("")
print("Per-layer breakdown (MoE layer):")
print("  Before: 416 ms/layer (MLP: 325ms, Attn: 71ms)")
print("  After:  172 ms/layer (MLP: 110ms, Attn: 65ms)")
layer_speedup = 416 / 172
print(f"  Layer speedup: {layer_speedup:.2f}x")
print("")
mlp_speedup = 325 / 110
print(f"MLP (MoE) improvement: 325ms -> 110ms = {mlp_speedup:.2f}x")
print("=" * 65)
