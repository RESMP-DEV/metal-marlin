import time

import torch

import metal_marlin


def benchmark(name, fn, warmup=3, runs=10):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        torch.mps.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    avg = sum(times) / len(times)
    print(f"{name}: {avg:.2f}ms (min: {min(times):.2f}, max: {max(times):.2f})")
    return avg

# Test suite
m, k, n = 4096, 4096, 4096

a = torch.randn(m, k, device='mps', dtype=torch.float16)
b = torch.randn(k, n, device='mps', dtype=torch.float16)

print("=== Kernel Benchmarks (Post-Refactor) ===")
benchmark("dense_gemm", lambda: metal_marlin.dense_gemm(a, b))

# Attention
q = torch.randn(1, 8, 128, 64, device='mps', dtype=torch.float16)
k_attn = torch.randn(1, 8, 128, 64, device='mps', dtype=torch.float16)
v = torch.randn(1, 8, 128, 64, device='mps', dtype=torch.float16)
benchmark("attention", lambda: metal_marlin.attention(q, k_attn, v))

# MoE dispatch
hidden = torch.randn(128, 4096, device='mps', dtype=torch.float16)
experts = torch.randn(64, 4096, 4096, device='mps', dtype=torch.float16)
indices = torch.randint(0, 64, (128, 2), device='mps')
benchmark("moe_dispatch", lambda: metal_marlin.expert_gather(experts, indices, hidden))
