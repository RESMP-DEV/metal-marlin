#!/usr/bin/env python3
import time

import torch

# Simulate router computation
batch_size, seq_len, num_experts = 1, 128, 64
hidden_dim = 4096

print("Testing Router Top-K Performance:")
print("=" * 60)

# Create test inputs on MPS
hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device='mps', dtype=torch.float16)
router_weights = torch.randn(hidden_dim, num_experts, device='mps', dtype=torch.float16)

# Test router computation
times = []
for _ in range(10):
    t0 = time.perf_counter()

    # Router forward: linear + softmax + topk
    router_logits = torch.matmul(hidden_states, router_weights)  # [1, 128, 64]
    router_probs = torch.softmax(router_logits, dim=-1)
    topk_values, topk_indices = torch.topk(router_probs, k=3, dim=-1)

    torch.mps.synchronize()
    times.append(time.perf_counter() - t0)

avg_time = sum(times) / len(times)
print(f"Router time: {avg_time*1000:.1f}ms")
print(f"  Linear: ~{avg_time*0.3*1000:.1f}ms (estimated)")
print(f"  Softmax: ~{avg_time*0.3*1000:.1f}ms (estimated)")
print(f"  TopK: ~{avg_time*0.4*1000:.1f}ms (estimated)")

if avg_time > 0.05:
    print("\n⚠️  Router is slow - may need optimization")
else:
    print("\n✓ Router performance reasonable")
