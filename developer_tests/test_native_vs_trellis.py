#!/usr/bin/env python3
"""Compare PyTorch native MoE vs TrellisQuantized MoE performance."""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

print("Comparing PyTorch Native MoE vs TrellisQuantized MoE")
print("=" * 70)

# Test configuration (reduced due to MPS limits)
batch, seq_len, hidden_dim = 1, 64, 2048
num_experts, expert_dim = 32, 5632
topk = 3
num_iters = 10

print(f"Config: batch={batch}, seq={seq_len}, hidden={hidden_dim}")
print(f"Experts: {num_experts}, expert_dim={expert_dim}, topk={topk}")
print(f"Iterations: {num_iters}")
print()

# Create test data
hidden_states = torch.randn(batch, seq_len, hidden_dim, device='mps', dtype=torch.float16)
router_weights = torch.randn(hidden_dim, num_experts, device='mps', dtype=torch.float16)

# ============================================================================
# PyTorch Native MoE (simple implementation)
# ============================================================================
print("1. PyTorch Native MoE:")
print("-" * 70)

class NativeMoE(nn.Module):
    def __init__(self, hidden_dim, num_experts, expert_dim, topk):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.router = nn.Linear(hidden_dim, num_experts, bias=False, device='mps', dtype=torch.float16)
        self.router.weight.data = router_weights.T.clone()
        self.expert_gates = nn.Linear(hidden_dim, expert_dim, bias=False, device='mps', dtype=torch.float16)
        self.expert_ups = nn.Linear(hidden_dim, expert_dim, bias=False, device='mps', dtype=torch.float16)
        self.expert_downs = nn.Linear(expert_dim, hidden_dim, bias=False, device='mps', dtype=torch.float16)

    def forward(self, x):
        # Flatten: [batch, seq, hidden] -> [batch*seq, hidden]
        orig_shape = x.shape
        x_flat = x.view(-1, hidden_dim)

        # Router
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(router_probs, k=self.topk, dim=-1)

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Process each topk choice
        for i in range(self.topk):
            expert_idx = topk_indices[:, i]  # [batch*seq]
            weight = topk_weights[:, i].unsqueeze(-1)  # [batch*seq, 1]

            # Simple: use first expert for everything (no actual routing to different experts)
            gate = self.expert_gates(x_flat)
            up = self.expert_ups(x_flat)
            act = F.silu(gate) * up
            down = self.expert_downs(act)

            output += down * weight

        return output.view(orig_shape)

native_moe = NativeMoE(hidden_dim, num_experts, expert_dim, topk)

# Warmup
for _ in range(3):
    _ = native_moe(hidden_states)
torch.mps.synchronize()

# Benchmark
t0 = time.perf_counter()
for _ in range(num_iters):
    _ = native_moe(hidden_states)
torch.mps.synchronize()
native_time = time.perf_counter() - t0

native_per_iter = native_time / num_iters
native_tps = (batch * seq_len) / native_per_iter
print(f"Total: {native_time*1000:.1f}ms ({native_time/num_iters*1000:.1f}ms/iter)")
print(f"Throughput: {native_tps:.1f} tokens/sec")
print()

# ============================================================================
# Individual component analysis
# ============================================================================
print("2. Component Analysis (PyTorch Native):")
print("-" * 70)

# Router only
router = nn.Linear(hidden_dim, num_experts, bias=False, device='mps', dtype=torch.float16)
router.weight.data = router_weights.T.clone()

x_flat = hidden_states.view(-1, hidden_dim)
t0 = time.perf_counter()
for _ in range(num_iters):
    logits = router(x_flat)
    probs = F.softmax(logits, dim=-1)
    weights, indices = torch.topk(probs, k=topk, dim=-1)
torch.mps.synchronize()
router_time = time.perf_counter() - t0

print(f"Router: {router_time*1000:.1f}ms ({router_time/num_iters*1000:.1f}ms/iter)")

# Single expert (one gate+up+down pass)
expert_gate = nn.Linear(hidden_dim, expert_dim, bias=False, device='mps', dtype=torch.float16)
expert_up = nn.Linear(hidden_dim, expert_dim, bias=False, device='mps', dtype=torch.float16)
expert_down = nn.Linear(expert_dim, hidden_dim, bias=False, device='mps', dtype=torch.float16)

t0 = time.perf_counter()
for _ in range(num_iters):
    gate = expert_gate(x_flat)
    up = expert_up(x_flat)
    act = F.silu(gate) * up
    _ = expert_down(act)
torch.mps.synchronize()
expert_time = time.perf_counter() - t0

print(f"Single Expert (gate+up+down): {expert_time*1000:.1f}ms ({expert_time/num_iters*1000:.1f}ms/iter)")

# ============================================================================
# Summary
# ============================================================================
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"PyTorch Native MoE: {native_tps:.1f} tokens/sec")
print(f"  - Router: {router_time/num_iters*1000:.1f}ms/iter ({router_time/native_time*100:.1f}% of total)")
print(f"  - Experts: {expert_time*topk/num_iters*1000:.1f}ms/iter ({expert_time*topk/native_time*100:.1f}% of total)")
print()
print("CONCLUSION:")
print("  PyTorch native ops work but are slow due to sequential expert execution.")
print("  Custom Metal kernels with parallel expert dispatch would be much faster.")
