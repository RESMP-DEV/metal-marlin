#!/usr/bin/env python3
import time

import torch

print("Testing PyTorch Native MoE Operations:")
print("=" * 60)

# Larger config hits MPS dimension limits (MPSGraph does not support tensor dims > INT_MAX)
# Use max feasible config for native PyTorch
batch, seq_len, hidden_dim = 1, 128, 2048
num_experts, expert_dim = 32, 5632
topk = 3

# Create test data
hidden_states = torch.randn(batch, seq_len, hidden_dim, device='mps', dtype=torch.float16)
router_weights = torch.randn(hidden_dim, num_experts, device='mps', dtype=torch.float16)
expert_weights_gate = torch.randn(num_experts, expert_dim, hidden_dim, device='mps', dtype=torch.float16)
expert_weights_up = torch.randn(num_experts, expert_dim, hidden_dim, device='mps', dtype=torch.float16)
expert_weights_down = torch.randn(num_experts, hidden_dim, expert_dim, device='mps', dtype=torch.float16)

# Router
t0 = time.perf_counter()
router_logits = torch.matmul(hidden_states, router_weights)
router_probs = torch.softmax(router_logits, dim=-1)
topk_weights, topk_indices = torch.topk(router_probs, k=topk, dim=-1)
torch.mps.synchronize()
router_time = time.perf_counter() - t0

# Expert dispatch (naive implementation)
t0 = time.perf_counter()
outputs = []
for i in range(topk):
    expert_idx = topk_indices[..., i]  # [batch, seq_len]

    # Simplified: use first expert for all
    # Reshape hidden_states for batched matmul: [batch*seq, hidden_dim]
    h_flat = hidden_states.view(-1, hidden_dim)
    # gate/up: [num_experts, expert_dim, hidden_dim]
    # down: [num_experts, hidden_dim, expert_dim]
    gate = torch.matmul(h_flat, expert_weights_gate[0].T)  # [b*s, expert_dim]
    up = torch.matmul(h_flat, expert_weights_up[0].T)      # [b*s, expert_dim]
    act = torch.nn.functional.silu(gate) * up              # [b*s, expert_dim]
    down = torch.matmul(act, expert_weights_down[0].T)    # [b*s, hidden_dim]
    down = down.view(batch, seq_len, hidden_dim)           # [b, s, hidden_dim]

    outputs.append(down * topk_weights[..., i].unsqueeze(-1))

final = torch.stack(outputs, dim=-2).sum(dim=-2)
torch.mps.synchronize()
expert_time = time.perf_counter() - t0

total_time = router_time + expert_time

print(f"Router: {router_time*1000:.1f}ms")
print(f"Experts: {expert_time*1000:.1f}ms")
print(f"Total: {total_time*1000:.1f}ms")
print(f"Throughput: {seq_len/total_time:.1f} tok/s")
