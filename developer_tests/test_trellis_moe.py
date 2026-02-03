#!/usr/bin/env python3
"""Test Trellis MoE implementation performance."""
import time

import torch

print("Testing Trellis MoE Implementation")
print("=" * 70)

# Test configuration (same as native test)
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

# Try to import Trellis
try:
    from metal_marlin.trellis.config import TrellisModelConfig
    from metal_marlin.trellis.moe import TrellisMoELayer, get_moe_dispatch_stats

    print("Trellis MoE Implementation:")
    print("-" * 70)

    # Create config
    config = TrellisModelConfig(
        hidden_size=hidden_dim,
        intermediate_size=expert_dim,
        num_experts=num_experts,
        num_experts_per_tok=topk,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_kv_heads=8,
        rope_theta=10000.0,
        vocab_size=128256,
    )

    # Create dummy weights
    layer_weights = {}
    for i in range(num_experts):
        layer_weights[f"experts.{i}.gate_proj"] = {
            "indices": torch.randint(0, 16, (hidden_dim, expert_dim // 16)),
            "scales": torch.randn(hidden_dim, expert_dim // 16, dtype=torch.float16),
            "su": torch.randint(0, 16, (expert_dim,)),
            "sv": torch.randint(0, 16, (hidden_dim,)),
            "bits": 4,
            "original_shape": (expert_dim, hidden_dim),
        }
        layer_weights[f"experts.{i}.up_proj"] = {
            "indices": torch.randint(0, 16, (hidden_dim, expert_dim // 16)),
            "scales": torch.randn(hidden_dim, expert_dim // 16, dtype=torch.float16),
            "su": torch.randint(0, 16, (expert_dim,)),
            "sv": torch.randint(0, 16, (hidden_dim,)),
            "bits": 4,
            "original_shape": (expert_dim, hidden_dim),
        }
        layer_weights[f"experts.{i}.down_proj"] = {
            "indices": torch.randint(0, 16, (hidden_dim, expert_dim // 16)),
            "scales": torch.randn(hidden_dim, expert_dim // 16, dtype=torch.float16),
            "su": torch.randint(0, 16, (expert_dim,)),
            "sv": torch.randint(0, 16, (hidden_dim,)),
            "bits": 4,
            "original_shape": (hidden_dim, expert_dim),
        }

    router_weight = torch.randn(num_experts, hidden_dim, dtype=torch.float16)

    # Create MoE layer
    moe_layer = TrellisMoELayer(
        config=config,
        layer_weights=layer_weights,
        router_weight=router_weight,
        layer_idx=0,
        device='mps',
        enable_cache=False,  # Disable cache for fair comparison
    )

    # Warmup
    for _ in range(3):
        _ = moe_layer(hidden_states)
    torch.mps.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(num_iters):
        _ = moe_layer(hidden_states)
    torch.mps.synchronize()
    trellis_time = time.perf_counter() - t0

    trellis_per_iter = trellis_time / num_iters
    trellis_tps = (batch * seq_len) / trellis_per_iter

    print(f"Total: {trellis_time*1000:.1f}ms ({trellis_per_iter*1000:.1f}ms/iter)")
    print(f"Throughput: {trellis_tps:.1f} tokens/sec")

    # Get dispatch stats
    stats = get_moe_dispatch_stats()
    print()
    print("Dispatch stats:")
    print(f"  Metal router calls: {stats['metal_router_calls']}")
    print(f"  Metal router success: {stats['metal_router_success']}")
    print(f"  CPU router fallback: {stats['cpu_router_fallback']}")
    print(f"  Total tokens processed: {stats['total_tokens_processed']}")
    print(f"  Total experts activated: {stats['total_experts_activated']}")

except ImportError as e:
    print(f"Cannot import Trellis MoE: {e}")
    print("Missing dependencies (transformers may be incompatible)")
except Exception as e:
    print(f"Error running Trellis MoE: {e}")
    import traceback
    traceback.print_exc()
