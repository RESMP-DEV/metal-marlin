#!/usr/bin/env python3
"""Benchmark MoE kernel variants for different batch sizes.

Measures the execution time of different kernel implementations to
determine the optimal selection strategy.
"""

import time
import torch
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metal_marlin.metal_dispatch import MetalKernelLibrary
from metal_marlin.trellis.moe_dispatch import dispatch_moe_trellis_swiglu, MoEBufferPool, create_cached_weight_buffers

def get_dummy_weights(device="mps"):
    num_experts = 8
    hidden = 2048
    intermediate = 5632
    top_k = 2
    bits = 4
    group_size = 128
    n_levels = 1 << bits
    tile_size = 16
    packed_bytes_per_tile = (tile_size * tile_size * bits + 7) // 8

    num_tiles_k_gate = (hidden + tile_size - 1) // tile_size
    num_tiles_n_gate = (intermediate + tile_size - 1) // tile_size
    num_tiles_k_down = (intermediate + tile_size - 1) // tile_size
    num_tiles_n_down = (hidden + tile_size - 1) // tile_size

    n_groups_gate = (hidden + group_size - 1) // group_size
    n_groups_down = (intermediate + group_size - 1) // group_size

    return {
        "num_experts": num_experts,
        "hidden_dim": hidden,
        "intermediate_dim": intermediate,
        "top_k": top_k,
        "bits": bits,
        "gate_weights": torch.randint(0, 256, (num_experts, num_tiles_k_gate, num_tiles_n_gate, packed_bytes_per_tile), dtype=torch.uint8, device=device),
        "gate_scales": torch.randn(num_experts, n_groups_gate, intermediate, dtype=torch.float16, device=device),
        "gate_su": torch.randn(num_experts, hidden, dtype=torch.float16, device=device),
        "gate_sv": torch.randn(num_experts, intermediate, dtype=torch.float16, device=device),
        "up_weights": torch.randint(0, 256, (num_experts, num_tiles_k_gate, num_tiles_n_gate, packed_bytes_per_tile), dtype=torch.uint8, device=device),
        "up_scales": torch.randn(num_experts, n_groups_gate, intermediate, dtype=torch.float16, device=device),
        "up_su": torch.randn(num_experts, hidden, dtype=torch.float16, device=device),
        "up_sv": torch.randn(num_experts, intermediate, dtype=torch.float16, device=device),
        "down_weights": torch.randint(0, 256, (num_experts, num_tiles_k_down, num_tiles_n_down, packed_bytes_per_tile), dtype=torch.uint8, device=device),
        "down_scales": torch.randn(num_experts, n_groups_down, hidden, dtype=torch.float16, device=device),
        "down_su": torch.randn(num_experts, intermediate, dtype=torch.float16, device=device),
        "down_sv": torch.randn(num_experts, hidden, dtype=torch.float16, device=device),
        "grid": torch.randn(n_levels, dtype=torch.float16, device=device),
    }

def benchmark():
    if not torch.backends.mps.is_available():
        print("MPS not available")
        return

    device = "mps"
    try:
        lib = MetalKernelLibrary.from_source_dir()
    except Exception as e:
        print(f"Failed to load Metal library: {e}")
        return

    w = get_dummy_weights(device)
    
    # Initialize pools
    buffer_pool = MoEBufferPool(
        device=lib.device,
        hidden_dim=w["hidden_dim"],
        max_batch=64,
        top_k_values=(w["top_k"],)
    )
    
    cached_buffers = create_cached_weight_buffers(
        device=lib.device,
        gate_weights=w["gate_weights"],
        gate_scales=w["gate_scales"],
        up_weights=w["up_weights"],
        up_scales=w["up_scales"],
        down_weights=w["down_weights"],
        down_scales=w["down_scales"],
        gate_su=w["gate_su"],
        gate_sv=w["gate_sv"],
        up_su=w["up_su"],
        up_sv=w["up_sv"],
        down_su=w["down_su"],
        down_sv=w["down_sv"],
        grid=w["grid"]
    )
    
    batch_sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64]
    kernels = [
        "decode",
        "prefill4",
        "base",
        "large_batch"
    ]
    
    print(f"{'Batch':<8} {'Kernel':<15} {'Time (ms)':<10}")
    print("-" * 40)

    results = []

    for b in batch_sizes:
        activations = torch.randn(b, w["hidden_dim"], dtype=torch.float16, device=device)
        expert_ids = torch.randint(0, w["num_experts"], (b, w["top_k"]), dtype=torch.int32, device=device)
        expert_probs = torch.rand(b, w["top_k"], dtype=torch.float16, device=device)
        
        # Warmup
        for _ in range(5):
             dispatch_moe_trellis_swiglu(
                lib=lib,
                activations=activations,
                gate_weights=None, gate_scales=None, up_weights=None, up_scales=None, down_weights=None, down_scales=None,
                gate_su=None, gate_sv=None, up_su=None, up_sv=None, down_su=None, down_sv=None, grid=None,
                expert_ids=expert_ids,
                expert_probs=expert_probs,
                hidden_dim=w["hidden_dim"],
                intermediate_dim=w["intermediate_dim"],
                num_experts=w["num_experts"],
                top_k=w["top_k"],
                bits=w["bits"],
                cached_buffers=cached_buffers,
                buffer_pool=buffer_pool,
                kernel_override="base" 
            )
        torch.mps.synchronize()
        
        best_time = float('inf')
        best_kernel = ""

        for k in kernels:
            if k == "decode" and b > 1:
                continue
                
            start = time.perf_counter()
            for _ in range(50):
                dispatch_moe_trellis_swiglu(
                    lib=lib,
                    activations=activations,
                    gate_weights=None, gate_scales=None, up_weights=None, up_scales=None, down_weights=None, down_scales=None,
                    gate_su=None, gate_sv=None, up_su=None, up_sv=None, down_su=None, down_sv=None, grid=None,
                    expert_ids=expert_ids,
                    expert_probs=expert_probs,
                    hidden_dim=w["hidden_dim"],
                    intermediate_dim=w["intermediate_dim"],
                    num_experts=w["num_experts"],
                    top_k=w["top_k"],
                    bits=w["bits"],
                    cached_buffers=cached_buffers,
                    buffer_pool=buffer_pool,
                    kernel_override=k
                )
            torch.mps.synchronize()
            avg = (time.perf_counter() - start) / 50 * 1000
            
            print(f"{b:<8} {k:<15} {avg:.3f}")
            if avg < best_time:
                best_time = avg
                best_kernel = k
        
        print(f"--> Batch {b} Best: {best_kernel} ({best_time:.3f} ms)")
        results.append({"batch": b, "best_kernel": best_kernel, "time": best_time})
        print()

    rec_file = Path(__file__).parent / "results" / "recommendations.json"
    rec_file.parent.mkdir(exist_ok=True, parents=True)
    with open(rec_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Recommendations saved to {rec_file}")

if __name__ == "__main__":
    benchmark()
