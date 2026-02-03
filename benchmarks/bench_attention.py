import os
import sys
import time
from collections.abc import Callable
from typing import Any

import torch

# Add current directory to sys.path to allow importing metal_marlin
sys.path.append(os.getcwd())

from metal_marlin.attention import scaled_dot_product_attention_metal
from metal_marlin.flash_attention_v2 import flash_attention_v2
from metal_marlin.fused_attention_mps import fused_attention


def benchmark_impl(
    name: str,
    impl_fn: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    causal: bool = False,
    warmup: int = 5,
    iterations: int = 20,
    ref_output: torch.Tensor | None = None,
) -> dict[str, Any]:

    # Warmup
    try:
        for _ in range(warmup):
            _ = impl_fn(q, k, v, mask=mask, causal=causal)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception as e:
        return {"name": name, "error": str(e), "time_ms": 0, "memory_mb": 0, "accuracy": "Error"}

    # Time measurement
    try:
        start_time = time.perf_counter()
        for _ in range(iterations):
            output = impl_fn(q, k, v, mask=mask, causal=causal)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        end_time = time.perf_counter()

        avg_time_ms = ((end_time - start_time) / iterations) * 1000

        # Memory measurement (approximate peak allocation)
        torch.mps.empty_cache()
        start_mem = torch.mps.current_allocated_memory() if torch.backends.mps.is_available() else 0
        # Run once for memory
        output = impl_fn(q, k, v, mask=mask, causal=causal)
        end_mem = torch.mps.current_allocated_memory() if torch.backends.mps.is_available() else 0
        mem_used_mb = (end_mem - start_mem) / (1024 * 1024)

        # Accuracy check
        accuracy_msg = "N/A"
        if ref_output is not None:
            # Normalize shapes if needed (e.g. for GQA or squeeze/unsqueeze)
            if output.shape != ref_output.shape:
                 # Try to match reference shape if it's just a singleton dimension diff
                 if output.numel() == ref_output.numel():
                     output = output.view_as(ref_output)

            if output.shape == ref_output.shape:
                diff = torch.norm(output.float() - ref_output.float())
                accuracy_msg = f"Diff: {diff.item():.4f}"
                if diff > 1e-2: # Tolerance
                    accuracy_msg += " (MISMATCH)"
            else:
                accuracy_msg = f"Shape mismatch: {output.shape} vs {ref_output.shape}"

    except Exception as e:
        return {"name": name, "error": str(e), "time_ms": 0, "memory_mb": 0, "accuracy": "Error"}

    return {
        "name": name,
        "time_ms": avg_time_ms,
        "memory_mb": mem_used_mb,
        "accuracy": accuracy_msg
    }

def run_benchmarks():
    print(f"Running Attention Benchmarks on {torch.device('mps') if torch.backends.mps.is_available() else 'cpu'}")

    seq_lens = [1, 512, 2048, 8192]
    batch_size = 1
    num_heads = 32
    head_dim = 128
    dtype = torch.float16
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    print(f"Config: Batch={batch_size}, Heads={num_heads}, HeadDim={head_dim}, Dtype={dtype}")
    print("-" * 120)
    print(f"{'Seq Len':<10} | {'Implementation':<25} | {'Time (ms)':<10} | {'Memory (MB)':<12} | {'Accuracy/Error':<40}")
    print("-" * 120)

    # Standard Wrapper
    def run_std(q, k, v, mask=None, causal=False):
        return scaled_dot_product_attention_metal(q, k, v, attn_mask=mask, is_causal=causal)

    # Fused Wrapper
    def run_fused(q, k, v, mask=None, causal=False):
        return fused_attention(q, k, v, mask=mask, causal=causal)

    # Flash Wrapper
    def run_flash(q, k, v, mask=None, causal=False):
        if mask is not None:
            # Flash V2 implies causal mask if causal=True, but doesn't take explicit mask tensor
            pass
        return flash_attention_v2(q, k, v, causal=causal)

    for seq_len in seq_lens:
        # Prepare inputs
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        # Standard Attention (Baseline)
        try:
            ref_out = run_std(q, k, v, causal=True)
            if torch.backends.mps.is_available():
                 torch.mps.synchronize()
        except Exception as e:
            print(f"Standard attention failed for seq_len={seq_len}: {e}")
            ref_out = None

        # 1. Standard
        res_std = benchmark_impl(
            "Standard",
            run_std,
            q, k, v,
            causal=True,
            ref_output=ref_out
        )
        print(f"{seq_len:<10} | {res_std['name']:<25} | {res_std.get('time_ms', 0):<10.2f} | {res_std.get('memory_mb', 0):<12.2f} | {res_std.get('error', res_std.get('accuracy', 'Error'))}")

        # 2. Fused QKV
        res_fused = benchmark_impl(
            "Fused QKV",
            run_fused,
            q, k, v,
            causal=True,
            ref_output=ref_out
        )
        print(f"{seq_len:<10} | {res_fused['name']:<25} | {res_fused.get('time_ms', 0):<10.2f} | {res_fused.get('memory_mb', 0):<12.2f} | {res_fused.get('error', res_fused.get('accuracy', 'Error'))}")

        # 3. Flash Attention V2
        res_flash = benchmark_impl(
            "Flash Attention V2",
            run_flash,
            q, k, v,
            causal=True,
            ref_output=ref_out
        )
        print(f"{seq_len:<10} | {res_flash['name']:<25} | {res_flash.get('time_ms', 0):<10.2f} | {res_flash.get('memory_mb', 0):<12.2f} | {res_flash.get('error', res_flash.get('accuracy', 'Error'))}")

        # 4. GQA (Grouped Query Attention)
        # Ratio 4:1 (32 heads Q, 8 heads KV)
        num_kv_heads = 8
        k_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
        v_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)

        # For reference
        k_gqa_expanded = k_gqa.repeat_interleave(num_heads // num_kv_heads, dim=1)
        v_gqa_expanded = v_gqa.repeat_interleave(num_heads // num_kv_heads, dim=1)
        try:
            ref_out_gqa = run_std(q, k_gqa_expanded, v_gqa_expanded, causal=True)
        except:
            ref_out_gqa = None

        res_gqa = benchmark_impl(
            "GQA (Flash V2)",
            run_flash,
            q, k_gqa, v_gqa,
            causal=True,
            ref_output=ref_out_gqa
        )
        print(f"{seq_len:<10} | {res_gqa['name']:<25} | {res_gqa.get('time_ms', 0):<10.2f} | {res_gqa.get('memory_mb', 0):<12.2f} | {res_gqa.get('error', res_gqa.get('accuracy', 'Error'))}")

        print("-" * 120)

if __name__ == "__main__":
    run_benchmarks()
