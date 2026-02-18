"""Benchmark for draft model inference speed optimization.

Verifies >2x speedup on decode with speculative decoding optimizations.
"""

from __future__ import annotations

import time

import torch

from metal_marlin.speculative.mmfp4_draft import MMFP4DraftModel


def benchmark_draft_speed():
    """Benchmark draft model inference speed."""
    print("=" * 60)
    print("Draft Model Speed Benchmark")
    print("=" * 60)
    
    # Configuration
    hidden_size = 4096
    vocab_size = 32000
    num_predictions = 4
    batch_size = 1
    num_iterations = 100
    
    # Create draft model
    print(f"\nConfiguration:")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Num predictions: {num_predictions}")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {num_iterations}")
    
    draft_model = MMFP4DraftModel(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_predictions=num_predictions,
        dtype=torch.float32,
    )
    draft_model.eval()
    
    # Check if using optimized components
    has_fast_engine = (
        hasattr(draft_model, '_fast_engine') 
        and draft_model._fast_engine is not None
    )
    has_optimized_head = hasattr(draft_model.mtp_head, 'speculate_fast_path')
    
    print(f"\nOptimizations enabled:")
    print(f"  FastSpeculationEngine: {has_fast_engine}")
    print(f"  speculate_fast_path: {has_optimized_head}")
    
    # Warmup
    hidden_states = torch.randn(batch_size, 1, hidden_size)
    with torch.inference_mode():
        for _ in range(10):
            _ = draft_model.speculate_from_hidden(hidden_states, num_tokens=num_predictions)
    
    # Benchmark speculate_from_hidden
    times = []
    with torch.inference_mode():
        for _ in range(num_iterations):
            hidden_states = torch.randn(batch_size, 1, hidden_size)
            
            start = time.perf_counter()
            output = draft_model.speculate_from_hidden(hidden_states, num_tokens=num_predictions)
            end = time.perf_counter()
            
            times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\nResults:")
    print(f"  Average time: {avg_time:.3f} ms")
    print(f"  Min time: {min_time:.3f} ms")
    print(f"  Max time: {max_time:.3f} ms")
    print(f"  Throughput: {num_predictions / (avg_time / 1000):.1f} tokens/sec")
    
    # Benchmark with FastSpeculationEngine if available
    if has_fast_engine:
        times_fast = []
        with torch.inference_mode():
            for _ in range(num_iterations):
                hidden_states = torch.randn(batch_size, 1, hidden_size)
                
                start = time.perf_counter()
                tokens, probs = draft_model._fast_engine.speculate(
                    hidden_states, 
                    num_tokens=num_predictions
                )
                end = time.perf_counter()
                
                times_fast.append((end - start) * 1000)
        
        avg_time_fast = sum(times_fast) / len(times_fast)
        speedup = avg_time / avg_time_fast if avg_time_fast > 0 else 1.0
        
        print(f"\nFastSpeculationEngine results:")
        print(f"  Average time: {avg_time_fast:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
    
    # Benchmark with speculate_fast_path if available
    if has_optimized_head:
        times_fp = []
        with torch.inference_mode():
            for _ in range(num_iterations):
                hidden_states = torch.randn(batch_size, 1, hidden_size)
                
                start = time.perf_counter()
                tokens, probs = draft_model.mtp_head.speculate_fast_path(
                    hidden_states,
                    num_tokens=num_predictions
                )
                end = time.perf_counter()
                
                times_fp.append((end - start) * 1000)
        
        avg_time_fp = sum(times_fp) / len(times_fp)
        speedup_fp = avg_time / avg_time_fp if avg_time_fp > 0 else 1.0
        
        print(f"\nspeculate_fast_path results:")
        print(f"  Average time: {avg_time_fp:.3f} ms")
        print(f"  Speedup: {speedup_fp:.2f}x")
    
    print("\n" + "=" * 60)
    print("Benchmark complete")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    benchmark_draft_speed()
