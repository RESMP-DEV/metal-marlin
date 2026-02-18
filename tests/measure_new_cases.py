
import torch
from tests.fixtures.synthetic_mixed_moe import create_synthetic_model, benchmark_forward

def measure():
    print("Creating model...")
    model = create_synthetic_model(device="mps")
    
    # 1. Batched Decode (Batch=8, Seq=1)
    print("Measuring Batched Decode (Batch=8, Seq=1)...")
    res_batched = benchmark_forward(
        model, batch_size=8, seq_len=1, warmup=5, iterations=20, device="mps"
    )
    print(f"Batched Decode: {res_batched.throughput_tokens_per_sec:.1f} tok/s")
    
    # 2. Long Context Prefill (Batch=1, Seq=2048)
    print("Measuring Long Context Prefill (Batch=1, Seq=2048)...")
    res_long = benchmark_forward(
        model, batch_size=1, seq_len=2048, warmup=2, iterations=5, device="mps"
    )
    print(f"Long Context Prefill: {res_long.throughput_tokens_per_sec:.1f} tok/s")

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        measure()
    else:
        print("MPS not available")
