"""Benchmark trellis dequantization and inference performance."""

import time
from contextlib import contextmanager

import torch


@contextmanager
def timer(name: str):
    torch.mps.synchronize()
    start = time.perf_counter()
    yield
    torch.mps.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    print(f"{name}: {elapsed:.2f} ms")


def benchmark_dequant(model_path: str, iterations: int = 100):
    """Benchmark dequantization performance."""
    from metal_marlin.trellis.linear import TrellisLinear
    from metal_marlin.trellis.loader import TrellisModelLoader

    loader = TrellisModelLoader(model_path)
    weights = loader.load_layer(0)

    # Get a representative weight
    weight_name = [n for n in weights if "gate_proj" in n][0]
    linear = TrellisLinear.from_trellis_weight(weights[weight_name], device="mps")

    print(f"\nWeight: {weight_name}")
    print(f"Shape: ({linear.in_features}, {linear.out_features})")

    # Warmup
    for _ in range(5):
        linear.clear_cache()
        _ = linear.dequantize(use_cache=False)

    # Benchmark
    times = []
    for _ in range(iterations):
        linear.clear_cache()
        torch.mps.synchronize()
        start = time.perf_counter()
        _ = linear.dequantize(use_cache=False)
        torch.mps.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    avg = sum(times) / len(times)
    print(f"Dequant time: {avg:.2f} ms (avg over {iterations} runs)")

    # Calculate throughput
    elements = linear.in_features * linear.out_features
    throughput = elements / (avg / 1000) / 1e9
    print(f"Throughput: {throughput:.2f} G elements/s")


def benchmark_forward(model_path: str, batch_sizes: list = [1, 4, 16, 64]):
    """Benchmark forward pass with different batch sizes."""
    from metal_marlin.trellis.linear import TrellisLinear
    from metal_marlin.trellis.loader import TrellisModelLoader

    loader = TrellisModelLoader(model_path)
    weights = loader.load_layer(0)

    weight_name = [n for n in weights if "gate_proj" in n][0]
    linear = TrellisLinear.from_trellis_weight(weights[weight_name], device="mps")

    print("\nForward pass benchmark")
    print(f"Weight shape: ({linear.in_features}, {linear.out_features})")

    for batch_size in batch_sizes:
        x = torch.randn(batch_size, linear.in_features, dtype=torch.float16, device="mps")

        # Warmup (with caching)
        for _ in range(5):
            _ = linear(x)

        # Benchmark
        torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            _ = linear(x)
        torch.mps.synchronize()
        elapsed = (time.perf_counter() - start) * 1000 / 100

        tokens_per_sec = batch_size / (elapsed / 1000)
        print(f"Batch {batch_size:3d}: {elapsed:.2f} ms ({tokens_per_sec:.0f} tok/s)")


def benchmark_moe(model_path: str):
    """Benchmark MoE layer performance."""
    from metal_marlin.trellis.moe import TrellisMoEConfig, TrellisMoELayer

    from metal_marlin.trellis.loader import TrellisModelLoader

    loader = TrellisModelLoader(model_path)
    config = TrellisMoEConfig()
    router_weight = torch.randn(config.num_experts, config.hidden_size)

    weights = loader.load_layer(2)
    moe = TrellisMoELayer(config, weights, router_weight, layer_idx=2, device="mps")

    print("\nMoE layer benchmark (64 experts, top-8)")

    for seq_len in [1, 8, 32, 128]:
        x = torch.randn(1, seq_len, 2048, dtype=torch.float16, device="mps")

        # Warmup
        for _ in range(3):
            _ = moe(x)

        # Benchmark
        torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            _ = moe(x)
        torch.mps.synchronize()
        elapsed = (time.perf_counter() - start) * 1000 / 20

        tokens_per_sec = seq_len / (elapsed / 1000)
        print(f"Seq {seq_len:3d}: {elapsed:.2f} ms ({tokens_per_sec:.0f} tok/s)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/GLM-4.7-Flash-EXL3-3bpw")
    args = parser.parse_args()

    benchmark_dequant(args.model)
    benchmark_forward(args.model)

    try:
        benchmark_moe(args.model)
    except Exception as e:
        print(f"\nMoE benchmark skipped: {e}")
