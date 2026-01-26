"""
Memory footprint analysis.
"""


def calculate_weight_memory(K: int, N: int, quant_type: str, group_size: int = 128) -> dict:
    """Calculate memory for a single weight matrix."""
    fp16_bytes = K * N * 2

    if quant_type == "fp4":
        packed_bytes = K * N // 2  # 4 bits per weight
        scales_bytes = (K // group_size) * N * 2  # FP16 scales
        total_bytes = packed_bytes + scales_bytes
    elif quant_type == "int4":
        packed_bytes = K * N // 2
        scales_bytes = (K // group_size) * N * 2
        zeros_bytes = (K // group_size) * N * 2  # Zero points
        total_bytes = packed_bytes + scales_bytes + zeros_bytes
    else:
        total_bytes = fp16_bytes

    return {
        "fp16_mb": fp16_bytes / 1024 / 1024,
        "quantized_mb": total_bytes / 1024 / 1024,
        "compression": fp16_bytes / total_bytes,
    }


def analyze_llama_memory():
    """Memory analysis for Llama models."""
    models = {
        "Llama-7B": {
            "hidden": 4096,
            "intermediate": 11008,
            "layers": 32,
            "heads": 32,
        },
        "Llama-13B": {
            "hidden": 5120,
            "intermediate": 13824,
            "layers": 40,
            "heads": 40,
        },
        "Llama-70B": {
            "hidden": 8192,
            "intermediate": 28672,
            "layers": 80,
            "heads": 64,
        },
    }

    for name, config in models.items():
        h = config["hidden"]
        ff = config["intermediate"]
        n_layers = config["layers"]

        # Per-layer weights
        qkv_size = h * h * 3  # Q, K, V projections
        o_size = h * h  # Output projection
        gate_up_size = h * ff * 2  # Gate and up projections
        down_size = ff * h  # Down projection

        layer_params = qkv_size + o_size + gate_up_size + down_size
        total_params = layer_params * n_layers

        fp16_gb = total_params * 2 / 1024 / 1024 / 1024
        fp4_gb = total_params * 0.5 / 1024 / 1024 / 1024  # Rough

        print(f"{name}:")
        print(f"  Parameters: {total_params / 1e9:.1f}B")
        print(f"  FP16: {fp16_gb:.1f} GB")
        print(f"  FP4: {fp4_gb:.1f} GB")
        print(f"  Compression: {fp16_gb / fp4_gb:.1f}x")
        print()


def calculate_kv_cache_memory(
    batch: int,
    heads: int,
    seq_k: int,
    head_dim: int,
    quant_type: str,
) -> dict:
    """Calculate KV cache memory footprint for a single layer."""
    kv_elements = batch * heads * seq_k * head_dim * 2  # K and V
    fp16_bytes = kv_elements * 2

    if quant_type in {"fp4", "int4"}:
        packed_bytes = kv_elements // 2  # 4 bits per value
        scales_bytes = batch * heads * seq_k * 2 * 2  # per-row scale for K and V
        total_bytes = packed_bytes + scales_bytes
    else:
        total_bytes = fp16_bytes

    return {
        "fp16_mb": fp16_bytes / 1024 / 1024,
        "quantized_mb": total_bytes / 1024 / 1024,
        "compression": fp16_bytes / total_bytes,
    }


def analyze_kv_cache_memory():
    """Report KV cache memory savings vs FP16."""
    configs = [
        (1, 32, 4096, 128),
        (1, 32, 32768, 128),
        (8, 32, 4096, 128),
    ]

    for batch, heads, seq_k, head_dim in configs:
        fp4_stats = calculate_kv_cache_memory(batch, heads, seq_k, head_dim, "fp4")
        int4_stats = calculate_kv_cache_memory(batch, heads, seq_k, head_dim, "int4")
        print(f"KV cache B={batch} H={heads} Sk={seq_k} D={head_dim}:")
        print(f"  FP16: {fp4_stats['fp16_mb']:.1f} MB")
        print(f"  FP4:  {fp4_stats['quantized_mb']:.1f} MB "
              f"({fp4_stats['compression']:.1f}x)")
        print(f"  INT4: {int4_stats['quantized_mb']:.1f} MB "
              f"({int4_stats['compression']:.1f}x)")
        print()


if __name__ == "__main__":
    analyze_llama_memory()
    analyze_kv_cache_memory()
