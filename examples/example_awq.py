"""Example: Basic AWQ quantization and dequantization.

This example demonstrates basic AWQ usage with synthetic data.
"""

import numpy as np

from metal_marlin import (
    awq_dequantize,
    awq_quantize,
    compute_activation_stats,
    find_salient_weights,
)

# Example 1: Basic quantization and dequantization
print("\n" + "=" * 70)
print("Example 1: Basic AWQ Quantization")
print("=" * 70 + "\n")

# Create dummy weights (simulating a linear layer)
in_features = 512
out_features = 256
weights = np.random.randn(in_features, out_features).astype(np.float32) * 0.1

print(f"Original weights shape: {weights.shape}")
print(f"Original weights dtype: {weights.dtype}")
print(f"Original weights range: [{weights.min():.4f}, {weights.max():.4f}]")

# Create dummy activation statistics
# In production, collect this from running model on calibration data
batch_size = 10
seq_len = 128
activations = np.random.randn(batch_size, seq_len, in_features).astype(np.float32)

# Simulate some high-magnitude activation channels
activations[:, :, :50] *= 2.0

# Compute activation statistics (for demonstration)
stats = compute_activation_stats(activations, method="rms")
print(f"\nActivation stats shape: {stats.shape}")
print(f"Activation stats range: [{stats.min():.4f}, {stats.max():.4f}]")

# Find salient weights (for demonstration)
salient_mask, importance = find_salient_weights(weights, stats, salient_ratio=0.01)
print(f"\nSalient weights: {np.sum(salient_mask)} / {salient_mask.size}")
print(f"  ({np.sum(salient_mask) / salient_mask.size * 100:.2f}% of weights)")

# Quantize with AWQ
print("\nQuantizing with AWQ...")
result = awq_quantize(
    weights,
    activations,
    group_size=128,
    salient_ratio=0.01,
    activation_method="rms",
)

print(f"Quantized weights shape: {result.Q.shape}")
print(f"Scales shape: {result.scales.shape}")
print(f"Zeros shape: {result.zeros.shape}")
print(f"Salient scale shape: {result.q_scale.shape}")
print(f"Quantization error: {result.quantization_error:.6f}")

# Dequantize
print("\nDequantizing...")
meta = {
    "in_features": in_features,
    "out_features": out_features,
    "group_size": 128,
    "quant_type": "awq_int4",
}
dequantized = awq_dequantize(result.Q, result.scales, result.zeros, result.q_scale, meta)

# Compute reconstruction error
error = np.mean((weights - dequantized) ** 2)
relative_error = np.linalg.norm(weights - dequantized) / np.linalg.norm(weights)

print(f"\nReconstruction error (MSE): {error:.6f}")
print(f"Relative error: {relative_error:.4f} ({relative_error * 100:.2f}%)")

# Compute compression ratio
original_bytes = weights.nbytes
quantized_bytes = (
    result.Q.nbytes + result.scales.nbytes + result.zeros.nbytes + result.q_scale.nbytes
)
compression_ratio = original_bytes / quantized_bytes

print("\nCompression:")
print(f"  Original: {original_bytes / 1024:.2f} KB")
print(f"  Quantized: {quantized_bytes / 1024:.2f} KB")
print(f"  Ratio: {compression_ratio:.2f}x")

# Example 2: Comparing different salient ratios
print("\n" + "=" * 70)
print("Example 2: Comparing Different Salient Ratios")
print("=" * 70 + "\n")

salient_ratios = [0.005, 0.01, 0.02]
results = {}

for ratio in salient_ratios:
    result = awq_quantize(
        weights,
        activations,
        group_size=128,
        salient_ratio=ratio,
        activation_method="rms",
    )

    dequantized = awq_dequantize(result.Q, result.scales, result.zeros, result.q_scale, meta)
    error = np.mean((weights - dequantized) ** 2)

    results[ratio] = {"error": error, "result": result}

    quantized_bytes = (
        result.Q.nbytes + result.scales.nbytes + result.zeros.nbytes + result.q_scale.nbytes
    )
    compression = original_bytes / quantized_bytes

    print(f"Salient ratio {ratio:.3f}:")
    print(f"  Reconstruction error: {error:.6f}")
    print(f"  Compression ratio: {compression:.2f}x")
    print(f"  Model size: {quantized_bytes / 1024:.2f} KB")

# Find best tradeoff
best_ratio = min(results.keys(), key=lambda r: results[r]["error"])
print(f"\nBest accuracy: salient_ratio={best_ratio:.3f}")

best_size = max(
    results.keys(),
    key=lambda r: original_bytes
    / (
        results[r]["result"].Q.nbytes
        + results[r]["result"].scales.nbytes
        + results[r]["result"].zeros.nbytes
        + results[r]["result"].q_scale.nbytes
    ),
)
print(f"Best compression: salient_ratio={best_size:.3f}")

# Example 3: Comparing activation methods
print("\n" + "=" * 70)
print("Example 3: Comparing Activation Methods")
print("=" * 70 + "\n")

activation_methods = ["mean", "max", "rms"]

for method in activation_methods:
    result = awq_quantize(
        weights,
        activations,
        group_size=128,
        salient_ratio=0.01,
        activation_method=method,
    )

    dequantized = awq_dequantize(result.Q, result.scales, result.zeros, result.q_scale, meta)
    error = np.mean((weights - dequantized) ** 2)

    print(f"Method '{method}': Reconstruction error = {error:.6f}")

print("\n" + "=" * 70)
print("Examples Complete!")
print("=" * 70 + "\n")
