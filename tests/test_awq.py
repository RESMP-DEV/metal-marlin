"""Tests for AWQ (Activation-aware Weight Quantization)."""

import numpy as np

from metal_marlin.awq import (
    AWQResult,
    awq_dequantize,
    awq_quantize,
    compute_activation_stats,
    compute_salient_scaling,
    find_salient_weights,
    pack_awq_weights,
)


def test_compute_activation_stats():
    """Test activation statistics computation."""
    # Create dummy activations
    batch_size = 10
    seq_len = 128
    in_features = 512

    activations = np.random.randn(batch_size, seq_len, in_features).astype(np.float32)

    # Test mean method
    stats_mean = compute_activation_stats(activations, method="mean")
    assert stats_mean.shape == (in_features,)
    assert np.all(stats_mean >= 0)

    # Test max method
    stats_max = compute_activation_stats(activations, method="max")
    assert stats_max.shape == (in_features,)
    assert np.all(stats_max >= 0)

    # Test rms method
    stats_rms = compute_activation_stats(activations, method="rms")
    assert stats_rms.shape == (in_features,)
    assert np.all(stats_rms >= 0)

    print("  ✓ compute_activation_stats tests passed")


def test_find_salient_weights():
    """Test salient weight identification."""
    in_features = 512
    out_features = 256

    # Create weights with some large values
    weights = np.random.randn(in_features, out_features).astype(np.float32)

    # Create activation stats
    activation_stats = np.random.rand(in_features).astype(np.float32)

    salient_mask, importance = find_salient_weights(weights, activation_stats, salient_ratio=0.01)

    assert salient_mask.shape == (in_features, out_features)
    assert salient_mask.dtype == bool
    assert importance.shape == (in_features,)

    # Check that approximately 1% of weights are marked salient
    expected_salient = int(in_features * out_features * 0.01)
    actual_salient = np.sum(salient_mask)
    assert abs(actual_salient - expected_salient) <= expected_salient // 2

    print("  ✓ find_salient_weights tests passed")


def test_compute_salient_scaling():
    """Test salient weight scaling computation."""
    in_features = 512
    out_features = 256
    group_size = 128

    weights = np.random.randn(in_features, out_features).astype(np.float32)
    scales = np.abs(np.random.randn(in_features // group_size, out_features)).astype(np.float32)
    salient_mask = np.random.rand(in_features, out_features) < 0.01

    q_scale = compute_salient_scaling(weights, salient_mask, scales, group_size)

    assert q_scale.shape == (out_features,)
    assert np.all(q_scale > 0)

    print("  ✓ compute_salient_scaling tests passed")


def test_pack_awq_weights():
    """Test AWQ weight packing."""
    in_features = 512
    out_features = 256
    group_size = 128

    # Create quantized weights (INT4 symmetric range: -8 to 7)
    weights = np.random.randint(-8, 8, size=(in_features, out_features)).astype(np.int8)
    scales = np.abs(np.random.randn(in_features // group_size, out_features)).astype(np.float32)
    zeros = np.zeros((in_features // group_size, out_features), dtype=np.float32)

    packed, meta = pack_awq_weights(weights, scales, zeros, group_size)

    assert packed.dtype == np.uint32
    assert packed.shape == (in_features // 8, out_features)
    assert meta["in_features"] == in_features
    assert meta["out_features"] == out_features
    assert meta["group_size"] == group_size
    assert meta["quant_type"] == "awq_int4"

    print("  ✓ pack_awq_weights tests passed")


def test_awq_quantize_dequantize():
    """Test full AWQ quantization and dequantization pipeline."""
    in_features = 512
    out_features = 256
    group_size = 128

    # Create weights
    weights = np.random.randn(in_features, out_features).astype(np.float32)

    # Create dummy activations
    batch_size = 10
    seq_len = 128
    activations = np.random.randn(batch_size, seq_len, in_features).astype(np.float32)

    # Quantize
    result = awq_quantize(
        weights,
        activations,
        group_size=group_size,
        salient_ratio=0.01,
        activation_method="rms",
    )

    assert isinstance(result, AWQResult)
    assert result.Q.shape[0] == in_features // 8
    assert result.Q.shape[1] == out_features
    assert result.q_scale.shape == (out_features,)
    assert result.salient_ratio == 0.01

    # Dequantize
    meta = {
        "in_features": in_features,
        "out_features": out_features,
        "group_size": group_size,
        "quant_type": "awq_int4",
    }

    dequantized = awq_dequantize(result.Q, result.scales, result.zeros, result.q_scale, meta)

    assert dequantized.shape == (in_features, out_features)

    # Check reconstruction error (should be reasonable)
    error = np.mean((weights - dequantized) ** 2)
    assert error < 1.0  # Relaxed threshold for 4-bit quantization

    print("  ✓ awq_quantize/dequantize tests passed")


def test_awq_reconstruction_quality():
    """Test AWQ reconstruction quality on synthetic data."""
    in_features = 1024
    out_features = 512
    group_size = 128

    # Create structured weights (not pure random)
    # Simulate realistic weight distributions
    weights = np.random.randn(in_features, out_features).astype(np.float32) * 0.1

    # Create activations with structure
    batch_size = 20
    seq_len = 256
    activations = np.random.randn(batch_size, seq_len, in_features).astype(np.float32)
    # Add some high-magnitude activation channels
    activations[:, :, :100] *= 2.0

    # Quantize with AWQ
    result = awq_quantize(
        weights,
        activations,
        group_size=group_size,
        salient_ratio=0.01,
        activation_method="rms",
    )

    # Dequantize
    meta = {
        "in_features": in_features,
        "out_features": out_features,
        "group_size": group_size,
        "quant_type": "awq_int4",
    }

    dequantized = awq_dequantize(result.Q, result.scales, result.zeros, result.q_scale, meta)

    # Compute relative error
    weight_norm = np.linalg.norm(weights)
    error_norm = np.linalg.norm(weights - dequantized)
    relative_error = error_norm / weight_norm

    # AWQ should achieve reasonable error for 4-bit quantization
    # Note: Without real calibration data, error may be higher
    assert relative_error < 0.15

    print(f"  ✓ AWQ reconstruction quality: {relative_error:.4f} relative error (target < 0.15)")


def test_awq_edge_cases():
    """Test AWQ edge cases."""
    in_features = 512
    out_features = 256

    # Test with all-zero weights
    weights = np.zeros((in_features, out_features), dtype=np.float32)
    activations = np.random.randn(10, 128, in_features).astype(np.float32)

    result = awq_quantize(weights, activations, group_size=128, salient_ratio=0.01)

    assert isinstance(result, AWQResult)
    assert result.quantization_error == 0.0

    # Test with very large weights
    weights = np.random.randn(in_features, out_features).astype(np.float32) * 100.0
    result = awq_quantize(weights, activations, group_size=128, salient_ratio=0.01)

    assert isinstance(result, AWQResult)
    assert np.all(np.abs(result.q_scale) > 0)

    print("  ✓ AWQ edge cases tests passed")


def run_all_tests():
    """Run all AWQ tests."""
    print("\n" + "=" * 70)
    print("Running AWQ Tests")
    print("=" * 70 + "\n")

    test_compute_activation_stats()
    test_find_salient_weights()
    test_compute_salient_scaling()
    test_pack_awq_weights()
    test_awq_quantize_dequantize()
    test_awq_reconstruction_quality()
    test_awq_edge_cases()

    print("\n" + "=" * 70)
    print("All AWQ tests passed! ✓")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
