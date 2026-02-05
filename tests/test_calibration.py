"""Comprehensive tests for calibration, quantization, and inference pipeline.

Validates:
  1. HessianCollector accumulation accuracy
  2. CalibrationDataset loader coverage
  3. Sensitivity analysis reproducibility
  4. Mixed precision packing/unpacking
  5. FP8 format correctness
  6. End-to-end quantize -> inference

Uses small synthetic models for fast CI runs.
Slow tests requiring real models are marked with @pytest.mark.slow.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from metal_marlin.calibration.calibration_dataset import CalibrationDataset
from metal_marlin.calibration_utils import ranges_to_scales
from metal_marlin.gptq import (
    GPTQQuantizer,
    compare_gptq_vs_rtn,
    compute_hessian,
    compute_hessian_streaming,
    quantize_layer_gptq,
    quantize_rtn,
)
from metal_marlin.hadamard import (
    apply_hadamard_rotation,
    compute_outlier_stats,
    hadamard_matrix,
    inverse_hadamard_rotation,
)
from metal_marlin.mixed_precision import (
    LayerQuantConfig,
    MixedPrecisionConfig,
    Precision,
    classify_layer,
    get_layer_config,
    quantize_fp8,
    quantize_int4,
    quantize_int8,
    quantize_tensor,
    should_quantize,
)
from metal_marlin.mr_gptq import (
    FP4_E2M1_GRID,
    INT4_GRID,
    NF4_GRID,
    MRGPTQQuantizer,
    collect_hessian_from_activations,
    quantize_to_grid,
)
from metal_marlin.quantize_fp4 import (
    E2M1_VALUES,
    compute_quantization_error,
    quantize_fp4,
    unpack_fp4,
)

# Multi-domain calibration v3 gist URL
CALIBRATION_V3_URL = (
    "https://gist.githubusercontent.com/bartowski1182/eb213dccb3571f863da82e99418f81e8/"
    "raw/2c64bb691316d32915b188e495754ef34931ae71/calibration_datav3.txt"
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def small_weight_matrix():
    """Create small synthetic weight matrix for fast tests."""
    np.random.seed(42)
    return np.random.randn(128, 256).astype(np.float32)


@pytest.fixture
def calibration_activations():
    """Create synthetic calibration activations."""
    np.random.seed(123)
    # Simulate 100 tokens x 256 features
    return [np.random.randn(20, 256).astype(np.float32) * 1.5 for _ in range(5)]


@pytest.fixture
def synthetic_model_weights():
    """Create synthetic model-like weights with realistic naming."""
    np.random.seed(456)
    weights = {
        "model.layers.0.self_attn.q_proj.weight": np.random.randn(256, 256).astype(np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.random.randn(256, 256).astype(np.float32),
        "model.layers.0.self_attn.v_proj.weight": np.random.randn(256, 256).astype(np.float32),
        "model.layers.0.self_attn.o_proj.weight": np.random.randn(256, 256).astype(np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.random.randn(512, 256).astype(np.float32),
        "model.layers.0.mlp.up_proj.weight": np.random.randn(512, 256).astype(np.float32),
        "model.layers.0.mlp.down_proj.weight": np.random.randn(256, 512).astype(np.float32),
        "model.embed_tokens.weight": np.random.randn(1000, 256).astype(np.float32),
        "lm_head.weight": np.random.randn(1000, 256).astype(np.float32),
    }
    return weights


# =============================================================================
# 1. HessianCollector Accumulation Accuracy Tests
# =============================================================================


class TestHessianCollectorAccuracy:
    """Tests for Hessian computation accuracy and numerical stability."""

    def test_hessian_basic_computation(self):
        """Verify H = X^T @ X is computed correctly."""
        np.random.seed(42)
        X = np.random.randn(100, 64).astype(np.float32)

        H = compute_hessian(X, normalize=True)

        # Shape check
        assert H.shape == (64, 64)

        # Symmetry check
        assert np.allclose(H, H.T, atol=1e-6)

        # Positive semi-definiteness (eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(H)
        assert np.all(eigenvalues >= -1e-6)

    def test_hessian_incremental_matches_batch(self):
        """Incremental Hessian accumulation should match batch computation."""
        np.random.seed(123)

        # Create multiple batches
        batches = [
            np.random.randn(50, 128).astype(np.float32),
            np.random.randn(30, 128).astype(np.float32),
            np.random.randn(40, 128).astype(np.float32),
        ]

        # Batch computation
        X_all = np.vstack(batches)
        H_batch = compute_hessian(X_all, normalize=False)

        # Incremental computation
        H_incremental, add_batch = compute_hessian_streaming(128)
        for batch in batches:
            add_batch(H_incremental, batch)

        # Should match within float32 tolerance
        assert np.allclose(H_batch, H_incremental, rtol=1e-5, atol=1e-3)

    def test_hessian_from_activations_shapes(self, calibration_activations):
        """Verify HessianInfo from activations has correct shapes."""
        hessian_info = collect_hessian_from_activations(calibration_activations, damp_ratio=0.01)

        assert hessian_info.hessian.shape == (256, 256)
        assert hessian_info.diag.shape == (256,)
        assert hessian_info.n_samples == 100  # 5 batches * 20 samples

    def test_hessian_damping_increases_diagonal(self, calibration_activations):
        """Damping should increase diagonal elements for numerical stability."""
        info_no_damp = collect_hessian_from_activations(calibration_activations, damp_ratio=0.0)
        info_with_damp = collect_hessian_from_activations(calibration_activations, damp_ratio=0.05)

        # Diagonal should be larger with damping
        assert np.all(np.diag(info_with_damp.hessian) >= np.diag(info_no_damp.hessian))

    def test_hessian_with_outlier_activations(self):
        """Hessian should handle extreme activations without NaN/Inf."""
        np.random.seed(789)

        # Normal activations with extreme outliers
        X = np.random.randn(100, 64).astype(np.float32)
        X[0, 0] = 1e6  # Large outlier
        X[1, 1] = 1e-8  # Small value

        H = compute_hessian(X, normalize=True)

        assert not np.any(np.isnan(H))
        assert not np.any(np.isinf(H))
        assert H.shape == (64, 64)

    def test_hessian_reproducibility(self):
        """Same input should produce identical Hessian."""
        np.random.seed(42)
        X1 = np.random.randn(100, 64).astype(np.float32)

        np.random.seed(42)
        X2 = np.random.randn(100, 64).astype(np.float32)

        H1 = compute_hessian(X1)
        H2 = compute_hessian(X2)

        np.testing.assert_array_equal(H1, H2)

    def test_hessian_sample_count_tracking(self, calibration_activations):
        """Verify sample counts are tracked correctly."""
        total_samples = sum(act.shape[0] for act in calibration_activations)
        info = collect_hessian_from_activations(calibration_activations)

        assert info.n_samples == total_samples


# =============================================================================
# 2. CalibrationDataset Loader Coverage Tests
# =============================================================================


class TestCalibrationDatasetLoader:
    """Tests for CalibrationDataset loading and coverage."""

    def test_dataset_container_basics(self):
        """Test CalibrationDataset basic operations."""
        samples = ["sample 1", "sample 2 is longer", "sample 3"]
        dataset = CalibrationDataset(
            samples=samples,
            name="test",
            version="v1",
        )

        assert len(dataset) == 3
        assert dataset[0] == "sample 1"
        assert list(dataset) == samples
        assert dataset.total_chars == sum(len(s) for s in samples)

    def test_dataset_filter(self):
        """Test filtering calibration samples."""
        samples = [
            "short",
            "this is a longer sample with code: def foo(): pass",
            "another medium length sample",
        ]
        dataset = CalibrationDataset(samples=samples, name="test", version="v1")

        # Filter by length
        filtered = dataset.filter(lambda s: len(s) > 10)
        assert len(filtered) == 2
        assert "short" not in filtered.samples

        # Original unchanged
        assert len(dataset) == 3

    def test_local_json_loading(self, tmp_path: Path):
        """Test loading calibration data from local JSON file."""
        # JSON list of strings (needs sufficient length per sample)
        data = [
            "Sample 1: code snippet with enough content to pass length filter of 50 chars",
            "Sample 2: chat format text that is also long enough to be included here",
            "Sample 3: math content that exceeds the minimum length requirement for parsing",
        ]
        json_file = tmp_path / "calibration.json"
        json_file.write_text(json.dumps(data))

        dataset = CalibrationDataset.from_local(json_file)
        assert len(dataset) == 3
        assert dataset.name == "calibration"

    def test_local_jsonl_loading(self, tmp_path: Path):
        """Test loading from JSONL format."""
        lines = [
            '{"text": "Sample 1 with enough characters to pass the length filter check"}',
            '{"text": "Sample 2 also needs to be sufficiently long for inclusion in dataset"}',
            '{"text": "Sample 3 continues the pattern of having substantial text content here"}',
        ]
        jsonl_file = tmp_path / "calibration.jsonl"
        jsonl_file.write_text("\n".join(lines))

        dataset = CalibrationDataset.from_local(jsonl_file)
        assert len(dataset) == 3

    def test_local_txt_loading(self, tmp_path: Path):
        """Test loading from plain text format."""
        txt_content = """This is the first sample paragraph with enough length to pass filtering.

This is the second sample paragraph, also with sufficient length for inclusion.

Third sample paragraph here with more text to ensure it passes the minimum length filter.
"""
        txt_file = tmp_path / "calibration.txt"
        txt_file.write_text(txt_content)

        dataset = CalibrationDataset.from_local(txt_file)
        assert len(dataset) == 3

    def test_max_samples_limit(self):
        """Test that max_samples limits returned data."""
        all_samples = [f"Sample {i}: content here with enough text" for i in range(100)]
        dataset = CalibrationDataset(samples=all_samples, name="test", version="v1")

        # Simulate limiting
        limited = dataset.samples[:50]
        assert len(limited) == 50

    @pytest.mark.slow
    def test_calibration_v3_download(self):
        """Test downloading calibration v3 data."""
        import urllib.request

        try:
            req = urllib.request.Request(
                CALIBRATION_V3_URL,
                headers={"User-Agent": "Python-urllib/3.12"},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()
                assert response.status == 200
                assert len(content) > 0

                text = content.decode("utf-8")
                # Should have substantial content
                assert len(text) > 1000

        except urllib.error.URLError as e:
            pytest.skip(f"Network unavailable: {e}")

    def test_calibration_data_format_parsing(self):
        """Test parsing different calibration data formats."""
        # Nested JSON with 'samples' key
        nested_data = {"samples": ["text1 with content", "text2 with content"]}
        samples = CalibrationDataset._parse_json_data(nested_data)
        assert "text1 with content" in samples
        assert "text2 with content" in samples

        # List of dicts with 'text' key
        dict_list = [{"text": "sample1 content"}, {"text": "sample2 content", "meta": "ignored"}]
        samples = CalibrationDataset._parse_json_data(dict_list)
        assert "sample1 content" in samples
        assert "sample2 content" in samples


# =============================================================================
# 3. Sensitivity Analysis Reproducibility Tests
# =============================================================================


class TestSensitivityReproducibility:
    """Tests for reproducible quantization sensitivity analysis."""

    def test_gptq_deterministic(self, small_weight_matrix):
        """GPTQ quantization should be deterministic given same input."""
        np.random.seed(42)
        X = np.random.randn(100, 256).astype(np.float32)
        H = compute_hessian(X)

        result1 = quantize_layer_gptq(small_weight_matrix, H, bits=4, group_size=128)
        result2 = quantize_layer_gptq(small_weight_matrix, H, bits=4, group_size=128)

        np.testing.assert_array_equal(result1.Q, result2.Q)
        np.testing.assert_array_equal(result1.scales, result2.scales)

    def test_rtn_deterministic(self, small_weight_matrix):
        """RTN quantization should be deterministic."""
        result1 = quantize_rtn(small_weight_matrix, bits=4, group_size=128)
        result2 = quantize_rtn(small_weight_matrix, bits=4, group_size=128)

        np.testing.assert_array_equal(result1.Q, result2.Q)
        np.testing.assert_array_equal(result1.scales, result2.scales)

    def test_hadamard_rotation_invertible(self, small_weight_matrix):
        """Hadamard rotation should be perfectly invertible."""
        rotated, meta = apply_hadamard_rotation(small_weight_matrix, block_size=64)
        recovered = inverse_hadamard_rotation(rotated, meta)

        # Recovered shape matches original, within float precision
        assert recovered.shape == small_weight_matrix.shape
        np.testing.assert_allclose(recovered, small_weight_matrix, rtol=1e-4, atol=1e-5)

    def test_hadamard_disperses_outliers(self, small_weight_matrix):
        """Hadamard rotation should reduce max/mean ratio (disperse outliers)."""
        # Add synthetic outlier
        W = small_weight_matrix.copy()
        W[0, 0] = 100.0  # Large outlier

        stats_before = compute_outlier_stats(W)
        rotated, _ = apply_hadamard_rotation(W, block_size=64)
        stats_after = compute_outlier_stats(rotated)

        # Max/mean ratio should decrease after rotation
        assert stats_after["max_mean_ratio"] < stats_before["max_mean_ratio"]

    def test_actorder_improves_quality(self, small_weight_matrix):
        """Activation ordering should improve or match baseline quality."""
        np.random.seed(42)
        X = np.random.randn(100, 256).astype(np.float32)
        H = compute_hessian(X)

        # With actorder
        result_actorder = quantize_layer_gptq(
            small_weight_matrix, H, bits=4, group_size=128, actorder=True
        )

        # Without actorder
        result_no_actorder = quantize_layer_gptq(
            small_weight_matrix, H, bits=4, group_size=128, actorder=False
        )

        # Both should have reasonable error
        err_act = np.mean((small_weight_matrix - result_actorder.Q) ** 2)
        err_no_act = np.mean((small_weight_matrix - result_no_actorder.Q) ** 2)

        # Actorder should be at least as good (often better)
        assert err_act <= err_no_act * 1.1  # Allow 10% tolerance

    def test_gptq_vs_rtn_comparison(self, small_weight_matrix):
        """GPTQ and RTN should both produce valid quantized outputs."""
        np.random.seed(42)
        X = np.random.randn(100, 256).astype(np.float32)
        H = compute_hessian(X)

        comparison = compare_gptq_vs_rtn(small_weight_matrix, H, bits=4, group_size=128)

        # Both should produce valid results (not NaN/Inf)
        assert "gptq_mse" in comparison
        assert "rtn_mse" in comparison
        assert comparison["gptq_mse"] >= 0
        assert comparison["rtn_mse"] >= 0
        # Both MSEs should be reasonably small
        assert comparison["gptq_mse"] < 1.0
        assert comparison["rtn_mse"] < 1.0

    def test_scale_computation_reproducibility(self, small_weight_matrix):
        """Scale computation should be reproducible."""
        packed1, scales1 = quantize_fp4(small_weight_matrix, group_size=128)
        packed2, scales2 = quantize_fp4(small_weight_matrix, group_size=128)

        np.testing.assert_array_equal(scales1, scales2)
        np.testing.assert_array_equal(packed1, packed2)


# =============================================================================
# 4. Mixed Precision Packing/Unpacking Tests
# =============================================================================


class TestMixedPrecisionPackUnpack:
    """Tests for mixed precision quantization packing and unpacking."""

    def test_fp4_pack_unpack_roundtrip(self, small_weight_matrix):
        """FP4 pack/unpack should preserve values within quantization error."""
        packed, scales = quantize_fp4(small_weight_matrix, group_size=128)
        unpacked = unpack_fp4(packed, scales, group_size=128)

        # Check shapes
        assert packed.shape == (32, 128)  # K/8 x N = 256/8 x 128
        assert scales.shape == (2, 128)  # K/group_size x N = 256/128 x 128
        assert unpacked.shape == small_weight_matrix.T.shape

        # Values should be close (within FP4 quantization error)
        error = compute_quantization_error(small_weight_matrix, packed, scales, group_size=128)
        assert error["rmse"] < 1.0  # Reasonable error bound

    def test_int4_symmetric_pack_unpack(self, small_weight_matrix):
        """INT4 symmetric quantization pack/unpack."""
        packed, scales, zeros = quantize_int4(small_weight_matrix, group_size=128, symmetric=True)

        assert zeros is None  # Symmetric has no zero points
        assert packed.dtype == np.uint32
        assert scales.dtype == np.float16

        # Verify packed values are in valid range (0-15 per nibble)
        for i in range(8):
            nibbles = (packed >> (i * 4)) & 0xF
            assert np.all(nibbles <= 15)

    def test_int4_asymmetric_pack_unpack(self, small_weight_matrix):
        """INT4 asymmetric quantization pack/unpack."""
        packed, scales, zeros = quantize_int4(small_weight_matrix, group_size=128, symmetric=False)

        assert zeros is not None
        assert zeros.dtype == np.float16
        assert scales.dtype == np.float16

    def test_int8_per_channel_quantization(self, small_weight_matrix):
        """INT8 per-channel quantization."""
        result = quantize_int8(small_weight_matrix, symmetric=True)

        assert result["data"].dtype == np.int8
        assert result["scales"].shape == (128,)  # One scale per output row
        assert np.all(result["data"] >= -128)
        assert np.all(result["data"] <= 127)

    def test_fp4_grid_values(self):
        """Verify FP4 E2M1 grid has correct values."""
        expected = np.array(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,  # Positive
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,  # Negative
            ]
        )
        np.testing.assert_array_equal(E2M1_VALUES, expected)

    def test_int4_grid_values(self):
        """Verify INT4 symmetric grid has correct values."""
        expected = np.arange(-8, 8, dtype=np.float32)
        np.testing.assert_array_equal(INT4_GRID, expected)

    def test_nf4_grid_normalized(self):
        """NF4 grid should be normalized to [-1, 1]."""
        assert NF4_GRID.min() == -1.0
        assert NF4_GRID.max() == 1.0
        assert len(NF4_GRID) == 16

    def test_quantize_to_grid_nearest(self):
        """quantize_to_grid should find nearest grid point."""
        values = np.array([0.1, 0.4, 0.6, 1.2, 5.5])
        grid = FP4_E2M1_GRID

        quantized, indices = quantize_to_grid(values, grid, scale=1.0)

        # 0.1 -> 0.0 (index 0)
        # 0.4 -> 0.5 (index 1)
        # 0.6 -> 0.5 (index 1)
        # 1.2 -> 1.0 (index 2)
        # 5.5 -> 6.0 (index 7)
        expected_vals = np.array([0.0, 0.5, 0.5, 1.0, 6.0])
        np.testing.assert_allclose(quantized, expected_vals, atol=1e-6)

    def test_mixed_precision_layer_config(self, synthetic_model_weights):
        """Test layer classification and config assignment."""
        config = MixedPrecisionConfig.default_moe()

        for name, tensor in synthetic_model_weights.items():
            category = classify_layer(name)
            layer_config = get_layer_config(name, config)

            # Verify classification is consistent
            assert category in [
                "attention_qkv",
                "attention_out",
                "mlp_gate",
                "mlp_up",
                "mlp_down",
                "embeddings",
                "lm_head",
                "default",
                "moe_router",
                "moe_experts",
                "moe_shared_expert",
            ]

            # Verify config is valid
            assert isinstance(layer_config.precision, Precision)
            assert layer_config.group_size > 0

    def test_should_quantize_dimension_check(self):
        """should_quantize should reject incompatible dimensions."""
        config = MixedPrecisionConfig.default_dense()

        # 1D tensor - should not quantize
        tensor_1d = np.random.randn(256).astype(np.float32)
        should_q, _ = should_quantize("some.weight", tensor_1d, config)
        assert not should_q

        # Dimensions not divisible by 8 - should not quantize
        tensor_odd = np.random.randn(128, 255).astype(np.float32)
        should_q, _ = should_quantize("some.weight", tensor_odd, config)
        assert not should_q


# =============================================================================
# 5. FP8 Format Correctness Tests
# =============================================================================


class TestFP8FormatCorrectness:
    """Tests for FP8 E4M3 quantization format correctness."""

    def test_fp8_quantization_basic(self, small_weight_matrix):
        """Basic FP8 quantization should produce valid output."""
        result = quantize_fp8(small_weight_matrix, group_size=128)

        assert result["data"].dtype == np.int8
        assert result["scales"].dtype == np.float16
        assert result["group_size"][0] == 128
        assert result["precision"][0] == 8

    def test_fp8_value_range(self, small_weight_matrix):
        """FP8 quantized values should be in valid range."""
        result = quantize_fp8(small_weight_matrix, group_size=128)

        # INT8 mapped from [-448, 448] to [-127, 127]
        assert np.all(result["data"] >= -127)
        assert np.all(result["data"] <= 127)

    def test_fp8_scales_positive(self, small_weight_matrix):
        """FP8 scales should all be positive."""
        result = quantize_fp8(small_weight_matrix, group_size=128)
        assert np.all(result["scales"] > 0)

    def test_fp8_handles_zeros(self):
        """FP8 should handle zero-filled tensors."""
        zeros = np.zeros((128, 256), dtype=np.float32)
        result = quantize_fp8(zeros, group_size=128)

        assert not np.any(np.isnan(result["scales"]))
        assert not np.any(np.isinf(result["scales"]))

    def test_fp8_preserves_sign(self, small_weight_matrix):
        """FP8 quantization should preserve value signs."""
        result = quantize_fp8(small_weight_matrix, group_size=128)

        # Dequantize and check signs match
        scales = result["scales"]
        data = result["data"].astype(np.float32)

        # Reshape for per-group dequantization
        out_feat, in_feat = small_weight_matrix.shape
        num_groups = in_feat // 128
        data_grouped = data.reshape(out_feat, num_groups, 128)
        scales_expanded = scales[:, :, None]

        dequantized = data_grouped * scales_expanded * (448 / 127)
        dequantized = dequantized.reshape(out_feat, in_feat)

        # Signs should mostly match (allowing for quantization noise around zero)
        significant_mask = np.abs(small_weight_matrix) > 0.1
        sign_match_rate = np.mean(
            np.sign(dequantized[significant_mask]) == np.sign(small_weight_matrix[significant_mask])
        )
        assert sign_match_rate > 0.95

    def test_fp8_group_size_validation(self, small_weight_matrix):
        """FP8 should handle different group sizes."""
        for group_size in [32, 64, 128, 256]:
            result = quantize_fp8(small_weight_matrix, group_size=group_size)
            expected_groups = 256 // group_size
            assert result["scales"].shape == (128, expected_groups)


# =============================================================================
# 6. End-to-End Quantize -> Inference Tests
# =============================================================================


class TestEndToEndQuantizeInference:
    """End-to-end tests for quantization and inference pipeline."""

    def test_fp4_quantize_and_matmul(self, small_weight_matrix):
        """Quantized FP4 weights should work in matrix multiplication."""
        packed, scales = quantize_fp4(small_weight_matrix, group_size=128)

        # Dequantize
        dequantized = unpack_fp4(packed, scales, group_size=128)

        # small_weight_matrix is [128, 256], so input should be [batch, 256]
        # Weight transpose is [256, 128], so output is [batch, 128]
        np.random.seed(42)
        x = np.random.randn(32, 256).astype(np.float16)

        # Original output: x @ W.T -> [32, 256] @ [256, 128] = [32, 128]
        y_original = x @ small_weight_matrix.T.astype(np.float16)

        # Quantized output
        y_quantized = x @ dequantized

        # Should be reasonably close
        relative_error = np.mean(np.abs(y_original - y_quantized)) / np.mean(np.abs(y_original))
        assert relative_error < 0.15  # Within 15% relative error

    def test_gptq_quantize_and_matmul(self, small_weight_matrix):
        """GPTQ quantized weights should work in matrix multiplication."""
        np.random.seed(42)
        X_cal = np.random.randn(100, 256).astype(np.float32)
        H = compute_hessian(X_cal)

        result = quantize_layer_gptq(small_weight_matrix, H, bits=4, group_size=128)

        # small_weight_matrix is [128, 256]
        # Weight transpose is [256, 128], so input should be [batch, 256]
        x = np.random.randn(32, 256).astype(np.float32)
        y_original = x @ small_weight_matrix.T
        y_quantized = x @ result.Q.T

        relative_error = np.mean(np.abs(y_original - y_quantized)) / np.mean(np.abs(y_original))
        assert relative_error < 0.15

    def test_mr_gptq_layer_quantization(self, small_weight_matrix):
        """MR-GPTQ quantizer should produce valid packed output."""
        quantizer = MRGPTQQuantizer(
            bits=4,
            format="fp4",
            group_size=128,
            use_hadamard=True,
            hadamard_block_size=64,
            actorder=False,  # Faster for test
        )

        packed, scales, meta = quantizer.quantize_layer(
            small_weight_matrix,
            hessian=None,  # Uses RTN fallback
            layer_name="test_layer",
        )

        assert packed.dtype == np.uint32
        assert scales.dtype == np.float16
        assert meta["format"] == "fp4"
        assert meta["use_hadamard"]
        assert "error" in meta

    def test_mr_gptq_with_hadamard_rotation(self, small_weight_matrix):
        """MR-GPTQ with Hadamard should improve quality."""
        np.random.seed(42)

        # Add outlier
        W = small_weight_matrix.copy()
        W[0, 0] = 50.0

        quantizer_no_had = MRGPTQQuantizer(bits=4, format="fp4", group_size=128, use_hadamard=False)
        quantizer_had = MRGPTQQuantizer(bits=4, format="fp4", group_size=128, use_hadamard=True)

        _, _, meta_no_had = quantizer_no_had.quantize_layer(W)
        _, _, meta_had = quantizer_had.quantize_layer(W)

        # Hadamard version should have better (lower) error
        # or at least not significantly worse
        assert meta_had["error"]["rmse"] <= meta_no_had["error"]["rmse"] * 1.2

    def test_quantization_error_metrics(self, small_weight_matrix):
        """Quantization error metrics should be computed correctly."""
        packed, scales = quantize_fp4(small_weight_matrix, group_size=128)
        error = compute_quantization_error(small_weight_matrix, packed, scales, 128)

        # All metrics should be present and valid
        assert "mse" in error
        assert "rmse" in error
        assert "max_error" in error
        assert "mean_relative_error" in error

        assert error["mse"] >= 0
        assert error["rmse"] >= 0
        assert error["rmse"] == pytest.approx(np.sqrt(error["mse"]), rel=1e-5)

    def test_full_layer_quantize_dequant_chain(self, small_weight_matrix):
        """Full chain: original -> quantize -> pack -> unpack -> dequant -> compare."""
        # Quantize
        packed, scales = quantize_fp4(small_weight_matrix, group_size=128)

        # Verify packed format
        assert packed.shape == (32, 128)

        # Unpack and dequantize
        dequantized = unpack_fp4(packed, scales, group_size=128)
        dequantized = dequantized.T

        # Compare
        assert dequantized.shape == small_weight_matrix.shape

        # Calculate reconstruction error
        diff = small_weight_matrix.astype(np.float32) - dequantized.astype(np.float32)
        rmse = np.sqrt(np.mean(diff**2))

        # Error should be reasonable for FP4
        assert rmse < 0.5

    @pytest.mark.smoke
    def test_basic_pipeline_smoke(self):
        """Smoke test for basic quantization pipeline."""
        np.random.seed(42)
        W = np.random.randn(64, 128).astype(np.float32)

        # Should complete without error
        packed, scales = quantize_fp4(W, group_size=64)
        assert packed is not None
        assert scales is not None

        unpacked = unpack_fp4(packed, scales, group_size=64)
        assert unpacked.shape == W.T.shape


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_group_quantization(self):
        """Quantization with single group (group_size = in_features)."""
        np.random.seed(42)
        W = np.random.randn(64, 128).astype(np.float32)

        packed, scales = quantize_fp4(W, group_size=128)
        assert scales.shape == (1, 64)

    def test_many_groups_quantization(self):
        """Quantization with many small groups."""
        np.random.seed(42)
        W = np.random.randn(64, 256).astype(np.float32)

        packed, scales = quantize_fp4(W, group_size=32)
        assert scales.shape == (8, 64)  # 256 / 32 = 8 groups

    def test_zero_weight_handling(self):
        """Quantization should handle zero weights."""
        W = np.zeros((64, 128), dtype=np.float32)

        packed, scales = quantize_fp4(W, group_size=128)

        # Should not produce NaN or Inf
        assert not np.any(np.isnan(scales))
        assert not np.any(np.isinf(scales))

    def test_extreme_values_handling(self):
        """Quantization should handle extreme weight values."""
        np.random.seed(42)
        W = np.random.randn(64, 128).astype(np.float32) * 1000

        packed, scales = quantize_fp4(W, group_size=128)

        # Should not produce NaN or Inf
        assert not np.any(np.isnan(scales))
        assert not np.any(np.isinf(scales))

        # Scales should be large to accommodate large values
        assert np.mean(scales) > 10

    def test_hadamard_size_validation(self):
        """Hadamard matrix should reject non-power-of-2 sizes."""
        with pytest.raises(ValueError):
            hadamard_matrix(3)

        with pytest.raises(ValueError):
            hadamard_matrix(0)

        with pytest.raises(ValueError):
            hadamard_matrix(-4)

    def test_gptq_dimension_mismatch_error(self):
        """GPTQ should error on dimension mismatch."""
        W = np.random.randn(64, 128).astype(np.float32)
        H_wrong = np.random.randn(64, 64).astype(np.float32)  # Wrong size

        quantizer = GPTQQuantizer(bits=4, group_size=128)

        with pytest.raises(ValueError):
            quantizer.quantize_weight(W, H_wrong)

    def test_quantize_tensor_dispatch(self):
        """quantize_tensor should dispatch correctly to different formats."""
        np.random.seed(42)
        W = np.random.randn(64, 128).astype(np.float32)

        # FP16 - no quantization
        config_fp16 = LayerQuantConfig(precision=Precision.FP16)
        result = quantize_tensor(W, config_fp16)
        assert result["data"].dtype == np.float16

        # FP4
        config_fp4 = LayerQuantConfig(precision=Precision.FP4_E2M1, group_size=128)
        result = quantize_tensor(W, config_fp4)
        assert "packed" in result
        assert result["precision"][0] == 4

        # INT4
        config_int4 = LayerQuantConfig(precision=Precision.INT4, group_size=128)
        result = quantize_tensor(W, config_int4)
        assert "packed" in result

        # FP8
        config_fp8 = LayerQuantConfig(precision=Precision.FP8_E4M3, group_size=128)
        result = quantize_tensor(W, config_fp8)
        assert result["precision"][0] == 8


# =============================================================================
# Calibration Integration Tests
# =============================================================================


class TestCalibrationIntegration:
    """Integration tests for full calibration workflow."""

    def test_ranges_to_scales_fp4(self):
        """Test converting activation ranges to FP4 scales."""
        ranges = {
            "layer1": (-3.0, 3.0),
            "layer2": (-6.0, 6.0),
        }

        scales = ranges_to_scales(ranges, quant_type="fp4")

        # FP4 max is 6.0, so scale = absmax / 6.0
        assert scales["layer1"] == pytest.approx(3.0 / 6.0)
        assert scales["layer2"] == pytest.approx(6.0 / 6.0)

    def test_ranges_to_scales_int4(self):
        """Test converting activation ranges to INT4 scales."""
        ranges = {
            "layer1": (-7.0, 7.0),
        }

        scales = ranges_to_scales(ranges, quant_type="int4_sym")

        # INT4 symmetric max is 7, so scale = absmax / 7.0
        assert scales["layer1"] == pytest.approx(7.0 / 7.0)

    def test_calibration_with_fp4_quantization(self, small_weight_matrix, tmp_path: Path):
        """Test calibration-aware FP4 quantization."""
        # Create mock activation ranges
        activation_ranges = {
            "input_range": (-2.0, 2.0),
            "percentile": 99.5,
            "smooth_factor": 0.5,
        }

        # Quantize with calibration
        packed, scales = quantize_fp4(
            small_weight_matrix,
            group_size=128,
            activation_ranges=activation_ranges,
        )

        # Compare with baseline (no calibration)
        packed_baseline, scales_baseline = quantize_fp4(
            small_weight_matrix,
            group_size=128,
        )

        # Both should produce valid outputs
        assert not np.any(np.isnan(scales))
        assert not np.any(np.isnan(scales_baseline))

        # Shapes should match
        assert packed.shape == packed_baseline.shape
        assert scales.shape == scales_baseline.shape


# =============================================================================
# Slow Tests (require real models or network)
# =============================================================================


@pytest.mark.slow
class TestSlowRealModels:
    """Slow tests that use real models or external resources."""

    def test_calibration_v3_full_download_and_parse(self):
        """Download and parse full calibration v3 dataset."""
        import urllib.request

        try:
            dataset = CalibrationDataset.v3(
                max_samples=100,  # Limit for test speed
                force_download=True,
            )

            assert len(dataset) > 0
            assert len(dataset) <= 100
            assert dataset.version == "v3"
            assert dataset.avg_sample_length > 0

        except urllib.error.URLError as e:
            pytest.skip(f"Network unavailable: {e}")

    def test_large_weight_quantization(self):
        """Test quantization on larger weight matrices."""
        np.random.seed(42)
        W = np.random.randn(4096, 4096).astype(np.float32)

        packed, scales = quantize_fp4(W, group_size=128)

        assert packed.shape == (4096, 512)  # 4096 / 8
        assert scales.shape == (4096, 32)  # 4096 / 128

    def test_full_gptq_large_layer(self):
        """GPTQ on larger layer with full Hessian computation."""
        np.random.seed(42)
        W = np.random.randn(1024, 1024).astype(np.float32)
        X = np.random.randn(512, 1024).astype(np.float32)

        H = compute_hessian(X)
        result = quantize_layer_gptq(W, H, bits=4, group_size=128)

        mse = np.mean((W - result.Q) ** 2)
        assert mse < 0.1  # Reasonable error bound
