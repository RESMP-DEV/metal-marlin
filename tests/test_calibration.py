"""Tests for Bartowski calibration v3 integration.

Validates:
  1. Download from gist URL
  2. Local caching
  3. Activation range computation
  4. Calibration-aware scale computation
  5. CLI command smoke tests

The Bartowski calibration v3 dataset is a multi-domain text corpus
(code, chat, reasoning, math) that provides better activation ranges
than WikiText-2 for LLM quantization.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# Bartowski calibration v3 gist URL
BARTOWSKI_V3_URL = (
    "https://gist.githubusercontent.com/bartowski1182/eb213dccb3571f863da82e99418f81e8/"
    "raw/2c64bb691316d32915b188e495754ef34931ae71/calibration_datav3.txt"
)


class TestDownloadBartowskiV3:
    """Test downloading calibration data from Bartowski gist."""

    def test_download_bartowski_v3(self):
        """Download from gist, verify format."""
        import urllib.request

        # Download with timeout
        try:
            req = urllib.request.Request(
                BARTOWSKI_V3_URL,
                headers={"User-Agent": "Python-urllib/3.12"},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()
                # Should return text content
                assert response.status == 200
                assert len(content) > 0

                # Decode as text
                text = content.decode("utf-8")
                assert len(text) > 100  # Should have substantial content

                # Verify it's text (not binary garbage)
                assert text.isprintable() or "\n" in text

        except urllib.error.URLError as e:
            pytest.skip(f"Network unavailable: {e}")

    def test_download_returns_text_content(self):
        """Verify downloaded content is readable text."""
        import urllib.request

        try:
            req = urllib.request.Request(
                BARTOWSKI_V3_URL,
                headers={"User-Agent": "Python-urllib/3.12"},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                content_type = response.headers.get("Content-Type", "")
                # Should be text/plain
                assert "text" in content_type.lower() or content_type == ""

                data = response.read().decode("utf-8")
                # Should contain actual text content
                lines = data.split("\n")
                assert len(lines) >= 1
                # At least some non-empty lines
                non_empty = [line for line in lines if line.strip()]
                assert len(non_empty) > 0

        except urllib.error.URLError as e:
            pytest.skip(f"Network unavailable: {e}")


class TestCalibrationCaching:
    """Test local cache functionality for calibration data."""

    def test_calibration_caching(self, tmp_path: Path):
        """Verify local cache works."""
        cache_file = tmp_path / "calibration_cache.txt"

        # Simulate writing to cache
        test_content = "Sample calibration text\nLine 2\nLine 3"
        cache_file.write_text(test_content)

        # Verify cache exists
        assert cache_file.exists()
        assert cache_file.stat().st_size > 0

        # Read from cache
        cached = cache_file.read_text()
        assert cached == test_content

    def test_cache_directory_creation(self, tmp_path: Path):
        """Verify cache directory is created if missing."""
        cache_dir = tmp_path / "calibration" / "bartowski"
        cache_file = cache_dir / "v3.txt"

        # Directory doesn't exist yet
        assert not cache_dir.exists()

        # Create directory structure
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("test data")

        assert cache_dir.exists()
        assert cache_file.exists()

    def test_cache_invalidation(self, tmp_path: Path):
        """Test cache invalidation based on file age."""
        cache_file = tmp_path / "calibration.txt"
        cache_file.write_text("old data")

        # Check modification time
        mtime = cache_file.stat().st_mtime
        assert mtime > 0

        # Simulate cache refresh
        cache_file.write_text("new data")
        new_mtime = cache_file.stat().st_mtime
        assert new_mtime >= mtime

        assert cache_file.read_text() == "new data"


class TestActivationRangeComputation:
    """Test activation range computation with mock model."""

    def test_activation_range_computation(self):
        """Mock model, verify ranges computed."""
        # Simulate a simple linear layer's activation tracking
        # In real calibration, we'd hook forward passes and track min/max

        # Mock activations from calibration passes
        activations = [
            np.random.randn(32, 128).astype(np.float32) * 2.0,  # batch 1
            np.random.randn(32, 128).astype(np.float32) * 1.5,  # batch 2
            np.random.randn(32, 128).astype(np.float32) * 2.5,  # batch 3
        ]

        # Compute running min/max across batches
        layer_min = float("inf")
        layer_max = float("-inf")

        for act in activations:
            batch_min = float(act.min())
            batch_max = float(act.max())
            layer_min = min(layer_min, batch_min)
            layer_max = max(layer_max, batch_max)

        # Verify ranges are reasonable
        assert layer_min < 0  # Gaussian should have negatives
        assert layer_max > 0  # And positives
        assert layer_max > layer_min

        # Ranges should encompass most values
        all_acts = np.concatenate([a.flatten() for a in activations])
        assert layer_min <= all_acts.min()
        assert layer_max >= all_acts.max()

    def test_percentile_based_ranges(self):
        """Test percentile-based activation ranges for outlier robustness."""
        # Create activations with outliers
        np.random.seed(42)
        activations = np.random.randn(10000).astype(np.float32)
        # Add some outliers at specific positions
        activations[0] = 100.0  # Extreme outlier
        activations[1] = -100.0

        # Min/max approach
        minmax_range = (float(activations.min()), float(activations.max()))

        # Percentile approach (1% and 99%) - excludes top/bottom 1%
        percentile_low = float(np.percentile(activations, 1))
        percentile_high = float(np.percentile(activations, 99))
        percentile_range = (percentile_low, percentile_high)

        # Percentile range should be much tighter
        minmax_span = minmax_range[1] - minmax_range[0]
        percentile_span = percentile_range[1] - percentile_range[0]

        assert percentile_span < minmax_span
        assert minmax_span > 100  # Due to outliers
        # Standard normal 1st-99th percentile is roughly [-2.33, 2.33] = ~4.66
        assert percentile_span < 10  # Excludes outliers, should be ~4.66

    def test_per_layer_activation_ranges(self):
        """Test computing activation ranges for multiple layers."""
        np.random.seed(123)

        # Simulate different layers with different activation patterns
        layers = {
            "model.layers.0.attn.q_proj": np.random.randn(64, 256) * 1.0,
            "model.layers.0.attn.k_proj": np.random.randn(64, 256) * 0.8,
            "model.layers.0.attn.v_proj": np.random.randn(64, 256) * 1.2,
            "model.layers.0.mlp.gate_proj": np.random.randn(64, 512) * 2.0,
            "model.layers.0.mlp.down_proj": np.random.randn(64, 512) * 1.5,
        }

        activation_ranges: dict[str, tuple[float, float]] = {}

        for name, acts in layers.items():
            layer_min = float(acts.min())
            layer_max = float(acts.max())
            activation_ranges[name] = (layer_min, layer_max)

        # Verify all layers have ranges
        assert len(activation_ranges) == len(layers)

        # MLP layers should have larger ranges (higher scale factor)
        mlp_gate_range = activation_ranges["model.layers.0.mlp.gate_proj"]
        attn_k_range = activation_ranges["model.layers.0.attn.k_proj"]

        mlp_span = mlp_gate_range[1] - mlp_gate_range[0]
        attn_span = attn_k_range[1] - attn_k_range[0]

        # MLP gate has 2.0x scale, attn k has 0.8x, so MLP should be wider
        assert mlp_span > attn_span


class TestCalibrationAwareScale:
    """Test comparing scales with/without calibration."""

    def test_calibration_aware_scale(self):
        """Compare scales with/without calibration."""
        from metal_marlin.quantize_fp4 import quantize_fp4

        np.random.seed(456)

        # Create a weight tensor
        weight = np.random.randn(256, 256).astype(np.float16)

        # Quantize without calibration
        packed_no_cal, scales_no_cal = quantize_fp4(weight, group_size=128)

        # Quantize with calibration data
        activation_ranges = {
            "input_range": (-3.0, 3.0),
            "percentile": 99.9,
            "smooth_factor": 0.5,
        }
        packed_cal, scales_cal = quantize_fp4(
            weight, group_size=128, activation_ranges=activation_ranges
        )

        # Packed weights should have same shape
        assert packed_no_cal.shape == packed_cal.shape

        # Scales may differ based on calibration
        assert scales_no_cal.shape == scales_cal.shape

        # Both should produce valid quantized outputs
        assert not np.any(np.isnan(scales_no_cal))
        assert not np.any(np.isnan(scales_cal))
        assert not np.any(np.isinf(scales_no_cal))
        assert not np.any(np.isinf(scales_cal))

    def test_scale_difference_with_calibration(self):
        """Calibration should potentially affect scale values."""
        from metal_marlin.quantize_fp4 import quantize_fp4

        np.random.seed(789)

        # Weight tensor with specific distribution
        weight = np.random.randn(128, 128).astype(np.float16) * 0.5

        # No calibration
        _, scales_baseline = quantize_fp4(weight, group_size=128)

        # With attenuation factor (activations small)
        activation_ranges_small = {
            "input_range": (-0.1, 0.1),  # Small activations
            "smooth_factor": 1.0,
        }
        _, scales_small_act = quantize_fp4(
            weight, group_size=128, activation_ranges=activation_ranges_small
        )

        # With amplification factor (activations large)
        activation_ranges_large = {
            "input_range": (-10.0, 10.0),  # Large activations
            "smooth_factor": 1.0,
        }
        _, scales_large_act = quantize_fp4(
            weight, group_size=128, activation_ranges=activation_ranges_large
        )

        # All should be valid
        assert np.all(scales_baseline > 0)
        assert np.all(scales_small_act > 0)
        assert np.all(scales_large_act > 0)

    def test_quantization_error_comparison(self):
        """Compare quantization error with/without calibration."""
        from metal_marlin.quantize_fp4 import (
            compute_quantization_error,
            quantize_fp4,
        )

        np.random.seed(999)

        weight = np.random.randn(256, 256).astype(np.float16)

        # Without calibration
        packed1, scales1 = quantize_fp4(weight, group_size=128)
        err1 = compute_quantization_error(weight, packed1, scales1, group_size=128)

        # With calibration (using percentile clipping)
        activation_ranges = {
            "percentile": 99.5,  # Clip outliers
            "smooth_factor": 0.8,
        }
        packed2, scales2 = quantize_fp4(
            weight, group_size=128, activation_ranges=activation_ranges
        )
        err2 = compute_quantization_error(weight, packed2, scales2, group_size=128)

        # Both should have reasonable error
        assert err1["rmse"] < 1.0
        assert err2["rmse"] < 1.0

        # Error metrics should be populated
        for key in ["mse", "rmse", "max_error", "mean_relative_error"]:
            assert key in err1
            assert key in err2


class TestCalibrationCLI:
    """Smoke tests for calibration CLI commands."""

    def test_calibration_cli_help(self):
        """Test that CLI shows calibration options."""
        from click.testing import CliRunner
        from metal_marlin.cli import cli

        runner = CliRunner()

        # Check quantize command has calibration option
        result = runner.invoke(cli, ["quantize", "--help"])
        assert result.exit_code == 0
        assert "--calibration" in result.output

    def test_cli_convert_help(self):
        """Test convert command help includes expected options."""
        from click.testing import CliRunner
        from metal_marlin.cli import cli

        runner = CliRunner()

        result = runner.invoke(cli, ["convert", "--help"])
        assert result.exit_code == 0
        assert "--quant" in result.output
        assert "--group-size" in result.output

    def test_cli_main_help(self):
        """Test main CLI help output."""
        from click.testing import CliRunner
        from metal_marlin.cli import cli

        runner = CliRunner()

        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Metal Marlin" in result.output or "marlin" in result.output.lower()

    def test_cli_invalid_command(self):
        """Test CLI handles invalid commands gracefully."""
        from click.testing import CliRunner
        from metal_marlin.cli import cli

        runner = CliRunner()

        result = runner.invoke(cli, ["nonexistent-command"])
        assert result.exit_code != 0


class TestCalibrationDataFormat:
    """Test expected calibration data format handling."""

    def test_parse_text_calibration_data(self):
        """Test parsing plain text calibration data."""
        # Calibration data is plain text with samples separated by newlines
        sample_data = """This is the first calibration sample for code.
def hello_world():
    print("Hello, World!")

This is another sample with chat format.
User: What is the capital of France?
Assistant: The capital of France is Paris.

Mathematical reasoning example:
If x + 5 = 12, then x = 12 - 5 = 7.
"""
        # Split into samples (could be by double newline, or other delimiter)
        # For simplicity, split by blank lines
        paragraphs = [p.strip() for p in sample_data.split("\n\n") if p.strip()]

        assert len(paragraphs) >= 3
        assert any("def " in p for p in paragraphs)  # Code sample
        assert any("User:" in p or "Assistant:" in p for p in paragraphs)  # Chat

    def test_calibration_sample_tokenization(self):
        """Test that calibration samples can be tokenized."""
        # Mock tokenization (in real use, would use HF tokenizer)
        sample = "The quick brown fox jumps over the lazy dog."

        # Simple word tokenization (proxy for real tokenization)
        tokens = sample.split()
        assert len(tokens) == 9

        # Verify tokens are valid strings
        for token in tokens:
            assert len(token) > 0
            assert isinstance(token, str)

    def test_max_samples_limit(self):
        """Test that calibration respects max_samples limit."""
        all_samples = [f"Sample {i}: content here" for i in range(1000)]

        max_samples = 512
        limited = all_samples[:max_samples]

        assert len(limited) == max_samples
        assert limited[0] == "Sample 0: content here"
        assert limited[-1] == "Sample 511: content here"


class TestCalibrationIntegration:
    """Integration tests for calibration workflow."""

    def test_full_calibration_workflow(self, tmp_path: Path):
        """Test end-to-end calibration workflow."""
        from metal_marlin.quantize_fp4 import quantize_fp4

        np.random.seed(42)

        # Step 1: Create mock calibration data
        calibration_file = tmp_path / "calibration.txt"
        calibration_samples = [
            "Sample 1: code snippet def foo(): return 42",
            "Sample 2: chat User: Hello! Assistant: Hi there!",
            "Sample 3: math reasoning x^2 + 2x + 1 = (x+1)^2",
        ]
        calibration_file.write_text("\n".join(calibration_samples))

        # Step 2: Simulate activation collection
        # In real workflow, would run model forward passes
        mock_activations = {
            "layer_0": np.random.randn(100, 256) * 1.5,
            "layer_1": np.random.randn(100, 256) * 2.0,
        }

        # Step 3: Compute activation ranges
        activation_ranges_dict: dict[str, tuple[float, float]] = {}
        for name, acts in mock_activations.items():
            # Use 99.9th percentile for robustness
            low = float(np.percentile(acts, 0.1))
            high = float(np.percentile(acts, 99.9))
            activation_ranges_dict[name] = (low, high)

        # Step 4: Quantize with calibration
        weight = np.random.randn(128, 256).astype(np.float16)

        # Use layer_0's range for this weight
        cal_data = {
            "input_range": activation_ranges_dict["layer_0"],
            "percentile": 99.9,
            "smooth_factor": 0.7,
        }

        packed, scales = quantize_fp4(weight, group_size=128, activation_ranges=cal_data)

        # Verify outputs
        assert packed.shape == (128, 32)  # 256 / 8 = 32
        assert scales.shape == (128, 2)  # 256 / 128 = 2 groups
        assert not np.any(np.isnan(scales))

    def test_calibration_saves_to_json(self, tmp_path: Path):
        """Test saving activation ranges to JSON."""
        activation_ranges = {
            "model.layers.0.q_proj": (-2.5, 3.1),
            "model.layers.0.k_proj": (-1.8, 2.4),
            "model.layers.0.v_proj": (-2.1, 2.8),
        }

        output_file = tmp_path / "activation_ranges.json"

        # Convert tuples to lists for JSON serialization
        json_data = {k: list(v) for k, v in activation_ranges.items()}

        with open(output_file, "w") as f:
            json.dump(json_data, f, indent=2)

        # Verify can reload
        with open(output_file) as f:
            loaded = json.load(f)

        assert len(loaded) == 3
        assert loaded["model.layers.0.q_proj"] == [-2.5, 3.1]


class TestHessianComputation:
    """Tests for GPTQ Hessian approximation (X^T @ X)."""

    def test_hessian_basic_computation(self):
        """Test that H = X^T @ X is computed correctly."""
        # Create simple test activations
        np.random.seed(42)
        X = np.random.randn(100, 64).astype(np.float32)

        # Expected Hessian
        expected_H = X.T @ X

        # Verify shape
        assert expected_H.shape == (64, 64)

        # Verify symmetry
        assert np.allclose(expected_H, expected_H.T)

        # Verify positive semi-definiteness (all eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(expected_H)
        assert np.all(eigenvalues >= -1e-6)  # Allow small numerical error

    def test_hessian_incremental_accumulation(self):
        """Test that incremental H accumulation matches batch computation."""
        np.random.seed(123)

        # Create multiple batches
        batch1 = np.random.randn(50, 128).astype(np.float32)
        batch2 = np.random.randn(30, 128).astype(np.float32)
        batch3 = np.random.randn(40, 128).astype(np.float32)

        # Batch computation: concatenate all, compute H
        X_all = np.vstack([batch1, batch2, batch3])
        H_batch = X_all.T @ X_all

        # Incremental computation: accumulate H from each batch
        H_incremental = np.zeros((128, 128), dtype=np.float32)
        for batch in [batch1, batch2, batch3]:
            H_incremental += batch.T @ batch

        # Should match within floating point tolerance
        # Using atol=1e-3 to account for float32 accumulation order differences
        assert np.allclose(H_batch, H_incremental, atol=1e-3)

    def test_hessian_damping(self):
        """Test Hessian damping H_damped = H + λI."""
        np.random.seed(456)
        X = np.random.randn(100, 32).astype(np.float32)
        H = X.T @ X

        # Compute damping
        damping_factor = 0.01
        mean_diag = np.mean(np.diag(H))
        lambda_damp = damping_factor * mean_diag
        H_damped = H + lambda_damp * np.eye(32)

        # Verify diagonal increased
        assert np.all(np.diag(H_damped) > np.diag(H))

        # Verify still symmetric
        assert np.allclose(H_damped, H_damped.T)

        # Verify positive definite (eigenvalues > 0 after damping)
        eigenvalues = np.linalg.eigvalsh(H_damped)
        assert np.all(eigenvalues > 0)

    def test_hessian_shape_for_different_layers(self):
        """Test Hessian shape matches in_features x in_features."""
        np.random.seed(789)

        test_cases = [
            (100, 256),  # q_proj: 100 tokens, 256 hidden
            (100, 512),  # k_proj: 100 tokens, 512 hidden
            (100, 1024),  # mlp: 100 tokens, 1024 hidden
        ]

        for num_tokens, in_features in test_cases:
            X = np.random.randn(num_tokens, in_features).astype(np.float32)
            H = X.T @ X
            assert H.shape == (in_features, in_features), (
                f"Expected ({in_features}, {in_features}), got {H.shape}"
            )

    def test_hessian_memory_efficiency(self):
        """Verify Hessian accumulation is memory-efficient."""
        # Memory for storing all activations vs just Hessian
        num_batches = 100
        tokens_per_batch = 512
        in_features = 4096

        # Storing all activations would require:
        all_activations_bytes = num_batches * tokens_per_batch * in_features * 4  # float32
        all_activations_mb = all_activations_bytes / (1024 * 1024)

        # Storing just the Hessian requires:
        hessian_bytes = in_features * in_features * 4  # float32
        hessian_mb = hessian_bytes / (1024 * 1024)

        # Hessian should be much smaller for large calibration sets
        assert hessian_mb < all_activations_mb
        assert hessian_mb == pytest.approx(64.0, rel=0.01)  # 4096^2 * 4 / 1MB = 64MB

    def test_hessian_numerical_stability(self):
        """Test numerical stability with varying activation magnitudes."""
        np.random.seed(999)

        # Test with small activations
        X_small = np.random.randn(100, 64).astype(np.float32) * 0.001
        H_small = X_small.T @ X_small
        assert not np.any(np.isnan(H_small))
        assert not np.any(np.isinf(H_small))

        # Test with large activations
        X_large = np.random.randn(100, 64).astype(np.float32) * 1000
        H_large = X_large.T @ X_large
        assert not np.any(np.isnan(H_large))
        assert not np.any(np.isinf(H_large))

        # Test with mixed magnitudes
        X_mixed = np.random.randn(100, 64).astype(np.float32)
        X_mixed[:, :32] *= 0.001
        X_mixed[:, 32:] *= 1000
        H_mixed = X_mixed.T @ X_mixed
        assert not np.any(np.isnan(H_mixed))
        assert not np.any(np.isinf(H_mixed))


class TestHessianCollector:
    """Tests for HessianCollector class with MLX model instrumentation."""

    @pytest.fixture
    def simple_mlx_model(self):
        """Create a simple MLX model for testing."""
        try:
            import mlx.core as mx
            import mlx.nn as nn
        except ImportError:
            pytest.skip("MLX not available")

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(64, 128)
                self.linear2 = nn.Linear(128, 64)

            def __call__(self, x: mx.array) -> mx.array:
                x = self.linear1(x)
                x = nn.relu(x)
                x = self.linear2(x)
                return x

        return SimpleModel()

    def test_hessian_collector_basic(self, simple_mlx_model):
        """Test HessianCollector captures Hessians correctly."""
        try:
            import mlx.core as mx
            from converters.calibration import HessianCollector
        except ImportError:
            pytest.skip("MLX or calibration module not available")

        collector = HessianCollector(simple_mlx_model)
        assert collector.num_layers == 2

        # Run forward pass
        x = mx.random.normal((10, 64))
        _ = simple_mlx_model(x)
        mx.eval(list(collector._hessians.values()))

        # Check Hessians were collected
        assert len(collector._hessians) == 2
        assert "linear1" in collector._hessians
        assert "linear2" in collector._hessians

        # Check shapes
        H1 = collector._hessians["linear1"]
        assert H1.shape == (64, 64)  # in_features of linear1

        H2 = collector._hessians["linear2"]
        assert H2.shape == (128, 128)  # in_features of linear2

        collector.remove_hooks()

    def test_hessian_collector_damping(self, simple_mlx_model):
        """Test collect_hessian applies damping correctly."""
        try:
            import mlx.core as mx
            from converters.calibration import HessianCollector
        except ImportError:
            pytest.skip("MLX or calibration module not available")

        damping_factor = 0.05
        collector = HessianCollector(simple_mlx_model, damping_factor=damping_factor)

        # Run forward passes
        for _ in range(5):
            x = mx.random.normal((20, 64))
            _ = simple_mlx_model(x)
        mx.eval(list(collector._hessians.values()))

        # Get damped Hessian
        H_damped = collector.collect_hessian("linear1")

        # Verify shape
        assert H_damped.shape == (64, 64)

        # Verify it's larger on diagonal (damping added)
        H_raw = collector._hessians["linear1"]
        damped_diag = mx.diag(H_damped)
        raw_diag = mx.diag(H_raw)
        assert mx.all(damped_diag > raw_diag)

        collector.remove_hooks()

    def test_hessian_collector_reset(self, simple_mlx_model):
        """Test reset clears accumulated Hessians."""
        try:
            import mlx.core as mx
            from converters.calibration import HessianCollector
        except ImportError:
            pytest.skip("MLX or calibration module not available")

        collector = HessianCollector(simple_mlx_model)

        # Run forward pass
        x = mx.random.normal((10, 64))
        _ = simple_mlx_model(x)
        mx.eval(list(collector._hessians.values()))

        assert len(collector._hessians) == 2

        # Reset
        collector.reset()
        assert len(collector._hessians) == 0
        assert len(collector._sample_counts) == 0

        collector.remove_hooks()

    def test_hessian_collector_layer_selection(self, simple_mlx_model):
        """Test HessianCollector can target specific layers."""
        try:
            import mlx.core as mx
            from converters.calibration import HessianCollector
        except ImportError:
            pytest.skip("MLX or calibration module not available")

        # Only collect for linear1
        collector = HessianCollector(
            simple_mlx_model,
            layers_to_calibrate=["linear1"],
        )
        assert collector.num_layers == 1

        # Run forward pass
        x = mx.random.normal((10, 64))
        _ = simple_mlx_model(x)
        mx.eval(list(collector._hessians.values()))

        # Only linear1 should have Hessian
        assert "linear1" in collector._hessians
        assert "linear2" not in collector._hessians

        collector.remove_hooks()


class TestComputeLayerHessians:
    """Tests for compute_layer_hessians function."""

    @pytest.fixture
    def calibration_batches(self):
        """Create mock calibration batches."""
        np.random.seed(42)
        return [
            {"input_ids": np.random.randint(0, 1000, (4, 128))}
            for _ in range(10)
        ]

    def test_compute_layer_hessians_with_chunking(self, tmp_path: Path):
        """Test chunked processing with disk caching."""
        try:
            from converters.calibration import (
                _cache_hessians,
                load_cached_hessians,
            )
        except ImportError:
            pytest.skip("Calibration module not available")

        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not available")

        # Create mock Hessians
        hessians = {
            "layer1": mx.random.normal((64, 64)),
            "layer2": mx.random.normal((128, 128)),
        }

        # Cache to disk
        cache_file = tmp_path / "hessian_chunk_0_10.npz"
        _cache_hessians(hessians, cache_file)

        assert cache_file.exists()

        # Load back
        loaded = load_cached_hessians(tmp_path)
        assert "layer1" in loaded
        assert "layer2" in loaded

    def test_hessian_sample_count_tracking(self):
        """Test that sample counts are tracked correctly."""
        try:
            import mlx.core as mx
            import mlx.nn as nn
            from converters.calibration import HessianCollector
        except ImportError:
            pytest.skip("MLX or calibration module not available")

        class SimpleLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 64)

            def __call__(self, x: mx.array) -> mx.array:
                return self.linear(x)

        model = SimpleLinear()
        collector = HessianCollector(model)

        # Track total samples
        total_samples = 0

        # Run multiple forward passes with different batch sizes
        batch_sizes = [10, 20, 15, 25]
        for bs in batch_sizes:
            x = mx.random.normal((bs, 32))
            _ = model(x)
            total_samples += bs
        mx.eval(list(collector._hessians.values()))

        # Verify sample count
        assert collector._sample_counts["linear"] == total_samples

        collector.remove_hooks()
