"""Tests for mixed bit-width kernel auto-tuning system."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from metal_marlin.trellis.autotune_mixed_bpw import (
    AutotuneConfig,
    BenchmarkResult,
    KernelConfig,
    MixedBPWAutoTuner,
)


class TestKernelConfig:
    """Test KernelConfig dataclass."""

    def test_kernel_config_creation(self):
        """Test creating a KernelConfig."""
        config = KernelConfig(
            kernel_name="test_kernel",
            tile_size_m=64,
            tile_size_n=64,
            simdgroups=8,
            use_fp32_acc=False,
            kernel_variant="base",
        )
        assert config.kernel_name == "test_kernel"
        assert config.tile_size_m == 64
        assert config.tile_size_n == 64
        assert config.simdgroups == 8
        assert not config.use_fp32_acc
        assert config.kernel_variant == "base"


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """Test creating a BenchmarkResult."""
        config = KernelConfig(
            kernel_name="test",
            tile_size_m=64,
            tile_size_n=64,
            simdgroups=8,
        )
        result = BenchmarkResult(
            config=config,
            latency_ms=1.5,
            throughput_gbps=100.0,
            memory_footprint_mb=256.0,
        )
        assert result.config == config
        assert result.latency_ms == 1.5
        assert result.throughput_gbps == 100.0
        assert result.memory_footprint_mb == 256.0


class TestAutotuneConfig:
    """Test AutotuneConfig dataclass."""

    def test_default_config(self):
        """Test default AutotuneConfig."""
        config = AutotuneConfig(device_name="M4 Max", device_family="M4")
        assert config.device_name == "M4 Max"
        assert config.device_family == "M4"
        assert 2 in config.tile_size_mapping
        assert config.tile_size_mapping[2] == 128
        assert 4 in config.simdgroup_mapping
        assert config.simdgroup_mapping[4] == 8

    def test_tile_size_mapping(self):
        """Test tile size mapping for different bit widths."""
        config = AutotuneConfig(device_name="M4 Max", device_family="M4")

        # 2-bit should have larger tiles
        assert config.tile_size_mapping[2] == 128

        # 4-bit should have smaller tiles
        assert config.tile_size_mapping[4] == 64

        # 8-bit should use smallest tiles
        assert config.tile_size_mapping[8] == 64


class TestMixedBPWAutoTuner:
    """Test MixedBPWAutoTuner class."""

    def test_initialization(self):
        """Test tuner initialization."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")
        assert tuner.device_name == "M4 Max"
        assert tuner.device_family == "M4"
        assert len(tuner._benchmark_results) == 0

    def test_device_detection(self):
        """Test automatic device detection."""
        tuner = MixedBPWAutoTuner()  # No device_name specified
        assert tuner.device_name is not None
        assert tuner.device_family in ["M1", "M2", "M3", "M4", "Unknown"]

    def test_kernel_name_generation(self):
        """Test kernel name generation."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")

        # Base variant
        base_name = tuner._get_kernel_name("base", False, 8)
        assert "moe_trellis_mixed_swiglu" in base_name

        # Decode variant
        decode_name = tuner._get_kernel_name("decode", False, 8)
        assert "decode" in decode_name

        # FP32 accumulator
        fp32_name = tuner._get_kernel_name("base", True, 8)
        assert "fp32acc" in fp32_name

    def test_config_generation_single_bit(self):
        """Test config generation for single bit width."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")
        configs = tuner._generate_configs(bit_widths=[4], use_decode=True, use_prefill=True)

        # Should have multiple configs (different tile sizes, simdgroups, variants)
        assert len(configs) > 0

        # All configs should have bit-width specific tile sizes
        tile_sizes = {cfg.tile_size_n for cfg in configs}
        assert len(tile_sizes) > 1  # Multiple tile sizes

        # Should have decode and base variants
        variants = {cfg.kernel_variant for cfg in configs}
        assert "decode" in variants or "base" in variants

    def test_config_generation_mixed_bits(self):
        """Test config generation for mixed bit widths."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")
        configs = tuner._generate_configs(bit_widths=[2, 3, 4], use_decode=True)

        # Should have configs for each bit width
        bit_widths_in_configs = set()
        for cfg in configs:
            # Config doesn't store bit_width directly, but we can infer from tile sizes
            # 2-bit should have larger tiles (128)
            if cfg.tile_size_n >= 128:
                bit_widths_in_configs.add(2)
            # 4-bit should have smaller tiles (64)
            elif cfg.tile_size_n <= 64:
                bit_widths_in_configs.add(4)

        # At least 2 and 4-bit should be represented
        assert 4 in bit_widths_in_configs

    def test_synthetic_input_creation(self):
        """Test synthetic input tensor creation."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")

        batch_size = 4
        hidden_dim = 4096
        out_dim = 2048

        input_tensor, output_tensor = tuner._create_synthetic_input(
            batch_size, hidden_dim, out_dim, device="cpu"
        )

        assert input_tensor.shape == (batch_size, hidden_dim)
        assert output_tensor.shape == (batch_size, out_dim)
        assert input_tensor.dtype == torch.float16
        assert output_tensor.dtype == torch.float16

    def test_nearest_key_finding(self):
        """Test finding nearest key in a set."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")

        keys = {1, 4, 8, 16, 32}

        # Exact match
        assert tuner._find_nearest_key(8, keys) == 8

        # Nearest below
        assert tuner._find_nearest_key(6, keys) == 4

        # Nearest above
        assert tuner._find_nearest_key(10, keys) == 8

        # Below minimum
        assert tuner._find_nearest_key(0, keys) == 1

        # Above maximum
        assert tuner._find_nearest_key(100, keys) == 32

        # Empty set
        assert tuner._find_nearest_key(8, set()) is None

    def test_kernel_selection_with_lookup_table(self):
        """Test kernel selection from lookup table."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")

        # Populate lookup table
        batch_size = 4
        hidden_dim = 4096
        bit_widths = [2, 4]
        bit_tuple = tuple(sorted(bit_widths))

        config = KernelConfig(
            kernel_name="test_kernel",
            tile_size_m=64,
            tile_size_n=64,
            simdgroups=8,
        )

        tuner.config.lookup_table[batch_size][hidden_dim][bit_tuple] = config

        # Select kernel
        selected = tuner.select_kernel(batch_size, hidden_dim, bit_widths)

        assert selected is not None
        assert selected.kernel_name == "test_kernel"

    def test_kernel_selection_nearest_batch(self):
        """Test kernel selection with nearest batch size."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")

        # Populate lookup table for batch_size=4
        batch_size = 4
        hidden_dim = 4096
        bit_widths = [4]
        bit_tuple = tuple(sorted(bit_widths))

        config = KernelConfig(
            kernel_name="test_kernel",
            tile_size_m=64,
            tile_size_n=64,
            simdgroups=8,
        )

        tuner.config.lookup_table[batch_size][hidden_dim][bit_tuple] = config

        # Select kernel for batch_size=6 (should find nearest)
        selected = tuner.select_kernel(6, hidden_dim, bit_widths)

        assert selected is not None
        assert selected.kernel_name == "test_kernel"

    def test_kernel_selection_fallback(self):
        """Test kernel selection with fallback to default."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")

        # Empty lookup table
        selected = tuner.select_kernel(
            batch_size=4,
            hidden_dim=4096,
            bit_widths=[4],
            fallback_to_default=True,
        )

        assert selected is not None
        assert selected.kernel_name is not None
        assert selected.tile_size_m == tuner.config.tile_size_mapping[4]
        assert selected.tile_size_n == tuner.config.tile_size_mapping[4]

    def test_kernel_selection_no_fallback(self):
        """Test kernel selection without fallback."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")

        # Empty lookup table
        selected = tuner.select_kernel(
            batch_size=4,
            hidden_dim=4096,
            bit_widths=[4],
            fallback_to_default=False,
        )

        assert selected is None

    def test_latency_recording(self):
        """Test recording kernel latency."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")

        kernel_name = "test_kernel"
        batch_size = 4
        latency_ms = 1.5

        tuner.record_latency(kernel_name, batch_size, latency_ms)

        key = f"{kernel_name}_bs{batch_size}"
        assert key in tuner.config.adaptation_timings
        assert len(tuner.config.adaptation_timings[key]) == 1
        assert tuner.config.adaptation_timings[key][0] == latency_ms

    def test_latency_recording_with_context(self):
        """Test recording latency with hidden_dim and bit_widths context."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")

        kernel_name = "test_kernel"
        batch_size = 4
        latency_ms = 1.5
        hidden_dim = 4096
        bit_widths = [2, 4]

        tuner.record_latency(kernel_name, batch_size, latency_ms, hidden_dim, bit_widths)

        key = f"{kernel_name}_bs{batch_size}_hd{hidden_dim}_bits{(2, 4)}"
        assert key in tuner.config.adaptation_timings
        assert tuner.config.adaptation_timings[key][0] == latency_ms

    def test_adaptation_history_trimming(self):
        """Test that adaptation history is trimmed to max length."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")
        tuner.config.adaptation_history_len = 5

        kernel_name = "test_kernel"
        batch_size = 4

        # Record more latencies than history length
        for i in range(10):
            tuner.record_latency(kernel_name, batch_size, float(i))

        key = f"{kernel_name}_bs{batch_size}"
        assert len(tuner.config.adaptation_timings[key]) == 5
        # Should keep most recent
        assert tuner.config.adaptation_timings[key][-1] == 9.0

    def test_adaptation_selection_no_history(self):
        """Test adaptation selection with no history."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")

        # No adaptation history
        adapted = tuner.adapt_selection(
            batch_size=4,
            hidden_dim=4096,
            bit_widths=[4],
        )

        assert adapted is None

    def test_adaptation_selection_insufficient_samples(self):
        """Test adaptation selection with insufficient samples."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")

        # Populate lookup table
        config = KernelConfig(
            kernel_name="test_kernel",
            tile_size_m=64,
            tile_size_n=64,
            simdgroups=8,
        )
        tuner.config.lookup_table[4][4096][(4,)] = config

        # Record insufficient samples (less than 5)
        for _ in range(3):
            tuner.record_latency("test_kernel", 4, 1.0, 4096, [4])

        adapted = tuner.adapt_selection(batch_size=4, hidden_dim=4096, bit_widths=[4])
        assert adapted is None

    def test_adaptation_selection_sufficient_samples(self):
        """Test adaptation selection with sufficient samples."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")

        # Populate lookup table
        config = KernelConfig(
            kernel_name="test_kernel",
            tile_size_m=64,
            tile_size_n=64,
            simdgroups=8,
        )
        tuner.config.lookup_table[4][4096][(4,)] = config

        # Record sufficient samples
        for _ in range(5):
            tuner.record_latency("test_kernel", 4, 1.0, 4096, [4])

        adapted = tuner.adapt_selection(batch_size=4, hidden_dim=4096, bit_widths=[4])
        assert adapted is not None
        assert adapted.kernel_name == "test_kernel"

    def test_clear_adaptation_history(self):
        """Test clearing adaptation history."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")

        # Record some latencies
        tuner.record_latency("test_kernel", 4, 1.0)

        assert len(tuner.config.adaptation_timings) > 0

        # Clear history
        tuner.clear_adaptation_history()

        assert len(tuner.config.adaptation_timings) == 0

    def test_reset(self):
        """Test resetting tuner to initial state."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")

        # Populate state
        tuner.config.lookup_table[4][4096][(4,)] = KernelConfig(
            kernel_name="test",
            tile_size_m=64,
            tile_size_n=64,
            simdgroups=8,
        )
        tuner.record_latency("test_kernel", 4, 1.0)

        assert len(tuner.config.lookup_table) > 0
        assert len(tuner.config.adaptation_timings) > 0

        # Reset
        tuner.reset()

        assert len(tuner.config.lookup_table) == 0
        assert len(tuner.config.adaptation_timings) == 0
        assert len(tuner._benchmark_results) == 0

    def test_export_config(self):
        """Test exporting config to JSON."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")

        # Populate some state
        config = KernelConfig(
            kernel_name="test_kernel",
            tile_size_m=64,
            tile_size_n=64,
            simdgroups=8,
        )
        tuner.config.lookup_table[4][4096][(4,)] = config

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "test_config.json"

            json_str = tuner.export_config(export_path)

            assert export_path.exists()

            # Verify JSON can be parsed
            parsed = json.loads(json_str)
            assert parsed["device_name"] == "M4 Max"
            assert parsed["device_family"] == "M4"

            # Verify file content matches
            with open(export_path, "r") as f:
                file_json = json.load(f)
                assert file_json == parsed

    def test_export_and_load_config(self):
        """Test exporting and loading config roundtrip."""
        tuner1 = MixedBPWAutoTuner(device_name="M4 Max")

        # Populate state
        config = KernelConfig(
            kernel_name="test_kernel",
            tile_size_m=64,
            tile_size_n=64,
            simdgroups=8,
        )
        tuner1.config.lookup_table[4][4096][(4,)] = config
        tuner1.config.adaptation_enabled = False

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "test_config.json"

            # Export
            tuner1.export_config(export_path)

            # Load into new tuner
            tuner2 = MixedBPWAutoTuner.load_config(export_path)

            # Verify state matches
            assert tuner2.device_name == tuner1.device_name
            assert tuner2.device_family == tuner1.device_family
            assert tuner2.config.adaptation_enabled == tuner1.config.adaptation_enabled

            # Verify lookup table
            assert 4 in tuner2.config.lookup_table
            assert 4096 in tuner2.config.lookup_table[4]
            assert (4,) in tuner2.config.lookup_table[4][4096]
            loaded_config = tuner2.config.lookup_table[4][4096][(4,)]
            assert loaded_config.kernel_name == config.kernel_name
            assert loaded_config.tile_size_m == config.tile_size_m
            assert loaded_config.tile_size_n == config.tile_size_n

    def test_get_statistics(self, tmp_path):
        """Test getting tuner statistics."""
        config_file = tmp_path / "test.json"
        tuner = MixedBPWAutoTuner(device_name="M4 Max", config_path=str(config_file))

        # Populate some state
        config = KernelConfig(
            kernel_name="test_kernel",
            tile_size_m=64,
            tile_size_n=64,
            simdgroups=8,
        )
        tuner.config.lookup_table[4][4096][(4,)] = config
        tuner.record_latency("test_kernel", 4, 1.0)
        tuner.record_latency("test_kernel", 4, 1.5)

        stats = tuner.get_statistics()

        assert stats["device_name"] == "M4 Max"
        assert stats["device_family"] == "M4"
        assert stats["lookup_table_entries"] == 1
        assert stats["adaptation_enabled"] is True
        assert stats["adaptation_samples"] == 2
        assert stats["active_adaptation_keys"] == 1

    def test_empty_statistics(self):
        """Test statistics for empty tuner."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")

        stats = tuner.get_statistics()

        assert stats["device_name"] == "M4 Max"
        assert stats["total_benchmarks"] == 0
        assert stats["lookup_table_entries"] == 0
        assert stats["adaptation_samples"] == 0
        assert stats["active_adaptation_keys"] == 0

    def test_config_path_attribute(self):
        """Test config_path is properly set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test.json"
            tuner = MixedBPWAutoTuner(config_path=config_path)

            assert tuner.config_path == config_path

            stats = tuner.get_statistics()
            assert str(config_path) in stats["config_path"]

    def test_multiple_bit_widths_in_lookup(self):
        """Test lookup table with multiple bit-width combinations."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")

        # Add entries for different bit-width combinations
        configs = [
            (2, 4),
            (3, 4),
            (2, 3, 4),
            (4,),
        ]

        for bit_widths in configs:
            config = KernelConfig(
                kernel_name=f"kernel_{'_'.join(map(str, bit_widths))}",
                tile_size_m=64,
                tile_size_n=64,
                simdgroups=8,
            )
            tuner.config.lookup_table[4][4096][bit_widths] = config

        # Select for different bit-width combinations
        selected_2_4 = tuner.select_kernel(4, 4096, [2, 4])
        assert selected_2_4 is not None
        assert "2_4" in selected_2_4.kernel_name

        selected_3_4 = tuner.select_kernel(4, 4096, [3, 4])
        assert selected_3_4 is not None
        assert "3_4" in selected_3_4.kernel_name

    def test_hidden_dim_fallback(self):
        """Test kernel selection with nearest hidden_dim."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")

        # Populate lookup table for hidden_dim=4096
        config = KernelConfig(
            kernel_name="test_kernel",
            tile_size_m=64,
            tile_size_n=64,
            simdgroups=8,
        )
        tuner.config.lookup_table[4][4096][(4,)] = config

        # Select for hidden_dim=5120 (should find nearest)
        selected = tuner.select_kernel(4, 5120, [4])

        assert selected is not None
        assert selected.kernel_name == "test_kernel"

    def test_mixed_bit_width_config_mapping(self):
        """Test that config mapping has entries for all bit widths."""
        config = AutotuneConfig(device_name="M4 Max", device_family="M4")

        # Check tile size mapping
        for bit_width in [2, 3, 4, 8]:
            assert bit_width in config.tile_size_mapping
            assert isinstance(config.tile_size_mapping[bit_width], int)
            assert config.tile_size_mapping[bit_width] > 0

        # Check simdgroup mapping
        for bit_width in [2, 3, 4, 8]:
            assert bit_width in config.simdgroup_mapping
            assert isinstance(config.simdgroup_mapping[bit_width], int)
            assert config.simdgroup_mapping[bit_width] > 0

    def test_bit_width_priority_in_tile_sizes(self):
        """Test that lower bit-widths get larger tiles."""
        tuner = MixedBPWAutoTuner(device_name="M4 Max")

        configs = tuner._generate_configs(bit_widths=[2, 4], use_decode=False)

        # Find max tile size for each bit width
        max_tile_2bit = max(cfg.tile_size_n for cfg in configs if cfg.tile_size_n >= 96)
        max_tile_4bit = max(cfg.tile_size_n for cfg in configs if cfg.tile_size_n <= 64)

        # 2-bit should have larger or equal tiles
        assert max_tile_2bit >= max_tile_4bit
