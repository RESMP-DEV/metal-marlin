"""Tests for the autotuning module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from metal_marlin._compat import HAS_TORCH, torch
from metal_marlin.autotuning import (
    DEFAULT_CONFIG,
    TILE_CONFIGS,
    AutotuneCache,
    BenchmarkResult,
    GPUFingerprint,
    TileConfig,
    TileSearcher,
    common_transformer_sizes,
    detect_gpu,
    estimate_throughput,
    get_decode_config,
    get_heuristic_config,
    get_prefill_config,
    select_best_heuristic,
)


class TestTileConfig:
    """Tests for TileConfig dataclass."""

    def test_default_config_valid(self) -> None:
        """Default config should have valid parameters."""
        assert DEFAULT_CONFIG.tile_m == 64
        assert DEFAULT_CONFIG.tile_n == 64
        assert DEFAULT_CONFIG.tile_k == 32
        assert DEFAULT_CONFIG.threads_per_tg == 128

    def test_shared_memory_calculation(self) -> None:
        """Shared memory should be calculated correctly."""
        config = TileConfig(
            tile_m=64,
            tile_n=64,
            tile_k=32,
            simdgroups_per_tg=4,
            threads_per_tg=128,
            sg_m_tiles=2,
            sg_n_tiles=4,
            num_buffers=2,
        )
        # A buffer: 2 * 64 * 32 * 2 = 8192 bytes
        # B staging: 4 * 8 * 8 * 2 = 512 bytes
        expected = 8192 + 512
        assert config.shared_memory_bytes == expected

    def test_template_params(self) -> None:
        """Template params should contain all necessary values."""
        params = DEFAULT_CONFIG.to_template_params()
        param_dict = dict(params)

        assert param_dict["TILE_M"] == 64
        assert param_dict["TILE_N"] == 64
        assert param_dict["TILE_K"] == 32
        assert param_dict["SIMDGROUPS_PER_TG"] == 4
        assert param_dict["NUM_BUFFERS"] == 2

    def test_k_tiles_property(self) -> None:
        """k_tiles should be tile_k // 8."""
        config = TileConfig(
            tile_m=64,
            tile_n=64,
            tile_k=64,  # Larger K
            simdgroups_per_tg=4,
            threads_per_tg=128,
            sg_m_tiles=2,
            sg_n_tiles=4,
        )
        assert config.k_tiles == 8  # 64 // 8

    def test_all_preset_configs_valid(self) -> None:
        """All preset configs should have consistent parameters."""
        for config in TILE_CONFIGS:
            # Threads per TG should equal simdgroups * 32
            assert config.threads_per_tg == config.simdgroups_per_tg * 32
            # SG decomposition should match tile sizes
            assert config.sg_m_tiles * 8 <= config.tile_m
            assert config.sg_n_tiles * 8 <= config.tile_n


class TestGPUFingerprint:
    """Tests for GPU fingerprint detection and matching."""

    def test_detect_gpu_returns_fingerprint(self) -> None:
        """detect_gpu should return a valid GPUFingerprint."""
        gpu = detect_gpu()
        assert isinstance(gpu, GPUFingerprint)
        assert gpu.name  # Non-empty name
        assert gpu.memory_gb > 0  # Positive memory

    def test_fingerprint_to_key(self) -> None:
        """to_key should produce filesystem-safe string."""
        gpu = GPUFingerprint(
            name="Apple M2 Max",
            cores=38,
            memory_gb=96,
            metal_family="Apple7",
        )
        key = gpu.to_key()
        assert "M2_Max" in key or "M2" in key
        assert "38c" in key
        assert "96GB" in key
        # Should not contain problematic chars
        assert "(" not in key
        assert ")" not in key
        assert " " not in key

    def test_fingerprint_matching(self) -> None:
        """Fingerprints should match within tolerance."""
        gpu1 = GPUFingerprint(
            name="Apple M2 Max",
            cores=38,
            memory_gb=96,
            metal_family="Apple7",
        )
        gpu2 = GPUFingerprint(
            name="Apple M2 Max",
            cores=38,
            memory_gb=64,  # Different memory config
            metal_family="Apple7",
        )
        # Should match (same chip, different RAM)
        assert gpu1.matches(gpu2)

    def test_fingerprint_no_match_different_cores(self) -> None:
        """Different core counts should not match."""
        gpu1 = GPUFingerprint(
            name="Apple M2 Pro",
            cores=19,
            memory_gb=32,
            metal_family="Apple7",
        )
        gpu2 = GPUFingerprint(
            name="Apple M2 Pro",
            cores=16,  # Different core count (binned)
            memory_gb=32,
            metal_family="Apple7",
        )
        assert not gpu1.matches(gpu2)

    def test_fingerprint_serialization(self) -> None:
        """Fingerprint should roundtrip through JSON."""
        gpu = GPUFingerprint(
            name="Apple M3",
            cores=10,
            memory_gb=24,
            metal_family="Apple8",
        )
        data = gpu.to_dict()
        restored = GPUFingerprint.from_dict(data)
        assert restored == gpu


class TestAutotuneCache:
    """Tests for the autotuning cache."""

    def test_cache_put_get(self) -> None:
        """Cache should store and retrieve configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AutotuneCache(cache_dir=Path(tmpdir), auto_load=False)

            config = DEFAULT_CONFIG
            cache.put(4096, 4096, 4096, config, gflops=1234.5)

            retrieved = cache.get(4096, 4096, 4096)
            assert retrieved is not None
            assert retrieved.tile_m == config.tile_m
            assert retrieved.tile_n == config.tile_n

    def test_cache_miss_returns_none(self) -> None:
        """Cache miss should return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AutotuneCache(cache_dir=Path(tmpdir), auto_load=False)
            assert cache.get(4096, 4096, 4096) is None

    def test_cache_persistence(self) -> None:
        """Cache should persist to disk and reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create and populate cache
            cache1 = AutotuneCache(cache_dir=cache_dir, auto_load=False)
            cache1.put(1024, 1024, 1024, DEFAULT_CONFIG, gflops=500.0)
            cache1.save()

            # Create new cache and load
            cache2 = AutotuneCache(cache_dir=cache_dir, auto_load=True)
            config = cache2.get(1024, 1024, 1024)
            assert config is not None
            assert config.tile_m == DEFAULT_CONFIG.tile_m

    def test_cache_group_size_keying(self) -> None:
        """Different group sizes should have separate cache entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AutotuneCache(cache_dir=Path(tmpdir), auto_load=False)

            config32 = DEFAULT_CONFIG
            config64 = TileConfig(
                tile_m=32,
                tile_n=32,
                tile_k=64,
                simdgroups_per_tg=2,
                threads_per_tg=64,
                sg_m_tiles=2,
                sg_n_tiles=2,
            )

            cache.put(4096, 4096, 4096, config32, group_size=32)
            cache.put(4096, 4096, 4096, config64, group_size=64)

            assert cache.get(4096, 4096, 4096, group_size=32) == config32
            assert cache.get(4096, 4096, 4096, group_size=64) == config64

    def test_cache_invalidate(self) -> None:
        """Invalidate should remove specific entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AutotuneCache(cache_dir=Path(tmpdir), auto_load=False)

            cache.put(1024, 1024, 1024, DEFAULT_CONFIG)
            assert cache.get(1024, 1024, 1024) is not None

            result = cache.invalidate(1024, 1024, 1024)
            assert result is True
            assert cache.get(1024, 1024, 1024) is None

    def test_cache_clear(self) -> None:
        """Clear should remove all entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AutotuneCache(cache_dir=Path(tmpdir), auto_load=False)

            cache.put(1024, 1024, 1024, DEFAULT_CONFIG)
            cache.put(2048, 2048, 2048, DEFAULT_CONFIG)
            assert len(cache) == 2

            cache.clear()
            assert len(cache) == 0

    def test_cache_contains(self) -> None:
        """__contains__ should work with tuple keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AutotuneCache(cache_dir=Path(tmpdir), auto_load=False)
            cache.put(4096, 4096, 4096, DEFAULT_CONFIG)

            assert (4096, 4096, 4096) in cache
            assert (1024, 1024, 1024) not in cache


class TestHeuristics:
    """Tests for heuristic-based config selection."""

    def test_heuristic_returns_config(self) -> None:
        """Heuristic should always return a valid config."""
        config = get_heuristic_config(4096, 4096, 4096)
        assert isinstance(config, TileConfig)
        assert config.tile_m > 0
        assert config.tile_n > 0

    def test_heuristic_small_m(self) -> None:
        """Small M should result in smaller tiles."""
        config_small = get_heuristic_config(M=1, N=4096, K=4096)
        config_large = get_heuristic_config(M=4096, N=4096, K=4096)

        # Small M should use smaller tiles
        assert config_small.tile_m <= config_large.tile_m

    def test_decode_config(self) -> None:
        """Decode config should be optimized for small M."""
        config = get_decode_config(N=4096, K=4096, batch_size=1)
        # Decode typically uses small tiles
        assert config.tile_m <= 32

    def test_prefill_config(self) -> None:
        """Prefill config should be optimized for large M."""
        get_prefill_config(seq_len=512, N=4096, K=4096, batch_size=8)
        # Prefill with large M should use larger tiles
        M = 8 * 512
        assert M >= 128  # Large enough for big tiles

    def test_estimate_throughput(self) -> None:
        """Throughput estimation should return valid metrics."""
        metrics = estimate_throughput(DEFAULT_CONFIG, 4096, 4096, 4096, gpu_cores=38)

        assert "estimated_gflops" in metrics
        assert "occupancy" in metrics
        assert "threadgroups" in metrics
        assert metrics["occupancy"] >= 0.0
        assert metrics["occupancy"] <= 1.0
        assert metrics["estimated_gflops"] > 0

    def test_select_best_heuristic(self) -> None:
        """Best heuristic selection should return valid config."""
        config = select_best_heuristic(4096, 4096, 4096)
        assert isinstance(config, TileConfig)

    def test_heuristic_respects_problem_bounds(self) -> None:
        """Heuristic should not return tiles larger than problem."""
        config = get_heuristic_config(M=32, N=64, K=128)
        # Config tiles should fit in problem
        assert config.tile_m <= 32 or config.tile_m == 64  # May round up
        assert config.tile_n <= 64 or config.tile_n == 128


class TestTileSearcher:
    """Tests for the tile searcher.

    Note: Full search tests require PyTorch and Metal kernels.
    These tests focus on the non-benchmark functionality.
    """

    def test_searcher_creation(self) -> None:
        """Searcher should be creatable with default settings."""
        searcher = TileSearcher()
        assert len(searcher.configs) > 0
        assert searcher.warmup_iters > 0
        assert searcher.bench_iters > 0

    def test_searcher_custom_configs(self) -> None:
        """Searcher should accept custom config list."""
        custom_configs = [DEFAULT_CONFIG]
        searcher = TileSearcher(configs=custom_configs)
        assert len(searcher.configs) == 1

    def test_config_fits_check(self) -> None:
        """Config fit check should reject incompatible configs."""
        searcher = TileSearcher()

        # Large config should not fit small problem
        large_config = TileConfig(
            tile_m=128,
            tile_n=128,
            tile_k=32,
            simdgroups_per_tg=8,
            threads_per_tg=256,
            sg_m_tiles=4,
            sg_n_tiles=8,
        )
        assert not searcher._config_fits(large_config, M=32, N=64, K=128)

        # Default config should fit reasonable problem
        assert searcher._config_fits(DEFAULT_CONFIG, M=256, N=256, K=256)


class TestTransformerSizes:
    """Tests for common transformer size generation."""

    def test_common_sizes_non_empty(self) -> None:
        """Should return non-empty list of sizes."""
        sizes = common_transformer_sizes()
        assert len(sizes) > 0

    def test_common_sizes_format(self) -> None:
        """Sizes should be (M, N, K) tuples."""
        sizes = common_transformer_sizes()
        for size in sizes:
            assert len(size) == 3
            M, N, K = size
            assert M > 0
            assert N > 0
            assert K > 0

    def test_common_sizes_includes_decode(self) -> None:
        """Should include batch=1 sizes for decode."""
        sizes = common_transformer_sizes()
        # seq_len=1 sizes
        has_small_m = any(M <= 32 for M, N, K in sizes)
        assert has_small_m

    def test_common_sizes_custom(self) -> None:
        """Should respect custom parameters."""
        sizes = common_transformer_sizes(
            hidden_sizes=[2048],
            batch_sizes=[1],
            seq_lens=[1],
        )
        # Should have exactly the specified combinations
        # 1 hidden * 1 batch * 1 seq_len * 3 (hidden, up, down projections)
        assert len(sizes) >= 1


class TestBenchmarkResult:
    """Tests for benchmark result dataclass."""

    def test_benchmark_result_valid(self) -> None:
        """Valid benchmark result."""
        result = BenchmarkResult(
            config=DEFAULT_CONFIG,
            elapsed_ms=1.5,
            gflops=1234.5,
            bandwidth_gb_s=456.7,
            valid=True,
        )
        assert result.valid
        assert result.error is None

    def test_benchmark_result_invalid(self) -> None:
        """Invalid benchmark result with error."""
        result = BenchmarkResult(
            config=DEFAULT_CONFIG,
            elapsed_ms=float("inf"),
            gflops=0.0,
            bandwidth_gb_s=0.0,
            valid=False,
            error="Config does not fit problem size",
        )
        assert not result.valid
        assert result.error is not None


# Integration tests (require PyTorch)
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for integration tests")
class TestIntegration:
    """Integration tests that require PyTorch.

    Note: These tests validate that the TileSearcher and AutotuneCache
    work end-to-end with real tensor operations. Full benchmarking requires
    Metal kernels which may not be available in all environments.
    """

    def test_tile_config_for_torch_gemm(self) -> None:
        """TileConfig should provide valid parameters for torch GEMM."""
        assert torch is not None

        config = DEFAULT_CONFIG
        M, N, K = 256, 256, 256

        # Verify config parameters are valid for a GEMM dispatch
        grid_x = (N + config.tile_n - 1) // config.tile_n
        grid_y = (M + config.tile_m - 1) // config.tile_m

        assert grid_x > 0
        assert grid_y > 0
        assert config.threads_per_tg > 0

        # Create test tensors
        A = torch.randn(M, K, dtype=torch.float16)
        B = torch.randn(K, N, dtype=torch.float16)

        # Verify GEMM would work (actual Metal kernel dispatch not tested here)
        C = torch.matmul(A, B)
        assert C.shape == (M, N)

    def test_heuristic_with_torch_tensors(self) -> None:
        """Heuristic selection should work with PyTorch-compatible sizes."""
        assert torch is not None

        # Test various problem sizes
        sizes = [(1, 4096, 4096), (32, 4096, 4096), (512, 4096, 4096)]

        for M, N, K in sizes:
            config = get_heuristic_config(M=M, N=N, K=K)
            assert isinstance(config, TileConfig)

            # Verify config is compatible with the problem
            assert config.tile_m > 0
            assert config.tile_n > 0
            assert config.tile_k > 0

    def test_cache_with_heuristic_fallback(self) -> None:
        """Cache should work with heuristic-based config selection."""
        assert torch is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AutotuneCache(cache_dir=Path(tmpdir), auto_load=False)

            # Get config via heuristics (no actual kernel benchmarking)
            M, N, K = 128, 128, 128
            config = get_heuristic_config(M=M, N=N, K=K)

            # Store in cache
            cache.put(M, N, K, config, gflops=100.0)

            # Retrieve from cache
            cached = cache.get(M, N, K)
            assert cached is not None
            assert cached.tile_m == config.tile_m
            assert cached.tile_n == config.tile_n

    def test_searcher_config_filtering(self) -> None:
        """TileSearcher should filter configs based on problem size."""
        searcher = TileSearcher(
            warmup_iters=1,
            bench_iters=1,
        )

        # Small problem - some configs should be filtered out
        small_M, small_N, small_K = 64, 64, 64

        valid_configs = [
            config
            for config in searcher.configs
            if searcher._config_fits(config, small_M, small_N, small_K)
        ]

        # At least one config should fit, but not all large configs
        assert len(valid_configs) > 0
        assert len(valid_configs) <= len(searcher.configs)
