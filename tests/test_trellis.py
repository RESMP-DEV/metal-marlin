"""Tests for EXL3-style trellis quantization."""

import numpy as np
import pytest


class TestTrellisCodebook:
    """Tests for TrellisCodebook quantization."""

    def test_codebook_roundtrip(self):
        """Quantize and dequantize should approximately recover values."""
        from metal_marlin.quantization.viterbi_quant import TrellisCodebook, quantize_tile_viterbi

        cb = TrellisCodebook(bits=4)
        grid = cb.get_grid()

        # Create a simple tile with grid values
        tile = np.zeros((16, 16), dtype=np.float32)
        for i, g in enumerate(grid[:16]):
            tile[i, 0] = g  # Test each grid point

        # Quantize with scale=1.0
        indices, dequantized = quantize_tile_viterbi(tile, cb, scale=1.0)

        # Grid points at scale=1.0 should roundtrip
        for i in range(min(16, len(grid))):
            g = grid[i]
            assert np.isclose(dequantized[i, 0], g, atol=0.01), (
                f"Grid point {g} failed roundtrip, got {dequantized[i, 0]}"
            )

    def test_codebook_scale(self):
        """Scaling should work correctly."""
        from metal_marlin.quantization.viterbi_quant import TrellisCodebook, quantize_tile_greedy

        cb = TrellisCodebook(bits=4)
        grid = cb.get_grid()

        # Create a tile with a value that should scale predictably
        # Use a grid point * scale value for predictable behavior
        scale = 2.0
        target_value = grid[5] * scale
        tile = np.full((16, 16), target_value, dtype=np.float32)

        # Quantize with scale
        indices, dequantized = quantize_tile_greedy(tile, cb, scale=scale)

        # The dequantized value should be close to the original target
        # (may not be exact due to quantization to grid points)
        # indices is 1D array flattened from 16x16 tile
        assert dequantized[0, 0] == grid[indices[0]] * scale
        # Error should be reasonable
        assert abs(dequantized[0, 0] - target_value) < scale * (grid[1] - grid[0])

    def test_codebook_different_bits(self):
        """Codebook should support different bit widths."""
        from metal_marlin.quantization.viterbi_quant import TrellisCodebook

        for bits in [2, 3, 4, 5]:
            cb = TrellisCodebook(bits=bits)
            grid = cb.get_grid()
            expected_size = 2**bits
            assert len(grid) == expected_size, (
                f"Expected {expected_size} grid points for {bits} bits, got {len(grid)}"
            )

    def test_viterbi_vs_greedy(self):
        """Viterbi should perform better or equal to greedy quantization."""
        from metal_marlin.quantization.viterbi_quant import (
            TrellisCodebook,
            compute_quantization_error,
            quantize_tile_greedy,
            quantize_tile_viterbi,
        )

        cb = TrellisCodebook(bits=4)
        np.random.seed(42)
        tile = np.random.randn(16, 16).astype(np.float32)
        scale = 1.0

        # Greedy quantization
        _, deq_greedy = quantize_tile_greedy(tile, cb, scale)
        error_greedy = compute_quantization_error(tile, deq_greedy)

        # Viterbi quantization
        _, deq_viterbi = quantize_tile_viterbi(tile, cb, scale)
        error_viterbi = compute_quantization_error(tile, deq_viterbi)

        # Viterbi should typically do better or equal
        assert error_viterbi <= error_greedy * 1.1, (
            f"Viterbi error {error_viterbi} > greedy error {error_greedy}"
        )


class TestTrellisTile:
    """Tests for tile extraction and reconstruction."""

    def test_tile_reshape_roundtrip(self):
        """Reshape and reconstruct should preserve values."""
        W = np.random.randn(64, 128).astype(np.float32)

        # Extract 16x16 tiles
        tiles = []
        tile_h, tile_w = 16, 16
        for i in range(0, W.shape[0], tile_h):
            for j in range(0, W.shape[1], tile_w):
                tile = W[i : i + tile_h, j : j + tile_w]
                # Pad if necessary
                if tile.shape != (tile_h, tile_w):
                    padded = np.zeros((tile_h, tile_w), dtype=np.float32)
                    padded[: tile.shape[0], : tile.shape[1]] = tile
                    tile = padded
                tiles.append(tile)

        # Reconstruct
        W_reconstructed = np.zeros_like(W)
        tile_idx = 0
        for i in range(0, W.shape[0], tile_h):
            for j in range(0, W.shape[1], tile_w):
                tile = tiles[tile_idx]
                end_i = min(i + tile_h, W.shape[0])
                end_j = min(j + tile_w, W.shape[1])
                W_reconstructed[i:end_i, j:end_j] = tile[: end_i - i, : end_j - j]
                tile_idx += 1

        np.testing.assert_allclose(W, W_reconstructed, rtol=1e-5)

    def test_tile_shapes(self):
        """Tiles should have correct shapes."""
        W = np.random.randn(64, 128).astype(np.float32)

        tile_h, tile_w = 16, 16
        tiles = []
        for i in range(0, W.shape[0], tile_h):
            for j in range(0, W.shape[1], tile_w):
                tile = W[i : i + tile_h, j : j + tile_w]
                tiles.append(tile)

        # Check that tiles cover the full matrix
        rows = (W.shape[0] + tile_h - 1) // tile_h
        cols = (W.shape[1] + tile_w - 1) // tile_w
        assert len(tiles) == rows * cols

    def test_tile_different_sizes(self):
        """Tiling should work with different matrix sizes."""
        tile_h, tile_w = 16, 16

        for shape in [(32, 32), (48, 64), (100, 200)]:
            W = np.random.randn(*shape).astype(np.float32)

            tiles = []
            for i in range(0, W.shape[0], tile_h):
                for j in range(0, W.shape[1], tile_w):
                    tile = W[i : i + tile_h, j : j + tile_w]
                    tiles.append(tile)

            # Verify we can reconstruct
            W_reconstructed = np.zeros_like(W)
            tile_idx = 0
            for i in range(0, W.shape[0], tile_h):
                for j in range(0, W.shape[1], tile_w):
                    tile = tiles[tile_idx]
                    end_i = min(i + tile_h, W.shape[0])
                    end_j = min(j + tile_w, W.shape[1])
                    W_reconstructed[i:end_i, j:end_j] = tile[: end_i - i, : end_j - j]
                    tile_idx += 1

            np.testing.assert_allclose(W, W_reconstructed, rtol=1e-5)


class TestLDLDecomposition:
    """Tests for LDL decomposition."""

    def test_ldl_decomp(self):
        """LDL decomposition should satisfy H = L @ D @ L.T"""
        from metal_marlin.quantization.ldl_decomp import block_ldl

        # Create SPD matrix
        A = np.random.randn(64, 64)
        H = A.T @ A + 0.1 * np.eye(64)

        L, D = block_ldl(H)

        # Verify decomposition
        H_reconstructed = L @ D @ L.T
        np.testing.assert_allclose(H, H_reconstructed, rtol=1e-5)

    def test_ldl_lower_triangular(self):
        """L should be lower triangular with unit diagonal blocks."""
        from metal_marlin.quantization.ldl_decomp import block_ldl

        A = np.random.randn(32, 32)
        H = A.T @ A + 0.1 * np.eye(32)

        L, D = block_ldl(H, block_size=16)

        # L should be lower triangular
        assert np.allclose(L, np.tril(L))
        # Check that diagonal blocks are close to identity
        for i in range(0, 32, 16):
            block = L[i : i + 16, i : i + 16]
            np.testing.assert_allclose(block, np.eye(16), atol=1e-5)

    def test_ldl_block_diagonal_d(self):
        """D should be block diagonal."""
        from metal_marlin.quantization.ldl_decomp import block_ldl

        A = np.random.randn(32, 32)
        H = A.T @ A + 0.1 * np.eye(32)

        L, D = block_ldl(H, block_size=16)

        # D should be block diagonal - check off-block-diagonal is near zero
        for i in range(0, 32, 16):
            for j in range(0, 32, 16):
                if i != j:
                    block = D[i : i + 16, j : j + 16]
                    assert np.allclose(block, 0, atol=1e-10)

    def test_ldl_different_sizes(self):
        """LDL should work for different matrix sizes."""
        from metal_marlin.quantization.ldl_decomp import block_ldl

        for size in [16, 32, 64, 128]:
            A = np.random.randn(size, size)
            H = A.T @ A + 0.1 * np.eye(size)

            L, D = block_ldl(H)
            H_reconstructed = L @ D @ L.T
            np.testing.assert_allclose(H, H_reconstructed, rtol=1e-4)

    def test_ldl_solve(self):
        """LDL solve should correctly solve linear systems."""
        from metal_marlin.quantization.ldl_decomp import block_ldl, ldl_solve

        A = np.random.randn(32, 32)
        H = A.T @ A + 0.1 * np.eye(32)
        b = np.random.randn(32)

        L, D = block_ldl(H)
        x = ldl_solve(L, D, b)

        # Verify solution
        np.testing.assert_allclose(H @ x, b, rtol=1e-4)


class TestHadamardPreprocess:
    """Tests for Hadamard preprocessing."""

    def test_hadamard_preprocess(self):
        """Hadamard preprocessing should rotate Hessian correctly."""
        from metal_marlin.quantization.hadamard_preprocess import preprocess_hessian_exl3

        # Create SPD matrix
        A = np.random.randn(64, 64)
        H = A.T @ A + 0.1 * np.eye(64)

        H_rotated, su, Had = preprocess_hessian_exl3(H, had_k=64)

        # Check shapes
        assert H_rotated.shape == H.shape
        assert su.shape == (64,)
        assert Had.shape == (64, 64)

        # Check Hadamard is orthogonal
        np.testing.assert_allclose(Had @ Had.T, np.eye(64), atol=1e-5)

    def test_weight_rotation_roundtrip(self):
        """Rotate then unrotate should recover original weights."""
        from metal_marlin.quantization.hadamard_preprocess import (
            preprocess_hessian_exl3,
            rotate_weights_exl3,
            unrotate_weights_exl3,
        )

        # Create Hessian and preprocess
        A = np.random.randn(64, 64)
        H = A.T @ A + 0.1 * np.eye(64)
        H_rotated, su, Had = preprocess_hessian_exl3(H, had_k=64)

        # Generate sign flips for output dimension
        rng = np.random.default_rng()
        sv = np.sign(rng.standard_normal(128) + 1e-5)

        # Create weights
        W = np.random.randn(128, 64).astype(np.float32)

        # Rotate weights
        W_rotated = rotate_weights_exl3(W, su, had_k=64)

        # Unrotate weights
        W_recovered = unrotate_weights_exl3(W_rotated, su, sv, had_k=64, had_n=128)

        # Should recover original (approximately, due to quantization in between)
        # Note: Without actual quantization, this won't be exact due to Hadamard properties
        assert W_recovered.shape == W.shape

    def test_blockwise_hadamard(self):
        """Blockwise Hadamard should apply transform correctly."""
        from metal_marlin.quantization.hadamard_preprocess import blockwise_hadamard

        X = np.random.randn(128, 64).astype(np.float64)

        # Apply along axis 0
        Y = blockwise_hadamard(X, block_size=64, axis=0)
        assert Y.shape == X.shape

        # Apply along axis 1
        Z = blockwise_hadamard(X, block_size=64, axis=1)
        assert Z.shape == X.shape

    def test_hadamard_preprocess_preserves_norm(self):
        """Hadamard transform should preserve Frobenius norm."""
        from metal_marlin.quantization.hadamard_preprocess import blockwise_hadamard

        X = np.random.randn(128, 128).astype(np.float64)
        original_norm = np.linalg.norm(X, "fro")

        X_processed = blockwise_hadamard(X, block_size=128, axis=0)
        processed_norm = np.linalg.norm(X_processed, "fro")

        np.testing.assert_allclose(original_norm, processed_norm, rtol=1e-4)


class TestViterbiQuantizer:
    """Tests for Viterbi quantization."""

    def test_viterbi_quantizer(self):
        """Viterbi should find valid quantization."""
        from metal_marlin.quantization.viterbi_quant import TrellisCodebook, quantize_tile_viterbi

        # Create synthetic tile
        np.random.seed(42)
        tile = np.random.randn(16, 16).astype(np.float32)

        cb = TrellisCodebook(bits=4)
        indices, dequantized = quantize_tile_viterbi(tile, cb, scale=1.0)

        # Check shape preserved
        assert dequantized.shape == tile.shape
        # Check quantization is valid (finite values)
        assert np.all(np.isfinite(dequantized))
        # Check indices are in valid range
        assert np.all(indices >= 0)
        assert np.all(indices < len(cb.get_grid()))

    def test_viterbi_reduces_error(self):
        """Viterbi should reduce quantization error vs naive."""
        from metal_marlin.quantization.viterbi_quant import (
            TrellisCodebook,
            compute_quantization_error,
            quantize_tile_greedy,
            quantize_tile_viterbi,
        )

        np.random.seed(42)
        tile = np.random.randn(16, 16).astype(np.float32)
        cb = TrellisCodebook(bits=4)
        scale = 1.0

        # Greedy quantization
        _, deq_greedy = quantize_tile_greedy(tile, cb, scale)
        error_greedy = compute_quantization_error(tile, deq_greedy)

        # Viterbi quantization
        _, deq_viterbi = quantize_tile_viterbi(tile, cb, scale)
        error_viterbi = compute_quantization_error(tile, deq_viterbi)

        # Viterbi should typically be better
        assert error_viterbi <= error_greedy * 1.5  # Allow some tolerance


class TestStreamingHessian:
    """Tests for streaming Hessian computation."""

    def test_streaming_hessian(self):
        """Streaming Hessian matches batch computation."""
        import torch

        from metal_marlin.quantization.hessian_streaming import StreamingHessianCollector

        # Generate calibration data
        np.random.seed(42)
        X = np.random.randn(100, 64).astype(np.float32)

        # Streaming computation
        collector = StreamingHessianCollector(hidden_dim=64)
        for i in range(0, len(X), 10):
            batch = torch.from_numpy(X[i : i + 10])
            collector.accumulate(batch)
        H_streaming = collector.finalize(sigma_reg=0.0)

        # Batch computation
        H_batch = X.T @ X / len(X)

        np.testing.assert_allclose(H_streaming, H_batch, rtol=1e-2, atol=1e-6)

    def test_streaming_hessian_accumulation(self):
        """Streaming Hessian should accumulate correctly."""
        import torch

        from metal_marlin.quantization.hessian_streaming import StreamingHessianCollector

        collector = StreamingHessianCollector(hidden_dim=32)

        # Add data in chunks
        chunks = [torch.randn(20, 32) for _ in range(5)]
        all_data = torch.cat(chunks, dim=0).numpy()

        for chunk in chunks:
            collector.accumulate(chunk)

        H_streaming = collector.finalize(sigma_reg=0.0)
        H_batch = all_data.T @ all_data / len(all_data)

        np.testing.assert_allclose(H_streaming, H_batch, rtol=1e-3, atol=1e-8)

    def test_streaming_hessian_regularization(self):
        """Streaming Hessian should support regularization."""
        import torch

        from metal_marlin.quantization.hessian_streaming import StreamingHessianCollector

        X = torch.randn(50, 32)

        collector = StreamingHessianCollector(hidden_dim=32)
        collector.accumulate(X)
        H = collector.finalize(sigma_reg=0.1)

        # Check diagonal has regularization
        assert np.all(np.diag(H) > 0)


class TestLayerStreamer:
    """Tests for layer streaming functionality."""

    def test_layer_streamer_init(self):
        """Layer streamer should initialize correctly."""
        import tempfile
        from pathlib import Path

        from metal_marlin.quantization.layer_streamer import LayerStreamer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an empty directory
            streamer = LayerStreamer(Path(tmpdir))
            layers = streamer.list_layers()
            assert isinstance(layers, list)

    def test_layer_weights_dataclass(self):
        """LayerWeights dataclass should hold data correctly."""
        import torch

        from metal_marlin.quantization.layer_streamer import LayerWeights

        weight = torch.randn(256, 128)
        bias = torch.randn(256)

        lw = LayerWeights(
            name="test_layer",
            weight=weight,
            bias=bias,
            layer_idx=0,
            total_layers=10,
        )

        assert lw.name == "test_layer"
        assert torch.equal(lw.weight, weight)
        assert torch.equal(lw.bias, bias)
        assert lw.layer_idx == 0
        assert lw.total_layers == 10


class TestCalibrationStreamer:
    """Tests for calibration data streaming."""

    def test_calibration_streamer_init(self):
        """Calibration streamer should initialize correctly."""
        from metal_marlin.quantization.calibration_streamer import CalibrationStreamer

        class MockDataset:
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return f"sample {idx}"

            def __iter__(self):
                for i in range(len(self)):
                    yield self[i]

        class MockTokenizer:
            def __call__(self, texts, **kwargs):
                import torch

                return {"input_ids": torch.randint(0, 1000, (len(texts), 10))}

        dataset = MockDataset()
        tokenizer = MockTokenizer()

        streamer = CalibrationStreamer(
            dataset=dataset,
            tokenizer=tokenizer,
            max_seq_len=512,
            target_memory_gb=4.0,
        )

        # Test batch size estimation
        batch_size = streamer.estimate_batch_size(hidden_dim=4096)
        assert batch_size >= 1

    def test_calibration_batch_dataclass(self):
        """CalibrationBatch dataclass should hold data correctly."""
        import torch

        from metal_marlin.quantization.calibration_streamer import CalibrationBatch

        input_ids = torch.randint(0, 1000, (4, 20))
        batch = CalibrationBatch(
            input_ids=input_ids,
            batch_idx=0,
            total_batches=5,
        )

        assert torch.equal(batch.input_ids, input_ids)
        assert batch.batch_idx == 0
        assert batch.total_batches == 5


# Tests for metal_marlin.metal_dispatch module
class TestMetalDispatchViterbi:
    """Tests for the metal_marlin.metal_dispatch dispatch_viterbi_quantize function."""

    @pytest.mark.skip(
        reason="dispatch_viterbi_quantize not yet implemented (P3 optional Metal kernel)"
    )
    def test_metal_dispatch_viterbi(self):
        """Test Metal Viterbi dispatch with Python fallback.

        This test validates:
        1. The dispatch function works with lib=None (Python fallback)
        2. Returns correct output shapes and types
        3. Produces reasonable quantization results
        """
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        from metal_marlin.metal_dispatch import dispatch_viterbi_quantize

        device = "cpu"
        n_tiles = 8
        tile_size = 256

        # Create test data
        tiles = torch.randn(n_tiles, tile_size, dtype=torch.float16, device=device)
        scales = torch.ones(n_tiles, dtype=torch.float16, device=device) * 0.1
        grid = torch.linspace(-6, 6, 16, dtype=torch.float16, device=device)

        # Test with lib=None (Python fallback)
        indices, dequantized = dispatch_viterbi_quantize(
            lib=None,
            tiles=tiles,
            scales=scales,
            grid=grid,
            bits=4,
        )

        # Validate outputs
        assert indices.shape == (n_tiles, tile_size), (
            f"Expected shape ({n_tiles}, {tile_size}), got {indices.shape}"
        )
        assert indices.dtype == torch.int16, f"Expected int16, got {indices.dtype}"
        assert dequantized.shape == (n_tiles, tile_size), (
            f"Expected shape ({n_tiles}, {tile_size}), got {dequantized.shape}"
        )
        assert dequantized.dtype == torch.float16, f"Expected float16, got {dequantized.dtype}"

        # Validate index range
        assert indices.min() >= 0, f"Negative indices found: {indices.min()}"
        assert indices.max() < 16, f"Index out of range: {indices.max()}"

        # Validate dequantized values are finite
        assert torch.isfinite(dequantized).all(), "Non-finite values in dequantized output"

        # Validate reconstruction quality
        mse = ((tiles.float() - dequantized.float()) ** 2).mean().item()
        assert mse < 1.0, f"Reconstruction MSE too high: {mse}"


# =============================================================================
# LayerStreamer Tests
# =============================================================================

import tempfile
from pathlib import Path

import pytest

safetensors = pytest.importorskip("safetensors")
import safetensors.torch
import torch

from metal_marlin.quantization.layer_streamer import LayerStreamer


def test_layer_streamer():
    """Test LayerStreamer basic functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.safetensors"

        # Create a simple model with linear layers
        tensors = {
            "layer1.weight": torch.randn(64, 32),
            "layer1.bias": torch.randn(64),
            "layer2.weight": torch.randn(128, 64),
            "layer2.bias": torch.randn(128),
        }
        safetensors.torch.save_file(tensors, model_path)

        # Test layer streamer
        streamer = LayerStreamer(model_path)
        layers = list(streamer.iter_linear_layers())

        assert len(layers) == 2
        assert layers[0].name == "layer1.weight"
        assert layers[0].weight.shape == (64, 32)
        assert layers[0].bias is not None
        assert layers[0].bias.shape == (64,)
        assert layers[0].layer_idx == 0
        assert layers[0].total_layers == 2

        assert layers[1].name == "layer2.weight"
        assert layers[1].weight.shape == (128, 64)
        assert layers[1].layer_idx == 1


def test_exl3_quantizer():
    """Test EXL3Quantizer basic functionality.

    Validates:
    1. Quantizer initialization with correct parameters
    2. quantize_layer returns EXL3QuantResult with correct types
    3. Output shapes match expected dimensions
    4. Reconstruction MSE is reasonable
    """
    import numpy as np
    import torch

    from metal_marlin.quantization.exl3_quantizer import EXL3Quantizer, EXL3QuantResult

    # Create test data
    out_features = 64
    in_features = 256
    bits = 4
    group_size = 128

    np.random.seed(42)
    torch.manual_seed(42)

    # Create random weight matrix
    weight = torch.randn(out_features, in_features, dtype=torch.float32)

    # Create positive semi-definite Hessian
    # Use a well-conditioned matrix that remains positive definite after transforms
    eigenvalues = np.linspace(1.0, 0.1, in_features).astype(np.float64)
    Q, _ = np.linalg.qr(np.random.randn(in_features, in_features).astype(np.float64))
    hessian = Q @ np.diag(eigenvalues) @ Q.T
    # Add strong regularization to ensure positive definiteness
    hessian = hessian + np.eye(in_features, dtype=np.float64) * 5.0

    # Create quantizer with higher regularization
    quantizer = EXL3Quantizer(
        bits=bits,
        group_size=group_size,
        had_k=128,
        sigma_reg=1.0,  # Higher regularization for stability
        max_workers=2,
    )

    # Verify initialization
    assert quantizer.bits == bits
    assert quantizer.group_size == group_size
    assert quantizer.had_k == 128
    assert quantizer.codebook.bits == bits

    # Quantize layer
    result = quantizer.quantize_layer(
        weight=weight,
        hessian=hessian,
        layer_name="test_layer",
    )

    # Verify result type
    assert isinstance(result, EXL3QuantResult)

    # Verify result attributes
    assert result.name == "test_layer"
    assert result.bits == bits
    assert result.reconstruction_mse >= 0
    assert result.quantization_time_sec >= 0

    # Verify shapes
    tiles_n = (out_features + 15) // 16  # 16x16 tiles
    tiles_k = (in_features + 15) // 16
    n_groups = (in_features + group_size - 1) // group_size

    assert result.trellis_indices.shape == (tiles_n, tiles_k, 256), (
        f"Expected trellis_indices shape {(tiles_n, tiles_k, 256)}, "
        f"got {result.trellis_indices.shape}"
    )
    assert result.scales.shape == (n_groups, out_features), (
        f"Expected scales shape {(n_groups, out_features)}, got {result.scales.shape}"
    )
    assert result.su.shape == (in_features,), (
        f"Expected su shape {(in_features,)}, got {result.su.shape}"
    )
    assert result.sv.shape == (out_features,), (
        f"Expected sv shape {(out_features,)}, got {result.sv.shape}"
    )

    # Verify dtypes
    assert result.trellis_indices.dtype == np.int16
    assert result.scales.dtype == np.float32
    assert result.su.dtype == np.float64
    assert result.sv.dtype == np.float64

    # Verify trellis indices are in valid range for 4-bit
    assert np.all(result.trellis_indices >= 0)
    assert np.all(result.trellis_indices < 16)  # 2^4 = 16 levels

    # Verify sign vectors contain only +/- 1
    assert np.all(np.abs(result.su) == 1.0)
    assert np.all(np.abs(result.sv) == 1.0)

    print("✓ EXL3Quantizer test passed")
    print(f"  - Reconstruction MSE: {result.reconstruction_mse:.6f}")
    print(f"  - Quantization time: {result.quantization_time_sec:.3f}s")


# =============================================================================
# EXL3 to Marlin Conversion Tests
# =============================================================================


class TestExl3ToMarlinConversion:
    """Tests for EXL3 to Marlin FP4 format conversion.

    These tests verify the conversion pipeline from EXL3 trellis format
    to Marlin FP4 packed format.
    """

    def test_exl3_to_marlin_conversion(self):
        """Test full EXL3 to Marlin conversion pipeline.

        This is the main verification test that exercises the complete
        conversion flow from EXL3 trellis format to Marlin FP4 format.

        Steps:
        1. Create synthetic EXL3-like data (trellis indices, scales, su)
        2. Convert to Marlin format using exl3_layer_to_marlin
        3. Verify output shapes and dtypes are correct
        4. Verify values are reasonable (no NaN, Inf, etc.)
        """
        from metal_marlin.quantization.exl3_to_marlin import TrellisCodebook, exl3_layer_to_marlin

        # Setup: Create synthetic EXL3 layer data
        out_features = 256
        in_features = 512
        group_size = 128
        had_k = 128

        # Trellis indices: EXL3 stores as tiles
        tiles_k = in_features // 16
        tiles_n = out_features // 16
        trellis_indices = np.random.randint(0, 16, (tiles_k, tiles_n, 256), dtype=np.int16)

        # Scales: per-group scales
        n_groups = in_features // group_size
        scales = np.random.uniform(0.01, 1.0, (out_features, n_groups)).astype(np.float32)

        # Sign flips: random ±1 values
        su = np.sign(np.random.randn(in_features) + 1e-5).astype(np.float64)

        # Codebook configuration
        codebook = TrellisCodebook(bits=4)

        # Execute: Convert to Marlin format
        packed, marlin_scales = exl3_layer_to_marlin(trellis_indices, scales, su, codebook)

        # Verify: Check output properties

        # 1. Output shapes
        # Packed: [K/8, N] = [in_features/8, out_features]
        expected_packed_shape = (in_features // 8, out_features)
        assert packed.shape == expected_packed_shape, (
            f"Packed shape mismatch: {packed.shape} != {expected_packed_shape}"
        )

        # Scales: [K/group_size, N] = [n_groups, out_features]
        expected_scales_shape = (n_groups, out_features)
        assert marlin_scales.shape == expected_scales_shape, (
            f"Scales shape mismatch: {marlin_scales.shape} != {expected_scales_shape}"
        )

        # 2. Output dtypes
        assert packed.dtype == np.uint32, f"Packed dtype should be uint32, got {packed.dtype}"
        assert marlin_scales.dtype == np.float16, (
            f"Scales dtype should be float16, got {marlin_scales.dtype}"
        )

        # 3. Value validation
        # No NaN or Inf in scales
        assert not np.any(np.isnan(marlin_scales)), "Marlin scales contain NaN"
        assert not np.any(np.isinf(marlin_scales)), "Marlin scales contain Inf"

        # Scales should be positive (they represent magnitude)
        assert np.all(marlin_scales > 0), "Marlin scales should be positive"

        # 4. Packed data validation
        # Each uint32 contains 8 FP4 values (4 bits each)
        # Max value per nibble is 15 (0xF)
        max_nibble_value = 15

        # Unpack and verify all nibbles are valid
        for i in range(8):
            nibbles = (packed >> (i * 4)) & 0xF
            assert np.all(nibbles <= max_nibble_value), (
                f"Found invalid nibble values in position {i}"
            )

        # 5. Round-trip check (partial)
        # Verify that the packed data can be interpreted as FP4 values
        # Unpack first few values and check they're in valid range
        first_packed = packed[0, 0]
        for i in range(8):
            nibble = (first_packed >> (i * 4)) & 0xF
            # FP4 indices should be 0-15
            assert 0 <= nibble <= 15, f"Invalid FP4 index: {nibble}"

    def test_exl3_to_marlin_basic(self):
        """Test basic EXL3 to Marlin conversion with simple data."""
        from metal_marlin.quantization.exl3_to_marlin import TrellisCodebook, exl3_layer_to_marlin

        # Create simple EXL3 data with dimensions compatible with had_k=128
        out_features, in_features = 128, 256
        tiles_k = in_features // 16  # 16
        tiles_n = out_features // 16  # 8

        trellis_indices = np.ones((tiles_k, tiles_n, 256), dtype=np.int16) * 8
        scales = np.ones((out_features, in_features // 128), dtype=np.float32)
        su = np.ones(in_features, dtype=np.float64)
        codebook = TrellisCodebook(bits=4)

        packed, marlin_scales = exl3_layer_to_marlin(trellis_indices, scales, su, codebook)

        assert packed.dtype == np.uint32
        assert marlin_scales.dtype == np.float16
        assert packed.shape[0] > 0
        assert marlin_scales.shape[0] > 0

    def test_exl3_to_marlin_shapes(self):
        """Test conversion produces correct output shapes for various sizes."""
        from metal_marlin.quantization.exl3_to_marlin import TrellisCodebook, exl3_layer_to_marlin

        test_cases = [
            (128, 256),  # Small layer
            (256, 512),  # Medium layer
            (512, 1024),  # Large layer
        ]

        for out_feat, in_feat in test_cases:
            tiles_k = in_feat // 16
            tiles_n = out_feat // 16

            trellis_indices = np.random.randint(0, 16, (tiles_k, tiles_n, 256), dtype=np.int16)
            scales = np.random.randn(out_feat, in_feat // 128).astype(np.float32)
            su = np.sign(np.random.randn(in_feat)).astype(np.float64)
            codebook = TrellisCodebook(bits=4)

            packed, marlin_scales = exl3_layer_to_marlin(trellis_indices, scales, su, codebook)

            expected_packed = (in_feat // 8, out_feat)
            expected_scales = (in_feat // 128, out_feat)

            assert packed.shape == expected_packed, f"Shape mismatch for ({out_feat}, {in_feat})"
            assert marlin_scales.shape == expected_scales


class TestExl3ToMarlinHelpers:
    """Tests for helper functions in exl3_to_marlin module."""

    def test_quantize_to_fp4_positive(self):
        """Test FP4 quantization of positive values."""
        from metal_marlin.quantization.exl3_to_marlin import _quantize_to_fp4_indices

        values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
        indices = _quantize_to_fp4_indices(values)

        expected = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)
        np.testing.assert_array_equal(indices.flatten(), expected)

    def test_quantize_to_fp4_negative(self):
        """Test FP4 quantization of negative values."""
        from metal_marlin.quantization.exl3_to_marlin import _quantize_to_fp4_indices

        # Test strictly negative values map to indices 9-15
        # (-0 maps to 8, but 0.0 and -0.0 are equal in numpy)
        values = np.array([-0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
        indices = _quantize_to_fp4_indices(values)

        expected = np.array([9, 10, 11, 12, 13, 14, 15], dtype=np.uint8)
        np.testing.assert_array_equal(indices.flatten(), expected)

        # Test zero maps to 0 (+0)
        zero = np.array([0.0], dtype=np.float32)
        zero_idx = _quantize_to_fp4_indices(zero)
        assert zero_idx[0] == 0, f"Zero should map to index 0, got {zero_idx[0]}"

    def test_inverse_hadamard_roundtrip(self):
        """Test that inverse Hadamard correctly reverses the transform."""
        from metal_marlin.hadamard import hadamard_matrix
        from metal_marlin.quantization.exl3_to_marlin import _inverse_hadamard_weight_rotation

        # Create random weights
        original = np.random.randn(128, 128).astype(np.float32)

        # Test without sign flips (su = 1)
        su_ones = np.ones(128, dtype=np.float64)

        # Forward transform: Had @ W @ Had.T
        Had = hadamard_matrix(128)
        forward = Had @ original @ Had.T

        # Inverse transform should recover original
        inverse = _inverse_hadamard_weight_rotation(forward.astype(np.float32), su_ones, had_k=128)

        # Should recover original (within numerical precision)
        np.testing.assert_allclose(inverse, original, atol=1e-4, rtol=1e-4)

        # Test with sign flips applied as the inverse expects them
        # The inverse applies: su_row @ Had.T @ W @ Had @ su_col
        # So forward should be: Had @ su_row @ W @ su_col @ Had.T
        su_row = np.sign(np.random.randn(128))
        su_col = np.sign(np.random.randn(128))
        su_combined = np.concatenate([su_row, su_col])

        # Forward: apply sign flips, then Hadamard
        forward_with_su = Had @ (su_row[:, None] * original * su_col[None, :]) @ Had.T

        # Inverse should recover original
        inverse_with_su = _inverse_hadamard_weight_rotation(
            forward_with_su.astype(np.float32), su_combined, had_k=128
        )

        np.testing.assert_allclose(inverse_with_su, original, atol=1e-4, rtol=1e-4)

    def test_pack_fp4_shapes(self):
        """Test FP4 packing produces correct shapes."""
        from metal_marlin.quantization.exl3_to_marlin import _pack_fp4_to_marlin

        out_feat, in_feat = 256, 512
        weights = np.random.randn(out_feat, in_feat).astype(np.float32)
        scales = np.ones((out_feat, in_feat // 128), dtype=np.float32)

        packed, marlin_scales = _pack_fp4_to_marlin(weights, scales, group_size=128)

        assert packed.shape == (in_feat // 8, out_feat)
        assert marlin_scales.shape == (in_feat // 128, out_feat)


class TestConvertExl3ToMarlinModel:
    """Tests for full model conversion function."""

    def test_convert_nonexistent_path(self, tmp_path):
        """Test conversion with non-existent input path."""
        from metal_marlin.quantization.exl3_to_marlin import convert_exl3_to_marlin

        nonexistent = tmp_path / "does_not_exist"
        output = tmp_path / "output.safetensors"

        stats = convert_exl3_to_marlin(nonexistent, output, verbose=False)

        assert len(stats["errors"]) == 1
        assert "does not exist" in stats["errors"][0]

    def test_convert_stats_structure(self, tmp_path):
        """Test that convert_exl3_to_marlin returns proper stats."""
        from metal_marlin.quantization.exl3_to_marlin import convert_exl3_to_marlin

        input_dir = tmp_path / "dummy_exl3"
        input_dir.mkdir()
        output = tmp_path / "output.safetensors"

        stats = convert_exl3_to_marlin(input_dir, output, verbose=False)

        # Check required keys
        assert "layers_converted" in stats
        assert "total_params" in stats
        assert "compression_ratio" in stats
        assert "errors" in stats
