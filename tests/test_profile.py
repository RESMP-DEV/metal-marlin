"""Tests for metal_marlin.profile module."""

from __future__ import annotations

import pytest

from metal_marlin.profile import (
    LayerFLOPs,
    LayerFLOPsCalculator,
    LayerFLOPsCounter,
    TransformerLayerFLOPs,
    calculate_attention_flops,
    calculate_embedding_flops,
    calculate_ffn_flops,
    calculate_layernorm_flops,
    calculate_layer_flops,
    calculate_matmul_flops,
    estimate_marlin_linear_flops,
    profile_model_flops,
    profile_model_layers,
)


class TestMatmulFLOPs:
    """Test matmul FLOPs calculations."""

    def test_standard_gemm(self) -> None:
        """Test standard GEMM FLOPs calculation."""
        # C = A @ B where A is [M, K] and B is [K, N]
        # FLOPs = 2 * M * N * K
        flops = calculate_matmul_flops(M=1024, N=1024, K=1024)
        expected = 2 * 1024 * 1024 * 1024  # 2,147,483,648
        assert flops == expected

    def test_quantized_gemm(self) -> None:
        """Test quantized GEMM includes dequant overhead."""
        flops = calculate_matmul_flops(M=1024, N=1024, K=1024, quantized=True)
        # Standard GEMM + dequant overhead (2 * M * N * K)
        expected_base = 2 * 1024 * 1024 * 1024
        expected_dequant = 1024 * 1024 * 1024 * 2
        assert flops == expected_base + expected_dequant

    def test_matmul_tflops_property(self) -> None:
        """Test TFLOPs property conversion."""
        flops = calculate_matmul_flops(M=4096, N=4096, K=4096)
        layer = LayerFLOPs(name="test", total_flops=flops, matmul_flops=flops)
        expected_tflops = flops / 1e12
        assert abs(layer.tflops - expected_tflops) < 1e-10


class TestAttentionFLOPs:
    """Test attention FLOPs calculations."""

    def test_attention_basic(self) -> None:
        """Test basic attention FLOPs."""
        flops = calculate_attention_flops(
            batch=2, seq_len=128, num_heads=8, head_dim=64
        )
        # Should be positive
        assert flops > 0

    def test_attention_causal(self) -> None:
        """Test causal attention reduces FLOPs."""
        full_flops = calculate_attention_flops(
            batch=2, seq_len=128, num_heads=8, head_dim=64, causal=False
        )
        causal_flops = calculate_attention_flops(
            batch=2, seq_len=128, num_heads=8, head_dim=64, causal=True
        )
        # Causal should be roughly 50% of full
        assert causal_flops < full_flops
        assert causal_flops == int(full_flops * 0.5)


class TestFFNFLOPs:
    """Test FFN FLOPs calculations."""

    def test_ffn_basic(self) -> None:
        """Test basic FFN FLOPs."""
        flops = calculate_ffn_flops(
            batch=2, seq_len=128, hidden_dim=512, ffn_dim=2048
        )
        assert flops > 0

    def test_ffn_gated(self) -> None:
        """Test gated FFN has more FLOPs."""
        standard_flops = calculate_ffn_flops(
            batch=2, seq_len=128, hidden_dim=512, ffn_dim=2048, gated=False
        )
        gated_flops = calculate_ffn_flops(
            batch=2, seq_len=128, hidden_dim=512, ffn_dim=2048, gated=True
        )
        assert gated_flops > standard_flops


class TestLayerNormFLOPs:
    """Test LayerNorm FLOPs calculations."""

    def test_layernorm(self) -> None:
        """Test LayerNorm FLOPs."""
        flops = calculate_layernorm_flops(batch=2, seq_len=128, hidden_dim=512)
        # ~5 ops per element
        expected = 2 * 128 * 512 * 5
        assert flops == expected


class TestEmbeddingFLOPs:
    """Test embedding FLOPs calculations."""

    def test_embedding(self) -> None:
        """Test embedding FLOPs."""
        flops = calculate_embedding_flops(
            batch=2, seq_len=128, vocab_size=32000, hidden_dim=512
        )
        # 1 op per element
        expected = 2 * 128 * 512
        assert flops == expected


class TestMarlinLinearFLOPs:
    """Test MarlinLinear-specific FLOPs calculations."""

    def test_marlin_linear_basic(self) -> None:
        """Test basic Marlin linear FLOPs."""
        flops = estimate_marlin_linear_flops(
            in_features=4096,
            out_features=11008,
            batch_size=1,
            seq_len=1,
            quantized=True,
        )
        assert flops > 0

    def test_marlin_linear_vs_standard(self) -> None:
        """Test quantized has overhead vs standard."""
        standard = estimate_marlin_linear_flops(
            in_features=4096, out_features=11008, batch_size=8, seq_len=128, quantized=False
        )
        quantized = estimate_marlin_linear_flops(
            in_features=4096, out_features=11008, batch_size=8, seq_len=128, quantized=True
        )
        assert quantized > standard

    def test_marlin_linear_with_bias(self) -> None:
        """Test bias adds FLOPs."""
        without_bias = estimate_marlin_linear_flops(
            in_features=4096,
            out_features=11008,
            batch_size=8,
            seq_len=128,
            include_bias=False,
        )
        with_bias = estimate_marlin_linear_flops(
            in_features=4096,
            out_features=11008,
            batch_size=8,
            seq_len=128,
            include_bias=True,
        )
        assert with_bias > without_bias


class TestCalculateLayerFLOPs:
    """Test generic layer FLOPs calculation."""

    def test_linear_layer(self) -> None:
        """Test linear layer FLOPs."""
        flops = calculate_layer_flops(
            "linear",
            in_features=4096,
            out_features=11008,
            batch_size=8,
            seq_len=128,
        )
        assert flops > 0

    def test_attention_layer(self) -> None:
        """Test attention layer FLOPs."""
        flops = calculate_layer_flops(
            "attention",
            batch=8,
            seq_len=128,
            num_heads=32,
            head_dim=128,
            causal=True,
        )
        assert flops > 0

    def test_ffn_layer(self) -> None:
        """Test FFN layer FLOPs."""
        flops = calculate_layer_flops(
            "ffn",
            batch=8,
            seq_len=128,
            hidden_dim=4096,
            ffn_dim=11008,
            gated=True,
        )
        assert flops > 0

    def test_layernorm_layer(self) -> None:
        """Test layernorm layer FLOPs."""
        flops = calculate_layer_flops(
            "layernorm",
            batch=8,
            seq_len=128,
            hidden_dim=4096,
        )
        assert flops > 0

    def test_invalid_layer_type(self) -> None:
        """Test invalid layer type raises error."""
        with pytest.raises(ValueError, match="Unknown layer_type"):
            calculate_layer_flops("invalid_layer", foo=123)


class TestLayerFLOPsCounter:
    """Test LayerFLOPsCounter functionality."""

    def test_empty_counter(self) -> None:
        """Test empty counter has zero FLOPs."""
        counter = LayerFLOPsCounter()
        assert counter.total_flops == 0
        assert counter.total_tflops == 0.0

    def test_add_matmul(self) -> None:
        """Test adding matmul layer."""
        counter = LayerFLOPsCounter()
        counter.add_matmul("test_matmul", M=1024, N=1024, K=1024)
        assert counter.total_flops > 0
        layer = counter.get_layer("test_matmul")
        assert layer is not None
        assert layer.name == "test_matmul"

    def test_add_attention(self) -> None:
        """Test adding attention layer."""
        counter = LayerFLOPsCounter()
        counter.add_attention("test_attn", batch=2, seq_len=128, num_heads=8, head_dim=64)
        assert counter.total_flops > 0

    def test_add_ffn(self) -> None:
        """Test adding FFN layer."""
        counter = LayerFLOPsCounter()
        counter.add_ffn("test_ffn", batch=2, seq_len=128, hidden_dim=512, ffn_dim=2048)
        assert counter.total_flops > 0

    def test_add_transformer_layer(self) -> None:
        """Test adding full transformer layer."""
        counter = LayerFLOPsCounter()
        counter.add_transformer_layer(
            "layer_0",
            batch=2,
            seq_len=128,
            hidden_dim=512,
            num_heads=8,
            ffn_dim=2048,
        )
        assert counter.total_flops > 0

    def test_clear(self) -> None:
        """Test clearing counter."""
        counter = LayerFLOPsCounter()
        counter.add_matmul("test", M=1024, N=1024, K=1024)
        assert counter.total_flops > 0
        counter.clear()
        assert counter.total_flops == 0


class TestLayerFLOPsCalculator:
    """Test LayerFLOPsCalculator functionality."""

    def test_calculator_init(self) -> None:
        """Test calculator initialization."""
        calc = LayerFLOPsCalculator(batch_size=8, seq_len=2048)
        assert calc.config.batch_size == 8
        assert calc.config.seq_len == 2048

    def test_empty_calculator(self) -> None:
        """Test empty calculator."""
        calc = LayerFLOPsCalculator()
        assert calc.total_flops == 0
        assert calc.total_params == 0
        assert len(calc.get_results()) == 0

    def test_get_layer_not_found(self) -> None:
        """Test getting non-existent layer returns None."""
        calc = LayerFLOPsCalculator()
        assert calc.get_layer("nonexistent") is None

    def test_clear_calculator(self) -> None:
        """Test clearing calculator."""
        calc = LayerFLOPsCalculator()
        # Would need to add a module first
        calc.clear()
        assert calc.total_flops == 0


class TestTransformerLayerFLOPs:
    """Test TransformerLayerFLOPs dataclass."""

    def test_from_config(self) -> None:
        """Test creating from config."""
        tf_flops = TransformerLayerFLOPs.from_config(
            batch=2,
            seq_len=128,
            hidden_dim=512,
            num_heads=8,
            ffn_dim=2048,
        )
        assert tf_flops.attention > 0
        assert tf_flops.ffn > 0
        assert tf_flops.layernorm > 0
        assert tf_flops.total > 0
        assert tf_flops.total == tf_flops.attention + tf_flops.ffn + tf_flops.layernorm


class TestProfileModelFLOPs:
    """Test high-level model profiling."""

    def test_profile_model(self) -> None:
        """Test profiling a model configuration."""
        counter = profile_model_flops(
            batch=1,
            seq_len=512,
            num_layers=2,
            hidden_dim=512,
            num_heads=8,
            ffn_dim=2048,
            vocab_size=1000,
        )
        assert counter.total_flops > 0
        assert len(counter.get_layers()) > 0


class TestModuleImports:
    """Test all module exports are available."""

    def test_all_exports(self) -> None:
        """Test that all expected exports exist."""
        from metal_marlin.profile import (
            LayerFLOPs,
            LayerFLOPsCounter,
            LayerFLOPsCalculator,
            TransformerLayerFLOPs,
            calculate_attention_flops,
            calculate_embedding_flops,
            calculate_ffn_flops,
            calculate_layernorm_flops,
            calculate_layer_flops,
            calculate_matmul_flops,
            estimate_marlin_linear_flops,
            profile_model_flops,
            profile_model_layers,
        )

        # Just verify they exist and are callable/types
        assert callable(calculate_matmul_flops)
        assert callable(calculate_attention_flops)
        assert callable(calculate_ffn_flops)
        assert callable(calculate_layernorm_flops)
        assert callable(calculate_embedding_flops)
        assert callable(calculate_layer_flops)
        assert callable(estimate_marlin_linear_flops)
        assert callable(profile_model_flops)
        assert callable(profile_model_layers)
