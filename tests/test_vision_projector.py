"""Tests for vision-language projector modules.

Tests cover:
- All projector architectures (LLaVA, Qwen2-VL, InternVL)
- Config parsing from HuggingFace configs
- Auto-detection of projector types
- Single image, multi-image, and video frame inputs
- Numpy fallback when MLX unavailable
"""

from __future__ import annotations

import numpy as np
import pytest

from metal_marlin._compat import HAS_MLX, mx
from metal_marlin.vision import (
    InternVLProjector,
    LLaVAProjector,
    Qwen2VLProjector,
    VisionProjector,
    VisionProjectorConfig,
    detect_projector_type,
)
from metal_marlin.vision.projector import (
    IdentityProjector,
    LinearProjector,
    ProjectorType,
)

# =============================================================================
# Config Tests
# =============================================================================


class TestVisionProjectorConfig:
    """Tests for VisionProjectorConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = VisionProjectorConfig()

        assert config.projector_type == ProjectorType.LLAVA_MLP
        assert config.vision_hidden_size == 1024
        assert config.llm_hidden_size == 4096
        assert config.intermediate_size == 4096  # 4x vision_hidden_size
        assert config.num_query_tokens == 64
        assert config.activation == "gelu"
        assert config.precision == "fp16"

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = VisionProjectorConfig(
            projector_type=ProjectorType.PERCEIVER,
            vision_hidden_size=1408,
            llm_hidden_size=5120,
            intermediate_size=8192,
            num_query_tokens=128,
            activation="silu",
        )

        assert config.projector_type == ProjectorType.PERCEIVER
        assert config.vision_hidden_size == 1408
        assert config.llm_hidden_size == 5120
        assert config.intermediate_size == 8192
        assert config.num_query_tokens == 128
        assert config.activation == "silu"

    def test_intermediate_size_default(self) -> None:
        """Test intermediate_size defaults to 4x vision_hidden_size."""
        config = VisionProjectorConfig(vision_hidden_size=512)
        assert config.intermediate_size == 2048

    def test_from_llava_config(self) -> None:
        """Test parsing LLaVA-style HuggingFace config."""
        hf_config = {
            "model_type": "llava",
            "vision_config": {"hidden_size": 1024},
            "hidden_size": 4096,
            "projector_type": "mlp2x_gelu",
        }

        config = VisionProjectorConfig.from_hf_config(hf_config)

        assert config.projector_type == ProjectorType.LLAVA_MLP
        assert config.vision_hidden_size == 1024
        assert config.llm_hidden_size == 4096

    def test_from_qwen2vl_config(self) -> None:
        """Test parsing Qwen2-VL config."""
        hf_config = {
            "model_type": "qwen2_vl",
            "vision_config": {"hidden_size": 1280},
            "hidden_size": 3584,
            "num_query_tokens": 256,
            "resampler_num_heads": 16,
            "resampler_num_layers": 4,
        }

        config = VisionProjectorConfig.from_hf_config(hf_config)

        assert config.projector_type == ProjectorType.PERCEIVER
        assert config.vision_hidden_size == 1280
        assert config.llm_hidden_size == 3584
        assert config.num_query_tokens == 256
        assert config.num_attention_heads == 16
        assert config.num_resampler_layers == 4

    def test_from_internvl_config(self) -> None:
        """Test parsing InternVL config."""
        hf_config = {
            "model_type": "internvl",
            "visual_hidden_size": 1408,
            "text_config": {"hidden_size": 5120},
            "use_qllama_projector": True,
            "num_image_tokens": 256,
        }

        config = VisionProjectorConfig.from_hf_config(hf_config)

        assert config.projector_type == ProjectorType.QLLAMA
        assert config.vision_hidden_size == 1408
        assert config.llm_hidden_size == 5120
        assert config.num_query_tokens == 256


# =============================================================================
# Projector Type Detection Tests
# =============================================================================


class TestProjectorTypeDetection:
    """Tests for auto-detection of projector type."""

    def test_detect_llava(self) -> None:
        """Test detection of LLaVA projector."""
        config = {"model_type": "llava"}
        assert detect_projector_type(config) == ProjectorType.LLAVA_MLP

    def test_detect_qwen2vl(self) -> None:
        """Test detection of Qwen2-VL Perceiver."""
        config = {"model_type": "qwen2_vl"}
        assert detect_projector_type(config) == ProjectorType.PERCEIVER

        config = {"model_type": "qwen2-vl"}
        assert detect_projector_type(config) == ProjectorType.PERCEIVER

    def test_detect_internvl(self) -> None:
        """Test detection of InternVL QLLaMA projector."""
        config = {"model_type": "internvl"}
        assert detect_projector_type(config) == ProjectorType.QLLAMA

    def test_detect_explicit_type(self) -> None:
        """Test explicit projector_type in config."""
        config = {"projector_type": "linear"}
        assert detect_projector_type(config) == ProjectorType.LINEAR

        config = {"mm_projector_type": "identity"}
        assert detect_projector_type(config) == ProjectorType.IDENTITY

    def test_detect_default(self) -> None:
        """Test default fallback to LLaVA MLP."""
        config = {"model_type": "unknown_model"}
        assert detect_projector_type(config) == ProjectorType.LLAVA_MLP


# =============================================================================
# LLaVA Projector Tests
# =============================================================================


class TestLLaVAProjector:
    """Tests for LLaVA 2-layer MLP projector."""

    @pytest.fixture
    def config(self) -> VisionProjectorConfig:
        return VisionProjectorConfig(
            projector_type=ProjectorType.LLAVA_MLP,
            vision_hidden_size=1024,
            llm_hidden_size=4096,
            intermediate_size=4096,
        )

    def test_init(self, config: VisionProjectorConfig) -> None:
        """Test projector initialization."""
        projector = LLaVAProjector(config)

        assert projector.vision_hidden_size == 1024
        assert projector.llm_hidden_size == 4096
        assert projector.intermediate_size == 4096

    def test_forward_numpy(self, config: VisionProjectorConfig) -> None:
        """Test forward pass with numpy input."""
        projector = LLaVAProjector(config)

        # Single image: [batch, num_patches, vision_hidden]
        x = np.random.randn(1, 576, 1024).astype(np.float32)
        output = projector(x)

        assert output.shape == (1, 576, 4096)

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_forward_mlx(self, config: VisionProjectorConfig) -> None:
        """Test forward pass with MLX input."""
        projector = LLaVAProjector(config)

        x = mx.random.normal((1, 576, 1024))
        output = projector(x)

        assert output.shape == (1, 576, 4096)

    def test_forward_batched(self, config: VisionProjectorConfig) -> None:
        """Test batched forward pass."""
        projector = LLaVAProjector(config)

        # Batch of 4 images
        x = np.random.randn(4, 576, 1024).astype(np.float32)
        output = projector(x)

        assert output.shape == (4, 576, 4096)

    def test_forward_variable_patches(self, config: VisionProjectorConfig) -> None:
        """Test with variable number of patches."""
        projector = LLaVAProjector(config)

        for num_patches in [256, 576, 1024]:
            x = np.random.randn(1, num_patches, 1024).astype(np.float32)
            output = projector(x)
            assert output.shape == (1, num_patches, 4096)


# =============================================================================
# Qwen2-VL Perceiver Tests
# =============================================================================


class TestQwen2VLProjector:
    """Tests for Qwen2-VL Perceiver resampler."""

    @pytest.fixture
    def config(self) -> VisionProjectorConfig:
        return VisionProjectorConfig(
            projector_type=ProjectorType.PERCEIVER,
            vision_hidden_size=1280,
            llm_hidden_size=3584,
            num_query_tokens=64,
            num_attention_heads=16,
            num_resampler_layers=2,
        )

    def test_init(self, config: VisionProjectorConfig) -> None:
        """Test projector initialization."""
        projector = Qwen2VLProjector(config)

        assert projector.vision_hidden_size == 1280
        assert projector.llm_hidden_size == 3584
        assert projector.num_query_tokens == 64
        assert projector.num_heads == 16
        assert projector.num_layers == 2

    def test_forward_numpy(self, config: VisionProjectorConfig) -> None:
        """Test forward pass with numpy (simplified fallback)."""
        projector = Qwen2VLProjector(config)

        # Variable length input gets resampled to fixed query tokens
        x = np.random.randn(1, 576, 1280).astype(np.float32)
        output = projector(x)

        # Output should be [batch, num_query_tokens, llm_hidden]
        assert output.shape == (1, 64, 3584)

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_forward_mlx(self, config: VisionProjectorConfig) -> None:
        """Test forward pass with MLX (full cross-attention)."""
        projector = Qwen2VLProjector(config)

        x = mx.random.normal((1, 576, 1280))
        output = projector(x)

        assert output.shape == (1, 64, 3584)

    def test_forward_multi_image(self, config: VisionProjectorConfig) -> None:
        """Test with multiple images (concatenated patches)."""
        projector = Qwen2VLProjector(config)

        # 3 images * 576 patches each = 1728 total patches
        x = np.random.randn(1, 1728, 1280).astype(np.float32)
        output = projector(x)

        # Still resampled to fixed query tokens
        assert output.shape == (1, 64, 3584)


# =============================================================================
# InternVL Projector Tests
# =============================================================================


class TestInternVLProjector:
    """Tests for InternVL QLLaMA-style projector."""

    @pytest.fixture
    def config(self) -> VisionProjectorConfig:
        return VisionProjectorConfig(
            projector_type=ProjectorType.QLLAMA,
            vision_hidden_size=1408,
            llm_hidden_size=5120,
            num_query_tokens=256,
            num_attention_heads=20,
            num_resampler_layers=3,
            max_image_tokens=1024,
        )

    def test_init(self, config: VisionProjectorConfig) -> None:
        """Test projector initialization."""
        projector = InternVLProjector(config)

        assert projector.vision_hidden_size == 1408
        assert projector.llm_hidden_size == 5120
        assert projector.num_query_tokens == 256
        assert projector.max_patches == 1024

    def test_forward_numpy(self, config: VisionProjectorConfig) -> None:
        """Test forward pass with numpy."""
        projector = InternVLProjector(config)

        x = np.random.randn(1, 576, 1408).astype(np.float32)
        output = projector(x)

        assert output.shape == (1, 256, 5120)

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_forward_mlx(self, config: VisionProjectorConfig) -> None:
        """Test forward pass with MLX."""
        projector = InternVLProjector(config)

        x = mx.random.normal((1, 576, 1408))
        output = projector(x)

        assert output.shape == (1, 256, 5120)

    def test_forward_high_resolution(self, config: VisionProjectorConfig) -> None:
        """Test with high-resolution image (more patches)."""
        projector = InternVLProjector(config)

        # High-res: 32x32 = 1024 patches
        x = np.random.randn(1, 1024, 1408).astype(np.float32)
        output = projector(x)

        # Resampled to query tokens
        assert output.shape == (1, 256, 5120)


# =============================================================================
# Linear and Identity Projector Tests
# =============================================================================


class TestLinearProjector:
    """Tests for simple linear projector."""

    def test_forward(self) -> None:
        """Test linear projection."""
        config = VisionProjectorConfig(
            projector_type=ProjectorType.LINEAR,
            vision_hidden_size=768,
            llm_hidden_size=4096,
        )
        projector = LinearProjector(config)

        x = np.random.randn(1, 196, 768).astype(np.float32)
        output = projector(x)

        assert output.shape == (1, 196, 4096)


class TestIdentityProjector:
    """Tests for identity projector."""

    def test_matching_dims(self) -> None:
        """Test identity with matching dimensions."""
        config = VisionProjectorConfig(
            projector_type=ProjectorType.IDENTITY,
            vision_hidden_size=4096,
            llm_hidden_size=4096,
        )
        projector = IdentityProjector(config)

        x = np.random.randn(1, 576, 4096).astype(np.float32)
        output = projector(x)

        np.testing.assert_array_equal(output, x)

    def test_mismatched_dims_raises(self) -> None:
        """Test identity raises error for mismatched dimensions."""
        config = VisionProjectorConfig(
            projector_type=ProjectorType.IDENTITY,
            vision_hidden_size=1024,
            llm_hidden_size=4096,
        )

        with pytest.raises(ValueError, match="IdentityProjector requires matching dimensions"):
            IdentityProjector(config)


# =============================================================================
# Factory Tests
# =============================================================================


class TestVisionProjectorFactory:
    """Tests for VisionProjector factory class."""

    def test_from_config_llava(self) -> None:
        """Test factory creates LLaVA projector."""
        config = VisionProjectorConfig(projector_type=ProjectorType.LLAVA_MLP)
        projector = VisionProjector.from_config(config)

        assert isinstance(projector, LLaVAProjector)

    def test_from_config_perceiver(self) -> None:
        """Test factory creates Perceiver projector."""
        config = VisionProjectorConfig(projector_type=ProjectorType.PERCEIVER)
        projector = VisionProjector.from_config(config)

        assert isinstance(projector, Qwen2VLProjector)

    def test_from_config_qllama(self) -> None:
        """Test factory creates QLLaMA projector."""
        config = VisionProjectorConfig(projector_type=ProjectorType.QLLAMA)
        projector = VisionProjector.from_config(config)

        assert isinstance(projector, InternVLProjector)

    def test_from_hf_config(self) -> None:
        """Test factory from HuggingFace config."""
        hf_config = {
            "model_type": "llava",
            "vision_config": {"hidden_size": 1024},
            "hidden_size": 4096,
        }

        projector = VisionProjector.from_hf_config(hf_config)
        assert isinstance(projector, LLaVAProjector)


# =============================================================================
# Multi-Image and Video Tests
# =============================================================================


class TestMultiImageInput:
    """Tests for multi-image input handling."""

    def test_interleaved_images(self) -> None:
        """Test projector with interleaved image tokens."""
        config = VisionProjectorConfig(
            projector_type=ProjectorType.LLAVA_MLP,
            vision_hidden_size=1024,
            llm_hidden_size=4096,
        )
        projector = LLaVAProjector(config)

        # 3 images, each with 576 patches
        batch_size = 1
        num_images = 3
        patches_per_image = 576
        total_patches = num_images * patches_per_image

        x = np.random.randn(batch_size, total_patches, 1024).astype(np.float32)
        output = projector(x)

        assert output.shape == (batch_size, total_patches, 4096)

    def test_video_frames(self) -> None:
        """Test projector with video frame sequence."""
        config = VisionProjectorConfig(
            projector_type=ProjectorType.PERCEIVER,
            vision_hidden_size=1280,
            llm_hidden_size=3584,
            num_query_tokens=128,  # More tokens for video
            supports_video=True,
        )
        projector = Qwen2VLProjector(config)

        # 8 video frames, each with 196 patches (14x14)
        num_frames = 8
        patches_per_frame = 196
        total_patches = num_frames * patches_per_frame

        x = np.random.randn(1, total_patches, 1280).astype(np.float32)
        output = projector(x)

        # Resampled to fixed query tokens
        assert output.shape == (1, 128, 3584)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_patch(self) -> None:
        """Test with single patch input."""
        config = VisionProjectorConfig(
            projector_type=ProjectorType.LLAVA_MLP,
            vision_hidden_size=1024,
            llm_hidden_size=4096,
        )
        projector = LLaVAProjector(config)

        x = np.random.randn(1, 1, 1024).astype(np.float32)
        output = projector(x)

        assert output.shape == (1, 1, 4096)

    def test_large_batch(self) -> None:
        """Test with large batch size."""
        config = VisionProjectorConfig(
            projector_type=ProjectorType.LLAVA_MLP,
            vision_hidden_size=512,
            llm_hidden_size=2048,
        )
        projector = LLaVAProjector(config)

        x = np.random.randn(32, 256, 512).astype(np.float32)
        output = projector(x)

        assert output.shape == (32, 256, 2048)

    def test_2d_input(self) -> None:
        """Test with 2D input (no batch dimension)."""
        config = VisionProjectorConfig(
            projector_type=ProjectorType.LLAVA_MLP,
            vision_hidden_size=1024,
            llm_hidden_size=4096,
        )
        projector = LLaVAProjector(config)

        x = np.random.randn(576, 1024).astype(np.float32)
        output = projector(x)

        assert output.shape == (576, 4096)

    def test_different_activations(self) -> None:
        """Test with different activation functions."""
        for activation in ["gelu", "silu", "relu"]:
            config = VisionProjectorConfig(
                projector_type=ProjectorType.LLAVA_MLP,
                vision_hidden_size=256,
                llm_hidden_size=512,
                activation=activation,
            )
            projector = LLaVAProjector(config)

            x = np.random.randn(1, 16, 256).astype(np.float32)
            output = projector(x)

            assert output.shape == (1, 16, 512)
            # Output should be finite
            assert np.all(np.isfinite(output))
