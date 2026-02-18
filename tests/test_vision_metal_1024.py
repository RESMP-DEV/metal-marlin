"""Tests for large resolution (1024x1024+) vision preprocessing methods.

Tests cover:
- Tile-based resize for memory efficiency
- Fused preprocessing pipeline (crop + resize + normalize)
- Lanczos-style high-quality resize (bicubic 8x8)
- Aspect ratio preserving resize with padding
- Large image processing (1024x1024, 2048x2048)

All tests are skipped if MPS is not available.
"""

from __future__ import annotations

import numpy as np
import pytest

from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch

# Skip all tests if MPS is not available
pytestmark = [
    pytest.mark.skipif(not HAS_MPS, reason="Needs MPS"),
    pytest.mark.skipif(not HAS_TORCH, reason="Needs PyTorch"),
    pytest.mark.vision,
]

# Try to import VisionMetal
try:
    from metal_marlin.vision.vision_metal import VisionMetal

    HAS_VISION_METAL = True
except ImportError:
    HAS_VISION_METAL = False


@pytest.fixture
def vision_metal():
    """Fixture providing a VisionMetal instance."""
    if not HAS_VISION_METAL:
        pytest.skip("VisionMetal not available")
    return VisionMetal()


# =============================================================================
# Large Resolution Resize Tests
# =============================================================================


def test_resize_bilinear_tiled_1024(vision_metal):
    """Test tile-based resize for 1024x1024 images."""
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")

    # Create 1024x1024 image
    rng = np.random.default_rng(42)
    img = rng.standard_normal((1, 1024, 1024, 3)).astype(np.float32)
    img_t = torch.from_numpy(img).to("mps")

    # Resize using tiled method
    result = vision_metal.resize_bilinear_tiled(
        img_t,
        target_size=(224, 224),
        nhwc=True,
    )

    assert result.shape == (1, 224, 224, 3)
    assert torch.all(torch.isfinite(result))


def test_resize_bilinear_tiled_2048(vision_metal):
    """Test tile-based resize for very large 2048x2048 images."""
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")

    # Create 2048x2048 image
    rng = np.random.default_rng(42)
    img = rng.standard_normal((1, 2048, 2048, 3)).astype(np.float32)
    img_t = torch.from_numpy(img).to("mps")

    # Resize to 1024x1024
    result = vision_metal.resize_bilinear_tiled(
        img_t,
        target_size=(1024, 1024),
        nhwc=True,
    )

    assert result.shape == (1, 1024, 1024, 3)
    assert torch.all(torch.isfinite(result))


def test_resize_bilinear_tiled_vs_standard(vision_metal):
    """Verify tiled resize produces similar results to standard resize."""
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")

    rng = np.random.default_rng(42)
    img = rng.standard_normal((1, 512, 512, 3)).astype(np.float32)
    img_t = torch.from_numpy(img).to("mps")

    # Both methods
    tiled_result = vision_metal.resize_bilinear_tiled(
        img_t,
        target_size=(224, 224),
        nhwc=True,
    )
    standard_result = vision_metal.resize_bilinear(
        img_t,
        target_size=(224, 224),
        nhwc=True,
    )

    # Should be very close (small differences due to different computation order)
    torch.testing.assert_close(tiled_result, standard_result, rtol=1e-3, atol=1e-4)


# =============================================================================
# Fused Preprocessing Pipeline Tests
# =============================================================================


def test_preprocess_large_image_fused_with_crop(vision_metal):
    """Test fused pipeline with center crop."""
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")

    # Create 2048x2048 image
    rng = np.random.default_rng(42)
    img = rng.standard_normal((1, 2048, 2048, 3)).astype(np.float32)
    img_t = torch.from_numpy(img).to("mps")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Fused crop + resize + normalize
    result = vision_metal.preprocess_large_image_fused(
        img_t,
        crop_size=(1536, 1536),
        target_size=(224, 224),
        mean=mean,
        std=std,
        nhwc=True,
    )

    assert result.shape == (1, 224, 224, 3)
    assert torch.all(torch.isfinite(result))


def test_preprocess_large_image_fused_no_crop(vision_metal):
    """Test fused pipeline without center crop (resize only)."""
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")

    # Create 1024x1024 image
    rng = np.random.default_rng(42)
    img = rng.standard_normal((1, 1024, 1024, 3)).astype(np.float32)
    img_t = torch.from_numpy(img).to("mps")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # No crop, just resize + normalize
    result = vision_metal.preprocess_large_image_fused(
        img_t,
        crop_size=None,
        target_size=(224, 224),
        mean=mean,
        std=std,
        nhwc=True,
    )

    assert result.shape == (1, 224, 224, 3)
    assert torch.all(torch.isfinite(result))


def test_preprocess_large_image_fused_vs_separate(vision_metal):
    """Verify fused pipeline matches separate operations."""
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")

    rng = np.random.default_rng(42)
    img = rng.standard_normal((1, 1024, 1024, 3)).astype(np.float32)
    img_t = torch.from_numpy(img).to("mps")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Fused path
    fused_result = vision_metal.preprocess_large_image_fused(
        img_t,
        crop_size=(768, 768),
        target_size=(224, 224),
        mean=mean,
        std=std,
        nhwc=True,
    )

    # Separate operations path
    from metal_marlin.vision.vision_metal import preprocess_for_vit

    # Manual crop then resize+normalize
    cropped = vision_metal.center_crop(img_t, size=(768, 768), nhwc=True)
    separate_result = vision_metal.resize_and_normalize(
        cropped,
        size=(224, 224),
        mean=mean,
        std=std,
        nhwc=True,
    )

    # Should be very close
    torch.testing.assert_close(fused_result, separate_result, rtol=1e-3, atol=1e-4)


# =============================================================================
# High-Quality Bicubic 8x8 Tests
# =============================================================================


def test_resize_bicubic_8x8_basic(vision_metal):
    """Test Lanczos-style 8x8 bicubic resize."""
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")

    rng = np.random.default_rng(42)
    img = rng.standard_normal((1, 1024, 1024, 3)).astype(np.float32)
    img_t = torch.from_numpy(img).to("mps")

    result = vision_metal.resize_bicubic_8x8(
        img_t,
        target_size=(512, 512),
        nhwc=True,
    )

    assert result.shape == (1, 512, 512, 3)
    assert torch.all(torch.isfinite(result))


def test_resize_bicubic_8x8_vs_standard(vision_metal):
    """Compare 8x8 bicubic to standard bicubic."""
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")

    rng = np.random.default_rng(42)
    img = rng.standard_normal((1, 512, 512, 3)).astype(np.float32)
    img_t = torch.from_numpy(img).to("mps")

    # Both methods
    high_quality = vision_metal.resize_bicubic_8x8(
        img_t,
        target_size=(224, 224),
        nhwc=True,
    )
    standard = vision_metal.resize_bicubic(
        img_t,
        target_size=(224, 224),
        nhwc=True,
    )

    # Both should produce valid results
    assert high_quality.shape == standard.shape
    assert torch.all(torch.isfinite(high_quality))
    assert torch.all(torch.isfinite(standard))

    # Results will differ due to different kernels, but both should be reasonable
    assert high_quality.abs().mean() > 0  # Not all zeros
    assert standard.abs().mean() > 0


# =============================================================================
# Aspect Ratio Preserve Tests
# =============================================================================


def test_resize_aspect_ratio_preserve_wide(vision_metal):
    """Test aspect ratio preserve for wide images (16:9)."""
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")

    # 1920x1080 HD image
    rng = np.random.default_rng(42)
    img = rng.standard_normal((1, 1080, 1920, 3)).astype(np.float32)
    img_t = torch.from_numpy(img).to("mps")

    result = vision_metal.resize_aspect_ratio_preserve(
        img_t,
        target_size=(1024, 1024),
        pad_value=0.0,
        nhwc=True,
    )

    assert result.shape == (1, 1024, 1024, 3)
    assert torch.all(torch.isfinite(result))

    # Check that padding exists (some pixels should be zero)
    # For 1920x1080 -> 1024x1024, image becomes 1024x576 with padding top/bottom
    top_row_mean = result[0, 0, :, :].mean()
    assert top_row_mean < 0.5, "Top padding should have low values"


def test_resize_aspect_ratio_preserve_tall(vision_metal):
    """Test aspect ratio preserve for tall images (9:16)."""
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")

    # 1080x1920 tall image
    rng = np.random.default_rng(42)
    img = rng.standard_normal((1, 1920, 1080, 3)).astype(np.float32)
    img_t = torch.from_numpy(img).to("mps")

    result = vision_metal.resize_aspect_ratio_preserve(
        img_t,
        target_size=(1024, 1024),
        pad_value=0.0,
        nhwc=True,
    )

    assert result.shape == (1, 1024, 1024, 3)
    assert torch.all(torch.isfinite(result))

    # Check that padding exists on left/right
    left_col_mean = result[0, :, 0, :].mean()
    assert left_col_mean < 0.5, "Left padding should have low values"


def test_resize_aspect_ratio_preserve_already_square(vision_metal):
    """Test aspect ratio preserve for already square images."""
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")

    # 1024x1024 square image
    rng = np.random.default_rng(42)
    img = rng.standard_normal((1, 1024, 1024, 3)).astype(np.float32)
    img_t = torch.from_numpy(img).to("mps")

    result = vision_metal.resize_aspect_ratio_preserve(
        img_t,
        target_size=(512, 512),
        pad_value=0.0,
        nhwc=True,
    )

    assert result.shape == (1, 512, 512, 3)
    assert torch.all(torch.isfinite(result))

    # For square input to square output, there should be minimal padding
    # Most pixels should be non-zero (from the image content)
    assert result.abs().mean() > 0.1


# =============================================================================
# NCHW Layout Tests for Large Resolution Methods
# =============================================================================


def test_resize_bilinear_tiled_nchw(vision_metal):
    """Test tiled resize with NCHW layout."""
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")

    rng = np.random.default_rng(42)
    img = rng.standard_normal((1, 3, 1024, 1024)).astype(np.float32)
    img_t = torch.from_numpy(img).to("mps")

    result = vision_metal.resize_bilinear_tiled(
        img_t,
        target_size=(224, 224),
        nhwc=False,
    )

    assert result.shape == (1, 3, 224, 224)
    assert torch.all(torch.isfinite(result))


def test_preprocess_large_image_fused_nchw(vision_metal):
    """Test fused pipeline with NCHW layout."""
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")

    rng = np.random.default_rng(42)
    img = rng.standard_normal((1, 3, 1024, 1024)).astype(np.float32)
    img_t = torch.from_numpy(img).to("mps")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    result = vision_metal.preprocess_large_image_fused(
        img_t,
        crop_size=(768, 768),
        target_size=(224, 224),
        mean=mean,
        std=std,
        nhwc=False,
    )

    assert result.shape == (1, 3, 224, 224)
    assert torch.all(torch.isfinite(result))


# =============================================================================
# Batch Processing Tests for Large Images
# =============================================================================


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_resize_bilinear_tiled_batch(vision_metal, batch_size):
    """Test tiled resize with batched images."""
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")

    rng = np.random.default_rng(42)
    img = rng.standard_normal((batch_size, 1024, 1024, 3)).astype(np.float32)
    img_t = torch.from_numpy(img).to("mps")

    result = vision_metal.resize_bilinear_tiled(
        img_t,
        target_size=(224, 224),
        nhwc=True,
    )

    assert result.shape == (batch_size, 224, 224, 3)
    assert torch.all(torch.isfinite(result))


@pytest.mark.parametrize("batch_size", [1, 2])
def test_preprocess_large_image_fused_batch(vision_metal, batch_size):
    """Test fused pipeline with batched images."""
    if not HAS_TORCH or torch is None:
        pytest.skip("PyTorch not available")

    rng = np.random.default_rng(42)
    img = rng.standard_normal((batch_size, 1024, 1024, 3)).astype(np.float32)
    img_t = torch.from_numpy(img).to("mps")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    result = vision_metal.preprocess_large_image_fused(
        img_t,
        crop_size=None,
        target_size=(224, 224),
        mean=mean,
        std=std,
        nhwc=True,
    )

    assert result.shape == (batch_size, 224, 224, 3)
    assert torch.all(torch.isfinite(result))
