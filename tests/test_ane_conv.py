"""Correctness tests for ANE conv implementations.

Tests ANEConv1d wrapper for numerical accuracy and functionality
compared to PyTorch equivalents.
"""

import pytest
import torch
import torch.nn as nn

try:
    from metal_marlin.ane import ANEConv1d, create_ane_conv1d, is_ane_available
    from metal_marlin.ane.depthwise_conv_ane import ANEDepthwiseConv1d, create_ane_depthwise_conv1d

    HAS_ANE = is_ane_available()
except ImportError:
    HAS_ANE = False


@pytest.mark.skipif(not HAS_ANE, reason="ANE not available")
class TestANEConv:
    """Tests for ANE-accelerated Conv1d implementation."""

    def test_pointwise_conv_matches(self):
        """Test ANEConv1d produces identical results to PyTorch Conv1d for pointwise conv."""
        # Create pointwise convolution (kernel_size=1)
        conv = nn.Conv1d(512, 512, 1, bias=True)
        conv.eval()  # Ensure deterministic behavior

        # Create ANE wrapper
        ane_conv = ANEConv1d(conv)

        # Test input
        x = torch.randn(1, 512, 1000)

        # Get expected and actual outputs
        with torch.no_grad():
            expected = conv(x)
            actual = ane_conv(x.to("mps")).cpu()

        # Verify numerical equivalence
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_depthwise_conv_matches(self):
        """Test ANEConv1d produces identical results for depthwise convolution."""
        # Create depthwise convolution
        conv = nn.Conv1d(256, 256, 3, padding=1, groups=256, bias=False)
        conv.eval()

        # Create ANE wrapper
        ane_conv = ANEConv1d(conv)

        # Test input
        x = torch.randn(2, 256, 512)

        # Get expected and actual outputs
        with torch.no_grad():
            expected = conv(x)
            actual = ane_conv(x.to("mps")).cpu()

        # Verify numerical equivalence
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_expanded_conv_matches(self):
        """Test ANEConv1d with expanded convolutions (channels != kernel_size)."""
        # Create expanded convolution
        conv = nn.Conv1d(128, 256, 3, padding=1, bias=True)
        conv.eval()

        # Create ANE wrapper
        ane_conv = ANEConv1d(conv)

        # Test input
        x = torch.randn(4, 128, 256)

        # Get expected and actual outputs
        with torch.no_grad():
            expected = conv(x)
            actual = ane_conv(x.to("mps")).cpu()

        # Verify numerical equivalence
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_strided_conv_matches(self):
        """Test ANEConv1d with strided convolutions."""
        # Create strided convolution
        conv = nn.Conv1d(64, 128, 3, stride=2, padding=1, bias=True)
        conv.eval()

        # Create ANE wrapper
        ane_conv = ANEConv1d(conv)

        # Test input
        x = torch.randn(2, 64, 1024)

        # Get expected and actual outputs
        with torch.no_grad():
            expected = conv(x)
            actual = ane_conv(x.to("mps")).cpu()

        # Verify numerical equivalence
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_different_batch_sizes(self):
        """Test ANEConv1d with different batch sizes."""
        conv = nn.Conv1d(128, 128, 1, bias=False)
        conv.eval()
        ane_conv = ANEConv1d(conv)

        # Test different batch sizes
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 128, 512)

            with torch.no_grad():
                expected = conv(x)
                actual = ane_conv(x.to("mps")).cpu()

            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_different_sequence_lengths(self):
        """Test ANEConv1d with different sequence lengths."""
        conv = nn.Conv1d(64, 64, 3, padding=1, bias=True)
        conv.eval()
        ane_conv = ANEConv1d(conv)

        # Test different sequence lengths
        for seq_len in [64, 128, 256, 512, 1024]:
            x = torch.randn(2, 64, seq_len)

            with torch.no_grad():
                expected = conv(x)
                actual = ane_conv(x.to("mps")).cpu()

            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


class TestANEDepthwiseConv:
    """Tests for ANE-optimized depthwise convolution."""

    def test_create_ane_depthwise_conv1d(self):
        """Test creation of ANE depthwise convolution."""
        depthwise_conv = create_ane_depthwise_conv1d(256, 3, padding=1)

        assert isinstance(depthwise_conv, ANEDepthwiseConv1d)
        assert depthwise_conv.in_channels == 256
        assert depthwise_conv.out_channels == 256
        assert depthwise_conv.kernel_size == 3
        assert depthwise_conv.padding == 1
        assert depthwise_conv.groups == 256

    def test_depthwise_conv_from_conv(self):
        """Test creating ANE depthwise conv from existing Conv1d."""
        # Create original depthwise conv
        conv = nn.Conv1d(128, 128, 5, padding=2, groups=128, bias=False)
        conv.eval()

        # Create ANE version
        ane_depthwise = ANEDepthwiseConv1d.from_conv(conv)

        assert isinstance(ane_depthwise, ANEDepthwiseConv1d)
        assert ane_depthwise.in_channels == conv.in_channels
        assert ane_depthwise.out_channels == conv.out_channels

        # Test functionality
        x = torch.randn(2, 128, 256)

        with torch.no_grad():
            expected = conv(x)
            actual = ane_depthwise(x)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_depthwise_conv_equivalence(self):
        """Test ANE depthwise conv produces same results as PyTorch."""
        # Test various configurations
        configs = [
            {"channels": 64, "kernel_size": 3, "padding": 1},
            {"channels": 128, "kernel_size": 5, "padding": 2},
            {"channels": 256, "kernel_size": 7, "padding": 3},
        ]

        for config in configs:
            # Create equivalent convolutions
            conv = nn.Conv1d(
                config["channels"],
                config["channels"],
                config["kernel_size"],
                padding=config["padding"],
                groups=config["channels"],
                bias=False,
            )
            conv.eval()

            ane_conv = ANEDepthwiseConv1d.from_conv(conv)

            # Test
            x = torch.randn(2, config["channels"], 256)

            with torch.no_grad():
                expected = conv(x)
                actual = ane_conv(x)

            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


class TestANEConvUtility:
    """Tests for ANE conv utility functions."""

    def test_is_ane_available(self):
        """Test ANE availability check."""
        result = is_ane_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(not HAS_ANE, reason="ANE not available")
    def test_create_ane_conv1d(self):
        """Test utility function for creating ANE conv."""
        conv = nn.Conv1d(64, 64, 3, padding=1)
        ane_conv = create_ane_conv1d(conv)

        assert isinstance(ane_conv, ANEConv1d)

        # Test functionality
        x = torch.randn(1, 64, 128)
        with torch.no_grad():
            expected = conv(x)
            actual = ane_conv(x.to("mps")).cpu()

        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_fallback_behavior(self):
        """Test fallback behavior when ANE is not available."""
        conv = nn.Conv1d(32, 32, 1)

        # Test with disabled ANE
        from metal_marlin.ane import maybe_ane_conv1d

        fallback_conv = maybe_ane_conv1d(conv, use_ane=False)

        # Should return original conv when ANE is disabled
        assert fallback_conv is conv

        # Test with enabled ANE but unavailable
        if not HAS_ANE:
            fallback_conv = maybe_ane_conv1d(conv, use_ane=True)
            # Should still return original conv when ANE is not available
            assert fallback_conv is conv


@pytest.mark.skipif(not HAS_ANE, reason="ANE not available")
class TestANEConvEdgeCases:
    """Tests for edge cases in ANE conv implementations."""

    def test_conv_with_bias(self):
        """Test ANE conv with bias terms."""
        conv = nn.Conv1d(128, 256, 3, padding=1, bias=True)
        ane_conv = ANEConv1d(conv)

        x = torch.randn(2, 128, 256)

        with torch.no_grad():
            expected = conv(x)
            actual = ane_conv(x.to("mps")).cpu()

        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_conv_without_bias(self):
        """Test ANE conv without bias terms."""
        conv = nn.Conv1d(128, 256, 3, padding=1, bias=False)
        ane_conv = ANEConv1d(conv)

        x = torch.randn(2, 128, 256)

        with torch.no_grad():
            expected = conv(x)
            actual = ane_conv(x.to("mps")).cpu()

        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_large_kernel_conv(self):
        """Test ANE conv with large kernel sizes."""
        conv = nn.Conv1d(64, 64, 31, padding=15, bias=True)
        ane_conv = ANEConv1d(conv)

        x = torch.randn(1, 64, 512)

        with torch.no_grad():
            expected = conv(x)
            actual = ane_conv(x.to("mps")).cpu()

        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_dilated_conv(self):
        """Test ANE conv with dilation."""
        conv = nn.Conv1d(128, 128, 3, padding=2, dilation=2, bias=True)
        ane_conv = ANEConv1d(conv)

        x = torch.randn(2, 128, 256)

        with torch.no_grad():
            expected = conv(x)
            actual = ane_conv(x.to("mps")).cpu()

        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_single_channel_conv(self):
        """Test ANE conv with single channel."""
        conv = nn.Conv1d(1, 1, 3, padding=1, bias=True)
        ane_conv = ANEConv1d(conv)

        x = torch.randn(4, 1, 100)

        with torch.no_grad():
            expected = conv(x)
            actual = ane_conv(x.to("mps")).cpu()

        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_mixed_precision(self):
        """Test ANE conv with different precisions."""
        conv = nn.Conv1d(64, 64, 3, padding=1, bias=True)
        ane_conv = ANEConv1d(conv)

        # Test float32
        x_fp32 = torch.randn(2, 64, 128, dtype=torch.float32)
        with torch.no_grad():
            expected_fp32 = conv(x_fp32)
            actual_fp32 = ane_conv(x_fp32.to("mps")).cpu()
        torch.testing.assert_close(actual_fp32, expected_fp32, rtol=1e-3, atol=1e-3)

        # Test float16
        x_fp16 = torch.randn(2, 64, 128, dtype=torch.float16)
        conv_fp16 = conv.to(torch.float16)
        ane_conv_fp16 = ANEConv1d(conv_fp16)

        with torch.no_grad():
            expected_fp16 = conv_fp16(x_fp16)
            actual_fp16 = ane_conv_fp16(x_fp16.to("mps")).cpu()
        torch.testing.assert_close(actual_fp16, expected_fp16, rtol=1e-2, atol=1e-2)


class TestConformerLikePattern:
    """Test patterns similar to those used in Conformer blocks."""

    def test_depthwise_pointwise_sequence(self):
        """Test depthwise followed by pointwise convolution pattern."""
        # Input channels
        in_channels = 256
        expansion = 4
        expanded_channels = in_channels * expansion

        # Create depthwise conv (like in Conformer)
        depthwise_conv = nn.Conv1d(
            expanded_channels, expanded_channels, 31, padding=15, groups=expanded_channels
        )

        # Create pointwise convs
        pointwise_expansion = nn.Conv1d(in_channels, expanded_channels, 1)
        pointwise_projection = nn.Conv1d(expanded_channels, in_channels, 1)

        # Set to eval mode
        depthwise_conv.eval()
        pointwise_expansion.eval()
        pointwise_projection.eval()

        # Test without ANE
        x = torch.randn(2, in_channels, 512)
        with torch.no_grad():
            # Standard forward
            expanded = pointwise_expansion(x)
            depthwise_out = depthwise_conv(expanded)
            final_out = pointwise_projection(depthwise_out)

        # Test with ANE depthwise
        if HAS_ANE:
            ane_depthwise = ANEDepthwiseConv1d.from_conv(depthwise_conv)

            with torch.no_grad():
                expanded_ane = pointwise_expansion(x)
                depthwise_out_ane = ane_depthwise(expanded_ane)
                final_out_ane = pointwise_projection(depthwise_out_ane)

            torch.testing.assert_close(final_out_ane, final_out, rtol=1e-3, atol=1e-3)
