"""
Hybrid Conformer Block for ASR models with GPU+ANE co-processing.

This module implements a conformer block that intelligently splits computation
between GPU and Apple Neural Engine (ANE) for optimal performance on Apple Silicon.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ane.ane_interface import ANEInterface
from ..core.metal_ops import MetalOps


class MHSAWithANE(nn.Module):
    """Multi-Head Self Attention with GPU+ANE co-processing."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        ane_heads: int = 4,  # Heads processed on ANE
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ane_heads = ane_heads
        self.gpu_heads = num_heads - ane_heads

        assert self.gpu_heads >= 0, "GPU heads must be non-negative"
        assert self.ane_heads <= num_heads, "ANE heads cannot exceed total heads"

        self.scale = head_dim**-0.5

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, dim, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # ANE interface (available on Apple Silicon)
        self._ane_interface = None
        self._metal_ops = None

        # Initialize ANE interface if available
        try:
            self._ane_interface = ANEInterface()
            self._metal_ops = MetalOps()
        except Exception:
            # ANE not available, fallback to GPU-only
            self.ane_heads = 0
            self.gpu_heads = num_heads

    def _split_attention_heads(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split attention heads between GPU and ANE."""
        B, H, W, C = x.shape

        # Split channels for GPU and ANE
        gpu_channels = self.gpu_heads * self.head_dim
        ane_channels = self.ane_heads * self.head_dim

        gpu_x = x[..., :gpu_channels]
        ane_x = x[..., gpu_channels : gpu_channels + ane_channels]

        return gpu_x, ane_x

    def _gpu_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Standard GPU attention computation."""
        B, H, N, C = q.shape

        # Reshape for batch matrix multiplication
        q = q.transpose(1, 2)  # (B, N, H, C)
        k = k.transpose(1, 2)  # (B, N, H, C)
        v = v.transpose(1, 2)  # (B, N, H, C)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        output = torch.matmul(attn_weights, v)

        # Reshape back
        output = output.transpose(1, 2)  # (B, H, N, C)
        return output

    def _ane_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """ANE-accelerated attention computation."""
        if self._ane_interface is None or self._metal_ops is None:
            # Fallback to GPU if ANE unavailable
            return self._gpu_attention(q, k, v)

        # Reshape for ANE processing
        B, H, N, C = q.shape

        # Process through ANE interface
        try:
            # ANE expects (B, N, H*C) format
            q_flat = q.permute(0, 2, 1, 3).reshape(B, N, -1)
            k_flat = k.permute(0, 2, 1, 3).reshape(B, N, -1)
            v_flat = v.permute(0, 2, 1, 3).reshape(B, N, -1)

            # Use Metal-ANE bridge for attention
            output_flat = self._ane_interface.compute_attention(
                q_flat, k_flat, v_flat, num_heads=self.ane_heads, head_dim=self.head_dim
            )

            # Reshape back to (B, H, N, C)
            output = output_flat.reshape(B, N, self.ane_heads, self.head_dim).permute(0, 2, 1, 3)
            return output

        except Exception:
            # Fallback to GPU if ANE processing fails
            return self._gpu_attention(q, k, v)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass with hybrid GPU+ANE attention.

        Args:
            x: Input tensor (B, N, C)
            mask: Optional attention mask

        Returns:
            Output tensor (B, N, C)
        """
        B, N, C = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Split heads between GPU and ANE
        if self.ane_heads > 0:
            q_gpu, q_ane = q[:, : self.gpu_heads], q[:, self.gpu_heads :]
            k_gpu, k_ane = k[:, : self.gpu_heads], k[:, self.gpu_heads :]
            v_gpu, v_ane = v[:, : self.gpu_heads], v[:, self.gpu_heads :]

            # Process in parallel
            gpu_output = self._gpu_attention(q_gpu, k_gpu, v_gpu)
            ane_output = self._ane_attention(q_ane, k_ane, v_ane)

            # Concatenate results
            attn_output = torch.cat([gpu_output, ane_output], dim=1)
        else:
            # GPU-only processing
            attn_output = self._gpu_attention(q, k, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, -1)
        output = self.out_proj(attn_output)

        return output


class ConvolutionModuleWithANE(nn.Module):
    """1D Convolution module with GPU+ANE processing."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout: float = 0.1,
        use_ane: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.use_ane = use_ane and (channels <= 512)  # ANE works best with smaller channels

        # Pointwise convolutions
        self.pointwise_conv1 = nn.Conv1d(channels, channels * expansion_factor, kernel_size=1)
        self.pointwise_conv2 = nn.Conv1d(channels * expansion_factor, channels, kernel_size=1)

        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            channels * expansion_factor,
            channels * expansion_factor,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels * expansion_factor,
        )

        # Normalization and activation
        self.batch_norm = nn.BatchNorm1d(channels * expansion_factor)
        self.activation = nn.GLU(dim=1)
        self.dropout = nn.Dropout(dropout)

        # ANE interface for depthwise convolution
        self._ane_interface = None
        self._metal_ops = None

        if self.use_ane:
            try:
                self._ane_interface = ANEInterface()
                self._metal_ops = MetalOps()
            except Exception:
                self.use_ane = False

    def _ane_depthwise_conv(self, x: torch.Tensor) -> torch.Tensor:
        """ANE-accelerated depthwise convolution."""
        if self._ane_interface is None or self._metal_ops is None:
            return self.depthwise_conv(x)

        try:
            # Process through ANE-optimized depthwise convolution
            return self._ane_interface.depthwise_conv1d(
                x,
                self.depthwise_conv.weight,
                padding=self.depthwise_conv.padding[0],
                groups=self.depthwise_conv.groups,
            )
        except Exception:
            return self.depthwise_conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with hybrid processing.

        Args:
            x: Input tensor (B, C, T)

        Returns:
            Output tensor (B, C, T)
        """
        # Pointwise convolution expansion (GPU)
        x = self.pointwise_conv1(x)
        x = self.batch_norm(x)

        # Depthwise convolution (ANE if available)
        if self.use_ane:
            x = self._ane_depthwise_conv(x)
        else:
            x = self.depthwise_conv(x)

        x = self.activation(x)

        # Pointwise convolution projection (GPU)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        return x


class FeedForwardWithANE(nn.Module):
    """Feed-forward module with GPU+ANE processing."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1, use_ane: bool = True):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.use_ane = use_ane and (hidden_dim <= 2048)  # ANE limit

        # Linear layers
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Activation
        self.activation = nn.GELU()

        # ANE interface
        self._ane_interface = None
        self._metal_ops = None

        if self.use_ane:
            try:
                self._ane_interface = ANEInterface()
                self._metal_ops = MetalOps()
            except Exception:
                self.use_ane = False

    def _ane_linear(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        """ANE-accelerated linear layer."""
        if self._ane_interface is None:
            return F.linear(x, weight, bias)

        try:
            return self._ane_interface.linear(x, weight, bias)
        except Exception:
            return F.linear(x, weight, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with hybrid processing.

        Args:
            x: Input tensor (B, N, C)

        Returns:
            Output tensor (B, N, C)
        """
        # First linear layer
        if self.use_ane:
            x = self._ane_linear(x, self.fc1.weight, self.fc1.bias)
        else:
            x = self.fc1(x)

        x = self.activation(x)
        x = self.dropout(x)

        # Second linear layer
        if self.use_ane:
            x = self._ane_linear(x, self.fc2.weight, self.fc2.bias)
        else:
            x = self.fc2(x)

        return x


class HybridConformerBlock(nn.Module):
    """
    Hybrid Conformer Block with intelligent GPU+ANE workload distribution.

    This block automatically determines the optimal split between GPU and ANE
    based on tensor sizes and available resources.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        ane_heads_ratio: float = 0.5,  # Ratio of attention heads for ANE
        enable_ane_conv: bool = True,
        enable_ane_ffn: bool = True,
    ):
        super().__init__()
        self.dim = dim

        # Calculate ANE heads count
        ane_heads = int(num_heads * ane_heads_ratio)

        # Multi-head self attention with hybrid processing
        self.mhsa = MHSAWithANE(
            dim=dim,
            num_heads=num_heads,
            head_dim=dim // num_heads,
            ane_heads=ane_heads,
            dropout=dropout,
        )

        # Convolution module with hybrid processing
        self.conv = ConvolutionModuleWithANE(
            channels=dim, kernel_size=conv_kernel, dropout=dropout, use_ane=enable_ane_conv
        )

        # Feed-forward module with hybrid processing
        self.ffn = FeedForwardWithANE(
            dim=dim, hidden_dim=ffn_dim, dropout=dropout, use_ane=enable_ane_ffn
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass through hybrid conformer block.

        Args:
            x: Input tensor (B, T, C)
            mask: Optional attention mask

        Returns:
            Output tensor (B, T, C)
        """
        # Residual connection + MHSA
        residual = x
        x = self.norm1(x)
        x = self.mhsa(x, mask)
        x = self.dropout(x) + residual

        # Residual connection + Convolution (need to transpose for conv1d)
        residual = x
        x = self.norm2(x)
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, T, C)
        x = residual + x

        # Residual connection + FFN
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = self.dropout(x) + residual

        return x

    def get_device_usage_stats(self) -> dict:
        """Get statistics about GPU vs ANE usage."""
        return {
            "attention": {
                "total_heads": self.mhsa.num_heads,
                "ane_heads": self.mhsa.ane_heads,
                "gpu_heads": self.mhsa.gpu_heads,
                "ane_ratio": self.mhsa.ane_heads / self.mhsa.num_heads
                if self.mhsa.num_heads > 0
                else 0,
            },
            "convolution": {"use_ane": self.conv.use_ane, "channels": self.conv.channels},
            "feedforward": {"use_ane": self.ffn.use_ane, "hidden_dim": self.ffn.hidden_dim},
        }


# Factory function for creating blocks with optimal ANE usage
def create_hybrid_conformer_block(
    dim: int,
    target_device: str = "auto",
    optimize_for: str = "throughput",  # "throughput" or "latency"
) -> HybridConformerBlock:
    """
    Create a hybrid conformer block with optimal ANE configuration.

    Args:
        dim: Model dimension
        target_device: "gpu", "ane", or "auto" for automatic selection
        optimize_for: "throughput" or "latency" optimization goal

    Returns:
        Configured HybridConformerBlock
    """
    if target_device == "gpu":
        # GPU-only configuration
        return HybridConformerBlock(
            dim=dim, ane_heads_ratio=0.0, enable_ane_conv=False, enable_ane_ffn=False
        )
    elif target_device == "ane":
        # Maximize ANE usage
        return HybridConformerBlock(
            dim=dim, ane_heads_ratio=0.7, enable_ane_conv=True, enable_ane_ffn=True
        )
    else:
        # Auto-optimization based on model size
        if optimize_for == "throughput":
            # Favor ANE for larger models where parallelism helps
            ane_ratio = min(0.6, dim / 1024)
            enable_conv = dim <= 512
            enable_ffn = dim <= 1024
        else:
            # Favor GPU for lower latency
            ane_ratio = min(0.3, dim / 2048)
            enable_conv = dim <= 256
            enable_ffn = dim <= 512

        return HybridConformerBlock(
            dim=dim,
            ane_heads_ratio=ane_ratio,
            enable_ane_conv=enable_conv,
            enable_ane_ffn=enable_ffn,
        )
