"""Per-layer FLOPs calculation utilities for Metal Marlin operations.

This module provides tools to calculate theoretical FLOPs (floating-point
operations) for various neural network layers and quantized operations.
Useful for roofline analysis and performance profiling.

Example:
    from metal_marlin.utils.profile_ops import (
        calculate_matmul_flops,
        calculate_attention_flops,
        LayerFLOPsCounter
    )

    # Calculate FLOPs for a matrix multiplication
    flops = calculate_matmul_flops(M=4096, N=4096, K=4096)
    print(f"GEMM FLOPs: {flops / 1e12:.2f} TFLOPs")

    # Profile a model's layers
    counter = LayerFLOPsCounter()
    for name, layer in model.named_modules():
        counter.add_layer(name, layer, input_shape=(batch, seq, hidden))
    counter.print_summary()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LayerFLOPs:
    """FLOPs breakdown for a single layer.

    Attributes:
        name: Layer identifier (e.g., "transformer.layer.0.attention").
        total_flops: Total FLOPs for this layer.
        matmul_flops: FLOPs from matrix multiplications.
        activation_flops: FLOPs from non-linearities (GELU, SiLU, etc).
        attention_flops: FLOPs from attention operations.
        other_flops: FLOPs from other operations (normalization, etc).
        metadata: Optional context (input shape, layer config, etc).
    """

    name: str
    total_flops: int = 0
    matmul_flops: int = 0
    activation_flops: int = 0
    attention_flops: int = 0
    other_flops: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tflops(self) -> float:
        """Total FLOPs in trillions (TFLOPs)."""
        return self.total_flops / 1e12

    @property
    def gflops(self) -> float:
        """Total FLOPs in billions (GFLOPs)."""
        return self.total_flops / 1e9


def calculate_matmul_flops(M: int, N: int, K: int, *, quantized: bool = False) -> int:
    """Calculate FLOPs for a matrix multiplication C = A @ B.

    Args:
        M: Number of rows in A and C.
        N: Number of columns in B and C.
        K: Number of columns in A / rows in B.
        quantized: If True, accounts for dequantization overhead.
            For FP4 quantization, adds 2 ops per element (scale + shift).

    Returns:
        Total FLOPs (2*M*N*K for standard GEMM, more for quantized).

    Example:
        # Standard FP16 GEMM
        flops = calculate_matmul_flops(4096, 4096, 4096)
        # 2 * 4096^3 = 137,438,953,472 FLOPs

        # Quantized GEMM with dequant overhead
        flops = calculate_matmul_flops(4096, 4096, 4096, quantized=True)
    """
    # Standard GEMM: M*N dot products, each requiring K multiply-adds (2K ops)
    gemm_flops = 2 * M * N * K

    if quantized:
        # Dequantization: scale + zero-point per element
        # For group quantization with G groups: K*N/G scales to load
        # Simplified: assume 2 extra ops (multiply + add) per element
        dequant_flops = M * N * K * 2
        return gemm_flops + dequant_flops

    return gemm_flops


def calculate_attention_flops(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    *,
    causal: bool = False,
) -> int:
    """Calculate FLOPs for scaled dot-product attention.

    Computes FLOPs for: softmax(Q @ K^T / sqrt(d)) @ V

    Args:
        batch: Batch size.
        seq_len: Sequence length (both queries and keys).
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        causal: If True, computes causal masking overhead (roughly 0.5x FLOPs).

    Returns:
        Total FLOPs for attention operation.

    Example:
        # Standard self-attention with 32 heads
        flops = calculate_attention_flops(
            batch=8, seq_len=2048, num_heads=32, head_dim=128
        )
    """
    # Q @ K^T: (B, H, S, D) @ (B, H, D, S) -> (B, H, S, S)
    qk_flops = batch * num_heads * calculate_matmul_flops(seq_len, seq_len, head_dim)

    # Softmax: roughly 5 ops per element (exp, sum, div)
    # Only computed over the attention scores
    softmax_flops = batch * num_heads * seq_len * seq_len * 5

    # Scores @ V: (B, H, S, S) @ (B, H, S, D) -> (B, H, S, D)
    sv_flops = batch * num_heads * calculate_matmul_flops(seq_len, head_dim, seq_len)

    total = qk_flops + softmax_flops + sv_flops

    # Causal masking reduces computation by ~50% (triangular matrix)
    if causal:
        total = int(total * 0.5)

    return total


def calculate_ffn_flops(
    batch: int,
    seq_len: int,
    hidden_dim: int,
    ffn_dim: int,
    *,
    gated: bool = False,
    quantized: bool = False,
) -> int:
    """Calculate FLOPs for a feedforward network (FFN) layer.

    Args:
        batch: Batch size.
        seq_len: Sequence length.
        hidden_dim: Hidden dimension (input/output).
        ffn_dim: FFN intermediate dimension (usually 4x hidden_dim).
        gated: If True, uses gated activation (SwiGLU, GeGLU) which
            requires 2 weight matrices for up-projection.
        quantized: If True, adds dequantization overhead.

    Returns:
        Total FLOPs for FFN layer.

    Example:
        # Standard FFN: linear -> GELU -> linear
        flops = calculate_ffn_flops(8, 2048, 4096, 16384)

        # SwiGLU FFN with quantization
        flops = calculate_ffn_flops(8, 2048, 4096, 16384, gated=True, quantized=True)
    """
    tokens = batch * seq_len

    # Up-projection: (tokens, hidden) -> (tokens, ffn_dim)
    up_flops = calculate_matmul_flops(tokens, ffn_dim, hidden_dim, quantized=quantized)

    if gated:
        # Gated activation needs 2 weight matrices
        # gate = linear_gate(x), value = linear_value(x)
        # output = gate * activation(value)
        up_flops *= 2  # Two projections
        gate_flops = tokens * ffn_dim  # Element-wise multiply

    # Activation: roughly 8 ops per element for GELU, 3 for SiLU
    activation_flops = tokens * ffn_dim * 8

    # Down-projection: (tokens, ffn_dim) -> (tokens, hidden)
    down_flops = calculate_matmul_flops(tokens, hidden_dim, ffn_dim, quantized=quantized)

    total = up_flops + activation_flops + down_flops
    if gated:
        total += gate_flops

    return total


def calculate_layernorm_flops(batch: int, seq_len: int, hidden_dim: int) -> int:
    """Calculate FLOPs for LayerNorm or RMSNorm.

    Args:
        batch: Batch size.
        seq_len: Sequence length.
        hidden_dim: Hidden dimension.

    Returns:
        Total FLOPs (approximately 5 ops per element for mean/var/normalize).

    Example:
        flops = calculate_layernorm_flops(8, 2048, 4096)
    """
    num_elements = batch * seq_len * hidden_dim
    # Mean: 1 op, Variance: 2 ops, Normalize: 2 ops (subtract mean, div by std)
    return num_elements * 5


def calculate_embedding_flops(batch: int, seq_len: int, vocab_size: int, hidden_dim: int) -> int:
    """Calculate FLOPs for embedding lookup.

    Args:
        batch: Batch size.
        seq_len: Sequence length.
        vocab_size: Vocabulary size.
        hidden_dim: Embedding dimension.

    Returns:
        FLOPs (essentially memory bound, but counted as 1 op per element).

    Note:
        Embedding lookup is memory-bound and doesn't involve FP operations,
        but we count it as 1 op per element for accounting purposes.
    """
    return batch * seq_len * hidden_dim


@dataclass
class TransformerLayerFLOPs:
    """FLOPs breakdown for a full Transformer layer.

    Includes attention, FFN, and normalization.
    """

    attention: int = 0
    ffn: int = 0
    layernorm: int = 0
    total: int = 0

    @classmethod
    def from_config(
        cls,
        batch: int,
        seq_len: int,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        *,
        causal: bool = True,
        gated_ffn: bool = False,
        quantized: bool = False,
    ) -> TransformerLayerFLOPs:
        """Calculate FLOPs for a Transformer layer from config.

        Args:
            batch: Batch size.
            seq_len: Sequence length.
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            ffn_dim: FFN intermediate dimension.
            causal: Causal attention masking.
            gated_ffn: Use gated FFN (SwiGLU).
            quantized: Use quantized weights.

        Returns:
            TransformerLayerFLOPs with per-component breakdown.
        """
        head_dim = hidden_dim // num_heads

        attention = calculate_attention_flops(
            batch, seq_len, num_heads, head_dim, causal=causal
        )
        ffn = calculate_ffn_flops(
            batch, seq_len, hidden_dim, ffn_dim, gated=gated_ffn, quantized=quantized
        )
        # LayerNorm before attention + before FFN
        layernorm = calculate_layernorm_flops(batch, seq_len, hidden_dim) * 2

        total = attention + ffn + layernorm
        return cls(attention=attention, ffn=ffn, layernorm=layernorm, total=total)


class LayerFLOPsCounter:
    """Accumulates FLOPs for multiple layers in a model.

    Example:
        counter = LayerFLOPsCounter()

        # Add individual layers
        counter.add_matmul("layer.0.qkv", M=8192, N=12288, K=4096, quantized=True)
        counter.add_attention("layer.0.attn", batch=8, seq_len=2048, num_heads=32, head_dim=128)
        counter.add_ffn("layer.0.ffn", batch=8, seq_len=2048, hidden=4096, ffn=16384)

        # Print summary
        counter.print_summary()
        print(f"Total: {counter.total_tflops:.2f} TFLOPs")
    """

    def __init__(self) -> None:
        self._layers: list[LayerFLOPs] = []

    def add_layer(self, layer: LayerFLOPs) -> None:
        """Add a pre-computed LayerFLOPs."""
        self._layers.append(layer)

    def add_matmul(
        self,
        name: str,
        M: int,
        N: int,
        K: int,
        *,
        quantized: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a matrix multiplication layer."""
        flops = calculate_matmul_flops(M, N, K, quantized=quantized)
        layer = LayerFLOPs(
            name=name,
            total_flops=flops,
            matmul_flops=flops,
            metadata=metadata or {},
        )
        self._layers.append(layer)

    def add_attention(
        self,
        name: str,
        batch: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        *,
        causal: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an attention layer."""
        flops = calculate_attention_flops(batch, seq_len, num_heads, head_dim, causal=causal)
        layer = LayerFLOPs(
            name=name,
            total_flops=flops,
            attention_flops=flops,
            metadata=metadata or {},
        )
        self._layers.append(layer)

    def add_ffn(
        self,
        name: str,
        batch: int,
        seq_len: int,
        hidden_dim: int,
        ffn_dim: int,
        *,
        gated: bool = False,
        quantized: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an FFN layer."""
        flops = calculate_ffn_flops(
            batch, seq_len, hidden_dim, ffn_dim, gated=gated, quantized=quantized
        )
        layer = LayerFLOPs(
            name=name,
            total_flops=flops,
            matmul_flops=flops,  # Dominated by matmuls
            metadata=metadata or {},
        )
        self._layers.append(layer)

    def add_transformer_layer(
        self,
        name: str,
        batch: int,
        seq_len: int,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        *,
        causal: bool = True,
        gated_ffn: bool = False,
        quantized: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a full Transformer layer."""
        tf_flops = TransformerLayerFLOPs.from_config(
            batch, seq_len, hidden_dim, num_heads, ffn_dim,
            causal=causal, gated_ffn=gated_ffn, quantized=quantized
        )
        layer = LayerFLOPs(
            name=name,
            total_flops=tf_flops.total,
            matmul_flops=tf_flops.ffn,
            attention_flops=tf_flops.attention,
            other_flops=tf_flops.layernorm,
            metadata=metadata or {},
        )
        self._layers.append(layer)

    @property
    def total_flops(self) -> int:
        """Total FLOPs across all layers."""
        return sum(layer.total_flops for layer in self._layers)

    @property
    def total_tflops(self) -> float:
        """Total FLOPs in trillions."""
        return self.total_flops / 1e12

    @property
    def total_gflops(self) -> float:
        """Total FLOPs in billions."""
        return self.total_flops / 1e9

    def get_layer(self, name: str) -> LayerFLOPs | None:
        """Get a layer by name."""
        for layer in self._layers:
            if layer.name == name:
                return layer
        return None

    def get_layers(self) -> list[LayerFLOPs]:
        """Get all layers."""
        return list(self._layers)

    def print_summary(self, *, top_n: int = 20) -> None:
        """Print formatted summary of FLOPs by layer.

        Args:
            top_n: Number of top layers to show (default 20).
        """
        if not self._layers:
            print("No layers profiled")
            return

        sorted_layers = sorted(self._layers, key=lambda x: x.total_flops, reverse=True)

        print(f"{'Layer':<50} {'TFLOPs':>10} {'% Total':>8}")
        print("-" * 70)

        total = self.total_flops
        for layer in sorted_layers[:top_n]:
            pct = (layer.total_flops / total) * 100 if total > 0 else 0
            print(f"{layer.name:<50} {layer.tflops:>10.3f} {pct:>7.1f}%")

        print("-" * 70)
        print(f"{'TOTAL':<50} {self.total_tflops:>10.3f} {'100.0':>7}%")

        if len(sorted_layers) > top_n:
            print(f"\n(Showing top {top_n} of {len(sorted_layers)} layers)")

    def clear(self) -> None:
        """Clear all accumulated layers."""
        self._layers.clear()


# Convenience function for quick profiling
def profile_model_flops(
    batch: int,
    seq_len: int,
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    ffn_dim: int,
    vocab_size: int,
    *,
    causal: bool = True,
    gated_ffn: bool = False,
    quantized: bool = False,
) -> LayerFLOPsCounter:
    """Profile FLOPs for a standard Transformer model.

    Args:
        batch: Batch size.
        seq_len: Sequence length.
        num_layers: Number of Transformer layers.
        hidden_dim: Hidden dimension.
        num_heads: Number of attention heads.
        ffn_dim: FFN intermediate dimension.
        vocab_size: Vocabulary size.
        causal: Causal attention masking.
        gated_ffn: Use gated FFN (SwiGLU).
        quantized: Use quantized weights.

    Returns:
        LayerFLOPsCounter with per-layer breakdown.

    Example:
        # Profile Llama-7B model
        counter = profile_model_flops(
            batch=1, seq_len=2048, num_layers=32,
            hidden_dim=4096, num_heads=32, ffn_dim=11008,
            vocab_size=32000, causal=True, gated_ffn=True, quantized=True
        )
        counter.print_summary()
    """
    counter = LayerFLOPsCounter()

    # Embedding
    embed_flops = calculate_embedding_flops(batch, seq_len, vocab_size, hidden_dim)
    counter.add_layer(LayerFLOPs(name="embedding", total_flops=embed_flops))

    # Transformer layers
    for i in range(num_layers):
        counter.add_transformer_layer(
            name=f"layer.{i}",
            batch=batch,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            causal=causal,
            gated_ffn=gated_ffn,
            quantized=quantized,
        )

    # Output projection (lm_head)
    tokens = batch * seq_len
    lm_head_flops = calculate_matmul_flops(tokens, vocab_size, hidden_dim, quantized=quantized)
    counter.add_layer(LayerFLOPs(name="lm_head", total_flops=lm_head_flops, matmul_flops=lm_head_flops))

    return counter
