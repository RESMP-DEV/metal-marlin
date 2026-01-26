"""Differential Transformer attention implementation for Metal Marlin.

Microsoft's Differential Transformer (2024) modifies standard attention to reduce
noise and improve long-context performance by computing differential attention:

    Output = softmax(Q1 @ K1^T / sqrt(d)) @ V - lambda * softmax(Q2 @ K2^T / sqrt(d)) @ V

Key features:
- Two attention heads per "logical" head (splits Q/K projections)
- Lambda parameter controls the subtraction weight (learnable or fixed)
- Reduces attention noise by subtracting a "baseline" attention pattern
- Improved performance on long-context tasks and retrieval

References:
    - Paper: "Differential Transformer" (Microsoft Research, 2024)
    - https://arxiv.org/abs/2410.05258

Usage:
    from metal_marlin.architectures.diff_transformer import (
        DifferentialAttentionConfig,
        DifferentialMarlinAttention,
        parse_diff_transformer_config,
    )

    # Parse config from HuggingFace model config
    config = parse_diff_transformer_config(hf_config_dict)

    # Create attention layer
    attn = DifferentialMarlinAttention(
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        lambda_init=0.8,
        lambda_learnable=True,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._compat import HAS_MLX, from_numpy, mx, nn, require_mlx, to_numpy
from ..layers import MarlinLinear

if TYPE_CHECKING:
    import mlx.core as mx
    import mlx.nn as nn


if HAS_MLX and mx is not None:
    _DIFF_ATTN_SOURCE: str | None = None
    _diff_attn_kernel: object | None = None
    _diff_attn_causal_kernel: object | None = None

    _DIFF_ATTN_ROWS_PER_TG = 4
    _DIFF_ATTN_THREADS_PER_TG = 32 * _DIFF_ATTN_ROWS_PER_TG

    def _get_diff_attn_source() -> str:
        global _DIFF_ATTN_SOURCE
        if _DIFF_ATTN_SOURCE is None:
            shader_path = Path(__file__).parent.parent / "src" / "diff_attention.metal"
            _DIFF_ATTN_SOURCE = shader_path.read_text()
        return _DIFF_ATTN_SOURCE

    def _get_diff_attn_kernel(causal: bool) -> object:
        global _diff_attn_kernel, _diff_attn_causal_kernel
        if causal:
            if _diff_attn_causal_kernel is None:
                _diff_attn_causal_kernel = mx.fast.metal_kernel(
                    name="diff_attention_causal",
                    input_names=[
                        "Q1",
                        "Q2",
                        "K1",
                        "K2",
                        "V",
                        "lambda_vals",
                        "batch",
                        "num_heads_q",
                        "num_heads_k",
                        "seq_q",
                        "seq_k",
                        "head_dim",
                        "scale",
                        "lambda_per_head",
                    ],
                    output_names=["O"],
                    source=_get_diff_attn_source(),
                    ensure_row_contiguous=True,
                )
            return _diff_attn_causal_kernel

        if _diff_attn_kernel is None:
            _diff_attn_kernel = mx.fast.metal_kernel(
                name="diff_attention",
                input_names=[
                    "Q1",
                    "Q2",
                    "K1",
                    "K2",
                    "V",
                    "lambda_vals",
                    "batch",
                    "num_heads_q",
                    "num_heads_k",
                    "seq_q",
                    "seq_k",
                    "head_dim",
                    "scale",
                    "lambda_per_head",
                ],
                output_names=["O"],
                source=_get_diff_attn_source(),
                ensure_row_contiguous=True,
            )
        return _diff_attn_kernel


@dataclass
class DifferentialAttentionConfig:
    """Configuration for Differential Transformer attention.

    Attributes:
        hidden_size: Model hidden dimension.
        num_attention_heads: Number of "logical" attention heads.
            Each logical head is split into two sub-heads for differential attention.
        num_key_value_heads: Number of KV heads (for GQA). Must divide num_attention_heads.
        head_dim: Dimension per head. Defaults to hidden_size // num_attention_heads.
        lambda_init: Initial value for lambda parameter. Default: 0.8.
            Microsoft recommends initializing around 0.8 for best results.
        lambda_learnable: Whether lambda is learnable (True) or fixed (False).
            Default: True. Learnable lambda adapts per-layer.
        lambda_per_head: Whether to use separate lambda per head (True) or
            shared across all heads (False). Default: False.
        sublayer_norm: Whether to apply LayerNorm after differential attention
            (before output projection). Default: False.
        rope_theta: Base frequency for RoPE embeddings. Default: 10000.0.
        max_position_embeddings: Maximum sequence length for RoPE. Default: 4096.
        use_bias: Whether Q/K/V/O projections have bias. Default: False.
        q1k1_quant_type: Optional quantization type for Q1/K1 projections.
            When set, enables separate Q1/K1 projections for mixed precision.
        q2k2_quant_type: Optional quantization type for Q2/K2 projections.
            When set, enables separate Q2/K2 projections for mixed precision.
        use_fused_kernel: Enable fused differential attention kernel (MLX only).
    """

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    lambda_init: float = 0.8
    lambda_learnable: bool = True
    lambda_per_head: bool = False
    sublayer_norm: bool = False
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096
    use_bias: bool = False
    q1k1_quant_type: Literal["fp4", "fp16"] | None = None
    q2k2_quant_type: Literal["fp4", "fp16"] | None = None
    use_fused_kernel: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        # Validate GQA configuration
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be "
                f"divisible by num_key_value_heads ({self.num_key_value_heads})"
            )


def parse_diff_transformer_config(config_dict: dict[str, Any]) -> DifferentialAttentionConfig:
    """Parse Differential Transformer config from HuggingFace config.json.

    Supports both native DiffTransformer configs and standard transformer configs
    that need to be adapted for differential attention.

    Args:
        config_dict: Dictionary loaded from config.json.

    Returns:
        DifferentialAttentionConfig with parsed settings.

    Example config.json fields (native DiffTransformer):
        {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "diff_attn_lambda_init": 0.8,
            "diff_attn_lambda_learnable": true,
            "diff_attn_lambda_per_head": false,
            "diff_attn_sublayer_norm": false,
            "rope_theta": 10000.0,
            ...
        }
    """
    # Core dimensions (standard transformer fields)
    hidden_size = config_dict.get("hidden_size", config_dict.get("d_model", 4096))
    num_heads = config_dict.get("num_attention_heads", config_dict.get("n_head", 32))
    num_kv_heads = config_dict.get(
        "num_key_value_heads",
        config_dict.get("num_kv_heads", num_heads)
    )
    head_dim = config_dict.get(
        "head_dim",
        config_dict.get("kv_channels", hidden_size // num_heads)
    )

    # Differential attention specific fields
    lambda_init = config_dict.get("diff_attn_lambda_init", 0.8)
    lambda_learnable = config_dict.get("diff_attn_lambda_learnable", True)
    lambda_per_head = config_dict.get("diff_attn_lambda_per_head", False)
    sublayer_norm = config_dict.get("diff_attn_sublayer_norm", False)
    q1k1_quant_type = config_dict.get("diff_attn_q1k1_quant_type")
    q2k2_quant_type = config_dict.get("diff_attn_q2k2_quant_type")
    use_fused_kernel = config_dict.get("diff_attn_use_fused_kernel", True)

    # RoPE settings
    rope_theta = config_dict.get("rope_theta", 10000.0)
    max_position = config_dict.get("max_position_embeddings", 4096)

    # Bias
    use_bias = config_dict.get("attention_bias", config_dict.get("use_bias", False))

    return DifferentialAttentionConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        lambda_init=lambda_init,
        lambda_learnable=lambda_learnable,
        lambda_per_head=lambda_per_head,
        sublayer_norm=sublayer_norm,
        rope_theta=rope_theta,
        max_position_embeddings=max_position,
        use_bias=use_bias,
        q1k1_quant_type=q1k1_quant_type,
        q2k2_quant_type=q2k2_quant_type,
        use_fused_kernel=use_fused_kernel,
    )


class RoPE:
    """Rotary Position Embedding for differential attention.

    Identical to standard RoPE but handles the split head dimensions
    appropriately for differential attention.
    """

    def __init__(self, dims: int, traditional: bool = False, base: float = 10000.0):
        self.dims = dims
        self.traditional = traditional
        self.base = base

    def __call__(self, x: Any, offset: int = 0) -> Any:
        """Apply RoPE to input tensor.

        Args:
            x: Input tensor [batch, num_heads, seq_len, head_dim]
            offset: Position offset for KV cache

        Returns:
            Tensor with RoPE applied
        """
        if not HAS_MLX:
            return self._apply_numpy(to_numpy(x), offset)

        shape = x.shape
        seq_len = shape[2]
        shape[3]

        # Compute position indices
        positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)

        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (mx.arange(0, self.dims, 2, dtype=mx.float32) / self.dims))

        # Compute angles: [seq_len, dims/2]
        freqs = mx.outer(positions, inv_freq)

        # Compute cos and sin
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)

        # Expand dims for broadcasting: [1, 1, seq_len, dims/2]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        # Split into even and odd indices
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # Apply rotation (Llama-style)
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_odd * cos + x_even * sin

        # Interleave back
        x_rotated = mx.concatenate(
            [x_rotated_even[..., None], x_rotated_odd[..., None]], axis=-1
        ).reshape(shape)

        return x_rotated

    def _apply_numpy(self, x: np.ndarray, offset: int = 0) -> np.ndarray:
        """NumPy fallback for RoPE."""
        shape = x.shape
        seq_len = shape[2]

        positions = np.arange(offset, offset + seq_len, dtype=np.float32)
        inv_freq = 1.0 / (self.base ** (np.arange(0, self.dims, 2, dtype=np.float32) / self.dims))
        freqs = np.outer(positions, inv_freq)

        cos = np.cos(freqs)[None, None, :, :]
        sin = np.sin(freqs)[None, None, :, :]

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_odd * cos + x_even * sin

        x_rotated = np.zeros_like(x)
        x_rotated[..., ::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd

        return x_rotated


def _fused_diff_attention_mlx(
    q1: Any,
    q2: Any,
    k1: Any,
    k2: Any,
    v: Any,
    lambda_vals: Any,
    scale: float,
    lambda_per_head: bool,
    use_causal: bool,
) -> Any:
    if not HAS_MLX or mx is None:
        raise RuntimeError("Fused differential attention requires MLX.")

    batch, num_heads_q, seq_q, head_dim = q1.shape
    num_heads_k = k1.shape[1]
    seq_k = k1.shape[2]

    if head_dim > 128 or head_dim % 32 != 0:
        raise ValueError(
            "Fused differential attention requires head_dim <= 128 and "
            "head_dim divisible by 32."
        )

    kernel = _get_diff_attn_kernel(causal=use_causal)

    q1 = q1.astype(mx.float16)
    q2 = q2.astype(mx.float16)
    k1 = k1.astype(mx.float16)
    k2 = k2.astype(mx.float16)
    v = v.astype(mx.float16)
    lambda_vals = lambda_vals.astype(mx.float16)

    batch_const = mx.array([batch], dtype=mx.uint32)
    num_heads_q_const = mx.array([num_heads_q], dtype=mx.uint32)
    num_heads_k_const = mx.array([num_heads_k], dtype=mx.uint32)
    seq_q_const = mx.array([seq_q], dtype=mx.uint32)
    seq_k_const = mx.array([seq_k], dtype=mx.uint32)
    head_dim_const = mx.array([head_dim], dtype=mx.uint32)
    scale_const = mx.array([scale], dtype=mx.float32)
    lambda_per_head_const = mx.array([lambda_per_head], dtype=mx.bool_)

    grid_x = num_heads_q
    grid_y = (seq_q + _DIFF_ATTN_ROWS_PER_TG - 1) // _DIFF_ATTN_ROWS_PER_TG
    grid_z = batch

    output_size = batch * num_heads_q * seq_q * head_dim

    outputs = kernel(
        inputs=[
            q1.reshape(-1),
            q2.reshape(-1),
            k1.reshape(-1),
            k2.reshape(-1),
            v.reshape(-1),
            lambda_vals.reshape(-1),
            batch_const,
            num_heads_q_const,
            num_heads_k_const,
            seq_q_const,
            seq_k_const,
            head_dim_const,
            scale_const,
            lambda_per_head_const,
        ],
        grid=(grid_x, grid_y, grid_z),
        threadgroup=(_DIFF_ATTN_THREADS_PER_TG, 1, 1),
        output_shapes=[(output_size,)],
        output_dtypes=[mx.float16],
    )

    return outputs[0].reshape(batch, num_heads_q, seq_q, head_dim)


class DifferentialAttention:
    """Pure differential attention computation (no projections).

    This class implements the core differential attention mechanism:
        Output = softmax(Q1 @ K1^T / sqrt(d)) @ V - lambda * softmax(Q2 @ K2^T / sqrt(d)) @ V

    The Q, K, V inputs are expected to already be split into sub-head pairs.
    This separation allows using this class with either FP16 or quantized projections.

    Args:
        num_heads: Number of logical attention heads.
        head_dim: Dimension per head.
        lambda_init: Initial lambda value. Default: 0.8.
        lambda_learnable: Whether lambda is trainable. Default: True.
        lambda_per_head: Use separate lambda per head. Default: False.
        sublayer_norm: Apply LayerNorm after attention. Default: False.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        lambda_init: float = 0.8,
        lambda_learnable: bool = True,
        lambda_per_head: bool = False,
        sublayer_norm: bool = False,
        use_fused_kernel: bool = True,
    ):
        if HAS_MLX and nn is not None:
            nn.Module.__init__(self)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.sublayer_norm = sublayer_norm
        self.lambda_per_head = lambda_per_head
        self.use_fused_kernel = use_fused_kernel

        # Lambda parameter
        # Shape: [num_heads] if per_head, else [1]
        lambda_shape = (num_heads,) if lambda_per_head else (1,)

        if HAS_MLX and mx is not None:
            if lambda_learnable:
                # Learnable parameter stored as log to ensure positivity
                # lambda = exp(lambda_log) where lambda_log is initialized to log(lambda_init)
                init_val = math.log(lambda_init) if lambda_init > 0 else -2.0
                self.lambda_log = mx.full(lambda_shape, init_val, dtype=mx.float16)
            else:
                # Fixed lambda
                self.lambda_log = None
                self.lambda_fixed = mx.full(lambda_shape, lambda_init, dtype=mx.float16)

            if sublayer_norm:
                # LayerNorm after differential attention
                self.sublayer_ln = nn.LayerNorm(head_dim)
        else:
            # NumPy fallback
            if lambda_learnable:
                init_val = math.log(lambda_init) if lambda_init > 0 else -2.0
                self.lambda_log = np.full(lambda_shape, init_val, dtype=np.float16)
            else:
                self.lambda_log = None
                self.lambda_fixed = np.full(lambda_shape, lambda_init, dtype=np.float16)
            self.sublayer_ln = None

        self.lambda_learnable = lambda_learnable

    def get_lambda(self) -> Any:
        """Get current lambda value(s)."""
        if self.lambda_learnable:
            if HAS_MLX and mx is not None:
                return mx.exp(self.lambda_log).astype(mx.float16)
            else:
                return np.exp(self.lambda_log).astype(np.float16)
        else:
            return self.lambda_fixed

    def __call__(
        self,
        q1: Any,  # [batch, num_heads, seq_q, head_dim]
        k1: Any,  # [batch, num_kv_heads, seq_k, head_dim]
        v: Any,   # [batch, num_kv_heads, seq_k, head_dim]
        q2: Any,  # [batch, num_heads, seq_q, head_dim]
        k2: Any,  # [batch, num_kv_heads, seq_k, head_dim]
        attention_mask: Any | None = None,
        attention_mask_is_causal: bool = False,
    ) -> Any:
        """Compute differential attention.

        Args:
            q1, k1: First attention sub-head Q/K (positive component)
            v: Value tensor (shared between both attention computations)
            q2, k2: Second attention sub-head Q/K (negative/baseline component)
            attention_mask: Optional causal mask with -inf for masked positions
            attention_mask_is_causal: True if the mask is causal (allows fused kernel)

        Returns:
            Differential attention output [batch, num_heads, seq_q, head_dim]
        """
        if HAS_MLX:
            return self._forward_mlx(
                q1,
                k1,
                v,
                q2,
                k2,
                attention_mask,
                attention_mask_is_causal,
            )
        else:
            return self._forward_numpy(
                to_numpy(q1), to_numpy(k1), to_numpy(v),
                to_numpy(q2), to_numpy(k2),
                to_numpy(attention_mask) if attention_mask is not None else None
            )

    def _forward_mlx(
        self,
        q1: Any, k1: Any, v: Any,
        q2: Any, k2: Any,
        attention_mask: Any | None,
        attention_mask_is_causal: bool,
    ) -> Any:
        """MLX implementation of differential attention."""
        batch_size, num_heads, seq_q, head_dim = q1.shape
        num_kv_heads = k1.shape[1]

        use_fused = self.use_fused_kernel and (
            attention_mask is None or attention_mask_is_causal
        )

        if use_fused:
            try:
                return _fused_diff_attention_mlx(
                    q1=q1,
                    q2=q2,
                    k1=k1,
                    k2=k2,
                    v=v,
                    lambda_vals=self.get_lambda(),
                    scale=self.scale,
                    lambda_per_head=self.lambda_per_head,
                    use_causal=attention_mask_is_causal,
                )
            except (ValueError, RuntimeError):
                # Fall back to reference path when kernel constraints are not met.
                use_fused = False

        if attention_mask is None and attention_mask_is_causal:
            attention_mask = create_causal_mask(seq_q, k1.shape[2])

        if num_kv_heads < num_heads:
            repeat_factor = num_heads // num_kv_heads
            k1 = mx.repeat(k1, repeat_factor, axis=1)
            k2 = mx.repeat(k2, repeat_factor, axis=1)
            v = mx.repeat(v, repeat_factor, axis=1)

        # Compute attention scores for both sub-heads
        # attn1 = softmax(Q1 @ K1^T / sqrt(d)) @ V
        # attn2 = softmax(Q2 @ K2^T / sqrt(d)) @ V
        scores1 = (q1 @ k1.transpose(0, 1, 3, 2)) * self.scale
        scores2 = (q2 @ k2.transpose(0, 1, 3, 2)) * self.scale

        # Apply attention mask (causal mask with -inf)
        if attention_mask is not None:
            scores1 = scores1 + attention_mask
            scores2 = scores2 + attention_mask

        # Softmax
        weights1 = mx.softmax(scores1, axis=-1)
        weights2 = mx.softmax(scores2, axis=-1)

        # Compute weighted values
        attn1 = weights1 @ v
        attn2 = weights2 @ v

        # Differential attention: output = attn1 - lambda * attn2
        lambda_val = self.get_lambda()
        if self.lambda_per_head:
            # lambda_val: [num_heads] -> [1, num_heads, 1, 1]
            lambda_val = lambda_val[None, :, None, None]
        else:
            # lambda_val: [1] -> scalar broadcast
            pass

        output = attn1 - lambda_val * attn2

        # Optional sublayer norm
        if self.sublayer_norm and self.sublayer_ln is not None:
            # Apply per head dim: [batch, heads, seq, head_dim]
            orig_shape = output.shape
            output = output.reshape(-1, head_dim)
            output = self.sublayer_ln(output)
            output = output.reshape(orig_shape)

        return output

    def _forward_numpy(
        self,
        q1: np.ndarray, k1: np.ndarray, v: np.ndarray,
        q2: np.ndarray, k2: np.ndarray,
        attention_mask: np.ndarray | None,
    ) -> np.ndarray:
        """NumPy fallback implementation."""
        batch_size, num_heads, seq_q, head_dim = q1.shape
        num_kv_heads = k1.shape[1]

        # Handle GQA
        if num_kv_heads < num_heads:
            repeat_factor = num_heads // num_kv_heads
            k1 = np.repeat(k1, repeat_factor, axis=1)
            k2 = np.repeat(k2, repeat_factor, axis=1)
            v = np.repeat(v, repeat_factor, axis=1)

        # Compute attention scores
        scores1 = (q1 @ k1.transpose(0, 1, 3, 2)) * self.scale
        scores2 = (q2 @ k2.transpose(0, 1, 3, 2)) * self.scale

        if attention_mask is not None:
            scores1 = scores1 + attention_mask
            scores2 = scores2 + attention_mask

        # Stable softmax
        def softmax(x, axis=-1):
            x_max = np.max(x, axis=axis, keepdims=True)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

        weights1 = softmax(scores1)
        weights2 = softmax(scores2)

        attn1 = weights1 @ v
        attn2 = weights2 @ v

        lambda_val = self.get_lambda()
        if self.lambda_per_head:
            lambda_val = lambda_val[None, :, None, None]

        output = attn1 - lambda_val * attn2

        return output


class DifferentialMarlinAttention:
    """Full Differential Transformer attention layer with Marlin-quantized projections.

    This is a drop-in replacement for standard attention layers in transformer models.
    It uses MarlinLinear for quantized Q/K/V/O projections and implements the
    differential attention mechanism from Microsoft's Differential Transformer.

    The key difference from standard attention:
    - Q projection outputs 2x the normal size (Q1 and Q2 concatenated)
    - K projection outputs 2x the normal size (K1 and K2 concatenated)
    - V projection outputs normal size (shared between both attention paths)
    - Output = softmax(Q1@K1^T/sqrt(d))@V - lambda * softmax(Q2@K2^T/sqrt(d))@V

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of logical attention heads.
        num_kv_heads: Number of KV heads (for GQA). Defaults to num_heads.
        head_dim: Dimension per head. Defaults to hidden_size // num_heads.
        lambda_init: Initial lambda value. Default: 0.8.
        lambda_learnable: Whether lambda is trainable. Default: True.
        lambda_per_head: Use separate lambda per head. Default: False.
        sublayer_norm: Apply LayerNorm after attention. Default: False.
        quant_type: Quantization type for projections. Default: "fp4".
        q1k1_quant_type: Optional quantization type for Q1/K1 projections.
        q2k2_quant_type: Optional quantization type for Q2/K2 projections.
        group_size: Quantization group size. Default: 128.
        rope_theta: RoPE base frequency. Default: 10000.0.
        max_position_embeddings: Maximum position for RoPE. Default: 4096.
        bias: Whether projections have bias. Default: False.
        use_fused_kernel: Enable fused differential attention kernel (MLX only).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        lambda_init: float = 0.8,
        lambda_learnable: bool = True,
        lambda_per_head: bool = False,
        sublayer_norm: bool = False,
        quant_type: Literal["fp4", "fp16"] = "fp4",
        q1k1_quant_type: Literal["fp4", "fp16"] | None = None,
        q2k2_quant_type: Literal["fp4", "fp16"] | None = None,
        group_size: int = 128,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 4096,
        bias: bool = False,
        use_fused_kernel: bool = True,
    ):
        if HAS_MLX and nn is not None:
            nn.Module.__init__(self)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.quant_type = quant_type
        self.q1k1_quant_type = q1k1_quant_type
        self.q2k2_quant_type = q2k2_quant_type
        self.split_qk_projections = (
            q1k1_quant_type is not None or q2k2_quant_type is not None
        )

        def _build_proj(out_features: int, proj_quant_type: Literal["fp4", "fp16"]) -> Any:
            if proj_quant_type == "fp4":
                return MarlinLinear(
                    hidden_size,
                    out_features,
                    bias=bias,
                    quant_type="fp4",
                    group_size=group_size,
                )
            require_mlx("FP16 differential attention projections")
            return nn.Linear(hidden_size, out_features, bias=bias)

        v_size = self.num_kv_heads * self.head_dim
        o_size = num_heads * self.head_dim

        if self.split_qk_projections:
            q1_type = q1k1_quant_type or quant_type
            q2_type = q2k2_quant_type or quant_type
            k1_type = q1k1_quant_type or quant_type
            k2_type = q2k2_quant_type or quant_type

            self.q_proj = None
            self.k_proj = None
            self.q1_proj = _build_proj(num_heads * self.head_dim, q1_type)
            self.q2_proj = _build_proj(num_heads * self.head_dim, q2_type)
            self.k1_proj = _build_proj(self.num_kv_heads * self.head_dim, k1_type)
            self.k2_proj = _build_proj(self.num_kv_heads * self.head_dim, k2_type)
        else:
            q_size = num_heads * self.head_dim * 2
            k_size = self.num_kv_heads * self.head_dim * 2

            if quant_type == "fp4":
                self.q_proj = MarlinLinear(
                    hidden_size, q_size, bias=bias,
                    quant_type="fp4", group_size=group_size,
                )
                self.k_proj = MarlinLinear(
                    hidden_size, k_size, bias=bias,
                    quant_type="fp4", group_size=group_size,
                )
            else:
                require_mlx("FP16 differential attention projections")
                self.q_proj = nn.Linear(hidden_size, q_size, bias=bias)
                self.k_proj = nn.Linear(hidden_size, k_size, bias=bias)

            self.q1_proj = None
            self.q2_proj = None
            self.k1_proj = None
            self.k2_proj = None

        if quant_type == "fp4":
            self.v_proj = MarlinLinear(
                hidden_size, v_size, bias=bias,
                quant_type="fp4", group_size=group_size,
            )
            self.o_proj = MarlinLinear(
                o_size, hidden_size, bias=bias,
                quant_type="fp4", group_size=group_size,
            )
        else:
            require_mlx("FP16 differential attention projections")
            self.v_proj = nn.Linear(hidden_size, v_size, bias=bias)
            self.o_proj = nn.Linear(o_size, hidden_size, bias=bias)

        # Differential attention core
        self.diff_attn = DifferentialAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            lambda_init=lambda_init,
            lambda_learnable=lambda_learnable,
            lambda_per_head=lambda_per_head,
            sublayer_norm=sublayer_norm,
            use_fused_kernel=use_fused_kernel,
        )

        # RoPE embeddings
        self.rope = RoPE(self.head_dim, base=rope_theta)

    def __call__(
        self,
        hidden_states: Any,  # [batch, seq_len, hidden_size]
        attention_mask: Any | None = None,
        position_ids: Any | None = None,
        kv_cache: Any | None = None,
        layer_idx: int = 0,
        attention_mask_is_causal: bool = False,
    ) -> Any:
        """Forward pass.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Optional position IDs for RoPE
            kv_cache: Optional KV cache for autoregressive generation
            layer_idx: Layer index for KV cache

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        if HAS_MLX:
            return self._forward_mlx(
                hidden_states,
                attention_mask,
                position_ids,
                kv_cache,
                layer_idx,
                attention_mask_is_causal,
            )
        else:
            if attention_mask is None and attention_mask_is_causal:
                seq_len = hidden_states.shape[1]
                attention_mask = create_causal_mask(seq_len, seq_len)
            return self._forward_numpy(
                hidden_states,
                attention_mask,
                position_ids,
                kv_cache,
                layer_idx,
            )

    def _forward_mlx(
        self,
        hidden_states: Any,
        attention_mask: Any | None,
        position_ids: Any | None,
        kv_cache: Any,
        layer_idx: int,
        attention_mask_is_causal: bool,
    ) -> Any:
        """MLX forward pass."""
        batch_size, seq_len, _ = hidden_states.shape

        v = self.v_proj(hidden_states)  # [batch, seq, num_kv_heads * head_dim]
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.transpose(0, 2, 1, 3)

        if self.split_qk_projections:
            q1 = self.q1_proj(hidden_states)
            q2 = self.q2_proj(hidden_states)
            k1 = self.k1_proj(hidden_states)
            k2 = self.k2_proj(hidden_states)

            q1 = q1.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            q2 = q2.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            q1 = q1.transpose(0, 2, 1, 3)
            q2 = q2.transpose(0, 2, 1, 3)

            k1 = k1.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            k2 = k2.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            k1 = k1.transpose(0, 2, 1, 3)
            k2 = k2.transpose(0, 2, 1, 3)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)

            q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim * 2)
            q1 = q[..., :self.head_dim]
            q2 = q[..., self.head_dim:]
            q1 = q1.transpose(0, 2, 1, 3)
            q2 = q2.transpose(0, 2, 1, 3)

            k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim * 2)
            k1 = k[..., :self.head_dim]
            k2 = k[..., self.head_dim:]
            k1 = k1.transpose(0, 2, 1, 3)
            k2 = k2.transpose(0, 2, 1, 3)

        # Apply RoPE to Q1, Q2, K1, K2
        position_offset = kv_cache.seq_len if kv_cache else 0
        q1 = self.rope(q1, offset=position_offset)
        q2 = self.rope(q2, offset=position_offset)
        k1 = self.rope(k1, offset=position_offset)
        k2 = self.rope(k2, offset=position_offset)

        # Update KV cache if provided
        if kv_cache is not None:
            # Cache stores [k1, k2] concatenated and v
            k_combined = mx.concatenate([k1, k2], axis=-1)  # [batch, kv_heads, seq, head_dim*2]
            k_combined, v = kv_cache.update(layer_idx, k_combined, v)
            # Split back
            k1 = k_combined[..., :self.head_dim]
            k2 = k_combined[..., self.head_dim:]

        # Compute differential attention
        attn_output = self.diff_attn(
            q1,
            k1,
            v,
            q2,
            k2,
            attention_mask,
            attention_mask_is_causal,
        )

        # Reshape and project output
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        output = self.o_proj(attn_output)

        return output

    def _forward_numpy(
        self,
        hidden_states: Any,
        attention_mask: Any | None,
        position_ids: Any | None,
        kv_cache: Any,
        layer_idx: int,
    ) -> Any:
        """NumPy fallback forward pass."""
        hidden_np = to_numpy(hidden_states)
        batch_size, seq_len, _ = hidden_np.shape

        v = self.v_proj(hidden_states)

        if self.split_qk_projections:
            q1 = to_numpy(self.q1_proj(hidden_states))
            q2 = to_numpy(self.q2_proj(hidden_states))
            k1 = to_numpy(self.k1_proj(hidden_states))
            k2 = to_numpy(self.k2_proj(hidden_states))

            q1 = q1.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            q2 = q2.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            q1 = q1.transpose(0, 2, 1, 3)
            q2 = q2.transpose(0, 2, 1, 3)

            k1 = k1.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            k2 = k2.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            k1 = k1.transpose(0, 2, 1, 3)
            k2 = k2.transpose(0, 2, 1, 3)
        else:
            q = to_numpy(self.q_proj(hidden_states))
            k = to_numpy(self.k_proj(hidden_states))

            q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim * 2)
            q1 = q[..., :self.head_dim]
            q2 = q[..., self.head_dim:]
            q1 = q1.transpose(0, 2, 1, 3)
            q2 = q2.transpose(0, 2, 1, 3)

            k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim * 2)
            k1 = k[..., :self.head_dim]
            k2 = k[..., self.head_dim:]
            k1 = k1.transpose(0, 2, 1, 3)
            k2 = k2.transpose(0, 2, 1, 3)

        v = to_numpy(v)

        # V is not split
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.transpose(0, 2, 1, 3)

        # Apply RoPE
        position_offset = 0  # KV cache not supported in numpy fallback
        q1 = self.rope._apply_numpy(q1, offset=position_offset)
        q2 = self.rope._apply_numpy(q2, offset=position_offset)
        k1 = self.rope._apply_numpy(k1, offset=position_offset)
        k2 = self.rope._apply_numpy(k2, offset=position_offset)

        # Compute differential attention
        mask_np = to_numpy(attention_mask) if attention_mask is not None else None
        attn_output = self.diff_attn._forward_numpy(q1, k1, v, q2, k2, mask_np)

        # Reshape and project output
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )

        if HAS_MLX:
            attn_output = from_numpy(attn_output, backend="mlx")

        output = self.o_proj(attn_output)

        return output

    @classmethod
    def from_config(
        cls,
        config: DifferentialAttentionConfig,
        quant_type: Literal["fp4", "fp16"] = "fp4",
        group_size: int = 128,
    ) -> DifferentialMarlinAttention:
        """Create layer from DifferentialAttentionConfig.

        Args:
            config: Attention configuration.
            quant_type: Quantization type for projections.
            group_size: Quantization group size.

        Returns:
            Configured DifferentialMarlinAttention layer.
        """
        return cls(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            lambda_init=config.lambda_init,
            lambda_learnable=config.lambda_learnable,
            lambda_per_head=config.lambda_per_head,
            sublayer_norm=config.sublayer_norm,
            quant_type=quant_type,
            q1k1_quant_type=config.q1k1_quant_type,
            q2k2_quant_type=config.q2k2_quant_type,
            group_size=group_size,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.use_bias,
            use_fused_kernel=config.use_fused_kernel,
        )


def create_causal_mask(seq_len: int, kv_seq_len: int | None = None) -> Any:
    """Create causal attention mask.

    Args:
        seq_len: Query sequence length
        kv_seq_len: Key/value sequence length (defaults to seq_len)

    Returns:
        Causal mask with -inf for masked positions
    """
    kv_seq_len = kv_seq_len or seq_len

    if seq_len == 1:
        # Single token decode - no masking needed
        return None

    if HAS_MLX and mx is not None:
        mask = mx.triu(mx.full((seq_len, kv_seq_len), float("-inf")), k=1)
        return mask[None, None, :, :]
    else:
        mask = np.triu(np.full((seq_len, kv_seq_len), float("-inf")), k=1)
        return mask[None, None, :, :]


# Make classes inherit from nn.Module when MLX is available
if HAS_MLX and nn is not None:
    _OriginalDiffAttn = DifferentialAttention
    _OriginalDiffMarlinAttn = DifferentialMarlinAttention

    class DifferentialAttention(nn.Module):  # type: ignore[no-redef]
        """Differential attention computation with learnable lambda."""

        __doc__ = _OriginalDiffAttn.__doc__

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            _OriginalDiffAttn.__init__(self, *args, **kwargs)

        __call__ = _OriginalDiffAttn.__call__
        _forward_mlx = _OriginalDiffAttn._forward_mlx
        _forward_numpy = _OriginalDiffAttn._forward_numpy
        get_lambda = _OriginalDiffAttn.get_lambda

    class DifferentialMarlinAttention(nn.Module):  # type: ignore[no-redef]
        """Full differential attention layer with Marlin-quantized projections."""

        __doc__ = _OriginalDiffMarlinAttn.__doc__

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            _OriginalDiffMarlinAttn.__init__(self, *args, **kwargs)

        __call__ = _OriginalDiffMarlinAttn.__call__
        _forward_mlx = _OriginalDiffMarlinAttn._forward_mlx
        _forward_numpy = _OriginalDiffMarlinAttn._forward_numpy
        from_config = _OriginalDiffMarlinAttn.from_config
