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
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attention import RoPE

if TYPE_CHECKING:
    from ..kv_cache import KVCache


def _get_device() -> torch.device:
    """Get the appropriate device (MPS on Apple Silicon, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        use_fused_kernel: Enable fused differential attention kernel (when available).
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

    def __post_init__(self) -> None:
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
        "num_key_value_heads", config_dict.get("num_kv_heads", num_heads)
    )
    head_dim = config_dict.get("head_dim", config_dict.get("kv_channels", hidden_size // num_heads))

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


class DifferentialAttention(nn.Module):
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
        use_fused_kernel: Enable fused attention (uses PyTorch SDPA). Default: True.
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
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.sublayer_norm = sublayer_norm
        self.lambda_per_head = lambda_per_head
        self.use_fused_kernel = use_fused_kernel
        self.lambda_learnable = lambda_learnable

        # Lambda parameter
        # Shape: [num_heads] if per_head, else [1]
        lambda_shape = (num_heads,) if lambda_per_head else (1,)

        if lambda_learnable:
            # Learnable parameter stored as log to ensure positivity
            # lambda = exp(lambda_log) where lambda_log is initialized to log(lambda_init)
            init_val = math.log(lambda_init) if lambda_init > 0 else -2.0
            self.lambda_log = nn.Parameter(torch.full(lambda_shape, init_val, dtype=torch.float32))
        else:
            # Fixed lambda (register as buffer, not parameter)
            self.register_buffer(
                "lambda_fixed", torch.full(lambda_shape, lambda_init, dtype=torch.float32)
            )
            self.lambda_log = None

        if sublayer_norm:
            # LayerNorm after differential attention
            self.sublayer_ln = nn.LayerNorm(head_dim)
        else:
            self.sublayer_ln = None

    def get_lambda(self) -> torch.Tensor:
        """Get current lambda value(s)."""
        if self.lambda_learnable and self.lambda_log is not None:
            return torch.exp(self.lambda_log)
        else:
            return self.lambda_fixed

    def forward(
        self,
        q1: torch.Tensor,  # [batch, num_heads, seq_q, head_dim]
        k1: torch.Tensor,  # [batch, num_kv_heads, seq_k, head_dim]
        v: torch.Tensor,  # [batch, num_kv_heads, seq_k, head_dim]
        q2: torch.Tensor,  # [batch, num_heads, seq_q, head_dim]
        k2: torch.Tensor,  # [batch, num_kv_heads, seq_k, head_dim]
        attention_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Compute differential attention.

        Args:
            q1, k1: First attention sub-head Q/K (positive component)
            v: Value tensor (shared between both attention computations)
            q2, k2: Second attention sub-head Q/K (negative/baseline component)
            attention_mask: Optional causal mask with -inf for masked positions
            is_causal: If True, uses causal masking (more efficient than explicit mask)

        Returns:
            Differential attention output [batch, num_heads, seq_q, head_dim]
        """
        batch_size, num_heads, seq_q, head_dim = q1.shape
        num_kv_heads = k1.shape[1]
        device = q1.device
        dtype = q1.dtype

        # Handle GQA: repeat K/V heads if needed
        if num_kv_heads < num_heads:
            repeat_factor = num_heads // num_kv_heads
            k1 = k1.repeat_interleave(repeat_factor, dim=1)
            k2 = k2.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        if self.use_fused_kernel and (attention_mask is None or is_causal):
            # Use PyTorch's optimized SDPA for both attention computations
            attn1 = F.scaled_dot_product_attention(
                q1,
                k1,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=self.scale,
            )
            attn2 = F.scaled_dot_product_attention(
                q2,
                k2,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=self.scale,
            )
        else:
            # Manual attention computation with explicit mask
            # Compute attention scores for both sub-heads
            scores1 = (q1 @ k1.transpose(-2, -1)) * self.scale
            scores2 = (q2 @ k2.transpose(-2, -1)) * self.scale

            # Apply attention mask (causal mask with -inf)
            if attention_mask is not None:
                scores1 = scores1 + attention_mask
                scores2 = scores2 + attention_mask

            # Softmax
            weights1 = F.softmax(scores1, dim=-1)
            weights2 = F.softmax(scores2, dim=-1)

            # Compute weighted values
            attn1 = weights1 @ v
            attn2 = weights2 @ v

        # Differential attention: output = attn1 - lambda * attn2
        lambda_val = self.get_lambda().to(dtype=dtype, device=device)
        if self.lambda_per_head:
            # lambda_val: [num_heads] -> [1, num_heads, 1, 1]
            lambda_val = lambda_val.view(1, -1, 1, 1)

        output = attn1 - lambda_val * attn2

        # Optional sublayer norm
        if self.sublayer_norm and self.sublayer_ln is not None:
            # Apply per head dim: [batch, heads, seq, head_dim]
            orig_shape = output.shape
            output = output.reshape(-1, head_dim)
            output = self.sublayer_ln(output)
            output = output.reshape(orig_shape)

        return output


class DifferentialMarlinAttention(nn.Module):
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
        use_fused_kernel: Enable fused differential attention kernel. Default: True.
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
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.quant_type = quant_type
        self.q1k1_quant_type = q1k1_quant_type
        self.q2k2_quant_type = q2k2_quant_type
        self.split_qk_projections = q1k1_quant_type is not None or q2k2_quant_type is not None

        def _build_proj(
            in_features: int,
            out_features: int,
            proj_quant_type: Literal["fp4", "fp16"],
        ) -> nn.Module:
            # Always use nn.Linear for initialization.
            # For quantized inference, use MarlinLinear.from_linear() to convert
            # the layers after loading weights, or load pre-quantized checkpoints.
            # This matches the pattern where models are trained in FP16 and
            # quantized post-training.
            return nn.Linear(in_features, out_features, bias=bias)

        v_size = self.num_kv_heads * self.head_dim
        o_size = num_heads * self.head_dim

        if self.split_qk_projections:
            q1_type = q1k1_quant_type or quant_type
            q2_type = q2k2_quant_type or quant_type
            k1_type = q1k1_quant_type or quant_type
            k2_type = q2k2_quant_type or quant_type

            self.q_proj = None
            self.k_proj = None
            self.q1_proj = _build_proj(hidden_size, num_heads * self.head_dim, q1_type)
            self.q2_proj = _build_proj(hidden_size, num_heads * self.head_dim, q2_type)
            self.k1_proj = _build_proj(hidden_size, self.num_kv_heads * self.head_dim, k1_type)
            self.k2_proj = _build_proj(hidden_size, self.num_kv_heads * self.head_dim, k2_type)
        else:
            q_size = num_heads * self.head_dim * 2
            k_size = self.num_kv_heads * self.head_dim * 2

            self.q_proj = _build_proj(hidden_size, q_size, quant_type)
            self.k_proj = _build_proj(hidden_size, k_size, quant_type)

            self.q1_proj = None
            self.q2_proj = None
            self.k1_proj = None
            self.k2_proj = None

        self.v_proj = _build_proj(hidden_size, v_size, quant_type)
        self.o_proj = _build_proj(o_size, hidden_size, quant_type)

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

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, hidden_size]
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Optional position IDs for RoPE
            kv_cache: Optional KV cache for autoregressive generation
            layer_idx: Layer index for KV cache
            is_causal: If True, uses causal masking

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # V projection (shared between both attention paths)
        v = self.v_proj(hidden_states)  # [batch, seq, num_kv_heads * head_dim]
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.split_qk_projections:
            # Separate projections for Q1/K1 and Q2/K2
            q1 = self.q1_proj(hidden_states)
            q2 = self.q2_proj(hidden_states)
            k1 = self.k1_proj(hidden_states)
            k2 = self.k2_proj(hidden_states)

            q1 = q1.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            q2 = q2.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k1 = k1.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            k2 = k2.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        else:
            # Combined projections, split after
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)

            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim * 2)
            q1 = q[..., : self.head_dim].transpose(1, 2)
            q2 = q[..., self.head_dim :].transpose(1, 2)

            k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim * 2)
            k1 = k[..., : self.head_dim].transpose(1, 2)
            k2 = k[..., self.head_dim :].transpose(1, 2)

        # Apply RoPE to Q1, Q2, K1, K2
        position_offset = kv_cache.seq_len if kv_cache else 0
        q1 = self.rope(q1, offset=position_offset)
        q2 = self.rope(q2, offset=position_offset)
        k1 = self.rope(k1, offset=position_offset)
        k2 = self.rope(k2, offset=position_offset)

        # Update KV cache if provided
        if kv_cache is not None:
            # Cache stores [k1, k2] concatenated and v
            k_combined = torch.cat([k1, k2], dim=-1)  # [batch, kv_heads, seq, head_dim*2]
            k_combined, v = kv_cache.update(layer_idx, k_combined, v)
            # Split back
            k1 = k_combined[..., : self.head_dim]
            k2 = k_combined[..., self.head_dim :]

        # Compute differential attention
        attn_output = self.diff_attn(
            q1,
            k1,
            v,
            q2,
            k2,
            attention_mask=attention_mask,
            is_causal=is_causal,
        )

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.head_dim)
        )
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


def create_causal_mask(
    seq_len: int,
    kv_seq_len: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    """Create causal attention mask.

    Args:
        seq_len: Query sequence length
        kv_seq_len: Key/value sequence length (defaults to seq_len)
        device: Device to create the mask on

    Returns:
        Causal mask with -inf for masked positions, or None for single-token decode
    """
    kv_seq_len = kv_seq_len or seq_len

    if seq_len == 1:
        # Single token decode - no masking needed
        return None

    if device is None:
        device = _get_device()

    mask = torch.triu(
        torch.full((seq_len, kv_seq_len), float("-inf"), device=device),
        diagonal=1,
    )
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, kv_seq_len]
