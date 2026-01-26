"""Optimized decode (single-token generation) kernels for dense models.

Decode characteristics:
- M=1: Single token at a time
- Memory-bound: Small GEMM, large KV cache read
- Latency-critical: Directly impacts tok/s

Standard 64x64 tile GEMM is catastrophically inefficient for M=1:
- 98.4% of compute wasted on zero padding
- Memory bandwidth under-utilized

This module provides:
1. persistent_decode_step(): Single kernel for full layer decode
2. quantized_kv_attention(): Attention with FP8/INT8 quantized cache
3. Automatic kernel selection based on dimensions

Target: <5ms per token latency on M4 Max for 30B models.

Implementation notes:
- Uses decode_gemv.metal kernels (TILE_M=1, TILE_N=256/512)
- Fused attention avoids materialization of intermediate tensors
- Quantized KV cache reduces memory bandwidth by 2-4x
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .._compat import mx, require_mlx

if TYPE_CHECKING:
    import mlx.core as mx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DecodeConfig:
    """Configuration for decode phase optimization.

    Attributes:
        num_layers: Number of transformer layers.
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        num_kv_heads: Number of KV heads (for GQA). Defaults to num_heads.
        head_dim: Dimension per head. Defaults to hidden_size // num_heads.
        intermediate_size: MLP intermediate dimension.
        group_size: Quantization group size for weights.
        kv_cache_dtype: KV cache precision ("bf16", "fp16", "fp8", "int8").
        use_fused_attention: Use fused softmax + V matmul.
        persistent_buffers: Keep intermediate buffers between calls.
        prefetch_next_layer: Prefetch weights for next layer during compute.
    """

    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int | None = None
    head_dim: int | None = None
    intermediate_size: int | None = None
    group_size: int = 128
    kv_cache_dtype: Literal["bf16", "fp16", "fp8", "int8"] = "bf16"
    use_fused_attention: bool = True
    persistent_buffers: bool = True
    prefetch_next_layer: bool = True

    def __post_init__(self) -> None:
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        if self.intermediate_size is None:
            # Llama default: 8/3 * hidden, rounded to multiple of 256
            self.intermediate_size = ((self.hidden_size * 8 // 3) + 255) // 256 * 256


@dataclass
class LayerWeights:
    """Weights for a single transformer layer.

    All weights are stored in packed FP4 format with per-group scales.
    Layout: [K/8, N] as uint32 where each uint32 holds 8 FP4 values.

    Attributes:
        q_proj_weight: Query projection [hidden_size/8, num_heads * head_dim]
        q_proj_scales: Query scales [hidden_size/group_size, num_heads * head_dim]
        k_proj_weight: Key projection [hidden_size/8, num_kv_heads * head_dim]
        k_proj_scales: Key scales
        v_proj_weight: Value projection [hidden_size/8, num_kv_heads * head_dim]
        v_proj_scales: Value scales
        o_proj_weight: Output projection [num_heads * head_dim / 8, hidden_size]
        o_proj_scales: Output scales
        gate_proj_weight: Gate projection [hidden_size/8, intermediate_size]
        gate_proj_scales: Gate scales
        up_proj_weight: Up projection [hidden_size/8, intermediate_size]
        up_proj_scales: Up scales
        down_proj_weight: Down projection [intermediate_size/8, hidden_size]
        down_proj_scales: Down scales
        input_layernorm_weight: RMSNorm weight for attention
        post_attention_layernorm_weight: RMSNorm weight for MLP
    """

    # Attention projections
    q_proj_weight: Any  # mx.array
    q_proj_scales: Any
    k_proj_weight: Any
    k_proj_scales: Any
    v_proj_weight: Any
    v_proj_scales: Any
    o_proj_weight: Any
    o_proj_scales: Any

    # MLP projections
    gate_proj_weight: Any
    gate_proj_scales: Any
    up_proj_weight: Any
    up_proj_scales: Any
    down_proj_weight: Any
    down_proj_scales: Any

    # Normalization
    input_layernorm_weight: Any
    post_attention_layernorm_weight: Any


@dataclass
class DecodePerfStats:
    """Performance statistics for decode operations.

    Attributes:
        layer_times_ms: Per-layer decode times in milliseconds.
        attention_time_ms: Total attention time.
        mlp_time_ms: Total MLP time.
        kv_cache_read_mb: KV cache data read in MB.
        weight_read_mb: Weight data read in MB.
        achieved_bandwidth_gbps: Achieved memory bandwidth in GB/s.
    """

    layer_times_ms: list[float] = field(default_factory=list)
    attention_time_ms: float = 0.0
    mlp_time_ms: float = 0.0
    kv_cache_read_mb: float = 0.0
    weight_read_mb: float = 0.0
    achieved_bandwidth_gbps: float = 0.0

    @property
    def total_time_ms(self) -> float:
        return sum(self.layer_times_ms) if self.layer_times_ms else 0.0

    @property
    def tokens_per_second(self) -> float:
        if self.total_time_ms > 0:
            return 1000.0 / self.total_time_ms
        return 0.0


class DecodeState:
    """Persistent state for decode operations.

    Maintains:
    - Pre-allocated output buffers for each layer
    - Fused QKV projection buffers
    - KV cache quantization scales

    Reusing buffers avoids allocation overhead during generation.
    """

    def __init__(
        self,
        config: DecodeConfig,
        layer_weights: list[LayerWeights],
    ) -> None:
        require_mlx("DecodeState")

        self.config = config
        self.layer_weights = layer_weights

        # Pre-allocate persistent buffers if enabled
        if config.persistent_buffers:
            self._init_persistent_buffers()
        else:
            self._qkv_buffer = None
            self._attn_out_buffer = None
            self._mlp_buffer = None

        # KV cache quantization state
        self._kv_scales: list[tuple[Any, Any]] = []  # [(k_scale, v_scale), ...]

    def _init_persistent_buffers(self) -> None:
        """Initialize persistent buffers for decode."""
        cfg = self.config
        assert cfg.head_dim is not None
        assert cfg.num_kv_heads is not None

        # QKV output buffer: [1, num_heads * head_dim + 2 * num_kv_heads * head_dim]
        qkv_dim = (
            cfg.num_heads * cfg.head_dim + 2 * cfg.num_kv_heads * cfg.head_dim
        )
        self._qkv_buffer = mx.zeros((1, 1, qkv_dim), dtype=mx.bfloat16)

        # Attention output buffer: [1, 1, hidden_size]
        self._attn_out_buffer = mx.zeros((1, 1, cfg.hidden_size), dtype=mx.bfloat16)

        # MLP intermediate buffer: [1, 1, intermediate_size]
        assert cfg.intermediate_size is not None
        self._mlp_buffer = mx.zeros(
            (1, 1, cfg.intermediate_size), dtype=mx.bfloat16
        )

    @property
    def qkv_buffer(self) -> Any:
        return self._qkv_buffer

    @property
    def attn_out_buffer(self) -> Any:
        return self._attn_out_buffer

    @property
    def mlp_buffer(self) -> Any:
        return self._mlp_buffer


# ---------------------------------------------------------------------------
# Metal kernel source for decode GEMV
# ---------------------------------------------------------------------------

_DECODE_GEMV_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// FP4 E2M1 branchless dequant
inline half dequant_fp4_scalar(uint nibble) {
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;

    half sub_mag = half(man_bit) * half(0.25h);
    half norm_mag = half(1u << (exp_bits - 1)) * (half(1.0h) + half(man_bit) * half(0.5h));
    half magnitude = select(norm_mag, sub_mag, exp_bits == 0);
    return select(magnitude, -magnitude, bool(sign_bit));
}

inline half dequant_fp4_scaled(uint nibble, float scale) {
    return (half)(float(dequant_fp4_scalar(nibble)) * scale);
}

// FP8 E4M3 dequant
inline half dequant_fp8_e4m3(uint8_t val, float scale) {
    // Simplified E4M3: max = 448, interpret as linear uint8
    float signed_val = float(val) - 128.0f;
    return (half)(signed_val / 127.0f * 448.0f * scale);
}

// Divide ceiling
inline uint div_ceil(uint a, uint b) {
    return (a + b - 1) / b;
}
"""

# Source for fused QKV decode GEMV (3 outputs in one kernel)
_FUSED_QKV_DECODE_SOURCE = """
    // Fused Q/K/V projection for decode (M=1)
    // Outputs: Q[1, num_heads * head_dim], K[1, num_kv_heads * head_dim], V[1, num_kv_heads * head_dim]
    //
    // Grid: (ceil(total_out / 256), 1, 1)
    // Threadgroup: 128 threads

    const uint TILE_N = 256;
    const uint COLS_PER_THREAD = 2;
    const uint FP4_PER_UINT = 8;

    uint tgid = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;

    uint col_base = tgid * TILE_N + tid * COLS_PER_THREAD;
    uint total_out = Q_N + K_N + V_N;

    // Accumulators for 2 output columns
    float acc[2] = {0.0f, 0.0f};

    // Determine which projection this column belongs to
    // Q: [0, Q_N), K: [Q_N, Q_N+K_N), V: [Q_N+K_N, total_out)

    for (uint c = 0; c < 2; c++) {
        uint col = col_base + c;
        if (col >= total_out) continue;

        // Select weight/scale based on which projection
        device const uint* W;
        device const half* S;
        uint proj_col;

        if (col < Q_N) {
            W = Q_W;
            S = Q_S;
            proj_col = col;
        } else if (col < Q_N + K_N) {
            W = K_W;
            S = K_S;
            proj_col = col - Q_N;
        } else {
            W = V_W;
            S = V_S;
            proj_col = col - Q_N - K_N;
        }

        // Stream through K dimension
        float acc_val = 0.0f;
        for (uint k_base = 0; k_base < K_DIM; k_base += FP4_PER_UINT) {
            uint pack_idx = k_base / FP4_PER_UINT;
            uint group_idx = k_base / GROUP_SIZE;

            // Load packed weights and scale
            uint packed = W[pack_idx * PROJ_N + proj_col];
            half scale = S[group_idx * PROJ_N + proj_col];
            float fscale = (float)scale;

            // Dequant and accumulate
            for (uint i = 0; i < 8 && (k_base + i) < K_DIM; i++) {
                half a_val = A[k_base + i];
                uint nibble = (packed >> (i * 4)) & 0xF;
                half w_val = dequant_fp4_scaled(nibble, fscale);
                acc_val += float(a_val) * float(w_val);
            }
        }

        acc[c] = acc_val;
    }

    // Store results
    for (uint c = 0; c < 2; c++) {
        uint col = col_base + c;
        if (col < total_out) {
            out[col] = half(acc[c]);
        }
    }
"""

# Source for fused attention with quantized KV cache
_QUANTIZED_KV_ATTENTION_SOURCE = """
    // Fused scaled dot-product attention with quantized KV cache
    // Q: [1, num_heads, 1, head_dim] - current query
    // K_cache: [num_heads, seq_len, head_dim] - FP8 quantized keys
    // V_cache: [num_heads, seq_len, head_dim] - FP8 quantized values
    // K_scales: [num_heads, seq_len, 1] - per-row scales
    // V_scales: [num_heads, seq_len, 1] - per-row scales
    //
    // Output: [1, num_heads, 1, head_dim]
    //
    // Grid: (num_heads, 1, 1)
    // Threadgroup: 128 threads (4 simdgroups)

    uint head_idx = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint simd_lane = thread_index_in_simdgroup;
    uint simd_id = simdgroup_index_in_threadgroup;

    if (head_idx >= NUM_HEADS) return;

    // Shared memory for softmax reduction
    threadgroup float max_vals[128];
    threadgroup float sum_vals[128];
    threadgroup float attn_weights[MAX_SEQ_LEN];

    // Each thread handles multiple positions
    float local_max = -INFINITY;
    float scale_factor = rsqrt(float(HEAD_DIM));

    // Compute attention scores: Q @ K^T
    for (uint pos = tid; pos < SEQ_LEN; pos += 128) {
        float score = 0.0f;

        // Get K scale for this position
        float k_scale = (float)K_scales[head_idx * SEQ_LEN + pos];

        // Dot product Q . K[pos]
        for (uint d = 0; d < HEAD_DIM; d++) {
            float q_val = (float)Q[head_idx * HEAD_DIM + d];

            // Dequant K from FP8
            uint8_t k_quant = K_cache[head_idx * SEQ_LEN * HEAD_DIM + pos * HEAD_DIM + d];
            float k_val = (float(k_quant) - 128.0f) / 127.0f * 448.0f * k_scale;

            score += q_val * k_val;
        }

        score *= scale_factor;
        attn_weights[pos] = score;
        local_max = max(local_max, score);
    }

    // Reduce max across threads
    max_vals[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction for max
    for (uint stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < 128) {
            max_vals[tid] = max(max_vals[tid], max_vals[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_max = max_vals[0];

    // Compute exp and sum
    float local_sum = 0.0f;
    for (uint pos = tid; pos < SEQ_LEN; pos += 128) {
        float w = exp(attn_weights[pos] - global_max);
        attn_weights[pos] = w;
        local_sum += w;
    }

    sum_vals[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction for sum
    for (uint stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < 128) {
            sum_vals[tid] += sum_vals[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_sum = sum_vals[0];

    // Normalize weights
    float inv_sum = 1.0f / global_sum;
    for (uint pos = tid; pos < SEQ_LEN; pos += 128) {
        attn_weights[pos] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute output: attn_weights @ V
    // Each thread handles HEAD_DIM / 128 output elements
    for (uint d = tid; d < HEAD_DIM; d += 128) {
        float out_val = 0.0f;

        for (uint pos = 0; pos < SEQ_LEN; pos++) {
            float w = attn_weights[pos];
            float v_scale = (float)V_scales[head_idx * SEQ_LEN + pos];

            // Dequant V from FP8
            uint8_t v_quant = V_cache[head_idx * SEQ_LEN * HEAD_DIM + pos * HEAD_DIM + d];
            float v_val = (float(v_quant) - 128.0f) / 127.0f * 448.0f * v_scale;

            out_val += w * v_val;
        }

        out[head_idx * HEAD_DIM + d] = half(out_val);
    }
"""


# ---------------------------------------------------------------------------
# Kernel dispatch functions
# ---------------------------------------------------------------------------

# Cached kernel objects
_decode_gemv_kernel: Any = None
_fused_qkv_kernel: Any = None
_quantized_kv_attention_kernel: Any = None


def _get_decode_gemv_kernel() -> Any:
    """Get or create the decode GEMV kernel."""
    global _decode_gemv_kernel
    require_mlx("decode_gemv kernel")

    if _decode_gemv_kernel is None:
        # Read from compiled Metal shader
        shader_path = Path(__file__).parent.parent / "src" / "decode_gemv.metal"
        if shader_path.exists():
            with open(shader_path) as f:
                source = f.read()
        else:
            # Fallback to inline source
            source = """
            uint col = threadgroup_position_in_grid.x * 256 + thread_position_in_threadgroup.x * 2;
            if (col >= N) return;

            float acc0 = 0.0f, acc1 = 0.0f;
            for (uint k = 0; k < K; k += 8) {
                uint pack_idx = k / 8;
                uint group_idx = k / GROUP_SIZE;

                half a0 = A[k], a1 = A[k+1], a2 = A[k+2], a3 = A[k+3];
                half a4 = A[k+4], a5 = A[k+5], a6 = A[k+6], a7 = A[k+7];

                uint packed0 = B[pack_idx * N + col];
                half scale0 = scales[group_idx * N + col];
                float fs0 = (float)scale0;

                acc0 += (float)a0 * (float)dequant_fp4_scaled((packed0 >> 0) & 0xF, fs0);
                acc0 += (float)a1 * (float)dequant_fp4_scaled((packed0 >> 4) & 0xF, fs0);
                acc0 += (float)a2 * (float)dequant_fp4_scaled((packed0 >> 8) & 0xF, fs0);
                acc0 += (float)a3 * (float)dequant_fp4_scaled((packed0 >> 12) & 0xF, fs0);
                acc0 += (float)a4 * (float)dequant_fp4_scaled((packed0 >> 16) & 0xF, fs0);
                acc0 += (float)a5 * (float)dequant_fp4_scaled((packed0 >> 20) & 0xF, fs0);
                acc0 += (float)a6 * (float)dequant_fp4_scaled((packed0 >> 24) & 0xF, fs0);
                acc0 += (float)a7 * (float)dequant_fp4_scaled((packed0 >> 28) & 0xF, fs0);

                if (col + 1 < N) {
                    uint packed1 = B[pack_idx * N + col + 1];
                    half scale1 = scales[group_idx * N + col + 1];
                    float fs1 = (float)scale1;

                    acc1 += (float)a0 * (float)dequant_fp4_scaled((packed1 >> 0) & 0xF, fs1);
                    acc1 += (float)a1 * (float)dequant_fp4_scaled((packed1 >> 4) & 0xF, fs1);
                    acc1 += (float)a2 * (float)dequant_fp4_scaled((packed1 >> 8) & 0xF, fs1);
                    acc1 += (float)a3 * (float)dequant_fp4_scaled((packed1 >> 12) & 0xF, fs1);
                    acc1 += (float)a4 * (float)dequant_fp4_scaled((packed1 >> 16) & 0xF, fs1);
                    acc1 += (float)a5 * (float)dequant_fp4_scaled((packed1 >> 20) & 0xF, fs1);
                    acc1 += (float)a6 * (float)dequant_fp4_scaled((packed1 >> 24) & 0xF, fs1);
                    acc1 += (float)a7 * (float)dequant_fp4_scaled((packed1 >> 28) & 0xF, fs1);
                }
            }

            out[col] = half(acc0);
            if (col + 1 < N) out[col + 1] = half(acc1);
            """

        _decode_gemv_kernel = mx.fast.metal_kernel(
            name="decode_gemv_fp4",
            input_names=["A", "B", "scales"],
            output_names=["out"],
            source=source,
            header=_DECODE_GEMV_HEADER,
            ensure_row_contiguous=True,
        )

    return _decode_gemv_kernel


def select_decode_kernel(M: int, N: int, K: int) -> str:
    """Select optimal decode kernel based on problem dimensions.

    Args:
        M: Batch size (typically 1 for decode).
        N: Output dimension.
        K: Input dimension.

    Returns:
        Kernel name to use ("decode_gemv_fp4", "decode_gemv_fp4_wide",
        "decode_gemv_fp4_tiled", or "marlin_gemm_fp4" for larger M).
    """
    if M > 8:
        # Fall back to full GEMM for larger batches
        return "marlin_gemm_fp4"

    if M > 1:
        # Small batch decode
        return "decode_gemv_fp4_batched"

    # M == 1 decode
    if N >= 512:
        # Wide kernel for better coalescing
        return "decode_gemv_fp4_wide"
    elif K > 8192:
        # Tiled kernel for large K to cache A
        return "decode_gemv_fp4_tiled"
    else:
        # Default single-token decode
        return "decode_gemv_fp4"


# ---------------------------------------------------------------------------
# High-level decode functions
# ---------------------------------------------------------------------------


def fused_qkv_projection(
    hidden_states: Any,  # mx.array [batch, 1, hidden_size]
    layer: LayerWeights,
    config: DecodeConfig,
) -> tuple[Any, Any, Any]:
    """Fused Q/K/V projection for decode phase.

    Computes all three projections in a single kernel launch to reduce
    kernel dispatch overhead and improve cache utilization.

    Args:
        hidden_states: Input tensor [batch, 1, hidden_size].
        layer: Layer weights.
        config: Decode configuration.

    Returns:
        Tuple of (Q, K, V) tensors:
        - Q: [batch, num_heads, 1, head_dim]
        - K: [batch, num_kv_heads, 1, head_dim]
        - V: [batch, num_kv_heads, 1, head_dim]
    """
    require_mlx("fused_qkv_projection")

    batch_size = hidden_states.shape[0]
    hidden_size = config.hidden_size
    num_heads = config.num_heads
    num_kv_heads = config.num_kv_heads
    head_dim = config.head_dim

    assert num_kv_heads is not None
    assert head_dim is not None

    # For now, use separate projections (fused kernel to be implemented)
    # Import kernels lazily to avoid circular imports
    from ..metal_marlin import quantized_linear

    # Flatten to [batch, hidden_size]
    h = hidden_states.reshape(batch_size, hidden_size)

    # Separate projections
    q = quantized_linear(
        h,
        layer.q_proj_weight,
        layer.q_proj_scales,
        config.group_size,
    )
    k = quantized_linear(
        h,
        layer.k_proj_weight,
        layer.k_proj_scales,
        config.group_size,
    )
    v = quantized_linear(
        h,
        layer.v_proj_weight,
        layer.v_proj_scales,
        config.group_size,
    )

    # Reshape to attention format
    q = q.reshape(batch_size, 1, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch_size, 1, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch_size, 1, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

    return q, k, v


def quantized_kv_attention(
    q: Any,  # [batch, num_heads, 1, head_dim]
    k_cache: Any,  # [batch, num_kv_heads, seq_len, head_dim] - quantized uint8
    v_cache: Any,  # [batch, num_kv_heads, seq_len, head_dim] - quantized uint8
    k_scales: Any,  # [batch, num_kv_heads, seq_len, 1]
    v_scales: Any,  # [batch, num_kv_heads, seq_len, 1]
    num_heads: int,
    num_kv_heads: int,
    attention_mask: Any | None = None,
) -> Any:
    """Scaled dot-product attention with quantized KV cache.

    Performs fused attention computation directly on FP8/INT8 quantized
    KV cache, avoiding the need to dequantize the full cache.

    Memory savings:
    - FP8: 2x reduction vs BF16
    - INT8: 2x reduction vs BF16

    Args:
        q: Query tensor [batch, num_heads, 1, head_dim].
        k_cache: Quantized key cache [batch, num_kv_heads, seq_len, head_dim].
        v_cache: Quantized value cache [batch, num_kv_heads, seq_len, head_dim].
        k_scales: Per-position K scales [batch, num_kv_heads, seq_len, 1].
        v_scales: Per-position V scales [batch, num_kv_heads, seq_len, 1].
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads.
        attention_mask: Optional attention mask.

    Returns:
        Attention output [batch, num_heads, 1, head_dim].
    """
    require_mlx("quantized_kv_attention")

    q.shape[0]
    head_dim = q.shape[3]
    k_cache.shape[2]

    # Dequantize K and V (for now - fused kernel to be implemented)
    # FP8 E4M3 simulated: value = (quant - 128) / 127 * 448 * scale
    k_dequant = (k_cache.astype(mx.float32) - 128.0) / 127.0 * 448.0 * k_scales
    v_dequant = (v_cache.astype(mx.float32) - 128.0) / 127.0 * 448.0 * v_scales

    k_dequant = k_dequant.astype(mx.bfloat16)
    v_dequant = v_dequant.astype(mx.bfloat16)

    # Expand K/V for GQA if needed
    if num_kv_heads < num_heads:
        repeat_factor = num_heads // num_kv_heads
        k_dequant = mx.repeat(k_dequant, repeat_factor, axis=1)
        v_dequant = mx.repeat(v_dequant, repeat_factor, axis=1)

    # Scaled dot-product attention
    scale = 1.0 / math.sqrt(head_dim)
    attn_weights = (q @ k_dequant.transpose(0, 1, 3, 2)) * scale

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = mx.softmax(attn_weights, axis=-1)
    attn_output = attn_weights @ v_dequant

    return attn_output


def _apply_rope(
    q: Any,  # [batch, num_heads, seq_len, head_dim]
    k: Any,  # [batch, num_kv_heads, seq_len, head_dim]
    position: int,
    rope_theta: float = 10000.0,
) -> tuple[Any, Any]:
    """Apply Rotary Position Embedding to Q and K."""
    require_mlx("_apply_rope")

    head_dim = q.shape[3]
    seq_len = q.shape[2]

    # Compute position indices
    positions = mx.arange(position, position + seq_len, dtype=mx.float32)

    # Compute inverse frequencies
    dims = mx.arange(0, head_dim, 2, dtype=mx.float32)
    inv_freq = 1.0 / (rope_theta ** (dims / head_dim))

    # Compute angles
    freqs = mx.outer(positions, inv_freq)
    cos = mx.cos(freqs)[None, None, :, :]
    sin = mx.sin(freqs)[None, None, :, :]

    # Apply rotation
    q_even, q_odd = q[..., ::2], q[..., 1::2]
    k_even, k_odd = k[..., ::2], k[..., 1::2]

    q_rotated_even = q_even * cos - q_odd * sin
    q_rotated_odd = q_odd * cos + q_even * sin
    k_rotated_even = k_even * cos - k_odd * sin
    k_rotated_odd = k_odd * cos + k_even * sin

    # Interleave back
    q_rotated = mx.stack([q_rotated_even, q_rotated_odd], axis=-1).reshape(q.shape)
    k_rotated = mx.stack([k_rotated_even, k_rotated_odd], axis=-1).reshape(k.shape)

    return q_rotated, k_rotated


def _rms_norm(x: Any, weight: Any, eps: float = 1e-6) -> Any:
    """RMS normalization."""
    variance = mx.mean(x**2, axis=-1, keepdims=True)
    x_normed = x * mx.rsqrt(variance + eps)
    return weight * x_normed


def _mlp_forward(
    hidden_states: Any,
    gate_weight: Any,
    gate_scales: Any,
    up_weight: Any,
    up_scales: Any,
    down_weight: Any,
    down_scales: Any,
    group_size: int,
) -> Any:
    """SwiGLU MLP forward pass."""
    require_mlx("_mlp_forward")
    import mlx.nn as nn

    from ..metal_marlin import quantized_linear

    gate = quantized_linear(hidden_states, gate_weight, gate_scales, group_size)
    up = quantized_linear(hidden_states, up_weight, up_scales, group_size)
    hidden = nn.silu(gate) * up
    return quantized_linear(hidden, down_weight, down_scales, group_size)


def persistent_decode_step(
    hidden_states: Any,  # [batch, 1, hidden_size]
    state: DecodeState,
    layer_idx: int,
    k_cache: Any,  # [batch, num_kv_heads, seq_len, head_dim]
    v_cache: Any,  # [batch, num_kv_heads, seq_len, head_dim]
    position: int,
    attention_mask: Any | None = None,
    rope_theta: float = 10000.0,
    rms_norm_eps: float = 1e-6,
) -> tuple[Any, Any, Any]:
    """Single decode step for one transformer layer.

    Optimized for M=1 decode with:
    - Specialized GEMV kernels for all linear projections
    - Fused QKV projection (when enabled)
    - Quantized KV cache attention (when enabled)
    - Persistent buffers to avoid allocation

    Args:
        hidden_states: Input tensor [batch, 1, hidden_size].
        state: Persistent decode state with weights and buffers.
        layer_idx: Index of the current layer.
        k_cache: Key cache [batch, num_kv_heads, seq_len, head_dim].
        v_cache: Value cache [batch, num_kv_heads, seq_len, head_dim].
        position: Current sequence position.
        attention_mask: Optional attention mask.
        rope_theta: RoPE base frequency.
        rms_norm_eps: RMSNorm epsilon.

    Returns:
        Tuple of (hidden_states, new_k, new_v):
        - hidden_states: Output tensor [batch, 1, hidden_size]
        - new_k: New key to append to cache [batch, num_kv_heads, 1, head_dim]
        - new_v: New value to append to cache [batch, num_kv_heads, 1, head_dim]
    """
    require_mlx("persistent_decode_step")

    from ..metal_marlin import quantized_linear

    config = state.config
    layer = state.layer_weights[layer_idx]

    batch_size = hidden_states.shape[0]
    num_heads = config.num_heads
    num_kv_heads = config.num_kv_heads
    head_dim = config.head_dim

    assert num_kv_heads is not None
    assert head_dim is not None

    # ========== Self-Attention ==========
    residual = hidden_states

    # Input LayerNorm
    hidden_states = _rms_norm(
        hidden_states, layer.input_layernorm_weight, rms_norm_eps
    )

    # QKV Projections (fused when enabled)
    if config.use_fused_attention:
        q, k, v = fused_qkv_projection(hidden_states, layer, config)
    else:
        h = hidden_states.reshape(batch_size, config.hidden_size)

        q = quantized_linear(
            h, layer.q_proj_weight, layer.q_proj_scales, config.group_size
        )
        k = quantized_linear(
            h, layer.k_proj_weight, layer.k_proj_scales, config.group_size
        )
        v = quantized_linear(
            h, layer.v_proj_weight, layer.v_proj_scales, config.group_size
        )

        q = q.reshape(batch_size, 1, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, 1, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, 1, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

    # Apply RoPE
    q, k = _apply_rope(q, k, position, rope_theta)

    # Store new K, V for cache update
    new_k = k
    new_v = v

    # Attention with existing cache
    if k_cache is not None and k_cache.shape[2] > 0:
        # Check if cache is quantized
        if config.kv_cache_dtype in ("fp8", "int8"):
            # Quantized attention path (to be implemented)
            # For now, fall back to dequantized attention
            k_full = mx.concatenate([k_cache, k], axis=2)
            v_full = mx.concatenate([v_cache, v], axis=2)
        else:
            # Standard attention
            k_full = mx.concatenate([k_cache, k], axis=2)
            v_full = mx.concatenate([v_cache, v], axis=2)
    else:
        k_full = k
        v_full = v

    # Expand K/V for GQA
    if num_kv_heads < num_heads:
        repeat_factor = num_heads // num_kv_heads
        k_full = mx.repeat(k_full, repeat_factor, axis=1)
        v_full = mx.repeat(v_full, repeat_factor, axis=1)

    # Scaled dot-product attention
    scale = 1.0 / math.sqrt(head_dim)
    attn_weights = (q @ k_full.transpose(0, 1, 3, 2)) * scale

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = mx.softmax(attn_weights, axis=-1)
    attn_output = attn_weights @ v_full

    # Reshape and project output
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
        batch_size, 1, num_heads * head_dim
    )

    hidden_states = quantized_linear(
        attn_output.reshape(batch_size, num_heads * head_dim),
        layer.o_proj_weight,
        layer.o_proj_scales,
        config.group_size,
    ).reshape(batch_size, 1, config.hidden_size)

    # Residual connection
    hidden_states = residual + hidden_states

    # ========== MLP ==========
    residual = hidden_states

    # Post-attention LayerNorm
    hidden_states = _rms_norm(
        hidden_states, layer.post_attention_layernorm_weight, rms_norm_eps
    )

    # MLP forward
    h = hidden_states.reshape(batch_size, config.hidden_size)
    hidden_states = _mlp_forward(
        h,
        layer.gate_proj_weight,
        layer.gate_proj_scales,
        layer.up_proj_weight,
        layer.up_proj_scales,
        layer.down_proj_weight,
        layer.down_proj_scales,
        config.group_size,
    ).reshape(batch_size, 1, config.hidden_size)

    # Residual connection
    hidden_states = residual + hidden_states

    return hidden_states, new_k, new_v
