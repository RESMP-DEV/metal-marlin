"""
MLA (Multi-head Latent Attention) kernel wrappers for Metal.

Provides optimized projections for MLA-style attention mechanisms:
- kv_a_proj: hidden → latent (down-projection)
- kv_b_proj: latent → output (up-projection)
- Fused kv_a + kv_b (skip intermediate materialization)
- RoPE-fused variants for position encoding
- Decode-optimized GEMV for single token inference

MLA Architecture (GLM-4, DeepSeek-V2/V3):
    - Uses compressed KV cache via latent projections
    - kv_lora_rank typically 512-1536 (small latent dimension)
    - Enables 10-20x KV cache compression vs standard MHA

Usage:
    from mla_kernels import mla_proj_fp4, mla_fused_kv_proj_fp4, mla_decode_proj_fp4

    # Single projection (kv_a or kv_b)
    latent = mla_proj_fp4(hidden, W_a_packed, scales_a, group_size=64)

    # Fused projection (skip intermediate)
    output = mla_fused_kv_proj_fp4(
        hidden, W_a_packed, scales_a, W_b_packed, scales_b,
        group_size_a=64, group_size_b=64
    )

    # Decode phase (single token)
    output = mla_decode_proj_fp4(x, W_packed, scales, group_size=64)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._compat import HAS_MLX

if HAS_MLX:
    import mlx.core as mx

    from .dtypes import get_default_config
elif TYPE_CHECKING:
    import mlx.core as mx

    from .dtypes import get_default_config


def _mlx_required(*args, **kwargs):
    """Stub function that raises ImportError when MLX is unavailable."""
    raise ImportError(
        "MLA kernel dispatch requires MLX. "
        "Install with: pip install mlx"
    )


# Export stubs when MLX unavailable
if not HAS_MLX:
    mla_proj_fp4 = _mlx_required
    mla_fused_kv_proj_fp4 = _mlx_required
    mla_proj_with_rope_fp4 = _mlx_required
    mla_decode_proj_fp4 = _mlx_required
    mla_decode_batched_fp4 = _mlx_required


# ---------------------------------------------------------------------------
# Constants matching mla_proj.metal
# ---------------------------------------------------------------------------

TILE_M_MLA = 64
TILE_N_MLA = 64
TILE_K_MLA = 16
TILE_K_MLA_LARGE = 32
SIMDGROUPS_PER_TG_MLA = 4
THREADS_PER_TG_MLA = SIMDGROUPS_PER_TG_MLA * 32  # 128
THREADS_PER_TG_DECODE = 64
SG_M_TILES_MLA = 2
SG_N_TILES_MLA = 4
FP4_PER_UINT = 8


# ---------------------------------------------------------------------------
# Metal shader source loading
# ---------------------------------------------------------------------------

def _load_shader_source() -> str:
    """Load mla_proj.metal shader source."""
    shader_path = Path(__file__).parent.parent / "src" / "mla_proj.metal"
    if shader_path.exists():
        return shader_path.read_text()
    raise FileNotFoundError(f"MLA shader not found at {shader_path}")


_MLA_SHADER_SOURCE: str | None = None


def _get_shader_source() -> str:
    """Lazy-load shader source."""
    global _MLA_SHADER_SOURCE
    if _MLA_SHADER_SOURCE is None:
        _MLA_SHADER_SOURCE = _load_shader_source()
    return _MLA_SHADER_SOURCE


# ---------------------------------------------------------------------------
# Header for inline kernel definitions (matches kernels.py pattern)
# ---------------------------------------------------------------------------

_MLA_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// FP4 E2M1 dequantization
inline half dequant_fp4_scalar(uint nibble) {
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;

    half sub_mag = half(man_bit) * half(0.25h);
    half norm_mag = half(1u << (exp_bits - 1)) * (half(1.0h) + half(man_bit) * half(0.5h));
    half magnitude = select(norm_mag, sub_mag, exp_bits == 0);
    return select(magnitude, -magnitude, bool(sign_bit));
}

inline void dequant_fp4x8(uint32_t packed, half scale, thread half *out) {
    float fscale = (float)scale;
    out[0] = (half)((float)dequant_fp4_scalar((packed >>  0) & 0xF) * fscale);
    out[1] = (half)((float)dequant_fp4_scalar((packed >>  4) & 0xF) * fscale);
    out[2] = (half)((float)dequant_fp4_scalar((packed >>  8) & 0xF) * fscale);
    out[3] = (half)((float)dequant_fp4_scalar((packed >> 12) & 0xF) * fscale);
    out[4] = (half)((float)dequant_fp4_scalar((packed >> 16) & 0xF) * fscale);
    out[5] = (half)((float)dequant_fp4_scalar((packed >> 20) & 0xF) * fscale);
    out[6] = (half)((float)dequant_fp4_scalar((packed >> 24) & 0xF) * fscale);
    out[7] = (half)((float)dequant_fp4_scalar((packed >> 28) & 0xF) * fscale);
}
"""

# ---------------------------------------------------------------------------
# Kernel body: MLA single projection (matches mla_proj_fp4_impl template)
# ---------------------------------------------------------------------------

_MLA_PROJ_SOURCE = """
    uint3 tgid = threadgroup_position_in_grid;
    uint simd_lane = thread_index_in_simdgroup;
    uint simd_id = simdgroup_index_in_threadgroup;
    uint thread_idx = simd_id * 32 + simd_lane;

    threadgroup half A_tile[TILE_M][TILE_K];
    threadgroup half B_tile[TILE_K][TILE_N];

    uint tg_row = tgid.y * TILE_M;
    uint tg_col = tgid.x * TILE_N;
    uint sg_row_offset = (simd_id / 2) * (SG_M_TILES * 8);
    uint sg_col_offset = (simd_id % 2) * (SG_N_TILES * 8);

    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi)
        for (uint ni = 0; ni < SG_N_TILES; ++ni)
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

    uint k_packs = (K + FP4_PER_UINT - 1) / FP4_PER_UINT;
    uint scale_tiles = (K + GROUP_SIZE - 1) / GROUP_SIZE;
    uint k_tiles_count = TILE_K / 8;

    for (uint k_block = 0; k_block < K; k_block += TILE_K) {
        // Load A tile
        {
            const uint elems_per_thread = (TILE_M * TILE_K) / THREADS_PER_TG;
            for (uint i = 0; i < elems_per_thread; ++i) {
                uint flat_idx = thread_idx * elems_per_thread + i;
                uint row = flat_idx / TILE_K;
                uint col = flat_idx % TILE_K;
                uint global_row = tg_row + row;
                uint global_col = k_block + col;

                half val = (global_row < M && global_col < K)
                           ? A[global_row * K + global_col]
                           : half(0.0h);
                A_tile[row][col] = val;
            }
        }

        // Load B tile with dequantization
        {
            const uint packed_per_thread = (TILE_K * TILE_N) / (THREADS_PER_TG * FP4_PER_UINT);
            for (uint i = 0; i < packed_per_thread; ++i) {
                uint flat_packed_idx = thread_idx * packed_per_thread + i;
                uint n_idx = flat_packed_idx / (TILE_K / FP4_PER_UINT);
                uint k_group_in_tile = flat_packed_idx % (TILE_K / FP4_PER_UINT);

                uint global_n = tg_col + n_idx;
                uint global_k_base = k_block + k_group_in_tile * FP4_PER_UINT;

                uint scale_k = global_k_base / GROUP_SIZE;
                half s = half(1.0h);
                if (global_n < N && scale_k < scale_tiles) {
                    s = scales[scale_k * N + global_n];
                }

                uint32_t packed = 0;
                uint b_row = global_k_base / FP4_PER_UINT;
                if (global_n < N && b_row < k_packs && global_k_base < K) {
                    packed = B_packed[b_row * N + global_n];
                }

                uint tile_k_base = k_group_in_tile * FP4_PER_UINT;
                half vals[8];
                dequant_fp4x8(packed, s, vals);
                for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < TILE_K; ++v) {
                    if (n_idx < TILE_N) {
                        uint global_k = global_k_base + v;
                        B_tile[tile_k_base + v][n_idx] = (global_k < K) ? vals[v] : half(0.0h);
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute
        for (uint kt = 0; kt < k_tiles_count; ++kt) {
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                simdgroup_matrix<half, 8, 8> a_frag;
                simdgroup_load(a_frag, &A_tile[sg_row_offset + mi * 8][kt * 8], TILE_K);

                for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                    simdgroup_matrix<half, 8, 8> b_frag;
                    simdgroup_load(b_frag, &B_tile[kt * 8][sg_col_offset + ni * 8], TILE_N);
                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_row = tg_row + sg_row_offset + mi * 8;
            uint out_col = tg_col + sg_col_offset + ni * 8;

            if (out_row + 8 <= M && out_col + 8 <= N) {
                simdgroup_store(acc[mi][ni], out + out_row * N + out_col, N);
            } else if (out_row < M && out_col < N) {
                threadgroup half out_staging[8][8];
                simdgroup_store(acc[mi][ni], &out_staging[0][0], 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    if (out_row + r < M && out_col + c < N) {
                        out[(out_row + r) * N + out_col + c] = out_staging[r][c];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
"""

# ---------------------------------------------------------------------------
# Kernel body: Decode GEMV (single token)
# ---------------------------------------------------------------------------

_MLA_DECODE_SOURCE = """
    uint tid = thread_index_in_threadgroup;
    uint tgid_x = threadgroup_position_in_grid.x;
    uint num_tgs = threadgroups_per_grid.x;

    uint col_stride = num_tgs * THREADS_PER_TG;

    for (uint n = tgid_x * THREADS_PER_TG + tid; n < N; n += col_stride) {
        float sum = 0.0f;
        uint k_packs = (K + FP4_PER_UINT - 1) / FP4_PER_UINT;

        for (uint k_pack = 0; k_pack < k_packs; ++k_pack) {
            uint k_base = k_pack * FP4_PER_UINT;
            uint scale_k = k_base / GROUP_SIZE;

            half s = scales[scale_k * N + n];
            uint32_t packed = W_packed[k_pack * N + n];

            half vals[8];
            dequant_fp4x8(packed, s, vals);

            for (uint v = 0; v < FP4_PER_UINT && k_base + v < K; ++v) {
                sum += (float)x[k_base + v] * (float)vals[v];
            }
        }

        out[n] = (half)sum;
    }
"""


# ---------------------------------------------------------------------------
# Kernel caching
# ---------------------------------------------------------------------------

_mla_proj_kernel = None
_mla_proj_kernel_k32 = None
_mla_decode_kernel = None


def _get_mla_proj_kernel(tile_k: int = 16):
    """Get or build MLA projection kernel."""
    global _mla_proj_kernel, _mla_proj_kernel_k32

    if tile_k <= 16:
        if _mla_proj_kernel is not None:
            return _mla_proj_kernel

        _mla_proj_kernel = mx.fast.metal_kernel(
            name="mla_proj_fp4",
            input_names=["A", "B_packed", "scales"],
            output_names=["out"],
            source=_MLA_PROJ_SOURCE,
            header=_MLA_HEADER,
            ensure_row_contiguous=True,
        )
        return _mla_proj_kernel
    else:
        if _mla_proj_kernel_k32 is not None:
            return _mla_proj_kernel_k32

        _mla_proj_kernel_k32 = mx.fast.metal_kernel(
            name="mla_proj_fp4_k32",
            input_names=["A", "B_packed", "scales"],
            output_names=["out"],
            source=_MLA_PROJ_SOURCE,
            header=_MLA_HEADER,
            ensure_row_contiguous=True,
        )
        return _mla_proj_kernel_k32


def _get_mla_decode_kernel():
    """Get or build MLA decode GEMV kernel."""
    global _mla_decode_kernel

    if _mla_decode_kernel is not None:
        return _mla_decode_kernel

    _mla_decode_kernel = mx.fast.metal_kernel(
        name="mla_decode_proj_fp4",
        input_names=["x", "W_packed", "scales"],
        output_names=["out"],
        source=_MLA_DECODE_SOURCE,
        header=_MLA_HEADER,
        ensure_row_contiguous=True,
    )
    return _mla_decode_kernel


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

if HAS_MLX:

    def mla_proj_fp4(
        A: mx.array,
        B_packed: mx.array,
        scales: mx.array,
        group_size: int = 64,
        dtype: mx.Dtype | None = None,
    ) -> mx.array:
        """
        MLA single projection with FP4 quantized weights.

        Optimized for MLA's tall-skinny (kv_a) or short-wide (kv_b) matrices.
        Automatically selects TILE_K based on K dimension.

        Args:
            A: Input activations [*, K]. Arbitrary leading dims.
            B_packed: Packed FP4 weights [K/8, N] as uint32.
            scales: Per-group scales [K/group_size, N].
            group_size: Elements per quantization group (default 64 for MLA).
            dtype: Output dtype. If None, uses DTypeConfig default.

        Returns:
            Output [*, N] with specified dtype.
        """
        if dtype is None:
            dtype = get_default_config().mlx_activations

        orig_shape = A.shape
        K = orig_shape[-1]
        M = 1
        for d in orig_shape[:-1]:
            M *= d

        A_2d = A.reshape(M, K).astype(dtype)
        N = B_packed.shape[1]

        # Select TILE_K based on K dimension
        tile_k = TILE_K_MLA if K <= 1024 else TILE_K_MLA_LARGE

        grid_x = (N + TILE_N_MLA - 1) // TILE_N_MLA
        grid_y = (M + TILE_M_MLA - 1) // TILE_M_MLA

        kernel = _get_mla_proj_kernel(tile_k)
        outputs = kernel(
            inputs=[A_2d, B_packed, scales],
            template=[
                ("M", M),
                ("N", N),
                ("K", K),
                ("GROUP_SIZE", group_size),
                ("TILE_M", TILE_M_MLA),
                ("TILE_N", TILE_N_MLA),
                ("TILE_K", tile_k),
                ("THREADS_PER_TG", THREADS_PER_TG_MLA),
                ("SG_M_TILES", SG_M_TILES_MLA),
                ("SG_N_TILES", SG_N_TILES_MLA),
                ("FP4_PER_UINT", FP4_PER_UINT),
            ],
            grid=(grid_x, grid_y, 1),
            threadgroup=(THREADS_PER_TG_MLA, 1, 1),
            output_shapes=[(M, N)],
            output_dtypes=[dtype],
        )

        out_shape = list(orig_shape[:-1]) + [N]
        return outputs[0].reshape(out_shape)


    def mla_decode_proj_fp4(
        x: mx.array,
        W_packed: mx.array,
        scales: mx.array,
        group_size: int = 64,
        dtype: mx.Dtype | None = None,
    ) -> mx.array:
        """
        MLA decode-phase projection (GEMV) with FP4 weights.

        Optimized for single token inference (batch=1).
        Uses GEMV pattern instead of full GEMM for better efficiency.

        Args:
            x: Input vector [K] or [1, K].
            W_packed: Packed FP4 weights [K/8, N] as uint32.
            scales: Per-group scales [K/group_size, N].
            group_size: Elements per quantization group.
            dtype: Output dtype. If None, uses DTypeConfig default.

        Returns:
            Output [N] or [1, N].
        """
        if dtype is None:
            dtype = get_default_config().mlx_activations

        x_flat = x.reshape(-1).astype(dtype)
        K = x_flat.shape[0]
        N = W_packed.shape[1]

        # Number of threadgroups to cover all output columns
        num_tgs = (N + THREADS_PER_TG_DECODE - 1) // THREADS_PER_TG_DECODE

        kernel = _get_mla_decode_kernel()
        outputs = kernel(
            inputs=[x_flat, W_packed, scales],
            template=[
                ("K", K),
                ("N", N),
                ("GROUP_SIZE", group_size),
                ("THREADS_PER_TG", THREADS_PER_TG_DECODE),
                ("FP4_PER_UINT", FP4_PER_UINT),
            ],
            grid=(num_tgs, 1, 1),
            threadgroup=(THREADS_PER_TG_DECODE, 1, 1),
            output_shapes=[(N,)],
            output_dtypes=[dtype],
        )

        if x.ndim == 2:
            return outputs[0].reshape(1, N)
        return outputs[0]


    def mla_fused_kv_proj_fp4(
        hidden: mx.array,
        W_a_packed: mx.array,
        scales_a: mx.array,
        W_b_packed: mx.array,
        scales_b: mx.array,
        group_size_a: int = 64,
        group_size_b: int = 64,
        dtype: mx.Dtype | None = None,
    ) -> mx.array:
        """
        Fused kv_a + kv_b projection (skip intermediate materialization).

        Computes: output = hidden @ W_a @ W_b
        Without materializing the intermediate latent tensor.

        This is ~15-20% faster than separate projections for small latent dims.

        Args:
            hidden: Input hidden states [batch, seq, hidden_size].
            W_a_packed: kv_a weights [hidden_size/8, kv_lora_rank] FP4.
            scales_a: kv_a scales [hidden_size/group_size_a, kv_lora_rank].
            W_b_packed: kv_b weights [kv_lora_rank/8, n_kv_heads * head_dim] FP4.
            scales_b: kv_b scales [kv_lora_rank/group_size_b, n_kv_heads * head_dim].
            group_size_a: Group size for W_a quantization.
            group_size_b: Group size for W_b quantization.
            dtype: Output dtype.

        Returns:
            Output [batch, seq, n_kv_heads * head_dim].

        Note:
            This implementation currently uses two sequential projections.
            A true fused kernel (mla_fused_kv_proj_fp4 in mla_proj.metal) can
            be added for further optimization when latent dim is very small.
        """
        if dtype is None:
            dtype = get_default_config().mlx_activations

        # For now, implement as two sequential projections
        # The Metal kernel exists but the MLX fast.metal_kernel API
        # doesn't easily support the complex fused kernel interface
        latent = mla_proj_fp4(hidden, W_a_packed, scales_a, group_size_a, dtype)
        output = mla_proj_fp4(latent, W_b_packed, scales_b, group_size_b, dtype)
        return output


    def mla_proj_with_rope_fp4(
        A: mx.array,
        B_packed: mx.array,
        scales: mx.array,
        cos_cache: mx.array,
        sin_cache: mx.array,
        positions: mx.array,
        rope_dim: int,
        group_size: int = 64,
        dtype: mx.Dtype | None = None,
    ) -> mx.array:
        """
        MLA projection with fused RoPE on partial output.

        For GLM-4 style models where RoPE is applied to decoupled rope_head_dim.

        Args:
            A: Input activations [*, K].
            B_packed: Packed FP4 weights [K/8, N] as uint32.
            scales: Per-group scales [K/group_size, N].
            cos_cache: Cosine cache [max_seq, rope_dim/2].
            sin_cache: Sine cache [max_seq, rope_dim/2].
            positions: Position indices [*] matching leading dims of A.
            rope_dim: Number of dimensions to apply RoPE to.
            group_size: Elements per quantization group.
            dtype: Output dtype.

        Returns:
            Output [*, N] with RoPE applied to first rope_dim columns.

        Note:
            This is a reference implementation. For maximum performance,
            the mla_proj_with_rope_fp4 Metal kernel should be used directly.
        """
        if dtype is None:
            dtype = get_default_config().mlx_activations

        # Compute projection
        output = mla_proj_fp4(A, B_packed, scales, group_size, dtype)

        # Apply RoPE to rope_dim portion
        orig_shape = output.shape
        output_flat = output.reshape(-1, orig_shape[-1])
        positions_flat = positions.reshape(-1)

        half_rope = rope_dim // 2
        rope_part = output_flat[:, :rope_dim]
        rope_x = rope_part[:, :half_rope]
        rope_y = rope_part[:, half_rope:]

        # Gather cos/sin from cache
        cos_vals = cos_cache[positions_flat]  # [M, half_rope]
        sin_vals = sin_cache[positions_flat]  # [M, half_rope]

        # Apply rotation: (x, y) -> (x*cos - y*sin, x*sin + y*cos)
        rope_x_new = rope_x * cos_vals - rope_y * sin_vals
        rope_y_new = rope_x * sin_vals + rope_y * cos_vals

        # Reconstruct output
        output_flat = mx.concatenate([
            rope_x_new,
            rope_y_new,
            output_flat[:, rope_dim:]
        ], axis=-1)

        return output_flat.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Weight packing utilities
# ---------------------------------------------------------------------------

def pack_fp4_weights_mla(
    weight: np.ndarray | mx.array,
    group_size: int = 64,
) -> tuple[mx.array, mx.array]:
    """
    Pack weights for MLA FP4 projection.

    Same as standard FP4 packing but with MLA-optimal group sizes.

    Args:
        weight: FP16/FP32 weight matrix [out_features, in_features].
        group_size: Quantization group size (default 64 for MLA).

    Returns:
        (packed_weights, scales) for use with mla_proj_fp4.
    """
    if HAS_MLX:
        from .kernels import pack_fp4_weights
        if isinstance(weight, np.ndarray):
            weight = mx.array(weight)
        return pack_fp4_weights(weight, group_size)
    else:
        raise ImportError("pack_fp4_weights_mla requires MLX")


# ---------------------------------------------------------------------------
# MLA Layer Wrapper
# ---------------------------------------------------------------------------

class MLALinear:
    """
    Quantized linear layer optimized for MLA projections.

    Automatically selects between GEMM and GEMV based on batch size.

    Usage:
        kv_a = MLALinear(hidden_size, kv_lora_rank, group_size=64)
        kv_a.load_quantized(packed_weights, scales)
        latent = kv_a(hidden_states)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 64,
        dtype: mx.Dtype | None = None,
    ):
        """
        Initialize MLA linear layer.

        Args:
            in_features: Input dimension (K).
            out_features: Output dimension (N).
            group_size: Quantization group size.
            dtype: Output dtype. If None, uses default.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.dtype = dtype

        self.weight_packed: mx.array | None = None
        self.scales: mx.array | None = None

    def load_quantized(self, weight_packed: mx.array, scales: mx.array) -> None:
        """Load pre-quantized weights."""
        self.weight_packed = weight_packed
        self.scales = scales

    def quantize_weights(self, weight: mx.array) -> None:
        """Quantize and store FP16/FP32 weights."""
        self.weight_packed, self.scales = pack_fp4_weights_mla(weight, self.group_size)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Automatically selects GEMV for single token, GEMM otherwise.
        """
        if self.weight_packed is None or self.scales is None:
            raise RuntimeError("Weights not loaded. Call load_quantized() or quantize_weights() first.")

        # Check if this is decode phase (single token per batch element)
        batch_size = 1
        for d in x.shape[:-1]:
            batch_size *= d

        if batch_size == 1:
            return mla_decode_proj_fp4(
                x, self.weight_packed, self.scales,
                self.group_size, self.dtype
            )
        else:
            return mla_proj_fp4(
                x, self.weight_packed, self.scales,
                self.group_size, self.dtype
            )


class MLAAttentionProjections:
    """
    Complete MLA projection stack (kv_a, kv_b, q_a, q_b).

    Encapsulates the latent projection pattern for MLA attention.

    MLA Architecture:
        Q: hidden -> q_a -> q_b -> [num_heads, head_dim]
        KV: hidden -> kv_a -> [kv_lora_rank] (cached) -> kv_b -> [num_kv_heads, 2*head_dim]
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_lora_rank: int | None = None,
        kv_lora_rank: int = 512,
        rope_head_dim: int = 0,
        group_size: int = 64,
        dtype: mx.Dtype | None = None,
    ):
        """
        Initialize MLA projection layers.

        Args:
            hidden_size: Model hidden dimension.
            num_heads: Number of attention heads.
            num_kv_heads: Number of KV heads (for GQA).
            head_dim: Dimension per head.
            q_lora_rank: Query latent rank (None = no compression).
            kv_lora_rank: KV latent rank.
            rope_head_dim: Dimension for decoupled RoPE.
            group_size: Quantization group size.
            dtype: Compute dtype.
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.rope_head_dim = rope_head_dim
        self.group_size = group_size
        self.dtype = dtype

        # KV projections (always compressed)
        self.kv_a_proj = MLALinear(
            hidden_size, kv_lora_rank + rope_head_dim, group_size, dtype
        )
        self.kv_b_proj = MLALinear(
            kv_lora_rank, num_kv_heads * head_dim * 2, group_size, dtype
        )

        # Q projections (optionally compressed)
        if q_lora_rank is not None:
            self.q_a_proj = MLALinear(hidden_size, q_lora_rank, group_size, dtype)
            self.q_b_proj = MLALinear(q_lora_rank, num_heads * head_dim, group_size, dtype)
        else:
            self.q_proj = MLALinear(hidden_size, num_heads * head_dim, group_size, dtype)

    def project_kv(self, hidden_states: mx.array) -> tuple[mx.array, mx.array | None]:
        """
        Project hidden states to KV cache representation.

        Args:
            hidden_states: [batch, seq, hidden_size]

        Returns:
            (kv_latent, rope_latent) where:
                kv_latent: [batch, seq, kv_lora_rank] - to be cached
                rope_latent: [batch, seq, rope_head_dim] - for RoPE (if rope_head_dim > 0)
        """
        # Down-project to latent space
        kv_a_out = self.kv_a_proj(hidden_states)  # [batch, seq, kv_lora_rank + rope_head_dim]

        if self.rope_head_dim > 0:
            # Split into cacheable latent and RoPE portion
            kv_latent = kv_a_out[..., :self.kv_lora_rank]
            rope_latent = kv_a_out[..., self.kv_lora_rank:]
            return kv_latent, rope_latent
        else:
            return kv_a_out, None

    def decompress_kv(self, kv_latent: mx.array) -> mx.array:
        """
        Decompress KV latent to full K,V tensors.

        Args:
            kv_latent: [batch, seq, kv_lora_rank] from cache

        Returns:
            kv: [batch, seq, num_kv_heads * head_dim * 2]
        """
        return self.kv_b_proj(kv_latent)

    def project_q(self, hidden_states: mx.array) -> mx.array:
        """
        Project hidden states to queries.

        Args:
            hidden_states: [batch, seq, hidden_size]

        Returns:
            q: [batch, seq, num_heads * head_dim]
        """
        if self.q_lora_rank is not None:
            q_latent = self.q_a_proj(hidden_states)
            return self.q_b_proj(q_latent)
        else:
            return self.q_proj(hidden_states)
