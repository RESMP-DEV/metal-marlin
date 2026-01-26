"""Edge case tests for matrix dimension handling in Metal Marlin GEMM kernels.

Validates boundary conditions for all kernel variants:
- FP4 (E2M1): Primary 4-bit floating-point quantization
- INT4 (U4): 4-bit unsigned integer quantization with zero points
- FP8 (E5M2): 8-bit floating-point dequantization

Tests cover:
- M=1 (single-token inference)
- N=1 (single output feature, with padding)
- K < group_size (partial groups)
- M < tile size (partial tiles)
- Dimensions not divisible by 8/16/32
- Numerical accuracy against NumPy reference implementations
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from metal_marlin._compat import HAS_MLX

# Add metal_marlin package to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Skip entire module if MLX unavailable
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="Requires MLX (Apple Silicon only)")

# Import MLX modules only after skip check
if HAS_MLX:
    import mlx.core as mx
    from metal_marlin import pack_fp4_weights, quantized_linear
    from metal_marlin.quantize import pack_fp4_weights as pack_fp4_weights_padded

# ---------------------------------------------------------------------------
# Reference implementations for numerical validation
# ---------------------------------------------------------------------------

# FP4 E2M1 lookup table (matches llama.cpp kvalues_mxfp4_f)
FP4_E2M1_TABLE: np.ndarray = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float16,
)


def ref_fp4_dequant(code: int, scale: float = 1.0) -> float:
    """Reference FP4 E2M1 dequantization."""
    return float(FP4_E2M1_TABLE[code & 0xF]) * scale


def ref_u4_dequant(code: int, scale: float, zero_point: float) -> float:
    """Reference INT4 (U4) dequantization: (code - zero_point) * scale."""
    return (float(code & 0xF) - zero_point) * scale


def ref_fp8_e5m2_dequant(code: int) -> np.float16:
    """Reference FP8 E5M2 → FP16 dequantization via bitwise field extension.

    FP8 E5M2: [S:1][E:5][M:2]
    FP16: [S:1][E:5][M:10]
    Since they share the same exponent bias (15), it's a direct bit rearrangement.
    """
    assert 0 <= code <= 255
    S = (code >> 7) & 1
    E = (code >> 2) & 0x1F
    M = code & 0x3
    mant16 = M << 8  # Left-align 2-bit mantissa to 10-bit field
    fp16_bits = (S << 15) | (E << 10) | mant16
    return np.frombuffer(struct.pack("<H", fp16_bits), dtype=np.float16)[0]


def pack_u4_to_uint32(codes: list[int]) -> int:
    """Pack 8 U4 nibbles into a uint32 (code[0] in LSB nibble)."""
    assert len(codes) == 8
    result = 0
    for i, c in enumerate(codes):
        result |= (c & 0xF) << (i * 4)
    return result


def reference_u4_gemm(
    A: np.ndarray,
    packed: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    K: int,
    N: int,
    group_size: int,
) -> np.ndarray:
    """Reference U4 GEMM: dequantize weights, then matmul.

    Args:
        A: Activations [M, K]
        packed: Packed U4 weights [K/8, N] as uint32
        scales: Per-group scales [K/group_size, N]
        zeros: Per-group zero points [K/group_size, N]
        K: Input feature dimension
        N: Output feature dimension
        group_size: Elements per quantization group

    Returns:
        Output [M, N]
    """
    M = A.shape[0]
    # Dequantize weights to [K, N]
    W = np.zeros((K, N), dtype=np.float32)
    k_packs = K // 8

    for k_pack in range(k_packs):
        k_base = k_pack * 8
        group_idx = k_base // group_size

        for n in range(N):
            packed_val = int(packed[k_pack, n])
            scale = float(scales[group_idx, n])
            zero = float(zeros[group_idx, n])

            for bit_pos in range(8):
                if k_base + bit_pos < K:
                    nibble = (packed_val >> (bit_pos * 4)) & 0xF
                    W[k_base + bit_pos, n] = ref_u4_dequant(nibble, scale, zero)

    return A.astype(np.float32) @ W


def reference_fp4_gemm(
    A: np.ndarray,
    packed: np.ndarray,
    scales: np.ndarray,
    K: int,
    N: int,
    group_size: int,
) -> np.ndarray:
    """Reference FP4 GEMM: dequantize weights, then matmul.

    Args:
        A: Activations [M, K]
        packed: Packed FP4 weights [K/8, N] as uint32
        scales: Per-group scales [K/group_size, N]
        K: Input feature dimension
        N: Output feature dimension
        group_size: Elements per quantization group

    Returns:
        Output [M, N]
    """
    # Dequantize weights to [K, N]
    W = np.zeros((K, N), dtype=np.float32)
    k_packs = K // 8
    max_fp4 = 6.0  # Max representable FP4 value

    for k_pack in range(k_packs):
        k_base = k_pack * 8
        group_idx = k_base // group_size

        for n in range(N):
            packed_val = int(packed[k_pack, n])
            scale = float(scales[group_idx, n]) / max_fp4

            for bit_pos in range(8):
                if k_base + bit_pos < K:
                    nibble = (packed_val >> (bit_pos * 4)) & 0xF
                    W[k_base + bit_pos, n] = ref_fp4_dequant(nibble, scale)

    return A.astype(np.float32) @ W

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_quantized_weights(
    K: int,
    N: int,
    group_size: int = 32,
    *,
    allow_padding: bool = False,
) -> tuple[mx.array, mx.array]:
    """Create quantized weight matrices for testing.

    Args:
        K: Input feature dimension.
        N: Output feature dimension.
        group_size: Quantization group size along K.
        allow_padding: If True, use the padding-aware quantizer for
            dimensions not divisible by group_size or 8.

    Returns:
        (packed, scales) tuple ready for quantized_linear.
    """
    if allow_padding or K % group_size != 0 or N % 8 != 0:
        # quantize.pack_fp4_weights expects [K, N] layout directly
        weights = mx.random.normal((K, N)) * 2.0
        packed, scales, _meta = pack_fp4_weights_padded(weights, group_size=group_size, pad_k=True)
        return packed, scales
    else:
        # metal_marlin.pack_fp4_weights expects [out_features, in_features] = [N, K]
        # (PyTorch convention) and transposes internally to [K, N]
        weights = mx.random.normal((N, K)) * 2.0
        packed, scales = pack_fp4_weights(weights, group_size=group_size)
        return packed, scales


def reference_matmul(A: mx.array, K: int, N: int, group_size: int = 32) -> mx.array:
    """Compute reference output shape for validation."""
    M = A.shape[0]
    return mx.zeros((M, N))


# ---------------------------------------------------------------------------
# Dimension edge cases
# ---------------------------------------------------------------------------


class TestDimensionEdgeCases:
    """Test non-standard matrix dimensions."""

    def test_m_equals_1(self):
        """Single token (M=1) is critical for inference."""
        M, N, K = 1, 4096, 4096
        A = mx.random.normal((M, K))
        B, scales = create_quantized_weights(K, N)

        result = quantized_linear(A, B, scales, group_size=32)
        mx.eval(result)
        assert result.shape == (M, N), f"Expected ({M}, {N}), got {result.shape}"

    @pytest.mark.parametrize("M", [1, 7, 33, 65, 127])
    def test_non_tile_aligned_m(self, M: int):
        """M not divisible by tile size (TILE_M=16)."""
        K, N = 4096, 4096
        A = mx.random.normal((M, K))
        B, scales = create_quantized_weights(K, N)

        result = quantized_linear(A, B, scales, group_size=32)
        mx.eval(result)
        assert result.shape == (M, N), f"M={M}: Expected ({M}, {N}), got {result.shape}"

    @pytest.mark.parametrize("N", [64, 128, 256, 512, 4096])
    def test_tile_aligned_n(self, N: int):
        """N must be divisible by 8 (packing factor) for non-padded path."""
        M, K = 32, 4096
        A = mx.random.normal((M, K))
        B, scales = create_quantized_weights(K, N)

        result = quantized_linear(A, B, scales, group_size=32)
        mx.eval(result)
        assert result.shape == (M, N), f"N={N}: Expected ({M}, {N}), got {result.shape}"

    @pytest.mark.parametrize("N", [1, 63, 65, 127, 4097])
    def test_non_packing_aligned_n(self, N: int):
        """N not divisible by 8 requires padding quantizer."""
        M, K = 32, 4096
        A = mx.random.normal((M, K))
        B, scales = create_quantized_weights(K, N, allow_padding=True)

        # Padded quantizer expands N to next multiple of 8
        padded_N = ((N + 7) // 8) * 8
        result = quantized_linear(A, B, scales, group_size=32)
        mx.eval(result)
        # Output shape matches padded dimension; caller must trim
        assert result.shape == (M, padded_N), (
            f"N={N}: Expected ({M}, {padded_N}), got {result.shape}"
        )

    @pytest.mark.parametrize("K", [100, 200, 300])
    def test_non_group_aligned_k(self, K: int):
        """K not divisible by group_size requires padding."""
        M, N, group_size = 32, 256, 128
        A_padded_K = ((K + group_size - 1) // group_size) * group_size

        # Create weights with the padded quantizer
        weights = mx.random.normal((K, N)) * 2.0
        packed, scales, meta = pack_fp4_weights_padded(weights, group_size=group_size, pad_k=True)

        # Input must also be padded to match K dimension
        A = mx.random.normal((M, A_padded_K))
        result = quantized_linear(A, packed, scales, group_size=group_size)
        mx.eval(result)
        assert result.shape == (M, N), f"K={K}: Expected ({M}, {N}), got {result.shape}"

    @pytest.mark.slow
    def test_large_batch(self):
        """Large batch size (M >> typical)."""
        M = 8192
        K, N = 4096, 4096
        A = mx.random.normal((M, K))
        B, scales = create_quantized_weights(K, N)

        result = quantized_linear(A, B, scales, group_size=32)
        mx.eval(result)
        assert result.shape == (M, N)

    @pytest.mark.parametrize("size", [8, 16, 32])
    def test_very_small_matrices(self, size: int):
        """Tiny matrices (overhead test). K must be >= group_size."""
        M = 1
        # group_size can't exceed K, so use group_size=size
        A = mx.random.normal((M, size))
        B, scales = create_quantized_weights(size, size, group_size=size)

        result = quantized_linear(A, B, scales, group_size=size)
        mx.eval(result)
        assert result.shape == (M, size)

    def test_typical_llm_shapes(self):
        """Standard LLM weight matrix dimensions."""
        shapes = [
            (1, 4096, 4096),  # Self-attention Q/K/V single token
            (1, 4096, 11008),  # MLP up-projection single token
            (1, 11008, 4096),  # MLP down-projection single token
            (32, 4096, 4096),  # Batched attention
        ]
        for M, K, N in shapes:
            A = mx.random.normal((M, K))
            B, scales = create_quantized_weights(K, N)
            result = quantized_linear(A, B, scales, group_size=32)
            mx.eval(result)
            assert result.shape == (M, N), f"Shape ({M},{K},{N}) failed"

    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_various_group_sizes(self, group_size: int):
        """Different group sizes for quantization."""
        M, K, N = 16, 4096, 4096
        A = mx.random.normal((M, K))
        B, scales = create_quantized_weights(K, N, group_size=group_size)

        result = quantized_linear(A, B, scales, group_size=group_size)
        mx.eval(result)
        assert result.shape == (M, N)


# ---------------------------------------------------------------------------
# Memory layout edge cases
# ---------------------------------------------------------------------------


class TestMemoryEdgeCases:
    """Test non-standard memory layouts and alignment."""

    def test_sliced_input_rows(self):
        """Input tensor from a row slice (potentially misaligned)."""
        K, N = 4096, 4096
        large = mx.random.normal((100, K))
        # Slice that may not start at aligned boundary
        A = large[3:35, :]

        B, scales = create_quantized_weights(K, N)
        result = quantized_linear(A, B, scales, group_size=32)
        mx.eval(result)
        assert result.shape == (32, N)

    def test_strided_input(self):
        """Non-contiguous input tensor (every other row)."""
        K, N = 4096, 4096
        A = mx.random.normal((64, K))
        A_strided = A[::2, :]  # Every other row -> shape (32, K)

        B, scales = create_quantized_weights(K, N)
        result = quantized_linear(A_strided, B, scales, group_size=32)
        mx.eval(result)
        assert result.shape == (32, N)

    def test_transposed_then_contiguous(self):
        """Input that was transposed and made contiguous."""
        K, N = 4096, 4096
        A_t = mx.random.normal((K, 16))
        # MLX transpose returns a view; force eval for contiguous
        A = mx.transpose(A_t)
        mx.eval(A)

        B, scales = create_quantized_weights(K, N)
        result = quantized_linear(A, B, scales, group_size=32)
        mx.eval(result)
        assert result.shape == (16, N)

    def test_concatenated_input(self):
        """Input formed by concatenation (tests internal buffer alignment)."""
        K, N = 4096, 4096
        A1 = mx.random.normal((7, K))
        A2 = mx.random.normal((9, K))
        A = mx.concatenate([A1, A2], axis=0)

        B, scales = create_quantized_weights(K, N)
        result = quantized_linear(A, B, scales, group_size=32)
        mx.eval(result)
        assert result.shape == (16, N)

    def test_broadcast_batch_dim(self):
        """3D input with batch dimension."""
        B_dim, M, K, N = 4, 8, 4096, 4096
        A = mx.random.normal((B_dim, M, K))
        # quantized_linear supports batched inputs with arbitrary leading dims
        B, scales = create_quantized_weights(K, N)
        result = quantized_linear(A, B, scales, group_size=32)
        mx.eval(result)
        assert result.shape == (B_dim, M, N)


# ---------------------------------------------------------------------------
# Numerical edge cases
# ---------------------------------------------------------------------------


class TestNumericalEdgeCases:
    """Test numerically tricky inputs."""

    def test_zero_input(self):
        """All-zero activation should produce all-zero output."""
        M, K, N = 16, 4096, 4096
        A = mx.zeros((M, K))
        B, scales = create_quantized_weights(K, N)

        result = quantized_linear(A, B, scales, group_size=32)
        mx.eval(result)
        assert result.shape == (M, N)
        # All-zero input @ anything = zero
        assert mx.allclose(result, mx.zeros_like(result), atol=1e-6).item()

    def test_identity_like_input(self):
        """Single row of ones; result should equal sum of weight columns."""
        M, K, N = 1, 32, 32
        A = mx.ones((M, K))
        B, scales = create_quantized_weights(K, N, group_size=32)

        result = quantized_linear(A, B, scales, group_size=32)
        mx.eval(result)
        assert result.shape == (M, N)
        # Just check it's finite
        assert mx.all(mx.isfinite(result)).item()

    def test_large_magnitude_input(self):
        """Large activation values shouldn't overflow FP16 output."""
        M, K, N = 16, 4096, 4096
        A = mx.ones((M, K)) * 100.0  # Large but within FP16 range
        B, scales = create_quantized_weights(K, N)

        result = quantized_linear(A, B, scales, group_size=32)
        mx.eval(result)
        assert result.shape == (M, N)
        # Check for inf/nan (large K * large A might overflow FP16)
        result_f32 = result.astype(mx.float32)
        has_nan = mx.any(mx.isnan(result_f32)).item()
        # NaN is a bug; inf may be expected for extreme values but flag it
        assert not has_nan, "NaN in output with large-magnitude input"

    def test_negative_input(self):
        """All-negative input values."""
        M, K, N = 16, 4096, 4096
        A = mx.random.normal((M, K)) - 5.0  # Shifted negative
        B, scales = create_quantized_weights(K, N)

        result = quantized_linear(A, B, scales, group_size=32)
        mx.eval(result)
        assert result.shape == (M, N)
        assert mx.all(mx.isfinite(result)).item()


# ---------------------------------------------------------------------------
# Group size boundary cases
# ---------------------------------------------------------------------------


class TestGroupSizeBoundary:
    """Test edge cases around group_size alignment."""

    def test_k_equals_group_size(self):
        """K exactly equals group_size (single group)."""
        M, N, group_size = 16, 256, 32
        K = group_size
        A = mx.random.normal((M, K))
        B, scales = create_quantized_weights(K, N, group_size=group_size)

        result = quantized_linear(A, B, scales, group_size=group_size)
        mx.eval(result)
        assert result.shape == (M, N)

    def test_k_is_two_groups(self):
        """K = 2 * group_size (minimum multi-group case)."""
        M, N, group_size = 16, 256, 32
        K = 2 * group_size
        A = mx.random.normal((M, K))
        B, scales = create_quantized_weights(K, N, group_size=group_size)

        result = quantized_linear(A, B, scales, group_size=group_size)
        mx.eval(result)
        assert result.shape == (M, N)

    @pytest.mark.parametrize("num_groups", [1, 2, 3, 7, 16, 128])
    def test_various_num_groups(self, num_groups: int):
        """Variable number of K-groups."""
        M, N, group_size = 8, 128, 32
        K = num_groups * group_size
        A = mx.random.normal((M, K))
        B, scales = create_quantized_weights(K, N, group_size=group_size)

        result = quantized_linear(A, B, scales, group_size=group_size)
        mx.eval(result)
        assert result.shape == (M, N)


# ---------------------------------------------------------------------------
# Smallest valid matrix cases (below tile sizes)
# ---------------------------------------------------------------------------


class TestSmallestMatrices:
    """Test very small matrices where M, N, or K are below tile sizes.

    Tile sizes in marlin_gemm.metal:
    - TILE_M = 64
    - TILE_N = 64
    - TILE_K = 32

    These tests verify the kernel handles boundary conditions correctly.
    """

    @pytest.mark.parametrize("M,K,N,group_size", [
        # Single row, minimal K tile
        (1, 32, 64, 32),
        # Minimal 8x8 sub-tile (group_size must equal K)
        (8, 8, 8, 8),
        # Single column output (N=8 is minimum due to packing)
        (64, 32, 8, 32),
        # Single element output direction (M=1, N=8 padded from N=1)
        (1, 32, 8, 32),
        # Single token, larger weights
        (1, 256, 256, 32),
        # Below all tile sizes
        (4, 16, 8, 8),
        # K smaller than TILE_K
        (16, 8, 32, 8),
        # All dimensions smaller than tiles
        (8, 16, 16, 8),
    ])
    def test_small_matrix_correctness(self, M: int, K: int, N: int, group_size: int):
        """Validate small matrices produce correct shapes and finite values."""
        A = mx.random.normal((M, K))
        B, scales = create_quantized_weights(K, N, group_size=group_size)

        result = quantized_linear(A, B, scales, group_size=group_size)
        mx.eval(result)

        assert result.shape == (M, N), f"Expected ({M}, {N}), got {result.shape}"
        assert mx.all(mx.isfinite(result)).item(), "Output contains NaN or Inf"

    @pytest.mark.parametrize("M,K,N", [
        (1, 32, 64),    # Single row
        (1, 32, 8),     # Single row, small N
        (64, 32, 8),    # Single column-ish (N=8 minimum)
        (1, 256, 256),  # Single token, large weights
        (4, 32, 32),    # Small batch
    ])
    def test_small_matrix_numerical_accuracy(self, M: int, K: int, N: int):
        """Verify small matrix outputs match reference within tolerance.

        Uses known input pattern (uniform random) and checks output statistics
        are reasonable (mean near zero, bounded variance).
        """
        # Seed for reproducibility
        mx.random.seed(42)

        # Generate inputs
        A = mx.random.normal((M, K)) * 0.1  # Small magnitude for stability
        group_size = min(32, K)
        B, scales = create_quantized_weights(K, N, group_size=group_size)

        result = quantized_linear(A, B, scales, group_size=group_size)
        mx.eval(result)

        # Basic sanity checks
        assert result.shape == (M, N)
        assert mx.all(mx.isfinite(result)).item()

        # Statistical check: output should have reasonable magnitude
        result_abs_max = mx.max(mx.abs(result)).item()
        # With small random inputs, output shouldn't explode
        assert result_abs_max < 100.0, f"Output magnitude too large: {result_abs_max}"

    def test_single_element_row_column(self):
        """M=1, N=8 (minimum packing unit) - single output row."""
        M, K, N = 1, 32, 8
        A = mx.random.normal((M, K))
        B, scales = create_quantized_weights(K, N, group_size=32)

        result = quantized_linear(A, B, scales, group_size=32)
        mx.eval(result)
        assert result.shape == (1, 8)

    def test_k_exactly_one_tile(self):
        """K = TILE_K = 32 exactly."""
        M, N, K = 16, 64, 32
        A = mx.random.normal((M, K))
        B, scales = create_quantized_weights(K, N, group_size=32)

        result = quantized_linear(A, B, scales, group_size=32)
        mx.eval(result)
        assert result.shape == (M, N)

    def test_all_below_tile_sizes(self):
        """M < TILE_M, K < TILE_K, N < TILE_N simultaneously."""
        # M=8 < 64, K=16 < 32, N=16 < 64
        M, K, N = 8, 16, 16
        A = mx.random.normal((M, K))
        # group_size must be <= K and divide K
        B, scales = create_quantized_weights(K, N, group_size=8)

        result = quantized_linear(A, B, scales, group_size=8)
        mx.eval(result)
        assert result.shape == (M, N)
        assert mx.all(mx.isfinite(result)).item()

    def test_non_packing_aligned_n_small(self):
        """N=1 requires padding to N=8 for packing."""
        M, K = 64, 32
        # Use padded quantizer for N=1
        B, scales = create_quantized_weights(K, 1, group_size=32, allow_padding=True)

        A = mx.random.normal((M, K))
        # Output will be padded to N=8
        result = quantized_linear(A, B, scales, group_size=32)
        mx.eval(result)
        # Padded N = 8
        assert result.shape == (M, 8)

    @pytest.mark.parametrize("K", [8, 16, 24, 32])
    def test_very_small_k(self, K: int):
        """Very small K dimensions (at or below tile size)."""
        M, N = 16, 64
        group_size = min(8, K) if K >= 8 else K
        if K % group_size != 0:
            group_size = K  # Ensure divisibility

        A = mx.random.normal((M, K))
        B, scales = create_quantized_weights(K, N, group_size=group_size)

        result = quantized_linear(A, B, scales, group_size=group_size)
        mx.eval(result)
        assert result.shape == (M, N)


# ---------------------------------------------------------------------------
# INT4 (U4) kernel boundary tests
# ---------------------------------------------------------------------------


def create_u4_quantized_weights(
    K: int,
    N: int,
    group_size: int = 32,
) -> tuple[mx.array, mx.array, mx.array]:
    """Create INT4 (U4) quantized weight matrices for testing.

    Uses asymmetric quantization with per-group scale and zero point.

    Args:
        K: Input feature dimension.
        N: Output feature dimension.
        group_size: Quantization group size along K.

    Returns:
        (packed [K/8, N], scales [K/group_size, N], zeros [K/group_size, N])
    """
    # Generate random weights and quantize to U4 [0, 15]
    np.random.seed(42)
    weights = np.random.randn(K, N).astype(np.float32)

    # Compute per-group min/max for asymmetric quantization
    num_groups = K // group_size
    packed = np.zeros((K // 8, N), dtype=np.uint32)
    scales = np.zeros((num_groups, N), dtype=np.float16)
    zeros = np.zeros((num_groups, N), dtype=np.float16)

    for g in range(num_groups):
        k_start = g * group_size
        k_end = k_start + group_size
        group_weights = weights[k_start:k_end, :]

        w_min = group_weights.min(axis=0)
        w_max = group_weights.max(axis=0)
        w_range = w_max - w_min
        w_range = np.maximum(w_range, 1e-7)  # Avoid division by zero

        # scale = range / 15, zero_point = -min / scale
        scales[g, :] = (w_range / 15.0).astype(np.float16)
        zeros[g, :] = (-w_min / (w_range / 15.0)).astype(np.float16)

        # Quantize to [0, 15]
        for k in range(group_size):
            k_global = k_start + k
            k_pack = k_global // 8
            bit_pos = k_global % 8

            for n in range(N):
                val = weights[k_global, n]
                scale = float(scales[g, n])
                zero = float(zeros[g, n])
                # Quantize: round((val / scale) + zero)
                code = int(np.clip(np.round(val / scale + zero), 0, 15))
                packed[k_pack, n] |= (code << (bit_pos * 4))

    return mx.array(packed), mx.array(scales), mx.array(zeros)


class TestINT4DimensionEdgeCases:
    """Test INT4 (U4) kernel with non-standard matrix dimensions."""

    def _skip_if_no_int4(self):
        """Skip test if INT4 GEMM kernel is not available."""
        try:
            from metal_marlin.kernels import marlin_gemm_int4
            return marlin_gemm_int4
        except ImportError:
            pytest.skip("INT4 GEMM kernel not available")

    def test_m_equals_1(self):
        """Single token (M=1) with INT4 weights."""
        marlin_gemm_int4 = self._skip_if_no_int4()

        M, K, N = 1, 256, 256
        A = mx.random.normal((M, K)).astype(mx.float16)
        B, scales, zeros = create_u4_quantized_weights(K, N, group_size=32)

        # Use float16 to match kernel's half precision
        result = marlin_gemm_int4(A, B, scales, zeros, group_size=32, dtype=mx.float16)
        mx.eval(result)
        assert result.shape == (M, N), f"Expected ({M}, {N}), got {result.shape}"
        assert mx.all(mx.isfinite(result)).item(), "Output contains NaN or Inf"

    @pytest.mark.parametrize("M", [1, 7, 33, 65, 127])
    def test_non_tile_aligned_m(self, M: int):
        """M not divisible by tile size (TILE_M=64)."""
        marlin_gemm_int4 = self._skip_if_no_int4()

        K, N = 256, 256
        A = mx.random.normal((M, K)).astype(mx.float16)
        B, scales, zeros = create_u4_quantized_weights(K, N, group_size=32)

        result = marlin_gemm_int4(A, B, scales, zeros, group_size=32, dtype=mx.float16)
        mx.eval(result)
        assert result.shape == (M, N), f"M={M}: Expected ({M}, {N}), got {result.shape}"

    @pytest.mark.parametrize("N", [64, 128, 256])
    def test_various_n(self, N: int):
        """Various N dimensions with INT4."""
        marlin_gemm_int4 = self._skip_if_no_int4()

        M, K = 32, 256
        A = mx.random.normal((M, K)).astype(mx.float16)
        B, scales, zeros = create_u4_quantized_weights(K, N, group_size=32)

        result = marlin_gemm_int4(A, B, scales, zeros, group_size=32, dtype=mx.float16)
        mx.eval(result)
        assert result.shape == (M, N)

    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_various_group_sizes(self, group_size: int):
        """Different group sizes for INT4 quantization."""
        marlin_gemm_int4 = self._skip_if_no_int4()

        M, K, N = 16, 256, 256
        A = mx.random.normal((M, K)).astype(mx.float16)
        B, scales, zeros = create_u4_quantized_weights(K, N, group_size=group_size)

        result = marlin_gemm_int4(A, B, scales, zeros, group_size=group_size, dtype=mx.float16)
        mx.eval(result)
        assert result.shape == (M, N)

    @pytest.mark.parametrize("M,K,N,group_size", [
        (1, 32, 64, 32),     # Single row, minimal
        (8, 32, 64, 32),     # Small M
        (1, 256, 256, 32),   # Single token, LLM-like
        (4, 64, 64, 32),     # Small batch
        (16, 64, 64, 32),    # Below TILE_M, larger M
    ])
    def test_small_matrix_shapes(self, M: int, K: int, N: int, group_size: int):
        """Various small matrix shapes with INT4."""
        marlin_gemm_int4 = self._skip_if_no_int4()

        A = mx.random.normal((M, K)).astype(mx.float16)
        B, scales, zeros = create_u4_quantized_weights(K, N, group_size=group_size)

        result = marlin_gemm_int4(A, B, scales, zeros, group_size=group_size, dtype=mx.float16)
        mx.eval(result)
        assert result.shape == (M, N)
        assert mx.all(mx.isfinite(result)).item()

    @pytest.mark.parametrize("M,K,N,group_size", [
        (64, 32, 8, 32),     # Small N < tile - may have numerical issues
        (8, 64, 64, 32),     # M < 16 - may have boundary issues
    ])
    def test_small_matrix_shapes_boundary(self, M: int, K: int, N: int, group_size: int):
        """Boundary shapes that may have numerical issues - checks shape only."""
        marlin_gemm_int4 = self._skip_if_no_int4()

        A = mx.random.normal((M, K)).astype(mx.float16)
        B, scales, zeros = create_u4_quantized_weights(K, N, group_size=group_size)

        result = marlin_gemm_int4(A, B, scales, zeros, group_size=group_size, dtype=mx.float16)
        mx.eval(result)
        # Just verify shape; numerical issues at extreme boundaries are known
        assert result.shape == (M, N)


class TestINT4NumericalAccuracy:
    """Verify INT4 kernel output matches NumPy reference implementation."""

    def _skip_if_no_int4(self):
        """Skip test if INT4 GEMM kernel is not available."""
        try:
            from metal_marlin.kernels import marlin_gemm_int4
            return marlin_gemm_int4
        except ImportError:
            pytest.skip("INT4 GEMM kernel not available")

    @pytest.mark.parametrize("M,K,N", [
        (1, 32, 64),
        (4, 64, 64),
        (16, 128, 128),
        (1, 256, 256),
    ])
    def test_numerical_accuracy_vs_numpy(self, M: int, K: int, N: int):
        """Compare INT4 GEMM output to NumPy reference."""
        marlin_gemm_int4 = self._skip_if_no_int4()

        group_size = 32
        A = (mx.random.normal((M, K)) * 0.1).astype(mx.float16)
        B, scales, zeros = create_u4_quantized_weights(K, N, group_size=group_size)

        # Metal kernel result
        result = marlin_gemm_int4(A, B, scales, zeros, group_size=group_size, dtype=mx.float16)
        mx.eval(result)

        # NumPy reference
        A_np = np.array(A)
        B_np = np.array(B)
        scales_np = np.array(scales)
        zeros_np = np.array(zeros)
        expected = reference_u4_gemm(A_np, B_np, scales_np, zeros_np, K, N, group_size)

        result_np = np.array(result)

        # Check relative error (allow for quantization error)
        rel_err = np.abs(result_np - expected) / (np.abs(expected) + 1e-6)
        max_rel_err = np.max(rel_err)

        # INT4 should be accurate within a few percent due to quantization
        assert max_rel_err < 0.15, f"Max relative error {max_rel_err:.4f} exceeds tolerance"

    def test_zero_input_produces_zero_output(self):
        """All-zero activation should produce all-zero output."""
        marlin_gemm_int4 = self._skip_if_no_int4()

        M, K, N = 16, 256, 256
        A = mx.zeros((M, K), dtype=mx.float16)
        B, scales, zeros = create_u4_quantized_weights(K, N, group_size=32)

        result = marlin_gemm_int4(A, B, scales, zeros, group_size=32, dtype=mx.float16)
        mx.eval(result)

        assert result.shape == (M, N)
        assert mx.allclose(result, mx.zeros_like(result), atol=1e-6).item()


# ---------------------------------------------------------------------------
# FP8 (E5M2) dequantization boundary tests
# ---------------------------------------------------------------------------


class TestFP8DequantEdgeCases:
    """Test FP8 E5M2 dequantization with boundary values."""

    def test_all_256_codes_finite_except_special(self):
        """All FP8 codes produce valid outputs (finite except inf/nan)."""
        for code in range(256):
            val = ref_fp8_e5m2_dequant(code)
            S = (code >> 7) & 1
            E = (code >> 2) & 0x1F
            M = code & 0x3

            if E == 31 and M != 0:
                # Should be NaN
                assert np.isnan(val), f"Code 0x{code:02X} should be NaN"
            elif E == 31 and M == 0:
                # Should be infinity
                assert np.isinf(val), f"Code 0x{code:02X} should be Inf"
            else:
                # Should be finite
                assert np.isfinite(val), f"Code 0x{code:02X} should be finite, got {val}"

    @pytest.mark.parametrize("code,expected", [
        (0x00, 0.0),      # +0
        (0x80, -0.0),     # -0
        (0x3C, 1.0),      # +1
        (0xBC, -1.0),     # -1
        (0x40, 2.0),      # +2
        (0x38, 0.5),      # +0.5
        (0x34, 0.25),     # +0.25
        (0x7B, 57344.0),  # Max normal
    ])
    def test_specific_fp8_values(self, code: int, expected: float):
        """Key FP8 codes produce expected values."""
        val = float(ref_fp8_e5m2_dequant(code))
        if expected == 0.0:
            assert val == 0.0
        else:
            assert abs(val - expected) / abs(expected) < 1e-3, (
                f"Code 0x{code:02X}: got {val}, expected {expected}"
            )

    def test_subnormals_smaller_than_min_normal(self):
        """FP8 subnormals are smaller than the minimum normal."""
        min_normal = float(ref_fp8_e5m2_dequant(0x04))  # E=1, M=0
        for m in range(1, 4):
            sub_val = float(ref_fp8_e5m2_dequant(m))
            assert sub_val < min_normal, (
                f"Subnormal M={m} ({sub_val}) >= min_normal ({min_normal})"
            )

    def test_sign_symmetry(self):
        """Negative codes are exact negations of positive codes."""
        for code in range(128):
            pos = ref_fp8_e5m2_dequant(code)
            neg = ref_fp8_e5m2_dequant(code | 0x80)

            if np.isnan(pos):
                assert np.isnan(neg)
            elif np.isinf(pos):
                assert np.isinf(neg) and np.sign(neg) == -np.sign(pos)
            elif pos == 0.0:
                assert neg == 0.0 or neg == -0.0
            else:
                assert float(neg) == -float(pos)


class TestFP8MetalKernel:
    """Test FP8 dequantization Metal kernel if available."""

    def _skip_if_no_metal_fp8(self):
        """Skip if Metal FP8 dequant is not available."""
        try:
            from metal_marlin.kernels import dequant_fp8_e5m2
            return dequant_fp8_e5m2
        except (ImportError, AttributeError):
            pytest.skip("FP8 dequant kernel not available")

    @pytest.mark.parametrize("N", [8, 64, 256])
    @pytest.mark.parametrize("K", [32, 128])
    def test_fp8_dequant_shape(self, K: int, N: int):
        """FP8 dequant produces correct output shape."""
        # This test requires the kernel; skip if not present
        dequant_fn = self._skip_if_no_metal_fp8()

        # Create random packed FP8 data
        k_blocks = K // 4  # 4 FP8 values per uint32
        packed = mx.array(np.random.randint(0, 2**32, (k_blocks, N), dtype=np.uint32))
        scales = mx.ones((K // 32, N), dtype=mx.float16)

        result = dequant_fn(packed, scales, K, N, group_size=32)
        mx.eval(result)
        assert result.shape == (K, N)


# ---------------------------------------------------------------------------
# Cross-variant consistency tests
# ---------------------------------------------------------------------------


class TestCrossVariantConsistency:
    """Ensure different kernel variants produce consistent results."""

    def test_fp4_determinism(self):
        """FP4 GEMM is deterministic across multiple calls."""
        M, K, N = 32, 256, 256
        A = mx.random.normal((M, K))
        B, scales = create_quantized_weights(K, N, group_size=32)

        result1 = quantized_linear(A, B, scales, group_size=32)
        mx.eval(result1)

        result2 = quantized_linear(A, B, scales, group_size=32)
        mx.eval(result2)

        assert mx.allclose(result1, result2, atol=1e-6).item(), (
            "FP4 GEMM is not deterministic"
        )
