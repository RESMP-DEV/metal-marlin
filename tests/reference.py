"""
Reference (non-optimized) implementations for testing.
"""
import numpy as np


def dequant_fp4_reference(
    packed: np.ndarray,
    scales: np.ndarray,
    group_size: int = 128,
) -> np.ndarray:
    """
    Reference FP4 E2M1 dequantization.

    FP4 E2M1 encoding:
    - 1 sign bit, 2 exponent bits, 1 mantissa bit
    - Bias = 1
    - No denormals, no infinity/NaN
    """
    # Unpack 4-bit values
    K8, N = packed.shape
    K = K8 * 8

    unpacked = np.zeros((K, N), dtype=np.float32)

    for k8 in range(K8):
        word = packed[k8, :]
        for i in range(8):
            fp4_code = (word >> (i * 4)) & 0xF

            # Decode FP4 E2M1
            sign = (fp4_code >> 3) & 1
            exp = (fp4_code >> 1) & 0x3
            mant = fp4_code & 1

            if exp == 0:
                # Subnormal: (-1)^S * 0.5 * M
                # M=0 -> zero, M=1 -> +/- 0.5
                val = 0.5 * mant
                if sign:
                    val = -val
            else:
                # Normal: (-1)^S * 2^(E-1) * (1 + M*0.5)
                val = (1.0 + mant * 0.5) * (2.0 ** (exp - 1))
                if sign:
                    val = -val

            unpacked[k8 * 8 + i, :] = val

    # Apply scales per group
    num_groups = K // group_size
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        unpacked[start:end, :] *= scales[g, :]

    return unpacked.astype(np.float16)


def dequant_int4_reference(
    packed: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    group_size: int = 128,
) -> np.ndarray:
    """
    Reference INT4 (unsigned with zero-point) dequantization.

    Formula: fp_val = scale * (int4_val - zero_point)
    """
    K8, N = packed.shape
    K = K8 * 8

    unpacked = np.zeros((K, N), dtype=np.float32)

    for k8 in range(K8):
        word = packed[k8, :]
        for i in range(8):
            int4_val = (word >> (i * 4)) & 0xF
            unpacked[k8 * 8 + i, :] = int4_val

    # Apply scales and zeros per group
    num_groups = K // group_size
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        unpacked[start:end, :] = (
            unpacked[start:end, :] - zeros[g, :]
        ) * scales[g, :]

    return unpacked.astype(np.float16)


def gemm_reference(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """FP16 GEMM reference using FP32 accumulation."""
    return (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
