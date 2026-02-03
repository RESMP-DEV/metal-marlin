#!/usr/bin/env python3
"""
Kernel Variant Registry for A/B Testing.

Defines variant configurations for Metal kernels that can be compared.
Each variant represents a different optimization strategy or configuration
that can be benchmarked and statistically compared.

Variant Categories:
1. Tile size variations (TILE_M, TILE_N, TILE_K)
2. Buffer count variations (single vs double buffering)
3. Vectorization strategies (scalar vs SIMD)
4. Memory access patterns (coalesced vs strided)
5. Accumulator precision (FP16 vs FP32)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class KernelType(Enum):
    """Types of kernels that can be tested."""

    GEMM_TRELLIS = "gemm_trellis"
    GEMM_FP4 = "gemm_fp4"
    ATTENTION_SIMD = "attention_simd"
    MOE_DISPATCH = "moe_dispatch"
    DEQUANT_TRELLIS = "dequant_trellis"


class OptimizationType(Enum):
    """Types of optimizations that can be compared."""

    TILE_SIZE = "tile_size"
    BUFFER_STRATEGY = "buffer_strategy"
    VECTORIZATION = "vectorization"
    ACCUMULATOR_PRECISION = "accumulator_precision"
    MEMORY_LAYOUT = "memory_layout"


@dataclass
class KernelConfig:
    """Configuration for a kernel variant."""

    name: str
    description: str
    kernel_type: KernelType
    optimization_type: OptimizationType

    # Compile-time constants (must match Metal kernel)
    tile_m: int = 64
    tile_n: int = 64
    tile_k: int = 32
    num_buffers: int = 2
    simdgroups_per_threadgroup: int = 4

    # Runtime parameters
    use_fp32_accumulator: bool = False
    use_texture_cache: bool = False

    # Metal shader function name
    shader_function: str = "gemm_trellis_packed"

    # Additional parameters for dispatch
    extra_params: dict[str, int | bool | str] = field(default_factory=dict)

    def __post_init__(self):
        # Validate tile dimensions are multiples of 8 for simdgroup_matrix
        assert self.tile_m % 8 == 0, f"tile_m must be multiple of 8, got {self.tile_m}"
        assert self.tile_n % 8 == 0, f"tile_n must be multiple of 8, got {self.tile_n}"
        assert self.tile_k % 8 == 0, f"tile_k must be multiple of 8, got {self.tile_k}"


# =============================================================================
# GEMM Trellis Kernel Variants
# =============================================================================

GEMM_TRELLIS_VARIANTS: dict[str, KernelConfig] = {
    # Baseline configuration
    "baseline_64x64x32": KernelConfig(
        name="baseline_64x64x32",
        description="Baseline 64x64x32 tiles with FP16 accumulation",
        kernel_type=KernelType.GEMM_TRELLIS,
        optimization_type=OptimizationType.TILE_SIZE,
        tile_m=64,
        tile_n=64,
        tile_k=32,
        shader_function="gemm_trellis_packed",
    ),
    # FP32 accumulation variant
    "fp32_acc_64x64x32": KernelConfig(
        name="fp32_acc_64x64x32",
        description="64x64x32 tiles with FP32 accumulation for stability",
        kernel_type=KernelType.GEMM_TRELLIS,
        optimization_type=OptimizationType.ACCUMULATOR_PRECISION,
        tile_m=64,
        tile_n=64,
        tile_k=32,
        use_fp32_accumulator=True,
        shader_function="gemm_trellis_packed_fp32acc",
    ),
    # Decode-optimized variant (smaller M, larger N)
    "decode_32x128x32": KernelConfig(
        name="decode_32x128x32",
        description="Decode-optimized 32x128x32 for small batch inference",
        kernel_type=KernelType.GEMM_TRELLIS,
        optimization_type=OptimizationType.TILE_SIZE,
        tile_m=32,
        tile_n=128,
        tile_k=32,
        shader_function="gemm_trellis_packed_decode",
    ),
    # Large tile variant for prefill
    "prefill_128x64x32": KernelConfig(
        name="prefill_128x64x32",
        description="Prefill-optimized 128x64x32 for large batch",
        kernel_type=KernelType.GEMM_TRELLIS,
        optimization_type=OptimizationType.TILE_SIZE,
        tile_m=128,
        tile_n=64,
        tile_k=32,
        # Note: This would require a new shader variant
        shader_function="gemm_trellis_packed",
        extra_params={"tile_m_override": 128},
    ),
    # Texture-cached grid lookup
    "texture_grid_64x64x32": KernelConfig(
        name="texture_grid_64x64x32",
        description="64x64x32 with texture cache for codebook",
        kernel_type=KernelType.GEMM_TRELLIS,
        optimization_type=OptimizationType.MEMORY_LAYOUT,
        tile_m=64,
        tile_n=64,
        tile_k=32,
        use_texture_cache=True,
        shader_function="gemm_trellis_packed",
        extra_params={"use_texture_grid": True},
    ),
}

# =============================================================================
# MoE Dispatch Kernel Variants
# =============================================================================

MOE_DISPATCH_VARIANTS: dict[str, KernelConfig] = {
    "fused_32x64x32": KernelConfig(
        name="fused_32x64x32",
        description="Fused dispatch with 32x64x32 tiles",
        kernel_type=KernelType.MOE_DISPATCH,
        optimization_type=OptimizationType.TILE_SIZE,
        tile_m=32,
        tile_n=64,
        tile_k=32,
        shader_function="moe_dispatch_fused",
    ),
    "grouped_32x64x32": KernelConfig(
        name="grouped_32x64x32",
        description="Pre-grouped token dispatch",
        kernel_type=KernelType.MOE_DISPATCH,
        optimization_type=OptimizationType.MEMORY_LAYOUT,
        tile_m=32,
        tile_n=64,
        tile_k=32,
        shader_function="moe_dispatch_grouped",
    ),
}

# =============================================================================
# Attention Kernel Variants
# =============================================================================

ATTENTION_VARIANTS: dict[str, KernelConfig] = {
    "simdgroup_32x64": KernelConfig(
        name="simdgroup_32x64",
        description="SIMD-group tiled attention (32 query rows, 64 KV rows)",
        kernel_type=KernelType.ATTENTION_SIMD,
        optimization_type=OptimizationType.TILE_SIZE,
        tile_m=32,
        tile_n=64,
        tile_k=128,  # head_dim
        shader_function="simdgroup_attention",
    ),
    "flash_8x24": KernelConfig(
        name="flash_8x24",
        description="Flash-style fused attention (8 queries, 24 KV per tile)",
        kernel_type=KernelType.ATTENTION_SIMD,
        optimization_type=OptimizationType.TILE_SIZE,
        tile_m=8,
        tile_n=24,
        tile_k=128,
        shader_function="simdgroup_attention",
        extra_params={"flash_style": True},
    ),
}

# =============================================================================
# Problem Size Configurations
# =============================================================================


@dataclass
class ProblemSize:
    """Problem size for kernel benchmarking."""

    name: str
    M: int
    N: int
    K: int
    description: str


# Standard LLM problem sizes
LLM_PROBLEM_SIZES: list[ProblemSize] = [
    # Decode (small batch)
    ProblemSize("decode_1", 1, 4096, 4096, "Single token decode"),
    ProblemSize("decode_8", 8, 4096, 4096, "8-token decode batch"),
    ProblemSize("decode_32", 32, 4096, 4096, "32-token decode batch"),
    # Prefill (larger batch)
    ProblemSize("prefill_128", 128, 4096, 4096, "128-token prefill"),
    ProblemSize("prefill_512", 512, 4096, 4096, "512-token prefill"),
    ProblemSize("prefill_2048", 2048, 4096, 4096, "2048-token prefill"),
    # MLP shapes
    ProblemSize("mlp_up_8b", 1, 14336, 4096, "Llama-3-8B up_proj"),
    ProblemSize("mlp_down_8b", 1, 4096, 14336, "Llama-3-8B down_proj"),
    # MoE shapes
    ProblemSize("moe_expert", 16, 14336, 4096, "MoE expert (16 tokens)"),
]

# Attention-specific problem sizes
ATTENTION_PROBLEM_SIZES: list[ProblemSize] = [
    ProblemSize("attn_short", 128, 128, 128, "Short context (128 tokens)"),
    ProblemSize("attn_medium", 512, 512, 128, "Medium context (512 tokens)"),
    ProblemSize("attn_long", 2048, 2048, 128, "Long context (2048 tokens)"),
    ProblemSize("attn_very_long", 8192, 8192, 128, "Very long context (8K tokens)"),
]


# =============================================================================
# Variant Selection Utilities
# =============================================================================


def get_variants_for_kernel(kernel_type: KernelType) -> dict[str, KernelConfig]:
    """Get all variants for a specific kernel type."""
    if kernel_type == KernelType.GEMM_TRELLIS:
        return GEMM_TRELLIS_VARIANTS
    elif kernel_type == KernelType.MOE_DISPATCH:
        return MOE_DISPATCH_VARIANTS
    elif kernel_type == KernelType.ATTENTION_SIMD:
        return ATTENTION_VARIANTS
    else:
        return {}


def get_recommended_pairs(kernel_type: KernelType) -> list[tuple[str, str, str]]:
    """
    Get recommended A/B test pairs for a kernel type.

    Returns list of (test_name, variant_a, variant_b) tuples.
    """
    if kernel_type == KernelType.GEMM_TRELLIS:
        return [
            ("precision", "baseline_64x64x32", "fp32_acc_64x64x32"),
            ("decode_tiles", "baseline_64x64x32", "decode_32x128x32"),
            ("texture_cache", "baseline_64x64x32", "texture_grid_64x64x32"),
        ]
    elif kernel_type == KernelType.MOE_DISPATCH:
        return [
            ("dispatch_strategy", "fused_32x64x32", "grouped_32x64x32"),
        ]
    elif kernel_type == KernelType.ATTENTION_SIMD:
        return [
            ("tile_size", "simdgroup_32x64", "flash_8x24"),
        ]
    return []


def print_available_variants() -> None:
    """Print all available kernel variants."""
    print("Available Kernel Variants")
    print("=" * 60)

    for kernel_type in KernelType:
        variants = get_variants_for_kernel(kernel_type)
        if variants:
            print(f"\n{kernel_type.value}:")
            for name, config in variants.items():
                print(f"  {name}:")
                print(f"    {config.description}")
                print(f"    Tiles: {config.tile_m}x{config.tile_n}x{config.tile_k}")
                print(f"    Shader: {config.shader_function}")


if __name__ == "__main__":
    print_available_variants()

    print("\n" + "=" * 60)
    print("Recommended A/B Test Pairs")
    print("=" * 60)

    for kernel_type in KernelType:
        pairs = get_recommended_pairs(kernel_type)
        if pairs:
            print(f"\n{kernel_type.value}:")
            for test_name, va, vb in pairs:
                print(f"  {test_name}: {va} vs {vb}")
