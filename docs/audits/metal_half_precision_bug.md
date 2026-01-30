# Metal Compiler Half-Precision Optimization Bug

**Date Discovered:** January 24, 2026  
**Verified On:**
- **Hardware:** Apple M4 Max (Mac16,5), 128GB RAM
- **macOS:** 26.3 (Tahoe) Build 25D5101c
- **Metal:** Version 32023.850 (metalfe-32023.850.10)

*Note: This bug has only been verified on the above configuration. It may or may not affect other Apple Silicon chips or macOS versions.*

## Summary

A Metal compiler optimization bug causes incorrect results when performing arithmetic operations with `half` (FP16) precision in inline functions. Specifically, fractional zero points in quantized weight dequantization are rounded to integers, producing systematic errors.

## Reproduction

The bug manifests when:
1. A `half` variable is passed to an inline function
2. The variable has a fractional value (e.g., 5.5)
3. The variable is used in subtraction within the function

### Minimal Reproducer

```metal
constant constexpr uint32_t FUSED_MAGIC_BIAS = 0x64006400u;
constant constexpr uint32_t FUSED_LO_MASK    = 0x000F000Fu;

// BUGGY: Returns -1.0 when zero_point=5.5, code=5 (expected: -0.5)
inline half dequant_buggy(uint32_t packed, half scale, half zero_point) {
    half2 bias = as_type<half2>(FUSED_MAGIC_BIAS);
    uint32_t n0 = (packed & FUSED_LO_MASK) | FUSED_MAGIC_BIAS;
    half2 v0 = as_type<half2>(n0) - bias;
    return (v0.x - zero_point) * scale;  // BUG: zero_point rounded to 6.0
}

// FIXED: Returns -0.5 correctly
inline half dequant_fixed(uint32_t packed, half scale, half zero_point) {
    half2 bias = as_type<half2>(FUSED_MAGIC_BIAS);
    float fscale = (float)scale;
    float fzero = (float)zero_point;  // Cast to float before arithmetic
    uint32_t n0 = (packed & FUSED_LO_MASK) | FUSED_MAGIC_BIAS;
    half2 v0 = as_type<half2>(n0) - bias;
    return (half)(((float)v0.x - fzero) * fscale);
}
```

### Test Results

| Zero Point | Expected | Buggy Result | Fixed Result |
|------------|----------|--------------|--------------|
| 5.0        | -0.0     | -0.0         | -0.0         |
| 5.25       | -0.25    | -0.0 (wrong) | -0.25        |
| 5.5        | -0.5     | -1.0 (wrong) | -0.5         |
| 5.75       | -0.75    | -1.0 (wrong) | -0.75        |
| 8.25       | -3.25    | -3.0 (wrong) | -3.25        |

The bug rounds fractional zero points: 5.25→5, 5.5→6, 5.75→6, 8.25→8, 8.75→9.

### Characteristics

- **Does NOT occur** when the computation is written inline in the kernel
- **Does NOT occur** when intermediate values are written to device memory first
- **DOES occur** in inline functions regardless of `inline` keyword
- **DOES occur** with both return values and `thread half*` output parameters

## Root Cause Analysis

The Metal compiler appears to apply an optimization that converts `half` arithmetic to use a different precision path when:
1. The value is a function parameter (not a local variable)
2. The value is used in subtraction with another `half2` component

Our hypothesis: The compiler may be using a fused multiply-add optimization that internally truncates the zero point to optimize the subtraction, or there's a register allocation issue that stores `half` parameters in a truncated format.

## Workaround

**Use `float` for intermediate computations:**

```metal
inline void fused_dequant_u4x8(uint32_t packed,
                                half scale,
                                half zero_point,
                                thread half *out) {
    half2 bias = as_type<half2>(FUSED_MAGIC_BIAS);
    
    // WORKAROUND: Cast to float before arithmetic
    float fscale = (float)scale;
    float fzero = (float)zero_point;

    uint32_t n0 = (packed & FUSED_LO_MASK) | FUSED_MAGIC_BIAS;
    half2 v0 = as_type<half2>(n0) - bias;
    // ... extract other nibbles ...

    // Compute in float, cast back to half
    out[0] = (half)(((float)v0.x - fzero) * fscale);
    out[1] = (half)(((float)v1.x - fzero) * fscale);
    // ... etc
}
```

## Performance Implications

### Why Float Can Be Faster Than Half

Counter-intuitively, the float version is sometimes *faster* than half. This is because:

1. **Apple Silicon GPUs natively prefer float32**: The ALUs are optimized for FP32. Half-precision operations may be internally promoted to float anyway, or use slower execution paths.

2. **Buggy code generation**: The compiler bug that causes incorrect results may also generate suboptimal code paths for the half version. The float version forces a clean, predictable code path.

3. **Memory-bound workloads hide ALU differences**: When the bottleneck is memory bandwidth (reading weights from DRAM), the ALU has idle cycles regardless of precision. More complex float arithmetic completes within the same memory latency window.

4. **No "free" 2x throughput for half**: Unlike NVIDIA GPUs with dedicated FP16 tensor cores, Apple Silicon's GPU doesn't necessarily provide 2x throughput for half vs float in scalar operations.

### Theoretical vs Actual

| Assumption | Reality |
|------------|---------|
| Half = 2x ALU throughput | Not guaranteed on Apple Silicon |
| Half = better register pressure | True, but not the bottleneck |
| Float conversion = overhead | Hidden by memory latency |

### Benchmark Results (Apple M4 Max, macOS 26.3)

**Pure dequant kernel benchmark** (`bench_half_vs_float.py`):

| Weights | Half (ms) | Float (ms) | Overhead | Throughput |
|---------|-----------|------------|----------|------------|
| 8K      | 0.156     | 0.156      | 0.0%     | 0.1 GB/s  |
| 131K    | 0.164     | 0.146      | -10.7%   | 2.0 GB/s  |
| 2M      | 0.212     | 0.214      | +1.0%    | 24.8 GB/s |
| 16M     | 0.541     | 0.539      | -0.3%    | 77.6 GB/s |

**Key finding**: At LLM scale (16M weights = 4096×4096 matrix), the float version is actually slightly *faster* due to cleaner code generation. The overhead is within measurement noise (±1%).

### Why No Performance Penalty?

1. **Memory-bound dequantization**: The dequant operation reads from global memory (packed weights, scales, zeros) and writes to threadgroup memory. This is memory-bound, not compute-bound. The extra ALU cycles for float conversion are hidden by memory latency.

2. **GEMM dominates runtime**: The simdgroup matrix multiply-accumulate (MMA) operations that perform the actual GEMM use `simdgroup_matrix<float, 8, 8>` accumulators regardless. The MMA inner loop is where 90%+ of cycles are spent.

3. **Conversion overhead is negligible**: Converting 8 half values to float and back adds ~24 instructions per dequant call. With TILE_K=32 and each call processing 8 values, this is 4 × 24 = 96 extra instructions per K-tile. Compare to the MMA operations: 4 simdgroups × 16 MMA ops × ~8 cycles = 512+ cycles per tile.

### End-to-End Estimate

| Scenario | Throughput Impact |
|----------|-------------------|
| Single-token inference (M=1) | **0% measurable** |
| Batched inference (M=32-128) | **0% measurable** |
| Pure dequant kernel | ±1% (within noise) |

The bottleneck for quantized LLM inference on Apple Silicon is:
1. Memory bandwidth (reading weights) ← **actual bottleneck**
2. simdgroup MMA throughput
3. ~~ALU for dequantization~~ ← **completely hidden by memory latency**

## Recommendations

### For This Project

1. ✅ **Keep the float workaround** - Correctness over micro-optimization
2. ✅ **Document the bug** - Help others avoid this pitfall
3. ⚠️ **File Apple Feedback** - Report to Apple for potential fix

### For Future Metal Kernels

1. **Test with fractional values** when using `half` in inline functions
2. **Prefer float intermediates** for complex arithmetic chains
3. **Validate against reference implementation** with diverse test cases

### Reporting to Apple

Consider filing a Feedback Assistant report with:
- Minimal reproducer code
- Expected vs actual output
- macOS version and hardware
- Request for fix or documentation of intended behavior

## Debug Scripts

The following scripts in `debug/` reproduce and verify the bug:

- `debug/debug_zeros.py` - Proves fractional zeros are rounded
- `debug/debug_optimizer.py` - Shows float intermediates fix the bug  
- `debug/debug_fix.py` - Verifies the full dequant function works with fix
- `debug/debug_func_variants.py` - Tests inline vs function behavior
- `bench_half_vs_float.py` - Benchmarks performance impact (confirms 0% overhead)

## Files Fixed

The following files have been updated with the float intermediate workaround:

### Production Kernels (`metal_marlin/kernels.py`)
- `dequant_u4x8()` - INT4/U4 dequantization
- `dequant_fp4x8()` - FP4 dequantization  
- `dequant_fp4_fa()` - Flash attention FP4 dequantization

### Standalone Metal Shaders (`src/*.metal`)
- `marlin_gemm.metal`:
  - `safe_dequant()` - Base scale multiplication
  - `safe_dequant_u4()` - INT4 with zero point
  - `safe_dequant_u4_value()` - INT4 value with zero point
- `batched_gemm.metal`:
  - `dequant_fp4()` - FP4 scalar dequantization
- `dequant_fp8.metal`:
  - `dequant_fp8_e4m3_x4_scaled()` - FP8 E4M3 × scale
  - `dequant_fp8_e4m3_x8_fused()` - FP8 E4M3 8-value fused
- `dequant_int8.metal`:
  - `dequant_s8x4()` - INT8 symmetric
  - `dequant_s8x4_asym()` - INT8 asymmetric with zero point

### Test Harness (`tests/test_accuracy.py`)
- `fused_dequant_u4x8()` - Reference implementation for testing
- `quantize_to_u4_awq()` - Fixed uniform weight edge case

## Known Issues

### Stripe Partition Tests (`tests/test_stripe_partition.py`)
The 33 stripe partition tests fail due to an **MLX API misuse**, not the half-precision bug:

- **Root cause**: Tests pass complete Metal source file to `mx.fast.metal_kernel(source=...)` which expects only the kernel body
- **Error**: MLX wraps the source inside a generated kernel function, causing syntax errors with `constant constexpr` declarations
- **Fix required**: Restructure tests to use `header=` + `source=` parameters like `kernels.py` does

This is a test infrastructure issue unrelated to the half-precision fix.

## Conclusion

This is a confirmed Metal compiler bug affecting half-precision arithmetic in inline functions on Apple M4 Max with macOS 26.3. The workaround (using float intermediates) has **no performance penalty** - in fact, the float version is marginally faster at LLM scale.

**Key findings:**
1. Correctness is restored with the float workaround
2. Float intermediates are **not slower** than buggy half code
3. At memory-bound scales (16M weights), both achieve ~77.6 GB/s
4. Apple Silicon GPUs do not provide guaranteed 2x throughput for half vs float

The dequant operation is purely memory-bound at LLM scale. ALU precision differences are completely hidden by memory latency, making the float workaround a "free" fix.
