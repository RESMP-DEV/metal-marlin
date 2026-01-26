# BF16 Compatibility Audit (metal_marlin)

Scope: `contrib/metal_marlin/src/bf16_compat.metal`

This audit focuses on BF16 conversion paths, the instruction-level work implied by the current code, and optimization opportunities. Instruction counts below are source-level estimates (bit ops, shifts, converts) and not the final ISA produced by the Metal compiler; the compiler may fuse, vectorize, or reorder operations.

## Conversion paths and estimated instruction counts

### Scalar helpers

- `bf16_to_float(bf16_t v)`
  - Path: BF16 -> FP32 (bitcast+shift)
  - Ops per element (est.): 1x zero-extend to uint, 1x shift (<<16), 1x bitcast to float

- `bf16_to_half(bf16_t v)`
  - Path: BF16 -> FP32 -> FP16
  - Ops per element (est.): `bf16_to_float` (3 ops) + 1x FP32->FP16 convert

- `bf16_from_half(half h)`
  - Path: FP16 -> FP32 -> BF16
  - Ops per element (est.): 1x FP16->FP32 convert + 1x bitcast to uint + 1x shift (>>16) + 1x truncate to ushort

- `bf16_from_float_rtz(float f)`
  - Path: FP32 -> BF16 (truncate)
  - Ops per element (est.): 1x bitcast to uint + 1x shift (>>16) + 1x truncate to ushort

- `bf16_from_float_rne(float f)`
  - Path: FP32 -> BF16 (RNE)
  - Ops per element (est., non-NaN/Inf):
    - 1x bitcast to uint
    - 1x shift + 1x AND for `exp_bits`
    - 1x compare
    - 1x shift + 1x AND for tie bit
    - 1x add (rounding)
    - 1x add to f_bits
    - 1x shift (>>16) + 1x truncate to ushort
    - NaN/Inf path adds a couple of AND/or ops

### Vector helpers

- `bf16x4_to_float4(ushort4 packed)`
  - Path: 4x (BF16 -> FP32)
  - Ops per element (est.): 1x zero-extend to uint + 1x shift + 1x bitcast
  - Total for 4 lanes: 4x shifts + 4x bitcasts + 4x zero-extends

- `float4_to_bf16x4_rtz(float4 vals)`
  - Path: 4x (FP32 -> BF16, truncate)
  - Ops per element (est.): 1x bitcast + 1x shift + 1x truncate to ushort
  - Total for 4 lanes: 4x bitcasts + 4x shifts + 4x truncates

- `float4_to_bf16x4_rne(float4 vals)`
  - Path: 4x (FP32 -> BF16, RNE)
  - Ops per element (est.): see `bf16_from_float_rne` above
  - Total for 4 lanes: ~4x (bitcast, shifts, ands, adds, compare, truncate)

- `bf16x4_to_half4(ushort4 packed)`
  - Path: BF16 -> FP32 -> FP16
  - Ops per element (est.): same as `bf16x4_to_float4` + 1x FP32->FP16 convert

### Load/store helpers used by GEMM integration

- `bf16_load_as_half8(...)`
  - Path: 8x BF16 -> FP32 -> FP16 via `bf16x4_to_half4`
  - Ops per element (est.): 1x load + 1x zero-extend + 1x shift + 1x bitcast + 1x FP32->FP16 convert
  - Total for 8 elements: 8 loads + 8 shifts + 8 bitcasts + 8 converts + packing overhead

- `bf16_store_from_half8(...)`
  - Path: 8x FP16 -> FP32 -> BF16 via `float4_to_bf16x4_rne`
  - Ops per element (est.): 1x FP16->FP32 convert + ~8-10 integer ops (RNE) + 1x store
  - Total for 8 elements: 8 converts + ~64-80 integer ops + 8 stores + packing overhead

### Bulk conversion kernels

- `bf16_to_half_kernel`: per element same as `bf16x4_to_half4`
- `half_to_bf16_kernel`: per element same as `float4_to_bf16x4_rne` after FP16->FP32
- `bf16_to_float_kernel`: per element same as `bf16x4_to_float4`
- `float_to_bf16_kernel`: per element same as `float4_to_bf16x4_rtz` or `_rne`

## Callers and GEMM usage

Grep results in `contrib/metal_marlin` show no in-repo callers of:
- `bf16_load_as_half8`
- `bf16_store_from_half8`
- `bf16x4_to_float4`
- `bf16_to_half_kernel`, `half_to_bf16_kernel`, `bf16_to_float_kernel`, `float_to_bf16_kernel`

These functions are only referenced in `contrib/metal_marlin/src/bf16_compat.metal`. No GEMM kernels in `contrib/metal_marlin/src` currently invoke these helpers directly. If GEMM uses BF16, it is likely via host-side dispatch of these kernels or via a separate Metal library not checked into this repo.

## Optimization opportunities

### 1) Direct BF16 <-> FP32 (single shift, no FP16 intermediate)

The BF16 <-> FP32 conversion can remain a single bit-shift with a bitcast. Use these in any path that currently goes through FP16 unnecessarily.

Proposed helpers (conceptual):

```metal
inline float4 bf16x4_to_float4_fast(ushort4 packed) {
    // Vectorized zero-extend + shift + bitcast.
    uint4 u = uint4(packed) << 16;
    return as_type<float4>(u);
}

inline ushort4 float4_to_bf16x4_rtz_fast(float4 vals) {
    // Vectorized bitcast + shift.
    uint4 u = as_type<uint4>(vals) >> 16;
    return ushort4(u);
}
```

Implications:
- Use `bf16x4_to_float4_fast` in any kernel that consumes BF16 into FP32 (no intermediate FP16).
- Use `float4_to_bf16x4_rtz_fast` when RNE is not required.
- For RNE, you can still keep `bf16_from_float_rne`, but consider vectorizing its internal ops (see below).

### 2) Avoid BF16 -> FP32 -> FP16 in `bf16_load_as_half8`

If the target compute is FP16, a direct BF16 -> FP16 conversion path can avoid the FP32 intermediate. This is not currently implemented. Options:

- **Option A (low risk, minimal change):** Keep FP32 conversion but make it vectorized and avoid scalar lane-by-lane conversions.
- **Option B (higher effort):** Implement BF16 -> FP16 bit conversion with special handling for exponent/mantissa mapping (including subnormals/NaN/Inf). This avoids the FP32 path but requires careful correctness work.

A practical short-term win is to keep FP32 but vectorize:

```metal
inline half4 bf16x4_to_half4_fast(ushort4 packed) {
    uint4 u = uint4(packed) << 16;
    return half4(as_type<float4>(u));
}
```

This still uses FP32 for the final `half4` conversion, but avoids per-lane scalar ops.

### 3) Avoid FP16 -> FP32 -> BF16 in `bf16_store_from_half8`

Similar to loads, you can vectorize the FP16->FP32 conversion and the RNE path. For RNE, vectorize per-lane bit ops to reduce scalar overhead.

Conceptual direction:

```metal
inline ushort4 float4_to_bf16x4_rne_fast(float4 vals) {
    // Implement RNE in vector form using uint4 lanes.
    uint4 f_bits = as_type<uint4>(vals);
    uint4 rounding = (1u << 15) + ((f_bits >> 16) & 1u);
    f_bits += rounding;
    return ushort4(f_bits >> 16);
}
```

You still need NaN/Inf handling; that can be implemented with vector compares and masked selects to preserve correctness.

### 4) Vectorized conversions using `simd_shuffle`

Current vector helpers build float4 or ushort4 from scalar lane accesses. A more SIMD-friendly approach is to use packed vector operations and `simd_shuffle` to expand 16-bit lanes into 32-bit lanes in bulk.

Conceptual approach (API details may vary by Metal version):

```metal
inline float4 bf16x4_to_float4_simd(ushort4 packed) {
    // Treat packed as two uints containing 4x16-bit values.
    uint2 u16 = as_type<uint2>(packed);

    // Use simd_shuffle to expand to 4x32-bit lanes.
    // mask selects the low/high 16-bit parts as needed.
    uint4 u32 = simd_shuffle(u16, /*mask to extract 4x16*/);

    u32 <<= 16;
    return as_type<float4>(u32);
}
```

If `simd_shuffle` is not available on the target Metal version, a plain `uint4(packed) << 16` is already better than scalar lane-by-lane work.

### 5) Use `metal::simd_broadcast` for shared scale factors

`bf16_compat.metal` does not currently apply scale factors, but if these conversions are used in dequant or epilog paths, broadcasting the shared scale into a vector once can reduce redundant scalar-to-vector conversions:

```metal
float scale = ...;  // per-tile or per-group scale
float4 scale_vec = metal::simd_broadcast(scale);
vals *= scale_vec;
```

This is most useful when the scale is shared across lanes or threads within a SIMD group and is applied to multiple vector operations.

## Summary of recommended changes

- Vectorize `bf16x4_to_float4` and `float4_to_bf16x4_rtz` by operating on `uint4` directly.
- Consider a vectorized RNE path for `float4_to_bf16x4_rne` (with NaN/Inf masking).
- Replace `bf16x4_to_half4` with a vectorized version to remove scalar lane accesses; defer direct BF16->FP16 bit conversion unless you need to avoid FP32 entirely.
- If BF16 conversion is used in GEMM epilog or dequant paths, use `simd_broadcast` for shared scales.

## Notes

- No in-repo GEMM kernel calls these helpers today, so the optimization impact depends on whether host code dispatches the conversion kernels or external Metal libraries include `bf16_compat.metal`.
- Any direct BF16 -> FP16 bit conversion must handle exponent bias, subnormals, NaN, and Inf correctly. If correctness risk is high, prefer vectorized BF16 -> FP32 -> FP16.
