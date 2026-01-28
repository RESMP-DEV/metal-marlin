# Mixed precision accumulation

This document describes the mixed precision hierarchy used in Metal Marlin GEMM kernels
and the rationale for FP32 accumulation. The goal is to preserve numerical stability while
maintaining throughput with quantized weights.

## GEMM precision hierarchy

1. Weights: FP4/INT4/FP8 (quantized)
2. Activations: FP16 (input)
3. Intermediate: FP16 (dequantized weights)
4. Accumulation: FP32 (dot product sums)
5. Output: FP16 (final result)

## Why FP32 accumulation matters

FP32 accumulation is critical for:

- Large K dimension (>2048)
- Avoiding overflow in dot products
- Numerical stability when summing long reduction chains

Even when weights and activations are low precision, the accumulation path dominates error
for large K. Keeping the accumulator in FP32 prevents premature saturation and reduces
catastrophic cancellation.

## Metal implementation note

Metal's `simdgroup_matrix` performs FP32 accumulation internally when the accumulator type
is `float`, even if inputs are `half`. Convert to FP16 only at the final store.

```metal
// simdgroup_matrix uses FP32 accumulation internally
simdgroup_matrix<float, 8, 8> C_acc;  // FP32 accumulator

// Convert to FP16 only at final store
for output in C_acc:
    C[...] = half(output);
```

## Testing guidance

Verify that large-K reductions do not overflow:

- Use a pathological test with all-ones weights and activations.
- Set K = 32768 and confirm that the output remains finite.
- Validate that switching accumulation to FP16 causes overflow or large error.

This test ensures the accumulation path is FP32 and guards against accidental regressions
in the kernel or in later refactors.
