# Paged Attention Metal Kernel: Correctness Investigation Plan

**Date:** April 16, 2026
**Target Kernel:** `kernels-community/paged-attention` (v1)
**File:** `paged-attention-metal/attention/paged_attention.metal`
**Upstream:** `mistral.rs` / `EricB/kernels-paged-attention-metal` (June 2025)
**Consumers:** `mistral.rs`, experimental `vllm-project/vllm-metal` fork

**Investigator Hardware (to be filled in during reproduction):**
- Apple M4 Max (Mac16,5), 128GB RAM
- macOS 26.3 (Tahoe) Build 25D5101c
- Metal 32023.850 (metalfe-32023.850.10)

*Note: Findings will be scoped to the above configuration. Other Apple Silicon chips and macOS versions may or may not exhibit the same behavior.*

## Summary

This kernel is a mechanical Metal port of the vLLM CUDA paged attention kernel, with the `Vec<T,N>` / `FloatVec` / `Qk_dot` template machinery transliterated largely unchanged. The port does not account for two previously documented Metal compiler foot-guns on M4 Max / macOS 26.3:

1. **Half-precision arithmetic miscompilation in inline functions with half parameters** (see `metal_half_precision_bug.md`). Fractional half values passed as function parameters and subjected to arithmetic are silently rounded in some code paths.
2. **Array-reference parameter miscompilation** (see `metal_array_parameter_bugs.md`). Functions receiving `T (&)[N]`-style array references may not be inlined even with the `inline` keyword, causing all indexed accesses to collapse to index 0.

The kernel does not use `__attribute__((always_inline))` anywhere, does not cast half parameters to float before arithmetic in the `mul<half,...>` / `fma(half,...)` overloads, and threads array references through template-heavy code paths (`qk_dot_`, `Qk_dot::dot`) that are prime candidates for both bugs. The upstream test suite was adapted from vLLM's CUDA tests and validates numerical equivalence with `allclose` at fp16-typical tolerance (roughly 1e-3), which is not tight enough to detect the drift patterns these compiler bugs produce.

This document lays out a minimal reproduction plan to determine whether correctness is actually compromised on M4 Max / macOS 26.3, and in which code paths. The investigation is structured so each hypothesis has a dedicated, isolated test that cannot be explained away by other factors.

## Background: Why This Matters

The label "Paged attention kernels from vLLM and mistral.rs" on the Hugging Face model card implies broader validation than actually exists. In reality:

- Mainline vLLM has no Metal backend. The `vllm-project/vllm-metal` repo is a separate experimental fork by unaffiliated contributors.
- MLX has no PagedAttention primitive. The "Updated from MLX commit hash f70764a" comment at the top of the file most likely references MLX's bfloat shim patterns, not a PagedAttention implementation.
- The kernel's primary production consumer is `mistral.rs`, which inherits any correctness issues silently because its reference path is the same kernel on CUDA.

If this kernel drifts on Apple Silicon, every downstream deployment inherits the drift. Users running quantized models (especially FP8 KV cache) on Mac would see degraded generation quality attributed to quantization when the actual cause is compiler miscompilation of the attention path itself.

Extending this to 4-bit weight schemes such as MXFP4 amplifies the problem: dequantization tolerance budgets are much tighter, and any systematic fractional-value rounding in the attention accumulator compounds with quantization noise instead of averaging out.

## Hypothesis Enumeration

### H1: Half-precision parameter bug is latent in `fma(half, half, float)` overload

**Location:** Lines defining `inline float fma(half a, half b, float c) { return (float)a * (float)b + c; }` and its vector variants.

**Trigger condition:** These overloads take `half` by value and cast to float inside the body. The bug from `metal_half_precision_bug.md` fires when the compiler applies an optimization that uses the half representation before the cast. The cast-inside-the-body pattern is exactly the shape that report flagged as unreliable; the working pattern required casting to float at the call site and passing float through.

**Expected symptom if bug fires:** QK dot products whose exact fp16 representation is near half-integer boundaries (e.g., scores around 5.5 before scaling) land on the wrong integer. After softmax, this produces systematically skewed attention weights.

**Test:** Construct Q and K vectors whose unscaled dot products land on known fractional values in fp16. Run through `paged_attention<half, half, ...>`. Compare against a float32 reference computed on CPU.

### H2: Array-reference parameter bug fires in `qk_dot_`

**Location:**
```metal
template <int THREAD_GROUP_SIZE, typename Vec, int N>
inline float qk_dot_(const threadgroup Vec (&q)[N], const thread Vec (&k)[N]) {
    A_vec qk_vec = mul<A_vec, Vec, Vec>(q[0], k[0]);
#pragma unroll
    for (int ii = 1; ii < N; ++ii) {
        qk_vec = fma(q[ii], k[ii], qk_vec);
    }
    ...
}
```

**Trigger condition:** `q` and `k` are array references passed as template parameters. The `#pragma unroll` might save it if the compiler fully unrolls and constant-folds, but if the loop is lowered as a real function call (not inlined), the reference decay ABI collapses all indices to `[0]`, matching the column-repetition symptom from `metal_array_parameter_bugs.md`.

**Expected symptom if bug fires:** Every QK dot product within a given thread group becomes `q[0] * k[0] * N` instead of the actual dot product. Output would be catastrophically wrong, not subtly wrong, so this is likely NOT firing at the default optimization level (otherwise `mistral.rs` would be visibly broken). But it may fire at specific template instantiations or under pressure.

**Test:** Inspect generated AIR / metallib for a representative instantiation (e.g., `paged_attention_half_cache_half_hs128_bs16_nt256_nsl32_ps0`). Determine whether `qk_dot_` is actually inlined. If not, construct inputs that would distinguish correct vs index-0-collapse behavior.

### H3: `Qk_dot::dot` static member wrapper adds a second call boundary

**Location:**
```metal
template <typename T, int THREAD_GROUP_SIZE> struct Qk_dot {
    template <typename Vec, int N>
    static inline float dot(const threadgroup Vec (&q)[N], const thread Vec (&k)[N]) {
        return qk_dot_<THREAD_GROUP_SIZE>(q, k);
    }
};
```

**Trigger condition:** Even if `qk_dot_` inlines, the `Qk_dot::dot` wrapper is another inline function taking array references. Two nested inline-with-reference calls may behave differently from one.

**Test:** Differential test between direct calls to `qk_dot_` and calls via `Qk_dot::dot`, with identical inputs.

### H4: FP8 KV cache path has a distinct bug surface

**Location:** `fp8_convert<K_vec, Quant_vec>(k_vec_quant, *k_scale)` call sites inside the main kernel.

**Trigger condition:** The FP8 path performs `fp8_e4m3_to_float(v) * scale` with scale being dereferenced as `float` from device memory, then casts to half/bfloat16 via constructor. This pattern is structurally safe (all arithmetic in float), but:
- If `*k_scale` or `*v_scale` happen to equal fractional values that hit the half-precision bug after the final cast, there could be drift.
- The cast-back-to-half at the end of `fp8_convert<Half8_, Uchar8_>` produces values that will then be multiplied in half by subsequent `mul<half4, half4, half4>` calls, potentially re-entering the bug surface.

**Test:** Compare FP8 cache output against fp16 cache output on identical logical data (same values, different storage format). Any systematic divergence beyond the expected FP8 quantization noise floor is suspect.

### H5: V-accumulation path is safe but worth confirming

**Location:** The V-side accumulation uses `float accs[NUM_ROWS_PER_THREAD]` and `dot(logits_vec, v_vec)` where `dot` dispatches to `sum(mul<A,T,T>(a,b))`. For `T = half4`, `A = float4`, this calls `mul<float4, half4, half4>(half4, half4) { return (float4)a * (float4)b; }` which casts to float at the call site.

**Expected behavior:** Safe by construction. Confirm via direct test.

## Reproduction Scripts

### Script 1: Kernel Acquisition and Basic Sanity

```bash
# Clone the kernel package
pip install kernels
python -c "from kernels import get_kernel; m = get_kernel('kernels-community/paged-attention', version=1); print(dir(m))"

# Locate the metallib in the installed package
find ~/.cache/huggingface -name "*.metallib" -path "*paged*"

# Extract kernel names from the metallib
xcrun metal-nm <metallib_path> | grep -i paged_attention | head -40
```

### Script 2: Minimal Reproducer for H1 (Half-Precision Path)

The goal is to construct Q and K vectors whose dot product lands on exactly fractional values known to trigger the `metal_half_precision_bug.md` pattern after scale multiplication.

```python
# test_h1_half_fma_bug.py
import numpy as np
import mlx.core as mx
from kernels import get_kernel

paged = get_kernel("kernels-community/paged-attention", version=1)

# Construct a minimal scenario: 1 sequence, 1 head, 1 KV head, head_size=128
# Single KV block of size 16. Query designed so that dot product with a
# specific K entry lands on a value that fp16 cannot represent exactly
# AND is near a half-integer boundary.

num_seqs = 1
num_heads = 1
num_kv_heads = 1
head_size = 128
block_size = 16
context_len = 16

# Build Q so each element is 1/sqrt(head_size) in fp16
# Build K[0] so dot(Q, K[0]) = 5.5 (fractional, fires H1 if it exists)
# Build K[1..15] so their dots are well-separated from 5.5

q = np.full((num_seqs, num_heads, head_size), 1.0 / np.sqrt(head_size), dtype=np.float16)

# Target dot product values that, after scale=1.0, land on problem values
target_dots = [5.5, 5.25, 8.25, 8.75, 3.0, 2.0, 1.0, 0.5,
               0.25, 0.125, 4.5, 6.5, 7.5, 9.5, 10.0, 11.0]

k_cache_logical = np.zeros((context_len, head_size), dtype=np.float16)
for i, target in enumerate(target_dots):
    # k[i] scaled so dot(q, k[i]) = target
    # dot(q, k) = (1/sqrt(D)) * sum(k) => sum(k) = target * sqrt(D)
    k_cache_logical[i, 0] = target * np.sqrt(head_size)

# Compute reference softmax in float64
scale = 1.0
dots = (q[0, 0].astype(np.float64) @ k_cache_logical.astype(np.float64).T) * scale
ref_weights = np.exp(dots - dots.max())
ref_weights /= ref_weights.sum()

print("Reference softmax weights:")
for i, (target, w) in enumerate(zip(target_dots, ref_weights)):
    print(f"  token {i:2d}: dot={target:6.3f}  weight={w:.8f}")

# Now invoke the kernel and compare
# (Specific MLX dispatch code depends on kernel's Python binding.
#  For mistral.rs, route through its PyO3 bindings; for kernels package,
#  use the provided Python API.)

# [kernel invocation placeholder]
# kernel_out = paged.paged_attention_v1(...)

# Compare
# kernel_weights_effective = ...
# drift = np.abs(kernel_weights_effective - ref_weights)
# print(f"Max drift: {drift.max():.6e}")
# print(f"Mean drift: {drift.mean():.6e}")

# If H1 fires: tokens with target=5.5, 5.25, 8.25, 8.75 should show
# disproportionately large drift compared to integer-valued targets.
```

### Script 3: AIR Inspection for H2 (Array-Reference Inlining)

```bash
# Extract a single kernel instantiation from the metallib and disassemble to AIR
xcrun metal-nm <metallib_path> | grep paged_attention_half_cache_half_hs128_bs16_nt256_nsl32_ps0
xcrun metal-objdump -disassemble <metallib_path> > paged_attention_disasm.txt

# Search for evidence of qk_dot_ as a called function vs fully inlined
grep -i "qk_dot\|call\|bl " paged_attention_disasm.txt | head -100

# Also check for the characteristic pattern of array-reference passing:
# if arguments include register pairs for (base_ptr, length), the function
# is being called through the reference ABI. If the loop body is inlined
# with direct indexed loads from threadgroup memory, inlining succeeded.
```

### Script 4: Differential Test for H4 (FP8 vs FP16 Cache)

```python
# test_h4_fp8_vs_fp16_drift.py
import numpy as np
from kernels import get_kernel

paged = get_kernel("kernels-community/paged-attention", version=1)

# Same Q, same logical KV values, stored once as fp16 and once as FP8 E4M3.
# Expected drift = FP8 quantization noise floor (known).
# Observed drift > expected = bug in fp8 path.

# Use realistic but controlled inputs
rng = np.random.default_rng(42)
q = rng.standard_normal((1, 8, 128)).astype(np.float16) * 0.1  # 8 heads
k_fp16 = rng.standard_normal((1, 8, 128, 256)).astype(np.float16) * 0.1
v_fp16 = rng.standard_normal((1, 8, 128, 256)).astype(np.float16) * 0.1

# Quantize to FP8 E4M3 with known scale
k_scale = np.float32(np.abs(k_fp16).max() / 448.0)  # E4M3 max = 448
v_scale = np.float32(np.abs(v_fp16).max() / 448.0)

def quantize_fp8_e4m3(x, scale):
    # Simulate FP8 E4M3 roundtrip in numpy (use ml_dtypes if available)
    import ml_dtypes
    x_scaled = x / scale
    x_fp8 = x_scaled.astype(ml_dtypes.float8_e4m3fn)
    return x_fp8.astype(np.uint8).view(np.uint8)

k_fp8 = quantize_fp8_e4m3(k_fp16, k_scale)
v_fp8 = quantize_fp8_e4m3(v_fp16, v_scale)

# Run kernel twice: once with fp16 cache, once with fp8 cache + scales
# Compute attention on both paths; compare

# [kernel dispatch for both paths]

# Expected: FP8 drift stays within E4M3 noise floor (roughly 1/256 relative)
# If drift exceeds this systematically, fp8_convert or subsequent half mul
# is corrupting values beyond quantization noise.
```

### Script 5: Direct-Call vs Wrapped-Call Differential for H3

This requires compiling a modified copy of the kernel file with an exposed variant that calls `qk_dot_` directly, bypassing `Qk_dot::dot`. Output the per-token QK values to a debug buffer and compare bit-exact.

```metal
// Add this variant kernel for differential testing only
[[kernel]] void paged_attention_debug_direct_qkdot(
    // ... same signature but write per-token qk to debug buffer
) {
    // same as paged_attention but replace:
    //   Qk_dot<T, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs)
    // with:
    //   qk_dot_<THREAD_GROUP_SIZE>(q_vecs[thread_group_offset], k_vecs)
    // and dump qk to a device buffer before applying softcapping/alibi
}
```

Compare per-token QK values between the two variants. Any bit-exact mismatch confirms H3.

## Success Criteria for Each Hypothesis

| Hypothesis | Confirmed if | Refuted if |
|---|---|---|
| H1 (half fma bug) | Drift on fractional-dot tokens > drift on integer-dot tokens by >10x | Drift distribution is uniform across test cases |
| H2 (array-ref bug) | AIR shows `qk_dot_` as an emitted function call, OR kernel output matches index-0-collapse signature | AIR shows full inlining; outputs match fp32 reference to within fp16 noise |
| H3 (wrapper layer) | Direct-call variant differs bit-exactly from wrapped-call variant | Bit-exact match between variants |
| H4 (fp8 path) | FP8 vs FP16 drift exceeds E4M3 quantization noise floor by >2x systematically | Drift within expected FP8 noise floor |
| H5 (v-accum) | Unexpected drift in V-side reduction | Matches reference within fp16 noise |

## Anticipated Results (Predictions, Not Observations)

Based on the structure of the kernel and the patterns documented in the prior bug reports:

- **H1 is most likely to partially fire.** The `fma(half, half, float)` pattern does the float cast inside the function body rather than forcing float through the boundary. On M4 Max / macOS 26.3, this is exactly the pattern that produced fractional-zero-point rounding. Expected symptom is mild: drift bounded by fp16 ULP on most tokens, but specific fractional values producing larger errors. Likely invisible in perplexity measurements, visible in raw logit comparisons.

- **H2 is most likely to NOT fire in the common case,** because if it did, mistral.rs would be visibly broken and someone would have noticed. More likely is that specific template instantiations (particular combinations of HEAD_SIZE, BLOCK_SIZE, NUM_THREADS) trigger it under specific compiler decisions. Worth enumerating.

- **H3 is speculative** but cheap to test.

- **H4 is the most practically important** because FP8 KV cache is the primary reason to use this kernel on Apple Silicon. If the fp8 path has drift beyond quantization noise, anyone using FP8 KV cache for Mac inference is getting degraded outputs silently.

- **H5 is a control.** Expected to pass.

## Reporting Format

For each confirmed hypothesis, produce a `bug_report_H<n>.md` following the structure of `metal_half_precision_bug.md`:

1. Summary
2. Reproduction (minimal reproducer)
3. Test results (table of expected vs buggy vs fixed)
4. Characteristics (when it fires, when it doesn't)
5. Root cause hypothesis
6. Workaround
7. Performance implications of the workaround

## Out-of-Scope

- Fixing the kernel. If bugs are confirmed, the fix is structurally obvious (add `__attribute__((always_inline))`, cast half parameters to float at call sites) but not the investigator's responsibility to ship.
- Filing issues upstream. That is a judgment call for after results are in. The current framing of the kernel package is misleading enough that a bug report would need to be carefully worded to avoid being dismissed as "this is mistral.rs's problem, not ours."
- Extending the reproduction to MXFP4 or other 4-bit weight schemes. That amplification is noted in the summary as a concern but is a separate investigation.

## Caveat on Publishing

Any public version of this investigation should omit or genericize the specific input constructions that most reliably trigger the bugs. The combination of "here is a Metal compiler miscompilation" and "here is a kernel shape that makes it fire hard" could be weaponized into a soft-break-your-GPU payload, which is not the goal. The minimal reproducer should demonstrate the issue without serving as a recipe for anything worse.
