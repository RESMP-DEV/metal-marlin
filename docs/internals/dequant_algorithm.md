# Marlin Bitwise FP4/INT4 to FP16 Dequantization

This document analyzes the bitwise dequantization algorithms from vLLM's Marlin
implementation (`csrc/quantization/marlin/dequant.h`), explains why they are
fundamentally faster than lookup-table approaches, and details how to port them
to Metal.

Reference: https://github.com/vllm-project/vllm/tree/main/csrc/quantization/marlin


## 1. FP4 (E2M1 / NVFP4) to FP16 Conversion

### 1.1 The FP4 E2M1 Format

FP4 E2M1 uses 4 bits per value: 1 sign, 2 exponent, 1 mantissa.

```
Bit layout:  [S][E1 E0][M0]
              3   2  1   0
```

The exponent bias is 1 (bias = 2^(2-1) - 1 = 1). This gives 16 representable
values:

| Binary | Sign | Exp | Mant | Value |
|--------|------|-----|------|-------|
| 0000   | +    | 0   | 0    | 0     |
| 0001   | +    | 0   | 1    | 0.5   |
| 0010   | +    | 1   | 0    | 1.0   |
| 0011   | +    | 1   | 1    | 1.5   |
| 0100   | +    | 2   | 0    | 2.0   |
| 0101   | +    | 2   | 1    | 3.0   |
| 0110   | +    | 3   | 0    | 4.0   |
| 0111   | +    | 3   | 1    | 6.0   |
| 1000   | -    | 0   | 0    | -0    |
| 1001   | -    | 0   | 1    | -0.5  |
| 1010   | -    | 1   | 0    | -1.0  |
| 1011   | -    | 1   | 1    | -1.5  |
| 1100   | -    | 2   | 0    | -2.0  |
| 1101   | -    | 2   | 1    | -3.0  |
| 1110   | -    | 3   | 0    | -4.0  |
| 1111   | -    | 3   | 1    | -6.0  |

Note: exp=0 with mantissa=1 is subnormal (0.5), computed as 0 + 1 * 2^(1-1) * 0.5 = 0.5.

### 1.2 FP16 Format Recap

FP16: 1 sign, 5 exponent, 10 mantissa. Exponent bias = 15.

```
Bit layout:  [S][E4 E3 E2 E1 E0][M9 M8 M7 M6 M5 M4 M3 M2 M1 M0]
             15  14 13 12 11 10   9  8  7  6  5  4  3  2  1  0
```

### 1.3 The Bitwise Conversion Algorithm

Marlin packs two FP4 values into specific bit positions within a 32-bit integer,
arranged so that they occupy the sign+exponent+mantissa fields of two packed
FP16 values (half2). The packing places each 4-bit FP4 value at bits [15:12]
of each 16-bit half within the 32-bit word.

The key insight: if you place the 4-bit FP4 `[S E1 E0 M0]` at bits [15:12]
of a 16-bit half word, you get:

```
FP16 bit positions:
  Bit 15 = S  (sign, already correct)
  Bit 14 = E1 (FP4 exponent high bit, in FP16 exponent field)
  Bit 13 = E0 (FP4 exponent low bit, in FP16 exponent field)
  Bit 12 = M0 (FP4 mantissa bit, in FP16 exponent field - WRONG position)
```

The FP4 exp+mantissa bits `[E1 E0 M0]` occupy bits [14:12]. For a valid FP16,
the exponent is at bits [14:10] and the mantissa at bits [9:0]. The algorithm
shifts the exp+mantissa bits right to properly separate them into FP16 fields.

#### Source code (from vLLM `dequant.h`):

```cuda
template <>
__device__ inline void dequant<half2, vllm::kFE2M1f.id(), true>(int q,
                                                                half2* frag_b) {
  constexpr int FP4_EXPONENT = 2, FP16_EXPONENT = 5;
  constexpr int RIGHT_SHIFT = FP16_EXPONENT - FP4_EXPONENT;  // = 3
  constexpr int MASK = 0x70007000;

  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 4;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

  frag_b[1] = *reinterpret_cast<const half2*>(&Out1);
  frag_b[0] = *reinterpret_cast<const half2*>(&Out2);
}
```

### 1.4 Mask Derivation

**Why 0x80008000 (sign mask)?**

The 32-bit int contains two 16-bit halves. Bit 15 of each half is the sign bit.
`0x80008000` extracts bit 15 from both halves simultaneously:

```
0x80008000 = 1000_0000_0000_0000  1000_0000_0000_0000
                                  ^                   ^
                            sign of half[1]     sign of half[0]
```

The FP4 sign bit is at position 3 within the nibble. When the nibble is placed
at bits [15:12], the sign bit lands at bit 15, which is exactly where FP16
expects it. No shifting needed for the sign.

**Why 0x70007000 (exp+mantissa mask)?**

`0x70007000` extracts bits [14:12] from each half:

```
0x70007000 = 0111_0000_0000_0000  0111_0000_0000_0000
              ^^^                  ^^^
         bits 14,13,12          bits 14,13,12
```

These are the 3 non-sign bits of the FP4 value: `[E1 E0 M0]`.

**Why RIGHT_SHIFT = 3?**

The FP4 exponent has 2 bits; FP16 exponent has 5 bits. The difference is 3.
After extracting `[E1 E0 M0]` at bits [14:12], shifting right by 3 moves them
to bits [11:9]:

```
Before shift: 0 E1 E0 M0 0 0 0 0 0 0 0 0 0 0 0 0
After >> 3:   0 0  0  0  E1 E0 M0 0 0 0 0 0 0 0 0
                          ^^ ^^  ^
                     bits: 11 10  9
```

This produces a 16-bit value with:
- Sign at bit 15 (from the OR with sign mask)
- Exponent bits E1,E0 at positions [11:10] of the FP16 exponent field [14:10]
- Mantissa bit M0 at position [9] of the FP16 mantissa field [9:0]

The resulting FP16 has exponent `000 E1 E0` and mantissa `M0 0000 00000`.
This is NOT yet the correct FP16 value because the exponent bias differs.

### 1.5 Exponent Bias Correction via Multiplication

The `skip_flop=true` version produces a raw bit pattern. The `skip_flop=false`
version applies bias correction:

```cuda
template <>
__device__ inline void dequant<half2, vllm::kFE2M1f.id(), false>(
    int q, half2* frag_b) {
  dequant<half2, vllm::kFE2M1f.id(), true>(q, frag_b);

  constexpr int FP4_EXPONENT = 2, FP16_EXPONENT = 5;
  constexpr int BIAS_OFFSET =
      (1 << (FP16_EXPONENT - 1)) - (1 << (FP4_EXPONENT - 1));
      // = (1 << 4) - (1 << 1) = 16 - 2 = 14
  const half2 bias_reg = __float2half2_rn(float(1 << BIAS_OFFSET));
      // = float(1 << 14) = 16384.0

  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
}
```

**Why multiply by 2^14?**

The raw bit-shifted result interprets the FP4 exponent (bias=1) directly in the
FP16 exponent field (bias=15). The bias mismatch is 15 - 1 = 14. To correct
this, we multiply by 2^14, which adds 14 to the effective exponent.

For example, FP4 value `0110` (binary) = +4.0:
- E1E0 = 11 (binary) = 3, actual exponent = 3 - 1 = 2
- M0 = 0, mantissa = 1.0
- Value = 1.0 * 2^2 = 4.0

After bit manipulation, raw FP16 has exponent field = `00011` = 3:
- FP16 interprets this as 2^(3-15) = 2^(-12), so raw value = 1.0 * 2^(-12)
- Multiply by 2^14: 2^(-12) * 2^14 = 2^2 = 4.0. Correct.

This works for normal values. For the subnormal case (exp=0, mantissa=1 giving
0.5), the bit shift produces FP16 exponent=0, mantissa MSB=1, which FP16 also
interprets as subnormal: 0 + 2^(1-15) * (1/2) * 2^14 = 2^(-15) * 0.5 * 16384
= 0.5. This is correct because FP4 and FP16 subnormal handling aligns after
the multiplication.

### 1.6 Processing Two Pairs from One int32

The input `q` contains 8 FP4 values (32 bits / 4 bits each). The packing puts
two FP4 values in the upper nibbles of each half-word (bits [15:12] of each
16-bit lane). The first extraction gets the first pair, then `q <<= 4` shifts
the next pair into position for the second extraction.

```
q (32 bits): [A3 A2 A1 A0 B3 B2 B1 B0 | ... | C3 C2 C1 C0 D3 D2 D1 D0]
              ^ first FP4 pair extracted         ^ ... processed after <<= 4
```

The reverse indexing (`frag_b[1]` before `frag_b[0]`) accounts for the specific
weight permutation used by Marlin's tensor-core-friendly memory layout.

### 1.7 Avoiding the LUT

The entire conversion is 3 ALU operations per pair of values:
1. `AND` to extract sign
2. `AND` + `SHIFT` to extract and reposition exp+mantissa
3. `OR` to combine sign with repositioned exp+mantissa

Plus one `MUL` for bias correction (fused with scale in practice).

No memory access is required. The 16-element lookup table is replaced by
pure register arithmetic.


## 2. INT4 (U4B8) to FP16 Conversion

### 2.1 The Magic Number Trick

Integer-to-float conversion on GPUs traditionally requires either an `int2float`
instruction (high latency on older architectures) or a lookup table. Marlin
instead exploits the structure of IEEE 754 to embed integer bits directly into
a floating-point number's mantissa, then subtract the bias.

### 2.2 Source Code

```cuda
template <>
__device__ inline void dequant<half2, vllm::kU4B8.id(), true>(int q,
                                                              half2* frag_b) {
  const int MASK = 0x000f000f;
  const int EX = 0x64006400;
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

  frag_b[0] = *reinterpret_cast<half2*>(&lo);
  frag_b[1] = *reinterpret_cast<half2*>(&hi);
}

template <>
__device__ inline void dequant<half2, vllm::kU4B8.id(), false>(int q,
                                                               half2* frag_b) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);

  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
}
```

### 2.3 The LOP3 Instruction

`lop3.b32` is a CUDA PTX instruction that computes a 3-input bitwise logical
operation in a single cycle. It takes three 32-bit inputs (a, b, c) and a
lookup table (LUT) that defines the output for each combination of input bits.

The LUT `(0xf0 & 0xcc) | 0xaa` computes to `0xea`:

```
Truth table derivation:
  a b c | 0xf0 | 0xcc | 0xf0&0xcc | 0xaa | result
  0 0 0 |  0   |  0   |     0     |  0   |   0
  0 0 1 |  0   |  0   |     0     |  1   |   1
  0 1 0 |  0   |  1   |     0     |  0   |   0
  0 1 1 |  0   |  1   |     0     |  1   |   1
  1 0 0 |  1   |  0   |     0     |  0   |   0
  1 0 1 |  1   |  0   |     0     |  1   |   1
  1 1 0 |  1   |  1   |     1     |  0   |   1
  1 1 1 |  1   |  1   |     1     |  1   |   1
                                           = 0b11101010 = 0xea
```

Reading the truth table: the output is 1 when `(a AND b) OR c`. So:

```
lop3<0xea>(a, b, c) = (a & b) | c
```

Applied to `lop3<0xea>(q, MASK, EX)`:

```
result = (q & MASK) | EX
       = (q & 0x000f000f) | 0x64006400
```

This extracts the low 4 bits of each 16-bit lane (`& 0x000f`) and ORs them
with the constant `0x6400`.

On Metal, this decomposes to two instructions: `(q & MASK) | EX`. The lop3
instruction saves one cycle on CUDA by fusing both operations.

### 2.4 Why 0x6400 (Bias = 1024.0)?

`0x6400` is the FP16 representation of 1024.0:

```
0x6400 = 0 11001 0000000000
         S  exp    mantissa

exponent = 0b11001 = 25
value = 2^(25 - 15) * 1.0 = 2^10 = 1024.0
```

When we OR a 4-bit integer N (0-15) into the mantissa of 0x6400:

```
0x6400 | N = 0 11001 00000N3N2N1N0

mantissa = N / 1024 (since mantissa is 10 bits)
value = 2^10 * (1 + N/1024) = 1024 + N
```

So `(q & 0x000f) | 0x6400` produces an FP16 whose value is `1024 + N` where N
is the 4-bit integer value. To recover N, subtract 1024.

This is the "magic number" trick: by placing integer bits directly into a
floating-point mantissa with a known exponent, we convert integer to float
using only bitwise operations and one subtraction.

### 2.5 U4B8: The Biased Format

The "B8" in U4B8 means the 4-bit values are stored with a bias of 8. An
unsigned 4-bit value N (0-15) represents the signed integer N-8 (range -8 to
+7). The subtraction constant 0x6408 encodes 1024 + 8 = 1032 in FP16:

```
0x6408 = 0 11001 0000001000
value = 2^10 * (1 + 8/1024) = 1024 + 8 = 1032
```

Subtracting 1032 from (1024 + N) gives N - 8, producing the signed result.

### 2.6 High Nibble Processing

For the high nibble, Marlin uses `HI = 0x00f000f0` to extract bits [7:4].
After the LOP3, the result is `(q & 0x00f000f0) | 0x64006400`. The 4-bit
value is now at mantissa bits [7:4] instead of [3:0], which means the embedded
value is 16x larger: `1024 + 16*N`.

To correct this, Marlin uses FMA:
- `MUL = 0x2c00` = FP16 for 1/16 = 0.0625
- `ADD = 0xd480` = FP16 for -(1024*0.0625 + 8) = -(64 + 8) = -72

The FMA computes: `hi_fp16 * (1/16) + (-72)`
= `(1024 + 16N) / 16 - 72`
= `64 + N - 72`
= `N - 8`

This gives the same signed result as the low-nibble path.

### 2.7 The Complete U4 (Non-Biased) Variant

For `kU4` (unsigned without B8 bias), the subtraction constant differs:

```cuda
const int SUB = 0x64006400;  // subtract 1024 (not 1032)
const int MUL = 0x2c002c00;  // 1/16
const int ADD = 0xd400d400;  // -(64) = -64
```

This produces unsigned values 0-15 directly.


## 3. Comparison with llama.cpp's Lookup Table Approach

### 3.1 llama.cpp MXFP4 Dequantization

```metal
constexpr constant static float kvalues_mxfp4_f[16] = {
    0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
    -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

void dequantize_mxfp4(device const block_mxfp4* xb, short il,
                       thread type4x4& reg) {
    device const uint8_t* q2 = (device const uint8_t*)xb->qs;
    const float d = e8m0_to_fp32(xb->e);
    const uint8_t shr = il >= 1 ? 4 : 0;

    for (int i = 0; i < 4; ++i) {
        reg[i][0] = d * kvalues_mxfp4_f[(q2[4*i + 0] >> shr) & 0x0F];
        reg[i][1] = d * kvalues_mxfp4_f[(q2[4*i + 1] >> shr) & 0x0F];
        reg[i][2] = d * kvalues_mxfp4_f[(q2[4*i + 2] >> shr) & 0x0F];
        reg[i][3] = d * kvalues_mxfp4_f[(q2[4*i + 3] >> shr) & 0x0F];
    }
}
```

The optimized mat-vec kernel preloads the LUT into threadgroup memory:

```metal
shmem_f32[tiisg] = kvalues_mxfp4_f[tiisg % 16];
threadgroup_barrier(mem_flags::mem_threadgroup);
// ... then indexes: shmem_f32[q2[i] & 0x0F]
```

### 3.2 Operation Count Comparison

**llama.cpp (per value):**

| Operation | Type | Count |
|-----------|------|-------|
| Shift (`>> shr`) | ALU | 1 |
| Mask (`& 0x0F`) | ALU | 1 |
| LUT load (`shmem[idx]`) | Memory (shared/threadgroup) | 1 |
| Scale multiply (`d *`) | FP MUL | 1 |
| **Total** | | **2 ALU + 1 MEM + 1 FP** |

**Marlin FP4 (per 2 values):**

| Operation | Type | Count |
|-----------|------|-------|
| AND sign (`& 0x80008000`) | ALU | 1 |
| AND exp+mant (`& 0x70007000`) | ALU | 1 |
| Shift right (`>> 3`) | ALU | 1 |
| OR combine | ALU | 1 |
| Bias multiply (fused with scale) | FP MUL | 1 |
| **Total** | | **4 ALU + 1 FP (for 2 values)** |

Per value: **2 ALU + 0.5 FP** vs **2 ALU + 1 MEM + 1 FP**.

**Marlin INT4 (per 2 values, `skip_flop=true`):**

| Operation | Type | Count |
|-----------|------|-------|
| LOP3 / AND+OR | ALU | 1 (CUDA) or 2 (Metal) |
| **Total** | | **1-2 ALU (for 2 values)** |

For `skip_flop=false` (with zero-point correction), add 1 SUB and 1 FMA.


## 4. Latency Analysis: Why Bitwise is Faster on GPU

### 4.1 Memory vs. ALU on Modern GPUs

On both NVIDIA and Apple Silicon GPUs, the fundamental performance characteristic
is the same: ALU operations are cheap compared to memory accesses, even for
threadgroup/shared memory. Exact cycle counts vary by GPU generation, clocks,
and compiler scheduling, so treat the ordering below as relative:

| Operation | Relative cost (order-of-magnitude) |
|-----------|------------------------------------|
| Integer ALU (AND, OR, SHIFT) | 1x |
| FP16 MUL/FMA | ~1-2x |
| Shared/TG memory load | ~5-10x |
| L1 cache hit | ~8-15x |
| Global memory (uncached) | ~50-200x |

### 4.2 Throughput Bottleneck Analysis

For a quantized GEMM, the dequantization throughput must match the MMA
(matrix multiply-accumulate) throughput to avoid being the bottleneck.

**Tensor cores** consume packed `FragB` values at extremely high throughput.
If each value requires a shared-memory lookup, that lookup can become the
bottleneck before the MMA pipeline saturates. The exact break-even point
depends on GPU generation and clock, but the qualitative outcome is stable:
shared-memory LUT loads compete with MMA demand.

**Bitwise approach:**
All operations are register ALU. Integer ALU throughput is typically orders
of magnitude higher than the dequantization demand, so the bitwise path
rarely becomes the bottleneck.

### 4.3 Bank Conflicts

Shared memory LUT access patterns suffer from bank conflicts when multiple
threads in a warp access the same bank. The 16-entry FP4 LUT is small, but
many threads can still contend for the same bank or cache line when they
decode the same 4-bit value, serializing access.

The bitwise approach has zero bank conflicts because it uses no shared memory.

### 4.4 Register Pressure

The LUT approach requires either:
- Threadgroup memory allocation (barrier overhead, occupancy impact), or
- Constant memory (cache pollution, limited bandwidth), or
- Register-resident LUT (16 * 32-bit = 16 registers per thread, non-trivial)

The bitwise approach uses only immediate constants embedded in the instruction
stream and 2-4 temporary registers.

### 4.5 Pipeline Considerations for Fused Dequant-GEMM

The critical advantage of bitwise dequant becomes apparent in Marlin's fused
kernel architecture:

```
Traditional (llama.cpp):
  Global → Shared (async copy, wait)
  Shared → Registers (LUT indexed load)
  Registers → Tensor Core

Marlin (fused):
  Global → Registers (packed INT4/FP4 load)
  Registers → Registers (bitwise dequant, ~1-2 cycles)
  Registers → Tensor Core
```

The shared memory stage is eliminated entirely. This saves:
1. Shared memory allocation (more warps can be resident)
2. Barrier synchronization (threadgroup_barrier latency)
3. One memory-to-register transfer

On Apple Silicon, threadgroup memory allocation can still reduce occupancy and
adds synchronization overhead, so eliminating the shared-memory stage remains
advantageous even if the underlying physical implementation differs from
NVIDIA's shared-memory model.

### 4.6 Instruction-Level Parallelism

The bitwise operations have no data dependencies between the two half-words
being processed. The GPU's instruction scheduler can overlap the sign extraction,
exp+mantissa extraction, and shift operations across multiple values. In
contrast, LUT loads create a data-dependent chain:

```
shift → mask → load (stall) → multiply
```

The load cannot begin until the index is computed, and the multiply cannot
begin until the load completes. This is a 3-stage dependent chain with a
memory access in the middle.


## 5. Metal Translation

### 5.1 FP4 E2M1 to FP16

Direct translation; no special primitives needed:

```metal
inline half2 dequant_fp4_pair(uint q_pair) {
    // q_pair has two FP4 values at bits [15:12] of each 16-bit lane
    constexpr uint SIGN_MASK = 0x80008000;
    constexpr uint EXP_MANT_MASK = 0x70007000;
    constexpr int RIGHT_SHIFT = 3;  // FP16_EXP(5) - FP4_EXP(2)

    uint out = (q_pair & SIGN_MASK) | ((q_pair & EXP_MANT_MASK) >> RIGHT_SHIFT);
    return as_type<half2>(out);
}

// With bias correction (complete dequant):
inline half2 dequant_fp4_pair_biased(uint q_pair) {
    half2 raw = dequant_fp4_pair(q_pair);
    constexpr half2 BIAS = half2(16384.0h);  // 2^14
    return raw * BIAS;
}
```

### 5.2 INT4 U4B8 to FP16

The lop3 decomposes into AND+OR:

```metal
inline half2 dequant_int4_lo(uint q) {
    constexpr uint LO_MASK = 0x000f000f;
    constexpr uint EX = 0x64006400;  // FP16 1024.0 in both lanes

    uint lo = (q & LO_MASK) | EX;
    return as_type<half2>(lo);
}

inline half2 dequant_int4_hi(uint q) {
    constexpr uint HI_MASK = 0x00f000f0;
    constexpr uint EX = 0x64006400;

    uint hi = (q & HI_MASK) | EX;
    return as_type<half2>(hi);
}

// With zero-point correction for U4B8:
inline void dequant_u4b8(uint q, thread half2& out_lo, thread half2& out_hi) {
    constexpr half2 SUB_LO = as_type<half2>(uint(0x64086408));  // 1032.0
    constexpr half2 MUL_HI = as_type<half2>(uint(0x2c002c00));  // 1/16
    constexpr half2 ADD_HI = as_type<half2>(uint(0xd480d480));  // -72.0

    out_lo = dequant_int4_lo(q) - SUB_LO;
    out_hi = fma(dequant_int4_hi(q), MUL_HI, ADD_HI);
}
```

### 5.3 Performance on Metal

On Metal:
- The lop3 decomposition costs one extra integer op (AND+OR vs single lop3).
- The overall pipeline benefit is preserved because the shared memory stage is
  still eliminated.
- simdgroup_multiply_accumulate can consume dequantized values directly from
  registers.

The real-world speedup depends on whether the kernel is compute-bound or
memory-bound. For batch size 1 (often memory-bound), the dequant approach
matters less. As batch size grows and compute dominates, removing shared
memory traffic and barriers more directly improves throughput. Measure with
the in-repo profiling tools for device-specific results.
