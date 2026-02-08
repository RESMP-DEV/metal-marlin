# Dequantization Algorithms (Overview)

This document is a high-level overview. For exact bitwise masks, packing rules,
and bias correction used in the kernels, see [dequant_algorithm.md](../internals/dequant_algorithm.md).

## FP4 E2M1 Dequantization

### Scalar reference implementation

```metal
half dequant_fp4_scalar(uchar fp4_code, half scale) {
    // Extract fields
    bool sign = (fp4_code >> 3) & 1;
    uchar exp = (fp4_code >> 1) & 0x3;
    uchar mant = fp4_code & 1;

    // Compute value
    half val;
    if (exp == 0) {
        val = half(mant) * 0.5h;  // Subnormal: 0.0 or 0.5
    } else {
        // Normal: 2^(exp-1) * (1 + mant/2)
        val = half(1 << (exp - 1)) * (1.0h + half(mant) * 0.5h);
    }

    // Apply sign
    if (sign) {
        val = -val;
    }

    return val * scale;
}
```

### Packed bitwise approach (kernel path)

In the kernel path, FP4 values are packed into a `uint32_t` (8 nibbles per
word). The conversion is done by placing each nibble into bits [15:12] of each
16-bit lane, then:

1. Mask the sign bits.
2. Mask and shift the exponent+mantissa bits into FP16 field positions.
3. Multiply by `2^14` to correct the bias mismatch (FP4 bias 1 vs FP16 bias 15).
4. Multiply by the per-group scale (often fused with step 3).

See [`dequant_algorithm.md`](../internals/dequant_algorithm.md) for the exact masks and code.

## INT4 Magic-Bias Dequantization

### Magic-bias trick

The "magic bias" approach embeds a small integer into the FP16 mantissa by
OR-ing with a constant that has a known exponent:

```
0x6400 = 0b0_11001_0000000000  // FP16 value 1024.0
```

If `N` is a 4-bit integer (0-15), then:

```
as_fp16(0x6400 | N) = 1024.0 + N
```

Subtracting 1024.0 recovers `N` without an integer-to-float conversion. For
U4B8, subtract 1032.0 instead (1024 + bias 8) to yield `N - 8`.

### Vectorized INT4 dequant (pseudocode)

```metal
half8 dequant_u4x8(uint packed, half scale, half zero) {
    // Unpack into 8 x uint16 with magic bias
    ushort8 unpacked;
    for (int i = 0; i < 8; ++i) {
        ushort val = (packed >> (i * 4)) & 0xF;
        unpacked[i] = val | 0x6400;  // Add magic bias
    }

    // Interpret as FP16 and subtract bias
    half8 result = as_type<half8>(unpacked) - half8(1024.0h);

    // Apply scale and zero
    return (result - zero) * scale;
}
```

## Performance Notes (Qualitative)

- Scalar FP4 is branchy and slow; use only for validation.
- Bitwise FP4 avoids LUT loads and integer-to-float conversions.
- Magic-bias INT4 is the fastest path for 4-bit integer dequant.
- Actual throughput depends on memory traffic and kernel fusion.

## See Also

- [dequant_algorithm.md](../internals/dequant_algorithm.md) for the exact bitwise algorithms and constants.
