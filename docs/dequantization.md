# Dequantization Algorithms

## FP4 E2M1 Dequantization

### Scalar Implementation

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
    if (sign) val = -val;

    return val * scale;
}
```

### Optimized Vectorized Implementation

The scalar loop is slow. Instead, we use bitwise operations to dequant
8 values in parallel:

```metal
// Process 8 FP4 values packed in one uint32
half8 dequant_fp4_x8(uint packed, half scale) {
    half8 result;

    // Extract sign bits
    uint signs = (packed >> 3) & 0x11111111u;  // Every 4th bit starting at bit 3

    // Extract exponent (2 bits each)
    uint exps = (packed >> 1) & 0x33333333u;

    // Extract mantissa (1 bit each)
    uint mants = packed & 0x11111111u;

    // Vectorized decode using lookup or bit manipulation
    // ...

    return result * scale;
}
```

### CUDA lop3.b32 Approach (Reference)

CUDA Marlin uses `lop3.b32` (3-input LUT) for efficient bit manipulation:

```cuda
// Extract low nibbles (values 0,2,4,6) and high nibbles (1,3,5,7)
uint32_t lo = lop3<0xea>(packed, 0x0f0f0f0f, 0x00ff00ff);  // Complex LUT
uint32_t hi = lop3<0xea>(packed >> 4, 0x0f0f0f0f, 0x00ff00ff);
```

Metal doesn't have lop3, so we compose from AND/OR/XOR.

## INT4 Magic-Bias Dequantization

### The Magic-Bias Trick

Instead of:
```
fp_value = (int4_value - zero_point) * scale
```

We use FP16 bit manipulation:
```
// FP16 representation: s | eeeee | mmmmmmmmmm
// Adding 1024.0 (0x6400) to a small integer creates:
// 0 | 10100 | int_value << 6

// This trick extracts 4-bit int into FP16 mantissa position
half val = as_half((uint16_t)(int4_value | 0x6400)) - 1024.0h;
```

This avoids integer-to-float conversion instructions.

### Vectorized INT4 Dequant

```metal
half8 dequant_u4x8(uint packed, half scale, half zero) {
    // Unpack into 8 x uint16 with magic bias
    ushort8 unpacked;
    for (int i = 0; i < 8; ++i) {
        ushort val = (packed >> (i * 4)) & 0xF;
        unpacked[i] = val | 0x6400;  // Add magic bias
    }

    // Interpret as FP16 and subtract bias
    half8 result = as_half8(unpacked) - half8(1024.0h);

    // Apply scale and zero
    return (result - zero) * scale;
}
```

## Performance Comparison

| Method | Cycles/element | Notes |
|--------|----------------|-------|
| Scalar FP4 | ~8 | Branch-heavy |
| Vectorized FP4 | ~2 | Bitwise parallel |
| Scalar INT4 | ~4 | int->float conversion |
| Magic-bias INT4 | ~1.5 | No conversion |
