# CUDA to Metal Porting Guide

This guide documents the systematic approach used to port Marlin from CUDA to Metal.

## Execution Model Mapping

| CUDA | Metal | Notes |
|------|-------|-------|
| Thread | Thread | Smallest execution unit |
| Warp (32) | Simdgroup (32) | SIMT execution group |
| Block | Threadgroup | Shared memory scope |
| Grid | Grid | Dispatch dimensions |

## Memory Hierarchy

| CUDA | Metal | Size (M4 Max) |
|------|-------|---------------|
| Global memory | Device memory | 128 GB |
| Shared memory | Threadgroup memory | 32 KB/TG |
| Registers | Thread registers | ~128/thread |
| L2 cache | Device cache | 48 MB SLC |

## Intrinsic Mapping

### Tensor Core → Simdgroup Matrix

CUDA:
```cuda
wmma::mma_sync(C, A, B, C);  // 16x8x16
```

Metal:
```metal
simdgroup_matrix<half, 8, 8> A_mat, B_mat, C_mat;
simdgroup_multiply_accumulate(C_mat, A_mat, B_mat, C_mat);  // 8x8
```

**Key difference:** CUDA mma.sync is 16x8x16, Metal is 8x8. Need 4x more ops.

### Warp Shuffle → Simd Shuffle

CUDA:
```cuda
float val = __shfl_xor_sync(0xffffffff, my_val, 16);
```

Metal:
```metal
float val = simd_shuffle_xor(my_val, 16);
```

### Async Copy → Manual Pipeline

CUDA:
```cuda
__pipeline_memcpy_async(dst, src, 16);
__pipeline_commit();
__pipeline_wait_prior(0);
```

Metal (no direct equivalent):
```metal
// Use double-buffering with explicit loads
threadgroup half buf[2][TILE][TILE];
// ... manual ping-pong ...
```

### LOP3.b32 → Composed Operations

CUDA:
```cuda
uint32_t result = lop3<0xea>(a, b, c);  // 3-input LUT
```

Metal:
```metal
// Decompose LUT 0xea = (a & b) | (a & c) | (b & c)
uint result = (a & b) | (a & c) | (b & c);
```

### PRMT.b32 → Extract + Shift

CUDA:
```cuda
uint32_t result = __prmt(a, b, selector);  // Byte permute
```

Metal:
```metal
// Manual byte extraction
uchar4 bytes_a = as_type<uchar4>(a);
uchar4 bytes_b = as_type<uchar4>(b);
uchar4 result = select_bytes(bytes_a, bytes_b, selector);
```

## Common Pitfalls

1. **Barrier semantics**: CUDA `__syncthreads()` = Metal `threadgroup_barrier(mem_flags::mem_threadgroup)`

2. **Memory coherence**: Metal requires explicit barriers between TG memory writes and reads

3. **Half precision**: Metal `half` has different rounding than CUDA `__half`

4. **Atomics**: Metal atomics are typed; CUDA allows more flexibility

## Performance Tuning

1. **Occupancy**: Metal GPU cores may have different limits than CUDA SMs

2. **Memory coalescing**: Both benefit from aligned, sequential access

3. **Register pressure**: Metal compiler is aggressive; may need manual register limiting
