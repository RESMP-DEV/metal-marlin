# Metal Compiler Error Guide

Common Metal shader compilation errors and solutions for Metal-Marlin.

## Thread Execution Width Errors

### Error: "Thread execution width must be a multiple of SIMD group size"

**Cause:** Threadgroup size not aligned with GPU SIMD width (32 on Apple Silicon).

**Solution:**
```cpp
// Bad
[[kernel]] void compute(uint tid [[thread_position_in_grid]]) {
    // threadgroup size: 31
}

// Good
[[kernel]] void compute(uint tid [[thread_position_in_grid]]) {
    // threadgroup size: 32, 64, 128, 256
}
```

Set `threadsPerThreadgroup` to multiples of 32.

## Memory Access Errors

### Error: "Address space mismatch"

**Cause:** Incorrect pointer address space qualifier.

**Solution:**
```cpp
// Bad
void helper(float* data) { }

// Good - specify address space
void helper(device float* data) { }
void helper(threadgroup float* data) { }
void helper(constant float* data) { }
```

### Error: "Threadgroup memory size exceeds limit"

**Cause:** Threadgroup memory allocation &gt; 32 KB (Apple Silicon limit).

**Solution:**
```cpp
// Bad - 64 KB
threadgroup float cache[16384];

// Good - 8 KB
threadgroup float cache[2048];

// Or reduce precision
threadgroup half cache[4096]; // 8 KB
```

## Type Errors

### Error: "Cannot convert between vector types"

**Cause:** Incompatible vector operations.

**Solution:**
```cpp
// Bad
float4 a = float4(1.0);
half4 b = a; // implicit conversion not allowed

// Good
float4 a = float4(1.0);
half4 b = half4(a);
```

### Error: "Atomic operations require device or threadgroup memory"

**Cause:** Using atomics on wrong memory type.

**Solution:**
```cpp
// Bad
void compute(constant atomic_int* counter) { }

// Good
void compute(device atomic_int* counter) { }
void compute(threadgroup atomic_int* counter) { }
```

## Kernel Signature Errors

### Error: "Buffer index X exceeds maximum"

**Cause:** Buffer binding index &gt; 30 (Metal limit: 0-30 for buffers).

**Solution:**
```cpp
// Bad
[[kernel]] void compute(
    device float* buf [[buffer(31)]]
) { }

// Good
[[kernel]] void compute(
    device float* buf [[buffer(0)]]
) { }
```

Reuse buffer indices or pack data into fewer buffers.

### Error: "Kernel must have at least one threadgroup dimension"

**Cause:** Invalid dispatch dimensions.

**Solution:**
```cpp
// Python side
# Bad
encoder.dispatchThreads((0, 1, 1), threadsPerThreadgroup=(32, 1, 1))

# Good
encoder.dispatchThreads((32, 1, 1), threadsPerThreadgroup=(32, 1, 1))
```

## Synchronization Errors

### Error: "Threadgroup barrier in non-uniform control flow"

**Cause:** Barrier inside conditional that not all threads execute.

**Solution:**
```cpp
// Bad
if (tid &lt; 128) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Good
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid &lt; 128) {
    // work
}
```

All threads in threadgroup must execute barrier.

## Resource Limit Errors

### Error: "Function exceeds register limit"

**Cause:** Too many registers used, causing spills.

**Solution:**
- Reduce local variables
- Use smaller data types (half instead of float)
- Split kernel into multiple passes
- Reduce loop unrolling

### Error: "Texture binding limit exceeded"

**Cause:** Too many texture arguments (&gt;128 on M1/M2, &gt;256 on M3+).

**Solution:**
- Combine textures into arrays
- Use buffer-backed textures
- Split kernel into multiple dispatches

## Compilation Flags

### Enable optimizations
```bash
# For release builds
-ffast-math -O3

# Preserve debug info
-g -O0
```

### Check for warnings
```bash
# Treat warnings as errors during development
-Werror

# Enable all warnings
-Wall -Wextra
```

## Debug Tips

### 1. Check intermediate representation
```bash
xcrun metal -c kernel.metal -o kernel.air
xcrun metal-ar -t kernel.air
```

### 2. Validate threadgroup size
```python
assert threads_per_group[0] % 32 == 0
assert threads_per_group[0] * threads_per_group[1] * threads_per_group[2] &lt;= 1024
```

### 3. Profile memory usage
```python
# After kernel compilation
print(f"Threadgroup memory: {function.threadgroupMemoryLength} bytes")
print(f"Max threads: {function.maxTotalThreadsPerThreadgroup}")
```

## Common Patterns

### Safe buffer indexing
```cpp
uint idx = gid.x;
if (idx &lt; buffer_size) {
    output[idx] = input[idx];
}
```

### Threadgroup reduction
```cpp
threadgroup float shared[256];
uint tid = tpg.x; // thread position in threadgroup
uint gid = grid.x; // global ID

shared[tid] = input[gid];
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint s = 128; s &gt; 0; s &gt;&gt;= 1) {
    if (tid &lt; s) {
        shared[tid] += shared[tid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid == 0) {
    output[grid.x / 256] = shared[0];
}
```

## References

- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Best Practices Guide](https://developer.apple.com/documentation/metal/best_practices)
- [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
