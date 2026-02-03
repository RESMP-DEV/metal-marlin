# Performance Analysis

## Kernel Dispatch Latency

### Precompiled Metallib (Recommended)

```
Library load (once): ~10-50 ms
Kernel lookup: ~0.01 ms
Total first kernel: ~10-50 ms
Subsequent kernels: ~0.01 ms
```

### JIT Compilation (Fallback)

```
Shader compile (per file): ~50-100 ms
Library creation: ~1-5 ms
Kernel lookup: ~0.1 ms
Total first kernel: ~50-100 ms
Subsequent kernels (same shader): ~0.1 ms
New shader: ~50-100 ms again
```

## Impact on Inference

For a model with 100 unique kernels:

| Method | Cold Start | Warm Throughput |
|--------|------------|-----------------|
| Metallib | ~50ms | ~0.01ms/dispatch |
| JIT | ~5-10 seconds | ~0.1ms/dispatch |

## Memory Usage

- Metallib file: ~5-20 MB
- Loaded in memory: ~10-50 MB
- JIT compilation cache: ~50-100 MB

## Recommendations

1. Always prebuild metallib for production
2. Use JIT only for development/debugging new kernels
3. Monitor with `enable_timing()` to catch regressions
