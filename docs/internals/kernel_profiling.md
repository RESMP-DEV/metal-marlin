# Metal Kernel Profiling Guide

This document describes how to profile Metal compute kernels for occupancy, utilization, and performance bottleneck identification on Apple Silicon GPUs.

## 1. Profiling Tools

### 1.1 Programmatic Profiling (Python)

The `benchmarks/profile_kernel_metal.py` script provides programmatic profiling:

```bash
# Profile default kernel
uv run python benchmarks/profile_kernel_metal.py

# Profile specific kernel
uv run python benchmarks/profile_kernel_metal.py --kernel gemm_trellis_packed --bits 3

# Profile all GEMM kernels
uv run python benchmarks/profile_kernel_metal.py --all-kernels

# Export results to JSON
uv run python benchmarks/profile_kernel_metal.py --export-json results/profile.json

# Single configuration
uv run python benchmarks/profile_kernel_metal.py --single-config 128,4096,4096
```

### 1.2 Instruments.app (Apple Developer Tools)

For detailed hardware profiling, use Instruments:

1. **Metal System Trace**: GPU timeline, kernel dispatch analysis
   ```bash
   xcrun xctrace record --template 'Metal System Trace' --output trace.trace \
       --launch -- python your_script.py
   ```

2. **GPU Counters**: Hardware performance counters
   - Open Instruments.app
   - Choose 'GPU Counters' template
   - Select metrics: ALU Utilization, Memory Bandwidth, Occupancy
   - Profile your application

3. **Metal Debugger** (Xcode GPU Frame Capture):
   ```bash
   export METAL_DEVICE_WRAPPER_TYPE=1
   # Run app from Xcode with GPU Frame Capture enabled
   ```

4. **Metal Performance HUD** (real-time overlay):
   ```bash
   export MTL_HUD_ENABLED=1
   python your_script.py
   ```

## 2. Key Metrics

### 2.1 Occupancy

**Definition**: Ratio of active threads to maximum possible threads per compute unit.

```
Occupancy = (threads_per_threadgroup / max_threads_per_threadgroup) * 100
```

**Apple Silicon specifics**:
- Max threads per threadgroup: 1024 (typical)
- SIMD width: 32 threads (fixed on Apple Silicon)
- Max simdgroups per threadgroup: 32

**Optimization targets**:
- Target occupancy: >50% for latency hiding
- Trade-off: Higher occupancy reduces register/threadgroup memory per thread

### 2.2 Compute Utilization

**Definition**: Ratio of achieved TFLOPS to theoretical peak.

```
Compute Utilization = (achieved_tflops / peak_tflops) * 100
```

**Peak FP16 TFLOPS by chip** (approximate):

| Chip | FP16 TFLOPS | Memory BW (GB/s) |
|------|-------------|------------------|
| M1 | 5.5 | 68 |
| M1 Max | 17.8 | 400 |
| M2 Max | 27.2 | 400 |
| M3 Max | 28.0 | 400 |
| M4 Max | 33.6 | 546 |

### 2.3 Memory Bandwidth Utilization

**Definition**: Ratio of achieved memory bandwidth to theoretical peak.

```
Memory Utilization = (achieved_gb_s / peak_memory_bw) * 100
```

**Calculation for quantized GEMM**:
```
bytes_moved = bytes_A + bytes_B_packed + bytes_scales + bytes_C
            = M*K*2 + K*N*bits/8 + (K/group_size)*N*2 + M*N*2
```

### 2.4 Arithmetic Intensity

**Definition**: Compute-to-memory ratio (FLOP/byte).

```
Arithmetic Intensity = FLOPS / bytes_moved
                     = (2*M*N*K) / bytes_moved
```

The arithmetic intensity determines whether a kernel is memory-bound or compute-bound.

## 3. Roofline Model

The roofline model identifies performance bottlenecks:

```
Achievable TFLOPS = min(Peak_TFLOPS, AI * Memory_BW)
```

**Ridge point**: Where memory and compute roofs intersect.
```
Ridge Point = Peak_TFLOPS * 1000 / Memory_BW
```

| Chip | Ridge Point (FLOP/byte) |
|------|------------------------|
| M1 | ~80 |
| M1 Max | ~44 |
| M4 Max | ~61 |

**Interpretation**:
- AI < Ridge Point: **Memory bound** - optimize memory access patterns
- AI > Ridge Point: **Compute bound** - optimize ALU utilization
- AI ~ Ridge Point: **Balanced** - both matter equally

## 4. Bottleneck Identification

### 4.1 Memory Bound Kernels

**Symptoms**:
- High memory utilization (>70%)
- Low compute utilization (<30%)
- Arithmetic intensity below ridge point

**Common causes**:
1. Uncoalesced memory access
2. Cache thrashing
3. Insufficient prefetching
4. Memory-intensive operations (dequantization, attention)

**Optimizations**:
- Use double-buffering for latency hiding
- Ensure coalesced threadgroup memory access
- Increase tile size to improve data reuse
- Use simdgroup operations for register-level data sharing

### 4.2 Compute Bound Kernels

**Symptoms**:
- High compute utilization (>70%)
- Low memory utilization (<30%)
- Arithmetic intensity above ridge point

**Common causes**:
1. Insufficient parallelism
2. Register pressure limiting occupancy
3. Complex per-element operations

**Optimizations**:
- Increase threadgroup size (more simdgroups)
- Use simdgroup_matrix operations for dense GEMM
- Fuse operations to reduce kernel launch overhead
- Consider mixed-precision (FP16 compute, FP32 accumulation)

### 4.3 Low Occupancy

**Symptoms**:
- Occupancy below 50%
- Variable latency
- Underutilization of both compute and memory

**Common causes**:
1. Excessive threadgroup memory usage
2. High register pressure
3. Improper threadgroup sizing

**Optimizations**:
- Reduce per-thread register usage
- Split large threadgroup memory allocations
- Use simdgroup_barrier instead of threadgroup_barrier where possible
- Profile with Xcode GPU Frame Capture for register count

## 5. Metal-Specific Considerations

### 5.1 Threadgroup Memory

Apple Silicon threadgroup memory limits:
- Maximum: 32 KB per threadgroup
- Sweet spot: 8-16 KB for optimal occupancy

Current kernel budgets (gemm_trellis_packed):
```
A_tiles[2][64][32] * 2B     = 8,192 bytes
B_staging[4][8][8] * 2B     =   512 bytes
epilogue_staging[4][8][8]   =   512 bytes
────────────────────────────────────────────
Total                        ≈ 9,216 bytes (OK)
```

### 5.2 SIMD Group Operations

Apple Silicon supports simdgroup_matrix operations:
- Fragment size: 8x8 matrix elements
- Supported: simdgroup_load, simdgroup_store, simdgroup_multiply_accumulate
- Requirement: Metal 3.0 language version

Use simdgroup operations for:
- Dense matrix multiplication
- Reduction operations
- Register-level data sharing

### 5.3 Memory Access Patterns

Optimal access patterns:
- **Threadgroup**: Contiguous 4-byte aligned access (coalesced)
- **Device**: 16-byte aligned for full bandwidth
- **Constant**: Use for small, read-only data (cached aggressively)

### 5.4 Double Buffering

For memory-bound kernels, double buffering hides latency:

```metal
threadgroup half A_tiles[2][TILE_M][TILE_K];  // Double buffer

// Pipeline: load next tile while computing current
for (int k = 0; k < K; k += TILE_K) {
    // Async load tile[1-buf] while computing tile[buf]
    simdgroup_barrier(...);
    // Compute with tile[buf]
    simdgroup_multiply_accumulate(...);
    buf = 1 - buf;
}
```

## 6. Profiling Workflow

### Step 1: Baseline Profile
```bash
uv run python benchmarks/profile_kernel_metal.py --kernel marlin_gemm_fp4
```

### Step 2: Identify Bottleneck
Check the output:
- Bottleneck: MEMORY/COMPUTE/BALANCED
- Roofline Attainment: How close to theoretical maximum

### Step 3: Detailed Hardware Analysis
Use Instruments.app for:
- Per-cycle ALU utilization
- Cache hit rates
- Memory stall breakdown
- Shader invocation counts

### Step 4: Optimize and Re-profile
Based on bottleneck:
- Memory bound: Improve data reuse, prefetching
- Compute bound: Increase parallelism, use matrix ops
- Low occupancy: Reduce resource usage

### Step 5: Validate Across Configurations
```bash
uv run python benchmarks/profile_kernel_metal.py --all-kernels
```

## 7. Example Analysis

### 7.1 Decode Kernel (M=1)

```
Profile: gemm_trellis_packed_decode
M=1, N=4096, K=4096, 3-bit

Timing:           0.042 ms
Throughput:       1.63 TFLOPS, 285 GB/s
Compute Util:     4.9%
Memory Util:      52.2%
Bottleneck:       MEMORY bound
Roofline Attain:  71.3%
```

**Analysis**:
- Low compute utilization expected for M=1 (single token decode)
- Memory utilization indicates good bandwidth usage
- 71% roofline attainment leaves room for improvement
- Optimization: Wider N tiles, prefetching

### 7.2 Prefill Kernel (M=512)

```
Profile: gemm_trellis_packed
M=512, N=4096, K=4096, 3-bit

Timing:           0.85 ms
Throughput:       20.1 TFLOPS, 89 GB/s
Compute Util:     59.8%
Memory Util:      16.3%
Bottleneck:       COMPUTE bound
Roofline Attain:  65.2%
```

**Analysis**:
- Good compute utilization for batch processing
- Low memory utilization (compute-bound)
- Optimization: Increase occupancy with more simdgroups

## 8. Reference: Metal Pipeline State Properties

The profiler queries these pipeline state properties:

```python
pipeline.maxTotalThreadsPerThreadgroup()   # Max threads per TG
pipeline.threadExecutionWidth()            # SIMD width (32)
pipeline.staticThreadgroupMemoryLength()   # Compile-time TG memory
```

For device capabilities:
```python
device.name()                              # Device name string
device.supportsFamily_(MTLGPUFamilyApple9) # M3/M4 features
device.maxThreadsPerThreadgroup()          # Device-wide limit
device.recommendedMaxWorkingSetSize()      # Memory guidance
device.maxThreadgroupMemoryLength()        # TG memory limit
```

## 9. Further Reading

- [Apple Metal Best Practices Guide](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [GPU Frame Capture](https://developer.apple.com/documentation/xcode/capturing-gpu-command-data-programmatically)
- [Roofline Model](https://en.wikipedia.org/wiki/Roofline_model)
