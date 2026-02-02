# MPS Async Dispatch Investigation

## Summary

This document describes the investigation into async kernel dispatch patterns for
pipelining MoE layer execution on Metal Performance Shaders (MPS).

**Result: Implementation NOT recommended.** The measured improvement (7.7%) is below
the 15% threshold required to justify the added complexity.

## Key Findings

### 1. MPS Async Execution Model

Metal completion handlers are **supported** and functional:
- Handler callback latency: 1.447 ms
- Commit overhead: 0.072 ms

This enables async notification when kernels complete. However, PyObjC's
MTLSharedEvent binding is not available, limiting cross-queue synchronization options.

### 2. PyTorch MPS Async Patterns

Batch synchronization vs per-op synchronization:
- Sync after each op: 291.47 ms
- Batch sync (submit all, sync once): 270.64 ms
- **Speedup: 1.08x (7.7% improvement)**

The MPS backend already provides implicit command buffer queuing. Explicit async
dispatch provides only marginal improvement because:
1. MPS batches operations internally
2. `torch.mps.synchronize()` forces a full pipeline flush
3. Most time is spent in actual GPU execution, not dispatch overhead

### 3. MoE Layer Overlap Potential

Measured timings (batch=128, hidden=4096, intermediate=14336, top_k=4):
- Router time: 2.82 ms (+/- 9.37)
- Single expert GEMM: 9.45 ms (+/- 3.96)
- Theoretical overlap: 2.82 ms
- **Overlap potential: 6.9%**

The overlap potential is limited because:
1. Router computation is already fast (2.82 ms)
2. Expert GEMMs dominate execution time
3. The fused MoE kernel already handles all experts in a single dispatch

### 4. Multi-Encoder Command Buffers

Using multiple compute encoders in a single command buffer shows near-linear scaling:
- 2 kernels: 1.11 ms total
- 4 kernels: 2.33 ms total
- 8 kernels: 4.41 ms total

This indicates Metal is already executing kernels efficiently with minimal overhead.

### 5. Async Dispatch Patterns

Testing multiple command buffers in flight:
- 2 in flight: 1.246 ms/kernel
- 4 in flight: 1.304 ms/kernel
- 8 in flight: 1.254 ms/kernel

Baseline sync dispatch: 2.401 ms/kernel

The per-kernel time improves with multiple in-flight buffers (~48% reduction from
sync baseline), but this improvement is already captured by the current architecture
that uses PyTorch's implicit async execution.

## Why Async Pipelining Doesn't Help

The current Metal Marlin implementation already benefits from MPS async execution:

1. **Implicit Batching**: PyTorch MPS automatically batches operations
2. **Fused Kernels**: The `moe_trellis_swiglu` kernel fuses all expert execution
3. **No Sequential Bottleneck**: The per-expert loop was eliminated by the fused kernel

The remaining synchronization points are:
- `dispatch_kernel(..., wait=True)` - Required for correctness (output must be ready)
- `torch.mps.synchronize()` - Used sparingly for timing/debugging

Removing these would require fundamental architectural changes that risk correctness.

## Recommendations

### Do NOT Implement Async Pipelining

The 7.7% measured improvement does not justify:
- Increased code complexity
- Risk of subtle synchronization bugs
- Memory overhead from double buffering
- Harder debugging and profiling

### Alternative Optimizations

Instead of async pipelining, consider these higher-impact optimizations:

1. **Larger Batch Sizes**: Better GPU utilization scales with batch size
2. **Kernel Fusion**: Continue fusing operations (already done with SwiGLU)
3. **Weight Prefetching**: Already implemented with `CachedWeightBuffers`
4. **Quantization Improvements**: 2-bit or 1.5-bit for memory-bound layers

## Memory Overhead Analysis

Double buffering for pipelining would require:
- Per-layer intermediate buffer: batch_size * intermediate_dim * 2 bytes (FP16)
- With batch=128, intermediate=14336: ~3.5 MB per buffer
- 2 buffers for double buffering: ~7 MB total

This is negligible compared to model weights (~GB scale), but the complexity
overhead is not justified by the minimal performance gain.

## Conclusion

**Do not proceed with async pipelining implementation.**

The MPS execution model already provides sufficient async behavior through implicit
command buffer queuing. The measured 7.7% improvement from explicit async patterns
is below the 15% threshold and does not justify the added complexity.

The fused MoE kernel (`moe_trellis_swiglu`) already eliminates the sequential
expert iteration bottleneck that would have been the primary target for pipelining.

## Test Methodology

Tests run using `contrib/metal_marlin/scripts/prototype_async_moe.py`:

1. **Sync Dispatch**: Single command buffer with `waitUntilCompleted()`
2. **Async Dispatch**: Multiple command buffers committed before waiting
3. **Completion Handlers**: Using `addCompletedHandler_` for async notification
4. **Multi-Encoder**: Multiple compute encoders in single command buffer
5. **PyTorch MPS**: Comparing `torch.mps.synchronize()` patterns
6. **MoE Overlap**: Measuring router vs expert execution time

Hardware: Apple Silicon with MPS support (results may vary by chip generation).
