# simdgroup_async_copy (latency hiding)

## Summary

`simdgroup_async_copy` is an undocumented Metal instruction that lets a SIMD-group schedule a device<->threadgroup copy that overlaps with compute. It is central to Apple GPU GEMM kernels that aim to hide device memory latency and keep the SIMD-group matrix hardware fed. The instruction is not described in the Metal Shading Language (MSL) spec, but it appears in leaked headers and in working GEMM examples.

## Signature and low-level form

The leaked header signatures (Xcode 14.2-14.3 era) expose templated overloads for 1D and 2D copies and a `simdgroup_future<void>` synchronization primitive: 
- `simdgroup_future<void> simdgroup_async_copy(threadgroup T *dst, const device T *src, ulong n_elements);`
- `simdgroup_future<void> simdgroup_async_copy(device T *dst, const threadgroup T *src, ulong n_elements);`
- 2D tiled variants with element/stride/tile parameters and clamp modes.

Source: `philipturner/metal-benchmarks` header reconstruction. 
https://github.com/philipturner/metal-benchmarks (see README header excerpt)

Percisely’s GEMM writeup shows the private assembly entrypoints used by the compiler and a minimal helper:

```text
thread _simdgroup_event_t* __metal_simdgroup_async_copy_2d(...)
  __asm("air.simdgroup_async_copy_2d.p3i8.p1i8");

void __metal_wait_simdgroup_events(...)
  __asm("air.wait_simdgroup_events");
```

Source: https://percisely.xyz/gemm

These two functions are enough to use async copy in custom kernels when the public overloads are unavailable.

## Usage pattern (GEMM mainloop)

A typical GEMM mainloop stages tiles of A and B from device to threadgroup memory using async copies, waits on the events, then performs compute on the threadgroup tiles. The minimal pattern is:

1. Issue `simdgroup_async_copy` for A and B tiles into threadgroup memory.
2. Wait on the returned events (or future) before using the data.
3. Compute with the staged tiles.
4. Repeat for the next K tile (often in a pipeline with double buffering).

Percisely’s example uses two async copies per loop iteration, followed by `__metal_wait_simdgroup_events(...)` before the compute phase. 
Source: https://percisely.xyz/gemm

## Availability in current Xcode

- The instruction is **not** in the public MSL spec. 
- Leaked headers indicate the public overloads existed in Xcode 14.2-14.3 and were usable in shader code at that time. 
  Source: https://github.com/philipturner/metal-benchmarks (README header excerpt)
- More recent work shows a workaround: accessing the async copy instruction from JIT-compiled shader source without relying on those headers. 
  Source: https://gist.github.com/philipturner/84f613a5cc745460a914d2c6ad226131 (UnifiedGEMMKernel.swift)

**Practical check:** compile a minimal kernel that declares `__metal_simdgroup_async_copy_2d` and `__metal_wait_simdgroup_events` (as in `src/test_async_copy.metal`). If the Metal compiler accepts the inline asm symbols, the toolchain still recognizes the instruction.

## Relationship with simdgroup_matrix_storage.load

`simdgroup_matrix_storage.load` loads a SIMD-group matrix from threadgroup memory. In practice, `simdgroup_async_copy` is the latency-hiding path that stages device memory into threadgroup tiles, and `simdgroup_matrix_storage.load` is the compute-facing load that consumes those tiles. The two are complementary: async copy hides device latency, while matrix-storage load feeds the SIMD-group matrix pipelines.

This relationship is conceptual rather than API-coupled: you can use async copy for any threadgroup tile, but it becomes especially valuable when the next step is a cooperative matrix load/compute sequence.

## Performance characteristics

Percisely’s microbenchmark shows a counterintuitive behavior: copying the entire tile with **one processor (one SIMD-group)** is faster than splitting the copy across multiple processors. The likely reason is that the async copy instruction dispatches extra integer address-generation instructions, which are expensive on Apple GPUs; spreading the work across multiple processors can amplify this overhead. 
Source: https://percisely.xyz/gemm

Takeaways:
- The async copy instruction has **fixed overhead**, so a single SIMD-group (or even a single lane) can be optimal for issuing the copy.
- You still want **overlap**: schedule the next async copy while current compute is running, then wait right before consuming the tile.
- Use **pipeline stages** (double or triple buffering) to maximize overlap.

## Test shader (this repo)

Path: `contrib/metal_marlin/src/test_async_copy.metal`

Contents:
- `test_async_copy_gemm`: tile-based GEMM using `simdgroup_async_copy` to stage A and B.
- `test_sync_copy_gemm`: same GEMM structure, but uses synchronous loads into threadgroup memory.

Both kernels emit `stats` for bandwidth reporting:
- `stats[0]` = bytes read from A
- `stats[1]` = bytes read from B
- `stats[2]` = bytes written to C
- `stats[3]` = FLOPs (2*M*N*K)

### How to measure effective bandwidth

1. Time each kernel execution on the host (e.g., with a GPU timestamp or CPU wall clock).
2. Compute total bytes transferred: `bytes = stats[0] + stats[1] + stats[2]`.
3. Bandwidth (GB/s) = `bytes / time_seconds / 1e9`.
4. Compare `test_async_copy_gemm` vs `test_sync_copy_gemm` for the same M/N/K and grid size.

### Notes

- The async kernel assumes M, N, and K are multiples of 8 and uses a threadgroup size of `(8, 8, 1)`.
- This shader intentionally uses the private instruction form to avoid relying on headers that may not exist in modern SDKs.

## References

- Percisely GEMM writeup and private `__metal_simdgroup_async_copy_2d` usage: https://percisely.xyz/gemm
- Leaked header signatures for `simdgroup_async_copy`: https://github.com/philipturner/metal-benchmarks
- JIT compiler access to async copy in UnifiedGEMMKernel.swift: https://gist.github.com/philipturner/84f613a5cc745460a914d2c6ad226131
