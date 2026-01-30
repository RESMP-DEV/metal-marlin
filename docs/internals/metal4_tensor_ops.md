# Metal 4 Tensor Ops (matmul2d) Research Notes

Scope: Forward-looking investigation of Metal 4 tensor APIs for matmul, with a focus on API surface, memory behavior, alignment requirements, and how this abstraction compares to our current Metal 3.1 kernels.

## Sources
- https://github.com/liuliu/example_matmul_metal4
- https://raw.githubusercontent.com/liuliu/example_matmul_metal4/main/Sources/matmul/shader.metal
- https://raw.githubusercontent.com/liuliu/example_matmul_metal4/main/README.md
- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- https://developer.apple.com/videos/play/wwdc2025/315/

## 1) Tensor API signatures for matmul (observed)
The example repo compiles with `metal_tensor` and `MetalPerformancePrimitives` headers and uses `mpp::tensor_ops`.

Key includes and namespace:
```cpp
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;
```

### Tensor construction
Dynamic extents:
```cpp
auto A = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
    A_buf, dextents<int32_t, 2>(256, 128));
```

Static extents:
```cpp
auto A = tensor<device half, extents<int32_t, 256, 128>, tensor_inline>(
    A_buf, extents<int32_t, 256, 128>());
```

Observed components:
- `tensor<address_space T, extents_type, layout_tag>`
- `dextents<index_t, rank>(...)` for dynamic extents
- `extents<index_t, ...>()` for static extents
- `tensor_inline` layout tag

### Slicing
Dynamic slices:
```cpp
auto mA = A.slice(0, tgid.y * 64);
```

Static slices:
```cpp
auto mA = A.slice<16, 64>(k, tgid.y * 64);
```

### Matmul descriptor and op
```cpp
constexpr auto matmulDescriptor = matmul2d_descriptor(
    64,
    32,
    dynamic_length_v<int>,
    false,
    false,
    false,
    matmul2d_descriptor::mode::multiply);

matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;
matmulOp.run(mA, mB, mC);
```

And for accumulate mode with explicit K tile:
```cpp
constexpr auto matmulDescriptor = matmul2d_descriptor(
    64,
    32,
    16,
    false,
    false,
    false,
    matmul2d_descriptor::mode::multiply_accumulate);
```

Notes:
- The descriptor signature appears to be: `(tile_m, tile_n, tile_k, flag0, flag1, flag2, mode)`.
- The three boolean flags are not documented in the example; likely transpose / layout flags, but need confirmation from the MSL spec.
- `dynamic_length_v<int>` is used as the tile-K when automatic slicing is desired.
- `matmul2d<descriptor, execution_simdgroups<N>>` suggests the op is parametrized by tile + SIMD-group count.

## 2) Memory management notes (device vs threadgroup)
Observations from the example repo:
- Tensors are created directly from `device` pointers (e.g. `device half *A_buf`), implying data resides in device memory and is bound via standard Metal 3 buffer bindings.
- The README states Apple has abstracted tensor APIs heavily; a disassembly indicates calls into packaged Apple APIs rather than user-managed shared memory.
- The README further states no special memory regions are needed beyond device and threadgroup memory; this contrasts with CUDA tensor memory or explicit shared-memory tiling.

Implications:
- The tensor op appears to manage its own internal threadgroup usage.
- Our host-side memory management likely remains buffer-centric (Metal 3-style), even when targeting Metal 4 tensor ops.
- The `tensor_inline` tag likely indicates inline layout without requiring explicit threadgroup staging.

## 3) K-dimension alignment requirement
From the example repo README:
- On Xcode 26.1, `matmul2d_descriptor` with an explicit K requires K to be a multiple of 32; otherwise it is truncated to a multiple of 32, producing incorrect results.
- Using `dynamic_length_v<int>` avoids the truncation with minimal performance impact.

Actionable takeaway:
- Treat the explicit K parameter as alignment-sensitive (multiple of 32) unless verified otherwise.
- Prefer `dynamic_length_v<int>` if K is not a clean multiple of 32 or if K varies at runtime.

## 4) Abstraction vs our current kernels
Current Metal 3.1 kernels:
- Manually manage tiles, threadgroup memory, and the K-loop.
- Explicitly control data layout, swizzling, and synchronization.
- Fuse quantization/dequantization logic with compute for bandwidth savings.

Metal 4 tensor ops (matmul2d):
- Provide a higher-level interface that abstracts tiling and use of ML accelerators.
- Kernel code becomes a lightweight orchestrator: build tensors, slice, and call `matmulOp.run`.
- Less explicit control over memory placement, layout, and custom fusion.

Expected tradeoffs:
- Pros: simpler code, likely better utilization of Apple ML hardware, reduced manual tuning.
- Cons: reduced ability to fuse custom quantization and sparse layouts; less control over tile sizes and threadgroup usage beyond descriptor parameters.

## 5) Relation to MLX (WWDC 2025 session 315)
The MLX session focuses on unified memory and custom Metal kernels for high-level ML workflows. It does not document tensor ops directly, but it reinforces:
- Unified memory programming model (CPU/GPU share memory).
- Custom Metal kernels can be injected into MLX as a JIT operation.

This suggests tensor ops are one (lower-level) path for acceleration, while MLX targets higher-level array/tensor workflows. Metal 4 tensor ops could be a backend for MLX-style kernels, but that link is not documented in the session.

## Open questions / follow-ups
- Confirm the exact meaning of the three boolean parameters in `matmul2d_descriptor` from the MSL v4 spec.
- Check whether `tensor` supports bfloat or only half/float for cooperative tensor ops.
- Validate the K alignment behavior on real hardware (Xcode 26.1+), especially for dynamic slicing.
- Measure performance and compatibility on macOS 26+ while keeping Metal 3.1 fallback for macOS 14+.
