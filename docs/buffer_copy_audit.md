# Buffer Copy Audit (PyObjC/MPS)

## Scope
- `contrib/metal_marlin/metal_marlin/sampler_metal.py`
- `contrib/metal_marlin/metal_marlin/autotune.py`
- `contrib/metal_marlin/metal_marlin/inference_metal.py`
- `contrib/metal_marlin/metal_marlin/kernels.py`
- `contrib/metal_marlin/metal_marlin/mla_kernels.py`
- `contrib/metal_marlin/metal_marlin/_compat.py`

## Summary
- Zero-copy MPS->Metal binding is implemented in `metal_dispatch.py` and `sampler_metal.py` via `newBufferWithBytesNoCopy(..., MTLResourceStorageModeShared, ...)`.
- Multiple CPU round-trips exist for sampling and CPU fallback paths (expected but costly when hit).
- `newBufferWithBytes` is used for scalar/param buffers and for numpy-backed inputs in the sampler; those are real copies.

## Findings by File

### `contrib/metal_marlin/metal_marlin/_compat.py`
- `to_numpy()` uses `arr.detach().cpu().numpy()` for torch tensors, which forces an MPS->CPU copy for MPS tensors. This will copy any tensor that passes through `to_numpy`.
- `from_numpy()` uses `torch.from_numpy(np_arr.copy())`, guaranteeing a copy even if `np_arr` is already contiguous. Comment notes this is to avoid negative strides; this is a deliberate CPU-side copy.
- No MPS tensor -> `MTLBuffer` binding logic exists here; that is handled in `contrib/metal_marlin/metal_marlin/metal_dispatch.py`.
- Shared memory mode for MPS/Metal interop is not configured here.

### `contrib/metal_marlin/metal_marlin/sampler_metal.py`
- `_create_buffer_from_numpy()` uses `newBufferWithBytes_length_options_` with `arr.tobytes()`, which copies numpy data into a new Metal buffer.
- `_create_buffer_from_tensor()` uses `tensor.cpu().numpy()` (MPS->CPU copy) then calls `_create_buffer_from_numpy()` (CPU->Metal copy). This is a double copy; note the method is currently unused in this file.
- `_mps_tensor_to_buffer()` uses `newBufferWithBytesNoCopy_length_options_deallocator_` with `MTLResourceStorageModeShared` for zero-copy MPS tensors.
- Readback paths:
  - `np.frombuffer(self._result_idx_buf.contents().as_buffer(4), ...)` reads from the shared MTLBuffer into CPU; the resulting torch tensor creation (`torch.tensor(..., device="mps")`) copies back to MPS.
  - `topk_vals` and `topk_idxs` read from `.contents().as_buffer(...)` and then `.copy()`, forcing a CPU-side copy of the top-k results.
- Sampling paths using `probs.cpu().numpy()` (`_topp_sample`, `sample`) move MPS tensors to CPU for numpy RNG. This is a per-sample MPS->CPU copy.

### `contrib/metal_marlin/metal_marlin/autotune.py`
- Uses `newBufferWithBytes_length_options_` for scalar constants (`M`, `N`, `K`, `group_size`). These are tiny copies and likely fine.
- Metal buffer binding uses `mps_tensor_to_metal_buffer()` (zero-copy) for tensor inputs/outputs.

### `contrib/metal_marlin/metal_marlin/kernels.py`
- Weight packers (`pack_fp4_weights`, `pack_int2_weights`, `pack_int3_weights`) use `.cpu().float().numpy()` to quantize in numpy. This forces a full MPS->CPU copy and CPU quantization. If inputs are already on CPU, this is expected; if inputs are on MPS, it is an intentional downshift.
- `dequant_u4_standalone`, `dequant_int2`, `dequant_int3` use CPU numpy paths (`.cpu().numpy()`), then copy back to MPS with `torch.from_numpy(...).to("mps")`. These are CPU fallback/debug paths with explicit round-trips.
- Kernel dispatch paths use `mps_tensor_to_metal_buffer()` with shared storage; params use `newBufferWithBytes_length_options_` (copy).

### `contrib/metal_marlin/metal_marlin/mla_kernels.py`
- Metal dispatch functions (`_dispatch_mla_proj`, `_dispatch_mla_decode`, `_dispatch_mla_fused`) create scalar constant buffers with `newBufferWithBytes_length_options_` (copies).
- numpy->MPS conversions:
  - `torch.from_numpy(np.asarray(...)).to("mps")` is used in multiple functions (`mla_proj_fp4`, `mla_decode_proj_fp4`, `mla_fused_kv_proj_fp4`, `mla_proj_with_rope_fp4`). This always performs a CPU->MPS copy; `np.asarray` may also copy if the input is not already an ndarray.
- No `.cpu()` calls appear in the hot Metal dispatch path, but the numpy fallback path materializes CPU arrays and computes on CPU (expected).

### `contrib/metal_marlin/metal_marlin/inference_metal.py`
- No direct `newBufferWithBytes`/`newBufferWithBytesNoCopy`, `.cpu()`, or `np.frombuffer` usages found here. This file relies on `metal_dispatch` for buffer binding.

## MPS Tensor -> MTLBuffer Binding / Shared Memory
- The binding is implemented in `contrib/metal_marlin/metal_marlin/metal_dispatch.py`:
  - `mps_tensor_to_metal_buffer()` uses `tensor.untyped_storage().data_ptr()` + `storage.nbytes()` and `device.newBufferWithBytesNoCopy_length_options_deallocator_(..., Metal.MTLResourceStorageModeShared, None)`.
  - `MetalKernelLibrary._get_metal_buffer()` uses `tensor.data_ptr()` + `tensor.numel() * tensor.element_size()` with the same `newBufferWithBytesNoCopy` + `MTLResourceStorageModeShared`.
- Shared storage mode is explicitly used for MPS interop (`MTLResourceStorageModeShared`).

## Notable Copy Hotspots (by pattern)
- `newBufferWithBytes`: scalar params and numpy buffers (sampler + autotune + kernels + mla kernels). All are copies.
- `numpy -> Metal` via `torch.from_numpy(...).to("mps")`: MLA wrappers, CPU fallbacks, and weight packers (if invoked on MPS inputs).
- `.contents().as_buffer() + np.frombuffer`: sampler readback; `metal_dispatch.metal_buffer_to_numpy()` uses `.copy()`.
- `.cpu()` before Metal dispatch: none in the core dispatch paths, but `_create_buffer_from_tensor()` (unused) and pack/dequant helpers perform MPS->CPU copies.
