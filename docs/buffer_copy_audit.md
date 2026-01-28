# Buffer Copy Audit (PyObjC/MPS)

Status (2026-01-28)
- Zero-copy MPS -> Metal bindings are still used where possible.
- Private-buffer staging (managed -> private blit) was added for some GEMM paths,
  which reduces shared-memory usage but does not eliminate CPU/GPU copies.
- CPU round-trips remain in sampler and CPU fallback paths.

## Scope
- `contrib/metal_marlin/metal_marlin/sampler_metal.py`
- `contrib/metal_marlin/metal_marlin/autotune.py`
- `contrib/metal_marlin/metal_marlin/inference_metal.py`
- `contrib/metal_marlin/metal_marlin/kernels.py`
- `contrib/metal_marlin/metal_marlin/mla_kernels.py`
- `contrib/metal_marlin/metal_marlin/_compat.py`
- `contrib/metal_marlin/metal_marlin/metal_dispatch.py`

## Summary
- Zero-copy MPS->Metal binding is implemented in `metal_dispatch.py` and
  `sampler_metal.py` via `newBufferWithBytesNoCopy(..., MTLResourceStorageModeShared, ...)`.
- Private-buffer staging is used in `kernels.py` and parts of `metal_dispatch.py`
  (`_private_buffer_from_bytes` / `_private_buffer_from_tensor`) to move data into
  GPU-only buffers via a blit. This avoids shared buffers but is still a copy.
- CPU round-trips exist for sampling and CPU fallback paths (expected but costly
  when hit).
- `newBufferWithBytes` is still used for scalar/param buffers and for numpy-backed
  inputs in the sampler; those are real copies.

## Findings by file

### `contrib/metal_marlin/metal_marlin/_compat.py`
- `to_numpy()` uses `arr.detach().cpu().numpy()` for torch tensors, forcing an
  MPS->CPU copy for MPS tensors.
- `from_numpy()` uses `torch.from_numpy(np_arr.copy())`, guaranteeing a copy even
  if `np_arr` is already contiguous (intentional to avoid negative strides).

### `contrib/metal_marlin/metal_marlin/sampler_metal.py`
- `_create_buffer_from_numpy()` uses `newBufferWithBytes_length_options_` with
  `arr.tobytes()`, copying numpy data into a new Metal buffer.
- `_create_buffer_from_tensor()` uses `tensor.cpu().numpy()` (MPS->CPU copy), then
  `_create_buffer_from_numpy()` (CPU->Metal copy). Method remains unused.
- `_mps_tensor_to_buffer()` uses `newBufferWithBytesNoCopy...` (zero-copy).
- Readback paths:
  - `np.frombuffer(self._result_idx_buf.contents().as_buffer(4), ...)` reads from
    shared MTLBuffer into CPU; converting back to MPS (`torch.tensor(..., device="mps")`)
    copies again.
  - `topk_vals` and `topk_idxs` read from `.contents().as_buffer(...)` and then
    `.copy()`, forcing CPU-side copies.
- Sampling paths using `probs.cpu().numpy()` (`_topp_sample`, `sample`) move MPS
  tensors to CPU for numpy RNG (per-sample MPS->CPU copy).

### `contrib/metal_marlin/metal_marlin/autotune.py`
- Uses `newBufferWithBytes_length_options_` for scalar constants (`M`, `N`, `K`,
  `group_size`). These are small copies.
- Uses `mps_tensor_to_metal_buffer()` for tensor inputs/outputs (zero-copy on MPS).

### `contrib/metal_marlin/metal_marlin/kernels.py`
- GEMM dispatch paths now use private buffers for weights/params via staging +
  blit (`_private_buffer_from_bytes`, `_private_buffer_from_tensor`).
- Weight packers (`pack_fp4_weights`, `pack_int2_weights`, `pack_int3_weights`)
  use `.cpu().float().numpy()` to quantize in numpy (MPS->CPU copy).
- Dequant helpers (`dequant_u4_standalone`, `dequant_int2`, `dequant_int3`) use
  CPU numpy paths then copy back to MPS; these are CPU fallback/debug paths.

### `contrib/metal_marlin/metal_marlin/mla_kernels.py`
- Scalar params are built with `newBufferWithBytes_length_options_` (copies).
- `torch.from_numpy(...).to("mps")` is used in multiple MLA wrappers, which always
  performs a CPU->MPS copy.

### `contrib/metal_marlin/metal_marlin/metal_dispatch.py`
- Zero-copy MPS binding uses `newBufferWithBytesNoCopy...` with shared storage.
- Some dispatch paths still allocate small parameter buffers via
  `newBufferWithBytes_length_options_` (copies).

### `contrib/metal_marlin/metal_marlin/inference_metal.py`
- No direct `newBuffer*`/`.cpu()`/`np.frombuffer` usage found; relies on
  `metal_dispatch` for buffer binding.

## Notable copy hotspots (by pattern)
- `newBufferWithBytes`: scalar params and numpy buffers (sampler/autotune/MLA/metal_dispatch).
- `numpy -> MPS`: `torch.from_numpy(...).to("mps")` in MLA wrappers and CPU fallbacks.
- `.contents().as_buffer() + np.frombuffer`: sampler readback paths and
  `metal_dispatch.metal_buffer_to_numpy()`.
- `.cpu()` before Metal dispatch: pack/dequant helpers and sampler CPU fallbacks.
