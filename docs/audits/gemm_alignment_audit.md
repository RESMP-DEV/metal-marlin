# GEMM alignment audit (M/N/K divisibility by 8)

Status (2026-01-28)
- `metal_dispatch.dispatch_gemm_fp4` now pads M/N/K (and packed weights/scales)
  when padding is enabled, addressing the earlier “no N/K padding” gap for that
  dispatch path.
- `metal_marlin/kernels.py` pads M to full tiles for `marlin_gemm_fp4` and uses
  `dispatch_gemm_fp4(..., enable_padding=False)` after that manual padding.
- Weight packers in `hf_loader.py` and `mr_gptq.py` still require K to be divisible
  by 8 and `group_size` (no automatic padding there).

## Why this matters
Metal MPS matmul performance drops when M, N, or K are not divisible by 8. Most
kernels use simdgroup 8x8 tiles and FP4 packing (8 values per uint32), so alignment
affects both throughput and row-stride efficiency.

## Current padding behavior

### `contrib/metal_marlin/metal_marlin/metal_dispatch.py`
- `dispatch_gemm_fp4`:
  - Pads K to `max(_PAD_MULTIPLE, group_size)`.
  - Pads N to `pad_n_multiple` (and TILE sizes for fused kernel variants).
  - Pads A on M (and K) to match padded sizes.
  - Pads packed weights/scales to the same padded K/N.

### `contrib/metal_marlin/metal_marlin/kernels.py`
- `marlin_gemm_fp4`:
  - Pads M to full `TILE_M` before dispatch, then slices output back.
  - Calls `dispatch_gemm_fp4(..., enable_padding=False)` after M padding.
- `marlin_gemm_int4` (metal path):
  - Pads M to a multiple of 8 (no N/K padding here).

### `contrib/metal_marlin/metal_marlin/hf_loader.py`
- `should_quantize_tensor` rejects `in_feat % 8 != 0` and `in_feat % group_size != 0`.

### `contrib/metal_marlin/metal_marlin/mr_gptq.py`
- Packing helpers require `in_feat % 8 == 0` and `in_feat % group_size == 0`.

## Remaining gaps
- N/K padding is only handled in `dispatch_gemm_fp4`; other GEMM kernels still
  assume aligned packed layouts and rely on bounds checks.
- Weight packing paths reject non-8-aligned K instead of padding.
- Decode paths (M=1) still incur partial tiles unless padded externally.

## Resolved issues
- “No N/K padding in GEMM dispatch” is resolved for `dispatch_gemm_fp4`.
  See `docs/resolved_bugs.md`.
