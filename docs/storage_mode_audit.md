# Storage Mode Audit (Metal buffers)

Scope: `metal_marlin/kernels.py`, `metal_marlin/sampler_metal.py`,
`metal_marlin/autotune.py`, `metal_marlin/inference_metal.py`,
`metal_marlin/mla_kernels.py`.

Rules applied:
- GPU-only buffers -> `MTLResourceStorageModePrivate`
- CPU-write, GPU-read -> `MTLResourceStorageModeManaged` (with explicit sync, e.g.
  `didModifyRange_` after CPU writes)
- Frequent CPU read-back -> `MTLResourceStorageModeShared`

## `metal_marlin/autotune.py`

Function: `run_kernel` (around line 260)

| Buffer | Current mode | Purpose | Access pattern | Suggested mode |
| --- | --- | --- | --- | --- |
| `M_buf` | Shared | Constant M parameter | CPU write, GPU read | Managed |
| `N_buf` | Shared | Constant N parameter | CPU write, GPU read | Managed |
| `K_buf` | Shared | Constant K parameter | CPU write, GPU read | Managed |
| `gs_buf` | Shared | Constant group_size parameter | CPU write, GPU read | Managed |

## `metal_marlin/kernels.py`

Function: `marlin_gemm_fp4` (around line 1146)

| Buffer | Current mode | Purpose | Access pattern | Suggested mode |
| --- | --- | --- | --- | --- |
| `params_buf` | Shared | Constant params (M, N, K, group_size) | CPU write, GPU read | Managed |

Function: `marlin_gemm_int4` (around line 1230)

| Buffer | Current mode | Purpose | Access pattern | Suggested mode |
| --- | --- | --- | --- | --- |
| `params_buf` | Shared | Constant params (M, N, K, group_size) | CPU write, GPU read | Managed |

Function: `dequant_fp4` (around line 1295)

| Buffer | Current mode | Purpose | Access pattern | Suggested mode |
| --- | --- | --- | --- | --- |
| `params_buf` | Shared | Constant params (K, N, group_size) | CPU write, GPU read | Managed |

Function: `hadamard_transform` (around line 1528)

| Buffer | Current mode | Purpose | Access pattern | Suggested mode |
| --- | --- | --- | --- | --- |
| `params_buf` | Shared | Constant params (n, normalize flag) | CPU write, GPU read | Managed |

Function: `decode_gemv_fp4` (around line 1601)

| Buffer | Current mode | Purpose | Access pattern | Suggested mode |
| --- | --- | --- | --- | --- |
| `params_buf` | Shared | Constant params (K, N, group_size) | CPU write, GPU read | Managed |

## `metal_marlin/mla_kernels.py`

Function: `_dispatch_mla_proj` (around line 212)

| Buffer | Current mode | Purpose | Access pattern | Suggested mode |
| --- | --- | --- | --- | --- |
| `M_buf` | Shared | Constant M parameter | CPU write, GPU read | Managed |
| `N_buf` | Shared | Constant N parameter | CPU write, GPU read | Managed |
| `K_buf` | Shared | Constant K parameter | CPU write, GPU read | Managed |
| `gs_buf` | Shared | Constant group_size parameter | CPU write, GPU read | Managed |

Function: `_dispatch_mla_decode` (around line 274)

| Buffer | Current mode | Purpose | Access pattern | Suggested mode |
| --- | --- | --- | --- | --- |
| `K_buf` | Shared | Constant K parameter | CPU write, GPU read | Managed |
| `N_buf` | Shared | Constant N parameter | CPU write, GPU read | Managed |
| `gs_buf` | Shared | Constant group_size parameter | CPU write, GPU read | Managed |

Function: `_dispatch_mla_fused` (around line 336)

| Buffer | Current mode | Purpose | Access pattern | Suggested mode |
| --- | --- | --- | --- | --- |
| `M_buf` | Shared | Constant M parameter | CPU write, GPU read | Managed |
| `Kh_buf` | Shared | Constant K_hidden parameter | CPU write, GPU read | Managed |
| `Kl_buf` | Shared | Constant K_latent parameter | CPU write, GPU read | Managed |
| `Nout_buf` | Shared | Constant N_out parameter | CPU write, GPU read | Managed |
| `gsa_buf` | Shared | Constant group_size_a parameter | CPU write, GPU read | Managed |
| `gsb_buf` | Shared | Constant group_size_b parameter | CPU write, GPU read | Managed |

## `metal_marlin/sampler_metal.py`

Function: `_init_buffers` (around line 765)

| Buffer | Current mode | Purpose | Access pattern | Suggested mode |
| --- | --- | --- | --- | --- |
| `_scalar_buf` | Shared | Reserved scalar scratch (unused in code) | Unknown (no observed use) | Private if GPU-only; otherwise Shared |
| `_result_idx_buf` | Shared | Argmax index output | GPU write, CPU read | Shared |
| `_result_val_buf` | Shared | Argmax value output | GPU write, no CPU read observed | Private |
| `_block_maxes_buf` | Shared | Softmax block maxes (multi-pass) | GPU write/read | Private |
| `_block_sums_buf` | Shared | Softmax block sums (multi-pass) | GPU write/read | Private |
| `_topk_vals_buf` | Shared | Top-k values output | GPU write, CPU read | Shared |
| `_topk_idxs_buf` | Shared | Top-k indices output | GPU write, CPU read | Shared |

Function: `_create_buffer_from_numpy` (around line 794)

| Buffer | Current mode | Purpose | Access pattern | Suggested mode |
| --- | --- | --- | --- | --- |
| `buf` | Shared | Constant buffers (vocab_size, batch_size, k, etc.) | CPU write, GPU read | Managed |

Function: `_mps_tensor_to_buffer` (around line 819)

| Buffer | Current mode | Purpose | Access pattern | Suggested mode |
| --- | --- | --- | --- | --- |
| `buffer` | Shared | Alias MPS tensor storage (zero-copy) | GPU read, CPU access not used here | Shared (required for no-copy) |

Function: `_greedy_sample_metal` (around line 895)

| Buffer | Current mode | Purpose | Access pattern | Suggested mode |
| --- | --- | --- | --- | --- |
| `results_buf` | Shared | Batched argmax indices | GPU write, CPU read | Shared |

## `metal_marlin/inference_metal.py`

No `newBuffer*` calls in this file.
