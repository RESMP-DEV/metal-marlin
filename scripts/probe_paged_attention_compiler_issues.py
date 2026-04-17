#!/usr/bin/env python3
"""Probe potential paged-attention compiler hazards on Apple Metal.

This script investigates two hypotheses from
`docs/paged_attention_metal_correctness_investigation.md` without depending on
the external `kernels-community/paged-attention` package:

1. H1: `fma(half, half, float)`-style accumulation drifts for fractional FP16
   values when the cast to float happens inside the helper.
2. H2/H3: array-reference helpers like `qk_dot_` and the `Qk_dot::dot`
   wrapper miscompile when passed `threadgroup`/`thread` array references.

The probes run through `metal_marlin`'s existing PyTorch MPS + Metal dispatch
path so they exercise the local Metal compiler/runtime stack directly.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metal_marlin._compat import HAS_MPS
from metal_marlin.metal_dispatch import (
    MetalKernelLibrary,
    _copy_buffer_to_tensor,
    _CopyBackBuffer,
    mps_tensor_to_metal_buffer,
)

PROBE_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

constant constexpr uint H1_HEAD_DIM = 128;
constant constexpr uint H2_VEC_COUNT = 4;

inline float fma_buggy(half a, half b, float c) {
    return (float)a * (float)b + c;
}

inline float fma_safe(float a, float b, float c) {
    return a * b + c;
}

inline float dot_buggy(device const half* q, device const half* k_row) {
    float acc = 0.0f;
#pragma unroll
    for (uint i = 0; i < H1_HEAD_DIM; ++i) {
        acc = fma_buggy(q[i], k_row[i], acc);
    }
    return acc;
}

inline float dot_safe(device const half* q, device const half* k_row) {
    float acc = 0.0f;
#pragma unroll
    for (uint i = 0; i < H1_HEAD_DIM; ++i) {
        acc = fma_safe(float(q[i]), float(k_row[i]), acc);
    }
    return acc;
}

kernel void probe_half_param_fma(
    device const half* q              [[buffer(0)]],
    device const half* k              [[buffer(1)]],
    device float* out                 [[buffer(2)]],
    uint tid                          [[thread_position_in_grid]]
) {
    device const half* k_row = k + tid * H1_HEAD_DIM;
    out[tid] = dot_buggy(q, k_row);
    out[16 + tid] = dot_safe(q, k_row);
}

template <typename Vec, int N>
inline float qk_dot_(const threadgroup Vec (&q)[N], const thread Vec (&k)[N]) {
    float4 acc = float4(q[0]) * float4(k[0]);
#pragma unroll
    for (int ii = 1; ii < N; ++ii) {
        acc += float4(q[ii]) * float4(k[ii]);
    }
    return acc.x + acc.y + acc.z + acc.w;
}

template <typename T, int THREAD_GROUP_SIZE>
struct Qk_dot {
    template <typename Vec, int N>
    inline static float dot(const threadgroup Vec (&q)[N], const thread Vec (&k)[N]) {
        return qk_dot_<Vec, N>(q, k);
    }
};

kernel void probe_array_ref_qk(
    device const half4* q_in          [[buffer(0)]],
    device const half4* k_in          [[buffer(1)]],
    device float* out                 [[buffer(2)]],
    uint tid                          [[thread_index_in_threadgroup]]
) {
    threadgroup half4 q_local[H2_VEC_COUNT];
    if (tid < H2_VEC_COUNT) {
        q_local[tid] = q_in[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    half4 k_local[H2_VEC_COUNT];
    for (uint i = 0; i < H2_VEC_COUNT; ++i) {
        k_local[i] = k_in[i];
    }

    if (tid == 0) {
        out[0] = qk_dot_<half4, H2_VEC_COUNT>(q_local, k_local);
        out[1] = Qk_dot<half, 32>::dot(q_local, k_local);
    }
}
"""


def _copy_back_if_needed(buffer_wrapper: object, tensor: torch.Tensor) -> None:
    if isinstance(buffer_wrapper, _CopyBackBuffer):
        _copy_buffer_to_tensor(buffer_wrapper.buffer, tensor)
    torch.mps.synchronize()


def _to_mps_tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(array).to(device="mps", dtype=dtype).contiguous()


def _run_half_param_probe(lib: MetalKernelLibrary) -> dict[str, object]:
    head_dim = 128
    targets = np.array(
        [
            5.5,
            5.25,
            8.25,
            8.75,
            3.0,
            2.0,
            1.0,
            0.5,
            0.25,
            0.125,
            4.5,
            6.5,
            7.5,
            9.5,
            10.0,
            11.0,
        ],
        dtype=np.float32,
    )

    q = np.full((head_dim,), 1.0 / math.sqrt(head_dim), dtype=np.float16)
    k = np.zeros((targets.size, head_dim), dtype=np.float16)
    k[:, 0] = (targets * math.sqrt(head_dim)).astype(np.float16)

    q_mps = _to_mps_tensor(q, torch.float16)
    k_mps = _to_mps_tensor(k, torch.float16)
    out = torch.empty((targets.size * 2,), dtype=torch.float32, device="mps")

    q_buf = mps_tensor_to_metal_buffer(q_mps, lib.device)
    k_buf = mps_tensor_to_metal_buffer(k_mps, lib.device)
    out_wrapper = mps_tensor_to_metal_buffer(out, lib.device, copy_back=True)
    out_buf = out_wrapper.buffer if isinstance(out_wrapper, _CopyBackBuffer) else out_wrapper

    kernel = lib.get_pipeline("probe_half_param_fma", "paged_attention_compiler_probe")
    lib._dispatch(kernel, (targets.size, 1, 1), (1, 1, 1), q_buf, k_buf, out_buf)
    _copy_back_if_needed(out_wrapper, out)

    out_np = out.cpu().numpy()
    buggy = out_np[: targets.size]
    safe = out_np[targets.size :]
    ref = (q.astype(np.float64)[None, :] * k.astype(np.float64)).sum(axis=1)

    buggy_abs = np.abs(buggy - ref)
    safe_abs = np.abs(safe - ref)
    buggy_vs_safe = np.abs(buggy - safe)

    return {
        "targets": [float(x) for x in targets],
        "buggy": [float(x) for x in buggy],
        "safe": [float(x) for x in safe],
        "reference": [float(x) for x in ref],
        "max_buggy_abs_diff": float(np.max(buggy_abs)),
        "max_safe_abs_diff": float(np.max(safe_abs)),
        "max_buggy_vs_safe_diff": float(np.max(buggy_vs_safe)),
        "suspicious": bool(np.max(buggy_vs_safe) > 5e-3 or np.max(buggy_abs) > 5e-3),
    }


def _run_array_ref_probe(lib: MetalKernelLibrary) -> dict[str, object]:
    q = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=np.float16,
    )
    k = np.array(
        [
            [0.5, 1.0, 1.5, 2.0],
            [2.5, 3.0, 3.5, 4.0],
            [4.5, 5.0, 5.5, 6.0],
            [6.5, 7.0, 7.5, 8.0],
        ],
        dtype=np.float16,
    )

    q_mps = _to_mps_tensor(q, torch.float16)
    k_mps = _to_mps_tensor(k, torch.float16)
    out = torch.empty((2,), dtype=torch.float32, device="mps")

    q_buf = mps_tensor_to_metal_buffer(q_mps, lib.device)
    k_buf = mps_tensor_to_metal_buffer(k_mps, lib.device)
    out_wrapper = mps_tensor_to_metal_buffer(out, lib.device, copy_back=True)
    out_buf = out_wrapper.buffer if isinstance(out_wrapper, _CopyBackBuffer) else out_wrapper

    kernel = lib.get_pipeline("probe_array_ref_qk", "paged_attention_compiler_probe")
    lib._dispatch(kernel, (1, 1, 1), (32, 1, 1), q_buf, k_buf, out_buf)
    _copy_back_if_needed(out_wrapper, out)

    direct, wrapped = [float(x) for x in out.cpu().numpy()]
    ref = float(np.sum(q.astype(np.float32) * k.astype(np.float32)))
    collapse_ref = float(q.shape[0] * np.sum(q[0].astype(np.float32) * k[0].astype(np.float32)))

    return {
        "direct": direct,
        "wrapped": wrapped,
        "reference": ref,
        "index0_collapse_reference": collapse_ref,
        "direct_abs_diff": abs(direct - ref),
        "wrapped_abs_diff": abs(wrapped - ref),
        "wrapper_vs_direct_diff": abs(wrapped - direct),
        "suspicious": bool(
            abs(direct - ref) > 1e-3
            or abs(wrapped - ref) > 1e-3
            or abs(wrapped - direct) > 1e-3
        ),
    }


def main() -> None:
    if not HAS_MPS:
        raise SystemExit("This probe requires PyTorch MPS on Apple Silicon.")

    lib = MetalKernelLibrary()
    lib.compile_source("paged_attention_compiler_probe", PROBE_SOURCE)

    half_probe = _run_half_param_probe(lib)
    array_probe = _run_array_ref_probe(lib)

    report = {
        "source": str(
            Path("contrib/metal_marlin/docs/paged_attention_metal_correctness_investigation.md")
        ),
        "half_param_probe": half_probe,
        "array_ref_probe": array_probe,
        "summary": {
            "half_param_issue_detected": half_probe["suspicious"],
            "array_ref_issue_detected": array_probe["suspicious"],
        },
    }

    print("=== Paged Attention Compiler Probe ===")
    print(f"H1 suspicious: {half_probe['suspicious']}")
    print(
        "  max buggy abs diff vs ref:",
        f"{half_probe['max_buggy_abs_diff']:.6e}",
        "| max safe abs diff vs ref:",
        f"{half_probe['max_safe_abs_diff']:.6e}",
        "| max buggy-safe gap:",
        f"{half_probe['max_buggy_vs_safe_diff']:.6e}",
    )
    print(f"H2/H3 suspicious: {array_probe['suspicious']}")
    print(
        "  direct:",
        f"{array_probe['direct']:.6f}",
        "| wrapped:",
        f"{array_probe['wrapped']:.6f}",
        "| ref:",
        f"{array_probe['reference']:.6f}",
        "| collapse-ref:",
        f"{array_probe['index0_collapse_reference']:.6f}",
    )
    print("--- JSON ---")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
