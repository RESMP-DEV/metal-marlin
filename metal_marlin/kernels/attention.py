"""Attention dispatch kernels extracted from the legacy kernels monolith."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


_PAGED_ATTENTION_SHADER_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "paged_attention.metal"
)

_QUANTIZED_KV_ATTN_DECODE_KERNEL = """
#include <metal_stdlib>
using namespace metal;

constant constexpr uint THREADS_PER_TG = 128;
constant constexpr uint NUM_SIMDGROUPS = THREADS_PER_TG / 32;

inline float reduce_sum_tg(threadgroup float* scratch, float value, uint tid) {
    float sg_sum = simd_sum(value);
    uint lane = tid & 31u;
    uint sg_id = tid >> 5u;
    if (lane == 0u) {
        scratch[sg_id] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        float v = (lane < NUM_SIMDGROUPS) ? scratch[lane] : 0.0f;
        float total = simd_sum(v);
        if (lane == 0u) {
            scratch[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

inline half load_quantized_kv(
    device const uint8_t* cache,
    device const half* scales,
    uint seq_idx,
    uint kv_head,
    uint dim_idx,
    uint num_kv_heads,
    uint head_dim,
    uint num_scale_groups,
    uint group_size,
    uint quant_mode
) {
    uint group_idx = dim_idx / max(group_size, 1u);
    if (group_idx >= num_scale_groups) {
        group_idx = num_scale_groups - 1u;
    }
    uint scale_idx = (seq_idx * num_kv_heads + kv_head) * num_scale_groups + group_idx;
    float s = float(scales[scale_idx]);

    if (quant_mode == 1u) {
        // FP4 packed, 2 values per byte.
        uint packed_dim = (head_dim + 1u) / 2u;
        uint base = (seq_idx * num_kv_heads + kv_head) * packed_dim;
        uint8_t packed = cache[base + (dim_idx >> 1u)];
        uint8_t nibble = (dim_idx & 1u) ? ((packed >> 4u) & 0xFu) : (packed & 0xFu);
        float q = float(int(nibble) - 8);
        return half(q * s);
    }

    // INT8 symmetric stored as uint8 with +128 offset.
    uint base = (seq_idx * num_kv_heads + kv_head) * head_dim;
    uint8_t raw = cache[base + dim_idx];
    float q = float(int(raw) - 128);
    return half(q * s);
}

kernel void quantized_kv_attention_decode(
    device const half* q                [[buffer(0)]],   // [num_heads_q, head_dim]
    device const uint8_t* k_cache       [[buffer(1)]],   // [seq_len, num_kv_heads, packed_or_head_dim]
    device const uint8_t* v_cache       [[buffer(2)]],   // [seq_len, num_kv_heads, packed_or_head_dim]
    device const half* k_scales         [[buffer(3)]],   // [seq_len, num_kv_heads, num_scale_groups]
    device const half* v_scales         [[buffer(4)]],   // [seq_len, num_kv_heads, num_scale_groups]
    device half* out                    [[buffer(5)]],   // [num_heads_q, head_dim]
    constant uint& seq_len              [[buffer(6)]],
    constant uint& num_heads_q          [[buffer(7)]],
    constant uint& num_kv_heads         [[buffer(8)]],
    constant uint& head_dim             [[buffer(9)]],
    constant uint& group_size           [[buffer(10)]],
    constant uint& num_scale_groups     [[buffer(11)]],
    constant uint& quant_mode           [[buffer(12)]],  // 1=fp4, 2=int8
    constant float& attn_scale          [[buffer(13)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint tid                            [[thread_index_in_threadgroup]]
) {
    uint head_q = tgid.x;
    if (head_q >= num_heads_q || num_kv_heads == 0u || num_scale_groups == 0u) return;
    uint kv_head = head_q % num_kv_heads;

    threadgroup float scratch[NUM_SIMDGROUPS];
    threadgroup float running_max;
    threadgroup float running_sum;
    threadgroup float alpha_shared;
    threadgroup float beta_shared;

    if (tid == 0u) {
        running_max = -INFINITY;
        running_sum = 0.0f;
        alpha_shared = 0.0f;
        beta_shared = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float qv = (tid < head_dim) ? float(q[head_q * head_dim + tid]) : 0.0f;
    float out_acc = 0.0f;

    for (uint t = 0u; t < seq_len; ++t) {
        float kv = 0.0f;
        if (tid < head_dim) {
            kv = float(load_quantized_kv(
                k_cache, k_scales, t, kv_head, tid,
                num_kv_heads, head_dim, num_scale_groups, group_size, quant_mode
            ));
        }

        float dot = qv * kv;
        float dot_sum = reduce_sum_tg(scratch, dot, tid);

        if (tid == 0u) {
            float score = dot_sum * attn_scale;
            float new_max = max(running_max, score);
            float alpha = exp(running_max - new_max);
            float beta = exp(score - new_max);
            running_sum = running_sum * alpha + beta;
            running_max = new_max;
            alpha_shared = alpha;
            beta_shared = beta;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < head_dim) {
            float vv = float(load_quantized_kv(
                v_cache, v_scales, t, kv_head, tid,
                num_kv_heads, head_dim, num_scale_groups, group_size, quant_mode
            ));
            out_acc = out_acc * alpha_shared + beta_shared * vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < head_dim) {
        float denom = max(running_sum, 1e-8f);
        out[head_q * head_dim + tid] = half(out_acc / denom);
    }
}
"""


def build_attention_exports(
    *,
    np: Any,
    torch: Any,
    require_mps: Callable[..., Any],
    get_default_library: Callable[..., Any],
    _ensure_kernel_compiled: Callable[..., Any],
    _private_buffer_from_tensor: Callable[..., Any],
    _params_buffer: Callable[..., Any],
    mps_tensor_to_metal_buffer: Callable[..., Any],
    dispatch_kernel: Callable[..., Any],
    FP4_PER_UINT: int,
) -> dict[str, Callable[..., Any]]:
    """Build extracted attention exports using helpers provided by kernels.py."""

    def paged_attention_v1(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Dispatch paged attention v1 kernel."""
        require_mps()

        batch, num_heads_q, seq_len, head_dim = q.shape
        del seq_len
        _, num_kv_heads, block_size, _ = k_cache.shape
        max_blocks_per_seq = block_tables.shape[1]

        q_flat = q.reshape(batch, num_heads_q, head_dim).half().contiguous()
        output = torch.empty(
            (batch, num_heads_q, head_dim),
            dtype=torch.float16,
            device="mps",
        )

        lib = get_default_library()
        device = lib.device

        if not _PAGED_ATTENTION_SHADER_PATH.exists():
            raise FileNotFoundError(
                f"Shader file not found: {_PAGED_ATTENTION_SHADER_PATH}"
            )

        _ensure_kernel_compiled(
            lib,
            "paged_attention",
            _PAGED_ATTENTION_SHADER_PATH.read_text(encoding="utf-8"),
        )

        q_buf = _private_buffer_from_tensor(q_flat, lib, device, cache=False)
        k_buf = _private_buffer_from_tensor(
            k_cache.half().contiguous(), lib, device, cache=True
        )
        v_buf = _private_buffer_from_tensor(
            v_cache.half().contiguous(), lib, device, cache=True
        )

        if block_tables.dtype != torch.int32:
            block_tables = block_tables.to(torch.int32)
        if context_lens.dtype != torch.int32:
            context_lens = context_lens.to(torch.int32)

        block_tables_buf = _private_buffer_from_tensor(
            block_tables.contiguous(), lib, device, cache=False
        )
        context_lens_buf = _private_buffer_from_tensor(
            context_lens.contiguous(), lib, device, cache=False
        )
        out_buf = mps_tensor_to_metal_buffer(output, device, copy_back=True)

        num_seqs_buf = _params_buffer(lib, device, np.array([batch], dtype=np.uint32))
        num_heads_q_buf = _params_buffer(
            lib, device, np.array([num_heads_q], dtype=np.uint32)
        )
        num_kv_heads_buf = _params_buffer(
            lib, device, np.array([num_kv_heads], dtype=np.uint32)
        )
        head_dim_buf = _params_buffer(lib, device, np.array([head_dim], dtype=np.uint32))
        max_blocks_buf = _params_buffer(
            lib, device, np.array([max_blocks_per_seq], dtype=np.uint32)
        )
        scale_buf = _params_buffer(lib, device, np.array([scale], dtype=np.float32))

        dispatch_kernel(
            lib,
            function_name="paged_attention_v1",
            grid=(batch, num_heads_q, 1),
            threadgroup=(128, 1, 1),
            buffers=[
                q_buf,
                k_buf,
                v_buf,
                block_tables_buf,
                context_lens_buf,
                out_buf,
                num_seqs_buf,
                num_heads_q_buf,
                num_kv_heads_buf,
                head_dim_buf,
                max_blocks_buf,
                scale_buf,
            ],
            wait=True,
        )

        return output.reshape(batch, num_heads_q, 1, head_dim)

    def quantized_kv_attention_decode(
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scales: torch.Tensor,
        v_scales: torch.Tensor,
        num_heads_q: int | None = None,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        quant_dtype: str = "fp4",
        group_size: int = 128,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Fused decode attention over FP4/INT8 KV cache."""
        require_mps()

        if quant_dtype not in {"fp4", "int8"}:
            raise ValueError(f"quant_dtype must be 'fp4' or 'int8', got {quant_dtype!r}")
        if group_size <= 0:
            raise ValueError("group_size must be > 0")

        if query.ndim == 3 and query.shape[0] == 1:
            query = query.squeeze(0)
        if query.ndim != 2:
            raise ValueError(f"query must have shape [heads, head_dim], got {tuple(query.shape)}")

        inferred_heads_q, inferred_head_dim = query.shape
        if num_heads_q is None:
            num_heads_q = int(inferred_heads_q)
        if head_dim is None:
            head_dim = int(inferred_head_dim)
        if num_heads_q != inferred_heads_q or head_dim != inferred_head_dim:
            raise ValueError(
                "query shape does not match provided num_heads_q/head_dim: "
                f"query={tuple(query.shape)}, num_heads_q={num_heads_q}, head_dim={head_dim}"
            )

        if k_cache.ndim != 3 or v_cache.ndim != 3:
            raise ValueError(
                "k_cache and v_cache must have shape [seq_len, num_kv_heads, packed_or_head_dim]"
            )
        if k_scales.ndim != 3 or v_scales.ndim != 3:
            raise ValueError(
                "k_scales and v_scales must have shape [seq_len, num_kv_heads, num_scale_groups]"
            )

        seq_len = int(k_cache.shape[0])
        if seq_len == 0:
            return torch.zeros((num_heads_q, head_dim), dtype=torch.float16, device=query.device)

        if num_kv_heads is None:
            num_kv_heads = int(k_cache.shape[1])
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be > 0")

        if tuple(k_cache.shape[:2]) != (seq_len, num_kv_heads) or tuple(v_cache.shape[:2]) != (
            seq_len,
            num_kv_heads,
        ):
            raise ValueError("k_cache/v_cache leading dimensions must match [seq_len, num_kv_heads]")

        if tuple(k_scales.shape[:2]) != (seq_len, num_kv_heads) or tuple(v_scales.shape[:2]) != (
            seq_len,
            num_kv_heads,
        ):
            raise ValueError("k_scales/v_scales leading dimensions must match [seq_len, num_kv_heads]")

        num_scale_groups = int(k_scales.shape[2])
        if num_scale_groups <= 0:
            raise ValueError("num_scale_groups must be > 0")

        expected_cache_last = (head_dim + 1) // 2 if quant_dtype == "fp4" else head_dim
        if int(k_cache.shape[2]) != expected_cache_last or int(v_cache.shape[2]) != expected_cache_last:
            raise ValueError(
                f"Expected cache last dim {expected_cache_last} for {quant_dtype}, got "
                f"k={k_cache.shape[2]}, v={v_cache.shape[2]}"
            )

        if scale is None:
            scale = float(head_dim) ** -0.5

        q = query.to(device="mps", dtype=torch.float16).contiguous()
        k = k_cache.to(device="mps", dtype=torch.uint8).contiguous()
        v = v_cache.to(device="mps", dtype=torch.uint8).contiguous()
        ks = k_scales.to(device="mps", dtype=torch.float16).contiguous()
        vs = v_scales.to(device="mps", dtype=torch.float16).contiguous()
        out = torch.empty((num_heads_q, head_dim), dtype=torch.float16, device="mps")

        lib = get_default_library()
        device = lib.device
        _ensure_kernel_compiled(
            lib,
            "quantized_kv_attention",
            _QUANTIZED_KV_ATTN_DECODE_KERNEL,
        )

        q_buf = _private_buffer_from_tensor(q, lib, device, cache=False)
        k_buf = _private_buffer_from_tensor(k, lib, device, cache=False)
        v_buf = _private_buffer_from_tensor(v, lib, device, cache=False)
        ks_buf = _private_buffer_from_tensor(ks, lib, device, cache=False)
        vs_buf = _private_buffer_from_tensor(vs, lib, device, cache=False)
        out_buf = mps_tensor_to_metal_buffer(out, device, copy_back=True)

        seq_len_buf = _params_buffer(lib, device, np.array([seq_len], dtype=np.uint32))
        heads_q_buf = _params_buffer(lib, device, np.array([num_heads_q], dtype=np.uint32))
        heads_kv_buf = _params_buffer(lib, device, np.array([num_kv_heads], dtype=np.uint32))
        head_dim_buf = _params_buffer(lib, device, np.array([head_dim], dtype=np.uint32))
        group_size_buf = _params_buffer(lib, device, np.array([group_size], dtype=np.uint32))
        num_groups_buf = _params_buffer(
            lib, device, np.array([num_scale_groups], dtype=np.uint32)
        )
        quant_mode = 1 if quant_dtype == "fp4" else 2
        quant_mode_buf = _params_buffer(lib, device, np.array([quant_mode], dtype=np.uint32))
        scale_buf = _params_buffer(lib, device, np.array([scale], dtype=np.float32))

        dispatch_kernel(
            lib,
            function_name="quantized_kv_attention_decode",
            grid=(num_heads_q, 1, 1),
            threadgroup=(128, 1, 1),
            buffers=[
                q_buf,
                k_buf,
                v_buf,
                ks_buf,
                vs_buf,
                out_buf,
                seq_len_buf,
                heads_q_buf,
                heads_kv_buf,
                head_dim_buf,
                group_size_buf,
                num_groups_buf,
                quant_mode_buf,
                scale_buf,
            ],
            wait=True,
        )

        return out

    def paged_attention_fp4(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Paged attention fallback over FP4-interleaved block cache."""
        del value_cache

        num_seqs, num_heads, seq_len, head_dim = query.shape
        device = query.device

        _, _, block_size, num_kv_heads, _ = key_cache.shape
        max_blocks = block_tables.shape[1]
        max_context = max_blocks * block_size

        flat_indices = block_tables.reshape(-1).long()
        gathered = key_cache[flat_indices]
        gathered = gathered.view(num_seqs, max_blocks, 2, block_size, num_kv_heads, head_dim)
        gathered = gathered.permute(0, 2, 1, 3, 4, 5)
        gathered = gathered.reshape(num_seqs, 2, max_context, num_kv_heads, head_dim)

        keys = gathered[:, 0]
        values = gathered[:, 1]
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        if num_kv_heads < num_heads:
            repeat_factor = num_heads // num_kv_heads
            keys = keys.repeat_interleave(repeat_factor, dim=1)
            values = values.repeat_interleave(repeat_factor, dim=1)

        attn_weights = (query @ keys.transpose(-2, -1)) * scale

        kv_positions = torch.arange(max_context, device=device)[None, :]
        context_lens_2d = context_lens[:, None].long()
        valid_mask = kv_positions < context_lens_2d
        valid_mask = valid_mask[:, None, None, :]
        attn_weights = torch.where(
            valid_mask,
            attn_weights,
            torch.tensor(float("-inf"), device=device),
        )

        if seq_len > 1:
            q_positions = torch.arange(seq_len, device=device)[None, None, :, None]
            kv_pos_expanded = kv_positions[None, None, None, :]
            offsets = context_lens_2d[:, None, None, :] - seq_len + q_positions
            causal_mask = kv_pos_expanded <= offsets
            attn_weights = torch.where(
                causal_mask,
                attn_weights,
                torch.tensor(float("-inf"), device=device),
            )

        attn_weights = torch.softmax(attn_weights, dim=-1)
        return attn_weights @ values

    def flash_attention_kv_fp4(
        Q: torch.Tensor,
        K_packed: torch.Tensor,
        V_packed: torch.Tensor,
        K_scales: torch.Tensor,
        V_scales: torch.Tensor,
        scale: float,
        num_heads_q: int | None = None,
        num_heads_k: int | None = None,
    ) -> torch.Tensor:
        """Flash Attention with FP4-quantized KV cache."""

        def _dequantize_fp4_blockscaled(
            packed: torch.Tensor,
            scales: torch.Tensor,
            head_dim: int,
        ) -> torch.Tensor:
            packed_cpu = packed.detach().cpu()
            scales_cpu = scales.detach().cpu()
            if scales_cpu.dim() == 4 and scales_cpu.shape[-1] == 1:
                scales_cpu = scales_cpu[..., 0]

            if packed_cpu.dtype != torch.uint32:
                packed_cpu = packed_cpu.to(torch.uint32)

            batch, heads, seq, packed_dim = packed_cpu.shape
            unpacked_dim = packed_dim * FP4_PER_UINT
            if unpacked_dim < head_dim:
                raise ValueError(
                    f"Packed FP4 head_dim too small: packed_dim={packed_dim} "
                    f"(unpacked {unpacked_dim}) < head_dim={head_dim}"
                )

            fp4_table = torch.tensor(
                [
                    0.0,
                    0.25,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    -0.0,
                    -0.25,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ],
                dtype=torch.float32,
            )

            packed_i64 = packed_cpu.to(torch.int64)
            scales_expanded = scales_cpu.to(torch.float32).unsqueeze(-1)
            out = torch.empty((batch, heads, seq, unpacked_dim), dtype=torch.float32)

            for i in range(FP4_PER_UINT):
                nibbles = (packed_i64 >> (i * 4)) & 0xF
                vals = fp4_table[nibbles]
                out[..., i::FP4_PER_UINT] = vals * scales_expanded

            out = out[..., :head_dim].to(torch.float16)
            return out.to(packed.device)

        head_dim = Q.shape[-1]
        K = _dequantize_fp4_blockscaled(K_packed, K_scales, head_dim)
        V = _dequantize_fp4_blockscaled(V_packed, V_scales, head_dim)

        heads_q = num_heads_q if num_heads_q is not None else Q.shape[1]
        heads_k = num_heads_k if num_heads_k is not None else K.shape[1]
        if heads_q != heads_k:
            if heads_k <= 0 or heads_q % heads_k != 0:
                raise ValueError(
                    f"Invalid GQA head counts: num_heads_q={heads_q}, num_heads_k={heads_k}"
                )
            repeat_factor = heads_q // heads_k
            K = K.repeat_interleave(repeat_factor, dim=1)
            V = V.repeat_interleave(repeat_factor, dim=1)

        return torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale)

    return {
        "flash_attention_kv_fp4": flash_attention_kv_fp4,
        "paged_attention_fp4": paged_attention_fp4,
        "paged_attention_v1": paged_attention_v1,
        "quantized_kv_attention_decode": quantized_kv_attention_decode,
    }
