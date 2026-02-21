"""MMFP4 expert container for GLM4-MoE-Lite."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_attr(obj: Any, dotted_name: str) -> Any:
    cur = obj
    for part in dotted_name.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _first_tensor(obj: Any, names: tuple[str, ...]) -> torch.Tensor | None:
    for name in names:
        value = _resolve_attr(obj, name)
        if isinstance(value, (torch.Tensor, nn.Parameter)):
            return value.detach()
    return None


def _first_int(obj: Any, names: tuple[str, ...]) -> int | None:
    for name in names:
        value = _resolve_attr(obj, name)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return int(value)
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return int(value.item())
    return None


def _copy_with_optional_transpose(dst: torch.Tensor, src: torch.Tensor, name: str) -> None:
    src = src.detach()
    if src.shape == dst.shape:
        src_copy = src
    elif src.ndim == dst.ndim and src.transpose(-1, -2).shape == dst.shape:
        src_copy = src.transpose(-1, -2)
    else:
        raise ValueError(
            f"Shape mismatch for {name}: got {tuple(src.shape)}, expected {tuple(dst.shape)}"
        )
    dst.copy_(src_copy.to(device=dst.device, dtype=dst.dtype))


def _batch_dequant_fp4_experts(
    packed: "torch.Tensor",   # [k, out_features, in_packed] uint32
    scales: "torch.Tensor",   # [k, n_groups, out_features] fp16
    group_size: int,
) -> "torch.Tensor":          # [k, out_features, in_features] fp16
    """Dequantize FP4 weights for k experts in one vectorized GPU pass.

    Eliminates the need to loop over experts or call tolist().
    Memory: ~18 MB for k=4 experts (intermediate tensors freed immediately).
    """
    from .layers.mmfp4_linear import (_SHIFT_4BIT, _get_cached_e2m1_table,
                                      _get_cached_group_ids)

    k, out_f, in_packed = packed.shape
    in_features = in_packed * 8
    device = packed.device
    n_groups = scales.shape[1]

    # Unpack all k experts' nibbles in one shot
    shifts = _SHIFT_4BIT.to(device=device, dtype=torch.int32)  # [8]
    # [k, out_f, in_packed, 1]
    words = packed.unsqueeze(-1).to(torch.int32)
    nibbles = torch.bitwise_and(
        torch.bitwise_right_shift(words, shifts.view(1, 1, 1, 8)), 0xF
    )  # [k, out_f, in_packed, 8]
    nibbles_flat = nibbles.contiguous().view(-1).long()

    table = _get_cached_e2m1_table(device, torch.float32)
    dequant = table[nibbles_flat].view(k, out_f, in_features)

    # Scale: [k, n_groups, out_f] -> [k, out_f, n_groups] -> expand to [k, out_f, in_features]
    scales_t = scales.to(torch.float32).permute(
        0, 2, 1)          # [k, out_f, n_groups]
    group_ids = _get_cached_group_ids(
        in_features, group_size, n_groups, device)
    # [k, out_f, in_features]
    expanded = scales_t[:, :, group_ids]

    return (dequant * expanded).to(torch.float16)


class QuantizedGlm4MoEExperts(nn.Module):
    """MMFP4-quantized MoE experts for GLM4-MoE-Lite.

    Replaces Glm4MoeLiteNaiveMoe. Stores per-expert packed FP4 weights
    and dispatches to Metal kernels for expert GEMM.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        group_size: int = 128,
        device: str = "mps",
    ):
        super().__init__()

        if num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {num_experts}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {hidden_size}")
        if intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be > 0, got {intermediate_size}"
            )
        if group_size <= 0:
            raise ValueError(f"group_size must be > 0, got {group_size}")
        if hidden_size % 8 != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by 8 for MMFP4 packing"
            )
        if intermediate_size % 8 != 0:
            raise ValueError(
                f"intermediate_size ({intermediate_size}) must be divisible by 8 for MMFP4 packing"
            )
        if hidden_size % group_size != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size})"
            )
        if intermediate_size % group_size != 0:
            raise ValueError(
                f"intermediate_size ({intermediate_size}) must be divisible by group_size ({group_size})"
            )

        self.num_experts = int(num_experts)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.group_size = int(group_size)

        hidden_packed = self.hidden_size // 8
        intermediate_packed = self.intermediate_size // 8
        hidden_groups = self.hidden_size // self.group_size
        intermediate_groups = self.intermediate_size // self.group_size
        target_device = torch.device(device)

        # Register buffers for each expert's packed weights and scales.
        # Packed shape: [num_experts, out_dim, in_dim // 8] (row-packed for mmfp4_gemm)
        # Scales shape: [num_experts, n_groups, out_dim] - PRE-TRANSPOSED for kernel
        # NOTE: Scales are stored transposed to avoid creating temporary tensors
        # during forward pass, which caused data_ptr() collisions in Metal buffer cache.
        self.register_buffer(
            "gate_proj_packed",
            torch.zeros(
                (self.num_experts, self.intermediate_size, hidden_packed),
                dtype=torch.uint32,
                device=target_device,
            ),
        )
        self.register_buffer(
            "gate_proj_scales",
            torch.ones(
                (self.num_experts, hidden_groups, self.intermediate_size),
                dtype=torch.float16,
                device=target_device,
            ),
        )
        self.register_buffer(
            "up_proj_packed",
            torch.zeros(
                (self.num_experts, self.intermediate_size, hidden_packed),
                dtype=torch.uint32,
                device=target_device,
            ),
        )
        self.register_buffer(
            "up_proj_scales",
            torch.ones(
                (self.num_experts, hidden_groups, self.intermediate_size),
                dtype=torch.float16,
                device=target_device,
            ),
        )
        self.register_buffer(
            "down_proj_packed",
            torch.zeros(
                (self.num_experts, self.hidden_size, intermediate_packed),
                dtype=torch.uint32,
                device=target_device,
            ),
        )
        self.register_buffer(
            "down_proj_scales",
            torch.ones(
                (self.num_experts, intermediate_groups, self.hidden_size),
                dtype=torch.float16,
                device=target_device,
            ),
        )

        # LRU dequant cache: expert_idx → (gate_w, up_w, down_w) in FP16
        # Eliminates repeated _fast_dequant calls on the same experts during decode.
        # Memory per expert: ~18.9MB FP16. Cap at 16 (covers top_k=6 with large margin).
        self._dequant_cache: OrderedDict[int, tuple[torch.Tensor,
                                                    torch.Tensor, torch.Tensor]] = OrderedDict()
        self._dequant_cache_max: int = 16
        # Batch path tuning knobs:
        # - _dispatch_chunk_size bounds temporary padded tensors used by vectorized dispatch.
        # - _vectorized_batch_dispatch toggles the chunked batched-bmm route.
        self._dispatch_chunk_size: int = 16
        self._vectorized_batch_dispatch: bool = True

        # Pre-allocated reusable output buffer for single-token decode.
        # Avoids repeated torch.zeros allocation (one per MoE layer per step).
        # Pre-allocated here so it's a normal tensor (not inference_mode-tainted).
        self._decode_buf: torch.Tensor = torch.zeros(
            (1, self.hidden_size), dtype=torch.float16, device=target_device
        )

    def _get_expert_weights(
        self, expert_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (gate_w, up_w, down_w) FP16 for expert_idx; uses LRU cache."""
        if expert_idx in self._dequant_cache:
            self._dequant_cache.move_to_end(expert_idx)
            return self._dequant_cache[expert_idx]

        from .layers.mmfp4_linear import _fast_dequant
        gate_w = _fast_dequant(
            self.gate_proj_packed[expert_idx], self.gate_proj_scales[expert_idx], self.group_size)
        up_w = _fast_dequant(
            self.up_proj_packed[expert_idx], self.up_proj_scales[expert_idx], self.group_size)
        down_w = _fast_dequant(
            self.down_proj_packed[expert_idx], self.down_proj_scales[expert_idx], self.group_size)

        self._dequant_cache[expert_idx] = (gate_w, up_w, down_w)
        if len(self._dequant_cache) > self._dequant_cache_max:
            self._dequant_cache.popitem(last=False)
        return gate_w, up_w, down_w

    def clear_dequant_cache(self) -> None:
        """Clear the dequant weight cache (call between generations if needed)."""
        self._dequant_cache.clear()

    def _get_stacked_expert_weights(
        self,
        expert_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return stacked (gate, up, down) FP16 weights for expert_indices."""
        if expert_indices.ndim != 1:
            raise ValueError(
                f"expert_indices must be rank-1, got shape={tuple(expert_indices.shape)}"
            )
        if expert_indices.numel() == 0:
            raise ValueError("expert_indices must be non-empty")

        expert_list = expert_indices.to("cpu", dtype=torch.int64).tolist()
        missing: list[int] = []
        for expert_idx in expert_list:
            if expert_idx in self._dequant_cache:
                self._dequant_cache.move_to_end(expert_idx)
            else:
                missing.append(expert_idx)

        if missing:
            missing_t = torch.tensor(
                missing,
                dtype=torch.long,
                device=self.gate_proj_packed.device,
            )
            gate_missing = _batch_dequant_fp4_experts(
                self.gate_proj_packed.index_select(0, missing_t),
                self.gate_proj_scales.index_select(0, missing_t),
                self.group_size,
            )
            up_missing = _batch_dequant_fp4_experts(
                self.up_proj_packed.index_select(0, missing_t),
                self.up_proj_scales.index_select(0, missing_t),
                self.group_size,
            )
            down_missing = _batch_dequant_fp4_experts(
                self.down_proj_packed.index_select(0, missing_t),
                self.down_proj_scales.index_select(0, missing_t),
                self.group_size,
            )

            for pos, expert_idx in enumerate(missing):
                self._dequant_cache[expert_idx] = (
                    gate_missing[pos],
                    up_missing[pos],
                    down_missing[pos],
                )
                self._dequant_cache.move_to_end(expert_idx)
                if len(self._dequant_cache) > self._dequant_cache_max:
                    self._dequant_cache.popitem(last=False)

        gate_weights: list[torch.Tensor] = []
        up_weights: list[torch.Tensor] = []
        down_weights: list[torch.Tensor] = []
        for expert_idx in expert_list:
            gate_w, up_w, down_w = self._dequant_cache[expert_idx]
            gate_weights.append(gate_w)
            up_weights.append(up_w)
            down_weights.append(down_w)

        return (
            torch.stack(gate_weights, dim=0),
            torch.stack(up_weights, dim=0),
            torch.stack(down_weights, dim=0),
        )

    def _dispatch_batch_legacy(
        self,
        sorted_inputs: torch.Tensor,
        active_experts_t: torch.Tensor,
        expert_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Legacy per-expert dispatch path kept as a safe fallback."""
        compute_device = sorted_inputs.device
        expert_indices: list[int] = []
        expert_inputs: list[torch.Tensor] = []
        for expert_pos in range(active_experts_t.numel()):
            start = int(expert_offsets[expert_pos].item())
            end = int(expert_offsets[expert_pos + 1].item())
            expert_indices.append(int(active_experts_t[expert_pos].item()))
            expert_inputs.append(sorted_inputs.narrow(0, start, end - start))

        if compute_device.type == "mps" and len(expert_inputs) > 1:
            expert_outputs_list = self._run_expert_mlp_batched(
                expert_indices, expert_inputs)
        else:
            expert_outputs_list = [
                self._run_expert_mlp(expert_idx, expert_in)
                for expert_idx, expert_in in zip(expert_indices, expert_inputs)
            ]

        sorted_outputs = torch.empty(
            (sorted_inputs.shape[0], self.hidden_size),
            dtype=torch.float16,
            device=compute_device,
        )
        for expert_pos, expert_out in enumerate(expert_outputs_list):
            start = int(expert_offsets[expert_pos].item())
            end = int(expert_offsets[expert_pos + 1].item())
            sorted_outputs[start:end] = expert_out
        return sorted_outputs

    def _dispatch_batch_vectorized(
        self,
        sorted_inputs: torch.Tensor,
        active_experts_t: torch.Tensor,
        expert_counts: torch.Tensor,
        expert_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Chunked batched expert dispatch to reduce Python loop overhead."""
        compute_device = sorted_inputs.device
        sorted_inputs_f16 = (
            sorted_inputs if sorted_inputs.dtype == torch.float16 else sorted_inputs.to(torch.float16)
        )
        sorted_outputs = torch.empty(
            (sorted_inputs.shape[0], self.hidden_size),
            dtype=torch.float16,
            device=compute_device,
        )

        total_active = int(active_experts_t.numel())
        chunk_size = max(1, min(self._dispatch_chunk_size, total_active))
        for chunk_start in range(0, total_active, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_active)
            assign_start = int(expert_offsets[chunk_start].item())
            assign_end = int(expert_offsets[chunk_end].item())
            if assign_start == assign_end:
                continue

            chunk_inputs = sorted_inputs_f16.narrow(0, assign_start, assign_end - assign_start)
            chunk_counts = expert_counts.narrow(0, chunk_start, chunk_end - chunk_start)
            chunk_experts = active_experts_t.narrow(0, chunk_start, chunk_end - chunk_start)

            if chunk_experts.numel() == 1:
                expert_idx = int(chunk_experts[0].item())
                sorted_outputs[assign_start:assign_end] = self._run_expert_mlp(expert_idx, chunk_inputs)
                continue

            local_expert_count = int(chunk_experts.numel())
            max_tokens = int(chunk_counts.max().item())
            local_expert_ids = torch.repeat_interleave(
                torch.arange(local_expert_count, device=compute_device, dtype=torch.long),
                chunk_counts,
            )
            local_offsets = torch.zeros(
                local_expert_count + 1,
                dtype=torch.long,
                device=compute_device,
            )
            local_offsets[1:] = torch.cumsum(chunk_counts, dim=0)
            local_base = local_offsets[:-1].repeat_interleave(chunk_counts)
            local_pos = (
                torch.arange(chunk_inputs.shape[0], device=compute_device, dtype=torch.long)
                - local_base
            )

            padded_inputs = torch.zeros(
                (local_expert_count, max_tokens, self.hidden_size),
                dtype=torch.float16,
                device=compute_device,
            )
            padded_inputs[local_expert_ids, local_pos] = chunk_inputs

            gate_w, up_w, down_w = self._get_stacked_expert_weights(chunk_experts)
            gate_out = torch.bmm(padded_inputs, gate_w.transpose(1, 2))
            up_out = torch.bmm(padded_inputs, up_w.transpose(1, 2))
            activated = F.silu(gate_out) * up_out
            padded_outputs = torch.bmm(activated, down_w.transpose(1, 2))

            sorted_outputs[assign_start:assign_end] = padded_outputs[local_expert_ids, local_pos]
        return sorted_outputs

    def _mmfp4_forward(
        self,
        x: torch.Tensor,
        packed_kernel: torch.Tensor,
        scales_kernel: torch.Tensor,
    ) -> torch.Tensor:
        x_fp16 = x if x.dtype == torch.float16 else x.to(torch.float16)

        # Use mmfp4_linear.mmfp4_gemm which expects row-packed [out, in//8]
        try:
            from .layers.mmfp4_linear import mmfp4_gemm as _row_mmfp4_gemm

            return _row_mmfp4_gemm(
                x_fp16,
                packed_kernel,
                scales_kernel,
                group_size=self.group_size,
            )
        except Exception:
            pass

        # Fallback: metal_kernels.mmfp4_gemm expects [in//8, out] - transpose needed
        try:
            from .kernels import mmfp4_gemm as _col_mmfp4_gemm

            # Transpose from [out, in//8] to [in//8, out] for kernel layout
            packed_col = packed_kernel.transpose(0, 1).contiguous()
            scales_col = scales_kernel.transpose(0, 1).contiguous()
            return _col_mmfp4_gemm(
                x_fp16,
                packed_col,
                scales_col,
                group_size=self.group_size,
            )
        except Exception:
            pass

        # CPU/non-MPS fallback: dequantize then run linear.
        from .layers.mmfp4_linear import _fast_dequant

        packed_row = packed_kernel.transpose(0, 1).contiguous()
        scales_row = scales_kernel.transpose(0, 1).contiguous()
        weight = _fast_dequant(packed_row, scales_row, self.group_size)
        return F.linear(
            x_fp16.to(device=weight.device, dtype=weight.dtype),
            weight,
            None,
        )

    def _mmfp4_fused_gate_up(
        self,
        x: torch.Tensor,
        gate_packed: torch.Tensor,
        gate_scales: torch.Tensor,
        up_packed: torch.Tensor,
        up_scales: torch.Tensor,
    ) -> torch.Tensor:
        x_fp16 = x if x.dtype == torch.float16 else x.to(torch.float16)
        try:
            from .kernels import mmfp4_fused_gate_up as _kernel
            return _kernel(x_fp16, gate_packed, gate_scales, up_packed, up_scales,
                           group_size=self.group_size)
        except Exception:
            pass
        gate = self._mmfp4_forward(x_fp16, gate_packed, gate_scales)
        up = self._mmfp4_forward(x_fp16, up_packed, up_scales)
        return F.silu(gate) * up

    def _run_expert_mlp(self, expert_idx: int, expert_in: torch.Tensor) -> torch.Tensor:
        x_fp16 = expert_in if expert_in.dtype == torch.float16 else expert_in.to(
            torch.float16)
        gate_w, up_w, down_w = self._get_expert_weights(expert_idx)
        activated = F.silu(F.linear(x_fp16, gate_w)) * F.linear(x_fp16, up_w)
        return F.linear(activated, down_w)

    def _accumulate_expert_into(
        self,
        out: torch.Tensor,
        expert_idx: int,
        expert_in: torch.Tensor,
        weight: float | torch.Tensor,
    ) -> None:
        """Run expert and add weighted result directly into out[0]. No intermediate allocation."""
        gate_w, up_w, down_w = self._get_expert_weights(expert_idx)
        activated = F.silu(F.linear(expert_in, gate_w)) * \
            F.linear(expert_in, up_w)
        expert_out = F.linear(activated, down_w)[0]
        if isinstance(weight, torch.Tensor):
            out.add_(expert_out * weight.to(device=out.device, dtype=out.dtype))
            return
        out.add_(expert_out, alpha=weight)

    def _run_expert_mlp_batched(
        self,
        expert_indices: list[int],
        expert_inputs: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if len(expert_indices) != len(expert_inputs):
            raise ValueError(
                f"expert_indices ({len(expert_indices)}) and expert_inputs ({len(expert_inputs)}) "
                "must have matching lengths"
            )
        if not expert_indices:
            return []

        # All paths use _run_expert_mlp which reads from the LRU cache.
        # Sequential F.linear is faster than torch.stack+bmm (avoids 113MB/layer copy).
        return [
            self._run_expert_mlp(expert_idx, expert_in)
            for expert_idx, expert_in in zip(expert_indices, expert_inputs)
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Match Glm4MoeLiteNaiveMoe.forward() signature exactly."""
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, got {hidden_states.shape[-1]}"
            )
        if top_k_index.shape != top_k_weights.shape:
            raise ValueError(
                "top_k_index and top_k_weights must have identical shapes, got "
                f"{tuple(top_k_index.shape)} vs {tuple(top_k_weights.shape)}"
            )
        if top_k_index.ndim < 2:
            raise ValueError(
                f"Expected top_k_index to be at least rank-2, got shape={tuple(top_k_index.shape)}"
            )

        input_dtype = hidden_states.dtype
        input_device = hidden_states.device
        compute_device = self.gate_proj_packed.device

        hidden_2d = hidden_states.reshape(-1, self.hidden_size)
        if hidden_2d.shape[0] == 0:
            return hidden_states.new_zeros(hidden_states.shape)
        if hidden_2d.device != compute_device:
            hidden_2d = hidden_2d.to(compute_device)

        top_k = int(top_k_index.shape[-1])
        routed_experts = top_k_index.reshape(-1, top_k).to(
            device=compute_device, dtype=torch.long
        )
        routed_weights = top_k_weights.reshape(-1, top_k).to(
            device=compute_device, dtype=torch.float16
        )
        if routed_experts.shape[0] != hidden_2d.shape[0]:
            raise ValueError(
                "Routing tensor/token count mismatch: "
                f"hidden tokens={hidden_2d.shape[0]}, routing tokens={routed_experts.shape[0]}"
            )

        # Fast path for single-token decode (M=1).
        # Uses LRU-cached float16 weights (one-time dequant per expert) + in-place accumulation.
        if hidden_2d.shape[0] == 1:
            x_f16 = hidden_2d if hidden_2d.dtype == torch.float16 else hidden_2d.to(
                torch.float16)
            expert_ids_t = routed_experts[0]
            weights_t = routed_weights[0]

            # Reuse pre-allocated decode buffer to avoid torch.zeros per layer.
            if self._decode_buf.device != compute_device:
                with torch.inference_mode(False):
                    self._decode_buf = torch.zeros(
                        (1, self.hidden_size), dtype=torch.float16, device=compute_device
                    )
            self._decode_buf.zero_()

            for slot in range(top_k):
                expert_idx = int(expert_ids_t[slot].item())
                self._accumulate_expert_into(
                    self._decode_buf[0],
                    expert_idx,
                    x_f16,
                    weights_t[slot],
                )

            out = self._decode_buf.to(dtype=input_dtype)
            if out.device != input_device:
                out = out.to(input_device)
            return out.reshape(*hidden_states.shape[:-1], self.hidden_size)

        # Batch path (prefill / multi-token): allocate output and sort by expert.
        output = torch.zeros(
            (hidden_2d.shape[0], self.hidden_size),
            dtype=torch.float16,
            device=compute_device,
        )

        # Flatten assignments then sort once so each expert's tokens are contiguous.
        flat_experts = routed_experts.reshape(-1)
        flat_weights = routed_weights.reshape(-1)
        token_base = torch.arange(
            hidden_2d.shape[0],
            device=compute_device,
            dtype=torch.long,
        )
        flat_token_idx = token_base.unsqueeze(1).expand(-1, top_k).reshape(-1)

        sorted_order = torch.argsort(flat_experts)
        sorted_experts = flat_experts.index_select(0, sorted_order)
        sorted_weights = flat_weights.index_select(0, sorted_order)
        sorted_token_idx = flat_token_idx.index_select(0, sorted_order)
        sorted_inputs = hidden_2d.index_select(0, sorted_token_idx)

        active_experts_t, expert_counts = torch.unique_consecutive(
            sorted_experts,
            return_counts=True,
        )
        if active_experts_t.numel() > 0:
            bad_mask = (active_experts_t < 0) | (active_experts_t >= self.num_experts)
            if bool(bad_mask.any().item()):
                bad_idx = int(active_experts_t[bad_mask][0].item())
                raise IndexError(
                    f"Expert index {bad_idx} is out of range [0, {self.num_experts})"
                )

        expert_offsets = torch.zeros(
            active_experts_t.numel() + 1,
            dtype=torch.long,
            device=compute_device,
        )
        expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

        if self._vectorized_batch_dispatch and active_experts_t.numel() > 0:
            try:
                sorted_outputs = self._dispatch_batch_vectorized(
                    sorted_inputs,
                    active_experts_t,
                    expert_counts,
                    expert_offsets,
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                sorted_outputs = self._dispatch_batch_legacy(
                    sorted_inputs,
                    active_experts_t,
                    expert_offsets,
                )
        else:
            sorted_outputs = self._dispatch_batch_legacy(
                sorted_inputs,
                active_experts_t,
                expert_offsets,
            )

        output.index_add_(0, sorted_token_idx,
                          sorted_outputs * sorted_weights.unsqueeze(-1))

        output = output.to(dtype=input_dtype)
        if output.device != input_device:
            output = output.to(input_device)
        return output.reshape(*hidden_states.shape[:-1], self.hidden_size)

    @classmethod
    def from_naive_moe(
        cls,
        naive_moe: nn.Module,
        group_size: int = 128,
        device: str = "mps",
    ) -> QuantizedGlm4MoEExperts:
        """Create quantized experts from a Glm4MoeLiteNaiveMoe-like module.

        This infers model dimensions from the source module and performs a
        best-effort copy of already-quantized expert tensors when present.
        """
        gate_packed = _first_tensor(
            naive_moe,
            (
                "gate_proj_packed",
                "experts.gate_proj_packed",
                "gate_proj_packed_weights",
                "experts.gate_proj_packed_weights",
                "experts.gate_proj.packed_weights",
            ),
        )
        gate_scales = _first_tensor(
            naive_moe,
            (
                "gate_proj_scales",
                "experts.gate_proj_scales",
                "experts.gate_proj.scales",
            ),
        )
        up_packed = _first_tensor(
            naive_moe,
            (
                "up_proj_packed",
                "experts.up_proj_packed",
                "up_proj_packed_weights",
                "experts.up_proj_packed_weights",
                "experts.up_proj.packed_weights",
            ),
        )
        up_scales = _first_tensor(
            naive_moe,
            (
                "up_proj_scales",
                "experts.up_proj_scales",
                "experts.up_proj.scales",
            ),
        )
        down_packed = _first_tensor(
            naive_moe,
            (
                "down_proj_packed",
                "experts.down_proj_packed",
                "down_proj_packed_weights",
                "experts.down_proj_packed_weights",
                "experts.down_proj.packed_weights",
            ),
        )
        down_scales = _first_tensor(
            naive_moe,
            (
                "down_proj_scales",
                "experts.down_proj_scales",
                "experts.down_proj.scales",
            ),
        )

        num_experts = _first_int(
            naive_moe,
            ("num_experts", "n_routed_experts", "num_local_experts"),
        )
        hidden_size = _first_int(
            naive_moe, ("hidden_size", "hidden_dim", "model_dim"))
        intermediate_size = _first_int(
            naive_moe,
            ("intermediate_size", "moe_intermediate_size", "ffn_hidden_size"),
        )

        if gate_packed is not None and gate_packed.ndim == 3:
            num_experts = int(
                gate_packed.shape[0]) if num_experts is None else num_experts
            intermediate_size = (
                int(gate_packed.shape[1]
                    ) if intermediate_size is None else intermediate_size
            )
            hidden_size = int(
                gate_packed.shape[2] * 8) if hidden_size is None else hidden_size
        if down_packed is not None and down_packed.ndim == 3:
            num_experts = int(
                down_packed.shape[0]) if num_experts is None else num_experts
            hidden_size = int(
                down_packed.shape[1]) if hidden_size is None else hidden_size
            intermediate_size = (
                int(down_packed.shape[2] * 8)
                if intermediate_size is None
                else intermediate_size
            )

        float_gate_weight = _first_tensor(
            naive_moe,
            (
                "experts.gate_proj.weight",
                "experts.gate_proj",
                "experts.up_proj.weight",
                "experts.up_proj",
                "experts.down_proj.weight",
                "experts.down_proj",
            ),
        )
        if float_gate_weight is not None and float_gate_weight.ndim == 3:
            num_experts = int(
                float_gate_weight.shape[0]) if num_experts is None else num_experts
            if hidden_size is None and float_gate_weight.shape[-1] % 8 == 0:
                hidden_size = int(float_gate_weight.shape[-1])
            if intermediate_size is None:
                intermediate_size = int(float_gate_weight.shape[1])

        if num_experts is None or hidden_size is None or intermediate_size is None:
            raise ValueError(
                "Unable to infer num_experts/hidden_size/intermediate_size from naive_moe. "
                "Expose these attrs or provide quantized expert tensors."
            )

        module = cls(
            num_experts=int(num_experts),
            hidden_size=int(hidden_size),
            intermediate_size=int(intermediate_size),
            group_size=group_size,
            device=device,
        )

        if gate_packed is not None:
            _copy_with_optional_transpose(
                module.gate_proj_packed, gate_packed, "gate_proj_packed"
            )
        if gate_scales is not None:
            _copy_with_optional_transpose(
                module.gate_proj_scales, gate_scales, "gate_proj_scales"
            )
        if up_packed is not None:
            _copy_with_optional_transpose(
                module.up_proj_packed, up_packed, "up_proj_packed"
            )
        if up_scales is not None:
            _copy_with_optional_transpose(
                module.up_proj_scales, up_scales, "up_proj_scales"
            )
        if down_packed is not None:
            _copy_with_optional_transpose(
                module.down_proj_packed, down_packed, "down_proj_packed"
            )
        if down_scales is not None:
            _copy_with_optional_transpose(
                module.down_proj_scales, down_scales, "down_proj_scales"
            )

        return module


__all__ = ["QuantizedGlm4MoEExperts"]
