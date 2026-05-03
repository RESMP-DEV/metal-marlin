"""MoE dispatch kernels extracted from the legacy kernels monolith."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def build_moe_exports(
    *,
    np: Any,
    torch: Any,
    require_mps: Callable[..., Any],
    group_tokens_by_expert_full: Callable[..., Any],
    gather_for_experts: Callable[..., Any],
    scatter_expert_outputs: Callable[..., Any],
    get_default_library: Callable[..., Any],
    _ensure_kernel_compiled: Callable[..., Any],
    _private_buffer_from_tensor: Callable[..., Any],
    _params_buffer: Callable[..., Any],
    mps_tensor_to_metal_buffer: Callable[..., Any],
    dispatch_kernel: Callable[..., Any],
    get_shader_source: Callable[..., str],
    marlin_gemm_fp4: Callable[..., Any],
    dequant_fp4: Callable[..., Any],
    FP4_PER_UINT: int,
) -> dict[str, Callable[..., Any]]:
    """Build extracted MoE exports using helpers provided by the kernel facades."""

    logger.info("build_moe_exports starting")
    def _flatten_moe_hidden(
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int, tuple[int, ...]]:
        logger.debug("_flatten_moe_hidden called with hidden_states=%s", hidden_states)
        if hidden_states.dim() < 2:
            raise ValueError("hidden_states must be at least 2D [tokens, hidden]")
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]
        num_tokens = 1
        for dim in orig_shape[:-1]:
            num_tokens *= dim
        hidden_2d = hidden_states.reshape(num_tokens, hidden_dim)
        if hidden_2d.dtype != torch.float16:
            hidden_2d = hidden_2d.half()
        return hidden_2d.contiguous(), num_tokens, hidden_dim, orig_shape

    def moe_shared_expert_fused(
        hidden_states: torch.Tensor,
        shared_expert_w: torch.Tensor,
        routed_expert_w: torch.Tensor,
        router_probs: torch.Tensor,
        router_indices: torch.Tensor,
        group_size: int = 128,
    ) -> torch.Tensor:
        """Fused shared+routed expert aggregation with FP16 weights."""
        logger.debug("moe_shared_expert_fused called with hidden_states=%s, shared_expert_w=%s, routed_expert_w=%s", hidden_states, shared_expert_w, routed_expert_w)
        require_mps()

        hidden_2d, num_tokens, hidden_dim, orig_shape = _flatten_moe_hidden(
            hidden_states
        )

        if shared_expert_w.dim() != 2:
            raise ValueError("shared_expert_w must be [hidden, intermediate]")
        if shared_expert_w.shape[0] != hidden_dim:
            raise ValueError("shared_expert_w hidden dim mismatch")

        intermediate_dim = int(shared_expert_w.shape[1])

        if routed_expert_w.dim() != 3:
            raise ValueError(
                "routed_expert_w must be [num_experts, hidden, intermediate]"
            )
        if (
            routed_expert_w.shape[1] != hidden_dim
            or routed_expert_w.shape[2] != intermediate_dim
        ):
            raise ValueError("routed_expert_w shape mismatch")

        if router_probs.dim() != 2 or router_probs.shape[0] != num_tokens:
            raise ValueError("router_probs must be [tokens, top_k]")
        if router_indices.dim() != 2 or router_indices.shape[0] != num_tokens:
            raise ValueError("router_indices must be [tokens, top_k]")

        top_k = int(router_probs.shape[1])
        if router_indices.shape[1] != top_k:
            raise ValueError("router_probs and router_indices must have same top_k")

        num_experts = int(routed_expert_w.shape[0])

        shared_w = (
            shared_expert_w.half().contiguous()
            if shared_expert_w.dtype != torch.float16
            else shared_expert_w.contiguous()
        )
        routed_w = (
            routed_expert_w.half().contiguous()
            if routed_expert_w.dtype != torch.float16
            else routed_expert_w.contiguous()
        )
        probs = (
            router_probs.half().contiguous()
            if router_probs.dtype != torch.float16
            else router_probs.contiguous()
        )
        if router_indices.dtype not in (torch.int32, torch.uint32):
            indices = router_indices.to(torch.int32).contiguous()
        else:
            indices = router_indices.contiguous()

        out = torch.empty(
            (num_tokens, intermediate_dim), dtype=torch.float16, device="mps"
        )

        lib = get_default_library()
        device = lib.device

        a_buf = _private_buffer_from_tensor(hidden_2d, lib, device, cache=False)
        shared_buf = _private_buffer_from_tensor(shared_w, lib, device, cache=True)
        routed_buf = _private_buffer_from_tensor(routed_w, lib, device, cache=True)
        probs_buf = _private_buffer_from_tensor(probs, lib, device, cache=False)
        indices_buf = _private_buffer_from_tensor(indices, lib, device, cache=False)
        out_buf = mps_tensor_to_metal_buffer(out, device, copy_back=True)

        num_tokens_buf = _params_buffer(
            lib, device, np.array([num_tokens], dtype=np.uint32)
        )
        hidden_buf = _params_buffer(lib, device, np.array([hidden_dim], dtype=np.uint32))
        intermediate_buf = _params_buffer(
            lib, device, np.array([intermediate_dim], dtype=np.uint32)
        )
        topk_buf = _params_buffer(lib, device, np.array([top_k], dtype=np.uint32))
        num_experts_buf = _params_buffer(
            lib, device, np.array([num_experts], dtype=np.uint32)
        )
        group_buf = _params_buffer(lib, device, np.array([group_size], dtype=np.uint32))

        tile_m = 64
        tile_n = 64
        threads_per_tg = 128
        grid_x = (intermediate_dim + tile_n - 1) // tile_n
        grid_y = (num_tokens + tile_m - 1) // tile_m

        dispatch_kernel(
            lib,
            function_name="moe_shared_expert_fused",
            grid=(grid_x, grid_y, 1),
            threadgroup=(threads_per_tg, 1, 1),
            buffers=[
                a_buf,
                shared_buf,
                routed_buf,
                probs_buf,
                indices_buf,
                out_buf,
                num_tokens_buf,
                hidden_buf,
                intermediate_buf,
                topk_buf,
                num_experts_buf,
                group_buf,
            ],
            wait=True,
        )

        return out.reshape(*orig_shape[:-1], intermediate_dim)

    def moe_shared_expert_fused_fp4(
        hidden_states: torch.Tensor,
        shared_expert_packed: torch.Tensor,
        shared_expert_scales: torch.Tensor,
        routed_expert_packed: torch.Tensor,
        routed_expert_scales: torch.Tensor,
        router_probs: torch.Tensor,
        router_indices: torch.Tensor,
        group_size: int = 128,
    ) -> torch.Tensor:
        """Fused shared+routed expert aggregation with FP4 weights."""
        logger.debug("moe_shared_expert_fused_fp4 called with hidden_states=%s, shared_expert_packed=%s, shared_expert_scales=%s", hidden_states, shared_expert_packed, shared_expert_scales)
        require_mps()

        hidden_2d, num_tokens, hidden_dim, orig_shape = _flatten_moe_hidden(
            hidden_states
        )

        if hidden_dim % FP4_PER_UINT != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by {FP4_PER_UINT}"
            )
        if hidden_dim % group_size != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by group_size ({group_size})"
            )

        if shared_expert_packed.dim() != 2:
            raise ValueError("shared_expert_packed must be [hidden/8, intermediate]")
        packed_k = int(shared_expert_packed.shape[0])
        if packed_k * FP4_PER_UINT != hidden_dim:
            raise ValueError("shared_expert_packed hidden dim mismatch")

        intermediate_dim = int(shared_expert_packed.shape[1])
        scale_rows = hidden_dim // group_size
        if shared_expert_scales.shape != (scale_rows, intermediate_dim):
            raise ValueError("shared_expert_scales shape mismatch")

        if routed_expert_packed.dim() != 3:
            raise ValueError(
                "routed_expert_packed must be [num_experts, hidden/8, intermediate]"
            )
        if (
            routed_expert_packed.shape[1] != packed_k
            or routed_expert_packed.shape[2] != intermediate_dim
        ):
            raise ValueError("routed_expert_packed shape mismatch")

        if routed_expert_scales.dim() != 3:
            raise ValueError(
                "routed_expert_scales must be [num_experts, hidden/group, intermediate]"
            )
        if (
            routed_expert_scales.shape[1] != scale_rows
            or routed_expert_scales.shape[2] != intermediate_dim
        ):
            raise ValueError("routed_expert_scales shape mismatch")

        if router_probs.dim() != 2 or router_probs.shape[0] != num_tokens:
            raise ValueError("router_probs must be [tokens, top_k]")
        if router_indices.dim() != 2 or router_indices.shape[0] != num_tokens:
            raise ValueError("router_indices must be [tokens, top_k]")

        top_k = int(router_probs.shape[1])
        if router_indices.shape[1] != top_k:
            raise ValueError("router_probs and router_indices must have same top_k")

        num_experts = int(routed_expert_packed.shape[0])

        shared_packed = shared_expert_packed.contiguous()
        shared_scales = (
            shared_expert_scales.half().contiguous()
            if shared_expert_scales.dtype != torch.float16
            else shared_expert_scales.contiguous()
        )
        routed_packed = routed_expert_packed.contiguous()
        routed_scales = (
            routed_expert_scales.half().contiguous()
            if routed_expert_scales.dtype != torch.float16
            else routed_expert_scales.contiguous()
        )
        probs = (
            router_probs.half().contiguous()
            if router_probs.dtype != torch.float16
            else router_probs.contiguous()
        )
        if router_indices.dtype not in (torch.int32, torch.uint32):
            indices = router_indices.to(torch.int32).contiguous()
        else:
            indices = router_indices.contiguous()

        out = torch.empty(
            (num_tokens, intermediate_dim), dtype=torch.float16, device="mps"
        )

        lib = get_default_library()
        device = lib.device

        a_buf = _private_buffer_from_tensor(hidden_2d, lib, device, cache=False)
        shared_packed_buf = _private_buffer_from_tensor(
            shared_packed, lib, device, cache=True
        )
        shared_scales_buf = _private_buffer_from_tensor(
            shared_scales, lib, device, cache=True
        )
        routed_packed_buf = _private_buffer_from_tensor(
            routed_packed, lib, device, cache=True
        )
        routed_scales_buf = _private_buffer_from_tensor(
            routed_scales, lib, device, cache=True
        )
        probs_buf = _private_buffer_from_tensor(probs, lib, device, cache=False)
        indices_buf = _private_buffer_from_tensor(indices, lib, device, cache=False)
        out_buf = mps_tensor_to_metal_buffer(out, device, copy_back=True)

        num_tokens_buf = _params_buffer(
            lib, device, np.array([num_tokens], dtype=np.uint32)
        )
        hidden_buf = _params_buffer(lib, device, np.array([hidden_dim], dtype=np.uint32))
        intermediate_buf = _params_buffer(
            lib, device, np.array([intermediate_dim], dtype=np.uint32)
        )
        topk_buf = _params_buffer(lib, device, np.array([top_k], dtype=np.uint32))
        num_experts_buf = _params_buffer(
            lib, device, np.array([num_experts], dtype=np.uint32)
        )
        group_buf = _params_buffer(lib, device, np.array([group_size], dtype=np.uint32))

        tile_m = 64
        tile_n = 64
        threads_per_tg = 128
        grid_x = (intermediate_dim + tile_n - 1) // tile_n
        grid_y = (num_tokens + tile_m - 1) // tile_m

        dispatch_kernel(
            lib,
            function_name="moe_shared_expert_fused_fp4",
            grid=(grid_x, grid_y, 1),
            threadgroup=(threads_per_tg, 1, 1),
            buffers=[
                a_buf,
                shared_packed_buf,
                shared_scales_buf,
                routed_packed_buf,
                routed_scales_buf,
                probs_buf,
                indices_buf,
                out_buf,
                num_tokens_buf,
                hidden_buf,
                intermediate_buf,
                topk_buf,
                num_experts_buf,
                group_buf,
            ],
            wait=True,
        )

        return out.reshape(*orig_shape[:-1], intermediate_dim)

    def moe_shared_expert_scatter(
        hidden_states: torch.Tensor,
        shared_expert_w: torch.Tensor,
        routed_expert_w: torch.Tensor,
        router_probs: torch.Tensor,
        router_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter-style shared expert aggregation for small token counts."""
        logger.debug("moe_shared_expert_scatter called with hidden_states=%s, shared_expert_w=%s, routed_expert_w=%s", hidden_states, shared_expert_w, routed_expert_w)
        require_mps()

        hidden_2d, num_tokens, hidden_dim, orig_shape = _flatten_moe_hidden(
            hidden_states
        )

        if shared_expert_w.dim() != 2:
            raise ValueError("shared_expert_w must be [hidden, intermediate]")
        if shared_expert_w.shape[0] != hidden_dim:
            raise ValueError("shared_expert_w hidden dim mismatch")

        intermediate_dim = int(shared_expert_w.shape[1])

        if routed_expert_w.dim() != 3:
            raise ValueError(
                "routed_expert_w must be [num_experts, hidden, intermediate]"
            )
        if (
            routed_expert_w.shape[1] != hidden_dim
            or routed_expert_w.shape[2] != intermediate_dim
        ):
            raise ValueError("routed_expert_w shape mismatch")

        if router_probs.dim() != 2 or router_probs.shape[0] != num_tokens:
            raise ValueError("router_probs must be [tokens, top_k]")
        if router_indices.dim() != 2 or router_indices.shape[0] != num_tokens:
            raise ValueError("router_indices must be [tokens, top_k]")

        top_k = int(router_probs.shape[1])
        if router_indices.shape[1] != top_k:
            raise ValueError("router_probs and router_indices must have same top_k")

        num_experts = int(routed_expert_w.shape[0])

        shared_w = (
            shared_expert_w.half().contiguous()
            if shared_expert_w.dtype != torch.float16
            else shared_expert_w.contiguous()
        )
        routed_w = (
            routed_expert_w.half().contiguous()
            if routed_expert_w.dtype != torch.float16
            else routed_expert_w.contiguous()
        )
        probs = (
            router_probs.half().contiguous()
            if router_probs.dtype != torch.float16
            else router_probs.contiguous()
        )
        if router_indices.dtype not in (torch.int32, torch.uint32):
            indices = router_indices.to(torch.int32).contiguous()
        else:
            indices = router_indices.contiguous()

        out = torch.empty(
            (num_tokens, intermediate_dim), dtype=torch.float16, device="mps"
        )

        lib = get_default_library()
        device = lib.device

        a_buf = _private_buffer_from_tensor(hidden_2d, lib, device, cache=False)
        shared_buf = _private_buffer_from_tensor(shared_w, lib, device, cache=True)
        routed_buf = _private_buffer_from_tensor(routed_w, lib, device, cache=True)
        probs_buf = _private_buffer_from_tensor(probs, lib, device, cache=False)
        indices_buf = _private_buffer_from_tensor(indices, lib, device, cache=False)
        out_buf = mps_tensor_to_metal_buffer(out, device, copy_back=True)

        num_tokens_buf = _params_buffer(
            lib, device, np.array([num_tokens], dtype=np.uint32)
        )
        hidden_buf = _params_buffer(lib, device, np.array([hidden_dim], dtype=np.uint32))
        intermediate_buf = _params_buffer(
            lib, device, np.array([intermediate_dim], dtype=np.uint32)
        )
        topk_buf = _params_buffer(lib, device, np.array([top_k], dtype=np.uint32))
        num_experts_buf = _params_buffer(
            lib, device, np.array([num_experts], dtype=np.uint32)
        )

        dispatch_kernel(
            lib,
            function_name="moe_shared_expert_scatter",
            grid=(num_tokens, 1, 1),
            threadgroup=(128, 1, 1),
            buffers=[
                a_buf,
                shared_buf,
                routed_buf,
                probs_buf,
                indices_buf,
                out_buf,
                num_tokens_buf,
                hidden_buf,
                intermediate_buf,
                topk_buf,
                num_experts_buf,
            ],
            wait=True,
        )

        return out.reshape(*orig_shape[:-1], intermediate_dim)

    def moe_shared_expert_fp4(
        hidden_states: torch.Tensor,
        gate_up_packed: torch.Tensor,
        gate_up_scales: torch.Tensor,
        down_packed: torch.Tensor,
        down_scales: torch.Tensor,
        group_size: int = 128,
        shared_prob: float = 1.0,
    ) -> torch.Tensor:
        """Shared expert forward pass with FP4 quantized weights."""
        logger.debug("moe_shared_expert_fp4 called with hidden_states=%s, gate_up_packed=%s, gate_up_scales=%s", hidden_states, gate_up_packed, gate_up_scales)
        require_mps()

        hidden_2d, num_tokens, hidden_dim, orig_shape = _flatten_moe_hidden(
            hidden_states
        )

        if hidden_dim % FP4_PER_UINT != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by {FP4_PER_UINT}"
            )
        if hidden_dim % group_size != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by group_size ({group_size})"
            )

        if gate_up_packed.dim() != 2:
            raise ValueError("gate_up_packed must be [hidden/8, 2*intermediate]")
        if gate_up_packed.shape[0] * FP4_PER_UINT != hidden_dim:
            raise ValueError("gate_up_packed hidden dim mismatch")

        gate_up_out = int(gate_up_packed.shape[1])
        if gate_up_out % 2 != 0:
            raise ValueError("gate_up_packed output dim must be even (gate+up)")
        intermediate = gate_up_out // 2

        if intermediate % group_size != 0:
            raise ValueError(
                f"intermediate ({intermediate}) must be divisible by group_size ({group_size})"
            )

        scale_rows = hidden_dim // group_size
        if gate_up_scales.shape != (scale_rows, gate_up_out):
            raise ValueError("gate_up_scales shape mismatch")

        if down_packed.dim() != 2:
            raise ValueError("down_packed must be [intermediate/8, hidden]")
        if down_packed.shape[0] * FP4_PER_UINT != intermediate:
            raise ValueError("down_packed intermediate dim mismatch")
        if down_packed.shape[1] != hidden_dim:
            raise ValueError("down_packed output dim mismatch")

        down_scale_rows = intermediate // group_size
        if down_scales.shape != (down_scale_rows, hidden_dim):
            raise ValueError("down_scales shape mismatch")

        lib = get_default_library()
        device = lib.device

        def _dispatch_shared_gemm(
            activations: torch.Tensor,
            weights: torch.Tensor,
            scales: torch.Tensor,
            *,
            k_dim: int,
            out_dim: int,
            prob: float,
        ) -> torch.Tensor:
            logger.debug("_dispatch_shared_gemm called with activations=%s, weights=%s, scales=%s", activations, weights, scales)
            out = torch.zeros(
                (num_tokens, out_dim), dtype=torch.float16, device="mps"
            )

            a_buf = _private_buffer_from_tensor(activations, lib, device, cache=False)
            w_buf = _private_buffer_from_tensor(weights, lib, device, cache=True)
            s_buf = _private_buffer_from_tensor(scales, lib, device, cache=True)
            out_buf = mps_tensor_to_metal_buffer(out, device, copy_back=True)

            batch_buf = _params_buffer(
                lib, device, np.array([num_tokens], dtype=np.uint32)
            )
            hidden_buf = _params_buffer(lib, device, np.array([k_dim], dtype=np.uint32))
            out_buf_param = _params_buffer(
                lib, device, np.array([out_dim], dtype=np.uint32)
            )
            group_buf = _params_buffer(
                lib, device, np.array([group_size], dtype=np.uint32)
            )
            prob_buf = _params_buffer(lib, device, np.array([prob], dtype=np.float16))

            tile_m = 64
            tile_n = 64
            threads_per_tg = 128
            grid_x = (out_dim + tile_n - 1) // tile_n
            grid_y = (num_tokens + tile_m - 1) // tile_m

            dispatch_kernel(
                lib,
                function_name="moe_expert_gemm_shared_fp4",
                grid=(grid_x, grid_y, 1),
                threadgroup=(threads_per_tg, 1, 1),
                buffers=[
                    a_buf,
                    w_buf,
                    s_buf,
                    out_buf,
                    batch_buf,
                    hidden_buf,
                    out_buf_param,
                    group_buf,
                    prob_buf,
                ],
                wait=True,
            )

            return out

        gate_up = _dispatch_shared_gemm(
            hidden_2d,
            gate_up_packed.contiguous(),
            gate_up_scales.half().contiguous(),
            k_dim=hidden_dim,
            out_dim=gate_up_out,
            prob=1.0,
        )
        gate = gate_up[:, :intermediate]
        up = gate_up[:, intermediate:]
        act = torch.nn.functional.silu(gate) * up

        output = _dispatch_shared_gemm(
            act,
            down_packed.contiguous(),
            down_scales.half().contiguous(),
            k_dim=intermediate,
            out_dim=hidden_dim,
            prob=float(shared_prob),
        )

        return output.reshape(*orig_shape[:-1], hidden_dim)

    def moe_expert_gemm_fp4(
        activations: torch.Tensor,
        expert_weights: torch.Tensor,
        scales: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_probs: torch.Tensor,
        group_size: int = 128,
    ) -> torch.Tensor:
        """MoE expert GEMM with FP4-quantized expert weights."""
        logger.debug("moe_expert_gemm_fp4 called with activations=%s, expert_weights=%s, scales=%s", activations, expert_weights, scales)
        require_mps()

        orig_dtype = activations.dtype
        batch_size = activations.shape[0]
        hidden_dim = activations.shape[1]
        num_experts = expert_weights.shape[0]
        out_dim = expert_weights.shape[-1]

        dispatch_info = group_tokens_by_expert_full(expert_ids, num_experts)
        gathered = gather_for_experts(activations, dispatch_info)

        expert_probs_sorted = expert_probs[
            dispatch_info.sorted_token_indices,
            dispatch_info.sorted_expert_indices,
        ]

        try:
            lib = get_default_library()
            _ensure_kernel_compiled(
                lib,
                "moe_expert_gemm",
                get_shader_source("moe_expert_gemm"),
            )

            device = lib.device

            act_contig = gathered.half().contiguous()
            weights_contig = expert_weights.contiguous()
            scales_contig = scales.half().contiguous()
            sorted_token_ids = dispatch_info.sorted_token_indices.int().contiguous()
            expert_offsets = dispatch_info.expert_offsets.int().contiguous()
            probs_sorted = expert_probs_sorted.half().contiguous()

            output = torch.zeros(
                batch_size,
                out_dim,
                dtype=torch.float16,
                device=activations.device,
            )

            act_buf = mps_tensor_to_metal_buffer(act_contig, device)
            weights_buf = mps_tensor_to_metal_buffer(weights_contig, device)
            scales_buf = mps_tensor_to_metal_buffer(scales_contig, device)
            sorted_token_buf = mps_tensor_to_metal_buffer(sorted_token_ids, device)
            offsets_buf = mps_tensor_to_metal_buffer(expert_offsets, device)
            probs_buf = mps_tensor_to_metal_buffer(probs_sorted, device)
            output_buf = mps_tensor_to_metal_buffer(output, device, copy_back=True)

            params = np.array(
                [
                    batch_size,
                    hidden_dim,
                    out_dim,
                    num_experts,
                    expert_ids.shape[1],
                    group_size,
                    0,
                    0,
                ],
                dtype=np.uint32,
            )
            params_buf = _params_buffer(lib, device, params)

            tile_n = 64
            grid_x = (out_dim + tile_n - 1) // tile_n
            grid_y = num_experts

            dispatch_kernel(
                lib,
                function_name="moe_expert_gemm_fp4_grouped",
                grid=(grid_x, grid_y, 1),
                threadgroup=(128, 1, 1),
                buffers=[
                    act_buf,
                    weights_buf,
                    scales_buf,
                    sorted_token_buf,
                    offsets_buf,
                    probs_buf,
                    output_buf,
                    params_buf,
                ],
                wait=True,
            )

            if output.dtype != orig_dtype:
                output = output.to(orig_dtype)
            return output

        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "MoE Metal dispatch failed, using fallback: %s",
                exc,
            )

            expert_outputs = torch.empty(
                (dispatch_info.total_assignments, out_dim),
                dtype=torch.float16,
                device=activations.device,
            )

            for expert_idx in range(num_experts):
                start = int(dispatch_info.expert_offsets[expert_idx].item())
                end = int(dispatch_info.expert_offsets[expert_idx + 1].item())
                if start == end:
                    continue
                try:
                    expert_outputs[start:end] = marlin_gemm_fp4(
                        gathered[start:end],
                        expert_weights[expert_idx],
                        scales[expert_idx],
                        group_size,
                    )
                except Exception:
                    k_dim = gathered.shape[1]
                    n_dim = expert_weights.shape[-1]
                    dequant = dequant_fp4(
                        expert_weights[expert_idx],
                        scales[expert_idx],
                        k_dim,
                        n_dim,
                        group_size,
                    )
                    expert_outputs[start:end] = gathered[start:end] @ dequant

            result = scatter_expert_outputs(
                expert_outputs,
                expert_probs,
                dispatch_info,
            )
            if result.dtype != orig_dtype:
                result = result.to(orig_dtype)
            return result

    def moe_router_topk(
        hidden: torch.Tensor,
        router_weights: torch.Tensor,
        top_k: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MoE router with top-k expert selection."""
        logger.debug("moe_router_topk called with hidden=%s, router_weights=%s, top_k=%s", hidden, router_weights, top_k)
        logits = torch.matmul(hidden, router_weights)
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_ids = torch.topk(probs, k=top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        return topk_ids, topk_probs

    def moe_fused_dispatch_shared_fp4(
        hidden_states: torch.Tensor,
        shared_gate_up_packed: torch.Tensor,
        shared_gate_up_scales: torch.Tensor,
        shared_down_packed: torch.Tensor,
        shared_down_scales: torch.Tensor,
        routed_gate_up_packed: torch.Tensor,
        routed_gate_up_scales: torch.Tensor,
        routed_down_packed: torch.Tensor,
        routed_down_scales: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_probs: torch.Tensor,
        group_size: int = 128,
    ) -> torch.Tensor:
        """Fused routed+shared expert decode with FP4 weights."""
        logger.debug("moe_fused_dispatch_shared_fp4 called with hidden_states=%s, shared_gate_up_packed=%s, shared_gate_up_scales=%s", hidden_states, shared_gate_up_packed, shared_gate_up_scales)
        require_mps()

        hidden_2d, num_tokens, hidden_dim, orig_shape = _flatten_moe_hidden(
            hidden_states
        )

        if hidden_dim % FP4_PER_UINT != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by {FP4_PER_UINT}"
            )
        if hidden_dim % group_size != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by group_size ({group_size})"
            )

        intermediate_dim = shared_gate_up_packed.shape[1] // 2
        if intermediate_dim % group_size != 0:
            raise ValueError(
                f"intermediate ({intermediate_dim}) must be divisible by group_size ({group_size})"
            )

        num_experts = routed_gate_up_packed.shape[0]
        top_k = expert_ids.shape[1]

        def _prepare_tensor(t: torch.Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
            logger.debug("_prepare_tensor called with t=%s, dtype=%s", t, dtype)
            tensor = t.contiguous()
            if tensor.dtype != dtype:
                tensor = tensor.to(dtype)
            return tensor

        hidden_contig = _prepare_tensor(hidden_2d)
        shared_gate_up_p = _prepare_tensor(shared_gate_up_packed, torch.uint32)
        shared_gate_up_s = _prepare_tensor(shared_gate_up_scales)
        shared_down_p = _prepare_tensor(shared_down_packed, torch.uint32)
        shared_down_s = _prepare_tensor(shared_down_scales)
        routed_gate_up_p = _prepare_tensor(routed_gate_up_packed, torch.uint32)
        routed_gate_up_s = _prepare_tensor(routed_gate_up_scales)
        routed_down_p = _prepare_tensor(routed_down_packed, torch.uint32)
        routed_down_s = _prepare_tensor(routed_down_scales)
        expert_ids_prepared = _prepare_tensor(expert_ids, torch.int32)
        expert_probs_prepared = _prepare_tensor(expert_probs, torch.float16)

        out = torch.empty((num_tokens, hidden_dim), dtype=torch.float16, device="mps")

        lib = get_default_library()
        device = lib.device

        a_buf = _private_buffer_from_tensor(hidden_contig, lib, device, cache=False)
        sg_p_buf = _private_buffer_from_tensor(shared_gate_up_p, lib, device, cache=True)
        sg_s_buf = _private_buffer_from_tensor(shared_gate_up_s, lib, device, cache=True)
        sd_p_buf = _private_buffer_from_tensor(shared_down_p, lib, device, cache=True)
        sd_s_buf = _private_buffer_from_tensor(shared_down_s, lib, device, cache=True)
        rg_p_buf = _private_buffer_from_tensor(routed_gate_up_p, lib, device, cache=True)
        rg_s_buf = _private_buffer_from_tensor(routed_gate_up_s, lib, device, cache=True)
        rd_p_buf = _private_buffer_from_tensor(routed_down_p, lib, device, cache=True)
        rd_s_buf = _private_buffer_from_tensor(routed_down_s, lib, device, cache=True)
        ids_buf = _private_buffer_from_tensor(
            expert_ids_prepared, lib, device, cache=False
        )
        probs_buf = _private_buffer_from_tensor(
            expert_probs_prepared, lib, device, cache=False
        )
        out_buf = mps_tensor_to_metal_buffer(out, device, copy_back=True)

        params = np.array(
            [num_tokens, hidden_dim, intermediate_dim, num_experts, top_k, group_size],
            dtype=np.uint32,
        )
        params_buf = _params_buffer(lib, device, params)

        tile_n = 64
        grid_x = (hidden_dim + tile_n - 1) // tile_n
        grid_y = num_tokens

        dispatch_kernel(
            lib,
            function_name="moe_fused_dispatch_shared_decode_fp4",
            grid=(grid_x, grid_y, 1),
            threadgroup=(128, 1, 1),
            buffers=[
                a_buf,
                sg_p_buf,
                sg_s_buf,
                sd_p_buf,
                sd_s_buf,
                rg_p_buf,
                rg_s_buf,
                rd_p_buf,
                rd_s_buf,
                ids_buf,
                probs_buf,
                out_buf,
                params_buf,
            ],
            wait=True,
        )

        return out.reshape(*orig_shape[:-1], hidden_dim)

    def moe_add_shared_expert_fp4(
        hidden_states: torch.Tensor,
        moe_output: torch.Tensor,
        shared_gate_up_packed: torch.Tensor,
        shared_gate_up_scales: torch.Tensor,
        shared_down_packed: torch.Tensor,
        shared_down_scales: torch.Tensor,
        group_size: int = 128,
    ) -> torch.Tensor:
        """Add the shared expert contribution to an existing MoE output tensor."""
        logger.debug("moe_add_shared_expert_fp4 called with hidden_states=%s, moe_output=%s, shared_gate_up_packed=%s", hidden_states, moe_output, shared_gate_up_packed)
        require_mps()

        hidden_2d, num_tokens, hidden_dim, orig_shape = _flatten_moe_hidden(
            hidden_states
        )

        if hidden_dim % FP4_PER_UINT != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by {FP4_PER_UINT}"
            )

        intermediate_dim = shared_gate_up_packed.shape[1] // 2
        if intermediate_dim % group_size != 0:
            raise ValueError(
                f"intermediate ({intermediate_dim}) must be divisible by group_size ({group_size})"
            )

        if moe_output.shape != hidden_2d.shape:
            raise ValueError(
                f"moe_output shape {moe_output.shape} doesn't match hidden shape {hidden_2d.shape}"
            )

        out = moe_output.half().contiguous()

        lib = get_default_library()
        device = lib.device

        a_buf = _private_buffer_from_tensor(hidden_2d, lib, device, cache=False)
        sg_p_buf = _private_buffer_from_tensor(
            shared_gate_up_packed.contiguous(), lib, device, cache=True
        )
        sg_s_buf = _private_buffer_from_tensor(
            shared_gate_up_scales.half().contiguous(), lib, device, cache=True
        )
        sd_p_buf = _private_buffer_from_tensor(
            shared_down_packed.contiguous(), lib, device, cache=True
        )
        sd_s_buf = _private_buffer_from_tensor(
            shared_down_scales.half().contiguous(), lib, device, cache=True
        )
        out_buf = mps_tensor_to_metal_buffer(out, device, copy_back=True)

        params = np.array(
            [num_tokens, hidden_dim, intermediate_dim, 0, 0, group_size],
            dtype=np.uint32,
        )
        params_buf = _params_buffer(lib, device, params)

        tile_n = 64
        grid_x = (hidden_dim + tile_n - 1) // tile_n
        grid_y = num_tokens

        dispatch_kernel(
            lib,
            function_name="moe_add_shared_expert_fp4",
            grid=(grid_x, grid_y, 1),
            threadgroup=(128, 1, 1),
            buffers=[
                a_buf,
                sg_p_buf,
                sg_s_buf,
                sd_p_buf,
                sd_s_buf,
                out_buf,
                params_buf,
            ],
            wait=True,
        )

        return out.reshape(*orig_shape[:-1], hidden_dim)

    return {
        "group_tokens_by_expert_full": group_tokens_by_expert_full,
        "gather_for_experts": gather_for_experts,
        "scatter_expert_outputs": scatter_expert_outputs,
        "moe_shared_expert_fused": moe_shared_expert_fused,
        "moe_shared_expert_fused_fp4": moe_shared_expert_fused_fp4,
        "moe_shared_expert_scatter": moe_shared_expert_scatter,
        "moe_shared_expert_fp4": moe_shared_expert_fp4,
        "moe_expert_gemm_fp4": moe_expert_gemm_fp4,
        "moe_router_topk": moe_router_topk,
        "moe_fused_dispatch_shared_fp4": moe_fused_dispatch_shared_fp4,
        "moe_add_shared_expert_fp4": moe_add_shared_expert_fp4,
    }
