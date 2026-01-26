"""MLA KV cache with latent storage and optional FP8 quantization.

Stores compressed KV latents (c_kv) plus decoupled RoPE positions (k_pe).
Supports dynamic growth when sequence length exceeds the pre-allocated capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._compat import HAS_TORCH, require_torch, torch

_FP8_E4M3_MAX = 448.0


def _resolve_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, str):
        if not HAS_TORCH or torch is None:
            raise RuntimeError("PyTorch is required to resolve dtype strings")
        mapping = {
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype string: {dtype}")
        return mapping[dtype]
    return dtype


@dataclass
class MLAKVCache:
    """Compressed KV cache for MLA attention.

    Stores:
        - c_kv: latent KV cache [num_layers, batch, seq, kv_lora_rank]
        - k_pe: RoPE portion [num_layers, batch, seq, qk_rope_head_dim]

    Latent storage supports optional FP8 quantization (uint8 + scale).
    """

    batch_size: int
    num_layers: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    max_seq_len: int
    device: str = "mps"
    dtype: torch.dtype | str = "float16"
    quantize_mode: Literal["none", "fp8"] = "none"
    fp8_scale_method: Literal["tensor", "channel"] = "tensor"

    def __post_init__(self) -> None:
        require_torch("MLA KV cache")
        if torch is None:
            raise RuntimeError("PyTorch is required for MLAKVCache")

        self.dtype = _resolve_dtype(self.dtype)
        self.seq_lens: list[int] = [0] * self.num_layers
        self._scale_dtype = torch.float16

        self._allocate(self.max_seq_len)

    def _allocate(self, max_seq_len: int) -> None:
        if self.quantize_mode not in ("none", "fp8"):
            raise ValueError(f"Unsupported quantize_mode: {self.quantize_mode}")

        if self.quantize_mode == "fp8":
            self.c_kv = torch.zeros(
                (self.num_layers, self.batch_size, max_seq_len, self.kv_lora_rank),
                dtype=torch.uint8,
                device=self.device,
            )
            scale_dim = self.kv_lora_rank if self.fp8_scale_method == "channel" else 1
            self.c_kv_scales = torch.zeros(
                (self.num_layers, self.batch_size, max_seq_len, scale_dim),
                dtype=self._scale_dtype,
                device=self.device,
            )
        else:
            self.c_kv = torch.zeros(
                (self.num_layers, self.batch_size, max_seq_len, self.kv_lora_rank),
                dtype=self.dtype,
                device=self.device,
            )
            self.c_kv_scales = None

        self.k_pe = torch.zeros(
            (self.num_layers, self.batch_size, max_seq_len, self.qk_rope_head_dim),
            dtype=self.dtype,
            device=self.device,
        )

    def _ensure_capacity(self, required_len: int) -> None:
        if required_len <= self.max_seq_len:
            return

        new_max = max(required_len, max(1, self.max_seq_len * 2))
        old_max = self.max_seq_len

        if self.quantize_mode == "fp8":
            new_c_kv = torch.zeros(
                (self.num_layers, self.batch_size, new_max, self.kv_lora_rank),
                dtype=torch.uint8,
                device=self.device,
            )
            scale_dim = self.kv_lora_rank if self.fp8_scale_method == "channel" else 1
            new_scales = torch.zeros(
                (self.num_layers, self.batch_size, new_max, scale_dim),
                dtype=self._scale_dtype,
                device=self.device,
            )
            new_c_kv[:, :, :old_max, :] = self.c_kv
            new_scales[:, :, :old_max, :] = self.c_kv_scales
            self.c_kv = new_c_kv
            self.c_kv_scales = new_scales
        else:
            new_c_kv = torch.zeros(
                (self.num_layers, self.batch_size, new_max, self.kv_lora_rank),
                dtype=self.dtype,
                device=self.device,
            )
            new_c_kv[:, :, :old_max, :] = self.c_kv
            self.c_kv = new_c_kv

        new_k_pe = torch.zeros(
            (self.num_layers, self.batch_size, new_max, self.qk_rope_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        new_k_pe[:, :, :old_max, :] = self.k_pe
        self.k_pe = new_k_pe

        self.max_seq_len = new_max

    @property
    def seq_len(self) -> int:
        """Current sequence length (assumes all layers are in sync)."""
        return self.seq_lens[0] if self.seq_lens else 0

    def update(
        self,
        layer_idx: int,
        c_kv_new: torch.Tensor,  # [batch, new_seq, kv_lora_rank]
        k_pe_new: torch.Tensor,  # [batch, new_seq, qk_rope_head_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store compressed KV and return full cache for this layer."""
        new_seq_len = c_kv_new.shape[1]
        start = self.seq_lens[layer_idx]
        end = start + new_seq_len

        self._ensure_capacity(end)

        if c_kv_new.device.type != self.device:
            c_kv_new = c_kv_new.to(self.device)
        if k_pe_new.device.type != self.device:
            k_pe_new = k_pe_new.to(self.device)

        if self.quantize_mode == "fp8":
            c_kv_quant, c_kv_scale = self._quantize_fp8(c_kv_new)
            self.c_kv[layer_idx, :, start:end, :] = c_kv_quant
            self.c_kv_scales[layer_idx, :, start:end, :] = c_kv_scale
        else:
            if c_kv_new.dtype != self.dtype:
                c_kv_new = c_kv_new.to(self.dtype)
            self.c_kv[layer_idx, :, start:end, :] = c_kv_new

        if k_pe_new.dtype != self.dtype:
            k_pe_new = k_pe_new.to(self.dtype)
        self.k_pe[layer_idx, :, start:end, :] = k_pe_new

        self.seq_lens[layer_idx] = end

        if self.quantize_mode == "fp8":
            c_kv_full = self._dequant_fp8(
                self.c_kv[layer_idx, :, :end, :],
                self.c_kv_scales[layer_idx, :, :end, :],
            ).to(self.dtype)
        else:
            c_kv_full = self.c_kv[layer_idx, :, :end, :]

        k_pe_full = self.k_pe[layer_idx, :, :end, :]

        return c_kv_full, k_pe_full

    def advance(self, num_tokens: int) -> None:
        """Advance cache position for all layers."""
        if num_tokens <= 0:
            return
        for idx in range(self.num_layers):
            self.seq_lens[idx] += num_tokens

    def reset(self) -> None:
        """Clear cache for new sequence (lengths only)."""
        self.seq_lens = [0] * self.num_layers

    def memory_usage_mb(self) -> float:
        """Return current memory usage in MB."""
        max_seq = max(self.seq_lens) if self.seq_lens else 0
        if max_seq == 0:
            return 0.0

        k_pe_bytes = (
            self.num_layers
            * self.batch_size
            * max_seq
            * self.qk_rope_head_dim
            * 2
        )

        if self.quantize_mode == "fp8":
            c_kv_bytes = (
                self.num_layers
                * self.batch_size
                * max_seq
                * self.kv_lora_rank
                * 1
            )
            scale_dim = self.kv_lora_rank if self.fp8_scale_method == "channel" else 1
            scale_bytes = (
                self.num_layers
                * self.batch_size
                * max_seq
                * scale_dim
                * 2
            )
        else:
            c_kv_bytes = (
                self.num_layers
                * self.batch_size
                * max_seq
                * self.kv_lora_rank
                * 2
            )
            scale_bytes = 0

        total_bytes = k_pe_bytes + c_kv_bytes + scale_bytes
        return total_bytes / 1024 / 1024

    def _quantize_fp8(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fp8_scale_method == "channel":
            abs_val = tensor.abs()
            scale = torch.clamp(abs_val, min=1e-8) / _FP8_E4M3_MAX
        else:
            abs_max = tensor.abs().amax(dim=-1, keepdim=True)
            abs_max = torch.clamp(abs_max, min=1e-8)
            scale = abs_max / _FP8_E4M3_MAX

        scaled = tensor / scale
        scaled = torch.clamp(scaled, -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
        quantized = torch.round(scaled / _FP8_E4M3_MAX * 127.0) + 128.0
        quantized = torch.clamp(quantized, 0, 255).to(torch.uint8)

        return quantized, scale.to(self._scale_dtype)

    def _dequant_fp8(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        signed = quantized.float() - 128.0
        return signed / 127.0 * _FP8_E4M3_MAX * scale.float()
