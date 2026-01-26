from dataclasses import dataclass

import torch


@dataclass
class MLAKVCache:
    """Compressed KV cache for MLA attention.

    Instead of storing full K,V tensors, stores the compressed latent
    representation (c_kv) plus the positional k_pe component.
    """

    batch_size: int
    num_layers: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    max_seq_len: int
    device: str = "mps"

    def __post_init__(self):
        # Storage: [batch, layers, max_seq_len, kv_lora_rank + qk_rope_head_dim]
        cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self._cache = torch.zeros(
            (self.batch_size, self.num_layers, self.max_seq_len, cache_dim),
            dtype=torch.float16,
            device=self.device,
        )
        self._seq_len = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def update(
        self,
        layer_idx: int,
        c_kv: torch.Tensor,  # [batch, seq_len, kv_lora_rank]
        k_pe: torch.Tensor,  # [batch, seq_len, qk_rope_head_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store compressed KV and return full cache for this layer."""
        new_seq_len = c_kv.shape[1]
        start = self._seq_len
        end = start + new_seq_len

        # Concatenate and store
        combined = torch.cat([c_kv, k_pe], dim=-1)
        self._cache[:, layer_idx, start:end] = combined

        # Return full cache for attention
        full = self._cache[:, layer_idx, :end]
        c_kv_full = full[..., :self.kv_lora_rank]
        k_pe_full = full[..., self.kv_lora_rank:]

        return c_kv_full, k_pe_full

    def advance(self, num_tokens: int):
        self._seq_len += num_tokens

    def reset(self):
        self._seq_len = 0
        self._cache.zero_()
