"""Batched model execution for continuous batching inference.

Executes a transformer model on mixed prefill/decode batches using paged
KV cache. Each scheduler iteration produces a SchedulerOutput with
prefill and decode requests; the runner flattens these into a single
forward pass with appropriate masking and position encodings.

The runner does NOT own the model or the block allocator. It receives
them at construction and operates on the block pool in-place.

Usage:
    runner = BatchedModelRunner(model, config, allocator)
    logits = runner.execute(schedule)
    # logits: [num_sequences, vocab_size] - next-token logits per sequence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
import torch.nn as nn

from ..kernels import HAS_METAL, paged_attention_fp4
from .request import SchedulerOutput

if TYPE_CHECKING:
    from ..paged.allocator import BlockAllocator


def _get_device() -> torch.device:
    """Get the appropriate device (MPS on Apple Silicon, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RoPEModule(Protocol):
    """Protocol for RoPE modules."""

    dims: int
    traditional: bool
    base: float


@dataclass
class ModelConfig:
    """Configuration for the causal LM being served."""

    vocab_size: int = 32000
    hidden_size: int = 4096
    num_heads: int = 32
    num_kv_heads: int = 32
    num_layers: int = 32
    intermediate_size: int = 11008
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096
    quant_type: str = "fp4"
    group_size: int = 128


@dataclass
class _BatchMetadata:
    """Internal metadata for a flattened batch."""

    input_ids: torch.Tensor  # [total_tokens]
    positions: torch.Tensor  # [total_tokens]
    block_tables: torch.Tensor  # [num_seqs, max_blocks_per_seq]
    context_lens: torch.Tensor  # [num_seqs]
    slot_offsets: torch.Tensor  # [num_seqs] - write offset for KV cache
    seq_starts: list[int]  # Start index in flattened input per sequence
    seq_lengths: list[int]  # Number of input tokens per sequence
    num_prefill: int  # Number of prefill sequences
    num_decode: int  # Number of decode sequences


class BatchedModelRunner:
    """Executes a causal LM on batched prefill+decode inputs with paged KV.

    Handles the complexity of mixed-length sequences in a single batch:
    - Prefill sequences contribute their full prompt as input
    - Decode sequences contribute only the last generated token
    - Each sequence has its own block table pointing into the shared pool
    - Position IDs are per-token within each sequence
    - KV writes target the correct block/slot via the block table

    The model must expose:
    - embed_tokens: nn.Embedding
    - layers: list[MarlinTransformerBlock]
    - norm: RMSNorm
    - lm_head: nn.Linear or MarlinLinear
    """

    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        allocator: BlockAllocator,
        use_metal_kernels: bool = True,
    ):
        self.model = model
        self.config = config
        self.allocator = allocator
        self.device = _get_device()
        self._layer_pools: list[torch.Tensor] | None = None
        self._use_metal_kernels = use_metal_kernels and HAS_METAL

    def execute(self, schedule: SchedulerOutput) -> torch.Tensor:
        """Execute one iteration of batched generation.

        Processes all prefill and decode requests in a single forward pass.
        Updates the paged KV cache in-place for each sequence.

        Args:
            schedule: Scheduler output containing prefill and decode requests.

        Returns:
            Next-token logits for each sequence [num_seqs, vocab_size].
            Prefill sequences return logits for the last prompt token.
            Returns empty array if schedule is empty.
        """
        if schedule.is_empty:
            return torch.empty(0, dtype=torch.float16, device=self.device)

        batch = self._build_batch(schedule)
        logits = self._forward(batch)
        return logits

    def _build_batch(self, schedule: SchedulerOutput) -> _BatchMetadata:
        """Flatten prefill + decode requests into contiguous batch tensors."""
        input_ids_list: list[int] = []
        positions_list: list[int] = []
        block_tables_list: list[list[int]] = []
        context_lens_list: list[int] = []
        slot_offsets_list: list[int] = []
        seq_starts: list[int] = []
        seq_lengths: list[int] = []

        offset = 0

        # Prefill: full prompt as input
        for req in schedule.prefill_requests:
            seq_len = req.num_prompt_tokens
            seq_starts.append(offset)
            seq_lengths.append(seq_len)

            input_ids_list.extend(req.prompt_tokens)
            positions_list.extend(range(seq_len))
            block_tables_list.append(list(req.block_indices))
            context_lens_list.append(seq_len)
            slot_offsets_list.append(0)  # Writing from position 0

            offset += seq_len

        # Decode: only the last token as input
        for req in schedule.decode_requests:
            seq_starts.append(offset)
            seq_lengths.append(1)

            last_token = req.output_tokens[-1] if req.output_tokens else req.prompt_tokens[-1]
            input_ids_list.append(last_token)
            positions_list.append(req.num_tokens - 1)
            block_tables_list.append(list(req.block_indices))
            context_lens_list.append(req.num_tokens)
            # Writing at the position after existing tokens
            slot_offsets_list.append(req.num_tokens - 1)

            offset += 1

        # Pad block tables to uniform width
        max_blocks = max(len(bt) for bt in block_tables_list) if block_tables_list else 0
        block_tables_padded = [bt + [0] * (max_blocks - len(bt)) for bt in block_tables_list]

        return _BatchMetadata(
            input_ids=torch.tensor(input_ids_list, dtype=torch.int32, device=self.device),
            positions=torch.tensor(positions_list, dtype=torch.int32, device=self.device),
            block_tables=torch.tensor(block_tables_padded, dtype=torch.int32, device=self.device),
            context_lens=torch.tensor(context_lens_list, dtype=torch.int32, device=self.device),
            slot_offsets=torch.tensor(slot_offsets_list, dtype=torch.int32, device=self.device),
            seq_starts=seq_starts,
            seq_lengths=seq_lengths,
            num_prefill=len(schedule.prefill_requests),
            num_decode=len(schedule.decode_requests),
        )

    def _forward(self, batch: _BatchMetadata) -> torch.Tensor:
        """Run the model forward pass with paged attention.

        For each transformer layer:
        1. Compute Q/K/V projections on the flattened input
        2. Write new K/V into the block pool
        3. Run paged attention (reading full context from blocks)
        4. Apply output projection and residual

        Returns:
            Logits for the last token of each sequence [num_seqs, vocab_size].
        """
        num_seqs = len(batch.seq_starts)

        # Token embedding
        hidden = self.model.embed_tokens(batch.input_ids)  # [total_tokens, hidden_size]

        # Process each transformer layer
        for layer_idx, layer in enumerate(self.model.layers):
            hidden = self._layer_forward(layer, layer_idx, hidden, batch)

        # Final norm
        hidden = self.model.norm(hidden)

        # Extract the last token per sequence for logits
        last_indices = torch.tensor(
            [batch.seq_starts[i] + batch.seq_lengths[i] - 1 for i in range(num_seqs)],
            dtype=torch.long,
            device=self.device,
        )
        last_hidden = hidden[last_indices]  # [num_seqs, hidden_size]

        # LM head projection
        logits = self.model.lm_head(last_hidden)  # [num_seqs, vocab_size]

        return logits

    def _layer_forward(
        self,
        layer: nn.Module,
        layer_idx: int,
        hidden: torch.Tensor,
        batch: _BatchMetadata,
    ) -> torch.Tensor:
        """Forward one transformer layer with paged KV cache.

        Decomposes the block's pre-norm -> attention -> residual -> norm -> MLP
        pattern to intercept the attention computation with paged KV.
        """
        num_seqs = len(batch.seq_starts)
        head_dim = self.config.head_dim
        num_heads = self.config.num_heads
        num_kv_heads = self.config.num_kv_heads
        block_size = self.block_size

        # === Attention sub-block ===
        residual = hidden
        hidden = layer.input_layernorm(hidden)

        # Compute Q/K/V projections [total_tokens, hidden_size] -> [total_tokens, proj_dim]
        q = layer.self_attn.q_proj(hidden)
        k = layer.self_attn.k_proj(hidden)
        v = layer.self_attn.v_proj(hidden)

        # Reshape to per-head: [total_tokens, num_heads, head_dim]
        q = q.view(-1, num_heads, head_dim)
        k = k.view(-1, num_kv_heads, head_dim)
        v = v.view(-1, num_kv_heads, head_dim)

        # Apply RoPE using position IDs
        q = self._apply_rope_flat(layer.self_attn.rope, q, batch.positions)
        k = self._apply_rope_flat(layer.self_attn.rope, k, batch.positions)

        # Write K/V into block pool for this layer
        self._write_kv_to_cache(k, v, batch, layer_idx)

        # Run paged attention per sequence
        attn_outputs: list[torch.Tensor] = []
        for seq_idx in range(num_seqs):
            start = batch.seq_starts[seq_idx]
            length = batch.seq_lengths[seq_idx]

            # Query for this sequence: [1, num_heads, seq_len, head_dim]
            q_seq = q[start : start + length]
            q_seq = q_seq.view(1, length, num_heads, head_dim).transpose(1, 2)

            # Block table and context length for this sequence
            seq_block_table = batch.block_tables[seq_idx : seq_idx + 1]
            seq_context_len = batch.context_lens[seq_idx : seq_idx + 1]

            # Paged attention
            attn_out = self._paged_attention(
                query=q_seq,
                block_pool=self._get_layer_pool(layer_idx),
                block_tables=seq_block_table,
                context_lens=seq_context_len,
                scale=layer.self_attn.scale,
                num_kv_heads=num_kv_heads,
                block_size=block_size,
            )
            # attn_out: [1, num_heads, seq_len, head_dim]
            # Reshape to [seq_len, num_heads * head_dim]
            attn_out = attn_out.transpose(1, 2).reshape(length, num_heads * head_dim)
            attn_outputs.append(attn_out)

        # Concatenate all sequences back to [total_tokens, hidden_size]
        attn_hidden = torch.cat(attn_outputs, dim=0)

        # Output projection
        attn_hidden = layer.self_attn.o_proj(attn_hidden)

        # Residual
        hidden = residual + attn_hidden

        # === MLP sub-block ===
        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = layer.mlp(hidden)
        hidden = residual + hidden

        return hidden

    def _apply_rope_flat(
        self,
        rope: RoPEModule,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RoPE to flattened tokens using per-token position IDs.

        Args:
            rope: RoPE module.
            x: Tensor [total_tokens, num_heads, head_dim].
            positions: Position IDs [total_tokens].

        Returns:
            Tensor with RoPE applied [total_tokens, num_heads, head_dim].
        """
        dims = rope.dims
        device = x.device
        dtype = x.dtype

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            rope.base ** (torch.arange(0, dims, 2, dtype=torch.float32, device=device) / dims)
        )

        # Angles: [total_tokens, dims/2]
        pos_float = positions.float()
        freqs = pos_float[:, None] * inv_freq[None, :]

        cos = torch.cos(freqs)[:, None, :]  # [total_tokens, 1, dims/2]
        sin = torch.sin(freqs)[:, None, :]

        # Split even/odd
        x_even = x[..., ::2]  # [total_tokens, num_heads, head_dim/2]
        x_odd = x[..., 1::2]

        # Apply rotation (Llama-style by default matches rope.traditional=False)
        if rope.traditional:
            rot_even = x_even * cos - x_odd * sin
            rot_odd = x_even * sin + x_odd * cos
        else:
            rot_even = x_even * cos - x_odd * sin
            rot_odd = x_odd * cos + x_even * sin

        # Interleave back to [total_tokens, num_heads, head_dim]
        result = torch.stack([rot_even, rot_odd], dim=-1).view(x.shape)

        return result.to(dtype)

    def _paged_attention(
        self,
        query: torch.Tensor,
        block_pool: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        scale: float,
        num_kv_heads: int,
        block_size: int = 16,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention with paged KV cache.

        Gathers K/V from the block pool using per-sequence block tables,
        expands KV heads for GQA, and computes masked attention.

        Args:
            query: Query tensor [num_seqs, num_heads, seq_len, head_dim].
            block_pool: Pre-allocated KV storage
                [num_blocks, 2, block_size, num_kv_heads, head_dim].
            block_tables: Block indices per sequence [num_seqs, max_blocks_per_seq].
            context_lens: Number of valid KV tokens per sequence [num_seqs].
            scale: Attention scale factor (typically head_dim ** -0.5).
            num_kv_heads: Number of KV heads (for GQA expansion).
            block_size: Tokens per block.

        Returns:
            Attention output [num_seqs, num_heads, seq_len, head_dim].
        """
        # Try Metal kernel dispatch first if available and enabled
        if self._use_metal_kernels and HAS_METAL and query.device.type == "mps":
            return paged_attention_fp4(
                query=query,
                key_cache=block_pool,
                value_cache=block_pool,
                block_tables=block_tables,
                context_lens=context_lens,
                scale=scale,
            )

        # Fallback to existing PyTorch implementation
        num_seqs = query.shape[0]
        num_heads = query.shape[1]
        seq_len = query.shape[2]
        head_dim = query.shape[3]
        max_blocks = block_tables.shape[1]
        max_context = max_blocks * block_size
        device = query.device

        # Gather K and V from block pool for each sequence
        flat_indices = block_tables.reshape(-1).long()  # [num_seqs * max_blocks]

        # Gather: [num_seqs * max_blocks, 2, block_size, num_kv_heads, head_dim]
        gathered = block_pool[flat_indices]

        # Reshape to [num_seqs, max_blocks * block_size, num_kv_heads, head_dim]
        gathered = gathered.view(num_seqs, max_blocks, 2, block_size, num_kv_heads, head_dim)
        gathered = gathered.permute(0, 2, 1, 3, 4, 5)  # [num_seqs, 2, max_blocks, block_size, ...]
        gathered = gathered.reshape(num_seqs, 2, max_context, num_kv_heads, head_dim)

        # Split K and V: each [num_seqs, max_context, num_kv_heads, head_dim]
        keys = gathered[:, 0]  # [num_seqs, max_context, num_kv_heads, head_dim]
        values = gathered[:, 1]  # [num_seqs, max_context, num_kv_heads, head_dim]

        # Transpose to [num_seqs, num_kv_heads, max_context, head_dim]
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        # GQA expansion: repeat KV heads to match query heads
        if num_kv_heads < num_heads:
            repeat_factor = num_heads // num_kv_heads
            keys = keys.repeat_interleave(repeat_factor, dim=1)
            values = values.repeat_interleave(repeat_factor, dim=1)

        # Compute attention scores: [num_seqs, num_heads, seq_len, max_context]
        attn_weights = (query @ keys.transpose(-2, -1)) * scale

        # Build validity mask from context_lens
        kv_positions = torch.arange(max_context, device=device)[None, :]  # [1, max_context]
        context_lens_2d = context_lens[:, None].long()  # [num_seqs, 1]
        valid_mask = kv_positions < context_lens_2d  # [num_seqs, max_context]

        # Expand for broadcasting: [num_seqs, 1, 1, max_context]
        valid_mask = valid_mask[:, None, None, :]
        attn_weights = torch.where(
            valid_mask, attn_weights, torch.tensor(float("-inf"), device=device)
        )

        # Causal mask for prefill (seq_len > 1)
        if seq_len > 1:
            q_positions = torch.arange(seq_len, device=device)[
                None, None, :, None
            ]  # [1, 1, seq_len, 1]
            kv_pos_expanded = kv_positions[None, None, None, :]  # [1, 1, 1, max_context]

            offsets = context_lens_2d[:, None, None, :] - seq_len + q_positions
            causal_mask = kv_pos_expanded <= offsets
            attn_weights = torch.where(
                causal_mask, attn_weights, torch.tensor(float("-inf"), device=device)
            )

        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Compute output: [num_seqs, num_heads, seq_len, head_dim]
        output = attn_weights @ values

        return output

    def _write_kv_to_cache(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        batch: _BatchMetadata,
        layer_idx: int,
    ) -> None:
        """Write computed K/V into the paged block pool.

        For each sequence, determines the target block and slot from the
        block table and slot offset, then writes K/V values.
        """
        num_seqs = len(batch.seq_starts)
        block_size = self.block_size
        pool = self._get_layer_pool(layer_idx)

        for seq_idx in range(num_seqs):
            start = batch.seq_starts[seq_idx]
            length = batch.seq_lengths[seq_idx]
            base_offset = int(batch.slot_offsets[seq_idx].item())
            seq_blocks = batch.block_tables[seq_idx]

            k_seq = keys[start : start + length]  # [length, num_kv_heads, head_dim]
            v_seq = values[start : start + length]

            for tok_idx in range(length):
                abs_pos = base_offset + tok_idx
                block_in_seq = abs_pos // block_size
                slot_in_block = abs_pos % block_size
                phys_block = int(seq_blocks[block_in_seq].item())

                pool[phys_block, 0, slot_in_block] = k_seq[tok_idx]
                pool[phys_block, 1, slot_in_block] = v_seq[tok_idx]

    def _get_layer_pool(self, layer_idx: int) -> torch.Tensor:
        """Get the KV block pool for a specific layer.

        If the model uses per-layer pools (stored as a list on the runner),
        returns the appropriate one. Otherwise returns the allocator's
        shared storage.
        """
        if self._layer_pools is not None:
            return self._layer_pools[layer_idx]
        return self.allocator._storage

    def _set_layer_pool(self, layer_idx: int, pool: torch.Tensor) -> None:
        """Update the KV block pool for a specific layer."""
        if self._layer_pools is not None:
            self._layer_pools[layer_idx] = pool
        else:
            self.allocator._storage = pool

    def init_layer_pools(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        """Initialize per-layer KV cache pools.

        Allocates separate block pools for each transformer layer.
        This avoids block-table aliasing across layers and is the
        recommended setup for multi-layer models.

        Args:
            num_blocks: Total number of blocks in the pool.
            block_size: Number of tokens per block.
            num_kv_heads: Number of KV heads.
            head_dim: Dimension per attention head.
        """
        self._layer_pools = [
            torch.zeros(
                (num_blocks, 2, block_size, num_kv_heads, head_dim),
                dtype=torch.float16,
                device=self.device,
            )
            for _ in range(self.config.num_layers)
        ]

    @property
    def block_size(self) -> int:
        """Block size from the allocator."""
        return getattr(self.allocator, "block_size", 16)

    @property
    def num_free_blocks(self) -> int:
        """Number of free blocks in the allocator."""
        return getattr(self.allocator, "num_free", 0)
