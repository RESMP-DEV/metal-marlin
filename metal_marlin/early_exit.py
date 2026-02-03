from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trellis.model import (
    CausalLMOutput,
    TrellisForCausalLM,
    TrellisModelConfig,
)


@dataclass
class EarlyExitOutput(CausalLMOutput):
    """Output from EarlyExitModel including exit layer index."""
    exit_layer_idx: int = -1

class EarlyExitModel(TrellisForCausalLM):
    """Trellis model with early exit capabilities.
    
    Adds a classifier head after each layer to allow exiting inference early
    if the model is confident in its prediction.
    """

    def __init__(self, config: TrellisModelConfig, exit_threshold: float = 0.9):
        super().__init__(config)
        self.exit_threshold = exit_threshold

        # Add classifier head after each layer
        # In a real scenario, these would be trained. Here we initialize them as linear projections.
        # We assume they project hidden_states -> vocab_size directly (after norm).
        self.exit_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.num_hidden_layers)
        ])

        # Initialize heads with main head weights if available (better than random)
        # But main head is initialized in super().__init__ which calls _load_base_weights
        # if from_pretrained is used. Here we are in __init__.
        # If initialized from scratch, weights are random.

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache = None,
        prefetch_kv: bool = True,
    ) -> EarlyExitOutput:
        """Forward pass with early exit.
        
        Iterates through layers and checks classifier confidence after each layer.
        If confidence > threshold, exits early and returns logits.
        """
        hidden_states = self.model.embed_tokens(input_ids)
        batch_size, seq_len = input_ids.shape
        device = hidden_states.device

        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = self.model._make_causal_mask(seq_len, device)

        # Get rope cache (sin, cos) for position-based lookup
        sin_cache = self.rope_sin_cache
        cos_cache = self.rope_cos_cache

        # Move caches to same device as hidden states
        sin_cache = sin_cache.to(device=device, dtype=hidden_states.dtype)
        cos_cache = cos_cache.to(device=device, dtype=hidden_states.dtype)

        # Pass position_ids and full caches
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # Get SHARED library for batch context
        lib = self._setup_shared_lib()

        use_prefetch = (
            prefetch_kv
            and kv_cache is not None
            and seq_len == 1
            and kv_cache.seq_len > 0
        )

        num_layers = len(self.model.layers)

        # We execute layers sequentially to allow early exit
        # This prevents using batch_dispatch for the whole model at once,
        # but individual layers (if MoE) can still use optimized kernels.

        # Region C: 128-256MB (Shared layer scratch) from TrellisModel
        mb = 1024 * 1024
        offset_scratch = 128 * mb

        exit_layer_idx = -1
        final_logits = None

        for i, layer in enumerate(self.model.layers):
            # Prefetch next layer's KV cache
            if use_prefetch and i + 1 < num_layers:
                # We can't use lib.batch_dispatch context here effectively if we break loop
                # So we use standard prefetch
                if lib is not None:
                    kv_cache.prefetch_layer_async(i + 1, lib=lib)
                else:
                    kv_cache.prefetch_layer(i + 1)

            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                rope_cos=cos_cache, # Fixed: passing the tensors directly as TrellisModel expects
                rope_sin=sin_cache,
                workspace=self.model.workspace,
                workspace_offset=offset_scratch,
            )

            # Check early exit confidence
            # Apply normalization (using the model's final norm)
            # Note: This assumes the intermediate hidden state is compatible with final norm.
            # In many Early Exit implementations, each layer has a dedicated norm.
            # Here we use the shared final norm for simplicity as per instructions.
            norm_hidden_states = self.model.norm(hidden_states)

            # Compute logits using this layer's exit head
            # Cast to fp32 for stability
            head_logits = self.exit_heads[i](norm_hidden_states.to(torch.float32))

            # Check confidence (for the last token in sequence)
            # We only check the last token for early exit decisions in generation
            last_token_logits = head_logits[:, -1, :]
            probs = F.softmax(last_token_logits, dim=-1)
            max_prob, _ = torch.max(probs, dim=-1)
            min_conf = torch.min(max_prob).item()

            if min_conf > self.exit_threshold:
                final_logits = head_logits
                exit_layer_idx = i
                break

        if final_logits is None:
            # Finished all layers without early exit
            norm_hidden_states = self.model.norm(hidden_states)
            final_logits = self.lm_head(norm_hidden_states.to(torch.float32))
            exit_layer_idx = num_layers - 1

        return EarlyExitOutput(logits=final_logits, exit_layer_idx=exit_layer_idx)
