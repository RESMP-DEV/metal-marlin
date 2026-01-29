"""Text generation for trellis-quantized models.

This module provides high-level text generation capabilities for models using
trellis quantization. It follows the HuggingFace generate() API for interface
compatibility while leveraging Metal-accelerated inference.

Usage:
    from metal_marlin.trellis_generate import TrellisGenerator, GenerationConfig
    from metal_marlin.trellis_linear import TrellisModelWrapper
    from transformers import AutoTokenizer

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("model_name")
    model = TrellisModelWrapper(base_model, "quantized_model_dir")
    model.replace_linear_layers()

    # Create generator and generate text
    generator = TrellisGenerator(model, tokenizer)
    config = GenerationConfig(max_new_tokens=256, temperature=0.7)
    output = generator.generate("Hello, world!", config=config)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._compat import require_torch, torch
from .trellis_kv_cache import TrellisKVCache

if TYPE_CHECKING:
    import torch as torch_typing


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Attributes:
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature. 1.0 = no change, < 1.0 = less random,
            > 1.0 = more random.
        top_k: Number of highest probability tokens to keep for top-k sampling.
            0 disables top-k.
        top_p: Cumulative probability threshold for nucleus sampling.
            1.0 disables top-p.
        repetition_penalty: Penalty for repeating tokens. 1.0 = no penalty.
        do_sample: Whether to use sampling or greedy decoding.
        eos_token_id: Token ID(s) that signal end of generation.
        pad_token_id: Token ID for padding.
    """

    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    do_sample: bool = True
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None


class TrellisGenerator:
    """Text generation with trellis-quantized models.

    This class provides a HuggingFace-compatible generate() interface for
    trellis-quantized models with support for temperature, top-k, top-p,
    repetition penalty, and streaming generation.

    Args:
        model: The trellis-quantized model (typically TrellisModelWrapper).
        tokenizer: HuggingFace tokenizer for encoding/decoding text.

    Example:
        >>> generator = TrellisGenerator(model, tokenizer)
        >>> config = GenerationConfig(max_new_tokens=100, temperature=0.7)
        >>> output = generator.generate("Hello", config=config)
        >>> print(output)
        'Hello, how can I help you today?'
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
    ):
        require_torch()

        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | list[str],
        config: GenerationConfig | None = None,
        stream: bool = False,
    ) -> str | list[str] | Iterator[str]:
        """Generate text from prompt(s).

        Generates text autoregressively from the given prompt(s) using the
        configured sampling strategy.

        Args:
            prompt: Input prompt string or list of strings for batch generation.
            config: Generation configuration. Uses defaults if not provided.
            stream: If True, yields text chunks as they are generated.

        Returns:
            Generated text string, list of strings (for batch), or iterator
            of text chunks (if streaming).

        Raises:
            RuntimeError: If PyTorch is not available.
            ValueError: If sequence length exceeds maximum cache size.
        """
        if config is None:
            config = GenerationConfig()

        # Handle single vs batch prompts
        if isinstance(prompt, str):
            prompts = [prompt]
            single = True
        else:
            prompts = prompt
            single = False

        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]

        # Get model config for KV cache
        model_config = self._get_model_config()
        num_layers = model_config.get("num_hidden_layers", 32)
        kv_lora_rank = model_config.get("kv_lora_rank", 512)

        # Initialize KV cache
        kv_cache = TrellisKVCache(
            num_layers=num_layers,
            batch_size=batch_size,
            max_seq_len=prompt_len + config.max_new_tokens,
            kv_lora_rank=kv_lora_rank,
            device=self.device,
        )

        if stream:
            # Return streaming generator for first sequence
            return self._generate_stream(
                input_ids[0:1],
                attention_mask[0:1] if attention_mask is not None else None,
                config,
                kv_cache,
            )

        # Generate for all sequences
        all_output_ids = []
        for i in range(batch_size):
            seq_input_ids = input_ids[i : i + 1]
            seq_attention_mask = (
                attention_mask[i : i + 1] if attention_mask is not None else None
            )

            # Reset cache for each sequence
            kv_cache.reset()

            output_ids = self._generate_single(
                seq_input_ids,
                seq_attention_mask,
                config,
                kv_cache,
            )
            all_output_ids.append(output_ids)

        # Decode outputs
        outputs = self.tokenizer.batch_decode(all_output_ids, skip_special_tokens=True)

        return outputs[0] if single else outputs

    @torch.inference_mode()
    def batch_generate(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
        batch_size: int = 8,
    ) -> list[str]:
        """Generate for multiple prompts with batching.

        Handles variable-length prompts by padding and tracking
        completion per sequence. More efficient than calling
        generate() individually for each prompt.

        Args:
            prompts: List of input prompt strings.
            config: Generation configuration. Uses defaults if not provided.
            batch_size: Number of prompts to process in each batch.

        Returns:
            List of generated text strings, one per prompt.

        Example:
            >>> prompts = ["Hello", "How are you?", "What is AI?", "Tell me a joke"]
            >>> outputs = generator.batch_generate(prompts, batch_size=2)
        """
        if config is None:
            config = GenerationConfig()

        outputs = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_outputs = self._generate_batch(batch_prompts, config)
            outputs.extend(batch_outputs)

        return outputs

    def _generate_batch(
        self,
        prompts: list[str],
        config: GenerationConfig,
    ) -> list[str]:
        """Generate for a single batch of prompts.

        Processes all prompts in the batch together, handling variable
        lengths through padding and tracking completion per sequence.

        Args:
            prompts: List of prompt strings for this batch.
            config: Generation configuration.

        Returns:
            List of generated text strings for the batch.
        """
        batch_size = len(prompts)

        # Tokenize with padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        seq_lens = attention_mask.sum(dim=1)  # Track actual lengths

        # Get model config for KV cache
        model_config = self._get_model_config()
        num_layers = model_config.get("num_hidden_layers", 32)
        kv_lora_rank = model_config.get("kv_lora_rank", 512)

        # Initialize KV cache
        max_len = input_ids.shape[1] + config.max_new_tokens
        kv_cache = TrellisKVCache(
            num_layers=num_layers,
            batch_size=batch_size,
            max_seq_len=max_len,
            kv_lora_rank=kv_lora_rank,
            device=self.device,
        )

        # Track which sequences are done
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        generated = [[] for _ in range(batch_size)]

        # Prefill phase: process entire prompt
        logits = self.model(input_ids, attention_mask=attention_mask, kv_cache=kv_cache)

        for step in range(config.max_new_tokens):
            # Get logits for last non-padding position
            next_token_logits = self._get_last_logits(logits, seq_lens)

            # Sample
            next_tokens = self._sample_batch(next_token_logits, config, generated)

            # Add to generated (unless finished)
            for j in range(batch_size):
                if not finished[j]:
                    generated[j].append(next_tokens[j].item())

            # Check EOS
            if config.eos_token_id is not None:
                eos_mask = self._is_eos_batch(next_tokens, config.eos_token_id)
                finished = finished | eos_mask

            if finished.all():
                break

            # Forward next token for unfinished sequences
            next_tokens_expanded = next_tokens.unsqueeze(1)  # [batch, 1]
            logits = self.model(
                next_tokens_expanded,
                kv_cache=kv_cache,
            )
            seq_lens = seq_lens + 1

        # Decode outputs
        outputs = []
        for j in range(batch_size):
            prompt_len = int(seq_lens[j].item()) - len(generated[j])
            full_ids = input_ids[j, :prompt_len].tolist() + generated[j]
            outputs.append(self.tokenizer.decode(full_ids, skip_special_tokens=True))

        return outputs

    def _get_last_logits(
        self,
        logits: torch_typing.Tensor,
        seq_lens: torch_typing.Tensor,
    ) -> torch_typing.Tensor:
        """Get logits at the last non-padding position for each sequence.

        This handles both the prefill phase (where logits has full sequence length)
        and the generation phase (where logits only has the last token).

        Args:
            logits: Logits tensor [batch, seq_len, vocab_size].
            seq_lens: Actual sequence lengths [batch].

        Returns:
            Logits at last position [batch, vocab_size].
        """
        # If logits only has one position (generation phase), use it directly
        if logits.shape[1] == 1:
            return logits[:, -1, :]

        # Prefill phase: need to get logits at actual sequence positions
        batch_size = logits.shape[0]
        last_logits = []
        for i in range(batch_size):
            # Get logits at position seq_lens[i] - 1 (last non-padding token)
            last_pos = int(seq_lens[i].item()) - 1
            last_logits.append(logits[i, last_pos, :])
        return torch.stack(last_logits)

    def _sample_batch(
        self,
        logits: torch_typing.Tensor,
        config: GenerationConfig,
        generated_tokens: list[list[int]],
    ) -> torch_typing.Tensor:
        """Sample next tokens for a batch.

        Args:
            logits: Logits tensor [batch, vocab_size].
            config: Generation configuration.
            generated_tokens: List of generated token lists per sequence.

        Returns:
            Sampled token IDs [batch].
        """
        batch_size = logits.shape[0]

        # Apply temperature scaling
        if config.temperature != 1.0 and config.temperature > 0:
            logits = logits / config.temperature

        # Apply repetition penalty per sequence
        if config.repetition_penalty != 1.0:
            logits = logits.clone()
            for i in range(batch_size):
                if generated_tokens[i]:
                    unique_tokens = set(generated_tokens[i])
                    for token_id in unique_tokens:
                        if logits[i, token_id] > 0:
                            logits[i, token_id] /= config.repetition_penalty
                        else:
                            logits[i, token_id] *= config.repetition_penalty

        # Greedy decoding
        if not config.do_sample:
            return logits.argmax(dim=-1)

        # Top-k filtering
        if config.top_k > 0:
            indices_to_remove = logits < torch.topk(logits, config.top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Top-p (nucleus) filtering
        if config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > config.top_p
            # Keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Sample from distribution
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _is_eos_batch(
        self,
        tokens: torch_typing.Tensor,
        eos_token_id: int | list[int] | None,
    ) -> torch_typing.Tensor:
        """Check which tokens are EOS tokens.

        Args:
            tokens: Token IDs [batch].
            eos_token_id: EOS token ID(s) to check against.

        Returns:
            Boolean tensor [batch] indicating EOS tokens.
        """
        if eos_token_id is None:
            return torch.zeros_like(tokens, dtype=torch.bool)

        if isinstance(eos_token_id, int):
            return tokens == eos_token_id

        # Multiple EOS tokens
        eos_mask = torch.zeros_like(tokens, dtype=torch.bool)
        for eos_id in eos_token_id:
            eos_mask = eos_mask | (tokens == eos_id)
        return eos_mask

    def _generate_single(
        self,
        input_ids: torch_typing.Tensor,
        attention_mask: torch_typing.Tensor | None,
        config: GenerationConfig,
        kv_cache: TrellisKVCache,
    ) -> torch_typing.Tensor:
        """Generate tokens for a single sequence.

        Args:
            input_ids: Input token IDs [1, seq_len].
            attention_mask: Optional attention mask [1, seq_len].
            config: Generation configuration.
            kv_cache: KV cache for the model.

        Returns:
            Full sequence tensor [1, total_seq_len] including prompt and generated.
        """
        # Prefill phase: process entire prompt
        logits = self.model(input_ids, attention_mask=attention_mask, kv_cache=kv_cache)
        next_token_logits = logits[:, -1, :]

        generated_tokens: list[int] = []
        max_length = input_ids.shape[1] + config.max_new_tokens

        for _ in range(config.max_new_tokens):
            # Sample next token
            next_token = self._sample(next_token_logits, config, generated_tokens)

            # Check for EOS
            if self._is_eos(next_token, config.eos_token_id):
                break

            generated_tokens.append(next_token.item())

            # Check if we've reached max length
            if len(generated_tokens) + input_ids.shape[1] >= max_length:
                break

            # Forward with new token
            next_token_tensor = torch.tensor(
                [[next_token]], device=self.device, dtype=torch.long
            )
            logits = self.model(
                next_token_tensor,
                kv_cache=kv_cache,
            )
            next_token_logits = logits[:, -1, :]

        # Combine prompt and generated tokens
        if generated_tokens:
            generated = torch.tensor(
                [generated_tokens], device=self.device, dtype=torch.long
            )
            output_ids = torch.cat([input_ids, generated], dim=1)
        else:
            output_ids = input_ids

        return output_ids

    def _generate_stream(
        self,
        input_ids: torch_typing.Tensor,
        attention_mask: torch_typing.Tensor | None,
        config: GenerationConfig,
        kv_cache: TrellisKVCache,
    ) -> Iterator[str]:
        """Stream generated text tokens.

        Args:
            input_ids: Input token IDs [1, seq_len].
            attention_mask: Optional attention mask [1, seq_len].
            config: Generation configuration.
            kv_cache: KV cache for the model.

        Yields:
            Decoded text chunks as they are generated.
        """
        # Prefill phase
        logits = self.model(input_ids, attention_mask=attention_mask, kv_cache=kv_cache)
        next_token_logits = logits[:, -1, :]

        generated_tokens: list[int] = []
        max_length = input_ids.shape[1] + config.max_new_tokens

        for _ in range(config.max_new_tokens):
            # Sample next token
            next_token = self._sample(next_token_logits, config, generated_tokens)

            # Check for EOS
            if self._is_eos(next_token, config.eos_token_id):
                break

            generated_tokens.append(next_token.item())

            # Yield decoded text
            yield self.tokenizer.decode([next_token.item()], skip_special_tokens=True)

            # Check if we've reached max length
            if len(generated_tokens) + input_ids.shape[1] >= max_length:
                break

            # Forward with new token
            next_token_tensor = torch.tensor(
                [[next_token]], device=self.device, dtype=torch.long
            )
            logits = self.model(
                next_token_tensor,
                kv_cache=kv_cache,
            )
            next_token_logits = logits[:, -1, :]

    def _sample(
        self,
        logits: torch_typing.Tensor,
        config: GenerationConfig,
        generated_tokens: list[int],
    ) -> torch_typing.Tensor:
        """Sample next token from logits.

        Args:
            logits: Logits tensor [batch, vocab_size].
            config: Generation configuration.
            generated_tokens: List of previously generated token IDs.

        Returns:
            Sampled token ID tensor [batch].
        """
        # Apply temperature scaling
        if config.temperature != 1.0 and config.temperature > 0:
            logits = logits / config.temperature

        # Apply repetition penalty
        if config.repetition_penalty != 1.0 and generated_tokens:
            logits = self._apply_repetition_penalty(
                logits, generated_tokens, config.repetition_penalty
            )

        # Greedy decoding
        if not config.do_sample:
            return logits.argmax(dim=-1)

        # Top-k filtering
        if config.top_k > 0:
            indices_to_remove = logits < torch.topk(logits, config.top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Top-p (nucleus) filtering
        if config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > config.top_p
            # Keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Sample from distribution
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _apply_repetition_penalty(
        self,
        logits: torch_typing.Tensor,
        generated_tokens: list[int],
        penalty: float,
    ) -> torch_typing.Tensor:
        """Apply repetition penalty to logits.

        Args:
            logits: Original logits tensor.
            generated_tokens: List of previously generated token IDs.
            penalty: Repetition penalty factor (>1 penalizes repetition).

        Returns:
            Modified logits with repetition penalty applied.
        """
        if not generated_tokens:
            return logits

        # Get unique tokens that have been generated
        unique_tokens = set(generated_tokens)

        # Apply penalty: reduce logits for tokens that have appeared
        for token_id in unique_tokens:
            if logits[0, token_id] > 0:
                logits = logits.clone()
                logits[0, token_id] /= penalty
            else:
                logits = logits.clone()
                logits[0, token_id] *= penalty

        return logits

    def _is_eos(
        self,
        token: torch_typing.Tensor,
        eos_token_id: int | list[int] | None,
    ) -> bool:
        """Check if token is an end-of-sequence token.

        Args:
            token: Token ID tensor.
            eos_token_id: EOS token ID(s) to check against.

        Returns:
            True if token is an EOS token.
        """
        if eos_token_id is None:
            return False

        token_val = token.item()

        if isinstance(eos_token_id, int):
            return token_val == eos_token_id

        return token_val in eos_token_id

    def stream_generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        """Generate text token-by-token with streaming output.

        Yields decoded text incrementally as tokens are generated.
        Each yield returns the full decoded text so far (not just the delta).

        Args:
            prompt: Input prompt string.
            config: Generation configuration (uses defaults if None).

        Yields:
            Decoded text string after each token is generated (cumulative).

        Example:
            >>> for text in generator.stream_generate("Hello", config):
            ...     print(text, end="\r")
        """
        if config is None:
            config = GenerationConfig()

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        # Get model config for KV cache
        model_config = self._get_model_config()
        num_layers = model_config.get("num_hidden_layers", 32)
        kv_lora_rank = model_config.get("kv_lora_rank", 512)

        # Initialize KV cache
        kv_cache = TrellisKVCache(
            num_layers=num_layers,
            batch_size=1,
            max_seq_len=input_ids.shape[1] + config.max_new_tokens,
            kv_lora_rank=kv_lora_rank,
            device=self.device,
        )

        # Prefill phase
        logits = self.model(input_ids, attention_mask=attention_mask, kv_cache=kv_cache)
        next_token_logits = logits[:, -1, :]

        # Track generated text for proper decoding
        generated_tokens: list[int] = []

        for _ in range(config.max_new_tokens):
            # Sample next token
            next_token = self._sample(next_token_logits, config, generated_tokens)
            generated_tokens.append(next_token.item())

            # Decode incrementally
            new_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )

            yield new_text

            # Check for EOS
            if self._is_eos(next_token, config.eos_token_id):
                break

            # Forward with new token
            next_token_tensor = torch.tensor(
                [[next_token]], device=self.device, dtype=torch.long
            )
            logits = self.model(
                next_token_tensor,
                kv_cache=kv_cache,
            )
            next_token_logits = logits[:, -1, :]

    def stream_generate_tokens(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[tuple[int, str]]:
        """Generate tokens with both ID and decoded text.

        Yields (token_id, decoded_token) tuples for each generated token.
        This is useful when you need both the raw token ID and its text representation.

        Args:
            prompt: Input prompt string.
            config: Generation configuration (uses defaults if None).

        Yields:
            Tuples of (token_id, decoded_text) for each generated token.

        Example:
            >>> for token_id, text in generator.stream_generate_tokens("Hello", config):
            ...     print(f"Token {token_id}: '{text}'")
        """
        if config is None:
            config = GenerationConfig()

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        # Get model config for KV cache
        model_config = self._get_model_config()
        num_layers = model_config.get("num_hidden_layers", 32)
        kv_lora_rank = model_config.get("kv_lora_rank", 512)

        # Initialize KV cache
        kv_cache = TrellisKVCache(
            num_layers=num_layers,
            batch_size=1,
            max_seq_len=input_ids.shape[1] + config.max_new_tokens,
            kv_lora_rank=kv_lora_rank,
            device=self.device,
        )

        # Prefill phase
        logits = self.model(input_ids, attention_mask=attention_mask, kv_cache=kv_cache)
        next_token_logits = logits[:, -1, :]

        generated_tokens: list[int] = []

        for _ in range(config.max_new_tokens):
            # Sample next token
            next_token = self._sample(next_token_logits, config, generated_tokens)
            token_id = next_token.item()
            generated_tokens.append(token_id)

            # Decode single token
            decoded = self.tokenizer.decode([token_id], skip_special_tokens=False)

            yield token_id, decoded

            # Check for EOS
            if self._is_eos(next_token, config.eos_token_id):
                break

            # Forward with new token
            next_token_tensor = torch.tensor(
                [[next_token]], device=self.device, dtype=torch.long
            )
            logits = self.model(
                next_token_tensor,
                kv_cache=kv_cache,
            )
            next_token_logits = logits[:, -1, :]

    def _get_model_config(self) -> dict:
        """Get model configuration from the wrapped model.

        Returns:
            Dictionary with model configuration.
        """
        # Try to get config from wrapped model
        if hasattr(self.model, "config"):
            config = self.model.config
            if hasattr(config, "to_dict"):
                return config.to_dict()
            if isinstance(config, dict):
                return config

        # Try to get from underlying model
        if hasattr(self.model, "model") and hasattr(self.model.model, "config"):
            config = self.model.model.config
            if hasattr(config, "to_dict"):
                return config.to_dict()
            if isinstance(config, dict):
                return config

        # Return defaults
        return {
            "num_hidden_layers": 32,
            "kv_lora_rank": 512,
        }


# Type hint for PreTrainedTokenizer to avoid direct import
class PreTrainedTokenizer:
    """Protocol for HuggingFace tokenizer interface."""

    def __call__(
        self,
        text: str | list[str],
        return_tensors: str | None = None,
        padding: bool = False,
        truncation: bool = False,
        **kwargs,
    ) -> dict:
        ...

    def decode(
        self,
        token_ids: list[int] | torch_typing.Tensor,
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        ...

    def batch_decode(
        self,
        sequences: list[list[int]] | torch_typing.Tensor,
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> list[str]:
        ...


__all__ = [
    "GenerationConfig",
    "TrellisGenerator",
]
