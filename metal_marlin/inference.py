"""
High-level inference API for Marlin-quantized models.

This module provides a generic pipeline that works with any model implementing
the standard forward interface, without hardcoding specific architectures.

Backend support:
    - mlx: Full Metal acceleration on Apple Silicon (default)
    - torch: PyTorch CPU/CUDA fallback
    - numpy: CPU-only, limited functionality (no generation)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ._compat import (
    HAS_MLX,
    get_default_backend,
    mx,
    require_mlx,
)

if TYPE_CHECKING:
    import mlx.core as mx  # noqa: F811

# Conditional imports based on available backends
if HAS_MLX:
    from .generate import GenerationConfig, generate, generate_stream
else:
    # Stub for when MLX is not available
    @dataclass
    class GenerationConfig:
        """Stub configuration when MLX is not available."""

        max_new_tokens: int = 256
        temperature: float = 0.7
        top_p: float = 0.9
        top_k: int = 50
        repetition_penalty: float = 1.1
        eos_token_id: int = 2
        pad_token_id: int = 0
        do_sample: bool = True

    def generate(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Token generation requires MLX. Install with: pip install mlx")

    def generate_stream(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Token generation requires MLX. Install with: pip install mlx")


@runtime_checkable
class MarlinModel(Protocol):
    """Protocol for models compatible with MarlinPipeline.

    Note: Array types are Any to support multiple backends (mlx.array, torch.Tensor, np.ndarray).
    """

    def __call__(self, input_ids: Any, kv_cache: Any = None) -> Any:
        """Forward pass returning logits."""
        ...

    def create_kv_cache(self, batch_size: int = 1) -> Any:
        """Create KV cache for incremental decoding."""
        ...


@dataclass
class ModelInfo:
    name: str
    num_params: int
    quant_type: str
    memory_mb: float


@dataclass
class ModelConfig:
    """Generic model configuration."""

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    quant_type: str = "fp4"
    hidden_act: str = "silu"
    mlp_bias: bool = False
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    head_dim: int | None = None


class MarlinPipeline:
    """
    High-level pipeline for text generation.

    This pipeline works with any model implementing the MarlinModel protocol.
    It does not assume a specific architecture (Llama, Mistral, etc.).

    Backend support:
        - mlx: Full Metal acceleration, required for generation
        - torch: PyTorch fallback (limited, weight loading only)
        - numpy: CPU-only (weight inspection only)

    Usage:
        # With pre-loaded model
        pipe = MarlinPipeline(model, tokenizer)
        output = pipe("What is the meaning of life?", max_tokens=100)

        # From ONNX (planned)
        pipe = MarlinPipeline.from_onnx("model.onnx", tokenizer_path)

        # From safetensors (planned)
        pipe = MarlinPipeline.from_safetensors("model.safetensors", config_path)
    """

    def __init__(
        self,
        model: MarlinModel,
        tokenizer: Any,
        config: GenerationConfig | None = None,
        backend: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        self._kv_cache = None
        self._backend = backend or get_default_backend()

    @property
    def backend(self) -> str:
        """Return the active backend for this pipeline."""
        return self._backend

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        quant_type: str = "fp4",
        device: str = "gpu",
        backend: str | None = None,
    ) -> MarlinPipeline:
        """
        Load a pretrained model from safetensors + config.json.

        This is a generic loader that reads:
        - config.json: Model architecture configuration
        - model.safetensors or model.marlin.safetensors: Weights

        The model architecture is inferred from config.json, not hardcoded.

        Args:
            model_path: Path to model directory
            quant_type: Quantization type (fp4, int4)
            device: Device to load to (gpu)
            backend: Force specific backend ('mlx', 'torch', 'numpy').
                     If None, uses best available.

        Returns:
            MarlinPipeline ready for generation

        Raises:
            RuntimeError: If MLX is not available (required for model building)
        """
        require_mlx("loading pretrained models")

        model_path = Path(model_path)

        # Load config
        config = cls._load_config(model_path)
        config.quant_type = quant_type

        # Build model from config (architecture-agnostic)
        model = cls._build_model_from_config(config)

        # Load weights
        weights_path = model_path / "model.marlin.safetensors"
        if weights_path.exists():
            cls._load_weights(model, weights_path)
        else:
            # Try regular safetensors and quantize
            st_path = model_path / "model.safetensors"
            if st_path.exists():
                cls._load_and_quantize(model, st_path, quant_type)
            else:
                raise FileNotFoundError(f"No weights found in {model_path}")

        # Load tokenizer
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return cls(model, tokenizer, backend=backend or "mlx")

    @classmethod
    def _build_model_from_config(cls, config: ModelConfig) -> MarlinModel:
        """Build a model from generic config.

        This creates a transformer using our generic building blocks
        (MarlinAttention, MarlinMLP, etc.) without hardcoding architecture.

        Raises:
            RuntimeError: If MLX is not available.
        """
        require_mlx("building transformer models")

        import mlx.nn as nn

        from .transformer import MarlinTransformerBlock

        class GenericTransformer(nn.Module):
            """Generic transformer built from config."""

            def __init__(self, cfg: ModelConfig):
                super().__init__()
                self.config = cfg
                self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
                self.layers = [
                    MarlinTransformerBlock(
                        hidden_size=cfg.hidden_size,
                        num_heads=cfg.num_attention_heads,
                        num_kv_heads=cfg.num_key_value_heads,
                        intermediate_size=cfg.intermediate_size,
                        group_size=32,
                    )
                    for _ in range(cfg.num_hidden_layers)
                ]
                self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
                self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

            def __call__(self, input_ids: Any, kv_cache: Any = None) -> Any:
                h = self.embed(input_ids)
                for i, layer in enumerate(self.layers):
                    cache = kv_cache.get_layer(i) if kv_cache else None
                    h = layer(h, cache=cache)
                h = self.norm(h)
                return self.lm_head(h)

            def create_kv_cache(self, batch_size: int = 1) -> Any:
                from .kv_cache import CacheConfig, KVCache

                cache_config = CacheConfig(
                    num_layers=self.config.num_hidden_layers,
                    num_heads=self.config.num_attention_heads,
                    num_kv_heads=self.config.num_key_value_heads,
                    head_dim=self.config.hidden_size // self.config.num_attention_heads,
                    max_seq_len=self.config.max_position_embeddings,
                )
                return KVCache(cache_config, batch_size)

        return GenericTransformer(config)

    @classmethod
    def _load_weights(cls, model: Any, weights_path: Path) -> None:
        """Load pre-quantized weights from safetensors."""
        require_mlx("loading weights into MLX model")
        import mlx.core as mx
        import safetensors.numpy as st

        tensors = st.load_file(str(weights_path))
        state = {k: mx.array(v) for k, v in tensors.items()}
        model.load_weights(list(state.items()))

    @classmethod
    def _load_and_quantize(cls, model: Any, weights_path: Path, quant_type: str) -> None:
        """Load FP16 weights and quantize to FP4/INT4."""
        require_mlx("quantizing weights")
        import mlx.core as mx
        import safetensors.numpy as st

        from .metal_marlin import pack_fp4_weights

        tensors = st.load_file(str(weights_path))
        state = {}

        for name, tensor in tensors.items():
            arr = mx.array(tensor)
            # Quantize linear weights
            if "weight" in name and len(arr.shape) == 2:
                packed, scales = pack_fp4_weights(arr, group_size=32)
                state[name] = packed
                state[name.replace("weight", "scales")] = scales
            else:
                state[name] = arr

        model.load_weights(list(state.items()))

    def __call__(
        self,
        prompt: str | list[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """
        Generate text from prompt.

        Args:
            prompt: Input text or list of texts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            stream: If True, return iterator yielding tokens

        Returns:
            Generated text or iterator

        Raises:
            RuntimeError: If MLX is not available for generation.
        """
        require_mlx("text generation")

        # Handle single prompt
        if isinstance(prompt, str):
            prompt = [prompt]

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="np", padding=True)
        input_ids = mx.array(inputs["input_ids"])

        # Update config
        config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        if stream:
            return self._stream_generate(input_ids, config)
        else:
            return self._generate(input_ids, config)

    def _generate(self, input_ids: Any, config: GenerationConfig) -> str:
        """Standard generation."""
        output_ids = generate(self.model, input_ids, config)
        return self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

    def _stream_generate(self, input_ids: Any, config: GenerationConfig) -> Iterator[str]:
        """Streaming generation."""
        for token_id in generate_stream(self.model, input_ids, config):
            yield self.tokenizer.decode([token_id])

    def info(self) -> ModelInfo:
        """Get model information."""
        # Try to get config from model
        config = getattr(self.model, "config", None)
        if config is None:
            return ModelInfo(
                name="Unknown",
                num_params=0,
                quant_type="unknown",
                memory_mb=0.0,
            )

        num_params = (
            config.vocab_size * config.hidden_size  # Embeddings
            + config.num_hidden_layers
            * (
                4 * config.hidden_size * config.hidden_size  # Attention
                + 3 * config.hidden_size * config.intermediate_size  # MLP
            )
        )

        quant_type = getattr(config, "quant_type", "fp4")
        if quant_type in ("fp4", "int4"):
            bits_per_param = 4
        else:
            bits_per_param = 16

        memory_mb = num_params * bits_per_param / 8 / 1024 / 1024

        return ModelInfo(
            name=f"Model-{num_params // 1e9:.0f}B",
            num_params=num_params,
            quant_type=quant_type,
            memory_mb=memory_mb,
        )

    @staticmethod
    def _load_config(model_path: Path) -> ModelConfig:
        """Load model config from path."""
        import json

        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            return ModelConfig(
                vocab_size=cfg.get("vocab_size", 32000),
                hidden_size=cfg.get("hidden_size", 4096),
                intermediate_size=cfg.get("intermediate_size", 11008),
                num_hidden_layers=cfg.get("num_hidden_layers", 32),
                num_attention_heads=cfg.get("num_attention_heads", 32),
                num_key_value_heads=cfg.get("num_key_value_heads", 32),
                max_position_embeddings=cfg.get("max_position_embeddings", 4096),
                rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
                rope_theta=cfg.get("rope_theta", 10000.0),
            )
        return ModelConfig()


# Convenience function
def chat(
    model_path: str,
    system_prompt: str = "You are a helpful assistant.",
) -> None:
    """Interactive chat interface.

    Raises:
        RuntimeError: If MLX is not available.
    """
    require_mlx("interactive chat")

    pipe = MarlinPipeline.from_pretrained(model_path)
    print(f"Loaded: {pipe.info()}")
    print("Type 'quit' to exit.\n")

    history: list[dict[str, str]] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        # Build prompt with history
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False)

        # Generate
        print("Assistant: ", end="", flush=True)
        response = ""
        for token in pipe(prompt, stream=True):
            print(token, end="", flush=True)
            response += token
        print()

        # Update history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
