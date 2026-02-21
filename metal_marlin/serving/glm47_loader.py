"""GLM-4.7-Flash MMFP4 loader for serving integration.

This module provides:
- `is_glm47_model()` for format detection from `config.json`
- `GLM47Loader` for loading GLM-4.7-Flash MMFP4 checkpoints
- `load_glm47()` convenience helper
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from .._compat import require_torch, torch

GLM47_ARCHITECTURE = "GLM4ForCausalLM"
GLM47_DEFAULT_TOKENIZER_ID = "zai-org/GLM-4.7-Flash"

try:
    from ..glm4_moe_experts import QuantizedGlm4MoEExperts as GLM4MoEExperts
except Exception:
    GLM4MoEExperts = None  # type: ignore[assignment]


def _load_config_dict(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path)
    config_path = path / "config.json" if path.is_dir() else path
    if config_path.name != "config.json":
        raise FileNotFoundError(f"Expected config.json, got: {config_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {config_path}: expected object")
    return data


def _coerce_int(config: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return None


def is_glm47_model(model_path: str | Path) -> bool:
    """Return True if `config.json` identifies a GLM-4.7 model checkpoint."""
    try:
        config = _load_config_dict(model_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError):
        return False

    architectures = config.get("architectures")
    return bool(
        isinstance(architectures, list) and GLM47_ARCHITECTURE in architectures
    )


class GLM47Loader:
    """Loader for GLM-4.7-Flash MMFP4 checkpoints used by serving."""

    expected_num_layers: int = 47
    expected_num_experts: int = 64
    expected_top_k: int = 4

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "mps",
        max_seq_len: int = 4096,
        max_batch_size: int = 1,
        tokenizer_id: str | None = None,
        trust_remote_code: bool = True,
        kv_quantize_mode: str = "none",
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.max_seq_len = int(max_seq_len)
        self.max_batch_size = int(max_batch_size)
        self.tokenizer_id = tokenizer_id
        self.trust_remote_code = trust_remote_code
        self.kv_quantize_mode = kv_quantize_mode

        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.kv_cache: Any | None = None
        self.weight_loader: Any | None = None
        self.expert_layers: list[Any] = []

    def _warn_if_unexpected_shape(self, config: dict[str, Any]) -> None:
        num_layers = _coerce_int(config, "num_hidden_layers")
        if num_layers is not None and num_layers != self.expected_num_layers:
            warnings.warn(
                f"GLM47Loader expected {self.expected_num_layers} layers, found {num_layers}.",
                stacklevel=2,
            )

        num_experts = _coerce_int(config, "num_local_experts", "num_experts", "n_routed_experts")
        if num_experts is not None and num_experts != self.expected_num_experts:
            warnings.warn(
                f"GLM47Loader expected {self.expected_num_experts} experts, found {num_experts}.",
                stacklevel=2,
            )

        top_k = _coerce_int(config, "num_experts_per_tok", "num_experts_per_token")
        if top_k is not None and top_k != self.expected_top_k:
            warnings.warn(
                f"GLM47Loader expected top_k={self.expected_top_k}, found top_k={top_k}.",
                stacklevel=2,
            )

    def _resolve_tokenizer_id(self, config: dict[str, Any]) -> str:
        if self.tokenizer_id:
            return self.tokenizer_id

        for key in ("tokenizer_name", "_name_or_path"):
            value = config.get(key)
            if isinstance(value, str) and value and not Path(value).exists():
                return value

        return GLM47_DEFAULT_TOKENIZER_ID

    def _load_tokenizer(self, config: dict[str, Any]) -> Any:
        from transformers import AutoTokenizer

        tokenizer_id = self._resolve_tokenizer_id(config)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id,
                trust_remote_code=self.trust_remote_code,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=self.trust_remote_code,
            )

        if (
            getattr(tokenizer, "pad_token_id", None) is None
            and getattr(tokenizer, "eos_token_id", None) is not None
            and getattr(tokenizer, "eos_token", None) is not None
        ):
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _find_glm4_expert_layers(self, model: Any) -> list[Any]:
        if GLM4MoEExperts is None:
            return []
        return [module for module in model.modules() if isinstance(module, GLM4MoEExperts)]

    def _initialize_glm4_experts(self, model: Any) -> list[Any]:
        expert_layers = self._find_glm4_expert_layers(model)
        if expert_layers:
            return expert_layers

        from .._quantized_weights import _apply_moe_expert_weights
        from ..layer_replacement import replace_glm4_moe_experts

        replace_glm4_moe_experts(model, device=self.device)
        if self.weight_loader is not None:
            _apply_moe_expert_weights(model, self.weight_loader, self.device)

        expert_layers = self._find_glm4_expert_layers(model)
        if not expert_layers:
            raise RuntimeError(
                "GLM4MoEExperts initialization failed: no expert layers found after replacement."
            )
        return expert_layers

    def _create_mla_kv_cache(
        self,
        model: Any,
        *,
        batch_size: int | None = None,
        max_seq_len: int | None = None,
    ) -> Any:
        require_torch("GLM47Loader KV cache setup")
        assert torch is not None

        from ..kv_cache import MLAKVCache

        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("Loaded model is missing config; cannot create MLAKVCache.")

        kv_lora_rank = getattr(config, "kv_lora_rank", None)
        if kv_lora_rank is None:
            raise ValueError(
                "GLM-4.7-Flash MLA config missing `kv_lora_rank`; cannot create MLAKVCache."
            )

        num_layers = int(getattr(config, "num_hidden_layers", self.expected_num_layers))
        qk_rope_head_dim = int(getattr(config, "qk_rope_head_dim", 64))
        cache_batch = int(batch_size if batch_size is not None else self.max_batch_size)
        cache_seq = int(
            max_seq_len
            if max_seq_len is not None
            else getattr(config, "max_position_embeddings", self.max_seq_len)
        )
        cache_dtype = getattr(model, "dtype", torch.float16)
        if not isinstance(cache_dtype, torch.dtype):
            cache_dtype = torch.float16

        return MLAKVCache(
            num_layers=num_layers,
            batch_size=cache_batch,
            max_seq_len=cache_seq,
            kv_lora_rank=int(kv_lora_rank),
            qk_rope_head_dim=qk_rope_head_dim,
            device=self.device,
            dtype=cache_dtype,
            quantize_mode=self.kv_quantize_mode,
        )

    def create_kv_cache(
        self,
        batch_size: int = 1,
        max_seq_len: int | None = None,
    ) -> Any:
        """Create a fresh MLAKVCache instance for serving requests."""
        if self.model is None:
            raise RuntimeError("Model has not been loaded yet.")
        return self._create_mla_kv_cache(
            self.model,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )

    def load(self) -> Any:
        """Load GLM-4.7-Flash model, tokenizer, experts, and MLA KV cache."""
        config = _load_config_dict(self.model_path)
        if not is_glm47_model(self.model_path):
            architectures = config.get("architectures", [])
            raise ValueError(
                "Unsupported model format for GLM47Loader: "
                f"architectures={architectures!r} (expected {GLM47_ARCHITECTURE!r})."
            )
        self._warn_if_unexpected_shape(config)

        from ..mmfp4_loader import MMFP4ModelLoader
        from ..model_utils import load_prequantized_mmfp4_model

        # Parse local quantized checkpoint metadata up front.
        self.weight_loader = MMFP4ModelLoader(self.model_path)

        # Load MMFP4-quantized weights from local checkpoint.
        model, _ = load_prequantized_mmfp4_model(
            str(self.model_path),
            device=self.device,
            bits=4,
        )

        # Ensure GLM4 MoE experts are present and initialized.
        self.expert_layers = self._initialize_glm4_experts(model)

        # HuggingFace tokenizer (remote ID preferred, local fallback).
        self.tokenizer = self._load_tokenizer(config)

        # MLA KV cache for serving/paged attention workflows.
        self.kv_cache = self._create_mla_kv_cache(model)

        # Attach serving-friendly handles to the loaded model.
        setattr(model, "_serving_tokenizer", self.tokenizer)
        setattr(model, "_serving_mla_kv_cache", self.kv_cache)
        setattr(model, "_serving_glm47_loader", self)
        setattr(model, "_serving_glm4_experts", self.expert_layers)
        if not hasattr(model, "create_kv_cache"):
            setattr(model, "create_kv_cache", self.create_kv_cache)

        self.model = model
        return model


def load_glm47(
    model_path: str | Path,
    *,
    device: str = "mps",
    max_seq_len: int = 4096,
    max_batch_size: int = 1,
    tokenizer_id: str | None = None,
    trust_remote_code: bool = True,
    kv_quantize_mode: str = "none",
) -> Any:
    """Load a GLM-4.7-Flash MMFP4 checkpoint and return a model ready for inference."""
    loader = GLM47Loader(
        model_path=model_path,
        device=device,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        tokenizer_id=tokenizer_id,
        trust_remote_code=trust_remote_code,
        kv_quantize_mode=kv_quantize_mode,
    )
    return loader.load()


__all__ = [
    "GLM4MoEExperts",
    "GLM47Loader",
    "is_glm47_model",
    "load_glm47",
]
