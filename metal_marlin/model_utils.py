from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from metal_marlin._quantized_weights import _apply_moe_expert_weights, _apply_quantized_weights
from metal_marlin.layer_replacement import replace_glm4_moe_experts, replace_linear_layers
from metal_marlin.layers.mtp_head import GLMMTPHead
from metal_marlin.mmfp4_loader import MMFP4ModelLoader

# --------------------------------------------------------------------------------------
# Qwen hybrid / DeltaNet-family config helpers
# --------------------------------------------------------------------------------------

_QWEN3_NEXT_MODEL_TYPES = frozenset({
    "qwen3_next",
    "qwen3_vl_moe_text",
    "qwen3_5_moe_text",
    "qwen3_6_moe_text",
})
_QWEN_MULTIMODAL_TYPES = frozenset({"qwen2_5_vl", "qwen3_vl_moe", "qwen2_vl", "qwen2_5_omni", "qwen3_omni_moe"})
_QWEN_DENSE_TYPES = frozenset({"qwen3", "qwen2", "qwen2_moe", "qwen3_moe"})


def get_effective_text_config(config: Any) -> Any:
    """Return the text sub-config for multimodal Qwen models.

    Qwen VL / VL-MoE models store language dimensions inside a nested
    ``text_config`` object (e.g. ``Qwen2_5_VLConfig``).  All other configs
    are returned as-is.

    Args:
        config: A HuggingFace PreTrainedConfig (or dict-compatible object).

    Returns:
        The config object that carries ``vocab_size`` / ``hidden_size`` / etc.
        For multimodal models this is ``config.text_config``; otherwise ``config``.
    """
    if hasattr(config, "text_config"):
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "vocab_size"):
            return text_cfg
    return config


def get_layer_types(config: Any) -> list[str] | None:
    """Return the ``layer_types`` list for a Qwen-family config.

    Checks both the top-level attribute and the effective text config
    (for multimodal models) before returning ``None``.

    Args:
        config: A HuggingFace PreTrainedConfig.

    Returns:
        The ``layer_types`` list (``["full_attention", "linear_attention", ...]``)
        when present, otherwise ``None``.
    """
    # Direct attribute first (flat configs: Qwen3Config, Qwen3NextConfig, VLText configs)
    if hasattr(config, "layer_types"):
        val = getattr(config, "layer_types")
        if isinstance(val, list):
            return val

    # Multimodal: check nested text_config
    text_cfg = get_effective_text_config(config)
    if text_cfg is not config and hasattr(text_cfg, "layer_types"):
        val = getattr(text_cfg, "layer_types")
        if isinstance(val, list):
            return val

    return None


def get_full_attention_interval(config: Any) -> int | None:
    """Return ``full_attention_interval`` for Qwen3-Next-family configs.

    ``full_attention_interval`` is stored as a ``kwargs`` default in
    ``Qwen3NextConfig.__init__`` and is only accessible via
    ``config.to_dict()`` or direct ``getattr`` on the config object.

    Checks both the top-level config and the nested ``text_config``
    (for multimodal models) before returning ``None``.

    Args:
        config: A HuggingFace PreTrainedConfig.

    Returns:
        The interval as an integer if present, otherwise ``None``.
    """
    # Check top-level first, then multimodal text sub-config
    candidates = [config]
    text_cfg = get_effective_text_config(config)
    if text_cfg is not config:
        candidates.append(text_cfg)

    for candidate in candidates:
        if hasattr(candidate, "full_attention_interval"):
            val = getattr(candidate, "full_attention_interval")
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass

        # Fallback: look inside the raw config dict stored by PreTrainedConfig
        as_dict: dict[str, Any] = {}
        if hasattr(candidate, "to_dict"):
            try:
                as_dict = candidate.to_dict()
            except Exception:
                pass
        elif hasattr(candidate, "__dict__"):
            as_dict = candidate.__dict__

        if "full_attention_interval" in as_dict:
            val = as_dict["full_attention_interval"]
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass

    return None


def get_deltanet_metadata(config: Any) -> dict[str, Any] | None:
    """Surface DeltaNet linear-attention metadata from a Qwen3-Next-family config.

    Returns ``None`` when the config is not a DeltaNet-family model or the
    fields are absent.

    The five DeltaNet-specific fields read from ``Qwen3NextConfig`` are:

    - ``linear_key_head_dim``
    - ``linear_value_head_dim``
    - ``linear_num_key_heads``
    - ``linear_num_value_heads``
    - ``linear_conv_kernel_dim``

    Args:
        config: A HuggingFace PreTrainedConfig.

    Returns:
        A dict with the five fields above when available, otherwise ``None``.
    """
    # Resolve the config object that actually carries DeltaNet fields.
    # For multimodal wrappers the fields live in text_config; for flat
    # configs they live at top level.
    source_cfg = config
    model_type = getattr(config, "model_type", "") or ""
    if model_type not in _QWEN3_NEXT_MODEL_TYPES:
        # Multimodal wrapper: check text sub-config
        text_cfg = get_effective_text_config(config)
        if text_cfg is not config:
            model_type = getattr(text_cfg, "model_type", "") or ""
            source_cfg = text_cfg
        else:
            model_type = ""

    if model_type not in _QWEN3_NEXT_MODEL_TYPES:
        return None

    fields = (
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_conv_kernel_dim",
    )
    result: dict[str, Any] = {}
    for name in fields:
        val = getattr(source_cfg, name, None)
        if val is None:
            return None  # Field missing → not a complete DeltaNet config
        result[name] = val

    return result


def is_qwen_hybrid_deltanet(config: Any) -> bool:
    """Detect whether a config belongs to the Qwen hybrid linear / full-attention family.

    This family is characterised by a ``layer_types`` list containing
    ``"linear_attention"`` (DeltaNet) alongside ``"full_attention"`` entries.

    Backward-compatible: returns ``False`` for dense Qwen models (``Qwen3Config``,
    ``Qwen2Config``, …) that only use ``"full_attention"`` and
    ``"sliding_attention"``.

    Args:
        config: A HuggingFace PreTrainedConfig.

    Returns:
        ``True`` if the config describes a DeltaNet-family hybrid model.
    """
    layer_types = get_layer_types(config)
    if layer_types is None:
        return False
    return "linear_attention" in layer_types


# --------------------------------------------------------------------------------------
# Existing public API
# --------------------------------------------------------------------------------------


logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Qwen hybrid / DeltaNet-family config helpers
# --------------------------------------------------------------------------------------

_QWEN3_NEXT_MODEL_TYPES = frozenset({
    "qwen3_next",
    "qwen3_vl_moe_text",
    "qwen3_5_moe_text",
    "qwen3_6_moe_text",
})
_QWEN_MULTIMODAL_TYPES = frozenset({"qwen2_5_vl", "qwen3_vl_moe", "qwen2_vl", "qwen2_5_omni", "qwen3_omni_moe"})
_QWEN_DENSE_TYPES = frozenset({"qwen3", "qwen2", "qwen2_moe", "qwen3_moe"})


def get_effective_text_config(config: Any) -> Any:
    """Return the text sub-config for multimodal Qwen models.

    Qwen VL / VL-MoE models store language dimensions inside a nested
    ``text_config`` object (e.g. ``Qwen2_5_VLConfig``).  All other configs
    are returned as-is.

    Args:
        config: A HuggingFace PreTrainedConfig (or dict-compatible object).

    Returns:
        The config object that carries ``vocab_size`` / ``hidden_size`` / etc.
        For multimodal models this is ``config.text_config``; otherwise ``config``.
    """
    logger.debug("get_effective_text_config called with config=%s", config)
    if hasattr(config, "text_config"):
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "vocab_size"):
            return text_cfg
    return config


def get_layer_types(config: Any) -> list[str] | None:
    """Return the ``layer_types`` list for a Qwen-family config.

    Checks both the top-level attribute and the effective text config
    (for multimodal models) before returning ``None``.

    Args:
        config: A HuggingFace PreTrainedConfig.

    Returns:
        The ``layer_types`` list (``["full_attention", "linear_attention", ...]``)
        when present, otherwise ``None``.
    """
    # Direct attribute first (flat configs: Qwen3Config, Qwen3NextConfig, VLText configs)
    logger.debug("get_layer_types called with config=%s", config)
    if hasattr(config, "layer_types"):
        val = getattr(config, "layer_types")
        if isinstance(val, list):
            return val

    # Multimodal: check nested text_config
    text_cfg = get_effective_text_config(config)
    if text_cfg is not config and hasattr(text_cfg, "layer_types"):
        val = getattr(text_cfg, "layer_types")
        if isinstance(val, list):
            return val

    return None


def get_full_attention_interval(config: Any) -> int | None:
    """Return ``full_attention_interval`` for Qwen3-Next-family configs.

    ``full_attention_interval`` is stored as a ``kwargs`` default in
    ``Qwen3NextConfig.__init__`` and is only accessible via
    ``config.to_dict()`` or direct ``getattr`` on the config object.

    Checks both the top-level config and the nested ``text_config``
    (for multimodal models) before returning ``None``.

    Args:
        config: A HuggingFace PreTrainedConfig.

    Returns:
        The interval as an integer if present, otherwise ``None``.
    """
    # Check top-level first, then multimodal text sub-config
    logger.debug("get_full_attention_interval called with config=%s", config)
    candidates = [config]
    text_cfg = get_effective_text_config(config)
    if text_cfg is not config:
        candidates.append(text_cfg)

    for candidate in candidates:
        if hasattr(candidate, "full_attention_interval"):
            val = getattr(candidate, "full_attention_interval")
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass

        # Fallback: look inside the raw config dict stored by PreTrainedConfig
        as_dict: dict[str, Any] = {}
        if hasattr(candidate, "to_dict"):
            try:
                as_dict = candidate.to_dict()
            except Exception:
                pass
        elif hasattr(candidate, "__dict__"):
            as_dict = candidate.__dict__

        if "full_attention_interval" in as_dict:
            val = as_dict["full_attention_interval"]
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass

    return None


def get_deltanet_metadata(config: Any) -> dict[str, Any] | None:
    """Surface DeltaNet linear-attention metadata from a Qwen3-Next-family config.

    Returns ``None`` when the config is not a DeltaNet-family model or the
    fields are absent.

    The five DeltaNet-specific fields read from ``Qwen3NextConfig`` are:

    - ``linear_key_head_dim``
    - ``linear_value_head_dim``
    - ``linear_num_key_heads``
    - ``linear_num_value_heads``
    - ``linear_conv_kernel_dim``

    Args:
        config: A HuggingFace PreTrainedConfig.

    Returns:
        A dict with the five fields above when available, otherwise ``None``.
    """
    # Resolve the config object that actually carries DeltaNet fields.
    # For multimodal wrappers the fields live in text_config; for flat
    # configs they live at top level.
    logger.debug("get_deltanet_metadata called with config=%s", config)
    source_cfg = config
    model_type = getattr(config, "model_type", "") or ""
    if model_type not in _QWEN3_NEXT_MODEL_TYPES:
        # Multimodal wrapper: check text sub-config
        text_cfg = get_effective_text_config(config)
        if text_cfg is not config:
            model_type = getattr(text_cfg, "model_type", "") or ""
            source_cfg = text_cfg
        else:
            model_type = ""

    if model_type not in _QWEN3_NEXT_MODEL_TYPES:
        return None

    fields = (
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_conv_kernel_dim",
    )
    result: dict[str, Any] = {}
    for name in fields:
        val = getattr(source_cfg, name, None)
        if val is None:
            return None  # Field missing → not a complete DeltaNet config
        result[name] = val

    return result


def is_qwen_hybrid_deltanet(config: Any) -> bool:
    """Detect whether a config belongs to the Qwen hybrid linear / full-attention family.

    This family is characterised by a ``layer_types`` list containing
    ``"linear_attention"`` (DeltaNet) alongside ``"full_attention"`` entries.

    Backward-compatible: returns ``False`` for dense Qwen models (``Qwen3Config``,
    ``Qwen2Config``, …) that only use ``"full_attention"`` and
    ``"sliding_attention"``.

    Args:
        config: A HuggingFace PreTrainedConfig.

    Returns:
        ``True`` if the config describes a DeltaNet-family hybrid model.
    """
    logger.debug("is_qwen_hybrid_deltanet called with config=%s", config)
    layer_types = get_layer_types(config)
    if layer_types is None:
        return False
    return "linear_attention" in layer_types


# --------------------------------------------------------------------------------------
# Existing public API
# --------------------------------------------------------------------------------------


def _load_mtp_head_from_checkpoint(
    model: Any,
    config: Any,
    loader: MMFP4ModelLoader,
) -> GLMMTPHead | None:
    """Build a GLM MTP head from checkpoint tensors if present.

    GLM 4.7 Flash stores auxiliary next-token predictor weights under:
    `model.layers.<num_hidden_layers>.shared_head.head.weight`.
    """
    logger.info("_load_mtp_head_from_checkpoint called with model=%s, config=%s, loader=%s", model, config, loader)
    num_tokens_hint = (
        getattr(config, "num_nextn_predict_layers", None)
        or getattr(config, "num_mtp_heads", None)
        or 0
    )
    try:
        num_tokens = int(num_tokens_hint)
    except (TypeError, ValueError):
        num_tokens = 0

    hidden_size = getattr(config, "hidden_size", None)
    vocab_size = getattr(config, "vocab_size", None)
    if num_tokens <= 0 or hidden_size is None or vocab_size is None:
        return None

    core = getattr(model, "model", None)
    layer_count = len(getattr(core, "layers", []))

    # Prefer auxiliary layer index immediately after the decoder stack
    # (e.g. decoder has layers [0..46], MTP head in layer 47).
    candidate_indices: list[int] = []
    if layer_count > 0:
        candidate_indices.append(layer_count)
        candidate_indices.extend(layer_count + i for i in range(1, num_tokens))
        candidate_indices.append(layer_count - 1)
    else:
        candidate_indices.extend(range(num_tokens))

    deduped_indices: list[int] = []
    seen_indices: set[int] = set()
    for idx in candidate_indices:
        if idx in seen_indices or idx < 0:
            continue
        seen_indices.add(idx)
        deduped_indices.append(idx)

    weights: list[torch.Tensor] = []
    for layer_idx in deduped_indices:
        key = f"model.layers.{layer_idx}.shared_head.head.weight"
        try:
            weight = loader.load_tensor(key)
        except Exception:
            continue

        if (
            not torch.is_tensor(weight)
            or weight.ndim != 2
            or weight.is_meta
            or not torch.is_floating_point(weight)
        ):
            continue

        if tuple(weight.shape) == (vocab_size, hidden_size):
            weights.append(weight)
        elif tuple(weight.shape) == (hidden_size, vocab_size):
            weights.append(weight.T)

        if len(weights) >= num_tokens:
            break

    if not weights:
        return None

    mtp_head = GLMMTPHead(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_tokens=len(weights),
    )
    with torch.no_grad():
        for idx, weight in enumerate(weights):
            target = mtp_head.heads[idx].weight
            target.copy_(weight.to(device=target.device, dtype=target.dtype))
    return mtp_head


def load_prequantized_mmfp4_model(
    model_path: str,
    device: str = "mps",
    bits: int = 4,
) -> tuple[Any, Any]:
    """Load an MMFP4 quantized model with all optimizations enabled.

    This function loads the model architecture WITHOUT weights, replaces
    linear layers with FP4 quantized versions, then loads the packed
    quantized weights from safetensors.

    Args:
        model_path: Path to the model directory
        device: Device to load model on ("mps", "cuda", "cpu")
        bits: Quantization bits (default 4 for FP4)

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("load_prequantized_mmfp4_model called with model_path=%s, device=%s, bits=%s", model_path, device, bits)
    print(f"Loading MMFP4 model from {model_path}...")
    quantized_path = Path(model_path)

    # Step 1: Load config only (no weights)
    print("  Loading model config...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    causal_config = config

    # Qwen3.5 multimodal configs keep text dimensions under `text_config`.
    # AutoModelForCausalLM needs the text-only config object.
    if not hasattr(causal_config, "vocab_size"):
        text_config = getattr(config, "text_config", None)
        if text_config is not None and hasattr(text_config, "vocab_size"):
            causal_config = text_config

    # Step 2: Create model architecture WITHOUT weight initialization
    # Using meta device avoids allocating/initializing 30B params (saves minutes)
    print("  Creating model architecture (meta device)...")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            causal_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    # Materialize to empty tensors on CPU (fast, no random init)
    model = model.to_empty(device="cpu")

    # Step 3: Replace linear layers with quantized versions (empty, no RTN)
    # Skip lm_head - checkpoint has FP16 lm_head.weight, not quantized
    print("  Replacing linear layers with FP4 quantized versions...")
    replace_linear_layers(
        model,
        bits=bits,
        prequantized=True,
        skip_patterns=["lm_head"],  # Keep lm_head as FP16
    )

    # Step 4: Replace MoE experts with quantized versions
    print("  Replacing MoE experts with quantized versions...")
    replace_glm4_moe_experts(model)

    # Step 5+: Load quantized weights from safetensors
    if (quantized_path / "model.safetensors").exists() or (
        quantized_path / "model.safetensors.index.json"
    ).exists():
        print("  Loading quantized weights...")
        loader = MMFP4ModelLoader(quantized_path)
        loaded = _apply_quantized_weights(model, loader, device)
        print(f"  Loaded {loaded} quantized weight tensors")
        print("  Loading MoE expert weights...")
        count = _apply_moe_expert_weights(model, loader, device)
        print(f"  Loaded {count} MoE expert layers")

        # Load FP16 weights for lm_head and embed_tokens (not quantized)
        print("  Loading FP16 lm_head and embed_tokens...")
        try:
            lm_head_weight = loader.load_tensor("lm_head.weight")
            if lm_head_weight is not None:
                model.lm_head.weight.data.copy_(lm_head_weight.to(device))
                print(f"    Loaded lm_head.weight: {lm_head_weight.shape}")
        except Exception as e:
            print(f"    lm_head load failed: {e}")

        try:
            embed_weight = loader.load_tensor("model.embed_tokens.weight")
            if embed_weight is not None:
                model.model.embed_tokens.weight.data.copy_(
                    embed_weight.to(device))
                print(f"    Loaded embed_tokens.weight: {embed_weight.shape}")
        except Exception as e:
            print(f"    embed_tokens load failed: {e}")

        # Load FP16/BF16 weights that were NOT quantized:
        # - Layernorms (input_layernorm, post_attention_layernorm, q_a_layernorm, kv_a_layernorm)
        # - Final model norm
        # - Rotary embeddings (if any)
        print("  Loading non-quantized FP16/BF16 weights...")
        fp16_loaded = 0
        all_keys = set(loader._tensor_to_shard.keys())

        # Load final model norm
        # NOTE: The MMFP4 checkpoint has corrupted index - model.norm.weight
        # points to wrong shard. Load from separate file if it exists.
        norm_loaded = False
        missing_norm_path = quantized_path / "missing_norm.safetensors"
        if missing_norm_path.exists():
            from safetensors import safe_open
            with safe_open(missing_norm_path, framework="pt") as sf:
                if "model.norm.weight" in sf.keys():
                    norm_w = sf.get_tensor("model.norm.weight")
                    model.model.norm.weight.data.copy_(norm_w.to(device))
                    fp16_loaded += 1
                    norm_loaded = True

        if not norm_loaded and "model.norm.weight" in all_keys:
            try:
                norm_w = loader.load_tensor("model.norm.weight")
                model.model.norm.weight.data.copy_(norm_w.to(device))
                fp16_loaded += 1
                norm_loaded = True
            except Exception:
                pass  # Will fall through to init

        if not norm_loaded:
            # RMSNorm default is ones (but this will hurt quality)
            model.model.norm.weight.data.fill_(1.0)
            print(
                "    WARNING: model.norm.weight missing, initialized to 1.0")

        # Load per-layer layernorm weights
        for i in range(len(model.model.layers)):
            layer = model.model.layers[i]

            # input_layernorm
            key = f"model.layers.{i}.input_layernorm.weight"
            if key in all_keys:
                w = loader.load_tensor(key)
                layer.input_layernorm.weight.data.copy_(w.to(device))
                fp16_loaded += 1

            # post_attention_layernorm
            key = f"model.layers.{i}.post_attention_layernorm.weight"
            if key in all_keys:
                w = loader.load_tensor(key)
                layer.post_attention_layernorm.weight.data.copy_(w.to(device))
                fp16_loaded += 1

            # Attention layernorms (q_a_layernorm, kv_a_layernorm)
            attn = layer.self_attn
            for ln_name in ["q_a_layernorm", "kv_a_layernorm"]:
                key = f"model.layers.{i}.self_attn.{ln_name}.weight"
                if key in all_keys and hasattr(attn, ln_name):
                    w = loader.load_tensor(key)
                    getattr(attn, ln_name).weight.data.copy_(w.to(device))
                    fp16_loaded += 1

            # MoE router/gate weights (for MoE layers)
            mlp = layer.mlp
            if hasattr(mlp, 'gate') and hasattr(mlp.gate, 'weight'):
                # Router weight
                key = f"model.layers.{i}.mlp.gate.weight"
                if key in all_keys:
                    w = loader.load_tensor(key)
                    mlp.gate.weight.data.copy_(w.to(device))
                    fp16_loaded += 1

                # e_score_correction_bias (if present)
                key = f"model.layers.{i}.mlp.gate.e_score_correction_bias"
                if key in all_keys and hasattr(mlp.gate, 'e_score_correction_bias'):
                    w = loader.load_tensor(key)
                    mlp.gate.e_score_correction_bias.data.copy_(w.to(device))
                    fp16_loaded += 1

            # Shared experts (MoE layers have shared_experts with gate_proj, up_proj, down_proj)
            if hasattr(mlp, 'shared_experts'):
                shared = mlp.shared_experts
                for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                    proj = getattr(shared, proj_name, None)
                    if proj is not None:
                        # Shared experts may be regular Linear or quantized
                        key_w = f"model.layers.{i}.mlp.shared_experts.{proj_name}.weight"
                        key_s = f"model.layers.{i}.mlp.shared_experts.{proj_name}.scales"
                        if key_w in all_keys and hasattr(proj, 'weight'):
                            # Check if it's a quantized linear
                            if hasattr(proj, 'scales') and key_s in all_keys:
                                # Quantized - should be loaded by _apply_quantized_weights
                                pass
                            else:
                                # Regular FP16 linear
                                w = loader.load_tensor(key_w)
                                proj.weight.data.copy_(w.to(device))
                                fp16_loaded += 1

        print(f"    Loaded {fp16_loaded} non-quantized weight tensors")

        # GLM 4.7 Flash MTP (Multi-Token Prediction) auxiliary head.
        # Prefer extraction from loaded model modules, then fallback to
        # known checkpoint keys when model class omits MTP modules.
        print("  Loading GLM MTP head (if present)...")
        mtp_head = GLMMTPHead.from_model(model)
        if mtp_head is None:
            mtp_head = _load_mtp_head_from_checkpoint(model, config, loader)
        model.mtp_head = mtp_head
        if mtp_head is not None:
            print(
                f"    Loaded MTP head with {mtp_head.num_tokens} prediction head(s)"
            )
        else:
            print("    No MTP head found in model/checkpoint")
    else:
        raise ValueError(
            f"No quantized weights found at {quantized_path}. "
            "Expected model.safetensors or model.safetensors.index.json"
        )

    # Move model to device
    model = model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)

    # Pre-dequantize all expert weights into GPU float16 tables. This eliminates
    # the per-layer GPU→CPU sync from .tolist() during single-token decode,
    # enabling fully async GPU execution for expert computation.
    try:
        from .glm4_moe_experts import QuantizedGlm4MoEExperts
        expert_layers = [
            m for m in model.modules()
            if isinstance(m, QuantizedGlm4MoEExperts)
        ]
        if expert_layers:
            print(
                f"  Found {len(expert_layers)} MoE expert layers (sync-free decode enabled).")
    except Exception as e:
        print(f"  Warning: could not inspect expert layers: {e}")

    model.eval()
    return model, tokenizer
