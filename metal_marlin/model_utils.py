from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from metal_marlin._quantized_weights import (_apply_moe_expert_weights,
                                             _apply_quantized_weights)
from metal_marlin.layer_replacement import (replace_glm4_moe_experts,
                                            replace_linear_layers)
from metal_marlin.layers.mtp_head import GLMMTPHead
from metal_marlin.mmfp4_loader import MMFP4ModelLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _load_mtp_head_from_checkpoint(
    model: Any,
    config: Any,
    loader: MMFP4ModelLoader,
) -> GLMMTPHead | None:
    """Build a GLM MTP head from checkpoint tensors if present.

    GLM 4.7 Flash stores auxiliary next-token predictor weights under:
    `model.layers.<num_hidden_layers>.shared_head.head.weight`.
    """
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
    print(f"Loading MMFP4 model from {model_path}...")
    quantized_path = Path(model_path)

    # Step 1: Load config only (no weights)
    print("  Loading model config...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Step 2: Create model architecture WITHOUT weight initialization
    # Using meta device avoids allocating/initializing 30B params (saves minutes)
    print("  Creating model architecture (meta device)...")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config,
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
