from __future__ import annotations
import logging

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from metal_marlin.hf_loader import ModelConfig, load_layer_weights
from metal_marlin.mr_gptq import MRGPTQQuantizer, QuantizationFormat



logger = logging.getLogger(__name__)

def test_model_config_parses_nested_text_config() -> None:
    logger.info("running test_model_config_parses_nested_text_config")
    cfg = ModelConfig.from_dict(
        {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 3072,
                "num_hidden_layers": 48,
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "intermediate_size": 3072,
                "moe_intermediate_size": 1024,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "shared_expert_intermediate_size": 1024,
                "max_position_embeddings": 262144,
                "vocab_size": 248320,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "rope_theta": 10_000_000,
                    "rope_type": "default",
                },
            },
        }
    )

    assert cfg.model_type == "qwen3_5_moe_text"
    assert cfg.hidden_size == 3072
    assert cfg.num_hidden_layers == 48
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 2
    assert cfg.num_experts == 256
    assert cfg.num_experts_per_tok == 8
    assert cfg.shared_expert_intermediate_size == 1024
    assert cfg.max_position_embeddings == 262144
    assert cfg.vocab_size == 248320
    assert cfg.rope_theta == 10_000_000


def test_model_config_qwen36_official_config() -> None:
    """Test parsing official Qwen3.6-35B-A3B style config with nested text_config."""
    logger.info("running test_model_config_qwen36_official_config")
    cfg = ModelConfig.from_dict(
        {
            "model_type": "qwen3_6_moe",
            "architectures": ["Qwen3_6MoEForCausalLM"],
            "text_config": {
                "model_type": "qwen3_6_moe_text",
                "hidden_size": 4096,
                "num_hidden_layers": 48,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "intermediate_size": 5632,
                "moe_intermediate_size": 1408,
                "num_local_experts": 256,
                "num_experts_per_tok": 8,
                "shared_expert_intermediate_size": 5632,
                "max_position_embeddings": 262144,
                "vocab_size": 256000,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10_000_000,
                # Qwen3.6 specific fields
                "layer_types": ["full_attention"] * 40 + ["hybrid_attention"] * 8,
                "full_attention_interval": [0, 1, 2, 3],
                "use_delta": True,
                "delta_intermediate_size": 1024,
                # MTP fields
                "num_mtp_heads": 1,
                "num_nextn_predict_layers": 1,
            },
        }
    )

    # Verify model_type comes from text_config, not outer wrapper
    assert cfg.model_type == "qwen3_6_moe_text"

    # Basic architecture
    assert cfg.hidden_size == 4096
    assert cfg.num_hidden_layers == 48
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 4
    assert cfg.intermediate_size == 5632
    assert cfg.vocab_size == 256000
    assert cfg.max_position_embeddings == 262144

    # MoE config
    assert cfg.num_experts == 256
    assert cfg.num_experts_per_tok == 8
    assert cfg.moe_intermediate_size == 1408
    assert cfg.shared_expert_intermediate_size == 5632

    # Qwen3.6 layer_types
    assert cfg.layer_types is not None
    assert len(cfg.layer_types) == 48
    assert cfg.layer_types[0] == "full_attention"
    assert cfg.layer_types[40] == "hybrid_attention"

    # Qwen3.6 full_attention_interval
    assert cfg.full_attention_interval == [0, 1, 2, 3]

    # DeltaNet support
    assert cfg.use_delta is True
    assert cfg.delta_intermediate_size == 1024
    assert cfg.has_delta is True

    # MTP config
    assert cfg.num_mtp_heads == 1
    assert cfg.has_mtp is True


def test_model_config_qwen36_dense_fallback() -> None:
    """Test Qwen3.6 dense model config without MoE fields."""
    logger.info("running test_model_config_qwen36_dense_fallback")
    cfg = ModelConfig.from_dict(
        {
            "model_type": "qwen3_6",
            "text_config": {
                "model_type": "qwen3_6_text",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "intermediate_size": 11008,
                "max_position_embeddings": 32768,
                "vocab_size": 151936,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1_000_000,
                # No MoE fields - should be dense model
            },
        }
    )

    assert cfg.model_type == "qwen3_6_text"
    assert cfg.is_moe is False
    assert cfg.num_experts is None
    assert cfg.num_experts_per_tok is None
    assert cfg.shared_expert_intermediate_size is None


def test_model_config_preserves_outer_fields_when_no_text_config() -> None:
    """Test that outer fields are used when text_config is absent."""
    logger.info("running test_model_config_preserves_outer_fields_when_no_text_config")
    cfg = ModelConfig.from_dict(
        {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
        }
    )

    assert cfg.model_type == "llama"
    assert cfg.hidden_size == 4096
    assert cfg.num_hidden_layers == 32
    assert cfg.is_moe is False


def test_model_config_shared_expert_alternate_names() -> None:
    """Test shared_expert_intermediate_size with alternate field name."""
    logger.info("running test_model_config_shared_expert_alternate_names")
    cfg = ModelConfig.from_dict(
        {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "intermediate_size": 2048,
                "moe_intermediate_size": 512,
                "num_local_experts": 64,
                "num_experts_per_tok": 4,
                # Use alternate field name
                "shared_expert_ffn_hidden_size": 2048,
                "max_position_embeddings": 32768,
                "vocab_size": 151936,
                "rms_norm_eps": 1e-6,
            },
        }
    )

    assert cfg.shared_expert_intermediate_size == 2048


def test_model_config_layer_types_normalization() -> None:
    """Test that layer_types is properly normalized to list of strings."""
    # Test with mixed types (int and str)
    logger.info("running test_model_config_layer_types_normalization")
    cfg = ModelConfig.from_dict(
        {
            "model_type": "qwen3_6_moe",
            "text_config": {
                "model_type": "qwen3_6_moe_text",
                "hidden_size": 4096,
                "num_hidden_layers": 4,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "intermediate_size": 5632,
                "vocab_size": 256000,
                "max_position_embeddings": 32768,
                "layer_types": ["full_attention", "hybrid_attention", "full_attention", "delta"],
            },
        }
    )

    assert cfg.layer_types == ["full_attention", "hybrid_attention", "full_attention", "delta"]


def test_model_config_full_attention_interval_int() -> None:
    """Test that full_attention_interval as int is normalized to list."""
    logger.info("running test_model_config_full_attention_interval_int")
    cfg = ModelConfig.from_dict(
        {
            "model_type": "qwen3_6_moe",
            "text_config": {
                "model_type": "qwen3_6_moe_text",
                "hidden_size": 4096,
                "num_hidden_layers": 48,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "intermediate_size": 5632,
                "vocab_size": 256000,
                "max_position_embeddings": 32768,
                "full_attention_interval": 4,  # Single int
            },
        }
    )

    assert cfg.full_attention_interval == [4]


def test_model_config_moe_intermediate_size_alternate_names() -> None:
    """Test moe_intermediate_size with alternate field name expert_intermediate_size."""
    logger.info("running test_model_config_moe_intermediate_size_alternate_names")
    cfg = ModelConfig.from_dict(
        {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "intermediate_size": 2048,
                "num_local_experts": 64,
                "num_experts_per_tok": 4,
                "shared_expert_intermediate_size": 2048,
                # Use alternate field name
                "expert_intermediate_size": 512,
                "max_position_embeddings": 32768,
                "vocab_size": 151936,
                "rms_norm_eps": 1e-6,
            },
        }
    )

    assert cfg.moe_intermediate_size == 512


def test_load_layer_weights_supports_language_model_prefix(tmp_path) -> None:
    logger.info("running test_load_layer_weights_supports_language_model_prefix")
    tensor_name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    save_file(
        {
            tensor_name: torch.randn(2, 16, 16, dtype=torch.float32),
        },
        str(shard_path),
    )

    weight_map = {
        tensor_name: shard_path.name,
    }
    loaded = load_layer_weights(tmp_path, layer_idx=0, weight_map=weight_map)
    assert tensor_name in loaded
    assert loaded[tensor_name].shape == (2, 16, 16)


def test_mrgptq_quantizes_stacked_expert_tensor(tmp_path) -> None:
    logger.info("running test_mrgptq_quantizes_stacked_expert_tensor")
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "out"
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensor_name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    save_file(
        {
            tensor_name: torch.randn(2, 16, 16, dtype=torch.float32),
            "model.language_model.embed_tokens.weight": torch.randn(
                32, 16, dtype=torch.float32
            ),
        },
        str(model_dir / "model.safetensors"),
    )

    quantizer = MRGPTQQuantizer(
        bits=4,
        format=QuantizationFormat.FP4,
        group_size=8,
        use_hadamard=False,
    )
    report = quantizer.quantize_model(model_dir, output_path=out_dir, verbose=False)
    assert report.quantized_layers >= 1

    with safe_open(str(out_dir / "model.safetensors"), framework="pt") as sf:
        assert tensor_name in sf.keys()
        assert f"{tensor_name}.scales" in sf.keys()
        assert f"{tensor_name}.group_size" in sf.keys()

        packed = sf.get_tensor(tensor_name).cpu().numpy()
        scales = sf.get_tensor(f"{tensor_name}.scales").cpu().numpy()
        group_size = sf.get_tensor(f"{tensor_name}.group_size").cpu().numpy()

    assert packed.dtype == np.uint32
    assert packed.shape[0] == 2  # num experts
    assert scales.dtype == np.float16
    assert scales.shape[0] == 2  # num experts
    assert int(group_size[0]) == 8
