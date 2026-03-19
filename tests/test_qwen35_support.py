from __future__ import annotations

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from metal_marlin.hf_loader import ModelConfig, load_layer_weights
from metal_marlin.mr_gptq import MRGPTQQuantizer, QuantizationFormat


def test_model_config_parses_nested_text_config() -> None:
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
    assert cfg.max_position_embeddings == 262144
    assert cfg.vocab_size == 248320
    assert cfg.rope_theta == 10_000_000


def test_load_layer_weights_supports_language_model_prefix(tmp_path) -> None:
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
