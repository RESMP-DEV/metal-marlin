"""Tests for Qwen3.6 nested text-config parsing in ``ModelConfig``."""

from metal_marlin.hf_loader import ModelConfig


def test_qwen36_config():
    """Qwen3.6 nested text configs should normalize cadence fields consistently."""

    cfg = ModelConfig.from_dict(
        {
            "model_type": "qwen3_6_moe",
            "text_config": {
                "model_type": "qwen3_6_moe_text",
                "layer_types": ["dense", "moe", "dan"],
                "full_attention_interval": 2,
                "use_delta": True,
                "delta_intermediate_size": 2048,
                "num_mtp_heads": 1,
                "mtp_num_hidden_layers": 1,
                "mtp_expansion_factor": 2,
            },
        }
    )

    assert cfg.model_type == "qwen3_6_moe_text"
    assert cfg.layer_types == ["dense", "moe", "dan"]
    assert cfg.full_attention_interval == [2]
    assert cfg.use_delta is True
    assert cfg.delta_intermediate_size == 2048
    assert cfg.num_mtp_heads == 1
    assert cfg.mtp_num_hidden_layers == 1
    assert cfg.mtp_expansion_factor == 2
