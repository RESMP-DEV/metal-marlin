from dataclasses import dataclass


@dataclass
class TDTConfig:
    vocab_size: int = 1024  # BPE vocab
    joint_hidden_size: int = 640
    predictor_hidden_size: int = 640
    predictor_num_layers: int = 2
    max_duration: int = 4  # TDT-specific: max frames to skip
    # Encoder output dim (from ConformerConfig)
    encoder_hidden_size: int = 512
