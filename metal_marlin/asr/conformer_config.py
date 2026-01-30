from dataclasses import dataclass


@dataclass
class ConformerConfig:
    num_layers: int = 17
    hidden_size: int = 512
    num_attention_heads: int = 8
    ffn_intermediate_size: int = 2048
    conv_kernel_size: int = 31
    dropout: float = 0.1
    # Mel spectrogram params
    n_mels: int = 80
    sample_rate: int = 16000
    # Subsampling
    subsampling_factor: int = 4
