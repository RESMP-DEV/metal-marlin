from dataclasses import dataclass


@dataclass
class ParakeetQuantPolicy:
    # Layers to quantize with Metal Marlin (GEMM-heavy)
    quantize_attention_qkv: bool = True
    quantize_attention_out: bool = True
    quantize_ffn: bool = True
    quantize_joint: bool = True
    quantize_predictor_lstm: bool = False  # LSTM has special handling

    # Layers to keep FP16 (for ANE offload)
    keep_fp16_conv: bool = True
    keep_fp16_embedding: bool = True
    keep_fp16_layernorm: bool = True

    # Quantization params
    bits: int = 4  # 4-bit default
    group_size: int = 128
    format: str = "fp4"  # fp4, int4, trellis
