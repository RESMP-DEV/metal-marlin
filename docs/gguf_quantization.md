# GGUF quantization compatibility

Metal Marlin can ingest llama.cpp GGUF weights and convert them to the
Marlin FP4 format used by the fused Metal kernels. Conversion happens by
reading GGUF block-quantized weights, dequantizing to FP32, and then
re-quantizing to Marlin FP4 with per-group scales.

Supported GGUF block quants:
- K-quants: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
- I-quants: IQ2_XXS, IQ3_XXS, IQ4_NL, IQ4_XS
- Legacy: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
- MXFP4: GGML_TYPE_MXFP4 (direct conversion)

Note on K_M variants:
- GGUF encodes Q4_K_M and Q5_K_M as Q4_K/Q5_K tensor types with a
  file_type metadata hint. Conversion treats them as their base K-quant types.

Notes on IQ2/IQ3:
- llama.cpp uses importance-guided codebooks (imatrix) at quantization time.
- GGUF conversion dequantizes the stored weights; it does not recreate the
  original importance matrix or its search process.

Approximate quality-size tradeoffs (relative to FP16):
- IQ2_XXS: ~2.1 bpw, +0.8-1.2 PPL
- IQ3_XXS: ~3.1 bpw, +0.3-0.5 PPL
- IQ4_XS:  ~4.3 bpw, +0.05-0.10 PPL
- Q4_K_M:  ~4.8 bpw, +0.05-0.15 PPL
- Q5_K_M:  ~5.7 bpw, +0.01-0.05 PPL
- Q6_K:    ~6.6 bpw, negligible PPL loss

These values are rough guidelines; real numbers depend on model size and
calibration data.
