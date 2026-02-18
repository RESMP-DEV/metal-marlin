# MMFP4 Inference

Metal Marlin supports MMFP4-quantized GLM-4.7-Flash with fused MoE kernels.

## Quick Start

```bash
# Generate text
metal-marlin mmfp4 generate -m models/GLM-4.7-Flash-Marlin-MMFP4 -p "Hello, world!"

# Start API server
metal-marlin mmfp4 serve -m models/GLM-4.7-Flash-Marlin-MMFP4 --port 8000
```

## Python API

```python
from metal_marlin.inference_metal import MetalQuantizedLinear
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load with MMFP4 weights
model = AutoModelForCausalLM.from_pretrained(
    "models/GLM-4.7-Flash-Marlin-MMFP4",
    device_map="mps"
)

# Generate
tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash")
input_ids = tokenizer("Hello!", return_tensors="pt").input_ids.to("mps")
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
```

## More Details

- [MMFP4 Inference Guide](../guides/mmfp4_inference.md) - Detailed usage
- [MMFP4 Troubleshooting](../guides/mmfp4_troubleshooting.md) - Common issues
