# Integration Guide

## MLX Integration

### Basic Usage

```python
from metal_marlin import MarlinLinear, quantize_model
import mlx.nn as nn

# Option 1: Replace individual layers
model.fc = MarlinLinear.from_linear(model.fc, quant_type="fp4")

# Option 2: Quantize entire model
quantize_model(model, quant_type="fp4", group_size=128)
```

### Custom Model

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Use MarlinLinear directly
        self.linear1 = MarlinLinear(4096, 4096, quant_type="fp4")
        self.linear2 = MarlinLinear(4096, 11008, quant_type="fp4")

    def __call__(self, x):
        x = mx.gelu(self.linear1(x))
        return self.linear2(x)
```

## llama.cpp Integration

### Patch and Build

```bash
cd llama.cpp
git apply /path/to/metal-marlin/ggml_integration/llama-cpp-marlin.patch

mkdir build && cd build
cmake .. -DLLAMA_METAL=ON -DLLAMA_MARLIN=ON
make -j
```

### Runtime Usage

```bash
# Convert weights to Marlin format
python -m metal_marlin.convert --input model.gguf --output model.marlin.gguf

# Run with Marlin acceleration
./main -m model.marlin.gguf --marlin
```

## Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM
from metal_marlin import quantize_model

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Convert to MLX
import mlx.utils
mlx_model = mlx.utils.convert_from_torch(model)

# Quantize
quantize_model(mlx_model, quant_type="fp4")
```

## Saving and Loading Quantized Models

```python
# Save
from metal_marlin.io import save_quantized_model
save_quantized_model(model, "model_fp4.safetensors")

# Load
from metal_marlin.io import load_quantized_model
model = load_quantized_model("model_fp4.safetensors")
```
