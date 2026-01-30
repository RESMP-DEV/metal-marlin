# Distributed and GPU-Accelerated Quantization

Metal Marlin supports GPU-accelerated GPTQ quantization with multiple backend options and distributed processing for maximum throughput on large models.

## Overview

Quantizing a 30B MoE model like GLM-4.7-Flash with full GPTQ + Hessian calibration takes ~12 hours on a single Mac with NumPy. With GPU acceleration and distributed processing, this can be reduced to **~30 minutes**.

| Backend | Time (30B model) | Hardware |
|---------|-----------------|----------|
| NumPy (CPU) | ~12 hours | Any |
| Metal MPS | ~2.5 hours | Apple Silicon (M1/M2/M3/M4) |
| Local CUDA | ~20 minutes | NVIDIA GPU (A100, H100, etc.) |
| Remote CUDA | ~30 minutes | Remote NVIDIA server |
| Distributed (2 workers) | ~15 minutes | Multiple machines |

## Quick Start

### Using GPU-Accelerated Backend (Local)

```python
from metal_marlin.mr_gptq import AcceleratedMRGPTQQuantizer
from metal_marlin.calibration import CalibrationDatasetLoader

# Auto-detect best available backend (MPS on Mac, CUDA on Linux)
quantizer = AcceleratedMRGPTQQuantizer.create(backend="auto")

# Or explicitly use MPS (Apple Silicon)
quantizer = AcceleratedMRGPTQQuantizer.create(backend="mps")

# Load calibration data
calibration = CalibrationDatasetLoader.v3()

# Quantize model
report = quantizer.quantize_model_with_calibration(
    model_path="zai-org/GLM-4.7-Flash",
    calibration=calibration,
    output_path="./glm47_fp4",
    tokenizer=tokenizer,
    verbose=True,
)
```

### Using Remote CUDA Server

If you have access to a machine with NVIDIA GPUs, you can offload quantization:

**On the CUDA server:**
```bash
# Install metal_marlin on CUDA machine
pip install torch transformers safetensors

# Start GPTQ server
python -m metal_marlin.gptq_accelerated server --port 5556
```

**On your Mac:**
```python
from metal_marlin.mr_gptq import AcceleratedMRGPTQQuantizer

quantizer = AcceleratedMRGPTQQuantizer.create(
    backend="remote_cuda",
    remote_address="cuda-server.local:5556"
)

report = quantizer.quantize_model_with_calibration(
    model_path="path/to/model",
    calibration=calibration,
    output_path="./output",
)
```

### Distributed Quantization (Multiple Workers)

For maximum throughput, use multiple workers:

**Start CUDA workers (on remote machines):**
```bash
# Machine 1
python -m metal_marlin.gptq_accelerated server --port 5556

# Machine 2
python -m metal_marlin.gptq_accelerated server --port 5556
```

**Run distributed quantization (on coordinator):**
```python
from metal_marlin.distributed_gptq import DistributedQuantizer

quantizer = DistributedQuantizer(
    workers=["cuda-server-1:5556", "cuda-server-2:5556"],
    local_workers=2,  # Also use local MPS
)

import asyncio

report = asyncio.run(quantizer.quantize_model(
    model_path="path/to/model",
    calibration=calibration,
    output_path="./output",
))
```

Or via CLI:
```bash
python -m metal_marlin.distributed_gptq model/ output/ \
    --workers cuda-server-1:5556 cuda-server-2:5556 \
    --local-workers 2
```

## Backend Details

### NumPy Backend

- Pure Python implementation
- Works everywhere
- Slowest option, ~12 hours for 30B model
- Good for small models or debugging

### Metal MPS Backend

- Uses PyTorch MPS for Apple Silicon GPUs
- 5-10x faster than NumPy on M1/M2/M3/M4
- Optimal for local Mac quantization
- Memory-efficient streaming

### CUDA Backend

- Uses PyTorch CUDA for NVIDIA GPUs
- Fastest option with local GPU
- Uses cuSOLVER for Cholesky decomposition
- Supports multi-GPU with device selection

### Remote CUDA Backend

- Offloads computation to remote machine
- Network overhead is minimal (~10% slower than local)
- Great for Macs without NVIDIA GPUs
- Secure binary protocol (consider VPN for production)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Coordinator (Mac)                           │
│   - Loads model weights (safetensors)                          │
│   - Runs calibration forward passes                            │
│   - Collects Hessians locally                                  │
│   - Distributes layer quantization                             │
│   - Aggregates results                                         │
└─────────────────────────────────────────────────────────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │  Worker 1     │   │  Worker 2     │   │  Worker 3     │
    │  (CUDA A100)  │   │  (CUDA H100)  │   │  (Mac MPS)    │
    │               │   │               │   │               │
    │  - Receives   │   │  - Receives   │   │  - Receives   │
    │    weights    │   │    weights    │   │    weights    │
    │  - Receives   │   │  - Receives   │   │  - Receives   │
    │    Hessian    │   │    Hessian    │   │    Hessian    │
    │  - Computes   │   │  - Computes   │   │  - Computes   │
    │    H^{-1}     │   │    H^{-1}     │   │    H^{-1}     │
    │  - Quantizes  │   │  - Quantizes  │   │  - Quantizes  │
    │  - Returns    │   │  - Returns    │   │  - Returns    │
    │    result     │   │    result     │   │    result     │
    └───────────────┘   └───────────────┘   └───────────────┘
```

## Memory Optimization

Quantization is memory-intensive. Here's how to manage it:

### Streaming Hessian Collection

Instead of storing all activations, we compute Hessians incrementally:

```python
# Memory-efficient Hessian accumulation
# H = 2 * X^T @ X (accumulated over batches)

# For 30B model with hidden_dim=4096:
# Full activations: ~50GB
# Streaming Hessian: ~134MB per layer
```

### Layer-by-Layer Processing

For extremely large models, process one layer at a time:

```python
quantizer = AcceleratedMRGPTQQuantizer.create(backend="mps")

# Process layers sequentially with GPU memory cleanup
for layer_name in layers:
    weights = load_layer_weights(layer_name)
    hessian = hessians[layer_name]
    
    packed, scales, meta = quantizer.quantize_layer_accelerated(
        weights, hessian, layer_name
    )
    
    save_quantized_layer(output_path, layer_name, packed, scales)
    
    # Free GPU memory
    gc.collect()
    torch.mps.empty_cache()
```

### Checkpointing

For long-running jobs, enable checkpointing:

```python
report = quantizer.quantize_model_with_calibration(
    model_path="model/",
    calibration=calibration,
    output_path="output/",
    checkpoint_dir="checkpoints/",  # Resume from here after interruption
    resume=True,
)
```

## Benchmarking

Run the built-in benchmark to compare backends:

```bash
python -m metal_marlin.gptq_accelerated benchmark --size 4096 --samples 1000
```

Example output:
```
Benchmarking GPTQ backends (matrix size: 4096x4096)
Calibration samples: 1000

--- NUMPY Backend ---
Hessian computation: 2.341s
Cholesky inverse: 0.523s
Full quantization: 45.231s
Quantization error: 0.012345

--- MPS Backend ---
Hessian computation: 0.234s
Cholesky inverse: 0.089s
Full quantization: 8.451s
Quantization error: 0.012341

Benchmark complete!
```

## Troubleshooting

### MPS Backend Not Available

```
RuntimeError: MPS backend not available
```

**Solution:** Ensure you're on macOS 12.3+ with Apple Silicon and PyTorch 2.0+.

```bash
pip install torch>=2.0.0
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Remote Connection Refused

```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Solution:** Ensure the GPTQ server is running on the remote machine:

```bash
# On remote machine
python -m metal_marlin.gptq_accelerated server --port 5556

# Test connection
nc -zv cuda-server.local 5556
```

### Out of Memory on GPU

For very large models, reduce batch size or use streaming:

```python
report = quantizer.quantize_model_with_calibration(
    model_path="model/",
    calibration=calibration,
    batch_size=1,  # Reduce batch size
    max_seq_len=1024,  # Reduce sequence length
)
```

### Cholesky Decomposition Fails

If Hessian is near-singular, increase damping:

```python
quantizer = AcceleratedMRGPTQQuantizer.create(
    backend="auto",
    percdamp=0.05,  # Increase from default 0.01
)
```

## API Reference

### `AcceleratedMRGPTQQuantizer`

```python
class AcceleratedMRGPTQQuantizer:
    @classmethod
    def create(
        cls,
        backend: str = "auto",  # auto, numpy, mps, cuda, remote_cuda
        remote_address: str | None = None,
        format: str = "fp4",
        group_size: int = 128,
        use_hadamard: bool = True,
        hadamard_block_size: int = 64,
        actorder: bool = True,
        percdamp: float = 0.01,
    ) -> AcceleratedMRGPTQQuantizer: ...
    
    def quantize_layer_accelerated(
        self,
        weights: NDArray,
        hessian: NDArray,
        layer_name: str = "",
    ) -> tuple[NDArray, NDArray, dict]: ...
    
    def quantize_model_parallel(
        self,
        model_path: str | Path,
        calibration_data: CalibrationDataset,
        output_path: str | Path,
        max_workers: int = 4,
    ) -> QuantizationReport: ...
```

### `DistributedQuantizer`

```python
class DistributedQuantizer:
    def __init__(
        self,
        workers: list[str] | None = None,  # Remote addresses
        local_workers: int = 1,
        local_backend: Backend = Backend.AUTO,
    ): ...
    
    async def quantize_model(
        self,
        model_path: str | Path,
        calibration: CalibrationDataset,
        output_path: str | Path,
    ) -> DistributedQuantizationReport: ...
```

### `RemoteGPTQServer`

```python
class RemoteGPTQServer:
    def __init__(self, port: int = 5556, device_id: int = 0): ...
    def run(self) -> None: ...  # Blocking server loop
```
