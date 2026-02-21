# GLM-4.7-Flash Serving Guide

**End-to-End OpenAI-Compatible Server with MMFP4 Quantization**

This guide walks through serving GLM-4.7-Flash (35 tok/s, 4.9× optimized) with OpenAI-compatible API on Apple Silicon.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Server Architecture](#server-architecture)
4. [Configuration](#configuration)
5. [API Reference](#api-reference)
6. [Client Examples](#client-examples)
7. [Performance Tuning](#performance-tuning)
8. [Monitoring](#monitoring)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

- **Apple Silicon**: M4 Max (or M3/M2 Pro/Max/Ultra)
- **Memory**: 32GB+ unified memory
- **Storage**: 20GB+ for quantized model weights

### Software Stack

```bash
# 1. Ensure Python 3.12 via uv
cd /Users/kearm/AlphaHENG/contrib/metal_marlin
uv sync --extra all

# 2. Verify Metal GPU access
uv run python -c "import torch; print(torch.backends.mps.is_available())"
# Should print: True
```

### Model Weights

Download GLM-4.7-Flash quantized weights (MMFP4 format):

```bash
# Option A: From HuggingFace (when published)
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('zai-org/GLM-4.7-Flash-MMFP4', local_dir='./models/glm47-flash')
"

# Option B: Local conversion from FP16
uv run python scripts/convert_glm47_to_mmfp4.py \
  --model-path zai-org/GLM-4.7-Flash \
  --output-dir ./models/glm47-flash-mmfp4
```

---

## Quick Start

### 1. Start the Server

```bash
# Basic launch (default: localhost:8000)
uv run python -m metal_marlin.serving.server \
  --model-path ./models/glm47-flash-mmfp4 \
  --device mps

# Production launch with tuning
uv run python -m metal_marlin.serving.server \
  --model-path ./models/glm47-flash-mmfp4 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-batch-size 32 \
  --max-wait-ms 10 \
  --max-seq-len 8192
```

### 2. Test the Server

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7-flash",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in 2 sentences"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### 3. Use with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Local server, no auth
)

response = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[
        {"role": "user", "content": "Write a haiku about GPUs"}
    ],
    max_tokens=50,
    temperature=0.7
)

print(response.choices[0].message.content)
```

---

## Server Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI HTTP Server                        │
│  (OpenAI-compatible endpoints: /v1/chat/completions, etc.)  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              MMFP4Server (Request Batching)                  │
│  • RequestBatcher: Groups requests for efficiency           │
│  • KVCacheSharing: Prefix caching with COW                  │
│  • ContinuousBatchingMetrics: Performance tracking          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            BatchScheduler (Continuous Batching)              │
│  • PagedAttention KV cache management                       │
│  • Prefill/decode scheduling                                │
│  • Preemption and swapping                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              ServingEngine (Model Execution)                 │
│  • MMFP4 quantized layers                                   │
│  • Fused MLA attention kernel                               │
│  • Quantized KV cache (int8)                                │
│  • Buffer pool (zero allocation)                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Metal GPU (M4 Max)                          │
│  • 35 tok/s decode throughput                               │
│  • 28.6ms/step latency                                      │
│  • Zero GPU→CPU syncs                                        │
└─────────────────────────────────────────────────────────────┘
```

### Request Flow

1. **HTTP Request** → FastAPI endpoint
2. **Validation** → OpenAI schema check (ChatCompletionRequest)
3. **Queuing** → RequestBatcher groups with other requests (max 32, max 10ms wait)
4. **Scheduling** → BatchScheduler allocates KV cache, schedules prefill/decode
5. **Execution** → ServingEngine runs MMFP4 layers on Metal GPU
6. **Response** → Streaming or batched JSON response

---

## Configuration

### Server Configuration

**Environment Variables:**

```bash
# Model paths
export METAL_MARLIN_MODEL_PATH="./models/glm47-flash-mmfp4"

# Performance tuning
export METAL_MARLIN_MAX_BATCH_SIZE=32
export METAL_MARLIN_MAX_WAIT_MS=10
export METAL_MARLIN_MAX_SEQ_LEN=8192

# Device selection
export METAL_MARLIN_DEVICE=mps  # or "cpu" for testing
```

**Python Configuration:**

```python
from metal_marlin.serving.server import configure
from metal_marlin.serving.engine import EngineConfig
from metal_marlin.serving.mmfp4_server import BatchConfig, SchedulerConfig

# Engine config
engine_config = EngineConfig(
    model_path="./models/glm47-flash-mmfp4",
    device="mps",
    max_seq_len=8192,
    request_timeout=60.0,  # seconds
)

# Batch config
batch_config = BatchConfig(
    max_batch_size=32,
    max_wait_ms=10.0,
    min_batch_size=1,
)

# Scheduler config
scheduler_config = SchedulerConfig(
    max_num_seqs=32,
    max_num_batched_tokens=2048,
    block_size=16,
)

# Configure server
configure(
    model_path="./models/glm47-flash-mmfp4",
    device="mps",
    batch_config=batch_config,
    scheduler_config=scheduler_config,
)
```

### Model Configuration

GLM-4.7-Flash specific settings (auto-loaded from `config.json`):

```python
from metal_marlin.trellis.config import TrellisModelConfig

config = TrellisModelConfig.from_pretrained("./models/glm47-flash-mmfp4")

print(config)
# TrellisModelConfig(
#   hidden_size=4096,
#   num_hidden_layers=47,
#   num_experts=64,
#   num_experts_per_tok=4,
#   kv_lora_rank=512,  # MLA attention
#   vocab_size=154880,
#   ...
# )
```

---

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/v1/models` | GET | List available models |
| `/v1/models/{model_id}` | GET | Model metadata |
| `/v1/chat/completions` | POST | Chat completion (OpenAI SDK compatible) |
| `/v1/completions` | POST | Text completion (legacy) |
| `/metrics` | GET | Performance metrics |

### Chat Completions

**Request Schema:**

```json
{
  "model": "glm-4.7-flash",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "top_p": 1.0,
  "max_tokens": 256,
  "stream": false,
  "stop": ["\\n", "END"],
  "n": 1,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0
}
```

**Response Schema:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1705800000,
  "model": "glm-4.7-flash",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 8,
    "total_tokens": 20
  }
}
```

### Streaming

Set `"stream": true` for token-by-token responses:

```python
stream = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[{"role": "user", "content": "Count to 10"}],
    stream=True,
    max_tokens=50
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**Streaming Response (SSE):**

```
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"content":"1"},"index":0}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"content":","},"index":0}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"content":" 2"},"index":0}]}
```

---

## Client Examples

### cURL

```bash
# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7-flash",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "glm-4.7-flash",
    "messages": [{"role": "user", "content": "Count to 10"}],
    "stream": true
  }'
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Basic chat
response = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[
        {"role": "system", "content": "You are a coding assistant"},
        {"role": "user", "content": "Write a Python function to sort a list"}
    ],
    max_tokens=256,
    temperature=0.7
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[{"role": "user", "content": "Explain quantum entanglement"}],
    stream=True
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

### JavaScript (fetch)

```javascript
const response = await fetch('http://localhost:8000/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'glm-4.7-flash',
    messages: [{ role: 'user', content: 'Hello!' }],
    max_tokens: 50,
    temperature: 0.7
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

### LangChain Integration

```python
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="glm-4.7-flash",
    temperature=0.7
)

response = llm.invoke([HumanMessage(content="Hello!")])
print(response.content)
```

### Multiple Concurrent Requests

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

async def generate(prompt: str, idx: int):
    response = await client.chat.completions.create(
        model="glm-4.7-flash",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50
    )
    return idx, response.choices[0].message.content

async def main():
    prompts = [
        "What is AI?",
        "Explain machine learning",
        "Define neural networks",
        "Describe deep learning",
        "What is NLP?"
    ]
    
    tasks = [generate(p, i) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    
    for idx, content in sorted(results):
        print(f"[{idx}] {content[:100]}...")

asyncio.run(main())
```

**Expected throughput**: 5 concurrent requests × 35 tok/s = 175 tok/s aggregate

---

## Performance Tuning

### Batch Size Optimization

```bash
# Low latency (single user)
--max-batch-size 1 --max-wait-ms 0

# Balanced (10-20 users)
--max-batch-size 16 --max-wait-ms 10

# High throughput (50+ users)
--max-batch-size 32 --max-wait-ms 20
```

### KV Cache Tuning

```python
# For long conversations
scheduler_config = SchedulerConfig(
    block_size=16,  # tokens per cache block
    max_num_batched_tokens=4096,  # KV cache budget
)

# For short queries
scheduler_config = SchedulerConfig(
    block_size=16,
    max_num_batched_tokens=1024,
)
```

### Memory Optimization

```bash
# Reduce memory for long contexts
--max-seq-len 4096  # instead of 8192

# Enable quantized KV cache (int8)
export METAL_MARLIN_QUANTIZED_KV=true
```

### Benchmarking

```bash
# Run performance benchmark
cd /Users/kearm/AlphaHENG/contrib/metal_marlin
uv run python benchmarks/comprehensive_benchmark.py

# Expected output:
# Throughput: 35.2 tok/s
# TTFT (prompt=256): 45.3ms
# TPOT (decode): 28.4ms
# Memory: 12.4 GB
```

---

## Monitoring

### Metrics Endpoint

```bash
curl http://localhost:8000/metrics
```

**Response:**

```json
{
  "throughput_tok_sec": 35.2,
  "avg_latency_ms": 28.6,
  "queue_depth": 3,
  "active_requests": 12,
  "kv_cache_utilization": 0.72,
  "prefill_tokens_total": 15420,
  "decode_tokens_total": 8930,
  "requests_completed": 145,
  "preemptions": 2
}
```

### Dashboard

```bash
# Real-time TUI dashboard
cd /Users/kearm/AlphaHENG/contrib/metal_marlin
uv run python scripts/monitor_server.py
```

**Dashboard Display:**

```
┌──────────────────────────────────────────────┐
│  GLM-4.7-Flash Server Dashboard              │
├──────────────────────────────────────────────┤
│  Throughput: 35.2 tok/s                      │
│  Latency:    28.6 ms/step                    │
│  Queue:      3 requests                      │
│  Active:     12 concurrent                   │
│  KV Cache:   72% utilized                    │
├──────────────────────────────────────────────┤
│  Requests: 145 completed, 0 failed           │
│  Tokens:    15.4K prefill, 8.9K decode       │
└──────────────────────────────────────────────┘
```

### Logging

```python
import logging

# Enable detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Server logs
# 2026-02-20 12:34:56 - metal_marlin.serving.server - INFO - Request queued: req-abc123
# 2026-02-20 12:34:56 - metal_marlin.serving.mmfp4_server - INFO - Batch: 8 requests, 10ms wait
# 2026-02-20 12:34:56 - metal_marlin.serving.engine - INFO - Prefill: 256 tokens, 15.2ms
# 2026-02-20 12:34:56 - metal_marlin.serving.engine - INFO - Decode: 35 tok/s, 28.6ms/step
```

---

## Troubleshooting

### Common Issues

#### 1. Model Not Found

```bash
# Error
curl: (7) Failed to connect to localhost port 8000

# Solution
uv run python -m metal_marlin.serving.server \
  --model-path ./models/glm47-flash-mmfp4 \
  --port 8000
```

#### 2. Out of Memory

```bash
# Error
RuntimeError: MPS backend out of memory

# Solution 1: Reduce batch size
--max-batch-size 8

# Solution 2: Reduce sequence length
--max-seq-len 4096

# Solution 3: Enable quantized KV cache
export METAL_MARLIN_QUANTIZED_KV=true
```

#### 3. Slow First Request

```bash
# Cause: Kernel compilation on first run
# Solution: Warmup request

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-4.7-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

#### 4. Slow Streaming

```bash
# Cause: Small chunks cause network overhead
# Solution: Batch tokens

# Set minimum tokens per chunk
export METAL_MARLIN_STREAM_CHUNK_SIZE=10
```

#### 5. Request Timeout

```python
# Increase timeout
engine_config = EngineConfig(
    model_path="./models/glm47-flash-mmfp4",
    request_timeout=120.0,  # 2 minutes
)
```

### Debug Mode

```bash
# Enable debug logging
export METAL_MARLIN_DEBUG=1

# Verbose server output
uv run python -m metal_marlin.serving.server \
  --model-path ./models/glm47-flash-mmfp4 \
  --log-level DEBUG
```

### Performance Debugging

```bash
# Profile single request
uv run python -c "
from metal_marlin.inference.mmfp4_pipeline import MMFP4Pipeline
import time

pipeline = MMFP4Pipeline.from_pretrained('./models/glm47-flash-mmfp4')

start = time.time()
result = pipeline('Hello', max_tokens=50, temperature=0.7)
elapsed = time.time() - start

print(f'Output: {result}')
print(f'Time: {elapsed:.2f}s')
print(f'Throughput: {50/elapsed:.1f} tok/s')
"
```

---

## Production Checklist

- [ ] Download quantized model weights
- [ ] Configure batch size based on workload
- [ ] Set appropriate max sequence length
- [ ] Enable quantized KV cache for memory efficiency
- [ ] Configure monitoring/metrics endpoint
- [ ] Set up request rate limiting
- [ ] Test with production-like load
- [ ] Configure graceful shutdown
- [ ] Set up logging/aggregation
- [ ] Document API endpoints for clients

---

## References

- **Model**: GLM-4.7-Flash (zai-org/GLM-4.7-Flash)
- **Tokenizer**: vocab size 154,880
- **Architecture**: 47 layers, 64 experts, top-4 routing, MLA attention
- **Quantization**: MMFP4 (4-bit with mixed precision)
- **Performance**: 35 tok/s on M4 Max

### Related Documentation

- [Optimization Status](optimization_status_final.md) - 35 tok/s achievement details
- [Completed Optimizations](../tasks/completed_optimizations.yaml) - Full task list
- [MMFP4 Quantization](mmfp4_quantization.md) - Quantization details
- [API Reference](api_reference.md) - Full API documentation

---

**Last Updated**: February 2026  
**Version**: 1.0.0
