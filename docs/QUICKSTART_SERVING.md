# GLM-4.7-Flash Serving Quick Reference

**35 tok/s on M4 Max • OpenAI-Compatible API • Zero-GPU Syncs**

## Quick Start

```bash
# 1. Start server
cd /Users/kearm/AlphaHENG/contrib/metal_marlin
uv run python scripts/serve_glm47.py --model-path ./models/glm47-flash-mmfp4

# 2. Test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-4.7-flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## Client Code

### Python (OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=50
)
print(response.choices[0].message.content)
```

### cURL
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-4.7-flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## Performance

| Metric | Value |
|--------|-------|
| Throughput | 35 tok/s |
| Latency | 28.6 ms/step |
| Batch Size | 32 concurrent |
| Memory | 12.4 GB |

## Endpoints

- `/health` - Health check
- `/v1/models` - Model list
- `/v1/chat/completions` - Chat (OpenAI compatible)
- `/v1/completions` - Text completion
- `/metrics` - Performance metrics

## Full Documentation

See [serving_guide.md](serving_guide.md) for complete details.
