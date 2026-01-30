# Serving Models

Metal Marlin provides an OpenAI-compatible API server for serving quantized models on Apple Silicon.

## Quick Start

```bash
# Serve a quantized model
metal-marlin serve ./models/qwen3_4b_fp4 --port 8000

# Use with OpenAI SDK
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="qwen3_4b_fp4",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completion |
| `/v1/completions` | POST | Text completion |
| `/health` | GET | Health check |
| `/metrics` | GET | Server metrics |

### Chat Completions

#### Request

```python
{
  "model": "qwen3_4b_fp4",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "temperature": 0.7,
  "max_tokens": 512,
  "stream": false,
  "top_p": 0.9,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stop": null
}
```

**Parameters:**

- `model` (string, required): Model identifier
- `messages` (array, required): Array of message objects with `role` and `content`
- `temperature` (number, optional, default 0.7): Sampling temperature (0.0-2.0)
- `max_tokens` (integer, optional, default 512): Maximum tokens to generate
- `stream` (boolean, optional, default false): Enable streaming responses
- `top_p` (number, optional, default 0.9): Nucleus sampling threshold
- `frequency_penalty` (number, optional, default 0.0): Frequency penalty (-2.0 to 2.0)
- `presence_penalty` (number, optional, default 0.0): Presence penalty (-2.0 to 2.0)
- `stop` (string|array, optional): Stop sequences

#### Response

```python
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1699012345,
  "model": "qwen3_4b_fp4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'm doing well, thank you for asking!"
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

### Text Completions

#### Request

```python
{
  "model": "qwen3_4b_fp4",
  "prompt": "The quick brown fox",
  "temperature": 0.7,
  "max_tokens": 100,
  "stream": false
}
```

#### Response

```python
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1699012345,
  "model": "qwen3_4b_fp4",
  "choices": [
    {
      "index": 0,
      "text": " jumps over the lazy dog.",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 6,
    "total_tokens": 10
  }
}
```

### Models Endpoint

#### Request

```bash
curl http://localhost:8000/v1/models
```

#### Response

```python
{
  "object": "list",
  "data": [
    {
      "id": "qwen3_4b_fp4",
      "object": "model",
      "created": 1699012345,
      "owned_by": "metal-marlin"
    }
  ]
}
```

### Health Check

```bash
curl http://localhost:8000/health
```

Response: `{"status": "ok", "model_loaded": true}`

## Attention Mode

The server supports two attention modes:

### Non-Paged Attention (Current Default)

The server currently uses **non-paged attention** for maximum compatibility:

- Each request runs sequentially through the model pipeline
- No KV cache sharing between requests
- Best for single-user or low-concurrency scenarios
- Simple and reliable

**To use:** This is the default, no configuration needed.

### Paged Attention (CLI Enabled)

Enable continuous batching with shared KV cache:

```bash
metal-marlin serve ./models/qwen3_4b_fp4 --enable-batching

# With custom KV cache sizing
metal-marlin serve ./models/qwen3_4b_fp4 \
  --enable-batching \
  --num-kv-blocks 1024 \
  --block-size 32
```

Benefits:
- Higher throughput for concurrent requests
- KV cache block allocation and reuse
- Dynamic batch composition

### Metrics

```bash
curl http://localhost:8000/metrics
```

Response:
```python
{
  "requests_total": 1234,
  "requests_active": 2,
  "tokens_generated_total": 567890,
  "avg_tokens_per_second": 45.2,
  "memory_usage_mb": 2048,
  "gpu_memory_usage_mb": 4096
}
```

## Streaming

Streaming responses use Server-Sent Events (SSE) format.

### Chat Completions Streaming

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
stream = client.chat.completions.create(
    model="qwen3_4b_fp4",
    messages=[{"role": "user", "content": "Tell me a short story."}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### SSE Format

Each SSE event:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1699012345,"model":"qwen3_4b_fp4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1699012345,"model":"qwen3_4b_fp4","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1699012345","model":"qwen3_4b_fp4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## Configuration

### Command Line Options

```bash
metal-marlin serve [MODEL_PATH] [OPTIONS]
```

**Positional Arguments:**

- `MODEL_PATH`: Path to quantized model directory

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `0.0.0.0` | Host address to bind |
| `--port` | `8000` | Port number |
| `--workers` | `1` | Number of worker processes |
| `--max-concurrent-requests` | `10` | Maximum concurrent requests |
| `--max-tokens` | `512` | Default max tokens per request |
| `--timeout` | `120` | Request timeout in seconds |
| `--log-level` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `--enable-metrics` | `true` | Enable metrics endpoint |
| `--enable-batching` | `false` | Enable paged attention with continuous batching |
| `--num-kv-blocks` | `512` | Number of KV cache blocks (with --enable-batching) |
| `--block-size` | `16` | Tokens per KV cache block (with --enable-batching) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `METAL_MARLIN_HOST` | `0.0.0.0` | Host address |
| `METAL_MARLIN_PORT` | `8000` | Port number |
| `METAL_MARLIN_MODEL_PATH` | - | Model path (overrides CLI arg) |
| `METAL_MARLIN_WORKERS` | `1` | Number of workers |
| `METAL_MARLIN_LOG_LEVEL` | `INFO` | Log level |
| `METAL_MARLIN_API_KEY` | - | API key for authentication (optional) |

### Example Configurations

```bash
# Basic production server
metal-marlin serve ./models/qwen3_4b_fp4 \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --max-concurrent-requests 50 \
  --log-level INFO

# Development server with verbose logging
metal-marlin serve ./models/qwen3_4b_fp4 \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1 \
  --log-level DEBUG

# Production with API key authentication
export METAL_MARLIN_API_KEY="your-secret-key"
metal-marlin serve ./models/qwen3_4b_fp4 \
  --workers 4 \
  --max-concurrent-requests 100
```

## Deployment

### systemd Service

Create `/etc/systemd/system/metal-marlin.service`:

```ini
[Unit]
Description=Metal Marlin Server
After=network.target

[Service]
Type=simple
User=marlin
WorkingDirectory=/opt/metal-marlin
Environment="METAL_MARLIN_HOST=0.0.0.0"
Environment="METAL_MARLIN_PORT=8000"
Environment="METAL_MARLIN_WORKERS=4"
ExecStart=/opt/metal-marlin/venv/bin/metal-marlin serve /opt/models/qwen3_4b_fp4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable metal-marlin
sudo systemctl start metal-marlin
sudo systemctl status metal-marlin
```

### Docker

**Dockerfile:**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Metal Marlin
COPY . /app
RUN pip install -e .

# Copy models
COPY ./models /models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["metal-marlin", "serve", "/models/qwen3_4b_fp4", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  metal-marlin:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models:ro
    environment:
      - METAL_MARLIN_HOST=0.0.0.0
      - METAL_MARLIN_PORT=8000
      - METAL_MARLIN_WORKERS=4
      - METAL_MARLIN_LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Nginx reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - metal-marlin
    restart: unless-stopped
```

Build and run:

```bash
docker-compose up -d
docker-compose logs -f metal-marlin
```

### Nginx Reverse Proxy

**nginx.conf:**

```nginx
events {
    worker_connections 1024;
}

http {
    upstream metal_marlin {
        least_conn;
        server metal-marlin:8000;
    }

    server {
        listen 80;
        server_name api.example.com;

        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name api.example.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;

        # Rate limiting
        limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

        location / {
            limit_req zone=api_limit burst=20;
            proxy_pass http://metal_marlin;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 120s;
            proxy_read_timeout 120s;

            # Streaming support
            proxy_buffering off;
            proxy_cache off;
        }
    }
}
```

## Testing

Run the server test suite (28 tests):

```bash
cd contrib/metal_marlin

# All tests (uses mock model, fast)
uv run pytest tests/test_openai_server.py -v

# Specific test categories
uv run pytest tests/test_openai_server.py -v -k "streaming"
uv run pytest tests/test_openai_server.py -v -k "concurrent"
uv run pytest tests/test_openai_server.py -v -k "validation"

# With real model (slower)
METAL_MARLIN_MOCK_MODEL=0 uv run pytest tests/test_openai_server.py -v
```

**Test coverage:**
- Basic functionality: health, models, chat, completions
- Streaming responses with SSE format
- Concurrent requests (10 requests, 5 workers)
- Input validation (missing fields, wrong types → 422)
- Error handling (wrong model → 404, no model loaded → 503)

## Production Checklist

- [ ] Configure HTTPS with valid SSL certificates
- [ ] Set up rate limiting
- [ ] Configure request timeout appropriately
- [ ] Enable health checks and monitoring
- [ ] Set up log aggregation (e.g., ELK, Loki)
- [ ] Configure metrics collection (Prometheus)
- [ ] Set up alerts for failures and high latency
- [ ] Test load balancing with multiple workers
- [ ] Configure automatic restarts
- [ ] Document API key management
- [ ] Set up backup strategy for models
- [ ] Test failover scenarios
- [ ] Monitor GPU memory usage
- [ ] Tune `max-concurrent-requests` based on hardware
