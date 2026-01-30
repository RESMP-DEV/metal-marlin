from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .engine import EngineConfig, ServingEngine
from .errors import (
    ModelNotFoundError,
    ModelNotLoadedError,
    ServingError,
)
from .metrics import MetricsCollector
from .openai_schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelList,
)

engine: ServingEngine | None = None
metrics = MetricsCollector()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Metal Marlin OpenAI API",
    description="OpenAI-compatible API for quantized LLM inference on Apple Silicon",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(ServingError)
async def serving_error_handler(request: Request, exc: ServingError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc),
                "type": exc.error_type,
                "code": exc.status_code,
            }
        },
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    # Log the error for debugging
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "code": 500,
            }
        },
    )


def configure(model_path: str, device: str = "mps", **kwargs):
    global engine
    engine = ServingEngine(EngineConfig(model_path=model_path, device=device, **kwargs))

    def get_queue_depth() -> int:
        if engine and engine.scheduler:
            return engine.scheduler.num_waiting
        return 0

    metrics.set_queue_depth_callback(get_queue_depth)


@app.get("/v1/models")
async def list_models() -> ModelList:
    if engine is None:
        raise ModelNotLoadedError()
    return ModelList(data=[ModelInfo(id=engine.model_name, created=0)])


@app.get("/v1/models/{model_id}")
async def get_model_info(model_id: str):
    if engine is None:
        raise ModelNotLoadedError()
    if model_id != engine.model_name:
        raise ModelNotFoundError(f"Model '{model_id}' not found. Loaded: {engine.model_name}")
    return engine.get_model_info()


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if engine is None:
        raise ModelNotLoadedError()

    # Validate model name
    if request.model != engine.model_name:
        raise ModelNotFoundError(
            f"Model '{request.model}' not found. Available: {engine.model_name}"
        )

    result = await engine.chat_completion(request)

    if request.stream:

        async def stream():
            async for chunk in result:
                yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        metrics.start_request("/v1/chat/completions")
        return StreamingResponse(stream(), media_type="text/event-stream")

    # Track metrics for non-streaming requests
    req_metrics = metrics.start_request("/v1/chat/completions")
    req_metrics.prompt_tokens = result.usage.prompt_tokens
    req_metrics.completion_tokens = result.usage.completion_tokens
    metrics.end_request(req_metrics)

    return result


@app.post("/v1/completions")
async def completions(request: CompletionRequest) -> CompletionResponse:
    if engine is None:
        raise ModelNotLoadedError()

    if request.model != engine.model_name:
        raise ModelNotFoundError(
            f"Model '{request.model}' not found. Available: {engine.model_name}"
        )

    result = await engine.completion(request)

    # Track metrics
    req_metrics = metrics.start_request("/v1/completions")
    req_metrics.prompt_tokens = result.usage.prompt_tokens
    req_metrics.completion_tokens = result.usage.completion_tokens
    metrics.end_request(req_metrics)

    return result


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": engine is not None}


@app.get("/metrics")
async def get_metrics():
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(content=metrics.to_prometheus(), media_type="text/plain")


def run_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    device: str = "mps",
    batch_size: int = 32,
    enable_batching: bool = False,
    num_kv_blocks: int = 512,
    block_size: int = 16,
):
    """Start the OpenAI-compatible API server.

    Args:
        model_path: Path to the quantized model
        host: Bind address for the server
        port: Port number for the server
        device: Device to use (mps/cpu)
        batch_size: Max concurrent requests
        enable_batching: Enable continuous batching
        num_kv_blocks: Number of KV cache blocks to allocate
        block_size: Number of tokens per KV cache block
    """
    import signal
    import sys

    configure(
        model_path,
        device=device,
        max_batch_size=batch_size,
        enable_batching=enable_batching,
        num_kv_blocks=num_kv_blocks,
        block_size=block_size,
    )

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nShutting down server...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print(f"Starting Metal Marlin server on {host}:{port}")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Continuous batching: {'enabled' if enable_batching else 'disabled'}")
    print("Press Ctrl+C to stop")

    uvicorn.run(app, host=host, port=port)
