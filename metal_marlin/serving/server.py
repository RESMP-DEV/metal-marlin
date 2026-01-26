from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .engine import EngineConfig, ServingEngine
from .openai_schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelList,
)

engine: ServingEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: engine initialized via configure()
    yield
    # Shutdown: cleanup if needed


app = FastAPI(
    title="Metal Marlin OpenAI API",
    description="OpenAI-compatible API for quantized LLM inference on Apple Silicon",
    version="0.1.0",
    lifespan=lifespan,
)


def configure(model_path: str, device: str = "mps", **kwargs):
    """Configure the serving engine before starting."""
    global engine
    engine = ServingEngine(EngineConfig(model_path=model_path, device=device, **kwargs))


@app.get("/v1/models")
async def list_models() -> ModelList:
    if engine is None:
        raise HTTPException(500, "Engine not configured")
    return ModelList(data=[ModelInfo(id=engine.model_name, created=0)])


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if engine is None:
        raise HTTPException(500, "Engine not configured")

    result = await engine.chat_completion(request)

    if request.stream:
        async def stream():
            async for chunk in result:
                yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream")

    return result


@app.post("/v1/completions")
async def completions(request: CompletionRequest) -> CompletionResponse:
    if engine is None:
        raise HTTPException(500, "Engine not configured")
    return await engine.completion(request)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": engine is not None}


def run_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    device: str = "mps",
):
    """Run the OpenAI-compatible server."""
    configure(model_path, device)
    uvicorn.run(app, host=host, port=port)
