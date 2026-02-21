from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager
from dataclasses import dataclass

import httpx
import pytest
import pytest_asyncio

from metal_marlin.serving import server as serving_server


@dataclass(frozen=True)
class _ServerInfo:
    model_name: str


@contextmanager
def _mock_model_env(model_name: str) -> Generator[None, None, None]:
    keys = ("METAL_MARLIN_MOCK_MODEL", "METAL_MARLIN_MOCK_MODEL_NAME")
    previous = {key: os.environ.get(key) for key in keys}

    os.environ["METAL_MARLIN_MOCK_MODEL"] = "1"
    os.environ["METAL_MARLIN_MOCK_MODEL_NAME"] = model_name

    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest_asyncio.fixture
async def openai_client() -> AsyncGenerator[tuple[httpx.AsyncClient, _ServerInfo], None]:
    model_name = "glm47-mock"

    with tempfile.TemporaryDirectory() as model_dir, _mock_model_env(model_name):
        serving_server.configure(model_dir, device="cpu")
        transport = httpx.ASGITransport(app=serving_server.app, raise_app_exceptions=False)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=20.0,
        ) as client:
            yield client, _ServerInfo(model_name=model_name)

    serving_server.engine = None


@pytest_asyncio.fixture
async def timeout_client() -> AsyncGenerator[tuple[httpx.AsyncClient, _ServerInfo], None]:
    model_name = "glm47-timeout-mock"

    with tempfile.TemporaryDirectory() as model_dir, _mock_model_env(model_name):
        serving_server.configure(model_dir, device="cpu", request_timeout=0.05)
        assert serving_server.engine is not None

        original_run_pipeline = serving_server.engine._run_pipeline

        async def _slow_run_pipeline(
            prompt: str,
            max_tokens: int,
            temperature: float,
            top_p: float,
        ) -> str:
            _ = (prompt, max_tokens, temperature, top_p)
            await asyncio.sleep(0.2)
            return "delayed"

        serving_server.engine._run_pipeline = _slow_run_pipeline  # type: ignore[method-assign]

        transport = httpx.ASGITransport(app=serving_server.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=20.0,
        ) as client:
            yield client, _ServerInfo(model_name=model_name)

        serving_server.engine._run_pipeline = original_run_pipeline

    serving_server.engine = None


@pytest.mark.asyncio
async def test_v1_models_endpoint(openai_client: tuple[httpx.AsyncClient, _ServerInfo]) -> None:
    client, server_info = openai_client
    response = await client.get("/v1/models")
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
    assert len(data["data"]) == 1

    model = data["data"][0]
    assert model["id"] == server_info.model_name
    assert model["object"] == "model"
    assert model["owned_by"] == "metal-marlin"


@pytest.mark.asyncio
async def test_chat_completions_non_streaming(
    openai_client: tuple[httpx.AsyncClient, _ServerInfo],
) -> None:
    client, server_info = openai_client
    payload = {
        "model": server_info.model_name,
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 16,
        "stream": False,
    }

    response = await client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")
    assert data["model"] == server_info.model_name
    assert isinstance(data["created"], int)

    assert isinstance(data["choices"], list)
    assert len(data["choices"]) == 1
    choice = data["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["message"]["content"].strip()
    assert choice["finish_reason"] in {"stop", "length", "content_filter"}

    usage = data["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@pytest.mark.asyncio
async def test_chat_completions_streaming(
    openai_client: tuple[httpx.AsyncClient, _ServerInfo],
) -> None:
    client, server_info = openai_client
    payload = {
        "model": server_info.model_name,
        "messages": [{"role": "user", "content": "Count to three."}],
        "max_tokens": 16,
        "stream": True,
    }

    chunks: list[dict] = []
    saw_done = False

    async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        async for line in response.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            event = line[6:]
            if event == "[DONE]":
                saw_done = True
                continue
            chunks.append(json.loads(event))

    assert saw_done
    assert chunks
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_completions_non_streaming(
    openai_client: tuple[httpx.AsyncClient, _ServerInfo],
) -> None:
    client, server_info = openai_client
    payload = {
        "model": server_info.model_name,
        "prompt": "Hello from completion endpoint",
        "max_tokens": 16,
    }

    response = await client.post("/v1/completions", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "text_completion"
    assert data["id"].startswith("cmpl-")
    assert data["model"] == server_info.model_name
    assert isinstance(data["choices"], list)
    assert len(data["choices"]) == 1
    assert isinstance(data["choices"][0]["text"], str)
    assert data["choices"][0]["finish_reason"] in {"stop", "length"}

    usage = data["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@pytest.mark.asyncio
async def test_error_model_not_found(openai_client: tuple[httpx.AsyncClient, _ServerInfo]) -> None:
    client, _ = openai_client
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8,
        },
    )

    assert response.status_code == 404
    data = response.json()
    assert data["error"]["type"] == "model_not_found"
    assert data["error"]["code"] == 404
    assert "nonexistent-model" in data["error"]["message"]


@pytest.mark.asyncio
async def test_error_timeout(timeout_client: tuple[httpx.AsyncClient, _ServerInfo]) -> None:
    client, server_info = timeout_client
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": server_info.model_name,
            "messages": [{"role": "user", "content": "Trigger timeout"}],
            "max_tokens": 8,
        },
    )

    assert response.status_code == 500
    data = response.json()
    assert data["error"]["type"] == "server_error"
    assert data["error"]["code"] == 500
    assert data["error"]["message"] == "Internal server error"
