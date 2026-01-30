from __future__ import annotations

import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import tempfile
import time
from collections.abc import Generator
from pathlib import Path

import pytest
import requests

from metal_marlin._compat import HAS_MPS, HAS_TORCH

_PORT = 8123
_BASE_URL = f"http://localhost:{_PORT}"
_MODEL_NAME = "qwen3_4b_fp4"
_USE_MOCK = os.getenv("METAL_MARLIN_MOCK_MODEL", "1") == "1"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _model_path() -> Path:
    return _project_root() / "benchmarks" / "results" / _MODEL_NAME


def _wait_for_health(timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{_BASE_URL}/health", timeout=1.0)
            if response.status_code == 200:
                return
        except Exception as exc:  # pragma: no cover - retry loop
            last_error = exc
        time.sleep(0.5)
    detail = f" last error: {last_error}" if last_error else ""
    raise AssertionError(f"Server did not become healthy within {timeout_s}s.{detail}")


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


@pytest.fixture(scope="module")
def server_process() -> Generator[subprocess.Popen[str], None, None]:
    if not _USE_MOCK:
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        if not HAS_MPS:
            pytest.skip("MPS not available")

    model_path = _model_path()
    if not _USE_MOCK and not model_path.exists():
        pytest.skip(f"Model path not found: {model_path}")

    if not _port_available(_PORT):
        pytest.skip(f"Port {_PORT} is already in use")

    log_file = tempfile.NamedTemporaryFile(mode="w+", delete=False)
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "metal_marlin",
        "serve",
        str(model_path),
        "--port",
        str(_PORT),
    ]

    env = os.environ.copy()
    if _USE_MOCK:
        env["METAL_MARLIN_MOCK_MODEL"] = "1"
        env["METAL_MARLIN_MOCK_MODEL_NAME"] = _MODEL_NAME

    process = subprocess.Popen(
        cmd,
        cwd=_project_root(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=env,
    )

    try:
        _wait_for_health()
        yield process
    except Exception:
        log_file.flush()
        log_file.seek(0)
        logs = log_file.read()
        raise AssertionError(f"Server failed to start. Logs:\n{logs}") from None
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except Exception:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                pass
        log_file.close()
        try:
            os.unlink(log_file.name)
        except OSError:
            pass


def test_health(server_process: subprocess.Popen[str]) -> None:
    response = requests.get(f"{_BASE_URL}/health", timeout=5.0)
    assert response.status_code == 200


def test_list_models(server_process: subprocess.Popen[str]) -> None:
    response = requests.get(f"{_BASE_URL}/v1/models", timeout=5.0)
    assert response.status_code == 200
    data = response.json()
    assert "data" in data


def test_get_model_info(server_process: subprocess.Popen[str]) -> None:
    """Test detailed model info endpoint."""
    response = requests.get(
        f"{_BASE_URL}/v1/models/{_MODEL_NAME}",
        timeout=10.0,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == _MODEL_NAME
    assert "capabilities" in data
    assert data["capabilities"]["chat_completions"] is True
    assert "config" in data
    assert "format" in data["config"]


def test_get_model_info_not_found(server_process: subprocess.Popen[str]) -> None:
    """Test model info for nonexistent model returns 404."""
    response = requests.get(
        f"{_BASE_URL}/v1/models/nonexistent_model",
        timeout=10.0,
    )
    assert response.status_code == 404
    data = response.json()
    assert data["error"]["type"] == "model_not_found"


def test_chat_completion(server_process: subprocess.Popen[str]) -> None:
    response = requests.post(
        f"{_BASE_URL}/v1/chat/completions",
        json={
            "model": _MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
        },
        timeout=30.0,
    )
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert data["choices"][0]["message"]["content"]


def test_chat_completion_streaming(server_process: subprocess.Popen[str]) -> None:
    """Test streaming chat completion."""
    response = requests.post(
        f"{_BASE_URL}/v1/chat/completions",
        json={
            "model": _MODEL_NAME,
            "messages": [{"role": "user", "content": "Count to 5"}],
            "max_tokens": 50,
            "stream": True,
        },
        stream=True,
        timeout=60.0,
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    chunks = []
    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))

    assert len(chunks) > 0
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_completion_basic(server_process: subprocess.Popen[str]) -> None:
    """Test basic text completion endpoint."""
    response = requests.post(
        f"{_BASE_URL}/v1/completions",
        json={
            "model": _MODEL_NAME,
            "prompt": "Hello, my name is",
            "max_tokens": 10,
        },
        timeout=30.0,
    )
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert data["choices"][0]["text"]


def test_concurrent_requests(server_process: subprocess.Popen[str]) -> None:
    """Test multiple concurrent chat completions."""

    def make_request(idx: int) -> dict:
        response = requests.post(
            f"{_BASE_URL}/v1/chat/completions",
            json={
                "model": _MODEL_NAME,
                "messages": [{"role": "user", "content": f"Say {idx}"}],
                "max_tokens": 5,
            },
            timeout=60.0,
        )
        return {"status": response.status_code, "idx": idx}

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request, i) for i in range(10)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # All requests should succeed
    assert all(r["status"] == 200 for r in results)
    assert len(results) == 10


def test_concurrent_streaming(server_process: subprocess.Popen[str]) -> None:
    """Test multiple concurrent streaming requests."""

    def stream_request(idx: int) -> int:
        response = requests.post(
            f"{_BASE_URL}/v1/chat/completions",
            json={
                "model": _MODEL_NAME,
                "messages": [{"role": "user", "content": f"Count {idx}"}],
                "stream": True,
                "max_tokens": 10,
            },
            stream=True,
            timeout=60.0,
        )
        chunk_count = sum(1 for _ in response.iter_lines() if _)
        return chunk_count

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(stream_request, i) for i in range(3)]
        results = [f.result() for f in futures]

    # Each stream should have chunks
    assert all(r > 2 for r in results)


def test_invalid_message_format(server_process: subprocess.Popen[str]) -> None:
    """Test malformed messages array returns 422."""
    response = requests.post(
        f"{_BASE_URL}/v1/chat/completions",
        json={
            "model": _MODEL_NAME,
            "messages": [{"wrong_key": "no role or content"}],
        },
        timeout=10.0,
    )
    # Pydantic validation error returns 422
    assert response.status_code == 422


def test_invalid_model_name(server_process: subprocess.Popen[str]) -> None:
    """Test request with wrong model name returns 404."""
    response = requests.post(
        f"{_BASE_URL}/v1/chat/completions",
        json={
            "model": "nonexistent_model_xyz",
            "messages": [{"role": "user", "content": "Hi"}],
        },
        timeout=10.0,
    )
    # Should return 404 (model not found), not 500
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["type"] == "model_not_found"


def test_chat_completion_missing_model(server_process: subprocess.Popen[str]) -> None:
    """Test request without model field returns 422."""
    response = requests.post(
        f"{_BASE_URL}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hi"}],
        },
        timeout=10.0,
    )
    assert response.status_code == 422


def test_chat_completion_missing_messages(server_process: subprocess.Popen[str]) -> None:
    """Test request without messages field returns 422."""
    response = requests.post(
        f"{_BASE_URL}/v1/chat/completions",
        json={
            "model": _MODEL_NAME,
        },
        timeout=10.0,
    )
    assert response.status_code == 422


def test_chat_completion_empty_messages(server_process: subprocess.Popen[str]) -> None:
    """Test request with empty messages array returns 422."""
    response = requests.post(
        f"{_BASE_URL}/v1/chat/completions",
        json={
            "model": _MODEL_NAME,
            "messages": [],
        },
        timeout=10.0,
    )
    assert response.status_code == 422


def test_chat_message_missing_role(server_process: subprocess.Popen[str]) -> None:
    """Test message without role returns 422."""
    response = requests.post(
        f"{_BASE_URL}/v1/chat/completions",
        json={
            "model": _MODEL_NAME,
            "messages": [{"content": "Hi"}],
        },
        timeout=10.0,
    )
    assert response.status_code == 422


def test_chat_message_missing_content(server_process: subprocess.Popen[str]) -> None:
    """Test message without content returns 422."""
    response = requests.post(
        f"{_BASE_URL}/v1/chat/completions",
        json={
            "model": _MODEL_NAME,
            "messages": [{"role": "user"}],
        },
        timeout=10.0,
    )
    assert response.status_code == 422


def test_completion_missing_model(server_process: subprocess.Popen[str]) -> None:
    """Test completion request without model returns 422."""
    response = requests.post(
        f"{_BASE_URL}/v1/completions",
        json={
            "prompt": "Hello",
        },
        timeout=10.0,
    )
    assert response.status_code == 422


def test_completion_missing_prompt(server_process: subprocess.Popen[str]) -> None:
    """Test completion request without prompt returns 422."""
    response = requests.post(
        f"{_BASE_URL}/v1/completions",
        json={
            "model": _MODEL_NAME,
        },
        timeout=10.0,
    )
    assert response.status_code == 422


def test_chat_completion_streaming_missing_model(server_process: subprocess.Popen[str]) -> None:
    """Test streaming request without model field returns 422."""
    response = requests.post(
        f"{_BASE_URL}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
        timeout=10.0,
    )
    assert response.status_code == 422


def test_chat_completion_streaming_missing_messages(server_process: subprocess.Popen[str]) -> None:
    """Test streaming request without messages field returns 422."""
    response = requests.post(
        f"{_BASE_URL}/v1/chat/completions",
        json={
            "model": _MODEL_NAME,
            "stream": True,
        },
        timeout=10.0,
    )
    assert response.status_code == 422


def test_chat_completion_model_null(server_process: subprocess.Popen[str]) -> None:
    """Test request with null model field returns 422."""
    response = requests.post(
        f"{_BASE_URL}/v1/chat/completions",
        json={
            "model": None,
            "messages": [{"role": "user", "content": "Hi"}],
        },
        timeout=10.0,
    )
    assert response.status_code == 422


def test_chat_completion_messages_null(server_process: subprocess.Popen[str]) -> None:
    """Test request with null messages field returns 422."""
    response = requests.post(
        f"{_BASE_URL}/v1/chat/completions",
        json={
            "model": _MODEL_NAME,
            "messages": None,
        },
        timeout=10.0,
    )
    assert response.status_code == 422


def test_completion_streaming_missing_model(server_process: subprocess.Popen[str]) -> None:
    """Test streaming completion request without model returns 422."""
    response = requests.post(
        f"{_BASE_URL}/v1/completions",
        json={
            "prompt": "Hello",
            "stream": True,
        },
        timeout=10.0,
    )
    assert response.status_code == 422


def test_completion_streaming_missing_prompt(server_process: subprocess.Popen[str]) -> None:
    """Test streaming completion request without prompt returns 422."""
    response = requests.post(
        f"{_BASE_URL}/v1/completions",
        json={
            "model": _MODEL_NAME,
            "stream": True,
        },
        timeout=10.0,
    )
    assert response.status_code == 422


def test_chat_completion_messages_wrong_type(server_process: subprocess.Popen[str]) -> None:
    """Test request with messages as object instead of array returns 422."""
    response = requests.post(
        f"{_BASE_URL}/v1/completions",
        json={
            "model": _MODEL_NAME,
            "messages": {"role": "user", "content": "Hi"},
        },
        timeout=10.0,
    )
    assert response.status_code == 422


def test_chat_completion_model_wrong_type(server_process: subprocess.Popen[str]) -> None:
    """Test request with model as number instead of string returns 422."""
    response = requests.post(
        f"{_BASE_URL}/v1/completions",
        json={
            "model": 12345,
            "messages": [{"role": "user", "content": "Hi"}],
        },
        timeout=10.0,
    )
    assert response.status_code == 422


def test_chat_completion_message_element_wrong_type(server_process: subprocess.Popen[str]) -> None:
    """Test request with non-object element in messages array returns 422."""
    response = requests.post(
        f"{_BASE_URL}/v1/completions",
        json={
            "model": _MODEL_NAME,
            "messages": ["not an object"],
        },
        timeout=10.0,
    )
    assert response.status_code == 422


def test_completion_prompt_wrong_type(server_process: subprocess.Popen[str]) -> None:
    """Test request with prompt as number instead of string returns 422."""
    response = requests.post(
        f"{_BASE_URL}/v1/completions",
        json={
            "model": _MODEL_NAME,
            "prompt": 12345,
        },
        timeout=10.0,
    )
    assert response.status_code == 422
