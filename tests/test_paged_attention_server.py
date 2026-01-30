"""Tests for paged attention mode in the OpenAI server."""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest
import requests

_PORT = 8125
_BASE_URL = f"http://localhost:{_PORT}"
_MODEL_NAME = "test-paged-model"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _wait_for_health(timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            response = requests.get(f"{_BASE_URL}/health", timeout=1.0)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise AssertionError(f"Server did not become healthy within {timeout_s}s")


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


@pytest.fixture(scope="module")
def paged_server():
    """Start server with paged attention enabled."""
    if not _port_available(_PORT):
        pytest.skip(f"Port {_PORT} is already in use")

    log_file = tempfile.NamedTemporaryFile(mode="w+", delete=False)
    cmd = [
        "uv", "run", "python", "-m", "metal_marlin", "serve",
        "/tmp/any",
        "--port", str(_PORT),
        "--enable-batching",
        "--num-kv-blocks", "64",
        "--block-size", "8",
    ]

    env = os.environ.copy()
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


def test_health_with_batching(paged_server):
    """Health check works with batching enabled."""
    response = requests.get(f"{_BASE_URL}/health", timeout=5.0)
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_chat_completion_with_batching(paged_server):
    """Chat completion works with batching enabled."""
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
