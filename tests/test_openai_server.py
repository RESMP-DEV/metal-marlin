from __future__ import annotations

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
