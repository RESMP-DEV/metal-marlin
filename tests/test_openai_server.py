import pytest
from fastapi.testclient import TestClient

from metal_marlin.serving.server import app, configure


@pytest.fixture(scope="module")
def client():
    configure("benchmarks/results/qwen3_4b_fp4", device="mps")
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_list_models(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert len(response.json()["data"]) > 0


def test_chat_completion(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3_4b_fp4",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 10,
            "temperature": 0.0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "4" in data["choices"][0]["message"]["content"]


def test_completion(client):
    response = client.post(
        "/v1/completions",
        json={
            "model": "qwen3_4b_fp4",
            "prompt": "The capital of France is",
            "max_tokens": 5,
            "temperature": 0.0,
        },
    )
    assert response.status_code == 200
    assert "Paris" in response.json()["choices"][0]["text"]
