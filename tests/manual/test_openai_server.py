#!/usr/bin/env python3
"""Test script for GLM-4.7-Flash OpenAI-compatible server.

This script tests the serving endpoints with various request patterns.

Prerequisites:
    1. Start the server:
        uv run python scripts/serve_glm47.py --model-path ./models/glm47-flash-mmfp4
    
    2. Run tests:
        uv run python tests/manual/test_openai_server.py
"""

import json
import sys
import time
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8000"


def test_health():
    """Test health endpoint."""
    print("\n[1/7] Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        print("  ✅ Health check passed")
        return True
    except Exception as e:
        print(f"  ❌ Health check failed: {e}")
        return False


def test_list_models():
    """Test model listing endpoint."""
    print("\n[2/7] Testing /v1/models endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/v1/models", timeout=5)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "data" in data, "Missing 'data' field"
        assert len(data["data"]) > 0, "No models returned"

        model = data["data"][0]
        print(f"  ✅ Models listed: {model['id']}")
        return True
    except Exception as e:
        print(f"  ❌ List models failed: {e}")
        return False


def test_chat_completion_basic():
    """Test basic chat completion."""
    print("\n[3/7] Testing basic chat completion...")
    try:
        payload = {
            "model": "glm-4.7-flash",
            "messages": [
                {"role": "user", "content": "Hello! Respond with just 'Hi'."}
            ],
            "max_tokens": 10,
            "temperature": 0.7
        }

        start = time.time()
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=30
        )
        elapsed = time.time() - start

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "choices" in data, "Missing 'choices' field"
        assert len(data["choices"]) > 0, "No choices returned"

        content = data["choices"][0]["message"]["content"]
        print(f"  ✅ Response: {content[:50]}... ({elapsed:.2f}s)")
        return True
    except Exception as e:
        print(f"  ❌ Basic chat failed: {e}")
        return False


def test_chat_completion_streaming():
    """Test streaming chat completion."""
    print("\n[4/7] Testing streaming chat completion...")
    try:
        payload = {
            "model": "glm-4.7-flash",
            "messages": [
                {"role": "user", "content": "Count from 1 to 5"}
            ],
            "max_tokens": 30,
            "temperature": 0.7,
            "stream": True
        }

        start = time.time()
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=30
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        chunks = []
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk != "[DONE]":
                        chunks.append(chunk)

        elapsed = time.time() - start
        print(f"  ✅ Streaming: {len(chunks)} chunks ({elapsed:.2f}s)")
        return True
    except Exception as e:
        print(f"  ❌ Streaming failed: {e}")
        return False


def test_text_completion():
    """Test legacy text completion endpoint."""
    print("\n[5/7] Testing text completion...")
    try:
        payload = {
            "model": "glm-4.7-flash",
            "prompt": "The capital of France is",
            "max_tokens": 10,
            "temperature": 0.7
        }

        start = time.time()
        response = requests.post(
            f"{BASE_URL}/v1/completions",
            json=payload,
            timeout=30
        )
        elapsed = time.time() - start

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "choices" in data, "Missing 'choices' field"

        text = data["choices"][0]["text"]
        print(f"  ✅ Completion: {text[:50]}... ({elapsed:.2f}s)")
        return True
    except Exception as e:
        print(f"  ❌ Text completion failed: {e}")
        return False


def test_concurrent_requests():
    """Test multiple concurrent requests."""
    print("\n[6/7] Testing concurrent requests...")
    try:
        import concurrent.futures

        prompts = [
            "What is AI?",
            "Explain machine learning",
            "Define neural networks",
            "Describe deep learning",
            "What is NLP?"
        ]

        def make_request(prompt):
            payload = {
                "model": "glm-4.7-flash",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 20,
                "temperature": 0.7
            }
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                json=payload,
                timeout=30
            )
            return time.time() - start, response.status_code

        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, p) for p in prompts]
            results = [f.result()
                       for f in concurrent.futures.as_completed(futures)]

        total_time = time.time() - start
        successful = sum(1 for _, status in results if status == 200)

        print(
            f"  ✅ Concurrent: {successful}/5 successful in {total_time:.2f}s")
        print(
            f"  Throughput: ~{successful * 20 / total_time:.1f} tok/s aggregate")
        return successful == 5
    except Exception as e:
        print(f"  ❌ Concurrent test failed: {e}")
        return False


def test_metrics():
    """Test metrics endpoint."""
    print("\n[7/7] Testing /metrics endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=5)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()

        print("  ✅ Metrics:")
        print(f"    Throughput: {data.get('throughput_tok_sec', 'N/A')} tok/s")
        print(f"    Latency: {data.get('avg_latency_ms', 'N/A')} ms")
        print(f"    Queue: {data.get('queue_depth', 'N/A')}")
        print(f"    Active: {data.get('active_requests', 'N/A')}")
        return True
    except Exception as e:
        print(f"  ❌ Metrics failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("GLM-4.7-Flash OpenAI Server Tests")
    print("="*60)

    # Check if server is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except Exception:
        print("\n❌ Server not running at", BASE_URL)
        print("\nStart the server first:")
        print(
            "  uv run python scripts/serve_glm47.py --model-path ./models/glm47-flash-mmfp4")
        sys.exit(1)

    # Run tests
    results = []
    results.append(test_health())
    results.append(test_list_models())
    results.append(test_chat_completion_basic())
    results.append(test_chat_completion_streaming())
    results.append(test_text_completion())
    results.append(test_concurrent_requests())
    results.append(test_metrics())

    # Summary
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)

    if passed == total:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
