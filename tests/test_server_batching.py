"""Test the continuous batching server implementation."""

import asyncio
from unittest.mock import MagicMock

import pytest


def test_server_batching_imports():
    """Test that _server_batching can be imported."""
    from metal_marlin.serving.mmfp4_server import _server_batching, MMFP4Server

    assert callable(_server_batching)
    assert isinstance(MMFP4Server, type)


def test_mmfp4server_has_continuous_batching_methods():
    """Test that MMFP4Server has the continuous batching methods."""
    from metal_marlin.serving.mmfp4_server import MMFP4Server

    # Check methods exist
    assert hasattr(MMFP4Server, "start_continuous_batching")
    assert hasattr(MMFP4Server, "stop_continuous_batching")
    assert hasattr(MMFP4Server, "submit_request_to_queue")
    assert hasattr(MMFP4Server, "request_queue")
    assert hasattr(MMFP4Server, "result_queue")


def test_server_batching_signature():
    """Test that _server_batching has the correct signature."""
    import inspect
    from metal_marlin.serving.mmfp4_server import _server_batching

    sig = inspect.signature(_server_batching)
    params = list(sig.parameters.keys())

    assert "server" in params
    assert "request_queue" in params
    assert "result_queue" in params
    assert "shutdown_event" in params


@pytest.mark.asyncio
async def test_continuous_batching_queues_initialized():
    """Test that continuous batching properly initializes queues."""
    from metal_marlin.serving.mmfp4_server import MMFP4Server

    # Mock engine
    mock_engine = MagicMock()
    mock_engine.config.request_timeout = 30.0

    # Create server
    server = MMFP4Server(mock_engine)

    # Queues should be None before start
    assert server.request_queue is None
    assert server.result_queue is None


@pytest.mark.asyncio
async def test_submit_request_to_queue_raises_when_not_started():
    """Test that submit_request_to_queue raises when continuous batching not started."""
    from metal_marlin.serving.mmfp4_server import MMFP4Server, GenerationRequest

    # Mock engine
    mock_engine = MagicMock()

    # Create server
    server = MMFP4Server(mock_engine)

    # Try to submit without starting
    mock_request = MagicMock(spec=GenerationRequest)

    with pytest.raises(RuntimeError, match="Continuous batching mode is not active"):
        server.submit_request_to_queue(mock_request)


def test_server_batching_docstring():
    """Test that _server_batching has proper documentation."""
    from metal_marlin.serving.mmfp4_server import _server_batching

    assert _server_batching.__doc__ is not None
    assert "continuous batching" in _server_batching.__doc__.lower()
    assert "request_queue" in _server_batching.__doc__
    assert "result_queue" in _server_batching.__doc__
