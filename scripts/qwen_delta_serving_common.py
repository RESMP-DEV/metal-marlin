"""Shared DeltaNet serving wrapper for Qwen3.5 / Qwen3.6 hybrid MoE models.

Extracts the common ``MMFP4Pipeline`` -> serving-adapter bridge so that
``serve_qwen35.py`` and ``serve_qwen36.py`` share one implementation of:

* ``install_mmfp4_adapter()`` - wires ``MMFP4Pipeline`` as the serving pipeline
* ``QwenDeltaServeConfig`` - frozen dataclass mirroring the server config shape
* ``add_serving_args()`` / ``build_config()`` - argument handling with env vars
* ``validate_serving_args()`` - validates parsed arguments
* ``run_serving_server()`` - launches the server with the config
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from dataclasses import dataclass

__all__ = [
    "QwenDeltaServeConfig",
    "add_serving_args",
    "build_config",
    "install_mmfp4_adapter",
    "run_serving_server",
    "validate_serving_args",
]

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def env_int(name: str, default: int) -> int:
    """Parse a required integer from ``os.environ``; exit on bad value."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid integer for {name}: {value!r}") from exc


# ---------------------------------------------------------------------------
# ServeConfig - shared dataclass shape
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QwenDeltaServeConfig:
    """FROZEN config object passed to the server entry point.

    Attributes mirror the metal_marlin.serving.server.run_server() signature
    plus model-path / host / port fields.
    """

    model_path: str
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "mps"
    batch_size: int = 32
    enable_batching: bool = True
    use_paged_attention: bool = True
    num_kv_blocks: int = 4096
    block_size: int = 16
    metrics_port: int | None = None


# ---------------------------------------------------------------------------
# Serving pipeline wrapper
# ---------------------------------------------------------------------------

def _enable_paged_attention_on_layers(pipeline_model: object) -> None:
    """Walk model layers and set ``use_paged_attention = True`` on any
    MMFP4 attention module that exposes the flag.

    This mirrors the per-layer patching performed by the adapter classes in
    both ``serve_qwen35.py`` and ``serve_glm47.py``.
    """
    model = getattr(pipeline_model, "model", pipeline_model)
    model_stack = getattr(model, "model", model)
    layers = getattr(model_stack, "layers", [])
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "use_paged_attention"):
            setattr(attn, "use_paged_attention", True)


class _QwenDeltaServingPipeline:
    """Thin adapter that exposes the serving engine's call contract.

    Wraps ``MMFP4Pipeline`` loaded via
    ``MMFP4Pipeline.from_pretrained(..., enable_persistent_cache=True)`` and
    translates ``max_tokens`` -> ``max_new_tokens`` so the caller never needs to
    know about internal naming.

    The class is registered into ``metal_marlin.serving.engine.MarlinPipeline``
    at startup so the server discovers it via the same import path as the
    existing GLM-4.7-Flash adapter.

    Args:
        pipeline: An already-constructed ``MMFP4Pipeline`` instance.
        use_paged_attention: Whether to enable paged attention on all
            ``self_attn`` modules that support the flag.
    """

    def __init__(
        self,
        pipeline: object,  # MMFP4Pipeline but avoid circular import
        use_paged_attention: bool = True,
    ) -> None:
        self._pipeline = pipeline
        self.tokenizer = getattr(pipeline, "tokenizer")
        self.model = getattr(pipeline, "model")
        self.config = getattr(self.model, "config", None)
        if use_paged_attention:
            _enable_paged_attention_on_layers(self._pipeline)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "mps",
        use_paged_attention: bool = True,
    ) -> _QwenDeltaServingPipeline:
        """Load an MMFP4-quantized Qwen DeltaNet checkpoint.

        Passes ``enable_persistent_cache=True`` to the underlying pipeline
        so that multi-turn conversations reuse KV tensors across turns.
        ``use_paged_attention`` is forwarded to the wrapper so paged KV
        management can be toggled independently of the pipeline.
        """
        from metal_marlin.inference.mmfp4_pipeline import MMFP4Pipeline

        pipeline = MMFP4Pipeline.from_pretrained(
            model_path,
            device=device,
            enable_persistent_cache=True,
            use_paged_attention=use_paged_attention,
        )
        return cls(pipeline, use_paged_attention=use_paged_attention)

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> str | Iterator[str]:
        # The serving engine passes max_tokens; MMFP4Pipeline expects
        # max_new_tokens.
        return self._pipeline(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
        )


def install_mmfp4_adapter(
    enable_persistent_cache: bool = True,
    use_paged_attention: bool = True,
) -> None:
    """Install the Qwen DeltaNet serving adapter.

    This function registers ``_QwenDeltaServingPipeline`` as the
    ``MarlinPipeline`` class in ``metal_marlin.serving.engine``, so that
    the server discovers it via the standard import path.

    Args:
        enable_persistent_cache: If True, the underlying MMFP4Pipeline
            will use persistent KV cache across multi-turn conversations.
        use_paged_attention: If True, enables paged attention on all
            MMFP4 attention layers that support the flag.
    """
    from metal_marlin.serving import engine as serving_engine

    # Create a closure that captures the configuration
    class _ConfiguredPipeline(_QwenDeltaServingPipeline):
        @classmethod
        def from_pretrained(
            cls,
            model_path: str,
            device: str = "mps",
            use_paged_attention: bool = use_paged_attention,
        ) -> _ConfiguredPipeline:
            from metal_marlin.inference.mmfp4_pipeline import MMFP4Pipeline

            pipeline = MMFP4Pipeline.from_pretrained(
                model_path,
                device=device,
                enable_persistent_cache=enable_persistent_cache,
                use_paged_attention=use_paged_attention,
            )
            return cls(pipeline, use_paged_attention=use_paged_attention)

    serving_engine.MarlinPipeline = _ConfiguredPipeline  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def add_serving_args(
    parser: argparse.ArgumentParser,
    env_prefix: str,
    default_model_path: str | None = None,
) -> None:
    """Add shared serving arguments to an argument parser.

    Args:
        parser: The argument parser to add arguments to.
        env_prefix: Upper-case prefix for environment variables, e.g. ``QWEN35``.
        default_model_path: Fallback model directory when the env var is unset.
    """
    env_path = f"{env_prefix}_MODEL_PATH"
    env_port = f"{env_prefix}_PORT"
    env_batch = f"{env_prefix}_BATCH_SIZE"

    parser.add_argument(
        "--model-path",
        default=os.getenv(env_path, default_model_path),
        help=f"Path to the model directory (env: {env_path}).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address.")
    parser.add_argument(
        "--port",
        type=int,
        default=env_int(env_port, 8000),
        help=f"Server port (env: {env_port}).",
    )
    parser.add_argument("--device", default="mps", help="Device to run on (mps/cpu).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=env_int(env_batch, 32),
        help=f"Maximum concurrent requests (env: {env_batch}).",
    )
    parser.add_argument(
        "--num-kv-blocks",
        type=int,
        default=4096,
        help="KV cache blocks for continuous batching.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Tokens per KV cache block.",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Optional dedicated Prometheus metrics port.",
    )
    parser.add_argument(
        "--no-paged-attention",
        action="store_true",
        help="Disable paged attention toggle on MMFP4 attention layers.",
    )


def validate_serving_args(args: argparse.Namespace, env_prefix: str) -> None:
    """Validate parsed serving arguments; exit with error if invalid.

    Args:
        args: Parsed argument namespace.
        env_prefix: Upper-case prefix used in error messages.
    """
    if not args.model_path:
        raise SystemExit(
            f"Missing model path. Provide --model-path or set {env_prefix}_MODEL_PATH."
        )
    for name, val in [
        ("--port", args.port),
        ("--batch-size", args.batch_size),
        ("--num-kv-blocks", args.num_kv_blocks),
        ("--block-size", args.block_size),
    ]:
        if (val or 0) <= 0:
            raise SystemExit(f"{name} must be > 0.")


def build_config(
    args: argparse.Namespace,
    env_prefix: str,
) -> QwenDeltaServeConfig:
    """Build a ``QwenDeltaServeConfig`` from parsed arguments.

    Args:
        args: Parsed argument namespace.
        env_prefix: Upper-case prefix (unused but kept for API symmetry).
    """
    _ = env_prefix  # Prefix already applied during parsing
    return QwenDeltaServeConfig(
        model_path=str(args.model_path),
        host=str(args.host),
        port=int(args.port),
        device=str(args.device),
        batch_size=int(args.batch_size),
        enable_batching=True,
        use_paged_attention=not bool(args.no_paged_attention),
        num_kv_blocks=int(args.num_kv_blocks),
        block_size=int(args.block_size),
        metrics_port=args.metrics_port,
    )


# ---------------------------------------------------------------------------
# Server launch helper
# ---------------------------------------------------------------------------

def run_serving_server(config: QwenDeltaServeConfig) -> int:
    """Apply the serving adapter and call ``run_server()``.

    Args:
        config: The serving configuration.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    from metal_marlin.serving.server import run_server

    run_server(
        model_path=config.model_path,
        host=config.host,
        port=config.port,
        device=config.device,
        batch_size=config.batch_size,
        enable_batching=config.enable_batching,
        num_kv_blocks=config.num_kv_blocks,
        block_size=config.block_size,
        metrics_port=config.metrics_port,
    )
    return 0
