#!/usr/bin/env python3
"""Launch the GLM-4.7-Flash OpenAI-compatible server.

This wrapper uses `metal_marlin.serving.server.run_server()` with GLM-4.7-Flash
defaults and wires serving to `MMFP4Pipeline` (which uses `MLAKVCache` for MLA
models).

Examples:
    # 1) Use environment defaults
    #    GLM47_MODEL_PATH is required unless --model-path is passed.
    GLM47_MODEL_PATH=models/GLM-4.7-Flash-Marlin-MMFP4 \
    uv run python scripts/serve_glm47.py

    # 2) Explicit arguments
    uv run python scripts/serve_glm47.py \
      --model-path models/GLM-4.7-Flash-Marlin-MMFP4 \
      --port 8000 \
      --batch-size 32

    # 3) Environment variable overrides
    GLM47_MODEL_PATH=models/GLM-4.7-Flash-Marlin-MMFP4 \
    GLM47_PORT=8080 \
    GLM47_BATCH_SIZE=24 \
    uv run python scripts/serve_glm47.py
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid integer for {name}: {value!r}") from exc


@dataclass(frozen=True)
class GLM47ServeConfig:
    model_path: str
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "mps"
    batch_size: int = 32
    enable_batching: bool = True
    use_paged_attention: bool = True
    num_kv_blocks: int = 4096  # Enough for 64K context
    block_size: int = 16
    metrics_port: int | None = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch GLM-4.7-Flash OpenAI server with MMFP4 + PagedAttention defaults.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("GLM47_MODEL_PATH"),
        help="Path to the GLM-4.7-Flash model directory (env: GLM47_MODEL_PATH).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address.")
    parser.add_argument(
        "--port",
        type=int,
        default=_env_int("GLM47_PORT", 8000),
        help="Server port (env: GLM47_PORT).",
    )
    parser.add_argument("--device", default="mps", help="Device to run on (mps/cpu).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_env_int("GLM47_BATCH_SIZE", 32),
        help="Maximum concurrent requests (env: GLM47_BATCH_SIZE).",
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
    return parser


def _install_mmfp4_adapter(use_paged_attention: bool) -> None:
    """Make serving load MMFP4Pipeline while keeping serving's call contract."""
    import metal_marlin.serving.engine as serving_engine
    from metal_marlin.inference.mmfp4_pipeline import MMFP4Pipeline
    from metal_marlin.kv_cache import MLAKVCache

    _ = MLAKVCache  # Document intentional reliance on MLA KV cache support.

    class MMFP4ServingPipeline:
        def __init__(self, pipeline: MMFP4Pipeline) -> None:
            self._pipeline = pipeline
            self.tokenizer = pipeline.tokenizer
            self.model = pipeline.model
            self.config = getattr(pipeline.model, "config", None)
            if use_paged_attention:
                self._enable_paged_attention()

        @classmethod
        def from_pretrained(
            cls, model_path: str, device: str = "mps"
        ) -> "MMFP4ServingPipeline":
            pipeline = MMFP4Pipeline.from_pretrained(model_path, device=device)
            return cls(pipeline)

        def _enable_paged_attention(self) -> None:
            # MMFP4 attention layers expose `use_paged_attention`; enable it
            # where present while keeping compatibility with other layer types.
            model = getattr(self._pipeline, "model", None)
            model_stack = getattr(model, "model", model)
            layers = getattr(model_stack, "layers", [])
            for layer in layers:
                attn = getattr(layer, "self_attn", None)
                if attn is not None and hasattr(attn, "use_paged_attention"):
                    setattr(attn, "use_paged_attention", True)

        def __call__(
            self,
            prompt: str,
            max_tokens: int = 256,
            temperature: float = 0.7,
            top_p: float = 0.9,
            stream: bool = False,
        ) -> str | Iterator[str]:
            return self._pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
            )

    serving_engine.MarlinPipeline = MMFP4ServingPipeline  # type: ignore[assignment]


def _parse_config(argv: list[str] | None = None) -> GLM47ServeConfig:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.model_path:
        parser.error("Missing model path. Provide --model-path or set GLM47_MODEL_PATH.")
    if args.port <= 0:
        parser.error("--port must be > 0.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0.")
    if args.num_kv_blocks <= 0:
        parser.error("--num-kv-blocks must be > 0.")
    if args.block_size <= 0:
        parser.error("--block-size must be > 0.")

    return GLM47ServeConfig(
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


def main(argv: list[str] | None = None) -> int:
    config = _parse_config(argv)
    _install_mmfp4_adapter(use_paged_attention=config.use_paged_attention)

    # Entry point requested by task.
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


if __name__ == "__main__":
    raise SystemExit(main())
