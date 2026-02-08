"""Trace all command buffer commits during one forward pass."""
import os
os.environ["METAL_MARLIN_TRACE_BATCH"] = "1"

import logging
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import torch


_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Set up tracing logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("metal_marlin.batch_trace")

_TRACE_OUTPUT = Path(
    os.environ.get("METAL_MARLIN_COMMIT_TRACE_OUTPUT", str(_THIS_DIR / "commit_traces.txt"))
)
_COMMIT_EVENTS: list[dict[str, Any]] = []
_BATCH_LOG_EVENTS: list[str] = []
_PATCH_WARNINGS: list[str] = []


class _BatchTraceCaptureHandler(logging.Handler):
    """Capture existing batch trace logger output."""

    def emit(self, record: logging.LogRecord) -> None:
        _BATCH_LOG_EVENTS.append(self.format(record))


class _CommandBufferProxy:
    """Fallback wrapper when monkey-patching the command buffer instance fails."""

    def __init__(self, inner: Any, source: str) -> None:
        self._inner = inner
        self._source = source

    def commit(self, *args: Any, **kwargs: Any) -> Any:
        _record_commit(self._source)
        return self._inner.commit(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _record_commit(source: str) -> None:
    stack_lines = traceback.format_stack(limit=64)
    stack_text = "".join(stack_lines[:-2]) if len(stack_lines) >= 2 else "".join(stack_lines)
    _COMMIT_EVENTS.append(
        {
            "index": len(_COMMIT_EVENTS) + 1,
            "source": source,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stack": stack_text,
        }
    )
    logger.debug("TRACE COMMIT %d: %s", len(_COMMIT_EVENTS), source)


def _patch_method(owner: Any, method_name: str, source_name: str) -> bool:
    method = getattr(owner, method_name, None)
    if not callable(method):
        return False
    if getattr(method, "_mm_commit_trace_wrapped", False):
        return True

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        _record_commit(source_name)
        return method(*args, **kwargs)

    wrapped._mm_commit_trace_wrapped = True  # type: ignore[attr-defined]
    try:
        setattr(owner, method_name, wrapped)
    except Exception as exc:
        _PATCH_WARNINGS.append(f"Failed to patch {source_name}: {exc}")
        return False
    return True


def _patch_function(module: Any, function_name: str, source_name: str) -> bool:
    fn = getattr(module, function_name, None)
    if not callable(fn):
        return False
    if getattr(fn, "_mm_commit_trace_wrapped", False):
        return True

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        _record_commit(source_name)
        return fn(*args, **kwargs)

    wrapped._mm_commit_trace_wrapped = True  # type: ignore[attr-defined]
    try:
        setattr(module, function_name, wrapped)
    except Exception as exc:
        _PATCH_WARNINGS.append(f"Failed to patch {source_name}: {exc}")
        return False
    return True


def _instrument_command_buffer(command_buffer: Any) -> Any:
    if command_buffer is None:
        return None

    if getattr(command_buffer, "_mm_trace_wrapped", False):
        return command_buffer

    original_commit = getattr(command_buffer, "commit", None)
    if not callable(original_commit):
        return command_buffer

    def traced_commit(*args: Any, **kwargs: Any) -> Any:
        _record_commit("MTLCommandBuffer.commit")
        return original_commit(*args, **kwargs)

    try:
        setattr(command_buffer, "commit", traced_commit)
        setattr(command_buffer, "_mm_trace_wrapped", True)
        return command_buffer
    except Exception:
        return _CommandBufferProxy(command_buffer, "MTLCommandBuffer.commit (proxy)")


def _patch_pyobjc_command_queue() -> bool:
    try:
        import Metal
    except Exception as exc:  # pragma: no cover - platform dependent
        _PATCH_WARNINGS.append(f"PyObjC Metal unavailable: {exc}")
        return False

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        _PATCH_WARNINGS.append("No default Metal device found")
        return False

    queue = device.newCommandQueue()
    if queue is None:
        _PATCH_WARNINGS.append("Failed to create Metal command queue")
        return False

    queue_type = type(queue)
    command_buffer_method = getattr(queue_type, "commandBuffer", None)
    if not callable(command_buffer_method):
        _PATCH_WARNINGS.append("MTLCommandQueue.commandBuffer() not callable")
        return False
    if getattr(command_buffer_method, "_mm_commit_trace_wrapped", False):
        return True

    def wrapped_command_buffer(self: Any, *args: Any, **kwargs: Any) -> Any:
        command_buffer = command_buffer_method(self, *args, **kwargs)
        return _instrument_command_buffer(command_buffer)

    wrapped_command_buffer._mm_commit_trace_wrapped = True  # type: ignore[attr-defined]

    try:
        setattr(queue_type, "commandBuffer", wrapped_command_buffer)
    except Exception as exc:
        _PATCH_WARNINGS.append(f"Failed to patch MTLCommandQueue.commandBuffer: {exc}")
        return False

    return True


def _patch_cpp_extension() -> None:
    try:
        from metal_marlin._compat import _metal_dispatch_ext
    except Exception as exc:
        _PATCH_WARNINGS.append(f"Failed importing _compat: {exc}")
        return

    if _metal_dispatch_ext is None:
        _PATCH_WARNINGS.append("metal_marlin._cpp_ext not available")
        return

    patched_any = False

    # Direct C++ dispatch function (contains command buffer commit internally).
    patched_any = _patch_function(
        _metal_dispatch_ext,
        "dispatch_kernel",
        "_cpp_ext.dispatch_kernel (internal commit)",
    ) or patched_any

    for class_name in ("BatchDispatch", "QueueManager"):
        klass = getattr(_metal_dispatch_ext, class_name, None)
        if klass is None:
            continue
        for method_name in ("commit", "commit_all"):
            patched_any = _patch_method(
                klass,
                method_name,
                f"_cpp_ext.{class_name}.{method_name}",
            ) or patched_any

    if not patched_any:
        _PATCH_WARNINGS.append("No patchable C++ commit surfaces were found in _cpp_ext")


def _patch_python_commit_surfaces() -> None:
    try:
        from metal_marlin import metal_dispatch
        from metal_marlin.trellis import async_dispatch
    except Exception as exc:
        _PATCH_WARNINGS.append(f"Failed importing python commit surfaces: {exc}")
        return

    _patch_function(metal_dispatch, "dispatch_kernel", "metal_dispatch.dispatch_kernel")
    _patch_function(metal_dispatch, "_blit_copy", "metal_dispatch._blit_copy")
    _patch_function(metal_dispatch, "_blit_copy_async", "metal_dispatch._blit_copy_async")

    _patch_method(metal_dispatch.MetalKernelLibrary, "commit_prefill", "MetalKernelLibrary.commit_prefill")
    _patch_method(metal_dispatch.MetalKernelLibrary, "commit_decode", "MetalKernelLibrary.commit_decode")

    _patch_method(
        async_dispatch.AsyncCommandBufferManager,
        "dispatch_immediate",
        "AsyncCommandBufferManager.dispatch_immediate",
    )
    _patch_method(
        async_dispatch.AsyncCommandBufferManager,
        "commit_batch",
        "AsyncCommandBufferManager.commit_batch",
    )
    _patch_method(
        async_dispatch.AsyncCommandBufferManager,
        "commit_and_wait",
        "AsyncCommandBufferManager.commit_and_wait",
    )


def _install_commit_tracing() -> None:
    handler = _BatchTraceCaptureHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)

    _patch_pyobjc_command_queue()
    _patch_cpp_extension()
    _patch_python_commit_surfaces()


def _write_trace_report(model_path: Path, failure: str | None) -> None:
    lines: list[str] = []
    lines.append("Command Buffer Commit Trace Report")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model path: {model_path}")
    lines.append(f"METAL_MARLIN_TRACE_BATCH={os.environ.get('METAL_MARLIN_TRACE_BATCH')}")
    lines.append(f"Total commit events: {len(_COMMIT_EVENTS)}")
    lines.append(f"Captured batch trace logs: {len(_BATCH_LOG_EVENTS)}")
    lines.append("")

    if _PATCH_WARNINGS:
        lines.append("Patch warnings:")
        for warning in _PATCH_WARNINGS:
            lines.append(f"- {warning}")
        lines.append("")

    if failure:
        lines.append("Failure:")
        lines.append(failure.rstrip())
        lines.append("")

    if _BATCH_LOG_EVENTS:
        lines.append("Batch trace logs:")
        for log_entry in _BATCH_LOG_EVENTS:
            lines.append(log_entry)
        lines.append("")

    if _COMMIT_EVENTS:
        for event in _COMMIT_EVENTS:
            lines.append("=" * 80)
            lines.append(f"Commit #{event['index']}: {event['source']}")
            lines.append(f"Timestamp: {event['time']}")
            lines.append("Stack trace:")
            lines.append(event["stack"].rstrip() or "<empty stack>")
            lines.append("")
    else:
        lines.append("No commit events captured.")

    _TRACE_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def _run_single_forward_pass() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required to trace Metal command buffer commits.")

    from metal_marlin.trellis.lm import TrellisForCausalLM

    model_path = _THIS_DIR / "fixtures" / "synthetic_trellis_smoke"
    if not model_path.exists():
        raise FileNotFoundError(f"Synthetic fixture not found: {model_path}")

    model = TrellisForCausalLM.from_pretrained(str(model_path), device="mps")
    input_ids = torch.randint(0, model.config.vocab_size, (1, 4), device="mps")

    with torch.no_grad():
        _ = model(input_ids)
    torch.mps.synchronize()


def main() -> int:
    model_path = _THIS_DIR / "fixtures" / "synthetic_trellis_smoke"
    failure: str | None = None

    _install_commit_tracing()
    try:
        _run_single_forward_pass()
    except Exception:
        failure = traceback.format_exc()
        logger.exception("Commit tracing run failed")
    finally:
        _write_trace_report(model_path=model_path, failure=failure)
        print(f"Wrote commit trace report: {_TRACE_OUTPUT}")

    return 0 if failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
