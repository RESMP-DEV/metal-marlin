from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from pathlib import Path

importlib.import_module("metal_marlin")

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "metal_marlin"
    / "serving"
    / "glm47_loader.py"
)
_SERVING_DIR = _MODULE_PATH.parent
_SERVING_PACKAGE = "metal_marlin.serving"
if _SERVING_PACKAGE not in sys.modules:
    serving_pkg = types.ModuleType(_SERVING_PACKAGE)
    serving_pkg.__path__ = [str(_SERVING_DIR)]  # type: ignore[attr-defined]
    sys.modules[_SERVING_PACKAGE] = serving_pkg

_SPEC = importlib.util.spec_from_file_location(
    f"{_SERVING_PACKAGE}.glm47_loader",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

DEFAULT_KV_QUANTIZE_MODE = _MODULE.DEFAULT_KV_QUANTIZE_MODE
GLM47Loader = _MODULE.GLM47Loader
load_glm47 = _MODULE.load_glm47


def test_glm47_loader_defaults_kv_quantization_to_int8() -> None:
    loader = GLM47Loader("contrib/metal_marlin/models/placeholder")
    assert loader.kv_quantize_mode == DEFAULT_KV_QUANTIZE_MODE


def test_load_glm47_signature_uses_default_kv_quantization() -> None:
    signature = inspect.signature(load_glm47)
    assert signature.parameters["kv_quantize_mode"].default == DEFAULT_KV_QUANTIZE_MODE
