"""Layer package shim with legacy `layers.py` compatibility.

Historically, Metal Marlin exposed layer classes from `metal_marlin/layers.py`.
This package keeps that API stable while allowing submodules like
`metal_marlin.layers.mmfp4_linear`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_legacy_layers_module() -> ModuleType:
    module_name = "metal_marlin._layers_legacy"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    legacy_path = Path(__file__).resolve().parent.parent / "layers.py"
    spec = importlib.util.spec_from_file_location(module_name, legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load legacy layers module from {legacy_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_legacy_layers = _load_legacy_layers_module()

MarlinLinear = _legacy_layers.MarlinLinear
MixedPrecisionLinear = _legacy_layers.MixedPrecisionLinear

from .mmfp4_linear import MMFP4Linear

try:
    from .mmfp4_mla import MMFP4MLA
except ImportError:
    MMFP4MLA = None

try:
    from .mmfp4_moe import MMFP4MoE
except ImportError:
    MMFP4MoE = None

__all__ = [
    "MarlinLinear",
    "MixedPrecisionLinear",
    "MMFP4Linear",
    "MMFP4MLA",
    "MMFP4MoE",
]
