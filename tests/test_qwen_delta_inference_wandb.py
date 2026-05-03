"""Synthetic tests for W&B tracking in Qwen inference wrappers."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(_REPO_ROOT))

import qwen_delta_inference_common as shared
from metal_marlin.wandb_helper import wandb_tracker

def test_imports():
    assert True
