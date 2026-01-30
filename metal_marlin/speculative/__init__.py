"""Speculative decoding: draft-then-verify for faster autoregressive inference."""

from .draft import DraftModel, DraftOutput, NGramDraft, SmallModelDraft
from .eagle import EagleHead, TreeDraftOutput
from .engine import GenerationStats, SpeculativeConfig, SpeculativeEngine, StepResult
from .verify import (
    EagleTreeVerifyResult,
    VerifyResult,
    verify_eagle_tree,
    verify_speculative,
)

__all__ = [
    "DraftModel",
    "DraftOutput",
    "EagleHead",
    "EagleTreeVerifyResult",
    "GenerationStats",
    "NGramDraft",
    "SmallModelDraft",
    "SpeculativeConfig",
    "SpeculativeEngine",
    "StepResult",
    "TreeDraftOutput",
    "VerifyResult",
    "verify_eagle_tree",
    "verify_speculative",
]
