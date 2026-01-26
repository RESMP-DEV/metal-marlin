"""Speculative decoding: draft-then-verify for faster autoregressive inference."""

from .draft import DraftModel, DraftOutput, NGramDraft, SmallModelDraft
from .engine import GenerationStats, SpeculativeConfig, SpeculativeEngine, StepResult
from .verify import VerifyResult, verify_speculative

__all__ = [
    "DraftModel",
    "DraftOutput",
    "GenerationStats",
    "NGramDraft",
    "SmallModelDraft",
    "SpeculativeConfig",
    "SpeculativeEngine",
    "StepResult",
    "VerifyResult",
    "verify_speculative",
]
