"""Speculative decoding: draft-then-verify for faster autoregressive inference."""

from .draft import DraftModel, DraftOutput, NGramDraft, SmallModelDraft
from .eagle import EagleHead, TreeDraftOutput
from .engine import GenerationStats, SpeculativeConfig, SpeculativeEngine, StepResult
from .mmfp4_draft import MMFP4DraftModel, MMFP4DraftModelWithTarget
from .mtp_draft import MTPDraft
from .token_acceptance import (
    AcceptanceResult,
    AcceptanceStats,
    TokenAcceptanceTracker,
    compute_acceptance_probabilities,
    create_acceptance_report,
    estimate_optimal_speculation_length,
)
from .verify import (
    EagleTreeVerifyResult,
    VerifyResult,
    verify_eagle_tree,
    verify_speculative,
)

__all__ = [
    "AcceptanceResult",
    "AcceptanceStats",
    "DraftModel",
    "DraftOutput",
    "EagleHead",
    "EagleTreeVerifyResult",
    "GenerationStats",
    "MMFP4DraftModel",
    "MMFP4DraftModelWithTarget",
    "MTPDraft",
    "NGramDraft",
    "SmallModelDraft",
    "SpeculativeConfig",
    "SpeculativeEngine",
    "StepResult",
    "TokenAcceptanceTracker",
    "TreeDraftOutput",
    "VerifyResult",
    "compute_acceptance_probabilities",
    "create_acceptance_report",
    "estimate_optimal_speculation_length",
    "verify_eagle_tree",
    "verify_speculative",
]
