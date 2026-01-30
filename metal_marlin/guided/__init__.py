"""Guided/constrained generation for structured outputs.

This module provides logit processors that constrain token generation to
match specific patterns: JSON schemas, regular expressions, or grammars.

The core approach:
1. Build a constraint state machine (FSM/DFA/parser) at init time
2. During generation, query which tokens are valid from current state
3. Mask invalid tokens to -inf before sampling
4. Update state after each generated token

Key exports:
- LogitProcessor: Base protocol for all constraint processors
- LogitProcessorList: Chain multiple processors together
- JSONSchemaProcessor: Constrain output to valid JSON matching a schema
- RegexProcessor: Constrain output to match a regular expression
- GrammarProcessor: Constrain output to match a context-free grammar
- apply_logit_mask: Metal-accelerated logit masking (when MLX available)

Example usage:
    from metal_marlin.guided import JSONSchemaProcessor, LogitProcessorList

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    processor = JSONSchemaProcessor(schema, tokenizer)

    # In generation loop:
    logits = processor(logits, generated_ids)
"""

from __future__ import annotations

from .grammar import (
    CFGRule,
    GrammarProcessor,
    GrammarState,
    parse_bnf_grammar,
)
from .json_schema import (
    JSONSchemaProcessor,
    JSONState,
)
from .logit_processor import (
    LogitProcessor,
    LogitProcessorList,
    MaskingMode,
    apply_logit_mask,
)
from .regex import (
    DFAState,
    RegexProcessor,
    build_dfa_from_regex,
)

__all__ = [
    # Base classes
    "LogitProcessor",
    "LogitProcessorList",
    "MaskingMode",
    "apply_logit_mask",
    # JSON Schema
    "JSONSchemaProcessor",
    "JSONState",
    # Regex
    "RegexProcessor",
    "DFAState",
    "build_dfa_from_regex",
    # Grammar
    "GrammarProcessor",
    "GrammarState",
    "CFGRule",
    "parse_bnf_grammar",
]
