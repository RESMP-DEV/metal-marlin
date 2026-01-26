"""Context-Free Grammar to token constraints for constrained generation.

This module implements grammar-constrained generation by:
1. Parsing a grammar specification (BNF/EBNF format)
2. Using an Earley parser to track valid parse states
3. At each step, determining which tokens can extend a valid partial parse
4. Masking invalid tokens to guide generation

Why CFG over regex/DFA?
- CFGs can express nested structures (balanced parens, XML, code)
- Many DSLs and programming languages are context-free
- More expressive than regular expressions

Key insight: We don't need to complete the parse, just know if a continuation
is possible. Earley parsing gives us exactly this - it tracks all valid parse
states in parallel and tells us which terminals can extend them.

Implementation notes:
- Uses Earley parsing algorithm for O(n³) worst case, O(n²) typical
- Supports BNF grammar format with extensions for character classes
- Terminals can be strings or character class patterns
- Precomputes terminal-to-token mapping for efficient lookup
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol

from .logit_processor import BaseLogitProcessor, MaskingMode

# ---------------------------------------------------------------------------
# Grammar data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CFGRule:
    """A production rule in a context-free grammar.

    A rule has a left-hand side (non-terminal) and right-hand side
    (sequence of terminals and non-terminals).
    """

    lhs: str  # Non-terminal being defined
    rhs: tuple[str, ...]  # Sequence of symbols (terminals or non-terminals)

    def __repr__(self) -> str:
        return f"{self.lhs} -> {' '.join(self.rhs) if self.rhs else 'ε'}"


@dataclass
class Grammar:
    """Context-free grammar.

    Consists of:
    - Rules: Production rules
    - Start symbol: Non-terminal that must be derived
    - Terminals: Symbols that appear in output
    - Non-terminals: Symbols that expand via rules
    """

    rules: list[CFGRule]
    start_symbol: str
    terminals: set[str]
    non_terminals: set[str]

    def rules_for(self, non_terminal: str) -> list[CFGRule]:
        """Get all rules with given non-terminal on LHS."""
        return [r for r in self.rules if r.lhs == non_terminal]

    def is_terminal(self, symbol: str) -> bool:
        """Check if symbol is a terminal."""
        return symbol in self.terminals

    def is_non_terminal(self, symbol: str) -> bool:
        """Check if symbol is a non-terminal."""
        return symbol in self.non_terminals


# ---------------------------------------------------------------------------
# Grammar parsing (BNF format)
# ---------------------------------------------------------------------------


def parse_bnf_grammar(text: str, start_symbol: str | None = None) -> Grammar:
    """Parse a BNF grammar specification.

    Format:
        <non_terminal> ::= symbol1 symbol2 | symbol3
        <another> ::= "literal" <non_terminal>

    Conventions:
    - Non-terminals in angle brackets: <name>
    - Literals in quotes: "text" or 'text'
    - Alternation with |
    - Whitespace separates symbols

    Special terminals:
    - [a-z] - character class (regex-style)
    - \\d, \\w, \\s - regex character classes
    - . - any character

    Args:
        text: Grammar specification
        start_symbol: Starting non-terminal (defaults to first rule's LHS)

    Returns:
        Parsed Grammar object
    """
    rules: list[CFGRule] = []
    terminals: set[str] = set()
    non_terminals: set[str] = set()

    # Remove comments and blank lines
    lines = []
    for line in text.strip().split('\n'):
        line = line.split('#')[0].strip()
        if line:
            lines.append(line)

    # Join continuation lines
    full_text = ' '.join(lines)

    # Parse rules
    rule_pattern = re.compile(r'<(\w+)>\s*::=\s*(.+?)(?=<\w+>\s*::=|$)')
    for match in rule_pattern.finditer(full_text):
        lhs = match.group(1)
        rhs_text = match.group(2)
        non_terminals.add(lhs)

        # Split alternatives
        alternatives = rhs_text.split('|')
        for alt in alternatives:
            symbols = _parse_rhs(alt.strip())
            for sym in symbols:
                if sym.startswith('<') and sym.endswith('>'):
                    non_terminals.add(sym[1:-1])
                else:
                    terminals.add(sym)
            # Store rule with angle brackets stripped from non-terminals
            processed_rhs = tuple(
                s[1:-1] if s.startswith('<') and s.endswith('>') else s
                for s in symbols
            )
            rules.append(CFGRule(lhs, processed_rhs))

    # Determine start symbol
    if start_symbol is None:
        if rules:
            start_symbol = rules[0].lhs
        else:
            raise ValueError("Empty grammar")

    # Convert non-terminal references to plain names
    processed_terminals = {t for t in terminals if not (t.startswith('<') and t.endswith('>'))}

    return Grammar(
        rules=rules,
        start_symbol=start_symbol,
        terminals=processed_terminals,
        non_terminals=non_terminals,
    )


def _parse_rhs(text: str) -> list[str]:
    """Parse right-hand side of a production rule.

    Handles:
    - Quoted strings: "text" or 'text'
    - Non-terminals: <name>
    - Character classes: [a-z]
    - Bare words (terminals)
    """
    symbols: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        # Skip whitespace
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break

        if text[i] == '"' or text[i] == "'":
            # Quoted string
            quote = text[i]
            i += 1
            start = i
            while i < n and text[i] != quote:
                if text[i] == '\\':
                    i += 2
                else:
                    i += 1
            symbols.append(text[start:i])
            i += 1  # Skip closing quote

        elif text[i] == '<':
            # Non-terminal
            start = i
            while i < n and text[i] != '>':
                i += 1
            symbols.append(text[start:i + 1])
            i += 1

        elif text[i] == '[':
            # Character class
            start = i
            while i < n and text[i] != ']':
                i += 1
            symbols.append(text[start:i + 1])
            i += 1

        elif text[i] == '\\':
            # Escape sequence (\d, \w, etc.)
            symbols.append(text[i:i + 2])
            i += 2

        elif text[i] == '.':
            # Any character
            symbols.append('.')
            i += 1

        else:
            # Bare word
            start = i
            while i < n and not text[i].isspace() and text[i] not in '"\'<>[].\\|':
                i += 1
            if start < i:
                symbols.append(text[start:i])

    return symbols


# ---------------------------------------------------------------------------
# Earley parser for grammar constraint checking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EarleyItem:
    """An item in the Earley parser.

    Represents a partial parse of a rule: how much of the RHS has been matched.
    """

    rule: CFGRule
    dot: int  # Position in RHS (0 = nothing matched, len(rhs) = complete)
    start: int  # Position in input where this rule started

    @property
    def is_complete(self) -> bool:
        """Check if rule is fully matched."""
        return self.dot >= len(self.rule.rhs)

    @property
    def next_symbol(self) -> str | None:
        """Get symbol after dot, or None if complete."""
        if self.dot < len(self.rule.rhs):
            return self.rule.rhs[self.dot]
        return None

    def advance(self) -> EarleyItem:
        """Return new item with dot advanced."""
        return EarleyItem(self.rule, self.dot + 1, self.start)

    def __repr__(self) -> str:
        rhs_with_dot = list(self.rule.rhs)
        rhs_with_dot.insert(self.dot, '•')
        return f"[{self.start}] {self.rule.lhs} -> {' '.join(rhs_with_dot)}"


class GrammarState(Enum):
    """State of grammar parsing."""

    PARSING = auto()  # In progress
    COMPLETE = auto()  # Full valid parse
    STUCK = auto()  # No valid continuation


@dataclass
class EarleyParser:
    """Earley parser for tracking valid parse states.

    The parser maintains a chart of item sets, where each set contains
    all valid partial parses at that position in the input.
    """

    grammar: Grammar
    chart: list[set[EarleyItem]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize chart with start items."""
        self.reset()

    def reset(self) -> None:
        """Reset parser to initial state."""
        self.chart = [set()]
        # Add all rules for start symbol
        for rule in self.grammar.rules_for(self.grammar.start_symbol):
            self.chart[0].add(EarleyItem(rule, 0, 0))
        # Close under prediction
        self._close_set(0)

    def _close_set(self, pos: int) -> None:
        """Close item set under prediction (expand non-terminals)."""
        added = True
        while added:
            added = False
            items_to_add: set[EarleyItem] = set()

            for item in self.chart[pos]:
                next_sym = item.next_symbol
                if next_sym and self.grammar.is_non_terminal(next_sym):
                    # Predict: add items for all rules expanding this non-terminal
                    for rule in self.grammar.rules_for(next_sym):
                        new_item = EarleyItem(rule, 0, pos)
                        if new_item not in self.chart[pos]:
                            items_to_add.add(new_item)

            if items_to_add:
                self.chart[pos].update(items_to_add)
                added = True

    def scan(self, terminal: str) -> bool:
        """Advance parser by consuming a terminal.

        Args:
            terminal: The terminal symbol to consume

        Returns:
            True if terminal was accepted, False if no valid parse
        """
        pos = len(self.chart) - 1
        new_items: set[EarleyItem] = set()

        # Scan: advance items expecting this terminal
        for item in self.chart[pos]:
            if item.next_symbol == terminal:
                new_items.add(item.advance())

        if not new_items:
            return False

        # Add new chart set
        self.chart.append(new_items)
        new_pos = len(self.chart) - 1

        # Complete and predict
        self._complete_and_predict(new_pos)

        return True

    def _complete_and_predict(self, pos: int) -> None:
        """Complete and predict until fixed point."""
        added = True
        while added:
            added = False
            items_to_add: set[EarleyItem] = set()

            for item in self.chart[pos]:
                if item.is_complete:
                    # Complete: advance items waiting for this non-terminal
                    for waiting in self.chart[item.start]:
                        if waiting.next_symbol == item.rule.lhs:
                            new_item = waiting.advance()
                            if new_item not in self.chart[pos]:
                                items_to_add.add(new_item)
                else:
                    # Predict: add items for non-terminals
                    next_sym = item.next_symbol
                    if next_sym and self.grammar.is_non_terminal(next_sym):
                        for rule in self.grammar.rules_for(next_sym):
                            new_item = EarleyItem(rule, 0, pos)
                            if new_item not in self.chart[pos]:
                                items_to_add.add(new_item)

            if items_to_add:
                self.chart[pos].update(items_to_add)
                added = True

    def get_expected_terminals(self) -> set[str]:
        """Get terminals that can extend the current parse.

        Returns:
            Set of terminal symbols that are valid next
        """
        pos = len(self.chart) - 1
        expected: set[str] = set()

        for item in self.chart[pos]:
            next_sym = item.next_symbol
            if next_sym and self.grammar.is_terminal(next_sym):
                expected.add(next_sym)

        return expected

    def is_complete(self) -> bool:
        """Check if parse is complete (accepts empty continuation)."""
        pos = len(self.chart) - 1
        for item in self.chart[pos]:
            if (item.is_complete and
                item.rule.lhs == self.grammar.start_symbol and
                item.start == 0):
                return True
        return False

    def can_continue(self) -> bool:
        """Check if there are any valid continuations."""
        return bool(self.get_expected_terminals())


# ---------------------------------------------------------------------------
# Grammar processor
# ---------------------------------------------------------------------------


class Tokenizer(Protocol):
    """Protocol for tokenizers used in guided generation."""

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...

    @property
    def vocab_size(self) -> int: ...


class GrammarProcessor(BaseLogitProcessor):
    """Constrain generation to strings matching a context-free grammar.

    This processor uses an Earley parser to track valid parse states and
    determines which tokens can legally extend the parse.

    Usage:
        grammar_text = '''
        <expr> ::= <term> | <expr> "+" <term>
        <term> ::= <factor> | <term> "*" <factor>
        <factor> ::= <number> | "(" <expr> ")"
        <number> ::= [0-9]+
        '''
        processor = GrammarProcessor(grammar_text, tokenizer)

        # In generation loop:
        logits = processor(logits, generated_ids)
    """

    def __init__(
        self,
        grammar_text: str | None = None,
        grammar: Grammar | None = None,
        tokenizer: Tokenizer | None = None,
        masking_mode: MaskingMode = MaskingMode.NEGATIVE_INF,
    ) -> None:
        """Initialize grammar processor.

        Args:
            grammar_text: BNF grammar specification (parsed if provided)
            grammar: Pre-parsed Grammar object (alternative to grammar_text)
            tokenizer: Tokenizer with encode/decode methods
            masking_mode: How to mask invalid tokens
        """
        if grammar is not None:
            self.grammar = grammar
        elif grammar_text is not None:
            self.grammar = parse_bnf_grammar(grammar_text)
        else:
            raise ValueError("Must provide grammar_text or grammar")

        if tokenizer is None:
            raise ValueError("Must provide tokenizer")

        super().__init__(tokenizer.vocab_size, masking_mode)
        self.tokenizer = tokenizer

        # Build token-terminal mapping
        self._build_token_tables()

        # Initialize parser
        self._parser = EarleyParser(self.grammar)

    def _build_token_tables(self) -> None:
        """Build mapping from terminals to tokens.

        For each terminal in the grammar, find tokens that:
        1. Match the terminal exactly
        2. Start with the terminal (partial match)
        3. Can produce the terminal (character class match)
        """
        self._terminal_to_tokens: dict[str, set[int]] = defaultdict(set)
        self._token_to_terminals: dict[int, set[str]] = defaultdict(set)

        for token_id in range(self.vocab_size):
            try:
                token_text = self.tokenizer.decode([token_id])
                if not token_text:
                    continue

                for terminal in self.grammar.terminals:
                    if self._matches_terminal(token_text, terminal):
                        self._terminal_to_tokens[terminal].add(token_id)
                        self._token_to_terminals[token_id].add(terminal)

            except Exception:
                continue

    def _matches_terminal(self, token_text: str, terminal: str) -> bool:
        """Check if token text matches or could extend a terminal.

        Args:
            token_text: Decoded token string
            terminal: Grammar terminal

        Returns:
            True if token matches terminal
        """
        # Exact match
        if token_text == terminal:
            return True

        # Token is prefix of terminal
        if terminal.startswith(token_text):
            return True

        # Terminal is prefix of token (token extends terminal)
        if token_text.startswith(terminal):
            return True

        # Character class matching
        if terminal.startswith('[') and terminal.endswith(']'):
            pattern = f"^{terminal}$"
            return any(re.match(pattern, c) for c in token_text)

        # Regex escape sequences
        if terminal.startswith('\\'):
            if terminal == '\\d':
                return any(c.isdigit() for c in token_text)
            elif terminal == '\\w':
                return any(c.isalnum() or c == '_' for c in token_text)
            elif terminal == '\\s':
                return any(c.isspace() for c in token_text)

        # Any character
        if terminal == '.':
            return len(token_text) > 0

        return False

    def reset(self) -> None:
        """Reset parser for new generation."""
        self._parser.reset()

    def get_valid_tokens(self, generated_ids: list[int]) -> set[int]:
        """Get tokens that can extend the current parse.

        Args:
            generated_ids: Previously generated token IDs

        Returns:
            Set of valid token IDs
        """
        # Update parser state based on last token
        if generated_ids:
            last_token = generated_ids[-1]
            self._consume_token(last_token)

        # Get expected terminals
        expected = self._parser.get_expected_terminals()

        # Map to token IDs
        valid: set[int] = set()
        for terminal in expected:
            valid |= self._terminal_to_tokens.get(terminal, set())

        return valid

    def _consume_token(self, token_id: int) -> None:
        """Update parser state by consuming token's terminals.

        Args:
            token_id: Token that was generated
        """
        # Get terminals this token produces
        terminals = self._token_to_terminals.get(token_id, set())

        # Try each terminal
        for terminal in terminals:
            if self._parser.scan(terminal):
                break  # Found a valid parse

    def get_state(self) -> GrammarState:
        """Get current parsing state."""
        if self._parser.is_complete():
            return GrammarState.COMPLETE
        elif self._parser.can_continue():
            return GrammarState.PARSING
        else:
            return GrammarState.STUCK


# ---------------------------------------------------------------------------
# Common grammar templates
# ---------------------------------------------------------------------------


JSON_GRAMMAR = """
<value> ::= <object> | <array> | <string> | <number> | "true" | "false" | "null"
<object> ::= "{" <members> "}" | "{" "}"
<members> ::= <pair> | <pair> "," <members>
<pair> ::= <string> ":" <value>
<array> ::= "[" <elements> "]" | "[" "]"
<elements> ::= <value> | <value> "," <elements>
<string> ::= '"' <chars> '"'
<chars> ::= <char> <chars> |
<char> ::= [a-zA-Z0-9 _]
<number> ::= <int> | <int> <frac>
<int> ::= <digit> | <digit> <int>
<frac> ::= "." <int>
<digit> ::= [0-9]
"""


ARITHMETIC_GRAMMAR = """
<expr> ::= <term> | <expr> "+" <term> | <expr> "-" <term>
<term> ::= <factor> | <term> "*" <factor> | <term> "/" <factor>
<factor> ::= <number> | "(" <expr> ")"
<number> ::= <digit> | <digit> <number>
<digit> ::= [0-9]
"""


SQL_SELECT_GRAMMAR = """
<select> ::= "SELECT" <columns> "FROM" <table> <where>
<columns> ::= <column> | <column> "," <columns> | "*"
<column> ::= <identifier>
<table> ::= <identifier>
<where> ::= "WHERE" <condition> |
<condition> ::= <column> <op> <value>
<op> ::= "=" | "!=" | "<" | ">" | "<=" | ">="
<value> ::= <string> | <number>
<identifier> ::= [a-zA-Z_][a-zA-Z0-9_]*
<string> ::= "'" <chars> "'"
<chars> ::= [a-zA-Z0-9 ]*
<number> ::= [0-9]+
"""
