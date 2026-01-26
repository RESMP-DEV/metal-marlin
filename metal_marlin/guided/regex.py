"""Regular expression to DFA to token masks for constrained generation.

This module implements regex-constrained generation by:
1. Compiling a regex pattern to a Deterministic Finite Automaton (DFA)
2. At each generation step, determining which tokens lead to valid DFA states
3. Masking invalid tokens to guide generation to match the pattern

The DFA approach is efficient because:
- State transitions are O(1) lookups
- Token validity can be cached per state (finite states)
- No backtracking needed during generation

Key insight: We don't need the token to complete a match, just to keep us
in a state from which a match is still possible. This means we accept
tokens that lead to any non-dead state in the DFA.

Implementation notes:
- Uses Python's `re` module for regex parsing (via sre_parse)
- Builds NFA from parsed regex, then converts to DFA via powerset construction
- Supports common regex features: ., *, +, ?, [], |, (), character classes
- Anchors (^, $) are implicit (full match required)
"""

from __future__ import annotations

import re
import sre_parse
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

from .logit_processor import BaseLogitProcessor, MaskCache, MaskingMode

# ---------------------------------------------------------------------------
# DFA data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DFAState:
    """A state in the DFA.

    States are identified by frozenset of NFA states (from powerset construction).
    We use integer IDs for efficiency and the frozen NFA state set for identity.
    """

    id: int
    nfa_states: frozenset[int]
    is_accepting: bool = False


@dataclass
class DFA:
    """Deterministic Finite Automaton for regex matching.

    The DFA is constructed from an NFA via powerset construction.
    Transitions are stored as a nested dict: transitions[state_id][char] = next_state_id
    """

    states: dict[int, DFAState]
    start_state: int
    accepting_states: set[int]
    transitions: dict[int, dict[str, int]]
    alphabet: set[str]
    # Dead state - transitions here mean the regex can never match
    dead_state: int | None = None

    def transition(self, state_id: int, char: str) -> int | None:
        """Get next state for transition on character.

        Args:
            state_id: Current state
            char: Input character

        Returns:
            Next state ID, or None if no valid transition
        """
        state_trans = self.transitions.get(state_id, {})
        return state_trans.get(char)

    def is_dead(self, state_id: int) -> bool:
        """Check if state is dead (no path to accepting state)."""
        return state_id == self.dead_state

    def can_accept_from(self, state_id: int) -> bool:
        """Check if an accepting state is reachable from given state.

        This is used to determine if we should continue generating or if
        we've entered a dead end.
        """
        if state_id in self.accepting_states:
            return True
        if self.is_dead(state_id):
            return False

        # BFS to find accepting state
        visited = {state_id}
        frontier = [state_id]
        while frontier:
            current = frontier.pop()
            for next_state in self.transitions.get(current, {}).values():
                if next_state in self.accepting_states:
                    return True
                if next_state not in visited and not self.is_dead(next_state):
                    visited.add(next_state)
                    frontier.append(next_state)
        return False


# ---------------------------------------------------------------------------
# NFA data structures and construction
# ---------------------------------------------------------------------------


@dataclass
class NFAFragment:
    """A fragment of an NFA with a single start and end state.

    Used during Thompson's construction to build NFA from regex.
    """

    start: int
    end: int


class NFA:
    """Non-deterministic Finite Automaton.

    Transitions can be on characters or epsilon (empty string).
    """

    def __init__(self) -> None:
        self.transitions: dict[int, dict[str | None, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.start_state: int = 0
        self.accepting_states: set[int] = set()
        self._next_state: int = 0

    def new_state(self) -> int:
        """Create a new state and return its ID."""
        state = self._next_state
        self._next_state += 1
        return state

    def add_transition(self, from_state: int, to_state: int, char: str | None) -> None:
        """Add a transition.

        Args:
            from_state: Source state
            to_state: Destination state
            char: Character for transition, or None for epsilon
        """
        self.transitions[from_state][char].add(to_state)

    def epsilon_closure(self, states: set[int]) -> frozenset[int]:
        """Compute epsilon closure of a set of states.

        Returns all states reachable via epsilon transitions.
        """
        closure = set(states)
        stack = list(states)
        while stack:
            state = stack.pop()
            for next_state in self.transitions[state].get(None, set()):
                if next_state not in closure:
                    closure.add(next_state)
                    stack.append(next_state)
        return frozenset(closure)


def _build_nfa_from_parsed(nfa: NFA, parsed: list, start: int) -> NFAFragment:
    """Build NFA fragment from sre_parse output.

    Args:
        nfa: NFA being built
        parsed: Parsed regex from sre_parse
        start: Starting state ID

    Returns:
        NFAFragment with start and end states
    """
    if not parsed:
        # Empty pattern - single state
        end = nfa.new_state()
        nfa.add_transition(start, end, None)
        return NFAFragment(start, end)

    # Process each element in sequence
    current_start = start
    fragments: list[NFAFragment] = []

    for op, av in parsed:
        if op == sre_parse.LITERAL:
            # Single character
            end = nfa.new_state()
            nfa.add_transition(current_start, end, chr(av))
            fragment = NFAFragment(current_start, end)
            current_start = end

        elif op == sre_parse.NOT_LITERAL:
            # Any character except this one
            end = nfa.new_state()
            # Add transitions for all printable chars except the one
            for c in range(32, 127):
                if c != av:
                    nfa.add_transition(current_start, end, chr(c))
            fragment = NFAFragment(current_start, end)
            current_start = end

        elif op == sre_parse.ANY:
            # Dot - any character except newline
            end = nfa.new_state()
            for c in range(32, 127):
                if c != ord('\n'):
                    nfa.add_transition(current_start, end, chr(c))
            fragment = NFAFragment(current_start, end)
            current_start = end

        elif op == sre_parse.IN:
            # Character class [...]
            end = nfa.new_state()
            chars = _expand_character_class(av)
            for c in chars:
                nfa.add_transition(current_start, end, c)
            fragment = NFAFragment(current_start, end)
            current_start = end

        elif op == sre_parse.BRANCH:
            # Alternation (a|b)
            branch_start = current_start
            branch_end = nfa.new_state()
            for branch in av[1]:  # av is (None, [branches])
                sub = _build_nfa_from_parsed(nfa, branch, nfa.new_state())
                nfa.add_transition(branch_start, sub.start, None)
                nfa.add_transition(sub.end, branch_end, None)
            fragment = NFAFragment(branch_start, branch_end)
            current_start = branch_end

        elif op == sre_parse.SUBPATTERN:
            # Group (...)
            sub = _build_nfa_from_parsed(nfa, av[3], current_start)
            fragment = sub
            current_start = sub.end

        elif op == sre_parse.MAX_REPEAT or op == sre_parse.MIN_REPEAT:
            # Repetition: *, +, ?, {n,m}
            min_count, max_count, subpattern = av

            if max_count == sre_parse.MAXREPEAT:
                # Unbounded: *, +
                if min_count == 0:
                    # * - zero or more
                    loop_start = nfa.new_state()
                    sub = _build_nfa_from_parsed(nfa, subpattern, nfa.new_state())
                    loop_end = nfa.new_state()
                    nfa.add_transition(current_start, loop_start, None)
                    nfa.add_transition(loop_start, sub.start, None)
                    nfa.add_transition(loop_start, loop_end, None)  # Skip
                    nfa.add_transition(sub.end, loop_start, None)  # Loop back
                    nfa.add_transition(sub.end, loop_end, None)
                    fragment = NFAFragment(current_start, loop_end)
                    current_start = loop_end
                else:
                    # + - one or more
                    sub = _build_nfa_from_parsed(nfa, subpattern, current_start)
                    loop_end = nfa.new_state()
                    nfa.add_transition(sub.end, sub.start, None)  # Loop back
                    nfa.add_transition(sub.end, loop_end, None)
                    fragment = NFAFragment(current_start, loop_end)
                    current_start = loop_end
            else:
                # Bounded: ?, {n,m}
                if min_count == 0 and max_count == 1:
                    # ? - zero or one
                    sub = _build_nfa_from_parsed(nfa, subpattern, nfa.new_state())
                    end = nfa.new_state()
                    nfa.add_transition(current_start, sub.start, None)
                    nfa.add_transition(current_start, end, None)  # Skip
                    nfa.add_transition(sub.end, end, None)
                    fragment = NFAFragment(current_start, end)
                    current_start = end
                else:
                    # General {n,m} - expand explicitly
                    # For simplicity, treat as {0,m} for now
                    end = current_start
                    for _ in range(max_count):
                        sub = _build_nfa_from_parsed(nfa, subpattern, nfa.new_state())
                        nfa.add_transition(end, sub.start, None)
                        nfa.add_transition(end, sub.end, None) if min_count == 0 else None
                        end = sub.end
                    fragment = NFAFragment(current_start, end)
                    current_start = end

        elif op == sre_parse.AT:
            # Anchors - ignore for token-level matching
            end = nfa.new_state()
            nfa.add_transition(current_start, end, None)
            fragment = NFAFragment(current_start, end)
            current_start = end

        elif op == sre_parse.CATEGORY:
            # Character categories like \d, \w, \s
            end = nfa.new_state()
            chars = _expand_category(av)
            for c in chars:
                nfa.add_transition(current_start, end, c)
            fragment = NFAFragment(current_start, end)
            current_start = end

        else:
            # Unknown op - skip with epsilon
            end = nfa.new_state()
            nfa.add_transition(current_start, end, None)
            fragment = NFAFragment(current_start, end)
            current_start = end

        fragments.append(fragment)

    # Return fragment spanning entire sequence
    if fragments:
        return NFAFragment(fragments[0].start, fragments[-1].end)
    else:
        end = nfa.new_state()
        nfa.add_transition(start, end, None)
        return NFAFragment(start, end)


def _expand_character_class(items: list) -> set[str]:
    """Expand a character class [...] into set of characters.

    Args:
        items: List from sre_parse for IN node

    Returns:
        Set of characters in the class
    """
    chars: set[str] = set()
    negate = False

    for op, av in items:
        if op == sre_parse.NEGATE:
            negate = True
        elif op == sre_parse.LITERAL:
            chars.add(chr(av))
        elif op == sre_parse.RANGE:
            lo, hi = av
            for c in range(lo, hi + 1):
                chars.add(chr(c))
        elif op == sre_parse.CATEGORY:
            chars |= _expand_category(av)

    if negate:
        # Return complement
        all_chars = set(chr(c) for c in range(32, 127))
        chars = all_chars - chars

    return chars


def _expand_category(category) -> set[str]:
    r"""Expand a regex category (\d, \w, etc.) to character set.

    Args:
        category: Category constant from sre_parse

    Returns:
        Set of characters in the category
    """
    if category == sre_parse.CATEGORY_DIGIT:
        return set("0123456789")
    elif category == sre_parse.CATEGORY_NOT_DIGIT:
        return set(chr(c) for c in range(32, 127)) - set("0123456789")
    elif category == sre_parse.CATEGORY_WORD:
        return set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    elif category == sre_parse.CATEGORY_NOT_WORD:
        word = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
        return set(chr(c) for c in range(32, 127)) - word
    elif category == sre_parse.CATEGORY_SPACE:
        return set(" \t\n\r\f\v")
    elif category == sre_parse.CATEGORY_NOT_SPACE:
        return set(chr(c) for c in range(32, 127)) - set(" \t\n\r\f\v")
    else:
        return set()


# ---------------------------------------------------------------------------
# NFA to DFA conversion
# ---------------------------------------------------------------------------


def _nfa_to_dfa(nfa: NFA, fragment: NFAFragment) -> DFA:
    """Convert NFA to DFA via powerset construction.

    Args:
        nfa: The NFA
        fragment: The NFA fragment representing the full pattern

    Returns:
        Equivalent DFA
    """
    # Compute alphabet from NFA transitions
    alphabet: set[str] = set()
    for state_trans in nfa.transitions.values():
        for char in state_trans.keys():
            if char is not None:
                alphabet.add(char)

    # Set accepting state
    nfa.accepting_states = {fragment.end}

    # Start state is epsilon closure of NFA start
    start_nfa_states = nfa.epsilon_closure({fragment.start})
    start_accepting = bool(start_nfa_states & nfa.accepting_states)

    dfa_states: dict[int, DFAState] = {}
    dfa_transitions: dict[int, dict[str, int]] = defaultdict(dict)
    accepting_states: set[int] = set()

    # Map from frozenset of NFA states to DFA state ID
    nfa_to_dfa_map: dict[frozenset[int], int] = {}
    next_dfa_id = 0

    # Create start state
    start_state = DFAState(next_dfa_id, start_nfa_states, start_accepting)
    dfa_states[next_dfa_id] = start_state
    nfa_to_dfa_map[start_nfa_states] = next_dfa_id
    if start_accepting:
        accepting_states.add(next_dfa_id)
    next_dfa_id += 1

    # BFS to build DFA
    worklist = [start_nfa_states]
    while worklist:
        current_nfa_states = worklist.pop()
        current_dfa_id = nfa_to_dfa_map[current_nfa_states]

        for char in alphabet:
            # Compute next NFA states on this character
            next_nfa_states: set[int] = set()
            for nfa_state in current_nfa_states:
                if char in nfa.transitions[nfa_state]:
                    next_nfa_states |= nfa.transitions[nfa_state][char]

            if not next_nfa_states:
                continue  # No transition on this character

            # Epsilon closure
            next_nfa_states_closed = nfa.epsilon_closure(next_nfa_states)

            if next_nfa_states_closed not in nfa_to_dfa_map:
                # New DFA state
                is_accepting = bool(next_nfa_states_closed & nfa.accepting_states)
                new_state = DFAState(next_dfa_id, next_nfa_states_closed, is_accepting)
                dfa_states[next_dfa_id] = new_state
                nfa_to_dfa_map[next_nfa_states_closed] = next_dfa_id
                if is_accepting:
                    accepting_states.add(next_dfa_id)
                worklist.append(next_nfa_states_closed)
                next_dfa_id += 1

            # Add DFA transition
            dfa_transitions[current_dfa_id][char] = nfa_to_dfa_map[next_nfa_states_closed]

    # Create dead state for transitions that lead nowhere
    dead_state = next_dfa_id
    dfa_states[dead_state] = DFAState(dead_state, frozenset(), False)

    return DFA(
        states=dfa_states,
        start_state=0,
        accepting_states=accepting_states,
        transitions=dict(dfa_transitions),
        alphabet=alphabet,
        dead_state=dead_state,
    )


def build_dfa_from_regex(pattern: str) -> DFA:
    """Build a DFA from a regex pattern string.

    Args:
        pattern: Regular expression pattern

    Returns:
        DFA that accepts strings matching the pattern
    """
    # Parse regex
    try:
        parsed = sre_parse.parse(pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {e}")

    # Build NFA
    nfa = NFA()
    start = nfa.new_state()
    fragment = _build_nfa_from_parsed(nfa, list(parsed), start)

    # Convert to DFA
    return _nfa_to_dfa(nfa, fragment)


# ---------------------------------------------------------------------------
# Regex processor
# ---------------------------------------------------------------------------


class Tokenizer(Protocol):
    """Protocol for tokenizers used in guided generation."""

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...

    @property
    def vocab_size(self) -> int: ...


class RegexProcessor(BaseLogitProcessor):
    """Constrain generation to strings matching a regular expression.

    This processor uses a DFA to track the match state and determines which
    tokens can legally follow to keep the DFA in a non-dead state.

    Usage:
        processor = RegexProcessor(r"[a-z]+@[a-z]+\\.com", tokenizer)

        # In generation loop:
        logits = processor(logits, generated_ids)
    """

    def __init__(
        self,
        pattern: str,
        tokenizer: Tokenizer,
        masking_mode: MaskingMode = MaskingMode.NEGATIVE_INF,
        partial_match: bool = True,
    ) -> None:
        """Initialize regex processor.

        Args:
            pattern: Regular expression pattern
            tokenizer: Tokenizer with encode/decode methods
            masking_mode: How to mask invalid tokens
            partial_match: If True, allow tokens that keep a match possible
                          If False, only allow tokens that extend a match
        """
        super().__init__(tokenizer.vocab_size, masking_mode)
        self.pattern = pattern
        self.tokenizer = tokenizer
        self.partial_match = partial_match

        # Build DFA
        self.dfa = build_dfa_from_regex(pattern)

        # Build token transition table
        self._build_token_tables()

        # Initialize state
        self._current_state = self.dfa.start_state
        self._generated_text = ""

        # Mask cache
        self._mask_cache = MaskCache(max_size=1024, vocab_size=self.vocab_size)

    def _build_token_tables(self) -> None:
        """Build mapping from tokens to DFA transitions they enable.

        For each token, we compute which DFA states it can transition from
        and to. This enables efficient lookup during generation.
        """
        # For each token, cache: {state_id: resulting_state_id}
        self._token_transitions: dict[int, dict[int, int | None]] = {}

        for token_id in range(self.vocab_size):
            try:
                token_text = self.tokenizer.decode([token_id])
                if not token_text:
                    continue

                # For each DFA state, compute where this token leads
                transitions: dict[int, int | None] = {}
                for state_id in self.dfa.states:
                    if self.dfa.is_dead(state_id):
                        transitions[state_id] = self.dfa.dead_state
                        continue

                    # Simulate consuming token character by character
                    current = state_id
                    valid = True
                    for char in token_text:
                        next_state = self.dfa.transition(current, char)
                        if next_state is None:
                            valid = False
                            break
                        current = next_state

                    transitions[state_id] = current if valid else self.dfa.dead_state

                self._token_transitions[token_id] = transitions

            except Exception:
                # Skip tokens that can't be decoded
                continue

    def reset(self) -> None:
        """Reset to start state for new generation."""
        self._current_state = self.dfa.start_state
        self._generated_text = ""

    def get_valid_tokens(self, generated_ids: list[int]) -> set[int]:
        """Get tokens that lead to valid DFA states.

        Args:
            generated_ids: Previously generated token IDs

        Returns:
            Set of valid token IDs
        """
        # Update state based on what was generated
        if generated_ids:
            last_token = generated_ids[-1]
            self._update_state(last_token)

        # Get valid tokens from current state
        return self._get_valid_from_state(self._current_state)

    def _update_state(self, token_id: int) -> None:
        """Update DFA state based on generated token.

        Args:
            token_id: The token that was just generated
        """
        if token_id in self._token_transitions:
            transitions = self._token_transitions[token_id]
            if self._current_state in transitions:
                next_state = transitions[self._current_state]
                if next_state is not None:
                    self._current_state = next_state

        # Also update text for debugging
        try:
            self._generated_text += self.tokenizer.decode([token_id])
        except Exception:
            pass

    def _get_valid_from_state(self, state_id: int) -> set[int]:
        """Get all tokens that lead to valid states from given state.

        Args:
            state_id: Current DFA state

        Returns:
            Set of valid token IDs
        """
        # Check cache
        def compute_valid() -> set[int]:
            valid: set[int] = set()

            for token_id, transitions in self._token_transitions.items():
                if state_id not in transitions:
                    continue

                next_state = transitions[state_id]
                if next_state is None or self.dfa.is_dead(next_state):
                    continue

                if self.partial_match:
                    # Allow if any path to accepting state
                    if self.dfa.can_accept_from(next_state):
                        valid.add(token_id)
                else:
                    # Only allow if extends current match
                    valid.add(token_id)

            return valid

        # Use cache
        mask = self._mask_cache.get_or_compute(state_id, compute_valid)

        # Convert mask back to set (for interface compatibility)
        # In practice, we'd return the mask directly to apply_logit_mask
        return {i for i in range(self.vocab_size) if mask[i]}

    def is_complete(self) -> bool:
        """Check if current state is accepting (full match)."""
        return self._current_state in self.dfa.accepting_states

    def can_complete(self) -> bool:
        """Check if an accepting state is still reachable."""
        return self.dfa.can_accept_from(self._current_state)
