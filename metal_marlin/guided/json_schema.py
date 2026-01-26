"""JSON Schema to token constraints for structured generation.

This module implements constrained generation for JSON output that conforms
to a given JSON Schema. The approach:

1. Parse the JSON Schema into a structural representation
2. Track parsing state (where are we in the JSON structure?)
3. At each step, determine which characters/tokens can legally follow
4. Map valid characters to valid tokens via tokenizer lookup

Key insight from jsonformer/outlines: JSON has a regular structure that
can be represented as a state machine. At any point, we know:
- Are we expecting a key, value, comma, colon, bracket, etc.?
- If expecting a value, what type? (string, number, boolean, array, object)
- Have we satisfied required fields?

This enables token-level guidance without full backtracking.

References:
- outlines: https://github.com/outlines-dev/outlines
- jsonformer: https://github.com/1rgs/jsonformer
- guidance: https://github.com/guidance-ai/guidance
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol

from .logit_processor import BaseLogitProcessor, MaskingMode


class JSONValueType(Enum):
    """JSON value types from JSON Schema."""

    STRING = auto()
    NUMBER = auto()
    INTEGER = auto()
    BOOLEAN = auto()
    NULL = auto()
    ARRAY = auto()
    OBJECT = auto()
    ANY = auto()  # No type constraint


@dataclass
class SchemaNode:
    """Parsed representation of a JSON Schema node.

    Captures the essential constraints for generation:
    - Type of value expected
    - For objects: properties, required fields
    - For arrays: item schema
    - For strings: pattern, enum, min/max length
    - For numbers: minimum, maximum, multipleOf
    """

    value_type: JSONValueType
    properties: dict[str, SchemaNode] = field(default_factory=dict)
    required: set[str] = field(default_factory=set)
    items: SchemaNode | None = None
    enum: list[Any] | None = None
    pattern: str | None = None
    min_length: int | None = None
    max_length: int | None = None
    minimum: float | None = None
    maximum: float | None = None
    additional_properties: bool = True
    const: Any | None = None


def parse_schema(schema: dict[str, Any]) -> SchemaNode:
    """Parse a JSON Schema dict into a SchemaNode tree.

    Args:
        schema: JSON Schema as a dictionary

    Returns:
        SchemaNode representing the schema constraints
    """
    # Determine type
    type_str = schema.get("type", "any")
    if isinstance(type_str, list):
        # Union type - for now, use ANY (could be improved)
        value_type = JSONValueType.ANY
    else:
        type_map = {
            "string": JSONValueType.STRING,
            "number": JSONValueType.NUMBER,
            "integer": JSONValueType.INTEGER,
            "boolean": JSONValueType.BOOLEAN,
            "null": JSONValueType.NULL,
            "array": JSONValueType.ARRAY,
            "object": JSONValueType.OBJECT,
            "any": JSONValueType.ANY,
        }
        value_type = type_map.get(type_str, JSONValueType.ANY)

    node = SchemaNode(value_type=value_type)

    # Object properties
    if "properties" in schema:
        for name, prop_schema in schema["properties"].items():
            node.properties[name] = parse_schema(prop_schema)

    # Required fields
    if "required" in schema:
        node.required = set(schema["required"])

    # Array items
    if "items" in schema:
        node.items = parse_schema(schema["items"])

    # String constraints
    node.enum = schema.get("enum")
    node.pattern = schema.get("pattern")
    node.min_length = schema.get("minLength")
    node.max_length = schema.get("maxLength")

    # Number constraints
    node.minimum = schema.get("minimum")
    node.maximum = schema.get("maximum")

    # Object constraints
    node.additional_properties = schema.get("additionalProperties", True)
    node.const = schema.get("const")

    return node


class JSONState(Enum):
    """States in the JSON parsing state machine."""

    START = auto()  # Beginning of JSON value
    OBJECT_OPEN = auto()  # Just saw {
    OBJECT_KEY = auto()  # Expecting a key (string)
    OBJECT_COLON = auto()  # Expecting :
    OBJECT_VALUE = auto()  # Expecting a value
    OBJECT_COMMA = auto()  # Expecting , or }
    ARRAY_OPEN = auto()  # Just saw [
    ARRAY_VALUE = auto()  # Expecting array element
    ARRAY_COMMA = auto()  # Expecting , or ]
    STRING_CONTENT = auto()  # Inside a string
    STRING_ESCAPE = auto()  # After backslash in string
    NUMBER_INT = auto()  # In integer part of number
    NUMBER_FRAC = auto()  # In fractional part
    NUMBER_EXP = auto()  # In exponent part
    DONE = auto()  # Complete valid JSON


@dataclass
class JSONParseContext:
    """Tracks the current position in JSON structure during parsing.

    This is a stack-based parser context that knows:
    - Current parse state
    - Schema node we're validating against
    - For objects: which keys we've seen, current key being parsed
    - For arrays: how many elements we've seen
    - For strings: content accumulated so far
    """

    state: JSONState
    schema: SchemaNode
    parent: JSONParseContext | None = None
    # Object tracking
    seen_keys: set[str] = field(default_factory=set)
    current_key: str = ""
    # Array tracking
    array_index: int = 0
    # String tracking
    string_buffer: str = ""
    # Number tracking
    number_buffer: str = ""


class Tokenizer(Protocol):
    """Protocol for tokenizers used in guided generation."""

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...

    @property
    def vocab_size(self) -> int: ...


class JSONSchemaProcessor(BaseLogitProcessor):
    """Constrain generation to valid JSON matching a schema.

    This processor tracks JSON parsing state and determines which tokens
    can legally follow based on:
    1. JSON syntax (braces, brackets, colons, commas, quotes)
    2. Schema constraints (types, required fields, patterns)
    3. Tokenizer vocabulary (which tokens produce valid continuations)

    Usage:
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        processor = JSONSchemaProcessor(schema, tokenizer)

        # In generation loop:
        logits = processor(logits, generated_ids)
    """

    def __init__(
        self,
        schema: dict[str, Any],
        tokenizer: Tokenizer,
        masking_mode: MaskingMode = MaskingMode.NEGATIVE_INF,
        strict: bool = True,
    ) -> None:
        """Initialize JSON Schema processor.

        Args:
            schema: JSON Schema as dictionary
            tokenizer: Tokenizer with encode/decode methods
            masking_mode: How to mask invalid tokens
            strict: If True, enforce all schema constraints strictly
        """
        super().__init__(tokenizer.vocab_size, masking_mode)
        self.schema = parse_schema(schema)
        self.tokenizer = tokenizer
        self.strict = strict

        # Build token lookup tables
        self._build_token_tables()

        # Initialize parse state
        self._context = JSONParseContext(
            state=JSONState.START,
            schema=self.schema,
        )
        self._generated_text = ""

    def _build_token_tables(self) -> None:
        """Build lookup tables mapping characters to tokens.

        This is the key optimization: precompute which tokens can produce
        each character/character class. At generation time, we just look
        up valid tokens instead of scanning the whole vocabulary.
        """
        self._char_to_tokens: dict[str, set[int]] = {}
        self._prefix_tokens: dict[str, set[int]] = {}

        # Characters we care about for JSON structure
        structural_chars = '{}[]:,"'
        whitespace = " \t\n\r"
        digit_chars = "0123456789"
        sign_chars = "+-"
        exp_chars = "eE"
        bool_chars = "truefalsn"  # true, false, null

        set(structural_chars + whitespace + digit_chars + sign_chars + exp_chars + bool_chars)

        # Scan vocabulary for tokens that start with each character
        for token_id in range(self.vocab_size):
            try:
                token_text = self.tokenizer.decode([token_id])
                if not token_text:
                    continue

                first_char = token_text[0]
                if first_char not in self._char_to_tokens:
                    self._char_to_tokens[first_char] = set()
                self._char_to_tokens[first_char].add(token_id)

                # Also track tokens that are exact matches for common patterns
                if token_text in ('true', 'false', 'null', '{', '}', '[', ']', ':', ',', '"'):
                    if token_text not in self._prefix_tokens:
                        self._prefix_tokens[token_text] = set()
                    self._prefix_tokens[token_text].add(token_id)

            except Exception:
                # Skip tokens that can't be decoded
                continue

        # Precompute common token sets
        self._whitespace_tokens: set[int] = set()
        for c in whitespace:
            self._whitespace_tokens |= self._char_to_tokens.get(c, set())

        self._digit_tokens: set[int] = set()
        for c in digit_chars:
            self._digit_tokens |= self._char_to_tokens.get(c, set())

        self._string_start_tokens = self._char_to_tokens.get('"', set())
        self._object_start_tokens = self._char_to_tokens.get('{', set())
        self._array_start_tokens = self._char_to_tokens.get('[', set())
        self._colon_tokens = self._char_to_tokens.get(':', set())
        self._comma_tokens = self._char_to_tokens.get(',', set())
        self._object_end_tokens = self._char_to_tokens.get('}', set())
        self._array_end_tokens = self._char_to_tokens.get(']', set())

    def reset(self) -> None:
        """Reset state for new generation."""
        self._context = JSONParseContext(
            state=JSONState.START,
            schema=self.schema,
        )
        self._generated_text = ""

    def get_valid_tokens(self, generated_ids: list[int]) -> set[int]:
        """Determine which tokens are valid given current JSON parse state.

        Args:
            generated_ids: Previously generated token IDs

        Returns:
            Set of valid token IDs
        """
        # Update state based on what was generated
        if generated_ids:
            last_token = generated_ids[-1]
            self._update_state(last_token)

        # Get valid tokens based on current state
        return self._get_valid_for_state()

    def _update_state(self, token_id: int) -> None:
        """Update parse state based on newly generated token.

        Args:
            token_id: The token that was just generated
        """
        token_text = self.tokenizer.decode([token_id])
        self._generated_text += token_text

        # Process each character in the token
        for char in token_text:
            self._process_char(char)

    def _process_char(self, char: str) -> None:
        """Update state machine based on a single character.

        Args:
            char: Character to process
        """
        ctx = self._context
        state = ctx.state

        # Skip whitespace in most states
        if char in ' \t\n\r' and state not in (JSONState.STRING_CONTENT, JSONState.STRING_ESCAPE):
            return

        if state == JSONState.START:
            self._handle_start(char)
        elif state == JSONState.OBJECT_OPEN:
            self._handle_object_open(char)
        elif state == JSONState.OBJECT_KEY:
            self._handle_object_key(char)
        elif state == JSONState.OBJECT_COLON:
            self._handle_object_colon(char)
        elif state == JSONState.OBJECT_VALUE:
            self._handle_object_value(char)
        elif state == JSONState.OBJECT_COMMA:
            self._handle_object_comma(char)
        elif state == JSONState.ARRAY_OPEN:
            self._handle_array_open(char)
        elif state == JSONState.ARRAY_VALUE:
            self._handle_array_value(char)
        elif state == JSONState.ARRAY_COMMA:
            self._handle_array_comma(char)
        elif state == JSONState.STRING_CONTENT:
            self._handle_string_content(char)
        elif state == JSONState.STRING_ESCAPE:
            self._handle_string_escape(char)
        elif state == JSONState.NUMBER_INT:
            self._handle_number_int(char)
        elif state == JSONState.NUMBER_FRAC:
            self._handle_number_frac(char)
        elif state == JSONState.NUMBER_EXP:
            self._handle_number_exp(char)

    def _handle_start(self, char: str) -> None:
        """Handle character at start of value."""
        vtype = self._context.schema.value_type

        if char == '{' and vtype in (JSONValueType.OBJECT, JSONValueType.ANY):
            self._context.state = JSONState.OBJECT_OPEN
        elif char == '[' and vtype in (JSONValueType.ARRAY, JSONValueType.ANY):
            self._context.state = JSONState.ARRAY_OPEN
        elif char == '"' and vtype in (JSONValueType.STRING, JSONValueType.ANY):
            self._context.state = JSONState.STRING_CONTENT
            self._context.string_buffer = ""
        elif char in '-0123456789' and vtype in (JSONValueType.NUMBER, JSONValueType.INTEGER, JSONValueType.ANY):
            self._context.state = JSONState.NUMBER_INT
            self._context.number_buffer = char
        elif char == 't' and vtype in (JSONValueType.BOOLEAN, JSONValueType.ANY):
            # Start of 'true' - would need multi-char handling
            pass
        elif char == 'f' and vtype in (JSONValueType.BOOLEAN, JSONValueType.ANY):
            # Start of 'false'
            pass
        elif char == 'n' and vtype in (JSONValueType.NULL, JSONValueType.ANY):
            # Start of 'null'
            pass

    def _handle_object_open(self, char: str) -> None:
        """Handle character after {."""
        if char == '"':
            self._context.state = JSONState.OBJECT_KEY
            self._context.string_buffer = ""
        elif char == '}':
            # Empty object - check required fields
            self._pop_context()

    def _handle_object_key(self, char: str) -> None:
        """Handle character while parsing object key."""
        if char == '"':
            # End of key
            self._context.current_key = self._context.string_buffer
            self._context.state = JSONState.OBJECT_COLON
        elif char == '\\':
            self._context.state = JSONState.STRING_ESCAPE
        else:
            self._context.string_buffer += char

    def _handle_object_colon(self, char: str) -> None:
        """Handle character expecting colon."""
        if char == ':':
            self._context.state = JSONState.OBJECT_VALUE

    def _handle_object_value(self, char: str) -> None:
        """Handle character at start of object value."""
        # Push new context for the value
        key = self._context.current_key
        value_schema = self._context.schema.properties.get(
            key,
            SchemaNode(JSONValueType.ANY) if self._context.schema.additional_properties else None
        )
        if value_schema:
            new_context = JSONParseContext(
                state=JSONState.START,
                schema=value_schema,
                parent=self._context,
            )
            self._context = new_context
            self._handle_start(char)

    def _handle_object_comma(self, char: str) -> None:
        """Handle character expecting comma or }."""
        if char == ',':
            self._context.state = JSONState.OBJECT_OPEN
        elif char == '}':
            self._pop_context()

    def _handle_array_open(self, char: str) -> None:
        """Handle character after [."""
        if char == ']':
            # Empty array
            self._pop_context()
        else:
            # Start of first element
            self._context.state = JSONState.ARRAY_VALUE
            item_schema = self._context.schema.items or SchemaNode(JSONValueType.ANY)
            new_context = JSONParseContext(
                state=JSONState.START,
                schema=item_schema,
                parent=self._context,
            )
            self._context = new_context
            self._handle_start(char)

    def _handle_array_value(self, char: str) -> None:
        """Handle character expecting array value."""
        item_schema = self._context.schema.items or SchemaNode(JSONValueType.ANY)
        new_context = JSONParseContext(
            state=JSONState.START,
            schema=item_schema,
            parent=self._context,
        )
        self._context = new_context
        self._handle_start(char)

    def _handle_array_comma(self, char: str) -> None:
        """Handle character expecting comma or ]."""
        if char == ',':
            self._context.array_index += 1
            self._context.state = JSONState.ARRAY_VALUE
        elif char == ']':
            self._pop_context()

    def _handle_string_content(self, char: str) -> None:
        """Handle character inside string."""
        if char == '"':
            # End of string
            self._complete_value()
        elif char == '\\':
            self._context.state = JSONState.STRING_ESCAPE
        else:
            self._context.string_buffer += char

    def _handle_string_escape(self, char: str) -> None:
        """Handle character after backslash."""
        # Accept any escape sequence character
        self._context.string_buffer += '\\' + char
        self._context.state = JSONState.STRING_CONTENT

    def _handle_number_int(self, char: str) -> None:
        """Handle character in integer part of number."""
        if char in '0123456789':
            self._context.number_buffer += char
        elif char == '.':
            self._context.number_buffer += char
            self._context.state = JSONState.NUMBER_FRAC
        elif char in 'eE':
            self._context.number_buffer += char
            self._context.state = JSONState.NUMBER_EXP
        else:
            # End of number - process char in parent context
            self._complete_value()
            self._process_char(char)

    def _handle_number_frac(self, char: str) -> None:
        """Handle character in fractional part of number."""
        if char in '0123456789':
            self._context.number_buffer += char
        elif char in 'eE':
            self._context.number_buffer += char
            self._context.state = JSONState.NUMBER_EXP
        else:
            self._complete_value()
            self._process_char(char)

    def _handle_number_exp(self, char: str) -> None:
        """Handle character in exponent part of number."""
        if char in '0123456789+-':
            self._context.number_buffer += char
        else:
            self._complete_value()
            self._process_char(char)

    def _complete_value(self) -> None:
        """Called when a value is complete. Pop context and update parent."""
        if self._context.parent:
            parent = self._context.parent
            if parent.state == JSONState.OBJECT_VALUE:
                parent.seen_keys.add(parent.current_key)
                parent.state = JSONState.OBJECT_COMMA
            elif parent.state == JSONState.ARRAY_VALUE:
                parent.state = JSONState.ARRAY_COMMA
            self._context = parent
        else:
            self._context.state = JSONState.DONE

    def _pop_context(self) -> None:
        """Pop current context and return to parent."""
        if self._context.parent:
            self._complete_value()
        else:
            self._context.state = JSONState.DONE

    def _get_valid_for_state(self) -> set[int]:
        """Get valid tokens for current parse state.

        Returns:
            Set of valid token IDs
        """
        ctx = self._context
        state = ctx.state
        schema = ctx.schema

        valid: set[int] = set()

        # Add whitespace tokens for most states
        if state not in (JSONState.STRING_CONTENT, JSONState.STRING_ESCAPE):
            valid |= self._whitespace_tokens

        if state == JSONState.START:
            valid |= self._get_valid_for_value_start(schema.value_type)

        elif state == JSONState.OBJECT_OPEN:
            valid |= self._string_start_tokens  # Key must be string
            # Can close empty object if no required fields
            if not schema.required or schema.required <= ctx.seen_keys:
                valid |= self._object_end_tokens

        elif state == JSONState.OBJECT_KEY:
            # Allow any string characters
            valid |= self._get_valid_string_chars()

        elif state == JSONState.OBJECT_COLON:
            valid |= self._colon_tokens

        elif state == JSONState.OBJECT_VALUE:
            key = ctx.current_key
            value_schema = schema.properties.get(key, SchemaNode(JSONValueType.ANY))
            valid |= self._get_valid_for_value_start(value_schema.value_type)

        elif state == JSONState.OBJECT_COMMA:
            valid |= self._comma_tokens
            # Can close object if all required fields present
            if not schema.required or schema.required <= ctx.seen_keys:
                valid |= self._object_end_tokens

        elif state == JSONState.ARRAY_OPEN:
            valid |= self._array_end_tokens  # Empty array
            if schema.items:
                valid |= self._get_valid_for_value_start(schema.items.value_type)
            else:
                valid |= self._get_valid_for_value_start(JSONValueType.ANY)

        elif state == JSONState.ARRAY_VALUE:
            if schema.items:
                valid |= self._get_valid_for_value_start(schema.items.value_type)
            else:
                valid |= self._get_valid_for_value_start(JSONValueType.ANY)

        elif state == JSONState.ARRAY_COMMA:
            valid |= self._comma_tokens
            valid |= self._array_end_tokens

        elif state == JSONState.STRING_CONTENT:
            valid |= self._get_valid_string_chars()

        elif state == JSONState.STRING_ESCAPE:
            # Allow escape sequence characters
            for c in 'nrtbf\\"u/':
                valid |= self._char_to_tokens.get(c, set())

        elif state in (JSONState.NUMBER_INT, JSONState.NUMBER_FRAC, JSONState.NUMBER_EXP):
            valid |= self._get_valid_number_chars(state)

        elif state == JSONState.DONE:
            # No more tokens valid (generation should stop)
            pass

        return valid

    def _get_valid_for_value_start(self, value_type: JSONValueType) -> set[int]:
        """Get tokens valid at the start of a JSON value.

        Args:
            value_type: Expected type of the value

        Returns:
            Set of valid token IDs
        """
        valid: set[int] = set()

        if value_type in (JSONValueType.OBJECT, JSONValueType.ANY):
            valid |= self._object_start_tokens

        if value_type in (JSONValueType.ARRAY, JSONValueType.ANY):
            valid |= self._array_start_tokens

        if value_type in (JSONValueType.STRING, JSONValueType.ANY):
            valid |= self._string_start_tokens

        if value_type in (JSONValueType.NUMBER, JSONValueType.INTEGER, JSONValueType.ANY):
            valid |= self._digit_tokens
            valid |= self._char_to_tokens.get('-', set())

        if value_type in (JSONValueType.BOOLEAN, JSONValueType.ANY):
            valid |= self._char_to_tokens.get('t', set())  # true
            valid |= self._char_to_tokens.get('f', set())  # false

        if value_type in (JSONValueType.NULL, JSONValueType.ANY):
            valid |= self._char_to_tokens.get('n', set())  # null

        return valid

    def _get_valid_string_chars(self) -> set[int]:
        """Get tokens valid inside a JSON string.

        Returns:
            Set of valid token IDs
        """
        valid: set[int] = set()

        # End quote
        valid |= self._string_start_tokens

        # Escape start
        valid |= self._char_to_tokens.get('\\', set())

        # Regular printable characters (excluding control chars and quotes)
        for c in string.printable:
            if c not in '"\\\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f':
                valid |= self._char_to_tokens.get(c, set())

        return valid

    def _get_valid_number_chars(self, state: JSONState) -> set[int]:
        """Get tokens valid in number context.

        Args:
            state: Current number parsing state

        Returns:
            Set of valid token IDs
        """
        valid: set[int] = set()

        # Digits always valid in numbers
        valid |= self._digit_tokens

        if state == JSONState.NUMBER_INT:
            # Can start fractional or exponent part
            valid |= self._char_to_tokens.get('.', set())
            valid |= self._char_to_tokens.get('e', set())
            valid |= self._char_to_tokens.get('E', set())
            # Number terminators (will complete number)
            valid |= self._comma_tokens
            valid |= self._object_end_tokens
            valid |= self._array_end_tokens
            valid |= self._whitespace_tokens

        elif state == JSONState.NUMBER_FRAC:
            valid |= self._char_to_tokens.get('e', set())
            valid |= self._char_to_tokens.get('E', set())
            valid |= self._comma_tokens
            valid |= self._object_end_tokens
            valid |= self._array_end_tokens
            valid |= self._whitespace_tokens

        elif state == JSONState.NUMBER_EXP:
            valid |= self._char_to_tokens.get('+', set())
            valid |= self._char_to_tokens.get('-', set())
            valid |= self._comma_tokens
            valid |= self._object_end_tokens
            valid |= self._array_end_tokens
            valid |= self._whitespace_tokens

        return valid
