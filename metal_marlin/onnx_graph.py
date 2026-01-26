"""ONNX graph structure parser for architecture-agnostic inference.

This module parses ONNX graphs to understand execution order and detect
model architectures from op patterns, enabling architecture-agnostic
inference without hardcoding model-specific logic.

Example:
    ops = parse_onnx_graph("model.onnx")
    arch = detect_architecture(ops)
    print(f"Detected: {arch}")  # e.g., "llama", "gpt", "moe"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ONNXOp:
    """Represents a single ONNX graph operation.

    This is a lightweight representation focused on execution order and
    architecture detection, as opposed to the full ONNXNode in converters
    which is designed for execution.

    Attributes:
        name: Unique node name within the graph.
        op_type: ONNX op type (e.g., MatMul, Softmax, LayerNorm).
        inputs: List of input tensor names.
        outputs: List of output tensor names.
        attributes: Node attributes (axis, epsilon, etc.).
    """

    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ONNXGraphInfo:
    """Parsed ONNX graph metadata.

    Provides high-level information about the graph structure without
    loading weights into memory.
    """

    ops: list[ONNXOp]
    input_names: list[str]
    output_names: list[str]
    initializer_names: set[str]
    opset_version: int
    ir_version: int
    producer_name: str


def parse_onnx_graph(onnx_path: str | Path) -> list[ONNXOp]:
    """Parse ONNX graph into execution order.

    Loads the ONNX model and extracts the graph structure in topological
    order (the order nodes appear in the ONNX graph, which is guaranteed
    to be a valid execution order by the ONNX spec).

    Args:
        onnx_path: Path to .onnx file.

    Returns:
        List of ONNXOp in execution order.

    Raises:
        ImportError: If onnx package is not installed.
        ValueError: If the file is not a valid ONNX model.
    """
    try:
        import onnx
    except ImportError as e:
        raise ImportError("pip install onnx to use ONNX graph parsing") from e

    path = Path(onnx_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX file not found: {path}")

    model = onnx.load(str(path))

    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        raise ValueError(f"Invalid ONNX model: {e}") from e

    graph = model.graph
    ops: list[ONNXOp] = []

    for node in graph.node:
        ops.append(
            ONNXOp(
                name=node.name or f"unnamed_{len(ops)}",
                op_type=node.op_type,
                inputs=list(node.input),
                outputs=list(node.output),
                attributes=_parse_attributes(node),
            )
        )

    return ops


def parse_onnx_graph_full(onnx_path: str | Path) -> ONNXGraphInfo:
    """Parse ONNX graph with full metadata.

    Similar to parse_onnx_graph but returns additional graph metadata
    useful for analysis and debugging.

    Args:
        onnx_path: Path to .onnx file.

    Returns:
        ONNXGraphInfo with ops and metadata.
    """
    try:
        import onnx
    except ImportError as e:
        raise ImportError("pip install onnx to use ONNX graph parsing") from e

    path = Path(onnx_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX file not found: {path}")

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    graph = model.graph

    ops: list[ONNXOp] = []
    for node in graph.node:
        ops.append(
            ONNXOp(
                name=node.name or f"unnamed_{len(ops)}",
                op_type=node.op_type,
                inputs=list(node.input),
                outputs=list(node.output),
                attributes=_parse_attributes(node),
            )
        )

    initializer_names = {init.name for init in graph.initializer}
    input_names = [i.name for i in graph.input if i.name not in initializer_names]
    output_names = [o.name for o in graph.output]

    opset_version = 0
    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            opset_version = opset.version
            break

    return ONNXGraphInfo(
        ops=ops,
        input_names=input_names,
        output_names=output_names,
        initializer_names=initializer_names,
        opset_version=opset_version,
        ir_version=model.ir_version,
        producer_name=model.producer_name or "unknown",
    )


def _parse_attributes(node) -> dict[str, Any]:
    """Parse ONNX node attributes to Python types."""
    try:
        import onnx
    except ImportError:
        return {}

    attrs: dict[str, Any] = {}
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.FLOAT:
            attrs[attr.name] = attr.f
        elif attr.type == onnx.AttributeProto.INT:
            attrs[attr.name] = attr.i
        elif attr.type == onnx.AttributeProto.STRING:
            attrs[attr.name] = attr.s.decode("utf-8")
        elif attr.type == onnx.AttributeProto.FLOATS:
            attrs[attr.name] = list(attr.floats)
        elif attr.type == onnx.AttributeProto.INTS:
            attrs[attr.name] = list(attr.ints)
        elif attr.type == onnx.AttributeProto.STRINGS:
            attrs[attr.name] = [s.decode("utf-8") for s in attr.strings]
        elif attr.type == onnx.AttributeProto.TENSOR:
            attrs[attr.name] = "<tensor>"
        elif attr.type == onnx.AttributeProto.GRAPH:
            attrs[attr.name] = "<subgraph>"
        elif attr.type == onnx.AttributeProto.TENSORS:
            attrs[attr.name] = "<tensors>"
        elif attr.type == onnx.AttributeProto.GRAPHS:
            attrs[attr.name] = "<subgraphs>"
    return attrs


def detect_architecture(ops: list[ONNXOp]) -> str:
    """Detect model architecture from op patterns.

    Analyzes the sequence and types of operations to identify the
    transformer architecture variant. This enables architecture-specific
    optimizations while maintaining a generic inference path.

    Detection heuristics:
    - Llama: RMSNorm, SiLU/Swish activation, RoPE patterns
    - GPT: LayerNorm, GELU activation, absolute position embeddings
    - Mistral: Similar to Llama but with sliding window attention patterns
    - MoE: Router + TopK + Expert dispatch patterns
    - BERT: Bidirectional attention, MLM head patterns
    - T5: Encoder-decoder structure, relative position bias
    - Unknown: No recognized pattern

    Args:
        ops: List of ONNXOp from parse_onnx_graph.

    Returns:
        Architecture name: "llama", "gpt", "mistral", "moe", "bert", "t5",
        or "unknown".
    """
    op_types = [op.op_type for op in ops]
    op_type_set = set(op_types)
    op_counts = _count_op_types(ops)

    # Check for MoE patterns first (most specific)
    if _has_moe_pattern(ops, op_type_set):
        return "moe"

    # Check for encoder-decoder (T5-like)
    if _has_encoder_decoder_pattern(ops):
        return "t5"

    # Check for BERT-like bidirectional patterns
    if _has_bert_pattern(ops, op_type_set, op_counts):
        return "bert"

    # Check for Llama-like patterns
    if _has_llama_pattern(ops, op_type_set, op_counts):
        # Distinguish Mistral from Llama via sliding window hints
        if _has_sliding_window_pattern(ops):
            return "mistral"
        return "llama"

    # Check for GPT-like patterns
    if _has_gpt_pattern(ops, op_type_set, op_counts):
        return "gpt"

    # Check for basic transformer structure
    if _has_basic_transformer_pattern(op_type_set, op_counts):
        return "transformer"

    return "unknown"


def _count_op_types(ops: list[ONNXOp]) -> dict[str, int]:
    """Count occurrences of each op type."""
    counts: dict[str, int] = {}
    for op in ops:
        counts[op.op_type] = counts.get(op.op_type, 0) + 1
    return counts


def _has_moe_pattern(ops: list[ONNXOp], op_type_set: set[str]) -> bool:
    """Detect Mixture of Experts patterns.

    MoE models have:
    - Router layer (typically a Linear followed by Softmax/TopK)
    - Multiple expert paths that get selectively activated
    - Scatter/Gather or conditional dispatch patterns
    """
    # Look for TopK which is used for expert selection
    has_topk = "TopK" in op_type_set

    # Look for patterns suggesting router + experts
    # Experts typically have parallel MatMul paths
    matmul_count = sum(1 for op in ops if op.op_type == "MatMul")

    # Look for Scatter/ScatterND/ScatterElements for expert dispatch
    has_scatter = any(
        op_type in op_type_set for op_type in ("Scatter", "ScatterND", "ScatterElements")
    )

    # Look for Gather which is used to select expert outputs
    has_gather_patterns = "Gather" in op_type_set or "GatherElements" in op_type_set

    # MoE typically has TopK for routing and many MatMuls for experts
    if has_topk and matmul_count > 20:
        return True

    # Alternative: Scatter + Gather pattern with many MatMuls
    if has_scatter and has_gather_patterns and matmul_count > 15:
        return True

    # Check for expert naming patterns in node names
    expert_names = sum(1 for op in ops if "expert" in op.name.lower())
    if expert_names > 4:
        return True

    return False


def _has_encoder_decoder_pattern(ops: list[ONNXOp]) -> bool:
    """Detect encoder-decoder architecture (T5-like).

    T5 and similar models have:
    - Separate encoder and decoder blocks
    - Cross-attention between encoder outputs and decoder
    - Relative position bias patterns
    """
    # Look for cross-attention patterns (decoder attending to encoder)
    # This typically shows as attention where Q comes from decoder
    # and K/V come from encoder (different sources)

    # Check for relative position bias patterns
    has_relative_position = any(
        "relative" in op.name.lower() or "position_bias" in op.name.lower() for op in ops
    )

    # Check for encoder/decoder naming
    encoder_count = sum(1 for op in ops if "encoder" in op.name.lower())
    decoder_count = sum(1 for op in ops if "decoder" in op.name.lower())

    if encoder_count > 2 and decoder_count > 2:
        return True

    # Look for cross attention patterns
    cross_attn = sum(1 for op in ops if "cross" in op.name.lower())
    if cross_attn > 2 and has_relative_position:
        return True

    return False


def _has_bert_pattern(
    ops: list[ONNXOp], op_type_set: set[str], op_counts: dict[str, int]
) -> bool:
    """Detect BERT-like bidirectional encoder patterns.

    BERT models have:
    - LayerNorm (not RMSNorm)
    - GELU activation
    - No causal masking (bidirectional attention)
    - Often have [CLS] token handling
    """
    has_layernorm = "LayerNormalization" in op_type_set
    has_gelu = "Gelu" in op_type_set

    # BERT uses LayerNorm, not RMSNorm
    if not has_layernorm:
        return False

    # Check for BERT-specific naming patterns
    bert_names = sum(
        1 for op in ops if any(x in op.name.lower() for x in ("bert", "pooler", "cls"))
    )

    # BERT typically has many attention heads with no causal patterns
    matmul_count = op_counts.get("MatMul", 0)
    softmax_count = op_counts.get("Softmax", 0)

    # BERT pattern: LayerNorm + GELU + attention structure
    if has_layernorm and has_gelu and matmul_count > 10 and softmax_count > 0:
        if bert_names > 0:
            return True
        # Also match if we see typical BERT structure without explicit naming
        layernorm_count = op_counts.get("LayerNormalization", 0)
        if layernorm_count > 20:  # Many LayerNorm layers is BERT-like
            return True

    return False


def _has_llama_pattern(
    ops: list[ONNXOp], op_type_set: set[str], op_counts: dict[str, int]
) -> bool:
    """Detect Llama-like decoder patterns.

    Llama models have:
    - RMSNorm (not LayerNorm)
    - SiLU/Swish activation (x * sigmoid(x))
    - Rotary Position Embeddings (RoPE)
    - Gated MLP (gate * up then down projection)
    """
    # Check for SiLU which is characteristic of Llama
    has_silu = "Silu" in op_type_set

    # Check for Mul + Sigmoid pattern (manual SiLU implementation)
    has_mul = "Mul" in op_type_set
    has_sigmoid = "Sigmoid" in op_type_set

    # Check for RoPE patterns (complex number operations or Sin/Cos)
    has_rope_ops = "Sin" in op_type_set and "Cos" in op_type_set

    # Look for RMSNorm-like patterns
    # RMSNorm doesn't have mean subtraction, just variance normalization
    # This is hard to detect directly, but Llama typically doesn't have LayerNorm
    has_layernorm = "LayerNormalization" in op_type_set

    # Llama uses SiLU and often has RoPE
    if has_silu or (has_mul and has_sigmoid):
        # Additional confidence from RoPE patterns
        if has_rope_ops:
            return True
        # Or from lack of LayerNorm (Llama uses RMSNorm)
        if not has_layernorm:
            # Check for typical Llama structure
            matmul_count = op_counts.get("MatMul", 0)
            if matmul_count > 10:
                return True

    # Check for llama-specific naming
    llama_names = sum(
        1 for op in ops if any(x in op.name.lower() for x in ("llama", "rmsnorm", "rope"))
    )
    if llama_names > 2:
        return True

    return False


def _has_sliding_window_pattern(ops: list[ONNXOp]) -> bool:
    """Detect sliding window attention (Mistral).

    Mistral uses sliding window attention where each token only
    attends to a fixed window of previous tokens.
    """
    # Look for sliding window hints in names
    for op in ops:
        name_lower = op.name.lower()
        if "sliding" in name_lower or "window" in name_lower:
            return True

    # Look for attention mask patterns that suggest windowing
    # This would typically show as specialized masking operations
    for op in ops:
        if op.op_type == "Where" or op.op_type == "Select":
            # Check if it's used in attention context
            for output in op.outputs:
                if "mask" in output.lower() or "attn" in output.lower():
                    return True

    return False


def _has_gpt_pattern(
    ops: list[ONNXOp], op_type_set: set[str], op_counts: dict[str, int]
) -> bool:
    """Detect GPT-like decoder patterns.

    GPT models have:
    - LayerNorm (not RMSNorm)
    - GELU activation
    - Absolute position embeddings (not RoPE)
    - Causal attention masking
    """
    has_layernorm = "LayerNormalization" in op_type_set
    has_gelu = "Gelu" in op_type_set

    # GPT uses LayerNorm and GELU
    if not has_layernorm:
        return False

    # Check for GPT-specific patterns
    gpt_names = sum(
        1 for op in ops if any(x in op.name.lower() for x in ("gpt", "wte", "wpe"))
    )

    # GELU is characteristic of GPT family
    if has_gelu:
        matmul_count = op_counts.get("MatMul", 0)
        if matmul_count > 10:
            if gpt_names > 0:
                return True
            # Generic GPT-like: LayerNorm + GELU + attention
            return True

    # Check for absolute position embeddings (Add at start)
    # GPT adds position embeddings early in the forward pass
    early_adds = sum(1 for i, op in enumerate(ops[:10]) if op.op_type == "Add")
    if has_layernorm and early_adds > 0 and op_counts.get("MatMul", 0) > 10:
        return True

    return False


def _has_basic_transformer_pattern(
    op_type_set: set[str], op_counts: dict[str, int]
) -> bool:
    """Detect basic transformer structure without specific architecture."""
    # Core transformer ops
    has_matmul = "MatMul" in op_type_set or "Gemm" in op_type_set
    has_softmax = "Softmax" in op_type_set
    has_norm = "LayerNormalization" in op_type_set or any(
        "norm" in op.lower() for op in op_type_set
    )

    if not (has_matmul and has_softmax):
        return False

    matmul_count = op_counts.get("MatMul", 0) + op_counts.get("Gemm", 0)
    softmax_count = op_counts.get("Softmax", 0)

    # Basic transformer: multiple attention heads (MatMul -> Softmax -> MatMul)
    if matmul_count >= 4 and softmax_count >= 1:
        return True

    return False


def get_layer_count(ops: list[ONNXOp]) -> int:
    """Estimate the number of transformer layers.

    Counts repeated structural patterns to estimate layer count.
    """
    # Count softmax operations as proxy for attention layers
    softmax_count = sum(1 for op in ops if op.op_type == "Softmax")

    # Each transformer layer typically has one Softmax in attention
    return softmax_count


def get_attention_heads(ops: list[ONNXOp]) -> int | None:
    """Attempt to infer number of attention heads from Reshape operations.

    Returns None if unable to determine.
    """
    for op in ops:
        if op.op_type == "Reshape":
            # Look for reshape that splits hidden_dim into heads
            # Shape often appears as [batch, seq, num_heads, head_dim]
            # This is a heuristic and may not work for all models
            pass
    return None


def summarize_graph(ops: list[ONNXOp]) -> dict[str, Any]:
    """Generate a summary of the ONNX graph structure.

    Returns:
        Dictionary with:
        - op_counts: Count of each operation type
        - estimated_layers: Estimated number of transformer layers
        - architecture: Detected architecture
        - total_ops: Total number of operations
    """
    op_counts = _count_op_types(ops)
    return {
        "op_counts": op_counts,
        "estimated_layers": get_layer_count(ops),
        "architecture": detect_architecture(ops),
        "total_ops": len(ops),
    }
