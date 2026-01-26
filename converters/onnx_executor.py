"""ONNX graph execution with Metal Marlin kernels.

This module provides a generic way to run ONNX models using Metal Marlin's
quantized GEMM kernels, without hardcoding model architectures.

Architecture:
1. Parse ONNX graph structure (nodes, edges, initializers)
2. Load weights from ONNX initializers or external files
3. Detect multi-head attention patterns and fuse into single calls
4. For each node, dispatch to appropriate kernel:
   - MatMul/Gemm -> marlin_gemm_fp4 (if weights are quantized)
   - Fused MHA -> MarlinAttention (decomposed Q/K/V pattern)
   - Attention op -> flash_attention_metal (native ONNX Attention)
   - LayerNorm, RoPE, etc. -> PyTorch standard ops

This approach supports any transformer architecture that uses standard ONNX ops,
without needing model-specific code like llama.py or mistral.py.

Requirements:
    pip install onnx onnxruntime torch
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    pass

# Re-export pack_fp4_weights for convenience
try:
    from ..metal_marlin import pack_fp4_weights_cpu as pack_fp4_weights
except ImportError:
    from metal_marlin import pack_fp4_weights_cpu as pack_fp4_weights

__all__ = ["ONNXExecutor", "load_onnx_model", "pack_fp4_weights"]


def _get_device() -> torch.device:
    """Get the appropriate device (MPS on Apple Silicon, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class MHAPattern:
    """A detected multi-head attention subgraph.

    Represents the decomposed attention pattern:
        Q_proj (MatMul) -> Reshape -> Transpose ->
        K_proj (MatMul) -> Reshape -> Transpose ->
        V_proj (MatMul) -> Reshape -> Transpose ->
        QK^T (MatMul) -> Softmax -> AV (MatMul) ->
        Reshape -> Transpose -> O_proj (MatMul)
    """

    # Indices of all nodes consumed by this fused op
    node_indices: list[int]
    # The first node index (entry point for execution ordering)
    entry_idx: int
    # Weight tensor names for Q, K, V, O projections
    q_weight: str
    k_weight: str
    v_weight: str
    o_weight: str
    # Inferred geometry
    num_heads: int
    head_dim: int
    hidden_size: int
    # Input tensor name (hidden_states feeding Q/K/V projections)
    input_name: str
    # Output tensor name (final result after O projection)
    output_name: str


@dataclass
class ONNXNode:
    """Represents a single ONNX graph node."""

    op_type: str
    name: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, object] = field(default_factory=dict)


@dataclass
class ONNXGraph:
    """Parsed ONNX model graph."""

    nodes: list[ONNXNode]
    inputs: list[str]
    outputs: list[str]
    initializers: dict[str, torch.Tensor]  # Weight tensors


class ONNXExecutor:
    """Execute ONNX graphs using Metal Marlin kernels.

    Example:
        executor = ONNXExecutor.from_file("model.onnx")
        output = executor(input_ids=tokens)
    """

    def __init__(self, graph: ONNXGraph, quantized_weights: dict[str, tuple] | None = None) -> None:
        """Initialize executor with parsed graph.

        Args:
            graph: Parsed ONNX graph
            quantized_weights: Optional dict of {weight_name: (packed, scales)}
                               for quantized layers
        """
        self.graph = graph
        self.quantized_weights = quantized_weights or {}
        self._mha_patterns = _detect_mha_patterns(graph)
        self._fused_node_indices: set[int] = set()
        for pattern in self._mha_patterns:
            self._fused_node_indices.update(pattern.node_indices)
        self._op_dispatch = self._build_dispatch_table()
        self._device = _get_device()

    @classmethod
    def from_file(cls, path: str | Path, quantize: bool = True) -> ONNXExecutor:
        """Load ONNX model and optionally quantize weights.

        Args:
            path: Path to .onnx file
            quantize: If True, quantize linear layer weights to FP4

        Returns:
            Configured executor
        """
        graph = load_onnx_model(path)

        if quantize:
            quantized = _quantize_linear_weights(graph)
            return cls(graph, quantized)

        return cls(graph)

    def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Execute the graph on given inputs.

        Args:
            **inputs: Named input tensors (e.g., input_ids, attention_mask)

        Returns:
            Dict of output tensor names to values
        """
        # Tensor value map: starts with inputs + initializers
        values: dict[str, torch.Tensor] = {**self.graph.initializers, **inputs}

        # Build index mapping: entry_idx -> pattern for fast lookup
        entry_to_pattern: dict[int, MHAPattern] = {p.entry_idx: p for p in self._mha_patterns}

        # Topological execution
        for idx, node in enumerate(self.graph.nodes):
            # If this node is the entry point of a fused MHA pattern,
            # execute the entire pattern as one fused call
            if idx in entry_to_pattern:
                pattern = entry_to_pattern[idx]
                result = self._handle_fused_attention(pattern, values)
                values[pattern.output_name] = result
                continue

            # Skip nodes that are interior to a fused pattern
            if idx in self._fused_node_indices:
                continue

            node_inputs = [values[name] for name in node.inputs if name in values]
            node_outputs = self._dispatch_node(node, node_inputs)

            for name, value in zip(node.outputs, node_outputs, strict=False):
                values[name] = value

        # Return requested outputs
        return {name: values[name] for name in self.graph.outputs}

    def _dispatch_node(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Dispatch a single node to the appropriate kernel."""
        handler = self._op_dispatch.get(node.op_type)
        if handler is None:
            raise NotImplementedError(f"ONNX op not supported: {node.op_type}")
        return handler(node, inputs)

    def _build_dispatch_table(self) -> dict:
        """Build op_type -> handler function mapping."""
        return {
            # Matrix operations
            "MatMul": self._handle_matmul,
            "Gemm": self._handle_gemm,
            # Normalization
            "LayerNormalization": self._handle_layernorm,
            # Activation functions
            "Relu": self._handle_relu,
            "Sigmoid": self._handle_sigmoid,
            "Tanh": self._handle_tanh,
            "Gelu": self._handle_gelu,
            "Silu": self._handle_silu,
            "Softmax": self._handle_softmax,
            # Element-wise operations
            "Mul": self._handle_mul,
            "Add": self._handle_add,
            "Sub": self._handle_sub,
            "Div": self._handle_div,
            "Pow": self._handle_pow,
            "Sqrt": self._handle_sqrt,
            "Neg": self._handle_neg,
            # Shape operations
            "Reshape": self._handle_reshape,
            "Transpose": self._handle_transpose,
            "Squeeze": self._handle_squeeze,
            "Unsqueeze": self._handle_unsqueeze,
            "Concat": self._handle_concat,
            "Split": self._handle_split,
            "Gather": self._handle_gather,
            # Reduction operations
            "ReduceMean": self._handle_reduce_mean,
            "ReduceSum": self._handle_reduce_sum,
            # Attention
            "Attention": self._handle_attention_op,
            "MultiHeadAttention": self._handle_fused_attention_dispatch,
        }

    def _handle_fused_attention(self, pattern: MHAPattern, values: dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute a detected MHA pattern as a single fused MarlinAttention call.

        Instead of running ~14 individual ops (Q/K/V projections, reshapes,
        transposes, two matmuls, softmax, output projection), we run one
        MarlinAttention forward pass with quantized Q/K/V/O projections.

        Args:
            pattern: The detected MHA pattern with weight names and geometry
            values: Current tensor value map

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        from ..metal_marlin.attention import MarlinAttention

        hidden_states = values[pattern.input_name]

        # Build MarlinAttention with the correct geometry
        attn = MarlinAttention(
            hidden_size=pattern.hidden_size,
            num_heads=pattern.num_heads,
            num_kv_heads=pattern.num_heads,  # MHA, not GQA
            head_dim=pattern.head_dim,
            bias=False,
        )

        # Load weights from graph initializers into the attention module
        _load_proj_weights(attn.q_proj, pattern.q_weight, self.graph, self.quantized_weights)
        _load_proj_weights(attn.k_proj, pattern.k_weight, self.graph, self.quantized_weights)
        _load_proj_weights(attn.v_proj, pattern.v_weight, self.graph, self.quantized_weights)
        _load_proj_weights(attn.o_proj, pattern.o_weight, self.graph, self.quantized_weights)

        # Move attention module to device
        attn = attn.to(self._device)

        # Run fused attention (no KV cache, no mask for now)
        return attn(hidden_states)

    def _handle_fused_attention_dispatch(
        self, node: ONNXNode, inputs: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Handle explicit MultiHeadAttention op (custom domain or fused graph).

        This handles the case where an ONNX graph has already been fused into
        a single MultiHeadAttention node (e.g., via onnxruntime graph optimization).
        """
        from ..metal_marlin.attention import MarlinAttention

        hidden_states = inputs[0]
        batch_size, seq_len, hidden_size = hidden_states.shape

        num_heads = int(node.attrs.get("num_heads", 1))
        head_dim = hidden_size // num_heads

        attn = MarlinAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=False,
        )

        # Load Q/K/V/O weights from node inputs if available
        for i, proj in enumerate([attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj]):
            weight_idx = i + 1  # inputs[0] is hidden_states
            if weight_idx < len(node.inputs):
                weight_name = node.inputs[weight_idx]
                _load_proj_weights(proj, weight_name, self.graph, self.quantized_weights)

        # Move attention module to device
        attn = attn.to(self._device)

        return [attn(hidden_states)]

    def _handle_attention_op(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle native ONNX Attention op via scaled_dot_product_attention.

        The ONNX Attention op computes scaled dot-product attention directly.
        We dispatch to PyTorch's optimized SDPA for O(N) memory and fused softmax.

        ONNX Attention op inputs:
            - query: [batch, num_heads, seq_q, head_dim]
            - key: [batch, num_heads, seq_k, head_dim]
            - value: [batch, num_heads, seq_k, head_dim]
            - Optional: mask, scale

        We reshape to match SDPA's expected layout.
        """
        query = inputs[0]
        key = inputs[1]
        value = inputs[2]

        # Extract scale from attrs or compute from head_dim
        head_dim = query.shape[-1]
        scale = node.attrs.get("scale", head_dim**-0.5)

        # Check for causal mask attribute
        causal = bool(node.attrs.get("causal", 0))

        # Determine if we need GQA handling
        num_q_heads = query.shape[1] if query.ndim == 4 else 1
        num_kv_heads = key.shape[1] if key.ndim == 4 else 1

        # Expand K/V for GQA if needed
        if num_q_heads > num_kv_heads:
            repeat_factor = num_q_heads // num_kv_heads
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)

        # Use PyTorch's optimized scaled_dot_product_attention
        output = F.scaled_dot_product_attention(
            query, key, value,
            scale=scale,
            is_causal=causal,
        )

        return [output]

    def _handle_matmul(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle MatMul - use quantized kernel if weights are quantized."""
        a, b = inputs

        # Check if B is a quantized weight
        weight_name = node.inputs[1]
        if weight_name in self.quantized_weights:
            try:
                from ..metal_marlin import quantized_linear
            except ImportError:
                from metal_marlin import quantized_linear

            packed, scales = self.quantized_weights[weight_name]
            return [quantized_linear(a, packed, scales, group_size=32)]

        # Fall back to standard matmul
        return [torch.matmul(a, b)]

    def _handle_gemm(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle Gemm (MatMul + bias)."""
        a, b = inputs[:2]
        bias = inputs[2] if len(inputs) > 2 else None

        alpha = node.attrs.get("alpha", 1.0)
        beta = node.attrs.get("beta", 1.0)
        trans_a = node.attrs.get("transA", 0)
        trans_b = node.attrs.get("transB", 0)

        if trans_a:
            a = a.T
        if trans_b:
            b = b.T

        result = alpha * torch.matmul(a, b)
        if bias is not None:
            result = result + beta * bias
        return [result]

    def _handle_layernorm(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle LayerNormalization.

        ONNX LayerNormalization has inputs: X, Scale, [Bias]
        Bias is optional (can be 2 or 3 inputs).
        """
        x = inputs[0]
        scale = inputs[1]
        bias = inputs[2] if len(inputs) > 2 else None

        epsilon = node.attrs.get("epsilon", 1e-5)
        axis = node.attrs.get("axis", -1)

        # Handle multi-axis normalization (axis can be negative)
        if isinstance(axis, int):
            axes = [axis]
        else:
            axes = list(axis)

        # Normalize axes to positive indices for computation
        ndim = x.ndim
        axes = [(a % ndim) for a in axes]

        # Compute mean and variance
        mean = x.mean(dim=axes, keepdim=True)
        var = x.var(dim=axes, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + epsilon)
        result = normalized * scale
        if bias is not None:
            result = result + bias
        return [result]

    def _handle_softmax(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle Softmax."""
        x = inputs[0]
        axis = node.attrs.get("axis", -1)
        return [F.softmax(x, dim=axis)]

    def _handle_mul(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle element-wise multiply."""
        return [inputs[0] * inputs[1]]

    def _handle_add(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle element-wise add."""
        return [inputs[0] + inputs[1]]

    def _handle_reshape(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle Reshape."""
        x, shape = inputs
        return [x.reshape(shape.tolist())]

    def _handle_transpose(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle Transpose."""
        x = inputs[0]
        perm = node.attrs.get("perm")
        if perm:
            return [x.permute(perm)]
        # Default: reverse all dimensions
        return [x.T]

    # -------------------------------------------------------------------------
    # Activation functions
    # -------------------------------------------------------------------------

    def _handle_relu(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle ReLU activation."""
        return [F.relu(inputs[0])]

    def _handle_sigmoid(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle Sigmoid activation."""
        return [torch.sigmoid(inputs[0])]

    def _handle_tanh(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle Tanh activation."""
        return [torch.tanh(inputs[0])]

    def _handle_gelu(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle GELU activation (approximate or exact based on attrs)."""
        x = inputs[0]
        approximate = node.attrs.get("approximate", "none")
        if approximate == "tanh":
            # Fast approximation
            return [F.gelu(x, approximate="tanh")]
        else:
            # Exact
            return [F.gelu(x, approximate="none")]

    def _handle_silu(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle SiLU/Swish activation: x * sigmoid(x)."""
        return [F.silu(inputs[0])]

    # -------------------------------------------------------------------------
    # Additional element-wise operations
    # -------------------------------------------------------------------------

    def _handle_sub(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle element-wise subtract."""
        return [inputs[0] - inputs[1]]

    def _handle_div(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle element-wise divide."""
        return [inputs[0] / inputs[1]]

    def _handle_pow(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle element-wise power."""
        return [torch.pow(inputs[0], inputs[1])]

    def _handle_sqrt(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle element-wise sqrt."""
        return [torch.sqrt(inputs[0])]

    def _handle_neg(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle negation."""
        return [-inputs[0]]

    # -------------------------------------------------------------------------
    # Shape operations
    # -------------------------------------------------------------------------

    def _handle_squeeze(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle Squeeze - remove dimensions of size 1."""
        x = inputs[0]
        axes = node.attrs.get("axes")
        if axes is None and len(inputs) > 1:
            # ONNX opset >= 13: axes as second input
            axes = inputs[1].tolist()
        if axes is not None:
            # Squeeze specific dimensions
            for ax in sorted(axes, reverse=True):  # Reverse to handle index shifts
                x = x.squeeze(ax)
            return [x]
        return [x.squeeze()]

    def _handle_unsqueeze(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle Unsqueeze - add dimensions of size 1."""
        x = inputs[0]
        axes = node.attrs.get("axes")
        if axes is None and len(inputs) > 1:
            axes = inputs[1].tolist()
        if axes:
            for ax in sorted(axes):
                x = x.unsqueeze(ax)
        return [x]

    def _handle_concat(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle Concat - concatenate tensors along axis."""
        axis = node.attrs.get("axis", 0)
        return [torch.cat(inputs, dim=axis)]

    def _handle_split(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle Split - split tensor into chunks."""
        x = inputs[0]
        axis = node.attrs.get("axis", 0)
        num_outputs = node.attrs.get("num_outputs")
        split = node.attrs.get("split")

        if split is None and len(inputs) > 1:
            split = inputs[1].tolist()

        if split is not None:
            # Split into specified sizes
            return list(torch.split(x, split, dim=axis))
        elif num_outputs:
            # Split into equal parts
            chunk_size = x.shape[axis] // num_outputs
            return list(torch.split(x, chunk_size, dim=axis))
        else:
            return [x]

    def _handle_gather(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle Gather - select slices from tensor."""
        data, indices = inputs
        axis = node.attrs.get("axis", 0)
        return [torch.index_select(data, axis, indices.flatten()).reshape(*indices.shape, *data.shape[axis+1:])]

    # -------------------------------------------------------------------------
    # Reduction operations
    # -------------------------------------------------------------------------

    def _handle_reduce_mean(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle ReduceMean."""
        x = inputs[0]
        axes = node.attrs.get("axes")
        keepdims = bool(node.attrs.get("keepdims", 1))

        if axes is None and len(inputs) > 1:
            axes = inputs[1].tolist()

        return [x.mean(dim=axes, keepdim=keepdims)]

    def _handle_reduce_sum(self, node: ONNXNode, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Handle ReduceSum."""
        x = inputs[0]
        axes = node.attrs.get("axes")
        keepdims = bool(node.attrs.get("keepdims", 1))

        if axes is None and len(inputs) > 1:
            axes = inputs[1].tolist()

        return [x.sum(dim=axes, keepdim=keepdims)]


# ---------------------------------------------------------------------------
# MHA Pattern Detection
# ---------------------------------------------------------------------------


def _detect_mha_patterns(graph: ONNXGraph) -> list[MHAPattern]:
    """Detect decomposed multi-head attention patterns in the ONNX graph.

    Scans for the canonical attention decomposition:
        Q = x @ W_q          (MatMul with weight initializer)
        K = x @ W_k          (MatMul with weight initializer)
        V = x @ W_v          (MatMul with weight initializer)
        Q' = Reshape(Q)      (split into heads)
        K' = Reshape(K)
        V' = Reshape(V)
        Q'' = Transpose(Q')  (permute to [batch, heads, seq, dim])
        K'' = Transpose(K')
        V'' = Transpose(V')
        scores = Q'' @ K''^T (MatMul, no weight initializer)
        attn = Softmax(scores)
        context = attn @ V'' (MatMul, no weight initializer)
        context' = Transpose(context)  (merge heads)
        context'' = Reshape(context')
        output = context'' @ W_o       (MatMul with weight initializer)

    The pattern matcher is permissive: it allows optional scale (Mul/Div)
    and mask (Add) nodes between Q@K^T and Softmax.

    Returns:
        List of detected patterns (may be empty if no patterns found)
    """
    # Build producer map: output_name -> (node_index, node)
    producer: dict[str, tuple[int, ONNXNode]] = {}
    for idx, node in enumerate(graph.nodes):
        for out in node.outputs:
            producer[out] = (idx, node)

    # Build consumer map: input_name -> list[(node_index, node)]
    consumer: dict[str, list[tuple[int, ONNXNode]]] = {}
    for idx, node in enumerate(graph.nodes):
        for inp in node.inputs:
            consumer.setdefault(inp, []).append((idx, node))

    patterns: list[MHAPattern] = []
    used_nodes: set[int] = set()

    # Strategy: find Softmax nodes (unique anchor in attention pattern),
    # then trace backward to Q@K^T and forward to attn@V
    for sm_idx, sm_node in enumerate(graph.nodes):
        if sm_node.op_type != "Softmax" or sm_idx in used_nodes:
            continue

        # Trace backward from Softmax input to find QK^T matmul.
        # Allow optional scale (Mul/Div) and mask (Add) between QK^T and Softmax.
        qk_matmul_idx, qk_node, intermediate_indices = _trace_back_to_matmul(
            sm_node.inputs[0], producer, graph
        )
        if qk_node is None:
            continue
        # QK^T matmul should NOT have a weight initializer (it's activation @ activation)
        if _has_weight_initializer(qk_node, graph):
            continue

        # Trace forward from Softmax output to find attn @ V matmul
        av_idx, av_node = _find_consumer_matmul(sm_node.outputs[0], consumer, graph)
        if av_node is None:
            continue
        if _has_weight_initializer(av_node, graph):
            continue

        # Trace Q branch: QK^T.inputs[0] <- Transpose <- Reshape <- Q_proj (MatMul)
        q_chain = _trace_proj_chain(qk_node.inputs[0], producer, graph)
        if q_chain is None:
            continue

        # Trace K branch: QK^T.inputs[1] <- Transpose <- Reshape <- K_proj (MatMul)
        k_chain = _trace_proj_chain(qk_node.inputs[1], producer, graph)
        if k_chain is None:
            continue

        # Trace V branch: AV.inputs[1] <- Transpose <- Reshape <- V_proj (MatMul)
        v_chain = _trace_proj_chain(av_node.inputs[1], producer, graph)
        if v_chain is None:
            continue

        # Verify Q, K, V projections share the same input (hidden_states)
        q_input = graph.nodes[q_chain.proj_idx].inputs[0]
        k_input = graph.nodes[k_chain.proj_idx].inputs[0]
        v_input = graph.nodes[v_chain.proj_idx].inputs[0]
        if not (q_input == k_input == v_input):
            continue

        # Trace output: AV.output -> Transpose -> Reshape -> O_proj (MatMul)
        o_chain = _trace_output_chain(av_node.outputs[0], consumer, producer, graph)
        if o_chain is None:
            continue

        # Infer geometry from reshape shape constants
        num_heads, head_dim = _infer_head_geometry(q_chain, graph)
        if num_heads is None:
            continue

        # Determine hidden_size from Q projection weight shape
        q_weight_name = graph.nodes[q_chain.proj_idx].inputs[1]
        if q_weight_name in graph.initializers:
            hidden_size = graph.initializers[q_weight_name].shape[0]
        else:
            hidden_size = num_heads * head_dim

        # Collect all node indices consumed by this pattern
        all_indices: set[int] = set()
        all_indices.update(q_chain.indices)
        all_indices.update(k_chain.indices)
        all_indices.update(v_chain.indices)
        all_indices.add(qk_matmul_idx)
        all_indices.update(intermediate_indices)
        all_indices.add(sm_idx)
        all_indices.add(av_idx)
        all_indices.update(o_chain.indices)

        # Skip if any node already consumed by another pattern
        if all_indices & used_nodes:
            continue

        entry_idx = min(all_indices)
        used_nodes.update(all_indices)

        o_proj_node = graph.nodes[o_chain.proj_idx]
        patterns.append(
            MHAPattern(
                node_indices=sorted(all_indices),
                entry_idx=entry_idx,
                q_weight=graph.nodes[q_chain.proj_idx].inputs[1],
                k_weight=graph.nodes[k_chain.proj_idx].inputs[1],
                v_weight=graph.nodes[v_chain.proj_idx].inputs[1],
                o_weight=o_proj_node.inputs[1],
                num_heads=num_heads,
                head_dim=head_dim,
                hidden_size=hidden_size,
                input_name=q_input,
                output_name=o_proj_node.outputs[0],
            )
        )

    return patterns


@dataclass
class _ProjChain:
    """Traced projection chain: MatMul -> Reshape -> Transpose."""

    proj_idx: int  # Index of the projection MatMul node
    reshape_idx: int  # Index of the Reshape node
    transpose_idx: int  # Index of the Transpose node
    indices: list[int]  # All node indices in this chain


@dataclass
class _OutputChain:
    """Traced output chain: Transpose -> Reshape -> O_proj MatMul."""

    transpose_idx: int
    reshape_idx: int
    proj_idx: int
    indices: list[int]


def _trace_proj_chain(
    tensor_name: str,
    producer: dict[str, tuple[int, ONNXNode]],
    graph: ONNXGraph,
) -> _ProjChain | None:
    """Trace backward from a tensor to find Transpose <- Reshape <- MatMul(proj).

    The tensor_name is the input to the QK^T or AV matmul, which should be
    the output of a Transpose node.
    """
    if tensor_name not in producer:
        return None

    # Level 1: Transpose
    tp_idx, tp_node = producer[tensor_name]
    if tp_node.op_type != "Transpose":
        return None

    # Level 2: Reshape (input to transpose)
    reshape_input = tp_node.inputs[0]
    if reshape_input not in producer:
        return None
    rs_idx, rs_node = producer[reshape_input]
    if rs_node.op_type != "Reshape":
        return None

    # Level 3: MatMul projection (input to reshape)
    proj_input = rs_node.inputs[0]
    if proj_input not in producer:
        return None
    proj_idx, proj_node = producer[proj_input]
    if proj_node.op_type not in ("MatMul", "Gemm"):
        return None

    # The projection must have a weight initializer (it's x @ W_proj)
    if not _has_weight_initializer(proj_node, graph):
        return None

    return _ProjChain(
        proj_idx=proj_idx,
        reshape_idx=rs_idx,
        transpose_idx=tp_idx,
        indices=[proj_idx, rs_idx, tp_idx],
    )


def _trace_output_chain(
    tensor_name: str,
    consumer: dict[str, list[tuple[int, ONNXNode]]],
    producer: dict[str, tuple[int, ONNXNode]],
    graph: ONNXGraph,
) -> _OutputChain | None:
    """Trace forward from AV output: Transpose -> Reshape -> O_proj (MatMul)."""
    consumers = consumer.get(tensor_name, [])
    if not consumers:
        return None

    # Level 1: Transpose (merge heads permutation)
    tp_idx, tp_node = None, None
    for idx, node in consumers:
        if node.op_type == "Transpose":
            tp_idx, tp_node = idx, node
            break
    if tp_node is None:
        return None

    # Level 2: Reshape (merge heads dimension)
    tp_out = tp_node.outputs[0]
    rs_consumers = consumer.get(tp_out, [])
    rs_idx, rs_node = None, None
    for idx, node in rs_consumers:
        if node.op_type == "Reshape":
            rs_idx, rs_node = idx, node
            break
    if rs_node is None:
        return None

    # Level 3: O projection (MatMul with weight)
    rs_out = rs_node.outputs[0]
    o_consumers = consumer.get(rs_out, [])
    o_idx, o_node = None, None
    for idx, node in o_consumers:
        if node.op_type in ("MatMul", "Gemm") and _has_weight_initializer(node, graph):
            o_idx, o_node = idx, node
            break
    if o_node is None:
        return None

    return _OutputChain(
        transpose_idx=tp_idx,
        reshape_idx=rs_idx,
        proj_idx=o_idx,
        indices=[tp_idx, rs_idx, o_idx],
    )


def _trace_back_to_matmul(
    tensor_name: str,
    producer: dict[str, tuple[int, ONNXNode]],
    graph: ONNXGraph,
    max_depth: int = 3,
) -> tuple[int | None, ONNXNode | None, list[int]]:
    """Trace backward through optional scale/mask ops to find the QK^T MatMul.

    Allows up to max_depth intermediate nodes (Mul for scale, Add for mask,
    Div for scale normalization) between the MatMul and the Softmax.

    Returns:
        (matmul_index, matmul_node, list_of_intermediate_indices)
    """
    intermediate: list[int] = []
    current = tensor_name

    for _ in range(max_depth + 1):
        if current not in producer:
            return None, None, []
        idx, node = producer[current]
        if node.op_type == "MatMul":
            return idx, node, intermediate
        # Allow scale/mask operations between QK^T and Softmax
        if node.op_type in ("Mul", "Div", "Add"):
            intermediate.append(idx)
            current = node.inputs[0]
            continue
        break

    return None, None, []


def _find_consumer_matmul(
    tensor_name: str,
    consumer: dict[str, list[tuple[int, ONNXNode]]],
    graph: ONNXGraph,
) -> tuple[int | None, ONNXNode | None]:
    """Find the attn @ V MatMul that consumes the Softmax output."""
    consumers = consumer.get(tensor_name, [])
    for idx, node in consumers:
        if node.op_type == "MatMul":
            return idx, node
    return None, None


def _has_weight_initializer(node: ONNXNode, graph: ONNXGraph) -> bool:
    """Check if a MatMul/Gemm node has a weight tensor in graph initializers."""
    for inp in node.inputs[1:]:
        if inp in graph.initializers:
            return True
    return False


def _infer_head_geometry(chain: _ProjChain, graph: ONNXGraph) -> tuple[int | None, int | None]:
    """Infer num_heads and head_dim from the Reshape node's shape constant.

    The Reshape after Q/K/V projection typically reshapes
    [batch, seq_len, hidden_size] -> [batch, seq_len, num_heads, head_dim]
    The shape tensor is usually a constant initializer like [0, 0, num_heads, head_dim]
    or [-1, seq_len, num_heads, head_dim].
    """
    reshape_node = graph.nodes[chain.reshape_idx]
    if len(reshape_node.inputs) < 2:
        return None, None

    shape_name = reshape_node.inputs[1]
    if shape_name not in graph.initializers:
        return None, None

    shape_tensor = graph.initializers[shape_name]
    shape_list = shape_tensor.tolist()

    # Expected: 4 elements [batch_dim, seq_dim, num_heads, head_dim]
    # batch_dim and seq_dim may be 0 (passthrough) or -1 (infer)
    if len(shape_list) != 4:
        return None, None

    num_heads = int(shape_list[2])
    head_dim = int(shape_list[3])

    if num_heads <= 0 or head_dim <= 0:
        return None, None

    return num_heads, head_dim


def _load_proj_weights(
    proj_layer,
    weight_name: str,
    graph: ONNXGraph,
    quantized_weights: dict[str, tuple],
) -> None:
    """Load weights into a MarlinLinear projection layer.

    If the weight is already quantized (present in quantized_weights),
    load the packed representation directly. Otherwise, quantize on the fly.
    """
    if weight_name in quantized_weights:
        packed, scales = quantized_weights[weight_name]
        proj_layer.weight_packed = packed
        proj_layer.scales = scales
    elif weight_name in graph.initializers:
        weight = graph.initializers[weight_name]
        packed, scales, _ = pack_fp4_weights(weight, group_size=proj_layer.group_size)
        proj_layer.weight_packed = packed
        proj_layer.scales = scales


# ---------------------------------------------------------------------------
# ONNX Loading
# ---------------------------------------------------------------------------


def load_onnx_model(path: str | Path) -> ONNXGraph:
    """Parse ONNX file into our graph representation.

    Args:
        path: Path to .onnx file

    Returns:
        Parsed graph structure
    """
    try:
        import onnx
    except ImportError as e:
        raise ImportError("pip install onnx to use ONNX model loading") from e

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    graph = model.graph

    device = _get_device()

    # Parse nodes
    nodes = [
        ONNXNode(
            op_type=node.op_type,
            name=node.name,
            inputs=list(node.input),
            outputs=list(node.output),
            attrs={attr.name: _parse_attr(attr) for attr in node.attribute},
        )
        for node in graph.node
    ]

    # Parse initializers (weights)
    initializers = {}
    for init in graph.initializer:
        np_array = onnx.numpy_helper.to_array(init)
        initializers[init.name] = torch.from_numpy(np_array).to(device)

    # Parse inputs/outputs
    input_names = [i.name for i in graph.input if i.name not in initializers]
    output_names = [o.name for o in graph.output]

    return ONNXGraph(
        nodes=nodes,
        inputs=input_names,
        outputs=output_names,
        initializers=initializers,
    )


def _parse_attr(attr):
    """Parse ONNX attribute to Python value."""
    import onnx

    if attr.type == onnx.AttributeProto.FLOAT:
        return attr.f
    elif attr.type == onnx.AttributeProto.INT:
        return attr.i
    elif attr.type == onnx.AttributeProto.STRING:
        return attr.s.decode("utf-8")
    elif attr.type == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    elif attr.type == onnx.AttributeProto.INTS:
        return list(attr.ints)
    return None


def _quantize_linear_weights(graph: ONNXGraph) -> dict[str, tuple]:
    """Quantize linear layer weights to FP4.

    Finds MatMul/Gemm nodes and quantizes their weight initializers.

    Args:
        graph: Parsed ONNX graph

    Returns:
        Dict of {weight_name: (packed_weights, scales)}
    """
    quantized = {}

    for node in graph.nodes:
        if node.op_type in ("MatMul", "Gemm"):
            # Weight is typically the second input
            weight_name = node.inputs[1] if len(node.inputs) > 1 else None
            if weight_name and weight_name in graph.initializers:
                weight = graph.initializers[weight_name]
                # ONNX MatMul weights are [K, N] (in_features, out_features).
                # pack_fp4_weights expects [N, K] (out_features, in_features)
                # following PyTorch convention, and transposes internally.
                # For Gemm with transB=1, weight is already [N, K].
                trans_b = node.attrs.get("transB", 0) if node.op_type == "Gemm" else 0
                if not trans_b:
                    weight = weight.T
                packed, scales, _ = pack_fp4_weights(weight, group_size=32)
                quantized[weight_name] = (packed, scales)

    return quantized
