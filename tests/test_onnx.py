"""Tests for ONNX model loading, parsing, and execution via ONNXExecutor.

All test models are created programmatically with onnx.helper (no external files).
Tests exercise:
1. Model loading and parsing
2. FP16 matmul (unquantized)
3. Quantized matmul via pack_fp4_weights
4. Two-layer FFN (Linear -> ReLU -> Linear)
5. LayerNorm
6. Transformer block (attention + FFN)
7. Weight extraction and initializer validation
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Add metal_marlin package to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import importlib.util

import mlx.core as mx
import numpy as np
import onnx
import pytest
from metal_marlin.quantize import pack_fp4_weights, unpack_fp4_weights
from onnx import TensorProto, helper, numpy_helper

# Load onnx_executor directly to avoid safetensors_loader import issues
_spec = importlib.util.spec_from_file_location(
    "onnx_executor", _ROOT / "converters" / "onnx_executor.py"
)
_onnx_mod = importlib.util.module_from_spec(_spec)
sys.modules["onnx_executor"] = _onnx_mod
_spec.loader.exec_module(_onnx_mod)
ONNXExecutor = _onnx_mod.ONNXExecutor
ONNXGraph = _onnx_mod.ONNXGraph
ONNXNode = _onnx_mod.ONNXNode
load_onnx_model = _onnx_mod.load_onnx_model
_quantize_linear_weights = _onnx_mod._quantize_linear_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_model(model: onnx.ModelProto) -> Path:
    """Save ONNX model to a temp file, returning its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx.save(model, tmp.name)
    tmp.close()
    return Path(tmp.name)


def _make_matmul_model(
    M: int, K: int, N: int, *, dtype: int = TensorProto.FLOAT16, seed: int = 42
) -> onnx.ModelProto:
    """Create a single-MatMul model: Y = X @ W where W is an initializer."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((K, N)).astype(np.float16)

    X = helper.make_tensor_value_info("X", dtype, [M, K])
    Y = helper.make_tensor_value_info("Y", dtype, [M, N])
    W_init = numpy_helper.from_array(W, name="W")

    matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"])

    graph = helper.make_graph([matmul_node], "matmul_graph", [X], [Y], [W_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _make_ffn_model(
    M: int, K: int, H: int, N: int, *, seed: int = 77
) -> onnx.ModelProto:
    """Build a 2-layer FFN: X @ W1 + b1 -> ReLU -> @ W2 + b2."""
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((K, H)).astype(np.float16)
    b1 = rng.standard_normal((H,)).astype(np.float16)
    W2 = rng.standard_normal((H, N)).astype(np.float16)
    b2 = rng.standard_normal((N,)).astype(np.float16)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [M, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [M, N])

    nodes = [
        helper.make_node("MatMul", ["X", "W1"], ["mm1"]),
        helper.make_node("Add", ["mm1", "b1"], ["lin1"]),
        helper.make_node("Relu", ["lin1"], ["relu_out"]),
        helper.make_node("MatMul", ["relu_out", "W2"], ["mm2"]),
        helper.make_node("Add", ["mm2", "b2"], ["Y"]),
    ]

    initializers = [
        numpy_helper.from_array(W1, "W1"),
        numpy_helper.from_array(b1, "b1"),
        numpy_helper.from_array(W2, "W2"),
        numpy_helper.from_array(b2, "b2"),
    ]

    graph = helper.make_graph(nodes, "ffn", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _make_layernorm_model(
    M: int, N: int, *, epsilon: float = 1e-5, seed: int = 55
) -> onnx.ModelProto:
    """Create LayerNorm model with learned scale and bias."""
    rng = np.random.default_rng(seed)
    scale = rng.uniform(0.5, 2.0, size=(N,)).astype(np.float32)
    bias = rng.standard_normal((N,)).astype(np.float32)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, N])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])

    ln_node = helper.make_node(
        "LayerNormalization",
        inputs=["X", "scale", "bias"],
        outputs=["Y"],
        epsilon=epsilon,
        axis=-1,
    )

    initializers = [
        numpy_helper.from_array(scale, "scale"),
        numpy_helper.from_array(bias, "bias"),
    ]

    graph = helper.make_graph([ln_node], "ln_graph", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _make_transformer_model(
    M: int, D: int, H: int, *, seed: int = 2024
) -> tuple[onnx.ModelProto, dict[str, np.ndarray]]:
    """Build simplified transformer: LN -> Attn(Q@K^T -> Softmax -> @V) -> Add -> LN -> FFN -> Add.

    Single-head attention with ReLU FFN. All weights float32.

    Args:
        M: Sequence length
        D: Model dimension
        H: FFN hidden dimension

    Returns:
        (model, weight_dict) for reference computation
    """
    rng = np.random.default_rng(seed)

    Wq = rng.standard_normal((D, D)).astype(np.float32) * 0.02
    Wk = rng.standard_normal((D, D)).astype(np.float32) * 0.02
    Wv = rng.standard_normal((D, D)).astype(np.float32) * 0.02
    Wo = rng.standard_normal((D, D)).astype(np.float32) * 0.02
    W1 = rng.standard_normal((D, H)).astype(np.float32) * 0.02
    W2 = rng.standard_normal((H, D)).astype(np.float32) * 0.02
    ln1_scale = np.ones((D,), dtype=np.float32)
    ln1_bias = np.zeros((D,), dtype=np.float32)
    ln2_scale = np.ones((D,), dtype=np.float32)
    ln2_bias = np.zeros((D,), dtype=np.float32)
    scale_val = np.array([1.0 / np.sqrt(D)], dtype=np.float32)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, D])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, D])

    nodes = [
        # Pre-norm 1
        helper.make_node("LayerNormalization", ["X", "ln1_s", "ln1_b"], ["ln1_out"],
                         epsilon=1e-5, axis=-1),
        # Q, K, V projections
        helper.make_node("MatMul", ["ln1_out", "Wq"], ["Q"]),
        helper.make_node("MatMul", ["ln1_out", "Wk"], ["K"]),
        helper.make_node("MatMul", ["ln1_out", "Wv"], ["V"]),
        # K^T
        helper.make_node("Transpose", ["K"], ["Kt"], perm=[1, 0]),
        # Attention scores: Q @ K^T * scale
        helper.make_node("MatMul", ["Q", "Kt"], ["qk"]),
        helper.make_node("Mul", ["qk", "attn_scale"], ["qk_scaled"]),
        helper.make_node("Softmax", ["qk_scaled"], ["attn_weights"], axis=-1),
        # Attention output
        helper.make_node("MatMul", ["attn_weights", "V"], ["attn_out"]),
        helper.make_node("MatMul", ["attn_out", "Wo"], ["proj_out"]),
        # Residual
        helper.make_node("Add", ["X", "proj_out"], ["res1"]),
        # Pre-norm 2
        helper.make_node("LayerNormalization", ["res1", "ln2_s", "ln2_b"], ["ln2_out"],
                         epsilon=1e-5, axis=-1),
        # FFN
        helper.make_node("MatMul", ["ln2_out", "W1"], ["ffn_h"]),
        helper.make_node("Relu", ["ffn_h"], ["ffn_relu"]),
        helper.make_node("MatMul", ["ffn_relu", "W2"], ["ffn_out"]),
        # Residual
        helper.make_node("Add", ["res1", "ffn_out"], ["Y"]),
    ]

    initializers = [
        numpy_helper.from_array(Wq, "Wq"),
        numpy_helper.from_array(Wk, "Wk"),
        numpy_helper.from_array(Wv, "Wv"),
        numpy_helper.from_array(Wo, "Wo"),
        numpy_helper.from_array(W1, "W1"),
        numpy_helper.from_array(W2, "W2"),
        numpy_helper.from_array(ln1_scale, "ln1_s"),
        numpy_helper.from_array(ln1_bias, "ln1_b"),
        numpy_helper.from_array(ln2_scale, "ln2_s"),
        numpy_helper.from_array(ln2_bias, "ln2_b"),
        numpy_helper.from_array(scale_val, "attn_scale"),
    ]

    graph = helper.make_graph(nodes, "transformer", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    weights = {"Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo,
               "W1": W1, "W2": W2, "scale": scale_val}
    return model, weights


def _numpy_layer_norm(
    x: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray | None,
    axis: int = -1,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Reference LayerNorm matching ONNX spec: normalize over [axis, ..., rank-1]."""
    rank = x.ndim
    ax = axis if axis >= 0 else axis + rank
    axes = tuple(range(ax, rank))
    mean = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    normalized = (x - mean) / np.sqrt(var + epsilon)
    result = normalized * scale
    if bias is not None:
        result = result + bias
    return result


# ---------------------------------------------------------------------------
# Test 1: Load and parse single-layer model
# ---------------------------------------------------------------------------


class TestLoadSimpleOnnx:
    """Load and parse a single-layer ONNX model via onnx.helper."""

    def test_parse_graph_structure(self):
        model = _make_matmul_model(M=4, K=64, N=32)
        path = _save_model(model)

        graph = load_onnx_model(path)

        assert isinstance(graph, ONNXGraph)
        assert len(graph.nodes) == 1
        assert graph.nodes[0].op_type == "MatMul"
        assert graph.nodes[0].inputs == ["X", "W"]
        assert graph.nodes[0].outputs == ["Y"]

    def test_input_output_names(self):
        model = _make_matmul_model(M=4, K=64, N=32)
        path = _save_model(model)

        graph = load_onnx_model(path)

        assert "X" in graph.inputs
        assert "W" not in graph.inputs  # Initializers excluded from inputs
        assert "Y" in graph.outputs

    def test_initializer_loaded_as_mx_array(self):
        model = _make_matmul_model(M=4, K=64, N=32)
        path = _save_model(model)

        graph = load_onnx_model(path)

        assert "W" in graph.initializers
        w = graph.initializers["W"]
        assert isinstance(w, mx.array)
        assert w.shape == (64, 32)

    def test_multi_node_graph(self):
        """Parse a graph with multiple nodes (MatMul + Add for linear layer)."""
        rng = np.random.default_rng(7)
        K, N = 64, 32

        W = rng.standard_normal((K, N)).astype(np.float16)
        bias = rng.standard_normal((N,)).astype(np.float16)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [None, K])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [None, N])
        W_init = numpy_helper.from_array(W, name="W")
        bias_init = numpy_helper.from_array(bias, name="bias")

        matmul = helper.make_node("MatMul", ["X", "W"], ["mm_out"])
        add = helper.make_node("Add", ["mm_out", "bias"], ["Y"])

        graph = helper.make_graph([matmul, add], "linear", [X], [Y], [W_init, bias_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        path = _save_model(model)

        parsed = load_onnx_model(path)

        assert len(parsed.nodes) == 2
        assert parsed.nodes[0].op_type == "MatMul"
        assert parsed.nodes[1].op_type == "Add"
        assert "W" in parsed.initializers
        assert "bias" in parsed.initializers

    def test_node_attributes_parsed(self):
        """Gemm node attributes (transB, alpha, beta) are correctly parsed."""
        rng = np.random.default_rng(44)
        K, N = 64, 32
        W = rng.standard_normal((N, K)).astype(np.float32)  # Transposed layout
        bias = rng.standard_normal((N,)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, K])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, N])

        gemm = helper.make_node(
            "Gemm", ["X", "W", "bias"], ["Y"],
            transB=1, alpha=2.0, beta=0.5,
        )

        initializers = [
            numpy_helper.from_array(W, "W"),
            numpy_helper.from_array(bias, "bias"),
        ]

        graph = helper.make_graph([gemm], "gemm_test", [X], [Y], initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        path = _save_model(model)

        parsed = load_onnx_model(path)

        node = parsed.nodes[0]
        assert node.op_type == "Gemm"
        assert node.attrs["transB"] == 1
        assert node.attrs["alpha"] == pytest.approx(2.0)
        assert node.attrs["beta"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Test 2: Standard matmul without quantization
# ---------------------------------------------------------------------------


class TestMatmulFP16:
    """FP16 MatMul execution without quantization."""

    def test_fp16_matmul_accuracy(self):
        """Execute MatMul in FP16, compare to numpy reference."""
        M, K, N = 8, 64, 32
        model = _make_matmul_model(M, K, N)
        path = _save_model(model)

        executor = ONNXExecutor.from_file(path, quantize=False)

        rng = np.random.default_rng(123)
        x_np = rng.standard_normal((M, K)).astype(np.float16)
        x = mx.array(x_np)

        result = executor(X=x)
        y = result["Y"]

        assert y.shape == (M, N)

        w_np = np.array(executor.graph.initializers["W"].astype(mx.float32))
        ref = x_np.astype(np.float32) @ w_np
        y_np = np.array(y.astype(mx.float32))

        np.testing.assert_allclose(y_np, ref, rtol=5e-3, atol=1e-3)

    def test_unquantized_no_quantized_weights(self):
        """When quantize=False, executor has no quantized weights."""
        model = _make_matmul_model(4, 64, 32)
        path = _save_model(model)

        executor = ONNXExecutor.from_file(path, quantize=False)
        assert executor.quantized_weights == {}

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batch_sizes(self, batch_size: int):
        """Output shape is correct for varying batch sizes."""
        K, N = 128, 64
        model = _make_matmul_model(batch_size, K, N)
        path = _save_model(model)

        executor = ONNXExecutor.from_file(path, quantize=False)
        x = mx.random.normal((batch_size, K))
        result = executor(X=x)

        assert result["Y"].shape == (batch_size, N)

    def test_dispatch_fallback_to_matmul(self):
        """MatMul without quantized weights uses mx.matmul (exact result)."""
        M, K, N = 4, 32, 16

        mx.random.seed(123)
        weight_kn = mx.random.normal((K, N)).astype(mx.float16)
        x = mx.random.normal((M, K)).astype(mx.float16)
        mx.eval(weight_kn, x)

        ref = x @ weight_kn
        mx.eval(ref)

        node = ONNXNode(op_type="MatMul", name="mm", inputs=["X", "W"], outputs=["Y"])
        graph = ONNXGraph(
            nodes=[node], inputs=["X"], outputs=["Y"],
            initializers={"W": weight_kn},
        )
        executor = ONNXExecutor(graph)

        result = executor(X=x)
        y = result["Y"]
        mx.eval(y)

        np.testing.assert_allclose(
            np.array(y.astype(mx.float32)),
            np.array(ref.astype(mx.float32)),
            rtol=1e-3,
        )


# ---------------------------------------------------------------------------
# Test 3: Quantized matmul via pack_fp4_weights
# ---------------------------------------------------------------------------


class TestMatmulQuantized:
    """Quantized MatMul via pack_fp4_weights and ONNXExecutor."""

    def test_pack_unpack_roundtrip(self):
        """pack_fp4_weights -> unpack_fp4_weights approximately recovers original."""
        rng = np.random.default_rng(99)
        K, N = 128, 64
        w = mx.array(rng.standard_normal((K, N)).astype(np.float16))

        packed, scales, meta = pack_fp4_weights(w, group_size=128)

        assert packed.dtype == mx.uint32
        assert scales.dtype == mx.float16
        assert meta["orig_K"] == K
        assert meta["orig_N"] == N

        recovered = unpack_fp4_weights(packed, scales, meta)
        assert recovered.shape == (K, N)

        w_f32 = np.array(w.astype(mx.float32))
        r_f32 = np.array(recovered.astype(mx.float32))
        mask = np.abs(w_f32) > 0.1
        if mask.sum() > 0:
            rel_error = np.abs(w_f32[mask] - r_f32[mask]) / (np.abs(w_f32[mask]) + 1e-7)
            assert np.median(rel_error) < 0.5

    def test_quantized_executor_output_shape(self):
        """ONNXExecutor with quantize=True produces correct output shape."""
        M, K, N = 8, 128, 64
        model = _make_matmul_model(M, K, N)
        path = _save_model(model)

        executor = ONNXExecutor.from_file(path, quantize=True)

        assert "W" in executor.quantized_weights
        packed, scales = executor.quantized_weights["W"]
        assert packed.dtype == mx.uint32

        x = mx.random.normal((M, K))
        result = executor(X=x)

        assert result["Y"].shape == (M, N)

    def test_quantize_linear_weights_finds_matmul(self):
        """_quantize_linear_weights correctly identifies MatMul weight initializers."""
        M, K, N = 4, 128, 64
        model = _make_matmul_model(M, K, N)
        path = _save_model(model)

        graph = load_onnx_model(path)
        quantized = _quantize_linear_weights(graph)

        assert "W" in quantized
        packed, scales = quantized["W"]
        assert isinstance(packed, mx.array)
        assert isinstance(scales, mx.array)

    def test_padding_for_non_aligned_k(self):
        """pack_fp4_weights handles K not divisible by group_size."""
        K, N = 100, 64
        w = mx.random.normal((K, N))

        packed, scales, meta = pack_fp4_weights(w, group_size=128, pad_k=True)

        assert meta["orig_K"] == 100
        assert meta["padded_K"] == 128
        assert meta["orig_N"] == 64

        recovered = unpack_fp4_weights(packed, scales, meta)
        assert recovered.shape == (100, 64)

    def test_quantized_vs_fp16_closeness(self):
        """Quantized MatMul output is within FP4 tolerance of FP16 reference."""
        M, K, N = 4, 64, 32
        group_size = 32

        mx.random.seed(42)
        weight_kn = mx.random.normal((K, N)).astype(mx.float16)
        x = mx.random.normal((M, K)).astype(mx.float16)
        mx.eval(weight_kn, x)

        ref = x @ weight_kn
        mx.eval(ref)

        # Quantize: ONNX weight is [K,N], pack_fp4_weights wants [N,K]
        packed, scales = _onnx_mod.pack_fp4_weights(weight_kn.T, group_size=group_size)

        node = ONNXNode(op_type="MatMul", name="mm", inputs=["X", "W"], outputs=["Y"])
        graph = ONNXGraph(
            nodes=[node], inputs=["X"], outputs=["Y"],
            initializers={"W": weight_kn},
        )
        executor = ONNXExecutor(graph, quantized_weights={"W": (packed, scales)})

        result = executor(X=x)
        y = result["Y"]
        mx.eval(y)

        ref_np = np.array(ref.astype(mx.float32))
        y_np = np.array(y.astype(mx.float32))

        # FP4/INT4 allow higher outliers; check bulk accuracy and cap worst-case error.
        abs_diff = np.abs(y_np - ref_np)
        assert np.percentile(abs_diff, 95) < 2.0, "95th percentile error too high"
        assert np.max(abs_diff) < 10.0, "Max error too high"

    def test_dispatch_routing_wrong_name(self):
        """Quantized path is NOT used when weight name doesn't match."""
        K, N = 64, 32
        group_size = 32

        mx.random.seed(11)
        weight_kn = mx.random.normal((K, N)).astype(mx.float16)
        x = mx.random.normal((2, K)).astype(mx.float16)
        mx.eval(weight_kn, x)

        packed, scales = _onnx_mod.pack_fp4_weights(weight_kn.T, group_size=group_size)

        node = ONNXNode(op_type="MatMul", name="mm", inputs=["X", "W"], outputs=["Y"])
        graph = ONNXGraph(
            nodes=[node], inputs=["X"], outputs=["Y"],
            initializers={"W": weight_kn},
        )
        # Register under wrong name
        executor = ONNXExecutor(graph, quantized_weights={"OTHER_W": (packed, scales)})

        result = executor(X=x)
        y = result["Y"]
        mx.eval(y)

        ref = x @ weight_kn
        mx.eval(ref)
        np.testing.assert_allclose(
            np.array(y.astype(mx.float32)),
            np.array(ref.astype(mx.float32)),
            rtol=1e-3,
        )


# ---------------------------------------------------------------------------
# Test 4: Feed-forward: Linear -> ReLU -> Linear
# ---------------------------------------------------------------------------


class TestTwoLayerFFN:
    """Two-layer feed-forward network: Linear -> ReLU -> Linear."""

    def test_ffn_structure(self):
        model = _make_ffn_model(4, 128, 64, 32)
        path = _save_model(model)

        graph = load_onnx_model(path)
        assert len(graph.nodes) == 5
        op_types = [n.op_type for n in graph.nodes]
        assert op_types == ["MatMul", "Add", "Relu", "MatMul", "Add"]

    def test_ffn_execution_unquantized(self):
        """Execute 2-layer FFN without quantization, compare to numpy."""
        M, K, H, N = 8, 128, 64, 32
        model = _make_ffn_model(M, K, H, N)
        path = _save_model(model)

        executor = ONNXExecutor.from_file(path, quantize=False)

        rng = np.random.default_rng(200)
        x_np = rng.standard_normal((M, K)).astype(np.float16)
        x = mx.array(x_np)

        result = executor(X=x)
        y = np.array(result["Y"].astype(mx.float32))

        # Reference computation in float32
        W1 = np.array(executor.graph.initializers["W1"].astype(mx.float32))
        b1 = np.array(executor.graph.initializers["b1"].astype(mx.float32))
        W2 = np.array(executor.graph.initializers["W2"].astype(mx.float32))
        b2 = np.array(executor.graph.initializers["b2"].astype(mx.float32))

        x_f32 = x_np.astype(np.float32)
        ref = x_f32 @ W1 + b1
        ref = np.maximum(ref, 0)  # ReLU
        ref = ref @ W2 + b2

        np.testing.assert_allclose(y, ref, rtol=5e-2, atol=5e-2)

    def test_ffn_quantized_shape(self):
        """Quantized FFN produces correct output shape."""
        M, K, H, N = 4, 128, 64, 32
        model = _make_ffn_model(M, K, H, N)
        path = _save_model(model)

        executor = ONNXExecutor.from_file(path, quantize=True)

        # Both W1 and W2 should be quantized
        assert "W1" in executor.quantized_weights
        assert "W2" in executor.quantized_weights

        x = mx.random.normal((M, K))
        result = executor(X=x)
        assert result["Y"].shape == (M, N)

    def test_ffn_initializer_count(self):
        """FFN model has exactly 4 initializers (W1, b1, W2, b2)."""
        model = _make_ffn_model(4, 128, 64, 32)
        path = _save_model(model)
        graph = load_onnx_model(path)

        assert set(graph.initializers.keys()) == {"W1", "b1", "W2", "b2"}
        assert graph.initializers["W1"].shape == (128, 64)
        assert graph.initializers["b1"].shape == (64,)
        assert graph.initializers["W2"].shape == (64, 32)
        assert graph.initializers["b2"].shape == (32,)

    @pytest.mark.parametrize("hidden_size", [32, 64, 128])
    def test_ffn_hidden_sizes(self, hidden_size: int):
        """FFN works with various hidden layer sizes."""
        M, K, N = 4, 128, 32
        model = _make_ffn_model(M, K, hidden_size, N, seed=hidden_size)
        path = _save_model(model)

        executor = ONNXExecutor.from_file(path, quantize=False)
        x = mx.random.normal((M, K))
        result = executor(X=x)

        assert result["Y"].shape == (M, N)


# ---------------------------------------------------------------------------
# Test 5: Standalone LayerNorm validation
# ---------------------------------------------------------------------------


class TestLayerNorm:
    """LayerNorm accuracy tests using onnx.helper-created models."""

    def test_layernorm_accuracy(self):
        """LayerNorm output matches numpy reference."""
        M, N = 4, 128
        epsilon = 1e-5
        model = _make_layernorm_model(M, N, epsilon=epsilon)
        path = _save_model(model)

        executor = ONNXExecutor.from_file(path, quantize=False)

        rng = np.random.default_rng(300)
        x_np = rng.standard_normal((M, N)).astype(np.float32)
        x = mx.array(x_np)

        result = executor(X=x)
        y = np.array(result["Y"])

        scale = np.array(executor.graph.initializers["scale"])
        bias = np.array(executor.graph.initializers["bias"])
        ref = _numpy_layer_norm(x_np, scale, bias, epsilon=epsilon)

        np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-5)

    def test_layernorm_unit_params(self):
        """LayerNorm with scale=1, bias=0 is equivalent to standardization."""
        M, N = 8, 64
        rng = np.random.default_rng(88)

        scale = np.ones((N,), dtype=np.float32)
        bias = np.zeros((N,), dtype=np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, N])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
        ln_node = helper.make_node(
            "LayerNormalization", ["X", "scale", "bias"], ["Y"],
            epsilon=1e-5, axis=-1,
        )
        initializers = [
            numpy_helper.from_array(scale, "scale"),
            numpy_helper.from_array(bias, "bias"),
        ]
        graph = helper.make_graph([ln_node], "ln", [X], [Y], initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        path = _save_model(model)

        executor = ONNXExecutor.from_file(path, quantize=False)
        x_np = rng.standard_normal((M, N)).astype(np.float32)
        result = executor(X=mx.array(x_np))
        y = np.array(result["Y"])

        # With unit scale and zero bias, output should have mean~0, std~1 per row
        assert np.allclose(y.mean(axis=-1), 0, atol=1e-5)
        assert np.allclose(y.var(axis=-1), 1.0, atol=1e-4)

    def test_layernorm_preserves_shape(self):
        M, N = 16, 256
        model = _make_layernorm_model(M, N)
        path = _save_model(model)

        executor = ONNXExecutor.from_file(path, quantize=False)
        x = mx.random.normal((M, N))
        result = executor(X=x)

        assert result["Y"].shape == (M, N)

    def test_layernorm_numerical_stability(self):
        """Near-zero variance input doesn't produce inf/nan."""
        M, N = 4, 32
        x_np = np.full((M, N), 3.14, dtype=np.float32)
        x_np[0, 0] += 1e-7

        scale = np.ones((N,), dtype=np.float32)
        bias = np.zeros((N,), dtype=np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, N])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
        ln_node = helper.make_node(
            "LayerNormalization", ["X", "scale", "bias"], ["Y"],
            epsilon=1e-5, axis=-1,
        )
        initializers = [
            numpy_helper.from_array(scale, "scale"),
            numpy_helper.from_array(bias, "bias"),
        ]
        graph = helper.make_graph([ln_node], "ln", [X], [Y], initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        path = _save_model(model)

        executor = ONNXExecutor.from_file(path, quantize=False)
        result = executor(X=mx.array(x_np))
        y = np.array(result["Y"])

        assert np.all(np.isfinite(y))

    def test_layernorm_no_bias(self):
        """LayerNorm with only scale (2 inputs, no bias)."""
        M, N = 4, 16
        rng = np.random.default_rng(123)
        scale = np.ones(N, dtype=np.float32) * 2.0

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, N])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
        ln_node = helper.make_node(
            "LayerNormalization", ["X", "scale"], ["Y"],
            epsilon=1e-5, axis=-1,
        )
        initializers = [numpy_helper.from_array(scale, "scale")]
        graph = helper.make_graph([ln_node], "ln", [X], [Y], initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        path = _save_model(model)

        executor = ONNXExecutor.from_file(path, quantize=False)
        x_np = rng.standard_normal((M, N)).astype(np.float32)
        result = executor(X=mx.array(x_np))
        y = np.array(result["Y"])

        ref = _numpy_layer_norm(x_np, scale, None, epsilon=1e-5)
        np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 6: Single transformer layer
# ---------------------------------------------------------------------------


class TestTransformerBlock:
    """Single transformer layer: pre-norm attention + FFN with residual."""

    def test_transformer_structure(self):
        """Verify graph has expected node count and types."""
        model, _ = _make_transformer_model(M=8, D=64, H=128)
        path = _save_model(model)
        graph = load_onnx_model(path)

        assert len(graph.nodes) == 16
        op_types = [n.op_type for n in graph.nodes]
        # 8 MatMuls: Q/K/V projections (3), Q@K^T (1), attn@V (1), O proj (1), FFN up/down (2)
        assert op_types.count("MatMul") == 8
        assert op_types.count("LayerNormalization") == 2
        assert op_types.count("Softmax") == 1
        assert op_types.count("Add") == 2
        assert op_types.count("Relu") == 1
        assert op_types.count("Transpose") == 1
        assert op_types.count("Mul") == 1

    def test_transformer_execution(self):
        """Execute transformer block and verify output matches reference."""
        M, D, H = 8, 64, 128
        model, weights = _make_transformer_model(M, D, H)
        path = _save_model(model)

        executor = ONNXExecutor.from_file(path, quantize=False)

        rng = np.random.default_rng(500)
        x_np = rng.standard_normal((M, D)).astype(np.float32)
        x = mx.array(x_np)

        result = executor(X=x)
        y = np.array(result["Y"])

        # Reference computation
        def ln(x, eps=1e-5):
            mean = x.mean(axis=-1, keepdims=True)
            var = x.var(axis=-1, keepdims=True)
            return (x - mean) / np.sqrt(var + eps)

        h = ln(x_np)
        Q = h @ weights["Wq"]
        K = h @ weights["Wk"]
        V = h @ weights["Wv"]

        scores = (Q @ K.T) * weights["scale"]
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        attn_out = attn @ V
        proj = attn_out @ weights["Wo"]
        res1 = x_np + proj

        h2 = ln(res1)
        ffn = np.maximum(h2 @ weights["W1"], 0) @ weights["W2"]
        ref = res1 + ffn

        np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4)

    def test_transformer_output_shape(self):
        """Output shape matches input shape (residual connections)."""
        M, D, H = 16, 32, 64
        model, _ = _make_transformer_model(M, D, H)
        path = _save_model(model)

        executor = ONNXExecutor.from_file(path, quantize=False)
        x = mx.random.normal((M, D))
        result = executor(X=x)

        assert result["Y"].shape == (M, D)

    def test_transformer_initializer_count(self):
        """Transformer has 11 initializers (6 weight matrices + 4 LN params + scale)."""
        model, _ = _make_transformer_model(M=8, D=64, H=128)
        path = _save_model(model)
        graph = load_onnx_model(path)

        expected_names = {"Wq", "Wk", "Wv", "Wo", "W1", "W2",
                          "ln1_s", "ln1_b", "ln2_s", "ln2_b", "attn_scale"}
        assert set(graph.initializers.keys()) == expected_names

    @pytest.mark.parametrize("seq_len", [4, 8, 16])
    def test_transformer_sequence_lengths(self, seq_len: int):
        """Transformer handles various sequence lengths."""
        D, H = 32, 64
        model, _ = _make_transformer_model(M=seq_len, D=D, H=H, seed=seq_len)
        path = _save_model(model)

        executor = ONNXExecutor.from_file(path, quantize=False)
        x = mx.random.normal((seq_len, D))
        result = executor(X=x)

        assert result["Y"].shape == (seq_len, D)


# ---------------------------------------------------------------------------
# Test 7: Extract and validate initializers
# ---------------------------------------------------------------------------


class TestWeightExtraction:
    """Extract and validate ONNX initializers (weights)."""

    def test_extract_all_initializers(self):
        """All weight initializers are correctly extracted."""
        rng = np.random.default_rng(11)
        K, N = 128, 64
        W = rng.standard_normal((K, N)).astype(np.float16)
        bias = rng.standard_normal((N,)).astype(np.float16)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [None, K])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [None, N])

        nodes = [
            helper.make_node("MatMul", ["X", "W"], ["mm_out"]),
            helper.make_node("Add", ["mm_out", "bias"], ["Y"]),
        ]
        initializers = [
            numpy_helper.from_array(W, "W"),
            numpy_helper.from_array(bias, "bias"),
        ]

        graph = helper.make_graph(nodes, "linear", [X], [Y], initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        path = _save_model(model)

        parsed = load_onnx_model(path)

        assert set(parsed.initializers.keys()) == {"W", "bias"}
        assert parsed.initializers["W"].shape == (K, N)
        assert parsed.initializers["bias"].shape == (N,)

    def test_initializer_values_preserved(self):
        """Weight values survive the save-load-parse cycle exactly."""
        rng = np.random.default_rng(22)
        W_np = rng.standard_normal((64, 32)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 64])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 32])
        W_init = numpy_helper.from_array(W_np, "W")

        node = helper.make_node("MatMul", ["X", "W"], ["Y"])
        graph = helper.make_graph([node], "test", [X], [Y], [W_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        path = _save_model(model)

        parsed = load_onnx_model(path)
        loaded_w = np.array(parsed.initializers["W"])

        np.testing.assert_array_equal(loaded_w, W_np)

    def test_multiple_dtypes(self):
        """Initializers with different dtypes are handled correctly."""
        W_f32 = np.ones((32, 16), dtype=np.float32) * 0.5
        W_f16 = np.ones((16, 8), dtype=np.float16) * 0.25

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 32])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [None, 8])

        nodes = [
            helper.make_node("MatMul", ["X", "W1"], ["h"]),
            helper.make_node("MatMul", ["h", "W2"], ["Y"]),
        ]
        initializers = [
            numpy_helper.from_array(W_f32, "W1"),
            numpy_helper.from_array(W_f16, "W2"),
        ]

        graph = helper.make_graph(nodes, "multi_dtype", [X], [Y], initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        path = _save_model(model)

        parsed = load_onnx_model(path)

        w1 = parsed.initializers["W1"]
        w2 = parsed.initializers["W2"]

        np.testing.assert_allclose(np.array(w1.astype(mx.float32)), W_f32, atol=1e-6)
        np.testing.assert_allclose(
            np.array(w2.astype(mx.float32)), W_f16.astype(np.float32), atol=1e-3
        )

    def test_large_initializer_count(self):
        """Graph with many initializers parses all of them."""
        num_weights = 20
        rng = np.random.default_rng(33)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 32])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 32])

        initializers = []
        nodes = []
        prev_output = "X"

        for i in range(num_weights):
            w_name = f"W{i}"
            out_name = f"h{i}" if i < num_weights - 1 else "Y"
            W = rng.standard_normal((32, 32)).astype(np.float32) * 0.01
            initializers.append(numpy_helper.from_array(W, w_name))
            nodes.append(helper.make_node("MatMul", [prev_output, w_name], [out_name]))
            prev_output = out_name

        graph = helper.make_graph(nodes, "deep", [X], [Y], initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        path = _save_model(model)

        parsed = load_onnx_model(path)

        assert len(parsed.initializers) == num_weights
        for i in range(num_weights):
            assert f"W{i}" in parsed.initializers
            assert parsed.initializers[f"W{i}"].shape == (32, 32)

    def test_scalar_initializer(self):
        """Scalar (0-d) initializer is handled correctly."""
        scale = np.array(0.125, dtype=np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 32])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 32])
        scale_init = numpy_helper.from_array(scale, "scale")

        node = helper.make_node("Mul", ["X", "scale"], ["Y"])
        graph = helper.make_graph([node], "scalar", [X], [Y], [scale_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        path = _save_model(model)

        parsed = load_onnx_model(path)

        s = parsed.initializers["scale"]
        assert s.shape == ()
        np.testing.assert_allclose(float(s), 0.125, atol=1e-7)

    def test_1d_initializer(self):
        """1-D bias vector is correctly extracted."""
        bias = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 4])
        bias_init = numpy_helper.from_array(bias, "bias")

        node = helper.make_node("Add", ["X", "bias"], ["Y"])
        graph = helper.make_graph([node], "bias_add", [X], [Y], [bias_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        path = _save_model(model)

        parsed = load_onnx_model(path)

        b = parsed.initializers["bias"]
        assert b.shape == (4,)
        np.testing.assert_array_equal(np.array(b), bias)
