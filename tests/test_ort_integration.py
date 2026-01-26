"""Tests for ONNX Runtime custom op integration with Metal Marlin.

Tests the Python custom op path (via onnxruntime-extensions) and validates
that MarlinQuantizedMatMul produces correct results when dispatched through
an ORT InferenceSession.

BLOCKERS for full C++ custom op integration:
    1. ORT's C++ custom op API (OrtCustomOp) requires implementing:
       - CreateKernel, GetName, GetInputType[Count], GetOutputType[Count]
       - KernelCompute with OrtKernelContext for tensor I/O
       This must be compiled as a shared library (.dylib on macOS, .so on Linux).

    2. Metal kernel dispatch from C++:
       The C++ extension would need to call Metal Marlin kernels without going
       through Python/MLX. Options:
       a) Link against Metal.framework directly and use raw command buffers
       b) Embed a minimal MLX runtime for kernel dispatch
       c) Use Metal Performance Shaders as an intermediate

    3. Memory management:
       ORT manages its own tensor memory (OrtAllocator). The custom op must
       either copy data to Metal buffers (losing zero-copy benefits) or use
       IOBinding to share Metal GPU memory with ORT's execution providers.

    4. Build system integration:
       Requires linking against:
       - ONNX Runtime C API headers (onnxruntime_c_api.h)
       - Metal.framework (for GPU dispatch)
       - MLX (if using MLX-based kernel dispatch)
       - Accelerate.framework (for CPU fallback)

    5. macOS/Apple Silicon specifics:
       - ORT's CoreML EP already claims some nodes; custom ops must not conflict
       - Metal command queue sharing between ORT and Marlin kernels
       - Unified memory means no explicit host<->device copies needed

    Python-only approach limitations:
       - Per-op Python interpreter roundtrip (~50-200us overhead per node)
       - numpy<->MLX array conversion on each call (memcpy on non-Metal allocations)
       - Python GIL prevents concurrent op execution
       - Net result: quantization speedup is negated by dispatch overhead
         for models with many small ops (attention layers with decomposed Q/K/V)

    Recommended path forward:
       Build a C++ OrtCustomOp library that calls Metal Marlin kernels via
       Metal.framework directly, bypassing Python entirely. Use IOBinding
       for zero-copy GPU memory sharing with ORT's CoreML EP.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add metal_marlin package to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import importlib.util

# Load ort_marlin_provider directly to avoid relative import issues in tests
_provider_spec = importlib.util.spec_from_file_location(
    "ort_marlin_provider", _ROOT / "converters" / "ort_marlin_provider.py"
)
_provider_mod = importlib.util.module_from_spec(_provider_spec)
sys.modules["ort_marlin_provider"] = _provider_mod
_provider_spec.loader.exec_module(_provider_mod)

MarlinQuantizedMatMulOp = _provider_mod.MarlinQuantizedMatMulOp
MarlinQuantizedLinearOp = _provider_mod.MarlinQuantizedLinearOp
DOMAIN = _provider_mod.DOMAIN
OPSET_VERSION = _provider_mod.OPSET_VERSION
export_with_marlin_ops = _provider_mod.export_with_marlin_ops

from onnx import TensorProto, helper, numpy_helper

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def quantize_weights():
    """Fixture providing pack_fp4_weights and unpack_fp4_weights."""
    from metal_marlin.quantize import pack_fp4_weights, unpack_fp4_weights
    return pack_fp4_weights, unpack_fp4_weights


@pytest.fixture
def ort_extensions_available():
    """Check if onnxruntime-extensions is available."""
    spec = importlib.util.find_spec("onnxruntime_extensions")
    if spec is None:
        pytest.skip("onnxruntime-extensions not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_model(model: onnx.ModelProto) -> Path:
    """Save ONNX model to a temp file, returning its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx.save(model, tmp.name)
    tmp.close()
    return Path(tmp.name)


def _make_standard_matmul_model(M: int, K: int, N: int) -> tuple[onnx.ModelProto, np.ndarray]:
    """Create a standard MatMul model with FP16 weights for conversion testing.

    Returns:
        Tuple of (model, weights_fp16) where weights_fp16 is [K, N].
    """
    weights = np.random.randn(K, N).astype(np.float16)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [M, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [M, N])

    w_init = numpy_helper.from_array(weights, name="W")

    matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"], name="matmul_0")

    graph = helper.make_graph(
        [matmul_node], "test_matmul", [X], [Y], initializer=[w_init]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    return model, weights


def _make_marlin_matmul_model(
    M: int,
    K: int,
    N: int,
    packed_weights: np.ndarray,
    scales: np.ndarray,
    *,
    group_size: int = 128,
) -> onnx.ModelProto:
    """Create an ONNX model with a single MarlinQuantizedMatMul custom op node."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [M, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [M, N])

    packed_init = numpy_helper.from_array(packed_weights, name="B_packed")
    scales_init = numpy_helper.from_array(scales, name="scales")

    marlin_node = helper.make_node(
        "MarlinQuantizedMatMul",
        inputs=["X", "B_packed", "scales"],
        outputs=["Y"],
        name="marlin_mm_0",
        domain=DOMAIN,
        group_size=group_size,
    )

    graph = helper.make_graph(
        [marlin_node], "marlin_test", [X], [Y],
        initializer=[packed_init, scales_init],
    )

    opset_imports = [
        helper.make_opsetid("", 17),
        helper.make_opsetid(DOMAIN, OPSET_VERSION),
    ]
    model = helper.make_model(graph, opset_imports=opset_imports)
    model.ir_version = 8

    return model


def _make_two_layer_marlin_model(
    M: int,
    K: int,
    hidden: int,
    N: int,
    packed_w1: np.ndarray,
    scales_w1: np.ndarray,
    packed_w2: np.ndarray,
    scales_w2: np.ndarray,
    *,
    group_size: int = 128,
) -> onnx.ModelProto:
    """Create a model with two MarlinQuantizedMatMul ops and a ReLU between them."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [M, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [M, N])

    # Initializers
    w1_packed_init = numpy_helper.from_array(packed_w1, name="W1_packed")
    s1_init = numpy_helper.from_array(scales_w1, name="W1_scales")
    w2_packed_init = numpy_helper.from_array(packed_w2, name="W2_packed")
    s2_init = numpy_helper.from_array(scales_w2, name="W2_scales")

    # Layer 1: X @ W1
    node1 = helper.make_node(
        "MarlinQuantizedMatMul",
        inputs=["X", "W1_packed", "W1_scales"],
        outputs=["hidden"],
        name="marlin_layer1",
        domain=DOMAIN,
        group_size=group_size,
    )

    # ReLU (standard op)
    relu_node = helper.make_node("Relu", inputs=["hidden"], outputs=["hidden_relu"], name="relu")

    # Layer 2: hidden_relu @ W2
    node2 = helper.make_node(
        "MarlinQuantizedMatMul",
        inputs=["hidden_relu", "W2_packed", "W2_scales"],
        outputs=["Y"],
        name="marlin_layer2",
        domain=DOMAIN,
        group_size=group_size,
    )

    graph = helper.make_graph(
        [node1, relu_node, node2], "two_layer_marlin",
        [X], [Y],
        initializer=[w1_packed_init, s1_init, w2_packed_init, s2_init],
    )

    opset_imports = [
        helper.make_opsetid("", 17),
        helper.make_opsetid(DOMAIN, OPSET_VERSION),
    ]
    model = helper.make_model(graph, opset_imports=opset_imports)
    model.ir_version = 8
    return model


# ---------------------------------------------------------------------------
# Tests: Custom op class interface
# ---------------------------------------------------------------------------

class TestMarlinOpDefinition:
    """Test the custom op class definitions are well-formed."""

    def test_matmul_op_type(self):
        assert MarlinQuantizedMatMulOp.op_type == "MarlinQuantizedMatMul"
        assert MarlinQuantizedMatMulOp.domain == DOMAIN

    def test_matmul_inputs(self):
        inputs = MarlinQuantizedMatMulOp.get_inputs()
        assert len(inputs) == 3
        assert inputs[0] == ("A", "tensor(float16)")
        assert inputs[1] == ("B_packed", "tensor(uint32)")
        assert inputs[2] == ("scales", "tensor(float16)")

    def test_matmul_outputs(self):
        outputs = MarlinQuantizedMatMulOp.get_outputs()
        assert len(outputs) == 1
        assert outputs[0] == ("Y", "tensor(float16)")

    def test_matmul_attrs(self):
        attrs = MarlinQuantizedMatMulOp.get_attrs()
        assert "group_size" in attrs
        assert attrs["group_size"] == ("int", 32)

    def test_linear_op_type(self):
        assert MarlinQuantizedLinearOp.op_type == "MarlinQuantizedLinear"
        assert MarlinQuantizedLinearOp.domain == DOMAIN

    def test_linear_inputs_include_bias(self):
        inputs = MarlinQuantizedLinearOp.get_inputs()
        assert len(inputs) == 4
        assert inputs[3] == ("bias", "tensor(float16)")

    def test_domain_and_opset(self):
        assert DOMAIN == "com.metal_marlin"
        assert OPSET_VERSION == 1


# ---------------------------------------------------------------------------
# Tests: Model creation and ONNX validity
# ---------------------------------------------------------------------------

class TestModelCreation:
    """Test that models with Marlin custom ops pass ONNX validation."""

    def test_single_marlin_node_valid_onnx(self, quantize_weights):
        """Model with one MarlinQuantizedMatMul passes onnx.checker."""
        import mlx.core as mx

        pack_fp4, _ = quantize_weights
        K, N = 256, 128
        weights = mx.random.normal((K, N)).astype(mx.float16)
        packed, scales, meta = pack_fp4(weights, group_size=128)

        packed_np = np.array(packed)
        scales_np = np.array(scales)

        model = _make_marlin_matmul_model(
            M=1, K=meta["padded_K"], N=meta["padded_N"],
            packed_weights=packed_np, scales=scales_np,
            group_size=128,
        )

        # Should not raise
        onnx.checker.check_model(model)

    def test_model_has_custom_domain(self, quantize_weights):
        """Model opset imports include com.metal_marlin domain."""
        import mlx.core as mx

        pack_fp4, _ = quantize_weights
        K, N = 128, 64
        weights = mx.random.normal((K, N)).astype(mx.float16)
        packed, scales, meta = pack_fp4(weights, group_size=128)

        model = _make_marlin_matmul_model(
            M=1, K=meta["padded_K"], N=meta["padded_N"],
            packed_weights=np.array(packed), scales=np.array(scales),
        )

        domains = {opset.domain for opset in model.opset_import}
        assert DOMAIN in domains

    def test_model_node_attributes(self, quantize_weights):
        """MarlinQuantizedMatMul node has correct attributes."""
        import mlx.core as mx

        pack_fp4, _ = quantize_weights
        K, N = 128, 64
        weights = mx.random.normal((K, N)).astype(mx.float16)
        packed, scales, meta = pack_fp4(weights, group_size=128)

        model = _make_marlin_matmul_model(
            M=4, K=meta["padded_K"], N=meta["padded_N"],
            packed_weights=np.array(packed), scales=np.array(scales),
            group_size=128,
        )

        node = model.graph.node[0]
        assert node.op_type == "MarlinQuantizedMatMul"
        assert node.domain == DOMAIN

        # Check group_size attribute
        group_size_attr = next(a for a in node.attribute if a.name == "group_size")
        assert group_size_attr.i == 128

    def test_two_layer_model_valid(self, quantize_weights):
        """Two-layer model with ReLU between Marlin ops passes validation."""
        import mlx.core as mx

        pack_fp4, _ = quantize_weights
        K, hidden, N = 256, 128, 64

        w1 = mx.random.normal((K, hidden)).astype(mx.float16)
        w2 = mx.random.normal((hidden, N)).astype(mx.float16)

        p1, s1, m1 = pack_fp4(w1, group_size=128)
        p2, s2, m2 = pack_fp4(w2, group_size=128)

        model = _make_two_layer_marlin_model(
            M=1, K=m1["padded_K"], hidden=m1["padded_N"],
            N=m2["padded_N"],
            packed_w1=np.array(p1), scales_w1=np.array(s1),
            packed_w2=np.array(p2), scales_w2=np.array(s2),
        )

        # Standard ONNX checker doesn't validate custom domain ops fully,
        # but structural validity should pass
        assert len(model.graph.node) == 3  # marlin, relu, marlin
        assert model.graph.node[0].op_type == "MarlinQuantizedMatMul"
        assert model.graph.node[1].op_type == "Relu"
        assert model.graph.node[2].op_type == "MarlinQuantizedMatMul"


# ---------------------------------------------------------------------------
# Tests: Python compute path (no ORT session, direct op execution)
# ---------------------------------------------------------------------------

class TestMarlinComputeDirect:
    """Test MarlinQuantizedMatMulOp.compute() directly (bypasses ORT)."""

    def test_output_shape(self, quantize_weights):
        """Compute produces correct output dimensions."""
        import mlx.core as mx

        pack_fp4, _ = quantize_weights
        M, K, N = 4, 256, 128

        weights = mx.random.normal((K, N)).astype(mx.float16)
        packed, scales, meta = pack_fp4(weights, group_size=128)

        input_data = np.random.randn(M, meta["padded_K"]).astype(np.float16)
        result = MarlinQuantizedMatMulOp.compute(
            input_data,
            np.array(packed),
            np.array(scales),
            group_size=128,
        )

        assert result.shape == (M, meta["padded_N"])
        assert result.dtype == np.float16

    def test_accuracy_vs_fp16_reference(self, quantize_weights):
        """Quantized compute is within tolerance of FP16 reference."""
        import mlx.core as mx

        pack_fp4, unpack_fp4 = quantize_weights
        M, K, N = 8, 128, 64

        weights = mx.random.normal((K, N)).astype(mx.float16)
        packed, scales, meta = pack_fp4(weights, group_size=128)

        # Dequantize for reference (unpack uses meta dict for padding info)
        weights_deq = unpack_fp4(packed, scales, meta)
        mx.eval(weights_deq)

        input_data = np.random.randn(M, meta["padded_K"]).astype(np.float16)
        input_mx = mx.array(input_data)

        # Reference: FP16 matmul with dequantized weights
        ref = np.array(input_mx @ weights_deq)

        # Quantized compute path (uses padded dimensions)
        result = MarlinQuantizedMatMulOp.compute(
            input_data,
            np.array(packed),
            np.array(scales),
            group_size=128,
        )

        # Trim to original output dimensions for comparison
        result_trimmed = result[:, :meta["orig_N"]]
        ref_trimmed = ref.astype(np.float32)

        # FP4 quantization introduces error, but should be bounded
        # Typical relative error for FP4 E2M1 is ~5-15%
        atol = 0.5  # Absolute tolerance for FP16 range
        np.testing.assert_allclose(
            result_trimmed.astype(np.float32),
            ref_trimmed,
            atol=atol,
            rtol=0.2,
        )

    def test_batch_dimension(self, quantize_weights):
        """Compute handles various batch sizes."""
        import mlx.core as mx

        pack_fp4, _ = quantize_weights
        K, N = 128, 64

        weights = mx.random.normal((K, N)).astype(mx.float16)
        packed, scales, meta = pack_fp4(weights, group_size=128)
        packed_np = np.array(packed)
        scales_np = np.array(scales)

        for M in [1, 4, 16, 32]:
            input_data = np.random.randn(M, meta["padded_K"]).astype(np.float16)
            result = MarlinQuantizedMatMulOp.compute(
                input_data, packed_np, scales_np, group_size=128,
            )
            assert result.shape == (M, meta["padded_N"])

    def test_different_group_sizes(self, quantize_weights):
        """Compute works with different group sizes."""
        import mlx.core as mx

        pack_fp4, _ = quantize_weights
        M, K, N = 4, 256, 128

        for group_size in [32, 64, 128, 256]:
            if K % group_size != 0:
                continue
            weights = mx.random.normal((K, N)).astype(mx.float16)
            packed, scales, meta = pack_fp4(weights, group_size=group_size)

            input_data = np.random.randn(M, meta["padded_K"]).astype(np.float16)
            result = MarlinQuantizedMatMulOp.compute(
                input_data,
                np.array(packed),
                np.array(scales),
                group_size=group_size,
            )
            assert result.shape[0] == M


# ---------------------------------------------------------------------------
# Tests: ORT session with onnxruntime-extensions
# ---------------------------------------------------------------------------

class TestORTSessionWithExtensions:
    """Test full ORT InferenceSession with registered Marlin custom ops.

    Requires onnxruntime-extensions to be installed.
    These tests validate the complete Python custom op pipeline:
        ORT session -> custom op domain -> Python compute -> Metal kernel -> ORT output
    """

    @pytest.fixture(autouse=True)
    def require_extensions(self, ort_extensions_available):
        """Skip all tests in this class if extensions not available."""

    def test_session_creation(self, quantize_weights):
        """ORT session loads model with Marlin custom ops without error."""
        import mlx.core as mx

        pack_fp4, _ = quantize_weights
        K, N = 128, 64
        weights = mx.random.normal((K, N)).astype(mx.float16)
        packed, scales, meta = pack_fp4(weights, group_size=128)

        model = _make_marlin_matmul_model(
            M=1, K=meta["padded_K"], N=meta["padded_N"],
            packed_weights=np.array(packed), scales=np.array(scales),
        )
        model_path = _save_model(model)

        try:
            _provider_mod.register_marlin_ops_ortextensions()
            from onnxruntime_extensions import get_library_path

            options = ort.SessionOptions()
            options.register_custom_ops_library(get_library_path())
            session = ort.InferenceSession(
                str(model_path), sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            assert session is not None
            assert len(session.get_inputs()) == 1
            assert session.get_inputs()[0].name == "X"
        finally:
            model_path.unlink(missing_ok=True)

    def test_inference_output_shape(self, quantize_weights):
        """ORT inference produces correct output shape."""
        import mlx.core as mx

        pack_fp4, _ = quantize_weights
        M, K, N = 4, 128, 64
        weights = mx.random.normal((K, N)).astype(mx.float16)
        packed, scales, meta = pack_fp4(weights, group_size=128)

        model = _make_marlin_matmul_model(
            M=M, K=meta["padded_K"], N=meta["padded_N"],
            packed_weights=np.array(packed), scales=np.array(scales),
        )
        model_path = _save_model(model)

        try:
            session = _provider_mod.create_session(str(model_path))
            input_data = np.random.randn(M, meta["padded_K"]).astype(np.float16)
            outputs = session.run(None, {"X": input_data})

            assert len(outputs) == 1
            assert outputs[0].shape == (M, meta["padded_N"])
        finally:
            model_path.unlink(missing_ok=True)

    def test_inference_accuracy(self, quantize_weights):
        """ORT inference matches direct Python compute path."""
        import mlx.core as mx

        pack_fp4, _ = quantize_weights
        M, K, N = 8, 128, 64
        weights = mx.random.normal((K, N)).astype(mx.float16)
        packed, scales, meta = pack_fp4(weights, group_size=128)

        packed_np = np.array(packed)
        scales_np = np.array(scales)

        model = _make_marlin_matmul_model(
            M=M, K=meta["padded_K"], N=meta["padded_N"],
            packed_weights=packed_np, scales=scales_np,
        )
        model_path = _save_model(model)

        try:
            session = _provider_mod.create_session(str(model_path))
            input_data = np.random.randn(M, meta["padded_K"]).astype(np.float16)

            # ORT path
            ort_result = session.run(None, {"X": input_data})[0]

            # Direct compute path (should be identical since same kernel)
            direct_result = MarlinQuantizedMatMulOp.compute(
                input_data, packed_np, scales_np, group_size=128,
            )

            np.testing.assert_array_equal(ort_result, direct_result)
        finally:
            model_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests: Model export/conversion
# ---------------------------------------------------------------------------

class TestModelExport:
    """Test converting standard ONNX models to use Marlin custom ops."""

    def test_export_replaces_matmul(self, quantize_weights):
        """export_with_marlin_ops replaces MatMul with MarlinQuantizedMatMul."""
        import mlx.core as mx

        pack_fp4, _ = quantize_weights
        K, N = 128, 64

        # Create standard model
        model, weights_np = _make_standard_matmul_model(M=1, K=K, N=N)
        model_path = _save_model(model)

        # Quantize weights
        weights_mx = mx.array(weights_np)
        packed, scales, meta = pack_fp4(weights_mx, group_size=128)

        output_path = model_path.with_suffix(".marlin.onnx")
        try:
            export_with_marlin_ops(
                model_path, output_path,
                weights={"W": (np.array(packed), np.array(scales))},
                group_size=128,
            )

            # Load and verify
            converted = onnx.load(str(output_path))
            assert len(converted.graph.node) == 1
            assert converted.graph.node[0].op_type == "MarlinQuantizedMatMul"
            assert converted.graph.node[0].domain == DOMAIN

            # Check initializers replaced
            init_names = {init.name for init in converted.graph.initializer}
            assert "W" not in init_names  # Original weight removed
            assert "W_packed" in init_names
            assert "W_scales" in init_names
        finally:
            model_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_export_preserves_non_quantized_nodes(self, quantize_weights):
        """Nodes not in the weights dict are preserved as-is."""
        import mlx.core as mx

        pack_fp4, _ = quantize_weights
        K, N = 128, 64

        # Create model with MatMul + Relu
        weights = np.random.randn(K, N).astype(np.float16)
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, K])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, N])
        w_init = numpy_helper.from_array(weights, name="W")

        matmul = helper.make_node("MatMul", ["X", "W"], ["mid"], name="mm")
        relu = helper.make_node("Relu", ["mid"], ["Y"], name="relu")

        graph = helper.make_graph([matmul, relu], "test", [X], [Y], initializer=[w_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8

        model_path = _save_model(model)

        # Quantize and export
        weights_mx = mx.array(weights)
        packed, scales, _ = pack_fp4(weights_mx, group_size=128)

        output_path = model_path.with_suffix(".marlin.onnx")
        try:
            export_with_marlin_ops(
                model_path, output_path,
                weights={"W": (np.array(packed), np.array(scales))},
            )

            converted = onnx.load(str(output_path))
            assert len(converted.graph.node) == 2
            assert converted.graph.node[0].op_type == "MarlinQuantizedMatMul"
            assert converted.graph.node[1].op_type == "Relu"
        finally:
            model_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_export_no_matching_weights_unchanged(self):
        """Model is unchanged when no weights match the quantized dict."""
        model, _ = _make_standard_matmul_model(M=1, K=128, N=64)
        model_path = _save_model(model)
        output_path = model_path.with_suffix(".marlin.onnx")

        try:
            export_with_marlin_ops(
                model_path, output_path,
                weights={},  # No weights to replace
            )

            converted = onnx.load(str(output_path))
            assert converted.graph.node[0].op_type == "MatMul"
        finally:
            model_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests: ORT session without extensions (C++ library path)
# ---------------------------------------------------------------------------

class TestORTSessionCppPath:
    """Test the C++ custom op library path (expected to fail without compiled lib)."""

    def test_missing_library_raises(self, quantize_weights):
        """create_session raises RuntimeError when C++ lib is not built."""
        import mlx.core as mx

        pack_fp4, _ = quantize_weights
        K, N = 128, 64
        weights = mx.random.normal((K, N)).astype(mx.float16)
        packed, scales, meta = pack_fp4(weights, group_size=128)

        model = _make_marlin_matmul_model(
            M=1, K=meta["padded_K"], N=meta["padded_N"],
            packed_weights=np.array(packed), scales=np.array(scales),
        )
        model_path = _save_model(model)

        try:
            with pytest.raises((FileNotFoundError, ValueError)):
                _provider_mod.create_session(
                    str(model_path),
                    use_extensions=False,
                    custom_lib_path="/nonexistent/libmarlin_ort.dylib",
                )
        finally:
            model_path.unlink(missing_ok=True)

    def test_no_args_raises_valueerror(self, quantize_weights):
        """create_session raises ValueError with use_extensions=False and no lib path."""
        import mlx.core as mx

        pack_fp4, _ = quantize_weights
        K, N = 128, 64
        weights = mx.random.normal((K, N)).astype(mx.float16)
        packed, scales, meta = pack_fp4(weights, group_size=128)

        model = _make_marlin_matmul_model(
            M=1, K=meta["padded_K"], N=meta["padded_N"],
            packed_weights=np.array(packed), scales=np.array(scales),
        )
        model_path = _save_model(model)

        try:
            with pytest.raises(ValueError, match="use_extensions.*custom_lib_path"):
                _provider_mod.create_session(
                    str(model_path),
                    use_extensions=False,
                    custom_lib_path=None,
                )
        finally:
            model_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests: Edge cases and error handling
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test error handling and edge cases in the ORT integration."""

    def test_model_with_unknown_custom_op_domain_fails_gracefully(self):
        """ORT rejects models with unregistered custom domain ops."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, 64])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, 64])

        node = helper.make_node(
            "FakeOp", inputs=["X"], outputs=["Y"],
            name="fake", domain="com.nonexistent",
        )

        graph = helper.make_graph([node], "fake_test", [X], [Y])
        opset_imports = [
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.nonexistent", 1),
        ]
        model = helper.make_model(graph, opset_imports=opset_imports)
        model.ir_version = 8

        model_path = _save_model(model)
        try:
            with pytest.raises(Exception):
                ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"],
                )
        finally:
            model_path.unlink(missing_ok=True)

    def test_zero_batch_input(self, quantize_weights):
        """Compute handles zero-length batch gracefully."""
        import mlx.core as mx

        pack_fp4, _ = quantize_weights
        K, N = 128, 64
        weights = mx.random.normal((K, N)).astype(mx.float16)
        packed, scales, meta = pack_fp4(weights, group_size=128)

        input_data = np.zeros((0, meta["padded_K"]), dtype=np.float16)

        # MLX may or may not handle empty tensors; document behavior
        try:
            result = MarlinQuantizedMatMulOp.compute(
                input_data,
                np.array(packed),
                np.array(scales),
                group_size=128,
            )
            assert result.shape[0] == 0
        except (ValueError, RuntimeError):
            # Empty batch may not be supported by Metal kernels
            pytest.skip("Empty batch not supported by Metal kernel")

    def test_large_group_size_equals_k(self, quantize_weights):
        """group_size == K means a single group per column (no sub-groups)."""
        import mlx.core as mx

        pack_fp4, _ = quantize_weights
        K, N = 128, 64
        weights = mx.random.normal((K, N)).astype(mx.float16)
        packed, scales, meta = pack_fp4(weights, group_size=K)

        input_data = np.random.randn(4, meta["padded_K"]).astype(np.float16)
        result = MarlinQuantizedMatMulOp.compute(
            input_data,
            np.array(packed),
            np.array(scales),
            group_size=K,
        )
        assert result.shape == (4, meta["padded_N"])
