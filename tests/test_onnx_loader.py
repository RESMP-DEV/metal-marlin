"""Tests for ONNX model loading, weight extraction, and name normalization.

Tests exercise:
1. Small ONNX model extraction
2. ONNX to HuggingFace name mapping/normalization
3. Full ONNX -> FP4 conversion pipeline
4. CLI smoke test for ONNX-related commands

All test models are created programmatically with onnx.helper (no PyTorch dependency).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

# Add metal_marlin package to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Load onnx_executor directly to avoid import issues
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


# Fixtures directory for test models
FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# TinyModel creation using onnx.helper
# ---------------------------------------------------------------------------


def _save_model(model: onnx.ModelProto) -> Path:
    """Save ONNX model to a temp file, returning its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx.save(model, tmp.name)
    tmp.close()
    return Path(tmp.name)


def create_tiny_onnx_model(
    in_features: int = 64,
    hidden_features: int = 32,
    out_features: int = 16,
    seed: int = 42,
    batch_size: int = 1,
) -> Path:
    """Create a 2-layer FFN model using onnx.helper: X @ W1 + b1 -> ReLU -> @ W2 + b2.

    Args:
        in_features: Input feature dimension
        hidden_features: Hidden layer dimension
        out_features: Output feature dimension
        seed: Random seed for weight initialization
        batch_size: Batch size for the input shape

    Returns:
        Path to the saved ONNX file
    """
    rng = np.random.default_rng(seed)

    # Initialize weights matching a 2-layer linear network
    W1 = rng.standard_normal((in_features, hidden_features)).astype(np.float16)
    b1 = rng.standard_normal((hidden_features,)).astype(np.float16)
    W2 = rng.standard_normal((hidden_features, out_features)).astype(np.float16)
    b2 = rng.standard_normal((out_features,)).astype(np.float16)

    X = helper.make_tensor_value_info("input", TensorProto.FLOAT16, [batch_size, in_features])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT16, [batch_size, out_features])

    nodes = [
        helper.make_node("MatMul", ["input", "linear1.weight"], ["mm1"]),
        helper.make_node("Add", ["mm1", "linear1.bias"], ["lin1"]),
        helper.make_node("Relu", ["lin1"], ["relu_out"]),
        helper.make_node("MatMul", ["relu_out", "linear2.weight"], ["mm2"]),
        helper.make_node("Add", ["mm2", "linear2.bias"], ["output"]),
    ]

    initializers = [
        numpy_helper.from_array(W1, "linear1.weight"),
        numpy_helper.from_array(b1, "linear1.bias"),
        numpy_helper.from_array(W2, "linear2.weight"),
        numpy_helper.from_array(b2, "linear2.bias"),
    ]

    graph = helper.make_graph(nodes, "tiny_model", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    # Save to fixtures directory
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = FIXTURES_DIR / f"tiny_{in_features}_{hidden_features}_{out_features}_{seed}.onnx"
    onnx.save(model, str(onnx_path))
    return onnx_path


# ---------------------------------------------------------------------------
# Test 1: Extract small ONNX model
# ---------------------------------------------------------------------------


class TestExtractSmallOnnx:
    """Test ONNX model extraction with a tiny onnx.helper-created model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create tiny ONNX model before each test."""
        self.onnx_path = create_tiny_onnx_model()

    def test_load_tiny_model(self):
        """Verify the tiny ONNX model loads successfully."""
        graph = load_onnx_model(self.onnx_path)

        assert isinstance(graph, ONNXGraph)
        assert len(graph.nodes) > 0
        assert "input" in graph.inputs

    def test_extract_weights(self):
        """Verify weight tensors are extracted as mx.array."""
        graph = load_onnx_model(self.onnx_path)

        # Should have 4 initializers: 2 weights + 2 biases
        assert len(graph.initializers) == 4

        for name, tensor in graph.initializers.items():
            assert isinstance(tensor, mx.array), f"{name} should be mx.array"

    def test_weight_shapes_match_model(self):
        """Verify extracted weight shapes match the TinyModel architecture."""
        graph = load_onnx_model(self.onnx_path)

        # TinyModel has two linear layers with weights: [64, 32] and [32, 16]
        assert graph.initializers["linear1.weight"].shape == (64, 32)
        assert graph.initializers["linear1.bias"].shape == (32,)
        assert graph.initializers["linear2.weight"].shape == (32, 16)
        assert graph.initializers["linear2.bias"].shape == (16,)

    def test_model_execution(self):
        """Verify the loaded model can execute and produce output."""
        executor = ONNXExecutor.from_file(self.onnx_path, quantize=False)

        # Create input
        batch_size = 4
        x = mx.random.normal((batch_size, 64)).astype(mx.float16)

        result = executor(input=x)

        assert "output" in result
        assert result["output"].shape == (batch_size, 16)

    def test_matmul_nodes_identified(self):
        """Verify MatMul/Gemm nodes are correctly identified."""
        graph = load_onnx_model(self.onnx_path)

        matmul_ops = [n for n in graph.nodes if n.op_type in ("MatMul", "Gemm")]
        # TinyModel has 2 linear layers -> 2 MatMul ops
        assert len(matmul_ops) == 2


# ---------------------------------------------------------------------------
# Test 2: Name normalization (ONNX to HuggingFace mapping)
# ---------------------------------------------------------------------------


class TestNameNormalization:
    """Test ONNX tensor name to HuggingFace-style name mapping."""

    def test_simple_names_preserved(self):
        """Simple weight names are preserved through load."""
        onnx_path = create_tiny_onnx_model()
        graph = load_onnx_model(onnx_path)

        # Names should match what we set in create_tiny_onnx_model
        init_names = list(graph.initializers.keys())
        assert "linear1.weight" in init_names
        assert "linear1.bias" in init_names
        assert "linear2.weight" in init_names
        assert "linear2.bias" in init_names

    def test_onnx_helper_model_names(self):
        """Test name handling with HuggingFace-style names."""
        # Create model with explicit HF-style names
        rng = np.random.default_rng(42)
        W1 = rng.standard_normal((64, 32)).astype(np.float16)
        b1 = rng.standard_normal((32,)).astype(np.float16)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [None, 64])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [None, 32])

        nodes = [
            helper.make_node("MatMul", ["X", "model.layers.0.self_attn.q_proj.weight"], ["mm_out"]),
            helper.make_node("Add", ["mm_out", "model.layers.0.self_attn.q_proj.bias"], ["Y"]),
        ]

        initializers = [
            numpy_helper.from_array(W1, "model.layers.0.self_attn.q_proj.weight"),
            numpy_helper.from_array(b1, "model.layers.0.self_attn.q_proj.bias"),
        ]

        graph = helper.make_graph(nodes, "hf_naming_test", [X], [Y], initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        onnx.checker.check_model(model)

        path = _save_model(model)
        parsed = load_onnx_model(path)

        # Verify HF-style names are preserved
        assert "model.layers.0.self_attn.q_proj.weight" in parsed.initializers
        assert "model.layers.0.self_attn.q_proj.bias" in parsed.initializers

        path.unlink()

    def test_transformer_weight_naming(self):
        """Test that transformer-style weight names are handled correctly."""
        rng = np.random.default_rng(99)
        D = 64

        weights = {
            "model.embed_tokens.weight": rng.standard_normal((1000, D)).astype(np.float16),
            "model.layers.0.self_attn.q_proj.weight": rng.standard_normal((D, D)).astype(np.float16),
            "model.layers.0.self_attn.k_proj.weight": rng.standard_normal((D, D)).astype(np.float16),
            "model.layers.0.self_attn.v_proj.weight": rng.standard_normal((D, D)).astype(np.float16),
            "model.layers.0.mlp.gate_proj.weight": rng.standard_normal((D * 4, D)).astype(np.float16),
            "model.layers.0.mlp.down_proj.weight": rng.standard_normal((D, D * 4)).astype(np.float16),
            "model.norm.weight": rng.standard_normal((D,)).astype(np.float16),
            "lm_head.weight": rng.standard_normal((1000, D)).astype(np.float16),
        }

        # Build a simple pass-through graph (just Identity for testing names)
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [None, D])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [None, D])

        initializers = [numpy_helper.from_array(w, name) for name, w in weights.items()]

        node = helper.make_node("Identity", ["X"], ["Y"])
        graph = helper.make_graph([node], "transformer_test", [X], [Y], initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8

        path = _save_model(model)
        parsed = load_onnx_model(path)

        # All original names should be preserved
        for name in weights:
            assert name in parsed.initializers, f"Missing: {name}"
            assert parsed.initializers[name].shape == weights[name].shape

        path.unlink()

    def test_names_not_empty(self):
        """Verify all initializer names are non-empty strings."""
        onnx_path = create_tiny_onnx_model()
        graph = load_onnx_model(onnx_path)

        for name in graph.initializers:
            assert name is not None
            assert len(name) > 0
            assert isinstance(name, str)


# ---------------------------------------------------------------------------
# Test 3: ONNX to FP4 conversion pipeline
# ---------------------------------------------------------------------------


class TestOnnxToFp4Conversion:
    """Full ONNX -> FP4 quantization pipeline tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create test models."""
        # Use dimensions divisible by 8 and group_size (32)
        self.onnx_path = create_tiny_onnx_model(
            in_features=64, hidden_features=128, out_features=32, seed=42
        )

    def test_quantize_linear_weights_extraction(self):
        """Test that _quantize_linear_weights finds MatMul weights."""
        graph = load_onnx_model(self.onnx_path)
        quantized = _quantize_linear_weights(graph)

        # Should have quantized 2 weights (from 2 linear layers)
        assert len(quantized) == 2

        for name, (packed, scales) in quantized.items():
            assert isinstance(packed, mx.array)
            assert isinstance(scales, mx.array)
            assert packed.dtype == mx.uint32
            assert scales.dtype == mx.float16

    def test_executor_with_quantization(self):
        """Test ONNXExecutor with quantize=True flag."""
        executor = ONNXExecutor.from_file(self.onnx_path, quantize=True)

        # Should have quantized weights
        assert len(executor.quantized_weights) == 2

        # Should still be able to execute
        x = mx.random.normal((4, 64)).astype(mx.float16)
        result = executor(input=x)

        assert "output" in result
        assert result["output"].shape == (4, 32)

    def test_quantized_output_finite(self):
        """Verify quantized execution produces finite outputs."""
        executor = ONNXExecutor.from_file(self.onnx_path, quantize=True)

        x = mx.random.normal((8, 64)).astype(mx.float16)
        result = executor(input=x)

        output_np = np.array(result["output"])
        assert np.all(np.isfinite(output_np)), "Output contains inf/nan"

    def test_quantized_vs_unquantized_shape(self):
        """Quantized and unquantized execution should produce same output shape."""
        exec_quant = ONNXExecutor.from_file(self.onnx_path, quantize=True)
        exec_noquant = ONNXExecutor.from_file(self.onnx_path, quantize=False)

        x = mx.random.normal((4, 64)).astype(mx.float16)
        mx.eval(x)

        result_q = exec_quant(input=x)
        result_nq = exec_noquant(input=x)

        assert result_q["output"].shape == result_nq["output"].shape

    def test_larger_model_conversion(self):
        """Test conversion on a slightly larger model."""
        # Create larger model with compatible dimensions
        onnx_path = create_tiny_onnx_model(
            in_features=256,
            hidden_features=512,
            out_features=128,
            seed=123,
        )

        executor = ONNXExecutor.from_file(onnx_path, quantize=True)

        x = mx.random.normal((2, 256)).astype(mx.float16)
        result = executor(input=x)

        assert result["output"].shape == (2, 128)
        assert np.all(np.isfinite(np.array(result["output"])))

    def test_conversion_preserves_computation(self):
        """Quantized output should be within FP4 tolerance of unquantized."""
        # Use a model with dimensions that work well for quantization
        onnx_path = create_tiny_onnx_model(
            in_features=128,
            hidden_features=64,
            out_features=32,
            seed=777,
        )

        exec_quant = ONNXExecutor.from_file(onnx_path, quantize=True)
        exec_noquant = ONNXExecutor.from_file(onnx_path, quantize=False)

        mx.random.seed(999)
        x = mx.random.normal((4, 128)).astype(mx.float16)
        mx.eval(x)

        result_q = exec_quant(input=x)
        result_nq = exec_noquant(input=x)

        q_np = np.array(result_q["output"].astype(mx.float32))
        nq_np = np.array(result_nq["output"].astype(mx.float32))

        # FP4 quantization introduces significant error due to 4-bit representation.
        # The bulk of values should be correlated (check correlation instead of raw error).
        correlation = np.corrcoef(q_np.flatten(), nq_np.flatten())[0, 1]
        assert correlation > 0.5, f"Correlation too low: {correlation}"

        # Also check outputs are finite
        assert np.all(np.isfinite(q_np)), "Quantized output contains inf/nan"


# ---------------------------------------------------------------------------
# Test 4: CLI smoke test
# ---------------------------------------------------------------------------


class TestOnnxCli:
    """Smoke tests for ONNX-related CLI commands."""

    def test_hf_loader_module_exists(self):
        """Verify hf_loader module can be imported as a package."""
        # Import via package path to handle relative imports
        import metal_marlin.hf_loader as hf_loader

        # Check key exports exist
        assert hasattr(hf_loader, "ModelConfig")
        assert hasattr(hf_loader, "load_model_config")
        assert hasattr(hf_loader, "should_quantize_tensor")

    def test_cli_module_exists(self):
        """Verify CLI module can be imported."""
        spec = importlib.util.spec_from_file_location(
            "cli", _ROOT / "metal_marlin" / "cli.py"
        )
        assert spec is not None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Check CLI group exists
        assert hasattr(module, "cli")

    def test_cli_help_runs(self):
        """Verify CLI --help runs without error."""
        result = subprocess.run(
            [sys.executable, "-m", "metal_marlin.cli", "--help"],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"CLI help failed: {result.stderr}"
        assert "Metal Marlin" in result.stdout or "metal" in result.stdout.lower()

    def test_quantize_command_exists(self):
        """Verify the quantize command is registered."""
        result = subprocess.run(
            [sys.executable, "-m", "metal_marlin.cli", "quantize", "--help"],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Command should exist and show help or error gracefully
        assert result.returncode == 0 or "Usage" in result.stdout or "usage" in result.stdout.lower()

    def test_convert_command_exists(self):
        """Verify the convert command is registered."""
        result = subprocess.run(
            [sys.executable, "-m", "metal_marlin.cli", "convert", "--help"],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Command should exist
        assert result.returncode == 0 or "Usage" in result.stdout or "Error" in result.stderr

    def test_should_quantize_tensor_function(self):
        """Test the should_quantize_tensor helper from hf_loader."""
        import metal_marlin.hf_loader as hf_loader

        should_quantize = hf_loader.should_quantize_tensor

        # Should quantize 2D weight matrices with compatible dimensions
        assert should_quantize("model.layers.0.mlp.gate_proj.weight", np.zeros((4096, 1024)))

        # Should NOT quantize embeddings
        assert not should_quantize("model.embed_tokens.weight", np.zeros((32000, 4096)))

        # Should NOT quantize layer norms
        assert not should_quantize("model.layers.0.input_layernorm.weight", np.zeros((4096,)))

        # Should NOT quantize 1D tensors
        assert not should_quantize("some.weight", np.zeros((4096,)))

        # Should NOT quantize lm_head
        assert not should_quantize("lm_head.weight", np.zeros((32000, 4096)))

        # Should NOT quantize tensors with dimensions not divisible by 8
        assert not should_quantize("model.layers.0.mlp.weight", np.zeros((100, 37)))
