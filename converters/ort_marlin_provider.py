"""ONNX Runtime custom op provider for Metal Marlin quantized GEMM.

This module registers quantized MatMul operations with ONNX Runtime so that
models exported with the ``com.metal_marlin`` custom domain can dispatch
directly to Metal Marlin FP4 kernels without the intermediate ONNXExecutor.

Integration approaches:
    1. **onnxruntime-extensions (Python path)**: Register a Python-based custom
       op via ``ort_custom_ops.PyCustomOpDef``. Lower throughput due to Python
       GIL and data copy overhead, but works without compiling a shared library.

    2. **C++ OrtCustomOp (production path)**: Compile a shared library
       implementing ``OrtCustomOp`` that calls Metal Marlin kernels directly.
       The library is loaded via ``SessionOptions.register_custom_ops_library``.
       This avoids Python entirely in the inference path.

    3. **ONNX Runtime Execution Provider (full integration)**: Implement an
       ``IExecutionProvider`` that claims quantized MatMul nodes during graph
       partitioning. Most complex but enables graph-level optimization (fusion,
       memory planning). Requires building ONNX Runtime from source.

This file implements approach 1 (Python skeleton) and documents the C++ interface
needed for approach 2.

Requirements:
    pip install onnxruntime onnxruntime-extensions

Usage:
    from converters.ort_marlin_provider import create_session

    sess = create_session("model.onnx", group_size=32)
    outputs = sess.run(None, {"input": input_array})

C++ shared library interface (approach 2):
    The compiled .dylib/.so must export ``RegisterCustomOps`` with signature:
        OrtStatus* RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api)

    The custom op domain is ``com.metal_marlin`` with ops:
        - MarlinQuantizedMatMul (inputs: A, B_packed, scales; attrs: group_size)
        - MarlinQuantizedLinear (inputs: X, W_packed, scales, bias; attrs: group_size)
        - MarlinFlashAttention (inputs: Q, K, V; attrs: scale, causal, num_kv_heads)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Custom op domain for all Metal Marlin operations
DOMAIN = "com.metal_marlin"
OPSET_VERSION = 1


class MarlinQuantizedMatMulOp:
    """Custom op definition for quantized MatMul using Metal Marlin kernels.

    ONNX op signature:
        domain: com.metal_marlin
        op_type: MarlinQuantizedMatMul
        inputs:
            A: tensor(float16)        - Activations [M, K]
            B_packed: tensor(uint32)  - Packed FP4 weights [K/8, N]
            scales: tensor(float16)   - Per-group scales [K/group_size, N]
        outputs:
            Y: tensor(float16)        - Result [M, N]
        attributes:
            group_size: int (default 32)
    """

    op_type = "MarlinQuantizedMatMul"
    domain = DOMAIN

    @staticmethod
    def get_inputs() -> list[tuple[str, str]]:
        """Return input tensor specifications."""
        return [
            ("A", "tensor(float16)"),
            ("B_packed", "tensor(uint32)"),
            ("scales", "tensor(float16)"),
        ]

    @staticmethod
    def get_outputs() -> list[tuple[str, str]]:
        """Return output tensor specifications."""
        return [("Y", "tensor(float16)")]

    @staticmethod
    def get_attrs() -> dict[str, tuple[str, Any]]:
        """Return attribute specifications: name -> (type, default)."""
        return {"group_size": ("int", 32)}

    @staticmethod
    def compute(
        A: NDArray[np.float16],
        B_packed: NDArray[np.uint32],
        scales: NDArray[np.float16],
        *,
        group_size: int = 32,
    ) -> NDArray[np.float16]:
        """Execute quantized MatMul via Metal Marlin.

        This crosses the Python/Metal boundary. For production use, the C++
        custom op path eliminates this overhead entirely.
        """
        import mlx.core as mx

        # Convert numpy -> MLX arrays (zero-copy when possible on Metal)
        a_mx = mx.array(A)
        b_mx = mx.array(B_packed)
        s_mx = mx.array(scales)

        # Dispatch to Metal Marlin kernel
        try:
            from ..metal_marlin import quantized_linear
        except ImportError:
            from metal_marlin import quantized_linear

        result = quantized_linear(a_mx, b_mx, s_mx, group_size)

        # Force evaluation and convert back to numpy
        mx.eval(result)
        return np.array(result, dtype=np.float16)


class MarlinQuantizedLinearOp:
    """Custom op for quantized linear (MatMul + optional bias).

    ONNX op signature:
        domain: com.metal_marlin
        op_type: MarlinQuantizedLinear
        inputs:
            X: tensor(float16)        - Input activations [batch, seq, hidden]
            W_packed: tensor(uint32)  - Packed FP4 weights
            scales: tensor(float16)   - Per-group scales
            bias: tensor(float16)     - Optional bias (can be empty tensor)
        outputs:
            Y: tensor(float16)        - Result
        attributes:
            group_size: int (default 32)
    """

    op_type = "MarlinQuantizedLinear"
    domain = DOMAIN

    @staticmethod
    def get_inputs() -> list[tuple[str, str]]:
        return [
            ("X", "tensor(float16)"),
            ("W_packed", "tensor(uint32)"),
            ("scales", "tensor(float16)"),
            ("bias", "tensor(float16)"),
        ]

    @staticmethod
    def get_outputs() -> list[tuple[str, str]]:
        return [("Y", "tensor(float16)")]

    @staticmethod
    def compute(
        X: NDArray[np.float16],
        W_packed: NDArray[np.uint32],
        scales: NDArray[np.float16],
        bias: NDArray[np.float16] | None = None,
        *,
        group_size: int = 32,
    ) -> NDArray[np.float16]:
        """Execute quantized linear via Metal Marlin."""
        import mlx.core as mx

        x_mx = mx.array(X)
        w_mx = mx.array(W_packed)
        s_mx = mx.array(scales)

        try:
            from ..metal_marlin import quantized_linear
        except ImportError:
            from metal_marlin import quantized_linear

        result = quantized_linear(x_mx, w_mx, s_mx, group_size)

        if bias is not None and bias.size > 0:
            result = result + mx.array(bias)

        mx.eval(result)
        return np.array(result, dtype=np.float16)


class MarlinFlashAttentionOp:
    """Custom op for flash attention via Metal Marlin kernels.

    ONNX op signature:
        domain: com.metal_marlin
        op_type: MarlinFlashAttention
        inputs:
            Q: tensor(float16)  - [batch, num_heads, seq_q, head_dim]
            K: tensor(float16)  - [batch, num_kv_heads, seq_k, head_dim]
            V: tensor(float16)  - [batch, num_kv_heads, seq_k, head_dim]
        outputs:
            Y: tensor(float16)  - [batch, num_heads, seq_q, head_dim]
        attributes:
            scale: float (default -1.0, meaning auto = head_dim^-0.5)
            causal: int (default 1)
            num_kv_heads: int (default 0, meaning same as num_heads)
    """

    op_type = "MarlinFlashAttention"
    domain = DOMAIN

    @staticmethod
    def get_inputs() -> list[tuple[str, str]]:
        return [
            ("Q", "tensor(float16)"),
            ("K", "tensor(float16)"),
            ("V", "tensor(float16)"),
        ]

    @staticmethod
    def get_outputs() -> list[tuple[str, str]]:
        return [("Y", "tensor(float16)")]

    @staticmethod
    def compute(
        Q: NDArray[np.float16],
        K: NDArray[np.float16],
        V: NDArray[np.float16],
        *,
        scale: float = -1.0,
        causal: int = 1,
        num_kv_heads: int = 0,
    ) -> NDArray[np.float16]:
        """Execute flash attention via Metal Marlin."""
        import mlx.core as mx

        q_mx = mx.array(Q)
        k_mx = mx.array(K)
        v_mx = mx.array(V)

        if scale < 0:
            head_dim = Q.shape[-1]
            scale = float(head_dim**-0.5)

        try:
            from ..metal_marlin.metal_marlin import flash_attention_metal
        except ImportError:
            from metal_marlin.metal_marlin import flash_attention_metal

        kwargs: dict[str, Any] = {"scale": scale, "causal": bool(causal)}
        if num_kv_heads > 0:
            kwargs["num_kv_heads"] = num_kv_heads

        result = flash_attention_metal(q_mx, k_mx, v_mx, **kwargs)
        mx.eval(result)
        return np.array(result, dtype=np.float16)


# All ops in the custom domain
ALL_OPS: list[type] = [
    MarlinQuantizedMatMulOp,
    MarlinQuantizedLinearOp,
    MarlinFlashAttentionOp,
]


def register_marlin_ops_ortextensions() -> None:
    """Register Metal Marlin ops using onnxruntime-extensions Python API.

    This uses the PyCustomOpDef interface from onnxruntime-extensions to
    register Python-callable custom ops. The ops run in-process but cross
    the Python/Metal boundary on each call.

    Raises:
        ImportError: If onnxruntime-extensions is not installed.
    """
    from onnxruntime_extensions import PyCustomOpDef, onnx_op

    @onnx_op(
        op_type="MarlinQuantizedMatMul",
        inputs=[PyCustomOpDef.dt_float16, PyCustomOpDef.dt_uint32, PyCustomOpDef.dt_float16],
        outputs=[PyCustomOpDef.dt_float16],
        attrs={"group_size": PyCustomOpDef.dt_int64},
    )
    def marlin_quantized_matmul(A, B_packed, scales, **kwargs):
        group_size = int(kwargs.get("group_size", 32))
        return MarlinQuantizedMatMulOp.compute(A, B_packed, scales, group_size=group_size)

    @onnx_op(
        op_type="MarlinQuantizedLinear",
        inputs=[
            PyCustomOpDef.dt_float16,
            PyCustomOpDef.dt_uint32,
            PyCustomOpDef.dt_float16,
            PyCustomOpDef.dt_float16,
        ],
        outputs=[PyCustomOpDef.dt_float16],
        attrs={"group_size": PyCustomOpDef.dt_int64},
    )
    def marlin_quantized_linear(X, W_packed, scales, bias, **kwargs):
        group_size = int(kwargs.get("group_size", 32))
        bias_arr = bias if bias.size > 0 else None
        return MarlinQuantizedLinearOp.compute(X, W_packed, scales, bias_arr, group_size=group_size)

    @onnx_op(
        op_type="MarlinFlashAttention",
        inputs=[PyCustomOpDef.dt_float16, PyCustomOpDef.dt_float16, PyCustomOpDef.dt_float16],
        outputs=[PyCustomOpDef.dt_float16],
        attrs={
            "scale": PyCustomOpDef.dt_float,
            "causal": PyCustomOpDef.dt_int64,
            "num_kv_heads": PyCustomOpDef.dt_int64,
        },
    )
    def marlin_flash_attention(Q, K, V, **kwargs):
        scale = float(kwargs.get("scale", -1.0))
        causal = int(kwargs.get("causal", 1))
        num_kv_heads = int(kwargs.get("num_kv_heads", 0))
        return MarlinFlashAttentionOp.compute(Q, K, V, scale=scale, causal=causal, num_kv_heads=num_kv_heads)


def create_session(
    model_path: str | Path,
    *,
    group_size: int = 32,
    use_extensions: bool = True,
    custom_lib_path: str | Path | None = None,
) -> Any:
    """Create an ONNX Runtime InferenceSession with Metal Marlin ops registered.

    Args:
        model_path: Path to .onnx file using com.metal_marlin domain ops.
        group_size: Quantization group size (must match model weights).
        use_extensions: If True, use onnxruntime-extensions Python path.
            If False, requires custom_lib_path to a compiled .dylib/.so.
        custom_lib_path: Path to compiled custom op shared library.
            Required when use_extensions=False.

    Returns:
        onnxruntime.InferenceSession configured with Metal Marlin ops.

    Raises:
        ImportError: If required packages are not installed.
        FileNotFoundError: If custom_lib_path does not exist.
    """
    import onnxruntime as ort

    session_options = ort.SessionOptions()

    if use_extensions:
        # Register Python custom ops via onnxruntime-extensions
        register_marlin_ops_ortextensions()

        from onnxruntime_extensions import get_library_path

        session_options.register_custom_ops_library(get_library_path())
    elif custom_lib_path is not None:
        # Load compiled C++ custom op library
        lib_path = Path(custom_lib_path)
        if not lib_path.exists():
            raise FileNotFoundError(f"Custom op library not found: {lib_path}")
        session_options.register_custom_ops_library(str(lib_path))
    else:
        raise ValueError("Either use_extensions=True or provide custom_lib_path")

    # Prefer CoreML EP on macOS for non-custom ops, fall back to CPU
    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    return ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=providers,
    )


def export_with_marlin_ops(
    onnx_model_path: str | Path,
    output_path: str | Path,
    weights: dict[str, tuple[NDArray[np.uint32], NDArray[np.float16]]],
    *,
    group_size: int = 32,
) -> None:
    """Rewrite an ONNX model to use com.metal_marlin custom ops.

    Takes a standard ONNX model and replaces MatMul nodes (whose weights
    appear in the ``weights`` dict) with MarlinQuantizedMatMul custom ops.

    Args:
        onnx_model_path: Path to input .onnx model with FP16 weights.
        output_path: Where to save the rewritten model.
        weights: Dict mapping weight initializer names to
            (packed_uint32, scales_fp16) tuples from pack_fp4_weights.
        group_size: Quantization group size.
    """
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(onnx_model_path))
    graph = model.graph

    # Add custom op domain to model
    opset = helper.make_opsetid(DOMAIN, OPSET_VERSION)
    model.opset_import.append(opset)

    # Build set of weight names to replace
    quantized_weight_names = set(weights.keys())

    new_nodes: list[Any] = []
    new_initializers: list[Any] = []

    for node in graph.node:
        if node.op_type in ("MatMul", "Gemm") and len(node.input) > 1:
            weight_name = node.input[1]
            if weight_name in quantized_weight_names:
                packed, scales = weights[weight_name]

                # Create initializer tensors for packed weights and scales
                packed_name = f"{weight_name}_packed"
                scales_name = f"{weight_name}_scales"

                packed_tensor = helper.make_tensor(
                    packed_name,
                    TensorProto.UINT32,
                    list(packed.shape),
                    packed.tobytes(),
                    raw=True,
                )
                scales_tensor = helper.make_tensor(
                    scales_name,
                    TensorProto.FLOAT16,
                    list(scales.shape),
                    scales.tobytes(),
                    raw=True,
                )

                new_initializers.extend([packed_tensor, scales_tensor])

                # Replace MatMul node with MarlinQuantizedMatMul
                custom_node = helper.make_node(
                    "MarlinQuantizedMatMul",
                    inputs=[node.input[0], packed_name, scales_name],
                    outputs=list(node.output),
                    name=f"{node.name}_marlin",
                    domain=DOMAIN,
                    group_size=group_size,
                )
                new_nodes.append(custom_node)

                # Remove original weight initializer
                continue

        new_nodes.append(node)

    # Replace nodes
    del graph.node[:]
    graph.node.extend(new_nodes)

    # Remove replaced weight initializers, add packed versions
    remaining_inits = [
        init for init in graph.initializer if init.name not in quantized_weight_names
    ]
    del graph.initializer[:]
    graph.initializer.extend(remaining_inits)
    graph.initializer.extend(new_initializers)

    onnx.save(model, str(output_path))
