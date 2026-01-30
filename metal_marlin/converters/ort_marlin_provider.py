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
    pip install onnxruntime onnxruntime-extensions torch

Usage:
    from metal_marlin.converters.ort_marlin_provider import create_session

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

# Check PyTorch MPS availability
try:
    import torch

    HAS_TORCH = True
    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False
    torch = None


def _to_mps_tensor(arr: NDArray[Any], dtype: torch.dtype | None = None) -> torch.Tensor:
    """Convert numpy array to PyTorch MPS tensor.

    Args:
        arr: Input numpy array.
        dtype: Optional torch dtype. If None, infers from numpy dtype.

    Returns:
        PyTorch tensor on MPS device.
    """
    if not HAS_MPS or torch is None:
        raise RuntimeError("PyTorch MPS is required for Metal kernel dispatch")

    # Map numpy dtypes to torch dtypes
    dtype_map = {
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.uint32: torch.int32,  # MPS doesn't have native uint32
        np.int32: torch.int32,
    }

    if dtype is None:
        np_dtype = arr.dtype.type
        dtype = dtype_map.get(np_dtype, torch.float32)

    # Convert to tensor (ensure contiguous)
    tensor = torch.from_numpy(np.ascontiguousarray(arr))

    # Handle uint32 -> int32 view for MPS compatibility
    if arr.dtype == np.uint32:
        tensor = tensor.view(torch.int32)

    return tensor.to(device="mps", dtype=dtype if dtype != torch.int32 else None)


def _from_mps_tensor(tensor: torch.Tensor, dtype: np.dtype = np.float16) -> NDArray[Any]:
    """Convert PyTorch MPS tensor to numpy array.

    Args:
        tensor: PyTorch tensor (must be on MPS device).
        dtype: Target numpy dtype.

    Returns:
        numpy array with specified dtype.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required")

    return tensor.cpu().numpy().astype(dtype)


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
        if not HAS_MPS:
            raise RuntimeError(
                "PyTorch MPS is required for Metal kernel dispatch. "
                "Ensure you're on Apple Silicon with PyTorch >= 2.0"
            )

        # Import Metal Marlin kernel functions
        try:
            from ..metal_marlin.kernels import marlin_gemm_fp4
        except ImportError:
            from metal_marlin.kernels import marlin_gemm_fp4

        # Convert numpy -> PyTorch MPS tensors
        a_mps = _to_mps_tensor(A, torch.float16)
        b_mps = _to_mps_tensor(B_packed)  # uint32 as int32 view
        s_mps = _to_mps_tensor(scales, torch.float16)

        # Dispatch to Metal Marlin kernel
        result = marlin_gemm_fp4(a_mps, b_mps, s_mps, group_size)

        # Synchronize and convert back to numpy
        torch.mps.synchronize()
        return _from_mps_tensor(result, np.float16)


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
        if not HAS_MPS:
            raise RuntimeError(
                "PyTorch MPS is required for Metal kernel dispatch. "
                "Ensure you're on Apple Silicon with PyTorch >= 2.0"
            )

        try:
            from ..metal_marlin.kernels import marlin_gemm_fp4
        except ImportError:
            from metal_marlin.kernels import marlin_gemm_fp4

        x_mps = _to_mps_tensor(X, torch.float16)
        w_mps = _to_mps_tensor(W_packed)
        s_mps = _to_mps_tensor(scales, torch.float16)

        result = marlin_gemm_fp4(x_mps, w_mps, s_mps, group_size)

        if bias is not None and bias.size > 0:
            bias_mps = _to_mps_tensor(bias, torch.float16)
            result = result + bias_mps

        torch.mps.synchronize()
        return _from_mps_tensor(result, np.float16)


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
        """Execute flash attention via Metal Marlin.

        Uses PyTorch's scaled_dot_product_attention as the Metal flash attention
        kernel is still using MLX internally. For full Metal performance, the
        flash_attention_v2.metal kernel needs to be integrated with PyTorch MPS.
        """
        if not HAS_MPS:
            raise RuntimeError(
                "PyTorch MPS is required for Metal kernel dispatch. "
                "Ensure you're on Apple Silicon with PyTorch >= 2.0"
            )

        # Convert to MPS tensors
        q_mps = _to_mps_tensor(Q, torch.float16)
        k_mps = _to_mps_tensor(K, torch.float16)
        v_mps = _to_mps_tensor(V, torch.float16)

        if scale < 0:
            head_dim = Q.shape[-1]
            scale = float(head_dim**-0.5)

        # Handle GQA by repeating KV heads
        if num_kv_heads > 0 and num_kv_heads != q_mps.shape[1]:
            num_heads_q = q_mps.shape[1]
            repeat_factor = num_heads_q // num_kv_heads
            k_mps = k_mps.repeat_interleave(repeat_factor, dim=1)
            v_mps = v_mps.repeat_interleave(repeat_factor, dim=1)

        # Use PyTorch's scaled dot product attention
        # This will use Metal Performance Shaders under the hood on MPS
        result = torch.nn.functional.scaled_dot_product_attention(
            q_mps,
            k_mps,
            v_mps,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=bool(causal),
            scale=scale,
        )

        torch.mps.synchronize()
        return _from_mps_tensor(result, np.float16)


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
        return MarlinFlashAttentionOp.compute(
            Q, K, V, scale=scale, causal=causal, num_kv_heads=num_kv_heads
        )


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
