"""Test MoE kernel functionality."""

import torch

from metal_marlin.kernels import moe_expert_gemm_fp4
from metal_marlin.quantize_fp4 import quantize_fp4


def test_moe_kernel_basic():
    """Basic test of MoE expert GEMM kernel."""
    batch = 4
    hidden = 128
    intermediate = 64
    num_experts = 8
    top_k = 2
    group_size = 32

    # Create expert weights and quantize
    expert_weights_fp16 = torch.randn(num_experts, hidden, intermediate, dtype=torch.float16)

    packed_list = []
    scales_list = []
    for e in range(num_experts):
        # quantize_fp4 expects [out_features, in_features]
        # our weights are [hidden, intermediate] = [in, out]
        # transpose to [intermediate, hidden] = [out, in] before quantizing
        weights_e = expert_weights_fp16[e].T.numpy()  # [intermediate, hidden]
        packed_e, scales_e = quantize_fp4(weights_e, group_size=group_size, marlin_layout=True)
        packed_list.append(torch.from_numpy(packed_e))
        scales_list.append(torch.from_numpy(scales_e))

    # Stack: [num_experts, hidden/8, intermediate]
    expert_weights_packed = torch.stack(packed_list).to("mps")
    scales = torch.stack(scales_list).to("mps")

    # Create input tensors
    activations = torch.randn(batch, hidden, dtype=torch.float16, device="mps")
    expert_ids = torch.randint(0, num_experts, (batch, top_k), device="mps")
    expert_probs = torch.softmax(torch.randn(batch, top_k, device="mps"), dim=-1).half()

    print(f"Activations: {activations.shape}")
    print(f"Expert weights: {expert_weights_packed.shape}")
    print(f"Scales: {scales.shape}")
    print(f"Expert IDs: {expert_ids.shape}")
    print(f"Expert probs: {expert_probs.shape}")

    # Run kernel
    output = moe_expert_gemm_fp4(
        activations,
        expert_weights_packed,
        scales,
        expert_ids,
        expert_probs,
        group_size=group_size,
    )

    print(f"Output: {output.shape}")
    assert output.shape == (batch, intermediate), f"Expected {(batch, intermediate)}"
    print("MoE kernel test PASSED!")


def test_moe_kernel_glm_dimensions():
    """Test with GLM-4.7-Flash-like dimensions."""
    # GLM-4.7-Flash: 30B-A3B MoE
    batch = 8
    hidden = 2048  # actual GLM hidden
    intermediate = 1536  # GLM moe_intermediate
    num_experts = 64  # GLM has 64 routed experts
    top_k = 4  # activates 4 experts per token
    group_size = 128

    print("\nGLM-4.7-Flash dimensions test:")
    print(f"  batch={batch}, hidden={hidden}, intermediate={intermediate}")
    print(f"  num_experts={num_experts}, top_k={top_k}")

    # Create expert weights and quantize
    expert_weights_fp16 = torch.randn(num_experts, hidden, intermediate, dtype=torch.float16)

    packed_list = []
    scales_list = []
    for e in range(num_experts):
        weights_e = expert_weights_fp16[e].T.numpy()  # [intermediate, hidden]
        packed_e, scales_e = quantize_fp4(weights_e, group_size=group_size, marlin_layout=True)
        packed_list.append(torch.from_numpy(packed_e))
        scales_list.append(torch.from_numpy(scales_e))

    expert_weights_packed = torch.stack(packed_list).to("mps")
    scales = torch.stack(scales_list).to("mps")

    activations = torch.randn(batch, hidden, dtype=torch.float16, device="mps")
    expert_ids = torch.randint(0, num_experts, (batch, top_k), device="mps")
    expert_probs = torch.softmax(torch.randn(batch, top_k, device="mps"), dim=-1).half()

    print(f"  Expert weights: {expert_weights_packed.shape}")
    print(f"  Scales: {scales.shape}")

    output = moe_expert_gemm_fp4(
        activations,
        expert_weights_packed,
        scales,
        expert_ids,
        expert_probs,
        group_size=group_size,
    )

    print(f"  Output: {output.shape}")
    assert output.shape == (batch, intermediate), f"Expected {(batch, intermediate)}"
    print("GLM dimensions test PASSED!")


if __name__ == "__main__":
    test_moe_kernel_basic()
    test_moe_kernel_glm_dimensions()
