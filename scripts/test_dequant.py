#!/usr/bin/env python3
"""Compare TrellisLinear forward vs manual matmul with dequantized weights."""

import torch

from metal_marlin.trellis.testing import create_mock_trellis_linear

torch.manual_seed(0)
linear = create_mock_trellis_linear(256, 128, bits=3, device="mps")

# Metal dequant
dq = linear.dequantize()
print(f"Dequant shape: {dq.shape}")  # Should be [out=128, in=256]
print(f"Dequant range: [{dq.min():.4f}, {dq.max():.4f}]")

# Test if TrellisLinear forward matches manual matmul with dequant
torch.manual_seed(100)
x = torch.randn(1, 256, dtype=torch.float16, device="mps")
print(f"\nInput x shape: {x.shape}, range: [{x.min():.4f}, {x.max():.4f}]")

out_kernel = linear(x)
print(
    f"TrellisLinear forward: max={out_kernel.abs().max():.2f}, mean={out_kernel.abs().mean():.2f}"
)

# Manual: y = x @ W^T where W is [out, in]
out_manual = x @ dq.T
print(f"Manual x @ dq.T: max={out_manual.abs().max():.2f}, mean={out_manual.abs().mean():.2f}")

# Check diff
diff = (out_kernel - out_manual).abs()
print(f"Diff: max={diff.max():.2f}, mean={diff.mean():.2f}")
print(
    f"Ratio (manual/kernel): {out_manual.abs().max().item() / out_kernel.abs().max().item():.2f}x"
)
