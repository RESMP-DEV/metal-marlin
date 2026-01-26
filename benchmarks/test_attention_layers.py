#!/usr/bin/env python3
"""Test attention layers for RTN vs MR-GPTQ comparison."""

import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent.parent))
from metal_marlin.mr_gptq import MRGPTQQuantizer

model_dir = Path(__file__).parent.parent / "models" / "Qwen3-30B-A3B"
st_file = model_dir / "model-00001-of-00016.safetensors"

rtn_q = MRGPTQQuantizer(4, "fp4", 128, use_hadamard=False)
mr_q = MRGPTQQuantizer(4, "fp4", 128, use_hadamard=True, hadamard_block_size=64)

print("=" * 70)
print("Testing ATTENTION Layers (larger, more outliers)")
print("=" * 70 + "\n")

with safe_open(str(st_file), framework="pt") as f:
    # Test attention QKV projections
    attn_keys = [k for k in f.keys() if "self_attn" in k and "proj" in k and "weight" in k][:4]

    rtn_res, mr_res = [], []
    for name in attn_keys:
        tensor = f.get_tensor(name).float().numpy()
        if tensor.ndim != 2 or tensor.shape[1] % 128 != 0:
            continue

        print(f"{name}: {tensor.shape}")

        _, _, m_rtn = rtn_q.quantize_layer(tensor, None, name)
        _, _, m_mr = mr_q.quantize_layer(tensor, None, name)

        rtn_rmse = m_rtn["error"]["rmse"]
        mr_rmse = m_mr["error"]["rmse"]
        imp = (rtn_rmse - mr_rmse) / rtn_rmse * 100

        print(f"  RTN:     RMSE={rtn_rmse:.6f}")
        print(f"  MR-GPTQ: RMSE={mr_rmse:.6f} ({imp:+.1f}%)")
        print()

        rtn_res.append(rtn_rmse)
        mr_res.append(mr_rmse)

avg_imp = (np.mean(rtn_res) - np.mean(mr_res)) / np.mean(rtn_res) * 100
print(f"Average RMSE Improvement on Attention: {avg_imp:+.1f}%")
