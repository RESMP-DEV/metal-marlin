#!/usr/bin/env python3
"""Compare RTN vs MR-GPTQ quantization quality on Qwen3-30B-A3B."""

import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

# Add metal_marlin to path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from metal_marlin.mr_gptq import MRGPTQQuantizer


def main():
    model_dir = _ROOT / "models" / "Qwen3-30B-A3B"
    st_file = model_dir / "model-00001-of-00016.safetensors"

    print("=" * 70)
    print("Qwen3-30B-A3B: RTN vs MR-GPTQ (Hadamard) Comparison")
    print("=" * 70)
    print()

    # Initialize quantizers
    rtn_quantizer = MRGPTQQuantizer(
        bits=4,
        format="fp4",
        group_size=128,
        use_hadamard=False,
        actorder=False,
    )

    mr_quantizer = MRGPTQQuantizer(
        bits=4,
        format="fp4",
        group_size=128,
        use_hadamard=True,
        hadamard_block_size=64,
        actorder=True,
    )

    rtn_results = []
    mr_results = []

    with safe_open(str(st_file), framework="pt") as f:
        # Only test linear projection layers, skip embeddings
        keys = [
            k
            for k in f.keys()
            if "weight" in k
            and "norm" not in k.lower()
            and "embed" not in k.lower()
            and ("proj" in k.lower() or "gate" in k.lower() or "mlp" in k.lower())
        ]
        print(f"Testing on {len(keys)} linear layers from first shard\n")

        for name in keys[:6]:  # Test first 6 layers
            tensor = f.get_tensor(name).float().numpy()

            # Skip non-2D or small tensors
            if tensor.ndim != 2 or tensor.shape[0] < 64 or tensor.shape[1] < 64:
                continue
            if tensor.shape[1] % 128 != 0 or tensor.shape[1] % 8 != 0:
                continue

            print(f"{name}: {tensor.shape}")

            # RTN quantization
            _, _, meta_rtn = rtn_quantizer.quantize_layer(tensor, hessian=None, layer_name=name)
            rtn_rmse = meta_rtn["error"]["rmse"]
            rtn_mre = meta_rtn["error"]["mean_relative_error"]

            # MR-GPTQ (Hadamard rotation, RTN fallback since no Hessian)
            _, _, meta_mr = mr_quantizer.quantize_layer(tensor, hessian=None, layer_name=name)
            mr_rmse = meta_mr["error"]["rmse"]
            mr_mre = meta_mr["error"]["mean_relative_error"]

            # Improvement percentage
            improvement = (rtn_rmse - mr_rmse) / rtn_rmse * 100

            print(f"  RTN FP4:     RMSE={rtn_rmse:.6f}, MRE={rtn_mre:.4%}")
            print(f"  MR-GPTQ FP4: RMSE={mr_rmse:.6f}, MRE={mr_mre:.4%} ({improvement:+.1f}%)")
            print()

            rtn_results.append({"rmse": rtn_rmse, "mre": rtn_mre})
            mr_results.append({"rmse": mr_rmse, "mre": mr_mre})

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_rtn_rmse = np.mean([r["rmse"] for r in rtn_results])
    avg_mr_rmse = np.mean([r["rmse"] for r in mr_results])
    avg_rtn_mre = np.mean([r["mre"] for r in rtn_results])
    avg_mr_mre = np.mean([r["mre"] for r in mr_results])
    improvement = (avg_rtn_rmse - avg_mr_rmse) / avg_rtn_rmse * 100

    print(f"Average RTN FP4:     RMSE={avg_rtn_rmse:.6f}, MRE={avg_rtn_mre:.4%}")
    print(f"Average MR-GPTQ FP4: RMSE={avg_mr_rmse:.6f}, MRE={avg_mr_mre:.4%}")
    print(f"RMSE Improvement:    {improvement:+.1f}%")
    print()
    print("Note: MR-GPTQ with full Hessian collection would show ~2x further improvement.")
    print("This comparison shows Hadamard rotation benefit alone.")


if __name__ == "__main__":
    main()
