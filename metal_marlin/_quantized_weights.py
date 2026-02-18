from typing import Any

import torch
from metal_marlin.mmfp4_loader import MMFP4ModelLoader


def _apply_quantized_weights(
    model: Any,
    loader: MMFP4ModelLoader,
    device: str,
) -> int:
    """Apply quantized weights from loader to model.

    Safetensors stores weights in [out_features, in_features//pack_factor] format,
    but MetalQuantizedLinear expects [in_features//pack_factor, out_features]
    (K-major packing for Metal kernels). This function handles the transpose.

    Args:
        model: The model with quantized linear layers
        loader: MMFP4ModelLoader with quantized weights
        device: Target device

    Returns:
        Number of weights loaded
    """
    from metal_marlin.inference_metal import MetalQuantizedLinear

    loaded = 0
    for name, module in model.named_modules():
        if not isinstance(module, MetalQuantizedLinear):
            continue

        # Try to find matching weight in loader
        weight_name = f"{name}.weight"
        try:
            qweight, scales = loader.get_quantized_weight(weight_name)
            if qweight is not None and scales is not None:
                # Safetensors: [out_features, in_features//pack_factor]
                # MetalQuantizedLinear: [in_features//pack_factor, out_features]
                # Need to TRANSPOSE both!
                qweight = qweight.T.contiguous()
                scales = scales.T.contiguous()

                # Move to device with correct dtypes
                qweight = qweight.to(device=device, dtype=torch.uint32)
                scales = scales.to(device=device, dtype=torch.float16)

                # Handle padding if needed
                if hasattr(module, "_needs_output_slice") and module._needs_output_slice:
                    pad_cols = module._padded_out_features - module.out_features
                    qweight = torch.nn.functional.pad(
                        qweight, (0, pad_cols, 0, 0))
                    scales = torch.nn.functional.pad(
                        scales, (0, pad_cols, 0, 0))

                # Copy to module
                if hasattr(module, "weight_packed"):
                    module.weight_packed.copy_(qweight)
                if hasattr(module, "scales"):
                    module.scales.copy_(scales)
                loaded += 1
        except Exception:
            # Weight not found in quantized checkpoint, skip
            pass

    return loaded
