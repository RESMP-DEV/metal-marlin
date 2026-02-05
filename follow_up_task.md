# Follow-up Task: Remove remaining `int16` for trellis indices

**Context:**
The migration to `uint8` for trellis indices is incomplete.
A final check revealed remaining references to `np.int16` and `torch.int16` for trellis and other quantization indices in the `metal_marlin` codebase.

**Files to investigate:**
- `contrib/metal_marlin/metal_marlin/quantization/exl3_quantizer.py`
- `contrib/metal_marlin/metal_marlin/quantization/exl3_to_marlin.py`
- `contrib/metal_marlin/metal_marlin/quantization/ldlq.py`
- `contrib/metal_marlin/metal_marlin/metal_dispatch.py`
- `contrib/metal_marlin/metal_marlin/trellis/packing.py`
- `contrib/metal_marlin/metal_marlin/trellis/loader.py`

**Action:**
Refactor the code in the files listed above to use `uint8` instead of `int16` for trellis and quantization indices.
Ensure that all related calculations and data structures are updated to handle the change in data type.
The goal is to completely remove the usage of `int16` for these indices.
