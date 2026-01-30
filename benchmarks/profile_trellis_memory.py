"""Profile memory usage of trellis inference."""

import gc

import torch


def get_mps_memory() -> dict:
    """Get MPS memory statistics."""
    return {
        "allocated": torch.mps.current_allocated_memory() / 1024**3,
        "reserved": torch.mps.driver_allocated_memory() / 1024**3,
    }

def profile_layer_loading(model_path: str, num_layers: int = 5):
    """Profile memory during layer loading."""
    from metal_marlin.trellis.loader import TrellisModelLoader

    gc.collect()
    torch.mps.empty_cache()

    baseline = get_mps_memory()
    print(f"Baseline: {baseline['allocated']:.2f} GB allocated")

    loader = TrellisModelLoader(model_path)

    for layer_idx in range(num_layers):
        gc.collect()
        before = get_mps_memory()

        weights = loader.load_layer(layer_idx)

        gc.collect()
        after = get_mps_memory()

        delta = after['allocated'] - before['allocated']
        print(f"Layer {layer_idx}: +{delta*1024:.1f} MB "
              f"(total: {after['allocated']:.2f} GB)")

        # Test that clearing works
        loader.clear_layer_cache(layer_idx)
        gc.collect()
        torch.mps.empty_cache()

        cleared = get_mps_memory()
        print(f"  After clear: {cleared['allocated']:.2f} GB")

def profile_inference(model_path: str):
    """Profile memory during inference."""
    from metal_marlin.trellis.linear import TrellisLinear
    from metal_marlin.trellis.loader import TrellisModelLoader

    gc.collect()
    torch.mps.empty_cache()

    loader = TrellisModelLoader(model_path)
    weights = loader.load_layer(0)

    weight_name = list(weights.keys())[0]
    linear = TrellisLinear.from_trellis_weight(weights[weight_name], device="mps")

    print(f"\nTrellisLinear created: {get_mps_memory()['allocated']:.2f} GB")

    # Forward pass
    x = torch.randn(1, linear.in_features, dtype=torch.float16, device="mps")

    before = get_mps_memory()
    y = linear(x)
    torch.mps.synchronize()
    after = get_mps_memory()

    print(f"After forward: {after['allocated']:.2f} GB "
          f"(delta: {(after['allocated']-before['allocated'])*1024:.1f} MB)")

    # Check cache behavior
    linear.clear_cache()
    gc.collect()
    torch.mps.empty_cache()

    print(f"After cache clear: {get_mps_memory()['allocated']:.2f} GB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/GLM-4.7-Flash-EXL3-3bpw")
    parser.add_argument("--layers", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("Layer Loading Memory Profile")
    print("=" * 60)
    profile_layer_loading(args.model, args.layers)

    print("\n" + "=" * 60)
    print("Inference Memory Profile")
    print("=" * 60)
    profile_inference(args.model)
