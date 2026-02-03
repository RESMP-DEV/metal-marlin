#!/usr/bin/env python3
"""Verify that vision_preprocess.metal has all required kernels for 1024x1024+ images."""

from pathlib import Path

REQUIRED_KERNELS = [
    "image_resize_bilinear_tiled",
    "image_resize_bilinear_tiled_shared",
    "image_resize_bilinear_4pixel",
    "center_crop_large",
    "preprocess_large_image_fused",
    "extract_patches",
    "extract_patches_vec4",
    "image_resize_bicubic_8x8",
    "resize_aspect_ratio_preserve",
]

def verify_kernels():
    shader_path = Path(__file__).parent / "src" / "vision_preprocess.metal"
    
    if not shader_path.exists():
        print(f"❌ ERROR: {shader_path} not found")
        return False
    
    shader_content = shader_path.read_text()
    
    print(f"✓ Found shader file: {shader_path}")
    print(f"✓ File size: {len(shader_content)} bytes ({len(shader_content.splitlines())} lines)")
    
    missing = []
    found = []
    
    for kernel in REQUIRED_KERNELS:
        if f"kernel void {kernel}(" in shader_content:
            found.append(kernel)
        else:
            missing.append(kernel)
    
    print(f"\n✓ Found {len(found)}/{len(REQUIRED_KERNELS)} required kernels for 1024x1024+ support:")
    for k in found:
        print(f"  ✓ {k}")
    
    if missing:
        print(f"\n❌ Missing {len(missing)} kernels:")
        for k in missing:
            print(f"  ✗ {k}")
        return False
    
    # Check for optimization constants
    if "TILE_SIZE = 16" in shader_content:
        print("\n✓ Found TILE_SIZE optimization constant (16x16 tiles)")
    
    if "1024x1024" in shader_content or "1024 x 1024" in shader_content:
        print("✓ Documentation mentions 1024x1024 support")
    
    print("\n✅ All required kernels for 1024x1024+ images are implemented!")
    return True

if __name__ == "__main__":
    import sys
    success = verify_kernels()
    sys.exit(0 if success else 1)
