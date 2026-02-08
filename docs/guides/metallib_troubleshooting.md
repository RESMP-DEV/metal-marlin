# Troubleshooting

## Metallib Issues

### "Metallib not found"

Build it:
```bash
cd contrib/metal_marlin
./scripts/build_metallib.sh
```

### "Multiple symbols" linker error

Duplicate kernel definitions. Find them:
```bash
grep -rh "^kernel void" src/ metal_marlin/ --include="*.metal" | \
  awk '{print $3}' | cut -d'(' -f1 | sort | uniq -c | awk '$1>1'
```

### Kernel not found in metallib

Check if kernel exists:
```bash
xcrun -sdk macosx metal-objdump -t metal_marlin/lib/metal_marlin.metallib | grep KERNEL_NAME
```

### Metallib is stale

Force rebuild:
```bash
./scripts/build_metallib.sh --force
```

Or check in Python:
```python
from metal_marlin import is_metallib_stale
if is_metallib_stale():
    print("Rebuild needed!")
```

## Performance Issues

### Slow kernel dispatch

Enable timing to find slow paths:
```python
from metal_marlin import enable_timing, get_timing_stats
enable_timing()
# ... run workload ...
print(get_timing_stats())
```

### High memory usage

Metallib is loaded once and cached. ~10-50MB is normal.
