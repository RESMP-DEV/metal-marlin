#!/usr/bin/env python3
"""Analyze optimization results and report the best variant."""

import json
from pathlib import Path

results_dir = Path('agent_workspace/opt_20260128_205817')
kernel_path = Path('/Users/kearm/AlphaHENG/contrib/metal_marlin/src/fusion/norm_linear.metal')

# Load baseline
baseline_file = results_dir / 'baseline.json'
if not baseline_file.exists():
    print('ERROR: Baseline results not found')
    exit(1)

baseline = json.loads(baseline_file.read_text())
baseline_results = baseline.get('results', {})

# Load all variant results
variants = []
for f in sorted(results_dir.glob('*.json')):
    if f.name in ['baseline.json', 'best_variant.json', 'llm_hypotheses.json']:
        continue
    try:
        data = json.loads(f.read_text())
        if data.get('compile_success') and data.get('results'):
            # Compute average speedup across problem sizes
            speedups = []
            for key, variant_us in data['results'].items():
                if key in baseline_results:
                    speedups.append(baseline_results[key] / variant_us)
            if speedups:
                avg_speedup = sum(speedups) / len(speedups)
                variants.append({
                    'name': data['variant'],
                    'speedup': avg_speedup,
                    'results': data['results'],
                    'file': f.name,
                })
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not parse {f.name}: {e}")
        continue

# Sort by speedup
variants.sort(key=lambda x: x['speedup'], reverse=True)

print('='*60)
print('Optimization Session: 20260128_205817')
print(f'Kernel: {kernel_path}')
print('='*60)
print()
print('Baseline results:')
for key, us in baseline_results.items():
    print(f'  {key}: {us:.2f} us')
print()
print(f'Total variants analyzed: {len(variants)}')
print()
print('Results (sorted by speedup):')
for v in variants[:15]:
    print(f'  {v["name"]}: {v["speedup"]:.3f}x')
print()

# Save best variant info for apply task
if variants and variants[0]['speedup'] > 1.02:
    best = variants[0]
    best_info = {
        'variant_name': best['name'],
        'speedup': best['speedup'],
        'kernel_path': str(kernel_path),
        'session_id': '20260128_205817',
    }
    (results_dir / 'best_variant.json').write_text(json.dumps(best_info, indent=2))
    print(f'BEST: {best["name"]} ({best["speedup"]:.3f}x speedup)')
    print('Saved to best_variant.json for auto-apply')
else:
    print('No significant improvement found (threshold: >2%)')
