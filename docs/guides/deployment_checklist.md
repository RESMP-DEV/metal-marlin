# Production Deployment Checklist

## Pre-deployment

- [ ] Run full test suite: `uv run pytest tests/ -v`
- [ ] Run benchmarks to establish baseline
- [ ] Verify memory fits in target device
- [ ] Test with representative prompts

## Configuration

- [ ] Set appropriate batch size for memory
- [ ] Enable/disable fast MoE based on stability
- [ ] Configure fallback behavior
- [ ] Set up monitoring

## Monitoring

- [ ] Watch for NaN rates
- [ ] Monitor latency P99
- [ ] Track memory usage
- [ ] Alert on fallback rate > threshold

## Rollback

- [ ] Document how to disable fast MoE
- [ ] Keep slow path tested and ready
