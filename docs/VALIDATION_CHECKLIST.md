# GLM-4.7-Flash Validation Checklist

**End-to-End Testing for 35 tok/s Production Server**

## Pre-Test Setup

- [ ] **Hardware Check**: Verify M4 Max or equivalent Apple Silicon
  ```bash
  sysctl -n machdep.cpu.brand_string
  # Should show: Apple M4 Max
  ```

- [ ] **Memory Check**: Verify 32GB+ unified memory
  ```bash
  sysctl -n hw.memsize
  # Should be >= 34359738368 (32GB)
  ```

- [ ] **Model Weights**: Download/convert quantized weights
  ```bash
  cd /Users/kearm/AlphaHENG/contrib/metal_marlin
  uv run python scripts/convert_glm47_to_mmfp4.py \
    --model-path zai-org/GLM-4.7-Flash \
    --output-dir ./models/glm47-flash-mmfp4
  ```

- [ ] **Dependencies**: Install required packages
  ```bash
  uv sync --extra all
  uv add requests openai  # For test scripts
  ```

## Server Startup

- [ ] **Start Server**: Launch with production config
  ```bash
  uv run python scripts/serve_glm47.py \
    --model-path ./models/glm47-flash-mmfp4 \
    --host 0.0.0.0 \
    --port 8000 \
    --max-batch-size 32 \
    --max-seq-len 8192
  ```

- [ ] **Verify Startup**: Check server logs for successful initialization
  - Should see: "Model path verified"
  - Should see: "MPS device available"
  - Should see: "Ready to serve requests!"

- [ ] **Health Check**: Test basic endpoint
  ```bash
  curl http://localhost:8000/health
  # Expected: {"status":"ok","model_loaded":true}
  ```

## Performance Tests

### Test 1: Single Request TPS

- [ ] **Run TPS Benchmark**
  ```bash
  uv run python tests/manual/test_e2e_validation.py
  ```
  
- [ ] **Expected Results**:
  - Average TPS: **≥35.0 tok/s**
  - Latency: **<30.0 ms/token**

### Test 2: Concurrent Requests

- [ ] **Run Concurrent Test**
  ```bash
  uv run python tests/manual/test_e2e_validation.py
  ```
  
- [ ] **Expected Results**:
  - 5 concurrent requests complete successfully
  - Aggregate TPS: **≥100 tok/s** (5 × 20 tok/s minimum)

### Test 3: Latency Breakdown

- [ ] **Run Latency Test**
  ```bash
  uv run python tests/manual/test_e2e_validation.py
  ```
  
- [ ] **Expected Results**:
  - TTFT (Time to First Token): **<50 ms**
  - TPOT (Time Per Output Token): **<30 ms**

## Quality Tests

### Test 4: Model Quality

- [ ] **Run Quality Test**
  ```bash
  uv run python tests/manual/test_e2e_validation.py
  ```
  
- [ ] **Expected Results**:
  - Arithmetic test: "2 + 2 = ?" → should contain "4"
  - Knowledge test: "Capital of France" → should contain "Paris"
  - Instruction test: "Convert to lowercase: HELLO" → should contain "hello"

### Test 5: Perplexity

- [ ] **Run Perplexity Benchmark**
  ```bash
  uv run python tests/manual/benchmark_perplexity.py \
    --model-path ./models/glm47-flash-mmfp4 \
    --dataset wikitext
  ```
  
- [ ] **Expected Results**:
  - Perplexity: **<25.0** (lower is better)
  - Good models typically: 8-15 on WikiText
  - Acceptable: <25

### Test 6: Streaming

- [ ] **Test Streaming Response**
  ```bash
  curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -N \
    -d '{
      "model":"glm-4.7-flash",
      "messages":[{"role":"user","content":"Count to 10"}],
      "stream":true,
      "max_tokens":50
    }'
  ```
  
- [ ] **Expected Results**:
  - Should see multiple `data:` chunks
  - Each chunk should have `chat.completion.chunk` object
  - Final message: `data: [DONE]`

## API Compatibility Tests

### Test 7: OpenAI SDK Compatibility

- [ ] **Test with OpenAI Python SDK**
  ```python
  from openai import OpenAI
  
  client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
  response = client.chat.completions.create(
      model="glm-4.7-flash",
      messages=[{"role": "user", "content": "Hello"}],
      max_tokens=50
  )
  print(response.choices[0].message.content)
  ```
  
- [ ] **Expected Results**:
  - No errors
  - Coherent response generated
  - Response object has all expected fields (id, choices, usage)

### Test 8: List Models

- [ ] **Test Model Listing**
  ```bash
  curl http://localhost:8000/v1/models
  ```
  
- [ ] **Expected Results**:
  - Returns JSON with `data` array
  - Contains model with id "glm-4.7-flash"

### Test 9: Error Handling

- [ ] **Test Invalid Model Name**
  ```bash
  curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"invalid-model","messages":[{"role":"user","content":"test"}]}'
  ```
  
- [ ] **Expected Results**:
  - Returns 404 error
  - Error message mentions model not found

## Monitoring Tests

### Test 10: Metrics Endpoint

- [ ] **Test Metrics**
  ```bash
  curl http://localhost:8000/metrics
  ```
  
- [ ] **Expected Results**:
  - Returns Prometheus-format metrics
  - Contains `throughput_tok_sec` metric
  - Contains `avg_latency_ms` metric

## Production Readiness

### Load Testing

- [ ] **Sustained Load Test** (Optional)
  ```bash
  # Run 100 requests in parallel
  for i in {1..100}; do
    curl -X POST http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"glm-4.7-flash","messages":[{"role":"user","content":"test"}],"max_tokens":20}' &
  done
  wait
  ```
  
- [ ] **Expected Results**:
  - All requests complete (some may timeout)
  - No server crashes
  - Memory usage stable (~12-14GB)

### Long-Running Stability

- [ ] **24-Hour Test** (Optional)
  - Keep server running for 24 hours
  - Periodically send test requests
  - Monitor memory growth
  
- [ ] **Expected Results**:
  - No memory leaks
  - Consistent performance
  - No crashes

## Final Validation

- [ ] **Run Complete Validation Suite**
  ```bash
  uv run python tests/manual/test_e2e_validation.py
  ```
  
- [ ] **All Tests Pass**:
  - Test 1 (TPS): ✅ ≥35.0 tok/s
  - Test 2 (Latency): ✅ <30 ms/token
  - Test 3 (Concurrent): ✅ Success
  - Test 4 (Quality): ✅ All pass
  - Test 5 (Perplexity): ✅ <25.0
  - Test 6 (Streaming): ✅ Works

## Success Criteria

**Minimum Requirements**:
- ✅ Throughput: ≥35.0 tok/s (single request)
- ✅ Latency: <30.0 ms/token
- ✅ Perplexity: <25.0
- ✅ Quality tests: 3/3 pass
- ✅ OpenAI API: Full compatibility
- ✅ Concurrent: 5+ simultaneous requests

**Production Ready**:
- ✅ All above tests pass
- ✅ No memory leaks after 1 hour
- ✅ Graceful error handling
- ✅ Metrics endpoint functional

## Troubleshooting

### Low TPS (<35 tok/s)
- Check GPU memory usage: `sudo powermetrics --samplers gpu_power`
- Reduce batch size: `--max-batch-size 16`
- Check for thermal throttling: `sudo powermetrics --samplers smc`

### High Perplexity (>25)
- Verify model weights downloaded correctly
- Check model path is correct
- Re-run quantization with more calibration data

### Memory Issues
- Reduce sequence length: `--max-seq-len 4096`
- Enable quantized KV cache
- Check for memory leaks in logs

### Connection Refused
- Verify server is running: `lsof -i :8000`
- Check firewall settings
- Verify correct host/port

## References

- [Serving Guide](serving_guide.md) - Complete documentation
- [Quick Reference](QUICKSTART_SERVING.md) - Fast startup guide
- [Completed Optimizations](../tasks/completed_optimizations.yaml) - Performance history

---

**Last Updated**: February 2026  
**Version**: 1.0
