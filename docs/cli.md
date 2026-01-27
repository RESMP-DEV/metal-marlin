# CLI Reference

Metal Marlin ships a unified `metal-marlin` CLI (Click-based). Run
`metal-marlin --help` to see all commands.

## Command Summary

- `quantize` - Convert HuggingFace or local models to Metal Marlin format
- `convert` - Convert between HF/GGUF/Marlin formats
- `generate` - One-shot text generation
- `chat` - Interactive chat session
- `serve` - OpenAI-compatible HTTP server
- `bench` - Throughput benchmark
- `eval` - Perplexity / KL evaluation framework
- `analyze` - Layer sensitivity analysis

## Quantize (`metal-marlin quantize`)

Quantize a model into Metal Marlin format.

```bash
metal-marlin quantize -i MODEL -o OUTPUT [OPTIONS]
```

**Key options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --method` | `rtn` | `rtn`, `gptq`, `mr-gptq` |
| `-b, --bits` | `4` | Bit width (2/3/4/8) |
| `-f, --format` | `fp4` | `fp4`, `int4`, `nf4`, `int3`, `int2`, `int8` |
| `-g, --group-size` | `128` | Elements per quantization group |
| `-c, --calibration` | none | `bartowski-v3`, `wikitext2`, `c4`, or file path |
| `-s, --samples` | all | Calibration sample count |
| `--mixed-precision` | none | `dense`, `moe`, `moe-mtp`, `quality`, `speed` |
| `--precision-config` | none | `moe-balanced`, `dense-optimal`, `quality-max`, `speed-max` |
| `--layerwise` | false | Memory-efficient layer-wise conversion |
| `--validate/--no-validate` | true | Compute quantization error stats |
| `-w, --workers` | `1` | Parallel layer workers |
| `--token` | env | HuggingFace token (gated models) |
| `-v, --verbose` | false | Verbose output |

**Examples:**

```bash
# RTN quantization (fast)
metal-marlin quantize -i Qwen/Qwen3-4B -o ./qwen3_4b_fp4

# MR-GPTQ with calibration
metal-marlin quantize \
  -i zai-org/GLM-4.7-Flash \
  -o ./glm47_fp4 \
  -m mr-gptq \
  -c bartowski-v3 \
  -g 128
```

## Convert (`metal-marlin convert`)

Convert between formats (HF, GGUF, Marlin).

```bash
metal-marlin convert -i INPUT -o OUTPUT [OPTIONS]
```

**Common options:**
- `--from-format` (`auto`, `safetensors`, `gguf`, `pytorch`, `hf`)
- `--to-format` (`marlin`, `safetensors`, `gguf`)
- `--quant` (`fp4`, `int4`, `nf4`, `int3`, `int2`, `int8`)
- `--group-size` (default 128)
- `--dequant-first` (dequantize to FP16 before re-quantizing)

**Example:**
```bash
metal-marlin convert -i model.gguf -o ./model-marlin --dequant-first
```

## Generate (`metal-marlin generate`)

One-shot generation from a prompt:

```bash
metal-marlin generate -m ./glm47_fp4 -p "Hello" --max-tokens 64 --quant fp4
```

## Chat (`metal-marlin chat`)

Interactive chat session:

```bash
metal-marlin chat -m ./glm47_fp4 --system "You are a helpful assistant."
```

## Serve (`metal-marlin serve`)

Start an OpenAI-compatible HTTP server:

```bash
metal-marlin serve ./glm47_fp4 --host 0.0.0.0 --port 8000 --device mps
```

## Bench (`metal-marlin bench`)

Simple throughput benchmark:

```bash
metal-marlin bench --model ./glm47_fp4 --prompt-len 128 --gen-len 128 --batch-size 1
```

## Eval (`metal-marlin eval`)

Evaluation framework (perplexity / KL):

```bash
metal-marlin eval -m ./glm47_fp4 --metric perplexity --dataset wikitext2
```

## Analyze (`metal-marlin analyze`)

Layer sensitivity analysis for precision planning:

```bash
metal-marlin analyze -i zai-org/GLM-4.7-Flash -o sensitivity_report.json --samples 256
```
