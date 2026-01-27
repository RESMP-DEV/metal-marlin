"""Command-line interface for Metal Marlin inference and quantization."""

import json
import os
import tempfile
from pathlib import Path

import click

# Quantization methods
QUANT_METHODS = ["rtn", "gptq", "mr-gptq"]

# Quantization formats
QUANT_FORMATS = ["fp4", "int4", "nf4", "int3", "int2", "int8"]

# Calibration sources
CALIBRATION_SOURCES = ["bartowski-v3", "wikitext2", "c4"]

# Quality benchmark datasets (aliases normalized below)
QUALITY_DATASETS = ["wikitext", "wikitext2", "wikitext-2", "c4", "bartowski-v3"]


@click.group()
def cli():
    """Metal Marlin: FP4-quantized LLM inference on Apple Silicon."""


@cli.command()
@click.option("--model", "-m", required=True, help="Model path")
@click.option("--prompt", "-p", required=True, help="Input prompt")
@click.option("--max-tokens", default=256, type=int)
@click.option("--temperature", default=0.7, type=float)
@click.option("--top-p", default=0.9, type=float)
@click.option("--quant", default="fp4", type=click.Choice(["fp4", "int4"]))
@click.option("--stream", is_flag=True)
def generate(model, prompt, max_tokens, temperature, top_p, quant, stream):
    """Generate text from a prompt."""
    from .inference import MarlinPipeline

    pipe = MarlinPipeline.from_pretrained(model, quant_type=quant)

    if stream:
        for token in pipe(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        ):
            click.echo(token, nl=False)
        click.echo()
    else:
        output = pipe(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        click.echo(output)


@cli.command()
@click.option("--model", "-m", required=True, help="Model path")
@click.option("--system", default="You are a helpful assistant.")
@click.option("--quant", default="fp4", type=click.Choice(["fp4", "int4"]))
def chat(model, system, quant):
    """Interactive chat session."""
    from .inference import chat as run_chat

    run_chat(model, system_prompt=system)


@click.command()
@click.argument("model_path", type=click.Path())
@click.option("--host", default="0.0.0.0", help="Bind address")
@click.option("--port", default=8000, type=int, help="Port number")
@click.option("--device", default="mps", help="Device (mps/cuda/cpu)")
def serve(model_path: str, host: str, port: int, device: str):
    """Start OpenAI-compatible API server.

    Example:
        metal-marlin serve benchmarks/results/qwen3_4b_fp4 --port 8000

    Then use with:
        curl http://localhost:8000/v1/chat/completions \
          -H "Content-Type: application/json" \
          -d '{"model": "qwen3_4b_fp4", "messages": [{"role": "user", "content": "Hello"}]}'
    """
    from .serving.server import run_server

    if os.getenv("METAL_MARLIN_MOCK_MODEL") != "1":
        if not Path(model_path).exists():
            raise click.ClickException(f"Model path not found: {model_path}")

    run_server(model_path, host=host, port=port, device=device)


if "serve" not in cli.commands:
    cli.add_command(serve)


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Input model path")
@click.option("--output", "-o", "output_path", required=True, help="Output path")
@click.option(
    "--from-format",
    "from_format",
    default="auto",
    type=click.Choice(["auto", "safetensors", "gguf", "pytorch", "hf"]),
    help="Input format (auto-detected if not specified)",
)
@click.option(
    "--to-format",
    "to_format",
    default="marlin",
    type=click.Choice(["marlin", "safetensors", "gguf"]),
    help="Output format",
)
@click.option(
    "--quant",
    default="fp4",
    type=click.Choice(QUANT_FORMATS, case_sensitive=False),
    help="Quantization format for output",
)
@click.option("--group-size", "-g", default=128, type=int, help="Group size for quantization")
@click.option(
    "--dequant-first",
    is_flag=True,
    default=False,
    help="Dequantize to FP16 before re-quantizing (for format conversion)",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def convert(
    input_path: str,
    output_path: str,
    from_format: str,
    to_format: str,
    quant: str,
    group_size: int,
    dequant_first: bool,
    verbose: bool,
):
    """Convert model between formats.

    \b
    Supported conversions:
      - HuggingFace (safetensors/pytorch) → Marlin FP4/INT4
      - GGUF → Marlin FP4/INT4 (dequant + requant)
      - Marlin → safetensors (for inspection)

    \b
    Examples:
      # Convert HF model to Marlin FP4
      metal-marlin convert -i ./model -o ./model-marlin --quant fp4

      # Convert GGUF to Marlin (dequant + requant)
      metal-marlin convert -i model.gguf -o ./model-marlin --dequant-first

      # Convert existing quantized model to different format
      metal-marlin convert -i ./model-int4 -o ./model-fp4 --quant fp4 --dequant-first
    """
    if verbose:
        click.echo(f"Converting: {input_path} → {output_path}")
        click.echo(f"  From format: {from_format}")
        click.echo(f"  To format:   {to_format}")
        click.echo(f"  Quant type:  {quant}")
        click.echo(f"  Group size:  {group_size}")

    # Auto-detect input format
    if from_format == "auto":
        input_path_lower = input_path.lower()
        if input_path_lower.endswith(".gguf"):
            from_format = "gguf"
        elif Path(input_path).is_dir():
            # Check for safetensors or pytorch files
            p = Path(input_path)
            if list(p.glob("*.safetensors")):
                from_format = "safetensors"
            elif list(p.glob("*.bin")) or list(p.glob("*.pt")):
                from_format = "pytorch"
            else:
                from_format = "hf"
        else:
            from_format = "safetensors"

        if verbose:
            click.echo(f"  Detected input format: {from_format}")

    # Route to appropriate converter
    if from_format == "gguf":
        from .gguf_to_marlin import convert_gguf_to_marlin

        stats = convert_gguf_to_marlin(
            input_path,
            output_path,
            quant_type=quant,
            group_size=group_size,
            verbose=verbose,
        )
    elif from_format in ("safetensors", "pytorch", "hf"):
        if dequant_first:
            # Dequantize then requantize
            from .hf_loader import convert_model_to_fp4

            stats = convert_model_to_fp4(
                model_path=input_path,
                output_path=output_path,
                group_size=group_size,
                validate=True,
                verbose=verbose,
            )
        else:
            from .safetensors_loader import convert_model_to_marlin

            stats = convert_model_to_marlin(
                input_path,
                output_path,
                quant_type=quant,
                group_size=group_size,
            )
    else:
        click.echo(f"Error: Unsupported input format: {from_format}", err=True)
        raise click.Abort()

    click.echo(f"Converted model saved to {output_path}")
    if isinstance(stats, dict) and "quantized_count" in stats:
        click.echo(f"  Quantized: {stats['quantized_count']} tensors")


@cli.command()
@click.option("--model", "-m", required=True)
@click.option("--prompt-len", default=128, type=int)
@click.option("--gen-len", default=128, type=int)
@click.option("--batch-size", default=1, type=int)
def bench(model, prompt_len, gen_len, batch_size):
    """Benchmark inference performance."""
    from .benchmarks import benchmark_inference

    benchmark_inference(
        model,
        prompt_len=prompt_len,
        gen_len=gen_len,
        batch_size=batch_size,
    )


@cli.command("quantize")
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    help="Input model path (HuggingFace ID or local directory)",
)
@click.option(
    "--output", "-o", "output_path", required=True, help="Output directory for quantized model"
)
@click.option(
    "--method",
    "-m",
    default="rtn",
    type=click.Choice(QUANT_METHODS, case_sensitive=False),
    help="Quantization method: rtn (round-to-nearest), gptq, or mr-gptq (Marlin-Replica GPTQ with Hadamard)",
)
@click.option(
    "--bits",
    "-b",
    default=4,
    type=click.Choice([2, 3, 4, 8], case_sensitive=False),
    help="Bit width for quantization",
)
@click.option(
    "--format",
    "-f",
    "quant_format",
    default="fp4",
    type=click.Choice(QUANT_FORMATS, case_sensitive=False),
    help="Quantization format (fp4=E2M1, int4=symmetric, nf4=NormalFloat)",
)
@click.option(
    "--group-size",
    "-g",
    default=128,
    type=click.Choice([32, 64, 128, 256], case_sensitive=False),
    help="Elements per quantization group",
)
@click.option(
    "--calibration",
    "-c",
    default=None,
    help="Calibration dataset: bartowski-v3, wikitext2, c4, or path to custom file",
)
@click.option(
    "--samples",
    "-s",
    default=None,
    type=int,
    help="Number of calibration samples (default: all available)",
)
@click.option(
    "--no-hadamard",
    is_flag=True,
    default=False,
    help="Disable Hadamard rotation (only applies to mr-gptq method)",
)
@click.option(
    "--hadamard-kurtosis-threshold",
    default=None,
    type=float,
    help=("Apply Hadamard only when excess kurtosis exceeds this threshold (mr-gptq only)."),
)
@click.option(
    "--actorder/--no-actorder",
    default=True,
    help="Enable activation-order quantization for GPTQ methods (default: enabled)",
)
@click.option(
    "--damp",
    default=0.01,
    type=float,
    help="Hessian damping factor for GPTQ methods (default: 0.01)",
)
@click.option(
    "--mixed-precision",
    "mixed_precision",
    default=None,
    type=click.Choice(["dense", "moe", "moe-mtp", "quality", "speed"]),
    help="Mixed-precision preset (auto-detected if not specified)",
)
@click.option(
    "--layerwise",
    is_flag=True,
    default=False,
    help="Use memory-efficient layer-wise conversion (for large models)",
)
@click.option(
    "--validate/--no-validate",
    default=True,
    help="Compute quantization error metrics (default: enabled)",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose output")
@click.option("--token", default=None, help="HuggingFace API token for gated models")
@click.option(
    "--workers",
    "-w",
    default=1,
    type=int,
    help="Number of parallel workers for layer processing (default: 1)",
)
@click.option(
    "--precision-config",
    "precision_config",
    default=None,
    type=click.Choice(["moe-balanced", "dense-optimal", "quality-max", "speed-max"]),
    help="Precision configuration preset for layer-specific bit widths",
)
@click.option(
    "--use-transformers",
    is_flag=True,
    help="Load via Transformers and quantize in-place before saving",
)
def quantize(
    input_path: str,
    output_path: str,
    method: str,
    bits: int,
    quant_format: str,
    group_size: int,
    calibration: str | None,
    samples: int | None,
    no_hadamard: bool,
    hadamard_kurtosis_threshold: float | None,
    actorder: bool,
    damp: float,
    mixed_precision: str | None,
    layerwise: bool,
    validate: bool,
    verbose: bool,
    token: str | None,
    workers: int,
    precision_config: str | None,
    use_transformers: bool,
):
    """Quantize a model to Metal Marlin format.

    Supports three quantization methods:

    \b
    RTN (Round-to-Nearest):
      Fast, no calibration needed. Lower quality than GPTQ methods.
      Example: metal-marlin quantize -i model -o output -m rtn

    \b
    GPTQ (Gradient-based PTQ):
      Uses Hessian from calibration data for error compensation.
      Example: metal-marlin quantize -i model -o output -m gptq -c bartowski-v3

    \b
    MR-GPTQ (Marlin-Replica GPTQ):
      GPTQ with Hadamard rotation for outlier dispersal. Best quality.
      Example: metal-marlin quantize -i model -o output -m mr-gptq -c bartowski-v3

    \b
    Full example with all options:
      metal-marlin quantize \\
          --input Qwen/Qwen3-32B \\
          --output ./Qwen3-32B-MR-GPTQ-FP4 \\
          --method mr-gptq \\
          --bits 4 \\
          --format fp4 \\
          --group-size 128 \\
          --calibration bartowski-v3 \\
          --samples 512
    """
    import os

    # Get token from environment if not provided
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # Validate method-specific requirements
    if method in ("gptq", "mr-gptq") and calibration is None:
        click.echo(
            f"Warning: {method.upper()} method without calibration data will use "
            "weight-only Hessian approximation. For best results, provide "
            "--calibration bartowski-v3 or similar.",
            err=True,
        )

    # Validate format vs bits compatibility
    format_bits_map = {
        "fp4": 4,
        "int4": 4,
        "nf4": 4,
        "int3": 3,
        "int2": 2,
        "int8": 8,
    }
    expected_bits = format_bits_map.get(quant_format, bits)
    if bits != expected_bits:
        click.echo(
            f"Warning: --bits {bits} does not match --format {quant_format} "
            f"(expected {expected_bits} bits). Using format's bit width.",
            err=True,
        )
        bits = expected_bits

    # Display configuration
    if verbose:
        click.echo("=" * 60)
        click.echo("Quantization Configuration")
        click.echo("=" * 60)
        click.echo(f"  Input:        {input_path}")
        click.echo(f"  Output:       {output_path}")
        click.echo(f"  Method:       {method.upper()}")
        click.echo(f"  Format:       {quant_format.upper()} ({bits}-bit)")
        click.echo(f"  Group size:   {group_size}")
        click.echo(f"  Calibration:  {calibration or 'None (weight-only)'}")
        if samples:
            click.echo(f"  Samples:      {samples}")
        if method == "mr-gptq":
            click.echo(f"  Hadamard:     {'disabled' if no_hadamard else 'enabled'}")
            if hadamard_kurtosis_threshold is not None:
                click.echo(f"  Hadamard kurtosis threshold: {hadamard_kurtosis_threshold}")
        if method in ("gptq", "mr-gptq"):
            click.echo(f"  Act-order:    {'enabled' if actorder else 'disabled'}")
            click.echo(f"  Damp:         {damp}")
        if mixed_precision:
            click.echo(f"  Mixed prec:   {mixed_precision}")
        if precision_config:
            click.echo(f"  Prec config:  {precision_config}")
        click.echo(f"  Layer-wise:   {'enabled' if layerwise else 'disabled'}")
        click.echo(f"  Transformers: {'enabled' if use_transformers else 'disabled'}")
        click.echo(f"  Workers:      {workers}")
        click.echo("=" * 60)

    # Route to appropriate quantization function
    if method == "rtn":
        _quantize_rtn(
            input_path=input_path,
            output_path=output_path,
            quant_format=quant_format,
            group_size=group_size,
            calibration=calibration,
            samples=samples,
            mixed_precision=mixed_precision,
            precision_config=precision_config,
            layerwise=layerwise,
            use_transformers=use_transformers,
            validate=validate,
            verbose=verbose,
            token=token,
            workers=workers,
        )
    elif method == "gptq":
        _quantize_gptq(
            input_path=input_path,
            output_path=output_path,
            quant_format=quant_format,
            group_size=group_size,
            calibration=calibration,
            samples=samples,
            actorder=actorder,
            damp=damp,
            mixed_precision=mixed_precision,
            precision_config=precision_config,
            validate=validate,
            verbose=verbose,
            token=token,
            workers=workers,
        )
    elif method == "mr-gptq":
        _quantize_mr_gptq(
            input_path=input_path,
            output_path=output_path,
            quant_format=quant_format,
            group_size=group_size,
            calibration=calibration,
            samples=samples,
            use_hadamard=not no_hadamard,
            hadamard_kurtosis_threshold=hadamard_kurtosis_threshold,
            actorder=actorder,
            damp=damp,
            mixed_precision=mixed_precision,
            precision_config=precision_config,
            validate=validate,
            verbose=verbose,
            token=token,
            workers=workers,
        )


def _quantize_rtn(
    input_path: str,
    output_path: str,
    quant_format: str,
    group_size: int,
    calibration: str | None,
    samples: int | None,
    mixed_precision: str | None,
    precision_config: str | None,
    layerwise: bool,
    use_transformers: bool,
    validate: bool,
    verbose: bool,
    token: str | None,
    workers: int,
):
    """RTN (Round-to-Nearest) quantization - existing implementation."""
    from .hf_loader import (
        CalibrationData,
        convert_model_layerwise,
        convert_model_to_fp4,
        convert_model_transformers,
        convert_model_with_calibration,
    )

    # Determine calibration path vs source
    cal_path = None
    cal_source = None
    if calibration:
        if Path(calibration).exists() and calibration.endswith(".json"):
            cal_path = calibration
        elif calibration in CALIBRATION_SOURCES:
            cal_source = calibration
        else:
            # Assume custom file path
            cal_path = calibration

    # Route based on options
    if use_transformers:
        if layerwise:
            click.echo("Warning: --use-transformers ignores --layerwise", err=True)

        # Load mixed-precision config if requested
        mp_config = None
        if mixed_precision:
            from .mixed_precision import MixedPrecisionConfig as MPConfig

            preset_map = {
                "dense": MPConfig.default_dense,
                "moe": MPConfig.default_moe,
                "moe-mtp": MPConfig.default_moe_mtp,
                "quality": MPConfig.quality_first,
                "speed": MPConfig.speed_first,
            }
            mp_config = preset_map[mixed_precision]()

        stats = convert_model_transformers(
            model_id=input_path,
            output_path=output_path,
            group_size=group_size,
            mixed_precision=mp_config,
            calibration=cal_path,
            calibration_source=cal_source,
            validate=validate,
            verbose=verbose,
            token=token,
        )
    elif layerwise:
        # Load mixed-precision config
        mp_config = None
        if mixed_precision:
            from .mixed_precision import MixedPrecisionConfig as MPConfig

            preset_map = {
                "dense": MPConfig.default_dense,
                "moe": MPConfig.default_moe,
                "moe-mtp": MPConfig.default_moe_mtp,
                "quality": MPConfig.quality_first,
                "speed": MPConfig.speed_first,
            }
            mp_config = preset_map[mixed_precision]()

        stats = convert_model_layerwise(
            model_path=input_path,
            output_path=output_path,
            group_size=group_size,
            mixed_precision=mp_config,
            calibration=cal_path,
            validate=validate,
            verbose=verbose,
            token=token,
        )
    elif mixed_precision or cal_source:
        stats = convert_model_with_calibration(
            model_path=input_path,
            output_path=output_path,
            calibration_path=cal_path,
            calibration_source=cal_source,
            mixed_precision=mixed_precision,
            group_size=group_size,
            validate=validate,
            verbose=verbose,
            token=token,
        )
    else:
        # Load calibration data if path provided
        calib_data = None
        if cal_path:
            calib_data = CalibrationData.from_json(cal_path)

        stats = convert_model_to_fp4(
            model_path=input_path,
            output_path=output_path,
            group_size=group_size,
            validate=validate,
            verbose=verbose,
            calibration=calib_data,
        )

    _print_quantization_summary(stats, output_path, "RTN")


def _quantize_gptq(
    input_path: str,
    output_path: str,
    quant_format: str,
    group_size: int,
    calibration: str | None,
    samples: int | None,
    actorder: bool,
    damp: float,
    mixed_precision: str | None,
    precision_config: str | None,
    validate: bool,
    verbose: bool,
    token: str | None,
    workers: int,
):
    """GPTQ quantization with Hessian-based error compensation."""
    # Check if GPTQ implementation exists
    try:
        from .gptq import GPTQQuantizer
    except ImportError:
        click.echo(
            "Error: GPTQ quantization not yet implemented.\n"
            "The GPTQ module is planned for Phase 21.3.\n"
            "For now, use --method rtn for RTN quantization.",
            err=True,
        )
        raise click.Abort()

    # Load calibration data
    calib_dataset = _load_calibration_dataset(calibration, samples, verbose)

    quantizer = GPTQQuantizer(
        bits=4 if quant_format in ("fp4", "int4", "nf4") else 8,
        group_size=group_size,
        sym=quant_format != "nf4",  # NF4 is asymmetric
        actorder=actorder,
        damp=damp,
    )

    # Run quantization
    stats = quantizer.quantize_model(
        model_path=input_path,
        output_path=output_path,
        calibration_data=calib_dataset,
        quant_format=quant_format,
        mixed_precision=mixed_precision,
        validate=validate,
        verbose=verbose,
        token=token,
    )

    _print_quantization_summary(stats, output_path, "GPTQ")


def _quantize_mr_gptq(
    input_path: str,
    output_path: str,
    quant_format: str,
    group_size: int,
    calibration: str | None,
    samples: int | None,
    use_hadamard: bool,
    hadamard_kurtosis_threshold: float | None,
    actorder: bool,
    damp: float,
    mixed_precision: str | None,
    precision_config: str | None,
    validate: bool,
    verbose: bool,
    token: str | None,
    workers: int,
):
    """MR-GPTQ quantization with Hadamard rotation and GPTQ."""
    # Check if MR-GPTQ implementation exists
    try:
        from .mr_gptq import MRGPTQQuantizer
    except ImportError:
        click.echo(
            "Error: MR-GPTQ quantization not yet implemented.\n"
            "The MR-GPTQ module is planned for Phase 21.4.\n"
            "For now, use --method rtn for RTN quantization.",
            err=True,
        )
        raise click.Abort()

    # Load calibration data
    calib_dataset = _load_calibration_dataset(calibration, samples, verbose)

    # Determine hadamard block size (typically matches group_size)
    hadamard_block_size = min(group_size, 64)  # Cap at 64 for efficiency

    quantizer = MRGPTQQuantizer(
        bits=4 if quant_format in ("fp4", "int4", "nf4") else 8,
        format=quant_format,
        group_size=group_size,
        use_hadamard=use_hadamard,
        hadamard_block_size=hadamard_block_size,
        hadamard_kurtosis_threshold=hadamard_kurtosis_threshold,
        actorder=actorder,
        damp=damp,
    )

    # Run quantization
    stats = quantizer.quantize_model(
        model_path=input_path,
        calibration_data=calib_dataset,
        output_path=output_path,
        mixed_precision=mixed_precision,
        validate=validate,
        verbose=verbose,
        token=token,
    )

    _print_quantization_summary(stats, output_path, "MR-GPTQ")


def _load_calibration_dataset(
    calibration: str | None,
    samples: int | None,
    verbose: bool,
):
    """Load calibration dataset from source or file."""
    if calibration is None:
        return None

    from .calibration import BartowskiCalibration, CalibrationDataset

    if calibration in CALIBRATION_SOURCES:
        if verbose:
            click.echo(f"Loading calibration dataset: {calibration}")

        if calibration == "bartowski-v3":
            return BartowskiCalibration.v3(max_samples=samples)
        elif calibration == "wikitext2":
            # Load WikiText-2 for calibration
            from .eval_perplexity import load_wikitext2

            texts = load_wikitext2(max_samples=samples)
            return CalibrationDataset(
                samples=texts,
                name="wikitext2",
                version="test",
            )
        elif calibration == "c4":
            from .hf_loader import _load_hf_calibration_dataset

            texts = _load_hf_calibration_dataset("c4", num_samples=samples)
            return CalibrationDataset(
                samples=texts,
                name="c4",
                version="train",
            )
    elif Path(calibration).exists():
        if verbose:
            click.echo(f"Loading calibration from file: {calibration}")
        return BartowskiCalibration.from_local(calibration)
    else:
        raise click.BadParameter(
            f"Unknown calibration source: {calibration}. "
            f"Use one of {CALIBRATION_SOURCES} or provide a file path."
        )


def _print_quantization_summary(stats: dict, output_path: str, method: str):
    """Print quantization summary."""
    click.echo()
    click.echo("=" * 60)
    click.echo(f"{method} Quantization Complete")
    click.echo("=" * 60)
    click.echo(f"  Output:       {output_path}")
    click.echo(f"  Quantized:    {stats.get('quantized_count', 'N/A')} tensors")
    click.echo(f"  Skipped:      {stats.get('skipped_count', 'N/A')} tensors")

    if "compression_ratio" in stats:
        click.echo(f"  Compression:  {stats['compression_ratio']:.2f}x")

    if "mean_rmse" in stats:
        click.echo(f"  Mean RMSE:    {stats['mean_rmse']:.6f}")

    if "max_error" in stats:
        click.echo(f"  Max error:    {stats['max_error']:.6f}")

    click.echo("=" * 60)


def _normalize_quality_dataset(dataset: str) -> str:
    dataset = dataset.lower()
    if dataset in ("wikitext", "wikitext2"):
        return "wikitext-2"
    return dataset


def _load_quality_texts(dataset: str, samples: int) -> list[str]:
    from .benchmark_models import load_eval_data

    dataset = _normalize_quality_dataset(dataset)
    return load_eval_data(dataset, num_samples=samples)


def _resolve_model_path(model: str, verbose: bool = False) -> Path | None:
    model_path = Path(model)
    if model_path.exists():
        return model_path

    try:
        from .hf_loader import download_model

        if verbose:
            click.echo(f"Downloading reference model: {model}")
        return download_model(model)
    except Exception as exc:
        if verbose:
            click.echo(f"Warning: could not resolve model path ({exc})", err=True)
        return None


def _estimate_model_memory_gb(model_path: Path, overhead: float = 1.1) -> float | None:
    if model_path is None or not model_path.exists():
        return None
    total_bytes = 0
    for file_path in model_path.glob("*.safetensors"):
        total_bytes += file_path.stat().st_size
    if total_bytes == 0:
        return None
    return (total_bytes / 1e9) * overhead


def _load_quantization_metadata(
    quantized_path: Path,
    bits: int | None = None,
    quant_format: str | None = None,
    group_size: int | None = None,
) -> tuple[dict[str, int | str | None], float | None]:
    meta: dict[str, object] = {}
    config_path = quantized_path / "quantization_config.json"
    if config_path.exists():
        meta = json.loads(config_path.read_text())

    format_bits_map = {
        "fp4": 4,
        "int4": 4,
        "nf4": 4,
        "int3": 3,
        "int2": 2,
        "int8": 8,
    }

    if bits is not None:
        try:
            bits = int(bits)
        except (TypeError, ValueError):
            bits = None

    raw_format = (
        quant_format
        or meta.get("quant_format")
        or meta.get("format")
        or meta.get("quantization_format")
    )
    fmt = None
    if raw_format:
        fmt = str(raw_format).lower().replace("marlin_", "").replace("marlin-", "")
        if fmt.startswith("fp4"):
            fmt = "fp4"

    if bits is None:
        if "bits" in meta and meta.get("bits") is not None:
            try:
                bits = int(meta["bits"])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                bits = None
        elif fmt in format_bits_map:
            bits = format_bits_map[fmt]

    if fmt is None and bits in format_bits_map.values():
        # Default to fp4 when bits=4 and format not provided
        fmt = "fp4" if bits == 4 else None

    if group_size is None:
        try:
            group_size = int(meta.get("group_size", 0)) or None
        except (TypeError, ValueError):
            group_size = None

    quant_info = {
        "bits": bits,
        "format": fmt,
        "group_size": group_size,
    }

    mean_rmse = meta.get("mean_rmse")
    if mean_rmse is not None:
        try:
            mean_rmse = float(mean_rmse)  # type: ignore[assignment]
        except (TypeError, ValueError):
            mean_rmse = None

    return quant_info, mean_rmse


def _write_quality_report(report: dict, output_path: str | None) -> None:
    payload = json.dumps(report, indent=2)
    if output_path:
        Path(output_path).write_text(payload)
        click.echo(f"Wrote quality report to {output_path}")
    else:
        click.echo(payload)


def _compute_layer_rmse(
    model_path: Path,
    quantized_path: Path,
    verbose: bool = False,
) -> tuple[dict[str, float], float | None]:
    from .hf_loader import load_layer_weights, load_model_config, load_quantized_weights
    from .quantize_fp4 import compute_quantization_error

    if model_path is None or not model_path.exists():
        raise click.ClickException("Reference model path not found for layer analysis.")

    quantized = load_quantized_weights(quantized_path)
    layer_map: dict[int, list[str]] = {}
    for name, entry in quantized.items():
        if "packed" not in entry:
            continue
        if not name.startswith("model.layers."):
            continue
        parts = name.split(".")
        if len(parts) < 3:
            continue
        try:
            layer_idx = int(parts[2])
        except ValueError:
            continue
        layer_map.setdefault(layer_idx, []).append(name)

    config = load_model_config(model_path)
    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        num_layers = max(layer_map.keys(), default=-1) + 1

    layer_rmse: dict[str, float] = {}
    rmse_values: list[float] = []

    for layer_idx in range(num_layers):
        names = layer_map.get(layer_idx)
        if not names:
            continue
        if verbose:
            click.echo(f"Analyzing layer {layer_idx} ({len(names)} tensors)...")
        weights = load_layer_weights(model_path, layer_idx)
        for name in names:
            entry = quantized.get(name, {})
            if "packed" not in entry:
                continue
            original = weights.get(name)
            if original is None:
                if verbose:
                    click.echo(f"  Warning: missing reference tensor {name}", err=True)
                continue
            if original.ndim != 2:
                continue
            err = compute_quantization_error(
                original,
                entry["packed"],
                entry["scales"],
                group_size=entry.get("group_size", 128),
            )
            layer_rmse[name] = float(err["rmse"])
            rmse_values.append(float(err["rmse"]))

    mean_rmse = float(sum(rmse_values) / len(rmse_values)) if rmse_values else None
    return layer_rmse, mean_rmse


@cli.group("quality")
def quality():
    """Quality benchmarks for quantized models."""


@quality.command("compare")
@click.option("--model", "-m", required=True, help="Reference model path or HuggingFace ID")
@click.option("--quantized", "-q", required=True, help="Path to quantized model")
@click.option(
    "--dataset",
    default="wikitext",
    type=click.Choice(QUALITY_DATASETS, case_sensitive=False),
    help="Evaluation dataset",
)
@click.option("--samples", default=100, type=int, help="Number of evaluation samples")
@click.option("--max-length", default=512, type=int, help="Max sequence length")
@click.option("--output", "-o", default=None, help="Output JSON path (stdout if omitted)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def quality_compare(
    model: str,
    quantized: str,
    dataset: str,
    samples: int,
    max_length: int,
    output: str | None,
    verbose: bool,
):
    """Compare quantized model quality against a BF16 reference."""
    from .benchmark_models import PUBLISHED_FP16_PPL, _benchmark_marlin, _compute_kl_divergence

    quantized_path = Path(quantized)
    if not quantized_path.exists():
        raise click.ClickException(f"Quantized model path not found: {quantized}")

    model_path = _resolve_model_path(model, verbose=verbose)

    quant_info, mean_rmse = _load_quantization_metadata(quantized_path)

    try:
        texts = _load_quality_texts(dataset, samples)
    except Exception as exc:
        if verbose:
            click.echo(f"Warning: failed to load dataset ({exc})", err=True)
        texts = []

    ppl_quant, throughput_quant, memory_quant = _benchmark_marlin(
        quantized_path, texts, max_length, verbose
    )

    ppl_ref = PUBLISHED_FP16_PPL.get(model)
    if ppl_ref is None and verbose:
        click.echo("Warning: reference perplexity not found; using null", err=True)

    delta_pct = None
    if ppl_ref is not None and ppl_ref != 0:
        delta_pct = ((ppl_quant - ppl_ref) / ppl_ref) * 100.0

    kl_mean = None
    kl_max = None
    if model_path is not None:
        try:
            kl_mean, kl_max, _ = _compute_kl_divergence(
                model_path,
                quantized_path,
                texts[: max(1, min(samples, 50))],
                max_length,
                verbose=verbose,
            )
        except Exception as exc:
            if verbose:
                click.echo(f"Warning: KL divergence skipped ({exc})", err=True)

    ref_memory = _estimate_model_memory_gb(model_path) if model_path else None
    compression = None
    if ref_memory and memory_quant:
        compression = ref_memory / memory_quant if memory_quant > 0 else None

    speedup = None
    ref_throughput = None
    if ref_throughput and throughput_quant:
        speedup = throughput_quant / ref_throughput

    report = {
        "model_id": model,
        "quantization": quant_info,
        "perplexity": {"ref": ppl_ref, "quant": ppl_quant, "delta_pct": delta_pct},
        "kl_divergence": {"mean": kl_mean, "max": kl_max},
        "layer_rmse": {},
        "mean_rmse": mean_rmse,
        "throughput": {
            "ref_tok_s": ref_throughput,
            "quant_tok_s": throughput_quant,
            "speedup": speedup,
        },
        "memory_gb": {"ref": ref_memory, "quant": memory_quant, "compression": compression},
    }

    _write_quality_report(report, output)


@quality.command("quick")
@click.option("--model", "-m", required=True, help="Reference model path or HuggingFace ID")
@click.option(
    "--bits",
    default=4,
    type=click.Choice([2, 3, 4, 8], case_sensitive=False),
    help="Quantization bit width",
)
@click.option(
    "--format",
    "quant_format",
    default="fp4",
    type=click.Choice(QUANT_FORMATS, case_sensitive=False),
    help="Quantization format",
)
@click.option("--group-size", "-g", default=128, type=int, help="Quantization group size")
@click.option(
    "--dataset",
    default="wikitext",
    type=click.Choice(QUALITY_DATASETS, case_sensitive=False),
    help="Evaluation dataset",
)
@click.option("--samples", default=20, type=int, help="Number of evaluation samples")
@click.option("--max-length", default=512, type=int, help="Max sequence length")
@click.option("--output", "-o", default=None, help="Output JSON path (stdout if omitted)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def quality_quick(
    model: str,
    bits: int,
    quant_format: str,
    group_size: int,
    dataset: str,
    samples: int,
    max_length: int,
    output: str | None,
    verbose: bool,
):
    """Quick quality check (reduced samples, no layer RMSE)."""
    try:
        bits = int(bits)
    except (TypeError, ValueError):
        raise click.BadParameter("Bits must be an integer.")

    if bits != 4 or str(quant_format).lower() != "fp4":
        raise click.BadParameter("quality quick currently supports fp4 4-bit only.")

    from .benchmark_models import PUBLISHED_FP16_PPL, _benchmark_marlin, _compute_kl_divergence
    from .hf_loader import convert_model_parallel

    model_path = _resolve_model_path(model, verbose=verbose)

    with tempfile.TemporaryDirectory(prefix="metal_marlin_quality_") as tmp_dir:
        quantized_path = Path(tmp_dir)
        if verbose:
            click.echo("Quantizing model (quick mode)...")
        convert_model_parallel(
            model_path or model,
            quantized_path,
            group_size=group_size,
            validate=False,
            verbose=verbose,
        )

        quant_info = {"bits": bits, "format": "fp4", "group_size": group_size}

        try:
            texts = _load_quality_texts(dataset, samples)
        except Exception as exc:
            if verbose:
                click.echo(f"Warning: failed to load dataset ({exc})", err=True)
            texts = []

        ppl_quant, throughput_quant, memory_quant = _benchmark_marlin(
            quantized_path, texts, max_length, verbose
        )

        ppl_ref = PUBLISHED_FP16_PPL.get(model)
        delta_pct = None
        if ppl_ref is not None and ppl_ref != 0:
            delta_pct = ((ppl_quant - ppl_ref) / ppl_ref) * 100.0

        kl_mean = None
        kl_max = None
        if model_path is not None:
            try:
                kl_mean, kl_max, _ = _compute_kl_divergence(
                    model_path,
                    quantized_path,
                    texts[: max(1, min(samples, 50))],
                    max_length,
                    verbose=verbose,
                )
            except Exception as exc:
                if verbose:
                    click.echo(f"Warning: KL divergence skipped ({exc})", err=True)

        ref_memory = _estimate_model_memory_gb(model_path) if model_path else None
        compression = None
        if ref_memory and memory_quant:
            compression = ref_memory / memory_quant if memory_quant > 0 else None

        report = {
            "model_id": model,
            "quantization": quant_info,
            "perplexity": {"ref": ppl_ref, "quant": ppl_quant, "delta_pct": delta_pct},
            "kl_divergence": {"mean": kl_mean, "max": kl_max},
            "layer_rmse": {},
            "mean_rmse": None,
            "throughput": {
                "ref_tok_s": None,
                "quant_tok_s": throughput_quant,
                "speedup": None,
            },
            "memory_gb": {"ref": ref_memory, "quant": memory_quant, "compression": compression},
        }

        _write_quality_report(report, output)


@quality.command("layers")
@click.option("--model", "-m", required=True, help="Reference model path or HuggingFace ID")
@click.option("--quantized", "-q", required=True, help="Path to quantized model")
@click.option("--output", "-o", default=None, help="Output JSON path (stdout if omitted)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def quality_layers(
    model: str,
    quantized: str,
    output: str | None,
    verbose: bool,
):
    """Layer-by-layer RMSE analysis for a quantized model."""
    quantized_path = Path(quantized)
    if not quantized_path.exists():
        raise click.ClickException(f"Quantized model path not found: {quantized}")

    model_path = _resolve_model_path(model, verbose=verbose)
    quant_info, mean_rmse_meta = _load_quantization_metadata(quantized_path)

    layer_rmse = {}
    mean_rmse = mean_rmse_meta
    try:
        layer_rmse, mean_rmse = _compute_layer_rmse(
            model_path,
            quantized_path,
            verbose=verbose,
        )
    except Exception as exc:
        if verbose:
            click.echo(f"Warning: layer RMSE skipped ({exc})", err=True)

    report = {
        "model_id": model,
        "quantization": quant_info,
        "perplexity": {"ref": None, "quant": None, "delta_pct": None},
        "kl_divergence": {"mean": None, "max": None},
        "layer_rmse": layer_rmse,
        "mean_rmse": mean_rmse,
        "throughput": {"ref_tok_s": None, "quant_tok_s": None, "speedup": None},
        "memory_gb": {"ref": None, "quant": None, "compression": None},
    }

    _write_quality_report(report, output)


@cli.command("eval")
@click.option("--model", "-m", required=True, help="Path to quantized model")
@click.option(
    "--reference", "-r", default=None, help="Path to reference model for comparison (optional)"
)
@click.option(
    "--metric",
    default="perplexity",
    type=click.Choice(["perplexity", "kl-divergence", "all"]),
    help="Evaluation metric",
)
@click.option(
    "--dataset",
    "-d",
    default="bartowski-v3",
    type=click.Choice(["bartowski-v3", "wikitext2", "c4"]),
    help="Evaluation dataset (bartowski-v3 recommended)",
)
@click.option("--samples", "-s", default=100, type=int, help="Number of evaluation samples")
@click.option("--context-length", default=2048, type=int, help="Context window size for perplexity")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def eval_model(
    model: str,
    reference: str | None,
    metric: str,
    dataset: str,
    samples: int,
    context_length: int,
    verbose: bool,
):
    """Evaluate quantized model quality.

    \b
    Examples:
      # Compute perplexity on WikiText-2
      metal-marlin eval -m ./model-fp4/ --metric perplexity

      # Compare against reference model
      metal-marlin eval -m ./model-fp4/ -r ./model-bf16/ --metric all

      # KL divergence from reference
      metal-marlin eval -m ./model-fp4/ -r ./model-bf16/ --metric kl-divergence
    """
    from .eval_perplexity import load_tokenizer

    click.echo(f"Loading model: {model}")
    load_tokenizer(model)

    # Note: Full evaluation requires inference implementation
    # For now, provide framework and documentation
    click.echo()
    click.echo("Note: Full model evaluation requires inference kernel integration.")
    click.echo("This command provides the evaluation framework.")
    click.echo()
    click.echo("To implement evaluation:")
    click.echo("  1. Load quantized model weights")
    click.echo("  2. Implement forward pass using Metal Marlin kernels")
    click.echo("  3. Pass logits function to evaluation metrics")
    click.echo()
    click.echo("Available metrics:")
    click.echo("  - perplexity: Sliding-window perplexity (llama.cpp compatible)")
    click.echo("  - kl-divergence: KL(reference || quantized)")
    click.echo()
    click.echo(f"Configured: {metric} on {dataset} ({samples} samples)")
    if reference:
        click.echo(f"Reference model: {reference}")


@cli.command("analyze")
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    help="Input model path (HuggingFace ID or local directory)",
)
@click.option(
    "--calibration",
    "-c",
    default="bartowski-v3",
    help="Calibration dataset for sensitivity analysis",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    default="sensitivity_report.json",
    help="Output path for sensitivity report",
)
@click.option("--samples", "-s", default=128, type=int, help="Number of calibration samples")
@click.option(
    "--layers", default=None, help="Specific layers to analyze (comma-separated, or 'all')"
)
@click.option("--bits", "-b", default="2,3,4,8", help="Bit widths to test (comma-separated)")
@click.option("--formats", "-f", default="fp4,int4,nf4", help="Formats to test (comma-separated)")
@click.option("--token", default=None, help="HuggingFace API token for gated models")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def analyze(
    input_path: str,
    calibration: str,
    output_path: str,
    samples: int,
    layers: str | None,
    bits: str,
    formats: str,
    token: str | None,
    verbose: bool,
):
    """Analyze layer sensitivity to quantization.

    Computes per-layer quantization error metrics across multiple bit widths
    and formats to identify optimal precision allocation.

    \b
    Output includes:
      - Per-layer RMSE and max error at each bit width
      - Hessian trace (importance) for GPTQ-based methods
      - Recommended precision config based on sensitivity

    \b
    Examples:
      # Analyze all layers with default settings
      metal-marlin analyze -i ./model -o sensitivity.json

      # Analyze specific layers
      metal-marlin analyze -i ./model --layers "model.layers.0,model.layers.1"

      # Test specific bit widths
      metal-marlin analyze -i ./model --bits "3,4" --formats "fp4,int4"

      # Full analysis with custom calibration
      metal-marlin analyze \\
          --input GLM-4/glm-4-9b-chat \\
          --calibration bartowski-v3 \\
          --output sensitivity_report.json \\
          --samples 256
    """
    import json
    import os

    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # Parse bit widths and formats
    bit_widths = [int(b.strip()) for b in bits.split(",")]
    format_list = [f.strip() for f in formats.split(",")]

    # Parse layers
    layer_list = None
    if layers and layers != "all":
        layer_list = [layer.strip() for layer in layers.split(",")]

    if verbose:
        click.echo("=" * 60)
        click.echo("Sensitivity Analysis Configuration")
        click.echo("=" * 60)
        click.echo(f"  Input:        {input_path}")
        click.echo(f"  Calibration:  {calibration}")
        click.echo(f"  Samples:      {samples}")
        click.echo(f"  Bit widths:   {bit_widths}")
        click.echo(f"  Formats:      {format_list}")
        click.echo(f"  Layers:       {layer_list or 'all'}")
        click.echo(f"  Output:       {output_path}")
        click.echo("=" * 60)

    # Load calibration dataset
    calib_dataset = _load_calibration_dataset(calibration, samples, verbose)

    # Run sensitivity analysis
    try:
        from .calibration import SensitivityAnalyzer
    except ImportError:
        click.echo(
            "Error: SensitivityAnalyzer not yet implemented.\n"
            "Creating placeholder report with analysis framework.",
            err=True,
        )
        # Create placeholder report
        report = {
            "model": input_path,
            "calibration": calibration,
            "samples": samples,
            "bit_widths_tested": bit_widths,
            "formats_tested": format_list,
            "status": "placeholder",
            "note": "Full sensitivity analysis requires SensitivityAnalyzer implementation",
            "framework": {
                "per_layer_metrics": ["rmse", "max_error", "hessian_trace"],
                "precision_recommendation": "computed from sensitivity scores",
            },
        }
        Path(output_path).write_text(json.dumps(report, indent=2))
        click.echo(f"Placeholder report written to {output_path}")
        return

    analyzer = SensitivityAnalyzer(
        model_path=input_path,
        calibration_data=calib_dataset,
        token=token,
    )

    report = analyzer.analyze(
        bit_widths=bit_widths,
        formats=format_list,
        layers=layer_list,
        verbose=verbose,
    )

    # Write report
    Path(output_path).write_text(json.dumps(report, indent=2))
    click.echo(f"Sensitivity report written to {output_path}")

    # Print summary
    if "summary" in report:
        click.echo()
        click.echo("=" * 60)
        click.echo("Sensitivity Analysis Summary")
        click.echo("=" * 60)
        summary = report["summary"]
        if "most_sensitive_layers" in summary:
            click.echo("Most sensitive layers (keep higher precision):")
            for layer in summary["most_sensitive_layers"][:5]:
                click.echo(f"  - {layer['name']}: error={layer['error']:.6f}")
        if "least_sensitive_layers" in summary:
            click.echo("Least sensitive layers (can use lower precision):")
            for layer in summary["least_sensitive_layers"][:5]:
                click.echo(f"  - {layer['name']}: error={layer['error']:.6f}")
        if "recommended_config" in summary:
            click.echo(f"Recommended config: {summary['recommended_config']}")
        click.echo("=" * 60)


def main():
    cli()


if __name__ == "__main__":
    main()
