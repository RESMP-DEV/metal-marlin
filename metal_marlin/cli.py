"""Command-line interface for Metal Marlin inference and quantization."""

from pathlib import Path

import click

# Quantization methods
QUANT_METHODS = ["rtn", "gptq", "mr-gptq"]

# Quantization formats
QUANT_FORMATS = ["fp4", "int4", "nf4", "int3", "int2", "int8"]

# Calibration sources
CALIBRATION_SOURCES = ["bartowski-v3", "wikitext2", "c4"]


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


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Input model path")
@click.option("--output", "-o", "output_path", required=True, help="Output path")
@click.option("--quant", default="fp4", type=click.Choice(["fp4", "int4"]))
@click.option("--group-size", default=128, type=int)
def convert(input_path, output_path, quant, group_size):
    """Convert model to Marlin format."""
    from .safetensors_loader import convert_model_to_marlin

    convert_model_to_marlin(
        input_path,
        output_path,
        quant_type=quant,
        group_size=group_size,
    )
    click.echo(f"Converted model saved to {output_path}")


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
    "--input", "-i", "input_path", required=True,
    help="Input model path (HuggingFace ID or local directory)"
)
@click.option(
    "--output", "-o", "output_path", required=True,
    help="Output directory for quantized model"
)
@click.option(
    "--method", "-m", default="rtn",
    type=click.Choice(QUANT_METHODS, case_sensitive=False),
    help="Quantization method: rtn (round-to-nearest), gptq, or mr-gptq (Marlin-Replica GPTQ with Hadamard)"
)
@click.option(
    "--bits", "-b", default=4,
    type=click.Choice([2, 3, 4, 8], case_sensitive=False),
    help="Bit width for quantization"
)
@click.option(
    "--format", "-f", "quant_format", default="fp4",
    type=click.Choice(QUANT_FORMATS, case_sensitive=False),
    help="Quantization format (fp4=E2M1, int4=symmetric, nf4=NormalFloat)"
)
@click.option(
    "--group-size", "-g", default=128,
    type=click.Choice([32, 64, 128, 256], case_sensitive=False),
    help="Elements per quantization group"
)
@click.option(
    "--calibration", "-c", default=None,
    help="Calibration dataset: bartowski-v3, wikitext2, c4, or path to custom file"
)
@click.option(
    "--samples", "-s", default=None, type=int,
    help="Number of calibration samples (default: all available)"
)
@click.option(
    "--no-hadamard", is_flag=True, default=False,
    help="Disable Hadamard rotation (only applies to mr-gptq method)"
)
@click.option(
    "--actorder/--no-actorder", default=True,
    help="Enable activation-order quantization for GPTQ methods (default: enabled)"
)
@click.option(
    "--damp", default=0.01, type=float,
    help="Hessian damping factor for GPTQ methods (default: 0.01)"
)
@click.option(
    "--mixed-precision", "mixed_precision", default=None,
    type=click.Choice(["dense", "moe", "moe-mtp", "quality", "speed"]),
    help="Mixed-precision preset (auto-detected if not specified)"
)
@click.option(
    "--layerwise", is_flag=True, default=False,
    help="Use memory-efficient layer-wise conversion (for large models)"
)
@click.option(
    "--validate/--no-validate", default=True,
    help="Compute quantization error metrics (default: enabled)"
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False,
    help="Verbose output"
)
@click.option(
    "--token", default=None,
    help="HuggingFace API token for gated models"
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
    actorder: bool,
    damp: float,
    mixed_precision: str | None,
    layerwise: bool,
    validate: bool,
    verbose: bool,
    token: str | None,
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
        "fp4": 4, "int4": 4, "nf4": 4,
        "int3": 3, "int2": 2, "int8": 8,
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
        if method in ("gptq", "mr-gptq"):
            click.echo(f"  Act-order:    {'enabled' if actorder else 'disabled'}")
            click.echo(f"  Damp:         {damp}")
        if mixed_precision:
            click.echo(f"  Mixed prec:   {mixed_precision}")
        click.echo(f"  Layer-wise:   {'enabled' if layerwise else 'disabled'}")
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
            layerwise=layerwise,
            validate=validate,
            verbose=verbose,
            token=token,
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
            validate=validate,
            verbose=verbose,
            token=token,
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
            actorder=actorder,
            damp=damp,
            mixed_precision=mixed_precision,
            validate=validate,
            verbose=verbose,
            token=token,
        )


def _quantize_rtn(
    input_path: str,
    output_path: str,
    quant_format: str,
    group_size: int,
    calibration: str | None,
    samples: int | None,
    mixed_precision: str | None,
    layerwise: bool,
    validate: bool,
    verbose: bool,
    token: str | None,
):
    """RTN (Round-to-Nearest) quantization - existing implementation."""
    from .hf_loader import (
        CalibrationData,
        convert_model_layerwise,
        convert_model_to_fp4,
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
    if layerwise:
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
    validate: bool,
    verbose: bool,
    token: str | None,
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
    actorder: bool,
    damp: float,
    mixed_precision: str | None,
    validate: bool,
    verbose: bool,
    token: str | None,
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


@cli.command("eval")
@click.option(
    "--model", "-m", required=True,
    help="Path to quantized model"
)
@click.option(
    "--reference", "-r", default=None,
    help="Path to reference model for comparison (optional)"
)
@click.option(
    "--metric", default="perplexity",
    type=click.Choice(["perplexity", "kl-divergence", "all"]),
    help="Evaluation metric"
)
@click.option(
    "--dataset", "-d", default="wikitext2",
    type=click.Choice(["wikitext2", "c4"]),
    help="Evaluation dataset"
)
@click.option(
    "--samples", "-s", default=100, type=int,
    help="Number of evaluation samples"
)
@click.option(
    "--context-length", default=2048, type=int,
    help="Context window size for perplexity"
)
@click.option(
    "--verbose", "-v", is_flag=True,
    help="Verbose output"
)
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
    from .eval_perplexity import (
        load_tokenizer,
    )

    click.echo(f"Loading model: {model}")
    tokenizer = load_tokenizer(model)

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


def main():
    cli()


if __name__ == "__main__":
    main()
