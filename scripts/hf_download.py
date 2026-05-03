#!/usr/bin/env python3
"""
Model downloader for Metal Marlin Trellis models.

Downloads Hugging Face models and caches them locally for quantization.
Standalone script - no AlphaHENG dependencies.
"""

import argparse
import logging
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("ERROR: huggingface_hub not installed. Install with:")
    print("  pip install huggingface_hub")
    sys.exit(1)



logger = logging.getLogger(__name__)

# Common Trellis model mappings
TRELLIS_MODELS = {
    "trellis-1b": "microsoft/trellis-1b",
    "trellis-3b": "microsoft/trellis-3b",
    "trellis-7b": "microsoft/trellis-7b",
    "trellis-13b": "microsoft/trellis-13b",
    "trellis-70b": "microsoft/trellis-70b",
}

# Common model architectures
MODEL_PRESETS = {
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-13b": "meta-llama/Llama-2-13b-hf",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-v0.1",
    "qwen-7b": "Qwen/Qwen-7B",
    "qwen3.5-35b-a3b": "Qwen/Qwen3.5-35B-A3B",
    "qwen3.6-35b-a3b": "Qwen/Qwen3.6-35B-A3B",
    "qwen3-coder-next": "Qwen/Qwen3-Coder-Next",
    "qwen3.6-35b-a3b-fp8": "Qwen/Qwen3.6-35B-A3B-FP8",
    "deepseek-7b": "deepseek-ai/deepseek-llm-7b-base",
}

ALL_PRESETS = {**TRELLIS_MODELS, **MODEL_PRESETS}


def get_cache_dir():
    """Get default cache directory."""
    # Use HF_HOME if set, otherwise ~/.cache/huggingface
    logger.debug("get_cache_dir called")
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    return cache_dir


def download_model(
    model_id: str,
    cache_dir: Path | None = None,
    token: str | None = None,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    force: bool = False,
) -> Path:
    """
    Download a Hugging Face model.

    Args:
        model_id: HuggingFace model ID (e.g., "microsoft/trellis-7b")
        cache_dir: Local cache directory (default: ~/.cache/huggingface/hub)
        token: HuggingFace API token for private models
        allow_patterns: Patterns to include (e.g., ["*.safetensors", "config.json"])
        ignore_patterns: Patterns to exclude (e.g., ["*.bin"])
        force: Force re-download even if cached

    Returns:
        Path to downloaded model directory
    """
    logger.info("download_model called with model_id=%s, cache_dir=%s, token=%s, allow_patterns=%s", model_id, cache_dir, token, allow_patterns)
    if cache_dir is None:
        cache_dir = get_cache_dir()

    # Default patterns: skip large checkpoints, only get model weights + config
    if allow_patterns is None:
        allow_patterns = [
            "*.safetensors",
            "*.json",
            "tokenizer.model",
            "*.tiktoken",
        ]

    if ignore_patterns is None:
        ignore_patterns = [
            "*.bin",  # Skip PyTorch checkpoints (use safetensors)
            "*.h5",
            "*.msgpack",
            "*.ot",
        ]

    print(f"📥 Downloading {model_id}")
    print(f"📁 Cache: {cache_dir}")

    try:
        local_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            token=token,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            force_download=force,
            resume_download=True,
        )
        print(f"✅ Downloaded to: {local_path}")
        return Path(local_path)

    except HfHubHTTPError as e:
        if e.response.status_code == 401:
            print("❌ Authentication failed. Provide a valid HuggingFace token:")
            print("   --token YOUR_TOKEN")
            print("   or set HF_TOKEN environment variable")
        elif e.response.status_code == 404:
            print(f"❌ Model not found: {model_id}")
            print("   Check the model ID is correct and accessible")
        else:
            print(f"❌ HTTP error: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)


def list_presets():
    """List available model presets."""
    logger.debug("list_presets called")
    print("📋 Available model presets:")
    print("\nTrellis models:")
    for name, repo in TRELLIS_MODELS.items():
        print(f"  {name:20s} → {repo}")

    print("\nOther models:")
    for name, repo in MODEL_PRESETS.items():
        print(f"  {name:20s} → {repo}")

    print("\nUsage: hf_download.py --preset <name>")


def download_file(
    repo_id: str,
    filename: str,
    cache_dir: Path | None = None,
    token: str | None = None,
) -> Path:
    """Download a single file from a repository."""
    logger.info("download_file called with repo_id=%s, filename=%s, cache_dir=%s, token=%s", repo_id, filename, cache_dir, token)
    if cache_dir is None:
        cache_dir = get_cache_dir()

    print(f"📥 Downloading {filename} from {repo_id}")

    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            token=token,
        )
        print(f"✅ Downloaded to: {local_path}")
        return Path(local_path)

    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)


def main():
    logger.info("main starting")
    parser = argparse.ArgumentParser(
        description="Download HuggingFace models for Metal Marlin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download by preset name
  %(prog)s --preset trellis-7b

  # Download by repo ID
  %(prog)s --model microsoft/trellis-7b

  # Download with custom cache location
  %(prog)s --preset trellis-7b --cache ./models

  # Download single file
  %(prog)s --model microsoft/trellis-7b --file config.json

  # List available presets
  %(prog)s --list

Lightweight investigation pull (Qwen DeltaNet-family models):
  # Inspect model config before full download
  %(prog)s --preset qwen3.6-35b-a3b --file config.json
  %(prog)s --preset qwen3.6-35b-a3b --file tokenizer.json

  # Lightweight pull: config + tokenizer only (no weights)
  %(prog)s --preset qwen3.6-35b-a3b --allow config.json tokenizer.json tokenizer_config.json

  # Qwen DeltaNet-family investigation pattern
  %(prog)s --preset qwen3-coder-next --allow config.json tokenizer*.json qwen3*.json
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace model ID (e.g., microsoft/trellis-7b)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(ALL_PRESETS.keys()),
        help="Use a preset model name",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Download a single file instead of entire model",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        help=f"Cache directory (default: {get_cache_dir()})",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--allow",
        type=str,
        nargs="+",
        help="File patterns to include (e.g., *.safetensors *.json)",
    )
    parser.add_argument(
        "--ignore",
        type=str,
        nargs="+",
        help="File patterns to exclude (e.g., *.bin)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available model presets",
    )

    args = parser.parse_args()

    # List presets
    if args.list:
        list_presets()
        return

    # Get model ID from preset or direct
    model_id = None
    if args.preset:
        model_id = ALL_PRESETS[args.preset]
        print(f"Using preset: {args.preset} → {model_id}")
    elif args.model:
        model_id = args.model
    else:
        parser.error("Must specify --model or --preset (or --list to see options)")

    # Download single file or entire model
    if args.file:
        download_file(
            repo_id=model_id,
            filename=args.file,
            cache_dir=args.cache,
            token=args.token,
        )
    else:
        download_model(
            model_id=model_id,
            cache_dir=args.cache,
            token=args.token,
            allow_patterns=args.allow,
            ignore_patterns=args.ignore,
            force=args.force,
        )


if __name__ == "__main__":
    main()
