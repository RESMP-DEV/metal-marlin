"""Allow running as `python -m metal_marlin`."""
import logging

from .cli import cli, serve


logger = logging.getLogger(__name__)

if "serve" not in cli.commands:
    cli.add_command(serve)

if __name__ == "__main__":
    cli()
