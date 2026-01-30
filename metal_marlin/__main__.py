"""Allow running as `python -m metal_marlin`."""

from .cli import cli, serve

if "serve" not in cli.commands:
    cli.add_command(serve)

if __name__ == "__main__":
    cli()
