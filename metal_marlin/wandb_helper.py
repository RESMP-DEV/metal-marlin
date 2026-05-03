"""Optional Weights & Biases tracking helper for metal_marlin.

Provides a safe wrapper around wandb that becomes a no-op if the library
is not installed or tracking is disabled.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False

class WandbHelper:
    def __init__(self) -> None:
        self.enabled = False
        self.run = None

    def init(self, project: str | None = None, name: str | None = None, config: dict[str, Any] | None = None, tags: list[str] | None = None, **kwargs: Any) -> Any:
        if not HAS_WANDB:
            logger.warning("wandb is not installed. Tracking is disabled.")
            return None

        # Check for API key in env
        api_key = os.environ.get("WB_API_KEY") or os.environ.get("WANDB_API_KEY")
        if not api_key:
            logger.warning("No WANDB_API_KEY or WB_API_KEY found in environment. Tracking is disabled.")
            return None

        # Proceed with init
        try:
            if "WANDB_API_KEY" not in os.environ and "WB_API_KEY" in os.environ:
                os.environ["WANDB_API_KEY"] = os.environ["WB_API_KEY"]
                
            self.enabled = True
            self.run = wandb.init(project=project, name=name, config=config, tags=tags, **kwargs)
            return self.run
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            self.enabled = False
            return None

    def log(self, data: dict[str, Any], **kwargs: Any) -> None:
        if self.enabled and wandb is not None:
            try:
                wandb.log(data, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")

    def finish(self) -> None:
        if self.enabled and wandb is not None:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to finish wandb run: {e}")
            finally:
                self.enabled = False
                self.run = None

# Global instance
wandb_tracker = WandbHelper()
