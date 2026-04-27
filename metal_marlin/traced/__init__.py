"""MPSGraph traced execution for fused inference."""
import logging

from .mpsgraph_layer import TracedDecoderLayer


logger = logging.getLogger(__name__)

__all__ = ["TracedDecoderLayer"]
