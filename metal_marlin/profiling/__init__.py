"""Kernel timing profiler for Metal Marlin."""
import logging
from .kernel_timer import KernelTimer, get_timer


logger = logging.getLogger(__name__)

__all__ = ["KernelTimer", "get_timer"]
