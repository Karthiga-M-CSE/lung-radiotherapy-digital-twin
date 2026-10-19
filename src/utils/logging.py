"""Centralized logging utility for the Lung Radiotherapy Digital Twin system."""

import logging
import sys
from typing import Optional
from src.config import LOG_LEVEL, LOG_FORMAT


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger with consistent formatting.

    Args:
        name: Name of the logger, typically __name__.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name or "lung_digital_twin")
    numeric_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
