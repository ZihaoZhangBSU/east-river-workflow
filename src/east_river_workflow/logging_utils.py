"""Logging setup."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(output_dir: Path, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("east_river_workflow")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(output_dir / "logs" / "workflow.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
