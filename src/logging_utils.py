from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from .config import DEBUG, ROOT


def get_logger(name: str = "kb") -> logging.Logger:
    """Return a configured logger that logs to console and a rotating file."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if DEBUG else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "app.log", maxBytes=2_000_000, backupCount=3
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
