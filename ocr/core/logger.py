"""Structured logging setup for the OCR Benchmark Framework.

Creates three rotating log files:
  - application.log  (all levels ≥ configured threshold)
  - ocr.log          (OCR-specific events: engine init, run, output)
  - errors.log       (WARNING and above only)

All loggers emit structured records compatible with log aggregation tools.
Call configure_logging() once at application startup.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ocr.core.config import LoggingConfig


# =============================================================================
# Logger names
# =============================================================================

APP_LOGGER = "ocr_benchmark"
OCR_LOGGER = "ocr_benchmark.ocr"
ERROR_LOGGER = "ocr_benchmark.error"


# =============================================================================
# Formatter
# =============================================================================


class _StructuredFormatter(logging.Formatter):
    """Compact structured formatter: timestamp | level | logger | message."""

    FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATEFMT = "%Y-%m-%dT%H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FMT, datefmt=self.DATEFMT)


# =============================================================================
# Public API
# =============================================================================


def configure_logging(cfg: "LoggingConfig", log_dir: Path) -> None:
    """Configure all application loggers.

    Must be called **once** at application startup before any logger is used.

    Args:
        cfg: LoggingConfig from the loaded AppConfig.
        log_dir: Directory where log files will be written.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, cfg.level.upper(), logging.INFO)
    formatter = _StructuredFormatter()

    def _make_rotating_handler(filename: str) -> logging.handlers.RotatingFileHandler:
        path = log_dir / filename
        handler = logging.handlers.RotatingFileHandler(
            filename=path,
            maxBytes=cfg.max_bytes,
            backupCount=cfg.backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(formatter)
        return handler

    # --- Root application logger ---
    app_logger = logging.getLogger(APP_LOGGER)
    app_logger.setLevel(numeric_level)
    app_logger.propagate = False

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    app_logger.addHandler(console_handler)

    # application.log
    app_file = cfg.files.get("application", "application.log")
    app_logger.addHandler(_make_rotating_handler(app_file))

    # --- OCR-specific logger ---
    ocr_logger = logging.getLogger(OCR_LOGGER)
    ocr_logger.setLevel(numeric_level)
    ocr_logger.propagate = True  # Propagates to app_logger

    ocr_file = cfg.files.get("ocr", "ocr.log")
    ocr_file_handler = _make_rotating_handler(ocr_file)
    ocr_file_handler.setLevel(numeric_level)
    ocr_logger.addHandler(ocr_file_handler)

    # --- Error logger ---
    err_logger = logging.getLogger(ERROR_LOGGER)
    err_logger.setLevel(logging.WARNING)
    err_logger.propagate = True

    err_file = cfg.files.get("errors", "errors.log")
    err_file_handler = _make_rotating_handler(err_file)
    err_file_handler.setLevel(logging.WARNING)
    err_logger.addHandler(err_file_handler)

    app_logger.info(
        "Logging configured — level=%s dir=%s", cfg.level.upper(), log_dir
    )


def get_logger(name: str = APP_LOGGER) -> logging.Logger:
    """Return a named logger under the ocr_benchmark hierarchy.

    Usage::

        from ocr.core.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Processing image %s", image_name)

    Args:
        name: Logger name. If not already prefixed with 'ocr_benchmark',
              it will be automatically prefixed.

    Returns:
        A configured Logger instance.
    """
    if not name.startswith(APP_LOGGER):
        name = f"{APP_LOGGER}.{name}"
    return logging.getLogger(name)
