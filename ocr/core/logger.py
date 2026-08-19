from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ocr.core.config import LoggingConfig

APP_LOGGER = "ocr_benchmark"
OCR_LOGGER = "ocr_benchmark.ocr"
ERROR_LOGGER = "ocr_benchmark.error"


class _StructuredFormatter(logging.Formatter):
    FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATEFMT = "%Y-%m-%dT%H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FMT, datefmt=self.DATEFMT)


def configure_logging(cfg: "LoggingConfig", log_dir: Path) -> None:
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

    app_logger = logging.getLogger(APP_LOGGER)
    app_logger.setLevel(numeric_level)
    app_logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    app_logger.addHandler(console_handler)

    app_file = cfg.files.get("application", "application.log")
    app_logger.addHandler(_make_rotating_handler(app_file))

    ocr_logger = logging.getLogger(OCR_LOGGER)
    ocr_logger.setLevel(numeric_level)
    ocr_logger.propagate = True

    ocr_file = cfg.files.get("ocr", "ocr.log")
    ocr_file_handler = _make_rotating_handler(ocr_file)
    ocr_file_handler.setLevel(numeric_level)
    ocr_logger.addHandler(ocr_file_handler)

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
    if not name.startswith(APP_LOGGER):
        name = f"{APP_LOGGER}.{name}"
    return logging.getLogger(name)
