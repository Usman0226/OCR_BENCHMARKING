from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytesseract
from PIL import Image

from ocr.core.logger import get_logger
from ocr.core.models import BoundingBox, EngineType, NormalizedWord
from ocr.engines.base import OCREngine
from ocr.utils.image import load_image_pil, preprocess_for_tesseract

if TYPE_CHECKING:
    from ocr.core.config import AppConfig

logger = get_logger(__name__)

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class TesseractEngine(OCREngine):
    """OCR engine backed by Tesseract 5 via pytesseract."""

    engine_type = EngineType.TESSERACT

    def __init__(self, config: "AppConfig") -> None:
        super().__init__(config)

    def initialize(self) -> None:
        cfg = self._config.engines.tesseract

        if os.name == "nt":
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        try:
            version = pytesseract.get_tesseract_version()
            logger.info("Tesseract version: %s", version)
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "Tesseract binary not found. "
                "Ensure tesseract-ocr is installed in the Docker image."
            ) from exc

        available_langs = pytesseract.get_languages()
        requested = [lang.strip() for lang in cfg.lang.split("+")]
        missing = [lang for lang in requested if lang not in available_langs]
        if missing:
            raise RuntimeError(
                f"Tesseract language pack(s) not installed: {missing}. "
                f"Available: {available_langs}. "
                "Install via: apt-get install tesseract-ocr-<lang>"
            )

        self._initialized = True
        logger.info(
            "TesseractEngine initialized — lang=%s psm=%d oem=%d",
            cfg.lang,
            cfg.psm,
            cfg.oem,
        )

    def shutdown(self) -> None:
        self._initialized = False
        logger.info("TesseractEngine shut down.")

    def _run_raw(self, image_path: Path) -> list[NormalizedWord]:
        cfg = self._config.engines.tesseract

        pil_image = load_image_pil(image_path)
        if self._config.preprocessing.enabled:
            pil_image = preprocess_for_tesseract(
                pil_image,
                grayscale=self._config.preprocessing.tesseract_grayscale,
                max_long_edge=self._config.preprocessing.max_long_edge,
            )

        config_parts = [
            f"--psm {cfg.psm}",
            f"--oem {cfg.oem}",
            f"--dpi {cfg.dpi}",
        ]
        if cfg.config_extra:
            config_parts.append(cfg.config_extra)
        tess_config = " ".join(config_parts)

        data = pytesseract.image_to_data(
            pil_image,
            lang=cfg.lang,
            config=tess_config,
            output_type=pytesseract.Output.DICT,
        )

        words: list[NormalizedWord] = []
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            if data["level"][i] != 5:
                continue

            text = str(data["text"][i]).strip()
            if not text:
                continue

            text = _CONTROL_CHAR_RE.sub("", text)
            if not text:
                continue

            try:
                conf_raw = float(data["conf"][i])
            except (ValueError, TypeError):
                continue

            if conf_raw < 0:
                continue
            confidence = float(conf_raw) / 100.0

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            if w <= 0 or h <= 0:
                continue

            bbox = BoundingBox(
                x_min=float(x),
                y_min=float(y),
                x_max=float(x + w),
                y_max=float(y + h),
            )

            page_num = int(data["page_num"][i]) if "page_num" in data else 1
            page = max(1, page_num)

            word = NormalizedWord(
                text=text,
                bbox=bbox,
                confidence=min(max(confidence, 0.0), 1.0),
                page=page,
                image_name=image_path.name,
                engine=self.engine_type,
                polygon=None,
            )
            words.append(word)

        logger.debug(
            "TesseractEngine extracted %d words from '%s'.",
            len(words),
            image_path.name,
        )
        return words
