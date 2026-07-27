"""Tesseract OCR engine implementation.

Uses pytesseract (Python wrapper for Tesseract 5).
Supports French + English (fra+eng) by default, configurable via config.yaml.
"""

from __future__ import annotations

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

# Regex to strip Tesseract control characters from output
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class TesseractEngine(OCREngine):
    """OCR engine backed by Tesseract 5 via pytesseract.

    Configuration keys (all from config.yaml → engines.tesseract):
      lang, psm, oem, config_extra, dpi
    """

    engine_type = EngineType.TESSERACT

    def __init__(self, config: "AppConfig") -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Verify Tesseract is installed and the requested languages are available."""
        cfg = self._config.engines.tesseract

        try:
            version = pytesseract.get_tesseract_version()
            logger.info("Tesseract version: %s", version)
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "Tesseract binary not found. "
                "Ensure tesseract-ocr is installed in the Docker image."
            ) from exc

        # Validate that all requested language packs are present
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
        """No persistent resources to release for Tesseract."""
        self._initialized = False
        logger.info("TesseractEngine shut down.")

    # ------------------------------------------------------------------
    # Raw run
    # ------------------------------------------------------------------

    def _run_raw(self, image_path: Path) -> list[NormalizedWord]:
        """Run Tesseract on *image_path* and return normalized words."""
        cfg = self._config.engines.tesseract

        pil_image = load_image_pil(image_path)
        if self._config.preprocessing.enabled:
            pil_image = preprocess_for_tesseract(
                pil_image,
                grayscale=self._config.preprocessing.tesseract_grayscale,
                max_long_edge=self._config.preprocessing.max_long_edge,
            )

        # Build tesseract custom config string
        config_parts = [
            f"--psm {cfg.psm}",
            f"--oem {cfg.oem}",
            f"--dpi {cfg.dpi}",
        ]
        if cfg.config_extra:
            config_parts.append(cfg.config_extra)
        tess_config = " ".join(config_parts)

        # Run Tesseract and get word-level bounding boxes
        data = pytesseract.image_to_data(
            pil_image,
            lang=cfg.lang,
            config=tess_config,
            output_type=pytesseract.Output.DICT,
        )

        words: list[NormalizedWord] = []
        n_boxes = len(data["level"])
        img_w, img_h = pil_image.size

        for i in range(n_boxes):
            # level 5 = word
            if data["level"][i] != 5:
                continue

            text: str = str(data["text"][i]).strip()
            if not text:
                continue

            # Clean control characters
            text = _CONTROL_CHAR_RE.sub("", text)
            if not text:
                continue

            # Tesseract confidence is 0–100; normalize to 0–1
            conf_raw = data["conf"][i]
            if conf_raw < 0:
                # -1 means Tesseract could not compute confidence — skip
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

            # Page number (Tesseract page index is 0-based in image_to_data)
            page_num = int(data["page_num"][i]) if "page_num" in data else 1
            page = max(1, page_num)

            word = NormalizedWord(
                text=text,
                bbox=bbox,
                confidence=min(max(confidence, 0.0), 1.0),
                page=page,
                image_name=image_path.name,
                engine=self.engine_type,
                polygon=None,  # Tesseract doesn't produce quadrilaterals at word level
            )
            words.append(word)

        logger.debug(
            "TesseractEngine extracted %d words from '%s'.",
            len(words),
            image_path.name,
        )
        return words
