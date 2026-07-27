"""PaddleOCR engine implementation.

CPU-only. Models are downloaded on first run and cached in PADDLE_MODEL_DIR.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ocr.core.logger import get_logger
from ocr.core.models import BoundingBox, EngineType, NormalizedWord, Polygon
from ocr.engines.base import OCREngine
from ocr.utils.image import load_image_rgb

if TYPE_CHECKING:
    from ocr.core.config import AppConfig

logger = get_logger(__name__)


class PaddleEngine(OCREngine):
    """OCR engine backed by PaddleOCR (CPU-only).

    Configuration keys (all from config.yaml → engines.paddle):
      lang, use_angle_cls, use_gpu, det_db_thresh, det_db_box_thresh,
      rec_batch_num, enable_mkldnn, show_log, cls_thresh
    """

    engine_type = EngineType.PADDLE

    def __init__(self, config: "AppConfig") -> None:
        super().__init__(config)
        self._paddle: Any = None  # paddleocr.PaddleOCR instance

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Load PaddleOCR model.

        Models are stored in PADDLE_MODEL_DIR to avoid re-downloading
        on every container restart.
        """
        cfg = self._config.engines.paddle
        model_dir = self._config.paths.paddle_model_dir
        model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initializing PaddleOCR — lang=%s use_angle_cls=%s model_dir=%s",
            cfg.lang,
            cfg.use_angle_cls,
            model_dir,
        )

        try:
            from paddleocr import PaddleOCR  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. "
                "Ensure paddleocr is in requirements.txt and the Docker image is rebuilt."
            ) from exc

        self._paddle = PaddleOCR(
            lang=cfg.lang,
            use_angle_cls=cfg.use_angle_cls,
            use_gpu=cfg.use_gpu,
            det_db_thresh=cfg.det_db_thresh,
            det_db_box_thresh=cfg.det_db_box_thresh,
            rec_batch_num=cfg.rec_batch_num,
            enable_mkldnn=cfg.enable_mkldnn,
            show_log=cfg.show_log,
            cls_thresh=cfg.cls_thresh,
        )
        self._initialized = True
        logger.info("PaddleOCR initialized successfully.")

    def shutdown(self) -> None:
        """Release PaddleOCR resources."""
        self._paddle = None
        self._initialized = False
        logger.info("PaddleOCR engine shut down.")

    # ------------------------------------------------------------------
    # Raw run
    # ------------------------------------------------------------------

    def _run_raw(self, image_path: Path) -> list[NormalizedWord]:
        """Run PaddleOCR on *image_path* and return normalized words."""
        # PaddleOCR accepts file paths directly
        raw_result = self._paddle.ocr(str(image_path), cls=True)

        words: list[NormalizedWord] = []

        if not raw_result or raw_result[0] is None:
            logger.debug("PaddleOCR returned no results for '%s'.", image_path.name)
            return words

        # PaddleOCR output structure (per page, per line):
        # [ [ [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence) ], ... ]
        for page_idx, page in enumerate(raw_result):
            if page is None:
                continue
            for line in page:
                if line is None or len(line) < 2:
                    continue
                quad_points: list[list[float]] = line[0]
                text_conf: tuple[str, float] = line[1]

                text = str(text_conf[0])
                confidence = float(text_conf[1])

                polygon = Polygon.from_list(quad_points)
                bbox = polygon.to_bounding_box()

                word = NormalizedWord(
                    text=text,
                    bbox=bbox,
                    confidence=min(max(confidence, 0.0), 1.0),
                    page=page_idx + 1,
                    image_name=image_path.name,
                    engine=self.engine_type,
                    polygon=polygon,
                )
                words.append(word)

        logger.debug(
            "PaddleOCR extracted %d words from '%s'.", len(words), image_path.name
        )
        return words
