from __future__ import annotations

import abc
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ocr.core.logger import get_logger
from ocr.core.models import EngineType, NormalizedWord, OCRResult

if TYPE_CHECKING:
    from ocr.core.config import AppConfig

logger = get_logger(__name__)


class OCREngine(abc.ABC):
    engine_type: EngineType

    def __init__(self, config: "AppConfig") -> None:
        self._config = config
        self._initialized = False

    @abc.abstractmethod
    def initialize(self) -> None:
        pass

    @abc.abstractmethod
    def shutdown(self) -> None:
        pass

    @abc.abstractmethod
    def _run_raw(self, image_path: Path) -> list[NormalizedWord]:
        pass

    def run(self, image_path: Path) -> OCRResult:
        if not self._initialized:
            raise RuntimeError(
                f"Engine '{self.engine_type.value}' has not been initialized. "
                "Call initialize() before run()."
            )

        image_name = image_path.name
        logger.info(
            "Running %s on '%s'", self.engine_type.value, image_name
        )

        start = time.perf_counter()
        try:
            words = self._run_raw(image_path)
            elapsed = time.perf_counter() - start
            logger.info(
                "%s processed '%s' — %d words in %.2fs",
                self.engine_type.value,
                image_name,
                len(words),
                elapsed,
            )
            return OCRResult(
                image_name=image_name,
                image_path=image_path,
                engine=self.engine_type,
                words=words,
                processing_time_s=elapsed,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error(
                "%s failed on '%s' after %.2fs: %s",
                self.engine_type.value,
                image_name,
                elapsed,
                exc,
                exc_info=True,
            )
            return OCRResult(
                image_name=image_name,
                image_path=image_path,
                engine=self.engine_type,
                words=[],
                processing_time_s=elapsed,
                error=str(exc),
            )

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return f"{self.__class__.__name__}(engine={self.engine_type.value}, {status})"


def create_engine(engine_type: EngineType, config: "AppConfig") -> OCREngine:
    if engine_type == EngineType.PADDLE:
        from ocr.engines.paddle_engine import PaddleEngine
        return PaddleEngine(config)
    if engine_type == EngineType.TESSERACT:
        from ocr.engines.tesseract_engine import TesseractEngine
        return TesseractEngine(config)
    raise ValueError(
        f"No engine implementation registered for '{engine_type.value}'."
    )
