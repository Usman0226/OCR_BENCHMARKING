"""Abstract base class for all OCR engines.

To add a new engine:
  1. Create a new file: ocr/engines/my_engine.py
  2. Subclass OCREngine
  3. Implement all abstract methods
  4. Register the engine name in EngineType (ocr/core/models.py)
  5. Update the engine factory in this file

No other files need modification.
"""

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
    """Strategy interface for OCR engines.

    Every concrete engine must implement all abstract methods.
    The engine lifecycle is:

        engine = SomeEngine(config)
        engine.initialize()          # load models, warm up
        try:
            result = engine.run(image_path)
        finally:
            engine.shutdown()        # release resources
    """

    engine_type: EngineType  # Must be set on concrete subclass

    def __init__(self, config: "AppConfig") -> None:
        self._config = config
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle methods
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def initialize(self) -> None:
        """Load models and warm up the engine.

        Called once before the first ``run()`` call.
        Must set ``self._initialized = True`` on success.
        """

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release all resources held by the engine.

        Called after all images have been processed.
        """

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _run_raw(self, image_path: Path) -> list[NormalizedWord]:
        """Run OCR on *image_path* and return normalized words.

        This is the engine-specific implementation.  ``run()`` wraps it
        with timing, error handling, and logging.

        Args:
            image_path: Absolute path to the input image.

        Returns:
            List of NormalizedWord (never None).
        """

    def run(self, image_path: Path) -> OCRResult:
        """Public entry point: run OCR on *image_path*.

        Wraps ``_run_raw()`` with:
          - Pre-condition check (engine initialized)
          - Wall-clock timing
          - Structured logging
          - Exception capture into OCRResult.error

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with words or error message.
        """
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

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return f"{self.__class__.__name__}(engine={self.engine_type.value}, {status})"


# =============================================================================
# Engine factory
# =============================================================================


def create_engine(engine_type: EngineType, config: "AppConfig") -> OCREngine:
    """Instantiate the correct engine for *engine_type*.

    Import is deferred to avoid loading heavy libraries (PaddleOCR, etc.)
    until they are actually needed.

    Args:
        engine_type: Which engine to create.
        config: Full application configuration.

    Returns:
        An uninitialized OCREngine subclass instance.

    Raises:
        ValueError: If the engine type is unknown.
    """
    if engine_type == EngineType.PADDLE:
        from ocr.engines.paddle_engine import PaddleEngine
        return PaddleEngine(config)
    if engine_type == EngineType.TESSERACT:
        from ocr.engines.tesseract_engine import TesseractEngine
        return TesseractEngine(config)
    raise ValueError(
        f"No engine implementation registered for '{engine_type.value}'. "
        "Add it to ocr/engines/ and register it in create_engine()."
    )
