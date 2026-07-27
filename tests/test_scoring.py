"""Unit tests for ocr.core.scoring."""

from __future__ import annotations

from pathlib import Path

import pytest

from ocr.core.models import (
    AnnotationWord,
    BoundingBox,
    EngineType,
    ImageAnnotation,
    NormalizedWord,
    OCRResult,
)
from ocr.core.scoring import _safe_divide, score_all, score_image


# =============================================================================
# Helpers
# =============================================================================


def _make_matching_cfg(
    strategy: str = "iou_and_text",
    iou_threshold: float = 0.5,
    text_threshold: float = 0.8,
) -> "object":
    class _Cfg:
        def __init__(self) -> None:
            self.strategy = strategy
            self.iou_threshold = iou_threshold
            self.text_similarity_threshold = text_threshold

    return _Cfg()


def _ocr_result(
    words: list[NormalizedWord],
    image_name: str = "test.jpg",
    engine: EngineType = EngineType.TESSERACT,
    error: str | None = None,
) -> OCRResult:
    return OCRResult(
        image_name=image_name,
        image_path=Path(f"/workspace/images/{image_name}"),
        engine=engine,
        words=words,
        error=error,
    )


def _annotation(
    image_name: str, words: list[AnnotationWord]
) -> ImageAnnotation:
    return ImageAnnotation(image_name=image_name, words=words)


def _word(
    text: str,
    x: float = 0.0,
    y: float = 0.0,
    w: float = 100.0,
    h: float = 30.0,
    engine: EngineType = EngineType.TESSERACT,
) -> NormalizedWord:
    return NormalizedWord(
        text=text,
        bbox=BoundingBox(x, y, x + w, y + h),
        confidence=0.9,
        page=1,
        image_name="test.jpg",
        engine=engine,
    )


def _ann(
    text: str,
    x: float = 0.0,
    y: float = 0.0,
    w: float = 100.0,
    h: float = 30.0,
) -> AnnotationWord:
    return AnnotationWord(
        text=text,
        bbox=BoundingBox(x, y, x + w, y + h),
        label="text",
        annotation_id="a",
        image_name="test.jpg",
    )


# =============================================================================
# _safe_divide
# =============================================================================


class TestSafeDivide:
    def test_normal(self) -> None:
        assert _safe_divide(3, 4) == pytest.approx(0.75)

    def test_zero_denominator(self) -> None:
        assert _safe_divide(5, 0) == 0.0

    def test_zero_numerator(self) -> None:
        assert _safe_divide(0, 4) == 0.0


# =============================================================================
# score_image
# =============================================================================


class TestScoreImage:
    def test_perfect_match(self) -> None:
        """All OCR words match annotations → P=R=F1=1."""
        ocr = _ocr_result([_word("Hello"), _word("World", x=110)])
        ann = _annotation(
            "test.jpg", [_ann("Hello"), _ann("World", x=110)]
        )
        result = score_image(ocr, ann, _make_matching_cfg())
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)
        assert result.f1 == pytest.approx(1.0)
        assert result.true_positives == 2
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_no_ocr_output(self) -> None:
        """Empty OCR → P=1.0 (vacuously) R=0 F1=0."""
        ocr = _ocr_result([])
        ann = _annotation("test.jpg", [_ann("Hello"), _ann("World", x=110)])
        result = score_image(ocr, ann, _make_matching_cfg())
        assert result.recall == pytest.approx(0.0)
        assert result.f1 == pytest.approx(0.0)
        assert result.false_negatives == 2
        assert len(result.missed_words) == 2

    def test_no_annotations(self) -> None:
        """Empty annotation → R=1.0 (vacuously) P=0 F1=0."""
        ocr = _ocr_result([_word("Ghost")])
        ann = _annotation("test.jpg", [])
        result = score_image(ocr, ann, _make_matching_cfg())
        assert result.precision == pytest.approx(0.0)
        assert result.f1 == pytest.approx(0.0)
        assert result.false_positives == 1
        assert len(result.invented_words) == 1
        assert "Ghost" in result.invented_words

    def test_partial_match(self) -> None:
        """One match, one miss, one invented."""
        ocr = _ocr_result([_word("Hello"), _word("Ghost", x=200)])
        ann = _annotation("test.jpg", [_ann("Hello"), _ann("Missed", x=400)])
        result = score_image(ocr, ann, _make_matching_cfg())
        assert result.true_positives == 1
        assert result.false_positives == 1
        assert result.false_negatives == 1
        assert "Missed" in result.missed_words
        assert "Ghost" in result.invented_words

    def test_ocr_error_returns_zero_score(self) -> None:
        """An OCR result with an error returns all-zero metrics."""
        ocr = _ocr_result([], error="Engine crashed")
        ann = _annotation("test.jpg", [_ann("Hello")])
        result = score_image(ocr, ann, _make_matching_cfg())
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0
        assert result.error == "Engine crashed"

    def test_coverage(self) -> None:
        """Coverage = TP / total annotation words."""
        ocr = _ocr_result([_word("Hello"), _word("World", x=110)])
        ann = _annotation(
            "test.jpg",
            [_ann("Hello"), _ann("World", x=110), _ann("Missing", x=300)],
        )
        result = score_image(ocr, ann, _make_matching_cfg())
        # 2 out of 3 annotation words matched
        assert result.coverage == pytest.approx(2 / 3, rel=1e-4)


# =============================================================================
# score_all
# =============================================================================


class TestScoreAll:
    def test_aggregate_metrics(self) -> None:
        ocr_results = [
            _ocr_result([_word("A")], image_name="img1.jpg"),
            _ocr_result([_word("B", x=0), _word("C", x=110)], image_name="img2.jpg"),
        ]
        annotations = {
            "img1.jpg": _annotation("img1.jpg", [_ann("A")]),
            "img2.jpg": _annotation(
                "img2.jpg", [_ann("B", x=0), _ann("C", x=110)]
            ),
        }
        cfg = _make_matching_cfg()
        result = score_all(
            ocr_results, annotations, cfg, engine=EngineType.TESSERACT
        )
        assert result.image_count == 2
        assert result.mean_f1 == pytest.approx(1.0)

    def test_missing_annotation_skipped(self) -> None:
        ocr_results = [
            _ocr_result([_word("A")], image_name="orphan.jpg"),
        ]
        annotations: dict = {}
        cfg = _make_matching_cfg()
        result = score_all(
            ocr_results, annotations, cfg, engine=EngineType.TESSERACT
        )
        assert result.image_count == 0

    def test_empty_results(self) -> None:
        result = score_all([], {}, _make_matching_cfg(), engine=EngineType.PADDLE)
        assert result.image_count == 0
        assert result.mean_f1 == 0.0
