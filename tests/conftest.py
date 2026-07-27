"""Shared pytest fixtures for the OCR Benchmark test suite."""

from __future__ import annotations

import pytest

from ocr.core.models import (
    AnnotationWord,
    BoundingBox,
    EngineType,
    NormalizedWord,
    Polygon,
)


# =============================================================================
# BoundingBox fixtures
# =============================================================================


@pytest.fixture()
def unit_box() -> BoundingBox:
    """A 1×1 box at the origin."""
    return BoundingBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)


@pytest.fixture()
def overlapping_box() -> BoundingBox:
    """A box that half-overlaps unit_box."""
    return BoundingBox(x_min=0.5, y_min=0.5, x_max=1.5, y_max=1.5)


@pytest.fixture()
def non_overlapping_box() -> BoundingBox:
    """A box that does not overlap unit_box."""
    return BoundingBox(x_min=2.0, y_min=2.0, x_max=3.0, y_max=3.0)


@pytest.fixture()
def contained_box() -> BoundingBox:
    """A box fully inside unit_box."""
    return BoundingBox(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8)


# =============================================================================
# NormalizedWord fixtures
# =============================================================================


@pytest.fixture()
def ocr_word_hello() -> NormalizedWord:
    """OCR word 'Hello' with a known bbox."""
    return NormalizedWord(
        text="Hello",
        bbox=BoundingBox(x_min=10.0, y_min=20.0, x_max=60.0, y_max=40.0),
        confidence=0.95,
        page=1,
        image_name="test.jpg",
        engine=EngineType.TESSERACT,
    )


@pytest.fixture()
def ocr_word_world() -> NormalizedWord:
    """OCR word 'World' adjacent to 'Hello'."""
    return NormalizedWord(
        text="World",
        bbox=BoundingBox(x_min=70.0, y_min=20.0, x_max=120.0, y_max=40.0),
        confidence=0.88,
        page=1,
        image_name="test.jpg",
        engine=EngineType.TESSERACT,
    )


@pytest.fixture()
def annotation_word_hello() -> AnnotationWord:
    """Ground-truth annotation for 'Hello' at the same position."""
    return AnnotationWord(
        text="Hello",
        bbox=BoundingBox(x_min=10.0, y_min=20.0, x_max=60.0, y_max=40.0),
        label="text",
        annotation_id="ann-001",
        image_name="test.jpg",
    )


@pytest.fixture()
def annotation_word_world() -> AnnotationWord:
    """Ground-truth annotation for 'World'."""
    return AnnotationWord(
        text="World",
        bbox=BoundingBox(x_min=70.0, y_min=20.0, x_max=120.0, y_max=40.0),
        label="text",
        annotation_id="ann-002",
        image_name="test.jpg",
    )


@pytest.fixture()
def annotation_word_missing() -> AnnotationWord:
    """An annotated word that has no corresponding OCR detection."""
    return AnnotationWord(
        text="Missed",
        bbox=BoundingBox(x_min=200.0, y_min=200.0, x_max=260.0, y_max=220.0),
        label="text",
        annotation_id="ann-003",
        image_name="test.jpg",
    )
