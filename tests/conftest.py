from __future__ import annotations

import pytest

from ocr.core.models import (
    AnnotationWord,
    BoundingBox,
    EngineType,
    NormalizedWord,
)


@pytest.fixture()
def unit_box() -> BoundingBox:
    return BoundingBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)


@pytest.fixture()
def overlapping_box() -> BoundingBox:
    return BoundingBox(x_min=0.5, y_min=0.5, x_max=1.5, y_max=1.5)


@pytest.fixture()
def non_overlapping_box() -> BoundingBox:
    return BoundingBox(x_min=2.0, y_min=2.0, x_max=3.0, y_max=3.0)


@pytest.fixture()
def contained_box() -> BoundingBox:
    return BoundingBox(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8)


@pytest.fixture()
def ocr_word_hello() -> NormalizedWord:
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
    return AnnotationWord(
        text="Hello",
        bbox=BoundingBox(x_min=10.0, y_min=20.0, x_max=60.0, y_max=40.0),
        label="text",
        annotation_id="ann-001",
        image_name="test.jpg",
    )


@pytest.fixture()
def annotation_word_world() -> AnnotationWord:
    return AnnotationWord(
        text="World",
        bbox=BoundingBox(x_min=70.0, y_min=20.0, x_max=120.0, y_max=40.0),
        label="text",
        annotation_id="ann-002",
        image_name="test.jpg",
    )


@pytest.fixture()
def annotation_word_missing() -> AnnotationWord:
    return AnnotationWord(
        text="Missed",
        bbox=BoundingBox(x_min=200.0, y_min=200.0, x_max=260.0, y_max=220.0),
        label="text",
        annotation_id="ann-003",
        image_name="test.jpg",
    )
