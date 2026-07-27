"""Unit tests for ocr.core.models — serialization and validation."""

from __future__ import annotations

import pytest

from ocr.core.models import (
    BoundingBox,
    EngineType,
    NormalizedWord,
    Polygon,
)


# =============================================================================
# BoundingBox
# =============================================================================


class TestBoundingBox:
    def test_valid_construction(self) -> None:
        box = BoundingBox(0, 0, 10, 20)
        assert box.width == 10
        assert box.height == 20
        assert box.area == 200

    def test_zero_area(self) -> None:
        box = BoundingBox(5, 5, 5, 5)
        assert box.area == 0.0

    def test_invalid_x(self) -> None:
        with pytest.raises(ValueError, match="x_min"):
            BoundingBox(x_min=10, y_min=0, x_max=5, y_max=10)

    def test_invalid_y(self) -> None:
        with pytest.raises(ValueError, match="y_min"):
            BoundingBox(x_min=0, y_min=15, x_max=10, y_max=5)

    def test_center(self) -> None:
        box = BoundingBox(0, 0, 10, 10)
        assert box.center == (5.0, 5.0)

    def test_to_dict_roundtrip(self) -> None:
        box = BoundingBox(1.5, 2.5, 10.5, 20.5)
        d = box.to_dict()
        assert d["x_min"] == 1.5
        assert d["x_max"] == 10.5

    def test_from_points(self) -> None:
        points = [[0, 5], [10, 0], [10, 10], [0, 10]]
        box = BoundingBox.from_points(points)
        assert box.x_min == 0
        assert box.y_min == 0
        assert box.x_max == 10
        assert box.y_max == 10

    def test_to_xywh(self) -> None:
        box = BoundingBox(10, 20, 110, 70)
        assert box.to_xywh() == (10, 20, 100, 50)


# =============================================================================
# EngineType
# =============================================================================


class TestEngineType:
    def test_from_str_paddle(self) -> None:
        assert EngineType.from_str("paddle") == EngineType.PADDLE

    def test_from_str_case_insensitive(self) -> None:
        assert EngineType.from_str("TESSERACT") == EngineType.TESSERACT

    def test_from_str_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown engine"):
            EngineType.from_str("easyocr")


# =============================================================================
# NormalizedWord
# =============================================================================


class TestNormalizedWord:
    def test_valid_word(self) -> None:
        word = NormalizedWord(
            text="Bonjour",
            bbox=BoundingBox(0, 0, 100, 30),
            confidence=0.95,
            page=1,
            image_name="doc.jpg",
            engine=EngineType.PADDLE,
        )
        assert word.text == "Bonjour"

    def test_confidence_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="Confidence"):
            NormalizedWord(
                text="x",
                bbox=BoundingBox(0, 0, 10, 10),
                confidence=1.5,  # invalid
                page=1,
                image_name="doc.jpg",
                engine=EngineType.TESSERACT,
            )

    def test_page_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="Page"):
            NormalizedWord(
                text="x",
                bbox=BoundingBox(0, 0, 10, 10),
                confidence=0.5,
                page=0,  # invalid
                image_name="doc.jpg",
                engine=EngineType.TESSERACT,
            )

    def test_to_dict_roundtrip(self) -> None:
        original = NormalizedWord(
            text="Test",
            bbox=BoundingBox(10, 20, 110, 50),
            confidence=0.88,
            page=2,
            image_name="image.png",
            engine=EngineType.TESSERACT,
            polygon=None,
        )
        d = original.to_dict()
        recovered = NormalizedWord.from_dict(d)
        assert recovered.text == original.text
        assert recovered.confidence == pytest.approx(original.confidence)
        assert recovered.page == original.page
        assert recovered.engine == original.engine
        assert recovered.bbox.x_min == pytest.approx(original.bbox.x_min)

    def test_to_dict_with_polygon(self) -> None:
        polygon = Polygon.from_list([[0, 0], [10, 0], [10, 10], [0, 10]])
        word = NormalizedWord(
            text="poly",
            bbox=BoundingBox(0, 0, 10, 10),
            confidence=0.7,
            page=1,
            image_name="doc.jpg",
            engine=EngineType.PADDLE,
            polygon=polygon,
        )
        d = word.to_dict()
        assert d["polygon"] is not None
        assert len(d["polygon"]) == 4

        recovered = NormalizedWord.from_dict(d)
        assert recovered.polygon is not None
        assert len(recovered.polygon.points) == 4


# =============================================================================
# Polygon
# =============================================================================


class TestPolygon:
    def test_from_list(self) -> None:
        poly = Polygon.from_list([[0, 0], [10, 0], [10, 10], [0, 10]])
        assert len(poly.points) == 4

    def test_to_bounding_box(self) -> None:
        poly = Polygon.from_list([[2, 3], [8, 3], [8, 9], [2, 9]])
        bbox = poly.to_bounding_box()
        assert bbox.x_min == 2
        assert bbox.y_min == 3
        assert bbox.x_max == 8
        assert bbox.y_max == 9

    def test_to_list_roundtrip(self) -> None:
        points = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        poly = Polygon.from_list(points)
        recovered = poly.to_list()
        for orig, rec in zip(points, recovered):
            assert rec[0] == pytest.approx(orig[0])
            assert rec[1] == pytest.approx(orig[1])
