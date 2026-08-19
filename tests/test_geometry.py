from __future__ import annotations

import pytest

from ocr.core.geometry import (
    bbox_coverage,
    bbox_intersection,
    bbox_iou,
    bbox_union,
    denormalize_bbox,
    merge_bboxes,
    normalize_bbox,
    polygon_area,
    polygon_to_bbox,
)
from ocr.core.models import BoundingBox, Polygon


class TestBboxIntersection:
    def test_identical_boxes(self, unit_box: BoundingBox) -> None:
        assert bbox_intersection(unit_box, unit_box) == pytest.approx(1.0)

    def test_no_overlap(
        self, unit_box: BoundingBox, non_overlapping_box: BoundingBox
    ) -> None:
        assert bbox_intersection(unit_box, non_overlapping_box) == 0.0

    def test_half_overlap(
        self, unit_box: BoundingBox, overlapping_box: BoundingBox
    ) -> None:
        result = bbox_intersection(unit_box, overlapping_box)
        assert result == pytest.approx(0.25)

    def test_contained(
        self, unit_box: BoundingBox, contained_box: BoundingBox
    ) -> None:
        result = bbox_intersection(unit_box, contained_box)
        assert result == pytest.approx(0.36)

    def test_touching_edge(self) -> None:
        a = BoundingBox(0, 0, 1, 1)
        b = BoundingBox(1, 0, 2, 1)
        assert bbox_intersection(a, b) == 0.0

    def test_symmetry(
        self, unit_box: BoundingBox, overlapping_box: BoundingBox
    ) -> None:
        assert bbox_intersection(unit_box, overlapping_box) == pytest.approx(
            bbox_intersection(overlapping_box, unit_box)
        )


class TestBboxUnion:
    def test_identical_boxes(self, unit_box: BoundingBox) -> None:
        assert bbox_union(unit_box, unit_box) == pytest.approx(1.0)

    def test_no_overlap(
        self, unit_box: BoundingBox, non_overlapping_box: BoundingBox
    ) -> None:
        assert bbox_union(unit_box, non_overlapping_box) == pytest.approx(2.0)

    def test_half_overlap(
        self, unit_box: BoundingBox, overlapping_box: BoundingBox
    ) -> None:
        assert bbox_union(unit_box, overlapping_box) == pytest.approx(1.75)


class TestBboxIoU:
    def test_identical_boxes_iou_is_one(self, unit_box: BoundingBox) -> None:
        assert bbox_iou(unit_box, unit_box) == pytest.approx(1.0)

    def test_no_overlap_iou_is_zero(
        self, unit_box: BoundingBox, non_overlapping_box: BoundingBox
    ) -> None:
        assert bbox_iou(unit_box, non_overlapping_box) == 0.0

    def test_half_overlap(
        self, unit_box: BoundingBox, overlapping_box: BoundingBox
    ) -> None:
        iou = bbox_iou(unit_box, overlapping_box)
        assert iou == pytest.approx(0.25 / 1.75, rel=1e-4)

    def test_zero_area_box(self) -> None:
        zero = BoundingBox(0, 0, 0, 0)
        normal = BoundingBox(0, 0, 1, 1)
        assert bbox_iou(zero, normal) == 0.0

    def test_symmetry(
        self, unit_box: BoundingBox, overlapping_box: BoundingBox
    ) -> None:
        assert bbox_iou(unit_box, overlapping_box) == pytest.approx(
            bbox_iou(overlapping_box, unit_box)
        )

    def test_contained_box(
        self, unit_box: BoundingBox, contained_box: BoundingBox
    ) -> None:
        iou = bbox_iou(unit_box, contained_box)
        assert 0 < iou < 1.0
        assert iou == pytest.approx(0.36 / 1.0, rel=1e-4)

    def test_iou_range(
        self, unit_box: BoundingBox, overlapping_box: BoundingBox
    ) -> None:
        iou = bbox_iou(unit_box, overlapping_box)
        assert 0.0 <= iou <= 1.0


class TestBboxCoverage:
    def test_identical_coverage_is_one(self, unit_box: BoundingBox) -> None:
        assert bbox_coverage(unit_box, unit_box) == pytest.approx(1.0)

    def test_no_overlap_coverage_is_zero(
        self, unit_box: BoundingBox, non_overlapping_box: BoundingBox
    ) -> None:
        assert bbox_coverage(unit_box, non_overlapping_box) == 0.0

    def test_contained_coverage(
        self, unit_box: BoundingBox, contained_box: BoundingBox
    ) -> None:
        result = bbox_coverage(unit_box, contained_box)
        assert result == pytest.approx(0.36, rel=1e-4)

    def test_zero_area_reference(self) -> None:
        zero = BoundingBox(0, 0, 0, 0)
        normal = BoundingBox(0, 0, 1, 1)
        assert bbox_coverage(zero, normal) == 0.0


class TestCoordinateNormalization:
    def test_denormalize_simple(self) -> None:
        bbox = denormalize_bbox(
            x_pct=10.0, y_pct=20.0, width_pct=30.0, height_pct=40.0,
            image_width=100, image_height=200,
        )
        assert bbox.x_min == pytest.approx(10.0)
        assert bbox.y_min == pytest.approx(40.0)
        assert bbox.x_max == pytest.approx(40.0)
        assert bbox.y_max == pytest.approx(120.0)

    def test_normalize_roundtrip(self) -> None:
        original = BoundingBox(x_min=50, y_min=100, x_max=150, y_max=300)
        x_pct, y_pct, w_pct, h_pct = normalize_bbox(original, 200, 400)
        recovered = denormalize_bbox(x_pct, y_pct, w_pct, h_pct, 200, 400)
        assert recovered.x_min == pytest.approx(original.x_min, rel=1e-6)
        assert recovered.y_min == pytest.approx(original.y_min, rel=1e-6)
        assert recovered.x_max == pytest.approx(original.x_max, rel=1e-6)
        assert recovered.y_max == pytest.approx(original.y_max, rel=1e-6)

    def test_normalize_zero_dimension_raises(self) -> None:
        bbox = BoundingBox(0, 0, 10, 10)
        with pytest.raises(ValueError):
            normalize_bbox(bbox, 0, 100)


class TestPolygonArea:
    def test_unit_square(self) -> None:
        square = Polygon(points=((0, 0), (1, 0), (1, 1), (0, 1)))
        assert polygon_area(square) == pytest.approx(1.0)

    def test_right_triangle(self) -> None:
        triangle = Polygon(points=((0, 0), (4, 0), (0, 3)))
        assert polygon_area(triangle) == pytest.approx(6.0)

    def test_degenerate_line(self) -> None:
        line = Polygon(points=((0, 0), (1, 0)))
        assert polygon_area(line) == 0.0

    def test_polygon_to_bbox(self) -> None:
        quad = Polygon(points=((1, 2), (5, 2), (5, 8), (1, 8)))
        bbox = polygon_to_bbox(quad)
        assert bbox.x_min == 1.0
        assert bbox.y_min == 2.0
        assert bbox.x_max == 5.0
        assert bbox.y_max == 8.0


class TestMergeBboxes:
    def test_merge_two(self) -> None:
        a = BoundingBox(0, 0, 5, 5)
        b = BoundingBox(3, 3, 10, 10)
        merged = merge_bboxes([a, b])
        assert merged.x_min == 0
        assert merged.y_min == 0
        assert merged.x_max == 10
        assert merged.y_max == 10

    def test_merge_single(self) -> None:
        box = BoundingBox(1, 2, 3, 4)
        assert merge_bboxes([box]) == box

    def test_merge_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            merge_bboxes([])
