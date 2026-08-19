from __future__ import annotations

from ocr.core.models import BoundingBox, Polygon


def bbox_intersection(a: BoundingBox, b: BoundingBox) -> float:
    inter_x_min = max(a.x_min, b.x_min)
    inter_y_min = max(a.y_min, b.y_min)
    inter_x_max = min(a.x_max, b.x_max)
    inter_y_max = min(a.y_max, b.y_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    return inter_w * inter_h


def bbox_union(a: BoundingBox, b: BoundingBox) -> float:
    intersection = bbox_intersection(a, b)
    return a.area + b.area - intersection


def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    if a.area == 0.0 or b.area == 0.0:
        return 0.0
    union = bbox_union(a, b)
    if union == 0.0:
        return 0.0
    return bbox_intersection(a, b) / union


def bbox_coverage(
    reference: BoundingBox,
    query: BoundingBox,
) -> float:
    if reference.area == 0.0:
        return 0.0
    return bbox_intersection(reference, query) / reference.area


def polygon_area(polygon: Polygon) -> float:
    pts = polygon.points
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def polygon_to_bbox(polygon: Polygon) -> BoundingBox:
    return polygon.to_bounding_box()


def denormalize_bbox(
    x_pct: float,
    y_pct: float,
    width_pct: float,
    height_pct: float,
    image_width: int,
    image_height: int,
) -> BoundingBox:
    x_min = (x_pct / 100.0) * image_width
    y_min = (y_pct / 100.0) * image_height
    x_max = x_min + (width_pct / 100.0) * image_width
    y_max = y_min + (height_pct / 100.0) * image_height
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def normalize_bbox(
    bbox: BoundingBox,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    if image_width == 0 or image_height == 0:
        raise ValueError("Image dimensions must be positive non-zero values.")
    x_pct = (bbox.x_min / image_width) * 100.0
    y_pct = (bbox.y_min / image_height) * 100.0
    w_pct = (bbox.width / image_width) * 100.0
    h_pct = (bbox.height / image_height) * 100.0
    return x_pct, y_pct, w_pct, h_pct


def merge_bboxes(boxes: list[BoundingBox]) -> BoundingBox:
    if not boxes:
        raise ValueError("Cannot merge an empty list of bounding boxes.")
    x_min = min(b.x_min for b in boxes)
    y_min = min(b.y_min for b in boxes)
    x_max = max(b.x_max for b in boxes)
    y_max = max(b.y_max for b in boxes)
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)