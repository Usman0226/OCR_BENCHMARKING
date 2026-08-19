from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ocr.core.geometry import denormalize_bbox
from ocr.core.logger import get_logger
from ocr.core.models import (
    AnnotationWord,
    BoundingBox,
    EngineType,
    ImageAnnotation,
    NormalizedWord,
    OCRResult,
    Polygon,
)
from ocr.utils.image import get_image_dimensions

logger = get_logger(__name__)


def load_label_studio_annotations(
    export_path: Path,
    images_dir: Path,
) -> dict[str, ImageAnnotation]:
    if not export_path.exists():
        raise FileNotFoundError(f"Label Studio export not found: {export_path}")

    raw: list[dict[str, Any]] = json.loads(export_path.read_text(encoding="utf-8"))
    annotations_map: dict[str, ImageAnnotation] = {}

    for task in raw:
        task_id: int = task.get("id", -1)
        image_url: str = task.get("data", {}).get("image", "")
        image_name = Path(image_url).name

        image_path = images_dir / image_name
        if not image_path.exists():
            import re
            match = re.match(r"^[0-9a-fA-F]{8}-(.+)$", image_name)
            if match:
                fallback_name = match.group(1)
                fallback_path = images_dir / fallback_name
                if fallback_path.exists():
                    image_name = fallback_name
                    image_path = fallback_path

        try:
            img_w, img_h = get_image_dimensions(image_path)
        except Exception:
            logger.warning(
                "Could not read dimensions for '%s' — skipping task %d",
                image_name,
                task_id,
            )
            continue

        img_annotation = ImageAnnotation(image_name=image_name, task_id=task_id)

        for annotation in task.get("annotations", []):
            for result in annotation.get("result", []):
                word = _parse_result(result, image_name, img_w, img_h)
                if word is not None:
                    img_annotation.words.append(word)

        annotations_map[image_name] = img_annotation
        logger.debug(
            "Loaded %d words for image '%s'",
            img_annotation.word_count,
            image_name,
        )

    logger.info(
        "Loaded annotations for %d images from '%s'",
        len(annotations_map),
        export_path.name,
    )
    return annotations_map


def _parse_result(
    result: dict[str, Any],
    image_name: str,
    image_width: int,
    image_height: int,
) -> AnnotationWord | None:
    result_type = result.get("type", "")
    value = result.get("value", {})
    annotation_id = str(result.get("id", ""))

    if result_type == "rectanglelabels":
        labels: list[str] = value.get("rectanglelabels", [])
        label = labels[0] if labels else "text"
        text = value.get("text", "")

        try:
            bbox = denormalize_bbox(
                x_pct=float(value["x"]),
                y_pct=float(value["y"]),
                width_pct=float(value["width"]),
                height_pct=float(value["height"]),
                image_width=image_width,
                image_height=image_height,
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Malformed rectanglelabels result '%s': %s", annotation_id, exc
            )
            return None

        return AnnotationWord(
            text=text,
            bbox=bbox,
            label=label,
            annotation_id=annotation_id,
            image_name=image_name,
        )

    if result_type == "polygonlabels":
        labels = value.get("polygonlabels", [])
        label = labels[0] if labels else "text"
        text = value.get("text", "")

        raw_points: list[list[float]] = value.get("points", [])
        if not raw_points:
            return None

        pixel_points = [
            [p[0] / 100.0 * image_width, p[1] / 100.0 * image_height]
            for p in raw_points
        ]
        polygon = Polygon.from_list(pixel_points)
        bbox = polygon.to_bounding_box()

        return AnnotationWord(
            text=text,
            bbox=bbox,
            label=label,
            annotation_id=annotation_id,
            image_name=image_name,
            polygon=polygon,
        )

    return None


def save_ocr_result(result: OCRResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.image_name).stem
    filename = f"{stem}.json"
    out_path = output_dir / filename
    out_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.debug("Saved OCR result → %s", out_path)
    return out_path


def load_ocr_result(json_path: Path) -> OCRResult:
    data: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    words = [NormalizedWord.from_dict(w) for w in data.get("words", [])]
    return OCRResult(
        image_name=data["image_name"],
        image_path=Path(data["image_path"]),
        engine=EngineType(data["engine"]),
        words=words,
        processing_time_s=float(data.get("processing_time_s", 0.0)),
        error=data.get("error"),
    )


def load_all_ocr_results(
    output_dir: Path,
    engine: EngineType | None = None,
) -> list[OCRResult]:
    pattern = "*.json"
    paths = sorted(output_dir.glob(pattern))

    results: list[OCRResult] = []
    for path in paths:
        try:
            result = load_ocr_result(path)
            if engine is None or result.engine == engine:
                results.append(result)
        except Exception as exc:
            logger.warning("Failed to load OCR result from '%s': %s", path, exc)

    logger.info(
        "Loaded %d OCR results from '%s'%s",
        len(results),
        output_dir,
        f" (engine={engine.value})" if engine else "",
    )
    return results


def load_annotations_from_directory(annotations_dir: Path) -> dict[str, ImageAnnotation]:
    all_annotations: dict[str, ImageAnnotation] = {}

    single_export = annotations_dir / "export.json"
    if single_export.exists():
        logger.warning(
            "Found export.json but image dir is needed for coordinate conversion. "
            "Use load_label_studio_annotations() with images_dir instead."
        )
        return all_annotations

    for json_path in sorted(annotations_dir.glob("*_annotations.json")):
        try:
            data: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
            image_name: str = data.get("image_name", json_path.stem)
            img_ann = ImageAnnotation(image_name=image_name)
            for w in data.get("words", []):
                bbox_data = w["bbox"]
                bbox = BoundingBox(**bbox_data)
                polygon: Polygon | None = None
                if w.get("polygon"):
                    polygon = Polygon.from_list(w["polygon"])
                word = AnnotationWord(
                    text=w["text"],
                    bbox=bbox,
                    label=w.get("label", "text"),
                    annotation_id=w.get("annotation_id", ""),
                    image_name=image_name,
                    page=int(w.get("page", 1)),
                    polygon=polygon,
                )
                img_ann.words.append(word)
            all_annotations[image_name] = img_ann
        except Exception as exc:
            logger.warning("Failed to parse annotation file '%s': %s", json_path, exc)

    return all_annotations
