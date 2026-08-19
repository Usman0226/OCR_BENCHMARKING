#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from shapely.geometry import box as shapely_box
from shapely.ops import unary_union


def load_annotations(ann_path: Path) -> dict[str, list]:
    raw: list[dict] = json.loads(ann_path.read_text(encoding="utf-8"))
    annotation_map: dict[str, list] = {}

    for task in raw:
        stem = _file_to_stem(task.get("file_upload", ""))
        boxes: list = []

        for annotation in task.get("annotations", []):
            for item in annotation.get("result", []):
                if item.get("type") != "rectanglelabels":
                    continue

                value = item["value"]
                img_w: float = float(item["original_width"])
                img_h: float = float(item["original_height"])

                x_min = (value["x"] / 100.0) * img_w
                y_min = (value["y"] / 100.0) * img_h
                x_max = x_min + (value["width"] / 100.0) * img_w
                y_max = y_min + (value["height"] / 100.0) * img_h

                if x_max > x_min and y_max > y_min:
                    boxes.append(shapely_box(x_min, y_min, x_max, y_max))

        annotation_map[stem] = boxes

    return annotation_map


def _file_to_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"^[0-9a-fA-F]{8}-", "", stem)
    return stem


def load_ocr_boxes(json_path: Path) -> list:
    data: dict = json.loads(json_path.read_text(encoding="utf-8"))
    boxes: list = []

    for word in data.get("words", []):
        b = word.get("bbox", {})
        x_min = float(b.get("x_min", 0))
        y_min = float(b.get("y_min", 0))
        x_max = float(b.get("x_max", 0))
        y_max = float(b.get("y_max", 0))
        if x_max > x_min and y_max > y_min:
            boxes.append(shapely_box(x_min, y_min, x_max, y_max))

    return boxes


def score_image(
    gt_boxes: list,
    ocr_boxes: list,
    engine: str,
) -> dict:
    words_total = len(gt_boxes)

    if words_total == 0:
        return {
            "words_total": 0,
            "found": 0,
            "missed": 0,
            "coverage_pct": "",
            "invented_boxes": len(ocr_boxes),
            "mean_iou": "",
        }

    if not ocr_boxes:
        return {
            "words_total": words_total,
            "found": 0,
            "missed": words_total,
            "coverage_pct": 0.0,
            "invented_boxes": 0,
            "mean_iou": 0.0 if engine == "tesseract" else "",
        }

    tool_union = unary_union(ocr_boxes)

    found = 0
    missed = 0
    for gt in gt_boxes:
        if gt.area == 0:
            continue
        covered_fraction = gt.intersection(tool_union).area / gt.area
        if covered_fraction >= 0.50:
            found += 1
        else:
            missed += 1

    coverage_pct = round(found / words_total * 100, 2)

    gt_union = unary_union(gt_boxes)
    invented_boxes = sum(
        1 for ob in ocr_boxes if ob.intersection(gt_union).area == 0
    )

    mean_iou: float | str = ""
    if engine == "tesseract":
        matched_ious: list[float] = []
        for gt in gt_boxes:
            best_iou = 0.0
            for ob in ocr_boxes:
                union_area = gt.union(ob).area
                if union_area == 0:
                    continue
                iou = gt.intersection(ob).area / union_area
                if iou > best_iou:
                    best_iou = iou
            if best_iou >= 0.5:
                matched_ious.append(best_iou)
        mean_iou = (
            round(sum(matched_ious) / len(matched_ious), 4)
            if matched_ious
            else 0.0
        )

    return {
        "words_total": words_total,
        "found": found,
        "missed": missed,
        "coverage_pct": coverage_pct,
        "invented_boxes": invented_boxes,
        "mean_iou": mean_iou,
    }


CSV_FIELDS = [
    "image",
    "tool",
    "words_total",
    "found",
    "missed",
    "coverage_pct",
    "invented_boxes",
    "mean_iou",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score OCR boxes against Label Studio annotations.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("images/"),
        help="Images directory.",
    )
    parser.add_argument(
        "--ann",
        type=Path,
        default=Path("annotations/"),
        help="Label Studio JSON export file or directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    ann_path: Path = args.ann
    if ann_path.is_dir():
        candidates = sorted(ann_path.glob("*.json"))
        candidates = [p for p in candidates if p.stat().st_size > 1000]
        if not candidates:
            print(f"ERROR: No JSON export found in {ann_path}", file=sys.stderr)
            sys.exit(1)
        ann_path = candidates[0]
        print(f"Auto-detected annotation file: {ann_path.name}")

    if not ann_path.exists():
        print(f"ERROR: Annotation file not found: {ann_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading annotations from: {ann_path}")
    annotations = load_annotations(ann_path)
    annotated = sum(1 for v in annotations.values() if v)
    print(f"  {len(annotations)} tasks loaded, {annotated} with word boxes\n")

    ocr_root = Path("ocr_output")
    engines = ["paddle", "tesseract"]
    rows: list[dict] = []

    for engine in engines:
        ocr_dir = ocr_root / engine
        if not ocr_dir.exists():
            print(f"WARNING: {ocr_dir} not found -- skipping {engine}\n")
            continue

        ocr_files = sorted(ocr_dir.glob("*.json"))
        scored = skipped_no_ann = 0
        print(f"Scoring {engine} ({len(ocr_files)} OCR files)...")

        for ocr_path in ocr_files:
            stem = ocr_path.stem

            if stem not in annotations:
                skipped_no_ann += 1
                continue

            gt_boxes = annotations[stem]
            ocr_boxes = load_ocr_boxes(ocr_path)

            metrics = score_image(gt_boxes, ocr_boxes, engine)

            rows.append({
                "image": stem + ".jpg",
                "tool": engine,
                **metrics,
            })
            scored += 1

            gt_count = metrics["words_total"]
            found = metrics["found"]
            cov = metrics["coverage_pct"]
            inv = metrics["invented_boxes"]
            print(
                f"  {stem:12s}  GT={gt_count:3d}  found={found:3d}  "
                f"coverage={cov!s:>6}%  invented={inv}"
            )

        print(
            f"  => scored {scored} image(s), "
            f"skipped {skipped_no_ann} (not in annotation export)\n"
        )

    if not rows:
        print("No results to write.", file=sys.stderr)
        sys.exit(1)

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"=> {args.out}  ({len(rows)} rows written)")


if __name__ == "__main__":
    main()

