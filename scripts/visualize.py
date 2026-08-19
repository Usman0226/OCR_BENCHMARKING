#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import click
import cv2

sys.path.insert(0, str(Path(__file__).parents[1]))

from ocr.core.config import load_config
from ocr.core.io import (
    load_all_ocr_results,
    load_label_studio_annotations,
)
from ocr.core.logger import configure_logging, get_logger
from ocr.core.matching import match_words
from ocr.core.models import EngineType, MatchStatus
from ocr.utils.image import draw_bboxes_on_image, load_image_rgb, save_image

logger = get_logger(__name__)

COLOR_TP = (0, 200, 0)
COLOR_FP = (220, 30, 30)
COLOR_FN = (255, 165, 0)


@click.command()
@click.option(
    "--engine",
    "-e",
    default="all",
    type=click.Choice(["paddle", "tesseract", "all"], case_sensitive=False),
)
@click.option(
    "--image",
    "-i",
    default=None,
    help="Restrict to a single image filename (e.g. 'invoice_001.jpg').",
)
@click.option(
    "--annotations",
    "-a",
    default=None,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Path to Label Studio JSON export.",
)
@click.option(
    "--config",
    "-c",
    default=None,
    type=click.Path(exists=True, file_okay=True, path_type=Path),
)
def main(
    engine: str,
    image: str | None,
    annotations: Path | None,
    config: Path | None,
) -> None:
    cfg = load_config(config)
    configure_logging(cfg.logging, cfg.paths.logs_dir)

    vis_dir = cfg.paths.reports_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    ann_path = annotations or _find_export(cfg.paths.annotations_dir)
    if ann_path is None or not ann_path.exists():
        click.echo("ERROR: No annotation export found. Use --annotations.", err=True)
        sys.exit(1)

    annotation_map = load_label_studio_annotations(
        export_path=ann_path,
        images_dir=cfg.paths.images_dir,
    )

    engines = (
        list(EngineType) if engine == "all" else [EngineType.from_str(engine)]
    )

    for eng in engines:
        ocr_results = load_all_ocr_results(cfg.paths.ocr_output_dir, engine=eng)
        if not ocr_results:
            click.echo(f"[SKIP] No OCR results for '{eng.value}'.")
            continue

        for result in ocr_results:
            if image and result.image_name != image:
                continue

            ann = annotation_map.get(result.image_name)
            if ann is None:
                logger.warning("No annotation for '%s' — skipping.", result.image_name)
                continue

            img_path = cfg.paths.images_dir / result.image_name
            if not img_path.exists():
                logger.warning("Image file not found: '%s'", img_path)
                continue

            try:
                rgb = load_image_rgb(img_path)
            except Exception as exc:
                logger.error("Failed to load '%s': %s", img_path, exc)
                continue

            matched = match_words(
                ocr_words=result.words,
                annotation_words=ann.words,
                cfg=cfg.matching,
            )

            bboxes = []
            colors = []
            for m in matched:
                if m.status == MatchStatus.TRUE_POSITIVE and m.ocr_word:
                    bboxes.append(m.ocr_word.bbox.to_xywh()[:2] + m.ocr_word.bbox.to_xywh()[2:])
                    colors.append(COLOR_TP)
                elif m.status == MatchStatus.FALSE_POSITIVE and m.ocr_word:
                    bboxes.append(
                        (
                            m.ocr_word.bbox.x_min,
                            m.ocr_word.bbox.y_min,
                            m.ocr_word.bbox.x_max,
                            m.ocr_word.bbox.y_max,
                        )
                    )
                    colors.append(COLOR_FP)
                elif m.status == MatchStatus.FALSE_NEGATIVE and m.annotation_word:
                    bboxes.append(
                        (
                            m.annotation_word.bbox.x_min,
                            m.annotation_word.bbox.y_min,
                            m.annotation_word.bbox.x_max,
                            m.annotation_word.bbox.y_max,
                        )
                    )
                    colors.append(COLOR_FN)

            annotated = draw_bboxes_on_image(rgb, bboxes, colors, thickness=2)
            _draw_legend(annotated)

            stem = Path(result.image_name).stem
            out_path = vis_dir / f"{stem}_{eng.value}_vis.jpg"
            save_image(annotated, out_path)
            click.echo(f"  Saved → {out_path}")

    click.echo(f"\nVisualizations saved to: {vis_dir}")


def _draw_legend(image: "cv2.typing.MatLike") -> None:
    legend_items = [
        (COLOR_TP, "True Positive"),
        (COLOR_FP, "False Positive"),
        (COLOR_FN, "False Negative"),
    ]
    x, y = 10, 10
    for color, label in legend_items:
        bgr = (color[2], color[1], color[0])
        cv2.rectangle(image, (x, y), (x + 20, y + 15), bgr, -1)
        cv2.putText(
            image,
            label,
            (x + 25, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 22


def _find_export(annotations_dir: Path) -> Path | None:
    if not annotations_dir.exists():
        return None
    for name in ("export.json", "annotations.json", "labelstudio_export.json"):
        p = annotations_dir / name
        if p.exists():
            return p
    jsons = sorted(annotations_dir.glob("*.json"))
    return jsons[0] if jsons else None


if __name__ == "__main__":
    main()
