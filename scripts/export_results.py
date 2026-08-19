#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import click

sys.path.insert(0, str(Path(__file__).parents[1]))

from ocr.core.config import load_config
from ocr.core.logger import configure_logging, get_logger
from ocr.core.models import AggregateScoringResult

if TYPE_CHECKING:
    from ocr.core.config import ReportsConfig

logger = get_logger(__name__)


def export_all(
    aggregate: AggregateScoringResult,
    reports_dir: Path,
    reports_cfg: "ReportsConfig",
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    engine_name = aggregate.engine.value

    if "json" in reports_cfg.formats:
        _export_json(aggregate, reports_dir, engine_name)

    if "csv" in reports_cfg.formats:
        _export_csv(aggregate, reports_dir, engine_name, reports_cfg)

    if "markdown" in reports_cfg.formats:
        _export_markdown(aggregate, reports_dir, engine_name, reports_cfg)

    logger.info(
        "Exported reports for '%s' to '%s' (formats: %s)",
        engine_name,
        reports_dir,
        reports_cfg.formats,
    )


def _export_json(
    aggregate: AggregateScoringResult,
    reports_dir: Path,
    engine_name: str,
) -> Path:
    out_path = reports_dir / f"{engine_name}_results.json"
    out_path.write_text(
        json.dumps(aggregate.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.debug("JSON report → %s", out_path)
    return out_path


def _export_csv(
    aggregate: AggregateScoringResult,
    reports_dir: Path,
    engine_name: str,
    cfg: "ReportsConfig",
) -> Path:
    out_path = reports_dir / f"{engine_name}_results.csv"

    rows = [r.to_flat_dict() for r in aggregate.per_image]

    summary = {
        "image_name": "AGGREGATE",
        "engine": engine_name,
        "true_positives": aggregate.total_true_positives,
        "false_positives": aggregate.total_false_positives,
        "false_negatives": aggregate.total_false_negatives,
        "precision": round(aggregate.mean_precision, 4),
        "recall": round(aggregate.mean_recall, 4),
        "f1": round(aggregate.mean_f1, 4),
        "coverage": round(aggregate.mean_coverage, 4),
        "missed_word_count": aggregate.total_missed_words,
        "invented_word_count": aggregate.total_invented_words,
        "processing_time_s": sum(
            r.processing_time_s for r in aggregate.per_image
        ),
        "error": None,
    }
    rows.append(summary)

    if not rows:
        return out_path

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.debug("CSV report → %s", out_path)
    return out_path


def _export_markdown(
    aggregate: AggregateScoringResult,
    reports_dir: Path,
    engine_name: str,
    cfg: "ReportsConfig",
) -> Path:
    out_path = reports_dir / f"{engine_name}_results.md"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        f"# OCR Benchmark Report — {engine_name.title()}",
        f"",
        f"_Generated: {now}_",
        f"",
        f"## Aggregate Results",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Engine | `{engine_name}` |",
        f"| Images scored | {aggregate.image_count} |",
        f"| Mean Precision | **{aggregate.mean_precision:.4f}** |",
        f"| Mean Recall | **{aggregate.mean_recall:.4f}** |",
        f"| Mean F1 | **{aggregate.mean_f1:.4f}** |",
        f"| Mean Coverage | **{aggregate.mean_coverage:.4f}** |",
        f"| Total True Positives | {aggregate.total_true_positives} |",
        f"| Total False Positives | {aggregate.total_false_positives} |",
        f"| Total False Negatives | {aggregate.total_false_negatives} |",
        f"| Total Missed Words | {aggregate.total_missed_words} |",
        f"| Total Invented Words | {aggregate.total_invented_words} |",
        f"",
    ]

    if cfg.include_per_image and aggregate.per_image:
        lines += [
            f"## Per-Image Results",
            f"",
            f"| Image | Precision | Recall | F1 | Coverage | TP | FP | FN | Error |",
            f"|-------|-----------|--------|----|----------|----|----|----|-------|",
        ]
        for r in aggregate.per_image:
            err = r.error or ""
            lines.append(
                f"| {r.image_name} "
                f"| {r.precision:.4f} "
                f"| {r.recall:.4f} "
                f"| {r.f1:.4f} "
                f"| {r.coverage:.4f} "
                f"| {r.true_positives} "
                f"| {r.false_positives} "
                f"| {r.false_negatives} "
                f"| {err} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.debug("Markdown report → %s", out_path)
    return out_path


@click.command()
@click.option(
    "--engine",
    "-e",
    default="all",
    type=click.Choice(["paddle", "tesseract", "all"], case_sensitive=False),
    help="Engine results to export. Defaults to 'all'.",
)
@click.option(
    "--config",
    "-c",
    default=None,
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    help="Path to config.yaml.",
)
def main(engine: str, config: Path | None) -> None:
    cfg = load_config(config)
    configure_logging(cfg.logging, cfg.paths.logs_dir)

    from ocr.core.models import AggregateScoringResult, EngineType

    results_dir = cfg.paths.results_dir

    engines = (
        list(EngineType)
        if engine == "all"
        else [EngineType.from_str(engine)]
    )

    for eng in engines:
        json_path = results_dir / f"{eng.value}_results.json"
        if not json_path.exists():
            click.echo(
                f"[SKIP] No results file found for '{eng.value}': {json_path}"
            )
            continue

        data = json.loads(json_path.read_text(encoding="utf-8"))
        from ocr.core.models import ImageScoringResult
        per_image = []
        for img in data.get("per_image", []):
            per_image.append(
                ImageScoringResult(
                    image_name=img["image_name"],
                    engine=EngineType(img["engine"]),
                    true_positives=img["true_positives"],
                    false_positives=img["false_positives"],
                    false_negatives=img["false_negatives"],
                    precision=img["precision"],
                    recall=img["recall"],
                    f1=img["f1"],
                    coverage=img["coverage"],
                    missed_words=img.get("missed_words", []),
                    invented_words=img.get("invented_words", []),
                    processing_time_s=img.get("processing_time_s", 0.0),
                    error=img.get("error"),
                )
            )
        agg = AggregateScoringResult(
            engine=EngineType(data["engine"]),
            image_count=data["image_count"],
            total_true_positives=data["total_true_positives"],
            total_false_positives=data["total_false_positives"],
            total_false_negatives=data["total_false_negatives"],
            mean_precision=data["mean_precision"],
            mean_recall=data["mean_recall"],
            mean_f1=data["mean_f1"],
            mean_coverage=data["mean_coverage"],
            total_missed_words=data["total_missed_words"],
            total_invented_words=data["total_invented_words"],
            per_image=per_image,
        )
        export_all(agg, cfg.paths.reports_dir, cfg.reports)
        click.echo(f"Exported reports for '{eng.value}' → {cfg.paths.reports_dir}")


if __name__ == "__main__":
    main()
