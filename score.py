#!/usr/bin/env python3
"""Score OCR results against annotation ground truth and produce reports.

Reads:
    annotations/         — Label Studio JSON export (export.json) or
                           per-image annotation JSONs
    ocr_output/paddle/   — PaddleOCR word-box JSONs (written by run_ocr.py)
    ocr_output/tesseract/— Tesseract word-box JSONs (written by run_ocr.py)

Writes (in project root):
    results.csv          — per-image scores for every engine
    results_summary.md   — final aggregate table + observations

Usage:
    python score.py
    python score.py --engine paddle
    python score.py --annotations annotations/my_export.json
    python score.py --config configs/config.yaml
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))

from ocr.core.config import load_config
from ocr.core.io import load_all_ocr_results, load_label_studio_annotations
from ocr.core.logger import configure_logging, get_logger
from ocr.core.models import AggregateScoringResult, EngineType
from ocr.core.scoring import score_all

console = Console()


@click.command()
@click.option(
    "--engine",
    "-e",
    default="all",
    type=click.Choice(["paddle", "tesseract", "all"], case_sensitive=False),
    help="Which engine results to score. Default: all.",
    show_default=True,
)
@click.option(
    "--annotations",
    "-a",
    default=None,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Label Studio JSON export file. Auto-detected if omitted.",
)
@click.option(
    "--config",
    "-c",
    default=None,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to config.yaml.",
)
@click.option(
    "--out-dir",
    "-o",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Where to write results.csv and results_summary.md. Defaults to project root.",
)
def main(
    engine: str,
    annotations: Path | None,
    config: Path | None,
    out_dir: Path | None,
) -> None:
    """Score OCR output against annotations and write results.csv + results_summary.md."""
    cfg = load_config(config)
    configure_logging(cfg.logging, cfg.paths.logs_dir)
    logger = get_logger(__name__)

    # Output directory — default to project root (same dir as this script)
    output_root = out_dir or Path(__file__).parent
    output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load annotations
    # ------------------------------------------------------------------ #
    ann_path = annotations or _find_annotation_export(cfg.paths.annotations_dir)
    if ann_path is None or not ann_path.exists():
        console.print(
            "[bold red]ERROR:[/bold red] No annotation export found.\n"
            f"  Place a Label Studio JSON export in '{cfg.paths.annotations_dir}'\n"
            "  or pass --annotations <file>.",
        )
        sys.exit(1)

    console.print(f"\n[bold]Annotations:[/bold] {ann_path}")

    try:
        annotation_map = load_label_studio_annotations(
            export_path=ann_path,
            images_dir=cfg.paths.images_dir,
        )
    except Exception as exc:
        console.print(f"[bold red]ERROR:[/bold red] Failed to load annotations: {exc}")
        sys.exit(1)

    if not annotation_map:
        console.print("[yellow]WARNING:[/yellow] No annotations loaded. Nothing to score.")
        sys.exit(0)

    console.print(
        f"[green]Loaded annotations for {len(annotation_map)} image(s).[/green]\n"
    )

    # ------------------------------------------------------------------ #
    # 2. Score each engine
    # ------------------------------------------------------------------ #
    engines_to_score: list[EngineType] = (
        list(EngineType) if engine == "all" else [EngineType.from_str(engine)]
    )

    all_aggregates: list[AggregateScoringResult] = []

    for eng in engines_to_score:
        # Read from engine-specific subdir: ocr_output/paddle/ or ocr_output/tesseract/
        engine_output_dir = cfg.paths.ocr_output_dir / eng.value
        ocr_results = load_all_ocr_results(engine_output_dir, engine=eng)

        if not ocr_results:
            console.print(
                f"[yellow]No OCR results for '{eng.value}' in '{engine_output_dir}'. "
                f"Run: python run_ocr.py --engine {eng.value}[/yellow]"
            )
            continue

        console.print(
            f"Scoring [bold cyan]{eng.value}[/bold cyan] "
            f"({len(ocr_results)} image(s))..."
        )

        aggregate = score_all(
            ocr_results=ocr_results,
            annotations=annotation_map,
            matching_cfg=cfg.matching,
            engine=eng,
        )
        all_aggregates.append(aggregate)

    if not all_aggregates:
        console.print("[yellow]No results to report.[/yellow]")
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # 3. Write results.csv  (per-image rows for all engines)
    # ------------------------------------------------------------------ #
    csv_path = output_root / "results.csv"
    _write_results_csv(all_aggregates, csv_path)
    console.print(f"\n[green]✓[/green] results.csv  → {csv_path}")

    # ------------------------------------------------------------------ #
    # 4. Write results_summary.md
    # ------------------------------------------------------------------ #
    md_path = output_root / "results_summary.md"
    _write_results_summary(all_aggregates, md_path)
    console.print(f"[green]✓[/green] results_summary.md → {md_path}")

    # ------------------------------------------------------------------ #
    # 5. Print console table
    # ------------------------------------------------------------------ #
    _print_summary_table(all_aggregates)


# ======================================================================== #
# Writers
# ======================================================================== #


def _write_results_csv(
    aggregates: list[AggregateScoringResult],
    path: Path,
) -> None:
    """Write one CSV row per (image × engine) + an aggregate summary row."""
    rows: list[dict[str, Any]] = []

    for agg in aggregates:
        for img in agg.per_image:
            rows.append(
                {
                    "engine": img.engine.value,
                    "image_name": img.image_name,
                    "precision": round(img.precision, 4),
                    "recall": round(img.recall, 4),
                    "f1": round(img.f1, 4),
                    "coverage": round(img.coverage, 4),
                    "true_positives": img.true_positives,
                    "false_positives": img.false_positives,
                    "false_negatives": img.false_negatives,
                    "missed_words": len(img.missed_words),
                    "invented_words": len(img.invented_words),
                    "processing_time_s": round(img.processing_time_s, 3),
                    "error": img.error or "",
                }
            )

        # Aggregate row at bottom of each engine block
        rows.append(
            {
                "engine": agg.engine.value,
                "image_name": "AGGREGATE",
                "precision": round(agg.mean_precision, 4),
                "recall": round(agg.mean_recall, 4),
                "f1": round(agg.mean_f1, 4),
                "coverage": round(agg.mean_coverage, 4),
                "true_positives": agg.total_true_positives,
                "false_positives": agg.total_false_positives,
                "false_negatives": agg.total_false_negatives,
                "missed_words": agg.total_missed_words,
                "invented_words": agg.total_invented_words,
                "processing_time_s": round(
                    sum(r.processing_time_s for r in agg.per_image), 3
                ),
                "error": "",
            }
        )

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_results_summary(
    aggregates: list[AggregateScoringResult],
    path: Path,
) -> None:
    """Write a Markdown summary with aggregate table + per-image breakdown."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "# OCR Benchmark — Results Summary",
        "",
        f"_Generated: {now}_",
        "",
        "## Aggregate Comparison",
        "",
        "| Engine | Images | Precision | Recall | F1 | Coverage | TP | FP | FN | Missed | Invented |",
        "|--------|--------|-----------|--------|-----|----------|----|----|----|--------|----------|",
    ]

    for agg in aggregates:
        lines.append(
            f"| **{agg.engine.value}** "
            f"| {agg.image_count} "
            f"| {agg.mean_precision:.4f} "
            f"| {agg.mean_recall:.4f} "
            f"| **{agg.mean_f1:.4f}** "
            f"| {agg.mean_coverage:.4f} "
            f"| {agg.total_true_positives} "
            f"| {agg.total_false_positives} "
            f"| {agg.total_false_negatives} "
            f"| {agg.total_missed_words} "
            f"| {agg.total_invented_words} |"
        )

    lines += ["", "---", ""]

    # Per-image breakdown for each engine
    for agg in aggregates:
        lines += [
            f"## Per-Image Results — {agg.engine.value.title()}",
            "",
            "| Image | Precision | Recall | F1 | Coverage | TP | FP | FN | Error |",
            "|-------|-----------|--------|----|----------|----|----|----|-------|",
        ]
        for r in agg.per_image:
            lines.append(
                f"| {r.image_name} "
                f"| {r.precision:.4f} "
                f"| {r.recall:.4f} "
                f"| {r.f1:.4f} "
                f"| {r.coverage:.4f} "
                f"| {r.true_positives} "
                f"| {r.false_positives} "
                f"| {r.false_negatives} "
                f"| {r.error or ''} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## Observations",
        "",
        "_Fill in your observations here after reviewing the results._",
        "",
        "- **PaddleOCR strengths/weaknesses:**",
        "- **Tesseract strengths/weaknesses:**",
        "- **Images that caused problems:**",
        "- **Overall recommendation:**",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


# ======================================================================== #
# Console output
# ======================================================================== #


def _print_summary_table(aggregates: list[AggregateScoringResult]) -> None:
    table = Table(title="OCR Benchmark — Aggregate Results", show_lines=True)
    table.add_column("Engine", style="bold cyan")
    table.add_column("Images", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right", style="bold green")
    table.add_column("Coverage", justify="right")
    table.add_column("TP", justify="right")
    table.add_column("FP", justify="right")
    table.add_column("FN", justify="right")
    table.add_column("Missed", justify="right")
    table.add_column("Invented", justify="right")

    for agg in aggregates:
        table.add_row(
            agg.engine.value,
            str(agg.image_count),
            f"{agg.mean_precision:.4f}",
            f"{agg.mean_recall:.4f}",
            f"{agg.mean_f1:.4f}",
            f"{agg.mean_coverage:.4f}",
            str(agg.total_true_positives),
            str(agg.total_false_positives),
            str(agg.total_false_negatives),
            str(agg.total_missed_words),
            str(agg.total_invented_words),
        )

    console.print()
    console.print(table)
    console.print()


# ======================================================================== #
# Helpers
# ======================================================================== #


def _find_annotation_export(annotations_dir: Path) -> Path | None:
    """Search for a Label Studio JSON export in annotations_dir."""
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
