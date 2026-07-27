#!/usr/bin/env python3
"""CLI: Score OCR results against Label Studio annotations.

Loads previously generated OCR JSON files, matches them against annotations,
computes precision/recall/F1, and writes reports to the reports directory.

Usage:
    python scripts/score.py
    python scripts/score.py --engine paddle
    python scripts/score.py --engine tesseract --config configs/config.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parents[1]))

from ocr.core.config import load_config
from ocr.core.io import load_all_ocr_results, load_label_studio_annotations
from ocr.core.logger import configure_logging, get_logger
from ocr.core.models import AggregateScoringResult, EngineType
from ocr.core.scoring import score_all
from scripts.export_results import export_all

console = Console()


@click.command()
@click.option(
    "--engine",
    "-e",
    default=None,
    type=click.Choice(["paddle", "tesseract", "all"], case_sensitive=False),
    help="Engine to score. Defaults to 'all' (scores all available results).",
)
@click.option(
    "--annotations",
    "-a",
    default=None,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Path to Label Studio JSON export file.",
)
@click.option(
    "--config",
    "-c",
    default=None,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to config.yaml.",
)
def main(
    engine: str | None,
    annotations: Path | None,
    config: Path | None,
) -> None:
    """Score OCR results against Label Studio annotations and generate reports."""
    cfg = load_config(config)
    configure_logging(cfg.logging, cfg.paths.logs_dir)
    logger = get_logger(__name__)

    # Locate annotation export file
    annotations_path = annotations or _find_annotation_export(cfg.paths.annotations_dir)
    if annotations_path is None or not annotations_path.exists():
        console.print(
            "[bold red]ERROR:[/bold red] No annotation export file found. "
            "Export your Label Studio project as JSON and place it in "
            f"'{cfg.paths.annotations_dir}' or pass --annotations.",
        )
        sys.exit(1)

    console.print(f"\n[bold]Annotation file:[/bold] {annotations_path}")

    # Load annotations
    try:
        annotation_map = load_label_studio_annotations(
            export_path=annotations_path,
            images_dir=cfg.paths.images_dir,
        )
    except Exception as exc:
        console.print(f"[bold red]ERROR:[/bold red] Failed to load annotations: {exc}")
        sys.exit(1)

    if not annotation_map:
        console.print("[yellow]WARNING:[/yellow] No annotations loaded. Nothing to score.")
        sys.exit(0)

    console.print(f"[green]Loaded annotations for {len(annotation_map)} images.[/green]\n")

    # Determine which engines to score
    engines_to_score: list[EngineType]
    if engine is None or engine == "all":
        engines_to_score = list(EngineType)
    else:
        engines_to_score = [EngineType.from_str(engine)]

    aggregate_results: list[AggregateScoringResult] = []

    for eng in engines_to_score:
        ocr_results = load_all_ocr_results(cfg.paths.ocr_output_dir, engine=eng)
        if not ocr_results:
            console.print(
                f"[yellow]No OCR results found for engine '{eng.value}' "
                f"in '{cfg.paths.ocr_output_dir}'. Skipping.[/yellow]"
            )
            continue

        console.print(
            f"Scoring [bold cyan]{eng.value}[/bold cyan] "
            f"({len(ocr_results)} images)..."
        )

        aggregate = score_all(
            ocr_results=ocr_results,
            annotations=annotation_map,
            matching_cfg=cfg.matching,
            engine=eng,
        )
        aggregate_results.append(aggregate)

        # Export reports for this engine
        export_all(aggregate, cfg.paths.reports_dir, cfg.reports)

    if not aggregate_results:
        console.print("[yellow]No results to report.[/yellow]")
        sys.exit(0)

    # Pretty console summary table
    _print_summary_table(aggregate_results)


def _find_annotation_export(annotations_dir: Path) -> Path | None:
    """Search for a Label Studio export JSON in the annotations directory."""
    if not annotations_dir.exists():
        return None
    # Check common filenames first
    for name in ("export.json", "annotations.json", "labelstudio_export.json"):
        candidate = annotations_dir / name
        if candidate.exists():
            return candidate
    # Fall back to the first JSON file found
    json_files = sorted(annotations_dir.glob("*.json"))
    return json_files[0] if json_files else None


def _print_summary_table(results: list[AggregateScoringResult]) -> None:
    """Render a rich summary table to the console."""
    table = Table(title="OCR Benchmark Results", show_lines=True)
    table.add_column("Engine", style="bold cyan")
    table.add_column("Images", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("TP", justify="right")
    table.add_column("FP", justify="right")
    table.add_column("FN", justify="right")
    table.add_column("Missed", justify="right")
    table.add_column("Invented", justify="right")

    for r in results:
        table.add_row(
            r.engine.value,
            str(r.image_count),
            f"{r.mean_precision:.3f}",
            f"{r.mean_recall:.3f}",
            f"{r.mean_f1:.3f}",
            f"{r.mean_coverage:.3f}",
            str(r.total_true_positives),
            str(r.total_false_positives),
            str(r.total_false_negatives),
            str(r.total_missed_words),
            str(r.total_invented_words),
        )

    console.print()
    console.print(table)
    console.print()


if __name__ == "__main__":
    main()
