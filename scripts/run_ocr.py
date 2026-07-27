#!/usr/bin/env python3
"""CLI: Run an OCR engine on a directory of images.

Usage:
    python scripts/run_ocr.py --engine paddle --images images/
    python scripts/run_ocr.py --engine tesseract --images images/
    python scripts/run_ocr.py --engine tesseract --images images/ --config configs/config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from tqdm import tqdm

# Ensure project root is on PYTHONPATH when running directly
sys.path.insert(0, str(Path(__file__).parents[1]))

from ocr.core.config import load_config
from ocr.core.io import save_ocr_result
from ocr.core.logger import configure_logging, get_logger
from ocr.core.models import EngineType
from ocr.engines.base import create_engine
from ocr.utils.files import list_images


@click.command()
@click.option(
    "--engine",
    "-e",
    required=True,
    type=click.Choice(["paddle", "tesseract"], case_sensitive=False),
    help="OCR engine to use.",
)
@click.option(
    "--images",
    "-i",
    required=False,
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory of images to process. Defaults to config value.",
)
@click.option(
    "--config",
    "-c",
    default=None,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to config.yaml. Defaults to CONFIG_PATH env var or configs/config.yaml.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Output directory for OCR JSON results. Defaults to config value.",
)
@click.option(
    "--lang",
    default=None,
    help=(
        "Override language for the selected engine. "
        "For Tesseract: 'fra+eng'. For Paddle: see PaddleOCR docs."
    ),
)
def main(
    engine: str,
    images: Path | None,
    config: Path | None,
    output: Path | None,
    lang: str | None,
) -> None:
    """Run an OCR engine on all images in a directory and save normalized output."""
    cfg = load_config(config)

    # Override lang from CLI if provided
    if lang:
        if engine.lower() == "tesseract":
            cfg.engines.tesseract.lang = lang
        elif engine.lower() == "paddle":
            cfg.engines.paddle.lang = lang

    configure_logging(cfg.logging, cfg.paths.logs_dir)
    logger = get_logger(__name__)

    images_dir = images or cfg.paths.images_dir
    output_dir = output or cfg.paths.ocr_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    engine_type = EngineType.from_str(engine)
    ocr_engine = create_engine(engine_type, cfg)

    # Discover images
    try:
        image_paths = list_images(images_dir)
    except NotADirectoryError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)

    if not image_paths:
        click.echo(
            f"No supported images found in '{images_dir}'. "
            "Supported formats: jpg, jpeg, png, tiff, bmp, webp",
            err=True,
        )
        sys.exit(1)

    click.echo(
        f"\n{'─'*60}\n"
        f"  Engine  : {engine_type.value}\n"
        f"  Images  : {images_dir} ({len(image_paths)} files)\n"
        f"  Output  : {output_dir}\n"
        f"{'─'*60}\n"
    )

    # Initialize engine
    try:
        ocr_engine.initialize()
    except RuntimeError as exc:
        click.echo(f"ERROR: Engine initialization failed — {exc}", err=True)
        sys.exit(1)

    # Process images
    success_count = 0
    error_count = 0

    try:
        for image_path in tqdm(image_paths, desc=f"[{engine_type.value}]", unit="img"):
            result = ocr_engine.run(image_path)
            save_ocr_result(result, output_dir)

            if result.succeeded:
                success_count += 1
            else:
                error_count += 1
                logger.warning("Failed: %s — %s", image_path.name, result.error)
    finally:
        ocr_engine.shutdown()

    click.echo(
        f"\n{'─'*60}\n"
        f"  Done!  ✓ {success_count} succeeded  ✗ {error_count} failed\n"
        f"  Results saved to: {output_dir}\n"
        f"{'─'*60}\n"
    )

    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
