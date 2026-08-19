#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import click
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

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
    default="all",
    type=click.Choice(["paddle", "tesseract", "all"], case_sensitive=False),
    help="Which engine to run. Default: both engines.",
    show_default=True,
)
@click.option(
    "--images",
    "-i",
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory of images. Defaults to config value.",
)
@click.option(
    "--config",
    "-c",
    default=None,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to config.yaml.",
)
@click.option(
    "--lang",
    default=None,
    help="Override Tesseract language (e.g. 'fra', 'fra+eng'). Ignored for Paddle.",
)
def main(
    engine: str,
    images: Path | None,
    config: Path | None,
    lang: str | None,
) -> None:
    cfg = load_config(config)
    configure_logging(cfg.logging, cfg.paths.logs_dir)
    logger = get_logger(__name__)

    images_dir = images or cfg.paths.images_dir

    if lang:
        cfg.engines.tesseract.lang = lang

    try:
        image_paths = list_images(images_dir)
    except NotADirectoryError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)

    if not image_paths:
        click.echo(
            f"No supported images found in '{images_dir}'.\n"
            "Supported: jpg, jpeg, png, tiff, bmp, webp",
            err=True,
        )
        sys.exit(1)

    engines_to_run: list[EngineType]
    if engine == "all":
        engines_to_run = [EngineType.PADDLE, EngineType.TESSERACT]
    else:
        engines_to_run = [EngineType.from_str(engine)]

    overall_errors = 0

    for eng_type in engines_to_run:
        engine_output_dir = cfg.paths.ocr_output_dir / eng_type.value
        engine_output_dir.mkdir(parents=True, exist_ok=True)

        click.echo(
            f"\n{'─'*60}\n"
            f"  Engine  : {eng_type.value}\n"
            f"  Images  : {images_dir}  ({len(image_paths)} files)\n"
            f"  Output  : {engine_output_dir}\n"
            f"{'─'*60}"
        )

        ocr_engine = create_engine(eng_type, cfg)
        try:
            ocr_engine.initialize()
        except RuntimeError as exc:
            click.echo(f"ERROR: Cannot initialize '{eng_type.value}': {exc}", err=True)
            overall_errors += 1
            continue

        success_count = 0
        error_count = 0

        try:
            for image_path in tqdm(
                image_paths, desc=f"[{eng_type.value}]", unit="img"
            ):
                result = ocr_engine.run(image_path)
                save_ocr_result(result, engine_output_dir)
                if result.succeeded:
                    success_count += 1
                else:
                    error_count += 1
                    logger.warning(
                        "Failed: %s — %s", image_path.name, result.error
                    )
        finally:
            ocr_engine.shutdown()

        click.echo(
            f"  ✓ {success_count} succeeded   ✗ {error_count} failed\n"
            f"  Saved → {engine_output_dir}\n"
        )
        overall_errors += error_count

    if overall_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
