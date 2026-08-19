from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ocr.core.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
)


def load_image_rgb(image_path: Path) -> np.ndarray:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_image_pil(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    return img


def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size


def preprocess_for_tesseract(
    image: Image.Image,
    grayscale: bool = True,
    max_long_edge: int | None = 4096,
) -> Image.Image:
    if max_long_edge is not None:
        w, h = image.size
        long_edge = max(w, h)
        if long_edge > max_long_edge:
            scale = max_long_edge / long_edge
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            logger.debug(
                "Resized image from %dx%d to %dx%d for Tesseract.", w, h, new_w, new_h
            )

    if grayscale:
        image = image.convert("L")

    return image


def draw_bboxes_on_image(
    image: np.ndarray,
    bboxes: list[tuple[float, float, float, float]],
    colors: list[tuple[int, int, int]],
    thickness: int = 2,
) -> np.ndarray:
    output = image.copy()
    for (x_min, y_min, x_max, y_max), color in zip(bboxes, colors):
        bgr_color = (color[2], color[1], color[0])
        cv2.rectangle(
            output,
            (int(x_min), int(y_min)),
            (int(x_max), int(y_max)),
            bgr_color,
            thickness,
        )
    return output


def save_image(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr)
